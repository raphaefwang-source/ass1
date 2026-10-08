#!/usr/bin/env python3
"""
Which velocity save interval resolves the VACF? (Used only to choose the production save interval; it says nothing
about the convergence of the long-time diffusion coefficient.)

Input:
- toy_run.py runs at N = 64, laws A and B, frames saved every step (dt = 0.005).

For each run and each candidate interval Delta in {0.01, 0.025, 0.05, 0.1}, compared with the every-step reference:
- quadrature only: the reference VACF (all time origins), read at lags k Delta, trapezoid with step Delta. This is the
  pure discretisation error of the time grid.
- production-like: VACF from the frames a run with that interval would store (origins and lags every Delta).
  This adds the effect of fewer time origins, which is statistical.
- early decay: C(Delta) / C(0) and the times at which the normalised VACF first falls below 1/2 and 1/e (linear
  interpolation between stored lags), against the every-step values.
- Green-Kubo: D(T) = (1/3) int_0^T <v(0).v(tau)> dtau for T = 0.2, 0.5, 1, 2, 3 (same upper limits for all
  intervals); relative difference to the every-step value and the ratio to the statistical error of the reference
  (block SEM, 8 blocks of the production run).
- reference quadrature error: estimated by Richardson, (D(0.005) - D_quad(0.01)) / 3.

Selection rule, fixed before the comparison:
- the largest Delta whose quadrature-only Green-Kubo bias is <= 0.2 % at every T for both laws and every run;
- at least 5 stored lags before the normalised VACF falls to 1/2;
- interpolated tau_1/2 within 2 % of the every-step value.
If no candidate passes, saving every step is kept.

    python3 vacf_sampling_check.py RUN_DIR [RUN_DIR ...] --out vacf_sampling_results
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toy_run as tr  # noqa: E402

DT = 0.005
DELTAS = (0.005, 0.01, 0.025, 0.05, 0.1)
T_GK = (0.2, 0.5, 1.0, 2.0, 3.0)
TAU_MAX = 3.0
RULE = dict(gk_bias_max=0.002, min_lags_before_half=5, tau_half_rel_max=0.02)
_trapz = getattr(np, "trapezoid", None) or np.trapz          # NumPy < 2 has only trapz


def vacf(V, stride, kmax):
    """<v_i(t0) . v_i(t0 + k stride dt)>, averaged over particles and all stored time origins, k = 0..kmax."""
    X = V[::stride].reshape(V[::stride].shape[0], -1)
    F = X.shape[0]
    nfft = 1 << int(np.ceil(np.log2(2 * F)))
    Xf = np.fft.rfft(X, n=nfft, axis=0)
    acf = np.fft.irfft(Xf.real ** 2 + Xf.imag ** 2, n=nfft, axis=0)[:kmax + 1].sum(axis=1)
    return acf / ((F - np.arange(kmax + 1)) * V.shape[1])


def crossing(c, delta, level):
    """First time the normalised VACF falls below level (linear interpolation); nan if it never does."""
    idx = np.nonzero(c < level)[0]
    if len(idx) == 0:
        return np.nan
    k = idx[0]
    return delta * (k - 1 + (c[k - 1] - level) / (c[k - 1] - c[k]))


def gk(C, delta, T):
    k = int(round(T / delta))
    return float(_trapz(C[:k + 1], dx=delta)) / 3.0


def analyse(run_dir):
    R = tr.load_run(run_dir)
    cfg = R["config"]
    fs = R["frame_step"]
    assert np.all(np.diff(fs) == 1), "frames must be saved every step"
    V = R["V"]
    kref = int(round(TAU_MAX / DT))
    Cref = vacf(V, 1, kref)
    cref = Cref / Cref[0]
    out = dict(run=str(run_dir), law=cfg["resolved"]["law"], N=cfg["resolved"]["N"], seed=cfg["plan"]["seed"],
               frames=len(fs), production_time=len(fs) * DT - DT, C0=float(Cref[0]),
               tau_half_ref=crossing(cref, DT, 0.5), tau_1e_ref=crossing(cref, DT, np.exp(-1)),
               D_ref={str(T): gk(Cref, DT, T) for T in T_GK})
    nb = 8
    Lb = len(V) // nb
    Db = np.array([[gk(vacf(V[b * Lb:(b + 1) * Lb], 1, kref), DT, T) for T in T_GK] for b in range(nb)])
    out["D_block_sem"] = {str(T): float(Db[:, i].std(ddof=1) / np.sqrt(nb)) for i, T in enumerate(T_GK)}
    C2 = Cref[::2]
    out["D_ref_quadrature_error_richardson"] = {str(T): (gk(Cref, DT, T) - gk(C2, 2 * DT, T)) / 3 for T in T_GK}
    rows = []
    for delta in DELTAS[1:]:
        s = int(round(delta / DT))
        k = int(round(TAU_MAX / delta))
        Cq = Cref[::s][:k + 1]                       # quadrature only (all origins)
        Cp = vacf(V, s, k)                           # production-like (origins every delta)
        cp = Cp / Cp[0]
        th = crossing(cp, delta, 0.5)
        row = dict(law=out["law"], seed=out["seed"], delta=delta, stride=s,
                   c_first_lag=float(cp[1]), c_first_lag_ref=float(cref[s]),
                   lags_before_half=int(np.nonzero(cp < 0.5)[0][0]) if np.any(cp < 0.5) else None,
                   tau_half=th, tau_half_rel=(th - out["tau_half_ref"]) / out["tau_half_ref"],
                   tau_1e=crossing(cp, delta, np.exp(-1)),
                   tau_1e_rel=(crossing(cp, delta, np.exp(-1)) - out["tau_1e_ref"]) / out["tau_1e_ref"])
        for T in T_GK:
            dq, dp, dr = gk(Cq, delta, T), gk(Cp, delta, T), out["D_ref"][str(T)]
            row[f"gk_quad_rel_T{T}"] = (dq - dr) / dr
            row[f"gk_prod_rel_T{T}"] = (dp - dr) / dr
            row[f"gk_quad_bias_over_sem_T{T}"] = (dq - dr) / out["D_block_sem"][str(T)]
        rows.append(row)
    out["rows"] = rows
    out["vacf_ref_early"] = cref[:int(round(0.5 / DT)) + 1].tolist()
    out["vacf_prod_early"] = {str(d): (vacf(V, int(round(d / DT)), int(round(0.5 / d))) /
                                       vacf(V, int(round(d / DT)), 0)[0]).tolist() for d in DELTAS[1:]}
    return out


def choose(results):
    ok = {}
    for delta in DELTAS[1:]:
        rs = [r for res in results for r in res["rows"] if r["delta"] == delta]
        ok[delta] = all(max(abs(r[f"gk_quad_rel_T{T}"]) for T in T_GK) <= RULE["gk_bias_max"]
                        and (r["lags_before_half"] or 0) >= RULE["min_lags_before_half"]
                        and abs(r["tau_half_rel"]) <= RULE["tau_half_rel_max"] for r in rs)
    passing = [d for d in DELTAS[1:] if ok[d]]
    return (max(passing) if passing else DT), ok


def figure(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    ramp = {0.01: "#6da7ec", 0.025: "#2a78d6", 0.05: "#1c5cab", 0.1: "#0d366b"}     # sequential blue: larger = darker
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF, "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, law in zip(axes[:2], ("A", "B")):
        res = [r for r in results if r["law"] == law][0]
        tref = np.arange(len(res["vacf_ref_early"])) * DT
        for d in DELTAS[1:][::-1]:
            c = res["vacf_prod_early"][str(d)]
            ax.plot(np.arange(len(c)) * d, c, "-", color=ramp[d], lw=1.5, marker="o", ms=5, mec=SURF, mew=0.8,
                    label=f"stored every {d}")
        ax.plot(tref, res["vacf_ref_early"], ":", color=INK, lw=1.8, zorder=5, label="stored every step (0.005)")
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_xlim(0, 0.5)
        ax.set_title(f"Law {law}: normalised VACF, early decay (seed {res['seed']})", loc="left", fontsize=10)
        ax.set_xlabel("lag τ")
        ax.grid(color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("C(τ) / C(0)")
    axes[0].legend(fontsize=7, frameon=False, handlelength=3.5)
    ax = axes[2]
    for law, ls in (("A", "-"), ("B", "--")):
        rs = [r for res in results if res["law"] == law for r in res["rows"]]
        for T in (0.5, 3.0):
            y = [max(abs(r[f"gk_quad_rel_T{T}"]) for r in rs if r["delta"] == d) for d in DELTAS[1:]]
            ax.plot(DELTAS[1:], y, ls, marker="o", lw=2, ms=6, mec=SURF, color="#2a78d6" if T == 0.5 else "#eb6834",
                    label=f"law {law}, T = {T}")
    ax.axhline(RULE["gk_bias_max"], color=INK2, lw=1)
    ax.annotate("selection rule: 0.2 %", (0.04, RULE["gk_bias_max"] * 1.3), fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("save interval Δ")
    ax.set_ylabel("|Green-Kubo bias| (quadrature only), relative")
    ax.set_title("Green-Kubo quadrature bias vs every-step value", loc="left", fontsize=10)
    ax.grid(color=GRID, lw=0.8, which="major")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, frameon=False, handlelength=3.5)
    fig.tight_layout()
    fig.savefig(out / "vacf_sampling.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=str(HERE / "vacf_sampling_results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    results = [analyse(d) for d in args.runs]
    best, ok = choose(results)
    summary = dict(rule=RULE, deltas=DELTAS, T_gk=T_GK, passing={str(k): v for k, v in ok.items()},
                   recommended_save_interval=best, results=[{k: v for k, v in r.items() if not k.startswith("vacf_")}
                                                            for r in results])
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    rows = [r for res in results for r in res["rows"]]
    with open(out / "intervals.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    figure(results, out)
    for res in results:
        print(f"law {res['law']} seed {res['seed']}: tau_1/2 {res['tau_half_ref']:.4f}, tau_1/e {res['tau_1e_ref']:.4f}, "
              f"D_ref(T=3) {res['D_ref']['3.0']:.5f} +- {res['D_block_sem']['3.0']:.5f} (block SEM), Richardson "
              f"quadrature error of the reference {res['D_ref_quadrature_error_richardson']['3.0'] / res['D_ref']['3.0']:+.1e}")
        for r in res["rows"]:
            print(f"   Delta {r['delta']:5.3f}: C(Delta)/C0 {r['c_first_lag']:.3f} (ref {r['c_first_lag_ref']:.3f}), lags before 1/2 "
                  f"{r['lags_before_half']}, tau_1/2 {r['tau_half_rel']:+.2%}, GK quad bias " +
                  " ".join(f"T{T}:{r[f'gk_quad_rel_T{T}']:+.2%}" for T in T_GK) + " | prod-like " +
                  " ".join(f"T{T}:{r[f'gk_prod_rel_T{T}']:+.2%}" for T in T_GK))
    print("passing:", ok, "-> recommended save interval", best)


if __name__ == "__main__":
    main()
