#!/usr/bin/env python3
"""
Large-N extension of the fixed-rank direct-start Lanczos budget (N = 8000, 16000).

Same direct-start Lanczos as test_lanczos_fdt.py: q1 = z/||z||, literal PPPM Gamma_h action,
Pi after every action, three-term recurrence + two passes of full reorthogonalisation, Pi again.
Same density (30/216), same Gamma_h parameters (PARAMS), same dt normalisation
(dt = 0.5 m / lambda_max, lambda_max = largest Ritz value of the reference run), beta = m = 1.
Seeds: positions default_rng(1000 + N), Gaussian vectors = first 3 draws of default_rng(1001 + N).

Reference: eta at rank S >= 260 (increased only if needed), accepted only if the successive-rank
difference is <= 1e-12 and ||eta(S) - eta(S-20)|| / ||eta(S)|| <= 1e-11.
N <= 4000 results are reused from fixed_rank_budget_results/REPORT.md and lanczos_fdt_results/scaling.json.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_ewald_longitudinal import set_style, INK2  # noqa: E402
from test_lanczos_fdt import GammaH, PARAMS, tridiag, proj, MASS, BETA  # noqa: E402

RANKS = [32, 40, 48]
NS_NEW = [8000, 16000]
N_SEEDS = 3
TAU = 0.5
BUDGET = 1e-3
OUT = HERE / "fixed_rank_budget_results"


def lanczos_timed(op, z, smax):
    """Direct-start Lanczos identical to test_lanczos_fdt.lanczos, with cumulative per-step timings."""
    n = z.size
    Q = np.zeros((smax + 1, n))
    al, be = np.zeros(smax), np.zeros(smax)
    nz = np.linalg.norm(z)
    Q[0] = z / nz
    t_act, t_re = np.zeros(smax), np.zeros(smax)
    ca = cr = 0.0
    for k in range(smax):
        t0 = time.perf_counter()
        w = proj(op.matvec(Q[k]))
        ca += time.perf_counter() - t0
        t0 = time.perf_counter()
        al[k] = Q[k] @ w
        w = w - al[k] * Q[k] - (be[k - 1] * Q[k - 1] if k > 0 else 0.0)
        for _ in range(2):
            w -= Q[:k + 1].T @ (Q[:k + 1] @ w)
        w = proj(w)
        be[k] = np.linalg.norm(w)
        Q[k + 1] = w / be[k]
        cr += time.perf_counter() - t0
        t_act[k], t_re[k] = ca, cr
    return dict(Q=Q, al=al, be=be, nz=nz, s=smax, t_act=t_act, t_re=t_re)


def eta_at(L, r, f):
    t0 = time.perf_counter()
    th, U = np.linalg.eigh(tridiag(L["al"], L["be"], r))
    if th[0] <= 0:
        raise RuntimeError(f"non-positive Ritz value {th[0]:.3e} at rank {r}")
    e = proj(L["nz"] * (L["Q"][:r].T @ (U @ (f(th) * U[0]))))
    return e, th, time.perf_counter() - t0


def run_seed(op, z, N, say):
    for smax in (260, 360, 460, 600):
        L = lanczos_timed(op, z, smax)
        th_all = np.linalg.eigvalsh(tridiag(L["al"], L["be"], smax))
        lam_max = float(th_all[-1])
        dt = TAU * MASS / lam_max
        f = lambda l: np.sqrt(MASS / BETA * (-np.expm1(-2 * dt * l / MASS)))
        ref, _, _ = eta_at(L, smax, f)
        succ = np.linalg.norm(ref - eta_at(L, smax - 1, f)[0]) / np.linalg.norm(ref)
        d20 = np.linalg.norm(ref - eta_at(L, smax - 20, f)[0]) / np.linalg.norm(ref)
        ok = succ <= 1e-12 and d20 <= 1e-11
        say(f"    N={N}: reference rank {smax}: successive {succ:.1e}, 20-step {d20:.1e} -> {'accepted' if ok else 'extend'}")
        if ok:
            break
    else:
        raise RuntimeError("reference not converged by rank 600")
    row = dict(S_ref=smax, ref_succ=float(succ), ref_d20=float(d20), lam_max=lam_max, ritz_min_ref=float(th_all[0]),
               t_action_per=float(L["t_act"][-1] / smax))
    for r in RANKS:
        e, th, t_f = eta_at(L, r, f)
        row[f"err_{r}"] = float(np.linalg.norm(e - ref) / np.linalg.norm(ref))
        row[f"ritz_min_{r}"], row[f"ritz_max_{r}"] = float(th[0]), float(th[-1])
        row[f"mom_{r}"] = float(np.linalg.norm(e.reshape(N, 3).sum(axis=0)) / np.linalg.norm(e))
        row[f"t_actions_{r}"] = float(L["t_act"][r - 1])
        row[f"t_reorth_{r}"] = float(L["t_re"][r - 1])
        row[f"t_feval_{r}"] = float(t_f)
        row[f"t_root_{r}"] = row[f"t_actions_{r}"] + row[f"t_reorth_{r}"] + t_f
    return row


def old_table():
    """(median, max) over the 2 previous seeds for r = 32, 40 at N <= 4000, dt lambda_max/m = 0.5."""
    txt = (OUT / "REPORT.md").read_text().split("## dt lambda_max / m = 2.0")[0]
    old = {}
    for line in txt.splitlines():
        m = re.match(r"\| (\d+) \| (.+) \|$", line)
        if m and "/" in m.group(2):
            cells = [c.strip() for c in m.group(2).split("|")]
            vals = [[float(v) for v in c.split("/")] for c in cells]      # r = 16, 24, 32, 40: median / p90 / max
            old[int(m.group(1))] = {32: (vals[2][0], vals[2][2]), 40: (vals[3][0], vals[3][2])}
    return old


def main():
    log = []
    say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(t) for t in a)))
    rows = []
    stop = False
    for N in NS_NEW:
        if stop:
            say(f"N={N}: skipped (rank 48 already failed the 1e-3 budget)")
            break
        dens = 30 / 216.0
        Lbox = (N / dens) ** (1 / 3)
        x = np.random.default_rng(1000 + N).uniform(0, Lbox, size=(N, 3))
        t0 = time.perf_counter()
        op = GammaH(x, Lbox, **PARAMS)
        say(f"N={N}: L={Lbox:.2f}, mesh M={op.mesh.M}, real-space pairs {len(op.w)}, setup {time.perf_counter() - t0:.1f}s")
        rng = np.random.default_rng(1001 + N)
        for v in range(N_SEEDS):
            z = proj(rng.normal(size=3 * N))
            row = run_seed(op, z, N, say)
            row.update(N=N, seed=v, L=Lbox, M=op.mesh.M)
            rows.append(row)
            say(f"    seed {v}: " + ", ".join(f"r={r}: {row[f'err_{r}']:.2e}" for r in RANKS)
                + f"; t_root(r=32) {row['t_root_32']:.2f}s, t_action {row['t_action_per'] * 1e3:.0f}ms")
        if max(r_["err_48"] for r_ in rows if r_["N"] == N) > BUDGET:
            stop = True
    (OUT / "large_n_raw.json").write_text(json.dumps(rows, indent=1) + "\n")

    # ---- combine with the stored N <= 4000 results -------------------------------------------
    old = old_table()
    prev = json.loads((HERE / "lanczos_fdt_results" / "scaling.json").read_text())
    agg = {}
    for N in sorted(old):
        agg[N] = {r: old[N][r] for r in (32, 40)}
        agg[N][48] = None
    for N in sorted({r["N"] for r in rows}):
        rr = [r for r in rows if r["N"] == N]
        agg[N] = {k: (float(np.median([r[f"err_{k}"] for r in rr])), float(max(r[f"err_{k}"] for r in rr))) for k in RANKS}
    Ns = sorted(agg)

    # timings: new N measured directly; N <= 4000 estimated from stored per-action and full-run reorth times
    tim = {}
    for N in Ns:
        if any(r["N"] == N for r in rows):
            rr = [r for r in rows if r["N"] == N]
            tim[N] = {k: dict(root=float(np.median([r[f"t_root_{k}"] for r in rr])),
                              actions=float(np.median([r[f"t_actions_{k}"] for r in rr])),
                              reorth=float(np.median([r[f"t_reorth_{k}"] for r in rr])),
                              feval=float(np.median([r[f"t_feval_{k}"] for r in rr])), measured=True) for k in RANKS}
        else:
            pp = [p for p in prev if p["N"] == N]
            ta = float(np.mean([p["t_action_per"] for p in pp]))
            tr = float(np.mean([p["t_reorth"] for p in pp]))
            tim[N] = {k: dict(root=k * ta + tr * (k / 400) ** 2, actions=k * ta, reorth=tr * (k / 400) ** 2, feval=None,
                              measured=False) for k in RANKS}

    def passes(k):
        # r = 48 has no stored values at N <= 4000; there it is bounded above by the r = 40 value
        if not all(N in agg for N in NS_NEW):
            return False
        return all((agg[N][k] if agg[N][k] is not None else agg[N][40])[1] <= BUDGET for N in Ns)

    passing = [k for k in RANKS if passes(k)]
    best = min(passing) if passing else None

    def expo(Nlist, k):
        y = [tim[N][k]["root"] for N in Nlist]
        return float(np.polyfit(np.log(Nlist), np.log(y), 1)[0])

    k_t = best if best else 48
    large = [N for N in Ns if N >= 2048]
    meas = [N for N in Ns if tim[N][k_t]["measured"]]
    expo_large = expo(large, k_t)
    expo_meas = expo(meas, k_t) if len(meas) >= 2 else None
    nlogn = {N: tim[N][k_t]["root"] / (N * np.log(N)) for N in Ns}

    # ---- report ------------------------------------------------------------------------------------
    R = ["", "---", "", "# Large-N extension (N = 8000, 16000)", "",
         "Script: `fixed_rank_large_n.py`; raw per-seed data: `large_n_raw.json`; figure: `fixed_rank_error_large_n.png`.",
         f"Same direct-start Lanczos, density 30/216, Γ_h parameters {PARAMS}, dt·λ_max/m = {TAU}, β = m = 1.",
         f"There are {N_SEEDS} fixed Gaussian seeds per new N. Rows for N ≤ 4000 are reused from the tables above",
         "(2 seeds, not rerun).", "",
         "Reference checks (new N):", ""]
    for r in rows:
        R.append(f"- N={r['N']} seed {r['seed']}: rank {r['S_ref']}, successive {r['ref_succ']:.1e} (≤ 1e-12), "
                 f"20-step {r['ref_d20']:.1e} (≤ 1e-11); λ_max {r['lam_max']:.3f}, smallest Ritz value {r['ritz_min_ref']:.3e}")
    R += ["", "Relative root-action error, median / max over seeds:", "", "| N | r=32 | r=40 | r=48 |", "|---|---|---|---|"]
    for N in Ns:
        cells = []
        for k in RANKS:
            if agg[N][k] is None:
                cells.append(f"≤ {agg[N][40][1]:.2e} (bounded by r=40; not evaluated)")
            else:
                cells.append(f"{agg[N][k][0]:.2e} / {agg[N][k][1]:.2e}")
        R.append(f"| {N} | " + " | ".join(cells) + " |")
    R += ["", "Diagnostics for the new N (median over seeds):", "",
          "| N | r | t_root [s] | Γ_h actions [s] | reorth [s] | f(T_r) [s] | Ritz min | Ritz max | max momentum residual |",
          "|---|---|---|---|---|---|---|---|---|"]
    for N in NS_NEW:
        rr = [r for r in rows if r["N"] == N]
        if not rr:
            continue
        for k in RANKS:
            R.append(f"| {N} | {k} | {np.median([r[f't_root_{k}'] for r in rr]):.2f} | {np.median([r[f't_actions_{k}'] for r in rr]):.2f} | "
                     f"{np.median([r[f't_reorth_{k}'] for r in rr]):.3f} | {np.median([r[f't_feval_{k}'] for r in rr]):.4f} | "
                     f"{min(r[f'ritz_min_{k}'] for r in rr):.3e} | {max(r[f'ritz_max_{k}'] for r in rr):.3f} | "
                     f"{max(r[f'mom_{k}'] for r in rr):.1e} |")
    R += ["", f"Root-action time at the chosen rank r = {k_t} and T/(N log N):", "",
          "| N | T_root [s] | source | T/(N log N) [s] |", "|---|---|---|---|"]
    for N in Ns:
        R.append(f"| {N} | {tim[N][k_t]['root']:.3f} | {'measured' if tim[N][k_t]['measured'] else 'estimated from stored timings'} | "
                 f"{nlogn[N]:.2e} |")
    ans = []
    a32 = {N: agg[N][32][1] for N in NS_NEW if N in agg}
    a40 = {N: agg[N][40][1] for N in NS_NEW if N in agg}
    a48 = {N: agg[N][48][1] for N in NS_NEW if N in agg}
    yn = lambda v: "yes" if v <= BUDGET else "no"
    ans.append(f"1. **Rank 32 at N = 8000:** {yn(a32[8000])} (max over seeds {a32[8000]:.2e}).")
    ans.append("2. **Rank 32 at N = 16000:** " + (f"{yn(a32[16000])} (max {a32[16000]:.2e})." if 16000 in a32 else "not run."))
    ans.append("3. **Rank 40:** " + ", ".join(f"N = {N}: {yn(v)} ({v:.2e})" for N, v in a40.items()) + ".")
    ans.append("4. **Rank 48:** " + ", ".join(f"N = {N}: {yn(v)} ({v:.2e})" for N, v in a48.items()) + ".")
    ans.append(f"5. **Smallest single rank below 1e-3 for every N = 128–16000:** "
               + (f"r = {best}." if best else "none of r = 32, 40, 48 (stopped as instructed)."))
    if best:
        vals = [nlogn[N] for N in Ns]
        ans.append(f"6. **T/(N log N) at r = {best}:** ranges from {min(vals):.2e} to {max(vals):.2e} s over N = 128–16000, "
                   f"a max/min ratio of {max(vals) / min(vals):.2f}. Over the large-N points (N ≥ 2048) it goes "
                   + ", ".join(f"{nlogn[N]:.2e}" for N in large) + ".")
    ans.append(f"7. **Timing exponent (T_root ∝ N^α at r = {k_t}):** α = {expo_large:.3f} over N = 2048–16000"
               + (f"; α = {expo_meas:.3f} over the measured N = 8000–16000 alone" if expo_meas is not None else "") + ".")
    R += ["", "## Answers", ""] + ans + ["",
          "Notes:",
          "- For N ≤ 4000, T_root is estimated as r × (stored per-action time) + (stored rank-400 reorth time) × (r/400)².",
          "  The new N are measured directly with the same code; the stored N ≤ 4000 timings come from an earlier run in the same environment, so the combined exponent mixes measured and estimated points.",
          "- Rank 48 was not part of the previous N ≤ 4000 set, so its column there is bounded by the rank-40 values",
          "  (the error decreased monotonically with rank in every run).",
          "- No dynamics claim is made."]
    with open(OUT / "REPORT.md", "a") as fh:
        fh.write("\n".join(R) + "\n")
    summary = dict(agg={str(N): {str(k): v for k, v in d.items()} for N, d in agg.items()},
                   timing={str(N): {str(k): v for k, v in d.items()} for N, d in tim.items()},
                   best_rank=best, timing_rank=k_t, exponent_2048_16000=expo_large, exponent_measured=expo_meas,
                   T_over_NlogN={str(N): v for N, v in nlogn.items()})
    (OUT / "large_n_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    say("\n".join(R))

    # ---- figures -------------------------------------------------------------------------------------
    set_style()
    PAL = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axs[0]
    for ci, k in enumerate(RANKS):
        pts = [(N, agg[N][k]) for N in Ns if agg[N][k] is not None]
        ax.loglog([p[0] for p in pts], [p[1][1] for p in pts], "o-", color=PAL[ci], label=f"r = {k} (max over seeds)")
        ax.loglog([p[0] for p in pts], [p[1][0] for p in pts], ":", color=PAL[ci], lw=1)
    ax.axhline(BUDGET, color=INK2, lw=1.0)
    ax.text(Ns[0], BUDGET * 1.25, "1e-3 budget", color=INK2, fontsize=8)
    ax.axvline(6000, color=INK2, lw=0.6, ls="--")
    ax.set(xlabel="N (fixed density)", ylabel="relative root-action error",
           title="Fixed-rank error vs N (right of dashed line: new runs)")
    ax.legend(fontsize=8)
    ax = axs[1]
    ax.loglog(Ns, [tim[N][k_t]["root"] for N in Ns], "o-", color=PAL[0], label=f"T_root, r = {k_t}")
    ax.loglog([N for N in Ns if tim[N][k_t]["measured"]], [tim[N][k_t]["root"] for N in Ns if tim[N][k_t]["measured"]],
              "s", color=PAL[1], ms=8, mfc="none", label="measured (new N)")
    nn = np.array(Ns, float)
    c = tim[Ns[-1]][k_t]["root"] / (nn[-1] * np.log(nn[-1]))
    ax.loglog(nn, c * nn * np.log(nn), color=INK2, lw=0.8, label="~ N log N (through last point)")
    ax.set(xlabel="N", ylabel="root-action time [s]", title=f"Root-action time at r = {k_t}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fixed_rank_error_large_n.png")
    (OUT / "large_n_run_log.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
