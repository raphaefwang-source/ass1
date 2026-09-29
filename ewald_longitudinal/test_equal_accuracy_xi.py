#!/usr/bin/env python3
"""
Equal-accuracy optimisation over the Ewald parameter xi.

For every xi and target tol, the remaining parameters (s, eta_N, p, and hence M) are
searched until the ACTUAL relative spectral error of the particle graph operator
    ||Gamma_h - Gamma||_2 / ||Gamma||_2 <= tol
holds in BOTH dense reference configurations (A: N=30, L=6; B: N=40, L=10), always with
rc = s/xi, kc = 2 xi s and eta_actual = kc h/pi < 1.  Only accepted parameter sets are
timed; the cost compared across xi is the cost of one Gamma_h X action at equal accuracy.

Search (explicit, reported):
    s     in s0' * {1, 1.025, 1.05, 1.075, 1.1},  s0' = sqrt(log(2/tol))
    p     in {3, ..., 8}
    eta_N in {0.8, 0.7, 0.6, 0.5, 0.4, 0.3}  (tried in this order = increasing M; first pass wins)
    M     = next 2*3*5-smooth integer >= ceil(L/h_m) (FFT-friendly), so h <= h_m and eta_actual <= eta_N
For each (tol, xi) the tested candidates are, per p, the passing (s, eta_N) with the smallest
mesh on the large system; these (<= 6) candidates are timed on A, B and on a timing-only
system C (N = 4000, L = 30, same parameters, accuracy calibrated on A and B).

Model unchanged; no Q projector; no clipping of Khat_L; no stabilisation of Gamma_h.
"""
import argparse
import json
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
from verify_periodic_graph_psd import Profiles, periodic_kernel  # noqa: E402
from test_parameter_selection import (Config, Mesh, lap_full, lattice_modes, khat_AB, fourier_pairs,  # noqa: E402
                                      ewald, count_images, real_space_action, ROUND)

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
KAPPA = 1.0
S_FACT = [1.0, 1.025, 1.05, 1.075, 1.1]
ETAS = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
PS = [3, 4, 5, 6, 7, 8]
M_C_MAX = 300                      # largest mesh timed on the N = 4000 system (memory/time bound)


def next_smooth(n):
    """Smallest m >= n whose prime factors are 2, 3, 5 (FFT-friendly; a prime M makes numpy's FFT very slow)."""
    m = int(n)
    while True:
        k = m
        for f in (2, 3, 5):
            while k % f == 0:
                k //= f
        if k == 1:
            return m
        m += 1


def make_mesh(L, K, kc, eta, p, rfft_grid=True):
    """Mesh with M = next_smooth(ceil(L/h_m)), h_m = eta pi/kc; the realised h <= h_m, eta_actual <= eta_N < 1."""
    M = next_smooth(np.ceil(L / (eta * np.pi / kc) - 1e-9))
    m = Mesh(L, K, kc, kc * (L / M) / np.pi * (1 + 1e-12), p, rfft_grid=rfft_grid)
    assert m.M == M and m.eta_actual < 1
    m.eta_requested = eta
    return m


def time_action(x, L, K, s, eta, p, reps=3, seed=0):
    rc, kc = s / K.xi, 2 * K.xi * s
    X = np.random.default_rng(seed).normal(size=x.shape)
    mesh = make_mesh(L, K, kc, eta, p)
    tr, td, tm = [], [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        _, nn = real_space_action(x, L, K, rc, X)
        t1 = time.perf_counter()
        idx, wt = mesh.flat(x)
        D = mesh.degree(idx, wt)
        t2 = time.perf_counter()
        mesh.graph_action(idx, wt, X, D)
        t3 = time.perf_counter()
        tr.append(t1 - t0)
        td.append(t2 - t1)
        tm.append(t3 - t2)
    return dict(neighbors=nn, n_modes=mesh.n_modes, M=mesh.M, M3=mesh.M ** 3, h=mesh.h, eta_actual=mesh.eta_actual,
                t_real=min(tr), t_mesh=min(td) + min(tm), t_total=min(tr) + min(td) + min(tm))


class Evaluator:
    def __init__(self, cfgs):
        self.cfgs = {c.name: c for c in cfgs}
        self._GS, self._tails, self._err = {}, {}, {}
        self.n_evals = 0

    def GS_rc(self, c, xi, s):
        key = (c.name, xi, s)
        if key not in self._GS:
            prof = Profiles(ewald(KAPPA, xi))
            self._GS[key] = c.lap(periodic_kernel(prof, c.d, "S", c.L, s / xi))
        return self._GS[key]

    def tails(self, c, xi, s):
        key = (c.name, xi, s)
        if key not in self._tails:
            GS, GL, _ = c.split(xi)
            eS = np.linalg.norm(self.GS_rc(c, xi, s) - GS, 2) / c.norm2
            mv, kv, _ = lattice_modes(c.L, 2 * xi * s)
            A, B = khat_AB(ewald(KAPPA, xi), np.linalg.norm(kv, axis=1))
            eF = np.linalg.norm(c.lap(fourier_pairs(c.d, kv, A, B, c.V)) - GL, 2) / c.norm2
            self._tails[key] = (float(eS), float(eF))
        return self._tails[key]

    def Gh(self, c, xi, s, eta, p):
        mesh = make_mesh(c.L, ewald(KAPPA, xi), 2 * xi * s, eta, p, rfft_grid=False)
        W, _ = mesh.pair_matrix(c.x)
        return self.GS_rc(c, xi, s) + lap_full(W), mesh

    def err(self, c, xi, s, eta, p):
        key = (c.name, xi, s, eta, p)
        if key not in self._err:
            Gh, mesh = self.Gh(c, xi, s, eta, p)
            self.n_evals += 1
            self._err[key] = (float(np.linalg.norm(Gh - c.G, 2) / c.norm2), mesh.M, mesh.eta_actual)
        return self._err[key]

    def full_metrics(self, c, xi, s, eta, p, rng):
        Gh, mesh = self.Gh(c, xi, s, eta, p)
        D = Gh - c.G
        err_abs = float(np.linalg.norm(D, 2))
        X = rng.normal(size=(100, 3 * c.N))
        probe = np.linalg.norm(X @ D.T, axis=1) / np.linalg.norm(X @ c.G.T, axis=1)
        nGh = float(np.linalg.norm(Gh, 2))
        lam_h = c.restricted_min(Gh)
        eS, eF = self.tails(c, xi, s)
        return dict(config=c.name, xi=xi, s=s, rc=s / xi, kc=2 * xi * s, eta_N=eta, p=p, M=mesh.M, h=mesh.h,
                    eta_actual=mesh.eta_actual, neighbors=count_images(c.d, c.L, s / xi), n_modes=mesh.n_modes,
                    M3=mesh.M ** 3, err_2=err_abs / c.norm2, err_abs=err_abs,
                    err_F=float(np.linalg.norm(D) / np.linalg.norm(c.G)), probe_median=float(np.median(probe)),
                    probe_max=float(probe.max()), eps_S=eS, eps_F=eF, lam_star=c.lam_star, lam_h=lam_h,
                    sym_resid=float(np.linalg.norm(Gh - Gh.T) / np.linalg.norm(Gh)),
                    trans_resid=float(max(np.linalg.norm(Gh @ c.T[:, k]) for k in range(3)) / nGh),
                    psd_observed=bool(lam_h >= -ROUND * nGh), certified=bool(err_abs < c.lam_star),
                    weyl_holds=bool(lam_h >= c.lam_star - err_abs - 1e-13 * c.norm2))


def search(ev, tol, xi, say):
    """Explicit search for passing (s, eta, p) in BOTH configurations; returns all passes found."""
    cA, cB = ev.cfgs["A"], ev.cfgs["B"]
    s0 = np.sqrt(np.log(2.0 / tol))
    passes, skipped = [], []
    for f in S_FACT:
        s = s0 * f
        tA, tB = ev.tails(cA, xi, s), ev.tails(cB, xi, s)
        if sum(tA) > 0.95 * tol or sum(tB) > 0.95 * tol:
            skipped.append((round(f, 3), max(sum(tA), sum(tB))))
            continue
        for p in PS:
            for eta in ETAS:
                eB, MB, etaB = ev.err(cB, xi, s, eta, p)
                if eB > tol:
                    continue
                eA, MA, _ = ev.err(cA, xi, s, eta, p)
                if eA > tol:
                    continue
                kc = 2 * xi * s
                M_C = next_smooth(np.ceil(30.0 / (eta * np.pi / kc) - 1e-9))
                passes.append(dict(tol=tol, xi=xi, s=s, s_factor=f, eta_N=eta, p=p, err_A=eA, err_B=eB,
                                   M_A=MA, M_B=MB, M_C=M_C))
                break
    if skipped:
        say(f"    tol={tol:.0e} xi={xi:g}: s factors skipped (Ewald tails alone > 0.95 tol): {skipped}")
    # tested candidates: per p, the pass with the smallest large-system mesh (then smaller s)
    cands = []
    for p in PS:
        pp = [r for r in passes if r["p"] == p]
        if pp:
            cands.append(min(pp, key=lambda r: (r["M_C"], r["s"])))
    return passes, cands


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tols", type=float, nargs="+", default=[1e-5, 1e-7, 1e-9])
    ap.add_argument("--xis", type=float, nargs="+", default=[0.5, 0.7, 1.0, 1.4, 2.0])
    args = ap.parse_args()
    out = HERE / "equal_accuracy_xi_results"
    out.mkdir(exist_ok=True)
    log = []
    say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(t) for t in a)))
    rng = np.random.default_rng(777)
    t0 = time.perf_counter()
    cfgs = [Config("A", 30, 6.0, 7, KAPPA), Config("B", 40, 10.0, 13, KAPPA)]
    big = Config("C", 4000, 30.0, 99, KAPPA, reference=False)
    ev = Evaluator(cfgs)

    all_pass, timed = [], []
    for tol in args.tols:
        for xi in args.xis:
            passes, cands = search(ev, tol, xi, say)
            all_pass += passes
            say(f"  tol={tol:.0e} xi={xi:g}: {len(passes)} passing (s, eta, p) combos, {len(cands)} candidates "
                f"(evaluations so far {ev.n_evals}, {time.perf_counter() - t0:.0f}s)")
            for c in cands:
                K = ewald(KAPPA, xi)
                row = dict(c)
                for cfg in cfgs:
                    T = time_action(cfg.x, cfg.L, K, c["s"], c["eta_N"], c["p"], reps=9)
                    row[f"t_{cfg.name}"] = T["t_total"]
                    row[f"t_real_{cfg.name}"] = T["t_real"]
                    row[f"t_mesh_{cfg.name}"] = T["t_mesh"]
                if c["M_C"] <= M_C_MAX:
                    T = time_action(big.x, big.L, K, c["s"], c["eta_N"], c["p"], reps=2 if c["M_C"] <= 200 else 1)
                    row.update(t_C=T["t_total"], t_real_C=T["t_real"], t_mesh_C=T["t_mesh"],
                               neighbors_C=T["neighbors"], n_modes_C=T["n_modes"], M3_C=T["M3"])
                else:
                    row.update(t_C=np.nan, t_real_C=np.nan, t_mesh_C=np.nan, neighbors_C=np.nan, n_modes_C=np.nan,
                               M3_C=c["M_C"] ** 3)
                timed.append(row)
                say(f"     p={c['p']} s={c['s']:.3f} eta={c['eta_N']} M_A={c['M_A']} M_B={c['M_B']} M_C={c['M_C']} "
                    f"err_A={c['err_A']:.1e} err_B={c['err_B']:.1e} | t_A={row['t_A'] * 1e3:.2f}ms "
                    f"t_B={row['t_B'] * 1e3:.2f}ms t_C={row['t_C'] * 1e3:.0f}ms")

    # ---- cheapest accepted run per (tol, xi, system) ------------------------
    best = []
    for tol in args.tols:
        for xi in args.xis:
            rows = [r for r in timed if r["tol"] == tol and r["xi"] == xi]
            for sysname in ("A", "B", "C"):
                ok = [r for r in rows if np.isfinite(r[f"t_{sysname}"])]
                if not ok:
                    best.append(dict(tol=tol, xi=xi, system=sysname, feasible=False))
                    continue
                b = min(ok, key=lambda r: r[f"t_{sysname}"])
                acc = ev.full_metrics(ev.cfgs["B" if sysname == "C" else sysname], xi, b["s"], b["eta_N"], b["p"], rng)
                acc.update(tol=tol, system=sysname, feasible=True, t_real=b[f"t_real_{sysname}"],
                           t_mesh=b[f"t_mesh_{sysname}"], t_total=b[f"t_{sysname}"], s_factor=b["s_factor"],
                           accuracy_checked_on="A and B")
                if sysname == "C":
                    acc.update(neighbors=b["neighbors_C"], n_modes=b["n_modes_C"], M3=b["M3_C"], M=b["M_C"],
                               h=30.0 / b["M_C"], eta_actual=2 * xi * b["s"] * (30.0 / b["M_C"]) / np.pi,
                               metrics_config="B (dense reference)", M_B=b["M_B"], h_B=10.0 / b["M_B"])
                best.append(acc)

    # ---- tables -------------------------------------------------------------
    cols = ["tol", "system", "xi", "s", "rc", "kc", "eta_N", "p", "M", "h", "eta_actual", "neighbors", "n_modes",
            "M3", "t_real", "t_mesh", "t_total", "err_2", "err_F", "probe_max", "lam_star", "lam_h", "err_abs",
            "psd_observed", "certified"]
    fmt = lambda v: ("yes" if v is True else "no" if v is False else f"{v:.4g}" if isinstance(v, (float, np.floating))
                     else str(v))
    md = ["# Equal-accuracy xi optimisation: cheapest accepted run per (tol, xi, system)", "",
          "Every row meets ||Gamma_h - Gamma||_2/||Gamma||_2 <= tol in BOTH dense configurations A and B. "
          "Error/eigenvalue columns come from the dense reference (for system C: configuration B); "
          "times are for one Gamma_h X action on the named system.", ""]
    for tol in args.tols:
        md += [f"## tol = {tol:.0e}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in [r for r in best if r["tol"] == tol and r["feasible"]]:
            md.append("| " + " | ".join(fmt(r[k]) for k in cols) + " |")
        md.append("")
    (out / "accepted_runs.md").write_text("\n".join(md) + "\n")
    clean = lambda rows: [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()} for r in rows]
    for name, rows in (("all_passing", all_pass), ("timed_candidates", timed), ("accepted_best", best)):
        (out / f"{name}.json").write_text(json.dumps(clean(rows), indent=1) + "\n")
        ks = sorted({k for r in rows for k in r})
        (out / f"{name}.csv").write_text(",".join(ks) + "\n" + "\n".join(",".join(str(r.get(k, "")) for k in ks)
                                                                        for r in rows) + "\n")

    # ---- plots --------------------------------------------------------------
    set_style()
    fig, axs = plt.subplots(1, len(args.tols), figsize=(5.2 * len(args.tols), 4.4))
    axs = np.atleast_1d(axs)
    for ax, tol in zip(axs, args.tols):
        for i, sysname in enumerate(("A", "B", "C")):
            rr = [r for r in best if r["tol"] == tol and r["system"] == sysname and r["feasible"]]
            ax.semilogy([r["xi"] for r in rr], [r["t_total"] * 1e3 for r in rr], "os^"[i] + "-", color=PAL[i],
                        label={"A": "A: N=30, L=6", "B": "B: N=40, L=10", "C": "C: N=4000, L=30 (timing)"}[sysname])
            if rr:
                b = min(rr, key=lambda r: r["t_total"])
                ax.plot([b["xi"]], [b["t_total"] * 1e3], "o", ms=12, mfc="none", mec=PAL[i])
        ax.set(xlabel="xi", ylabel="minimum measured action time [ms]",
               title=f"tol = {tol:.0e}: cheapest accepted run vs xi")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "min_cost_vs_xi.png")
    plt.close(fig)
    for tol in args.tols:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for i, sysname in enumerate(("A", "B", "C")):
            rr = [r for r in best if r["tol"] == tol and r["system"] == sysname and r["feasible"]]
            ax.semilogy([r["xi"] for r in rr], [r["t_total"] * 1e3 for r in rr], "os^"[i] + "-", color=PAL[i],
                        label=f"system {sysname}")
        ax.set(xlabel="xi", ylabel="minimum measured action time [ms]", title=f"tol = {tol:.0e}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"min_cost_vs_xi_tol{tol:.0e}.png")
        plt.close(fig)

    # ---- summary --------------------------------------------------------------
    S = ["# Equal-accuracy xi optimisation: summary", ""]
    for tol in args.tols:
        for sysname in ("A", "B", "C"):
            rr = [r for r in best if r["tol"] == tol and r["system"] == sysname and r["feasible"]]
            if not rr:
                continue
            b = min(rr, key=lambda r: r["t_total"])
            S.append(f"- tol {tol:.0e}, system {sysname}: " + ", ".join(
                f"xi={r['xi']:g}: {r['t_total'] * 1e3:.2f} ms (p={r['p']}, eta={r['eta_N']}, M={r['M']})" for r in rr)
                + f"  ->  cheapest tested xi = {b['xi']:g}")
    inf = [r for r in best if not r["feasible"]]
    if inf:
        S.append("- not feasible / not timed: " + ", ".join(f"tol {r['tol']:.0e} xi {r['xi']:g} system {r['system']}"
                                                           for r in inf))
    S.append(f"- all accepted runs PSD observed: {all(r['psd_observed'] for r in best if r['feasible'])}; "
             f"certified: {sum(r['certified'] for r in best if r['feasible'])}/{sum(r['feasible'] for r in best)}")
    S.append(f"- dense evaluations: {ev.n_evals}; wall time {time.perf_counter() - t0:.0f}s")
    text = "\n".join(S)
    say("\n" + text)
    (out / "summary.md").write_text(text + "\n")
    (out / "run_log.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
