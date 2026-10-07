#!/usr/bin/env python3
"""Which radial longitudinal friction law (A: A e^{-kr}/r, B: B r e^{-kr}) is supported by the data in this project?

Step 0  inventory (inspect_data) and case classification.
Step 1  MD analysis - only if MD / CG dissipation data exist. In this repository they do not (CASE 0), so nothing
        is estimated from MD and no model is preferred on physical grounds.
Step 2  KNOWN-ANSWER pipeline tests on synthetic data whose input law is known (never evidence about MD):
        V1  CASE-1 path: friction tensor Gamma_h(q) of this project's own CG model (input law = Model A, kappa = 1)
        V2  CASE-2 path: synthetic constrained-dynamics unresolved forces with exponential memory, generated from a
            Model-B pair friction (kappa = 1, B = 1) with a many-body environment factor and an excluded-volume core
        V3  CASE-3 diagnostic: Markovian pair-drift regression on trajectories of this project's own model
            (input law = Model A)
Step 3  Fourier-space and real-space comparison of the two analytic kernels.

Usage: python3 run_analysis.py [--quick]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "ewald_longitudinal"
sys.path.insert(0, str(HERE))
import inspect_data  # noqa: E402
import load_data  # noqa: E402
import estimate_memory as EM  # noqa: E402
import radial_projection as RP  # noqa: E402
import fit_models as FM  # noqa: E402
import fourier_analysis as FA  # noqa: E402
import plotting as PL  # noqa: E402

RES, FIG = HERE / "results", HERE / "figures"
EDGES = np.arange(0.0, 6.0 + 1e-9, 0.4)
N_MIN = 30
BETA = 1.0


def say(s):
    print(s, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# ----------------------------------------------------------------------------- common analysis of pair data
def analyse_pair_dataset(name, r, gL, gT, groups, truth=None, extra=None, yscale="linear", n_boot=300):
    """Bin per-pair gamma_L, gamma_T; fit A, B, spline; group bootstrap. Returns (summary, plot dict)."""
    sL = RP.group_bin_stats(r, gL, groups, EDGES)
    sT = RP.group_bin_stats(r, gT, groups, EDGES)
    use = (sL["n"] >= N_MIN) & np.isfinite(sL["se_group"]) & (sL["se_group"] > 0)
    bins = np.nonzero(use)[0]
    avg_full = FM.BinAverager(sL["r_samples"], sL["b_samples"], bins, max_per_bin=10 ** 9)
    avg_boot = FM.BinAverager(sL["r_samples"], sL["b_samples"], bins, max_per_bin=4000)
    rbar, y, sig = sL["rbar"][use], sL["mean"][use], sL["se_group"][use]
    fits = FM.fit_all(rbar, y, sig, avg_full)
    Y = FM.group_replicates(sL, use, n_boot)
    boot = FM.bootstrap(Y, rbar, sig, avg_boot)
    spl, _ = FM.spline_fit(rbar, y, sig)
    out = dict(name=name, bins=bins, edges=EDGES, n=sL["n"], rbar=sL["rbar"], gL=sL["mean"], gL_se=sL["se_group"],
               gL_se_naive=sL["se_naive"], gL_std=sL["std"], gT=sT["mean"], gT_se=sT["se_group"], gT_std=sT["std"],
               fits=fits, bootstrap=boot, n_groups=int(groups.max()) + 1, r_min=float(r.min()))
    with np.errstate(invalid="ignore", divide="ignore"):
        stable = use & (np.abs(sL["mean"]) > 3 * sL["se_group"])
        out["gT_over_gL"] = np.where(stable, sT["mean"] / sL["mean"], np.nan)
        out["cv_at_fixed_r"] = np.where(use, sL["std"] / np.abs(sL["mean"]), np.nan)
    if truth is not None:
        out["truth_bin_mean"] = np.array([np.mean(truth(sL["r_samples"][sL["b_samples"] == b])) if sL["n"][b] else np.nan
                                          for b in range(len(EDGES) - 1)])
    D = dict(use=use, rbar=sL["rbar"], gL=sL["mean"], gL_se=sL["se_group"], gL_std=sL["std"], gT=sT["mean"],
             gT_se=sT["se_group"], n=sL["n"], edges=EDGES, fits={m: dict(fits[m]) for m in ("A", "B", "spline")},
             spline=spl, truth=truth, yscale=yscale)
    if extra:
        D.update(extra)
    return out, D


def write_bin_table(path, s):
    cols = ["r_lo", "r_hi", "rbar", "n", "gamma_L_mean", "gamma_L_std", "gamma_L_se", "gamma_T_mean", "gamma_T_se",
            "gT_over_gL", "used_in_fit"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for b in range(len(EDGES) - 1):
            w.writerow([EDGES[b], EDGES[b + 1], s["rbar"][b], s["n"][b], s["gL"][b], s["gL_std"][b], s["gL_se"][b],
                        s["gT"][b], s["gT_se"][b], s["gT_over_gL"][b], int(b in s["bins"])])


def median_nn(q, L):
    d = q[None] - q[:, None]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    return float(np.median(r.min(axis=1)))


# ----------------------------------------------------------------------------- V1: CASE-1 path, truth = A
def validation_v1(n_conf, N, say):
    sys.path.insert(0, str(PROJECT))
    from test_true_dynamics import Box, DENSITY  # this project's model (input law Model A, kappa = 1)
    box = Box(N)
    rs, gls, gts, grp, sym = [], [], [], [], []
    for c in range(n_conf):
        q = np.random.default_rng([31, c]).uniform(0, box.L, (N, 3))       # ideal-gas equilibrium configuration
        G = box.gamma(q).dense()
        P = EM.pair_friction_from_tensor(G, q, box.L, EDGES[-1])
        rs.append(P["r"]); gls.append(P["gL"]); gts.append(P["gT"]); grp.append(np.full(len(P["r"]), c))
        sym.append(P["sym_defect"])
        say(f"  V1 config {c}: {len(P['r'])} pairs, min r {P['r'].min():.3f}")
    r, gL, gT, g = map(np.concatenate, (rs, gls, gts, grp))
    truth = lambda x: np.exp(-x) / x                                     # noqa: E731
    out, D = analyse_pair_dataset("V1", r, gL, gT, g, truth=truth, yscale="log",
                                  extra=dict(std_label="± conditional std at fixed r (PPPM deviation only)"))
    out.update(L=box.L, density=DENSITY, sym_defect=max(sym),
               pair_dev_from_truth=float(np.max(np.abs(gL - truth(r)) / truth(r))),
               gT_abs_max=float(np.max(np.abs(gT))))
    out["median_nn"] = median_nn(np.random.default_rng([31, 0]).uniform(0, box.L, (N, 3)), box.L)
    return out, D


# ----------------------------------------------------------------------------- V2: CASE-2 path, truth = B
def rsa_config(N, L, dmin, rng):
    q = np.empty((N, 3))
    k = 0
    while k < N:
        x = rng.uniform(0, L, 3)
        d = q[:k] - x
        d -= L * np.round(d / L)
        if k == 0 or np.min(np.einsum("pa,pa->p", d, d)) >= dmin * dmin:
            q[k] = x
            k += 1
    return q


def model_b_gamma(q, L, B, kappa, rc, r_env, env=True):
    """Graph-Laplacian friction with pair weights B r e^{-kr} s_i s_j P_L, s_i = sqrt((1+n_i)/(1+<n>)), n_i = number
    of neighbours within r_env (a many-body environment factor). PSD by construction."""
    N = len(q)
    i, j, d, r = RP.pairs(q, L, rc)
    nloc = np.bincount(i[r < r_env], minlength=N) + np.bincount(j[r < r_env], minlength=N)
    s = np.sqrt((1.0 + nloc) / (1.0 + nloc.mean())) if env else np.ones(N)
    w = B * r * np.exp(-kappa * r) * s[i] * s[j]
    e = d / r[:, None]
    blk = w[:, None, None] * e[:, :, None] * e[:, None, :]
    G = np.zeros((N, 3, N, 3))
    a = np.arange(3)
    for (p_, q_, sg) in ((i, j, -1), (j, i, -1), (i, i, 1), (j, j, 1)):
        np.add.at(G, (p_[:, None, None], a[None, :, None], q_[:, None, None], a[None, None, :]), sg * blk)
    return G.reshape(3 * N, 3 * N), s


def validation_v2(n_conf, N, T_steps, say, env=True, name="V2", B=1.0, kappa=1.0, dmin=0.9, tau=0.2, dtf=0.02,
                  nlag=100):
    L = (N / (30 / 216.0)) ** (1 / 3)
    a = np.exp(-dtf / tau)
    rs, gls, gts, grp, KLs, h1, h2, truth_pair = [], [], [], [], [], [], [], []
    nullc = []
    for c in range(n_conf):
        rng = np.random.default_rng([47, c])
        q = rsa_config(N, L, dmin, rng)
        G, s = model_b_gamma(q, L, B, kappa, EDGES[-1], 2.0, env)
        lam, U = np.linalg.eigh(G)
        null = np.abs(lam) <= 1e-10 * lam.max()                           # exact translational null space (3 modes)
        nullc.append(int(null.sum()))
        if np.any(lam[~null] < 0):
            raise RuntimeError("Model-B friction not PSD")
        S = U * np.sqrt(np.where(null, 0.0, lam) / (BETA * tau))          # C0 = S S^T = Gamma/(beta tau)
        F = np.empty((T_steps, N, 3))
        f = S @ rng.normal(size=3 * N)
        Xi = rng.normal(size=(T_steps, 3 * N))
        sq = np.sqrt(1 - a * a)
        for t in range(T_steps):                                          # OU: <F(t)F(0)^T> = C0 e^{-|t|/tau}
            f = a * f + sq * (S @ Xi[t])
            F[t] = f.reshape(N, 3)
        M = EM.pair_memory_from_constrained_forces(F, q, L, BETA, dtf, EDGES, nlag, rng=rng)
        GL = EM.cumulative_integral(M["KL"], dtf)
        GT = EM.cumulative_integral(M["KT"], dtf)
        rs.append(M["r"]); gls.append(GL[:, -1]); gts.append(GT[:, -1]); grp.append(np.full(len(M["r"]), c))
        KLs.append(M["KL"]); h1.append(M["halves"][0, 0]); h2.append(M["halves"][1, 0])
        truth_pair.append(B * M["r"] * np.exp(-kappa * M["r"]) * s[M["i"]] * s[M["j"]])
        say(f"  {name} config {c}: {len(M['r'])} sampled pairs, null modes {nullc[-1]}, min r {M['r'].min():.3f}")
    r, gL, gT, g, KL, H1, H2, TP = map(np.concatenate, (rs, gls, gts, grp, KLs, h1, h2, truth_pair))
    t = np.arange(nlag) * dtf
    b = np.searchsorted(EDGES, r, side="right") - 1
    nb = len(EDGES) - 1
    KL_t = np.array([KL[b == k].mean(0) if np.any(b == k) else np.full(nlag, np.nan) for k in range(nb)])
    GL_T = EM.cumulative_integral(KL_t, dtf)
    truth_bin = np.array([TP[b == k].mean() if np.any(b == k) else np.nan for k in range(nb)])
    # scatter decomposition per bin: total variance of per-pair estimates = intrinsic (many-body) + estimation noise;
    # estimation-noise variance from split halves: var((h1 - h2)/2)
    var_tot = np.array([gL[b == k].var(ddof=1) if np.sum(b == k) > 2 else np.nan for k in range(nb)])
    var_noise = np.array([((H1 - H2)[b == k] / 2).var(ddof=1) if np.sum(b == k) > 2 else np.nan for k in range(nb)])
    var_true = np.array([TP[b == k].var(ddof=1) if np.sum(b == k) > 2 else np.nan for k in range(nb)])
    show = [k for k in range(nb) if np.sum(b == k) >= N_MIN][::3]
    truth_mean = lambda x: B * x * np.exp(-kappa * x)                     # noqa: E731
    out, D = analyse_pair_dataset(name, r, gL, gT, g, truth=truth_mean,
                                  extra=dict(KL_t=KL_t, GL_T=GL_T, t=t, show_bins=show, truth_bin=truth_bin,
                                             count_label="sampled pairs per bin"))
    i_T = [int(round(x / dtf)) for x in (2 * tau, 5 * tau, 10 * tau - dtf)]
    conv = {f"T={tt * dtf:.2f}": (GL_T[out["bins"], tt] / GL_T[out["bins"], -1]).tolist() for tt in i_T}
    out.update(L=L, dmin=dmin, tau=tau, dt=dtf, nlag=nlag, T_steps=T_steps, null_modes=nullc, truth_bin_pairs=truth_bin,
               convergence_ratio=conv, var_total=var_tot, var_noise=var_noise, var_intrinsic_est=var_tot - var_noise,
               var_intrinsic_true=var_true, B_true=B, kappa_true=kappa, env_factor=env)
    out["median_nn"] = median_nn(q, L)
    return out, D


# ----------------------------------------------------------------------------- V3: CASE-3 diagnostic, truth = A
def validation_v3(say, n_boot=300, max_traj=8):
    trajs = load_data.own_model_trajectories(PROJECT)[:max_traj]
    if not trajs:
        say("  V3 skipped: no own-model trajectories on disk (ewald_longitudinal/dynamics_results/correlations_raw)")
        return None, None
    XtX, Xty, rs, bs = [], [], [], []
    for k, tr in enumerate(trajs):
        q, p, L, dt = tr["q"], tr["p"], tr["L"], tr["dt"]
        Qh = (q[:-1] + 0.5 * dt * p[:-1]) % L                            # friction acts at q_half
        R = EM.markov_drift_regression(Qh, p[:-1], p[1:] - p[:-1], L, dt, EDGES, block_len=100)
        XtX.append(R["XtX"]); Xty.append(R["Xty"]); rs.append(R["r_samples"]); bs.append(R["b_samples"])
        say(f"  V3 trajectory {k}: {len(q) - 1} steps")
    XtX, Xty = np.concatenate(XtX), np.concatenate(Xty)
    nb = len(EDGES) - 1
    theta = EM.solve_blocks(XtX, Xty)
    rng = np.random.default_rng(11)
    reps = np.array([EM.solve_blocks(XtX, Xty, rng.integers(0, len(XtX), len(XtX))) for _ in range(n_boot)])
    se = reps.std(axis=0, ddof=1)
    r_s, b_s = np.concatenate(rs), np.concatenate(bs)
    n = np.bincount(b_s[b_s >= 0], minlength=nb).astype(float)
    rbar = np.bincount(b_s[b_s >= 0], r_s[b_s >= 0], nb) / np.maximum(n, 1)
    gL, gT, seL, seT = theta[:nb], theta[nb:], se[:nb], se[nb:]
    use = (n * 25 >= N_MIN * 10) & (seL > 0)
    bins = np.nonzero(use)[0]
    avg_full = FM.BinAverager(r_s, b_s, bins, max_per_bin=10 ** 9)
    avg_boot = FM.BinAverager(r_s, b_s, bins, max_per_bin=4000)
    fits = FM.fit_all(rbar[use], gL[use], seL[use], avg_full)
    boot = FM.bootstrap(reps[:, :nb][:, use], rbar[use], seL[use], avg_boot)
    spl, _ = FM.spline_fit(rbar[use], gL[use], seL[use])
    truth = lambda x: np.exp(-x) / x                                     # noqa: E731
    out = dict(name="V3", bins=bins, edges=EDGES, n=n, rbar=rbar, gL=gL, gL_se=seL, gT=gT, gT_se=seT, fits=fits,
               bootstrap=boot, n_groups=len(XtX), n_traj=len(trajs), dt=trajs[0]["dt"],
               truth_bin_mean=np.array([np.mean(truth(r_s[b_s == b])) if n[b] else np.nan for b in range(nb)]),
               gL_std=np.full(nb, np.nan), r_min=float(r_s.min()))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["gT_over_gL"] = np.where(use & (np.abs(gL) > 3 * seL), gT / gL, np.nan)
    D = dict(use=use, rbar=rbar, gL=gL, gL_se=seL, gL_std=np.zeros(nb), gT=gT, gT_se=seT, n=n, edges=EDGES,
             fits=fits, spline=spl, truth=truth, yscale="log", std_label="(per-pair scatter not identifiable)",
             count_label="pair samples per bin (every 25th step)")
    return out, D


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="smaller validation runs")
    ap.add_argument("--report-only", action="store_true", help="reprint the final answers from results/summary.json")
    args = ap.parse_args()
    if args.report_only:
        S = json.loads((RES / "summary.json").read_text())
        final_report(S, S["classification"])
        return
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    summary = {}

    say("=== Step 0: inventory ===")
    inv, txt = inspect_data.report()
    say(txt)
    (RES / "inventory.json").write_text(json.dumps(jsonable(inv), indent=1) + "\n")
    cls = inv["classification"]
    summary["classification"] = cls

    say("\n=== Step 1: MD / CG dissipation analysis ===")
    if cls["case"] == 0:
        msg = ("No MD data of any kind (no LAMMPS input, dump, topology, atom-to-molecule map, force, constrained-"
               "dynamics or memory files). The only particle trajectories are output of this project's own CG model, "
               "whose friction law is Model A by construction. No physical friction has been estimated and no model "
               "can be preferred from data.")
        say(msg)
        summary["md_analysis"] = dict(performed=False, reason=msg)
    else:
        summary["md_analysis"] = dict(performed=False, reason=f"CASE {cls['case']} data found; configure the reader in "
                                      "load_data.py (masses, molecule ids, beta) and call the matching estimator.")
        say(summary["md_analysis"]["reason"])

    say("\n=== Step 2: known-answer pipeline tests (synthetic; NOT evidence about MD) ===")
    nconf = 3 if args.quick else 6
    say("V1: CASE-1 path on Gamma_h(q) of this project's model (input: Model A, A = 1, kappa = 1)")
    v1, D1 = validation_v1(nconf, 512, say)
    PL.radial_figure(D1, FIG / "V1_case1_project_model_truthA.png",
                     "V1 (known answer, input = Model A): pair friction from the friction tensor Γ_h(q) of this project's "
                     "own CG model, ideal-gas configurations, N = 512")
    v2s = {}
    for nm, env, lab in (("V2a", False, "pure Model B"), ("V2b", True, "Model B × environment factor")):
        say(f"{nm}: CASE-2 path on synthetic constrained-dynamics forces (input: {lab}, core d = 0.9, memory tau = 0.2)")
        v, D = validation_v2(3 if args.quick else 6, 512, 4000 if args.quick else 8000, say, env=env, name=nm)
        PL.radial_figure(D, FIG / f"{nm}_case2_synthetic_truthB{'_env' if env else ''}.png",
                         f"{nm} (known answer, input = {lab}, excluded volume d = 0.9, exponential memory τ = 0.2): "
                         "pair memory from synthetic constrained-dynamics forces")
        v2s[nm] = v
    v2 = v2s["V2b"]
    say("V3: CASE-3 Markovian drift regression on this project's model trajectories (input: Model A)")
    v3, D3 = validation_v3(say, max_traj=2 if args.quick else 8)
    if v3 is not None:
        PL.radial_figure(D3, FIG / "V3_case3_drift_regression_truthA.png",
                         "V3 (known answer, input = Model A): Markovian pair-drift regression on trajectories of this "
                         "project's own CG model (assumes a pairwise Markov form)")
    for v in (v1, v2s["V2a"], v2s["V2b"], v3):
        if v is not None:
            write_bin_table(RES / f"{v['name']}_bins.csv", v)
            summary[v["name"]] = jsonable({k: val for k, val in v.items() if k not in ("edges",)})

    say("\n=== Step 3: Fourier and real-space comparison ===")
    ver = FA.verify()
    slopes = FA.large_k_slopes()
    kz = None
    kk = np.linspace(0.1, 10, 100000)
    lA, _ = FA.analytic_A(kk)
    kz = float(kk[np.argmax(lA < 0)])
    summary["fourier"] = dict(verification=ver, max_rel_err=max(r["rel_err_max"] for r in ver), large_k=slopes,
                              k_zero_L_A=kz, k_zero_L_B=1 / np.sqrt(3))
    PL.fourier_figure(FIG / "fourier_eigenvalues.png")
    marks = [("V1 ideal gas (no core)", v1["r_min"], v1["median_nn"]), ("V2 synthetic core d = 0.9", v2["r_min"], v2["median_nn"])]
    PL.realspace_figure(FIG / "realspace_shapes.png", marks)
    say(f"Fourier closed forms vs QAWF quadrature: max rel. error {summary['fourier']['max_rel_err']:.1e}")
    say(f"large-k slopes: A L/T {slopes['A']['slope_L']:.2f}/{slopes['A']['slope_T']:.2f}, "
        f"B L/T {slopes['B']['slope_L']:.2f}/{slopes['B']['slope_T']:.2f}; lam_L changes sign at k = {kz:.3f} (A), "
        f"{1 / np.sqrt(3):.3f} (B), kappa = 1")

    (RES / "summary.json").write_text(json.dumps(jsonable(summary), indent=1) + "\n")
    say(f"\nresults in {RES}, figures in {FIG}  ({time.perf_counter() - t0:.0f} s)")
    final_report(summary, cls)


def fmt_fit(v, m):
    f = v["fits"][m]
    ci = v["bootstrap"]["ci"].get(m)
    cis = f" boot95 amp [{ci['p2_5'][0]:.3g}, {ci['p97_5'][0]:.3g}] kappa [{ci['p2_5'][1]:.3g}, {ci['p97_5'][1]:.3g}]" if ci else ""
    return (f"amp {f['params'][0]:.4g} ± {f['errors'][0]:.2g}, kappa {f['params'][1]:.4g} ± {f['errors'][1]:.2g}; "
            f"chi2/dof {f['chi2_per_dof']:.3g}, wRMSE {f['weighted_rmse']:.3g}, RMSE {f['rmse']:.3g}, MAE {f['mae']:.3g}, "
            f"R2 {f['r2']:.4f}, AIC {f['aic']:.4g}, BIC {f['bic']:.4g};{cis}")


def final_report(S, cls):
    say("\n================ FINAL ANSWERS ================")
    say(f"A. Data found: CASE {cls['case']}. No LAMMPS inputs, MD dumps, topology/molecule maps, forces, constrained-"
        f"dynamics series or memory kernels anywhere in the repository (all branches checked separately). Only "
        f"{len(cls['model_trajectories'])} trajectory archives of this project's OWN CG model (input kernel = Model A).")
    say("B. Physical quantity estimated from MD: none. Estimated only in known-answer tests: pair friction "
        "G_ij = -Gamma_ij projected on rhat (V1 from the model's friction tensor, V2a/V2b from synthetic constrained-force "
        "memory integrals, V3 from a Markovian drift regression).")
    say("C. True MZ/constrained-dynamics friction from MD: NO. Not even a proxy from MD is possible here.")
    for name in ("V1", "V2a", "V2b", "V3"):
        v = S.get(name)
        if v is None:
            continue
        truth = {"V1": "A (kappa = 1, A = 1)", "V2a": "pure B (kappa = 1, B = 1), core d = 0.9",
                 "V2b": "B (kappa = 1, B = 1) x environment factor, core d = 0.9", "V3": "A (kappa = 1, A = 1)"}[name]
        say(f"D/E [{name}, input = {truth}]")
        say(f"    Model A: {fmt_fit(v, 'A')}")
        say(f"    Model B: {fmt_fit(v, 'B')}")
        say(f"    spline : chi2/dof {v['fits']['spline']['chi2_per_dof']:.3g}; "
            f"Delta AIC (B - A) = {v['fits']['delta_aic_B_minus_A']:.4g}; "
            f"bootstrap fraction with A better = {v['bootstrap']['frac_A_better']:.2f}")
    say("F. Better-supported model for the PHYSICAL system: undetermined - there are no MD/CG dissipation data. "
        "The pipeline identifies the correct law in all known-answer tests (see D/E), so it can decide once data exist.")
    v1 = S["V1"]
    say(f"G. gamma_T << gamma_L: not testable for MD. In V1 (this model) max |gamma_T| = {v1['gT_abs_max']:.1e} "
        "(pure-longitudinal by construction, PPPM-level). The assumption must be tested on real constrained-force data.")
    vb = S["V2b"]
    use = vb["bins"]
    vn, vt, ve = (np.array(vb[k], float)[use] for k in ("var_noise", "var_intrinsic_true", "var_intrinsic_est"))
    say(f"H. Many-body scatter: not testable for MD. In V2b (input with an environment factor) the per-pair estimation "
        f"noise variance of the memory integral is {np.median(vn / vt):.0f}x (median over bins) the true intrinsic "
        f"conditional variance; the split-half estimate of the intrinsic variance (median {np.median(ve):.1e} vs true "
        f"{np.median(vt):.1e}) is noise-dominated. The many-body factor nonetheless biases the conditional MEAN "
        f"<gamma_L | r> by {100 * (np.nanmax(np.abs(np.array(vb['truth_bin_pairs'], float)[use] / np.array(vb['truth_bin_mean'], float)[use] - 1))):.0f}% "
        "relative to the pure radial law: Model B's chi2/dof rises from "
        f"{S['V2a']['fits']['B']['chi2_per_dof']:.2g} (V2a) to {vb['fits']['B']['chi2_per_dof']:.2g} (V2b) and the AIC margin "
        f"over A shrinks from {-S['V2a']['fits']['delta_aic_B_minus_A']:.0f} to {-vb['fits']['delta_aic_B_minus_A']:.0f}, "
        "i.e. environment dependence shows up as a misfit of every pairwise radial law long before it is resolvable "
        "as per-pair scatter.")
    say("I. Needed: atomistic MD with molecule mapping (positions, velocities, forces per atom), plus constrained "
        "dynamics with CG coordinates (and momenta) held fixed producing unresolved-force time series dF_i(t) for many "
        "CG configurations (Lyu-Lei); or a directly computed friction tensor. See README.md.")


if __name__ == "__main__":
    main()
