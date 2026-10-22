#!/usr/bin/env python3
"""
Experiment 2: Candidate 1 (theta = 0, existing positive sharp Laplace split) vs Candidate 2 (theta = 1/2, isotropic
Gaussian-corrected split) on the SAME kernel (c = 1), positions, dense references (raw/exp_refs_c1, shared with
accuracy_sweeps), probe vectors, cutoffs, mesh, B-spline order and FFT implementation. Only the splitting changes:
    K_S^theta = K_S^0 - c theta G(r) I,  K_L^theta = K_L^0 + c theta G(r) I,  G(r) = g_3(xi^2) e^{-xi^2 r^2},
    Khat_L^theta = Khat_L^0 + c theta g_3(xi^2) (pi/xi^2)^{3/2} e^{-k^2/(4 xi^2)} I     (k = 0 included).
No eigenvalue clipping or PSD repair anywhere.

Pilot: N = 256, rho = 0.75, xi = 0.85, kappa in (0.7, 0.2, 2.0), uniform / moderate / strong, seed 1 (--seeds to
expand); s_R = 2 ... 6.5 with s_F = 6.5, and s_F = 2 ... 6.5 with s_R = 6.5 (the headline points 3, 4, 5 are in
the grid); M = 80, p = 9; every point also with the mesh-free Fourier sum (E_G_meshfree) and the mesh-only error so
that mesh error cannot mask cutoff differences.

Stages: smoke | checks | pilot | ratios | eqacc | eqtime | plots | tables | all (= checks pilot ratios eqacc eqtime
plots tables)

    python3 run_candidate2_pilot.py smoke
    python3 run_candidate2_pilot.py all --workers 3 [--seeds 1 2 3]
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "accuracy_sweeps"))

import ewald_exp_lib as E                # noqa: E402
import common as C                       # noqa: E402
import numpy as np                       # noqa: E402
from scipy import integrate, special     # noqa: E402

from neighbor_images import kdtree_images  # noqa: E402
import run_accuracy_sweeps as A1         # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
THETAS = (0.0, 0.5)
KAPPAS = (0.7, 0.2, 2.0)
S_GRID = A1.S_GRID
S_ACC = 6.5                               # "sufficiently accurate" companion cutoff
HEADLINE = (3.0, 4.0, 5.0)
PILOT = dict(E.BASE, M=80, p=9)
TOLS = (1e-4, 1e-6, 1e-8, 1e-10)
P_EQ = (4, 5, 6, 7, 8, 9)


# ============================================================================ checks
def tensors6(theta_long, iso, d):
    r2 = np.einsum("pa,pa->p", d, d)
    w = theta_long / r2
    T = np.stack([w * d[:, a] * d[:, b] for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], 1)
    T[:, :3] += iso[:, None]
    return T


def fro6(T):
    return np.sqrt(np.sum(T[:, :3] ** 2, axis=-1) + 2 * np.sum(T[:, 3:] ** 2, axis=-1))


def stage_checks(args):
    rec, rows = {}, []
    xi = E.BASE["xi"]
    # (1) g_3 = Laplace density of e^{-kappa r}/r^3, and theta_S e^{xi^2 r^2} -> g_3(xi^2) (leading tail)
    for kap in KAPPAS:
        K = E.kernel(kap, xi, 0.5)
        r = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]) / xi
        tail = {}
        rr = np.geomspace(0.05, 12, 60)
        # int_0^inf g3(s) e^{-s r^2} ds by tanh-sinh-free composite quadrature in u = log s
        u = np.linspace(-30, 14, 40001)
        S = np.exp(u)
        f = E.rho_density(S, kap)[None, :] * np.exp(-np.outer(rr ** 2, S)) * S[None, :]
        lap = integrate.simpson(f, x=u, axis=1)
        rec[f"g3_laplace_kappa{kap:g}_max_rel"] = float(np.max(np.abs(lap - np.exp(-kap * rr) / rr ** 3)
                                                            / (np.exp(-kap * rr) / rr ** 3)))
        for rv in r:
            tail[f"{xi * rv:g}"] = float(K.theta_S_scaled_closed(np.array([rv]))[0] / K.g3)
        rec[f"thetaS_scaled_over_g3_kappa{kap:g}_vs_xi_r"] = tail
        rec[f"g3_kappa{kap:g}"] = K.g3
    # (2) real-space splitting identity K = K_S^theta + K_L^theta on all pair images |d| <= 8 and a radial sweep
    for kap in KAPPAS:
        for theta in THETAS:
            K = E.kernel(kap, xi, theta)
            worst = 0.0
            for fam in E.FAMS:
                for sd in E.SEEDS:
                    cfg = E.config(fam, sd, 0.75)
                    il = kdtree_images(cfg["q"], cfg["L"], 8.0)
                    r = np.linalg.norm(il.d, axis=1)
                    T = tensors6(K.theta_full(r), np.zeros_like(r), il.d)
                    TS = tensors6(K.theta_S(r), K.iso_S(r), il.d)
                    TL = tensors6(K.theta_L(r), K.iso_L(r), il.d)
                    dT = T - TS - TL
                    rel = float(np.sqrt(np.sum(fro6(dT) ** 2) / np.sum(fro6(T) ** 2)))
                    pt = float(np.max(fro6(dT) / fro6(T)))
                    worst = max(worst, pt)
                    rows.append(dict(check="split_identity_configs", kappa=kap, theta=theta, config=cfg["name"],
                                     n_images=int(len(r)), rel_fro=rel, max_rel_pointwise=pt))
            rr = np.geomspace(1e-6, 12.0, 601)
            dirs = np.random.default_rng(3).standard_normal((rr.size, 3))
            d = rr[:, None] * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
            dT = tensors6(K.theta_full(rr), 0 * rr, d) - tensors6(K.theta_S(rr), K.iso_S(rr), d) - \
                tensors6(K.theta_L(rr), K.iso_L(rr), d)
            relr = fro6(dT) / fro6(tensors6(K.theta_full(rr), 0 * rr, d))
            rows.append(dict(check="split_identity_radial", kappa=kap, theta=theta, n_images=int(rr.size),
                             max_rel_pointwise=float(relr.max()), r_at_max=float(rr[np.argmax(relr)])))
            # eigenvalues of the short-range pair tensor (Candidate 2: negative transverse eigenvalue)
            rr2 = np.linspace(0.8, 8.0 / xi, 400)
            rows.append(dict(check="KS_pair_eigs", kappa=kap, theta=theta,
                             min_transverse=float(np.min(K.iso_S(rr2))),
                             min_longitudinal=float(np.min(K.theta_S(rr2) + K.iso_S(rr2))),
                             max_abs_tail_ratio_vs_theta0=float(np.max(np.maximum(np.abs(K.theta_S(rr2) + K.iso_S(rr2)),
                                                                                  np.abs(K.iso_S(rr2))) / K.theta_S(rr2)))))
    # (3) analytic Fourier correction vs direct evaluations
    for kap in KAPPAS:
        K = E.kernel(kap, xi, 0.5)
        k = np.linspace(0.0, 15.0, 61)
        rq = np.linspace(0, 12 / xi, 40001)
        Gr = K.beta0 * np.exp(-(xi * rq) ** 2)
        num = 4 * np.pi * integrate.simpson(rq[None, :] ** 2 * Gr[None, :] * np.sinc(np.outer(k, rq) / np.pi), x=rq, axis=1)
        A0 = E.kernel(kap, xi, 0.0).AB(k)[0]
        Ah = K.AB(k)[0]
        rec[f"fourier_correction_radialFT_kappa{kap:g}_max_rel"] = float(np.max(np.abs((Ah - A0) - num)) / np.max(np.abs(num)))
        # Poisson: direct periodic image sum of G vs lattice Fourier sum (k = 0 included) on configuration pairs
        cfg = E.config("strong", 1, 0.75)
        q, L = cfg["q"], cfg["L"]
        pi_, pj_ = np.triu_indices(len(q), 1)
        sel = np.random.default_rng(7).choice(len(pi_), 3000, replace=False)
        d0 = q[pi_[sel]] - q[pj_[sel]]
        d0 -= L * np.round(d0 / L)
        sh = E.DP.shifts_for(L, 12 / xi)
        dsum = np.zeros(len(d0))
        for s0 in range(0, len(sh), 64):
            dd = d0[None] + sh[s0:s0 + 64, None]
            dsum += (K.beta0 * np.exp(-xi ** 2 * np.einsum("spa,spa->sp", dd, dd))).sum(0)
        mv, _, _ = E.mesh_modes(2 * xi * 8.0, L)
        kv = 2 * np.pi / L * mv
        Ak = K.beta0 * (np.pi / xi ** 2) ** 1.5 * np.exp(-np.einsum("ka,ka->k", kv, kv) / (4 * xi ** 2))
        fsum = np.cos(d0 @ kv.T) @ Ak / L ** 3
        rec[f"fourier_correction_poisson_kappa{kap:g}_max_abs_over_G0"] = float(np.max(np.abs(fsum - dsum)) / K.beta0)
        big = dsum > 1e-6 * K.beta0
        rec[f"fourier_correction_poisson_kappa{kap:g}_max_rel_where_sum_gt_1e-6_G0"] = float(
            np.max(np.abs(fsum - dsum)[big] / dsum[big]))
        k0 = np.all(mv == 0, axis=1)
        rec[f"fourier_correction_k0_term_kappa{kap:g}"] = float(Ak[k0][0] / L ** 3)
        rec[f"fourier_correction_k0_expected_kappa{kap:g}"] = float(K.beta0 * (np.pi / xi ** 2) ** 1.5 / L ** 3)
        # (4) complete operators at large cutoffs, mesh-free and PPPM, vs Gamma_ref
        G, GLF = E.load_reference(cfg, kap)
        nref = E.norm2_sym(G)[0]
        rc, kc, _ = E.cutoffs(dict(E.BASE, sR=E.RC_FAR_S, sF=E.SF_FAR), L)
        GF = {}
        for theta in THETAS:
            Kt = E.kernel(kap, xi, theta)
            mesh, _ = E.check_mesh(L, Kt, kc, 96, 9)
            op = E.ThetaOp(q, L, Kt, rc, mesh)
            GS = E.real_dense(op)
            GF[theta] = E.fourier_dense(op.x, L, Kt, kc)
            GM = E.lap_full(E.mode_pair_matrix_fast(mesh, op.x))
            rec[f"full_meshfree_vs_ref_theta{theta:g}_kappa{kap:g}"] = E.norm2_sym(GS + GF[theta] - G)[0] / nref
            rec[f"full_pppm_M96p9_vs_ref_theta{theta:g}_kappa{kap:g}"] = E.norm2_sym(GS + GM - G)[0] / nref
        # Gamma_L^{1/2} - Gamma_L^0 (Fourier) vs the Laplacian of the directly summed periodic Gaussian
        Wd = np.zeros((len(q), len(q)))
        pi2, pj2 = np.triu_indices(len(q), 1)
        dd0 = q[pi2] - q[pj2]
        dd0 -= L * np.round(dd0 / L)
        acc = np.zeros(len(pi2))
        for s0 in range(0, len(sh), 32):
            dd = dd0[None] + sh[s0:s0 + 32, None]
            acc += (K.beta0 * np.exp(-xi ** 2 * np.einsum("spa,spa->sp", dd, dd))).sum(0)
        Wd[pi2, pj2] = acc
        Wd = Wd + Wd.T
        Wfull = np.zeros((len(q), len(q), 3, 3))
        for a in range(3):
            Wfull[:, :, a, a] = Wd
        Gdir = E.lap_full(Wfull)
        dF = GF[0.5] - GF[0.0]
        rec[f"isoGaussian_fourier_vs_direct_laplacian_kappa{kap:g}"] = E.norm2_sym(dF - Gdir)[0] / E.norm2_sym(Gdir)[0]
        C.log(f"checks kappa {kap:g}: radial FT {rec[f'fourier_correction_radialFT_kappa{kap:g}_max_rel']:.1e}, Poisson "
              f"{rec[f'fourier_correction_poisson_kappa{kap:g}_max_abs_over_G0']:.1e}, iso Laplacian "
              f"{rec[f'isoGaussian_fourier_vs_direct_laplacian_kappa{kap:g}']:.1e}, full mesh-free "
              f"{rec[f'full_meshfree_vs_ref_theta0_kappa{kap:g}']:.1e} / {rec[f'full_meshfree_vs_ref_theta0.5_kappa{kap:g}']:.1e}")
    C.write_csv(RESULTS / "checks_split.csv", rows)
    C.write_json(RESULTS / "checks.json", dict(rec, amplitude=E.amplitude_checks(),
                                               split_identity_max_pointwise=max(r["max_rel_pointwise"] for r in rows
                                                                                if r["check"].startswith("split")),
                                               provenance=E.provenance()))


# ============================================================================ pilot
def pilot_points(kappas=KAPPAS):
    pts = []
    for kap in kappas:
        for theta in THETAS:
            for s in S_GRID:
                pts.append(("sR", f"theta{theta:g}", s, dict(PILOT, kappa=kap, theta=theta, sR=s, sF=S_ACC)))
                pts.append(("sF", f"theta{theta:g}", s, dict(PILOT, kappa=kap, theta=theta, sR=S_ACC, sF=s)))
    return pts


def stage_pilot(args):
    rows = A1.stage_sweeps(args, points=pilot_points(), out=RESULTS / "pilot.csv",
                           seeds=tuple(args.seeds), meshfree=True)
    return rows


def floor_of(r):
    return max(r.get("ref_floor_G") or 0.0, r.get("E_G_mesh_only") or 0.0, 1e-15)


def stage_ratios(args):
    rows = E.read_csv(RESULTS / "pilot.csv")
    idx = {}
    for r in rows:
        idx[(r["config"], r["kappa"], r["sweep"], r["value"], r["theta"])] = r
    out = []
    for (cfg, kap, sw, val, th), r0 in sorted(idx.items()):
        if th != 0.0:
            continue
        r1 = idx.get((cfg, kap, sw, val, 0.5))
        if r1 is None:
            continue
        rec = dict(config=cfg, family=r0["family"], seed=r0["seed"], kappa=kap, sweep=sw, value=val,
                   sF_eff=r0["sF_eff"], floor=max(floor_of(r0), floor_of(r1)))
        for m in ("E_G", "E_GX", "E_G_meshfree", "E_GX_slow_max", "E_GX_fast"):
            a, b = r0.get(m), r1.get(m)
            rec[f"{m}_theta0"], rec[f"{m}_theta0.5"] = a, b
            ok = a is not None and b is not None and min(a, b) > 10 * rec["floor"]
            rec[f"ratio_{m}"] = a / b if ok else None
        rec.update(lam_min_theta0=r0["lam_min_h"], lam_min_theta0p5=r1["lam_min_h"], lam_min_ref=r0["lam_min_ref"],
                   n_neg_theta0=r0["n_negative_h"], n_neg_theta0p5=r1["n_negative_h"],
                   t_matvec_theta0=r0["t_matvec_s"], t_matvec_theta0p5=r1["t_matvec_s"],
                   t_setup_theta0=r0["t_setup_s"], t_setup_theta0p5=r1["t_setup_s"])
        out.append(rec)
    C.write_csv(RESULTS / "ratios.csv", out)
    C.log(f"ratios: {len(out)} paired points")


# ============================================================================ equal-accuracy runtime comparison
class Probe:
    """Measured E_Gamma of one candidate on one configuration, cached by parameter set."""

    def __init__(self, cfg, kap, theta):
        self.cfg, self.kap, self.theta = cfg, kap, theta
        self.G, self.GLF = E.load_reference(cfg, kap)
        self.pr, _ = E.probes_for(self.G)
        self.meshes = E.MeshCache()
        self.cache = {}
        self.n_eval = 0

    def err(self, sR, sF, M=80, p=9, meshfree=False):
        key = (round(sR, 6), round(sF, 6), M, p, meshfree)
        if key not in self.cache:
            par = dict(PILOT, kappa=self.kap, theta=self.theta, sR=sR, sF=sF, M=M, p=p)
            r = E.evaluate(self.cfg, self.G, self.GLF, self.pr, par, self.meshes, timing=False, meshfree=meshfree)
            self.n_eval += 1
            self.cache[key] = r
        return self.cache[key]


def min_s(fun, tol, lo=1.5, hi=6.5, step=0.05):
    """Smallest s on a step grid with fun(s) <= tol (bisection; truncation errors decrease with s)."""
    if fun(hi) > tol:
        return None
    a, b = int(round(lo / step)), int(round(hi / step))
    while b - a > 1:
        m = (a + b) // 2
        if fun(m * step) <= tol:
            b = m
        else:
            a = m
    return b * step


def stage_eqacc(args):
    """Per candidate and tolerance on E_Gamma: smallest s_R (real-space error <= tol/3 with mesh-free Fourier sum at
    s_F = 6.5), smallest shell s_F (Fourier error <= tol/3 at s_R = 6.5), then for each p the smallest valid smooth M
    with total PPPM E_Gamma <= tol; the (p, M) with the fastest measured matvec is selected (separately for each
    candidate). Design configuration: strong seed 1, kappa 0.7. Validation: all 9 configurations, serial timing."""
    kap = 0.7
    design = E.config("strong", 1, 0.75)
    rows = []
    for theta in THETAS:
        P = Probe(design, kap, theta)
        for tol in TOLS:
            sR = min_s(lambda s: P.err(s, S_ACC, meshfree=True)["E_G_meshfree"], tol / 3)
            sF = min_s(lambda s: P.err(S_ACC, s, meshfree=True)["E_G_meshfree"], tol / 3)
            best = None
            for p in P_EQ:
                rc, kc, _ = E.cutoffs(dict(PILOT, sR=sR, sF=sF), design["L"])
                mc = int(np.floor(kc * design["L"] / (2 * np.pi) + 1e-12))
                for M in [m for m in A1.M_MOD + (90, 96) if m > 2 * mc]:
                    r = P.err(sR, sF, M, p)
                    if not r.get("mesh_valid") or r["E_G"] > tol:
                        continue
                    t = timed(design, kap, theta, sR, sF, M, p)
                    cand = dict(p=p, M=M, E_G=r["E_G"], t_matvec_s=t)
                    if best is None or t < best["t_matvec_s"]:
                        best = cand
                    break
            rows.append(dict(theta=theta, tol=tol, sR=sR, sF=sF, sF_eff=P.err(S_ACC, sF, meshfree=True)["sF_eff"],
                             **{f"design_{k}": v for k, v in (best or {}).items()}, n_eval=P.n_eval))
            C.log(f"eqacc theta {theta:g} tol {tol:g}: sR {sR}, sF {sF}, best {best}")
    # validation on all configurations, serial timing
    val = []
    for r in rows:
        if r.get("design_M") is None:
            continue
        for fam in E.FAMS:
            for sd in E.SEEDS:
                cfg = E.config(fam, sd, 0.75)
                G, GLF = E.load_reference(cfg, kap)
                pr, _ = E.probes_for(G)
                par = dict(PILOT, kappa=kap, theta=r["theta"], sR=r["sR"], sF=r["sF"], M=r["design_M"], p=r["design_p"])
                e = E.evaluate(cfg, G, GLF, pr, par, E.MeshCache(), timing=True)
                val.append(dict(tol=r["tol"], **e))
    C.write_csv(RESULTS / "eqacc_design.csv", rows)
    C.write_csv(RESULTS / "eqacc_validation.csv", val)
    summ = []
    for r in rows:
        vs = [v for v in val if v["tol"] == r["tol"] and v["theta"] == r["theta"]]
        if not vs:
            continue
        summ.append(dict(theta=r["theta"], tol=r["tol"], sR=r["sR"], sF=r["sF"], M=r["design_M"], p=r["design_p"],
                         E_G_max=max(v["E_G"] for v in vs), E_GX_max=max(v["E_GX"] for v in vs),
                         n_fail=int(sum(v["E_G"] > r["tol"] for v in vs)),
                         lam_min_rel_dev_max=max(abs(v["lam_min_h"] / v["lam_min_ref"] - 1) for v in vs),
                         n_images_median=float(np.median([v["n_images"] for v in vs])),
                         t_setup_median_s=float(np.median([v["t_setup_s"] for v in vs])),
                         t_matvec_median_s=float(np.median([v["t_matvec_s"] for v in vs]))))
    C.write_csv(RESULTS / "eqacc_summary.csv", summ)


def stage_eqtime(args, rounds=7, reps=7):
    """Paired, interleaved serial timing of the two equal-accuracy designs (eqacc_summary.csv) on all 9
    configurations: rounds alternate theta = 0 / 1/2 so that machine drift affects both alike."""
    summ = E.read_csv(RESULTS / "eqacc_summary.csv")
    kap = 0.7
    out = []
    for tol in TOLS:
        des = {r["theta"]: r for r in summ if r["tol"] == tol}
        if len(des) < 2:
            continue
        for fam in E.FAMS:
            for sd in E.SEEDS:
                cfg = E.config(fam, sd, 0.75)
                ops = {}
                for th, r in des.items():
                    K = E.kernel(kap, 0.85, th)
                    rc, kc, _ = E.cutoffs(dict(PILOT, sR=r["sR"], sF=r["sF"]), cfg["L"])
                    t0 = time.perf_counter()
                    mesh, _ = E.check_mesh(cfg["L"], K, kc, r["M"], r["p"])
                    t_mesh = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    op = E.ThetaOp(cfg["q"], cfg["L"], K, rc, mesh)
                    ops[th] = (op, t_mesh, time.perf_counter() - t0)
                X = np.random.default_rng(11).standard_normal((cfg["N"], 3))
                ts = {th: [] for th in ops}
                for _ in range(rounds):
                    for th, (op, _, _) in ops.items():
                        ts[th] += list(C.median_time(lambda: op.apply(X), reps, warm=1))
                rec = dict(tol=tol, config=cfg["name"])
                for th, (op, tm, tsu) in ops.items():
                    st = C.tstats(ts[th])
                    rec.update({f"t_matvec_theta{th:g}_s": st["median"], f"t_matvec_iqr_theta{th:g}": st["iqr_rel"],
                                f"t_setup_theta{th:g}_s": tsu, f"t_mesh_theta{th:g}_s": tm,
                                f"n_images_theta{th:g}": op.n_images, f"M_theta{th:g}": op.mesh.M,
                                f"p_theta{th:g}": op.mesh.p})
                rec["matvec_ratio_theta0.5_over_0"] = rec["t_matvec_theta0.5_s"] / rec["t_matvec_theta0_s"]
                out.append(rec)
        sel = [r["matvec_ratio_theta0.5_over_0"] for r in out if r["tol"] == tol]
        C.log(f"eqtime tol {tol:g}: matvec ratio theta 1/2 / theta 0 median {np.median(sel):.3f} "
              f"[{min(sel):.3f}, {max(sel):.3f}]")
    C.write_csv(RESULTS / "eqacc_paired_timing.csv", out)


def timed(cfg, kap, theta, sR, sF, M, p, reps=9):
    K = E.kernel(kap, 0.85, theta)
    rc, kc, _ = E.cutoffs(dict(PILOT, sR=sR, sF=sF), cfg["L"])
    mesh, info = E.check_mesh(cfg["L"], K, kc, M, p)
    op = E.ThetaOp(cfg["q"], cfg["L"], K, rc, mesh)
    X = np.random.default_rng(1).standard_normal((cfg["N"], 3))
    return C.tstats(C.median_time(lambda: op.apply(X), reps, warm=2))["median"]


# ============================================================================ plots / tables
def stage_plots(args):
    """Paper-style figures (../paper_figures.py) from the saved CSV files."""
    import paper_figures
    paper_figures.exp2(RESULTS, FIGURES, E.read_csv)
    C.log("plots written")


def stage_tables(args):
    rows = E.read_csv(RESULTS / "pilot.csv")
    rat = E.read_csv(RESULTS / "ratios.csv")
    out = ["# Experiment 2 tables (generated by run_candidate2_pilot.py tables)", ""]
    for sw in ("sR", "sF"):
        out += [f"## {sw} = 3, 4, 5 (companion cutoff 6.5, M = 80, p = 9)", "",
                "| kappa | config | s | E_Γ θ=0 | E_Γ θ=½ | ratio | E_ΓX θ=0 | E_ΓX θ=½ | ratio | E_Γ mesh-free θ=0 / θ=½ | "
                "mesh-only θ=0 / θ=½ | λmin θ=0 / θ=½ / ref | t_mv θ=0 / θ=½ [ms] |", "|" + "---|" * 13]
        for kap in KAPPAS:
            for s in HEADLINE:
                for r in [q for q in rat if q["kappa"] == kap and q["sweep"] == sw and q["value"] == s]:
                    r0 = next(q for q in rows if q["config"] == r["config"] and q["kappa"] == kap and q["sweep"] == sw
                              and q["value"] == s and q["theta"] == 0.0)
                    r1 = next(q for q in rows if q["config"] == r["config"] and q["kappa"] == kap and q["sweep"] == sw
                              and q["value"] == s and q["theta"] == 0.5)
                    rg = lambda m: "–" if r.get(f"ratio_{m}") is None else f"{r[f'ratio_{m}']:.2f}"
                    out.append(f"| {kap:g} | {r['config']} | {s:g}{f' ({r0['sF_eff']:.3f})' if sw == 'sF' else ''} | "
                               f"{r0['E_G']:.2e} | {r1['E_G']:.2e} | {rg('E_G')} | {r0['E_GX']:.2e} | {r1['E_GX']:.2e} | "
                               f"{rg('E_GX')} | {r0['E_G_meshfree']:.1e} / {r1['E_G_meshfree']:.1e} | "
                               f"{r0['E_G_mesh_only']:.1e} / {r1['E_G_mesh_only']:.1e} | {r0['lam_min_h']:.5f} / "
                               f"{r1['lam_min_h']:.5f} / {r0['lam_min_ref']:.5f} | {1e3 * r0['t_matvec_s']:.1f} / "
                               f"{1e3 * r1['t_matvec_s']:.1f} |")
        out.append("")
    out += ["## Improvement ratio E_Γ(θ=0)/E_Γ(θ=½) above the floor (all pilot points)", "",
            "| kappa | sweep | n points | median | min | max |", "|---|---|---|---|---|---|"]
    for kap in KAPPAS:
        for sw in ("sR", "sF"):
            v = [r["ratio_E_G"] for r in rat if r["kappa"] == kap and r["sweep"] == sw and r.get("ratio_E_G") is not None]
            if v:
                out.append(f"| {kap:g} | {sw} | {len(v)} | {np.median(v):.2f} | {min(v):.2f} | {max(v):.2f} |")
    out += ["", "## PSD of the complete Γ_h on Range(Π) (all pilot points)", "",
            "| theta | kappa | n | min λmin(Γ_h) | n with negative eig. | max |λmin_h/λmin_ref − 1| |", "|---|---|---|---|---|---|"]
    for th in THETAS:
        for kap in KAPPAS:
            sel = [r for r in rows if r["theta"] == th and r["kappa"] == kap and r.get("lam_min_h") is not None]
            out.append(f"| {th:g} | {kap:g} | {len(sel)} | {min(r['lam_min_h'] for r in sel):.5f} | "
                       f"{sum(r['n_negative_h'] > 0 for r in sel)} | "
                       f"{max(abs(r['lam_min_h'] / r['lam_min_ref'] - 1) for r in sel):.1e} |")
    f = RESULTS / "eqacc_summary.csv"
    if f.exists():
        out += ["", "## Equal-accuracy runtime (kappa = 0.7, design strong_s1, validated on 9 configurations, serial)", "",
                "| target E_Γ | theta | s_R | s_F | M | p | max E_Γ (9 cfg) | fails | images | setup [ms] | matvec [ms] |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(E.read_csv(f), key=lambda r: (-r["tol"], r["theta"])):
            out.append(f"| {r['tol']:g} | {r['theta']:g} | {r['sR']:.2f} | {r['sF']:.2f} | {r['M']} | {r['p']} | "
                       f"{r['E_G_max']:.2e} | {r['n_fail']} | {r['n_images_median']:.0f} | "
                       f"{1e3 * r['t_setup_median_s']:.1f} | {1e3 * r['t_matvec_median_s']:.1f} |")
    (RESULTS / "tables.md").write_text("\n".join(out) + "\n")
    C.log("tables written")


# ============================================================================ smoke
def stage_smoke(args):
    out = RESULTS / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    kap = 2.0
    cfg = E.config("strong", 1, 0.75)
    if not E.ref_path(cfg, kap).exists():
        E.compute_reference(cfg, kap, say=C.log)
    G, GLF = E.load_reference(cfg, kap)
    pr, _ = E.probes_for(G)
    K = E.kernel(kap, 0.85, 0.5)
    rq = np.linspace(0, 12 / 0.85, 20001)
    k = np.linspace(0, 12, 25)
    num = 4 * np.pi * integrate.simpson(rq ** 2 * K.beta0 * np.exp(-(0.85 * rq) ** 2) * np.sinc(np.outer(k, rq) / np.pi),
                                        x=rq, axis=1)
    ft = float(np.max(np.abs(K.AB(k)[0] - E.kernel(kap, 0.85, 0.0).AB(k)[0] - num)) / np.max(num))
    meshes = E.MeshCache()
    rows = []
    for th in THETAS:
        for sw, s in (("sR", 3.0), ("sR", 5.0), ("sF", 3.0), ("sF", 5.0)):
            par = dict(PILOT, kappa=kap, theta=th, **({"sR": s, "sF": S_ACC} if sw == "sR" else {"sR": S_ACC, "sF": s}))
            rows.append(dict(sweep=sw, value=s, **E.evaluate(cfg, G, GLF, pr, par, meshes, meshfree=True)))
    C.write_csv(out / "smoke_pilot.csv", rows)
    for r in rows:
        C.log(f"smoke theta {r['theta']:g} {r['sweep']}={r['value']:g}: E_G {r['E_G']:.2e} (mesh-free {r['E_G_meshfree']:.2e}),"
              f" lam_min {r['lam_min_h']:.5f} (ref {r['lam_min_ref']:.5f}), mf/dense {r['mf_vs_dense']:.1e}")
    C.write_json(out / "smoke_checks.json", dict(fourier_correction_radial_FT_max_rel=ft, provenance=E.provenance()))
    C.log(f"smoke: Fourier correction vs radial FT {ft:.1e}")
    assert ft < 1e-10
    assert all(r["mf_vs_dense"] < 1e-12 for r in rows)


STAGES = dict(smoke=stage_smoke, checks=stage_checks, pilot=stage_pilot, ratios=stage_ratios, eqacc=stage_eqacc,
              eqtime=stage_eqtime, plots=stage_plots, tables=stage_tables)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=list(STAGES) + ["all"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    E.log_cmd(RESULTS)
    stages = ["checks", "pilot", "ratios", "eqacc", "eqtime", "plots", "tables"] if "all" in args.stages else args.stages
    for st in stages:
        t0 = time.perf_counter()
        C.log(f"=== stage {st}")
        STAGES[st](args)
        C.log(f"=== stage {st} done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
