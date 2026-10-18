#!/usr/bin/env python3
"""
Method A at N = 512 (GitHub issue #2). Stage 1: joint cost-aware search over (xi, rc, kc, M, p) with the
Hurwitz-zeta mesh bound and a per-p cost model, then MEASURED selection. Stage 2: dense Gamma_h vs matrix-free PPPM
for the very same discrete operator. Every output goes to results_psearch/ (results/, results_joint/ are only read).

Fixed: common.py constants (kernel A, gamma 0.5, kappa 0.7, r_ref 1.3, rho 0.75, full periodic lattice sum, k = 0
retained), configurations configs/*_N512_s1.npz, the Ewald split at s = xi^2, the existing PPPM (Mesh) and the
independent dense full-periodic reference raw/<config>_Gref.npy (v2.LatticeFriction, rel_tol 1e-13), tau = 2e-7.

Stage 1
  calibrate  per configuration, isolated N = 512 timings for both action implementations (cached D-H action,
             CachedGammaH.apply; existing pair-difference action, GammaH.matvec on the same images and mesh):
             real space vs N_img, spread + gather for every p = 4..11 at three M, FFT + influence vs M, and full
             actions at four (rc, M) per p -> PCostModel per implementation (results_psearch/cost_model_p.json)
  select     per configuration and p in --p (default 5..10): approach C of joint_optimizer (joint budget, HZ bound,
             exact reciprocal shells, 2,3,5-smooth M <= M_CAP_P, smallest admissible rc by bisection), xi on the 0.025
             grid 0.40..1.60 and a 0.005 refinement around each p's best xi; the eps_S curves of one xi are shared by
             all p; (kc, M) alternatives with a different M at each p's xi_opt; ranking model = the implementation
             with the smaller calibrated real-space cost (both predictions are recorded)
  bench      every candidate (per p: optimum, two alternatives with another M, xi_opt +- 0.025, +- 0.05; the global
             model top 8): Mesh build (parameter-only) and setups of both implementations (3 interleaved builds),
             then interleaved screening of both actions (R_SCREEN rounds, random order per round, same X for every
             candidate in a round)
  errors     E_op = ||Gamma_h - Gamma_ref||_2/||Gamma_ref||_2 of every candidate (GammaH.dense, the project's
             independent assembly), structure, probe actions
  confirm    the 6 fastest passing candidates and the fastest passing candidate of every p, both actions,
             R_CONF interleaved rounds; the measured optimum = fastest passing (candidate, implementation)
Stage 2 (dense)
  For the measured optimum of every configuration: the same positions, the same 10 deterministic X; dense Gamma_h
  from positions by (a) GammaH.dense (project routine: np.add.at real blocks + Mesh.pair_matrix + lap_full),
  (b) the same assembly vectorised (bincount real blocks, real-arithmetic mode sum over +-k pairs), (c) basis
  vectors through the matrix-free action; interleaved build timings with both PPPM setups; interleaved GEMV
  (G @ x, one BLAS thread) vs both matrix-free actions; identity checks; one-time Mesh and Gamma_ref creation timed
  separately; end-to-end T_build + q T_GEMV vs T_setup + q T_action.

    python3 run_psearch_dense512.py --stage calibrate select bench errors confirm dense
All timings: one process, OMP/OPENBLAS/MKL threads = 1 (verified with threadpoolctl), nothing else running.
"""
import argparse
import gc
import time

import common as C                       # first: sets OMP/OPENBLAS/MKL_NUM_THREADS = 1 before numpy loads OpenBLAS
import numpy as np
from scipy.optimize import nnls
from threadpoolctl import threadpool_info, threadpool_limits

from clustered_configurations import load
from cached_operator import CachedGammaH, mesh_for, pair_difference_action, short_range_weights
from neighbor_images import kdtree_images
from parameter_optimizer import RealSpaceTail, A_MAX, R_PAD, XI_GRID, shell_kc
from joint_optimizer import evaluate_xi, finalize, PCostModel, TAU, G_RC
from error_validation import (reference_cached, reference_summary, probes, validate_parameter_set, REF_REL_TOL,
                              PAIRS6)
from test_lanczos_fdt import GammaH
from test_parameter_selection import Mesh, lap_full

RESULTS_PSEARCH = C.HERE / "results_psearch"
OLD_RESULTS = C.HERE / "results"
CONFIGS = ("strong_N512_s1", "uniform_N512_s1", "moderate_N512_s1")
P_SET = (5, 6, 7, 8, 9, 10)
P_CAL = tuple(range(4, 12))                  # calibrated and alias-validated orders (4 and 11: boundary neighbours)
IMPLS = ("cached", "pairdiff")
M_CAP_P = 128
TOPK = 80
N_ALT_M = 2
XI_NEIGH = (-0.05, -0.025, 0.025, 0.05)
N_GLOBAL = 8
R_SCREEN, R_CONF, R_BUILD, R_GEMV = 40, 150, 30, 200
N_CONF = 6
CAL_XI = 0.75
CAL_FULL = ((4.5, 32), (6.0, 45), (5.0, 60), (6.5, 36))
CAL_FFT_M = (20, 24, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 90, 96, 100, 108, 120, 128)
SEED = 20261018


# ============================================================================ helpers
_LIMITS = threadpool_limits(limits=1)    # every BLAS / OpenMP pool of this process: one thread


def blas_info():
    return threadpool_info()


def assert_one_thread():
    """Timing stages refuse to run unless every loaded BLAS/OpenMP pool has exactly one thread."""
    info = threadpool_info()
    bad = [(i.get("internal_api"), i.get("num_threads")) for i in info if i.get("num_threads") != 1]
    if bad:
        raise SystemExit(f"thread pools not limited to 1: {bad}")
    return info


def x_inputs(N, k=10):
    """The deterministic particle-vector inputs (same for every candidate, both actions and the dense GEMV)."""
    rng = np.random.default_rng([SEED, 512, 7])
    return [rng.standard_normal((N, 3)) for _ in range(k)]


def interleaved(fns, rounds, inputs, warm=3, seed=0):
    """Run every fn once per round in a fresh random order, all on inputs[r % len(inputs)]; times per fn."""
    rng = np.random.default_rng([SEED, 3, seed])
    keys = list(fns)
    for w in range(warm):
        for k in keys:
            fns[k](inputs[w % len(inputs)])
    t = {k: np.empty(rounds) for k in keys}
    gc_was = gc.isenabled()
    gc.disable()
    try:
        for r in range(rounds):
            X = inputs[r % len(inputs)]
            for o in rng.permutation(len(keys)):
                k = keys[o]
                t0 = time.perf_counter()
                fns[k](X)
                t[k][r] = time.perf_counter() - t0
    finally:
        if gc_was:
            gc.enable()
    return t


def st(t):
    s = C.tstats(t)
    return dict(median_ms=1e3 * s["median"], q1_ms=1e3 * s["q1"], q3_ms=1e3 * s["q3"], min_ms=1e3 * s["min"],
                max_ms=1e3 * s["max"], n=s["n"], iqr_rel=s["iqr_rel"])


def pairdiff_setup(x, L, K, rc, mesh):
    """Setup of the existing pair-difference operator (GammaH: pairs, weights, mesh stencil, mesh degree) on the
    periodic KD-tree image list (identical images to CachedGammaH)."""
    t0 = time.perf_counter()
    op = GammaH.__new__(GammaH)
    x = np.mod(x, L)
    il = kdtree_images(x, L, rc)
    op.x, op.L, op.N, op.rc, op.kc = x, float(L), len(x), float(rc), mesh.kc
    op.i, op.j, op.d = il.i, il.j, il.d
    op.w = short_range_weights(K, op.d)
    op.mesh = mesh
    op.idx, op.wt = mesh.flat(x)
    op.D = mesh.degree(op.idx, op.wt)
    op.n_actions, op.t_actions = 0, 0.0
    op.t_setup = time.perf_counter() - t0
    return op


def action_fns(op):
    """Both production actions on the same images and mesh, as functions of X (N, 3)."""
    pd = op.existing_gammah()
    return {"cached": op.apply, "pairdiff": lambda X: pd.matvec(X.ravel())}


def cand_key(r):
    return f"p{r['p']}_xi{r['xi']:.4f}_kc{r['kc']:.6f}_M{r['M']}_rc{r['rc']:.6f}"


def path(kind, name, tag=None):
    return RESULTS_PSEARCH / (f"{kind}_{name}" + (f"_{tag}" if tag else "") + ".json")


def cost_models(name):
    d = C.read_json(RESULTS_PSEARCH / "cost_model_p.json")["configs"][name]
    return {impl: PCostModel.from_dict(d["models"][impl]) for impl in IMPLS}, d["rank_impl"]


def refs():
    return C.read_json(OLD_RESULTS / "references.json")["configs"]


# ============================================================================ calibrate
def fit_rel(f, y, intercept=True):
    """y = slope f (+ intercept), coefficients >= 0, least squares in relative error."""
    A = np.stack([f, np.ones_like(f)], 1) if intercept else f[:, None]
    coef, _ = nnls(A / y[:, None], np.ones_like(y))
    pred = A @ coef
    return coef, float(np.max(np.abs(pred - y) / y))


def calibrate_config(name):
    x, L = load(name)
    x = np.mod(x, L)
    N = len(x)
    K = C.kernel(CAL_XI)
    Xs = x_inputs(N)
    rows = []
    small = mesh_for(L, K, 2 * np.pi * 2 / L, 16, 4)
    for rc in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        op = CachedGammaH(x, L, K, rc, small)
        t = interleaved({"cached": op.real_H,
                         "pairdiff": lambda X, op=op: pair_difference_action(op.i, op.j, op.d, op.w, X, N)}, 30, Xs)
        for impl in IMPLS:
            rows.append(dict(config=name, kind="real", impl=impl, rc=rc, n_img=op.n_images, p=4, M=16,
                             t=float(np.median(t[impl])), t_iqr_rel=C.tstats(t[impl])["iqr_rel"]))
    for p in P_CAL:
        for M in (30, 45, 64):
            op = CachedGammaH(x, L, K, 1.5, mesh_for(L, K, 2 * np.pi * 4 / L, M, p))
            ph = op.fft_influence(op.spread(Xs[0]))
            t = interleaved({"spread": op.spread, "gather": lambda X, op=op, ph=ph: op.gather(ph)}, 30, Xs)
            ts, tg = float(np.median(t["spread"])), float(np.median(t["gather"]))
            rows.append(dict(config=name, kind="spread_gather", impl="both", p=p, M=M, n_img=op.n_images, rc=1.5,
                             t=ts + tg, t_spread=ts, t_gather=tg))
    for M in CAL_FFT_M:
        op = CachedGammaH(x, L, K, 1.5, mesh_for(L, K, 2 * np.pi * 4 / L, M, 4))
        g = op.spread(Xs[0])
        t = interleaved({"fft": lambda X, op=op, g=g: op.fft_influence(g)}, 30, Xs)
        rows.append(dict(config=name, kind="fft", impl="both", M=M, p=4, rc=1.5, n_img=op.n_images,
                         t=float(np.median(t["fft"]))))
    for p in P_CAL:
        for rc, M in CAL_FULL:
            op = CachedGammaH(x, L, K, rc, mesh_for(L, K, 0.5 * np.pi * M / L, M, p))
            t = interleaved(action_fns(op), 30, Xs)
            for impl in IMPLS:
                rows.append(dict(config=name, kind="full", impl=impl, p=p, M=M, rc=rc, n_img=op.n_images,
                                 t=float(np.median(t[impl])), t_iqr_rel=C.tstats(t[impl])["iqr_rel"]))
        C.log(f"calibrate {name}: p={p} full actions done")
    # ---- fits
    fft = [r for r in rows if r["kind"] == "fft"]
    fM = np.array([r["M"] ** 3 * np.log(r["M"] ** 3) for r in fft], float)
    (C_F, b_F), e_F = fit_rel(fM, np.array([r["t"] for r in fft]))
    per_impl = {}
    for impl in IMPLS:
        real = [r for r in rows if r["kind"] == "real" and r["impl"] == impl]
        (C_R, b_R), e_R = fit_rel(np.array([r["n_img"] for r in real], float), np.array([r["t"] for r in real]))
        per_p, info = {}, {}
        for p in P_CAL:
            sg = [r for r in rows if r["kind"] == "spread_gather" and r["p"] == p]
            (G_p, S_p), e_P = fit_rel(np.array([r["M"] ** 3 for r in sg], float), np.array([r["t"] for r in sg]))
            full = [r for r in rows if r["kind"] == "full" and r["impl"] == impl and r["p"] == p]
            comp = lambda r: (C_R * r["n_img"] + b_R + S_p + G_p * r["M"] ** 3  # noqa: E731
                              + C_F * r["M"] ** 3 * np.log(r["M"] ** 3) + b_F)
            glue = float(np.median([r["t"] - comp(r) for r in full]))
            per_p[p] = dict(S=float(S_p), G=float(G_p), C_0=float(b_R + b_F + glue))
            info[str(p)] = dict(spread_gather_fit_max_rel_err=e_P, glue=glue, C_P_equiv=float(S_p / (N * p ** 3)),
                           full_rel_err=[float((comp(r) + glue) / r["t"] - 1) for r in full])
        cm = PCostModel(C_R, C_F, per_p, N=N, intercepts=dict(real=float(b_R), fft=float(b_F)),
                        max_rel_fit_err=dict(real=e_R, fft=e_F), per_p_info=info)
        per_impl[impl] = cm.as_dict()
        for r in rows:
            if r["kind"] == "full" and r["impl"] == impl:
                r["t_pred"] = cm.predict(r["n_img"], N, r["p"], r["M"])
    rank = min(IMPLS, key=lambda i: per_impl[i]["C_R"])
    return rows, dict(models=per_impl, rank_impl=rank, xi=CAL_XI)


def stage_calibrate(args):
    assert_one_thread()
    out = C.read_json(RESULTS_PSEARCH / "cost_model_p.json", {"configs": {}}) if args.resume else {"configs": {}}
    allrows = []
    for name in args.configs:
        t0 = time.perf_counter()
        rows, d = calibrate_config(name)
        out["configs"][name] = d
        allrows += rows
        m = d["models"]
        C.log(f"cost model {name}: rank impl {d['rank_impl']}; C_R cached {m['cached']['C_R']:.3e} pairdiff "
              f"{m['pairdiff']['C_R']:.3e}; C_F {m['cached']['C_F']:.3e}; S_p [ms] "
              + " ".join(f"{p}:{1e3 * m['cached']['per_p'][str(p)]['S']:.2f}" for p in P_CAL)
              + "; full rel err cached " + " ".join(f"{e:+.2f}" for p in P_CAL
                                                    for e in m['cached']['per_p_info'][str(p)]['full_rel_err'])
              + f" ({time.perf_counter() - t0:.0f} s)")
        out.update(provenance=C.provenance(blas=blas_info()), note="seconds; one action, 1 thread, N = 512")
        C.write_json(RESULTS_PSEARCH / "cost_model_p.json", out)
    if allrows:
        C.write_csv(RESULTS_PSEARCH / "calibration_timings_p.csv", allrows)


# ============================================================================ select
def with_preds(rec, models, N):
    rec = dict(rec)
    for impl, cm in models.items():
        parts = cm.parts(rec["n_img_mean"], N, rec["p"], rec["M"])
        rec[f"T_pred_{impl}_ms"] = 1e3 * float(sum(parts.values()))
        rec.update({f"T_pred_{impl}_{k}_ms": 1e3 * float(v) for k, v in parts.items()})
    rec["eps_S_abs"] = rec["eps_S_abs_per_seed"][0]
    rec["eps_cert_rel_no_quad"] = (rec["eps_S_abs"] + rec["eps_F_abs"] + rec["eps_mesh_abs"]) / rec["lam_lb_per_seed"][0]
    return rec


def select_config(name, ps, models, rank_impl, R):
    x, L = load(name)
    N = len(x)
    seed = dict(name=name, rst=RealSpaceTail(x, L, (A_MAX + R_PAD) / XI_GRID.min()), lam_lb=R[name]["lam_lb"])
    cm = models[rank_impl]
    res = {}

    def run_xi(xi, plist):
        curves = None
        for p in plist:
            if (p, xi) in res:
                continue
            out, _, curves = evaluate_xi(xi, [seed], L, N, cm, p=p, approaches=("C",), curves=curves,
                                         variants=("hz",), M_cap=M_CAP_P, topk=1)
            res[(p, xi)] = None if out["C"] is None else with_preds(out["C"]["best"], models, N)

    t0 = time.perf_counter()
    for xi in XI_GRID:
        run_xi(round(float(xi), 4), ps)
    C.log(f"select {name}: coarse grid done ({time.perf_counter() - t0:.0f} s)")
    want = {}
    for p in ps:
        coarse = {xi: r for (q, xi), r in res.items() if q == p and r is not None}
        xb = min(coarse, key=lambda k: coarse[k]["T_pred"])
        for xi in np.round(np.arange(xb - 0.025, xb + 0.02501, 0.005), 4):
            if XI_GRID.min() <= xi <= XI_GRID.max():
                want.setdefault(round(float(xi), 4), []).append(p)
    for xi in sorted(want):
        run_xi(xi, want[xi])
    per_p = {}
    for p in ps:
        allx = {xi: r for (q, xi), r in res.items() if q == p and r is not None}
        xo = min(allx, key=lambda k: allx[k]["T_pred"])
        out, tab, curves = evaluate_xi(xo, [seed], L, N, cm, p=p, approaches=("C",), variants=("hz",),
                                       M_cap=M_CAP_P, topk=TOPK)
        opt = allx[xo]
        alts, Ms = [], {opt["M"]}
        for t in out["C"]["top"]:
            if len(alts) >= N_ALT_M:
                break
            if t["M"] in Ms:
                continue
            Ms.add(t["M"])
            a = finalize(xo, tab, curves, [seed], t["i"], t["j"], np.array(t["B"]), np.array(t["g"]), "hz", cm, N,
                         p, TAU)
            alts.append(with_preds(a, models, N))
        per_p[str(p)] = dict(opt=opt, alternatives=alts, n_candidates_at_opt=out["C"]["n_candidates"],
                             per_xi={f"{k:.4f}": allx[k] for k in sorted(allx)})
        C.log(f"select {name} p={p}: xi {opt['xi']:.3f} kc {opt['kc']:.3f} M {opt['M']} rc {opt['rc']:.3f} n_img "
              f"{opt['n_img_mean']:.0f} cert {opt['cert_rel_max']:.3e} T_pred {rank_impl} "
              f"{opt[f'T_pred_{rank_impl}_ms']:.2f} ms; alt M {[a['M'] for a in alts]}")
    best = min(r[f"T_pred_{rank_impl}_ms"] for g in per_p.values() for r in g["per_xi"].values())
    fft_cap = 1e3 * float(cm.C_F * M_CAP_P ** 3 * np.log(M_CAP_P ** 3))
    return dict(config=name, N=N, L=L, lam_lb=R[name]["lam_lb"], norm2=R[name]["norm2"], tau=TAU, ps=list(ps),
                rank_impl=rank_impl, M_cap=M_CAP_P, G_RC=G_RC, xi_grid=[float(XI_GRID.min()), float(XI_GRID.max())],
                fft_term_at_M_cap_ms=fft_cap, best_T_pred_ms=best, per_p=per_p,
                wall_s=time.perf_counter() - t0)


def stage_select(args):
    R = refs()
    for name in args.configs:
        models, rank = cost_models(name)
        f = path("selection", name)
        old = C.read_json(f) if (args.resume and f.exists()) else None
        ps = [p for p in args.p if old is None or str(p) not in old["per_p"]]
        if not ps:
            continue
        d = select_config(name, ps, models, rank, R)
        if old is not None:
            old["per_p"].update(d["per_p"])
            old["ps"] = sorted(set(old["ps"]) | set(d["ps"]))
            old["best_T_pred_ms"] = min(old["best_T_pred_ms"], d["best_T_pred_ms"])
            d = old
        d["provenance"] = C.provenance()
        C.write_json(f, d)
        C.log(f"select {name}: best predicted {d['best_T_pred_ms']:.2f} ms; FFT term alone at M_cap = {M_CAP_P}: "
              f"{d['fft_term_at_M_cap_ms']:.1f} ms ({d['wall_s']:.0f} s)")


# ============================================================================ bench (screening)
def candidates(sel, ps):
    out = []
    allrec = []
    for p in ps:
        g = sel["per_p"][str(p)]
        out.append(("opt", g["opt"]))
        out += [(f"alt_M{t + 1}", a) for t, a in enumerate(g["alternatives"])]
        per = g["per_xi"]
        grid = np.array([float(k) for k in per])
        xo = g["opt"]["xi"]
        for o in XI_NEIGH:
            k = grid[np.argmin(np.abs(grid - (xo + o)))]
            if abs(k - xo) > 1e-9 and abs(k - (xo + o)) < 0.003:
                out.append((f"xi{o:+.3f}", per[f"{k:.4f}"]))
        allrec += list(per.values())
    rk = sel["rank_impl"]
    out += [("global_top", r) for r in sorted(allrec, key=lambda r: r[f"T_pred_{rk}_ms"])[:N_GLOBAL]]
    seen, uniq = {}, []
    for role, r in out:
        k = cand_key(r)
        if k in seen:
            seen[k]["roles"].append(role)
            continue
        c = dict(r, roles=[role], key=k)
        seen[k] = c
        uniq.append(c)
    return uniq


def build_candidate(x, L, c):
    K = C.kernel(c["xi"])
    kc = shell_kc(c["kc"], L)
    t0 = time.perf_counter()
    mesh = mesh_for(L, K, kc, c["M"], c["p"])
    t_mesh = time.perf_counter() - t0
    assert mesh.mc == int(np.floor(kc * L / (2 * np.pi) + 1e-12)) and \
        np.isclose(np.max(np.linalg.norm(mesh.kv, axis=1)), kc, rtol=1e-12), "outer shell lost"
    return K, kc, mesh, t_mesh


def stage_bench(args):
    assert_one_thread()
    for name in args.configs:
        sel = C.read_json(path("selection", name))
        ps = [p for p in sel["ps"] if p in args.p] if args.only_p else sel["ps"]
        cands = candidates(sel, ps)
        x, L = load(name)
        x = np.mod(x, L)
        N = len(x)
        Xs = x_inputs(N)
        C.log(f"bench {name}: {len(cands)} candidates over p {ps}")
        ops, setup = {}, {c["key"]: {"cached": [], "pairdiff": []} for c in cands}
        meshes = {}
        for c in cands:
            K, kc, mesh, t_mesh = build_candidate(x, L, c)
            meshes[c["key"]] = (K, mesh)
            c.update(kc_shell=kc, t_mesh_build_ms=1e3 * t_mesh, n_modes=mesh.n_modes, n_excluded=mesh.n_excluded,
                     eta_actual=mesh.eta_actual)
        rng = np.random.default_rng([SEED, 11, N])
        for r in range(3):
            for o in rng.permutation(len(cands)):
                c = cands[o]
                K, mesh = meshes[c["key"]]
                op = CachedGammaH(x, L, K, c["rc"], mesh)
                setup[c["key"]]["cached"].append(op.t_setup)
                ops[c["key"]] = op
                pd = pairdiff_setup(x, L, K, c["rc"], mesh)
                setup[c["key"]]["pairdiff"].append(pd.t_setup)
                del pd
        fns = {}
        for c in cands:
            op = ops[c["key"]]
            f = action_fns(op)
            dev = max(float(np.linalg.norm(f["cached"](X).ravel() - f["pairdiff"](X)) /
                            np.linalg.norm(f["cached"](X))) for X in Xs[:3])
            c.update(n_img=op.n_images, cached_vs_pairdiff_rel=dev,
                     **{f"T_setup_{i}_ms": 1e3 * float(np.median(setup[c["key"]][i])) for i in IMPLS})
            for i in IMPLS:
                fns[(c["key"], i)] = f[i]
        t0 = time.perf_counter()
        t = interleaved(fns, R_SCREEN, Xs, seed=N)
        C.log(f"bench {name}: screening {len(fns)} actions x {R_SCREEN} rounds ({time.perf_counter() - t0:.0f} s)")
        raw = {}
        for c in cands:
            for i in IMPLS:
                s = st(t[(c["key"], i)])
                c.update({f"T_action_{i}_screen_{k}": v for k, v in s.items()})
                raw[f"{c['key']}|{i}"] = (1e3 * t[(c["key"], i)]).tolist()
            c["T_action_screen_best_ms"] = min(c[f"T_action_{i}_screen_median_ms"] for i in IMPLS)
            c["T_action_screen_best_impl"] = min(IMPLS, key=lambda i: c[f"T_action_{i}_screen_median_ms"])
        rk = sel["rank_impl"]
        for c in sorted(cands, key=lambda c: c["T_action_screen_best_ms"]):
            C.log(f"  {c['key']:52s} {','.join(c['roles']):24s} n_img {c['n_img']:7d}: cached "
                  f"{c['T_action_cached_screen_median_ms']:6.2f} pairdiff {c['T_action_pairdiff_screen_median_ms']:6.2f} "
                  f"pred {c['T_pred_' + rk + '_ms']:6.2f} setup {c['T_setup_cached_ms']:.0f}/"
                  f"{c['T_setup_pairdiff_ms']:.0f} ms")
        C.write_json(path("bench", name, args.tag), dict(config=name, ps=ps, candidates=cands, rounds=R_SCREEN,
                                                         provenance=C.provenance(blas=blas_info())))
        C.write_json(path("raw_screen", name, args.tag), dict(unit="ms", times=raw))
        del ops, fns, meshes
        gc.collect()


# ============================================================================ errors
def stage_errors(args):
    for name in args.configs:
        bench = C.read_json(path("bench", name, args.tag))["candidates"]
        f = path("errors", name)
        out = C.read_json(f, {"rows": {}}) if f.exists() else {"rows": {}}
        x, L = load(name)
        x = np.mod(x, L)
        N = len(x)
        G = reference_cached(name, x, L)
        rs, Q = reference_summary(G, N, C.lattice(L))
        P = probes(x, L, N, G, Q, np.random.default_rng([SEED, 13, N]))
        for c in bench:
            if c["key"] in out["rows"]:
                continue
            t0 = time.perf_counter()
            rec, act = validate_parameter_set(name, x, L, G, rs, Q, P, xi=c["xi"], rc=c["rc"], kc=c["kc_shell"],
                                              M=c["M"], p=c["p"])
            rec.update(key=c["key"], norm2_ref=rs["norm2"], passes=bool(rec["E_op"] <= TAU),
                       cert_rel=c["cert_rel_max"], probe_opnorm_max=max(v["op_normalised"] for v in act.values()))
            out["rows"][c["key"]] = rec
            C.log(f"errors {name} {c['key']}: E_op {rec['E_op']:.3e} (cert {c['cert_rel_max']:.3e}) "
                  f"cached-vs-dense {rec['cached_vs_dense_max']:.1e} ({time.perf_counter() - t0:.1f} s)")
            C.write_json(f, dict(out, provenance=C.provenance()))
        C.write_json(f, dict(out, provenance=C.provenance()))


# ============================================================================ confirm
def stage_confirm(args):
    assert_one_thread()
    for name in args.configs:
        bench = C.read_json(path("bench", name, args.tag))
        cands = bench["candidates"]
        err = C.read_json(path("errors", name))["rows"]
        for c in cands:
            e = err[c["key"]]
            c.update(E_op=e["E_op"], passes=bool(e["E_op"] <= TAU and c["cert_rel_max"] <= TAU * (1 + 1e-12)))
        ok = sorted([c for c in cands if c["passes"]], key=lambda c: c["T_action_screen_best_ms"])
        pick = {c["key"]: c for c in ok[:N_CONF]}
        for p in bench["ps"]:
            cp = [c for c in ok if c["p"] == p]
            if cp:
                pick.setdefault(cp[0]["key"], cp[0])
        pick = list(pick.values())
        x, L = load(name)
        x = np.mod(x, L)
        N = len(x)
        Xs = x_inputs(N)
        fns = {}
        for c in pick:
            K, kc, mesh, _ = build_candidate(x, L, c)
            op = CachedGammaH(x, L, K, c["rc"], mesh)
            for i, fn in action_fns(op).items():
                fns[(c["key"], i)] = fn
        t0 = time.perf_counter()
        t = interleaved(fns, R_CONF, Xs, warm=5, seed=N + 1)
        C.log(f"confirm {name}: {len(fns)} actions x {R_CONF} rounds ({time.perf_counter() - t0:.0f} s)")
        raw, rows = {}, []
        for c in pick:
            for i in IMPLS:
                s = st(t[(c["key"], i)])
                rows.append(dict(key=c["key"], impl=i, p=c["p"], xi=c["xi"], kc=c["kc_shell"], M=c["M"], rc=c["rc"],
                                 n_img=c["n_img"], E_op=c["E_op"], cert_rel=c["cert_rel_max"],
                                 T_pred_ms=c[f"T_pred_{i}_ms"], T_setup_ms=c[f"T_setup_{i}_ms"],
                                 **{f"T_action_{k}": v for k, v in s.items()}))
                raw[f"{c['key']}|{i}"] = (1e3 * t[(c["key"], i)]).tolist()
        rows.sort(key=lambda r: r["T_action_median_ms"])
        best = rows[0]
        for r in rows:
            C.log(f"  {r['key']:52s} {r['impl']:8s} {r['T_action_median_ms']:6.2f} ms [{r['T_action_q1_ms']:.2f}, "
                  f"{r['T_action_q3_ms']:.2f}] pred {r['T_pred_ms']:.2f} E_op {r['E_op']:.2e}")
        pmin, pmax = min(bench["ps"]), max(bench["ps"])
        boundary = best["p"] in (pmin, pmax)
        neighbour = (best["p"] - 1) if best["p"] == pmin else (best["p"] + 1) if best["p"] == pmax else None
        opt = dict(config=name, best=best, chosen_impl=best["impl"], boundary_p=boundary,
                   neighbour_p_to_check=neighbour if boundary else None,
                   neighbour_supported=bool(neighbour in P_CAL) if boundary else None,
                   candidate=next(c for c in pick if c["key"] == best["key"]), confirm_rows=rows, rounds=R_CONF,
                   ps_searched=bench["ps"], provenance=C.provenance(blas=blas_info()))
        C.write_json(path("optimum", name, args.tag), opt)
        C.write_json(path("raw_confirm", name, args.tag), dict(unit="ms", times=raw))
        C.log(f"confirm {name}: measured optimum {best['key']} {best['impl']} {best['T_action_median_ms']:.2f} ms, "
              f"E_op {best['E_op']:.2e}" + (f"; p at search boundary -> check p = {neighbour}" if boundary else ""))


# ============================================================================ Stage 2: dense vs PPPM
def mode_pair_matrix_fast(mesh, x):
    """Mesh.pair_matrix in real arithmetic: the +-k terms of the mode sum are complex conjugates (real stencil
    weights, even G_h), so W = V^-1 [S_0 S_0^T G_h(0) + 2 sum_{k in half} (Re S Re S^T + Im S Im S^T) G_h(k)]."""
    s = mesh.stencil_dft(x)
    o, mv = mesh.mc, mesh.mv
    half = (mv[:, 2] > 0) | ((mv[:, 2] == 0) & (mv[:, 1] > 0)) | ((mv[:, 2] == 0) & (mv[:, 1] == 0) & (mv[:, 0] > 0))
    zero = np.all(mv == 0, axis=1)
    assert 2 * int(half.sum()) + int(zero.sum()) == len(mv)
    S = s[:, 0, mv[half, 0] + o] * s[:, 1, mv[half, 1] + o] * s[:, 2, mv[half, 2] + o]
    A = np.concatenate([S.real, S.imag], axis=1)
    Gh = mesh.Gm[half]
    N = len(x)
    W = np.empty((N, N, 3, 3))
    S0 = (s[:, 0, o] * s[:, 1, o] * s[:, 2, o]).real if zero.any() else None
    for a, b in PAIRS6:
        g = np.concatenate([Gh[:, a, b], Gh[:, a, b]])
        Z = (A * g) @ A.T * (2.0 / mesh.V)
        if S0 is not None:
            Z += np.outer(S0, S0) * (mesh.Gm[zero][0, a, b] / mesh.V)
        W[:, :, a, b] = W[:, :, b, a] = Z
    return W


def dense_project(x, L, K, rc, mesh):
    """Dense Gamma_h from positions with the project's GammaH.dense (unchanged)."""
    t = {}
    t0 = time.perf_counter()
    x = np.mod(x, L)
    il = kdtree_images(x, L, rc)
    t["images"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    op = GammaH.__new__(GammaH)
    op.x, op.L, op.N, op.rc, op.kc = x, float(L), len(x), float(rc), mesh.kc
    op.i, op.j, op.d = il.i, il.j, il.d
    op.w = short_range_weights(K, op.d)
    op.mesh = mesh
    t["weights"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    G = GammaH.dense(op)
    t["dense"] = time.perf_counter() - t2
    t["total"] = time.perf_counter() - t0
    return G, t


def dense_fast(x, L, K, rc, mesh):
    """The same dense Gamma_h, vectorised: real blocks by bincount, mode sum in real arithmetic, one lap_full."""
    t = {}
    t0 = time.perf_counter()
    x = np.mod(x, L)
    N = len(x)
    il = kdtree_images(x, L, rc)
    t["images"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    d = il.d
    w = short_range_weights(K, d)
    wd = w[:, None] * d
    flat = il.i.astype(np.int64) * N + il.j
    W = np.empty((N, N, 3, 3))
    for a, b in PAIRS6:
        W[:, :, a, b] = W[:, :, b, a] = np.bincount(flat, wd[:, a] * d[:, b], N * N).reshape(N, N)
    W = W + W.transpose(1, 0, 2, 3)
    t["real_blocks"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    W += mode_pair_matrix_fast(mesh, x)
    t["mode_pair_matrix"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    G = lap_full(W)
    t["laplacian"] = time.perf_counter() - t3
    t["total"] = time.perf_counter() - t0
    return G, t


def time_basis(op_fn, N):
    n = 3 * N
    G = np.empty((n, n))
    E = np.zeros(n)
    t0 = time.perf_counter()
    for c in range(n):
        E[c] = 1.0
        G[:, c] = op_fn(E.reshape(N, 3)).ravel()
        E[c] = 0.0
    return G, time.perf_counter() - t0


def stage_dense(args):
    assert_one_thread()
    for name in args.configs:
        opt = C.read_json(path("optimum", name, args.opt_tag))
        c = opt["candidate"]
        impl = opt["chosen_impl"]
        x, L = load(name)
        x = np.mod(x, L)
        N = len(x)
        Xs = x_inputs(N)
        xi, rc, kc, M, p = c["xi"], c["rc"], shell_kc(c["kc_shell"], L), c["M"], c["p"]
        K = C.kernel(xi)
        C.log(f"dense {name}: p={p} xi={xi:.4f} rc={rc:.6f} kc={kc:.6f} M={M}, matrix-free production action {impl}")
        info = dict(config=name, N=N, L=L, xi=xi, rc=rc, kc=kc, M=M, p=p, eta=kc * (L / M) / np.pi,
                    production_impl=impl, blas=blas_info())
        # -- one-time, parameter-only: Mesh (influence function on the M^3 grid and mode-space G_h)
        tm = []
        for _ in range(5):
            t0 = time.perf_counter()
            mesh = mesh_for(L, K, kc, M, p)
            tm.append(time.perf_counter() - t0)
        info["T_mesh_influence_once"] = st(np.array(tm))
        # -- one-time, accuracy only: the independent Gamma_ref (regenerated in memory, compared with raw/)
        t0 = time.perf_counter()
        Gref_new, _ = C.reference(x, L, rel_tol=REF_REL_TOL)
        info["T_reference_generation_s"] = time.perf_counter() - t0
        Gref = reference_cached(name, x, L)
        info["reference_regenerated_vs_raw_maxabs"] = float(np.abs(Gref_new - Gref).max())
        del Gref_new
        ref_norm = float(np.linalg.norm(Gref, 2))
        # -- builds from positions, interleaved: dense (project, vectorised) and PPPM setups (cached, pair-difference)
        builds = {"dense_project": lambda: dense_project(x, L, K, rc, mesh),
                  "dense_fast": lambda: dense_fast(x, L, K, rc, mesh),
                  "setup_cached": lambda: CachedGammaH(x, L, K, rc, mesh),
                  "setup_pairdiff": lambda: pairdiff_setup(x, L, K, rc, mesh)}
        for k in builds:
            builds[k]()
        tb = {k: np.empty(R_BUILD) for k in builds}
        parts = {"dense_project": [], "dense_fast": [], "setup_cached": []}
        rng = np.random.default_rng([SEED, 17, N])
        t0 = time.perf_counter()
        for r in range(R_BUILD):
            for o in rng.permutation(len(builds)):
                k = list(builds)[o]
                gc.collect()
                t1 = time.perf_counter()
                res = builds[k]()
                tb[k][r] = time.perf_counter() - t1
                if k in ("dense_project", "dense_fast"):
                    parts[k].append(res[1])
                elif k == "setup_cached":
                    parts[k].append(dict(res.t_setup_parts))
                del res
        C.log(f"dense {name}: {R_BUILD} interleaved build rounds ({time.perf_counter() - t0:.0f} s): "
              + ", ".join(f"{k} {1e3 * np.median(v):.1f} ms" for k, v in tb.items()))
        info["builds"] = {k: st(v) for k, v in tb.items()}
        info["build_parts_median_ms"] = {k: {q: 1e3 * float(np.median([d[q] for d in v])) for q in v[0]}
                                         for k, v in parts.items()}
        raw = dict(unit="ms", builds={k: (1e3 * v).tolist() for k, v in tb.items()})
        # -- the operators used below
        Gd, _ = dense_project(x, L, K, rc, mesh)
        Gf, _ = dense_fast(x, L, K, rc, mesh)
        op = CachedGammaH(x, L, K, rc, mesh)
        pd = pairdiff_setup(x, L, K, rc, mesh)
        f_mf = {"cached": op.apply, "pairdiff": lambda X: pd.matvec(X.ravel()).reshape(N, 3)}
        info.update(dense_shape=list(Gd.shape), dense_dtype=str(Gd.dtype), dense_MiB=Gd.nbytes / 2 ** 20,
                    dense_c_contiguous=bool(Gd.flags.c_contiguous))
        # -- basis-vector assembly through the matrix-free production action (3N actions), 3 timings
        tbasis, Gb = [], None
        for _ in range(3):
            Gb, tt = time_basis(f_mf[impl], N)
            tbasis.append(tt)
        info["T_dense_basis"] = st(np.array(tbasis))
        raw["basis_s"] = tbasis
        # -- identity checks
        nGd = float(np.linalg.norm(Gd, 2))
        idc = dict(norm2_dense=nGd,
                   fast_vs_project_rel2=float(np.linalg.norm(Gf - Gd, 2) / nGd),
                   fast_vs_project_maxabs_over_maxabs=float(np.abs(Gf - Gd).max() / np.abs(Gd).max()),
                   basis_vs_project_rel2=float(np.linalg.norm(Gb - Gd, 2) / nGd),
                   basis_vs_project_maxabs_over_maxabs=float(np.abs(Gb - Gd).max() / np.abs(Gd).max()),
                   symmetry_project=float(np.linalg.norm(Gd - Gd.T) / np.linalg.norm(Gd)),
                   E_op_project=float(np.linalg.norm(Gd - Gref, 2) / ref_norm),
                   E_op_fast=float(np.linalg.norm(Gf - Gref, 2) / ref_norm),
                   E_op_basis=float(np.linalg.norm(Gb - Gref, 2) / ref_norm), norm2_ref=ref_norm)
        vec = []
        for k, X in enumerate(Xs):
            gx = Gd @ X.ravel()
            row = dict(kind="random", k=k, norm_GX=float(np.linalg.norm(gx)))
            for i, fn in f_mf.items():
                dv = float(np.linalg.norm(gx - fn(X).ravel()))
                row[f"rel_{i}"] = dv / max(row["norm_GX"], 1e-300)
            row["rel_fast_dense"] = float(np.linalg.norm(gx - Gf @ X.ravel())) / row["norm_GX"]
            vec.append(row)
        for a in range(3):
            T = np.zeros((N, 3))
            T[:, a] = 1.0
            gx = Gd @ T.ravel()
            small = 1e-12 * nGd * np.linalg.norm(T)
            row = dict(kind="translation", k=a, norm_GX=float(np.linalg.norm(gx)),
                       null_dense_opnorm=float(np.linalg.norm(gx) / (nGd * np.linalg.norm(T))))
            for i, fn in f_mf.items():
                y = fn(T).ravel()
                row[f"null_{i}_opnorm"] = float(np.linalg.norm(y) / (nGd * np.linalg.norm(T)))
                row[f"rel_{i}"] = float(np.linalg.norm(gx - y)) / max(row["norm_GX"], small)
            vec.append(row)
        idc["vectors"] = vec
        idc["max_rel_random"] = {i: max(r[f"rel_{i}"] for r in vec if r["kind"] == "random") for i in IMPLS}
        idc["max_rel_translation_vs_small"] = {i: max(r[f"rel_{i}"] for r in vec if r["kind"] == "translation")
                                               for i in IMPLS}
        info["identity"] = idc
        C.log(f"dense {name}: identity fast/project {idc['fast_vs_project_rel2']:.1e}, basis/project "
              f"{idc['basis_vs_project_rel2']:.1e}, max rel dense-vs-matrix-free {idc['max_rel_random']}, "
              f"E_op {idc['E_op_project']:.3e}/{idc['E_op_basis']:.3e}")
        del Gf, Gb
        # -- GEMV vs matrix-free actions, interleaved, the same 10 X
        assert Gd.dtype == np.float64 and Gd.flags.c_contiguous
        fns = {"dense_GEMV": lambda X: Gd @ X.ravel(), "pppm_cached": op.apply,
               "pppm_pairdiff": lambda X: pd.matvec(X.ravel())}
        t0 = time.perf_counter()
        t = interleaved(fns, R_GEMV, Xs, warm=10, seed=N + 2)
        C.log(f"dense {name}: {R_GEMV} interleaved action rounds ({time.perf_counter() - t0:.0f} s): "
              + ", ".join(f"{k} {1e3 * np.median(v):.3f} ms" for k, v in t.items()))
        info["actions"] = {k: st(v) for k, v in t.items()}
        raw["actions"] = {k: (1e3 * v).tolist() for k, v in t.items()}
        # -- end-to-end, per configuration of positions: build once, q actions
        Ta = info["actions"][f"pppm_{impl}"]["median_ms"]
        Ts = info["builds"][f"setup_{impl}"]["median_ms"]
        Tg = info["actions"]["dense_GEMV"]["median_ms"]
        e2e = {}
        for bname, Tb in (("dense_project", info["builds"]["dense_project"]["median_ms"]),
                          ("dense_fast", info["builds"]["dense_fast"]["median_ms"]),
                          ("dense_basis", info["T_dense_basis"]["median_ms"])):
            qs = {}
            for q in (1, 10, 40, 80):
                qs[q] = dict(dense_ms=Tb + q * Tg, pppm_ms=Ts + q * Ta, ratio_dense_over_pppm=(Tb + q * Tg) / (Ts + q * Ta))
            be = (Tb - Ts) / (Ta - Tg) if Ta > Tg else None
            e2e[bname] = dict(T_build_ms=Tb, q=qs, break_even_q=be,
                              note=("dense cheaper for q > break_even_q" if be is not None and be > 0 else
                                    "dense cheaper for every q" if be is not None else
                                    "GEMV not faster than the matrix-free action: no break-even"))
        info["end_to_end"] = dict(T_setup_ms=Ts, T_action_ms=Ta, T_GEMV_ms=Tg, builds=e2e,
                                  ratio_GEMV_over_action=Tg / Ta,
                                  ratio_note="T_dense_GEMV / T_PPPM_action; > 1 means PPPM faster")
        info["provenance"] = C.provenance(blas=blas_info())
        C.write_json(path("dense512", name), info)
        C.write_json(path("raw_dense512", name), raw)
        C.log(f"dense {name}: GEMV {Tg:.3f} ms vs PPPM {impl} {Ta:.2f} ms (ratio {Tg / Ta:.3f}); build project "
              f"{e2e['dense_project']['T_build_ms']:.0f} ms, fast {e2e['dense_fast']['T_build_ms']:.0f} ms, basis "
              f"{e2e['dense_basis']['T_build_ms']:.0f} ms vs setup {Ts:.1f} ms; break-even q "
              + ", ".join(f"{k} {v['break_even_q']}" for k, v in e2e.items()))
        del Gd, op, pd
        gc.collect()


STAGES = dict(calibrate=stage_calibrate, select=stage_select, bench=stage_bench, errors=stage_errors,
              confirm=stage_confirm, dense=stage_dense)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", required=True, choices=list(STAGES))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--p", type=int, nargs="+", default=list(P_SET))
    ap.add_argument("--only-p", action="store_true", help="bench: only the orders given by --p")
    ap.add_argument("--tag", default="main", help="bench/errors/confirm file tag")
    ap.add_argument("--opt-tag", default="main", help="dense: tag of the optimum file to use")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    C.RESULTS = RESULTS_PSEARCH
    RESULTS_PSEARCH.mkdir(parents=True, exist_ok=True)
    C.log_command()
    for s in args.stage:
        t0 = time.perf_counter()
        STAGES[s](args)
        C.log(f"stage {s} done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
