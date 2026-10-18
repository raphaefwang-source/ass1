#!/usr/bin/env python3
"""
Fixed-accuracy scaling benchmark of Method A: pair-summed direct periodic sum vs optimised matrix-free PPPM,
N = 128 ... 4096 at rho = 0.75, tau = 2e-7 (relative spectral norm). Outputs: results_scaling/, configs_scaling/,
logs_scaling/ (earlier results are only read).

Physical operator (manuscript Eqs. kernelA, Kper, Gamma): (Gamma X)_i = sum_{j != i} K^per(q_i - q_j)(X_i - X_j),
K(r) = c e^{-kappa r}/r rhat rhat^T, full periodic lattice sum, k = 0 retained, no minimum image; common.py constants.

Methods (same positions, box, kernel, inputs and tau for every N):
  Direct  direct_periodic.DirectPairs: setup tabulates K^per_{R_d}(q_i - q_j) for all N(N-1)/2 pairs (truncated
          image sum, streamed over lattice shifts); action loops over the pairs. R_d = smallest radius on a 0.125 grid
          whose certified bound 2 [max_i lambda_max(discarded K) + tail] <= tau Lambda_lb.
          One-time (per N, configuration independent): pair index arrays and lattice-shift list (DirectGeometry).
  PPPM    Method A (Ewald split at s = xi^2, existing B-spline PPPM), (xi, rc, kc, M, p) from the joint certified
          budget with the Hurwitz-zeta mesh bound (joint_optimizer, approach C) for every N; the fastest MEASURED
          passing candidate and the faster of the two numerically identical actions (cached D-H / pair-difference).
          One-time (per parameter set, configuration independent): the Mesh (influence function).

Stages
  configs    strong and uniform at every N (clustered_configurations.generate, seed 1; N = 256, 512: the existing
             configs/*_s1.npz are reused unchanged); new ones in configs_scaling/
  refs       Gamma_ref (LatticeFriction, rel_tol 1e-13, raw/<name>_Gref.npy, git-ignored); ||Gamma_ref||_2 (dense
             eigvalsh for N <= 2048, ARPACK lambda_max of the PSD matrix for N = 4096), Lambda_lb (exact)
  calibrate  per-N per-p cost model (strong configuration; reused for uniform at the same N), both actions
  select     per configuration: p in P_SCALE, xi on [XI_LO, XI_HI] (0.025 grid without exact rc, then 0.005 around
             each p's best), exact rc and errors (finalize) for each p's optimum, alternatives and xi neighbours
  bench      screening (interleaved) of every p's model optimum and the alternatives of the 3 best predicted p
  confirm    top candidates + best of every p, both actions, interleaved; measured optimum
  verify     E_op of the confirmed PPPM candidates: dense (N <= 1024) and/or ARPACK (N >= 512)
  direct     certified R_d, the direct operator's measured accuracy (dense for N <= 2048, ARPACK), reference check
             Gamma_D(36) vs Gamma_ref, optional image-resolved identity check (N <= 256)
  scale      final interleaved timing of direct vs PPPM (setups, actions), T_total(q)
  memory     tracemalloc peaks of setup and action, persistent bytes (separate pass, not timed)

    python3 run_scaling.py --stage <stage> [--N ...] [--fam strong uniform]
"""
import argparse
import gc
import time
import tracemalloc

import common as C                       # first: sets OMP/OPENBLAS/MKL_NUM_THREADS = 1 before numpy loads OpenBLAS
import numpy as np
from scipy.optimize import nnls
from scipy.sparse.linalg import LinearOperator, eigsh

import clustered_configurations as CC
from cached_operator import CachedGammaH, mesh_for, pair_difference_action
from parameter_optimizer import RealSpaceTail, A_MAX, R_PAD, shell_kc, lambda_lower_bound, smooth_numbers
from joint_optimizer import evaluate_xi, finalize, PCostModel, TAU
from direct_periodic import DirectPairs, DirectGeometry, DirectImages, degree_bound_curve, tensors_to_dense
from error_validation import REF_REL_TOL
from run_psearch_dense512 import (x_inputs, interleaved, st, pairdiff_setup, action_fns, cand_key, blas_info,
                                  assert_one_thread, fit_rel, with_preds, dense_fast, IMPLS)

RESULTS = C.HERE / "results_scaling"
CONFIGS_SCALING = C.HERE / "configs_scaling"
NS = (128, 256, 512, 1024, 2048, 4096)
FAMS = ("strong", "uniform")
P_SCALE = (6, 7, 8, 9, 10, 11, 12)
XI_LO, XI_HI = 0.50, 1.30
TOPK = 80
N_TOP_P = 3
R_DIRECT_REF = 36.0
CERT_EDGES = np.round(np.concatenate([np.arange(22.0, 34.0001, 0.125), [40.0]]), 4)
QS = (1, 10, 40, 80)
SEED = 20261018


def rounds(N, table):
    return table[min(table, key=lambda k: abs(np.log(k) - np.log(N)))]


R_CAL = {128: 30, 256: 30, 512: 30, 1024: 20, 2048: 12, 4096: 8}
R_SCREEN = {128: 40, 256: 40, 512: 40, 1024: 30, 2048: 20, 4096: 15}
R_CONF = {128: 150, 256: 150, 512: 150, 1024: 100, 2048: 60, 4096: 40}
R_SET = {128: 20, 256: 20, 512: 20, 1024: 10, 2048: 6, 4096: 5}
R_ACT = {128: 200, 256: 200, 512: 200, 1024: 100, 2048: 50, 4096: 30}
DENSE_MAX_PPPM = 1024                    # dense Gamma_h (vectorised mode-space assembly) up to this N
DENSE_MAX_DIRECT = 2048                  # dense Gamma_D (pair tensors -> 3N x 3N) up to this N


def name_of(fam, N):
    return f"{fam}_N{N}_s1"


def load_cfg(name):
    f = CONFIGS_SCALING / f"{name}.npz"
    if not f.exists():
        f = C.CONFIGS / f"{name}.npz"
    with np.load(f) as d:
        return np.mod(d["q"].copy(), float(d["L"])), float(d["L"])


def path(kind, name=None):
    return RESULTS / (f"{kind}_{name}.json" if name else f"{kind}.json")


def M_ref(N):
    return 40.0 * (N / 512) ** (1 / 3)


def M_cap(N):
    return int(round(128 * (N / 512) ** (1 / 3)))


def smooth_near(m):
    return int(smooth_numbers(int(np.ceil(m)), int(np.ceil(m)) + 64)[0])


def gref(name, x, L):
    """Gamma_ref from raw/ (computed and stored if missing). Returns (G, generation time or None)."""
    f = C.RAW / f"{name}_Gref.npy"
    if f.exists():
        return np.load(f), None
    t0 = time.perf_counter()
    G, _ = C.reference(x, L, rel_tol=REF_REL_TOL)
    t = time.perf_counter() - t0
    C.RAW.mkdir(parents=True, exist_ok=True)
    np.save(f, G)
    return G, t


def sym_norm2(A):
    """Spectral norm of a matrix that is symmetric up to round-off: max |eig(sym part)| + ||antisym part||_F
    (an upper bound; the antisymmetric part is reported separately)."""
    S = 0.5 * (A + A.T)
    lam = np.linalg.eigvalsh(S)
    asym = float(np.linalg.norm(0.5 * (A - A.T)))
    return float(max(abs(lam[0]), abs(lam[-1]))), asym


def arpack_norm(matvec, n, which="LM", tol=1e-8, seed=0):
    """Largest-magnitude (or largest algebraic) eigenvalue of a symmetric operator by ARPACK (Krylov estimate,
    converged to tol; a lower bound of the spectral norm in exact arithmetic) and its residual."""
    v0 = np.random.default_rng([SEED, 31, n, seed]).standard_normal(n)
    cnt = {"n": 0}

    def mv(v):
        cnt["n"] += 1
        return matvec(v)
    op = LinearOperator((n, n), matvec=mv, dtype=float)
    t0 = time.perf_counter()
    lam, vec = eigsh(op, k=1, which=which, tol=tol, v0=v0, ncv=24, maxiter=2000)
    v = vec[:, 0]
    res = float(np.linalg.norm(matvec(v) - lam[0] * v))
    return dict(value=float(abs(lam[0])), eig=float(lam[0]), residual=res, matvecs=cnt["n"], tol=tol, which=which,
                t_s=time.perf_counter() - t0)


# ============================================================================ configs
def stage_configs(args):
    out = C.read_json(path("configurations"), {"rows": {}})
    CONFIGS_SCALING.mkdir(parents=True, exist_ok=True)
    for N in args.N:
        for fam in args.fam:
            name = name_of(fam, N)
            if (C.CONFIGS / f"{name}.npz").exists():
                x, L = load_cfg(name)
                hist, stt = CC.structure_stats(x, L)
                out["rows"][name] = dict(name=name, source="configs/ (earlier rounds, unchanged)", N=N, L=L, **stt)
                C.log(f"configs {name}: reused from configs/ (d_min {stt['d_min_periodic']:.4f})")
                continue
            if (CONFIGS_SCALING / f"{name}.npz").exists() and name in out["rows"]:
                continue
            t0 = time.perf_counter()
            x, rec = CC.generate(fam, N, 1)
            hist, stt = CC.structure_stats(x, rec["L"])
            np.savez_compressed(CONFIGS_SCALING / f"{name}.npz", q=x, L=rec["L"],
                                labels=rec.get("labels", np.zeros(N, int)), hist=hist["hist"], g=hist["g"],
                                hist_edges=CC.HIST_EDGES, nb_first=hist["nb_first"], nb_rc=hist["nb_rc"],
                                Sk_small=hist["Sk_small"])
            row = {k: v for k, v in rec.items() if k != "labels"}
            row.update(stt, name=name, source="configs_scaling/ (clustered_configurations.generate, seed 1)",
                       t_generate_s=time.perf_counter() - t0)
            out["rows"][name] = row
            C.log(f"configs {name}: L={rec['L']:.4f} attempt={rec['attempt']} iters={rec['push_iterations']} "
                  f"d_min={stt['d_min_periodic']:.4f} sigma_real/req={rec.get('sigma_realised_over_requested')} "
                  f"({row['t_generate_s']:.0f} s)")
            C.write_json(path("configurations"), dict(out, provenance=C.provenance()))
    C.write_json(path("configurations"), dict(out, provenance=C.provenance()))


# ============================================================================ refs
def stage_refs(args):
    for N in args.N:
        for fam in args.fam:
            name = name_of(fam, N)
            f = path("reference", name)
            if f.exists() and not args.force:
                continue
            x, L = load_cfg(name)
            G, t_gen = gref(name, x, L)
            n = 3 * N
            rec = dict(name=name, N=N, L=L, t_generate_s=t_gen, nbytes=int(G.nbytes),
                       sym=float(np.linalg.norm(G - G.T) / np.linalg.norm(G)), lam_lb=lambda_lower_bound(G, N))
            T = np.zeros((N, 3))
            T[:, 0] = 1.0
            rec["null_rel"] = float(np.linalg.norm(G @ T.ravel()) / (np.abs(G).max() * np.sqrt(N)))
            t0 = time.perf_counter()
            if N <= 2048:
                lam = np.linalg.eigvalsh(G)
                rec.update(norm2=float(lam[-1]), lam_min_all=float(lam[0]), norm2_method="dense eigvalsh")
            else:
                a = arpack_norm(lambda v: G @ v, n, which="LA", tol=1e-10)
                rec.update(norm2=a["value"], norm2_method="ARPACK lambda_max (PSD)", arpack=a)
            rec["t_norm_s"] = time.perf_counter() - t0
            C.write_json(f, dict(rec, provenance=C.provenance()))
            C.log(f"refs {name}: ||G||={rec['norm2']:.4f} ({rec['norm2_method']}) Lambda_lb={rec['lam_lb']:.4f} "
                  f"gen {t_gen} s, sym {rec['sym']:.1e}")
            del G
            gc.collect()


def ref_info(name):
    return C.read_json(path("reference", name))


# ============================================================================ calibrate
def stage_calibrate(args):
    assert_one_thread()
    for N in args.N:
        name = name_of("strong", N)
        x, L = load_cfg(name)
        K = C.kernel(0.75)
        Xs = x_inputs(N)
        R = rounds(N, R_CAL)
        Mr = M_ref(N)
        rows = []
        t0 = time.perf_counter()
        small = mesh_for(L, K, 2 * np.pi * 2 / L, 16, 4)
        for rc in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
            op = CachedGammaH(x, L, K, rc, small)
            t = interleaved({"cached": op.real_H,
                             "pairdiff": lambda X, op=op: pair_difference_action(op.i, op.j, op.d, op.w, X, N)}, R, Xs)
            for impl in IMPLS:
                rows.append(dict(kind="real", impl=impl, rc=rc, n_img=op.n_images, p=4, M=16, t=float(np.median(t[impl]))))
        Msg = sorted({smooth_near(f * Mr) for f in (0.75, 1.0, 1.5)})
        for p in P_SCALE:
            for M in Msg:
                op = CachedGammaH(x, L, K, 1.5, mesh_for(L, K, 2 * np.pi * 4 / L, M, p))
                ph = op.fft_influence(op.spread(Xs[0]))
                t = interleaved({"spread": op.spread, "gather": lambda X, op=op, ph=ph: op.gather(ph)}, R, Xs)
                rows.append(dict(kind="spread_gather", impl="both", p=p, M=M, rc=1.5, n_img=op.n_images,
                                 t=float(np.median(t["spread"]) + np.median(t["gather"]))))
        Mfft = [m for m in smooth_numbers(int(0.5 * Mr), int(3.3 * Mr))]
        Mfft = Mfft[::max(1, len(Mfft) // 18)]
        for M in Mfft:
            op = CachedGammaH(x, L, K, 1.5, mesh_for(L, K, 2 * np.pi * 4 / L, M, 4))
            g = op.spread(Xs[0])
            t = interleaved({"fft": lambda X, op=op, g=g: op.fft_influence(g)}, R, Xs)
            rows.append(dict(kind="fft", impl="both", M=M, p=4, rc=1.5, n_img=op.n_images, t=float(np.median(t["fft"]))))
        pts = [(4.5, smooth_near(0.8 * Mr)), (6.0, smooth_near(1.1 * Mr)), (5.0, smooth_near(1.5 * Mr)),
               (6.5, smooth_near(0.9 * Mr))]
        for p in P_SCALE:
            for rc, M in pts:
                op = CachedGammaH(x, L, K, rc, mesh_for(L, K, 0.5 * np.pi * M / L, M, p))
                t = interleaved(action_fns(op), R, Xs)
                for impl in IMPLS:
                    rows.append(dict(kind="full", impl=impl, p=p, M=M, rc=rc, n_img=op.n_images,
                                     t=float(np.median(t[impl]))))
            del op
            gc.collect()
        fft = [r for r in rows if r["kind"] == "fft"]
        (C_F, b_F), e_F = fit_rel(np.array([r["M"] ** 3 * np.log(r["M"] ** 3) for r in fft], float),
                                  np.array([r["t"] for r in fft]))
        models = {}
        for impl in IMPLS:
            real = [r for r in rows if r["kind"] == "real" and r["impl"] == impl]
            (C_R, b_R), e_R = fit_rel(np.array([r["n_img"] for r in real], float), np.array([r["t"] for r in real]))
            per_p, info = {}, {}
            for p in P_SCALE:
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
            models[impl] = cm.as_dict()
            for r in rows:
                if r["kind"] == "full" and r["impl"] == impl:
                    r["t_pred"] = cm.predict(r["n_img"], N, r["p"], r["M"])
        rank = min(IMPLS, key=lambda i: models[i]["C_R"])
        C.write_json(path("cost_model", f"N{N}"), dict(N=N, config=name, models=models, rank_impl=rank, rows=rows,
                                                       rounds=R, provenance=C.provenance(blas=blas_info())))
        errs = [abs(e) for i in IMPLS for v in models[i]["per_p_info"].values() for e in v["full_rel_err"]]
        C.log(f"calibrate N={N}: rank {rank}; C_R {models['cached']['C_R'] * 1e9:.1f}/{models['pairdiff']['C_R'] * 1e9:.1f} ns; "
              f"C_F {C_F * 1e9:.2f} ns; S_p [ms] " + " ".join(f"{p}:{1e3 * models['pairdiff']['per_p'][str(p)]['S']:.2f}"
                                                            for p in P_SCALE)
              + f"; full-action fit max err {max(errs):.2f} ({time.perf_counter() - t0:.0f} s)")


def cost_models(N):
    d = C.read_json(path("cost_model", f"N{N}"))
    return {impl: PCostModel.from_dict(d["models"][impl]) for impl in IMPLS}, d["rank_impl"]


# ============================================================================ select
def stage_select(args):
    for N in args.N:
        models, rank = cost_models(N)
        cm = models[rank]
        for fam in args.fam:
            name = name_of(fam, N)
            if path("selection", name).exists() and not args.force:
                continue
            t0 = time.perf_counter()
            x, L = load_cfg(name)
            ri = ref_info(name)
            seed = dict(name=name, rst=RealSpaceTail(x, L, (A_MAX + R_PAD) / XI_LO), lam_lb=ri["lam_lb"])
            mcap = M_cap(N)
            grid = {}

            def scan(xi, plist):
                curves = None
                for p in plist:
                    if (p, xi) in grid:
                        continue
                    out, _, curves = evaluate_xi(xi, [seed], L, N, cm, p=p, approaches=("C",), curves=curves,
                                                 variants=("hz",), M_cap=mcap, topk=1, finalize_best=False)
                    r = out["C"]
                    grid[(p, xi)] = None if r is None else dict(r["top"][0], p=p, n_candidates=r["n_candidates"])

            for xi in np.round(np.arange(XI_LO, XI_HI + 1e-9, 0.025), 4):
                scan(float(xi), args.p)
            t1 = time.perf_counter()
            want = {}
            for p in args.p:
                g = {xi: r for (q, xi), r in grid.items() if q == p and r is not None}
                xb = min(g, key=lambda k: g[k]["T_pred_grid"])
                for xi in np.round(np.arange(xb - 0.025, xb + 0.02501, 0.005), 4):
                    if XI_LO <= xi <= XI_HI:
                        want.setdefault(float(xi), []).append(p)
            for xi in sorted(want):
                scan(xi, want[xi])
            t2 = time.perf_counter()
            per_p = {}
            for p in args.p:
                g = {xi: r for (q, xi), r in grid.items() if q == p and r is not None}
                xo = min(g, key=lambda k: g[k]["T_pred_grid"])
                out, tab, curves = evaluate_xi(xo, [seed], L, N, cm, p=p, approaches=("C",), variants=("hz",),
                                               M_cap=mcap, topk=TOPK)
                opt = with_preds(out["C"]["best"], models, N)
                alts, Ms = [], {opt["M"]}
                for t in out["C"]["top"]:
                    if t["M"] in Ms:
                        continue
                    Ms.add(t["M"])
                    alts.append(with_preds(finalize(xo, tab, curves, [seed], t["i"], t["j"], np.array(t["B"]),
                                                    np.array(t["g"]), "hz", cm, N, p, TAU), models, N))
                    break
                neigh = []
                for o in (-0.025, 0.025):
                    xn = round(xo + o, 4)
                    if XI_LO <= xn <= XI_HI:
                        on, _, _ = evaluate_xi(xn, [seed], L, N, cm, p=p, approaches=("C",), variants=("hz",),
                                               M_cap=mcap, topk=1)
                        if on["C"] is not None:
                            neigh.append(with_preds(on["C"]["best"], models, N))
                per_p[str(p)] = dict(opt=opt, alternatives=alts, xi_neighbours=neigh,
                                     xi_at_boundary=bool(xo <= XI_LO + 1e-9 or xo >= XI_HI - 1e-9),
                                     M_at_cap=bool(opt["M"] >= mcap),
                                     grid={f"{xi:.4f}": g[xi] for xi in sorted(g)})
                C.log(f"select {name} p={p}: xi {opt['xi']:.3f} kc {opt['kc']:.3f} M {opt['M']} rc {opt['rc']:.3f} "
                      f"n_img {opt['n_img_mean']:.0f} cert {opt['cert_rel_max']:.4e} pred {opt[f'T_pred_{rank}_ms']:.2f} ms")
            best_p = min(per_p, key=lambda k: per_p[k]["opt"][f"T_pred_{rank}_ms"])
            ps = sorted(int(k) for k in per_p)
            res = dict(config=name, N=N, L=L, lam_lb=ri["lam_lb"], norm2=ri["norm2"], tau=TAU, ps=ps, rank_impl=rank,
                       M_cap=mcap, xi_range=[XI_LO, XI_HI], best_p_model=int(best_p),
                       p_at_boundary=bool(int(best_p) in (ps[0], ps[-1])),
                       fft_term_at_M_cap_ms=1e3 * float(cm.C_F * mcap ** 3 * np.log(mcap ** 3)),
                       per_p=per_p, t_coarse_s=t1 - t0, t_refine_s=t2 - t1, t_total_s=time.perf_counter() - t0,
                       provenance=C.provenance())
            C.write_json(path("selection", name), res)
            C.log(f"select {name}: model best p={best_p} ({per_p[best_p]['opt'][f'T_pred_{rank}_ms']:.2f} ms), boundary "
                  f"p {res['p_at_boundary']}, xi boundary {[k for k, v in per_p.items() if v['xi_at_boundary']]}, "
                  f"M cap {[k for k, v in per_p.items() if v['M_at_cap']]} ({res['t_total_s']:.0f} s)")
            del seed
            gc.collect()


# ============================================================================ bench (screening)
def candidates(sel):
    rk = sel["rank_impl"]
    out = [("opt", g["opt"]) for g in sel["per_p"].values()]
    top_p = sorted(sel["per_p"], key=lambda k: sel["per_p"][k]["opt"][f"T_pred_{rk}_ms"])[:N_TOP_P]
    for k in top_p:
        g = sel["per_p"][k]
        out += [("alt_M", a) for a in g["alternatives"]] + [("xi_neighbour", a) for a in g["xi_neighbours"]]
    seen, uniq = {}, []
    for role, r in out:
        key = cand_key(r)
        if key in seen:
            seen[key]["roles"].append(role)
            continue
        c = dict(r, roles=[role], key=key)
        seen[key] = c
        uniq.append(c)
    return uniq


def build_mesh(L, c):
    K = C.kernel(c["xi"])
    kc = shell_kc(c["kc"], L)
    t0 = time.perf_counter()
    mesh = mesh_for(L, K, kc, c["M"], c["p"])
    t = time.perf_counter() - t0
    assert mesh.mc == int(np.floor(kc * L / (2 * np.pi) + 1e-12)) and \
        np.isclose(np.max(np.linalg.norm(mesh.kv, axis=1)), kc, rtol=1e-12), "outer shell lost"
    return K, kc, mesh, t


def stage_bench(args):
    assert_one_thread()
    for N in args.N:
        for fam in args.fam:
            name = name_of(fam, N)
            sel = C.read_json(path("selection", name))
            cands = candidates(sel)
            x, L = load_cfg(name)
            Xs = x_inputs(N)
            C.log(f"bench {name}: {len(cands)} candidates")
            meshes, ops, setup = {}, {}, {c["key"]: {i: [] for i in IMPLS} for c in cands}
            for c in cands:
                K, kc, mesh, tm = build_mesh(L, c)
                meshes[c["key"]] = (K, mesh)
                c.update(kc_shell=kc, t_mesh_build_ms=1e3 * tm, n_modes=mesh.n_modes, eta_actual=mesh.eta_actual)
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
                f = action_fns(ops[c["key"]])
                y = f["cached"](Xs[0]).ravel()
                c.update(n_img=ops[c["key"]].n_images,
                         cached_vs_pairdiff_rel=float(np.linalg.norm(y - f["pairdiff"](Xs[0])) / np.linalg.norm(y)),
                         **{f"T_setup_{i}_ms": 1e3 * float(np.median(setup[c["key"]][i])) for i in IMPLS})
                for i in IMPLS:
                    fns[(c["key"], i)] = f[i]
            R = rounds(N, R_SCREEN)
            t0 = time.perf_counter()
            t = interleaved(fns, R, Xs, seed=N + 100)
            raw = {}
            for c in cands:
                for i in IMPLS:
                    s = st(t[(c["key"], i)])
                    c.update({f"T_action_{i}_screen_{k}": v for k, v in s.items()})
                    raw[f"{c['key']}|{i}"] = (1e3 * t[(c["key"], i)]).tolist()
                c["T_action_screen_best_ms"] = min(c[f"T_action_{i}_screen_median_ms"] for i in IMPLS)
            C.log(f"bench {name}: screening {len(fns)} actions x {R} rounds ({time.perf_counter() - t0:.0f} s); best "
                  + ", ".join(f"{c['key']} {c['T_action_screen_best_ms']:.2f}"
                              for c in sorted(cands, key=lambda c: c["T_action_screen_best_ms"])[:4]))
            C.write_json(path("bench", name), dict(config=name, N=N, candidates=cands, rounds=R,
                                                   provenance=C.provenance(blas=blas_info())))
            C.write_json(path("raw_screen", name), dict(unit="ms", times=raw))
            del ops, fns, meshes
            gc.collect()


# ============================================================================ confirm
def stage_confirm(args):
    assert_one_thread()
    for N in args.N:
        for fam in args.fam:
            name = name_of(fam, N)
            bench = C.read_json(path("bench", name))
            cands = sorted(bench["candidates"], key=lambda c: c["T_action_screen_best_ms"])
            pick = {c["key"]: c for c in cands[:4]}
            for p in sorted({c["p"] for c in cands}):
                cp = [c for c in cands if c["p"] == p]
                pick.setdefault(cp[0]["key"], cp[0])
            pick = list(pick.values())
            x, L = load_cfg(name)
            Xs = x_inputs(N)
            fns = {}
            for c in pick:
                K, kc, mesh, _ = build_mesh(L, c)
                op = CachedGammaH(x, L, K, c["rc"], mesh)
                for i, fn in action_fns(op).items():
                    fns[(c["key"], i)] = fn
            R = rounds(N, R_CONF)
            t0 = time.perf_counter()
            t = interleaved(fns, R, Xs, warm=5, seed=N + 101)
            rows, raw = [], {}
            for c in pick:
                for i in IMPLS:
                    s = st(t[(c["key"], i)])
                    rows.append(dict(key=c["key"], impl=i, p=c["p"], xi=c["xi"], kc=c["kc_shell"], M=c["M"], rc=c["rc"],
                                     n_img=c["n_img"], cert_rel=c["cert_rel_max"], T_pred_ms=c[f"T_pred_{i}_ms"],
                                     T_setup_ms=c[f"T_setup_{i}_ms"], **{f"T_action_{k}": v for k, v in s.items()}))
                    raw[f"{c['key']}|{i}"] = (1e3 * t[(c["key"], i)]).tolist()
            rows.sort(key=lambda r: r["T_action_median_ms"])
            C.write_json(path("confirm", name), dict(config=name, N=N, rows=rows, rounds=R,
                                                     candidates={c["key"]: c for c in pick},
                                                     provenance=C.provenance(blas=blas_info())))
            C.write_json(path("raw_confirm", name), dict(unit="ms", times=raw))
            C.log(f"confirm {name}: {len(fns)} actions x {R} rounds ({time.perf_counter() - t0:.0f} s): "
                  + "; ".join(f"{r['key']} {r['impl']} {r['T_action_median_ms']:.2f}" for r in rows[:6]))


# ============================================================================ verify (PPPM accuracy)
def pppm_error(name, c, x, L, G, nref, dense, arpack):
    N = len(x)
    K, kc, mesh, _ = build_mesh(L, c)
    op = CachedGammaH(x, L, K, c["rc"], mesh)
    pd = op.existing_gammah()
    out = dict(key=c["key"])
    if dense:
        t0 = time.perf_counter()
        Gh, _ = dense_fast(x, L, K, c["rc"], mesh)
        nd, asym = sym_norm2(Gh - G)
        out.update(E_op_dense=nd / nref, E_op_dense_asym_fro=asym / nref, t_dense_s=time.perf_counter() - t0)
        Xs = x_inputs(N)[:3]
        out["matrix_free_vs_dense"] = max(float(np.linalg.norm(pd.matvec(X.ravel()) - Gh @ X.ravel())
                                                / np.linalg.norm(Gh @ X.ravel())) for X in Xs)
        del Gh
    if arpack:
        a = arpack_norm(lambda v: pd.matvec(v) - G @ v, 3 * N, which="LM", tol=1e-6)
        out.update(E_op_arpack=a["value"] / nref, arpack=a)
    out["E_op"] = out.get("E_op_dense", out.get("E_op_arpack"))
    out["E_op_method"] = "dense (eigvalsh of the symmetric part + antisymmetric Frobenius)" if dense else \
        "ARPACK largest |eigenvalue| of Gamma_h - Gamma_ref (matrix-free Gamma_h, dense Gamma_ref GEMV)"
    return out


def stage_verify(args):
    for N in args.N:
        for fam in args.fam:
            name = name_of(fam, N)
            conf = C.read_json(path("confirm", name))
            x, L = load_cfg(name)
            G, _ = gref(name, x, L)
            nref = ref_info(name)["norm2"]
            f = path("verify", name)
            res = C.read_json(f, {"rows": {}})
            rows = conf["rows"]
            keys = list(dict.fromkeys(r["key"] for r in rows))
            if N > DENSE_MAX_PPPM:
                keys = keys[:2]                                  # measured optimum and runner-up
            for k in keys:
                if k in res["rows"]:
                    continue
                c = conf["candidates"][k]
                dense = N <= DENSE_MAX_PPPM
                arp = (N >= 512 and k == keys[0]) or N > DENSE_MAX_PPPM
                t0 = time.perf_counter()
                out = pppm_error(name, c, x, L, G, nref, dense, arp)
                out.update(cert_rel=c["cert_rel_max"], passes=bool(out["E_op"] <= TAU), t_s=time.perf_counter() - t0)
                res["rows"][k] = out
                C.log(f"verify {name} {k}: E_op {out['E_op']:.3e} ({out['E_op_method'][:5]})"
                      + (f" ARPACK {out['E_op_arpack']:.3e} ({out['arpack']['matvecs']} mv)" if "E_op_arpack" in out else "")
                      + f" cert {c['cert_rel_max']:.4e} ({out['t_s']:.0f} s)")
                C.write_json(f, dict(res, provenance=C.provenance()))
            ok = [r for r in rows if res["rows"].get(r["key"], {}).get("passes")]
            best = ok[0] if ok else None
            res["optimum"] = dict(best, E_op=res["rows"][best["key"]]["E_op"]) if best else None
            res["note"] = ("measured optimum = fastest confirmed (candidate, implementation) whose E_op was verified "
                           "<= tau; all candidates are certified (bound <= tau)")
            C.write_json(f, dict(res, provenance=C.provenance()))
            if best:
                C.log(f"verify {name}: optimum {best['key']} {best['impl']} {best['T_action_median_ms']:.2f} ms, "
                      f"E_op {res['optimum']['E_op']:.3e}")
            del G
            gc.collect()


# ============================================================================ direct accuracy and R_d
def stage_direct(args):
    for N in args.N:
        for fam in args.fam:
            name = name_of(fam, N)
            f = path("direct", name)
            if f.exists() and not args.force:
                continue
            x, L = load_cfg(name)
            ri = ref_info(name)
            lam, nref = ri["lam_lb"], ri["norm2"]
            t0 = time.perf_counter()
            b, lmax, tail, cnt = degree_bound_curve(x, L, CERT_EDGES)
            t_cert = time.perf_counter() - t0
            rel = b[:-1] / lam
            ok = np.nonzero(rel <= TAU)[0]
            k = int(ok[0])
            R_d = float(CERT_EDGES[k])
            rec = dict(config=name, N=N, L=L, R_d=R_d, bound_abs=float(b[k]), bound_rel_lam=float(rel[k]),
                       bound_curve=dict(R=CERT_EDGES[:-1].tolist(), rel_lam=rel.tolist()), R_far_cert=float(CERT_EDGES[-1]),
                       tail_beyond_Rfar_per_particle=tail, t_certify_s=t_cert, lam_lb=lam, norm2_ref=nref)
            C.log(f"direct {name}: certified R_d = {R_d} (bound/Lambda_lb {rel[k]:.4e}; R - 0.125 gives "
                  f"{rel[k - 1]:.4e}) tail(R_far) {tail:.2e} ({t_cert:.0f} s)")
            G, _ = gref(name, x, L)
            Dd = DirectPairs(x, L, R_d)
            Dr = DirectPairs(x, L, R_DIRECT_REF)
            rec.update(n_images=Dd.n_images, n_images_ref=Dr.n_images, n_pairs=int(len(Dd.pi)))
            n = 3 * N
            if N <= DENSE_MAX_DIRECT:
                t1 = time.perf_counter()
                Gd = tensors_to_dense(Dd.pi, Dd.pj, Dd.T, N)
                e, a = sym_norm2(Gd - G)
                rec.update(E_ref_dense=e / nref, E_ref_dense_asym=a / nref)
                Gr = tensors_to_dense(Dr.pi, Dr.pj, Dr.T, N)
                e2, a2 = sym_norm2(Gr - G)
                rec.update(E_refcheck_dense=e2 / nref)
                lamc = np.linalg.eigvalsh(Gr - Gd)
                rec.update(E_conv_dense=float(lamc[-1]) / nref, E_conv_dense_min_eig=float(lamc[0]) / nref,
                           t_dense_s=time.perf_counter() - t1)
                del Gd, Gr
            if N >= 1024:
                a1 = arpack_norm(lambda v: Dd.apply(v).ravel() - G @ v, n, which="LM", tol=1e-6)
                a2 = arpack_norm(lambda v: Dr.apply(v).ravel() - Dd.apply(v).ravel(), n, which="LA", tol=1e-6, seed=1)
                rec.update(E_ref_arpack=a1["value"] / nref, arpack_ref=a1, E_conv_arpack=a2["value"] / nref, arpack_conv=a2)
                if N > DENSE_MAX_DIRECT:
                    a3 = arpack_norm(lambda v: Dr.apply(v).ravel() - G @ v, n, which="LM", tol=1e-6, seed=2)
                    rec.update(E_refcheck_arpack=a3["value"] / nref, arpack_refcheck=a3)
            rec["E_ref"] = rec.get("E_ref_dense", rec.get("E_ref_arpack"))
            rec["E_ref_method"] = "dense" if "E_ref_dense" in rec else "ARPACK"
            if N <= 256:
                Di = DirectImages(x, L, R_d)
                Xs = x_inputs(N)[:3]
                rec["image_resolved_vs_pair_summed"] = max(float(np.linalg.norm(Di.apply(X) - Dd.apply(X))
                                                                 / np.linalg.norm(Dd.apply(X))) for X in Xs)
                rec["image_resolved_n_images"] = Di.n_images
                del Di
            rec["t_total_s"] = time.perf_counter() - t0
            C.write_json(f, dict(rec, provenance=C.provenance()))
            C.log(f"direct {name}: E_ref {rec['E_ref']:.3e} ({rec['E_ref_method']}), E_conv "
                  f"{rec.get('E_conv_dense', rec.get('E_conv_arpack')):.3e}, Gamma_D({R_DIRECT_REF}) vs ref "
                  f"{rec.get('E_refcheck_dense', rec.get('E_refcheck_arpack', float('nan'))):.2e}"
                  + (f", ARPACK E_ref {rec['E_ref_arpack']:.3e}" if "E_ref_arpack" in rec else "")
                  + f"; {Dd.n_images} images ({rec['t_total_s']:.0f} s)")
            del G, Dd, Dr
            gc.collect()


# ============================================================================ final scaling timing
def stage_scale(args):
    assert_one_thread()
    for N in args.N:
        for fam in args.fam:
            name = name_of(fam, N)
            ver = C.read_json(path("verify", name))
            dr = C.read_json(path("direct", name))
            conf = C.read_json(path("confirm", name))
            o = ver["optimum"]
            c = conf["candidates"][o["key"]]
            impl = o["impl"]
            x, L = load_cfg(name)
            Xs = x_inputs(N)
            R_d = dr["R_d"]
            info = dict(config=name, N=N, L=L, R_d=R_d, pppm=dict(p=c["p"], xi=c["xi"], rc=c["rc"], kc=c["kc_shell"],
                                                                     M=c["M"], eta=c["eta_actual"], impl=impl,
                                                                     n_img=c["n_img"], E_op=o["E_op"],
                                                                     cert_rel=c["cert_rel_max"]),
                        direct=dict(R_d=R_d, E_ref=dr["E_ref"], bound_rel_lam=dr["bound_rel_lam"], n_images=dr["n_images"],
                                    n_pairs=dr["n_pairs"]), blas=blas_info())
            tg, tm = [], []
            for _ in range(3):
                t0 = time.perf_counter()
                geom = DirectGeometry(N, L, R_d)
                tg.append(time.perf_counter() - t0)
                K, kc, mesh, t_m = build_mesh(L, c)
                tm.append(t_m)
            info["once_direct_geometry"] = st(np.array(tg))
            info["once_pppm_mesh"] = st(np.array(tm))
            setups = {"direct": lambda: DirectPairs(x, L, R_d, geom=geom),
                      "pppm_cached": lambda: CachedGammaH(x, L, K, c["rc"], mesh),
                      "pppm_pairdiff": lambda: pairdiff_setup(x, L, K, c["rc"], mesh)}
            Rs = rounds(N, R_SET)
            ts = {k: np.empty(Rs) for k in setups}
            rng = np.random.default_rng([SEED, 51, N])
            keys = list(setups)
            for k in keys:
                obj = setups[k]()
                del obj
            t0 = time.perf_counter()
            for r in range(Rs):
                for o_ in rng.permutation(len(keys)):
                    gc.collect()
                    t1 = time.perf_counter()
                    obj = setups[keys[o_]]()
                    ts[keys[o_]][r] = time.perf_counter() - t1
                    del obj
            C.log(f"scale {name}: {Rs} setup rounds ({time.perf_counter() - t0:.0f} s): "
                  + ", ".join(f"{k} {1e3 * np.median(v):.1f} ms" for k, v in ts.items()))
            D = DirectPairs(x, L, R_d, geom=geom)
            op = CachedGammaH(x, L, K, c["rc"], mesh)
            f = action_fns(op)
            fns = {"direct": D.apply, "pppm_cached": f["cached"], "pppm_pairdiff": f["pairdiff"]}
            G, _ = gref(name, x, L)
            nref = ref_info(name)["norm2"]
            acc = {}
            for k, fn in fns.items():
                errs = [float(np.linalg.norm(np.asarray(fn(X)).ravel() - G @ X.ravel()) / (nref * np.linalg.norm(X)))
                        for X in Xs]
                acc[k] = max(errs)
            del G
            gc.collect()
            Ra = rounds(N, R_ACT)
            t0 = time.perf_counter()
            ta = interleaved(fns, Ra, Xs, warm=3, seed=N + 102)
            C.log(f"scale {name}: {Ra} action rounds ({time.perf_counter() - t0:.0f} s): "
                  + ", ".join(f"{k} {1e3 * np.median(v):.3f} ms" for k, v in ta.items()))
            info.update(setups={k: st(v) for k, v in ts.items()}, actions={k: st(v) for k, v in ta.items()},
                        action_err_opnorm_10X=acc, rounds_setup=Rs, rounds_action=Ra)
            Tsd, Tad = info["setups"]["direct"]["median_ms"], info["actions"]["direct"]["median_ms"]
            Tsp, Tap = info["setups"][f"pppm_{impl}"]["median_ms"], info["actions"][f"pppm_{impl}"]["median_ms"]
            info["summary"] = dict(
                T_setup_direct_ms=Tsd, T_action_direct_ms=Tad, T_setup_pppm_ms=Tsp, T_action_pppm_ms=Tap,
                action_ratio_direct_over_pppm=Tad / Tap, setup_ratio_direct_over_pppm=Tsd / Tsp,
                total_direct_ms={q: Tsd + q * Tad for q in QS}, total_pppm_ms={q: Tsp + q * Tap for q in QS},
                total_ratio_direct_over_pppm={q: (Tsd + q * Tad) / (Tsp + q * Tap) for q in QS},
                break_even_q=((Tsp - Tsd) / (Tad - Tap)) if (Tad - Tap) != 0 and (Tsp - Tsd) / (Tad - Tap) > 0 else None,
                note="ratios > 1: PPPM faster; break_even_q: q at which T_setup + q T_action are equal (None: no "
                     "positive crossing)")
            info["provenance"] = C.provenance(blas=blas_info())
            C.write_json(path("scale", name), info)
            C.write_json(path("raw_scale", name), dict(unit="ms", setups={k: (1e3 * v).tolist() for k, v in ts.items()},
                                                       actions={k: (1e3 * v).tolist() for k, v in ta.items()}))
            s = info["summary"]
            C.log(f"scale {name}: action direct {Tad:.3f} / PPPM {Tap:.3f} ms (x{s['action_ratio_direct_over_pppm']:.2f}); "
                  f"setup direct {Tsd:.0f} / PPPM {Tsp:.1f} ms; break-even q {s['break_even_q']}")
            del D, op, fns, geom, mesh
            gc.collect()


# ============================================================================ memory
def traced(fn):
    gc.collect()
    tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    out = fn()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, peak - base, cur - base


def op_bytes(obj):
    return int(sum(v.nbytes for v in vars(obj).values() if isinstance(v, np.ndarray)))


def stage_memory(args):
    for N in args.N:
        for fam in args.fam:
            name = name_of(fam, N)
            sc = C.read_json(path("scale", name))
            conf = C.read_json(path("confirm", name))
            ver = C.read_json(path("verify", name))
            c = conf["candidates"][ver["optimum"]["key"]]
            x, L = load_cfg(name)
            X = x_inputs(N)[0]
            R_d = sc["R_d"]
            rec = dict(config=name, N=N)
            geom, gpeak, gkeep = traced(lambda: DirectGeometry(N, L, R_d))
            rec.update(direct_geometry_peak=gpeak, direct_geometry_bytes=int(geom.pairs[0].nbytes + geom.pairs[1].nbytes
                                                                             + geom.shifts.nbytes))
            D, dpeak, dkeep = traced(lambda: DirectPairs(x, L, R_d, geom=geom))
            _, apeak, _ = traced(lambda: D.apply(X))
            rec.update(direct_setup_peak=dpeak, direct_persistent=D.nbytes(), direct_action_peak=apeak)
            del D
            (K, kc, mesh, _), mpeak, _ = traced(lambda: build_mesh(L, c))
            rec.update(pppm_mesh_peak=mpeak, pppm_mesh_bytes=op_bytes(mesh))
            op, ppeak, pkeep = traced(lambda: CachedGammaH(x, L, K, c["rc"], mesh))
            _, capeak, _ = traced(lambda: op.apply(X))
            pdo, pdpeak, pdkeep = traced(lambda: pairdiff_setup(x, L, K, c["rc"], mesh))
            _, papeak, _ = traced(lambda: pdo.matvec(X.ravel()))
            rec.update(pppm_cached_setup_peak=ppeak, pppm_cached_persistent=op_bytes(op), pppm_cached_action_peak=capeak,
                       pppm_pairdiff_setup_peak=pdpeak, pppm_pairdiff_persistent=op_bytes(pdo),
                       pppm_pairdiff_action_peak=papeak)
            C.write_json(path("memory", name), dict(rec, unit="bytes", method="tracemalloc (NumPy allocations), "
                                                    "persistent = nbytes of the arrays held by the operator",
                                                    provenance=C.provenance()))
            C.log(f"memory {name}: direct persistent {rec['direct_persistent'] / 2**20:.1f} MiB, setup peak "
                  f"{dpeak / 2**20:.1f}, action peak {apeak / 2**20:.1f}; PPPM pairdiff persistent "
                  f"{rec['pppm_pairdiff_persistent'] / 2**20:.1f} MiB (+ mesh {rec['pppm_mesh_bytes'] / 2**20:.1f}), setup "
                  f"peak {pdpeak / 2**20:.1f}, action peak {papeak / 2**20:.1f}")
            del op, pdo, mesh, geom
            gc.collect()


STAGES = dict(configs=stage_configs, refs=stage_refs, calibrate=stage_calibrate, select=stage_select,
              bench=stage_bench, confirm=stage_confirm, verify=stage_verify, direct=stage_direct, scale=stage_scale,
              memory=stage_memory)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", required=True, choices=list(STAGES))
    ap.add_argument("--N", type=int, nargs="+", default=list(NS))
    ap.add_argument("--fam", nargs="+", default=list(FAMS))
    ap.add_argument("--p", type=int, nargs="+", default=list(P_SCALE))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    C.RESULTS = RESULTS
    RESULTS.mkdir(parents=True, exist_ok=True)
    C.log_command()
    for s in args.stage:
        t0 = time.perf_counter()
        STAGES[s](args)
        C.log(f"stage {s} done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
