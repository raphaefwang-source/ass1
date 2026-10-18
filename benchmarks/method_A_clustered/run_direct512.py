#!/usr/bin/env python3
"""
Stage 2 at N = 512: the direct periodic action of the physical kernel A vs the optimised PPPM of Stage 1 vs the
preassembled dense physical Gamma. Same configuration, box, density, kernel, full periodic convention (k = 0 retained,
no minimum image) and the same 10 deterministic input vectors for every method. Outputs: results_direct512/.

Stages
  converge  direct truncation radius by a convergence test (direct_periodic.annulus_tensors up to R_MAX):
            E_ref(R)  = ||Gamma_D(R) - Gamma_ref||_2 / ||Gamma_ref||_2   (Gamma_ref: independent LatticeFriction,
                        rel_tol 1e-13, raw/<config>_Gref.npy)
            E_conv(R) = ||Gamma_D(R) - Gamma_D(R_MAX)||_2 / ||Gamma_D(R_MAX)||_2   (self-convergence)
            bound(R)  = 2 [max_i lambda_max(discarded K) + analytic tail beyond R_MAX] / Lambda_lb   (certified)
            R_meas = smallest grid R with E_ref and E_conv <= tau there and at every larger grid R;
            R_cert = smallest grid R with bound <= tau (the analogue of the certified PPPM selection)
  bench     setups (interleaved, R_SET rounds) and actions (interleaved: R_FAST rounds for the fast actions,
            R_SLOW rounds including the image-resolved direct action) of
              direct_images  image-resolved direct action (Eq. 7), R_meas and R_cert
              direct_pairs   pair-summed direct action (Eq. Gamma with K^per_R tabulated per pair), R_meas and R_cert
              dense_phys     preassembled dense physical Gamma_D(R_meas) @ x (BLAS GEMV), assembled from the
                             direct_pairs tensors; a separate comparator, not the direct algorithm
              pppm           the measured-fastest accuracy-passing Stage 1 parameters, both action implementations
            accuracy of every action against Gamma_ref X on the same X, E_op of the assembled operators, identity of
            the direct variants, one-time costs (PPPM mesh/influence, Gamma_ref generation) separately, and the
            changing-position scenario T_setup + q T_action, q = 1, 10, 40, 80.

    python3 run_direct512.py --stage converge bench
"""
import argparse
import gc
import time

import common as C                       # first: sets OMP/OPENBLAS/MKL_NUM_THREADS = 1 before numpy loads OpenBLAS
import numpy as np

from clustered_configurations import load
from cached_operator import CachedGammaH, mesh_for
from parameter_optimizer import shell_kc, lambda_lower_bound
from error_validation import reference_cached, REF_REL_TOL, dense_independent
from direct_periodic import (DirectImages, DirectPairs, annulus_tensors, tensors_to_dense, degree_lambda_max,
                             tail_bound_per_particle)
from run_psearch_dense512 import (x_inputs, interleaved, st, pairdiff_setup, blas_info, assert_one_thread, path as ps_path,
                                  RESULTS_PSEARCH, TAU)

RESULTS_DIRECT = C.HERE / "results_direct512"
CONFIGS = ("strong_N512_s1", "uniform_N512_s1", "moderate_N512_s1")
EDGES = np.round(np.concatenate([np.arange(12.0, 32.0001, 0.25), np.arange(33.0, 42.0001, 1.0)]), 4)
R_SET, R_PPPM_SET, R_FAST, R_SLOW = 10, 30, 200, 30
QS = (1, 10, 40, 80)


def out_path(kind, name):
    return RESULTS_DIRECT / f"{kind}_{name}.json"


# ============================================================================ converge
def stage_converge(args):
    for name in args.configs:
        t0 = time.perf_counter()
        x, L = load(name)
        x = np.mod(x, L)
        N = len(x)
        Gref = reference_cached(name, x, L)
        nref = float(np.linalg.norm(Gref, 2))
        lam = lambda_lower_bound(Gref, N)
        pi, pj, acc, count = annulus_tensors(x, L, EDGES)
        C.log(f"converge {name}: annulus tensors up to R_max {EDGES[-1]} ({count.sum()} images, "
              f"{time.perf_counter() - t0:.0f} s)")
        np.cumsum(acc, axis=0, out=acc)                       # acc[k] = K^per_{R_k} per pair
        nimg = np.cumsum(count)
        G_max = tensors_to_dense(pi, pj, acc[-1], N)
        nmax = float(np.linalg.norm(G_max, 2))
        tail = tail_bound_per_particle(N, L, EDGES[-1])
        rows = []
        for k, R in enumerate(EDGES):
            G = tensors_to_dense(pi, pj, acc[k], N)
            disc = acc[-1] - acc[k]
            lmax = degree_lambda_max(pi, pj, disc, N) if k < len(EDGES) - 1 else 0.0
            bound_abs = 2 * (lmax + tail)
            rows.append(dict(R=float(R), n_img=int(nimg[k]), E_ref=float(np.linalg.norm(G - Gref, 2) / nref),
                             E_conv=float(np.linalg.norm(G - G_max, 2) / nmax), bound_abs=bound_abs,
                             bound_rel_lam=bound_abs / lam, bound_rel_norm=bound_abs / nref))
            if k % 8 == 0 or k == len(EDGES) - 1:
                r = rows[-1]
                C.log(f"  R={R:6.2f} n_img={r['n_img']:9d} E_ref={r['E_ref']:.3e} E_conv={r['E_conv']:.3e} "
                      f"bound/Lambda={r['bound_rel_lam']:.3e}")
        ok_meas = [max(r["E_ref"], r["E_conv"]) <= TAU for r in rows]
        k_meas = next(k for k in range(len(rows)) if all(ok_meas[k:]))
        k_cert = next(k for k in range(len(rows)) if rows[k]["bound_rel_lam"] <= TAU)
        out = dict(config=name, N=N, L=L, tau=TAU, norm2_ref=nref, lam_lb=lam, R_max=float(EDGES[-1]),
                   tail_beyond_Rmax_per_particle=tail, E_ref_at_Rmax=rows[-1]["E_ref"],
                   R_meas=rows[k_meas]["R"], R_cert=rows[k_cert]["R"], row_meas=rows[k_meas], row_cert=rows[k_cert],
                   rows=rows, wall_s=time.perf_counter() - t0, provenance=C.provenance())
        C.write_json(out_path("convergence", name), out)
        C.log(f"converge {name}: R_meas {out['R_meas']} (E_ref {rows[k_meas]['E_ref']:.3e}, E_conv "
              f"{rows[k_meas]['E_conv']:.3e}), R_cert {out['R_cert']} (bound {rows[k_cert]['bound_rel_lam']:.3e}); "
              f"Gamma_D(R_max) vs Gamma_ref {rows[-1]['E_ref']:.2e} ({out['wall_s']:.0f} s)")
        del acc, G_max
        gc.collect()


# ============================================================================ bench
def acc_metrics(fn, Xs, Gref, nref, N):
    rel_op, rel_act = [], []
    for X in Xs:
        y = np.asarray(fn(X)).ravel()
        r = Gref @ X.ravel()
        rel_op.append(float(np.linalg.norm(y - r) / (nref * np.linalg.norm(X))))
        rel_act.append(float(np.linalg.norm(y - r) / np.linalg.norm(r)))
    null = []
    for a in range(3):
        T = np.zeros((N, 3))
        T[:, a] = 1.0
        null.append(float(np.linalg.norm(np.asarray(fn(T)).ravel()) / (nref * np.linalg.norm(T))))
    return dict(max_err_opnorm=max(rel_op), max_err_rel_action=max(rel_act), translation_null_opnorm=max(null))


def stage_bench(args):
    assert_one_thread()
    for name in args.configs:
        conv = C.read_json(out_path("convergence", name))
        opt = C.read_json(ps_path("optimum", name, args.opt_tag))
        c, impl = opt["candidate"], opt["chosen_impl"]
        x, L = load(name)
        x = np.mod(x, L)
        N = len(x)
        Xs = x_inputs(N)
        Rm, Rc = conv["R_meas"], conv["R_cert"]
        xi, rc, M, p = c["xi"], c["rc"], c["M"], c["p"]
        kc = shell_kc(c["kc_shell"], L)
        K = C.kernel(xi)
        info = dict(config=name, N=N, L=L, tau=TAU, R_meas=Rm, R_cert=Rc,
                    pppm=dict(xi=xi, rc=rc, kc=kc, M=M, p=p, eta=kc * (L / M) / np.pi, impl=impl,
                              stage1_E_op=c["E_op"], stage1_cert_rel=c["cert_rel_max"]),
                    blas=blas_info())
        C.log(f"bench {name}: R_meas {Rm} R_cert {Rc}; PPPM p={p} xi={xi:.4f} rc={rc:.4f} kc={kc:.4f} M={M} ({impl})")
        # ---- one-time costs (not part of setup or action)
        tm = []
        for _ in range(5):
            t0 = time.perf_counter()
            mesh = mesh_for(L, K, kc, M, p)
            tm.append(time.perf_counter() - t0)
        info["once_pppm_mesh_influence"] = st(np.array(tm))
        tr = []
        for _ in range(3):
            t0 = time.perf_counter()
            Gnew, _ = C.reference(x, L, rel_tol=REF_REL_TOL)
            tr.append(time.perf_counter() - t0)
        Gref = reference_cached(name, x, L)
        info["once_reference_generation"] = st(np.array(tr))
        info["reference_regenerated_vs_raw_maxabs"] = float(np.abs(Gnew - Gref).max())
        del Gnew
        nref = float(np.linalg.norm(Gref, 2))
        # ---- setups, interleaved
        setups = {"direct_images_Rmeas": lambda: DirectImages(x, L, Rm),
                  "direct_images_Rcert": lambda: DirectImages(x, L, Rc),
                  "direct_pairs_Rmeas": lambda: DirectPairs(x, L, Rm),
                  "direct_pairs_Rcert": lambda: DirectPairs(x, L, Rc),
                  "dense_phys_Rmeas": lambda: DirectPairs(x, L, Rm).dense(),
                  "pppm_setup_cached": lambda: CachedGammaH(x, L, K, rc, mesh),
                  "pppm_setup_pairdiff": lambda: pairdiff_setup(x, L, K, rc, mesh)}
        ts = {k: np.empty(R_SET) for k in setups}
        rng = np.random.default_rng([20261018, 41, N])
        keys = list(setups)
        t0 = time.perf_counter()
        for r in range(R_SET):
            for o in rng.permutation(len(keys)):
                gc.collect()
                t1 = time.perf_counter()
                obj = setups[keys[o]]()
                ts[keys[o]][r] = time.perf_counter() - t1
                del obj
        C.log(f"bench {name}: {R_SET} interleaved setup rounds ({time.perf_counter() - t0:.0f} s): "
              + ", ".join(f"{k} {1e3 * np.median(v):.0f} ms" for k, v in ts.items()))
        tp = {k: np.empty(R_PPPM_SET) for k in ("pppm_setup_cached", "pppm_setup_pairdiff")}
        for r in range(R_PPPM_SET):
            for k in rng.permutation(list(tp)):
                gc.collect()
                t1 = time.perf_counter()
                obj = setups[k]()
                tp[k][r] = time.perf_counter() - t1
                del obj
        info["setups"] = {k: st(v) for k, v in ts.items()}
        info["setups_pppm_extra"] = {k: st(v) for k, v in tp.items()}
        raw = dict(unit="ms", setups={k: (1e3 * v).tolist() for k, v in ts.items()},
                   setups_pppm_extra={k: (1e3 * v).tolist() for k, v in tp.items()})
        # ---- operators
        D1m, D1c = DirectImages(x, L, Rm), DirectImages(x, L, Rc)
        D2m, D2c = DirectPairs(x, L, Rm), DirectPairs(x, L, Rc)
        Gd = D2m.dense()
        assert Gd.shape == (3 * N, 3 * N) and Gd.dtype == np.float64
        Gd = np.ascontiguousarray(Gd)
        op = CachedGammaH(x, L, K, rc, mesh)
        pd = pairdiff_setup(x, L, K, rc, mesh)
        info["n_images"] = dict(direct_images_Rmeas=D1m.n_images, direct_images_Rcert=D1c.n_images,
                                direct_pairs_Rmeas=D2m.n_images, direct_pairs_Rcert=D2c.n_images,
                                n_pairs=int(len(D2m.pi)), pppm_real_space=op.n_images,
                                direct_pairs_n_shifts=dict(Rmeas=D2m.n_shifts, Rcert=D2c.n_shifts))
        info["dense_phys"] = dict(shape=list(Gd.shape), dtype=str(Gd.dtype), MiB=Gd.nbytes / 2 ** 20)
        fns = {"direct_pairs_Rmeas": D2m.apply, "direct_pairs_Rcert": D2c.apply,
               "dense_phys_GEMV": lambda X: Gd @ X.ravel(), "dense_ref_GEMV": lambda X: Gref @ X.ravel(),
               "pppm_cached": op.apply, "pppm_pairdiff": lambda X: pd.matvec(X.ravel())}
        slow = dict(fns, direct_images_Rmeas=D1m.apply, direct_images_Rcert=D1c.apply)
        # ---- accuracy (same X) and identity
        accm = {k: acc_metrics(f, Xs, Gref, nref, N) for k, f in slow.items() if k != "dense_ref_GEMV"}
        T1 = D1m.pair_tensors()
        ident = dict(pair_tensors_images_vs_streamed_maxrel=float(np.abs(T1 - D2m.T).max() / np.abs(D2m.T).max()),
                     n_images_equal=bool(D1m.n_images == D2m.n_images))
        for k, f in (("direct_images_vs_pairs", D1m.apply), ("dense_phys_vs_pairs", lambda X: Gd @ X.ravel())):
            ident[k] = max(float(np.linalg.norm(np.asarray(f(X)).ravel() - D2m.apply(X).ravel())
                                 / np.linalg.norm(D2m.apply(X).ravel())) for X in Xs)
        del T1
        Gh = dense_independent(op)
        Gc = D2c.dense()
        Eop = dict(dense_phys_Rmeas=float(np.linalg.norm(Gd - Gref, 2) / nref),
                   dense_phys_Rcert=float(np.linalg.norm(Gc - Gref, 2) / nref),
                   pppm_Gamma_h=float(np.linalg.norm(Gh - Gref, 2) / nref))
        ident["pppm_matrix_free_vs_dense_Gamma_h"] = max(float(np.linalg.norm(op.apply(X).ravel() - Gh @ X.ravel())
                                                             / np.linalg.norm(Gh @ X.ravel())) for X in Xs)
        del Gh, Gc
        info.update(accuracy=accm, E_op=Eop, identity=ident, norm2_ref=nref)
        C.log(f"bench {name}: E_op dense_phys(R_meas) {Eop['dense_phys_Rmeas']:.3e} (R_cert) {Eop['dense_phys_Rcert']:.3e}"
              f" PPPM {Eop['pppm_Gamma_h']:.3e}; action errors " + ", ".join(f"{k} {v['max_err_opnorm']:.2e}"
                                                                            for k, v in accm.items())
              + f"; identity {ident}")
        # ---- actions, interleaved
        t0 = time.perf_counter()
        tf = interleaved(fns, R_FAST, Xs, warm=10, seed=N + 40)
        t1 = time.perf_counter()
        tsl = interleaved(slow, R_SLOW, Xs, warm=2, seed=N + 41)
        C.log(f"bench {name}: fast {R_FAST} rounds ({t1 - t0:.0f} s): "
              + ", ".join(f"{k} {1e3 * np.median(v):.3f} ms" for k, v in tf.items())
              + f"; slow {R_SLOW} rounds ({time.perf_counter() - t1:.0f} s): "
              + ", ".join(f"{k} {1e3 * np.median(v):.2f} ms" for k, v in tsl.items()))
        info["actions_fast"] = {k: st(v) for k, v in tf.items()}
        info["actions_slow"] = {k: st(v) for k, v in tsl.items()}
        raw["actions_fast"] = {k: (1e3 * v).tolist() for k, v in tf.items()}
        raw["actions_slow"] = {k: (1e3 * v).tolist() for k, v in tsl.items()}
        # ---- summary: production timings (fast session where available) and the changing-position scenario
        act = {k: info["actions_fast"][k]["median_ms"] for k in tf}
        act.update({k: info["actions_slow"][k]["median_ms"] for k in ("direct_images_Rmeas", "direct_images_Rcert")})
        setup = {k: info["setups"][k]["median_ms"] for k in ts}
        methods = {
            "direct_images_Rmeas": (setup["direct_images_Rmeas"], act["direct_images_Rmeas"]),
            "direct_images_Rcert": (setup["direct_images_Rcert"], act["direct_images_Rcert"]),
            "direct_pairs_Rmeas": (setup["direct_pairs_Rmeas"], act["direct_pairs_Rmeas"]),
            "direct_pairs_Rcert": (setup["direct_pairs_Rcert"], act["direct_pairs_Rcert"]),
            "dense_phys_Rmeas": (setup["dense_phys_Rmeas"], act["dense_phys_GEMV"]),
            "pppm": (info["setups"][f"pppm_setup_{impl}"]["median_ms"], act[f"pppm_{impl}"]),
        }
        Tsp, Tap = methods["pppm"]
        summ = {}
        for k, (Ts_, Ta_) in methods.items():
            qs = {q: Ts_ + q * Ta_ for q in QS}
            be = None
            if k != "pppm" and Ta_ != Tap:
                q0 = (Tsp - Ts_) / (Ta_ - Tap)
                be = float(q0) if q0 > 0 else None
            summ[k] = dict(T_setup_ms=Ts_, T_action_ms=Ta_, action_speedup_vs_pppm=Ta_ / Tap,
                           setup_ratio_vs_pppm=Ts_ / Tsp, total_ms=qs,
                           total_ratio_vs_pppm={q: qs[q] / (Tsp + q * Tap) for q in QS},
                           break_even_q_vs_pppm=be)
        info["summary"] = summ
        info["summary_note"] = ("action_speedup_vs_pppm = T_action(method)/T_action(PPPM): > 1 means PPPM is faster; "
                                "break_even_q_vs_pppm: number of actions per configuration at which "
                                "T_setup + q T_action of the method equals that of PPPM")
        info["provenance"] = C.provenance(blas=blas_info())
        C.write_json(out_path("stage2", name), info)
        C.write_json(out_path("stage2_raw", name), raw)
        C.log(f"bench {name}: " + "; ".join(f"{k}: setup {v['T_setup_ms']:.1f} ms, action {v['T_action_ms']:.3f} ms "
                                             f"(x{v['action_speedup_vs_pppm']:.3f} of PPPM)" for k, v in summ.items()))
        del D1m, D1c, D2m, D2c, Gd, op, pd, Gref
        gc.collect()


STAGES = dict(converge=stage_converge, bench=stage_bench)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", required=True, choices=list(STAGES))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--opt-tag", default="main")
    args = ap.parse_args()
    C.RESULTS = RESULTS_DIRECT
    RESULTS_DIRECT.mkdir(parents=True, exist_ok=True)
    C.log_command()
    for s in args.stage:
        t0 = time.perf_counter()
        STAGES[s](args)
        C.log(f"stage {s} done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
