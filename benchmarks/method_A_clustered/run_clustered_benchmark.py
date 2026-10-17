#!/usr/bin/env python3
"""
Theory-guided equal-accuracy benchmark of the cached Method A action Gamma_h X on clustered configurations.

Stages (each reads the previous stages' files in results/ and can be rerun on its own):
  refs       dense full-periodic Gamma_ref of every configuration (raw/, git-ignored), ||Gamma_ref||_2, Lambda_lb,
             lambda_min on Range(Pi), reference convergence (LatticeFriction rel_tol 1e-10 vs 1e-13) and self-check
             against a brute-force image sum
  calibrate  isolated timings -> C_R, C_P, C_F, C_0 of the action model and the setup model (results/cost_model.json)
  select     certified a_tol, b_tol, rc, kc, M (p = 7) and the exact N_img(rc) on the xi grid; family parameters
             (admissible for every seed of a clustering level); predicted T_action, T_setup; xi_opt per level and
             for T_setup + s T_action, s = 1, 10, 20 (bounded 1-D minimisation: grid, then a 0.005 refinement)
  bench      measured cached action at xi_opt and nearby xi (and a coarse xi scan for the two primary N = 256
             configurations), component breakdown, setup; the validated baseline set at the same state point
  errors     E_op and structure for every benchmarked set; basis-vector assembly and the measured components at xi_opt
  empirical  separate DIAGNOSTIC: smallest parameters whose MEASURED components meet the budget (N = 64, 256)

    python3 run_clustered_benchmark.py --stage refs calibrate select bench errors empirical
All timings: one process, OMP/OPENBLAS/MKL threads = 1, nothing else running, medians over >= 15 repetitions after
3 warm-up calls.
"""
import argparse
import time

import numpy as np

import common as C
from clustered_configurations import load, all_names
from cached_operator import CachedGammaH, mesh_for, baseline_mesh
from parameter_optimizer import (RealSpaceTail, FourierShells, select_for_xi, lambda_lower_bound, M_tol, CostModel,
                                 XI_GRID, A_MAX, R_PAD, SPLIT, eps_mesh, C_p, shell_kc)
from error_validation import (reference_cached, reference_summary, probes, validate_parameter_set, Components,
                              REF_REL_TOL)
import prl_toy_models as v2

FAMILIES = ("uniform", "moderate", "strong", "lj_equilibrated")
PRIMARY = ("strong_N256_s1", "uniform_N256_s1")
REUSE = (1, 10, 20)
EMP_XI = (0.5, 0.6, 0.7, 0.85, 1.0, 1.2)


def family_of(name):
    return name.split("_N")[0]


def N_of(name):
    return int(name.split("_N")[1].split("_")[0])


def groups(names):
    g = {}
    for n in names:
        g.setdefault((family_of(n), N_of(n)), []).append(n)
    return g


def names_for(args):
    return [n for n in all_names(Ns=tuple(args.N)) if N_of(n) in args.N]


# ============================================================================ refs
def reference_selfcheck(L, rng, n=48):
    lat = v2.LatticeFriction(L, kernel=C.LAW, gamma=C.GAMMA, kappa=C.KAPPA, r_ref=C.R_REF, rel_tol=REF_REL_TOL)
    dr = rng.uniform(-L / 2, L / 2, (n, 3))
    shifts = v2.lattice_shifts(lat.box, lat.r_tail + np.sqrt(3) * L / 2 + 5.0)
    brute = v2.image_sum_tensors(dr, shifts, kernel=C.LAW, gamma=C.GAMMA, kappa=C.KAPPA, r_ref=C.R_REF)
    fast = lat.pair_tensors(dr)
    return dict(n_displacements=n, n_shifts=int(len(shifts)), r_tail=lat.r_tail,
                max_abs_err_over_max=float(np.abs(fast - brute).max() / np.abs(brute).max()))


def stage_refs(args):
    out = C.read_json(C.RESULTS / "references.json", {"configs": {}, "checks": {}})
    rng = np.random.default_rng([20261017, 5])
    for name in names_for(args):
        x, L = load(name)
        N = len(x)
        t0 = time.perf_counter()
        G = reference_cached(name, x, L)
        rs, _ = reference_summary(G, N, C.lattice(L))
        rs.update(lam_lb=lambda_lower_bound(G, N), L=L, N=N, t_s=time.perf_counter() - t0)
        out["configs"][name] = rs
        C.log(f"ref {name}: ||G||={rs['norm2']:.4f} Lambda_lb={rs['lam_lb']:.4f} lam_min(Pi)={rs['lam_min_PiRange']:.4f} "
              f"({rs['t_s']:.1f} s)")
    for N in args.N:
        name = f"uniform_N{N}_s1"
        x, L = load(name)
        G = reference_cached(name, x, L)
        G10, _ = C.reference(x, L, rel_tol=1e-10)
        out["checks"][f"N{N}"] = dict(conv_rel_tol_1e10_vs_1e13=float(np.linalg.norm(G10 - G, 2) / np.linalg.norm(G, 2)),
                                      selfcheck=reference_selfcheck(L, rng))
        C.log(f"ref checks N={N}: {out['checks'][f'N{N}']}")
    out["rel_tol"] = REF_REL_TOL
    out["provenance"] = C.provenance()
    C.write_json(C.RESULTS / "references.json", out)


# ============================================================================ timing helpers
def reps_for(t_single, lo=15, hi=60, budget=1.5):
    return int(min(hi, max(lo, budget / max(t_single, 1e-6))))


def timed(fn, warm=3, budget=1.5):
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter() - t0
    return C.tstats(C.median_time(fn, reps_for(t1, budget=budget), warm=warm))


def x_vectors(N, k=4):
    rng = np.random.default_rng([20261017, 7, N])
    return [rng.standard_normal((N, 3)) for _ in range(k)]


def cycle(fn, Xs):
    st = {"k": 0}

    def g():
        st["k"] += 1
        return fn(Xs[st["k"] % len(Xs)])
    return g


def measure_point(x, L, xi, rc, kc, M, p, mesh=None, components=True, n_setup=3):
    N = len(x)
    K = C.kernel(xi)
    t0 = time.perf_counter()
    if mesh is None:
        mesh = mesh_for(L, K, kc, M, p)
    t_param = time.perf_counter() - t0
    setups = []
    for _ in range(n_setup):
        op = CachedGammaH(x, L, K, rc, mesh)
        setups.append((op.t_setup, dict(op.t_setup_parts)))
    ts = sorted(setups, key=lambda s: s[0])[len(setups) // 2]
    Xs = x_vectors(N)
    act = timed(cycle(op.apply, Xs))
    row = dict(N=N, L=L, xi=xi, rc=rc, kc=kc, M=mesh.M, p=p, eta=mesh.eta_actual, n_modes=mesh.n_modes,
               n_excluded=mesh.n_excluded, n_img=op.n_images, n_unique=op.images.unique_pairs(),
               t_param_s=t_param, t_setup_s=ts[0], **{f"t_setup_{k}_s": v for k, v in ts[1].items()},
               t_action_s=act["median"], t_action_q1_s=act["q1"], t_action_q3_s=act["q3"], t_action_min_s=act["min"],
               t_action_max_s=act["max"], t_action_n=act["n"], t_action_iqr_rel=act["iqr_rel"])
    if components:
        X = Xs[0]
        grids = op.spread(X)
        phi = op.fft_influence(grids)
        old = op.existing_gammah()
        for nm, fn in (("real", cycle(op.real_H, Xs)), ("spread", cycle(op.spread, Xs)),
                       ("fft_influence", lambda: op.fft_influence(grids)), ("gather", lambda: op.gather(phi)),
                       ("degree", cycle(op.degree_term, Xs)),
                       ("existing_pairdiff_matvec", cycle(lambda X_: old.matvec(X_.ravel()), Xs))):
            row[f"t_{nm}_s"] = timed(fn, budget=0.6)["median"]
    return row, op


# ============================================================================ calibrate
def build3(x, L, K, rc, mesh):
    ops = [CachedGammaH(x, L, K, rc, mesh) for _ in range(3)]
    return ops[0], float(np.median([o.t_setup for o in ops]))


def stage_calibrate(args):
    rows = []
    K = C.kernel(0.85)
    for N in args.N:
        name = f"uniform_N{N}_s1"
        x, L = load(name)
        X = x_vectors(N)
        small = mesh_for(L, K, 2 * np.pi * 2 / L, 16, 4)
        for rc in (2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0):
            if N / 2 * C.DENSITY * 4 / 3 * np.pi * rc ** 3 > 1.3e6:
                continue
            op, tsu = build3(x, L, K, rc, small)
            t = timed(cycle(op.real_H, X))
            rows.append(dict(kind="real", N=N, rc=rc, n_img=op.n_images, p=4, M=16, t=t["median"], t_iqr_rel=t["iqr_rel"],
                             t_setup=tsu))
        for p in (4, 5, 6, 7, 8):
            mesh = mesh_for(L, K, 2 * np.pi * 6 / L, 32, p)
            op, tsu = build3(x, L, K, 1.5, mesh)
            g = op.spread(X[0])
            ph = op.fft_influence(g)
            ts, tg = timed(cycle(op.spread, X)), timed(lambda: op.gather(ph))
            rows.append(dict(kind="spread_gather", N=N, p=p, M=32, n_img=op.n_images, rc=1.5, t=ts["median"] + tg["median"],
                             t_spread=ts["median"], t_gather=tg["median"], t_setup=tsu))
        for M in ((16, 24, 32, 40, 48, 60, 64, 80, 96, 120) if N == 256 else (24, 48, 96)):
            mesh = mesh_for(L, K, 2 * np.pi * max(1, M // 4) / L, M, 4)
            op, tsu = build3(x, L, K, 1.5, mesh)
            g = op.spread(X[0])
            t = timed(lambda: op.fft_influence(g))
            rows.append(dict(kind="fft", N=N, M=M, p=4, rc=1.5, n_img=op.n_images, t=t["median"], t_setup=tsu))
        for rc, M, p in ((3.5, 24, 7), (5.0, 40, 7), (7.0, 60, 7), (4.5, 32, 5)):
            mesh = mesh_for(L, K, 0.5 * np.pi * M / L, M, p)
            op, tsu = build3(x, L, K, rc, mesh)
            t = timed(cycle(op.apply, X))
            rows.append(dict(kind="full", N=N, rc=rc, M=M, p=p, n_img=op.n_images, t=t["median"], t_iqr_rel=t["iqr_rel"],
                             t_setup=tsu))
        C.log(f"calibrate N={N}: {len(rows)} timings so far")
    feats = lambda r: (r["n_img"], r["N"] * r["p"] ** 3, r["M"] ** 3 * np.log(r["M"] ** 3))   # noqa: E731
    from scipy.optimize import nnls

    def fit(kind, col):
        """t = slope * feature + intercept, both >= 0, least squares in RELATIVE error."""
        sel = [r for r in rows if r["kind"] == kind]
        f = np.array([feats(r)[col] for r in sel], float)
        y = np.array([r["t"] for r in sel])
        A = np.stack([f, np.ones_like(f)], 1) / y[:, None]
        coef, _ = nnls(A, np.ones_like(y))
        pred = coef[0] * f + coef[1]
        return float(coef[0]), float(coef[1]), float(np.max(np.abs(pred - y) / y))

    C_R, b_R, e_R = fit("real", 0)
    C_P, b_P, e_P = fit("spread_gather", 1)
    C_F, b_F, e_F = fit("fft", 2)
    full = [r for r in rows if r["kind"] == "full"]
    comp = lambda r: C_R * r["n_img"] + C_P * r["N"] * r["p"] ** 3 + C_F * r["M"] ** 3 * np.log(r["M"] ** 3)  # noqa: E731
    # C_0 = component intercepts + the remaining glue (degree term D_i X_i, array allocation, Python calls), the latter
    # as the median relative residual of the full-action calibration points
    glue = float(np.median([(r["t"] - comp(r) - b_R - b_P - b_F) for r in full]))
    C_0 = b_R + b_P + b_F + glue
    cm = CostModel(C_R, C_P, C_F, C_0, intercepts=dict(real=b_R, spread_gather=b_P, fft=b_F, glue=glue),
                   max_rel_fit_err=dict(real=e_R, spread_gather=e_P, fft=e_F),
                   full_rel_err=[float((comp(r) + C_0) / r["t"] - 1) for r in full])
    # setup model: T_setup = S_R n_img + S_P N p^3 + S_F M^3 log M^3 + S_0 (non-negative, relative least squares over
    # every calibration build; each t_setup is the median of 3 builds)
    A = np.array([[*feats(r), 1.0] for r in rows], float)
    y = np.array([r["t_setup"] for r in rows])
    s_coef, _ = nnls(A / y[:, None], np.ones_like(y))
    setup = dict(S_R=float(s_coef[0]), S_P=float(s_coef[1]), S_F=float(s_coef[2]), S_0=float(s_coef[3]),
                 max_rel_err=float(np.max(np.abs(A @ s_coef - y) / y)),
                 median_rel_err=float(np.median(np.abs(A @ s_coef - y) / y)))
    for r in full:
        r["t_pred"] = cm.predict(r["n_img"], r["N"], r["p"], r["M"])
    C.write_csv(C.RESULTS / "calibration_timings.csv", rows)
    C.write_json(C.RESULTS / "cost_model.json", dict(action=cm.as_dict(), setup=setup, provenance=C.provenance(),
                                                     note="coefficients in seconds; one cached action, 1 thread"))
    C.log(f"cost model: C_R={C_R:.3e} C_P={C_P:.3e} C_F={C_F:.3e} C_0={C_0:.3e}; fit errs real {e_R:.2f} sg {e_P:.2f} "
          f"fft {e_F:.2f}; full {cm.extra['full_rel_err']}; setup {setup}")


def cost_models():
    d = C.read_json(C.RESULTS / "cost_model.json")
    return CostModel.from_dict(d["action"]), d["setup"]


def setup_pred(sm, n_img, N, p, M):
    return sm["S_R"] * n_img + sm["S_P"] * N * p ** 3 + sm["S_F"] * M ** 3 * np.log(M ** 3) + sm["S_0"]


# ============================================================================ select
def family_params(xi, seeds, L, N, p=C.P_ORDER, eps=C.EPS_TARGET):
    """Certified parameters admissible for every configuration of the group (largest rc, kc; M re-solved at that kc)."""
    rows = [select_for_xi(xi, s["rst"], L, N, s["lam_lb"], fs=s["fs"], prep=s["rst"].prepare(xi)) for s in seeds]
    if not all(r["feasible"] for r in rows):
        return None, rows
    rc = max(r["rc"] for r in rows)
    kc = max(r["kc"] for r in rows)
    fs = seeds[0]["fs"]
    Ms = [M_tol(fs, kc, p, SPLIT["mesh"] * eps * s["lam_lb"], N / L ** 3) for s in seeds]
    if any(M is None for M in Ms):
        return None, rows
    M = max(Ms)
    n_img = [s["rst"].n_img(rc) for s in seeds]
    n_uni = [s["rst"].n_unique(rc) for s in seeds]
    return dict(xi=float(xi), rc=rc, kc=kc, M=M, p=p, a_tol=xi * rc, b_tol=kc / (2 * xi),
                eta=kc * (L / M) / np.pi, n_img_mean=float(np.mean(n_img)), n_img_max=int(max(n_img)),
                n_unique_mean=float(np.mean(n_uni)), images_per_pair=float(np.mean(n_img) / np.mean(n_uni)),
                n_img_per_seed=n_img), rows


def stage_select(args):
    cm, sm = cost_models()
    refs = C.read_json(C.RESULTS / "references.json")["configs"]
    grid_rows, fam_rows, summary = [], [], {}
    for (fam, N), names in sorted(groups(names_for(args)).items(), key=lambda kv: (kv[0][1], kv[0][0])):
        t0 = time.perf_counter()
        seeds = []
        for name in names:
            x, L = load(name)
            seeds.append(dict(name=name, rst=RealSpaceTail(x, L, (A_MAX + R_PAD) / XI_GRID.min()),
                              lam_lb=refs[name]["lam_lb"], norm2=refs[name]["norm2"]))
        fsc = {}

        def evaluate(xi):
            fs = fsc.setdefault(round(float(xi), 6), FourierShells(xi, L))
            for s in seeds:
                s["fs"] = fs
            fp, rows = family_params(xi, seeds, L, N)
            for s, r in zip(seeds, rows):
                r.update(config=s["name"], family=fam, N=N, norm2_ref=s["norm2"])
                if r["feasible"]:
                    r["T_pred_own"] = cm.predict(r["n_img"], N, r["p"], r["M"])
            if fp is None:
                return None, rows
            parts = cm.parts(fp["n_img_mean"], N, fp["p"], fp["M"])
            fp.update(family=fam, N=N, L=L, T_pred=sum(parts.values()), **{f"T_pred_{k}": v for k, v in parts.items()},
                      T_setup_pred=setup_pred(sm, fp["n_img_mean"], N, fp["p"], fp["M"]))
            for s_ in REUSE:
                fp[f"J_pred_s{s_}"] = fp["T_setup_pred"] + s_ * fp["T_pred"]
            return fp, rows

        fps = []
        for xi in XI_GRID:
            fp, rows = evaluate(xi)
            grid_rows += rows
            if fp is not None:
                fps.append(fp)
        fam_rows += [dict(f, grid="coarse") for f in fps]
        out = {}
        for obj in ["T_pred"] + [f"J_pred_s{s_}" for s_ in REUSE]:
            best = min(fps, key=lambda f: f[obj])
            fine = [evaluate(xi)[0] for xi in np.round(np.arange(best["xi"] - 0.025, best["xi"] + 0.02501, 0.005), 4)
                    if XI_GRID.min() <= xi <= XI_GRID.max()]
            fine = [f for f in fine if f is not None]
            fam_rows += [dict(f, grid=f"fine_{obj}") for f in fine]
            opt = min(fine + [best], key=lambda f: f[obj])
            out[obj] = {k: opt[k] for k in ("xi", "rc", "kc", "M", "p", "a_tol", "b_tol", "eta", "n_img_mean",
                                            "n_unique_mean", "images_per_pair", "T_pred", "T_setup_pred", obj)}
        summary[f"{fam}_N{N}"] = dict(family=fam, N=N, L=L, configs=names, opt=out)
        C.log(f"select {fam} N={N}: xi_opt(action)={out['T_pred']['xi']:.3f} M={out['T_pred']['M']} "
              f"rc={out['T_pred']['rc']:.3f} T_pred={1e3 * out['T_pred']['T_pred']:.2f} ms; "
              + ", ".join(f"s={s_}: {out[f'J_pred_s{s_}']['xi']:.3f}" for s_ in REUSE)
              + f"  ({time.perf_counter() - t0:.0f} s)")
        del seeds
    C.write_csv(C.RESULTS / "selection_grid.csv", grid_rows)
    C.write_csv(C.RESULTS / "selection_family.csv", [{k: v for k, v in r.items() if k != "n_img_per_seed"} for r in fam_rows])
    C.write_json(C.RESULTS / "selection_summary.json", dict(groups=summary, split=SPLIT, eps=C.EPS_TARGET, p=C.P_ORDER,
                                                            C_p=C_p(C.P_ORDER), xi_grid=XI_GRID, provenance=C.provenance()))


# ============================================================================ bench
def family_lookup():
    fam = {}
    with open(C.RESULTS / "selection_family.csv") as fh:
        import csv
        for r in csv.DictReader(fh):
            key = (r["family"], int(r["N"]), round(float(r["xi"]), 4))
            fam[key] = dict(xi=float(r["xi"]), rc=float(r["rc"]), kc=shell_kc(float(r["kc"]), float(r["L"])),
                            M=int(r["M"]), p=int(r["p"]),
                            T_pred=float(r["T_pred"]), T_setup_pred=float(r["T_setup_pred"]),
                            **{k: float(r[k]) for k in r if k.startswith("T_pred_") or k.startswith("J_pred_")})
    return fam


def bench_xis(xi_opt, N, name):
    g = XI_GRID
    snap = lambda v: float(g[np.argmin(np.abs(g - v))])            # noqa: E731
    if N == 64:
        offs = ()
    elif N == 512:
        offs = (-0.2, -0.1, 0.1, 0.2)
    else:
        offs = (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2)
    xs = {round(xi_opt, 4)} | {snap(xi_opt + o) for o in offs}
    if name in PRIMARY:
        xs |= {round(float(v), 4) for v in np.arange(0.4, 1.6001, 0.1)}
    return sorted(xs)


def stage_bench(args):
    cm, sm = cost_models()
    summ = C.read_json(C.RESULTS / "selection_summary.json")["groups"]
    fam = family_lookup()
    rows = C.read_json(C.RESULTS / "bench.json", {"rows": []})["rows"] if args.resume else []
    done = {(r["config"], r["set"], round(r["xi"], 4)) for r in rows}
    for name in names_for(args):
        x, L = load(name)
        N = len(x)
        g = summ[f"{family_of(name)}_N{N}"]["opt"]
        xi_opt = round(g["T_pred"]["xi"], 4)
        todo = [("certified", xi) for xi in bench_xis(xi_opt, N, name)]
        todo += [("certified_reuse", round(g[f"J_pred_s{s_}"]["xi"], 4)) for s_ in REUSE]
        todo.append(("baseline", C.BASELINE["xi"]))
        seen = set()
        for kind, xi in todo:
            if (kind.replace("_reuse", ""), xi) in seen or (name, kind.replace("_reuse", ""), xi) in done:
                continue
            seen.add((kind.replace("_reuse", ""), xi))
            if kind == "baseline":
                b = C.BASELINE
                K = C.kernel(b["xi"])
                rc, kc = b["s"] / b["xi"], 2 * b["xi"] * b["s"]
                t0 = time.perf_counter()
                mesh = baseline_mesh(L, K, kc, b["eta"], b["p"])
                tp = time.perf_counter() - t0
                row, _ = measure_point(x, L, b["xi"], rc, kc, mesh.M, b["p"], mesh=mesh)
                row["t_param_s"] = tp
                pred = cm.predict(row["n_img"], N, b["p"], mesh.M)
                st = "baseline"
            else:
                fp = fam.get((family_of(name), N, xi))
                if fp is None:
                    fp = next(v for k, v in fam.items() if k[0] == family_of(name) and k[1] == N and abs(k[2] - xi) < 1e-9)
                row, op = measure_point(x, L, xi, fp["rc"], fp["kc"], fp["M"], fp["p"])
                assert op.mesh.mc == int(np.floor(fp["kc"] * L / (2 * np.pi) + 1e-12)) and \
                    np.isclose(np.max(np.linalg.norm(op.mesh.kv, axis=1)), fp["kc"], rtol=1e-12), "outer shell lost"
                pred = cm.predict(row["n_img"], N, fp["p"], fp["M"])
                st = "certified"
            parts = cm.parts(row["n_img"], N, row["p"], row["M"])
            row.update(config=name, family=family_of(name), set=st, xi_opt_pred=xi_opt, is_xi_opt=bool(abs(xi - xi_opt) < 1e-9),
                       T_pred_s=pred, **{f"T_pred_{k}_s": v for k, v in parts.items()},
                       T_setup_pred_s=setup_pred(sm, row["n_img"], N, row["p"], row["M"]))
            rows.append(row)
            C.log(f"bench {name} {st} xi={xi:.3f} M={row['M']} rc={row['rc']:.3f} n_img={row['n_img']}: "
                  f"action {1e3 * row['t_action_s']:.2f} ms (pred {1e3 * pred:.2f}, IQR {100 * row['t_action_iqr_rel']:.0f}%), "
                  f"setup {1e3 * row['t_setup_s']:.1f} ms")
            C.write_json(C.RESULTS / "bench.json", dict(rows=rows, provenance=C.provenance()))
    C.write_json(C.RESULTS / "bench.json", dict(rows=rows, provenance=C.provenance()))


# ============================================================================ errors
def stage_errors(args):
    bench = C.read_json(C.RESULTS / "bench.json")["rows"]
    summ = C.read_json(C.RESULTS / "selection_summary.json")["groups"]
    sel_rows = {}
    import csv
    with open(C.RESULTS / "selection_grid.csv") as fh:
        for r in csv.DictReader(fh):
            sel_rows[(r["config"], round(float(r["xi"]), 4))] = r
    out = C.read_json(C.RESULTS / "errors.json", {"rows": [], "probes": []}) if args.resume else {"rows": [], "probes": []}
    done = {(r["config"], r["set"], round(r["xi"], 4)) for r in out["rows"]}
    for name in names_for(args):
        x, L = load(name)
        x = np.mod(x, L)
        N = len(x)
        G = reference_cached(name, x, L)
        rs, Q = reference_summary(G, N, C.lattice(L))
        P = probes(x, L, N, G, Q, np.random.default_rng([20261017, 11, N, sum(map(ord, name))]))
        xi_opt = round(summ[f"{family_of(name)}_N{N}"]["opt"]["T_pred"]["xi"], 4)
        sets = {}
        for r in bench:
            if r["config"] == name:
                sets[(r["set"], round(r["xi"], 4))] = r
        for (st, xi), r in sorted(sets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if (name, st, xi) in done:
                continue
            t0 = time.perf_counter()
            basis = st == "certified" and xi == xi_opt and (N <= 256 or name == "strong_N512_s1")
            mesh_kw = dict(xi=r["xi"], rc=r["rc"], kc=r["kc"], M=r["M"], p=r["p"])
            rec, act = validate_parameter_set(name, x, L, G, rs, Q, P, basis=basis, **mesh_kw)
            rec.update(config=name, family=family_of(name), N=N, set=st, is_xi_opt=bool(st == "certified" and xi == xi_opt),
                       **mesh_kw, norm2_ref=rs["norm2"], eps_target=C.EPS_TARGET, passes=bool(rec["E_op"] <= C.EPS_TARGET))
            if st == "certified" and xi == xi_opt and N <= 256:
                cp = Components(x, L, G, r["xi"])
                rec.update(cp.measured(r["rc"], r["kc"], r["M"], r["p"]))
                # certified pieces evaluated at the benchmarked (family) parameters for this configuration
                from parameter_optimizer import FourierShells as FS, eps_mesh as em
                fs = FS(r["xi"], L)
                prep = cp.rst.prepare(r["xi"])
                lam = lambda_lower_bound(G, N)
                rec.update(cert_S_abs=cp.rst.eps_S(prep, r["rc"]), cert_F_abs=fs.eps_F(r["kc"], N / L ** 3),
                           cert_mesh_abs=em(fs, r["kc"], r["M"], r["p"], N / L ** 3), lam_lb=lam)
                own = sel_rows.get((name, xi))
                if own:
                    rec["eps_quad_abs_est"] = float(own["eps_quad_abs_est"])
            out["rows"].append(rec)
            out["probes"] += [dict(config=name, set=st, xi=r["xi"], probe=k, **v) for k, v in act.items()]
            C.log(f"errors {name} {st} xi={r['xi']:.3f} M={r['M']}: E_op={rec['E_op']:.3e} (k!=0: {rec['E_op_k0']:.2e}) "
                  f"lam_min {rec['lam_min_h']:.4f}/{rec['lam_min_ref']:.4f} sym {rec['sym_resid']:.1e} null {rec['null_resid']:.1e}"
                  + (f" basis {rec['basis_vs_independent']:.1e}" if basis else "")
                  + (f" meas R/F/M {rec['meas_real_rel']:.2e}/{rec['meas_fourier_rel']:.2e}/{rec['meas_mesh_rel']:.2e}"
                     if "meas_real_rel" in rec else "") + f" ({time.perf_counter() - t0:.1f} s)")
            C.write_json(C.RESULTS / "errors.json", dict(out, provenance=C.provenance()))
    C.write_json(C.RESULTS / "errors.json", dict(out, provenance=C.provenance()))


# ============================================================================ empirical (diagnostic)
def stage_empirical(args):
    cm, sm = cost_models()
    import csv
    sel_rows = {}
    with open(C.RESULTS / "selection_grid.csv") as fh:
        for r in csv.DictReader(fh):
            sel_rows[(r["config"], round(float(r["xi"]), 4))] = r
    res = C.read_json(C.RESULTS / "empirical.json", {"seeds": [], "family": [], "timing": [], "check": []})
    done_f = {(r["family"], r["N"], round(r["xi"], 4)) for r in res["family"]}
    for (fam, N), names in sorted(groups([n for n in names_for(args) if N_of(n) in (64, 256)]).items(),
                                  key=lambda kv: (kv[0][1], kv[0][0])):
        data = []
        for name in names:
            x, L = load(name)
            x = np.mod(x, L)
            G = reference_cached(name, x, L)
            data.append((name, x, L, G, RealSpaceTail(x, L, (A_MAX + R_PAD) / min(EMP_XI))))
        for xi in EMP_XI:
            if (fam, N, round(xi, 4)) in done_f:
                continue
            t0 = time.perf_counter()
            per = []
            for name, x, L, G, rst in data:
                own = sel_rows[(name, round(xi, 4))]
                cp = Components(x, L, G, xi, rst)
                e = cp.empirical(C.EPS_TARGET, C.P_ORDER, float(own["rc"]), shell_kc(float(own["kc"]), L))
                e.update(config=name, family=fam, N=N, rc_cert=float(own["rc"]), kc_cert=shell_kc(float(own["kc"]), L),
                         M_cert=int(own["M"]))
                per.append(e)
                res["seeds"].append({k: v for k, v in e.items()})
            rc, kc, M = max(e["rc_emp"] for e in per), max(e["kc_emp"] for e in per), max(e["M_emp"] for e in per)
            n_img = [rst.n_img(rc) for *_, rst in data]
            f = dict(family=fam, N=N, xi=xi, rc=rc, kc=kc, M=M, p=C.P_ORDER, a_emp=xi * rc, b_emp=kc / (2 * xi),
                     n_img_mean=float(np.mean(n_img)), T_pred=cm.predict(np.mean(n_img), N, C.P_ORDER, M),
                     T_setup_pred=setup_pred(sm, np.mean(n_img), N, C.P_ORDER, M))
            for s_ in REUSE:
                f[f"J_pred_s{s_}"] = f["T_setup_pred"] + s_ * f["T_pred"]
            # measured total E_op of the family empirical set on every configuration of the group
            for name, x, L, G, rst in data:
                rs, Q = reference_summary(G, len(x), C.lattice(L))
                P = probes(x, L, len(x), G, Q, np.random.default_rng([20261017, 11, len(x), sum(map(ord, name))]))
                rec, _ = validate_parameter_set(name, x, L, G, rs, Q, P, xi, rc, kc, M, C.P_ORDER)
                res["check"].append(dict(config=name, family=fam, N=N, xi=xi, rc=rc, kc=kc, M=M, E_op=rec["E_op"],
                                         E_op_k0=rec["E_op_k0"], lam_min_h=rec["lam_min_h"], passes=rec["E_op"] <= C.EPS_TARGET))
            res["family"].append(f)
            C.log(f"empirical {fam} N={N} xi={xi}: rc {rc:.3f} kc {kc:.3f} M {M} (certified M "
                  f"{max(e['M_cert'] for e in per)}), T_pred {1e3 * f['T_pred']:.2f} ms, E_op max "
                  f"{max(c['E_op'] for c in res['check'] if c['family'] == fam and c['N'] == N and c['xi'] == xi):.2e} "
                  f"({time.perf_counter() - t0:.0f} s)")
            C.write_json(C.RESULTS / "empirical.json", dict(res, provenance=C.provenance()))
    # measured timing at the empirical family sets (primary N = 256 configurations)
    if not args.no_emp_timing:
        res["timing"] = []
        for name in names_for(args):
            if N_of(name) != 256 or not name.endswith("_s1"):
                continue
            x, L = load(name)
            for f in res["family"]:
                if f["family"] == family_of(name) and f["N"] == 256:
                    row, _ = measure_point(x, L, f["xi"], f["rc"], f["kc"], f["M"], f["p"], components=False)
                    row.update(config=name, family=f["family"], set="empirical", T_pred_s=cm.predict(row["n_img"], 256, f["p"], f["M"]))
                    res["timing"].append(row)
                    C.log(f"empirical timing {name} xi={f['xi']}: action {1e3 * row['t_action_s']:.2f} ms "
                          f"(pred {1e3 * row['T_pred_s']:.2f})")
    C.write_json(C.RESULTS / "empirical.json", dict(res, provenance=C.provenance()))


STAGES = dict(refs=stage_refs, calibrate=stage_calibrate, select=stage_select, bench=stage_bench, errors=stage_errors,
              empirical=stage_empirical)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", required=True, choices=list(STAGES))
    ap.add_argument("--N", type=int, nargs="+", default=[64, 256, 512])
    ap.add_argument("--resume", action="store_true", help="keep finished rows of bench/errors")
    ap.add_argument("--no-emp-timing", action="store_true")
    args = ap.parse_args()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    C.log_command()
    for st in args.stage:
        t0 = time.perf_counter()
        STAGES[st](args)
        C.log(f"stage {st} done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
