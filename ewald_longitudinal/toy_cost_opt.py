#!/usr/bin/env python3
"""
Error-constrained cost optimisation of the PPPM + Lanczos friction thermostat (toy application, laws A and B).

Model kept fixed: pure longitudinal K = g(r) P, full periodic images, k = 0 retained, g(r_ref) = gamma = 0.5,
kappa = 0.7, r_ref = 1.3, kT = 0.7, m = 1, dt = 0.005, density 64/5.5^3. Baseline ("high accuracy", kept as the
reference version): PPPM xi = 0.7, s = 4.10, eta = 0.7, p = 7; Lanczos noise rank 40, damping rank 16.

Stages (each writes JSON/CSV to cost_optimization_results/ and can be rerun independently):
  profile   time breakdown of one full step (forces, operator construction, real-space matvec, mesh spread/gather,
            FFT, influence multiply, Lanczos reorthogonalisation / small eigenproblem / other)
  search    staged operator search per law and error budget: (1) cutoff scale s per xi with a fine mesh,
            (2) coarsest mesh per assignment order p = 5, 6, 7; error = relative spectral error against the dense
            full-periodic reference (v2 LatticeFriction); timing of each accepted candidate
  ranks     noise sqrt-action and damping-action errors vs Lanczos rank, against dense f(Gamma_h) (Krylov part) and
            dense f(Gamma_ref) (total), separately from Gamma v
  refine    mesh rescan at a large box (N = 512, matrix-free spectral error) for the cheapest (xi, s) per budget
  modes     rank requirement per N: overall, k = 2 pi / L plane-wave and lowest-eigenmode Lanczos errors
  slowmodes final static checks of the selected sets (N = 64, 512) + slow-mode errors at their ranks
  ranks_large  large-N rank errors (converged-Lanczos reference) and sampled-row Gamma v error vs direct sum
  validate  short fixed-time dynamics, candidate vs baseline on common initial states and common noise
  timing    dense vs direct full-periodic sum + Lanczos vs PPPM + Lanczos at several N (one process per case)
  table     final table (cost_table.csv, timing_table.md); report: cost_optimization_results/REPORT.md
Error budgets: "5e-8" (law B only; ~ B baseline error), "2e-7" (~ A baseline error), "1e-6" and "1e-4" (relative spectral operator
error against the full-periodic reference, and relative Lanczos action errors).
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import prl_toy_models as v2  # noqa: E402
import radial_kernels as rk  # noqa: E402
import toy_dynamics as td  # noqa: E402
from test_lanczos_fdt import DenseRef, lanczos, tridiag, proj  # noqa: E402
from test_production_timing_rank40_16 import timed_apply  # noqa: E402
from test_equal_accuracy_xi import next_smooth  # noqa: E402

OUT = HERE / "cost_optimization_results"
CKPT = HERE / "toy_models" / "v2_results" / "restart_checkpoints"
TOY = dict(gamma=0.5, kappa=0.7, r_ref=1.3, kT=0.7, mass=1.0, dt=0.005)
RHO = 64 / 5.5 ** 3
BASE = dict(xi=0.7, s=4.10, eta=0.7, p=7, rank_noise=40, rank_damp=16)
TOLS = {"5e-8": 5e-8, "2e-7": 2e-7, "1e-6": 1e-6, "1e-4": 1e-4}   # baseline errors: 1.3e-7 (A), 4.5e-8 (B)
XIS = (0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5, 1.8)
S_GRID = tuple(np.round(np.arange(2.0, 4.81, 0.1), 2))
ETAS = (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3)
PS = (5, 6, 7)
RANKS_N = (6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)
RANKS_D = (2, 3, 4, 5, 6, 8, 12, 16, 24)


def box(N):
    return (N / RHO) ** (1 / 3)


def kernel(law, xi):
    return rk.toy_kernel(law, TOY["gamma"], TOY["kappa"], TOY["r_ref"], xi)


def fast_op(L, law, xi, s, eta, p, pair_search="tree"):
    """pair_search "tree" (rk.real_pairs_tree) gives the same pair set as the original "images" enumeration."""
    return rk.FastFriction(L, kernel(law, xi), s, eta, p, pair_search=pair_search)


def lattice_matrix(lat, q, chunk=8192):
    """lat.matrix(q) with the pair tensors evaluated in chunks (the Chebyshev far field needs ~14 kB per pair)."""
    i, j, dr, _ = v2.pair_geometry(q, lat.box)
    K = np.concatenate([lat.pair_tensors(dr[c:c + chunk]) for c in range(0, len(dr), chunk)])
    return v2.assemble_friction(i, j, K, len(q))


def lattice(L, law):
    return v2.LatticeFriction(L, kernel=law, gamma=TOY["gamma"], kappa=TOY["kappa"], r_ref=TOY["r_ref"])


def config(N, law, potential="double_well", seed=101, rng_seed=0):
    """N = 64: v2 final state; N = 8 * 64: 2x2x2 replica of it with 0.02 jitter; other N: jittered lattice."""
    q64 = None
    f = CKPT / f"{potential}_burn140" / "lattice" / f"{potential}_{law}" / f"seed_{seed}" / "restart.npz"
    with np.load(f) as ck:
        q64 = ck["Q"] % 5.5
    rng = np.random.default_rng(rng_seed)
    L = box(N)
    if N == 64:
        return q64, L
    k = round((N / 64) ** (1 / 3))
    if k ** 3 * 64 == N:
        shifts = np.stack(np.meshgrid(*[np.arange(k)] * 3, indexing="ij"), -1).reshape(-1, 3) * 5.5
        q = (q64[None] + shifts[:, None]).reshape(-1, 3) + rng.uniform(-0.02, 0.02, (N, 3))
        return q % L, L
    m = int(np.ceil(N ** (1 / 3)))
    while m ** 3 < N:
        m += 1
    g = np.stack(np.meshgrid(*[np.arange(m)] * 3, indexing="ij"), -1).reshape(-1, 3)
    g = g[rng.permutation(len(g))[:N]]
    return ((g + 0.5) * L / m + rng.uniform(-0.05, 0.05, (N, 3))) % L, L


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def load(name):
    return json.loads((OUT / name).read_text())


def f_noise(dt):
    return lambda l: np.sqrt(TOY["kT"] * TOY["mass"] * -np.expm1(-2 * dt * l / TOY["mass"]))


def f_damp(dt):
    return lambda l: np.exp(-dt * l / TOY["mass"])


# ----------------------------------------------------------------------------
# profile: one full step, broken down
# ----------------------------------------------------------------------------
def matvec_breakdown(op, X, reps=20):
    """Times the pieces of GammaH.matvec (same operations, same order) and checks the sum against op.matvec."""
    mesh = op.mesh
    N = op.N
    M3 = mesh.M ** 3
    t = dict(real_space=0.0, spread=0.0, fft_forward=0.0, influence=0.0, fft_inverse=0.0, gather=0.0, degree_term=0.0)
    for _ in range(reps):
        t0 = time.perf_counter()
        f = (op.w * np.einsum("pa,pa->p", op.d, X[op.i] - X[op.j]))[:, None] * op.d
        out = np.stack([np.bincount(op.i, f[:, c], N) - np.bincount(op.j, f[:, c], N) for c in range(3)], axis=1)
        t1 = time.perf_counter()
        grids = [np.bincount(op.idx.ravel(), (op.wt * X[:, c, None]).ravel(), M3).reshape((mesh.M,) * 3) for c in range(3)]
        t2 = time.perf_counter()
        rho_hat = [np.fft.rfftn(g) for g in grids]
        t3 = time.perf_counter()
        prods = [sum(mesh._G(d, c) * rho_hat[c] for c in range(3)) for d in range(3)]
        t4 = time.perf_counter()
        phi = [np.fft.irfftn(pr, s=(mesh.M,) * 3, axes=(0, 1, 2)) for pr in prods]
        t5 = time.perf_counter()
        u = np.stack([(phi[d].ravel()[op.idx] * op.wt).sum(axis=1) for d in range(3)], axis=1) / mesh.h ** 3
        t6 = time.perf_counter()
        res = out + np.einsum("nab,nb->na", op.D, X) - u
        t7 = time.perf_counter()
        for k, a, b in (("real_space", t0, t1), ("spread", t1, t2), ("fft_forward", t2, t3), ("influence", t3, t4),
                        ("fft_inverse", t4, t5), ("gather", t5, t6), ("degree_term", t6, t7)):
            t[k] += (b - a) / reps
    ref = op.matvec(X.ravel()).reshape(N, 3)
    t0 = time.perf_counter()
    for _ in range(reps):
        op.matvec(X.ravel())
    t["matvec_total"] = (time.perf_counter() - t0) / reps
    t["breakdown_vs_matvec_maxdiff"] = float(np.abs(res - ref).max() / np.abs(ref).max())
    return t


def build_breakdown(fast, x):
    """Operator construction for one configuration, split as in rk.FastFriction.gamma_h."""
    t = {}
    t0 = time.perf_counter()
    x = np.asarray(x, float) % fast.L
    i, j, d = fast._pairs(x, fast.L, fast.rc)
    t1 = time.perf_counter()
    r = np.linalg.norm(d, axis=1)
    w = fast.K.theta_S_scaled_closed(r) * np.exp(-(fast.K.xi * r) ** 2) / r ** 2
    t2 = time.perf_counter()
    idx, wt = fast.mesh.flat(x)
    t3 = time.perf_counter()
    D = fast.mesh.degree(idx, wt)
    t4 = time.perf_counter()
    t.update(neighbour_pairs=t1 - t0, real_weights=t2 - t1, stencil=t3 - t2, degree=t4 - t3, n_real_pairs=len(r))
    return t


def timed_step(L, law, potential, q, p, xi_noise, fast, rank_noise, rank_damp, force_method="neighbor"):
    dt, m = TOY["dt"], TOY["mass"]
    force_fn = td.FORCES[force_method]
    t = {}
    t0 = time.perf_counter()
    force, _, _ = force_fn(q, L, potential)
    t["force_1"] = time.perf_counter() - t0
    p = p + 0.5 * dt * force
    q = q + 0.5 * dt / m * p
    t0 = time.perf_counter()
    op = fast.gamma_h(q)
    t["operator_build"] = time.perf_counter() - t0
    pm = p.mean(axis=0)
    t0 = time.perf_counter()
    damp, idamp = timed_apply(op, proj(p.ravel()), rank_damp, f_damp(dt))
    noise, inoise = timed_apply(op, proj(xi_noise.ravel()), rank_noise, f_noise(dt))
    t["lanczos_total"] = time.perf_counter() - t0
    p = pm[None] + (proj(damp) + proj(noise)).reshape(p.shape)
    q = q + 0.5 * dt / m * p
    t0 = time.perf_counter()
    force, _, _ = force_fn(q, L, potential)
    t["force_2"] = time.perf_counter() - t0
    p = p + 0.5 * dt * force
    for k in ("t_action", "t_reorth", "t_small", "t_other"):
        t["lanczos_" + k[2:]] = idamp[k] + inoise[k]
    t["n_actions"] = idamp["actions"] + inoise["actions"]
    return q, p, t


def stage_profile(args):
    res = {}
    for N in args.N:
        for law in ("A", "B"):
            q, L = config(N, law)
            pot = "double_well"
            fast = fast_op(L, law, BASE["xi"], BASE["s"], BASE["eta"], BASE["p"], pair_search=args.pair_search)
            rng = np.random.default_rng(1)
            p = rng.normal(0, np.sqrt(TOY["kT"]), q.shape)
            p -= p.mean(axis=0)
            steps = []
            for k in range(2 + args.reps):
                q, p, t = timed_step(L, law, pot, q, p, rng.standard_normal(q.shape), fast, BASE["rank_noise"], BASE["rank_damp"])
                if k >= 2:
                    steps.append(t)
            med = {k: float(np.median([s[k] for s in steps])) for k in steps[0]}
            med["step_total"] = float(np.median([sum(s[k] for k in ("force_1", "operator_build", "lanczos_total", "force_2")) for s in steps]))
            op = fast.gamma_h(q)
            mv = matvec_breakdown(op, rng.normal(size=q.shape))
            bb = build_breakdown(fast, q)
            fa = {}
            for fm in ("allpairs", "neighbor"):
                fn = td.FORCES[fm]
                fn(q, L, pot)
                t0 = time.perf_counter()
                for _ in range(5):
                    fn(q, L, pot)
                fa[fm] = (time.perf_counter() - t0) / 5
            res[f"N{N}_{law}"] = dict(N=N, L=L, law=law, pair_search=args.pair_search, mesh_M=fast.mesh.M, rc=fast.rc,
                                     kc=fast.kc, step_median=med,
                                     matvec=mv, build=bb, force_one_call=fa,
                                     share_actions=med["lanczos_action"] / med["step_total"],
                                     timed_steps=args.reps, warmup_steps=2)
            r = res[f"N{N}_{law}"]
            print(f"N={N} {law}: step {1e3 * med['step_total']:.1f} ms, actions {med['n_actions']:.0f} = "
                  f"{100 * r['share_actions']:.1f}%, build {1e3 * med['operator_build']:.1f} ms, forces "
                  f"{1e3 * (med['force_1'] + med['force_2']):.2f} ms; matvec {1e3 * mv['matvec_total']:.2f} ms "
                  f"(real {1e3 * mv['real_space']:.2f}, spread {1e3 * mv['spread']:.2f}, fft {1e3 * (mv['fft_forward'] + mv['fft_inverse']):.2f}, "
                  f"infl {1e3 * mv['influence']:.2f}, gather {1e3 * mv['gather']:.2f})", flush=True)
    save(f"profile_{args.pair_search}.json", res)


# ----------------------------------------------------------------------------
# search: staged operator search
# ----------------------------------------------------------------------------
class ErrorCases:
    """Fixed configurations with their dense full-periodic reference matrices (computed once)."""

    def __init__(self, law, N=64):
        self.law, self.cases = law, []
        specs = [("double_well", 101), ("lj", 101)] if N == 64 else [("double_well", 101)]
        for pot, seed in specs:
            q, L = config(N, law, pot, seed)
            G = lattice_matrix(lattice(L, law), q)
            self.cases.append(dict(q=q, L=L, G=G, nrm=float(np.linalg.norm(G, 2)), name=f"{pot}_{seed}_N{N}"))

    def error(self, xi, s, eta, p):
        errs = []
        for c in self.cases:
            fast = fast_op(c["L"], self.law, xi, s, eta, p)
            Gh = fast.gamma_h(c["q"]).dense()
            errs.append(float(np.linalg.norm(Gh - c["G"], 2) / c["nrm"]))
        return max(errs), fast.mesh.M


def time_candidate(law, N, xi, s, eta, p, reps=10):
    q, L = config(N, law)
    t0 = time.perf_counter()
    fast = fast_op(L, law, xi, s, eta, p)
    t_pre = time.perf_counter() - t0
    fast.gamma_h(q)
    t0 = time.perf_counter()
    for _ in range(3):
        op = fast.gamma_h(q)
    t_build = (time.perf_counter() - t0) / 3
    X = np.random.default_rng(0).normal(size=q.shape).ravel()
    op.matvec(X)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        op.matvec(X)
        ts.append(time.perf_counter() - t0)
    mv = float(np.median(ts))
    return dict(precompute=t_pre, build=t_build, matvec=mv, M=fast.mesh.M, n_real=len(op.w), rc=fast.rc,
                est_step_56=t_build + 56 * mv)


def stage_search(args):
    out = {}
    path = OUT / "search.json"
    if path.exists():
        out = load("search.json")
    for law in args.laws:
        cases = ErrorCases(law)
        for tname in args.tols:
            key = f"{law}_{tname}"
            if key in out and not args.redo:
                continue
            tol = TOLS[tname]
            rows, evals = [], 0
            t_stage = time.process_time()
            for xi in XIS:
                s_min = None
                for s in S_GRID:                                  # cutoff scale with a fine mesh (eta 0.3, p 7)
                    if 2 * xi * s * 5.5 / (2 * np.pi) > 40:       # keep the mode count bounded
                        break
                    e, _ = cases.error(xi, s, 0.3, 7)
                    evals += 1
                    if e <= tol / 2:
                        s_min = float(s)
                        break
                if s_min is None:
                    rows.append(dict(xi=xi, status="no s in grid meets tol/2"))
                    continue
                for p in PS:                                       # coarsest mesh meeting the full tolerance
                    best, seen = None, set()
                    for eta in ETAS:
                        M = next_smooth(np.ceil(5.5 / (eta * np.pi / (2 * xi * s_min)) - 1e-9))
                        if M in seen:
                            continue
                        seen.add(M)
                        e, Mm = cases.error(xi, s_min, eta, p)
                        evals += 1
                        if e <= tol:
                            best = dict(xi=xi, s=s_min, eta=eta, p=p, M64=Mm, err64=e)
                            break
                    rows.append(best if best else dict(xi=xi, s=s_min, p=p, status="no eta in grid meets tol"))
                print(f"[{key}] xi={xi}: s_min={s_min} -> " + ", ".join(
                    f"p{r['p']}: eta {r.get('eta')} M {r.get('M64')} err {r.get('err64', float('nan')):.1e}" for r in rows[-3:] if 'p' in r), flush=True)
            ok = [r for r in rows if "eta" in r]
            for r in ok:                                           # cost of each accepted candidate
                for N in args.timing_N:
                    r[f"cost_N{N}"] = time_candidate(law, N, r["xi"], r["s"], r["eta"], r["p"])
            out[key] = dict(law=law, tol=tol, candidates=rows, evaluations=evals,
                            cpu_seconds=time.process_time() - t_stage)
            save("search.json", out)
    # baseline errors / costs for reference
    if "baseline" not in out or args.redo:
        base = {}
        for law in args.laws:
            cases = ErrorCases(law)
            e, M = cases.error(BASE["xi"], BASE["s"], BASE["eta"], BASE["p"])
            base[law] = dict(err64=e, M64=M, **{f"cost_N{N}": time_candidate(law, N, BASE["xi"], BASE["s"], BASE["eta"], BASE["p"]) for N in args.timing_N})
        out["baseline"] = base
        save("search.json", out)


def spectral_err_matfree(op, Gr, nrm, seed=0):
    """|| Gamma_h - Gamma_ref ||_2 / || Gamma_ref ||_2 with the literal matvec (ARPACK, largest-magnitude eigenvalue of
    the symmetric difference) and the dense reference matrix."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    n = Gr.shape[0]
    E = LinearOperator((n, n), matvec=lambda v: op.matvec(np.ravel(v)) - Gr @ np.ravel(v), dtype=float)
    v0 = np.random.default_rng(seed).standard_normal(n)
    lam = eigsh(E, k=1, which="LM", tol=1e-3, ncv=24, v0=v0, return_eigenvectors=False)
    return float(abs(lam[0]) / nrm)


def stage_refine(args):
    """Mesh choice at a large box (default N = 512), where the N = 64 mesh rounding (M = next_smooth(...)) does not
    transfer: for the cheapest (xi, s) of each law / budget from `search`, scan eta from coarse to fine for every p
    and keep the coarsest mesh whose spectral error at this N (matrix-free, vs dense full-periodic reference) is
    <= tol. The transferable mesh parameter is the realised eta_actual (h kc / pi): any box with
    M = next_smooth(ceil(L kc / (eta_actual pi))) has a mesh at least as fine."""
    search = load("search.json")
    out = load("refine.json") if (OUT / "refine.json").exists() else {}
    N = args.N[0]
    for law in args.laws:
        q, L = config(N, law)
        Gr = lattice_matrix(lattice(L, law), q)
        nrm = float(np.linalg.eigvalsh(Gr)[-1])
        fb = fast_op(L, law, BASE["xi"], BASE["s"], BASE["eta"], BASE["p"])
        check = dict(matfree=spectral_err_matfree(fb.gamma_h(q), Gr, nrm),
                     dense=float(np.linalg.norm(fb.gamma_h(q).dense() - Gr, 2) / nrm))
        out[f"{law}_baseline_N{N}"] = dict(law=law, N=N, params=dict(BASE), err=check["matfree"], check=check,
                                           M=fb.mesh.M, eta_actual=fb.mesh.eta_actual,
                                           cost=time_candidate(law, N, BASE["xi"], BASE["s"], BASE["eta"], BASE["p"]))
        print(f"[{law} baseline N={N}] spectral error matrix-free {check['matfree']:.3e} vs dense {check['dense']:.3e}", flush=True)
        for tname in args.tols:
            if f"{law}_{tname}" not in search:
                continue
            tol = TOLS[tname]
            ok = [r for r in search[f"{law}_{tname}"]["candidates"] if "eta" in r]
            best = {}
            for r in ok:
                k = (r["xi"], r["s"])
                best[k] = min(best.get(k, np.inf), r[f"cost_N{N}"]["est_step_56"])
            for xi, s_ in sorted(best, key=best.get)[: args.top]:
                kc = 2 * xi * s_
                for p_ in PS:
                    key = f"{law}_{tname}_xi{xi}_s{s_}_p{p_}_N{N}"
                    if key in out:
                        continue
                    seen, acc, tried = set(), None, []
                    for eta in ETAS:
                        M = next_smooth(np.ceil(L / (eta * np.pi / kc) - 1e-9))
                        if M in seen:
                            continue
                        seen.add(M)
                        fast = fast_op(L, law, xi, s_, eta, p_)
                        e = spectral_err_matfree(fast.gamma_h(q), Gr, nrm)
                        tried.append(dict(eta=eta, M=int(M), eta_actual=fast.mesh.eta_actual, err=e))
                        if e <= tol:
                            acc = dict(xi=xi, s=s_, p=p_, eta=eta, M=int(M), eta_actual=fast.mesh.eta_actual, err=e,
                                       cost=time_candidate(law, N, xi, s_, eta, p_))
                            break
                    out[key] = dict(law=law, tol=tname, N=N, accepted=acc, tried=tried)
                    save("refine.json", out)
                    if acc:
                        print(f"[{key}] eta {acc['eta']} M {acc['M']} (eta_actual {acc['eta_actual']:.3f}) err {acc['err']:.2e} "
                              f"build {1e3 * acc['cost']['build']:.1f} ms matvec {1e3 * acc['cost']['matvec']:.2f} ms", flush=True)
                    else:
                        print(f"[{key}] no mesh in grid meets tol", flush=True)
        save("refine.json", out)


# ----------------------------------------------------------------------------
# ranks: Lanczos matrix-function errors (not Gamma v)
# ----------------------------------------------------------------------------
def krylov_errors(op, Dh, Dr, vecs, fun, ranks):
    """Lanczos f(Gamma_h) v for several ranks (one Lanczos run per vector, truncated), with relative errors against
    dense f(Gamma_h) ("krylov") and dense f(Gamma_ref) ("total"). Dh, Dr: DenseRef of Gamma_h and Gamma_ref.
    Returns the error table, the smallest Ritz value seen, and the approximations / exact vectors for O-step checks."""
    smax = min(max(ranks), 3 * op.N - 3)
    res = {r: dict(krylov=0.0, total=0.0) for r in ranks}
    lam_min = np.inf
    approx_all, exact_h_all, exact_r_all = [], [], []
    for v in vecs:
        v = proj(v)
        exact_h, exact_r = Dh.apply(fun, v), Dr.apply(fun, v)
        Lz = lanczos(op, v, smax)
        ap = {}
        for r in ranks:
            s = min(r, Lz["s"])
            th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], s))
            lam_min = min(lam_min, th[0])
            if th[0] <= 0:
                res[r]["negative_ritz"] = True
                continue
            ap[r] = proj(Lz["nz"] * (Lz["Q"][:s].T @ (U @ (fun(th) * U[0]))))
            res[r]["krylov"] = max(res[r]["krylov"], float(np.linalg.norm(ap[r] - exact_h) / np.linalg.norm(exact_h)))
            res[r]["total"] = max(res[r]["total"], float(np.linalg.norm(ap[r] - exact_r) / np.linalg.norm(exact_r)))
        approx_all.append(ap)
        exact_h_all.append(exact_h)
        exact_r_all.append(exact_r)
    return res, float(lam_min), (approx_all, exact_h_all, exact_r_all)


def pick_candidates(search, law, tname, N=512):
    """Accepted operator candidates sorted by the estimated step cost build + 56 Gamma v at N (baseline ranks)."""
    ok = [r for r in search[f"{law}_{tname}"]["candidates"] if "eta" in r and f"cost_N{N}" in r]
    return sorted(ok, key=lambda r: r[f"cost_N{N}"]["est_step_56"])


def stage_ranks(args):
    """Per candidate and N: operator checks (Gamma v error, symmetry, momentum null space, lambda_min on Range(Pi)),
    Lanczos noise sqrt-action and damping-action errors per rank, and the O-step error for every (noise, damping)
    rank pair: | R~ p + S~ z - (R p + S z) | / | R p + S z | with the exact R, S of Gamma_h (Krylov part, which is
    what breaks the friction-noise pairing) and of Gamma_ref (total). Same vectors for every candidate."""
    search = load("search.json")
    out = load("ranks.json") if (OUT / "ranks.json").exists() and not args.redo else {}
    refs = {}
    for law in args.laws:
        plan = {}
        for tname in args.tols:
            for c in pick_candidates(search, law, tname)[: args.top]:
                plan.setdefault(f"{law}_xi{c['xi']}_s{c['s']}_eta{c['eta']}_p{c['p']}", (tname, c))
        plan[f"{law}_baseline"] = ("baseline", dict(xi=BASE["xi"], s=BASE["s"], eta=BASE["eta"], p=BASE["p"]))
        for key, (tname, c) in plan.items():
            if key in out:
                continue
            rows = {}
            for N in args.N:
                q, L = config(N, law)
                if (N, law) not in refs:
                    Gr = lattice_matrix(lattice(L, law), q)
                    refs[(N, law)] = (Gr, DenseRef(Gr, N))
                Gr, Dr = refs[(N, law)]
                op = fast_op(L, law, c["xi"], c["s"], c["eta"], c["p"]).gamma_h(q)
                Gh = op.dense()
                Dh = DenseRef(Gh, N)
                rng = np.random.default_rng([5, N])
                zs = [rng.standard_normal(3 * N) for _ in range(args.vectors)]
                ps = [rng.normal(0, np.sqrt(TOY["kT"] * TOY["mass"]), 3 * N) for _ in range(args.vectors)]
                noise, lmin_n, (an, nh, nr) = krylov_errors(op, Dh, Dr, zs, f_noise(TOY["dt"]), RANKS_N)
                damp, lmin_d, (ad, dh, dr) = krylov_errors(op, Dh, Dr, ps, f_damp(TOY["dt"]), RANKS_D)
                ostep = {}
                for rn in RANKS_N:
                    for rd in RANKS_D:
                        eh = er = 0.0
                        for k in range(args.vectors):
                            if rn not in an[k] or rd not in ad[k]:
                                eh = er = np.inf
                                break
                            y = an[k][rn] + ad[k][rd]
                            eh = max(eh, float(np.linalg.norm(y - nh[k] - dh[k]) / np.linalg.norm(nh[k] + dh[k])))
                            er = max(er, float(np.linalg.norm(y - nr[k] - dr[k]) / np.linalg.norm(nr[k] + dr[k])))
                        ostep[f"n{rn}_d{rd}"] = dict(krylov=eh, total=er)
                X = np.stack([proj(z) for z in zs])
                gv_err = max(float(np.linalg.norm(Gh @ x - Gr @ x) / np.linalg.norm(Gr @ x)) for x in X)
                lit = max(float(np.linalg.norm(op.matvec(x) - Gh @ x) / np.linalg.norm(Gh @ x)) for x in X)
                T = np.kron(np.ones((N, 1)), np.eye(3))
                rows[f"N{N}"] = dict(noise=noise, damp=damp, ostep=ostep, lam_min=float(Dh.lam[0]),
                                     lam_max=float(Dh.lam[-1]), dt_lam_max=TOY["dt"] * float(Dh.lam[-1]),
                                     lam_min_ref=float(Dr.lam[0]), min_ritz=min(lmin_n, lmin_d),
                                     sym_resid=float(np.linalg.norm(Gh - Gh.T) / np.linalg.norm(Gh)),
                                     null_resid=float(np.abs(Gh @ T).max() / np.abs(Gh).max()),
                                     op_err=float(np.linalg.norm(Gh - Gr, 2) / np.linalg.norm(Gr, 2)),
                                     gv_err=gv_err, literal_vs_dense=lit, M=op.mesh.M, n_real=len(op.w),
                                     vectors=args.vectors)
                print(f"[{key}] N={N}: op_err {rows[f'N{N}']['op_err']:.1e} Gv {gv_err:.1e}, lam [{Dh.lam[0]:.3f},{Dh.lam[-1]:.1f}], "
                      "noise krylov " + " ".join(f"r{r}:{noise[r]['krylov']:.0e}" for r in RANKS_N) + " | damp "
                      + " ".join(f"r{r}:{damp[r]['krylov']:.0e}" for r in RANKS_D), flush=True)
            out[key] = dict(law=law, tol=tname, params=c, by_N=rows)
            save("ranks.json", out)


def gv_rows_error(op, lat, q, X, rows, chunk=8192):
    """Gamma v error on a subset of particles against the direct full-periodic sum (v2 LatticeFriction pair
    tensors, k = 0 retained): (Gamma_ref X)_i = sum_j K_ij (X_i - X_j) for i in rows; O(len(rows) N)."""
    N = len(q)
    ii = np.repeat(rows, N)
    jj = np.tile(np.arange(N), len(rows))
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    dr = q[ii] - q[jj]
    dr -= lat.box * np.round(dr / lat.box)
    K = np.concatenate([lat.pair_tensors(dr[c:c + chunk]) for c in range(0, len(dr), chunk)])
    f = np.einsum("pab,pb->pa", K, X[ii] - X[jj])
    ref = np.stack([np.bincount(np.searchsorted(rows, ii), f[:, c], len(rows)) for c in range(3)], axis=1)
    fast = op.matvec(X.ravel()).reshape(N, 3)[rows]
    return float(np.linalg.norm(fast - ref) / np.linalg.norm(ref))


def stage_ranks_large(args):
    """Large N without dense matrices: Lanczos errors per rank against a converged high-rank Lanczos result
    (noise rank 96, damping rank 32; convergence shown by the rank-80 / rank-24 difference), extreme Ritz values,
    and the Gamma v error on 48 sampled particles against the direct full-periodic sum."""
    cands = json.loads(Path(args.candidates).read_text())
    out = load("ranks_large.json") if (OUT / "ranks_large.json").exists() else {}
    for c in cands:
        for N in args.N:
            key = f"{c['law']}_{c['tag']}_N{N}"
            if key in out:
                continue
            t0 = time.process_time()
            q, L = config(N, c["law"])
            op = fast_op(L, c["law"], c["xi"], c["s"], c["eta"], c["p"]).gamma_h(q)
            rng = np.random.default_rng([7, N])
            res = dict(noise={r: 0.0 for r in RANKS_N}, damp={r: 0.0 for r in RANKS_D}, noise_conv=0.0, damp_conv=0.0,
                       ritz_min=np.inf, ritz_max=0.0)
            for v in range(args.vectors):
                for kind, vec, fun, ranks, smax, sconv in (
                        ("noise", rng.standard_normal(3 * N), f_noise(TOY["dt"]), RANKS_N, 96, 80),
                        ("damp", rng.normal(0, np.sqrt(TOY["kT"] * TOY["mass"]), 3 * N), f_damp(TOY["dt"]), RANKS_D, 32, 24)):
                    Lz = lanczos(op, proj(vec), smax)
                    fr = {}
                    for r in list(ranks) + [sconv, smax]:
                        sr = min(r, Lz["s"])
                        th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], sr))
                        if th[0] <= 0:
                            raise ValueError("non-positive Ritz value")
                        res["ritz_min"], res["ritz_max"] = min(res["ritz_min"], th[0]), max(res["ritz_max"], th[-1])
                        fr[r] = Lz["Q"][:sr].T @ (U @ (fun(th) * U[0]))
                    nref = np.linalg.norm(fr[smax])
                    for r in ranks:
                        res[kind][r] = max(res[kind][r], float(np.linalg.norm(fr[r] - fr[smax]) / nref))
                    res[kind + "_conv"] = max(res[kind + "_conv"], float(np.linalg.norm(fr[sconv] - fr[smax]) / nref))
            lat = lattice(L, c["law"])
            rows = np.sort(rng.choice(N, 48, replace=False))
            res["gv_rows_err"] = max(gv_rows_error(op, lat, q, rng.standard_normal((N, 3)), rows) for _ in range(2))
            res.update(law=c["law"], tag=c["tag"], N=N, params=c, M=op.mesh.M, n_real=len(op.w), vectors=args.vectors,
                       cpu_s=time.process_time() - t0)
            out[key] = res
            save("ranks_large.json", out)
            print(f"[{key}] Ritz [{res['ritz_min']:.3f},{res['ritz_max']:.1f}] Gv(rows) {res['gv_rows_err']:.1e} noise "
                  + " ".join(f"r{r}:{res['noise'][r]:.0e}" for r in RANKS_N if r <= 48) + f" (conv {res['noise_conv']:.0e})"
                  + " | damp " + " ".join(f"r{r}:{res['damp'][r]:.0e}" for r in RANKS_D if r <= 12)
                  + f" (conv {res['damp_conv']:.0e}) [{res['cpu_s']:.0f} s]", flush=True)


def stage_slowmodes(args):
    """Final static check of the selected candidates at N <= 512: spectral operator error against the dense full-periodic
    reference (N = 64: DW and LJ states), symmetry, momentum null space, k = 0 retained, lambda_min on Range(Pi); and the
    mode-resolved Lanczos error at N <= 512 (dense Gamma_h): relative error of the component of f(Gamma_h) v in the
    span of the n_slow lowest eigenvectors on Range(Pi) (the long-wavelength modes, slowest Krylov convergence),
    for the noise sqrt-action and the damping action at the candidate ranks."""
    cands = json.loads(Path(args.candidates).read_text())
    out = load("slowmodes.json") if (OUT / "slowmodes.json").exists() else {}
    for c in cands:
        for N in args.N:
            key = f"{c['law']}_{c['tag']}_N{N}"
            if key in out:
                continue
            q, L = config(N, c["law"])
            op = fast_op(L, c["law"], c["xi"], c["s"], c["eta"], c["p"]).gamma_h(q)
            Gh = op.dense()
            Dh = DenseRef(Gh, N)
            errs = []
            for pot in (["double_well", "lj"] if N == 64 else ["double_well"]):
                qq, _ = config(N, c["law"], pot)
                Gr = lattice_matrix(lattice(L, c["law"]), qq)
                Gq = op.dense() if pot == "double_well" else fast_op(L, c["law"], c["xi"], c["s"], c["eta"], c["p"]).gamma_h(qq).dense()
                errs.append(float(np.linalg.norm(Gq - Gr, 2) / np.linalg.norm(Gr, 2)))
            T = np.kron(np.ones((N, 1)), np.eye(3))
            static = dict(op_err=max(errs), op_err_cases=errs, M=op.mesh.M, eta_actual=op.mesh.eta_actual,
                          sym_resid=float(np.linalg.norm(Gh - Gh.T) / np.linalg.norm(Gh)),
                          null_resid=float(np.abs(Gh @ T).max() / np.abs(Gh).max()),
                          lam_min_range=float(Dh.lam[0]), lam_max=float(Dh.lam[-1]),
                          min_full_eig=float(Dh.full_eigs[0]), k0_retained=bool(np.any(np.all(op.mesh.mv == 0, axis=1))))
            rng = np.random.default_rng([9, N])
            res = {}
            for n_slow in (3, 12, 48):
                res[n_slow] = dict(noise=0.0, damp=0.0, lam_hi=float(Dh.lam[n_slow - 1]))
            for kind, fun, rank in (("noise", f_noise(TOY["dt"]), c["rank_noise"]), ("damp", f_damp(TOY["dt"]), c["rank_damp"])):
                for _ in range(args.vectors):
                    v = proj(rng.standard_normal(3 * N) * (1.0 if kind == "noise" else np.sqrt(TOY["kT"] * TOY["mass"])))
                    exact = Dh.apply(fun, v)
                    Lz = lanczos(op, v, rank)
                    sr = min(rank, Lz["s"])
                    th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], sr))
                    approx = proj(Lz["nz"] * (Lz["Q"][:sr].T @ (U @ (fun(th) * U[0]))))
                    for n_slow in res:
                        B = Dh.U[:, :n_slow]
                        e, a_ = B.T @ exact, B.T @ approx
                        res[n_slow][kind] = max(res[n_slow][kind], float(np.linalg.norm(a_ - e) / np.linalg.norm(e)))
            out[key] = dict(law=c["law"], tag=c["tag"], N=N, rank_noise=c["rank_noise"], rank_damp=c["rank_damp"],
                            lam_min=float(Dh.lam[0]), slow=res, static=static, params=c)
            save("slowmodes.json", out)
            print(f"[{key}] M {static['M']} op_err {static['op_err']:.2e} sym {static['sym_resid']:.0e} null {static['null_resid']:.0e} "
                  f"lam_min {static['lam_min_range']:.3f} | n{c['rank_noise']}/d{c['rank_damp']}: " + "; ".join(
                f"lowest {k} modes (lam <= {v['lam_hi']:.2f}): noise {v['noise']:.1e} damp {v['damp']:.1e}" for k, v in res.items()), flush=True)


def stage_modes(args):
    """Rank requirement per N, resolved by mode. One operator per law (the Krylov error depends on the spectrum, which
    all accepted operators share to <= 1e-4). For each rank: relative error of the Lanczos f(Gamma_h) v
    (noise sqrt-action and damping action) (a) overall, (b) in the k = 2 pi / L plane-wave components
    J(k) = sum_i u_i exp(i k.x_i), k along x, y, z (the hydrodynamic modes of C_L, C_T), (c) in the span of the
    3 / 12 lowest eigenvectors on Range(Pi) (N <= 512 only). Reference: dense f(Gamma_h) for N <= 512; for larger N
    the rank-96 (noise) / rank-32 (damping) Lanczos result, with the rank-80 / rank-24 difference as convergence check."""
    cands = json.loads(Path(args.candidates).read_text())
    out = load("modes.json") if (OUT / "modes.json").exists() else {}
    for law in args.laws:
        c = next(x for x in cands if x["law"] == law)
        for N in args.N:
            key = f"{law}_N{N}"
            if key in out:
                continue
            t0 = time.process_time()
            q, L = config(N, law)
            x = q % L
            op = fast_op(L, law, c["xi"], c["s"], c["eta"], c["p"]).gamma_h(q)
            dense = N <= 512
            if dense:
                Dh = DenseRef(op.dense(), N)
            ph = np.exp(1j * 2 * np.pi / L * x)                           # (N, 3): k = 2 pi / L along x, y, z
            rng = np.random.default_rng([11, N])
            res = {}
            for kind, fun, ranks, smax, sconv in (("noise", f_noise(TOY["dt"]), RANKS_N, 96, 80),
                                                  ("damp", f_damp(TOY["dt"]), RANKS_D, 32, 24)):
                tab = {r: dict(overall=0.0, kmin=0.0, low3=0.0, low12=0.0) for r in ranks}
                conv = 0.0
                ritz = [np.inf, 0.0]
                for _ in range(args.vectors):
                    v = proj(rng.standard_normal(3 * N) * (1.0 if kind == "noise" else np.sqrt(TOY["kT"] * TOY["mass"])))
                    Lz = lanczos(op, v, min(smax, 3 * N - 3))
                    fr = {}
                    for r in sorted(set(ranks) | {sconv, smax}):
                        sr = min(r, Lz["s"])
                        th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], sr))
                        ritz = [min(ritz[0], th[0]), max(ritz[1], th[-1])]
                        fr[r] = proj(Lz["nz"] * (Lz["Q"][:sr].T @ (U @ (fun(th) * U[0]))))
                    ref = Dh.apply(fun, v) if dense else fr[smax]
                    if not dense:
                        conv = max(conv, float(np.linalg.norm(fr[sconv] - ref) / np.linalg.norm(ref)))
                    Jref = ph.T @ ref.reshape(N, 3)
                    for r in ranks:
                        d = fr[r] - ref
                        Jd = ph.T @ d.reshape(N, 3)
                        e = tab[r]
                        e["overall"] = max(e["overall"], float(np.linalg.norm(d) / np.linalg.norm(ref)))
                        e["kmin"] = max(e["kmin"], float(np.max(np.linalg.norm(Jd, axis=1) / np.linalg.norm(Jref, axis=1))))
                        if dense:
                            for n_low, name in ((3, "low3"), (12, "low12")):
                                B = Dh.U[:, :n_low]
                                e[name] = max(e[name], float(np.linalg.norm(B.T @ d) / np.linalg.norm(B.T @ ref)))
                        else:
                            e["low3"] = e["low12"] = None
                res[kind] = dict(table=tab, conv=conv, ritz=ritz)
            out[key] = dict(law=law, N=N, L=L, operator=c, reference="dense" if dense else "lanczos_96_32",
                            lam_min=float(Dh.lam[0]) if dense else res["noise"]["ritz"][0],
                            lam_max=float(Dh.lam[-1]) if dense else res["noise"]["ritz"][1],
                            vectors=args.vectors, res=res, cpu_s=time.process_time() - t0)
            save("modes.json", out)
            for kind in ("noise", "damp"):
                T = res[kind]["table"]
                print(f"[{key}] {kind}: " + " ".join(
                    f"r{r}:{T[r]['overall']:.0e}/{T[r]['kmin']:.0e}" + (f"/{T[r]['low3']:.0e}" if T[r]['low3'] is not None else "")
                    for r in T) + f" conv {res[kind]['conv']:.0e}", flush=True)


# ----------------------------------------------------------------------------
# validate: short fixed-time dynamics, candidate vs baseline
# ----------------------------------------------------------------------------
def observables_at(tr, q0):
    out = {}
    for k, t in enumerate(tr["t"]):
        if k == 0:
            continue
        V, Q = tr["V"][k], tr["Q"][k]
        v2_ = np.einsum("nc,nc->n", V, V)
        dq = Q - q0
        out[float(t)] = dict(KE=0.5 * TOY["mass"] * v2_.sum() / len(V), U=float(tr["diagnostics"][k, 2]),
                             M4=float(np.mean(v2_ ** 2)), MSD=float(np.einsum("nc,nc->", dq, dq) / len(V)))
    return out


def run_validation_path(job):
    """One coupled path: arm = "baseline" (original code: images pair enumeration, xi 0.7, s 4.10, eta 0.7, p 7,
    ranks 40/16) or a candidate (tree pair enumeration). Same initial state and same noise seed for every arm."""
    law, state, path, cand = job
    tag = cand["tag"]
    f = OUT / "validate_paths" / f"{law}_{tag}_state{state}_path{path:02d}.json"
    if f.exists():
        return json.loads(f.read_text())
    with np.load(CKPT / "double_well_burn140" / "lattice" / f"double_well_{law}" / f"seed_{state}" / "restart.npz") as ck:
        q0, p0 = ck["Q"].copy(), ck["p"].copy()
    seed = int(np.random.SeedSequence([20261009, state, path]).generate_state(1)[0])
    th = td.Thermostat("pppm_lanczos", 5.5, law, TOY["gamma"], TOY["kappa"], TOY["r_ref"], TOY["dt"], TOY["kT"],
                       TOY["mass"], pppm=dict(xi=cand["xi"], s=cand["s"], eta=cand["eta"], p=cand["p"]),
                       rank_noise=cand["rank_noise"], rank_damp=cand["rank_damp"],
                       pair_search=cand.get("pair_search", "tree"))
    t0 = time.perf_counter()
    tr = td.run("double_well", th, q0, p0, steps=400, stride=100, seed=seed)
    rec = dict(law=law, state=state, path=path, noise_seed=seed, arm=cand, obs=observables_at(tr, q0),
               wall=time.perf_counter() - t0, thermostat=th.record, steps=400, stride=100, dt=TOY["dt"])
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, indent=1) + "\n")
    return json.loads(f.read_text())                       # observable keys as strings, as when cached


def stage_validate(args):
    """Fixed-time paired comparison, candidate - baseline, at t = 0.5, 1, 2 (KE per particle, U per particle, <|v|^4>,
    MSD). Per state: mean over paths; across states: mean and Student-t 95 % CI (df = n_states - 1)."""
    from concurrent.futures import ProcessPoolExecutor
    from scipy import stats
    cands = json.loads(Path(args.candidates).read_text())
    laws = sorted({c["law"] for c in cands})
    base = {law: dict(BASE, tag="baseline", law=law, pair_search="images") for law in laws}
    jobs = [(law, s, j, base[law]) for law in laws for s in args.states for j in range(args.paths)]
    jobs += [(c["law"], s, j, c) for c in cands for s in args.states for j in range(args.paths)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        recs = list(pool.map(run_validation_path, jobs))
    B = {(r["law"], r["state"], r["path"]): r for r in recs if r["arm"]["tag"] == "baseline"}
    rows = []
    for c in cands:
        R = [r for r in recs if r["arm"]["tag"] == c["tag"] and r["law"] == c["law"]]
        for t in ("0.5", "1.0", "2.0"):
            for ob in ("KE", "U", "M4", "MSD"):
                d = {r["state"]: [] for r in R}
                for r in R:
                    d[r["state"]].append(r["obs"][t][ob] - B[(r["law"], r["state"], r["path"])]["obs"][t][ob])
                means = np.array([np.mean(v) for v in d.values()])
                allv = np.concatenate([v for v in d.values()])
                n = len(means)
                m, se = float(means.mean()), float(means.std(ddof=1) / np.sqrt(n))
                tc = stats.t.ppf(0.975, n - 1)
                bvals = np.array([B[(r["law"], r["state"], r["path"])]["obs"][t][ob] for r in R])
                rows.append(dict(law=c["law"], tag=c["tag"], t=float(t), observable=ob, n_states=n,
                                 paths_per_state=args.paths, mean_diff=m, ci95_low=m - tc * se, ci95_high=m + tc * se,
                                 baseline_mean=float(bvals.mean()), rel_mean_diff=m / abs(bvals.mean()),
                                 rms_pathwise=float(np.sqrt(np.mean(allv ** 2))), max_abs_pathwise=float(np.abs(allv).max()),
                                 baseline_path_sd=float(bvals.std(ddof=1)),
                                 per_state_mean_diff={str(k): float(np.mean(v)) for k, v in d.items()}))
        walls = [r["wall"] for r in R]
        bw = [B[(r["law"], r["state"], r["path"])]["wall"] for r in R]
        print(f"{c['law']} {c['tag']}: wall per 400-step path {np.mean(walls):.1f} s vs baseline {np.mean(bw):.1f} s", flush=True)
    save("validate.json", rows)
    import csv
    with open(OUT / "validate_summary.csv", "w", newline="") as fh:
        keys = [k for k in rows[0] if k != "per_state_mean_diff"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['law']} {r['tag']} t={r['t']} {r['observable']}: d={r['mean_diff']:+.2e} "
              f"[{r['ci95_low']:+.2e},{r['ci95_high']:+.2e}] rel {r['rel_mean_diff']:+.1e} rms={r['rms_pathwise']:.1e} "
              f"max={r['max_abs_pathwise']:.1e} (baseline path sd {r['baseline_path_sd']:.1e})")


# ----------------------------------------------------------------------------
# timing: dense vs direct periodic sum + Lanczos vs PPPM + Lanczos
# ----------------------------------------------------------------------------
class DirectLatticeOp:
    """Matrix-free direct full-periodic sum: K_ij = sum_n g(|r_ij + nL|) rhat rhat^T for all pairs (v2 LatticeFriction
    pair tensors, k = 0 retained), action sum_j K_ij (x_i - x_j). Same model as the dense reference, O(N^2)."""

    def __init__(self, lat, x, chunk=8192):
        i, j, dr, _ = v2.pair_geometry(x, lat.box)
        K = np.concatenate([lat.pair_tensors(dr[c:c + chunk]) for c in range(0, len(dr), chunk)])
        self.i, self.j, self.K, self.N = i, j, K, len(x)
        self.n_actions, self.t_actions = 0, 0.0

    def matvec(self, v):
        t0 = time.perf_counter()
        X = v.reshape(self.N, 3)
        f = np.einsum("pab,pb->pa", self.K, X[self.i] - X[self.j])
        out = np.stack([np.bincount(self.i, f[:, c], self.N) - np.bincount(self.j, f[:, c], self.N) for c in range(3)], axis=1)
        self.n_actions += 1
        self.t_actions += time.perf_counter() - t0
        return out.ravel()


def timing_case(method, N, law, params, reps, warm):
    q, L = config(N, law)
    pot = "double_well"
    rng = np.random.default_rng(2)
    p = rng.normal(0, np.sqrt(TOY["kT"]), q.shape)
    p -= p.mean(axis=0)
    dt, m = TOY["dt"], TOY["mass"]
    t0 = time.perf_counter()
    if method == "dense":
        lat = lattice(L, law)
    elif method == "direct_lanczos":
        lat = lattice(L, law)
    else:
        fast = fast_op(L, law, params["xi"], params["s"], params["eta"], params["p"],
                       pair_search=params.get("pair_search", "tree"))
    t_pre = time.perf_counter() - t0
    steps, t_act, t_build, t_fun = [], [], [], []
    for k in range(warm + reps):
        ts = time.perf_counter()
        force, _, _ = td.conservative_force_neighbor(q, L, pot)
        p = p + 0.5 * dt * force
        q = q + 0.5 * dt / m * p
        xi = rng.standard_normal(q.shape)
        tb = time.perf_counter()
        if method == "dense":
            G = lattice_matrix(lat, q % L)
            tf = time.perf_counter()
            ref = DenseRef(G, N)
            damp, noise = ref.apply(f_damp(dt), proj(p.ravel())), ref.apply(f_noise(dt), proj(xi.ravel()))
        else:
            op = DirectLatticeOp(lat, q % L) if method == "direct_lanczos" else fast.gamma_h(q)
            tf = time.perf_counter()
            damp, _ = timed_apply(op, proj(p.ravel()), params["rank_damp"], f_damp(dt))
            noise, _ = timed_apply(op, proj(xi.ravel()), params["rank_noise"], f_noise(dt))
            if k >= warm:
                t_act.append(op.t_actions / max(op.n_actions, 1))
        if k >= warm:
            t_build.append(tf - tb)
            t_fun.append(time.perf_counter() - tf)
        p = p.mean(axis=0)[None] + (proj(damp) + proj(noise)).reshape(p.shape)
        q = q + 0.5 * dt / m * p
        force, _, _ = td.conservative_force_neighbor(q, L, pot)
        p = p + 0.5 * dt * force
        if k >= warm:
            steps.append(time.perf_counter() - ts)
    return dict(method=method, N=N, law=law, L=L, params=params, precompute=t_pre, step_median=float(np.median(steps)),
                step_min=float(np.min(steps)), step_max=float(np.max(steps)), reps=reps, warmup=warm,
                gamma_v=float(np.median(t_act)) if t_act else None, operator_build=float(np.median(t_build)),
                matrix_function=float(np.median(t_fun)),
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def stage_timing(args):
    cands = json.loads(Path(args.candidates).read_text()) if args.candidates else []
    res = load("timing.json") if (OUT / "timing.json").exists() else []
    done = {(r["method"], r["N"], r["law"], r.get("tag")) for r in res}
    plan = []
    for N in args.N:
        for law in args.laws:
            plan.append(("dense", N, law, dict(BASE), "dense"))
            plan.append(("direct_lanczos", N, law, dict(BASE), "direct_base_ranks"))
            plan.append(("pppm_lanczos", N, law, dict(BASE, pair_search="images"), "baseline_images"))
            plan.append(("pppm_lanczos", N, law, dict(BASE, pair_search="tree"), "baseline_tree"))
            for c in cands:
                if c["law"] == law:
                    cc = dict(c)
                    if N > 512 and str(N) in c.get("ranks_by_N", {}):      # rank requirement grows with N
                        cc["rank_noise"], cc["rank_damp"] = c["ranks_by_N"][str(N)]
                    plan.append(("pppm_lanczos", N, law, cc, c["tag"]))
                    if c.get("time_direct"):
                        plan.append(("direct_lanczos", N, law, cc, "direct_" + c["tag"] + "_ranks"))
    for method, N, law, params, tag in plan:
        if (method, N, law, tag) in done or (method == "dense" and N > args.dense_max):
            continue
        if method == "direct_lanczos" and (N > args.direct_max or (N > 512 and tag != "direct_base_ranks")):
            continue
        if tag == "baseline_images" and N > args.images_max:
            continue
        reps = args.reps if not (method == "dense" and N >= 512) else max(2, args.reps // 2)
        cmd = [sys.executable, str(Path(__file__)), "--stage", "_timing_case", "--case",
               json.dumps(dict(method=method, N=N, law=law, params=params, reps=reps, warm=args.warm))]
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        outp = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if outp.returncode != 0:
            print("FAILED", method, N, law, tag, outp.stderr[-800:], flush=True)
            continue
        r = json.loads(outp.stdout.strip().splitlines()[-1])
        r["tag"] = tag
        r["threads"] = dict(OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1)
        r["cpu"] = os.uname().machine + " " + (open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(" :\t")
                                                 if Path("/proc/cpuinfo").exists() else "")
        res.append(r)
        save("timing.json", res)
        print(f"{method:15s} {tag:28s} N={N:5d} {law}: pre {r['precompute']:.2f}s step {1e3 * r['step_median']:.1f} ms "
              f"Gv {('%.2f ms' % (1e3 * r['gamma_v'])) if r['gamma_v'] else '-'} RSS {r['peak_rss_mb']:.0f} MB", flush=True)


# ----------------------------------------------------------------------------
# summary: rank choice per candidate and the final table
# ----------------------------------------------------------------------------
def choose_ranks(rows, tol):
    """Smallest noise / damping ranks whose Krylov errors (noise sqrt-action, damping action and the combined O-step,
    all against dense f(Gamma_h)) are <= tol at every tested N, with all Ritz values positive."""
    rn = next((r for r in RANKS_N if all(R["noise"][str(r)]["krylov"] <= tol and not R["noise"][str(r)].get("negative_ritz")
                                         for R in rows.values())), None)
    rd = next((r for r in RANKS_D if all(R["damp"][str(r)]["krylov"] <= tol and not R["damp"][str(r)].get("negative_ritz")
                                         for R in rows.values())), None)
    if rn is None or rd is None:
        return None
    while not all(R["ostep"][f"n{rn}_d{rd}"]["krylov"] <= tol for R in rows.values()):
        bigger = [r for r in RANKS_N if r > rn]
        if not bigger:
            return None
        rn = bigger[0]
    return rn, rd


def stage_summary(args):
    search, ranks = load("search.json"), load("ranks.json")
    table = []
    for key, R in ranks.items():
        law, tname, c = R["law"], R["tol"], R["params"]
        tol = TOLS.get(tname, TOLS["2e-7"])
        rows = R["by_N"]
        chosen = {t: choose_ranks(rows, TOLS[t]) for t in TOLS}
        for t, rr in chosen.items():
            if rr is None or max(rows[n]["op_err"] for n in rows) > TOLS[t]:
                continue
            rn, rd = rr
            cost = {}
            for n in (64, 512):
                src = (search["baseline"][law] if tname == "baseline" else
                       next(x for x in search[f"{law}_{tname}"]["candidates"]
                            if x.get("eta") == c["eta"] and x["xi"] == c["xi"] and x["p"] == c["p"]))
                cst = src.get(f"cost_N{n}")
                if cst:
                    cost[f"est_step_N{n}"] = cst["build"] + (rn + rd) * cst["matvec"]
            table.append(dict(law=law, budget=t, source=key, xi=c["xi"], s=c["s"], eta=c["eta"], p=c["p"],
                              rank_noise=rn, rank_damp=rd,
                              op_err_max=max(rows[n]["op_err"] for n in rows),
                              gv_err_max=max(rows[n]["gv_err"] for n in rows),
                              noise_krylov=max(rows[n]["noise"][str(rn)]["krylov"] for n in rows),
                              damp_krylov=max(rows[n]["damp"][str(rd)]["krylov"] for n in rows),
                              ostep_krylov=max(rows[n]["ostep"][f"n{rn}_d{rd}"]["krylov"] for n in rows),
                              ostep_total=max(rows[n]["ostep"][f"n{rn}_d{rd}"]["total"] for n in rows),
                              sym_max=max(rows[n]["sym_resid"] for n in rows),
                              null_max=max(rows[n]["null_resid"] for n in rows),
                              lam_min=min(rows[n]["lam_min"] for n in rows),
                              min_ritz=min(rows[n]["min_ritz"] for n in rows),
                              tested_N=sorted(int(n[1:]) for n in rows), **cost))
    table.sort(key=lambda r: (r["law"], r["budget"], r.get("est_step_N512", np.inf)))
    save("rank_choice.json", table)
    for r in table:
        print(f"{r['law']} {r['budget']:5s} {r['source']:36s} n{r['rank_noise']:>2d}/d{r['rank_damp']:>2d} "
              f"op {r['op_err_max']:.1e} ostep {r['ostep_krylov']:.1e}/{r['ostep_total']:.1e} "
              f"est N64 {1e3 * r.get('est_step_N64', np.nan):.1f} ms N512 {1e3 * r.get('est_step_N512', np.nan):.1f} ms")


def stage_table(args):
    """Final table: original vs recommended parameters, errors, measured step times and speedups (timing.json),
    fixed-time dynamics differences (validate.json). Writes cost_table.csv and cost_table.md."""
    import csv
    C = load("candidates.json")
    T = load("timing.json") if (OUT / "timing.json").exists() else []
    V = load("validate.json") if (OUT / "validate.json").exists() else []
    S = load("slowmodes.json")
    RL = load("ranks_large.json")
    step = {(r["tag"], r["N"], r["law"]): r for r in T}
    rows = []
    Ns = sorted({r["N"] for r in T})
    for c in C:
        law, tag = c["law"], c["tag"]
        st64, st512 = S.get(f"{law}_{tag}_N64", {}).get("static", {}), S.get(f"{law}_{tag}_N512", {}).get("static", {})
        v = [r for r in V if r["tag"] == tag]
        row = dict(law=law, budget=c["budget"],
                   original="xi 0.7, s 4.10, eta 0.7, p 7, ranks 40/16, image-enumerated pairs",
                   recommended=f"xi {c['xi']}, s {c['s']}, eta {c['eta']:.4f}, p {c['p']}, ranks {c['rank_noise']}/{c['rank_damp']} (N<=512), KD-tree pairs",
                   ranks_N1728="%d/%d" % tuple(c["ranks_by_N"]["1728"]), ranks_N4096="%d/%d" % tuple(c["ranks_by_N"]["4096"]),
                   M_N64=st64.get("M"), M_N512=st512.get("M"),
                   op_err_N64=st64.get("op_err"), op_err_N512=st512.get("op_err"),
                   gv_rows_err_N1728=RL.get(f"{law}_{tag}_N1728", {}).get("gv_rows_err"),
                   gv_rows_err_N4096=RL.get(f"{law}_{tag}_N4096", {}).get("gv_rows_err"),
                   dyn_validated=bool(v),
                   dyn_max_abs_rel_mean_diff=max((abs(r["rel_mean_diff"]) for r in v), default=None),
                   dyn_all_ci_include_zero=all(r["ci95_low"] <= 0 <= r["ci95_high"] for r in v) if v else None)
        for N in Ns:
            cand, b_img, b_tree = step.get((tag, N, law)), step.get(("baseline_images", N, law)), step.get(("baseline_tree", N, law))
            if cand:
                row[f"step_ms_N{N}"] = 1e3 * cand["step_median"]
                orig = b_img or b_tree
                row[f"speedup_vs_original_N{N}"] = orig["step_median"] / cand["step_median"] if orig else None
                row[f"speedup_vs_baseline_tree_N{N}"] = b_tree["step_median"] / cand["step_median"] if b_tree else None
                row[f"rss_mb_N{N}"] = cand["peak_rss_mb"]
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "cost_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    base = []
    for N in Ns:
        for law in ("A", "B"):
            for tag in ("dense", "direct_base_ranks", "baseline_images", "baseline_tree"):
                r = step.get((tag, N, law))
                if r:
                    base.append(f"| {law} | {N} | {tag} | {r['precompute']:.3f} | {1e3 * r['step_median']:.1f} "
                                f"({1e3 * r['step_min']:.1f}-{1e3 * r['step_max']:.1f}) | {1e3 * r['operator_build']:.1f} | "
                                f"{('%.2f' % (1e3 * r['gamma_v'])) if r['gamma_v'] else '-'} | {r['peak_rss_mb']:.0f} |")
    md = ["| law | N | method | precompute s | step ms median (min-max) | operator build ms | Gamma v ms | peak RSS MB |",
          "|---|---|---|---|---|---|---|---|"] + base
    for N in Ns:
        for c in C:
            r = step.get((c["tag"], N, c["law"]))
            if r:
                md.append(f"| {c['law']} | {N} | {c['tag']} (ranks {r['params']['rank_noise']}/{r['params']['rank_damp']}) | "
                          f"{r['precompute']:.3f} | {1e3 * r['step_median']:.1f} ({1e3 * r['step_min']:.1f}-{1e3 * r['step_max']:.1f}) | "
                          f"{1e3 * r['operator_build']:.1f} | {1e3 * r['gamma_v']:.2f} | {r['peak_rss_mb']:.0f} |")
            r = step.get(("direct_" + c["tag"] + "_ranks", N, c["law"]))
            if r:
                md.append(f"| {c['law']} | {N} | direct sum, ranks of {c['tag']} | {r['precompute']:.3f} | "
                          f"{1e3 * r['step_median']:.1f} ({1e3 * r['step_min']:.1f}-{1e3 * r['step_max']:.1f}) | "
                          f"{1e3 * r['operator_build']:.1f} | {1e3 * r['gamma_v']:.2f} | {r['peak_rss_mb']:.0f} |")
    (OUT / "timing_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    for r in rows:
        print(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["profile", "search", "ranks", "ranks_large", "slowmodes", "modes", "refine", "table", "validate", "timing", "summary", "_timing_case"])
    ap.add_argument("--N", type=int, nargs="+", default=[64, 512])
    ap.add_argument("--laws", nargs="+", default=["A", "B"])
    ap.add_argument("--tols", nargs="+", default=list(TOLS))
    ap.add_argument("--timing-N", type=int, nargs="+", default=[64, 512])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--vectors", type=int, default=4)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--states", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    ap.add_argument("--paths", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dense-max", type=int, default=512)
    ap.add_argument("--direct-max", type=int, default=1024)
    ap.add_argument("--images-max", type=int, default=512)
    ap.add_argument("--case", default=None)
    ap.add_argument("--pair-search", default="tree", choices=["tree", "images"])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage == "_timing_case":
        c = json.loads(args.case)
        print(json.dumps(timing_case(c["method"], c["N"], c["law"], c["params"], c["reps"], c["warm"])))
        return
    t0, c0 = time.time(), time.process_time()
    {"profile": stage_profile, "search": stage_search, "ranks": stage_ranks, "ranks_large": stage_ranks_large, "slowmodes": stage_slowmodes, "modes": stage_modes, "table": stage_table, "refine": stage_refine,
     "validate": stage_validate,
     "timing": stage_timing, "summary": stage_summary}[args.stage](args)
    log = OUT / "budget_log.json"
    entries = json.loads(log.read_text()) if log.exists() else []
    ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    entries.append(dict(stage=args.stage, argv=sys.argv[1:], wall_s=time.time() - t0, cpu_s_main=time.process_time() - c0,
                        cpu_s_children=ch.ru_utime + ch.ru_stime))
    log.write_text(json.dumps(entries, indent=1) + "\n")


if __name__ == "__main__":
    main()
