#!/usr/bin/env python3
"""
Short benchmark: where does PPPM/FFT beat the direct full-periodic sum, and by how much?

All three methods use the same physical model:
- pure longitudinal friction, full periodic images, k = 0 retained;
- g(r_ref) = 0.5, kappa = 0.7, r_ref = 1.3, kT = 0.7, m = 1, dt = 0.005, density 64/5.5^3;
- the same BAOAB step (toy_dynamics.run), neighbour-list double-well forces evaluated once per step, the same
  noise stream.

Methods:
  dense    v2 LatticeFriction matrix (near images + Chebyshev far field, k = 0) rebuilt every step + eigh;
           exact f(Gamma) on Range(Pi) (the 3 translation eigenvectors dropped).
  direct   the same v2 pair tensors for all pairs, built once per O step and reused by every Lanczos action
           (toy_cost_opt.DirectLatticeOp), + Lanczos (test_true_dynamics.lanczos_apply).
  pppm     production operator of toy_configs.costopt_hiacc (KD-tree real space + PPPM mesh, k = 0) rebuilt
           every step + the same Lanczos routine.
direct and pppm use the same noise / damping ranks:
- N <= 512: the costopt_hiacc ranks;
- N = 1728: ranks chosen here by the same rule (budget/10 on every test state).

Stages (outputs in fft_vs_direct_results/):
  states    test configurations per N (v2 / replicated / lattice and the saved liquid-like clustered states),
            structure stats, sha256
  accuracy  per law, N, state: direct matvec vs dense reference; PPPM spectral operator error; Lanczos noise and
            damping errors for a rank grid (overall, k = 2 pi/L, lowest 3 / 12 eigenmodes), direct against exact
            f(Gamma_ref), PPPM against f(Gamma_h) (dense for N <= 512, converged rank-96 / 32 Lanczos for 1728); the
            lean dense O step against the reference DenseRef implementation
  timing    one subprocess per (method, law, N, state), single thread: one-off initialisation, then warm-up and
            timed BAOAB steps (raw per-step CSV), Gamma v after a build, peak RSS
  report    summary CSV / JSON, T(direct)/T(pppm), figures
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import radial_kernels as rk  # noqa: E402
import toy_configs as tc  # noqa: E402
import toy_cost_opt as tco  # noqa: E402
import toy_dynamics as td  # noqa: E402
from test_lanczos_fdt import DenseRef, lanczos, proj, tridiag  # noqa: E402
from test_true_dynamics import lanczos_apply  # noqa: E402

OUT = HERE / "fft_vs_direct_results"
STATES = OUT / "states"
M = tc.MODEL
DT = tc.DT_VALIDATED
RANKS_N = (8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80)
RANKS_D = (3, 4, 5, 6, 8, 10, 12, 16)
REF_RANKS = (96, 32)                 # converged Lanczos reference of Gamma_h at N > 512
NVEC = 4


def f_noise(l):
    return np.sqrt(M["kT"] * M["mass"] * -np.expm1(-2 * DT * l / M["mass"]))


def f_damp(l):
    return np.exp(-DT * l / M["mass"])


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def load(name):
    return json.loads((OUT / name).read_text())


def env_info():
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next(ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor()
    import scipy
    return dict(cpu=cpu, cores=len(os.sched_getaffinity(0)), python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, threads={v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS",
                                                                                   "OMP_NUM_THREADS", "MKL_NUM_THREADS")})


# ----------------------------------------------------------------------------
# states
# ----------------------------------------------------------------------------
def state_names(N):
    k = round((N / 64) ** (1 / 3))
    first = "v2_state" if N == 64 else ("replicated_v2" if 64 * k ** 3 == N else "jittered_lattice")
    return [first, "liquid_clustered"]


def state_file(N, law, name):
    return STATES / f"N{N}_{name}{'_' + law if name != 'liquid_clustered' else ''}.npz"


def stage_states(args):
    import verify_production_configs as vpc                       # liquid-like states (cached, hash-named)
    STATES.mkdir(parents=True, exist_ok=True)
    man = {}
    for N in args.N:
        L = tc.box_length(N)
        for law in ("A", "B"):
            for name in state_names(N):
                f = state_file(N, law, name)
                if not f.exists():
                    q = vpc.liquid_state(N) if name == "liquid_clustered" else tco.config(N, law)[0]
                    np.savez(f, q=q % L)
                with np.load(f) as d:
                    q = d["q"]
                x = q % L
                Sk = []
                for n in np.array([[a, b, c] for a in range(0, 3) for b in range(-2, 3) for c in range(-2, 3)]):
                    if 0 < n @ n <= 4 and (n[0] > 0 or (n[0] == 0 and (n[1] > 0 or (n[1] == 0 and n[2] > 0)))):
                        Sk.append(np.abs(np.exp(1j * x @ (2 * np.pi / L * n)).sum()) ** 2 / N)
                from scipy.spatial import cKDTree
                nb = np.array([len(v) - 1 for v in cKDTree(x, boxsize=L).query_ball_point(x, 1.6)])
                _, _, rmin = td.conservative_force_neighbor(q, L, "double_well")
                man[f"N{N}_{law}_{name}"] = dict(file=str(f.relative_to(HERE)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                                                 N=N, law=law, state=name, S_k_max=float(max(Sk)),
                                                 neighbours_1p6_mean=float(nb.mean()), rmin=float(rmin),
                                                 origin=("scalar-Langevin double-well liquid, conservative forces only, "
                                                         "t = 50 from fcc/sc (verify_production_configs.liquid_state)"
                                                         if name == "liquid_clustered" else
                                                         "toy_cost_opt.config(N, law): v2 state (N=64), 2x2x2 / 3x3x3 "
                                                         "replica + 0.02 jitter, or jittered lattice (N=256)"))
                print(f"N={N} {law} {name}: S(k)max {man[f'N{N}_{law}_{name}']['S_k_max']:.1f}, neighbours<1.6 "
                      f"{nb.mean():.1f}, rmin {rmin:.3f}")
    save("states.json", man)


def load_state(N, law, name):
    with np.load(state_file(N, law, name)) as d:
        return d["q"].copy()


# ----------------------------------------------------------------------------
# operators
# ----------------------------------------------------------------------------
def pppm_params(law, N):
    """Production PPPM set; for N > 512 (refused by the production config) the same set, ranks given here."""
    return dict(tc.CONFIGS["costopt_hiacc"]["pppm"][law])


def prod_ranks(law, N):
    if N <= 512:
        r = tc.resolve("costopt_hiacc", law, N, "double_well", allow_unverified=True)   # N=108: the N<=256 row
        return r["rank_noise"], r["rank_damp"]
    sel = load("ranks_selected.json")
    return tuple(sel[f"{law}_N{N}"]["ranks"])


def make_fast(law, L):
    pp = pppm_params(law, None)
    K = rk.toy_kernel(law, M["gamma"], M["kappa"], M["r_ref"], pp["xi"])
    return rk.FastFriction(L, K, pp["s"], pp["eta"], pp["p"], pair_search="tree")


class LeanDense:
    """f(Gamma) on Range(Pi) from eigh of the full 3N x 3N matrix; the three null (translation) eigenvectors, the
    three eigenvalues closest to 0 of a PSD matrix whose Range(Pi) spectrum starts at >= 0.2, are dropped."""

    def __init__(self, G):
        w, V = np.linalg.eigh(0.5 * (G + G.T))
        self.null = float(np.abs(w[:3]).max() / w[-1])
        self.w, self.V = w[3:], V[:, 3:]

    def apply(self, fun, v):
        return self.V @ (fun(self.w) * (self.V.T @ v))


# ----------------------------------------------------------------------------
# accuracy
# ----------------------------------------------------------------------------
def krylov_table(op, vecs, ref_fun, ranks, fun, ph, low_basis, smax=None):
    """Errors of Lanczos f(Gamma)v per rank against ref_fun(v): overall, k = 2pi/L, lowest 3 / 12 eigenmodes."""
    N = op.N
    tab = {r: dict(overall=0.0, kmin=0.0, low3=0.0, low12=0.0) for r in ranks}
    conv = 0.0
    for v in vecs:
        Lz = lanczos(op, v, smax or max(ranks))
        ref = ref_fun(v, Lz)
        Jr = ph.T @ ref.reshape(N, 3)
        for r in ranks:
            s = min(r, Lz["s"])
            th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], s))
            if th[0] <= 0:
                raise ValueError("non-positive Ritz value")
            d = proj(Lz["nz"] * (Lz["Q"][:s].T @ (U @ (fun(th) * U[0])))) - ref
            e = tab[r]
            e["overall"] = max(e["overall"], float(np.linalg.norm(d) / np.linalg.norm(ref)))
            e["kmin"] = max(e["kmin"], float(np.max(np.linalg.norm(ph.T @ d.reshape(N, 3), axis=1)
                                                    / np.linalg.norm(Jr, axis=1))))
            for n_low, nm in ((3, "low3"), (12, "low12")):
                B = low_basis[:, :n_low]
                e[nm] = max(e[nm], float(np.linalg.norm(B.T @ d) / np.linalg.norm(B.T @ ref)))
    return {str(r): v for r, v in tab.items()}


def lanczos_converged(fun, smax, sconv):
    def ref(v, Lz):
        out = {}
        for s in (sconv, smax):
            ss = min(s, Lz["s"])
            th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], ss))
            out[s] = proj(Lz["nz"] * (Lz["Q"][:ss].T @ (U @ (fun(th) * U[0]))))
        ref.conv = max(getattr(ref, "conv", 0.0), float(np.linalg.norm(out[sconv] - out[smax]) / np.linalg.norm(out[smax])))
        return out[smax]
    return ref


def accuracy_case(law, N, name):
    t_cpu = time.process_time()
    q = load_state(N, law, name)
    L = tc.box_length(N)
    x = q % L
    lat = tco.lattice(L, law)
    t0 = time.process_time()
    Gr = tco.lattice_matrix(lat, x)
    Dr = LeanDense(Gr)
    t_dense = time.process_time() - t0
    T = np.kron(np.ones((N, 1)), np.eye(3))
    rec = dict(law=law, N=N, state=name, lam_min=float(Dr.w[0]), lam_max=float(Dr.w[-1]), dense_null=Dr.null,
               dense_null_space_resid=float(np.abs(Gr @ T).max() / np.abs(Gr).max()),
               dense_sym=float(np.abs(Gr - Gr.T).max() / np.abs(Gr).max()))
    rng = np.random.default_rng([23, N])
    zs = [proj(rng.standard_normal(3 * N)) for _ in range(NVEC)]
    ps = [proj(rng.normal(0, np.sqrt(M["kT"] * M["mass"]), 3 * N)) for _ in range(NVEC)]
    if N <= 512:                                                     # lean dense vs the reference implementation
        R = DenseRef(Gr, N)
        rec["lean_dense_vs_DenseRef"] = max(float(np.linalg.norm(Dr.apply(f, v) - R.apply(f, v)) / np.linalg.norm(R.apply(f, v)))
                                            for f, vs in ((f_noise, zs), (f_damp, ps)) for v in vs)
    ph = np.exp(1j * 2 * np.pi / L * x)
    # direct: operator identical to the reference; Krylov errors against exact f(Gamma_ref)
    t0 = time.process_time()
    od = tco.DirectLatticeOp(lat, x)
    rec["direct_matvec_vs_dense"] = max(float(np.linalg.norm(od.matvec(v) - Gr @ v) / np.linalg.norm(Gr @ v)) for v in zs)
    rec["direct_noise"] = krylov_table(od, zs, lambda v, Lz: Dr.apply(f_noise, v), RANKS_N, f_noise, ph, Dr.V)
    rec["direct_damp"] = krylov_table(od, ps, lambda v, Lz: Dr.apply(f_damp, v), RANKS_D, f_damp, ph, Dr.V)
    t_direct = time.process_time() - t0
    # pppm: operator error and Krylov errors against f(Gamma_h)
    t0 = time.process_time()
    fast = make_fast(law, L)
    op = fast.gamma_h(q)
    rec["pppm_mesh"] = fast.record()
    rec["pppm_op_err"] = tco.spectral_err_matfree(op, Gr, float(Dr.w[-1]))
    if N <= 512:
        Gh = op.dense()
        rec["pppm_op_err_dense"] = float(np.linalg.norm(Gh - Gr, 2) / np.linalg.norm(Gr, 2))
        Dh = LeanDense(Gh)
        rec["pppm_noise"] = krylov_table(op, zs, lambda v, Lz: Dh.apply(f_noise, v), RANKS_N, f_noise, ph, Dh.V)
        rec["pppm_damp"] = krylov_table(op, ps, lambda v, Lz: Dh.apply(f_damp, v), RANKS_D, f_damp, ph, Dh.V)
        rec["pppm_reference"] = "dense f(Gamma_h)"
    else:
        rn, rd = lanczos_converged(f_noise, *REF_RANKS[:1], 80), lanczos_converged(f_damp, REF_RANKS[1], 24)
        rec["pppm_noise"] = krylov_table(op, zs, rn, RANKS_N, f_noise, ph, Dr.V, smax=REF_RANKS[0])
        rec["pppm_damp"] = krylov_table(op, ps, rd, RANKS_D, f_damp, ph, Dr.V, smax=REF_RANKS[1])
        rec["pppm_reference"] = (f"converged Lanczos rank {REF_RANKS} (rank 80 / 24 difference "
                                 f"{max(rn.conv, rd.conv):.1e}); low-mode subspaces of Gamma_ref")
        rec["pppm_reference_conv"] = max(rn.conv, rd.conv)
    # total error of the PPPM O-step functions against exact f(Gamma_ref) (operator + Krylov), at the top rank
    rec["pppm_total_vs_ref_top_rank"] = max(
        float(np.linalg.norm(lanczos_apply(op, v, rk_, f)[0] - Dr.apply(f, v)) / np.linalg.norm(Dr.apply(f, v)))
        for f, vs, rk_ in ((f_noise, zs, max(RANKS_N)), (f_damp, ps, max(RANKS_D))) for v in vs)
    t_pppm = time.process_time() - t0
    rec["cpu_s"] = dict(dense_reference=t_dense, direct_checks=t_direct, pppm_checks=t_pppm,
                        total=time.process_time() - t_cpu)
    return rec


def worst(tab, rank):
    return max(tab[str(rank)].values())


def stage_accuracy(args):
    db = load("accuracy.json") if (OUT / "accuracy.json").exists() else {}
    for N in args.N:
        for law in ("A", "B"):
            for name in state_names(N):
                key = f"{law}_N{N}_{name}"
                if key in db:
                    continue
                db[key] = accuracy_case(law, N, name)
                save("accuracy.json", db)
                e = db[key]
                print(f"[{key}] lam [{e['lam_min']:.3f}, {e['lam_max']:.1f}] direct matvec {e['direct_matvec_vs_dense']:.1e} "
                      f"pppm op {e['pppm_op_err']:.2e} | cpu {e['cpu_s']['total']:.0f} s", flush=True)
    # rank selection for N > 512 (the production config has no rows there): same rule as production, budget/10 on
    # every state, judged on the direct method against exact f(Gamma_ref); then both methods use them
    sel = load("ranks_selected.json") if (OUT / "ranks_selected.json").exists() else {}
    for N in [n for n in args.N if n > 512]:
        for law in ("A", "B"):
            tol = tc.CONFIGS["costopt_hiacc"]["operator_budget"][law] / 10
            es = [db[f"{law}_N{N}_{s}"] for s in state_names(N)]
            rn = next((r for r in RANKS_N if all(worst(e["direct_noise"], r) <= tol for e in es)), None)
            rd = next((r for r in RANKS_D if all(worst(e["direct_damp"], r) <= tol for e in es)), None)
            if rn is None or rd is None:
                print(f"N={N} {law}: no rank in the grid meets budget/10 -> no timing at this N")
                continue
            sel[f"{law}_N{N}"] = dict(ranks=[rn, rd], rule="smallest grid ranks with all four Krylov error measures "
                                      "<= budget/10 on every state (direct method, exact reference)", budget=tol * 10)
            print(f"N={N} {law}: selected ranks {rn}/{rd}")
    save("ranks_selected.json", sel)
    # pass / fail at the ranks that are timed
    rows = []
    for key, e in db.items():
        law, N = e["law"], e["N"]
        try:
            rn, rd = prod_ranks(law, N)
        except (KeyError, FileNotFoundError):
            continue
        tol = tc.CONFIGS["costopt_hiacc"]["operator_budget"][law]
        row = dict(law=law, N=N, state=e["state"], rank_noise=rn, rank_damp=rd, budget=tol,
                   lam_min=e["lam_min"], lam_max=e["lam_max"], dense_null=e["dense_null"],
                   lean_dense_vs_DenseRef=e.get("lean_dense_vs_DenseRef"),
                   direct_matvec_vs_dense=e["direct_matvec_vs_dense"],
                   direct_noise_err=worst(e["direct_noise"], rn), direct_damp_err=worst(e["direct_damp"], rd),
                   pppm_op_err=e["pppm_op_err"], pppm_noise_err=worst(e["pppm_noise"], rn),
                   pppm_damp_err=worst(e["pppm_damp"], rd), pppm_M=e["pppm_mesh"]["M"],
                   accuracy_cpu_s=e["cpu_s"]["total"])
        row["direct_pass"] = (row["direct_matvec_vs_dense"] <= 1e-12 and row["direct_noise_err"] <= tol
                              and row["direct_damp_err"] <= tol)
        row["pppm_pass"] = row["pppm_op_err"] <= tol and row["pppm_noise_err"] <= tol and row["pppm_damp_err"] <= tol
        row["dense_pass"] = row["dense_null"] <= 1e-10 and (row["lean_dense_vs_DenseRef"] or 0.0) <= 1e-10
        rows.append(row)
    rows.sort(key=lambda r: (r["N"], r["law"], r["state"]))
    with open(OUT / "accuracy_at_timed_ranks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"N={r['N']:5d} {r['law']} {r['state']:18s} ranks {r['rank_noise']}/{r['rank_damp']}: direct "
              f"{r['direct_noise_err']:.1e}/{r['direct_damp_err']:.1e} {'PASS' if r['direct_pass'] else 'FAIL'} | pppm op "
              f"{r['pppm_op_err']:.2e} kry {r['pppm_noise_err']:.1e}/{r['pppm_damp_err']:.1e} "
              f"{'PASS' if r['pppm_pass'] else 'FAIL'} | dense {'PASS' if r['dense_pass'] else 'FAIL'}")


# ----------------------------------------------------------------------------
# timing (one subprocess per case)
# ----------------------------------------------------------------------------
def timing_case(method, law, N, name, reps, warm, n_gv):
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    L = tc.box_length(N)
    q = load_state(N, law, name)
    rn, rd = prod_ranks(law, N) if method != "dense" else (None, None)
    rng = np.random.default_rng(7)
    p = rng.normal(0, np.sqrt(M["kT"] * M["mass"]), q.shape)
    p -= p.mean(axis=0)
    t0 = time.perf_counter()
    if method in ("dense", "direct"):
        lat = tco.lattice(L, law)
    else:
        fast = make_fast(law, L)
    t_init = time.perf_counter() - t0
    rss_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    m, dt = M["mass"], DT
    force, _, _ = td.conservative_force_neighbor(q, L, "double_well")
    rows, op = [], None
    for k in range(warm + reps):
        ts = time.perf_counter()
        xi = rng.standard_normal(q.shape)
        p = p + 0.5 * dt * force
        q = q + 0.5 * dt / m * p
        to = time.perf_counter()
        pm, pp, z = p.mean(axis=0), proj(p.ravel()), proj(xi.ravel())
        tb = time.perf_counter()
        if method == "dense":
            G = tco.lattice_matrix(lat, q % L)
            t_build = time.perf_counter() - tb
            te = time.perf_counter()
            D = LeanDense(G)
            t_eig = time.perf_counter() - te
            damp, noise = D.apply(f_damp, pp), D.apply(f_noise, z)
            n_act = 0
        else:
            op = tco.DirectLatticeOp(lat, q % L) if method == "direct" else fast.gamma_h(q)
            t_build = time.perf_counter() - tb
            t_eig = 0.0
            damp, _ = lanczos_apply(op, pp, rd, f_damp)
            noise, _ = lanczos_apply(op, z, rn, f_noise)
            n_act = op.n_actions
        p = pm[None] + (proj(damp) + proj(noise)).reshape(p.shape)
        t_matfun = time.perf_counter() - tb - t_build
        t_ostep = time.perf_counter() - to
        q = q + 0.5 * dt / m * p
        tf = time.perf_counter()
        force, _, _ = td.conservative_force_neighbor(q, L, "double_well")
        t_force = time.perf_counter() - tf
        p = p + 0.5 * dt * force
        t_step = time.perf_counter() - ts
        rows.append(dict(rep=k, warmup=k < warm, t_step=t_step, t_ostep=t_ostep, t_build=t_build, t_eigh=t_eig,
                         t_matfun=t_matfun, t_force=t_force, n_actions=n_act,
                         t_actions_in_lanczos=(op.t_actions if op is not None and method != "dense" else 0.0)))
    gv = []
    if method != "dense":
        X = np.random.default_rng(1).standard_normal(3 * N)
        for _ in range(3):
            op.matvec(X)
        for _ in range(n_gv):
            t0 = time.perf_counter()
            op.matvec(X)
            gv.append(time.perf_counter() - t0)
    return dict(method=method, law=law, N=N, state=name, L=L, ranks=[rn, rd], t_init=t_init, rows=rows, gv=gv,
                rss_after_imports_mb=rss0, rss_after_init_mb=rss_init,
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                pppm=(fast.record() if method == "pppm" else None), env=env_info())


def stage_timing(args):
    res = load("timing_raw.json") if (OUT / "timing_raw.json").exists() else []
    done = {(r["method"], r["law"], r["N"], r["state"]) for r in res}
    for N in args.N:
        warm, reps = (2, 5) if N <= 512 else (1, 3)
        for law in ("A", "B"):
            for name in state_names(N):
                for method in ("dense", "direct", "pppm"):
                    if (method, law, N, name) in done:
                        continue
                    if method == "dense" and N > 512 and name != state_names(N)[1]:
                        continue                         # dense cost does not depend on the state; one state at 1728
                    w, r_ = (1, 2) if (method == "dense" and N > 512) else (warm, reps)
                    case = dict(method=method, law=law, N=N, name=name, reps=r_, warm=w, n_gv=15 if N <= 512 else 7)
                    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    t0 = time.time()
                    out = subprocess.run([sys.executable, str(Path(__file__)), "--stage", "_case", "--case",
                                          json.dumps(case)], capture_output=True, text=True, env=env)
                    if out.returncode != 0:
                        print("FAILED", case, out.stderr[-1500:], flush=True)
                        continue
                    r = json.loads(out.stdout.strip().splitlines()[-1])
                    r["wall_subprocess_s"] = time.time() - t0
                    res.append(r)
                    save("timing_raw.json", res)
                    T = [x for x in r["rows"] if not x["warmup"]]
                    print(f"{method:6s} {law} N={N:5d} {name:18s}: init {r['t_init']:.2f}s step "
                          f"{1e3 * np.median([x['t_step'] for x in T]):.1f} ms (build {1e3 * np.median([x['t_build'] for x in T]):.1f}"
                          f", O {1e3 * np.median([x['t_ostep'] for x in T]):.1f}) Gv "
                          f"{(1e3 * np.median(r['gv'])) if r['gv'] else float('nan'):.2f} ms peak RSS {r['peak_rss_mb']:.0f} MB",
                          flush=True)
    with open(OUT / "timing_raw.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "law", "N", "state", "rank_noise", "rank_damp", "rep", "warmup", "t_step_s", "t_ostep_s",
                    "t_build_s", "t_eigh_s", "t_matfun_s", "t_force_s", "n_actions", "t_init_s", "peak_rss_mb"])
        for r in res:
            for x in r["rows"]:
                w.writerow([r["method"], r["law"], r["N"], r["state"], *r["ranks"], x["rep"], x["warmup"], x["t_step"],
                            x["t_ostep"], x["t_build"], x["t_eigh"], x["t_matfun"], x["t_force"], x["n_actions"],
                            r["t_init"], r["peak_rss_mb"]])
    with open(OUT / "gamma_v_raw.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "law", "N", "state", "sample", "t_gamma_v_s"])
        for r in res:
            for i, g in enumerate(r["gv"]):
                w.writerow([r["method"], r["law"], r["N"], r["state"], i, g])


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def stage_report(args):
    res = load("timing_raw.json")
    acc = {(r["law"], int(r["N"]), r["state"]): r for r in csv.DictReader(open(OUT / "accuracy_at_timed_ranks.csv"))}
    summ = []
    for r in res:
        T = [x for x in r["rows"] if not x["warmup"]]
        med = lambda k: float(np.median([x[k] for x in T]))  # noqa: E731
        a = acc.get((r["law"], r["N"], r["state"]))
        passed = None if a is None else (a[f"{r['method']}_pass"] == "True")
        summ.append(dict(method=r["method"], law=r["law"], N=r["N"], state=r["state"], rank_noise=r["ranks"][0],
                         rank_damp=r["ranks"][1], reps=len(T), init_s=r["t_init"], step_s=med("t_step"),
                         step_min_s=float(min(x["t_step"] for x in T)), step_max_s=float(max(x["t_step"] for x in T)),
                         ostep_s=med("t_ostep"), build_s=med("t_build"), eigh_s=med("t_eigh"), matfun_s=med("t_matfun"),
                         force_s=med("t_force"), gamma_v_s=float(np.median(r["gv"])) if r["gv"] else None,
                         n_actions=int(T[0]["n_actions"]), peak_rss_mb=r["peak_rss_mb"],
                         rss_after_imports_mb=r["rss_after_imports_mb"], accuracy_pass=passed,
                         pppm_M=(r["pppm"] or {}).get("M")))
    summ.sort(key=lambda s: (s["N"], s["law"], s["state"], s["method"]))
    with open(OUT / "timing_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    by = {(s["method"], s["law"], s["N"], s["state"]): s for s in summ}
    ratios = []
    for (meth, law, N, st), s in by.items():
        if meth != "pppm":
            continue
        d, de = by.get(("direct", law, N, st)), by.get(("dense", law, N, st))
        if d is None:
            continue
        ratios.append(dict(law=law, N=N, state=st, step_direct_over_pppm=d["step_s"] / s["step_s"],
                           ostep_direct_over_pppm=d["ostep_s"] / s["ostep_s"],
                           build_direct_over_pppm=d["build_s"] / s["build_s"],
                           gamma_v_direct_over_pppm=d["gamma_v_s"] / s["gamma_v_s"],
                           step_dense_over_pppm=(de["step_s"] / s["step_s"]) if de else None,
                           both_pass_accuracy=bool(s["accuracy_pass"] and d["accuracy_pass"])))
    ratios.sort(key=lambda r: (r["N"], r["law"], r["state"]))
    with open(OUT / "ratios.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ratios[0]))
        w.writeheader()
        w.writerows(ratios)
    for r in ratios:
        print(f"N={r['N']:5d} {r['law']} {r['state']:18s} T_direct/T_pppm: step {r['step_direct_over_pppm']:.2f} O-step "
              f"{r['ostep_direct_over_pppm']:.2f} build {r['build_direct_over_pppm']:.1f} Gv {r['gamma_v_direct_over_pppm']:.2f}"
              f" | dense/pppm {r['step_dense_over_pppm'] if r['step_dense_over_pppm'] is None else round(r['step_dense_over_pppm'], 2)}"
              f" | accuracy {'PASS' if r['both_pass_accuracy'] else 'FAIL'}")
    make_figures(summ)


def make_figures(summ):
    """Two static figures. Palette: first three categorical slots of the dataviz reference palette (validated
    all-pairs, light surface); method = colour, state = marker fill and line dash; text in text tokens."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    col = {"pppm": "#2a78d6", "direct": "#eb6834", "dense": "#1baf7a"}
    lab = {"pppm": "PPPM/FFT + Lanczos", "direct": "direct periodic sum + Lanczos", "dense": "dense matrix + eigh"}
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF, "font.size": 9})

    def style(ax, title):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(color=GRID, lw=0.8, which="major")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, color=INK, fontsize=10, loc="left")

    ok = lambda s: s.get("accuracy_pass") in (True, "True")  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, law in zip(axes, ("A", "B")):
        for meth in ("dense", "direct", "pppm"):
            for clustered in (False, True):
                S_all = sorted([s for s in summ if s["method"] == meth and s["law"] == law
                                and (s["state"] == "liquid_clustered") == clustered], key=lambda s: s["N"])
                S = [s for s in S_all if ok(s)]                  # curves: accuracy-passing cases only
                bad = [s for s in S_all if not ok(s)]
                if bad:
                    ax.plot([s["N"] for s in bad], [s["step_s"] for s in bad], linestyle="none", marker="X", ms=11,
                            mew=1.5, mec=SURF, color=col[meth], zorder=5,
                            label=f"{lab[meth]}, lattice state: fails accuracy check (excluded from curve)")
                    for s in bad:
                        ax.annotate("fails accuracy\n(excluded)", (s["N"], s["step_s"]), xytext=(14, -22),
                                    textcoords="offset points", ha="left", fontsize=7, color=INK2,
                                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
                if not S:
                    continue
                ax.plot([s["N"] for s in S], [s["step_s"] for s in S], "--" if clustered else "-", lw=2,
                        solid_capstyle="round", marker="o", ms=7, mew=1.5, mec=col[meth] if clustered else SURF,
                        mfc=SURF if clustered else col[meth], color=col[meth],
                        label=f"{lab[meth]}, {'clustered' if clustered else 'v2 / replica / lattice'} state")
                if clustered:
                    ax.annotate(lab[meth].split(" + ")[0], (S[-1]["N"], S[-1]["step_s"]), xytext=(6, 0),
                                textcoords="offset points", va="center", fontsize=8, color=INK2)
        style(ax, f"Law {law}: one full BAOAB step, single thread (median)")
        ax.set_xlabel("N (density 64/5.5³)")
        ax.set_xlim(right=ax.get_xlim()[1] * 2.2)
    axes[0].set_ylabel("seconds per step (log)")
    axes[0].legend(fontsize=7, loc="upper left", frameon=False, handlelength=4.5)
    hb, lb = axes[1].get_legend_handles_labels()
    keep = [i for i, t in enumerate(lb) if "fails accuracy" in t]
    if keep:
        axes[1].legend([hb[i] for i in keep], [lb[i] for i in keep], fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "step_time_vs_N.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, law in zip(axes, ("A", "B")):
        for meth, key, ls, name in (("direct", "build_s", "-", "direct: pair-tensor build per step"),
                                    ("direct", "gamma_v_s", ":", "direct: one Γv after build"),
                                    ("pppm", "build_s", "-", "PPPM: build per step (pairs, stencil, degree)"),
                                    ("pppm", "gamma_v_s", ":", "PPPM: one Γv after build"),
                                    ("dense", "build_s", "-", "dense: matrix build per step"),
                                    ("dense", "eigh_s", "-.", "dense: eigendecomposition per step")):
            S = sorted([s for s in summ if s["method"] == meth and s["law"] == law and s["state"] == "liquid_clustered"
                        and s.get(key)], key=lambda s: s["N"])
            if S:
                ax.plot([s["N"] for s in S], [s[key] for s in S], ls, lw=2, marker="o", ms=6, mec=SURF, mew=1.2,
                        color=col[meth], label=name)
        style(ax, f"Law {law}: cost components, clustered state (median)")
        ax.set_xlabel("N (density 64/5.5³)")
    axes[0].set_ylabel("seconds (log)")
    axes[0].legend(fontsize=7, loc="upper left", frameon=False, handlelength=4.5)
    fig.tight_layout()
    fig.savefig(OUT / "build_vs_gamma_v.png", dpi=130)
    plt.close(fig)


def load_summary():
    rows = list(csv.DictReader(open(OUT / "timing_summary.csv")))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("method", "law", "state", "accuracy_pass"):
                continue
            r[k] = None if v in ("", "None") else (int(v) if k in ("N", "rank_noise", "rank_damp", "reps", "n_actions",
                                                                    "pppm_M") else float(v))
    return rows


def stage_figures(args):
    make_figures(load_summary())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["states", "accuracy", "timing", "report", "figures", "_case"])
    ap.add_argument("--N", type=int, nargs="+", default=[64, 256, 512])
    ap.add_argument("--case")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage == "_case":
        c = json.loads(args.case)
        print(json.dumps(timing_case(c["method"], c["law"], c["N"], c["name"], c["reps"], c["warm"], c["n_gv"]),
                         default=float))
        return
    t0, c0 = time.time(), time.process_time()
    {"states": stage_states, "accuracy": stage_accuracy, "timing": stage_timing, "report": stage_report,
     "figures": stage_figures}[args.stage](args)
    log = OUT / "budget_log.json"
    L = json.loads(log.read_text()) if log.exists() else []
    ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    L.append(dict(stage=args.stage, argv=sys.argv[1:], wall_s=time.time() - t0, cpu_s_main=time.process_time() - c0,
                  cpu_s_children=ch.ru_utime + ch.ru_stime, env=env_info()))
    log.write_text(json.dumps(L, indent=1) + "\n")


if __name__ == "__main__":
    main()
