#!/usr/bin/env python3
"""
Section 6 dynamics: ideal-gas conservation and equilibrium (former experiment 6), dynamics reference comparison
(former 7) and time-step check (former 9). One integrator, one set of initial states and noise streams.

Model and step (unchanged): U = 0, periodic, kappa = beta = m = 1, rho = 30/216. One step A(dt/2) O(dt) A(dt/2):
    q_half = q + dt/(2m) p (mod L);  Gamma_h = Gamma_h(q_half)
    p_new  = p_mean + exp(-dt Gamma_h/m) Pi p + f_dt(Gamma_h) Pi xi,   f_dt(l) = sqrt((m/beta)(-expm1(-2 dt l/m)))
    q_new  = q_half + dt/(2m) p_new
The centre-of-mass momentum is copied exactly. Spatial set, dt and production ranks come from sec6_config.json.

Methods (same initial state, same dt, same Gaussian stream):
    dense   eigendecomposition of the assembled Gamma_h on Range(Pi)                    (small N only)
    hi      Lanczos at high ranks (rn_hi, rd_hi); each step also evaluates ranks -20 and -40 of the same recursion
            and records the self-differences, so the reference quality is checked step by step
    prod    Lanczos at the production ranks of sec6_config
Optional spatial variant: --set T9 (fine spatial set) for the spatial-sensitivity comparison.

Stages
  ideal     production runs, N = 64 (16 trajectories) and N = 256 (4 trajectories); O-step inputs (q_half, p, xi) of
            every step are saved (raw/, git-ignored) for the offline matrix-function checks
  offline   dense f(Gamma_h) vs the production ranks on saved O-steps: a regular sample plus the steps with the largest
            spectral range (largest Ritz lambda_max estimate)
  compare   dense and hi runs on the ideal-gas initial states and streams (N = 64), paired with the production runs
  spatial   dense T7 vs dense T9 on the same states and streams (N = 64)
  dtstudy   dense at dt, dt/2 (and dt/4 if requested) to a fixed physical end time with nested streams
            xi_h = S_h^{-1}(R_{h/2} S_{h/2} xi_1 + S_{h/2} xi_2)  (exactly N(0, Pi) for frozen Gamma_h)
  large     short coupled prod vs hi runs at N = 4000 and 16000
  analyze   statistics, tables, figures
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sec6_config  # noqa: E402
from sec6_common import Box, f_noise, f_damp, lanczos, tridiag, proj, DenseRef, MASS, BETA  # noqa: E402

SCRIPT = "sec6_dynamics.py"
OUT_IG = sec6_config.RESULTS / "ideal_gas"
OUT_DY = sec6_config.RESULTS / "dynamics"
SEED_STATE, SEED_NOISE = 20261016, 20261017
T0 = np.array([2.0, 0.5, 0.5])                       # initial component temperatures
VCM = np.array([0.3, -0.2, 0.1])                     # initial centre-of-mass velocity


class NegativeRitzError(Exception):
    pass


# ----------------------------------------------------------------------------
# step
# ----------------------------------------------------------------------------
def lanczos_fun(op, v, rank, fun, backs=(), extra=()):
    """f(Gamma_h) v at the given rank from one recursion; also at rank-b for b in backs (self-check) and at the
    absolute ranks in extra (e.g. the production rank evaluated inside a high-rank reference recursion)."""
    nv = np.linalg.norm(v)
    if nv == 0:
        return np.zeros_like(v), {}, {}
    Lz = lanczos(op, v, rank)
    s = Lz["s"]
    res, ritz = {}, {}
    want = (s,) + tuple(s - b for b in backs if s - b >= 1) + tuple(r for r in extra if 1 <= r < s)
    for r in dict.fromkeys(want):
        th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], r))
        if r == s:
            ritz = dict(min=float(th[0]), max=float(th[-1]), n_neg=int((th < 0).sum()))
            if th[0] < 0:
                raise NegativeRitzError(f"negative Ritz value {th[0]:.3e} at rank {s}")
        res[r] = proj(Lz["nz"] * (Lz["Q"][:r].T @ (U @ (fun(th) * U[0]))))
    out = res[s]
    chk = {f"back{s - r}": float(np.linalg.norm(res[r] - out) / np.linalg.norm(out)) for r in res
           if r != s and r not in extra}
    chk.update({f"rank{r}": float(np.linalg.norm(res[r] - out) / np.linalg.norm(out)) for r in res if r in extra})
    return out, ritz, chk


def o_step(box, qh, p, xi, dt, method, ranks, coarse=None):
    N = box.N
    op = box.gamma(qh)
    pm = p.mean(axis=0)
    pp = proj(p.ravel())
    info = {}
    if method == "dense":
        ref = DenseRef(op.dense(), N)
        U, lam = ref.U, ref.lam
        if coarse is not None:                                   # nested stream for the dt study
            h2 = 0.5 * dt
            fs, fs2, fr2 = f_noise(dt)(lam), f_noise(h2)(lam), f_damp(h2)(lam)
            x1, x2 = U.T @ proj(coarse[0].ravel()), U.T @ proj(coarse[1].ravel())
            xi = (U @ ((fr2 * fs2 * x1 + fs2 * x2) / fs)).reshape(N, 3)
            info["xi_built"] = xi
        damp = U @ (f_damp(dt)(lam) * (U.T @ pp))
        noise = U @ (f_noise(dt)(lam) * (U.T @ proj(xi.ravel())))
        info.update(lam_max=float(lam[-1]), lam_min=float(lam[0]), lam_source="dense")
    else:
        rn, rd = ranks
        rn, rd = min(rn, 3 * N - 3), min(rd, 3 * N - 3)
        backs = (20, 40) if method == "hi" else ()
        pr_n, pr_d = prod_ranks() if method == "hi" else (None, None)
        damp, rit_d, chk_d = lanczos_fun(op, pp, rd, f_damp(dt), backs, (pr_d,) if pr_d else ())
        noise, rit_n, chk_n = lanczos_fun(op, proj(xi.ravel()), rn, f_noise(dt), backs, (pr_n,) if pr_n else ())
        info.update(lam_max=max(rit_d.get("max", 0), rit_n.get("max", 0)),
                    lam_min=min(rit_d.get("min", np.inf), rit_n.get("min", np.inf)), lam_source="Ritz (estimate)",
                    n_actions=op.n_actions, chk_noise=chk_n, chk_damp=chk_d)
    p_new = pm[None, :] + (proj(damp) + proj(noise)).reshape(N, 3)
    return p_new, info


def observables(q, p, L, P0):
    N = len(p)
    P = p.sum(axis=0)
    vcm = P / (N * MASS)
    dv = p / MASS - vcm
    T = MASS * (dv ** 2).sum(axis=0) / (N - 1)                 # N - 1 degrees of freedom per component
    return dict(P=P, dP=float(np.linalg.norm(P - P0) / np.linalg.norm(P0)), T=T, Tkin=float(T.mean()),
                Ok=float(np.mean(np.cos(2 * np.pi / L * q[:, 0]))))


def initial_state(N, L, traj):
    rng = np.random.default_rng([SEED_STATE, N, traj])
    q = rng.uniform(0.0, L, size=(N, 3))
    pth = rng.normal(size=(N, 3)) * np.sqrt(MASS * T0)
    pth -= pth.mean(axis=0)
    return q, pth + MASS * VCM


def noise_stream(N, traj, nsteps, level=1):
    """Gaussian stream of a trajectory. level = refinement factor of the finest grid (1: dt, 2: dt/2, 4: dt/4)."""
    rng = np.random.default_rng([SEED_NOISE, N, traj, level])
    return rng.normal(size=(level * nsteps, N, 3))


def run(box, q0, p0, xis, dt, nsteps, method, ranks=None, coarse_stream=None, save_inputs=False, save_path=False):
    q, p = q0.copy(), p0.copy()
    L = box.L
    P0 = p0.sum(axis=0)
    rec = {k: [] for k in ("t", "T", "Tkin", "Ok", "dP", "lam_max", "lam_min", "n_actions", "t_step",
                           "chk_noise20", "chk_noise40", "chk_damp20", "chk_damp40", "prodrank_noise_err",
                           "prodrank_damp_err")}
    inputs = {k: [] for k in ("qh", "p", "xi")}
    path = {k: [] for k in ("q", "p")}
    built = []

    def push(t, o, info, ts):
        rec["t"].append(t)
        rec["T"].append(o["T"])
        rec["Tkin"].append(o["Tkin"])
        rec["Ok"].append(o["Ok"])
        rec["dP"].append(o["dP"])
        rec["lam_max"].append(info.get("lam_max", np.nan))
        rec["lam_min"].append(info.get("lam_min", np.nan))
        rec["n_actions"].append(info.get("n_actions", np.nan))
        rec["t_step"].append(ts)
        for k in ("noise", "damp"):
            c = info.get(f"chk_{k}", {})
            rec[f"chk_{k}20"].append(c.get("back20", np.nan))
            rec[f"chk_{k}40"].append(c.get("back40", np.nan))
            pr = [v for kk, v in c.items() if kk.startswith("rank")]
            rec[f"prodrank_{k}_err"].append(pr[0] if pr else np.nan)

    push(0.0, observables(q, p, L, P0), {}, np.nan)
    if save_path:
        path["q"].append(q.copy())
        path["p"].append(p.copy())
    for n in range(nsteps):
        t0 = time.perf_counter()
        coarse = (coarse_stream[2 * n], coarse_stream[2 * n + 1]) if coarse_stream is not None else None
        xi = xis[n] if xis is not None else None
        qh = (q + 0.5 * dt / MASS * p) % L
        if save_inputs:
            inputs["qh"].append(qh.copy())
            inputs["p"].append(p.copy())
        p, info = o_step(box, qh, p, xi, dt, method, ranks, coarse)
        if coarse is not None:
            built.append(info["xi_built"])
        if save_inputs:
            inputs["xi"].append(xi.copy() if xi is not None else info["xi_built"].copy())
        q = (qh + 0.5 * dt / MASS * p) % L
        push((n + 1) * dt, observables(q, p, L, P0), info, time.perf_counter() - t0)
        if save_path:
            path["q"].append(q.copy())
            path["p"].append(p.copy())
    out = {k: np.array(v, dtype=float) for k, v in rec.items()}
    if save_inputs:
        out.update({f"in_{k}": np.array(v) for k, v in inputs.items()})
    if save_path:
        out.update({f"path_{k}": np.array(v) for k, v in path.items()})
    return out, (np.array(built) if built else None)


# ----------------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------------
def prod_ranks():
    pr = sec6_config.production()
    if pr["noise_rank"] is None or pr["damping_rank"] is None:
        raise SystemExit("production ranks not set in sec6_config.json (run sec6_lanczos.py select --write-config)")
    return int(pr["noise_rank"]), int(pr["damping_rank"])


def raw_dir(out):
    d = out / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ideal_job(job):
    N, traj, T_end = job
    pr = sec6_config.production()
    dt = pr["dt"]
    box = Box(N, pr["spatial_set"])
    nsteps = int(round(T_end / dt))
    q0, p0 = initial_state(N, box.L, traj)
    xis = noise_stream(N, traj, nsteps)
    t0 = time.perf_counter()
    rec, _ = run(box, q0, p0, xis, dt, nsteps, "prod", prod_ranks(), save_inputs=True, save_path=(N <= 64))
    rec["wall_s"] = time.perf_counter() - t0
    np.savez_compressed(raw_dir(OUT_IG) / f"prod_N{N}_traj{traj:02d}.npz", **rec,
                        meta=json.dumps(dict(N=N, traj=traj, dt=dt, nsteps=nsteps, L=box.L, ranks=prod_ranks(),
                                             box=box.describe(), provenance=sec6_config.provenance(SCRIPT, True))))
    return N, traj, rec["wall_s"], float(np.nanmax(rec["dP"])), rec["T"][-1].tolist()


def _compare_job(job):
    """dense, hi (and optionally T9 dense) on the same initial state and stream as the production run."""
    N, traj, T_end, method, set_name, hi_ranks = job
    pr = sec6_config.production()
    dt = pr["dt"]
    box = Box(N, set_name)
    nsteps = int(round(T_end / dt))
    q0, p0 = initial_state(N, box.L, traj)
    xis = noise_stream(N, traj, nsteps)              # prefix of the production run's stream (same seed)
    t0 = time.perf_counter()
    rec, _ = run(box, q0, p0, xis, dt, nsteps, method, hi_ranks, save_path=True)
    rec["wall_s"] = time.perf_counter() - t0
    tag = f"{method}_{set_name}_N{N}_traj{traj:02d}"
    np.savez_compressed(raw_dir(OUT_DY) / f"cmp_{tag}.npz", **rec,
                        meta=json.dumps(dict(N=N, traj=traj, dt=dt, nsteps=nsteps, method=method, set=set_name,
                                             hi_ranks=hi_ranks, provenance=sec6_config.provenance(SCRIPT, True))))
    return tag, rec["wall_s"]


def _dt_job(job):
    N, traj, T_end, levels = job
    pr = sec6_config.production()
    dt = pr["dt"]
    box = Box(N, pr["spatial_set"])
    nsteps = int(round(T_end / dt))
    q0, p0 = initial_state(N, box.L, traj)
    finest = max(levels)
    stream = noise_stream(N, traj, nsteps, level=finest)          # raw stream on the finest grid
    out = {}
    cur = stream
    for lev in sorted(levels, reverse=True):                     # finest first; coarser levels nest the finer noise
        h = dt / lev
        if lev == finest:
            rec, built = run(box, q0, p0, cur, h, lev * nsteps, "dense", save_path=True)
            built = cur
        else:
            rec, built = run(box, q0, p0, None, h, lev * nsteps, "dense", coarse_stream=cur, save_path=True)
        out[lev] = rec
        cur = built
    np.savez_compressed(raw_dir(OUT_DY) / f"dt_N{N}_traj{traj:02d}.npz",
                        **{f"L{lev}_{k}": v for lev, r in out.items() for k, v in r.items()},
                        meta=json.dumps(dict(N=N, traj=traj, dt=dt, nsteps=nsteps, levels=levels,
                                             provenance=sec6_config.provenance(SCRIPT, True))))
    return traj, {lev: out[lev]["T"][-1].tolist() for lev in out}


def _large_job(job):
    N, traj, nsteps, method, hi_ranks = job
    pr = sec6_config.production()
    dt = pr["dt"]
    box = Box(N, pr["spatial_set"])
    q0, p0 = initial_state(N, box.L, traj)
    xis = noise_stream(N, traj, nsteps)
    ranks = prod_ranks() if method == "prod" else hi_ranks
    t0 = time.perf_counter()
    rec, _ = run(box, q0, p0, xis, dt, nsteps, method, ranks, save_path=True)
    rec["wall_s"] = time.perf_counter() - t0
    np.savez_compressed(raw_dir(OUT_DY) / f"large_{method}_N{N}_traj{traj:02d}.npz", **rec,
                        meta=json.dumps(dict(N=N, traj=traj, dt=dt, nsteps=nsteps, method=method, ranks=ranks,
                                             box=box.describe(), provenance=sec6_config.provenance(SCRIPT, True))))
    return N, traj, method, rec["wall_s"]


# ----------------------------------------------------------------------------
# offline O-step checks
# ----------------------------------------------------------------------------
def _offline_one(job):
    path, steps = job
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    N, dt = meta["N"], meta["dt"]
    pr = sec6_config.production()
    box = Box(N, pr["spatial_set"])
    rn, rd = meta["ranks"]
    rows = []
    for n in steps:
        qh, p, xi = z["in_qh"][n], z["in_p"][n], z["in_xi"][n]
        op = box.gamma(qh)
        ref = DenseRef(op.dense(), N)
        pp, zz = proj(p.ravel()), proj(xi.ravel())
        d_ref, n_ref = ref.apply(f_damp(dt), pp), ref.apply(f_noise(dt), zz)
        d_l, rit_d, _ = lanczos_fun(op, pp, min(rd, 3 * N - 3), f_damp(dt))
        n_l, rit_n, _ = lanczos_fun(op, zz, min(rn, 3 * N - 3), f_noise(dt))
        rows.append(dict(file=Path(path).name, step=int(n), lam_max=ref.lam_max, lam_min=ref.lam_star,
                         tau=dt * ref.lam_max / MASS, ritz_max_noise=rit_n["max"], ritz_min_noise=rit_n["min"],
                         noise_rel_err=float(np.linalg.norm(n_l - n_ref) / np.linalg.norm(n_ref)),
                         damp_rel_err=float(np.linalg.norm(d_l - d_ref) / np.linalg.norm(d_ref))))
    return rows


def offline_steps(z, n_regular=12, n_top=6):
    lm = z["lam_max"][1:]                                     # per-step Ritz estimate recorded during the run
    nst = len(lm)
    reg = list(np.linspace(0, nst - 1, n_regular).round().astype(int))
    top = list(np.argsort(-np.nan_to_num(lm, nan=-1))[:n_top])
    return sorted(set(int(s) for s in reg + top))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["ideal", "offline", "compare", "spatial", "dtstudy", "large", "analyze"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--N", type=int, nargs="*")
    ap.add_argument("--ntraj", type=int, nargs="*")
    ap.add_argument("--T", type=float, nargs="*", help="end time per N")
    ap.add_argument("--hi", type=int, nargs=2, help="hi noise and damping ranks")
    ap.add_argument("--levels", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--nsteps", type=int, default=25)
    ap.add_argument("--trajs", type=int, nargs="*")
    ap.add_argument("--spec", nargs="*", help="ideal stage: N:T_end:first_traj:count entries")
    args = ap.parse_args()
    t0 = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] stage {args.stage}", flush=True)
    if args.stage == "ideal":
        if args.spec:
            jobs = [(int(N), k, float(T)) for e in args.spec for N, T, a, n in [e.split(":")]
                    for k in range(int(a), int(a) + int(n))]
            jobs.sort(key=lambda j: -j[0] * j[2])                        # longest first
        else:
            Ns, nt, Ts = args.N or [64, 256], args.ntraj or [16, 4], args.T or [40.0, 30.0]
            jobs = [(N, k, T) for N, n, T in zip(Ns, nt, Ts) for k in range(n)]
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(_ideal_job, jobs):
                print(f"  ideal N={r[0]} traj={r[1]}: {r[2]:.0f}s max dP {r[3]:.1e} final T {np.round(r[4], 3)}",
                      flush=True)
    elif args.stage == "offline":
        files = sorted(raw_dir(OUT_IG).glob("prod_N*_traj*.npz"))
        jobs = []
        for f in files:
            z = np.load(f)
            jobs.append((str(f), offline_steps(z)))
        rows = []
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(_offline_one, jobs):
                rows += r
                print(f"  offline {r[0]['file']}: noise max {max(x['noise_rel_err'] for x in r):.2e} damp max "
                      f"{max(x['damp_rel_err'] for x in r):.2e} tau max {max(x['tau'] for x in r):.2f}", flush=True)
        OUT_IG.mkdir(parents=True, exist_ok=True)
        (OUT_IG / "offline_ostep_checks.json").write_text(json.dumps(dict(
            provenance=sec6_config.provenance(SCRIPT, True), ranks=prod_ranks(), rows=rows), indent=1) + "\n")
    elif args.stage in ("compare", "spatial"):
        N, T_end = (args.N or [64])[0], (args.T or [6.0])[0]
        trajs = args.trajs if args.trajs is not None else list(range((args.ntraj or [16])[0]))
        if args.stage == "compare":
            jobs = [(N, k, T_end, m, sec6_config.production()["spatial_set"], tuple(args.hi) if m == "hi" else None)
                    for k in trajs for m in ("dense", "hi")]
        else:
            jobs = [(N, k, T_end, "dense", sec6_config.production()["fine_spatial_set"], None) for k in trajs]
        with ProcessPoolExecutor(args.workers) as ex:
            for tag, w in ex.map(_compare_job, jobs):
                print(f"  {tag}: {w:.0f}s", flush=True)
    elif args.stage == "dtstudy":
        N, T_end = (args.N or [64])[0], (args.T or [6.0])[0]
        trajs = args.trajs if args.trajs is not None else list(range((args.ntraj or [16])[0]))
        jobs = [(N, k, T_end, args.levels) for k in trajs]
        with ProcessPoolExecutor(args.workers) as ex:
            for traj, T in ex.map(_dt_job, jobs):
                print(f"  dt traj {traj}: final T {T}", flush=True)
    elif args.stage == "large":
        jobs = []
        for N in args.N or [4000, 16000]:
            for k in (args.trajs if args.trajs is not None else [0]):
                if N >= 16000 and k > 0:                       # one representative trajectory at N = 16000
                    continue
                for m in ("hi", "prod"):
                    jobs.append((N, k, args.nsteps, m, tuple(args.hi)))
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(_large_job, jobs):
                print(f"  large N={r[0]} traj={r[1]} {r[2]}: {r[3]:.0f}s", flush=True)
    else:
        import sec6_dynamics_analyze
        sec6_dynamics_analyze.main()
    print(f"[{time.strftime('%H:%M:%S')}] stage {args.stage} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
