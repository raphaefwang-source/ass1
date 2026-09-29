#!/usr/bin/env python3
"""
First time-dependent test of the PPPM + direct-start Lanczos finite-time FDT thermostat:
ideal gas, U(q) = 0, beta = m = 1.

One step of B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2) with U = 0:
    q_half = q_n + dt/(2m) p_n (mod L);  Gamma_h = Gamma_h(q_half)
    p_new  = p_mean + R_dt Pi p_n + S_dt Pi xi_n,  R_dt = exp(-dt Gamma_h/m),
             S_dt = f_dt(Gamma_h), f_dt(l) = sqrt(beta^{-1} m [-expm1(-2 dt l/m)])
    q_{n+1} = q_half + dt/(2m) p_new (mod L)
Gamma_h is the validated PPPM operator (test_lanczos_fdt.PARAMS); the mesh/influence function
is built once per box and reused, only the particle-dependent parts are rebuilt each step.

Methods (common initial states and common Gaussian streams):
    dense      eigendecomposition of the assembled Gamma_h on Range(Pi)          (reference)
    tight      dense, with a tighter Gamma_h (tol-1e-9 PPPM set)                   (spatial-error proxy)
    lanczos_hi PPPM action + Lanczos, noise rank 90, damping rank 24               (converged Krylov)
    rank40     PPPM action + Lanczos, noise rank 40, damping rank RD (chosen)
    rank48     PPPM action + Lanczos, noise rank 48, damping rank RD
    indep      negative control: correct damping, independent per-particle noise
Temporal refinement: dense at 2dt, dt, dt/2, dt/4 with nested streams
    xi_coarse = S_h^{-1} (R_{h/2} S_{h/2} xi_1 + S_{h/2} xi_2)   (exactly N(0, Pi) for frozen Gamma_h)

Model unchanged; no transverse kernel; no clipping of Ritz values; no potential invented.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_lanczos_fdt import GammaH, PARAMS, lanczos, tridiag, proj, DenseRef, real_pairs  # noqa: E402
from test_parameter_selection import ewald  # noqa: E402
from test_equal_accuracy_xi import make_mesh  # noqa: E402

OUT = HERE / "dynamics_results"
BETA, MASS = 1.0, 1.0
DENSITY = 30 / 216.0
TIGHT = dict(xi=0.7, s=4.628, eta=0.6, p=8)          # accepted at operator tol 1e-9 (equal-accuracy study)
RD = 12                                              # damping rank (static test: <= 1e-10 up to dt lambda_max = 5)
T0 = np.array([2.0, 0.5, 0.5])                        # initial component temperatures
VCM = np.array([0.3, -0.2, 0.1])                      # prescribed centre-of-mass velocity


class NegativeRitz(Exception):
    pass


# ----------------------------------------------------------------------------
# Gamma_h(q) with a reusable mesh
# ----------------------------------------------------------------------------
class Box:
    def __init__(self, N, params=PARAMS):
        self.N, self.L = N, (N / DENSITY) ** (1 / 3)
        self.K = ewald(1.0, params["xi"])
        self.rc, self.kc = params["s"] / params["xi"], 2 * params["xi"] * params["s"]
        self.mesh = make_mesh(self.L, self.K, self.kc, params["eta"], params["p"])

    def gamma(self, x):
        op = GammaH.__new__(GammaH)
        op.x, op.L, op.N, op.rc, op.kc = x, self.L, len(x), self.rc, self.kc
        op.i, op.j, op.d = real_pairs(x, self.L, self.rc)
        r = np.linalg.norm(op.d, axis=1)
        op.w = self.K.theta_S_scaled_closed(r) * np.exp(-(self.K.xi * r) ** 2) / r ** 2
        op.mesh = self.mesh
        op.idx, op.wt = self.mesh.flat(x)
        op.D = self.mesh.degree(op.idx, op.wt)
        op.n_actions, op.t_actions = 0, 0.0
        return op


def f_noise(dt):
    return lambda l: np.sqrt(MASS / BETA * (-np.expm1(-2 * dt * l / MASS)))


def f_damp(dt):
    return lambda l: np.exp(-dt * l / MASS)


def lanczos_apply(op, v, rank, fun):
    """Direct-start Lanczos on Pi v (v already in Range(Pi)); returns projected result, Ritz extrema."""
    nv = np.linalg.norm(v)
    if nv == 0:
        return np.zeros_like(v), (np.nan, np.nan)
    Lz = lanczos(op, v, rank)
    s = Lz["s"]
    th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], s))
    if th[0] <= 0:
        raise NegativeRitz(f"Ritz value {th[0]:.3e}")
    return proj(Lz["nz"] * (Lz["Q"][:s].T @ (U @ (fun(th) * U[0])))), (float(th[0]), float(th[-1]))


# ----------------------------------------------------------------------------
# One integrator step
# ----------------------------------------------------------------------------
def step(box, q, p, xi, dt, method, coarse=None):
    """Returns q_new, p_new, info.  xi: (N,3) standard normal (projected inside).  coarse: (xi1, xi2) for nested
    stream construction (dense only): then xi is replaced by S_dt^{-1}(R_{dt/2} S_{dt/2} xi1 + S_{dt/2} xi2)."""
    N, L = box.N, box.L
    t0 = time.perf_counter()
    qh = (q + 0.5 * dt / MASS * p) % L
    op = box.gamma(qh)
    t_setup = time.perf_counter() - t0
    pm = p.mean(axis=0)
    pp = proj(p.ravel())
    info = {}
    t1 = time.perf_counter()
    if method in ("dense", "tight", "indep"):
        ref = DenseRef(op.dense(), N)
        U, lam = ref.U, ref.lam
        if coarse is not None:
            h2 = 0.5 * dt
            fs, fs2, fr2 = f_noise(dt)(lam), f_noise(h2)(lam), f_damp(h2)(lam)
            x1, x2 = U.T @ proj(coarse[0].ravel()), U.T @ proj(coarse[1].ravel())
            c = (fr2 * fs2 * x1 + fs2 * x2) / fs
            xi = (U @ c).reshape(N, 3)
            info["xi_built"] = xi
        damp = U @ (f_damp(dt)(lam) * (U.T @ pp))
        if method == "indep":
            s2 = float(np.mean(f_noise(dt)(lam) ** 2))           # same average scalar variance, no correlation
            noise = np.sqrt(s2) * xi.ravel()                       # NOT projected: independent per-particle noise
        else:
            noise = U @ (f_noise(dt)(lam) * (U.T @ proj(xi.ravel())))
        info["lam_max"], info["lam_min"] = float(lam[-1]), float(lam[0])
    else:
        rn = {"lanczos_hi": 90, "rank40": 40, "rank48": 48}[method]
        rd = 24 if method == "lanczos_hi" else RD
        rn = min(rn, 3 * N - 3)
        damp, (dmin, dmax) = lanczos_apply(op, pp, rd, f_damp(dt))
        noise, (nmin, nmax) = lanczos_apply(op, proj(xi.ravel()), rn, f_noise(dt))
        info["lam_max"], info["lam_min"] = max(dmax, nmax), min(dmin, nmin)
        info["n_actions"] = op.n_actions
    t_mat = time.perf_counter() - t1
    p_new = (pm[None, :] + (proj(damp) + (noise if method == "indep" else proj(noise))).reshape(N, 3))
    q_new = (qh + 0.5 * dt / MASS * p_new) % L
    info.update(t_setup=t_setup, t_matfun=t_mat, t_total=time.perf_counter() - t0)
    return q_new, p_new, info


def observables(q, p, L, P0):
    N = len(p)
    P = p.sum(axis=0)
    vcm = P / (N * MASS)
    dv = p / MASS - vcm
    T = MASS * (dv ** 2).sum(axis=0) / (N - 1)          # conditional (N - 1 degrees of freedom per component)
    Ok = float(np.mean(np.cos(2 * np.pi / L * q[:, 0])))
    return dict(P=P, dP=float(np.linalg.norm(P - P0) / max(1.0, np.linalg.norm(P0))), vcm=vcm, T=T,
                Tkin=float(T.mean()), Ok=Ok)


def initial_state(N, L, seed):
    rng = np.random.default_rng([20260929, seed])
    q = rng.uniform(0, L, size=(N, 3))
    pth = rng.normal(size=(N, 3)) * np.sqrt(MASS * T0)
    pth -= pth.mean(axis=0)
    p = pth + MASS * VCM
    return q, p, rng


def run(box, q0, p0, xis, dt, nsteps, method, coarse_stream=None, record_every=1, stop=None, snap_from=None, snap_every=10):
    q, p = q0.copy(), p0.copy()
    P0 = p0.sum(axis=0)
    rec = {k: [] for k in ("t", "T", "Tkin", "Ok", "dP", "vcm", "dt_lmax", "lam_min", "t_step", "n_actions")}
    snaps = []
    built = []
    o = observables(q, p, box.L, P0)

    def push(t, o, info):
        rec["t"].append(t)
        rec["T"].append(o["T"].tolist())
        rec["Tkin"].append(o["Tkin"])
        rec["Ok"].append(o["Ok"])
        rec["dP"].append(o["dP"])
        rec["vcm"].append(o["vcm"].tolist())
        rec["dt_lmax"].append(dt * info.get("lam_max", np.nan) / MASS)
        rec["lam_min"].append(info.get("lam_min", np.nan))
        rec["t_step"].append(info.get("t_total", np.nan))
        rec["n_actions"].append(info.get("n_actions", np.nan))

    push(0.0, o, {})
    for n in range(nsteps):
        coarse = (coarse_stream[2 * n], coarse_stream[2 * n + 1]) if coarse_stream is not None else None
        xi = xis[n] if xis is not None else None
        q, p, info = step(box, q, p, xi, dt, method, coarse)
        if coarse is not None:
            built.append(info["xi_built"])
        if (n + 1) % record_every == 0:
            push((n + 1) * dt, observables(q, p, box.L, P0), info)
        if snap_from is not None and n + 1 >= snap_from and (n + 1 - snap_from) % snap_every == 0:
            snaps.append(p - p.mean(axis=0))                      # Pi p (zero-total-momentum part)
        if stop and stop():
            break
    out = {k: np.array(v) for k, v in rec.items()}
    if snaps:
        out["snaps"] = np.array(snaps)
    return out, (np.array(built) if built else None), (q, p)


# ----------------------------------------------------------------------------
def small_study(args, say):
    N = args.N_small
    box, tbox = Box(N), Box(N, TIGHT)
    # dt from the initial configurations: dt lambda_max(q0)/m = 0.2 (median over trajectories)
    lmax0 = []
    for s in range(args.ntraj):
        q0, p0, _ = initial_state(N, box.L, s)
        lmax0.append(DenseRef(box.gamma(q0).dense(), N).lam_max)
    dt = 0.2 * MASS / float(np.median(lmax0))
    nsteps = int(round(args.T_end / dt))
    nsteps -= nsteps % 4                                                # divisible for the 2dt level
    T_end = nsteps * dt
    params = dict(N=N, L=box.L, density=DENSITY, beta=BETA, m=MASS, dt=dt, T_end=T_end, nsteps=nsteps,
                  lam_max_initial=lmax0, damping_rank=RD, ntraj=args.ntraj, T0=T0.tolist(), vcm=VCM.tolist(),
                  pppm=PARAMS, pppm_tight=TIGHT, mesh_M=box.mesh.M, mesh_M_tight=tbox.mesh.M,
                  seeds=[[20260929, s] for s in range(args.ntraj)])
    say(f"N={N} L={box.L:.3f}: lambda_max(q0) median {np.median(lmax0):.3f} (range {min(lmax0):.3f}-{max(lmax0):.3f}); "
        f"dt = {dt:.5f}, T_end = {T_end:.3f}, {nsteps} steps")
    results = {}
    for s in (args.trajs if args.trajs else range(args.ntraj)):
        q0, p0, rng = initial_state(N, box.L, s)
        xi_fine = rng.normal(size=(4 * nsteps, N, 3))                   # finest (dt/4) stream
        R = {}
        t0 = time.perf_counter()
        R["dense_dt4"], built_h2, _ = run(box, q0, p0, xi_fine, dt / 4, 4 * nsteps, "dense")
        R["dense_dt2"], built_h1, _ = run(box, q0, p0, None, dt / 2, 2 * nsteps, "dense", coarse_stream=xi_fine)
        R["dense"], built_base, _ = run(box, q0, p0, None, dt, nsteps, "dense", coarse_stream=built_h1,
                                        snap_from=int(0.5 * nsteps))
        R["dense_2dt"], _, _ = run(box, q0, p0, None, 2 * dt, nsteps // 2, "dense", coarse_stream=built_base)
        R["tight"], _, _ = run(tbox, q0, p0, built_base, dt, nsteps, "tight")
        for m_ in ("lanczos_hi", "rank40", "rank48"):
            R[m_], _, _ = run(box, q0, p0, built_base, dt, nsteps, m_, snap_from=int(0.5 * nsteps) if m_ == "rank40" else None)
        results[s] = R
        save_small(params, {s: R}, OUT)
        say(f"  trajectory {s}: done in {time.perf_counter() - t0:.0f}s; final T dense {R['dense']['T'][-1].round(3)}, "
            f"rank40 {R['rank40']['T'][-1].round(3)}; max dt*lambda_max {np.nanmax(R['dense']['dt_lmax']):.2f}")
    return params, results


def save_small(params, results, out):
    out.mkdir(exist_ok=True)
    (out / "params_small.json").write_text(json.dumps(params, indent=1) + "\n")
    flat = {}
    for s, R in results.items():
        for m_, rec in R.items():
            for k, v in rec.items():
                flat[f"s{s}__{m_}__{k}"] = v
    for s in results:
        np.savez_compressed(out / f"raw_small_traj{s:02d}.npz", **{k: v for k, v in flat.items() if k.startswith(f"s{s}__")})


def control_run(args, say):
    """Negative control: correct damping, but independent per-particle Gaussian noise drawn from a RAW
    (unprojected) stream with the same mean scalar variance as f_dt(Gamma_h)^2 on Range(Pi)."""
    params = json.loads((OUT / "params_small.json").read_text())
    N, dt, nsteps = params["N"], params["dt"], params["nsteps"]
    box = Box(N)
    for s in range(args.n_indep):
        q0, p0, _ = initial_state(N, box.L, s)
        xi_raw = np.random.default_rng([20260929, s, 7]).normal(size=(nsteps, N, 3))
        rec, _, _ = run(box, q0, p0, xi_raw, dt, nsteps, "indep")
        np.savez_compressed(OUT / f"raw_control_traj{s:02d}.npz", **rec)
        say(f"control trajectory {s}: max momentum defect {rec['dP'].max():.2e}, final T {rec['T'][-1].round(3)}")


def large_run(args, say):
    """One production trajectory (run as an independent process so that trajectories can run in parallel)."""
    N, s, rank = args.N, args.seed, args.rank
    box = Box(N)
    q0, p0, rng = initial_state(N, box.L, 1000 + s)
    dt = args.dt
    nsteps = int(round(args.T_large / dt))
    xis = rng.normal(size=(nsteps, N, 3))
    method = {40: "rank40", 48: "rank48"}[rank]
    t0 = time.perf_counter()
    damp_check = []

    rec, _, (q, p) = run(box, q0, p0, xis, dt, nsteps, method)
    # damping-rank monitor on the final configuration: rank RD vs RD + 12
    op = box.gamma(q)
    pp = proj(p.ravel())
    a, _ = lanczos_apply(op, pp, RD, f_damp(dt))
    b, _ = lanczos_apply(op, pp, RD + 12, f_damp(dt))
    damp_check.append(float(np.linalg.norm(a - b) / np.linalg.norm(b)))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"raw_large_N{N}_seed{s}_rank{rank}.npz", damp_check=np.array(damp_check),
                        p_final=p.astype(np.float32), **rec)
    say(f"N={N} seed={s} rank={rank}: {nsteps} steps in {time.perf_counter() - t0:.0f}s; final T {rec['T'][-1].round(3)}; "
        f"max dP {rec['dP'].max():.1e}; dt*lambda_max range {np.nanmin(rec['dt_lmax'][1:]):.2f}-{np.nanmax(rec['dt_lmax']):.2f}; "
        f"damping check {damp_check[0]:.1e}")


def timing_run(args, say):
    rows = []
    for N in args.timing_N:
        box = Box(N)
        q0, p0, rng = initial_state(N, box.L, 5000)
        xis = rng.normal(size=(args.timing_steps, N, 3))
        rec, _, _ = run(box, q0, p0, xis, args.dt, args.timing_steps, "rank40")
        ts = rec["t_step"][1:]
        rows.append(dict(N=N, t_step_median=float(np.median(ts)), t_step_all=ts.tolist(),
                         n_actions=float(np.median(rec["n_actions"][1:]))))
        say(f"timing N={N}: {np.median(ts):.3f} s/step ({np.median(rec['n_actions'][1:]):.0f} Gamma_h actions/step)")
    OUT.mkdir(exist_ok=True)
    (OUT / "timing.json").write_text(json.dumps(rows, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["small", "large", "timing", "report", "control"], required=True)
    ap.add_argument("--N-small", type=int, default=64)
    ap.add_argument("--ntraj", type=int, default=16)
    ap.add_argument("--trajs", type=int, nargs="+", default=None, help="subset of trajectory indices (parallel runs)")
    ap.add_argument("--n-indep", type=int, default=4)
    ap.add_argument("--T-end", type=float, default=6.0)
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=40)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--T-large", type=float, default=3.0)
    ap.add_argument("--timing-N", type=int, nargs="+", default=[64, 512, 4000, 16000])
    ap.add_argument("--timing-steps", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tag = {"large": f"_N{args.N}_s{args.seed}_r{args.rank}",
           "small": ("_traj" + "-".join(map(str, args.trajs))) if args.trajs else ""}.get(args.stage, "")
    logf = OUT / f"log_{args.stage}{tag}.txt"
    lines = []

    def say(*a):
        msg = " ".join(str(t) for t in a)
        print(msg, flush=True)
        lines.append(msg)
        logf.write_text("\n".join(lines) + "\n")

    if args.dt is None and args.stage in ("large", "timing"):
        args.dt = json.loads((OUT / "params_small.json").read_text())["dt"]
    if args.stage == "small":
        small_study(args, say)
    elif args.stage == "large":
        large_run(args, say)
    elif args.stage == "control":
        control_run(args, say)
    elif args.stage == "timing":
        timing_run(args, say)
    else:
        report(say)



# ----------------------------------------------------------------------------
# Report stage: merge runs, paired error decomposition, Maxwell check, plots
# ----------------------------------------------------------------------------
OBS = ["Tx", "Ty", "Tz", "Tkin", "Ok"]


def obs_series(rec):
    T = rec["T"]
    return {"Tx": T[:, 0], "Ty": T[:, 1], "Tz": T[:, 2], "Tkin": rec["Tkin"], "Ok": rec["Ok"]}


def load_small():
    params = json.loads((OUT / "params_small.json").read_text())
    runs = {}
    for f in sorted(OUT.glob("raw_small_traj*.npz")):
        z = np.load(f)
        for key in z.files:
            s, m_, k = key.split("__")
            runs.setdefault(int(s[1:]), {}).setdefault(m_, {})[k] = z[key]
    return params, runs


def paired(runs, a, b, stride_a=1, stride_b=1, trajs=None):
    """Per-observable arrays (traj, time) of O_a - O_b on the common grid."""
    trajs = [s for s in sorted(runs) if a in runs[s] and b in runs[s]] if trajs is None else trajs
    out = {}
    for o in OBS:
        A = np.array([obs_series(runs[s][a])[o][::stride_a] for s in trajs])
        B = np.array([obs_series(runs[s][b])[o][::stride_b] for s in trajs])
        n = min(A.shape[1], B.shape[1])
        out[o] = A[:, :n] - B[:, :n]
    return out, len(trajs)


def summarize(D, ntraj):
    res = {}
    for o, d in D.items():
        m = d.mean(axis=0)
        se = d.std(axis=0, ddof=1) / np.sqrt(ntraj) if ntraj > 1 else np.zeros_like(m)
        res[o] = dict(max_abs_mean=float(np.abs(m).max()), rms_mean=float(np.sqrt((m ** 2).mean())),
                      max_abs_path=float(np.abs(d).max()), rms_path=float(np.sqrt((d ** 2).mean())),
                      max_se=float(se.max()))
    return res


def moments(x):
    x = np.asarray(x).ravel()
    m, v = x.mean(), x.var()
    return dict(n=int(x.size), mean=float(m), var=float(v), skew=float(((x - m) ** 3).mean() / v ** 1.5),
                exkurt=float(((x - m) ** 4).mean() / v ** 2 - 3), se_mean=float(np.sqrt(v / x.size)),
                se_var=float(v * np.sqrt(2 / x.size)))


def report(say):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from verify_ewald_longitudinal import set_style, INK2
    PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
    set_style()
    params, runs = load_small()
    for s in runs:
        runs[s].pop("indep", None)                      # superseded by the raw-stream control stage
    for f in sorted(OUT.glob("raw_control_traj*.npz")):
        runs[int(f.stem[-2:])]["indep"] = dict(np.load(f))
    trajs = sorted(runs)
    nt = len(trajs)
    dt = params["dt"]
    t = runs[trajs[0]]["dense"]["t"]
    S = {}

    # ---- momentum --------------------------------------------------------------------------
    mom = {m_: float(max(runs[s][m_]["dP"].max() for s in trajs if m_ in runs[s]))
           for m_ in ("dense", "dense_dt2", "dense_dt4", "dense_2dt", "tight", "lanczos_hi", "rank40", "rank48", "indep")}
    S["momentum_defect_max"] = mom

    # ---- relaxation and equilibrium --------------------------------------------------------------
    burn = t >= 3.0
    eq = {}
    for m_ in ("dense", "rank40", "rank48", "lanczos_hi"):
        per = np.array([[obs_series(runs[s][m_])[o][burn].mean() for o in ("Tx", "Ty", "Tz", "Tkin")] for s in trajs])
        mu, se = per.mean(0), per.std(0, ddof=1) / np.sqrt(nt)
        eq[m_] = {o: dict(mean=float(mu[k]), ci95=[float(mu[k] - 1.96 * se[k]), float(mu[k] + 1.96 * se[k])])
                  for k, o in enumerate(("Tx", "Ty", "Tz", "Tkin"))}
    S["late_time_temperatures_t>=3"] = eq
    Tinit = np.array([runs[s]["dense"]["T"][0] for s in trajs])
    S["initial_component_temperatures_mean"] = Tinit.mean(0).tolist()
    S["dt_lambda_max_range_dense"] = [float(min(np.nanmin(runs[s]["dense"]["dt_lmax"][1:]) for s in trajs)),
                                      float(max(np.nanmax(runs[s]["dense"]["dt_lmax"]) for s in trajs))]
    S["dt_lambda_max_range_rank40"] = [float(min(np.nanmin(runs[s]["rank40"]["dt_lmax"][1:]) for s in trajs)),
                                       float(max(np.nanmax(runs[s]["rank40"]["dt_lmax"]) for s in trajs))]
    S["lambda_min_positive_rank40"] = float(min(np.nanmin(runs[s]["rank40"]["lam_min"][1:]) for s in trajs))

    # ---- paired error decomposition ------------------------------------------------------------------
    comps = {
        "rank40 - dense": paired(runs, "rank40", "dense"),
        "rank48 - dense": paired(runs, "rank48", "dense"),
        "lanczos_hi - dense": paired(runs, "lanczos_hi", "dense"),
        "rank40 - lanczos_hi": paired(runs, "rank40", "lanczos_hi"),
        "rank48 - rank40": paired(runs, "rank48", "rank40"),
        "tight - dense (spatial)": paired(runs, "tight", "dense"),
        "dense(dt) - dense(dt/2)": paired(runs, "dense", "dense_dt2", 1, 2),
        "dense(dt/2) - dense(dt/4)": paired(runs, "dense_dt2", "dense_dt4", 1, 2),
        "dense(2dt) - dense(dt)": paired(runs, "dense_2dt", "dense", 1, 2),
    }
    S["paired"] = {k: summarize(*v) for k, v in comps.items()}
    samp = {o: float(max(np.array([obs_series(runs[s]["dense"])[o] for s in trajs]).std(0, ddof=1) / np.sqrt(nt)))
            for o in OBS}
    S["sampling_SE_max_over_t"] = samp

    # ---- Maxwell --------------------------------------------------------------------------------------
    N = params["N"]
    mx = {}
    for m_ in ("dense", "rank40"):
        snaps = np.concatenate([runs[s][m_]["snaps"] for s in trajs if "snaps" in runs[s][m_]])
        x = snaps / np.sqrt(MASS / BETA * (N - 1) / N)
        mx[m_] = dict(all=moments(x), **{c: moments(x[..., k]) for k, c in enumerate("xyz")},
                      n_snapshots=int(snaps.shape[0]))
    S["maxwell_conditional"] = mx

    # ---- large N ----------------------------------------------------------------------------------------
    large = {}
    for f in sorted(OUT.glob("raw_large_N*_seed*_rank*.npz")):
        nm = f.stem.split("_")
        N_, s_, r_ = int(nm[2][1:]), int(nm[3][4:]), int(nm[4][4:])
        large[(N_, s_, r_)] = dict(np.load(f))
    L_ = {}
    for (N_, s_, r_), rec in large.items():
        L_.setdefault(str(N_), {})[f"seed{s_}_rank{r_}"] = dict(
            dP_max=float(rec["dP"].max()), T_final=rec["T"][-1].tolist(), T_init=rec["T"][0].tolist(),
            T_late_mean=rec["T"][rec["t"] >= 2.0].mean(0).tolist(),
            dt_lmax_range=[float(np.nanmin(rec["dt_lmax"][1:])), float(np.nanmax(rec["dt_lmax"]))],
            damping_check=rec["damp_check"].tolist(), steps=int(len(rec["t"]) - 1),
            t_step_median=float(np.nanmedian(rec["t_step"][1:])))
    for N_ in {k[0] for k in large}:
        if (N_, 0, 40) in large and (N_, 0, 48) in large:
            a, b = large[(N_, 0, 48)], large[(N_, 0, 40)]
            n = min(len(a["t"]), len(b["t"]))
            L_[str(N_)]["rank48_minus_rank40_seed0"] = {
                o: float(np.abs(obs_series(a)[o][:n] - obs_series(b)[o][:n]).max()) for o in OBS}
    S["large_N"] = L_
    timing = json.loads((OUT / "timing.json").read_text()) if (OUT / "timing.json").exists() else []
    S["timing"] = timing
    if len(timing) >= 2:
        Ns = np.array([r["N"] for r in timing], float)
        ts = np.array([r["t_step_median"] for r in timing])
        big = Ns >= 512
        S["timing_exponent_N>=512"] = float(np.polyfit(np.log(Ns[big]), np.log(ts[big]), 1)[0]) if big.sum() >= 2 else None
        S["T_over_NlogN"] = {str(int(n)): float(x / (n * np.log(n))) for n, x in zip(Ns, ts)}
    (OUT / "summary.json").write_text(json.dumps(S, indent=1) + "\n")

    # ---- plots ----------------------------------------------------------------------------------------------
    def band(ax, m_, o, col, lab, ls="-"):
        A = np.array([obs_series(runs[s][m_])[o] for s in trajs])
        mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(nt)
        tt = runs[trajs[0]][m_]["t"]
        ax.plot(tt, mu, ls, color=col, label=lab)
        if ls == "-":
            ax.fill_between(tt, mu - se, mu + se, color=col, alpha=0.15, lw=0)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, o in enumerate(("Tx", "Ty", "Tz")):
        band(axs[0], "dense", o, PAL[k], f"{o} dense (±1 SE, {nt} traj)")
        band(axs[0], "rank40", o, PAL[k], f"{o} rank 40", ls="--")
    axs[0].axhline(1.0, color=INK2, lw=0.8)
    axs[0].set(xlabel="t", ylabel="component temperature", title=f"1a. N = {N}: T_x, T_y, T_z")
    axs[0].legend(fontsize=7)
    for (N_, s_, r_), rec in sorted(large.items()):
        if r_ != 40:
            continue
        for k in range(3):
            axs[1].plot(rec["t"], rec["T"][:, k], color=PAL[k], lw=0.8 if N_ == 4000 else 1.4,
                        ls="-" if N_ == 16000 else ":", label=f"N={N_}, T_{'xyz'[k]}" if s_ == 0 else None)
    axs[1].axhline(1.0, color=INK2, lw=0.8)
    axs[1].set(xlabel="t", ylabel="component temperature", title="1b. Large N, rank 40 (dotted 4000, solid 16000)")
    axs[1].legend(fontsize=7)
    fig.tight_layout(), fig.savefig(OUT / "plot1_component_temperatures.png"), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, m_ in enumerate(("dense", "rank40", "rank48", "lanczos_hi", "tight")):
        band(ax, m_, "Tkin", PAL[k], m_, ls="-" if m_ == "dense" else "--")
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set(xlabel="t", ylabel="T_kin", title=f"2. Kinetic temperature (N = {N}, mean over {nt} trajectories)")
    ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(OUT / "plot2_Tkin.png"), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, m_ in enumerate(("dense", "rank40", "rank48", "lanczos_hi", "indep")):
        A = np.array([runs[s][m_]["dP"] for s in trajs if m_ in runs[s]])
        ax.semilogy(runs[trajs[0]][m_]["t"], np.maximum(A.max(0), 1e-18), color=PAL[k],
                    label=m_ + (" (negative control)" if m_ == "indep" else ""))
    for (N_, s_, r_), rec in sorted(large.items()):
        ax.semilogy(rec["t"], np.maximum(rec["dP"], 1e-18), ":", color=PAL[5] if N_ == 4000 else PAL[6], lw=0.8,
                    label=f"N={N_} rank {r_}" if s_ == 0 else None)
    ax.set(xlabel="t", ylabel="||P(t) - P(0)|| / max(1, ||P(0)||)", title="3. Total-momentum defect (max over trajectories)")
    ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(OUT / "plot3_momentum.png"), plt.close(fig)

    for fig_i, key, fname in ((4, "rank40 - dense", "plot4_rank40_error.png"), (5, "rank48 - dense", "plot5_rank48_error.png")):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
        D, _ = comps[key]
        Dt, _ = comps["dense(dt) - dense(dt/2)"]
        for ax, o in zip(axs, ("Tx", "Ok")):
            A = np.array([obs_series(runs[s]["dense"])[o] for s in trajs])
            tt = t[:D[o].shape[1]]
            ax.semilogy(tt, np.maximum(np.abs(D[o].mean(0)), 1e-18), color=PAL[0], label=f"|mean paired {key}|")
            ax.semilogy(tt, np.maximum(np.sqrt((D[o] ** 2).mean(0)), 1e-18), ":", color=PAL[0], label="pathwise RMS")
            ax.semilogy(t[:Dt[o].shape[1]], np.maximum(np.abs(Dt[o].mean(0)), 1e-18), color=PAL[1],
                        label="|mean paired dense(dt) - dense(dt/2)| (temporal)")
            ax.semilogy(t, A.std(0, ddof=1) / np.sqrt(nt), color=INK2, lw=0.8, label="sampling SE of the mean")
            ax.set(xlabel="t", ylabel=f"difference in {o}", title=f"{fig_i}. {key}: {o}")
            ax.legend(fontsize=7)
        fig.tight_layout(), fig.savefig(OUT / fname), plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, o in zip(axs, ("Tx", "Ok")):
        for k, key in enumerate(("dense(2dt) - dense(dt)", "dense(dt) - dense(dt/2)", "dense(dt/2) - dense(dt/4)")):
            D, _ = comps[key]
            step_ = 2 * dt if key.startswith("dense(2dt)") else dt
            tt = np.arange(D[o].shape[1]) * step_
            ax.semilogy(tt, np.maximum(np.abs(D[o].mean(0)), 1e-18), color=PAL[k], label=f"|mean| {key}")
            ax.semilogy(tt, np.sqrt((D[o] ** 2).mean(0)), ":", color=PAL[k], lw=1)
        D, _ = comps["rank40 - dense"]
        ax.semilogy(t[:D[o].shape[1]], np.maximum(np.sqrt((D[o] ** 2).mean(0)), 1e-18), color=PAL[4], lw=1.4,
                    label="rank40 - dense (pathwise RMS)")
        ax.set(xlabel="t", ylabel=f"paired difference in {o} (dotted: pathwise RMS)", title=f"6. Temporal refinement: {o}")
        ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(OUT / "plot6_temporal_refinement.png"), plt.close(fig)

    from scipy import stats as sst
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, m_ in enumerate(("dense", "rank40")):
        snaps = np.concatenate([runs[s][m_]["snaps"] for s in trajs if "snaps" in runs[s][m_]])
        x = np.sort((snaps / np.sqrt(MASS / BETA * (N - 1) / N)).ravel())
        qn = sst.norm.ppf((np.arange(1, x.size + 1) - 0.5) / x.size)
        axs[0].plot(qn[::7], x[::7], ".", ms=2, color=PAL[k], label=m_)
        axs[1].hist(x, bins=80, density=True, histtype="step", color=PAL[k], label=m_)
    xx = np.linspace(-4.5, 4.5, 300)
    axs[0].plot(xx, xx, color=INK2, lw=0.8, label="y = x")
    axs[1].plot(xx, sst.norm.pdf(xx), color=INK2, lw=0.8, label="standard normal")
    axs[0].set(xlabel="normal quantile", ylabel="sample quantile", title="7a. QQ plot, Pi p / sqrt(m/beta (N-1)/N), t >= T/2")
    axs[1].set(xlabel="normalised momentum component", ylabel="density", title="7b. Conditional Maxwell histogram")
    axs[0].legend(fontsize=8), axs[1].legend(fontsize=8)
    fig.tight_layout(), fig.savefig(OUT / "plot7_maxwell.png"), plt.close(fig)

    if timing:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        Ns = np.array([r["N"] for r in timing], float)
        ts = np.array([r["t_step_median"] for r in timing])
        ax.loglog(Ns, ts, "o-", color=PAL[0], label="measured wall time per step (rank 40 + damping rank 12)")
        ax.loglog(Ns, ts[-1] * Ns * np.log(Ns) / (Ns[-1] * np.log(Ns[-1])), color=INK2, lw=0.8, label="~ N log N")
        ax.set(xlabel="N", ylabel="s / step", title="8. Wall time per step vs N (single process)")
        ax.legend(fontsize=8)
        fig.tight_layout(), fig.savefig(OUT / "plot8_time_per_step.png"), plt.close(fig)
    say(json.dumps({k: S[k] for k in ("momentum_defect_max", "sampling_SE_max_over_t")}, indent=1))
    for k, v in S["paired"].items():
        say(f"{k:28s} " + "  ".join(f"{o}: mean {v[o]['max_abs_mean']:.1e} path {v[o]['rms_path']:.1e}" for o in OBS))
    return S


if __name__ == "__main__":
    main()
