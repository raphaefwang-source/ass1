#!/usr/bin/env python3
"""
Fixed-time paired validation of the fast operator (PPPM + Lanczos 40/16) against the dense full-periodic reference,
with a dense time-step refinement as the yardstick for "small". Toy double well, law A, N = 64 (same BAOAB, pair
potential and friction kernel as toy_dynamics.py).

Design
    initial states   the five v2 final states (seeds 101-105) used by production_t60
    noise paths      16 independent paths per initial state; path (s, j) draws its finest-level Gaussians from
                     SeedSequence([20261008, s, j])
    levels (dense)   dt/4 = 0.00125 (finest), dt/2, dt = 0.005 (production), 2 dt
    fast             PPPM + Lanczos 40/16 at dt, on exactly the same dt-level noise as dense at dt
    observables      at t = 0.5, 1, 2 (integer steps of every level): K/N = sum|p|^2/(2 m N), U/N,
                     M4 = mean_i |v_i|^4, MSD = mean_i |q_i(t) - q_i(0)|^2 (unwrapped positions)
Brownian coupling across step sizes (nested streams, as in test_true_dynamics): a level with step h consumes, per
step, the two Gaussians xi_1, xi_2 of the next finer level (step h/2) through
    xi_h = S_h^{-1} (R_{h/2} S_{h/2} xi_1 + S_{h/2} xi_2),   R_h = exp(-h Gamma/m), S_h = f_h(Gamma),
with Gamma = Gamma(q_half) of the coarse run itself. For a frozen Gamma this is exactly N(0, Pi) and reproduces the
fine noise integrated over the coarse step; while Gamma(q) moves it is an approximate pathwise coupling and exact in
distribution. The constructed xi_h is stored and passed on to the next coarser level. Sharing a random seed alone
would NOT couple the Brownian paths across step sizes.

Stages
    --stage run [--states 101 ...] [--paths 16] [--workers 4]   per-path results -> toy_fixed_time_results/paths/
    --stage analyze                                             paths.csv, per-state and across-state tables, figure
    --stage selftest                                            O-step identity, nested-noise covariance, frozen-Gamma
                                                                two-fine-steps == one-coarse-step check
Layering follows the error chain of the fast-particle Landau preprint (Zhao, Dou, Lei): operator accuracy is tested
on static configurations elsewhere (radial_kernel_static_results); this script tests fixed-time statistics, with the
production chain's paired difference judged against the dense temporal bias on the same Brownian path indices.
Statistics: per initial state over its noise paths (paths are independent given the state), and across the five
initial states using the per-state means as the units (df = 4). Noise replicates are never counted as independent
initial states. No convergence order is assumed or claimed.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import prl_toy_models as v2  # noqa: E402
import toy_dynamics as td  # noqa: E402
from test_lanczos_fdt import DenseRef, proj  # noqa: E402

OUT = HERE / "toy_fixed_time_results"
PATHS = OUT / "paths"
CKPT = HERE / "toy_models" / "v2_results" / "restart_checkpoints" / "double_well_burn140" / "lattice" / "double_well_A"
POT, LAW = "double_well", "A"
TOY = dict(gamma=0.5, kappa=0.7, r_ref=1.3, kT=0.7, mass=1.0, L=5.5)
DT = 0.005
LEVELS = (0.01, 0.005, 0.0025, 0.00125)                  # 2dt, dt, dt/2, dt/4 (finest)
TIMES = (0.5, 1.0, 2.0)
T_END = max(TIMES)
STATES = (101, 102, 103, 104, 105)
ENTROPY = 20261008
OBS = ("KE_per_particle", "U_per_particle", "M4_velocity", "MSD")


def load_state(seed):
    with np.load(CKPT / f"seed_{seed}" / "restart.npz") as ck:
        return ck["Q"].copy(), ck["p"].copy()


class DenseO:
    """Dense full-periodic O-step (same matrices as toy_dynamics method 'reference'), optional nested noise."""

    def __init__(self, dt):
        self.dt, self.m, self.kT = dt, TOY["mass"], TOY["kT"]
        self.lat = v2.LatticeFriction(TOY["L"], kernel=LAW, gamma=TOY["gamma"], kappa=TOY["kappa"], r_ref=TOY["r_ref"])

    def __call__(self, q_half, p, xi=None, fine_pair=None):
        N = len(p)
        ref = DenseRef(self.lat.matrix(q_half % TOY["L"]), N)
        lam, U = ref.lam, ref.U
        if lam[0] <= 0:
            raise ValueError(f"non-positive eigenvalue on Range(Pi): {lam[0]:.3e}")
        fn = lambda l, h: np.sqrt(self.kT * self.m * -np.expm1(-2 * h * l / self.m))
        fs = fn(lam, self.dt)
        if fine_pair is not None:                       # build this level's noise from the finer level's pair
            h2 = 0.5 * self.dt
            x1, x2 = U.T @ proj(fine_pair[0].ravel()), U.T @ proj(fine_pair[1].ravel())
            c = (np.exp(-h2 * lam / self.m) * fn(lam, h2) * x1 + fn(lam, h2) * x2) / fs
            xi = (U @ c).reshape(N, 3)
        z = U.T @ proj(xi.ravel())
        pm = p.mean(axis=0)
        damp = U @ (np.exp(-self.dt * lam / self.m) * (U.T @ proj(p.ravel())))
        noise = U @ (fs * z)
        return pm[None, :] + (proj(damp) + proj(noise)).reshape(N, 3), xi


class FastO:
    def __init__(self, dt):
        self.th = td.Thermostat("pppm_lanczos", TOY["L"], LAW, TOY["gamma"], TOY["kappa"], TOY["r_ref"], dt,
                                TOY["kT"], TOY["mass"])

    def __call__(self, q_half, p, xi=None, fine_pair=None):
        assert fine_pair is None
        return self.th.o_step(q_half, p, xi)[0], xi


def observe(q, p, q0, U):
    m, N = TOY["mass"], len(p)
    v = p / m
    v2_ = np.einsum("nc,nc->n", v, v)
    dq = q - q0
    return dict(KE_per_particle=float(0.5 * m * v2_.sum() / N), U_per_particle=float(U / N),
                M4_velocity=float(np.mean(v2_ ** 2)), MSD=float(np.einsum("nc,nc->", dq, dq) / N))


def integrate(ostep, dt, q0, p0, noises=None, fine=None):
    """BAOAB from (q0, p0) to T_END. Either `noises` (n_steps, N, 3) or `fine` (2 n_steps, N, 3) for nested build.

    Returns observables at TIMES, the states there, and the noise actually consumed (for the next coarser level).
    """
    m, L = TOY["mass"], TOY["L"]
    n_steps = int(round(T_END / dt))
    rec_steps = {int(round(t / dt)): t for t in TIMES}
    q, p = q0.astype(float).copy(), p0.astype(float).copy()
    force, energy, rmin = v2.conservative_force(q, L, POT)
    used = np.empty((n_steps,) + p.shape)
    out, states = {}, {}
    for n in range(n_steps):
        p = p + 0.5 * dt * force
        q = q + 0.5 * dt / m * p
        if fine is not None:
            p, xi = ostep(q, p, fine_pair=(fine[2 * n], fine[2 * n + 1]))
        else:
            p, xi = ostep(q, p, xi=noises[n])
        used[n] = xi
        q = q + 0.5 * dt / m * p
        force, energy, rmin = v2.conservative_force(q, L, POT)
        p = p + 0.5 * dt * force
        if not np.all(np.isfinite(p)) or rmin < 0.45:
            raise RuntimeError("unstable close encounter")
        if n + 1 in rec_steps:
            t = rec_steps[n + 1]
            out[t] = observe(q, p, q0, energy)
            states[t] = (q.copy(), p.copy())
    return out, states, used


def run_path(job):
    state, path = job
    f = PATHS / f"state{state}_path{path:02d}.npz"
    if f.exists():
        return str(f), "exists"
    q0, p0 = load_state(state)
    rng = np.random.default_rng(np.random.SeedSequence([ENTROPY, state, path]))
    finest = LEVELS[-1]
    noise = rng.standard_normal((int(round(T_END / finest)),) + p0.shape)
    res, st, timing = {}, {}, {}
    t0 = time.perf_counter()
    res[("dense", finest)], st[("dense", finest)], level_noise = integrate(DenseO(finest), finest, q0, p0, noises=noise)
    timing[("dense", finest)] = time.perf_counter() - t0
    base_noise = None
    for dt in sorted(LEVELS[:-1]):                         # dt/2, dt, 2dt: each built from the next finer level
        t0 = time.perf_counter()
        res[("dense", dt)], st[("dense", dt)], level_noise = integrate(DenseO(dt), dt, q0, p0, fine=level_noise)
        timing[("dense", dt)] = time.perf_counter() - t0
        if abs(dt - DT) < 1e-12:
            base_noise = level_noise.copy()
    t0 = time.perf_counter()
    res[("fast", DT)], st[("fast", DT)], _ = integrate(FastO(DT), DT, q0, p0, noises=base_noise)
    timing[("fast", DT)] = time.perf_counter() - t0
    rows = [dict(state=state, path=path, method=m, dt=dt, t=t, **vals)
            for (m, dt), r in res.items() for t, vals in r.items()]
    np.savez_compressed(f, rows=np.array(json.dumps(rows)), timing=np.array(json.dumps({f"{m}@{dt}": v for (m, dt), v in timing.items()})),
                        seed=np.array(json.dumps(dict(entropy=[ENTROPY, state, path], generator="numpy default_rng(SeedSequence)",
                                                      finest_dt=finest, draws=list(noise.shape)))),
                        **{f"{m}_{dt}_t{t}_{k}": st[(m, dt)][t][i] for (m, dt) in st for t in TIMES for i, k in enumerate(("q", "p"))})
    return str(f), f"{sum(timing.values()):.0f}s"


def stage_selftest(args):
    q0, p0 = load_state(STATES[0])
    rng = np.random.default_rng(1)
    q_half = q0 + 0.5 * DT * p0
    xi, xi1, xi2 = (rng.standard_normal(p0.shape) for _ in range(3))
    dense = DenseO(DT)
    ref = td.Thermostat("reference", TOY["L"], LAW, TOY["gamma"], TOY["kappa"], TOY["r_ref"], DT, TOY["kT"], TOY["mass"])
    a, _ = dense(q_half, p0, xi=xi)
    b = ref.o_step(q_half, p0, xi)[0]
    out = dict(dense_O_vs_toy_dynamics_reference=float(np.abs(a - b).max()))
    # nested construction at a fixed configuration: coarse noise = A xi1 + B xi2 with A^2 + B^2 = 1 on Range(Pi)
    G = DenseRef(dense.lat.matrix(q_half % TOY["L"]), len(p0))
    lam = G.lam
    h = DT / 2
    fn = lambda l, hh: np.sqrt(TOY["kT"] * TOY["mass"] * -np.expm1(-2 * hh * l / TOY["mass"]))
    A = np.exp(-h * lam / TOY["mass"]) * fn(lam, h) / fn(lam, DT)
    B = fn(lam, h) / fn(lam, DT)
    out["nested_noise_A2_plus_B2_minus_1"] = float(np.abs(A ** 2 + B ** 2 - 1).max())
    # frozen Gamma: one coarse O-step with the constructed noise == two fine O-steps with xi1, xi2
    U = G.U
    R = lambda hh: U @ np.diag(np.exp(-hh * lam / TOY["mass"])) @ U.T
    S = lambda hh: U @ np.diag(fn(lam, hh)) @ U.T
    pp, z1, z2 = proj(p0.ravel()), proj(xi1.ravel()), proj(xi2.ravel())
    two_fine = R(h) @ (R(h) @ pp + S(h) @ z1) + S(h) @ z2
    _, xi_c = dense(q_half, p0, fine_pair=(xi1, xi2))
    one_coarse = R(DT) @ pp + S(DT) @ proj(xi_c.ravel())
    out["frozen_gamma_two_fine_vs_one_coarse"] = float(np.abs(two_fine - one_coarse).max())
    (OUT / "selftest.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def stage_run(args):
    PATHS.mkdir(parents=True, exist_ok=True)
    jobs = [(s, j) for s in args.states for j in range(args.paths)]
    meta = dict(command=" ".join(["python3", Path(__file__).name] + sys.argv[1:]), toy=TOY, potential=POT, law=LAW,
                levels=LEVELS, base_dt=DT, times=TIMES, states=list(args.states), paths=args.paths, entropy=ENTROPY,
                fast=dict(method="pppm_lanczos", pppm=td.PPPM_PRODUCTION, rank_noise=td.RANK_NOISE, rank_damp=td.RANK_DAMP),
                initial_states=str(CKPT.relative_to(HERE)))
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f, status in pool.map(run_path, jobs):
            print(f"done {f} ({status})", flush=True)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
COMPARISONS = {                                            # name: (a, b) -> a - b, per path
    "fast_minus_dense@dt": (("fast", 0.005), ("dense", 0.005)),
    "dense@2dt_minus_dense@dt/4": (("dense", 0.01), ("dense", 0.00125)),
    "dense@dt_minus_dense@dt/4": (("dense", 0.005), ("dense", 0.00125)),
    "dense@dt/2_minus_dense@dt/4": (("dense", 0.0025), ("dense", 0.00125)),
    "dense@2dt_minus_dense@dt": (("dense", 0.01), ("dense", 0.005)),
    "dense@dt_minus_dense@dt/2": (("dense", 0.005), ("dense", 0.0025)),
}


def ci(x, level=0.95):
    x = np.asarray(x, float)
    n = len(x)
    mean, sd = x.mean(), x.std(ddof=1)
    sem = sd / np.sqrt(n)
    tc = stats.t.ppf(0.5 + level / 2, n - 1)
    return n, mean, sd, sem, tc, mean - tc * sem, mean + tc * sem


def stage_analyze(args):
    files = sorted(PATHS.glob("state*_path*.npz"))
    rows, timings = [], []
    for f in files:
        with np.load(f) as z:
            rows += json.loads(str(z["rows"]))
            timings.append(json.loads(str(z["timing"])))
    if not rows:
        raise SystemExit("no per-path results")
    keys = ["state", "path", "method", "dt", "t"] + list(OBS)
    with open(OUT / "paths.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["state"], r["path"], r["method"], r["dt"], r["t"])):
            w.writerow({k: (f"{r[k]:.12g}" if isinstance(r[k], float) else r[k]) for k in keys})
    val = {(r["state"], r["path"], r["method"], r["dt"], r["t"]): r for r in rows}
    states = sorted({r["state"] for r in rows})
    paths_of = {s: sorted({r["path"] for r in rows if r["state"] == s}) for s in states}
    if min(len(v) for v in paths_of.values()) < 2:
        raise SystemExit("need at least two noise paths per initial state for a per-state interval")
    per_state, overall = [], []
    for t in TIMES:
        for ob in OBS:
            for name, (a, b) in COMPARISONS.items():
                state_means = []
                for s in states:
                    P = paths_of[s]
                    d = [val[(s, j) + a + (t,)][ob] - val[(s, j) + b + (t,)][ob] for j in P]
                    base = [val[(s, j, "dense", DT, t)][ob] for j in P]
                    n, mean, sd, sem, tc, lo, hi = ci(d)
                    ref_mean, ref_sd = float(np.mean(base)), float(np.std(base, ddof=1))
                    per_state.append(dict(t=t, observable=ob, comparison=name, state=s, n_paths=n, mean_diff=mean,
                                          sd_paths=sd, sem=sem, t_crit_95=tc, ci95_low=lo, ci95_high=hi,
                                          ci_excludes_zero=bool(lo > 0 or hi < 0),
                                          rms_pathwise=float(np.sqrt(np.mean(np.square(d)))),
                                          dense_dt_mean=ref_mean, dense_dt_sd_over_paths=ref_sd,
                                          rel_to_dense_mean=mean / abs(ref_mean),
                                          in_units_of_path_sd=mean / ref_sd if ref_sd > 0 else np.nan))
                    state_means.append(mean)
                if len(states) >= 2:
                    n, mean, sd, sem, tc, lo, hi = ci(state_means)
                    base_all = np.mean([val[(s, j, "dense", DT, t)][ob] for s in states for j in paths_of[s]])
                    overall.append(dict(t=t, observable=ob, comparison=name, n_states=n,
                                        n_paths_per_state=",".join(str(len(paths_of[s])) for s in states),
                                        mean_of_state_means=mean, sd_between_states=sd, sem_states=sem, t_crit_95=tc,
                                        ci95_low=lo, ci95_high=hi, ci_excludes_zero=bool(lo > 0 or hi < 0),
                                        dense_dt_mean=float(base_all), rel_to_dense_mean=mean / abs(base_all),
                                        **{f"state{s}_mean": m for s, m in zip(states, state_means)}))
    write(OUT / "fixed_time_per_state.csv", per_state)
    if len(states) < 2:
        print("only one initial state: per-state table written; across-state summary needs >= 2 states")
        return
    write(OUT / "fixed_time_across_states.csv", overall)
    verdict = []
    for t in TIMES:
        for ob in OBS:
            fd = next(r for r in overall if r["t"] == t and r["observable"] == ob and r["comparison"] == "fast_minus_dense@dt")
            te = next(r for r in overall if r["t"] == t and r["observable"] == ob and r["comparison"] == "dense@dt_minus_dense@dt/4")
            fd_s = [r for r in per_state if r["t"] == t and r["observable"] == ob and r["comparison"] == "fast_minus_dense@dt"]
            te_s = [r for r in per_state if r["t"] == t and r["observable"] == ob and r["comparison"] == "dense@dt_minus_dense@dt/4"]
            resolved = te["ci_excludes_zero"]
            fd_bound = max(abs(fd["ci95_low"]), abs(fd["ci95_high"]))
            verdict.append(dict(t=t, observable=ob,
                                fast_dense_mean=fd["mean_of_state_means"], fast_dense_ci_low=fd["ci95_low"],
                                fast_dense_ci_high=fd["ci95_high"], fast_dense_abs_ci_bound=fd_bound,
                                time_error_dt_mean=te["mean_of_state_means"], time_error_dt_ci_low=te["ci95_low"],
                                time_error_dt_ci_high=te["ci95_high"], time_error_resolved=resolved,
                                fast_dense_bound_below_abs_time_error=(bool(fd_bound < abs(te["mean_of_state_means"])) if resolved else "time error not resolved"),
                                rms_pathwise_fast_dense_max_state=max(r["rms_pathwise"] for r in fd_s),
                                rms_pathwise_time_error_dt_min_state=min(r["rms_pathwise"] for r in te_s)))
    write(OUT / "fixed_time_verdict.csv", verdict)
    tim = {}
    for d in timings:
        for k, v in d.items():
            tim.setdefault(k, []).append(v)
    (OUT / "timing.json").write_text(json.dumps({k: dict(mean_s=float(np.mean(v)), n=len(v)) for k, v in tim.items()}, indent=1) + "\n")
    figure(overall, per_state)
    print(f"{len(files)} paths, states {states}, paths per state {[len(paths_of[s]) for s in states]}")
    for v in verdict:
        print(f"t={v['t']:<4} {v['observable']:<16} fast-dense {v['fast_dense_mean']:+.2e} [{v['fast_dense_ci_low']:+.2e},{v['fast_dense_ci_high']:+.2e}]"
              f"  dense dt - dt/4 {v['time_error_dt_mean']:+.2e} [{v['time_error_dt_ci_low']:+.2e},{v['time_error_dt_ci_high']:+.2e}]"
              f"  resolved={v['time_error_resolved']}  rms(fd)max={v['rms_pathwise_fast_dense_max_state']:.1e} rms(te)min={v['rms_pathwise_time_error_dt_min_state']:.1e}")


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


def figure(overall, per_state=None):
    """Log-scale magnitudes, no lines across magnitudes.
    Top: 95% bound on |mean difference| over the 5 initial states; filled = CI excludes 0 (marker at |mean|, bar to
    the CI end farthest from 0), open = CI includes 0 (marker at the bound, i.e. an upper bound only).
    Bottom: pathwise RMS of the per-path difference (median over initial states, bar = min..max over states)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"fast_minus_dense@dt": ("#dc7627", "o", "fast − dense, both at dt"),
              "dense@2dt_minus_dense@dt/4": ("#94a3b8", "v", "dense 2dt − dense dt/4"),
              "dense@dt_minus_dense@dt/4": ("#2563eb", "s", "dense dt − dense dt/4"),
              "dense@dt/2_minus_dense@dt/4": ("#0f766e", "^", "dense dt/2 − dense dt/4")}
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), layout="constrained", sharex=True)
    for col, ob in enumerate(OBS):
        for k, (name, (c, mk, lab)) in enumerate(styles.items()):
            x0 = np.arange(len(TIMES)) + 0.15 * (k - 1.5)
            for xi_, t in zip(x0, TIMES):
                r = next(r for r in overall if r["observable"] == ob and r["comparison"] == name and r["t"] == t)
                lo, hi, m = r["ci95_low"], r["ci95_high"], r["mean_of_state_means"]
                bound = max(abs(lo), abs(hi))
                if r["ci_excludes_zero"]:
                    near = min(abs(lo), abs(hi))
                    axes[0, col].errorbar([xi_], [abs(m)], yerr=[[abs(m) - near], [bound - abs(m)]], color=c, marker=mk,
                                          ms=7, capsize=3, lw=1)
                else:
                    axes[0, col].plot([xi_], [bound], marker=mk, mfc="none", mec=c, ms=7, ls="none")
                if per_state is not None:
                    rms = [q["rms_pathwise"] for q in per_state if q["observable"] == ob and q["comparison"] == name and q["t"] == t]
                    med = float(np.median(rms))
                    axes[1, col].errorbar([xi_], [med], yerr=[[med - min(rms)], [max(rms) - med]], color=c, marker=mk,
                                          ms=6, capsize=3, lw=1, ls="none")
            axes[0, col].plot([], [], color=c, marker=mk, ls="none", label=lab)
        axes[0, col].set(yscale="log", title=ob.replace("_", " "), ylabel="|mean difference|: 95% bound over 5 states")
        axes[1, col].set(yscale="log", ylabel="pathwise RMS of the difference", xlabel="time t",
                         xticks=range(len(TIMES)), xticklabels=[str(t) for t in TIMES])
        axes[0, col].legend(fontsize=7, title="filled: CI excludes 0\nopen: upper bound only", title_fontsize=7)
    fig.suptitle("double_well_A, N = 64, 5 initial states x 16 coupled noise paths: fast vs dense at the production step dt, "
                 "and dense time-step differences (nested coupled noise). No convergence order assumed.", fontsize=10)
    fig.savefig(OUT / "fixed_time_differences.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["run", "analyze", "selftest"], required=True)
    ap.add_argument("--states", type=int, nargs="+", default=list(STATES))
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"run": stage_run, "analyze": stage_analyze, "selftest": stage_selftest}[args.stage](args)


if __name__ == "__main__":
    main()
