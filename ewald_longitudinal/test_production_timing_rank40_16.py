#!/usr/bin/env python3
"""
Production timing of the PPPM + direct-start Lanczos finite-time FDT thermostat step with the final ranks:
noise rank 40, damping rank 16 (the earlier timing in dynamics_results/timing.json used damping rank 12).

One real A(dt/2) O(dt) A(dt/2) ideal-gas step, exactly as test_true_dynamics.step:
    q_half = q + dt/(2m) p (mod L);  Gamma_h = Gamma_h(q_half)
    p_new  = p_mean + Pi exp(-dt Gamma_h/m) Pi p + Pi f_dt(Gamma_h) Pi xi
    q_new  = q_half + dt/(2m) p_new (mod L)
Model, density, kernel, PPPM parameters, neighbour logic, dt, beta, m and the Lanczos routine are imported
unchanged from test_true_dynamics / test_lanczos_fdt. Ritz values are checked, never clipped.

Run single-threaded:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 test_production_timing_rank40_16.py
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_lanczos_fdt import lanczos, tridiag, proj  # noqa: E402
from test_true_dynamics import Box, NegativeRitz, f_noise, f_damp, initial_state, MASS, OUT  # noqa: E402

RANK_NOISE, RANK_DAMP = 40, 16
OLD = {512: 0.790305835500476, 4000: 7.438761513996724, 16000: 40.37323408099837}   # timing.json, 40 + 12
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def timed_apply(op, v, rank, fun):
    """Direct-start Lanczos f(Gamma_h) v with a time breakdown. v is in Range(Pi)."""
    a0, n0 = op.t_actions, op.n_actions
    t0 = time.perf_counter()
    Lz = lanczos(op, v, rank)
    t_lz = time.perf_counter() - t0
    t1 = time.perf_counter()
    s = Lz["s"]
    th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], s))
    if th[0] <= 0:
        raise NegativeRitz(f"Ritz value {th[0]:.3e}")
    c = Lz["nz"] * (U @ (fun(th) * U[0]))
    t_small = time.perf_counter() - t1
    t2 = time.perf_counter()
    raw = Lz["Q"][:s].T @ c
    out = proj(raw)
    t_comb = time.perf_counter() - t2
    t_act = op.t_actions - a0
    nrm = np.linalg.norm(raw)
    return out, dict(actions=op.n_actions - n0, rank=s, t_action=t_act, t_reorth=Lz["t_reorth"], t_small=t_small,
                     t_other=t_lz - t_act - Lz["t_reorth"] + t_comb, t_total=t_lz + t_small + t_comb,
                     theta_min=float(th[0]), theta_max=float(th[-1]),
                     off_pi=float(np.linalg.norm(raw - out) / nrm) if nrm > 0 else 0.0)


def timed_step(box, q, p, xi, dt):
    N, L = box.N, box.L
    t0 = time.perf_counter()
    qh = (q + 0.5 * dt / MASS * p) % L
    op = box.gamma(qh)
    t_setup = time.perf_counter() - t0
    t1 = time.perf_counter()
    pm = p.mean(axis=0)
    pp = proj(p.ravel())
    zx = proj(xi.ravel())
    t_vec = time.perf_counter() - t1
    damp, d = timed_apply(op, pp, RANK_DAMP, f_damp(dt))
    noise, n = timed_apply(op, zx, RANK_NOISE, f_noise(dt))
    t2 = time.perf_counter()
    p_new = pm[None, :] + (proj(damp) + proj(noise)).reshape(N, 3)
    q_new = (qh + 0.5 * dt / MASS * p_new) % L
    t_vec += time.perf_counter() - t2
    t_total = time.perf_counter() - t0
    P0 = p.sum(axis=0)
    info = dict(t_setup=t_setup, t_vec=t_vec, t_step=t_total, t_thermo=t_total - t_setup,
                n_actions=op.n_actions, damp=d, noise=n,
                mom_res=float(np.linalg.norm(p_new.sum(axis=0) - P0) / max(1.0, np.linalg.norm(P0))))
    return q_new, p_new, info


def time_N(N, dt, warm, steps, say):
    box = Box(N)
    q, p, rng = initial_state(N, box.L, 7000)
    xis = rng.normal(size=(warm + steps, N, 3))          # drawn before timing
    infos = []
    for k in range(warm + steps):
        q, p, info = timed_step(box, q, p, xis[k], dt)
        if k >= warm:
            infos.append(info)
    med = lambda f: float(np.median([f(i) for i in infos]))     # noqa: E731
    mx = lambda f: float(np.max([f(i) for i in infos]))         # noqa: E731
    ts = np.array([i["t_step"] for i in infos])
    row = dict(
        N=N, L=box.L, mesh_M=int(box.mesh.M) if hasattr(box.mesh, "M") else None,
        noise_rank=RANK_NOISE, damping_rank=RANK_DAMP, warmup_steps=warm, timed_steps=steps,
        actual_action_count=med(lambda i: i["n_actions"]),
        action_count_min=int(min(i["n_actions"] for i in infos)), action_count_max=int(max(i["n_actions"] for i in infos)),
        setup_time=med(lambda i: i["t_setup"]),
        action_time=med(lambda i: i["damp"]["t_action"] + i["noise"]["t_action"]),
        time_per_action=med(lambda i: (i["damp"]["t_action"] + i["noise"]["t_action"]) / i["n_actions"]),
        damping_time=med(lambda i: i["damp"]["t_total"]),
        damping_action_time=med(lambda i: i["damp"]["t_action"]),
        damping_reorth_time=med(lambda i: i["damp"]["t_reorth"]),
        damping_small_matrix_time=med(lambda i: i["damp"]["t_small"]),
        damping_other_vector_time=med(lambda i: i["damp"]["t_other"]),
        noise_time=med(lambda i: i["noise"]["t_total"]),
        noise_action_time=med(lambda i: i["noise"]["t_action"]),
        noise_reorth_time=med(lambda i: i["noise"]["t_reorth"]),
        noise_small_matrix_time=med(lambda i: i["noise"]["t_small"]),
        noise_other_vector_time=med(lambda i: i["noise"]["t_other"]),
        reorth_time=med(lambda i: i["damp"]["t_reorth"] + i["noise"]["t_reorth"]),
        projection_vector_overhead=med(lambda i: i["t_vec"] + i["damp"]["t_other"] + i["noise"]["t_other"]),
        total_thermostat_time=med(lambda i: i["t_thermo"]),
        total_step_time=float(np.median(ts)),
        step_time_min=float(ts.min()), step_time_max=float(ts.max()),
        step_time_rel_spread=float((ts.max() - ts.min()) / np.median(ts)),
        step_times=ts.tolist(),
        momentum_residual=mx(lambda i: i["mom_res"]),
        off_pi_noise=mx(lambda i: i["noise"]["off_pi"]), off_pi_damp=mx(lambda i: i["damp"]["off_pi"]),
        theta_min_noise=float(min(i["noise"]["theta_min"] for i in infos)),
        theta_min_damp=float(min(i["damp"]["theta_min"] for i in infos)),
        dt_lambda_max=float(dt * max(max(i["noise"]["theta_max"], i["damp"]["theta_max"]) for i in infos)),
        negative_ritz=0,
    )
    row["t_over_NlnN"] = row["total_step_time"] / (N * np.log(N))
    row["old_rank12_step_time"] = OLD[N]
    row["ratio_new_over_old"] = row["total_step_time"] / OLD[N]
    say(f"N={N}: step {row['total_step_time']:.3f} s (min {ts.min():.3f}, max {ts.max():.3f}); actions "
        f"{row['actual_action_count']:.0f}; setup {row['setup_time']:.3f}; actions {row['action_time']:.3f}; reorth "
        f"{row['reorth_time']:.3f}; momentum {row['momentum_residual']:.1e}; offPi {row['off_pi_noise']:.1e}; "
        f"theta_min {row['theta_min_noise']:.3e}/{row['theta_min_damp']:.3e}; new/old {row['ratio_new_over_old']:.3f}")
    return row


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = np.array([r["N"] for r in rows], float)
    T = np.array([r["total_step_time"] for r in rows])
    To = np.array([r["old_rank12_step_time"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    a.loglog(N, T, "o-", color=PAL[0], label="step, noise 40 + damping 16")
    a.loglog(N, [r["total_thermostat_time"] for r in rows], "s--", color=PAL[2], ms=4, label="thermostat part")
    a.loglog(N, [r["action_time"] for r in rows], "^:", color=PAL[3], ms=4, label="Γ_h actions only")
    a.loglog(N, To, "x", color=PAL[1], label="historical: noise 40 + damping 12")
    Ng = np.geomspace(N[0], N[-1], 50)
    a.loglog(Ng, T[0] * Ng / N[0], color="0.6", lw=0.8, label="∝ N")
    a.loglog(Ng, T[0] * Ng * np.log(Ng) / (N[0] * np.log(N[0])), color="0.3", lw=0.8, ls="--", label="∝ N ln N")
    a.set_xlabel("N"), a.set_ylabel("median wall time per step [s]"), a.legend(fontsize=8)
    a.set_title("single-process step time")
    b = ax[1]
    b.semilogx(N, T / (N * np.log(N)), "o-", color=PAL[0], label="noise 40 + damping 16")
    b.semilogx(N, To / (N * np.log(N)), "x--", color=PAL[1], label="historical 40 + 12")
    b.set_ylim(0, None), b.set_xlabel("N"), b.set_ylabel("T / (N ln N) [s]"), b.legend(fontsize=8)
    b.set_title("normalised cost")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--N", type=int, nargs="+", default=[512, 4000, 16000])
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--steps-if-noisy", type=int, default=20)
    ap.add_argument("--noisy", type=float, default=0.25, help="relative (max - min)/median spread triggering a rerun")
    args = ap.parse_args()
    dt = json.loads((OUT / "params_small.json").read_text())["dt"]
    say = lambda s: print(s, flush=True)     # noqa: E731
    rows = []
    for N in args.N:                          # sequential: one N at a time in this single process
        row = time_N(N, dt, args.warm, args.steps, say)
        if row["step_time_rel_spread"] > args.noisy:
            say(f"N={N}: spread {row['step_time_rel_spread']:.2f} > {args.noisy}; repeating with {args.steps_if_noisy} steps")
            row = time_N(N, dt, args.warm, args.steps_if_noisy, say)
        rows.append(row)
    N = np.array([r["N"] for r in rows], float)
    T = np.array([r["total_step_time"] for r in rows])
    alpha, lnC = np.polyfit(np.log(N), np.log(T), 1)
    alpha_old = np.polyfit(np.log(N), np.log([OLD[int(n)] for n in N]), 1)[0]
    summary = dict(dt=dt, log_convention="natural log", noise_rank=RANK_NOISE, damping_rank=RANK_DAMP,
                   threads="OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, single process, sequential N",
                   alpha_fit=float(alpha), C_fit=float(np.exp(lnC)), alpha_fit_old_same_N=float(alpha_old),
                   t_over_NlnN_ratio_max_min=float((T / (N * np.log(N))).max() / (T / (N * np.log(N))).min()),
                   rows=rows)
    (OUT / "production_timing_rank40_16.json").write_text(json.dumps(summary, indent=1) + "\n")
    cols = ["N", "noise_rank", "damping_rank", "actual_action_count", "setup_time", "action_time", "time_per_action",
            "damping_time", "damping_action_time", "damping_reorth_time", "damping_small_matrix_time",
            "noise_time", "noise_action_time", "noise_reorth_time", "noise_small_matrix_time", "reorth_time",
            "projection_vector_overhead", "total_thermostat_time", "total_step_time", "step_time_min",
            "step_time_max", "t_over_NlnN", "old_rank12_step_time", "ratio_new_over_old", "momentum_residual",
            "off_pi_noise", "theta_min_noise", "theta_min_damp", "timed_steps"]
    with open(OUT / "production_timing_rank40_16.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    plot(rows, OUT / "production_timing_rank40_16.png")
    say(f"alpha = {alpha:.3f} (old 40+12 on same N: {alpha_old:.3f}); "
        "T/(N ln N) = " + ", ".join(f"{r['t_over_NlnN']:.3e}" for r in rows))


if __name__ == "__main__":
    main()
