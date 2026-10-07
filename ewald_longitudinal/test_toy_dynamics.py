#!/usr/bin/env python3
"""
Fast (PPPM + Lanczos) vs reference (dense full-periodic) dynamics for the two toy potentials and two radial laws.

Stages
    --stage run --tag coupled_short [--steps 2000]   coupled runs: for every potential x law x seed, the three
                                                     methods of toy_dynamics start from the same equilibrated state
                                                     and consume the same Gaussian stream
    --stage analyze --tag TAG                        pathwise divergence, VACF, distance-binned VCCF, longitudinal /
                                                     transverse current correlations, distinct van Hove; paired
                                                     method differences per seed and their SEM across seeds
Initial states: final frames of the v2 full-lattice runs (toy_models/v2_results/restart_checkpoints/*_burn140/lattice,
effective burn-in 140 + production 60 under the reference model). The fast operator differs from the reference by
<= 1.3e-7 (radial_kernel_static_results), so these states are equally valid starting points for both. This does NOT
prove equilibrium; see the v2 notes for the slow-relaxation caveats.
Raw trajectories go to toy_dynamics_raw/ (git-ignored); figures and statistics to toy_dynamics_results/.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import toy_dynamics as td  # noqa: E402
import prl_toy_models as v2  # noqa: E402
from ensemble import VCCF_EDGES, mean_squared_displacement  # noqa: E402

RAW = HERE / "toy_dynamics_raw"
OUT = HERE / "toy_dynamics_results"
CKPT = HERE / "toy_models" / "v2_results" / "restart_checkpoints"
TOY = dict(gamma=0.5, kappa=0.7, r_ref=1.3, dt=0.005, kT=0.7, mass=1.0, L=5.5)
POTENTIALS, LAWS = ("lj", "double_well"), ("A", "B")
COL = {"A": "#2563eb", "B": "#dc7627"}
LS = {"reference": "-", "pppm_dense": "--", "pppm_lanczos": ":"}
NOISE_SEED_OFFSET = 3000


def checkpoint(potential, law, seed):
    with np.load(CKPT / f"{potential}_burn140" / "lattice" / f"{potential}_{law}" / f"seed_{seed}" / "restart.npz") as ck:
        return ck["Q"].copy(), ck["p"].copy(), float(ck["box"][0])


def raw_path(tag, potential, law, method, seed):
    return RAW / tag / f"{potential}_{law}" / f"{method}_seed{seed}.npz"


def run_job(job):
    tag, potential, law, method, seed, steps, stride = job
    path = raw_path(tag, potential, law, method, seed)
    if path.exists():
        return str(path), "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    q0, p0, L = checkpoint(potential, law, seed)
    th = td.Thermostat(method, L, law, TOY["gamma"], TOY["kappa"], TOY["r_ref"], TOY["dt"], TOY["kT"], TOY["mass"])
    out = td.run(potential, th, q0, p0, steps=steps, stride=stride, seed=seed + NOISE_SEED_OFFSET)
    meta = json.loads(str(out["metadata"]))
    meta.update(initial_state=f"v2 restart checkpoint {potential}_burn140/lattice/{potential}_{law}/seed_{seed}",
                effective_prior_time=200.0, tag=tag)
    out["metadata"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **out)
    return str(path), f"{meta['wall_time']:.0f}s"


def stage_run(args):
    jobs = [(args.tag, pot, law, m, s, args.steps, args.stride) for pot in args.potentials for law in args.laws
            for s in args.seeds for m in args.methods]
    # slowest first so the pool stays busy
    jobs.sort(key=lambda j: {"pppm_lanczos": 0, "pppm_dense": 1, "reference": 2}[j[3]])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path, status in pool.map(run_job, jobs):
            print(f"done {path} ({status})", flush=True)


def observables(tr, potential, max_lag_time, origins):
    frame = float(tr["t"][1] - tr["t"][0])
    max_lag = int(round(max_lag_time / frame))
    obs = v2.compute_observables(tr["Q"], tr["V"], tr["t"], tr["box"], max_lag=max_lag, n_lags=max_lag + 1,
                                 origins=origins, vccf_edges=VCCF_EDGES[potential])
    obs["msd"] = mean_squared_displacement(tr["Q"], obs["time_origins"], obs["lag_frames"])
    return obs


KEYS = ("vacf", "vccf_normalized", "hydro_L", "hydro_T", "van_hove_distinct", "rdf", "msd")


def stage_analyze(args):
    root = RAW / args.tag
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    report = dict(tag=args.tag, toy=TOY, max_lag_time=args.max_lag_time, origins=args.origins, cases={})
    stats = {}
    for pot in POTENTIALS:
        for law in LAWS:
            case = f"{pot}_{law}"
            files = sorted((root / case).glob("*_seed*.npz"))
            if not files:
                continue
            seeds = sorted({int(f.stem.split("_seed")[1]) for f in files})
            methods = [m for m in td.METHODS if all((root / case / f"{m}_seed{s}.npz").exists() for s in seeds)]
            data, path_div, diag = {}, {}, {}
            for m in methods:
                obs_list, d_list = [], []
                for s in seeds:
                    with np.load(root / case / f"{m}_seed{s}.npz") as tr:
                        tr = {k: tr[k] for k in tr.files}
                    obs_list.append(observables(tr, pot, args.max_lag_time, args.origins))
                    d_list.append(tr)
                data[m] = obs_list
                diag[m] = d_list
            # pathwise divergence from the reference (same start, same noise stream)
            for m in methods:
                if m == "reference":
                    continue
                base = "pppm_dense" if m == "pppm_lanczos" and "pppm_dense" in methods else "reference"
                for b in sorted({base, "reference"}):
                    dq = np.stack([np.abs(diag[m][i]["Q"] - diag[b][i]["Q"]).max(axis=(1, 2)) for i in range(len(seeds))])
                    dv = np.stack([np.abs(diag[m][i]["V"] - diag[b][i]["V"]).max(axis=(1, 2)) for i in range(len(seeds))])
                    path_div[f"{m}-{b}"] = dict(t=diag[m][0]["t"].tolist(), dq_max=dq.max(axis=0).tolist(),
                                                dv_max=dv.max(axis=0).tolist())
            st = {m: {k: np.stack([np.asarray(o[k], float) for o in data[m]]) for k in KEYS} for m in methods}
            for m in methods:
                for k in ("hydro_L", "hydro_T"):
                    st[m][k + "_mode_mean"] = st[m][k].mean(axis=-1)
            stats[case] = dict(st=st, lag=data[methods[0]][0]["lag_time"], r=data[methods[0]][0]["radial_centers"],
                               edges=data[methods[0]][0]["vccf_edges"], methods=methods, path=path_div)
            # paired differences per seed, SEM across seeds
            comp = {}
            for m in methods:
                if m == "reference":
                    continue
                cmp_ = {}
                for k in ("vacf", "vccf_normalized", "hydro_L_mode_mean", "hydro_T_mode_mean", "van_hove_distinct", "rdf"):
                    # Paired difference (same start, same noise) compared with the sampling error bar of the
                    # observable itself (reference SEM across seeds). Points where the reference has no spread
                    # (t = 0 normalisation, empty van Hove bins) carry no statistical information and are skipped.
                    diff = st[m][k] - st["reference"][k]
                    mean = diff.mean(axis=0)
                    sem_ref = st["reference"][k].std(axis=0, ddof=1) / np.sqrt(len(seeds))
                    scale = np.nanmax(np.abs(st["reference"][k]))
                    ok = np.isfinite(mean) & np.isfinite(sem_ref) & (sem_ref > 1e-12 * scale)
                    ratio = np.abs(mean[ok]) / sem_ref[ok]
                    cmp_[k] = dict(max_abs_paired_diff=float(np.nanmax(np.abs(mean))),
                                   max_reference_sem=float(np.nanmax(sem_ref)),
                                   max_diff_over_reference_sem=float(ratio.max()),
                                   fraction_within_2_reference_sem=float(np.mean(ratio <= 2.0)))
                comp[m] = cmp_
            Ts = {m: [float(np.mean(d["diagnostics"][:, 1])) for d in diag[m]] for m in methods}
            Us = {m: [float(np.mean(d["diagnostics"][:, 2])) for d in diag[m]] for m in methods}
            report["cases"][case] = dict(seeds=seeds, methods=methods, comparisons=comp,
                                         T_mean={m: [float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))] for m, v in Ts.items()},
                                         U_per_particle={m: [float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))] for m, v in Us.items()},
                                         max_total_momentum={m: float(max(d["diagnostics"][:, 4].max() for d in diag[m])) for m in methods},
                                         lam_range={m: [float(min(json.loads(str(d["metadata"]))["lam_range"][0] for d in diag[m])),
                                                        float(max(json.loads(str(d["metadata"]))["lam_range"][1] for d in diag[m]))] for m in methods},
                                         wall_time_per_step={m: float(np.mean([json.loads(str(d["metadata"]))["wall_time"] / json.loads(str(d["metadata"]))["steps"] for d in diag[m]])) for m in methods},
                                         pathwise=path_div)
            print(f"[{case}] seeds {seeds} methods {methods}", flush=True)
            for m, c in comp.items():
                print("   " + m + ": " + "  ".join(f"{k}: {v['max_abs_paired_diff']:.1e} (max {v['max_diff_over_reference_sem']:.2f} ref-SEM, "
                                                 f"{100 * v['fraction_within_2_reference_sem']:.0f}% within 2)" for k, v in c.items()), flush=True)
    (out / "comparison.json").write_text(json.dumps(report, indent=1) + "\n")
    figures(stats, out, args.tag)


def figures(stats, out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for pot in POTENTIALS:
        cases = [c for c in stats if c.startswith(pot)]
        if not cases:
            continue
        fig, axes = plt.subplots(3, 4, figsize=(18, 11), layout="constrained")
        panels = [("vacf", "VACF"), ("vccf_normalized", "VCCF / <|v|^2>, first shell"),
                  ("hydro_L_mode_mean", "C_L, |k| = 2pi/L"), ("hydro_T_mode_mean", "C_T, |k| = 2pi/L")]
        for case in cases:
            law = case.split("_")[-1]
            S = stats[case]
            t, n = S["lag"], S["st"]["reference"]["vacf"].shape[0]
            for col, (key, title) in enumerate(panels):
                for m in S["methods"]:
                    y = S["st"][m][key]
                    if key == "vccf_normalized":
                        y = y[..., 0]
                    axes[0, col].plot(t, y.mean(axis=0), color=COL[law], ls=LS[m], lw=1.6 if m == "reference" else 1.2,
                                      label=f"{law}, {m}")
                    if m != "reference":
                        yr = S["st"]["reference"][key]
                        if key == "vccf_normalized":
                            yr = yr[..., 0]
                        diff = y - yr
                        mean = diff.mean(axis=0)
                        sem_ref = yr.std(axis=0, ddof=1) / np.sqrt(n)
                        ax = axes[1 if law == "A" else 2, col]
                        if m == "pppm_lanczos":
                            ax.fill_between(t, -2 * sem_ref, 2 * sem_ref, color=".75", alpha=.5, lw=0,
                                            label="+/-2 SEM of the reference (sampling)")
                        ax.plot(t, mean, color=COL[law], ls=LS[m], label=f"{m} - reference (paired mean)")
                axes[0, col].set(title=title, xlabel="lag time", xlim=(0, 1.5))
        for col in range(4):
            axes[0, col].axhline(0, color=".6", lw=.6)
            for row in (1, 2):
                axes[row, col].axhline(0, color=".6", lw=.6)
                axes[row, col].set(xlabel="lag time", xlim=(0, 1.5),
                                   title=f"law {'A' if row == 1 else 'B'}: fast - reference (grey: +/-2 ref. SEM)")
                axes[row, col].legend(fontsize=7)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{pot}: fast PPPM operator vs dense full-periodic reference ({tag}); coupled seeds, "
                     f"paired differences. Toy model, no MD/PRL validation.", fontsize=11)
        fig.savefig(out / f"fig_{pot}_correlations.png", dpi=140)
        plt.close(fig)
        # van Hove and RDF
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
        for case in cases:
            law = case.split("_")[-1]
            S = stats[case]
            t, r, n = S["lag"], S["r"], S["st"]["reference"]["vacf"].shape[0]
            for m in S["methods"]:
                for target, alpha in ((0.0, 1.0), (0.5, .7), (2.0, .45)):
                    k = int(np.argmin(np.abs(t - target)))
                    if t[k] < target - 0.05:
                        continue
                    axes[0, 0 if law == "A" else 1].plot(r, S["st"][m]["van_hove_distinct"][:, k].mean(axis=0), color=COL[law],
                                                         ls=LS[m], alpha=alpha, label=f"{m}, t={t[k]:.2f}")
                if m != "reference":
                    for target in (0.0, 0.5):
                        k = int(np.argmin(np.abs(t - target)))
                        yr = S["st"]["reference"]["van_hove_distinct"][:, k]
                        mean = (S["st"][m]["van_hove_distinct"][:, k] - yr).mean(axis=0)
                        sem_ref = yr.std(axis=0, ddof=1) / np.sqrt(n)
                        ax = axes[1, 0 if law == "A" else 1]
                        if m == "pppm_lanczos":
                            ax.fill_between(r, -2 * sem_ref, 2 * sem_ref, color=".75", alpha=.35, lw=0)
                        ax.plot(r, mean, color=COL[law], ls=LS[m], label=f"{m} - ref, t={t[k]:.2f}")
            for row in (0, 1):
                axes[row, 0 if law == "A" else 1].set(xlabel="minimum-image distance r", title=f"law {law}: "
                                                      + ("distinct van Hove g_d(r,t)" if row == 0 else "fast - reference (grey: +/-2 ref. SEM)"))
                axes[row, 0 if law == "A" else 1].legend(fontsize=7)
        fig.suptitle(f"{pot}: distinct van Hove, fast vs reference ({tag})", fontsize=11)
        fig.savefig(out / f"fig_{pot}_van_hove.png", dpi=140)
        plt.close(fig)
    # pathwise divergence
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
    for case, S in stats.items():
        law = case.split("_")[-1]
        for key, d in S["path"].items():
            ls = "-" if case.startswith("lj") else "--"
            ax = axes[0] if key.endswith("-reference") else axes[1]
            ax.semilogy(d["t"], np.maximum(d["dv_max"], 1e-17), color=COL[law], ls=ls, label=f"{case}: {key}")
    axes[0].set(xlabel="time since common start", ylabel="max_i |v_i - v_i^ref|", title="PPPM methods vs dense full-periodic reference")
    axes[1].set(xlabel="time since common start", ylabel="max_i |v_i^Lanczos - v_i^PPPM dense|", title="Krylov (rank 40/16) vs dense, same PPPM Gamma_h")
    for ax in axes:
        ax.legend(fontsize=6)
    fig.suptitle("Pathwise divergence of coupled runs (same start, same noise); growth is chaotic amplification of the operator difference", fontsize=10)
    fig.savefig(out / "fig_pathwise_divergence.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["run", "analyze"], required=True)
    ap.add_argument("--tag", default="coupled_short")
    ap.add_argument("--potentials", nargs="+", default=list(POTENTIALS), choices=POTENTIALS)
    ap.add_argument("--laws", nargs="+", default=list(LAWS), choices=LAWS)
    ap.add_argument("--methods", nargs="+", default=list(td.METHODS), choices=td.METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-lag-time", type=float, default=3.0)
    ap.add_argument("--origins", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    if args.stage == "run":
        stage_run(args)
    else:
        stage_analyze(args)


if __name__ == "__main__":
    main()
