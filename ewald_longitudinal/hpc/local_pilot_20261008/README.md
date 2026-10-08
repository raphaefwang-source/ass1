# Local pilot, 2026-10-08 (emulated, not on an HPC system)

The pilot script `hpc/pilot.sbatch` was run under bash in the development container, with no Slurm, using a local
`cluster.env`. This is not an HPC validation; repeat the pilot on the cluster (`hpc/README.md`).

**Machine:** Intel Xeon @ 2.10 GHz, 4 cores visible; 1 thread per run (OPENBLAS/OMP/MKL = 1); Python 3.13.16,
NumPy 2.5.3, SciPy 1.18.1 (`env_report.json`). This VM was 1.2–1.4× slower than the one used for
`cost_optimization_results/timing.json`: the same benchmark cases, rerun on it, gave 605 vs 500 ms and 441 vs 314 ms.

| run | ranks | steps | step time median (p90) | peak RSS | checks |
|---|---|---|---|---|---|
| double_well A, N = 512, costopt_hiacc | 40/5 | 200 | 666 ms (751) | 126 MB | PASS |
| double_well B, N = 512, costopt_hiacc | 24/8 | 200 | 458 ms (527) | 125 MB | PASS |
| double_well A, N = 512, baseline_hiacc (original) | 40/16 | 30 | 2324 ms (2610) | 1135 MB | (timing only) |
| double_well B, N = 512, baseline_hiacc (original) | 40/16 | 30 | 2347 ms (2499) | 1136 MB | (timing only) |

**Speedup, same machine and same runner:** ×3.5 (A) and ×5.1 (B) against the original high-accuracy version (image
pairs, all-pair forces, 40/16) (`same_machine_comparison.json`).

**Restart consistency, all bitwise** (q, p, RNG state, diagnostics except wall time, frames, monitor rows):
- N = 64 law A, 60 steps split 25 + 20 + rest: PASS (`restart_check_segments.json`);
- N = 64 law B, 400 steps stopped by SIGTERM: PASS (`restart_check_signal.json`);
- `array.sbatch` with USR1 forwarding, resume and a no-op rerun: PASS (`array_signal_resume_check.json`).

**Resource estimate** (`resource_estimate.json`):
- default task list, 40 tasks, burn-in t = 140 + production t = 60, 40 000 steps each;
- 184 core-hours of run time on this VM, 276 core-hours with a 1.5 margin;
- N = 256 is scaled from the N = 512 pilot by the earlier benchmark ratio, not measured;
- queue waiting time cannot be estimated here.

The cluster pilot replaces these numbers.

Files: `config.json`, `status.json` and `run.log` of the two pilot runs, `pilot_check.json`, `pilot_stdout.log`.
Paths are rewritten as `$RUN_ROOT` / `$REPO_DIR`. Raw npz output is not kept.
