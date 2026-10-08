# Local emulation of the HPC pilot, 2026-10-08 (not on an HPC system)

`hpc/pilot_task.sbatch` run under bash in the development container, with no Slurm, once per kernel:

```bash
PILOT_STEPS=300 PILOT_LOCAL_SIGNAL_AFTER=60 KERNEL=A TOY_ENV_FILE=<local cluster.env> TOY_HPC_DIR=hpc bash hpc/pilot_task.sbatch
```

`PILOT_LOCAL_SIGNAL_AFTER` makes the script send USR1 to its own batch shell, standing in for Slurm's
`--signal=B:USR1@2160`. The file system was the container's ext4. Paths are rewritten to `$RUN_ROOT` / `$REPO_DIR`.
The code was the working tree just before the commit that added this directory (`dirty=True` in the logs).

**This is not an HPC validation.** It shows that the script's sequence works: signal → exit 75, duplicate start
refused with exit 3, resume to completion, checks, summary. Scheduler signal delivery, cluster file-system locking,
the cluster's software build and its step times are tested only by the pilot on the cluster (`hpc/README.md`,
Section 2).

| kernel | steps | segments (exit) | duplicate | step median (p90) | peak RSS | accuracy warnings | error estimate max | job wall | accepted |
|---|---|---|---|---|---|---|---|---|---|
| A, 40/5, M 36 | 300 | 75 (SIGUSR1 at step 91), 75 (200-step limit), 0 | exit 3 | 649 ms (727) | 129 MB | 0 | 9.6e-11 | 219 s | yes |
| B, 24/8, M 40 | 300 | 75 (SIGUSR1 at step 124), 0 | exit 3 | 463 ms (529) | 128 MB | 0 | 1.3e-14 | 160 s | yes |

For B the signal came late enough that segment 2 reached the end within its 200-step limit; the duplicate start was
still tested while segment 2 held the lock.

**Files per kernel:** `pilot_summary.json`, `pilot_check.json` (check_run), `lock_check.json`, `env_report.json`,
`restart_check.log` (N = 64, bitwise PASS), `run_status.json`, `run.log`, `duplicate.log`, `job_stdout.log`.

`resource_estimate.json`: `estimate_resources.py` on these two runs with the default 40-task list (every-step frames,
margin 1.5): ≈ 182 core-hours of run time, ≈ 273 with margin, ≈ 8.4 GB of disk. N = 256 is scaled, not measured.

`round_budget.json`: local compute used in this round, ≈ 0.74 core-hours (partly estimated); 0 on HPC.
