# Running the toy friction dynamics on an HPC cluster (Slurm)

**What has and has not been run.** Nothing in this directory has been run on an HPC system yet. The development
container has no cluster connection (no ssh target, no Slurm), so the two pilot jobs of Section 2 have **not** been
submitted. Locally, with no Slurm, the following have been checked:
- the runner tests (`../test_toy_run.py`, 10 tests, including the directory lock and `status.py`);
- `lock_check.py` on the container's own file system (ext4);
- `pilot_task.sbatch` run under bash for A and B with 300 steps and a self-sent USR1 instead of Slurm's signal
  (`local_pilot_emulation_20261008/`);
- earlier: the old pilot and the array script under bash, and the submission wrapper with a fake `sbatch`
  (`local_pilot_20261008/`, historical).

None of these local runs is an HPC validation. Scheduler signals, the cluster file system's locking, the cluster's
Python/BLAS build and its step times are checked only by the pilot on the cluster.

**Scope:** no long runs were started. The model is unchanged: pure longitudinal friction, full periodic images,
k = 0, g(r_ref) = 0.5, kT = 0.7, dt = 0.005, density 64/5.5³. No random batch.

## Files

| file | purpose |
|---|---|
| `../toy_configs.py` | named configurations; every parameter explicit, N-dependent Lanczos ranks, refusals |
| `../toy_run.py` | unified, restartable runner (entry point) |
| `../run_lock.py` | per-run-directory lock held for the whole life of a `toy_run.py` process |
| `../verify_production_configs.py` | static confirmation of the configurations at exactly their parameters |
| `../test_toy_run.py` | runner tests (bitwise vs `toy_dynamics.run`, restart, signals, overwrite protection, lock, status) |
| `../vacf_sampling_check.py` | choice of the save interval from every-step VACFs (`../vacf_sampling_results/REPORT.md`) |
| `cluster.env.example` | the cluster settings you fill in (copy to `cluster.env`, which git ignores) |
| `setup_venv.sh`, `requirements.txt`, `requirements-range.txt` | Python environment: pinned versions (Python ≥ 3.12) or version ranges |
| `submit.sh` | submission wrapper; `pilot` submits the two pilot jobs, `array` only prints unless `--submit` |
| `pilot_task.sbatch` | one pilot job (one kernel): speed, memory, accuracy monitor, lock, signal and resume |
| `lock_check.py` | checks that the run-directory lock works on a given file system |
| `array.sbatch` | one trajectory per array task, always `--resume` |
| `make_tasklist.py` | potential × kernel × N × seed task list (TSV) |
| `status.py` | per-task state (active / resumable / unknown / ...) and the `--array` string of what is safe to resubmit |
| `check_run.py` | acceptance checks of finished runs |
| `restart_check.py` | continuous vs stopped-and-resumed run, bitwise |
| `estimate_resources.py` | `--time` / `--mem` / core-hours / disk from pilot runs |
| `env_report.py` | CPU, cores, Python, NumPy/SciPy, BLAS, threads |
| `tasks/production_N256_N512.tsv` | default plan, 40 tasks (see below) |

## Configurations (`toy_configs.py`)

| name | law | PPPM ξ, s, η*, p | ranks (noise/damping), N ≤ 256 / N ≤ 512 | pairs | forces |
|---|---|---|---|---|---|
| `costopt_hiacc` | A (budget 2e-7) | 0.85, 4.1, 0.6779, 7 | 32/5 / 40/5 | KD-tree | neighbour list |
| `costopt_hiacc` | B (budget 5e-8) | 1.0, 3.9, 0.6828, 7 | 16/6 / 24/8 | KD-tree | neighbour list |
| `baseline_hiacc` | A, B | 0.7, 4.10, 0.7, 7 | 40/16 | image enumeration | all pairs |

**Why the costopt ranks are higher than in the report.** They exceed the cost-optimisation report's ranks
(A 32/4, B 12/5). Those were chosen on replicated states. A liquid-like test state has a much wider friction
spectrum, and B 12/5 fails on it. The ranks here meet budget/10 on every test state
(`../cost_optimization_results/REPORT.md`, Section 10).

**`baseline_hiacc`** reproduces runs made before 2026-10-08.

**Refusals and records:**
- `costopt_hiacc` refuses N > 512 and dt ≠ 0.005 unless `--allow-unverified` is given; explicit `--ranks` also
  needs it.
- At start-up the runner prints and stores every parameter: mesh M, η_actual, r_c, k_c, k = 0, ranks with
  provenance, pair search, force method, verification status, hashes, git commit and thread settings.

## 1. Environment (once, on a login node)

```bash
git clone https://github.com/raphaefwang-source/ass1.git && cd ass1 && git checkout claude/new-session-nmi9ux
# Python >= 3.12 for the pinned versions (requirements.txt); otherwise add --range (requirements-range.txt)
bash ewald_longitudinal/hpc/setup_venv.sh $HOME/venvs/toy python3            # prints the PY_SETUP line
bash ewald_longitudinal/hpc/setup_venv.sh $HOME/venvs/toy python3.11 --range  # e.g. for Python 3.10 / 3.11
cp ewald_longitudinal/hpc/cluster.env.example ewald_longitudinal/hpc/cluster.env
$EDITOR ewald_longitudinal/hpc/cluster.env                                    # replace every FILL_ME
python3 ewald_longitudinal/hpc/lock_check.py --dir <RUN_ROOT>                 # optional, on the login node
```

**Which versions have been checked:**
- **Pinned set:** Python 3.13.16 with NumPy 2.5.3, SciPy 1.18.1 and Matplotlib 3.11.2 (`requirements.txt`). Every
  local check of this round used it.
- **Range set:** Python 3.11.17 with NumPy 2.4.6 and SciPy 1.17.1 (`requirements-range.txt`); `test_toy_run.py`
  passed in an earlier round.
- **Python 3.10 and older:** the code parses with the 3.9 grammar, but nothing has been run.

Results from different NumPy/SciPy/BLAS builds are not bitwise comparable with each other. Restart consistency on
the cluster's own installation is checked by the pilot.

Use your site's module or conda instead of a venv if that is the norm there.

## 2. Pilot: two short jobs (the only jobs to submit for now)

```bash
cd ass1/ewald_longitudinal
bash hpc/submit.sh pilot --dry-run      # prints the two sbatch commands (job names toy_pilot_A, toy_pilot_B)
bash hpc/submit.sh pilot                # submits both; each 1 task, 1 core, 1 thread, 2 GB, 40 min limit
squeue --me
sacct -j <JOBID_A>,<JOBID_B> --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,ReqMem,Timelimit
less $LOG_ROOT/toy_pilot_A-<JOBID_A>.out
cat  $RUN_ROOT/pilot/<JOBID_A>_A/pilot_summary.json $RUN_ROOT/pilot/<JOBID_B>_B/pilot_summary.json
```

**Each job** (`pilot_task.sbatch`, `KERNEL=A` or `B`) runs N = 512, `costopt_hiacc`, double well, seed 101,
1500 steps, frames every step, monitor every 50 steps, in `$RUN_ROOT/pilot/<JOBID>_<K>/`:
1. `env_report.json`; `lock_check.py` on `$RUN_ROOT/pilot` (`lock_check.json`).
2. Segment 1 in the background behind the same USR1 trap as `array.sbatch`. `--signal=B:USR1@2160` makes Slurm
   signal the batch shell about 4 min after the start (up to 60 s earlier); the run must checkpoint and exit 75.
3. Segment 2 (`--resume --max-segment-steps 200`). While it holds the lock, a second `toy_run.py --resume` on the
   same directory must exit 3 and change nothing (`duplicate.log`).
4. Resume to completion (exit 0).
5. `check_run.py` (`pilot_check.json`) and `restart_check.py` (N = 64, bitwise; `restart_check/`).
6. `pilot_summary.json`; the job exits 0 only if the pilot is accepted.

**Pilot acceptance.** In each `pilot_summary.json`, `"accepted": true`, which requires:
1. `signal_exercised`: segment 1 exited 75 with stop reason `signal SIGUSR1`, i.e. Slurm's signal reached the run.
2. `duplicate_exit_code` 3: the directory lock refused the second process.
3. `segment_exit_codes.final` 0 and `check_run_passed`:
   - config `costopt_hiacc`, ranks 40/5 (A) and 24/8 (B), KD-tree pairs, neighbour forces;
   - static verification PASS for these parameters;
   - all steps contiguous and finite, one frame per step;
   - total momentum drift ≤ 1e-9·√(N m kT); r_min > 0.45; mean temperature within 20 % of 0.7 (a sanity bound,
     not an equilibration test);
   - all Ritz values > 0 and Lanczos error estimates ≤ budget.
4. `restart_check_passed`: q, p, RNG, diagnostics, frames and monitor rows bitwise equal after a stop and resume.
5. `lock_check_passed`. Also read `lock_check.json["warnings"]`: a single job cannot detect node-local locks
   (Lustre `localflock`), so the mount options are reported there.
6. `accuracy_warnings` 0.

**Also check by hand:**
- `step_ms_median` of the same order as locally (A ≈ 0.65 s, B ≈ 0.46 s per step at N = 512, below);
- `MaxRSS` well below 2 GB (≈ 0.13 GB locally);
- `env_report.json` shows 1 thread for OPENBLAS/OMP/MKL and the expected Python/NumPy;
- `TotalCPU` ≈ `Elapsed` (one busy core).

**Expected cost on a node like the local VM:** A ≈ 17 min, B ≈ 12 min (1500 steps plus ≈ 20 s of checks), about
0.5 core-hours for both; at most 2 × 40 min = 1.3 core-hours reserved. Frames add ≈ 37 MB per job.

**If `signal_exercised` is false with stop reason `complete`,** the node was so fast that 1500 steps finished
before the signal. Resubmit with more steps: `PILOT_STEPS=4000 bash hpc/submit.sh pilot` (the variable is passed
through `--export=ALL`).

**Local emulation (not HPC):** `local_pilot_emulation_20261008/` holds the same script run under bash, 300 steps,
`PILOT_LOCAL_SIGNAL_AFTER=60` instead of Slurm's signal, on the working tree just before this commit. Both kernels
were accepted:

| kernel | step time median (p90) | peak RSS | accuracy warnings | Lanczos error estimate max | job wall |
|---|---|---|---|---|---|
| A, ranks 40/5, M 36 | 649 ms (727) | 129 MB | 0 | 9.6e-11 (budget 2e-7) | 219 s |
| B, ranks 24/8, M 40 | 463 ms (529) | 128 MB | 0 | 1.3e-14 (budget 5e-8) | 160 s |

## 3. Production plan and resources (prepare; do not submit yet)

```bash
python3 hpc/make_tasklist.py --out hpc/tasks/production_N256_N512.tsv
#   ids 1-10  N=256 A | 11-20 N=256 B | 21-30 N=512 A | 31-40 N=512 B
#   (double_well and lj x seeds 101-105; burn-in t=140, production t=60, q and v saved every step, t=0.005)
python3 hpc/estimate_resources.py \
    --pilot $RUN_ROOT/pilot/<JOBID_A>_A/double_well_A_N512_s101 $RUN_ROOT/pilot/<JOBID_B>_B/double_well_B_N512_s101 \
    --tasklist hpc/tasks/production_N256_N512.tsv --margin 1.5 [--max-job-hours <SITE LIMIT>] [--max-parallel 20]
```

**Save interval: every step.** Law B's VACF falls to 1/2 in about 3.4 steps. Saving every 0.01 already biases its
Green–Kubo integral by +1.5 to +3.6 %, and the old 0.1 by +170 to +430 % (`../vacf_sampling_results/REPORT.md`).

**What the estimate gives, per (N, kernel):**
- run time per task (40 000 steps × median pilot step time);
- `--time` (× margin, rounded up, + 10 min);
- `--mem` (2 × pilot peak RSS, at least 1 GB);
- core-hours;
- disk.

N = 256 is scaled from the N = 512 pilot by the earlier local benchmark ratio and is labelled as such.

**Local reference (local emulation, slow VM, 1.5 margin; `local_pilot_emulation_20261008/resource_estimate.json`):**
- per task: A N=512 ≈ 7.2 h run (`--time 11:10:00`); B N=512 ≈ 5.1 h (`07:55:00`); A N=256 ≈ 3.4 h (`05:25:00`,
  scaled); B N=256 ≈ 2.4 h (`03:55:00`, scaled); `--mem 1G`;
- all 40 tasks: ≈ 182 core-hours of run time, ≈ 273 core-hours with margin;
- disk ≈ 8.4 GB: ≈ 144 MB per N = 256 task and ≈ 284 MB per N = 512 task, almost all frames (2 × N × 3 × 8 bytes
  per step). The old 0.1 interval needed ≈ 0.4 GB.

**Queue waiting time is not part of any estimate.** It depends on your site.

**The initial states are not equilibrated:**
- N = 512 starts from 2×2×2 copies of a v2 N = 64 state, with 0.02 jitter and fresh Maxwell momenta;
- N = 256 starts from an fcc lattice with 0.05 jitter;
- both are recorded as `equilibrated: false`.

t = 140 of burn-in is an initial plan, not a guarantee. Check the diagnostics before using production data:
- T, U/N and S(k_min) in the chunks and monitor rows;
- the per-checkpoint log lines.

**Large boxes may cluster or phase-separate.** A liquid-like test state at N = 256 and 512 showed strong
long-wavelength density fluctuations (Section 10 of the report). The runner records S(k_min) every 500 steps. It
also records the Lanczos error estimates and warns in `run.log` and `status.json` if they exceed the budget or if
the spectrum falls below the verified range. A spectrum warning gives the error estimate at that step: a smallest
Ritz value below the verified range with an error estimate far under the budget is not an accuracy failure.

## 4. Submitting the array (when you decide to)

Use the times from your pilot's estimate. `--max-parallel` caps the number of concurrent tasks (`%K`):

```bash
bash hpc/submit.sh array hpc/tasks/production_N256_N512.tsv --ids 21-30 --time <HH:MM:SS> --mem 1G --max-parallel 10           # prints only
bash hpc/submit.sh array hpc/tasks/production_N256_N512.tsv --ids 21-30 --time <HH:MM:SS> --mem 1G --max-parallel 10 --submit  # submits
```

The array's job name is `toy_<task list stem>` (here `toy_production_N256_N512`); `status.py` uses it.

**Signals and the wall limit:**
- Every task gets `--signal=B:USR1@300`. The batch script forwards the signal; the run checkpoints and exits 75.
- `MAX_WALL_HOURS` (the limit minus 10 min) stops it cleanly even if no signal arrives.

**`--time` format:** HH:MM:SS. If a task needs more than your site's limit, use a shorter `--time` and resubmit:
each resubmission continues from the last checkpoint.

## 5. Status, resubmission, single-run resume

```bash
python3 hpc/status.py hpc/tasks/production_N256_N512.tsv --run-root $RUN_ROOT
#   per task: category, state, lock, Slurm state, step; then e.g.  safe to (re)submit: --array=23,27-30
bash hpc/submit.sh array hpc/tasks/production_N256_N512.tsv --ids 23,27-30 --time <HH:MM:SS> --mem 1G --submit
python3 toy_run.py --resume --out $RUN_ROOT/costopt_hiacc/double_well_A_N512_s103      # one run, by hand
python3 hpc/check_run.py $RUN_ROOT/costopt_hiacc/*_N512_s10? --expect-config costopt_hiacc
```

**Directory lock.** Every `toy_run.py` process takes an exclusive lock on `<run dir>/.run.lock` before it reads,
resumes or changes anything, and holds it until it exits:
- A second process on the same directory waits up to `--lock-wait` seconds (default 30), then exits 3 without
  touching `status.json`, checkpoints, chunks or the log.
- The lock is a POSIX record lock (`fcntl.lockf`). The kernel drops it when the process ends in any way, SIGKILL
  included. Liveness is never inferred from a PID file; the host and PID written into the lock file are only a note.
- The lock is only as good as the file system's locking: on Lustre `RUN_ROOT` must be mounted with `flock`
  (`localflock` locks per node only); NFS must not be mounted `nolock`. `lock_check.py` reports the mount options.
- `--unsafe-no-lock` turns the lock off (only for file systems without lock support; then no two processes may
  ever share a directory).

**`status.py` categories:**

| category | condition | in "safe to (re)submit"? |
|---|---|---|
| active | queued or running in Slurm (job name `toy_<stem>`), or lock held | no |
| complete | state complete | no |
| failed | state failed | no; inspect, then `toy_run.py --resume --retry-failed` |
| resumable | state incomplete (stopped cleanly) | yes |
| resumable_unclean | state running, Slurm queried and the task not queued, lock free | yes |
| unknown | state running and Slurm not queried (`--no-slurm`, or `squeue` unavailable) | no: check `sacct` / `squeue` |
| not_started | no status.json and not queued | yes |

Without Slurm, "running" is never treated as terminated: a free lock alone does not prove the process is gone on
a file system with node-local locks.

**How resume behaves:**
- **Complete runs** exit at once.
- **Failed runs** (exit 2: close encounter or negative Ritz value) are not resumed automatically. The same state and
  noise would fail again. Inspect `run.log` first; `--retry-failed` forces a retry.
- **Unclean ends** (state still "running" after the process died, e.g. node failure or SIGKILL) resume from the
  last checkpoint; the log notes it and the new segment records the previous state.
- **Overwrite protection:** a new run refuses a non-empty directory. `--resume` refuses arguments that differ from
  the stored `config.json`. Chunks written after the last checkpoint (by a kill between the two writes) are moved
  to `chunks/stale/`.

**Exit codes and Slurm states:**

| exit code | meaning | Slurm state |
|---|---|---|
| 0 | complete | COMPLETED |
| 75 | stopped early, resumable | FAILED (expected; resubmit) |
| 3 | run directory locked by another live process; nothing changed | FAILED (find the other job) |
| 2 | failed | FAILED |
| 1 | configuration error | FAILED |

## 6. Reading results

```python
import sys; sys.path.insert(0, "ewald_longitudinal")
import toy_run
R = toy_run.load_run("$RUN_ROOT/costopt_hiacc/double_well_A_N512_s101")
```

**Fields of `R`:**
- `diag`: every step from 0, with columns `step, t, T_kin, U_per_N, KE_per_N, P_norm, rmin, step_wall_s, ritz_min,
  ritz_max`;
- `frame_step`, `Q` (unwrapped positions), `V` (velocities): every save interval in production (every step by
  default); also in burn-in with `--save-burnin`;
- `monitor`: columns `step, noise_err_est, noise_err_est_kmin, damp_err_est, damp_err_est_kmin, ritz_min_mon,
  ritz_max_mon, S_kmin_max`;
- `config`, `status`.

Production time is t − burn-in time. The state at the end of burn-in is in `checkpoint_burnin_end.npz`.

For law B, integrate the VACF with a higher-order or Richardson-corrected rule, or cross-check with the MSD: the
every-step trapezoid rule over-estimates D by up to ≈ 1.2 % (`../vacf_sampling_results/REPORT.md`).

## 7. Cluster information you need to provide

| setting | where it goes |
|---|---|
| Slurm account, CPU partition, QoS (if your site uses one) | `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_QOS` in `cluster.env` |
| absolute path of the cloned repository | `REPO_DIR` in `cluster.env` |
| absolute path for run output, on a file system with coherent POSIX locks (Lustre with `flock`, GPFS, NFS with locking) and quota ≥ 10 GB | `RUN_ROOT` in `cluster.env` |
| absolute path for Slurm stdout/stderr files | `LOG_ROOT` in `cluster.env` |
| how Python is provided (module / conda / venv) and the exact activation commands | `PY_SETUP` (and `PYTHON`) in `cluster.env` |
| whether a 1-core, 2 GB, 40 min job is allowed on the partition (some sites require whole nodes or a minimum core count) | adjust the `#SBATCH` lines of `pilot_task.sbatch` |
| maximum wall time per job, maximum array size and concurrent-job limits, memory-per-CPU defaults | flags for `estimate_resources.py` and `submit.sh` |
| whether jobs may be preempted or requeued | if yes, add `--requeue`; resuming is safe |
| scratch purge policy | runs are ≈ 144–284 MB per trajectory; copy them out before a purge |
