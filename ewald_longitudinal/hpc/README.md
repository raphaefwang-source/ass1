# Running the toy friction dynamics on an HPC cluster (Slurm)

**What has and has not been run.** Nothing in this directory has been run on an HPC system yet. Locally, with no
Slurm, the following have been checked:
- the runner tests;
- the pilot script under bash;
- the array script under bash, including USR1 forwarding and resume;
- the submission wrapper, with a fake `sbatch`.

Results: `local_pilot_20261008/`.

**Scope of this round:** no long runs were started. The model is unchanged: pure longitudinal friction, full
periodic images, k = 0, g(r_ref) = 0.5, kT = 0.7, dt = 0.005, density 64/5.5³. No random batch.

## Files

| file | purpose |
|---|---|
| `../toy_configs.py` | named configurations; every parameter explicit, N-dependent Lanczos ranks, refusals |
| `../toy_run.py` | unified, restartable runner (entry point) |
| `../verify_production_configs.py` | static confirmation of the configurations at exactly their parameters |
| `../test_toy_run.py` | runner tests (bitwise vs `toy_dynamics.run`, restart, signals, overwrite protection) |
| `cluster.env.example` | the cluster settings you fill in (copy to `cluster.env`, which git ignores) |
| `setup_venv.sh`, `requirements.txt`, `requirements-range.txt` | Python environment: pinned versions (Python ≥ 3.12) or version ranges |
| `submit.sh` | submission wrapper; submits the pilot, only prints array commands unless `--submit` |
| `pilot.sbatch` | pilot job |
| `array.sbatch` | one trajectory per array task, always `--resume` |
| `make_tasklist.py` | potential × kernel × N × seed task list (TSV) |
| `status.py` | per-task state and the `--array` string of what to (re)submit |
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
```

**Which versions have been checked:**
- **Pinned set:** Python 3.13.16 with NumPy 2.5.3, SciPy 1.18.1 and Matplotlib 3.11.2 (`requirements.txt`). Every
  local check of this round used it.
- **Range set:** Python 3.11.17 with NumPy 2.4.6 and SciPy 1.17.1 (`requirements-range.txt`); `test_toy_run.py`
  passes.
- **Python 3.10 and older:** the code parses with the 3.9 grammar, but nothing has been run.

Results from different NumPy/SciPy/BLAS builds are not bitwise comparable with each other. Restart consistency on
the cluster's own installation is checked by the pilot.

Use your site's module or conda instead of a venv if that is the norm there.

## 2. Pilot (the only job to submit for now)

```bash
cd ass1/ewald_longitudinal
bash hpc/submit.sh pilot --dry-run      # shows the sbatch command
bash hpc/submit.sh pilot                # submits; 1 core, 2 GB, 30 min limit
squeue -u $USER
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,Timelimit
less $LOG_ROOT/toy_pilot-<JOBID>.out
ls $RUN_ROOT/pilot/<JOBID>/             # pilot_check.json, resource_estimate.json, restart_check*/, test_toy_run.log
```

The pilot runs:
- `test_toy_run.py`;
- N = 512 double-well, laws A and B, 200 steps each;
- two restart-consistency checks (N = 64): a step-limit split and SIGTERM;
- `check_run.py` and `estimate_resources.py` on the default task list.

**Pilot acceptance.** All of these must hold:
1. Job state COMPLETED, exit code 0, and the last log line is `PILOT PASSED`.
2. `test_toy_run.log` ends with `OK` (6 tests).
3. Both `restart_check*/restart_check.json` have `"passed": true`: bitwise q, p, RNG, diagnostics, frames and
   monitor rows.
4. `pilot_check.json` passes for both runs:
   - config `costopt_hiacc`, ranks 40/5 (A) and 24/8 (B), KD-tree pairs, neighbour forces;
   - static verification PASS for these parameters;
   - 200 contiguous steps, all values finite;
   - total momentum drift ≤ 1e-9·√(N m kT); r_min > 0.45; mean temperature within 20 % of 0.7 (a sanity bound,
     not an equilibration test);
   - all Ritz values > 0; Lanczos error estimates ≤ budget; no accuracy warnings; 11 frames.
5. Step time of the same order as locally (A ~0.5–0.7 s, B ~0.35–0.5 s per step at N = 512); `MaxRSS` well below
   the 2 GB requested (~0.13 GB locally).
6. `env_report.json` shows 1 thread for OPENBLAS/OMP/MKL and the expected Python/NumPy.

## 3. Production plan and resources (prepare; do not submit yet)

```bash
python3 hpc/make_tasklist.py --out hpc/tasks/production_N256_N512.tsv
#   ids 1-10  N=256 A | 11-20 N=256 B | 21-30 N=512 A | 31-40 N=512 B
#   (double_well and lj x seeds 101-105; burn-in t=140, production t=60, frames every t=0.1)
python3 hpc/estimate_resources.py --pilot $RUN_ROOT/pilot/<JOBID>/double_well_?_N512_s101 \
    --tasklist hpc/tasks/production_N256_N512.tsv --margin 1.5 [--max-job-hours <SITE LIMIT>] [--max-parallel 20]
```

**What the estimate gives, per (N, kernel):**
- run time per task (40 000 steps × median pilot step time);
- `--time` (× margin, rounded up, + 10 min);
- `--mem` (2 × pilot peak RSS, at least 1 GB);
- core-hours;
- disk.

N = 256 is scaled from the N = 512 pilot by the earlier local benchmark ratio and is labelled as such.

**Local reference (slow VM, 1.5 margin):**
- per task: A N=512 ≈ 7.4 h run (`--time 11:25:00`); B N=512 ≈ 5.1 h (`07:55:00`); A N=256 ≈ 3.5 h; B N=256 ≈ 2.4 h;
- all 40 tasks: ≈ 184 core-hours of run time, ≈ 276 core-hours with margin, ≈ 0.5 GB of disk.

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
the spectrum falls below the verified range.

## 4. Submitting the array (when you decide to)

Use the times from your pilot's estimate. `--max-parallel` caps the number of concurrent tasks (`%K`):

```bash
bash hpc/submit.sh array hpc/tasks/production_N256_N512.tsv --ids 21-30 --time <HH:MM:SS> --mem 1G --max-parallel 10           # prints only
bash hpc/submit.sh array hpc/tasks/production_N256_N512.tsv --ids 21-30 --time <HH:MM:SS> --mem 1G --max-parallel 10 --submit  # submits
```

**Signals and the wall limit:**
- Every task gets `--signal=B:USR1@300`. The batch script forwards the signal; the run checkpoints and exits 75.
- `MAX_WALL_HOURS` (the limit minus 10 min) stops it cleanly even if no signal arrives.

**`--time` format:** HH:MM:SS. If a task needs more than your site's limit, use a shorter `--time` and resubmit:
each resubmission continues from the last checkpoint.

## 5. Status, resubmission, single-run resume

```bash
python3 hpc/status.py hpc/tasks/production_N256_N512.tsv --run-root $RUN_ROOT
#   prints per-task state and e.g.  resubmit (not started / incomplete / killed): --array=23,27-30
bash hpc/submit.sh array hpc/tasks/production_N256_N512.tsv --ids 23,27-30 --time <HH:MM:SS> --mem 1G --submit
python3 toy_run.py --resume --out $RUN_ROOT/costopt_hiacc/double_well_A_N512_s103      # one run, by hand
python3 hpc/check_run.py $RUN_ROOT/costopt_hiacc/*_N512_s10? --expect-config costopt_hiacc
```

**How resume behaves:**
- **Complete runs** exit at once.
- **Failed runs** (exit 2: close encounter or negative Ritz value) are not resumed automatically. The same state and
  noise would fail again. Inspect `run.log` first; `--retry-failed` forces a retry.
- **Overwrite protection:** a new run refuses a non-empty directory. `--resume` refuses arguments that differ from
  the stored `config.json`. Chunks written after the last checkpoint (by a kill between the two writes) are moved
  to `chunks/stale/`.

**Exit codes and Slurm states:**

| exit code | meaning | Slurm state |
|---|---|---|
| 0 | complete | COMPLETED |
| 75 | stopped early, resumable | FAILED (expected; resubmit) |
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
- `frame_step`, `Q` (unwrapped positions), `V` (velocities): every save interval in production; also in burn-in
  with `--save-burnin`;
- `monitor`: columns `step, noise_err_est, noise_err_est_kmin, damp_err_est, damp_err_est_kmin, ritz_min_mon,
  ritz_max_mon, S_kmin_max`;
- `config`, `status`.

Production time is t − burn-in time. The state at the end of burn-in is in `checkpoint_burnin_end.npz`.

## 7. Cluster information you need to provide

| setting | where it goes |
|---|---|
| Slurm account, CPU partition, QoS (if your site uses one) | `cluster.env` |
| absolute paths: repository, run output (scratch or project space with quota), Slurm logs | `cluster.env` |
| how Python is provided (module / conda / venv) and the exact activation commands | `PY_SETUP` in `cluster.env` |
| maximum wall time per job, maximum array size and concurrent-job limits, memory-per-CPU defaults | flags for `estimate_resources.py` and `submit.sh` |
| whether jobs may be preempted or requeued | if yes, add `--requeue`; resuming is safe |
| scratch purge policy | runs are small (≈ 10–17 MB per trajectory) but should be copied out before a purge |
