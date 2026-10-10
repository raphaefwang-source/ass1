# LJ law A: HPC short pilot (dense reference vs fast PPPM-Lanczos, dt 0.005 / 0.01, t = 1)

**Where this ran: LOCAL (no Slurm job id): not an HPC result.** No Slurm job id is attached to any segment. Every number below is from this machine, not from the cluster; the cluster run is still to be done (commands in Section 9).

Scope: one shared initial state, t = 1 per case. This pilot checks that the runs, the comparison and the restart machinery work and measures cost. It does not show statistical equivalence of the dynamics or long-time equilibrium.

## 1. Setup

- Model: `lj_rho0.75_kT1.0_A_prod`, LJ, law A, N 256, ρ* 0.75, kT 1.0, L 6.988644, γ 0.5, κ 0.7, r_ref 1.3; full periodic friction, k = 0 mode kept in both operators.
- Fast: PPPM ξ 0.85, s 4.1, η 0.6779, p 8 → mesh M 24; Lanczos ranks noise/damping [14, 5]; one Γ_h(q_half) for damping and noise.
- Dense: full periodic lattice-sum Γ (`v2.LatticeFriction`: 33 near + 988 Chebyshev far images, tail ≈ 1e-11 per pair, k = 0 coefficient retained), exact matrix functions by eigendecomposition on Range(Π) (`DenseRef`). Not an eigendecomposition of the PPPM Γ_h.
- Both: BAOAB with the finite-time FDT O-step, the same runner (`toy_run.py`, `--method`), the same conservative force (`neighbor` = `toy_dynamics.conservative_force_neighbor`).
- Initial state: `lj075_results/hpc_pilot/init_state_eqA_lgv_s305_t50.npz` = end of law A run `eq_A_lgv_s305` at t = 50 (Langevin-prepared canonical start; REPORT.md §4: canonical starts need no burn-in). T_kin of this snapshot 0.920, U/N -4.7625. Identical in all cases (checked bitwise).
- Noise: fast and dense at the same dt share the seed, so they receive the same standard-normal input at every step (checked: identical final RNG state, identical state at t = 0). dt 0.005 and dt 0.01 use different seeds (7005, 7010) and are **not coupled**; no cross-dt pathwise comparison is made. (A shared seed would not couple different step sizes. The verified OU coupling of `lj075_coupled.py` makes the dt 0.01 input depend on Γ, so fast and dense at 0.01 would no longer get identical inputs.)
- Output: positions (unwrapped), velocities and diagnostics every step; accuracy monitor every 10 steps; checkpoint every 25 steps counted from each segment's start, and at every stop (`--checkpoint-every-steps` is passed to every segment, since `toy_run.py --resume` takes the cadence from its command line).
- Threads: OPENBLAS/OMP/MKL_NUM_THREADS = 1; OS threads per process at segment end: [(1,)].
- Hardware: host vm, Intel(R) Xeon(R) Processor @ 2.80GHz, 4 cores visible, Linux-6.18.44-fc-v114-x86_64-with-glibc2.39; Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, BLAS scipy-openblas.
- Code: git 2e8576741f3e20f165767c90852862b686f0d4b9 (dirty=False).

## 2. Workflow checks

| check | dense_dt0.005 | fast_dt0.005 | dense_dt0.01 | fast_dt0.01 | pair_0.005 | pair_0.01 | cross_dt |
|---|---|---|---|---|---|---|---|
| complete | PASS | PASS | PASS | PASS |  |  |  |
| every_step_frames | PASS | PASS | PASS | PASS |  |  |  |
| shared_initial_state | PASS | PASS | PASS | PASS |  |  |  |
| finite | PASS | PASS | PASS | PASS |  |  |  |
| momentum_within_roundoff | PASS | PASS | PASS | PASS |  |  |  |
| segments_contiguous | PASS | PASS | PASS | PASS |  |  |  |
| checkpoint_cadence_in_all_segments | PASS | PASS | PASS | PASS |  |  |  |
| monitor_points_sampled | PASS | PASS | PASS | PASS |  |  |  |
| monitor_within_budget | PASS | PASS | PASS | PASS |  |  |  |
| no_accuracy_warnings | PASS | PASS | PASS | PASS |  |  |  |
| exit_codes_as_expected | PASS | PASS | PASS | PASS |  |  |  |
| signal_stop_exercised | PASS | PASS | PASS | PASS |  |  |  |
| duplicate_refused_by_lock | PASS | PASS | PASS | PASS |  |  |  |
| restart_bitwise | PASS | PASS | PASS | PASS |  |  |  |
| lock_check | PASS | PASS | PASS | PASS |  |  |  |
| lock_check_without_warnings | PASS | PASS | PASS | PASS |  |  |  |
| single_thread_env | PASS | PASS | PASS | PASS |  |  |  |
| single_os_thread | PASS | PASS | PASS | PASS |  |  |  |
| true_lanczos_error_within_budget |  | PASS |  | PASS |  |  |  |
| same_seed |  |  |  |  | PASS | PASS |  |
| same_final_rng_state |  |  |  |  | PASS | PASS |  |
| identical_at_t0 |  |  |  |  | PASS | PASS |  |
| same_force_method |  |  |  |  | PASS | PASS |  |
| different_seeds_no_coupling_claimed |  |  |  |  |  |  | PASS |

All checks passed: **True**.

`lock_check` runs `hpc/lock_check.py` on the case directory; `duplicate_refused_by_lock` starts a second `--resume` on the same node while a segment runs. Neither shows that locks exclude processes on *other* nodes (Lustre `localflock`, NFS `nolock`); `lock_check.py` warns about such mounts, and it gave no warning here.

## 3. Four cases

| | dense_dt0.005 | fast_dt0.005 | dense_dt0.01 | fast_dt0.01 |
|---|---|---|---|---|
| O-step operator | full periodic Γ (lattice sum, k=0 kept), dense eigendecomposition | PPPM Γ_h (M 24, k=0 kept), Lanczos 14/5 | full periodic Γ (lattice sum, k=0 kept), dense eigendecomposition | PPPM Γ_h (M 24, k=0 kept), Lanczos 14/5 |
| dt / steps / frames | 0.005 / 200 / 201 | 0.005 / 200 / 201 | 0.01 / 100 / 101 | 0.01 / 100 / 101 |
| seed (noise stream) | 7005 | 7005 | 7010 | 7010 |
| host / CPU | vm / Intel(R) Xeon(R) Processor @ 2.80GHz | vm / Intel(R) Xeon(R) Processor @ 2.80GHz | vm / Intel(R) Xeon(R) Processor @ 2.80GHz | vm / Intel(R) Xeon(R) Processor @ 2.80GHz |
| state / all values finite | complete / yes | complete / yes | complete / yes | complete / yes |
| max \|P(t)−P(0)\| (1st half, 2nd half) | 1.3e-13 (1.3e-13, 1.3e-13) | 1.3e-13 (6.3e-14, 1.3e-13) | 8.5e-14 (2.8e-14, 8.5e-14) | 6.7e-14 (2.8e-14, 6.7e-14) |
| max \|P−P0\| / (ε Σ\|p\|); drift | 1.0; none detectable (round-off) | 1.0; none detectable (round-off) | 0.6; none detectable (round-off) | 0.5; none detectable (round-off) |
| T_kin: t=0 / mean / min–max | 0.9203 / 0.9826 / 0.864–1.134 | 0.9203 / 0.9826 / 0.864–1.134 | 0.9203 / 0.9706 / 0.873–1.079 | 0.9203 / 0.9706 / 0.873–1.079 |
| U/N: t=0 / mean / end | -4.7625 / -4.6791 / -4.5984 | -4.7625 / -4.6791 / -4.5984 | -4.7625 / -4.6839 / -4.7128 | -4.7625 / -4.6839 / -4.7128 |
| min pair distance | 0.877 | 0.877 | 0.892 | 0.892 |
| spectrum (Ritz) range | 4.653 – 12.31 | 4.659 – 12.28 | 4.653 – 12.13 | 4.659 – 12.09 |
| monitor points / max error estimate (budget) | 21 / n/a (exact matrix functions) | 21 / 4.9e-11 (2e-07) | 11 / n/a (exact matrix functions) | 11 / 4.8e-11 (2e-07) |
| actual Lanczos error, max (noise, damping) | n/a | 4.3e-12, 4.9e-13 | n/a | 3.3e-12, 1.5e-11 |
| init (process start → first step) [s] | 3.99 (imports 0.91) | 0.95 (imports 0.90) | 4.48 (imports 0.92) | 1.03 (imports 0.99) |
| integration step: median / p90 / max [s] | 1.129 / 2.643 / 3.082 | 0.208 / 0.243 / 0.289 | 1.034 / 2.568 / 2.921 | 0.204 / 0.245 / 0.302 |
| monitor per point [s] | 0.000 | 0.325 | 0.000 | 0.300 |
| output per step (frame every step, checkpoint every 25 steps) [s] | 0.0004 | 0.0004 | 0.0005 | 0.0005 |
| full per step (integration + monitor + output) [s] | 1.584 | 0.249 | 1.504 | 0.248 |
| run CPU [s] / run process wall [s] | 323.7 / 328.7 | 52.2 / 53.9 | 153.6 / 156.3 | 27.5 / 28.6 |
| peak RSS [MB] | 604 | 121 | 604 | 121 |
| segments / checkpoints at / restart bitwise / duplicate refused | 3 / [0, 25, 26, 51, 56, 81, 106, 131, 156, 181, 200] / yes / yes | 3 / [0, 25, 26, 51, 56, 81, 106, 131, 156, 181, 200] / yes / yes | 3 / [0, 25, 26, 51, 56, 81, 100] / yes / yes | 3 / [0, 25, 26, 51, 56, 81, 100] / yes / yes |
| threads (env / OS threads) | 1 / [1] | 1 / [1] | 1 / [1] | 1 / [1] |
| raw output on disk [MB] | 2.6 | 2.6 | 1.3 | 1.3 |
| est. core-s per time unit (prod. monitor cadence) | 317 | 43 | 150 | 21 |


Init = process start to the first step (imports, configuration, initial state, operator set-up, initial force), first segment. Integration step = one BAOAB step without monitor or output. Output = chunk and checkpoint files, status and log writes. Step, output and full per-step costs leave out segment 2, during which the duplicate lock test runs (on a one-CPU allocation it shares the core). Run CPU / wall: the three run segments (the twin restart check is extra).
Integration step times vary widely in dense_dt0.005 (p90 / median 2.3), dense_dt0.01 (p90 / median 2.5); the cost estimates use the mean step. The cause was not investigated (the dense step builds and diagonalises a 768 × 768 matrix with large temporaries).

## 4. Fast vs dense at the same dt

| dt | t | Δq rms | Δq max | Δv rms | Δv max | Δv rms / v rms | ΔT_kin | ΔU/N |
|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.005 | 2.88e-11 | 7.96e-11 | 1.15e-08 | 3.18e-08 | 6.96e-09 | -1.7e-10 | -2.3e-11 |
| 0.005 | 0.05 | 7.99e-10 | 2.26e-09 | 2.63e-08 | 6.10e-08 | 1.57e-08 | -2.1e-09 | -3.0e-10 |
| 0.005 | 0.1 | 1.82e-09 | 4.53e-09 | 3.26e-08 | 8.77e-08 | 1.90e-08 | -2.5e-09 | +5.9e-10 |
| 0.005 | 0.25 | 4.58e-09 | 1.13e-08 | 4.09e-08 | 9.13e-08 | 2.40e-08 | -1.5e-09 | +2.0e-09 |
| 0.005 | 0.5 | 8.25e-09 | 2.84e-08 | 6.32e-08 | 1.71e-07 | 3.62e-08 | +4.2e-09 | -7.9e-10 |
| 0.005 | 1 | 1.96e-08 | 6.28e-08 | 1.28e-07 | 4.85e-07 | 7.44e-08 | -1.8e-10 | +3.7e-10 |
| 0.01 | 0.01 | 7.89e-11 | 1.86e-10 | 1.57e-08 | 3.69e-08 | 9.56e-09 | +6.0e-11 | -1.0e-10 |
| 0.01 | 0.05 | 8.44e-10 | 2.12e-09 | 2.62e-08 | 5.68e-08 | 1.56e-08 | -1.7e-09 | -1.5e-09 |
| 0.01 | 0.1 | 1.80e-09 | 4.69e-09 | 3.16e-08 | 7.56e-08 | 1.85e-08 | -2.4e-09 | -4.9e-10 |
| 0.01 | 0.25 | 4.64e-09 | 1.41e-08 | 4.34e-08 | 1.09e-07 | 2.57e-08 | -1.4e-09 | -2.2e-09 |
| 0.01 | 0.5 | 9.15e-09 | 2.88e-08 | 6.54e-08 | 2.04e-07 | 3.74e-08 | +5.1e-09 | -1.3e-08 |
| 0.01 | 1 | 2.22e-08 | 7.81e-08 | 1.35e-07 | 5.14e-07 | 7.78e-08 | +1.2e-08 | -1.8e-08 |


- dt 0.005: identical at t = 0 (True); max over t ≤ 1: |Δq| 6.3e-08, |Δv| 4.9e-07, |ΔT_kin| 8.3e-09, |ΔU/N| 1.1e-08; at t = 1 the rms velocity difference is 7.4e-08 of the rms velocity; fitted exponential growth rate of the rms difference for t ≥ 0.1: Δq 2.18, Δv 1.45 per time unit.
- dt 0.01: identical at t = 0 (True); max over t ≤ 1: |Δq| 7.8e-08, |Δv| 5.1e-07, |ΔT_kin| 1.3e-08, |ΔU/N| 1.8e-08; at t = 1 the rms velocity difference is 7.8e-08 of the rms velocity; fitted exponential growth rate of the rms difference for t ≥ 0.1: Δq 2.42, Δv 1.51 per time unit.

The fast and dense O-steps differ by two approximations, measured at the fast runs' monitor configurations and inputs (Section 6):
- PPPM Γ_h vs full periodic Γ: 4.1e-08–4.3e-08 (relative Frobenius norm on Range(Π)); in the O-step matrix functions noise 1.9e-08–2.1e-08, damping 1.9e-09–4.1e-09;
- Lanczos truncation (ranks [14, 5]) against exact f(Γ_h): noise ≤ 4.3e-12, damping ≤ 1.5e-11.
The operator term is the larger one by at least a factor 131 in both functions, so the path difference comes mainly from the PPPM operator approximation.

The differences grow over t ≤ 1 (`pilot_fast_vs_dense.png`); the fitted exponential rates above summarise the growth over t ∈ [0.1, 1] only and are not Lyapunov exponents. One initial state and one noise path per dt: these numbers describe this path pair only and are not a statistical comparison of the two methods.

## 5. Momentum, temperature, potential energy

- dense_dt0.005: max |P(t) − P(0)| = 1.3e-13 (relative to √(NmkT) = 16: 8.2e-15); first / second half 1.3e-13 / 1.3e-13; ε Σ|p| = 1.3e-13. **No drift detectable: max |ΔP| = 1.0 ε Σ|p|, within the round-off of forming Σp (random-walk scale over 200 steps: 14 ε Σ|p|).** T_kin mean 0.9826 (range 0.864–1.134), U/N mean -4.6791; all values finite: True.
- fast_dt0.005: max |P(t) − P(0)| = 1.3e-13 (relative to √(NmkT) = 16: 8.3e-15); first / second half 6.3e-14 / 1.3e-13; ε Σ|p| = 1.3e-13. **No drift detectable: max |ΔP| = 1.0 ε Σ|p|, within the round-off of forming Σp (random-walk scale over 200 steps: 14 ε Σ|p|).** T_kin mean 0.9826 (range 0.864–1.134), U/N mean -4.6791; all values finite: True.
- dense_dt0.01: max |P(t) − P(0)| = 8.5e-14 (relative to √(NmkT) = 16: 5.3e-15); first / second half 2.8e-14 / 8.5e-14; ε Σ|p| = 1.3e-13. **No drift detectable: max |ΔP| = 0.6 ε Σ|p|, within the round-off of forming Σp (random-walk scale over 100 steps: 10 ε Σ|p|).** T_kin mean 0.9706 (range 0.873–1.079), U/N mean -4.6839; all values finite: True.
- fast_dt0.01: max |P(t) − P(0)| = 6.7e-14 (relative to √(NmkT) = 16: 4.2e-15); first / second half 2.8e-14 / 6.7e-14; ε Σ|p| = 1.3e-13. **No drift detectable: max |ΔP| = 0.5 ε Σ|p|, within the round-off of forming Σp (random-walk scale over 100 steps: 10 ε Σ|p|).** T_kin mean 0.9706 (range 0.873–1.079), U/N mean -4.6839; all values finite: True.

|P(0)| = 6.6e-13 (round-off of the source run). The O-step carries the mean momentum exactly and projects the friction update, and the pair forces cancel in pairs, so P changes only by floating-point round-off. ε Σ|p| is the round-off of forming Σp once; a deviation within ε Σ|p| √n (n steps) cannot be told apart from round-off, and no drift is detectable then. A systematic error of 2e-14 per step (0.15 ε Σ|p|) would exceed that scale within t = 1 at both step sizes. T_kin over t = 1 from one state is a single correlated sample; its mean is not a temperature test.

## 6. Lanczos error monitor (fast runs)

- fast_dt0.005: 21 monitor points at steps [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]. Monitor estimate |f_r − f_(r+e)| max: noise 4.3e-12, damping 4.9e-13. Actual error against dense f(Γ_h) at the same configurations and inputs: noise 4.3e-12, damping 4.9e-13. Budget 2e-07. Smallest eigenvalue of Γ_h / Γ: 4.655 / 4.655.
- fast_dt0.01: 11 monitor points at steps [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]. Monitor estimate |f_r − f_(r+e)| max: noise 3.3e-12, damping 1.5e-11. Actual error against dense f(Γ_h) at the same configurations and inputs: noise 3.3e-12, damping 1.5e-11. Budget 2e-07. Smallest eigenvalue of Γ_h / Γ: 4.657 / 4.657.

The monitor points include steps after both restarts. The dense runs have no Lanczos step; their monitor rows hold S(k_min) only.

## 7. Checkpoint save and restore

- dense_dt0.005: segment 1 stopped by SIGUSR1 (sent by the pilot) → exit 75 at step 26; segment 2 (30-step limit) exit 75 at step 56, duplicate resume during it exit 3 (3 = refused by the lock); final exit [0]; checkpoints at [0, 25, 26, 51, 56, 81, 106, 131, 156, 181, 200] (expected [0, 25, 26, 51, 56, 81, 106, 131, 156, 181, 200]). Continuous twin to step 66 (exit 0, past both restarts: True): diagnostics True, frames True, monitor rows True, q/p True (bitwise).
- fast_dt0.005: segment 1 stopped by SIGUSR1 (sent by the pilot) → exit 75 at step 26; segment 2 (30-step limit) exit 75 at step 56, duplicate resume during it exit 3 (3 = refused by the lock); final exit [0]; checkpoints at [0, 25, 26, 51, 56, 81, 106, 131, 156, 181, 200] (expected [0, 25, 26, 51, 56, 81, 106, 131, 156, 181, 200]). Continuous twin to step 66 (exit 0, past both restarts: True): diagnostics True, frames True, monitor rows True, q/p True (bitwise).
- dense_dt0.01: segment 1 stopped by SIGUSR1 (sent by the pilot) → exit 75 at step 26; segment 2 (30-step limit) exit 75 at step 56, duplicate resume during it exit 3 (3 = refused by the lock); final exit [0]; checkpoints at [0, 25, 26, 51, 56, 81, 100] (expected [0, 25, 26, 51, 56, 81, 100]). Continuous twin to step 66 (exit 0, past both restarts: True): diagnostics True, frames True, monitor rows True, q/p True (bitwise).
- fast_dt0.01: segment 1 stopped by SIGUSR1 (sent by the pilot) → exit 75 at step 26; segment 2 (30-step limit) exit 75 at step 56, duplicate resume during it exit 3 (3 = refused by the lock); final exit [0]; checkpoints at [0, 25, 26, 51, 56, 81, 100] (expected [0, 25, 26, 51, 56, 81, 100]). Continuous twin to step 66 (exit 0, past both restarts: True): diagnostics True, frames True, monitor rows True, q/p True (bitwise).

## 8. Cost and resource recommendation

Cost per physical time unit, from the measured mean integration step, the measured output cost with a frame every step, and the monitor at the production cadence (every 500 steps):

| case | steps / time unit | step [s] | core-s / time unit | storage MB / time unit (every-step frames) | core-h per 100 time units | core-h, 10 replicas × 55 | peak RSS [MB] | --mem |
|---|---|---|---|---|---|---|---|---|
| dense_dt0.005 | 200 | 1.583 | 317 | 2.5 | 8.80 | 48.4 | 604 | 1.5G |
| fast_dt0.005 | 200 | 0.214 | 43 | 2.5 | 1.19 | 6.6 | 121 | 1G |
| dense_dt0.01 | 100 | 1.503 | 150 | 1.2 | 4.18 | 23.0 | 604 | 1.5G |
| fast_dt0.01 | 100 | 0.213 | 21 | 1.2 | 0.59 | 3.3 | 121 | 1G |

Recommendation for the formal law A runs (fast method only; the dense reference costs about 7× more per time unit at dt 0.005 and is for spot checks):

- one replica = one single-core task, OPENBLAS/OMP/MKL threads 1; 5 + 50 time units at dt 0.005 ≈ 0.66 h of compute on this CPU. Request `--time` ≈ 1.5 × the cluster-measured estimate + 10 min, with `--signal=B:USR1@600` and `--resume` so a job stopped at the limit continues;
- `--mem=1G` for the fast runs (1.5 × peak RSS 121 MB + 256 MB, rounded up); `--mem=1.5G` for the dense reference (peak RSS 604 MB, lattice set-up);
- storage with every-step frames: 2.5 MB per time unit at dt 0.005 (135 MB per replica of 55);
- 10 replicas × 55 time units: ≈ 6.6 core-h at dt 0.005, ≈ 3.3 core-h at dt 0.01 (dt 0.01 is not validated for production; REPORT_observable_dt.md);
- these numbers come from this local machine; the per-step time on the cluster's CPU must replace them before sizing the array (rerun this pilot there).

## 9. Running it on the cluster

```bash
cp hpc/cluster.env.example hpc/cluster.env   # fill in every FILL_ME
hpc/submit.sh lj075-pilot --dry-run          # prints the five sbatch commands
hpc/submit.sh lj075-pilot                    # 4 single-core case jobs + 1 analysis job (afterany)
squeue -u $USER; sacct -j <ids> --format=JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,NodeList
```

The analysis job writes `$RUN_ROOT/lj075_pilot/<tag>/report/` (this report, `pilot_table.md`, `pilot_results.json`, figures); copy that directory back into `lj075_results/hpc_pilot/` for review.

## 10. What this pilot does not show

- No statistical equivalence of fast and dense dynamics and no dt conclusion: one initial state, one noise path per dt, t = 1.
- No long-time equilibrium or stationarity: T_kin and U/N over t = 1 are single correlated samples.
- dt 0.005 and dt 0.01 are not pathwise coupled; nothing here compares them pathwise.
- Locking across nodes: the duplicate test runs on one node (see Section 2).
- Slurm's own time-limit signal: segment 1 is stopped by a SIGUSR1 that the pilot sends itself. The path from `--signal=B:USR1@120` through the batch-shell trap to the run is used only if a job nears its limit; this pilot does not force it (the older N = 512 pilot, `hpc/pilot_task.sbatch`, does).
- Nothing about the cluster: its file system's locking, its Python/BLAS build and its step times are tested only when these jobs run there.

Analysis wall time 85 s (Lanczos/operator check 85 s).
