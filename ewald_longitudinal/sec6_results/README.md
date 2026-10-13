# Section 6 experiments: shared configuration, status and results

This directory collects the Section 6 experiments that follow Experiment 1. The stages are:

- the Lanczos accuracy and rank selection (former experiments 4, 5 and 8);
- the ideal-gas conservation and equilibrium check (former 6);
- the dynamics reference comparison and time-step check (former 7 and 9);
- the full-step equal-accuracy timing (former 10).

Every stage reads its parameters from `../sec6_config.json` through `../sec6_config.py`. No script hard-codes model
constants, spatial parameters, dt or ranks. Each output carries a `provenance` record with:

- the git commit and dirty flag;
- the **config hash**, covering model, spatial sets, production settings and budgets;
- the **code hash**, a sha256 over the core numerical source files and the producing script;
- versions, host and thread settings.

`sec6_config.reusable(record, script)` accepts a stored result only when both hashes match.

## Model (unchanged)

- Kernel K_κ(r) = e^{−κr}/r · r rᵀ/r².
- κ = β = m = 1.
- Full periodic lattice sum, with k = 0 retained.
- Fixed density ρ = 30/216, so L = (N/ρ)^{1/3}.

The Lennard-Jones application (`lj075_*`, `toy_*`) has a different kernel amplitude, κ, density and dt = 0.005.
None of these are used here.

## Stage 1: Experiment 1 close-out and shared configuration (done)

### Integrity

`python3 exp1_finalize.py` writes `exp1_results/integrity.json`. It finds:

- every CSV has its designed row count;
- there are no NaN or Inf cells;
- the rerun log `exp1_results/run_2026-10-13_exact_k.log` ends with exit 0, after the stages split, cutoffs, mesh,
  action and precision.

`khat_AB` (`test_parameter_selection.py`) deduplicates on the unrounded float64 |k|, with no `np.round`.

### Precision after the fix

Mesh-free Fourier sum against the real-space image sum Γ_L, relative to ‖Γ_ref‖₂ (`exp1_results/fourier_precision.csv`):

| variant | floor |
|---|---|
| \|k\| rounded to 12 decimals (removed routine) | 2.3e-14 – 1.5e-13 |
| unrounded \|k\| (current) | 6.3e-17 – 7.6e-16 |
| float64 vs 80-bit evaluation of the same sum | 4.2e-17 – 6.7e-16 |
| image sum, R = 40 vs 45 | 4.1e-17 – 4.3e-16 |

The former ~1e-13 plateau came from the |k| rounding. It is not a float64 summation floor.

### Measured scope of the spatial parameter sets

The relative operator error is ‖Γ_h − Γ_ref‖₂/‖Γ_ref‖₂, from `exp1_results/parameter_set_status.json` and `.csv`. All
three sets were accepted on cases A and B only.

| set | ξ, s, η_N, p | nominal tol | pass / tested | largest rel. error | abs. error range | failing configurations |
|---|---|---|---|---|---|---|
| T5 | 0.7, 3.4937, 0.8, 6 | 1e-5 | 10 / 14 | 1.29e-5 | 7.7e-6 – 3.3e-5 | perturbed lattices N=128 s1–s3, N=256 |
| T7 | 0.7, 4.1002, 0.7, 7 | 1e-7 | 39 / 44 | 2.11e-7 | 5.7e-8 – 5.5e-7 | perturbed lattices N=128 s1–s3, N=256; uniform `fixn_L6_N30_s3` |
| T9 | 0.7, 4.6278, 0.6, 8 | 1e-9 | 10 / 14 | 2.55e-9 | 1.3e-9 – 7.1e-9 | perturbed lattices N=128 s1–s3, N=256 |

- No set reaches its nominal tolerance on every tested configuration.
- T7 fails on a uniform configuration with the same N and L as case A. The failures are therefore not specific to
  lattices.
- The probe actions (relative action errors) are larger still. At T7, all 39 configurations that pass in operator
  norm have a transverse k_min probe above 1e-7, with a maximum of 4.5e-7.

**Production spatial set: T7.** It is used with its measured scope: relative operator error ≤ 2.2e-7 on the 44
static configurations tested, not a general 1e-7. T9 is the finer set for spatial-sensitivity comparisons.

### Shared configuration (`sec6_config.json`)

- **Model:** the constants above.
- **Spatial sets:** T5, T7 and T9.
- **Production settings:**
  - spatial set T7, fine set T9;
  - fixed dt = 0.042917661805891890;
  - Lanczos conventions: direct start, full reorthogonalisation, Π after every action, centre-of-mass momentum copied,
    f_dt via `expm1`, negative Ritz values recorded and never clipped.
- **Budgets:**
  - noise relative action error 1e-3 and damping 1e-6;
  - noise high-accuracy levels 1e-6 and 1e-8;
  - reference margin 0.1 × tol.
- **N scan:** 128 … 16000.

The production noise and damping ranks are `null` until the rank-selection stage writes them.

The dt is the one the pure-kernel dynamics actually used: `dynamics_results/params_small.json`,
dt = 0.2 m / median λ_max(q₀) over the 16 initial N = 64 configurations. The same physical dt applies at every N.

### Reuse of earlier results

The earlier Lanczos and dynamics results do not match the current code and parameters for two reasons:

- they were computed with the rounded-|k| `khat_AB`;
- they used `test_lanczos_fdt.PARAMS` with s = 4.100 instead of the accepted s = 4.100151562. Γ_h then differs by
  about 1e-8 in absolute action on a test vector.

The affected directories are:

- `lanczos_fdt_results/`
- `lanczos_range_start_results/`
- `fixed_rank_budget_results/`
- `dynamics_results/`
- `fft_vs_direct_results/`

They are therefore **not reused** as numbers. They stay in the repository unchanged for traceability, and the new
stages recompute everything they report. Experiment 1 results (`exp1_results/`) match the current code and are the
inputs of this configuration.

## Stage 2: Lanczos accuracy and rank selection (former 4, 5, 8) (done)

### Commands

```bash
python3 sec6_lanczos.py small --workers 4     # 54 s
python3 sec6_lanczos.py large --workers 4     # 24 min on 4 cores
python3 sec6_rank_select.py --write-config     # selection rule, writes the production ranks into sec6_config.json
python3 sec6_lanczos.py report                 # tables and figure
```

- Log: `lanczos/run_lanczos.log`.
- Raw per-case error curves, every rank, with Ritz data and provenance: `lanczos/raw/{small,large}_cases.json`.
  These are committed.

### Design

**Functions.** Each recursion is evaluated for:

- the noise action η = f_dt(Γ_h)Πz, with f_dt = √((m/β)(−expm1(−2dtλ/m)));
- the damping action d = exp(−dtΓ_h/m)Πv.

Both use the same Γ_h (production set T7), the same start vector and the same dt.

**Lanczos.** Direct start, full two-pass reorthogonalisation, Π after every literal action. These are the unchanged
`test_lanczos_fdt.lanczos`. One recursion per (Γ_h, vector) gives every rank and every dt, so ranks are never
re-run. Both functions use the same Gaussian start vector; in equilibrium Πp has the same law as Πz.

**Time step.**

- Primary: the production dt = 0.0429177 from `sec6_config`.
- Diagnostic only: fixed τ = dt·λ_max/m ∈ {0.2, 0.5, 2, 5}. Here λ_max is exact for the small cases. For the large
  cases it is the top Ritz value of the reference recursion and is therefore an **estimate**.

**Small cases** (dense reference on the same Γ_h).

- The 14 Experiment 1 configurations: cases A and B, uniform, perturbed lattice and clustered at N = 128 (3 seeds
  each), and one per family at N = 256. The densities are as in Experiment 1.
- 4 Gaussian vectors each, 56 cases.
- Reference: eigendecomposition on Range(Π).
- The literal action agrees with the assembled Γ_h to ≤ 6.5e-16.

**Large cases** (fixed density 30/216).

- N = 512, 1024, 2048 and 8000: 1 position each. N = 4000 and 16000: 3 independent positions each.
- 3 Gaussian vectors per position, 30 cases.
- Position seeds `[20261014, N, pos]` and vector seeds `[20261015, 2, N, pos, k]` are separate.
- Reference: the same recursion continued to S = 260 (N = 512) … 380 (N = 16000).
- The reference is accepted for a tolerance tol only if ‖η_S − η_{S−20}‖ and ‖η_S − η_{S−40}‖ are both ≤ 0.1·tol.
  Required ranks are then read only up to S − 40.
- All 30 references are accepted for every budget, so nothing is unresolved:

  | function | worst S−20 difference | worst S−40 difference |
  |---|---|---|
  | noise | 9.1e-13 | 3.1e-12 |
  | damping | 3.4e-15 | 3.7e-15 |

**Errors in coefficient space.** Errors against the reference are computed in the Lanczos coefficient space, using
‖Q^TQ − I‖ ≤ 2.1e-15. They agree with vector-space errors to 1.2e-5 relative wherever the error is above 1e-12.

**Negative Ritz values.** None, in 86 recursions and over every rank.

**Spectral range.** dt·λ_max is 0.03–0.55 in the small cases (exact). In the large cases it is 0.38–3.83, estimated
from Ritz values; 3.83 comes from one N = 4000 position with a close pair, λ_max ≈ 89.

### Results

The table gives the largest required rank over all tested cases at the production dt. The required rank is the
smallest r after which the error stays ≤ tol for every computed rank. These are sample maxima, not bounds
(`lanczos/rank_selection.json`, `required_ranks.csv`, `error_at_ranks.csv`).

| budget | N=30 (A) | 40 (B) | 128 | 256 | 512 | 1024 | 2048 | 4000 | 8000 | 16000 |
|---|---|---|---|---|---|---|---|---|---|---|
| noise 1e-3 | 9 | 15 | 22 | 23 | 18 | 21 | 24 | 32 | 34 | **42** |
| damping 1e-6 | 4 | 4 | 5 | 5 | 5 | 5 | 5 | 6 | 6 | 6 |
| noise 1e-6 | 19 | 36 | 75 | 85 | 46 | 59 | 71 | 94 | 114 | 140 |
| noise 1e-8 | 25 | 48 | 111 | 128 | 65 | 82 | 103 | 134 | 169 | 216 |

Notes on the table:

- At N = 128 and 256 the maximum comes from the clustered configurations. The uniform configurations there need 13
  and 16 for noise 1e-3.
- The noise rank grows with N.
- **Rank 40 is not sufficient at N = 16000.** At position 0 the sample maximum is 1.09e-3 > 1e-3.
- The earlier "noise 40 / damping 16" choice was made at τ = 0.5, on one position per N. It is superseded.

**Production ranks: noise 48, damping 10** (`sec6_config.json`).

The rule is `budgets.rank_rule`: the smallest r such that the sample maximum stays ≤ budget/1.5 for all ranks ≥ r. It
must hold both at the production dt and at the diagnostic τ = 5, which lies beyond the largest static dt·λ_max seen
(3.83). Dynamics with close pairs reached dt·λ_max ≈ 10 in earlier large-N runs.

| function | minimal at production dt | with margin, production dt | with margin, τ = 5 | chosen |
|---|---|---|---|---|
| noise | 42 | 47 | 48 | 48 |
| damping | 6 | 6 | 10 | 10 |

Sample maximum errors at the chosen ranks:

- **Noise at rank 48, production dt:** 6.2e-4 (N = 16000), 2.8e-4 (8000), 1.3e-4 (4000), ≤ 6.0e-5 (N ≤ 2048).
- **Damping at rank 10:** ≤ 3.4e-13 at the production dt, and ≤ 5.7e-7 at τ = 5.

**Validated scope:**

- static uniform configurations at ρ = 30/216 with N ≤ 16000;
- the Experiment 1 families at N ≤ 256;
- dt·λ_max up to 3.8 at the production dt, and up to the τ = 5 diagnostic.

Larger N, or states with a larger spectral range, are not covered. Actual dynamics states are checked offline in
stage 3.

**High-accuracy cost.** At N = 16000 the noise needs rank 140 for 1e-6 and 216 for 1e-8. These are minimal ranks with
no margin.

## Stage 3: ideal-gas conservation and equilibrium (former 6) (done)

### Commands

```bash
python3 sec6_dynamics.py ideal --workers 4                                  # N=64 x16 (t=40), N=256 x4 (t=30)
python3 sec6_dynamics.py ideal --workers 4 --spec 256:80:0:8 64:40:16:16    # extension, see below
python3 sec6_dynamics.py offline --workers 4                                # dense check of saved O-steps
python3 sec6_dynamics.py analyze
```

### Setup

- U = 0, periodic, κ = β = m = 1, ρ = 30/216, uniform initial positions.
- Initial component temperatures (2, 0.5, 0.5), with the sample mean removed; centre-of-mass velocity
  (0.3, −0.2, 0.1).
- Production settings from `sec6_config`: spatial set T7, dt = 0.0429177, noise rank 48, damping rank 10. That is 58
  Γ_h actions per step at every step.
- Step A(dt/2) O(dt) A(dt/2), with Γ_h evaluated at q_half. Mean momentum copied.
- Initial states use seed `[20261016, N, traj]`; noise streams use `[20261017, N, traj, 1]`.
- Each step's O-step inputs (q_half, p, ξ) are saved in `ideal_gas/raw/` (git-ignored, 0.5 GB).
- `ideal_gas/observables_N{64,256}.csv.gz` hold the per-step T_x, T_y, T_z, momentum error and Ritz λ estimates of
  every trajectory.

### Statistics

- **Temperatures:** T_a = m Σ_i (v_ia − v_cm,a)² / (N − 1).
- **Confidence intervals:** t-intervals across independent trajectories. For time averages, each trajectory's
  post-burn-in average is one sample.
- **Burn-in:** t_b = max(5 τ_fit, 3/(2λ*)).
  - τ_fit comes from the initial contiguous decay of the ensemble-mean temperatures and anisotropy.
  - λ* is the median Ritz estimate of the smallest eigenvalue on T⊥, so 3/(2λ*) is three e-folds of the slowest
    momentum mode.
  - The window after t_b is checked by comparing its two halves, paired over trajectories.
- **Maxwell marginals:** u = (p_ia − P_a/N)/√(m kT (N−1)/N) at every post-burn-in step. Variance and excess kurtosis
  are estimated per trajectory, then a t-CI is taken over trajectories. The pooled KS statistic is descriptive only,
  because snapshots within a trajectory are correlated.

### Results (`ideal_gas/summary.json`, `fig_ideal_gas.png`)

| | N = 64 | N = 256 |
|---|---|---|
| trajectories × length | 32 × t = 40 (932 steps) | 8 × t = 80 (1864 steps) |
| max ‖P(t) − P(0)‖/‖P(0)‖ | 1.4e-14 | 3.1e-14 |
| τ_fit (anisotropy, components) | 1.2 | 1.3 |
| λ* (Ritz estimate, median) | 0.089 | 0.048 |
| burn-in t_b | 16.8 | 31.4 |
| T_x, T_y, T_z after burn-in (95% CI) | 0.994 ± 0.023, 0.997 ± 0.016, 0.991 ± 0.016 | 1.001 ± 0.011, 0.993 ± 0.016, 0.992 ± 0.018 |
| T_kin | 0.994 ± 0.010 | 0.996 ± 0.011 |
| second half − first half | consistent with 0 in all three components | consistent with 0 in all three components |
| Maxwell variance of u | 0.994 ± 0.010 | 0.996 ± 0.011 |
| excess kurtosis of u | 0.009 ± 0.026 | 0.014 ± 0.018 |
| dt·λ_max visited (Ritz estimate) | 0.06 – 8.8 | 0.12 – 7.5 |

### Statistical gaps found and closed

- With the first 16 N = 64 trajectories, the excess kurtosis was 0.034 ± 0.031, marginally excluding 0. With 32
  trajectories it is 0.009 ± 0.026.
- The first N = 256 runs (4 × t = 30) were shorter than the burn-in rule (3/(2λ*) = 31.3). They therefore showed
  relaxation only, and equilibrium was not claimed. They were rerun as 8 × t = 80 with the same seeds, so the first
  t = 30 are identical. The superseded files are kept in `raw/superseded_T30/`.
- The first 16 N = 64 trajectories ran with `sec6_dynamics.py` as committed in ab94081. The later runs used a version
  that differs only in argument parsing (`--spec`). The two code hashes are recorded per trajectory in `summary.json`.

### Offline O-step checks (`ideal_gas/offline_ostep_checks.json`)

- **Steps checked:** 718 actual O-steps, i.e. per trajectory 12 regularly spaced steps plus the 6 steps with the
  largest Ritz λ_max.
- **Method:** on the saved (q_half, p, ξ), dense f(Γ_h) is compared with the production ranks 48/10. The damping
  vector is the actual momentum Πp, not a random vector.
- **Noise relative error:** ≤ 7.4e-11 at N = 64 and ≤ 1.1e-7 at N = 256.
- **Damping relative error:** ≤ 4.1e-15.
- **Spectral range:** the checked states reach dt·λ_max = 8.8 (exact, dense), beyond the τ = 5 diagnostic used to
  choose the damping rank.

## Stage 4: dynamics reference comparison and time-step check (former 7, 9) (done)

### Commands

```bash
bash sec6_results/dynamics_stage4_small.sh    # offline checks, compare N=64 and N=256, spatial T9, dt study
bash sec6_results/dynamics_stage4_large.sh    # short coupled runs at N = 4000 (2) and N = 16000 (1)
python3 sec6_dynamics.py analyze
```

Logs: `dynamics/run_stage4_small.log`, `dynamics/run_stage4_large.log`. The raw paths are git-ignored. Summary:
`dynamics/summary.json`.

### Design

Three arms share the stage 3 setup: the same initial states, dt, Gaussian stream (prefix of the production stream)
and spatial set.

- **dense:** f(Γ_h) by eigendecomposition of the assembled Γ_h.
- **hi:** Lanczos at noise/damping ranks 120/40 (N = 64), 160/40 (N = 256, 4000, 16000). Every step also evaluates
  ranks −20 and −40 of the same recursion as a self-check, and the production ranks 48/10 against the hi result.
- **prod:** ranks 48/10. For N = 64 and 256 these are the stage 3 trajectories themselves, compared over their first
  t = 6. They were reused, not rerun: the config hash is the same (`808cc0703acf6e50`), and the stage 3 → 4 code
  change only adds stages and record fields. The paired prod − dense differences of order 1e-11 at N = 64 confirm that
  initial states and noise are identical. For N = 4000 and 16000 the prod arm is a new run (`large` stage).

Further comparisons:

- **Spatial sensitivity**, separately: dense with the fine set T9 against dense T7 on the same states and streams
  (N = 64). This is not a Krylov comparison.
- **dt study:** dense at dt and dt/2 to the fixed end time t = 6.008, with nested noise
  ξ_h = S_h⁻¹(R_{h/2}S_{h/2}ξ₁ + S_{h/2}ξ₂) built from the dt/2 stream. That stream is exactly N(0, Π) for a frozen
  Γ_h, and only approximately pathwise-coupled once positions differ. N = 64, 16 trajectories.
- **Statistics:** paired differences per trajectory; mean over trajectories with a t-CI; RMS over paths and times;
  pathwise position and momentum RMS (minimum image) per particle.

### Same Γ_h: Krylov effect of the production ranks

All differences are paired arm A − arm B. Each entry is the largest over time of the |mean over trajectories| of
T_x, T_y, T_z, T_kin and O_k = ⟨cos 2πx/L⟩. Pathwise RMS position and momentum differences are per particle.

| N, runs | comparison | temperatures | O_k | path q / p RMS | resolved from 0? |
|---|---|---|---|---|---|
| 64, 16 × t = 6 | prod − dense | ≤ 2.4e-11 | 1.5e-12 | ≤ 1.1e-9 / 1.1e-9 | no (CI ≈ 2–5e-11) |
| 64, 16 × t = 6 | hi − dense | ≤ 2.0e-13 | 3.4e-14 | ≤ 1.0e-11 / 7.6e-12 | no |
| 256, 4 × t = 6 | prod − dense | ≤ 5.1e-7 | 4.4e-9 | — | no (CI ≈ 1e-6) |
| 256, 4 × t = 6 | hi − dense | ≤ 6.1e-13 | 3.9e-15 | ≤ 4e-12 / 2.4e-11 | no |
| 4000, 2 × 25 steps (t = 1.07) | prod − hi | ≤ 7.8e-6 | ≤ 2.0e-6 | ≤ 1.3e-4 / 2.2e-4 | single paths |
| 16000, 1 × 25 steps | prod − hi | ≤ 1.4e-5 | 7.6e-7 | 5.3e-4 / 9.3e-4 | single path |

Reference checks of the hi arm, per step, as the largest self-difference against ranks −20 and −40 of the same
recursion:

| N | noise, rank −20 | noise, rank −40 | damping, rank −20 |
|---|---|---|---|
| 64 | 3.7e-14 | 1.3e-14 | 2.8e-15 |
| 256 | 7.0e-15 | 8.3e-15 | 2.3e-15 |
| 4000 | 8.7e-9 | 7.9e-8 | 1.7e-15 |
| 16000 | 7.9e-7 | 3.6e-6 | 1.5e-15 |

At N = 16000 the hi arm is a valid reference for the production comparison: its self-difference is two to three orders
below the production-rank error. It is **not** an exact reference. It is not called converged beyond about 1e-6.

**Per-step error of the production ranks on the visited states.** In the hi arm, every step also evaluates ranks
48/10 inside the same recursion, relative to the hi result (same Γ_h, vector and dt). These are sample maxima over
the visited O-steps, not bounds (`dynamics/summary.json`, keys `prodrank_err_in_hi`):

| N | paths | noise, max | noise, median | damping, max |
|---|---|---|---|---|
| 64 | 16 × t = 6 | 3.4e-11 | — | 2.9e-15 |
| 256 | 4 × t = 6 | 2.2e-7 | — | 2.8e-15 |
| 4000 | traj 0, 25 steps | 1.4e-4 | 9.0e-5 | 5.5e-14 |
| 4000 | traj 1, 25 steps | 1.7e-4 | 1.0e-4 | 9.5e-13 |
| 16000 | traj 0, 25 steps | 5.3e-4 | 4.0e-4 | 1.0e-12 |

- Every visited step is inside the noise budget 1e-3 and inside the margin target 1e-3/1.5 used for the rank
  choice.
- At N = 4000 the dynamic maximum (1.7e-4) is slightly above the static stage 2 sample maximum at the same N (1.3e-4).
  Visited states are not identical in law to the static uniform samples. It is still six times below the budget.
- At N = 16000 the hi self-difference (≤ 3.6e-6) is below 1% of these errors, so the listed values are not limited by
  the reference.

The momentum is conserved to ≤ 2.6e-14 in every arm. dt·λ_max (Ritz estimate) reached 1.9 at N = 4000 and 4.3 at
N = 16000.

### Spatial sensitivity (separate from the Krylov comparison)

The spatial check compares dense T9 against dense T7 on the same states and streams. It is the spatial effect at one
Γ_h refinement; it does not validate the full error of the original model.

- **N = 64:**
  - temperatures differ by ≤ 8.8e-7 and O_k by 1.4e-7;
  - the pathwise RMS is q ≤ 2.0e-5 and p ≤ 1.9e-5;
  - only T_kin is marginally resolved: 4.0e-7 against a CI of 3.9e-7, at a single time out of 141.
- **All timed N (static refinement check, `timing/spatial_check.json`):**
  ‖(Γ_T7 − Γ_T9)X‖/‖Γ_T9 X‖ = 5.6e-8 – 1.0e-7 for N = 256 … 16000.

At N = 64 the spatial effect on the dynamics (~1e-6) is four to five orders larger than the Krylov effect of the
production ranks (~1e-11).

### Time step

Setting: N = 64, 16 trajectories, dense dt against dt/2 with nested noise, fixed end time t = 6.008. The table gives
the paired mean difference, dt − dt/2, ± its 95% CI.

| observable | t = 1.5 | t = 3.0 | t = 6.0 |
|---|---|---|---|
| T_x | 0.0034 ± 0.0058 | −0.0097 ± 0.0168 | −0.032 ± 0.040 |
| T_y | −0.0008 ± 0.0043 | −0.0089 ± 0.0217 | −0.012 ± 0.053 |
| T_z | 0.0001 ± 0.0070 | −0.0030 ± 0.0125 | 0.015 ± 0.053 |
| T_kin | 0.0009 ± 0.0029 | −0.0072 ± 0.0111 | −0.0096 ± 0.0202 |
| O_k | −0.0005 ± 0.0010 | −0.0029 ± 0.0041 | −0.0010 ± 0.0177 |

- **Not resolved at any of these times.** Over all 141 recorded times, the pointwise CI excludes 0 at 10 (T_x), 1, 2,
  5 and 1 times. At 5% per time about 7 are expected, so this is no evidence of a resolved bias.
- **Paths decorrelate.** The pathwise RMS of the temperature differences is 0.04–0.05. The coupling across step sizes
  is only approximate once positions differ, and this decorrelation sets the uncertainty.
- **Conclusion: the time-step bias is not clearly resolved at dt.** No convergence order is claimed. dt/4 was not run,
  because a finer level cannot change this judgement while the dt − dt/2 difference itself is unresolved.
- Indirect evidence: at the production dt the stage 3 equilibrium temperatures and Maxwell moments agree with their
  exact values within about 1% (95% CIs). A stationary dt bias larger than that is not seen.

## Stage 5: equal-accuracy full-step timing (former 10) (done)

### Commands

```bash
bash sec6_results/timing_stage5.sh    # direct_accuracy, spatial_check, time (pppm + direct), dense, report
```

- Logs: `timing/run_direct_accuracy.log`, `timing/run_spatial_check.log`, `timing/run_timing.log`.
- Raw per-step timings with every component: `timing/timing_raw.json`.
- Table: `timing/timing_summary.csv`. Fits and accuracy: `timing/summary.json`. Figure: `timing/fig_timing.png`.

### Protocol

- **Machine:** one cloud VM, Intel Xeon @ 2.80 GHz, Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1. One process with
  OMP/OPENBLAS/MKL threads = 1.
- **Isolation:** the timing ran alone, after the stage 4 runs had finished. Every number below comes from this one
  machine and run; nothing is estimated or taken from another machine.
- **What is timed:** one complete step A(dt/2) O(dt) A(dt/2) at the production settings (T7, dt = 0.0429177, noise
  rank 48, damping rank 10), starting from initial state index 900 with its own noise stream.
- **Repeats:** 2 warm-up steps, then 5 consecutive timed steps. The tables give the median and the min–max.
- **Γ_h calls:** 58 actions per step (48 noise + 10 damping, direct start), at every N and every repeat.
- **Inside the step:** the per-step setup (pair list, pair weights, B-spline stencils, degree term) is included,
  because the positions change every step.
- **Outside the step:** the one-time preprocessing (mesh choice, influence function) is listed separately.
- **Timed Lanczos:** `lanczos_timed` does the same arithmetic as `test_lanczos_fdt.lanczos`. Only the timers are
  added.

### Accuracy at each timed N

- **Spatial parameters:** the T7 rule at every N. The mesh M and the actual η follow from it (table). The scope of T7
  is the one measured in stage 1: relative operator error ≤ 2.2e-7 on the 44 static configurations.
- **Refinement check at the timing state** (`timing/spatial_check.json`): ‖(Γ_T7 − Γ_T9)X‖/‖Γ_T9 X‖ is 5.6e-8 –
  1.03e-7 for N = 256 … 16000. This is a refinement indicator, not an exact error.
- **Krylov:** at rank 48 the noise sample maxima are ≤ 6.2e-4 static (stage 2) and ≤ 5.3e-4 on dynamic states
  (stage 4). Damping at rank 10 is ≤ 1e-12.

### PPPM + Lanczos, production settings

| N | L | M | η actual | real-space pairs | step: median (min – max) [s] | T/(N ln N) [µs] | preprocessing [s] | mesh | real space | setup | reorth. | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 12.26 | 36 | 0.622 | 14949 | 0.79 (0.73 – 0.85) | 555 | 0.04 | 82.3% | 13.5% | 2.4% | 0.46% | 1.6% |
| 512 | 15.45 | 45 | 0.627 | 29787 | 1.23 (1.08 – 1.43) | 387 | 0.02 | 82.0% | 14.2% | 2.3% | 0.43% | 1.2% |
| 1024 | 19.46 | 54 | 0.659 | 59772 | 2.57 (2.36 – 2.62) | 362 | 0.06 | 81.5% | 14.2% | 2.4% | 0.40% | 0.8% |
| 2048 | 24.52 | 72 | 0.622 | 119320 | 6.45 (6.05 – 6.93) | 413 | 0.10 | 85.1% | 11.5% | 2.4% | 0.34% | 0.4% |
| 4000 | 30.65 | 81 | 0.691 | 233624 | 9.74 (9.69 – 10.90) | 294 | 0.15 | 82.2% | 14.4% | 2.6% | 0.41% | 0.4% |
| 6000 | 35.09 | 96 | 0.668 | 350824 | 19.77 (19.24 – 20.25) | 379 | 0.24 | 84.8% | 12.3% | 2.3% | 0.31% | 0.3% |
| 8000 | 38.62 | 108 | 0.653 | 468413 | 31.94 (29.89 – 32.18) | 444 | 0.27 | 85.8% | 10.9% | 2.3% | 0.28% | 0.2% |
| 11000 | 42.94 | 120 | 0.654 | 641255 | 44.81 (41.74 – 45.12) | 438 | 0.40 | 85.7% | 11.0% | 2.7% | 0.29% | 0.2% |
| 16000 | 48.66 | 128 | 0.695 | 936295 | 66.11 (63.76 – 67.75) | 427 | 0.55 | 85.8% | 10.8% | 2.8% | 0.31% | 0.2% |

Column meanings:

- **mesh:** the PPPM part of the 58 Γ_h actions (spreading, FFTs, influence, gathering).
- **real space:** the K^S pair part of the same actions.
- **setup:** the per-step pair list, weights, stencils and degree term.
- **other:** the tridiagonal eigenproblem, f evaluation, combination, vector work and updates.

Observations:

- **The mesh part dominates** at 81–86% of the step. Reorthogonalisation is ≤ 0.5% at the production ranks.
- **T(N) is not smooth**, because M takes discrete values from the accuracy rule. For example, from N = 2048 to 4000
  M only goes from 72 to 81, so the step time rises by 1.5× while N rises by 2×.
- **T/(N ln N)** is 290–440 µs for N ≥ 512 (555 µs at N = 256).
- **Fit of log T on log N:** slope 1.11 over N = 256 … 16000 (9 points), and 1.21 over N = 2048 … 16000 (6 points).
  These slopes describe this measured range on this machine only. They are not an asymptotic complexity.

### Baselines

**Direct sum + Lanczos.**

- The unsplit kernel is summed in real space over all images |r + nL| ≤ R_d. The pair list comes from a periodic
  replica tree. The Lanczos, ranks, dt and step are the same as for PPPM.
- R_d = 19 is the smallest cutoff whose largest relative operator error over the 14 Experiment 1 configurations
  (1.03e-7) does not exceed that of the production PPPM set on the same configurations (2.11e-7). See
  `timing/direct_accuracy.json`.

**Dense route.** The dense Γ_h is assembled and eigendecomposed once per step, which gives exact matrix functions of
the same T7 operator. It is listed as a cost only, since it has no Krylov error and so is not an equal-accuracy
comparison.

| N | direct sum, median (min – max) [s] | image pairs | real-space share | PPPM [s] | direct / PPPM | dense assembly + eig [s] |
|---|---|---|---|---|---|---|
| 64 | — | — | — | — | — | 0.05 + 0.01 |
| 128 | — | — | — | — | — | 0.20 + 0.04 |
| 256 | 5.5 (5.3 – 6.0) | 1022122 | 96.4% | 0.79 | 6.9 | 0.61 + 0.28 |
| 512 | 14.4 (14.0 – 15.3) | 2042224 | 96.8% | 1.23 | 11.6 | 5.92 + 2.07 |
| 1024 | 35.4 (31.8 – 36.1) | 4082686 | 97.7% | 2.57 | 13.8 | 51.56 + 18.71 |
| 2048 | 87.0 (79.6 – 93.9) | 8169312 | 96.0% | 6.45 | 13.5 | — |
| 4000 | 156.6 (151.5 – 159.0) | 15955968 | 96.3% | 9.74 | 16.1 | — |

- **Direct sum:** the log–log slope is 1.24 over N = 256 … 4000. At fixed density and fixed R_d the image pair count
  is proportional to N.
- **Direct / PPPM ratio:** it grows from 7 to 16 over the measured range. It depends on R_d, i.e. on the accuracy
  matched, and on this implementation.
- **Dense route:** at N = 256 it costs about one PPPM + Lanczos step (0.89 s against 0.79 s). At N = 1024 it costs
  70 s against 2.6 s. It was not timed above N = 1024.

## Final summary for the paper

### Recommended production parameters and validated scope

All values are in `sec6_config.json` and are read by every script.

| item | value | validated scope (tested, not a bound) |
|---|---|---|
| spatial set | T7: ξ = 0.7, s = 4.100151562, η_N = 0.7, p = 7 | relative operator error ≤ 2.2e-7 on the 44 static Experiment 1 configurations; 39 of them meet the nominal 1e-7. T9 − T7 refinement 5.6e-8 – 1.0e-7 at ρ = 30/216, N = 256 … 16000 (one state per N). Dynamics T9 − T7 ≤ 8.8e-7 in temperatures at N = 64 |
| fine set (comparisons only) | T9: s = 4.6278, η_N = 0.6, p = 8 | ≤ 2.6e-9 on the 14 tested configurations; 10 of 14 meet the nominal 1e-9 |
| dt | 0.042917661805891890 | ideal gas at N = 64 and 256: equilibrium temperatures and Maxwell moments consistent with the exact values within about 1% (95% CI). dt − dt/2 at N = 64 not resolved |
| noise rank | 48 | ρ = 30/216 static uniform, N ≤ 16000; Experiment 1 families N ≤ 256; diagnostic τ ≤ 5. Dynamic states: N = 64, 256 (718 offline O-steps, exact dt·λ_max ≤ 8.8); N = 4000, 16000 (3 short paths, estimated dt·λ_max ≤ 4.3). Largest sample error 6.2e-4 static, 5.3e-4 dynamic |
| damping rank | 10 | same cases; ≤ 3.4e-13 at production dt, ≤ 5.7e-7 at τ = 5, ≤ 1.0e-12 on dynamic states |
| Lanczos conventions | direct start, full two-pass reorthogonalisation, Π after every action, centre-of-mass momentum copied, f_dt via `expm1`, negative Ritz values recorded and never clipped | no negative Ritz value occurred in any recursion |

**Not covered:**

- N > 16000;
- other densities, κ or amplitudes (in particular the Lennard-Jones application settings);
- states with a larger spectral range than above;
- interacting potentials (U ≠ 0);
- multithreaded or other-machine timings.

### Status of each result

| result | setting | criterion / reference | status |
|---|---|---|---|
| Exp1 Fourier floor after the \|k\| fix | precision stage | float64 vs 80-bit, image sum R = 40 vs 45 | **pass**: floor 6e-17 – 8e-16, at the level of both checks |
| Exp1 spatial sets at nominal tol | T5 / T7 / T9, 14 / 44 / 14 configs | real-space image sum | **fail** on 4 / 5 / 4 configs; used with measured scope |
| large-N Lanczos references | 30 cases, S = 260 … 380 | S − 20 and S − 40 ≤ 0.1 tol | **pass**, all budgets; none unresolved |
| noise rank 48 at 1e-3 | 86 static cases, 5 dynamic sets | dense (small), high order (large) | **pass** (sample max 6.2e-4) |
| damping rank 10 at 1e-6 | same | same | **pass** |
| negative Ritz values | 86 static recursions plus every dynamic Lanczos step | recorded; a dynamic run aborts on one | none (smallest Ritz value 4.5e-3, N = 16000) |
| momentum conservation | stages 3–4 | ‖ΔP‖/‖P‖ | **pass** (≤ 3.1e-14) |
| relaxation and burn-in | N = 64, 256 | halves after t_b | **pass** (halves consistent) |
| equilibrium T_a, T_kin, Maxwell variance and kurtosis | N = 64, 256 | exact values, 95% CI | **pass** (CIs cover exact) |
| offline O-step checks | 718 actual steps | dense f(Γ_h) | **pass** (noise ≤ 1.1e-7, damping ≤ 4.1e-15) |
| prod vs dense dynamics | N = 64 (16), 256 (4), t = 6 | paired differences, t-CI | **pass**: difference not distinguishable from 0 (≤ 2.4e-11, ≤ 5.1e-7) |
| prod vs hi, short coupled runs | N = 4000 (2 paths), 16000 (1 path), 25 steps | hi self-check ≤ 3.6e-6 | **measured**: ≤ 7.8e-6 and 1.4e-5 in temperatures. Single paths, no CI |
| spatial T9 vs T7 dynamics | N = 64 (16), t = 6 | paired differences | **pass**: ≤ 8.8e-7, one marginal time of 141 |
| time step dt vs dt/2 | N = 64 (16), t = 6.008 | paired differences | **unresolved**: the time-step bias is not clearly resolved; no order claimed; dt/4 not run |
| timing | N = 256 … 16000 (PPPM), 256 … 4000 (direct) | one machine, 1 thread | measured; fits describe this range only |

### Reused, rerun and why

- **Experiment 1:** the split, cutoffs, mesh, action and precision stages were rerun after the |k| rounding fix
  (`exp1_results/run_2026-10-13_exact_k.log`). The earlier log is kept (`run_2026-10-13_initial_rounded_k.log`).
- **Earlier Lanczos, dynamics and FFT-vs-direct directories:** not reused. They used the rounded |k| and s = 4.100.
  They are kept unchanged for traceability (stage 1).
- **dt:** the value is taken from `dynamics_results/params_small.json`. It is a definition, so no recomputation is
  needed.
- **Stage 2:** the selection rule was added after the runs, in `sec6_rank_select.py`. The run-time
  `sec6_lanczos.py` is unchanged, so its code hash `9e76b0026a743fff` matches the raw files.
- **Stage 3:**
  - The N = 256, t = 30 runs were superseded by t = 80 runs with the same seeds, because t = 30 was shorter than the
    burn-in (`ideal_gas/raw/superseded_T30/`).
  - The N = 64 runs were extended by 16 trajectories.
- **Stage 4:** reused the stage 3 trajectories as the prod arm at N = 64 and 256, with the same config hash and
  identical inputs (see stage 4).
- **Stage 5:** all new. The direct-sum cutoff was chosen for this comparison.

### Code versions, seeds and hashes

- Config hash with ranks: `808cc0703acf6e50`, used by stages 3–5. Without ranks: `69e23b991db39a74`, used by
  stage 2.
- Code hashes are in each `summary.json` under `provenance`, and per run in the raw records.
  - Stage 2 runs: `9e76b0026a743fff`.
  - Stage 3 extension and stage 4 small runs: `768a0aec214aba9b`, i.e. `sec6_dynamics.py` as committed in
    0bcacd1.
  - Stage 4 large runs and the analysis: `e957c1a529956d34`. This version differs from the previous one only in
    the `large` job list, which skips trajectories > 0 at N = 16000.
  - Stage 5: `0b21c2ade2937f73`.
- Seeds:
  - positions `[20261014, N, pos]`;
  - Lanczos vectors `[20261015, …]`;
  - initial states `[20261016, N, traj]`;
  - noise `[20261017, N, traj, 1]`;
  - timing state index 900.

### Stable paths for the paper (relative to `ewald_longitudinal/`)

| content | path |
|---|---|
| shared configuration and loader | `sec6_config.json`, `sec6_config.py` |
| Exp1 summary, integrity, set status | `exp1_results/summary.json`, `exp1_results/integrity.json`, `exp1_results/parameter_set_status.{json,csv}` |
| Exp1 precision after the fix | `exp1_results/fourier_precision.csv` |
| Exp1 figures | `exp1_results/figures/exp1_action.{pdf,png}`, `exp1_results/figures/exp1_components.{pdf,png}` |
| Lanczos ranks and errors | `sec6_results/lanczos/{rank_selection.json, required_ranks.csv, error_at_ranks.csv, reference_checks.csv, summary.json, fig_lanczos_rank.png}` |
| ideal gas | `sec6_results/ideal_gas/{summary.json, offline_ostep_checks.json, observables_N64.csv.gz, observables_N256.csv.gz, fig_ideal_gas.png}` |
| dynamics comparison and dt | `sec6_results/dynamics/summary.json` |
| timing | `sec6_results/timing/{timing_summary.csv, summary.json, timing_raw.json, direct_accuracy.json, spatial_check.json, dense_cost.json, fig_timing.png}` |
| this document | `sec6_results/README.md` |
