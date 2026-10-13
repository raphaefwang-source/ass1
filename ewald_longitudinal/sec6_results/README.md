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
