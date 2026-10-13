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
