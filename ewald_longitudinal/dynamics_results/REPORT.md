# First time-dependent test: ideal-gas PPPM–Lanczos finite-time FDT thermostat

Script: `test_true_dynamics.py`, run in stages: small, control, equil, large, timing, report.
Data: `summary.json`, `params_small.json`, `timing.json`, `damping_rank_high_tau.json`, `raw_*.npz`; logs in `log_*.txt`.
Plots: `plot1` … `plot8`.

## Setup

- **Model and potential.** Ideal gas U = 0, β = m = 1, density 30/216. Same screened pure-longitudinal Γ_h (PPPM
  ξ = 0.7, s = 4.10, η = 0.7, p = 7) with unchanged spatial accuracy. No transverse kernel.
- **Soft-repulsive fluid.** The draft specifies a periodic soft-repulsive fluid but does not define a unique
  potential, and the repository contains none, so no potential was invented and this study stops at the ideal gas.
- **Integrator.** A(dt/2) O(dt) A(dt/2). The O-step is p ← p_mean + R_dt Πp + f_dt(Γ_h) Πξ with
  R_dt = exp(−dtΓ_h/m) and f_dt = √(β⁻¹m[−expm1(−2dtλ/m)]). Γ_h is evaluated at q_half. The mesh and influence
  function are built once and reused; only the particle-dependent parts are rebuilt each step.
- **Formula check.** Γ_h·1 = 0 and f(0) = 0, so the mean momentum is copied exactly. The O-step preserves
  N(p_mean, β⁻¹mΠ) exactly for a frozen Γ_h, so the late-time momentum law carries no dt bias.
- **Nested streams.** ξ_coarse = S_h⁻¹(R_{h/2}S_{h/2}ξ₁ + S_{h/2}ξ₂) is exactly N(0, Π) for a frozen Γ_h, and is
  used for the dt/4 → dt/2 → dt → 2dt coupling.
- **Initial state.** Uniform positions. Thermal part with (T_x, T_y, T_z) = (2, 0.5, 0.5), sample mean removed, then
  v_cm = (0.3, −0.2, 0.1) added. The actual mean initial temperatures at N = 64 are (1.959, 0.528, 0.492), using the
  conditional N − 1 normalization.
- **Time step.** dt = 0.2 m / median λ_max(q₀) = 0.04292 at N = 64, where the median λ_max is 4.66 (range
  2.49–12.7). The same physical dt is used at every N.
  - Encountered dt·λ_max(t): 0.07–3.66 at N = 64, 0.43–3.65 at N = 4000 and 0.73–10.06 at N = 16000.
  - With U = 0, particles can approach arbitrarily closely, and each close pair spikes λ_max. The exponential step
    is unconditionally stable, so dt was not tuned around the spikes.
- **Ranks.** Damping rank 12, chosen from a static test (rank 5 already gives 1e-8 at dt·λ_max = 0.2). Noise
  ranks 40 (production) and 48; high-rank Lanczos uses 90 for noise and 24 for damping.
- **Coupled methods at N = 64** (16 trajectories, T_end = 6.0, 140 steps; same initial states and Gaussian streams):
  - dense eigendecomposition (reference);
  - "tight" dense with the tol-1e-9 PPPM set (a spatial-error proxy);
  - PPPM high-rank Lanczos;
  - PPPM with noise rank 40 or 48, damping rank 12;
  - dense at 2dt, dt/2 and dt/4.
- **Additional runs.**
  - Long equilibrium runs: 16 × T = 30 at rank 40, with burn-in to t = 15 (more than twice the slowest relaxation
    time 1/(2λ_*) ≈ 6.7).
  - Negative control: 4 trajectories with raw independent noise.
  - N = 4000 and N = 16000: 4 seeds each at rank 40, plus seed 0 at rank 48 on the same stream; T = 3, 70 steps.

## Results

**Momentum.** The largest ‖P(t) − P(0)‖/max(1, ‖P(0)‖) over every coupled method at N = 64 is 1.6e-14. It is
1.2e-14 in the long runs, at most 2.5e-14 at N = 4000 and at most 3.8e-14 at N = 16000 (plot 3). The negative
control with independent noise drifts to 1.0–1.7, i.e. O(1).

**Paired differences at N = 64.** Each entry is the largest over time of |mean over trajectories| of the paired
difference; the pathwise RMS is similar in magnitude.

| comparison | T_x | T_y | T_z | T_kin | O_k = ⟨cos(2πx/L)⟩ |
|---|---|---|---|---|---|
| rank 40 − dense | 6.6e-10 | 6.1e-10 | 1.0e-9 | 4.1e-10 | 5.6e-11 |
| rank 48 − dense | 1.2e-11 | 7.3e-12 | 5.1e-12 | 3.7e-12 | 4.3e-13 |
| high-rank Lanczos − dense | 3.0e-14 | 1.1e-13 | 5.7e-14 | 2.3e-14 | 7.6e-15 |
| tight − dense (spatial) | 2.4e-7 | 3.1e-7 | 5.6e-7 | 1.7e-7 | 4.2e-8 |
| dense(dt) − dense(dt/2) | 2.8e-2 (SE 2.3e-2) | 3.3e-2 (1.9e-2) | 3.4e-2 (2.4e-2) | 1.1e-2 (1.1e-2) | 1.5e-2 (9e-3) |
| dense(dt/2) − dense(dt/4) | 2.6e-2 (1.8e-2) | 1.7e-2 (1.6e-2) | 3.5e-2 (2.4e-2) | 1.1e-2 (1.0e-2) | 2.5e-3 (5e-3) |
| dense(2dt) − dense(dt) | 2.8e-2 (2.9e-2) | 2.7e-2 (2.6e-2) | 4.1e-2 (3.1e-2) | 2.7e-2 (1.3e-2) | 2.9e-3 (9e-3) |
| sampling SE of the mean (max over t) | 8.4e-2 | 4.9e-2 | 6.1e-2 | 3.1e-2 | 2.9e-2 |

- The nested streams are exact in distribution, but only an approximate pathwise coupling while Γ_h(q) moves.
- The temporal differences are therefore at or below about 3e-2 and comparable to their own SE: the temporal weak
  error is not resolved beyond about 1e-2 to 3e-2. No weak order is claimed.
- The ordering is unambiguous: Krylov (≤ 1e-9) ≪ spatial (≤ 6e-7) ≪ temporal and sampling (about 1e-2).

**Rank 48 vs rank 40 at large N** (the only regime where rank 40 carries a measurable truncation error). Largest
paired difference over the trajectory, on the same seed and stream:

| N | T_x | T_y | T_z | T_kin | O_k |
|---|---|---|---|---|---|
| 4000 | 3.0e-5 | 3.9e-5 | 4.1e-5 | 1.9e-5 | 4.4e-6 |
| 16000 | 1.4e-4 | 9.4e-5 | 9.0e-5 | 9.3e-5 | 1.7e-5 |

For comparison, the seed-to-seed standard deviation of the late-time temperatures is 0.009–0.019 (N = 4000) and
0.002–0.007 (N = 16000). The temporal differences at N = 64 are about 1e-2 to 3e-2.

**Equilibrium** (long runs, N = 64; statistics are per trajectory, then across 16 trajectories):

- 95% confidence intervals: T_x 1.002 [0.977, 1.028], T_y 1.000 [0.963, 1.036], T_z 0.996 [0.977, 1.016],
  T_kin 0.999 [0.979, 1.020]. All contain 1/β = 1.
- Conditional Maxwell check for Πp / √(m/β·(N−1)/N): variance 0.9995 ± 0.010, skewness −0.006 ± 0.007, excess
  kurtosis −0.023 ± 0.025. Each component is individually consistent. The QQ plot and histogram match the standard
  normal (plot 7).
- The short-run window t ∈ [3, 6] had not equilibrated the slowest mode: its T_y interval [0.87, 0.98] excluded 1.
  The long runs resolve this.

**Relaxation at large N.** All seeds relax from (2.0, 0.5, 0.5) towards 1 (plot 1b). By t = 3 they reach
T_x ≈ 1.05–1.10 and T_y, T_z ≈ 0.94–0.97; relaxation is visible but not complete. Equilibrium at large N was not
checked: the slow long-wavelength modes have a relaxation time of about 1/(2λ_*) ≈ 50 or more.

**Damping rank.** Relative error of rank 12 against a converged exp(−dtΓ_h/m)Πp:

| dt·λ_max | rank-12 error | rank-16 error |
|---|---|---|
| 5 | ≤ 7.7e-10 | ≤ 1.1e-14 |
| 10 | ≤ 7.0e-7 | ≤ 1.6e-10 |
| 20 | ≤ 1.8e-4 | ≤ 5.7e-7 |

Rank 12 therefore met the 1e-6 target throughout the observed range (dt·λ_max ≤ 10.06), but only marginally. On
the final configurations, rank 12 and rank 24 agreed to at most 1.2e-15.

**Cost.** Single-process wall time per step with 52 Γ_h actions (40 noise + 12 damping):

| N | 64 | 512 | 4000 | 16000 |
|---|---|---|---|---|
| s / step | 0.132 | 0.79 | 7.44 | 40.4 |
| T/(N log N) [s] | 4.95e-4 | 2.47e-4 | 2.24e-4 | 2.61e-4 |

For N ≥ 512 the fitted exponent is 1.14; N log N over the same range corresponds to about 1.13.

## Answers

1. **Anisotropic temperatures relax to the target.** At N = 64, T_x goes 1.96 → 1 and T_y, T_z go 0.5 → 1 by
   t ≈ 3–4. The long runs give T_a = 1 within 95% confidence for every component. At N = 4000 and 16000 the same
   relaxation is clearly under way by t = 3.
2. **Total momentum is conserved pathwise to roundoff** (≤ 3.8e-14 in every run, at every N). The independent-noise
   control breaks it at O(1).
3. **The dense finite-time FDT dynamics behaves as expected:** exact conditional-Maxwell equilibrium within
   uncertainty, no momentum drift, and stable steps even at dt·λ_max = 3.7 (N = 64) and 10 (N = 16000).
4. **PPPM high-rank Lanczos reproduces the dense dynamics to 1e-13** in every observable. The PPPM spatial effect
   itself, measured as tight minus dense, is at most 6e-7.
5. **Rank 40 is not dynamically distinguishable.** At N = 64 it differs from dense by at most 1e-9. At large N,
   rank 48 minus rank 40 is at most 4e-5 (N = 4000) and 1.4e-4 (N = 16000). That is 2–3 orders below the
   trajectory scatter and the temporal differences.
6. **Rank 48 is not materially better in the observables.** Its changes relative to rank 40 are far below every
   other error source.
7. **Damping rank:** 12 was sufficient here (≤ 1e-6 up to the observed dt·λ_max = 10). Because close-pair spikes
   can exceed 10 and rank 12 degrades to 1.8e-4 at dt·λ_max = 20, rank 16 is recommended for margin: ≤ 6e-7 at 20,
   for about 8% more cost per step.
8. **Fixed-rank Lanczos errors are far below the temporal discretization error at the intended dt:** about 1e-9 at
   N = 64 and at most 1.4e-4 at N = 16000, against a temporal difference of about 1e-2 to 3e-2 at dt.
9. **At N = 16000, rank 40 still gives acceptable dynamics,** consistent with its static 1e-3 root-action budget
   (9.1e-4 at N = 16000). Only one coupled rank-48 trajectory per large N was run.
10. **Production ranks:** noise rank 40 (rank 48 buys nothing measurable up to N = 16000) and damping rank 16
    (12 was adequate in these runs but marginal at the largest spikes).
11. **Is the cost O(N log N)?** Yes, within the stated scope. With fixed ranks (noise 40, damping 12) chosen from
    the tested accuracy budget, the measured wall time per step follows N log N from N = 512 to 16000:
    T/(N log N) stays at 2.2–2.6e-4 s, with an exponent of 1.14. This is a measured statement over the tested
    range only. The fixed noise rank is verified to meet its 1e-3 budget only up to N = 16000, where its margin is
    thin. No asymptotic claim is made.

## Caveats

- **Temporal error** is bounded (about 1e-2 to 3e-2 at dt), not resolved into a convergence order: the nested
  coupling is only approximate while Γ_h(q) evolves, and 16 trajectories leave SE ≈ 1e-2.
- **Large-N statistics** are limited: 4 seeds per N, T = 3, and one coupled rank-48 trajectory per N. Large-N runs
  were executed as 5 concurrent single-threaded processes, so their logged per-step times (about 10 s at N = 4000,
  about 53 s at N = 16000) are inflated by contention. The cost table uses the separate single-process timing stage.
- **At N = 64,** rank 40 covers most of the 189-dimensional space, so this test cannot stress it. The informative
  rank-40 test is the large-N rank-48 comparison.
