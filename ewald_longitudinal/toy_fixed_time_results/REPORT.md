# Fixed-time paired validation: fast (PPPM + Lanczos 40/16) vs dense, with dense time-step refinement

Script: `test_toy_fixed_time.py` (stages `selftest`, `run`, `analyze`). Toy double well, law A, N = 64, the same
BAOAB, pair potential and friction kernel as the long runs. The layering follows the error chain of the
fast-particle Landau preprint (Zhao, Dou, Lei, §§1 and 5.2):
- operator errors on static configurations (`../radial_kernel_static_results`);
- then fixed-time statistics, with the production chain's paired difference judged against the dense temporal bias
  on the same Brownian path indices.

## Design

| item | value |
|---|---|
| initial states | the 5 v2 final states (seeds 101–105) used by `production_t60` |
| noise paths | 16 per initial state. Path (s, j) draws its finest-level Gaussians from `SeedSequence([20261008, s, j])` |
| dense step ladder | 2dt = 0.01, dt = 0.005 (production), dt/2, dt/4 = 0.00125 (finest, used as the comparison level) |
| fast | PPPM + Lanczos (noise rank 40, damping rank 16, ξ = 0.7, s = 4.10, η = 0.7, p = 7) at dt, on exactly the dt-level noise of dense at dt |
| observables at t = 0.5, 1, 2 | K/N = Σ\|p\|²/(2mN), U/N, ⟨\|v\|⁴⟩ = mean_i \|v_i\|⁴, MSD = mean_i \|q_i(t) − q_i(0)\|² (unwrapped) |
| parameters | γ = 0.5, κ = 0.7, r_ref = 1.3, kT = 0.7, m = 1, L = 5.5 |

**Brownian coupling across step sizes.** This uses nested streams, as in `test_true_dynamics`. A level with step h
consumes the two Gaussians ξ₁, ξ₂ of the next finer level per step:

ξ_h = S_h⁻¹ (R_{h/2} S_{h/2} ξ₁ + S_{h/2} ξ₂), with R_h = e^{−hΓ/m}, S_h = [kT m (1 − e^{−2hΓ/m})]^{1/2},
and Γ = Γ(q_half) of the coarse run itself.

The constructed ξ_h is passed on to the next coarser level. Sharing a seed alone would not couple the paths.

**Self-test** (`selftest.json`):
- the dense O-step equals the production reference O-step: max difference 0;
- the nested noise is exactly N(0, Π) for frozen Γ: A² + B² − 1 ≤ 4e-16;
- for frozen Γ, one coarse step reproduces two fine steps to 7e-15.

**Statistics.**
- Per initial state, over its 16 paths: the paths are independent given the state, so df = 15.
- Across the 5 initial states, with the per-state means as the units (df = 4). Noise replicates are never counted as
  independent initial states.
- No convergence order is assumed or fitted.
- When a time-step difference is not resolved (its CI includes 0), only absolute values are compared; no ratios are
  formed.

**Cost** (one core, mean per path to t = 2): dense dt/4 31.5 s, dt/2 17.1 s, dt 9.9 s, 2dt 5.8 s; fast at dt 35.4 s.
At N = 64 the fast chain is slower than dense, so no performance claim is made. The 80 paths took about 33 min on
4 cores. Four of them (states 101 and 102, paths 0–1) come from the pilot invocation with identical seeds. The full
command is in `run_meta.json`.

## Results (across 5 initial states; per-state values in `fixed_time_per_state.csv`)

| t | observable | dense mean (dt) | fast − dense [95% CI over 5 states] | relative | dense dt − dt/4 [95% CI] | dense 2dt − dt/4 [95% CI] | pathwise RMS: fast−dense (max state) / dense dt−dt/4 (min state) |
|---|---|---|---|---|---|---|---|
| 0.5 | K/N | 1.015 | +5.5e-10 [-2.5e-09, +3.6e-09] | +5e-10 | -6.1e-04 [-2.7e-03, +1.5e-03] | -4.4e-03 [-8.0e-03, -7.1e-04] **(excl. 0)** | 1e-08 / 3e-03 |
| 0.5 | U/N | -5.681 | -2.1e-09 [-6.0e-09, +1.8e-09] | -4e-10 | +2.7e-04 [-1.2e-03, +1.8e-03] | +2.7e-04 [-3.3e-03, +3.8e-03] | 2e-08 / 5e-03 |
| 0.5 | ⟨|v|⁴⟩ | 6.861 | -6.5e-10 [-5.0e-08, +4.9e-08] | -9e-11 | -4.9e-03 [-3.5e-02, +2.6e-02] | -6.8e-02 [-8.3e-02, -5.4e-02] **(excl. 0)** | 2e-07 / 6e-02 |
| 0.5 | MSD | 0.09628 | +2.3e-09 [+1.5e-09, +3.1e-09] **(excl. 0)** | +2e-08 | -4.7e-05 [-1.5e-04, +5.4e-05] | -4.2e-05 [-2.8e-04, +2.0e-04] | 3e-09 / 3e-04 |
| 1 | K/N | 1.021 | +2.4e-09 [-2.2e-08, +2.7e-08] | +2e-09 | -2.8e-03 [-9.8e-03, +4.1e-03] | -3.7e-03 [-8.6e-03, +1.2e-03] | 1e-07 / 1e-02 |
| 1 | U/N | -5.666 | -6.0e-09 [-3.4e-08, +2.2e-08] | -1e-09 | +4.0e-03 [-1.5e-03, +9.5e-03] | -3.1e-04 [-1.0e-02, +9.8e-03] | 1e-07 / 2e-02 |
| 1 | ⟨|v|⁴⟩ | 6.973 | -8.2e-09 [-3.2e-07, +3.0e-07] | -1e-09 | -9.6e-02 [-2.8e-01, +9.2e-02] | -1.1e-01 [-3.3e-01, +1.2e-01] | 1e-06 / 3e-01 |
| 1 | MSD | 0.1785 | +3.7e-09 [+1.4e-09, +6.0e-09] **(excl. 0)** | +2e-08 | -2.5e-04 [-1.2e-03, +7.3e-04] | -7.4e-04 [-2.2e-03, +7.3e-04] | 8e-09 / 2e-03 |
| 2 | K/N | 1.042 | -3.8e-07 [-8.9e-07, +1.3e-07] | -4e-07 | +7.3e-03 [-1.8e-02, +3.3e-02] | -1.4e-02 [-5.8e-02, +3.0e-02] | 5e-06 / 7e-02 |
| 2 | U/N | -5.71 | -2.9e-08 [-5.8e-07, +5.2e-07] | -5e-09 | -5.8e-03 [-1.3e-02, +1.3e-03] | +4.7e-03 [-4.0e-02, +5.0e-02] | 3e-06 / 8e-02 |
| 2 | ⟨|v|⁴⟩ | 7.195 | -7.5e-06 [-2.0e-05, +4.7e-06] | -1e-06 | +7.7e-02 [-2.6e-01, +4.1e-01] | -1.6e-01 [-9.6e-01, +6.4e-01] | 1e-04 / 1e+00 |
| 2 | MSD | 0.3255 | +6.7e-09 [-1.3e-07, +1.4e-07] | +2e-08 | +5.9e-03 [-8.2e-03, +2.0e-02] | -2.0e-03 [-1.2e-02, +8.5e-03] | 5e-07 / 2e-02 |


Notes on the table:
- The dense dt mean is the mean of 80 paths.
- "Relative" is the fast − dense mean divided by |dense mean|.
- "(excl. 0)" marks a 95 % CI that excludes 0.
- The pathwise RMS is the root mean square of the per-path difference: the largest per-state value for fast − dense,
  the smallest for dense dt − dt/4.

## Reading

1. **Fast − dense at the production step.**
   - **Size:** |mean| ≤ 7.5e-6 and 95 % bounds ≤ 2e-5 for every observable and time. Relative to the observable this
     is ≤ 1e-6. The difference grows with t, from about 1e-9 at t = 0.5 to about 1e-7–1e-5 at t = 2, as expected for
     chaotic amplification of the 1e-7 operator difference.
   - **The one resolved systematic difference** is in the MSD at t = 0.5 and 1: +2.3e-9 and +3.7e-9 (≈ 2e-8 relative),
     positive in all 5 initial states at t = 0.5. This is consistent with a first-order effect of the measured 1e-7
     operator difference. It is far too small to matter at the scale of any temporal error below.
2. **Dense temporal bias.**
   - **At dt:** not resolved for any observable or time at this sample size. The 95 % intervals bound it, for example
     |K/N| ≤ 2.7e-3 at t = 0.5 and ≤ 3.3e-2 at t = 2, and ⟨|v|⁴⟩ ≤ 0.41 at t = 2.
   - **At 2dt:** resolved at t = 0.5 for K/N (−4.4e-3, −0.43 %) and ⟨|v|⁴⟩ (−6.8e-2, −1.0 %). Every initial state has
     the same sign there.
   - **At later times:** the pathwise spread of the coupled dt-vs-dt/4 differences grows quickly with t, so later times
     resolve less.
   - Whether the temporal bias at dt is smaller than these bounds, and how it scales with the step, is not established.
3. **Is fast − dense clearly smaller than the temporal error?**
   - **Against the resolved 2dt bias** (t = 0.5): yes. The fast − dense bounds are 3.6e-9 (K/N) and 5e-8 (⟨|v|⁴⟩),
     about 6 orders of magnitude below it.
   - **Against the dt bias:** that bias is not resolved, so no ratio is formed and no claim is made that fast − dense is
     smaller than the dt bias itself. What holds is that the fast − dense bounds are 4–6 orders of magnitude below the
     resolution of the dt time-step test (its CI half-widths).
   - **Pathwise:** fast − dense is 4–6 orders of magnitude smaller than the coupled time-step differences at every time
     and for every observable.
   - Within this fixed-time test the production chain is subdominant to time discretisation and to sampling. A CI that
     includes 0 is not read as a proof of equivalence.
4. **Scope.** This is one case (double_well_A), N = 64, t ≤ 2, and five initial states taken from equilibrated long
   runs. It does not address the long-time diffusion difference seen in `production_t60` (t up to 60, after pathwise
   decorrelation at t ≈ 6). That difference remains unexplained, and the fixed-time test neither confirms nor rules
   it out.
