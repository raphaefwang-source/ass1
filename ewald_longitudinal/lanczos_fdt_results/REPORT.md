# Lanczos matrix-function stage of the discrete-FDT thermostat: report

Script: `test_lanczos_fdt.py`; full run took 2115 s. Detailed tables are in `summary.md` and the data in `*.json`;
the ten plots are `plot01` … `plot10`.

**Setup**
- Γ_h is the PPPM operator calibrated in the equal-accuracy experiment: ξ = 0.7, s = 4.10, η_N = 0.7, p = 7
  (operator error about 1e-7 in configurations A and B). β = m = 1.
- Every Lanczos action is the literal Γ_h action (real-space pair sum plus spread/FFT/G/IFFT/gather) followed by Π.
- Lanczos uses full reorthogonalization. Dense references use the eigendecomposition of Γ_h restricted to Range(Π).
- Nothing is clipped: no Ritz values, no function arguments, no negative eigenvalue of K̂_L.

**Formula check.** C = β⁻¹m(I − R²) = f_dt(Γ_h)² and C_E = β⁻¹m(I − R_E²) are both correct.
The Euler covariance is non-negative exactly when 0 ≤ dt λ/m ≤ 2.

## Answers

1. **Does Lanczos reproduce f_dt(Γ_h) z?** Yes. For every function and every tested time step it converges to
   the dense result, reaching an error of at most 2.5e-15 at the final rank.
   - Identity checks: ‖QᵀQ − I‖ ≤ 1.2e-15, recurrence residual ≤ 4.9e-16, and T_s = Q_sᵀΓ_hQ_s to 9.2e-16.
   - The literal PPPM action matches the dense Γ_h inside the recurrence to 3.0e-15.
   - Covariance built with the literal action matches the batched dense-matrix path to 1.2e-15.

2. **Ranks needed** (sustained, over 32 vectors; median / max):

   | config | 1e-4 | 1e-6 | 1e-8 | 1e-10 | dim Range(Π) |
   |---|---|---|---|---|---|
   | A (λ_max/λ_* = 33) | 12 / 13 | 18–19 / 19 | 25 / 26 | 31 / 32 | 87 |
   | B (λ_max/λ_* = 228) | 22 / 23 | 36 / 37 | 48 / 49 | 58–59 / 60 | 117 |

   The rank is nearly independent of dt·λ_max/m (0.05–2) and is the same for √λ and for f_E. All of these
   functions share the √λ branch point at λ = 0.

3. **Dependence on λ_*/λ_max.** Strong; the near-zero spectrum is the bottleneck.
   - Across 12 dense configurations, the median rank for 1e-8 rises from 21–25 at a condition ratio of about 26–33
     to 52 at a ratio of 987 (plot 3).
   - The rank tracks roughly √(λ_max/λ_*) × log(1/tol), with some scatter from how the spectrum is distributed
     (E4, at ratio 270, needs fewer ranks than B, at 228).
   - Isolated large eigenvalues cost little. N = 512 has λ_max ≈ 51 from one close pair, yet its rank follows the
     smooth trend.

4. **Momentum.** Yes, projecting with Π keeps zero total momentum to roundoff at every rank.
   ‖(I − Π)η_s‖/‖η_s‖ ≤ 1.7e-16 and |Σ_i η_i|/‖η_s‖ ≤ 1.0e-15 (plot 6); for N = 4000 it is 9e-16.

5. **Negative Ritz values.** None, in any run. The smallest Ritz value over λ_* is 0.99999999999997 or larger,
   and the largest over λ_max is at most 1 + 1.3e-15 (plot 5).
   - The only non-evaluable case is Euler exactly at the stability boundary dt λ_max/m = 2 (64 runs). There the
     top Ritz value exceeds λ_max by about 1 ulp, so 2 − dt θ/m < 0 and f_E is NaN.
   - This is reported, not clipped. At dt λ_max/m = 1.9 the Euler runs converge normally.
   - The first-pass summary mislabelled these 64 runs as "negative Ritz"; the label is corrected in `checks.json`
     and `summary.md`.

6. **Best stopping rule.** Use d_s ≤ tol/10, where d_s = ‖η_s − η_{s−1}‖/‖η_s‖.
   - It is the only rule tested with zero false stops across all tolerances, both configurations and 32 vectors
     per case. It over-solves by a factor of 1.08–1.36 in rank, and the error at stop is 0.04–0.2 × tol.
   - The two-step difference d2_s ≤ tol is almost as good: false stops 0% except 1.9% in B at 1e-6, over-solve
     1.00–1.18.
   - Plain d_s ≤ tol, and d_s ≤ tol at two consecutive ranks, under-solve in the small-gap configuration B:
     52–99% and 7.5–74% false stops. At N = 4000 the two-consecutive rule stopped with a true error of 4.8e-8 at
     tol 1e-8.
   - The generalized residual β_s |e_sᵀf(T_s)e₁| ‖z‖ systematically underestimates the error, by about 1.5–30×,
     giving 94–100% false stops.
   - None of these is a rigorous bound. d_s tracks the true error closely because convergence is linear
     (plot 4); the factor 10 is an empirical safety margin.

7. **Covariance.** Yes, once Monte Carlo noise is separated out, the Lanczos noise reproduces the discrete-FDT
   covariance. Here dt λ_max/m = 0.5 and the same ξ_l are used for Lanczos and for the dense root.

   | | Monte Carlo error (1e3 / 1e4 / 1e5 samples) | Krylov error at stop tol 1e-4 / 1e-6 / 1e-8 / 1e-10 (1e4 samples) |
   |---|---|---|
   | A | 0.22 / 0.070 / 0.023 | 2.4e-5 / 2.0e-7 / 1.8e-9 / 1.1e-11 |
   | B | 0.22 / 0.071 / 0.022 | 3.9e-5 / 2.8e-7 / 2.2e-9 / 1.5e-11 |

   - At a fixed rank of 48 or more the Krylov covariance error is 3e-15 (plots 7 and 8).
   - Exact FDT identity R(β⁻¹mI)Rᵀ + C_exact = β⁻¹mI: residual 3.3e-15 (full space) and 3.5e-15 (Range(Π)).
   - With the Lanczos sample covariance the FDT deviation is 1.21e-2 (A) and 1.02e-2 (B) at 1e4 samples, identical
     to the dense same-sample value. It is all Monte Carlo.
   - Conditional Maxwell step p₁ = R p₀ + η with p₀ ~ N(0, β⁻¹mΠ): the covariance deviation is Monte Carlo only;
     the Krylov part is ≤ 5.4e-10 at stop tol 1e-8. Total momentum stays ≤ 7e-16.

8. **Is the rank bounded in N?** No. At fixed density and fixed spatial accuracy it grows slowly.

   | N | 128 | 256 | 512 | 1024 | 2048 | 4000 |
   |---|---|---|---|---|---|---|
   | rank for 1e-6 | 29 | 36 | 45 | 57 | 71 | 85 |
   | rank for 1e-8 | 39 | 50 | 64 | 83 | 102 | 127 |
   | smallest Ritz value at s = 400 | 6.9e-2 | 4.9e-2 | 3.2e-2 | 2.2e-2 | 1.5e-2 | 1.0e-2 |

   - Fitted exponents: rank ∝ N^0.32 (1e-6) and N^0.34 (1e-8). The smallest positive scale falls as N^−0.56, close to
     the L⁻² ∝ N^−2/3 of the longest-wavelength collective mode.
   - Rank ∝ (smallest Ritz value)^−0.6, consistent with the √(λ_max/λ_*) mechanism of answer 3.
   - All scaling runs converged: the final d_s is at most 5e-15, with orthogonality at most 1.8e-15.

9. **What dominates runtime?** The Γ_h actions.
   - At N = 4000, 400 actions take 48–50 s, while full reorthogonalization at rank 400 takes 0.7 s and evaluating
     f(T_s) at every rank about 3 s. At the ranks actually needed (≤ 130) reorthogonalization is even cheaper.
   - One action costs 3.0 ms at N = 128 and 123 ms at N = 4000, i.e. ∝ N^1.09. Normalized by N log N it is roughly
     constant (3.7–5.2e-6 s).

10. **Is the full PPPM–Lanczos thermostat O(N log N) at fixed accuracy?** Not on this evidence.
    - Each Γ_h action is O(N log N) as measured, but the required rank grows like N^0.33 (the smallest collective
      eigenvalue falls roughly as N^−0.56).
    - The measured cost of one thermostat increment at 1e-8 is 0.13 s → 15.7 s for N = 128 → 4000,
      ∝ N^1.43 ≈ N^{4/3} log N.
    - O(N log N) would need a rank that stays bounded in N, and that did not happen here. Getting there would take
      something beyond plain Lanczos on Γ_h, for example preconditioning or a separate treatment of the
      long-wavelength modes. Those were not tested.
