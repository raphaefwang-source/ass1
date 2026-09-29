# Direct-start vs Landau-style range-start Lanczos: report

Script: `test_lanczos_range_start.py`; the run took 800 s. Detailed tables are in `summary.md`, the data in `*.json`,
and the ten plots are `plot01` … `plot10`.

**Setup**
- Same Γ_h as before (literal PPPM action, Π after every action). The functions are f_dt with dt λ_max/m = 0.5 and
  β = m = 1, and the √λ debug case.
- The dense configurations, Gaussian vectors and scaling seeds are identical to `test_lanczos_fdt.py`. Direct-start
  medians reproduce the previous run exactly in all 12 configurations.
- Range start is b = ΠΓ_h z, q₁ = b/‖b‖, η_r = ‖b‖ Q_r g_dt(T_r) e₁ with g_dt = f_dt/λ, evaluated through expm1.
  Every Ritz value is checked before g is evaluated, and nothing is clipped or thresholded.
- Rank r in range start costs r + 1 Γ_h actions; timings include the action that forms b.

**Formula check.** f(Γ)z = g(Γ)Γz holds on Range(Γ_h), which equals Range(Π) here. The small-λ limit
g ≈ √(2dt/β)/√λ is correct. One structural point matters: K_r(Γ, Γz) ⊂ K_{r+1}(Γ, z), so range start at rank r
searches a subspace of direct start at rank r + 1. Its implicit polynomial must approximate a function g that is more
singular at λ → 0 than f.

## Answers

1. **Same target?** Yes. Range start converges to the dense f_dt(Γ_h)z to at most 9.2e-15 (direct: 4.2e-15). At
   N = 128–4000 the two methods' converged results agree to at most 8.8e-15.

2. **Rank on configurations A and B:** range start needs more ranks, not fewer. Medians (and maxima):

   | config | tol | direct | range | range/direct |
   |---|---|---|---|---|
   | A | 1e-6 | 18 (19) | 23 (24) | 1.28 |
   | A | 1e-8 | 25 | 30 | 1.20 |
   | A | 1e-10 | 31 | 36 | 1.16 |
   | B | 1e-6 | 36 | 45 (46) | 1.25 |
   | B | 1e-8 | 48 (49) | 57 (58) | 1.19 |
   | B | 1e-10 | 58 (60) | 67 | 1.14 |

   - All 12 dense configurations show the same pattern: range/direct = 1.13–1.43.
   - The √λ debug test (Landau's own Γ^{−1/2} b construction) gives the same ranks as f_dt: A 25 → 30, B 48 → 57 at 1e-8.
   - Rank versus condition ratio has the same slope for both methods (plot 3); range start is shifted up by about
     5–9 ranks.

3. **At N = 4000, tol 1e-8:** range start needs 169 and 166 ranks against 128 and 126 for direct start (+32%). At
   1e-6 it is 120 and 119 against 85 (+41%).

4. **Does the rank still grow with N?** Yes, and faster for range start:

   | N | 128 | 256 | 512 | 1024 | 2048 | 4000 |
   |---|---|---|---|---|---|---|
   | direct, 1e-8 | 39 | 50 | 64 | 82.5 | 102 | 127 |
   | range, 1e-8 | 47.5 | 62 | 79 | 103 | 128.5 | 167.5 |
   | direct, 1e-6 | 28.5 | 36 | 45 | 56.5 | 71 | 85 |
   | range, 1e-6 | 36 | 47 | 60 | 76 | 94.5 | 119.5 |

   The range/direct ratio rises from 1.22 to 1.32 at 1e-8, and from 1.26 to 1.41 at 1e-6 (plot 7).

5. **Fitted exponent:** α_range = 0.363 at 1e-8 and 0.345 at 1e-6, against α_direct = 0.344 and 0.320. It is not
   close to zero, and not smaller than direct start.

6. **Is the rank explained by low-mode spectral weight?** No, not in the direction the motivation assumed. Range
   start does suppress low-mode weight strongly (plot 4). In B, modes below 2λ_* carry 7.6% of ‖z‖² but only 0.013%
   of ‖b‖². The rank is set anyway by how hard the function is to approximate near the bottom of the spectrum.
   Direct start must represent f ~ √λ; range start must represent g ~ λ^{−1/2} and then multiply by λ. Resolving the
   singular g on [λ_*, λ_max] costs more ranks than the λ² reweighting saves. The √(λ_max/λ_*) dependence remains.

7. **Share of the stochastic increment from the lowest modes:**
   - Dense configurations: modes below 2λ_* carry 9.0% of ‖f_dt(Γ)z‖² in A and 0.60% in B. Modes below 5λ_* carry
     40% and 3.0%.
   - Large N (Gauss-quadrature estimate from the converged Lanczos): the share below 5 × the smallest Ritz value
     falls from 18% (N = 128) to 7–10% (256), 2% (512), 0.6% (1024), 0.15% (2048) and 0.05% (N = 4000).
   - So the troublesome modes are physically negligible at large N. They still have to be resolved to meet a
     relative tolerance of 1e-6–1e-8, which is far below their share.

8. **Is range start faster overall, counting the extra action?** No. Production-style timed solves with the
   d_r ≤ tol/10 rule at tol 1e-8:

   | N | direct actions / time | range actions / time |
   |---|---|---|
   | 1024 | 87–90 / 2.5–2.6 s | 108–110 / 3.0 s |
   | 4000 | 131–132 / 15.8–15.9 s | 172–175 / 20.6–20.9 s |

   - Range start is about 20–35% slower at every N and at both tolerances.
   - Both methods' final errors meet the tolerance; at N = 4000 they are 5.5–6.7e-9 (direct) and 4.9–5.3e-9 (range).
   - Time exponents are 1.38 (direct) and 1.38 (range) at 1e-8, and 1.34 and 1.36 at 1e-6.
   - Reorthogonalization plus tridiagonal work is at most 0.5 s of about 21 s at N = 4000. The Γ_h actions dominate.

9. **Is d_r ≤ tol/10 still a safe stopping rule for range start?** Yes. There were zero false stops in all 224 dense
   runs at every tolerance, with over-solve of 1.08–1.20 (median) and errors at stop of at most 0.25 × tol. At
   N = 1024–4000 it stopped with errors of 3.6–5.3e-9 at tol 1e-8.
   - d2_r ≤ tol had 0.9% false stops at 1e-4 and 1e-6; plain d_r ≤ tol had 11–54%.
   - None of these is a rigorous bound.

10. **Negative Ritz values?** None. In range start every Ritz value stayed at or above λ_* to within 2e-14 relative,
    including in all 12 dense configurations. There were no Ritz stops.
    - At large N the smallest range-start Ritz value equals the direct-start one (1.01e-2 at N = 4000).
    - The largest value of g evaluated at a Ritz value was 2.6, so the λ^{−1/2} singularity is never approached.
    - Momentum and nullspace residuals are at roundoff: ‖(I−Π)η_r‖/‖η_r‖ ≤ 1.4e-16, total momentum ≤ 8.4e-16, and
      ‖(I−Π)Γ_h z‖/‖b‖ ≤ 1.1e-16 before projection.

11. **Does the Landau-style threshold (1e-12‖T_r‖) help?** It is irrelevant here. It discarded exactly zero Ritz
    mass in every run, because the smallest Ritz value (≥ λ_*) is about 1e-2 of ‖T_r‖, ten orders above the
    threshold. The action bias, covariance bias and rank change are all exactly 0. Its purpose — protecting
    C^{−1/2} from null or negative spectrum — does not arise for Γ_h with Π projection.

12. **Production choice:** direct-start Lanczos with Π after every action and the d_r ≤ tol/10 stopping rule. Range
    start computes the same result but needs 13–43% more ranks, one extra action, and grows at least as fast with N.

**Main question: does range start restore a bounded accuracy-resolved rank, and so make an O(N log N) thermostat
plausible?** No. The rank grows as N^0.36 with range start and N^0.34 with direct start. The complete root-action cost
grows about as N^1.38 for both. The O(N log N) claim is not supported for fixed-density random systems at fixed
relative tolerance with either start vector. The rank is set by λ_max/λ_*, and λ_* falls with the longest collective
wavelength. Those modes carry a vanishing share of the noise (0.05% at N = 4000), which suggests one direction worth
testing: relax the tolerance on the low-mode subspace, or deflate or precondition it separately. This experiment
does not test that.
