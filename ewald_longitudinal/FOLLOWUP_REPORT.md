# Follow-up: equal-accuracy ξ optimisation and exact mesh-error decomposition (κ = 1)

Scripts: `test_equal_accuracy_xi.py` (results in `equal_accuracy_xi_results/`) and
`test_mesh_error_decomposition.py` (results in `mesh_error_decomposition_results/`).
The model is unchanged: no Q projector, no clipping of K̂_L, no stabilisation of Γ_h.

## Formula check

- All the stated equations are correct. With the code's convention K_h,ij = (1/V) Σ_k G_h S_i* S_j
  and S_h = Ŵ e^{−ik·q}(1+α), the error K_h − K_F equals
  (1/V) Σ_k e^{ik·r_ij}[D_h + H_h(α_i* + α_j + α_i*α_j)] exactly.
- For ideal deconvolution, D_h is identically zero in exact arithmetic, because Ŵ is real and nonzero for |k_a h| < 2π.
- ξh = η_N π/(2s) and q = η/(2−η) are both correct.
- One note affects how to measure the h^p scaling. The linear α terms carry the phase e^{−i2πn·q/h}, so their
  average over global mesh offsets is zero while their RMS is not. The h-slope is therefore fitted to the RMS.

## Experiment A: cost at equal actual accuracy

In every accepted run, ‖Γ_h − Γ‖₂/‖Γ‖₂ ≤ tol holds in both dense configurations, A (N = 30, L = 6) and
B (N = 40, L = 10). The searched parameters are:

- s = s₀′·{1, 1.025, …, 1.1}, with s₀′ = √log(2/tol)
- p ∈ 3…8
- η_N ∈ {0.8, …, 0.3}
- M = the next 2·3·5-smooth integer ≥ L/h_m. This keeps h ≤ h_m, and it matters because a prime M
  (for example 167) made numpy's FFT about 10× slower.

That is 2030 dense evaluations in total. C (N = 4000, L = 30) is timed only, using parameters calibrated on A and B.

Cheapest accepted action time (ms):

| tol | system | ξ = 0.5 | 0.7 | 1.0 | 1.4 | 2.0 | cheapest ξ |
|---|---|---|---|---|---|---|---|
| 1e-5 | C | 164 | **129** | 270 | 663 | 1999 | 0.7 |
| 1e-7 | C | 332 | **311** | 532 | 1366 | 8145 | 0.7 |
| 1e-9 | C | **592** | 739 | 2791 | 12704 | (M > 300, not timed) | 0.5 |
| 1e-5 | B | 5.7 | **2.4** | 5.2 | 11.2 | 34.5 | 0.7 |
| 1e-7 | B | **7.0** | 8.4 | 9.6 | 23.3 | 119 | 0.5 |
| 1e-9 | B | **9.6** | 14.1 | 35.0 | 166 | 534 | 0.5 |
| 1e-5 | A | 7.6 | 3.8 | 4.2 | **3.3** | 8.5 | 1.4 (within timing noise of 0.7) |
| 1e-7 | A | 8.5 | **4.7** | 5.8 | 9.6 | 20.5 | 0.7 |
| 1e-9 | A | **10.2** | 11.7 | 11.8 | 25.4 | 82.6 | 0.5 |

All 44 accepted runs are PSD and certified. The accepted s factors lie between 1.0 and 1.075 × √log(2/tol).
Full per-run tables are in `equal_accuracy_xi_results/accepted_runs.md`.

## Experiment B: exact mesh-error decomposition and h^p

Setup: ξ = 1, fixed kc ∈ {4, 6}, M ∈ {12, …, 64} with η_actual < 1, p = 2…8, configurations A and B.
There are 203 fixed-configuration meshes plus 64 random global mesh offsets each (13 195 mesh operators).

- **Consistency of the implementations.** The literal spread/FFT/G/IFFT/gather matches the mode-space matrix to
  1.4e-15. K_h − K_F matches the D_h + α reconstruction to 1.8e-15 of max|K_F|. The truncated Poisson sum over n
  matches the exact stencil DFT to 3e-5 of |S|; p = 2 converges slowest.
- **D_h is roundoff.** max |D_h|/|K̂_L| = 1.8e-16, its operator contribution is ≤ 4.7e-17, and no influence mode
  was excluded. **The complete off-grid mesh error is alias-induced through the spread/gather cross terms.**
- **The linear term carries the error.** The linear part α_i* + α_j is 0.92–1.03 of the total. The quadratic
  part is 2.5e-11 to 0.09 of the total.
- **Scaling.** Fitted RMS slopes on the finest four meshes are p to p + 1.5 (A: 2.02…8.76, B: 1.96…9.55).
  The excess over p is pre-asymptotic. The dominant alias ratio along one axis is exactly (x/(π−x))^p with
  x = kh/2, so its local slope is p·π/(π−x), which tends to p as h → 0.
- **Asymptotic range.** The local slope is within 20% of p for h ≈ 0.125–0.375 (ξh ≈ 0.1–0.4, config A) and
  h ≈ 0.21–0.5 (config B); the range is narrower for p ≥ 7.
- **C_M = ε/(ξh)^p.** It is stable to a factor of 1.04–1.4 on the finest three meshes in A, and to 1.1–2.1 in B,
  where p ≥ 7 is still drifting. Apparent asymptotic values:

  | p | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|---|
  | C_M range | 0.021–0.035 | 0.0059–0.011 | 0.0027–0.0061 | 0.0011–0.0030 | 5.9e-4–2.1e-3 | 2.8e-4–1.3e-3 | 1.6e-4–1.1e-3 |

  C_M depends on the configuration (B/A about 1.5–3×) and on kc (up to 2.3× between kc = 4 and 6).
- **Predictors against the 181 earlier parameter-selection mesh runs (predicted/actual):**
  - C_M(ξh)^p: median 0.93, range 0.25–2.1, rms |log10| 0.14.
  - C_M(η_N π/(2s))^p: median 0.98, range 0.31–2.2.
  - q^p: median 1.9e3, range 37–4e5, rms |log10| 3.6.
- **What the old labels measured**, identified from the code:
  - `alias_TI` summed only the n = n′ ≠ 0 terms of Σ_n Ŵ(k_n)²e^{ik_n·r}. That is exactly the n = n′ part of the
    quadratic term α_i*α_j, which is O(|Ŵ(k_n)/Ŵ(k)|²) ~ h^{2p}.
  - "transfer" was defined as mesh − K_F − alias_TI. That is the whole linear term α_i* + α_j (the n = 0,
    n′ ≠ 0 pairs, which are O(h^p)) plus the n ≠ n′ quadratic terms. The two agree to 8e-5, a difference that
    comes from truncating the old sum at |n| ≤ 40.
  - So the 22×–2e5× gap was a first-order versus second-order alias gap, roughly (Ŵ(k_n)/Ŵ(k))⁻¹ ~ (kh)⁻ᵖ.
    It was not a separate "transfer" physics.
- **PSD.** Classes over all 13 195 mesh operators: 9882 certified (A), 3313 PSD but uncertified (B),
  0 indefinite (C). λ_h/λ_* ≥ 0.970. The bound λ_h ≥ λ_* − ‖Γ_h − Γ‖₂ held everywhere. The uncertified runs are
  all in config B: 3185 at kc = 4, where the fixed Fourier cutoff alone gives ε_F = 7.2e-3, above
  λ_*/‖Γ‖ = 4.4e-3 whatever the mesh; the other 128 at kc = 6 with p = 2 on the three coarsest meshes (M = 20, 24, 32).

## Answers

1. **Cheapest ξ at equal actual accuracy.** On the large system C, ξ = 0.7 is cheapest at tol 1e-5 and 1e-7,
   and ξ = 0.5 at tol 1e-9. On the small systems it is 0.5–0.7; the one exception, A at 1e-5, picks 1.4 by
   0.5 ms, which is within timing noise.
2. **Yes, the 0.5–0.7 conclusion survives.** Cost rises steeply above ξ = 1. At 1e-7 on C, ξ = 1, 1.4 and 2
   cost 1.7×, 4.4× and 26× the minimum. The ξ = 0.5 versus 0.7 difference is only 5–30%, and the optimum
   shifts toward 0.5 as the tolerance tightens.
3. **Yes, the complete mesh error scales as h^p.** Fitted slopes are p to p + 1.5, and the excess is the known
   pre-asymptotic factor π/(π − kh/2).
4. **Asymptotic range:** ξh ≈ 0.1–0.4 for p ≤ 6, and a narrower window for p = 7–8.
5. **C_M is nearly stable:** within a factor of 1.04–1.4 in A and up to about 2 in B for p ≥ 7. It depends mildly
   on configuration and kc (about 3×).
6. **Yes, D_h is roundoff:** ≤ 1.8e-16 relative and ≤ 4.7e-17 in the operator.
7. **Yes, the old "transfer" is the full alias cross-term response:** the linear α terms plus the n ≠ n′
   quadratic terms.
8. **The old alias diagnostic kept only second-order terms.** It kept only the translation-invariant n = n′ ≠ 0
   part of the second-order term, which scales as h^{2p}, and dropped the first-order O(h^p) cross terms.
9. **Yes, C_M(ξh)^p is far better than q^p.** Its median error is 0.93× against 1900× for q^p (rms |log10| 0.14
   against 3.6). q^p uses the edge of the retained sphere, |k| = kc, but the error is weighted by K̂_L(k),
   which is concentrated at small k where the alias ratio is much smaller than q.
10. **Yes, raw Γ_h stayed PSD throughout:** no indefinite operator among 13 195 mesh tests and 44 accepted runs.
11. **The certificate is very conservative.** The actual gap shift is a median 2.4% (at most 7%) of ‖Γ_h − Γ‖₂.
    It fails to certify only when ‖Γ_h − Γ‖ exceeds λ_*, which in these tests was driven by the Fourier cutoff.
12. **Recommended rule** (replacing q^p):
    - s = √log(2/tol), r_c = s/ξ, k_c = 2ξs; the accepted runs needed at most 1.075·s.
    - ξ ≈ 0.5–0.7 per unit κ at this density: 0.7 for tol ≥ 1e-7, 0.5 for tighter tolerances.
    - Choose (p, h) so that C_M(p)(ξh)^p ≤ tol/2, using the conservative (largest) C_M from the table above:
      h = (tol/(2 C_M))^{1/p}/ξ, with η_actual = k_c h/π ≤ 0.8.
    - Prefer large p (6–8) for tol ≤ 1e-7, because spreading is cheap compared with the FFT.
    - Round M up to an FFT-friendly size.
    - Optionally certify PSD with ‖Γ_h − Γ‖ < λ_*. Stabilisation was never needed.
