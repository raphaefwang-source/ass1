# Method A: joint cost-aware error-budget allocation and the Hurwitz-zeta alias bound

This round extends the clustered-particle benchmark of `benchmark_report.md` (same branch, configurations, references,
kernel, split, PPPM solver and k = 0 treatment). Its outputs are kept separate from the previous ones:
`results_joint/`, `figures_joint/`, `logs_joint/`. Nothing in `results/`, `figures/` or the manuscript was changed.

The VM was rebooted before this round, so every timing compared here, the baseline included, was measured again in
this session. The machine is the same Intel Xeon @ 2.10 GHz with 1 thread.

## 0. Results in brief (target: certified relative operator error τ = 2e-7, p = 7)

| | A: fixed 33 % split, original C(p) bound | B: joint, original bound | C: joint, Hurwitz-zeta bound | A_hz: fixed split, Hurwitz-zeta bound (diagnostic) |
|---|---|---|---|---|
| predicted ξ_opt, N = 256 | 0.555 – 0.570 | 0.575 – 0.595 | **0.710 – 0.730** | 0.655 – 0.730 |
| M at the optimum, N = 256 | 48 | 45 | **40** | 40 – 45 |
| measured T_action, N = 256 [ms] | 18.9 – 22.0 | 16.9 – 19.5 | **10.9 – 11.9** | 11.7 – 14.0 |
| measured speedup over A (N = 64 / 256 / 512) | 1 | 1.10–1.22 / 1.05–1.30 / 1.04–1.23 | **1.66–1.91 / 1.70–2.01 / 1.88–1.96** | 1.43–1.74 / 1.49–1.70 / 1.63–1.70 |
| measured E_op at the optimum (all N) | 2.96 – 3.20e-8 | 1.84 – 2.46e-8 | 1.76 – 2.56e-8 | 2.97 – 3.77e-8 |
| measured speedup over the existing baseline (N = 64 / 256 / 512) | 0.38–0.42 / 0.29–0.31 / 0.23–0.26 | 0.42–0.48 / 0.31–0.37 / 0.27–0.30 | **0.65–0.79 / 0.51–0.58 / 0.45–0.49** | 0.57–0.71 / 0.45–0.53 / 0.39–0.43 |

- **Overall gain.** Joint allocation with the improved bound (C) makes the certified Method A **1.7–2.0× faster** than the
  fixed split (A).
  - The improved bound alone (A_hz) gives 1.4–1.7×.
  - Joint allocation alone (B) gives 1.05–1.3×.
  - The two effects compound.
- **Against the existing baseline.** The existing validated baseline (ξ = 0.85, s = 4.1, η = 0.6779, **p = 8**; 1.5–1.8 /
  5.7–6.7 / 12.0–12.3 ms at N = 64 / 256 / 512) is still **1.3–2.2× faster** than C. Its accuracy is established only by
  measurement (E_op 5.0–6.9e-8 here), not certified. The remaining gap is the conservatism of the a-priori Fourier and
  mesh bounds (§4).
- **Accuracy.** Every one of the 306 measured parameter sets meets 2e-7 against the dense full-periodic reference (maximum
  8.6e-8).
- **Cost model.** At N = 256 the predicted optimum agrees with the measured time within −8…+8 %. The predicted ξ_opt and M
  of C are those of the measured fastest point in all 7 configurations (§5).
- **Clustering.** It does not change the optimum: ξ_opt(C) = 0.720 / 0.725 / 0.730 / 0.710 and M = 40 for uniform / moderate /
  strong / LJ-equilibrated at N = 256.

## 1. The joint problem and how it is solved (`joint_optimizer.py`)

**Problem.**

  min over (ξ, k_c, M, r_c) of T_pred = C_R N_img(r_c) + C_P N p³ + C_F M³ log M³ + C_0,
  subject to ε_S(r_c) + ε_F(k_c) + ε_mesh(k_c, M) + ε_quad(k_c, M) ≤ τ Λ_lb for **every** configuration of the level.

Here p = 7 and Λ_lb = max_i λ_max(Γ_ref,ii) ≤ ‖Γ_ref‖₂, so dividing by Λ_lb never understates the relative bound. The
bounds are exactly those of the previous round. Only the allocation, and in C/A_hz the mesh bound, change.

**Algorithm.** For every candidate (ξ, k_c, M):

1. **Fixed bounds.** Compute ε_F(k_c) (reciprocal-lattice tail, a priori), ε_mesh(k_c, M) (a priori alias bound) and
   ε_quad(k_c, M). ε_quad is the production-vs-refined Gauss–Legendre difference of A and B, propagated through
   2ρ Σ (1+α)² ‖δK̂_L‖. It is a **numerical estimate, not a certificate**, and is about 1e-8 of the budget.
2. **Remaining budget.** B = τ Λ_lb − ε_F − ε_mesh − ε_quad, separately for each configuration.
3. **Real-space cutoff.**
   - r_c is the smallest radius with the certified, configuration-specific ε_S(r_c) = 2 max_i λ_max(Σ_{discarded
     images} K_S) ≤ B.
   - ε_S(r_c) is tabulated on 700 points of a = ξ r_c ∈ [3.0, 5.5] as a non-increasing upper envelope.
   - The selected r_c is then refined by bisection on the exact bound to 1e-7 relative.
   - The level uses the largest r_c over its configurations.
4. **Cost.** T_pred uses the exact full-periodic image count N_img(r_c) (i < j, all lattice shifts; the KD-tree enumeration
   validated in the previous round) of each configuration, averaged over the level.
5. **Selection.** The candidate with the smallest T_pred wins.

**Search space.**
- k_c runs over reciprocal-lattice shell radii, so the discrete cutoff is exact.
- M runs over 2,3,5-smooth sizes ≤ 400 with all retained modes below Nyquist (m_c < M/2, η < 1, the constraints of the
  existing `Mesh`/`make_mesh`).
- Shells and meshes whose ε is already below 1e-4 τ Λ_lb end the search along that axis.
- About 500–860 feasible (k_c, M) pairs are scored per ξ at N = 256.
- ξ runs over 0.40–1.60 in steps of 0.025, then 0.005 within ±0.025 of each approach's best.

**Fixed split (A, A_hz).** It reuses the same tables with ε_S, ε_F, ε_mesh ≤ 0.33 τ Λ_lb each. A reproduces the previous
round's parameters exactly at every level (for example moderate N = 256: ξ 0.570, k_c 5.165, M 48, r_c 7.286), even with
this session's recalibrated cost model.

**Cost model** (recalibrated in this session, `results_joint/cost_model.json`):
- C_R = 66.5 ns per image interaction;
- C_P = 15.5 ns per N p³;
- C_F = 6.17 ns per M³ ln M³;
- C_0 = 0.38 ms.

The 12 full-action calibration points are reproduced within −11 … +10 %.

## 2. The mode-dependent Hurwitz-zeta alias bound and its validation (`test_alias_bound.py`)

This conversation did not spell out the improved bound, so this is the exact form implemented. Check that it matches the
one you had in mind.

**Per axis.** With h = L/M and x = k_a h/(2π) = m_a/M (|x| < 1/2 for retained modes), the existing order-p B-spline
stencil gives, through the Poisson sum,

  s(k_a) = Ŵ(k_a) e^{−ik_a q} (1 + β),  |β| ≤ Σ_{m≠0} |Ŵ(k_a + 2πm/h)/Ŵ(k_a)| = Σ_{m≠0} |x/(x+m)|^p
          = |x|^p [ζ(p, 1+|x|) + ζ(p, 1−|x|)] =: σ_p(x)   (Hurwitz zeta; an identity for the sum, no inequality used).

**3D, tensor-product stencil.**
- |α_j(k)| = |Π_a(1+β_a) − 1| ≤ **α_HZ(k) = Π_a(1 + σ_p(m_a/M)) − 1**.
- The original bound used α(k) = C(p) max_a (x_a/(1−x_a))^p with C(p) = (1+2ζ(p))³ − 1 = 26.45 for p = 7.
- In both cases ε_mesh = 2ρ Σ_{|k|≤k_c} α(2+α) ‖K̂_L(k)‖₂.

**Validation on the existing code** (`results_joint/alias_bound_validation.json`; all checks pass):

| check | result |
|---|---|
| FT of `bspline` = sinc^p (p = 2…8, dense quadrature) | max abs. deviation 2.3e-12 (p = 3…8), 2.1e-10 (p = 2, kinked spline) |
| stencil covers the whole support | partition of unity to 4.4e-16; B-spline exactly 0 just outside the stencil |
| Poisson representation vs `Mesh.stencil_dft` (explicit series, \|m'\| ≤ 2000) | 2.1e-13 |
| 1D bound \|β\| ≤ σ_p, all particles of 5 configurations, M = 12…64, all retained modes | max \|β\|/σ_p = 0.99949 (≤ 1: valid) |
| sharpness: sup over the particle offset of \|β\|/σ_p | 0.9992–0.9995: the bound is attained, so it is the tightest of this per-axis form |
| 3D bound from the actual 3D stencil DFT | max \|α_j\|/α_HZ = 0.99934 |
| α_HZ / α_orig over all retained modes | 0.038 – 0.20 |
| influence function divides by exactly Ŵ² (`Mesh.W2`) | 8.6e-16, so the deconvolution residual D_h = 0 |
| Hurwitz ζ (`scipy.special.zeta`) vs direct summation | ≤ 1.1e-15 relative |
| operator level: measured ‖Γ_L^mesh − Γ_L^{k_c}‖₂ ≤ HZ bound ≤ original bound (3 configurations, M from η ≈ 1 to 0.13) | always; HZ/measured = 115–874, original/measured = 2317–17026, original/HZ = 14.6–24.4 |

**Where the gain comes from.**
- Per axis, σ_p(x) is at most 2× tighter than 2ζ(p)(x/(1−x))^p (ratio → 1 for small x, 0.50 at x = 1/2).
- The 15–24× operator-level gain comes from replacing C(7) = 26.45 × max_a q_a^p by the product: for a mode with one
  dominant axis, Π_a(1+σ_a) − 1 ≈ σ_p(x_a) ≈ 2 q_a^p.
- The remaining 100–900× is the N-coherent mode sum (‖e_k e_k^H‖ = N for every mode), which this bound does not address.

Figure J1 shows the per-axis bound with the sharpness points (a), and the operator-level bounds against the measured mesh
error (b).

## 3. Selected parameters and measured performance

### 3.1 N = 256, every configuration

Columns:
- "shares" = certified ε_S / ε_F / ε_mesh as fractions of τ · min Λ_lb of the level;
- "pred / meas T" = predicted and measured cached-action median;
- "measured fastest" = the fastest of all timed candidates of that approach (ξ / M: ms (number timed));
- speedups are measured, against A and against the existing baseline.

| config | appr. | ξ | a | b | r_c | k_c | M (η) | image interactions | shares S/F/mesh | E_op | pred T [ms] | meas T [ms] | measured fastest | ×A | ×baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform_N256_s1 | A | 0.560 | 4.172 | 4.541 | 7.451 | 5.086 | 48 (0.236) | 165548 | 0.33/0.2968/0.31 | 2.96e-08 | 20.67 | 20.33 | 0.560/48: 20.33 (14) | 1.00 | 0.30 |
| uniform_N256_s1 | B | 0.585 | 4.221 | 5.434 | 7.216 | 6.357 | 45 (0.314) | 150377 | 0.23/0.0000/0.78 | 2.02e-08 | 18.16 | 18.65 | 0.585/45: 17.88 (22) | 1.09 | 0.33 |
| uniform_N256_s1 | **C** | 0.720 | 4.230 | 5.371 | 5.874 | 7.734 | 40 (0.430) | 81300 | 0.19/0.0001/0.81 | 1.76e-08 | 11.52 | 11.47 | 0.720/40: 11.18 (23) | **1.77** | 0.53 |
| uniform_N256_s2 | A | 0.560 | 4.172 | 4.541 | 7.451 | 5.086 | 48 | 165642 | 0.33/0.2968/0.31 | 3.05e-08 | 20.68 | 21.63 | 0.560/48: 21.63 (1) | 1.00 | 0.29 |
| uniform_N256_s2 | B | 0.585 | 4.221 | 5.434 | 7.216 | 6.357 | 45 | 150513 | 0.23/0.0000/0.78 | 2.03e-08 | 18.17 | 19.03 | 0.585/45: 18.21 (10) | 1.14 | 0.32 |
| uniform_N256_s2 | **C** | 0.720 | 4.230 | 5.371 | 5.874 | 7.734 | 40 | 81148 | 0.19/0.0001/0.81 | 1.81e-08 | 11.51 | 11.32 | 0.720/40: 11.32 (10) | **1.91** | 0.55 |
| moderate_N256_s1 | A | 0.570 | 4.153 | 4.530 | 7.286 | 5.165 | 48 (0.239) | 153930 | 0.33/0.2833/0.32 | 3.16e-08 | 19.90 | 19.82 | 0.570/48: 19.82 (1) | 1.00 | 0.31 |
| moderate_N256_s1 | B | 0.590 | 4.186 | 5.111 | 7.095 | 6.031 | 45 (0.298) | 142091 | 0.26/0.0008/0.74 | 2.46e-08 | 17.61 | 17.72 | 0.590/45: 17.65 (10) | 1.12 | 0.35 |
| moderate_N256_s1 | **C** | 0.725 | 4.182 | 5.405 | 5.768 | 7.838 | 40 (0.436) | 76546 | 0.26/0.0001/0.76 | 2.38e-08 | 11.20 | 11.24 | 0.725/40: 10.81 (10) | **1.76** | 0.55 |
| moderate_N256_s2 | A | 0.570 | 4.153 | 4.530 | 7.286 | 5.165 | 48 | 154198 | 0.33/0.2833/0.32 | 3.20e-08 | 19.92 | 19.71 | 0.570/48: 19.71 (1) | 1.00 | 0.31 |
| moderate_N256_s2 | B | 0.590 | 4.186 | 5.111 | 7.095 | 6.031 | 45 | 142411 | 0.26/0.0008/0.74 | 2.41e-08 | 17.63 | 17.92 | 0.590/45: 17.70 (10) | 1.10 | 0.34 |
| moderate_N256_s2 | **C** | 0.725 | 4.182 | 5.405 | 5.768 | 7.838 | 40 | 76443 | 0.26/0.0001/0.76 | 2.52e-08 | 11.19 | 11.22 | 0.725/40: 10.48 (10) | **1.76** | 0.54 |
| strong_N256_s1 | A | 0.570 | 4.147 | 4.530 | 7.275 | 5.165 | 48 (0.239) | 157292 | 0.33/0.2690/0.31 | 3.09e-08 | 20.12 | 18.92 | 0.570/48: 18.92 (14) | 1.00 | 0.30 |
| strong_N256_s1 | B | 0.595 | 4.182 | 5.342 | 7.029 | 6.357 | 45 (0.314) | 142033 | 0.24/0.0001/0.77 | 2.30e-08 | 17.61 | 17.96 | 0.595/45: 17.25 (19) | 1.05 | 0.31 |
| strong_N256_s1 | **C** | 0.730 | 4.201 | 5.297 | 5.754 | 7.734 | 40 (0.430) | 75192 | 0.24/0.0003/0.77 | 2.26e-08 | 11.11 | 11.12 | 0.730/40: 10.32 (23) | **1.70** | 0.51 |
| strong_N256_s2 | A | 0.570 | 4.147 | 4.530 | 7.275 | 5.165 | 48 | 157265 | 0.33/0.2690/0.31 | 3.11e-08 | 20.12 | 21.93 | 0.570/48: 21.93 (1) | 1.00 | 0.29 |
| strong_N256_s2 | B | 0.595 | 4.182 | 5.342 | 7.029 | 6.357 | 45 | 142043 | 0.24/0.0001/0.77 | 2.29e-08 | 17.61 | 16.92 | 0.595/45: 16.92 (10) | 1.30 | 0.37 |
| strong_N256_s2 | **C** | 0.730 | 4.201 | 5.297 | 5.754 | 7.734 | 40 | 75373 | 0.24/0.0003/0.77 | 2.18e-08 | 11.12 | 10.93 | 0.730/40: 10.89 (10) | **2.01** | 0.58 |
| lj_equilibrated_N256 | A | 0.555 | 4.182 | 4.436 | 7.536 | 4.924 | 48 (0.228) | 171221 | 0.33/0.2818/0.32 | 3.06e-08 | 21.05 | 21.98 | 0.555/48: 21.98 (1) | 1.00 | 0.31 |
| lj_equilibrated_N256 | B | 0.575 | 4.210 | 5.067 | 7.322 | 5.827 | 45 (0.288) | 157169 | 0.26/0.0016/0.74 | 2.40e-08 | 18.61 | 19.50 | 0.525/40: 19.20 (10) | 1.13 | 0.35 |
| lj_equilibrated_N256 | **C** | 0.710 | 4.245 | 5.297 | 5.979 | 7.522 | 40 (0.418) | 85726 | 0.20/0.0003/0.80 | 1.84e-08 | 11.81 | 11.85 | 0.710/40: 11.27 (10) | **1.85** | 0.57 |

**Existing baseline, measured in this session** (ξ 0.85, r_c 4.824, k_c 6.970, M 24, p 8):

| config | T [ms] | E_op |
|---|---|---|
| uniform_N256_s1 | 6.13 | 6.31e-8 |
| uniform_N256_s2 | 6.18 | 6.15e-8 |
| moderate_N256_s1 | 6.20 | 5.54e-8 |
| moderate_N256_s2 | 6.10 | 5.68e-8 |
| strong_N256_s1 | 5.65 | 5.19e-8 |
| strong_N256_s2 | 6.31 | 5.16e-8 |
| lj_equilibrated_N256 | 6.73 | 6.88e-8 |

The A_hz rows are in `results_joint/joint_parameter_results.csv`:
- ξ 0.655–0.730, M 40–45;
- 11.7–14.0 ms;
- 1.49–1.70 × A.

### 3.2 N = 64 and N = 512 (seed 1 shown; all rows in `results_joint/joint_parameter_results.csv`)

| config | appr. | ξ | r_c | k_c | M | image interactions | E_op | pred / meas T [ms] | ×A | ×baseline |
|---|---|---|---|---|---|---|---|---|---|---|
| strong_N64_s1 | A | 0.600 | 6.958 | 5.340 | 27 | 33238 | 3.04e-8 | 4.13 / 4.14 | 1.00 | 0.42 |
| strong_N64_s1 | B | 0.650 | 6.476 | 6.694 | 27 | 26659 | 2.19e-8 | 3.70 / 3.61 | 1.15 | 0.48 |
| strong_N64_s1 | **C** | 0.800 | 5.261 | 8.322 | 24 | 14436 | 2.07e-8 | 2.50 / 2.20 | **1.88** | 0.79 |
| uniform_N64_s1 | **C** | 0.790 | 5.332 | 8.443 | 24 | 15038 | 2.07e-8 | 2.54 / 2.35 | 1.73 | 0.65 |
| moderate_N64_s1 | **C** | 0.800 | 5.279 | 8.443 | 24 | 14610 | 1.88e-8 | 2.51 / 2.26 | 1.91 | 0.72 |
| strong_N512_s1 | A | 0.535 | 7.764 | 4.840 | 60 | 377030 | 3.04e-8 | 44.54 / 45.81 | 1.00 | 0.26 |
| strong_N512_s1 | B | 0.585 | 7.198 | 6.262 | 60 | 294220 | 2.01e-8 | 39.03 / 43.84 | 1.04 | 0.27 |
| strong_N512_s1 | **C** | 0.680 | 6.152 | 7.347 | 50 | 178548 | 2.56e-8 | 24.02 / 24.35 | **1.88** | 0.49 |
| uniform_N512_s1 | **C** | 0.670 | 6.300 | 7.064 | 50 | 200678 | 2.14e-8 | 25.49 / 26.82 | 1.96 | 0.45 |
| moderate_N512_s1 | **C** | 0.680 | 6.163 | 7.347 | 50 | 187053 | 2.45e-8 | 24.59 / 25.23 | 1.95 | 0.49 |

The baseline here is 1.53–1.75 ms at N = 64 (M = 15) and 12.0–12.3 ms at N = 512 (M = 30).

## 4. Where the budget goes, and why the joint optimum looks the way it does

**The joint optimum raises k_c far above the fixed-split value** (b = k_c/(2ξ) = 5.07–5.43 instead of 4.44–4.54).
- With the deconvolved influence function, k_c only selects which modes enter G_h. The FFT cost depends on M alone, and
  k_c does not appear in T_pred except through m_c < M/2.
- ε_F can therefore be driven to ≤ 0.2 % of the budget at no cost, as long as the extra high-|k| modes do not inflate the
  mesh bound much. They do not: ‖K̂_L‖ is ~e^{−b²} there.
- The freed budget goes to the mesh term (74–81 %) and the real-space term (18–26 %). The fixed split forces 27–30 % onto a
  term that costs nothing to remove (Figure J2a).

**Optimum is a plateau in k_c.**
- The best (k_c, M) alternatives at ξ_opt share M and r_c (N_img within 0.1 %) and differ only in k_c, so they are the
  same-cost operator.
- They do identical work by construction: the same M, the same r_c and image list, and k_c only changes which entries of
  the fixed-size M³ influence grid are non-zero.
- Their measured times still differ by 0–8 %. This is run-to-run variation between separately timed operators, and it can
  exceed the within-run IQR (e.g. strong_N256_s1 C: optimum 11.12 ms [q1 10.91, q3 11.40], alternative 10.32 ms
  [10.14, 10.47]).
- This is why the "measured fastest" C point is sometimes an alternative. It is not a cheaper operator.

**Measured vs certified components** at the optima (N ≤ 256, divided by ‖Γ_ref‖₂; `results_joint/errors.json`):

| | measured real space | measured Fourier | measured mesh | real-space bound / measured | mesh bound / measured (HZ; original) |
|---|---|---|---|---|---|
| A | 2.96 – 3.21e-8 | 7.3e-11 – 1.3e-9 | 1.3 – 4.7e-11 | 1.75 – 1.90 | 70 – 252; 1192 – 4234 |
| B | 1.99 – 2.46e-8 | 5.0e-14 – 2.5e-12 | 2.7 – 10e-11 | 1.75 – 1.91 | 76 – 277; 1349 – 4757 |
| C | 1.76 – 2.51e-8 | 8.4e-14 – 6.0e-13 | 0.35 – 1.2e-9 | 1.73 – 1.88 | 115 – 398; 2259 – 7460 |

- **Measured E_op** is dominated by the real-space truncation for every approach: E_op ≈ measured ε_S.
- **The certified real-space bound is tight** (≈ 1.8×).
- **Why C is still slower than the baseline.** Even with the improved bound, the certified mesh term over-estimates by
  10²–4·10². The certified mesh (M = 40 at ξ ≈ 0.72) is therefore finer than the measured accuracy needs. In the previous
  round's empirical diagnostic, M = 24 sufficed at ξ = 0.7. This is what still keeps C slower than the uncertified baseline.

## 5. Accuracy of the predicted optimum

- **Prediction at the optima:**

  | N | predicted / measured − 1 |
  |---|---|
  | 64 | −4 … +18 % (36 optima) |
  | 256 | −8 … +8 % (28 optima) |
  | 512 | −12 … −1 % (12 optima) |

  Over all 227 timed N = 256 candidates the median is −2 %. The extremes (−45 … −56 %) are A's large-mesh scan points
  (ξ = 1.5–1.6, M = 180–192), as in the previous round.
- **N = 256, approach C:** in every configuration the measured fastest candidate has the predicted ξ_opt and M = 40, and
  differs only in k_c (same-cost plateau, §4). Ten to 23 candidates were timed per configuration, including ξ_opt ±0.05,
  ±0.1, ±0.2 and, for strong_N256_s1 and uniform_N256_s1, a full ξ scan in steps of 0.1 (Figure J3).
- **N = 256, approach B:** the same holds except LJ-equilibrated. There the measured fastest is ξ = 0.525 / M = 40 at
  19.20 ms against 19.50 ms at the predicted ξ = 0.575, a 1.5 % gap that is within noise.
- **N = 256, approach A:** on the two primary configurations the predicted optimum was the fastest of 14 timed ξ.
- **N = 512, C:** measured 24.4–26.8 ms against predicted 24.0–25.5 ms. The ξ ± 0.1 neighbours were slower.

## 6. Validation against the dense full-periodic reference (`results_joint/errors.json`, `joint_errors.csv`)

- **Coverage:** 306 parameter sets, i.e. every timed point: all candidates of every approach plus the baseline, on all 19
  configurations.
- **Accuracy:** E_op ≤ 8.6e-8 for all 306, so **all ≤ 2e-7**.
- **Certified total at the selected optima:** ≤ τ for every level. A uses 1.31–1.88e-7 of the 2e-7 budget; B and C use
  1.989–2.000e-7.
- **Structure:**
  - symmetry ≤ 2.5e-17;
  - translation null space ≤ 1.6e-15;
  - λ_min(Γ_h) on Ran Π equal to λ_min(Γ_ref) within 1.1e-6 (Weyl-consistent with ‖ΔΓ‖);
  - no PSD stabilisation.
- **Cached action vs independent dense assembly:** ≤ 1.3e-15 on three X per set.
- **Basis-vector assembly** of the B and C optima (32 sets, N ≤ 256) agrees with the independent mode-space assembly
  to ≤ 1.8e-15.

## 7. Figures (`figures_joint/`)

- **J1** `figJ1_alias_bound.png`: per-axis Hurwitz-zeta vs original bound, with the measured sharpness (a); operator-level
  bounds vs measured mesh error (b).
- **J2** `figJ2_budget_allocation.png`: certified shares of ε_S / ε_F / ε_mesh at each approach's optimum (a); measured
  components and E_op (b).
- **J3** `figJ3_time_vs_xi.png`: predicted (best k_c, M, r_c per ξ) and measured action time against ξ for A, A_hz, B and C,
  with the measured baseline; strong_N256_s1 and uniform_N256_s1.
- **J4** `figJ4_speedup_error.png`: measured speedup over the baseline and measured E_op at each approach's optimum, all
  configurations.

## 8. Limitations

- **Certification scope.** ε_S, ε_F and ε_mesh are certified in exact arithmetic for the computed kernel coefficients.
  - ε_quad is a numerical estimate (~1e-8 of the budget).
  - Floating-point round-off (~1e-15) is not bounded.
  - Λ_lb comes from the diagonal blocks of the dense reference.
- **Search space.** p is fixed at 7. M is restricted to 2,3,5-smooth sizes ≤ 400. Contributions below 1e-4 of the budget
  end the k_c and M search.
- **Timing.** One VM, one thread, NumPy implementation.
  - Run-to-run variation between separately timed operators of identical work is up to ~8 %.
  - The cost model under-predicts large meshes (M ≥ 150).
  - All timings are medians with IQR and min/max in `results_joint/joint_timing.csv`.
- **Coverage.** N = 512 has one configuration per level and no measured component decomposition. Components were measured
  at N = 64 and 256.

## 9. Files

New code, all reusing the existing modules unchanged:
- `test_alias_bound.py`: validation of the Hurwitz-zeta bound;
- `joint_optimizer.py`: bounds, ε_S curves, joint and fixed-split selection;
- `run_joint_benchmark.py`: stages calibrate / select / bench / errors;
- `make_figures_joint.py`.

`results_joint/`:

| file | content |
|---|---|
| `alias_bound_validation.json` | Hurwitz-zeta bound validation |
| `cost_model.json`, `calibration_timings.csv` | cost model of this session |
| `joint_selection.json` | every level, approach and ξ: best (k_c, M, r_c) with the certified decomposition; alternatives at ξ_opt |
| `joint_selection_grid.csv` | flat version of the selection |
| `bench.json`, `joint_timing.csv` | every timed point |
| `errors.json`, `joint_errors.csv` | dense validation, including every probe |
| `joint_summary.csv`, `joint_parameter_results.csv` | per configuration and approach: predicted and measured optimal parameters, errors, times, speedups |
| `reproducibility_joint.json` | environment, problem definition, command log, sha256 of every output |
| `commands.log` | time-stamped command log |

Logs are in `logs_joint/`.

## 10. Commands executed (from `benchmarks/method_A_clustered/`; time-stamped log: `results_joint/commands.log`)

```bash
python3 test_alias_bound.py                                    # run twice: the second run after removing a duplicated
                                                               # M from the operator-level list (same results)
python3 run_joint_benchmark.py --stage calibrate               # cost model of this session
python3 run_joint_benchmark.py --stage select                  # 10 levels x 4 approaches, 29 min
python3 run_joint_benchmark.py --stage bench                   # 306 timed points (234 at N = 256), run alone (12 min)
python3 run_joint_benchmark.py --stage errors                  # 306 dense validations (9 min)
python3 make_figures_joint.py                                  # run 3 times (figure layout fixes); last run is final
```

During development a short scratch script also scored three ξ values for the strong N = 256 level with the previous
round's cost model. It produced no stored result.
