# Method A on clustered particles: theory-guided, equal-accuracy benchmark of the cached action Γ_h X

Every number in this report is read from a result file in this directory. Section 10 lists the commands that
produced them. The run was on one Intel Xeon @ 2.10 GHz VM (4 cores) with OMP/OPENBLAS/MKL threads = 1, Python
3.13.16, NumPy 2.5.3 and SciPy 1.18.1.

## 0. Answers

| question | answer (measured unless marked) |
|---|---|
| 1. Theoretically selected ξ_opt | **N = 256:** 0.560 (uniform), 0.570 (moderate), 0.570 (strong), 0.555 (LJ equilibrated).<br>**N = 64:** 0.640 (uniform), 0.600 (moderate, strong).<br>**N = 512:** 0.525 (uniform), 0.535 (moderate, strong).<br>All with p = 7, certified ε ≤ 2e-7, and the bounded 1-D minimisation of the calibrated cost model. |
| 2. Accuracy of the cost model | **At ξ_opt** (predicted / measured):<br>• N = 64: 0.89–1.10;<br>• N = 256: 0.89–0.96;<br>• N = 512: 0.82–0.90.<br>**Location of the optimum:**<br>• N = 256: the predicted ξ_opt is the measured fastest point in 6 of 7 configurations. In the seventh (strong_N256_s2) the measured best ξ = 0.525 is 0.2 % faster, inside the 16 % IQR.<br>• N = 512: the measured best is ξ = 0.55–0.56 against a predicted 0.525–0.535. Running at the predicted ξ costs 3–18 % more time.<br>**Over all 124 certified points:** the model under-predicts large meshes by up to 62 % (M ≥ 144) and by 7–20 % (96 < M < 144), and is within −32…+5 % for M ≤ 96. |
| 3. Does clustering change ξ_opt? | **No, beyond ±0.015 at N = 256** (0.555–0.570 across all four levels), and the measured fastest ξ agrees. At ξ_opt, rc ≈ 7.3 > L = 6.99, so the retained image count follows the mean density. For rc ≥ 3 the exact N_img(rc) of the strongest clustering differs from the homogeneous mean-density estimate by ≤ 5 % at N = 256 (≤ 14 % at N = 512).<br>Clustering does raise ‖Γ_ref‖₂ (13.7 → 15.9 at N = 256) and lowers λ_min on Ran Π (4.37 → 3.46). It does not change the selected cutoffs materially (a_tol 4.15–4.18). |
| 4. Dominant cost | **At the certified optimum:** real space 53–57 % and FFT + influence 35–39 % (spread + gather 7–9 %).<br>**Setup** is 3.1–5.2 × one action. Its largest part is the periodic image enumeration (35–62 ms at N = 256). |
| 5. Are the bounds excessively conservative? | **Real space:** no, the bound is 1.8–1.9 × the measured error.<br>**Fourier:** yes, 32–40 × (N = 64) and 107–136 × (N = 256). The bound is N-coherent, so the factor grows with N.<br>**Mesh, with C(7) = (1 + 2ζ(7))³ − 1 = 26.45:** yes, 1.2–1.4·10³ × (N = 64) and 3.8–4.2·10³ × (N = 256).<br>**Effect on parameters:** the certified mesh is 2.25–3.75 × finer per axis than the empirically sufficient one.<br>**Effect on cost:** the certified optimum is **2.3–4.3 × slower** than the validated baseline (whose accuracy was established empirically). |
| 6. Does the cached Γ_h X stay accurate? | **Yes.** Against the independent dense full-periodic reference, 124 certified sets give E_op = 2.77e-8 … 3.63e-8 (target 2e-7).<br>• Cached action vs the independent dense assembly, on 3 X vectors per set: ≤ 1.4e-15.<br>• Basis-vector assembly vs mode-space assembly: ≤ 2.7e-15.<br>• Symmetry ≤ 2.5e-17; translation null space ≤ 1.7e-15; λ_min on Ran Π within 4.5e-7 of Γ_ref's.<br>• No PSD stabilisation is applied. |
| 7. Speedup over the baseline | **Certified selection:** 0.35–0.44 (N = 64), 0.27–0.31 (N = 256), 0.23–0.25 (N = 512), i.e. slower.<br>**Separate empirical diagnostic** (N = 256, ξ = 1.0, p = 7, M = 30): 1.07–1.20 × faster than the baseline, at measured E_op ≤ 1.14e-7. This uses measured, not certified, error budgets. |
| 8. Effect of setup cost | Including setup moves the optimum to larger ξ (fewer pair images to enumerate).<br>**s = 1:** predicted 0.655–0.705 at N = 256; measured best 0.625–0.705.<br>**s = 10 and 20:** the optimum returns towards the action-only value (0.555–0.625). |

## 1. Repository state and scope decisions

- **Starting point.** No `benchmarks/method_A_clustered/` existed on any branch, so this directory was created
  from scratch. The work branch `claude/intelligent-bell-62t6pg` was reset onto the latest project branch
  `origin/claude/new-session-nmi9ux` (2f41eb2); before that it held only the Q4 notebook.
- **State point** ("the already agreed Method A"): the LJ + kernel A study of `manuscript/main.tex` §5.1 and
  `toy_configs.CONFIGS["lj_rho0.75_kT1.0_A_prod"]`.
  - Kernel K(r) = c e^{−κr}/r · r r^T/r² with γ = 0.5, κ = 0.7, r_ref = 1.3, c = 1.6148.
  - ρ = 0.75; full periodic lattice sum with k = 0 retained; ε_op ≤ 2e-7 in the spectral norm.
  - This is the only parameter set in the repository with a friction amplitude, the 2e-7 budget and LJ minimum
    separations.
  - The Section 6 unit-kernel set (κ = 1, ρ = 30/216) and the toy state point (ρ = 64/5.5³, kT = 0.7) were not
    used or mixed in.
- **Baseline**: ξ = 0.85, s* = 4.1 (rc = 4.824, kc = 6.97), η = 0.6779, **p = 8** (M = 15 / 24 / 30 at
  N = 64 / 256 / 512 via `make_mesh`). The manuscript raised p from 7 to 8 because p = 7 gave 2.04–2.15e-7 on fcc
  states.
- **Reused unchanged:**
  - `radial_kernels.RadialEwald` (split at s = ξ²);
  - `test_parameter_selection.Mesh` (B-spline spread / FFT / K̂_L/|Ŵ|² / IFFT / adjoint gather, mesh degree, mode-space
    `pair_matrix`);
  - `test_equal_accuracy_xi.make_mesh` / `next_smooth`;
  - `test_lanczos_fdt.GammaH` (existing action and its independent dense assembly);
  - `prl_toy_models.LatticeFriction` (reference);
  - `lj075_common.rsa_positions`.

  The kernel, the split, the influence function, the k = 0 treatment and the PPPM solver were not modified.
- **New code:**
  - `neighbor_images.py`: periodic KD-tree candidates plus all lattice shifts;
  - `cached_operator.py`: `CachedGammaH`, D_i X_i − H_i(X), with D_i cached;
  - `clustered_configurations.py`;
  - `parameter_optimizer.py`: bounds, selection and cost model;
  - `error_validation.py`;
  - `run_clustered_benchmark.py`, `make_figures.py`, `test_neighbor_images.py`.
- **Existing clustered states:**
  - `ewald_longitudinal/fft_vs_direct_results/states/*_liquid_clustered.npz` are double-well liquids at the toy
    state point (ρ = 0.385, kT = 0.7). They were excluded because they belong to a different state point.
  - The committed equilibrated LJ + A state `lj075_results/hpc_pilot/init_state_eqA_lgv_s305_t50.npz` (N = 256,
    t = 50 of law-A dynamics, r_min = 0.924) is included as the physically motivated validation set
    "lj_equilibrated".
- **Not done:** no Lanczos, no BAOAB, no alternative splits, and no change to the manuscript or to older result
  files.

## 2. Periodic KD-tree neighbour search and full-periodic image multiplicity

Retained set: I(rc) = {(i, j, n) : i < j, n ∈ ℤ³, |q_i − q_j + nL| ≤ rc}. Self images (i = j) are excluded because they
multiply X_i − X_i = 0.

**Method.**
- `cKDTree(x, boxsize=L).query_pairs(rc)` returns the unique unordered pairs whose **minimum-image** distance is ≤ rc.
  These are only the *candidates*, but the set is complete: if any image of a pair lies within rc, so does its
  minimum image. The candidate radius is inflated by 1e-12 relative, so KD-tree rounding at the cutoff cannot
  drop a pair.
- For every candidate pair, all shifts with |d0 + nL| ≤ rc are enumerated, where d0 = x_i − x_j and x is wrapped to
  [0, L)³. Two shift ranges are implemented:
  - "cube": |n_a| ≤ ceil(rc/L) + 1, the safe range; ceil(rc/L) already suffices.
  - "tight": n_a ∈ [ceil((−rc − d0_a)/L), floor((rc − d0_a)/L)]. For rc < L/2 this reduces to the single
    minimum-image shift, which is provably the only possible image.

**Validation** (`test_neighbor_images.py` → `results/neighbor_validation.{json,csv}`, `logs/neighbor_validation.log`):

**Exact set tests (100):**
- **Coverage:** 9 configurations (uniform and clustered with N = 8, 27, 64, 128; a boundary case with particles on
  the faces, a pair across the boundary and a pair exactly at the cutoff) × rc/L ∈ {0.2, 0.25, 0.35, 0.49, 0.5,
  0.51, 0.7, 0.866, 0.9, 1.2, 1.6, 2.3}.
- **Methods compared with brute force** (all pairs × |n_a| ≤ ceil(rc/L) + 2): KD-tree + tight, KD-tree + cube, and
  the no-tree variant.
- **Result:** identical (i, j, n) sets in all 100 tests, with bit-identical displacements. Candidates were always
  complete and no self image ever appeared.
- **Existing routines:** `test_lanczos_fdt.real_pairs` and `radial_kernels.real_pairs_tree` give the identical set in
  all 100 tests.

**Multiplicity:**
- rc < L/2: at most 1 image per pair (all tests).
- rc > L/2: up to 60 images per pair.

**Short-range action at the LJ + A state point** (RSA configurations with d_min = 0.9):

| N | rc (rc/L) | unique pairs | image interactions | images/pair | KD-tree vs brute force | vs dense brute-force Γ_S^{rc} X | D−H vs pair-difference form | error if only the minimum image is kept |
|---|---|---|---|---|---|---|---|---|
| 64 | 1.5 (0.34) | 313 | 313 | 1.00 | 9.3e-17 | 1.8e-16 | 1.8e-16 | 1.8e-16 |
| 64 | 2.6 (0.59) | 1568 | 1746 | 1.11 | 2.0e-16 | 3.4e-16 | 3.0e-16 | **1.2e-2** |
| 64 | 4.824 (1.10) | 2016 | 11093 | 5.50 | 5.3e-16 | 6.0e-16 | 7.2e-16 | **2.1e-2** |
| 256 | 3.0 (0.43) | 10744 | 10744 | 1.00 | 2.5e-16 | 4.4e-16 | 3.9e-16 | 4.4e-16 |
| 256 | 4.824 (0.69) | 30992 | 44920 | 1.45 | 5.7e-16 | 8.0e-16 | 7.3e-16 | **6.4e-5** |
| 256 | 6.0 (0.86) | 32640 | 86679 | 2.66 | 7.8e-16 | 1.1e-15 | 1.0e-15 | **6.4e-5** |

**Full operator:**
- The cached D_i X_i − H_i(X) operator with KD-tree images reproduces the existing `FastFriction`/`GammaH.matvec`
  (image-enumerated pairs, same mesh) to 2.3e-16 (N = 64) and 2.1e-16 (N = 256).
- The image counts are identical: 11047 and 44975.

**Enumeration time at rc = 4.824:**

| N | KD-tree + tight (this work) | existing `real_pairs` | existing `real_pairs_tree` |
|---|---|---|---|
| 256 | 12.2 ms | 193 ms | 8.4 ms |
| 512 | 19.2 ms | 777 ms | 16.3 ms |

## 3. Configurations (ρ = 0.75, periodic d_min = 0.9)

The same N, L, mean density, kernel and seeds are used throughout. Only the Gaussian-mixture widths and centres
differ.

**Generator.**
- Positions are drawn from the wrapped mixture.
- Overlaps are then removed by overdamped pairwise periodic push-apart. Every pair below 1.01 d_min is pushed apart,
  and each step is capped at 0.02 d_min.
- No pair below d_min is ever accepted.
- If the push-apart failed, the width would be relaxed by ×1.15 and the draw repeated. This was not needed.
- A first version without the step cap threw particles out of dense clouds and produced near-uniform "clusters"
  (σ_realised/σ_req ≈ 3, S(k) like uniform). It was replaced before any benchmark use.

| N = 256, seed 1 | requested width | realised width / requested | d_min (periodic) | neighbours within 1.5: mean ± sd [min, max] | max S(k), small k | images within rc = 4.824 / unique pairs | ‖Γ_ref‖₂ | λ_min(Γ_ref) on Ran Π |
|---|---|---|---|---|---|---|---|---|
| uniform | — | — | 0.9019 | 10.19 ± 2.38 [4, 17] | 1.74 | 45030 / 31047 | 13.67 | 4.369 |
| moderate (3 clouds) | 0.20 L = 1.398 | 1.010 | 0.9015 | 10.96 ± 3.19 [2, 18] | 6.26 | 45003 / 31150 | 15.01 | 4.262 |
| strong (2 clouds) | 0.08 L = 0.559 | **2.18** | 0.9001 | 13.23 ± 3.84 [4, 22] | 15.5 | 43023 / 30462 | 15.89 | 3.462 |
| LJ equilibrated | — | — | 0.9244 | 10.57 ± 1.41 [5, 15] | 0.32 | 44935 / 31003 | 11.92 | 4.658 |

- **The strong width is geometrically infeasible at d_min = 0.9.** The overlap removal widened the clouds to
  σ ≈ 1.22, which is close to two jammed droplets of radius ≈ 2.7 separated by voids.
- **Level separation** at N = 512 is S(k)_max = 2.3 / 17.1 / 31.2 for uniform / moderate / strong.
- **Seeds:** N = 64 seeds 1–3, N = 256 seeds 1–2, N = 512 seed 1.
- **Saved files:** positions, labels, pair-distance histograms (all images to r = 10), neighbour counts and S(k) are
  in `configs/*.npz`; the summary is in `results/configurations.csv` (Figure 1).

**Reference Γ_ref.**
- `LatticeFriction` with far-sum tolerance 1e-13. Against the default 1e-10, Γ_ref changes by 1.2e-11 (N = 64),
  3.8e-12 (N = 256) and 1.8e-12 (N = 512) relative.
- Self-check against a brute-force image sum: ≤ 1.2e-14.
- The dense matrices are cached in `raw/` (git-ignored) and regenerated by `--stage refs`.

## 4. Theory-guided parameter selection (`parameter_optimizer.py`)

**Error decomposition.** Γ_h − Γ_ref = −(K_S images beyond rc) + (Γ_L^{kc} − Γ_L) + (Γ_L^{mesh} − Γ_L^{kc}) + coefficient
errors. The **absolute** spectral-norm bounds below hold in exact arithmetic, given exact A, B and θ_S.

- **ε_S(rc; q)** = 2 max_i λ_max(Σ_{(i,j,n): |d|>rc} K_S(d)). This is configuration-specific, i.e. geometry-aware.
  - The discarded images form a graph Laplacian with PSD edge weights θ_S r̂r̂^T, and
    (v_i−v_j)^T B (v_i−v_j) ≤ 2v_i^T B v_i + 2v_j^T B v_j.
  - Images up to R_far = 7/ξ are summed explicitly.
  - Beyond R_far an analytic bound is added, using θ_S ≤ c e^{−ξ²r²}(e^{−z²}/r + ρ_κ(ξ²)) (from erfc x ≤ e^{−x²}) and
    the image count ≤ N(4π/3)(r + √3L/2)³/L³. It is ≤ 1e-17 here.
- **ε_F(kc)** = 2ρ Σ_{k∈(2π/L)ℤ³, |k|>kc} ‖K̂_L(k)‖₂, with ‖K̂_L‖₂ = max(|A|, |A + Bk²|).
  - This holds because ‖e_k e_k^H‖ = N for the pair matrix and its degree.
  - The sum runs shell by shell over the actual reciprocal lattice, so the discrete cutoff is exact (kc is always a
    shell radius).
  - Beyond k_far = 19.4 ξ the tail is bounded by A(k) ≤ A(0)e^{−k²/4ξ²} and |B(k)| ≤ |B(0)|e^{−k²/4ξ²}.
- **ε_mesh(M, p)** = 2ρ Σ_{|k|≤kc} α(k)(2 + α(k)) ‖K̂_L(k)‖₂, with **α(k) = C(p) q(k)^p**,
  **C(p) = (1+2ζ(p))³ − 1** (C(7) = 26.4534) and q(k) = max_a |k_a|h/(2π − |k_a|h).
  - From the Poisson sum, S_j(k) = Ŵ(k)e^{−ik·q_j}(1 + α_j(k)) with a deconvolved influence K̂_L/Ŵ².
  - Per axis, |Ŵ(k_a + 2πm/h)/Ŵ(k_a)| ≤ q_a^p |m|^{−p}, so |α_j| ≤ Π_a(1 + 2ζ(p)q_a^p) − 1 ≤ C(p) max_a q_a^p.
  - It then follows that ‖u u^H − e e^H‖ ≤ Nα(2+α).
  - No influence mode was excluded (n_excluded = 0 in every set), so the deconvolution residual is zero.
- **ε_quad** is a numerical **estimate**, not a certificate.
  - Production vs refined Gauss–Legendre A, B (the rule of `exp1_operator_action.RefinedEwald`) is propagated
    through 2ρ Σ (1+α)²‖δK̂‖.
  - Measured: 8.7e-16 – 3.0e-15 of ‖Γ_ref‖, against a budget of 2e-9.

**Normalisation.**
- Λ_lb = max_i λ_max(Γ_ref,ii) ≤ ‖Γ_ref‖₂ (a diagonal block of a PSD matrix). Dividing by Λ_lb therefore never
  understates the relative bound.
- The fixed budget split is ε_S : ε_F : ε_mesh : ε_quad = 0.33 : 0.33 : 0.33 : 0.01 of ε Λ_lb.

**Selection for each ξ.**
- a_tol = ξ · min{rc: ε_S ≤ 0.33εΛ_lb} (bisection);
- b_tol = kc/(2ξ), with kc the smallest admissible lattice shell;
- M = the smallest 2·3·5-smooth M with retained modes below Nyquist (m_c < M/2, η = kc h/π < 1, the constraints of
  `Mesh`/`make_mesh`) and ε_mesh ≤ 0.33εΛ_lb;
- p = 7.

**Clustering levels.** The parameters must be admissible for every seed of a level: the largest rc and kc are
taken, and M is re-solved at that kc.

**Optimisation.** The cost model is evaluated on ξ ∈ [0.40, 1.60] in steps of 0.025 (model evaluation only, no
benchmark sweep). The argmin is then refined in steps of 0.005 over ±0.025. Figures 3 and 4 show a_tol, b_tol, M
and the image counts against ξ.

## 5. Geometry-aware cost model (`results/cost_model.json`, `results/calibration_timings.csv`)

**Model.** T_pred = C_R N_img(rc) + C_P N p³ + C_F M³ log(M³) + C_0, where N_img counts each retained (i < j, n)
image once. One pass updates both rows, in both the cached action and the existing action.

**Calibration.**
- The coefficients come from 64 isolated timings on the uniform seed-1 configurations (N = 64, 256, 512):
  - real-space H alone, rc = 2…10;
  - spread + gather, p = 4…8;
  - FFT + influence alone, M = 16…120;
  - 12 full actions, used to fix C_0.
- Each fit is a non-negative least-squares fit in relative error. An earlier plain least-squares fit gave a
  non-physical C_0 < 0 and was replaced.

**Fitted values:**
- C_R = 61.8 ns per image interaction;
- C_P = 14.5 ns per N p³;
- C_F = 5.69 ns per M³ ln M³;
- C_0 = 0.57 ms.

**Fit quality:**
- Component fits: maximum relative error 21 % (real), 36 % (spread + gather) and 20 % (FFT).
- The 12 full-action calibration points are reproduced within −14 … +24 %.

**Setup model** (median of 3 builds per point): S_R = 0.39 µs per image, S_P = 49 ns, S_F = 6.3 ns, S_0 = 0.52 ms.
The median error is 7.5 % and the maximum 52 %.

## 6. Results

### 6.1 Selected parameters and measured performance at ξ_opt (`parameter_results.csv`)

| config | ξ_opt | a_tol | b_tol | rc | kc | M (η) | unique KD-tree pairs | image interactions (ratio) | measured E_op | T_pred [ms] | T_meas [ms] (IQR) | T_setup [ms] | measured fastest ξ | baseline T [ms] / E_op |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform_N64_s1 | 0.640 | 4.168 | 4.172 | 6.512 | 5.340 | 30 (0.249) | 2016 | 27277 (13.5) | 2.96e-8 | 4.14 | 4.67 (47 %) | 22.4 | 0.640 | 1.63 / 5.94e-8 |
| moderate_N64_s1 | 0.600 | 4.164 | 4.450 | 6.941 | 5.340 | 27 (0.277) | 2016 | 33122 (16.4) | 3.18e-8 | 4.04 | 4.05 (5 %) | 16.4 | 0.600 | 1.80 / 5.71e-8 |
| strong_N64_s1 | 0.600 | 4.175 | 4.450 | 6.958 | 5.340 | 27 (0.277) | 2016 | 33239 (16.5) | 3.04e-8 | 4.05 | 3.67 (9 %) | 15.2 | 0.600 | 1.51 / 5.64e-8 |
| uniform_N256_s1 | 0.560 | 4.172 | 4.541 | 7.451 | 5.086 | 48 (0.236) | 32640 | 165549 (5.07) | 2.96e-8 | 19.38 | 21.76 (13 %) | 73.1 | 0.560 | 5.86 / 6.31e-8 |
| uniform_N256_s2 | 0.560 | 4.172 | 4.541 | 7.451 | 5.086 | 48 | 32640 | 165642 (5.07) | 3.05e-8 | 19.38 | 21.20 (9 %) | 76.0 | 0.560 | 6.14 / 6.15e-8 |
| moderate_N256_s1 | 0.570 | 4.153 | 4.530 | 7.286 | 5.165 | 48 (0.239) | 32640 | 153931 (4.72) | 3.16e-8 | 18.66 | 19.54 (8 %) | 102.5 | 0.570 | 5.99 / 5.54e-8 |
| moderate_N256_s2 | 0.570 | 4.153 | 4.530 | 7.286 | 5.165 | 48 | 32640 | 154201 (4.72) | 3.20e-8 | 18.68 | 20.00 (10 %) | 71.8 | 0.570 | 6.01 / 5.68e-8 |
| **strong_N256_s1** | **0.570** | 4.147 | 4.530 | 7.276 | 5.165 | 48 (0.239) | 32640 | 157296 (4.82) | 3.09e-8 | 18.87 | 20.66 (16 %) | 72.6 | **0.570** | 5.57 / 5.19e-8 |
| strong_N256_s2 | 0.570 | 4.147 | 4.530 | 7.276 | 5.165 | 48 | 32640 | 157268 (4.82) | 3.11e-8 | 18.86 | 21.19 (16 %) | 65.1 | 0.525 | 5.99 / 5.16e-8 |
| lj_equilibrated_N256 | 0.555 | 4.182 | 4.436 | 7.536 | 4.924 | 48 (0.228) | 32640 | 171226 (5.25) | 3.06e-8 | 19.73 | 21.23 (9 %) | 84.2 | 0.555 | 5.75 / 6.88e-8 |
| uniform_N512_s1 | 0.525 | 4.163 | 4.559 | 7.929 | 4.787 | 60 (0.224) | 130816 | 400722 (3.06) | 3.06e-8 | 42.97 | 47.64 (4 %) | 180.8 | 0.550 | 11.87 / 5.79e-8 |
| moderate_N512_s1 | 0.535 | 4.145 | 4.523 | 7.748 | 4.840 | 60 (0.226) | 130816 | 369073 (2.82) | 3.13e-8 | 41.01 | 50.05 (18 %) | 192.9 | 0.560 | 12.15 / 5.04e-8 |
| strong_N512_s1 | 0.535 | 4.154 | 4.523 | 7.764 | 4.840 | 60 (0.226) | 130816 | 377038 (2.88) | 3.04e-8 | 41.50 | 50.54 (8 %) | 184.1 | 0.560 | 11.82 / 5.12e-8 |

- **Omitted rows:** N = 64 seeds 2–3 behave like seed 1. All 19 rows are in `parameter_results.csv`.
- **Measured ξ points:**

  | case | ξ values measured |
  |---|---|
  | strong_N256_s1, uniform_N256_s1 (primary) | 19 each (Figure 2) |
  | other N = 256 configurations | 7–9 |
  | N = 512 | 7 |
  | N = 64 | 2–3 |

- **Totals:** 144 timed points; 124 certified sets plus 19 baseline sets were error-validated.

### 6.2 Predicted vs measured time (Figure 2)

- **N = 256, strongly clustered:** the measured times follow the predicted curve over ξ = 0.4–1.6 (Figure 2a). Both
  minima are at ξ = 0.570 (20.66 ms measured, 18.87 ms predicted).
- **Prediction error:**

  | N | median | range | comment |
  |---|---|---|---|
  | 64 | −0.2 % | −12 … +10 % | |
  | 256 | −9 % | −62 … +5 % | −32 … +5 % for M ≤ 96; within ±0.1 of ξ_opt: median |8.5| %, −32 … −4 % |
  | 512 | −13 % | −26 … −1 % | |

- **Under-prediction:** the model under-predicts large FFTs (M ≥ 144) by up to 62 %, consistent with grids leaving
  cache; for 96 < M < 144 the under-prediction is 7–20 %. At N = 512 it under-predicts the per-image real-space cost
  for ~4·10⁵ images. This moves the measured optimum 0.025 above the predicted one (ξ = 0.55–0.56 vs 0.525–0.535). At
  the predicted ξ the cost is 3–18 % above the measured best: strong_N512 is 50.5 ms at 0.535 against 42.7 ms at 0.560.

### 6.3 Operator error against the dense full-periodic reference (Figure 5, `results/errors.json`)

**Certified sets (124; N = 64, 256, 512; all ξ that were timed):**
- E_op = 2.77e-8 … 3.63e-8, so **all ≤ 2e-7**.
- Relative to the k ≠ 0 part of Γ_ref: 6.2e-8 … 1.06e-7.
- E_Frobenius = 1.5e-8 … 3.8e-8.

**Probe actions** (the same probe vectors for every parameter set of a configuration), maximum operator-normalised
error ‖ΔΓX‖/(‖Γ_ref‖‖X‖):

| probe | max operator-normalised error |
|---|---|
| Gaussian | 2.6e-8 |
| zero-mean Gaussian | 2.6e-8 |
| longitudinal k_min modes | 3.5e-8 |
| transverse k_min modes | 2.5e-8 |
| approximate slow eigenvectors (lowest 6 + 10 % noise) | 2.5e-8 |
| translation | 9.8e-16 (absolute ≤ 3.9e-13) |

‖ΔΓX‖/‖Γ_ref X‖ is reported but not used as a criterion. It reaches 1.4e-7 for the slow vectors because
‖Γ_ref X‖ is small there; for translations it is undefined.

**Structure:**
- symmetry ≤ 2.5e-17; ‖Γ_h T‖/‖Γ_h‖ ≤ 1.7e-15;
- λ_min(Γ_h) on Ran Π ≥ 2.30, equal to λ_min(Γ_ref) within 4.5e-7;
- **no PSD stabilisation is used**, so there is no stabilisation perturbation to account for.

**Cached action and assembly:**
- On the selected sets, the cached D−H action applied to three fixed random X matches the independent dense assembly
  (`GammaH.dense`) to ≤ 1.4e-15.
- At ξ_opt (17 sets), Γ_h assembled by applying the matrix-free cached action to all 3N standard basis vectors
  matches the independent assembly to ≤ 2.7e-15. Its E_op differs by ≤ 2.4e-16.

**Baseline set (p = 8):** E_op = 5.0e-8 … 6.9e-8 on all 19 configurations, also within budget.

### 6.4 Certified bound vs measured error, per component, at ξ_opt (`results/error_contributions.{csv,md}`)

All values are divided by ‖Γ_ref‖₂.

| | certified bound | measured | bound / measured |
|---|---|---|---|
| real space (ε_S vs ‖Γ_S^{rc} − Γ_S‖) | 5.5e-8 – 6.0e-8 | 3.0e-8 – 3.2e-8 | **1.75 – 1.90** |
| Fourier truncation (ε_F vs ‖Γ_L^{kc} − Γ_L‖) | 2.6e-9 – 5.2e-8 | 7.3e-11 – 1.3e-9 | 32 – 40 (N = 64), **107 – 136 (N = 256)** |
| mesh, C(7) (ε_mesh vs ‖Γ_L^{mesh} − Γ_L^{kc}‖) | 5.3e-8 – 5.8e-8 | 1.3e-11 – 4.7e-11 | 1.2–1.4·10³ (N = 64), **3.8–4.2·10³ (N = 256)** |
| quadrature (estimate only, not certified) | — | 8.7e-16 – 3.0e-15 | — |
| total (Σ bounds vs measured E_op) | 1.14e-7 – 1.68e-7 | 2.96e-8 – 3.20e-8 | 3.6 – 5.4 |

- **Real-space bound:** tight. It is configuration-specific and sees the actual local geometry.
- **Fourier and mesh bounds:** they sum all modes coherently (‖e_k e_k^H‖ = N), so their over-estimate grows with N
  (about ×3.4 and ×3.2 from N = 64 to 256).
- **Mesh bound:** it also carries C(7) = 26.45, against roughly one dominant alias term in practice.
- **Why the total ratio stays small:** each bound is sized to its own 1/3 of the budget. The conservatism shows up in
  the parameters rather than in the final E_op. Measured errors at the certified sets are dominated by the real-space
  truncation.

### 6.5 Does clustering, or full-periodic image multiplicity, change ξ_opt?

- **Clustering: no.** ξ_opt at N = 256 is 0.560 / 0.570 / 0.570 / 0.555 for uniform / moderate / strong / LJ, and
  the measured fastest ξ is the same.
  - At ξ_opt, rc ≈ 7.3 exceeds L. At that range the image count is fixed by the mean density, not by the local
    arrangement: the homogeneous estimate N ρ (4π/3) rc³/2 is within 0.9 % (uniform) and 5 % (strong) of the exact
    N_img for rc ≥ 3.
  - The certified ε_S bound responds to clustering, but a_tol changes only between 4.147 and 4.182.
  - The larger ‖Γ_ref‖ of clustered states loosens the relative budget slightly.
  - At N = 64, uniform has ξ_opt = 0.640 against 0.600 for the clustered levels. This is not a pair-count effect.
    - At ξ = 0.6 all levels select the same shell (kc = 5.340).
    - Uniform needs M = 30 instead of 27. Its smallest normaliser (Λ_lb = 10.87, uniform seed 3, against ≥ 11.7 for
      the clustered seeds) tightens the absolute mesh budget.
    - That step lifts T_pred(0.6) from 4.04 to 4.52 ms. On this step-shaped landscape the optimum moves to the
      neighbouring ξ = 0.64 (4.14 ms).
- **Image multiplicity: yes, decisively.** At ξ_opt there are 4.7–5.3 retained image interactions per unique pair at
  N = 256 (13.5–16.5 at N = 64, 2.8–3.1 at N = 512).
  - If the same cost model counted unique KD-tree pairs instead (`query_pairs(rc)`, capped at N(N−1)/2), it would
    place ξ_opt at the lower edge of the search range (ξ = 0.40) and predict 5.4 ms.
  - The measured time there is 35.5–36.6 ms, against 19.5–21.8 ms at the true optimum.
  - Correct periodic-image counting is therefore what makes the model predictive (Figure 3).

### 6.6 Where the time goes (`timing_breakdown.csv`)

At the certified optimum, per isolated component (median ms):

| | real-space H | spread | FFT + influence | gather | D_i X_i | sum | full action | existing pair-difference matvec |
|---|---|---|---|---|---|---|---|---|
| strong_N256_s1 (ξ 0.570, M 48) | 10.62 | 0.81 | 7.14 | 0.73 | 0.01 | 19.29 | 20.66 | 18.44 |
| uniform_N256_s1 (ξ 0.560, M 48) | 11.35 | 0.84 | 7.76 | 0.76 | 0.01 | 20.71 | 21.76 | 19.41 |
| strong_N512_s1 (ξ 0.535, M 60) | 25.54 | 2.12 | 16.49 | 1.62 | 0.01 | 45.79 | 50.54 | 44.79 |

- **Breakdown:** over the 10 N ≥ 256 optima, the real space dominates (53–57 % of the component sum), followed by the
  FFT/influence step (35–39 %); spread + gather take 7–9 %.
- **Cached D−H form vs existing form:** the cached form is 1–14 % *slower* than the existing pair-difference matvec
  on the same operator at N = 256 (median 12 %). It needs two dot products per image (X_j·d and X_i·d) instead of one. Caching D_i removes no
  per-action work in the real-space part; H_i(X) still has to be recomputed for every X.
- **Setup breakdown:** at N = 256, the 65–103 ms setup splits into image enumeration 35–62 ms, θ_S weights and the
  real-space degree D^S_i 19–23 ms, the mesh stencil 2.4–3.4 ms and the mesh degree 9.3–14.7 ms.
- **Parameter-only precomputation** (the `Mesh` influence function, A and B quadrature) takes 1.1–3.4 ms and is
  reported separately as T_param.

### 6.7 Comparison with the validated baseline

| N | baseline T_action [ms] | certified-optimum T_action [ms] | speedup | baseline E_op | certified E_op |
|---|---|---|---|---|---|
| 64 | 1.51 – 1.80 | 3.67 – 4.67 | 0.35 – 0.44 | 5.5e-8 – 6.2e-8 | 2.96e-8 – 3.19e-8 |
| 256 | 5.57 – 6.14 | 19.54 – 21.76 | 0.27 – 0.31 | 5.2e-8 – 6.9e-8 | 2.96e-8 – 3.20e-8 |
| 512 | 11.82 – 12.15 | 47.64 – 50.54 | 0.23 – 0.25 | 5.0e-8 – 5.8e-8 | 3.04e-8 – 3.13e-8 |

The certified selection is 2.3–4.3× slower than the baseline. It is a *certificate*: the baseline (p = 8, M = 24)
has no certificate at this accuracy. Its ε_mesh bound with C(8) at η = 0.646 is far above the budget. The baseline
was accepted by measurement on 152 LJ states in the manuscript, and it also meets 2e-7 on every configuration here.

### 6.8 Setup cost and reuse (T_setup + s T_action; Figure 6b)

**Predicted ξ_opt (N = 256):**

| level | action only | s = 1 | s = 10 | s = 20 |
|---|---|---|---|---|
| uniform | 0.560 | 0.660 | 0.580 | 0.580 |
| moderate | 0.570 | 0.705 | 0.570 | 0.570 |
| strong | 0.570 | 0.675 | 0.625 | 0.625 |
| LJ equilibrated | 0.555 | 0.655 | 0.605 | 0.555 |

**Measured best ξ** (among the measured ξ; the predicted value is in brackets):
- strong_N256_s1: s = 1 → 0.625 (0.675); s = 10 → 0.625 (0.625); s = 20 → 0.625 (0.625).
- uniform_N256_s1: s = 1 → 0.650 (0.660); s = 10 → 0.560 (0.580); s = 20 → 0.560 (0.580).
- moderate_N256_s1 and the LJ-equilibrated configuration: every prediction is exact.

**Why the optimum moves.** Setup costs 0.39 µs per image against 0.062 µs per image for the action, so a
configuration used once (s = 1) favours larger ξ: fewer images, more mesh.

**At s = 20,** the certified best costs 480–509 ms against 137–147 ms for the baseline (N = 256).

### 6.9 Separate empirical equal-accuracy diagnostic (NOT a certificate; `results/empirical.json`)

**Procedure.**
- For ξ ∈ {0.5, 0.6, 0.7, 0.85, 1.0, 1.2}, three sequential 1-D searches were run on every seed of each level at
  N = 64 and 256. Each finds the smallest parameter whose *measured* component error is ≤ 0.33 ε ‖Γ_ref‖:
  - rc by bisection;
  - kc down the shell list from the certified kc;
  - M up the smooth-number list, with the next M required to pass too.
- The measured total is then ≤ 0.99 ε by the triangle inequality, and was confirmed with the measured E_op.
- This is in-sample (selected and validated on the same configurations) and uses a coarse ξ set.

**Parameters:**
- a_emp/a_tol = 0.974–0.991 (the real-space bound is tight).
- b_emp/b_tol = 0.82–0.89.
- M_cert/M_emp = 2.25–3.75. At ξ = 0.85, N = 256: M = 25 against 81–90.

**Accuracy:** the measured E_op of the empirical sets is ≤ 1.14e-7 in all 96 checks.

**Time:**
- The predicted empirical optimum is **ξ = 1.0** for every level at N = 64 and 256.
- Measured (N = 256, seed 1, M = 30): 5.13 ms (uniform), 4.98 ms (moderate), 5.22 ms (strong), all fastest among the
  six ξ.
- This is 1.07–1.20 × faster than the baseline on the same configurations.

The theory-guided *structure* (choose ξ by minimising the calibrated geometry-aware model at fixed accuracy)
therefore works. The *certified* error constants are what make it uncompetitive.

## 7. Figures (`figures/`)

1. `fig1_configurations.png`: 3D configurations (uniform, moderate, strong, LJ equilibrated; N = 256), pair-distance
   histogram, neighbour-count distribution, and S(k)_max against N.
2. `fig2_time_vs_xi.png`: predicted vs measured action time against ξ for the strongly clustered and the uniform
   N = 256 configurations (baseline marked), plus the predicted component breakdown.
3. `fig3_pairs_vs_xi.png`: image interactions vs unique KD-tree pairs vs the homogeneous estimate against ξ, and image
   multiplicity for N = 64 / 256 / 512.
4. `fig4_mesh_vs_xi.png`: certified M(ξ) for N = 64 / 256 / 512 with the empirical M, plus a_tol(ξ) and b_tol(ξ).
5. `fig5_error_vs_xi.png`: measured E_op against ξ (certified sets, empirical sets, baseline) with the certified bound
   and ε = 2e-7.
6. `fig6_xi_opt_vs_clustering.png`: ξ_opt against clustering level for N = 64 / 256 / 512, measured fastest ξ and
   empirical ξ_opt, plus the reuse counts s = 1, 10, 20.

The error-contribution table is `results/error_contributions.md`.

## 8. Limitations and things to keep apart

**What "certified" means here.**
- The bounds are certified in exact arithmetic for the given kernel coefficients.
- A and B come from Gauss–Legendre quadrature, whose error was only *estimated*: 1e-15 relative coefficient
  difference to a refined rule. Floating-point round-off (~1e-15) is not bounded.
- ε_S is certified per configuration, from the explicit image sum.
- The normalisation Λ_lb was taken from the diagonal blocks of the dense reference. It can be computed in O(N
  neighbours) without the dense matrix.

**Timing scope.**
- One VM, one NumPy implementation, single-threaded.
- Component timings use a short budget (≥ 15 repetitions, 0.6 s), so their sum can deviate from the full-action
  median by up to ~25 % on some baseline points.
- All rows report median, IQR and min/max.

**Clustering scope.**
- At ρ = 0.75 with d_min = 0.9, the requested strong clustering is infeasible.
- The realised level (two near-jammed droplets) is the strongest this generator reaches without overlaps.
- Larger N gives more contrast (S(k)_max 3–4 / 15 / 31 at N = 64 / 256 / 512).

**Coverage.**
- N = 512 has one seed per level and no dense empirical diagnostic.
- The empirical diagnostic is in-sample and coarse in ξ.

**A bug found and fixed during this run.**
- The first `bench`/`errors` pass read kc from a CSV written with 10 significant digits. In 59 of 124 certified rows
  that dropped the outermost reciprocal shell from the operator (`lattice_modes` keeps |k| ≤ kc(1 + 1e-12)).
- Fixes: full-precision CSVs, kc snapped to exact shell radii, and a run-time assertion that the outer shell is
  present.
- Those passes were discarded and `select`, `bench` and `errors` were rerun. All numbers here are from the rerun.

**A second, smaller bug.**
- In `errors.json`, the per-row summary fields `act_*_trans` were aggregated while `probe_class` still matched the
  translation probe as "trans". The transverse maxima there therefore also include the translation probe (≤ 1e-15),
  which does not change them.
- The per-probe records in `errors.json` are complete and correct. The class table used in §6.3
  (`results/action_errors_by_class.csv`) is re-aggregated from them with the fixed classification.

## 9. Files

- **Code:** `neighbor_images.py`, `test_neighbor_images.py`, `clustered_configurations.py`, `cached_operator.py`,
  `parameter_optimizer.py`, `error_validation.py`, `run_clustered_benchmark.py`, `make_figures.py`, `common.py`.
- **Requested deliverables:** `parameter_results.csv`, `timing_breakdown.csv`, `benchmark_report.md`,
  `reproducibility.json` (environment, constants, seeds, timing protocol, command log, sha256 of every result
  file), `figures/`.
- **`results/`:**
  - `neighbor_validation.{json,csv}`
  - `configurations.{csv,json}`
  - `references.json`
  - `cost_model.json`, `calibration_timings.csv`
  - `selection_grid.csv` (per configuration and ξ, coarse grid), `selection_family.csv`, `selection_summary.json`
  - `bench.json`
  - `errors.json` (including every probe), `action_errors_by_class.csv`
  - `error_contributions.{csv,md}`
  - `empirical.json`
  - `commands.log`
- **Other:** `configs/*.npz`; `logs/`. `raw/` (dense Γ_ref) is git-ignored and regenerated by `--stage refs`.

## 10. Commands executed

All commands were run from `benchmarks/method_A_clustered/` unless stated. The time-stamped log of the benchmark
stages is `results/commands.log`.

```bash
pip install scipy matplotlib                       # scipy 1.18.1, matplotlib 3.11.2 (not installed in the container)
git checkout -B claude/intelligent-bell-62t6pg origin/claude/new-session-nmi9ux       # repository root

python3 test_neighbor_images.py                    # run during development (initial, rc < L/2 fast path,
                                                   # chunked enumeration) and once more at the end (all pass)
python3 clustered_configurations.py                # first version (uncapped push-apart) replaced; final run
                                                   # repeated at the end, bit-identical positions
python3 parameter_optimizer.py --config strong_N256_s1 [--xi ...]   # quick looks at the certified selection

python3 run_clustered_benchmark.py --stage refs
python3 run_clustered_benchmark.py --stage calibrate   # twice: plain LSQ (C_0 < 0) -> NNLS refit + bincount setup
python3 run_clustered_benchmark.py --stage select      # three times: initial, after the rc-bisection speed-up
                                                       # (identical result), after the kc-precision fix (final)
python3 run_clustered_benchmark.py --stage bench       # twice; the first pass was discarded (kc rounding, §8)
python3 run_clustered_benchmark.py --stage errors      # twice; the first pass was discarded (kc rounding, §8)
python3 run_clustered_benchmark.py --stage empirical
python3 make_figures.py                                # figures, parameter_results.csv, timing_breakdown.csv,
                                                       # error_contributions, reproducibility.json
```

The final pipeline was chained so that the timing stages ran with nothing else on the machine:
`select > logs/select.log && bench > logs/bench.log && errors > logs/errors.log && empirical > logs/empirical.log`.
