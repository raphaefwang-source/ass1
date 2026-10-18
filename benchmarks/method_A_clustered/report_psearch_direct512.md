# Method A at N = 512: (1) joint optimisation of (ξ, r_c, k_c, M, p), (2) direct periodic kernel vs optimised PPPM

GitHub issue #2 (`raphaefwang-source/ass1`), as revised in the request of 2026-10-18. Starting point: commit `c0d77cd`
(branch `claude/intelligent-bell-62t6pg`); work branch `claude/issue-2-psearch-direct512`.

Every number below is read from a result file in this directory. Most tables are generated from those files by
`report_tables_psearch.py`, and the full generated tables are in `results_psearch/report_tables.md`. Earlier results
(`results/`, `results_joint/`, `figures*/`, `logs*/`, `benchmark_report.md`, `report_joint.md`) are untouched.

**Fixed throughout** (`common.py`, unchanged constants):
- Kernel A: K(r) = c e^{−κr}/r · r̂r̂ᵀ, with γ = 0.5, κ = 0.7, r_ref = 1.3, c = 1.6148.
- Density ρ = 0.75.
- Full periodic lattice sum: k = 0 retained, no minimum image.
- τ = EPS_TARGET = 2e-7, relative spectral norm.
- Configurations: `configs/{strong,uniform,moderate}_N512_s1.npz` (N = 512, L = 8.8051). Primary: `strong_N512_s1`.
- Independent reference Γ_ref: `prl_toy_models.LatticeFriction` at rel_tol 1e-13 (`raw/<config>_Gref.npy`). It is checked
  in this round against an independent direct image sum (§5.2).

Machine and timing conditions:
- One Intel Xeon @ 2.10 GHz VM, 4 cores, L2 8 MiB, L3 260 MiB.
- Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1.
- One process; OMP/OPENBLAS/MKL = 1, and **both OpenBLAS pools (NumPy's and SciPy's) at 1 thread, verified with
  threadpoolctl**. Every timing stage refuses to start otherwise (see §7.1 for a bug found and fixed on the way).
- Nothing else ran during any timing stage.

## 0. Results

### Stage 1: measured-fastest accuracy-passing PPPM parameters (p searched over 5…10, plus 11 as the boundary neighbour)

| config | p | ξ | r_c | k_c (shell) | M | η | image interactions | certified bound / Λ_lb | **measured E_op** | action | **T_action** median [q1, q3] | T_pred | T_setup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **strong_N512_s1** | **9** | 0.850 | 4.8878 | 9.3585 | 40 | 0.656 | 91 095 | 1.9935e-7 | **3.03e-8** | pair-difference | **15.84 ms** [15.20, 16.84] | 15.50 ms | 52 ms |
| uniform_N512_s1 | 9 | 0.840 | 4.9768 | 9.2766 | 40 | 0.650 | 98 794 | 1.9922e-7 | 2.73e-8 | pair-difference | 16.06 ms [15.58, 16.93] | 16.58 ms | 58 ms |
| moderate_N512_s1 | 9 | 0.860 | 4.9001 | 9.4667 | 40 | 0.663 | 97 098 | 1.9917e-7 | 1.82e-8 | pair-difference | 15.90 ms [15.27, 17.10] | 14.80 ms | 58 ms |

How these were obtained:
- T_action is the median of 150 interleaved rounds (confirmation run, §3.3); T_pred is the per-p cost model's
  prediction for the same parameters.
- E_op = ‖Γ_h − Γ_ref‖₂/‖Γ_ref‖₂, from the full dense operator, not from probes.

Main findings:
- **The optimal order is p = 9 in all three configurations.** It is interior: the neighbours p = 8 and p = 10 are
  measurably slower, and so is p = 11, which was checked because the model put p = 10 at the boundary.
- **Per-p best measured action** (confirmation run, strong_N512_s1):

  | p | 5 | 6 | 7 | 8 | **9** | 10 | 11 |
  |---|---|---|---|---|---|---|---|
  | T_action [ms] | 90.1 | 38.9 | 24.0 | 17.4 | **15.8** | 17.0 | 18.9 |

- **Freeing p gives a measured speedup** of 1.51× (strong), 1.56× (uniform) and 1.45× (moderate) over the best p = 7
  point, in the same interleaved run.
- **Every candidate passes the measured accuracy check.** All 117 timed candidates are certified (bound ≤ τ), and their
  measured E_op is 5.1e-9 … 9.1e-8.
- **Implementation choice.** The pair-difference action (`GammaH.matvec` form) has a 0.1–10.4 % lower median than the
  cached D−H action for every one of the 36 confirmed candidates, and is numerically identical to it (≤ 1.3e-15). It is
  the production action.
- **Model vs measurement.** The model's argmin is the measured optimum for strong and moderate. For uniform the model
  preferred p = 10 (15.13 ms predicted); measured, that point is 2.7 % slower than the p = 9 optimum.

### Stage 2: direct periodic kernel vs optimised PPPM vs preassembled dense physical Γ (N = 512)

The same positions, box, density, kernel A, full periodic convention (k = 0 retained, no minimum image) and the same 10
deterministic input vectors were used for every method. All methods were checked against the same Γ_ref.

strong_N512_s1 (`results_direct512/stage2_strong_N512_s1.json`; medians, n = number of timed repetitions):

| method | what is timed | setup [ms] | one action [ms] | action / PPPM action | E_op vs Γ_ref |
|---|---|---|---|---|---|
| **PPPM**, Stage 1 optimum (p = 9, pair-difference action) | setup: images, weights, stencil, mesh degree | **55.2** (n = 10) | **15.64** [14.86, 16.83] (n = 200) | 1 | 3.03e-8 |
| **direct, pair-summed**, R_d = R_meas = 25.75 | setup: 13 684 015 images summed into the 130 816 periodic pair tensors K^per_R(q_i − q_j); action: loop over pairs | 2089 (n = 10) | **11.42** [10.96, 12.13] (n = 200) | **0.730** | 1.77e-7 |
| direct, pair-summed, R_d = R_cert = 27.0 | as above, 15 821 858 images | 2284 | 11.40 | 0.729 | 7.55e-8 |
| **direct, image-resolved**, R_d = R_meas | setup: the image list; action: loop over all 13.7 M images (Eq. 7 literally) | 7195 | **1211** [1173, 1275] (n = 30) | **77.4** | 1.77e-7 |
| direct, image-resolved, R_d = R_cert | 15.8 M images | 9518 | 1409 (n = 30) | 90.1 | 7.55e-8 |
| **dense physical Γ_D(R_meas)** (separate comparator, not the direct algorithm) | setup: pair-summed direct + 3N × 3N assembly; action: BLAS GEMV, (1536, 1536) float64, 18 MiB | 2117 | **1.321** [1.278, 1.387] (n = 200) | **0.084** | 1.77e-7 |

Uniform and moderate give the same picture:

| quantity | uniform | moderate |
|---|---|---|
| PPPM action | 16.61 ms | 16.01 ms |
| direct pair-summed action (ratio to PPPM) | 11.85 ms (0.713) | 11.42 ms (0.713) |
| direct image-resolved action | 1240 ms | 1204 ms |
| dense GEMV | 1.388 ms | 1.343 ms |
| PPPM setup | 59 ms | 59 ms |
| direct pair-summed setup | 2.04 s | 2.02 s |
| dense setup | 2.12 s | 2.05 s |

**Answer, measured at N = 512 on this machine:**
- **Per action, PPPM vs direct:**
  - The optimised PPPM is **75–90 × faster** than the literal image-resolved direct sum.
  - It is **1.37–1.40 × slower** than the pair-summed direct action, i.e. direct with K^per tabulated per pair in setup.
  - It is **11.8–12.0 × slower** than a GEMV with the preassembled dense physical Γ.
- **Setup:** PPPM is **34–41 × cheaper** than the pair-summed direct and dense setups, and 114–172 × cheaper than the
  image-resolved one.
- **When positions change** (T_setup + q T_action per configuration):
  - PPPM is the cheapest method for q < 135–144 actions per configuration; above that, the dense physical Γ is.
  - The pair-summed direct action overtakes PPPM only beyond q = 416–482 (494–526 with R_cert).
  - The image-resolved direct sum never does.
  - For q = 1 / 10 / 40 / 80, dense/PPPM = 29.9 / 10.1 / 3.19 / 1.70 on strong_N512_s1.
- **Accuracy:** all methods meet τ against the same reference.
  - PPPM: E_op = 1.8–3.0e-8.
  - Direct at the convergence-test radius R_meas: 1.77–1.82e-7, by construction just inside τ.
  - Direct at the certified radius R_cert: 7.6–8.0e-8.
- **Scope:** this is a fixed-N measurement; a multi-N study is needed to locate the asymptotic crossover (§5.5).

## 1. What was done, in order

1. **Alias bound for every p** (`test_alias_bound_p.py`, `results_psearch/alias_bound_validation_p.json`). The
   Hurwitz-zeta bound was re-validated for p = 4…11 on the existing B-spline code (§2).
2. **Per-p cost model** (`run_psearch_dense512.py --stage calibrate`, `results_psearch/cost_model_p.json`).
   - Isolated N = 512 timings per configuration and for both action implementations: real space vs N_img;
     spread + gather for every p = 4…11 at M = 30, 45, 64; FFT + influence for 22 sizes M = 20…128; full actions at four
     (r_c, M) per p.
   - Model: T(p) = C_R N_img + [S_p + G_p M³] + C_F M³ ln M³ + C_0(p), with S_p, G_p, C_0(p) recalibrated per p.
3. **Joint search per p** (`--stage select`, `results_psearch/selection_<config>.json`).
   - Approach C of `joint_optimizer.py` run for every p: joint budget, HZ mesh bound, exact reciprocal shells, 2,3,5-smooth
     M ≤ 128, and the smallest admissible r_c from the remaining budget by bisection on the certified ε_S.
   - No fixed split.
   - ξ: grid 0.40…1.60 in steps of 0.025, then 0.005 steps within ±0.025 of each p's best ξ.
   - The ε_S curves of one ξ are shared by all p.
   - The model optimum touched p = 10, so p = 11 was added (§3.1).
4. **Screening** (`--stage bench`, `results_psearch/bench_<config>_main.json`).
   - Every candidate: per p, the model optimum, two (k_c, M) alternatives with another M, and ξ_opt ± 0.025 and ± 0.05;
     plus the model's global top 8. That is 36 / 43 / 38 candidates.
   - Both actions timed interleaved: 40 rounds, random order per round, the same X for every candidate within a round.
5. **Accuracy of every candidate** (`--stage errors`, `results_psearch/errors_<config>.json`): dense E_op against
   Γ_ref, structure, probe actions, and matrix-free vs dense.
6. **Confirmation** (`--stage confirm`, `results_psearch/optimum_<config>_main.json`).
   - The 6 fastest passing candidates plus the fastest passing candidate of every p, both actions, 150 interleaved
     rounds.
   - Measured optimum = the fastest passing (candidate, implementation).
7. **Stage 1 verification at the optimum** (`--stage dense`, `results_psearch/dense512_<config>.json`):
   - dense Γ_h vs the matrix-free action on 10 X and the translation modes;
   - basis-vector vs mode-space assembly;
   - E_op re-check (§4).
8. **Stage 2, only after 1–7** (`run_direct512.py`, `results_direct512/`): a convergence test for the direct truncation
   radius (§5.2), then the timing comparison (§5.3–5.5).

`joint_optimizer.py` was extended, not rewritten. The additions:
- a `PCostModel` (per-p coefficients);
- `predict_vec`;
- optional arguments `approaches`, `curves` (shared ε_S curves), `variants` and `M_cap` of `evaluate_xi`/`XiTables`.

The defaults are unchanged. A check reproduced two per-ξ records of `results_joint/joint_selection.json` bit for bit
(r_c, k_c, M, T_pred, cert identical).

## 2. Hurwitz-zeta alias bound for p = 4…11 (`results_psearch/alias_bound_validation_p.json`)

The bound is |β(k)| ≤ σ_p(x) = |x|^p [ζ(p, 1+|x|) + ζ(p, 1−|x|)] per axis, and |α| ≤ Π(1 + σ_p) − 1 in 3D. It is the
triangle inequality applied to the Poisson series of the code's stencil DFT, so it holds in exact arithmetic for every
p. The supporting facts were re-checked for every order:

| check | result |
|---|---|
| FT of the code's `bspline` vs sinc^p (p = 4…12) | ≤ 2.3e-12 (quadrature limit) |
| Partition of unity; B-spline value just outside the p-point stencil | ≤ 6.7e-16; exactly 0 |
| Hurwitz ζ (SciPy) vs direct summation; σ_p ≤ 2ζ(p)(x/(1−x))^p | ≤ 1e-12; holds for p = 4…12 |
| `Mesh.stencil_dft` vs the explicit Poisson series | ≤ 1.0e-12 (p = 11) … 1.1e-13 (p = 5) |
| Influence function: `Mesh.W2` = Ŵ² | ≤ 9.2e-16 |
| 1D: max β/σ_p over all particles, modes and M = 12…64, on modes with σ_p ≥ 1e-6 | ≤ 1 + 1e-9 for every p |
| 3D: max α/α_HZ, 5 configurations incl. all N = 512 ones, on modes with α_HZ ≥ 1e-6 | ≤ 1 + 1e-9 for every p |
| Operator level: measured ‖Γ_L^mesh − Γ_L^kc‖₂ vs the HZ bound (96 cases: strong N = 256, strong/uniform N = 512, p = 4…11, M = 24…60) | bound ≥ 335 × measured in every case |

Tightness:
- **Even p:** sup over particle position of β/σ_p = 1 up to ~1e-11, i.e. the bound is attained.
- **Odd p:** 0.993–0.99999.

**Floating point, reported rather than assumed away:**
- The computed β exceeds σ_p by up to 6.4e-15 in absolute terms (p = 11; ≤ 3.3e-16 at p = 4). This happens only
  where σ_p itself is below ~1e-11, or at m = 0 where β = 0 exactly. Example: σ_10(1/64) ≈ 2e-18.
- So the relative test β/σ_p ≤ 1 + 1e-12 used for p = 7 in round 2 **fails for p ≥ 8**, with ratios up to 4e4. This is
  purely a round-off effect.
- The new test therefore checks two things, reported separately:
  - an absolute round-off allowance: excess ≤ 2e-14 (1D) and ≤ 5e-14 (3D);
  - the ratio on modes with σ_p ≥ 1e-6.
- Inflating α_HZ by the largest measured excess changes ε_mesh by at most 2.3e-6 of the budget.
- This is a measured round-off level, not a certificate. Neither the round-off of the assembled operator nor that of
  the infinite-tail sums is certified (§7).

## 3. Stage 1 search and measurement

### 3.1 Per-p model optima (pair-difference cost model; full table with ε components in `results_psearch/report_tables.md`)

strong_N512_s1. Λ_lb = 15.178 and ‖Γ_ref‖₂ = 17.549. Absolute certified errors; cert_rel = (ε_S + ε_F + ε_mesh +
ε_quad)/Λ_lb.

| p | ξ | r_c | k_c | M | η | n_img | ε_S | ε_F | ε_mesh (HZ) | ε_quad (est.) | cert_rel | measured E_op | T_pred [ms] | T_action cached / pairdiff, screening [ms] | T_setup cached / pairdiff [ms] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.415 | 10.1368 | 4.3405 | 64 | 0.190 | 844088 | 5.90e-07 | 7.16e-10 | 2.44e-06 | 2.4e-14 | 1.9990e-07 | 1.980e-08 | 65.93 | 91.75 / 84.04 | 525 / 453 |
| 6 | 0.575 | 7.3261 | 6.0968 | 60 | 0.285 | 311864 | 6.49e-07 | 1.18e-09 | 2.38e-06 | 9.0e-14 | 1.9994e-07 | 1.941e-08 | 34.03 | 45.61 / 44.33 | 142 / 129 |
| 7 | 0.660 | 6.3419 | 7.1000 | 48 | 0.415 | 195953 | 6.95e-07 | 6.31e-10 | 2.34e-06 | 1.0e-13 | 1.9972e-07 | 2.583e-08 | 20.96 | 25.91 / 24.57 | 97 / 84 |
| 8 | 0.720 | 5.7943 | 7.8818 | 40 | 0.552 | 148645 | 7.07e-07 | 2.46e-10 | 2.33e-06 | 1.5e-13 | 1.9983e-07 | 2.728e-08 | 16.70 | 20.25 / 19.42 | 70 / 65 |
| 9 | 0.855 | 4.8869 | 9.4128 | 40 | 0.660 | 91054 | 7.43e-07 | 2.15e-10 | 2.29e-06 | 3.0e-13 | 1.9958e-07 | 2.491e-08 | 15.50 | 17.96 / 17.89 | 64 / 56 |
| 10 | 0.885 | 4.6787 | 9.7058 | 36 | 0.756 | 80861 | 9.96e-07 | 3.57e-10 | 2.04e-06 | 2.0e-13 | 1.9975e-07 | 3.353e-08 | 16.62 | 19.31 / 18.88 | 76 / 75 |
| 11 | 0.975 | 4.2279 | 10.7984 | 36 | 0.841 | 61728 | 1.12e-06 | 2.75e-10 | 1.90e-06 | 3.5e-13 | 1.9891e-07 | 4.019e-08 | 17.54 | 21.34 / 20.68 | 101 / 100 |

How the optimum moves with p:
- Raising p makes the alias term decay faster. The joint optimum responds by moving to larger ξ, a smaller real-space
  cutoff (r_c 10.1 → 4.2; image interactions 844k → 62k) and a coarser mesh relative to k_c (η 0.19 → 0.84).
- Spread + gather costs S_p = 0.42 ms at p = 4 rising to 9.8 ms at p = 11 (`cost_model_p.json`). These two trends
  balance at p = 9.
- Beyond p = 9, S_p keeps growing (5.9 → 7.0 → 9.8 ms for p = 9, 10, 11) faster than the savings in M (40 → 36) and
  N_img (91k → 81k → 62k).
- In every row the certified budget is used to 99.5–100 %, 63–80 % of it by the HZ mesh bound (Figure P3). The
  measured E_op is 0.1–0.2 τ.

Search boundary:
- The p = 5…10 search put the model optimum at the upper boundary p = 10 for uniform_N512_s1 (and at p = 9 for the other
  two).
- The neighbour p = 11 was therefore searched for all three configurations, and it is predicted and measured slower.
- p = 4 is not needed: the lower boundary p = 5 is 4.9–5.7 × slower than the optimum.

### 3.2 Cost model per p (`cost_model_p.json`, `calibration_timings_p.csv`)

Coefficients:
- C_R is 51.98–54.52 ns per image interaction for the pair-difference action and 57.44–60.53 ns for the cached action.
- C_F is 5.28–5.65 ns per M³ ln M³.
- C_P(p) = S_p/(N p³) is 10–16 ns for p = 4…11, so spread/gather is close to the nominal N p³ law at every p.

Fit quality and prediction:
- The full-action calibration points (4 per p and implementation) are reproduced within 23 % (per-p maximum).
- At the three measured optima the prediction is within −7 … +3 % of the confirmed median.
- In the screening run, measured times exceed the predictions by 9–35 % (Figure P2), most of all for the
  large-N_img orders p = 5, 6. Screening cycles through 72–86 large operators, so caches are cold. The calibration timed one
  operator at a time. The confirmation run (24 operators) is in between.
- The ranking of p is the same in all three settings.

### 3.3 Measured selection (screening 40 rounds, confirmation 150 rounds; `raw_screen_*`, `raw_confirm_*`)

Fastest passing candidate per p: from the confirmation run where that p was confirmed, otherwise from screening.

| config | p = 5 | 6 | 7 | 8 | **9** | 10 | 11 |
|---|---|---|---|---|---|---|---|
| strong_N512_s1 | 90.14 | 38.89 | 23.99 | 17.37 | **15.84** | 16.97 | 18.95 |
| uniform_N512_s1 | 90.71 | 42.33 | 25.13 | 18.35 | **16.06** | 16.50 | 19.15 |
| moderate_N512_s1 | 78.52 | 38.56 | 23.10 | 18.20 | **15.90** | 17.23 | 18.44 |

All values are confirmation medians in ms, pair-difference action.

How sure is the p = 9 result:
- **p = 9 vs p = 10:** p = 10 is 3–8 % slower in median. The interquartile ranges overlap slightly, for example on
  strong: [15.20, 16.84] vs [16.25, 18.20].
- **Within p = 9:** the ξ variants inside ± 0.02 of the optimum are statistically indistinguishable (same M = 40, N_img
  within 3 %). The reported optimum is the fastest median of that plateau, not a sharply resolved point.
- **Earlier round:** the previous round's p = 7 joint optimum at N = 512 (`results_joint/`, 24.4 ms) was measured in a
  different session. It is not used for any ratio here. Within this session, the best p = 7 point is 23.1–25.1 ms.

## 4. Stage 1 verification at the optimum (`results_psearch/dense512_<config>.json`)

| check | strong | uniform | moderate |
|---|---|---|---|
| E_op, dense Γ_h by `GammaH.dense` (mode-space assembly) | 3.034e-8 | 2.730e-8 | 1.822e-8 |
| E_op, Γ_h assembled from 1536 basis vectors through the matrix-free action | 3.034e-8 | 2.730e-8 | 1.822e-8 |
| ‖G_basis − G_dense‖₂/‖G_dense‖₂ | 1.1e-15 | 1.1e-15 | 1.2e-15 |
| vectorised assembly vs `GammaH.dense` | 1.7e-15 | 1.7e-15 | 2.0e-15 |
| max over 10 X of ‖G_dense X − Γ_h(X)‖/‖G_dense X‖ (pair-difference / cached) | 8.7e-16 / 8.5e-16 | 8.7e-16 / 8.6e-16 | 9.3e-16 / 9.1e-16 |

Two further facts:
- Every one of the 117 candidates passes E_op ≤ τ.
- The certified bound, expressed relative to ‖Γ_ref‖₂, is at least 1.9 × the measured E_op for every candidate
  (max E_op/(cert_rel Λ_lb/‖Γ_ref‖) = 0.49–0.53).

## 5. Stage 2: direct periodic kernel vs optimised PPPM vs dense physical Γ (`run_direct512.py`, `direct_periodic.py`)

### 5.1 Definitions and what each branch computes

- **Physical operator** (manuscript Eqs. kernelA, Kper, Gamma; the request's Eqs. 7–9):
  (ΓX)_i = Σ_{j≠i} Σ_{n∈Z³} K(q_i − q_j + nL)(X_i − X_j).
  - The kernel is evaluated by `prl_toy_models.radial_friction`, law A, the same function the reference uses.
  - No Ewald split, no minimum image.
  - Self images cancel and are not generated.
- **Direct, image-resolved** (`DirectImages`):
  - Setup: every image (i < j, n) with |q_i − q_j + nL| ≤ R_d, enumerated by the validated periodic KD-tree + exact
    lattice-shift routine (`neighbor_images.kdtree_images`), and its weight g(r)/r².
  - Action: a chunked pair-difference loop over all retained images.
- **Direct, pair-summed** (`DirectPairs`):
  - Setup: the truncated periodic pair kernel K^per_R(q_i − q_j) = Σ_{|d|≤R_d} K(d) for each of the 130 816 pairs,
    streamed over 251 lattice-shift blocks. Images are not stored.
  - Action: (ΓX)_i = Σ_{j≠i} K^per_R(q_i − q_j)(X_i − X_j), a vectorised loop over pairs (gather, 3 × 3 product,
    `bincount`).
  - This is the direct pairwise matrix-vector action of Eq. (Gamma), not a dense BLAS product.
- **Dense physical Γ_D(R_d)**, a separate comparator:
  - The pair-summed tensors are assembled into the 3N × 3N graph Laplacian with the reference's own
    `assemble_friction`: shape (1536, 1536), float64, 18.0 MiB.
  - Action: `G @ x`, an OpenBLAS dgemv on one thread.
  - GEMV with the reference matrix Γ_ref itself takes the same time (1.34–1.37 ms).
  - This is the physical Γ, not the discrete PPPM matrix Γ_h; Γ_h is only in the supplementary §6.
- **PPPM:** the measured-fastest accuracy-passing Stage 1 parameters (§0), with the pair-difference action. Both
  implementations were timed again.
- **Identity of the three direct variants** on the 10 X:
  - image-resolved vs pair-summed: ≤ 1.9e-15;
  - dense vs pair-summed: ≤ 1.1e-15;
  - pair tensors from the stored image list vs streamed: ≤ 9.4e-16;
  - identical image counts.

### 5.2 Direct truncation radius by a convergence test (`results_direct512/convergence_<config>.json`, Figure D1)

Pair tensors were accumulated per annulus up to R_max = 42 (59.5 M images), so Γ_D(R) is available on the whole grid
R = 12…32 (step 0.25) and 33…42. Three error measures per R:
- E_ref(R) = ‖Γ_D(R) − Γ_ref‖₂/‖Γ_ref‖₂;
- the self-convergence E_conv(R) = ‖Γ_D(R) − Γ_D(42)‖₂/‖Γ_D(42)‖₂, which is independent of Γ_ref;
- a certified tail bound, 2 [max_i λ_max(Σ discarded K) + a summation-by-parts tail beyond 42], divided by Λ_lb.

| config | R_meas (E_ref, E_conv ≤ τ here and at every larger grid R) | E_ref(R_meas) | images | R_cert (bound ≤ τ) | bound(R_cert) | images | ‖Γ_D(42) − Γ_ref‖/‖Γ_ref‖ |
|---|---|---|---|---|---|---|---|
| strong | 25.75 | 1.773e-7 | 13 684 015 | 27.0 | 1.718e-7 | 15 821 858 | 3.2e-12 |
| uniform | 26.0 | 1.766e-7 | 14 109 855 | 27.25 | 1.746e-7 | 16 243 985 | 3.8e-12 |
| moderate | 25.75 | 1.818e-7 | 13 685 441 | 27.0 | 1.783e-7 | 15 783 473 | 3.3e-12 |

What the convergence test shows:
- E_ref and E_conv coincide to plotting accuracy down to ~1e-11. The certified bound is ≈ 2.3 × the measured error.
- **Cross-check of the reference.** The unsplit direct sum and the reference (near images direct plus a Chebyshev far
  field) agree to 3.2–3.8e-12. So Γ_ref is confirmed by a second, independent algorithm.
- **Reference regeneration.** Γ_ref was regenerated in this session and is bitwise identical to the stored file.
- **The convergence is slow.** κ = 0.7 needs R_d ≈ 26 ≈ 3 L to reach 2e-7.
- **The truncated tail is nearly a mean field.** The isotropic estimate ρ ĝ_tail(R)/3, with
  ĝ_tail(R) = 4πc e^{−κR}(1+κR)/κ², divided by ‖Γ_ref‖, gives 1.67–1.74e-7 at R_meas. The measured value is
  1.77–1.82e-7.

Choice of radius:
- R_meas follows the request (convergence test). It is less conservative than the certified PPPM selection.
- R_cert is the like-for-like certified choice. It costs the pair-summed direct setup 9–18 % and the image-resolved
  action 13–16 %, and leaves the pair-summed action unchanged.

### 5.3 Timing protocol

- **Setups:** 10 interleaved rounds (random order per round; `gc.collect()` before each build, outside the timer) of
  every setup kind. PPPM setups were then timed in 30 further rounds (`setups_pppm_extra`; medians 56–59 ms pair-difference, 59–64 ms
  cached; consistent with the 10-round run).
- **Actions:**
  - a fast session: 200 interleaved rounds of the pair-summed direct (both radii), dense GEMV (Γ_D and Γ_ref) and both
    PPPM actions, 10 warm-up rounds;
  - a slow session: 30 interleaved rounds of all of these plus the image-resolved direct action (both radii).
  - Reported actions are from the fast session; the image-resolved direct action is from the slow one. In the slow
    session the fast actions are 2–14 % slower (cache pollution by the 13.7 M-image arrays).
- **Conditions:** the same 10 deterministic X (seed [20261018, 512, 7]) in the same order for every method; one process;
  both OpenBLAS pools at 1 thread (threadpoolctl).
- **One-time costs**, reported separately and not part of setup or action:
  - PPPM mesh and influence function: 10.6–12.0 ms;
  - Γ_ref generation: 2.29–2.72 s.

### 5.4 Results per configuration (setup / action medians with [q1, q3]; generated, `results_psearch/report_tables.md`)

#### strong_N512_s1

Direct truncation: R_meas = 25.75 (E_ref 1.773e-07, E_conv 1.773e-07, n_img 13684015), R_cert = 27.0 (bound 1.718e-07, n_img 15821858); Gamma_D(R_max = 42.0) vs Gamma_ref: 3.19e-12.

PPPM: p = 9, xi = 0.8500, r_c = 4.8878, k_c = 9.3585, M = 40, eta = 0.656, action implementation pairdiff.

| method | setup median [q1, q3] [ms] | action median [q1, q3] [ms] | T_action / T_action(PPPM) | T_setup / T_setup(PPPM) | E_op vs Gamma_ref | max ||A X - Gamma_ref X||/(||Gamma_ref|| ||X||) | translation null |
|---|---|---|---|---|---|---|---|
| direct image-resolved, R_meas | 7195.3 [6736.3, 7868.3] (n=10) | 1211.042 [1173.154, 1274.677] (n=30) | 77.429 | 130.35 | 1.773e-07 | 1.69e-07 | 0.0e+00 |
| direct image-resolved, R_cert | 9517.9 [8897.0, 10004.9] (n=10) | 1409.330 [1374.689, 1475.437] (n=30) | 90.107 | 172.43 | 7.551e-08 | 7.27e-08 | 0.0e+00 |
| direct pair-summed, R_meas | 2088.6 [2004.0, 2102.8] (n=10) | 11.421 [10.964, 12.134] (n=200) | 0.730 | 37.84 | 1.773e-07 | 1.69e-07 | 0.0e+00 |
| direct pair-summed, R_cert | 2284.3 [2215.7, 2309.7] (n=10) | 11.400 [10.833, 12.100] (n=200) | 0.729 | 41.38 | 7.551e-08 | 7.27e-08 | 0.0e+00 |
| dense physical Gamma_D(R_meas), GEMV | 2117.0 [1992.7, 2166.9] (n=10) | 1.321 [1.278, 1.387] (n=200) | 0.084 | 38.35 | 1.773e-07 | 1.69e-07 | 4.8e-16 |
| PPPM (Stage 1 optimum) | 55.2 [53.8, 58.0] (n=10) | 15.641 [14.863, 16.829] (n=200) | 1.000 | 1.00 | 3.034e-08 | 1.40e-08 | 0.0e+00 |

End-to-end per configuration of positions, T_setup + q T_action [ms] (measured medians):

| method | q = 1 | q = 10 | q = 40 | q = 80 | ratio to PPPM at q = 1 / 80 | break-even q vs PPPM |
|---|---|---|---|---|---|---|
| direct image-resolved, R_meas | 8406.3 | 19305.7 | 55636.9 | 104078.6 | 118.67 / 79.66 | – |
| direct image-resolved, R_cert | 10927.2 | 23611.2 | 65891.1 | 122264.3 | 154.25 / 93.58 | – |
| direct pair-summed, R_meas | 2100.0 | 2202.8 | 2545.4 | 3002.3 | 29.64 / 2.30 | 481.9 |
| direct pair-summed, R_cert | 2295.7 | 2398.3 | 2740.3 | 3196.3 | 32.41 / 2.45 | 525.7 |
| dense physical Gamma_D(R_meas), GEMV | 2118.3 | 2130.2 | 2169.8 | 2222.7 | 29.90 / 1.70 | 144.0 |
| PPPM (Stage 1 optimum) | 70.8 | 211.6 | 680.8 | 1306.5 | 1.00 / 1.00 | – |

One-time (not in setup or action): PPPM mesh + influence function 12.0 ms (n=5); Gamma_ref generation (LatticeFriction, rel_tol 1e-13) 2568 ms (n=3; regenerated vs stored max abs diff 0.0e+00). Dense GEMV with Gamma_ref itself: 1.343 ms. Images: {'direct_images_Rmeas': 13684015, 'direct_images_Rcert': 15821858, 'direct_pairs_Rmeas': 13684015, 'direct_pairs_Rcert': 15821858, 'n_pairs': 130816, 'pppm_real_space': 91095, 'direct_pairs_n_shifts': {'Rmeas': 251, 'Rcert': 251}}. Identity: {'pair_tensors_images_vs_streamed_maxrel': '9.4e-16', 'n_images_equal': True, 'direct_images_vs_pairs': '1.9e-15', 'dense_phys_vs_pairs': '1.1e-15', 'pppm_matrix_free_vs_dense_Gamma_h': '8.5e-16'}.

#### uniform_N512_s1

Direct truncation: R_meas = 26.0 (E_ref 1.766e-07, E_conv 1.766e-07, n_img 14109855), R_cert = 27.25 (bound 1.746e-07, n_img 16243985); Gamma_D(R_max = 42.0) vs Gamma_ref: 3.84e-12.

PPPM: p = 9, xi = 0.8400, r_c = 4.9768, k_c = 9.2766, M = 40, eta = 0.650, action implementation pairdiff.

| method | setup median [q1, q3] [ms] | action median [q1, q3] [ms] | T_action / T_action(PPPM) | T_setup / T_setup(PPPM) | E_op vs Gamma_ref | max ||A X - Gamma_ref X||/(||Gamma_ref|| ||X||) | translation null |
|---|---|---|---|---|---|---|---|
| direct image-resolved, R_meas | 7007.5 [6410.3, 7287.3] (n=10) | 1239.672 [1203.646, 1282.491] (n=30) | 74.645 | 118.56 | 1.766e-07 | 1.71e-07 | 0.0e+00 |
| direct image-resolved, R_cert | 8787.1 [8435.2, 9128.0] (n=10) | 1419.340 [1381.626, 1479.083] (n=30) | 85.463 | 148.66 | 7.659e-08 | 7.45e-08 | 0.0e+00 |
| direct pair-summed, R_meas | 2039.9 [1991.7, 2080.6] (n=10) | 11.849 [11.285, 13.077] (n=200) | 0.713 | 34.51 | 1.766e-07 | 1.71e-07 | 0.0e+00 |
| direct pair-summed, R_cert | 2389.5 [2310.9, 2466.5] (n=10) | 11.891 [11.315, 12.704] (n=200) | 0.716 | 40.43 | 7.659e-08 | 7.45e-08 | 0.0e+00 |
| dense physical Gamma_D(R_meas), GEMV | 2115.6 [2038.3, 2154.0] (n=10) | 1.388 [1.340, 1.504] (n=200) | 0.084 | 35.79 | 1.766e-07 | 1.71e-07 | 5.1e-16 |
| PPPM (Stage 1 optimum) | 59.1 [57.6, 62.6] (n=10) | 16.608 [15.590, 18.497] (n=200) | 1.000 | 1.00 | 2.730e-08 | 1.46e-08 | 0.0e+00 |

End-to-end per configuration of positions, T_setup + q T_action [ms] (measured medians):

| method | q = 1 | q = 10 | q = 40 | q = 80 | ratio to PPPM at q = 1 / 80 | break-even q vs PPPM |
|---|---|---|---|---|---|---|
| direct image-resolved, R_meas | 8247.1 | 19404.2 | 56594.3 | 106181.2 | 108.92 / 76.51 | – |
| direct image-resolved, R_cert | 10206.4 | 22980.5 | 65560.7 | 122334.3 | 134.80 / 88.15 | – |
| direct pair-summed, R_meas | 2051.7 | 2158.4 | 2513.9 | 2987.8 | 27.10 / 2.15 | 416.3 |
| direct pair-summed, R_cert | 2401.3 | 2508.4 | 2865.1 | 3340.7 | 31.72 / 2.41 | 494.1 |
| dense physical Gamma_D(R_meas), GEMV | 2117.0 | 2129.5 | 2171.1 | 2226.6 | 27.96 / 1.60 | 135.1 |
| PPPM (Stage 1 optimum) | 75.7 | 225.2 | 723.4 | 1387.7 | 1.00 / 1.00 | – |

One-time (not in setup or action): PPPM mesh + influence function 10.6 ms (n=5); Gamma_ref generation (LatticeFriction, rel_tol 1e-13) 2293 ms (n=3; regenerated vs stored max abs diff 0.0e+00). Dense GEMV with Gamma_ref itself: 1.367 ms. Images: {'direct_images_Rmeas': 14109855, 'direct_images_Rcert': 16243985, 'direct_pairs_Rmeas': 14109855, 'direct_pairs_Rcert': 16243985, 'n_pairs': 130816, 'pppm_real_space': 98794, 'direct_pairs_n_shifts': {'Rmeas': 251, 'Rcert': 251}}. Identity: {'pair_tensors_images_vs_streamed_maxrel': '8.3e-16', 'n_images_equal': True, 'direct_images_vs_pairs': '1.7e-15', 'dense_phys_vs_pairs': '1.1e-15', 'pppm_matrix_free_vs_dense_Gamma_h': '8.6e-16'}.

#### moderate_N512_s1

Direct truncation: R_meas = 25.75 (E_ref 1.818e-07, E_conv 1.818e-07, n_img 13685441), R_cert = 27.0 (bound 1.783e-07, n_img 15783473); Gamma_D(R_max = 42.0) vs Gamma_ref: 3.32e-12.

PPPM: p = 9, xi = 0.8600, r_c = 4.9001, k_c = 9.4667, M = 40, eta = 0.663, action implementation pairdiff.

| method | setup median [q1, q3] [ms] | action median [q1, q3] [ms] | T_action / T_action(PPPM) | T_setup / T_setup(PPPM) | E_op vs Gamma_ref | max ||A X - Gamma_ref X||/(||Gamma_ref|| ||X||) | translation null |
|---|---|---|---|---|---|---|---|
| direct image-resolved, R_meas | 6700.4 [6338.3, 7195.2] (n=10) | 1203.655 [1131.462, 1267.191] (n=30) | 75.182 | 113.68 | 1.818e-07 | 1.75e-07 | 0.0e+00 |
| direct image-resolved, R_cert | 9354.9 [8567.8, 10070.5] (n=10) | 1361.995 [1333.424, 1403.164] (n=30) | 85.072 | 158.71 | 7.967e-08 | 7.64e-08 | 0.0e+00 |
| direct pair-summed, R_meas | 2016.5 [1923.7, 2140.6] (n=10) | 11.423 [10.966, 12.279] (n=200) | 0.713 | 34.21 | 1.818e-07 | 1.75e-07 | 0.0e+00 |
| direct pair-summed, R_cert | 2372.0 [2287.3, 2499.3] (n=10) | 11.324 [10.894, 12.166] (n=200) | 0.707 | 40.24 | 7.967e-08 | 7.64e-08 | 0.0e+00 |
| dense physical Gamma_D(R_meas), GEMV | 2047.8 [1973.1, 2114.3] (n=10) | 1.343 [1.296, 1.412] (n=200) | 0.084 | 34.74 | 1.818e-07 | 1.75e-07 | 4.9e-16 |
| PPPM (Stage 1 optimum) | 58.9 [55.6, 60.8] (n=10) | 16.010 [15.217, 17.220] (n=200) | 1.000 | 1.00 | 1.822e-08 | 9.22e-09 | 0.0e+00 |

End-to-end per configuration of positions, T_setup + q T_action [ms] (measured medians):

| method | q = 1 | q = 10 | q = 40 | q = 80 | ratio to PPPM at q = 1 / 80 | break-even q vs PPPM |
|---|---|---|---|---|---|---|
| direct image-resolved, R_meas | 7904.1 | 18737.0 | 54846.6 | 102992.8 | 105.46 / 76.88 | – |
| direct image-resolved, R_cert | 10716.9 | 22974.8 | 63834.7 | 118314.5 | 142.98 / 88.31 | – |
| direct pair-summed, R_meas | 2028.0 | 2130.8 | 2473.5 | 2930.4 | 27.06 / 2.19 | 426.8 |
| direct pair-summed, R_cert | 2383.3 | 2485.3 | 2825.0 | 3277.9 | 31.80 / 2.45 | 493.6 |
| dense physical Gamma_D(R_meas), GEMV | 2049.1 | 2061.2 | 2101.5 | 2155.2 | 27.34 / 1.61 | 135.6 |
| PPPM (Stage 1 optimum) | 75.0 | 219.0 | 699.3 | 1339.7 | 1.00 / 1.00 | – |

One-time (not in setup or action): PPPM mesh + influence function 10.8 ms (n=5); Gamma_ref generation (LatticeFriction, rel_tol 1e-13) 2719 ms (n=3; regenerated vs stored max abs diff 0.0e+00). Dense GEMV with Gamma_ref itself: 1.351 ms. Images: {'direct_images_Rmeas': 13685441, 'direct_images_Rcert': 15783473, 'direct_pairs_Rmeas': 13685441, 'direct_pairs_Rcert': 15783473, 'n_pairs': 130816, 'pppm_real_space': 97098, 'direct_pairs_n_shifts': {'Rmeas': 251, 'Rcert': 251}}. Identity: {'pair_tensors_images_vs_streamed_maxrel': '8.3e-16', 'n_images_equal': True, 'direct_images_vs_pairs': '1.8e-15', 'dense_phys_vs_pairs': '1.1e-15', 'pppm_matrix_free_vs_dense_Gamma_h': '9.1e-16'}.

### 5.5 Interpretation

- **Why the pair-summed direct action beats PPPM per action at N = 512.**
  - Its per-action work is N(N − 1)/2 = 130 816 pair tensors, independent of R_d. The 13.7–16 M images were summed once
    in setup.
  - PPPM needs 91–99 k real-space images, spread and gather with p = 9 (5–6 ms) and an M = 40 FFT round trip.
  - The pair loop costs O(N²) per action and O(N²) memory (6 doubles per pair). PPPM costs O(N log N) at fixed accuracy.
  - At N = 512 the quadratic term is still the smaller one. Locating the crossover needs a multi-N study, which was not
    part of this experiment.
- **Why the dense GEMV is fastest per action.** The 18 MiB matrix stays in the 260 MiB L3 cache, and dgemv runs at
  ~3.5 Gflop/s on one thread.
- **What decides the end-to-end comparison: setup.** Every direct-type method must evaluate the unsplit, slowly decaying
  kernel on 13.7–16 M images for each new configuration. That takes 2.0–2.4 s; PPPM's real-space cutoff 4.9 needs 91–99 k
  images (55–59 ms).
- **For dynamics with moving particles** (positions change every step, q actions per step), PPPM is the cheapest method
  below q ≈ 135–144. The pair-summed direct action becomes cheaper than PPPM only beyond q ≈ 420–530.
- **For a frozen configuration** with many actions, the preassembled dense physical Γ is cheapest.
- **Accuracy.** All three methods meet τ against the same reference. PPPM has the largest margin, 6–11 × below τ,
  because its error budget is certified and therefore conservative. The direct sum at R_meas sits just below τ by
  construction.

## 6. Supplementary: dense discrete Γ_h of the same PPPM operator (issue #2's original Stage 2; `results_psearch/dense512_<config>.json`)

Issue #2 originally asked for this same-operator comparison. It is kept as a separate item and is not the Stage 2
answer:
- Here the dense matrix is the **discrete PPPM operator Γ_h** at the Stage 1 optimum, built from the same positions.
- §5 used the **physical Γ**.

Build methods:
- `GammaH.dense()` (project routine: `np.add.at` real-space blocks + `Mesh.pair_matrix` + `lap_full`);
- the same assembly vectorised (`bincount`; the real-arithmetic mode sum over ±k pairs);
- assembly from basis vectors through the matrix-free action.

Timings and checks: 30 interleaved builds; 200 interleaved GEMV/action rounds on the same 10 X.

| config | T_build project GammaH.dense [ms] | T_build vectorised [ms] | T_build basis [ms] | T_GEMV(Gamma_h) [ms] | T_PPPM_setup [ms] | T_PPPM_action [ms] | T_GEMV/T_action | break-even q (project build) | break-even q (vectorised) | max rel dense vs matrix-free | basis vs project | vectorised vs project | E_op |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strong_N512_s1 | 2434 | 730 | 22968 | 1.187 | 61.8 | 14.17 | 0.084 | 182.7 | 51.5 | 8.7e-16 | 1.1e-15 | 1.7e-15 | 3.034e-08 |
| uniform_N512_s1 | 2561 | 721 | 23318 | 1.235 | 59.4 | 14.65 | 0.084 | 186.4 | 49.3 | 8.7e-16 | 1.1e-15 | 1.7e-15 | 2.730e-08 |
| moderate_N512_s1 | 2615 | 735 | 22508 | 1.270 | 55.7 | 14.54 | 0.087 | 192.9 | 51.2 | 9.3e-16 | 1.2e-15 | 2.0e-15 | 1.822e-08 |

What this shows:
- The dense Γ_h GEMV is 0.084–0.087 × the PPPM action.
- **Break-even with PPPM** (positions changing):
  - 49–52 actions per configuration with the vectorised build;
  - 183–193 with the project routine;
  - ~1700 with basis assembly.
- **Session-to-session variation.** The PPPM action in this 3-method session is 14.2–14.7 ms, against 15.6–16.6 ms in
  the 8-method Stage 2 session. This is the cache effect of §7.

## Figures (`figures_psearch/`)
- **P1** `figP1_time_vs_p.png`: model optimum and measured best action per p (screening, both implementations;
  confirmation with IQR); the measured optimum is starred.
- **P2** `figP2_pred_vs_meas.png`: per-p model vs screening medians, all 117 candidates, both implementations.
- **P3** `figP3_error_budget_vs_p.png`: certified budget shares (ε_S, ε_F, ε_mesh HZ, ε_quad estimate) of each p's
  model optimum, and its measured E_op/τ.
- **P4** `figP4_xi_scan_strong.png`: predicted per-ξ optimum of every p, with the screened and confirmed points.
- **D1** `figD1_direct_convergence.png`: E_ref, E_conv and the certified bound vs the direct truncation radius.
- **D2** `figD2_setup_action_times.png`: setup and action medians of every Stage 2 method.
- **D3** `figD3_end_to_end.png`: T_setup + q T_action vs q.

## 7. Limitations and failures

### 7.1 Thread-pool bug in the first timing pass (found, fixed, re-run)

- **The bug.** The first calibration and screening (06:22 and earlier) ran with both OpenBLAS pools at 4 threads. The
  new runner imported NumPy before `common.py` set `OPENBLAS_NUM_THREADS=1`; the earlier rounds' runners share this
  import order.
- **Measured effect.** A same-process A/B test (60 interleaved rounds; `diag_blas_threads.py`,
  `results_psearch/run1_blas_pool4/blas_thread_ab.json`) gives a 4-thread/1-thread action time ratio of 0.97–1.008 for
  both implementations at p = 7, 9, 10. The PPPM actions call no BLAS, so the first pass was not distorted.
- **Fix and re-run.** The runners now import `common` first, call `threadpool_limits(1)` and abort any timing stage
  unless every pool reports 1 thread. Calibration, selection, screening and confirmation were then repeated. All
  reported Stage 1 and Stage 2 timings come from the re-run.
- **What was kept.** The first-pass files are kept unmodified in `results_psearch/run1_blas_pool4/` and
  `logs_psearch/run1_blas_pool4/`. Accuracy results do not depend on the thread count, so 21 first-pass E_op values were
  reused by candidate key.

### 7.2 Other limitations

- **Certificate scope.** The certificate covers the truncation errors in exact arithmetic:
  - ε_S: explicit image sum plus an analytic tail bound beyond R_far;
  - ε_F: shell sum plus a far-tail bound;
  - ε_mesh: Hurwitz-zeta bound;
  - for the direct sum, the summation-by-parts tail beyond R_max.

  It does not cover:
  - ε_quad, which is a numerical estimate (production vs refined Gauss–Legendre) included in the budget but not
    certified;
  - floating-point round-off in the operator and in the tail sums. The measured alias excess is reported in §2, and
    round-off is ~1e-15 relative, eight orders below τ, but not bounded rigorously.
- **Reference accuracy.** E_op is measured against Γ_ref, whose own error is ~1e-12 relative:
  - LatticeFriction rel_tol 1e-10 vs 1e-13 differ by 1.8e-12 (`results/references.json`);
  - the direct sum to R = 42 agrees with Γ_ref to 3.2–3.8e-12 (§5.1).
- **Timing variance.** Medians carry 7–20 % IQRs. The absolute action time of the same operator depends on how many
  other large operators are interleaved with it (cache state). For strong's optimum:
  - 17.6 ms in screening (72 actions),
  - 15.8 ms in confirmation (24),
  - 14.2 ms in the 3-method dense Γ_h run.

  Rankings are consistent across these settings; ratios should be read within one run.
- **Scope of the conclusions.** N = 512 only and one machine, so nothing here measures asymptotic scaling.

## 8. Reproducibility

- **Machine:** one Intel Xeon @ 2.10 GHz VM (4 cores, 1 thread per core, L2 8 MiB, L3 260 MiB), Linux 6.18.
- **Software:** Python 3.13.16, NumPy 2.5.3 (OpenBLAS 0.3.34), SciPy 1.18.1 (OpenBLAS 0.3.31), matplotlib 3.11.2,
  threadpoolctl 3.7.0.
- **Threads:** OMP/OPENBLAS/MKL_NUM_THREADS = 1, `threadpool_limits(1)`. The threadpool info is stored in every timing
  file's `provenance.blas`, and every pool reports 1 thread.
- **Seeds:**
  - X inputs: `default_rng([20261018, 512, 7])`;
  - interleaving order: `[20261018, 3, s]`;
  - accuracy probes: `[20261018, 13, N]`;
  - alias test: `[20261018, 21]`;
  - configurations: `configs/*.npz` (unchanged).
- **Code hash** (`common.code_hash`, sha256 over all `*.py` here plus the imported project modules; in every JSON
  `provenance`):
  - Stage 1 timing stages: `200345160cfecf7d`;
  - Stage 2: `c13747b41f70b433`.
  - Between the two, and after Stage 2, only the figure/table scripts changed (`make_figures_psearch.py`,
    `report_tables_psearch.py`). The committed tree hashes to `5bfab30fd1a04423`.
  - Git state during the runs: `c0d77cd` + uncommitted new files. The commit containing everything is the one adding
    this report.
- **Commands:** in order, with time stamps, in `results_psearch/commands.log` and `results_direct512/commands.log`.
  The final sequence, from `benchmarks/method_A_clustered/`:
  ```
  python3 test_alias_bound_p.py
  python3 diag_blas_threads.py                                   # documents the run-1 thread bug (§7.1)
  python3 run_psearch_dense512.py --stage calibrate
  python3 run_psearch_dense512.py --stage select --configs <config>            # x3 in parallel, p = 5..10
  python3 run_psearch_dense512.py --stage select --configs <config> --p 11 --resume
  python3 run_psearch_dense512.py --stage bench
  python3 run_psearch_dense512.py --stage errors --configs <config>            # x3 in parallel
  python3 run_psearch_dense512.py --stage confirm
  python3 run_psearch_dense512.py --stage dense                  # Stage 1 verification + supplementary Γ_h
  python3 run_direct512.py --stage converge --configs <config>   # x3 in parallel
  python3 run_direct512.py --stage bench
  python3 make_figures_psearch.py && python3 report_tables_psearch.py
  ```
  Parallel runs were only non-timing stages (selection, dense E_op, convergence). Every timing stage ran alone.
- **Raw timings (ms, every repetition):**
  - `results_psearch/raw_screen_<config>_main.json` (40 rounds × 72–86 actions);
  - `raw_confirm_<config>_main.json` (150 rounds × 24);
  - `raw_dense512_<config>.json` (builds, basis, actions);
  - `results_direct512/stage2_raw_<config>.json` (setups, fast and slow action sessions);
  - calibration medians in `results_psearch/calibration_timings_p.csv`.
- **Summary tables:** `results_psearch/psearch_per_p.csv`, `psearch_candidates.csv`, `report_tables.md`;
  `results_direct512/stage2_summary.csv`, `direct_convergence.csv`.
- **Reproducibility record:** `results_psearch/reproducibility_psearch.json` (seeds, every command, and the provenance of
  every stage: time, git state, code hash, versions, CPU, thread pools).
- **Logs:** `logs_psearch/` (run-1 logs in `logs_psearch/run1_blas_pool4/`).
- **Not committed:** dense reference matrices `raw/<config>_Gref.npy` (git-ignored, regenerated deterministically by
  `error_validation.reference_cached`; regenerated copies in this session were bitwise identical).

### New and changed code
- New:
  - `run_psearch_dense512.py` (Stage 1 stages + supplementary dense Γ_h);
  - `test_alias_bound_p.py`;
  - `direct_periodic.py`, `run_direct512.py` (Stage 2);
  - `diag_blas_threads.py`;
  - `make_figures_psearch.py`, `report_tables_psearch.py`.
- Changed: `joint_optimizer.py` (backward-compatible extension, §1).
- Not changed: physics, kernel, split, PPPM solver, reference, `common.py`, the manuscript, any earlier result.
- Not run: Lanczos and BAOAB.
