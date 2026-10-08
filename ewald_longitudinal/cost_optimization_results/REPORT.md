# Error-constrained cost optimisation of the PPPM + Lanczos thermostat (toy laws A and B)

Script: `toy_cost_opt.py` (stages `profile`, `search`, `refine`, `ranks`, `modes`, `slowmodes`, `ranks_large`,
`validate`, `timing`, `table`). All numbers below are read from the JSON/CSV files in this directory.

**What was held fixed:**
- the pure longitudinal model K = g(r) r̂r̂ᵀ;
- full periodic images, with k = 0 retained;
- the kernel normalisation g(r_ref) = γ = 0.5 (κ = 0.7, r_ref = 1.3);
- kT = 0.7, m = 1, dt = 0.005;
- density 64/5.5³.

**What was not done:**
- no random batch;
- no new long trajectory;
- none of the historical tests or long runs were rerun.

**Baseline (kept unchanged as the reference version):**
- PPPM ξ = 0.7, s = 4.10, η = 0.7, p = 7;
- Lanczos noise rank 40, damping rank 16;
- real-space pairs from the original image enumeration.

**Compute used:** 0.80 core-hours of the ~2 core-hour budget (`budget_log.json`, including an estimate for unlogged
checks).

## 0. Result table

> **Amendment (2026-10-08, Section 10).** The ranks in this table were chosen on replicated or v2 states. On a
> liquid-like test state, law B at 12/5 misses its budget at N = 256 and 512, and law A at 32/4 misses at N = 1728.
> The production configuration `costopt_hiacc` (`../toy_configs.py`) keeps these operator sets but uses higher
> ranks (A 32/5 for N ≤ 256 and 40/5 for N ≤ 512; B 16/6 and 24/8) and refuses N > 512. Its measured speedup at
> N = 512 against the original version, on one machine and with the same runner, is ×3.5 (A) and ×5.1 (B). The
> table below keeps the study's numbers as they were measured.

**How to read this table:**
- **Error budget:** the relative spectral error ‖Γ_h − Γ_ref‖₂/‖Γ_ref‖₂ against the dense full-periodic reference.
  The Lanczos noise and damping errors must meet the same budget. They are judged on the overall relative error,
  the k = 2π/L plane-wave components and the lowest 3 / 12 eigenmodes (strict rule, Section 4).
- **"=":** marks the equal-accuracy budget for each law. The baseline's own errors are 1.2e-7 (A) and 4.3e-8 (B).
- **Ranks:** "noise/damping" for N ≤ 512. Ranks for N = 1728 and 4096 are in brackets.
- **Mesh rule:** M = next_smooth(⌈L·k_c/(η*π)⌉), with k_c = 2ξs.
- **Step times:** measured medians of one full BAOAB step on 1 thread, in ms, at
  N = 64 / 256 / 512 / 1728 / 4096.
- **Speedups:** "×orig" is relative to the original code, "×tree" to the original parameters with the
  KD-tree pair search. For N ≥ 1728 both baselines are the same code path, because r_c < L/2.

| law, budget | recommended ξ, s, η*, p; ranks | op. error N=64 / 512 (Γv rows N=1728 / 4096) | step ms N=64 / 256 / 512 / 1728 / 4096 | measured speedup ×orig (×tree) | t ≤ 2 dynamics, N=64 | static checks and timing at |
|---|---|---|---|---|---|---|
| **A original** | 0.7, 4.10, 0.7, 7; 40/16 | 1.1e-7 / 1.2e-7 | 82.5 / 440 / 1634 / 1827 / 4718 | 1 | — | — |
| A = 2e-7 | 0.85, 4.1, 0.678, 7; **32/4** (32/4, 40/4) | 1.5e-7 / 1.4e-7 (7.7e-8 / 7.6e-8) | 36.7 / 149 / 314 / 1089 / 3796 | 2.2 / 2.9 / 5.2 / 1.7 / 1.2 (1.8 / 1.6 / 1.6 / 1.7 / 1.2) | max rel. 5e-7; 2 MSD CIs exclude 0 (−1.5e-8, −2.4e-8) | N = 64–4096 |
| A 1e-6 | 0.85, 3.9, 0.645, 6; **24/4** (32/4, 40/4) | 8.0e-7 / 7.0e-7 (4.1e-7 / 3.9e-7) | 25.6 / 98.5 / 193 / 1153 / 2728 | 3.2 / 4.5 / 8.5 / 1.6 / 1.7 (2.6 / 2.5 / 2.6 / 1.6 / 1.7) | max rel. 1.7e-6; no CI excludes 0 | N = 64–4096 |
| A 1e-4 | 0.85, 3.2, 0.794, 5; **16/3** (20/3, 24/3) | 6.0e-5 / 5.2e-5 (3.0e-5 / 3.0e-5) | 11.3 / 31.5 / 64.2 / 290 / 764 | 7.3 / 14 / 25 / 6.3 / 6.2 (5.8 / 7.8 / 7.8 / 6.3 / 6.2) | MSD(t=2) +1.1e-4 rel., CI [+7.9e-5, +1.5e-4] | N = 64–4096 |
| **B original** | 0.7, 4.10, 0.7, 7; 40/16 | 4.5e-8 / 4.3e-8 | 90.3 / 419 / 1216 / 1689 / 4579 | 1 | — | — |
| B = 5e-8 | 1.0, 3.9, 0.683, 7; **12/5** (16/5, 16/5) | 3.6e-8 / 3.5e-8 (2.0e-8 / 1.9e-8) | 18.9 / 80.5 / 172 / 751 / 1932 | 4.8 / 5.2 / 7.1 / 2.2 / 2.4 (3.3 / 3.2 / 2.7 / 2.2 / 2.4) | max rel. 2e-9; no CI excludes 0 | N = 64–4096 |
| B 2e-7 | 0.85, 3.8, 0.628, 6; 10/5 (12/5, 16/5) | **2.02e-7** / 1.9e-7 (9.2e-8 / 9.0e-8) | not timed | estimated only | not run | N = 64 is 1 % over budget; not recommended |
| B 1e-6 | 1.0, 3.5, 0.766, 6; **10/5** (12/5, 16/5) | 9.9e-7 / 9.5e-7 (4.2e-7 / 4.0e-7) | 11.6 / 40.1 / 74.3 / 322 / 926 | 7.8 / 10 / 16 / 5.3 / 4.9 (5.4 / 6.4 / 6.3 / 5.3 / 4.9) | max rel. 8e-8; 2 MSD CIs exclude 0 (+5e-8, +8e-8) | N = 64–4096 |
| B 1e-4 | 0.85, 2.9, 0.863, 5; **8/4** (8/4, 10/4) | 4.4e-5 / 4.7e-5 (2.6e-5 / 2.3e-5) | 7.4 / 19.1 / 36.5 / 143 / 355 | 12 / 22 / 33 / 12 / 13 (8.4 / 13 / 13 / 12 / 13) | MSD +5–6e-6 rel. at all t, CIs exclude 0 | N = 64–4096 |

All recommended sets use the KD-tree pair search (`FastFriction(..., pair_search="tree")`). The operator it
builds is identical to the original enumeration (Section 2).

**N = 64.** The dense full-periodic reference (matrix + eigendecomposition) is the cheapest equal-accuracy
method:
- dense: 16.7 ms (A) and 17.0 ms (B), with no operator or Krylov error;
- recommended PPPM: 36.7 ms (A) and 18.9 ms (B);
- PPPM is faster than dense at N = 64 only at the looser budgets.

**Crossover.** PPPM wins from N = 256 on:
- dense: 451 ms (A) and 441 ms (B);
- PPPM equal-accuracy: 149 ms (A) and 80 ms (B).

### Where the gains come from (N = 512, equal accuracy)

| law | original | + KD-tree pair search (same operator) | + ranks (same operator parameters) | + operator parameters |
|---|---|---|---|---|
| A | 1634 ms | 500 ms (×3.3) | ≈ 36·8.2 + 25 ≈ 320 ms (est.) | 314 ms (Γv 8.2 → 7.8 ms) |
| B | 1216 ms | 471 ms (×2.6) | ≈ 17·7.7 + 24 ≈ 155 ms (est.) | 172 ms (measured, Γv 8.6 ms) |

The "+ ranks" column is a cost-model estimate (operator build + ranks × Γv), not a measurement.

**Summary:**
- **At equal accuracy:**
  - pair search: ×2.6–3.3 at N ≤ 512;
  - Lanczos ranks: ×1.6 (A) and ×3 (B);
  - operator parameters: ≤ 10 % at N = 512, which is inside the timing spread. The B row's measured 172 ms is
    above its 155 ms estimate, also within that spread.
- **At looser budgets:** the operator parameters contribute substantially, ×2 (1e-6) to ×3–4 (1e-4) in Γv cost.

## 1. Where the time goes (`profile_images.json`, `profile_tree.json`)

One full step (two force calls, operator build, damping and noise Lanczos), baseline parameters, double-well
state, medians of 4 steps after 2 warm-up steps. Times in ms.

| code | N | law | step | forces | operator build | 56 Γv actions | reorth / small eig / other |
|---|---|---|---|---|---|---|---|
| original (image pairs) | 64 | A | 88.7 | 0.9 (1 %) | 30.0 (34 %) | 54.6 (62 %) | 0.9 / 0.4 / 1.8 |
| original (image pairs) | 512 | A | 2220 | 5.0 (0.2 %) | 1689 (76 %) | 518 (23 %) | 3.1 / 0.4 / 5.0 |
| KD-tree pairs | 64 | A | 66.9 | 0.9 (1 %) | 5.0 (8 %) | 56.9 (85 %) | 1.1 / 0.4 / 2.6 |
| KD-tree pairs | 512 | A | 521 | 5.1 (1 %) | 25.0 (5 %) | 480 (92 %) | 2.9 / 0.4 / 4.5 |

Law B is within a few % of law A (same files).

**One Γv action (N = 512, law A, original parameters), 8.1–8.4 ms:**

| part | time |
|---|---|
| real space | 4.2–4.5 ms |
| spread | 1.4 ms |
| forward + inverse FFT (M = 30) | 1.2 ms |
| influence multiply | 0.3 ms |
| gather | 1.3–1.4 ms |
| degree term | 0.03 ms |

The sum of the parts reproduces `GammaH.matvec` to 1.5e-16.

**Is the "56 Γv ≈ 96 %" figure true for the toy system?**
- **Not for the code as it was.** At N = 512 the Γv actions were only 23 % of the step. Operator construction took
  76 %: the real-space pair search enumerated all N(N−1)/2 pairs × (2⌈r_c/L⌉+3)³ image vectors whenever
  r_c = s/ξ = 5.86 ≥ L/2. At N = 512 that is 125 images: 16 M candidate vectors per step, 1.15 GB peak RSS.
- **Yes, after the exact fix (Section 2).** The Γv actions are 85 % (N = 64) and 92–93 % (N = 512) of the step,
  close to the project's 96 %.
- **Remaining parts:**
  - Lanczos overhead (reorthogonalisation, small eigenproblem, other) is 6 % at N = 64 and 1.5 % at N = 512;
  - forces are ≤ 1 % with the neighbour list.

## 2. Implementation changes that keep the operator and forces identical

**KD-tree pair search (`radial_kernels.real_pairs_tree`)**
- **Method:**
  - r_c < L/2: unchanged periodic KD-tree.
  - r_c ≥ L/2: a KD-tree on x is queried against a KD-tree on the (2m+1)³ replicas x_j − nL, with
    m = ⌊r_c/L⌋ + 1, keeping i < j.
  - Self images are skipped, as before.
- **Checks** (`test_radial_kernels.Enumeration`, and 36 cases with N = 64/216/512 and r_c from 0.18 L to 1.45 L):
  - identical pair-image sets;
  - displacement differences ≤ 3.6e-15;
  - Γ_h dense difference ≤ 6.2e-16 relative;
  - matvec difference ≤ 2.4e-16.
- **Speed and memory:**
  - pair search time at N = 512: 0.67–1.6 s → 11–12 ms;
  - peak RSS at N = 512: 1150 MB → 138 MB.
- **Default:** `pair_search="images"` stays the default of `FastFriction` and `Thermostat`, so the original
  thermostat code path is unchanged; `"tree"` is opt-in.

**Neighbour-list conservative force (`toy_dynamics.conservative_force_neighbor`, now the default in `run()`)**
- **Method:** periodic KD-tree within r_cut = 2.5σ, with the same `v2.pair_potential`, including the S′(r)u₀ switch
  term.
- **Checks** (42 configurations × 2 potentials, plus the unit test):
  - force difference ≤ 8.7e-16 relative;
  - energy difference ≤ 4.4e-16;
  - r_min identical.
- **Time per call:** 0.45 → 0.29 ms (N = 64), 27 → 2.4 ms (N = 512), 16 ms at N = 4096.
- **Reproducibility:** the summation order differs from the all-pair version, so a rerun of an earlier trajectory
  agrees only to round-off and then diverges chaotically. Pass `force_method="allpairs"` to reproduce runs from
  before this round bit for bit.

**No hidden O(N²) work in the step:**
- per-step diagnostics in `run()` (K, U, total momentum, r_min) are O(N) or come from the neighbour list;
- the Lanczos extreme Ritz values are O(rank);
- mesh stencil, degree term and spread/gather are O(N p³ + M³ log M).

The O(N²) pieces left are:
- the dense and direct reference methods, by design;
- the pair-binned VCCF and van Hove post-processing, which is analysis, not part of the step.

## 3. Staged operator search (`search.json`, `refine.json`)

The search ran in four stages:
1. **Cutoff.** For ξ ∈ {0.4, …, 1.8}, the smallest s on a 0.1 grid with error ≤ budget/2, on a fine mesh (η = 0.3,
   p = 7). Error = max relative spectral error over the double-well and LJ states at N = 64 (dense).
2. **Mesh.** For each p ∈ {5, 6, 7}, the coarsest mesh (largest η) with error ≤ budget.
3. **Mesh at a large box.** The N = 64 mesh choice does not transfer, because the rounding M = next_smooth(⌈…⌉)
   differs between boxes.
   - Example: A, ξ 0.85, s 3.9, η 0.95, p 7 gives 6.0e-7 at N = 64 but 3.4e-6 at N = 512.
   - So for the 3 cheapest (ξ, s) per budget and every p, η was rescanned at N = 512.
   - The spectral error there was computed matrix-free (ARPACK on Γ_h − Γ_ref). It equals the dense value for the
     baseline: 1.173e-7 vs 1.173e-7 (A) and 4.345e-8 vs 4.345e-8 (B).
4. **Transferable mesh.** The parameter that carries over is the realised η* = h k_c/π at N = 512. Any box with
   M = next_smooth(⌈L k_c/(η*π)⌉) has a mesh at least as fine.

**Re-checks of the selected sets:**
- at N = 64, on both states;
- at N = 512 (dense);
- at N = 1728 and 4096: Γv on 48 sampled particles against the direct full-periodic sum (`ranks_large.json`).

All are within budget except B 2e-7 at N = 64 (2.02e-7, 1 % over).

**Findings:**
- **Equal accuracy:** the operator parameters can hardly be improved. At N = 512 the cheapest set that meets the
  baseline accuracy has Γv 7.8 ms (A) and 7.0–8.6 ms (B), against 7.7–8.2 ms for the baseline, which is within the
  timing scatter.
- **What drives the cost:** the real-space part (∝ (s/ξ)³) and spread/gather (∝ p³) dominate; the FFT is minor. So
  ξ ≈ 0.85–1.0 is preferred to 0.7, and p = 5 or 6 pays off only at loose budgets.
- **Looser budgets:** Γv drops to 5.8–6.1 ms (A, 1e-6), 4.0–4.2 ms (B, 1e-6), and 2.0–2.6 ms (1e-4).

## 4. Lanczos ranks: noise and damping errors, separately from Γv (`ranks.json`, `modes.json`, `ranks_large.json`)

The errors are relative errors of Lanczos f(Γ_h)v with respect to dense f(Γ_h)v (Krylov part), for:
- the noise square root f = √(kT m(1 − e^{−2dtλ/m})) applied to z ~ N(0, I);
- the damping f = e^{−dtλ/m} applied to Maxwell momenta.

For N > 512 the reference is the rank-96 / rank-32 Lanczos result; its rank-80 / rank-24 difference is ≤ 4e-15.

**Spectrum on Range(Π)** (`lam` = dense eigenvalues, Ritz values for N > 512):

| law | N=64 | 512 | 1728 | 4096 | dt·λ_max |
|---|---|---|---|---|---|
| A | [1.73, 11.8] | [0.62, 11.9] | [0.45, 11.9] | [0.29, 12.0] | 0.06 |
| B | [30.0, 51.1] | [17.9, 54.8] | [13.5, 53.8] | [9.1, 54.8] | 0.27 |

λ_min falls with the box size, because the long-wavelength modes are the softest. The Krylov rank needed for the
square root therefore grows with N, mostly for law A.

**Ranks needed** (budget met at that N). "overall" uses only the random-vector relative error. "strict" also
requires the k = 2π/L plane-wave components J(k) = Σ u_i e^{ik·x_i}, which are the hydrodynamic modes of C_L and
C_T, and, for N ≤ 512, the lowest 3 / 12 eigenmodes:

| law, budget | N=64 overall / strict | N=512 | N=1728 | N=4096 |
|---|---|---|---|---|
| A 2e-7 | 16/4 / 16/4 | 20/4 / 32/4 | 24/4 / 32/4 | 32/4 / 40/4 |
| A 1e-6 | 12/3 / 16/4 | 20/3 / 24/4 | 20/3 / 32/4 | 24/3 / 40/4 |
| A 1e-4 | 8/2 / 10/3 | 10/2 / 16/3 | 10/2 / 20/3 | 12/2 / 24/3 |
| B 5e-8 | 8/4 / 8/5 | 10/5 / 12/5 | 12/5 / 16/5 | 12/5 / 16/5 |
| B 1e-6 | 6/4 / 6/4 | 8/4 / 10/5 | 8/4 / 12/5 | 10/4 / 16/5 |
| B 1e-4 | 6/3 / 6/3 | 6/3 / 8/4 | 6/3 / 8/4 | 6/3 / 10/4 |

**What this shows:**
- **The baseline ranks 40/16 are far beyond the operator error.** At N = 512:
  - noise Krylov error 2e-12 overall and 9e-11 in the k = 2π/L components;
  - damping 4e-15;
  - the operator error is 1e-7.
  - Against dense f(Γ_ref), the total error is floored at 4e-8 (A) and 8e-9 (B) by the operator, whatever the rank.
- **Damping rank 16 is oversized at every N.** Rank 4 (A) or 5 (B) meets the strictest budget, because
  dt·λ_max ≤ 0.27 makes e^{−dtλ} nearly a low-degree polynomial on the spectrum.
- **Noise rank 40 is needed only for law A at N = 4096 under the strict rule.** At N ≤ 512, ranks 32 (A) and 12 (B)
  meet the equal-accuracy budgets. The user-suggested grid noise 24/32/40 and damping 8/12/16 was extended down to
  6 and 2. The low ranks were measured, not assumed.
- **The overall error understates the error of the slowest modes** by 10–100× at the same rank. For example, A at
  N = 512 with rank 20: 1e-7 overall vs 4e-6 at k = 2π/L. The strict rule is used for the recommendations, because
  the toy analysis measures exactly those modes.

**Structure of every selected Γ_h** (N = 64 and 512, `slowmodes.json`):
- symmetry residual ≤ 2e-17;
- momentum null-space residual ≤ 3e-15;
- λ_min > 0 on Range(Π), the same as the reference to the digits shown;
- all Ritz values positive;
- k = 0 retained.

**Friction–noise matching:**
- Damping and noise are built from the same Γ_h instance in every step, so they are matched exactly at the
  operator level.
- The only mismatch is the Krylov truncation, bounded by the damping and noise errors above.
- The combined O-step error |R̃p + S̃z − (Rp + Sz)|/|Rp + Sz| was also tabulated for every rank pair
  (`ranks.json`, key `ostep`).

## 5. Short dynamics, candidate vs original version (`validate.json`, `validate_summary.csv`, `validate_paths/`)

**Setup:**
- double-well, N = 64;
- 5 initial states (v2 checkpoints seeds 101–105) × 2 noise paths;
- the same initial q, p and the same noise seed (`SeedSequence([20261009, state, path])`) for the original code
  (image pairs, 40/16) and each candidate;
- 400 steps;
- observables at t = 0.5, 1, 2: K/N, U/N, ⟨|v|⁴⟩ and MSD;
- per state: mean over paths; across states: mean and Student-t 95 % CI (df = 4).

| candidate | largest relative mean difference | CIs excluding zero (relative) | pathwise RMS / path-to-path SD of the baseline |
|---|---|---|---|
| A 2e-7 | 5.1e-7 (⟨|v|⁴⟩, t=2) | MSD t=0.5: −1.5e-8 [−2.7e-8, −3.6e-9]; t=1: −2.4e-8 [−4.1e-8, −6.7e-9] | ≤ 2.0e-5 |
| A 1e-6 | 1.7e-6 (K/N, t=2) | none | ≤ 4.7e-5 |
| A 1e-4 | 1.1e-4 (MSD, t=2) | MSD t=2: +1.1e-4 [+7.9e-5, +1.5e-4] | ≤ 3.0e-3 |
| B 5e-8 | 1.8e-9 (MSD, t=2) | none | ≤ 3.8e-8 |
| B 1e-6 | 7.9e-8 (MSD, t=1) | MSD t=0.5: +5.1e-8; t=1: +7.9e-8 | ≤ 6.9e-7 |
| B 1e-4 | 6.0e-6 (MSD, t=0.5) | MSD at t = 0.5, 1, 2: +5–6e-6 | ≤ 5.5e-5 |

**How to read the dynamics results:**
- **Detected differences are systematic offsets.** The MSD offsets that exclude zero match the operator error of
  each budget: the coupled paths make offsets of 1e-8–1e-4 detectable. A CI that includes zero is not a statement
  of equivalence; there it only bounds the difference.
- **Scale.** All differences are 2–8 orders of magnitude below:
  - the path-to-path spread;
  - the resolved 2dt time-step bias of the earlier fixed-time study (K/N −0.43 % at t = 0.5).
- **Coverage.** This is a fixed-time test at t ≤ 2 for N = 64 double-well only. It says nothing about the long-time
  law-A diffusion difference (+5.6 % to +8.0 %, unresolved; `../TOY_INTEGRATION_REPORT.md`, Layer 4), and does not
  claim to resolve it.

Wall time per 400-step path in this pool of 4 workers:
- original: 37.4 s;
- A sets: 15.0 / 10.3 / 4.7 s;
- B sets: 7.6 / 4.8 / 2.9 s.

## 6. Fair timing (`timing.json`, `timing_table.md`, `step_time_vs_N.png`)

**Conditions:**
- **Hardware:** Intel Xeon @ 2.30 GHz (4 cores).
- **Threads and processes:** every case runs alone in its own process, with OPENBLAS/OMP/MKL threads = 1.
- **State and step:** fixed density, double-well state, the same step code (neighbour-list forces for all
  methods).
- **Repeats:** 2 warm-up + 5 timed steps; dense at N = 512 2 timed; N ≥ 1728: 1 + 3.
- **Statistics:** median, with the min–max in `timing_table.md`.
- **Precompute:** one-off, not in the step: mesh and influence function for PPPM, the Chebyshev far-field fit
  (~1–2.5 s) for dense and direct.
- **Memory:** peak RSS of the process, including ~110 MB of Python/NumPy.

**Methods (all on the same full periodic model with k = 0, no minimum image):**
- **dense:** v2 `LatticeFriction` matrix (near images + Chebyshev far field) + eigendecomposition on Range(Π);
- **direct:** the same pair tensors for all pairs + matrix-free Lanczos;
- **PPPM:** + Lanczos.

**Results:**
- **N = 64:** dense 16.7–17.0 ms; direct 18.9–19.4 ms (Γv 0.10 ms); PPPM original 82–90 ms.
- **N = 256:** dense ≈ 445 ms; direct ≈ 380 ms; PPPM original ≈ 430 ms; PPPM equal accuracy 149 (A) and 80 (B) ms.
- **N = 512:**
  - dense 2.5–2.6 s;
  - direct 1.5 s, of which 1.1–1.2 s is building the pair tensors. Its Γv, 6.5 ms, is cheaper than PPPM's 7.8 ms.
  - PPPM original 1.2–1.6 s;
  - PPPM equal accuracy 314 (A) and 172 (B) ms.
- **N = 1728:** direct 20.6–21.0 s/step (O(N²)); PPPM original 1.7–1.8 s; equal accuracy 1.09 (A) and 0.75 (B) s.
- **N = 4096:** PPPM original 4.6–4.7 s; equal accuracy 3.8 (A, ranks 40/4) and 1.9 (B) s.

Dense was not run beyond N = 512, and direct not beyond 1728.

## 7. Verified vs estimated

**Verified:**
- identical operator and forces for the two enumeration changes;
- operator errors of every selected set:
  - spectral at N = 64 (2 states) and 512;
  - Γv on 48 sampled rows at N = 1728 and 4096;
- the Krylov noise and damping errors per rank at N = 64–4096 (overall, k = 2π/L, lowest eigenmodes for N ≤ 512);
- symmetry, null space, PSD, k = 0;
- t ≤ 2 paired dynamics at N = 64 (double-well, laws A and B);
- all step times and speedups in Section 0, measured as described in Section 6.

**Estimated or not verified:**
- the "+ ranks" decomposition column in Section 0;
- B 2e-7: not timed and not run dynamically, and 1 % over budget at N = 64;
- ranks for N > 4096: they will keep growing for law A;
- LJ dynamics with the new sets: only static LJ operator checks at N = 64;
- dynamics at N > 64;
- any statement about long-time transport.

The operator parameters were tuned on double-well and LJ states at the toy density only.

## 8. Remaining bottlenecks

**Equal accuracy, N ≥ 256:**
- the cost is ≈ 90 % Γv actions;
- within one Γv at N = 512, the real-space NumPy gather/einsum/bincount is ~55 % and spread/gather ~35 %;
- further gains would need a different implementation, for example a per-step sparse real-space block matrix or
  compiled kernels, rather than different parameters. Not tried; no gain is claimed.

**Law A at large N:** the strict noise rank grows from 32 (N = 512) to 40 (N = 4096) as λ_min falls. At N = 4096
the equal-accuracy speedup over the original is therefore only ×1.24.

**N ≈ 64:** the dense reference is cheaper than PPPM at equal accuracy and has no approximation error.

## 9. Commands

```
python3 toy_cost_opt.py --stage profile --N 64 512 --reps 4 --pair-search images   # (first run: profile_images.json)
python3 toy_cost_opt.py --stage profile --N 64 512 --reps 4 --pair-search tree
python3 toy_cost_opt.py --stage search --laws A B --timing-N 64 512
python3 toy_cost_opt.py --stage search --laws B --tols 5e-8 --timing-N 64 512
python3 toy_cost_opt.py --stage ranks --laws A --tols 2e-7 1e-6 1e-4 --N 64 512 --vectors 6 --top 2
python3 toy_cost_opt.py --stage ranks --laws B --tols 5e-8 2e-7 1e-6 1e-4 --N 64 512 --vectors 6 --top 2
python3 toy_cost_opt.py --stage refine --laws A B --tols 5e-8 2e-7 1e-6 1e-4 --N 512 --top 3
#   candidates.json: cheapest refined set per law/budget (ranks then set by the strict rule from modes.json)
python3 toy_cost_opt.py --stage slowmodes --candidates cost_optimization_results/candidates.json --N 64 512 --vectors 4
python3 toy_cost_opt.py --stage ranks_large --candidates cost_optimization_results/candidates.json --N 1728 4096 --vectors 3
python3 toy_cost_opt.py --stage modes --candidates cost_optimization_results/candidates.json --laws A B --N 64 512 1728 4096 --vectors 4
python3 toy_cost_opt.py --stage validate --candidates cost_optimization_results/candidates_validate.json --states 101 102 103 104 105 --paths 2 --workers 4
python3 toy_cost_opt.py --stage timing --candidates cost_optimization_results/candidates_timing.json --N 64 256 512 --reps 5 --warm 2
python3 toy_cost_opt.py --stage timing --candidates cost_optimization_results/candidates_timing.json --N 1728 4096 --reps 3 --warm 1 --direct-max 1728
python3 toy_cost_opt.py --stage table
```

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. `validate_paths/*.json` holds one file per
path: seed, initial state, observables, thermostat record.

## 10. Production configuration and verification (amendment, 2026-10-08)

The study's sets are packaged as the named configuration `costopt_hiacc` in `../toy_configs.py`:
- law A: the 2e-7 set, ξ 0.85, s 4.1, η* 0.6779, p 7;
- law B: the 5e-8 set, ξ 1.0, s 3.9, η* 0.6828, p 7;
- KD-tree pairs and neighbour-list forces.

The original version is kept as `baseline_hiacc`. The runner is `../toy_run.py`; the HPC files are in `../hpc/`.

**Static confirmation at exactly the configured parameters** (`production_config_verification.json`, written by
`../verify_production_configs.py`):
- **Cache key.** Each entry is keyed by the configuration's operator hash: config, law, N, dt, model, PPPM set, pair
  search and ranks. The protocol and the budget are part of the key too, so a parameter change always recomputes.
- **Test states** at N = 256 and 512:
  - the earlier state, a replicated v2 state or a jittered lattice;
  - a liquid-like double-well state, whose positions come from scalar Langevin dynamics with conservative forces
    only (t = 50 from fcc / sc, `verification_states/`). Positions are canonical whatever the friction, so the
    friction model plays no part in preparing them.
- **Checks** (`checks` / `passed` in each entry): operator error, Krylov errors at the configured ranks (overall,
  k = 2π/L, lowest 3 / 12 eigenmodes) and the O-step error. Structure: symmetry ≤ 2e-17, null space ≤ 3e-15,
  λ_min > 0, k = 0 retained, literal matvec equals dense assembly.

**Spectra on Range(Π), λ_min / λ_max of Γ_h:**

| law, N | earlier test state | liquid-like state |
|---|---|---|
| A 256 | 1.78 / 6.8 | 0.456 / 18.5 |
| A 512 | 0.620 / 11.9 | 0.215 / 21.5 |
| B 256 | 28.6 / 49.2 | 15.7 / 78.3 |
| B 512 | 17.9 / 54.8 | 9.31 / 96.6 |

The liquid-like state is much harder for Lanczos.

**Rank scan** (`production_rank_scan.json`): the ranks needed to meet the budget on every state are:

| law | N=64 | N=256 | N=512 |
|---|---|---|---|
| A (2e-7) | 16/4 | 32/4 | 32/5 |
| B (5e-8) | 8/5 | 16/6 | 20/6 |

The study's B ranks 12/5 fail: noise error 1.6e-5 at N = 512, against a budget of 5e-8. A 32/4 meets the budget
at N = 512 with little margin: 1.5e-7 against 2e-7. At N = 1728, A 32/4 fails on the liquid-like state (3.5e-5 at
k = 2π/L); `costopt_hiacc_A_N1728` is kept as evidence only.

**Configured ranks:**
- **Choice.** The smallest ranks of the scan that meet budget/10 on every state at that N:
  - A: 32/5 (N ≤ 256) and 40/5 (N ≤ 512);
  - B: 16/6 and 24/8.
- **Why the factor 10.** It is a margin for production states that are more clustered than the test state.
- **Refusal above N = 512.** No larger-N rows have been verified on liquid-like states.
- **Re-verification at the budget** with independent vectors:

| config, law | N | ranks | operator error | max Krylov error | result |
|---|---|---|---|---|---|
| costopt A | 64 | 32/5 | 1.48e-7 | 8.2e-12 | PASS |
| costopt A | 256 | 32/5 | 1.07e-7 | 1.5e-9 | PASS |
| costopt A | 512 | 40/5 | 1.36e-7 | 1.3e-9 | PASS |
| costopt B | 64 | 16/6 | 3.58e-8 | 2.1e-12 | PASS |
| costopt B | 256 | 16/6 | 4.26e-8 | 5.6e-9 | PASS |
| costopt B | 512 | 24/8 | 3.53e-8 | 2.3e-10 | PASS |
| baseline A | 64 / 256 / 512 | 40/16 | 1.27e-7 / 1.38e-7 / 1.17e-7 | ≤ 1.3e-9 | PASS (checked against 2e-7) |
| baseline B | 64 / 256 / 512 | 40/16 | 4.55e-8 / 4.41e-8 / 4.37e-8 | ≤ 2.1e-14 | PASS (checked against 5e-8) |

**Structure of the liquid-like state.** It shows strong long-wavelength density fluctuations:
- max S(k) over the wavevectors k = 2πn/L with 0 < |n| ≤ 2: 113 at N = 256 and 236 at N = 512;
- neighbours within 1.6σ: 19–21, against 12.3 in the v2 N = 64 state.

This is consistent with clustering or liquid–vapour separation at this density and temperature, which a 64-particle
box cannot show. For LJ, T = 0.7 is below the critical temperature and ρ = 0.385 lies between the coexisting
densities (literature values). It was not established here: there is one preparation per N and no test of
equilibration.

For production at N = 256 and 512 this means two things:
- The system may become inhomogeneous during burn-in, and the physics of the toy model then differs from the N = 64
  picture.
- The friction spectrum may become harder than the verified states.

The runner therefore monitors, every 500 steps, the Lanczos error estimates (overall and at k = 2π/L), the extreme
Ritz values and max S(k = 2π/L). It warns when an estimate exceeds the budget or the spectrum falls below the
verified λ_min.

**Measured cost of the production configuration** (`../hpc/local_pilot_20261008/`):
- N = 512, through `toy_run.py`, on one machine (Xeon @ 2.10 GHz, slower than the VM of Section 6), 1 thread:

  | law | production config | original version | speedup |
  |---|---|---|---|
  | A | 666 ms | 2324 ms | ×3.5 |
  | B | 458 ms | 2347 ms | ×5.1 |

- Peak RSS: 125 MB against 1135 MB.
- These replace the Section 0 speedups (A ×5.2, B ×7.1) for the production configuration. Those were for the
  lower study ranks on a faster VM.
- At equal accuracy the A operator parameters remain within timing noise of the original operator (Section 3). The
  gains come from the pair search and from damping rank 16 → 5.

**Cache keys of this study.**
- `toy_cost_opt.py` now puts a hash of the actual parameters into the keys of `slowmodes.json`, `ranks_large.json`,
  `modes.json` and the `validate_paths/` file names. Existing entries were migrated by the hash of the parameters
  stored in each entry.
- The `slowmodes.json` entries computed at early ranks (for example A 20/4) carry a `superseded` note. Their static
  parts are rank-independent and still feed `cost_table.csv`, which regenerates identically, as does
  `validate.json`.

