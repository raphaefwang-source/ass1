# PPPM/FFT vs direct full-periodic sum: short benchmark

Script: `../bench_fft_vs_direct.py`. Every number below is read from the CSV/JSON files in this directory.

**The question.** From which N does PPPM/FFT beat the direct sum, and by how much? This is not "optimised vs old
PPPM".

**Fixed for this round:**
- the production configuration of a9d2ae6 is unchanged; there was no parameter tuning;
- no long runs, no random batch.

**Compute used:** 0.64 core-hours of the 2 core-hour budget (`budget_log.json`).

## Setup

**Shared by all three methods:**
- the same model: pure longitudinal friction, full periodic images, k = 0 retained, g(r_ref) = 0.5, κ = 0.7,
  r_ref = 1.3, kT = 0.7, m = 1;
- dt = 0.005, density 64/5.5³;
- the same BAOAB step as `toy_dynamics.run`, with one neighbour-list double-well force evaluation per step and the
  same noise stream.

| method | per step |
|---|---|
| **dense** | rebuild the v2 `LatticeFriction` matrix (33 near images + Chebyshev far field, k = 0), then `eigh` and exact f(Γ) on Range(Π). The lean implementation agrees with the project's `DenseRef` to ≤ 1e-10 at N ≤ 512; not cross-checked at 864 or 1728. |
| **direct** | rebuild the same v2 pair tensors for all N(N−1)/2 pairs once per O step, reuse them in every Lanczos action (`toy_cost_opt.DirectLatticeOp`), then `test_true_dynamics.lanczos_apply`. |
| **pppm** | rebuild the production `costopt_hiacc` operator (KD-tree real space + PPPM mesh, k = 0), then the same Lanczos routine. |

**Lanczos ranks** (noise/damping), identical for direct and pppm:

| N | law A | law B | source |
|---|---|---|---|
| 64, 108, 256 | 32/5 | 16/6 | production configuration |
| 512 | 40/5 | 24/8 | production configuration |
| 864 | 48/5 | 32/8 | chosen here by the production rule (Krylov errors ≤ budget/10 on every state, direct method, exact reference) |
| 1728 | 64/5 | 32/8 | same rule (`ranks_selected.json`) |

N = 864 and 1728 are outside the production configuration, which refuses N > 512.

**Test configurations** (`states.json`, `states/`). Two per N, used by every method:

| state | N | how it was made |
|---|---|---|
| v2 equilibrium | 64 | v2 restart state |
| replicated v2 | 512, 1728 | 2×2×2 or 3×3×3 copies + 0.02 jitter |
| jittered lattice | 108, 256, 864 | jittered lattice |
| clustered | all | the saved liquid-like state: scalar Langevin with conservative forces only, t = 50 from fcc/sc |

Structure of the clustered states:
- max S(k) at |n| ≤ 2: 27, 43, 113, 236, 419 and 587 at N = 64, 108, 256, 512, 864 and 1728;
- 14–21 neighbours within 1.6σ, against 11–12 in the v2 state.

**Timing protocol** (`timing_raw.csv`, `gamma_v_raw.csv`):
- **Machine:** Intel Xeon @ 2.10 GHz; 1 thread (OPENBLAS/OMP/MKL = 1); Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1.
- **Isolation:** one subprocess per (method, law, N, state); nothing else running.
- **Steps:** 2 warm-up + 5 timed steps; at N = 1728, 1 + 3; dense at N > 512, 1 + 2 on the clustered state only,
  because its cost does not depend on the configuration.
- **Medians** are reported.
- **Γv:** timed after a build, 15 samples (7 at N = 1728), median.
- **One-off initialisation** is reported separately and never counted in a step: the Chebyshev far-field fit for
  dense/direct, 0.1–6 s; the mesh and influence function for PPPM, 0.01–0.07 s.
- **What each step includes:** the per-step rebuild of the position-dependent operator (dense matrix, direct pair
  tensors, PPPM operator).
- **Peak RSS:** the whole process; Python + NumPy alone take 106 MB.
- **Accuracy checks and diagnostics** are timed separately: `accuracy.json`, field `cpu_s`.

## Accuracy at the timed ranks (`accuracy_at_timed_ranks.csv`)

**Budgets:** the production operator budgets, 2e-7 (A) and 5e-8 (B).

**Pass rules:**
- **direct:** its matvec equals the dense reference to ≤ 1e-15. Its noise and damping Krylov errors must meet the
  budget; each error is the maximum over four measures (overall, k = 2π/L, lowest 3 eigenmodes, lowest 12
  eigenmodes), taken against exact f(Γ_ref).
- **pppm:** its spectral operator error must meet the budget, and so must its Krylov errors.
  - The operator error is computed matrix-free (ARPACK on Γ_h − Γ_ref) at every N. At N ≤ 512 it was also computed
    from the dense PPPM matrix; the two agree to ≤ 1.1e-6 relative.
  - At N ≤ 512 the Krylov errors are taken against dense f(Γ_h), from eigh of the dense PPPM matrix.
  - At N = 864 and 1728 they are taken against a converged rank-96 / rank-32 Lanczos result of Γ_h, whose rank-80 /
    rank-24 difference is ≤ 3e-15. The low-mode subspaces there come from Γ_ref.
  - *Correction (2026-10-08):* the earlier text said "dense for N ≤ 864"; the code (`accuracy_case`) uses dense only
    for N ≤ 512.
- **dense:** the null-space separation must be clean (≤ 1e-10); at N ≤ 512 also agreement with `DenseRef`.

**Results:**
- **All methods pass on all 24 cases except two:** PPPM law B on the jittered-lattice states at N = 108 and 864.
  Their operator errors are 5.67e-8 and 5.98e-8, above the 5e-8 budget by 13–20 %. The clustered states at the
  same N pass (3.7e-8, 3.4e-8). These two cases are excluded from the speedup conclusions below.
- **Largest errors that pass:**
  - Krylov: 3.6e-9;
  - PPPM operator: 1.6e-7 (A) and 4.3e-8 (B);
  - direct matvec vs dense: ≤ 1.6e-15.

**Spectra.** On the clustered states λ_min falls to 0.11–0.21 (A) and 4.9–9.3 (B) at N ≥ 512, against 0.45–0.62 and
13.5–17.9 on the replicated states. This is why the ranks grow with N.

**Note for production.** These results were not acted on in this round. N = 108 is allowed by the production
configuration (N ≤ 256 row, with a warning). On an ordered state there, the B operator misses its 5e-8 budget by 13 %.
The production verification covers N = 64, 256 and 512 only.

## Results

### Full BAOAB step, T(direct + Lanczos) / T(PPPM + Lanczos)

Values above 1 mean PPPM is faster. Rows marked † fail the PPPM accuracy check and carry no claim.

| law | N | state | step | O step | build | one Γv | dense / PPPM (step) | ESTIMATE: direct without its build / PPPM (O step)* |
|---|---|---|---|---|---|---|---|---|
| A | 64 | v2 | 0.33 | 0.33 | 3 | 0.11 | 0.53 | 0.11 |
| A | 64 | clustered | 0.35 | 0.34 | 3 | 0.11 | 0.30 | 0.12 |
| A | 108 | lattice | 0.96 | 0.96 | 11 | 0.21 | 1.10 | 0.18 |
| A | 108 | clustered | 0.79 | 0.79 | 9 | 0.12 | 0.70 | 0.14 |
| A | 256 | lattice | 2.28 | 2.29 | 27 | 0.36 | 2.51 | 0.40 |
| A | 256 | clustered | 1.76 | 1.76 | 16 | 0.37 | 2.20 | 0.33 |
| A | 512 | replicated | 4.08 | 4.10 | 70 | 0.81 | 4.95 | 0.73 |
| A | 512 | clustered | 3.26 | 3.28 | 62 | 0.65 | 4.61 | 0.62 |
| A | 864 | lattice | 6.31 | 6.33 | 100 | 1.40 | — | 1.43 |
| A | 864 | clustered | 3.79 | 3.81 | 93 | 0.92 | 5.09 | 0.86 |
| A | 1728 | replicated | 9.88 | 9.93 | 207 | 3.38 | — | 3.37 |
| A | 1728 | clustered | 7.85 | 7.89 | 184 | 2.53 | 12.07 | 2.68 |
| B | 64 | v2 | 0.58 | 0.58 | 4 | 0.07 | 0.76 | 0.15 |
| B | 64 | clustered | 0.55 | 0.55 | 4 | 0.07 | 0.54 | 0.15 |
| B | 108 | lattice † | 1.51 | 1.51 | 12 | 0.26 | 1.47 | 0.19 |
| B | 108 | clustered | 1.56 | 1.58 | 13 | 0.27 | 1.64 | 0.21 |
| B | 256 | lattice | 3.75 | 3.78 | 36 | 0.47 | 4.45 | 0.53 |
| B | 256 | clustered | 3.56 | 3.61 | 41 | 0.37 | 4.08 | 0.37 |
| B | 512 | replicated | 3.99 | 4.03 | 62 | 0.82 | 5.95 | 0.71 |
| B | 512 | clustered | 3.70 | 3.72 | 63 | 0.57 | 4.33 | 0.67 |
| B | 864 | lattice † | 6.29 | 6.32 | 124 | 1.03 | — | 1.16 |
| B | 864 | clustered | 5.92 | 5.99 | 96 | 0.98 | 8.41 | 1.14 |
| B | 1728 | replicated | 15.90 | 16.00 | 270 | 4.22 | — | 3.62 |
| B | 1728 | clustered | 11.71 | 11.81 | 213 | 2.81 | 20.72 | 2.60 |

\* Estimate, not a measurement: (direct O step − direct build) / PPPM O step, a derived bound for a hypothetical
direct method whose pair tensors cost nothing. See "Estimates" below.

**Absolute times in ms** (`timing_summary.csv`). Each cell gives the step time, with (build / one Γv) for direct and
PPPM and (build + eigh) for dense:

| law | N | state | dense step (build + eigh) | direct step (build / Γv) | PPPM step (build / Γv) | peak RSS MB: dense / direct / PPPM |
|---|---|---|---|---|---|---|
| A | 64 | clustered | 17.9 (13.5 + 3.4) | 20.5 (13.0 / 0.12) | 58.7 (4.3 / 1.15) | 499 / 499 / 111 |
| A | 256 | clustered | 501 (404 + 91) | 400 (323 / 1.9) | 228 (19.6 / 5.2) | 483 / 483 / 120 |
| A | 512 | clustered | 2581 (1801 + 664) | 1827 (1477 / 7.7) | 560 (24 / 11.8) | 382 / 382 / 135 |
| A | 864 | clustered | 7629 (4798 + 2810) | 5683 (4392 / 27.7) | 1499 (47 / 30.0) | 480 / 346 / 153 |
| A | 1728 | clustered | 45720 (21713 + 23900) | 29747 (19638 / 124) | 3789 (107 / 49.0) | 1559 / 536 / 198 |
| B | 64 | clustered | 17.5 (13.4 + 3.3) | 17.9 (12.6 / 0.13) | 32.5 (3.4 / 1.85) | 499 / 499 / 111 |
| B | 256 | clustered | 561 (419 + 97) | 490 (438 / 1.9) | 137 (10.8 / 5.0) | 499 / 499 / 119 |
| B | 512 | clustered | 2290 (1565 + 713) | 1955 (1597 / 10.9) | 529 (25 / 19.1) | 483 / 483 / 130 |
| B | 864 | clustered | 7904 (4851 + 3030) | 5567 (4495 / 22.5) | 940 (47 / 22.9) | 480 / 382 / 148 |
| B | 1728 | clustered | 44820 (21388 + 23300) | 25323 (19723 / 118) | 2163 (92 / 42.1) | 1559 / 536 / 181 |

The other states are in the CSV. Figures:
- `step_time_vs_N.png`: full step against N, three methods, both states. The two PPPM cases that fail the accuracy
  check (law B, lattice state, N = 108 and 864) are not on the curves; they are drawn as separate crosses labelled
  "fails accuracy (excluded)";
- `build_vs_gamma_v.png`: build and Γv components, clustered state.

## Answers

1. **Where the advantage starts (full step, accuracy-passing cases; measured):**
   - **N = 64:** PPPM is slower. The direct sum is ×2.9–3.0 faster (law A) and ×1.7–1.8 faster (law B); dense is
     about as fast as direct.
   - **Law B:** PPPM is ahead at N = 108 (×1.56 on the clustered state; the lattice state fails the PPPM accuracy
     check and is not counted).
   - **Law A:** at N = 108 PPPM is still level or behind (×0.96 lattice, ×0.79 clustered) and is ahead at 256.
   - **Measured bracket of the crossover:** 64 < N ≤ 108 for B, and 108 < N ≤ 256 for A. A point estimate inside the
     bracket is under "Estimates" below.
2. **How much faster** (full step, PPPM over direct):

   | N | law A | law B |
   |---|---|---|
   | 256 | ×1.8–2.3 | ×3.6–3.8 |
   | 512 | ×3.3–4.1 | ×3.7–4.0 |
   | 864 | ×3.8–6.3 | ×5.9 (clustered only) |
   | 1728 | ×7.9–9.9 | ×11.7–15.9 |

   Against dense, PPPM is ×4.3–6.0 faster at 512 and ×12–21 faster at 1728.
3. **Is one Γv still slower at N = 512? Yes.**
   - Measured: PPPM 9.0–19.1 ms against direct 7.3–10.9 ms, i.e. direct/PPPM = 0.57–0.82.
   - **Parity near N ≈ 864:** 0.92–1.40 (A) and 0.98–1.03 (B). The clustered states push the crossover up: they
     have more real-space pairs within r_c, and their PPPM Γv is 1.1–2.1× that of the other state.
   - **N = 1728:** PPPM's Γv is ×2.5–4.2 faster.
4. **Does including construction change the conclusion? Yes, completely, at N ≤ 864.**
   - **The direct cost is the build.** Each step the direct method rebuilds the full-periodic pair tensors for all
     pairs. That is O(N²) and about 11–13 µs per pair: 33 near images plus a Chebyshev far field. It takes 63–89 % of
     its step (1.3–1.7 s at N = 512), against 22–25 ms for the PPPM build.
   - **What decides the full-step crossover** (measured bracket 64–256) is the cost of rebuilding the periodic
     operator, not the FFT matvec. The measured Γv ratios show this (item 3): the direct Γv is cheaper up to N = 512
     and about equal at 864. That crossover is therefore specific to this direct implementation.

### Estimates (not measurements)

These numbers are derived from the measured values. They are not results of a run of any method.
- **Point estimates of the full-step crossover:** log interpolation of the measured T(direct)/T(PPPM) between the
  bracketing N gives N ≈ 85 for law B (clustered state) and N ≈ 110–140 for law A (lattice and clustered states).
  The true crossover can lie anywhere inside the measured brackets above.
- **"Direct build free" bound:** the last column of the ratio table is (direct O step − direct build) / PPPM O step,
  i.e. a hypothetical direct method whose periodic pair tensors cost nothing. By this bound, such a method would beat
  PPPM at N = 512 (both laws) and at 864 on the clustered law-A state. PPPM would win only from around N ≈ 900, and
  clearly at 1728 (×2.6–3.6).
  - No such implementation was built or timed.
  - Whether a cheaper evaluation of the periodic pair tensors (e.g. tabulation) would move the full-step crossover
    towards N ≈ 900 is not measured.
5. **Memory.** Peak RSS is 111–198 MB for PPPM, 346–536 MB for direct and 480–1559 MB for dense; 106 MB of each is
   Python + NumPy. At N ≤ 512, the direct and dense peaks come from the chunked Chebyshev temporaries; at 1728 dense
   needs 1.56 GB.

**Caveats:**
- The timings come from one shared VM: repeated steps vary by up to about ±15 % (min–max in the CSV). Ratios between
  0.9 and 1.1 should be read as parity.
- The speedups cover only the listed cases that pass the accuracy check. The two B lattice cases are excluded.
- The ranks at N = 864 and 1728 come from this round's rule. They are not part of the production configuration.

## Commands

```bash
cd ewald_longitudinal
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 bench_fft_vs_direct.py --stage states   --N 64 108 256 512 864 1728   # test configurations + structure stats
python3 bench_fft_vs_direct.py --stage accuracy --N 64 108 256 512 864 1728   # checks; also selects ranks for N > 512
python3 bench_fft_vs_direct.py --stage timing   --N 64 108 256 512 864 1728   # one subprocess per case; resumes
python3 bench_fft_vs_direct.py --stage report                                 # summary CSVs, ratios, figures
```

The stages cache their results (`accuracy.json`, `timing_raw.json`); delete those files to rerun from scratch.
`--N 64 256 512 1728` reproduces the first pass of this round. N = 108 and 864 were added to locate the two
crossovers.
