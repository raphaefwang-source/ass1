# Seed-paired check of the fast operator against the dense reference (production_t60)

Script: `analyze_toy_paired.py`. Outputs in this directory: `paired_summary.csv`, `lag_scan.csv`,
`window_summary.csv`, `operator_along_trajectories.csv` and `status.json`. All rows list the five per-seed values
of both methods and of d.

## Method

- **Pairing.** Run i of both methods starts from the same state (the v2 final state of seed i) and consumes the same
  Gaussian stream (seed i + 3000). The analysis therefore uses d_i = fast_i − reference_i for i = 101…105, even
  after the paths have separated. The two groups are never treated as independent.
- **Statistics.** mean d; SEM = std(d_i, ddof = 1)/√5; 95% CI = mean d ± t₀.₉₇₅,₄ · SEM with t₀.₉₇₅,₄ = 2.776.
  The relative difference is d/|mean reference|, so its sign is the sign of d (U/N is negative).
- **What is reported.** The paired t statistic (df = 4), the count of d_i > 0, and the exact sign-test p-value
  (≥ 0.0625 for n = 5). t is not translated into "σ".
- **Multiplicity.** About 30 full-run tests, 48 lag-scan rows and 80 window rows were computed. Isolated small
  p-values are expected by chance.
- **Equivalence.** A non-significant result is not read as equivalence. The CI shows which differences the data
  still allow.
- **Estimators** (same as before):
  - Green-Kubo D = ⟨|v|²⟩/3 ∫₀^τ VACF;
  - MSD D = the slope of the MSD on [τ/2, τ], divided by 6.
  - The full-run values use the cached observables (300 time origins).
  - The windows use all origins, via FFT; this was checked against direct sums to 4e-14.
- **Cache-only operation.** `paired_summary.csv` and `lag_scan.csv` need only the committed cache
  (`../observables/`). This was tested by pointing `--raw-dir` at a missing directory:
  - both files were recomputed byte-identically;
  - the window and operator stages were skipped, with the 40 missing files listed in `status.json`;
  - the existing `window_summary.csv` was left unchanged.

## 1. Full-run paired differences

| case | quantity | mean reference | mean d = fast − ref | d / abs(ref) [95% CI] | paired t (df 4) | d > 0 |
|---|---|---|---|---|---|---|
| lj_A | D_green_kubo | 0.06092 | -0.00266 | -4.36% [-8.54, -0.18]% | -2.89 | 1/5 |
| lj_A | D_msd | 0.06026 | -0.00231 | -3.83% [-9.50, +1.85]% | -1.87 | 1/5 |
| lj_A | T_mean | 0.6999 | +0.000303 | +0.04% [-0.76, +0.85]% | +0.15 | 2/5 |
| lj_A | U_per_particle | -3.170 | -0.0337 | -1.06% [-4.42, +2.30]% | -0.88 | 2/5 |
| lj_A | tau_T | 0.2408 | -0.0175 | -7.26% [-20.48, +5.96]% | -1.53 | 2/5 |
| lj_A | rdf_peak | 3.269 | +0.0396 | +1.21% [-2.53, +4.95]% | +0.90 | 3/5 |
| lj_B | D_green_kubo | 0.01004 | -4.13e-08 | -0.00% [-0.00, +0.00]% | -1.85 | 2/5 |
| lj_B | D_msd | 0.009275 | +9.97e-08 | +0.00% [-0.00, +0.00]% | +1.01 | 3/5 |
| lj_B | T_mean | 0.6996 | -6.43e-08 | -0.00% [-0.00, +0.00]% | -1.23 | 1/5 |
| lj_B | U_per_particle | -3.145 | -2.28e-06 | -0.00% [-0.00, +0.00]% | -0.88 | 1/5 |
| lj_B | tau_T | 0.01753 | +4.48e-07 | +0.00% [-0.00, +0.01]% | +1.82 | 5/5 |
| lj_B | rdf_peak | 3.253 | -0.00053 | -0.02% [-0.08, +0.05]% | -0.72 | 1/5 |
| double_well_A | D_green_kubo | 0.01937 | +0.00156 | +8.04% [+4.08, +12.01]% | +5.63 | 5/5 |
| double_well_A | D_msd | 0.02006 | +0.00112 | +5.56% [+2.58, +8.54]% | +5.17 | 5/5 |
| double_well_A | T_mean | 0.6993 | +0.00144 | +0.21% [-0.34, +0.76]% | +1.04 | 3/5 |
| double_well_A | U_per_particle | -5.756 | -0.00948 | -0.16% [-1.36, +1.03]% | -0.38 | 2/5 |
| double_well_A | tau_T | 0.105 | +0.0196 | +18.70% [-39.18, +76.57]% | +0.90 | 4/5 |
| double_well_A | rdf_peak | 4.272 | -0.0294 | -0.69% [-5.10, +3.73]% | -0.43 | 3/5 |
| double_well_B | D_green_kubo | 0.006427 | -0.000113 | -1.76% [-9.36, +5.85]% | -0.64 | 2/5 |
| double_well_B | D_msd | 0.005147 | -1.06e-05 | -0.21% [-6.10, +5.69]% | -0.10 | 2/5 |
| double_well_B | T_mean | 0.6992 | -8.97e-05 | -0.01% [-0.05, +0.02]% | -1.04 | 1/5 |
| double_well_B | U_per_particle | -5.703 | -0.00207 | -0.04% [-0.70, +0.63]% | -0.15 | 2/5 |
| double_well_B | tau_T | 0.0206 | +0.00178 | +8.62% [-1.94, +19.18]% | +2.27 | 4/5 |
| double_well_B | rdf_peak | 4.195 | -0.0272 | -0.65% [-1.55, +0.25]% | -1.99 | 2/5 |


## 2. double_well_A: do all five seeds go the same way?

Yes, at max lag 3, for both estimators:
- every d_i > 0;
- paired t = 5.63 (GK) and 5.17 (MSD), df = 4;
- the exact sign-test p is 0.0625, the smallest attainable with 5 pairs.

| quantity (max lag 3) | seed 101 | 102 | 103 | 104 | 105 | reference D per seed |
|---|---|---|---|---|---|---|
| D_green_kubo | +2.03e-03 | +1.14e-03 | +7.01e-04 | +1.75e-03 | +2.16e-03 | 0.0193, 0.0178, 0.0203, 0.0212, 0.0183 |
| D_msd | +1.02e-03 | +8.65e-04 | +4.91e-04 | +1.60e-03 | +1.60e-03 | 0.0209, 0.0186, 0.0203, 0.0206, 0.0198 |


The spread of these d_i (sd/D = 3.2% for GK, 2.4% for MSD) is much smaller than the window-level spread in Section 4:
- 15–18% of D per 15-unit window;
- roughly 8% for a full run if the windows were independent.

With only 4 degrees of freedom the sd itself is poorly determined. The large t partly reflects the fact that five
numbers happened to be similar.

## 3. Dependence on the maximum lag (`lag_scan.csv`)

| case | estimator | max lag 0.5 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| lj_A | D_green_kubo | -2.8% [-7.0, +1.4] 1/5 | -4.5% [-9.0, -0.0] 1/5 | -3.2% [-8.8, +2.3] 1/5 | -4.4% [-8.5, -0.2] 1/5 |
| lj_A | D_msd | -2.1% [-6.3, +2.1] 2/5 | -3.6% [-8.3, +1.1] 1/5 | -3.3% [-8.4, +1.8] 1/5 | -3.8% [-9.5, +1.8] 1/5 |
| double_well_A | D_green_kubo | +3.0% [-2.5, +8.4] 4/5 | +6.0% [-0.4, +12.5] 4/5 | +5.9% [+2.2, +9.7] 5/5 | +8.0% [+4.1, +12.0] 5/5 |
| double_well_A | D_msd | +1.3% [-4.9, +7.5] 3/5 | +3.5% [-1.9, +8.9] 3/5 | +4.9% [+0.5, +9.2] 5/5 | +5.6% [+2.6, +8.5] 5/5 |
| double_well_B | D_green_kubo | -0.0% [-2.8, +2.7] 2/5 | +0.0% [-3.4, +3.4] 2/5 | -0.8% [-6.3, +4.6] 2/5 | -1.8% [-9.4, +5.8] 2/5 |
| double_well_B | D_msd | -0.1% [-2.9, +2.7] 2/5 | -0.4% [-4.5, +3.7] 2/5 | -0.3% [-5.9, +5.2] 2/5 | -0.2% [-6.1, +5.7] 2/5 |
| lj_B | D_green_kubo | +2.6e-10 (abs.) | -8.5e-09 (abs.) | +1.1e-07 (abs.) | -4.1e-08 (abs.) |
| lj_B | D_msd | -4.1e-08 (abs.) | -4.0e-08 (abs.) | +6.3e-08 (abs.) | +1.0e-07 (abs.) |


Green-Kubo contributions of lag segments:

| case | lags 0–0.5 | 0.5–1 | 1–2 | 2–3 |
|---|---|---|---|---|
| lj_A | -1.8e-03 [-4.4e-03, +9.1e-04] 1/5 | -1.1e-03 [-2.1e-03, -8.5e-05] 0/5 | +8.8e-04 [-5.7e-04, +2.3e-03] 4/5 | -7.0e-04 [-2.2e-03, +8.2e-04] 2/5 |
| double_well_A | +7.7e-04 [-6.4e-04, +2.2e-03] 4/5 | +6.2e-04 [-4.7e-04, +1.7e-03] 4/5 | -1.5e-04 [-1.2e-03, +9.2e-04] 2/5 | +3.1e-04 [-1.3e-04, +7.6e-04] 3/5 |
| double_well_B | -3.4e-06 [-2.4e-04, +2.4e-04] 2/5 | +4.6e-06 [-9.9e-05, +1.1e-04] 3/5 | -5.5e-05 [-2.3e-04, +1.2e-04] 3/5 | -5.9e-05 [-2.8e-04, +1.6e-04] 3/5 |


- **double_well_A.** The difference grows with the integration limit:
  - Green-Kubo: +3.0% at lag 0.5 (CI includes 0, 4/5 positive) up to +8.0% at lag 3 (5/5);
  - MSD: +1.3% (3/5) up to +5.6% (5/5).
  - No single lag segment has a CI excluding 0. The 0–0.5 and 0.5–1 segments contribute about +7.7e-4 and +6.2e-4
    (4/5 positive each).
  - The difference therefore lives in the slow part of the correlation functions, not in the short-time decay.
- **lj_A.** The difference is negative at every lag (−2 to −4.5%, mostly 1/5 positive). The CI excludes 0 only for
  GK at lags 1 and 3, marginally.
- **double_well_B.** ±2% with CIs of ±3–9%.
- **lj_B.** |d| ≤ 1e-7 at every lag: these pairs stay pathwise coupled.

## 4. Time windows (`window_summary.csv`, raw trajectories)

| case | quantity | t 0–15 | 15–30 | 30–45 | 45–60 | per-seed trend of d per window, mean [95% CI], t (df 4) |
|---|---|---|---|---|---|---|
| double_well_A | D_green_kubo | +14.8% 5/5 | +15.4% 4/5 | +14.5% 4/5 | -16.0% 0/5 | -1.92e-03 [-3.77e-03, -5.89e-05], -2.86 |
| double_well_A | D_msd | +12.1% 4/5 | +18.9% 4/5 | +12.6% 3/5 | -11.4% 1/5 | -1.63e-03 [-3.69e-03, +4.42e-04], -2.18 |
| double_well_A | T_mean | +0.4% 3/5 | +0.7% 4/5 | +0.6% 4/5 | -0.9% 1/5 | -2.82e-03 [-5.45e-03, -1.76e-04], -2.96 |
| double_well_A | U_per_particle | +0.5% 4/5 | +1.6% 3/5 | +2e-03 abs. 2/5 | -2.8% 1/5 | -6.48e-02 [-1.27e-01, -2.36e-03], -2.88 |
| lj_A | D_green_kubo | -0.8% 2/5 | -8.4% 2/5 | -2.5% 1/5 | -8.4% 1/5 | -1.05e-03 [-4.97e-03, +2.88e-03], -0.74 |
| lj_A | D_msd | +2.6% 4/5 | -6.8% 1/5 | -5.3% 0/5 | -2.4% 1/5 | -7.89e-04 [-3.55e-03, +1.97e-03], -0.79 |
| lj_A | T_mean | +1e-04 abs. 3/5 | -0.4% 2/5 | +0.5% 3/5 | -3e-05 abs. 3/5 | +5.74e-04 [-2.35e-03, +3.49e-03], +0.55 |
| lj_A | U_per_particle | +0.3% 3/5 | -1.0% 2/5 | -2.1% 1/5 | -1.4% 2/5 | -1.96e-02 [-1.08e-01, +6.91e-02], -0.61 |
| double_well_B | D_green_kubo | -7e-09 abs. 2/5 | -0.1% 1/5 | -5.0% 1/5 | +0.9% 3/5 | -1.26e-05 [-3.51e-04, +3.26e-04], -0.10 |
| double_well_B | D_msd | +2e-11 abs. 3/5 | +5e-07 abs. 3/5 | -2.3% 1/5 | +0.7% 3/5 | +4.93e-07 [-1.90e-04, +1.91e-04], +0.01 |
| double_well_B | T_mean | -6e-09 abs. 2/5 | +3e-05 abs. 4/5 | -4e-05 abs. 2/5 | -0.1% 2/5 | -1.13e-04 [-4.50e-04, +2.25e-04], -0.93 |
| double_well_B | U_per_particle | -7e-09 abs. 3/5 | -4e-04 abs. 2/5 | -0.2% 3/5 | +0.1% 2/5 | +2.86e-04 [-3.74e-02, +3.80e-02], +0.02 |
| lj_B | D_green_kubo | +7e-11 abs. 5/5 | +4e-10 abs. 4/5 | +3e-08 abs. 3/5 | +2e-07 abs. 3/5 | +5.51e-08 [-1.08e-07, +2.18e-07], +0.94 |
| lj_B | D_msd | +6e-11 abs. 5/5 | +7e-11 abs. 3/5 | +7e-09 abs. 3/5 | +2e-07 abs. 3/5 | +6.91e-08 [-1.25e-07, +2.64e-07], +0.99 |
| lj_B | T_mean | +3e-12 abs. 2/5 | -1e-10 abs. 2/5 | +3e-08 abs. 3/5 | -3e-07 abs. 1/5 | -8.27e-08 [-2.71e-07, +1.06e-07], -1.22 |
| lj_B | U_per_particle | -9e-10 abs. 2/5 | +3e-08 abs. 3/5 | -5e-07 abs. 2/5 | -9e-06 abs. 1/5 | -2.65e-06 [-1.11e-05, +5.77e-06], -0.87 |


- **double_well_A.**
  - **D:** about +15% (fast higher) in t ∈ [0, 45], where 4–5 of 5 seeds are positive in every window, and −16% (GK)
    or −11% (MSD) in t ∈ [45, 60], where 0–1 of 5 are positive.
  - **T and U/N:** in the same windows the fast runs are slightly hotter (+0.4 to +0.7%) and less bound, then colder
    (−0.9%) and more bound (U/N −2.8%) in the last window.
  - **Trend:** the per-seed trend of d across windows is negative for D, T and U (t = −2.2 to −3.0, df 4).
  - This is not a constant offset. D, T and U move together, which is the signature of a slow collective
    (structural/energy) fluctuation. A looser, warmer configuration diffuses faster.
- **lj_A.** Mostly negative D differences (−0.8 to −8.4% GK) with no significant trend.
- **double_well_B.**
  - Window 0 (pairs still pathwise coupled): |d| ≤ 1e-8.
  - Window 1 (pairs starting to separate): |d| ≤ 0.1%.
  - Windows 2–3 (pairs separated): −5% to +1%.
- **lj_B.** |d| ≤ 1e-5 in every window (the pairs never separate).

## 5. Operator on the configurations actually visited (`operator_along_trajectories.csv`)

130 frames per case, every 5 time units of all 10 runs (both methods):

| case | median ‖Γ_h − Γ_ref‖₂/‖Γ_ref‖₂ | max | max abs. difference of restricted λ_min |
|---|---|---|---|
| lj_A | 1.27e-7 | 1.53e-7 | 3.9e-7 |
| lj_B | 3.94e-8 | 4.49e-8 | 9.3e-7 |
| double_well_A | 1.14e-7 | 1.35e-7 | 4.0e-7 |
| double_well_B | 4.60e-8 | 5.38e-8 | 1.0e-6 |

## Answers

1. **How large is the difference?**
   - Full run, max lag 3: double_well_A D is +8.0% (GK, CI +4.1 to +12.0%) and +5.6% (MSD, CI +2.6 to +8.5%).
     lj_A D is −4.4% (GK, CI −8.5 to −0.2%) and −3.8% (MSD, CI −9.5 to +1.9%).
   - double_well_B is within ±2% with CIs up to about ±9%. lj_B differs by ≤ 1e-7.
   - T, U/N and the RDF peak: all CIs include 0, with widths ≤ ±0.9% (T), ±4.4% (U/N) and ±5% (RDF peak).
2. **Does it depend on the integration limit or the time window? Yes, on both.**
   - double_well_A grows from +1–3% at lag 0.5 to +5.6–8% at lag 3.
   - It is +12–19% in the first three windows and −11 to −16% in the last, moving together with T and U.
   - lj_A is opposite in sign to double_well_A. A friction bias of the fast operator would shift D the same way in
     both potentials.
3. **What the data do and do not show.**
   - **Not ruled out:** the paired test detects a difference in double_well_A D that the earlier combined-SEM
     comparison missed. The data therefore do not establish that the two methods give the same D to better than
     several percent for law A.
   - **Not established:** a systematic effect of the fast algorithm. Arguing against one:
     - the dependence on window and lag;
     - the sign reversal between potentials and between windows;
     - the operator agreement of ≤ 1.5e-7 on the configurations visited;
     - the coupled pathwise agreement up to the Lyapunov horizon;
     - the number of comparisons made.
   - The available evidence is compatible with slow collective fluctuations sampled by only 5 pairs. It does not
     exclude a real effect of a few percent.
4. **Targeted rerun: recommended for law A** (double_well_A first, then lj_A). It was not started; it needs a
   decision on cost. Planning numbers from these data use the larger of the full-run sd and the window-based estimate
   (sd ≈ 8% of D for double_well_A, 5% for lj_A):

   | case | pairs for a 95% CI half-width of ±3% / ±2% |
   |---|---|
   | double_well_A | 29–36 / 61–77 |
   | lj_A | 12–13 / 23–26 |

   Suggested design:
   - New independent start states from a long reference burn-in. The v2 states are only 5 and are already used.
   - Coupled fast/reference pairs with t = 60, as here.
   - A **null-control arm**: reference vs the same reference started from positions perturbed by ~1e-8, on the same
     noise. This calibrates how large paired differences become between runs that differ only by chaotic
     amplification of a tiny perturbation.
   - Analysis: the same paired statistics, with windows and lag scan.
   - Cost per double_well_A pair is about 20 min (Lanczos) + 4 min (reference) + 4 min (null arm) on one core. 30
     pairs take about 14 core-hours (≈ 3.5 h on 4 cores), plus the burn-in of new start states.
