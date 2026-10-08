# Velocity save interval: VACF check at N = 64

Script: `../vacf_sampling_check.py`. Data: `summary.json`, `intervals.csv`, `vacf_sampling.png`.

**Scope.** This check only chooses the production save interval. It says nothing about the convergence of the
long-time diffusion coefficient.

## Runs

**Setup:**
- `toy_run.py`, configuration `costopt_hiacc` (unchanged), double well, N = 64;
- laws A (ranks 32/5) and B (16/6), seeds 101 and 102, each starting from the v2 N = 64 state of the same seed;
- burn-in t = 2, production t = 40 (8001 frames);
- q and v saved every step, i.e. Δ = dt = 0.005.

The raw frames (25 MB per run) are not committed. Commands are at the end.

**Cost:** 531, 546, 335 and 335 s of wall time on one core each.

**Peak RSS:** 127 MB.

**Runtime monitor:**
- Lanczos error estimates ≤ 2e-11 in every run, against budgets of 2e-7 (A) and 5e-8 (B).
- The law-A seed-102 run logged 2 spectrum warnings. Its smallest Ritz value, 1.545–1.550, was below 0.9 × the
  λ_min (1.729) verified at N = 64, which came from a single state.
- The Lanczos error estimates at those steps were ≤ 2e-11, so accuracy was not affected.
- The warning text now includes the error estimate.

## Method

For every run and every candidate interval Δ ∈ {0.01, 0.025, 0.05, 0.1}, the stored-every-step trajectory is
down-sampled offline and compared with the every-step result.

**VACF** C(τ) = ⟨v_i(t₀)·v_i(t₀+τ)⟩, averaged over particles and time origins, by FFT.

**Two versions per Δ:**
- **quadrature only:** the every-step VACF, with all time origins, read at lags kΔ. This isolates the
  time-grid error.
- **production-like:** the VACF computed only from the frames a run with interval Δ would store, so time origins
  and lags are both every Δ.

**Early decay:**
- C(Δ)/C(0);
- the number of stored lags before the normalised VACF falls to 1/2;
- the times τ½ and τ_{1/e} at which it falls to 1/2 and 1/e, interpolated linearly between stored lags.

**Green–Kubo:** D(T) = ⅓∫₀ᵀ C dτ by the trapezoid rule, with the same upper limits T = 0.2, 0.5, 1, 2, 3 for every Δ.
The statistical scale is the block SEM of the every-step value (8 blocks of t = 5).

**Selection rule, fixed before the comparison.** The largest Δ that satisfies all three conditions for both laws and
both seeds:
- quadrature-only GK bias ≤ 0.2 % at every T;
- at least 5 stored lags before the VACF falls to 1/2;
- τ½ within 2 % of the every-step value.

If no candidate passes, saving every step is kept.

## Results (`intervals.csv`)

| law | τ½ (every step) | τ_{1/e} | Δ = 0.01 | Δ = 0.025 | Δ = 0.05 | Δ = 0.1 |
|---|---|---|---|---|---|---|
| A | 0.058 (≈ 11.6 steps) | 0.073 | GK bias +0.07 to +0.14 %; 6 lags before ½; τ½ +0.06 % | +0.6 to +1.2 %; 3 lags | +2.4 to +5.0 %; 2 lags | +10 to +21 %; 1 lag |
| B | 0.017 (≈ 3.4 steps) | 0.024 | +1.5 to +3.6 %; 2 lags; τ½ +1.4 % | +12 to +30 %; 1 lag; τ½ +12 % | +48 to +118 % | +170 to +430 % |

GK biases are quadrature only, over T = 0.2–3 and both seeds. The production-like values differ from them by up to
2.6 percentage points for law A, and by up to 7.8 for law B (the largest at Δ = 0.1, where the bias itself is
+170 to +430 %). The difference is statistical: fewer time origins.

**Scale of the statistical error.** The block SEM of D(T = 3) in these short N = 64 runs is 7–11 %. Production
(N = 256 and 512, t = 60, 5 seeds × 2 potentials) will have a smaller error. A systematic bias of a few percent,
which is what law B gets even at Δ = 0.01, would then no longer be negligible.

**Every-step data are not quadrature-converged either for law B.** The Richardson estimate of the trapezoid error of
the every-step value, (D_0.005 − D_0.01)/3, is −0.5 to −1.2 % relative for law B and −0.02 to −0.05 % for law A. The
every-step trapezoid integral for law B therefore over-estimates D by up to about 1.2 %.

The analysis of law-B runs should use a higher-order or Richardson-corrected quadrature on the every-step data, or
cross-check against the MSD. This is possible only if every step is stored.

## Recommendation

No candidate interval passes the rule for law B; law A alone would pass at Δ = 0.01. **Keep saving q and v every
step (Δ = dt = 0.005) for both laws.**

**What changed:**
- the `toy_run.py` default and the task list (`hpc/make_tasklist.py --save-every 0.005`,
  `hpc/tasks/production_N256_N512.tsv`) now save every step;
- the earlier default of 0.1 would have biased the law-B Green–Kubo integral by +170 to +430 %.

**Disk.** Frames are 2 × N × 3 × 8 bytes per step:

| N | per task (t = 60, 12 001 frames) |
|---|---|
| 256 | ≈ 141 MB |
| 512 | ≈ 281 MB |

Per-step diagnostics add ≈ 3 MB per task. The default 40-task list needs ≈ 8.2 GB in all, against 0.4 GB at the old
0.1 interval.

Storing positions less often than velocities would roughly halve this. It is not implemented, so the single interval
applies to both.

## Commands

```bash
cd ewald_longitudinal
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for K in A B; do for S in 101 102; do
  python3 toy_run.py --config costopt_hiacc --potential double_well --kernel $K --N 64 --seed $S \
      --burn-in 2.0 --production 40.0 --save-every-steps 1 --checkpoint-every-steps 2000 --out RUNS/dw_${K}_N64_s$S &
done; done; wait
python3 vacf_sampling_check.py RUNS/dw_A_N64_s101 RUNS/dw_A_N64_s102 RUNS/dw_B_N64_s101 RUNS/dw_B_N64_s102
```
