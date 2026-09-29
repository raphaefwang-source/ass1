# Fixed-rank direct-start root-action budget

Direct-start Lanczos (full reorthogonalization, Pi after every literal PPPM Gamma_h action), f_dt with beta = m = 1.
Same Gamma_h parameters, configurations and Gaussian seeds (2 vectors per N) as the completed scaling run in
`lanczos_fdt_results/`. That run saved only summary ranks, not per-rank vectors, so these fixed-rank errors come
from a minimal direct-start rerun to rank 260. The reference is eta at rank 260, accepted only if the last three
successive-rank differences are <= 1e-12 and it agrees with rank S-30 to <= 1e-11.
Reference acceptance: 24/24 runs; worst successive difference 2.6e-15, worst rank-(S-30) agreement 1.9e-13.
lambda_max estimates match the previous run to 2.9e-15.

## dt lambda_max / m = 0.5 (primary)

Relative root-action error ||eta_r - eta_ref|| / ||eta_ref|| (median / 90th percentile / max over 2 seeds):

| N | error(r=16) | error(r=24) | error(r=32) | error(r=40) |
|---|---|---|---|---|
| 128 | 1.58e-04 / 1.74e-04 / 1.78e-04 | 5.75e-06 / 6.01e-06 / 6.07e-06 | 1.49e-07 / 1.52e-07 / 1.53e-07 | 4.48e-09 / 4.55e-09 / 4.56e-09 |
| 256 | 4.54e-04 / 4.75e-04 / 4.81e-04 | 3.75e-05 / 4.19e-05 / 4.30e-05 | 3.25e-06 / 3.40e-06 / 3.44e-06 | 2.43e-07 / 2.57e-07 / 2.61e-07 |
| 512 | 1.26e-03 / 1.35e-03 / 1.38e-03 | 1.57e-04 / 1.75e-04 / 1.80e-04 | 1.96e-05 / 2.01e-05 / 2.03e-05 | 3.26e-06 / 3.32e-06 / 3.34e-06 |
| 1024 | 1.99e-03 / 2.12e-03 / 2.16e-03 | 4.54e-04 / 4.73e-04 / 4.78e-04 | 9.10e-05 / 9.46e-05 / 9.55e-05 | 1.75e-05 / 1.79e-05 / 1.79e-05 |
| 2048 | 3.62e-03 / 3.75e-03 / 3.78e-03 | 9.66e-04 / 1.01e-03 / 1.02e-03 | 2.56e-04 / 2.60e-04 / 2.60e-04 | 7.78e-05 / 7.79e-05 / 7.79e-05 |
| 4000 | 4.50e-03 / 4.55e-03 / 4.56e-03 | 1.33e-03 / 1.38e-03 / 1.39e-03 | 4.85e-04 / 5.15e-04 / 5.22e-04 | 1.90e-04 / 1.99e-04 / 2.02e-04 |

| rank | largest error over all N | passes 1e-3? | passes 1e-4? |
|---|---|---|---|
| 16 | 4.56e-03 | no | no |
| 24 | 1.39e-03 | no | no |
| 32 | 5.22e-04 | yes | no |
| 40 | 2.02e-04 | yes | no |

## dt lambda_max / m = 2.0

Relative root-action error ||eta_r - eta_ref|| / ||eta_ref|| (median / 90th percentile / max over 2 seeds):

| N | error(r=16) | error(r=24) | error(r=32) | error(r=40) |
|---|---|---|---|---|
| 128 | 1.80e-04 / 2.00e-04 / 2.05e-04 | 6.54e-06 / 6.86e-06 / 6.95e-06 | 1.69e-07 / 1.72e-07 / 1.72e-07 | 5.08e-09 / 5.18e-09 / 5.21e-09 |
| 256 | 5.07e-04 / 5.26e-04 / 5.30e-04 | 4.17e-05 / 4.62e-05 / 4.73e-05 | 3.61e-06 / 3.74e-06 / 3.78e-06 | 2.70e-07 / 2.83e-07 / 2.86e-07 |
| 512 | 1.31e-03 / 1.41e-03 / 1.43e-03 | 1.63e-04 / 1.82e-04 / 1.87e-04 | 2.04e-05 / 2.09e-05 / 2.11e-05 | 3.39e-06 / 3.46e-06 / 3.48e-06 |
| 1024 | 2.18e-03 / 2.33e-03 / 2.37e-03 | 4.97e-04 / 5.19e-04 / 5.24e-04 | 9.93e-05 / 1.04e-04 / 1.05e-04 | 1.91e-05 / 1.95e-05 / 1.96e-05 |
| 2048 | 3.92e-03 / 4.06e-03 / 4.10e-03 | 1.05e-03 / 1.09e-03 / 1.10e-03 | 2.77e-04 / 2.81e-04 / 2.81e-04 | 8.40e-05 / 8.41e-05 / 8.41e-05 |
| 4000 | 4.72e-03 / 4.77e-03 / 4.78e-03 | 1.40e-03 / 1.44e-03 / 1.45e-03 | 5.08e-04 / 5.38e-04 / 5.46e-04 | 1.98e-04 / 2.08e-04 / 2.11e-04 |

| rank | largest error over all N | passes 1e-3? | passes 1e-4? |
|---|---|---|---|
| 16 | 4.78e-03 | no | no |
| 24 | 1.45e-03 | no | no |
| 32 | 5.46e-04 | yes | no |
| 40 | 2.11e-04 | yes | no |

Momentum / nullspace residual at all fixed ranks: max |total momentum|/||eta_r|| = 1.4e-15, max ||(I - Pi) eta_r|| / ||eta_r|| = 3.5e-17; smallest Ritz value at these ranks = 1.82e-02 (all positive).

## Answers (primary case dt lambda_max/m = 0.5; dt lambda_max/m = 2.0 gives the same conclusions)

1. **Is rank 24 enough for a 1e-3 budget?** No. Its error exceeds 1e-3 at N = 2048 (1.02e-3) and N = 4000
   (1.39e-3).
2. **Is rank 32 enough?** Yes, over the tested range. Its largest error over N = 128–4000 and both seeds is
   5.2e-4 (5.5e-4 at dt lambda_max/m = 2).
3. **Smallest fixed rank meeting 1e-3 for every N = 128–4000:** r = 32. Rank 40 also passes, with a largest
   error of 2.0e-4.
4. **Does any tested rank meet 1e-4 for every N?** No. The best is r = 40, which reaches 1.9–2.0e-4 at N = 4000
   and 7.8e-5 at N = 2048.
5. **Complexity statement.** With r = 32, the root action costs a constant 32 Γ_h actions (one extra for dt
   scaling if λ_max is re-estimated), each measured as O(N log N). Over N = 128–4000 the measured production
   complexity can therefore be tested as O(N log N).

**Caveats**
- The fixed-rank error still grows with N. At r = 32 it goes 1.5e-7, 3.4e-6, 2.0e-5, 9.6e-5, 2.6e-4, 5.2e-4, so it
  roughly doubles each time N doubles near the top of the range. The 1e-3 margin at N = 4000 is only about 1.9×.
  If that trend continues, r = 32 would exceed 1e-3 at roughly N ≈ 8000. The constant-rank statement is verified
  only for N ≤ 4000; beyond that the rank must be re-checked or raised.
- Only two Gaussian seeds per N were available from the completed run, so "90th percentile" and "max" are over
  two samples.
- No full-dynamics claim is made here.

(The numbers above come from a small post-processing script kept outside the repository, as requested; it reuses
`test_lanczos_fdt.py` with the same seeds.)

---

# Large-N extension (N = 8000, 16000)

Script: `fixed_rank_large_n.py`; raw per-seed data: `large_n_raw.json`; figure: `fixed_rank_error_large_n.png`.
Same direct-start Lanczos, density 30/216, Γ_h parameters {'xi': 0.7, 's': 4.1, 'eta': 0.7, 'p': 7}, dt·λ_max/m = 0.5, β = m = 1.
There are 3 fixed Gaussian seeds per new N. Rows for N ≤ 4000 are reused from the tables above
(2 seeds, not rerun).

Reference checks (new N):

- N=8000 seed 0: rank 360, successive 3.1e-15 (≤ 1e-12), 20-step 4.7e-15 (≤ 1e-11); λ_max 35.433, smallest Ritz value 6.859e-03
- N=8000 seed 1: rank 360, successive 3.0e-15 (≤ 1e-12), 20-step 3.8e-15 (≤ 1e-11); λ_max 35.433, smallest Ritz value 6.861e-03
- N=8000 seed 2: rank 360, successive 2.8e-15 (≤ 1e-12), 20-step 5.0e-15 (≤ 1e-11); λ_max 35.433, smallest Ritz value 6.847e-03
- N=16000 seed 0: rank 360, successive 8.1e-14 (≤ 1e-12), 20-step 2.4e-12 (≤ 1e-11); λ_max 88.804, smallest Ritz value 4.422e-03
- N=16000 seed 1: rank 360, successive 9.5e-14 (≤ 1e-12), 20-step 2.9e-12 (≤ 1e-11); λ_max 88.804, smallest Ritz value 4.361e-03
- N=16000 seed 2: rank 360, successive 1.1e-13 (≤ 1e-12), 20-step 2.9e-12 (≤ 1e-11); λ_max 88.804, smallest Ritz value 4.377e-03

Relative root-action error, median / max over seeds:

| N | r=32 | r=40 | r=48 |
|---|---|---|---|
| 128 | 1.49e-07 / 1.53e-07 | 4.48e-09 / 4.56e-09 | ≤ 4.56e-09 (bounded by r=40; not evaluated) |
| 256 | 3.25e-06 / 3.44e-06 | 2.43e-07 / 2.61e-07 | ≤ 2.61e-07 (bounded by r=40; not evaluated) |
| 512 | 1.96e-05 / 2.03e-05 | 3.26e-06 / 3.34e-06 | ≤ 3.34e-06 (bounded by r=40; not evaluated) |
| 1024 | 9.10e-05 / 9.55e-05 | 1.75e-05 / 1.79e-05 | ≤ 1.79e-05 (bounded by r=40; not evaluated) |
| 2048 | 2.56e-04 / 2.60e-04 | 7.78e-05 / 7.79e-05 | ≤ 7.79e-05 (bounded by r=40; not evaluated) |
| 4000 | 4.85e-04 / 5.22e-04 | 1.90e-04 / 2.02e-04 | ≤ 2.02e-04 (bounded by r=40; not evaluated) |
| 8000 | 9.22e-04 / 1.04e-03 | 4.44e-04 / 4.54e-04 | 2.09e-04 / 2.15e-04 |
| 16000 | 1.75e-03 / 1.78e-03 | 8.84e-04 / 9.07e-04 | 4.75e-04 / 4.83e-04 |

Diagnostics for the new N (median over seeds):

| N | r | t_root [s] | Γ_h actions [s] | reorth [s] | f(T_r) [s] | Ritz min | Ritz max | max momentum residual |
|---|---|---|---|---|---|---|---|---|
| 8000 | 32 | 10.62 | 10.58 | 0.047 | 0.0007 | 2.697e-02 | 35.433 | 1.8e-15 |
| 8000 | 40 | 13.49 | 13.43 | 0.064 | 0.0007 | 1.646e-02 | 35.433 | 1.9e-15 |
| 8000 | 48 | 15.87 | 15.79 | 0.075 | 0.0007 | 1.204e-02 | 35.433 | 1.4e-15 |
| 16000 | 32 | 22.41 | 22.34 | 0.067 | 0.0015 | 4.512e-02 | 88.804 | 2.2e-15 |
| 16000 | 40 | 28.33 | 28.24 | 0.086 | 0.0016 | 2.855e-02 | 88.804 | 2.0e-15 |
| 16000 | 48 | 34.19 | 34.08 | 0.112 | 0.0016 | 1.931e-02 | 88.804 | 2.7e-15 |

Root-action time at the chosen rank r = 40 and T/(N log N):

| N | T_root [s] | source | T/(N log N) [s] |
|---|---|---|---|
| 128 | 0.129 | estimated from stored timings | 2.08e-04 |
| 256 | 0.257 | estimated from stored timings | 1.81e-04 |
| 512 | 0.549 | estimated from stored timings | 1.72e-04 |
| 1024 | 1.131 | estimated from stored timings | 1.59e-04 |
| 2048 | 2.884 | estimated from stored timings | 1.85e-04 |
| 4000 | 4.939 | estimated from stored timings | 1.49e-04 |
| 8000 | 13.491 | measured | 1.88e-04 |
| 16000 | 28.326 | measured | 1.83e-04 |

## Answers

1. **Rank 32 at N = 8000:** no (max over seeds 1.04e-03).
2. **Rank 32 at N = 16000:** no (max 1.78e-03).
3. **Rank 40:** N = 8000: yes (4.54e-04), N = 16000: yes (9.07e-04).
4. **Rank 48:** N = 8000: yes (2.15e-04), N = 16000: yes (4.83e-04).
5. **Smallest single rank below 1e-3 for every N = 128–16000:** r = 40.
6. **T/(N log N) at r = 40:** ranges from 1.49e-04 to 2.08e-04 s over N = 128–16000, a max/min ratio of 1.40. Over the large-N points (N ≥ 2048) it goes 1.85e-04, 1.49e-04, 1.88e-04, 1.83e-04.
7. **Timing exponent (T_root ∝ N^α at r = 40):** α = 1.146 over N = 2048–16000; α = 1.070 over the measured N = 8000–16000 alone.

Notes:
- For N ≤ 4000, T_root is estimated as r × (stored per-action time) + (stored rank-400 reorth time) × (r/400)².
  The new N are measured directly with the same code; the stored N ≤ 4000 timings come from an earlier run in the same environment, so the combined exponent mixes measured and estimated points.
- Rank 48 was not part of the previous N ≤ 4000 set, so its column there is bounded by the rank-40 values
  (the error decreased monotonically with rank in every run).
- No dynamics claim is made.
