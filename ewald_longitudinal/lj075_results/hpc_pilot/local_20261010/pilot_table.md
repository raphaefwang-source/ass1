# HPC pilot: four cases

LOCAL (no Slurm job id): not an HPC result

| | dense_dt0.005 | fast_dt0.005 | dense_dt0.01 | fast_dt0.01 |
|---|---|---|---|---|
| O-step operator | full periodic Γ (lattice sum, k=0 kept), dense eigendecomposition | PPPM Γ_h (M 24, k=0 kept), Lanczos 14/5 | full periodic Γ (lattice sum, k=0 kept), dense eigendecomposition | PPPM Γ_h (M 24, k=0 kept), Lanczos 14/5 |
| dt / steps / frames | 0.005 / 200 / 201 | 0.005 / 200 / 201 | 0.01 / 100 / 101 | 0.01 / 100 / 101 |
| seed (noise stream) | 7005 | 7005 | 7010 | 7010 |
| host / CPU | vm / Intel(R) Xeon(R) Processor @ 2.80GHz | vm / Intel(R) Xeon(R) Processor @ 2.80GHz | vm / Intel(R) Xeon(R) Processor @ 2.80GHz | vm / Intel(R) Xeon(R) Processor @ 2.80GHz |
| state / all values finite | complete / yes | complete / yes | complete / yes | complete / yes |
| max \|P(t)−P(0)\| (1st half, 2nd half) | 1.3e-13 (1.3e-13, 1.3e-13) | 1.3e-13 (6.3e-14, 1.3e-13) | 8.5e-14 (2.8e-14, 8.5e-14) | 6.7e-14 (2.8e-14, 6.7e-14) |
| max \|P−P0\| / (ε Σ\|p\|); drift | 1.0; none detectable (round-off) | 1.0; none detectable (round-off) | 0.6; none detectable (round-off) | 0.5; none detectable (round-off) |
| T_kin: t=0 / mean / min–max | 0.9203 / 0.9826 / 0.864–1.134 | 0.9203 / 0.9826 / 0.864–1.134 | 0.9203 / 0.9706 / 0.873–1.079 | 0.9203 / 0.9706 / 0.873–1.079 |
| U/N: t=0 / mean / end | -4.7625 / -4.6791 / -4.5984 | -4.7625 / -4.6791 / -4.5984 | -4.7625 / -4.6839 / -4.7128 | -4.7625 / -4.6839 / -4.7128 |
| min pair distance | 0.877 | 0.877 | 0.892 | 0.892 |
| spectrum (Ritz) range | 4.653 – 12.31 | 4.659 – 12.28 | 4.653 – 12.13 | 4.659 – 12.09 |
| monitor points / max error estimate (budget) | 21 / n/a (exact matrix functions) | 21 / 4.9e-11 (2e-07) | 11 / n/a (exact matrix functions) | 11 / 4.8e-11 (2e-07) |
| actual Lanczos error, max (noise, damping) | n/a | 4.3e-12, 4.9e-13 | n/a | 3.3e-12, 1.5e-11 |
| init (process start → first step) [s] | 3.99 (imports 0.91) | 0.95 (imports 0.90) | 4.48 (imports 0.92) | 1.03 (imports 0.99) |
| integration step: median / p90 / max [s] | 1.129 / 2.643 / 3.082 | 0.208 / 0.243 / 0.289 | 1.034 / 2.568 / 2.921 | 0.204 / 0.245 / 0.302 |
| monitor per point [s] | 0.000 | 0.325 | 0.000 | 0.300 |
| output per step (frame every step, checkpoint every 25 steps) [s] | 0.0004 | 0.0004 | 0.0005 | 0.0005 |
| full per step (integration + monitor + output) [s] | 1.584 | 0.249 | 1.504 | 0.248 |
| run CPU [s] / run process wall [s] | 323.7 / 328.7 | 52.2 / 53.9 | 153.6 / 156.3 | 27.5 / 28.6 |
| peak RSS [MB] | 604 | 121 | 604 | 121 |
| segments / checkpoints at / restart bitwise / duplicate refused | 3 / [0, 25, 26, 51, 56, 81, 106, 131, 156, 181, 200] / yes / yes | 3 / [0, 25, 26, 51, 56, 81, 106, 131, 156, 181, 200] / yes / yes | 3 / [0, 25, 26, 51, 56, 81, 100] / yes / yes | 3 / [0, 25, 26, 51, 56, 81, 100] / yes / yes |
| threads (env / OS threads) | 1 / [1] | 1 / [1] | 1 / [1] | 1 / [1] |
| raw output on disk [MB] | 2.6 | 2.6 | 1.3 | 1.3 |
| est. core-s per time unit (prod. monitor cadence) | 317 | 43 | 150 | 21 |

## Fast minus dense, same dt, same initial state and noise

| dt | t | Δq rms | Δq max | Δv rms | Δv max | Δv rms / v rms | ΔT_kin | ΔU/N |
|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.005 | 2.88e-11 | 7.96e-11 | 1.15e-08 | 3.18e-08 | 6.96e-09 | -1.7e-10 | -2.3e-11 |
| 0.005 | 0.05 | 7.99e-10 | 2.26e-09 | 2.63e-08 | 6.10e-08 | 1.57e-08 | -2.1e-09 | -3.0e-10 |
| 0.005 | 0.1 | 1.82e-09 | 4.53e-09 | 3.26e-08 | 8.77e-08 | 1.90e-08 | -2.5e-09 | +5.9e-10 |
| 0.005 | 0.25 | 4.58e-09 | 1.13e-08 | 4.09e-08 | 9.13e-08 | 2.40e-08 | -1.5e-09 | +2.0e-09 |
| 0.005 | 0.5 | 8.25e-09 | 2.84e-08 | 6.32e-08 | 1.71e-07 | 3.62e-08 | +4.2e-09 | -7.9e-10 |
| 0.005 | 1 | 1.96e-08 | 6.28e-08 | 1.28e-07 | 4.85e-07 | 7.44e-08 | -1.8e-10 | +3.7e-10 |
| 0.01 | 0.01 | 7.89e-11 | 1.86e-10 | 1.57e-08 | 3.69e-08 | 9.56e-09 | +6.0e-11 | -1.0e-10 |
| 0.01 | 0.05 | 8.44e-10 | 2.12e-09 | 2.62e-08 | 5.68e-08 | 1.56e-08 | -1.7e-09 | -1.5e-09 |
| 0.01 | 0.1 | 1.80e-09 | 4.69e-09 | 3.16e-08 | 7.56e-08 | 1.85e-08 | -2.4e-09 | -4.9e-10 |
| 0.01 | 0.25 | 4.64e-09 | 1.41e-08 | 4.34e-08 | 1.09e-07 | 2.57e-08 | -1.4e-09 | -2.2e-09 |
| 0.01 | 0.5 | 9.15e-09 | 2.88e-08 | 6.54e-08 | 2.04e-07 | 3.74e-08 | +5.1e-09 | -1.3e-08 |
| 0.01 | 1 | 2.22e-08 | 7.81e-08 | 1.35e-07 | 5.14e-07 | 7.78e-08 | +1.2e-08 | -1.8e-08 |
