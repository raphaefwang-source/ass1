### References

| configuration | L | ||Γ_ref||_2 | Λ_lb | λ_min on Range Π | ||Γ_D55 − Γ_ref||_2/||Γ_ref||_2 | D55 tail bound (rel.) | ||Γ_ref T||/λ_max |
|---|---|---|---|---|---|---|---|
| moderate_N128_s1 | 5.5469 | 14.5573 | 12.7248 | 5.1545 | 3.66e-15 | 5.5e-14 | 4.0e-16 |
| moderate_N128_s2 | 5.5469 | 14.0656 | 12.6417 | 5.2543 | 3.91e-15 | 5.7e-14 | 4.6e-16 |
| moderate_N128_s3 | 5.5469 | 13.8278 | 12.0900 | 5.2486 | 3.93e-15 | 5.8e-14 | 4.0e-16 |
| moderate_N256_s1 | 6.9886 | 15.0052 | 13.4545 | 4.2622 | 2.82e-15 | 5.7e-14 | 6.7e-16 |
| moderate_N256_s2 | 6.9886 | 15.0554 | 13.7952 | 4.1140 | 2.98e-15 | 5.7e-14 | 6.3e-16 |
| moderate_N256_s3 | 6.9886 | 15.1693 | 13.6114 | 4.0754 | 2.78e-15 | 5.7e-14 | 6.6e-16 |
| strong_N128_s1 | 5.5469 | 14.5638 | 13.1546 | 4.6662 | 3.85e-15 | 5.5e-14 | 4.2e-16 |
| strong_N128_s2 | 5.5469 | 14.6319 | 13.2278 | 4.6671 | 4.10e-15 | 5.5e-14 | 4.3e-16 |
| strong_N128_s3 | 5.5469 | 14.2515 | 12.8890 | 4.7370 | 4.01e-15 | 5.7e-14 | 4.3e-16 |
| strong_N256_s1 | 6.9886 | 15.8946 | 14.3219 | 3.4616 | 2.96e-15 | 5.4e-14 | 6.5e-16 |
| strong_N256_s2 | 6.9886 | 15.8187 | 14.1728 | 3.4589 | 2.79e-15 | 5.4e-14 | 6.6e-16 |
| strong_N256_s3 | 6.9886 | 15.8463 | 14.1336 | 3.5040 | 2.87e-15 | 5.4e-14 | 6.8e-16 |
| strong_N512_s1 | 8.8051 | 17.5490 | 15.1781 | 2.3020 | 2.06e-15 | 5.3e-14 | 8.4e-16 |
| uniform_N128_s1 | 5.5469 | 13.2278 | 11.5463 | 5.4589 | 4.02e-15 | 6.1e-14 | 4.4e-16 |
| uniform_N128_s2 | 5.5469 | 13.4671 | 11.8629 | 5.5612 | 4.05e-15 | 6.0e-14 | 4.5e-16 |
| uniform_N128_s3 | 5.5469 | 13.2269 | 11.4907 | 5.4696 | 4.19e-15 | 6.1e-14 | 4.2e-16 |
| uniform_N256_s1 | 6.9886 | 13.6746 | 11.8188 | 4.3686 | 3.20e-15 | 6.3e-14 | 6.6e-16 |
| uniform_N256_s2 | 6.9886 | 13.3849 | 11.7370 | 4.4097 | 3.27e-15 | 6.4e-14 | 6.8e-16 |
| uniform_N256_s3 | 6.9886 | 13.4654 | 11.9469 | 4.4668 | 3.15e-15 | 6.4e-14 | 6.6e-16 |

### (b) split identity

| ξ | max rel. Frobenius (uniform / strong / moderate) | max pair rel. (all 9) | max abs. entry | radial sweep max (r ≥ 1e-6) | mpmath θ_S (ξr ≤ 7), rel. self | mpmath θ_L, rel. self | naive θ−θ_S at r = 1e-4 |
|---|---|---|---|---|---|---|---|
| 0.5 | 2.72e-16 / 2.68e-16 / 2.76e-16 | 1.83e-15 | 7.08e-16 | 1.36e-15 | 4.35e-16 | 6.72e-16 | 9.34e-03 |
| 0.6 | 2.64e-16 / 2.64e-16 / 2.65e-16 | 1.88e-15 | 8.92e-16 | 1.02e-15 | 1.35e-15 | 4.30e-16 | 4.20e-03 |
| 0.75 | 2.58e-16 / 2.60e-16 / 2.61e-16 | 1.57e-15 | 8.33e-16 | 1.25e-15 | 3.94e-15 | 6.72e-16 | 1.61e-03 |
| 0.85 | 2.67e-16 / 2.75e-16 / 2.69e-16 | 1.65e-15 | 6.80e-16 | 1.47e-15 | 2.81e-15 | 6.72e-16 | 2.29e-04 |
| 1.0 | 2.07e-16 / 2.00e-16 / 2.02e-16 | 1.89e-15 | 6.11e-16 | 1.13e-15 | 1.75e-15 | 6.72e-16 | 2.87e-04 |
| 1.2 | 2.82e-16 / 2.97e-16 / 2.85e-16 | 1.87e-15 | 7.22e-16 | 1.03e-15 | 8.80e-16 | 3.45e-16 | 4.66e-04 |
| 1.5 | 2.12e-16 / 2.16e-16 / 2.10e-16 | 1.69e-15 | 6.66e-16 | 1.13e-15 | 8.26e-15 | 3.52e-16 | 1.94e-04 |

Pair images per configuration (|d| ≤ 8): moderate_N256_s1 204371, moderate_N256_s2 204675, moderate_N256_s3 204478, strong_N256_s1 207191, strong_N256_s2 207197, strong_N256_s3 207012, uniform_N256_s1 205067, uniform_N256_s2 205119, uniform_N256_s3 205024.

### (c) short-range cutoff

| configuration | fit interval s | points | c (all) | c (s ≥ 3) | E_S at s = 4.1 / 4.2 | floor vs Γ_ref | floor vs Γ_D55 | bound/E (fit range) | Fourier tail of Γ_L^ex (certified, rel.) |
|---|---|---|---|---|---|---|---|---|---|
| uniform_N256_s1 | [2.0, 5.3] | 34 | 0.981 | 0.988 | 5.49e-08 / 2.40e-08 | 3.85e-15 | 1.83e-15 | 1.50–1.85 | 1.7e-17 |
| uniform_N256_s2 | [2.0, 5.3] | 34 | 0.981 | 0.988 | 5.63e-08 / 2.42e-08 | 3.96e-15 | 2.40e-15 | 1.55–1.85 | 1.7e-17 |
| uniform_N256_s3 | [2.0, 5.3] | 34 | 0.981 | 0.987 | 5.27e-08 / 2.41e-08 | 4.14e-15 | 2.12e-15 | 1.53–1.87 | 1.7e-17 |
| strong_N256_s1 | [2.0, 5.3] | 34 | 0.987 | 0.984 | 4.72e-08 / 2.13e-08 | 3.56e-15 | 1.80e-15 | 1.43–1.88 | 1.5e-17 |
| strong_N256_s2 | [2.0, 5.3] | 34 | 0.987 | 0.984 | 4.73e-08 / 2.10e-08 | 3.71e-15 | 1.92e-15 | 1.43–1.87 | 1.5e-17 |
| strong_N256_s3 | [2.0, 5.3] | 34 | 0.987 | 0.983 | 4.91e-08 / 2.15e-08 | 3.54e-15 | 1.80e-15 | 1.40–1.85 | 1.5e-17 |
| moderate_N256_s1 | [2.0, 5.3] | 34 | 0.984 | 0.988 | 4.82e-08 / 2.21e-08 | 3.52e-15 | 1.90e-15 | 1.55–1.86 | 1.6e-17 |
| moderate_N256_s2 | [2.0, 5.3] | 34 | 0.983 | 0.986 | 4.92e-08 / 2.15e-08 | 3.94e-15 | 1.77e-15 | 1.50–1.86 | 1.5e-17 |
| moderate_N256_s3 | [2.0, 5.3] | 34 | 0.984 | 0.987 | 4.82e-08 / 2.17e-08 | 3.90e-15 | 1.99e-15 | 1.50–1.87 | 1.5e-17 |

| family | c (median curve, s ∈ [2.0, 5.3]) | c (median curve, s ≥ 3) | median floor |
|---|---|---|---|
| uniform | 0.981 | 0.987 | 3.96e-15 |
| strong | 0.987 | 0.984 | 3.56e-15 |
| moderate | 0.983 | 0.987 | 3.90e-15 |

Largest relative difference between E_S and the exact-arithmetic tail ||Γ_S^full − Γ_S^rc||_2/||Γ_ref||_2 over all points with tail > 1e-12: 1.3e-03. Same-neighbour steps: 0.

### (d) mesh validity and retained modes

| N | cutoffs | r_c | k_c used (outermost shell n) | modes retained = shell count | valid M | invalid M (reason) | min \|W_hat\|^2 | excluded modes |
|---|---|---|---|---|---|---|---|---|
| 128 | production | 4.887756188 | 9.34081468247 (n = 68) | 2373 = 2373 | 18–64 | 16: Nyquist: largest retained \|m_a\| = 8 >= M/2 = 8 | 1.3e-03 | 0 |
| 128 | strict | 7.058823529 | 10.1946614731 (n = 81) | 3071 = 3071 | 20–64 | 16: Nyquist: largest retained \|m_a\| = 9 >= M/2 = 8; 18: Nyquist: largest retained \|m_a\| = 9 >= M/2 = 9 | 1.6e-03 | 0 |
| 256 | production | 4.887756188 | 9.34326884418 (n = 108) | 4729 = 4729 | 22–80 | 20: Nyquist: largest retained \|m_a\| = 10 >= M/2 = 10 | 8.4e-04 | 0 |
| 256 | strict | 7.058823529 | 10.1716627546 (n = 128) | 6043 = 6043 | 24–80 | 20: Nyquist: largest retained \|m_a\| = 11 >= M/2 = 10; 22: Nyquist: largest retained \|m_a\| = 11 >= M/2 = 11 | 9.4e-04 | 0 |
| 512 | production | 4.887756188 | 9.35853479262 (n = 172) | 9459 = 9459 | 28–96 | 26: Nyquist: largest retained \|m_a\| = 13 >= M/2 = 13 | 9.3e-04 | 0 |
| 512 | strict | 7.058823529 | 10.1919836793 (n = 204) | 12197 = 12197 | 30–96 | 26: Nyquist: largest retained \|m_a\| = 14 >= M/2 = 13; 28: Nyquist: largest retained \|m_a\| = 14 >= M/2 = 14 | 7.1e-04 | 0 |

N = 512 production: retained modes with the full-precision k_c = 9459; with k_c truncated to 9.3585 (old code) = 9435 (shell n = 172 dropped).

### (d) N = 512 strongly clustered, every M

| cutoffs | M | Gaussian E_complete (min–max of 10) | slow spectral E_complete (max of 3) | fast E_complete | Gaussian E_mesh (max) | slow E_mesh (max) | E_trunc Gaussian / slow (max) | dense E_op | dense E_op,mesh |
|---|---|---|---|---|---|---|---|---|---|
| production | 28 | 2.03e-08–2.20e-08 | 8.06e-08 | 9.35e-09 | 3.73e-09 | 5.10e-09 | 2.14e-08 / 8.05e-08 | 3.04e-08 | 8.07e-09 |
| production | 30 | 2.01e-08–2.14e-08 | 8.06e-08 | 8.43e-09 | 1.59e-09 | 1.96e-09 | 2.14e-08 / 8.05e-08 | 3.04e-08 | 3.86e-09 |
| production | 32 | 2.01e-08–2.14e-08 | 8.06e-08 | 8.85e-09 | 8.21e-10 | 9.02e-10 | 2.14e-08 / 8.05e-08 | 3.03e-08 | 1.95e-09 |
| production | 36 | 2.02e-08–2.14e-08 | 8.05e-08 | 8.65e-09 | 2.38e-10 | 3.12e-10 | 2.14e-08 / 8.05e-08 | 3.03e-08 | 5.15e-10 |
| production (production M) | 40 | 2.01e-08–2.14e-08 | 8.05e-08 | 8.60e-09 | 7.34e-11 | 8.71e-11 | 2.14e-08 / 8.05e-08 | 3.03e-08 | 1.90e-10 |
| production | 48 | 2.01e-08–2.14e-08 | 8.05e-08 | 8.61e-09 | 1.16e-11 | 1.48e-11 | 2.14e-08 / 8.05e-08 | 3.03e-08 | 2.78e-11 |
| production | 56 | 2.01e-08–2.14e-08 | 8.05e-08 | 8.61e-09 | 2.46e-12 | 3.31e-12 | 2.14e-08 / 8.05e-08 | 3.03e-08 | 5.98e-12 |
| production | 64 | 2.01e-08–2.14e-08 | 8.05e-08 | 8.61e-09 | 6.53e-13 | 8.66e-13 | 2.14e-08 / 8.05e-08 | 3.03e-08 | 1.56e-12 |
| production | 80 | 2.01e-08–2.14e-08 | 8.05e-08 | 8.61e-09 | 7.63e-14 | 9.86e-14 | 2.14e-08 / 8.05e-08 | 3.03e-08 | 1.84e-13 |
| production | 96 | 2.01e-08–2.14e-08 | 8.05e-08 | 8.61e-09 | 1.39e-14 | 1.89e-14 | 2.14e-08 / 8.05e-08 | 3.03e-08 | 3.10e-14 |
| strict | 30 | 1.46e-09–1.59e-09 | 1.96e-09 | 1.28e-09 | 1.59e-09 | 1.96e-09 | 2.52e-15 / 1.33e-14 | 3.86e-09 | 3.86e-09 |
| strict | 32 | 7.24e-10–8.21e-10 | 9.02e-10 | 6.98e-10 | 8.21e-10 | 9.02e-10 | 2.52e-15 / 1.33e-14 | 1.95e-09 | 1.95e-09 |
| strict | 36 | 2.08e-10–2.38e-10 | 3.12e-10 | 1.89e-10 | 2.38e-10 | 3.12e-10 | 2.52e-15 / 1.33e-14 | 5.15e-10 | 5.15e-10 |
| strict | 40 | 6.73e-11–7.34e-11 | 8.71e-11 | 5.68e-11 | 7.34e-11 | 8.71e-11 | 2.52e-15 / 1.33e-14 | 1.90e-10 | 1.90e-10 |
| strict | 48 | 1.05e-11–1.16e-11 | 1.48e-11 | 9.67e-12 | 1.16e-11 | 1.48e-11 | 2.52e-15 / 1.33e-14 | 2.78e-11 | 2.78e-11 |
| strict | 56 | 2.29e-12–2.46e-12 | 3.31e-12 | 2.33e-12 | 2.46e-12 | 3.31e-12 | 2.52e-15 / 1.33e-14 | 5.98e-12 | 5.98e-12 |
| strict | 64 | 6.03e-13–6.53e-13 | 8.67e-13 | 4.61e-13 | 6.53e-13 | 8.67e-13 | 2.52e-15 / 1.33e-14 | 1.55e-12 | 1.56e-12 |
| strict | 80 | 7.01e-14–7.64e-14 | 9.98e-14 | 6.00e-14 | 7.63e-14 | 9.86e-14 | 2.52e-15 / 1.33e-14 | 1.86e-13 | 1.84e-13 |
| strict | 96 | 1.22e-14–1.42e-14 | 2.17e-14 | 1.16e-14 | 1.39e-14 | 1.91e-14 | 2.52e-15 / 1.33e-14 | 3.21e-14 | 3.10e-14 |

### (d) N = 128, 256 (3 families x 3 seeds)

| N | cutoffs | M | Gaussian E_complete (max over 10 probes × 9 configs) | slow E_complete | fast E_complete | Gaussian E_mesh | slow E_mesh |
|---|---|---|---|---|---|---|---|
| 128 | production | 18 | 2.87e-08 | 6.74e-08 | 1.98e-08 | 3.40e-09 | 2.81e-09 |
| 128 | production | 36 | 2.84e-08 | 6.73e-08 | 1.95e-08 | 2.27e-12 | 1.97e-12 |
| 128 | production | 64 | 2.84e-08 | 6.73e-08 | 1.95e-08 | 9.16e-15 | 8.20e-15 |
| 128 | strict | 20 | 1.02e-09 | 8.20e-10 | 1.15e-09 | 1.02e-09 | 8.20e-10 |
| 128 | strict | 36 | 2.27e-12 | 1.97e-12 | 2.16e-12 | 2.27e-12 | 1.97e-12 |
| 128 | strict | 64 | 1.04e-14 | 1.35e-14 | 9.45e-15 | 9.19e-15 | 8.19e-15 |
| 256 | production | 22 | 2.76e-08 | 8.01e-08 | 2.00e-08 | 4.52e-09 | 4.86e-09 |
| 256 | production | 40 | 2.74e-08 | 8.01e-08 | 1.91e-08 | 7.56e-12 | 7.89e-12 |
| 256 | production | 80 | 2.74e-08 | 8.01e-08 | 1.91e-08 | 9.25e-15 | 9.62e-15 |
| 256 | strict | 24 | 1.59e-09 | 1.62e-09 | 1.57e-09 | 1.59e-09 | 1.62e-09 |
| 256 | strict | 40 | 7.56e-12 | 7.89e-12 | 7.81e-12 | 7.56e-12 | 7.89e-12 |
| 256 | strict | 80 | 9.93e-15 | 1.48e-14 | 1.06e-14 | 9.25e-15 | 9.62e-15 |

### (d) fitted rates

| N | cutoffs | probe class | quantity | M range | points | LSQ rate | R^2 |
|---|---|---|---|---|---|---|---|
| 128 | production | gaussian | E_mesh (max over tested probes/configs) | 18–48 | 8 | 10.29 | 0.9994 |
| 128 | production | gaussian | E_complete (max over tested probes/configs) | 18–64 | 10 | 0.01 | 0.3255 |
| 128 | production | spectral_slow | E_mesh (max over tested probes/configs) | 18–48 | 8 | 10.17 | 0.9990 |
| 128 | production | spectral_slow | E_complete (max over tested probes/configs) | 18–64 | 10 | 0.00 | 0.1647 |
| 128 | production | spectral_fast | E_mesh (max over tested probes/configs) | 18–48 | 8 | 10.31 | 0.9990 |
| 128 | production | spectral_fast | E_complete (max over tested probes/configs) | 18–64 | 10 | 0.00 | 0.0938 |
| 256 | production | gaussian | E_mesh (max over tested probes/configs) | 22–56 | 8 | 10.36 | 0.9994 |
| 256 | production | gaussian | E_complete (max over tested probes/configs) | 22–80 | 10 | 0.00 | 0.0523 |
| 256 | production | spectral_slow | E_mesh (max over tested probes/configs) | 22–56 | 8 | 10.44 | 0.9990 |
| 256 | production | spectral_slow | E_complete (max over tested probes/configs) | 22–80 | 10 | 0.00 | 0.4046 |
| 256 | production | spectral_fast | E_mesh (max over tested probes/configs) | 22–56 | 8 | 10.24 | 0.9995 |
| 256 | production | spectral_fast | E_complete (max over tested probes/configs) | 22–80 | 10 | 0.02 | 0.3900 |
| 512 | production | gaussian | E_mesh (max over tested probes/configs) | 28–64 | 8 | 10.42 | 0.9995 |
| 512 | production | gaussian | E_complete (max over tested probes/configs) | 28–96 | 10 | 0.01 | 0.2262 |
| 512 | production | spectral_slow | E_mesh (max over tested probes/configs) | 28–64 | 8 | 10.34 | 0.9985 |
| 512 | production | spectral_slow | E_complete (max over tested probes/configs) | 28–96 | 10 | 0.00 | 0.4049 |
| 512 | production | spectral_fast | E_mesh (max over tested probes/configs) | 28–64 | 8 | 10.40 | 0.9991 |
| 512 | production | spectral_fast | E_complete (max over tested probes/configs) | 28–96 | 10 | 0.03 | 0.1698 |
| 128 | strict | gaussian | E_mesh (max over tested probes/configs) | 20–48 | 7 | 10.15 | 0.9997 |
| 128 | strict | gaussian | E_complete (max over tested probes/configs) | 20–48 | 7 | 10.15 | 0.9997 |
| 128 | strict | spectral_slow | E_mesh (max over tested probes/configs) | 20–48 | 7 | 10.00 | 0.9992 |
| 128 | strict | spectral_slow | E_complete (max over tested probes/configs) | 20–48 | 7 | 10.00 | 0.9992 |
| 128 | strict | spectral_fast | E_mesh (max over tested probes/configs) | 20–48 | 7 | 10.29 | 0.9986 |
| 128 | strict | spectral_fast | E_complete (max over tested probes/configs) | 20–48 | 7 | 10.28 | 0.9986 |
| 256 | strict | gaussian | E_mesh (max over tested probes/configs) | 24–56 | 7 | 10.24 | 0.9996 |
| 256 | strict | gaussian | E_complete (max over tested probes/configs) | 24–56 | 7 | 10.24 | 0.9996 |
| 256 | strict | spectral_slow | E_mesh (max over tested probes/configs) | 24–56 | 7 | 10.29 | 0.9993 |
| 256 | strict | spectral_slow | E_complete (max over tested probes/configs) | 24–56 | 7 | 10.29 | 0.9993 |
| 256 | strict | spectral_fast | E_mesh (max over tested probes/configs) | 24–56 | 7 | 10.19 | 0.9993 |
| 256 | strict | spectral_fast | E_complete (max over tested probes/configs) | 24–56 | 7 | 10.19 | 0.9993 |
| 512 | strict | gaussian | E_mesh (max over tested probes/configs) | 30–64 | 7 | 10.32 | 0.9998 |
| 512 | strict | gaussian | E_complete (max over tested probes/configs) | 30–64 | 7 | 10.32 | 0.9998 |
| 512 | strict | spectral_slow | E_mesh (max over tested probes/configs) | 30–64 | 7 | 10.15 | 0.9993 |
| 512 | strict | spectral_slow | E_complete (max over tested probes/configs) | 30–64 | 7 | 10.15 | 0.9993 |
| 512 | strict | spectral_fast | E_mesh (max over tested probes/configs) | 30–64 | 7 | 10.35 | 0.9990 |
| 512 | strict | spectral_fast | E_complete (max over tested probes/configs) | 30–64 | 7 | 10.35 | 0.9990 |

p study (N = 512, mesh-only error, maximum over tested probes):

| cutoffs | p | probe class | M range | points | LSQ rate | R^2 |
|---|---|---|---|---|---|---|
| production | 4 | gaussian | 28–96 | 10 | 4.21 | 0.9995 |
| production | 4 | spectral_slow | 28–96 | 10 | 4.11 | 0.9989 |
| production | 6 | gaussian | 28–96 | 10 | 6.49 | 0.9995 |
| production | 6 | spectral_slow | 28–96 | 10 | 6.43 | 0.9988 |
| production | 8 | gaussian | 28–96 | 10 | 8.87 | 0.9993 |
| production | 8 | spectral_slow | 28–96 | 10 | 8.85 | 0.9988 |
| production | 9 | gaussian | 28–64 | 8 | 10.42 | 0.9995 |
| production | 9 | spectral_slow | 28–64 | 8 | 10.34 | 0.9985 |
| strict | 4 | gaussian | 30–96 | 9 | 4.19 | 0.9995 |
| strict | 4 | spectral_slow | 30–96 | 9 | 4.12 | 0.9987 |
| strict | 6 | gaussian | 30–96 | 9 | 6.46 | 0.9996 |
| strict | 6 | spectral_slow | 30–96 | 9 | 6.37 | 0.9990 |
| strict | 8 | gaussian | 30–96 | 9 | 8.81 | 0.9995 |
| strict | 8 | spectral_slow | 30–96 | 9 | 8.73 | 0.9994 |
| strict | 9 | gaussian | 30–64 | 7 | 10.32 | 0.9998 |
| strict | 9 | spectral_slow | 30–64 | 7 | 10.15 | 0.9993 |

### (d) translation, spectral probes and structural checks

| N | cutoffs | translation ‖Γ_h T‖/‖Γ_ref‖, pair-difference | cached D − H | Γ_ref itself | ‖Γ_L^h T‖ (abs.) |
|---|---|---|---|---|---|
| 128 | production | 1.50e-16 | 2.08e-16 | 3.15e-16 | 2.09e-15 |
| 128 | strict | 1.52e-16 | 2.87e-16 | 3.15e-16 | 2.22e-15 |
| 256 | production | 0.00e+00 | 9.25e-18 | 4.19e-16 | 0.00e+00 |
| 256 | strict | 0.00e+00 | 9.00e-18 | 4.19e-16 | 0.00e+00 |
| 512 | production | 1.04e-16 | 1.83e-16 | 4.92e-16 | 1.82e-15 |
| 512 | strict | 1.03e-16 | 2.70e-16 | 4.92e-16 | 1.81e-15 |

Spectral probes of strong_N512_s1 (eigenvectors of sym(Γ_ref)):

| probe | eigenvalue | ‖Γ_ref u − λu‖/λ | relative gap to neighbours | ‖Tᵀu‖ |
|---|---|---|---|---|
| spec_slow0 | 2.301966 | 7.7e-15 | 0.0109 | 5.9e-16 |
| spec_slow1 | 2.327088 | 6.6e-15 | 0.0108 | 6.4e-16 |
| spec_slow2 | 2.366316 | 6.8e-15 | 0.0166 | 3.6e-16 |
| spec_fast | 17.549015 | 1.1e-15 | 0.0040 | 1.9e-17 |

Translation eigenvalues of sym(Γ_ref) (N = 512): -1.7e-16, 2.0e-16, -1.1e-15; λ_min(positive) = 2.301966, λ_max = 17.549015.
Stricter LatticeFriction (rel_tol 1e-15, degree 20, near_cells 3) vs Γ_ref: 2.66e-15 (operator), 1.33e-14 (largest relative action difference over the non-null probes).
Checks over all 361 valid (configuration, cutoffs, M) cases: dense Γ_h vs matrix-free action ≤ 3.78e-15; symmetry defect ≤ 1.29e-17; negative eigenvalue on Range Π: False; cached vs pair-difference ≤ 2.16e-15; E_complete ≤ E_mesh + E_trunc in every row: True.

