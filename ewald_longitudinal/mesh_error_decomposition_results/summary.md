# Mesh error decomposition: summary

Tests: 203 fixed-configuration meshes, 203 phase-averaged meshes (64 offsets each).
- literal spread/FFT/G/IFFT/gather vs mode-space matrix: max 1.4e-15
- identity K_h - K_F = D_h + H(alpha_i^* + alpha_j + alpha_i^* alpha_j): max |deviation| / max|K_F| = 1.8e-15 (roundoff level); relative to the error itself 5.6e-15..8.0e-05 (largest only where the mesh error itself approaches roundoff)
- truncated Poisson sum (|n| <= 1000) vs exact stencil DFT, relative to the alias part: max 9.7e-05 (p=2 converges slowest); relative to |S|: max 3.1e-05
- max |D_h|/|Khat_L| = 1.8e-16; operator contribution of D_h max 4.7e-17; excluded influence modes: 0
- linear part / total: 0.919..1.031; quadratic / total: 2.5e-11..9.2e-02
- old 'transfer' = linear + quadratic(n != n'): max deviation 8.0e-05; old 'alias' = quadratic(n = n') part

| config | kc | p | slope (finest 4) | h fitted | asymptotic C_M | C_M spread (finest 3) | h range with local slope within 20% of p |
|---|---|---|---|---|---|---|---|
| A | 4 | 2 | 2.02 | 0.094-0.188 | 0.021 | 1.10 | 0.125-0.500 |
| A | 4 | 3 | 2.96 | 0.094-0.188 | 0.00589 | 1.08 | 0.125-0.500 |
| A | 4 | 4 | 4.26 | 0.094-0.188 | 0.00272 | 1.23 | 0.125-0.375 |
| A | 4 | 5 | 5.07 | 0.094-0.188 | 0.00112 | 1.05 | 0.125-0.375 |
| A | 4 | 6 | 6.30 | 0.094-0.188 | 0.000592 | 1.20 | 0.125-0.375 |
| A | 4 | 7 | 7.22 | 0.094-0.188 | 0.000277 | 1.04 | 0.125-0.375 |
| A | 4 | 8 | 8.76 | 0.125-0.250 | 0.000164 | 1.26 | 0.150-0.375 |
| A | 6 | 2 | 2.15 | 0.094-0.188 | 0.0216 | 1.15 | 0.125-0.250 |
| A | 6 | 3 | 2.86 | 0.094-0.188 | 0.00645 | 1.18 | 0.125-0.500 |
| A | 6 | 4 | 4.18 | 0.094-0.188 | 0.00329 | 1.17 | 0.125-0.375 |
| A | 6 | 5 | 4.91 | 0.094-0.188 | 0.00159 | 1.19 | 0.125-0.375 |
| A | 6 | 6 | 6.31 | 0.094-0.188 | 0.000946 | 1.20 | 0.125-0.375 |
| A | 6 | 7 | 7.27 | 0.094-0.188 | 0.000577 | 1.09 | 0.125-0.250 |
| A | 6 | 8 | 8.67 | 0.094-0.188 | 0.000382 | 1.41 | 0.125-0.250 |
| B | 4 | 2 | 1.96 | 0.156-0.312 | 0.0329 | 1.07 | 0.208-0.625 |
| B | 4 | 3 | 3.29 | 0.156-0.312 | 0.0104 | 1.31 | 0.208-0.625 |
| B | 4 | 4 | 4.33 | 0.156-0.312 | 0.00449 | 1.16 | 0.208-0.500 |
| B | 4 | 5 | 5.74 | 0.156-0.312 | 0.00214 | 1.62 | 0.208-0.500 |
| B | 4 | 6 | 6.74 | 0.156-0.312 | 0.00114 | 1.41 | 0.208-0.500 |
| B | 4 | 7 | 8.17 | 0.156-0.312 | 0.000642 | 2.03 | 0.208-0.500 |
| B | 4 | 8 | 9.12 | 0.156-0.312 | 0.000381 | 1.83 | 0.208-0.312 |
| B | 6 | 2 | 1.97 | 0.156-0.312 | 0.0351 | 1.23 | 0.417-0.417 |
| B | 6 | 3 | 3.48 | 0.156-0.312 | 0.0113 | 1.39 | 0.208-0.312 |
| B | 6 | 4 | 4.35 | 0.156-0.312 | 0.00607 | 1.37 | 0.208-0.500 |
| B | 6 | 5 | 5.94 | 0.156-0.312 | 0.00298 | 1.61 | 0.208-0.312 |
| B | 6 | 6 | 6.88 | 0.156-0.312 | 0.00206 | 1.54 | 0.208-0.312 |
| B | 6 | 7 | 8.41 | 0.156-0.312 | 0.00129 | 1.87 | 0.208-0.208 |
| B | 6 | 8 | 9.55 | 0.156-0.312 | 0.00108 | 2.06 | 0.208-0.312 |

Predictors vs the 181 parameter-selection mesh runs (pred/actual):
- pred_CM_h: median 0.932, range 0.248..2.07, rms |log10| 0.14
- pred_CM_eta: median 0.977, range 0.307..2.15, rms |log10| 0.12
- qp: median 1.92e+03, range 36.8..4e+05, rms |log10| 3.56

PSD classes over all fixed + offset meshes: {'A': 9882, 'B': 3313, 'C': 0}; Weyl bound held: 203/203 fixed, 203/203 offset sets
max symmetry residual 8.4e-17, max translation residual 6.5e-17, min lambda_h/lambda_* 0.9704
actual gap shift / ||Gamma_h - Gamma||_2: median 2.4e-02, max 0.07
