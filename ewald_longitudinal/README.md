# Ewald split of the screened pure-longitudinal kernel: numerical verification

```
python3 verify_ewald_longitudinal.py                 # kappa = 1, xi = 1 (about 25 s)
python3 verify_ewald_longitudinal.py --kappa 3 --xi 0.7 [--no-fft] [--outdir DIR]
```

Requires NumPy, SciPy and Matplotlib. Outputs:

- `results/summary_kappa{κ}_xi{ξ}.txt` and `.json`: every metric, plus PASS/FAIL for items 1–11.
- `figures/fig1…fig8`: splitting error, positivity and small-r behaviour, real-space and Fourier
  asymptotics with slope fits, the two cutoff tails, numerical/asymptotic ratios, and the
  Fourier cross-checks with the eigenvalues of K̂_L.
- `DERIVATIONS.md`: independent re-derivation of every formula, plus observations.

## How it is computed

- **Primary quantities come straight from the task's s-integrals.** θ_L, θ_S, A and B use composite
  Gauss–Legendre rules. Their panels are graded where the integrands have near-singular or
  boundary-layer behaviour.
- **Exponentially small values are carried in scaled form**: θ_S e^{ξ²r²}, (A, B) e^{k²/4ξ²},
  E_S e^{ξ²rc²} and E_F e^{kc²/4ξ²}. This keeps logs and ratios accurate out to r = 12 and k = 40.
  - The θ_S integral uses s = ξ² + t/r².
  - The A and B integrals use s = 1/(v + ξ⁻²) with v = 4τ/(k² + κ²). That integrand decays like e^{-τ} for
    every k ≥ 0, including k = 0.
- **Each primary quantity has independent cross-checks:**
  - adaptive `scipy.integrate.quad` on the literal formulas;
  - a closed form for θ_S;
  - a radial Hankel transform of θ_L for A and B;
  - a 192³ FFT of the full tensor K_L, compared with A I + B k k^T;
  - two different quadrature routes for each tail E_S and E_F.
- The model is not changed. ρ_κ and h are evaluated in algebraically identical erfcx forms that
  avoid cancellation, and the script checks them against the literal expressions.

## Results for κ = 1, ξ = 1 (z = 0.5, h = 0.3538549)

| check | result |
|---|---|
| formula cross-checks (quad, closed form, Hankel, FFT) | 1e-16 … 2e-11 |
| max splitting error, scalar / tensor | 3.6e-15 / 1.9e-15 |
| θ_S, θ_L > 0; transverse eigenvalues / θ | yes; ≤ 4.5e-16 |
| C = ∫₀^{ξ²}ρ: quadrature vs closed form; small-r fit | 0.1729020066, agree to 1e-15; fit to 2e-10 |
| slope of log θ_S vs r², window r ∈ [11, 12] | −1.00006 (target −1) |
| slope of log‖K̂_L‖ vs k², window k ∈ [38, 40] | −0.2499998 (target −0.25) |
| θ_S / asymptotic at r = 12 | 1.0076 (next-order prediction 1 + 1.1004/r²) |
| A / asymptotic at k = 40 | 0.9985 (prediction 1 − 2.4018/k²) |
| ‖K̂_L‖ / asymptotic at k = 40 | 0.99975 (prediction 1 − 0.4018/k²) |
| E_S / asymptotic at rc = 8 | 1.0248 (prediction 1 + 1.6004/rc²) |
| E_F / asymptotic at kc = 30 | 1.0018 (prediction 1 + 1.5982/kc²) |

## Periodic particle graph Laplacian (`verify_periodic_graph_psd.py`)

```
python3 verify_periodic_graph_psd.py        # case A: N=30, L=6; case B: N=40, L=10; plus a sweep (~50 s)
```

This builds Γ_full, Γ_S and Γ_L from periodic pair kernels summed directly over images
(converged to about 1e-15), then checks each Laplacian for PSD, the split, the null modes and the
quadratic-form identity. It then builds Γ_L from a truncated Fourier series at increasing kc.
No eigenvalue is clipped. The report is written to `results/periodic_graph_psd_kappa1_xi1.txt`.

## Automatic parameter selection (`test_parameter_selection.py`)

```
python3 test_parameter_selection.py        # about 5.5 min
```

This tests the balanced choice ξ·rc = kc/(2ξ) = s together with a PPPM long-range operator.
The PPPM uses B-spline order p, the influence function K̂_L/|Ŵ|², and adjoint gather. Both
parts are assembled as degree-minus-weight graph Laplacians and compared with the exact
periodic Γ (N = 30, L = 6 and N = 40, L = 10). The error is split into real-cutoff,
Fourier-cutoff, alias and transfer parts. Results, tables, the 8 plots and
`final_report.md` are in `parameter_selection_results/`.

## Follow-ups (`test_equal_accuracy_xi.py`, `test_mesh_error_decomposition.py`)

- `test_equal_accuracy_xi.py` (about 13 min): for each ξ and tol ∈ {1e-5, 1e-7, 1e-9} it searches s, η_N, p and
  an FFT-friendly M until the actual operator error meets tol in both dense configurations. It then times the
  cheapest accepted sets on A, B and an N = 4000 system. Output: `equal_accuracy_xi_results/`.
- `test_mesh_error_decomposition.py` (about 5 min): exact D_h + α decomposition of the PPPM error, checked across
  three implementations. Also a clean h^p test at fixed k_c with 64 random mesh offsets, the C_M and q^p
  predictors, and PSD checks. Output: `mesh_error_decomposition_results/`.
- The combined answers are in `FOLLOWUP_REPORT.md`.

## Lanczos discrete-FDT thermostat (`test_lanczos_fdt.py`)

```
python3 test_lanczos_fdt.py        # about 35 min
```

This validates η = f_dt(Γ_h)Πξ computed by Lanczos with full reorthogonalization, using the literal PPPM Γ_h
action. The Lanczos results are compared with dense references for configurations A and B and ten more
configurations. The script also measures the ranks needed, tests stopping rules, and separates the discrete-FDT
covariance error into its Krylov and Monte Carlo parts. It covers the Euler variant and rank/time scaling up to
N = 4000. Answers are in `lanczos_fdt_results/REPORT.md`.

## Range-start vs direct-start Lanczos (`test_lanczos_range_start.py`)

```
python3 test_lanczos_range_start.py        # about 13 min
```

This compares Landau-style range-start Lanczos (b = ΠΓ_h z, g = f/λ) with direct start. It uses the same Γ_h,
vectors and seeds as `test_lanczos_fdt.py`, and covers the dense configurations, scaling to N = 4000, timing,
stopping rules and a threshold ablation. Answers are in `lanczos_range_start_results/REPORT.md`.

## Ideal-gas dynamics test (`test_true_dynamics.py`)

This is the first time-dependent test: an ideal gas under the A–O–A finite-time FDT thermostat with the PPPM Γ_h.
It compares dense, high-rank Lanczos, rank-40 and rank-48 runs on coupled streams, with nested dt/4…2dt
refinement, a tight-PPPM spatial proxy, long equilibrium runs, a negative control, and N = 4000 and 16000
production runs with timing. Run it in stages (`--stage small|control|equil|large|timing|report`). The answers are
in `dynamics_results/REPORT.md`.
