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
