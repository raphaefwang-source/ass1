# Experiment 1 (manuscript Section 6.1): complete particle–mesh operator action

These files hold the data, fits and figures behind `paper/sec6_1_experiment1.tex`.

The model is the screened pure-longitudinal kernel K(r) = e^{-κr}/r P(r) with κ = 1, in a fully periodic cube with
edge length L. The k = 0 mode is retained. There is no free-space truncation and no energy null space.

## Reproduce

```bash
cd ewald_longitudinal
python3 exp1_operator_action.py all --workers 4   # all stages, including the slow 80-bit precision check
python3 exp1_plots.py                             # fits, figures, LaTeX tables, summary.json (~20 s)
cd ../paper && pdflatex exp1_standalone.tex        # standalone build of the section, for checking
```

- Stages can be run one at a time: `configs`, `reference`, `quadrature`, `split`, `cutoffs`, `mesh`, `action`,
  `precision`.
- The first seven stages take about 16 min on 4 cores. `precision` (80-bit accumulation) takes longer.
- Every invocation is appended to `manifest.json` with its command, git commit, versions, host and threads.
- `run_all.log` is the log of the run reported here.
- The dense matrices (Γ_ref and the exact Γ_S, Γ_L) are cached in `raw/` and are not committed. They are
  regenerated deterministically by the `reference` stage.

## Seeds

- Cases A and B: `default_rng(7)` and `default_rng(13)`. These are the original draws of
  `verify_periodic_graph_psd.py`.
- All other configurations: `default_rng([20261013, family code, N, seed])`, with family code 1 = uniform,
  2 = perturbed lattice, 3 = clustered, and seed ∈ {1, 2, 3}.
- Each configuration's generator record is stored in the `generator` column of `configs.csv`.
- Grid shifts: `default_rng([20261013, 7, N, seed, round(100 s_F)])`. Shift 0 is the unshifted grid.
- Probe vectors: `default_rng([20261013, 11, N, seed])`.

## Files

| file | content |
|---|---|
| `configs.csv`, `configs.npz` | 44 configurations: N, L, n = N/L³, κL, spacing, d_min, nearest-neighbour statistics, generator; positions |
| `reference_ladder.csv` | Γ_ref (unsplit kernel, lattice sum over \|r + mL\| ≤ R) for R = 15…45: change relative to R = 45 and to the next R; ‖Γ_ref‖₂, λ_min on T⊥, symmetry, translation residual |
| `quadrature.csv` | A(k), B(k), θ_L, θ_S under refined Gauss–Legendre rules; closed-form vs quadrature θ_S |
| `split_pairs.csv` | θ_S + θ_L − e^{-κr}/r (θ_L from its own s-integral): pair images and radial grid, ξ ∈ {0.5, 0.7, 1, 1.4, 2} |
| `split_operator.csv` | Γ_S (images) + Γ_L (mesh-free lattice Fourier sum, k_c = 2ξ·6.5) vs Γ_ref |
| `fourier_precision.csv` | floor of the mesh-free Fourier lattice sum against the image-sum Γ_L: float64 accumulation, 80-bit (`np.longdouble`) accumulation, and refined quadrature coefficients; case A, lattice_N128_s1, uniform_N128_s1 at ξ = 0.7 and 1 |
| `cutoffs.csv`, `fits_cutoffs.csv` | real-space and Fourier truncation sweeps (no other approximation), with flags for steps that admit no new pair image or lattice shell; Gaussian-rate fits (flat and empty steps excluded) |
| `mesh.csv`, `fits_mesh.csv` | Γ_L^mesh − Γ_L^{k_c} (RMS, max and min over grid shifts). Scans: s = 4.1 (production k_c), p = 3…8, nine meshes, 8 shifts, main and legacy; s = 5.5, p = 4, 6, 8, seven meshes, 8 shifts, five configurations; N = 256, p = 5, 7, six meshes, 4 shifts. Fits: h^p on the four finest meshes with error ≥ 1e-12, the slope of the max, and the coarse-mesh exponent (η ≥ 0.5) |
| `mesh_decomposition.csv` | exact D_h + H(α_i* + α_j + α_i*α_j) decomposition: identity residual, linear / quadratic / D_h shares, literal vs mode-space |
| `action.csv` | production `GammaH.matvec` at T5 / T7 / T9: spectral and Frobenius error, error budget (ε_S, ε_F, ε_mesh), literal vs mode-space, symmetry, translations, λ_min on T⊥, Weyl margin, timing |
| `action_probes.csv` | every probe: Gaussian, longitudinal and transverse k_min modes, and translations; relative and operator-normalized errors |
| `summary.json` | every number quoted in the LaTeX section |
| `figures/` | `exp1_components` and `exp1_action` (PDF and PNG); the PDFs are also copied to `paper/figures/` |

## Notes on interpretation

- Relative errors are divided by ‖Γ_ref‖₂. This norm is set by the closest pairs and spans 0.72–35 over the
  configurations.
- At fixed parameters the absolute error ‖Γ_h − Γ_ref‖₂ varies by less than a factor 10.
- The perturbed lattices, which have no close pairs, exceed the relative tolerances that the T5/T7/T9 sets were
  accepted for on cases A and B, by up to a factor 2.6.
- The mesh-free Fourier lattice sum has a float64 floor of about 1e-13 of ‖Γ_ref‖. For its precision checks, see
  `fourier_precision.csv`. Mesh-error points below 1e-12 are excluded from the h^p fits for this reason.
- The h^p rate is observed for η ≈ 0.23–0.50. On coarser meshes the exponent is 1.1–1.3·p.
