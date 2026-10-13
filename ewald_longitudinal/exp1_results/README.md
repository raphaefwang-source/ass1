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
- The first seven stages take about 16 min on 4 cores. `precision` (80-bit evaluation) takes about 19 min.
- Every invocation is appended to `manifest.json` with its command, git commit, versions, host and threads.
- Logs:
  - `run_2026-10-13_initial_rounded_k.log`: the first full run. Its `configs`, `reference` and `quadrature` stages
    are the ones still in use. Its later stages were computed with the rounded-|k| coefficients (see below) and
    are superseded.
  - `run_2026-10-13_exact_k.log`: the rerun of `split`, `cutoffs`, `mesh`, `action` and `precision` after the
    fix. All numbers in the files and in the LaTeX section come from this run.
  - Provenance: the `manifest.json` entries of this rerun record commit 257dff5 with `dirty=true`, and 257dff5
    still contains the rounded `khat_AB`. The code that produced these results is the working tree of 257dff5 plus
    the `khat_AB` fix and the new precision variants. It is committed together with the results, in the commit that
    adds `run_2026-10-13_exact_k.log`. Later edits change only docstrings, comments, figure layout and the summary
    and table post-processing of `exp1_plots.py`.
- The dense matrices (Γ_ref and the image sums Γ_S, Γ_L over |r + mL| ≤ 45) are cached in `raw/` and are not committed. They are
  regenerated deterministically by the `reference` stage.

## Wave-number rounding fix (2026-10-13)

`khat_AB` in `test_parameter_selection.py` evaluated A(k), B(k) at `np.round(|k|, 12)`, so the kernel input was
perturbed by up to 5e-13. Every Fourier-side quantity of this experiment uses `khat_AB`:

- the operator-level split check;
- the Fourier cutoff sweep;
- the mesh scans and the alias decomposition (through `Mesh`);
- the production `GammaH` (through `make_mesh` → `Mesh`);
- the precision check.

The rounding was the dominant source of the former ~1e-13 Fourier-vs-image-sum floor. The earlier precision check
(float64, 80-bit and refined quadrature) could not detect it, because all three variants shared the rounded
coefficients. `khat_AB` now deduplicates on the unrounded float64 |k|. `precision` reproduces the rounded path explicitly
(`floor_rounded12`) and also evaluates the coefficients on integer shells |m|², with k computed from the shell
(`floor_int_shell`).

The configurations, the reference matrices in `raw/`, `reference_ladder.csv` and `quadrature.csv` do not involve
`khat_AB` and were reused.

The same rounding pattern was also removed from `verify_periodic_graph_psd.py`, which is outside the Experiment 1
path. Other scripts that import `khat_AB` belong to other experiments: `test_parameter_selection.py`,
`test_equal_accuracy_xi.py`, `test_mesh_error_decomposition.py` and `test_radial_kernel_static.py`. Their stored
outputs predate the fix and were not rerun.

## Main manuscript and Experiment 2

The repository contains only this section (`paper/sec6_1_experiment1.tex`, a drop-in replacement for Section 6.1),
its tables and a standalone build. The main manuscript and the text of Experiment 2 are not in the repository.
Section 6.1 now states that the equal-accuracy timings of Experiment 2 are equal accuracy on cases A and B only.
The corresponding sentence in the Experiment 2 section, and any abstract or conclusion that quotes T5/T7/T9
accuracy or the former 1e-13 floor, must be updated in the manuscript itself.

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
| `fourier_precision.csv` | Mesh-free Fourier lattice sum (k_c = 2ξ·6.5) against the image-sum Γ_L, for case A, lattice_N128_s1 and uniform_N128_s1 at ξ = 0.7 and 1. Sources are separated:<br>• wave-number rounding: `floor_rounded12` (old path) vs `floor_float64` (unrounded \|k\|) vs `floor_int_shell`;<br>• quadrature: `floor_refined_coeffs`;<br>• floating point: `floor_longdouble` (whole sum evaluated in 80 bits) and `float64_vs_longdouble`;<br>• reference truncation and round-off: `GL_images_R40_vs_R45`. |
| `cutoffs.csv`, `fits_cutoffs.csv` | real-space and Fourier truncation sweeps (no other approximation), with flags for steps that admit no new pair image or lattice shell; Gaussian-rate fits (flat and empty steps and points below the fit floor excluded) |
| `mesh.csv`, `fits_mesh.csv` | Γ_L^mesh − Γ_L^{k_c} (RMS, max and min over grid shifts). Scans:<br>• s = 4.1 (production k_c): p = 3…8, nine meshes, 8 shifts, main and legacy;<br>• s = 5.5: p = 4, 6, 8, seven meshes, 8 shifts, five configurations;<br>• N = 256: p = 5, 7, six meshes, 4 shifts.<br>Fits, all on the RMS error:<br>• slope in h on the four finest meshes above the fit floor, and the slope of the max on the same meshes;<br>• slope over every mesh above the fit floor;<br>• local slopes;<br>• coarse-mesh exponent (η ≥ 0.5). |
| `mesh_decomposition.csv` | full D_h + H(α_i* + α_j + α_i*α_j) decomposition: identity residual, linear / quadratic / D_h shares, literal vs mode-space |
| `action.csv` | production `GammaH.matvec` at T5 / T7 / T9: spectral and Frobenius error, error budget (ε_S, ε_F, ε_mesh), literal vs mode-space, symmetry, translations, λ_min on T⊥, Weyl margin, timing |
| `action_probes.csv` | every probe: Gaussian, longitudinal and transverse k_min modes, and translations; relative and operator-normalized errors |
| `summary.json` | every number quoted in the LaTeX section |
| `figures/` | `exp1_components` and `exp1_action` (PDF and PNG); the PDFs are also copied to `paper/figures/` |

## Fit floor

The fit floor is set by the numerical floor ε_num and is computed in `exp1_plots.py`. ε_num is the largest
numerical discrepancy measured: the operator-level split identity, and every column of `fourier_precision.csv`
except the rounded-|k| one. Those columns are the Fourier-sum variants, the float64 vs 80-bit difference and the
R = 40 vs 45 image-sum change. Here ε_num = 1.1e-15, so the fit floor 100·ε_num = 1.1e-13, and the floor contributes
at most 1% to any fitted error.

Points below the fit floor stay in the CSV files. In panels (b) and (c) of `exp1_components` they lie below the
dotted line. In the mesh scans two meshes are excluded: s = 5.5, p = 8, η ≈ 0.25, for case A and uniform_N128_s1.
`summary.json` lists them under `numerical_floor`, `cut_*` and `mesh_*_below_floor`.

## Notes on interpretation

- **Relative errors** are divided by ‖Γ_ref‖₂, which spans 0.72–35 over the configurations. Close pairs strongly
  influence it, but the nearest pair alone does not set it: ‖Γ_ref‖₂/θ(d_min) is 2.0–4.1, and an isolated pair
  gives exactly 2.
- **Absolute errors:** at fixed parameters ‖Γ_h − Γ_ref‖₂ varies by about a factor 10 at T7 (44 configurations)
  and by a factor 4–6 at T5/T9 (14 configurations). The numerator depends on the configuration too.
- **Tolerance transfer:** 13 of the 72 (configuration, set) pairs exceed the relative tolerance that T5/T7/T9 were
  accepted for on cases A and B, by up to a factor 2.5. These are:
  - the perturbed lattices, at all three sets;
  - `fixn_L6_N30_s3` at T7 (1.40e-7). This is a uniform draw with the same N, L and generator as case A.
- **Action errors:** a relative operator error within tolerance does not imply relative action errors within it.
  At T7 every configuration that meets 1e-7 in operator norm has a transverse k_min probe above 1e-7. Among
  configurations within the operator tolerance, the largest action error is 4.5× the tolerance at T7 and 4.9× at T9.
- **Fourier floor:** after the fix, the mesh-free Fourier sum agrees with the real-space image sum to ~1e-16–1e-15
  of ‖Γ_ref‖₂. This remaining floor combines floating-point arithmetic (float64 vs 80-bit sums differ by up to 6.7e-16)
  with the numerical uncertainty of the image-sum reference (R = 40 vs 45 changes it by up to 4.3e-16). The two are
  not separated (`fourier_precision.csv`).
- **Mesh slopes:** on the four finest meshes (η ≈ 0.23–0.50) the slopes are consistent with the predicted h^p
  trend. Finite-mesh fits do not establish the asymptotic order. On coarser meshes the exponent is 1.1–1.3·p.
