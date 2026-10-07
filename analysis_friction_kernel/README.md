# Radial longitudinal friction law: Model A vs Model B

**Question.** Which pair-friction law is better supported by MD or coarse-grained dissipation data?

- Model A: γ_A(r) = A e^{−κr}/r
- Model B: γ_B(r) = B r e^{−κr}

Both are used as γ(r) P_L(r), with P_L = r̂r̂ᵀ.

**Answer.** Undetermined. This repository contains no MD or coarse-grained dissipation data of any kind. The only
particle trajectories come from this project's own CG model, whose input kernel is Model A with κ = 1. Fitting
them would be circular, so no physical model preference is claimed.

What is provided instead:

- a reusable pipeline for the three data cases;
- known-answer tests showing that the pipeline can tell A from B, and when it fails to;
- the Fourier and real-space comparison of the two analytic kernels.

```
python3 run_analysis.py            # full run (about 10 min, 4 threads); --quick for a smaller test
python3 run_analysis.py --report-only
python3 inspect_data.py            # inventory only
```

## Inventory (`inspect_data.py`, `results/inventory.json`)

- **MD-related files:** none. There are no LAMMPS inputs, dumps (`*.lammpstrj`, `*.dump`, `*.xyz`, `*.dcd`,
  `*.xtc`, …), topology or `data` files, atom-to-molecule maps, force dumps, constrained-dynamics or
  unresolved-force series, memory kernels or trained friction models.
- **Other branch:** the repository's other branch (`claude/optimistic-bell-qkn0l2`) contains only a finite-element
  notebook.
- **NumPy archives:** 46 hold observables only (temperatures, mode amplitudes). The other 12 hold q(t) and p(t) from
  `ewald_longitudinal/test_dynamic_correlations.py`. Those are output of this project's ideal-gas CG model, with
  friction Γ_h built from e^{−κr}/r P_L and κ = 1. They are excluded from git.
- **Classification:** CASE 0. There are not even ordinary MD positions and velocities.

## Conventions

- **Pair friction.** Lyu–Lei write the Markovian limit as K(Q,t) ≈ −Γ(Q)δ(t), with Γ positive semidefinite and
  Γ_ii = −Σ_{j≠i} Γ_ij. The pair friction is therefore taken as G_ij = −Γ_ij (i ≠ j), so that DPD-like friction
  gives G_ij = +γ(r) P_L. From it:
  - γ_L = r̂ᵀ G_ij r̂
  - γ_T = ½ tr(P_T G_ij)
- **Memory (CASE 2).** K_pair(r,t) = −β⟨δF_i(t) δF_j(0)ᵀ⟩, where δF = F − ⟨F | Z⟩ comes from constrained dynamics.
  The effective friction is the one-sided Green–Kubo integral G_eff(r; T) = ∫₀ᵀ K_pair dt, and convergence in T is
  checked explicitly.
- **Uncertainties.**
  - Errors are configuration-level: the standard error of the bin mean across independent configurations or time
    blocks. No frame-level bootstrap is used.
  - The bootstrap resamples configurations (V1, V2) or 100-step trajectory blocks (V3).
  - The conditional standard deviation at fixed r is reported separately and never hidden.
- **Model predictions** are exact bin averages over the sampled pair distances, so the 1/r law has no Jensen bias in
  wide bins. Only bins with at least 30 samples are fitted, so nothing is extrapolated toward r = 0.
- **Spline baseline.** A weighted smoothing spline of u = rγ, evaluated as u/r. It is regular for both laws.

## Modules

| file | role |
|---|---|
| `inspect_data.py` | recursive inventory and CASE classification |
| `load_data.py` | LAMMPS text-dump reader, generic npz reader, loader for own-model trajectories |
| `coarse_grain.py` | atom → molecule mapping: unwrap molecules, centre of mass, total momentum, total force |
| `estimate_memory.py` | CASE 1: tensor → pair friction. CASE 2: constrained-force pair memory, cumulative integrals and split-half noise estimate. CASE 3: Markovian drift regression (a diagnostic only) |
| `radial_projection.py` | pairs, L/T projections, radial binning with group-level errors |
| `fit_models.py` | weighted NLS on bin averages, spline baseline, metrics (χ², weighted and unweighted RMSE, MAE, R², AIC/BIC), group bootstrap |
| `fourier_analysis.py` | closed-form and QAWF-quadrature Fourier eigenvalues of both tensor kernels |
| `plotting.py` | figures |
| `run_analysis.py` | driver; writes `results/` and `figures/` |

## Known-answer tests (synthetic; not evidence about MD)

| test | data path | input law | outcome |
|---|---|---|---|
| V1 | CASE 1: Γ_h(q) of this project's model, 6 ideal-gas configurations, N = 512 | A, κ = 1 | A recovered; B rejected |
| V2a | CASE 2: synthetic constrained forces with exponential memory τ = 0.2, core d = 0.9 | pure B, κ = 1 | B recovered; A rejected |
| V2b | as V2a, with a many-body environment factor s_i s_j | B × environment | B preferred, but every pairwise law misfits (χ²/dof > 1) |
| V3 | CASE 3: Markovian drift regression on 8 own-model trajectories | A, κ = 1 | A recovered, with an O(dt) bias of about 10% |

The numbers are in `results/summary.json`, the per-bin tables in `results/V*_bins.csv`, and the figures, panels (a)
to (i), in `figures/`.

### Lessons for real data

1. **An excluded-volume core removes the region where A and B differ most** (1/r against r as r → 0).
   - On the sampled range, Model A with a small κ can mimic B.
   - With 3 configurations × 4000 force samples, an earlier test run selected the wrong model (A, with ΔAIC = 5 in
     its favour) when the input was B × environment.
   - With 6 × 8000 it selects B correctly. Model selection must therefore be accompanied by a power check.
2. **Per-pair memory integrals are noise-dominated** unless the constrained runs are very long. In V2b the per-pair
   estimation variance is about 200× the true many-body variance. Many-body dependence shows up first as a χ²/dof
   misfit of all pairwise laws, and only much later as resolvable per-pair scatter.
3. **A drift regression from ordinary trajectories estimates (I − e^{−dtΓ})/dt, not Γ,** and only for an assumed
   Markovian pairwise form with the conservative force removed. It is not a Mori–Zwanzig estimate.

## Fourier space (`figures/fourier_eigenvalues.png`)

The convention is F[f](k) = ∫ e^{−ik·r} f dr. For g(r) r̂r̂ᵀ the transform is λ_T(I − k̂k̂ᵀ) + λ_L k̂k̂ᵀ, with:

- λ_T = 4π∫ g r² j₁(kr)/(kr) dr
- λ_L = 4π∫ g r² [j₀ − 2j₁/(kr)] dr

**Model A** (closed form derived here):

- λ_T = 4πA[k − κ arctan(k/κ)]/k³
- λ_L = 4πA/(k² + κ²) − 2λ_T
- Both decay as k⁻². λ_L becomes negative for k > 1.515κ (asymptotically −4πA/k²).

**Model B:**

- K̂_B = 8πB/(k² + κ²)² I − 32πB/(k² + κ²)³ kkᵀ
- λ_T = 8πB/(k² + κ²)²
- λ_L = 8πB(κ² − 3k²)/(k² + κ²)³, which is negative for k > κ/√3 and tends to −24πB/k⁴.

The closed forms match QAWF quadrature to 8e-10. Both continuum kernels are indefinite. Positive semidefiniteness
of the particle friction comes from the graph-Laplacian construction with positive pair weights, not from the
continuum transform. The smoother k⁻⁴ decay of B is a numerical convenience only, and it must not be used to choose
between the models.

## What data would decide the question

1. **Atomistic MD of the target system:** per-atom positions, velocities and forces, molecule ids, masses, and β.
   `load_data.stack_dump` and `coarse_grain.com_map` read these.
2. **Constrained dynamics (Lyu–Lei):** for many independent CG configurations Z, run atomistic dynamics with the CG
   coordinates held fixed and record the unresolved forces δF_i(t) = F_i − ⟨F_i | Z⟩.
   - Each run must be long compared with the memory time.
   - From V2, a run length of several hundred memory times is needed per configuration for bin-level accuracy. Many
     more configurations are needed to resolve per-pair (many-body) scatter.
   - `estimate_memory.pair_memory_from_constrained_forces` consumes these directly.
3. **Alternatively, an already-trained friction tensor Γ(Q)** for sampled configurations: CASE 1,
   `pair_friction_from_tensor`.
4. **A power check:** repeat the known-answer test V2a at the real system's density, core size and run length to
   confirm that A and B are distinguishable before interpreting ΔAIC.
