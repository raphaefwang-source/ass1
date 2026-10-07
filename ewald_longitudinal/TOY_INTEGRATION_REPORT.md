# Toy application on the Ewald/PPPM longitudinal-friction project

Double-well and Lennard-Jones pair potentials, radial friction laws A and B, the project's PPPM
operator Γ_h and finite-time FDT thermostat, compared with dense full-periodic references.
This is a **toy application**. It is not a reproduction of Lyu & Lei, PRL 131, 177301 (2023), and no figure
here validates real MD. The friction is pure longitudinal, K = g(r) r̂r̂ᵀ, with no transverse kernel.

## Summary by validation layer (read this first)

The layers follow the error chain of the fast-particle Landau preprint (Zhao, Dou, Lei): from the most local
diagnostic to the most integrated, so that a good-looking trajectory plot cannot hide an operator, sampling or
time-discretisation floor.

**Layer 1 — operator accuracy** (static configurations, no time integration; Sections 2 and 8)
- Γ_h (production PPPM) vs two independent dense full-periodic references, on 6 fixed configurations: relative
  spectral error ≤ 1.3e-7 (law A) and ≤ 4.6e-8 (law B). The tight PPPM set gives ≤ 1.8e-9.
- Error split: real cutoff, Fourier cutoff, mesh alias and mesh transfer reported separately.
  - The exact split is ξ-independent to 1e-15.
  - The ξ dependence of the total error comes from mesh transfer.
  - k = 0 is retained.
- Structure and FDT:
  - PSD is Weyl-certified; total-momentum null space and symmetry hold to ~1e-16;
  - frozen-q FDT identity holds to 4e-15;
  - Lanczos 40/16 matches dense to 3e-15.
- On 130 configurations per case visited by the long runs, the operator error is ≤ 1.5e-7.

**Layer 2 — fixed-time statistical validation** (`toy_fixed_time_results/REPORT.md`, Section 9)
- Setup: double_well_A, N = 64, 5 initial states × 16 paired noise paths, t = 0.5, 1, 2. The dense step ladder
  2dt…dt/4 uses nested coupled noise.
- **fast − dense:** 95 % bounds ≤ 2e-5 (≤ 1e-6 relative) for K/N, U/N, ⟨|v|⁴⟩ and MSD. One resolved but tiny systematic
  MSD difference (≈ 2e-8 relative), consistent with the 1e-7 operator error.
- **Dense time-step bias:**
  - at the production dt it is not resolved; it is only bounded, e.g. |K/N| ≤ 2.7e-3 at t = 0.5;
  - at 2dt it is resolved at t = 0.5: K/N −0.43 %, ⟨|v|⁴⟩ −1.0 %.
- **Comparison:** fast − dense is about 6 orders of magnitude below the resolved 2dt bias and 4–6 orders below the
  resolution of the dt test. No claim is made relative to the unresolved dt bias, and no convergence order is
  claimed.

**Layer 3 — long-time physical trends** (Sections 5–6; `toy_dynamics_results/production_t60/ab_*.png`)
- The A/B figures of VACF, VCCF, C_L/C_T, RDF and distinct van Hove are kept for display:
  - law B damps much more strongly (LJ: D 0.010 vs 0.058);
  - static structure agrees between A and B within error.
- These are long-time trends of the toy model, not validated against MD, and the 5-seed error bars are optimistic
  given slow collective fluctuations.

**Layer 4 — unresolved: the law-A diffusion difference** (Section 8; `production_t60/paired/PAIRED_ANALYSIS.md`)
- **double_well_A:** seed-paired fast − dense D is +5.6 % (MSD) to +8.0 % (Green-Kubo) at max lag 3, with all 5 seeds
  positive.
- **Not a constant offset:** the difference depends on the integration limit and changes sign between time windows.
- **lj_A** shows −4 %.
- **Not explained:** this is neither explained nor established as an algorithmic effect. Layers 1–2 show no operator
  or short-time discrepancy at the 1e-7 / 1e-5 level, but they do not cover t ≫ 6, where the coupled paths have
  decorrelated.
- The targeted long rerun with a null-control arm (about 30–36 pairs for ±3 %) has been postponed and not started.

## 1. What was integrated (and what was not changed)

| piece | file | status |
|---|---|---|
| v2 dense reference kit, results, restart checkpoints | `toy_models/` | copied unchanged except two corrected statements in the notes (see `toy_models/INTEGRATION_NOTES.md`, top) |
| radial laws A and B with toy amplitude | `radial_kernels.py` | new; feeds the unchanged `Mesh`, `GammaH`, Lanczos code |
| static Γ·v comparison | `test_radial_kernel_static.py` → `radial_kernel_static_results/` | new |
| conservative toy dynamics + project O-step | `toy_dynamics.py` | new |
| fast vs reference dynamics and statistics | `test_toy_dynamics.py` → `toy_dynamics_results/` | new; raw trajectories in `toy_dynamics_raw/` (git-ignored) |
| unit checks | `test_radial_kernels.py` | new (6 tests) |

No existing project file was modified except one added line in `.gitignore` (`toy_dynamics_raw/`).

### Kernel conventions, unified

- Toy laws (as in v2): g_A(r) = γ (r_ref/r) e^{−κ(r−r_ref)} and g_B(r) = γ (r/r_ref) e^{−κ(r−r_ref)}, both with
  g(r_ref) = γ. This fixes one reference strength and nothing else; A and B still have different total friction.
- Project kernel: the unit-amplitude law A, e^{−κr}/r P. So Γ_toy,A = c_A Γ_project with c_A = γ r_ref e^{κ r_ref}.
  For law B, c_B = γ e^{κ r_ref}/r_ref.
- Toy parameters: γ = 0.5, κ = 0.7, r_ref = 1.3, L = 5.5, N = 64, kT = 0.7, m = 1, dt = 0.005.
- **Law B in the Ewald machinery.** K = c r rᵀ ∫₀^∞ μ(s) e^{−sr²} ds, with μ_A = ρ_κ (the project's density) and
  μ_B = w_κ = e^{−κ²/4s}/√(πs) = ρ_κ′. Law B needed no new splitting theory, because ∫w e^{−sr²} ds = e^{−κr}/r.
  - Both densities are positive, so θ_S and θ_L are nonnegative for both laws.
  - Closed form: θ_S^B e^{ξ²r²} = (r/2) e^{−z²}[erfcx(ξr+z) + erfcx(ξr−z)]. This is checked against the s-integral
    to 7e-16.
  - Â, B̂ use the project's substitution with μ_B in place of ρ_κ.
  - Â_L(0) + Â_S(0) = ĝ(0)/3 holds to ≤3e-15 for both laws: ĝ_A(0) = 4πc_A/κ² and ĝ_B(0) = 24πc_B/κ⁴.
- **Periodic convention.** The project PPPM part keeps every |k| ≤ k_c, *including k = 0*
  (`lattice_modes`, `khat_AB`). The real-space part sums all images within r_c. This is the full periodic-image
  convention, the same as v2's `LatticeFriction`, not v2's minimum-image default.
- **Temperature.** The project's `test_true_dynamics.f_noise` hard-codes β = 1. The toy runs at kT = 0.7, so
  `toy_dynamics.f_noise(dt, kT, m)` passes kT explicitly. A static check caught this: the noise came out
  1/√0.7 − 1 = 19.5 % off before the fix.

## 2. Static comparison on fixed configurations (`radial_kernel_static_results/`)

Six configurations, N = 64, L = 5.5: the final frames of the v2 full-lattice runs for LJ and double well
(effective burn-in 140 plus production 60) for laws A and B, and one uniform random configuration per law.

**References.** There are two independent dense full-periodic references:
- v2 `LatticeFriction`: real-space image sum with a Chebyshev far field and no Ewald split;
- the project-style direct image sum (`periodic_kernel`, r ≤ 50 for A and 65 for B).

They agree to ≤ 9e-12 (max entry, relative).

| quantity (production PPPM ξ=0.7, s=4.10, η=0.7, p=7, M=15) | law A | law B |
|---|---|---|
| ‖Γ_h − Γ_ref‖₂/‖Γ_ref‖₂ | 7.1e-8 – 1.3e-7 | 3.9e-8 – 4.6e-8 |
| max relative action error (50 random vectors) | ≤ 8.8e-8 | ≤ 2.3e-8 |
| same, tight set ξ=0.7, s=4.628, η=0.6, p=8 (M=20) | ≤ 1.8e-9 | ≤ 6.1e-10 |
| literal `GammaH.matvec` vs dense assembly | ≤ 4.6e-16 | ≤ 4.5e-16 |
| symmetry ‖Γ_h − Γ_hᵀ‖/‖Γ_h‖ | ≤ 3.5e-17 | ≤ 1.5e-17 |
| translation null space ‖Γ_h 1_c‖/‖Γ_h‖ (dense; literal action gives 0) | ≤ 2.3e-16 | ≤ 2.5e-16 |
| restricted smallest eigenvalue λ_* (Γ_h = Γ_ref to 4 digits) | 1.73 – 2.51 | 30.0 – 34.3 |
| PSD certified by Weyl (‖ΔΓ‖ < λ_*) | yes (all) | yes (all) |
| frozen-q FDT, R C Rᵀ + S Sᵀ = C with R, S from one Γ_h | ≤ 3.0e-15 | ≤ 3.8e-15 |
| S_dt(Γ_h) vs S_dt(Γ_ref) | ≤ 7.2e-8 | ≤ 2.2e-8 |
| Lanczos noise rank 40 / damping rank 16 vs dense f(Γ_h)z | ≤ 3.1e-15 / 2.5e-15 | ≤ 3.2e-15 / 2.5e-15 |
| dt·λ_max | 0.04 – 0.07 | 0.23 – 0.26 |

**Error budget and ξ dependence** (fig1, s = 4.10, η = 0.7, p = 7 fixed, ξ ∈ {0.5, 0.6, 0.7, 0.85, 1.0}).
Each source was checked separately:

- **Split implementation.** Γ_S(ξ) + Γ_L(ξ), each summed exactly, equals Γ_ref to ≤ 1.1e-15 at every ξ. The split
  itself introduces no ξ dependence.
- **Real-space cutoff ε_S and Fourier cutoff ε_F.** The parameter s fixes ξr_c and k_c/2ξ.
  - For law A both are nearly flat in ξ: ε_S is 2.8e-8 to 5.2e-8 and ε_F is 1.2e-8 to 2.9e-8.
  - For law B both fall about 7× from ξ = 0.5 to 1.0: ε_S goes from 2.9e-8 to 3.4e-9 and ε_F from 2.1e-8 to 2.6e-9.
    This is the law-dependent prefactor of the tails relative to ‖Γ‖.
- **Mesh.** The alias part is ≤ 1.1e-11. The *transfer* part dominates and is non-monotone in ξ: the mesh size M
  changes with ξ through k_c and the FFT-friendly rounding. For law A it alone produces the bump of the total error
  at ξ = 0.7–0.85 (up to 1.5e-7). For law B it adds a bump at ξ = 0.7 on top of the falling cutoff terms.
- **k = 0.** The k = 0 term is present at every ξ. Its long-range coefficient Â_L(0)/V itself depends on ξ:
  0.034–0.063 per pair for A and 0.55–0.60 for B. The full k = 0 coefficient of the unsplit periodic kernel is
  ξ-independent: 0.083 for A and 0.601 for B, as in v2.
  - Dropping only the reciprocal k = 0 term would therefore produce a ξ-dependent model.
  - Removing the full k = 0 coefficient from Γ_h makes it indefinite (λ_min −0.7 to −1.5 for A, −3.2 to −7.4 for B).
  - This is a diagnostic of what *would* happen. Nothing in the project drops k = 0, and the ξ dependence measured
    above is a mesh-transfer effect, not a k = 0 effect.

**Conclusion of the static stage.** With the production PPPM set, the fast operator and the dense full-periodic
reference describe the same physical model to ~1e-7 (A) and ~5e-8 (B). The friction and the noise use one Γ_h
instance. Symmetry, the total-momentum null space and positive semidefiniteness hold for both laws.

## 3. Short coupled dynamics (`toy_dynamics_results/coupled_short/`)

**Setup.**
- Cases: LJ and double well × laws A and B, seeds 101–103.
- Start states: the final frames of the v2 full-lattice runs at t = 200 under the reference model.
- Length: 2000 steps (t = 10), stride 4.
- Methods (same start, same Gaussian stream ξ_n):
  - `reference`: v2 lattice sum, dense;
  - `pppm_dense`: Γ_h, dense;
  - `pppm_lanczos`: Γ_h with Lanczos noise rank 40 / damping rank 16.
- Integrator: B A O A B with the toy pair forces (switch-derivative term included). R and S come from one Γ instance
  per step, at kT = 0.7.

**Conservation and thermostat.**
- max ‖P(t) − P(0)‖ ≤ 9e-13 for every method.
- T and U agree between methods within their seed scatter.
- Spectrum seen during the runs:
  - law A: λ ∈ [1.5, 13.2], dt·λ_max ≤ 0.07;
  - law B: λ ∈ [29, 51], dt·λ_max ≤ 0.26.
- Cost per step at N = 64 (one core): reference 18 ms, PPPM dense 44 ms, PPPM + Lanczos 95–100 ms (56 Γ_h actions).

**Pathwise divergence** (`fig_pathwise_divergence.png`):

| case | PPPM − reference, max_i Δv at t = 0.02 / 2 / 6 / 10 | Lanczos − PPPM dense at the same times |
|---|---|---|
| LJ A | 7.6e-8 / 2.7e-6 / 8.9e-4 / 1.1 | 1.6e-14 / 2.0e-12 / 5.4e-10 / 3.1e-7 |
| LJ B | 2.3e-8 / 3.4e-8 / 4.3e-8 / 9.4e-8 | 1.2e-14 / 1.7e-14 / 2.2e-14 / 1.1e-13 |
| double well A | 5.8e-8 / 2.2e-4 / 2.2 / 2.7 | 1.2e-14 / 3.5e-11 / 3.6e-5 / 1.7 |
| double well B | 3.5e-8 / 3.8e-8 / 2.4e-7 / 1.9e-6 | 1.0e-14 / 2.6e-14 / 3.1e-13 / 6.2e-12 |

- **Law A.** The initial offset equals the operator difference, and the Krylov difference starts at round-off.
  Both then grow at the same exponential rate, which is the chaotic amplification of the conservative dynamics.
  The coupled paths become effectively independent after t ≈ 6 (double well) and t ≈ 10 (LJ). This growth is a
  property of the dynamics, not a drift of either algorithm.
- **Law B.** The friction is strong (λ_min ≈ 30), so there is no exponential growth and the coupled paths stay
  within 2e-6 over the window.

**Statistics on these short runs** (3 seeds, t = 10; paired difference vs the reference's own sampling SEM):
- **Law B, both potentials.** VACF, VCCF, C_L, C_T and the RDF agree to ≤ 6e-9. The distinct van Hove agrees to
  ≤ 3e-4, i.e. a single pair crossing a bin edge. Every point is within 0.01 reference-SEM.
- **LJ A.** Every point is within 2 reference-SEM except a few van Hove bins (max 4.3).
- **Double well A.** 83–97 % of points are within 2 reference-SEM, with outliers up to ~25 on C_L. These runs do
  not resolve the law-A statistics. There are 3 seeds, the decorrelated parts of the pairs are finite independent
  samples, and the 3-seed SEM is itself very noisy. This design tests pathwise agreement up to the Lyapunov horizon,
  not statistical agreement. That is why long runs follow (Section 4).

## 4. Which long runs are needed (decision after Sections 2–3)

- **v2 minimum-image runs.** These are a different friction model; the fast operator is full-periodic. They are
  not affected and are not rerun.
- **v2 full-lattice runs** (5 seeds × t = 60 after effective burn-in 140). This is the same model as the fast
  operator to ≤ 1.3e-7, so they remain valid reference statistics and are not rerun.
- **New runs.** The fast production method (PPPM + Lanczos 40/16) and a fresh dense reference, coupled pairwise:
  - 4 cases × seeds 101–105, t = 60 each (12000 steps), started from the same v2 t = 200 states;
  - tag `production_t60`, about 2 h wall on 4 cores;
  - Law A needs these to compare statistics after decorrelation. Law B gets the same protocol so that the fast
    algorithm's figures cover both laws.

## 5. Long coupled runs: fast vs reference statistics (`toy_dynamics_results/production_t60/`)

**Setup.**
- Cases: LJ and double well × laws A and B, seeds 101–105.
- Start states: the v2 t = 200 states.
- Length: 12000 steps (t = 60), stride 4 (sample interval 0.02).
- Methods: `pppm_lanczos` (production 40/16) and `reference`, on the same noise stream (seed + 3000).
- Statistics: every lag up to 3.0, 300 time origins per run, VCCF bins as in v2.
- The independent v2 full-lattice ensembles (5 seeds, window t ∈ [140, 200]) are a second, uncoupled reference.

**Conservation and spectrum.**
- max ‖P(t) − P(0)‖ ≤ 1.1e-12 in all 40 runs.
- Spectrum: law A λ ∈ [1.47, 13.2], law B λ ∈ [28.7, 51.9].
- Wall time per step: reference 18.3–18.7 ms, PPPM + Lanczos 93–98 ms (N = 64, one core). At N = 64 the dense
  reference is cheaper. No performance claim is made here; the project's scaling study is in
  `dynamics_results/REPORT.md`.

**When the coupled pairs decorrelate** (max_i |Δv_i| reaches O(1)):

| case | t = 0.02 | t = 10 | t = 40 | t = 60 |
|---|---|---|---|---|
| LJ A | 7.6e-8 | 1.1 | 3.0 | 2.9 |
| LJ B | 2.6e-8 | 2.1e-7 | 1.6e-4 | 1.4e-2 |
| double well A | 5.8e-8 | 2.7 | 3.0 | 2.9 |
| double well B | 3.5e-8 | 2.7e-6 | 1.1 | 0.8 |

- LJ B stays pathwise coupled for essentially the whole window. Its fast and reference statistics agree to
  ≤ 1e-3 (van Hove) and ≤ 2e-5 (time correlations), at most 0.4 reference-SEM.
- Law A decorrelates after t ≈ 6–10, and double well B after t ≈ 30–40. For these cases the comparison is
  statistical.

**Curves, combined-SEM z between fast and coupled reference** (conservative while the pairs are still correlated).
*Superseded for the scalars by the seed-paired analysis in Section 8.* The combined-SEM z treats fast and reference
as independent groups and so ignores the pairing:

| case | VACF | VCCF (all bins) | C_L | C_T | RDF | distinct van Hove (all t) |
|---|---|---|---|---|---|---|
| LJ A | 95 % (max 2.8) | 90 % (5.3) | 85 % (3.6) | 97 % (3.1) | 88 % (3.6) | 95 % (7.9) |
| LJ B | 100 % (0.0) | 100 % (0.0) | 100 % (0.0) | 100 % (0.0) | 100 % (0.1) | 100 % (0.3) |
| double well A | 97 % (3.1) | 96 % (3.1) | 95 % (2.9) | 98 % (2.5) | 100 % (1.7) | 98 % (5.1) |
| double well B | 100 % (1.5) | 100 % (3.5) | 100 % (0.8) | 100 % (0.9) | 98 % (2.7) | 100 % (3.6) |

Entries are the fraction of lags or radii with |z| ≤ 2, with max |z| in brackets.

**Same-model control.** The dense reference against the independent v2 ensemble (`report.json` → `independent_vs_v2`)
gives a comparable spread: 80–100 % within |z| ≤ 2 and max |z| 1.4–7.8. The fast-vs-v2 numbers are almost the same
(`z_*_fast_vs_v2.png`: the solid fast curves and the dotted control curves make the same excursions at the same lags).
- The excursions are consistent with finite, slowly fluctuating samples (window difference, correlated lags,
  5 seeds) that both algorithms share.
  - *Correction:* this comparison alone does not show that the fast operator leaves the dynamics unchanged.
  - The seed-paired analysis (Section 8) finds a double-well A diffusion difference that this comparison did not
    resolve.
- The lag-by-lag fractions are descriptive, not a formal test.

**Scalars** (mean ± SEM over 5 seeds; z uses the combined SEM):

| case | quantity | fast PPPM+Lanczos | coupled dense reference | v2 lattice (t in [140,200]) | z fast-ref | z fast-v2 | z ref-v2 (control) |
|---|---|---|---|---|---|---|---|
| lj_A | T | 0.7002 ± 0.0019 | 0.6999 ± 0.0004 | 0.6975 ± 0.0014 | 0.2 | 1.1 | 1.6 |
| lj_A | U/N | -3.204 ± 0.045 | -3.170 ± 0.030 | -3.203 ± 0.029 | -0.6 | -0.0 | 0.8 |
| lj_A | D (Green-Kubo) | 0.0583 ± 0.0014 | 0.0609 ± 0.0018 | 0.0584 ± 0.0015 | -1.2 | -0.1 | 1.1 |
| lj_A | D (MSD) | 0.0580 ± 0.0018 | 0.0603 ± 0.0018 | 0.0588 ± 0.0017 | -0.9 | -0.3 | 0.6 |
| lj_A | tau_T | 0.223 ± 0.009 | 0.241 ± 0.010 | 0.215 ± 0.011 | -1.3 | 0.6 | 1.8 |
| lj_A | RDF peak | 3.309 ± 0.064 | 3.269 ± 0.046 | 3.318 ± 0.063 | 0.5 | -0.1 | -0.6 |
| lj_A | dU/N halves | 0.0099 ± 0.0414 | 0.0534 ± 0.0551 | -0.0413 ± 0.0223 | -0.6 | 1.1 | 1.6 |
| lj_B | T | 0.6996 ± 0.0010 | 0.6996 ± 0.0010 | 0.6987 ± 0.0008 | -0.0 | 0.7 | 0.7 |
| lj_B | U/N | -3.145 ± 0.010 | -3.145 ± 0.010 | -3.196 ± 0.018 | -0.0 | 2.5 | 2.5 |
| lj_B | D (Green-Kubo) | 0.0100 ± 0.0003 | 0.0100 ± 0.0003 | 0.0095 ± 0.0006 | -0.0 | 0.9 | 0.9 |
| lj_B | D (MSD) | 0.0093 ± 0.0001 | 0.0093 ± 0.0001 | 0.0090 ± 0.0001 | 0.0 | 1.8 | 1.8 |
| lj_B | tau_T | 0.0175 ± 0.0036 | 0.0175 ± 0.0036 | 0.0214 ± 0.0029 | 0.0 | -0.8 | -0.8 |
| lj_B | RDF peak | 3.252 ± 0.019 | 3.253 ± 0.019 | 3.334 ± 0.038 | -0.0 | -1.9 | -1.9 |
| lj_B | dU/N halves | 0.0302 ± 0.0590 | 0.0302 ± 0.0590 | 0.0355 ± 0.0324 | -0.0 | -0.1 | -0.1 |
| double_well_A | T | 0.7007 ± 0.0007 | 0.6993 ± 0.0018 | 0.6960 ± 0.0011 | 0.8 | 3.8 | 1.6 |
| double_well_A | U/N | -5.766 ± 0.040 | -5.756 ± 0.033 | -5.768 ± 0.073 | -0.2 | 0.0 | 0.1 |
| double_well_A | D (Green-Kubo) | 0.0209 ± 0.0006 | 0.0194 ± 0.0006 | 0.0202 ± 0.0015 | 1.7 | 0.4 | -0.5 |
| double_well_A | D (MSD) | 0.0212 ± 0.0005 | 0.0201 ± 0.0004 | 0.0211 ± 0.0010 | 1.8 | 0.1 | -0.9 |
| double_well_A | tau_T | 0.125 ± 0.017 | 0.105 ± 0.007 | 0.138 ± 0.014 | 1.1 | -0.6 | -2.2 |
| double_well_A | RDF peak | 4.243 ± 0.081 | 4.272 ± 0.074 | 4.242 ± 0.122 | -0.3 | 0.0 | 0.2 |
| double_well_A | dU/N halves | -0.0976 ± 0.0518 | 0.0409 ± 0.0419 | 0.0187 ± 0.0434 | -2.1 | -1.7 | 0.4 |
| double_well_B | T | 0.6991 ± 0.0009 | 0.6992 ± 0.0010 | 0.6984 ± 0.0009 | -0.1 | 0.5 | 0.5 |
| double_well_B | U/N | -5.705 ± 0.056 | -5.703 ± 0.051 | -5.682 ± 0.087 | -0.0 | -0.2 | -0.2 |
| double_well_B | D (Green-Kubo) | 0.0063 ± 0.0005 | 0.0064 ± 0.0004 | 0.0058 ± 0.0003 | -0.2 | 0.9 | 1.3 |
| double_well_B | D (MSD) | 0.0051 ± 0.0003 | 0.0051 ± 0.0002 | 0.0051 ± 0.0003 | -0.0 | 0.1 | 0.2 |
| double_well_B | tau_T | 0.0224 ± 0.0042 | 0.0206 ± 0.0035 | 0.0170 ± 0.0025 | 0.3 | 1.1 | 0.8 |
| double_well_B | RDF peak | 4.167 ± 0.108 | 4.195 ± 0.096 | 4.076 ± 0.150 | -0.2 | 0.5 | 0.7 |
| double_well_B | dU/N halves | -0.0264 ± 0.0682 | -0.0226 ± 0.0538 | -0.0035 ± 0.0370 | -0.0 | -0.3 | -0.3 |

- Fast vs coupled reference, combined-SEM z: every |z| ≤ 1.8, except the double-well A half-to-half energy drift
  statistic (z = −2.1).
  - *Correction:* this z ignores the pairing. Seed-paired, the double-well A diffusion difference is +8.0% (GK) and
    +5.6% (MSD), with all 5 seeds positive and paired t = 5.6 and 5.2 (df 4). See Section 8.
- The largest fast-vs-v2 gap is the double-well A temperature (z = 3.8). It comes from the v2 window's own low
  temperature, 0.6960 ± 0.0011, which v2 already reported. The fast run gives 0.7007 ± 0.0007.
- LJ B U/N differs between the new window and the v2 window by z = 2.5. It does so identically for fast and
  reference, because they are pathwise identical: a window effect, not an algorithm effect.

**Physics of the fast method** (`ab_{lj,double_well}_pppm_lanczos.png`, v2 layout and colours):
- Kernel A vs B: B damps far more strongly.
  - LJ: D = 0.058 (A) vs 0.010 (B); τ_T = 0.22 vs 0.018.
  - Double well: D = 0.021 vs 0.0063.
- First-shell VCCF peak: 0.038 (LJ) and 0.035 (double well) at t ≈ 0.15 for A; 0.006 at t ≈ 0.08 for B.
- First zero crossing:
  - law B: VACF at t = 0.08–0.10, C_L at 0.08–0.10, C_T at 0.12;
  - law A: VACF at 0.54 (LJ) and 0.14 (double well), C_L at 0.34 and 0.16, C_T at 1.02 and 0.54.
- Static structure (RDF, van Hove at t = 0) is the same for A and B within error, as it must be, since the friction
  does not change the Boltzmann distribution.

## 6. Limitations (kept from v2, not weakened)

- **No equilibrium proof.** The start states have an effective history of t = 200 under the reference model.
  "No detected drift" and "agreement within ~2σ" are consistency statements, not proofs of equilibrium.
  - Window-to-window shifts larger than the within-window SEM remain. Examples: LJ B U/N, z = 2.5 between windows;
    the double-well A temperature of the v2 window.
  - At ρ = 0.385, T = 0.7 the LJ system is likely in liquid–vapour coexistence, and the double well forms an ordered
    condensed cluster. Slow collective fluctuations make 5-seed SEMs optimistic for small differences.
- **Law A comparison is statistical.** Chaotic decorrelation after t ≈ 6–10 makes fast-vs-reference agreement for
  law A a statement about distributions, at the resolution of 5 seeds × t = 60. Pathwise agreement holds only up to
  the Lyapunov horizon. For law B the agreement is mostly pathwise.
- **Splitting-parameter dependence is attributed component by component** (Section 2): real cutoff, Fourier cutoff,
  mesh alias, mesh transfer and split implementation. It is not attributed to k = 0, which is retained.
- **PPPM parameters were not re-optimised.** The production and tight sets were tuned by the project at κ = 1. Here
  their accuracy was measured at κ = 0.7 for both laws on six configurations (≤ 1.3e-7 and ≤ 1.8e-9), but not
  re-optimised.
- **Single box.** One box (N = 64, L = 5.5); no finite-size study. Law-B friction is dominated by the uniform
  k = 0 part.
- **No external validation.** There is no MD or experimental data and no PRL reproduction; toy potentials and
  parameters only.

## 7. Files, data, commands

- Report: this file. The static stage is in `radial_kernel_static_results/` (`summary.json`, `fig1`, `fig2`).
- Short coupled runs: `toy_dynamics_results/coupled_short/`.
- Long runs: `toy_dynamics_results/production_t60/`:
  - `report.json`: scalars, z tables, paired and independent comparisons;
  - `comparison.json`: pathwise divergence, paired differences;
  - `ensemble_<case>_<method>.npz`: mean and SEM curves;
  - `observables/<case>/<method>_seed<s>.npz`: per-seed observables;
  - figures `ab_*`, `fig_*`, `z_*`.
- Raw trajectories: 76 files, 414 MB, git-ignored. They contain Q (unwrapped), V (velocity), t, box, diagnostics,
  metadata, and the final q, p and RNG state for an exact restart. SHA-256 per file is in
  `toy_dynamics_results/raw_trajectory_manifest.json`.
- v2 raw trajectories: 100 files, 845 MB. Their manifest is
  `toy_models/v2_results/restart_checkpoints/trajectory_manifest.json`, and the small restart checkpoints are in git.
- Commands: see the README section "Toy application". The long runs used `--stage run --tag production_t60
  --steps 12000 --seeds 101 102 103 104 105 --methods reference pppm_lanczos --workers 4`, followed by `--stage
  report` and `--stage analyze` with `--origins 300`.

## 8. Seed-paired re-analysis of production_t60 (`toy_dynamics_results/production_t60/paired/`)

`analyze_toy_paired.py`; full write-up in `paired/PAIRED_ANALYSIS.md`; tables in `paired_summary.csv`,
`lag_scan.csv`, `window_summary.csv` and `operator_along_trajectories.csv`.

**Method.**
- d_i = fast_i − reference_i per seed (same start, same noise); SEM = std(d_i, ddof = 1)/√5; Student-t 95% CI with
  df = 4.
- The two methods are not treated as independent even after the paths separate.
- The cache-only stages (full-run table, lag scan) run without raw files; the window and operator stages are skipped
  explicitly when raw files are missing.

**Results.**
- **double_well_A** at max lag 3: d/D = +8.0% (GK, CI +4.1 to +12.0%) and +5.6% (MSD, CI +2.6 to +8.5%). All 5 seeds
  are positive; paired t = 5.6 / 5.2.
  - The difference grows with the integration limit (+1–3% at lag 0.5).
  - It changes sign between time windows: about +15% for t ∈ [0, 45] and −11 to −16% for t ∈ [45, 60]. T and U/N
    move with it.
- **lj_A:** −4% (GK CI −8.5 to −0.2%), the opposite sign.
- **double_well_B:** ±2% with CIs up to ±9%.
- **lj_B:** ≤ 1e-7.
- **Operator on 130 visited configurations per case:** ‖Γ_h − Γ_ref‖/‖Γ_ref‖ ≤ 1.5e-7.

**Reading.**
- The data do not establish a systematic effect of the fast algorithm. Against one: the window and lag dependence,
  the sign reversals, the 1e-7 operator agreement, and multiplicity.
- The data also do not establish agreement in law-A diffusion to better than several percent.

**Recommendation.** A targeted rerun for law A (double_well_A first) with a null-control arm:
- double_well_A needs about 30–36 pairs for ±3% and 61–77 for ±2%; lj_A needs about 12–13 for ±3%.
- The design and cost are in `PAIRED_ANALYSIS.md`. It has not been started.

## 9. Fixed-time paired validation (`toy_fixed_time_results/`)

Script: `test_toy_fixed_time.py`. The report is `toy_fixed_time_results/REPORT.md`.

Outputs:
- `paths.csv`: every path × method × step × time;
- `fixed_time_per_state.csv` and `fixed_time_across_states.csv`;
- `fixed_time_verdict.csv`;
- `fixed_time_differences.png`;
- `selftest.json`, `run_meta.json`, `timing.json`;
- `paths/*.npz`: per-path observables, seeds, and the q and p at t = 0.5, 1, 2.

Summary in the Layer-2 paragraph above.
