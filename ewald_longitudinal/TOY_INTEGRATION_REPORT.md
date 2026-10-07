# Toy application on the Ewald/PPPM longitudinal-friction project

Double-well and Lennard-Jones pair potentials, radial friction laws A and B, the project's PPPM
operator Γ_h and finite-time FDT thermostat, compared with dense full-periodic references.
This is a **toy application**. It is not a reproduction of Lyu & Lei, PRL 131, 177301 (2023), and no figure
here validates real MD. The friction is pure longitudinal, K = g(r) r̂r̂ᵀ, with no transverse kernel.

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
