# Toy-model application: status, changes, conventions, results

> **Correction (later session).** This v2 note was written without access to the project. The
> Ewald/PPPM project exists on branch `claude/wizardly-shannon-n9enyq` (`ewald_longitudinal/`, checked at
> commit `6317ca5`); the statement below that it was not available was wrong (only the default branch
> had been inspected). The project's PPPM operator keeps k = 0 and uses the full periodic-image convention.
> The integration with the project (kernel B, toy amplitude, conservative potentials, fast vs reference
> comparisons) is documented in `../TOY_INTEGRATION_REPORT.md`. Statements here about the splitting
> parameter were also too strong and have been corrected in place (marked "corrected").

## 1. Status

The target long-range longitudinal friction / Ewald-PPPM project was NOT available in
this session (the only attached repository was an unrelated finite-element course
assignment). Therefore nothing here is wired into the project's integrator, Gamma
action, Ewald parameters, OU/Lanczos noise or output interface, and its periodic
convention could not be read from its code. Everything below is done on the
standalone dense reference kit, and is designed so the remaining project steps are
mechanical once the code is available (Section 6).

Toy models only. No figure or number here validates real MD or the PRL system
(Lyu & Lei, PRL 131, 177301 (2023)); no external physical reference data were used.
Friction remains longitudinal only, K_ij = g(r) rhat rhat^T; no transverse kernel.

## 2. Modification list

`prl_toy_models.py` (all additions; default behavior and plotting colors unchanged)
- `assemble_friction(i, j, K, n)`: graph-Laplacian assembly factored out of
  `friction_matrix` (identical result).
- `friction_tail_integral`: closed form of 4 pi int_rc^inf r^2 g(r) dr for A and B;
  rc=0 gives ghat(0), the k=0 transform.
- `lattice_shifts`, `image_sum_tensors`: vectorized brute-force periodic-image sums.
- `LatticeFriction`: full periodic-image friction
  K_ij = sum_n g(|r_ij + n L|) rhat rhat^T, converged to `rel_tol` (default 1e-10 of
  ghat(0)); exact direct sum over images with |nL| <= 2 min(L), smooth far-image
  remainder as a degree-16 tensor Chebyshev fit (agrees with brute force to ~1e-13).
  Nothing is removed in Fourier space; the k=0 coefficient is retained.
- `simulate(..., friction_images='minimum'|'lattice', lattice_tol=1e-10)` and CLI
  flags `--friction-images`, `--lattice-tol`. The same Gamma is used for the
  dissipative and noise factors of the exact frozen-q OU step, as before.
- `simulate(..., initial_state=(q, p))` restarts from a saved state (e.g. the last
  frame, p = m V); the default initial condition is unchanged.
- Trajectory metadata now records `V_kind` (velocity p/m, not momentum),
  `sample_interval`, `burn_in_time`, `production_time`, `t_origin`, and the friction
  convention record (truncation radius, tail estimate, image counts, ghat(0), k=0
  coefficient).

New files
- `test_periodic_friction.py`: 6 tests (Chebyshev vs brute force, cubic and
  orthorhombic boxes; truncation convergence; symmetry, translation null space,
  positivity and frozen-q FDT of the lattice Gamma; lattice == minimum image in the
  short-range limit; cell average of the periodic tensor == k=0 coefficient
  ghat(0)/(3V) I, i.e. the zero mode is retained; short lattice trajectory records
  its convention and conserves momentum).
- `ensemble.py`: independent-seed ensembles, all four statistics, per-seed scalars
  (T, U, half-vs-half drift, Green-Kubo and MSD diffusion, VACF minimum, transverse
  relaxation time, RDF peak), mean +/- SEM across seeds, A/B figures per potential
  and convention, and convention-comparison figures. `--continue-from DIR` restarts
  every seed from its last frame with a fresh noise stream and records the
  accumulated (effective) burn-in.
- `compare_action.py`: convention detector for a fast Gamma-action code (minimum
  image vs full lattice vs lattice with the k=0 coefficient removed).

## 3. Periodic conventions: what differs and by how much

The pair potential is short ranged (r_cut = 2.5 < L/2 = 2.75), so minimum image and
full periodic images are identical for the conservative force. They are NOT identical
for the friction, because neither kernel is short ranged relative to L = 5.5:
g_B(L/2)/g_B(r_ref) = 0.77, and ghat(0) = 41.4 (A) and 300.1 (B).

k=0 coefficient of the periodic pair tensor, c0 = ghat(0)/(3V) per pair:
A: 0.0830, B: 0.601 (compare g(r_ref) = gamma = 0.5). Its contribution to Gamma
acting on peculiar velocities is N c0 = 5.31 (A) and 38.5 (B).

Measured on pilot configurations (N = 64, L = 5.5, defaults):

| case | rel. Frobenius gap ‖Γ_lat−Γ_min‖/‖Γ_lat‖ | mean Γ_ii (lattice / minimum) | eig range, lattice | eig range, minimum | eig range, lattice with k=0 removed |
|---|---|---|---|---|---|
| double_well_A | 0.246 | 6.31 / 4.71 | [1.85, 11.4] | [0.40, 9.6] | [−2.92, 6.1] |
| double_well_B | 0.773 | 37.98 / 8.51 | [30.9, 48.0] | [2.41, 17.0] | [−6.67, 9.5] |
| lj_A | 0.308 | 5.28 / 3.62 | [2.31, 8.3] | [0.87, 6.5] | [−2.91, 3.0] |
| lj_B | 0.781 | 37.95 / 8.26 | [32.6, 46.0] | [3.33, 14.5] | [−5.46, 7.5] |

(eigenvalues in the zero-total-momentum subspace, per component, m = 1)

Consequences
- Minimum-image and full-lattice friction are two different models at this box
  size. Their difference must not be reported as Ewald/PPPM error.
- With full periodic images, kernel B is dominated by the uniform k=0 part: every
  peculiar velocity is damped at rate ~38, so B becomes strongly overdamped.
  N c0 = rho ghat(0)/3 is intensive, so this is the large-L limit of B, while the
  minimum-image B (truncated at the cell boundary) is strongly box-size dependent.
- Removing the k=0 coefficient makes Gamma indefinite, so the matched noise
  sqrt(2 kBT Gamma) does not exist and the FDT fails. In an Ewald split, dropping
  only the reciprocal k=0 term would remove (1/V) Khat_L(0), which depends on the
  splitting parameter. (corrected) A dependence of results on the splitting parameter
  is NOT by itself evidence of a k=0 problem: real-space truncation, Fourier
  truncation, mesh (alias/transfer) errors and the split implementation all vary
  with it and must be checked separately (see ../radial_kernel_static_results).
- Self-image terms (i = j, n != 0) act on v_i − v_i and cancel in Gamma. An Ewald code
  that adds a self term on the diagonal must subtract it identically off-diagonal.

## 4. Commands

```
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_physics.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_periodic_friction.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python ensemble.py --images minimum lattice \
    --seeds 101 102 103 104 105 --burn 8000 --steps 12000 --max-lag-time 3.0 \
    --origins 300 --workers 4 --out ensemble_results
# Extend burn-in by continuing every seed (used for the final results, effective burn-in 140):
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python ensemble.py --potentials double_well \
    --images minimum lattice --seeds 101 102 103 104 105 --burn 8000 --steps 12000 \
    --max-lag-time 3.0 --origins 300 --continue-from ensemble_results \
    --out ensemble_results_dw_continued        # same with --potentials lj -> lj_continued
# Time-step check (double well, minimum image):
python ensemble.py --potentials double_well --images minimum --dt 0.0025 --stride 8 \
    --burn 0 --steps 16000 --max-lag-time 1.0 --origins 100 \
    --continue-from ensemble_results_dw_continued --continue-seed-offset 2000 --out dtcheck_dt0025
# Single cases, original interface (now with a convention flag):
python prl_toy_models.py demo --kernels A B --friction-images lattice --burn 8000 --steps 12000
# Convention detector for the project's fast Gamma action:
python compare_action.py make-probe probe.npz --kernel A   # then add GammaV from the project
python compare_action.py check probe.npz
```

## 5. Results from the dense reference (independent seeds)

Parameters (all cases): N = 64, L = 5.5 (rho = 0.385), m = 1, kBT = 0.7, epsilon = sigma = 1,
dt = 0.005, gamma = 0.5, kappa = 0.7, r_ref = 1.3 (g_A(r_ref) = g_B(r_ref) = gamma),
BAOAB with the exact frozen-q OU step. Saved every 4 steps (sample interval 0.02);
V is velocity. Statistics: every lag up to 3.0 (150 frames), 300 time origins per
trajectory. VCCF bins by r_ij(t0): double well [0, 1.36, 1.90, 2.75] (barrier between
the wells, first minimum of g), LJ [0, 1.55, 2.40, 2.75] (minima of g).
Seeds 101-105; error bars = SEM across the 5 independent seeds (ddof = 1).

### Burn-in protocol (why t = 140)
1. Original pilots (t_burn = 4, one seed): U/N still falls during production in all
   four cases (e.g. double_well_A: -4.64 -> -5.29 between production halves).
2. Single-seed runs from t = 0 to 100: U/N relaxes over t ~ 20-30, then shows slow
   block-to-block fluctuations of order 0.2 (blocks of 10 time units).
3. Ensemble 1 (burn 40, production 60): double well still drifts significantly
   (half-vs-half dU/N: min A -0.11 +/- 0.04, min B -0.12 +/- 0.02, lattice A -0.15 +/- 0.06,
   lattice B -0.66 +/- 0.06). LJ shows no significant within-window drift, BUT the
   static observables disagree between friction models (lattice B: U/N = -3.01 +/- 0.04,
   RDF peak 3.06 +/- 0.06 vs minimum A: -3.25 +/- 0.01, 3.39 +/- 0.03). Static
   observables must be friction-independent at equilibrium (same Boltzmann
   distribution), so this was an equilibration failure of the slow (overdamped)
   lattice-B dynamics, not a physical effect.
4. All cases were continued from their final states with fresh noise streams
   (seed + 1000) for another burn of 40 and production of 60: effective burn-in 140.
5. Time-step check (double well, minimum image, 5 seeds, continuation of t = 40):
   dt = 0.005 vs 0.0025 gives T 0.6999 +/- 0.0030 vs 0.7007 +/- 0.0013 (A),
   0.6992 +/- 0.0017 vs 0.6988 +/- 0.0020 (B); U/N, D and RDF peak also agree within SEM.

### Final windows (effective burn-in 140, production 60, 5 seeds each)

| potential | friction images | kernel | T | U/N | dU/N (2nd-1st half) | D (Green-Kubo) | D (MSD) | VACF min | tau_T = int C_T dt | RDF peak |
|---|---|---|---|---|---|---|---|---|---|---|
| double_well | minimum | A | 0.6957 ± 0.0027 | -5.86 ± 0.02 | -0.04 ± 0.04 | 0.0251 ± 0.002 | 0.0252 ± 0.002 | -0.136 ± 0.006 | 0.252 ± 0.03 | 4.39 ± 0.06 |
| double_well | minimum | B | 0.6969 ± 0.0015 | -5.76 ± 0.05 | -0.06 ± 0.07 | 0.0172 ± 0.0009 | 0.0172 ± 0.0008 | -0.127 ± 0.004 | 0.12 ± 0.01 | 4.22 ± 0.09 |
| double_well | lattice | A | 0.6960 ± 0.0011 | -5.77 ± 0.07 | 0.02 ± 0.04 | 0.0202 ± 0.001 | 0.0211 ± 0.001 | -0.135 ± 0.006 | 0.138 ± 0.01 | 4.24 ± 0.12 |
| double_well | lattice | B | 0.6984 ± 0.0009 | -5.68 ± 0.09 | -0.00 ± 0.04 | 0.00581 ± 0.0003 | 0.00507 ± 0.0003 | -0.050 ± 0.002 | 0.017 ± 0.003 | 4.08 ± 0.15 |
| lj | minimum | A | 0.6985 ± 0.0021 | -3.18 ± 0.01 | 0.02 ± 0.07 | 0.0833 ± 0.002 | 0.084 ± 0.002 | -0.010 ± 0.001 | 0.391 ± 0.02 | 3.30 ± 0.02 |
| lj | minimum | B | 0.6999 ± 0.0020 | -3.13 ± 0.03 | 0.03 ± 0.05 | 0.0395 ± 0.0009 | 0.0395 ± 0.0008 | -0.019 ± 0.002 | 0.143 ± 0.01 | 3.22 ± 0.05 |
| lj | lattice | A | 0.6975 ± 0.0014 | -3.20 ± 0.03 | -0.04 ± 0.02 | 0.0584 ± 0.002 | 0.0588 ± 0.002 | -0.011 ± 0.001 | 0.215 ± 0.01 | 3.32 ± 0.06 |
| lj | lattice | B | 0.6987 ± 0.0008 | -3.20 ± 0.02 | 0.04 ± 0.03 | 0.00947 ± 0.0006 | 0.00901 ± 0.0001 | -0.029 ± 0.002 | 0.0214 ± 0.003 | 3.33 ± 0.04 |

Max |total momentum| <= 8e-13 in every run; min pair distance 0.69 (double well), 0.90 (LJ).

### Consistency checks that passed
- Static observables agree across kernels and conventions within ~2 SEM in the final
  windows (LJ U/N -3.13 to -3.20, RDF peak 3.22-3.33; double well U/N -5.68 to -5.86,
  RDF peak 4.08-4.39; the extreme pairs differ by ~2 SEM).
- Green-Kubo and MSD diffusion coefficients agree (within ~2 SEM; lattice double_well_B
  is the largest gap, 0.0058 vs 0.0051).
- VCCF at t = 0 is -1/(N-1) = -0.0159 in every distance bin (pair-weighted
  -0.0156 to -0.0162): the zero-total-momentum constraint. This finite-N offset
  must be accounted for when comparing VCCF across system sizes.
- No within-window drift is detectable in the final windows (|dU/N| <= 0.06, all
  within 1.5 SEM except LJ lattice A at 1.9 SEM).
- Original and modified code give bit-identical trajectories on the same machine
  (default minimum-image path). The shipped pilot differs from a rerun at round-off
  (1e-11 at the first saved frame), consistent with platform BLAS/eigh differences
  amplified by chaos.

### What the numbers say (this toy state point only)
- Kernel B vs A, minimum image: D smaller by ~2.1x (LJ 0.040 vs 0.083) and 1.5x
  (double well 0.017 vs 0.025); transverse current decays 2-3x faster.
- Full lattice vs minimum image: modest for A (LJ D 0.058 vs 0.083), large for B
  (LJ D 0.0095 vs 0.040, double well 0.0058 vs 0.017). Lattice B is overdamped: the
  VACF crosses zero at t = 0.08 and C_T at t = 0.12-0.16, and the first-shell VCCF
  peak drops to ~0.005 (from 0.03-0.05).
- First-shell VCCF is positive with a peak at t = 0.14-0.18 (t = 0.10 for lattice B):
  momentum transferred to neighbours. The peak is larger for A than for B in every
  potential/convention pair; outer shells start at -1/(N-1).

### Caveats (do not over-read)
- Window-to-window shifts are larger than within-window SEMs for some quantities
  (LJ minimum A: U/N -3.25 -> -3.18 and D 0.076 -> 0.083 between the t = 40-100 and
  t = 140-200 windows; double well U/N moved by -0.05 to -0.66). At rho = 0.385,
  T = 0.7 the LJ system is likely in liquid-vapor coexistence and the double well
  forms an ordered condensed cluster (RDF peak > 4), with slow collective
  fluctuations. The 5-seed SEMs therefore underestimate the uncertainty of small
  differences; the large A/B and convention effects (factors 1.5-8 in D) are robust.
  Equilibrium is not claimed.
- T in the final windows is 0.6957-0.6999 (up to 3.6 SEM below 0.7 for double well
  lattice A); the dt-halving check found no time-step bias at the 0.002 level, so
  this is attributed to the slow fluctuations above, not established as a bias.
- Minimum-image A has the slowest transverse decay and a tail (C_T(t=1) = 0.04 double
  well, 0.075 LJ). Part of this may be slow collective motion of the condensed
  cluster in a 64-particle box; it is not interpreted further.
- Single small box (N = 64). Finite-size effects are not studied. The minimum-image
  friction model in particular is box-size dependent for these kernels.
- No external physical reference data. This is a toy application of the PRL
  observables, not a reproduction of the PRL system.

## 6. Remaining steps that need the project code

1. Read the project's docs/AGENTS; locate the integrator, Gamma action, Ewald/PPPM
   parameters, OU/Lanczos noise and output interface.
2. Port `pair_potential`/`conservative_force` (switch-derivative term included).
   Minimum image is exact here because r_cut = 2.5 < L/2; keep the L > 5 sigma check.
3. Kernel A first, then B, with explicit gamma, kappa, r_ref and g(r_ref) = gamma.
   Record all A/B parameters in the trajectory metadata (as `simulate` does).
4. Convention: run `compare_action.py make-probe` / `check` against the project's
   Gamma action for A and B. The expected match is `lattice` at the Ewald tolerance.
   Repeat with 2-3 Ewald splitting parameters. (corrected) A change with the
   splitting parameter must be decomposed into real cutoff, Fourier cutoff, mesh and
   split-implementation parts before any attribution; k=0 is only one candidate.
   If the project turns out to be minimum-image, use `--friction-images minimum`.
5. Use the project's BAOAB and finite-time FDT. The noise operator must be
   S = [m kBT (I - exp(-2 dt Gamma/m))]^{1/2} with the SAME Gamma as the damping
   exp(-dt Gamma/m) (Lanczos can apply both functions of Gamma). Verify on N <= 64
   against `thermostat_factors` (frozen-q identity R C R^T + S S^T = C). Save Q, V
   (velocity), t, box plus sample interval and burn-in, as here.
6. Statistics: `prl_toy_models.py analyze traj.npz --vccf-edges ...` already reads
   project trajectories with fields Q, V, t, box.
7. Dense vs fast: identical noise is not available through Lanczos, so compare
   statistics, not trajectories. Run `ensemble.py`-style independent seeds with both
   and compare means within the combined SEM, plus the static action comparison of
   step 4. Use effective burn-in >= 140 (longer for overdamped lattice B).
