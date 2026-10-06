# Dynamical correlation functions of the ideal-gas PPPM–Lanczos thermostat

Script: `test_dynamic_correlations.py`, run as `--stage generate`, one trajectory per process, then
`--stage analyze`. Figures `fig1_vacf.png`, `fig2_vccf.png` and `fig3_hydro_modes.png` follow the spirit of
Figs. 1–3 of Wang et al., PRL 131, 177301 (2023), adapted to the screened pure-longitudinal Markovian model. All
numbers are in `correlations.json`.

## Data

The earlier dynamics runs saved only observables, not q_i(t) and p_i(t). New trajectories were therefore generated
by calling `test_true_dynamics.step` **unchanged**:

- **Unchanged inputs.** Same Box, PPPM parameters (ξ = 0.7, s = 4.10, η = 0.7, p = 7), dt = 0.04292 and β = m = 1.
- **Damping rank.** The production damping rank 16 is selected through the module constant `RD`.
- **System.** N = 512 at density 30/216, so L = 15.45 and the screening length is 1/κ = 1.
- **Equilibrium start.** q is uniform and p ~ N(0, β⁻¹mΠ), with total momentum 0. For U = 0 the A–O–A step
  preserves this law exactly, so there is no burn-in and every time origin is an equilibrium origin.
- **Runs.** 600 steps each (t = 25.8). Dense, PPPM + high-rank Lanczos (noise 90 / damping 24) and production PPPM
  + Lanczos (40 / 16) were run on **coupled** inputs (same q₀, p₀ and ξ stream) for seeds 0 and 1. Production
  rank 40 was also run for seeds 2–7.
- **Wall time per step** (4 jobs at once): dense 5.3 s, high-rank 1.9 s, production 0.9 s.
- **Raw trajectories.** These are 170 MB in `correlations_raw/`, which is git-ignored. They can be regenerated
  bit-for-bit from the seeds.

## Estimators

- **C_V(t).** ⟨Σ_i v_i(t)·v_i(0)⟩ / ⟨Σ_i|v_i|²⟩ over all time origins, computed with an FFT.
- **C_vv(t; r₀).** ⟨v_i(0)·v_j(t)⟩ / ⟨|v_i|²⟩ over ordered pairs with minimum-image r_ij(0) in r₀ ± 0.25, for
  r₀ = 0.5, 1, 2, 3 and 4 in units of 1/κ. Origins are taken every 5 steps. The average number of pairs per origin
  is 123, 462, 1802, 4042 and 7175 for the five bins.
  - The correlation is also split into a component along e = r̂_ij(0), C_∥ = ⟨(v_i(0)·e)(v_j(t)·e)⟩/(⟨|v|²⟩/3),
    and the remaining part across e, C_⊥.
  - Because Σ_j v_j = 0 exactly, C_vv(0; r₀) = −1/(N−1). This baseline is drawn dotted in the figure.
- **C_L(k,t) and C_T(k,t).** ũ(k,t) = N⁻¹Σ_j v_j e^{ik·q_j} for k = 2πn/L. The shells are |n| = 1, √2 and 2
  (3, 6 and 3 directions), giving k = 0.407, 0.575 and 0.813. C_T averages both transverse axes.
- **Error bands.** The bands are ±2 SE over the 8 production seeds. σ₁ denotes the single-trajectory scatter
  across those seeds.

### Two limiting references

- **Frozen-Γ_h prediction** (green circles). This is exp(−tΓ_h(q₀)/m)Π, evaluated by dense eigendecomposition on
  the t = 0 configurations of seeds 0 and 1. It is the model's thermostat with particle motion switched off, and is
  exact as t → 0.
- **Free streaming** (Fig. 3, dash-dot). This is the collisionless ideal gas with no thermostat:
  C_L = (1 − k²t²)e^{−k²t²/2} and C_T = e^{−k²t²/2}.

## Results

1. **VACF (Fig. 1).**
   - C_V decays smoothly and almost exponentially: it reaches 1/2 at t = 1.40 and 1/e at t = 2.05, with
     C_V(6) = 0.061. The integral up to t = 6 gives D ≥ 1.94; this is a lower bound because the tail is truncated.
   - The frozen-Γ_h curve is exact over the first step and then decays more slowly (0.45 against 0.37 at t = 2).
     As particles move, Γ_h(q) rotates its eigenvectors, so slow modes of one configuration are not slow in the
     next.
   - All three methods lie on one curve.

2. **VCCF (Fig. 2).**
   - Momentum transferred from i to j produces a delayed positive peak whose height falls and whose time grows
     with r₀:

     | r₀ | 0.5 | 1 | 2 | 3 | 4 |
     |---|---|---|---|---|---|
     | peak height | 0.113 | 0.052 | 0.015 | 0.0055 | 0.0025 |
     | peak time | 0.77 | 1.03 | 1.67 | 2.19 | 2.92 |

   - **Longitudinal versus transverse.** Beyond 1/κ, C_∥ dominates C_⊥: at r₀ = 2–4 the peak of C_∥ is 3.5–9
     times the peak of C_⊥. This is the signature of the pure-longitudinal kernel, which transfers momentum along
     r̂_ij. At r₀ ≤ 1 the two are comparable, because pairs rotate during the transfer time.
   - **Comparison with the frozen prediction.** The frozen-Γ_h prediction captures the initial rise of C_∥ but
     overshoots its peak by a factor of up to 2. It misses C_⊥ at short range, which comes from pair rotation.
   - Even at r₀ = 4 the signal is resolved above noise.

3. **Hydrodynamic modes (Fig. 3).** The measured curves sit between the two limits.
   - **Longitudinal.** C_L crosses zero at k·t ≈ 1.05–1.12, close to the free-streaming value of 1, and has a
     negative lobe. That lobe is shallower and later than in free streaming: −0.34, −0.35 and −0.30 against
     −0.45.
   - **Transverse.** C_T decays far more slowly than free-streaming dephasing: at k = 0.407, C_T(6) ≈ 0.30 against
     0.05. It is still much faster than the frozen thermostat.
   - So the momentum-conserving pair coupling produces a persistent transverse (viscous) momentum mode on top of the
     ideal-gas kinetics, while longitudinal relaxation stays dominated by streaming.

4. **Dense vs PPPM–Lanczos.**
   - **Pathwise (Fig. 1c).** With identical q₀, p₀ and ξ, the trajectories start apart by the per-step Lanczos
     error: about 1e-11 to 1e-10 for high-rank, and 2e-6 to 9e-6 for rank 40.
     - Both differences then grow exponentially at the **same** rate, about 1.0 per unit time (fits: 0.93–1.03).
       The growth comes in jumps at close encounters.
     - The model's coupled multiplicative-noise dynamics therefore amplifies any perturbation, including roundoff.
       The growth is not the Lanczos error accumulating.
     - Paths agree to 1e-3 until t ≈ 4 for rank 40 and t ≈ 13–18 for high-rank. After that they are independent
       realizations. (At N = 64 over 400 steps the paths had stayed together, to 4e-8.)
   - **Statistical (Fig. 1d and `stat_test_vs_dense`).** Once paths decorrelate, the right test is the difference
     of the 2-seed means relative to σ₁. Every point of every correlation lies within 2σ₁:
     - high-rank: max |Δ|/σ₁ = 0.05–0.11;
     - rank 40: max |Δ|/σ₁ = 0.63 (C_V), 1.1 (C_vv), 1.6 (C_∥), 1.7 (C_L) and 1.3 (C_T), with medians 0.14–0.42.
   - **Conclusion.** Production rank 40 / 16 reproduces the dense VACF, VCCF and hydrodynamic-mode correlations
     within statistical error. High-rank Lanczos is indistinguishable from dense.

5. **Conservation.** Total momentum stayed ≤ 6.2e-13 in all runs. The largest dt·λ_max encountered was 5.2
   (14.6 in one rank-40 seed). The smallest Ritz value or eigenvalue was 0.028; nothing was clipped.

## Caveats

- The model has U = 0, so these are the correlations of the thermostatted ideal gas. There is no structural
  correlation, so the VCCF reflects only dissipative and stochastic momentum exchange.
- Hydrodynamic modes at |n| = 1 are statistically limited (±2 SE ≈ 0.07 at late times). Only one box size was
  used, so finite-size effects in C_T (the slowest mode) are not quantified.
- The pathwise comparison is meaningful only before the divergence time. Beyond it the method comparison rests on
  2 coupled seeds against the 8-seed scatter.
