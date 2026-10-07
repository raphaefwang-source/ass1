# production_t60: long coupled runs (fast PPPM + Lanczos 40/16 vs dense full-periodic reference)

Setup: 4 cases (LJ and double well × laws A and B) × 5 seeds, t = 60 each, both methods started from the same v2
states with the same noise. See `../../TOY_INTEGRATION_REPORT.md`, Sections 5 and 8.

## Figures kept for display (long-time physical trends)

- `ab_lj_pppm_lanczos.png`, `ab_double_well_pppm_lanczos.png`: A/B physics of the fast method. They show VACF,
  VCCF grouped by r_ij(t0), C_L and C_T at |k| = 2π/L, RDF and distinct van Hove, as means of 5 seeds with ±1 SEM
  bands.
- `ab_*_reference.png`: the same for the coupled dense reference.
- `fig_*`, `z_*`: method comparisons.

## Caveat: law-A diffusion difference not fully explained

The seed-paired analysis (`paired/PAIRED_ANALYSIS.md`) finds, for double_well_A at max lag 3:
- fast − reference D = +8.0 % (Green-Kubo, 95 % CI +4.1 to +12.0 %) and +5.6 % (MSD, CI +2.6 to +8.5 %);
- all 5 seeds positive.

This difference is not a constant offset:
- it grows with the integration limit;
- it changes sign between time windows;
- lj_A shows the opposite sign (−4 %).

The fast operator agrees with the reference to ≤ 1.5e-7 on the configurations visited.

These figures are shown as long-time physical trends of the toy model. They do not establish that the two methods
give the same law-A diffusion coefficient to a few percent. That question is open, and the targeted long rerun with
a null-control arm has not been run. Fixed-time statistical validation at t ≤ 2 is in
`../../toy_fixed_time_results/`.
