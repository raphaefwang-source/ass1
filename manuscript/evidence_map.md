# Evidence map for `main.tex`

Repository `raphaefwang-source/ass1`, branch `claude/new-session-nmi9ux`, evidence frozen at HEAD `39bb4f6`.
Paths are relative to `ewald_longitudinal/` unless they start with `manuscript/`. "Commit" is the commit that added
or last changed the result file (`git log -- <file>`). Configuration names refer to `toy_configs.py`.

Status labels: **verified** (result meets a stated criterion), **provisional** (decision taken without meeting the
registered rule), **under-sampled** (measured, interval too wide), **historical** (other system; not the LJ+A state
point), **derived here** (computed for this draft by a script in `manuscript/scripts/`).

## 1. Model and structure (Sections 2–4)

| claim in paper | source | status |
|---|---|---|
| SDE, BAOAB with O-step `p_mean + R Πp + S Πξ`, one Γ_h per step for damping and noise | `toy_dynamics.py` (`Thermostat.o_step`, `run`), `test_true_dynamics.py:101-142` | code |
| LJ with C2 quintic switch on [2, 2.5]σ including S'(r)u0 | `toy_models/prl_toy_models.py:22-46`; finite-difference test `test_lj075_config.py` | code + test |
| kernel A: g = c e^{-κr}/r, c = γ r_ref e^{κ r_ref} = 1.6148 (γ 0.5, κ 0.7, r_ref 1.3) | `radial_kernels.py:1-60` (`amplitude`), `toy_configs.MODEL` | code |
| full periodic lattice sum, k = 0 retained, self images cancel | `toy_models/prl_toy_models.py:142-198` (`LatticeFriction`) | code |
| ρ ĝ(0)/3 = 10.35 at ρ = 0.75 | `lj075_results/verify_*_A.json` (`per_state.*.k0_shift`), REPORT.md §2.1 | verified |
| Lemma 3.1, θ_S closed form, E1 forms of φ, ψ, full-kernel Fourier form, λ∥(1.3) = −0.484 | `manuscript/scripts/verify_closed_forms.py` (run 2026-10-12; max rel. errors 0, 4e-16, 1.3e-15, 7.8e-16, FD 1.3e-8, 1.7e-11, 7.2e-10) | derived here |
| A(k), B(k) formulas and asymptotics, K̂_L not PSD | `DERIVATIONS.md` §5, §7; `verify_ewald_longitudinal.py`; `results/summary_kappa1_xi1.json` (commit 5daf203) | historical (κ = ξ = 1) |
| PPPM graph-Laplacian action, self-term cancellation, influence function K̂_L/|Ŵ|² | `test_parameter_selection.py:206-299` (`Mesh.apply/degree/graph_action`), `test_lanczos_fdt.py:92-127` (`GammaH`) | code |
| Lanczos with full reorthogonalisation, Π each step, abort on Ritz ≤ 0 | `test_lanczos_fdt.py:143-176`, `test_true_dynamics.py:85-98` | code |
| nested coupling a, b; 2 fine = 1 coarse to 3e-15 (dense), ≤ 4e-11 (Lanczos) | `lj075_coupled.py`; REPORT.md §2.2, §3.1 | verified |

## 2. Spatial accuracy (Section 5.2, Fig. 1, Table 2)

| claim | file | commit | status |
|---|---|---|---|
| ε_op ≤ 7.6e-8 on 152 states; vs k≠0 part ≤ 1.60e-7 | `lj075_results/verify_representative_A.json`, `verify_visited_eq_A.json`, `verify_visited_dt_A.json`; `summary_tables.md` | f2fb05e/d43b322, 4559a5c, 64583cd | verified |
| p raised 7 → 8 (p = 7 gave 2.04–2.15e-7 on fcc states) | `lj075_results/pppm_screen.json`; `toy_configs.py` comment | d43b322 | verified |
| alternative sets (η 0.6 p 7 M 27; s 4.4 η 0.6 p 7 M 30) | `verify_representative_A_0.85_4.1_0.6_7.0.json`, `..._4.4_0.6_7.0.json` | b8ee2cc | verified (static only) |
| rank 12 gives 2.88e-8 on `eq_A_fcc_s301_t5`; ranks 14/5 worst 6.4e-10 / 2.8e-11 | `verify_visited_eq_A.json`; `verification_runner.json` (entry `lj_rho0.75_kT1.0_A_prod_A_N256_dt0.005`) | 4559a5c, 80f8191 | verified |
| on-line monitor of dynamic runs ≤ 1.6e-10 (noise overall), 1.8e-9 (k_min) | REPORT.md §2.2; `dt_study.json` (`lanczos_monitor_max`) | df43e2d | verified |
| λ_min(Γ_h on RanΠ) 4.58–4.69; symmetry ≤ 2e-17; null ≤ 2e-15; literal vs dense ≤ 8e-16 | same verify JSONs (`lam_min`, `sym_resid`, `null_resid`, `literal_vs_dense`) | as above | verified |
| operator-sensitivity differences (T 5.9e-5, U/N 1.4e-5, MSD ≤ 1.3e-4, P 4.5e-4) | `lj075_results/operator_sensitivity.json`; tight set in `lj075_operator_sensitivity.py:37` | f3797b1 | observed maxima, n = 2 |
| Fig. 1 | `manuscript/figures/fig_operator_lanczos.pdf` from `manuscript/scripts/make_figures.py` | — | derived here (from cached JSON) |

## 3. Time discretisation (Section 5.3, Table 3, Figs. 2, 3, 6)

| claim | file | commit | status |
|---|---|---|---|
| criteria committed before runs | `lj075_results/dt_criteria.json` (91e20d4, 2026-10-09 17:38) precedes `dt_study.json` (df43e2d, 23:51); ancestry checked | 91e20d4 | verified ordering |
| Table 3 values, no dt validated, MSD(5) FAIL at 0.0025 | `lj075_results/dt_study.json`; `summary_tables.md` | df43e2d | **provisional** dt = 0.005 |
| coarse levels agree with each other (post hoc) | `lj075_results/dt_diagnostic.json` | 29a9ee5 | post hoc only |
| pathwise RMS dv 0.025/0.057/0.12 at t = 1 | `dt_study.json` (`pathwise_rms_dv`); REPORT.md §3.3 | df43e2d | verified |
| replicas needed (25 for MSD(0.5), 80 for MSD(1), 60–100 for pressure) | REPORT.md §3.3–3.4 | 80f8191 | estimate |
| D_MSD windows, plateau rule, dt ratio | `lj075_results/longD.json` | 199ab38 | **under-sampled** |
| config records `dt_validated=None`, `dt_provisional=0.005` | `toy_configs.py` (`lj_rho0.75_kT1.0_A_prod`) | 64583cd | — |
| Figs. 2, 3 | `manuscript/figures/fig_dt_forest.pdf`, `fig_diffusion.pdf` (script as above) | — | derived here |
| Fig. 6 | copy of `lj075_results/dt_study_A.png` | df43e2d | — |
| same-dt fast vs dense (single path) | `lj075_results/hpc_pilot/local_20261010/pilot_results.json` | 39bb4f6 | local emulation, n = 1 |

## 4. Observables (Section 5.4, Figs. 4, 5, 7, 8, 9)

| claim | file | commit | status |
|---|---|---|---|
| U/N, T, P, RDF peak pool means; burn-in 2 (recommend 5); ANOVA p 0.21–0.78; replica requirements | `lj075_results/equilibration.json`; `summary_tables.md` | 7119efe/80f8191 | burn-in verified; start independence under-sampled |
| save-interval rule (GK bias 0.23% at Δ = 0.01) | `lj075_results/sampling.json` | 7119efe | verified at dt 0.005 |
| pressure and T_kin definitions | `lj075_common.py:164-181` | — | code |
| observable definitions, reproduction ≤ 1e-15 | `lj075_results/observable_dt/definition_check.json`; `lj075_observable_dt.py` | 410ffca | verified |
| data groups (16 primary, 11 secondary; no pooling), origins, grid | `observable_dt/inventory.json` | 686eea6 | — |
| protocol v2 frozen, v3 amendment before differences | `observable_dt/protocol_v2_frozen.json` (c0434df), `protocol.json` (7484b7d), results (5649159) | — | ordering verified from git |
| deviations from protocol | `observable_dt/protocol_deviations.json` | 410ffca | — |
| band self-test (worst coverage 0.94) | `observable_dt/band_selftest.json` | c0434df | — |
| 1 SUPPORTED / 5 NOT SUPPORTED / 65 INCONCLUSIVE; 7 FAIL cells; resolved deviations; whole-curve bounds | `observable_dt/results.json`, `conclusions_table.md` (v3); v2 in `results_v2.json`, `conclusions_table_v2.md` | 5649159 | final = v3; v2 is not used for conclusions |
| ⟨|v|²⟩ −0.38% [−0.67, −0.09]; C_T(k1) half-decay +3.6% [0.6, 6.6] | `observable_dt/results.json`; REPORT_observable_dt.md | 5649159/410ffca | resolved, INCONCLUSIVE vs 1% |
| D_GK 0.01 vs 0.005 +0.55% [−0.78, 1.88] of D_GK | REPORT_observable_dt.md ("derived" from `per_replica` values) | 410ffca | not shown at 1% |
| burst follow-up (170 bursts, 6.8 core-h) | `observable_dt/followup_plan.json` | 410ffca | not run |
| Figs. 4, 5, 8, 9 | copies of `observable_dt/{vacf,vccf,currents,van_hove}_dt{,_secondary}.png` | 5649159/410ffca | — |
| Fig. 7 | copy of `lj075_results/equilibration_A.png` | 7119efe | — |

## 5. Cost (Section 5.5, Table 4)

| claim | file | commit | machine | status |
|---|---|---|---|---|
| ms/step 124.8 (12/5), 134.9 (14/5), 117.9, 140.0 | `lj075_results/pppm_cost.json` | 1ea4d2a/df43e2d | host "vm", 1 thread | measured |
| saving 48.7% / 50.4% of dt 0.01 over 0.005 | `observable_dt/timing.json` | bf6a13e | host "vm", 1 thread, separate session | measured |
| fast vs dense step 0.208 vs 1.129 s; RSS 121 vs 604 MB | `hpc_pilot/local_20261010/pilot_results.json`, `pilot_table.md` | 39bb4f6 | VM, Xeon 2.80 GHz, 1 thread | local emulation, single path |
| total recorded 32.3 core-h (law A) | `summary_tables.md` | 199ab38 | — | — |

Timings from different files are never combined into a speed-up ratio in the paper.

## 6. Historical evidence (Appendix D, Table 5, Fig. 10) — not the LJ+A state point

| directory | generating script | last commit | system |
|---|---|---|---|
| `results/`, `figures/` | `verify_ewald_longitudinal.py`, `verify_periodic_graph_psd.py` | 5daf203 / b2bf351 | κ = ξ = 1, random N = 30/40 |
| `parameter_selection_results/` | `test_parameter_selection.py` | 32c3068 | κ = 1 |
| `equal_accuracy_xi_results/` | `test_equal_accuracy_xi.py` | 6c5959e | κ = 1, N = 4000 timing (machine unrecorded) |
| `mesh_error_decomposition_results/` | `test_mesh_error_decomposition.py` | 6c5959e | κ = ξ = 1 |
| `radial_kernel_static_results/` | `test_radial_kernel_static.py` | 6b58244 | toy N = 64, L = 5.5, kT 0.7, laws A/B |
| `lanczos_fdt_results/` | `test_lanczos_fdt.py` | de846cd | unit kernel κ = 1, ρ = 30/216 |
| `lanczos_range_start_results/` | `test_lanczos_range_start.py` | e49f761 | same |
| `fixed_rank_budget_results/` | `fixed_rank_large_n.py` (N ≤ 4000 tables from a script not in the repo) | b524c56 | same, N ≤ 16000 |
| `dynamics_results/` | `test_true_dynamics.py`, `test_production_timing_rank40_16.py`, `test_dynamic_correlations.py` | 6317ca5 (main: 905f2a2, 8f596b5, bbe5851) | ideal gas κ = 1 |
| `toy_fixed_time_results/` | `test_toy_fixed_time.py` | 3cd9e8d | double well, A, N = 64 |
| `toy_dynamics_results/` | `toy_dynamics.py`, `test_toy_dynamics.py`, `analyze_toy_paired.py` | 0acdf60 | LJ/DW × A/B, N = 64 |
| `cost_optimization_results/` | `toy_cost_opt.py`, `verify_production_configs.py` | e48269e | toy state, 2.30 GHz VM |
| `fft_vs_direct_results/` | `bench_fft_vs_direct.py` | f6289b0 | toy state, double well, 2.10 GHz Xeon |
| `vacf_sampling_results/` | `vacf_sampling_check.py` | f6289b0 | double well, N = 64 |
| law B at LJ point (stopped) | `verify_representative_B*.json` | c6cbd6f | static only |

Supersession: `lj075_results/` supersedes all toy-state PPPM sets for the LJ+A point (p 7 → 8; ranks 32/4 and 12/5 → 14/5).
`hpc_pilot/local_20261010` supersedes the earlier raw pilot of commit 698fdcd (checkpoint-cadence bug); it does not
supersede `hpc/local_pilot_20261008` or `hpc/local_pilot_emulation_20261008`, which concern the double-well model.
Protocol v3 (`results.json`) supersedes v2 (`results_v2.json`) for the observable comparison.

## 7. Data availability

| item | in git | can be re-analysed here | note |
|---|---|---|---|
| LJ+A raw trajectories (`lj075_results/raw/`: dt study, long-D, equilibration, visited frames) | no (git-ignored) | **no** | SHA-256 in `observable_dt/inventory.json`; location TODO |
| LJ+A analysis caches (all JSON, `per_replica_observables.npz`) | yes | yes (figures, tables, re-binning of stored statistics) | — |
| HPC pilot cases and logs | yes | yes | — |
| `toy_dynamics_raw/` (76 files, 413.8 MB) | no | **no** | manifest `toy_dynamics_results/raw_trajectory_manifest.json` |
| `dynamics_results/correlations_raw/` (170 MB) | no | regenerable from seeds (not done) | — |
| `dynamics_results/raw_*.npz` (observable series, 5.6 MB) | yes | yes | not q/p trajectories |
| `toy_fixed_time_results/paths/` (80 npz, 4.35 MB) | yes | yes | — |
| real HPC/Slurm results | none exist | — | only local emulation |

## 8. Open TODOs in the manuscript

1. Author list and affiliations.
2. Bibliographic data for Lyu & Lei PRL 131, 177301 (title to verify) and for the Zhao–Dou–Lei Landau preprint (arXiv id, year).
3. Location of the raw LJ+A trajectories.
4. Which `lj075_hpc_pilot.py` monitor column is the noise-function Lanczos error (4.9e-11 vs 4.3e-12).
