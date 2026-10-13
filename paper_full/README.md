# Full paper (restructured Section 6, rewritten localization proofs)

Source: the complete LaTeX project from `~/papers/my-paper` at commit `44e4886` (Overleaf copy, unchanged there).
Build: `latexmk -pdf main.tex` (pdfLaTeX, see `.latexmkrc`). `main.pdf` is the compiled version (40 pages, no warnings).

## Section 6 mapping

| new | content | former experiments |
|---|---|---|
| 6.1 Operator accuracy and structural checks (`sec:pm-validation`) | mesh response, predictors, spectrum/positivity | 1, 3 |
| 6.2 Lanczos accuracy and production-rank selection (`sec:lanczos-exp`) | rank growth at 1e-8; fixed noise rank at 1e-3; damping rank; reference convergence | 4, 5, 8 |
| 6.3 Conservation, equilibrium, and dynamical accuracy (`sec:dyn-exp`) | momentum, equilibrium; Krylov / spatial / temporal / sampling separation; step sensitivity | 6, 7, 9 |
| 6.4 Computational cost and scaling (`sec:exp-timing`) | complete-step timing at ranks 40/16 | 10 |
| Appendix G (`app:equal-accuracy`) | equal-accuracy timing vs xi, summarized in Sec. 5.2 | 2 |

- Results of 6.2-6.4 are labelled as the earlier setting (rounded |k|, s = 4.100).
- `\pending{...}` marks the places where the Section 6 recomputation (branch `claude/new-session-nmi9ux`,
  `ewald_longitudinal/sec6_results/`) will insert results. No recomputed number has been inserted.
- The dynamics used damping rank 12 (noise 40/48); the timing used 40/16 (56 actions). Earlier 40/12 timing
  (52 actions) is superseded. Source: `ewald_longitudinal/dynamics_results/REPORT.md`.
- Section 7 keeps its own parameters. Its subsection titles no longer carry experiment numbers 11-15.

## Mathematical changes (Appendix C)

- Real-space decay: one integration by parts; explicit bound
  0 <= theta_S - rho(xi^2) e^{-xi^2 r^2} <= e^{-xi^2 r^2}/(sqrt(pi) xi r^2). Replaces the stretched-variable and
  dominated-convergence argument. Eq. (10) unchanged.
- Endpoint asymptotics: integration by parts in I_m (existing symbol kept; m = 5/2, 7/2), boundary term at s = 0
  shown to vanish, remainder bounded by 16(1+2|m-2|) xi^{7-2m} e^{-k^2/(4 xi^2)} / (sqrt(pi) k^4). Coefficients,
  signs and the spectral norm of Eqs. (14)-(15) re-derived; statements unchanged.

## Other edits

- Sec. 5.3: T_step = (r_noise + r_damp) T_Gamma_h + ...; 40/16 (56 actions) named as the earlier setting.
- Conclusion: Section 6 items rewritten to the four groups and qualified.
- Sec. 7: smallest eigenvalue on Range Pi corrected to 4.58-4.84 (`lj075_results/verify_*_A.json`, 152 states).
- Layout: overfull formulas split, breakable underscores in file names, ragged table columns, brace characters in
  a typewriter file name; log free of warnings.
