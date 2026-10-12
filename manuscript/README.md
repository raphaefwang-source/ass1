# Manuscript draft (LJ + kernel A)

- `main.tex`, `references.bib`, `figures/` — self-contained; upload the folder (or `overleaf_upload.zip`) to Overleaf
  and compile with pdfLaTeX + BibTeX (latexmk works locally: `latexmk -pdf main.tex`).
- `main.pdf` — compiled draft (28 pages).
- `evidence_map.md` — every quantitative claim mapped to result file, configuration and commit; data availability; open TODOs.
- `scripts/make_figures.py` — rebuilds `fig_operator_lanczos.pdf`, `fig_dt_forest.pdf`, `fig_diffusion.pdf` from the cached
  JSON in `ewald_longitudinal/lj075_results/` (run from the repository root; no trajectories needed).
- `scripts/verify_closed_forms.py` — independent numerical checks of the closed forms quoted in Sections 2–3 and Appendix A.
- All other figures are unmodified copies of figures in `ewald_longitudinal/` (source path given in each caption).

Red `[TODO: ...]` marks in the PDF are items that could not be confirmed from the repository.
