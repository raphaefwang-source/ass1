# Full paper: theory and methods + operator experiment

Entry point: `main.tex` (pdfLaTeX, `latexmk -pdf main.tex`). Compiled: `main.pdf` (24 pages, log free of warnings).
Overleaf: `overleaf_upload.zip` = `main.tex`, `references.bib`, `figures/` (only the four figures used).
Source of the text before this revision: Overleaf project at commit `7b428ef` (= this directory on branch
`claude/paper-structure-ibp`, commit `f3160c6`).

## Contents (this version)

- Sections 1-5: kernel, Ewald splitting, PPPM graph-Laplacian action, finite-time FDT thermostat with Lanczos, time
  integration, error budget and parameter selection. The rewritten proofs of Appendix C are kept.
- Section 6: the operator experiment (former Experiment 1), rewritten from `ewald_longitudinal/exp1_results/` after
  the |k|-rounding fix (`run_2026-10-13_exact_k.log`, exit 0; `integrity.json` all row counts ok, no NaN/Inf):
  configurations, reference, parameter sets, separate error sources, complete production action, probes,
  spectrum, box length. Technical details in Appendix G.
- Parameter-set status (`parameter_set_status.json`): T5 10/14, max 1.2875e-5; T7 39/44, max 2.1056e-7;
  T9 10/14, max 2.5451e-9.

## Removed in this version

Former Experiments 2-10 (equal-accuracy timing, spectrum study on old mesh operators, Lanczos ranks, dynamics,
timing), the Lennard-Jones application (Section 7, former Experiments 11-15, its appendix and supplementary
figures), the old mesh-decomposition study that formed the earlier Experiment 1, and every figure, table, cross
reference and conclusion that depended on them. Result files stay in `ewald_longitudinal/` and in Git history.

## Figures

`scripts/make_exp1_figures.py` regenerates `figures/exp1_{cutoffs,mesh,operator,probes_box}.pdf` (vector PDF,
6.5 in wide = text width, 8-9 pt text) from the CSV files; it reads only `exp1_results/` and writes only here.
