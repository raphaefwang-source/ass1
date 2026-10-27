# Positive-mesh B2 experiment

This is a separate B2 implementation and result set, based on commit `99b416fb8a5eba3d4130f62a39110f6f00197188` of `method-a-only-cleanup-review`. The original PPPM, configurations, scripts, plots and numerical results are preserved byte for byte. The 18 existing independent direct-reference caches are reused read-only. Missing independent references are stored in the separate ignored `raw/positive_mesh_b2_refs` directory, preserving the original runner's cache behavior.

- [Mathematical proof and assumptions](proof.md)
- [Full experiment report](report.md)
- Implementation: [`../../positive_mesh.py`](../../positive_mesh.py)
- Experiment 1: `results/sweeps.csv`, `results/mesh_rate_fits.csv`, `figures/fig1_operator_six_sweeps.png`, `figures/fig2_action_six_sweeps.png`
- Original PPPM comparison: `figures/fig3_original_pppm_comparison.png`; original six-panel figures remain in `../accuracy_sweeps/figures/`.
- Experiment 2: `results/theory_initialization.csv`, `results/search.csv`, `results/validation.csv`, `results/final_timing_raw.csv`, `results/final_timing.csv`, `results/best_configuration.json`.
- Setup, timing protocol, reference provenance, construction audits and all plotted series are saved separately.

## Reproduction

From this directory, using Python with NumPy, SciPy, Matplotlib and threadpoolctl:

```sh
python3 -m unittest discover -p 'test_*.py' -v
python3 build_references.py --workers 3
python3 run_experiments.py sweeps fits --workers 3
python3 audit.py
python3 optimize.py calibrate
python3 optimize.py search
python3 optimize.py local
python3 optimize.py boundary
# Wait for every accuracy/search/reference process to finish before timing.
python3 time_candidates.py
python3 final_diagnostics.py
python3 plot_results.py
python3 summarize_results.py
```

The sweep and search stages resume from CSV checkpoints. References use the original unsplit direct periodic kernel, keyed by screening, density, positions and amplitude. `time_candidates.py` refuses to replace an existing completed timing pass; archive the timing outputs before requesting a fresh pass. The checked-in figures and report can be regenerated without rerunning timings. The final timing scripts use one computational thread and randomized candidate blocks on the same host. Screen timings may overlap accuracy work and are used only as provisional cost guides.

## Operator API

```python
from radial_kernels import RadialEwald
from positive_mesh import PositiveMesh, PositiveGamma

K = RadialEwald('A', kappa=0.2, xi=xi, amp=1.0)
mesh = PositiveMesh(L, K, M=M, p=p, sT=36.0)
op = PositiveGamma(q, L, K, rc=rc, mesh=mesh)
y = op.apply(X)  # X and y have shape (N,3)
```

`sT=kappa*R_T` controls a cubical physical-image cutoff for constructing the grid table. It is **not** the original `sF=k_c/(2*xi)`. No Fourier cutoff, deconvolution or multiplier clipping is used. Parameter-only tables can be shared across configurations; `mesh.with_order(p)` also shares the table when changing assignment order. Configuration-dependent geometry, stencils and the complete degree field are cached by `PositiveGamma`.

The PSD theorem applies to every legal configuration and finite positive construction rule; float64 roundoff and accuracy certification are discussed explicitly in the proof. The implementation preserves the full screened zero mode and cancels the entire interpolated particle self response using the same table in both degree and field actions.
