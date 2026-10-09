#!/usr/bin/env python3
"""
PPPM parameter screening for law A at the lj075 state point (the costopt_hiacc set fails the 2e-7 budget on
fcc-lattice states at rho 0.75). Spectral operator error ||Gamma_h - Gamma_ref||_2 / ||Gamma_ref||_2 (full periodic
lattice-sum reference) on three fcc lattice states (jitter 0, 0.02, 0.05) and three canonical liquid states, plus
the timing of one Gamma_h build and one literal matvec (single thread). Also the error relative to the k != 0 part
of the reference, ||Gamma_h - Gamma_ref||_2 / ||Gamma_ref - c0 (N I - 1 1^T) (x) I_3||_2 (the k = 0 term is exact).
Law B: the costopt set and tighter candidates (its full-norm PASS rests largely on the exact k = 0 term).
Writes lj075_results/pppm_screen.json.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import lj075_common as C  # noqa: E402
import prl_toy_models as v2  # noqa: E402
import radial_kernels as rk  # noqa: E402
from lj075_verify import lattice_matrix  # noqa: E402

CANDS = {"A": [(0.85, 4.1, 0.6779116381586452, 7), (0.85, 4.1, 0.6779116381586452, 8), (0.85, 4.1, 0.6, 7),
               (0.85, 4.1, 0.55, 7), (0.85, 4.3, 0.6779116381586452, 7), (0.85, 4.4, 0.6, 7), (0.85, 4.3, 0.6, 8),
               (0.8, 4.1, 0.6, 7)],
         "B": [(1.0, 3.9, 0.6827747058642308, 7), (1.0, 3.9, 0.6827747058642308, 8), (1.0, 3.9, 0.6, 7),
               (1.0, 4.2, 0.6, 7), (1.0, 4.2, 0.6, 8), (1.0, 4.4, 0.55, 8)]}


def main():
    L = C.box_length(256, 0.75)
    states = {f"fcc_jitter{j}": C.fcc_positions(256, L, j, np.random.default_rng(5 + k))
              for k, j in enumerate((0.05, 0.02, 0.0))}
    for f in ("fcc_s1", "rsa_s4", "rsa_s6_x2"):
        states[f] = np.load(C.RAW / "states" / f"{f}.npz")["q"]
    out = dict(states=list(states), provenance=C.provenance(), budget=dict(A=2e-7, B=5e-8), candidates={})
    for law, cands in CANDS.items():
        lat = v2.LatticeFriction(L, kernel=law, gamma=0.5, kappa=0.7, r_ref=1.3)
        Gr = {k: lattice_matrix(lat, q % L) for k, q in states.items()}
        Gk0 = lat.zero_mode * np.kron(256 * np.eye(256) - np.ones((256, 256)), np.eye(3))
        nk = {k: np.linalg.norm(G - Gk0, 2) for k, G in Gr.items()}
        for xi, s, eta, p in cands:
            ff = rk.FastFriction(L, rk.toy_kernel(law, 0.5, 0.7, 1.3, xi), s, eta, p, pair_search="tree")
            dif = {k: np.linalg.norm(ff.gamma_h(q).dense() - Gr[k], 2) for k, q in states.items()}
            errs = {k: float(dif[k] / np.linalg.norm(Gr[k], 2)) for k in states}
            errs_k = {k: float(dif[k] / nk[k]) for k in states}
            op = ff.gamma_h(states["rsa_s4"])
            v = np.random.default_rng(0).normal(size=768)
            t0 = time.perf_counter()
            for _ in range(20):
                op.matvec(v)
            tm = (time.perf_counter() - t0) / 20
            t0 = time.perf_counter()
            for _ in range(5):
                ff.gamma_h(states["rsa_s4"])
            tb = (time.perf_counter() - t0) / 5
            key = f"{law} xi{xi} s{s} eta{eta:.4f} p{p}"
            rec = ff.record()
            out["candidates"][key] = dict(law=law, xi=xi, s=s, eta=eta, p=p, M=rec["M"], rc=rec["rc"], errors=errs,
                                          max_error=max(errs.values()), errors_vs_k_nonzero=errs_k,
                                          max_error_vs_k_nonzero=max(errs_k.values()),
                                          matvec_ms=tm * 1e3, build_ms=tb * 1e3,
                                          passes=max(errs.values()) <= out["budget"][law],
                                          passes_vs_k_nonzero=max(errs_k.values()) <= out["budget"][law])
            print(f"{key}: M {rec['M']} max {max(errs.values()):.2e} (k!=0 norm {max(errs_k.values()):.2e}) matvec "
                  f"{tm * 1e3:.2f} ms build {tb * 1e3:.1f} ms", flush=True)
    C.write_json(C.OUT / "pppm_screen.json", out)


if __name__ == "__main__":
    main()
