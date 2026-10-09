#!/usr/bin/env python3
"""
Extract configurations actually visited by the lj075 runs (production runner and dt study) for the static operator /
Lanczos verification (lj075_verify.py --tag visited --states 'raw/visited/*.npz').

Per equilibration run: frames at t = 0, 0.25, 1, 5, 20, 50 (early frames of the fcc / rsa starts are far from
equilibrium: lattice-like or high-energy), the frame with the smallest pair distance and the frame with the largest
Ritz value seen by the runner; the stored momenta p = m v are kept as the 'actual' damping input.
Per dt-study replica: the final frame and the smallest-rmin frame of the coarsest (dt = 0.01) level.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lj075_common as C  # noqa: E402

OUT = C.RAW / "visited"


def main():
    import toy_run
    OUT.mkdir(parents=True, exist_ok=True)
    L, kT = C.box_length(256, 0.75), 1.0
    n = 0
    man = {}
    for run in sorted((C.RAW / "runs").glob("eq_*")):
        st = json.loads((run / "status.json").read_text())
        if st["state"] != "complete":
            continue
        d = toy_run.load_run(run)
        dt = d["config"]["resolved"]["dt"]
        t = d["frame_step"] * dt
        diag = d["diag"]
        picks = {f"t{x:g}": int(np.argmin(np.abs(t - x))) for x in (0, 0.25, 1, 5, 20, 50)}
        picks["rmin"] = int(np.argmin(diag[:, 6]))
        picks["ritzmax"] = int(np.nanargmax(diag[:, 9]))
        for tag, k in picks.items():
            name = f"{run.name}_{tag}"
            np.savez(OUT / f"{name}.npz", q=d["Q"][k] % L, p=d["V"][k], L=L, kT=kT, N=256)
            man[name] = dict(run=run.name, frame=k, t=float(t[k]), rmin=float(diag[k, 6]))
            n += 1
    for f in sorted((C.RAW / "dt_study").glob("*_L4_T10.npz")):
        with np.load(f) as z:
            S = z["S_0.01"]
            for tag, k in (("final", len(S) - 1), ("rmin", int(np.argmin(S[:, 5])))):
                name = f"dt_{f.stem}_{tag}"
                np.savez(OUT / f"{name}.npz", q=z["Q_0.01"][k].astype(float) % L, p=z["V_0.01"][k].astype(float),
                         L=L, kT=kT, N=256)
                man[name] = dict(source=f.name, level=0.01, frame=k, t=float(S[k, 0]), rmin=float(S[k, 5]))
                n += 1
    C.write_json(C.OUT / "visited_states.json", dict(states=man, provenance=C.provenance()))
    print(f"{n} visited configurations written to {OUT}")


if __name__ == "__main__":
    main()
