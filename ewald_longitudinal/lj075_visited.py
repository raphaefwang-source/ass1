#!/usr/bin/env python3
"""
Extract configurations actually visited by the lj075 runs (production runner and dt study) for the static operator /
Lanczos verification (lj075_verify.py --tag visited --states 'raw/visited/*.npz').

Per equilibration run (frames every step): steps at t = 0, 0.25, 1, 5, 20, 49.995 (early frames of the fcc / rsa starts
are far from equilibrium: lattice-like or high-energy), the step with the smallest pair distance and the step with the
largest Ritz value seen by the runner. For each picked step n the EXACT inputs of the next O-step are rebuilt:
    p' = p_n + dt/2 F(q_n),  q_half = q_n + dt/(2m) p'   (damping input Pi p' at Gamma_h(q_half)),
    xi = the (n+1)-th standard-normal (N, 3) draw of the run's noise stream default_rng([seed, 1]) (noise input).
Per dt-study replica: the final frame and the smallest-rmin frame of the coarsest (dt = 0.01) level.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lj075_common as C  # noqa: E402
import toy_dynamics as td  # noqa: E402

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
        if not np.all(np.diff(d["frame_step"]) == 1) or d["frame_step"][0] != 0:
            raise ValueError(f"{run}: frames are not every step from step 0")
        nmax = len(t) - 2
        picks = {f"t{x:g}": min(nmax, int(np.argmin(np.abs(t - x)))) for x in (0, 0.25, 1, 5, 20, 50)}
        picks["rmin"] = min(nmax, int(np.argmin(diag[:-1, 6])))
        picks["ritzmax"] = min(nmax, int(np.nanargmax(diag[1:, 9])))     # diag ritz of step k+1 = O-step at k
        seed = d["config"]["plan"]["seed"]
        rng = np.random.default_rng([seed, 1])
        xis = {}
        for k in range(max(picks.values()) + 1):
            x = rng.standard_normal((256, 3))
            if k in picks.values():
                xis[k] = x
        m = d["config"]["resolved"]["model"]["mass"]
        for tag, k in picks.items():
            qn, pn = d["Q"][k], d["V"][k] * m
            F, _, _ = td.conservative_force_neighbor(qn, L, "lj")
            p1 = pn + 0.5 * dt * F
            qh = qn + 0.5 * dt / m * p1
            name = f"{run.name}_{tag}"
            np.savez(OUT / f"{name}.npz", q=qh % L, p=p1, xi=xis[k], L=L, kT=kT, N=256)
            man[name] = dict(run=run.name, step=int(d["frame_step"][k]), t=float(t[k]), rmin=float(diag[k, 6]),
                             inputs="exact: q_half, p' (damping input), xi (noise draw)")
            n += 1
    for f in sorted((C.RAW / "dt_study").glob("*_L4_T10.npz")):
        with np.load(f) as z:
            S = z["S_0.01"]
            for tag, k in (("final", len(S) - 1), ("rmin", int(np.argmin(S[:, 5])))):
                name = f"dt_{f.stem}_{tag}"
                np.savez(OUT / f"{name}.npz", q=z["Q_0.01"][k].astype(float) % L, p=z["V_0.01"][k].astype(float),
                         L=L, kT=kT, N=256)
                man[name] = dict(source=f.name, level=0.01, frame=k, t=float(S[k, 0]), rmin=float(S[k, 5]),
                                 inputs="full-step q and p of the coarsest level (not the exact O-step inputs)")
                n += 1
    C.write_json(C.OUT / "visited_states.json", dict(states=man, provenance=C.provenance()))
    print(f"{n} visited configurations written to {OUT}")


if __name__ == "__main__":
    main()
