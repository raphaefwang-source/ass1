#!/usr/bin/env python3
"""
Equilibration (burn-in), sampling requirements and long-time diffusion at the lj075 state point, with the production
runner toy_run.py and the named configuration lj_rho0.75_kT1.0_costopt.

Runs (per law A, B), all through toy_run.py (restartable, every parameter recorded in config.json):
    fcc   --init fcc --init-jitter 0.05              seeds 301 302   ordered start (must melt)
    rsa   --init rsa --init-dmin 0.9                 seeds 303 304   random packing, high energy
    lgv   --from-state raw/states/eqinit_s305/306    Langevin-equilibrated positions and momenta (near equilibrium)
    hot   --from-state raw/states/eqhot_s307/308     Langevin positions, momenta at kT = 2 (thermal relaxation)
  production 50 time units, frames EVERY STEP (q unwrapped, v; law B's VACF decays within ~0.02, so GK and the
  save-interval study need every step), monitor every 500 steps; burn-in is NOT imposed in the runner (burn-in 0)
  and is determined offline.

Stages:
    python3 lj075_equilibration.py prepare            # Langevin starting files (eqinit / eqhot)
    python3 lj075_equilibration.py tasks              # print the toy_run command lines (one per run)
    python3 lj075_equilibration.py analyze
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lj075_common as C  # noqa: E402

CONFIG = "lj_rho0.75_kT1.0_costopt"
RUNS = C.RAW / "runs"
DT = 0.005
T_PROD = 50.0
SAVE = 0.02
ICS = {"fcc": (301, 302), "rsa": (303, 304), "lgv": (305, 306), "hot": (307, 308)}


def prepare():
    import lj075_prepare_states as P
    for s in (305, 306, 307, 308):
        name = f"eqinit_s{s}"
        if not (P.STATES / f"{name}.npz").exists():
            P.prepare(name, 30.0)
    for s in (307, 308):
        with np.load(P.STATES / f"eqinit_s{s}.npz") as d:
            arr = {k: d[k] for k in d.files}
        rng = np.random.default_rng([20261009, s, 2])
        p = C.maxwell(rng, 256, 2.0, 1.0)
        arr["p"] = p
        rec = json.loads(str(arr["generation"]))
        rec.update(name=f"eqhot_s{s}", note="positions of eqinit; momenta fresh Maxwell at kT = 2 (hot start)")
        arr["generation"] = np.array(json.dumps(rec))
        np.savez(P.STATES / f"eqhot_s{s}.npz", **arr)
    for s in (309,):
        if not (P.STATES / f"eqinit_s{s}.npz").exists():
            P.prepare(f"eqinit_s{s}", 30.0)


def commands():
    cmds = []
    for law in ("A", "B"):
        for ic, seeds in ICS.items():
            for s in seeds:
                out = RUNS / f"eq_{law}_{ic}_s{s}"
                c = [sys.executable, str(HERE / "toy_run.py"), "--config", CONFIG, "--potential", "lj", "--kernel",
                     law, "--N", "256", "--seed", str(s), "--dt", str(DT), "--allow-unverified", "--production",
                     str(T_PROD), "--save-every-steps", "1", "--monitor-every-steps", "500", "--resume",
                     "--out", str(out), "--quiet"]
                if ic in ("fcc", "rsa"):
                    c += ["--init", ic] + (["--init-jitter", "0.05"] if ic == "fcc" else ["--init-dmin", "0.9"])
                else:
                    pre = "eqinit" if ic == "lgv" else "eqhot"
                    c += ["--from-state", str(C.RAW / "states" / f"{pre}_s{s}.npz")]
                cmds.append(c)
    return cmds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["prepare", "tasks", "analyze"])
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "tasks":
        for c in commands():
            print(" ".join(c))
    else:
        import lj075_equil_analysis
        lj075_equil_analysis.main()


if __name__ == "__main__":
    main()
