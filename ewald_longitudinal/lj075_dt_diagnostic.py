#!/usr/bin/env python3
"""
Post-hoc diagnostic of the lj075 dt study (NOT part of the pre-registered criteria): paired differences among the
coarser levels (0.005 vs 0.0025, 0.01 vs 0.0025, 0.01 vs 0.005) for the scalars and MSD(t). Written after the final
analysis showed the same MSD(5) offset for every level against 0.00125, to tell a fluctuation of the reference level
from a dt effect (a dt effect grows with dt; a reference fluctuation is common to all comparisons against it).
Writes lj075_results/dt_diagnostic.json.

    python3 lj075_dt_diagnostic.py [--law A]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lj075_common as C  # noqa: E402

LEVELS = (0.00125, 0.0025, 0.005, 0.01)
MSD_TIMES = (0.1, 0.5, 1.0, 2.0, 5.0)
PAIRS = ((0.005, 0.0025), (0.01, 0.0025), (0.01, 0.005))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--law", default="A")
    args = ap.parse_args()
    files = sorted((C.RAW / "dt_study").glob(f"{args.law}_rep*_L4_T10.npz"))
    obs = {dt: dict(T=[], U=[], P=[], msd=[]) for dt in LEVELS}
    for f in files:
        with np.load(f) as z:
            for dt in LEVELS:
                S = z[f"S_{dt}"]
                k = S[:, 0] >= 1.0 - 1e-12
                m = C.msd(z[f"Q_{dt}"][k].astype(float), int(round(max(MSD_TIMES) / dt)) + 1)
                obs[dt]["T"].append(S[k, 1].mean())
                obs[dt]["U"].append(S[k, 2].mean())
                obs[dt]["P"].append(S[k, 3].mean())
                obs[dt]["msd"].append([m[int(round(t / dt))] for t in MSD_TIMES])
    o = {dt: {k: np.array(v) for k, v in d.items()} for dt, d in obs.items()}
    res = dict(note=__doc__.split("\n\n")[0].strip(), n_replicas=len(files), provenance=C.provenance(),
               level_means={str(dt): dict(T_kin=C.t_ci(o[dt]["T"]), U_per_N=C.t_ci(o[dt]["U"]),
                                          pressure=C.t_ci(o[dt]["P"]),
                                          msd={f"{t:g}": C.t_ci(o[dt]["msd"][:, j]) for j, t in enumerate(MSD_TIMES)})
                            for dt in LEVELS},
               pairs={})
    for a, b in PAIRS:
        res["pairs"][f"{a}_vs_{b}"] = dict(
            T_kin_rel=C.t_ci((o[a]["T"] - o[b]["T"]) / o[b]["T"].mean()),
            U_per_N_rel=C.t_ci((o[a]["U"] - o[b]["U"]) / abs(o[b]["U"].mean())),
            pressure_abs=C.t_ci(o[a]["P"] - o[b]["P"]),
            msd_rel={f"{t:g}": C.t_ci((o[a]["msd"][:, j] - o[b]["msd"][:, j]) / o[b]["msd"][:, j].mean())
                     for j, t in enumerate(MSD_TIMES)})
        print(f"{a} vs {b}: " + ", ".join(f"MSD({t}) {100 * v[0]:+.2f}% [{100 * v[1]:+.2f}, {100 * v[2]:+.2f}]"
                                         for t, v in res["pairs"][f"{a}_vs_{b}"]["msd_rel"].items()) +
              f"; P {res['pairs'][f'{a}_vs_{b}']['pressure_abs']}", flush=True)
    C.write_json(C.OUT / "dt_diagnostic.json", res)


if __name__ == "__main__":
    main()
