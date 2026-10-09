#!/usr/bin/env python3
"""
Physical effect of the PPPM operator error at the lj075 state point (the operator error is not a physical error;
this measures the latter directly).

For each law, two operators are run in lock-step from the SAME canonical state with the SAME noise draws at
dt = 0.005 (the production integrator of lj075_coupled.Level, single level):
    production   the configuration's PPPM set (A: xi 0.85 s 4.1 eta* 0.6779 p 8; B: xi 1.0 s 3.9 eta* 0.6828 p 7)
    tight        a much tighter set (A: xi 0.85 s 4.4 eta 0.6 p 7, 1.7e-8; B: xi 1.0 s 4.2 eta 0.6 p 7, 4.3e-9 full /
                 2.4e-8 k!=0 norm), same Lanczos ranks
Recorded: pathwise RMS differences of q and v vs time, and the paired differences of the time averages of T_kin,
U/N and pressure over [1, t_end] and of the MSD at t = 0.5, 2, 5 (relative to the 1% dt-study tolerances).

    python3 lj075_operator_sensitivity.py --law B --replica 101 --t-end 10
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lj075_common as C  # noqa: E402
import lj075_coupled as LC  # noqa: E402
import toy_configs as tc  # noqa: E402
from test_lanczos_fdt import proj  # noqa: E402

CONFIG = "lj_rho0.75_kT1.0_costopt"
TIGHT = {"A": dict(xi=0.85, s=4.4, eta=0.6, p=7), "B": dict(xi=1.0, s=4.2, eta=0.6, p=7)}
OUT = C.RAW / "operator_sensitivity"
DT = 0.005


def run(law, rep, t_end):
    out = OUT / f"{law}_rep{rep}_T{t_end:g}.npz"
    if out.exists():
        print(f"{out.name} exists")
        return
    r = tc.resolve(CONFIG, law, 256, "lj", dt=DT, allow_unverified=True)
    r2 = dict(r, pppm=dict(TIGHT[law]))
    st = np.load(C.RAW / "states" / f"dtinit_s{rep}.npz")
    q0, p0 = st["q"], st["p"]
    lv = [LC.Level(r, DT, q0, p0, monitor_every=0), LC.Level(r2, DT, q0, p0, monitor_every=0)]
    rng = np.random.default_rng([20261009, 7, ord(law), rep])
    n = int(round(t_end / DT))
    S = np.zeros((2, n + 1, 4))
    Q = np.zeros((2, n + 1, 256, 3), np.float32)
    cpu0, wall0 = C.cpu_seconds(), time.perf_counter()
    for k, l in enumerate(lv):
        S[k, 0] = (0, C.kinetic_temperature(l.p, 1.0), l.U / 256, C.pressure(l.q, l.p, l.L, 1.0))
        Q[k, 0] = l.q
    dq, dv = np.zeros(n + 1), np.zeros(n + 1)
    for s in range(1, n + 1):
        xi = proj(rng.standard_normal(768))
        for k, l in enumerate(lv):
            l.step(xi=xi)
            S[k, s] = (s * DT, C.kinetic_temperature(l.p, 1.0), l.U / 256, C.pressure(l.q, l.p, l.L, 1.0))
            Q[k, s] = l.q
        dq[s] = np.sqrt(np.mean((lv[0].q - lv[1].q) ** 2))
        dv[s] = np.sqrt(np.mean((lv[0].p - lv[1].p) ** 2))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(out, S=S, Q=Q, dq=dq, dv=dv)
    meta = dict(law=law, replica=rep, t_end=t_end, dt=DT, production_pppm=r["pppm"], tight_pppm=TIGHT[law],
                ranks=(r["rank_noise"], r["rank_damp"]), init=f"dtinit_s{rep}", seed=[20261009, 7, ord(law), rep],
                cpu_s=C.cpu_seconds() - cpu0, wall_s=time.perf_counter() - wall0, provenance=C.provenance())
    C.write_json(out.with_suffix(".json"), meta)
    print(f"[{law} {rep}] done; dq(t_end) {dq[-1]:.2e} dv(t_end) {dv[-1]:.2e}", flush=True)


def analyze():
    res = dict(provenance=C.provenance())
    for law in ("A", "B"):
        files = sorted(OUT.glob(f"{law}_rep*_T*.npz"))
        if not files:
            continue
        rows, paths = [], []
        for f in files:
            d = np.load(f)
            S, Q = d["S"], d["Q"].astype(float)
            keep = S[0, :, 0] >= 1.0
            row = {}
            for j, name in ((1, "T_kin"), (2, "U_per_N"), (3, "pressure")):
                a, b = S[0, keep, j].mean(), S[1, keep, j].mean()
                row[name] = (a - b) / (abs(b) if name != "pressure" else 1.0)
            for tt in (0.5, 2.0, 5.0):
                k = int(round(tt / DT))
                m = [C.msd(Q[i][int(round(1.0 / DT))::10], k // 10 + 1)[-1] for i in range(2)]
                row[f"msd_{tt:g}"] = (m[0] - m[1]) / m[1]
            rows.append(row)
            paths.append(dict(dq={f"t={x:g}": float(d["dq"][int(round(x / DT))]) for x in (0.1, 1, 5, 10)
                                  if int(round(x / DT)) < len(d["dq"])},
                              dv={f"t={x:g}": float(d["dv"][int(round(x / DT))]) for x in (0.1, 1, 5, 10)
                                  if int(round(x / DT)) < len(d["dv"])}))
        res[law] = dict(n=len(rows), per_replica=rows, paths=paths,
                        max_abs={k: float(max(abs(r[k]) for r in rows)) for k in rows[0]},
                        note="production minus tight operator; relative except pressure (absolute)")
        print(f"[{law}] max |diff| {res[law]['max_abs']}; paths {paths}", flush=True)
    C.write_json(C.OUT / "operator_sensitivity.json", res)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--law")
    ap.add_argument("--replica", type=int)
    ap.add_argument("--t-end", type=float, default=10.0)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze()
    else:
        run(args.law, args.replica, args.t_end)


if __name__ == "__main__":
    main()
