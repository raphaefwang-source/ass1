#!/usr/bin/env python3
"""
Canonical starting / test structures at the lj075 state point (rho 0.75, kT 1.0, N = 256, switched LJ).

Scalar-friction Langevin BAOAB (exact OU step, friction 1.0, dt 0.005) with the conservative forces only. The
positions sample exp(-U/kT) whatever the friction model, so these are valid canonical structures for the static
operator / Lanczos verification and as common initial states of the dt study. They are NOT claimed to be
equilibrated under the friction dynamics; lj075_equilibration.py studies burn-in separately.

Starts: fcc (4x4x4 cells, jitter 0.05) and RSA (minimum distance 0.9), seeds as listed; energy traces are stored.
Each state is written as raw/states/<name>.npz with q, p, L, kT, N, generation record (git-ignored), and a summary
lj075_results/prepared_states.json.

    python3 lj075_prepare_states.py --t 40 --names fcc_s1 fcc_s2 fcc_s3 rsa_s4 rsa_s5 rsa_s6
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import lj075_common as C  # noqa: E402

STATES = C.RAW / "states"
GEN = dict(method="scalar Langevin BAOAB, conservative LJ forces only", gamma=1.0, dt=0.005, fcc_jitter=0.05,
           rsa_dmin=0.9, rng="default_rng([20261009, seed])")


def prepare(name, t, extra_snapshots=0, snap_gap=5.0):
    kind, seed = name.split("_s")
    seed = int(seed)
    N, kT, m = C.STATE["N"], C.STATE["kT"], C.STATE["mass"]
    L = C.box_length(N, C.STATE["rho"])
    rng = np.random.default_rng([20261009, seed])
    if kind == "fcc":
        q = C.fcc_positions(N, L, GEN["fcc_jitter"], rng)
    elif kind in ("rsa", "dtinit", "eqinit"):                  # dtinit / eqinit: RSA starts for the dt / burn-in studies
        q = C.rsa_positions(N, L, GEN["rsa_dmin"], rng)
    else:
        raise ValueError(f"unknown start kind {kind}")
    p = C.maxwell(rng, N, kT, m)
    t0, c0 = time.perf_counter(), time.process_time()
    q, p, tr = C.langevin_prepare(q, p, L, kT, m, t, GEN["dt"], GEN["gamma"], rng)
    rec = dict(name=name, start=kind, seed=seed, t=t, wall_s=time.perf_counter() - t0, cpu_s=time.process_time() - c0,
               U_per_N_last_quarter=float(tr[tr[:, 0] >= 0.75 * t, 1].mean()),
               T_last_quarter=float(tr[tr[:, 0] >= 0.75 * t, 2].mean()), generation=GEN)
    STATES.mkdir(parents=True, exist_ok=True)
    np.savez(STATES / f"{name}.npz", q=q, p=p, L=L, kT=kT, N=N, trace=tr,
             generation=np.array(json.dumps(rec)))
    out = [rec]
    for k in range(extra_snapshots):
        q, p, tr2 = C.langevin_prepare(q, p, L, kT, m, snap_gap, GEN["dt"], GEN["gamma"], rng)
        r2 = dict(rec, name=f"{name}_x{k + 1}", t=t + (k + 1) * snap_gap, parent=name)
        np.savez(STATES / f"{name}_x{k + 1}.npz", q=q, p=p, L=L, kT=kT, N=N, trace=tr2,
                 generation=np.array(json.dumps(r2)))
        out.append(r2)
    return out, tr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--t", type=float, default=40.0)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--extra-snapshots", type=int, default=0)
    args = ap.parse_args()
    summ_f = C.OUT / "prepared_states.json"
    summ = json.loads(summ_f.read_text()) if summ_f.exists() else {}
    for name in args.names:
        recs, tr = prepare(name, args.t, args.extra_snapshots)
        for r in recs:
            summ[r["name"]] = r
        print(f"{name}: U/N last quarter {recs[0]['U_per_N_last_quarter']:.4f}, T {recs[0]['T_last_quarter']:.4f}, "
              f"{recs[0]['cpu_s']:.0f} cpu s; U/N at t = 1, 2, 5, 10, 20: "
              + ", ".join(f"{np.interp(x, tr[:, 0], tr[:, 1]):.3f}" for x in (1, 2, 5, 10, 20)), flush=True)
        summ["_provenance"] = C.provenance()
        C.write_json(summ_f, summ)


if __name__ == "__main__":
    main()
