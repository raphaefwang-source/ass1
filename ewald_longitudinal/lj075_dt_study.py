#!/usr/bin/env python3
"""
Time-step study for the lj075 state point (rho 0.75, kT 1.0, N = 256, laws A and B), with cross-dt coupled noise.

Design (pre-registered in lj075_results/dt_criteria.json before any production run of this study):
  * levels dt = 0.01, 0.005, 0.0025 and (subset of replicas) 0.00125, integrated in lock-step from the SAME (q0, p0)
    to the SAME physical end time, with the nested exact-OU noise coupling of lj075_coupled.py (the finest level
    draws the Gaussians; every coarser level builds its input from two inputs of the next finer level with its own
    Gamma). The same seed alone would not couple the levels.
  * operator and Lanczos errors controlled first: PPPM set and ranks of lj075_verify.py (all four dt and the coupling
    maps), with an on-line Lanczos error estimate on the actual inputs every MONITOR steps of every level.
  * replicas: independent canonical starting states (lj075_prepare_states.py, Langevin with conservative forces);
    a relaxation window [0, T_DISCARD] is dropped from every level before averaging.
  * recorded per step: T_kin, U/N, pressure (virial), |P_total|, rmin; frames (unwrapped q, v; float32) every step of
    every level (raw, git-ignored).

Stages:
    python3 lj075_dt_study.py criteria                       # write the pre-registered criteria (once)
    python3 lj075_dt_study.py run --law A --replica 1 --levels 4 --t-end 10
    python3 lj075_dt_study.py analyze
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

CONFIG = "lj_rho0.75_kT1.0_costopt"
DT_ALL = (0.00125, 0.0025, 0.005, 0.01)
T_DISCARD = 1.0
MONITOR = 200
RAWDIR = C.RAW / "dt_study"
CRIT = C.OUT / "dt_criteria.json"
SEED_ENTROPY = 20261009


def criteria():
    if CRIT.exists():
        print(f"{CRIT} exists (pre-registered {json.loads(CRIT.read_text())['registered']}); not overwritten")
        return
    crit = dict(
        registered=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        reference="the finest level present in every replica used for a comparison (0.00125 where run, else 0.0025)",
        averaging=f"per replica: time average over t in [{T_DISCARD}, t_end] of each level; differences are paired by "
                  "replica (same q0, p0, coupled noise); 95% Student-t CI over replicas unless stated",
        tolerance=dict(
            T_kin="relative 1% of the reference mean",
            U_per_N="relative 1% of |reference mean|",
            pressure="absolute 0.01 * rho * kT = 0.0075 (relative 1% is ill-conditioned if P is small); relative "
                     "difference also reported",
            rdf="E_g = ||mean_dg||_w / ||g_ref - 1||_w with w = 4 pi rho r^2 dr on [0.5, L/2); 1% on the 95% "
                "bootstrap upper bound (replica resampling). Noise inflates E_g, so this is conservative",
            rdf_first_peak="height and position: relative 1%",
            vacf="E_V = max over the common grid t_j = j * 0.01, t_j <= t_max (A 2.0, B 0.5) of "
                 "|mean dC(t_j)| / C_ref(0) (absolute, normalised by the reference C(0): the VACF crosses zero); 1% "
                 "on the 95% bootstrap upper bound",
            msd="relative 1% at t = 0.1, 0.5, 1, 2, 5",
            D="long-time D is NOT judged from these short runs; it is studied separately (lj075_equilibration.py); "
              "the dt-study D estimates are reported for information only"),
        verdict=dict(PASS="the whole 95% interval lies inside the tolerance band",
                     FAIL="the whole 95% interval lies outside the tolerance band",
                     INCONCLUSIVE="otherwise (an interval containing 0 is NOT taken as equivalence)"),
        decision="first compare the finest two levels; if every criterion PASSES, the finest level is an adequate "
                 "reference. Then the recommended dt is the largest level for which EVERY criterion PASSES against the "
                 "reference. INCONCLUSIVE results lead to more replicas, not to a relaxed tolerance.",
        operator_control="PPPM set and Lanczos ranks verified by lj075_verify.py for all four dt and the coupling "
                         "maps (strict rule budget/10); on-line Lanczos estimates on actual inputs every 200 steps")
    C.write_json(CRIT, crit)
    print(json.dumps(crit, indent=1))


def resolved(law, dt, ranks):
    r = tc.resolve(CONFIG, law, 256, "lj", dt=dt, allow_unverified=True,
                   rank_override=(ranks[0], ranks[1]) if ranks else None)
    return r


def run(args):
    law, rep = args.law, args.replica
    levels = DT_ALL[4 - args.levels:]
    out = RAWDIR / f"{law}_rep{rep:02d}_L{args.levels}_T{args.t_end:g}.npz"
    meta_f = out.with_suffix(".json")
    if meta_f.exists() and json.loads(meta_f.read_text()).get("complete"):
        print(f"{out.name} complete; nothing to do")
        return
    st = np.load(C.RAW / "states" / f"{args.init_prefix}{rep}.npz")
    L = C.box_length(256, 0.75)
    if abs(float(st["L"]) - L) > 1e-9 or abs(float(st["kT"]) - 1.0) > 1e-12:
        raise ValueError("initial state is not at the lj075 state point")
    q0, p0 = st["q"].astype(float), st["p"].astype(float)
    ranks = tuple(args.ranks) if args.ranks else None
    r = resolved(law, levels[0], ranks)
    if ranks:
        r["rank_noise"], r["rank_damp"] = ranks[0], ranks[1]
    rc = ranks[2] if ranks else None
    N = 256
    nsave = {k: int(round(args.t_end / dt)) + 1 for k, dt in enumerate(levels)}
    Qs = {k: np.empty((n, N, 3), np.float32) for k, n in nsave.items()}
    Vs = {k: np.empty((n, N, 3), np.float32) for k, n in nsave.items()}
    scal = {k: np.empty((n, 6)) for k, n in nsave.items()}
    idx = {k: 0 for k in nsave}

    def record(k, lv):
        i = idx[k]
        Qs[k][i], Vs[k][i] = lv.q, lv.p / lv.m
        scal[k][i] = (lv.nstep * lv.dt, C.kinetic_temperature(lv.p, lv.m), lv.U / N,
                      C.pressure(lv.q, lv.p, lv.L, lv.m), np.linalg.norm(lv.p.sum(0)), lv.rmin)
        idx[k] = i + 1

    cpu0, wall0 = C.cpu_seconds(), time.perf_counter()
    res = LC.run_chain(r, list(levels), q0, p0, args.t_end, seed=[SEED_ENTROPY, ord(law), rep], on_step=record,
                       monitor_every=MONITOR, rank_couple=rc, log=lambda s: print(f"[{law} rep {rep}] {s}", flush=True))
    for k in nsave:
        if idx[k] != nsave[k]:
            raise RuntimeError(f"level {k}: recorded {idx[k]} of {nsave[k]} frames")
    RAWDIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, dt in enumerate(levels):
        arrays[f"Q_{dt}"], arrays[f"V_{dt}"], arrays[f"S_{dt}"] = Qs[k], Vs[k], scal[k]
        arrays[f"mon_{dt}"] = np.array(res["monitor"][k], float).reshape(-1, 4)
    np.savez(out, **arrays)
    meta = dict(law=law, replica=rep, levels=list(levels), t_end=args.t_end, init_state=f"{args.init_prefix}{rep}",
                init_record=json.loads(str(st["generation"])), seed=[SEED_ENTROPY, ord(law), rep],
                scal_cols=["t", "T_kin", "U_per_N", "pressure", "P_total_norm", "rmin"],
                resolved=r, rank_couple=rc or r["rank_noise"], monitor_every=MONITOR,
                wall_per_level=res["wall_per_level"], steps=res["steps"], lam=res["lam"],
                cpu_s=C.cpu_seconds() - cpu0, wall_s=time.perf_counter() - wall0, provenance=C.provenance(),
                complete=True)
    C.write_json(meta_f, meta)
    print(f"[{law} rep {rep}] done: wall per level {np.round(res['wall_per_level'], 1)} s, cpu {meta['cpu_s']:.0f} s; "
          f"lam range {res['lam']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["criteria", "run", "analyze"])
    ap.add_argument("--law")
    ap.add_argument("--replica", type=int)
    ap.add_argument("--levels", type=int, default=3, choices=[2, 3, 4])
    ap.add_argument("--t-end", type=float, default=10.0)
    ap.add_argument("--init-prefix", default="dtinit_s")
    ap.add_argument("--ranks", type=int, nargs=3, metavar=("NOISE", "DAMP", "COUPLE"))
    args = ap.parse_args()
    if args.stage == "criteria":
        criteria()
    elif args.stage == "run":
        if not CRIT.exists():
            raise SystemExit("write the pre-registered criteria first (stage criteria)")
        run(args)
    else:
        import lj075_dt_analysis
        lj075_dt_analysis.main()


if __name__ == "__main__":
    main()
