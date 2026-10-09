#!/usr/bin/env python3
"""
Cost of complete production steps (LJ force, Gamma_h build, damping and noise Lanczos) for candidate PPPM sets at the
lj075 state point, single thread, dt = 0.005, with the Lanczos ranks of each candidate's own 42-state verification
(verify_representative_<law>[_xi_s_eta_p].json; candidates without a passed verification are timed but flagged), plus
rank variants (RANK_VARIANTS: e.g. the noise rank 14 that the strict rule requires on the visited A states).

Timing: process CPU time per step; the candidates are interleaved in rounds (each round: STEPS steps per candidate
from the same canonical state, fresh Level) so that background load affects all candidates alike; the median over
rounds is reported with the min-max range.
Writes lj075_results/pppm_cost.json.

    python3 lj075_pppm_cost.py [--rounds 6] [--steps 40] [--laws A]
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
CANDS = {"A": [None, (0.85, 4.1, 0.6, 7), (0.85, 4.4, 0.6, 7)],
         "B": [None, (1.0, 4.2, 0.6, 7)]}
# (pppm candidate or None, (rank_noise, rank_damp), verification file whose chosen ranks justify them)
RANK_VARIANTS = {"A": [(None, (14, 5), "verify_visited_eq_A.json")]}
STATES = ("dtinit_s109", "dtinit_s110", "dtinit_s111")


def verification(law, cand):
    f = C.OUT / f"verify_representative_{law}{'' if cand is None else '_' + '_'.join(map(str, map(float, cand)))}.json"
    if not f.exists():
        return f.name, None
    v = json.loads(f.read_text())
    return f.name, v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--laws", nargs="+", default=["A"])
    args = ap.parse_args()
    out = dict(config=CONFIG, dt=0.005, rounds=args.rounds, steps_per_round=args.steps, states=STATES,
               provenance=C.provenance(), laws={})
    for law in args.laws:
        cands = CANDS[law]
        base = tc.resolve(CONFIG, law, 256, "lj", dt=0.005, allow_unverified=True)
        setups = []
        for cand in cands:
            fname, v = verification(law, cand)
            pp = dict(base["pppm"]) if cand is None else dict(xi=cand[0], s=cand[1], eta=cand[2], p=int(cand[3]))
            if v is not None:
                rn, rd = v["chosen_ranks"]["noise"], v["chosen_ranks"]["damp"]
                status = dict(file=fname, passed=bool(v["passed"]), op_err_max=max(s["op_err"] for s in
                                                                                    v["per_state"].values()),
                              op_err_vs_k_nonzero_max=v["operator_vs_k_nonzero_part_max"], M=v["mesh"]["M"])
            else:
                rn, rd = base["rank_noise"], base["rank_damp"]
                status = dict(file=None, passed=False, note="no 42-state verification; production ranks used")
            r = dict(base, pppm=pp, rank_noise=rn, rank_damp=rd)
            setups.append((("production" if cand is None else "xi{} s{} eta{} p{}".format(*cand)), r, status))
        for cand, (rn, rd), vfile in RANK_VARIANTS.get(law, []):
            v = json.loads((C.OUT / vfile).read_text())
            pp = dict(base["pppm"]) if cand is None else dict(xi=cand[0], s=cand[1], eta=cand[2], p=int(cand[3]))
            status = dict(file=vfile, passed=bool(v["passed"]), op_err_max=max(x["op_err"] for x in v["per_state"].values()),
                          op_err_vs_k_nonzero_max=v["operator_vs_k_nonzero_part_max"], M=v["mesh"]["M"],
                          chosen_ranks=v["chosen_ranks"])
            setups.append((f"{'production' if cand is None else 'xi{} s{} eta{} p{}'.format(*cand)} ranks {rn}/{rd}",
                           dict(base, pppm=pp, rank_noise=rn, rank_damp=rd), status))
        times = {name: [] for name, _, _ in setups}
        for rd_ in range(args.rounds):
            st = np.load(C.RAW / "states" / f"{STATES[rd_ % len(STATES)]}.npz")
            order = setups if rd_ % 2 == 0 else setups[::-1]
            for name, r, _ in order:
                lv = LC.Level(r, 0.005, st["q"], st["p"])
                rng = np.random.default_rng(rd_)
                lv.step(xi=LC.proj(rng.standard_normal(768)))          # warm-up (tree, FFT plans)
                c0 = time.process_time()
                for _ in range(args.steps):
                    lv.step(xi=LC.proj(rng.standard_normal(768)))
                times[name].append((time.process_time() - c0) / args.steps)
            print(f"[{law}] round {rd_}: " + ", ".join(f"{n} {times[n][-1] * 1e3:.1f} ms" for n, _, _ in setups),
                  flush=True)
        prod = np.median(times["production"])
        out["laws"][law] = {name: dict(pppm=r["pppm"], ranks=dict(noise=r["rank_noise"], damp=r["rank_damp"]),
                                       verification=status, cpu_ms_per_step_median=float(np.median(times[name]) * 1e3),
                                       cpu_ms_per_step_range=[float(min(times[name]) * 1e3),
                                                              float(max(times[name]) * 1e3)],
                                       relative_to_production=float(np.median(times[name]) / prod),
                                       core_hours_per_time_unit=float(np.median(times[name]) / 0.005 / 3600))
                            for name, r, status in setups}
    C.write_json(C.OUT / "pppm_cost.json", out)


if __name__ == "__main__":
    main()
