#!/usr/bin/env python3
"""
Long-time diffusion and its time-step dependence at the lj075 state point (separate from the short dt study, as
required: short-time agreement is not used as evidence for the long-time D).

Data: coupled chains of lj075_dt_study.py run with --frame-every 0.01 and t_end 100 (law A: dt 0.005 and 0.01; law B:
0.0025, 0.005 and 0.01), started from independent canonical states (dtinit_s113-115), noise coupled across dt.
For every level: MSD (all origins, COM removed) after T0 = 10 (burn-in margin; the starts are canonical), from frames
every 0.01; D_MSD on windows [1,2], [2,4], [4,8], [8,16], [16,32] by linear fits.
Plateau (per level): paired per-replica relative change between adjacent windows, whole 95% t-CI within +-2%;
  long-time D is reported only for a window where a plateau holds, else "not determined".
dt dependence: paired per-replica relative difference D_dt / D_ref - 1 at each window (ref = finest level), with
  95% t-CI, compared with the free-particle prior x coth x - 1, x = lambda dt / (2m), lambda in the measured
  spectrum (A [4.6, 12.1], B [63.7, 91.6]).
Writes lj075_results/longD.json and longD.png.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lj075_common as C  # noqa: E402

RAWDIR = C.RAW / "dt_study"
T0 = 10.0
WINDOWS = ((1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0))
LAM = {"A": (4.6, 12.1), "B": (63.7, 91.6)}


def prior(law, dt):
    out = []
    for lam in LAM[law]:
        x = lam * dt / 2
        out.append(x / np.tanh(x) - 1)
    return out


def load(law):
    reps = []
    for f in sorted(RAWDIR.glob(f"{law}_rep*_L*_T*_F0.01.json")):
        m = json.loads(f.read_text())
        if m.get("complete"):
            reps.append((m, f.with_suffix(".npz")))
    return reps


def dmsd(Q, fdt):
    k0 = int(round(T0 / fdt))
    Qp = Q[k0:].astype(float)
    nl = min(len(Qp) - 1, int(round(max(w[1] for w in WINDOWS) / fdt)) + 1)
    m = C.msd(Qp, nl)
    tm = np.arange(len(m)) * fdt
    out = {}
    for a, b in WINDOWS:
        sel = (tm >= a - 1e-9) & (tm <= b + 1e-9)
        if b <= tm[-1] + 1e-9 and b <= len(Qp) * fdt / 3:
            out[f"{a:g}-{b:g}"] = float(np.polyfit(tm[sel], m[sel], 1)[0] / 6)
    return out, (tm[::20].tolist(), m[::20].tolist())


def main():
    res = dict(provenance=C.provenance(), T0=T0, windows=[f"{a:g}-{b:g}" for a, b in WINDOWS])
    for law in ("A", "B"):
        reps = load(law)
        if len(reps) < 2:
            continue
        levels = reps[0][0]["levels"]
        per = {}
        curves = {}
        cpu = 0.0
        for m, f in reps:
            if m["levels"] != levels:
                raise ValueError("mixed level sets")
            fe = m["frame_every"]
            with np.load(f) as z:
                per[m["replica"]] = {str(dt): dmsd(z[f"Q_{dt}"], fe)[0] for dt in levels}
                curves[m["replica"]] = {str(dt): dmsd(z[f"Q_{dt}"], fe)[1] for dt in levels}
                per[m["replica"]]["_T"] = {str(dt): float(z[f"S_{dt}"][int(round(T0 / dt)):, 1].mean()) for dt in levels}
            cpu += m["cpu_s"]
        keys = [k for k in res["windows"] if all(k in per[r][str(levels[0])] for r in per)]
        lev = {}
        for dt in levels:
            D = {k: C.t_ci([per[r][str(dt)][k] for r in per]) for k in keys}
            pl = []
            for a, b in zip(keys[:-1], keys[1:]):
                rows = [(per[r][str(dt)][b] - per[r][str(dt)][a]) / per[r][str(dt)][a] for r in per]
                mm, lo, hi = C.t_ci(rows)
                pl.append(dict(windows=f"{a} -> {b}", rel_change=mm, ci95=[lo, hi], plateau=bool(-0.02 <= lo and hi <= 0.02)))
            det = [p for p in pl if p["plateau"]]
            lev[str(dt)] = dict(D_MSD=D, plateau=pl, long_time_D_determined=bool(det),
                                D_long=(D[det[0]["windows"].split(" -> ")[1]] if det else None),
                                T_kin=C.t_ci([per[r]["_T"][str(dt)] for r in per]))
        ref = str(levels[0])
        cmp_ = {}
        for dt in levels[1:]:
            rows = {k: C.t_ci([per[r][str(dt)][k] / per[r][ref][k] - 1 for r in per]) for k in keys}
            cmp_[f"{dt}_vs_{ref}"] = dict(rel_diff=rows, prior_xcothx_minus_1_vs_continuum=prior(law, dt),
                                          prior_ref=prior(law, float(ref)))
        res[law] = dict(levels=levels, n_replicas=len(per), replicas=sorted(per), cpu_core_hours=cpu / 3600,
                        per_level=lev, dt_comparison=cmp_, per_replica=per, _curves=curves)
        print(f"[{law}] levels {levels}, n {len(per)}:", flush=True)
        for dt in levels:
            print(f"   dt {dt}: D_MSD {[(k, round(v[0], 6)) for k, v in lev[str(dt)]['D_MSD'].items()]}; plateau "
                  f"{[(p['windows'], round(p['rel_change'], 4), p['plateau']) for p in lev[str(dt)]['plateau']]}",
                  flush=True)
        for k, v in cmp_.items():
            print(f"   {k}: {[(w, round(x[0], 4), round(x[1], 4), round(x[2], 4)) for w, x in v['rel_diff'].items()]}; "
                  f"prior {np.round(v['prior_xcothx_minus_1_vs_continuum'], 4)}", flush=True)
    C.write_json(C.OUT / "longD.json", res)


if __name__ == "__main__":
    main()
