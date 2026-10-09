#!/usr/bin/env python3
"""
Save-interval study for the lj075 state point: offline thinning of every-step trajectories of the production runner
(lj075_equilibration.py runs, frames every step), after the burn-in.

For save interval Delta = s * dt (s = 1, 2, 4, 10, 20):
  quadrature-only  every-step, all-origin VACF / MSD read at lags k * Delta; isolates the time-grid effect (GK
                   trapezoid on the coarser grid; the MSD values themselves are exact at those lags)
  production-like  origins AND lags every Delta (what a run saving every Delta can compute); adds origin-count noise
Metrics vs the every-step reference of the same trajectory (paired; 95% t-CI over runs):
  VACF: tau_1/2 (linear interpolation), number of stored lags before C/C0 = 1/2;
  GK:   D_GK(T) = (1/3) int_0^T C dt by trapezoid, T = 0.1, 0.5, 2, 8;
  MSD:  MSD(t) at t = 0.1, 0.5, 2, 8 and D_MSD from the fit window [2, 8].
Rule (the project's existing save-interval rule of vacf_sampling_check.py, applied to EVERY run): GK quadrature bias
  <= 0.2% at every T, >= 5 stored lags before 1/2, tau_1/2 within 2%; extended here by: production-like MSD(t) and
  D_MSD[2, 8] relative difference with the whole 95% t-CI over runs within +-0.2%.
Reference: the every-step trapezoid GK. For this integrator it equals the chain's own long-time MSD diffusion (the
  force term of the position update telescopes), so it has no quadrature error relative to the simulated dynamics;
  its distance from the dt -> 0 limit is a time-step effect (estimated by the Richardson offset (D_s1 - D_s2)/3 and
  judged in the dt study), not a save-interval effect. If even stride 1 fails the lag rule, the VACF is NOT resolved
  at this dt and this is stated.
Writes lj075_results/sampling.json and sampling_interval.png. Does not touch vacf_sampling_results/.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lj075_common as C  # noqa: E402

RUNS = C.RAW / "runs"
STRIDES = (1, 2, 4, 10, 20)
GK_T = (0.1, 0.5, 2.0, 8.0)
MSD_T = (0.1, 0.5, 2.0, 8.0)
FIT = (2.0, 8.0)
RULE = dict(gk_bias=0.002, lags_before_half=5, tau_half_rel=0.02, msd_rel=0.002)


def tau_half(t, c):
    c = c / c[0]
    k = np.argmax(c < 0.5)
    if k == 0:
        return np.nan, 0
    return float(t[k - 1] + (0.5 - c[k - 1]) / (c[k] - c[k - 1]) * (t[k] - t[k - 1])), int(k)


def gk(t, c, T):
    sel = t <= T + 1e-12
    return float(np.trapezoid(c[sel], t[sel]) / 3)


def one_run(run, burn):
    import toy_run
    d = toy_run.load_run(run)
    dt = d["config"]["resolved"]["dt"]
    k0 = int(round(burn / dt))
    fs = d["frame_step"]
    if not np.all(np.diff(fs) == 1):
        raise ValueError(f"{run}: frames are not every step")
    Q, V = d["Q"][k0:], d["V"][k0:]
    nlag = int(round(max(GK_T) / dt)) + 1
    c_all = C.vacf(V, nlag)
    t_all = np.arange(nlag) * dt
    m_lag = int(round(max(MSD_T) / dt)) + 1
    msd_all = C.msd(Q, m_lag)
    tm_all = np.arange(m_lag) * dt
    out = {}
    for s in STRIDES:
        rec = {}
        for variant in ("quadrature", "production"):
            if variant == "quadrature":
                c, tt = c_all[::s], t_all[::s]
                msd, tm = msd_all[::s], tm_all[::s]
            else:
                c = C.vacf(V[::s], nlag // s + 1)
                tt = np.arange(len(c)) * s * dt
                msd = C.msd(Q[::s], m_lag // s + 1)
                tm = np.arange(len(msd)) * s * dt
            th, nb = tau_half(tt, c)
            sel = (tm >= FIT[0]) & (tm <= FIT[1])
            rec[variant] = dict(tau_half=th, lags_before_half=nb,
                                gk={str(T): gk(tt, c, T) for T in GK_T},
                                msd={str(x): float(np.interp(x, tm, msd)) for x in MSD_T},
                                D_msd=float(np.polyfit(tm[sel], msd[sel], 1)[0] / 6))
        out[str(s)] = rec
    return dt, out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--laws", nargs="+", default=["A"], help="laws to analyse (default A: the lj075 campaign "
                    "continued with law A only; B was stopped)")
    args = ap.parse_args()
    eq = json.loads((C.OUT / "equilibration.json").read_text())
    res = dict(rule=RULE, strides=STRIDES, provenance=C.provenance())
    for law in args.laws:
        if law not in eq:
            continue
        burn = eq[law]["burn_in"]["estimate"]
        runs = [RUNS / n for n in eq[law]["runs"]]
        per = {}
        dt = None
        for r in runs:
            dt, per[r.name] = one_run(r, burn)
        summ = {}
        for s in STRIDES:
            row = {}
            for variant in ("quadrature", "production"):
                rel = lambda f: [f(p[str(s)][variant]) / f(p["1"]["quadrature"]) - 1 for p in per.values()]  # noqa: E731
                v = dict(tau_half=C.t_ci(rel(lambda x: x["tau_half"])),
                         lags_before_half=int(min(p[str(s)][variant]["lags_before_half"] for p in per.values())),
                         gk={T: C.t_ci(rel(lambda x, T=T: x["gk"][T])) for T in map(str, GK_T)},
                         msd={x: C.t_ci(rel(lambda y, x=x: y["msd"][x])) for x in map(str, MSD_T)},
                         D_msd=C.t_ci(rel(lambda y: y["D_msd"])))
                row[variant] = v
            q, pr = row["quadrature"], row["production"]
            per_run_gk = [abs(p[str(s)]["quadrature"]["gk"][T] / p["1"]["quadrature"]["gk"][T] - 1)
                          for p in per.values() for T in map(str, GK_T)]
            per_run_tau = [abs(p[str(s)]["quadrature"]["tau_half"] / p["1"]["quadrature"]["tau_half"] - 1)
                           for p in per.values()]
            ok_vacf = (max(per_run_gk) <= RULE["gk_bias"] and q["lags_before_half"] >= RULE["lags_before_half"]
                       and max(per_run_tau) <= RULE["tau_half_rel"])
            ok_msd = (all(-RULE["msd_rel"] <= x[1] and x[2] <= RULE["msd_rel"] for x in pr["msd"].values())
                      and -RULE["msd_rel"] <= pr["D_msd"][1] and pr["D_msd"][2] <= RULE["msd_rel"])
            row["passes_vacf_gk_rule"] = bool(ok_vacf)
            row["passes_msd_rule"] = bool(ok_msd)
            row["passes_rule"] = bool(ok_vacf and ok_msd)
            row["max_run_gk_bias"] = float(max(per_run_gk))
            row["save_interval"] = s * dt
            summ[str(s)] = row
        ref = {k: C.t_ci([p["1"]["quadrature"][k] for p in per.values()]) for k in ("tau_half",)}
        ref["gk"] = {T: C.t_ci([p["1"]["quadrature"]["gk"][T] for p in per.values()]) for T in map(str, GK_T)}
        rich = {T: C.t_ci([(p["1"]["quadrature"]["gk"][T] - p["2"]["quadrature"]["gk"][T]) / 3
                           / p["1"]["quadrature"]["gk"][T] for p in per.values()]) for T in map(str, GK_T)}
        res[law] = dict(dt=dt, burn_in=burn, runs=[r.name for r in runs], every_step_reference=ref, by_stride=summ,
                        largest_passing=max([s for s in STRIDES if summ[str(s)]["passes_rule"]], default=None),
                        largest_passing_msd_only=max([s for s in STRIDES if summ[str(s)]["passes_msd_rule"]], default=None),
                        vacf_resolved_at_every_step=bool(summ["1"]["quadrature"]["lags_before_half"]
                                                         >= RULE["lags_before_half"]),
                        richardson_offset_rel=rich)
        print(f"[{law}] tau_half {ref['tau_half']}; passing strides "
              f"{[s for s in STRIDES if summ[str(s)]['passes_rule']]}", flush=True)
        for s in STRIDES:
            q = summ[str(s)]["quadrature"]
            print(f"   stride {s}: GK bias {[round(x[0] * 100, 3) for x in q['gk'].values()]} %, lags before 1/2 "
                  f"{q['lags_before_half']}, tau_half {q['tau_half'][0] * 100:+.2f} %", flush=True)
    C.write_json(C.OUT / "sampling.json", res)


if __name__ == "__main__":
    main()
