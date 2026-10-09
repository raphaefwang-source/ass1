#!/usr/bin/env python3
"""
Analysis of the lj075 time-step study (lj075_dt_study.py), following the pre-registered criteria in
lj075_results/dt_criteria.json. Writes lj075_results/dt_study.json and dt_study_<law>.png.

All comparisons are PAIRED by replica (same q0, p0, coupled noise) against the reference level (the finest level
common to the replicas used). Scalars: Student-t 95% CI over replicas of the per-replica difference of time
averages over [T_DISCARD, t_end]. Function-valued observables: the global error measure of the mean difference with a
95% bootstrap upper bound (replicas resampled with replacement, 4000 resamples).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lj075_common as C  # noqa: E402

RAWDIR = C.RAW / "dt_study"
T_DISCARD = 1.0
GRID = 0.01                                   # common time grid (the largest dt)
VACF_TMAX = {"A": 2.0, "B": 0.5}
MSD_TIMES = (0.1, 0.5, 1.0, 2.0, 5.0)
RDF_EDGES = np.arange(0.5, 3.48, 0.02)
RDF_STRIDE_T = 0.02
N_BOOT = 4000
TOL = 0.01


def load(law):
    reps = []
    for meta_f in sorted(RAWDIR.glob(f"{law}_rep*_L*_T*.json")):
        meta = json.loads(meta_f.read_text())
        if not meta.get("complete"):
            continue
        reps.append((meta, meta_f.with_suffix(".npz")))
    return reps


N_WIN = 3                                     # stationarity check: equal windows of [T_DISCARD, t_end]


def per_level(npz, dt, law, L):
    """Per-replica, per-level observables (full averaging window and N_WIN sub-windows for scalars and RDF peak)."""
    S = npz[f"S_{dt}"]
    t = S[:, 0]
    keep = t >= T_DISCARD - 1e-12
    out = dict(T_kin=S[keep, 1].mean(), U_per_N=S[keep, 2].mean(), pressure=S[keep, 3].mean(),
               max_P_total=float(S[:, 4].max()), rmin=float(S[:, 5].min()))
    edges = np.linspace(T_DISCARD, t[-1], N_WIN + 1)
    win = [(t >= edges[w] - 1e-12) & (t < edges[w + 1] - 1e-12 if w < N_WIN - 1 else t <= edges[-1] + 1e-12)
           for w in range(N_WIN)]
    out["win"] = {k: np.array([S[m, c].mean() for m in win]) for k, c in (("T_kin", 1), ("U_per_N", 2), ("pressure", 3))}
    out["win_edges"] = edges
    V = npz[f"V_{dt}"][keep].astype(float)
    Q = npz[f"Q_{dt}"][keep].astype(float)
    step_grid = int(round(GRID / dt))
    nlag_v = int(round(VACF_TMAX[law] / dt)) + 1
    c = C.vacf(V, nlag_v)
    out["vacf_grid"] = c[::step_grid]                             # at t_j = j * 0.01
    out["vacf_full"] = c
    out["D_GK_info"] = float(np.trapezoid(c, dx=dt) / 3)
    nlag_m = int(round(max(MSD_TIMES) / dt)) + 1
    if nlag_m > len(Q) // 2:
        raise ValueError(f"t_end too short for MSD at t = {max(MSD_TIMES)}")
    m = C.msd(Q, nlag_m)
    out["msd"] = np.array([m[int(round(x / dt))] for x in MSD_TIMES])
    rs = int(round(RDF_STRIDE_T / dt))
    tq = t[keep][::rs]
    hs = [C.rdf_counts(Q[::rs][(tq >= edges[w] - 1e-12) & (tq <= edges[w + 1] + 1e-12)], L, RDF_EDGES)
          for w in range(N_WIN)]
    nf = [int(((tq >= edges[w] - 1e-12) & (tq <= edges[w + 1] + 1e-12)).sum()) for w in range(N_WIN)]
    h = C.rdf_counts(Q[::rs], L, RDF_EDGES)
    out["rdf"] = C.rdf_from_counts(h, len(Q[::rs]), Q.shape[1], L, RDF_EDGES)
    out["win"]["rdf_peak"] = np.array([C.rdf_from_counts(hw, n, Q.shape[1], L, RDF_EDGES).max() for hw, n in zip(hs, nf)])
    return out


def ci_t(d):
    return C.t_ci(d, 0.95)


def verdict(lo, hi, tol):
    if -tol <= lo and hi <= tol:
        return "PASS"
    if hi < -tol or lo > tol:
        return "FAIL"
    return "INCONCLUSIVE"


def boot_upper(D, measure, rng):
    """95% bootstrap upper bound of measure(mean over replicas of D) (D: replicas x ...)."""
    n = len(D)
    vals = np.array([measure(D[rng.integers(0, n, n)].mean(axis=0)) for _ in range(N_BOOT)])
    return float(measure(D.mean(axis=0))), float(np.percentile(vals, 95))


def pathwise(npz, levels):
    """RMS velocity difference (per component) of each level vs the finest, on the common 0.01 grid."""
    ref = levels[0]
    Vr = npz[f"V_{ref}"][:: int(round(GRID / ref))].astype(float)
    out = {}
    for dt in levels[1:]:
        Vd = npz[f"V_{dt}"][:: int(round(GRID / dt))].astype(float)
        n = min(len(Vr), len(Vd))
        out[str(dt)] = np.sqrt(np.mean((Vd[:n] - Vr[:n]) ** 2, axis=(1, 2)))
    return out


def analyze_law(law, rng):
    reps = load(law)
    if not reps:
        return None
    L = C.box_length(256, 0.75)
    level_sets = sorted({tuple(m["levels"]) for m, _ in reps}, key=len)
    results = {}
    for levels in level_sets:
        rs = [(m, f) for m, f in reps if tuple(m["levels"]) == levels]
        ref = levels[0]
        obs = []
        paths, mon = [], []
        cpu = 0.0
        for meta, f in rs:
            with np.load(f) as npz:
                obs.append({dt: per_level(npz, dt, law, L) for dt in levels})
                paths.append(pathwise(npz, levels))
                mon.append({str(dt): np.nanmax(npz[f"mon_{dt}"][:, 1:], axis=0).tolist() for dt in levels})
            cpu += meta["cpu_s"]
        tag = f"levels_{'_'.join(map(str, levels))}"
        res = dict(levels=levels, reference=ref, n_replicas=len(rs), replicas=[m["replica"] for m, _ in rs],
                   t_end=rs[0][0]["t_end"], cpu_core_hours=cpu / 3600, comparisons={},
                   lanczos_monitor_max=dict((str(dt), np.max([mo[str(dt)] for mo in mon], axis=0).tolist())
                                            for dt in levels),
                   lanczos_monitor_cols=["damp_err_est", "noise_err_est", "coupling_err_est"])
        means = {}
        for dt in levels:
            means[str(dt)] = {k: ci_t([o[dt][k] for o in obs]) for k in ("T_kin", "U_per_N", "pressure")}
            means[str(dt)]["D_GK_info"] = ci_t([o[dt]["D_GK_info"] for o in obs])
            means[str(dt)]["msd"] = [ci_t([o[dt]["msd"][j] for o in obs]) for j in range(len(MSD_TIMES))]
        res["level_means"] = means
        ref_mean = {k: np.mean([o[ref][k] for o in obs]) for k in ("T_kin", "U_per_N", "pressure")}
        for dt in levels[1:]:
            cmp_ = {}
            for k in ("T_kin", "U_per_N"):
                d = [(o[dt][k] - o[ref][k]) / abs(ref_mean[k]) for o in obs]
                m, lo, hi = ci_t(d)
                cmp_[k] = dict(rel_diff=m, ci95=[lo, hi], tol=TOL, verdict=verdict(lo, hi, TOL))
            d = [o[dt]["pressure"] - o[ref]["pressure"] for o in obs]
            m, lo, hi = ci_t(d)
            tolp = 0.01 * 0.75 * 1.0
            cmp_["pressure"] = dict(abs_diff=m, ci95=[lo, hi], tol_abs=tolp, verdict=verdict(lo, hi, tolp),
                                    rel_diff=m / abs(ref_mean["pressure"]))
            # RDF
            Dg = np.array([o[dt]["rdf"] - o[ref]["rdf"] for o in obs])
            gref = np.mean([o[ref]["rdf"] for o in obs], axis=0)
            rc = 0.5 * (RDF_EDGES[1:] + RDF_EDGES[:-1])
            w = 4 * np.pi * 0.75 * rc ** 2 * np.diff(RDF_EDGES)
            den = np.sqrt(np.sum(w * (gref - 1) ** 2))
            Eg, Eg_up = boot_upper(Dg, lambda x: np.sqrt(np.sum(w * x ** 2)) / den, rng)
            flips_g = np.array([np.sqrt(np.sum(w * ((Dg * rng.choice([-1, 1], len(Dg))[:, None]).mean(0)) ** 2)) / den
                                for _ in range(400)])
            ipk = int(np.argmax(gref))
            ph = [(o[dt]["rdf"][ipk] - o[ref]["rdf"][ipk]) / gref[ipk] for o in obs]
            m, lo, hi = ci_t(ph)
            cmp_["rdf"] = dict(E_g=Eg, E_g_upper95=Eg_up, tol=TOL, noise_floor_median=float(np.median(flips_g)), verdict="PASS" if Eg_up <= TOL else
                               ("FAIL" if Eg > TOL else "INCONCLUSIVE"),
                               first_peak_height=dict(rel_diff=m, ci95=[lo, hi], verdict=verdict(lo, hi, TOL),
                                                      r_peak=float(rc[ipk]), g_peak=float(gref[ipk])))
            # VACF on the common grid
            Dv = np.array([o[dt]["vacf_grid"] - o[ref]["vacf_grid"] for o in obs])
            c0 = np.mean([o[ref]["vacf_grid"][0] for o in obs])
            Ev, Ev_up = boot_upper(Dv, lambda x: np.max(np.abs(x)) / c0, rng)
            # noise floor of the sup-norm measure: same statistic on replica-sign-flipped differences (zero mean)
            flips = np.array([np.max(np.abs((Dv * rng.choice([-1, 1], len(Dv))[:, None]).mean(0))) / c0
                              for _ in range(400)])
            cmp_["vacf"] = dict(E_V=Ev, E_V_upper95=Ev_up, tol=TOL, t_max=VACF_TMAX[law],
                                noise_floor_median=float(np.median(flips)),
                                verdict="PASS" if Ev_up <= TOL else ("FAIL" if Ev > TOL else "INCONCLUSIVE"))
            # MSD
            msd_c = []
            for j, tt in enumerate(MSD_TIMES):
                d = [(o[dt]["msd"][j] - o[ref]["msd"][j]) / np.mean([oo[ref]["msd"][j] for oo in obs]) for o in obs]
                m, lo, hi = ci_t(d)
                msd_c.append(dict(t=tt, rel_diff=m, ci95=[lo, hi], verdict=verdict(lo, hi, TOL)))
            cmp_["msd"] = msd_c
            d = [(o[dt]["D_GK_info"] - o[ref]["D_GK_info"]) / np.mean([oo[ref]["D_GK_info"] for oo in obs]) for o in obs]
            m, lo, hi = ci_t(d)
            cmp_["D_GK_info_short"] = dict(rel_diff=m, ci95=[lo, hi], note="GK to t_max only; information")
            # stationarity of the paired differences (they start at 0 from the common state): per-window CIs and a
            # late-minus-early trend; a diagnostic, not a change of the pre-registered criteria
            st = {}
            for k in ("T_kin", "U_per_N", "pressure", "rdf_peak"):
                scale = 1.0 if k == "pressure" else abs(np.mean([o[ref]["win"][k].mean() for o in obs]))
                dw = np.array([(o[dt]["win"][k] - o[ref]["win"][k]) / scale for o in obs])
                st[k] = dict(windows=[list(map(float, obs[0][ref]["win_edges"][w:w + 2])) for w in range(N_WIN)],
                             per_window=[ci_t(dw[:, w]) for w in range(N_WIN)],
                             late_minus_early=ci_t(dw[:, -1] - dw[:, 0]),
                             last_window_verdict=verdict(*ci_t(dw[:, -1])[1:],
                                                         0.0075 if k == "pressure" else TOL),
                             unit="absolute" if k == "pressure" else "relative")
            cmp_["stationarity"] = st
            allv = [cmp_[k]["verdict"] for k in ("T_kin", "U_per_N", "pressure", "rdf", "vacf")]
            allv += [cmp_["rdf"]["first_peak_height"]["verdict"]] + [x["verdict"] for x in msd_c]
            cmp_["all_pass"] = all(v == "PASS" for v in allv)
            cmp_["any_fail"] = any(v == "FAIL" for v in allv)
            # pathwise coupling
            P = np.array([p[str(dt)] for p in paths])
            tg = np.arange(P.shape[1]) * GRID
            cmp_["pathwise_rms_dv"] = {f"t={x:g}": float(np.median(P[:, int(round(x / GRID))]))
                                       for x in (0.05, 0.2, 0.5, 1, 2, 5) if x / GRID < P.shape[1]}
            res["comparisons"][f"{dt}_vs_{ref}"] = cmp_
            res.setdefault("_curves", {})[str(dt)] = dict(t=tg.tolist(), rms_dv=np.median(P, axis=0).tolist(),
                                                          vacf_diff=(Dv.mean(0) / c0).tolist(),
                                                          rdf_diff=Dg.mean(0).tolist())
        res["_curves_ref"] = dict(vacf=(np.mean([o[ref]["vacf_grid"] for o in obs], axis=0) / c0).tolist(),
                                  rdf=gref.tolist(), r=rc.tolist())
        results[tag] = res
    return results


def figure(law, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag = max(res, key=lambda k: res[k]["n_replicas"])
    r = res[tag]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, (dt, cur) in enumerate(r["_curves"].items()):
        t = np.array(cur["t"])
        ax[0].semilogy(t, np.maximum(cur["rms_dv"], 1e-16), color=cols[i], label=f"dt {dt} vs {r['reference']}")
        tv = np.arange(len(cur["vacf_diff"])) * GRID
        ax[1].plot(tv, cur["vacf_diff"], color=cols[i], label=f"dt {dt}")
        ax[2].plot(r["_curves_ref"]["r"], cur["rdf_diff"], color=cols[i], label=f"dt {dt}")
    ax[0].set(xlabel="t", ylabel="median RMS |v_dt - v_ref|", title="pathwise coupling (median over replicas)")
    ax[1].axhspan(-TOL, TOL, color="#eeeeee")
    ax[1].set(xlabel="t", ylabel="mean ΔC(t)/C_ref(0)", title="VACF difference (band ±1%)")
    ax[2].set(xlabel="r", ylabel="mean Δg(r)", title="RDF difference")
    for a in ax:
        a.legend(fontsize=7)
    fig.suptitle(f"lj075 time-step study, law {law}: {r['n_replicas']} replicas, levels {r['levels']}, "
                 f"t_end {r['t_end']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.OUT / f"dt_study_{law}.png", dpi=120)
    plt.close(fig)


def main():
    rng = np.random.default_rng(99)
    out = dict(criteria=json.loads((C.OUT / "dt_criteria.json").read_text()), provenance=C.provenance())
    for law in ("A", "B"):
        res = analyze_law(law, rng)
        if res:
            figure(law, res)
            out[law] = res
            for tag, r in res.items():
                for k, c in r["comparisons"].items():
                    print(f"[{law} {tag}] {k}: T {c['T_kin']['verdict']} {c['T_kin']['rel_diff']:+.2e} "
                          f"{np.round(c['T_kin']['ci95'], 4)}; U {c['U_per_N']['verdict']} "
                          f"{c['U_per_N']['rel_diff']:+.2e} {np.round(c['U_per_N']['ci95'], 4)}; P "
                          f"{c['pressure']['verdict']} {c['pressure']['abs_diff']:+.2e}; RDF {c['rdf']['verdict']} "
                          f"{c['rdf']['E_g_upper95']:.2e}; VACF {c['vacf']['verdict']} {c['vacf']['E_V_upper95']:.2e}; "
                          f"MSD {[x['verdict'][0] for x in c['msd']]}", flush=True)
    C.write_json(C.OUT / "dt_study.json", out)


if __name__ == "__main__":
    main()
