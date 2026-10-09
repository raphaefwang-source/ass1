#!/usr/bin/env python3
"""
Analysis of the lj075 time-step study (lj075_dt_study.py), following the pre-registered criteria in
lj075_results/dt_criteria.json. Writes lj075_results/dt_study.json and dt_study_<law>.png.

All comparisons are PAIRED by replica (same q0, p0, coupled noise) against the reference level (the finest level
common to the replicas used). Scalars: Student-t 95% CI over replicas of the per-replica difference of time
averages over [T_DISCARD, t_end].
Function-valued observables (amendment after code review, before any multi-replica analysis; more conservative than
the registered percentile bootstrap, which is anti-conservative for few replicas and is still reported):
  per grid point / bin j: mean difference Dbar_j and Student-t 95% half-width h_j = t_{0.975,n-1} s_j / sqrt(n);
  VACF: upper bound max_j (|Dbar_j| + h_j) / C_ref(0); lower bound max_j max(|Dbar_j| - h_j, 0) / C_ref(0);
  RDF:  the squared weighted norm of the mean difference is biased upward by the bin noise (it grows with the
        number of bins); the decision uses the debiased estimate E2 = ||Dbar||_w^2 - sum_j w_j s_j^2 / n (unbiased
        for ||E[D]||_w^2) with a jackknife (over replicas) one-sided 95% t-bound: upper/lower bound of E_g =
        sqrt(max(E2 +- t_{0.95,n-1} SE_jack, 0)) / ||g_ref - 1||_w. The pointwise envelope bound
        ||(|Dbar| + h)||_w / ||g_ref - 1||_w (very conservative) is reported as well.
  PASS if the upper bound <= 1%, FAIL if the lower bound > 1%, INCONCLUSIVE otherwise; n >= 3 replicas required.
RDF first peak: height and position per replica and level by a quadratic fit through the 3 bins around the maximum
(sub-bin position), paired relative differences with t-CIs.
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
    if nlag_m > 0.75 * len(Q):
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
    out["peak_height"], out["peak_pos"] = peak(out["rdf"], RDF_EDGES)
    out["win"]["rdf_peak"] = np.array([C.rdf_from_counts(hw, n, Q.shape[1], L, RDF_EDGES).max() for hw, n in zip(hs, nf)])
    return out


N_MIN = 3


def ci_t(d):
    return C.t_ci(d, 0.95)


def peak(g, edges):
    """First-peak height and position by a quadratic through the 3 bins around the maximum."""
    rc = 0.5 * (edges[1:] + edges[:-1])
    i = int(np.argmax(g))
    y0, y1, y2 = g[i - 1], g[i], g[i + 1]
    den = y0 - 2 * y1 + y2
    x = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    h = rc[1] - rc[0]
    return float(y1 - 0.25 * (y0 - y2) * x), float(rc[i] + x * h)


def t_bounds(D):
    """Per-point mean and Student-t 95% half-width over replicas (rows)."""
    from scipy import stats
    n = len(D)
    return D.mean(0), stats.t.ppf(0.975, n - 1) * D.std(0, ddof=1) / np.sqrt(n)


def debiased_norm_bounds(D, w, den):
    """Debiased ||E[D]||_w and one-sided 95% jackknife-t bounds (see module docstring)."""
    from scipy import stats
    n = len(D)
    if n < 3:
        return float("nan"), float("nan"), float("nan")

    def e2(X):
        k = len(X)
        return np.sum(w * X.mean(0) ** 2) - np.sum(w * X.var(0, ddof=1)) / k
    full = e2(D)
    jk = np.array([e2(np.delete(D, i, axis=0)) for i in range(n)])
    se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    tq = stats.t.ppf(0.95, n - 1)
    f = lambda x: float(np.sqrt(max(x, 0.0)) / den)                      # noqa: E731
    return f(full), f(full + tq * se), f(full - tq * se)


def norm_verdict(up, lo, tol, n):
    if n < N_MIN:
        return "INCONCLUSIVE"
    return "PASS" if up <= tol else ("FAIL" if lo > tol else "INCONCLUSIVE")


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
    level_sets = sorted({(tuple(m["levels"]), m["t_end"]) for m, _ in reps}, key=lambda x: (len(x[0]), x[1]))
    results = {}
    for levels, t_end in level_sets:
        rs = [(m, f) for m, f in reps if tuple(m["levels"]) == levels and m["t_end"] == t_end]
        ids = [(m["replica"], m["init_state"]) for m, _ in rs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate replicas in group {levels}, t_end {t_end}: {ids}")
        sig = {(m["resolved"]["rank_noise"], m["resolved"]["rank_damp"], m["rank_couple"],
                json.dumps(m["resolved"]["pppm"], sort_keys=True)) for m, _ in rs}
        if len(sig) != 1:
            raise ValueError(f"replicas with different ranks / PPPM in one group: {sig}")
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
        tag = f"levels_{'_'.join(map(str, levels))}_T{t_end:g}"
        ver = json.loads((C.OUT / f"verify_representative_{law}.json").read_text())
        budget = ver["budget"]
        res = dict(levels=levels, reference=ref, n_replicas=len(rs), replicas=[m["replica"] for m, _ in rs],
                   t_end=rs[0][0]["t_end"], cpu_core_hours=cpu / 3600, comparisons={},
                   lanczos_monitor_max=dict((str(dt), np.max([mo[str(dt)] for mo in mon], axis=0).tolist())
                                            for dt in levels),
                   lanczos_monitor_cols=["damp_err_est", "noise_err_est", "coupling_err_est"],
                   operator_ok={str(dt): bool(np.nanmax([np.nanmax(np.array(mo[str(dt)], float)) for mo in mon])
                                              <= budget / 10) for dt in levels},
                   operator_budget=budget)
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
            Eg, Eg_boot = boot_upper(Dg, lambda x: np.sqrt(np.sum(w * x ** 2)) / den, rng)
            mg, hg = t_bounds(Dg)
            Eg_env_up = float(np.sqrt(np.sum(w * (np.abs(mg) + hg) ** 2)) / den)
            Eg_deb, Eg_up, Eg_lo = debiased_norm_bounds(Dg, w, den)
            flips_g = np.array([np.sqrt(np.sum(w * ((Dg * rng.choice([-1, 1], len(Dg))[:, None]).mean(0)) ** 2)) / den
                                for _ in range(400)])
            ph = [(o[dt]["peak_height"] - o[ref]["peak_height"]) / o[ref]["peak_height"] for o in obs]
            m, lo, hi = ci_t(ph)
            pp_ = [(o[dt]["peak_pos"] - o[ref]["peak_pos"]) / o[ref]["peak_pos"] for o in obs]
            mp, lop, hip = ci_t(pp_)
            cmp_["rdf"] = dict(E_g=Eg, E_g_debiased=Eg_deb, E_g_upper95=Eg_up, E_g_lower95=Eg_lo,
                               E_g_envelope_upper95=Eg_env_up, E_g_bootstrap_p95=Eg_boot, tol=TOL,
                               noise_floor_median=float(np.median(flips_g)), n=len(obs),
                               verdict=norm_verdict(Eg_up, Eg_lo, TOL, len(obs)),
                               first_peak_height=dict(rel_diff=m, ci95=[lo, hi], verdict=verdict(lo, hi, TOL),
                                                      g_peak=float(np.mean([o[ref]["peak_height"] for o in obs]))),
                               first_peak_position=dict(rel_diff=mp, ci95=[lop, hip], verdict=verdict(lop, hip, TOL),
                                                        r_peak=float(np.mean([o[ref]["peak_pos"] for o in obs]))))
            # VACF on the common grid
            Dv = np.array([o[dt]["vacf_grid"] - o[ref]["vacf_grid"] for o in obs])
            c0 = np.mean([o[ref]["vacf_grid"][0] for o in obs])
            Ev, Ev_boot = boot_upper(Dv, lambda x: np.max(np.abs(x)) / c0, rng)
            mv, hv = t_bounds(Dv)
            Ev_up = float(np.max(np.abs(mv) + hv) / c0)
            Ev_lo = float(np.max(np.maximum(np.abs(mv) - hv, 0)) / c0)
            # noise floor of the sup-norm measure: same statistic on replica-sign-flipped differences (zero mean)
            flips = np.array([np.max(np.abs((Dv * rng.choice([-1, 1], len(Dv))[:, None]).mean(0))) / c0
                              for _ in range(400)])
            cmp_["vacf"] = dict(E_V=Ev, E_V_upper95=Ev_up, E_V_lower95=Ev_lo, E_V_bootstrap_p95=Ev_boot, tol=TOL,
                                t_max=VACF_TMAX[law], noise_floor_median=float(np.median(flips)), n=len(obs),
                                verdict=norm_verdict(Ev_up, Ev_lo, TOL, len(obs)))
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
            allv += [cmp_["rdf"]["first_peak_height"]["verdict"], cmp_["rdf"]["first_peak_position"]["verdict"]]
            allv += [x["verdict"] for x in msd_c]
            if len(obs) < N_MIN:
                allv.append("INCONCLUSIVE")
            cmp_["operator_ok"] = bool(res["operator_ok"][str(dt)] and res["operator_ok"][str(ref)])
            cmp_["all_pass"] = all(v == "PASS" for v in allv) and cmp_["operator_ok"]
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
