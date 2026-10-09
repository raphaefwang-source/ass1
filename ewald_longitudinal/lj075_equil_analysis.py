#!/usr/bin/env python3
"""
Analysis of the lj075 equilibration runs (lj075_equilibration.py): burn-in, initial-state dependence, sampling
requirements, long-time diffusion. Writes lj075_results/equilibration.json and equilibration_<law>.png.

Burn-in (per run, per observable): MSER truncation (White 1997) on 0.5-time-unit window means of
    U/N, T_kin, pressure, RDF first-peak height, S(k) on the smallest-k shell, cell-count variance (27 cells),
    the RDF distance ||g_w - g_ref||_w (relative to ||g_ref - 1||_w)
  t_MSER = argmin_d sum_{i>d} (x_i - mean_{>d})^2 / (n - d)^2 over d <= n/2.
  burn-in estimate = max over runs and observables; cross-check: after discarding it, the run means of each
  observable must be mutually consistent (chi^2 of run means about the pooled mean with autocorrelation-corrected
  standard errors, p > 0.01) and the four initial-state groups must agree (max |group mean - pooled| / SE).
Sampling (after burn-in): integrated autocorrelation time tau_int (Sokal window), per-run SE, between-run scatter vs
  predicted SE, and the production time needed for the pre-stated targets (95% half-width):
    T_kin, U/N 0.2%;  pressure 0.0075 (= the dt-study tolerance);  RDF first peak 0.5%;  D 5%.
Diffusion: MSD (all origins, COM removed) and GK (VACF integral) per run; local slope D_MSD(t) = (1/6) dMSD/dt and
  D_GK(tau) = (1/3) int_0^tau C_v; plateau test: relative change < 2% between tau and 2 tau (GK) or between fit windows
  [t, 2t] and [2t, 4t] (MSD), with 95% CIs over runs. No plateau -> long-time D reported as NOT determined.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lj075_common as C  # noqa: E402

RUNS = C.RAW / "runs"
W = 0.5                                         # window length for burn-in series
RDF_EDGES = np.arange(0.5, 3.48, 0.02)
TARGETS = dict(T_kin=0.002, U_per_N=0.002, pressure_abs=0.0075, rdf_peak=0.005, D=0.05)
GK_TAUS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0)
MSD_WINDOWS = ((0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0))


def mser(x):
    n = len(x)
    best, d_best = np.inf, 0
    for d in range(0, n // 2 + 1):
        y = x[d:]
        v = np.sum((y - y.mean()) ** 2) / (n - d) ** 2
        if v < best:
            best, d_best = v, d
    return d_best


def cell_variance(Q, L, m=3):
    """Variance / mean of particle counts in m^3 cells (per frame)."""
    idx = np.floor((Q % L) / (L / m)).astype(int).clip(0, m - 1)
    flat = idx[..., 0] * m * m + idx[..., 1] * m + idx[..., 2]
    counts = np.stack([np.bincount(f, minlength=m ** 3) for f in flat])
    return counts.var(axis=1) / counts.mean(axis=1)


def load(out):
    import toy_run
    d = toy_run.load_run(out)
    return d


def window_series(d, L, dt, ref_g=None):
    diag = d["diag"]
    t = diag[:, 1]
    nw = int(np.floor(t[-1] / W + 1e-9))
    fsteps = d["frame_step"]
    tf = fsteps * dt
    Q, V = d["Q"], d["V"]
    stride = int(round(0.02 / dt))                       # structural observables every 0.02
    sel = np.arange(0, len(tf), stride)
    out = {k: np.zeros(nw) for k in ("U_per_N", "T_kin", "pressure", "rdf_peak", "S_kmin", "cell_var")}
    g_w = np.zeros((nw, len(RDF_EDGES) - 1))
    shells, Sk = C.structure_factor(Q[sel], L, 1)
    cv = cell_variance(Q[sel], L)
    P = np.array([C.pressure(Q[i], V[i], L, 1.0) for i in sel])
    for w in range(nw):
        m = (t >= w * W - 1e-12) & (t < (w + 1) * W - 1e-12)
        out["U_per_N"][w] = diag[m, 3].mean()
        out["T_kin"][w] = diag[m, 2].mean()
        mf = (tf[sel] >= w * W - 1e-12) & (tf[sel] < (w + 1) * W - 1e-12)
        out["pressure"][w] = P[mf].mean()
        out["S_kmin"][w] = Sk[mf, 0].mean()
        out["cell_var"][w] = cv[mf].mean()
        h = C.rdf_counts(Q[sel][mf], L, RDF_EDGES)
        g_w[w] = C.rdf_from_counts(h, mf.sum(), Q.shape[1], L, RDF_EDGES)
        out["rdf_peak"][w] = g_w[w].max()
    return out, g_w, dict(t_step=t, U=diag[:, 3], T=diag[:, 2], t_frame=tf[sel], P=P, Sk=Sk[:, 0], cv=cv)


def diffusion(Q, V, dt, t0):
    """Per-run MSD and GK diffusion diagnostics from every-step frames after burn-in t0."""
    k0 = int(round(t0 / dt))
    Qp, Vp = Q[k0:], V[k0:]
    Tn = len(Qp)
    nlag_m = min(Tn - 1, int(round(16.0 / dt)) + 1)
    # MSD on a coarser grid for cost: every 0.01 (exact lags; frames subsampled)
    s = max(1, int(round(0.01 / dt)))
    m = C.msd(Qp[::s], min(len(Qp[::s]) - 1, int(16.0 / (s * dt)) + 1))
    tm = np.arange(len(m)) * s * dt
    msd_fit = {}
    for a, b in MSD_WINDOWS:
        sel = (tm >= a) & (tm <= b)
        if sel.sum() >= 3 and b <= tm[-1] * 0.5 + 1e-9:
            msd_fit[f"{a}-{b}"] = float(np.polyfit(tm[sel], m[sel], 1)[0] / 6)
    nl = min(Tn - 1, int(round(max(GK_TAUS) / dt)) + 1)
    c = C.vacf(Vp, nl)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (c[1:] + c[:-1]) * dt)]) / 3
    gk = {str(tau): float(cum[int(round(tau / dt))]) for tau in GK_TAUS if int(round(tau / dt)) < len(cum)}
    return dict(msd_fit=msd_fit, gk=gk, msd_curve=(tm[::10].tolist(), m[::10].tolist()),
                vacf_curve=(np.arange(len(c))[:: max(1, len(c) // 400)] * dt).tolist(),
                vacf=(c[:: max(1, len(c) // 400)] / c[0]).tolist(), c0=float(c[0]), T_eff=float(c[0] / 3))


def analyze_law(law):
    runs = sorted(RUNS.glob(f"eq_{law}_*"))
    runs = [r for r in runs if (r / "status.json").exists() and json.loads((r / "status.json").read_text())["state"]
            == "complete"]
    if not runs:
        return None
    L = C.box_length(256, 0.75)
    data = {}
    for r in runs:
        d = load(r)
        dt = d["config"]["resolved"]["dt"]
        ws, gw, series = window_series(d, L, dt)
        data[r.name] = dict(ws=ws, gw=gw, series=series, dt=dt, d=d,
                            ic=r.name.split("_")[2], cpu=sum(s.get("cpu_s", 0) or 0 for s in d["status"]["segments"]),
                            config=d["config"])
    # reference RDF: pooled second half of all runs
    nw = min(len(v["gw"]) for v in data.values())
    g_ref = np.mean([v["gw"][nw // 2:nw].mean(0) for v in data.values()], axis=0)
    rc = 0.5 * (RDF_EDGES[1:] + RDF_EDGES[:-1])
    w = 4 * np.pi * 0.75 * rc ** 2 * np.diff(RDF_EDGES)
    den = np.sqrt(np.sum(w * (g_ref - 1) ** 2))
    for v in data.values():
        v["ws"]["rdf_dist"] = np.sqrt(np.sum(w * (v["gw"] - g_ref) ** 2, axis=1)) / den
    obs = list(next(iter(data.values()))["ws"])
    mser_t = {name: {k: mser(v["ws"][k]) * W for k in obs} for name, v in data.items()}
    burn = max(max(x.values()) for x in mser_t.values())
    # consistency after burn-in
    consist = {}
    for k in ("U_per_N", "T_kin", "pressure", "rdf_peak", "S_kmin", "cell_var"):
        means, ses, ics = [], [], []
        for name, v in data.items():
            if k in ("U_per_N", "T_kin"):
                t, x = v["series"]["t_step"], v["series"]["U" if k == "U_per_N" else "T"]
            elif k == "pressure":
                t, x = v["series"]["t_frame"], v["series"]["P"]
            elif k == "S_kmin":
                t, x = v["series"]["t_frame"], v["series"]["Sk"]
            elif k == "cell_var":
                t, x = v["series"]["t_frame"], v["series"]["cv"]
            else:
                t, x = np.arange(len(v["ws"][k])) * W, v["ws"][k]
            sel = t >= burn
            m_, se, tau = C.mean_se(x[sel])
            means.append(m_)
            ses.append(se)
            ics.append(v["ic"])
        means, ses = np.array(means), np.array(ses)
        wts = 1 / ses ** 2
        pooled = np.sum(wts * means) / np.sum(wts)
        chi2 = float(np.sum(((means - pooled) / ses) ** 2))
        from scipy import stats
        pval = float(stats.chi2.sf(chi2, len(means) - 1))
        grp = {}
        for ic in sorted(set(ics)):
            sel = np.array([i == ic for i in ics])
            gm = means[sel].mean()
            gse = np.sqrt(np.sum(ses[sel] ** 2)) / sel.sum()
            grp[ic] = dict(mean=float(gm), z=float((gm - pooled) / gse))
        sd_runs = float(np.std(means, ddof=1))
        consist[k] = dict(pooled=float(pooled), chi2=chi2, dof=len(means) - 1, p=pval, groups=grp,
                          between_run_sd=sd_runs, mean_predicted_se=float(np.mean(ses)),
                          ratio_sd_to_se=float(sd_runs / np.mean(ses)))
    # sampling requirements: tau_int from step / frame series after burn-in, pooled over runs
    req = {}
    for k, target, rel in (("U_per_N", TARGETS["U_per_N"], True), ("T_kin", TARGETS["T_kin"], True),
                           ("pressure", TARGETS["pressure_abs"], False)):
        taus, vars_ = [], []
        for v in data.values():
            if k == "pressure":
                t, x, dts = v["series"]["t_frame"], v["series"]["P"], 0.02
            else:
                t, x, dts = v["series"]["t_step"], v["series"]["U" if k == "U_per_N" else "T"], v["dt"]
            x = x[t >= burn]
            tau, _, g = C.integrated_act(x)
            taus.append(tau * dts)
            vars_.append(x.var())
        tau_t = float(np.median(taus))
        var = float(np.mean(vars_))
        mean = consist[k]["pooled"]
        eps = target * abs(mean) if rel else target
        # 95% half-width 1.96 sqrt(var * 2 tau / T_total) <= eps
        T_total = 1.96 ** 2 * var * 2 * tau_t / eps ** 2
        req[k] = dict(tau_int_time=tau_t, var=var, target_half_width=eps, total_time_needed=float(T_total))
    # diffusion
    dif = {}
    for name, v in data.items():
        dd = v["d"]
        dif[name] = diffusion(dd["Q"], dd["V"], v["dt"], burn)
    gk_keys = list(next(iter(dif.values()))["gk"])
    msd_keys = list(next(iter(dif.values()))["msd_fit"])
    D_gk = {k: C.t_ci([x["gk"][k] for x in dif.values() if k in x["gk"]]) for k in gk_keys}
    D_msd = {k: C.t_ci([x["msd_fit"][k] for x in dif.values() if k in x["msd_fit"]]) for k in msd_keys}
    taus = [float(k) for k in gk_keys]
    gk_plateau = []
    for i, tau in enumerate(taus):
        if 2 * tau in taus:
            a, b = D_gk[str(tau)][0], D_gk[str(2 * tau)][0]
            gk_plateau.append(dict(tau=tau, rel_change=float((b - a) / b)))
    msd_plateau = []
    for i in range(len(msd_keys) - 1):
        a, b = D_msd[msd_keys[i]][0], D_msd[msd_keys[i + 1]][0]
        msd_plateau.append(dict(windows=f"{msd_keys[i]} -> {msd_keys[i + 1]}", rel_change=float((b - a) / b)))
    res = dict(runs=sorted(data), n_runs=len(data), dt=next(iter(data.values()))["dt"],
               cpu_core_hours=sum(v["cpu"] for v in data.values()) / 3600,
               mser_burn_in_by_run=mser_t, burn_in_estimate=burn, consistency_after_burn_in=consist,
               sampling_requirements=req, targets=TARGETS,
               diffusion=dict(D_GK=D_gk, D_MSD=D_msd, gk_plateau=gk_plateau, msd_plateau=msd_plateau,
                              per_run={k: dict(gk=v["gk"], msd_fit=v["msd_fit"], T_eff=v["T_eff"])
                                       for k, v in dif.items()}),
               _curves=dict(window_t=(np.arange(nw) * W + W / 2).tolist(),
                            ws={name: {k: v["ws"][k][:nw].tolist() for k in obs} for name, v in data.items()},
                            ic={name: v["ic"] for name, v in data.items()},
                            vacf={name: (dif[name]["vacf_curve"], dif[name]["vacf"]) for name in dif},
                            msd={name: dif[name]["msd_curve"] for name in dif}))
    return res


def figure(law, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"fcc": "#2a78d6", "rsa": "#eb6834", "lgv": "#1baf7a", "hot": "#eda100"}
    cur = res["_curves"]
    t = np.array(cur["window_t"])
    keys = ["U_per_N", "T_kin", "pressure", "rdf_peak", "rdf_dist", "S_kmin"]
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    ax = ax.ravel()
    for i, k in enumerate(keys):
        for name, ws in cur["ws"].items():
            ic = cur["ic"][name]
            ax[i].plot(t, ws[k], color=cols[ic], lw=1, alpha=0.8, label=ic if name.endswith(("s301", "s303", "s305",
                                                                                                 "s307")) else None)
        ax[i].axvline(res["burn_in_estimate"], color="0.3", ls="--", lw=0.8)
        ax[i].set(xlabel="t", title=k)
        if k in ("U_per_N", "T_kin", "rdf_dist"):
            ax[i].set_xscale("symlog", linthresh=1.0)
    ax[0].legend(fontsize=8, title="start")
    for name, (tv, v) in cur["vacf"].items():
        ax[6].plot(tv, v, color=cols[cur["ic"][name]], lw=0.8)
    ax[6].axhline(0, color="0.6", lw=0.6)
    ax[6].set(xlabel="t", ylabel="C_v(t)/C_v(0)", title="VACF after burn-in", xscale="log")
    taus = [float(k) for k in res["diffusion"]["D_GK"]]
    g = np.array([res["diffusion"]["D_GK"][k] for k in res["diffusion"]["D_GK"]])
    ax[7].errorbar(taus, g[:, 0], yerr=[g[:, 0] - g[:, 1], g[:, 2] - g[:, 0]], fmt="o-", color="#2a78d6",
                   label="D_GK(τ), 95% CI over runs")
    mk = list(res["diffusion"]["D_MSD"])
    mm = np.array([res["diffusion"]["D_MSD"][k] for k in mk])
    mid = [np.mean([float(x) for x in k.split("-")]) for k in mk]
    ax[7].errorbar(mid, mm[:, 0], yerr=[mm[:, 0] - mm[:, 1], mm[:, 2] - mm[:, 0]], fmt="s-", color="#eb6834",
                   label="D_MSD (window centre)")
    ax[7].set(xscale="log", xlabel="τ or window centre", ylabel="D", title="diffusion: plateau check")
    ax[7].legend(fontsize=8)
    fig.suptitle(f"lj075 equilibration and sampling, law {law} ({res['n_runs']} runs, dt {res['dt']}); dashed: "
                 f"MSER burn-in estimate {res['burn_in_estimate']:.1f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.OUT / f"equilibration_{law}.png", dpi=110)
    plt.close(fig)


def main():
    out = dict(provenance=C.provenance())
    for law in ("A", "B"):
        res = analyze_law(law)
        if res is None:
            continue
        figure(law, res)
        out[law] = res
        print(f"[{law}] burn-in (MSER max) {res['burn_in_estimate']:.1f}; consistency p-values "
              f"{ {k: round(v['p'], 3) for k, v in res['consistency_after_burn_in'].items()} }", flush=True)
        print(f"[{law}] sampling {res['sampling_requirements']}", flush=True)
        print(f"[{law}] D_GK {res['diffusion']['D_GK']}\n[{law}] D_MSD {res['diffusion']['D_MSD']}", flush=True)
    C.write_json(C.OUT / "equilibration.json", out)


if __name__ == "__main__":
    main()
