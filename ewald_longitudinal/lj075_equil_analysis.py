#!/usr/bin/env python3
"""
Analysis of the lj075 equilibration runs (lj075_equilibration.py): burn-in, initial-state dependence, sampling
requirements, long-time diffusion. Writes lj075_results/equilibration.json and equilibration_<law>.png.

Rules (stated here before the analysis of the 50-unit runs; revised after code review, see REPORT):

Run set: all complete runs eq_<law>_<ic>_s<seed>; they must share config, dt, L, kT and save frames every step from
  step 0 (asserted). Starting types (ic): fcc (lattice, must melt), rsa (random packing), hot (canonical positions,
  momenta at kT = 2), lgv (canonical positions and momenta: the equilibrium-start null reference).

Observables per 0.5-unit window: U/N, T_kin, pressure (virial, every 0.02), RDF first-peak height, S(k) on the
  smallest-k shell, cell-count variance (27 cells), RDF distance ||g_w - g_pool||_w / ||g_pool - 1||_w.

Burn-in:
  stationary pool = all windows with t >= T_POOL (= 25) of all runs; its first and second halves are compared
  (replica t-test) to check that the pool itself is stationary.
  For block length B = 2 time units: s_B = standard deviation of 2-unit block means in the pool (all runs, all blocks);
  for each start type g (n_g runs) and observable X: d_g(b) = (mean over the g runs of block b) - pool mean;
  t_eq(g, X) = end of the LAST block with |d_g(b)| > 4 s_B / sqrt(n_g) (0 if none). With 4 sigma the false-alarm
  probability per block is ~6e-5, so isolated noise does not set the burn-in.
  burn-in = max over g and X of t_eq; MSER (White 1997, d <= n/2) per run and observable is reported for information
  with d = n/2 flagged as "not determined".
Initial-state dependence after burn-in: one-way ANOVA of the per-run means across start types (replica-based; no
  autocorrelation model needed); p < 0.01 = dependence detected. Autocorrelation-based z-scores are reported for
  information with a reliability flag (series length >= 50 tau_int).
Errors of single series: tau_int by Sokal's window AND by block averaging (blocks of >= 10 tau, plateau of the block
  SE); the larger SE is used; flagged unreliable if T < 50 tau.
Sampling requirements (after burn-in): per observable, s_run = standard deviation over runs of the per-run means
  (run length T_run). Total production time for a 95% half-width eps: T_tot = (1.96 s_run)^2 T_run / eps^2 (valid if
  T_run >> tau). Targets: T_kin and U/N 0.2% (well below the 1% dt tolerance), pressure 0.0075 (= the dt-study
  tolerance), RDF first peak 0.5%, D 5%.
Diffusion (after burn-in, every-step frames): D_GK(tau) = (1/3) int_0^tau C_v (every-step trapezoid; for this
  integrator it equals the chain's own long-time MSD diffusion, so it carries the chain's dt bias but no quadrature
  error), D_MSD from linear fits on windows [1,2], [2,4], [4,8], [8,16]. Plateau test with paired per-run differences:
  GK: D(2 tau) - D(tau); MSD: adjacent windows. A plateau is declared only if the whole 95% t-CI of the relative change
  lies within +-2% (status "plateau"); "drift" if the whole CI lies beyond +-2% (a resolved change), else
  "undetermined" (CI too wide: under-sampled). Long-time D counts as determined only if every step from some limit /
  window up to the longest one is "plateau" (for GK and MSD alike); otherwise it is reported as NOT determined (values are lower/upper bounds
  of the running estimate only).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lj075_common as C  # noqa: E402

RUNS = C.RAW / "runs"
W = 0.5
B = 2.0
T_POOL = 25.0
RDF_EDGES = np.arange(0.5, 3.48, 0.02)
TARGETS = dict(T_kin=0.002, U_per_N=0.002, pressure=0.0075, rdf_peak=0.005, D=0.05)
REL = dict(T_kin=True, U_per_N=True, pressure=False, rdf_peak=True)
GK_TAUS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
MSD_WINDOWS = ((1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0))
OBS = ("U_per_N", "T_kin", "pressure", "rdf_peak", "S_kmin", "cell_var", "rdf_dist")


def mser(x):
    n = len(x)
    best, d_best = np.inf, 0
    for d in range(0, n // 2 + 1):
        y = x[d:]
        v = np.sum((y - y.mean()) ** 2) / (n - d) ** 2
        if v < best:
            best, d_best = v, d
    return d_best, d_best == n // 2


def robust_se(x, dt_sample):
    """SE of the mean: max of Sokal-window and block-average estimates; tau in time units; reliability flag."""
    x = np.asarray(x, float)
    m, se_s, tau = C.mean_se(x)
    tau_t = tau * dt_sample
    nb_len = max(1, int(np.ceil(10 * tau)))
    nb = len(x) // nb_len
    se_b = C.block_se(x, nb) if nb >= 4 else np.nan
    se = np.nanmax([se_s, se_b])
    return dict(mean=float(m), se=float(se), se_sokal=float(se_s), se_block=float(se_b), tau=float(tau_t),
                reliable=bool(len(x) >= 50 * tau))


def cell_variance(Q, L, m=3):
    idx = np.floor((Q % L) / (L / m)).astype(int).clip(0, m - 1)
    flat = idx[..., 0] * m * m + idx[..., 1] * m + idx[..., 2]
    counts = np.stack([np.bincount(f, minlength=m ** 3) for f in flat])
    return counts.var(axis=1) / counts.mean(axis=1)


def load_runs(law):
    import toy_run
    out = {}
    ref = None
    for r in sorted(RUNS.glob(f"eq_{law}_*")):
        st = json.loads((r / "status.json").read_text()) if (r / "status.json").exists() else {}
        if st.get("state") != "complete":
            continue
        d = toy_run.load_run(r)
        rs = d["config"]["resolved"]
        key = (rs["config"], rs["dt"], round(rs["L"], 12), rs["model"]["kT"], rs["law"], rs["potential"])
        if ref is None:
            ref = key
        if key != ref:
            raise ValueError(f"{r.name}: run parameters {key} differ from {ref}; refusing to pool")
        if d["frame_step"][0] != 0 or not np.all(np.diff(d["frame_step"]) == 1):
            raise ValueError(f"{r.name}: frames are not every step from step 0")
        out[r.name] = d
    return out, ref


def window_series(d, L, dt):
    diag = d["diag"]
    t = diag[:, 1]
    nw = int(np.floor(t[-1] / W + 1e-9))
    Q, V = d["Q"], d["V"]
    stride = int(round(0.02 / dt))
    sel = np.arange(0, len(Q), stride)
    tf = sel * dt
    out = {k: np.zeros(nw) for k in OBS}
    g_w = np.zeros((nw, len(RDF_EDGES) - 1))
    _, Sk = C.structure_factor(Q[sel], L, 1)
    cv = cell_variance(Q[sel], L)
    P = np.array([C.pressure(Q[i], V[i], L, 1.0) for i in sel])
    for w in range(nw):
        m = (t >= w * W - 1e-12) & (t < (w + 1) * W - 1e-12)
        out["U_per_N"][w] = diag[m, 3].mean()
        out["T_kin"][w] = diag[m, 2].mean()
        mf = (tf >= w * W - 1e-12) & (tf < (w + 1) * W - 1e-12)
        out["pressure"][w] = P[mf].mean()
        out["S_kmin"][w] = Sk[mf, 0].mean()
        out["cell_var"][w] = cv[mf].mean()
        h = C.rdf_counts(Q[sel][mf], L, RDF_EDGES)
        g_w[w] = C.rdf_from_counts(h, mf.sum(), Q.shape[1], L, RDF_EDGES)
        out["rdf_peak"][w] = g_w[w].max()
    series = dict(t_step=t, U=diag[:, 3], T=diag[:, 2], t_frame=tf, P=P)
    return out, g_w, series


def diffusion(Q, V, dt, t0):
    k0 = int(round(t0 / dt))
    Qp, Vp = Q[k0:], V[k0:]
    s = max(1, int(round(0.01 / dt)))
    Qs = Qp[::s]
    nl = min(len(Qs) - 1, int(round(max(w[1] for w in MSD_WINDOWS) / (s * dt))) + 1)
    m = C.msd(Qs, nl)
    tm = np.arange(len(m)) * s * dt
    msd_fit = {}
    for a, b in MSD_WINDOWS:
        sel = (tm >= a - 1e-9) & (tm <= b + 1e-9)
        if sel.sum() >= 3 and b <= tm[-1] + 1e-9 and b <= (len(Qs) * s * dt) / 2:
            msd_fit[f"{a:g}-{b:g}"] = float(np.polyfit(tm[sel], m[sel], 1)[0] / 6)
    nlv = min(len(Vp) - 1, int(round(max(GK_TAUS) / dt)) + 1)
    c = C.vacf(Vp, nlv)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (c[1:] + c[:-1]) * dt)]) / 3
    gk = {f"{tau:g}": float(cum[int(round(tau / dt))]) for tau in GK_TAUS if int(round(tau / dt)) < len(cum)}
    # stride-2 trapezoid (Richardson estimate of the every-step value's distance from its dt -> 0 extrapolation)
    c2 = c[::2]
    cum2 = np.concatenate([[0.0], np.cumsum(0.5 * (c2[1:] + c2[:-1]) * 2 * dt)]) / 3
    rich = {f"{tau:g}": float((cum[int(round(tau / dt))] - cum2[int(round(tau / (2 * dt)))]) / 3)
            for tau in GK_TAUS if int(round(tau / (2 * dt))) < len(cum2)}
    ii = np.unique(np.geomspace(1, len(c) - 1, 300).astype(int))
    return dict(msd_fit=msd_fit, gk=gk, gk_richardson_offset=rich, T_eff=float(c[0] / 3),
                vacf_curve=((ii * dt).tolist(), (c[ii] / c[0]).tolist()),
                msd_curve=(tm[::max(1, len(tm) // 400)].tolist(), m[::max(1, len(tm) // 400)].tolist()))


def plateau(per_run, keys, label):
    """Paired per-run relative change between consecutive keys; plateau if the whole 95% CI is within +-2%."""
    out = []
    for a, b in zip(keys[:-1], keys[1:]):
        rows = [(v[b] - v[a]) / v[a] for v in per_run.values() if a in v and b in v]
        if len(rows) < 3:
            continue
        m, lo, hi = C.t_ci(rows)
        out.append(dict(**{label: f"{a} -> {b}"}, rel_change=m, ci95=[lo, hi], n=len(rows),
                        plateau=bool(-0.02 <= lo and hi <= 0.02), status=plateau_status(lo, hi)))
    return out


def settled(pl):
    """Plateau entries followed only by plateau entries (an undetermined later step could hide a slow drift)."""
    return [p for i, p in enumerate(pl) if all(q["status"] == "plateau" for q in pl[i:])]


def plateau_status(lo, hi, tol=0.02):
    """plateau: CI inside +-tol; drift: CI entirely outside (a resolved change > tol); else undetermined (CI too
    wide to decide, i.e. under-sampled)."""
    if -tol <= lo and hi <= tol:
        return "plateau"
    if lo > tol or hi < -tol:
        return "drift"
    return "undetermined"



def analyze_law(law):
    runs, key = load_runs(law)
    if len(runs) < 3:
        return None
    L, dt = key[2], key[1]
    data = {}
    for name, d in runs.items():
        ws, gw, series = window_series(d, L, dt)
        data[name] = dict(ws=ws, gw=gw, series=series, ic=name.split("_")[2],
                          cpu=sum(s.get("cpu_s", 0) or 0 for s in d["status"]["segments"]),
                          wall=sum(s.get("wall_s", 0) or 0 for s in d["status"]["segments"]),
                          command=d["config"].get("command"), seed=d["config"]["plan"]["seed"],
                          init=d["config"]["init"])
    nw = min(len(v["gw"]) for v in data.values())
    tw = np.arange(nw) * W
    pool_w = tw >= T_POOL
    g_pool = np.mean([v["gw"][:nw][pool_w].mean(0) for v in data.values()], axis=0)
    rc = 0.5 * (RDF_EDGES[1:] + RDF_EDGES[:-1])
    wgt = 4 * np.pi * 0.75 * rc ** 2 * np.diff(RDF_EDGES)
    den = np.sqrt(np.sum(wgt * (g_pool - 1) ** 2))
    for v in data.values():
        v["ws"]["rdf_dist"] = np.sqrt(np.sum(wgt * (v["gw"] - g_pool) ** 2, axis=1)) / den
    # --- stationary pool checks
    nb = int(round(B / W))
    nblk = nw // nb
    blk_t = np.arange(nblk) * B
    pool_b = blk_t >= T_POOL
    bm = {name: {k: v["ws"][k][:nblk * nb].reshape(nblk, nb).mean(1) for k in OBS} for name, v in data.items()}
    pool_stat, burn_tab = {}, {}
    ics = sorted({v["ic"] for v in data.values()})
    for k in OBS:
        allpool = np.concatenate([bm[n][k][pool_b] for n in data])
        mu, sB = allpool.mean(), allpool.std(ddof=1)
        half = int(pool_b.sum()) // 2
        first = [bm[n][k][pool_b][:half].mean() for n in data]
        second = [bm[n][k][pool_b][half:].mean() for n in data]
        tt = stats.ttest_rel(first, second)
        pool_stat[k] = dict(mean=float(mu), block_sd=float(sB), halves_paired_t_p=float(tt.pvalue))
        for g in ics:
            names = [n for n in data if data[n]["ic"] == g]
            dg = np.mean([bm[n][k] for n in names], axis=0) - mu
            h = 4 * sB / np.sqrt(len(names))
            exc = np.nonzero(np.abs(dg) > h)[0]
            burn_tab.setdefault(g, {})[k] = float((exc[-1] + 1) * B) if len(exc) else 0.0
    burn = max(max(x.values()) for x in burn_tab.values())
    mser_t = {name: {k: dict(zip(("d", "at_boundary"), mser(v["ws"][k][:nw]))) for k in OBS} for name, v in data.items()}
    mser_t = {n: {k: dict(t=x["d"] * W, at_boundary=x["at_boundary"]) for k, x in o.items()} for n, o in mser_t.items()}
    # --- IC dependence after burn-in: ANOVA on per-run means + autocorrelation-based info
    anova = {}
    per_run_means = {}
    for k in ("U_per_N", "T_kin", "pressure", "rdf_peak", "S_kmin", "cell_var"):
        rm, se_info = {}, {}
        for name, v in data.items():
            if k in ("U_per_N", "T_kin"):
                t, x, ds = v["series"]["t_step"], v["series"]["U" if k == "U_per_N" else "T"], dt
            elif k == "pressure":
                t, x, ds = v["series"]["t_frame"], v["series"]["P"], 0.02
            else:
                t, x, ds = tw[:nw], v["ws"][k][:nw], W
            r = robust_se(x[t >= burn], ds)
            rm[name] = r["mean"]
            se_info[name] = r
        per_run_means[k] = rm
        groups = [[rm[n] for n in data if data[n]["ic"] == g] for g in ics]
        F = stats.f_oneway(*groups) if all(len(gx) >= 2 for gx in groups) else None
        anova[k] = dict(F=float(F.statistic) if F else None, p=float(F.pvalue) if F else None,
                        dependence_detected=bool(F is not None and F.pvalue < 0.01),
                        group_means={g: float(np.mean(gx)) for g, gx in zip(ics, groups)},
                        between_run_sd=float(np.std(list(rm.values()), ddof=1)),
                        tau_int={n: se_info[n]["tau"] for n in data},
                        tau_reliable={n: se_info[n]["reliable"] for n in data},
                        se_single_run_median=float(np.median([se_info[n]["se"] for n in data])))
    # --- diffusion
    dif = {name: diffusion(runs[name]["Q"], runs[name]["V"], dt, burn) for name in runs}
    gk_keys = [f"{t:g}" for t in GK_TAUS if all(f"{t:g}" in x["gk"] for x in dif.values())]
    msd_keys = [f"{a:g}-{b:g}" for a, b in MSD_WINDOWS if all(f"{a:g}-{b:g}" in x["msd_fit"] for x in dif.values())]
    D_gk = {k: C.t_ci([x["gk"][k] for x in dif.values()]) for k in gk_keys}
    D_msd = {k: C.t_ci([x["msd_fit"][k] for x in dif.values()]) for k in msd_keys}
    gk_pl = plateau({n: x["gk"] for n, x in dif.items()}, gk_keys, "tau")
    msd_pl = plateau({n: x["msd_fit"] for n, x in dif.items()}, msd_keys, "windows")
    gk_det = settled(gk_pl)
    msd_det = settled(msd_pl)
    determined = bool(gk_det) and bool(msd_det)
    # --- sampling requirements
    T_run = (nw * W) - burn
    req = {}
    for k in ("T_kin", "U_per_N", "pressure", "rdf_peak"):
        vals = np.array(list(per_run_means[k].values()))
        s_run = vals.std(ddof=1)
        mu = vals.mean()
        eps = TARGETS[k] * abs(mu) if REL[k] else TARGETS[k]
        req[k] = dict(target_half_width=float(eps), s_run=float(s_run), run_length=float(T_run),
                      total_time_needed=float((1.96 * s_run) ** 2 * T_run / eps ** 2), n_runs=len(vals))
    # D: use the longest GK limit and MSD window available
    dreq = {}
    for lab, kk, src in (("D_GK", gk_keys[-1] if gk_keys else None, "gk"),
                         ("D_MSD", msd_keys[-1] if msd_keys else None, "msd_fit")):
        if kk is None:
            continue
        vals = np.array([x[src][kk] for x in dif.values()])
        s_run, mu = vals.std(ddof=1), vals.mean()
        n_need = None
        for n in range(3, 1000):
            if stats.t.ppf(0.975, n - 1) * s_run / np.sqrt(n) <= TARGETS["D"] * mu:
                n_need = n
                break
        dreq[lab] = dict(at=kk, mean=float(mu), s_run=float(s_run), run_length=float(T_run), n_runs=len(vals),
                         replicas_needed_at_this_length=n_need,
                         total_time_needed=float((1.96 * s_run) ** 2 * T_run / (TARGETS["D"] * mu) ** 2),
                         note="valid only if D at this limit is the long-time D (see plateau)")
    req["D"] = dreq
    res = dict(runs=sorted(data), n_runs=len(data), start_types=ics, dt=dt, L=L, config=key[0],
               run_records={n: dict(seed=v["seed"], init=v["init"], command=v["command"], cpu_s=v["cpu"],
                                    wall_s=v["wall"]) for n, v in data.items()},
               cpu_core_hours=sum(v["cpu"] for v in data.values()) / 3600,
               burn_in=dict(estimate=burn, rule=f"last 2-unit block with |group mean - pool| > 4 s_B/sqrt(n_g); pool t >= "
                                                 f"{T_POOL}", by_start_type=burn_tab, pool=pool_stat,
                            mser_info=mser_t),
               ic_dependence_after_burn_in=anova, sampling_requirements=req, targets=TARGETS,
               diffusion=dict(D_GK=D_gk, D_MSD=D_msd, gk_plateau=gk_pl, msd_plateau=msd_pl,
                              long_time_D_determined=determined,
                              gk_richardson_offset={k: C.t_ci([x["gk_richardson_offset"][k] for x in dif.values()])
                                                    for k in gk_keys},
                              per_run={k: dict(gk=v["gk"], msd_fit=v["msd_fit"], T_eff=v["T_eff"]) for k, v in dif.items()}),
               _curves=dict(window_t=(tw + W / 2).tolist(),
                            ws={name: {k: v["ws"][k][:nw].tolist() for k in OBS} for name, v in data.items()},
                            ic={name: v["ic"] for name, v in data.items()},
                            vacf={name: dif[name]["vacf_curve"] for name in dif},
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
    seen = set()
    for i, k in enumerate(keys):
        for name, ws in cur["ws"].items():
            ic = cur["ic"][name]
            ax[i].plot(t, ws[k], color=cols[ic], lw=0.9, alpha=0.8,
                       label=ic if (i == 0 and ic not in seen and not seen.add(ic)) else None)
        ax[i].axvline(res["burn_in"]["estimate"], color="0.3", ls="--", lw=0.8)
        ax[i].set(xlabel="t", title=k)
    ax[0].legend(fontsize=8, title="start")
    for name, (tv, v) in cur["vacf"].items():
        ax[6].plot(tv, v, color=cols[cur["ic"][name]], lw=0.8)
    ax[6].axhline(0, color="0.6", lw=0.6)
    ax[6].set(xlabel="t", ylabel="C_v(t)/C_v(0)", title="VACF after burn-in", xscale="log")
    dg = res["diffusion"]["D_GK"]
    taus = [float(k) for k in dg]
    g = np.array([dg[k] for k in dg])
    ax[7].errorbar(taus, g[:, 0], yerr=[g[:, 0] - g[:, 1], g[:, 2] - g[:, 0]], fmt="o-", color="#2a78d6",
                   label="D_GK(τ), 95% CI over runs")
    dm = res["diffusion"]["D_MSD"]
    if dm:
        mm = np.array([dm[k] for k in dm])
        mid = [np.sqrt(np.prod([float(x) for x in k.split("-")])) for k in dm]
        ax[7].errorbar(mid, mm[:, 0], yerr=[mm[:, 0] - mm[:, 1], mm[:, 2] - mm[:, 0]], fmt="s-", color="#eb6834",
                       label="D_MSD (window, geometric centre)")
    ax[7].set(xscale="log", xlabel="τ or window centre", ylabel="D",
              title="diffusion: " + ("plateau found" if res["diffusion"]["long_time_D_determined"]
                                     else "NO plateau: long-time D not determined"))
    ax[7].legend(fontsize=8)
    fig.suptitle(f"lj075 equilibration and sampling, law {law} ({res['n_runs']} runs, dt {res['dt']}); dashed: "
                 f"burn-in estimate {res['burn_in']['estimate']:.1f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.OUT / f"equilibration_{law}.png", dpi=110)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--laws", nargs="+", default=["A"], help="laws to analyse (default A: the lj075 campaign "
                    "continued with law A only; B was stopped)")
    args = ap.parse_args()
    out = dict(provenance=C.provenance())
    for law in args.laws:
        res = analyze_law(law)
        if res is None:
            continue
        figure(law, res)
        out[law] = res
        print(f"[{law}] burn-in {res['burn_in']['estimate']:.1f} by start type {res['burn_in']['by_start_type']}",
              flush=True)
        print(f"[{law}] IC dependence p {[(k, round(v['p'], 3) if v['p'] is not None else None) for k, v in res['ic_dependence_after_burn_in'].items()]}", flush=True)
        print(f"[{law}] D determined {res['diffusion']['long_time_D_determined']}; GK {res['diffusion']['D_GK']}", flush=True)
    C.write_json(C.OUT / "equilibration.json", out)


if __name__ == "__main__":
    main()
