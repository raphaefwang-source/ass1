"""
Analysis of the Section 6 dynamics stages (sec6_dynamics.py analyze). Reads the raw trajectories in
sec6_results/{ideal_gas,dynamics}/raw/ and writes summaries, tables and figures next to them.

Statistics
- Confidence intervals use independent trajectories: per-trajectory time averages (after burn-in) or per-trajectory
  paired differences, Student-t with n_traj - 1 degrees of freedom. Correlated snapshots of one trajectory are never
  treated as independent samples.
- Burn-in: t_b = max(5 tau_fit, 3/(2 lambda*)). tau_fit comes from an exponential fit of the initial contiguous decay
  (|dev| > 3 SE) of the ensemble-mean component temperatures and anisotropy; lambda* is the median Ritz estimate of
  the smallest eigenvalue on T_perp over the run (slowest momentum mode, relaxation rate 2 lambda*/m). The window
  after t_b is checked by comparing its first and second halves (paired over trajectories). A run shorter than t_b
  is reported as relaxation only, equilibrium not established.
- Maxwell marginals: u = (p_ia - P_a/N) / sqrt(m kT (N-1)/N) at every step after burn-in; per-trajectory variance and
  excess kurtosis (correlated snapshots are only averaged within a trajectory), then the t-CI over trajectories.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sec6_config  # noqa: E402
from sec6_dynamics import OUT_IG, OUT_DY, SCRIPT  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COMP = [("#2a78d6", "$T_x$"), ("#eb6834", "$T_y$"), ("#1baf7a", "$T_z$")]


def style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                         "savefig.dpi": 200, "lines.linewidth": 1.2, "mathtext.fontset": "cm"})


def tci(x, axis=0, level=0.95):
    """Mean, half-width of the Student-t CI, SE over independent samples along axis."""
    x = np.asarray(x, float)
    n = x.shape[axis]
    m = x.mean(axis=axis)
    se = x.std(axis=axis, ddof=1) / np.sqrt(n) if n > 1 else np.full_like(m, np.nan)
    hw = stats.t.ppf(0.5 + level / 2, n - 1) * se if n > 1 else np.full_like(m, np.nan)
    return m, hw, se


def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def fl(x):
    return [float(v) for v in np.ravel(x)]


# ----------------------------------------------------------------------------
# ideal gas
# ----------------------------------------------------------------------------
def relaxation_time(t, dev, se):
    """Fit |dev| ~ A exp(-t/tau) over the initial contiguous stretch where |dev| > 3 SE (t > 0). Returns tau or nan."""
    y = np.abs(dev)
    big = (y > 3 * se) & np.isfinite(se)
    stop = np.argmax(~big[1:]) + 1 if (~big[1:]).any() else len(y)
    ok = np.zeros_like(big)
    ok[1:stop] = True
    if ok.sum() < 4:
        return np.nan, int(ok.sum())
    sl, ic = np.polyfit(t[ok], np.log(y[ok]), 1)
    return (-1.0 / sl if sl < 0 else np.nan), int(ok.sum())


def ideal_gas():
    files = sorted((OUT_IG / "raw").glob("prod_N*_traj*.npz"))
    byN = {}
    for f in files:
        d = load(f)
        byN.setdefault(d["meta"]["N"], []).append(d)
    res = {}
    figs = {}
    for N, runs in sorted(byN.items()):
        nt = len(runs)
        nmin = min(len(r["t"]) for r in runs)
        t = runs[0]["t"][:nmin]
        dt = runs[0]["meta"]["dt"]
        T = np.array([r["T"][:nmin] for r in runs])                     # (traj, time, 3)
        dP = np.array([r["dP"][:nmin] for r in runs])
        m, hw, se = tci(T, axis=0)
        taus = [relaxation_time(t, m[:, a] - 1.0, se[:, a]) for a in range(3)]
        An = T[:, :, 0] - 0.5 * (T[:, :, 1] + T[:, :, 2])                 # anisotropy, largest initial signal
        mA_, _, seA_ = tci(An)
        taus.append(relaxation_time(t, mA_, seA_))
        tau_fit = np.nanmax([x[0] for x in taus]) if np.isfinite([x[0] for x in taus]).any() else np.nan
        lam_star = float(np.nanmedian(np.concatenate([r["lam_min"][1:] for r in runs])))   # Ritz estimate
        t_slow = 3.0 / (2.0 * lam_star)                                    # 3 e-folds of the slowest momentum mode
        tau_rel = tau_fit
        t_burn = max(5 * tau_fit if np.isfinite(tau_fit) else 0.0, t_slow)
        win = t >= t_burn
        nwin = int(win.sum())
        if nwin < 50:
            res[str(N)] = dict(n_traj=nt, nsteps=nmin - 1, T_end=float(t[-1]), dt=dt,
                               momentum_rel_err_max=float(dP.max()), initial_T_mean=fl(T[:, 0, :].mean(axis=0)),
                               final_T_mean=fl(m[-1]), final_T_ci95=fl(hw[-1]),
                               tau_fit_components_and_anisotropy=[float(x[0]) for x in taus], tau_fit=float(tau_fit),
                               lam_star_ritz_median=lam_star, t_slow_3efold=float(t_slow), burn_in=float(t_burn),
                               equilibrium_window_available=False,
                               note="run shorter than the burn-in rule: relaxation only, equilibrium NOT established",
                               lam_max_ritz_est_range=[float(np.nanmin([np.nanmin(r["lam_max"]) for r in runs])),
                                                       float(np.nanmax([np.nanmax(r["lam_max"]) for r in runs]))],
                               n_actions_per_step=sorted({int(x) for r in runs for x in r["n_actions"][1:]
                                                          if np.isfinite(x)}))
            figs[N] = dict(t=t, m=m, hw=hw, t_burn=t_burn, allu=np.array([]))
            continue
        # per-trajectory window averages
        Tav = T[:, win, :].mean(axis=1)                                 # (traj, 3)
        Tkin_av = Tav.mean(axis=1)
        mA, hA, _ = tci(Tav)
        mK, hK, _ = tci(Tkin_av)
        # halves check
        idx = np.where(win)[0]
        h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
        dhalf = T[:, h2, :].mean(axis=1) - T[:, h1, :].mean(axis=1)
        mH, hH, _ = tci(dhalf)
        # Maxwell marginals from saved paths/inputs: use the pre-O momenta p of each step (in_p) after burn-in
        stride = 1          # every step after burn-in; the CI is over independent trajectories, not over snapshots
        var_t, kurt_t, nsnap = [], [], []
        pooled = []
        for r in runs:
            P = r["in_p"][: nmin - 1]                                    # p_n before step n (n = 0..)
            tt = t[: nmin - 1]
            sel = np.where(tt >= t_burn)[0][::stride]
            u = []
            for n in sel:
                p = P[n]
                u.append(((p - p.mean(axis=0)) / np.sqrt(sec6_config.model()["mass"] / sec6_config.model()["beta"]
                                                          * (N - 1) / N)).ravel())
            u = np.concatenate(u) if u else np.array([])
            pooled.append(u)
            nsnap.append(len(sel))
            if u.size:
                var_t.append(u.var())
                kurt_t.append(((u - u.mean()) ** 4).mean() / u.var() ** 2 - 3.0)
        mV, hV, _ = tci(var_t)
        mKu, hKu, _ = tci(kurt_t)
        allu = np.concatenate(pooled)
        ks = stats.kstest(allu, "norm")
        res[str(N)] = dict(
            n_traj=nt, nsteps=nmin - 1, T_end=float(t[-1]), dt=dt,
            momentum_rel_err_max=float(dP.max()),
            initial_T_mean=fl(T[:, 0, :].mean(axis=0)),
            tau_fit_components_and_anisotropy=[float(x[0]) for x in taus], tau_fit_points=[x[1] for x in taus],
            tau_fit=float(tau_fit), lam_star_ritz_median=lam_star, t_slow_3efold=float(t_slow),
            burn_in_rule="t_b = max(5 tau_fit, 3/(2 lambda*)); tau_fit from the initial contiguous decay of the "
                         "ensemble-mean component temperatures and anisotropy; lambda* = median Ritz estimate of the "
                         "smallest eigenvalue on T_perp over the run",
            burn_in=float(t_burn), equilibrium_window_available=bool(nwin >= 50),
            window=[float(t[win][0]), float(t[-1])] if nwin else None,
            window_steps=nwin,
            T_window_mean=fl(mA), T_window_ci95=fl(hA), Tkin_window_mean=float(mK), Tkin_window_ci95=float(hK),
            T_window_contains_1=[bool(abs(a - 1) <= h) for a, h in zip(mA, hA)],
            halves_diff_mean=fl(mH), halves_diff_ci95=fl(hH),
            halves_consistent=[bool(abs(a) <= h) for a, h in zip(mH, hH)],
            maxwell=dict(snapshot_stride_steps=stride, snapshots_per_traj=nsnap, var_mean=float(mV),
                         var_ci95=float(hV), exkurt_mean=float(mKu), exkurt_ci95=float(hKu),
                         ks_pooled_stat=float(ks.statistic), ks_pooled_n=int(allu.size),
                         ks_note="pooled KS is descriptive only: snapshots within a trajectory are correlated"),
            lam_max_ritz_est_range=[float(np.nanmin([np.nanmin(r["lam_max"]) for r in runs])),
                                    float(np.nanmax([np.nanmax(r["lam_max"]) for r in runs]))],
            tau_actual_est_range=[float(dt * np.nanmin([np.nanmin(r["lam_max"]) for r in runs])),
                                  float(dt * np.nanmax([np.nanmax(r["lam_max"]) for r in runs]))],
            n_actions_per_step=sorted({int(x) for r in runs for x in r["n_actions"][1:] if np.isfinite(x)}),
            wall_s_per_step_median=float(np.median([np.nanmedian(r["t_step"]) for r in runs])),
        )
        figs[N] = dict(t=t, m=m, hw=hw, t_burn=t_burn, allu=allu)
    # compact per-step observables of every trajectory (the raw trajectories are git-ignored)
    import gzip
    for N, runs in sorted(byN.items()):
        with gzip.open(OUT_IG / f"observables_N{N}.csv.gz", "wt") as fh:
            fh.write("traj,step,t,Tx,Ty,Tz,dP,lam_max_ritz_est,lam_min_ritz_est,n_actions\n")
            for r in runs:
                k = r["meta"]["traj"]
                for i in range(len(r["t"])):
                    fh.write(f"{k},{i},{r['t'][i]:.10g},{r['T'][i, 0]:.10g},{r['T'][i, 1]:.10g},{r['T'][i, 2]:.10g},"
                             f"{r['dP'][i]:.4g},{r['lam_max'][i]:.8g},{r['lam_min'][i]:.8g},{r['n_actions'][i]:.0f}\n")
    out = dict(provenance=sec6_config.provenance(SCRIPT, True), stats=res,
               trajectories={str(N): [dict(traj=r["meta"]["traj"], nsteps=r["meta"]["nsteps"],
                                           code_hash=r["meta"]["provenance"]["code_hash"],
                                           time=r["meta"]["provenance"]["time"]) for r in runs]
                             for N, runs in sorted(byN.items())})
    off = OUT_IG / "offline_ostep_checks.json"
    if off.exists():
        o = json.loads(off.read_text())
        rows = o["rows"]
        out["offline"] = dict(ranks=o["ranks"], n_steps=len(rows),
                              noise_rel_err_max=max(r["noise_rel_err"] for r in rows),
                              damp_rel_err_max=max(r["damp_rel_err"] for r in rows),
                              tau_range=[min(r["tau"] for r in rows), max(r["tau"] for r in rows)],
                              lam_max_range=[min(r["lam_max"] for r in rows), max(r["lam_max"] for r in rows)],
                              by_N={str(n): dict(noise=max(r["noise_rel_err"] for r in rows if f"_N{n}_" in r["file"]),
                                                 damp=max(r["damp_rel_err"] for r in rows if f"_N{n}_" in r["file"]),
                                                 tau_max=max(r["tau"] for r in rows if f"_N{n}_" in r["file"]))
                                    for n in sorted({int(r["file"].split("_N")[1].split("_")[0]) for r in rows})})
    (OUT_IG / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    # figure
    style()
    fig, axs = plt.subplots(1, len(figs) + 1, figsize=(7.2, 2.5), constrained_layout=True)
    for ax, (N, F) in zip(axs, sorted(figs.items())):
        for a, (col, lab) in enumerate(COMP):
            ax.fill_between(F["t"], F["m"][:, a] - F["hw"][:, a], F["m"][:, a] + F["hw"][:, a], color=col, alpha=0.18, lw=0)
            ax.plot(F["t"], F["m"][:, a], color=col, label=lab)
        ax.axhline(1, color=INK2, lw=0.7, ls="--")
        ax.axvline(F["t_burn"], color=INK2, lw=0.7, ls=":")
        ax.set_xlabel("$t$")
        ax.set_ylabel("component temperature (N-1 dof)")
        ax.set_title(f"N={N}, {res[str(N)]['n_traj']} traj. (mean, 95% CI)", loc="left", fontsize=7.5)
        ax.legend(loc="upper right")
    ax = axs[-1]
    xs = np.linspace(-4, 4, 200)
    for N, F in sorted(figs.items()):
        if not F["allu"].size:
            continue
        ax.hist(F["allu"], bins=60, range=(-4, 4), density=True, histtype="step", label=f"N={N}")
    ax.plot(xs, stats.norm.pdf(xs), color=INK, lw=0.8, ls="--", label="N(0,1)")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 0.6)
    ax.set_xlabel("standardised momentum component")
    ax.set_title("Maxwell marginal, after burn-in", loc="left", fontsize=7.5)
    ax.legend(loc="lower center")
    fig.savefig(OUT_IG / "fig_ideal_gas.png")
    return out


# ----------------------------------------------------------------------------
# paired comparisons
# ----------------------------------------------------------------------------
OBS = ("Tx", "Ty", "Tz", "Tkin", "Ok")


def series(r):
    T = r["T"]
    return dict(Tx=T[:, 0], Ty=T[:, 1], Tz=T[:, 2], Tkin=r["Tkin"], Ok=r["Ok"])


def path_diff(a, b, L):
    dq = a["path_q"] - b["path_q"]
    dq -= L * np.round(dq / L)
    n = min(len(dq), len(b["path_p"]))
    N = dq.shape[1]
    return (np.sqrt((dq[:n] ** 2).sum(axis=(1, 2)) / N), np.sqrt(((a["path_p"][:n] - b["path_p"][:n]) ** 2).sum(axis=(1, 2)) / N))


def paired_stats(A, B, L, stride_a=1, stride_b=1):
    """A, B: lists of runs (same trajectories). Returns per observable: max_t |mean diff|, its CI, RMS pathwise."""
    out = {}
    for o in OBS:
        da = []
        for a, b in zip(A, B):
            sa, sb = series(a)[o][::stride_a], series(b)[o][::stride_b]
            n = min(len(sa), len(sb))
            da.append(sa[:n] - sb[:n])
        n = min(len(x) for x in da)
        D = np.array([x[:n] for x in da])
        m, hw, se = tci(D) if len(D) > 1 else (D[0], np.full(n, np.nan), np.full(n, np.nan))
        k = int(np.nanargmax(np.abs(m)))
        out[o] = dict(max_abs_mean=float(np.abs(m).max()), at_t_index=k, ci95_at_max=float(hw[k]) if len(D) > 1 else None,
                      resolved_at_max=bool(abs(m[k]) > hw[k]) if len(D) > 1 else None,
                      final_mean=float(m[-1]), final_ci95=float(hw[-1]) if len(D) > 1 else None,
                      rms_path=float(np.sqrt((D ** 2).mean())), max_abs_path=float(np.abs(D).max()))
    if stride_a == 1 and stride_b == 1 and "path_q" in A[0] and "path_q" in B[0]:
        q, p = zip(*[path_diff(a, b, L) for a, b in zip(A, B)])
        n = min(len(x) for x in q)
        out["path_q_rms"] = dict(max=float(np.max([x[:n] for x in q])), final_median=float(np.median([x[n - 1] for x in q])))
        out["path_p_rms"] = dict(max=float(np.max([x[:n] for x in p])), final_median=float(np.median([x[n - 1] for x in p])))
    return out


def compare():
    raw = OUT_DY / "raw"
    res = {}
    prod_set = sec6_config.production()["spatial_set"]
    fine_set = sec6_config.production()["fine_spatial_set"]
    Ns = sorted({int(f.stem.split("_N")[1].split("_")[0]) for f in raw.glob(f"cmp_dense_{prod_set}_N*_traj*.npz")})
    for N in Ns:
        pick = lambda pat: {int(f.stem.split("traj")[1]): load(f) for f in sorted(raw.glob(pat))}  # noqa: E731
        dense = pick(f"cmp_dense_{prod_set}_N{N}_traj*.npz")
        hi = pick(f"cmp_hi_{prod_set}_N{N}_traj*.npz")
        fine = pick(f"cmp_dense_{fine_set}_N{N}_traj*.npz")
        prod = {int(f.stem.split("traj")[1]): load(f) for f in sorted((OUT_IG / "raw").glob(f"prod_N{N}_traj*.npz"))}
        L = sec6_config.box_length(N)
        ks = sorted(set(dense) & set(prod))
        if not ks:
            continue
        nst = len(dense[ks[0]]["t"])

        def cut(r):
            return {k: (v[:nst] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) >= nst else v)
                    for k, v in r.items() if k not in ("in_qh", "in_p", "in_xi")}
        R = {}
        R["prod_minus_dense"] = paired_stats([cut(prod[k]) for k in ks], [dense[k] for k in ks], L)
        if hi:
            kh = sorted(set(hi) & set(dense))
            R["hi_minus_dense"] = paired_stats([hi[k] for k in kh], [dense[k] for k in kh], L)
            R["prod_minus_hi"] = paired_stats([cut(prod[k]) for k in kh], [hi[k] for k in kh], L)
            R["hi_selfcheck_max"] = dict(noise20=float(max(np.nanmax(hi[k]["chk_noise20"]) for k in kh)),
                                         noise40=float(max(np.nanmax(hi[k]["chk_noise40"]) for k in kh)),
                                         damp20=float(max(np.nanmax(hi[k]["chk_damp20"]) for k in kh)),
                                         damp40=float(max(np.nanmax(hi[k]["chk_damp40"]) for k in kh)),
                                         hi_ranks=hi[kh[0]]["meta"]["hi_ranks"])
        if fine:
            kf = sorted(set(fine) & set(dense))
            R["spatial_fine_minus_prod_dense"] = paired_stats([fine[k] for k in kf], [dense[k] for k in kf], L)
        R["n_traj"] = len(ks)
        R["T_end"] = float(dense[ks[0]]["t"][-1])
        R["nsteps"] = nst - 1
        R["momentum_max"] = dict(dense=float(max(np.max(dense[k]["dP"]) for k in ks)),
                                 hi=float(max(np.max(hi[k]["dP"]) for k in hi)) if hi else None,
                                 fine=float(max(np.max(fine[k]["dP"]) for k in fine)) if fine else None)
        R["lam_max_dense_range"] = [float(min(np.nanmin(dense[k]["lam_max"]) for k in ks)),
                                    float(max(np.nanmax(dense[k]["lam_max"]) for k in ks))]
        res[f"N{N}"] = R
    # dt study
    dts = sorted(raw.glob("dt_N64_traj*.npz"))
    if dts:
        Z = [np.load(f) for f in dts]
        levels = json.loads(str(Z[0]["meta"]))["levels"]
        mk = lambda z, lev: dict(T=z[f"L{lev}_T"], Tkin=z[f"L{lev}_Tkin"], Ok=z[f"L{lev}_Ok"], t=z[f"L{lev}_t"])  # noqa: E731
        dtr = {}
        for a, b in [(1, 2), (2, 4)]:
            if a in levels and b in levels:
                A = [mk(z, a) for z in Z]
                B = [mk(z, b) for z in Z]
                st = {}
                for o in OBS:
                    D = np.array([series(x)[o] - series(y)[o][::b // a] for x, y in zip(A, B)])
                    m, hw, se = tci(D)
                    tt = A[0]["t"]
                    pts = {}
                    for frac in (0.25, 0.5, 1.0):
                        i = int(round(frac * (len(tt) - 1)))
                        pts[f"t={tt[i]:.3f}"] = dict(mean=float(m[i]), ci95=float(hw[i]), se=float(se[i]),
                                                     resolved=bool(abs(m[i]) > hw[i]))
                    k = int(np.argmax(np.abs(m)))
                    st[o] = dict(at_times=pts, max_abs_mean=float(np.abs(m[k])), ci95_at_max=float(hw[k]),
                                 n_resolved_times=int((np.abs(m) > hw).sum()), n_times=int(len(m)),
                                 rms_path=float(np.sqrt((D ** 2).mean())))
                dtr[f"dt/{a} - dt/{b}"] = st
        res["dt_study"] = dict(levels=levels, n_traj=len(Z), T_end=float(mk(Z[0], 1)["t"][-1]), pairs=dtr)
    # large
    lg = {}
    for f in sorted(raw.glob("large_prod_N*_traj*.npz")):
        g = Path(str(f).replace("large_prod", "large_hi"))
        if not g.exists():
            continue
        a, b = load(f), load(g)
        N = a["meta"]["N"]
        st = paired_stats([a], [b], a["meta"]["box"]["L"])
        st["hi_selfcheck_max"] = dict(noise20=float(np.nanmax(b["chk_noise20"])), noise40=float(np.nanmax(b["chk_noise40"])),
                                      damp20=float(np.nanmax(b["chk_damp20"])), damp40=float(np.nanmax(b["chk_damp40"])))
        st["ranks"] = dict(prod=a["meta"]["ranks"], hi=b["meta"]["ranks"])
        st["nsteps"] = a["meta"]["nsteps"]
        st["lam_max_est_range"] = [float(np.nanmin(b["lam_max"])), float(np.nanmax(b["lam_max"]))]
        st["tau_est_range"] = [float(a["meta"]["dt"] * x) for x in st["lam_max_est_range"]]
        st["momentum_max"] = dict(prod=float(np.max(a["dP"])), hi=float(np.max(b["dP"])))
        st["wall_s"] = dict(prod=float(a["wall_s"]), hi=float(b["wall_s"]))
        lg[f"N{N}_traj{a['meta']['traj']}"] = st
    if lg:
        res["large"] = lg
    out = dict(provenance=sec6_config.provenance(SCRIPT, True), **res)
    OUT_DY.mkdir(parents=True, exist_ok=True)
    (OUT_DY / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main():
    a = ideal_gas() if (OUT_IG / "raw").exists() else None
    b = compare() if (OUT_DY / "raw").exists() else None
    print(json.dumps(dict(ideal=a and a["stats"], compare=b and {k: v for k, v in b.items() if k != "provenance"}),
                     indent=1, default=float)[:6000])
