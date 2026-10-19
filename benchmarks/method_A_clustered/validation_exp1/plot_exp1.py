#!/usr/bin/env python3
"""
Figure 1 (accuracy validation of the periodic Ewald-PPPM for Model A) and its supplementary figures, regenerated from
the saved results of run_exp1.py. Nothing is recomputed here; rates quoted in the figures are read from the saved
least-squares fits (results/action_rate_fits.csv, results/action_p_rate_fits.csv).

    python3 plot_exp1.py
Outputs: figures/fig1_spatial_validation.{pdf,png}      (a) configurations, (b) split identity, (c) short-range
                                                        cutoff, (d) matrix-free PPPM action
         figures/figS1_split_radial.{pdf,png}           radial sweep and small-r cancellation for (b)
         figures/figS2_action_detail.{pdf,png}          (d) complete error for every N, both cutoff settings, all probe
                                                        classes, with dense E_op
         figures/figS3_translation_residual.{pdf,png}   translation null space of Gamma_h, Gamma_ref
         figures/figS4_action_p.{pdf,png}               complete action error for p = 4, 6, 8, 9 (N = 512, strict)
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FAMS = ("uniform", "strong", "moderate")
FAM_LABEL = dict(uniform="uniform", strong="strongly clustered", moderate="moderately clustered")
FAM_COLOR = dict(uniform="#1f77b4", strong="#d62728", moderate="#2ca02c")
CLS_COLOR = dict(gaussian="#1f4e9c", spectral_slow="#c0392b", spectral_fast="#2e8b57")
CLS_LABEL = dict(gaussian="Gaussian", spectral_slow="slow spectral", spectral_fast="fast spectral")
P_COLOR = {4: "#4c72b0", 6: "#55a868", 8: "#dd8452", 9: "#8172b3"}
U_ROUND = 2.0 ** -53
S_PROD = 0.85 * 4.88775618844242
FLOOR_FACTOR = 2.0                       # (b): a point is resolved if the exact truncation tail >= 2 x measured floor

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.labelsize": 8.5,
    "legend.fontsize": 7.0, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "lines.linewidth": 1.2,
    "lines.markersize": 3.5, "axes.linewidth": 0.7, "legend.frameon": False, "savefig.dpi": 300,
})


def conv(v):
    if v == "" or v is None:
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def load(name):
    with open(RES / name) as fh:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def series(rows, xkey, ykey, agg=max, **sel):
    sub = [r for r in rows if all(r.get(k) == v for k, v in sel.items()) and r.get(ykey) is not None]
    xs = sorted({r[xkey] for r in sub})
    return np.array(xs, float), np.array([agg([r[ykey] for r in sub if r[xkey] == x]) for x in xs], float)


def tex(x, digits=1):
    m, e = f"{x:.{digits}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def style_legend(ax, entries, **kw):
    hs = [plt.Line2D([], [], **e) for _, e in entries]
    return ax.legend(hs, [lab for lab, _ in entries], **kw)


def log_m_axis(ax, ticks):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


# ---------------------------------------------------------------------------- (b) split identity
def panel_split(ax):
    rows = load("split_configs.csv")
    radial = load("split_radial.csv")
    for fam in FAMS:
        x, y = series(rows, "xi", "rel_fro", family=fam)
        ax.plot(x, y, "-o", color=FAM_COLOR[fam], ms=3 if fam != "uniform" else 4.5,
                mfc=FAM_COLOR[fam] if fam != "uniform" else "none")
        x, y = series(rows, "xi", "max_rel_pointwise", family=fam)
        ax.plot(x, y, "--s", color=FAM_COLOR[fam], mfc="none")
    x, y = series(radial, "xi", "rel_pointwise")
    ax.plot(x, y, ":^", color="k", mfc="none")
    ax.axhline(U_ROUND, color="0.5", lw=0.8, ls="-.")
    ax.text(1.49, U_ROUND * 1.15, r"$u=2^{-53}$", ha="right", va="bottom", fontsize=7, color="0.4")
    ax.set_yscale("log")
    ax.set_ylim(5e-17, 2e-13)
    ax.set_xlabel(r"splitting parameter $\xi$")
    ax.set_ylabel(r"relative error of $K-K_S-K_L$")
    ax.set_title("(b) Analytical Ewald splitting")
    leg = style_legend(ax, [(f, dict(color=FAM_COLOR[f], lw=2)) for f in FAMS], loc="upper left", ncol=3,
                       handlelength=1.4, columnspacing=1.0)
    ax.add_artist(leg)
    style_legend(ax, [("relative Frobenius, all pairs", dict(color="k", ls="-", marker="o", ms=3)),
                      ("max over pairs", dict(color="k", ls="--", marker="s", mfc="none", ms=3)),
                      (r"radial sweep $r\in[10^{-6},12]$", dict(color="k", ls=":", marker="^", mfc="none", ms=3))],
                 loc="upper left", bbox_to_anchor=(0.0, 0.89), handlelength=2.4)


# ---------------------------------------------------------------------------- (c) short-range cutoff
def shortrange_split():
    """Measured E_S per configuration, split into the resolved decay (exact truncation tail >= FLOOR_FACTOR x the
    configuration's measured floor) and the points at the reference / round-off floor. No value is altered."""
    rows = load("shortrange.csv")
    floor = {r["config"]: r["floor"] for r in load("shortrange_fits.csv")}
    for r in rows:
        r["resolved"] = r["E_trunc_exact"] >= FLOOR_FACTOR * floor[r["config"]]
    return rows, floor


def panel_shortrange(ax):
    rows, floor = shortrange_split()
    summ = json.loads((RES / "shortrange_summary.json").read_text())
    s_res = []
    for fam in FAMS:
        sub = [r for r in rows if r["family"] == fam]
        s_all = sorted({r["s"] for r in sub})
        s_ok = [s for s in s_all if all(r["resolved"] for r in sub if r["s"] == s)]
        E = np.array([[r["E_S"] for r in sub if r["s"] == s] for s in s_ok])
        ax.fill_between(s_ok, E.min(1), E.max(1), color=FAM_COLOR[fam], alpha=0.25, lw=0)
        ax.plot(s_ok, np.median(E, 1), "-", color=FAM_COLOR[fam], label=f"{FAM_LABEL[fam]}")
        fl = [r for r in sub if r["s"] not in s_ok]
        ax.plot([r["s"] for r in fl], [r["E_S"] for r in fl], "o", ms=3.2, mfc="none", mew=0.8,
                color=FAM_COLOR[fam], alpha=0.85)
        s_res.append(max(s_ok))
    s_hi = min(s_res)
    fl_lo, fl_hi = min(floor.values()), max(floor.values())
    ax.axhspan(fl_lo, fl_hi, color="0.6", alpha=0.35, lw=0)
    ax.text(6.68, fl_hi * 4.0, "reference /\nroundoff floor", ha="right", va="bottom", fontsize=5.7, color="0.25")
    s, B = series(rows, "s", "bound_eps_S", agg=np.median)
    keep = B >= 1e-16
    ax.plot(s[keep], B[keep], "-.", color="0.35", lw=0.9)
    s_all, Em = series(rows, "s", "E_S", agg=np.median)
    s0 = 3.0
    E0 = float(Em[np.argmin(abs(s_all - s0))])
    sg = np.linspace(2.0, s_hi, 100)
    ax.plot(sg, E0 * np.exp(-(sg ** 2 - s0 ** 2)), "k--", lw=0.9)
    ax.axvline(S_PROD, color="0.5", ls=":", lw=0.9)
    ax.text(S_PROD + 0.06, 1.4e-16, rf"production $s={S_PROD:.3f}$", fontsize=6.0, color="0.35", va="bottom", ha="left")
    c_lo, c_hi = summ["c_range_individual"]
    ax.text(0.97, 0.70, rf"$\log E_S=a-cs^2$: $c={c_lo:.3f}$–${c_hi:.3f}$" + "\n" + r"(LSQ, $s\in[2.0,5.3]$, 9 configs)",
            transform=ax.transAxes, fontsize=6.3, ha="right", va="top")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 5e-2)
    ax.set_xlim(1.9, 6.7)
    ax.set_xlabel(r"$s=\xi r_c$  ($\xi=0.85$, $N=256$)")
    ax.set_ylabel(r"$E_S=\|\Gamma_S^{r_c}+\Gamma_L^{\rm exact}-\Gamma_{\rm ref}\|_2/\|\Gamma_{\rm ref}\|_2$")
    ax.set_title("(c) Short-range cutoff error")
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.10), handlelength=2.0,
                    title="measured $E_S$ (median, band: 3 seeds)", title_fontsize=6.2)
    ax.add_artist(leg)
    style_legend(ax, [("measured at floor", dict(color="0.3", ls="none", marker="o", mfc="none", ms=3.2)),
                      ("certified bound (theory)", dict(color="0.35", ls="-.")),
                      (r"$e^{-s^2}$ guide", dict(color="k", ls="--"))],
                 loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=2.0)


# ---------------------------------------------------------------------------- (d) PPPM action
def class_curve(rows, key, cls, agg=max, **sel):
    return series(rows, "M", key, agg=agg, probe_class=cls, **sel)


N_STYLE = {128: dict(ls="-", lw=0.9), 256: dict(ls="--", lw=0.9), 512: dict(ls="-", lw=1.6)}
FAM_MARKER = dict(uniform="o", moderate="^", strong="s")


def family_curves(ax, rows, key, setting):
    """Gaussian probes, one curve per (family, N): maximum over the 10 probes (and the 3 seeds for N = 128, 256)."""
    for fam in FAMS:
        for N, st in N_STYLE.items():
            M, E = series(rows, "M", key, probe_class="gaussian", family=fam, N=N, setting=setting)
            if len(M):
                ax.plot(M, E, color=FAM_COLOR[fam], marker=FAM_MARKER[fam], ms=3.6 if N == 512 else 3.0,
                        mfc=FAM_COLOR[fam] if N == 512 else "none", mew=0.8, alpha=0.9, **st)


def panel_action(ax):
    rows = load("action_errors.csv")
    fits = [q for q in load("action_rate_fits.csv") if q["setting"] == "strict" and q["family"] != "all"
            and q["probe_class"] == "gaussian" and q["quantity"].startswith("E_complete")]
    family_curves(ax, rows, "E_complete", "strict")
    lo, hi = min(q["rate_lsq"] for q in fits), max(q["rate_lsq"] for q in fits)
    ax.text(0.97, 0.97, f"LSQ rate {lo:.1f}–{hi:.1f}\n(each family and $N$, $R^2\\geq{min(q['r2'] for q in fits):.3f}$)",
            transform=ax.transAxes, fontsize=6.3, ha="right", va="top")
    ax.text(0.03, 0.97, r"$\xi=0.85$, $r_c=6/\xi$, $k_c\leq12\xi$, $p=9$ fixed;" + "\nonly $M$ varies",
            transform=ax.transAxes, fontsize=6.3, ha="left", va="top", color="0.25")
    log_m_axis(ax, [18, 24, 32, 40, 48, 64, 80, 96])
    ax.set_ylim(3e-15, 3e-7)
    ax.set_xlabel(r"FFT grid size $M$ per dimension")
    ax.set_ylabel(r"$\|(\Gamma_h-\Gamma_{\rm ref})X\|_2/\|\Gamma_{\rm ref}X\|_2$" + "\n(maximum over tested probes/configurations)")
    ax.set_title("(d) Matrix-free PPPM action")
    leg = style_legend(ax, [(FAM_LABEL[f], dict(color=FAM_COLOR[f], lw=1.2, marker=FAM_MARKER[f], ms=3.6, mfc="none"))
                            for f in FAMS], loc="lower left", handlelength=2.0, fontsize=6.4)
    ax.add_artist(leg)
    style_legend(ax, [("$N=128$ (open markers)", dict(color="k", **N_STYLE[128])),
                      ("$N=256$ (open markers)", dict(color="k", **N_STYLE[256])),
                      ("$N=512$ (filled markers)", dict(color="k", marker="s", ms=3.4, **N_STYLE[512]))],
                 loc="upper right", bbox_to_anchor=(1.0, 0.86), handlelength=2.2, fontsize=6.2)


CONFIG_VIEWS = (("uniform_N256_s1", "uniform"), ("moderate_N256_s1", "moderately clustered"),
                ("strong_N256_s1", "strongly clustered"), ("strong_N512_s1", "strongly clustered"))


def load_config(name):
    for d in (HERE / "configs",):
        if (d / f"{name}.npz").exists():
            with np.load(d / f"{name}.npz") as z:
                return np.mod(z["q"], float(z["L"])), float(z["L"])
    raise FileNotFoundError(name)


def panel_configs(axs):
    """Schematic of the particle configurations: positions in the periodic box (depth shaded)."""
    edges = [((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 0, 1)), ((1, 1, 0), (0, 1, 0)),
             ((1, 1, 0), (1, 0, 0)), ((1, 1, 0), (1, 1, 1)), ((1, 0, 1), (0, 0, 1)), ((1, 0, 1), (1, 1, 1)),
             ((1, 0, 1), (1, 0, 0)), ((0, 1, 1), (0, 0, 1)), ((0, 1, 1), (1, 1, 1)), ((0, 1, 1), (0, 1, 0))]
    for ax, (name, lab) in zip(axs, CONFIG_VIEWS):
        x, L = load_config(name)
        fam = name.split("_")[0]
        u = x / L
        depth = (u[:, 0] + u[:, 1]) / 2 - u[:, 2] * 0.3
        a = 0.35 + 0.65 * (depth - depth.min()) / np.ptp(depth)
        cols = np.tile(matplotlib.colors.to_rgba(FAM_COLOR[fam]), (len(u), 1))
        cols[:, 3] = a
        ax.scatter(u[:, 0], u[:, 1], u[:, 2], s=5 if len(u) <= 256 else 3, c=cols, edgecolors="none", depthshade=False)
        for p0, p1 in edges:
            ax.plot(*zip(p0, p1), color="0.45", lw=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_box_aspect((1, 1, 1), zoom=1.12)
        ax.view_init(elev=18, azim=-58)
        ax.set_axis_off()
        N = len(x)
        ax.set_title(f"{lab}, $N={N}$", fontsize=6.4, fontweight="normal", pad=-4)


def fig1():
    fig = plt.figure(figsize=(7.4, 6.9))
    gs = GridSpec(2, 2, figure=fig, left=0.09, right=0.98, top=0.95, bottom=0.07, hspace=0.36, wspace=0.34)
    sub = gs[0, 0].subgridspec(2, 2, wspace=0.0, hspace=0.22)
    axs3 = [fig.add_subplot(sub[k // 2, k % 2], projection="3d") for k in range(4)]
    panel_configs(axs3)
    pos = gs[0, 0].get_position(fig)
    fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.012, "(a) Particle configurations", ha="center", va="bottom",
             fontsize=9.5, fontweight="bold")
    panel_split(fig.add_subplot(gs[0, 1]))
    panel_shortrange(fig.add_subplot(gs[1, 0]))
    panel_action(fig.add_subplot(gs[1, 1]))
    save(fig, "fig1_spatial_validation")


# ---------------------------------------------------------------------------- supplementary
def figS1():
    radial = load("split_radial.csv")
    mp_rows = load("split_mpmath.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.0))
    cm = plt.get_cmap("viridis")
    xis = sorted({r["xi"] for r in radial})
    for k, xi in enumerate(xis):
        sub = [r for r in radial if r["xi"] == xi]
        r = np.array([q["r"] for q in sub])
        e = np.maximum(np.array([q["rel_pointwise"] for q in sub]), 1e-18)
        axs[0].plot(r, e, ".", ms=1.6, color=cm(k / max(len(xis) - 1, 1)), label=rf"$\xi={xi}$")
    axs[0].axhline(U_ROUND, color="0.5", ls="-.", lw=0.8)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_ylim(5e-19, 1e-14)
    axs[0].set_xlabel("$r$")
    axs[0].set_ylabel(r"$|\theta-\theta_S-\theta_L|/\theta$")
    axs[0].set_title("identity, radial sweep (exact 0 shown at $10^{-18}$)", fontsize=8)
    axs[0].legend(ncol=2, markerscale=4, loc="upper left")
    for k, xi in enumerate(xis):
        sub = [q for q in mp_rows if q["xi"] == xi]
        r = np.array([q["r"] for q in sub])
        axs[1].plot(r, np.maximum([q["err_naive_L_rel_self"] for q in sub], 1e-18), "-", color=cm(k / max(len(xis) - 1, 1)),
                    lw=0.9)
        axs[1].plot(r, np.maximum([q["err_theta_L_rel_self"] for q in sub], 1e-18), "o", ms=2.5, mfc="none",
                    color=cm(k / max(len(xis) - 1, 1)))
    axs[1].plot([], [], "k-", label=r"naive $\theta-\theta_S$ (cancellation)")
    axs[1].plot([], [], "ko", mfc="none", label=r"$\theta_L$ by its own $s$-quadrature")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_ylim(5e-19, 1e-1)
    axs[1].set_xlabel("$r$")
    axs[1].set_ylabel(r"rel. error of $\theta_L$ (50-digit mpmath)")
    axs[1].set_title("small-$r$ cancellation in $K-K_S$", fontsize=8)
    axs[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "figS1_split_radial")


def figS2():
    rows = load("action_errors.csv")
    mesh = [m for m in load("action_mesh.csv") if m["valid"]]
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.8), sharey=True)
    for i, setting in enumerate(("production", "strict")):
        for j, N in enumerate((128, 256, 512)):
            ax = axs[i, j]
            sub = [r for r in rows if r["setting"] == setting and r["N"] == N]
            for cls in ("gaussian", "spectral_slow", "spectral_fast"):
                M, E = class_curve(sub, "E_complete", cls)
                ax.plot(M, E, "-o", color=CLS_COLOR[cls], ms=2.6)
            M, E = series([m for m in mesh if m["setting"] == setting and m["N"] == N], "M", "E_op")
            ax.plot(M, E, ":", color="k", lw=1.0)
            Ms = sorted({r["M"] for r in sub})
            log_m_axis(ax, Ms)
            ax.tick_params(axis="x", labelsize=6)
            cfgs = sorted({r["config"] for r in sub})
            ax.set_title(f"N={N}, {setting} cutoffs, p=9 ({len(cfgs)} configs)", fontsize=7.5)
            if j == 0:
                ax.set_ylabel("complete error vs $\\Gamma_{\\rm ref}$\n(maximum over tested probes/configurations)", fontsize=7)
            if i == 1:
                ax.set_xlabel("$M$")
    axs[0, 0].set_ylim(3e-15, 3e-6)
    style_legend(axs[0, 0], [(CLS_LABEL[c], dict(color=CLS_COLOR[c], lw=2)) for c in CLS_COLOR]
                 + [(r"dense $E_{\rm op}$", dict(color="k", ls=":"))], loc="lower left", fontsize=6)
    fig.tight_layout()
    save(fig, "figS2_action_detail")


def figS3():
    tr = load("action_translation.csv")
    zero = 1.5e-18                           # exact zeros are drawn on this line (log axis)
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    for ax, setting in zip(axs, ("production", "strict")):
        for N, c in ((128, "#7f7f7f"), (256, "#1f77b4"), (512, "#d62728")):
            sub = [r for r in tr if r["setting"] == setting and r["N"] == N]
            M, E = series(sub, "M", "opnorm_pairdiff")
            ax.plot(M, np.maximum(E, zero), "-o", color=c, ms=2.6, label=f"$N={N}$ pair-difference (production)")
            M, E = series(sub, "M", "opnorm_cached")
            ax.plot(M, np.maximum(E, zero), "--s", color=c, mfc="none", ms=2.6, lw=0.9, label=f"$N={N}$ cached $D-H$")
            ax.axhline(max(r["opnorm_ref"] for r in sub), color=c, ls=":", lw=0.9)
        ax.axhline(U_ROUND, color="0.5", ls="-.", lw=0.8)
        ax.axhline(zero, color="0.8", lw=3, zorder=0)
        ax.text(19, zero * 1.4, "exact 0", fontsize=6, color="0.4")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        Ms = sorted({r["M"] for r in tr})
        ax.set_xticks(Ms[::2])
        ax.set_xticklabels([str(m) for m in Ms[::2]], fontsize=6.5)
        ax.set_ylim(1e-18, 1e-14)
        ax.set_xlabel("$M$")
        ax.set_title(f"translation probes, {setting} cutoffs", fontsize=8)
    axs[0].set_ylabel(r"$\max_a\|\Gamma_h T_a\|_2/\|\Gamma_{\rm ref}\|_2$" + "\n(3 probes, maximum over configurations)",
                      fontsize=7.5)
    h, lab = axs[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="k", ls=":"), plt.Line2D([], [], color="0.5", ls="-.")]
    lab += [r"$\Gamma_{\rm ref}$ itself (dotted, per $N$)", r"$u=2^{-53}$"]
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=6.0, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "figS3_translation_residual")


def figS4():
    rows = [r for r in load("action_p.csv") if r["setting"] == "strict"]
    fits = [q for q in load("action_p_rate_fits.csv") if q["setting"] == "strict" and q["quantity"] == "E_complete"
            and q["probe_class"] == "gaussian"]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for p in (4, 6, 8, 9):
        M, E = class_curve([r for r in rows if r["p"] == p], "E_complete", "gaussian")
        f = [q for q in fits if q["p"] == p][0]
        ax.plot(M, E, "-o", color=P_COLOR[p], ms=2.8,
                label=f"$p={p}$: LSQ rate {f['rate_lsq']:.1f} (M {f['M_lo']}–{f['M_hi']}, $R^2$={f['r2']:.4f})")
    log_m_axis(ax, sorted({r["M"] for r in rows}))
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_ylim(3e-15, 1e-4)
    ax.set_xlabel("$M$")
    ax.set_ylabel("Gaussian probes: complete error\n(maximum over 10 probes)", fontsize=7.5)
    ax.set_title("N=512 strongly clustered, strict cutoffs", fontsize=8)
    ax.legend(fontsize=5.8, loc="lower left")
    fig.tight_layout()
    save(fig, "figS4_action_p")


if __name__ == "__main__":
    fig1()
    figS1()
    figS2()
    figS3()
    figS4()
    print("figures written to", FIG)
