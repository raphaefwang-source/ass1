#!/usr/bin/env python3
"""
Figure 1 (accuracy validation of the periodic Ewald-PPPM for Model A) and its supplementary figures, regenerated from
the saved results of run_exp1.py. Nothing is recomputed here; rates quoted in the figures are read from the saved
least-squares fits (results/action_rate_fits.csv, results/action_p_rate_fits.csv).

    python3 plot_exp1.py
Outputs: figures/fig1_spatial_validation.{pdf,png}      (a) configurations, (b) split identity, (c) short-range
                                                        cutoff, (d) matrix-free PPPM action
         figures/figS1_split_radial.{pdf,png}           radial sweep and small-r cancellation for (b)
         figures/figS2_action_detail.{pdf,png}          (d) for every N, cutoff setting and probe class, with dense E_op
         figures/figS3_translation_residual.{pdf,png}   translation null space of Gamma_h, Gamma_ref
         figures/figS4_action_p.{pdf,png}               mesh-only action error for p = 4, 6, 8, 9 (N = 512)
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
    ax.text(6.68, fl_hi * 2.4, "reference /\nroundoff floor", ha="right", va="bottom", fontsize=5.7, color="0.25")
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


def panel_action(ax1, ax2):
    rows = [r for r in load("action_errors.csv")]
    mesh = load("action_mesh.csv")
    fits = load("action_rate_fits.csv")
    # (d, left) production cutoffs, N = 512 strongly clustered: complete vs mesh-only
    prod = [r for r in rows if r["N"] == 512 and r["setting"] == "production"]
    for cls in ("gaussian", "spectral_slow"):
        c = CLS_COLOR[cls]
        M, Emax = class_curve(prod, "E_complete", cls)
        _, Emin = class_curve(prod, "E_complete", cls, agg=min)
        ax1.fill_between(M, Emin, Emax, color=c, alpha=0.18, lw=0)
        ax1.plot(M, Emax, "-o", color=c, ms=3)
        M, Emax = class_curve(prod, "E_mesh", cls)
        _, Emin = class_curve(prod, "E_mesh", cls, agg=min)
        ax1.fill_between(M, Emin, Emax, color=c, alpha=0.12, lw=0)
        ax1.plot(M, Emax, "--o", color=c, mfc="none", ms=3)
    inval = [m for m in mesh if m["N"] == 512 and m["setting"] == "production" and not m["valid"]]
    for m in inval:
        ax1.axvspan(m["M"] * 0.985, m["M"] * 1.015, color="0.75", lw=0)
    ax1.text(26.0, 3e-11, "Nyquist-invalid $M=26$", fontsize=5.6, color="0.3", ha="center", va="center", rotation=90)
    ax1.axvline(40, color="0.4", ls=":", lw=0.9)
    ax1.text(41.0, 6e-15, "production $M=40$", fontsize=6.0, color="0.3", va="bottom", rotation=90)
    fg = [f for f in fits if f["N"] == 512 and f["setting"] == "production" and f["probe_class"] == "gaussian"
          and f["quantity"].startswith("E_mesh")][0]
    ax1.text(0.97, 0.55, f"mesh-only, Gaussian: LSQ rate {fg['rate_lsq']:.1f}\n(M = {fg['M_lo']}–{fg['M_hi']}, "
             f"$R^2$ = {fg['r2']:.4f})", transform=ax1.transAxes, fontsize=6.2, ha="right", va="top")
    log_m_axis(ax1, [28, 32, 40, 48, 64, 80, 96])
    ax1.set_ylim(3e-15, 2e-5)
    ax1.set_xlabel(r"FFT grid size $M$ per dimension")
    ax1.set_ylabel("relative action error\n(maximum over tested probes/configurations)")
    ax1.set_title(r"production cutoffs, $N=512$ strongly clustered", fontsize=7.8, fontweight="normal")
    leg = style_legend(ax1, [(CLS_LABEL[c], dict(color=CLS_COLOR[c], lw=2)) for c in ("gaussian", "spectral_slow")],
                       loc="upper left", bbox_to_anchor=(0.06, 1.0), handlelength=1.4, fontsize=6.4)
    ax1.add_artist(leg)
    style_legend(ax1, [(r"complete $E_{\rm complete}$ (vs $\Gamma_{\rm ref}$)", dict(color="k", ls="-", marker="o", ms=3)),
                       (r"mesh-only $E_{\rm mesh}$ (vs $\Gamma_L^{k_c}$)", dict(color="k", ls="--", marker="o", mfc="none",
                                                                             ms=3))],
                 loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=2.2, fontsize=6.4)
    # (d, right) strict cutoffs: complete error for every size, maximum over tested probes and configurations
    st = [r for r in rows if r["setting"] == "strict"]
    for N, ls, lw in ((128, "-", 0.9), (256, "--", 0.9), (512, "-", 1.8)):
        for cls in ("gaussian", "spectral_slow"):
            M, E = class_curve(st, "E_complete", cls, N=N)
            ax2.plot(M, E, ls, color=CLS_COLOR[cls], lw=lw, marker="o" if N == 512 else None, ms=2.8,
                     alpha=1.0 if N == 512 else 0.75)
    fg = [f for f in fits if f["setting"] == "strict" and f["probe_class"] == "gaussian" and f["quantity"].startswith(
        "E_complete")]
    ax2.text(0.97, 0.97, "LSQ rate (Gaussian, max over probes/configs):\n" + ", ".join(
        f"$N={f['N']}$: {f['rate_lsq']:.1f}" for f in fg), transform=ax2.transAxes, fontsize=6.0, ha="right", va="top")
    trunc = max(m["E_op_trunc"] for m in mesh if m["setting"] == "strict")
    ax2.axhline(trunc, color="0.45", ls=":", lw=0.9)
    ax2.text(19.5, trunc * 1.4, rf"strict truncation floor $E_{{\rm op}}\leq{tex(trunc)}$", fontsize=5.8, color="0.3")
    log_m_axis(ax2, [20, 24, 32, 40, 48, 64, 80, 96])
    ax2.set_ylim(3e-15, 2e-5)
    ax2.set_xlabel(r"FFT grid size $M$ per dimension")
    ax2.set_title(r"strict cutoffs ($r_c=6/\xi$, $k_c\leq12\xi$): $E_{\rm complete}$", fontsize=7.8,
                  fontweight="normal")
    style_legend(ax2, [("$N=128$ (9 configs)", dict(color="k", ls="-", lw=0.9)),
                       ("$N=256$ (9 configs)", dict(color="k", ls="--", lw=0.9)),
                       ("$N=512$ strongly clustered", dict(color="k", ls="-", lw=1.8, marker="o", ms=2.8))],
                 loc="lower left", bbox_to_anchor=(0.0, 0.07), handlelength=2.2, fontsize=6.4)


CONFIG_VIEWS = (("uniform_N256_s1", "uniform"), ("moderate_N256_s1", "moderately clustered"),
                ("strong_N256_s1", "strongly clustered"), ("strong_N512_s1", "strongly clustered"))


def load_config(name):
    for d in (HERE / "configs", HERE.parent / "configs", HERE.parent / "configs_scaling"):
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
        ax.set_box_aspect((1, 1, 1), zoom=1.08)
        ax.view_init(elev=18, azim=-58)
        ax.set_axis_off()
        N = len(x)
        ax.set_title(f"{lab}\n$N={N}$, $L={L:.2f}$" + ("  (main test of (d))" if N == 512 else ""), fontsize=6.8,
                     fontweight="normal", pad=-2)


def fig1():
    fig = plt.figure(figsize=(7.4, 8.6))
    top = GridSpec(1, 4, figure=fig, left=0.06, right=0.98, top=0.935, bottom=0.765, wspace=0.02)
    axs3 = [fig.add_subplot(top[0, k], projection="3d") for k in range(4)]
    panel_configs(axs3)
    mid = GridSpec(1, 2, figure=fig, left=0.10, right=0.97, top=0.705, bottom=0.425, wspace=0.36)
    panel_split(fig.add_subplot(mid[0, 0]))
    panel_shortrange(fig.add_subplot(mid[0, 1]))
    bot = GridSpec(1, 2, figure=fig, left=0.10, right=0.97, top=0.305, bottom=0.05, wspace=0.22)
    ax1 = fig.add_subplot(bot[0, 0])
    ax2 = fig.add_subplot(bot[0, 1])
    panel_action(ax1, ax2)
    fig.text(0.5, 0.985, "(a) Particle configurations (periodic box, $\\rho=0.75$, periodic $d_{\\min}\\geq0.9$)",
             ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    fig.text(0.5, 0.335, r"(d) Matrix-free PPPM action convergence ($\xi$, $r_c$, $k_c$, $p=9$ fixed; only $M$ varies)",
             ha="center", va="bottom", fontsize=9.5, fontweight="bold")
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
                M, E = class_curve(sub, "E_mesh", cls)
                ax.plot(M, E, "--o", color=CLS_COLOR[cls], mfc="none", ms=2.6, lw=0.9)
            M, E = series([m for m in mesh if m["setting"] == setting and m["N"] == N], "M", "E_op")
            ax.plot(M, E, ":", color="k", lw=1.0)
            Ms = sorted({r["M"] for r in sub})
            log_m_axis(ax, Ms)
            ax.tick_params(axis="x", labelsize=6)
            cfgs = sorted({r["config"] for r in sub})
            ax.set_title(f"N={N}, {setting} cutoffs, p=9 ({len(cfgs)} config{'s' if len(cfgs) > 1 else ''})", fontsize=7.5)
            if j == 0:
                ax.set_ylabel("relative error\n(maximum over tested probes/configurations)", fontsize=7)
            if i == 1:
                ax.set_xlabel("$M$")
    axs[0, 0].set_ylim(3e-15, 3e-6)
    style_legend(axs[0, 0], [(CLS_LABEL[c], dict(color=CLS_COLOR[c], lw=2)) for c in CLS_COLOR]
                 + [("complete (solid)", dict(color="k", ls="-")), ("mesh-only (dashed)", dict(color="k", ls="--")),
                    (r"dense $E_{\rm op}$ (dotted)", dict(color="k", ls=":"))], loc="lower left", fontsize=5.8)
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
    rows = load("action_p.csv")
    fits = load("action_p_rate_fits.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, setting in zip(axs, ("production", "strict")):
        for p in (4, 6, 8, 9):
            sub = [r for r in rows if r["setting"] == setting and r["p"] == p]
            M, E = class_curve(sub, "E_mesh", "gaussian")
            f = [q for q in fits if q["setting"] == setting and q["p"] == p and q["probe_class"] == "gaussian"]
            lab = f"$p={p}$" + (f", LSQ rate {f[0]['rate_lsq']:.1f} (M {f[0]['M_lo']}–{f[0]['M_hi']})" if f else "")
            ax.plot(M, E, "--o", color=P_COLOR[p], mfc="none", ms=2.8, label=lab)
            M, E = class_curve(sub, "E_complete", "gaussian")
            ax.plot(M, E, "-", color=P_COLOR[p], lw=0.9)
        log_m_axis(ax, sorted({r["M"] for r in rows}))
        ax.tick_params(axis="x", labelsize=6)
        ax.set_ylim(3e-15, 1e-4)
        ax.set_xlabel("$M$")
        ax.set_title(f"N=512 strongly clustered, {setting} cutoffs", fontsize=8)
        ax.legend(fontsize=5.8, loc="lower left", title="dashed mesh-only, solid complete", title_fontsize=5.8)
    axs[0].set_ylabel("Gaussian probes: relative action error\n(maximum over 10 probes)", fontsize=7.5)
    fig.tight_layout()
    save(fig, "figS4_action_p")


if __name__ == "__main__":
    fig1()
    figS1()
    figS2()
    figS3()
    figS4()
    print("figures written to", FIG)
