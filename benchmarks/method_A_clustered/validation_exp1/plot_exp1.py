#!/usr/bin/env python3
"""
Figure 1 (spatial accuracy validation of the Model A Ewald-PPPM) and supplementary figures, regenerated from the
saved results of run_exp1.py (nothing is recomputed here except least-squares guides drawn on the saved data).

    python3 plot_exp1.py
Outputs: figures/fig1_spatial_validation.{pdf,png}, figures/figS1_split_radial.{pdf,png},
         figures/figS2_degree_all.{pdf,png}, figures/figS3_action_all.{pdf,png}
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.ticker                 # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FAMS = ("uniform", "strong", "moderate")
FAM_LABEL = dict(uniform="uniform", strong="strongly clustered", moderate="moderately clustered")
FAM_COLOR = dict(uniform="#1f77b4", strong="#d62728", moderate="#2ca02c")
P_COLOR = {4: "#4c72b0", 6: "#55a868", 8: "#dd8452", 9: "#8172b3"}
U_ROUND = 2.0 ** -53

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


# ---------------------------------------------------------------------------- panels
def tex(x, digits=1):
    m, e = f"{x:.{digits}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def style_legend(ax, entries, **kw):
    hs = [plt.Line2D([], [], **e) for _, e in entries]
    return ax.legend(hs, [lab for lab, _ in entries], **kw)


def panel_a(ax):
    rows = load("a_split_configs.csv")
    radial = load("a_split_radial.csv")
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
    ax.set_title("(a) Analytical Ewald identity")
    leg = style_legend(ax, [(f, dict(color=FAM_COLOR[f], lw=2)) for f in FAMS],
                       loc="upper left", handlelength=1.4)
    ax.add_artist(leg)
    style_legend(ax, [("relative Frobenius, all pairs", dict(color="k", ls="-", marker="o", ms=3)),
                      ("max over pairs", dict(color="k", ls="--", marker="s", mfc="none", ms=3)),
                      (r"radial sweep $r\in[10^{-6},12]$", dict(color="k", ls=":", marker="^", mfc="none", ms=3))],
                 loc="upper right", handlelength=2.4)


def panel_b(ax):
    rows = load("b_shortrange.csv")
    summ = json.loads((RES / "b_shortrange_summary.json").read_text())
    for fam in FAMS:
        sub = [r for r in rows if r["family"] == fam]
        s = np.array(sorted({r["s"] for r in sub}))
        E = np.array([[r["E_S"] for r in sub if r["s"] == v] for v in s])
        ax.fill_between(s, E.min(1), E.max(1), color=FAM_COLOR[fam], alpha=0.25, lw=0)
        ax.plot(s, np.median(E, 1), "-", color=FAM_COLOR[fam], label=f"{FAM_LABEL[fam]}")
    s, B = series(rows, "s", "bound_eps_S", agg=np.median)
    ax.plot(s, B, "-.", color="0.35", lw=0.9, label=r"certified bound $\varepsilon_S/\|\Gamma_{\rm ref}\|_2$")
    s_all, Em = series(rows, "s", "E_S", agg=np.median)
    s0 = 3.0
    E0 = float(Em[np.argmin(abs(s_all - s0))])
    sg = np.linspace(2.0, 5.85, 100)
    ax.plot(sg, E0 * np.exp(-(sg ** 2 - s0 ** 2)), "k--", lw=0.9, label=r"$e^{-s^2}$ guide")
    ax.axvline(0.85 * 4.8878, color="0.5", ls=":", lw=0.9)
    ax.text(0.85 * 4.8878 + 0.06, 1.5e-2, r"production $s=4.15$", fontsize=6.5, color="0.35", va="top")
    c_lo, c_hi = summ["c_range_individual"]
    ax.text(0.97, 0.80, "fit $\\log E=a-cs^2$, $s\\in[2.0,5.3]$:\n" + rf"$c={c_lo:.3f}$–${c_hi:.3f}$ (9 configs)",
            transform=ax.transAxes, fontsize=6.5, ha="right", va="top")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 5e-2)
    ax.set_xlabel(r"$s=\xi r_c$  ($\xi=0.85$, $N=256$)")
    ax.set_ylabel(r"$E_S=\|\Gamma_S^{r_c}+\Gamma_L^{\rm ex}-\Gamma_{\rm ref}\|_2/\|\Gamma_{\rm ref}\|_2$")
    ax.set_title("(b) Short-range cutoff")
    ax.legend(loc="lower left", handlelength=2.2, title="median over 3 seeds, shading min–max",
              title_fontsize=6.5)


def panel_c(ax, name="strong_N512_s1", setting="production"):
    rows = [r for r in load("c_degree.csv") if r["config"] == name and r["setting"] == setting and r["valid"]]
    fits = {r["p"]: r["rate_mesh_lsq"] for r in load("c_degree_rate_fits.csv")
            if r["config"] == name and r["setting"] == setting}
    for p in (4, 6, 8, 9):
        M, E = series(rows, "M", "E_D_complete", p=p)
        ax.plot(M, E, "-o", color=P_COLOR[p], label=f"$p={p}$ (mesh rate {fits[p]:.1f})")
        M, E = series(rows, "M", "E_D_mesh", p=p)
        ax.plot(M, E, "--", color=P_COLOR[p], marker="o", mfc="none", lw=0.9)
    floor = rows[0]["E_D_floor"]
    ax.axhline(floor, color="k", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([28, 32, 40, 48, 64, 80, 96])
    ax.set_xticklabels(["28", "32", "40", "48", "64", "80", "96"])
    ax.minorticks_off()
    ax.set_ylim(1e-16, 1e-3)
    ax.set_xlabel(r"FFT grid size $M$ per dimension (strong, $N=512$)")
    ax.set_ylabel(r"degree error $E_D$")
    ax.set_title("(c) Degree convergence")
    leg = ax.legend(loc="lower left", handlelength=2.2, title="LSQ rate of mesh-transfer error", title_fontsize=6.5)
    ax.add_artist(leg)
    style_legend(ax, [(r"complete, vs $\Gamma_{\rm ref}$", dict(color="k", ls="-", marker="o", ms=3)),
                      (r"mesh transfer, vs $\Gamma_L^{k_c}$", dict(color="k", ls="--", marker="o", mfc="none", ms=3)),
                      (rf"fixed-cutoff floor ${tex(floor)}$", dict(color="k", ls=":"))],
                 loc="upper right", handlelength=2.2)


def panel_d(ax):
    rows = [r for r in load("d_action.csv") if r["valid"]]
    fits = load("d_action_rate_fits.csv")
    for fam in FAMS:
        for N, ls in ((128, "-"), (256, "--")):
            M, E = series(rows, "M", "E_gauss_max", family=fam, N=N, setting="strict")
            ax.plot(M, E, ls, color=FAM_COLOR[fam], marker="o", mfc="none" if N == 256 else FAM_COLOR[fam],
                    ms=3 if fam != "uniform" else 4.5, label=f"{FAM_LABEL[fam]}, $N={N}$")
            M, E = series(rows, "M", "E_gauss_max", family=fam, N=N, setting="production")
            ax.plot(M, E, ls, color=FAM_COLOR[fam], lw=0.7, alpha=0.6)
    prod_floor = max(r["E_op"] for r in rows if r["setting"] == "production")
    prod_g = [r["E_gauss_max"] for r in rows if r["setting"] == "production"]
    ax.axhline(prod_floor, color="k", ls=":", lw=0.9)
    ax.text(0.03, 0.985, "production cutoffs (thin): Gaussian " + rf"$\leq{tex(max(prod_g))}$;"
            + "\n" + rf"dense $E_{{\rm op}}\leq{tex(prod_floor)}$ (dotted: fixed-cutoff floor)",
            transform=ax.transAxes, fontsize=6.3, color="0.2", va="top")
    rates = [r["rate_lsq"] for r in fits if r["quantity"].startswith("E_gauss")]
    ax.text(0.97, 0.56, "strict cutoffs (thick):\nLSQ rate " + rf"${min(rates):.2f}$–${max(rates):.2f}$",
            transform=ax.transAxes, fontsize=6.5, ha="right", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([18, 20, 24, 32, 40, 48, 64, 80])
    ax.set_xticklabels(["18", "20", "24", "32", "40", "48", "64", "80"])
    ax.minorticks_off()
    ax.set_ylim(3e-15, 3e-6)
    ax.set_xlabel(r"FFT grid size $M$ per dimension ($p=9$)")
    ax.set_ylabel(r"$\max_X\,\|\Gamma_h X-\Gamma_{\rm ref}X\|_2/\|\Gamma_{\rm ref}X\|_2$")
    ax.set_title("(d) Matrix-free friction action")
    ax.legend(loc="lower left", ncol=1, handlelength=2.2, fontsize=6.3)


def fig1():
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.4))
    panel_a(axs[0, 0])
    panel_b(axs[0, 1])
    panel_c(axs[1, 0])
    panel_d(axs[1, 1])
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    save(fig, "fig1_spatial_validation")


# ---------------------------------------------------------------------------- supplementary
def figS1():
    radial = load("a_split_radial.csv")
    mp_rows = load("a_split_mpmath.csv")
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
    rows = [r for r in load("c_degree.csv")]
    fig, axs = plt.subplots(2, 3, figsize=(9.0, 5.6), sharey=True)
    for i, setting in enumerate(("production", "strict")):
        for j, N in enumerate((128, 256, 512)):
            ax = axs[i, j]
            name = f"strong_N{N}_s1"
            sub = [r for r in rows if r["config"] == name and r["setting"] == setting]
            for p in (4, 6, 8, 9):
                M, E = series([r for r in sub if r["valid"]], "M", "E_D_complete", p=p)
                ax.plot(M, E, "-o", color=P_COLOR[p], label=f"$p={p}$")
                M, E = series([r for r in sub if r["valid"]], "M", "E_D_mesh", p=p)
                ax.plot(M, E, "--", color=P_COLOR[p], marker="o", mfc="none", lw=0.8)
            if N == 256:
                for fam in ("uniform", "moderate"):
                    s2 = [r for r in rows if r["config"] == f"{fam}_N256_s1" and r["setting"] == setting and r["valid"]]
                    M, E = series(s2, "M", "E_D_complete", p=9)
                    ax.plot(M, E, "-", color=FAM_COLOR[fam], lw=0.7, label=f"{fam} $p=9$")
            inval = sorted({r["M"] for r in sub if not r["valid"]})
            for M in inval:
                ax.axvline(M, color="0.7", lw=4, alpha=0.4)
            ax.axhline(sub[0]["E_D_floor"], color="k", ls=":", lw=0.9)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.minorticks_off()
            Ms = sorted({r["M"] for r in sub})
            ax.set_xticks(Ms)
            ax.set_xticklabels([str(m) for m in Ms], fontsize=6)
            ax.set_title(f"strong N={N}, {setting}" + (" (grey: invalid M)" if inval else ""), fontsize=8)
            if j == 0:
                ax.set_ylabel(r"$E_D$ (solid complete, dashed mesh)")
            if i == 1:
                ax.set_xlabel("$M$")
            if i == 0 and j == 1:
                ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save(fig, "figS2_degree_all")


def figS3():
    rows = [r for r in load("d_action.csv") if r["valid"]]
    keys = (("E_op", "dense $E_{\\rm op}$"), ("E_gauss_max", "Gaussian (max of 5)"), ("E_spec_min", "spectral min"),
            ("E_spec_max", "spectral max"), ("E_slow_max", "slow modes (max of 4)"))
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 5.8), sharey="row")
    axs[0, 0].set_ylim(1.5e-8, 1.2e-7)
    for i, setting in enumerate(("production", "strict")):
        for j, N in enumerate((128, 256)):
            ax = axs[i, j]
            for k, (key, lab) in enumerate(keys):
                M, E = series(rows, "M", key, N=N, setting=setting)
                ax.plot(M, E, "-o", color=plt.get_cmap("tab10")(k), label=lab)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            Ms = sorted({r["M"] for r in rows if r["N"] == N})
            ax.set_xticks(Ms)
            ax.set_xticklabels([str(m) for m in Ms], fontsize=6.5)
            ax.set_title(f"N={N}, {setting} cutoffs, p=9 (max over 9 configs)", fontsize=7.5)
            ax.set_xlabel("$M$")
            if j == 0:
                ax.set_ylabel("relative error")
            if j == 0:
                ax.legend(fontsize=6, ncol=2 if i == 0 else 1)
    fig.tight_layout()
    save(fig, "figS3_action_all")


if __name__ == "__main__":
    fig1()
    figS1()
    figS2()
    figS3()
    print("figures written to", FIG)
