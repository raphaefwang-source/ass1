"""
Publication figures for accuracy_sweeps/ and candidate2_pilot/, in the style of Fig. 1 of Zhao, Dou & Lei, "A fast
conservative particle method for the free-space Landau equation" (SIAM preprint): 2 x 2 panels with bold centred
titles, serif (STIX) fonts, boxed axes with inward ticks, light grid, framed in-panel legends, colour = family or
candidate, filled markers / solid lines vs open markers / dashed lines for the two quantities of a panel, shaded
min-max range over seeds, black dotted comparison guides and dotted, labelled thresholds.

Every figure is regenerated from the CSV files in results/ (no recomputation):
    accuracy_sweeps/figures/figA1_gamma_operator_6sweeps (E_G), figA2_gamma_action_6sweeps (E_GX): 2 x 3, the six
        one-at-a-time sweeps with identical panels and axes; figS1_reference_structure (supplementary);
        figA_plotted_series.csv (the plotted medians / ranges, for checking)
    candidate2_pilot/figures/fig3_candidate_comparison, fig4_psd_cost, figS2_pilot_all_kappa
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

FAMS = ("uniform", "moderate", "strong")
FAM_COL = {"uniform": "#2a78d6", "moderate": "#eb6834", "strong": "#1baf7a"}
FAM_MK = {"uniform": "o", "moderate": "s", "strong": "^"}
CAND_COL = {0.0: "#2a78d6", 0.5: "#eb6834"}
CAND_MK = {0.0: "o", 0.5: "s"}
CAND_LAB = {0.0: r"Candidate 1 ($\theta=0$)", 0.5: r"Candidate 2 ($\theta=1/2$)"}
KAP_COL = {0.2: "#1baf7a", 0.5: "#eda100", 0.7: "#6250d6", 1.0: "#2a78d6", 2.0: "#e34948"}
KAP_MK = {0.2: "^", 0.5: "v", 0.7: "o", 1.0: "s", 2.0: "D"}
INK, GREY, GRID = "#111111", "#6b6b6b", "#dcdcdc"


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.3,
        "axes.linewidth": 0.7, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "axes.grid": True, "axes.grid.which": "major", "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "#b8b8b8", "legend.fancybox": False,
        "legend.borderpad": 0.4, "legend.handlelength": 2.0, "legend.labelspacing": 0.3,
        "lines.linewidth": 1.1, "lines.markersize": 3.8, "lines.markeredgewidth": 0.9,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def stats(rows, xkey, ykey, keep=lambda r: True):
    """x values and median / min / max of ykey over the rows sharing each x."""
    xs = sorted({r[xkey] for r in rows if keep(r) and r.get(ykey) is not None})
    med, lo, hi = [], [], []
    for x in xs:
        v = np.array([r[ykey] for r in rows if r[xkey] == x and keep(r) and r.get(ykey) is not None], float)
        med.append(np.median(v)), lo.append(v.min()), hi.append(v.max())
    return np.array(xs, float), np.array(med), np.array(lo), np.array(hi)


def curve(ax, x, med, lo, hi, color, marker, dashed=False, label=None, band=True, ms=None):
    if band:
        ax.fill_between(x, lo, hi, color=color, alpha=0.16 if not dashed else 0.10, lw=0)
    kw = {} if ms is None else dict(ms=ms)
    ax.plot(x, med, "--" if dashed else "-", color=color, marker=marker, mfc="white" if dashed else color,
            mec=color, label=label, **kw)


def guide(ax, x, y, label):
    ax.plot(x, y, ":", color=INK, lw=1.1, label=label)


def threshold(ax, y, text, x_text=0.98, above=True, ha="right"):
    ax.axhline(y, color=INK, ls=(0, (1, 1.5)), lw=0.8)
    ax.annotate(text, xy=(x_text, y), xycoords=("axes fraction", "data"), ha=ha,
                va="bottom" if above else "top", fontsize=6.3, xytext=(0, 1.5 if above else -1.5),
                textcoords="offset points")


def anchored(xs, f, x0, y0, factor=3.0):
    """Guide c f(x) through factor * y0 at x0."""
    return f(xs) * factor * y0 / f(x0)


def fam_handles(extra=()):
    return [Line2D([], [], color=FAM_COL[f], marker=FAM_MK[f], label=f) for f in FAMS] + list(extra)


# ============================================================================ Experiment 1
# Panel definitions shared by Fig. A1 (E_G) and Fig. A2 (E_GX): one one-at-a-time sweep per panel, x read from the
# recorded parameters (k_c/xi = 2 sF_eff, the shell-snapped value actually used, never a relabelled request).
A_PANELS = (
    ("sR", "(a) Real-space cutoff", r"$\xi r_c$", lambda r: r["value"], False),
    ("sF", "(b) Fourier cutoff", r"$k_c/\xi$  (shell-snapped)", lambda r: 2.0 * r["sF_eff"], False),
    ("M", "(c) FFT mesh size", r"$M$", lambda r: r["value"], True),
    ("p", "(d) B-spline assignment order", r"$p$", lambda r: r["value"], False),
    ("rho", "(e) Particle density", r"$\rho$  ($L=(N/\rho)^{1/3}$)", lambda r: r["value"], False),
    ("kappa", "(f) Screening parameter", r"$\kappa$  ($c=1$)", lambda r: r["value"], True),
)
A_SETTINGS = (("baseline", False), ("moderate", True))     # solid / filled vs dashed / open
A_METRICS = {
    "E_G": dict(floor="ref_floor_G", fig="figA1_gamma_operator_6sweeps",
                ylabel=r"$E_\Gamma=\|\Gamma_h-\Gamma_{\rm ref}\|_2\,/\,\|\Gamma_{\rm ref}\|_2$",
                floor_label=r"reference floor $\|\Gamma_{LF}-\Gamma_D\|_2/\|\Gamma_D\|_2$"),
    "E_GX": dict(floor="ref_floor_GX", fig="figA2_gamma_action_6sweeps",
                 ylabel=r"$E_{\Gamma X}=\|(\Gamma_h-\Gamma_{\rm ref})X\|_F\,/\,\|\Gamma_{\rm ref}X\|_F$",
                 floor_label=r"reference floor $\|(\Gamma_{LF}-\Gamma_D)X\|_F/\|\Gamma_D X\|_F$"),
}
A_YLIM = (1e-16, 1e-1)
A_FAM_LAB = {"uniform": "uniform", "moderate": "moderately clustered", "strong": "strongly clustered"}
A_FAM_MS = {"uniform": 5.4, "moderate": 4.2, "strong": 3.0}    # decreasing sizes: overlapping families stay visible


def six_sweeps(rows, metric, figures, series):
    """2 x 3 figure of one metric over the six sweeps; the plotted medians / ranges are appended to `series`."""
    spec = A_METRICS[metric]
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.0), sharey=True)
    for ax, (sw, title, xl, xfun, logx) in zip(axs.ravel(), A_PANELS):
        pr = [dict(r, x=xfun(r)) for r in rows if r["sweep"] == sw]
        settings = [(st, d) for st, d in A_SETTINGS if any(r["setting"] == st for r in pr)]
        for st, dashed in settings:
            for fam in FAMS:
                fr = [r for r in pr if r["setting"] == st and r["family"] == fam]
                x, m, lo, hi = stats(fr, "x", metric)
                curve(ax, x, m, lo, hi, FAM_COL[fam], FAM_MK[fam], dashed=dashed, ms=A_FAM_MS[fam])
                for k in range(len(x)):
                    n = sum(1 for r in fr if r["x"] == x[k])
                    series.append(dict(figure=spec["fig"], panel=title[:3], sweep=sw, metric=metric, setting=st,
                                       family=fam, x=x[k], median=m[k], min=lo[k], max=hi[k], n_seeds=n))
        # reference floor: one line where the physical configuration is fixed, per-x values where it changes
        fx, fm, _, _ = stats(pr, "x", spec["floor"])
        if sw in ("rho", "kappa"):
            ax.plot(fx, fm, ls=(0, (1, 1.5)), color=INK, lw=0.8, marker="x", ms=3.5, mew=0.8)
        else:
            ax.axhline(float(np.median(fm)), color=INK, ls=(0, (1, 1.5)), lw=0.8)
        for k in range(len(fx)):
            series.append(dict(figure=spec["fig"], panel=title[:3], sweep=sw, metric=spec["floor"], setting="reference",
                               family="all", x=fx[k], median=fm[k], min=None, max=None, n_seeds=None))
        # theoretical Gaussian decay guides (not measured data), anchored 3x above the baseline median at s = 3
        base = [r for r in pr if r["setting"] == "baseline"]
        if sw == "sR":
            y0 = np.median([r[metric] for r in base if r["x"] == 3.0])
            g = np.linspace(2.0, 5.75, 60)
            guide(ax, g, anchored(g, lambda s: np.exp(-s ** 2), 3.0, y0), None)
        if sw == "sF":
            u0 = min({r["x"] for r in base}, key=lambda u: abs(u - 6.0))
            y0 = np.median([r[metric] for r in base if r["x"] == u0])
            g = np.linspace(min(r["x"] for r in base), 11.5, 60)
            guide(ax, g, anchored(g, lambda u: np.exp(-u ** 2 / 4), u0, y0), None)
            lo_u, hi_u = min(r["x"] for r in base), max(r["x"] for r in base)
            ax.annotate(fr"computed $k_c/\xi$: {lo_u:.2f} – {hi_u:.2f}", xy=(0.03, 0.10), xycoords="axes fraction",
                        fontsize=6.2, color=GREY)
        if sw in ("sR", "sF"):
            ax.annotate("baseline only", xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                        fontsize=6.2, color=GREY)
        ax.set_yscale("log"), ax.set_ylim(*A_YLIM)
        ax.set_yticks([1e-16, 1e-13, 1e-10, 1e-7, 1e-4, 1e-1])
        if sw == "M":
            ax.set_xscale("log"), ax.set_xticks([16, 24, 32, 48, 64, 80])
            ax.set_xticklabels(["16", "24", "32", "48", "64", "80"]), ax.minorticks_off()
        elif sw == "kappa":
            ax.set_xscale("log"), ax.set_xticks([0.2, 0.5, 0.7, 1.0, 2.0])
            ax.set_xticklabels(["0.2", "0.5", "0.7", "1", "2"]), ax.minorticks_off()
        elif sw == "sF":
            ax.set_xticks([4, 6, 8, 10, 12])
        elif sw == "p":
            ax.set_xticks(range(3, 10))
        elif sw == "rho":
            ax.set_xticks([0.3, 0.5, 0.75, 1.0])
        ax.set_xlabel(xl), ax.set_title(title)
    for ax in axs[:, 0]:
        ax.set_ylabel(spec["ylabel"])
    for ax in axs[:, 1:].ravel():
        ax.tick_params(labelleft=False)
    floor_h = Line2D([], [], color=INK, ls=(0, (1, 1.5)), lw=0.8, marker="x", ms=3.5, mew=0.8,
                     label=spec["floor_label"] + "\n" + r"($\times$: separate value for each $\rho$, $\kappa$)")
    h = [Line2D([], [], color=FAM_COL[f], marker=FAM_MK[f], ms=A_FAM_MS[f], label=A_FAM_LAB[f]) for f in FAMS] + [
        Line2D([], [], color=GREY, marker="o", label=r"baseline: $\xi r_c=6$, $k_c/\xi=12$, $M=80$, $p=9$"),
        Line2D([], [], color=GREY, ls="--", marker="o", mfc="white",
               label=r"moderate: $\xi r_c=4$, $k_c/\xi=8$, $M=40$, $p=7$"),
        Line2D([], [], color=INK, ls=":", lw=1.1, label=r"guides $e^{-(\xi r_c)^2}$ (a), $e^{-(k_c/\xi)^2/4}$ (b)"),
        floor_h]
    order = [0, 3, 1, 4, 2, 5, 6]      # column-major legend: families | settings | guide, floor
    fig.legend(handles=[h[i] for i in order], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005),
               columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(h_pad=1.0, w_pad=0.6, rect=(0, 0.085, 1, 1))
    save(fig, figures / spec["fig"])


def exp1(results, figures, read_csv):
    """Fig. A1 (E_G), Fig. A2 (E_GX) over the six sweeps, and Fig. S1 (references, probe classes, PSD, momentum,
    cost). Plotted medians / ranges are written to figures/figA_plotted_series.csv for checking."""
    style()
    rows = [r for r in read_csv(results / "sweeps.csv") if r.get("mesh_valid")]
    refs = read_csv(results / "references.csv")
    conv = read_csv(results / "reference_convergence.csv")
    series = []
    for metric in ("E_G", "E_GX"):
        six_sweeps(rows, metric, figures, series)
    keys = ("figure", "panel", "sweep", "metric", "setting", "family", "x", "median", "min", "max", "n_seeds")
    with open(figures / "figA_plotted_series.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for s_ in series:
            fh.write(",".join("" if s_[k] is None else (repr(float(s_[k])) if isinstance(s_[k], (float, np.floating))
                                                         else str(s_[k])) for k in keys) + "\n")
    for r in rows:
        r["x"] = r["value"]
    floor = float(np.median([r["ref_floor_GX"] for r in rows if r["sweep"] == "sR"]))

    # ---------------- Fig. S1: reference convergence, probe classes, PSD, momentum / symmetry, cost
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.0))
    ax = axs[0, 0]
    for kap in (0.2, 0.5, 0.7, 1.0, 2.0):
        cs = [c for c in conv if c["kappa"] == kap and c["rho"] == 0.75 and c["diff_to_RD_rel"] > 0]
        R_D = cs[0]["R_D"]
        x, m, lo, hi = stats(cs, "R", "diff_to_RD_rel")
        curve(ax, x, m, lo, hi, KAP_COL[kap], KAP_MK[kap], label=fr"$\kappa={kap:g}$, $R_D={R_D:g}$")
        x, m, _, _ = stats(cs, "R", "cert_bound_rel")
        ax.plot(x, m, "--", color=KAP_COL[kap], lw=0.8, mfc="white", marker=KAP_MK[kap], ms=2.6)
    ax.axhline(1e-13, color=INK, ls=(0, (1, 1.5)), lw=0.8)
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_ylim(1e-23, 1e-3)
    ax.set_yticks([1e-19, 1e-15, 1e-11, 1e-7, 1e-3])
    ax.set_xlabel(r"direct-sum radius $R$")
    ax.set_ylabel(r"$\|\Gamma_D(R)-\Gamma_D(R_D)\|_2/\|\Gamma_D\|_2$")
    ax.set_title("(a) Direct periodic reference")
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color=GREY, ls="--", marker="o", mfc="white", ms=2.6, lw=0.8),
          Line2D([], [], color=INK, ls=(0, (1, 1.5)), lw=0.8)]
    l += ["certified tail bound", r"certification $10^{-13}$"]
    ax.legend(h, l, loc="lower left", ncol=2, columnspacing=0.6, handlelength=1.4, fontsize=5.4)
    ax = axs[0, 1]
    tols = (1e-10, 1e-12, 1e-14)
    for kap in (0.2, 0.5, 0.7, 1.0, 2.0):
        rr = [r for r in refs if r["kappa"] == kap]
        med = [np.median([r[f"LF_{t:g}_vs_D_rel"] for r in rr]) for t in tols]
        lo = [min(r[f"LF_{t:g}_vs_D_rel"] for r in rr) for t in tols]
        hi = [max(r[f"LF_{t:g}_vs_D_rel"] for r in rr) for t in tols]
        curve(ax, np.array(tols), np.array(med), np.array(lo), np.array(hi), KAP_COL[kap], KAP_MK[kap],
              label=fr"$\kappa={kap:g}$")
    ax.set_xscale("log"), ax.set_yscale("log"), ax.invert_xaxis()
    ax.set_xlabel("LatticeFriction rel_tol"), ax.set_ylabel(r"$\|\Gamma_{LF}-\Gamma_D\|_2/\|\Gamma_D\|_2$")
    ax.set_title("(b) Two independent references")
    ax.legend(loc="upper right", ncol=2)
    ax = axs[0, 2]
    rs = [r for r in rows if r["sweep"] == "sR"]
    for key, lab, ls, mk in (("E_GX", r"Gaussian $X$ (10 vectors)", "-", "o"),
                             ("E_GX_slow_max", "3 slowest eigenvectors", "--", "s"),
                             ("E_GX_fast", "fastest eigenvector", "-.", "^")):
        x, m, lo, hi = stats(rs, "x", key)
        ax.fill_between(x, lo, hi, color=GREY, alpha=0.12, lw=0)
        ax.plot(x, m, ls, color=INK, marker=mk, mfc="white" if ls != "-" else INK, label=lab)
    threshold(ax, floor, "reference floor (Gaussian $X$)", x_text=0.02, ha="left", above=True)
    ax.set_yscale("log"), ax.set_ylim(2e-16, 0.3)
    ax.set_xlabel(r"$\xi r_c$  (baseline)")
    ax.set_ylabel(r"$\|(\Gamma_h-\Gamma_{\rm ref})X\|/\|\Gamma_{\rm ref}X\|$")
    ax.set_title("(c) Action error by probe class")
    ax.legend(loc="upper right")
    ax = axs[1, 0]
    for fam in FAMS:
        fr = [r for r in rows if r["family"] == fam]
        ax.plot([r["E_G"] for r in fr], [abs(r["lam_min_h"] / r["lam_min_ref"] - 1) for r in fr], FAM_MK[fam],
                color=FAM_COL[fam], ms=2.6, alpha=0.65, ls="none", label=fam)
    ax.plot([1e-16, 1], [1e-16, 1], ls=(0, (1, 1.5)), color=INK, lw=0.8, label=r"$y=x$")
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(5e-16, 0.1), ax.set_ylim(1e-17, 0.1)
    ax.set_xlabel(r"$E_\Gamma$"), ax.set_ylabel(r"$|\lambda_{\min}(\Gamma_h)/\lambda_{\min}(\Gamma_{\rm ref})-1|$")
    ax.set_title(r"(d) Smallest eigenvalue on Range($\Pi$)")
    ax.legend(loc="upper left")
    ax = axs[1, 1]
    for fam in FAMS:
        fr = [r for r in rows if r["family"] == fam]
        ax.plot([r["E_G"] for r in fr], [r["mom_opnorm_max"] for r in fr], FAM_MK[fam], color=FAM_COL[fam], ms=2.6,
                alpha=0.6, ls="none")
        ax.plot([r["E_G"] for r in fr], [r["sym_resid"] for r in fr], FAM_MK[fam], color=FAM_COL[fam], ms=2.6,
                mfc="white", alpha=0.6, ls="none")
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_ylim(5e-19, 1e-15)
    ax.set_xlabel(r"$E_\Gamma$"), ax.set_ylabel("residual")
    ax.set_title("(e) Momentum and symmetry")
    ax.legend(handles=[Line2D([], [], color=GREY, marker="o", ls="none", label=r"$\|\Gamma_h t\|/\|\Gamma_{\rm ref}\|$"),
                       Line2D([], [], color=GREY, marker="o", mfc="white", ls="none",
                              label=r"$\|\Gamma_h-\Gamma_h^T\|_F/\|\Gamma_h\|_F$")], loc="upper left")
    ax = axs[1, 2]
    for st, dashed in A_SETTINGS:
        rs = [r for r in rows if r["sweep"] == "M" and r["setting"] == st and r.get("t_matvec_s")]
        for key, col, mk in (("t_matvec_s", "#2a78d6", "o"), ("t_setup_s", "#eb6834", "s")):
            x, m, lo, hi = stats(rs, "x", key)
            curve(ax, x, 1e3 * m, 1e3 * lo, 1e3 * hi, col, mk, dashed=dashed)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xticks([16, 24, 32, 48, 64, 80]), ax.set_xticklabels(["16", "24", "32", "48", "64", "80"])
    ax.minorticks_off()
    ax.set_xlabel(r"FFT mesh size $M$"), ax.set_ylabel("time [ms]")
    ax.set_title("(f) Cost per configuration")
    ax.legend(handles=[Line2D([], [], color="#2a78d6", marker="o", label="cached matvec"),
                       Line2D([], [], color="#eb6834", marker="s", label="setup"),
                       Line2D([], [], color=GREY, label="baseline"),
                       Line2D([], [], color=GREY, ls="--", marker="o", mfc="white", label="moderate")],
              loc="upper left", ncol=2)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    save(fig, figures / "figS1_reference_structure")


# ============================================================================ Experiment 2
def exp2(results, figures, read_csv):
    style()
    rows = [r for r in read_csv(results / "pilot.csv") if r.get("E_G") is not None]
    for r in rows:
        r["x"] = r["sF_eff"] if r["sweep"] == "sF" else r["value"]
        r["floor"] = max(r["ref_floor_G"], r["E_G_mesh_only"], 1e-15)
    rat = read_csv(results / "ratios.csv")
    for r in rat:
        r["x"] = r["sF_eff"] if r["sweep"] == "sF" else r["value"]
        a, b = r["E_GX_slow_max_theta0"], r["E_GX_slow_max_theta0.5"]
        r["ratio_slow"] = a / b if min(a, b) > 10 * r["floor"] else None
    floor = lambda kap: float(np.median([r["floor"] for r in rows if r["kappa"] == kap]))

    def cand_panel(ax, kap, sw, legend=True):
        rs = [r for r in rows if r["kappa"] == kap and r["sweep"] == sw]
        for th in (0.0, 0.5):
            tr = [r for r in rs if r["theta"] == th]
            curve(ax, *stats(tr, "x", "E_G"), CAND_COL[th], CAND_MK[th])
            curve(ax, *stats(tr, "x", "E_GX"), CAND_COL[th], CAND_MK[th], dashed=True, band=False)
        x = np.linspace(2, 5.7, 50)
        y0 = np.median([r["E_G"] for r in rs if abs(r["x"] - 3.0) < 0.13 and r["theta"] == 0.0])
        guide(ax, x, anchored(x, lambda s: np.exp(-s ** 2), 3.0, y0), None)
        threshold(ax, floor(kap), "floor: max(reference, mesh-only)", x_text=0.02, ha="left", above=False)
        ax.set_yscale("log"), ax.set_ylim(1e-16, 0.3)
        if legend:
            ax.legend(handles=[Line2D([], [], color=CAND_COL[t], marker=CAND_MK[t], label=CAND_LAB[t]) for t in (0.0, 0.5)]
                      + [Line2D([], [], color=GREY, label=r"$E_\Gamma$"),
                         Line2D([], [], color=GREY, ls="--", marker="o", mfc="white", label=r"$E_{\Gamma X}$ (Gaussian)"),
                         Line2D([], [], color=INK, ls=":", label=r"$e^{-s^2}$ guide")], loc="upper right")

    # ---------------- Fig. 3: candidate comparison
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.2))
    cand_panel(axs[0, 0], 0.7, "sR")
    axs[0, 0].set_xlabel(r"$s_R=\xi r_c$  ($s_F=6.5$)"), axs[0, 0].set_ylabel("relative error")
    axs[0, 0].set_title(r"(a) Short-range cutoff, $\kappa=0.7$")
    cand_panel(axs[0, 1], 0.7, "sF", legend=False)
    axs[0, 1].set_xlabel(r"$s_F=k_c/(2\xi)$  ($s_R=6.5$)"), axs[0, 1].set_ylabel("relative error")
    axs[0, 1].set_title(r"(b) Fourier cutoff, $\kappa=0.7$")
    ax = axs[1, 0]
    for kap in (0.2, 0.7, 2.0):
        for sw, dashed in (("sR", False), ("sF", True)):
            rs = [r for r in rat if r["kappa"] == kap and r["sweep"] == sw and r.get("ratio_E_G") is not None]
            curve(ax, *stats(rs, "x", "ratio_E_G"), KAP_COL[kap], KAP_MK[kap], dashed=dashed)
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.axhline(2.0, color=INK, ls=(0, (1, 1.5)), lw=0.8)
    ax.set_ylim(0.85, 3.0)
    ax.set_xlabel(r"swept cutoff $s$"), ax.set_ylabel(r"$E_\Gamma(\theta=0)\,/\,E_\Gamma(\theta=1/2)$")
    ax.set_title("(c) Spectral-error improvement ratio")
    ax.legend(handles=[Line2D([], [], color=KAP_COL[k], marker=KAP_MK[k], label=fr"$\kappa={k:g}$") for k in (0.2, 0.7, 2.0)]
              + [Line2D([], [], color=GREY, label=r"$s_R$ sweep"),
                 Line2D([], [], color=GREY, ls="--", marker="o", mfc="white", label=r"$s_F$ sweep"),
                 Line2D([], [], color=INK, ls=(0, (1, 1.5)), lw=0.8, label="asymptotic tail ratio 2")],
              loc="upper right", ncol=2)
    ax = axs[1, 1]
    for kap in (0.2, 0.7, 2.0):
        for key, dashed in (("ratio_slow", False), ("ratio_E_GX", True)):
            rs = [r for r in rat if r["kappa"] == kap and r["sweep"] == "sR" and r.get(key) is not None]
            curve(ax, *stats(rs, "x", key), KAP_COL[kap], KAP_MK[kap], dashed=dashed)
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.annotate(r"ratio $<1$: Candidate 2 worse", xy=(0.98, 0.97), xycoords=("axes fraction", "data"), ha="right",
                va="top", fontsize=6.3)
    ax.set_yscale("log"), ax.set_ylim(0.2, 5)
    ax.set_yticks([0.25, 0.5, 1, 2, 4]), ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"]), ax.minorticks_off()
    ax.set_xlabel(r"$s_R=\xi r_c$  ($s_F=6.5$)"), ax.set_ylabel(r"$E_{\Gamma X}(\theta=0)\,/\,E_{\Gamma X}(\theta=1/2)$")
    ax.set_title("(d) Action-error ratio by probe class")
    ax.legend(handles=[Line2D([], [], color=KAP_COL[k], marker=KAP_MK[k], label=fr"$\kappa={k:g}$") for k in (0.2, 0.7, 2.0)]
              + [Line2D([], [], color=GREY, label="3 slowest eigenvectors"),
                 Line2D([], [], color=GREY, ls="--", marker="o", mfc="white", label="Gaussian $X$")],
              loc="lower right", ncol=2)
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    save(fig, figures / "fig3_candidate_comparison")

    # ---------------- Fig. 4: complete-operator PSD and equal-accuracy cost
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.2))
    ax = axs[0, 0]
    for kap, dashed in ((0.7, False), (2.0, True)):
        for th in (0.0, 0.5):
            rs = [r for r in rows if r["kappa"] == kap and r["sweep"] == "sR" and r["theta"] == th]
            for r in rs:
                r["dlam"] = r["lam_min_h"] / r["lam_min_ref"] - 1
            curve(ax, *stats(rs, "x", "dlam"), CAND_COL[th], CAND_MK[th], dashed=dashed)
    ax.axhline(0.0, color=INK, lw=0.8)
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_yticks([-1e-1, -1e-4, -1e-7, 0, 1e-7, 1e-4, 1e-1])
    ax.set_yticklabels(["$-10^{-1}$", "$-10^{-4}$", "$-10^{-7}$", "0", "$10^{-7}$", "$10^{-4}$", "$10^{-1}$"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    n_ops = sum(1 for r in rows)
    n_neg = sum(1 for r in rows if r["n_negative_h"] > 0)
    ax.annotate(f"negative eigenvalues: {n_neg} of {n_ops} operators", xy=(0.97, 0.04), xycoords="axes fraction",
                ha="right", fontsize=6.3)
    ax.set_xlabel(r"$s_R=\xi r_c$  ($s_F=6.5$)")
    ax.set_ylabel(r"$\lambda_{\min}(\Gamma_h)/\lambda_{\min}(\Gamma_{ref})-1$")
    ax.set_title(r"(a) Complete $\Gamma_h$ on Range($\Pi$), no repair")
    ax.legend(handles=[Line2D([], [], color=CAND_COL[t], marker=CAND_MK[t], label=CAND_LAB[t]) for t in (0.0, 0.5)]
              + [Line2D([], [], color=GREY, label=r"$\kappa=0.7$"),
                 Line2D([], [], color=GREY, ls="--", marker="o", mfc="white", label=r"$\kappa=2$")],
              loc="upper right")
    ax = axs[0, 1]
    for th in (0.0, 0.5):
        tr = [r for r in rows if r["theta"] == th]
        ax.plot([r["E_G"] for r in tr], [r["mom_opnorm_max"] for r in tr], CAND_MK[th], color=CAND_COL[th], ms=2.4,
                alpha=0.55, ls="none")
        ax.plot([r["E_G"] for r in tr], [r["sym_resid"] for r in tr], CAND_MK[th], color=CAND_COL[th], ms=2.4,
                mfc="white", alpha=0.55, ls="none")
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_ylim(5e-19, 2e-16)
    ax.set_xlabel(r"$E_\Gamma$"), ax.set_ylabel("residual")
    ax.set_title("(b) Momentum and symmetry residuals")
    ax.legend(handles=[Line2D([], [], color=CAND_COL[t], marker=CAND_MK[t], ls="none", label=CAND_LAB[t]) for t in (0.0, 0.5)]
              + [Line2D([], [], color=GREY, marker="o", ls="none", label=r"$\|\Gamma_h t\|/\|\Gamma_{ref}\|$"),
                 Line2D([], [], color=GREY, marker="o", mfc="white", ls="none", label="symmetry")], loc="upper left", ncol=2)
    es = read_csv(results / "eqacc_summary.csv")
    pt = read_csv(results / "eqacc_paired_timing.csv")
    tols = sorted({r["tol"] for r in es}, reverse=True)
    ax = axs[1, 0]
    for th in (0.0, 0.5):
        s_ = sorted([r for r in es if r["theta"] == th], key=lambda r: -r["tol"])
        med = np.array([np.median([q[f"t_matvec_theta{th:g}_s"] for q in pt if q["tol"] == r["tol"]]) for r in s_])
        lo = np.array([min(q[f"t_matvec_theta{th:g}_s"] for q in pt if q["tol"] == r["tol"]) for r in s_])
        hi = np.array([max(q[f"t_matvec_theta{th:g}_s"] for q in pt if q["tol"] == r["tol"]) for r in s_])
        curve(ax, np.array([r["tol"] for r in s_]), 1e3 * med, 1e3 * lo, 1e3 * hi, CAND_COL[th], CAND_MK[th],
              label=CAND_LAB[th])
        if th == 0.0:
            for k, (r, y) in enumerate(zip(s_, med)):
                r1 = next(q for q in es if q["tol"] == r["tol"] and q["theta"] == 0.5)
                ax.annotate(fr"$s_R$ {r['sR']:.2f}$\to${r1['sR']:.2f}", xy=(r["tol"], 1e3 * y),
                            xytext=(3 if k == 0 else (-3 if k == len(s_) - 1 else 0), -14), textcoords="offset points",
                            fontsize=5.8, color=GREY, ha="left" if k == 0 else ("right" if k == len(s_) - 1 else "center"))
    ax.set_xscale("log"), ax.invert_xaxis(), ax.set_ylim(0, None)
    ax.set_xlabel(r"target $E_\Gamma$ (validated on 9 configurations)"), ax.set_ylabel("matvec time [ms]")
    ax.set_title(r"(c) Equal-accuracy cost, $\kappa=0.7$")
    ax.legend(loc="upper left")
    ax = axs[1, 1]
    for k, tol in enumerate(tols):
        v = np.array([q["matvec_ratio_theta0.5_over_0"] for q in pt if q["tol"] == tol])
        jit = np.linspace(-0.12, 0.12, len(v))
        ax.plot(k + jit, v, "o", color=CAND_COL[0.5], ms=3, mfc="white", ls="none")
        ax.plot([k - 0.22, k + 0.22], [np.median(v)] * 2, color=INK, lw=1.4)
    ax.axhline(1.0, color=INK, lw=0.8, ls=(0, (1, 1.5)))
    ax.annotate("equal cost", xy=(0.98, 1.0), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=6.3, xytext=(0, 1.5), textcoords="offset points")
    ax.set_xticks(range(len(tols))), ax.set_xticklabels([fr"$10^{{{int(np.log10(t))}}}$" for t in tols])
    ax.set_xlim(-0.5, len(tols) - 0.5), ax.set_ylim(0.9, 1.4)
    ax.set_xlabel(r"target $E_\Gamma$"), ax.set_ylabel(r"$t_{\rm matvec}(\theta=1/2)\,/\,t_{\rm matvec}(\theta=0)$")
    ax.set_title("(d) Paired timing, 9 configurations")
    ax.legend(handles=[Line2D([], [], color=CAND_COL[0.5], marker="o", mfc="white", ls="none", label="configuration"),
                       Line2D([], [], color=INK, lw=1.4, label="median")], loc="upper left")
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    save(fig, figures / "fig4_psd_cost")

    # ---------------- Fig. S2: all kappa
    fig, axs = plt.subplots(3, 2, figsize=(6.6, 7.4))
    for i, kap in enumerate((0.7, 0.2, 2.0)):
        for j, sw in enumerate(("sR", "sF")):
            ax = axs[i, j]
            cand_panel(ax, kap, sw, legend=(i == 0 and j == 0))
            ax.set_xlabel(r"$s_R=\xi r_c$  ($s_F=6.5$)" if sw == "sR" else r"$s_F=k_c/(2\xi)$  ($s_R=6.5$)")
            ax.set_ylabel("relative error")
            ax.set_title(f"({'abcdef'[2 * i + j]}) " + ("Short-range" if sw == "sR" else "Fourier")
                         + fr" cutoff, $\kappa={kap:g}$")
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    save(fig, figures / "figS2_pilot_all_kappa")
