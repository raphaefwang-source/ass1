#!/usr/bin/env python3
"""Figures of the operator experiment (Section 6) from ewald_longitudinal/exp1_results/*.csv.

Reads only the result files written after the |k|-rounding fix (run_2026-10-13_exact_k.log); writes vector PDFs
sized for the final text width (6.5 in) into paper_full/figures/. Usage, from the repository root:
    python3 paper_full/scripts/make_exp1_figures.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "ewald_longitudinal" / "exp1_results"
OUT = ROOT / "paper_full" / "figures"

WIDTH = 6.5                      # \textwidth of the manuscript, inches
S_PROD = 4.100151562017953       # production s of the mesh scans
FIT_FLOOR = 1.1e-13              # 100 x largest numerical discrepancy (summary.json numerical_floor.fit_floor)
TOL = {"T5": 1e-5, "T7": 1e-7, "T9": 1e-9}
INK, GREY = "#333333", "#9a9a9a"
FAM = {  # colour, marker, label
    "caseA": ("#000000", "D", "case A"),
    "caseB": ("#7a7a7a", "D", "case B"),
    "uniform": ("#1f6fb4", "o", "uniform"),
    "lattice": ("#d1581c", "s", "perturbed lattice"),
    "cluster": ("#2c9c5a", "^", "clustered"),
}
FAMS = ("caseA", "caseB", "uniform", "lattice", "cluster")


def rd(name):
    with open(RES / name, newline="") as fh:
        return list(csv.DictReader(fh))


def f(r, k):
    return float(r[k])


def style():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "legend.frameon": False, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42,
        "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5,
    })


def title(ax, s):
    ax.set_title(s, loc="left", pad=4)


def cutoffs_figure(cut):
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.95), constrained_layout=True)
    for ax, part, xl, tt in ((axs[0], "real", r"$\xi r_c$", r"(a) real-space cutoff, $\xi=0.7$"),
                             (axs[1], "fourier", r"$k_c/(2\xi)$", r"(b) Fourier cutoff, $\xi=0.7$")):
        sub = [r for r in cut if r["part"] == part and f(r, "xi") == 0.7]
        xg = np.unique([f(r, "x") for r in sub])
        for fam in ("uniform", "lattice", "cluster"):
            col, _, lab = FAM[fam]
            names = sorted({r["name"] for r in sub if r["family"] == fam and r["set"] == "main"})
            E = np.array([[f(r, "err") for r in sorted((r for r in sub if r["name"] == nm), key=lambda r: f(r, "x"))]
                          for nm in names])
            ax.fill_between(xg, E.min(0), E.max(0), color=col, alpha=0.18, lw=0)
            ax.plot(xg, np.median(E, 0), "-", color=col, label=lab + " (3 seeds)")
        for fam in ("caseA", "caseB"):
            col, _, lab = FAM[fam]
            rr = sorted((r for r in sub if r["family"] == fam), key=lambda r: f(r, "x"))
            ax.plot([f(r, "x") for r in rr], [f(r, "err") for r in rr], ":", color=col, lw=1.1, label=lab)
        ax.plot(xg, np.exp(-xg ** 2), "--", color=GREY, lw=1.0, label=r"$e^{-x^2}$")
        ax.axhline(FIT_FLOOR, color=INK, lw=0.7, ls=(0, (1, 2)))
        ax.text(0.2, FIT_FLOOR * 3, "fit floor", fontsize=8, color=INK, ha="left", va="bottom")
        ax.set_yscale("log")
        ax.set_ylim(1e-17, 3)
        ax.set_xlim(0, 6.6)
        ax.set_xlabel(xl)
        title(ax, tt)
    axs[0].set_ylabel(r"truncation error / $\|\Gamma\|_2$")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3, handlelength=1.8, columnspacing=1.5)
    return fig


def mesh_figure(mesh, fits):
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.9), constrained_layout=True)
    ax = axs[0]
    rep = "uniform_N128_s1"
    sub = [r for r in mesh if r["name"] == rep and abs(f(r, "s_F") - S_PROD) < 1e-6]
    ps = sorted({int(f(r, "p")) for r in sub})
    cmap = plt.get_cmap("viridis")
    for i, p in enumerate(ps):
        rr = sorted((r for r in sub if int(f(r, "p")) == p), key=lambda r: f(r, "xi_h"))
        x = np.array([f(r, "xi_h") for r in rr])
        y = np.array([f(r, "err_rms") for r in rr])
        col = cmap(i / (len(ps) - 1) * 0.9)
        ax.fill_between(x, [f(r, "err_min") for r in rr], [f(r, "err_max") for r in rr], color=col, alpha=0.2, lw=0)
        ax.plot(x, y, "o-", color=col, ms=2.8, label=f"$p={p}$")
    ax.axhline(FIT_FLOOR, color=INK, lw=0.7, ls=(0, (1, 2)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.1, 0.2, 0.3, 0.4])
    ax.set_xticklabels(["0.1", "0.2", "0.3", "0.4"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(1e-13, 1e-3)
    ax.set_xlabel(r"$\xi h$")
    ax.set_ylabel(r"$\|\widetilde\Gamma_h^L-\Gamma_L^{k_c}\|_2/\|\Gamma\|_2$")
    ax.legend(loc="lower right", ncol=2, handlelength=1.4, columnspacing=0.8, labelspacing=0.25)
    title(ax, r"(a) mesh error, uniform $N=128$, seed 1")

    ax = axs[1]
    jit = {"caseA": -0.24, "caseB": -0.12, "uniform": 0.0, "lattice": 0.12, "cluster": 0.24}
    for fam in FAMS:
        col, mk, lab = FAM[fam]
        for sF, filled in ((S_PROD, True), (5.5, False)):
            rr = [r for r in fits if r["family"] == fam and abs(f(r, "s_F") - sF) < 1e-6 and r["slope"]]
            if not rr:
                continue
            ax.plot([int(f(r, "p")) + jit[fam] for r in rr], [f(r, "slope") for r in rr], mk, ms=4,
                    color=col, mfc=col if filled else "white", mec=col, mew=0.8, label=lab if filled else None)
    pp = np.array([2.6, 8.4])
    ax.plot(pp, pp, "-", color=GREY, lw=0.9, label="slope $=p$")
    ax.plot([], [], "o", color=INK, mfc="white", mew=0.8, ms=4, label=r"$s=5.5$")
    ax.set_xticks(range(3, 9))
    ax.set_xlabel("B-spline order $p$")
    ax.set_ylabel("fitted slope in $h$")
    ax.legend(loc="upper left", handlelength=1.2, labelspacing=0.25)
    title(ax, "(b) slopes on the four finest meshes")
    return fig


def operator_figure(act, refn):
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.0), constrained_layout=True)
    ax = axs[0]
    xpos = {"T9": 0, "T7": 1, "T5": 2}
    off = {"caseA": -0.3, "caseB": -0.18, "uniform": -0.06, "lattice": 0.06, "cluster": 0.18}
    for fam in FAMS:
        col, mk, lab = FAM[fam]
        for pname, t in TOL.items():
            for sets, filled in ((("legacy", "main"), True), (("check",), False)):
                v = [f(r, "err_spec_rel") / t for r in act
                     if r["family"] == fam and r["params"] == pname and r["set"] in sets]
                if v:
                    ax.plot([xpos[pname] + off[fam]] * len(v), v, mk, color=col, ms=4.5,
                            mfc=col if filled else "white", mec=col, mew=0.8,
                            label=lab if (pname == "T7" and filled) else None)
    bx = [f(r, "err_spec_rel") / TOL["T7"] for r in act if r["params"] == "T7" and r["set"].startswith("box")]
    ax.plot([xpos["T7"] + 0.3] * len(bx), bx, "x", color=GREY, ms=4, mew=0.9, label="box-length sets")
    ax.axhline(1.0, color=INK, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.01, 10)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r"T9 ($10^{-9}$)", r"T7 ($10^{-7}$)", r"T5 ($10^{-5}$)"])
    ax.set_ylabel("relative operator error / tolerance")
    title(ax, "(a) relative operator error")

    ax = axs[1]
    rows = [r for r in act if r["params"] == "T7"]
    for r in (r for r in rows if r["set"].startswith("box")):
        ax.plot(refn[r["name"]], f(r, "err_spec_rel") * refn[r["name"]], "x", color=GREY, ms=4, mew=0.9)
    for fam in FAMS:
        col, mk, _ = FAM[fam]
        for r in (r for r in rows if r["family"] == fam and not r["set"].startswith("box")):
            filled = r["set"] != "check"
            ax.plot(refn[r["name"]], f(r, "err_spec_rel") * refn[r["name"]], mk, color=col, ms=4.5,
                    mfc=col if filled else "white", mec=col, mew=0.8)
    g = np.array([0.5, 50.0])
    ax.plot(g, 1e-7 * g, "-", color=GREY, lw=0.9)
    ax.text(30, 1e-7 * 30 * 0.5, r"relative $10^{-7}$", fontsize=8, color=INK, ha="right", va="top", rotation=24)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 50)
    ax.set_ylim(4e-8, 2e-6)
    ax.set_xlabel(r"$\|\Gamma\|_2$")
    ax.set_ylabel(r"$\|\widetilde\Gamma_h-\Gamma\|_2$")
    title(ax, "(b) absolute operator error, set T7")
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], ls="", marker="o", color=INK, mfc="white", mew=0.8, ms=4.5))
    l.append(r"open: $N=256$")
    fig.legend(h, l, loc="outside upper center", ncol=7, handlelength=1.0, columnspacing=0.9, handletextpad=0.3)
    return fig


def probes_box_figure(probes, act, refn):
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.9), constrained_layout=True)
    ax = axs[0]
    kinds = [("gauss", "Gaussian"), ("long", "longitudinal\n$2\\pi/L$"), ("trans", "transverse\n$2\\pi/L$"),
             ("translation", "translation")]
    for gi, fam in enumerate(FAMS):
        col, mk, lab = FAM[fam]
        for ki, (kind, _) in enumerate(kinds):
            sub = [r for r in probes if r["family"] == fam and r["params"] == "T7" and r["probe"] == kind
                   and r["set"] in ("legacy", "main")]
            col_ = "opnorm_err" if kind == "translation" else "rel_err"
            v = [max(f(r, col_), 1e-18) for r in sub]
            if v:
                ax.plot([ki + (gi - 2) * 0.13] * len(v), v, mk, color=col, ms=3, alpha=0.8, mec="none",
                        label=lab if ki == 0 else None)
    ax.axhline(1e-7, color=INK, lw=0.8, ls="--")
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([k[1] for k in kinds])
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e-5)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("relative action error")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), handletextpad=0.2, labelspacing=0.25)
    title(ax, "(a) probe vectors, set T7")

    ax = axs[1]
    for setn, col, mk, lab, dx in (("box_fixedN", "#1f6fb4", "o", r"fixed $N=64$ (density varies)", -0.12),
                                   ("box_fixedn", "#d1581c", "s", r"fixed density $30/216$ ($N$ varies)", 0.12)):
        sub = [r for r in act if r["set"] == setn]
        Ls = sorted({f(r, "L") for r in sub})
        v = [[f(r, "err_spec_rel") * refn[r["name"]] for r in sub if f(r, "L") == L] for L in Ls]
        ax.plot(np.array(Ls) + dx, [np.median(x) for x in v], mk + "-", color=col, ms=4, label=lab)
        ax.vlines(np.array(Ls) + dx, [min(x) for x in v], [max(x) for x in v], color=col, lw=1.2)
    # The realised k_c h/pi depends only on L here (same k_c and mesh rounding in both sets): one label per L.
    etas = {}
    for r in act:
        if r["set"].startswith("box"):
            etas.setdefault(f(r, "L"), set()).add(round(f(r, "eta_actual"), 2))
    assert all(len(e) == 1 for e in etas.values())
    for L, e in etas.items():
        ax.text(L, 1.45e-6, f"{e.pop():.2f}", fontsize=8, color=INK, ha="center", va="center")
    ax.set_yscale("log")
    ax.set_ylim(3e-8, 2.2e-6)
    ax.set_xlim(2.8, 17)
    ax.set_xticks([4, 6, 8, 10, 12, 16])
    ax.set_xlabel("box length $L$")
    ax.set_ylabel(r"$\|\widetilde\Gamma_h-\Gamma\|_2$")
    ax.legend(loc="lower left", handlelength=1.6, labelspacing=0.25)
    title(ax, r"(b) box length, set T7; top row: $k_ch/\pi$")
    return fig


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    ref = rd("reference_ladder.csv")
    refn = {r["name"]: f(r, "norm2") for r in ref if f(r, "R") == 45.0}
    act, probes = rd("action.csv"), rd("action_probes.csv")
    figs = {
        "exp1_cutoffs": cutoffs_figure(rd("cutoffs.csv")),
        "exp1_mesh": mesh_figure(rd("mesh.csv"), rd("fits_mesh.csv")),
        "exp1_operator": operator_figure(act, refn),
        "exp1_probes_box": probes_box_figure(probes, act, refn),
    }
    for name, fig in figs.items():
        fig.savefig(OUT / f"{name}.pdf")
        plt.close(fig)
        print("wrote", OUT / f"{name}.pdf")


if __name__ == "__main__":
    main()
