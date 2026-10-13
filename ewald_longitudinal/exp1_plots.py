#!/usr/bin/env python3
"""
Experiment 1: fits, figures, LaTeX tables and the number summary cited in the manuscript text.
Reads only exp1_results/*.csv written by exp1_operator_action.py; writes

  exp1_results/fits_cutoffs.csv, fits_mesh.csv, summary.json
  exp1_results/figures/exp1_components.{pdf,png}, exp1_action.{pdf,png}
  paper/figures/exp1_components.pdf, exp1_action.pdf
  paper/tab_exp1_configs.tex, paper/tab_exp1_params.tex

Fit rules (stated in the text):
  cutoffs: log err = a - c x^2 on 1 <= x and err >= FIT_FLOOR (pre-plateau); x = xi rc or kc/(2 xi);
  mesh:    log err_rms = a + q log h on the four finest meshes with err_rms >= FIT_FLOOR, per (config, s_F, p).
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp1_results"
FIG = OUT / "figures"
PAPER = HERE.parent / "paper"
FIT_FLOOR = 1e-12

# reference palette (dataviz skill): categorical slots 1-3 for the three families, ink for legacy cases,
# ordinal blue ramp (steps 250..700) for the B-spline order p
FAM = {"uniform": ("#2a78d6", "o", "uniform"), "lattice": ("#eb6834", "s", "perturbed lattice"),
       "cluster": ("#1baf7a", "^", "clustered")}
LEG = {"caseA": ("#0b0b0b", "D", "case A"), "caseB": ("#52514e", "v", "case B")}
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def rd(name):
    with open(OUT / name) as fh:
        return list(csv.DictReader(fh))


def num(r, k):
    v = r.get(k, "")
    if v in ("", "nan", "None"):
        return np.nan
    if v in ("True", "False"):
        return v == "True"
    return float(v)


def style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
                         "lines.markersize": 4.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "legend.frameon": False, "savefig.dpi": 200, "pdf.fonttype": 42,
                         "mathtext.fontset": "cm"})


def panel_label(ax, s):
    ax.set_title(s, loc="left", fontsize=8.5)


# ----------------------------------------------------------------------------
def fit_cutoffs(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["name"], r["part"], float(r["xi"]))].append(r)
    fits = []
    for (name, part, xi), rr in by.items():
        x = np.array([num(r, "x") for r in rr])
        e = np.array([num(r, "err") for r in rr])
        nn = np.array([r["no_new"] == "True" for r in rr])
        cnt = np.array([num(r, "count") for r in rr])
        win = (x >= 1.0) & (e >= FIT_FLOOR)
        m = win & ~nn & (cnt > 0)                      # flat or empty staircase steps carry no decay information
        tail = e[x >= x.max() - 0.6]
        rec = dict(name=name, set=rr[0]["set"], family=rr[0]["family"], N=int(num(rr[0], "N")), part=part, xi=xi,
                   n_fit=int(m.sum()), x_fit_min=float(x[m].min()) if m.any() else np.nan,
                   x_fit_max=float(x[m].max()) if m.any() else np.nan, flat_steps_total=int(nn.sum()),
                   flat_or_empty_in_window_excluded=int((win & ~m).sum()),
                   err_at_xmax=float(e[-1]), tail_max=float(tail.max()),
                   plateau=bool(tail.min() > 0.3 * tail.max()))
        if m.sum() >= 3:
            c, a = np.polyfit(x[m] ** 2, np.log(e[m]), 1)
            rec.update(c=float(-c), a=float(a))
            c2, _ = np.polyfit(x[m] ** 2, np.log(e[m] / x[m]), 1)       # with the s e^{-s^2} prefactor removed
            rec["c_prefactor_removed"] = float(-c2)
        fits.append(rec)
    return fits


def fit_mesh(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["name"], float(r["s_F"]), int(float(r["p"])))].append(r)
    fits = []
    for (name, sF, p), rr in by.items():
        rr = sorted(rr, key=lambda r: num(r, "h"))
        h = np.array([num(r, "h") for r in rr])
        e = np.array([num(r, "err_rms") for r in rr])
        emax = np.array([num(r, "err_max") for r in rr])
        emin = np.array([num(r, "err_min") for r in rr])
        ok = e >= FIT_FLOOR
        idx = np.where(ok)[0][:4]
        rec = dict(name=name, set=rr[0]["set"], family=rr[0]["family"], N=int(num(rr[0], "N")), s_F=sF, p=p,
                   xi=num(rr[0], "xi"), n_meshes=len(rr), n_fit=len(idx), h_fit=[float(h[i]) for i in idx],
                   eta_range=[num(rr[-1], "eta"), num(rr[0], "eta")],
                   shift_max_over_rms=float(np.nanmax(emax / e)), shift_max_over_min=float(np.nanmax(emax / emin)),
                   min_err=float(e.min()), eps_F=num(rr[0], "eps_F"))
        eta = np.array([num(r, "eta") for r in rr])
        if len(idx) >= 3:
            q, a = np.polyfit(np.log(h[idx]), np.log(e[idx]), 1)
            qmax, _ = np.polyfit(np.log(h[idx]), np.log(emax[idx]), 1)
            rec.update(slope=float(q), CM=float(np.exp(a)), slope_of_max=float(qmax),
                       eta_fit=[float(eta[idx].min()), float(eta[idx].max())])
            loc = np.diff(np.log(e)) / np.diff(np.log(h))
            rec["local_slopes"] = [float(v) for v in loc]
        coarse = np.where(eta >= 0.5)[0]
        if len(coarse) >= 3:
            qc, _ = np.polyfit(np.log(h[coarse]), np.log(e[coarse]), 1)
            rec.update(slope_coarse=float(qc), eta_coarse=[float(eta[coarse].min()), float(eta[coarse].max())])
        rec["h_steps"] = [float(v) for v in h[1:] / h[:-1] - 1]
        fits.append(rec)
    return fits


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})


# ----------------------------------------------------------------------------
def fam_style(fam):
    return FAM.get(fam) or LEG.get(fam) or ("#52514e", "x", fam)


def components_figure(cut, mesh, msum, split_op, split_pairs, quad):
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.6), constrained_layout=True)
    # (a) split identity
    ax = axs[0, 0]
    xs = sorted({float(r["xi"]) for r in split_op})
    for fam in ("caseA", "caseB", "uniform", "lattice", "cluster"):
        sub = [r for r in split_op if any(k in r["name"] for k in ([fam] if fam in FAM else
                                                                   (["A"] if fam == "caseA" else ["B"])))]
        if fam in ("caseA", "caseB"):
            sub = [r for r in split_op if r["name"] == ("A" if fam == "caseA" else "B")]
        if not sub:
            continue
        col, mk, lab = fam_style(fam)
        for xi in xs:
            v = [num(r, "spec_rel") for r in sub if float(r["xi"]) == xi]
            ax.plot([xi] * len(v), v, mk, color=col, ms=4, alpha=0.9, mec="white", mew=0.5,
                    label=lab if xi == xs[0] else None)
    pr = [r for r in split_pairs if r["points"].startswith("pair")]
    ax.plot([num(r, "xi") for r in pr], [num(r, "max_rel") for r in pr], "-", color=INK2, lw=1,
            label=r"pair level, max over $r$")
    ax.set_yscale("log")
    ax.set_ylim(3e-17, 3e-12)
    ax.set_xlabel(r"splitting parameter $\xi$")
    ax.set_ylabel("relative error")
    ax.legend(loc="lower center", ncol=2, handletextpad=0.2, columnspacing=0.5, fontsize=6.3, markerscale=0.9)
    panel_label(ax, r"(a) $K=K^S+K^L$ against $\Gamma_{\rm ref}$")
    # (b), (c) cutoffs at the production xi = 0.7, family median with seed range; legacy cases thin
    for ax, part, xl, tt in ((axs[0, 1], "real", r"$\xi r_c$", r"(b) real-space cutoff, $\xi=0.7$"),
                             (axs[0, 2], "fourier", r"$k_c/(2\xi)$", r"(c) Fourier cutoff, $\xi=0.7$")):
        sub = [r for r in cut if r["part"] == part and float(r["xi"]) == 0.7]
        xg = np.unique([num(r, "x") for r in sub])
        for fam in ("uniform", "lattice", "cluster"):
            col, mk, lab = FAM[fam]
            E = np.array([[num(r, "err") for r in sorted([r for r in sub if r["name"] == nm],
                                                          key=lambda r: num(r, "x"))]
                          for nm in sorted({r["name"] for r in sub if r["family"] == fam and r["set"] == "main"})])
            ax.fill_between(xg, E.min(0), E.max(0), color=col, alpha=0.15, lw=0)
            ax.plot(xg, np.median(E, 0), "-", color=col, label=lab)
        for fam in ("caseA", "caseB"):
            col, mk, lab = LEG[fam]
            r_ = sorted([r for r in sub if r["family"] == fam], key=lambda r: num(r, "x"))
            ax.plot([num(r, "x") for r in r_], [num(r, "err") for r in r_], ":", color=col, lw=1, label=lab)
        ax.plot(xg, np.exp(-xg ** 2), "--", color=INK2, lw=0.8, label=r"$e^{-x^2}$")
        ax.set_yscale("log")
        ax.set_ylim(1e-17, 2)
        ax.set_xlabel(xl)
        ax.set_ylabel(r"$\|\Delta\Gamma\|_2/\|\Gamma_{\rm ref}\|_2$")
        panel_label(ax, tt)
        if part == "real":
            ax.legend(loc="lower left", handlelength=1.6, fontsize=6.5)
    # (d) mesh error vs xi h at production kc, representative configuration, p ramp, RMS and shift range
    ax = axs[1, 0]
    rep = "uniform_N128_s1"
    sub = [r for r in mesh if r["name"] == rep and abs(num(r, "s_F") - 4.100151562017953) < 1e-6]
    for i, p in enumerate(sorted({int(float(r["p"])) for r in sub})):
        rr = sorted([r for r in sub if int(float(r["p"])) == p], key=lambda r: num(r, "xi_h"))
        x = np.array([num(r, "xi_h") for r in rr])
        y = np.array([num(r, "err_rms") for r in rr])
        lo = np.array([num(r, "err_min") for r in rr])
        hi = np.array([num(r, "err_max") for r in rr])
        col = RAMP[min(i, len(RAMP) - 1)]
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(x, y, "o-", color=col, ms=3, label=f"$p={p}$")
        f = [r for r in msum if r["name"] == rep and r["p"] == p and abs(r["s_F"] - 4.100151562017953) < 1e-6]
        if f and "slope" in f[0]:
            xx = np.array([x[0], x[min(3, len(x) - 1)]])
            ax.plot(xx, y[0] * (xx / x[0]) ** p * 0.35, "-", color=INK2, lw=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.1, 0.2, 0.3, 0.4])
    ax.set_xticklabels(["0.1", "0.2", "0.3", "0.4"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$\xi h$")
    ax.set_ylabel(r"$\|\Gamma_L^{\rm mesh}-\Gamma_L^{k_c}\|_2/\|\Gamma_{\rm ref}\|_2$")
    ax.legend(loc="lower right", ncol=2, handlelength=1.2, columnspacing=0.6)
    panel_label(ax, r"(d) mesh error, uniform $N\!=\!128$, seed 1")
    # (e) fitted mesh slope vs p over configurations
    ax = axs[1, 1]
    for fam in ("caseA", "caseB", "uniform", "lattice", "cluster"):
        col, mk, lab = fam_style(fam)
        for sF, fill in ((4.100151562017953, True), (5.5, False)):
            f = [r for r in msum if r["family"] == fam and abs(r["s_F"] - sF) < 1e-6 and "slope" in r]
            if not f:
                continue
            jit = {"caseA": -0.24, "caseB": -0.12, "uniform": 0.0, "lattice": 0.12, "cluster": 0.24}[fam]
            ax.plot([r["p"] + jit for r in f], [r["slope"] for r in f], mk, color=col, ms=4,
                    mfc=col if fill else "white", mec=col if not fill else "white", mew=0.8 if not fill else 0.5,
                    label=lab if fill else None)
    pp = np.arange(2.5, 8.6, 0.5)
    ax.plot(pp, pp, "-", color=INK2, lw=0.8, label="slope $=p$")
    ax.set_xlabel("B-spline order $p$")
    ax.set_ylabel(r"fitted slope of RMS error in $h$")
    ax.legend(loc="upper left", handlelength=1.2)
    panel_label(ax, r"(e) $O(h^p)$ fits (open: $k_c$ at $s=5.5$)")
    # (f) grid-shift sensitivity: max over shifts / RMS
    ax = axs[1, 2]
    for fam in ("caseA", "caseB", "uniform", "lattice", "cluster"):
        col, mk, lab = fam_style(fam)
        sub = [r for r in mesh if r["family"] == fam and abs(num(r, "s_F") - 4.100151562017953) < 1e-6
               and int(num(r, "shifts")) == 8]
        if not sub:
            continue
        ax.plot([num(r, "eta") for r in sub], [num(r, "err_max") / num(r, "err_rms") for r in sub], mk, color=col,
                ms=2.5, alpha=0.6, mec="none", label=lab)
    ax.set_xlabel(r"$\eta = k_c h/\pi$")
    ax.set_ylabel("max / RMS over grid shifts")
    panel_label(ax, r"(f) grid-shift spread (8 shifts, $N\leq128$)")
    for a in axs.ravel():
        a.grid(True, which="major")
    return fig


def action_figure(act, probes, box, refn):
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)
    fams = ("caseA", "caseB", "uniform", "lattice", "cluster")
    tolv = {"T5": 1e-5, "T7": 1e-7, "T9": 1e-9}
    jit = {"caseA": 0.80, "caseB": 0.89, "uniform": 1.0, "lattice": 1.12, "cluster": 1.26}
    # (a) achieved relative spectral error against the tolerance the set was accepted for (on A and B)
    ax = axs[0, 0]
    for fam in fams:
        col, mk, lab = fam_style(fam)
        for pname, t in tolv.items():
            for sets, fill in ((("legacy", "main"), True), (("check",), False)):
                v = [num(r, "err_spec_rel") for r in act if r["family"] == fam and r["params"] == pname
                     and r["set"] in sets]
                if v:
                    ax.plot([t * jit[fam]] * len(v), v, mk, color=col, ms=4.5, mfc=col if fill else "white",
                            mec="white" if fill else col, mew=0.5 if fill else 0.9,
                            label=lab if (pname == "T7" and fill) else None)
    tt = np.array([4e-10, 2.5e-5])
    ax.plot(tt, tt, "-", color=INK2, lw=0.8)
    ax.text(3e-6, 2.0e-6, "error = tol", rotation=33, fontsize=6.5, color=INK2, ha="center", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1e-9, 1e-7, 1e-5])
    ax.set_xticklabels(["T9\n$10^{-9}$", "T7\n$10^{-7}$", "T5\n$10^{-5}$"])
    ax.set_xlabel("parameter set (tolerance accepted on A and B)")
    ax.set_ylabel(r"$\|\Gamma_h-\Gamma_{\rm ref}\|_2/\|\Gamma_{\rm ref}\|_2$")
    ax.legend(loc="upper left", handletextpad=0.2, fontsize=6.5, ncol=1)
    panel_label(ax, "(a) relative spectral error (open: $N=256$)")
    # (b) absolute error against ||Gamma_ref||_2 at T7
    ax = axs[0, 1]
    allr = [r for r in act + box if r["params"] == "T7"]
    nG = {r["name"]: refn[r["name"]] for r in allr}
    for r in [r for r in allr if r["set"].startswith("box")]:
        ax.plot(nG[r["name"]], num(r, "err_spec_rel") * nG[r["name"]], "x", color="#a3a29d", ms=3.5, mew=0.8)
    for fam in fams:
        col, mk, lab = fam_style(fam)
        sub = [r for r in allr if r["family"] == fam and not r["set"].startswith("box")]
        ax.plot([nG[r["name"]] for r in sub], [num(r, "err_spec_rel") * nG[r["name"]] for r in sub], mk, color=col,
                ms=4.5, mec="white", mew=0.5)
    g = np.array([0.5, 40.0])
    ax.plot(g, 1e-7 * g, "-", color=INK2, lw=0.8)
    ax.text(27, 1e-7 * 27 * 0.62, r"relative $10^{-7}$", fontsize=6.5, color=INK2, ha="right", va="top", rotation=28)
    ax.plot([], [], "x", color="#a3a29d", label="box-length sets")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\|\Gamma_{\rm ref}\|_2$ (set by the closest pairs)")
    ax.set_ylabel(r"$\|\Gamma_h-\Gamma_{\rm ref}\|_2$")
    ax.legend(loc="lower right", fontsize=6.5)
    panel_label(ax, "(b) absolute error, set T7")
    # (c) probes at T7
    ax = axs[1, 0]
    kinds = [("gauss", "Gaussian"), ("long", r"long. $k_{\min}$"), ("trans", r"transv. $k_{\min}$"),
             ("translation", "translation")]
    for gi, fam in enumerate(fams):
        col, mk, lab = fam_style(fam)
        for ki, (kind, _) in enumerate(kinds):
            sub = [r for r in probes if r["family"] == fam and r["params"] == "T7" and r["probe"] == kind
                   and r["set"] in ("legacy", "main")]
            if kind == "translation":
                v = [max(num(r, "opnorm_err"), 1e-18) for r in sub]
            else:
                v = [num(r, "rel_err") for r in sub]
            if v:
                ax.plot([ki + (gi - 2) * 0.14] * len(v), v, mk, color=col, ms=3, alpha=0.8, mec="none")
    ax.axhline(1e-7, color=INK2, lw=0.6, ls="--")
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([k[1] for k in kinds], fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(5e-19, 3e-6)
    ax.set_ylabel(r"$\|(\Gamma_h-\Gamma_{\rm ref})X\|/\|\Gamma_{\rm ref}X\|$" + "\n" +
                  r"translation: $/(\|\Gamma_{\rm ref}\|_2\|X\|)$")
    panel_label(ax, r"(c) probes through $\Gamma_h X$, set T7")
    # (d) box length at T7: absolute error, with the realised eta (M is rounded to a 2,3,5-smooth size)
    ax = axs[1, 1]
    for setn, col, mk, lab, dx in (("box_fixedN", "#2a78d6", "o", r"fixed $N=64$ ($n$ varies)", -0.12),
                                   ("box_fixedn", "#eb6834", "s", r"fixed $n=n_A$ ($N$ varies)", 0.12)):
        sub = [r for r in box if r["set"] == setn]
        Ls = sorted({num(r, "L") for r in sub})
        v = [[num(r, "err_spec_rel") * refn[r["name"]] for r in sub if num(r, "L") == L] for L in Ls]
        ax.plot(np.array(Ls) + dx, [np.median(x) for x in v], mk + "-", color=col, ms=4, label=lab)
        ax.vlines(np.array(Ls) + dx, [min(x) for x in v], [max(x) for x in v], color=col, lw=1.2)
        if setn == "box_fixedN":
            for L, x in zip(Ls, v):
                eta = num([r for r in sub if num(r, "L") == L][0], "eta_actual")
                ax.text(L + dx, max(x) * 1.25, rf"$\eta$={eta:.2f}", fontsize=5.8, color=INK2, ha="center")
    ax.set_yscale("log")
    ax.set_ylim(3e-8, 2e-6)
    ax.set_xlabel(r"box length $L$")
    ax.set_ylabel(r"$\|\Gamma_h-\Gamma_{\rm ref}\|_2$")
    ax.legend(loc="lower right", fontsize=6.5)
    panel_label(ax, "(d) box length, set T7 (3 seeds)")
    return fig


# ----------------------------------------------------------------------------
def tables(cfg, act):
    PAPER.mkdir(exist_ok=True)
    groups = [("legacy", "caseA"), ("legacy", "caseB")] + [("main", f) for f in ("uniform", "lattice", "cluster")] \
        + [("check", f) for f in ("uniform", "lattice", "cluster")] + [("box_fixedN", "uniform"),
                                                                       ("box_fixedn", "uniform")]
    gen_txt = {"caseA": "uniform, seed 7", "caseB": "uniform, seed 13",
               "uniform": "uniform i.i.d.", "lattice": r"bcc/fcc, $\sigma=0.1\,d_{\rm nn}$",
               "cluster": r"clusters of 16, $\sigma_c=0.5$, $d\ge0.25$"}
    name_txt = {("legacy", "caseA"): "A", ("legacy", "caseB"): "B", ("main", "uniform"): "U128",
                ("main", "lattice"): "L128", ("main", "cluster"): "C128", ("check", "uniform"): "U256",
                ("check", "lattice"): "L256", ("check", "cluster"): "C256", ("box_fixedN", "uniform"): "box-$N$",
                ("box_fixedn", "uniform"): "box-$n$"}
    lines = [r"\begin{tabular}{llrlllr}", r"\toprule",
             r"set & generation & $N$ & $L=\kappa L$ & $n=N/L^3$ & $d_{\min}$ & seeds \\", r"\midrule"]
    for sset, fam in groups:
        sub = [r for r in cfg if r["set"] == sset and r["family"] == fam]
        if not sub:
            continue
        Ns = sorted({int(r["N"]) for r in sub})
        Ls = sorted({float(r["L"]) for r in sub})
        ns = sorted({float(r["n"]) for r in sub})
        dm = [float(r["d_min"]) for r in sub]
        fmtL = lambda v: f"{v:.3g}" if v != int(v) else f"{int(v)}"                      # noqa: E731
        Ntxt = str(Ns[0]) if len(Ns) == 1 else f"{Ns[0]}--{Ns[-1]}"
        Ltxt = fmtL(Ls[0]) if len(Ls) == 1 else f"{fmtL(Ls[0])}--{fmtL(Ls[-1])} ({len(Ls)} values)"
        ntxt = f"{ns[0]:.3f}" if len(ns) == 1 or (max(ns) - min(ns)) < 1e-3 else f"{min(ns):.3f}--{max(ns):.2f}"
        kL = Ltxt
        dtxt = f"{min(dm):.3f}" if len(dm) == 1 else f"{min(dm):.3f}--{max(dm):.3f}"
        seeds = sorted({int(r["seed"]) for r in sub})
        gtxt = gen_txt[fam]
        if fam == "lattice":
            gtxt = (r"bcc $4^3$" if Ns == [128] else r"fcc $4^3$") + r", $\sigma=0.1\,d_{\rm nn}$"
        lines.append(f"{name_txt[(sset, fam)]} & {gtxt} & {Ntxt} & {Ltxt} & {ntxt} & {dtxt} & "
                     f"{len(seeds)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tab_exp1_configs.tex").write_text("\n".join(lines) + "\n")
    # numerical parameter sets (M depends on L)
    lines = [r"\begin{tabular}{lllllll}", r"\toprule",
             r"set & $\xi$ & $s$ & $r_c=s/\xi$ & $k_c=2\xi s$ & $\eta_N$, $p$ & $M$ (A / B / $N\!=\!128$ / $N\!=\!256$) \\",
             r"\midrule"]
    for pname in ("T5", "T7", "T9"):
        sub = [r for r in act if r["params"] == pname]
        r0 = sub[0]
        Ms = []
        for key in (("caseA",), ("caseB",), ("main",), ("check",)):
            m = sorted({int(float(r["M"])) for r in sub if r["family"] in key or r["set"] in key})
            Ms.append("/".join(str(v) for v in m))
        lines.append(f"{pname} & {num(r0, 'xi'):g} & {num(r0, 's'):.3f} & {num(r0, 'rc'):.3f} & {num(r0, 'kc'):.3f} & "
                     f"{num(r0, 'eta'):g}, {int(num(r0, 'p'))} & {' / '.join(Ms)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tab_exp1_params.tex").write_text("\n".join(lines) + "\n")


def summary(ref, quad, split_pairs, split_op, cutfits, msum, dec, act, probes, cfg, prec):
    """Every number quoted in the LaTeX section."""
    S = {}
    rl = [r for r in ref]
    nG = {r["name"]: num(r, "norm2") for r in rl if num(r, "R") == 45.0}
    S["reference"] = dict(
        n_configs=len({r["name"] for r in rl}),
        max_change_R40_to_R45=max(num(r, "spec_rel_vs_Rmax") for r in rl if num(r, "R") == 40.0),
        max_change_R35_to_R45=max(num(r, "spec_rel_vs_Rmax") for r in rl if num(r, "R") == 35.0),
        max_change_R30_to_R45=max(num(r, "spec_rel_vs_Rmax") for r in rl if num(r, "R") == 30.0),
        max_trans_resid=max(num(r, "trans_resid") for r in rl if num(r, "R") == 45.0),
        max_sym_resid=max(num(r, "sym_resid") for r in rl if num(r, "R") == 45.0),
        norm2_range=[min(nG.values()), max(nG.values())],
        lam_star_range=[min(num(r, "lam_star") for r in rl if num(r, "R") == 45.0),
                        max(num(r, "lam_star") for r in rl if num(r, "R") == 45.0)])
    S["quadrature"] = {f"xi{r['xi']}": {k: num(r, k) for k in r if k != "xi"} for r in quad}
    S["fourier_precision"] = [{k: (num(r, k) if k not in ("name", "longdouble_type") else r[k]) for k in r} for r in prec]
    S["split_pairs"] = {f"{r['xi']}|{r['points'][:4]}": num(r, "max_rel") for r in split_pairs}
    S["split_pairs_max"] = max(num(r, "max_rel") for r in split_pairs)
    S["split_operator_range"] = [min(num(r, "spec_rel") for r in split_op), max(num(r, "spec_rel") for r in split_op)]
    S["split_operator_by_xi"] = {str(xi): max(num(r, "spec_rel") for r in split_op if float(r["xi"]) == xi)
                                 for xi in sorted({float(r["xi"]) for r in split_op})}
    for part in ("real", "fourier"):
        for xi in (0.7, 1.0):
            f = [r for r in cutfits if r["part"] == part and r["xi"] == xi and "c" in r and r["set"] in ("legacy", "main")]
            fc = [r for r in cutfits if r["part"] == part and r["xi"] == xi and "c" in r and r["set"] == "check"]
            fa = [r for r in cutfits if r["part"] == part and r["xi"] == xi]
            S[f"cut_{part}_xi{xi:g}"] = dict(
                c_median=float(np.median([r["c"] for r in f])), c_min=min(r["c"] for r in f),
                c_max=max(r["c"] for r in f),
                c_pref_removed_median=float(np.median([r["c_prefactor_removed"] for r in f])),
                c_check=[round(r["c"], 3) for r in fc],
                err_at_xmax_max=max(r["err_at_xmax"] for r in fa), tail_max=max(r["tail_max"] for r in fa),
                plateau_configs=sorted({r["name"] for r in fa if r["plateau"]}),
                flat_steps_total=sum(r["flat_steps_total"] for r in fa),
                excluded_from_fit=sum(r["flat_or_empty_in_window_excluded"] for r in fa),
                configs_with_flat=sorted({r["name"] for r in fa if r["flat_steps_total"] > 0}))
    for sF in sorted({r["s_F"] for r in msum}):
        sub = [r for r in msum if r["s_F"] == sF and "slope" in r and r["set"] in ("legacy", "main")]
        chk = [r for r in msum if r["s_F"] == sF and "slope" in r and r["set"] == "check"]
        key = f"mesh_sF{sF:.3g}"
        S[key] = {str(p): dict(slope_median=float(np.median([r["slope"] for r in sub if r["p"] == p])),
                               slope_min=min(r["slope"] for r in sub if r["p"] == p),
                               slope_max=max(r["slope"] for r in sub if r["p"] == p),
                               n=len([r for r in sub if r["p"] == p]),
                               configs=sorted(r["name"] for r in sub if r["p"] == p),
                               check_slopes=[round(r["slope"], 2) for r in chk if r["p"] == p],
                               coarse_slope_median=(float(np.median([r["slope_coarse"] for r in sub if r["p"] == p
                                                                     and "slope_coarse" in r]))
                                                    if any("slope_coarse" in r for r in sub if r["p"] == p) else None),
                               coarse_slope_range=([min(r["slope_coarse"] for r in sub if r["p"] == p and "slope_coarse" in r),
                                                    max(r["slope_coarse"] for r in sub if r["p"] == p and "slope_coarse" in r)]
                                                   if any("slope_coarse" in r for r in sub if r["p"] == p) else None),
                               slope_of_max_minus_rms=[min(r["slope_of_max"] - r["slope"] for r in sub if r["p"] == p),
                                                       max(r["slope_of_max"] - r["slope"] for r in sub if r["p"] == p)])
                     for p in sorted({r["p"] for r in sub})}
        S[key + "_eta_fit"] = dict(min=min(r["eta_fit"][0] for r in sub + chk), max=max(r["eta_fit"][1] for r in sub + chk),
                                   median_lo=float(np.median([r["eta_fit"][0] for r in sub])),
                                   median_hi=float(np.median([r["eta_fit"][1] for r in sub])),
                                   max_main=max(r["eta_fit"][1] for r in sub))
        S[key + "_coarse_eta"] = [min(r["eta_coarse"][0] for r in sub if "eta_coarse" in r),
                                  max(r["eta_coarse"][1] for r in sub if "eta_coarse" in r)]
        S[key + "_h_steps"] = [min(min(r["h_steps"]) for r in sub + chk), max(max(r["h_steps"]) for r in sub + chk)]
        allr = sub + chk
        S[key + "_shift_max_over_rms"] = max(r["shift_max_over_rms"] for r in allr if r["set"] != "check")
        S[key + "_shift_max_over_min"] = max(r["shift_max_over_min"] for r in allr if r["set"] != "check")
        S[key + "_min_err"] = min(r["min_err"] for r in msum if r["s_F"] == sF)
        S[key + "_threshold_moved_window"] = sorted({(r["name"], r["p"]) for r in msum if r["s_F"] == sF
                                                     and r.get("eta_fit") and r["eta_fit"][0] > 0.3})
    S["mesh_no_fit"] = [(r["name"], r["s_F"], r["p"]) for r in msum if "slope" not in r]
    S["decomposition"] = dict(
        identity_resid_rel_to_K_max=max(num(r, "identity_resid_rel_to_K") for r in dec),
        identity_resid_rel_to_error_max=max(num(r, "identity_resid_rel_to_error") for r in dec),
        linear_over_total=[min(num(r, "linear_over_total") for r in dec), max(num(r, "linear_over_total") for r in dec)],
        quadratic_over_total=[min(num(r, "quadratic_over_total") for r in dec),
                              max(num(r, "quadratic_over_total") for r in dec)],
        Dh_over_total_max=max(num(r, "Dh_over_total") for r in dec),
        literal_vs_modespace_max=max(num(r, "literal_vs_modespace") for r in dec if r.get("literal_vs_modespace")),
        eta=[min(num(r, "eta") for r in dec), max(num(r, "eta") for r in dec)], configs=sorted({r["name"] for r in dec}))
    for pname in ("T5", "T7", "T9"):
        sub = [r for r in act if r["params"] == pname]
        S[f"action_{pname}"] = {
            grp: dict(err_spec=[min(num(r, "err_spec_rel") for r in g), max(num(r, "err_spec_rel") for r in g)],
                      err_abs=[min(num(r, "err_spec_rel") * nG[r["name"]] for r in g),
                               max(num(r, "err_spec_rel") * nG[r["name"]] for r in g)],
                      gauss_rel_max=max(num(r, "gauss_rel_max") for r in g),
                      long_rel_max=max(num(r, "long_rel_max") for r in g),
                      trans_rel_max=max(num(r, "trans_rel_max") for r in g),
                      trans_resid_h=max(num(r, "trans_resid_h_opnorm") for r in g),
                      trans_err_abs=max(num(r, "trans_err_abs_opnorm") for r in g),
                      sym=max(num(r, "sym_resid") for r in g),
                      literal_vs_modespace=max(num(r, "literal_vs_modespace") for r in g),
                      lam_rel_change=max(abs(num(r, "lam_star_h") / num(r, "lam_star_ref") - 1) for r in g),
                      weyl_margin_min=min(num(r, "weyl_margin") / num(r, "lam_star_ref") for r in g),
                      eps_S=max(num(r, "eps_S") for r in g), eps_F=max(num(r, "eps_F") for r in g),
                      eps_mesh=max(num(r, "eps_mesh") for r in g),
                      eps_S_gt_mesh=sorted(r["name"] for r in g if num(r, "eps_S") > num(r, "eps_mesh")),
                      budget_match=max(num(r, "real_matches_production") for r in g), n=len(g))
            for grp, g in (("legacy", [r for r in sub if r["set"] == "legacy"]),
                           ("main", [r for r in sub if r["set"] == "main"]),
                           ("check", [r for r in sub if r["set"] == "check"]),
                           ("box_fixedN", [r for r in sub if r["set"] == "box_fixedN"]),
                           ("box_fixedn", [r for r in sub if r["set"] == "box_fixedn"])) if g}
        S[f"action_{pname}_by_family"] = {
            f"{fam}|{st}": dict(err_spec=[min(num(r, "err_spec_rel") for r in g), max(num(r, "err_spec_rel") for r in g)],
                                err_abs=[min(num(r, "err_spec_rel") * nG[r["name"]] for r in g),
                                         max(num(r, "err_spec_rel") * nG[r["name"]] for r in g)])
            for fam in ("caseA", "caseB", "uniform", "lattice", "cluster") for st in ("legacy", "main", "check")
            for g in [[r for r in sub if r["family"] == fam and r["set"] == st]] if g}
        S[f"action_{pname}_abs_all"] = [min(num(r, "err_spec_rel") * nG[r["name"]] for r in sub),
                                        max(num(r, "err_spec_rel") * nG[r["name"]] for r in sub)]
    S["action_by_config_T7"] = {r["name"]: dict(err=num(r, "err_spec_rel"), err_abs=num(r, "err_spec_rel") * nG[r["name"]],
                                                eps_S=num(r, "eps_S"), eps_F=num(r, "eps_F"), eps_mesh=num(r, "eps_mesh"),
                                                lam_star=num(r, "lam_star_ref"), norm=nG[r["name"]], M=int(num(r, "M")),
                                                eta=num(r, "eta_actual"))
                                for r in act if r["params"] == "T7"}
    # box length at T7: medians over seeds of the absolute total and of each absolute budget term
    box = {}
    for st in ("box_fixedN", "box_fixedn"):
        g = [r for r in act if r["params"] == "T7" and r["set"] == st]
        for L in sorted({num(r, "L") for r in g}):
            gl = [r for r in g if num(r, "L") == L]
            med = lambda k: float(np.median([num(r, k) * nG[r["name"]] for r in gl]))              # noqa: E731
            box[f"{st}|L{L:g}"] = dict(N=int(num(gl[0], "N")), n=float(num(gl[0], "N")) / L ** 3,
                                       eta=num(gl[0], "eta_actual"), total_median=med("err_spec_rel"),
                                       total_range=[min(num(r, "err_spec_rel") * nG[r["name"]] for r in gl),
                                                    max(num(r, "err_spec_rel") * nG[r["name"]] for r in gl)],
                                       S_median=med("eps_S"), F_median=med("eps_F"), mesh_median=med("eps_mesh"))
    S["box_T7"] = box
    # probes at T7: maxima, smallest normalized reference action, and action error over the configuration's spectral error
    spec = {(r["name"], r["params"]): num(r, "err_spec_rel") for r in act}
    S["probes_T7"] = {}
    for grp, sets in (("legacy+main", ("legacy", "main")), ("check", ("check",)), ("box", ("box_fixedN", "box_fixedn"))):
        g = [r for r in probes if r["params"] == "T7" and r["set"] in sets]
        S["probes_T7"][grp] = {kind: dict(rel_max=max(num(r, "rel_err") for r in g if r["probe"] == kind),
                                          ref_action_min=min(num(r, "ref_action_opnorm") for r in g if r["probe"] == kind),
                                          over_spec_max=max(num(r, "rel_err") / spec[(r["name"], "T7")]
                                                            for r in g if r["probe"] == kind),
                                          frac_over_spec=float(np.mean([num(r, "rel_err") > spec[(r["name"], "T7")]
                                                                        for r in g if r["probe"] == kind])))
                               for kind in ("gauss", "long", "trans")}
    S["configs"] = {r["name"]: dict(N=int(r["N"]), L=float(r["L"]), n=float(r["n"]), d_min=float(r["d_min"]),
                                    nn_mean=float(r["nn_mean"]), generator=json.loads(r["generator"])) for r in cfg}
    (OUT / "summary.json").write_text(json.dumps(S, indent=1, default=float) + "\n")
    return S


def main():
    style()
    FIG.mkdir(exist_ok=True)
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    cfg, ref = rd("configs.csv"), rd("reference_ladder.csv")
    quad, sp, so = rd("quadrature.csv"), rd("split_pairs.csv"), rd("split_operator.csv")
    cut, mesh, dec = rd("cutoffs.csv"), rd("mesh.csv"), rd("mesh_decomposition.csv")
    act, probes = rd("action.csv"), rd("action_probes.csv")
    cutfits, msum = fit_cutoffs(cut), fit_mesh(mesh)
    write_csv(OUT / "fits_cutoffs.csv", cutfits)
    write_csv(OUT / "fits_mesh.csv", msum)
    prec = rd("fourier_precision.csv") if (OUT / "fourier_precision.csv").exists() else []
    S = summary(ref, quad, sp, so, cutfits, msum, dec, act, probes, cfg, prec)
    f1 = components_figure(cut, mesh, msum, so, sp, quad)
    refn = {r["name"]: num(r, "norm2") for r in ref if num(r, "R") == 45.0}
    f2 = action_figure([r for r in act if r["set"] in ("legacy", "main", "check")], probes,
                       [r for r in act if r["set"].startswith("box") and r["params"] == "T7"], refn)
    for fig, nm in ((f1, "exp1_components"), (f2, "exp1_action")):
        fig.savefig(FIG / f"{nm}.pdf")
        fig.savefig(FIG / f"{nm}.png")
        fig.savefig(PAPER / "figures" / f"{nm}.pdf")
    tables(cfg, [r for r in act if r["set"] in ("legacy", "main", "check")])
    print(json.dumps({k: v for k, v in S.items() if k not in ("configs", "action_by_config_T7")}, indent=1,
                     default=float)[:6000])


if __name__ == "__main__":
    main()
