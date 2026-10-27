"""Reproduce the three-method comparison figures from saved CSV files only.

Run ``python plot_results.py`` after the experiments.  Every figure is exported
as PNG and SVG, with its actual plotted statistics in ``figures/plot_data``.
Missing optional columns/files skip individual panels and are reported in
``figures/plot_status.json``. No old experiment or result file is imported or
modified. Zero corrections are preserved in CSV and explicitly marked on log
axes; displaying a zero at a plotting floor does not turn it into a measurement.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
METHODS = ("PPPM-original", "B2-positive", "PPPM-grid-PSD")
LABEL = {
    "PPPM-original": "Original PPPM",
    "B2-positive": "B2 Positive Mesh",
    "PPPM-grid-PSD": "Deconvolved PPPM + PSD",
}
COLOR = {"PPPM-original": "#2874b2", "B2-positive": "#df7b24", "PPPM-grid-PSD": "#218c64"}
MARKERS = ("o", "s", "^", "D", "P", "X")
LINESTYLES = ("-", "--", "-.", ":")
STATUS = {"generated": [], "skipped": [], "notes": []}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 9,
        "axes.titleweight": "bold", "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 6.5, "axes.linewidth": .7,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "axes.grid": True, "grid.color": "#dddddd",
        "grid.linewidth": .5, "legend.frameon": True, "legend.fancybox": False,
        "legend.edgecolor": "#b8b8b8", "lines.linewidth": 1.2,
        "lines.markersize": 3.8, "svg.fonttype": "none", "savefig.dpi": 240,
    })


def numeric(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def read_csv(path):
    if not path.exists():
        STATUS["notes"].append(f"Missing optional file: {path.name}")
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
            if key in ("id", "reference_id", "method", "reference_method", "family", "config", "sweep", "setting",
                       "source", "source_path", "stage", "status", "failure", "phase", "selection", "note", "timing_batch"):
                continue
            number = numeric(value)
            if number is not None:
                row[key] = number
        row["method"] = canonical(row.get("method", ""))
    return rows


def canonical(method):
    return {"PPPM": "PPPM-original", "original": "PPPM-original", "B2": "B2-positive",
            "grid-PSD": "PPPM-grid-PSD", "projected": "PPPM-grid-PSD"}.get(str(method), str(method))


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ["no_data"])
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name, rows, args):
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"{name}.{extension}", dpi=args.dpi,
                    bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    csv_write(args.output / "plot_data" / f"{name}.csv", rows)
    STATUS["generated"].append({"figure": name, "plotted_records": len(rows)})


def skip(name, reason):
    STATUS["skipped"].append({"figure": name, "reason": reason})


def present_methods(rows):
    found = {r.get("method") for r in rows}
    return [method for method in METHODS if method in found]


def families(rows):
    found = {str(r.get("family", "all")) for r in rows}
    preferred = [f for f in ("uniform", "moderate", "strong") if f in found]
    return preferred + sorted(found - set(preferred))


def fmarker(family, all_families):
    return MARKERS[all_families.index(family) % len(MARKERS)]


def mixed_fit(fit, rows):
    if fit.get("method") != "PPPM-grid-PSD":
        return False
    selected = [r for r in rows if r.get("method") == fit.get("method") and r.get("p") == fit.get("p")
                and r.get("family") == fit.get("family") and numeric(r.get("M")) is not None
                and fit.get("Mlo", math.inf) <= r["M"] <= fit.get("Mhi", -math.inf)]
    flags = [numeric(r.get("negative_blocks", r.get("neg_fraction"))) for r in selected]
    flags = [v > 0 for v in flags if v is not None]
    return bool(flags) and any(flags) and not all(flags)


def aggregate(rows, xkey, ykey):
    groups = defaultdict(list)
    for row in rows:
        x, y = numeric(row.get(xkey)), numeric(row.get(ykey))
        if x is not None and y is not None:
            groups[x].append(y)
    return [dict(x=x, median=float(np.median(y)), min=min(y), max=max(y),
                 n=len(y), n_zero=sum(v == 0 for v in y)) for x, y in sorted(groups.items())]


def floor_for(rows, fields):
    positive = [numeric(row.get(field)) for row in rows for field in fields]
    positive = [x for x in positive if x is not None and x > 0]
    return min(positive) * .25 if positive else 1e-18


@lru_cache(maxsize=256)
def last_reciprocal_shell(nmax):
    """Largest sum of three squares <= nmax; identical to saved shell snapping."""
    for n in range(nmax, 0, -1):
        for a in range(math.isqrt(n) + 1):
            for b in range(math.isqrt(n - a*a) + 1):
                c2 = n - a*a - b*b
                if math.isqrt(c2)**2 == c2:
                    return n
    return None


def fourier_x(row):
    actual = numeric(row.get("sF_eff"))
    if actual is not None:
        return actual
    # Older new-method checkpoints did not save sF_eff. Recover the exact same
    # reciprocal-shell parameter from recorded L/xi/request, without importing
    # or running an experiment and without changing its saved result CSV.
    L, xi = numeric(row.get("L")), numeric(row.get("xi"))
    requested = numeric(row.get("sF", row.get("sF_req", row.get("value"))))
    if L is not None and xi is not None and requested is not None and L > 0 and xi > 0:
        nmax = math.floor((xi*requested*L/math.pi)**2 * (1 + 1e-12))
        shell = last_reciprocal_shell(nmax)
        if shell is not None:
            return math.pi*math.sqrt(shell)/(L*xi)
    return row.get("value")


def draw_curve(ax, stats, color, marker="o", ls="-", label=None, floor=None):
    if not stats:
        return
    x = np.asarray([v["x"] for v in stats])
    median = np.asarray([v["median"] for v in stats])
    lo = np.asarray([v["min"] for v in stats])
    hi = np.asarray([v["max"] for v in stats])
    if floor is not None:
        # Keep true zeros in CSV; only the display coordinate uses a floor.
        plotted = np.where(median == 0, floor, median)
        lo = np.where(lo == 0, floor, lo)
        hi = np.where(hi == 0, floor, hi)
    else:
        plotted = median
    ax.fill_between(x, lo, hi, color=color, alpha=.09, lw=0)
    ax.plot(x, plotted, color=color, marker=marker, ls=ls, label=label,
            mfc="white" if ls != "-" else color, ms=3.5)
    if floor is not None and np.any(median == 0):
        mask = median == 0
        ax.scatter(x[mask], plotted[mask], marker="v", s=27, facecolors="white",
                   edgecolors=color, linewidths=.9, zorder=6)


def append_stats(output, stats, **metadata):
    output.extend(dict(metadata, **record) for record in stats)


def blank(ax, note="No saved data"):
    ax.text(.5, .5, note, transform=ax.transAxes, ha="center", va="center", fontsize=8,
            color="#777777")


def method_handles(methods=METHODS):
    return [Line2D([], [], color=COLOR[m], label=LABEL[m], lw=1.8) for m in methods]


def timing_batch(row):
    return str(row.get("timing_batch") or "unrecorded")


def batch_note(rows):
    grouped = defaultdict(set)
    for row in rows:
        if numeric(row.get("epsilon")) is not None:
            grouped[row["epsilon"]].add(timing_batch(row))
    return "; ".join(f"{epsilon:.0e}: {','.join(sorted(batches))}" for epsilon, batches in sorted(grouped.items(), reverse=True))


def six_sweeps(rows, metric, args):
    name = "fig1_operator_six_sweeps" if metric == "E_G" else "fig2_action_six_sweeps"
    if not any(numeric(r.get(metric)) is not None for r in rows):
        skip(name, f"No {metric} values")
        return
    fig = plt.figure(figsize=(11.2, 6.7))
    grid = fig.add_gridspec(2, 3, hspace=.42, wspace=.32)
    cutoff = grid[0, 1].subgridspec(1, 2, wspace=.31)
    specs = [
        (fig.add_subplot(grid[0, 0]), "sR", "(a) Real-space cutoff", r"$s_R=\xi r_c$", False),
        (fig.add_subplot(cutoff[0]), "sF", "(b1) Fourier cutoff", r"$s_F=k_c/(2\xi)$", False),
        (fig.add_subplot(cutoff[1]), "sT", "(b2) B2 image window", r"$s_T=\kappa R_T$", False),
        (fig.add_subplot(grid[0, 2]), "M", "(c) FFT mesh size", r"$M$", True),
        (fig.add_subplot(grid[1, 0]), "p", "(d) Assignment order", r"$p$", False),
        (fig.add_subplot(grid[1, 1]), "rho", "(e) Particle density", r"$\rho$", False),
        (fig.add_subplot(grid[1, 2]), "kappa", "(f) Screening parameter", r"$\kappa$", True),
    ]
    all_families = families(rows)
    settings = sorted({str(r.get("setting", "baseline")) for r in rows},
                      key=lambda value: (value != "baseline", value))
    output = []
    floor = floor_for(rows, [metric])
    for ax, sweep, title, xlabel, logx in specs:
        subset = [r for r in rows if str(r.get("sweep", "")) == sweep]
        # Accept a combined cutoff name if the method implies its meaning.
        subset += [r for r in rows if str(r.get("sweep", "")) in ("sF/sT", "cutoff")
                   and ((sweep == "sT" and r.get("method") == "B2-positive")
                        or (sweep == "sF" and r.get("method") != "B2-positive"))]
        if not subset:
            blank(ax)
        for method in present_methods(subset):
            for family in all_families:
                for si, setting in enumerate(settings):
                    selected = [r for r in subset if r.get("method") == method
                                and str(r.get("family", "all")) == family
                                and str(r.get("setting", "baseline")) == setting]
                    # Prefer the actual shell-snapped Fourier cutoff over its request.
                    selected = [dict(r, plot_x=fourier_x(r))
                                if sweep == "sF" else dict(r, plot_x=r.get("value")) for r in selected]
                    stats = aggregate(selected, "plot_x", metric)
                    draw_curve(ax, stats, COLOR[method], fmarker(family, all_families),
                               LINESTYLES[si % len(LINESTYLES)], floor=floor)
                    append_stats(output, stats, sweep=sweep, method=method, family=family,
                                 setting=setting, metric=metric, zero_display_floor=floor)
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        if sweep == "p":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title, fontsize=8 if sweep in ("sF", "sT") else 9)
        ax.set_xlabel(xlabel)
        if sweep in ("sR", "p"):
            ax.set_ylabel(r"$E_\Gamma$ (relative operator norm)" if metric == "E_G"
                          else r"$E_{\Gamma X}$ (relative action error)")
    handles = method_handles(present_methods(rows))
    handles += [Line2D([], [], color="#444444", marker=fmarker(f, all_families), ls="", label=f)
                for f in all_families]
    handles += [Line2D([], [], color="#444444", ls=LINESTYLES[i % len(LINESTYLES)], label=s)
                for i, s in enumerate(settings)]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.005))
    fig.text(.5, .075, "Lines: medians; bands: seed min–max. Fourier and B2 image cutoffs have different definitions.",
             ha="center", fontsize=7)
    fig.subplots_adjust(bottom=.19)
    save(fig, name, output, args)


def convergence(rows, fits, args):
    name = "fig3_mesh_convergence"
    if not rows:
        skip(name, "No convergence.csv rows")
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), sharey=True)
    ps = sorted({numeric(r.get("p")) for r in rows if numeric(r.get("p")) is not None})
    pc = {p: plt.get_cmap("viridis")((i + .4) / max(len(ps), 1)) for i, p in enumerate(ps)}
    all_families = families(rows)
    output = []
    floor = floor_for(rows, ["E_mesh", "E_G_mesh_only"])
    for ax, method in zip(axes, METHODS):
        subset = [dict(r, error=r.get("E_mesh", r.get("E_G_mesh_only")))
                  for r in rows if r.get("method") == method]
        for p in ps:
            for fi, family in enumerate(all_families):
                selected = [r for r in subset if r.get("p") == p and str(r.get("family", "all")) == family]
                stats = aggregate(selected, "M", "error")
                draw_curve(ax, stats, pc[p], fmarker(family, all_families),
                           LINESTYLES[fi % len(LINESTYLES)], floor=floor)
                intervals = [r for r in fits if r.get("method") == method and r.get("p") == p
                             and str(r.get("family", "all")) == family
                             and numeric(r.get("Mlo")) is not None and numeric(r.get("Mhi")) is not None]
                for interval in intervals:
                    segment = [r for r in stats if interval["Mlo"] <= r["x"] <= interval["Mhi"]]
                    if segment:
                        ax.plot([r["x"] for r in segment], [max(r["median"], floor) for r in segment],
                                color=pc[p], lw=2.5, alpha=.8, zorder=3)
                for record in stats:
                    record["in_fit_interval"] = int(any(r["Mlo"] <= record["x"] <= r["Mhi"] for r in intervals))
                append_stats(output, stats, method=method, family=family, p=p, metric="E_mesh",
                             zero_display_floor=floor)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$M$  ($h=L/M$)")
        ax.set_title(LABEL[method], color=COLOR[method])
        if not subset:
            blank(ax)
    axes[0].set_ylabel(r"$\|\Gamma_{h,F}-\Gamma_{D,F}\|_2/\|\Gamma_D\|_2$")
    handles = [Line2D([], [], color=pc[p], label=f"p={p:g}") for p in ps]
    handles += [Line2D([], [], color="#444444", marker=fmarker(f, all_families),
                       ls=LINESTYLES[i % len(LINESTYLES)], label=f) for i, f in enumerate(all_families)]
    fig.legend(handles=handles, loc="lower center", ncol=max(3, min(7, len(handles))))
    for threshold in sorted({r["floor"] for r in fits if numeric(r.get("floor")) is not None}):
        for ax in axes:
            ax.axhline(threshold, color="#777777", ls=":", lw=.8)
    fig.text(.5, .12, "Bands: seed min–max. Thick segments: recorded fit intervals; dotted horizontal: fit exclusion threshold (not measured data).",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .19, 1, 1))
    save(fig, name, output, args)
    if not any(numeric(r.get("order")) is not None for r in fits):
        skip("fig3b_observed_mesh_orders", "No mesh_rate_fits.csv fitted orders")
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    output = []
    for method in present_methods(fits):
        for fi, family in enumerate(families(fits)):
            selected = [r for r in fits if r.get("method") == method and str(r.get("family", "all")) == family]
            stats = aggregate(selected, "p", "order")
            draw_curve(ax, stats, COLOR[method], MARKERS[fi % len(MARKERS)],
                       LINESTYLES[fi % len(LINESTYLES)], label=f"{LABEL[method]} / {family}")
            append_stats(output, stats, method=method, family=family, metric="fitted_order")
    transitions = [r for r in fits if mixed_fit(r, rows)]
    for r in transitions:
        ax.scatter([r["p"]], [r["order"]], s=36, marker="o", facecolors="white",
                   edgecolors=COLOR[r["method"]], linewidths=1.2, zorder=6)
    if ps:
        ax.plot([min(ps), max(ps)], [min(ps), max(ps)], ":", color="black", label=r"guide: order $=p$")
        ax.axhline(2, color="#777777", ls="--", lw=.8, label="guide: order = 2")
    if transitions:
        ax.plot([], [], ls="", marker="o", mfc="white", mec="black",
                label="mixed active/inactive interval (not an order)")
    ax.set_xlabel("Assignment order p"); ax.set_ylabel("Formal fitted log–log slope q")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Recorded interval slopes; projection transitions are not convergence orders")
    ax.legend(ncol=2, fontsize=6.4)
    fig.text(.5, .015, "Large slopes across projection shutoff/cancellation are transition diagnostics, not high-order certificates.",
             ha="center", fontsize=6.6)
    fig.tight_layout(rect=(0, .05, 1, 1))
    # Include the actual fit interval for reproducibility, not just the aggregate.
    output += [dict(row, record_type="fit_input") for row in fits]
    save(fig, "fig3b_observed_mesh_orders", output, args)


def projection_diagnostics(rows, args):
    name = "fig4_projection_error_and_correction"
    projected = [dict(r) for r in rows if r.get("method") == "PPPM-grid-PSD"]
    if not projected:
        skip(name, "No projected-method convergence rows")
        return
    match_keys = ("family", "seed", "M", "p")
    originals = {tuple(r.get(k) for k in match_keys): r for r in rows if r.get("method") == "PPPM-original"}
    for row in projected:
        original = originals.get(tuple(row.get(k) for k in match_keys), {})
        row["E_before"] = row.get("E_original", original.get("E_G"))
        row["E_mesh_before"] = row.get("E_original_mesh", original.get("E_mesh"))
        row["neg_percent"] = 100 * row["neg_fraction"] if numeric(row.get("neg_fraction")) is not None else None
        row["max_correction_spectral"] = (row.get("correction_max_spectral")
                if numeric(row.get("correction_max_spectral")) is not None else row.get("max_correction"))
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4))
    panels = [
        (axes[0, 0], [("E_mesh_before", "before", "--"), ("E_mesh", "after", "-")],
         "(a) Mesh-only error", r"Relative mesh error", True),
        (axes[0, 1], [("E_before", "before", "--"), ("E_G", "after", "-")],
         "(b) Complete operator error", r"$E_\Gamma$", True),
        (axes[0, 2], [("E_proj", "operator correction", "-")],
         "(c) Operator correction", r"$\|\Gamma_{\rm PSD}-\Gamma_{\rm original}\|_2/\|\Gamma_D\|_2$", True),
        (axes[1, 0], [("neg_percent", "negative fraction", "-")],
         "(d) Raw grid blocks needing projection", "Negative-eigenvalue fraction (%)", True),
        (axes[1, 1], [("min_raw_eig", "raw minimum eigenvalue", "-")],
         "(e) Smallest raw grid eigenvalue", "Minimum raw eigenvalue", False),
        (axes[1, 2], [("max_correction_spectral", "max block spectral correction", "-")],
         "(f) Largest node correction", "Maximum block correction (spectral norm)", True),
    ]
    ps = sorted({r["p"] for r in projected if numeric(r.get("p")) is not None})
    pc = {p: plt.get_cmap("viridis")((i + .4) / max(len(ps), 1)) for i, p in enumerate(ps)}
    all_families = families(projected)
    output = []
    for ax, metrics, title, ylabel, logy in panels:
        floor = floor_for(projected, [metric for metric, _, _ in metrics])
        count = 0
        for p in ps:
            for family in all_families:
                selected = [r for r in projected if r.get("p") == p and str(r.get("family", "all")) == family]
                for metric, phase, ls in metrics:
                    stats = aggregate(selected, "M", metric)
                    count += len(stats)
                    draw_curve(ax, stats, pc[p], fmarker(family, all_families), ls,
                               floor=floor if logy else None)
                    append_stats(output, stats, metric=metric, phase=phase, method="PPPM-grid-PSD",
                                 p=p, family=family, zero_display_floor=floor if logy else "")
        if not count:
            blank(ax, "Column not available")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1e-15)
            ax.axhline(0, color="#777777", ls=":", lw=.8)
        ax.set_title(title); ax.set_xlabel("Mesh size M"); ax.set_ylabel(ylabel)
    handles = [Line2D([], [], color=pc[p], label=f"p={p:g}") for p in ps]
    handles += [Line2D([], [], color="#444444", marker=fmarker(f, all_families), ls="", label=f)
                for f in all_families]
    handles += [Line2D([], [], color="#444444", ls="--", label="before projection"),
                Line2D([], [], color="#444444", ls="-", label="after projection"),
                Line2D([], [], color="#444444", marker="v", mfc="white", ls="", label="exact zero (display floor)")]
    fig.legend(handles=handles, loc="lower center", ncol=5)
    fig.text(.5, .11, "Open downward triangles denote exact zero at a panel-specific display floor; CSV keeps zero and records the floor. "
             "Active/inactive projection regimes must be interpreted separately.", ha="center", fontsize=6.7)
    fig.tight_layout(rect=(0, .18, 1, 1), h_pad=1.4)
    save(fig, name, output, args)


def error_time(search, timing, frontier, epsilons, args):
    name = "fig5_error_vs_time_pareto"
    if not search and not timing and not frontier:
        skip(name, "No search/timing/frontier data")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9))
    output = []
    for method in present_methods(search):
        selected = [r for r in search if r.get("method") == method
                    and numeric(r.get("t_screen_s")) is not None and numeric(r.get("E_G")) is not None
                    and r["t_screen_s"] > 0 and r["E_G"] > 0]
        axes[0].scatter([r["t_screen_s"] * 1e3 for r in selected], [r["E_G"] for r in selected],
                        s=12, alpha=.4, color=COLOR[method], label=LABEL[method])
        output += [dict(r, dataset="screening", x_ms=r["t_screen_s"] * 1e3, y=r["E_G"]) for r in selected]
    validated = {}
    # The main confirmation Pareto must use this batch's root final_timing.csv
    # only. Secondary accuracy tiers may come from the archived primary pass.
    for row in timing:
        x, y = numeric(row.get("median_matvec_s")), numeric(row.get("max_E_G"))
        if x is not None and y is not None and x > 0 and y > 0:
            validated[(row.get("method"), str(row.get("id")), x, y)] = row
    validated = list(validated.values())
    for mi, method in enumerate(present_methods(validated)):
        selected = [r for r in validated if r.get("method") == method]
        feasible = [r for r in selected if r["max_E_G"] <= 1e-4]
        infeasible = [r for r in selected if r["max_E_G"] > 1e-4]
        axes[1].scatter([r["median_matvec_s"] * 1e3 for r in feasible], [r["max_E_G"] for r in feasible],
                        s=31, marker="o", color=COLOR[method], label=LABEL[method], zorder=4)
        axes[1].scatter([r["median_matvec_s"] * 1e3 for r in infeasible], [r["max_E_G"] for r in infeasible],
                        s=31, marker="o", edgecolors=COLOR[method], facecolors="white", zorder=4)
        pareto = []
        best_error = math.inf
        for row in sorted(feasible, key=lambda r: (r["median_matvec_s"], r["max_E_G"])):
            if row["max_E_G"] < best_error:
                best_error = row["max_E_G"]
                pareto.append(row)
        axes[1].plot([r["median_matvec_s"] * 1e3 for r in pareto], [r["max_E_G"] for r in pareto],
                     color=COLOR[method], lw=1.3)
        if feasible:
            winner = min(feasible, key=lambda r: r["median_matvec_s"])
            axes[1].scatter([winner["median_matvec_s"]*1e3], [winner["max_E_G"]],
                            marker="*", s=115, color=COLOR[method], edgecolors="black", linewidths=.5, zorder=7)
            parameter_text = ", ".join(f"{key}={numeric(winner.get(key)):g}" for key in ("xi", "sR", "M", "p")
                                       if numeric(winner.get(key)) is not None)
            axes[1].annotate(f"{LABEL[method]}: {parameter_text}",
                             xy=(winner["median_matvec_s"]*1e3, winner["max_E_G"]),
                             xytext=(.025, .96-mi*.10), textcoords="axes fraction", fontsize=6.1,
                             color=COLOR[method], va="top",
                             bbox=dict(boxstyle="square,pad=.15", fc="white", ec="none", alpha=.8))
        output += [dict(r, dataset="validated", x_ms=r["median_matvec_s"] * 1e3, y=r["max_E_G"],
                        on_pareto=int(r in pareto), feasible_main=int(r in feasible),
                        fastest_main=int(bool(feasible) and r is winner)) for r in selected]
    for i, ax in enumerate(axes):
        for epsilon in epsilons:
            ax.axhline(epsilon, color="#777777", ls=":", lw=.8)
            output.append(dict(dataset="target", panel=i, epsilon=epsilon))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Cached complete matvec time (ms)")
        ax.set_ylabel(r"$E_\Gamma$ on screening configuration" if i == 0 else r"Maximum validated $E_\Gamma$")
        if ax.has_data():
            ax.legend(loc="best")
        else:
            blank(ax)
    axes[0].set_title("(a) Screening under concurrent accuracy load\nHeuristic times only")
    axes[1].set_title(r"(b) Main confirmation batch; Pareto at $E_\Gamma\leq10^{-4}$")
    fig.text(.5, .025, "Dotted: target errors. Open circles: main-target infeasible, never joined. Stars: fastest main-target feasible points.\n"
             "Screen times were measured under concurrent accuracy load and are search heuristics only.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .09, 1, 1))
    save(fig, name, output, args)


def frontier_figures(rows, epsilons, args):
    valid = [r for r in rows if numeric(r.get("epsilon")) is not None
             and numeric(r.get("median_matvec_s")) is not None and r["median_matvec_s"] > 0
             and numeric(r.get("max_E_G")) is not None and r["max_E_G"] <= r["epsilon"]]
    if not valid:
        skip("fig6_optimal_speed_frontier", "No validated attainable frontier points")
        skip("fig7_optimal_parameters", "No validated attainable frontier points")
        return
    # If a schema contains multiple candidates per method/target, take the fastest.
    groups = defaultdict(list)
    for row in valid:
        groups[(row["method"], row["epsilon"])].append(row)
    best = [min(group, key=lambda r: r["median_matvec_s"]) for group in groups.values()]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))
    output = []
    original_time = {(r["epsilon"], timing_batch(r)): r["median_matvec_s"] for r in best if r["method"] == "PPPM-original"}
    b2_time = {(r["epsilon"], timing_batch(r)): r["median_matvec_s"] for r in best if r["method"] == "B2-positive"}
    eps_batches = defaultdict(set)
    for row in best:
        eps_batches[row["epsilon"]].add(timing_batch(row))
    inconsistent = {e for e, batches in eps_batches.items() if len(batches) > 1}
    if inconsistent:
        STATUS["notes"].append(f"Speed ratios omitted for mixed timing batches at epsilon={sorted(inconsistent)}")
    for mi, method in enumerate(METHODS):
        selected = sorted([r for r in best if r.get("method") == method], key=lambda r: r["epsilon"])
        axes[0].plot([r["epsilon"] for r in selected], [r["median_matvec_s"] * 1e3 for r in selected],
                     color=COLOR[method], marker=MARKERS[mi], label=LABEL[method])
        speedup = [dict(r, speedup_original=original_time[(r["epsilon"], timing_batch(r))] / r["median_matvec_s"])
                   for r in selected if (r["epsilon"], timing_batch(r)) in original_time and r["epsilon"] not in inconsistent]
        axes[1].plot([r["epsilon"] for r in speedup], [r["speedup_original"] for r in speedup],
                     color=COLOR[method], marker=MARKERS[mi], label=LABEL[method])
        b2_speedup = [dict(r, speedup_b2=b2_time[(r["epsilon"], timing_batch(r))] / r["median_matvec_s"])
                      for r in selected if (r["epsilon"], timing_batch(r)) in b2_time and r["epsilon"] not in inconsistent]
        axes[2].plot([r["epsilon"] for r in b2_speedup], [r["speedup_b2"] for r in b2_speedup],
                     color=COLOR[method], marker=MARKERS[mi], label=LABEL[method])
        output += [dict(r, metric="time", x=r["epsilon"], y=r["median_matvec_s"] * 1e3) for r in selected]
        output += [dict(r, metric="speedup", x=r["epsilon"], y=r["speedup_original"]) for r in speedup]
        output += [dict(r, metric="speedup_b2", x=r["epsilon"], y=r["speedup_b2"]) for r in b2_speedup]
    for ax in axes:
        ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel(r"Required error $\epsilon$ (stricter to right)")
        ax.legend()
    axes[0].set_yscale("log"); axes[0].set_ylabel("Fastest validated matvec (ms)")
    axes[0].set_title("(a) Equal-accuracy optimum")
    axes[1].axhline(1, color="#777777", ls=":")
    axes[1].set_ylabel("Original PPPM time / method time")
    axes[1].set_title("(b) Relative speed at equal target error")
    axes[2].axhline(1, color="#777777", ls=":")
    axes[2].set_ylabel("B2 time / method time")
    axes[2].set_title("(c) Speed relative to B2")
    missing = [f"{LABEL[m]}: {e:.0e}" for e in epsilons for m in METHODS
               if not any(r["method"] == m and r["epsilon"] == e for r in best)]
    note = "Unattained or unmeasured targets have no speedup point."
    if missing:
        note += " Missing: " + "; ".join(missing)
    note += "\nTiming batches by target: " + batch_note(best) + ". Ratios use the same target and batch only."
    fig.text(.5, .015, note, ha="center", fontsize=6.1, wrap=True)
    fig.tight_layout(rect=(0, .11, 1, 1))
    output += [dict(r, record_type="unattained_or_invalid") for r in rows if r not in valid]
    save(fig, "fig6_optimal_speed_frontier", output, args)
    fig = plt.figure(figsize=(11.2, 6.2))
    grid = fig.add_gridspec(2, 3, hspace=.42, wspace=.32)
    cutoff = grid[0, 2].subgridspec(1, 2, wspace=.53)
    specs = [(fig.add_subplot(grid[0, 0]), "xi", r"$\xi$"),
             (fig.add_subplot(grid[0, 1]), "sR", r"$s_R=\xi r_c$"),
             (fig.add_subplot(cutoff[0]), "sF", r"PPPM $s_F$ (pre-projection)"),
             (fig.add_subplot(cutoff[1]), "sT", r"B2 $s_T=\kappa R_T$"),
             (fig.add_subplot(grid[1, 0]), "M", "Mesh size M"),
             (fig.add_subplot(grid[1, 1]), "p", "Assignment order p"),
             (fig.add_subplot(grid[1, 2]), "max_E_G", "Maximum validated error")]
    output = []
    for ax, field, ylabel in specs:
        for mi, method in enumerate(METHODS):
            selected = []
            for row in best:
                if row["method"] != method:
                    continue
                if (field == "sF" and method == "B2-positive") or (field == "sT" and method != "B2-positive"):
                    continue
                value = row.get(field)
                if field in ("sF", "sT"):
                    value = row.get("sF/sT", row.get("cutoff")) if numeric(value) is None else value
                if numeric(value) is not None:
                    selected.append(dict(row, plot_y=value))
            selected.sort(key=lambda r: r["epsilon"])
            ax.plot([r["epsilon"] for r in selected], [r["plot_y"] for r in selected],
                    color=COLOR[method], marker=MARKERS[mi], label=LABEL[method])
            output += [dict(r, parameter=field, x=r["epsilon"], y=r["plot_y"]) for r in selected]
        ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel(r"Target $\epsilon$"); ax.set_ylabel(ylabel)
        if field in ("M", "p"):
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if field == "max_E_G":
            ax.set_yscale("log")
            ax.plot(epsilons, epsilons, ":", color="black", lw=.8)
        if not ax.has_data():
            blank(ax)
    fig.legend(handles=method_handles(present_methods(best)), loc="lower center", ncol=3)
    fig.text(.5, .075, "Separate cutoff facets: sF (PPPM before projection) / sT (B2 image window); values are not equivalent.\n"
             "Timing batches by target: " + batch_note(best) + ". Each target is compared within its own batch.",
             ha="center", fontsize=6.3)
    fig.subplots_adjust(bottom=.21)
    save(fig, "fig7_optimal_parameters", output, args)


def setup_components(timing, components, args):
    if not timing:
        skip("fig8_setup_and_cached_cost", "No final_timing.csv rows")
        return
    # All separately timed validated candidates are shown; ids prevent hidden selection.
    selected = [r for r in timing if numeric(r.get("median_matvec_s")) is not None]
    if not selected:
        skip("fig8_setup_and_cached_cost", "No complete matvec timing values")
        return
    selected.sort(key=lambda r: (METHODS.index(r["method"]) if r["method"] in METHODS else 99,
                                 r.get("median_matvec_s", math.inf)))
    def short_label(row):
        def value(key):
            number = numeric(row.get(key))
            return f"{number:g}" if number is not None else "N/A"
        cutoff = "sT" if row["method"] == "B2-positive" else "sF"
        return (f"{row['method']}\n"
                f"xi={value('xi')}, M={value('M')}, p={value('p')}\n"
                f"sR={value('sR')}, {cutoff}={value(cutoff)}")
    labels = [short_label(r) for r in selected]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3))
    fields = [("median_matvec_s", "(a) Cached complete matvec"),
              ("median_config_setup_s", "(b) Per-configuration setup"),
              ("median_table_setup_s", "(c) Per-parameter table setup")]
    output = []
    for ax, (field, title) in zip(axes, fields):
        for i, row in enumerate(selected):
            y = numeric(row.get(field))
            if y is not None:
                ax.bar(i, y * 1e3, color=COLOR.get(row["method"], "#777777"), width=.65)
                ax.text(i, y * 1e3, f"{y*1e3:.3g}", ha="center", va="bottom", fontsize=6)
                output.append(dict(row, phase=field, x=i, y_ms=y * 1e3))
        ax.set_xticks(range(len(selected))); ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7)
        ax.set_title(title); ax.set_ylabel("Time (ms)")
        ax.margins(y=.17)
    fig.tight_layout()
    save(fig, "fig8_setup_and_cached_cost", output, args)
    if not components:
        skip("fig8b_matvec_components", "No components_raw.csv rows")
        return
    phase_names = sorted({str(r.get("phase", "")) for r in components})
    # Total timings are inclusive; plotting them as stacked phases would double count.
    phase_names = [p for p in phase_names if p.lower() not in ("total", "matvec", "complete_matvec", "matvec_total")]
    if not phase_names:
        skip("fig8b_matvec_components", "Only inclusive total timings; no disjoint phase timings")
        return
    fig, ax = plt.subplots(figsize=(max(7., len(selected)*.65), 4.3))
    output = []
    for pi, phase in enumerate(phase_names):
        color = plt.get_cmap("tab10")(pi % 10)
        for i, row in enumerate(selected):
            values = [r["seconds"] for r in components if r.get("method") == row["method"]
                      and str(r.get("id")) == str(row.get("id")) and str(r.get("phase")) == phase
                      and numeric(r.get("seconds")) is not None]
            if not values:
                continue
            med = float(np.median(values))*1e3
            offset = (pi - (len(phase_names)-1)/2) * .8/len(phase_names)
            ax.bar(i + offset, med, width=.8/len(phase_names), color=color,
                   label=phase if i == 0 else None)
            output.append(dict(method=row["method"], id=row.get("id"), phase=phase, n=len(values),
                               median_ms=med, min_ms=min(values)*1e3, max_ms=max(values)*1e3))
    ax.set_xticks(range(len(selected))); ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7)
    ax.set_ylabel("Phase time (ms); medians over recorded repeats")
    ax.set_title("Measured cost components (separately timed; overlapping phases must not be added)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=plt.get_cmap("tab10")(pi % 10), label=phase)
                       for pi, phase in enumerate(phase_names)], ncol=min(4, len(phase_names)))
    fig.tight_layout()
    save(fig, "fig8b_matvec_components", output, args)


def controlled_xi(rows, args):
    """The same numeric parameters across methods; distinct from optimization."""
    name = "fig9_xi_controlled_sweep"
    if not any(numeric(r.get("xi")) is not None and numeric(r.get("E_G")) is not None for r in rows):
        skip(name, "No controlled xi_sweep.csv observations")
        return
    all_families = ("uniform", "moderate", "strong")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True, sharey="row")
    output = []
    xi_values = sorted({r["xi"] for r in rows if numeric(r.get("xi")) is not None})
    for ri, metric in enumerate(("E_G", "E_GX")):
        floor = floor_for(rows, [metric])
        for ci, family in enumerate(all_families):
            ax = axes[ri, ci]
            for mi, method in enumerate(METHODS):
                selected = [r for r in rows if r.get("method") == method and r.get("family") == family]
                stats = aggregate(selected, "xi", metric)
                draw_curve(ax, stats, COLOR[method], marker=MARKERS[mi], ls="-", floor=floor)
                append_stats(output, stats, method=method, family=family, metric=metric,
                             zero_display_floor=floor)
            ax.set_yscale("log")
            if ri == 0:
                ax.set_title(family)
            if ri == 1:
                ax.set_xlabel(r"Ewald splitting parameter $\xi$")
                ax.set_xticks(xi_values)
                ax.set_xticklabels([f"{x:g}" for x in xi_values])
            if ci == 0:
                ax.set_ylabel(r"$E_\Gamma$ (relative operator norm)" if metric == "E_G"
                              else r"$E_{\Gamma X}$ (Gaussian action error)")
            if not ax.has_data():
                blank(ax)
    fig.legend(handles=[Line2D([], [], color=COLOR[m], marker=MARKERS[i], label=LABEL[m])
                        for i, m in enumerate(METHODS)], loc="lower center", ncol=3)
    fig.text(.5, .075, r"Controlled sweep: $N=256,\rho=.75,\kappa=.2,M=20,p=5,s_R=3$; "
             r"PPPM $s_F=3$, B2 $s_T=36$. Lines: seed medians; bands: seed min–max.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .13, 1, 1), h_pad=1.1)
    save(fig, name, output, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--epsilons", type=float, nargs="*", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    tables = {name: read_csv(args.results / f"{name}.csv") for name in
              ("sweeps", "convergence", "mesh_rate_fits", "search", "validation_summary",
               "final_timing", "best_configuration", "components_raw", "frontier", "xi_sweep")}
    if not tables["frontier"]:
        tables["frontier"] = read_csv(args.results / "accuracy_time_frontier.csv")
    epsilons = sorted(set(args.epsilons + [r["epsilon"] for r in tables["frontier"]
                       if numeric(r.get("epsilon")) is not None and r["epsilon"] > 0]))
    if not epsilons:
        epsilons = [1e-2, 1e-3, 1e-4]
        STATUS["notes"].append("No target errors recorded: used display guides 1e-2, 1e-3, 1e-4.")
    jobs = [
        ("operator_sweeps", lambda: six_sweeps(tables["sweeps"], "E_G", args)),
        ("action_sweeps", lambda: six_sweeps(tables["sweeps"], "E_GX", args)),
        ("mesh_convergence", lambda: convergence(tables["convergence"], tables["mesh_rate_fits"], args)),
        ("projection", lambda: projection_diagnostics(tables["convergence"], args)),
        ("error_time", lambda: error_time(tables["search"], tables["final_timing"], tables["frontier"], epsilons, args)),
        ("frontier", lambda: frontier_figures(tables["frontier"], epsilons, args)),
        ("timings", lambda: setup_components(tables["best_configuration"] or tables["final_timing"], tables["components_raw"], args)),
        ("controlled_xi", lambda: controlled_xi(tables["xi_sweep"], args)),
    ]
    for name, job in jobs:
        try:
            job()
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            plt.close("all")
            skip(name, f"{type(exc).__name__}: {exc}")
    STATUS["results_directory"] = str(args.results.resolve())
    STATUS["epsilons"] = epsilons
    (args.output / "plot_status.json").write_text(json.dumps(STATUS, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"figures_generated": len(STATUS["generated"]), "skipped": STATUS["skipped"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
