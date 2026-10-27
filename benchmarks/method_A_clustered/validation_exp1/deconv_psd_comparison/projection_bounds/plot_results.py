"""Plot projection-correction bounds from this directory's CSVs only.

No old experiment, timing, or implementation is imported. PNG/SVG figures,
their exact aggregate source CSVs, and a skip/coverage manifest are saved in
figures/. The true zero mask is preserved; logarithmic display floors are not
measurements. Tightness uses independently verified correction-only norms.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
PARAMETERS = ("N", "rho", "kappa", "xi", "sR", "sF", "M", "p")
COLORS = {"old": "#2874b2", "new": "#218c64", "projection": "#df7b24",
          "simple": "#9467bd", "degree": "#c9474f", "merged": "#222222"}
REGIME = {"active": "#c9474f", "mixed": "#e3a527", "inactive": "#a9b9c6"}
STATUS = {"generated": [], "skipped": [], "notes": []}
TEXT_FIELDS = {"dataset", "case_id", "parameter_id", "data_id", "source_row_id", "config", "family", "source_csv",
               "correction_norm_source", "correction_norm_kind", "source_sweep", "source_setting", "sweep", "setting",
               "selected_label", "label", "case_label", "role", "phase", "method", "status", "mesh_reference"}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def truth(value):
    return value is True or str(value).strip().lower() in ("true", "1", "1.0", "yes")


def read(path):
    if not path.exists():
        STATUS["notes"].append(f"Missing optional input: {path.name}")
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
            if key not in TEXT_FIELDS and number(value) is not None:
                row[key] = number(value)
        if not row.get("dataset") and row.get("source_csv"):
            row["dataset"] = Path(str(row["source_csv"])).stem
        row.setdefault("case_id", row.get("data_id", ""))
        row.setdefault("parameter_id", row.get("table_key", row.get("id", "")))
        if number(row.get("negative_fraction")) is None and number(row.get("negative_blocks")) is not None and number(row.get("M")):
            row["negative_fraction"] = row["negative_blocks"] / row["M"]**3
        row["negative_percent"] = row["negative_fraction"] * 100 if number(row.get("negative_fraction")) is not None else None
        for bound in ("B_simple", "B_degree", "B"):
            if number(row.get(bound + "_rel")) is None and number(row.get(bound)) is not None and number(row.get("normGammaD")):
                row[bound + "_rel"] = row[bound] / row["normGammaD"]
        if verified_norm(row) and number(row.get("correction_norm")) is not None and row["correction_norm"] > 0:
            for bound in ("B_simple", "B_degree", "B"):
                if number(row.get(bound)) is not None:
                    row[bound + "_tightness"] = row[bound] / row["correction_norm"]
    return rows


def verified_norm(row):
    explicit = row.get("correction_norm_verified", row.get("correction_only_verified"))
    if explicit not in (None, ""):
        return truth(explicit)
    source = str(row.get("correction_norm_source", "")).lower().replace("-", "_")
    if any(word in source for word in ("proxy", "historical", "empirical_eproj", "reused_e_proj")):
        return False
    return ("correction" in source and any(word in source for word in ("dense", "direct", "eigval", "only")))


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": .7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": .5,
        "legend.frameon": True, "legend.fancybox": False, "lines.linewidth": 1.2,
        "lines.markersize": 3.5, "svg.fonttype": "none"})


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ["no_data"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def save(fig, name, rows, args):
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"{name}.{ext}", dpi=args.dpi, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    csv_write(args.output / "plot_data" / f"{name}.csv", rows)
    STATUS["generated"].append({"figure": name, "plotted_records": len(rows)})


def skip(name, reason):
    STATUS["skipped"].append({"figure": name, "reason": reason})


def logical_key(row):
    return tuple(row.get(k) for k in ("dataset", "config") + PARAMETERS)


def unique(rows):
    # The same physical case can appear in multiple old one-parameter sweeps.
    # It must not receive extra statistical weight at the common baseline.
    seen = {}
    for row in rows:
        seen.setdefault(logical_key(row), row)
    return list(seen.values())


def unique_cases(rows):
    seen = {}
    for row in rows:
        key = row.get("case_id") or tuple(row.get(k) for k in ("config",)+PARAMETERS)
        seen.setdefault(key, row)
    return list(seen.values())


def independent_timing_samples(rows):
    seen = {}
    for i, row in enumerate(rows):
        shared = truth(row.get("parameter_sample_shared_across_configs")) or row.get("phase") in ("bound_parameter_setup", "correction_fft")
        key = (row.get("parameter_id"), row.get("repeat"), row.get("phase")) if shared else ("row", i)
        seen.setdefault(key, row)
    return list(seen.values())


def subset(rows, dataset):
    return [r for r in rows if str(r.get("dataset", "")).removesuffix(".csv") == dataset]


def floor_for(rows, fields):
    vals = [number(r.get(f)) for r in rows for f in fields]
    vals = [v for v in vals if v is not None and v > 0]
    return min(vals) * .25 if vals else 1e-18


def stats(rows, xkey, ykey, floor):
    grouped = defaultdict(list)
    for row in rows:
        if number(row.get(xkey)) is not None and number(row.get(ykey)) is not None:
            grouped[row[xkey]].append(row)
    output = []
    for x, group in sorted(grouped.items()):
        vals = [r[ykey] for r in group]
        median = float(np.median(vals))
        output.append(dict(x=x, median=median, min=min(vals), max=max(vals), n=len(vals),
                n_configs=len({r.get("config") for r in group}), n_zero=sum(v == 0 for v in vals),
                plotted_median=floor if median == 0 else median, zero_display_floor=floor,
                case_ids=";".join(sorted({str(r.get("case_id", r.get("data_id", ""))) for r in group})),
                parameter_ids=";".join(sorted({str(r.get("parameter_id", "")) for r in group}))))
    return output


def curve(ax, records, color, ls="-", marker="o", label=None, logy=True):
    if not records:
        return
    x = np.array([r["x"] for r in records]); y = np.array([r["median"] for r in records])
    floor = records[0]["zero_display_floor"]
    low = np.array([r["min"] for r in records]); high = np.array([r["max"] for r in records])
    if logy:
        ax.fill_between(x, np.where(low == 0, floor, low), np.where(high == 0, floor, high),
                        color=color, alpha=.09, lw=0)
        # Do not draw a false convergence segment through an exact-zero floor.
        ax.plot(x, np.where(y > 0, y, np.nan), color=color, ls=ls, marker=marker, label=label,
                mfc="white" if ls != "-" else color)
        mask = y == 0
        if mask.any():
            ax.scatter(x[mask], np.full(mask.sum(), floor), marker="v", s=29,
                       facecolors="white", edgecolors=color, linewidths=.9, zorder=6)
    else:
        ax.fill_between(x, low, high, color=color, alpha=.09, lw=0)
        ax.plot(x, y, color=color, ls=ls, marker=marker, label=label)


def add_stats(output, records, **metadata):
    output.extend(dict(metadata, **row) for row in records)


def active(row):
    if row.get("active") not in (None, ""):
        return truth(row["active"])
    return (number(row.get("negative_blocks")) or 0) > 0


def regime_strip(ax, rows, xkey, output, **metadata):
    grouped = defaultdict(list)
    for row in rows:
        if number(row.get(xkey)) is not None:
            grouped[row[xkey]].append(row)
    for x, group in sorted(grouped.items()):
        count = sum(active(r) for r in group)
        regime = "inactive" if count == 0 else "active" if count == len(group) else "mixed"
        ax.scatter([x], [.985], transform=ax.get_xaxis_transform(), marker="s", s=22,
                   c=REGIME[regime], edgecolors="white", linewidths=.3, zorder=8)
        output.append(dict(record_type="activity", x=x, active_count=count, n=len(group), regime=regime,
                           case_ids=";".join(sorted({str(r.get("case_id", "")) for r in group})), **metadata))


def regime_handles():
    return [Line2D([], [], ls="", marker="s", color=color, label=f"{regime} configurations")
            for regime, color in REGIME.items()]


def blank(ax, text="No saved observations"):
    ax.text(.5, .5, text, transform=ax.transAxes, ha="center", va="center", color="#777777", fontsize=8)


CORE = [("E_old", "old", "--", "o", "Old error (empirical)"),
        ("E_new", "new", "-", "s", "Projected error (empirical)"),
        ("E_proj", "projection", "-", "^", r"$E_{\rm proj}$ (comparison)"),
        ("B_simple_rel", "simple", ":", "D", r"$N\delta/\|\Gamma_D\|_2$"),
        ("B_degree_rel", "degree", "-.", "x", r"$2D_{\max}/\|\Gamma_D\|_2$"),
        ("B_rel", "merged", "--", "+", r"$\min(N\delta,2D_{\max})/\|\Gamma_D\|_2$")]


def core_figures(rows, args):
    conv = unique(subset(rows, "convergence"))
    if not conv:
        skip("fig1a_complete_error_bounds", "No convergence observations")
        skip("fig1b_mesh_error_bounds", "No convergence observations")
        return
    ps = [p for p in (3, 5, 7, 9) if any(r.get("p") == p for r in conv)]
    for mesh, name in [(False, "fig1a_complete_error_bounds"), (True, "fig1b_mesh_error_bounds")]:
        metrics = [("E_mesh_old" if mesh and key == "E_old" else "E_mesh_new" if mesh and key == "E_new" else key,
                    color, ls, marker, label.replace("error", "mesh-only error") if mesh and key in ("E_old", "E_new") else label)
                   for key, color, ls, marker, label in CORE]
        if mesh and not any(number(r.get("E_mesh_old")) is not None and number(r.get("E_mesh_new")) is not None for r in conv):
            skip(name, "No matched old/new mesh-only errors")
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.1), sharex=True, sharey=True)
        output = []
        floor = floor_for(conv, [key for key, *_ in metrics])
        for ax, p in zip(axes.ravel(), ps):
            group = [r for r in conv if r.get("p") == p]
            for key, color, ls, marker, label in metrics:
                records = stats(group, "M", key, floor)
                curve(ax, records, COLORS[color], ls, marker, label)
                add_stats(output, records, record_type="curve", p=p, metric=key, error_scope="mesh-only" if mesh else "complete")
            regime_strip(ax, group, "M", output, p=p)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"p={p}"); ax.set_xlabel("Mesh size M")
            ax.set_ylabel(r"Relative error / correction bound ($\|\Gamma_D\|_2$ normalization)")
        for ax in axes.ravel()[len(ps):]:
            ax.set_visible(False)
        handles = [Line2D([], [], color=COLORS[c], ls=ls, marker=mk, label=label) for _, c, ls, mk, label in metrics]
        handles += regime_handles() + [Line2D([], [], ls="", marker="v", mfc="white", mec="black", label="exact zero at display floor")]
        fig.legend(handles=handles, loc="lower center", ncol=3)
        scope = "Mesh-only empirical errors; correction and bounds retain complete-graph normalization." if mesh else "Complete empirical operator errors against the independent numerical reference."
        fig.text(.5, .13, scope + "\nMedians/min–max over 9 configurations (3 families × 3 seeds). Top squares: projection activity.\n"
                 "E_old is measured, not a certified old-error bound. No mixed-transition convergence order is inferred.", ha="center", fontsize=7)
        fig.tight_layout(rect=(0, .23, 1, 1), h_pad=1.3)
        save(fig, name, output, args)


def tightness_figure(rows, args):
    name = "fig2_tightness_delta_negative_blocks"
    conv = unique(subset(rows, "convergence"))
    ps = [p for p in (2, 3, 5, 7, 9) if any(r.get("p") == p for r in conv)]
    if not ps:
        skip(name, "No convergence rows")
        return
    fig, axes = plt.subplots(3, len(ps), figsize=(3*len(ps), 8.3), squeeze=False, sharex="col", sharey="row")
    output = []
    for ci, p in enumerate(ps):
        group = [r for r in conv if r.get("p") == p]
        checked = [r for r in group if verified_norm(r) and number(r.get("correction_norm")) is not None and r["correction_norm"] > 0]
        ax = axes[0, ci]
        for key, color, ls, marker in [("B_simple_tightness", "simple", ":", "D"),
                                       ("B_degree_tightness", "degree", "-.", "x"),
                                       ("B_tightness", "merged", "-", "o")]:
            records = stats(checked, "M", key, 1e-18)
            curve(ax, records, COLORS[color], ls, marker)
            add_stats(output, records, p=p, metric=key, norm_scope="verified correction-only norm; zero excluded")
        ax.axhline(1, color="#777777", ls="--", lw=.8)
        ax.set_yscale("log"); ax.set_title(f"p={p}")
        if not checked:
            blank(ax, "No positive verified correction norm\n(exact zero: ratio undefined)")
        for ri, key, ylabel, color in [(1, "delta", r"$\delta=\max_d\|C_d\|_2$", "projection"),
                                       (2, "negative_percent", "Negative grid blocks (%)", "degree")]:
            floor = floor_for(group, [key])
            records = stats(group, "M", key, floor)
            curve(axes[ri, ci], records, COLORS[color])
            add_stats(output, records, p=p, metric=key)
            axes[ri, ci].set_yscale("log")
            if ci == 0:
                axes[ri, ci].set_ylabel(ylabel)
        axes[0, ci].set_xscale("log"); axes[0, ci].set_ylabel(r"$B_*/\|\Delta\Gamma\|_2$" if ci == 0 else "")
        axes[2, ci].set_xlabel("Mesh size M")
    handles = [Line2D([], [], color=COLORS[c], ls=ls, marker=mk, label=lab) for c, ls, mk, lab in
               [("simple", ":", "D", "N delta"), ("degree", "-.", "x", "2 Dmax"), ("merged", "-", "o", "merged bound")]]
    fig.legend(handles=handles, loc="lower center", ncol=3)
    fig.text(.5, .065, "Tightness uses only independently verified positive correction-only spectral norms; exact-zero ratios are undefined.\n"
             "Dotted reference at 1 is the real-arithmetic requirement. Small floating violations are retained; no clipping to 1.\n"
             "Ratios with a correction graph norm below the full-operator floating scale can be extremely large; all are retained.\n"
             "Bands: min–max over configurations. Open downward triangles: actual zeros shown at a recorded plotting floor.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .15, 1, 1), w_pad=1.1)
    save(fig, name, output, args)


def controlled_slice(rows, parameter, setting="baseline"):
    if parameter == "xi":
        return unique(subset(rows, "xi_sweep")), "controlled xi_sweep preset"
    source = subset(rows, "sweeps")
    explicit = [r for r in source if str(r.get("sweep", r.get("source_sweep", ""))) == parameter]
    if explicit:
        chosen = [r for r in explicit if str(r.get("setting", r.get("source_setting", ""))) == setting]
        if setting != "baseline" and not chosen:
            return [], f"No recorded {setting} controlled {parameter} sweep"
        return unique(chosen or explicit), f"explicit sweep; {setting} setting"
    source = unique(source)
    fixed = [key for key in PARAMETERS if key != parameter]
    groups = defaultdict(list)
    for row in source:
        if number(row.get(parameter)) is not None:
            groups[tuple(row.get(k) for k in fixed)].append(row)
    eligible = [v for v in groups.values() if len({r.get(parameter) for r in v}) > 1]
    if not eligible:
        return [], "No identifiable controlled slice"
    return max(eligible, key=lambda group: (len({r.get(parameter) for r in group}), len(group))), "largest slice with all other numeric parameters fixed"


def parameter_figure(rows, args):
    name = "fig3_parameter_bounds_activity"
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
    output = []
    specs = [("xi", r"Splitting parameter $\xi$"), ("sF", r"Pre-projection cutoff request $s_F$"),
             ("p", "Assignment order p"), ("rho", r"Density $\rho$"), ("kappa", r"Screening $\kappa$")]
    for ax, (parameter, xlabel) in zip(axes.ravel(), specs):
        group, rule = controlled_slice(rows, parameter)
        if not group:
            blank(ax, "Controlled source slice unavailable")
            STATUS["notes"].append(f"{name}: {parameter}: {rule}")
        metrics = [CORE[i] for i in (0, 1, 2, 5)]
        floor = floor_for(group, [key for key, *_ in metrics])
        for key, color, ls, marker, label in metrics:
            records = stats(group, parameter, key, floor)
            curve(ax, records, COLORS[color], ls, marker, label)
            add_stats(output, records, parameter=parameter, metric=key, slice_rule=rule, error_scope="complete")
        regime_strip(ax, group, parameter, output, parameter=parameter)
        ax.set_yscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel("Relative error / correction bound")
        if parameter == "p":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(parameter)
    axes[1, 2].axis("off")
    handles = [Line2D([], [], color=COLORS[c], ls=ls, marker=mk, label=label) for _, c, ls, mk, label in [CORE[i] for i in (0,1,2,5)]]
    axes[1, 2].legend(handles=handles + regime_handles(), loc="center", fontsize=8)
    fig.text(.5, .035, "Panels use their own controlled presets; xi is the separate shared-parameter scan. Medians and min–max across configurations.\n"
             "Top squares encode activity. No cross-panel speed claim or certified total-error bound is made.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .08, 1, 1))
    if output:
        save(fig, name, output, args)
    else:
        plt.close(fig); skip(name, "No controlled parameter slices")


def moderate_parameter_figure(rows, args):
    name="fig3b_parameter_bounds_activity_moderate"
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.6))
    output=[]
    specs=[("M","Mesh size M"),("p","Assignment order p"),("rho",r"Density $\rho$"),("kappa",r"Screening $\kappa$")]
    for ax,(parameter,xlabel) in zip(axes.ravel(),specs):
        group,rule=controlled_slice(rows,parameter,"moderate")
        if not group:blank(ax,"Recorded moderate slice unavailable")
        metrics=[CORE[i] for i in (0,1,2,5)]
        floor=floor_for(group,[key for key,*_ in metrics])
        for key,color,ls,marker,label in metrics:
            records=stats(group,parameter,key,floor)
            curve(ax,records,COLORS[color],ls,marker,label)
            add_stats(output,records,parameter=parameter,metric=key,slice_rule=rule,setting="moderate",error_scope="complete")
        regime_strip(ax,group,parameter,output,parameter=parameter,setting="moderate")
        ax.set_yscale("log");ax.set_xlabel(xlabel);ax.set_ylabel("Relative complete error / correction bound")
        ax.set_title(f"moderate: {parameter}")
        if parameter=="M":ax.set_xscale("log")
        if parameter=="p":ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    handles=[Line2D([],[],color=COLORS[c],ls=ls,marker=mk,label=label) for _,c,ls,mk,label in [CORE[i] for i in (0,1,2,5)]]
    fig.legend(handles=handles+regime_handles(),loc="lower center",ncol=3)
    fig.text(.5,.12,"Recorded moderate preset only: xi=.85, sR=sF=4, M=40, p=7, rho=.75, kappa=.7 except the scanned parameter.\n"
             "The historical study has no controlled moderate sF sweep; its baseline sF data remain in Fig. 3. No missing scan is invented.\n"
             "Top squares: activity. Curves and bands: medians/min–max over configurations; no transition order is inferred.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.23,1,1))
    if output:save(fig,name,output,args)
    else:plt.close(fig);skip(name,"No recorded moderate parameter slices")


def selection_label(row):
    for key in ("selected_label", "case_label", "label", "role", "selection"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return str(row.get("parameter_id") or row.get("case_id") or "unlabelled")


def selected_cases(rows, summaries):
    output = defaultdict(list)
    for summary in summaries:
        label = selection_label(summary)
        matched = [r for r in rows if summary.get("case_id") and r.get("case_id") == summary["case_id"]]
        if not matched and summary.get("parameter_id"):
            matched = [r for r in rows if r.get("parameter_id") == summary["parameter_id"]]
        if not matched and all(number(summary.get(k)) is not None for k in ("xi", "sR", "sF", "M", "p", "kappa", "rho")):
            matched = [r for r in rows if all(number(r.get(k)) == number(summary[k]) for k in ("xi", "sR", "sF", "M", "p", "kappa", "rho"))]
        output[label].extend(matched or [summary])
    return {label: unique_cases(group) for label, group in output.items()}


def case_figure(rows, summaries, timing, args):
    name = "fig4_winner_activecase_and_bound_cost"
    groups = selected_cases(rows, summaries)
    if not groups:
        skip(name, "No labelled cases_summary.csv rows")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.5))
    output = []
    labels = list(groups)
    display = [label.replace("_", " ") for label in labels]
    metrics = [("E_old", "old", "Empirical E_old"), ("E_new", "new", "Empirical E_new"),
               ("E_proj", "projection", "E_proj comparison"), ("B_rel", "merged", "Merged correction bound")]
    floor = floor_for([r for group in groups.values() for r in group], [key for key, *_ in metrics])
    width = .8 / len(metrics)
    for ci, (key, color, label) in enumerate(metrics):
        for gi, (case, group) in enumerate(groups.items()):
            values = [r[key] for r in group if number(r.get(key)) is not None]
            if not values:
                continue
            median, lo, hi = float(np.median(values)), min(values), max(values)
            x = gi + (ci-(len(metrics)-1)/2)*width
            if median > 0:
                axes[0].bar(x, median, width=width, color=COLORS[color], label=label if gi == 0 else None)
                axes[0].errorbar(x, median, yerr=[[median-max(lo,floor)], [hi-median]], fmt="none", color="#444444", lw=.7, capsize=2)
            else:
                axes[0].scatter([x], [floor], marker="v", s=30, facecolors="white", edgecolors=COLORS[color])
            output.append(dict(record_type="case_error", selected_label=case, metric=key, x=x, median=median, min=lo, max=hi,
                               n=len(values), zero_display_floor=floor, case_ids=";".join(str(r.get("case_id", "")) for r in group)))
    axes[0].set_yscale("log"); axes[0].set_ylabel("Relative complete error / correction bound")
    axes[0].set_title("(a) Winner and active stress case")
    axes[0].legend(fontsize=6)
    phases = sorted({str(r.get("phase")) for r in timing if number(r.get("seconds")) is not None})
    for pi, phase in enumerate(phases):
        color = plt.get_cmap("tab10")(pi % 10)
        for gi, case in enumerate(labels):
            group = groups[case]
            caseids = {str(r.get("case_id", "")) for r in group}
            parameterids = {str(r.get("parameter_id", "")) for r in group}
            samples = [r for r in timing if str(r.get("phase")) == phase and number(r.get("seconds")) is not None
                and (selection_label(r) == case or (r.get("case_id") and str(r["case_id"]) in caseids)
                     or (r.get("parameter_id") and str(r["parameter_id"]) in parameterids))]
            if not samples:
                continue
            samples = independent_timing_samples(samples)
            values = [r["seconds"]*1e3 for r in samples]
            x = gi + (pi-(len(phases)-1)/2)*.8/max(len(phases),1)
            median = float(np.median(values))
            axes[1].bar(x, median, width=.8/max(len(phases),1), color=color)
            q25,q75=float(np.quantile(values,.25)),float(np.quantile(values,.75))
            axes[1].errorbar(x,median,yerr=[[median-q25],[q75-median]],fmt="none",ecolor="#333333",lw=.7,capsize=2)
            output.append(dict(record_type="new_bound_setup", selected_label=case, phase=phase, median_ms=median,
                               q25_ms=q25,q75_ms=q75,min_ms=min(values), max_ms=max(values), n=len(values)))
    if phases:
        axes[1].legend(handles=[plt.Rectangle((0,0),1,1,color=plt.get_cmap("tab10")(i%10),label=phase) for i,phase in enumerate(phases)], fontsize=6)
    else:
        blank(axes[1], "New bound setup timing not saved")
    axes[1].set_ylabel("New, independently measured setup cost (ms)")
    axes[1].set_title("(b) Bound construction / evaluation only")
    for ax in axes:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(display, fontsize=7)
    fig.text(.5, .02, "Error ranges span saved configurations. Bound setup phases are independently measured and may overlap; do not add their medians.\n"
             "Timing error bars: q25–q75; configuration samples aggregate 9 configs × repeats. Shared parameter samples count once per repeat.\n"
             "No old timing batch is used to claim a speedup.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .09, 1, 1))
    save(fig, name, output, args)


def budget_figure(rows, summaries, args):
    name = "fig5_empirical_budget_envelope"
    groups = selected_cases(rows, summaries)
    if not groups:
        skip(name, "No labelled selected cases")
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5.4*len(groups),4.6), squeeze=False)
    output = []
    family_order = {"uniform":0,"moderate":1,"strong":2}
    for ax, (label, group) in zip(axes[0], groups.items()):
        group = sorted(group,key=lambda r:(family_order.get(str(r.get("family")),99),number(r.get("seed")) or 0))
        floor = floor_for(group,["E_old","E_new","empirical_envelope"])
        for field,color,ls,marker,legend in [("E_old","old","--","o","E_old measured"),
                ("E_new","new","-","s","E_new measured"),
                ("empirical_envelope","merged","-.","^","E_old measured + B_rel (envelope)")]:
            prepared = [dict(r,plot_x=i) for i,r in enumerate(group)]
            records=stats(prepared,"plot_x",field,floor)
            curve(ax,records,COLORS[color],ls,marker,legend)
            add_stats(output,records,selected_label=label,metric=field,scope="empirical reference; not certified total")
        ax.axhline(1e-4,color="#777777",ls=":",lw=1,label="empirical budget 1e-4")
        labels=[f"{str(r.get('family','?'))[:1].upper()}{number(r.get('seed')):g}" if number(r.get('seed')) is not None
                else str(r.get("config","?")) for r in group]
        ax.set_xticks(range(len(group)));ax.set_xticklabels(labels)
        ax.set_yscale("log");ax.set_ylabel("Relative empirical error / triangle envelope")
        ax.set_xlabel("Configuration (family initial + seed)");ax.set_title(label.replace("_"," "))
        ax.legend(fontsize=6.3)
    fig.text(.5,.02,"The envelope is relative to the numerical direct reference. No independently certified old-error bound is available.\n"
             "Envelope failure does not imply measured E_new failure; no total-error certification is claimed.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.09,1,1))
    save(fig,name,output,args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--output", type=Path, default=HERE/"figures")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    style()
    rows, timing, cases = (read(args.results/f"{name}.csv") for name in ("bounds", "timing_raw", "cases_summary"))
    jobs = [("core", lambda: core_figures(rows,args)), ("tightness",lambda:tightness_figure(rows,args)),
            ("parameters",lambda:parameter_figure(rows,args)), ("cases",lambda:case_figure(rows,cases,timing,args))]
    jobs.append(("empirical_budget",lambda:budget_figure(rows,cases,args)))
    jobs.append(("moderate_parameters",lambda:moderate_parameter_figure(rows,args)))
    for name, job in jobs:
        try:
            job()
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            plt.close("all"); skip(name, f"{type(exc).__name__}: {exc}")
    STATUS["input_rows"] = len(rows)
    STATUS["dataset_rows"] = dict(Counter(str(r.get("dataset")) for r in rows))
    STATUS["verified_positive_correction_norm_rows"] = sum(verified_norm(r) and (number(r.get("correction_norm")) or 0)>0 for r in rows)
    (args.output/"plot_status.json").write_text(json.dumps(STATUS,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(STATUS,ensure_ascii=False))


if __name__ == "__main__":
    main()
