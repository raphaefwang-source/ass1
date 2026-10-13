"""
Tables and figures of the full-step timing (sec6_timing.py report). Only measured points are shown and fitted; the
finite-range fit describes the measured range only.
"""
import csv
import json

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sec6_config  # noqa: E402
from sec6_timing import OUT  # noqa: E402

COMP = [("setup", ("setup_pairs", "setup_weights", "setup_stencil", "setup_degree"), "#86b6ef"),
        ("Gamma_h real-space", ("action_real",), "#2a78d6"), ("Gamma_h PPPM", ("action_mesh",), "#0d366b"),
        ("reorthogonalisation", ("damping_reorth", "noise_reorth"), "#eb6834"),
        ("tridiag + f, combine, vectors", ("damping_tridiag_feval", "noise_tridiag_feval", "damping_combine",
                                           "noise_combine", "damping_vector", "noise_vector"), "#1baf7a"),
        ("updates", ("q_half", "update"), "#52514e")]


def main():
    raw = json.loads((OUT / "timing_raw.json").read_text())
    rows = raw["rows"]
    out_rows = []
    for r in sorted(rows, key=lambda r: (r["method"], r["N"])):
        c = r["components_median"]
        row = dict(method=r["method"], N=r["N"], L=round(r["L"], 4), M=r["box"]["M"] if r["method"] == "pppm" else None,
                   eta_actual=r["box"]["eta_actual"] if r["method"] == "pppm" else None, R_d=r["R_d"],
                   noise_rank=r["noise_rank"], damping_rank=r["damping_rank"], gamma_calls=r["n_actions"][0],
                   gamma_calls_all_equal=len(set(r["n_actions"])) == 1, pairs=r["n_pairs"],
                   step_median_s=r["total_median"], step_min_s=r["total_min"], step_max_s=r["total_max"],
                   spread_rel=r["spread_rel"], T_over_NlogN=r["per_N_logN"], preprocess_s=r["t_preprocess"],
                   ritz_min=r["ritz_min"], ritz_max=r["ritz_max"], cpu=r["machine"]["cpu"],
                   threads=r["machine"]["threads"])
        for lab, keys, _ in COMP:
            row[lab] = sum(c.get(k, 0.0) for k in keys)
        row["unaccounted"] = c["total"] - c["accounted"]
        out_rows.append(row)
    with open(OUT / "timing_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    fits = {}
    for m in sorted({r["method"] for r in out_rows}):
        rr = [r for r in out_rows if r["method"] == m]
        N = np.array([r["N"] for r in rr], float)
        T = np.array([r["step_median_s"] for r in rr])
        if len(rr) >= 3:
            a, b = np.polyfit(np.log(N), np.log(T), 1)
            fits[m] = dict(exponent=float(a), N_range=[int(N.min()), int(N.max())], n_points=len(rr),
                           note="least-squares fit of log T on log N over the measured points only; describes this "
                                "range, not an asymptotic complexity")
            if len(rr) >= 4:
                big = N >= 2048
                if big.sum() >= 3:
                    a2, _ = np.polyfit(np.log(N[big]), np.log(T[big]), 1)
                    fits[m]["exponent_N_ge_2048"] = float(a2)
    summ = dict(provenance=sec6_config.provenance("sec6_timing.py", True), fits=fits, rows=out_rows,
                machine=rows[0]["machine"])
    dc = OUT / "dense_cost.json"
    if dc.exists():
        summ["dense_cost"] = json.loads(dc.read_text())["rows"]
    acc = OUT / "direct_accuracy.json"
    if acc.exists():
        a = json.loads(acc.read_text())
        summ["direct_accuracy"] = dict(R_d=a["R_d"], pppm_max=a["production_pppm_err_max_same_configs"],
                                       direct_max_at_R_d=max(r["err_rel"] for r in a["rows"] if r["R_d"] == a["R_d"])
                                       if a["R_d"] else None)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    # figure
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "legend.frameon": False, "savefig.dpi": 200})
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    for m, col, mk in (("pppm", "#2a78d6", "o"), ("direct", "#eb6834", "s")):
        rr = [r for r in out_rows if r["method"] == m]
        if not rr:
            continue
        N = [r["N"] for r in rr]
        axs[0].errorbar(N, [r["step_median_s"] for r in rr],
                        yerr=[[r["step_median_s"] - r["step_min_s"] for r in rr], [r["step_max_s"] - r["step_median_s"] for r in rr]],
                        fmt=mk + "-", color=col, ms=3.5, capsize=2, label=f"{m} (measured)")
        axs[1].plot(N, [r["T_over_NlogN"] * 1e6 for r in rr], mk + "-", color=col, ms=3.5, label=m)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("$N$")
    axs[0].set_ylabel("time per step [s] (median, min-max)")
    axs[0].legend(loc="upper left", fontsize=6.5)
    axs[0].set_title("full step, 1 thread", loc="left")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("$N$")
    axs[1].set_ylabel(r"$T/(N\ln N)$ [$\mu$s]")
    axs[1].set_title(r"$T/(N\ln N)$", loc="left")
    ax = axs[2]
    rr = [r for r in out_rows if r["method"] == "pppm"]
    x = np.arange(len(rr))
    bottom = np.zeros(len(rr))
    for lab, keys, col in COMP:
        v = np.array([r[lab] / r["step_median_s"] for r in rr])
        ax.bar(x, v, bottom=bottom, color=col, width=0.75, label=lab, edgecolor="white", linewidth=0.5)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["N"]) for r in rr], rotation=60, fontsize=6)
    ax.set_ylabel("fraction of step (pppm)")
    ax.set_title("cost breakdown (median step)", loc="left")
    ax.legend(loc="lower left", fontsize=5.4)
    fig.savefig(OUT / "fig_timing.png")
    print(json.dumps(dict(fits=fits, rows=[{k: r[k] for k in ("method", "N", "step_median_s", "spread_rel",
                                                               "gamma_calls", "T_over_NlogN")} for r in out_rows]),
                     indent=1))
