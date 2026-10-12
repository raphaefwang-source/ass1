#!/usr/bin/env python3
"""Build the new manuscript figures from cached JSON results only (no trajectories are read or rerun).

Inputs (relative to the repository root): ewald_longitudinal/lj075_results/{verify_*.json, dt_study.json, longD.json}.
Outputs: manuscript/figures/fig_operator_lanczos.pdf, fig_dt_forest.pdf, fig_diffusion.pdf.
Usage: python3 manuscript/scripts/make_figures.py   (from the repository root)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "ewald_longitudinal" / "lj075_results"
OUT = ROOT / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"0.0025": "#1baf7a", "0.005": "#2a78d6", "0.01": "#eb6834"}
SETS = [("verify_representative_A.json", "representative (42)"),
        ("verify_visited_eq_A.json", "visited, equilibration (72)"),
        ("verify_visited_dt_A.json", "visited, dt/long-D (38)")]


def load(name):
    return json.loads((RES / name).read_text())


# ---------------------------------------------------------------- Figure: operator + Lanczos
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
for x0, (f, lab) in enumerate(SETS):
    ps = load(f)["per_state"].values()
    e_full = np.array([s["op_err"] for s in ps])
    e_k = np.array([s["op_err_vs_k_nonzero_part"] for s in ps])
    jit = np.linspace(-0.15, 0.15, len(e_full))
    ax[0].plot(x0 + jit - 0.12, e_full, ".", ms=2.5, color="#2a78d6", label="full norm" if x0 == 0 else None)
    ax[0].plot(x0 + jit + 0.12, e_k, ".", ms=2.5, color="#eb6834", label=r"vs $k\neq0$ part" if x0 == 0 else None)
ax[0].axhline(2e-7, color="k", lw=0.8, ls="--", label="budget $2\\times10^{-7}$")
ax[0].set_yscale("log"); ax[0].set_ylim(1e-8, 5e-7)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["repr.\n(42)", "visited eq.\n(72)", "visited dt\n(38)"])
ax[0].set_ylabel(r"$\|\Gamma_h-\Gamma_{\rm ref}\|_2/\|\cdot\|_2$"); ax[0].set_title("(a) PPPM operator error")
ax[0].legend(frameon=False, loc="upper left")

allstates = []
for f, _ in SETS:
    allstates += list(load(f)["per_state"].values())
for k, (fun, dt, ls) in enumerate([("noise", "0.005", "-"), ("noise", "0.01", "--"),
                                    ("damp", "0.005", "-"), ("damp", "0.01", "--")]):
    key = f"{fun}_{dt}"
    ranks = sorted(int(r) for r in allstates[0]["krylov"][key])
    strict = [max(max(s["krylov"][key][str(r)].values()) for s in allstates) for r in ranks]
    ax[1].semilogy(ranks, strict, ls, color="#2a78d6" if fun == "noise" else "#1baf7a", marker=".", ms=3,
                   label=f"{fun}, dt={dt}")
ax[1].axhline(2e-8, color="k", lw=0.8, ls="--", label="budget/10")
ax[1].set_xlabel("Lanczos rank"); ax[1].set_ylabel("max strict error over 152 states")
ax[1].set_title("(b) Lanczos matrix-function error"); ax[1].legend(frameon=False); ax[1].set_xlim(1, 24)

lam = [(s["lam_min"], s["lam_max"]) for s in allstates]
lo, hi = np.array(lam).T
ax[2].hist(lo, bins=25, color="#2a78d6", alpha=0.8, label=r"$\lambda_{\min}$ on Range$(\Pi)$")
ax[2].hist(hi, bins=25, color="#eb6834", alpha=0.8, label=r"$\lambda_{\max}$")
ax[2].axvline(0.75 * 4 * np.pi * 0.5 * 1.3 * np.exp(0.91) / 0.49 / 3, color="k", lw=0.8, ls=":",
              label=r"$k=0$ shift $\rho\hat g(0)/3$")
ax[2].set_xlabel(r"eigenvalue of $\Gamma_h$"); ax[2].set_ylabel("states"); ax[2].set_title("(c) spectrum of $\\Gamma_h$")
ax[2].legend(frameon=False, fontsize=6)
fig.savefig(OUT / "fig_operator_lanczos.pdf")
print("operator: max eps_op", max(s["op_err"] for s in allstates), "lam_min", lo.min(), "lam_max", hi.max())

# ---------------------------------------------------------------- Figure: dt forest plot
g = load("dt_study.json")["A"]["levels_0.00125_0.0025_0.005_0.01_T10"]
comp = g["comparisons"]
rows = [("T_kin", r"$T_{\rm kin}$"), ("U_per_N", "$U/N$")] + [(f"msd{t}", f"MSD({t})") for t in (0.1, 0.5, 1, 2, 5)]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True, gridspec_kw={"width_ratios": [3, 1.2]})
for i, (key, lab) in enumerate(rows):
    for j, dt in enumerate(["0.0025", "0.005", "0.01"]):
        c = comp[f"{dt}_vs_0.00125"]
        if key.startswith("msd"):
            t = float(key[3:])
            rec = [m for m in c["msd"] if abs(m["t"] - t) < 1e-9][0]
        else:
            rec = c[key]
        y = -i + (1 - j) * 0.22
        m, (a, b) = 100 * rec["rel_diff"], [100 * v for v in rec["ci95"]]
        ax[0].errorbar(m, y, xerr=[[m - a], [b - m]], fmt="o", ms=3, color=COL[dt], capsize=2,
                       label=f"dt={dt}" if i == 0 else None)
ax[0].axvspan(-1, 1, color="0.92", zorder=0); ax[0].axvline(0, color="k", lw=0.5)
ax[0].set_yticks([-i for i in range(len(rows))]); ax[0].set_yticklabels([r[1] for r in rows])
ax[0].set_xlabel("relative difference vs dt=0.00125 (%)  [95% t-CI, n=16]")
ax[0].set_title("(a) pre-registered scalar criteria (band: tolerance 1%)"); ax[0].legend(frameon=False, loc="lower right")
for j, dt in enumerate(["0.0025", "0.005", "0.01"]):
    rec = comp[f"{dt}_vs_0.00125"]["pressure"]
    m, (a, b) = rec["abs_diff"], rec["ci95"]
    ax[1].errorbar(m, -j, xerr=[[m - a], [b - m]], fmt="o", ms=3, color=COL[dt], capsize=2)
ax[1].axvspan(-0.0075, 0.0075, color="0.92", zorder=0); ax[1].axvline(0, color="k", lw=0.5)
ax[1].set_yticks([0, -1, -2]); ax[1].set_yticklabels(["0.0025", "0.005", "0.01"])
ax[1].set_xlabel("pressure difference (abs.)"); ax[1].set_title("(b) pressure (tol. 0.0075)")
fig.savefig(OUT / "fig_dt_forest.pdf")

# ---------------------------------------------------------------- Figure: diffusion windows
d = load("longD.json")
A = d["A"]
wins = d["windows"]
mid = [np.sqrt(float(w.split("-")[0]) * float(w.split("-")[1])) for w in wins]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), constrained_layout=True)
for k, dt in enumerate(["0.005", "0.01"]):
    D = A["per_level"][dt]["D_MSD"]
    m = np.array([D[w][0] for w in wins]); lo_, hi_ = np.array([D[w][1] for w in wins]), np.array([D[w][2] for w in wins])
    x = np.array(mid) * (1 + 0.04 * (2 * k - 1))
    ax[0].errorbar(x, m, yerr=[m - lo_, hi_ - m], fmt="o-", ms=3, color=COL[dt], capsize=2, label=f"dt={dt}")
ax[0].set_xscale("log"); ax[0].set_xticks(mid); ax[0].set_xticklabels([f"[{w}]" for w in wins], fontsize=6)
ax[0].set_xlabel("MSD fit window"); ax[0].set_ylabel(r"$D_{\rm MSD}$ [95% CI, n=11]")
ax[0].set_title("(a) windowed diffusion coefficient"); ax[0].legend(frameon=False)
rd = A["dt_comparison"]["0.01_vs_0.005"]["rel_diff"]
m = np.array([100 * rd[w][0] for w in wins]); lo_, hi_ = np.array([100 * rd[w][1] for w in wins]), np.array([100 * rd[w][2] for w in wins])
ax[1].errorbar(mid, m, yerr=[m - lo_, hi_ - m], fmt="o", ms=3, color="k", capsize=2)
ax[1].axhspan(-2, 2, color="0.92", zorder=0); ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xscale("log"); ax[1].set_xticks(mid); ax[1].set_xticklabels([f"[{w}]" for w in wins], fontsize=6)
ax[1].set_ylabel(r"$D(0.01)/D(0.005)-1$ (%)"); ax[1].set_xlabel("MSD fit window")
ax[1].set_title("(b) paired dt effect (band: ±2%)")
fig.savefig(OUT / "fig_diffusion.pdf")
print("done")
