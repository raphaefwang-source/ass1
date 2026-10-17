#!/usr/bin/env python3
"""
Figures 1-6, parameter_results.csv, timing_breakdown.csv and the error-contribution table, from the result files.

    python3 make_figures.py
Colours: categorical slots 1-4 of the reference palette in fixed order (uniform, moderate, strong, LJ-equilibrated);
grey for the baseline and the target; markers and labels as secondary encoding.
"""
import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common as C  # noqa: E402
from clustered_configurations import load  # noqa: E402

FAM_COL = {"uniform": "#2a78d6", "moderate": "#eb6834", "strong": "#1baf7a", "lj_equilibrated": "#eda100"}
FAM_MK = {"uniform": "o", "moderate": "s", "strong": "D", "lj_equilibrated": "^"}
FAM_LAB = {"uniform": "uniform", "moderate": "moderately clustered", "strong": "strongly clustered",
           "lj_equilibrated": "LJ equilibrated (t = 50)"}
INK, INK2, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#7d7c78"


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
        "lines.markersize": 5.5, "text.color": INK, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb"})


def rd(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- data
def load_all():
    D = dict(refs=C.read_json(C.RESULTS / "references.json")["configs"],
             summ=C.read_json(C.RESULTS / "selection_summary.json")["groups"],
             fam=rd(C.RESULTS / "selection_family.csv"), grid=rd(C.RESULTS / "selection_grid.csv"),
             bench=C.read_json(C.RESULTS / "bench.json")["rows"], errors=C.read_json(C.RESULTS / "errors.json"),
             emp=C.read_json(C.RESULTS / "empirical.json", {"family": [], "check": [], "timing": [], "seeds": []}),
             cfgs={r["name"]: r for r in rd(C.RESULTS / "configurations.csv")},
             cost=C.read_json(C.RESULTS / "cost_model.json"))
    return D


def fam_curve(D, fam, N, key, grid="coarse"):
    rows = sorted([r for r in D["fam"] if r["family"] == fam and int(r["N"]) == N and r["grid"] == grid],
                  key=lambda r: float(r["xi"]))
    return np.array([float(r["xi"]) for r in rows]), np.array([fnum(r[key]) for r in rows])


def err_of(D, config, st, xi):
    for r in D["errors"]["rows"]:
        if r["config"] == config and r["set"] == st and abs(r["xi"] - xi) < 1e-9:
            return r
    return None


# ----------------------------------------------------------------------------- figure 1
def fig1(D):
    names = ["uniform_N256_s1", "moderate_N256_s1", "strong_N256_s1", "lj_equilibrated_N256_s305"]
    fig = plt.figure(figsize=(12.5, 6.4))
    for k, nm in enumerate(names):
        ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        x, L = load(nm)
        fam = nm.split("_N")[0]
        x = np.mod(x, L)
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=9, c=FAM_COL[fam], depthshade=True, edgecolors="none")
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([0, L]); ax.set_yticks([0, L]); ax.set_zticks([0, L])
        ax.set_xticklabels(["0", "L"]); ax.set_yticklabels(["0", "L"]); ax.set_zticklabels(["0", "L"])
        c = D["cfgs"][nm]
        ax.set_title(f"{FAM_LAB[fam]}\nN = 256, L = {L:.3f}, d_min = {float(c['d_min_periodic']):.3f}", fontsize=8.5)
    ax = fig.add_subplot(2, 3, 4)
    for nm in names:
        fam = nm.split("_N")[0]
        with np.load(C.CONFIGS / f"{nm}.npz") as d:
            e, g = d["hist_edges"], d["g"]
        mid = 0.5 * (e[1:] + e[:-1])
        ax.plot(mid, g, color=FAM_COL[fam], label=FAM_LAB[fam], lw=1.3)
    ax.axvline(C.D_MIN, color=GREY, ls=":", lw=1)
    ax.set_xlim(0, 7)
    ax.set_xlabel("pair distance r (all periodic images)")
    ax.set_ylabel("g(r)")
    ax.set_title("(e) pair-distance histogram, normalised", fontsize=9)
    ax.legend(fontsize=7.5)
    ax = fig.add_subplot(2, 3, 5)
    bins = np.arange(-0.5, 25.5, 1)
    for nm in names:
        fam = nm.split("_N")[0]
        with np.load(C.CONFIGS / f"{nm}.npz") as d:
            nb = d["nb_first"]
        ax.hist(nb, bins=bins, histtype="step", color=FAM_COL[fam], lw=1.6, label=FAM_LAB[fam])
    ax.set_xlabel("neighbours within r = 1.5")
    ax.set_ylabel("particles")
    ax.set_title("(f) local neighbour-count distribution", fontsize=9)
    ax = fig.add_subplot(2, 3, 6)
    fams = ["uniform", "moderate", "strong"]
    for fam in fams:
        Ns, S = [], []
        for N in (64, 256, 512):
            v = [float(r["S_k_max_small"]) for k, r in D["cfgs"].items() if r["family"] == fam and int(r["N"]) == N]
            Ns.append(N)
            S.append(np.mean(v))
        ax.plot(Ns, S, marker=FAM_MK[fam], color=FAM_COL[fam], label=FAM_LAB[fam])
    ax.plot([256], [float(D["cfgs"]["lj_equilibrated_N256_s305"]["S_k_max_small"])], marker="^", ls="none",
            color=FAM_COL["lj_equilibrated"], label=FAM_LAB["lj_equilibrated"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([64, 256, 512]); ax.set_xticklabels(["64", "256", "512"])
    ax.set_xlabel("N")
    ax.set_ylabel("max S(k), 0 < |k| <= 4 pi/L")
    ax.set_title("(g) long-wavelength density fluctuations", fontsize=9)
    ax.legend(fontsize=7.5)
    fig.suptitle("Figure 1. Configurations at rho = 0.75 with periodic minimum separation 0.9 (N = 256 shown)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig1_configurations")


# ----------------------------------------------------------------------------- figure 2
def fig2(D):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, nm in zip(axes[:2], ("strong_N256_s1", "uniform_N256_s1")):
        fam = nm.split("_N")[0]
        xs, tp = fam_curve(D, fam, 256, "T_pred")
        ax.plot(xs, 1e3 * tp, color=FAM_COL[fam], lw=1.6, label="predicted (cost model, certified parameters)")
        b = sorted([r for r in D["bench"] if r["config"] == nm and r["set"] == "certified"], key=lambda r: r["xi"])
        bx = np.array([r["xi"] for r in b])
        bt = np.array([r["t_action_s"] for r in b]) * 1e3
        lo = bt - np.array([r["t_action_q1_s"] for r in b]) * 1e3
        hi = np.array([r["t_action_q3_s"] for r in b]) * 1e3 - bt
        ax.errorbar(bx, bt, yerr=[lo, hi], fmt=FAM_MK[fam], color=INK, mfc="white", ms=5.5, capsize=2, lw=1,
                    label="measured median (IQR), cached action")
        xo = D["summ"][f"{fam}_N256"]["opt"]["T_pred"]["xi"]
        ax.axvline(xo, color=FAM_COL[fam], ls="--", lw=1)
        best = b[int(np.argmin(bt))]
        ax.text(0.97, 0.05, f"predicted xi_opt = {xo:.3f}\nmeasured fastest = {best['xi']:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=INK2)
        bl = [r for r in D["bench"] if r["config"] == nm and r["set"] == "baseline"]
        if bl:
            ax.axhline(1e3 * bl[0]["t_action_s"], color=GREY, ls=":", lw=1.2,
                       label=f"validated baseline (xi 0.85, p 8, M {bl[0]['M']}), measured")
        ax.set_yscale("log")
        ax.set_xlabel("Ewald parameter xi")
        ax.set_ylabel("T_action [ms]")
        ax.set_title(f"({'ab'[list(axes).index(ax)]}) {FAM_LAB[fam]}, {nm}", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
    ax = axes[2]
    for key, lab, col in (("T_pred_real", "real space C_R N_img", "#2a78d6"),
                          ("T_pred_spread_gather", "spread + gather C_P N p^3", "#eb6834"),
                          ("T_pred_fft", "FFT + influence C_F M^3 log M^3", "#1baf7a"),
                          ("T_pred_overhead", "constant C_0", "#eda100")):
        xs, v = fam_curve(D, "strong", 256, key)
        ax.plot(xs, 1e3 * v, color=col, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Ewald parameter xi")
    ax.set_ylabel("predicted component [ms]")
    ax.set_title("(c) predicted breakdown, strongly clustered N = 256", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("Figure 2. Predicted vs measured cached action time at certified equal accuracy (eps = 2e-7)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig2_time_vs_xi")


# ----------------------------------------------------------------------------- figure 3
def fig3(D):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    for fam in ("uniform", "moderate", "strong", "lj_equilibrated"):
        xs, ni = fam_curve(D, fam, 256, "n_img_mean")
        _, nu = fam_curve(D, fam, 256, "n_unique_mean")
        ax.plot(xs, ni, color=FAM_COL[fam], label=f"{FAM_LAB[fam]}: image interactions")
        ax.plot(xs, nu, color=FAM_COL[fam], ls="--", lw=1.1, label=f"{FAM_LAB[fam]}: unique KD-tree pairs")
    xs, rc = fam_curve(D, "uniform", 256, "rc")
    ax.plot(xs, 256 / 2 * C.DENSITY * 4 / 3 * np.pi * rc ** 3, color=GREY, ls=":", lw=1.2,
            label="homogeneous estimate N rho (4 pi/3) rc^3 / 2")
    ax.axhline(256 * 255 / 2, color=MUTED, lw=0.8)
    ax.text(1.58, 256 * 255 / 2 * 1.08, "N(N-1)/2", ha="right", fontsize=7.5, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("Ewald parameter xi  (rc = a_tol(xi)/xi)")
    ax.set_ylabel("retained real-space interactions")
    ax.set_title("(a) N = 256, certified rc(xi)", fontsize=9)
    ax.legend(fontsize=6.5, ncol=1)
    ax = axes[1]
    for fam in ("uniform", "moderate", "strong"):
        for N, ls in ((64, ":"), (256, "-"), (512, "--")):
            xs, ra = fam_curve(D, fam, N, "images_per_pair")
            if len(xs):
                ax.plot(xs, ra, color=FAM_COL[fam], ls=ls, lw=1.3, label=f"{FAM_LAB[fam]}, N = {N}")
    ax.axhline(1, color=GREY, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Ewald parameter xi")
    ax.set_ylabel("image interactions / unique pairs")
    ax.set_title("(b) full-periodic image multiplicity", fontsize=9)
    ax.legend(fontsize=6.5, ncol=1)
    fig.suptitle("Figure 3. Real-space interaction count vs xi (KD-tree candidates + all lattice shifts)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig3_pairs_vs_xi")


# ----------------------------------------------------------------------------- figure 4
def fig4(D):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    for N, ls in ((64, ":"), (256, "-"), (512, "--")):
        xs, M = fam_curve(D, "strong", N, "M")
        ax.plot(xs, M, color=FAM_COL["strong"], ls=ls, drawstyle="steps-mid", label=f"certified M, strong, N = {N}")
    xs, M = fam_curve(D, "uniform", 256, "M")
    ax.plot(xs, M, color=FAM_COL["uniform"], drawstyle="steps-mid", lw=1.1, label="certified M, uniform, N = 256")
    em = [f for f in D["emp"]["family"] if f["N"] == 256]
    for fam in ("uniform", "moderate", "strong"):
        e = sorted([f for f in em if f["family"] == fam], key=lambda f: f["xi"])
        if e:
            ax.plot([f["xi"] for f in e], [f["M"] for f in e], marker=FAM_MK[fam], color=FAM_COL[fam], mfc="white",
                    ls="none", label=f"empirical M (measured budget), {FAM_LAB[fam]}, N = 256")
    bl = [r for r in D["bench"] if r["config"] == "strong_N256_s1" and r["set"] == "baseline"]
    if bl:
        ax.plot([C.BASELINE["xi"]], [bl[0]["M"]], marker="*", ms=10, color=GREY, ls="none",
                label="validated baseline (p = 8), N = 256")
    ax.set_xlabel("Ewald parameter xi")
    ax.set_ylabel("FFT mesh size M (2,3,5-smooth)")
    ax.set_title("(a) mesh size, p = 7", fontsize=9)
    ax.legend(fontsize=6.5)
    ax = axes[1]
    for fam in ("uniform", "strong"):
        xs, kc = fam_curve(D, fam, 256, "kc")
        _, a = fam_curve(D, fam, 256, "a_tol")
        _, b = fam_curve(D, fam, 256, "b_tol")
        ax.plot(xs, a, color=FAM_COL[fam], label=f"a_tol = xi rc, {FAM_LAB[fam]}")
        ax.plot(xs, b, color=FAM_COL[fam], ls="--", drawstyle="steps-mid", label=f"b_tol = kc/(2 xi), {FAM_LAB[fam]}")
    ax.axhline(C.BASELINE["s"], color=GREY, ls=":", lw=1.2, label="baseline s* = 4.1 (a = b)")
    ax.set_xlabel("Ewald parameter xi")
    ax.set_ylabel("dimensionless cutoff")
    ax.set_title("(b) certified a_tol(xi), b_tol(xi), N = 256", fontsize=9)
    ax.legend(fontsize=6.5)
    fig.suptitle("Figure 4. Fourier mesh and cutoffs from the certified bounds (C(7) = (1 + 2 zeta(7))^3 - 1)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig4_mesh_vs_xi")


# ----------------------------------------------------------------------------- figure 5
def fig5(D):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    for ax, N in zip(axes, (256, 64)):
        for fam in ("uniform", "moderate", "strong", "lj_equilibrated"):
            rows = sorted([r for r in D["errors"]["rows"] if r["family"] == fam and r["N"] == N and r["set"] == "certified"],
                          key=lambda r: r["xi"])
            if rows:
                ax.plot([r["xi"] for r in rows], [r["E_op"] for r in rows], marker=FAM_MK[fam], color=FAM_COL[fam],
                        ls="none", label=f"measured E_op, certified set, {FAM_LAB[fam]}")
            cert = sorted([g for g in D["grid"] if g["family"] == fam and int(g["N"]) == N and g["feasible"] == "True"],
                          key=lambda g: float(g["xi"]))
            if cert and fam != "lj_equilibrated":
                by = {}
                for g in cert:
                    v = fnum(g["eps_cert_rel"]) * fnum(g["lam_lb"]) / fnum(g["norm2_ref"])
                    by.setdefault(float(g["xi"]), []).append(v)
                xs = sorted(by)
                ax.plot(xs, [max(by[x_]) for x_ in xs], color=FAM_COL[fam], lw=1, alpha=0.8,
                        label=f"certified bound / ||Gamma_ref||, {FAM_LAB[fam]}")
            ec = sorted([c for c in D["emp"]["check"] if c["family"] == fam and c["N"] == N], key=lambda c: c["xi"])
            if ec:
                ax.plot([c["xi"] for c in ec], [c["E_op"] for c in ec], marker="x", color=FAM_COL[fam], ls="none",
                        ms=6, label=f"measured E_op, empirical set, {FAM_LAB[fam]}")
        bl = [r for r in D["errors"]["rows"] if r["N"] == N and r["set"] == "baseline"]
        if bl:
            ax.plot([C.BASELINE["xi"]] * len(bl), [r["E_op"] for r in bl], marker="*", ms=9, color=GREY, ls="none",
                    label="validated baseline set (p = 8)")
        ax.axhline(C.EPS_TARGET, color=INK2, ls="--", lw=1.1)
        ax.text(0.41, C.EPS_TARGET * 1.12, "target eps = 2e-7", fontsize=7.5, color=INK2)
        ax.set_yscale("log")
        ax.set_xlabel("Ewald parameter xi")
        ax.set_ylabel("||Gamma_h - Gamma_ref||_2 / ||Gamma_ref||_2")
        ax.set_title(f"({'ab'[list(axes).index(ax)]}) N = {N}", fontsize=9)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Figure 5. Measured operator error vs xi against the dense full-periodic reference",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    save(fig, "fig5_error_vs_xi")


# ----------------------------------------------------------------------------- figure 6
def fig6(D):
    fams = ["uniform", "moderate", "strong", "lj_equilibrated"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    for N, mk, off in ((64, "v", -0.18), (256, "o", 0.0), (512, "s", 0.18)):
        xs, ys = [], []
        for k, fam in enumerate(fams):
            g = D["summ"].get(f"{fam}_N{N}")
            if g:
                xs.append(k + off)
                ys.append(g["opt"]["T_pred"]["xi"])
        ax.plot(xs, ys, marker=mk, color=INK2, ls="none", ms=6, label=f"predicted xi_opt (certified), N = {N}")
    xs, ys = [], []
    for k, fam in enumerate(fams):
        nm = f"{fam}_N256_s1" if fam != "lj_equilibrated" else "lj_equilibrated_N256_s305"
        b = [r for r in D["bench"] if r["config"] == nm and r["set"] == "certified"]
        if b:
            xs.append(k + 0.05)
            ys.append(min(b, key=lambda r: r["t_action_s"])["xi"])
    ax.plot(xs, ys, marker="*", ms=10, color="#2a78d6", ls="none", label="measured fastest xi (certified sets), N = 256")
    xs, ys = [], []
    for k, fam in enumerate(fams):
        e = [f for f in D["emp"]["family"] if f["family"] == fam and f["N"] == 256]
        if e:
            xs.append(k - 0.05)
            ys.append(min(e, key=lambda f: f["T_pred"])["xi"])
    ax.plot(xs, ys, marker="P", ms=8, color="#eb6834", ls="none", label="empirical-diagnostic xi_opt, N = 256")
    ax.axhline(C.BASELINE["xi"], color=GREY, ls=":", lw=1.1, label="baseline xi = 0.85")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(["uniform", "moderate", "strong", "LJ eq."])
    ax.set_ylabel("xi_opt")
    ax.set_title("(a) action-only objective", fontsize=9)
    ax.legend(fontsize=6.5)
    ax = axes[1]
    for s_, mk, col in ((None, "o", "#2a78d6"), (1, "s", "#eb6834"), (10, "D", "#1baf7a"), (20, "^", "#eda100")):
        key = "T_pred" if s_ is None else f"J_pred_s{s_}"
        xs, ys = [], []
        for k, fam in enumerate(fams):
            g = D["summ"].get(f"{fam}_N256")
            if g:
                xs.append(k)
                ys.append(g["opt"][key]["xi"])
        ax.plot(xs, ys, marker=mk, color=col, ls="-", lw=1,
                label="T_action only" if s_ is None else f"T_setup + {s_} T_action")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(["uniform", "moderate", "strong", "LJ eq."])
    ax.set_ylabel("predicted xi_opt (certified), N = 256")
    ax.set_title("(b) reuse count s of one configuration", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("Figure 6. Optimal xi vs clustering level", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig6_xi_opt_vs_clustering")


# ----------------------------------------------------------------------------- tables
def tables(D):
    summ = D["summ"]
    rows = []
    for name in sorted({r["config"] for r in D["bench"]}, key=lambda n: (int(n.split("_N")[1].split("_")[0]), n)):
        fam = name.split("_N")[0]
        N = int(name.split("_N")[1].split("_")[0])
        opt = summ[f"{fam}_N{N}"]["opt"]
        b = [r for r in D["bench"] if r["config"] == name and r["set"] == "certified"]
        xo = [r for r in b if abs(r["xi"] - opt["T_pred"]["xi"]) < 1e-9][0]
        fastest = min(b, key=lambda r: r["t_action_s"])
        eo = err_of(D, name, "certified", xo["xi"]) or {}
        bl = [r for r in D["bench"] if r["config"] == name and r["set"] == "baseline"][0]
        eb = err_of(D, name, "baseline", bl["xi"]) or {}
        rows.append(dict(
            config=name, family=fam, N=N, L=xo["L"], xi_opt_pred=opt["T_pred"]["xi"],
            a_tol=xo["xi"] * xo["rc"], b_tol=xo["kc"] / (2 * xo["xi"]), rc=xo["rc"], kc=xo["kc"], M=xo["M"], p=xo["p"],
            eta=xo["eta"], N_unique_pairs=xo["n_unique"], N_image_interactions=xo["n_img"],
            images_per_pair=xo["n_img"] / xo["n_unique"], E_op_measured=eo.get("E_op"),
            E_op_k_nonzero=eo.get("E_op_k0"), E_op_basis_assembly=eo.get("E_op_basis"),
            T_action_pred_ms=1e3 * xo["T_pred_s"], T_action_meas_ms=1e3 * xo["t_action_s"],
            T_action_IQR_rel=xo["t_action_iqr_rel"], pred_over_meas=xo["T_pred_s"] / xo["t_action_s"],
            T_setup_meas_ms=1e3 * xo["t_setup_s"], T_param_ms=1e3 * xo["t_param_s"],
            measured_fastest_xi=fastest["xi"], T_action_fastest_ms=1e3 * fastest["t_action_s"],
            n_xi_measured=len(b),
            xi_opt_pred_s1=opt["J_pred_s1"]["xi"], xi_opt_pred_s10=opt["J_pred_s10"]["xi"],
            xi_opt_pred_s20=opt["J_pred_s20"]["xi"],
            baseline_M=bl["M"], baseline_n_img=bl["n_img"], baseline_T_action_ms=1e3 * bl["t_action_s"],
            baseline_T_setup_ms=1e3 * bl["t_setup_s"], baseline_E_op=eb.get("E_op"),
            speedup_vs_baseline=bl["t_action_s"] / xo["t_action_s"]))
    C.write_csv(C.RESULTS.parent / "parameter_results.csv", rows)
    tb = []
    for r in D["bench"]:
        tb.append({k: r.get(k) for k in (
            "config", "family", "N", "set", "xi", "is_xi_opt", "rc", "kc", "M", "p", "eta", "n_unique", "n_img",
            "t_action_s", "t_action_q1_s", "t_action_q3_s", "t_action_min_s", "t_action_max_s", "t_action_n",
            "t_real_s", "t_spread_s", "t_fft_influence_s", "t_gather_s", "t_degree_s", "t_existing_pairdiff_matvec_s",
            "T_pred_s", "T_pred_real_s", "T_pred_spread_gather_s", "T_pred_fft_s", "T_pred_overhead_s",
            "t_setup_s", "t_setup_pairs_s", "t_setup_real_weights_degree_s", "t_setup_mesh_stencil_s",
            "t_setup_mesh_degree_s", "T_setup_pred_s", "t_param_s")})
    C.write_csv(C.RESULTS.parent / "timing_breakdown.csv", tb)
    # error contributions at xi_opt: certified (absolute bound / ||Gamma_ref||) vs measured
    from parameter_optimizer import FourierShells, eps_quad_estimate
    ec = []
    for r in D["errors"]["rows"]:
        if "meas_real_rel" not in r:
            continue
        n2 = r["norm2_ref"]
        _, L = load(r["config"])
        r["eps_quad_abs_est"] = eps_quad_estimate(r["xi"], L, r["kc"], r["M"], r["p"], r["N"] / L ** 3,
                                                  FourierShells(r["xi"], L))[0]
        ec.append(dict(config=r["config"], N=r["N"], xi=r["xi"], rc=r["rc"], kc=r["kc"], M=r["M"],
                       cert_S_rel=r["cert_S_abs"] / n2, meas_S_rel=r["meas_real_rel"],
                       ratio_S=r["cert_S_abs"] / n2 / max(r["meas_real_rel"], 1e-300),
                       cert_F_rel=r["cert_F_abs"] / n2, meas_F_rel=r["meas_fourier_rel"],
                       ratio_F=r["cert_F_abs"] / n2 / max(r["meas_fourier_rel"], 1e-300),
                       cert_mesh_rel=r["cert_mesh_abs"] / n2, meas_mesh_rel=r["meas_mesh_rel"],
                       ratio_mesh=r["cert_mesh_abs"] / n2 / max(r["meas_mesh_rel"], 1e-300),
                       quad_est_rel=r.get("eps_quad_abs_est", np.nan) / n2,
                       cert_total_rel=(r["cert_S_abs"] + r["cert_F_abs"] + r["cert_mesh_abs"]) / n2,
                       meas_E_op=r["E_op"], ratio_total=(r["cert_S_abs"] + r["cert_F_abs"] + r["cert_mesh_abs"]) / n2 / r["E_op"]))
    C.write_csv(C.RESULTS / "error_contributions.csv", ec)
    # probe-action errors per class (re-aggregated from the stored per-probe records)
    from error_validation import probe_class
    agg = {}
    for pr in D["errors"]["probes"]:
        k = (pr["config"], pr["set"], round(pr["xi"], 4), probe_class(pr["probe"]))
        a = agg.setdefault(k, dict(config=k[0], set=k[1], xi=k[2], probe_class=k[3], n=0, abs_max=0.0,
                                   op_normalised_max=0.0, rel_max=None))
        a["n"] += 1
        a["abs_max"] = max(a["abs_max"], pr["abs"])
        a["op_normalised_max"] = max(a["op_normalised_max"], pr["op_normalised"])
        if pr["rel"] is not None:
            a["rel_max"] = pr["rel"] if a["rel_max"] is None else max(a["rel_max"], pr["rel"])
    C.write_csv(C.RESULTS / "action_errors_by_class.csv", list(agg.values()))
    with open(C.RESULTS / "error_contributions.md", "w") as fh:
        fh.write("| config | xi | M | real: bound / measured | Fourier: bound / measured | mesh: bound / measured | "
                 "quad (estimate) | total: bound / measured E_op |\n|---|---|---|---|---|---|---|---|\n")
        for e in ec:
            fh.write(f"| {e['config']} | {e['xi']:.3f} | {e['M']} | {e['cert_S_rel']:.2e} / {e['meas_S_rel']:.2e} "
                     f"(x{e['ratio_S']:.1f}) | {e['cert_F_rel']:.2e} / {e['meas_F_rel']:.2e} (x{e['ratio_F']:.0f}) | "
                     f"{e['cert_mesh_rel']:.2e} / {e['meas_mesh_rel']:.2e} (x{e['ratio_mesh']:.0f}) | "
                     f"{e['quad_est_rel']:.1e} | {e['cert_total_rel']:.2e} / {e['meas_E_op']:.2e} (x{e['ratio_total']:.1f}) |\n")
    return rows


def reproducibility(D):
    import hashlib
    from parameter_optimizer import SPLIT, XI_GRID, A_MAX, R_PAD, C_p
    from error_validation import REF_REL_TOL
    from clustered_configurations import FAMILIES, SEEDS, EXCLUDED

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    files = sorted(list(C.CONFIGS.glob("*.npz")) + list(C.RESULTS.glob("*")) + list(C.FIGURES.glob("*.png"))
                   + [C.HERE / "parameter_results.csv", C.HERE / "timing_breakdown.csv"])
    cmds = (C.RESULTS / "commands.log").read_text().splitlines() if (C.RESULTS / "commands.log").exists() else []
    rec = dict(
        environment=C.provenance(), commands_executed=cmds,
        model=dict(state_point="LJ + kernel A, toy_configs " + C.BASE_CONFIG, law=C.LAW, gamma=C.GAMMA, kappa=C.KAPPA,
                   r_ref=C.R_REF, amplitude_c=C.AMP, density=C.DENSITY, periodic="full lattice sum, k = 0 retained",
                   kernel="K(r) = c exp(-kappa r)/r r r^T/r^2", split="Gaussian scale s = xi^2 (radial_kernels.RadialEwald)"),
        target=dict(eps=C.EPS_TARGET, norm="spectral, relative to ||Gamma_ref||_2", split=SPLIT, p=C.P_ORDER,
                    C_p=C_p(C.P_ORDER), xi_grid=[float(v) for v in XI_GRID], fine_step=0.005, a_max=A_MAX,
                    explicit_image_radius="(A_MAX + R_PAD)/xi", R_PAD=R_PAD),
        baseline=dict(C.BASELINE, source=f"toy_configs.CONFIGS['{C.BASE_CONFIG}']"),
        reference=dict(kind="v2.LatticeFriction (near images direct, far images tensor Chebyshev fit)",
                       rel_tol=REF_REL_TOL, convergence_and_selfcheck=C.read_json(C.RESULTS / "references.json")["checks"]),
        configurations=dict(families=FAMILIES, seeds=SEEDS, d_min=C.D_MIN, rng="default_rng([20261017, N, seed, attempt])",
                            excluded=EXCLUDED),
        timing_protocol=dict(threads="OMP/OPENBLAS/MKL_NUM_THREADS = 1, one process, nothing else running",
                             warmup_calls=3, repetitions="max(15, min(60, 1.5 s / single call)) for the action; "
                             "0.6 s budget for components; statistic = median, variability = IQR and min/max",
                             setup="median of 3 builds", X_vectors="default_rng([20261017, 7, N]), 4 vectors cycled"),
        not_committed=dict(raw="raw/<config>_Gref.npy dense references; regenerated deterministically by --stage refs"),
        files_sha256={str(f.relative_to(C.HERE)): sha(f) for f in files if f.is_file() and f.suffix != ".tmp"})
    C.write_json(C.HERE / "reproducibility.json", rec)


def main():
    C.log_command()
    style()
    D = load_all()
    fig1(D); fig2(D); fig3(D); fig4(D); fig5(D); fig6(D)
    rows = tables(D)
    reproducibility(D)
    C.log(f"figures written to {C.FIGURES}; parameter_results.csv with {len(rows)} rows")


if __name__ == "__main__":
    main()
