#!/usr/bin/env python3
"""
Figures of the fixed-accuracy scaling benchmark (run_scaling.py): direct pair-summed periodic sum vs optimised PPPM,
N = 128 ... 4096. Reads results_scaling/, writes figures_scaling/. Guide lines are labelled as such; they are not fits.

    python3 make_figures_scaling.py
"""
import common as C
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from run_scaling import RESULTS, NS, FAMS, name_of, path, QS  # noqa: E402

FIG = C.HERE / "figures_scaling"
TAU = C.EPS_TARGET
COL = {"direct": "#d62728", "pppm": "#1f77b4"}
LS = {"strong": "-", "uniform": "--"}
MK = {"strong": "o", "uniform": "s"}


def load_all():
    data = {}
    for fam in FAMS:
        for N in NS:
            name = name_of(fam, N)
            if not path("scale", name).exists():
                continue
            d = dict(scale=C.read_json(path("scale", name)), direct=C.read_json(path("direct", name)),
                     verify=C.read_json(path("verify", name)), sel=C.read_json(path("selection", name)),
                     ref=C.read_json(path("reference", name)))
            if path("memory", name).exists():
                d["mem"] = C.read_json(path("memory", name))
            data[(fam, N)] = d
    return data


def series(data, fam, fn):
    Ns = [N for N in NS if (fam, N) in data]
    return np.array(Ns, float), np.array([fn(data[(fam, N)]) for N in Ns], float)


def act(d, m):
    s = d["scale"]
    k = "direct" if m == "direct" else f"pppm_{s['pppm']['impl']}"
    a = s["actions"][k]
    return a["median_ms"], a["q1_ms"], a["q3_ms"]


def setup(d, m):
    s = d["scale"]
    k = "direct" if m == "direct" else f"pppm_{s['pppm']['impl']}"
    a = s["setups"][k]
    return a["median_ms"], a["q1_ms"], a["q3_ms"]


LAB = {"direct": "Direct (pair-summed $K^{per}$)", "pppm": "PPPM (optimised, matrix-free)"}


def fig_action(data):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for fam in FAMS:
        for m in ("direct", "pppm"):
            Ns, med = series(data, fam, lambda d: act(d, m)[0])
            if not len(Ns):
                continue
            _, q1 = series(data, fam, lambda d: act(d, m)[1])
            _, q3 = series(data, fam, lambda d: act(d, m)[2])
            ax.errorbar(Ns, med, yerr=[med - q1, q3 - med], fmt=MK[fam] + LS[fam], color=COL[m], ms=5, capsize=2,
                        mfc=COL[m] if fam == "strong" else "white", label=f"{LAB[m]}, {fam}")
    Ns = np.array(NS, float)
    if ("strong", 512) in data:
        a0 = act(data[("strong", 512)], "direct")[0]
        p0 = act(data[("strong", 512)], "pppm")[0]
        ax.plot(Ns, a0 * (Ns / 512) ** 2, ":", color="0.4", lw=1, label="guide $\\propto N^2$ (through direct, 512)")
        ax.plot(Ns, p0 * Ns * np.log(Ns) / (512 * np.log(512)), "-.", color="0.4", lw=1,
                label="guide $\\propto N\\log N$ (through PPPM, 512)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (fixed density $\\rho$ = 0.75)")
    ax.set_ylabel("one action $\\Gamma X$, median [ms]")
    ax.set_title("Single action at fixed accuracy $\\tau$ = 2e-7")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "figS1_action_vs_N.png", dpi=150)
    plt.close(fig)


def fig_normalised(data):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (lab, f) in zip(axs, (("$T_{action}/(N \\log N)$ [ns]", lambda N: N * np.log(N)),
                                  ("$T_{action}/N^2$ [ns]", lambda N: N ** 2.0))):
        for fam in FAMS:
            for m in ("direct", "pppm"):
                Ns, med = series(data, fam, lambda d: act(d, m)[0])
                if len(Ns):
                    ax.plot(Ns, 1e6 * med / f(Ns), MK[fam] + LS[fam], color=COL[m], mfc=COL[m] if fam == "strong" else "white",
                            label=f"{LAB[m]}, {fam}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, which="both")
    axs[0].legend(fontsize=7)
    fig.suptitle("Normalised single-action time (flat = that complexity over this range)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figS2_action_normalised.png", dpi=150)
    plt.close(fig)


def fig_setup(data):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for fam in FAMS:
        for m in ("direct", "pppm"):
            Ns, med = series(data, fam, lambda d: setup(d, m)[0])
            if len(Ns):
                ax.plot(Ns, med, MK[fam] + LS[fam], color=COL[m], mfc=COL[m] if fam == "strong" else "white",
                        label=f"{LAB[m]}, {fam}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("setup per configuration, median [ms]")
    ax.set_title("Setup (positions changed); one-time preprocessing excluded")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "figS3_setup_vs_N.png", dpi=150)
    plt.close(fig)


def fig_total(data):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(QS)))
    for fam in FAMS:
        for q, c in zip(QS, cols):
            Ns, r = series(data, fam, lambda d: d["scale"]["summary"]["total_ratio_direct_over_pppm"][str(q)])
            if len(Ns):
                ax.plot(Ns, r, MK[fam] + LS[fam], color=c, mfc=c if fam == "strong" else "white", label=f"q = {q}, {fam}")
        Ns, r = series(data, fam, lambda d: d["scale"]["summary"]["action_ratio_direct_over_pppm"])
        if len(Ns):
            ax.plot(Ns, r, "k" + MK[fam] + LS[fam], mfc="k" if fam == "strong" else "white", label=f"q $\\to\\infty$ (action only), {fam}")
    ax.axhline(1.0, color="r", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("$T_{total}^{Direct}(q)\\,/\\,T_{total}^{PPPM}(q)$  (> 1: PPPM faster)")
    ax.set_title("$T_{total}(q) = T_{setup} + q\\,T_{action}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "figS4_total_ratio.png", dpi=150)
    plt.close(fig)


def fig_memory(data):
    if not any("mem" in d for d in data.values()):
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    items = (("direct_persistent", "Direct: pair tensors + indices (persistent)", COL["direct"], "-"),
             ("direct_setup_peak", "Direct: setup peak", COL["direct"], ":"),
             ("pppm_pairdiff_persistent", "PPPM: images, stencil, degree (persistent)", COL["pppm"], "-"),
             ("pppm_mesh_bytes", "PPPM: mesh / influence (one-time)", COL["pppm"], "-."),
             ("pppm_pairdiff_setup_peak", "PPPM: setup peak", COL["pppm"], ":"))
    for k, lab, c, ls in items:
        Ns, v = series({k_: d for k_, d in data.items() if "mem" in d and k_[0] == "strong"}, "strong",
                       lambda d: d["mem"][k] / 2 ** 20)
        if len(Ns):
            ax.plot(Ns, v, "o" + ls, color=c, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("MiB (tracemalloc / array nbytes)")
    ax.set_title("Memory, strong configurations")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "figS5_memory.png", dpi=150)
    plt.close(fig)


def fig_params(data):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.6))
    specs = (("PPPM optimal p", lambda d: d["scale"]["pppm"]["p"]),
             ("PPPM $\\xi$", lambda d: d["scale"]["pppm"]["xi"]),
             ("PPPM $M^3/N$", lambda d: d["scale"]["pppm"]["M"] ** 3 / d["scale"]["N"]),
             ("PPPM real-space interactions / particle", lambda d: d["scale"]["pppm"]["n_img"] / d["scale"]["N"]),
             ("Direct $R_d$ (certified)", lambda d: d["direct"]["R_d"]),
             ("Direct images / particle (setup work)", lambda d: d["direct"]["n_images"] / d["direct"]["N"]))
    for ax, (lab, fn) in zip(axs.ravel(), specs):
        for fam in FAMS:
            Ns, v = series(data, fam, fn)
            if len(Ns):
                ax.plot(Ns, v, MK[fam] + LS[fam], color="k", mfc="k" if fam == "strong" else "white", label=fam)
        ax.set_xscale("log", base=2)
        ax.set_title(lab, fontsize=9)
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
    axs[0, 0].set_ylim(5, 13)
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "figS6_parameters_vs_N.png", dpi=150)
    plt.close(fig)


def fig_accuracy(data):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for fam in FAMS:
        Ns, e = series(data, fam, lambda d: d["scale"]["pppm"]["E_op"])
        if len(Ns):
            ax.plot(Ns, e, MK[fam] + LS[fam], color=COL["pppm"], mfc=COL["pppm"] if fam == "strong" else "white",
                    label=f"PPPM measured $E_{{op}}$, {fam}")
            _, c = series(data, fam, lambda d: d["scale"]["pppm"]["cert_rel"] * d["ref"]["lam_lb"] / d["ref"]["norm2"])
            ax.plot(Ns, c, MK[fam] + ":", color=COL["pppm"], alpha=0.5, label=f"PPPM certified bound / $\\|\\Gamma\\|$, {fam}")
        Ns, e = series(data, fam, lambda d: d["direct"]["E_ref"])
        if len(Ns):
            ax.plot(Ns, e, MK[fam] + LS[fam], color=COL["direct"], mfc=COL["direct"] if fam == "strong" else "white",
                    label=f"Direct measured $E_{{op}}$, {fam}")
            _, c = series(data, fam, lambda d: d["direct"]["bound_abs"] / d["ref"]["norm2"])
            ax.plot(Ns, c, MK[fam] + ":", color=COL["direct"], alpha=0.5, label=f"Direct certified bound / $\\|\\Gamma\\|$, {fam}")
    ax.axhline(TAU, color="k", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("relative spectral-norm error")
    ax.set_title("Accuracy: measured vs certified ($\\tau$ = 2e-7)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "figS7_accuracy_vs_N.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = load_all()
    if not data:
        raise SystemExit("no scale results")
    fig_action(data)
    fig_normalised(data)
    fig_setup(data)
    fig_total(data)
    fig_memory(data)
    fig_params(data)
    fig_accuracy(data)
    C.log(f"figures written to {FIG} ({len(data)} configurations)")


if __name__ == "__main__":
    main()
