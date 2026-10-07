"""Figures (matplotlib only). Colours: data = ink, Model A = blue, Model B = orange, spline = aqua."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fit_models import MODELS, LABELS  # noqa: E402
import fourier_analysis as FA  # noqa: E402

INK, MUTED = "#222222", "#8a8a8a"
COL = {"A": "#2a78d6", "B": "#eb6834", "spline": "#1baf7a", "T": "#4a3aa7"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "legend.frameon": False})


def radial_figure(D, path, title):
    """Panels (a)-(g) for one dataset D (dict with rbar, gL, gL_se, gL_std, gT, gT_se, n, use, fits, avg, spline,
    truth (optional callable), r_marks)."""
    use = D["use"]
    r, y, se = D["rbar"][use], D["gL"][use], D["gL_se"][use]
    fig, ax = plt.subplots(3, 3, figsize=(13, 11))
    ax = ax.ravel()
    rr = np.linspace(max(r.min() * 0.9, 1e-3), r.max() * 1.05, 300)
    # (a)
    a = ax[0]
    a.errorbar(r, y, yerr=2 * se, fmt="o", ms=4, color=INK, capsize=2, label="binned γ_L ± 2 SE")
    for m in MODELS:
        p = D["fits"][m]["params"]
        a.plot(r, D["fits"][m]["pred"], "-", color=COL[m], lw=2,
               label=f"{LABELS[m]} (bin avg): amp {p[0]:.3g}, κ {p[1]:.3g}, χ²/dof {D['fits'][m]['chi2_per_dof']:.3g}")
        a.plot(rr, MODELS[m](rr, *p), ":", color=COL[m], lw=1)
    if D.get("truth") is not None:
        a.plot(rr, D["truth"](rr), "--", color=MUTED, lw=1, label="input pair law (no environment factor)")
    a.set(xlabel="r", ylabel="γ_L(r)", title="(a) γ_L(r) and fitted models", yscale=D.get("yscale", "linear"))
    a.legend(fontsize=6.5, loc="upper right")
    # (b)
    a = ax[1]
    a.errorbar(r, D["gT"][use], yerr=2 * D["gT_se"][use], fmt="s", ms=4, color=COL["T"], capsize=2,
               label="binned γ_T ± 2 SE")
    a.errorbar(r, y, yerr=2 * se, fmt="o", ms=3, color=INK, alpha=0.35, label="γ_L (reference)")
    a.axhline(0, color=MUTED, lw=0.8)
    a.set(xlabel="r", ylabel="γ_T(r)", title="(b) transverse pair friction γ_T(r)")
    a.legend(fontsize=7)
    # (c)
    a = ax[2]
    stable = np.abs(y) > 3 * se
    ratio = D["gT"][use] / y
    rse = np.abs(ratio) * np.sqrt((D["gT_se"][use] / np.maximum(np.abs(D["gT"][use]), 1e-300)) ** 2 + (se / np.abs(y)) ** 2)
    a.errorbar(r[stable], ratio[stable], yerr=2 * np.minimum(rse[stable], 1e3), fmt="o", ms=4, color=COL["T"], capsize=2)
    a.axhline(0, color=MUTED, lw=0.8)
    a.set(xlabel="r", ylabel="γ_T/γ_L", title="(c) γ_T/γ_L (bins with |γ_L| > 3 SE)")
    # (d), (e)
    for k, m in ((3, "A"), (4, "B")):
        a = ax[k]
        res = (y - np.array(D["fits"][m]["pred"])) / se
        a.axhspan(-2, 2, color="#f0f0f0")
        a.plot(r, res, "o-", color=COL[m], ms=4, lw=1)
        a.axhline(0, color=MUTED, lw=0.8)
        a.set(xlabel="r", ylabel="(data − model)/SE",
              title=f"({'d' if m == 'A' else 'e'}) normalised residuals, Model {m}")
    # (f)
    a = ax[5]
    a.errorbar(r, y, yerr=2 * se, fmt="o", ms=4, color=INK, capsize=2, label="binned γ_L ± 2 SE")
    a.plot(rr, D["spline"](rr), "-", color=COL["spline"], lw=2,
           label=f"weighted smoothing spline, χ²/dof {D['fits']['spline']['chi2_per_dof']:.3g}")
    a.fill_between(r, D["gL"][use] - D["gL_std"][use], D["gL"][use] + D["gL_std"][use], color=INK, alpha=0.08,
                   label="± conditional std at fixed r" if D.get("std_label") is None else D["std_label"])
    a.set(xlabel="r", ylabel="γ_L(r)", title="(f) empirical γ_L with nonparametric baseline",
          yscale=D.get("yscale", "linear"))
    a.legend(fontsize=7)
    # (g)
    a = ax[6]
    w = np.diff(D["edges"])
    a.bar(D["edges"][:-1], D["n"], width=w, align="edge", color="#cfd8e3", edgecolor="white", linewidth=1)
    a.bar(D["edges"][:-1][use], D["n"][use], width=w[use], align="edge", color="#7d93ad", edgecolor="white",
          linewidth=1, label="bins used in fits")
    a.set(xlabel="r", ylabel=D.get("count_label", "pair samples per bin"), title="(g) samples per radial bin",
          yscale="log")
    a.legend(fontsize=7)
    # (h), (i) only if time-dependent memory is present
    if "KL_t" in D:
        a = ax[7]
        for kk, b in enumerate(D["show_bins"]):
            a.plot(D["t"], D["KL_t"][b], color=plt.cm.viridis(kk / max(len(D["show_bins"]) - 1, 1)),
                   label=f"r ≈ {D['rbar'][b]:.2f}")
        a.axhline(0, color=MUTED, lw=0.8)
        a.set(xlabel="t", ylabel="K_L(r,t)", title="(h) longitudinal pair memory K_L(r,t)")
        a.legend(fontsize=7)
        a = ax[8]
        for kk, b in enumerate(D["show_bins"]):
            a.plot(D["t"], D["GL_T"][b], color=plt.cm.viridis(kk / max(len(D["show_bins"]) - 1, 1)))
            if D.get("truth") is not None:
                a.axhline(D["truth_bin"][b], color=plt.cm.viridis(kk / max(len(D["show_bins"]) - 1, 1)), ls=":", lw=1)
        a.set(xlabel="upper limit T", ylabel="∫₀ᵀ K_L(r,t) dt", title="(i) cumulative integral vs T (dotted: input)")
    else:
        for a in ax[7:]:
            a.axis("off")
        ax[7].text(0, 0.9, "(h), (i): no time-dependent memory in this dataset", transform=ax[7].transAxes,
                   fontsize=9, color=MUTED)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fourier_figure(path, kappa=1.0, A=1.0, B=1.0):
    k = np.geomspace(1e-2, 1e3, 400)
    lA, tA = FA.analytic_A(k, A, kappa)
    lB, tB = FA.analytic_B(k, B, kappa)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a, (lam_l, lam_t, m) in zip(ax, ((lA, tA, "A"), (lB, tB, "B"))):
        a.loglog(k, np.where(lam_t > 0, lam_t, np.nan), "-", color=COL[m], lw=2, label="λ_T(k) > 0")
        a.loglog(k, np.where(lam_l > 0, lam_l, np.nan), "-", color=INK, lw=2, label="λ_L(k) > 0")
        a.loglog(k, np.where(lam_l < 0, -lam_l, np.nan), "--", color=INK, lw=2, label="−λ_L(k) (λ_L < 0)")
        p = -2 if m == "A" else -4
        a.loglog(k[k > 5], 30 * k[k > 5] ** float(p) * (1 if m == "A" else 10), ":", color=MUTED, label=f"∝ k^{p}")
        a.set(xlabel="|k| (κ = 1)", ylabel="|eigenvalue|",
              title=("Model A: A e^{-κr}/r P_L, A = 1" if m == "A" else "Model B: B r e^{-κr} P_L, B = 1"))
        a.legend(fontsize=8)
    fig.suptitle("Fourier eigenvalues of the candidate tensor kernels (closed forms, verified by QAWF quadrature)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def realspace_figure(path, marks, kappa=1.0):
    r = np.linspace(1e-3, 8, 800)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    gA = np.exp(-kappa * r) / r
    gB = r * np.exp(-kappa * r)
    ax.plot(r, gA / np.interp(1 / kappa, r, gA), color=COL["A"], lw=2, label="Model A ∝ e^{-κr}/r  (~1/r at r→0)")
    ax.plot(r, gB / np.interp(1 / kappa, r, gB), color=COL["B"], lw=2, label="Model B ∝ r e^{-κr}  (~r at r→0, peak r = 1/κ)")
    ax.axvline(1 / kappa, color=COL["B"], lw=0.8, ls=":")
    for k, (lab, rmin, rtyp) in enumerate(marks):
        c = ["#555555", "#999999", "#bbbbbb"][k % 3]
        ax.axvspan(0, rmin, color=c, alpha=0.12)
        ax.axvline(rmin, color=c, lw=1.2, ls="--", label=f"{lab}: min sampled r = {rmin:.2f}")
        ax.axvline(rtyp, color=c, lw=1.2, ls="-.", label=f"{lab}: median nearest-neighbour r = {rtyp:.2f}")
    ax.set(xlabel="r (κ = 1)", ylabel="γ(r)/γ(1/κ)", ylim=(0, 4), title="Real-space shapes, normalised at r = 1/κ")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
