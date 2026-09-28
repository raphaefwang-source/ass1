#!/usr/bin/env python3
"""
Exact decomposition of the particle-mesh (PPPM) error of the screened longitudinal
kernel, and a clean h^p scaling test at FIXED physical Fourier cutoff kc.

For a mesh spacing h, k_n = k + (2 pi/h) n, and a particle at q:
    S_h(k,q) = sum_n W_hat(k_n) e^{-i k_n.q} = W_hat(k) e^{-i k.q} [1 + alpha_h(k,q)],
    H_h(k)   = |W_hat(k)|^2 G_h(k),    D_h(k) = H_h(k) - Khat_L(k).
The spread/FFT/G/IFFT/gather pair response is K_h,ij = (1/V) sum_k G_h conj(S_i) S_j, hence
    K_h,ij - K_F,ij = (1/V) sum_k e^{i k.r_ij} [ D_h + H_h (alpha_i^* + alpha_j + alpha_i^* alpha_j) ].

Three independent computations per test:
  (1) K_F  exact retained lattice Fourier sum,
  (2) K_h  literal spread -> FFT -> G_h -> IFFT -> gather (one spread per source particle),
  (3) K_F + reconstruction from D_h and alpha (alpha from the stencil DFT; the Poisson form
      sum_n W_hat(k_n) e^{-i k_n.q} is checked separately with a truncated n-sum).
The previous 'alias' / 'transfer' diagnostics are recomputed and mapped onto these terms.

Model unchanged, no transverse projector, no clipping of Khat_L, no stabilisation of Gamma_h.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_ewald_longitudinal import set_style, INK2  # noqa: E402
from test_parameter_selection import (Config, Mesh, lap_full, lattice_modes, khat_AB, fourier_pairs,  # noqa: E402
                                      ewald, PAIRS6, ROUND)

I3 = np.eye(3)
XI = 1.0
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]   # fixed categorical order


# ----------------------------------------------------------------------------
def full_from_pairs(Kp, pi, pj, N):
    F = np.zeros((N, N, 3, 3))
    F[pi, pj] = Kp
    F[pj, pi] = Kp
    return F


def offdiag(N):
    return ~np.eye(N, dtype=bool)


def mesh_at(L, K, kc, M, p, rfft_grid=True):
    """Mesh with exactly M points per side (eta chosen so that ceil(L/h_m) = M)."""
    eta = kc * (L / M) / np.pi * (1 + 1e-12)
    m = Mesh(L, K, kc, eta, p, rfft_grid=rfft_grid)
    assert m.M == M
    return m


def literal_W(mesh, x):
    """Literal spread -> FFT -> G_h -> IFFT -> adjoint gather, one scalar spread per source particle."""
    M, N = mesh.M, len(x)
    idx, wt = mesh.flat(x)
    W = np.empty((N, N, 3, 3))
    for j in range(N):
        rh = np.fft.rfftn(np.bincount(idx[j], wt[j], M ** 3).reshape(M, M, M))
        for (a, b) in PAIRS6:
            phi = np.fft.irfftn(mesh._G(a, b) * rh, s=(M, M, M), axes=(0, 1, 2)).ravel()
            W[:, j, a, b] = W[:, j, b, a] = (phi[idx] * wt).sum(axis=1) / mesh.h ** 3
    return W


def decomposition(mesh, x):
    """D_h, linear-alpha and quadratic-alpha parts of K_h - K_F (full N x N x 3 x 3 arrays)."""
    N = len(x)
    s = mesh.stencil_dft(x)
    o, mv, kv = mesh.mc, mesh.mv, mesh.kv
    S = s[:, 0, mv[:, 0] + o] * s[:, 1, mv[:, 1] + o] * s[:, 2, mv[:, 2] + o]
    What = np.prod(np.sinc(kv * mesh.h / (2 * np.pi)) ** mesh.p, axis=1)
    U = np.exp(1j * x @ kv.T)                                    # e^{i k.x}
    alpha = S / (What * U.conj()) - 1.0
    Khat = mesh.A[:, None, None] * I3 + mesh.B[:, None, None] * kv[:, :, None] * kv[:, None, :]
    H = What[:, None, None] ** 2 * mesh.Gm
    Dk = H - Khat

    def blocks(Lm, Rm, T):
        out = np.empty((N, N, 3, 3))
        im = 0.0
        for (a, b) in PAIRS6:
            Z = (Lm * T[:, a, b]) @ Rm.T / mesh.V
            out[:, :, a, b] = out[:, :, b, a] = Z.real
            im = max(im, np.abs(Z.imag).max())
        return out, im

    ED, i1 = blocks(U, U.conj(), Dk)
    EL1, i2 = blocks(U * alpha.conj(), U.conj(), H)
    EL2, i3 = blocks(U, U.conj() * alpha, H)
    EQ, i4 = blocks(U * alpha.conj(), U.conj() * alpha, H)
    return dict(ED=ED, ELin=EL1 + EL2, EQuad=EQ, imag=max(i1, i2, i3, i4),
                D_rel=float(np.abs(Dk).max() / np.abs(Khat).max()), alpha_max=float(np.abs(alpha).max()))


def poisson_alpha_check(mesh, x, nmax=1000):
    """Per-axis S from the truncated Poisson sum vs the exact stencil DFT (relative to the alias part)."""
    s_ex = mesh.stencil_dft(x)
    k1 = 2 * np.pi / mesh.L * np.arange(-mesh.mc, mesh.mc + 1)
    n = np.arange(-nmax, nmax + 1)
    q = k1[:, None] + 2 * np.pi / mesh.h * n[None, :]
    w = np.sinc(q * mesh.h / (2 * np.pi)) ** mesh.p
    sP = np.einsum("mn,jamn->jam", w, np.exp(-1j * x[:, :, None, None] * q[None, None]))
    s0 = np.sinc(k1 * mesh.h / (2 * np.pi)) ** mesh.p * np.exp(-1j * x[:, :, None] * k1)
    return float(np.abs(sP - s_ex).max() / np.abs(s_ex - s0).max()), float(np.abs(sP - s_ex).max() / np.abs(s_ex).max())


# ----------------------------------------------------------------------------
def run(args):
    out = HERE / "mesh_error_decomposition_results"
    out.mkdir(exist_ok=True)
    log = []
    say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(t) for t in a)))
    K = ewald(1.0, XI)
    Ms = [12, 16, 20, 24, 32, 40, 48, 64]
    ps = [2, 3, 4, 5, 6, 7, 8]
    kcs = [4.0, 6.0]
    n_off = args.offsets
    rng = np.random.default_rng(424242)
    cfgs = [Config("A", 30, 6.0, 7, 1.0), Config("B", 40, 10.0, 13, 1.0)]
    fixed, phase = [], []
    t0 = time.perf_counter()
    for cfg in cfgs:
        GS, GL, _ = cfg.split(XI)
        N, pi, pj = cfg.N, cfg.pi, cfg.pj
        od = offdiag(N)
        for kc in kcs:
            mv, kv, _ = lattice_modes(cfg.L, kc)
            A, B = khat_AB(K, np.linalg.norm(kv, axis=1))
            KF = fourier_pairs(cfg.d, kv, A, B, cfg.V)
            KF_full = full_from_pairs(KF, pi, pj, N)
            GLkc = cfg.lap(KF)
            eps_F = np.linalg.norm(GLkc - GL, 2) / cfg.norm2
            for p in ps:
                for M in Ms:
                    h = cfg.L / M
                    if kc * h / np.pi >= 1.0:
                        continue
                    mesh = mesh_at(cfg.L, K, kc, M, p)
                    R = dict(config=cfg.name, kc=kc, p=p, M=M, h=h, eta_actual=mesh.eta_actual,
                             n_modes=mesh.n_modes, n_excluded=mesh.n_excluded, eps_F=eps_F,
                             q=mesh.eta_actual / (2 - mesh.eta_actual))
                    R["qp"] = R["q"] ** p
                    # (2) literal and mode-space responses
                    Wl = literal_W(mesh, cfg.x)
                    Wm, _ = mesh.pair_matrix(cfg.x)
                    R["literal_vs_modespace"] = float(np.abs(Wl - Wm)[od].max() / np.abs(Wl)[od].max())
                    # (3) reconstruction
                    dec = decomposition(mesh, cfg.x)
                    E_true = (Wl - KF_full)[od]
                    E_rec = (dec["ED"] + dec["ELin"] + dec["EQuad"])[od]
                    R["recon_identity_err"] = float(np.abs(E_true - E_rec).max() / np.abs(E_true).max())
                    R["recon_identity_err_relK"] = float(np.abs(E_true - E_rec).max() / np.abs(KF_full[od]).max())
                    R["recon_imag_rel"] = dec["imag"] / np.abs(E_true).max()
                    R["D_rel_max"] = dec["D_rel"]
                    R["alpha_max"] = dec["alpha_max"]
                    R["poisson_vs_stencil"], R["poisson_vs_stencil_relS"] = poisson_alpha_check(mesh, cfg.x, nmax=args.nmax)
                    # operator norms of every piece (Laplacians of the off-diagonal blocks)
                    nrm = lambda Wfull: float(np.linalg.norm(lap_full(Wfull), 2) / cfg.norm2)
                    GLm = lap_full(Wl)
                    R["eps_mesh"] = float(np.linalg.norm(GLm - GLkc, 2) / cfg.norm2)
                    R["eps_recon"] = nrm(dec["ED"] + dec["ELin"] + dec["EQuad"])
                    R["eps_D"] = nrm(dec["ED"])
                    R["eps_lin"] = nrm(dec["ELin"])
                    R["eps_quad"] = nrm(dec["EQuad"])
                    R["eps_alpha"] = nrm(dec["ELin"] + dec["EQuad"])
                    # old diagnostics (unchanged definitions from test_parameter_selection.py)
                    al = full_from_pairs(mesh.alias_TI(cfg.d), pi, pj, N)
                    tr = Wl - KF_full - al
                    R["old_alias"] = nrm(al)
                    R["old_transfer"] = nrm(tr)
                    R["quad_offdiag"] = nrm(dec["EQuad"] - al)
                    R["old_transfer_vs_lin_plus_quadoff"] = float(
                        np.abs((tr - (dec["ED"] + dec["ELin"] + dec["EQuad"] - al))[od]).max() / np.abs(tr[od]).max())
                    # operator / PSD checks (Gamma_S exact, so only mesh + fixed Fourier cutoff enter)
                    Gh = GS + GLm
                    nGh = np.linalg.norm(Gh, 2)
                    R["sym_resid"] = float(np.linalg.norm(Gh - Gh.T) / np.linalg.norm(Gh))
                    R["trans_resid"] = float(max(np.linalg.norm(Gh @ cfg.T[:, c]) for c in range(3)) / nGh)
                    R["err_abs"] = float(np.linalg.norm(Gh - cfg.G, 2))
                    R["err_2"] = R["err_abs"] / cfg.norm2
                    R["lam_star"] = cfg.lam_star
                    R["lam_h"] = cfg.restricted_min(Gh)
                    R["weyl_bound"] = cfg.lam_star - R["err_abs"]
                    R["weyl_holds"] = bool(R["lam_h"] >= R["weyl_bound"] - 1e-13 * cfg.norm2)
                    R["certified"] = bool(R["err_abs"] < cfg.lam_star)
                    R["indefinite"] = bool(R["lam_h"] < -ROUND * nGh)
                    R["psd_class"] = "A" if R["certified"] else ("C" if R["indefinite"] else "B")
                    fixed.append(R)
                    # (B) phase-averaged: random global mesh offsets in one cell, separations fixed
                    mesh_m = mesh
                    e_off, lam_off, err_off, cls_off = [], [], [], []
                    for _ in range(n_off):
                        delta = rng.uniform(0, h, size=3)
                        Wo, _ = mesh_m.pair_matrix(cfg.x + delta)
                        GLo = lap_full(Wo)
                        e_off.append(np.linalg.norm(GLo - GLkc, 2) / cfg.norm2)
                        Gho = GS + GLo
                        ea = np.linalg.norm(Gho - cfg.G, 2)
                        lo = cfg.restricted_min(Gho)
                        err_off.append(ea)
                        lam_off.append(lo)
                        cls_off.append("A" if ea < cfg.lam_star else ("C" if lo < -ROUND * np.linalg.norm(Gho, 2) else "B"))
                    e_off = np.array(e_off)
                    phase.append(dict(config=cfg.name, kc=kc, p=p, M=M, h=h, eta_actual=mesh.eta_actual,
                                      mean=float(e_off.mean()), rms=float(np.sqrt((e_off ** 2).mean())),
                                      std=float(e_off.std()), min=float(e_off.min()), max=float(e_off.max()),
                                      lam_h_min=float(min(lam_off)), err_abs_max=float(max(err_off)),
                                      lam_star=cfg.lam_star, classes={c: cls_off.count(c) for c in "ABC"},
                                      weyl_holds=bool(all(l >= cfg.lam_star - e - 1e-13 * cfg.norm2
                                                          for l, e in zip(lam_off, err_off))),
                                      qp=R["qp"]))
                say(f"  {cfg.name} kc={kc:g} p={p}: done ({time.perf_counter() - t0:.0f}s)")
    return out, fixed, phase, log, say


# ----------------------------------------------------------------------------
def fit_slopes(phase):
    """Slope of log(RMS) vs log(h) on the finest meshes above the roundoff floor; C_M = rms/(xi h)^p."""
    fits = []
    for key in sorted({(r["config"], r["kc"], r["p"]) for r in phase}):
        rows = sorted([r for r in phase if (r["config"], r["kc"], r["p"]) == key], key=lambda r: r["h"])
        valid = [r for r in rows if r["rms"] > 1e-12]
        use = valid[:4]                                   # finest four valid meshes
        h = np.array([r["h"] for r in use])
        e = np.array([r["rms"] for r in use])
        slope = float(np.polyfit(np.log(h), np.log(e), 1)[0]) if len(use) >= 3 else np.nan
        hh = np.array([r["h"] for r in rows])
        ee = np.array([r["rms"] for r in rows])
        local = np.diff(np.log(ee)) / np.diff(np.log(hh))
        p = key[2]
        CM = ee / (XI * hh) ** p
        cm_fine = CM[[rows.index(r) for r in valid[:3]]] if len(valid) >= 3 else CM[:1]
        # asymptotic range: meshes where the local slope is within 20% of p (and above floor)
        in_asym = [rows[i + 1]["h"] for i in range(len(local)) if abs(local[i] - p) <= 0.2 * p
                   and rows[i]["rms"] > 1e-12]
        fits.append(dict(config=key[0], kc=key[1], p=p, slope=slope, n_fit=len(use),
                         h_fit=[float(v) for v in h], CM_asym=float(np.mean(cm_fine)),
                         CM_spread=float(cm_fine.max() / cm_fine.min()),
                         CM_all=[float(v) for v in CM], h_all=[float(v) for v in hh],
                         local_slopes=[float(v) for v in local],
                         asym_h_range=[float(min(in_asym)), float(max(in_asym))] if in_asym else None))
    return fits


def compare_predictors(fits, say):
    """C_M (xi h)^p and q^p against the parameter-selection mesh runs (kc = 2 xi s, varied eta)."""
    ps_dir = HERE / "parameter_selection_results"
    rows = []
    for name in ("mesh_sweep.json", "extension.json"):
        f = ps_dir / name
        if f.exists():
            rows += json.loads(f.read_text())
    CM = {(f["config"], f["p"]): f["CM_asym"] for f in fits if f["kc"] == 6.0}
    out = []
    for r in rows:
        key = (r["config"], r["p"])
        if key not in CM or r["eps_mesh"] <= 0:
            continue
        pred_h = CM[key] * (r["xi"] * r["h"]) ** r["p"]
        pred_eta = CM[key] * (r["eta_N"] * np.pi / (2 * r["s"])) ** r["p"]
        out.append(dict(config=r["config"], p=r["p"], eta_N=r["eta_N"], s=r["s"], h=r["h"], actual=r["eps_mesh"],
                        pred_CM_h=pred_h, pred_CM_eta=pred_eta, qp=r["qp"]))
    stat = {}
    for k in ("pred_CM_h", "pred_CM_eta", "qp"):
        lr = np.log10(np.array([o[k] / o["actual"] for o in out]))
        stat[k] = dict(median_factor=float(10 ** np.median(lr)), min_factor=float(10 ** lr.min()),
                       max_factor=float(10 ** lr.max()), rms_log10=float(np.sqrt((lr ** 2).mean())))
        say(f"  predictor {k:12s}: pred/actual median {stat[k]['median_factor']:.3g}, range "
            f"{stat[k]['min_factor']:.3g} .. {stat[k]['max_factor']:.3g}, rms |log10| = {stat[k]['rms_log10']:.2f}")
    return out, stat


def plots(out, fixed, phase, fits, pred):
    set_style()
    keys = [("A", 4.0), ("A", 6.0), ("B", 4.0), ("B", 6.0)]
    ps = sorted({r["p"] for r in phase})
    col = {p: PAL[i] for i, p in enumerate(ps)}

    def four(title, ylab, fn, fname, xlog=True, ylog=True):
        fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
        for ax, (c, kc) in zip(axs.ravel(), keys):
            fn(ax, c, kc)
            ax.set(title=f"config {c}, kc = {kc:g}", xlabel="h", ylabel=ylab)
            if xlog:
                ax.set_xscale("log")
            if ylog:
                ax.set_yscale("log")
        axs[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / fname)
        plt.close(fig)

    def p1(ax, c, kc):
        for p in ps:
            rr = sorted([r for r in phase if r["config"] == c and r["kc"] == kc and r["p"] == p], key=lambda r: r["h"])
            ax.plot([r["h"] for r in rr], [r["rms"] for r in rr], "o-", ms=3, color=col[p], label=f"p = {p}")
    four("1. RMS phase-averaged mesh error vs h", "eps_mesh (RMS over offsets)", p1, "plot01_eps_vs_h.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, (c, kc) in enumerate(keys):
        ff = [f for f in fits if f["config"] == c and f["kc"] == kc]
        ax.plot([f["p"] for f in ff], [f["slope"] for f in ff], "osD^"[i], ms=7, color=PAL[i],
                label=f"config {c}, kc = {kc:g}")
    ax.plot([1.5, 8.5], [1.5, 8.5], color=INK2, lw=0.8, label="slope = p")
    ax.set(xlabel="B-spline order p", ylabel="fitted slope d log(eps)/d log(h)",
           title="2. Fitted h-slope vs p (finest valid meshes, RMS over offsets)")
    ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plot02_slope_vs_p.png"), plt.close(fig)

    def p3(ax, c, kc):
        for p in ps:
            f = next(f for f in fits if f["config"] == c and f["kc"] == kc and f["p"] == p)
            ax.plot(f["h_all"], f["CM_all"], "o-", ms=3, color=col[p], label=f"p = {p}")
    four("3. C_M(h) = eps_mesh / (xi h)^p", "C_M", p3, "plot03_CM_vs_h.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, c in enumerate(("A", "B")):
        rr = [r for r in fixed if r["config"] == c]
        ax.loglog([r["eps_mesh"] for r in rr], [r["eps_recon"] for r in rr], "os"[i], ms=4, color=PAL[i],
                  label=f"config {c}")
    lim = [1e-14, 1e-1]
    ax.plot(lim, lim, color=INK2, lw=0.8, label="y = x")
    ax.set(xlabel="full mesh error ||Gamma_L^mesh - Gamma_L^kc|| / ||Gamma|| (literal)",
           ylabel="reconstructed D_h + alpha terms", title="4. Full vs reconstructed mesh error")
    ax.text(0.03, 0.9, f"max |identity deviation| / max|K_F| = {max(r['recon_identity_err_relK'] for r in fixed):.1e}",
            transform=ax.transAxes, color=INK2, fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(), fig.savefig(out / "plot04_full_vs_reconstructed.png"), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, c in enumerate(("A", "B")):
        rr = [r for r in fixed if r["config"] == c]
        ax.loglog([r["eps_alpha"] for r in rr], [max(r["eps_D"], 1e-18) for r in rr], "os"[i], ms=4, color=PAL[i],
                  label=f"config {c}")
    ax.set(xlabel="alpha contribution (alpha_i^* + alpha_j + alpha_i^* alpha_j)", ylabel="D_h contribution",
           title="5. D_h contribution vs alpha contribution (operator norm / ||Gamma||)")
    ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plot05_D_vs_alpha.png"), plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, p in zip(axs, (2, 4, 6)):
        rr = sorted([r for r in fixed if r["config"] == "B" and r["kc"] == 6.0 and r["p"] == p], key=lambda r: r["h"])
        hh = [r["h"] for r in rr]
        for k, lab, cc, st in (("eps_mesh", "total K_h - K_F", PAL[0], "o-"), ("old_transfer", "old 'transfer'", PAL[1], "s--"),
                               ("eps_lin", "new: linear alpha_i^* + alpha_j", PAL[2], "^-"),
                               ("old_alias", "old 'alias' (= quad, n = n')", PAL[3], "D--"),
                               ("eps_quad", "new: quadratic alpha_i^* alpha_j", PAL[4], "v-"),
                               ("quad_offdiag", "new: quadratic, n != n'", PAL[5], "x:")):
            ax.loglog(hh, [r[k] for r in rr], st, ms=4, color=cc, label=lab)
        ax.set(xlabel="h", ylabel="operator norm / ||Gamma||", title=f"6. config B, kc = 6, p = {p}")
    axs[0].legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "plot06_old_vs_new_diagnostics.png"), plt.close(fig)

    def p7(ax, c, kc):
        for p in (2, 4, 6, 8):
            rr = sorted([r for r in phase if r["config"] == c and r["kc"] == kc and r["p"] == p], key=lambda r: r["h"])
            hh = [r["h"] for r in rr]
            ax.fill_between(hh, [r["min"] for r in rr], [r["max"] for r in rr], color=col[p], alpha=0.18, lw=0)
            ax.plot(hh, [r["rms"] for r in rr], "-", color=col[p], label=f"p = {p}: RMS, band = min..max")
            fx = sorted([r for r in fixed if r["config"] == c and r["kc"] == kc and r["p"] == p], key=lambda r: r["h"])
            ax.plot([r["h"] for r in fx], [r["eps_mesh"] for r in fx], "x", color=col[p], ms=5)
    four("7. Phase variation over random global mesh offsets (x = fixed configuration)", "eps_mesh", p7,
         "plot07_phase_variation.png")

    for k, fname, lab in (("pred_CM_h", "plot08_actual_vs_CM.png", "C_M (xi h)^p"),
                          ("qp", "plot09_actual_vs_qp.png", "q^p, q = eta_N/(2 - eta_N)")):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for i, c in enumerate(("A", "B")):
            rr = [o for o in pred if o["config"] == c]
            ax.loglog([o[k] for o in rr], [o["actual"] for o in rr], "os"[i], ms=4, color=PAL[i],
                      label=f"parameter-selection runs, config {c}")
        lim = [1e-11, 1.0]
        ax.plot(lim, lim, color=INK2, lw=0.8, label="y = x")
        ax.set(xlabel=f"predicted: {lab}", ylabel="actual mesh error", title=f"{fname[4:6]}. Actual mesh error vs {lab}")
        ax.legend(fontsize=8)
        fig.tight_layout(), fig.savefig(out / fname), plt.close(fig)

    def p10(ax, c, kc):
        for p in (2, 4, 6, 8):
            rr = sorted([r for r in fixed if r["config"] == c and r["kc"] == kc and r["p"] == p], key=lambda r: r["h"])
            hh = [r["h"] for r in rr]
            ax.plot(hh, [r["lam_h"] for r in rr], "o-", ms=3, color=col[p], label=f"p = {p}: lambda_h")
            ax.plot(hh, [r["weyl_bound"] for r in rr], "--", color=col[p], lw=1.0, label=f"p = {p}: lambda_* - ||dGamma||")
        ax.axhline(rr[0]["lam_star"], color=INK2, lw=0.8)
    four("10. lambda_h and the perturbation lower bound vs h (line: lambda_*)", "eigenvalue on T-perp", p10,
         "plot10_lambda_vs_h.png", ylog=False)


# ----------------------------------------------------------------------------
def report(fixed, phase, fits, stat):
    L = ["# Mesh error decomposition: summary", ""]
    L.append(f"Tests: {len(fixed)} fixed-configuration meshes, {len(phase)} phase-averaged meshes "
             f"({phase[0]['classes'] and sum(phase[0]['classes'].values())} offsets each).")
    L.append(f"- literal spread/FFT/G/IFFT/gather vs mode-space matrix: max {max(r['literal_vs_modespace'] for r in fixed):.1e}")
    L.append(f"- identity K_h - K_F = D_h + H(alpha_i^* + alpha_j + alpha_i^* alpha_j): max |deviation| / max|K_F| = "
             f"{max(r['recon_identity_err_relK'] for r in fixed):.1e} (roundoff level); relative to the error itself "
             f"{min(r['recon_identity_err'] for r in fixed):.1e}..{max(r['recon_identity_err'] for r in fixed):.1e} "
             "(largest only where the mesh error itself approaches roundoff)")
    L.append(f"- truncated Poisson sum (|n| <= 1000) vs exact stencil DFT, relative to the alias part: "
             f"max {max(r['poisson_vs_stencil'] for r in fixed):.1e} (p=2 converges slowest); relative to |S|: "
             f"max {max(r['poisson_vs_stencil_relS'] for r in fixed):.1e}")
    L.append(f"- max |D_h|/|Khat_L| = {max(r['D_rel_max'] for r in fixed):.1e}; operator contribution of D_h "
             f"max {max(r['eps_D'] for r in fixed):.1e}; excluded influence modes: {sum(r['n_excluded'] for r in fixed)}")
    rat = [r["eps_lin"] / r["eps_mesh"] for r in fixed]
    L.append(f"- linear part / total: {min(rat):.3f}..{max(rat):.3f}; quadratic / total: "
             f"{min(r['eps_quad'] / r['eps_mesh'] for r in fixed):.1e}..{max(r['eps_quad'] / r['eps_mesh'] for r in fixed):.1e}")
    L.append(f"- old 'transfer' = linear + quadratic(n != n'): max deviation "
             f"{max(r['old_transfer_vs_lin_plus_quadoff'] for r in fixed):.1e}; old 'alias' = quadratic(n = n') part")
    L.append("")
    L.append("| config | kc | p | slope (finest 4) | h fitted | asymptotic C_M | C_M spread (finest 3) | h range with local slope within 20% of p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for f in fits:
        L.append(f"| {f['config']} | {f['kc']:g} | {f['p']} | {f['slope']:.2f} | {min(f['h_fit']):.3f}-{max(f['h_fit']):.3f} | "
                 f"{f['CM_asym']:.3g} | {f['CM_spread']:.2f} | "
                 f"{'%.3f-%.3f' % tuple(f['asym_h_range']) if f['asym_h_range'] else 'none'} |")
    L.append("")
    L.append("Predictors vs the 181 parameter-selection mesh runs (pred/actual):")
    for k, v in stat.items():
        L.append(f"- {k}: median {v['median_factor']:.3g}, range {v['min_factor']:.3g}..{v['max_factor']:.3g}, "
                 f"rms |log10| {v['rms_log10']:.2f}")
    cls = {}
    for r in fixed:
        cls[r["psd_class"]] = cls.get(r["psd_class"], 0) + 1
    for r in phase:
        for c, n in r["classes"].items():
            cls[c] = cls.get(c, 0) + n
    L.append("")
    L.append(f"PSD classes over all fixed + offset meshes: {cls}; Weyl bound held: "
             f"{sum(r['weyl_holds'] for r in fixed)}/{len(fixed)} fixed, {sum(r['weyl_holds'] for r in phase)}/{len(phase)} offset sets")
    L.append(f"max symmetry residual {max(r['sym_resid'] for r in fixed):.1e}, max translation residual "
             f"{max(r['trans_resid'] for r in fixed):.1e}, min lambda_h/lambda_* {min(r['lam_h'] / r['lam_star'] for r in fixed):.4f}")
    sh = np.array([(r["lam_star"] - r["lam_h"]) / r["err_abs"] for r in fixed])
    L.append(f"actual gap shift / ||Gamma_h - Gamma||_2: median {np.median(np.abs(sh)):.1e}, max {np.abs(sh).max():.2f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--offsets", type=int, default=64)
    ap.add_argument("--nmax", type=int, default=1000)
    args = ap.parse_args()
    out, fixed, phase, log, say = run(args)
    fits = fit_slopes(phase)
    for f in fits:
        say(f"  fit {f['config']} kc={f['kc']:g} p={f['p']}: slope {f['slope']:.2f}, C_M {f['CM_asym']:.3g} "
            f"(spread {f['CM_spread']:.2f}), local slopes {np.round(f['local_slopes'], 2).tolist()}")
    pred, stat = compare_predictors(fits, say)
    plots(out, fixed, phase, fits, pred)
    rep = report(fixed, phase, fits, stat)
    say("\n" + rep)
    for name, rows in (("fixed_config", fixed), ("phase_averaged", phase), ("fits", fits), ("predictors", pred)):
        (out / f"{name}.json").write_text(json.dumps(rows, indent=1) + "\n")
        ks = [k for k in rows[0] if not isinstance(rows[0][k], (list, dict))]
        (out / f"{name}.csv").write_text(",".join(ks) + "\n" + "\n".join(",".join(str(r[k]) for k in ks) for r in rows) + "\n")
    (out / "predictor_stats.json").write_text(json.dumps(stat, indent=1) + "\n")
    (out / "summary.md").write_text(rep + "\n")
    (out / "run_log.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
