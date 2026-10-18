#!/usr/bin/env python3
"""
Validity of the mode-dependent Hurwitz-zeta alias bound for the EXISTING B-spline implementation
(test_parameter_selection.bspline / Mesh.stencil / Mesh.stencil_dft / Mesh influence function).

Claim. For one axis, with h = L/M, particle coordinate u = q/h, mode k = 2 pi m/L and x = k h/(2 pi) = m/M (|x| < 1/2),
the stencil DFT of the code, s(k) = sum_t w_t exp(-i k g_t h), satisfies

    s(k) = W_hat(k) e^{-ikq} (1 + beta(k)),   W_hat(k) = sinc^p(k h/2),
    beta(k) = sum_{m' != 0} e^{-i 2 pi m' u} [W_hat(k + 2 pi m'/h)/W_hat(k)],
    |beta(k)| <= sigma_p(x) = sum_{m' != 0} |x/(x + m')|^p = |x|^p [zeta(p, 1 + |x|) + zeta(p, 1 - |x|)]   (Hurwitz zeta),

and, because the 3D stencil is a tensor product, |alpha_j(k)| = |prod_a (1 + beta_a) - 1| <= prod_a (1 + sigma_p(x_a)) - 1.
This requires (i) Fourier transform of bspline = sinc^p, (ii) the p-point stencil covers the whole support, (iii) the
Poisson representation with the code's phase convention, (iv) the influence function divides by exactly W_hat^2.

Checks (all numerical, written to results_joint/alias_bound_validation.json):
  1. FT of the code's bspline vs sinc^p (dense quadrature), p = 2..8.
  2. Stencil coverage: partition of unity and zero B-spline value just outside the stencil.
  3. Poisson representation: Mesh.stencil_dft vs the explicitly summed Poisson series (|m'| <= 2000).
  4. 1D bound |beta| <= sigma_p(x) for every particle of every configuration and every retained mode, several M;
     tightness sup_u |beta|/sigma_p over a fine u grid.
  5. 3D bound |alpha_j(k)| <= prod(1 + sigma_p) - 1 from the actual 3D stencil DFT; comparison with C(p) max q^p.
  6. Hurwitz zeta (scipy.special.zeta(p, q)) vs direct summation; sigma_p <= 2 zeta(p) (x/(1-x))^p.
  7. Influence function: Mesh.W2 equals W_hat^2 (deconvolution residual D_h = 0 up to round-off).
  8. Operator level: measured ||Gamma_L^mesh - Gamma_L^kc||_2 <= both a priori mesh bounds, coarse to fine meshes.

    python3 test_alias_bound.py
"""
import time

import numpy as np
from scipy import special

import common as C
from clustered_configurations import load
from cached_operator import mesh_for
from joint_optimizer import sigma_hz, alpha_hz, alpha_orig, mesh_bound_one, RESULTS_JOINT
from parameter_optimizer import FourierShells, mesh_modes, C_p
from test_parameter_selection import bspline, lap_full
from error_validation import Components


def check_ft(rows):
    out = {}
    y = np.linspace(-4.5, 4.5, 360001)
    dy = y[1] - y[0]
    w = np.linspace(-3 * np.pi, 3 * np.pi, 61)
    for p in range(2, 9):
        B = bspline(y, p)
        ft = (B[None, :] * np.exp(-1j * w[:, None] * y[None, :])).sum(1) * dy
        ref = np.sinc(w / (2 * np.pi)) ** p
        out[p] = float(np.max(np.abs(ft - ref)))
    rows["1_bspline_ft_vs_sinc_p_maxabs"] = out
    return max(out.values()) < 1e-8


def check_stencil(rows, rng):
    worst_pu, worst_out = 0.0, 0.0
    for p in range(2, 9):
        u = np.concatenate([rng.uniform(0, 50, 20000), np.arange(0, 50, 0.5)])       # incl. integer/half-integer u
        g = np.floor(u - p / 2)[:, None] + 1 + np.arange(p)
        wt = bspline(g - u[:, None], p)
        worst_pu = max(worst_pu, float(np.max(np.abs(wt.sum(1) - 1))))
        out_lo = bspline(g[:, 0] - 1 - u, p)
        out_hi = bspline(g[:, -1] + 1 - u, p)
        worst_out = max(worst_out, float(np.max(np.abs(out_lo))), float(np.max(np.abs(out_hi))))
    rows["2_partition_of_unity_maxdev"] = worst_pu
    rows["2_bspline_outside_stencil_max"] = worst_out
    return worst_pu < 1e-12 and worst_out == 0.0


def poisson_series(u, x, p, mmax=2000):
    """1 + beta from the explicit Poisson series (per axis), u = q/h, x = m/M."""
    mp = np.arange(-mmax, mmax + 1)
    mp = mp[mp != 0]
    ratio = (np.sinc(x[None, :, None] + mp[None, None, :]) / np.sinc(x[None, :, None])) ** p
    return 1 + (np.exp(-2j * np.pi * mp[None, None, :] * u[:, None, None]) * ratio).sum(-1)


def check_1d(rows, configs, p=7):
    res = []
    ok = True
    for name in configs:
        x, L = load(name)
        x = np.mod(x, L)
        K = C.kernel(0.85)
        for M in (12, 16, 24, 32, 48, 64):
            mc = (M - 1) // 2
            kc = 2 * np.pi * mc / L
            mesh = mesh_for(L, K, kc, M, p)
            s = mesh.stencil_dft(x)                                     # (N, 3, 2 mc + 1)
            m = np.arange(-mesh.mc, mesh.mc + 1)
            k1 = 2 * np.pi / L * m
            Wh = np.sinc(k1 * mesh.h / (2 * np.pi)) ** p
            onepb = s / (Wh[None, None, :] * np.exp(-1j * x[:, :, None] * k1[None, None, :]))
            beta = np.abs(onepb - 1)
            sig = sigma_hz(m / M, p)
            nz = sig > 0
            ratio = beta[..., nz] / sig[None, None, nz]
            zero_mode = float(np.max(beta[..., ~nz]))                   # m = 0: s = sum w = 1 exactly
            pser = poisson_series(x[:8, 0] / mesh.h, m / M, p)          # explicit Poisson series, 8 particles
            prep = float(np.max(np.abs(pser - onepb[:8, 0, :])))
            res.append(dict(config=name, M=M, mc=int(mesh.mc), max_beta_over_sigma=float(ratio.max()),
                            median_beta_over_sigma=float(np.median(ratio)), beta_at_m0=zero_mode,
                            poisson_vs_stencil_dft=prep))
            ok &= ratio.max() <= 1 + 1e-12 and prep < 1e-10
    # tightness: sup over u of |beta| / sigma for a fine u grid
    tight = {}
    u = np.linspace(0, 1, 2001)
    for M in (16, 32, 64):
        xs = np.arange(1, (M - 1) // 2 + 1) / M
        pb = poisson_series(u, xs, p, mmax=400)
        tight[M] = float(np.max(np.abs(pb - 1).max(0) / sigma_hz(xs, p)))
    rows["4_one_axis"] = res
    rows["4_sup_u_beta_over_sigma"] = tight
    return ok


def check_3d(rows, configs, p=7):
    res = []
    ok = True
    for name in configs:
        x, L = load(name)
        x = np.mod(x, L)
        K = C.kernel(0.85)
        for M in (16, 24, 48):
            kc = 2 * np.pi * ((M - 1) // 2) / L * 0.999
            mesh = mesh_for(L, K, kc, M, p)
            s = mesh.stencil_dft(x)
            o = mesh.mc
            mv = mesh.mv
            S = s[:, 0, mv[:, 0] + o] * s[:, 1, mv[:, 1] + o] * s[:, 2, mv[:, 2] + o]
            kv = mesh.kv
            What = np.prod(np.sinc(kv * mesh.h / (2 * np.pi)) ** p, axis=1)
            alpha = np.abs(S / (What[None, :] * np.exp(-1j * x @ kv.T)) - 1)
            aHZ = alpha_hz(mv, M, p)
            aOR = alpha_orig(mv, M, p)
            nz = aHZ > 0
            res.append(dict(config=name, M=M, n_modes=int(len(mv)),
                            max_alpha_over_HZ=float((alpha[:, nz] / aHZ[nz]).max()),
                            max_alpha_over_orig=float((alpha[:, nz] / aOR[nz]).max()),
                            HZ_over_orig_min=float((aHZ[nz] / aOR[nz]).min()),
                            HZ_over_orig_median=float(np.median(aHZ[nz] / aOR[nz])),
                            HZ_over_orig_max=float((aHZ[nz] / aOR[nz]).max()),
                            W2_vs_What2=float(np.max(np.abs(mesh.W2 - What ** 2) / What ** 2))))
            ok &= res[-1]["max_alpha_over_HZ"] <= 1 + 1e-12 and res[-1]["HZ_over_orig_max"] <= 1 + 1e-12 \
                and res[-1]["W2_vs_What2"] < 1e-12
    rows["5_three_d"] = res
    return ok


def check_zeta(rows):
    out = []
    ok = True
    for p in (3, 5, 7, 8):
        for q in (0.5, 0.75, 1.0, 1.25, 1.5):
            n = np.arange(0, 200000)
            direct = np.sum((n + q) ** -float(p)) + (200000 + q) ** (1 - p) / (p - 1) - 0.5 * (200000 + q) ** -float(p)
            sc = special.zeta(p, q)
            out.append(dict(p=p, q=q, rel=float(abs(sc - direct) / direct)))
            ok &= abs(sc - direct) / direct < 1e-12
    x = np.linspace(1e-6, 0.5 - 1e-9, 20001)
    worst = {}
    for p in range(2, 9):
        old = 2 * special.zeta(p, 1) * (x / (1 - x)) ** p
        r = sigma_hz(x, p) / old
        worst[p] = dict(max_ratio_new_over_old=float(r.max()), ratio_at_x_0p5=float(r[-1]),
                        ratio_at_x_0p25=float(np.interp(0.25, x, r)))
        ok &= r.max() <= 1 + 1e-12
    rows["6_hurwitz_zeta_vs_direct"] = out
    rows["6_sigma_over_2zeta_q_p"] = worst
    return ok


def check_operator(rows, p=7):
    """Measured mesh error vs the two a priori bounds (absolute, spectral norm)."""
    res = []
    ok = True
    for name, xi in (("strong_N64_s1", 0.85), ("strong_N256_s1", 0.85), ("uniform_N256_s1", 1.0)):
        x, L = load(name)
        x = np.mod(x, L)
        N = len(x)
        G = np.load(C.RAW / f"{name}_Gref.npy")
        cp = Components(x, L, G, xi)
        fs = FourierShells(xi, L)
        kc = fs.kc_tol(1e-9 * 15, N / L ** 3)
        _, GF = cp.fourier_error(kc)
        mc = int(np.floor(kc * L / (2 * np.pi) + 1e-12))
        for M in sorted({m for m in (2 * mc + 1, 2 * mc + 2, 24, 30, 36, 48, 64, 90) if m > 2 * mc and kc * (L / m) / np.pi < 1}):
            meas, _ = cp.mesh_error(kc, M, p, GF)
            bo = mesh_bound_one(fs, kc, M, p, N / L ** 3, "orig")
            bh = mesh_bound_one(fs, kc, M, p, N / L ** 3, "hz")
            res.append(dict(config=name, xi=xi, kc=kc, M=M, eta=kc * (L / M) / np.pi, measured=meas, bound_orig=bo,
                            bound_hz=bh, orig_over_measured=bo / meas, hz_over_measured=bh / meas, orig_over_hz=bo / bh))
            ok &= meas <= bh <= bo
    rows["8_operator_level"] = res
    return ok


def main():
    C.RESULTS = RESULTS_JOINT              # this round's outputs and command log go to results_joint/
    C.log_command()
    t0 = time.perf_counter()
    rng = np.random.default_rng([20261018, 1])
    rows = {}
    cfgs1 = ["uniform_N64_s1", "strong_N64_s1", "strong_N256_s1", "moderate_N512_s1", "lj_equilibrated_N256_s305"]
    passes = {}
    passes["1_ft"] = check_ft(rows)
    passes["2_stencil"] = check_stencil(rows, rng)
    passes["3_4_poisson_and_1d_bound"] = check_1d(rows, cfgs1)
    passes["5_3d_bound"] = check_3d(rows, ["strong_N64_s1", "uniform_N256_s1", "strong_N256_s1"])
    passes["6_zeta"] = check_zeta(rows)
    passes["8_operator"] = check_operator(rows)
    rows["7_influence_W2_equals_What2"] = max(r["W2_vs_What2"] for r in rows["5_three_d"])
    summary = dict(passes=passes, all_pass=all(passes.values()), C_7=C_p(7),
                   max_beta_over_sigma=max(r["max_beta_over_sigma"] for r in rows["4_one_axis"]),
                   max_alpha_over_HZ=max(r["max_alpha_over_HZ"] for r in rows["5_three_d"]),
                   HZ_over_orig_range=[min(r["HZ_over_orig_min"] for r in rows["5_three_d"]),
                                       max(r["HZ_over_orig_max"] for r in rows["5_three_d"])],
                   wall_s=time.perf_counter() - t0)
    C.write_json(RESULTS_JOINT / "alias_bound_validation.json", dict(summary=summary, checks=rows, provenance=C.provenance()))
    C.log(f"alias bound validation: {summary}")
    for r in rows["8_operator_level"]:
        C.log(f"  {r['config']} M={r['M']} eta={r['eta']:.3f}: measured {r['measured']:.2e}  HZ {r['bound_hz']:.2e} "
              f"(x{r['hz_over_measured']:.0f})  orig {r['bound_orig']:.2e} (x{r['orig_over_measured']:.0f})")
    if not summary["all_pass"]:
        raise SystemExit("alias bound validation FAILED")


if __name__ == "__main__":
    main()
