#!/usr/bin/env python3
"""
Hurwitz-zeta alias bound of the existing B-spline PPPM for every assignment order of the p search (p = 4..11; the
search set is 5..10, 4 and 11 are the neighbouring orders checked when the optimum touches a boundary).

The bound |beta(k)| <= sigma_p(x) is the triangle inequality applied to the Poisson series of the stencil DFT, so it
holds in exact arithmetic for every p. In floating point the computed beta = |s/(W_hat e^{-ikq}) - 1| carries an
absolute round-off error of a few ulps (and is ~4e-16 instead of 0 at m = 0). For p >= 8 sigma_p drops below 1e-15 on
coarse modes, so the RELATIVE test |beta|/sigma <= 1 + 1e-12 of test_alias_bound.py (round 2, p = 7) fails there for a
purely numerical reason. The criteria used here, reported separately:
  (a) absolute: max(beta - sigma_p) <= DELTA_1D on every mode (and max(alpha - alpha_HZ) <= DELTA_3D in 3D), with the
      measured excess written out; DELTA is a round-off allowance (~100 ulp), not part of the certificate;
  (b) relative: beta/sigma_p <= 1 + 1e-9 on the modes with sigma_p >= SIG_REL, where a few-ulp error cannot matter.
Checks:
  1. FT of the code's bspline vs sinc^p, partition of unity and stencil coverage, p = 4..12;
  2. Hurwitz zeta vs direct summation and sigma_p <= 2 zeta(p) (x/(1-x))^p, p = 4..12;
  3. per axis |beta| <= sigma_p (stencil DFT of every particle, every retained mode, M = 12..64) and the explicit
     Poisson series vs Mesh.stencil_dft; sup_u |beta|/sigma_p on a fine u grid (tightness);
  4. 3D |alpha_j(k)| <= prod(1 + sigma_p) - 1 from the actual 3D stencil DFT, and Mesh.W2 = W_hat^2;
  5. operator level at N = 256 and N = 512: measured ||Gamma_L^mesh - Gamma_L^kc||_2 <= HZ bound on meshes around the
     search region; a measured value below the round-off floor 1e-14 ||Gamma_ref||_2 is flagged.
  6. effect of the measured round-off excess on eps_mesh: the bound re-evaluated with alpha_HZ + max excess.
Outputs: results_psearch/alias_bound_validation_p.json (nothing in results_joint/ is touched).

    python3 test_alias_bound_p.py
"""
import time

import common as C
import numpy as np
from scipy import special

from clustered_configurations import load
from cached_operator import mesh_for
from joint_optimizer import sigma_hz, alpha_hz, mesh_bound_one
from parameter_optimizer import FourierShells, mesh_modes
from test_parameter_selection import bspline
from error_validation import Components
from test_alias_bound import poisson_series

RESULTS_PSEARCH = C.HERE / "results_psearch"
P_ALL = tuple(range(4, 12))
ROUNDOFF = 1e-14
DELTA_1D = 2e-14
DELTA_3D = 5e-14
SIG_REL = 1e-6


def check_ft_p(rows, ps):
    out = {}
    y = np.linspace(-6.5, 6.5, 520001)
    dy = y[1] - y[0]
    w = np.linspace(-3 * np.pi, 3 * np.pi, 61)
    for p in ps:
        B = bspline(y, p)
        ft = (B[None, :] * np.exp(-1j * w[:, None] * y[None, :])).sum(1) * dy
        out[p] = float(np.max(np.abs(ft - np.sinc(w / (2 * np.pi)) ** p)))
    rows["1_bspline_ft_vs_sinc_p_maxabs"] = out
    return max(out.values()) < 1e-8


def check_stencil_p(rows, ps, rng):
    pu, outside = {}, {}
    for p in ps:
        u = np.concatenate([rng.uniform(0, 50, 20000), np.arange(0, 50, 0.5)])
        g = np.floor(u - p / 2)[:, None] + 1 + np.arange(p)
        wt = bspline(g - u[:, None], p)
        pu[p] = float(np.max(np.abs(wt.sum(1) - 1)))
        outside[p] = float(max(np.abs(bspline(g[:, 0] - 1 - u, p)).max(), np.abs(bspline(g[:, -1] + 1 - u, p)).max()))
    rows["2_partition_of_unity_maxdev"] = pu
    rows["2_bspline_outside_stencil_max"] = outside
    return max(pu.values()) < 1e-12 and max(outside.values()) == 0.0


def check_zeta_p(rows, ps):
    out, worst, ok = [], {}, True
    n = np.arange(0, 200000)
    for p in ps:
        for q in (0.5, 0.75, 1.0, 1.25, 1.5):
            direct = np.sum((n + q) ** -float(p)) + (200000 + q) ** (1 - p) / (p - 1) - 0.5 * (200000 + q) ** -float(p)
            rel = float(abs(special.zeta(p, q) - direct) / direct)
            out.append(dict(p=p, q=q, rel=rel))
            ok &= rel < 1e-12
        x = np.linspace(1e-6, 0.5 - 1e-9, 20001)
        r = sigma_hz(x, p) / (2 * special.zeta(p, 1) * (x / (1 - x)) ** p)
        worst[p] = dict(max_ratio_new_over_old=float(r.max()), ratio_at_x_0p5=float(r[-1]),
                        ratio_at_x_0p25=float(np.interp(0.25, x, r)))
        ok &= r.max() <= 1 + 1e-12
    rows["6_hurwitz_zeta_vs_direct"] = out
    rows["6_sigma_over_2zeta_q_p"] = worst
    return ok


def check_1d_p(configs, p):
    res, ok = [], True
    for name in configs:
        x, L = load(name)
        x = np.mod(x, L)
        K = C.kernel(0.85)
        for M in (12, 16, 24, 32, 48, 64):
            mc = (M - 1) // 2
            mesh = mesh_for(L, K, 2 * np.pi * mc / L, M, p)
            s = mesh.stencil_dft(x)
            m = np.arange(-mesh.mc, mesh.mc + 1)
            k1 = 2 * np.pi / L * m
            Wh = np.sinc(k1 * mesh.h / (2 * np.pi)) ** p
            onepb = s / (Wh[None, None, :] * np.exp(-1j * x[:, :, None] * k1[None, None, :]))
            beta = np.abs(onepb - 1)
            sig = sigma_hz(m / M, p)
            excess = float(np.max(beta - sig[None, None, :]))
            rel = sig >= SIG_REL
            ratio_rel = float((beta[..., rel] / sig[rel]).max()) if rel.any() else None
            nz = sig > 0
            pser = poisson_series(x[:8, 0] / mesh.h, m / M, p)
            prep = float(np.max(np.abs(pser - onepb[:8, 0, :])))
            res.append(dict(config=name, M=M, mc=int(mesh.mc), max_excess_abs=excess,
                            max_ratio_sigma_ge_SIG_REL=ratio_rel,
                            max_ratio_all_modes=float((beta[..., nz] / sig[nz]).max()),
                            min_sigma_nonzero=float(sig[nz].min()), beta_at_m0=float(beta[..., ~nz].max()),
                            poisson_vs_stencil_dft=prep))
            ok &= excess <= DELTA_1D and (ratio_rel is None or ratio_rel <= 1 + 1e-9) and prep < 1e-10
    tight = {}
    u = np.linspace(0, 1, 2001)
    for M in (16, 32, 64):
        xs = np.arange(1, (M - 1) // 2 + 1) / M
        sg = sigma_hz(xs, p)
        keep = sg >= SIG_REL
        if keep.any():
            pb = poisson_series(u, xs[keep], p, mmax=400)
            tight[M] = float(np.max(np.abs(pb - 1).max(0) / sg[keep]))
    return res, tight, bool(ok)


def check_3d_p(configs, p):
    res, ok = [], True
    for name in configs:
        x, L = load(name)
        x = np.mod(x, L)
        K = C.kernel(0.85)
        for M in (16, 24, 48):
            mesh = mesh_for(L, K, 2 * np.pi * ((M - 1) // 2) / L * 0.999, M, p)
            s = mesh.stencil_dft(x)
            o, mv, kv = mesh.mc, mesh.mv, mesh.kv
            S = s[:, 0, mv[:, 0] + o] * s[:, 1, mv[:, 1] + o] * s[:, 2, mv[:, 2] + o]
            What = np.prod(np.sinc(kv * mesh.h / (2 * np.pi)) ** p, axis=1)
            alpha = np.abs(S / (What[None, :] * np.exp(-1j * x @ kv.T)) - 1)
            aHZ = alpha_hz(mv, M, p)
            excess = float(np.max(alpha - aHZ[None, :]))
            rel = aHZ >= SIG_REL
            ratio_rel = float((alpha[:, rel] / aHZ[rel]).max()) if rel.any() else None
            res.append(dict(config=name, M=M, n_modes=int(len(mv)), max_excess_abs=excess,
                            max_ratio_alphaHZ_ge_SIG_REL=ratio_rel,
                            W2_vs_What2=float(np.max(np.abs(mesh.W2 - What ** 2) / What ** 2))))
            ok &= excess <= DELTA_3D and (ratio_rel is None or ratio_rel <= 1 + 1e-9) and res[-1]["W2_vs_What2"] < 1e-12
            del S, alpha
    return res, bool(ok)


def check_operator_p(rows, ps):
    """Measured mesh error vs the HZ bound, absolute spectral norm; one Components object per configuration."""
    cases = (("strong_N256_s1", 0.85, (24, 30, 36, 48)), ("strong_N512_s1", 0.75, (30, 36, 45, 60)),
             ("uniform_N512_s1", 0.75, (30, 36, 45, 60)))
    res, ok = [], True
    for name, xi, Ms in cases:
        x, L = load(name)
        x = np.mod(x, L)
        N = len(x)
        G = np.load(C.RAW / f"{name}_Gref.npy")
        cp = Components(x, L, G, xi)
        fs = FourierShells(xi, L)
        kc = fs.kc_tol(1.5e-9 * cp.norm_ref, N / L ** 3)
        _, GF = cp.fourier_error(kc)
        mc = int(np.floor(kc * L / (2 * np.pi) + 1e-12))
        for p in ps:
            for M in Ms:
                if M <= 2 * mc or kc * (L / M) / np.pi >= 1:
                    continue
                t0 = time.perf_counter()
                meas, n_excl = cp.mesh_error(kc, M, p, GF)
                bh = mesh_bound_one(fs, kc, M, p, N / L ** 3, "hz")
                floor = ROUNDOFF * cp.norm_ref
                below = bool(meas <= floor)
                passes = bool(meas <= bh) or below
                res.append(dict(config=name, xi=xi, kc=kc, M=M, p=p, eta=kc * (L / M) / np.pi, measured_abs=meas,
                                measured_rel=meas / cp.norm_ref, bound_hz_abs=bh, hz_over_measured=bh / meas,
                                roundoff_floor_abs=floor, measured_below_roundoff_floor=below,
                                bound_below_roundoff_floor=bool(bh <= floor), n_excluded=int(n_excl), passes=passes))
                ok &= passes
                C.log(f"  op {name} p={p} M={M} eta={res[-1]['eta']:.3f}: measured {meas:.2e} HZ {bh:.2e} "
                      f"(x{bh / meas:.0f}){' [round-off floor]' if below else ''} ({time.perf_counter() - t0:.1f} s)")
        del cp, GF
    rows["8_operator_level"] = res
    return ok


def roundoff_effect(rows, ps, delta):
    """eps_mesh with alpha_HZ + delta (the measured floating-point excess) vs eps_mesh, at representative meshes."""
    out = []
    lam = C.read_json(C.HERE / "results" / "references.json")["configs"]
    for name, xi in (("strong_N512_s1", 0.75), ("uniform_N512_s1", 0.75)):
        x, L = load(name)
        N = len(x)
        rho = N / L ** 3
        fs = FourierShells(xi, L)
        for p in ps:
            for kc_m, M in ((6, 30), (8, 40), (10, 48)):
                kc = 2 * np.pi / L * kc_m
                mv, n2, mc = mesh_modes(kc, L)
                if mc >= M / 2:
                    continue
                a = alpha_hz(mv, M, p)
                kn = fs.knorm_of_n[n2]
                e0 = 2 * rho * float(np.sum(a * (2 + a) * kn))
                ad = a + delta
                e1 = 2 * rho * float(np.sum(ad * (2 + ad) * kn))
                out.append(dict(config=name, xi=xi, p=p, kc=kc, M=M, eps_mesh_abs=e0, eps_mesh_with_excess_abs=e1,
                                increase_abs=e1 - e0, increase_rel_to_budget=(e1 - e0) / (C.EPS_TARGET * lam[name]["lam_lb"])))
    rows["9_roundoff_excess_effect"] = out
    return max(r["increase_rel_to_budget"] for r in out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, nargs="+", default=list(P_ALL), help="orders of the 1D/3D/operator checks")
    ap.add_argument("--out-dir", default=str(RESULTS_PSEARCH))
    ap.add_argument("--out-name", default="alias_bound_validation_p.json")
    args = ap.parse_args()
    P_RUN = tuple(args.p)
    out_dir = C.Path(args.out_dir)
    C.RESULTS = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    C.log_command()
    t0 = time.perf_counter()
    rng = np.random.default_rng([20261018, 21])
    rows, passes = {}, {}
    p_ft = tuple(range(4, max(13, max(P_RUN) + 2)))
    passes["1_ft"] = check_ft_p(rows, p_ft)
    passes["2_stencil"] = check_stencil_p(rows, p_ft, rng)
    passes["6_zeta"] = check_zeta_p(rows, p_ft)
    C.log(f"ft/stencil/zeta p={p_ft[0]}..{p_ft[-1]}: {passes}")
    cfg1 = ["uniform_N64_s1", "strong_N256_s1", "strong_N512_s1", "moderate_N512_s1", "uniform_N512_s1"]
    cfg3 = ["strong_N256_s1", "strong_N512_s1", "uniform_N512_s1"]
    per_p = {}
    for p in P_RUN:
        r1, tight, ok1 = check_1d_p(cfg1, p)
        r3, ok3 = check_3d_p(cfg3, p)
        per_p[p] = dict(one_axis=r1, sup_u_beta_over_sigma=tight, three_d=r3, pass_1d=ok1, pass_3d=ok3,
                        max_excess_1d=max(x["max_excess_abs"] for x in r1),
                        max_ratio_1d_sigma_ge_SIG_REL=max(x["max_ratio_sigma_ge_SIG_REL"] or 0 for x in r1),
                        max_ratio_1d_all_modes=max(x["max_ratio_all_modes"] for x in r1),
                        max_poisson_vs_stencil_dft=max(x["poisson_vs_stencil_dft"] for x in r1),
                        max_excess_3d=max(x["max_excess_abs"] for x in r3),
                        max_ratio_3d_alphaHZ_ge_SIG_REL=max(x["max_ratio_alphaHZ_ge_SIG_REL"] or 0 for x in r3),
                        max_W2_vs_What2=max(x["W2_vs_What2"] for x in r3))
        passes[f"3_4_p{p}"] = bool(ok1 and ok3)
        q = per_p[p]
        C.log(f"alias p={p}: 1D excess {q['max_excess_1d']:.1e}, ratio(sigma>=1e-6) {q['max_ratio_1d_sigma_ge_SIG_REL']:.6f}, "
              f"ratio(all) {q['max_ratio_1d_all_modes']:.3g}; 3D excess {q['max_excess_3d']:.1e}, ratio(>=1e-6) "
              f"{q['max_ratio_3d_alphaHZ_ge_SIG_REL']:.6f}; Poisson {q['max_poisson_vs_stencil_dft']:.1e}; "
              f"W2 {q['max_W2_vs_What2']:.1e} -> {'pass' if ok1 and ok3 else 'FAIL'}")
    rows["per_p"] = per_p
    delta = max(max(v["max_excess_1d"], v["max_excess_3d"]) for v in per_p.values())
    eff = roundoff_effect(rows, P_RUN, max(delta, 0.0))
    passes["8_operator"] = check_operator_p(rows, P_RUN)
    summary = dict(passes=passes, all_pass=all(passes.values()), p_tested=list(P_RUN), DELTA_1D=DELTA_1D,
                   DELTA_3D=DELTA_3D, SIG_REL=SIG_REL, max_fp_excess_any_p=delta,
                   max_eps_mesh_increase_from_excess_rel_to_budget=eff,
                   operator_min_hz_over_measured={p: min([r["hz_over_measured"] for r in rows["8_operator_level"]
                                                          if r["p"] == p and not r["measured_below_roundoff_floor"]],
                                                         default=None) for p in P_RUN},
                   wall_s=time.perf_counter() - t0)
    C.write_json(out_dir / args.out_name, dict(summary=summary, checks=rows, provenance=C.provenance()))
    C.log(f"alias bound validation p={list(P_RUN)}: {summary}")
    if not summary["all_pass"]:
        raise SystemExit("alias bound validation FAILED")


if __name__ == "__main__":
    main()
