#!/usr/bin/env python3
"""
Experiment 1 (Model A): spatial accuracy validation of the fully periodic pure-longitudinal Yukawa Ewald-PPPM.

Model (common.py, unchanged): K(r) = c e^{-kappa r}/r rhat rhat^T, kappa = 0.7, c = gamma r_ref e^{kappa r_ref}
(gamma = 0.5, r_ref = 1.3), rho = 0.75, (Gamma X)_i = sum_{j != i} sum_n K(q_i - q_j + nL)(X_i - X_j), full periodic
lattice sum, Fourier k = 0 retained. Split at s = xi^2 (radial_kernels.RadialEwald, law A): K_S = c r r^T int_{xi^2}^inf
rho_kappa e^{-s r^2} ds, K_L = c r r^T int_0^{xi^2} rho_kappa e^{-s r^2} ds. Production operator (unchanged):
Gamma_h X = Gamma_S^{rc} X + D_L^h X - H_L^h(X) (cached_operator.CachedGammaH; pair-difference GammaH.matvec).

References (independent of the split and of the PPPM):
  Gamma_ref  prl_toy_models.LatticeFriction, rel_tol 1e-13 (near images summed directly, far images by a tensor
             Chebyshev fit), raw/<config>_Gref.npy (git-ignored; generated here for new configurations)
  Gamma_D55  unsplit direct periodic image sum to R = 55 (direct_periodic.DirectPairs; certified tail bound
             <= 7e-14 relative), used only to measure the accuracy of Gamma_ref (stage refs)

Panels (results/):
  (a) split      K = K_S + K_L: theta_S closed form, theta_L by its own Gauss-Legendre s-integral (never K - K_S),
                 theta = c e^{-kappa r}/r; all pair images |d| <= 8 of N = 256 uniform/strong/moderate x 3 seeds,
                 xi in XI_A; a synthetic radial sweep r = 1e-6 ... 12; 50-digit mpmath values of each component
  (b) shortrange E_S(rc) = ||Gamma_S^{rc} + Gamma_L^{exact} - Gamma_ref||_2/||Gamma_ref||_2, xi = 0.85,
                 s = xi rc = 2.0 ... 6.6; Gamma_L^{exact} = mesh-free Fourier lattice sum to k_c = 2 xi 6.6 (no mesh);
                 also against Gamma_D55, the exact-arithmetic tail ||Gamma_S^{rc} - Gamma_S^{full}||, and the
                 certified RealSpaceTail bound
  (c) degree     D_i^h = diagonal blocks of the complete discrete Gamma_h (mesh self response excluded), checked
                 against the production FFT degree minus the mode-space self response; E_D vs Gamma_ref (complete),
                 vs the exact truncated-Fourier degree (mesh transfer only), and the fixed-cutoff floor; p = 4, 6, 8, 9
  (d) action     Gamma_h X (production pair-difference) vs Gamma_ref X: Gaussian, translation, spectral, slow-mode
                 probes; dense E_op, symmetry, translation null space, cached vs pair-difference, lambda_min on
                 Range(Pi) of Gamma_h; N = 128, 256 x 3 families x 3 seeds, p = 9
Cutoff settings of (c), (d): "production" xi = 0.85, rc = 4.8878, kc = 9.3585 (measured N = 512 optimum, k_c snapped
to the largest reciprocal shell <= 9.3585 of each box); "strict" xi = 0.85, rc = 6.0/xi, k_c <= 2 xi 6.0 (convergence
experiment only, so the mesh error is visible below the production truncation floor).

    python3 run_exp1.py all            # or any of: configs refs split shortrange degree action fits
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common as C                       # noqa: E402  (first: thread settings before numpy)
import numpy as np                       # noqa: E402
from scipy import linalg                 # noqa: E402

import clustered_configurations as CC    # noqa: E402
from neighbor_images import kdtree_images  # noqa: E402
from parameter_optimizer import RealSpaceTail, FourierShells, lambda_lower_bound, shell_kc, mesh_modes  # noqa: E402
from cached_operator import CachedGammaH, mesh_for  # noqa: E402
from error_validation import Components, REF_REL_TOL, PAIRS6  # noqa: E402
from direct_periodic import DirectPairs  # noqa: E402
from test_parameter_selection import lap_full  # noqa: E402

RESULTS = HERE / "results"
CONFIG_DIR = HERE / "configs"
FAMS = ("uniform", "strong", "moderate")
SEEDS = (1, 2, 3)
XI_A = (0.5, 0.6, 0.75, 0.85, 1.0, 1.2, 1.5)
R_PAIRS_A = 8.0
XI0 = 0.85
S_B = np.round(np.arange(2.0, 6.6001, 0.1), 4)
B_EXACT = 6.6                           # Gamma_L^exact: k_c = 2 xi B_EXACT (Fourier tail certified below)
R_D55 = 55.0
SETTINGS = {"production": dict(xi=0.85, rc=4.8878, kc_max=9.3585),
            "strict": dict(xi=0.85, rc=6.0 / 0.85, kc_max=2 * 0.85 * 6.0)}
P_C = (4, 6, 8, 9)
P_D = 9
M_LIST = {128: (18, 20, 24, 28, 32, 36, 40, 48, 56, 64),
          256: (22, 24, 28, 32, 36, 40, 48, 56, 64, 80),
          512: (28, 32, 36, 40, 48, 56, 64, 80, 96)}
SEED = 20261019


def name_of(fam, N, seed):
    return f"{fam}_N{N}_s{seed}"


def cfg_path(name):
    for d in (CONFIG_DIR, C.HERE / "configs_scaling", C.CONFIGS):
        if (d / f"{name}.npz").exists():
            return d / f"{name}.npz"
    return None


def load_cfg(name):
    with np.load(cfg_path(name)) as d:
        return np.mod(d["q"].copy(), float(d["L"])), float(d["L"])


def gref(name, x, L):
    f = C.RAW / f"{name}_Gref.npy"
    if f.exists():
        return np.load(f)
    G, _ = C.reference(x, L, rel_tol=REF_REL_TOL)
    np.save(f, G)
    return G


def kc_shell_below(kc_max, L):
    """Largest reciprocal-lattice shell radius 2 pi sqrt(n)/L <= kc_max."""
    n = int(np.floor((kc_max * L / (2 * np.pi)) ** 2 + 1e-9))
    while n > 0:
        mv, n2, _ = mesh_modes(2 * np.pi / L * np.sqrt(n) * (1 + 1e-12), L)
        if np.any(n2 == n):
            return float(2 * np.pi / L * np.sqrt(n))
        n -= 1
    raise ValueError("no shell")


def projector_basis(N):
    T = np.kron(np.ones((N, 1)), np.eye(3)) / np.sqrt(N)
    return linalg.null_space(T.T), T


def norm2_sym(A):
    """||A||_2 for A symmetric up to round-off: max |eig(sym A)|, plus ||antisym A||_F (reported)."""
    lam = np.linalg.eigvalsh(0.5 * (A + A.T))
    return float(max(abs(lam[0]), abs(lam[-1]))), float(np.linalg.norm(0.5 * (A - A.T)))


# ============================================================================ configs and references
def needed_configs():
    out = []
    for fam in FAMS:
        for s in SEEDS:
            out += [name_of(fam, 256, s), name_of(fam, 128, s)]
    out.append(name_of("strong", 512, 1))
    return out


def stage_configs(args):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = C.read_json(RESULTS / "configurations.json", {"rows": {}})
    for name in needed_configs():
        fam, N, seed = name.split("_")[0], int(name.split("_N")[1].split("_")[0]), int(name.split("_s")[-1])
        p = cfg_path(name)
        if p is not None and name in rows["rows"]:
            continue
        if p is None:
            x, rec = CC.generate(fam, N, seed)
            hist, st = CC.structure_stats(x, rec["L"])
            np.savez_compressed(CONFIG_DIR / f"{name}.npz", q=x, L=rec["L"], labels=rec.get("labels", np.zeros(N, int)))
            src = "validation_exp1/configs (clustered_configurations.generate)"
            extra = {k: v for k, v in rec.items() if k not in ("labels",)}
        else:
            x, L = load_cfg(name)
            hist, st = CC.structure_stats(x, L)
            src = str(p.parent.relative_to(C.HERE))
            extra = dict(N=N, L=L, family=fam, seed=seed)
        rows["rows"][name] = dict(name=name, source=src, **extra, **st)
        C.log(f"configs {name}: {src}, d_min {st['d_min_periodic']:.4f}")
    C.write_json(RESULTS / "configurations.json", dict(rows, provenance=C.provenance()))


def stage_refs(args):
    """Gamma_ref for every configuration, its norm and Lambda_lb, and its accuracy against the direct sum to R = 55."""
    out = C.read_json(RESULTS / "references.json", {"rows": {}})
    for name in needed_configs():
        if name in out["rows"]:
            continue
        x, L = load_cfg(name)
        N = len(x)
        t0 = time.perf_counter()
        G = gref(name, x, L)
        t_ref = time.perf_counter() - t0
        lam = np.linalg.eigvalsh(G)
        Q, T = projector_basis(N)
        lamPi = np.linalg.eigvalsh(Q.T @ G @ Q)
        rec = dict(name=name, N=N, L=L, norm2=float(lam[-1]), lam_lb=lambda_lower_bound(G, N),
                   lam_min_RangePi=float(lamPi[0]), sym=float(np.linalg.norm(G - G.T) / np.linalg.norm(G)),
                   null=float(np.linalg.norm(G @ T) / lam[-1]), t_ref_s=t_ref)
        t1 = time.perf_counter()
        D = DirectPairs(x, L, R_D55)
        GD = D.dense()
        np.save(C.RAW / f"exp1_{name}_GD55.npy", GD)
        from direct_periodic import tail_bound_per_particle
        tail = tail_bound_per_particle(N, L, R_D55)
        nrm, asym = norm2_sym(GD - G)
        rec.update(D55_vs_ref=nrm / rec["norm2"], D55_vs_ref_asym=asym / rec["norm2"], D55_n_images=D.n_images,
                   D55_tail_bound_rel=2 * tail / rec["norm2"], t_D55_s=time.perf_counter() - t1)
        out["rows"][name] = rec
        C.log(f"refs {name}: ||G|| {rec['norm2']:.4f}, Lambda_lb {rec['lam_lb']:.4f}, lam_min(Pi) {rec['lam_min_RangePi']:.4f}; "
              f"||G_D55 - G_ref||/||G_ref|| {rec['D55_vs_ref']:.2e} (D55 tail bound {rec['D55_tail_bound_rel']:.1e})")
        C.write_json(RESULTS / "references.json", dict(out, provenance=C.provenance()))
        del G, GD, D
    C.write_json(RESULTS / "references.json", dict(out, provenance=C.provenance()))


def ref_rec(name):
    return C.read_json(RESULTS / "references.json")["rows"][name]


# ============================================================================ (a) split identity
def mp_components(xi, r_list, dps=50):
    """50-digit theta, theta_S, theta_L (mpmath tanh-sinh quadrature, breakpoints on the e^{-s r^2} scale)."""
    import mpmath as mp
    mp.mp.dps = dps
    kap, c = mp.mpf(C.KAPPA), mp.mpf(C.AMP)
    xi2 = mp.mpf(xi) ** 2

    def rho(s):
        return 2 * mp.sqrt(s) / mp.sqrt(mp.pi) * mp.exp(-kap ** 2 / (4 * s)) - kap * mp.erfc(kap / (2 * mp.sqrt(s)))
    out = []
    for r in r_list:
        r = mp.mpf(r)
        th = c * mp.exp(-kap * r) / r
        thL = c * r ** 2 * mp.quad(lambda s: rho(s) * mp.exp(-s * r ** 2), [0, xi2 / 4, xi2])
        thS = c * r ** 2 * mp.quad(lambda s: rho(s) * mp.exp(-s * r ** 2),
                                   [xi2 + k / r ** 2 for k in (0, 0.25, 1, 3, 8, 20, 60)] + [mp.inf])
        out.append((float(th), float(thS), float(thL), float(abs(th - thS - thL) / th)))
    return out


def tensors(theta, d):
    """theta(r) d d^T / r^2 as (P, 6) independent components (xx, yy, zz, xy, xz, yz)."""
    r2 = np.einsum("pa,pa->p", d, d)
    w = theta / r2
    return np.stack([w * d[:, a] * d[:, b] for a, b in PAIRS6], 1)


def fro6(T):
    """Frobenius norm of the full 3x3 tensors from their 6 components."""
    return np.sqrt(np.sum(T[:, :3] ** 2) + 2 * np.sum(T[:, 3:] ** 2))


def stage_split(args):
    rows, radial, mp_rows = [], [], []
    rng = np.random.default_rng([SEED, 1])
    for fam in FAMS:
        for s in SEEDS:
            name = name_of(fam, 256, s)
            x, L = load_cfg(name)
            il = kdtree_images(x, L, R_PAIRS_A)
            d = il.d
            r = np.linalg.norm(d, axis=1)
            for xi in XI_A:
                K = C.kernel(xi)
                th = K.theta_full(r)
                thS = K.theta_S_scaled_closed(r) * np.exp(-(xi * r) ** 2)
                thL = K.theta_L(r)
                T, TS, TL = tensors(th, d), tensors(thS, d), tensors(thL, d)
                dT = T - TS - TL
                rel_pt = np.abs(th - thS - thL) / th
                rows.append(dict(config=name, family=fam, seed=s, N=256, xi=xi, n_pairs=int(len(r)),
                                 r_min=float(r.min()), r_max=float(r.max()),
                                 rel_fro=float(fro6(dT) / fro6(T)), max_abs=float(np.abs(dT).max()),
                                 max_abs_over_max_K=float(np.abs(dT).max() / np.abs(T).max()),
                                 max_rel_pointwise=float(rel_pt.max()),
                                 r_at_max_rel=float(r[np.argmax(rel_pt)]),
                                 rel_fro_short=float(fro6(dT[r < 1.5]) / fro6(T[r < 1.5])) if np.any(r < 1.5) else None))
            C.log(f"split {name}: max rel_fro over xi {max(q['rel_fro'] for q in rows if q['config'] == name):.2e}, "
                  f"max pointwise {max(q['max_rel_pointwise'] for q in rows if q['config'] == name):.2e}")
    rr = np.geomspace(1e-6, 12.0, 601)
    dirs = rng.standard_normal((rr.size, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for xi in XI_A:
        K = C.kernel(xi)
        th = K.theta_full(rr)
        thS = K.theta_S_scaled_closed(rr) * np.exp(-(xi * rr) ** 2)
        thL = K.theta_L(rr)
        d = rr[:, None] * dirs
        dT = tensors(th, d) - tensors(thS, d) - tensors(thL, d)
        rel = np.abs(th - thS - thL) / th
        for k in range(rr.size):
            radial.append(dict(xi=xi, r=float(rr[k]), rel_pointwise=float(rel[k]),
                               rel_tensor_fro=float(fro6(dT[k:k + 1]) / fro6(tensors(th[k:k + 1], d[k:k + 1]))),
                               theta=float(th[k]), theta_S=float(thS[k]), theta_L=float(thL[k])))
        rmp = np.geomspace(1e-4, 12.0, 25)
        ref = mp_components(xi, rmp)
        thS_f = K.theta_S_scaled_closed(rmp) * np.exp(-(xi * rmp) ** 2)
        thL_f = K.theta_L(rmp)
        th_f = K.theta_full(rmp)
        for k, (th_m, thS_m, thL_m, id_m) in enumerate(ref):
            mp_rows.append(dict(xi=xi, r=float(rmp[k]), theta_mp=th_m, theta_S_mp=thS_m, theta_L_mp=thL_m,
                                identity_mp=id_m,
                                err_theta_rel=abs(th_f[k] - th_m) / th_m,
                                err_theta_S_rel_to_theta=abs(thS_f[k] - thS_m) / th_m,
                                err_theta_L_rel_to_theta=abs(thL_f[k] - thL_m) / th_m,
                                err_theta_S_rel_self=abs(thS_f[k] - thS_m) / thS_m if thS_m > 0 else None,
                                err_theta_L_rel_self=abs(thL_f[k] - thL_m) / thL_m if thL_m > 0 else None,
                                err_naive_L_rel_self=abs(th_f[k] - thS_f[k] - thL_m) / thL_m if thL_m > 0 else None))
        C.log(f"split radial xi={xi}: max pointwise {max(q['rel_pointwise'] for q in radial if q['xi'] == xi):.2e} "
              f"(r < 0.9: {max(q['rel_pointwise'] for q in radial if q['xi'] == xi and q['r'] < 0.9):.2e}); mpmath: "
              f"theta_S {max(q['err_theta_S_rel_self'] for q in mp_rows if q['xi'] == xi and q['err_theta_S_rel_self'] is not None):.1e}, "
              f"theta_L {max(q['err_theta_L_rel_self'] for q in mp_rows if q['xi'] == xi and q['err_theta_L_rel_self'] is not None):.1e} (rel. to self)")
    C.write_csv(RESULTS / "a_split_configs.csv", rows)
    C.write_csv(RESULTS / "a_split_radial.csv", radial)
    C.write_csv(RESULTS / "a_split_mpmath.csv", mp_rows)
    C.write_json(RESULTS / "a_split_summary.json", dict(
        max_rel_fro=max(q["rel_fro"] for q in rows), max_abs=max(q["max_abs"] for q in rows),
        max_rel_pointwise_configs=max(q["max_rel_pointwise"] for q in rows),
        max_rel_pointwise_radial=max(q["rel_pointwise"] for q in radial),
        max_rel_pointwise_radial_r_lt_0p9=max(q["rel_pointwise"] for q in radial if q["r"] < 0.9),
        mp_max_err_theta_S_self=max(q["err_theta_S_rel_self"] for q in mp_rows if q["err_theta_S_rel_self"] is not None),
        mp_max_err_theta_S_self_xir_le_7=max(q["err_theta_S_rel_self"] for q in mp_rows
                                             if q["err_theta_S_rel_self"] is not None and q["xi"] * q["r"] <= 7.0),
        mp_max_err_theta_S_rel_to_theta=max(q["err_theta_S_rel_to_theta"] for q in mp_rows),
        mp_max_err_theta_L_self=max(q["err_theta_L_rel_self"] for q in mp_rows if q["err_theta_L_rel_self"] is not None),
        mp_max_err_theta_L_rel_to_theta=max(q["err_theta_L_rel_to_theta"] for q in mp_rows),
        mp_max_err_theta=max(q["err_theta_rel"] for q in mp_rows),
        mp_max_err_naive_L_self=max(q["err_naive_L_rel_self"] for q in mp_rows if q["err_naive_L_rel_self"] is not None),
        xi=list(XI_A), r_pairs=R_PAIRS_A,
        provenance=C.provenance()))


# ============================================================================ (b) short-range truncation
def stage_shortrange(args):
    rows, fits = [], []
    xi = XI0
    K = C.kernel(xi)
    for fam in FAMS:
        for s_ in SEEDS:
            name = name_of(fam, 256, s_)
            t0 = time.perf_counter()
            x, L = load_cfg(name)
            N = len(x)
            G = gref(name, x, L)
            GD = np.load(C.RAW / f"exp1_{name}_GD55.npy")
            rr = ref_rec(name)
            nref = rr["norm2"]
            cp = Components(x, L, G, xi)
            kc_ex = shell_kc(2 * xi * B_EXACT, L)
            fs = FourierShells(xi, L)
            eps_F_ex = fs.eps_F(kc_ex, N / L ** 3)
            GL = lap_full(cp.fourier_W(kc_ex))
            GS_full = cp.GS
            prep = cp.rst.prepare(xi)
            prev = -1
            for s in S_B:
                rc = float(s / xi)
                n = cp.rst.n_img(rc)
                GSr = cp.lap_images(0, n)
                E, asym = norm2_sym(GSr + GL - G)
                E55, _ = norm2_sym(GSr + GL - GD)
                lam_tail = float(np.linalg.eigvalsh(GS_full - GSr)[-1]) if n < cp.m_far else 0.0
                rows.append(dict(config=name, family=fam, seed=s_, N=N, xi=xi, s=float(s), rc=rc, n_img=n,
                                 same_neighbours_as_previous=bool(n == prev),
                                 E_S=E / nref, E_S_asym=asym / nref, E_S_vs_D55=E55 / nref,
                                 E_trunc_exact=lam_tail / nref,
                                 bound_eps_S=cp.rst.eps_S(prep, rc) / nref,
                                 bound_eps_S_over_lam_lb=cp.rst.eps_S(prep, rc) / rr["lam_lb"]))
                prev = n
            floor_ops = norm2_sym(GS_full + GL - G)[0] / nref
            mine = [q for q in rows if q["config"] == name]
            floor = float(np.median([q["E_S"] for q in mine[-5:]]))
            fit_pts = [q for q in mine if q["E_S"] > 100 * floor]
            if len(fit_pts) >= 3:
                sv = np.array([q["s"] for q in fit_pts])
                ev = np.log(np.array([q["E_S"] for q in fit_pts]))
                A = np.stack([np.ones_like(sv), -sv ** 2], 1)
                (a, c), *_ = np.linalg.lstsq(A, ev, rcond=None)
                sub = sv >= 3.0 - 1e-9
                (a3, c3), *_ = np.linalg.lstsq(A[sub], ev[sub], rcond=None)
                fits.append(dict(config=name, family=fam, seed=s_, s_lo=float(sv.min()), s_hi=float(sv.max()),
                                 n_points=len(fit_pts), a=float(a), c=float(c), c_s_ge_3=float(c3), a_s_ge_3=float(a3),
                                 floor=floor,
                                 ratio_bound_over_E_max=float(max(q["bound_eps_S"] / q["E_trunc_exact"] for q in fit_pts)),
                                 ratio_bound_over_E_min=float(min(q["bound_eps_S"] / q["E_trunc_exact"] for q in fit_pts)),
                                 floor_Gamma_S_full_plus_GL=floor_ops, kc_exact=kc_ex, eps_F_exact_rel=eps_F_ex / nref,
                                 n_flat_steps=int(sum(q["same_neighbours_as_previous"] for q in mine))))
            C.log(f"shortrange {name}: E_S(s=2) {mine[0]['E_S']:.2e}, floor {floor:.2e} (vs D55 "
                  f"{np.median([q['E_S_vs_D55'] for q in mine[-5:]]):.2e}); fit c = {fits[-1]['c']:.3f} on "
                  f"s in [{fits[-1]['s_lo']:.1f}, {fits[-1]['s_hi']:.1f}]; Fourier tail {eps_F_ex / nref:.1e} "
                  f"({time.perf_counter() - t0:.0f} s)")
    C.write_csv(RESULTS / "b_shortrange.csv", rows)
    C.write_csv(RESULTS / "b_shortrange_fits.csv", fits)
    fam_fits = {}
    for fam in FAMS:
        Ss = sorted({q["s"] for q in rows})
        med = np.array([np.median([q["E_S"] for q in rows if q["family"] == fam and q["s"] == s]) for s in Ss])
        floor = float(np.median(med[-5:]))
        keep = med > 100 * floor
        sv, ev = np.array(Ss)[keep], np.log(med[keep])
        A = np.stack([np.ones_like(sv), -sv ** 2], 1)
        (a, c), *_ = np.linalg.lstsq(A, ev, rcond=None)
        sub = sv >= 3.0 - 1e-9
        (a3, c3), *_ = np.linalg.lstsq(A[sub], ev[sub], rcond=None)
        fam_fits[fam] = dict(c_median_curve=float(c), a=float(a), s_lo=float(sv.min()), s_hi=float(sv.max()),
                             c_median_curve_s_ge_3=float(c3), a_s_ge_3=float(a3), floor_median=floor)
    C.write_json(RESULTS / "b_shortrange_summary.json", dict(
        xi=xi, s=S_B.tolist(), B_exact=B_EXACT, family_fits=fam_fits,
        c_range_individual=[min(f_["c"] for f_ in fits), max(f_["c"] for f_ in fits)],
        c_s_ge_3_range_individual=[min(f_["c_s_ge_3"] for f_ in fits), max(f_["c_s_ge_3"] for f_ in fits)],
        bound_over_E_range=[min(f_["ratio_bound_over_E_min"] for f_ in fits), max(f_["ratio_bound_over_E_max"] for f_ in fits)],
        n_flat_steps_total=int(sum(f_["n_flat_steps"] for f_ in fits)),
        provenance=C.provenance()))


# ============================================================================ shared: operators per mesh
def setting_params(setting, L):
    st = SETTINGS[setting]
    return st["xi"], st["rc"], kc_shell_below(st["kc_max"], L)


def try_mesh(L, K, kc, M, p):
    mc = int(np.floor(kc * L / (2 * np.pi) + 1e-12))
    eta = kc * (L / M) / np.pi
    if mc >= M / 2:
        return None, dict(valid=False, reason=f"Nyquist: m_c = {mc} >= M/2 = {M / 2}", mc=mc, eta=eta)
    if eta >= 1:
        return None, dict(valid=False, reason=f"eta = {eta:.3f} >= 1", mc=mc, eta=eta)
    mesh = mesh_for(L, K, kc, M, p)
    return mesh, dict(valid=True, reason="", mc=mc, eta=eta, n_modes=mesh.n_modes, n_excluded=mesh.n_excluded,
                      W2min=mesh.W2min)


def self_response(mesh, x):
    """Mesh self response W_ii = (1/V) sum_k |S_i(k)|^2 G_h(k) (mode space), shape (N, 3, 3)."""
    s = mesh.stencil_dft(x)
    o, mv = mesh.mc, mesh.mv
    S = s[:, 0, mv[:, 0] + o] * s[:, 1, mv[:, 1] + o] * s[:, 2, mv[:, 2] + o]
    a2 = np.abs(S) ** 2
    return np.einsum("nk,kab->nab", a2, mesh.Gm) / mesh.V


def diag_blocks(G, N):
    return G.reshape(N, 3, N, 3)[np.arange(N), :, np.arange(N), :]


def real_degree(op):
    N = op.N
    return op.DS


def fourier_degree(cp, kc):
    W = cp.fourier_W(kc)
    N = W.shape[0]
    idx = np.arange(N)
    W[idx, idx] = 0.0
    return W.sum(axis=1), W


# ============================================================================ (c) degree convergence
def stage_degree(args):
    rows = []
    cases = [(name_of("strong", N, 1), p, setting) for N in (128, 256, 512) for p in P_C for setting in SETTINGS]
    cases += [(name_of(fam, 256, 1), 9, setting) for fam in ("uniform", "moderate") for setting in SETTINGS]
    cache = {}
    for name, p, setting in cases:
        x, L = load_cfg(name)
        N = len(x)
        xi, rc, kc = setting_params(setting, L)
        K = C.kernel(xi)
        key = (name, setting)
        if key not in cache:
            G = gref(name, x, L)
            Dref = diag_blocks(G, N)
            cp = Components(x, L, G, xi)
            DLkc, _ = fourier_degree(cp, kc)
            op0 = CachedGammaH(x, L, K, rc, mesh_for(L, K, 2 * np.pi * 2 / L, 16, 4))
            DS = op0.DS.copy()
            nD = float(np.sqrt(np.sum(Dref ** 2)))
            floor = float(np.sqrt(np.sum((DS + DLkc - Dref) ** 2)) / nD)
            cache[key] = dict(G=G, Dref=Dref, DLkc=DLkc, DS=DS, nD=nD, floor=floor)
            del cp
        cc = cache[key]
        for M in M_LIST[N]:
            mesh, info = try_mesh(L, K, kc, M, p)
            row = dict(config=name, N=N, family=name.split("_")[0], setting=setting, xi=xi, rc=rc, kc=kc, p=p, M=M,
                       h=L / M, **info, E_D_floor=cc["floor"])
            if mesh is not None:
                idx, wt = mesh.flat(x)
                Dm = mesh.degree(idx, wt)                     # production FFT degree, mesh self response included
                Wii = self_response(mesh, x)
                DLh = Dm - Wii
                Dh = cc["DS"] + DLh
                nD = cc["nD"]
                row.update(E_D_complete=float(np.sqrt(np.sum((Dh - cc["Dref"]) ** 2)) / nD),
                           E_D_mesh=float(np.sqrt(np.sum((DLh - cc["DLkc"]) ** 2)) / nD),
                           E_D_self_not_removed=float(np.sqrt(np.sum((cc["DS"] + Dm - cc["Dref"]) ** 2)) / nD),
                           self_response_rel=float(np.sqrt(np.sum(Wii ** 2)) / nD))
                if N <= 256 and M in (M_LIST[N][0], M_LIST[N][len(M_LIST[N]) // 2]):
                    from run_psearch_dense512 import dense_fast
                    Gh, _ = dense_fast(x, L, K, rc, mesh)
                    row["diag_dense_vs_fft_minus_self"] = float(np.sqrt(np.sum((diag_blocks(Gh, N) - Dh) ** 2)) / nD)
                    del Gh
            rows.append(row)
        mine = [q for q in rows if q["config"] == name and q["p"] == p and q["setting"] == setting]
        C.log(f"degree {name} {setting} p={p}: " + ", ".join(
            f"M{q['M']}:{q['E_D_complete']:.1e}/{q['E_D_mesh']:.1e}" if q["valid"] else f"M{q['M']}:invalid"
            for q in mine) + f"; floor {cc['floor']:.1e}")
        C.write_csv(RESULTS / "c_degree.csv", rows)
    rates = []
    for name, p, setting in cases:
        mine = [q for q in rows if q["config"] == name and q["p"] == p and q["setting"] == setting and q["valid"]]
        for a, b in zip(mine[:-1], mine[1:]):
            rates.append(dict(config=name, p=p, setting=setting, M_lo=a["M"], M_hi=b["M"],
                              rate_mesh=float(-np.log(b["E_D_mesh"] / a["E_D_mesh"]) / np.log(b["M"] / a["M"])),
                              rate_complete=float(-np.log(b["E_D_complete"] / a["E_D_complete"]) / np.log(b["M"] / a["M"]))))
    C.write_csv(RESULTS / "c_degree_rates.csv", rates)
    gfit = []
    for name, p, setting in cases:
        mine = [q for q in rows if q["config"] == name and q["p"] == p and q["setting"] == setting and q["valid"]
                and q["E_D_mesh"] >= 1e-13]
        if len(mine) >= 3:
            lm = np.log([q["M"] for q in mine])
            le = np.log([q["E_D_mesh"] for q in mine])
            slope, icpt = np.polyfit(lm, le, 1)
            gfit.append(dict(config=name, p=p, setting=setting, M_lo=mine[0]["M"], M_hi=mine[-1]["M"], n_points=len(mine),
                             rate_mesh_lsq=float(-slope)))
    C.write_csv(RESULTS / "c_degree_rate_fits.csv", gfit)
    v = [q for q in rows if q["valid"]]
    C.write_json(RESULTS / "c_degree_summary.json", dict(
        settings=SETTINGS, M_list=M_LIST, p=list(P_C), n_rows=len(rows), n_valid=len(v),
        invalid=[dict(config=q["config"], setting=q["setting"], p=q["p"], M=q["M"], reason=q["reason"])
                 for q in rows if not q["valid"]],
        max_n_excluded=max(q["n_excluded"] for q in v), min_W2min=min(q["W2min"] for q in v),
        max_eta=max(q["eta"] for q in v),
        self_response_rel_range=[min(q["self_response_rel"] for q in v), max(q["self_response_rel"] for q in v)],
        max_diag_dense_vs_fft_minus_self=max(q["diag_dense_vs_fft_minus_self"] for q in v
                                             if q.get("diag_dense_vs_fft_minus_self") is not None),
        n_dense_checks=sum(1 for q in v if q.get("diag_dense_vs_fft_minus_self") is not None),
        floors={f"{q['config']}|{q['setting']}": q["E_D_floor"] for q in v},
        rate_fits=gfit, provenance=C.provenance()))


# ============================================================================ (d) action
def probe_set(x, L, N, G, Q):
    rng = np.random.default_rng([SEED, 4, N])
    P = {f"gauss{k}": rng.standard_normal(3 * N) for k in range(5)}
    lam, U = np.linalg.eigh(Q.T @ G @ Q)
    P["spec_min"] = Q @ U[:, 0]
    P["spec_max"] = Q @ U[:, -1]
    kmin = 2 * np.pi / L
    for a, b, tag in ((0, 0, "long_x"), (1, 1, "long_y"), (0, 1, "trans_xy"), (2, 0, "trans_zx")):
        X = np.zeros((N, 3))
        X[:, b] = np.cos(kmin * x[:, a])
        X -= X.mean(0)
        P[f"slow_{tag}"] = X.ravel()
    for a in range(3):
        T = np.zeros((N, 3))
        T[:, a] = 1.0
        P[f"translation_{'xyz'[a]}"] = T.ravel()
    return P, lam


def stage_action(args):
    rows, prow = [], []
    for N in (128, 256):
        for fam in FAMS:
            for s_ in SEEDS:
                name = name_of(fam, N, s_)
                x, L = load_cfg(name)
                G = gref(name, x, L)
                nref = ref_rec(name)["norm2"]
                Q, T = projector_basis(N)
                P, lamref = probe_set(x, L, N, G, Q)
                GP = {k: G @ v for k, v in P.items()}
                t0 = time.perf_counter()
                for setting in SETTINGS:
                    xi, rc, kc = setting_params(setting, L)
                    K = C.kernel(xi)
                    for M in M_LIST[N]:
                        mesh, info = try_mesh(L, K, kc, M, P_D)
                        row = dict(config=name, N=N, family=fam, seed=s_, setting=setting, xi=xi, rc=rc, kc=kc, p=P_D,
                                   M=M, h=L / M, **info)
                        if mesh is None:
                            rows.append(row)
                            continue
                        op = CachedGammaH(x, L, K, rc, mesh)
                        pd = op.existing_gammah()
                        Gh = pd.dense()
                        E, asym = norm2_sym(Gh - G)
                        lh = np.linalg.eigvalsh(Q.T @ (0.5 * (Gh + Gh.T)) @ Q)
                        rel, nullr, cvp = {}, {}, 0.0
                        for k, v in P.items():
                            y = pd.matvec(v)                         # production pair-difference action
                            yc = op.apply(v.reshape(N, 3)).ravel()   # cached D - H action
                            dv = float(np.linalg.norm(y - GP[k]))
                            on = nref * np.linalg.norm(v)
                            if k.startswith("translation"):
                                nullr[k] = dict(abs=float(np.linalg.norm(y)), opnorm=float(np.linalg.norm(y) / on),
                                                opnorm_cached=float(np.linalg.norm(yc) / on),
                                                opnorm_ref=float(np.linalg.norm(GP[k]) / on))
                                prow.append(dict(config=name, N=N, family=fam, seed=s_, setting=setting, M=M, probe=k,
                                                 abs=nullr[k]["abs"], opnorm=nullr[k]["opnorm"], rel=None,
                                                 opnorm_cached=nullr[k]["opnorm_cached"],
                                                 opnorm_ref=nullr[k]["opnorm_ref"]))
                            else:
                                cvp = max(cvp, float(np.linalg.norm(y - yc) / np.linalg.norm(y)))
                                rel[k] = dv / np.linalg.norm(GP[k])
                                prow.append(dict(config=name, N=N, family=fam, seed=s_, setting=setting, M=M, probe=k,
                                                 abs=dv, opnorm=dv / on, rel=rel[k], opnorm_cached=None, opnorm_ref=None))
                        row.update(E_op=E / nref, E_op_asym=asym / nref,
                                   E_gauss_max=max(v for k, v in rel.items() if k.startswith("gauss")),
                                   E_gauss_median=float(np.median([v for k, v in rel.items() if k.startswith("gauss")])),
                                   E_spec_min=rel["spec_min"], E_spec_max=rel["spec_max"],
                                   E_slow_max=max(v for k, v in rel.items() if k.startswith("slow")),
                                   translation_opnorm_max=max(v["opnorm"] for v in nullr.values()),
                                   translation_abs_max=max(v["abs"] for v in nullr.values()),
                                   translation_opnorm_cached_max=max(v["opnorm_cached"] for v in nullr.values()),
                                   translation_opnorm_ref_max=max(v["opnorm_ref"] for v in nullr.values()),
                                   null_dense_h=float(np.linalg.norm(Gh @ T) / (nref * np.sqrt(3))),
                                   sym=float(np.linalg.norm(Gh - Gh.T) / np.linalg.norm(Gh)),
                                   cached_vs_pairdiff=cvp,
                                   lam_min_h_RangePi=float(lh[0]), lam_min_ref_RangePi=float(lamref[0]),
                                   n_negative_RangePi=int(np.sum(lh < 0)))
                        rows.append(row)
                C.log(f"action {name}: " + "; ".join(
                    f"{st}: " + " ".join(f"{q['M']}:{q['E_gauss_max']:.1e}" for q in rows
                                         if q["config"] == name and q["setting"] == st and q["valid"])
                    for st in SETTINGS) + f" ({time.perf_counter() - t0:.0f} s)")
                C.write_csv(RESULTS / "d_action.csv", rows)
                C.write_csv(RESULTS / "d_action_probes.csv", prow)
    C.write_json(RESULTS / "d_action_summary.json", dict(
        settings=SETTINGS, p=P_D, M_list={k: v for k, v in M_LIST.items() if k <= 256},
        max_sym=max(q["sym"] for q in rows if q["valid"]),
        max_translation_opnorm=max(q["translation_opnorm_max"] for q in rows if q["valid"]),
        max_translation_opnorm_cached=max(q["translation_opnorm_cached_max"] for q in rows if q["valid"]),
        max_null_dense_h=max(q["null_dense_h"] for q in rows if q["valid"]),
        max_cached_vs_pairdiff=max(q["cached_vs_pairdiff"] for q in rows if q["valid"]),
        min_lam_min_h=min(q["lam_min_h_RangePi"] for q in rows if q["valid"]),
        any_negative=any(q["n_negative_RangePi"] > 0 for q in rows if q["valid"]),
        provenance=C.provenance()))


def stage_fits(args):
    """Least-squares rates of the saved action errors (strict cutoffs, points >= 1e-13; no recomputation)."""
    import csv
    with open(RESULTS / "d_action.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["valid"] == "True" and r["setting"] == "strict"]
    out = []
    for N in (128, 256):
        for fam in FAMS:
            for key in ("E_op", "E_gauss_max"):
                Ms = sorted({int(r["M"]) for r in rows if int(r["N"]) == N and r["family"] == fam})
                E = np.array([max(float(r[key]) for r in rows if int(r["N"]) == N and r["family"] == fam and int(r["M"]) == M)
                              for M in Ms])
                keep = E >= 1e-13
                slope, _ = np.polyfit(np.log(np.array(Ms, float)[keep]), np.log(E[keep]), 1)
                out.append(dict(N=N, family=fam, quantity=key + " (max over seeds)", M_lo=int(np.array(Ms)[keep][0]),
                                M_hi=int(np.array(Ms)[keep][-1]), n_points=int(keep.sum()), rate_lsq=float(-slope)))
    C.write_csv(RESULTS / "d_action_rate_fits.csv", out)
    C.log("action rates: " + ", ".join(f"N{q['N']} {q['family']} {q['quantity'][:7]} {q['rate_lsq']:.2f}" for q in out))


STAGES = dict(configs=stage_configs, refs=stage_refs, split=stage_split, shortrange=stage_shortrange,
              degree=stage_degree, action=stage_action, fits=stage_fits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=["all"])
    args = ap.parse_args()
    stages = list(STAGES) if args.stages == ["all"] else args.stages
    C.RESULTS = RESULTS
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "commands.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}  cd {HERE.relative_to(C.REPO)} && "
                 f"{' '.join(['python3'] + sys.argv)}\n")
    for s in stages:
        t0 = time.perf_counter()
        STAGES[s](args)
        C.log(f"stage {s} done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
