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

Figure 1 (results/; panel (a) only shows the configurations of results/configurations.json):
  (b) split      K = K_S + K_L: theta_S closed form, theta_L by its own Gauss-Legendre s-integral (never K - K_S),
                 theta = c e^{-kappa r}/r; all pair images |d| <= 8 of N = 256 uniform/strong/moderate x 3 seeds,
                 xi in XI_A; a synthetic radial sweep r = 1e-6 ... 12; 50-digit mpmath values of each component
                 -> split_*.csv/json
  (c) shortrange E_S(rc) = ||Gamma_S^{rc} + Gamma_L^{exact} - Gamma_ref||_2/||Gamma_ref||_2, xi = 0.85,
                 s = xi rc = 2.0 ... 6.6; Gamma_L^{exact} = mesh-free Fourier lattice sum to k_c = 2 xi 6.6 (no mesh);
                 also against Gamma_D55, the exact-arithmetic tail ||Gamma_S^{rc} - Gamma_S^{full}||, and the
                 certified RealSpaceTail bound -> shortrange*.csv/json
  (d) action     production matrix-free action Gamma_h(M) X (pair-difference GammaH.matvec; cached D - H as a second
                 path) with xi, rc, kc, p fixed and only M varied; probes: 10 Gaussian (fixed seeds, translation
                 removed), the 3 slowest and the fastest eigenvector of Gamma_ref, 3 translations (absolute /
                 operator-normalised residuals only);
                   E_complete = ||(Gamma_h - Gamma_ref) X|| / ||Gamma_ref X||          (the quantity of Fig. 1(d))
                 diagnostic columns (not plotted): E_mesh = ||(Gamma_L^h - Gamma_L^{kc}) X|| / ||Gamma_ref X|| and
                 E_trunc = ||(Gamma_S^{rc} + Gamma_L^{kc} - Gamma_ref) X|| / ||Gamma_ref X||;
                 N = 128, 256 (uniform/moderate/strong x 3 seeds) and N = 512 (uniform/moderate/strong, seed 1;
                 strongly clustered = the optimisation configuration); dense E_op, symmetry, lambda_min on Range(Pi)
                 for context -> action_*.csv/json; stage action_p: p = 4, 6, 8, 9 at N = 512 (supplementary)
                 -> action_p*.csv; stage action_fits: least-squares rates from the saved CSVs
Cutoff settings of (d): "production" = the measured N = 512 optimum in full double precision (xi = 0.85,
rc = 4.88775618844242, kc = 9.358534792617089 = 2 pi sqrt(172)/L_512, p = 9), kc snapped to the largest reciprocal
shell <= kc of each box; "strict" xi = 0.85, rc = 6/xi, kc <= 12 xi (convergence experiment only, so the mesh error
is visible below the production truncation floor). The diffusion-coefficient (degree D_i) study of the first version
is not part of Experiment 1 any more (see report_exp1.md, "Changes").

    python3 run_exp1.py all            # or any of: configs refs split shortrange action action_p action_fits
"""
import argparse
import hashlib
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
from cached_operator import CachedGammaH, mesh_for, short_range_weights  # noqa: E402
from error_validation import Components, REF_REL_TOL, PAIRS6  # noqa: E402
from direct_periodic import DirectPairs  # noqa: E402
from test_parameter_selection import lap_full, lattice_modes, INFL_THR  # noqa: E402
from exp1_dense_reference import mode_pair_matrix_fast  # noqa: E402
from prl_toy_models import LatticeFriction  # noqa: E402

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
# production = the measured N = 512 optimum (results_psearch/optimum_strong_N512_s1_main.json), full doubles: k_c is the
# exact shell radius 2 pi sqrt(172)/L of the N = 512 box; a decimal truncation (9.3585) drops that shell.
SETTINGS = {"production": dict(xi=0.85, rc=4.88775618844242, kc_max=9.358534792617089),
            "strict": dict(xi=0.85, rc=6.0 / 0.85, kc_max=2 * 0.85 * 6.0)}
P_ACTION = 9
P_STUDY = (4, 6, 8, 9)
M_PROD = 40
MAIN_C = "strong_N512_s1"
M_LIST = {128: (16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64),
          256: (20, 22, 24, 28, 32, 36, 40, 48, 56, 64, 80),
          512: (26, 28, 30, 32, 36, 40, 48, 56, 64, 80, 96)}
GAUSS_SEEDS = tuple(range(1001, 1011))
FIT_FLOOR = 1e-13                       # rate fits use only points above the round-off / reference level
SEED = 20261019


def name_of(fam, N, seed):
    return f"{fam}_N{N}_s{seed}"


def cfg_path(name):
    for d in (CONFIG_DIR,):
        if (d / f"{name}.npz").exists():
            return d / f"{name}.npz"
    return None


def load_cfg(name):
    with np.load(cfg_path(name)) as d:
        return np.mod(d["q"].copy(), float(d["L"])), float(d["L"])


def gref(name, x, L):
    C.RAW.mkdir(parents=True, exist_ok=True)
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
    out += [name_of(fam, 512, 1) for fam in FAMS]
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
    C.RAW.mkdir(parents=True, exist_ok=True)
    for name in needed_configs():
        if (name in out["rows"] and
            (C.RAW / f"{name}_Gref.npy").exists() and
            (C.RAW / f"exp1_{name}_GD55.npy").exists()):
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
    C.write_csv(RESULTS / "split_configs.csv", rows)
    C.write_csv(RESULTS / "split_radial.csv", radial)
    C.write_csv(RESULTS / "split_mpmath.csv", mp_rows)
    C.write_json(RESULTS / "split_summary.json", dict(
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
    C.write_csv(RESULTS / "shortrange.csv", rows)
    C.write_csv(RESULTS / "shortrange_fits.csv", fits)
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
    C.write_json(RESULTS / "shortrange_summary.json", dict(
        xi=xi, s=S_B.tolist(), B_exact=B_EXACT, family_fits=fam_fits,
        c_range_individual=[min(f_["c"] for f_ in fits), max(f_["c"] for f_ in fits)],
        c_s_ge_3_range_individual=[min(f_["c_s_ge_3"] for f_ in fits), max(f_["c_s_ge_3"] for f_ in fits)],
        bound_over_E_range=[min(f_["ratio_bound_over_E_min"] for f_ in fits), max(f_["ratio_bound_over_E_max"] for f_ in fits)],
        n_flat_steps_total=int(sum(f_["n_flat_steps"] for f_ in fits)),
        provenance=C.provenance()))


# ============================================================================ (c) matrix-free action convergence
def setting_params(setting, L):
    st = SETTINGS[setting]
    return st["xi"], st["rc"], kc_shell_below(st["kc_max"], L)


def sum_of_three_squares(n):
    """True if n = a^2 + b^2 + c^2 (Legendre: n is not of the form 4^a (8b + 7))."""
    while n > 0 and n % 4 == 0:
        n //= 4
    return n % 8 != 7


def shell_count(kc, L):
    """Outermost reciprocal shell |m|^2 = n_c <= (kc L/2 pi)^2 and an independent integer count of |m|^2 <= n_c."""
    n_c = int(np.floor((kc * L / (2 * np.pi)) ** 2 * (1 + 1e-12)))
    while not sum_of_three_squares(n_c):
        n_c -= 1
    m = np.arange(-int(np.sqrt(n_c)), int(np.sqrt(n_c)) + 1)
    n2 = (m[:, None, None] ** 2 + m[None, :, None] ** 2 + m[None, None, :] ** 2).ravel()
    return n_c, int(np.sum(n2 <= n_c))


def check_mesh(L, K, kc, M, p):
    """Validity of one FFT size at fixed (xi, rc, kc, p): Nyquist, eta < 1, every retained shell complete
    (independent integer count) and no mode below the influence-function threshold |W_hat|^2 <= INFL_THR."""
    mc = int(np.floor(kc * L / (2 * np.pi) + 1e-12))
    eta = kc * (L / M) / np.pi
    info = dict(M=M, h=L / M, mc=mc, eta=eta, valid=False, reason="")
    if not mc < M / 2:
        info["reason"] = f"Nyquist: largest retained |m_a| = {mc} >= M/2 = {M / 2:g}"
        return None, info
    if not eta < 1:
        info["reason"] = f"eta = k_c h/pi = {eta:.4f} >= 1"
        return None, info
    mesh = mesh_for(L, K, kc, M, p)
    n_c, n_exp = shell_count(kc, L)
    info.update(n_shell=n_c, n_modes=mesh.n_modes, n_excluded=mesh.n_excluded, n_modes_expected=n_exp,
                W2min=mesh.W2min, max_influence_amplification=float(1.0 / mesh.W2min))
    if mesh.n_modes + mesh.n_excluded != n_exp:
        info["reason"] = f"retained modes {mesh.n_modes + mesh.n_excluded} != shell count {n_exp}"
        return None, info
    if mesh.n_excluded > 0:
        info["reason"] = f"{mesh.n_excluded} modes with |W_hat|^2 <= {INFL_THR:g} (incomplete shells)"
        return None, info
    info["valid"] = True
    return mesh, info


def sha(v):
    return hashlib.sha256(np.ascontiguousarray(v, dtype=np.float64).tobytes()).hexdigest()


def action_probes(N, G):
    """Unit-norm probes: 10 Gaussian (fixed seeds, translation removed), the 3 slowest and the fastest eigenvector of
    sym(Gamma_ref) outside the translation space, and the 3 translations 1_N (x) e_a / sqrt(N)."""
    probes = []
    for a, sd in enumerate(GAUSS_SEEDS):
        X = np.random.default_rng(sd).standard_normal((N, 3))
        X -= X.mean(axis=0)
        probes.append(dict(probe_class="gaussian", probe_id=f"gauss{a}", seed=sd, v=X.ravel() / np.linalg.norm(X)))
    Q, T = projector_basis(N)
    lam, U = np.linalg.eigh(0.5 * (G + G.T))
    tr = np.argsort(np.abs(lam))[:3]
    pos = np.setdiff1d(np.arange(3 * N), tr)
    pos = pos[np.argsort(lam[pos])]
    assert lam[pos[0]] > 0
    spec = dict(translation_eigenvalues=lam[tr].tolist(),
                translation_overlap=np.linalg.norm(T.T @ U[:, tr], axis=0).tolist(),
                max_translation_overlap_other=float(np.linalg.norm(T.T @ U[:, pos], axis=0).max()),
                lambda_min_positive=float(lam[pos[0]]), lambda_max=float(lam[pos[-1]]))
    for tag, k in (("slow", 0), ("slow", 1), ("slow", 2), ("fast", len(pos) - 1)):
        j = pos[k]
        u = U[:, j] * np.sign(U[np.argmax(np.abs(U[:, j])), j])
        u = u / np.linalg.norm(u)
        nb = [lam[pos[q]] for q in (k - 1, k + 1) if 0 <= q < len(pos)]
        probes.append(dict(probe_class=f"spectral_{tag}", probe_id=f"spec_{tag}{k if tag == 'slow' else ''}",
                           eigen_index_positive=int(k), eigenvalue=float(lam[j]),
                           eigen_residual_rel=float(np.linalg.norm(G @ u - lam[j] * u) / abs(lam[j])),
                           rel_gap_to_neighbours=float(min(abs(lam[j] - b) for b in nb) / abs(lam[j])), v=u))
    for a in range(3):
        X = np.zeros((N, 3))
        X[:, a] = 1.0
        probes.append(dict(probe_class="translation", probe_id=f"trans_{'xyz'[a]}", v=X.ravel() / np.sqrt(N)))
    for pr in probes:
        pr.update(sha256=sha(pr["v"]), translation_component=float(np.linalg.norm(T.T @ pr["v"])))
    return probes, spec


def real_dense(x, L, K, rc):
    """Dense Gamma_S^{rc} on the production image list (the real-space blocks of run_psearch_dense512.dense_fast)."""
    N = len(x)
    il = kdtree_images(np.mod(x, L), L, rc)
    w = short_range_weights(K, il.d)
    wd = w[:, None] * il.d
    flat = il.i.astype(np.int64) * N + il.j
    W = np.empty((N, N, 3, 3))
    for a, b in PAIRS6:
        W[:, :, a, b] = W[:, :, b, a] = np.bincount(flat, wd[:, a] * il.d[:, b], N * N).reshape(N, N)
    return lap_full(W + W.transpose(1, 0, 2, 3)), len(il.i)


def action_case(name, x, L, G, GD, probes, setting, p, M_list, dense):
    """All M for one (configuration, cutoff setting, p): complete, mesh-only and truncation action errors."""
    fam, N, seed = name.split("_")[0], len(x), int(name.split("_s")[-1])
    xi, rc, kc = setting_params(setting, L)
    K = C.kernel(xi)
    nref = norm2_sym(G)[0]
    Xm = np.stack([pr["v"] for pr in probes], 1)
    Yref, Y55 = G @ Xm, GD @ Xm
    mv, _, _ = lattice_modes(L, kc)
    GL = lap_full(Components(x, L, G, xi).fourier_W(kc, mv=mv))        # exact truncated Fourier operator
    YL = GL @ Xm
    GS, n_img = real_dense(x, L, K, rc)
    Ytr = GS @ Xm + YL                                                # Gamma_S^{rc} + Gamma_L^{kc}: M-independent
    E_op_trunc = norm2_sym(GS + GL - G)[0] / nref
    Q, _ = projector_basis(N)
    head = dict(config=name, N=N, family=fam, seed=seed, setting=setting, xi=xi, rc=rc, kc=kc, p=p)
    rows, trows, mrows = [], [], []
    for M in M_list:
        mesh, info = check_mesh(L, K, kc, M, p)
        mrow = dict(head, **info, production_point=bool(setting == "production" and N == 512 and M == M_PROD and p == 9),
                    n_real_images=n_img, E_op_trunc=E_op_trunc)
        if mesh is None:
            mrows.append(mrow)
            continue
        op = CachedGammaH(x, L, K, rc, mesh)
        pd = op.existing_gammah()
        assert op.n_images == n_img
        if dense:
            GLh = lap_full(mode_pair_matrix_fast(mesh, x))
            Gh = GS + GLh
            lh = np.linalg.eigvalsh(Q.T @ (0.5 * (Gh + Gh.T)) @ Q)
            Yd = Gh @ Xm
            mrow.update(E_op=norm2_sym(Gh - G)[0] / nref, E_op_mesh=norm2_sym(GLh - GL)[0] / nref,
                        sym_defect_h=float(np.linalg.norm(Gh - Gh.T) / np.linalg.norm(Gh)),
                        lam_min_h_RangePi=float(lh[0]), n_negative_RangePi=int(np.sum(lh < 0)))
            del Gh, GLh
        dmax, dtr = 0.0, 0.0
        for k, pr in enumerate(probes):
            v = Xm[:, k]
            assert sha(v) == pr["sha256"]                         # identical input bytes for every M
            yh = pd.matvec(v.copy())                              # production pair-difference action
            yc = op.apply(v.reshape(N, 3)).ravel()                # cached D - H action
            yLh = pd.mesh.graph_action(pd.idx, pd.wt, v.reshape(N, 3), pd.D).ravel()    # Gamma_L^h X
            if dense and pr["probe_class"] != "translation":
                dmax = max(dmax, float(np.linalg.norm(Yd[:, k] - yh) / np.linalg.norm(yh)))
            if dense and pr["probe_class"] == "translation":
                dtr = max(dtr, float(np.linalg.norm(Yd[:, k] - yh)) / nref)
            if pr["probe_class"] == "translation":
                trows.append(dict(head, M=M, probe_id=pr["probe_id"], abs_pairdiff=float(np.linalg.norm(yh)),
                                  abs_cached=float(np.linalg.norm(yc)), abs_mesh_part=float(np.linalg.norm(yLh)),
                                  opnorm_pairdiff=float(np.linalg.norm(yh)) / nref,
                                  opnorm_cached=float(np.linalg.norm(yc)) / nref,
                                  abs_ref=float(np.linalg.norm(Yref[:, k])), opnorm_ref=float(np.linalg.norm(Yref[:, k])) / nref,
                                  abs_D55=float(np.linalg.norm(Y55[:, k])), abs_trunc=float(np.linalg.norm(Ytr[:, k])),
                                  sha256=pr["sha256"][:16]))
                continue
            nr = float(np.linalg.norm(Yref[:, k]))
            ec = float(np.linalg.norm(yh - Yref[:, k]))
            em = float(np.linalg.norm(yLh - YL[:, k])) / nr
            et = float(np.linalg.norm(Ytr[:, k] - Yref[:, k])) / nr
            rows.append(dict(head, M=M, h=L / M, eta=info["eta"], production_point=mrow["production_point"],
                             probe_class=pr["probe_class"], probe_id=pr["probe_id"], probe_seed=pr.get("seed"),
                             eigenvalue=pr.get("eigenvalue"), ref_action_norm=nr, E_complete=ec / nr, E_complete_abs=ec,
                             E_complete_opnorm=ec / nref, E_mesh=em, E_trunc=et,
                             triangle_ok=bool(ec / nr <= em + et + 1e-15),
                             E_complete_cached=float(np.linalg.norm(yc - Yref[:, k])) / nr,
                             cached_vs_pairdiff=float(np.linalg.norm(yc - yh) / np.linalg.norm(yh)),
                             E_D55_vs_ref=float(np.linalg.norm(Y55[:, k] - Yref[:, k])) / nr, sha256=pr["sha256"][:16]))
        if dense:
            mrow.update(dense_vs_matrix_free=dmax, dense_vs_matrix_free_translation_opnorm=dtr)
        mrows.append(mrow)
    return rows, trows, mrows


def action_configs():
    return [name_of(f, N, s) for N in (128, 256) for f in FAMS for s in SEEDS] + [name_of(f, 512, 1) for f in FAMS]


def typed(v):
    """CSV string -> bool / int / float / None (for resuming from saved rows)."""
    if v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def read_rows(name):
    import csv
    with open(RESULTS / name) as fh:
        return [{k: typed(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def rate_fit(Ms, E):
    keep = E >= FIT_FLOOR
    if keep.sum() < 4:
        return None
    lm, le = np.log(np.array(Ms, float)[keep]), np.log(E[keep])
    b, a = np.polyfit(lm, le, 1)
    r2 = 1 - np.sum((le - (a + b * lm)) ** 2) / np.sum((le - le.mean()) ** 2)
    return dict(M_lo=int(np.array(Ms)[keep][0]), M_hi=int(np.array(Ms)[keep][-1]), n_points=int(keep.sum()),
                rate_lsq=float(-b), r2=float(r2))


def write_action_fits():
    """Least-squares rates of the saved action errors (maximum over the tested probes, points >= FIT_FLOOR), per N and
    cutoff setting for all families pooled and for each family, and for the p study; no recomputation."""
    rows = read_rows("action_errors.csv")
    fits = []
    for setting in SETTINGS:
        for N in (128, 256, 512):
            for fam in ("all",) + FAMS:
                for cls in ("gaussian", "spectral_slow", "spectral_fast"):
                    for key in ("E_complete", "E_mesh"):
                        sub = [q for q in rows if q["N"] == N and q["setting"] == setting and q["probe_class"] == cls
                               and fam in ("all", q["family"])]
                        Ms = sorted({q["M"] for q in sub})
                        f = rate_fit(Ms, np.array([max(q[key] for q in sub if q["M"] == M) for M in Ms]))
                        if f is not None:
                            fits.append(dict(N=N, setting=setting, family=fam, probe_class=cls,
                                             quantity=key + " (max over tested probes/configs)", **f))
    C.write_csv(RESULTS / "action_rate_fits.csv", fits)
    if (RESULTS / "action_p.csv").exists():
        prow = read_rows("action_p.csv")
        pf = []
        for setting in SETTINGS:
            for p in P_STUDY:
                for cls in ("gaussian", "spectral_slow"):
                    for key in ("E_complete", "E_mesh"):
                        sub = [q for q in prow if q["setting"] == setting and q["p"] == p and q["probe_class"] == cls]
                        Ms = sorted({q["M"] for q in sub})
                        f = rate_fit(Ms, np.array([max(q[key] for q in sub if q["M"] == M) for M in Ms]))
                        if f is not None:
                            pf.append(dict(setting=setting, p=p, probe_class=cls, quantity=key, **f))
        C.write_csv(RESULTS / "action_p_rate_fits.csv", pf)


def stage_action_fits(args):
    write_action_fits()


def stage_action(args):
    """Panel (c): p = 9, production and strict cutoffs, all action configurations, dense E_op for context."""
    rows, trows, mrows, meta = [], [], [], {}
    if args.resume and (RESULTS / "action_mesh.csv").exists():
        rows, trows, mrows = (read_rows(f"action_{k}.csv") for k in ("errors", "translation", "mesh"))
        meta = C.read_json(RESULTS / "action_probes.json")["configs"]
    done = {q["config"] for q in mrows}
    for name in action_configs():
        if name in done:
            continue
        t0 = time.perf_counter()
        x, L = load_cfg(name)
        N = len(x)
        G = gref(name, x, L)
        GD = np.load(C.RAW / f"exp1_{name}_GD55.npy")
        probes, spec = action_probes(N, G)
        if name == MAIN_C:
            np.savez_compressed(RESULTS / f"action_probes_{name}.npz", X=np.stack([pr["v"] for pr in probes], 1),
                                ids=np.array([pr["probe_id"] for pr in probes]))
        meta[name] = dict(N=N, L=L, norm2_ref=norm2_sym(G)[0], spectral=spec,
                          probes=[{k: v for k, v in pr.items() if k != "v"} for pr in probes], settings={})
        for setting in SETTINGS:
            xi, rc, kc = setting_params(setting, L)
            n_c, n_exp = shell_count(kc, L)
            meta[name]["settings"][setting] = dict(xi=xi, rc=rc, kc=kc, n_shell=n_c, n_modes=n_exp,
                                                   n_modes_if_kc_rounded_to_9p3585=int(len(lattice_modes(
                                                       L, kc_shell_below(9.3585, L))[0])) if setting == "production" else None)
            r, t, m = action_case(name, x, L, G, GD, probes, setting, P_ACTION, M_LIST[N], dense=True)
            rows += r
            trows += t
            mrows += m
        g = [q for q in rows if q["config"] == name and q["probe_class"] == "gaussian"]
        C.log(f"action {name}: " + "; ".join(
            f"{st}: " + " ".join(f"{M}:{max(q['E_complete'] for q in g if q['setting'] == st and q['M'] == M):.1e}/"
                                 f"{max(q['E_mesh'] for q in g if q['setting'] == st and q['M'] == M):.1e}"
                                 for M in sorted({q['M'] for q in g if q['setting'] == st}))
            for st in SETTINGS) + f"  (Gaussian max complete/mesh; {time.perf_counter() - t0:.0f} s)")
        C.write_csv(RESULTS / "action_errors.csv", rows)
        C.write_csv(RESULTS / "action_translation.csv", trows)
        C.write_csv(RESULTS / "action_mesh.csv", mrows)
    # reference check of the main configuration against a stricter LatticeFriction
    x, L = load_cfg(MAIN_C)
    G = gref(MAIN_C, x, L)
    lat = LatticeFriction(L, kernel=C.LAW, gamma=C.GAMMA, kappa=C.KAPPA, r_ref=C.R_REF, rel_tol=1e-15, near_cells=3,
                          degree=20)
    Gs = C.lattice_matrix(lat, x)
    Xm = np.stack([pr["v"] for pr in action_probes(len(x), G)[0]], 1)
    dY = np.linalg.norm((Gs - G) @ Xm, axis=0) / np.maximum(np.linalg.norm(G @ Xm, axis=0), 1e-300)
    meta[MAIN_C]["strict_lattice_check"] = dict(rel_tol=1e-15, near_cells=3, degree=20,
                                                op_rel=norm2_sym(Gs - G)[0] / norm2_sym(G)[0],
                                                action_rel_max_nonnull=float(dY[:-3].max()))
    C.write_json(RESULTS / "action_probes.json", dict(
        settings=SETTINGS, p=P_ACTION, M_list={str(k): v for k, v in M_LIST.items()}, gauss_seeds=GAUSS_SEEDS,
        production_source="results_psearch/optimum_strong_N512_s1_main.json", M_production=M_PROD,
        configs=meta, provenance=C.provenance()))
    write_action_fits()
    v = [q for q in mrows if q["valid"]]
    C.write_json(RESULTS / "action_summary.json", dict(
        n_cases=len(mrows), n_valid=len(v), invalid=[dict(config=q["config"], setting=q["setting"], M=q["M"],
                                                          reason=q["reason"]) for q in mrows if not q["valid"]],
        max_n_excluded=max(q["n_excluded"] for q in v), min_W2min=min(q["W2min"] for q in v),
        max_dense_vs_matrix_free=max(q["dense_vs_matrix_free"] for q in v),
        max_sym_defect_h=max(q["sym_defect_h"] for q in v), any_negative=any(q["n_negative_RangePi"] > 0 for q in v),
        all_triangle_ok=all(q["triangle_ok"] for q in rows),
        max_cached_vs_pairdiff=max(q["cached_vs_pairdiff"] for q in rows),
        max_translation_opnorm_pairdiff=max(q["opnorm_pairdiff"] for q in trows),
        max_translation_opnorm_cached=max(q["opnorm_cached"] for q in trows),
        max_translation_opnorm_ref=max(q["opnorm_ref"] for q in trows), fit_floor=FIT_FLOOR,
        provenance=C.provenance()))


def stage_action_p(args):
    """Supplementary: mesh-only and complete action error for p = 4, 6, 8, 9 on the main configuration, both cutoff
    settings, every other parameter identical (no dense assembly)."""
    x, L = load_cfg(MAIN_C)
    N = len(x)
    G = gref(MAIN_C, x, L)
    GD = np.load(C.RAW / f"exp1_{MAIN_C}_GD55.npy")
    probes, _ = action_probes(N, G)
    probes = [pr for pr in probes if pr["probe_class"] != "translation"]
    rows, mrows = [], []
    for setting in SETTINGS:
        for p in P_STUDY:
            r, _, m = action_case(MAIN_C, x, L, G, GD, probes, setting, p, M_LIST[N], dense=False)
            rows += r
            mrows += m
            g = [q for q in r if q["probe_class"] == "gaussian"]
            C.log(f"action_p {setting} p={p}: " + " ".join(
                f"{M}:{max(q['E_mesh'] for q in g if q['M'] == M):.1e}" for M in sorted({q['M'] for q in g})))
            C.write_csv(RESULTS / "action_p.csv", rows)
            C.write_csv(RESULTS / "action_p_mesh.csv", mrows)
    write_action_fits()


STAGES = dict(configs=stage_configs, refs=stage_refs, split=stage_split, shortrange=stage_shortrange,
              action=stage_action, action_p=stage_action_p, action_fits=stage_action_fits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=["all"])
    ap.add_argument("--resume", action="store_true", help="action: keep saved configurations, compute only missing ones")
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
