"""
Shared machinery of the two new spatial-accuracy experiments (accuracy_sweeps/, candidate2_pilot/).

Physical model (amplitude fixed to c = gamma_0 l_0 = 1; the old normalisation c = gamma r_ref e^{kappa r_ref} of
common.py is NOT used here):
    K_kappa(r) = c e^{-kappa r}/r P_r,  P_r = r r^T/r^2,   c = AMP = 1
    (Gamma X)_i = sum_{j != i} sum_{n in Z^3} K(q_i - q_j + nL)(X_i - X_j)   full periodic lattice, k = 0 retained,
    self-particle edges excluded, no minimum image.

Ewald splittings at s = xi^2 (theta = 0: Candidate 1, the existing positive sharp Laplace split; theta = 1/2:
Candidate 2, isotropic Gaussian-corrected split):
    K_S^theta = K_S^0 - c theta G(r) I,    K_L^theta = K_L^0 + c theta G(r) I,    G(r) = g_3(xi^2) e^{-xi^2 r^2},
    Khat_L^theta(k) = Khat_L^0(k) + c theta g_3(xi^2) (pi/xi^2)^{3/2} e^{-k^2/(4 xi^2)} I     (k = 0 included).
g_3(s) is the Laplace density of e^{-kappa r}/r^3 in s = r^2, e^{-kappa r}/r^3 = int_0^inf g_3(s) e^{-s r^2} ds, i.e.
the project's rho_kappa(s) (verify_ewald_longitudinal.rho); K_S^0 = c theta_S(r) P_r with theta_S(r) ~ g_3(xi^2)
e^{-xi^2 r^2} as r -> infinity, so theta = 1/2 balances the longitudinal (+) and transverse (-) eigenvalues of the
leading real-space tail. K_S^0, K_L^0, the mesh, the B-spline assignment, the FFT normalisation and the degree /
self-cancellation logic are the existing ones (radial_kernels.RadialEwald law A with amp = 1, test_parameter_selection
.Mesh, cached_operator.CachedGammaH); Candidate 2 only adds the isotropic term to the real-space weights/degree and a
constant (in the e^{k^2/4xi^2}-scaled form) to A(k) of the influence function.

Dimensionless cutoffs: s_R = xi r_c, s_F = k_c/(2 xi); k_c is snapped to the largest reciprocal-lattice shell
2 pi sqrt(n)/L <= 2 xi s_F (exact float; the realised s_F is recorded).

References (cached in raw/exp_refs_c1/, keyed by kappa, c, rho, L, configuration name and position hash):
    Gamma_D   unsplit direct periodic image sum to R_D (direct_periodic.annulus_tensors, the convergence-test form of
              DirectPairs: same image set, accumulated per annulus so Gamma_D(R) is available on a radius grid).
              R_D per (kappa, rho, configuration) = smallest R on a 2.5 grid with the certified bound
              2 tail_bound_per_particle(R)/||Gamma||_2 <= REF_CERT_TOL.
    Gamma_LF  prl_toy_models.LatticeFriction (near images direct + Chebyshev far field) with rel_tol 1e-10, 1e-12,
              1e-14 (amplitude 1 via gamma = e^{-kappa}, r_ref = 1; c - 1 is recorded, a few ulp).
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import common as C                       # noqa: E402  (thread settings before numpy)
import numpy as np                       # noqa: E402
from scipy import linalg                 # noqa: E402

import radial_kernels as rk              # noqa: E402
import direct_periodic as DP             # noqa: E402
from verify_ewald_longitudinal import rho as rho_density  # noqa: E402
from prl_toy_models import LatticeFriction  # noqa: E402
from cached_operator import CachedGammaH  # noqa: E402
from clustered_configurations import min_periodic_separation  # noqa: E402
from parameter_optimizer import mesh_modes  # noqa: E402
from test_parameter_selection import lap_full  # noqa: E402
from exp1_dense_reference import mode_pair_matrix_fast  # noqa: E402
from run_exp1 import action_probes, projector_basis, norm2_sym, check_mesh, sha, shell_count  # noqa: E402

AMP = 1.0                                # c = gamma_0 l_0 = 1, exactly
N0 = 256
FAMS = ("uniform", "moderate", "strong")
SEEDS = (1, 2, 3)
RHO_BASE_CONFIGS = 0.75                  # density of the committed configs/*.npz
BASE = dict(kappa=0.7, rho=0.75, xi=0.85, sR=6.0, sF=6.0, M=80, p=9, theta=0.0)
MODERATE = dict(BASE, sR=4.0, sF=4.0, M=40, p=7)
REF_CERT_TOL = 1e-13                     # certified relative truncation bound of Gamma_D
LF_TOLS = (1e-10, 1e-12, 1e-14)
REF_DIR = C.RAW / "exp_refs_c1"
EXP_DIR = HERE
RC_FAR_S = 8.0                           # s_R of the "untruncated" real-space operator (e^{-64} tail)
SF_FAR = 7.0                             # s_F of the "untruncated" mesh-free Fourier operator
TIMING_REPS = 15


def g_phys(kappa, amp=AMP):
    """Radial profile of the physical kernel, g(r) = c e^{-kappa r}/r (c = 1)."""
    kappa, amp = float(kappa), float(amp)
    return lambda r: amp * np.exp(-kappa * np.asarray(r, float)) / np.asarray(r, float)


# ============================================================================ configurations
def base_config_path(fam, seed):
    return HERE / "configs" / f"{fam}_N{N0}_s{seed}.npz"


def config(fam, seed, rho):
    """Configuration (fam, seed) at density rho: the committed rho = 0.75 positions rescaled isotropically by
    L(rho)/L(0.75) (L = (N/rho)^{1/3}), so the geometry in box units is identical for every density; the minimum
    separation scales by the same factor and is recorded."""
    with np.load(base_config_path(fam, seed)) as d:
        q0, L0 = np.mod(d["q"].astype(float), float(d["L"])), float(d["L"])
    assert abs(N0 / L0 ** 3 - RHO_BASE_CONFIGS) < 1e-12
    L = (N0 / float(rho)) ** (1.0 / 3.0)
    q = q0 if float(rho) == RHO_BASE_CONFIGS else np.mod(q0 * (L / L0), L)
    name = f"{fam}_N{N0}_s{seed}_rho{float(rho):g}"
    return dict(name=name, family=fam, seed=int(seed), rho=float(rho), N=N0, L=float(L), q=q,
                scale=float(L / L0), hash=sha(np.concatenate([q.ravel(), [L]]))[:12],
                d_min=min_periodic_separation(q, L), generation="rescaled" if float(rho) != 0.75 else "original")


def config_record(cfg):
    return {k: v for k, v in cfg.items() if k != "q"}


# ============================================================================ references
def lattice_friction(L, kappa, rel_tol, degree=16, near_cells=2):
    """Independent unsplit reference with amplitude 1: radial_friction(r, A, gamma, kappa, r_ref) =
    gamma (r_ref/r) e^{-kappa (r - r_ref)}, gamma = e^{-kappa}, r_ref = 1 -> e^{-kappa r}/r."""
    return LatticeFriction(L, kernel="A", gamma=float(np.exp(-kappa)), kappa=float(kappa), r_ref=1.0,
                           rel_tol=rel_tol, degree=degree, near_cells=near_cells)


def lf_amplitude(kappa):
    return float(np.exp(-kappa) * 1.0 * np.exp(kappa * 1.0))


def ref_key(cfg, kappa):
    return f"{cfg['name']}_kap{float(kappa):g}_c{AMP:g}_L{cfg['L']:.10f}_{cfg['hash']}"


def ref_path(cfg, kappa):
    return REF_DIR / f"{ref_key(cfg, kappa)}.npz"


def choose_RD(N, L, kappa, norm_est, tol=REF_CERT_TOL, step=2.5):
    g = g_phys(kappa)
    R = 10.0
    while 2 * DP.tail_bound_per_particle(N, L, R, g=g) / norm_est > tol:
        R += step
    return R


def compute_reference(cfg, kappa, say=print):
    """Gamma_D(R_D) with its convergence record and Gamma_LF at increasing accuracy."""
    q, L, N = cfg["q"], cfg["L"], cfg["N"]
    g = g_phys(kappa)
    rec = dict(key=ref_key(cfg, kappa), config=cfg["name"], family=cfg["family"], seed=cfg["seed"],
               rho=cfg["rho"], kappa=float(kappa), c=AMP, N=N, L=L, hash=cfg["hash"], d_min=cfg["d_min"],
               c_lattice_friction=lf_amplitude(kappa), c_lattice_friction_minus_1=lf_amplitude(kappa) - 1.0)
    GLF = {}
    for tol in LF_TOLS:
        t0 = time.perf_counter()
        lat = lattice_friction(L, kappa, tol)
        GLF[tol] = C.lattice_matrix(lat, q)
        rec[f"t_LF_{tol:g}_s"] = time.perf_counter() - t0
        rec[f"LF_{tol:g}_r_tail"] = lat.r_tail
    nrm_est = float(np.linalg.eigvalsh(GLF[LF_TOLS[-1]])[-1])
    R_D = choose_RD(N, L, kappa, nrm_est)
    edges = np.round(R_D * np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]), 6)
    t0 = time.perf_counter()
    pi, pj, acc, count = DP.annulus_tensors(q, L, edges, g=g)
    t_D = time.perf_counter() - t0
    cum = np.cumsum(acc, axis=0)
    GD = DP.tensors_to_dense(pi, pj, cum[-1], N)
    nrm, _ = norm2_sym(GD)
    conv = []
    for k, R in enumerate(edges):
        Gk = DP.tensors_to_dense(pi, pj, cum[k], N)
        conv.append(dict(R=float(R), diff_to_RD_rel=norm2_sym(Gk - GD)[0] / nrm,
                         cert_bound_rel=2 * DP.tail_bound_per_particle(N, L, float(R), g=g) / nrm,
                         n_images=int(count[:k + 1].sum())))
    Q, T = projector_basis(N)
    lamPi = np.linalg.eigvalsh(Q.T @ (0.5 * (GD + GD.T)) @ Q)
    rec.update(R_D=float(R_D), t_direct_s=t_D, n_images_direct=int(count.sum()), norm2=nrm,
               cert_bound_rel=conv[-1]["cert_bound_rel"],
               conv_R_0p9_rel=conv[-3]["diff_to_RD_rel"], conv_R_0p95_rel=conv[-2]["diff_to_RD_rel"],
               sym_rel=float(np.linalg.norm(GD - GD.T) / np.linalg.norm(GD)),
               null_rel=float(np.linalg.norm(GD @ T) / nrm), lam_min_RangePi=float(lamPi[0]),
               lam_max=float(lamPi[-1]), k0_eigenvalue=float(cfg["rho"] * 4 * np.pi * AMP / kappa ** 2 / 3))
    for tol in LF_TOLS:
        rec[f"LF_{tol:g}_vs_D_rel"] = norm2_sym(GLF[tol] - GD)[0] / nrm
    rec["LF_1e-12_vs_1e-14_rel"] = norm2_sym(GLF[1e-12] - GLF[1e-14])[0] / nrm
    REF_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ref_path(cfg, kappa).with_suffix(".tmp.npz")
    np.savez(tmp, G_D=GD, G_LF=GLF[LF_TOLS[-1]], conv=np.array([[c["R"], c["diff_to_RD_rel"], c["cert_bound_rel"]]
                                                                  for c in conv]))
    os.replace(tmp, ref_path(cfg, kappa))
    say(f"ref {rec['key']}: R_D {R_D:g} ({rec['n_images_direct']:.3g} images, {t_D:.0f} s), ||G|| {nrm:.4g}, "
        f"cert {rec['cert_bound_rel']:.1e}, |G(0.95R_D)-G(R_D)| {rec['conv_R_0p95_rel']:.1e}, "
        f"|LF14 - D| {rec['LF_1e-14_vs_D_rel']:.1e}, |LF10 - D| {rec['LF_1e-10_vs_D_rel']:.1e}")
    return GD, GLF[LF_TOLS[-1]], rec, conv


def load_reference(cfg, kappa):
    """(Gamma_D, Gamma_LF(1e-14)) from the cache; never reused across kappa, rho, configuration or amplitude
    (all are part of the key, and the position hash is part of the file name)."""
    f = ref_path(cfg, kappa)
    if not f.exists():
        raise FileNotFoundError(f"reference {f.name} missing: run the 'refs' stage first")
    with np.load(f) as d:
        return d["G_D"], d["G_LF"]


# ============================================================================ candidate kernels
class ThetaKernel:
    """Ewald split kernel with the Gaussian correction theta (theta = 0: existing split, unchanged).

    Interface used by Mesh / CachedGammaH: xi, kap, theta_S_scaled_closed (longitudinal K_S^0 profile, scaled by
    e^{xi^2 r^2}), AB_scaled (Khat_L^theta scaled by e^{k^2/4xi^2}). Isotropic real-space part: iso_S(r) I."""

    def __init__(self, kappa, xi, theta, amp=AMP):
        self.rad = rk.RadialEwald("A", kappa, xi, amp)
        self.law, self.amp, self.xi, self.kap, self.theta = "A", float(amp), self.rad.xi, self.rad.kap, float(theta)
        self.g3 = float(rho_density(self.xi ** 2, self.kap))
        assert abs(self.g3 - self.rad.base.rho_xi2) <= 1e-15 * self.g3
        self.beta0 = self.amp * self.theta * self.g3                      # K_S^theta = K_S^0 - beta0 e^{-xi^2r^2} I
        self.A_shift_scaled = self.beta0 * (np.pi / self.xi ** 2) ** 1.5  # added to A(k) e^{k^2/4xi^2}

    def theta_full(self, r):
        return self.rad.theta_full(r)

    def theta_S_scaled_closed(self, r):
        return self.rad.theta_S_scaled_closed(r)

    def theta_S(self, r):
        return self.rad.theta_S(r)

    def theta_L(self, r):
        return self.rad.theta_L(r)

    def iso_S(self, r):
        return -self.beta0 * np.exp(-(self.xi * np.asarray(r, float)) ** 2)

    def iso_L(self, r):
        return self.beta0 * np.exp(-(self.xi * np.asarray(r, float)) ** 2)

    def AB_scaled(self, k):
        A, B = self.rad.AB_scaled(k)
        return A + self.A_shift_scaled, B

    def AB(self, k):
        A, B = self.AB_scaled(k)
        dmp = np.exp(-np.asarray(k, float) ** 2 / (4 * self.xi ** 2))
        return A * dmp, B * dmp


_KC = {}


def kernel(kappa, xi, theta):
    key = (round(float(kappa), 12), round(float(xi), 12), round(float(theta), 12))
    if key not in _KC:
        _KC[key] = ThetaKernel(kappa, xi, theta)
    return _KC[key]


# ============================================================================ operator
class ThetaOp(CachedGammaH):
    """CachedGammaH (existing cached D - H action) plus the isotropic short-range term beta(r) I of K_S^theta:
    D_i += sum_images beta I,  H_i += sum_images beta X_j.  theta = 0 adds nothing (identical to CachedGammaH)."""

    def __init__(self, x, L, K, rc, mesh, images=None):
        super().__init__(x, L, K, rc, mesh, images)
        t0 = time.perf_counter()
        self.iso = K.theta != 0.0
        if self.iso:
            r = np.linalg.norm(self.d, axis=1)
            self.beta = K.iso_S(r)
            deg = np.bincount(self.i, self.beta, self.N) + np.bincount(self.j, self.beta, self.N)
            self.DS = self.DS + deg[:, None, None] * np.eye(3)
            self.D = self.DS + self.Dm
        t = time.perf_counter() - t0
        self.t_setup_parts["iso_weights_degree"] = t
        self.t_setup += t

    def real_H(self, X):
        """theta = 0: CachedGammaH.real_H unchanged. theta != 0: the same scatter, with the isotropic term beta X_j
        (beta X_i) added to the per-image contribution before the bincount, so no extra pass over the images."""
        if not self.iso:
            return super().real_H(X)
        Xi, Xj = X[self.i], X[self.j]
        a = np.einsum("pa,pa->p", self.d, Xj)
        b = np.einsum("pa,pa->p", self.d, Xi)
        fi = a[:, None] * self.wd + self.beta[:, None] * Xj
        fj = b[:, None] * self.wd + self.beta[:, None] * Xi
        N = self.N
        return np.stack([np.bincount(self.i, fi[:, c], N) + np.bincount(self.j, fj[:, c], N) for c in range(3)],
                        axis=1)


def snap_kc(kc_req, L):
    """Largest reciprocal shell radius 2 pi sqrt(n)/L <= kc_req (exact; n a sum of three squares)."""
    n = int(np.floor((kc_req * L / (2 * np.pi)) ** 2 * (1 + 1e-12)))
    while n > 0:
        mv, n2, _ = mesh_modes(2 * np.pi / L * np.sqrt(n) * (1 + 1e-12), L)
        if np.any(n2 == n):
            return float(2 * np.pi / L * np.sqrt(n)), n
        n -= 1
    raise ValueError("no shell")


def cutoffs(par, L):
    xi = par["xi"]
    rc = par["sR"] / xi
    kc, n_shell = snap_kc(2 * xi * par["sF"], L)
    return rc, kc, n_shell


def real_dense(op):
    """Dense Gamma_S^{rc,theta} from the operator's own image list and weights (independent of the action code)."""
    N = op.N
    flat = op.i.astype(np.int64) * N + op.j
    W = np.empty((N, N, 3, 3))
    for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        v = op.wd[:, a] * op.d[:, b]
        if a == b and op.iso:
            v = v + op.beta
        W[:, :, a, b] = W[:, :, b, a] = np.bincount(flat, v, N * N).reshape(N, N)
    return lap_full(W + W.transpose(1, 0, 2, 3))


def fourier_dense(x, L, K, kc):
    """Mesh-free truncated lattice Fourier operator (1/V) sum_{|k| <= kc} Khat_L^theta(k) e^{ik.(q_i - q_j)}."""
    mv, _, _ = mesh_modes(kc, L)
    kv = 2 * np.pi / L * mv
    kn = np.linalg.norm(kv, axis=1)
    ku, inv = np.unique(kn, return_inverse=True)
    A, B = K.AB(ku)
    A, B = A[inv], B[inv]
    ph = x @ kv.T
    Cs, Sn = np.cos(ph), np.sin(ph)
    N = len(x)
    W = np.empty((N, N, 3, 3))
    for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        gg = A * (a == b) + B * kv[:, a] * kv[:, b]
        W[:, :, a, b] = W[:, :, b, a] = ((Cs * gg) @ Cs.T + (Sn * gg) @ Sn.T) / L ** 3
    return lap_full(W)


class MeshCache:
    """Parameter-only meshes (influence function) keyed by (L, kappa, xi, theta, kc, M, p)."""

    def __init__(self):
        self.m = {}

    def get(self, L, K, kc, M, p):
        key = (round(L, 12), K.kap, K.xi, K.theta, round(kc, 12), int(M), int(p))
        if key not in self.m:
            t0 = time.perf_counter()
            mesh, info = check_mesh(L, K, kc, M, p)
            info["t_mesh_build_s"] = time.perf_counter() - t0
            self.m[key] = (mesh, info)
        return self.m[key]


def probes_for(G):
    N = G.shape[0] // 3
    pr, spec = action_probes(N, G)
    return pr, spec


def evaluate(cfg, G, GLF, probes, par, meshes, timing=True, dense=True, meshfree=False, extra=None):
    """One (configuration, parameter set, candidate): operator errors, action errors and structure."""
    x, L, N = cfg["q"], cfg["L"], cfg["N"]
    K = kernel(par["kappa"], par["xi"], par["theta"])
    rc, kc, n_shell = cutoffs(par, L)
    row = dict(config=cfg["name"], family=cfg["family"], seed=cfg["seed"], rho=cfg["rho"], N=N, L=L,
               kappa=par["kappa"], c=AMP, xi=par["xi"], theta=par["theta"], sR=par["sR"], sF_req=par["sF"],
               sF_eff=kc / (2 * par["xi"]), rc=rc, kc=kc, kc_shell_n=n_shell, M=par["M"], p=par["p"])
    if extra:
        row.update(extra)
    mesh, info = meshes.get(L, K, kc, par["M"], par["p"])
    row.update(mesh_valid=info["valid"], mesh_reason=info["reason"], h=L / par["M"], eta=info.get("eta"),
               n_modes=info.get("n_modes"), t_mesh_build_s=info["t_mesh_build_s"])
    if mesh is None:
        return row
    nref = norm2_sym(G)[0]
    t0 = time.perf_counter()
    op = ThetaOp(x, L, K, rc, mesh)
    t_setup_wall = time.perf_counter() - t0
    row.update(n_images=op.n_images, t_setup_s=op.t_setup, t_setup_wall_s=t_setup_wall,
               **{f"t_setup_{k}_s": v for k, v in op.t_setup_parts.items()})
    Xm = np.stack([p_["v"] for p_ in probes], 1)
    Y = np.stack([op.apply(Xm[:, k].reshape(N, 3)).ravel() for k in range(Xm.shape[1])], 1)
    Yref, YLF = G @ Xm, GLF @ Xm
    cls = np.array([p_["probe_class"] for p_ in probes])
    gs, sl, fa, tr = cls == "gaussian", cls == "spectral_slow", cls == "spectral_fast", cls == "translation"
    rel = np.linalg.norm(Y - Yref, axis=0) / np.maximum(np.linalg.norm(Yref, axis=0), 1e-300)
    relLF = np.linalg.norm(Y - YLF, axis=0) / np.maximum(np.linalg.norm(YLF, axis=0), 1e-300)
    row.update(E_GX=float(np.linalg.norm(Y[:, gs] - Yref[:, gs]) / np.linalg.norm(Yref[:, gs])),
               E_GX_vsLF=float(np.linalg.norm(Y[:, gs] - YLF[:, gs]) / np.linalg.norm(YLF[:, gs])),
               E_GX_gauss_max=float(rel[gs].max()), E_GX_gauss_median=float(np.median(rel[gs])),
               E_GX_slow_max=float(rel[sl].max()), E_GX_fast=float(rel[fa].max()),
               E_GX_slow_max_vsLF=float(relLF[sl].max()),
               mom_abs_max=float(np.linalg.norm(Y[:, tr], axis=0).max()),
               mom_opnorm_max=float(np.linalg.norm(Y[:, tr], axis=0).max() / nref),
               ref_floor_GX=float(np.linalg.norm(YLF[:, gs] - Yref[:, gs]) / np.linalg.norm(Yref[:, gs])))
    if dense:
        GS = real_dense(op)
        GLh = lap_full(mode_pair_matrix_fast(mesh, op.x))
        Gh = GS + GLh
        Q, _ = projector_basis(N)
        lh = np.linalg.eigvalsh(Q.T @ (0.5 * (Gh + Gh.T)) @ Q)
        Yd = Gh @ Xm
        nz = ~tr
        row.update(E_G=norm2_sym(Gh - G)[0] / nref, E_G_vsLF=norm2_sym(Gh - GLF)[0] / nref,
                   ref_floor_G=norm2_sym(GLF - G)[0] / nref,
                   sym_resid=float(np.linalg.norm(Gh - Gh.T) / np.linalg.norm(Gh)),
                   lam_min_h=float(lh[0]), n_negative_h=int(np.sum(lh < 0)),
                   lam_min_ref=float(np.linalg.eigvalsh(Q.T @ (0.5 * (G + G.T)) @ Q)[0]),
                   mf_vs_dense=float((np.linalg.norm(Y[:, nz] - Yd[:, nz], axis=0) /
                                      np.linalg.norm(Yd[:, nz], axis=0)).max()))
        if meshfree:
            GF = fourier_dense(op.x, L, K, kc)
            Gt = GS + GF
            row.update(E_G_meshfree=norm2_sym(Gt - G)[0] / nref,
                       E_GX_meshfree=float(np.linalg.norm((Gt - G) @ Xm[:, gs]) / np.linalg.norm(Yref[:, gs])),
                       E_G_mesh_only=norm2_sym(GLh - GF)[0] / nref)
        if K.theta != 0.0 and op.n_images:
            r = np.linalg.norm(op.d, axis=1)
            tS = K.theta_S_scaled_closed(r) * np.exp(-(K.xi * r) ** 2)
            row.update(KS_min_transverse_eig=float(op.beta.min()), KS_max_long_eig=float((tS + op.beta).max()),
                       KS_min_long_eig=float((tS + op.beta).min()))
    if timing:
        X0 = Xm[:, 0].reshape(N, 3).copy()
        t = C.median_time(lambda: op.apply(X0), TIMING_REPS, warm=2)
        st = C.tstats(t)
        row.update(t_matvec_s=st["median"], t_matvec_iqr_rel=st["iqr_rel"])
    return row


# ============================================================================ io helpers
def write_rows(path, rows):
    C.write_csv(path, rows)


def read_csv(path):
    import csv
    from run_exp1 import typed
    with open(path) as fh:
        return [{k: typed(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def log_cmd(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "commands.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}  cd {Path.cwd().relative_to(C.REPO) if str(Path.cwd()).startswith(str(C.REPO)) else Path.cwd()}"
                 f" && {' '.join(['python3'] + sys.argv)}\n")


def provenance(**kw):
    p = C.provenance(**kw)
    p["amplitude_c"] = AMP
    return p


def amplitude_checks():
    """c = 1 in every path: physical g, RadialEwald, theta_full, LatticeFriction, direct sum."""
    r = np.geomspace(0.3, 30, 200)
    out = {}
    for kap in (0.2, 0.5, 0.7, 1.0, 2.0):
        K = kernel(kap, 0.85, 0.0)
        exact = np.exp(-kap * r) / r
        import prl_toy_models as v2
        lf = v2.radial_friction(r, kernel="A", gamma=np.exp(-kap), kappa=kap, r_ref=1.0)
        out[f"kappa{kap:g}"] = dict(
            RadialEwald_amp=K.rad.amp,
            theta_full_max_rel=float(np.max(np.abs(K.theta_full(r) - exact) / exact)),
            direct_g_max_rel=float(np.max(np.abs(g_phys(kap)(r) - exact) / exact)),
            lattice_friction_g_max_rel=float(np.max(np.abs(lf - exact) / exact)),
            lattice_friction_c_minus_1=lf_amplitude(kap) - 1.0,
            split_S_plus_L_max_rel=float(np.max(np.abs(K.theta_S(r) + K.theta_L(r) - exact) / exact)),
            ghat0_vs_4pi_over_kappa2=float(K.rad.ghat0() / (4 * np.pi / kap ** 2) - 1))
    return out


__all__ = ["AMP", "BASE", "MODERATE", "FAMS", "SEEDS", "config", "compute_reference", "load_reference", "kernel",
           "ThetaKernel", "ThetaOp", "evaluate", "MeshCache", "probes_for", "fourier_dense", "real_dense", "cutoffs"]
