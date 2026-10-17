#!/usr/bin/env python3
"""
Theory-guided parameter selection and geometry-aware cost model for Method A at a relative operator budget eps.

ABSOLUTE spectral-norm bounds (exact arithmetic, exact kernel coefficients), Gamma_h - Gamma_ref =
-(tail of Gamma_S beyond rc) + (Gamma_L^{kc} - Gamma_L) + (Gamma_L^{mesh} - Gamma_L^{kc}) + coefficient errors:

  eps_S(rc; q)  = 2 max_i lambda_max( sum_{(i,j,n): |d_ijn| > rc} K_S(d_ijn) )                     [configuration]
                  The discarded images form a graph Laplacian with PSD edge weights K_S = theta_S rhat rhat^T, and
                  (v_i - v_j)^T B (v_i - v_j) <= 2 v_i^T B v_i + 2 v_j^T B v_j gives ||Lap|| <= 2 max_i ||D_i||.
                  Images with |d| <= R_far are summed explicitly; beyond R_far, theta_S(r) <= c e^{-xi^2 r^2}
                  (e^{-z^2}/r + rho_kappa(xi^2)) (erfc(x) <= e^{-x^2}) and the image count within r is at most
                  N (4 pi/3)(r + sqrt(3) L/2)^3 / L^3.
  eps_F(kc)     = 2 rho sum_{k in (2 pi/L) Z^3, |k| > kc} ||Khat_L(k)||_2                            [a priori]
                  The removed modes give the pair matrix W_R = V^-1 sum_k e_k e_k^H (x) Khat_L(k), ||e_k e_k^H|| = N,
                  and its degree D_R,i = V^-1 sum_k e_ki (sum_j conj e_kj) Khat_L; both are <= rho sum ||Khat_L||.
                  The sum runs over the actual reciprocal lattice (shell by shell), so the discrete cutoff is exact:
                  kc is always a shell radius. ||Khat_L|| = max(|A|, |A + B k^2|).
  eps_mesh(M,p) = 2 rho sum_{|k| <= kc} alpha(k) (2 + alpha(k)) ||Khat_L(k)||_2,   alpha(k) = C(p) q(k)^p,
                  C(p) = (1 + 2 zeta(p))^3 - 1,  q(k) = max_a |k_a| h / (2 pi - |k_a| h)                  [a priori]
                  With S_j(k) = W_hat(k) e^{-ik.q_j} (1 + alpha_j(k)) (Poisson sum of the B-spline stencil) and the
                  deconvolved influence Khat_L/W_hat^2, the mesh pair matrix is V^-1 sum_k u_k u_k^H (x) Khat_L,
                  u_k = e_k o (1 + conj alpha_k). Per axis |W_hat(k_a + 2 pi m/h)/W_hat(k_a)| = |theta/(theta + pi m)|^p
                  <= q_a^p |m|^-p for |theta| = |k_a| h/2 < pi/2, so |alpha_j(k)| <= prod_a (1 + 2 zeta(p) q_a^p) - 1
                  <= C(p) max_a q_a^p, and ||u u^H - e e^H|| <= N alpha (2 + alpha) (same for the degree).
                  Excluded influence modes (|W_hat|^2 <= 1e-10) would add 2 rho ||Khat_L||; none occur here.
  eps_quad      numerical ESTIMATE, not a certificate: production vs refined Gauss-Legendre A, B, propagated with
                  2 rho sum (1 + alpha)^2 ||dKhat||; theta_S closed form vs its s-integral on a radial grid.

Relative target: eps_S + eps_F + eps_mesh + eps_quad <= eps * Lambda_lb, Lambda_lb = max_i lambda_max(Gamma_ref,ii)
<= ||Gamma_ref||_2 (a diagonal block of a PSD matrix), so dividing by Lambda_lb never understates the relative bound.
Budget split (fixed in advance): f_S = f_F = f_mesh = 0.33, f_quad = 0.01.

Selection for each xi: a_tol = xi * min{rc : eps_S <= f_S eps Lambda_lb};  b_tol = kc/(2 xi) with kc the smallest
lattice-shell radius such that eps_F <= f_F eps Lambda_lb;  M = smallest 2,3,5-smooth M with retained modes below
Nyquist (m_c < M/2, eta = kc h/pi < 1: implementation constraints of Mesh/make_mesh) and eps_mesh <= f_mesh eps
Lambda_lb;  p = 7. rc = a_tol/xi, kc = 2 xi b_tol.

Cost model (one cached action, N_img = retained (i < j, n) images, each processed once for both rows):
    T_pred(xi) = C_R N_img(rc) + C_P N p^3 + C_F M^3 log(M^3) + C_0,   coefficients from isolated timings.
"""
import argparse

import numpy as np
from scipy import special

import common as C
from neighbor_images import kdtree_images
from cached_operator import mesh_for, next_smooth
from verify_ewald_longitudinal import graded_rule

SPLIT = dict(S=0.33, F=0.33, mesh=0.33, quad=0.01)
XI_GRID = np.round(np.arange(0.40, 1.6001, 0.025), 4)
A_MAX = 5.5                 # largest a = xi rc searched (e^{-30}); a_tol = NaN would flag a budget beyond it
R_PAD = 1.5                 # explicit image sum up to R_far = (A_MAX + R_PAD)/xi = 7/xi (e^{-49}), analytic tail beyond
FOURIER_FAR = 9.7           # explicit lattice sum up to k_far = 2 xi FOURIER_FAR (e^{-94} relative), tail beyond


def C_p(p):
    return (1.0 + 2.0 * special.zeta(p, 1)) ** 3 - 1.0


# ----------------------------------------------------------------------------- Fourier side
class FourierShells:
    """||Khat_L|| on the reciprocal-lattice shells |m|^2 = n of one box, for one xi."""

    def __init__(self, xi, L):
        self.xi, self.L = float(xi), float(L)
        self.K = C.kernel(xi)
        kfar = 2 * xi * FOURIER_FAR
        mf = int(np.ceil(kfar * L / (2 * np.pi)))
        g = np.arange(-mf, mf + 1)
        n2 = (g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2).ravel()
        n2 = n2[n2 <= (kfar * L / (2 * np.pi)) ** 2]
        self.r3 = np.bincount(n2)                                     # multiplicity of each |m|^2
        self.n = np.nonzero(self.r3)[0]
        self.k = 2 * np.pi / L * np.sqrt(self.n)
        A, B = self.K.AB_scaled(self.k)
        dmp = np.exp(-self.k ** 2 / (4 * xi ** 2))
        A, B = A * dmp, B * dmp
        self.knorm = np.maximum(np.abs(A), np.abs(A + B * self.k ** 2))
        self.kfar = float(self.k[-1])
        w = self.r3[self.n] * self.knorm
        self.tail_after = np.concatenate([np.cumsum(w[::-1])[::-1][1:], [0.0]]) + self.far_tail()   # sum_{n' > n}
        self.knorm_of_n = np.zeros(self.n[-1] + 1)
        self.knorm_of_n[self.n] = self.knorm

    def far_tail(self):
        """Rigorous bound of sum_{|k| > k_far} ||Khat_L||: A(k) <= A(0) e^{-k^2/4xi^2}, |B(k)| <= |B(0)| e^{-k^2/4xi^2}."""
        A0, B0 = self.K.AB_scaled(np.array([0.0]))
        A0, B0 = float(A0[0]), abs(float(B0[0]))
        dk = 2 * np.pi / self.L
        tot, k = 0.0, self.kfar
        for _ in range(100000):
            kout = k + dk
            cnt = 4 / 3 * np.pi * (kout / dk + np.sqrt(3) / 2) ** 3          # lattice points with |k| <= kout
            term = cnt * np.exp(-k ** 2 / (4 * self.xi ** 2)) * (A0 + B0 * kout ** 2)
            tot += term
            if term < 1e-300 or (tot > 0 and term < 1e-30 * tot):
                break
            k = kout
        return tot

    def eps_F(self, kc, rho):
        """2 rho sum_{|k| > kc} ||Khat_L|| (kc is mapped to the shell it retains)."""
        nmax = int(np.floor((kc * self.L / (2 * np.pi)) ** 2 * (1 + 1e-12)))
        idx = np.searchsorted(self.n, nmax, side="right") - 1
        return 2 * rho * float(self.tail_after[idx])

    def kc_tol(self, target_abs, rho):
        """Smallest shell radius kc with eps_F(kc) <= target_abs."""
        ok = 2 * rho * self.tail_after <= target_abs
        if not ok.any():
            return np.nan
        first = int(np.argmax(ok))
        return float(self.k[first])


def mesh_modes(kc, L):
    mc = int(np.floor(kc * L / (2 * np.pi) + 1e-12))
    g = np.arange(-mc, mc + 1)
    mv = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    n2 = (mv ** 2).sum(1)
    keep = 2 * np.pi / L * np.sqrt(n2) <= kc * (1 + 1e-12)
    return mv[keep], n2[keep], mc


def eps_mesh(fs, kc, M, p, rho, return_alpha=False):
    """A priori mesh bound 2 rho sum_{|k| <= kc} alpha (2 + alpha) ||Khat_L||, alpha = C(p) max_a q_a^p."""
    L = fs.L
    mv, n2, mc = mesh_modes(kc, L)
    h = L / M
    if mc >= M / 2:
        return np.inf
    ka = 2 * np.pi / L * np.abs(mv)
    qa = ka * h / (2 * np.pi - ka * h)
    alpha = C_p(p) * qa.max(axis=1) ** p
    W2 = np.prod(np.sinc(2 * np.pi / L * mv * h / (2 * np.pi)) ** (2 * p), axis=1)
    kn = fs.knorm_of_n[n2]
    excl = W2 <= 1e-10
    val = 2 * rho * float(np.sum(np.where(excl, 1.0, alpha * (2 + alpha)) * kn))
    return (val, alpha) if return_alpha else val


def smooth_numbers(lo, hi):
    out, m = [], int(lo)
    while m <= hi:
        m = next_smooth(m)
        if m <= hi:
            out.append(m)
        m += 1
    return out


def M_tol(fs, kc, p, target_abs, rho, M_cap=400):
    L = fs.L
    mc = int(np.floor(kc * L / (2 * np.pi) + 1e-12))
    for M in smooth_numbers(2 * mc + 1, M_cap):
        if kc * (L / M) / np.pi >= 1:          # make_mesh / mesh_for require eta_actual < 1
            continue
        if eps_mesh(fs, kc, M, p, rho) <= target_abs:
            return M
    return None


def eps_quad_estimate(xi, L, kc, M, p, rho, fs):
    """Numerical estimate (NOT a certificate) of the coefficient-quadrature contribution."""
    K = C.kernel(xi)
    base = K.base
    saved = base.tau_rule
    try:
        base.tau_rule = graded_rule(100.0, 1e-12, 2 * 46, n=30)      # refined rule of exp1 RefinedEwald
        mv, n2, mc = mesh_modes(kc, L)
        nu = np.unique(n2)
        k = 2 * np.pi / L * np.sqrt(nu)
        Ar, Br = K.AB_scaled(k)
    finally:
        base.tau_rule = saved
    Ap, Bp = K.AB_scaled(k)
    dmp = np.exp(-k ** 2 / (4 * xi ** 2))
    dA, dB = (Ap - Ar) * dmp, (Bp - Br) * dmp
    dn = np.maximum(np.abs(dA), np.abs(dA + dB * k ** 2))
    dn_of = np.zeros(nu.max() + 1)
    dn_of[nu] = dn
    _, alpha = eps_mesh(fs, kc, M, p, rho, return_alpha=True)
    eF = 2 * rho * float(np.sum((1 + alpha) ** 2 * dn_of[n2]))
    rel_coeff = float(np.max(dn / np.maximum(fs.knorm_of_n[nu], 1e-300)))
    return eF, rel_coeff


def theta_S_closed_vs_quad(xi, rmin, rmax):
    K = C.kernel(xi)
    r = np.linspace(rmin, rmax, 400)
    a, b = K.theta_S_scaled_closed(r), K.theta_S_scaled_quad(r)
    return float(np.max(np.abs(a - b) / np.abs(b)))


# ----------------------------------------------------------------------------- real-space side
class RealSpaceTail:
    """All retained images of one configuration up to R_max (sorted by r^2); per-xi certified tail bound."""

    def __init__(self, x, L, R_max):
        self.x, self.L, self.N = np.mod(x, L), float(L), len(x)
        il = kdtree_images(self.x, L, R_max)
        r2 = np.einsum("pa,pa->p", il.d, il.d)
        o = np.argsort(r2, kind="stable")
        self.i, self.j, self.d, self.r2 = il.i[o], il.j[o], il.d[o], r2[o]
        self.R_max = float(R_max)
        self.key = self.i.astype(np.int64) * self.N + self.j
        _, first = np.unique(self.key, return_index=True)
        self.first_r2 = np.sort(self.r2[first])          # distance^2 at which each unique pair enters the list

    def n_img(self, rc):
        return int(np.searchsorted(self.r2, rc * rc, side="right"))

    def n_unique(self, rc):
        return int(np.searchsorted(self.first_r2, rc * rc, side="right"))

    def prepare(self, xi):
        """Pair tensors K_S(d) (6 components) for r <= R_far(xi)."""
        K = C.kernel(xi)
        R_far = min((A_MAX + R_PAD) / xi, self.R_max)
        m = int(np.searchsorted(self.r2, R_far * R_far, side="right"))
        d = self.d[:m]
        r = np.sqrt(self.r2[:m])
        w = K.theta_S_scaled_closed(r) * np.exp(-(xi * r) ** 2) / r ** 2
        comps = np.stack([w * d[:, a] * d[:, b] for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], 1)
        tail = self.analytic_tail(xi, R_far)
        return dict(xi=xi, m=m, comps=comps, R_far=R_far, tail=tail)

    def analytic_tail(self, xi, R):
        K = C.kernel(xi)
        z, c = K.z, K.amp
        rho_x2 = K.base.rho_xi2
        assert xi * R >= z
        th = lambda r: c * np.exp(-(xi * r) ** 2) * (np.exp(-z * z) / r + rho_x2)       # noqa: E731
        cnt = lambda r: self.N * 4 / 3 * np.pi * (r + np.sqrt(3) * self.L / 2) ** 3 / self.L ** 3   # noqa: E731
        tot, r, dr = 0.0, R, 0.05
        for _ in range(200000):
            term = cnt(r + dr) * th(r)
            tot += term
            if term < 1e-300 or term < 1e-30 * tot:
                break
            r += dr
        return tot

    def _tensor_sum(self, prep, lo, hi):
        """Per-particle sum of K_S (6 components) over the sorted images with index in [lo, hi)."""
        D = np.zeros((self.N, 6))
        if hi > lo:
            cs = prep["comps"][lo:hi]
            ii, jj = self.i[lo:hi], self.j[lo:hi]
            for c in range(6):
                D[:, c] = np.bincount(ii, cs[:, c], self.N) + np.bincount(jj, cs[:, c], self.N)
        return D

    def _bound(self, D, prep):
        T = D[:, [0, 3, 4, 3, 1, 5, 4, 5, 2]].reshape(self.N, 3, 3)
        return 2 * (float(np.linalg.eigvalsh(T)[:, -1].max()) + prep["tail"])

    def eps_S(self, prep, rc):
        """2 max_i [lambda_max(sum_{rc < r <= R_far} K_S) + analytic tail]."""
        lo = int(np.searchsorted(self.r2, rc * rc, side="right"))
        return self._bound(self._tensor_sum(prep, lo, prep["m"]), prep)

    def rc_tol(self, prep, target_abs, tol=1e-5):
        """Bisection on rc; the tensor sum beyond the current upper end is kept, so each step adds one shell."""
        xi, m = prep["xi"], prep["m"]
        idx = lambda r: min(int(np.searchsorted(self.r2, r * r, side="right")), m)     # noqa: E731
        lo, hi = 0.5 / xi, A_MAX / xi
        hi_i = idx(hi)
        base = self._tensor_sum(prep, hi_i, m)
        if self._bound(base, prep) > target_abs:
            return np.nan
        while hi - lo > tol * hi:
            mid = 0.5 * (lo + hi)
            mid_i = idx(mid)
            D = base + self._tensor_sum(prep, mid_i, hi_i)
            if self._bound(D, prep) <= target_abs:
                hi, hi_i, base = mid, mid_i, D
            else:
                lo = mid
        return hi


# ----------------------------------------------------------------------------- selection
def lambda_lower_bound(Gref, N):
    """max_i lambda_max(Gamma_ref,ii) <= ||Gamma_ref||_2."""
    B = Gref.reshape(N, 3, N, 3)[np.arange(N), :, np.arange(N), :]
    return float(np.linalg.eigvalsh(B)[:, -1].max())


def select_for_xi(xi, rst, L, N, lam_lb, eps=C.EPS_TARGET, p=C.P_ORDER, split=SPLIT, fs=None, prep=None):
    rho = N / L ** 3
    fs = fs or FourierShells(xi, L)
    prep = prep or rst.prepare(xi)
    tS, tF, tM = (split[k] * eps * lam_lb for k in ("S", "F", "mesh"))
    rc = rst.rc_tol(prep, tS)
    kc = fs.kc_tol(tF, rho)
    M = M_tol(fs, kc, p, tM, rho) if np.isfinite(kc) else None
    row = dict(xi=float(xi), rc=rc, kc=kc, a_tol=xi * rc, b_tol=kc / (2 * xi), M=M, p=p, lam_lb=lam_lb,
               feasible=bool(np.isfinite(rc) and np.isfinite(kc) and M is not None))
    if row["feasible"]:
        eS, eF, eMh = rst.eps_S(prep, rc), fs.eps_F(kc, rho), eps_mesh(fs, kc, M, p, rho)
        eQ, rel_coeff = eps_quad_estimate(xi, L, kc, M, p, rho, fs)
        row.update(eps_S_abs=eS, eps_F_abs=eF, eps_mesh_abs=eMh, eps_quad_abs_est=eQ, coeff_rel_quad_diff=rel_coeff,
                   eps_cert_rel=(eS + eF + eMh) / lam_lb, eps_total_rel_with_quad=(eS + eF + eMh + eQ) / lam_lb,
                   quad_within_budget=bool(eQ <= split["quad"] * eps * lam_lb),
                   eta=kc * (L / M) / np.pi, h=L / M, mc=int(np.floor(kc * L / (2 * np.pi) + 1e-12)),
                   n_img=rst.n_img(rc), n_unique=rst.n_unique(rc))
        row["images_per_pair"] = row["n_img"] / row["n_unique"]
    return row


# ----------------------------------------------------------------------------- cost model
class CostModel:
    def __init__(self, C_R, C_P, C_F, C_0, **extra):
        self.C_R, self.C_P, self.C_F, self.C_0 = C_R, C_P, C_F, C_0
        self.extra = extra

    def parts(self, n_img, N, p, M):
        return dict(real=self.C_R * n_img, spread_gather=self.C_P * N * p ** 3,
                    fft=self.C_F * M ** 3 * np.log(M ** 3), overhead=self.C_0)

    def predict(self, n_img, N, p, M):
        return float(sum(self.parts(n_img, N, p, M).values()))

    def as_dict(self):
        return dict(C_R=self.C_R, C_P=self.C_P, C_F=self.C_F, C_0=self.C_0, **self.extra)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        return cls(d.pop("C_R"), d.pop("C_P"), d.pop("C_F"), d.pop("C_0"), **d)


def main():
    ap = argparse.ArgumentParser(description="quick look at the certified selection for one configuration")
    ap.add_argument("--config", default="strong_N256_s1")
    ap.add_argument("--xi", type=float, nargs="+", default=[0.7, 0.85, 1.0, 1.2])
    args = ap.parse_args()
    from clustered_configurations import load
    x, L = load(args.config)
    N = len(x)
    G, _ = C.reference(x, L)
    lam = lambda_lower_bound(G, N)
    C.log(f"{args.config}: N={N} L={L:.4f} Lambda_lb={lam:.4f} ||Gref||={np.linalg.norm(G, 2):.4f}")
    rst = RealSpaceTail(x, L, (A_MAX + R_PAD) / min(args.xi))
    for xi in args.xi:
        C.log(" ".join(f"{k}={v:.4g}" if isinstance(v, (float, np.floating)) else f"{k}={v}" for k, v in select_for_xi(xi, rst, L, N, lam).items()))


if __name__ == "__main__":
    main()
