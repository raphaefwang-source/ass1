#!/usr/bin/env python3
"""
Numerical verification of the Ewald splitting of the screened, pure-longitudinal
tensor kernel

    K_kappa(r_vec) = exp(-kappa r)/r * P(r_vec),     P = r_vec r_vec^T / r^2,

based on the Gaussian-scale representation

    exp(-kappa r)/r^3 = int_0^inf rho_kappa(s) exp(-s r^2) ds,
    rho_kappa(s) = 2 sqrt(s)/sqrt(pi) exp(-kappa^2/(4s)) - kappa erfc(kappa/(2 sqrt(s))),

cut at s = xi^2 into K_L (s < xi^2) and K_S (s > xi^2).

Notation: kappa > 0 physical screening, xi > 0 Ewald splitting parameter,
k = |k_vec| Fourier wave number (k = 0 allowed), z = kappa/(2 xi),
h(z) = exp(-z^2) - sqrt(pi) z erfc(z).  There is no transverse projector.

Every primary quantity is computed directly from the s-integral definitions.
Closed forms (see DERIVATIONS.md), direct adaptive quadrature, a radial Hankel
transform and a 3D FFT are used only as independent cross-checks.
Exponentially small quantities are carried in scaled form, e.g.
theta_S(r) * exp(xi^2 r^2), so nothing underflows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, special

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SQPI = np.sqrt(np.pi)
PI32 = np.pi ** 1.5
EPS = np.finfo(float).eps
CHUNK = 1 << 21  # max number of (point, node) pairs evaluated at once


# ----------------------------------------------------------------------------
# Quadrature utilities
# ----------------------------------------------------------------------------
def composite_gl(breaks, n):
    """Composite Gauss-Legendre rule on the panels defined by `breaks`."""
    x, w = np.polynomial.legendre.leggauss(n)
    breaks = np.asarray(breaks, float)
    a, b = breaks[:-1, None], breaks[1:, None]
    half = 0.5 * (b - a)
    return (a + half * (x + 1.0)).ravel(), (half * w).ravel()


def graded_rule(upper, smallest, n_panels, n=20):
    """GL rule on [0, upper] with panels graded geometrically towards 0."""
    breaks = np.concatenate(([0.0], np.geomspace(smallest, upper, n_panels)))
    return composite_gl(breaks, n)


def blockwise(fn, x, n_nodes):
    """Evaluate a vectorised fn(points) on x in memory-bounded blocks."""
    x = np.asarray(x, float)
    flat = x.ravel()
    step = max(1, CHUNK // max(n_nodes, 1))
    out = np.concatenate([fn(flat[i:i + step]) for i in range(0, flat.size, step)], axis=0)
    return out.reshape(x.shape + out.shape[1:])


def ls_slope(x, y):
    """Least-squares slope of y against x along the last axis."""
    xm = x - x.mean(axis=-1, keepdims=True)
    ym = y - y.mean(axis=-1, keepdims=True)
    return (xm * ym).sum(axis=-1) / (xm * xm).sum(axis=-1)


# ----------------------------------------------------------------------------
# Special functions
# ----------------------------------------------------------------------------
_N_SER = np.arange(1, 26)
_SER_COEF = (-1.0) ** (_N_SER + 1) * special.factorial2(2 * _N_SER - 1, exact=False)


def p_aux(x):
    """p(x) = 1 - sqrt(pi) x erfcx(x), x >= 0, without cancellation at large x."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = x <= 8.0
    out[small] = 1.0 - SQPI * x[small] * special.erfcx(x[small])
    u = 0.5 / x[~small] ** 2
    out[~small] = (_SER_COEF * u[:, None] ** _N_SER).sum(axis=-1)
    return out


def h_literal(z):
    """h(z) exactly as written in the task."""
    return np.exp(-z ** 2) - SQPI * z * special.erfc(z)


def h_func(z):
    """h(z) = exp(-z^2) p(z): algebraically identical, stable for large z."""
    z = np.asarray(z, float)
    return np.exp(-z ** 2) * p_aux(z)


def c_aux(z):
    """c(z) = (1 + z^2) - sqrt(pi) z (3/2 + z^2) erfcx(z) = -1/2 + (3/2 + z^2) p(z), z >= 0.

    Evaluated via p(z) (moderate z) or its own asymptotic series (z > 8) to limit cancellation.
    """
    if z <= 8.0:
        return float(-0.5 + (1.5 + z * z) * p_aux(np.array([z]))[0])
    n = _N_SER[1:-1]
    return float(((1.5 * _SER_COEF[n - 1] + 0.5 * _SER_COEF[n]) * (0.5 / z ** 2) ** n).sum())


def rho_literal(s, kap):
    """rho_kappa(s) exactly as written in the task."""
    s = np.asarray(s, float)
    return 2.0 * np.sqrt(s) / SQPI * np.exp(-kap ** 2 / (4 * s)) - kap * special.erfc(kap / (2 * np.sqrt(s)))


def rho(s, kap):
    """rho_kappa(s) = 2 sqrt(s)/sqrt(pi) exp(-kappa^2/4s) p(kappa/(2 sqrt s)).

    Algebraically identical to rho_literal (kappa erfc(x) = 2 sqrt(s) x erfcx(x) e^{-x^2}
    with x = kappa/(2 sqrt s)); avoids the cancellation of the two terms for s << kappa^2.
    """
    s = np.asarray(s, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = kap / (2.0 * np.sqrt(s))
        out = 2.0 * np.sqrt(s) / SQPI * np.exp(-x ** 2) * p_aux(x)
    return np.where(s > 0, out, 0.0)


def w_density(s, kap):
    """Candidate derivative d rho/ds = exp(-kappa^2/(4s)) / sqrt(pi s)."""
    return np.exp(-kap ** 2 / (4 * s)) / np.sqrt(np.pi * s)


def j2_over_x2(x):
    """Spherical Bessel j_2(x)/x^2, finite at x = 0 (limit 1/15)."""
    x = np.asarray(x, float)
    small = x < 0.1
    xs = np.where(small, 1.0, x)
    x2 = x * x
    series = 1 / 15 - x2 / 210 + x2 ** 2 / 7560 - x2 ** 3 / 498960
    return np.where(small, series, special.spherical_jn(2, xs) / xs ** 2)


# ----------------------------------------------------------------------------
# The split kernel
# ----------------------------------------------------------------------------
class EwaldLongitudinal:
    def __init__(self, kappa, xi):
        self.kap, self.xi = float(kappa), float(xi)
        self.z = self.kap / (2 * self.xi)
        self.h = float(h_func(self.z))
        self.rho_xi2 = float(rho(self.xi ** 2, self.kap))
        xi2 = self.xi ** 2
        self.s_low = graded_rule(xi2, xi2 * 2.0 ** -48, 97)   # s in [0, xi^2], panel ratio sqrt(2)
        self.rho_low = rho(self.s_low[0], self.kap)
        self.t_rule = graded_rule(64.0, 1e-12, 56)             # t in [0, 64], weight e^{-t}
        self.tau_rule = graded_rule(80.0, 1e-10, 46)           # tau in [0, 80], weight ~e^{-tau}
        self.y_rule = graded_rule(64.0, 1e-8, 31, n=16)        # y in [0, 64], weight e^{-y}
        self.unit_rule = graded_rule(1.0, 1e-8, 24, n=16)      # u in [0, 1]
        # next-order coefficients: numerical/asymptotic = 1 + c/x^2 + O(x^-4) (DERIVATIONS.md)
        q = np.exp(-self.z ** 2) / self.h
        self.c_theta = q / (2 * xi2)
        self.c_A = 2 * xi2 * (1 - q)
        self.c_norm = 2 * xi2 * (2 - q)
        self.c_ES = (1 + q) / (2 * xi2)
        self.c_EF = 2 * xi2 * (3 - q)
        self._kstar = None

    # ---- real space -------------------------------------------------------
    def g_L(self, r):
        """int_0^{xi^2} rho(s) e^{-s r^2} ds, so K_L = g_L(r) r r^T and theta_L = r^2 g_L."""
        s, ws = self.s_low
        wr = ws * self.rho_low
        return blockwise(lambda rb: np.exp(-np.outer(rb ** 2, s)) @ wr, r, s.size)

    def theta_L(self, r):
        return np.asarray(r, float) ** 2 * self.g_L(r)

    def theta_S_scaled(self, r):
        """theta_S(r) e^{xi^2 r^2} = int_0^inf rho(xi^2 + t/r^2) e^{-t} dt  (s = xi^2 + t/r^2)."""
        t, wt = self.t_rule
        we = wt * np.exp(-t)
        f = lambda rb: rho(self.xi ** 2 + t[None, :] / rb[:, None] ** 2, self.kap) @ we
        return blockwise(f, r, t.size)

    def theta_S(self, r):
        r = np.asarray(r, float)
        return self.theta_S_scaled(r) * np.exp(-(self.xi * r) ** 2)

    def theta_S_scaled_closed(self, r):
        """Closed form (cross-check only): e^{-z^2}/(2r)[erfcx(xi r+z)+erfcx(xi r-z)] + rho(xi^2)."""
        r = np.asarray(r, float)
        a = self.xi * r
        return (np.exp(-self.z ** 2) / (2 * r) * (special.erfcx(a + self.z) + special.erfcx(a - self.z))
                + self.rho_xi2)

    def C_closed(self):
        """K_L = C r r^T + O(r^4) with C = int_0^{xi^2} rho ds (closed form, cross-check only)."""
        return 4 * self.xi ** 3 / (3 * SQPI) * np.exp(-self.z ** 2) * c_aux(self.z)

    # ---- Fourier space ----------------------------------------------------
    def AB_scaled(self, k):
        """(A(k), B(k)) * exp(k^2/(4 xi^2)) from the s-integrals of the task.

        J_n = int_0^{xi^2} rho s^{-n} e^{-k^2/4s} ds is mapped by s = 1/(v + xi^-2),
        v = tau/lam, lam = (k^2 + kappa^2)/4.  The integrand decays like e^{-tau}
        for every k >= 0 (the decay at k = 0 comes from kappa > 0).
        """
        tau, wt = self.tau_rule

        def f(kb):
            k2 = kb[:, None] ** 2
            lam = (k2 + self.kap ** 2) / 4
            v = tau[None, :] / lam
            s = 1.0 / (v + self.xi ** -2)
            base = rho(s, self.kap) * np.exp(-k2 * v / 4) * s ** 2 / lam
            J52 = (base * s ** -2.5) @ wt
            J72 = (base * s ** -3.5) @ wt
            return np.stack([0.5 * PI32 * J52, -0.25 * PI32 * J72], axis=-1)

        out = blockwise(f, k, tau.size)
        return out[..., 0], out[..., 1]

    def eigs_scaled(self, k):
        """Scaled A, B, longitudinal eigenvalue A + B k^2 and ||Khat_L||_2."""
        k = np.asarray(k, float)
        A, B = self.AB_scaled(k)
        lam_par = A + B * k ** 2
        return A, B, lam_par, np.maximum(np.abs(A), np.abs(lam_par))

    def _root(self, fn):
        g = lambda k: float(fn(np.array([k]))[0])
        return optimize.brentq(g, 1e-6, 60 * self.xi, xtol=1e-14, rtol=4 * EPS)

    @property
    def k_star(self):
        """Kink of the operator norm: 2A + B k^2 = 0 (norm = A below, -(A + B k^2) above)."""
        if self._kstar is None:
            self._kstar = self._root(lambda k: (lambda A, B: 2 * A + B * k ** 2)(*self.AB_scaled(k)))
        return self._kstar

    def k_zero(self):
        """Sign change of the longitudinal eigenvalue A + B k^2."""
        return self._root(lambda k: (lambda A, B: A + B * k ** 2)(*self.AB_scaled(k)))

    # ---- cutoff tails -----------------------------------------------------
    def ES_scaled_nested(self, rc):
        """E_S(rc) e^{xi^2 rc^2} = (2 pi/xi^2) int_0^inf r Theta_S(r) e^{-y} dy, r^2 = rc^2 + y/xi^2."""
        rc = np.asarray(rc, float)
        y, wy = self.y_rule
        r = np.sqrt(rc[:, None] ** 2 + y[None, :] / self.xi ** 2)
        return 2 * np.pi / self.xi ** 2 * (r * self.theta_S_scaled(r)) @ (wy * np.exp(-y))

    def ES_scaled_swap(self, rc):
        """Same quantity, r-integral done analytically: 4 pi int_{xi^2}^inf rho(s) int_rc^inf r^4 e^{-s r^2} dr ds."""
        rc = np.asarray(rc, float)
        t, wt = self.t_rule
        R = rc[:, None]
        s = self.xi ** 2 + t[None, :] / R ** 2
        r4 = R ** 3 / (2 * s) + 3 * R / (4 * s ** 2) + 3 * SQPI / (8 * s ** 2.5) * special.erfcx(np.sqrt(s) * R)
        return 4 * np.pi / rc ** 2 * ((rho(s, self.kap) * r4) @ (wt * np.exp(-t)))

    def EF_scaled_direct(self, kc):
        """E_F(kc) e^{kc^2/(4 xi^2)} = (xi^2/pi^2) int_0^inf k Nhat(k) e^{-y} dy, k^2 = kc^2 + 4 xi^2 y,
        with the y-range split at the kink k_star of ||Khat_L||_2."""
        kc = np.asarray(kc, float)
        ystar = np.maximum(self.k_star ** 2 - kc ** 2, 0.0) / (4 * self.xi ** 2)
        u, wu = self.unit_rule
        y, wy = self.y_rule
        Y = np.concatenate([ystar[:, None] * u, ystar[:, None] + y], axis=1)
        W = np.concatenate([ystar[:, None] * wu, np.broadcast_to(wy, (kc.size, y.size))], axis=1)
        k = np.sqrt(kc[:, None] ** 2 + 4 * self.xi ** 2 * Y)
        norm = self.eigs_scaled(k)[3]
        return self.xi ** 2 / np.pi ** 2 * (k * norm * np.exp(-Y) * W).sum(axis=1)

    def EF_scaled_swap(self, kc):
        """For kc >= k_star only (norm = -(A + B k^2)): k-integral done analytically."""
        kc = np.asarray(kc, float)
        tau, wt = self.tau_rule
        K = kc[:, None]
        lam = (K ** 2 + self.kap ** 2) / 4
        v = tau[None, :] / lam
        s = 1.0 / (v + self.xi ** -2)
        sig = 1.0 / (4 * s)
        ex = special.erfcx(np.sqrt(sig) * K)
        m2 = K / (2 * sig) + SQPI / (4 * sig ** 1.5) * ex
        m4 = K ** 3 / (2 * sig) + 3 * K / (4 * sig ** 2) + 3 * SQPI / (8 * sig ** 2.5) * ex
        f = rho(s, self.kap) * s ** 2 * (0.25 * s ** -3.5 * m4 - 0.5 * s ** -2.5 * m2) * np.exp(-K ** 2 * v / 4) / lam
        return f @ wt / (2 * SQPI)

    # ---- independent cross-checks of the Fourier formulas -----------------
    def hankel_AB(self, k):
        """A = (4pi/3) int r^2 theta_L (j0 + j2) dr,  B = -4pi int r^4 theta_L j2(kr)/(kr)^2 dr."""
        k = np.asarray(k, float)
        R = 40.0 / min(self.kap, self.xi) + 10.0
        dr = min(0.25, 1.5 / max(k.max(), 1e-9))
        r, wr = composite_gl(np.linspace(0.0, R, int(R / dr) + 1), 16)
        thL = self.theta_L(r)
        x = np.outer(k, r)
        A = 4 * np.pi / 3 * ((special.spherical_jn(0, x) + special.spherical_jn(2, x)) * r ** 2 * thL) @ wr
        B = -4 * np.pi * (j2_over_x2(x) * r ** 4 * thL) @ wr
        return A, B

    def fft_check(self):
        """Brute-force 3D FFT of K_L_ij(x) = g_L(r) x_i x_j vs A I + B k k^T on the FFT grid.

        Box L = 64/min(kappa, 1) (periodic images ~ e^{-kappa L/2}), spacing ~ 1/(3 xi);
        compared for |k| <= kmax with an aliasing margin of 12 xi.  Returns None if the
        grid would exceed 256^3 points.
        """
        L = 64.0 / min(self.kap, 1.0)
        N = int(min(256, 2 * np.ceil(1.5 * self.xi * L)))
        hg = L / N
        kmax = min(6.0 * self.xi, 2 * np.pi / hg - 12.0 * self.xi)
        if kmax <= 0:
            return None
        m = np.arange(N) - N // 2
        M2 = m[:, None, None] ** 2 + m[None, :, None] ** 2 + m[None, None, :] ** 2
        uniq, inv = np.unique(M2, return_inverse=True)
        g = self.g_L(hg * np.sqrt(uniq))[inv.reshape(M2.shape)]
        del M2, inv
        kf = 2 * np.pi * np.fft.fftfreq(N, d=hg)
        mask = (kf[:, None, None] ** 2 + kf[None, :, None] ** 2 + kf[None, None, :] ** 2) <= kmax ** 2
        idx = np.nonzero(mask)
        kv = np.stack([kf[i] for i in idx], axis=1)
        kmag = np.linalg.norm(kv, axis=1)
        ku, kinv = np.unique(kmag, return_inverse=True)
        A_u, B_u = self.AB_scaled(ku)
        dmp = np.exp(-ku ** 2 / (4 * self.xi ** 2))
        A, B = (A_u * dmp)[kinv], (B_u * dmp)[kinv]
        ref = A[:, None, None] * np.eye(3) + B[:, None, None] * kv[:, :, None] * kv[:, None, :]
        num = np.empty_like(ref)
        X = hg * m
        coords = [X[:, None, None], X[None, :, None], X[None, None, :]]
        imag_max = 0.0
        for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
            F = np.fft.fftn(np.fft.ifftshift(coords[i] * coords[j] * g)) * hg ** 3
            vals = F[mask]
            num[:, i, j] = num[:, j, i] = vals.real
            imag_max = max(imag_max, np.abs(vals.imag).max())
        nrm = np.linalg.norm(ref, 2, axis=(1, 2))
        rel = np.abs(num - ref).max(axis=(1, 2)) / nrm
        return kmag, rel, imag_max / nrm.max(), kmax


# ----------------------------------------------------------------------------
# Plot style (reference palette: slot 1 blue, 2 orange, 3 aqua; neutral inks)
# ----------------------------------------------------------------------------
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "xtick.color": INK2, "ytick.color": INK2,
        "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150,
    })


def refline(ax, y, label=None):
    ax.axhline(y, color=INK2, lw=0.8, zorder=0, label=label)


def padded_lim(*arrays, pad=0.05):
    v = np.concatenate(arrays)
    lo, hi = min(v.min(), 0.98), max(v.max(), 1.02)
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(outdir / name)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------
def run(kappa, xi, outdir, fft=True):
    figdir, resdir = outdir / "figures", outdir / "results"
    figdir.mkdir(parents=True, exist_ok=True)
    resdir.mkdir(parents=True, exist_ok=True)
    K = EwaldLongitudinal(kappa, xi)
    kap, z, h = K.kap, K.z, K.h
    R = {"kappa": kap, "xi": xi, "z": z, "h(z)": h, "rho(xi^2)": K.rho_xi2}
    qopt = dict(epsabs=0.0, epsrel=1e-13, limit=500)

    # Sampling in natural units: lengths ~ 1/xi, wave numbers ~ xi.  For large z the next-order
    # coefficients grow like z^2, so the "large x" region is pushed out by sc = max(1, 2z).
    sc = max(1.0, 2 * z)
    a_r, b_k = sc / xi, sc * xi
    R["sampling_scale_sc"] = sc

    # ---- 1. formula checks ------------------------------------------------
    s_chk = np.geomspace(max(1e-2 * xi ** 2, kap ** 2 / 20), 1e3 * max(xi, kap) ** 2, 400)
    R["rho_stable_vs_literal_maxrel"] = float(np.max(np.abs(rho(s_chk, kap) / rho_literal(s_chk, kap) - 1)))
    s_pts = np.array([0.05 * kap ** 2, 0.2 * kap ** 2, kap ** 2, 5 * kap ** 2, 0.5 * xi ** 2, xi ** 2, 3 * xi ** 2, 20 * xi ** 2])
    rho_int_w = np.array([integrate.quad(w_density, 0, s, args=(kap,), **qopt)[0] for s in s_pts])
    R["rho_equals_int_w_maxrel"] = float(np.max(np.abs(rho_int_w / rho_literal(s_pts, kap) - 1)))
    s_pos = np.geomspace(1e-6, 1e4, 2000) * xi ** 2
    R["min rho(s)e^{kappa^2/4s} (s in xi^2[1e-6,1e4])"] = float(
        (2 * np.sqrt(s_pos) / SQPI * p_aux(kap / (2 * np.sqrt(s_pos)))).min())

    def laplace_quad(r):
        f = lambda s: rho_literal(s, kap) * np.exp(-s * r * r)
        return (integrate.quad(f, 0, xi ** 2, **qopt)[0] + integrate.quad(f, xi ** 2, np.inf, **qopt)[0])

    r_pts = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]) / xi
    lap = np.array([laplace_quad(r) for r in r_pts])
    R["laplace_representation_maxrel"] = float(np.max(np.abs(lap / (np.exp(-kap * r_pts) / r_pts ** 3) - 1)))
    R["rho_xi2_vs_2xi_h/sqrtpi_rel"] = float(abs(rho_literal(xi ** 2, kap) / (2 * xi / SQPI * h_literal(z)) - 1))
    R["h_stable_vs_literal_rel"] = float(abs(h / h_literal(z) - 1))

    r_cf = np.geomspace(1e-3, 25, 500) / xi
    R["theta_S_def_vs_closed_maxrel"] = float(np.max(np.abs(K.theta_S_scaled(r_cf) / K.theta_S_scaled_closed(r_cf) - 1)))
    r_q = np.array([0.1, 0.5, 1.0, 2.0, 3.0]) / xi
    thS_q = np.array([r * r * integrate.quad(lambda s: rho_literal(s, kap) * np.exp(-s * r * r), xi ** 2, np.inf, **qopt)[0]
                      for r in r_q])
    thL_q = np.array([r * r * integrate.quad(lambda s: rho_literal(s, kap) * np.exp(-s * r * r), 0, xi ** 2, **qopt)[0]
                      for r in r_q])
    R["theta_S_GL_vs_quad_maxrel"] = float(np.max(np.abs(K.theta_S(r_q) / thS_q - 1)))
    R["theta_L_GL_vs_quad_maxrel"] = float(np.max(np.abs(K.theta_L(r_q) / thL_q - 1)))

    k_q = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0]) * xi
    AB_q = np.array([[integrate.quad(lambda s: c * rho_literal(s, kap) * s ** -n * np.exp(-k * k / (4 * s)),
                                     0, xi ** 2, **qopt)[0] for c, n in ((PI32 / 2, 2.5), (-PI32 / 4, 3.5))]
                     for k in k_q])
    A_s, B_s = K.AB_scaled(k_q)
    dmp = np.exp(-k_q ** 2 / (4 * xi ** 2))
    R["A_GL_vs_quad_maxrel"] = float(np.max(np.abs(A_s * dmp / AB_q[:, 0] - 1)))
    R["B_GL_vs_quad_maxrel"] = float(np.max(np.abs(B_s * dmp / AB_q[:, 1] - 1)))
    R["A(0)"], R["B(0)"] = float(A_s[0]), float(B_s[0])

    k_h = np.linspace(0.0, 6.0 * xi, 25)
    A_h, B_h = K.hankel_AB(k_h)
    A_f, B_f, lp_f, n_f = K.eigs_scaled(k_h)
    dmp = np.exp(-k_h ** 2 / (4 * xi ** 2))
    hank_A = np.abs(A_h / (A_f * dmp) - 1)
    hank_B = np.abs(B_h / (B_f * dmp) - 1)
    R["hankel_A_maxrel(k<=6xi)"] = float(hank_A.max())
    R["hankel_B_maxrel(k<=6xi)"] = float(hank_B.max())
    fft_res = K.fft_check() if fft else None
    if fft_res is not None:
        kf_mag, fft_rel, fft_imag, kf_max = fft_res
        R["fft_kmax"] = float(kf_max)
        R["fft_tensor_maxrel(|k|<=kmax)"] = float(fft_rel.max())
        R["fft_tensor_maxrel(|k|<=2kmax/3)"] = float(fft_rel[kf_mag <= 2 * kf_max / 3].max())
        R["fft_imag_part_rel"] = float(fft_imag)
        R["fft_n_kpoints"] = int(kf_mag.size)
    else:
        R["fft_check"] = "skipped"

    # ---- 2. splitting identity -------------------------------------------
    r_sp = np.geomspace(1e-3, 30, 600) / xi
    thS_sp, thL_sp = K.theta_S(r_sp), K.theta_L(r_sp)
    exact_sp = np.exp(-kap * r_sp) / r_sp
    split_rel = np.abs(thS_sp + thL_sp - exact_sp) / exact_sp
    R["split_maxrel_scalar"] = float(split_rel.max())

    rng = np.random.default_rng(20260928)
    nv = 4000
    dirs = rng.normal(size=(nv, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mags = 10 ** rng.uniform(-3, np.log10(20), nv) / xi
    rv = dirs * mags[:, None]
    outer = rv[:, :, None] * rv[:, None, :]
    P = outer / mags[:, None, None] ** 2
    ThS = K.theta_S_scaled(mags)                         # theta_S e^{xi^2 r^2}
    iS = ThS * np.exp(-(xi * mags) ** 2) / mags ** 2     # int_{xi^2}^inf rho e^{-s r^2} ds
    iL = K.g_L(mags)                                     # int_0^{xi^2}  rho e^{-s r^2} ds
    KS, KL = outer * iS[:, None, None], outer * iL[:, None, None]
    Kk = (np.exp(-kap * mags) / mags)[:, None, None] * P

    def fro(M):
        m = np.abs(M).max(axis=(1, 2))
        m = np.where(m > 0, m, 1.0)
        return m * np.linalg.norm(M / m[:, None, None], axis=(1, 2))

    split_mat = fro(KS + KL - Kk) / fro(Kk)
    R["split_maxrel_matrix"] = float(split_mat.max())
    I3 = np.eye(3)
    # K_S in scaled form (K_S e^{xi^2 r^2}) so that nothing underflows at large r
    for name, M, th in (("K_S", outer * (ThS / mags ** 2)[:, None, None], ThS), ("K_L", KL, iL * mags ** 2)):
        Mn = M / th[:, None, None]
        ev = np.linalg.eigvalsh(Mn)
        R[f"{name}_transverse_eigs_over_theta_max"] = float(np.abs(ev[:, :2]).max())
        R[f"{name}_longit_eig_vs_theta_maxrel"] = float(np.abs(ev[:, 2] - 1).max())
        R[f"{name}_(I-P)K_rel_max"] = float((fro((I3 - P) @ Mn) / fro(Mn)).max())
    R["min theta_S e^{xi^2 r^2} on split grid"] = float(K.theta_S_scaled(r_sp).min())
    R["min theta_L on split grid"] = float(thL_sp.min())
    R["theta_S_all_positive"] = bool(np.all(K.theta_S_scaled(r_sp) > 0))
    R["theta_L_all_positive"] = bool(np.all(thL_sp > 0))

    # ---- 4. K_L near r = 0 -----------------------------------------------
    C_gl = float(K.g_L(np.array([0.0]))[0])
    C_q = integrate.quad(lambda s: rho_literal(s, kap), 0, xi ** 2, **qopt)[0]
    D_q = integrate.quad(lambda s: s * rho_literal(s, kap), 0, xi ** 2, **qopt)[0]
    r_sm = np.linspace(1e-3, 0.2, 60) / xi
    c1, c0 = np.polyfit(r_sm ** 2, K.g_L(r_sm), 3)[-2:]
    R.update({"C_quad": C_q, "C_closed_form": float(K.C_closed()), "C_GL_at_r0": C_gl, "C_fit": float(c0),
              "D_quad(=-dg_L/d(r^2) at 0)": D_q, "D_fit": float(-c1)})
    R["|K_L - C r r^T|/|K_L| at xi r=1e-3"] = float(abs(K.g_L(np.array([1e-3 / xi]))[0] / C_q - 1))

    # ---- 5. real-space pointwise asymptotic ------------------------------
    r_as = np.linspace(0.5, 12.0, 461) * a_r
    ratio_th = K.theta_S_scaled(r_as) / K.rho_xi2
    rp = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]) * a_r
    rr = K.theta_S_scaled(rp) / K.rho_xi2
    R["theta_S_ratio"] = {f"r={x:.4g}": float(v) for x, v in zip(rp, rr)}
    R["theta_S_(ratio-1)r^2"] = {f"r={x:.4g}": float(v) for x, v in zip(rp, (rr - 1) * rp ** 2)}
    R["c_theta_predicted"] = K.c_theta

    # ---- 6/7. Fourier-space pointwise asymptotic -------------------------
    k_as = np.linspace(0.0, 40.0, 801) * b_k
    A_as, B_as, lp_as, n_as = K.eigs_scaled(k_as)
    A_asym = np.divide(4 * np.pi * h, k_as ** 2, out=np.full_like(k_as, np.inf), where=k_as > 0)
    N_asym = 2 * np.pi * h / xi ** 2
    kp = np.array([4.0, 6.0, 8.0, 12.0, 20.0, 30.0, 40.0]) * b_k
    Ap, Bp, lpp, npp = K.eigs_scaled(kp)
    R["A_ratio"] = {f"k={x:.4g}": float(v) for x, v in zip(kp, Ap / (4 * np.pi * h / kp ** 2))}
    R["A_(ratio-1)k^2"] = {f"k={x:.4g}": float(v) for x, v in zip(kp, (Ap / (4 * np.pi * h / kp ** 2) - 1) * kp ** 2)}
    R["norm_ratio"] = {f"k={x:.4g}": float(v) for x, v in zip(kp, npp / N_asym)}
    R["norm_(ratio-1)k^2"] = {f"k={x:.4g}": float(v) for x, v in zip(kp, (npp / N_asym - 1) * kp ** 2)}
    R["c_A_predicted"], R["c_norm_predicted"] = K.c_A, K.c_norm
    R["k_star(2A+Bk^2=0)"] = K.k_star
    R["k_zero(A+Bk^2=0)"] = K.k_zero()
    R["longit_eig_negative_for_k>k_zero"] = bool(np.all(lp_as[k_as > R["k_zero(A+Bk^2=0)"] + 1e-9] < 0))
    R["norm_is_A_below_kstar_and_|A+Bk2|_above"] = bool(
        np.all((2 * A_as + B_as * k_as ** 2)[k_as < K.k_star - 1e-9] > 0)
        and np.all((2 * A_as + B_as * k_as ** 2)[k_as > K.k_star + 1e-9] < 0))
    kdir = rng.normal(size=(k_as.size, 3))
    kdir /= np.linalg.norm(kdir, axis=1, keepdims=True)
    kvec = kdir * k_as[:, None]
    Mhat = A_as[:, None, None] * np.eye(3) + B_as[:, None, None] * kvec[:, :, None] * kvec[:, None, :]
    R["opnorm_numpy_vs_max|eig|_maxrel"] = float(np.max(np.abs(np.linalg.norm(Mhat, 2, axis=(1, 2)) / n_as - 1)))
    R["Khat_L(0)_is_A(0)I"] = float(np.abs(Mhat[0] - A_as[0] * np.eye(3)).max())

    # ---- 8. fit log theta_S vs r^2 ---------------------------------------
    ra = np.arange(1.0, 12.0) * a_r
    rw = ra[:, None] + np.linspace(0.0, a_r, 41)[None, :]
    log_thS = np.log(K.theta_S_scaled(rw)) - (xi * rw) ** 2
    slope_r = ls_slope(rw ** 2, log_thS)
    R["real_slope_windows"] = {f"r in [{a:.4g},{a + a_r:.4g}]": float(v) for a, v in zip(ra, slope_r)}
    rg = np.linspace(5.0, 10.0, 101) * a_r
    R["real_slope_global_(5..10)*sc/xi"] = float(ls_slope(rg ** 2, np.log(K.theta_S_scaled(rg)) - (xi * rg) ** 2))
    R["real_slope_target"] = -xi ** 2

    # ---- 9. fit log ||Khat_L|| vs k^2 ------------------------------------
    ka = np.arange(2.0, 40.0, 2.0) * b_k
    kw = ka[:, None] + np.linspace(0.0, 2 * b_k, 41)[None, :]
    log_n = np.log(K.eigs_scaled(kw)[3]) - kw ** 2 / (4 * xi ** 2)
    slope_k = ls_slope(kw ** 2, log_n)
    R["fourier_slope_windows"] = {f"k in [{a:.4g},{a + 2 * b_k:.4g}]": float(v) for a, v in zip(ka, slope_k)}
    kg = np.linspace(20.0, 40.0, 201) * b_k
    R["fourier_slope_global_(20..40)*sc*xi"] = float(
        ls_slope(kg ** 2, np.log(K.eigs_scaled(kg)[3]) - kg ** 2 / (4 * xi ** 2)))
    R["fourier_slope_target"] = -1 / (4 * xi ** 2)

    # ---- 10. real-space cutoff tail --------------------------------------
    rc = np.linspace(0.5, 8.0, 31) * a_r
    ES1, ES2 = K.ES_scaled_nested(rc), K.ES_scaled_swap(rc)
    ES_asym = 4 * SQPI / xi * h * rc
    R["E_S_two_routes_maxrel"] = float(np.max(np.abs(ES1 / ES2 - 1)))
    sel = rc >= 2.0 * a_r
    R["E_S_ratio"] = {f"rc={x:.4g}": float(v) for x, v in zip(rc[sel][::2], (ES1 / ES_asym)[sel][::2])}
    R["E_S_(ratio-1)rc^2"] = {f"rc={x:.4g}": float(v)
                              for x, v in zip(rc[sel][::2], ((ES1 / ES_asym - 1) * rc ** 2)[sel][::2])}
    R["c_ES_predicted"] = K.c_ES
    big = rc >= 4.0 * a_r
    R["E_S_slope_log(E/rc)_vs_rc^2_(large rc)"] = float(
        ls_slope(rc[big] ** 2, np.log(ES1[big] / rc[big]) - (xi * rc[big]) ** 2))

    # ---- 11. Fourier cutoff tail -----------------------------------------
    kc = np.linspace(0.25, 30.0, 120) * b_k
    EF1 = K.EF_scaled_direct(kc)
    EF_asym = 2 / np.pi * h * kc
    above = kc >= K.k_star
    EF2 = K.EF_scaled_swap(kc[above])
    R["E_F_two_routes_maxrel(kc>=k_star)"] = float(np.max(np.abs(EF1[above] / EF2 - 1)))
    kcp = np.array([4.0, 6.0, 8.0, 12.0, 20.0, 30.0]) * b_k
    EFp = K.EF_scaled_direct(kcp)
    R["E_F_ratio"] = {f"kc={x:.4g}": float(v) for x, v in zip(kcp, EFp / (2 / np.pi * h * kcp))}
    R["E_F_(ratio-1)kc^2"] = {f"kc={x:.4g}": float(v) for x, v in zip(kcp, (EFp / (2 / np.pi * h * kcp) - 1) * kcp ** 2)}
    R["c_EF_predicted"] = K.c_EF
    big = kc >= 15.0 * b_k
    R["E_F_slope_log(E/kc)_vs_kc^2_(large kc)"] = float(
        ls_slope(kc[big] ** 2, np.log(EF1[big] / kc[big]) - kc[big] ** 2 / (4 * xi ** 2)))

    # ---- verdicts (tolerances are generous multiples of what the quadrature achieves)
    last = lambda d: list(d.values())[-1]
    coef = lambda d, c: abs(last(d) / c - 1)
    R["next_order_coef_reldiff_at_largest_x"] = {
        "theta_S": coef(R["theta_S_(ratio-1)r^2"], K.c_theta), "A": coef(R["A_(ratio-1)k^2"], K.c_A),
        "norm": coef(R["norm_(ratio-1)k^2"], K.c_norm), "E_S": coef(R["E_S_(ratio-1)rc^2"], K.c_ES),
        "E_F": coef(R["E_F_(ratio-1)kc^2"], K.c_EF)}
    nc = R["next_order_coef_reldiff_at_largest_x"]
    R["verdicts"] = {
        "1 formulas (rho'=w, Laplace rep., rho(xi^2), closed forms, A/B vs quad/Hankel/FFT)": bool(
            max(R["rho_equals_int_w_maxrel"], R["laplace_representation_maxrel"], R["rho_xi2_vs_2xi_h/sqrtpi_rel"],
                R["theta_S_def_vs_closed_maxrel"], R["A_GL_vs_quad_maxrel"], R["B_GL_vs_quad_maxrel"],
                R["hankel_A_maxrel(k<=6xi)"], R["hankel_B_maxrel(k<=6xi)"]) < 1e-11
            and R.get("fft_tensor_maxrel(|k|<=kmax)", 0.0) < 1e-9),
        "2 K_S + K_L = K_kappa (max rel err < 1e-13)": bool(max(R["split_maxrel_scalar"], R["split_maxrel_matrix"]) < 1e-13),
        "3 pure longitudinal, theta_S > 0, theta_L > 0": bool(
            R["theta_S_all_positive"] and R["theta_L_all_positive"]
            and max(R["K_S_transverse_eigs_over_theta_max"], R["K_L_transverse_eigs_over_theta_max"]) < 1e-13),
        "4 K_L = C r r^T + O(r^4), C closed form": bool(abs(R["C_quad"] / R["C_closed_form"] - 1) < 1e-12
                                                   and abs(R["C_GL_at_r0"] / R["C_quad"] - 1) < 1e-12),
        "5 theta_S ~ rho(xi^2) exp(-xi^2 r^2)": bool(last(R["theta_S_ratio"]) - 1 < 0.01 and nc["theta_S"] < 0.01),
        "7a A ~ 4 pi h/k^2 exp(-k^2/4xi^2)": bool(abs(last(R["A_ratio"]) - 1) < 0.01 and nc["A"] < 0.01),
        "7b ||Khat_L|| ~ 2 pi h/xi^2 exp(-k^2/4xi^2)": bool(abs(last(R["norm_ratio"]) - 1) < 0.01 and nc["norm"] < 0.02),
        "8 slope log theta_S vs r^2 -> -xi^2": bool(abs(last(R["real_slope_windows"]) / R["real_slope_target"] - 1) < 1e-3),
        "9 slope log||Khat_L|| vs k^2 -> -1/(4xi^2)": bool(
            abs(last(R["fourier_slope_windows"]) / R["fourier_slope_target"] - 1) < 1e-3),
        "10 E_S cutoff law": bool(R["E_S_two_routes_maxrel"] < 1e-12 and abs(last(R["E_S_ratio"]) - 1) < 0.03
                                  and nc["E_S"] < 0.02),
        "11 E_F cutoff law": bool(R["E_F_two_routes_maxrel(kc>=k_star)"] < 1e-12 and abs(last(R["E_F_ratio"]) - 1) < 0.01
                                  and nc["E_F"] < 0.02),
    }

    # ---- 12. figures -------------------------------------------------------
    set_style()
    tag = f"(kappa = {kap:g}, xi = {xi:g})"

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.loglog(r_sp, np.maximum(split_rel, 1e-18), "o", ms=2.2, color=C1, label="scalar: |theta_S + theta_L - e^{-kappa r}/r| / (e^{-kappa r}/r)")
    ax.loglog(mags, np.maximum(split_mat, 1e-18), "o", ms=2.2, color=C2, alpha=0.6,
              label="tensor: ||K_S + K_L - K_kappa||_F / ||K_kappa||_F (random r_vec)")
    refline(ax, EPS, "machine epsilon")
    ax.set(xlabel="r", ylabel="relative error", title=f"Splitting identity K_S + K_L = K_kappa {tag}",
           ylim=(1e-18, max(1e-12, 10 * max(split_rel.max(), split_mat.max()))))
    ax.legend(loc="upper left")
    save(fig, figdir, "fig1_splitting_error.png")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    ax.loglog(r_sp, exact_sp, color=INK2, lw=1.0, label="e^{-kappa r}/r")
    ax.loglog(r_sp, thS_sp, color=C1, label="theta_S (short range)")
    ax.loglog(r_sp, thL_sp, color=C2, label="theta_L (long range)")
    ax.set(xlabel="r", ylabel="theta", title="theta_S, theta_L > 0 (log axis)", ylim=(1e-30, 1e4))
    ax.text(0.03, 0.05, f"min theta_S e^(xi^2 r^2) = {R['min theta_S e^{xi^2 r^2} on split grid']:.3g}\n"
            f"min theta_L = {R['min theta_L on split grid']:.3g}\n(r in [1e-3, 30]/xi)",
            transform=ax.transAxes, color=INK2, fontsize=9)
    ax.legend(loc="center left")
    ax = axs[1]
    r_s0 = np.linspace(0.0, 1.5, 301) / xi
    ax.plot(r_s0, K.g_L(r_s0), color=C1, label="theta_L / r^2 = g_L(r) (definition)")
    ax.plot(r_s0, C_q - D_q * r_s0 ** 2, "--", color=C2, label="C - D r^2 (Taylor)")
    ax.plot([0.0], [K.C_closed()], "o", ms=7, color=C3, label=f"C (closed form) = {K.C_closed():.6f}")
    ax.set(xlabel="r", ylabel="theta_L / r^2", title="K_L = C r r^T + O(r^4): smooth at r = 0")
    ax.legend(loc="lower left")
    save(fig, figdir, "fig2_positivity_and_small_r.png")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    ax.semilogy(r_as, K.theta_S(r_as), color=C1, label="theta_S (numerical)")
    ax.semilogy(r_as, K.rho_xi2 * np.exp(-(xi * r_as) ** 2), "--", color=C2, label="rho(xi^2) exp(-xi^2 r^2)")
    ax.set(xlabel="r", ylabel="theta_S = ||K_S||_2", title=f"Real-space pointwise asymptotic {tag}")
    ax.legend()
    ax = axs[1]
    rmid = ra + 0.5 * a_r
    ax.plot(rmid, slope_r, "o-", color=C1, label="fitted slope of log theta_S vs r^2 (windows)")
    rr_f = np.linspace(1.5, 11.5, 200) * a_r
    ax.plot(rr_f, -xi ** 2 - K.c_theta / rr_f ** 4, "--", color=C2, label="-xi^2 - c_theta / r^4 (next order)")
    refline(ax, -xi ** 2, "target -xi^2")
    ax.set(xlabel="window centre r", ylabel="slope", title="Fit of log theta_S vs r^2")
    ax.legend(loc="lower right")
    save(fig, figdir, "fig3_real_space_asymptotic.png")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    damp = np.exp(-k_as ** 2 / (4 * xi ** 2))
    kw20 = k_as <= 20 * b_k
    ax = axs[0]
    ax.semilogy(k_as[kw20], (n_as * damp)[kw20], color=C1, label="||Khat_L(k)||_2 (numerical)")
    ax.semilogy(k_as[kw20], (N_asym * damp)[kw20], "--", color=C2, label="2 pi h / xi^2 exp(-k^2/4xi^2)")
    ax.axvline(K.k_star, color=INK2, lw=0.8)
    ax.text(K.k_star + 0.3 * b_k, 0.05, f"k* = {K.k_star:.3f}\n(norm = A below,\n |A + B k^2| above)",
            color=INK2, fontsize=8, transform=ax.get_xaxis_transform())
    ax.set(xlabel="k", ylabel="operator norm", title=f"Fourier norm asymptotic {tag}")
    ax.legend()
    ax = axs[1]
    kk = (k_as > 0.5 * b_k) & kw20
    ax.semilogy(k_as[kk], (A_as * damp)[kk], color=C1, label="A(k) (numerical)")
    ax.semilogy(k_as[kk], (A_asym * damp)[kk], "--", color=C2, label="4 pi h / k^2 exp(-k^2/4xi^2)")
    ax.set(xlabel="k", ylabel="A(k)", title="A(k) asymptotic")
    ax.legend()
    ax = axs[2]
    ax.plot(ka + b_k, slope_k, "o-", color=C1, label="fitted slope of log||Khat_L|| vs k^2 (windows)")
    kk_f = np.linspace(3, 39, 200) * b_k
    ax.plot(kk_f, -1 / (4 * xi ** 2) - K.c_norm / kk_f ** 4, "--", color=C2, label="-1/(4xi^2) - c_norm / k^4 (next order)")
    refline(ax, -1 / (4 * xi ** 2), "target -1/(4 xi^2)")
    ax.set(xlabel="window centre k", ylabel="slope", title="Fit of log||Khat_L|| vs k^2")
    ax.legend(loc="upper right")
    save(fig, figdir, "fig4_fourier_asymptotic.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    dS = np.exp(-(xi * rc) ** 2)
    ax.semilogy(rc, ES1 * dS, color=C1, label="E_S(rc) numerical (nested quadrature)")
    ax.semilogy(rc, ES2 * dS, "o", ms=4, color=C3, label="E_S(rc) numerical (analytic r-integral)")
    ax.semilogy(rc, ES_asym * dS, "--", color=C2, label="4 sqrt(pi)/xi h rc exp(-xi^2 rc^2)")
    ax.set(xlabel="rc", ylabel="E_S(rc)", title=f"Real-space cutoff tail {tag}")
    ax.legend()
    save(fig, figdir, "fig5_real_space_cutoff.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    dF = np.exp(-kc ** 2 / (4 * xi ** 2))
    ax.semilogy(kc, EF1 * dF, color=C1, label="E_F(kc) numerical (norm split at k*)")
    ax.semilogy(kc[above][::3], (EF2 * dF[above])[::3], "o", ms=4, color=C3, label="E_F(kc) numerical (analytic k-integral, kc >= k*)")
    ax.semilogy(kc, EF_asym * dF, "--", color=C2, label="2/pi h kc exp(-kc^2/4xi^2)")
    ax.set(xlabel="kc", ylabel="E_F(kc)", title=f"Fourier-space cutoff tail {tag}")
    ax.legend()
    save(fig, figdir, "fig6_fourier_cutoff.png")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    m_r, m_rc = r_as >= a_r, rc >= a_r
    ax.plot(r_as[m_r], ratio_th[m_r], color=C1, label="theta_S / asymptotic  (x = r)")
    ax.plot(rc[m_rc], (ES1 / ES_asym)[m_rc], color=C2, label="E_S / asymptotic  (x = rc)")
    xr = np.linspace(1.0, 12.0, 300) * a_r
    ax.plot(xr, 1 + K.c_theta / xr ** 2, "--", color=C1, lw=1.0, label="1 + c_theta / x^2")
    ax.plot(xr[xr <= 8 * a_r], 1 + K.c_ES / xr[xr <= 8 * a_r] ** 2, "--", color=C2, lw=1.0, label="1 + c_ES / x^2")
    refline(ax, 1.0)
    ax.set(xlabel="x", ylabel="numerical / asymptotic", title=f"Real-space ratios {tag}",
           ylim=padded_lim(ratio_th[m_r], (ES1 / ES_asym)[m_rc]))
    ax.legend()
    ax = axs[1]
    m_k, m_kc = k_as >= 3 * b_k, kc >= 3 * b_k
    ax.plot(k_as[m_k], (A_as / A_asym)[m_k], color=C1, label="A / asymptotic  (x = k)")
    ax.plot(k_as[m_k], (n_as / N_asym)[m_k], color=C2, label="||Khat_L|| / asymptotic  (x = k)")
    ax.plot(kc[m_kc], (EF1 / EF_asym)[m_kc], color=C3, label="E_F / asymptotic  (x = kc)")
    xk = np.linspace(3.0, 40.0, 300) * b_k
    ax.plot(xk, 1 + K.c_A / xk ** 2, "--", color=C1, lw=1.0, label="1 + c_A / x^2")
    ax.plot(xk, 1 + K.c_norm / xk ** 2, "--", color=C2, lw=1.0, label="1 + c_norm / x^2")
    ax.plot(xk[xk <= 30 * b_k], 1 + K.c_EF / xk[xk <= 30 * b_k] ** 2, "--", color=C3, lw=1.0, label="1 + c_EF / x^2")
    refline(ax, 1.0)
    ax.set(xlabel="x", ylabel="numerical / asymptotic", title="Fourier-space ratios",
           ylim=padded_lim((A_as / A_asym)[m_k], (n_as / N_asym)[m_k], (EF1 / EF_asym)[m_kc]))
    ax.legend(ncol=2, loc="lower right")
    save(fig, figdir, "fig7_ratios.png")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    kk = k_as <= 12 * b_k
    ax.plot(k_as[kk], A_as[kk], color=C1, label="A e^{k^2/4xi^2}  (transverse eigenvalue, x2)")
    ax.plot(k_as[kk], lp_as[kk], color=C2, label="(A + B k^2) e^{k^2/4xi^2}  (longitudinal eigenvalue)")
    refline(ax, -2 * np.pi * h / xi ** 2, "-2 pi h / xi^2 (limit)")
    ax.axhline(0.0, color=GRID, lw=1.0, zorder=0)
    ax.set(xlabel="k", ylabel="scaled eigenvalue", title="Eigenvalues of Khat_L (scaled)")
    ax.legend(loc="upper right")
    ax = axs[1]
    ax.semilogy(k_h, np.maximum(hank_A, 1e-17), "o-", ms=3, color=C1, label="A: s-integral vs radial Hankel transform")
    ax.semilogy(k_h, np.maximum(hank_B, 1e-17), "s-", ms=3, color=C2, label="B: s-integral vs radial Hankel transform")
    if fft_res is not None:
        order = np.argsort(kf_mag)
        ax.semilogy(kf_mag[order][::40], np.maximum(fft_rel[order][::40], 1e-17), ".", ms=2, color=C3,
                    label="full tensor: A I + B k k^T vs 3D FFT of K_L")
    ax.set(xlabel="k", ylabel="relative difference", title="Independent checks of the Fourier formulas", ylim=(1e-17, 1e-4))
    ax.legend(loc="upper left")
    save(fig, figdir, "fig8_fourier_crosschecks.png")

    return R


def fmt(v):
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def summary_text(R):
    lines = [f"Ewald split of the screened longitudinal kernel: kappa = {R['kappa']:g}, xi = {R['xi']:g}",
             f"z = kappa/(2 xi) = {R['z']:.6g},  h(z) = {R['h(z)']:.12g},  rho(xi^2) = {R['rho(xi^2)']:.12g}", ""]
    for key, val in R.items():
        if key in ("kappa", "xi", "z", "h(z)", "rho(xi^2)"):
            continue
        if isinstance(val, dict):
            lines.append(f"{key}:")
            lines += [f"    {k2:>16s}  {fmt(v2)}" if not isinstance(v2, bool) else f"    {'PASS' if v2 else 'FAIL'}  {k2}"
                      for k2, v2 in val.items()]
        else:
            lines.append(f"{key:>48s}  {fmt(val)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kappa", type=float, default=1.0, help="physical screening parameter kappa > 0")
    ap.add_argument("--xi", type=float, default=1.0, help="Ewald splitting parameter xi > 0")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--no-fft", action="store_true", help="skip the 3D FFT cross-check")
    args = ap.parse_args()
    if args.kappa <= 0 or args.xi <= 0:
        ap.error("kappa and xi must be positive")
    R = run(args.kappa, args.xi, args.outdir, fft=not args.no_fft)
    text = summary_text(R)
    print(text)
    tag = f"kappa{args.kappa:g}_xi{args.xi:g}"
    (args.outdir / "results" / f"summary_{tag}.txt").write_text(text + "\n")
    (args.outdir / "results" / f"summary_{tag}.json").write_text(json.dumps(R, indent=2) + "\n")


if __name__ == "__main__":
    main()
