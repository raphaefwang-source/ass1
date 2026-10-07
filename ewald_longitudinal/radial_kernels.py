#!/usr/bin/env python3
"""
Radial laws A and B for the pure-longitudinal friction kernel, with explicit amplitude, on top of the
existing Ewald/PPPM machinery (verify_ewald_longitudinal, test_parameter_selection, test_lanczos_fdt).

    K(r_vec) = g(r) P(r_vec),  P = r_vec r_vec^T / r^2   (no transverse projector)
    law A: g(r) = c_A exp(-kappa r)/r,   c_A = gamma r_ref exp(kappa r_ref)
    law B: g(r) = c_B r exp(-kappa r),   c_B = gamma exp(kappa r_ref)/r_ref
so g_A(r_ref) = g_B(r_ref) = gamma (toy normalisation of prl_toy_models; not an equal-total-friction rule).

Gaussian-scale representation, split at s = xi^2:
    K = c r_vec r_vec^T int_0^inf mu(s) exp(-s r^2) ds,
    mu_A = rho_kappa (the project's density),   mu_B = w_kappa = exp(-kappa^2/(4s))/sqrt(pi s) = rho_kappa'.
Both densities are positive, so K_S (s > xi^2) and K_L (s < xi^2) are pure longitudinal with
nonnegative profiles theta_S, theta_L for both laws.

Closed forms (law B), from int_{xi^2}^inf w exp(-s r^2) ds = Y_S(r):
    Y_S(r) e^{xi^2 r^2} = e^{-z^2}/(2r) [erfcx(xi r + z) + erfcx(xi r - z)],   z = kappa/(2 xi)
    theta_S^B = r^2 Y_S,          theta_S^A = r^2 int rho e^{-s r^2} = rho(xi^2) e^{-xi^2 r^2} + Y_S   (project form)
Fourier coefficients (convention int e^{-ik.r} K dr), Khat_L = A(k) I + B(k) k k^T:
    A = (pi^{3/2}/2) int_0^{xi^2} mu s^{-5/2} e^{-k^2/4s} ds,   B = -(pi^{3/2}/4) int_0^{xi^2} mu s^{-7/2} e^{-k^2/4s} ds
evaluated with the project's substitution s = 1/(v + xi^-2), v = tau/lam, lam = (k^2+kappa^2)/4 (decay e^{-tau}
for every k >= 0, including k = 0, which is finite because kappa > 0).
Full-space integral (k = 0 of the unsplit kernel): ghat(0)/3 I with ghat_A(0) = 4 pi c_A/kappa^2,
ghat_B(0) = 24 pi c_B/kappa^4.

The fast operator below reuses GammaH.matvec, Mesh and the Lanczos routines unchanged; only the kernel object
(theta_S_scaled_closed, AB_scaled, xi, kap) is swapped. Model unchanged otherwise; nothing is clipped.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import special

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_ewald_longitudinal import EwaldLongitudinal, blockwise, rho, w_density, PI32  # noqa: E402
from test_lanczos_fdt import GammaH, real_pairs  # noqa: E402
from test_equal_accuracy_xi import make_mesh  # noqa: E402

LAWS = ("A", "B")


def amplitude(law, gamma, kappa, r_ref):
    """c such that g(r_ref) = gamma for the chosen law."""
    if law == "A":
        return gamma * r_ref * np.exp(kappa * r_ref)
    if law == "B":
        return gamma * np.exp(kappa * r_ref) / r_ref
    raise ValueError(f"law must be A or B, got {law!r}")


class RadialEwald:
    """Kernel object with the interface used by Mesh/GammaH (xi, kap, theta_S_scaled_closed, AB_scaled)."""

    def __init__(self, law, kappa, xi, amp=1.0):
        if law not in LAWS:
            raise ValueError(f"law must be A or B, got {law!r}")
        if kappa <= 0 or xi <= 0 or amp < 0:
            raise ValueError("Require kappa > 0, xi > 0 and amplitude >= 0.")
        self.law, self.amp = law, float(amp)
        self.base = EwaldLongitudinal(kappa, xi)          # quadrature rules and law-A formulas of the project
        self.kap, self.xi, self.z = self.base.kap, self.base.xi, self.base.z
        self.mu = (lambda s: rho(s, self.kap)) if law == "A" else (lambda s: w_density(s, self.kap))
        self.mu_low = self.mu(self.base.s_low[0])

    # ---- real space ----------------------------------------------------------
    def theta_full(self, r):
        r = np.asarray(r, float)
        g = np.exp(-self.kap * r) / r if self.law == "A" else r * np.exp(-self.kap * r)
        return self.amp * g

    def theta_S_scaled_closed(self, r):
        """theta_S(r) e^{xi^2 r^2} (closed form)."""
        r = np.asarray(r, float)
        if self.law == "A":
            return self.amp * self.base.theta_S_scaled_closed(r)
        a = self.xi * r
        return self.amp * 0.5 * r * np.exp(-self.z ** 2) * (special.erfcx(a + self.z) + special.erfcx(a - self.z))

    def theta_S_scaled_quad(self, r):
        """Same quantity from the s-integral (cross-check): r^2 int_0^inf mu(xi^2 + t/r^2) e^{-t} dt / r^2."""
        t, wt = self.base.t_rule
        we = wt * np.exp(-t)
        f = lambda rb: self.mu(self.xi ** 2 + t[None, :] / rb[:, None] ** 2) @ we
        return self.amp * blockwise(f, np.asarray(r, float), t.size)

    def theta_S(self, r):
        r = np.asarray(r, float)
        return self.theta_S_scaled_closed(r) * np.exp(-(self.xi * r) ** 2)

    def theta_L(self, r):
        """r^2 int_0^{xi^2} mu e^{-s r^2} ds (s-integral; no cancellation at small r)."""
        s, ws = self.base.s_low
        wr = ws * self.mu_low
        r = np.asarray(r, float)
        return self.amp * r ** 2 * blockwise(lambda rb: np.exp(-np.outer(rb ** 2, s)) @ wr, r, s.size)

    def theta(self, r, which):
        """Profiles-compatible profile: 'full', 'S' or 'L' (theta_L from full - S where well conditioned)."""
        r = np.asarray(r, float)
        if which == "full":
            return self.theta_full(r)
        th_S = self.theta_S(r)
        if which == "S":
            return th_S
        th_L = self.theta_full(r) - th_S
        near = r < 1.5 / self.xi
        if near.any():
            th_L[near] = self.theta_L(r[near])
        return th_L

    # ---- Fourier space -------------------------------------------------------
    def AB_scaled(self, k):
        """(A(k), B(k)) * exp(k^2/(4 xi^2)) for K_L, same substitution and rule as the project."""
        tau, wt = self.base.tau_rule

        def f(kb):
            k2 = kb[:, None] ** 2
            lam = (k2 + self.kap ** 2) / 4
            v = tau[None, :] / lam
            s = 1.0 / (v + self.xi ** -2)
            base = self.mu(s) * np.exp(-k2 * v / 4) * s ** 2 / lam
            return np.stack([0.5 * PI32 * ((base * s ** -2.5) @ wt), -0.25 * PI32 * ((base * s ** -3.5) @ wt)], axis=-1)

        out = blockwise(f, np.asarray(k, float), tau.size)
        return self.amp * out[..., 0], self.amp * out[..., 1]

    def ghat0(self):
        """int g(r) d^3r of the unsplit kernel (its k = 0 tensor is ghat0/3 I)."""
        if self.law == "A":
            return self.amp * 4 * np.pi / self.kap ** 2
        return self.amp * 24 * np.pi / self.kap ** 4


def toy_kernel(law, gamma, kappa, r_ref, xi):
    return RadialEwald(law, kappa, xi, amplitude(law, gamma, kappa, r_ref))


class FastFriction:
    """PPPM friction operator for one box: mesh/influence function built once, Gamma_h(q) per configuration.

    gamma_h(x) returns a GammaH instance whose matvec / dense are the project's literal implementations.
    """

    def __init__(self, L, kernel, s, eta, p):
        self.L, self.K, self.s, self.eta, self.p = float(L), kernel, float(s), float(eta), int(p)
        self.rc, self.kc = s / kernel.xi, 2 * kernel.xi * s
        self.mesh = make_mesh(self.L, kernel, self.kc, eta, p)

    def gamma_h(self, x):
        x = np.asarray(x, float) % self.L
        op = GammaH.__new__(GammaH)
        op.x, op.L, op.N, op.rc, op.kc = x, self.L, len(x), self.rc, self.kc
        op.i, op.j, op.d = real_pairs(x, self.L, self.rc)
        r = np.linalg.norm(op.d, axis=1)
        op.w = self.K.theta_S_scaled_closed(r) * np.exp(-(self.K.xi * r) ** 2) / r ** 2
        op.mesh = self.mesh
        op.idx, op.wt = self.mesh.flat(x)
        op.D = self.mesh.degree(op.idx, op.wt)
        op.n_actions, op.t_actions = 0, 0.0
        return op

    def record(self):
        m = self.mesh
        return dict(law=self.K.law, amplitude=self.K.amp, kappa=self.K.kap, xi=self.K.xi, s=self.s, rc=self.rc,
                    kc=self.kc, eta_N=self.eta, eta_actual=m.eta_actual, p=self.p, M=m.M, n_modes=m.n_modes,
                    n_excluded=m.n_excluded, k0_retained=bool(np.any(np.all(m.mv == 0, axis=1))))
