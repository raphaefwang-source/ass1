#!/usr/bin/env python3
"""Independent numerical checks of the closed forms quoted in the manuscript.

  - Lemma (integration by parts):  exp(-kappa r)/r^3 = int_0^inf rho_kappa(s) exp(-s r^2) ds
  - theta_S closed form at the production values kappa = 0.7, xi = 0.85
  - Hessian remark: psi, phi closed forms (E1) and K = Hess(phi) + psi I
  - Appendix: Fourier transform of the full unit kernel, a(k) I + b(k) khat khat^T
Requires numpy and scipy. Usage: python3 manuscript/scripts/verify_closed_forms.py
"""
import numpy as np
from scipy import integrate, special

KAP, XI = 0.7, 0.85


def rho(s, kap=KAP):
    return 2 * np.sqrt(s / np.pi) * np.exp(-kap ** 2 / (4 * s)) - kap * special.erfc(kap / (2 * np.sqrt(s)))


def sint(f, lo=0.0):
    q = lambda a, b: integrate.quad(f, a, b, limit=500, epsabs=0, epsrel=1e-13)[0]
    return q(lo, 1.0) + q(1.0, np.inf) if lo < 1 else q(lo, np.inf)


def main():
    worst = {}
    for r in (0.3, 1.0, 2.5, 5.0):
        x = KAP * r
        rep = sint(lambda s: rho(s) * np.exp(-s * r * r)) / (np.exp(-x) / r ** 3) - 1
        psi = sint(lambda s: rho(s) / (2 * s) * np.exp(-s * r * r))
        phi = sint(lambda s: rho(s) / (4 * s * s) * np.exp(-s * r * r))
        psic = np.exp(-x) / r - KAP * special.exp1(x)
        phic = ((1 - x) * np.exp(-x) + x * x * special.exp1(x)) / (2 * KAP)
        for k, v in (("representation", rep), ("psi", psi / psic - 1), ("phi", phi / phic - 1)):
            worst[k] = max(worst.get(k, 0), abs(v))

    z = KAP / (2 * XI)
    for r in (0.5, 1.5, 3.0, 4.8):
        q = r * r * sint(lambda s: rho(s) * np.exp(-s * r * r), lo=XI ** 2)
        c = (np.exp(KAP * r) * special.erfc(XI * r + z) + np.exp(-KAP * r) * special.erfc(XI * r - z)) / (2 * r) \
            + rho(XI ** 2) * np.exp(-XI ** 2 * r * r)
        worst["theta_S"] = max(worst.get("theta_S", 0), abs(q / c - 1))

    def phif(v):
        x = KAP * np.linalg.norm(v)
        return ((1 - x) * np.exp(-x) + x * x * special.exp1(x)) / (2 * KAP)

    v, h = np.array([0.7, -0.4, 1.1]), 1e-4
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * h, np.eye(3)[j] * h
            H[i, j] = (phif(v + ei + ej) - phif(v + ei - ej) - phif(v - ei + ej) + phif(v - ei - ej)) / (4 * h * h)
    r = np.linalg.norm(v)
    Kc = H + (np.exp(-KAP * r) / r - KAP * special.exp1(KAP * r)) * np.eye(3)
    Kt = np.exp(-KAP * r) * np.outer(v, v) / r ** 3
    worst["hessian (finite differences)"] = np.abs(Kc - Kt).max() / np.abs(Kt).max()

    j0 = lambda x: np.sin(x) / x
    j1 = lambda x: (np.sin(x) - x * np.cos(x)) / x ** 2
    for k in (0.05, 0.9, 1.3, 4.0):
        a = 4 * np.pi * (k - KAP * np.arctan(k / KAP)) / k ** 3
        b = 4 * np.pi / (k ** 2 + KAP ** 2) - 3 * a
        an = 4 * np.pi * integrate.quad(lambda r: r * np.exp(-KAP * r) * j1(k * r) / (k * r), 0, np.inf, limit=2000)[0]
        bn = 4 * np.pi * integrate.quad(lambda r: r * np.exp(-KAP * r) * (j0(k * r) - 3 * j1(k * r) / (k * r)),
                                        0, np.inf, limit=2000)[0]
        worst["fourier a"] = max(worst.get("fourier a", 0), abs(a / an - 1))
        worst["fourier b"] = max(worst.get("fourier b", 0), abs(b / bn - 1))
        if k == 1.3:
            print(f"lambda_parallel(1.3) = {a + b:.4f}")
    for k, v in worst.items():
        print(f"{k:32s} max relative error {v:.2e}")


if __name__ == "__main__":
    main()
