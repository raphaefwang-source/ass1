"""Fourier transforms of the two candidate tensor kernels g(r) P_L(r), convention F[f](k) = int e^{-ik.r} f dr.

For K = g(r) rhat rhat^T, F[K](k) = lam_T(k) (I - khat khat^T) + lam_L(k) khat khat^T with
    lam_T(k) = 4 pi int_0^inf g r^2 j1(kr)/(kr) dr,
    lam_L(k) = 4 pi int_0^inf g r^2 [j0(kr) - 2 j1(kr)/(kr)] dr,
(using <e^{-ikr mu}>_sphere = j0, <mu^2 e^{-ikr mu}> = j0 - 2 j1/x). These are evaluated numerically with
QUADPACK's Fourier-integral routine (QAWF) and compared with closed forms:

Model A, g = A e^{-kr}/r (derived here; checked numerically):
    tr = 4 pi A/(k^2+kappa^2),  lam_T = 4 pi A [k - kappa arctan(k/kappa)]/k^3,  lam_L = tr - 2 lam_T
    -> lam_T ~ 4 pi A/k^2, lam_L ~ -4 pi A/k^2 for k >> kappa.
Model B, g = B r e^{-kr} = B e^{-kr}/r * r^2:  F[r_i r_j Y] = -d^2/dk_i dk_j [4 pi/(k^2+kappa^2)]
    = 8 pi B/(k^2+kappa^2)^2 I - 32 pi B/(k^2+kappa^2)^3 k k^T
    -> lam_T = 8 pi B/(k^2+kappa^2)^2,  lam_L = 8 pi B (kappa^2 - 3k^2)/(k^2+kappa^2)^3  (~ -24 pi B/k^4)."""
import numpy as np
from scipy import integrate


def analytic_A(k, A=1.0, kappa=1.0):
    k = np.asarray(k, float)
    tr = 4 * np.pi * A / (k ** 2 + kappa ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        lt = np.where(k > 1e-3 * kappa, 4 * np.pi * A * (k - kappa * np.arctan(k / kappa)) / k ** 3,
                      4 * np.pi * A * (1 / (3 * kappa ** 2) - k ** 2 / (5 * kappa ** 4)))
    return tr - 2 * lt, lt


def analytic_B(k, B=1.0, kappa=1.0):
    s = np.asarray(k, float) ** 2 + kappa ** 2
    return 8 * np.pi * B * (kappa ** 2 - 3 * np.asarray(k) ** 2) / s ** 3, 8 * np.pi * B / s ** 2


def numeric(g, k):
    """lam_L, lam_T by QAWF. With x = kr: j0 = sin x/x, j1/x = (sin x - x cos x)/x^3,
    j0 - 2 j1/x = sin x/x - 2 (sin x - x cos x)/x^3."""
    def qs(f):
        return integrate.quad(f, 0, np.inf, weight="sin", wvar=k, limlst=200)[0]

    def qc(f):
        return integrate.quad(f, 0, np.inf, weight="cos", wvar=k, limlst=200)[0]
    # small-r part on [0, a] with the exact (non-oscillatory-weighted) integrand to avoid 1/r^n cancellations
    a = min(1.0, 2.0 / k)

    def jt(r):
        x = k * r
        return np.where(x < 1e-2, 1 / 3 - x ** 2 / 30, (np.sin(x) - x * np.cos(x)) / np.maximum(x, 1e-300) ** 3)

    def jl(r):
        x = k * r
        return np.where(x < 1e-2, 1 / 3 - 3 * x ** 2 / 30,
                        np.sin(x) / np.maximum(x, 1e-300) - 2 * (np.sin(x) - x * np.cos(x)) / np.maximum(x, 1e-300) ** 3)
    lt0 = integrate.quad(lambda r: g(r) * r * r * jt(r), 0, a, limit=200, epsabs=0, epsrel=1e-12)[0]
    ll0 = integrate.quad(lambda r: g(r) * r * r * jl(r), 0, a, limit=200, epsabs=0, epsrel=1e-12)[0]
    sh = lambda f: (lambda u: f(u + a))                                  # noqa: E731
    # tail on [a, inf): rewrite sin(k(u+a)) = sin(ku)cos(ka) + cos(ku)sin(ka), etc.
    ca, sa = np.cos(k * a), np.sin(k * a)

    def tail(fs, fc):
        """int_a^inf [fs(r) sin(kr) + fc(r) cos(kr)] dr."""
        Fs, Fc = sh(fs), sh(fc)
        return (ca * qs(Fs) + sa * qc(Fs)) + (ca * qc(Fc) - sa * qs(Fc))
    lt1 = tail(lambda r: g(r) / (k ** 3 * r), lambda r: -g(r) / k ** 2)
    ll1 = tail(lambda r: g(r) * r / k - 2 * g(r) / (k ** 3 * r), lambda r: 2 * g(r) / k ** 2)
    return 4 * np.pi * (ll0 + ll1), 4 * np.pi * (lt0 + lt1)


def verify(A=1.0, B=1.0, kappa=1.0, ks=(0.05, 0.3, 1.0, 3.0, 10.0, 30.0)):
    rows = []
    for k in ks:
        nA = numeric(lambda r: A * np.exp(-kappa * r) / r, k)
        nB = numeric(lambda r: B * r * np.exp(-kappa * r), k)
        aA, aB = analytic_A(k, A, kappa), analytic_B(k, B, kappa)
        rel = lambda n, a: float(abs(n - a) / max(abs(a), 1e-300))      # noqa: E731
        rows.append(dict(k=k, A_L_num=nA[0], A_L_ana=float(aA[0]), A_T_num=nA[1], A_T_ana=float(aA[1]),
                         B_L_num=nB[0], B_L_ana=float(aB[0]), B_T_num=nB[1], B_T_ana=float(aB[1]),
                         rel_err_max=max(rel(nA[0], aA[0]), rel(nA[1], aA[1]), rel(nB[0], aB[0]), rel(nB[1], aB[1]))))
    return rows


def large_k_slopes(A=1.0, B=1.0, kappa=1.0, k=(100.0, 1000.0)):
    out = {}
    for name, fn in (("A", lambda kk: analytic_A(kk, A, kappa)), ("B", lambda kk: analytic_B(kk, B, kappa))):
        (l1, t1), (l2, t2) = fn(np.array(k[0])), fn(np.array(k[1]))
        out[name] = dict(slope_L=float(np.log(abs(l2) / abs(l1)) / np.log(k[1] / k[0])),
                         slope_T=float(np.log(abs(t2) / abs(t1)) / np.log(k[1] / k[0])),
                         sign_L_large_k=float(np.sign(l2)), k_zero_L=None)
    out["A"]["k_zero_L"] = None
    out["B"]["k_zero_L"] = float(kappa / np.sqrt(3))
    return out
