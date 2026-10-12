"""
Numerical check of the asymptotic statements in "Fast Conserved DPD" (draft 3).

Every "~" in the draft is tested by computing the exact quantity with quadrature
and printing exact / asymptotic.  A correct leading-order formula gives a ratio
that tends to 1; the column s^2*(ratio-1) tending to a constant shows the
relative correction is O(1/s^2), with s = xi*r (real space) or s = k/(2 xi)
(Fourier space), the balanced variable of Eq. (79).

  Eq. (3)          closed form of rho_kappa(s)
  Eq. (10)         theta_S(r) ~ rho(xi^2) exp(-xi^2 r^2)
  Eqs. (11)-(13)   A, B against an independent radial Fourier transform of theta_L
  Eqs. (14), (15)  A, A + k^2 B and ||K^L(k)||_2 for large k
  Eq. (17)         tails E_S(r_c) and E_F(k_c)
  Eqs. (56), (58)  B-spline alias bound (167) and the h^p mesh law, via Eq. (54)
  Eq. (73)         f_dt(lambda) ~ sqrt(2 dt lambda / beta)

Usage:  python check_asymptotics.py [--kappa 1.0] [--xi 1.0] [--no-plot]
"""
import argparse

import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx, spherical_jn, zeta

SQPI = np.sqrt(np.pi)
PI32 = np.pi ** 1.5
GL_X, GL_W = np.polynomial.legendre.leggauss(20)


def gl_nodes(edges):
    """Composite Gauss-Legendre nodes and weights; panels along the last axis of edges."""
    a, b = edges[..., :-1, None], edges[..., 1:, None]
    half = 0.5 * (b - a)
    x = half * GL_X + 0.5 * (a + b)
    w = half * GL_W
    return x.reshape(*edges.shape[:-1], -1), w.reshape(*edges.shape[:-1], -1)


def graded_edges(length, n=50, first=1e-8):
    """Panel edges on [0, length] refined geometrically toward 0. length may be an array."""
    t = np.concatenate(([0.0], np.geomspace(first, 1.0, n)))
    return np.asarray(length)[..., None] * t


# --------------------------------------------------------------------------- #
# Section 2: kernel weights and coefficients
# --------------------------------------------------------------------------- #
def rho(s, kappa):
    """Eq. (3), closed form; erfcx avoids the cancellation for small s."""
    a = kappa / (2.0 * np.sqrt(s))
    return np.exp(-a ** 2) * (2.0 * np.sqrt(s) / SQPI - kappa * erfcx(a))


def rho_integral(s, kappa):
    """Eq. (3), defining integral, with t = s u^2."""
    u, w = gl_nodes(graded_edges(1.0, n=40, first=1e-3))
    s = np.asarray(s)[..., None]
    return 2.0 * np.sqrt(s[..., 0]) / SQPI * np.sum(w * np.exp(-kappa ** 2 / (4.0 * s * u ** 2)), axis=-1)


def theta_S_scaled_closed(r, kappa, xi):
    """exp(xi^2 r^2) * theta_S(r) from Eqs. (142)-(143), written with erfcx."""
    z = kappa / (2.0 * xi)
    c = rho(xi ** 2, kappa)
    return c + np.exp(-z ** 2) / (2.0 * r) * (erfcx(xi * r - z) + erfcx(xi * r + z))


def theta_S_scaled_quad(r, kappa, xi):
    """exp(xi^2 r^2) * r^2 int_{xi^2}^inf rho(s) exp(-s r^2) ds, Eq. (8), with s = xi^2 + u/r^2."""
    u, w = gl_nodes(graded_edges(60.0, n=40, first=1e-6))
    r = np.asarray(r)[..., None]
    return np.sum(w * rho(xi ** 2 + u / r ** 2, kappa) * np.exp(-u), axis=-1)


def theta_L(r, kappa, xi):
    """Eq. (7) via theta_L = exp(-kappa r)/r - theta_S, Eq. (144)."""
    return np.exp(-kappa * r) / r - np.exp(-(xi * r) ** 2) * theta_S_scaled_closed(r, kappa, xi)


def J(m, k, kappa, xi):
    """
    I_m(k) = int_0^{xi^2} rho(s) s^-m exp(-k^2/(4s)) ds = xi^(2-2m) exp(-a - z^2) J_m(k),
    a = k^2/(4 xi^2), z = kappa/(2 xi), obtained exactly with s = xi^2/(1+v).
    """
    z = kappa / (2.0 * xi)
    b = np.asarray(k) ** 2 / (4.0 * xi ** 2) + z ** 2
    v, w = gl_nodes(graded_edges(60.0 / b, n=60))
    t = np.sqrt(1.0 + v)
    rho_t = 2.0 * xi / (SQPI * t) - kappa * erfcx(z * t)
    return np.sum(w * rho_t * t ** (2 * m - 4) * np.exp(-b[..., None] * v), axis=-1)


def AB_scaled(k, kappa, xi):
    """exp(k^2/(4 xi^2)) * (A, B) of Eqs. (12)-(13)."""
    ez = np.exp(-(kappa / (2.0 * xi)) ** 2)
    A = 0.5 * PI32 * xi ** -3 * ez * J(2.5, k, kappa, xi)
    B = -0.25 * PI32 * xi ** -5 * ez * J(3.5, k, kappa, xi)
    return A, B


def AB(k, kappa, xi):
    A, B = AB_scaled(k, kappa, xi)
    damp = np.exp(-np.asarray(k) ** 2 / (4.0 * xi ** 2))
    return A * damp, B * damp


def AB_hankel(k, kappa, xi):
    """
    Independent route to (A, B): radial transforms of theta_L(r) P(r).
      tr K^L(k)            = 3A + k^2 B = 4 pi int r^2 theta_L j0(kr) dr
      k^T K^L(k) k / k^2   =  A + k^2 B = 4 pi int r^2 theta_L (j0 - 2 j2)/3 dr
    """
    r, w = gl_nodes(np.linspace(0.0, 45.0 / kappa + 10.0 / xi, 241))
    f = 4.0 * np.pi * w * r ** 2 * theta_L(r, kappa, xi)
    kr = np.asarray(k)[:, None] * r
    j0, j2 = spherical_jn(0, kr), spherical_jn(2, kr)
    tr = np.sum(f * j0, axis=-1)
    par = np.sum(f * (j0 - 2.0 * j2) / 3.0, axis=-1)
    A = 0.5 * (tr - par)
    return A, (par - A) / np.asarray(k) ** 2


def tail_ES_scaled(rc, kappa, xi):
    """exp(xi^2 rc^2) * E_S(rc), Eq. (157)."""
    rc = np.asarray(rc)[..., None]
    x, w = gl_nodes(graded_edges(np.full(rc.shape[:-1], 12.0 / xi), n=50))
    r = rc + x
    return 4.0 * np.pi * np.sum(
        w * r ** 2 * np.exp(-xi ** 2 * (x ** 2 + 2.0 * rc * x)) * theta_S_scaled_closed(r, kappa, xi), axis=-1)


def tail_EF_scaled(kc, kappa, xi):
    """exp(kc^2/(4 xi^2)) * E_F(kc), Eq. (157), with ||K^L||_2 = max(|A|, |A + k^2 B|)."""
    kc = np.asarray(kc)[..., None]
    y, w = gl_nodes(graded_edges(np.full(kc.shape[:-1], 24.0 * xi), n=40))
    k = kc + y
    A, B = AB_scaled(k, kappa, xi)
    norm = np.maximum(np.abs(A), np.abs(A + k ** 2 * B))
    return np.sum(w * k ** 2 * norm * np.exp(-(k ** 2 - kc ** 2) / (4.0 * xi ** 2)), axis=-1) / (2.0 * np.pi ** 2)


# --------------------------------------------------------------------------- #
# Section 3: B-spline aliasing and the mesh law
# --------------------------------------------------------------------------- #
def alias_1d(x, frac, p, nmax=100):
    """a(x, q) = sum_{n != 0} ((-1)^n x / (x + pi n))^p exp(-2 pi i n q/h), frac = q/h."""
    n = np.concatenate((np.arange(-nmax, 0), np.arange(1, nmax + 1)))
    ratio = ((-1.0) ** n * x[..., None] / (x[..., None] + np.pi * n)) ** p
    return np.sum(ratio * np.exp(-2j * np.pi * n * frac[..., None]), axis=-1)


def alias_factor(a):
    """alpha = prod_l (1 + a_l) - 1, expanded so that tiny alphas keep full precision (axis -2 = l)."""
    a1, a2, a3 = a[..., 0, :], a[..., 1, :], a[..., 2, :]
    return a1 + a2 + a3 + a1 * a2 + a1 * a3 + a2 * a3 + a1 * a2 * a3


def mesh_error(q, L, M, p, kc, kappa, xi, n_off=8, seed=1):
    """
    eps_mesh = 2 max_i sum_{j != i} ||E_ij||_2 with ideal deconvolution, Eq. (54):
      E_ij = (1/V) sum_{0<|k|<=kc} K^L(k) exp(ik.(qi-qj)) (conj(alpha_i) + alpha_j + conj(alpha_i) alpha_j).
    Returns the RMS over n_off random global grid offsets and max |alpha| / (h|k|/pi)^p.
    """
    h = L / M
    mmax = int(np.floor(kc * L / (2.0 * np.pi)))
    grid = np.arange(-mmax, mmax + 1)
    kint = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    kvec = 2.0 * np.pi / L * kint
    kabs = np.linalg.norm(kvec, axis=1)
    keep = (kabs > 0) & (kabs <= kc)
    kint, kvec, kabs = kint[keep], kvec[keep], kabs[keep]
    A, B = AB(kabs, kappa, xi)

    rng = np.random.default_rng(seed)
    frac = (q[None] + rng.uniform(0.0, h, (n_off, 1, 3))) / h                 # (o, N, 3)
    xcomp = (2.0 * np.pi / L * grid) * h / 2.0                                 # distinct k_l h/2
    a1 = alias_1d(xcomp[None, :, None, None], frac[:, None], p)               # (o, m, N, 3)
    a1 = np.moveaxis(a1, -1, 1)                                                # (o, 3, m, N)
    a = a1[:, np.arange(3), kint + mmax, :]                                    # (o, k, 3, N)
    alpha = alias_factor(a)                                                    # (o, k, N)

    phase = np.exp(1j * q @ kvec.T).T                                          # (k, N)
    c = (phase[:, :, None] * np.conj(phase[:, None, :]))[None] * (
        np.conj(alpha)[..., :, None] + alpha[..., None, :] + np.conj(alpha)[..., :, None] * alpha[..., None, :])
    V = L ** 3
    E = (np.einsum("okij,k->oij", c, A)[..., None, None] * np.eye(3)
         + np.einsum("okij,k,ka,kb->oijab", c, B, kvec, kvec)) / V
    imag = np.abs(E.imag).max() / max(np.abs(E.real).max(), 1e-300)
    norms = np.abs(np.linalg.eigvalsh(E.real)).max(-1)                         # (o, N, N)
    norms[:, np.arange(len(q)), np.arange(len(q))] = 0.0
    eps = 2.0 * norms.sum(-1).max(-1)
    bound_ratio = (np.abs(alpha) / ((h * kabs / np.pi) ** p)[None, :, None]).max()
    return np.sqrt(np.mean(eps ** 2)), bound_ratio, imag


# --------------------------------------------------------------------------- #
def table(title, header, rows, fmt):
    print(f"\n{title}")
    print("  " + "  ".join(f"{h:>13s}" for h in header))
    for row in rows:
        print("  " + "  ".join(f"{v:13{f}}" for v, f in zip(row, fmt)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--xi", type=float, default=1.0)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    kappa, xi = args.kappa, args.xi
    z = kappa / (2.0 * xi)
    c = rho(xi ** 2, kappa)
    print(f"kappa = {kappa}, xi = {xi}, z = kappa/(2 xi) = {z:.4f}, rho(xi^2) = {c:.6e}")

    # ---- Eq. (3) and quadrature sanity checks ------------------------------------------
    s_test = np.geomspace(1e-2, 1e2, 9)
    err3 = np.max(np.abs(rho(s_test, kappa) / rho_integral(s_test, kappa) - 1.0))
    k_test = np.array([0.3, 1.0, 3.0]) * xi
    I_quad = np.array([quad(lambda s: rho(s, kappa) * s ** -2.5 * np.exp(-k ** 2 / (4 * s)), 0, xi ** 2,
                            epsabs=0, epsrel=1e-12, limit=200)[0] for k in k_test])
    I_gl = xi ** -3 * np.exp(-k_test ** 2 / (4 * xi ** 2) - z ** 2) * J(2.5, k_test, kappa, xi)
    r_test = np.array([0.3, 1.0, 3.0]) / xi
    errS = np.max(np.abs(theta_S_scaled_closed(r_test, kappa, xi) / theta_S_scaled_quad(r_test, kappa, xi) - 1))
    print(f"Eq. (3)   closed form vs integral, max rel. diff          : {err3:.2e}")
    print(f"I_m(k)    Gauss-Legendre vs scipy.quad, max rel. diff      : {np.max(np.abs(I_gl / I_quad - 1)):.2e}")
    print(f"Eq. (142) theta_S closed form vs direct quadrature of (8)  : {errS:.2e}")

    # ---- Eqs. (11)-(13): Fourier transform against a radial Hankel transform -----------
    kf = np.array([0.25, 0.5, 1.0, 2.0, 3.0]) * xi
    A1, B1 = AB(kf, kappa, xi)
    A2, B2 = AB_hankel(kf, kappa, xi)
    dk = 1e-4 * kf
    Ap, _ = AB(kf + dk, kappa, xi)
    Am, _ = AB(kf - dk, kappa, xi)
    Bfd = (Ap - Am) / (2 * dk) / kf
    table("Eqs. (11)-(13): s-integral vs radial FT of theta_L(r)P(r); B = A'/k (Eq. 153)",
          ["k/xi", "A", "rel.diff A", "B", "rel.diff B", "B vs A'/k"],
          zip(kf / xi, A1, np.abs(A1 / A2 - 1), B1, np.abs(B1 / B2 - 1), np.abs(Bfd / B1 - 1)),
          ["g", ".6e", ".1e", ".6e", ".1e", ".1e"])

    # ---- Eq. (10) and E_S of Eq. (17) ---------------------------------------------------
    s = np.array([1, 1.5, 2, 3, 4, 6, 8, 12, 16], dtype=float)
    r = s / xi
    ratio10 = theta_S_scaled_closed(r, kappa, xi) / c
    ratioES = tail_ES_scaled(r, kappa, xi) / (2 * np.pi * c * r / xi ** 2)
    table("Eq. (10) theta_S(r) / [rho(xi^2) e^{-xi^2 r^2}]  and  Eq. (17) E_S / asymptote,  s = xi r",
          ["s", "ratio (10)", "s^2(r-1)", "ratio E_S", "s^2(r-1)"],
          zip(s, ratio10, s ** 2 * (ratio10 - 1), ratioES, s ** 2 * (ratioES - 1)),
          ["g", ".8f", ".5f", ".8f", ".5f"])
    c10 = 1.0 / (2.0 * (1.0 - SQPI * z * erfcx(z)))
    print(f"  exact limit of s^2(ratio-1) for Eq. (10): 1/(2(1 - sqrt(pi) z erfcx z)) = {c10:.5f}"
          f"  (~1/2 for z->0, ~z^2 for z>>1: the asymptote needs s >> max(1, z))")

    # ---- Eqs. (14), (15) and E_F of Eq. (17) -------------------------------------------
    k = 2 * xi * s
    As, Bs = AB_scaled(k, kappa, xi)
    lam_par = As + k ** 2 * Bs
    rA = As / (2 * PI32 / (xi * k ** 2) * c)
    rP = lam_par / (-PI32 / xi ** 3 * c)
    rN = np.maximum(np.abs(As), np.abs(lam_par)) / (PI32 / xi ** 3 * c)
    rEF = tail_EF_scaled(k, kappa, xi) / (c * k / (SQPI * xi))
    table("Eqs. (14),(15),(17) exact / asymptote,  s = k/(2 xi)",
          ["s", "A", "A+k^2B", "||K^L||_2", "s^2(r-1)", "E_F", "s^2(r-1)"],
          zip(s, rA, rP, rN, s ** 2 * (rN - 1), rEF, s ** 2 * (rEF - 1)),
          ["g", ".8f", ".8f", ".8f", ".5f", ".8f", ".5f"])

    kk = np.linspace(0.05, 12, 4000) * xi
    Ak, Bk = AB_scaled(kk, kappa, xi)
    lp = Ak + kk ** 2 * Bk
    k0 = kk[np.argmax(lp < 0)]
    k1 = kk[np.argmax(np.abs(lp) > np.abs(Ak))]
    print(f"\nSection 2.5: lambda_par < 0 for k/xi > {k0 / xi:.4f};  ||K^L||_2 = |lambda_par| for k/xi > {k1 / xi:.4f}"
          f"  (i.e. s > {k1 / (2 * xi):.3f})")

    # ---- Eq. (73) -----------------------------------------------------------------------
    beta, dt, m = 1.0, 0.01, 1.0
    lam = np.geomspace(1e-6, 1e2, 5)
    f = np.sqrt(m / beta * (1 - np.exp(-2 * dt * lam / m)))
    print("Eq. (73)  f(lambda)/sqrt(2 dt lambda/beta) at lambda = " + ", ".join(f"{x:.0e}" for x in lam) + " : "
          + ", ".join(f"{x:.6f}" for x in f / np.sqrt(2 * dt * lam / beta)))

    # ---- Eqs. (56), (58), (167): mesh law -----------------------------------------------
    L, N, kc = 6.0, 30, 4.0
    q = np.random.default_rng(0).uniform(0, L, (N, 3))
    Ms = np.array([12, 16, 20, 24, 32, 40, 48, 64, 80])
    ps = np.arange(2, 7)
    res = np.array([[mesh_error(q, L, M, p, kc, kappa, xi) for M in Ms] for p in ps])  # (p, M, 3)
    eps, bnd, imag = res[..., 0], res[..., 1], res[..., 2]
    xh = xi * L / Ms
    slopes = np.polyfit(np.log(xh[-3:]), np.log(eps[:, -3:].T), 1)[0]
    print(f"\nEqs. (54),(58): config N={N}, L={L}, kc={kc}, RMS over 8 grid offsets "
          f"(max |Im E|/|Re E| = {imag.max():.1e})")
    print("  xi*h        " + "  ".join(f"p={p:<9d}" for p in ps))
    for i, x in enumerate(xh):
        print(f"  {x:<10.4f}  " + "  ".join(f"{e:.3e}  " for e in eps[:, i]))
    print("  slope(3 finest) " + "  ".join(f"{sl:.3f}" for sl in slopes))
    print("  C_M = eps/(xi h)^p on finest mesh: " + "  ".join(f"{e:.2e}" for e in eps[:, -1] / xh[-1] ** ps))
    print("  Eq. (167): max |alpha|/(h|k|/pi)^p vs bound (1+2 zeta(p))^3 - 1:")
    for p, b in zip(ps, bnd.max(1)):
        print(f"    p={p}: {b:.4f} <= {(1 + 2 * zeta(p)) ** 3 - 1:.4f}  {'OK' if b <= (1 + 2 * zeta(p)) ** 3 - 1 else 'VIOLATED'}")

    if not args.no_plot:
        plot(s, [ratio10, ratioES], [rA, rN, rEF], xh, eps, ps, kappa, xi)


def plot(s, real_ratios, fourier_ratios, xh, eps, ps, kappa, xi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    markers = ["o", "s", "^", "D", "v"]
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "text.color": ink, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=surface)
    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(True, which="major", color=grid, lw=0.6)

    def rel_panel(ax, ratios, labels, title, xlabel):
        for i, (rr, lab) in enumerate(zip(ratios, labels)):
            ax.loglog(s, np.abs(rr - 1), color=series[i], lw=2, marker=markers[i], ms=5, label=lab)
            ax.annotate(lab, (s[-1], abs(rr[-1] - 1)), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=ink2)
        ref = np.abs(ratios[0][-1] - 1) * (s[-1] / s) ** 2
        ax.loglog(s, ref, color=ink2, lw=1, ls=":", label=r"$\propto s^{-2}$")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("|exact / asymptotic - 1|")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(s[0] * 0.9, s[-1] * 2.2)
        ax.legend(frameon=False, fontsize=8, loc="lower left")

    rel_panel(axes[0], real_ratios, [r"$\theta_S$ (10)", r"$E_S$ (17)"],
              "Real space: relative error of asymptote", r"$s=\xi r$")
    rel_panel(axes[1], fourier_ratios, [r"$A$ (14)", r"$\|\hat K^L\|_2$ (15)", r"$E_F$ (17)"],
              "Fourier space: relative error of asymptote", r"$s=k/(2\xi)$")

    ax = axes[2]
    for i, p in enumerate(ps):
        ax.loglog(xh, eps[i], color=series[i], lw=2, marker=markers[i], ms=5, label=f"p = {p}")
        ax.loglog(xh, eps[i, -1] * (xh / xh[-1]) ** p, color=series[i], lw=1, ls=":")
        ax.annotate(f"p={p}", (xh[0], eps[i, 0]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=ink2)
    ax.set_xlabel(r"$\xi h$")
    ax.set_ylabel(r"$\epsilon_{\rm mesh}$ (RMS over grid offsets)")
    ax.set_title(r"Mesh law (58): dotted = slope $p$", loc="left", fontsize=10)
    ax.set_xlim(xh[-1] * 0.9, xh[0] * 1.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(rf"Asymptotic checks, $\kappa={kappa:g}$, $\xi={xi:g}$", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    out = f"asymptotic_check_kappa{kappa:g}_xi{xi:g}.png"
    fig.savefig(out, dpi=150, facecolor=surface)
    print(f"\nfigure saved to {out}")


if __name__ == "__main__":
    main()
