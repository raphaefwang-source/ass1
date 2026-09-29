#!/usr/bin/env python3
"""
Is the periodic particle graph Laplacian built from the screened longitudinal
Ewald pieces positive semidefinite, even though Khat_L(k) is indefinite?

    (Gamma X)_i = sum_{j != i} K_ij (X_i - X_j),   K_ij = K^per(x_i - x_j),
    Gamma_ij = -K_ij (i != j),   Gamma_ii = sum_{j != i} K_ij,
    X^T Gamma X = 1/2 sum_{i != j} (X_i - X_j)^T K_ij (X_i - X_j).

If every block K_ij is PSD, every term of the quadratic form is >= 0, so Gamma >= 0.
That condition is pointwise in r.  PSD of Khat(k) is a different condition: it
controls the convolution operator f -> K * f (and Gram-type matrices
M_ij = K^per(x_i - x_j) including the self-image block), not the Laplacian.

(a) exact:     K^per by direct image sums (kappa > 0 makes them converge exponentially).
(b) truncated: K^{per,kc}(r) = (1/V) sum_{|k| <= kc} Khat_L(k) e^{i k.r}, k = 2 pi m / L.
No negative eigenvalue is clipped or otherwise modified anywhere.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_ewald_longitudinal import EwaldLongitudinal  # noqa: E402

I3 = np.eye(3)


# ----------------------------------------------------------------------------
# Radial profiles theta(r):  K(r_vec) = theta(r) P(r_vec)
# ----------------------------------------------------------------------------
class Profiles:
    def __init__(self, K):
        self.K = K

    def theta(self, r, which):
        K = self.K
        if which == "full":
            return np.exp(-K.kap * r) / r
        th_S = K.theta_S_scaled_closed(r) * np.exp(-(K.xi * r) ** 2)
        if which == "S":
            return th_S
        # theta_L = e^{-kappa r}/r - theta_S is well conditioned for xi r >= 1.5;
        # closer in, use the s-integral definition (no cancellation).
        th_L = np.exp(-K.kap * r) / r - th_S
        near = r < 1.5 / K.xi
        if near.any():
            th_L[near] = K.theta_L(r[near])
        return th_L


def image_vectors(L, rcut):
    nmax = int(np.ceil(rcut / L)) + 1
    m = np.arange(-nmax, nmax + 1)
    n = np.stack(np.meshgrid(m, m, m, indexing="ij"), axis=-1).reshape(-1, 3)
    return n * L


def periodic_kernel(prof, d, which, L, rcut, chunk=256):
    """sum_n K(d + nL) over |d + nL| <= rcut, excluding only an exactly zero vector (n = 0 self term)."""
    d = np.atleast_2d(d)
    nL = image_vectors(L, rcut)
    out = np.zeros((d.shape[0], 3, 3))
    for c0 in range(0, nL.shape[0], chunk):  # memory blocks over image vectors
        r = d[:, None, :] + nL[None, c0:c0 + chunk, :]
        rr = np.linalg.norm(r, axis=-1)
        keep = (rr <= rcut) & (rr > 0)
        w = np.zeros_like(rr)
        w[keep] = prof.theta(rr[keep], which) / rr[keep] ** 2
        out += np.einsum("mc,mci,mcj->mij", w, r, r)
    return out


# ----------------------------------------------------------------------------
# Graph Laplacian and friends
# ----------------------------------------------------------------------------
def laplacian(Kp, pi, pj, N):
    G = np.zeros((N, 3, N, 3))
    G[pi, :, pj, :] = -Kp
    G[pj, :, pi, :] = -Kp
    D = np.zeros((N, 3, 3))
    np.add.at(D, pi, Kp)
    np.add.at(D, pj, Kp)
    idx = np.arange(N)
    G[idx, :, idx, :] = D
    return G.reshape(3 * N, 3 * N)


def gram(Kp, Kself, pi, pj, N):
    """M_ij = K^per(x_i - x_j) including the self-image block M_ii = sum_{n != 0} K(nL)."""
    M = np.zeros((N, 3, N, 3))
    M[pi, :, pj, :] = Kp
    M[pj, :, pi, :] = Kp
    idx = np.arange(N)
    M[idx, :, idx, :] = Kself
    return M.reshape(3 * N, 3 * N)


def min_eig_blocks(Kb):
    """Smallest eigenvalue of each 3x3 block divided by its largest."""
    ev = np.linalg.eigvalsh(Kb)
    return ev[:, 0] / np.abs(ev).max(axis=1)


def fourier_kernel(dvec, kv, A, B, V):
    """(1/V) sum_k [A I + B k k^T] cos(k.d)   (Khat_L is real and even)."""
    C = np.cos(dvec @ kv.T)
    return ((C @ A) / V)[:, None, None] * I3 + np.einsum("pk,k,ki,kj->pij", C, B, kv, kv) / V


def spectrum(G):
    return np.linalg.eigvalsh(0.5 * (G + G.T))


def fmt_list(v):
    return " ".join(f"{x: .3e}" for x in v)


# ----------------------------------------------------------------------------
def run(kappa, xi, N, L, seed, rcut, kcs, out):
    K = EwaldLongitudinal(kappa, xi)
    prof = Profiles(K)
    V = L ** 3
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, L, size=(N, 3))
    pi, pj = np.triu_indices(N, 1)
    d = x[pi] - x[pj]
    d -= L * np.round(d / L)                      # minimum image (any lift gives the same lattice sum)
    rep = []
    P = rep.append
    P(f"kappa = {kappa:g}, xi = {xi:g}, N = {N}, L = {L:g}, seed = {seed}, image cutoff R = {rcut:g}")
    P(f"pairs = {len(pi)}, min pair distance = {np.linalg.norm(d, axis=1).min():.4f}, "
      f"lattice spacing 2pi/L = {2 * np.pi / L:.4f}")

    # ---- 2. direct image sums + convergence --------------------------------
    Kp = {w: periodic_kernel(prof, d, w, L, rcut) for w in ("full", "S", "L")}
    zero = np.zeros((1, 3))
    Kself = {w: periodic_kernel(prof, zero, w, L, rcut)[0] for w in ("full", "S", "L")}
    Kp_lo = {w: periodic_kernel(prof, d, w, L, rcut - 10) for w in ("full", "L")}
    P("\n[2] image-sum convergence: max|K(R) - K(R-10)| / max|K(R)|")
    for w in ("full", "L"):
        P(f"    {w:5s} {np.abs(Kp[w] - Kp_lo[w]).max() / np.abs(Kp[w]).max():.2e}   "
          f"(tail estimate e^(-kappa (R-10)) = {np.exp(-kappa * (rcut - 10)):.1e})")
    P(f"    pair-kernel split: max|K_full - K_S - K_L| / max|K_full| = "
      f"{np.abs(Kp['full'] - Kp['S'] - Kp['L']).max() / np.abs(Kp['full']).max():.2e}")

    # ---- 3. pointwise PSD of periodic kernels ------------------------------
    dr = rng.uniform(-L / 2, L / 2, size=(3000, 3))
    dr = np.concatenate([dr, d, np.array([[1e-3, 0, 0], [L / 2, L / 2, L / 2], [L / 2, 0, 0]])])
    P(f"\n[3] pointwise PSD of K^per(r): min over {len(dr)} separations of lambda_min/lambda_max of the 3x3 block")
    for w in ("full", "S", "L"):
        Kr = periodic_kernel(prof, dr, w, L, rcut)
        m = min_eig_blocks(Kr)
        P(f"    K_{w:4s}^per   min ratio = {m.min(): .3e}   blocks with ratio < 0 (raw): {(m < 0).sum():4d},"
          f"  < -1e-13 (beyond roundoff): {(m < -1e-13).sum()}")

    # ---- 4/5. Laplacians and spectra ---------------------------------------
    G = {w: laplacian(Kp[w], pi, pj, N) for w in ("full", "S", "L")}
    P(f"\n[5] ||Gamma_full - Gamma_S - Gamma_L||_max / ||Gamma_full||_max = "
      f"{np.abs(G['full'] - G['S'] - G['L']).max() / np.abs(G['full']).max():.2e}")
    ev = {w: spectrum(G[w]) for w in G}
    for w in ("full", "S", "L"):
        e = ev[w]
        P(f"    Gamma_{w:4s} lambda_max = {e[-1]:.4e}   smallest 10:")
        P(f"        {fmt_list(e[:10])}")
        P(f"        min/max = {e[0] / e[-1]: .2e},  #eig < 0 (raw) = {(e < 0).sum()},  "
          f"#eig < -1e-12 lambda_max = {(e < -1e-12 * e[-1]).sum()},  4th eigenvalue = {e[3]:.4e}")

    # ---- 6. translational null modes ---------------------------------------
    P("\n[6] translational null modes: ||Gamma X|| / (||Gamma||_2 ||X||)")
    for a, lab in enumerate("xyz"):
        X = np.zeros((N, 3))
        X[:, a] = 1.0
        X = X.ravel()
        P(f"    e_{lab}: " + "  ".join(f"{w}={np.linalg.norm(G[w] @ X) / (ev[w][-1] * np.linalg.norm(X)):.1e}"
                                         for w in ("full", "S", "L")))

    # ---- 7. quadratic form identity ----------------------------------------
    Xs = rng.normal(size=(500, N, 3))
    lhs = np.einsum("sa,ab,sb->s", Xs.reshape(500, -1), G["L"], Xs.reshape(500, -1))
    dX = Xs[:, pi] - Xs[:, pj]                                     # i<j pairs: 1/2 sum_{i!=j} = sum_{i<j}
    rhs = np.einsum("spi,pij,spj->s", dX, Kp["L"], dX)
    P(f"\n[7] X^T Gamma_L X vs 1/2 sum_(i!=j) dX^T K_L^per dX, 500 random X: "
      f"max rel diff = {np.max(np.abs(lhs / rhs - 1)):.2e}, min value = {rhs.min():.3e} (>0)")

    # ---- 8/10. Fourier-truncated K_L^per -----------------------------------
    kmax = max(kcs)
    nm = int(np.ceil(kmax * L / (2 * np.pi)))
    m = np.arange(-nm, nm + 1)
    kv = 2 * np.pi / L * np.stack(np.meshgrid(m, m, m, indexing="ij"), axis=-1).reshape(-1, 3)
    kn = np.linalg.norm(kv, axis=1)
    kv, kn = kv[kn <= kmax + 1e-12], kn[kn <= kmax + 1e-12]
    ku, kinv = np.unique(np.round(kn, 12), return_inverse=True)
    Au, Bu = K.AB_scaled(ku)
    dmp = np.exp(-ku ** 2 / (4 * xi ** 2))
    A, B = (Au * dmp)[kinv], (Bu * dmp)[kinv]
    lam_par = A + B * kn ** 2
    Gref = G["L"]
    P(f"\n[8] Fourier-truncated K_L^(per,kc) (includes k = 0); reference = direct-image Gamma_L")
    P("    kc      #k   lambda_min(Gamma)  #neg(raw) #neg(<-1e-12max)  lambda_4(Gamma)  row-sum err  "
      "||G_kc-G_L||/||G_L||  min block ratio  pairs w/ non-PSD block")
    rows = []
    for kc in kcs:
        sel = kn <= kc + 1e-12
        Kf = fourier_kernel(d, kv[sel], A[sel], B[sel], V)
        Gk = laplacian(Kf, pi, pj, N)
        e = spectrum(Gk)
        rowsum = np.abs(Gk.reshape(N, 3, N, 3).sum(axis=2)).max() / np.abs(Gk).max()
        err = np.abs(Gk - Gref).max() / np.abs(Gref).max()
        mb = min_eig_blocks(Kf)
        rows.append((kc, sel.sum(), e[0], (e < 0).sum(), (e < -1e-12 * e[-1]).sum(), rowsum, err, mb.min(), (mb < 0).sum()))
        P(f"    {kc:5.2f} {sel.sum():6d}   {e[0]: .4e}      {(e < 0).sum():4d}      {(e < -1e-12 * e[-1]).sum():4d}"
          f"          {e[3]: .4e}     {rowsum:.1e}      {err:.2e}           {mb.min(): .3e}      {(mb < 0).sum():4d}")
    P(f"    reference Gamma_L: lambda_4 = {ev['L'][3]:.4e}.  lambda_1..3 are the translational null modes (roundoff);")
    P("    lambda_4 is the smallest eigenvalue on the complement of translations.")
    P("    (row sums vanish identically for any graph Laplacian: Gamma_ii is built as sum_j K_ij)")

    # ---- 11. the two positivity notions ------------------------------------
    P("\n[11] two positivity notions")
    neg = lam_par < 0
    P(f"    lattice Khat_L(k) eigenvalues for 0 < |k| <= {kmax:g}: min(A+Bk^2) = {lam_par.min():.4e}, "
      f"negative at {neg.sum()} of {kv.shape[0]} lattice k (first at |k| = {kn[neg].min():.4f}; k0 = {K.k_zero():.4f})")
    P("    => the periodic convolution operator f -> K_L^per * f (eigenvalues Khat_L(k) on the lattice) is NOT PSD.")
    Mg = gram(Kp["L"], Kself["L"], pi, pj, N)
    em = spectrum(Mg)
    P(f"    Gram-type particle matrix M_ij = K_L^per(x_i - x_j) (incl. self image): smallest 5 = {fmt_list(em[:5])}")
    P(f"        #eig < -1e-12 lambda_max = {(em < -1e-12 * em[-1]).sum()}  "
      f"(its positivity is governed by Khat_L, since X^T M X = (1/V) sum_k S_k^* Khat_L(k) S_k)")
    Xn = np.linalg.eigh(0.5 * (Mg + Mg.T))[1][:, 0]
    P(f"        the same X gives X^T Gamma_L X = {Xn @ Gref @ Xn: .4e} >= 0")
    P(f"    Graph Laplacian Gamma_L: lambda_min/lambda_max = {ev['L'][0] / ev['L'][-1]: .2e}  -> PSD, "
      "because every block K_L^per(r_ij) is PSD (pointwise condition).")
    return rep, rows, ev, em


def sweep(kappa, xi, rcut, cases, kcs):
    K = EwaldLongitudinal(kappa, xi)
    prof = Profiles(K)
    lines = ["\n[sweep] other configurations: (a) direct-image Gamma_L, Gram matrix M; "
             "(b) min over kc of lambda_4(Gamma_L^kc)/lambda_max",
             "    N    L   seed   min eig Gamma_L/max  4th eig Gamma_L/max  min eig M/max  #neg M   "
             "(b) min_kc lambda_4/max (at kc)  #kc with genuinely neg. eig"]
    for N, L, seed in cases:
        x = np.random.default_rng(seed).uniform(0, L, size=(N, 3))
        pi, pj = np.triu_indices(N, 1)
        d = x[pi] - x[pj]
        d -= L * np.round(d / L)
        Kp = periodic_kernel(prof, d, "L", L, rcut)
        Ks = periodic_kernel(prof, np.zeros((1, 3)), "L", L, rcut)[0]
        e = spectrum(laplacian(Kp, pi, pj, N))
        em = spectrum(gram(Kp, Ks, pi, pj, N))
        nm = int(np.ceil(max(kcs) * L / (2 * np.pi)))
        m = np.arange(-nm, nm + 1)
        kv = 2 * np.pi / L * np.stack(np.meshgrid(m, m, m, indexing="ij"), axis=-1).reshape(-1, 3)
        kn = np.linalg.norm(kv, axis=1)
        ku, kinv = np.unique(np.round(kn, 12), return_inverse=True)
        Au, Bu = K.AB_scaled(ku)
        dmp = np.exp(-ku ** 2 / (4 * xi ** 2))
        A, B = (Au * dmp)[kinv], (Bu * dmp)[kinv]
        l4, nneg = [], 0
        for kc in kcs:
            sel = kn <= kc + 1e-12
            ek = spectrum(laplacian(fourier_kernel(d, kv[sel], A[sel], B[sel], L ** 3), pi, pj, N))
            l4.append(ek[3] / ek[-1])
            nneg += int((ek < -1e-12 * ek[-1]).any())
        i4 = int(np.argmin(l4))
        lines.append(f"   {N:3d} {L:4g} {seed:5d}      {e[0] / e[-1]: .2e}            {e[3] / e[-1]: .3e}       "
                     f"{em[0] / em[-1]: .3e}    {(em < -1e-12 * em[-1]).sum():3d}       "
                     f"{l4[i4]: .3e} ({kcs[i4]:g})                {nneg}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--xi", type=float, default=1.0)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--L", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rcut", type=float, default=45.0)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--single", action="store_true", help="only the configuration given by --N/--L/--seed")
    args = ap.parse_args()
    kcs = [0.0, 1.1, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    rep, *_ = run(args.kappa, args.xi, args.N, args.L, args.seed, args.rcut, kcs, None)
    if not args.single:
        # a sparser box (2 pi / L < k0) where the truncated Fourier Laplacian does go negative
        rep += ["", "=" * 100, "CASE B", "=" * 100]
        rep += run(args.kappa, args.xi, 40, 10.0, 13, args.rcut, kcs, None)[0]
    if not args.no_sweep:
        rep += sweep(args.kappa, args.xi, args.rcut - 5,
                     [(20, 3.0, 11), (20, 4.0, 1), (20, 6.0, 2), (30, 4.0, 3), (30, 8.0, 4), (40, 3.0, 12),
                      (40, 5.0, 5), (40, 8.0, 6), (40, 10.0, 13)], kcs)
    text = "\n".join(rep)
    print(text)
    outdir = Path(__file__).resolve().parent / "results"
    outdir.mkdir(exist_ok=True)
    (outdir / f"periodic_graph_psd_kappa{args.kappa:g}_xi{args.xi:g}.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
