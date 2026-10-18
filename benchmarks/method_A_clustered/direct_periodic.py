"""
Direct (unsplit) full-periodic action of the physical kernel A (manuscript Eqs. kernelA, Kper, Gamma):

    (Gamma X)_i = sum_{j != i} sum_{n in Z^3} K(q_i - q_j + n L) (X_i - X_j),   K(r) = c e^{-kappa r}/r rhat rhat^T,

with the kernel evaluated by the reference's own function (prl_toy_models.radial_friction, law A, gamma 0.5,
kappa 0.7, r_ref 1.3). No Ewald split, no minimum image: every image (i < j, n) with |q_i - q_j + n L| <= R_d is
summed, self images (i = j) cancel in X_i - X_i and are not generated. R_d is chosen by a convergence test.

Two direct pairwise algorithms (no dense BLAS):
  DirectImages     image-resolved: setup = periodic KD-tree candidates + exact lattice-shift enumeration
                   (neighbor_images.kdtree_images, validated against brute force) and the weights g(r)/r^2;
                   action = loop over all retained images (chunked pair-difference form, as Eq. (7))
  DirectPairs      pair-summed: setup = the periodic pair tensor K^per_R(q_i - q_j) = sum_{|d| <= R_d} K(d) of every
                   pair i < j, streamed over lattice-shift chunks; action = loop over the N(N-1)/2 pairs (Eq. Gamma)
Preassembled dense physical Gamma (a separate comparator): DirectPairs tensors scattered into the 3N x 3N
graph Laplacian with the reference's assemble_friction; action = BLAS GEMV.

Convergence test (annulus_tensors): pair tensors accumulated per annulus (R_{k-1}, R_k] up to R_max, so Gamma_D(R)
is available for every R of the grid; certified tail: 2 max_i lambda_max(sum of discarded K) + an analytic bound of
the images beyond R_max (count N 4 pi/3 (r + sqrt(3) L/2)^3 / L^3, g decreasing).
"""
import time

import numpy as np

import common as C
from neighbor_images import kdtree_images
import prl_toy_models as v2

KW = dict(kernel=C.LAW, gamma=C.GAMMA, kappa=C.KAPPA, r_ref=C.R_REF)
SYM_A, SYM_B = np.array([0, 1, 2, 0, 0, 1]), np.array([0, 1, 2, 1, 2, 2])
SYM_FULL = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])
CHUNK = 1 << 20


def g_over_r2(r2):
    """g(r)/r^2 of kernel A (prl_toy_models.radial_friction), so that K(d) = (g/r^2) d d^T."""
    return v2.radial_friction(np.sqrt(r2), **KW) / r2


def min_image_pairs(x, L):
    N = len(x)
    pi, pj = np.triu_indices(N, 1)
    d0 = x[pi] - x[pj]
    d0 -= L * np.round(d0 / L)
    return pi.astype(np.intp), pj.astype(np.intp), d0


def shifts_for(L, R):
    """All n L with |n L| <= R + sqrt(3) L/2: complete for minimum-image d0 (|d0_a| <= L/2)."""
    return v2.lattice_shifts(L, R + np.sqrt(3) * L / 2)


def tail_bound_per_particle(N, L, R, dr=0.05):
    """Upper bound of sum_{images of one particle, |d| > R} g(|d|). With n(r) the number of images within r and
    cnt(r) = N 4 pi/3 (r + sqrt(3) L/2)^3 / L^3 >= n(r), summation by parts gives
    sum = -g(R) n(R) + int_R^inf n(r) (-g'(r)) dr <= sum_s cnt(r_{s+1}) [g(r_s) - g(r_{s+1})]  (g decreasing)."""
    r = R + dr * np.arange(400001)
    g = v2.radial_friction(r, **KW)
    cnt = N * 4 / 3 * np.pi * (r[1:] + np.sqrt(3) * L / 2) ** 3 / L ** 3
    terms = cnt * (g[:-1] - g[1:])
    assert terms[-1] < 1e-30 * terms.sum()
    return float(terms.sum() + cnt[-1] * g[-1])


# ----------------------------------------------------------------------------- convergence test
def annulus_tensors(x, L, edges, budget=1 << 22):
    """acc[k, p, :] = sum of K(d) (6 components) over the images of pair p with edges[k-1] < |d| <= edges[k]
    (edges[-1] = R_max; images beyond are dropped). Returns pi, pj, acc and the image count per annulus."""
    x = np.mod(x, L)
    pi, pj, d0 = min_image_pairs(x, L)
    P = len(pi)
    edges = np.asarray(edges, float)
    nK = len(edges)
    e2 = edges ** 2
    S = shifts_for(L, edges[-1])
    acc = np.zeros((nK * P, 6))
    count = np.zeros(nK, np.int64)
    step = max(1, budget // P)
    pidx = np.arange(P)
    for s0 in range(0, len(S), step):
        d = d0[None, :, :] + S[s0:s0 + step, None, :]
        r2 = np.einsum("spa,spa->sp", d, d)
        k = np.searchsorted(e2, r2, side="left")          # first edge >= r: annulus index; nK = beyond R_max
        keep = k < nK
        kk = k[keep]
        dk = d[keep]
        w = g_over_r2(r2[keep])
        flat = kk * P + np.broadcast_to(pidx[None, :], k.shape)[keep]
        count += np.bincount(kk, minlength=nK)
        for c in range(6):
            acc[:, c] += np.bincount(flat, w * dk[:, SYM_A[c]] * dk[:, SYM_B[c]], nK * P)
    return pi, pj, acc.reshape(nK, P, 6), count


def tensors_to_dense(pi, pj, T6, N):
    """3N x 3N graph Laplacian from symmetric pair tensors (reference routine prl_toy_models.assemble_friction)."""
    return v2.assemble_friction(pi, pj, T6[:, SYM_FULL], N)


def degree_lambda_max(pi, pj, T6, N):
    D = np.stack([np.bincount(pi, T6[:, c], N) + np.bincount(pj, T6[:, c], N) for c in range(6)], 1)
    return float(np.linalg.eigvalsh(D[:, SYM_FULL]).max())


# ----------------------------------------------------------------------------- direct operators
class DirectImages:
    """Image-resolved direct action: all (i < j, n) with |d| <= R stored once, action loops over images."""

    def __init__(self, x, L, R):
        t0 = time.perf_counter()
        self.x, self.L, self.N, self.R = np.mod(x, L), float(L), len(x), float(R)
        il = kdtree_images(self.x, L, R)
        self.i, self.j, self.d = il.i, il.j, il.d
        t1 = time.perf_counter()
        self.w = g_over_r2(np.einsum("pa,pa->p", self.d, self.d))
        t2 = time.perf_counter()
        self.t_setup_parts = dict(images=t1 - t0, weights=t2 - t1)
        self.t_setup = t2 - t0

    @property
    def n_images(self):
        return int(len(self.i))

    def apply(self, X):
        X = np.asarray(X, float).reshape(self.N, 3)
        out = np.zeros((self.N, 3))
        for c0 in range(0, len(self.i), CHUNK):
            i, j, d = self.i[c0:c0 + CHUNK], self.j[c0:c0 + CHUNK], self.d[c0:c0 + CHUNK]
            f = (self.w[c0:c0 + CHUNK] * np.einsum("pa,pa->p", d, X[i] - X[j]))[:, None] * d
            for c in range(3):
                out[:, c] += np.bincount(i, f[:, c], self.N) - np.bincount(j, f[:, c], self.N)
        return out

    def pair_tensors(self):
        """Reduce to K^per_R per pair i < j (cross-check of DirectPairs)."""
        N = self.N
        key = self.i.astype(np.int64) * N + self.j
        T = np.stack([np.bincount(key, self.w * self.d[:, SYM_A[c]] * self.d[:, SYM_B[c]], N * N)
                      for c in range(6)], 1)
        pi, pj = np.triu_indices(N, 1)
        return T[pi * N + pj]


class DirectPairs:
    """Pair-summed direct action: K^per_R(q_i - q_j) tabulated for every pair i < j in setup (streamed over lattice
    shifts, images not stored), action loops over the N(N-1)/2 pairs."""

    def __init__(self, x, L, R, budget=1 << 22):
        t0 = time.perf_counter()
        self.x, self.L, self.N, self.R = np.mod(x, L), float(L), len(x), float(R)
        self.pi, self.pj, d0 = min_image_pairs(self.x, L)
        P = len(self.pi)
        S = shifts_for(L, R)
        T = np.zeros((P, 6))
        R2 = self.R * self.R
        n_img = 0
        step = max(1, budget // P)
        pidx = np.arange(P)
        for s0 in range(0, len(S), step):
            d = d0[None, :, :] + S[s0:s0 + step, None, :]
            r2 = np.einsum("spa,spa->sp", d, d)
            keep = r2 <= R2
            dk = d[keep]
            w = g_over_r2(r2[keep])
            p = np.broadcast_to(pidx[None, :], r2.shape)[keep]
            n_img += len(p)
            for c in range(6):
                T[:, c] += np.bincount(p, w * dk[:, SYM_A[c]] * dk[:, SYM_B[c]], P)
        self.T = T
        self.n_images = int(n_img)
        self.n_shifts = int(len(S))
        self.t_setup = time.perf_counter() - t0

    def apply(self, X):
        X = np.asarray(X, float).reshape(self.N, 3)
        dX = X[self.pi] - X[self.pj]
        T = self.T
        f = np.stack([T[:, 0] * dX[:, 0] + T[:, 3] * dX[:, 1] + T[:, 4] * dX[:, 2],
                      T[:, 3] * dX[:, 0] + T[:, 1] * dX[:, 1] + T[:, 5] * dX[:, 2],
                      T[:, 4] * dX[:, 0] + T[:, 5] * dX[:, 1] + T[:, 2] * dX[:, 2]], 1)
        N = self.N
        return np.stack([np.bincount(self.pi, f[:, c], N) - np.bincount(self.pj, f[:, c], N) for c in range(3)], 1)

    def dense(self):
        return tensors_to_dense(self.pi, self.pj, self.T, self.N)
