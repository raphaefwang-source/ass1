"""
Cached matrix-free Method A action for fixed positions q:

    (Gamma_h X)_i = D_i X_i - H_i(X),
    D_i = sum_{j != i} K^h_ij                       (configuration-dependent, built once in setup)
    H_i(X) = sum_{j != i} K^h_ij X_j                 (recomputed for every X)

with K^h = K_S over the retained images (i, j, n) (neighbor_images.kdtree_images) plus the existing PPPM pair
matrix of test_parameter_selection.Mesh (spread / FFT / influence / IFFT / adjoint gather). The PPPM self terms
W_ii X_i appear in both D_i and H_i and cancel exactly, as in Mesh.graph_action.

Setup (configuration-dependent): image enumeration, pair weights w = theta_S/r^2, the real-space degree
D^S_i = sum K_S, mesh stencil indices/weights (Mesh.flat), mesh degree D^mesh_i (Mesh.degree, three translation
fields). Parameter-only (per xi, rc, kc, M, p and box, not per configuration): the Mesh object with its influence
function, built by make_mesh(). The existing GammaH.matvec (pair-difference form) is kept as the comparison path.
"""
import time

import numpy as np

import common  # noqa: F401  (sets threads and sys.path)
from neighbor_images import kdtree_images
from test_lanczos_fdt import GammaH
from test_equal_accuracy_xi import make_mesh, next_smooth
from test_parameter_selection import Mesh


def mesh_for(L, K, kc, M, p):
    """Existing Mesh at an explicitly chosen FFT size M (same construction as make_mesh)."""
    m = Mesh(L, K, kc, kc * (L / M) / np.pi * (1 + 1e-12), p)
    assert m.M == M and m.eta_actual < 1, (m.M, M, m.eta_actual)
    return m


def short_range_weights(K, d):
    r = np.linalg.norm(d, axis=1)
    return K.theta_S_scaled_closed(r) * np.exp(-(K.xi * r) ** 2) / r ** 2


def pair_difference_action(i, j, d, w, X, N):
    """Real-space part of the existing GammaH.matvec: sum over images of K_S (X_i - X_j) to i and -(...) to j."""
    f = (w * np.einsum("pa,pa->p", d, X[i] - X[j]))[:, None] * d
    return np.stack([np.bincount(i, f[:, c], N) - np.bincount(j, f[:, c], N) for c in range(3)], axis=1)


class CachedGammaH:
    """Method A operator for one configuration with cached geometry and degree tensors."""

    def __init__(self, x, L, K, rc, mesh, images=None):
        t = {}
        t0 = time.perf_counter()
        self.x, self.L, self.N, self.rc, self.K, self.mesh = np.mod(x, L), float(L), len(x), float(rc), K, mesh
        self.images = images if images is not None else kdtree_images(self.x, L, rc)
        self.i, self.j, self.d = self.images.i, self.images.j, self.images.d
        t["pairs"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.w = short_range_weights(K, self.d)
        self.wd = self.w[:, None] * self.d
        N = self.N
        DS = np.empty((N, 3, 3))
        for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
            kab = self.wd[:, a] * self.d[:, b]
            DS[:, a, b] = DS[:, b, a] = np.bincount(self.i, kab, N) + np.bincount(self.j, kab, N)
        self.DS = DS
        t["real_weights_degree"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        self.idx, self.wt = mesh.flat(self.x)
        t["mesh_stencil"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        self.Dm = mesh.degree(self.idx, self.wt)
        t["mesh_degree"] = time.perf_counter() - t3
        self.D = self.DS + self.Dm
        self.t_setup_parts = t
        self.t_setup = time.perf_counter() - t0
        self.n_actions = 0

    @property
    def n_images(self):
        return int(len(self.i))

    # -- action ----------------------------------------------------------------
    def real_H(self, X):
        """H^S_i(X) = sum K_S(d) X_j over images: row i gets K_S X_j, row j gets K_S X_i."""
        a = np.einsum("pa,pa->p", self.d, X[self.j])
        b = np.einsum("pa,pa->p", self.d, X[self.i])
        fi, fj = a[:, None] * self.wd, b[:, None] * self.wd
        N = self.N
        return np.stack([np.bincount(self.i, fi[:, c], N) + np.bincount(self.j, fj[:, c], N) for c in range(3)],
                        axis=1)

    def apply(self, X):
        X = np.asarray(X, float).reshape(self.N, 3)
        H = self.real_H(X) + self.mesh.apply(self.idx, self.wt, X)
        self.n_actions += 1
        return np.einsum("nab,nb->na", self.D, X) - H

    def matvec(self, v):
        return self.apply(v).ravel()

    def degree_term(self, X):
        return np.einsum("nab,nb->na", self.D, X)

    # -- staged mesh action (same operations as Mesh.apply, split for timing) ----
    def spread(self, X):
        m = self.mesh
        M3 = m.M ** 3
        return [np.bincount(self.idx.ravel(), (self.wt * X[:, c, None]).ravel(), M3).reshape((m.M,) * 3)
                for c in range(3)]

    def fft_influence(self, grids):
        m = self.mesh
        rho_hat = [np.fft.rfftn(g) for g in grids]
        return m._conv(rho_hat, range(3))

    def gather(self, phi):
        m = self.mesh
        return np.stack([(phi[d].ravel()[self.idx] * self.wt).sum(axis=1) for d in range(3)], axis=1) / m.h ** 3

    def apply_staged(self, X):
        X = np.asarray(X, float).reshape(self.N, 3)
        u = self.gather(self.fft_influence(self.spread(X)))
        return self.degree_term(X) - self.real_H(X) - u

    # -- comparison path: the existing GammaH (pair-difference form) on the same pairs and mesh ----
    def existing_gammah(self):
        op = GammaH.__new__(GammaH)
        op.x, op.L, op.N, op.rc, op.kc = self.x, self.L, self.N, self.rc, self.mesh.kc
        op.i, op.j, op.d, op.w = self.i, self.j, self.d, self.w
        op.mesh, op.idx, op.wt, op.D = self.mesh, self.idx, self.wt, self.Dm
        op.n_actions, op.t_actions = 0, 0.0
        return op

    # -- testing only: dense assembly by applying the action to the 3N standard basis vectors ----
    def dense_by_basis(self, block=None):
        n = 3 * self.N
        G = np.empty((n, n))
        E = np.zeros(n)
        for c in range(n):
            E[c] = 1.0
            G[:, c] = self.matvec(E)
            E[c] = 0.0
        return G


def build(x, L, xi, rc, kc, M, p, mesh=None):
    K = common.kernel(xi)
    if mesh is None:
        mesh = mesh_for(L, K, kc, M, p)
    return CachedGammaH(x, L, K, rc, mesh)


def baseline_mesh(L, K, kc, eta, p):
    """Mesh of the validated baseline set (make_mesh: M = next_smooth(ceil(L kc/(eta pi))))."""
    return make_mesh(L, K, kc, eta, p)


__all__ = ["CachedGammaH", "mesh_for", "build", "baseline_mesh", "next_smooth", "pair_difference_action",
           "short_range_weights"]
