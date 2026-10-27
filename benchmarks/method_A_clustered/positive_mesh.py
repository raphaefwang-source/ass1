"""B2: positive-edge mesh for the unchanged Candidate-1 Laplace split.

The real-space table, NOT its Fourier multipliers, is positive semidefinite.
Positive quadrature and a cubical physical image window preserve this property
even before convergence. See validation_exp1/positive_mesh_b2/proof.md.
"""
import time
import copy

import common  # noqa: F401 (path and thread defaults)
import numpy as np
from scipy.sparse import csr_matrix
from verify_ewald_longitudinal import rho
from cached_operator import CachedGammaH

PAIRS6 = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def table_tail_bound(L, kappa, R, amp=1.):
    """Uniform norm bound for discarded physical images (outside the cube).

    K_L <= K_full in Loewner order; the outside-cube set is outside radius R.
    Lattice point count <= 4*pi*(r+sqrt(3)*L/2)^3/(3*L^3).
    Integration by parts, dropping the negative boundary term, then 1/r<=1/R.
    """
    a = np.sqrt(3)*L/2
    i0 = np.exp(-kappa*R)/kappa
    i1 = i0*(R+1/kappa)
    i2 = i0*(R*R+2*R/kappa+2/kappa**2)
    return amp*4*np.pi/(3*L**3)*(kappa+1/R)*(i2+3*a*i1+(3*a*a+a**3/R)*i0)


def positive_bspline(u, p):
    """Centered cardinal spline; only nonnegative recurrence coefficients."""
    if p == 1:
        return ((u >= -.5) & (u < .5)).astype(float)
    return (np.maximum(p / 2 + u, 0) * positive_bspline(u + .5, p - 1)
            + np.maximum(p / 2 - u, 0) * positive_bspline(u - .5, p - 1)) / (p - 1)


def positive_rule(kappa, xi, order=8, log_step=np.log(2), lower_exponent=64.):
    """Positive GL in log(s), s in [min(xi^2/2,kappa^2/256),xi^2]."""
    lo = min(xi * xi / 2, kappa * kappa / (4 * lower_exponent))
    a, b = np.log(lo), np.log(xi * xi)
    edges = np.linspace(a, b, max(1, int(np.ceil((b - a) / log_step))) + 1)
    z, w = np.polynomial.legendre.leggauss(order)
    half = np.diff(edges)[:, None] / 2
    t = (edges[:-1, None] + half * (1 + z)).ravel()
    s = np.exp(t)
    weights = (half * w).ravel() * s
    density = rho(s, kappa)
    if not (np.all(weights > 0) and np.all(density >= 0) and np.all(np.isfinite(density))):
        raise FloatingPointError("Invalid positive Laplace rule")
    return s, weights * density, lo


def moment_factors(L, M, R, s):
    """1D image moments, using a variance expressed as a sum of squares.

    Canonical displacements and the physical cutoff |t+nL|<=R ensure periodic
    image handling. Reflection at 0 and Nyquist is a positive averaging step.
    f0,f1,f2 give A0, A0*mean, A0*(mean^2+variance).
    """
    j = np.arange(M)
    t = np.minimum(j, M - j) * (L / M)
    n = np.arange(-int(np.ceil(R / L)) - 1, int(np.ceil(R / L)) + 2)
    r = t[:, None] + n[None, :] * L
    keep = np.abs(r) <= R
    a = np.exp(-s[:, None, None] * r[None, :, :] ** 2) * keep[None, :, :]
    f0 = a.sum(axis=2)
    # A zero A0 is allowed (e.g. R<L/2). It contributes the zero tensor.
    mean = np.divide((a * r).sum(axis=2), f0, out=np.zeros_like(f0), where=f0 > 0)
    mean[:, 0] = 0
    if M % 2 == 0:
        mean[:, M // 2] = 0
    var = np.divide((a * (r[None, :, :] - mean[:, :, None]) ** 2).sum(axis=2), f0,
                    out=np.zeros_like(f0), where=f0 > 0)
    mean[:, j > M / 2] *= -1
    return np.array([f0, f0 * mean, f0 * (mean ** 2 + var)])


class PositiveMesh:
    """Periodic positive tensor table + identical nonnegative spread/gather.

    Legal: finite L,kappa,xi>0, amplitude>=0, integer M>=2, 1<=p<=M,
    R>0 and positive integer quadrature order. Coincident particles are not
    legal for the singular full physical kernel (the long mesh alone is finite).
    """
    def __init__(self, L, K, M, p, sT=36., quad_order=8, lower_exponent=64.):
        start = time.perf_counter()
        vals = (L, K.kap, K.xi, K.amp, sT, lower_exponent)
        if not all(np.isfinite(v) for v in vals) or min(L, K.kap, K.xi, sT, lower_exponent) <= 0 or K.amp < 0:
            raise ValueError("Require finite positive L,kappa,xi,sT and nonnegative amplitude")
        if int(M) != M or int(p) != p or not 1 <= p <= M or M < 2 or int(quad_order) != quad_order or quad_order < 1:
            raise ValueError("Require integer M>=2, 1<=p<=M, quadrature order>=1")
        theta = getattr(K, 'theta', 0.)
        # ThetaKernel stores its blend as a number; RadialEwald instead has
        # a profile-evaluation method named theta. Both support pure law A.
        if (not callable(theta) and theta != 0.) or getattr(K, 'law', 'A') != 'A':
            raise ValueError("B2 implements Candidate 1 law A only")
        self.L, self.K, self.M, self.p = float(L), K, int(M), int(p)
        self.h, self.V = self.L / self.M, self.L ** 3
        self.sT, self.R = float(sT), float(sT / K.kap)
        self.s, self.qw, self.s_lower = positive_rule(K.kap, K.xi, quad_order, lower_exponent=lower_exponent)
        self.qw *= K.amp
        if not np.all(np.isfinite(self.qw)) or np.any(self.qw < 0):
            raise FloatingPointError("Non-finite or negative positive-rule coefficient")
        self.factors = moment_factors(self.L, self.M, self.R, self.s)
        if not np.all(np.isfinite(self.factors)):
            raise FloatingPointError("Non-finite image moments")
        # Sum positive separable tensors. This is a direct real-space table
        # construction; no analytic Fourier synthesis, deconvolution or repair.
        f0, f1, f2 = self.factors
        table = np.empty((6, self.M, self.M, self.M))
        for c, (a, b) in enumerate(PAIRS6):
            fs = [f0, f0, f0]
            fs[a] = f2 if a == b else f1
            if a != b:
                fs[b] = f1
            table[c] = np.einsum('qi,qj,qk,q->ijk', *fs, self.qw, optimize=True)
        if not np.all(np.isfinite(table)):
            raise FloatingPointError("Non-finite positive tensor table")
        # Inversion is already enforced in 1D factors (including Nyquist).
        # Keeping all FFT modes includes the actual table zero mode.
        self.table = table
        self.Gr = np.ascontiguousarray(np.fft.rfftn(table, axes=(1, 2, 3)).real)
        if not np.all(np.isfinite(self.Gr)):
            raise FloatingPointError("Non-finite FFT table")
        self.zero_mode = self.Gr[:, 0, 0, 0].copy()
        self.t_setup = time.perf_counter() - start

    def with_order(self, p):
        """The table is independent of assignment order; share immutable arrays."""
        if int(p) != p or not 1 <= p <= self.M:
            raise ValueError("Require integer 1<=p<=M")
        other = copy.copy(self)
        other.p = int(p)
        return other

    def stencil(self, x):
        u = np.mod(np.asarray(x, float), self.L) / self.h
        if self.p == 1:
            # Resolve exact half-grid ties explicitly. Evaluating B1(g-u)
            # with a left-closed support would otherwise select an empty
            # stencil at the tie chosen by floor(u+1/2).
            g = np.floor(u + .5).astype(int)[:, :, None]
            return g, np.ones_like(g, dtype=float)
        g = np.floor(u - self.p / 2).astype(int)[:, :, None] + 1 + np.arange(self.p)
        w = positive_bspline(g - u[:, :, None], self.p)
        w /= w.sum(axis=2, keepdims=True)
        return g, w

    def flat(self, x):
        g, w = self.stencil(x)
        gi, M = g % self.M, self.M
        idx = (gi[:, 0, :, None, None] * M + gi[:, 1, None, :, None]) * M + gi[:, 2, None, None, :]
        wt = w[:, 0, :, None, None] * w[:, 1, None, :, None] * w[:, 2, None, None, :]
        return idx.reshape(len(x), -1), wt.reshape(len(x), -1)

    def _G(self, a, b):
        return self.Gr[PAIRS6.index((min(a, b), max(a, b)))]

    def _conv(self, rho_hat, comps):
        return [np.fft.irfftn(sum(self._G(a, b) * rho_hat[b] for b in comps),
                              s=(self.M,) * 3, axes=(0, 1, 2)) for a in range(3)]

    def apply(self, idx, wt, X):
        src = [np.fft.rfftn(np.bincount(idx.ravel(), (wt * X[:, c, None]).ravel(), self.M ** 3)
                            .reshape((self.M,) * 3)) for c in range(3)]
        phi = self._conv(src, range(3))
        return np.stack([(z.ravel()[idx] * wt).sum(axis=1) for z in phi], axis=1)

    def degree(self, idx, wt):
        src = np.fft.rfftn(np.bincount(idx.ravel(), wt.ravel(), self.M ** 3).reshape((self.M,) * 3))
        D = np.empty((len(idx), 3, 3))
        for c, (a, b) in enumerate(PAIRS6):
            z = np.fft.irfftn(self.Gr[c] * src, s=(self.M,) * 3, axes=(0, 1, 2))
            D[:, a, b] = D[:, b, a] = (z.ravel()[idx] * wt).sum(axis=1)
        return D

    def graph_action(self, idx, wt, X, D=None):
        if D is None:
            D = self.degree(idx, wt)
        return np.einsum('nab,nb->na', D, X) - self.apply(idx, wt, X)

    def pair_tensors(self, x):
        """Independent direct stencil contraction of the separable table.

        No 3D FFT or operator basis actions; only used for dense validation.
        """
        g, w = self.stencil(x)
        return _pair_tensors(g, w, self.factors, self.qw, self.M)


def _pair_tensors(g, w, fac, qw, M):
    N, _, p = g.shape
    rows = np.repeat(np.arange(N), p)
    S = [csr_matrix((w[:,d].ravel(), (rows, (g[:,d] % M).ravel())), shape=(N,M)) for d in range(3)]
    delta = (np.arange(M)[:,None] - np.arange(M)[None,:]) % M
    pi,pj = np.triu_indices(N,1)
    out = np.zeros((len(pi),6))
    for q in range(len(qw)):
        # Sparse multiplication evaluates the two compact stencils literally.
        v = np.empty((3,3,len(pi)))
        for d in range(3):
            for r in range(3):
                z = S[d] @ fac[r,q,delta]
                z = (S[d] @ z.T).T
                v[d,r] = z[pi,pj]
        a0,b0,c0 = v[0,0],v[1,0],v[2,0]
        out[:,0] += qw[q]*v[0,2]*b0*c0
        out[:,1] += qw[q]*a0*v[1,2]*c0
        out[:,2] += qw[q]*a0*b0*v[2,2]
        out[:,3] += qw[q]*v[0,1]*v[1,1]*c0
        out[:,4] += qw[q]*v[0,1]*b0*v[2,1]
        out[:,5] += qw[q]*a0*v[1,1]*v[2,1]
    return out


class PositiveGamma(CachedGammaH):
    """Original cached short-range algorithm, with B2 mesh normalization."""
    def __init__(self, x, L, K, rc, mesh, images=None):
        if not np.isfinite(rc) or rc <= 0:
            raise ValueError("Require finite rc>0")
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.all(np.isfinite(x)):
            raise ValueError("Require nonempty finite positions of shape (N,3)")
        # Singular coincident distinct particles are outside the physical model.
        if len(np.unique(np.mod(x, L), axis=0)) != len(x):
            raise ValueError("Coincident periodic particles are not legal for the physical kernel")
        if mesh.L != L or mesh.K is not K:
            raise ValueError("Mesh and operator must share box and kernel")
        self.iso = False
        super().__init__(x, L, K, rc, mesh, images)
        if not np.all(np.isfinite(self.w)) or np.any(self.w < 0) or not np.all(np.isfinite(self.D)):
            raise FloatingPointError("Short weights and complete degree must be finite and weights nonnegative")

    def gather(self, phi):
        return np.stack([(z.ravel()[self.idx] * self.wt).sum(axis=1) for z in phi], axis=1)


__all__ = ['PositiveMesh', 'PositiveGamma', 'positive_bspline', 'positive_rule', 'moment_factors']
