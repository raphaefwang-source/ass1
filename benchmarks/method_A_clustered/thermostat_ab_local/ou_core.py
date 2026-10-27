"""Matrix-free frozen OU kernels, with explicit removal of the three COM modes.

The supplied action is the symmetric friction matrix G (not G/m).  For a
step h the two functions are exp(-h G/m) and
sqrt(m theta [I-exp(-2h G/m)]).  Lanczos is vector dependent at finite rank;
its noise action must therefore be tested statistically, not interpreted as
a fixed matrix assembled by applying it to coordinate columns.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.linalg import eigh_tridiagonal


def _vector(v):
    a = np.asarray(v, dtype=float)
    if a.ndim not in (1, 2) or (a.ndim == 2 and a.shape[1] != 3):
        raise ValueError("expected flat 3N vector or (N,3) array")
    if a.size == 0 or a.size % 3 or not np.isfinite(a).all():
        raise ValueError("expected a nonempty finite 3N vector")
    return a, a.reshape(-1, 3)


def project(v):
    """Apply Pi = (I - 11'/N) tensor I3 and preserve the input shape."""
    a, q = _vector(v)
    return (q - q.mean(axis=0)).reshape(a.shape)


def _parameters(dt, mass, theta):
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative")
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("mass must be finite and positive")
    if not np.isfinite(theta) or theta < 0:
        raise ValueError("theta must be finite and nonnegative")


class PSDViolation(ValueError):
    """Significant negative Ritz/edge eigenvalue; no OU step is produced."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _psd_spectrum(values, diagnostics):
    """Use one scale-aware roundoff rule for damping, noise, and edges."""
    values = np.asarray(values, dtype=float)
    scale = max(1., float(np.max(np.abs(values), initial=0.)))
    tolerance = 256 * np.finfo(float).eps * scale
    minimum = float(values.min(initial=np.inf))
    diagnostics.update(ritz_min_raw=minimum, negative_tolerance=tolerance,
                       negative_ritz_count=int(np.count_nonzero(values < 0)),
                       clipped_negative_count=int(np.count_nonzero((values < 0) & (values >= -tolerance))))
    if minimum < -tolerance:
        raise PSDViolation(f"friction is not PSD: minimum eigenvalue {minimum:.6g} < {-tolerance:.6g}", diagnostics)
    return np.maximum(values, 0.)


def lanczos_action(action, v, rank, dt, kind, mass=1., theta=1.):
    """Return one matrix-function action and measured Lanczos diagnostics.

    ``action`` receives and returns a flat 3N vector.  Two full classical
    Gram-Schmidt passes reorthogonalize each residual against every current
    basis vector.  No diagonal shift or regularization of null modes is used.
    Significant negative Ritz values reject BOTH matrix functions.  The rank
    is capped at 3N-3; happy breakdown can terminate it earlier.
    """
    start = perf_counter()
    _parameters(dt, mass, theta)
    aliases = {'damp': 'damping', 'exp': 'damping', 'sqrt': 'noise'}
    kind = aliases.get(kind, kind)
    if kind not in ('damping', 'noise'):
        raise ValueError("kind must be 'damping'/'damp' or 'noise'")
    if isinstance(rank, bool) or int(rank) != rank or rank < 1:
        raise ValueError("rank must be a positive integer")
    original, _ = _vector(v)
    vector = project(original).ravel()
    norm = float(np.linalg.norm(vector))
    dimension = max(0, vector.size - 3)
    limit = min(int(rank), dimension)
    diag = dict(kind=kind, rank_requested=int(rank), rank_used=0,
                projected_dimension=dimension, input_norm=norm,
                matvec_count=0, reorth_passes=0, reorth_inner_products=0,
                t_matvec_s=0., t_reorth_s=0., t_eigh_s=0.,
                ritz_min_raw=None, ritz_max_raw=None,
                negative_ritz_count=0, clipped_negative_count=0,
                happy_breakdown=False, beta_tail=0., orthogonality_error=0.)
    if norm == 0. or dimension == 0 or dt == 0 or (kind == 'noise' and theta == 0):
        output = vector if kind == 'damping' else np.zeros_like(vector)
        diag['happy_breakdown'] = norm == 0. or dimension == 0
        diag['t_total_s'] = perf_counter() - start
        return output.reshape(original.shape), diag

    Q = np.empty((limit, vector.size))
    Q[0] = vector / norm
    alphas, betas = [], []
    beta_previous = 0.
    for j in range(limit):
        t = perf_counter()
        w = np.asarray(action(Q[j].copy()), dtype=float).reshape(-1)
        diag['t_matvec_s'] += perf_counter() - t
        diag['matvec_count'] += 1
        if w.shape != vector.shape or not np.isfinite(w).all():
            raise ValueError("action returned an invalid vector")
        w = project(w).ravel()
        action_norm = float(np.linalg.norm(w))
        if j:
            w -= beta_previous * Q[j-1]
        alpha = float(Q[j] @ w)
        alphas.append(alpha)
        w -= alpha * Q[j]
        t = perf_counter()
        for _ in range(2):
            coefficients = Q[:j+1] @ w
            w -= coefficients @ Q[:j+1]
        w = project(w).ravel()
        diag['t_reorth_s'] += perf_counter() - t
        diag['reorth_passes'] += 2
        diag['reorth_inner_products'] += 2 * (j + 1)
        beta = float(np.linalg.norm(w))
        diag['beta_tail'] = beta
        # Relative to the observed action, not an arbitrary damping shift.
        breakdown = 64 * np.finfo(float).eps * max(action_norm, abs(alpha), beta_previous, np.finfo(float).tiny)
        if beta <= breakdown or j+1 == dimension:
            diag['happy_breakdown'] = True
            break
        if j+1 == limit:
            break
        betas.append(beta)
        Q[j+1] = w / beta
        beta_previous = beta

    r = len(alphas)
    diag['rank_used'] = r
    # Full reorthogonalization keeps the symmetric Lanczos projection tridiagonal
    # up to floating point roundoff; the diagonal/offdiagonal are stored directly.
    t = perf_counter()
    eigenvalues, U = eigh_tridiagonal(np.asarray(alphas), np.asarray(betas), check_finite=False)
    diag['t_eigh_s'] = perf_counter() - t
    diag['ritz_max_raw'] = float(eigenvalues[-1])
    eigenvalues = _psd_spectrum(eigenvalues, diag)
    if kind == 'damping':
        values = np.exp(-(dt / mass) * eigenvalues)
    else:
        values = np.sqrt(mass * theta * -np.expm1(-(2 * dt / mass) * eigenvalues))
    coefficients = U @ (values * U[0])
    output = project(norm * (coefficients @ Q[:r])).ravel()
    # Measured directly rather than inferred from the number of reorth passes.
    diag['orthogonality_error'] = float(np.linalg.norm(Q[:r] @ Q[:r].T - np.eye(r)))
    diag['t_total_s'] = perf_counter() - start
    return output.reshape(original.shape), diag


def _restore_com(relative, original):
    a, q = _vector(original)
    result = project(relative).reshape(-1, 3) + q.mean(axis=0)
    # One correction limits drift from accumulating roundoff over many substeps.
    result += q.mean(axis=0) - result.mean(axis=0)
    return result.reshape(a.shape)


def _ou(action, p, z, dt, r_d, r_n, mass, theta, prefix):
    start = perf_counter()
    p, _ = _vector(p)
    z, _ = _vector(z)
    if p.size != z.size:
        raise ValueError("p and z must have the same size")
    damp, dd = lanczos_action(action, p, r_d, dt, 'damping', mass, theta)
    noise, nd = lanczos_action(action, z.reshape(p.shape), r_n, dt, 'noise', mass, theta)
    output = _restore_com(damp + noise, p)
    diag = {'damping': dd, 'noise': nd,
            f't_{prefix}_lanczos_damp': dd['t_total_s'],
            f't_{prefix}_lanczos_noise': nd['t_total_s'],
            f't_{prefix}_matvec': dd['t_matvec_s'] + nd['t_matvec_s'],
            't_reorth': dd['t_reorth_s'] + nd['t_reorth_s'],
            't_ritz_eigh': dd['t_eigh_s'] + nd['t_eigh_s'],
            'matvec_count': dd['matvec_count'] + nd['matvec_count'],
            'damp_matvec_count': dd['matvec_count'], 'noise_matvec_count': nd['matvec_count'],
            'reorth_inner_products': dd['reorth_inner_products'] + nd['reorth_inner_products'],
            'damp_rank_used': dd['rank_used'], 'noise_rank_used': nd['rank_used'],
            'negative_ritz_count': dd['negative_ritz_count'] + nd['negative_ritz_count'],
            'clipped_negative_count': dd['clipped_negative_count'] + nd['clipped_negative_count'],
            'step_wall_s': perf_counter() - start}
    return output, diag


def ou_b(action, p, z, dt, r_d, r_n, mass=1., theta=1.):
    """Method B: full friction OU action; ``z`` is supplied standard normal."""
    return _ou(action, p, z, dt, r_d, r_n, mass, theta, 'full')


class EdgeOU:
    """Exact pair OU sweeps for full 3x3 PSD edge blocks.

    Edges can be an iterable of (i,j,K), or three separate arrays.  Repeated
    edges are permitted and retain their supplied order.  Eigensystems are
    cached once; time-dependent scalar factors are cached by (tau,m,theta).
    All noise arrays are indexed by ORIGINAL edge index, including reverse.
    """
    def __init__(self, edges, j=None, K=None, n_particles=None):
        start = perf_counter()
        if j is None and K is None:
            records = list(edges)
            i = [e[0] for e in records]
            j = [e[1] for e in records]
            K = [e[2] for e in records]
        elif j is None or K is None:
            raise ValueError("supply all of i,j,K")
        else:
            i = edges
        self.i = np.asarray(i, dtype=np.int64).reshape(-1)
        self.j = np.asarray(j, dtype=np.int64).reshape(-1)
        self.K = np.asarray(K, dtype=float).reshape(-1, 3, 3)
        if len(self.i) != len(self.j) or len(self.i) != len(self.K):
            raise ValueError("edge arrays have inconsistent sizes")
        if np.any(self.i < 0) or np.any(self.j < 0) or np.any(self.i == self.j):
            raise ValueError("edge endpoints must be distinct nonnegative indices")
        if not np.isfinite(self.K).all():
            raise ValueError("edge blocks must be finite")
        inferred = int(max(self.i.max(initial=-1), self.j.max(initial=-1)) + 1)
        self.n_particles = int(n_particles) if n_particles is not None else inferred
        if self.n_particles < inferred:
            raise ValueError("n_particles is smaller than an edge endpoint")
        asym = np.max(np.abs(self.K - self.K.transpose(0, 2, 1)), initial=0.)
        scale = max(1., float(np.max(np.abs(self.K), initial=0.)))
        if asym > 256 * np.finfo(float).eps * scale:
            raise ValueError("edge blocks must be symmetric")
        self.K = .5 * (self.K + self.K.transpose(0, 2, 1))
        raw, self.U = np.linalg.eigh(self.K)
        self.spectrum_diagnostics = {}
        self.eigenvalues = _psd_spectrum(raw.ravel(), self.spectrum_diagnostics).reshape(-1, 3) if len(raw) else raw
        # Keep the exposed K consistent with the same roundoff clipping rule.
        if self.spectrum_diagnostics.get('clipped_negative_count', 0):
            self.K = np.einsum('eik,ek,ejk->eij', self.U, self.eigenvalues, self.U)
        self._factors = {}
        self.t_setup = perf_counter() - start

    @classmethod
    def from_arrays(cls, i, j, K, n_particles=None):
        return cls(i, j, K, n_particles=n_particles)

    def _coefficients(self, tau, mass, theta):
        _parameters(tau, mass, theta)
        key = (float(tau), float(mass), float(theta))
        if key not in self._factors:
            damping = np.exp(-(2*tau/mass) * self.eigenvalues)
            noise = np.sqrt(2*mass*theta * -np.expm1(-(4*tau/mass) * self.eigenvalues))
            self._factors[key] = (damping, noise)
        return self._factors[key]

    def sweep(self, p, eta, tau, reverse=False, mass=1., theta=1.):
        start = perf_counter()
        original, q = _vector(p)
        if len(q) < self.n_particles:
            raise ValueError("p has too few particles for these edges")
        eta = np.asarray(eta, dtype=float)
        if eta.shape != (len(self.i), 3) or not np.isfinite(eta).all():
            raise ValueError("eta must be finite with shape (number of edges,3)")
        damping, noise = self._coefficients(tau, mass, theta)
        q = q.copy()
        order = range(len(self.i)-1, -1, -1) if reverse else range(len(self.i))
        for e in order:
            i, j, U = self.i[e], self.j[e], self.U[e]
            relative = q[i] - q[j]
            updated = U @ (damping[e] * (U.T @ relative) + noise[e] * (U.T @ eta[e]))
            increment = .5 * (updated - relative)
            q[i] += increment
            q[j] -= increment
        return q.reshape(original.shape), dict(pair_updates=len(self.i), t_pairOU=perf_counter()-start)

    def mean_sweep(self, p, tau, reverse=False, mass=1.):
        return self.sweep(p, np.zeros((len(self.i), 3)), tau, reverse, mass, 0.)[0]


def ou_a(long_action, p, z_long, eta_forward, eta_reverse, edges, dt, r_d, r_n, mass=1., theta=1.):
    """Method A: forward edge half sweep, long OU, reverse edge half sweep.

    Caller supplies three independent Gaussian arrays.  Independence is a
    probabilistic caller obligation; kernels never generate or reuse noise.
    """
    start = perf_counter()
    _parameters(dt, mass, theta)
    if not isinstance(edges, EdgeOU):
        edges = EdgeOU(edges)
    first, df = edges.sweep(p, eta_forward, dt/2, False, mass, theta)
    middle, diag = _ou(long_action, first, z_long, dt, r_d, r_n, mass, theta, 'long')
    result, dr = edges.sweep(middle, eta_reverse, dt/2, True, mass, theta)
    result = _restore_com(result, p)
    diag.update(t_pairOU_forward=df['t_pairOU'], t_pairOU_reverse=dr['t_pairOU'],
                pair_updates=df['pair_updates']+dr['pair_updates'], step_wall_s=perf_counter()-start)
    return result, diag
