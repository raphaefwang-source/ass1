"""Conservative model and scalar-Langevin preparation for the local pilot.

Only the conservative LJ force uses minimum images.  The production friction
is constructed separately by the existing complete-image Grid PSD operator.
"""
from __future__ import annotations

import hashlib
import json
from time import perf_counter
import numpy as np

N_PARTICLES = 256
RHO = .75
MASS = 1.
TEMPERATURE = 1.
KAPPA = .7
AMPLITUDE = .5 * 1.3 * np.exp(.7 * 1.3)
L_BOX = (N_PARTICLES / RHO) ** (1 / 3)

_PAIRS = {}


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def rng_sha(rng):
    return hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()


def switch(r):
    """Quintic switch and its radial derivative; both vanish at the cutoff."""
    r = np.asarray(r, dtype=float)
    z = np.clip((r - 2.) / .5, 0., 1.)
    S = 1. - 10. * z**3 + 15. * z**4 - 6. * z**5
    dS = (-30. * z**2 + 60. * z**3 - 30. * z**4) / .5
    dS = np.where((r > 2.) & (r < 2.5), dS, 0.)
    return S, dS


def lj_force(q, L=L_BOX):
    """Return force, total switched LJ potential, and minimum separation.

    Pair force is ``-[u_LJ'(r)*S(r)+u_LJ(r)*S'(r)] d/r``.
    The bincount scatter is antisymmetric pair by pair.  No friction kernel
    or historical LJ implementation is used here.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError('Require finite (N,3) positions')
    N = len(q)
    if N not in _PAIRS:
        _PAIRS[N] = np.triu_indices(N, 1)
    i, j = _PAIRS[N]
    d = q[i] - q[j]
    d -= float(L) * np.rint(d / float(L))
    r2 = np.einsum('pa,pa->p', d, d)
    if np.any(r2 <= 0):
        raise FloatingPointError('Coincident periodic particles in LJ force')
    minimum = float(np.sqrt(r2.min())) if len(r2) else float(L)
    keep = r2 < 2.5**2
    ii, jj, dd, rr2 = i[keep], j[keep], d[keep], r2[keep]
    r = np.sqrt(rr2)
    inverse6 = rr2**-3
    u = 4. * (inverse6**2 - inverse6)
    derivative = 24. * (inverse6 - 2. * inverse6**2) / r
    S, dS = switch(r)
    f = (-(derivative * S + u * dS) / r)[:, None] * dd
    force = np.stack([np.bincount(ii, f[:, a], N) - np.bincount(jj, f[:, a], N)
                      for a in range(3)], axis=1)
    potential = float(np.sum(u * S))
    if not np.isfinite(force).all() or not np.isfinite(potential):
        raise FloatingPointError('Nonfinite LJ force or potential')
    return force, potential, minimum


def fcc_configuration(N=N_PARTICLES, rho=RHO):
    """FCC lattice for N=4*ncell^3 (N=256 uses four cells per axis)."""
    ncell = int(round((int(N) / 4.) ** (1 / 3)))
    if int(N) != 4 * ncell**3:
        raise ValueError('FCC requires N=4*ncell^3')
    L = float((int(N) / float(rho)) ** (1 / 3))
    cells = np.indices((ncell,) * 3).reshape(3, -1).T
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    return ((cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) * L / ncell), L


def observables(q, p, L=L_BOX):
    _, potential, separation = lj_force(q, L)
    p = np.asarray(p)
    return dict(Tkin=float(np.sum((p - p.mean(axis=0))**2) / (3 * len(p) - 3)),
                U_per_particle=potential / len(p),
                momentum_x=float(p[:, 0].sum()), momentum_y=float(p[:, 1].sum()),
                momentum_z=float(p[:, 2].sum()), momentum_norm=float(np.linalg.norm(p.sum(axis=0))),
                min_pair_distance=separation)


def make_initial(seed, steps=6000, dt=.005, N=N_PARTICLES, rho=RHO,
                 scalar_gamma=1., record_stride=1):
    """Prepare one independent COM-constrained canonical-Langevin start.

    Returns ``(arrays, rows, metadata)``.  Production does not include this
    preparation time.  A 30-time-unit run is a pre-equilibration protocol,
    not a proof of exact canonical equilibration; its diagnostics are saved.
    ``rows`` retain every scalar pre-equilibration step by default.
    """
    if steps < 1 or dt <= 0 or scalar_gamma <= 0:
        raise ValueError('Require positive preparation length, dt, and gamma')
    started = perf_counter()
    rng = np.random.default_rng(int(seed))
    state_before = rng_sha(rng)
    q, L = fcc_configuration(N, rho)
    initial_q = q.copy()
    p = rng.standard_normal((N, 3))
    p -= p.mean(axis=0)
    initial_p = p.copy()
    force, potential, separation = lj_force(q, L)
    damping = np.exp(-scalar_gamma * dt)
    noise_scale = np.sqrt(-np.expm1(-2 * scalar_gamma * dt))
    rows = []
    noise_hash = hashlib.sha256()
    for step in range(1, int(steps) + 1):
        tick = perf_counter()
        ph = p + .5 * dt * force
        qm = q + .5 * dt * ph
        z = rng.standard_normal(p.shape)
        noise_hash.update(z.tobytes())
        z -= z.mean(axis=0)
        po = damping * ph + noise_scale * z
        # The COM mode is fixed, matching the production invariant subspace.
        po += ph.mean(axis=0) - po.mean(axis=0)
        q = qm + .5 * dt * po
        force, potential, separation = lj_force(np.mod(q, L), L)
        p = po + .5 * dt * force
        wall = perf_counter() - tick
        if not np.isfinite(q).all() or not np.isfinite(p).all():
            raise FloatingPointError(f'Nonfinite scalar preparation at step {step}')
        if step % record_stride == 0 or step == steps:
            rows.append(dict(step=step, time=step*dt, Tkin=float(np.sum((p-p.mean(axis=0))**2)/(3*N-3)),
                             U_per_particle=potential/N, min_pair_distance=separation,
                             momentum_norm=float(np.linalg.norm(p.sum(axis=0))),
                             q_displacement_rms=float(np.sqrt(np.mean((q-initial_q)**2))),
                             wall_s=wall))
    p -= p.mean(axis=0)
    arrays = dict(q_unwrapped=q.copy(), q=np.mod(q, L), p=p.copy(),
                  fcc_q=initial_q, initial_p=initial_p, L=np.asarray(L))
    meta = dict(seed=int(seed), bit_generator=type(rng.bit_generator).__name__,
                rng_state_before_sha256=state_before, rng_state_after_sha256=rng_sha(rng),
                raw_gaussian_stream_sha256=noise_hash.hexdigest(),
                initialization='FCC; independent Maxwell momentum; projected scalar Langevin BAOAB',
                N=int(N), rho=float(rho), L=L, dt=float(dt), steps=int(steps),
                duration=float(steps*dt), scalar_gamma=float(scalar_gamma),
                temperature=1., mass=1., initial_total_momentum_zero=True,
                total_wall_s=perf_counter()-started,
                q_sha256=array_sha(arrays['q']), p_sha256=array_sha(p),
                q_unwrapped_sha256=array_sha(q),
                canonical_claim='COM-constrained Langevin pre-equilibration; no exact equilibration proof',
                **{f'final_{k}':v for k,v in observables(arrays['q'], p, L).items()})
    return arrays, rows, meta
