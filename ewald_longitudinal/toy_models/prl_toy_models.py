#!/usr/bin/env python3
"""CPU reference simulations and PRL-style trajectory statistics.

Toy application, NOT a reproduction of Lyu & Lei PRL 131, 177301 (2023).
Equal masses, 3D orthorhombic periodic box, minimum-image pair interactions.
Use a small N: the thermostat uses a dense eigendecomposition every step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def minimum_image(delta, box):
    box = np.asarray(box, dtype=float)
    return delta - box * np.rint(delta / box)


def pair_potential(r, kind, epsilon=1.0, sigma=1.0, r_on=2.0, r_cut=2.5):
    """Return u(r), du/dr. Switching radii are in units of sigma.

    double_well: repulsive r^-12 core plus two attractive Gaussian wells.
    lj: 12-6 Lennard-Jones. Both use the same C2 quintic cutoff switch.
    """
    r = np.asarray(r, dtype=float)
    if np.any(r <= 0) or not 0 < r_on < r_cut:
        raise ValueError('Require positive separations and 0 < r_on < r_cut.')
    s = r / sigma
    if kind == 'lj':
        u = 4 * epsilon * (s**-12 - s**-6)
        du = 24 * epsilon / sigma * (s**-7 - 2*s**-13)
    elif kind == 'double_well':
        centers, width = (1.12, 1.60), 0.12
        gaussians = [np.exp(-0.5*((s-c)/width)**2) for c in centers]
        u = epsilon * (0.1*s**-12 - gaussians[0] - gaussians[1])
        du = epsilon / sigma * (-1.2*s**-13 + sum(
            g*(s-c)/width**2 for g, c in zip(gaussians, centers)))
    else:
        raise ValueError(f'Unknown potential: {kind}')
    x = np.clip((s-r_on)/(r_cut-r_on), 0, 1)
    switch = 1 - 10*x**3 + 15*x**4 - 6*x**5
    derivative = (-30*x**2 + 60*x**3 - 30*x**4) / (sigma*(r_cut-r_on))
    return switch*u, switch*du + derivative*u


def pair_geometry(q, box):
    i, j = np.triu_indices(len(q), 1)
    dr = minimum_image(q[i]-q[j], box)
    r = np.linalg.norm(dr, axis=1)
    if np.any(r < 1e-12):
        raise ValueError('Coincident particles: the longitudinal projector is undefined.')
    return i, j, dr, r


def conservative_force(q, box, potential, **kwargs):
    i, j, dr, r = pair_geometry(q, box)
    u, du = pair_potential(r, potential, **kwargs)
    fij = -du[:, None]*dr/r[:, None]
    force = np.zeros_like(q, dtype=float)
    np.add.at(force, i, fij)
    np.add.at(force, j, -fij)
    return force, float(u.sum()), float(r.min())


def radial_friction(r, kernel='A', gamma=0.5, kappa=0.7, r_ref=1.3):
    """Normalize BOTH laws by g(r_ref)=gamma; this is not a fitted calibration.

    A: g=gamma*(r_ref/r)*exp[-kappa*(r-r_ref)].
    B: g=gamma*(r/r_ref)*exp[-kappa*(r-r_ref)].
    A and B have different total friction even under this normalization.
    """
    if gamma < 0 or kappa <= 0 or r_ref <= 0 or np.any(np.asarray(r) <= 0):
        raise ValueError('Require gamma>=0, kappa>0, r_ref>0 and r>0.')
    if kernel not in ('A', 'B'):
        raise ValueError('kernel must be A or B')
    power = -1 if kernel == 'A' else 1
    return gamma*(np.asarray(r)/r_ref)**power*np.exp(-kappa*(np.asarray(r)-r_ref))


def friction_matrix(q, box, **kwargs):
    """Gamma_ii=sum K_ij, Gamma_ij=-K_ij, K_ij=g(r) rhat rhat^T.

    All minimum-image pairs are retained; no friction cutoff, no Ewald sum.
    """
    i, j, dr, r = pair_geometry(q, box)
    unit = dr/r[:, None]
    K = radial_friction(r, **kwargs)[:, None, None]*np.einsum('pi,pj->pij', unit, unit)
    return assemble_friction(i, j, K, len(q))


def assemble_friction(i, j, K, n):
    """Graph-Laplacian Gamma from symmetric 3x3 pair tensors K[p] for pairs (i[p], j[p])."""
    blocks = np.zeros((n, n, 3, 3))
    degree = np.zeros((n, 3, 3))
    np.add.at(degree, i, K)
    np.add.at(degree, j, K)
    blocks[i, j] = blocks[j, i] = -K
    blocks[np.arange(n), np.arange(n)] = degree
    return blocks.transpose(0, 2, 1, 3).reshape(3*n, 3*n)


def friction_tail_integral(r_cut, kernel='A', gamma=0.5, kappa=0.7, r_ref=1.3):
    """4*pi*int_{r_cut}^inf r^2 g(r) dr (closed form). r_cut=0 gives ghat(k=0).

    The k=0 Fourier coefficient of the periodic pair tensor is ghat(0)/(3V) * I.
    """
    radial_friction(np.array([r_ref]), kernel, gamma, kappa, r_ref)
    x = kappa*np.asarray(r_cut, dtype=float)
    if kernel == 'A':
        return 4*np.pi*gamma*r_ref*np.exp(kappa*r_ref-x)*(1+x)/kappa**2
    return 4*np.pi*gamma/r_ref*np.exp(kappa*r_ref-x)*(x**3+3*x**2+6*x+6)/kappa**4


def lattice_shifts(box, reach):
    """All lattice vectors n*box (n integer) with |n*box| <= reach, including n=0."""
    box = np.broadcast_to(np.asarray(box, dtype=float), (3,))
    axes = [np.arange(-k, k+1) for k in np.ceil(reach/box).astype(int)]
    shifts = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)*box
    return shifts[np.einsum('sc,sc->s', shifts, shifts) <= reach**2]


_SYM_A, _SYM_B = np.array([0, 1, 2, 0, 0, 1]), np.array([0, 1, 2, 1, 2, 2])
_SYM_FULL = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])


def image_sum_tensors(dr, shifts, chunk=1 << 21, **kwargs):
    """sum_s g(|dr+s|) (dr+s)(dr+s)^T/|dr+s|^2 for each row of dr; shape (P,3,3)."""
    dr = np.atleast_2d(np.asarray(dr, dtype=float))
    out = np.zeros((len(dr), 6))
    step = max(1, chunk//max(1, len(dr)))
    for start in range(0, len(shifts), step):
        d = dr[None]+shifts[start:start+step, None]
        r2 = np.einsum('spc,spc->sp', d, d)
        w = radial_friction(np.sqrt(r2), **kwargs)/r2
        out += np.einsum('sp,spm->pm', w, d[..., _SYM_A]*d[..., _SYM_B])
    return out[:, _SYM_FULL]


class LatticeFriction:
    """Full periodic-image friction K_ij = sum_n g(|r_ij+n*box|) rhat rhat^T.

    Real-space lattice sum over ALL images (no Ewald splitting, nothing removed in
    Fourier space): the k=0 coefficient ghat(0)/(3V)*I per pair is retained.
    Images with |n*box| <= near_cells*min(box) are summed directly. The remaining
    images are a smooth (analytic) function of the minimum-image displacement and
    are represented by a tensor Chebyshev fit on [-box/2, box/2]^3. The far sum is
    truncated where the closed-form tail estimate is below rel_tol*ghat(0).
    Self-image terms (i=j, n!=0) act on v_i-v_i=0 and cancel exactly in Gamma.
    """
    def __init__(self, box, kernel='A', gamma=0.5, kappa=0.7, r_ref=1.3,
                 rel_tol=1e-10, near_cells=2, degree=16):
        self.box = np.broadcast_to(np.asarray(box, dtype=float), (3,)).copy()
        self.kwargs = dict(kernel=kernel, gamma=gamma, kappa=kappa, r_ref=r_ref)
        self.degree, self.rel_tol = int(degree), float(rel_tol)
        half_diagonal = 0.5*np.linalg.norm(self.box)
        near = near_cells*self.box.min()
        if near <= half_diagonal:
            raise ValueError('near_cells*min(box) must exceed the half cell diagonal.')
        self.ghat0 = float(friction_tail_integral(0., **self.kwargs))
        self.volume = float(np.prod(self.box))
        radii = np.linspace(0., 400./kappa, 400001)
        rel = friction_tail_integral(radii, **self.kwargs)/max(self.ghat0, np.finfo(float).tiny)
        self.r_tail = float(radii[np.argmax(rel <= self.rel_tol)])
        self.tail_estimate = float(friction_tail_integral(self.r_tail, **self.kwargs)/self.volume)
        self.zero_mode = self.ghat0/(3*self.volume)
        shifts = lattice_shifts(self.box, max(near, self.r_tail+half_diagonal))
        is_near = np.einsum('sc,sc->s', shifts, shifts) <= near**2
        self.near_shifts, far_shifts = shifts[is_near], shifts[~is_near]
        nodes = np.cos(np.pi*(np.arange(self.degree+1)+0.5)/(self.degree+1))
        grid = np.stack(np.meshgrid(nodes, nodes, nodes, indexing='ij'), -1).reshape(-1, 3)
        far = image_sum_tensors(grid*self.box/2, far_shifts, **self.kwargs)
        values = far[:, _SYM_A, _SYM_B].reshape((self.degree+1,)*3+(6,))
        inv = np.linalg.inv(np.polynomial.chebyshev.chebvander(nodes, self.degree))
        self.coef = np.einsum('ai,bj,ck,ijkm->abcm', inv, inv, inv, values, optimize=True)
        self._coef_z = self.coef.transpose(2, 0, 1, 3).reshape(self.degree+1, -1)
        self.n_far_images = len(far_shifts)

    def pair_tensors(self, dr):
        """K for minimum-image displacements dr (|dr_c| <= box_c/2), shape (P,3,3)."""
        dr = np.atleast_2d(np.asarray(dr, dtype=float))
        x = np.clip(dr/(self.box/2), -1., 1.)
        bx, by, bz = (np.polynomial.chebyshev.chebvander(x[:, c], self.degree) for c in range(3))
        far = (bz@self._coef_z).reshape(len(dr), self.degree+1, self.degree+1, 6)
        far = np.einsum('pa,pam->pm', bx, np.einsum('pb,pabm->pam', by, far))
        return image_sum_tensors(dr, self.near_shifts, **self.kwargs)+far[:, _SYM_FULL]

    def matrix(self, q):
        i, j, dr, _ = pair_geometry(q, self.box)
        return assemble_friction(i, j, self.pair_tensors(dr), len(q))

    def record(self):
        return dict(images='lattice', rel_tol=self.rel_tol, r_tail=self.r_tail,
                    tail_estimate_per_pair=self.tail_estimate, near_images=len(self.near_shifts),
                    far_images=self.n_far_images, chebyshev_degree=self.degree,
                    ghat0=self.ghat0, zero_mode_per_pair=self.zero_mode)


def thermostat_factors(G, dt, mass, temperature):
    lam, vec = np.linalg.eigh(G)
    tol = 1e-10*max(1., np.abs(lam).max())
    if lam.min() < -tol:
        raise ValueError('Friction has a significantly negative eigenvalue.')
    lam = np.maximum(lam, 0.)  # Only roundoff-level negative eigenvalues.
    damping = np.exp(-dt*lam/mass)
    noise = np.sqrt(mass*temperature*(-np.expm1(-2*dt*lam/mass)))
    return vec, damping, noise


def thermostat_step(p, G, dt, mass, temperature, rng):
    """Exact frozen-q OU update, with total momentum copied explicitly."""
    vec, damping, noise = thermostat_factors(G, dt, mass, temperature)
    mean = p.mean(axis=0)
    pec = (p-mean).reshape(-1)
    z = rng.standard_normal(p.shape)
    z -= z.mean(axis=0)
    out = vec @ (damping*(vec.T@pec) + noise*(vec.T@z.reshape(-1)))
    out = out.reshape(p.shape)
    out -= out.mean(axis=0)
    return out + mean


def simulate(potential='lj', kernel='A', n=64, box=5.5, dt=0.005,
             burn=800, steps=1600, stride=4, temperature=0.7, mass=1.,
             epsilon=1., sigma=1., gamma=0.5, kappa=0.7, r_ref=1.3, seed=42,
             friction_images='minimum', lattice_tol=1e-10, initial_state=None):
    """BAOAB with dense OU thermostat. Save unwrapped positions and velocities.

    initial_state: optional (q, p) to continue an earlier run, e.g. the last saved
    frame (Q[-1], mass*V[-1]); default is a jittered cubic lattice with Maxwell momenta.

    friction_images='minimum': every pair uses only its minimum image (default).
    friction_images='lattice': full periodic-image sum, see LatticeFriction.
    The pair potential is unaffected: r_cut < box/2 makes both conventions identical.
    """
    cfg = dict(locals())
    cfg['initial_state'] = 'jittered lattice' if initial_state is None else 'provided (q, p)'
    if n < 2 or min(box, dt, temperature, mass, epsilon, sigma) <= 0:
        raise ValueError('Invalid physical parameters.')
    if burn < 0 or steps < stride or stride < 1:
        raise ValueError('Require burn>=0, steps>=stride>=1.')
    if box <= 5*sigma:
        raise ValueError('Require box > 2*r_cut = 5*sigma for the pair potential.')
    rng = np.random.default_rng(seed)
    if initial_state is None:
        m = int(np.ceil(n**(1/3)))
        while m**3 < n:
            m += 1
        grid = np.stack(np.meshgrid(*([np.arange(m)]*3), indexing='ij'), -1).reshape(-1, 3)
        q = (grid[:n]+0.5)*box/m + rng.uniform(-0.025, 0.025, (n, 3))*sigma
        p = rng.normal(0, np.sqrt(mass*temperature), (n, 3))
        p -= p.mean(axis=0)
    else:
        q, p = (np.array(a, dtype=float) for a in initial_state)
        if q.shape != (n, 3) or p.shape != (n, 3) or not (np.all(np.isfinite(q)) and np.all(np.isfinite(p))):
            raise ValueError('initial_state must be finite (q, p) arrays of shape (n, 3).')
    force_args = dict(epsilon=epsilon, sigma=sigma)
    fric_args = dict(kernel=kernel, gamma=gamma, kappa=kappa, r_ref=r_ref)
    if friction_images == 'minimum':
        gamma_matrix = lambda x: friction_matrix(x, box, **fric_args)
        cfg['friction_record'] = dict(images='minimum')
    elif friction_images == 'lattice':
        lattice = LatticeFriction(box, rel_tol=lattice_tol, **fric_args)
        gamma_matrix = lattice.matrix
        cfg['friction_record'] = lattice.record()
    else:
        raise ValueError("friction_images must be 'minimum' or 'lattice'.")
    cfg.update(V_kind='velocity (p/m), not momentum', sample_interval=stride*dt,
               burn_in_time=burn*dt, production_time=steps*dt,
               t_origin='t=0 is the first saved frame after burn-in')
    force, energy, rmin = conservative_force(q, box, potential, **force_args)
    Q, V, times, diag = [], [], [], []
    for step in range(burn+steps+1):
        kinetic = np.sum((p-p.mean(axis=0))**2)/(2*mass)
        if step % stride == 0:
            diag.append([step*dt, 2*kinetic/(3*(n-1)), energy/n,
                         kinetic/n, np.linalg.norm(p.sum(axis=0)), rmin])
        if step >= burn and (step-burn) % stride == 0:
            Q.append(q.copy()); V.append(p.copy()/mass); times.append((step-burn)*dt)
        if step == burn+steps:
            break
        p += 0.5*dt*force
        q += 0.5*dt*p/mass
        G = gamma_matrix(q)
        p = thermostat_step(p, G, dt, mass, temperature, rng)
        q += 0.5*dt*p/mass
        force, energy, rmin = conservative_force(q, box, potential, **force_args)
        p += 0.5*dt*force
        if not np.all(np.isfinite(p)) or rmin < 0.45*sigma:
            raise RuntimeError('Unstable close encounter: reduce dt; no force clipping was applied.')
    return dict(Q=np.asarray(Q), V=np.asarray(V), t=np.asarray(times),
                box=np.repeat(box, 3), diagnostics=np.asarray(diag),
                metadata=np.array(json.dumps(cfg)))


def compute_observables(Q, V, t, box, max_lag=None, n_lags=45, origins=40,
                        r_bins=60, vccf_edges=None, modes=None, remove_com=False):
    """Single-trajectory point estimates; no independent-path CI is inferred.

    Correlations average over common time origins. Ordered distinct pairs are
    used in VCCF and van Hove; minimum-image radial bins end at min(box)/2.
    van Hove output is the dimensionless distinct g_d(r,t), with g_d(r,0)=RDF.
    """
    Q, V, t = np.asarray(Q, float), np.asarray(V, float), np.asarray(t, float)
    box = np.broadcast_to(np.asarray(box, float), (3,)).copy()
    if Q.ndim != 3 or Q.shape != V.shape or Q.shape[2] != 3 or Q.shape[1] < 2:
        raise ValueError('Q and V must have equal shape (frames, particles>=2, 3).')
    if len(t) != len(Q) or len(t) < 3 or not np.all(np.diff(t) > 0) or np.any(box <= 0):
        raise ValueError('Need >=3 frames, increasing t and positive box lengths.')
    if not all(np.all(np.isfinite(a)) for a in (Q, V, t, box)):
        raise ValueError('Nonfinite input.')
    h = t[1]-t[0]
    if not np.allclose(np.diff(t), h, rtol=1e-6, atol=abs(h)*1e-8):
        raise ValueError('Uniformly sampled trajectory required.')
    if min(n_lags, origins, r_bins) < 1:
        raise ValueError('n_lags, origins and r_bins must be positive.')
    if remove_com:
        # Constant inertial-frame shift, NOT frame-by-frame velocity detrending.
        vc = V.mean(axis=(0, 1))
        V = V-vc
        Q = Q-(t-t[0])[:, None, None]*vc
    T, N, _ = Q.shape
    limit = (T-1)//2 if max_lag is None else min(T-1, int(max_lag))
    if limit < 1:
        raise ValueError('max_lag must be at least one saved frame.')
    lags = np.unique(np.rint(np.linspace(0, limit, n_lags)).astype(int))
    starts = np.unique(np.rint(np.linspace(0, T-limit-1, min(origins, T-limit))).astype(int))
    bins = np.linspace(0, box.min()/2, r_bins+1)
    vedges = np.asarray(vccf_edges if vccf_edges is not None else
                        np.linspace(0, box.min()/2, 6), float)
    if vedges.ndim != 1 or len(vedges) < 2 or np.any(np.diff(vedges) <= 0) or vedges[0] < 0:
        raise ValueError('vccf_edges must be increasing nonnegative bin edges.')
    modes = np.asarray(modes if modes is not None else [[1,0,0], [0,1,0], [0,0,1]])
    if modes.ndim != 2 or modes.shape[1] != 3 or len(modes) == 0 or not np.allclose(modes, np.rint(modes)):
        raise ValueError('modes must be a nonempty integer array of shape (modes,3).')
    if np.any(np.linalg.norm(modes, axis=1) == 0):
        raise ValueError('Zero mode is excluded.')
    waves = 2*np.pi*modes/box
    unit = waves/np.linalg.norm(waves, axis=1)[:, None]
    phase = np.exp(1j*np.einsum('tnc,kc->tnk', Q % box, waves))
    current = np.einsum('tnc,tnk->tkc', V, phase)/N
    longitudinal = np.einsum('tkc,kc->tk', current, unit)
    transverse = current-longitudinal[..., None]*unit[None, ...]
    L0 = np.mean(np.abs(longitudinal[starts])**2, axis=0)
    T0 = np.mean(np.sum(np.abs(transverse[starts])**2, axis=-1)/2, axis=0)
    v0sq = np.mean(np.sum(V[starts]**2, axis=-1))
    if v0sq <= np.finfo(float).tiny:
        raise ValueError('VACF cannot be normalized for a zero-velocity trajectory.')
    mask = ~np.eye(N, dtype=bool)
    labels, counts = [], np.zeros(len(vedges)-1, dtype=np.int64)
    for s in starts:
        r = np.linalg.norm(minimum_image(Q[s, :, None]-Q[s, None, :], box), axis=-1)[mask]
        lab = np.searchsorted(vedges, r, side='right')-1
        valid = (lab >= 0) & (lab < len(counts))
        labels.append((lab, valid))
        counts += np.bincount(lab[valid], minlength=len(counts))
    vacf, vccf, cl, ct, vh = [], [], [], [], []
    shells = 4*np.pi/3*np.diff(bins**3)
    for lag in lags:
        vacf.append(np.mean(np.sum(V[starts]*V[starts+lag], axis=-1))/v0sq)
        rawl = np.mean((longitudinal[starts+lag]*longitudinal[starts].conj()).real, axis=0)
        rawt = np.mean(np.sum((transverse[starts+lag]*transverse[starts].conj()).real, axis=-1)/2, axis=0)
        cl.append(np.divide(rawl, L0, out=np.full_like(L0, np.nan), where=L0>1e-30))
        ct.append(np.divide(rawt, T0, out=np.full_like(T0, np.nan), where=T0>1e-30))
        cross = np.zeros(len(counts)); hist = np.zeros(r_bins)
        for s, (lab, valid) in zip(starts, labels):
            # V_i(t0) dot V_j(t0+t), selected by r_ij(t0).
            dots = (V[s]@V[s+lag].T)[mask]
            cross += np.bincount(lab[valid], weights=dots[valid], minlength=len(counts))
            r = np.linalg.norm(minimum_image(Q[s+lag, :, None]-Q[s, None, :], box), axis=-1)[mask]
            hist += np.histogram(r, bins=bins)[0]
        vccf.append(np.divide(cross, counts, out=np.full_like(cross, np.nan), where=counts>0))
        vh.append(hist*np.prod(box)/(len(starts)*N*(N-1)*shells))
    return dict(lag_frames=lags, lag_time=lags*h, time_origins=starts,
                vacf=np.asarray(vacf), vccf=np.asarray(vccf),
                vccf_normalized=np.asarray(vccf)/v0sq, vccf_edges=vedges,
                vccf_pair_origin_counts=counts, modes=modes,
                hydro_L=np.asarray(cl), hydro_T=np.asarray(ct),
                radial_edges=bins, radial_centers=(bins[:-1]+bins[1:])/2,
                van_hove_distinct=np.asarray(vh), rdf=np.asarray(vh)[0],
                mean_squared_speed=np.array(v0sq), remove_com=np.array(remove_com))


def save_observables(data, output, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'observables.npz', **data)
    t = data['lag_time']
    headers = ['lag_time', 'VACF'] + [f'VCCF_bin_{i}' for i in range(data['vccf'].shape[1])]
    np.savetxt(output/'correlations.csv', np.column_stack([t, data['vacf'], data['vccf']]),
               delimiter=',', header=','.join(headers), comments='')
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    ax.plot(t, data['vacf'], lw=2); ax.axhline(0, color='.6', lw=.6)
    ax.set(xlabel='Lag time', ylabel='Normalized VACF', title=f'{label}: VACF')
    fig.savefig(output/'vacf.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    edges = data['vccf_edges']
    for i, count in enumerate(data['vccf_pair_origin_counts']):
        if count:
            ax.plot(t, data['vccf'][:, i], label=f'{edges[i]:.2f} <= r0 < {edges[i+1]:.2f}')
    ax.set(xlabel='Lag time', ylabel='Raw VCCF (velocity squared)', title=f'{label}: VCCF')
    ax.legend(fontsize=8); ax.axhline(0, color='.6', lw=.6)
    fig.savefig(output/'vccf.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for key, ax, title in zip(('hydro_L', 'hydro_T'), axes, ('Longitudinal', 'Transverse')):
        for i, mode in enumerate(data['modes']):
            ax.plot(t, data[key][:, i], label=str(tuple(map(int, mode))))
        ax.set(xlabel='Lag time', ylabel='Normalized correlation', title=title)
        ax.legend(fontsize=8); ax.axhline(0, color='.6', lw=.6)
    fig.suptitle(f'{label}: wave modes')
    fig.savefig(output/'hydrodynamic_modes.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    mesh = ax.pcolormesh(data['radial_centers'], t, data['van_hove_distinct'],
                         shading='nearest', cmap='viridis')
    ax.set(xlabel='Minimum-image distance r', ylabel='Lag time', title=f'{label}: distinct van Hove')
    fig.colorbar(mesh, ax=ax, label='g_d(r,t), finite-N normalization')
    fig.savefig(output/'van_hove.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    ax.plot(data['radial_centers'], data['rdf']); ax.axhline(1, color='.6', lw=.6)
    ax.set(xlabel='Distance r', ylabel='g(r)', title=f'{label}: RDF = g_d(r,0)')
    fig.savefig(output/'rdf.png', dpi=160); plt.close(fig)


def make_overview(folder, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for row, potential in enumerate(('double_well', 'lj')):
        r = np.linspace(.88, 2.7, 600)
        u, _ = pair_potential(r, potential)
        axes[row, 0].plot(r, u, color='#334155', lw=2)
        axes[row, 0].set(xlabel='Distance r', ylabel='Pair potential', ylim=(-1.3, 3), title=potential)
        for kernel, color in [('A', '#2563eb'), ('B', '#dc7627')]:
            key = f'{potential}_{kernel}'
            if key not in cases:
                continue
            obs, diag = cases[key]
            axes[row, 1].plot(obs['lag_time'], obs['vacf'], label=kernel, color=color)
            axes[row, 2].plot(diag[:, 0], diag[:, 1], label=kernel, color=color, alpha=.85)
        axes[row, 1].set(xlabel='Lag time', ylabel='Normalized VACF', title='Pilot VACF')
        axes[row, 2].set(xlabel='Simulation time (including burn-in)', ylabel='Temperature', title='Temperature diagnostic')
        for ax in axes[row, 1:]:
            ax.legend(title='Kernel')
    fig.suptitle('Pair double-well / Lennard-Jones: small-system toy pilots\nSingle trajectories; no PRL reproduction or equilibrium/convergence claim', fontsize=13)
    fig.savefig(Path(folder)/'overview.png', dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Run toy simulations and all four statistics.')
    demo.add_argument('--potentials', nargs='+', choices=['double_well','lj'], default=['double_well','lj'])
    demo.add_argument('--kernels', nargs='+', choices=['A','B'], default=['A'])
    demo.add_argument('--n', type=int, default=64); demo.add_argument('--box', type=float, default=5.5)
    demo.add_argument('--burn', type=int, default=800); demo.add_argument('--steps', type=int, default=1600)
    demo.add_argument('--stride', type=int, default=4); demo.add_argument('--dt', type=float, default=.005)
    demo.add_argument('--temperature', type=float, default=.7); demo.add_argument('--mass', type=float, default=1.)
    demo.add_argument('--gamma', type=float, default=.5); demo.add_argument('--kappa', type=float, default=.7)
    demo.add_argument('--r-ref', type=float, default=1.3); demo.add_argument('--seed', type=int, default=42)
    demo.add_argument('--friction-images', choices=['minimum', 'lattice'], default='minimum')
    demo.add_argument('--lattice-tol', type=float, default=1e-10)
    demo.add_argument('--out', default='results')
    analyze = sub.add_parser('analyze', help='Analyze external Q,V,t,box .npz trajectories.')
    analyze.add_argument('trajectory'); analyze.add_argument('--out', default='analysis')
    analyze.add_argument('--max-lag', type=int, default=None, help='Saved frames, NOT integrator steps.')
    analyze.add_argument('--origins', type=int, default=40); analyze.add_argument('--n-lags', type=int, default=45)
    analyze.add_argument('--vccf-edges', type=float, nargs='+'); analyze.add_argument('--remove-com', action='store_true')
    args = p.parse_args()
    if args.command == 'analyze':
        with np.load(args.trajectory, allow_pickle=False) as tr:
            obs = compute_observables(tr['Q'], tr['V'], tr['t'], tr['box'], max_lag=args.max_lag,
                    origins=args.origins, n_lags=args.n_lags, vccf_edges=args.vccf_edges, remove_com=args.remove_com)
        save_observables(obs, args.out, Path(args.trajectory).stem)
        print(f'Saved analysis to {args.out}', flush=True)
        return
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    cases, summaries = {}, {}
    for potential in args.potentials:
        for kernel in args.kernels:
            label = f'{potential}_{kernel}'
            print(f'Running {label}, N={args.n}', flush=True)
            opts = {k:v for k,v in vars(args).items() if k not in ('command','potentials','kernels','out')}
            tr = simulate(potential=potential, kernel=kernel, **opts)
            folder = root/label; folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/'trajectory.npz', **tr)
            obs = compute_observables(tr['Q'], tr['V'], tr['t'], tr['box'])
            save_observables(obs, folder, label+' (toy pilot)')
            d = tr['diagnostics']
            np.savetxt(folder/'diagnostics.csv', d, delimiter=',', comments='',
                       header='time,T,U_per_particle,KE_per_particle,total_momentum_norm,min_distance')
            cases[label] = (obs, d)
            summaries[label] = dict(config=json.loads(str(tr['metadata'])), frames=len(tr['t']),
                max_total_momentum=float(d[:,4].max()), min_distance=float(d[:,5].min()),
                production_mean_temperature=float(d[d[:,0]>=args.burn*args.dt,1].mean()))
            print(f'Finished {label}: max |P|={d[:,4].max():.2e}', flush=True)
    make_overview(root, cases)
    (root/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')


if __name__ == '__main__':
    main()
