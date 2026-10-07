#!/usr/bin/env python3
"""
Conservative toy dynamics (pair double well / Lennard-Jones) with the project's finite-time FDT thermostat.

One step, B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2):
    p += dt/2 F(q);  q_half = q + dt/(2m) p
    O:  Gamma = Gamma(q_half);  p_new = p_mean + R_dt Pi p + S_dt Pi xi,
        R_dt = exp(-dt Gamma/m),  S_dt = f_dt(Gamma) = sqrt(kT m [1 - exp(-2 dt Gamma/m)])
        (damping and noise come from the SAME Gamma instance; total momentum is copied exactly)
    q = q_half + dt/(2m) p_new;  p += dt/2 F(q)
F(q) = -grad sum_{i<j} u(r_ij), u = toy LJ or double well with the C2 quintic switch on [2, 2.5] sigma, including
the S'(r) u0(r) term (toy_models/prl_toy_models.pair_potential, unchanged). r_cut = 2.5 < L/2, so minimum image is
exact for the forces; only the friction needs the periodic-image convention.

Friction operators (same law, gamma, kappa, r_ref, box):
    reference     v2 full-periodic lattice sum (LatticeFriction), dense eigendecomposition on Range(Pi)
    pppm_dense    project PPPM Gamma_h (radial_kernels.FastFriction), dense eigendecomposition (DenseRef)
    pppm_lanczos  project PPPM Gamma_h literal action + direct-start Lanczos, noise rank 40 / damping rank 16
                  (production configuration of dynamics_results/REPORT.md), test_true_dynamics.lanczos_apply
All methods consume one standard-normal (N,3) draw per step from the same Generator, so runs started from the
same (q, p) and seed are coupled pathwise. Positions are stored unwrapped (the operators wrap them mod L);
V is velocity p/m. No clipping of Ritz values or eigenvalues: a negative Ritz value raises.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import prl_toy_models as v2  # noqa: E402
import radial_kernels as rk  # noqa: E402
from test_lanczos_fdt import DenseRef, proj  # noqa: E402
from test_true_dynamics import lanczos_apply  # noqa: E402

METHODS = ("reference", "pppm_dense", "pppm_lanczos")
PPPM_PRODUCTION = dict(xi=0.7, s=4.10, eta=0.7, p=7)
RANK_NOISE, RANK_DAMP = 40, 16


def f_noise(dt, kT, m):
    """Finite-time FDT noise function with explicit temperature (test_true_dynamics.f_noise assumes beta = 1)."""
    return lambda l: np.sqrt(kT * m * -np.expm1(-2 * dt * l / m))


def f_damp(dt, m):
    return lambda l: np.exp(-dt * l / m)


class Thermostat:
    def __init__(self, method, L, law, gamma, kappa, r_ref, dt, kT, mass, pppm=None,
                 rank_noise=RANK_NOISE, rank_damp=RANK_DAMP, pair_search="images"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method, self.L, self.dt, self.kT, self.m = method, float(L), dt, kT, mass
        self.fn, self.fd = f_noise(dt, kT, mass), f_damp(dt, mass)
        self.rank_noise, self.rank_damp = rank_noise, rank_damp
        self.pppm = dict(PPPM_PRODUCTION if pppm is None else pppm)
        self.record = dict(method=method, law=law, gamma=gamma, kappa=kappa, r_ref=r_ref, dt=dt, kT=kT, mass=mass,
                           amplitude=rk.amplitude(law, gamma, kappa, r_ref))
        if method == "reference":
            self.lat = v2.LatticeFriction(L, kernel=law, gamma=gamma, kappa=kappa, r_ref=r_ref)
            self.record["operator"] = self.lat.record()
        else:
            K = rk.toy_kernel(law, gamma, kappa, r_ref, self.pppm["xi"])
            self.fast = rk.FastFriction(L, K, self.pppm["s"], self.pppm["eta"], self.pppm["p"], pair_search=pair_search)
            self.record["operator"] = self.fast.record()
            if method == "pppm_lanczos":
                self.record.update(rank_noise=rank_noise, rank_damp=rank_damp)

    def o_step(self, q_half, p, xi):
        N = len(p)
        pm = p.mean(axis=0)
        pp = proj(p.ravel())
        z = proj(xi.ravel())
        info = {}
        if self.method == "pppm_lanczos":
            op = self.fast.gamma_h(q_half)                      # one Gamma_h for damping AND noise
            damp, (dmin, dmax) = lanczos_apply(op, pp, self.rank_damp, self.fd)
            noise, (nmin, nmax) = lanczos_apply(op, z, self.rank_noise, self.fn)
            info.update(lam_min=min(dmin, nmin), lam_max=max(dmax, nmax), n_actions=op.n_actions)
        else:
            G = self.lat.matrix(q_half % self.L) if self.method == "reference" else self.fast.gamma_h(q_half).dense()
            ref = DenseRef(G, N)
            if ref.lam[0] < -1e-10 * max(1.0, ref.lam[-1]):
                raise ValueError(f"Gamma has a negative eigenvalue on Range(Pi): {ref.lam[0]:.3e}")
            damp, noise = ref.apply(self.fd, pp), ref.apply(self.fn, z)
            info.update(lam_min=float(ref.lam[0]), lam_max=float(ref.lam[-1]))
        p_new = pm[None, :] + (proj(damp) + proj(noise)).reshape(N, 3)
        return p_new, info


def conservative_force_neighbor(q, box, potential, epsilon=1.0, sigma=1.0, r_on=2.0, r_cut=2.5):
    """Same pair potential, switch and switch-derivative term as v2.conservative_force (v2.pair_potential), but the
    pairs come from a periodic KD-tree within r_cut*sigma instead of all N(N-1)/2 pairs: O(N log N).

    Forces and energy are identical to the all-pair version up to summation order (pairs beyond r_cut contribute
    exactly zero there). rmin is the minimum over pairs within the cutoff; it equals the all-pair minimum whenever
    any pair is closer than r_cut (always at the toy densities). Requires r_cut*sigma < box/2.
    """
    L = float(np.asarray(box).ravel()[0])
    rc = r_cut * sigma
    if not rc < L / 2:
        raise ValueError("neighbour list requires r_cut*sigma < box/2")
    q = np.asarray(q, float)
    n = len(q)
    x = np.mod(q, L)
    x[x >= L] = 0.0                                          # guard the rounding case q % L == L
    pairs = cKDTree(x, boxsize=L).query_pairs(rc, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros_like(q), 0.0, np.inf
    i, j = pairs[:, 0], pairs[:, 1]
    dr = v2.minimum_image(q[i] - q[j], L)
    r = np.linalg.norm(dr, axis=1)
    if np.any(r < 1e-12):
        raise ValueError("Coincident particles.")
    u, du = v2.pair_potential(r, potential, epsilon=epsilon, sigma=sigma, r_on=r_on, r_cut=r_cut)
    fij = -du[:, None] * dr / r[:, None]
    force = np.stack([np.bincount(i, fij[:, c], n) - np.bincount(j, fij[:, c], n) for c in range(3)], axis=1)
    return force, float(u.sum()), float(r.min())


FORCES = {"neighbor": conservative_force_neighbor, "allpairs": v2.conservative_force}


def run(potential, thermostat, q0, p0, steps, stride, seed, burn=0, stop_rmin=0.45, force_method="neighbor"):
    """Integrate `burn + steps` steps from (q0, p0); save every `stride` steps after burn-in.

    Returns dict with Q (unwrapped), V (velocity), t, box, diagnostics, final state and RNG state for restart.
    """
    th = thermostat
    dt, m, L = th.dt, th.m, th.L
    rng = np.random.default_rng(seed)
    q, p = np.array(q0, float), np.array(p0, float)
    n = len(q)
    conservative_force = FORCES[force_method]               # runs before 2026-10-08 used "allpairs"
    force, energy, rmin = conservative_force(q, L, potential)
    Q, V, T, diag = [], [], [], []
    t_wall = time.perf_counter()
    lam_rng = [np.inf, -np.inf]
    for step in range(burn + steps + 1):
        if step % stride == 0:
            kin = np.sum((p - p.mean(axis=0)) ** 2) / (2 * m)
            diag.append([step * dt, 2 * kin / (3 * (n - 1)), energy / n, kin / n,
                         np.linalg.norm(p.sum(axis=0)), rmin])
            if step >= burn:
                Q.append(q.copy()); V.append(p / m); T.append((step - burn) * dt)
        if step == burn + steps:
            break
        xi = rng.standard_normal((n, 3))
        p = p + 0.5 * dt * force
        q = q + 0.5 * dt / m * p
        p, info = th.o_step(q, p, xi)
        lam_rng = [min(lam_rng[0], info["lam_min"]), max(lam_rng[1], info["lam_max"])]
        q = q + 0.5 * dt / m * p
        force, energy, rmin = conservative_force(q, L, potential)
        p = p + 0.5 * dt * force
        if not np.all(np.isfinite(p)) or rmin < stop_rmin:
            raise RuntimeError("Unstable close encounter: reduce dt (no force clipping is applied).")
    meta = dict(potential=potential, thermostat=th.record, steps=steps, burn=burn, stride=stride, seed=seed,
                force_method=force_method,
                sample_interval=stride * dt, burn_in_time=burn * dt, production_time=steps * dt,
                V_kind="velocity (p/m), not momentum", positions="unwrapped", n=n, box=L,
                lam_range=lam_rng, wall_time=time.perf_counter() - t_wall)
    return dict(Q=np.asarray(Q), V=np.asarray(V), t=np.asarray(T), box=np.repeat(L, 3),
                diagnostics=np.asarray(diag), q_final=q, p_final=p,
                rng_state=np.array(json.dumps(rng.bit_generator.state)), metadata=np.array(json.dumps(meta)))
