"""Fresh-q moving-particle BAOAB using the unchanged PR8/PR9 OU kernels."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
from dataclasses import dataclass
from pathlib import Path
import hashlib
import sys
from time import perf_counter
import numpy as np

HERE = Path(__file__).resolve().parent
THERMOSTAT = HERE.parent / 'thermostat_ab_local'
for _path in (THERMOSTAT, THERMOSTAT / 'optimized_pairou'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from ou_core import ou_a, ou_b
from compiled_edge import CompiledEdgeOU
from model import MASS, TEMPERATURE, L_BOX, array_sha, rng_sha, lj_force


class FFTMeter:
    """Scoped FFT-call wall meter; times are nested diagnostic subdivisions.

    The numerical FFT implementation and arguments are unchanged.  The local
    pilot is serial; this scoped instrumentation is not thread-safe.
    """
    def __enter__(self):
        self.elapsed = 0.
        self.calls = 0
        self.saved = {}
        for name in ('rfftn', 'irfftn'):
            original = getattr(np.fft, name)
            self.saved[name] = original
            def wrapped(*args, _original=original, **kwargs):
                t = perf_counter()
                out = _original(*args, **kwargs)
                self.elapsed += perf_counter()-t
                self.calls += 1
                return out
            setattr(np.fft, name, wrapped)
        return self

    def __exit__(self, *exc):
        for name, original in self.saved.items():
            setattr(np.fft, name, original)


@dataclass
class BuiltState:
    op: object
    edges: object
    metrics: dict
    q_mid: np.ndarray
    token: str

    def full_action(self, v):
        return self.op.matvec(np.asarray(v).ravel())

    def long_action(self, v):
        X = np.asarray(v).reshape(self.op.N, 3)
        return (np.einsum('nab,nb->na', self.op.Dm, X) -
                self.op.mesh.apply(self.op.idx, self.op.wt, X)).ravel()


class PilotSystem:
    """Parameter-only tables persist; every build creates a fresh q operator."""
    def __init__(self, params, L=None):
        from spatial_validation import build_mesh
        self.params = dict(params)
        self.L = float(L if L is not None else self.params.get('L', L_BOX))
        start = perf_counter()
        self.mesh = build_mesh(self.params, self.L)
        self.t_parameter_table_setup_s = perf_counter()-start
        self.table_sha256 = array_sha(self.mesh.Gr)
        self.builds = 0
        self.last_built = None

    def build(self, q, method='A', dt=.005):
        from spatial_validation import build_operator, near_edges
        if method not in ('A', 'B'):
            raise ValueError('method must be A or B')
        start = perf_counter()
        q = np.mod(np.asarray(q, dtype=float), self.L)
        token = array_sha(q)
        with FFTMeter() as fft:
            op = build_operator(q, self.params, mesh=self.mesh)
        op_elapsed = perf_counter()-start
        parts = op.t_setup_parts
        metrics = dict(t_operator_setup_s=op_elapsed,
                       t_near_neighbor_images_s=parts['pairs'],
                       t_near_weights_degree_s=parts['real_weights_degree'],
                       t_mesh_assignment_s=parts['mesh_stencil'],
                       t_mesh_degree_s=parts['mesh_degree'],
                       t_setup_fft_s=fft.elapsed, setup_fft_calls=fft.calls,
                       t_edge_aggregate_s=0., t_edge_constructor_s=0.,
                       t_edge_eigh_s=0., t_coefficient_setup_s=0.,
                       edge_count=0, near_image_count=op.n_images,
                       coefficient_cache_initial_size=0,
                       coefficient_first_prepare_hit=False,
                       active_backend='none',
                       fft_parameter_table_reused=True,
                       projection_active=bool(self.mesh.diagnostics['negative_blocks']))
        edges = None
        if method == 'A':
            t = perf_counter()
            records = near_edges(op)
            metrics['t_edge_aggregate_s'] = perf_counter()-t
            # Time precisely the reused constructor's actual eigensolve.
            original_eigh = np.linalg.eigh
            eigh_times = []
            def measured_eigh(*args, **kwargs):
                tick = perf_counter()
                value = original_eigh(*args, **kwargs)
                eigh_times.append(perf_counter()-tick)
                return value
            t = perf_counter()
            np.linalg.eigh = measured_eigh
            try:
                edges = CompiledEdgeOU(records, n_particles=len(q), configuration_token=token)
            finally:
                np.linalg.eigh = original_eigh
            metrics['t_edge_constructor_s'] = perf_counter()-t
            metrics['t_edge_eigh_s'] = sum(eigh_times)
            metrics['coefficient_cache_initial_size'] = edges.coefficient_cache_size
            prepared = edges.prepare(dt/2, MASS, TEMPERATURE)
            metrics.update(t_coefficient_setup_s=prepared['t_coefficient_setup_s'],
                           coefficient_first_prepare_hit=prepared['coefficient_cache_hit'],
                           edge_count=len(edges.i), active_backend=edges.backend)
            if edges.backend != 'compiled':
                raise RuntimeError('Production requires actual compiled backend: '+str(edges.compile_info))
            if np.any(edges.i == edges.j):
                raise RuntimeError('Self-particle near edge')
        metrics['t_dynamic_setup_s'] = perf_counter()-start
        self.builds += 1
        metrics['operator_build_index'] = self.builds
        built = BuiltState(op, edges, metrics, q.copy(), token)
        self.last_built = built
        return built

    def step(self, q_unwrapped, p, method, r_d, r_n, rng, dt=.005):
        """One true B-A-O-A-B step, including both conservative force calls.

        Full wall ends after the second momentum kick; hashing, observations,
        disk I/O, and spatial audits are outside and separately reported.
        Parameter-table creation is a separate one-time system cost.
        """
        q0 = np.asarray(q_unwrapped, dtype=float)
        p0 = np.asarray(p, dtype=float)
        rng_before = rng_sha(rng)
        start = perf_counter()
        t = perf_counter()
        force0, _, _ = lj_force(np.mod(q0, self.L), self.L)
        t_force0 = perf_counter()-t
        ph = p0 + .5*dt*force0
        qm = q0 + .5*dt*ph/MASS
        built = self.build(np.mod(qm, self.L), method, dt)
        t = perf_counter()
        z = rng.standard_normal(p0.shape)
        ef = er = None
        if method == 'A':
            ef = rng.standard_normal((len(built.edges.i), 3))
            er = rng.standard_normal((len(built.edges.i), 3))
        t_rng = perf_counter()-t
        with FFTMeter() as fft:
            if method == 'A':
                po, diag = ou_a(built.long_action, ph, z, ef, er, built.edges,
                                dt, int(r_d), int(r_n), MASS, TEMPERATURE)
            else:
                po, diag = ou_b(built.full_action, ph, z, dt, int(r_d), int(r_n), MASS, TEMPERATURE)
        q1 = qm + .5*dt*po/MASS
        t = perf_counter()
        force1, potential, minimum = lj_force(np.mod(q1, self.L), self.L)
        t_force1 = perf_counter()-t
        p1 = po + .5*dt*force1
        t_baoab = perf_counter()-start
        diagnostic_start = perf_counter()
        if not np.isfinite(q1).all() or not np.isfinite(p1).all():
            raise FloatingPointError('Nonfinite BAOAB state')
        prefix = 'long' if method == 'A' else 'full'
        row = dict(built.metrics, method=method, dt=float(dt), r_d=int(r_d), r_n=int(r_n),
                   r_d_used=diag['damp_rank_used'], r_n_used=diag['noise_rank_used'],
                   substeps=1, t_LJ_force_before_s=t_force0, t_LJ_force_after_s=t_force1,
                   t_LJ_forces_s=t_force0+t_force1, t_rng_s=t_rng,
                   t_pair_forward_s=diag.get('t_pairOU_forward',0.),
                   t_pair_reverse_s=diag.get('t_pairOU_reverse',0.),
                   t_lanczos_damp_s=diag[f't_{prefix}_lanczos_damp'],
                   t_lanczos_noise_s=diag[f't_{prefix}_lanczos_noise'],
                   t_reorth_s=diag['t_reorth'], t_ritz_eigh_s=diag['t_ritz_eigh'],
                   t_ou_wall_s=diag['step_wall_s'], t_ou_fft_s=fft.elapsed,
                   t_fft_s=fft.elapsed+built.metrics['t_setup_fft_s'],
                   ou_fft_calls=fft.calls,
                   t_baoab_s=t_baoab, t_baoab_wall_s=t_baoab,
                   negative_ritz_count=diag['negative_ritz_count'],
                   clipped_roundoff_ritz_count=diag['clipped_negative_count'],
                   ritz_min_damp=diag['damping']['ritz_min_raw'],
                   ritz_min_noise=diag['noise']['ritz_min_raw'],
                   matvec_count=diag['matvec_count'],
                   Tkin=float(np.sum((p1-p1.mean(axis=0))**2)/(3*len(p1)-3)),
                   U_per_particle=potential/len(p1), min_pair_distance=minimum,
                   momentum_x=float(p1[:,0].sum()), momentum_y=float(p1[:,1].sum()),
                   momentum_z=float(p1[:,2].sum()), momentum_norm=float(np.linalg.norm(p1.sum(axis=0))),
                   q_motion_rms=float(np.sqrt(np.mean((q1-q0)**2))),
                   q_mid_sha256=built.token, q_before_sha256=array_sha(q0),
                   q_after_sha256=array_sha(q1), p_before_sha256=array_sha(p0), p_after_sha256=array_sha(p1),
                   rng_state_before_sha256=rng_before, rng_state_after_sha256=rng_sha(rng),
                   z_sha256=array_sha(z), eta_forward_sha256=array_sha(ef) if ef is not None else '',
                   eta_reverse_sha256=array_sha(er) if er is not None else '',
                   q_dependent_setup_rebuilt=True, table_sha256=self.table_sha256)
        # These major phases are disjoint. FFT, eigensolve, and reorthogonalization
        # fields above are nested subdivisions and must never be added again.
        phase_sum = (row['t_dynamic_setup_s']+row['t_LJ_forces_s']+t_rng+row['t_ou_wall_s'])
        row['t_baoab_other_s'] = t_baoab-phase_sum
        row['t_evidence_diagnostics_s'] = perf_counter()-diagnostic_start
        return q1, p1, row
