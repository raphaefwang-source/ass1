"""Measured, finite-probe Krylov validation for the moving-particle pilot.

Dense references are assembled from pair tensors independently of the FFT
action, then diagonalized on the COM-free subspace. Candidate selection uses
the production FFT actions. These checks certify only the saved probes/states;
they are neither a uniform finite-rank error bound nor a Gibbs proof.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy import linalg

HERE = Path(__file__).resolve().parent
THERMOSTAT = HERE.parent / 'thermostat_ab_local'
for folder in (THERMOSTAT, THERMOSTAT / 'optimized_pairou'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from ou_core import lanczos_action, project, ou_a, ou_b, PSDViolation
from compiled_edge import CompiledEdgeOU
from protocol import DenseOU, ideal_a_mean

ACTION_TOLERANCE = 2e-8
DEFAULT_SEED = 202610280130


def array_sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')


def _csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relative_error(actual, expected):
    return float(np.linalg.norm(np.asarray(actual).ravel() - np.asarray(expected).ravel()) /
                 max(np.linalg.norm(expected), np.finfo(float).tiny))


def _norm_symmetric(matrix):
    eig = linalg.eigvalsh(.5 * (matrix + matrix.T), check_finite=False)
    return float(max(abs(eig[0]), abs(eig[-1])))


def directions(q, reference, p, z):
    """Slow/fast modes, a random vector, actual inputs, and low-k L/T modes.

    Low-wave-number sine/cosine polarizations cover each principal wavevector
    with one longitudinal and two transverse polarizations. Zero/near-zero
    probes are excluded explicitly rather than divided by a tiny norm.
    """
    q = np.asarray(q, float).reshape(-1, 3)
    n = len(q)
    box = (n / .75) ** (1/3)
    rng = np.random.default_rng(DEFAULT_SEED + n)
    out = reference.probes(rng)
    out.pop('COM', None)
    out['actual_momentum'] = project(np.asarray(p).ravel())
    out['actual_gaussian'] = project(np.asarray(z).ravel())
    for axis in range(3):
        for wave_name, values in (('sin', np.sin(2*np.pi*q[:, axis]/box)),
                                  ('cos', np.cos(2*np.pi*q[:, axis]/box))):
            for polarization in range(3):
                v = np.zeros((n, 3))
                v[:, polarization] = values
                v = project(v).ravel()
                if np.linalg.norm(v) <= 1e-12:
                    continue
                label = 'L' if polarization == axis else 'T'
                out[f'lowk_axis{axis}_{wave_name}_{label}{polarization}'] = v
    normalized = {}
    for name, vector in out.items():
        vector = project(vector).ravel()
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            normalized[name] = vector / norm
    return normalized


def validate_probe_set(action, reference, probes, rank, dt, kind,
                       threshold=ACTION_TOLERANCE, label=''):
    """Use the real supplied action, with a dense exact reference per probe."""
    rows, arrays = [], {}
    values = (np.exp(-dt*reference.eig) if kind == 'damping' else
              np.sqrt(-np.expm1(-2*dt*reference.eig)))
    for name, vector in probes.items():
        expected = reference.U @ (values * (reference.U.T @ project(vector).ravel()))
        actual, diag = lanczos_action(action, vector, int(rank), dt, kind)
        error = relative_error(actual, expected)
        row = dict(method=label, kind=kind, probe=name, rank_requested=int(rank),
                   rank_used=int(diag['rank_used']), relative_error=error,
                   threshold=float(threshold), passed=bool(error <= threshold),
                   ritz_min_raw=diag['ritz_min_raw'], ritz_max_raw=diag['ritz_max_raw'],
                   negative_ritz_count=diag['negative_ritz_count'],
                   clipped_negative_count=diag['clipped_negative_count'],
                   matvec_count=diag['matvec_count'],
                   orthogonality_error=diag['orthogonality_error'],
                   t_action_wall_s=diag['t_total_s'],
                   probe_sha256=array_sha(vector),
                   exact_sha256=array_sha(expected), actual_sha256=array_sha(actual))
        rows.append(row)
        arrays[f'{kind}__{name}__probe'] = vector
        arrays[f'{kind}__{name}__exact'] = expected
        arrays[f'{kind}__{name}__actual'] = actual
    return rows, arrays


def _prepared(q, p, params, seed, mesh=None, noise=None):
    from spatial_validation import build_operator, dense_parts, near_edges, long_action
    q, p = np.asarray(q, float), np.asarray(p, float).ravel()
    op = build_operator(q, params, mesh=mesh)
    gs, gl, gh = dense_parts(op)
    full_ref, long_ref = DenseOU(gh), DenseOU(gl)
    edges = CompiledEdgeOU(near_edges(op), n_particles=len(q),
                           configuration_token=array_sha(q), backend='compiled')
    if edges.backend != 'compiled':
        raise RuntimeError(f'Rank validation requires compiled A: {edges.compile_info}')
    if noise is None:
        rng = np.random.default_rng(seed)
        z = rng.normal(size=p.shape)
        ef = rng.normal(size=(len(edges.i), 3))
        er = rng.normal(size=(len(edges.i), 3))
    else:
        rng = np.random.default_rng(seed)
        z = np.asarray(noise['z'], float).ravel()
        ef = np.asarray(noise['ef'] if 'ef' in noise else rng.normal(size=(len(edges.i), 3)), float)
        er = np.asarray(noise['er'] if 'er' in noise else rng.normal(size=(len(edges.i), 3)), float)
        if z.shape != p.shape or ef.shape != (len(edges.i), 3) or er.shape != ef.shape:
            raise ValueError('Actual production Gaussian dimensions do not match the sampled graph')
        if not all(np.isfinite(a).all() for a in (z, ef, er)):
            raise ValueError('Actual production Gaussians must be finite')
    forward, _ = edges.sweep(p, ef, .005/2)
    return dict(op=op, gs=gs, gl=gl, gh=gh, full=full_ref, long=long_ref,
                edges=edges, z=z, ef=ef, er=er, forward=forward,
                actions={'A': lambda v: long_action(op, v), 'B': op.matvec},
                probes={'A': directions(q, long_ref, forward, z),
                        'B': directions(q, full_ref, p, z)})


def _candidate_ranks(start, dimension):
    values = sorted(set([start] + [v for v in (8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256)
                                  if v > start] + [dimension]))
    return [min(v, dimension) for v in values if v <= dimension]


def select_ranks(q, p, params, output_dir, *, tag='state', seed=DEFAULT_SEED,
                 dt=.005, threshold=ACTION_TOLERANCE, mesh=None):
    """Find separate A/long and B/full ranks on this actual saved state.

    Start B at its historical 5/14 only as candidates; every returned rank is
    newly checked against the supplied physical graph and actual FFT action.
    To use a shared rank over several states, take each method/kind maximum,
    then call validate_actions on every state with that common choice.
    """
    if dt != .005:
        raise ValueError('This Stage 0/1 validation is registered only for dt=.005')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / f'ranks_{tag}.json'
    if marker.exists():
        raise FileExistsError(f'Preserve completed rank evidence: {marker}')
    started = perf_counter()
    prepared = _prepared(q, p, params, seed, mesh)
    ranks = dict(A={}, B={})
    attempts = []
    selected_arrays = dict(q=np.asarray(q), p=np.asarray(p), z=prepared['z'],
                           eta_forward=prepared['ef'], eta_reverse=prepared['er'],
                           actual_A_middle_input=prepared['forward'])
    for method in ('B', 'A'):
        reference = prepared['full' if method == 'B' else 'long']
        for kind, key, start in (('damping', 'r_d', 5), ('noise', 'r_n', 14)):
            found = False
            for rank in _candidate_ranks(start, len(np.asarray(p).ravel()) - 3):
                rows, arrays = validate_probe_set(prepared['actions'][method], reference,
                    prepared['probes'][method], rank, dt, kind, threshold, method)
                for row in rows:
                    row.update(state=tag, stage='selection', dt=dt)
                attempts.extend(rows)
                _csv(output / f'rank_attempts_{tag}.csv', attempts)
                print(f'{tag} {method} {kind} r={rank}: max={max(r["relative_error"] for r in rows):.4g}', flush=True)
                if all(r['passed'] for r in rows):
                    ranks[method][key] = int(rank)
                    selected_arrays.update({f'{method}__{key2}': value for key2, value in arrays.items()})
                    found = True
                    break
            if not found:
                _save(output / f'rank_failure_{tag}.json', dict(state=tag, method=method,
                     kind=kind, threshold=threshold, message='No tested feasible rank meets threshold'))
                raise RuntimeError(f'{method}/{kind} could not meet {threshold} on {tag}')
    np.savez_compressed(output / f'rank_probe_outputs_{tag}.npz', **selected_arrays)
    result = dict(**ranks, threshold=threshold, dt=dt, substeps=1, state=tag,
                  seed=int(seed), q_sha256=array_sha(q), p_sha256=array_sha(p),
                  independent_gaussian_streams=dict(z=array_sha(prepared['z']),
                      eta_forward=array_sha(prepared['ef']), eta_reverse=array_sha(prepared['er'])),
                  elapsed_s=perf_counter()-started,
                  interpretation='Finite saved directions at this state only; no uniform rank or Gibbs guarantee')
    _save(marker, result)
    return result


def validate_actions(q, p, params, ranks, output_dir, *, tag='state', seed=DEFAULT_SEED,
                     dt=.005, threshold=ACTION_TOLERANCE, mesh=None,
                     splitting=True, noise=None):
    """Recheck common selected ranks and distinguish three error mechanisms."""
    started = perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / f'action_validation_{tag}.json'
    if marker.exists():
        raise FileExistsError(f'Preserve completed action evidence: {marker}')
    prepared = _prepared(q, p, params, seed, mesh, noise=noise)
    rows, arrays = [], dict(q=np.asarray(q), p=np.asarray(p), z=prepared['z'],
                            eta_forward=prepared['ef'], eta_reverse=prepared['er'])
    for method in ('B', 'A'):
        reference = prepared['full' if method == 'B' else 'long']
        for kind, key in (('damping', 'r_d'), ('noise', 'r_n')):
            batch, values = validate_probe_set(prepared['actions'][method], reference,
                prepared['probes'][method], int(ranks[method][key]), dt, kind, threshold, method)
            for row in batch:
                row.update(state=tag, stage='selected_validation', dt=dt)
            rows.extend(batch)
            arrays.update({f'{method}__{key2}': value for key2, value in values.items()})
    _csv(output / f'action_checks_{tag}.csv', rows)
    rfull, ffull, cfull = prepared['full'].matrices(dt)
    rlong, flong, _ = prepared['long'].matrices(dt)
    pflat = np.asarray(p).ravel()
    outb, db = ou_b(prepared['actions']['B'], pflat, prepared['z'], dt,
                    int(ranks['B']['r_d']), int(ranks['B']['r_n']))
    exactb = rfull @ pflat + ffull @ prepared['z']
    outa, da = ou_a(prepared['actions']['A'], pflat, prepared['z'], prepared['ef'],
                    prepared['er'], prepared['edges'], dt,
                    int(ranks['A']['r_d']), int(ranks['A']['r_n']))
    ideal_middle = rlong @ prepared['forward'] + flong @ prepared['z']
    ideal_a, _ = prepared['edges'].sweep(ideal_middle, prepared['er'], dt/2, reverse=True)
    arrays.update(B_actual=outb, B_exact=exactb, A_actual=outa, A_ideal_split=ideal_a)
    stochastic = dict(B_finite_vs_dense_relative=relative_error(outb, exactb),
                      A_finite_vs_ideal_split_relative=relative_error(outa, ideal_a),
                      A_damp_rank_used=int(da['damp_rank_used']),
                      A_noise_rank_used=int(da['noise_rank_used']),
                      B_damp_rank_used=int(db['damp_rank_used']),
                      B_noise_rank_used=int(db['noise_rank_used']),
                      negative_ritz_count=int(da['negative_ritz_count']+db['negative_ritz_count']),
                      clipped_negative_count=int(da['clipped_negative_count']+db['clipped_negative_count']),
                      A_COM_error=float(np.linalg.norm(outa.reshape(-1,3).sum(0)-pflat.reshape(-1,3).sum(0))),
                      B_COM_error=float(np.linalg.norm(outb.reshape(-1,3).sum(0)-pflat.reshape(-1,3).sum(0))))
    split_rows = []
    if splitting:
        for substeps in (1, 2, 4):
            ra = ideal_a_mean(prepared['edges'], prepared['gl'], dt, substeps)
            ca = prepared['full'].Pi - ra @ prepared['full'].Pi @ ra.T
            split_rows.append(dict(state=tag, dt=dt, substeps=substeps,
                 selected_substeps=1,
                 mean_operator_error=_norm_symmetric(ra-rfull),
                 covariance_operator_error=_norm_symmetric(ca-cfull),
                 covariance_relative_error=_norm_symmetric(ca-cfull)/max(_norm_symmetric(cfull),1e-300),
                 ideal_split_fdt_identity_residual=_norm_symmetric(ra@prepared['full'].Pi@ra.T+ca-prepared['full'].Pi)))
        _csv(output / f'splitting_{tag}.csv', split_rows)
    np.savez_compressed(output / f'action_outputs_{tag}.npz', **arrays)
    result = dict(state=tag, dt=dt, ranks={key:ranks[key] for key in ('A','B')},
                  threshold=threshold, all_probes_passed=all(row['passed'] for row in rows),
                  max_relative_action_error=max(row['relative_error'] for row in rows),
                  negative_ritz_count=stochastic['negative_ritz_count'],
                  stochastic_path_checks=stochastic, splitting=split_rows,
                  q_sha256=array_sha(q), p_sha256=array_sha(p),
                  noise_source=('saved production components with other components independently generated' if noise is not None else
                                'independent Stage 0 probe stream'),
                  supplied_noise_components=sorted(noise.keys()) if noise is not None else [],
                  supplied_noise_sha256={key:array_sha(value) for key,value in noise.items()} if noise is not None else {},
                  z_sha256=array_sha(prepared['z']),
                  eta_forward_sha256=array_sha(prepared['ef']),
                  eta_reverse_sha256=array_sha(prepared['er']),
                  actual_A_middle_sha256=array_sha(prepared['forward']),
                  elapsed_s=perf_counter()-started,
                  interpretation='Krylov-vs-discrete-graph, ideal splitting-vs-full OU, and spatial-vs-physical errors are separate; no full Gibbs or equal-dynamics claim')
    _save(marker, result)
    if not result['all_probes_passed']:
        raise RuntimeError(f'Selected ranks fail actual FFT action checks on {tag}')
    return result
