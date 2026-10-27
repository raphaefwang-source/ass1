"""Final evidence audit. Run after experiments/timing, never concurrently.

Failure means an evidence/integrity/declared deterministic check failed.
Approximate Monte Carlo intervals are described, never used as a proof gate.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import linalg
from report_generator import HERE, RESULTS, read, json_read


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete', action='store_true', help='report missing stages as N/A rather than a failure')
    args = parser.parse_args()
    checks, warnings, metrics = [], [], {}

    def check(name, condition, detail=None):
        checks.append(dict(name=name, passed=bool(condition), detail=detail))

    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=HERE, text=True).strip())
    snapshot_path = HERE / 'source_snapshot.json'
    snapshot = json.loads(snapshot_path.read_text())
    baseline = snapshot['tracked_file_sha256']
    changed = [name for name, expected in baseline.items() if not (repo / name).is_file() or sha(repo / name) != expected]
    check('all_baseline_tracked_hashes_preserved', not changed,
          dict(files_checked=len(baseline), changed_or_missing=changed, source_snapshot_sha256=sha(snapshot_path)))
    prefix = str(HERE.relative_to(repo)) + '/'
    diff = subprocess.check_output(['git', 'diff', '--name-only', snapshot['base_sha']], cwd=repo, text=True).splitlines()
    check('tracked_changes_stay_in_new_directory', all(name.startswith(prefix) for name in diff), diff)
    syntax_errors = []
    for path in HERE.glob('*.py'):
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            syntax_errors.append(str(exc))
    check('python_source_syntax', not syntax_errors, syntax_errors)

    for stage in ('accuracy', 'mc', 'timing'):
        complete = (RESULTS / (stage + '_done.json')).is_file()
        if not complete and args.allow_incomplete:
            warnings.append(stage + ' stage is N/A/incomplete')
        else:
            check(stage + '_completed_marker', complete)
    failures = json_read('failures.json', [])
    check('no_configuration_exceptions', not failures, failures)
    protocol = json_read('protocol.json', {})
    structures, selections = read('structure.csv'), read('selection.csv')
    ranks, splits = read('rank_errors.csv'), read('splitting.csv')
    modes, mf = read('mode_decay.csv'), read('matrixfree_checks.csv')
    mc, timing = read('mc.csv'), read('timing_raw.csv')
    planned = {row['config'] for row in protocol.get('cases', [])}
    actual = {row['config'] for row in structures}
    missing = sorted(planned - actual)
    metrics['coverage'] = dict(planned_systems=len(planned), actual_systems=len(actual), missing=missing,
                               N=sorted({int(row['N']) for row in structures}))
    if missing and args.allow_incomplete:
        warnings.append('Unrun planned configurations: ' + ', '.join(missing))
    else:
        check('declared_configuration_coverage', not missing, metrics['coverage'])
    check('unique_structure_configurations', len(actual) == len(structures))
    alphas = protocol.get('alphas', [.1, .5, 1., 2.])
    expected_selections = {(row['config'], float(alpha), method) for row in structures for alpha in alphas for method in ('A', 'B')}
    actual_selections = {(row['config'], row['alpha'], row['method']) for row in selections}
    check('every_saved_configuration_has_declared_alpha_method_selection', expected_selections == actual_selections,
          dict(expected=len(expected_selections), actual=len(actual_selections)))
    check('finite_structure_values', all(math.isfinite(v) for row in structures for v in row.values() if isinstance(v, (float, int))))

    input_issues, residuals = [], []
    for row in structures:
        path = HERE / 'inputs' / (row['config'] + '.npz')
        if not path.is_file():
            input_issues.append(dict(config=row['config'], error='missing saved input'))
            continue
        if row.get('input_sha256') != sha(path):
            input_issues.append(dict(config=row['config'], error='input hash mismatch'))
        with np.load(path, allow_pickle=False) as saved:
            N = int(row['N'])
            G, GS, GL = (saved[key] for key in ('G', 'GS', 'GL'))
            if any(matrix.shape != (3*N, 3*N) or not np.isfinite(matrix).all() for matrix in (G, GS, GL)):
                input_issues.append(dict(config=row['config'], error='invalid matrix shape/value'))
                continue
            norm = max(float(np.linalg.norm(G)), 1e-300)
            item = dict(config=row['config'], decomposition=float(np.linalg.norm(G-GS-GL)/norm),
                        symmetry=float(np.linalg.norm(G-G.T)/norm),
                        COM=float(np.linalg.norm(G @ np.tile(np.eye(3), (N, 1)))/norm))
            if saved['q'].shape != (N, 3) or saved['p'].size != 3*N or saved['z'].size != 3*N:
                input_issues.append(dict(config=row['config'], error='invalid q/p/z shape'))
            # Recompute independent small-N algebra from the recorded matrices.
            if N <= 16:
                Q = np.kron(linalg.helmert(N, full=False).T, np.eye(3))
                Pi = Q @ Q.T
                e, V = linalg.eigh(Q.T @ G @ Q, check_finite=False)
                item['saved_matrix_reduced_min_eigenvalue'] = float(e.min())
                U = Q @ V
                dt = 1. / row['lambda_max']
                R = np.eye(3*N) - Pi + (U * np.exp(-dt*e)) @ U.T
                F = (U * np.sqrt(-np.expm1(-2*dt*np.maximum(e, 0.)))) @ U.T
                item['recomputed_dense_FDT'] = float(np.linalg.norm(R @ Pi @ R.T + F @ F.T - Pi))
            residuals.append(item)
    check('saved_inputs_integrity', not input_issues, input_issues)
    check('saved_same_graph_decomposition_symmetry_COM',
          all(max(row['decomposition'], row['symmetry'], row['COM']) <= 1e-11 for row in residuals), residuals)
    check('small_N_saved_matrix_independent_FDT',
          all(row.get('recomputed_dense_FDT', 0.) <= 1e-10 for row in residuals))
    for key, threshold in [('mf_full_error', 1e-10), ('mf_long_error', 1e-10), ('dense_fdt_residual', 1e-10),
                           ('momentum_residual', 1e-10), ('symmetry_residual', 1e-10)]:
        values = [row[key] for row in structures if isinstance(row.get(key), (int, float))]
        metrics[key + '_max'] = max(values) if values else None
        check(key, bool(values) and max(values) <= threshold, metrics[key + '_max'])
    material_negatives = [row['config'] for row in structures if min(row.get('lambda_min', 0), row.get('near_min', 0), row.get('long_min', 0)) < -1e-10 * max(1., row['lambda_max'])]
    check('no_materially_negative_discrete_spectrum', not material_negatives, material_negatives)
    spatial_missing = [row['config'] for row in structures if not isinstance(row.get('spatial_error'), (int, float))]
    spatial_failed = [dict(config=row['config'], error=row['spatial_error'], epsilon=row['epsilon']) for row in structures
                      if isinstance(row.get('spatial_error'), (int, float)) and row['spatial_error'] > row['epsilon']]
    check('independent_spatial_accuracy', not spatial_failed and not spatial_missing,
          dict(failed=spatial_failed, missing=spatial_missing))
    physical_bound_issues = []
    reference_hashes = {}
    for row in structures:
        bound = row.get('reference_tail_bound_rel')
        source = Path(row.get('reference_source') or '')
        if not source.is_file():
            physical_bound_issues.append(dict(config=row['config'], error='independent reference source missing'))
            continue
        if str(source) not in reference_hashes:
            reference_hashes[str(source)] = sha(source)
        source_hash = reference_hashes[str(source)]
        if row.get('reference_sha256') != source_hash:
            physical_bound_issues.append(dict(config=row['config'], error='independent reference hash mismatch'))
        if not isinstance(bound, (int, float)) or not 0 <= bound < 1:
            physical_bound_issues.append(dict(config=row['config'], error='invalid reference tail bound'))
        elif not isinstance(row.get('spatial_error'), (int, float)) or (row['spatial_error']+bound)/(1-bound) > row['epsilon']:
            physical_bound_issues.append(dict(config=row['config'], error='spatial error plus analytic tail exceeds epsilon'))
    check('physical_spatial_tolerance_including_analytic_reference_tail', not physical_bound_issues,
          dict(issues=physical_bound_issues, sources_hashed=len(reference_hashes), floating_point_scope='real-arithmetic truncation bound; not interval-certified evaluation'))
    tail_audit = json_read('reference_tail_audit.json', {})
    tail_checks = tail_audit.get('numerical_checks', [])
    tail_valid = (tail_audit.get('status') == 'pass' and len(tail_checks) == 3 and
                  all(row['relative_discrepancy'] <= tail_audit['numerical_relative_tolerance'] for row in tail_checks) and
                  tail_audit.get('script_sha256') == sha(HERE / 'reference_tail_audit.py') and
                  tail_audit.get('spatial_source_sha256') == sha(HERE / 'spatial.py') and
                  tail_audit.get('snapshot_sha256') == sha(snapshot_path) and
                  not tail_audit.get('historical_snapshot_differences', ['missing']))
    if tail_audit.get('original_pilot_sha256'):
        pilot = HERE / tail_audit['original_pilot_evidence']
        tail_valid = tail_valid and pilot.is_file() and sha(pilot) == tail_audit['original_pilot_sha256']
    check('analytic_reference_tail_formula_independent_quadrature', tail_valid,
          dict(numerical_checks=len(tail_checks), maximum_relative_discrepancy=max((row['relative_discrepancy'] for row in tail_checks), default=None)))

    selection_issues = []
    indexed_selection = {(row['config'], row['alpha'], row['method']): row for row in selections}
    for sel in selections:
        if not sel['available']:
            warnings.append(f"N/A selection: {sel['config']} alpha={sel['alpha']} method={sel['method']}")
            continue
        local_tol = protocol.get('rank_tolerance', 1e-5) / sel['substeps']
        operator = 'long' if sel['method'] == 'A' else 'full'
        for kind, key in [('damp', 'r_d'), ('noise', 'r_n')]:
            matches = [row for row in ranks if row['config'] == sel['config'] and row['alpha'] == sel['alpha']
                       and row['operator'] == operator and row['kind'] == kind and row['rank'] == sel[key]]
            probes = set(protocol.get('rank_probes', ['slow', 'fast', 'random']))
            if not probes.issubset({row['probe'] for row in matches}) or any(row['error'] > local_tol + 1e-12 for row in matches):
                selection_issues.append(dict(config=sel['config'], alpha=sel['alpha'], method=sel['method'], kind=kind, error='selected rank/probe budget failed'))
        if sel['split_error'] > sel['tolerance'] + 1e-12:
            selection_issues.append(dict(config=sel['config'], error='splitting tolerance failed'))
        if sel['method'] == 'A':
            rr = [row for row in splits if row['config'] == sel['config'] and row['alpha'] == sel['alpha'] and row['substeps'] == sel['substeps']]
            if not rr or max(rr[0]['mean_operator_error'], rr[0]['cov_operator_error']) > sel['tolerance'] + 1e-12:
                selection_issues.append(dict(config=sel['config'], error='missing/invalid ideal-A split reference'))
    check('selected_ranks_and_splitting_evidence', not selection_issues, selection_issues)
    rank_diag_issues = [dict(config=row['config'], kind=row['kind'], rank=row['rank']) for row in ranks
                        if not math.isfinite(row['error']) or row['error'] < 0 or
                        row.get('diag_rank_used', 0) > row['rank'] or
                        (row.get('diag_ritz_min_raw') is not None and row['diag_ritz_min_raw'] < -row.get('diag_negative_tolerance', 1e-10))]
    check('rank_error_and_negative_Ritz_diagnostics', not rank_diag_issues, rank_diag_issues)
    mf_issues = []
    for row in mf:
        sel = indexed_selection[(row['config'], row['alpha'], row['method'])]
        tolerance = protocol.get('rank_tolerance', 1e-5) / sel['substeps']
        if row['matrixfree_vs_dense_lanczos'] > 1e-9 or row['exact_action_error'] > tolerance + 1e-9:
            mf_issues.append(row)
    check('selected_matrixfree_function_actions', not mf_issues, mf_issues)
    mode_issues = []
    for row in modes:
        sel = indexed_selection[(row['config'], row['alpha'], row['method'])]
        bound = sel['split_error'] + protocol.get('rank_tolerance', 1e-5) + 1e-8
        if row.get('full_exact_mean_action_error', 0.) > bound or row['COM_error'] > 1e-9:
            mode_issues.append(dict(config=row['config'], alpha=row['alpha'], method=row['method'], direction=row['direction'],
                                    actual_error=row.get('full_exact_mean_action_error'), evidence_budget=bound, COM=row['COM_error']))
    check('finite_mean_registered_directions_and_COM', not mode_issues, mode_issues)

    timing_issues = []
    batch = {row.get('batch') for row in timing}
    for row in timing:
        sel = indexed_selection.get((row['config'], row['alpha'], row['method']))
        if not sel or not sel['available'] or any(row.get(key) != sel.get(key) for key in ('r_d', 'r_n', 'substeps')):
            timing_issues.append(dict(config=row['config'], error='timing lacks identical selected ranks/substeps'))
        if row['t_total_frozen_step'] <= 0 or not row.get('gaussian_generation_included'):
            timing_issues.append(dict(config=row['config'], error='invalid total time scope'))
        calls = row.get('actual_full_actions', 0.) + row.get('actual_long_actions', 0.)
        if calls != row.get('matvec_count') or row.get('FFT') != 6*calls or row.get('near_actions') != row.get('actual_full_actions'):
            timing_issues.append(dict(config=row['config'], error='measured action/FFT/near counts disagree'))
        if (row['method'] == 'A' and row.get('actual_full_actions') != 0) or (row['method'] == 'B' and row.get('actual_long_actions') != 0):
            timing_issues.append(dict(config=row['config'], error='wrong friction action domain for method'))
    grouped = {}
    for row in timing:
        grouped.setdefault((row['config'], row['round'], row['rep']), []).append(row['method'])
    if any(sorted(methods) != ['A', 'B'] for methods in grouped.values()):
        timing_issues.append(dict(error='nonpaired same-round repetitions'))
    check('fresh_same_batch_equal_selection_total_timing', not timing_issues and len(batch) <= 1,
          dict(issues=timing_issues, batch=list(batch), rows=len(timing)))
    timing_done = json_read('timing_done.json', {})
    eligible = {row['config'] for row in structures if row['epsilon'] == 1e-4 and
                all(indexed_selection.get((row['config'], 1., method), {}).get('available') for method in ('A', 'B'))}
    expected_timing_rows = len(eligible) * timing_done.get('rounds', 0) * timing_done.get('reps', 0) * 2
    if (RESULTS / 'timing_done.json').exists():
        check('timing_complete_declared_repetitions', len(timing) == expected_timing_rows and timing_done.get('rows') == len(timing),
              dict(eligible_configs=len(eligible), expected_rows=expected_timing_rows, actual_rows=len(timing)))
        setup = read('setup_timing_raw.csv')
        setup_issues = []
        for row in setup:
            b = row['t_table_setup'] + row['t_config_setup']
            a = b + row['t_near_edge_build'] + row['t_near_edge_eigh'] + row['t_edge_dt_factors']
            if not row.get('independent_dense_assembly_excluded') or abs(row['B_fresh_setup']-b) > 1e-12 or abs(row['A_fresh_setup']-a) > 1e-12:
                setup_issues.append(dict(config=row['config'], rep=row['rep'], error='fresh setup phase scope/sum mismatch'))
        check('fresh_setup_costs_separate_and_declared_scope', not setup_issues and len(setup) == 3*sum(row['epsilon'] == 1e-4 for row in structures),
              dict(rows=len(setup), issues=setup_issues))
    if timing and not any('fft' in key.lower() for key in timing[0]):
        warnings.append('FFT counters absent from timing CSV; see matrix-free action implementation and phase costs')
    if timing and not any('near' in key.lower() and ('count' in key.lower() or 'actions' in key.lower()) for key in timing[0]):
        warnings.append('Near-action counters absent from timing CSV')
    mc_issues = [dict(config=row['config'], method=row['method'], phase=row['phase'], direction=row['direction']) for row in mc
                 if row['n_samples'] < 2 or not all(math.isfinite(row[key]) for key in ('mean', 'variance', 'variance_se', 'ci_low', 'ci_high', 'fourth_moment'))
                 or row['variance'] < 0 or not row['ci_low'] <= row['variance'] <= row['ci_high'] or row['com_max'] > 1e-9]
    check('MC_finite_statistics_interval_and_COM_integrity', not mc_issues, mc_issues)
    sample_issues = []
    mc_groups = {}
    for row in mc:
        mc_groups.setdefault((row['config'], row['method'], row['phase']), []).append(row)
    for (config, method, phase), rows in mc_groups.items():
        sample_path = HERE / 'inputs' / f'mc_{config}_{method}_{phase}.npz'
        base_path = HERE / 'inputs' / (config + '.npz')
        if not sample_path.is_file() or not base_path.is_file():
            sample_issues.append(dict(config=config, method=method, phase=phase, error='missing raw MC/input'))
            continue
        with np.load(sample_path, allow_pickle=False) as stored, np.load(base_path, allow_pickle=False) as base:
            samples = stored['samples']
            if samples.shape != (int(rows[0]['n_samples']), 3*int(rows[0]['N'])) or not np.isfinite(samples).all():
                sample_issues.append(dict(config=config, method=method, phase=phase, error='invalid raw MC shape/values'))
                continue
            ideal_samples = stored['ideal_samples'] if 'ideal_samples' in stored else None
            if ideal_samples is None or ideal_samples.shape != samples.shape or not np.isfinite(ideal_samples).all():
                sample_issues.append(dict(config=config, method=method, phase=phase, error='missing/invalid ideal coupled raw samples'))
                ideal_samples = None
            else:
                path_errors = np.linalg.norm(samples-ideal_samples, axis=1)/np.maximum(np.linalg.norm(ideal_samples, axis=1), 1e-300)
                if 'path_errors' not in stored or not np.allclose(stored['path_errors'], path_errors, atol=1e-13, rtol=1e-8):
                    sample_issues.append(dict(config=config, method=method, phase=phase, error='coupled path error mismatch'))
            for row in rows:
                direction = row['direction']
                v = (np.tile([1., 0., 0.], int(row['N'])) / np.sqrt(row['N']) if direction == 'COM' else base[direction])
                x = samples @ v
                values = dict(mean=float(x.mean()), variance=float(x.var(ddof=1)),
                              fourth_moment=float(np.mean((x-x.mean())**4)))
                if ideal_samples is not None:
                    idealx = ideal_samples @ v
                    paired_mean = x-idealx
                    paired_second = (x-row['ideal_method_mean'])**2 - (idealx-row['ideal_method_mean'])**2
                    values.update(paired_mean_error=float(paired_mean.mean()),
                                  paired_mean_se=float(paired_mean.std(ddof=1)/np.sqrt(len(x))),
                                  paired_second_error=float(paired_second.mean()),
                                  paired_second_se=float(paired_second.std(ddof=1)/np.sqrt(len(x))),
                                  coupled_path_error_max=float(path_errors.max()))
                for key, actual_value in values.items():
                    if abs(actual_value-row[key]) > 1e-11*max(1., abs(actual_value)):
                        sample_issues.append(dict(config=config, method=method, phase=phase, direction=direction,
                                                  field=key, saved=row[key], recomputed=actual_value))
    check('MC_statistics_recomputed_from_raw_actual_samples', not sample_issues, sample_issues)
    if (RESULTS / 'mc_done.json').exists():
        expected_mc_configs = {row['config'] for row in structures if row['epsilon'] == 1e-4 and row['seed'] == 2 and
                               (row['family'] == 'uniform' or row['N'] == 16)}
        expected_mc = {(config, method, phase, direction) for config in expected_mc_configs for method in ('A', 'B')
                       if indexed_selection.get((config, 1., method), {}).get('available')
                       for phase in ('conditional', 'equilibrium') for direction in ('slow', 'fast', 'random', 'COM')}
        actual_mc = {(row['config'], row['method'], row['phase'], row['direction']) for row in mc}
        check('MC_declared_case_phase_direction_coverage', expected_mc == actual_mc,
              dict(expected=len(expected_mc), actual=len(actual_mc)))
    metrics['MC_interval_role'] = 'pointwise approximate sample uncertainty; coverage is descriptive, not a proof/pass criterion'
    metrics['MC_rows'] = len(mc)
    metrics['MC_ideal_target_in_CI'] = sum(row['ci_low'] <= row['ideal_method_variance'] <= row['ci_high'] for row in mc if row['direction'] != 'COM')
    metrics['MC_nonCOM_rows'] = sum(row['direction'] != 'COM' for row in mc)
    refinement = json_read('rank_refinement_summary.json', {})
    if refinement:
        pilot = HERE / refinement['pilot_path']
        refinement_issues = []
        for name, expected in refinement.get('preserved_initial_evidence', {}).items():
            path = pilot / name
            if not path.is_file() or sha(path) != expected:
                refinement_issues.append(dict(file=str(path.relative_to(HERE)), error='archived pilot hash mismatch'))
        for name, expected in refinement.get('final_result_sha256', {}).items():
            path = RESULTS / name
            if not path.is_file() or sha(path) != expected:
                refinement_issues.append(dict(file=str(path.relative_to(HERE)), error='refined final evidence hash mismatch'))
        match = indexed_selection.get((refinement['config'], refinement['only_alpha'], refinement['only_method']))
        if not match or match['r_n'] != refinement['new_noise_rank'] or max(refinement['initial_path_errors'].values()) <= 1e-5:
            refinement_issues.append(dict(error='refinement trigger/final selection mismatch'))
        if max(refinement['refined_path_errors'].values()) > 1e-12:
            refinement_issues.append(dict(error='full projected dimension refined case did not reach roundoff'))
        check('post_MC_rank_refinement_disclosed_and_pilot_preserved', not refinement_issues,
              dict(issues=refinement_issues, pilot_files=len(refinement.get('preserved_initial_evidence', {})),
                   old_noise_rank=refinement['old_noise_rank'], new_noise_rank=refinement['new_noise_rank'],
                   observed_trigger=max(refinement['initial_path_errors'].values())))
    path_maximum = max((row.get('coupled_path_error_max', float('inf')) for row in mc), default=None)
    check('MC_observed_coupled_paths_within_declared_sample_guard',
          path_maximum is not None and path_maximum <= protocol.get('rank_tolerance', 1e-5) + 1e-10,
          dict(max_relative_path_error=path_maximum, guard=protocol.get('rank_tolerance', 1e-5),
               scope='observed samples only, separate from function-action rank criterion and any universal covariance bound'))
    stopping = json_read('stopping_summary.json', {})
    if stopping:
        from stopping_audit import evaluate, counts
        individual, aggregate = evaluate(ranks)
        check('stopping_diagnostic_counts_match_rank_evidence',
              stopping.get('source_sha256') == sha(RESULTS / 'rank_errors.csv') and
              stopping.get('per_direction') == counts(individual) and
              stopping.get('all_registered_directions') == counts(aggregate),
              dict(per_direction=counts(individual), all_registered_directions=counts(aggregate),
                   interpretation='false positives/negatives are descriptive; dense error selects benchmark ranks'))
    elif args.allow_incomplete:
        warnings.append('Stopping diagnostic audit is N/A')
    else:
        check('stopping_diagnostic_counts_match_rank_evidence', False, 'stopping_summary.json missing')
    uncensored = json_read('uncensored_stopping_summary.json', {})
    if uncensored:
        from uncensored_stopping import summarize
        from stopping_audit import _decision
        uncensored_rows = read('uncensored_stopping.csv')
        uncensored_both = read('uncensored_stopping_both_probes.csv')
        diagnostic_issues = [row for row in uncensored_rows + uncensored_both if
                             row['decision'] != _decision(row['step_change'], row['error'], row['tolerance'])]
        for key, field, records in [('per_direction', 'per_direction', uncensored_rows),
                                    ('both_registered_probes', 'both_registered_probes', uncensored_both)]:
            for tolerance, expected in uncensored.get(field, {}).items():
                observed = summarize([row for row in records if row['tolerance'] == float(tolerance)])
                if observed != expected:
                    diagnostic_issues.append(dict(scope=key, tolerance=tolerance, error='uncensored summary mismatch'))
        for config, expected in uncensored.get('input_sha256', {}).items():
            if sha(HERE / 'inputs' / (config + '.npz')) != expected:
                diagnostic_issues.append(dict(config=config, error='uncensored saved input hash mismatch'))
        check('uncensored_stopping_evidence_and_counts', not diagnostic_issues and
              uncensored.get('source_rank_errors_sha256_before') == uncensored.get('source_rank_errors_sha256_after'),
              dict(issues=diagnostic_issues, interpretation='diagnostic sample results, no general stopping certificate'))
    elif args.allow_incomplete:
        warnings.append('Uncensored stopping diagnostic is N/A')
    else:
        check('uncensored_stopping_evidence_and_counts', False, 'uncensored_stopping_summary.json missing')
    coupling = read('coupling_checks.csv')
    coupling_issues = [row for row in coupling if row.get('status') != 'run' or row.get('signflip_relative', 1.) > 1e-10 or
                       row.get('coupled_exact_relative', 1.) > 1e-3 or row.get('plus_COM_error', 1.) > 1e-9]
    expected_coupling = {(N, method) for N in (2, 8, 16, 64, 128, 256) for method in ('A', 'B')}
    actual_coupling = {(row['N'], row.get('method')) for row in coupling if row.get('status') == 'run'}
    if args.allow_incomplete and not coupling_issues and expected_coupling != actual_coupling:
        warnings.append('Some coupled physical paths are N/A')
    else:
        check('saved_physical_signflip_and_independent_same_noise_reference',
              not coupling_issues and expected_coupling == actual_coupling,
              dict(rows=len(coupling), issues=coupling_issues, scope='six saved uniform paths; no universal covariance assertion'))
    expected_figures = [HERE / 'figures' / (stem + '.' + ext) for stem in
                        ('rank_vs_error', 'directional_fdt', 'splitting_convergence', 'equal_accuracy_total_time') for ext in ('png', 'svg')]
    if not args.allow_incomplete:
        check('four_figures_in_PNG_and_SVG', all(path.is_file() and path.stat().st_size > 1000 for path in expected_figures))
    test_log = HERE / 'logs' / 'tests.log'
    if test_log.exists():
        test_text = test_log.read_text()
        counts = re.findall(r'Ran (\d+) tests? in ([\d.]+)s', test_text)
        check('logged_unit_and_spatial_tests_passed', bool(counts) and test_text.rstrip().endswith('OK') and 'FAILED (' not in test_text,
              dict(log='logs/tests.log', tests=int(counts[-1][0]) if counts else None,
                   elapsed_s=float(counts[-1][1]) if counts else None))
        metrics['unit_tests'] = int(counts[-1][0]) if counts else None
    else:
        check('logged_unit_and_spatial_tests_passed', False, 'logs/tests.log missing')
    evidence_files = sorted(path for path in HERE.rglob('*') if path.is_file() and
                            (path.suffix in ('.csv', '.json', '.npz', '.py', '.png', '.svg', '.log', '.md') or path.name == '.gitattributes') and
                            path.name not in ('audit.json',) and '__pycache__' not in path.parts)
    manifest = {str(path.relative_to(HERE)): sha(path) for path in evidence_files}
    failed = [row['name'] for row in checks if not row['passed']]
    result = dict(status='FAIL' if failed else ('PASS_WITH_NA' if warnings else 'PASS'),
                  timestamp_utc=datetime.now(timezone.utc).isoformat(), baseline_sha=snapshot['base_sha'],
                  checks=checks, failed_checks=failed, warnings=warnings, metrics=metrics,
                  evidence_sha256=manifest,
                  scope='frozen local thermostat only; deterministic checks have declared finite input domain; MC not exact FDT proof')
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), failed=failed, warnings=len(warnings), baseline_files=len(baseline)), ensure_ascii=False))
    return bool(failed)


if __name__ == '__main__':
    sys.exit(main())
