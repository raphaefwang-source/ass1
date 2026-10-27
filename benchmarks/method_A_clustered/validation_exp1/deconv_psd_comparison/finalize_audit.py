"""Audit saved three-method artifacts without rerunning numerical experiments.

Only results/audit_final.json is written. Missing or partial stages produce a
partial_fail result, so this script can be rerun after the pipeline finishes.
All historical files, direct references, CSV inputs and figures are read-only.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import struct
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
RESULTS = HERE/'results'
VALIDATION = HERE.parent
REPO = VALIDATION.parents[2]
METHODS = ('PPPM-original', 'B2-positive', 'PPPM-grid-PSD')
FAMILIES = ('uniform', 'moderate', 'strong')
FAMILY_SEEDS = set(itertools.product(FAMILIES, (1, 2, 3)))
CONFIGS = {f'{family}_N256_s{seed}_rho0.75' for family, seed in FAMILY_SEEDS}
FIGURES = ('fig1_operator_six_sweeps', 'fig2_action_six_sweeps',
           'fig3_mesh_convergence', 'fig3b_observed_mesh_orders',
           'fig4_projection_error_and_correction', 'fig5_error_vs_time_pareto',
           'fig6_optimal_speed_frontier', 'fig7_optimal_parameters',
           'fig8_setup_and_cached_cost', 'fig8b_matvec_components',
           'fig9_xi_controlled_sweep')
ISSUES = []
CHECKS = {}


def issue(check, message, kind='incomplete'):
    ISSUES.append(dict(check=check, kind=kind, message=message))


def need(condition, check, message, kind='integrity'):
    if not condition:
        issue(check, message, kind)
    return bool(condition)


def number(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def integer(value, default=None):
    value = number(value)
    return int(value) if value is not None and value == int(value) else default


def truth(value):
    return value is True or str(value).lower() == 'true'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path, required=True):
    if not path.exists():
        if required:
            issue('files', f'Missing {path.relative_to(REPO)}')
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        issue('files', f'Cannot read {path.relative_to(REPO)}: {exc}', 'integrity')
        return {}


def read_csv(path, required=True):
    if not path.exists():
        if required:
            issue('files', f'Missing {path.relative_to(REPO)}')
        return []
    try:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            return list(csv.DictReader(stream))
    except (OSError, ValueError, csv.Error) as exc:
        issue('files', f'Cannot read {path.relative_to(REPO)}: {exc}', 'integrity')
        return []


def hashes(expected, check, base=REPO):
    changed, missing = [], []
    for relative, sha256 in expected.items():
        path = base/relative
        if not path.exists():
            missing.append(relative)
        elif digest(path) != sha256:
            changed.append(relative)
    need(not missing and not changed, check,
         f'{len(missing)} missing and {len(changed)} changed files', 'preservation')
    return dict(checked=len(expected), missing=missing, changed=changed)


def preservation():
    initial = read_json(HERE/'audit_initial.json')
    original = initial.get('initial_tracked_file_sha256', initial.get('tracked_file_sha256', {}))
    need(bool(original), 'initial_snapshot', 'Initial tracked-file hash manifest is unavailable', 'preservation')
    result = dict(initial_tracked=hashes(original, 'initial_tracked'))
    # The full initial tracked snapshot includes these B2 sources. The B2's
    # own final source manifest supplies a second independent explicit check.
    b2_audit = read_json(VALIDATION/'positive_mesh_b2/results/final_audit.json')
    b2_sources = b2_audit.get('source_sha256', {})
    result['b2_sources'] = hashes(b2_sources, 'b2_sources')
    refs = initial.get('references', {}).get('rows', [])
    need(len(refs) == 72, 'reference_snapshot', f'Expected 72 initial reference hashes, found {len(refs)}', 'preservation')
    ref_hashes, meta_hashes = {}, {}
    for row in refs:
        relative = row.get('path')
        if relative and row.get('sha256'):
            ref_hashes[relative] = row['sha256']
            if row.get('metadata_sha256'):
                meta_hashes[str(Path(relative).with_suffix('.json'))] = row['metadata_sha256']
    result['references'] = hashes(ref_hashes, 'reference_npz')
    result['reference_metadata'] = hashes(meta_hashes, 'reference_metadata')
    return result


def family_seed(row):
    return row.get('family'), integer(row.get('seed'))


def audit_sweeps():
    rows = read_csv(RESULTS/'sweeps.csv')
    summary = {}
    for method in METHODS:
        own = [r for r in rows if r.get('method') == method]
        groups = defaultdict(list)
        for row in own:
            groups[(row.get('sweep'), row.get('setting'), number(row.get('value')))].append(row)
        duplicate = [key for key, count in Counter((r.get('sweep'), r.get('setting'), number(r.get('value')),
                                                     r.get('config')) for r in own).items() if count > 1]
        bad_groups = [list(key) for key, values in groups.items()
                      if len(values) != 9 or {family_seed(r) for r in values} != FAMILY_SEEDS]
        bad_errors = sum(number(r.get('E_G')) is None or number(r.get('E_GX')) is None for r in own)
        expected_sweeps = {'sR', 'M', 'p', 'rho', 'kappa', 'sT' if method == 'B2-positive' else 'sF'}
        need(len(own) == 927, 'sweeps', f'{method}: expected 927 rows, found {len(own)}', 'incomplete')
        need(len(groups) == 103, 'sweeps', f'{method}: expected 103 parameter groups, found {len(groups)}', 'incomplete')
        need(not duplicate and not bad_groups and not bad_errors, 'sweeps',
             f'{method}: duplicates={len(duplicate)}, incomplete family/seed groups={len(bad_groups)}, bad errors={bad_errors}')
        need({r.get('sweep') for r in own} == expected_sweeps, 'sweeps', f'{method}: six sweep names differ from expected')
        summary[method] = dict(rows=len(own), parameter_groups=len(groups), duplicate_keys=duplicate,
                               bad_configuration_groups=bad_groups, nonfinite_or_missing_errors=bad_errors)
    return summary


def audit_convergence():
    rows = read_csv(RESULTS/'convergence.csv')
    expected = {(family, seed, p, M) for family, seed in FAMILY_SEEDS for p in (2, 3, 5, 7, 9)
                for M in ((24, 32, 40, 48, 64, 80) if p < 5 else (24, 27, 30, 32, 40, 48, 64, 80))}
    summary = {}
    for method in METHODS:
        own = [r for r in rows if r.get('method') == method]
        counter = Counter((*family_seed(r), integer(r.get('p')), integer(r.get('M'))) for r in own)
        actual = set(counter)
        missing, unexpected = expected-actual, actual-expected
        duplicates = [key for key, count in counter.items() if count != 1]
        bad_errors = sum(number(r.get('E_mesh', r.get('E_G_mesh_only'))) is None for r in own)
        need(len(own) == 324, 'convergence', f'{method}: expected 324 rows, found {len(own)}', 'incomplete')
        need(not missing and not unexpected and not duplicates and not bad_errors, 'convergence',
             f'{method}: missing={len(missing)}, unexpected={len(unexpected)}, duplicates={len(duplicates)}, bad mesh errors={bad_errors}')
        summary[method] = dict(rows=len(own), expected_rows=324, missing=sorted(missing),
                               unexpected=sorted(unexpected, key=str), duplicate_keys=duplicates,
                               nonfinite_or_missing_errors=bad_errors)
    return summary


def audit_xi_sweep():
    rows = read_csv(RESULTS/'xi_sweep.csv')
    preset_path = RESULTS/'xi_sweep_preset.json'
    preset = read_json(preset_path)
    provenance = read_json(RESULTS/'xi_sweep_provenance.json')
    xis = (.35, .6, .85, 1., 1.25)
    fixed = dict(N=256, kappa=.2, rho=.75, M=20, p=5, sR=3., sF=3., sT=36., c=1.)
    expected = {(method, family, seed, xi) for method in METHODS
                for family, seed in FAMILY_SEEDS for xi in xis}
    counts = Counter((r.get('method'), *family_seed(r), number(r.get('xi'))) for r in rows)
    missing, unexpected = expected-set(counts), set(counts)-expected
    duplicates = [key for key, count in counts.items() if count != 1]
    invalid, nonfinite, per_method = [], [], {}
    for row in rows:
        key = (row.get('method'), *family_seed(row), number(row.get('xi')))
        wrong = [name for name, value in fixed.items()
                 if number(row.get('amplitude_c' if name == 'c' else name)) != value]
        if (wrong or row.get('sweep') != 'xi' or row.get('setting') != 'controlled_fixed_mesh'
                or number(row.get('value')) != number(row.get('xi')) or row.get('config') not in CONFIGS):
            invalid.append(dict(key=key, changed_fixed_parameters=wrong))
        if any(number(row.get(field)) is None for field in ('E_G', 'E_GX')):
            nonfinite.append(key)
    need(len(rows) == 135, 'xi_sweep', f'Expected 135 controlled xi rows, found {len(rows)}', 'incomplete')
    need(not missing and not unexpected and not duplicates and not invalid and not nonfinite,
         'xi_sweep', f'missing={len(missing)}, unexpected={len(unexpected)}, duplicate={len(duplicates)}, '
         f'changed preset/metadata={len(invalid)}, missing/nonfinite errors={len(nonfinite)}')
    preset_matches = (tuple(number(value) for value in preset.get('xi', ())) == xis
                      and preset.get('fixed') == fixed
                      and set(preset.get('families', ())) == set(FAMILIES)
                      and set(preset.get('seeds', ())) == {1, 2, 3}
                      and set(preset.get('methods', ())) == set(METHODS)
                      and integer(preset.get('expected_rows')) == 135
                      and preset.get('structure') is True and preset.get('actions') is True
                      and preset.get('mesh_only') is False)
    need(preset_matches, 'xi_sweep_preset', 'Saved controlled xi preset differs from the fixed study design')
    preset_hash_matches = bool(preset_path.exists() and provenance.get('preset_sha256') == digest(preset_path))
    need(preset_hash_matches and integer(provenance.get('rows')) == 135,
         'xi_sweep_provenance', 'Xi provenance row count or preset SHA-256 is missing/inconsistent')
    exited_fields = {key: value for key, value in provenance.items()
                     if key in ('workers_exited', 'worker_processes_exited', 'all_workers_exited')}
    need(all(truth(value) for value in exited_fields.values()), 'xi_sweep_provenance',
         'Xi provenance records worker processes that had not exited')
    for method in METHODS:
        per_method[method] = sum(r.get('method') == method for r in rows)
    return dict(rows=len(rows), expected_rows=135, rows_per_method=per_method,
                expected_xi=list(xis), fixed_parameters=fixed,
                missing=sorted(missing), unexpected=sorted(unexpected, key=str), duplicate_keys=duplicates,
                invalid_fixed_parameters=invalid, nonfinite_or_missing_errors=nonfinite,
                preset_matches=preset_matches, provenance_preset_sha256_matches=preset_hash_matches,
                provenance_worker_exit_fields=exited_fields,
                provenance_note=provenance.get('note'),
                csv_sha256=digest(RESULTS/'xi_sweep.csv') if (RESULTS/'xi_sweep.csv').exists() else None)


def audit_search_validation():
    search = read_csv(RESULTS/'search.csv')
    validation = read_csv(RESULTS/'validation.csv')
    summaries = read_csv(RESULTS/'validation_summary.csv')
    budget = read_json(RESULTS/'search_budget.json')
    need(bool(search), 'search', 'Search contains no rows', 'incomplete')
    duplicate_search = [key for key, count in Counter(r.get('id') for r in search).items() if count > 1]
    need(not duplicate_search, 'search', f'Duplicate search IDs: {duplicate_search}')
    by_method = {}
    for method in METHODS:
        own = [r for r in search if r.get('method') == method]
        failed = [r for r in own if r.get('status') != 'ok']
        no_reason = [r.get('id') for r in failed if not r.get('failure')]
        need(bool(own), 'search', f'{method}: search has no rows', 'incomplete')
        need(not no_reason, 'search', f'{method}: failed candidates lack saved reason: {no_reason}')
        expected_budget = budget.get(method, {})
        need(integer(expected_budget.get('independent_candidates')) == len(own), 'search_budget',
             f'{method}: saved search budget disagrees with {len(own)} retained rows')
        need(integer(expected_budget.get('failures')) == len(failed), 'search_budget',
             f'{method}: saved failure budget disagrees with {len(failed)} retained failures')
        by_method[method] = dict(candidates=len(own), failures_retained=len(failed),
                                 failure_ids=[r['id'] for r in failed], failures_without_reason=no_reason,
                                 stage_counts=dict(Counter(r.get('stage') for r in own)))
    groups = defaultdict(list)
    for row in validation:
        groups[row.get('id')].append(row)
    summary_by_id = {row.get('id'): row for row in summaries}
    need(len(summary_by_id) == len(summaries), 'validation', 'Validation summary has duplicate candidate IDs')
    incomplete, mismatch = [], []
    for candidate, own in groups.items():
        configs = [r.get('config') for r in own]
        if len(own) != 9 or set(configs) != CONFIGS or len(set(configs)) != 9:
            incomplete.append(candidate)
        summary = summary_by_id.get(candidate, {})
        if integer(summary.get('n_configs')) != 9:
            mismatch.append(dict(id=candidate, reason='missing nine-configuration summary'))
            continue
        good = all(row.get('status') == 'ok' and number(row.get('E_G')) is not None for row in own)
        if good:
            actual = max(float(row['E_G']) for row in own)
            recorded = number(summary.get('max_E_G'))
            if recorded is None or not math.isclose(actual, recorded, rel_tol=1e-12, abs_tol=1e-15):
                mismatch.append(dict(id=candidate, reason='summary max_E_G differs from actual nine rows'))
            if truth(summary.get('feasible')) != (actual <= 1e-4):
                mismatch.append(dict(id=candidate, reason='feasibility flag differs from actual spectral-norm constraint'))
    need(bool(groups), 'validation', 'No validation candidates', 'incomplete')
    need(not incomplete and not mismatch, 'validation',
         f'Incomplete nine-configuration candidates={len(incomplete)}, inconsistent summaries={len(mismatch)}')
    need(set(summary_by_id) == set(groups), 'validation', 'Summary/validation candidate ID sets differ')
    return dict(search_by_method=by_method, search_duplicate_ids=duplicate_search,
                validation_rows=len(validation), validation_candidates=len(groups),
                incomplete_ids=incomplete, summary_mismatches=mismatch), summary_by_id


def audit_timing(validated, directory=RESULTS, expected_candidates=None):
    summary = read_csv(directory/'final_timing.csv')
    raw = read_csv(directory/'final_timing_raw.csv')
    protocol = read_json(directory/'timing_protocol.json')
    best_json = read_json(directory/'best_configuration.json')
    best = {r.get('method'): r for r in best_json} if isinstance(best_json, list) else best_json
    need(integer(protocol.get('threads')) == 1, 'timing_protocol', 'Timing protocol does not confirm threads=1')
    need(integer(protocol.get('rounds')) == 5 and integer(protocol.get('reps')) == 7, 'timing_protocol',
         'Final protocol must use five rounds and seven samples per configuration')
    env = protocol.get('environment_threads', {})
    need(bool(env) and all(str(val) == '1' for val in env.values()), 'timing_protocol', 'Thread environment is missing or differs from one')
    pools = protocol.get('provenance', {}).get('threadpool', [])
    need(all(integer(pool.get('num_threads')) == 1 for pool in pools), 'timing_protocol', 'An active compute thread pool differs from one thread')
    by_id = {row.get('id'): row for row in summary}
    need(len(by_id) == len(summary) and bool(by_id), 'timing', 'Timing summary is empty or has duplicate IDs', 'incomplete')
    if expected_candidates is not None:
        need(len(summary) == expected_candidates and len(raw) == expected_candidates*315,
             'timing', f'{directory.name}: expected {expected_candidates} candidates and {expected_candidates*315} '
             f'raw samples; found {len(summary)} and {len(raw)}', 'incomplete')
    need({r.get('method') for r in summary} == set(METHODS), 'timing', 'Not every method has common timing', 'incomplete')
    groups = defaultdict(list)
    for row in raw:
        groups[row.get('id')].append(row)
    need(set(groups) == set(by_id), 'timing', 'Raw timing and summary candidate sets differ')
    expected = set(itertools.product(CONFIGS, range(5), range(7)))
    per_candidate = {}
    for candidate, row in by_id.items():
        own = groups.get(candidate, [])
        count = Counter((r.get('config'), integer(r.get('round')), integer(r.get('rep'))) for r in own)
        missing, extra = expected-set(count), set(count)-expected
        duplicate = [key for key, value in count.items() if value > 1]
        invalid = sum(number(r.get('seconds')) is None or number(r.get('seconds'), 0) <= 0
                      or r.get('method') != row.get('method') for r in own)
        need(len(own) == 315 and not missing and not extra and not duplicate and not invalid, 'timing',
             f'{candidate}: n={len(own)}, missing={len(missing)}, extra={len(extra)}, duplicate={len(duplicate)}, invalid={invalid}')
        if not invalid and len(own) == 315 and not missing:
            medians = [statistics.median(float(r['seconds']) for r in own if r['config'] == cfg) for cfg in CONFIGS]
            recomputed = statistics.median(medians)
            need(math.isclose(recomputed, number(row.get('median_matvec_s'), -1), rel_tol=1e-12, abs_tol=1e-15),
                 'timing', f'{candidate}: reported complete-action median differs from raw samples')
        need(candidate in validated and integer(validated[candidate].get('n_configs')) == 9,
             'timing', f'{candidate}: timed point lacks a complete validation certificate')
        required_fields = ('median_matvec_s', 'max_E_G', 'median_config_setup_s',
                           'median_table_setup_s', 'max_relative_iqr', 'table_array_bytes',
                           'median_config_array_bytes', 'max_temporary_estimated_bytes')
        bad_fields = [field for field in required_fields if number(row.get(field)) is None]
        need(not bad_fields, 'timing', f'{candidate}: missing/nonfinite summary fields {bad_fields}')
        per_candidate[candidate] = dict(samples=len(own), expected_samples=315, missing_samples=len(missing),
                                       extra_samples=len(extra), duplicate_samples=len(duplicate), invalid_samples=invalid)
    mainbest = {}
    for method in METHODS:
        row = best.get(method, {})
        candidate = row.get('id')
        error = number(row.get('max_E_G'))
        feasible = [r for r in summary if r.get('method') == method and number(r.get('max_E_G'), math.inf) <= 1e-4]
        fastest = min(feasible, key=lambda r: number(r.get('median_matvec_s'), math.inf)) if feasible else {}
        need(error is not None and error <= 1e-4 and candidate in validated and candidate in by_id,
             'main_best', f'{method}: missing validated/timed best or its max E_G exceeds 1e-4')
        need(candidate == fastest.get('id'), 'main_best', f'{method}: main best is not the fastest feasible common-timed candidate')
        if candidate in validated and error is not None:
            certified = number(validated[candidate].get('max_E_G'))
            need(certified is not None and math.isclose(error, certified, rel_tol=1e-12, abs_tol=1e-15),
                 'main_best', f'{method}: saved best error differs from its nine-configuration validation')
        mainbest[method] = dict(id=candidate, max_E_G=error, median_matvec_s=number(row.get('median_matvec_s')))
    setup_checks = {}
    for filename, keyfields, expected_reps in (
            ('table_setup_raw.csv', ('id',), 3),
            ('config_setup_raw.csv', ('id', 'config'), 3)):
        setup = read_csv(directory/filename)
        setup_groups = defaultdict(list)
        for row in setup:
            setup_groups[tuple(row.get(key) for key in keyfields)].append(row)
        expected_keys = {(candidate,) for candidate in by_id} if len(keyfields) == 1 else set(itertools.product(by_id, CONFIGS))
        invalid_keys = [key for key, values in setup_groups.items()
                        if len(values) != expected_reps or {integer(r.get('rep')) for r in values} != set(range(expected_reps))
                        or any(number(r.get('seconds')) is None or number(r.get('seconds'), -1) < 0 for r in values)]
        need(set(setup_groups) == expected_keys and not invalid_keys, 'timing_setup',
             f'{filename}: missing/extra setup keys or invalid three-fresh-replicate groups')
        setup_checks[filename] = dict(rows=len(setup), groups=len(setup_groups), invalid_groups=invalid_keys)
    return dict(directory=str(directory.relative_to(HERE)), common_timed_candidates=len(summary), raw_samples=len(raw),
                per_candidate=per_candidate, main_best=mainbest, setup=setup_checks)


def audit_primary_timing(validated):
    manifest_path = RESULTS/'primary_archive_manifest.json'
    directory = RESULTS/'primary_timing'
    if not manifest_path.exists() and not directory.exists():
        return dict(present=False)
    manifest = read_json(manifest_path)
    expected = manifest.get('sha256', {})
    need(bool(expected), 'primary_preservation', 'Primary timing archive manifest has no per-file SHA-256 entries', 'preservation')
    preserved = hashes(expected, 'primary_preservation', base=directory)
    required = {'final_timing.csv', 'final_timing_raw.csv', 'timing_protocol.json',
                'best_configuration.json', 'accuracy_time_frontier.csv', 'table_setup_raw.csv',
                'config_setup_raw.csv'}
    need(required <= set(expected), 'primary_preservation',
         f'Primary archive manifest lacks required files: {sorted(required-set(expected))}', 'preservation')
    timing = audit_timing(validated, directory=directory, expected_candidates=15)
    return dict(present=True, manifest_sha256=digest(manifest_path) if manifest_path.exists() else None,
                preserved=preserved, timing=timing)


def audit_cross_winners(validated):
    rows = read_csv(RESULTS/'cross_winner_summary.csv')
    snapshot = read_json(RESULTS/'cross_winner_source_snapshot.json')
    best = snapshot.get('best', {})
    pairs = Counter((row.get('method'), row.get('source_method')) for row in rows)
    expected = {(method, source) for method in METHODS for source in METHODS if method != source}
    need(len(rows) == 6 and set(pairs) == expected and all(count == 1 for count in pairs.values()),
         'cross_winner', f'Expected six unique directed method/source pairs, found {len(rows)} rows', 'incomplete')
    params = ('xi', 'sR', 'sF', 'sT', 'M', 'p', 'kappa', 'rho')
    invalid = []
    for row in rows:
        candidate = row.get('id')
        certificate = validated.get(candidate, {})
        source = best.get(row.get('source_method'), {})
        reasons = []
        if integer(row.get('n_configs')) != 9 or integer(certificate.get('n_configs')) != 9:
            reasons.append('cross candidate lacks complete nine-configuration validation')
        if row.get('status') != 'ok' or certificate.get('status') != 'ok':
            reasons.append('cross candidate validation failed')
        for field in ('max_E_G', 'max_E_GX'):
            saved, actual = number(row.get(field)), number(certificate.get(field))
            if saved is None or actual is None or not math.isclose(saved, actual, rel_tol=1e-12, abs_tol=1e-15):
                reasons.append(f'{field} is missing/nonfinite or differs from complete validation')
        if row.get('source_winner_id') != source.get('id') or any(number(row.get(k)) != number(source.get(k)) for k in params):
            reasons.append('eight cross parameters differ from the snapshotted source winner')
        if reasons:
            invalid.append(dict(id=candidate, reasons=reasons, screen_pass=truth(row.get('screen_pass'))))
    need(not invalid, 'cross_winner', f'{len(invalid)} cross winners have inconsistent parameters or incomplete validation')
    primary_best = RESULTS/'primary_timing/best_configuration.json'
    if primary_best.exists():
        need(snapshot.get('source_sha256') == digest(primary_best), 'cross_winner',
             'Cross-winner source snapshot SHA differs from preserved primary winners', 'preservation')
    return dict(rows=len(rows), expected_rows=6, invalid=invalid,
                screen_failures_with_nine_configuration_validation=sum(
                    not truth(r.get('screen_pass')) and integer(r.get('n_configs')) == 9 for r in rows),
                feasible_cross_candidates=sum(truth(r.get('feasible')) for r in rows))


def audit_frontier():
    rows = read_csv(RESULTS/'frontier.csv')
    epsilons = (1e-3, 1e-4, 1e-5, 1e-6)
    expected = set(itertools.product(METHODS, epsilons))
    counts = Counter((r.get('method'), number(r.get('epsilon'))) for r in rows)
    need(len(rows) == 12 and set(counts) == expected and all(count == 1 for count in counts.values()),
         'frontier', 'Expected 12 unique method/epsilon rows', 'incomplete')
    primary_exists = (RESULTS/'primary_timing').exists()
    batch_per_epsilon, invalid = {}, []
    sources = {}
    for eps in epsilons:
        own = [row for row in rows if number(row.get('epsilon')) == eps]
        batches = {row.get('timing_batch') for row in own}
        batch_per_epsilon[str(eps)] = sorted(batches, key=str)
        if primary_exists:
            batch = 'main_confirmation' if eps == 1e-4 else 'primary'
            source_name = 'accuracy_time_frontier.csv' if eps == 1e-4 else 'primary_timing/accuracy_time_frontier.csv'
            need(len(own) == 3 and batches == {batch}, 'frontier',
                 f'epsilon={eps:g}: all three methods must use the same {batch} timing batch')
            if source_name not in sources:
                sources[source_name] = read_csv(RESULTS/source_name)
            for row in own:
                source = next((r for r in sources[source_name] if r.get('method') == row.get('method')
                               and number(r.get('epsilon')) == eps), {})
                reasons = []
                if row.get('timing_source') != source_name:
                    reasons.append('timing source path differs from the required same-batch source')
                if not source or row.get('id') != source.get('id') or truth(row.get('available')) != truth(source.get('available')):
                    reasons.append('merged candidate/availability differs from preserved source')
                if truth(row.get('available')):
                    for field in ('median_matvec_s', 'max_E_G'):
                        saved, actual = number(row.get(field)), number(source.get(field))
                        if saved is None or actual is None or not math.isclose(saved, actual, rel_tol=1e-12, abs_tol=1e-15):
                            reasons.append(f'{field} differs from the same-batch source')
                    if number(row.get('max_E_G'), math.inf) > eps or integer(row.get('n_configs')) != 9:
                        reasons.append('frontier row lacks the matching nine-configuration error certificate')
                if reasons:
                    invalid.append(dict(method=row.get('method'), epsilon=eps, reasons=reasons))
        elif any(batches):
            need(len(batches) == 1, 'frontier', f'epsilon={eps:g}: methods mix timing batches')
    need(not invalid, 'frontier', f'{len(invalid)} frontier rows disagree with their same-batch sources')
    return dict(rows=len(rows), expected_rows=12, batch_per_epsilon=batch_per_epsilon, source_mismatches=invalid)


def audit_figures():
    figure_dir = HERE/'figures'
    manifest = read_json(figure_dir/'plot_status.json')
    need(not manifest.get('skipped'), 'figures', f'Plot stages skipped: {manifest.get("skipped")}')
    info = {}
    names = set(FIGURES)
    if figure_dir.exists():
        names.update(path.stem for path in figure_dir.glob('fig*.png'))
        names.update(path.stem for path in figure_dir.glob('fig*.svg'))
        names.update(path.stem for path in (figure_dir/'plot_data').glob('fig*.csv'))
    for name in sorted(names):
        png, svg, data = (figure_dir/f'{name}.png', figure_dir/f'{name}.svg', figure_dir/'plot_data'/f'{name}.csv')
        present = all(path.exists() and path.stat().st_size > 0 for path in (png, svg, data))
        need(present, 'figures', f'{name}: PNG, SVG or plotted-series CSV missing/empty', 'incomplete')
        if not present:
            info[name] = dict(complete=False)
            continue
        raw = png.read_bytes()
        valid_png = len(raw) >= 24 and raw[:8] == b'\x89PNG\r\n\x1a\n'
        dims = struct.unpack('>II', raw[16:24]) if valid_png else (0, 0)
        valid_png = valid_png and all(value > 0 for value in dims)
        text = svg.read_text()
        valid_svg = '<svg' in text and '</svg>' in text
        plotted = read_csv(data)
        need(valid_png and valid_svg and bool(plotted), 'figures', f'{name}: invalid image encoding or empty plotted data')
        info[name] = dict(complete=valid_png and valid_svg and bool(plotted), png_pixels=dims,
                          plotted_rows=len(plotted), png_sha256=digest(png), svg_sha256=digest(svg),
                          plot_data_sha256=digest(data))
    return info


def main():
    CHECKS['preservation'] = preservation()
    CHECKS['sweeps'] = audit_sweeps()
    CHECKS['convergence'] = audit_convergence()
    CHECKS['controlled_xi_sweep'] = audit_xi_sweep()
    CHECKS['search_validation'], validated = audit_search_validation()
    primary_present = (RESULTS/'primary_timing').exists() or (RESULTS/'primary_archive_manifest.json').exists()
    CHECKS['timing'] = audit_timing(validated, expected_candidates=6 if primary_present else None)
    CHECKS['primary_timing'] = audit_primary_timing(validated)
    CHECKS['cross_winners'] = audit_cross_winners(validated)
    CHECKS['frontier'] = audit_frontier()
    CHECKS['figures'] = audit_figures()
    report = HERE/'report.md'
    need(report.exists() and bool(re.search(r'[\u4e00-\u9fff]', report.read_text() if report.exists() else '')),
         'report', 'Chinese report.md is absent or contains no Chinese text', 'incomplete')
    new_artifacts = {str(path.relative_to(HERE)): digest(path)
                     for folder in (RESULTS, HERE/'figures') if folder.exists()
                     for path in sorted(folder.rglob('*'))
                     if path.is_file() and path.suffix.lower() in ('.csv', '.png', '.svg')}
    result = dict(timestamp=datetime.now(ZoneInfo('America/Detroit')).isoformat(),
                  status='pass' if not ISSUES else 'partial_fail', complete=not ISSUES,
                  issue_count=len(ISSUES), issues=ISSUES, checks=CHECKS,
                  expected=dict(sweep_rows_per_method=927, convergence_rows_per_method=324,
                                controlled_xi_sweep_rows=135,
                                primary_timing_candidates=15, primary_timing_samples=4725,
                                confirmation_timing_candidates=6, confirmation_timing_samples=1890,
                                cross_winner_candidates=6,
                                configurations_per_validation=9, timing_samples_per_candidate=315,
                                timing_threads=1, figure_png_svg_csv_triplets=len(FIGURES)),
                  new_artifact_sha256=new_artifacts,
                  audit_script_sha256=digest(Path(__file__)),
                  note='Saved-artifact audit only; no old input or result is rewritten, and experiments are not rerun.')
    RESULTS.mkdir(exist_ok=True)
    target = RESULTS/'audit_final.json'
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(status=result['status'], issues=len(ISSUES), output=str(target)), ensure_ascii=False))


if __name__ == '__main__':
    main()
