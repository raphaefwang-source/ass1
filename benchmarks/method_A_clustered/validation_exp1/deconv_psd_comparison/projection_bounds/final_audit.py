"""Resumable saved-artifact audit for projection bounds; never rerun accuracy.

Only results/audit_final.json is written. Mathematical formulas, uninflated
float64 bounds, correction-only measured norms and historical dense-difference
roundoff are checked and reported separately. Missing stages give partial_fail.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
RESULTS = HERE/'results'
REPO = HERE.parents[4]
EPS = 2.220446049250313e-16
ISSUES = []
WARNINGS = []


def number(value):
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def first(row, *names):
    return next((number(row.get(name)) for name in names if number(row.get(name)) is not None), None)


def truth(value):
    return value is True or str(value).lower() in ('true', '1')


def need(condition, check, detail, kind='integrity'):
    if not condition:
        ISSUES.append(dict(check=check, kind=kind, detail=detail))
    return bool(condition)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path, required=True):
    if not path.exists():
        if required:
            need(False, 'files', f'Missing {path.relative_to(HERE)}', 'incomplete')
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        need(False, 'files', f'Cannot read {path.name}: {exc}')
        return {}


def read_csv(path, required=True):
    if not path.exists():
        if required:
            need(False, 'files', f'Missing {path.relative_to(HERE)}', 'incomplete')
        return []
    try:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            return list(csv.DictReader(stream))
    except (OSError, ValueError, csv.Error) as exc:
        need(False, 'files', f'Cannot read {path.name}: {exc}')
        return []


def old_hashes():
    snapshot = read_json(HERE/'source_snapshot.json')
    expected = snapshot.get('tracked_file_sha256', {})
    changed, missing = [], []
    for relative, expected_sha in expected.items():
        path = REPO/relative
        if not path.exists():
            missing.append(relative)
        elif sha(path) != expected_sha:
            changed.append(relative)
    need(len(expected) == 668, 'preservation', f'Expected initial 668 old tracked files, found {len(expected)}', 'preservation')
    need(not changed and not missing, 'preservation', f'Changed={changed}, missing={missing}', 'preservation')
    references = snapshot.get('independent_references', [])
    reference_failures = []
    for row in references:
        for path_key, hash_key in (('path','sha256'), ('metadata_path','metadata_sha256')):
            path = REPO/row[path_key]
            if not path.exists() or sha(path) != row[hash_key]:
                reference_failures.append(row[path_key])
    need(len(references) == 72 and not reference_failures, 'reference_preservation',
         f'Expected 72 NPZ and sidecars; records={len(references)}, changed={reference_failures}', 'preservation')
    csv_failures = [relative for relative, expected_sha in snapshot.get('source_csv_sha256', {}).items()
                    if not (REPO/relative).exists() or sha(REPO/relative) != expected_sha]
    need(not csv_failures, 'source_csv_preservation', f'Changed source CSV files: {csv_failures}', 'preservation')
    return dict(old_tracked_files=len(expected), changed=changed, missing=missing,
                independent_reference_npz=len(references), independent_reference_sidecars=len(references),
                reference_failures=reference_failures, source_csv_failures=csv_failures)


def source_id(row):
    if row.get('source_row_id'):
        return row['source_row_id']
    name = row.get('source_csv')
    index = first(row, 'source_data_row', 'source_row', 'source_row_index')
    if name and index is not None:
        return f'{name}:{int(index)}'
    return None


def audit_bounds():
    catalog = [row for row in read_csv(HERE/'source_catalog.csv') if row.get('config')]
    rows = read_csv(RESULTS/'bounds.csv')
    by_source = {row['source_row_id']: row for row in catalog}
    cases = defaultdict(list)
    for source in catalog:
        cases[(Path(source['source_csv']).stem, source['data_id'])].append(source)
    expected_dataset = Counter(Path(row['source_csv']).stem for row in catalog)
    actual_dataset = Counter(row.get('dataset') for row in rows)
    need(len(catalog) == 1692 and len(rows) == 1692, 'coverage',
         f'Expected 1692 source configuration rows; catalog={len(catalog)}, bounds={len(rows)}', 'incomplete')
    need(actual_dataset == expected_dataset, 'coverage',
         f'Dataset counts differ: expected {dict(expected_dataset)}, actual {dict(actual_dataset)}')
    used = Counter()
    mismatch, bad_numeric, proxy_misuse = [], [], []
    parameter_tables = defaultdict(set)
    bound_formula_mismatch, certificate_mismatch = [], []
    strict_historical, larger_historical = [], []
    strict_correction, larger_correction = [], []
    inactive_count = dense_count = proxy_count = zero_norm_count = 0
    for index, row in enumerate(rows, 1):
        case = row.get('case_id', row.get('data_id'))
        sid = source_id(row)
        if sid:
            source = by_source.get(sid)
        else:
            choices = cases.get((row.get('dataset'), case), [])
            source = next((candidate for candidate in choices if not used[candidate['source_row_id']]), None)
            sid = source['source_row_id'] if source else None
        if source is None:
            mismatch.append(dict(row=index, case_id=case, reason='no matching unused historical source row'))
            continue
        used[sid] += 1
        if (case != source['data_id'] or row.get('dataset') != Path(source['source_csv']).stem):
            mismatch.append(dict(row=index, source_row_id=sid, reason='case_id/dataset differs from source lineage'))
        for field, aliases in (('E_old', ('E_old','Eold')), ('E_new', ('E_new','Enew')), ('E_proj', ('E_proj','Eproj'))):
            if first(row, *aliases) != number(source[field]):
                mismatch.append(dict(row=index, source_row_id=sid, field=field,
                                     source_value=source[field], new_value=first(row,*aliases),
                                     reason='historical empirical value changed'))
        pid = row.get('parameter_id')
        if pid:
            parameter_tables[pid].add(source['table_key'])
        else:
            mismatch.append(dict(row=index, source_row_id=sid, reason='missing parameter_id'))
        delta = first(row, 'delta')
        B_simple, B_degree, bound = (first(row, key) for key in ('B_simple','B_degree','B'))
        norm = first(row, 'normGammaD','norm_Gamma_D','norm_ref')
        N = first(row, 'N')
        active = truth(row.get('active'))
        offsets = first(row, 'active_correction_offsets','active_offsets')
        correction = first(row, 'correction_norm','correction_norm_absolute','delta_gamma_norm')
        tightness = first(row, 'tightness','bound_tightness','B_over_correction_norm')
        if (any(value is None for value in (delta,B_simple,B_degree,bound,norm,N,offsets))
                or norm <= 0 or min(delta,B_simple,B_degree,bound,offsets) < 0):
            bad_numeric.append(dict(row=index, source_row_id=sid, reason='missing/nonfinite/negative bound fields'))
            continue
        if not math.isclose(norm, number(source['normGammaD']), rel_tol=1e-12, abs_tol=1e-15):
            bad_numeric.append(dict(row=index, source_row_id=sid, reason='reference norm changed'))
        if N != number(source['N']) or active != (offsets > 0):
            bad_numeric.append(dict(row=index, source_row_id=sid, reason='N or active/offset flag inconsistent'))
        # Check the actual uninflated scalar formulas. The tolerance is only
        # an equality comparison; it never changes or replaces a stored bound.
        if not math.isclose(B_simple, N*delta if N>=2 else 0., rel_tol=2e-14, abs_tol=0.):
            bound_formula_mismatch.append(dict(row=index, source_row_id=sid, reason='B_simple differs from N*delta'))
        if bound != min(B_simple,B_degree):
            bound_formula_mismatch.append(dict(row=index, source_row_id=sid, reason='B differs from min(B_simple,B_degree)'))
        dmax = first(row, 'degree_max_eigenvalue','degree_lambda_max')
        if dmax is not None and not math.isclose(B_degree, 2*dmax, rel_tol=2e-14, abs_tol=0.):
            bound_formula_mismatch.append(dict(row=index, source_row_id=sid, reason='B_degree differs from 2*Dmax'))
        reported_relative = first(row, 'B_relative','B_rel','B_normalized','bound_relative')
        if reported_relative is not None and not math.isclose(reported_relative, bound/norm, rel_tol=2e-14, abs_tol=0.):
            bound_formula_mismatch.append(dict(row=index, source_row_id=sid, reason='reported normalized B differs from B/normGammaD'))
        if not active:
            inactive_count += 1
            if correction != 0. or delta != 0. or bound != 0. or tightness is not None:
                certificate_mismatch.append(dict(row=index, source_row_id=sid, reason='inactive true C norm/bound must be exactly zero and tightness unavailable'))
            zero_norm_count += correction == 0.
        elif offsets <= 5000:
            dense_count += 1
            if correction is None or correction < 0:
                certificate_mismatch.append(dict(row=index, source_row_id=sid, reason='active <=5000-offset case lacks correction-only dense norm'))
            elif correction == 0.:
                zero_norm_count += 1
                if tightness is not None:
                    certificate_mismatch.append(dict(row=index, source_row_id=sid, reason='zero true correction norm must not have a finite tightness'))
            elif tightness is None or not math.isclose(tightness, bound/correction, rel_tol=2e-12, abs_tol=0.):
                certificate_mismatch.append(dict(row=index, source_row_id=sid, reason='tightness differs from B divided by correction-only norm'))
        else:
            proxy_count += 1
            if correction is not None or tightness is not None:
                proxy_misuse.append(dict(row=index, source_row_id=sid, reason='>5000 offsets: historical proxy must not populate correction_norm or tightness'))
        empirical = number(source['E_proj'])
        rel = bound/norm
        historical_gap = empirical-rel
        if historical_gap > 0:
            item = dict(source_row_id=sid, case_id=case, active=active,
                        E_proj=empirical, bound_relative=rel, excess=historical_gap)
            strict_historical.append(item)
            comparison_scale = max(1., abs(number(source['E_old']) or 0.), abs(number(source['E_new']) or 0.))
            if historical_gap > 256*EPS*comparison_scale:
                larger_historical.append(item)
        if correction is not None and correction > bound:
            item = dict(source_row_id=sid, case_id=case, correction_norm=correction,
                        B=bound, excess=correction-bound)
            strict_correction.append(item)
            if correction-bound > 256*EPS*max(1., correction, bound):
                larger_correction.append(item)
        # A fully certified total bound is unavailable without a separately
        # certified original discretization error. A measured envelope is fine.
        for field in ('certified_total_bound','epsilon_old_certified','epsilon_old'):
            if number(row.get(field)) is not None:
                certificate_mismatch.append(dict(row=index, source_row_id=sid, reason=f'{field} is populated without an original error certificate'))
    missing = [sid for sid in by_source if not used[sid]]
    duplicate = [sid for sid, count in used.items() if count != 1]
    collisions = {pid:sorted(keys) for pid, keys in parameter_tables.items() if len(keys) != 1}
    table_parameters = defaultdict(set)
    for pid, tables in parameter_tables.items():
        for table in tables:
            table_parameters[table].add(pid)
    duplicated_parameter_ids = {table:sorted(ids) for table, ids in table_parameters.items() if len(ids) != 1}
    need(not missing and not duplicate and not mismatch, 'coverage',
         f'missing={len(missing)}, duplicate={len(duplicate)}, changed lineage/errors={len(mismatch)}')
    need(not collisions, 'parameter_identity', f'Parameter IDs map to multiple tables: {collisions}')
    need(not duplicated_parameter_ids and len(parameter_tables) == 141, 'parameter_identity',
         f'Expected 141 stable one-to-one table parameter IDs; found {len(parameter_tables)}, duplicate mappings={duplicated_parameter_ids}')
    need(not bad_numeric and not bound_formula_mismatch, 'bound_formulas',
         f'Invalid bound fields={len(bad_numeric)}, formula mismatches={len(bound_formula_mismatch)}')
    need(not proxy_misuse and not certificate_mismatch, 'correction_norm_scope',
         f'Historical proxy misuse={len(proxy_misuse)}, correction certificate/tightness mismatches={len(certificate_mismatch)}')
    need(not larger_correction, 'correction_bound', f'{len(larger_correction)} correction-only norm violations exceed the separately stated comparison tolerance')
    if strict_historical or strict_correction:
        WARNINGS.append('Strict float64 comparisons have positive residuals; they are recorded without inflating bounds or replacing historical errors.')
    return dict(source_configuration_rows=len(catalog), bound_rows=len(rows), expected_rows=1692,
                dataset_counts=dict(actual_dataset), expected_dataset_counts=dict(expected_dataset),
                unique_case_ids=len({row.get('case_id',row.get('data_id')) for row in rows}),
                unique_parameter_ids=len(parameter_tables), missing_source_rows=missing, duplicate_source_rows=duplicate,
                unique_table_configuration_pairs=len({(r.get('parameter_id'),r.get('config')) for r in rows}),
                unique_active_case_ids=len({r.get('case_id') for r in rows if truth(r.get('active'))}),
                unique_inactive_case_ids=len({r.get('case_id') for r in rows if not truth(r.get('active'))}),
                unique_active_dense_case_ids=len({r.get('case_id') for r in rows if truth(r.get('active')) and number(r.get('correction_norm')) is not None}),
                unique_active_dense_table_configuration_pairs=len({(r.get('parameter_id'),r.get('config')) for r in rows
                    if truth(r.get('active')) and number(r.get('correction_norm')) is not None}),
                unique_proxy_only_case_ids=len({r.get('case_id') for r in rows if truth(r.get('active')) and number(r.get('correction_norm')) is None}),
                source_mismatches=mismatch, parameter_id_collisions=collisions,
                table_to_multiple_parameter_ids=duplicated_parameter_ids,
                invalid_numeric_fields=bad_numeric, formula_mismatches=bound_formula_mismatch,
                correction_scope_mismatches=certificate_mismatch, historical_proxy_misuse=proxy_misuse,
                inactive_exact_zero_rows=inactive_count, active_dense_norm_rows=dense_count,
                active_historical_proxy_only_rows=proxy_count, exact_zero_true_norm_rows=zero_norm_count,
                strict_historical_Eproj_violations=strict_historical,
                historical_Eproj_violations_above_separate_roundoff_comparison=larger_historical,
                strict_correction_norm_violations=strict_correction,
                correction_norm_violations_above_separate_roundoff_comparison=larger_correction,
                roundoff_comparison='256*float64_epsilon*max(1, comparison magnitude); diagnostic comparison only, never added to B',
                certificate_scope='Exact-arithmetic correction theorem; float64 diagnostics are not interval-certified total discretization bounds.')


def audit_timing_and_case_summaries():
    timing = read_csv(RESULTS/'timing_raw.csv')
    summaries = read_csv(RESULTS/'cases_summary.csv')
    context = read_json(RESULTS/'timing_context.json')
    phases = {'bound_parameter_setup','correction_fft','bound_configuration_setup',
              'stencil','density_fft','degree_convolutions','degree_eigh'}
    cases = {row.get('case_id'):row for row in summaries}
    labels = Counter(row.get('selected_label') for row in summaries)
    need(len(summaries) == 18 and len(cases) == 18 and labels == {'winner':9,'active-badcase':9},
         'incremental_cost', f'Expected two nine-configuration case summaries, found {dict(labels)}', 'incomplete')
    counts = Counter((row.get('case_id'), number(row.get('repeat')), row.get('phase')) for row in timing)
    expected = {(case, float(rep), phase) for case in cases for rep in range(1,10) for phase in phases}
    invalid = [i for i,row in enumerate(timing,1) if number(row.get('seconds')) is None or number(row.get('seconds')) < 0
               or number(row.get('threads')) != 1 or truth(row.get('audit'))
               or row.get('batch') != 'isolated_incremental_bound_setup']
    need(len(timing) == 1134 and set(counts) == expected and all(value == 1 for value in counts.values()) and not invalid,
         'incremental_cost', f'Expected 18 cases x 9 repeats x 7 phases, threads1/auditFalse; '
         f'rows={len(timing)}, missing={len(expected-set(counts))}, invalid={invalid}')
    shared = defaultdict(list)
    for row in timing:
        if row.get('phase') in ('bound_parameter_setup','correction_fft'):
            shared[(row.get('parameter_id'), number(row.get('repeat')), row.get('phase'))].append(row)
    shared_invalid = [str(key) for key,rows in shared.items()
                      if len(rows) != 9 or len({number(row.get('seconds')) for row in rows}) != 1
                      or not all(truth(row.get('parameter_sample_shared_across_configs')) for row in rows)]
    need(not shared_invalid, 'incremental_cost', f'Parameter FFT samples must be shared, not represented as nine independent samples: {shared_invalid}')
    medians_invalid = []
    for case, row in cases.items():
        for phase, column in (('bound_parameter_setup','median_bound_parameter_setup_s'),
                              ('correction_fft','median_correction_fft_s'),
                              ('bound_configuration_setup','median_bound_configuration_setup_s')):
            values = [number(r.get('seconds')) for r in timing if r.get('case_id') == case and r.get('phase') == phase]
            if len(values) != 9 or any(value is None for value in values):
                continue
            saved = number(row.get(column))
            if saved is None or not math.isclose(saved,statistics.median(values),rel_tol=2e-12,abs_tol=0.):
                medians_invalid.append(dict(case_id=case,column=column))
    need(not medians_invalid, 'incremental_cost', f'Case summary setup medians differ from raw samples: {medians_invalid}')
    need(number(context.get('repeats')) == 9, 'incremental_cost', 'Timing context must record nine repeats')
    return dict(raw_rows=len(timing), case_summary_rows=len(summaries), labels=dict(labels),
                invalid_raw_rows=invalid, invalid_shared_samples=shared_invalid,
                median_mismatches=medians_invalid, distinct_shared_parameter_phase_repeats=len(shared),
                context=context)


def audit_verification_summary(bounds):
    summary = read_json(RESULTS/'verification_summary.json')
    expected = dict(source_rows=1692,unique_cases=1602,unique_tables=141,
                    unique_active_cases=bounds['unique_active_case_ids'],
                    literal_zero_cases=bounds['unique_inactive_case_ids'],
                    dense_correction_checked_unique_cases=bounds['unique_active_dense_case_ids'],
                    active_cases_using_historical_full_difference_only=bounds['unique_proxy_only_case_ids'])
    mismatches = {key:dict(recorded=summary.get(key), expected=value) for key,value in expected.items()
                  if number(summary.get(key)) != value}
    need(not mismatches, 'verification_summary', f'Summary counts differ from source rows: {mismatches}')
    need(summary.get('datasets') == bounds['dataset_counts'], 'verification_summary', 'Dataset summary differs from saved bounds CSV')
    need(not truth(summary.get('certified_total_bound_available')), 'verification_summary',
         'Measured original errors must not imply an available certified total bound')
    return dict(recorded=summary,count_mismatches=mismatches)


def audit_tests():
    summary = read_json(RESULTS/'tests_summary.json', required=False)
    path = RESULTS/'test_bounds.log'
    if not path.exists():
        candidates = [RESULTS/name for name in ('unit_tests.log','tests.log') if (RESULTS/name).exists()]
        path = candidates[0] if candidates else path
    log = path.read_text() if path.exists() else ''
    ran = re.search(r'Ran\s+(\d+)\s+tests?', log)
    log_ok = bool(ran and int(ran.group(1)) >= 10 and re.search(r'^OK\s*$', log, re.M)
                  and not re.search(r'^FAILED\b', log, re.M))
    need(log_ok, 'unit_tests', 'Expected saved unittest log confirming at least ten tests and OK', 'incomplete')
    if summary:
        need(truth(summary.get('ok')) and number(summary.get('tests_run')) is not None
             and number(summary.get('tests_run')) >= 10 and number(summary.get('failures')) == 0
             and number(summary.get('errors')) == 0, 'unit_tests', 'Machine-readable test summary does not confirm a clean pass')
    return dict(log=str(path.relative_to(HERE)), log_sha256=sha(path) if path.exists() else None,
                tests_run=int(ran.group(1)) if ran else None, log_ok=log_ok, summary=summary)


def audit_plots():
    folder = HERE/'figures'
    status = read_json(folder/'plot_status.json', required=False)
    names = {p.stem for p in folder.glob('fig*.png')} | {p.stem for p in folder.glob('fig*.svg')}
    names |= {p.stem for p in (folder/'plot_data').glob('fig*.csv')}
    need(len(names) >= 3, 'plots', f'Expected at least three complete figure groups, found {len(names)}', 'incomplete')
    if status:
        need(not status.get('failed') and not status.get('skipped'), 'plots', f'Failed/skipped plot stages: {status}')
    result = {}
    for name in sorted(names):
        png, svg, data = folder/f'{name}.png', folder/f'{name}.svg', folder/'plot_data'/f'{name}.csv'
        present = all(p.exists() and p.stat().st_size for p in (png,svg,data))
        need(present, 'plots', f'{name}: PNG/SVG/source CSV triplet missing or empty', 'incomplete')
        if not present:
            result[name] = dict(complete=False); continue
        raw = png.read_bytes()
        valid_png = len(raw)>=24 and raw[:8] == b'\x89PNG\r\n\x1a\n'
        dimensions = struct.unpack('>II',raw[16:24]) if valid_png else (0,0)
        text = svg.read_text()
        points = read_csv(data)
        valid = bool(valid_png and min(dimensions)>0 and '<svg' in text and '</svg>' in text and points)
        need(valid, 'plots', f'{name}: image encoding invalid or source CSV has no rows')
        result[name] = dict(complete=valid, png_pixels=dimensions, plot_data_rows=len(points),
                            png_sha256=sha(png),svg_sha256=sha(svg),plot_data_sha256=sha(data))
    return dict(groups=len(names), figures=result, plot_status=status)


def main():
    checks = dict(preservation=old_hashes(), bounds=audit_bounds(), tests=audit_tests(), plots=audit_plots())
    checks['verification_summary'] = audit_verification_summary(checks['bounds'])
    checks['incremental_cost'] = audit_timing_and_case_summaries()
    checks['tightness_outlier_diagnostic'] = read_json(RESULTS/'tightness_outlier_diagnostic.json',required=False)
    for name in ('README.md','proof.md','report.md'):
        path = HERE/name
        need(path.exists() and path.stat().st_size>0, 'documents', f'{name} is missing/empty', 'incomplete')
    report = HERE/'report.md'
    need(report.exists() and bool(re.search(r'[\u4e00-\u9fff]',report.read_text() if report.exists() else '')),
         'documents', 'Expected a Chinese report', 'incomplete')
    artifacts = {str(path.relative_to(HERE)):sha(path) for path in sorted(HERE.rglob('*'))
                 if path.is_file() and path != RESULTS/'audit_final.json' and '__pycache__' not in path.parts
                 and path.suffix in ('.py','.csv','.png','.svg','.md','.json','.log')}
    output = dict(timestamp=datetime.now(ZoneInfo('America/Detroit')).isoformat(),
                  status='pass' if not ISSUES else 'partial_fail', ok=not ISSUES,
                  issues=ISSUES, warnings=WARNINGS, checks=checks, artifact_sha256=artifacts,
                  note='Saved-artifact audit only. Historical files and results remain read-only; no bound is inflated and no empirical error is relabeled as certified.')
    RESULTS.mkdir(exist_ok=True)
    target=RESULTS/'audit_final.json'
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=output['status'], issues=len(ISSUES), output=str(target)),ensure_ascii=False))


if __name__=='__main__':
    main()
