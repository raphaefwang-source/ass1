"""Read-only inventory and preservation snapshot for the projection-bound study.

Only this new projection_bounds directory is written. No operator, reference,
accuracy measurement, or prior timing is recomputed. Repeated runs retain the
first snapshot and reject changed historical inputs.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent
RESULTS = PREVIOUS/'results'
REPO = PREVIOUS.parents[3]
SOURCE_NAMES = ('convergence', 'validation', 'sweeps', 'xi_sweep',
                'best_projection_diagnostics', 'final_timing')
OPTIONAL_NAMES = ('search', 'table_diagnostics', 'best_table_diagnostics',
                  'best_configuration', 'cross_winner_summary')
METHOD = 'PPPM-grid-PSD'
TABLE_FIELDS = ('L', 'kappa', 'xi', 'sF', 'M', 'p')
PARAM_FIELDS = ('xi', 'sR', 'sF', 'sT', 'M', 'p', 'kappa', 'rho')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_read(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def first(row, *names):
    return next((number(row.get(name)) for name in names if number(row.get(name)) is not None), None)


def keynumber(value):
    return format(float(value), '.12g') if value is not None else 'missing'


def physical(row):
    par = {key: number(row.get(key)) for key in PARAM_FIELDS}
    par['sF'] = first(row, 'sF', 'sF_req')
    par['N'] = first(row, 'N') or 256
    par['L'] = first(row, 'L')
    if par['L'] is None and par['rho']:
        par['L'] = (par['N']/par['rho'])**(1/3)
    return par


def table_key(par):
    return '_'.join(key+keynumber(par.get(key)) for key in TABLE_FIELDS)


def operator_key(row):
    par = physical(row)
    return (row.get('config'), *(keynumber(par.get(key)) for key in ('xi','sR','sF','M','p','kappa','rho')))


def run_git(*args):
    return subprocess.run(['git', *args], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout


def snapshot_sources(source_paths):
    rel_new = HERE.relative_to(REPO)
    tracked = [path for path in run_git('ls-files', '-z').split('\0') if path
               and not Path(path).is_relative_to(rel_new)]
    tracked_sha = {name: sha(REPO/name) for name in tracked}
    legacy = json.loads((PREVIOUS/'audit_initial.json').read_text())
    references = []
    for original in legacy['references']['rows']:
        relative = original['path']
        path = REPO/relative
        metadata = path.with_suffix('.json')
        actual, meta_actual = sha(path), sha(metadata)
        if actual != original['sha256'] or meta_actual != original['metadata_sha256']:
            raise RuntimeError(f'Historical independent reference changed: {relative}')
        references.append(dict(path=relative, sha256=actual, bytes=path.stat().st_size,
                               metadata_path=str(metadata.relative_to(REPO)), metadata_sha256=meta_actual,
                               source=original['source'], position_hash=original['position_hash']))
    sources = {str(path.relative_to(REPO)): sha(path) for path in source_paths}
    value = dict(created_at=datetime.now(ZoneInfo('America/Detroit')).isoformat(),
                 git=dict(head=run_git('rev-parse', 'HEAD').strip(),
                          branch=run_git('branch', '--show-current').strip(),
                          status=run_git('status', '--porcelain').strip()),
                 excluded_new_directory=str(rel_new), tracked_file_sha256=tracked_sha,
                 independent_references=references, source_csv_sha256=sources,
                 note='All old tracked files and 72 independent references/sidecars are read-only inputs.')
    target = HERE/'source_snapshot.json'
    if target.exists():
        old = json.loads(target.read_text())
        for key in ('tracked_file_sha256', 'independent_references', 'source_csv_sha256'):
            if old[key] != value[key]:
                raise RuntimeError(f'Historical input set changed since first bound-study snapshot: {key}')
        return old
    json_write(target, value)
    return value


def main():
    paths = [RESULTS/f'{name}.csv' for name in SOURCE_NAMES]
    paths += [RESULTS/f'{name}.csv' for name in OPTIONAL_NAMES if (RESULTS/f'{name}.csv').exists()]
    snapshot = snapshot_sources(paths)
    refs = csv_read(PREVIOUS.parent/'positive_mesh_b2/results/references.csv')
    reference_norms = {(row['config'], number(row['kappa'])): number(row['norm2']) for row in refs}
    source_summaries, catalog = {}, []
    original_mesh = {}
    for name in SOURCE_NAMES:
        rows = csv_read(RESULTS/f'{name}.csv')
        if name == 'convergence':
            original_mesh = {operator_key(row): first(row, 'E_mesh', 'E_G_mesh_only')
                             for row in rows if row.get('method') == 'PPPM-original'}
        grid = [row for row in rows if row.get('method') == METHOD]
        source_summaries[name] = dict(total_rows=len(rows), grid_rows=len(grid),
                                      columns=list(rows[0]) if rows else [],
                                      sha256=sha(RESULTS/f'{name}.csv'))
        for index, row in enumerate(rows, 1):
            if row.get('method') != METHOD:
                continue
            par = physical(row)
            opkey = operator_key(row)
            source_id = f'{name}.csv:{index}'
            payload = json.dumps(opkey, separators=(',', ':'))
            identity = 'grid_'+hashlib.sha256(payload.encode()).hexdigest()[:16]
            e_new, e_old, e_proj = first(row, 'E_G'), first(row, 'E_original'), first(row, 'E_proj')
            mesh_new = first(row, 'E_mesh', 'E_G_mesh_only')
            mesh_old = original_mesh.get(opkey) if name == 'convergence' else None
            norm = reference_norms.get((row.get('config'), par['kappa']))
            negative_blocks = first(row, 'negative_blocks')
            entry = dict(source_csv=f'{name}.csv', source_data_row=index, source_row_id=source_id,
                         data_id=identity, table_key=table_key(par), method=METHOD, id=row.get('id'),
                         config=row.get('config') or None, family=row.get('family') or None,
                         seed=number(row.get('seed')), sweep=row.get('sweep') or None,
                         setting=row.get('setting') or None, value=number(row.get('value')),
                         **par, E_old=e_old, E_new=e_new, E_proj=e_proj,
                         E_old_kind='empirical complete operator error against independent numerical reference',
                         E_new_kind='empirical complete operator error against independent numerical reference',
                         E_mesh_old=mesh_old, E_mesh_new=mesh_new,
                         mesh_reference='Gamma_D minus far-short at sR=8' if name == 'convergence' else
                                        ('Gamma_D minus far-short; old PPPM sweep mesh-only uses truncated Fourier instead' if name == 'sweeps' else None),
                         normGammaD=norm, E_proj_abs_empirical=e_proj*norm if e_proj is not None and norm is not None else None,
                         active=negative_blocks > 0 if negative_blocks is not None else None,
                         negative_blocks=negative_blocks,
                         negative_fraction=first(row, 'fraction_negative_blocks', 'neg_fraction'),
                         correction_fro=first(row, 'correction_fro'),
                         min_raw_eig=first(row, 'min_raw_eig', 'min_eig_raw', 'min_eig_before'),
                         min_symmetrized_eig=first(row, 'min_eig_symmetrized', 'min_eig_before'),
                         delta_existing_diagnostic=first(row, 'correction_max_spectral', 'max_negative_part_spectral'),
                         inversion_defect=first(row, 'inversion_residual_before'),
                         measured_setup_s=first(row, 't_setup_s'),
                         final_max_E_new=first(row, 'max_E_G'),
                         final_median_matvec_s=first(row, 'median_matvec_s'),
                         total_bound_certified=None,
                         certified_old_error_available=False)
            catalog.append(entry)
    groups = defaultdict(list)
    for row in catalog:
        groups[row['table_key']].append(row)
    unique_tables = []
    for key, rows in sorted(groups.items()):
        representative = rows[0]
        unique_tables.append(dict(table_key=key, **{field:representative[field] for field in TABLE_FIELDS},
                                   N=representative['N'], rho=representative['rho'],
                                   source_rows=len(rows), n_configs=len({row['config'] for row in rows if row['config']}),
                                   sources=sorted({row['source_csv'] for row in rows}),
                                   active_any=any(row['active'] for row in rows if row['active'] is not None)))
    candidates = [row for row in catalog if row['config'] and row['E_proj'] is not None and row['active']]
    worst = sorted(candidates, key=lambda row:row['E_proj'], reverse=True)[:12]
    # Use an actually present complete nine-configuration source as the main
    # active case. The earlier original-optimum failure exists only in the
    # optional screen file and must not be described as nine-configuration data.
    badcase_parameters = {key:worst[0][key] for key in ('xi','sR','sF','M','p','kappa','rho')} if worst else {}
    badcase = [row for row in catalog if row['config'] and badcase_parameters
               and all(row[key] == value for key,value in badcase_parameters.items())]
    old_optimum_screen = [row for row in csv_read(RESULTS/'search.csv')
                          if row.get('method')==METHOD and row.get('id')=='PPPM-grid-PSD_xi1.25_sR2.8_sF2.8_M16_p5']
    best = json.loads((RESULTS/'best_configuration.json').read_text())[METHOD]
    winners = [row for row in catalog if row['id']==best['id'] and row['config']]
    output = dict(created_at=datetime.now(ZoneInfo('America/Detroit')).isoformat(),
                  snapshot='source_snapshot.json', catalog_csv='source_catalog.csv',
                  required_sources=source_summaries,
                  optional_sources={name:dict(total_rows=len(csv_read(RESULTS/f'{name}.csv')),
                                             sha256=sha(RESULTS/f'{name}.csv')) for name in OPTIONAL_NAMES
                                    if (RESULTS/f'{name}.csv').exists()},
                  catalog_grid_source_rows=len(catalog),
                  unique_table_parameter_keys=len(unique_tables), unique_tables=unique_tables,
                  unique_operator_configuration_ids=len({row['data_id'] for row in catalog if row['config']}),
                  grid_source_rows_with_all_empirical_errors=sum(all(row[k] is not None for k in ('E_old','E_new','E_proj')) for row in catalog),
                  convergence_rows_with_both_mesh_only_errors=sum(row['source_csv']=='convergence.csv' and row['E_mesh_old'] is not None
                                                                 and row['E_mesh_new'] is not None for row in catalog),
                  original_numerical_reference_certificate='Reference tail bound is certified; measured E_old is not a certified total discretization bound.',
                  field_mapping=dict(E_old='E_original (complete operator, empirical)', E_new='E_G (complete operator, empirical)',
                    E_proj='E_proj (dense operator difference, empirical)',
                    E_mesh_new='E_mesh or E_G_mesh_only (Gamma_D minus far-short)',
                    E_mesh_old='convergence.csv PPPM-original same operator_key, E_mesh or E_G_mesh_only',
                    active='negative_blocks > 0, with literal zeros preserved',
                    delta_existing_diagnostic='correction_max_spectral or max_negative_part_spectral; recompute for the new bound',
                    normGammaD='positive_mesh_b2/results/references.csv norm2, same independent direct cache'),
                  typical_badcase_parameters=badcase_parameters,
                  typical_badcase_rows=badcase, worst_active_rows=worst,
                  original_optimum_optional_screen_rows=old_optimum_screen,
                  original_optimum_optional_screen_scope='Only uniform seed2 screen; not a complete nine-configuration validation.',
                  main_winner=best, main_winner_source_rows=winners,
                  snapshot_counts=dict(tracked_files=len(snapshot['tracked_file_sha256']),
                                       independent_references=len(snapshot['independent_references']),
                                       reference_sidecars=len(snapshot['independent_references'])))
    csv_write(HERE/'source_catalog.csv', catalog)
    json_write(HERE/'source_catalog.json', output)
    print(json.dumps(dict(tracked_files=len(snapshot['tracked_file_sha256']), refs=len(snapshot['independent_references']),
                          source_grid_rows=len(catalog), unique_grid_tables=len(unique_tables),
                          unique_grid_operator_configs=output['unique_operator_configuration_ids'],
                          source_counts={name:value['grid_rows'] for name,value in source_summaries.items()},
                          worst_active_source=worst[0]['source_row_id'] if worst else None,
                          worst_active_E_proj=worst[0]['E_proj'] if worst else None,
                          winner=best['id']), ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
