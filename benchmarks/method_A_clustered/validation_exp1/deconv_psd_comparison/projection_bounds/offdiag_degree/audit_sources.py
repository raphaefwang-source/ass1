"""Freeze PR #5 inputs for the self-excluded degree follow-up, read-only.

Only this new offdiag_degree directory is written. Repeated runs preserve the
first snapshot and reject changes to old files, references, CSVs or part caches.
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
STUDY = PREVIOUS.parent
REPO = HERE.parents[5]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError,ValueError):
        return None


def truth(value):
    return value is True or str(value).lower() in ('true','1')


def csv_read(path):
    with path.open(newline='',encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def save_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def git(*args):
    return subprocess.run(['git',*args],cwd=REPO,check=True,capture_output=True,text=True).stdout


def snapshot():
    excluded=HERE.relative_to(REPO)
    tracked=[name for name in git('ls-files','-z').split('\0') if name
             and not Path(name).is_relative_to(excluded)]
    hashes={name:digest(REPO/name) for name in tracked}
    if len(hashes)!=861:
        raise RuntimeError(f'Expected 861 prior tracked files, found {len(hashes)}')
    old=json.loads((PREVIOUS/'source_snapshot.json').read_text())
    references=[]
    for row in old['independent_references']:
        for filekey,hashkey in (('path','sha256'),('metadata_path','metadata_sha256')):
            if digest(REPO/row[filekey])!=row[hashkey]:
                raise RuntimeError(f'Independent reference changed since PR #5: {row[filekey]}')
        references.append(dict(row))
    if len(references)!=72:
        raise RuntimeError(f'Expected 72 independent reference/metadata pairs, found {len(references)}')
    csv_paths=[PREVIOUS/'results'/f'{name}.csv' for name in
               ('bounds','cases_summary','timing_raw')]
    csv_paths += [PREVIOUS/'source_catalog.csv']
    csv_paths += [STUDY/'results'/f'{name}.csv' for name in
                  ('convergence','validation','sweeps','xi_sweep','best_projection_diagnostics','final_timing')]
    parts=sorted((PREVIOUS/'results/parts').glob('*.json'))
    if len(parts)!=141:
        raise RuntimeError(f'Expected 141 reusable PR #5 table parts, found {len(parts)}')
    data=dict(created_at=datetime.now(ZoneInfo('America/Detroit')).isoformat(),
              git=dict(head=git('rev-parse','HEAD').strip(),branch=git('branch','--show-current').strip(),
                       status=git('status','--porcelain').strip()),
              excluded_new_directory=str(excluded),tracked_file_sha256=hashes,
              independent_references=references,
              source_csv_sha256={str(path.relative_to(REPO)):digest(path) for path in csv_paths},
              reused_part_sha256={str(path.relative_to(REPO)):digest(path) for path in parts},
              note='Read-only PR #5 sources, all old methods/results/figures/cached timings and direct references are preserved.')
    path=HERE/'source_snapshot.json'
    if path.exists():
        initial=json.loads(path.read_text())
        for key in ('tracked_file_sha256','independent_references','source_csv_sha256','reused_part_sha256'):
            if initial[key]!=data[key]:
                raise RuntimeError(f'Previous inputs changed since first offdiag snapshot: {key}')
        return initial
    save_json(path,data)
    return data


def main():
    initial=snapshot()
    bounds=csv_read(PREVIOUS/'results/bounds.csv')
    if len(bounds)!=1692:
        raise RuntimeError(f'Expected all 1692 PR #5 bound rows, found {len(bounds)}')
    fields=('source_row_id','source_csv','source_data_row','dataset','case_id','data_id','parameter_id','table_key',
            'config','family','seed','id','N','L','xi','sR','sF','sT','M','p','rho','kappa',
            'E_old','E_new','E_proj','E_mesh_old','E_mesh_new','normGammaD','active','delta',
            'negative_blocks','active_correction_offsets','correction_fro','min_raw_eig',
            'B_simple','B_degree','B','correction_norm','correction_norm_source','tightness','exact_zero')
    catalog=[]
    for index,row in enumerate(bounds,1):
        item=dict(previous_bounds_data_row=index,**{field:row.get(field,'') for field in fields})
        item['B_old']=row['B']
        item['B_total']=row['B_degree']
        item['original_errors_kind']='empirical against independent numerical reference; not certified epsilon_old'
        catalog.append(item)
    focus=[]
    predicates=(
      ('convergence',lambda r:r['dataset']=='convergence'),
      ('winner',lambda r:r['dataset']=='best_projection_diagnostics'),
      ('active-badcase',lambda r:r['dataset']=='sweeps' and number(r['kappa'])==.7 and number(r['xi'])==.85
        and number(r['sR'])==4 and number(r['sF'])==4 and number(r['M'])==16 and number(r['p'])==7 and number(r['rho'])==.75),
      ('self-dominated-outlier',lambda r:r['dataset']=='convergence' and number(r['kappa'])==.7 and number(r['xi'])==.85
        and number(r['sR'])==8 and number(r['sF'])==6 and number(r['M'])==64 and number(r['p'])==9 and number(r['rho'])==.75))
    focus_summary={}
    for label,predicate in predicates:
        chosen=[dict(row,focus_group=label) for row in catalog if predicate(row)]
        if len(chosen)!=(324 if label=='convergence' else 9):
            raise RuntimeError(f'Unexpected mandatory focus count {label}: {len(chosen)}')
        focus.extend(chosen)
        focus_summary[label]=dict(rows=len(chosen),unique_cases=len({r['case_id'] for r in chosen}),
                                  parameter_ids=sorted({r['parameter_id'] for r in chosen}))
    parts={}
    for path in sorted((PREVIOUS/'results/parts').glob('*.json')):
        data=json.loads(path.read_text())
        parts[data['parameter_id']]=dict(table_key=data['table_key'],configurations=len(data['configurations']),
          known_true_norm_configurations=sum(v.get('correction_norm') is not None for v in data['configurations'].values()),
          active_true_dense_configurations=sum(bool(v.get('active')) and v.get('correction_norm') is not None
                                               for v in data['configurations'].values()),
          unavailable_wide_configurations=sum(bool(v.get('active')) and v.get('correction_norm') is None
                                              for v in data['configurations'].values()))
    counts=dict(source_rows=len(bounds),unique_case_ids=len({r['case_id'] for r in bounds}),
                unique_parameter_ids=len({r['parameter_id'] for r in bounds}),
                unique_parameter_configuration_pairs=len({(r['parameter_id'],r['config']) for r in bounds}),
                datasets=dict(Counter(r['dataset'] for r in bounds)),
                true_norm_unavailable_source_rows=sum(number(r['correction_norm']) is None for r in bounds),
                true_norm_unavailable_case_ids=len({r['case_id'] for r in bounds if number(r['correction_norm']) is None}),
                active_dense_case_ids=len({r['case_id'] for r in bounds if truth(r['active']) and number(r['correction_norm']) is not None}),
                active_dense_parameter_configuration_pairs=sum(r['active_true_dense_configurations'] for r in parts.values()))
    output=dict(created_at=datetime.now(ZoneInfo('America/Detroit')).isoformat(),
                snapshot='source_snapshot.json',source_csv='source_catalog.csv',focus_csv='focus_catalog.csv',
                preservation=dict(tracked_files=len(initial['tracked_file_sha256']),
                                  independent_references=len(initial['independent_references']),
                                  sidecars=len(initial['independent_references']),parts=len(parts)),
                source_counts=counts,focus_groups=focus_summary,reused_parts=parts,
                field_mapping=dict(B_old='PR #5 B',B_total='PR #5 B_degree',B_simple='PR #5 N*delta',
                  correction_norm='PR #5 true correction-only norm, literal zero or all-offset dense; unavailable wide rows stay blank',
                  E_old='PR #5 E_old, unchanged empirical complete operator error',
                  E_new='PR #5 E_new, unchanged empirical complete operator error',
                  E_proj='PR #5 E_proj, unchanged historical full-operator difference; never a true correction norm proxy'),
                policy='Compute new self/offdiagonal degrees per table/configuration while preserving all 1692 source rows. '
                       'The 9 outlier rows are a subset of the 324 convergence rows; focus CSV repeats their label intentionally. '
                       'No certified total discretization error exists without a separately certified original error.')
    save_csv(HERE/'source_catalog.csv',catalog)
    save_csv(HERE/'focus_catalog.csv',focus)
    save_json(HERE/'source_catalog.json',output)
    print(json.dumps(dict(preservation=output['preservation'],source_counts=counts,
                          focus_counts={label:value['rows'] for label,value in focus_summary.items()}),indent=2))


if __name__=='__main__':
    main()
