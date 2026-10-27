"""Checkpointed bounds validation using immutable PR #4 measurements.

Only correction degrees and correction-only graphs are newly computed. Historical
complete-operator errors retain their source row IDs, including literal roundoff.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import study as S
from bounds import CorrectionBounds
np=S.np
RESULTS=HERE/'results'
PARTS=RESULTS/'parts'
DENSE_OFFSET_LIMIT=5000
ROUND_OFF_CLASSIFICATION=float(512*np.finfo(float).eps)


def json_scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError('Unsupported JSON value: '+type(value).__name__)


def save(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=json_scalar)+'\n')
    tmp.replace(path)


def write(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)


def read(path):
    with Path(path).open(newline='') as f: rows=list(csv.DictReader(f))
    for row in rows:
        for k,v in row.items():
            if v=='': row[k]=None
            elif v in ('True','False'):row[k]=v=='True'
            else:
                try:row[k]=float(v)
                except ValueError:pass
    return rows


def pid(table_key):
    return 'table_'+hashlib.sha256(table_key.encode()).hexdigest()[:16]


def parameters(row):
    return S.parameters(**{k:row[k] for k in ('xi','sR','sF','M','p','kappa','rho')})


def context():
    rows=[r for r in read(HERE/'source_catalog.csv') if r['config']]
    meta=json.loads((HERE/'source_catalog.json').read_text())
    groups={}
    for r in rows:groups.setdefault(r['table_key'],[]).append(r)
    selected={r['data_id']:'winner' for r in meta['main_winner_source_rows']}
    selected.update({r['data_id']:'active-badcase' for r in meta['typical_badcase_rows']})
    return rows,groups,selected


def run_table(job):
    table_key,rows,selected=job
    identifier=pid(table_key); destination=PARTS/(identifier+'.json')
    source_digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    if destination.exists():
        previous=json.loads(destination.read_text())
        if previous['source_digest']!=source_digest:raise RuntimeError('Changed checkpoint input: '+table_key)
        return identifier,'cached',len(previous['configurations'])
    start=time.perf_counter()
    par=parameters(rows[0]); mesh=S.table('PPPM-grid-PSD',rows[0]['L'],par)
    cb=CorrectionBounds(mesh,audit=True)
    # The same correction and degree are independent of short cutoff sR.
    configurations={}
    for row in rows:configurations.setdefault(row['config'],row)
    results={}
    for name,row in configurations.items():
        cfg=S.E.config(row['family'],int(row['seed']),row['rho'])
        assert cfg['name']==name and abs(cfg['L']-row['L'])<1e-12
        r,D=cb.bounds(cfg['q'])
        nref=row['normGammaD']
        r.update(config=name,parameter_id=identifier,h=mesh.h,
                 negative_fraction=r['fraction_negative_blocks'],
                 symmetrization_defect=r['raw_inversion_max_component_defect'],
                 correction_norm=None,correction_norm_source='not_computed',
                 tightness=None,exact_zero=not r['active'],
                 norm_comparison_residual=None,operator_comparison_residual=None,
                 correction_degree_dense_residual=None,correction_action_dense_residual=None,
                 correction_null_residual=None,correction_graph_min_eigenvalue=None)
        if not r['active']:
            r.update(correction_norm=0.,correction_norm_source='literal_zero_correction_table',
                     norm_comparison_residual=abs(row['E_proj']*nref))
        elif len(cb.active_offsets)<=DENSE_OFFSET_LIMIT:
            td=time.perf_counter(); W=cb.pair_matrix(cfg['q']); GC=S.lap_full(W)
            r['t_dense_correction_check_s']=time.perf_counter()-td
            lo=float(S.linalg.eigvalsh(GC,subset_by_index=[0,0])[0])
            hi=float(S.linalg.eigvalsh(GC,subset_by_index=[len(GC)-1,len(GC)-1])[0])
            cn=max(abs(lo),abs(hi))
            rng=np.random.default_rng(1000+int(row['seed']))
            probes=rng.standard_normal((3*cfg['N'],2))
            action=np.column_stack([cb.matvec(cfg['q'],v,D) for v in probes.T])
            T=np.tile(np.eye(3),(cfg['N'],1))/np.sqrt(cfg['N'])
            r.update(correction_norm=cn,correction_norm_source='correction_only_dense_all_offsets',
                     tightness=r['B']/cn if cn>0 else None,
                     norm_comparison_residual=abs(cn-row['E_proj']*nref),
                     correction_degree_dense_residual=float(np.max(np.abs(D-W.sum(axis=1)))),
                     correction_action_dense_residual=float(np.linalg.norm(action-GC@probes)/nref/np.linalg.norm(probes)),
                     correction_null_residual=float(np.linalg.norm(GC@T)/nref),
                     correction_graph_min_eigenvalue=lo)
            # Selected winner and bad case: compare the FULL historical operator
            # difference with C-only, rather than just comparing their norms.
            if any(selected.get(z['data_id']) for z in rows if z['config']==name):
                old=S.long_dense('PPPM-original',mesh.original,cfg['q'])
                new=S.long_dense('PPPM-grid-PSD',mesh,cfg['q'])
                r['operator_comparison_residual']=S.norm(new-old-GC)/nref
                r['full_difference_norm_recheck']=S.norm(new-old)/nref
            del W,GC
        results[name]=r
    payload=dict(parameter_id=identifier,table_key=table_key,source_digest=source_digest,
                 elapsed_s=time.perf_counter()-start,configurations=results)
    save(destination,payload)
    return identifier,'computed',len(results)


def merge():
    rows,groups,selected=context(); output=[]
    for table_key,source_rows in groups.items():
        part=PARTS/(pid(table_key)+'.json')
        if not part.exists():raise FileNotFoundError('Missing checkpoint '+str(part))
        data=json.loads(part.read_text())['configurations']
        for source in source_rows:
            row=dict(source); r=data[source['config']]
            row.update(r,dataset=Path(source['source_csv']).stem,case_id=source['data_id'],
                       selected_label=selected.get(source['data_id']),
                       source_sweep=source['sweep'],source_setting=source['setting'],source_value=source['value'])
            nref=source['normGammaD']
            for key in ('B_simple','B_degree','B'):row[key+'_rel']=r[key]/nref
            row['E_correction']=r['correction_norm']/nref if r['correction_norm'] is not None else None
            row['historical_operator_difference_norm']=source['E_proj']*nref
            row['norm_comparison_residual']=(abs(r['correction_norm']-source['E_proj']*nref)
                                              if r['correction_norm'] is not None else None)
            # Do NOT replace old E_proj by zero, clip it or inflate the bound.
            row['violation_abs']=max(source['E_proj']-row['B_rel'],0.)
            row['strict_float_violation']=row['violation_abs']>0
            row['roundoff_violation']=0<row['violation_abs']<=ROUND_OFF_CLASSIFICATION
            row['material_violation']=row['violation_abs']>ROUND_OFF_CLASSIFICATION
            row['correction_bound_violation_abs']=(max(r['correction_norm']-r['B'],0.)
                                                   if r['correction_norm'] is not None else None)
            row['empirical_envelope']=source['E_old']+row['B_rel']
            row['empirical_envelope_violation']=max(source['E_new']-row['empirical_envelope'],0.)
            row['certified_total_bound']=None
            output.append(row)
    write(RESULTS/'bounds.csv',output)
    unique={row['case_id']:row for row in output}
    exact=[r for r in unique.values() if r['correction_norm_source']=='correction_only_dense_all_offsets']
    inactive=[r for r in unique.values() if r['exact_zero']]
    uncertain=[r for r in unique.values() if r['correction_norm'] is None]
    physical={(r['parameter_id'],r['config']):r for r in output}
    selected_rows=list({r['case_id']:r for r in output if r['selected_label']}.values())
    write(RESULTS/'cases_summary.csv',selected_rows)
    if (RESULTS/'timing_raw.csv').exists():attach_timing_summaries()
    summary=dict(source_rows=len(output),unique_cases=len(unique),unique_tables=len(groups),
        computed_table_configurations=len(physical),
        computed_active_table_configurations=sum(r['active'] for r in physical.values()),
        computed_dense_active_table_configurations=sum(r['correction_norm_source']=='correction_only_dense_all_offsets' for r in physical.values()),
        computed_literal_zero_table_configurations=sum(r['exact_zero'] for r in physical.values()),
        datasets={key:sum(r['dataset']==key for r in output) for key in sorted({r['dataset'] for r in output})},
        unique_active_cases=sum(r['active'] for r in unique.values()),literal_zero_cases=len(inactive),
        dense_correction_checked_unique_cases=len(exact),dense_offset_limit=DENSE_OFFSET_LIMIT,
        active_cases_using_historical_full_difference_only=len(uncertain),
        strict_float_E_proj_violations=sum(r['strict_float_violation'] for r in output),
        roundoff_E_proj_violations=sum(r['roundoff_violation'] for r in output),
        material_E_proj_violations=sum(r['material_violation'] for r in output),
        max_E_proj_violation=max(r['violation_abs'] for r in output),
        roundoff_classification_relative_scale=ROUND_OFF_CLASSIFICATION,
        min_dense_tightness=min(r['tightness'] for r in exact),
        median_dense_tightness=float(np.median([r['tightness'] for r in exact])),
        max_dense_tightness=max(r['tightness'] for r in exact),
        max_correction_bound_violation=max(r['correction_bound_violation_abs'] or 0 for r in output),
        max_norm_comparison_residual_relative=max((r['norm_comparison_residual'] or 0)/r['normGammaD'] for r in output),
        max_operator_comparison_residual_relative=max(r['operator_comparison_residual'] or 0 for r in output),
        max_degree_dense_residual=max(r['correction_degree_dense_residual'] or 0 for r in output),
        max_action_dense_residual_relative=max(r['correction_action_dense_residual'] or 0 for r in output),
        max_null_residual_relative=max(r['correction_null_residual'] or 0 for r in output),
        empirical_envelope_material_violations=sum(r['empirical_envelope_violation']>ROUND_OFF_CLASSIFICATION for r in output),
        certified_total_bound_available=False,
        source_E_old_kind='empirical against the existing independent numerical direct reference',
        correction_norm_policy='literal zero or exact all-offset correction-only dense; uncomputed wide-table norms stay blank',
        recorded_at=datetime.now(timezone.utc).isoformat())
    save(RESULTS/'verification_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def timings(repeats=9):
    """A NEW isolated serial batch for incremental bound setup, no matvec timing."""
    if (RESULTS/'timing_raw.csv').exists():
        previous=json.loads((RESULTS/'timing_context.json').read_text())
        if previous['repeats']!=repeats:
            raise FileExistsError('Preserve/rename the existing timing batch before requesting a different repeat count')
        attach_timing_summaries()
        print('Reused preserved timing_raw.csv; no raw timing samples overwritten',flush=True)
        return
    rows,groups,selected=context()
    selected_tables={r['table_key'] for r in rows if selected.get(r['data_id'])}
    raw=[]
    # Alternating table order avoids assigning every early sample to one case.
    for repeat in range(repeats+1):
        keys=sorted(selected_tables,reverse=bool(repeat%2))
        for key in keys:
            source=[r for r in groups[key] if selected.get(r['data_id'])]
            source=list({r['config']:r for r in source}.values())
            par=parameters(source[0]); mesh=S.table('PPPM-grid-PSD',source[0]['L'],par)
            cb=CorrectionBounds(mesh,audit=False)
            for row in source:
                cfg=S.E.config(row['family'],int(row['seed']),row['rho'])
                result,_=cb.bounds(cfg['q'])
                if repeat==0:continue
                fields=dict(repeat=repeat,parameter_id=pid(key),case_id=row['data_id'],config=row['config'],
                    family=row['family'],seed=row['seed'],selected_label=selected[row['data_id']],
                    M=row['M'],p=row['p'],xi=row['xi'],sF=row['sF'],rho=row['rho'],kappa=row['kappa'],
                    threads=1,audit=False,batch='isolated_incremental_bound_setup')
                for phase,keytime in [('bound_parameter_setup','t_correction_parameter_setup_s'),
                                      ('correction_fft','t_correction_fft_setup_s'),
                                      ('bound_configuration_setup','t_bound_configuration_setup_s'),
                                      ('stencil','t_bound_stencil_s'),
                                      ('density_fft','t_bound_density_fft_s'),
                                      ('degree_convolutions','t_bound_degree_convolutions_s'),
                                      ('degree_eigh','t_bound_degree_eigh_s')]:
                    raw.append(dict(fields,phase=phase,seconds=result[keytime],
                                    parameter_sample_shared_across_configs=phase in ('bound_parameter_setup','correction_fft')))
    write(RESULTS/'timing_raw.csv',raw)
    timing_context=dict(repeats=repeats,warmups=1,threads=S.threadpool_info(),
        load_average=list(os.getloadavg()),platform=sys.platform,
        policy='Incremental correction-bound setup only; parameter sample is shared, not 9 independent FFT samples. No cached matvec re-timing or old-batch speed ratio.')
    save(RESULTS/'timing_context.json',timing_context)
    attach_timing_summaries()
    print('Saved isolated timing samples: '+str(len(raw)),flush=True)


def attach_timing_summaries():
    """Refresh derived medians without repeating/overwriting raw measurements."""
    raw=read(RESULTS/'timing_raw.csv')
    cases=read(RESULTS/'cases_summary.csv')
    for row in cases:
        for phase in ('bound_parameter_setup','correction_fft','bound_configuration_setup'):
            values=[r['seconds'] for r in raw if r['case_id']==row['case_id'] and r['phase']==phase]
            row['median_'+phase+'_s']=float(np.median(values))
            row['q25_'+phase+'_s']=float(np.quantile(values,.25))
            row['q75_'+phase+'_s']=float(np.quantile(values,.75))
    write(RESULTS/'cases_summary.csv',cases)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('compute','merge','timing','all'))
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=9)
    args=parser.parse_args()
    if args.stage in ('compute','all'):
        rows,groups,selected=context(); PARTS.mkdir(parents=True,exist_ok=True)
        # Small tables first: early complete checkpoints, bounded 8GB host use.
        jobs=sorted(((k,v,selected) for k,v in groups.items()),key=lambda j:j[1][0]['M'])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_table,j) for j in jobs]
            for i,future in enumerate(as_completed(futures),1):
                print(f'{i}/{len(futures)} {future.result()}',flush=True)
    if args.stage in ('merge','all'):merge()
    if args.stage in ('timing','all'):timings(args.repeats)


if __name__=='__main__':main()
