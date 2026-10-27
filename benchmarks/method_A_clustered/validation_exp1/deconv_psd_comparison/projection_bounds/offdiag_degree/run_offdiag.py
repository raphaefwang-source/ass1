"""Resumable self-excluded correction-degree study; all PR #5 inputs read-only."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run_bounds as PREV
from bounds import CorrectionBounds
from offdiag_bounds import OffdiagBounds
S=PREV.S;np=S.np
read=PREV.read;write=PREV.write;save=PREV.save
RESULTS=HERE/'results';PARTS=RESULTS/'parts'
ROUND_OFF_CLASSIFICATION=float(512*np.finfo(float).eps)


def spectral_max(A):
    eig=np.linalg.eigvalsh(A)
    return float(np.abs(eig).max())


def context():
    rows=read(HERE/'source_catalog.csv');groups={}
    for row in rows:groups.setdefault(row['parameter_id'],[]).append(row)
    focus=read(HERE/'focus_catalog.csv')
    selected={r['case_id']:r['focus_group'] for r in focus if r['focus_group']!='convergence'}
    full_audit={(r['parameter_id'],r['config']) for r in focus}
    return rows,groups,selected,full_audit


def parameters(row):
    return S.parameters(**{k:row[k] for k in ('xi','sR','sF','M','p','kappa','rho')})


def direct_audit(cb,ob,q,Doff,norm_correction,prefix):
    """Independent all-offset pair matrix; zero self BEFORE the offdegree sum."""
    W=cb.pair_matrix(q);index=np.arange(len(q))
    Cself=W[index,index].copy();Dtotal=W.sum(axis=1)
    W[index,index]=0.
    Ddirect=W.sum(axis=1)
    raw_error=spectral_max(ob.last_D_off_raw-Ddirect)
    final_error=spectral_max(Doff-Ddirect)
    eig=np.linalg.eigvalsh(Ddirect)
    result={prefix+'_N':len(q),prefix+'_checked':True,
        prefix+'_self_error_max_spectral':spectral_max(ob.last_C_self-Cself),
        prefix+'_total_error_max_spectral':spectral_max(ob.last_D_total-Dtotal),
        prefix+'_raw_off_error_max_spectral':raw_error,
        prefix+'_off_error_max_spectral':final_error,
        prefix+'_direct_off_min_eigenvalue':float(eig[:,0].min()),
        prefix+'_B_off_direct':float(2*eig[:,-1].max()),
        prefix+'_off_symmetry_residual':float(np.abs(Doff-Doff.transpose(0,2,1)).max()),
        prefix+'_raw_error_relative_true_correction_norm':raw_error/norm_correction if norm_correction and norm_correction>0 else None,
        prefix+'_error_relative_true_correction_norm':final_error/norm_correction if norm_correction and norm_correction>0 else None}
    return result,W


def run_table(job):
    parameter_id,source,selected,full_audit=job
    target=PARTS/(parameter_id+'.json')
    digest=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    algorithm_sha=hashlib.sha256((HERE/'offdiag_bounds.py').read_bytes()).hexdigest()
    if target.exists():
        old=json.loads(target.read_text())
        if old['source_digest']!=digest or old['algorithm_sha256']!=algorithm_sha:
            raise RuntimeError('Checkpoint input/implementation changed; preserve prior batch before recomputing '+parameter_id)
        return parameter_id,'cached',len(old['configurations'])
    start=time.perf_counter();par=parameters(source[0])
    mesh=S.table('PPPM-grid-PSD',source[0]['L'],par)
    cb=CorrectionBounds(mesh,audit=False);ob=OffdiagBounds(cb)
    previous=json.loads((HERE.parent/'results/parts'/(parameter_id+'.json')).read_text())['configurations']
    configurations={r['config']:r for r in source};output={}
    for config,row in configurations.items():
        cfg=S.E.config(row['family'],int(row['seed']),row['rho']);q=cfg['q']
        t0=time.perf_counter();r,Doff=ob.bounds(q,stable='auto')
        r.update(t_full_bound_configuration_setup_s=time.perf_counter()-t0,
                 parameter_id=parameter_id,config=config,h=mesh.h,
                 correction_norm=previous[config]['correction_norm'],
                 correction_norm_source=previous[config]['correction_norm_source'],
                 correction_norm_provenance='PR5 results/parts/'+parameter_id+'.json',
                 audit_full_checked=False,audit_literal_checked=False)
        cn=r['correction_norm']
        if (parameter_id,config) in full_audit:
            # Active focus cases are independently assembled in full N=256.
            # Literal-zero tables need no dense construction to check their zero.
            if r.get('active',cb.diagnostics['active']):
                a,_=direct_audit(cb,ob,q,Doff,cn,'audit_full');r.update(a)
            else:
                r.update(audit_full_checked=True,audit_full_N=len(q),audit_full_literal_zero=True,
                         audit_full_raw_off_error_max_spectral=0.,audit_full_off_error_max_spectral=0.,
                         audit_full_B_off_direct=0.,audit_full_direct_off_min_eigenvalue=0.)
        # Every physical table/configuration gets an independent small-N check.
        picks=np.random.default_rng(1973+int(row['seed'])).choice(len(q),4,replace=False)
        small=q[picks];rs,Ds=ob.bounds(small,stable='auto')
        if cb.diagnostics['active']:
            Ws=cb.pair_matrix(small);iself=np.arange(4);Ws[iself,iself]=0.
            norm_small=S.norm(S.lap_full(Ws))
            a,_=direct_audit(cb,ob,small,Ds,norm_small,'audit_small');r.update(a)
            r.update(audit_small_true_correction_norm=norm_small,
                     audit_small_true_exact_zero=norm_small==0.,
                     audit_small_raw_unreliable=rs.get('cancellation_unreliable_raw'),
                     audit_small_stable_fallback_used=rs.get('stable_fallback_used'))
            if row['family']=='uniform' and int(row['seed'])==1:
                # Literal p^6 contraction is confined to two-particle offline
                # validation, never the production self/bound calculation.
                literal=cb.direct_pair_matrix(small[:2]);fast=cb.pair_matrix(small[:2])
                r.update(audit_literal_checked=True,audit_literal_N=2,
                         audit_literal_pair_max_component_residual=float(np.abs(literal-fast).max()))
        else:
            r.update(audit_small_checked=True,audit_small_N=4,audit_small_true_exact_zero=True,
                audit_small_true_correction_norm=0.,audit_small_raw_off_error_max_spectral=0.,
                audit_small_off_error_max_spectral=0.,audit_small_B_off_direct=0.,
                audit_small_direct_off_min_eigenvalue=0.,audit_small_literal_zero=True)
        output[config]=r
    save(target,dict(parameter_id=parameter_id,source_digest=digest,algorithm_sha256=algorithm_sha,
                     elapsed_s=time.perf_counter()-start,configurations=output))
    return parameter_id,'computed',len(output)


def merge():
    rows,groups,selected,focus=context();merged=[]
    for pid,source in groups.items():
        data=json.loads((PARTS/(pid+'.json')).read_text())['configurations']
        for old in source:
            row=dict(old);r=data[old['config']]
            row.update(r,selected_label=selected.get(old['case_id']),
                       B_old_previous=old['B_old'],B_total_previous=old['B_total'],
                       t_legacy_matvec_s=None,certified_total_bound=None,
                       theory_certified_candidate=None)
            nref=old['normGammaD']
            for key in ('B_simple','B_total','B_off','B_old','B_improved','B_off_raw','B_improved_raw'):
                row[key+'_rel']=row[key]/nref if row.get(key) is not None else None
            row['B_total_source_residual_abs']=abs(r['B_total']-old['B_total'])
            row['B_old_source_residual_abs']=abs(r['B_old']-old['B_old'])
            row['degree_off_total_ratio']=r.get('degreeoff_total_ratio')
            # Small-N audits also use auto mode. For N=4, an unreliable raw
            # subtraction without a successful fallback is explicitly unavailable.
            row['audit_small_bound_numerical_available']=not (
                bool(r.get('audit_small_raw_unreliable')) and not bool(r.get('audit_small_stable_fallback_used')))
            row['audit_small_stable_fallback_unavailable']=not row['audit_small_bound_numerical_available']
            cn=row['correction_norm']
            row['E_correction']=cn/nref if cn is not None else None
            row['old_tightness']=row['B_old']/cn if cn is not None and cn>0 else None
            row['improved_tightness']=row['B_improved']/cn if cn is not None and cn>0 and row.get('B_improved') is not None else None
            row['empirical_envelope_old']=old['E_old']+row['B_old_rel']
            row['empirical_envelope_new']=old['E_old']+row['B_improved_rel'] if row.get('B_improved_rel') is not None else None
            row['empirical_old_budget_pass']=row['empirical_envelope_old']<=1e-4
            row['empirical_new_budget_pass']=row['empirical_envelope_new']<=1e-4 if row['empirical_envelope_new'] is not None else None
            row['independent_reference_empirical_pass']=old['E_new']<=1e-4
            row['true_norm_violation_abs']=max(cn-row['B_improved'],0.) if cn is not None and row.get('B_improved') is not None else None
            row['old_bound_monotonicity_violation_abs']=max(row['B_improved']-row['B_old'],0.) if row.get('B_improved') is not None else None
            row['historical_Eproj_violation_rel']=max(old['E_proj']-row['B_improved_rel'],0.) if row.get('B_improved_rel') is not None else None
            row['historical_Eproj_material_violation']=(row['historical_Eproj_violation_rel']>ROUND_OFF_CLASSIFICATION
                                                       if row['historical_Eproj_violation_rel'] is not None else None)
            merged.append(row)
    write(RESULTS/'bounds.csv',merged)
    physical={(r['parameter_id'],r['config']):r for r in merged}
    cases={r['case_id']:r for r in merged}
    selected_cases=list({r['case_id']:r for r in merged if r['selected_label']}.values())
    write(RESULTS/'cases_summary.csv',selected_cases)
    if (RESULTS/'timing_raw.csv').exists():attach_timing_summaries()
    def maxfield(field):return max((r.get(field) or 0) for r in physical.values())
    summary=dict(source_rows=len(merged),unique_case_ids=len(cases),unique_parameter_ids=len(groups),
        computed_table_configurations=len(physical),
        active_table_configurations=sum(bool(r['active']) for r in physical.values()),
        reused_active_dense_table_configurations=sum(r['correction_norm_source']=='correction_only_dense_all_offsets' for r in physical.values()),
        true_norm_unavailable_cases=sum(r['correction_norm'] is None for r in cases.values()),
        small_N_checked_table_configurations=sum(bool(r['audit_small_checked']) for r in physical.values()),
        full_N_checked_table_configurations=sum(bool(r['audit_full_checked']) for r in physical.values()),
        literal_N2_checked_tables=sum(bool(r['audit_literal_checked']) for r in physical.values()),
        stable_fallback_used_table_configurations=sum(bool(r.get('stable_fallback_used')) for r in physical.values()),
        cancellation_unreliable_raw_table_configurations=sum(bool(r.get('cancellation_unreliable_raw')) for r in physical.values()),
        small_N_unavailable_bounds=sum(bool(r['audit_small_stable_fallback_unavailable']) for r in physical.values()),
        unavailable_improved_bounds=sum(r.get('B_improved') is None for r in physical.values()),
        max_true_norm_violation_abs=maxfield('true_norm_violation_abs'),
        max_old_bound_monotonicity_violation_abs=maxfield('old_bound_monotonicity_violation_abs'),
        max_B_total_source_residual_abs=maxfield('B_total_source_residual_abs'),
        max_full_raw_off_error_relative_true_norm=maxfield('audit_full_raw_error_relative_true_correction_norm'),
        max_full_off_error_relative_true_norm=maxfield('audit_full_error_relative_true_correction_norm'),
        max_small_raw_off_error_relative_true_norm=maxfield('audit_small_raw_error_relative_true_correction_norm'),
        max_small_off_error_relative_true_norm=maxfield('audit_small_error_relative_true_correction_norm'),
        max_small_off_error_relative_true_norm_available=max(
            (r.get('audit_small_error_relative_true_correction_norm') or 0)
            for r in physical.values() if r['audit_small_bound_numerical_available']),
        small_N_error_policy='All-case maxima retain the raw diagnostics of unavailable auto bounds; available-only precision and extra-cost forced-stable audits are reported separately.',
        max_literal_pair_component_residual=maxfield('audit_literal_pair_max_component_residual'),
        historical_Eproj_material_violations=sum(bool(r['historical_Eproj_material_violation']) for r in merged),
        max_historical_Eproj_violation_relative=maxfield('historical_Eproj_violation_rel'),
        roundoff_classification_relative_scale=ROUND_OFF_CLASSIFICATION,
        certified_total_bound_available=False,
        policy='Sufficient/conservative correction bound; empirical-pass candidates are not rejected merely for envelope failure. No certified epsilon_old is available.',
        recorded_at=datetime.now(timezone.utc).isoformat())
    save(RESULTS/'verification_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=PREV.json_scalar),flush=True)


def attach_timing_summaries():
    raw=read(RESULTS/'timing_raw.csv');cases=read(RESULTS/'cases_summary.csv')
    for row in cases:
        for phase in ('correction_parameter_setup','offdiag_parameter_setup','bound_configuration_setup',
                      'total_degree','self_autocorrelation','stable_fallback','offdiag_incremental'):
            values=[r['seconds'] for r in raw if r['case_id']==row['case_id'] and r['phase']==phase]
            if values:
                for q,name in ((.5,'median'),(.25,'q25'),(.75,'q75')):
                    row[name+'_'+phase+'_s']=float(np.quantile(values,q))
    write(RESULTS/'cases_summary.csv',cases)


def timings(repeats=9):
    if (RESULTS/'timing_raw.csv').exists():
        old=json.loads((RESULTS/'timing_context.json').read_text())
        if old['repeats']!=repeats:raise FileExistsError('Preserve the existing timing batch before changing repeats')
        attach_timing_summaries();print('Reused preserved raw timing',flush=True);return
    rows,groups,selected,_=context();raw=[]
    tables={r['parameter_id'] for r in rows if selected.get(r['case_id'])}
    for repeat in range(repeats+1):
        for pid in sorted(tables,reverse=bool(repeat%2)):
            source=list({r['config']:r for r in groups[pid] if selected.get(r['case_id'])}.values())
            mesh=S.table('PPPM-grid-PSD',source[0]['L'],parameters(source[0]))
            t0=time.perf_counter();cb=CorrectionBounds(mesh,audit=False);t_cb=time.perf_counter()-t0
            t0=time.perf_counter();ob=OffdiagBounds(cb);t_ob=time.perf_counter()-t0
            for row in source:
                cfg=S.E.config(row['family'],int(row['seed']),row['rho'])
                t0=time.perf_counter();r,_=ob.bounds(cfg['q'],stable='auto');elapsed=time.perf_counter()-t0
                if repeat==0:continue
                metadata=dict(parameter_id=pid,case_id=row['case_id'],config=row['config'],
                    family=row['family'],seed=row['seed'],selected_label=selected[row['case_id']],
                    repeat=repeat,threads=1,audit=False,batch='isolated_offdiag_bound_setup',
                    M=row['M'],p=row['p'],xi=row['xi'],sF=row['sF'],rho=row['rho'],kappa=row['kappa'],
                    stable_fallback_used=r.get('stable_fallback_used'),degreeoff_method=r.get('degreeoff_method'))
                phases=dict(correction_parameter_setup=t_cb,offdiag_parameter_setup=t_ob,
                    bound_configuration_setup=elapsed,
                    total_degree=r.get('t_existing_total_bound_setup_s'),
                    self_autocorrelation=r.get('t_self_autocorrelation_s'),
                    stable_fallback=r.get('t_stable_offdiag_setup_s'),
                    offdiag_incremental=r.get('t_offdiag_incremental_s',r.get('t_offdiag_setup_s')))
                for phase,seconds in phases.items():
                    if seconds is None:raise RuntimeError('Missing phase '+phase+'; available '+str(list(r)))
                    raw.append(dict(metadata,phase=phase,seconds=seconds,
                                    parameter_sample_shared_across_configs=phase in ('correction_parameter_setup','offdiag_parameter_setup')))
    write(RESULTS/'timing_raw.csv',raw)
    save(RESULTS/'timing_context.json',dict(repeats=repeats,warmups=1,threads=S.threadpool_info(),
        load_average=list(os.getloadavg()),platform=sys.platform,
        policy='New isolated bound-only setup batch; parameter samples shared across configurations. Full bound includes correction total-degree FFT and self-exclusion. Fallback cost reported separately; no cached-matvec timings or old-batch speed ratios.'))
    attach_timing_summaries();print('Saved timing rows '+str(len(raw)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('compute','merge','timing','all'))
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--repeats',type=int,default=9)
    args=parser.parse_args()
    if args.stage in ('compute','all'):
        _,groups,selected,focus=context();PARTS.mkdir(parents=True,exist_ok=True)
        jobs=sorted(((k,v,selected,focus) for k,v in groups.items()),key=lambda j:j[1][0]['M'])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_table,j) for j in jobs]
            for i,f in enumerate(as_completed(futures),1):print(f'{i}/{len(futures)} {f.result()}',flush=True)
    if args.stage in ('merge','all'):merge()
    if args.stage in ('timing','all'):timings(args.repeats)


if __name__=='__main__':main()
