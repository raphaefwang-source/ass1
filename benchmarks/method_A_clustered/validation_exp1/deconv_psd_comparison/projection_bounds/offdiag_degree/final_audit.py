"""Independent, resumable saved-artifact audit of self-excluded degree bounds.

Only this follow-up's results/audit_final.json is written. All earlier methods,
figures, references, caches and timings remain read-only. Raw subtraction and
float comparison failures are reported without clipping or inflating bounds.
"""
from collections import Counter,defaultdict
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

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
REPO=HERE.parents[5]
ISSUES=[]
EPS=2.220446049250313e-16
LABELS=('winner','active-badcase','self-dominated-outlier')


def number(value):
    try:
        result=float(value)
        return result if math.isfinite(result) else None
    except (TypeError,ValueError):return None


def truth(value):return value is True or str(value).lower() in ('true','1')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def need(ok,check,detail,kind='integrity'):
    if not ok:ISSUES.append(dict(check=check,kind=kind,detail=detail))
    return bool(ok)


def read_json(path,required=True):
    if not path.exists():
        if required:need(False,'files',f'Missing {path.relative_to(HERE)}','incomplete')
        return {}
    try:return json.loads(path.read_text())
    except (OSError,ValueError) as exc:
        need(False,'files',f'Cannot read {path.name}: {exc}');return {}


def read_csv(path,required=True):
    if not path.exists():
        if required:need(False,'files',f'Missing {path.relative_to(HERE)}','incomplete')
        return []
    try:
        with path.open(newline='',encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))
    except (OSError,ValueError,csv.Error) as exc:
        need(False,'files',f'Cannot read {path.name}: {exc}');return []


def preservation():
    snapshot=read_json(HERE/'source_snapshot.json')
    result={}
    for key,count in (('tracked_file_sha256',861),('reused_part_sha256',141)):
        expected=snapshot.get(key,{})
        changed=[name for name,digest in expected.items() if not (REPO/name).exists() or sha(REPO/name)!=digest]
        need(len(expected)==count and not changed,'preservation',f'{key}: expected {count}, got {len(expected)}, changed={changed}','preservation')
        result[key]=dict(checked=len(expected),changed=changed)
    refs=snapshot.get('independent_references',[]);changed=[]
    for row in refs:
        for pathkey,hashkey in (('path','sha256'),('metadata_path','metadata_sha256')):
            path=REPO/row[pathkey]
            if not path.exists() or sha(path)!=row[hashkey]:changed.append(row[pathkey])
    need(len(refs)==72 and not changed,'preservation',f'Independent reference/sidecar pairs: {len(refs)}, changed={changed}','preservation')
    result['references']=dict(npz=len(refs),sidecars=len(refs),changed=changed)
    csv_hashes=snapshot.get('source_csv_sha256',{})
    changed=[name for name,digest in csv_hashes.items() if not (REPO/name).exists() or sha(REPO/name)!=digest]
    need(not changed,'preservation',f'Changed source CSVs: {changed}','preservation')
    result['source_csv']=dict(checked=len(csv_hashes),changed=changed)
    return result


def equal(a,b):
    return a is not None and b is not None and math.isclose(a,b,rel_tol=2e-12,abs_tol=0.)


def bounds_checks():
    source=read_csv(HERE/'source_catalog.csv');rows=read_csv(RESULTS/'bounds.csv')
    by_source={r['source_row_id']:r for r in source}
    counts=Counter(r.get('source_row_id') for r in rows)
    need(len(rows)==1692 and set(counts)==set(by_source) and all(v==1 for v in counts.values()),
         'coverage',f'Expected all 1692 source rows exactly once; found {len(rows)}','incomplete')
    expected_datasets=Counter(r.get('dataset') for r in source)
    need(Counter(r.get('dataset') for r in rows)==expected_datasets,'coverage','Dataset row counts changed')
    bad_source=[];bad_formula=[];bad_scope=[];bad_stability=[];bad_ratio=[]
    historical_strict=[];norm_strict=[];norm_material=[];monotonicity=[]
    physical={};case_map={};groups=defaultdict(set)
    for i,row in enumerate(rows,1):
        sid=row.get('source_row_id');old=by_source.get(sid)
        if old is None:bad_source.append(dict(row=i,reason='unknown source_row_id'));continue
        for key in ('case_id','parameter_id','config','dataset','correction_norm_source'):
            if row.get(key)!=old.get(key):bad_source.append(dict(source=sid,field=key,reason='source identity/provenance changed'))
        for key in ('E_old','E_new','E_proj','normGammaD','correction_norm','delta','N','L','M','p','xi','sR','sF','sT','rho','kappa'):
            if number(row.get(key))!=number(old.get(key)):bad_source.append(dict(source=sid,field=key,reason='old numerical input changed'))
        if truth(row.get('active'))!=truth(old.get('active')):bad_source.append(dict(source=sid,field='active',reason='old correction activity changed'))
        pid=row.get('parameter_id');case=row.get('case_id');physical[(pid,row.get('config'))]=row;case_map[case]=row
        groups[pid].add(old.get('table_key'))
        cn=number(row.get('correction_norm'));nref=number(row.get('normGammaD'))
        N=number(row.get('N'));delta=number(row.get('delta'))
        simple=number(row.get('B_simple'));total=number(row.get('B_total'));bo=number(row.get('B_old'))
        raw=number(row.get('B_off_raw'));iraw=number(row.get('B_improved_raw'))
        off=number(row.get('B_off'));improved=number(row.get('B_improved'))
        available=truth(row.get('numerical_bound_available'))
        if any(v is None for v in (N,delta,simple,total,bo,raw,iraw,nref)) or nref<=0:
            bad_formula.append(dict(source=sid,reason='missing/nonfinite required old/raw bound'));continue
        if not equal(simple,N*delta if N>=2 else 0.) or bo!=min(simple,total) or iraw!=min(simple,total,raw):
            bad_formula.append(dict(source=sid,reason='old or raw minimum formulas differ'))
        total_eigenvalue=number(row.get('total_max_eigenvalue'))
        if total_eigenvalue is None or not equal(total,2*total_eigenvalue):
            bad_formula.append(dict(source=sid,reason='B_total differs from twice total eigenvalue'))
        if number(row.get('B_old_previous'))!=number(old.get('B_old')) or number(row.get('B_total_previous'))!=number(old.get('B_total')):
            bad_source.append(dict(source=sid,reason='stored previous old bounds changed'))
        olddiff=abs(bo-number(old['B_old']));totaldiff=abs(total-number(old['B_total']))
        if number(row.get('B_old_source_residual_abs'))!=olddiff or number(row.get('B_total_source_residual_abs'))!=totaldiff:
            bad_formula.append(dict(source=sid,reason='old-bound replay residual differs'))
        if olddiff!=0 or totaldiff!=0:bad_source.append(dict(source=sid,reason='replayed old bounds differ from PR #5',old_difference=olddiff,total_difference=totaldiff))
        if available:
            om=number(row.get('offdiag_max_eigenvalue'))
            if off is None or improved is None or om is None or not equal(off,2*om) or improved!=min(simple,total,off):
                bad_formula.append(dict(source=sid,reason='available final bound/off-eigenvalue/minimum inconsistent'))
            if improved is not None and improved>bo:monotonicity.append(dict(source=sid,excess=improved-bo))
        elif any(v is not None for v in (off,improved,number(row.get('improved_tightness')),number(row.get('empirical_envelope_new')))):
            bad_scope.append(dict(source=sid,reason='unavailable numerical bound must leave final bound/tightness/envelope blank'))
        for key in ('B_simple','B_total','B_off','B_old','B_improved','B_off_raw','B_improved_raw'):
            value=number(row.get(key));relative=number(row.get(key+'_rel'))
            if (value is None)!=(relative is None) or (value is not None and not equal(relative,value/nref)):
                bad_formula.append(dict(source=sid,field=key,reason='normalization mismatch'))
        for key,bound in (('old_tightness',bo),('improved_tightness',improved)):
            ratio=number(row.get(key))
            expected=bound/cn if cn is not None and cn>0 and bound is not None else None
            if (expected is None)!=(ratio is None) or (expected is not None and not equal(ratio,expected)):
                bad_scope.append(dict(source=sid,field=key,reason='tightness must use only original true correction-only norm'))
        if not truth(row.get('active')) and (cn!=0 or simple!=0 or total!=0 or off!=0 or improved!=0):
            bad_scope.append(dict(source=sid,reason='literal inactive correction must remain exactly zero'))
        for key in ('certified_total_bound','theory_certified_candidate','t_legacy_matvec_s'):
            if row.get(key) not in (None,'','None','null'):bad_scope.append(dict(source=sid,field=key,reason='unavailable certificate/old-batch matvec must remain blank'))
        if truth(row.get('bound_float_interval_certified')):bad_scope.append(dict(source=sid,reason='float diagnostics cannot be interval certified'))
        unreliable=truth(row.get('cancellation_unreliable_raw'));fallback=truth(row.get('stable_fallback_used'))
        fallback_unavailable=truth(row.get('stable_fallback_unavailable'))
        if unreliable and not fallback and available and N!=1:bad_stability.append(dict(source=sid,reason='unreliable raw subtraction exposed as available without fallback'))
        if fallback and row.get('degreeoff_method')!='positive_local_occupancy':bad_stability.append(dict(source=sid,reason='fallback method label inconsistent'))
        if fallback_unavailable and (available or not unreliable):bad_stability.append(dict(source=sid,reason='unavailable stable path flags inconsistent'))
        trace_total=number(row.get('total_degree_trace_sum'));trace_self=number(row.get('self_degree_trace_sum'))
        trace_ratio=number(row.get('self_degree_trace_fraction'));degree_ratio=number(row.get('degreeoff_total_ratio'))
        totalmax=number(row.get('total_max_eigenvalue'));offmax=number(row.get('offdiag_max_eigenvalue'))
        if trace_total is None or trace_self is None or (trace_total!=0 and not equal(trace_ratio,trace_self/trace_total)) or (trace_total==0 and trace_ratio is not None):
            bad_ratio.append(dict(source=sid,reason='self trace ratio differs from unmodified degree traces'))
        if totalmax is None or offmax is None or (totalmax!=0 and not equal(degree_ratio,offmax/totalmax)) or (totalmax==0 and degree_ratio is not None):
            bad_ratio.append(dict(source=sid,reason='off/total eigenvalue ratio inconsistent'))
        eold=number(old['E_old']);enew=number(old['E_new']);eproj=number(old['E_proj'])
        envelope_old=eold+bo/nref;envelope_new=eold+improved/nref if improved is not None else None
        if not equal(number(row.get('empirical_envelope_old')),envelope_old) or ((envelope_new is None)!=(number(row.get('empirical_envelope_new')) is None)) or (envelope_new is not None and not equal(number(row.get('empirical_envelope_new')),envelope_new)):
            bad_formula.append(dict(source=sid,reason='empirical envelope is not measured E_old plus correction bound'))
        if truth(row.get('empirical_old_budget_pass'))!=(envelope_old<=1e-4) or truth(row.get('independent_reference_empirical_pass'))!=(enew<=1e-4):
            bad_formula.append(dict(source=sid,reason='empirical pass flags inconsistent'))
        if envelope_new is not None and truth(row.get('empirical_new_budget_pass'))!=(envelope_new<=1e-4):
            bad_formula.append(dict(source=sid,reason='new empirical budget flag inconsistent'))
        if improved is not None:
            gap=max(eproj-improved/nref,0.)
            if number(row.get('historical_Eproj_violation_rel'))!=gap or truth(row.get('historical_Eproj_material_violation'))!=(gap>512*EPS):
                bad_formula.append(dict(source=sid,reason='historical full-difference violation changed or inflated'))
            if gap>0:historical_strict.append(dict(source=sid,case_id=case,excess_relative=gap))
            expected_violation=max(cn-improved,0.) if cn is not None else None
            if number(row.get('true_norm_violation_abs'))!=expected_violation:bad_formula.append(dict(source=sid,reason='true correction norm violation not recorded literally'))
            if expected_violation is not None and expected_violation>0:
                item=dict(source=sid,case_id=case,excess_absolute=expected_violation,excess_relative_true_norm=expected_violation/cn if cn>0 else None)
                norm_strict.append(item)
                scale=max(simple,total,cn,1e-300)
                if expected_violation>512*EPS*scale:norm_material.append(item)
    need(len(case_map)==1602 and len(physical)==1269 and len(groups)==141,'coverage',f'Cases={len(case_map)}, table/config={len(physical)}, tables={len(groups)}')
    need(all(len(v)==1 for v in groups.values()),'coverage','Parameter ID has multiple physical table keys')
    for name,items in (('source',bad_source),('formulas',bad_formula),('scope',bad_scope),('stability',bad_stability),('ratios',bad_ratio)):
        need(not items,name,f'{len(items)} inconsistent rows')
    need(not monotonicity,'monotonicity',f'{len(monotonicity)} final bounds exceed old bounds')
    need(not norm_material,'true_norm',f'{len(norm_material)} true-norm violations exceed the separately reported float comparison scale')
    return dict(source_rows=len(rows),unique_cases=len(case_map),physical_table_configurations=len(physical),tables=len(groups),
                dataset_counts=dict(Counter(r.get('dataset') for r in rows)),source_mismatches=bad_source,formula_mismatches=bad_formula,
                scope_mismatches=bad_scope,stability_mismatches=bad_stability,ratio_mismatches=bad_ratio,
                unavailable_true_norm_cases=sum(number(r.get('correction_norm')) is None for r in case_map.values()),
                unavailable_improved_table_configurations=sum(not truth(r.get('numerical_bound_available')) for r in physical.values()),
                raw_unreliable_table_configurations=sum(truth(r.get('cancellation_unreliable_raw')) for r in physical.values()),
                stable_fallback_table_configurations=sum(truth(r.get('stable_fallback_used')) for r in physical.values()),
                strict_historical_Eproj_violations=historical_strict,strict_true_norm_violations=norm_strict,
                true_norm_violations_above_float_comparison_scale=norm_material,monotonicity_violations=monotonicity,
                float_comparison_scale='512 eps times max(N delta, B_total, true norm); diagnostics only, never added to bounds'),physical,case_map


def independent_audits(physical):
    focus=read_csv(HERE/'focus_catalog.csv')
    expected_full={(r['parameter_id'],r['config']) for r in focus}
    checked_full={key for key,r in physical.items() if truth(r.get('audit_full_checked'))}
    small_bad=[key for key,r in physical.items() if not truth(r.get('audit_small_checked')) or number(r.get('audit_small_N'))!=4]
    need(len(expected_full)==342 and checked_full==expected_full,'independent_checks',f'Expected 342 full-N focus checks; actual {len(checked_full)}')
    need(not small_bad,'independent_checks',f'{len(small_bad)} table/configurations lack N4 independent offdiagonal sum')
    literal={pid for (pid,config),r in physical.items() if truth(r.get('audit_literal_checked'))}
    active_tables={pid for (pid,config),r in physical.items() if truth(r.get('active'))}
    need(literal==active_tables,'independent_checks',f'Active tables={len(active_tables)}, literal-N2 checked={len(literal)}')
    bad=[];small_unavailable=[];small_availability_bad=[];small_available=[]
    for key,r in physical.items():
        small_is_unavailable=(truth(r.get('audit_small_raw_unreliable'))
                              and not truth(r.get('audit_small_stable_fallback_used')))
        if (r.get('audit_small_bound_numerical_available') in (None,'')
                or truth(r.get('audit_small_bound_numerical_available'))==small_is_unavailable
                or truth(r.get('audit_small_stable_fallback_unavailable'))!=small_is_unavailable):
            small_availability_bad.append(key)
        if small_is_unavailable:
            small_unavailable.append(dict(parameter_id=key[0],config=key[1],
                true_correction_norm=number(r.get('audit_small_true_correction_norm')),
                raw_error_relative_true_norm=number(r.get('audit_small_raw_error_relative_true_correction_norm')),
                final_exposed_raw_error_relative_true_norm=number(r.get('audit_small_error_relative_true_correction_norm')),
                reason='Raw cancellation unreliable; automatic stable path exceeds its query budget; no available numerical small-N bound is claimed.'))
        else:small_available.append(r)
        for prefix in ('audit_small','audit_full'):
            if not truth(r.get(prefix+'_checked')):continue
            if number(r.get(prefix+'_N'))!=(4 if prefix=='audit_small' else 256):bad.append(dict(key=key,prefix=prefix,reason='incorrect audit N'))
            error=number(r.get(prefix+'_off_error_max_spectral'))
            raw_error=number(r.get(prefix+'_raw_off_error_max_spectral'))
            direct=number(r.get(prefix+'_B_off_direct'))
            if error is None or raw_error is None or direct is None:bad.append(dict(key=key,prefix=prefix,reason='missing independent degree residual'))
        if truth(r.get('audit_literal_checked')) and (number(r.get('audit_literal_N'))!=2 or number(r.get('audit_literal_pair_max_component_residual')) is None):
            bad.append(dict(key=key,reason='missing literal N2 contraction residual'))
    need(not bad,'independent_checks',f'{len(bad)} incomplete independent residual records')
    need(not small_availability_bad,'independent_checks',f'Small-N availability flags missing/inconsistent: {small_availability_bad}')
    files=sorted((RESULTS/'parts').glob('*.json'));algorithm=sha(HERE/'offdiag_bounds.py') if (HERE/'offdiag_bounds.py').exists() else None
    bad_parts=[];part_configs=0;part_replay_mismatches=[]
    for path in files:
        data=read_json(path);part_configs+=len(data.get('configurations',{}))
        if data.get('algorithm_sha256')!=algorithm or data.get('parameter_id')!=path.stem:bad_parts.append(path.name)
        for config,checkpoint in data.get('configurations',{}).items():
            merged=physical.get((data.get('parameter_id'),config),{})
            for field in ('B_off','B_improved','B_off_raw','B_improved_raw','B_old','B_total',
                          'correction_norm','audit_small_true_correction_norm',
                          'audit_small_error_relative_true_correction_norm','audit_small_off_error_max_spectral'):
                if number(checkpoint.get(field))!=number(merged.get(field)):
                    part_replay_mismatches.append(dict(parameter_id=path.stem,config=config,field=field))
            for field in ('numerical_bound_available','stable_fallback_used','cancellation_unreliable_raw',
                          'audit_small_raw_unreliable','audit_small_stable_fallback_used'):
                if truth(checkpoint.get(field))!=truth(merged.get(field)):
                    part_replay_mismatches.append(dict(parameter_id=path.stem,config=config,field=field))
    need(len(files)==141 and part_configs==1269 and not bad_parts,'checkpoint_provenance',f'Parts={len(files)}, configs={part_configs}, stale algorithms={bad_parts}')
    need(not part_replay_mismatches,'checkpoint_provenance',f'{len(part_replay_mismatches)} primary records differ from frozen computation checkpoints')
    def maximum(field):return max((number(r.get(field)) or 0 for r in physical.values()),default=0)
    maximum_small_available=max((number(r.get('audit_small_error_relative_true_correction_norm')) or 0
                                 for r in small_available),default=0)
    worst_small_available=max(small_available,key=lambda r:number(r.get('audit_small_error_relative_true_correction_norm')) or 0,default={})
    return dict(full_focus_table_configurations=len(checked_full),expected_full_focus=len(expected_full),
                full_active_dense_checks=sum(truth(r.get('audit_full_checked')) and truth(r.get('active')) for r in physical.values()),
                full_inactive_zero_checks=sum(truth(r.get('audit_full_checked')) and not truth(r.get('active')) for r in physical.values()),
                small_N4_table_configurations=len(physical)-len(small_bad),literal_N2_active_tables=len(literal),
                max_full_raw_error_relative_true_norm=maximum('audit_full_raw_error_relative_true_correction_norm'),
                max_full_final_error_relative_true_norm=maximum('audit_full_error_relative_true_correction_norm'),
                max_small_raw_error_relative_true_norm=maximum('audit_small_raw_error_relative_true_correction_norm'),
                max_small_final_error_relative_true_norm_all_including_unavailable=maximum('audit_small_error_relative_true_correction_norm'),
                max_small_final_error_relative_true_norm_available_only=maximum_small_available,
                worst_available_small_N_case={k:worst_small_available.get(k) for k in
                    ('parameter_id','config','audit_small_true_correction_norm','audit_small_off_error_max_spectral',
                     'audit_small_error_relative_true_correction_norm')},
                small_N_unavailable_bounds=len(small_unavailable),small_N_unavailable_cases=small_unavailable,
                small_N_availability_flag_mismatches=small_availability_bad,
                max_literal_pair_component_residual=maximum('audit_literal_pair_max_component_residual'),
                incomplete_residuals=bad,parts=len(files),part_configurations=part_configs,algorithm_sha256=algorithm,stale_parts=bad_parts,
                primary_checkpoint_replay_mismatches=part_replay_mismatches)


def targeted_small_stability(physical):
    rows=read_csv(RESULTS/'smallN_stability_cases.csv');summary=read_json(RESULTS/'smallN_stability_summary.json')
    expected={key for key,r in physical.items()
              if (number(r.get('audit_small_error_relative_true_correction_norm')) or 0)>=1e-8
              or truth(r.get('audit_small_stable_fallback_unavailable'))}
    keys=Counter((r.get('parameter_id'),r.get('config')) for r in rows)
    need(len(rows)==6 and set(keys)==expected and all(count==1 for count in keys.values()),'targeted_stability',
         f'Expected six separately selected offline diagnostics, found {len(rows)}','incomplete')
    bad=[];strict=[]
    for r in rows:
        key=(r.get('parameter_id'),r.get('config'));primary=physical.get(key,{})
        norm=number(r.get('true_correction_norm'))
        if (number(r.get('audit_N'))!=4 or norm is None or norm<=0
                or number(r.get('primary_small_norm'))!=number(primary.get('audit_small_true_correction_norm'))
                or norm!=number(r.get('primary_small_norm'))
                or number(r.get('primary_small_error_ratio'))!=number(primary.get('audit_small_error_relative_true_correction_norm'))
                or not truth(r.get('primary_records_unchanged'))):
            bad.append(dict(key=key,reason='small-N truth/source/unchanged-primary claims inconsistent'));continue
        primary_available=truth(primary.get('audit_small_bound_numerical_available'))
        if truth(r.get('auto_numerical_bound_available'))!=primary_available or truth(r.get('auto_stable_fallback_unavailable'))==primary_available:
            bad.append(dict(key=key,reason='offline auto availability differs from unchanged primary result'))
        for prefix in ('auto','forced'):
            error=number(r.get(prefix+'_error_max_spectral'));ratio=number(r.get(prefix+'_error_relative_true_norm'))
            if error is None or not equal(ratio,error/norm):bad.append(dict(key=key,reason=prefix+' error ratio not derived from true small-N correction norm'))
        forced_bound=number(r.get('forced_B_improved'));saved_violation=number(r.get('forced_strict_norm_violation_abs'))
        if forced_bound is None or saved_violation!=max(norm-forced_bound,0.):bad.append(dict(key=key,reason='forced strict violation not recorded without inflation'))
        elif saved_violation>0:strict.append(dict(parameter_id=key[0],config=key[1],excess_absolute=saved_violation,excess_relative_true_norm=saved_violation/norm))
        auto_bound=number(r.get('auto_B_improved'));auto_violation=number(r.get('auto_strict_norm_violation_abs'))
        if not primary_available and any(number(r.get(k)) is not None for k in ('auto_B_improved','auto_B_off','auto_strict_norm_violation_abs')):
            bad.append(dict(key=key,reason='auto-unavailable diagnostic must keep auto numerical bound fields blank'))
        if primary_available and (auto_bound is None or auto_violation!=max(norm-auto_bound,0.)):
            bad.append(dict(key=key,reason='auto violation inconsistent'))
        queries=number(r.get('stable_queries'));budget=number(r.get('default_query_budget'))
        if queries is None or budget!=50_000_000 or truth(r.get('forced_exceeds_default_query_budget'))!=(queries>budget):
            bad.append(dict(key=key,reason='explicit forced budget exception missing/inconsistent'))
        if number(r.get('forced_positive_setup_seconds')) is None or number(r.get('forced_positive_setup_seconds'))<0:
            bad.append(dict(key=key,reason='extra forced setup cost missing'))
    algorithm=sha(HERE/'offdiag_bounds.py') if (HERE/'offdiag_bounds.py').exists() else None
    need(summary.get('algorithm_sha256')==algorithm,'targeted_stability','Forced audit algorithm differs from frozen primary implementation')
    need(summary.get('primary_results_modified') is False,'targeted_stability','Forced audit must not replace primary unavailable records')
    unavailable=sum(not truth(r.get('auto_numerical_bound_available')) for r in rows)
    beyond=sum(truth(r.get('forced_exceeds_default_query_budget')) for r in rows)
    maxerror=max((number(r.get('forced_error_relative_true_norm')) or 0 for r in rows),default=0)
    maxviolation=max((number(r.get('forced_strict_norm_violation_abs')) or 0 for r in rows),default=0)
    need(number(summary.get('selected_table_configurations'))==6 and number(summary.get('auto_unavailable'))==unavailable==3
         and number(summary.get('forced_exceeds_default_budget'))==beyond==3,'targeted_stability','Forced/auto-unavailable summary counts differ')
    need(number(summary.get('max_forced_relative_error'))==maxerror and number(summary.get('max_forced_strict_norm_violation_abs'))==maxviolation,
         'targeted_stability','Forced accuracy/strict-violation summary maxima differ')
    need(not bad,'targeted_stability',f'{len(bad)} targeted offline diagnostic inconsistencies')
    return dict(rows=len(rows),auto_unavailable=unavailable,forced_over_default_budget=beyond,
                max_forced_relative_error=maxerror,strict_forced_norm_violations=strict,
                forced_setup_seconds=[number(r.get('forced_positive_setup_seconds')) for r in rows],
                inconsistent_rows=bad,summary=summary,
                scope='Separate explicit-cost offline audit; primary auto-unavailable bounds remain unavailable; no bound inflation.')


def test_and_plots():
    tests=read_json(RESULTS/'tests_summary.json');logpath=RESULTS/'test_offdiag_bounds.log'
    text=logpath.read_text() if logpath.exists() else '';ran=re.search(r'Ran\s+(\d+)\s+tests?',text)
    ok=bool(ran and int(ran.group(1))>=12 and re.search(r'^OK\s*$',text,re.M) and not re.search(r'^FAILED\b',text,re.M))
    need(ok and truth(tests.get('ok')) and number(tests.get('failures'))==0 and number(tests.get('errors'))==0,'tests','Expected saved 12+ tests, zero errors/failures, OK','incomplete')
    folder=HERE/'figures';names={p.stem for p in folder.glob('fig*.png')}|{p.stem for p in folder.glob('fig*.svg')}
    names|={p.stem for p in (folder/'plot_data').glob('fig*.csv')}
    need(len(names)>=4,'plots',f'Expected four complete figure groups, found {len(names)}','incomplete')
    status=read_json(folder/'plot_status.json',required=False)
    if status:need(not status.get('failed') and not status.get('skipped'),'plots','Plots include failed/skipped stages')
    figures={}
    for name in sorted(names):
        png,svg,csvpath=folder/f'{name}.png',folder/f'{name}.svg',folder/'plot_data'/f'{name}.csv'
        exists=all(p.exists() and p.stat().st_size for p in (png,svg,csvpath))
        need(exists,'plots',f'{name}: PNG/SVG/CSV triplet missing','incomplete')
        if not exists:continue
        raw=png.read_bytes();pixels=struct.unpack('>II',raw[16:24]) if len(raw)>=24 and raw[:8]==b'\x89PNG\r\n\x1a\n' else (0,0)
        image=svg.read_text();points=read_csv(csvpath)
        need(min(pixels)>0 and '<svg' in image and '</svg>' in image and bool(points),'plots',f'{name}: invalid image or empty plotted data')
        figures[name]=dict(pixels=pixels,source_rows=len(points),png_sha256=sha(png),svg_sha256=sha(svg),csv_sha256=sha(csvpath))
    return dict(tests=tests,log_ok=ok,figures=figures,plot_status=status)


def timing_cases(case_map):
    raw=read_csv(RESULTS/'timing_raw.csv');summaries=read_csv(RESULTS/'cases_summary.csv');context=read_json(RESULTS/'timing_context.json')
    by_case={r.get('case_id'):r for r in summaries};labels=Counter(r.get('selected_label') for r in summaries)
    need(len(summaries)==27 and len(by_case)==27 and labels==dict.fromkeys(LABELS,9),'cost',f'Expected three nine-configuration cases, found {dict(labels)}','incomplete')
    phases=('correction_parameter_setup','offdiag_parameter_setup','bound_configuration_setup','total_degree','self_autocorrelation','stable_fallback','offdiag_incremental')
    counts=Counter((r.get('case_id'),number(r.get('repeat')),r.get('phase')) for r in raw)
    expected={(case,float(rep),phase) for case in by_case for rep in range(1,10) for phase in phases}
    invalid=[i for i,r in enumerate(raw,1) if number(r.get('seconds')) is None or number(r.get('seconds'))<0
             or number(r.get('threads'))!=1 or truth(r.get('audit')) or r.get('batch')!='isolated_offdiag_bound_setup']
    need(len(raw)==1701 and set(counts)==expected and all(v==1 for v in counts.values()) and not invalid,'cost',f'Expected 1701 isolated setup samples; rows={len(raw)}, invalid={invalid}')
    shared=defaultdict(list)
    for r in raw:
        if r.get('phase') in phases[:2]:shared[(r.get('parameter_id'),number(r.get('repeat')),r['phase'])].append(r)
    bad_shared=[str(key) for key,rs in shared.items() if len(rs)!=9 or len({number(r['seconds']) for r in rs})!=1
                or not all(truth(r.get('parameter_sample_shared_across_configs')) for r in rs)]
    need(not bad_shared,'cost',f'Shared parameter samples were misrepresented: {bad_shared}')
    bad_medians=[];budget={}
    for case,row in by_case.items():
        for phase in phases:
            values=[number(r['seconds']) for r in raw if r.get('case_id')==case and r.get('phase')==phase]
            if len(values)==9 and all(v is not None for v in values) and not equal(number(row.get('median_'+phase+'_s')),statistics.median(values)):
                bad_medians.append(dict(case_id=case,phase=phase))
        source=case_map.get(case,{})
        for field in ('E_old','E_new','E_proj','correction_norm','B_old','B_improved'):
            if number(row.get(field))!=number(source.get(field)):bad_medians.append(dict(case_id=case,field=field,reason='case summary differs from raw bound result'))
    for label in LABELS:
        own=[r for r in summaries if r.get('selected_label')==label]
        budget[label]=dict(configurations=len(own),old_envelope_pass=sum(truth(r.get('empirical_old_budget_pass')) for r in own),
                           improved_envelope_pass=sum(truth(r.get('empirical_new_budget_pass')) for r in own),
                           independent_reference_empirical_pass=sum(truth(r.get('independent_reference_empirical_pass')) for r in own))
    need(not bad_medians,'cost',f'Cost/case summary mismatches: {bad_medians}')
    need(number(context.get('repeats'))==9,'cost','Context must record nine repeats')
    return dict(raw_rows=len(raw),case_summary_rows=len(summaries),labels=dict(labels),invalid_samples=invalid,
                bad_shared_parameter_samples=bad_shared,summary_mismatches=bad_medians,budgets=budget,context=context)


def main():
    immutable=preservation();bound,physical,cases=bounds_checks();audits=independent_audits(physical)
    stability=targeted_small_stability(physical)
    tests=test_and_plots();cost=timing_cases(cases);verification=read_json(RESULTS/'verification_summary.json')
    for key,value in (('source_rows',1692),('unique_case_ids',1602),('unique_parameter_ids',141),('computed_table_configurations',1269),
                      ('small_N_checked_table_configurations',1269),('full_N_checked_table_configurations',342),('literal_N2_checked_tables',60),
                      ('small_N_unavailable_bounds',audits['small_N_unavailable_bounds'])):
        need(number(verification.get(key))==value,'verification_summary',f'{key}: expected {value}, recorded {verification.get(key)}')
    need(not truth(verification.get('certified_total_bound_available')),'scope','No certified original error is available')
    for name in ('README.md','proof.md','report.md'):
        p=HERE/name;need(p.exists() and p.stat().st_size,'documents',f'Missing/empty {name}','incomplete')
    report=HERE/'report.md';need(report.exists() and bool(re.search(r'[\u4e00-\u9fff]',report.read_text() if report.exists() else '')),'documents','Expected Chinese report','incomplete')
    artifacts={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p!=RESULTS/'audit_final.json'
               and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.csv','.png','.svg','.md','.log')}
    result=dict(timestamp=datetime.now(ZoneInfo('America/Detroit')).isoformat(),status='pass' if not ISSUES else 'partial_fail',
                ok=not ISSUES,issues=ISSUES,checks=dict(preservation=immutable,bounds=bound,independent_audits=audits,
                tests_and_plots=tests,incremental_cost=cost,targeted_small_N_stability=stability,verification_summary=verification),artifact_sha256=artifacts,
                scope='Exact-arithmetic sufficient correction bound; float64 diagnostics are not interval certificates. '
                      'Empirical envelope failure does not reject independently validated empirical-pass candidates.')
    RESULTS.mkdir(exist_ok=True);target=RESULTS/'audit_final.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],issues=len(ISSUES),output=str(target)),ensure_ascii=False))


if __name__=='__main__':main()
