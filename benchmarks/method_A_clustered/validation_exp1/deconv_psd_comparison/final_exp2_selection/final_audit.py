"""Saved-artifact final audit and reproducible seal for the four-step Exp. 2.

Run after all formal/scaling CPU work and structural_audit.py have finished.
Only this directory's results/audit_final.json and seal_manifest.json are
written. HEAD may advance; every pre-existing file/reference must stay intact.
"""
import ast
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
import subprocess
import sys
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
REPO=HERE.parents[4]
METHODS=('PPPM-original','B2-positive','PPPM-grid-PSD')
EPSILONS=(1e-3,1e-4,1e-5)
CONFIGS={f'{f}_N256_s{s}_rho0.75' for f in ('uniform','moderate','strong') for s in (1,2,3)}
BATCH='final_four_step_common_serial'
CORE=('protocol.py','search.py','timing.py')
ISSUES=[]
HASH_CACHE={}


def number(value):
    try:
        x=float(value);return x if math.isfinite(x) else None
    except (TypeError,ValueError):return None
def truth(value):return value is True or str(value).lower() in ('true','1')
def equal(a,b):return a is not None and b is not None and math.isclose(a,b,rel_tol=3e-12,abs_tol=1e-16)
def sha(path):
    path=Path(path)
    if path not in HASH_CACHE:HASH_CACHE[path]=hashlib.sha256(path.read_bytes()).hexdigest()
    return HASH_CACHE[path]
def need(ok,check,detail,kind='integrity'):
    if not ok:ISSUES.append(dict(check=check,kind=kind,detail=detail))
    return bool(ok)
def read_json(path,required=True):
    if not path.exists():
        if required:need(False,'files',f'Missing {path.relative_to(HERE)}','incomplete')
        return {}
    try:return json.loads(path.read_text())
    except (OSError,ValueError) as exc:need(False,'files',f'{path.name}: {exc}');return {}
def read_csv(path,required=True):
    if not path.exists():
        if required:need(False,'files',f'Missing {path.relative_to(HERE)}','incomplete')
        return []
    try:
        with path.open(newline='',encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))
    except (OSError,ValueError,csv.Error) as exc:need(False,'files',f'{path.name}: {exc}');return []
def git(*args):return subprocess.run(['git',*args],cwd=REPO,check=True,capture_output=True,text=True).stdout


def preservation():
    source=read_json(HERE/'source_snapshot.json');result={}
    for key,expected_count in (('tracked_file_sha256',1048),('old_results_sha256',773)):
        expected=source.get(key,{})
        changed=[name for name,digest in expected.items() if not (REPO/name).exists() or sha(REPO/name)!=digest]
        need(len(expected)==expected_count and not changed,'preservation',f'{key}: count={len(expected)}, changed={changed}','preservation')
        result[key]=dict(checked=len(expected),changed=changed)
    for key in ('source_csv_sha256','theory_source_sha256'):
        expected=source.get(key,{})
        changed=[name for name,digest in expected.items() if not (REPO/name).exists() or sha(REPO/name)!=digest]
        need(bool(expected) and not changed,'preservation',f'{key}: changed={changed}','preservation')
        result[key]=dict(checked=len(expected),changed=changed)
    refs=source.get('independent_references',[]);bad=[];known=set();directories=set()
    for row in refs:
        for pathkey,hashkey in (('path','sha256'),('metadata_path','metadata_sha256')):
            path=REPO/row[pathkey];known.add(path);directories.add(path.parent)
            if not path.exists() or sha(path)!=row[hashkey]:bad.append(row[pathkey])
    new_old_cache=[str(path.relative_to(REPO)) for directory in directories for path in directory.iterdir()
                   if path.is_file() and path.suffix in ('.npz','.json') and path not in known]
    need(len(refs)==72 and not bad and not new_old_cache,'reference_preservation',
         f'Reference pairs={len(refs)}, changed={bad}, unexpected old-cache writes={new_old_cache}','preservation')
    result['independent_references']=dict(npz=len(refs),sidecars=len(refs),changed=bad,new_files_in_old_cache=new_old_cache)
    old_names=set(source.get('tracked_file_sha256',{}));tracked=set(filter(None,git('ls-files','-z').split('\0')))
    outside=[name for name in tracked-old_names if not (REPO/name).is_relative_to(HERE)]
    untracked=list(filter(None,git('ls-files','--others','--exclude-standard','-z').split('\0')))
    outside+= [name for name in untracked if not (REPO/name).is_relative_to(HERE)]
    need(not outside,'new_scope',f'New files outside final_exp2_selection: {outside}')
    result['git']=dict(base_head=source.get('git',{}).get('head'),current_head=git('rev-parse','HEAD').strip(),
                       branch=git('branch','--show-current').strip(),new_files_outside_scope=outside,
                       policy='New commits are permitted; old-file SHA invariants, rather than HEAD equality, are required.')
    return result


def code_and_theory():
    protocol=read_json(RESULTS/'search_protocol.json');alias=read_json(RESULTS/'alias_calibration.json')
    seeds=read_csv(RESULTS/'theory_initialization.csv');candidates=read_csv(RESULTS/'candidates.csv')
    members=read_csv(RESULTS/'candidate_memberships.csv');search=read_csv(RESULTS/'search.csv')
    need(tuple(protocol.get('epsilons',()))==EPSILONS,'four_step','Mandatory epsilon tiers differ')
    need(number(protocol.get('eta_max'))==.95 and number(protocol.get('max_M'))==128,'four_step','Explicit current safety/resource preset differs')
    need(protocol.get('search_source_sha256')==sha(HERE/'search.py'),'core_provenance','search.py changed after search protocol snapshot')
    need(alias.get('certified') is False and number(alias.get('target_kappa'))==.2 and number(alias.get('target_rho'))==.75,
         'four_step','Alias calibration must be same-physics and empirical only')
    groups={(number(r.get('epsilon')),r.get('theory_seed_id')) for r in seeds}
    counts=Counter(number(r.get('epsilon')) for r in seeds)
    group_counts=Counter(eps for eps,_ in groups)
    need(len(groups)==18 and set(counts)==set(EPSILONS) and all(group_counts[eps]==6 for eps in EPSILONS),
         'four_step',f'Expected six xi start groups per epsilon; group counts={dict(group_counts)}')
    bad=[]
    for r in seeds:
        eps=number(r.get('epsilon'));eta=number(r.get('eta_prediction'));p=number(r.get('p'));M=number(r.get('M'))
        if (r.get('origin')!='four_step_theory' or truth(r.get('predicted_error_is_acceptance_constraint'))
                or truth(r.get('bound_used_as_constraint')) or truth(r.get('PSD_eigenvalue_used_as_constraint'))):
            bad.append(dict(id=r.get('theory_seed_id'),reason='theory origin or hard-constraint flag inconsistent'))
        if not equal(number(r.get('sR')),number(r.get('theory_sR'))) or not equal(number(r.get('sF')),number(r.get('theory_sF'))):
            bad.append(dict(id=r.get('theory_seed_id'),reason='tail initialization values not used'))
        if eps is None or not equal(number(r.get('cutoff_error_budget')),eps/3) or not equal(number(r.get('alias_error_budget')),eps/3):
            bad.append(dict(id=r.get('theory_seed_id'),reason='initial model budget provenance mismatch'))
        if eta is None or not 0<eta<=.95 or M is None or p is None or not 2<=p<=min(9,M):
            bad.append(dict(id=r.get('theory_seed_id'),reason='seed safety or finite searched p range inconsistent'));continue
        q=eta/(2-eta);prediction=number(r.get('alias_C'))*q**p
        if not equal(number(r.get('alias_q')),q) or not equal(number(r.get('alias_prediction')),prediction):
            bad.append(dict(id=r.get('theory_seed_id'),reason='alias predictor not traceable'))
        if r.get('seed_role')=='smallest_predicted_reasonable_p':
            if prediction>eps/3 or (p>2 and number(r.get('alias_C'))*q**(p-1)<=eps/3):
                bad.append(dict(id=r.get('theory_seed_id'),reason='chosen reasonable p is not the first predictor pass at that M'))
    need(not bad,'four_step',f'{len(bad)} seed provenance inconsistencies')
    byid={r['id']:r for r in candidates};searchids={r['id'] for r in search}
    need(len(byid)==len(candidates)==len(searchids)==len(search),'search','Candidate/search unique ID coverage differs')
    budget=read_json(RESULTS/'search_budget.json')
    need(number(budget.get('unique_fresh_candidates'))==len(search),'search','Retained search budget count differs')
    for eps in EPSILONS:
        grid=[r for r in members if r.get('method')=='PPPM-grid-PSD' and number(r.get('epsilon'))==eps]
        need({'seeds','local','boundary'}<={r.get('stage') for r in grid},'four_step',f'Grid PSD lacks a completed stage at epsilon={eps}')
    forbidden=[]
    for name in CORE:
        tree=ast.parse((HERE/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                module=getattr(node,'module','') or ''
                if 'offdiag_bounds' in module or module.endswith('bounds'):forbidden.append(dict(file=name,line=node.lineno,module=module))
    need(not forbidden,'code_scope',f'Physical selection core imports a projection-bound constraint implementation: {forbidden}')
    tests=read_json(RESULTS/'tests_summary.json');log=RESULTS/'test_selection.log';text=log.read_text() if log.exists() else ''
    need(truth(tests.get('ok')) and (number(tests.get('tests_run')) or 0)>=12 and number(tests.get('failures'))==0 and number(tests.get('errors'))==0
         and re.search(r'^OK\s*$',text,re.M),'tests','Expected 12+ selection tests with OK','incomplete')
    for name,digest in tests.get('source_sha256',{}).items():
        need((HERE/name).exists() and sha(HERE/name)==digest,'tests',f'{name} changed after the recorded tests')
    structural=read_json(HERE/'structural_audit.json')
    need(truth(structural.get('ok')) and structural.get('status')=='pass' and len(structural.get('records',[]))==2,
         'structure','Independent odd/even structural audit missing or failed','incomplete')
    need(not truth(structural.get('bound_used_as_constraint')) and not truth(structural.get('PSD_eigenvalue_used_as_constraint')),
         'structure','Structural diagnostics must not screen search candidates')
    for name in CORE:
        need(structural.get('source_sha256',{}).get(name)==sha(HERE/name),'structure',f'{name} differs from the structural audit source snapshot')
    return dict(search_protocol=protocol,alias_calibration=alias,seed_rows=len(seeds),seed_groups=len(groups),
                seed_mismatches=bad,unique_search_candidates=len(search),membership_rows=len(members),
                tests=tests,structural=structural,forbidden_constraint_imports=forbidden,
                reviewed_flow='Each candidate constructs its own S.table/Grid PSD mesh; protocol evaluates full GS+GL versus independent Gamma_D. Feasibility is fresh max-nine E_G only.',
                safety_scope='eta_max=.95 is this run explicit margin; previous implementation requires eta<1, complete shells, W2>1e-10. p=1 remains legal but untested; searched p is 2..9 and optima are local/tested only.')


def validate_physics(row,check):
    if row.get('status')!='ok':return
    need(number(row.get('N'))==256 and number(row.get('rho'))==.75 and number(row.get('kappa'))==.2
         and number(row.get('amplitude'))==1 and number(row.get('kernel_amplitude'))==1,check,f'{row.get("id")}: physical state differs')
    need(number(row.get('E_G')) is not None and number(row.get('E_G'))>=0 and number(row.get('E_Gamma'))==number(row.get('E_G')),
         check,f'{row.get("id")}: missing complete operator error')
    need(number(row.get('normGammaD')) is not None and number(row.get('normGammaD'))>0
         and equal(number(row.get('rc')),number(row.get('sR'))/number(row.get('xi'))),check,'Reference normalization or real cutoff provenance differs')
    need(not truth(row.get('bound_used_as_constraint')) and not truth(row.get('PSD_eigenvalue_used_as_constraint')),check,'Bound/PSD eigenvalue hard constraint flagged')
    p=number(row.get('p'));M=number(row.get('M'))
    need(p is not None and M is not None and 1<=p<=M and truth(row.get('mesh_legal')),check,'Illegal assignment/mesh accepted')
    if row.get('method')!='B2-positive':
        eta=number(row.get('eta'));denom=number(row.get('W2min'))
        need(eta is not None and eta<=.95 and eta<1 and number(row.get('n_excluded'))==0 and denom is not None and denom>1e-10,
             check,f'{row.get("id")}: FFT/denominator safety mismatch')
        need(equal(eta,number(row.get('kc'))*number(row.get('h'))/math.pi),check,'Actual eta does not use actual snapped kc/h')
    if row.get('method')=='PPPM-grid-PSD':need(truth(row.get('projection_constructed')),check,'Candidate own PSD mesh was not constructed')


def validation_audit():
    combined={};results={}
    for rawname,summaryname in (('validation','validation_summary'),('cross_validation','cross_validation_summary')):
        raw=read_csv(RESULTS/f'{rawname}.csv');summary=read_csv(RESULTS/f'{summaryname}.csv');groups=defaultdict(list)
        for row in raw:groups[row.get('id')].append(row);validate_physics(row,rawname)
        byid={row.get('id'):row for row in summary};bad=[]
        need(len(byid)==len(summary) and set(byid)==set(groups),rawname,'Summary IDs/duplicate coverage differs')
        for id_,rows in groups.items():
            if len(rows)!=9 or {r.get('config') for r in rows}!=CONFIGS:bad.append(dict(id=id_,reason='not exactly all nine configurations'))
            own=byid.get(id_,{});ok=all(r.get('status')=='ok' and number(r.get('E_G')) is not None for r in rows)
            actual=max((number(r.get('E_G')) for r in rows if number(r.get('E_G')) is not None),default=None)
            if number(own.get('n_configs'))!=9 or (ok and number(own.get('max_E_G'))!=actual):bad.append(dict(id=id_,reason='max-nine summary differs'))
            if ok and own.get('status')!='ok':bad.append(dict(id=id_,reason='summary success status differs'))
        need(not bad,rawname,f'{len(bad)} incomplete/inconsistent validation groups')
        combined.update({key:value for key,value in byid.items() if value.get('status')=='ok' and number(value.get('n_configs'))==9})
        results[rawname]=dict(rows=len(raw),candidates=len(groups),inconsistent=bad)
    plan=read_json(RESULTS/'cross_plan.json');planned={r['id'] for r in plan.get('candidates',[])}
    crossids={r['id'] for r in read_csv(RESULTS/'cross_validation_summary.csv')}
    need(len(planned)==11 and planned==crossids,'cross_controls',f'Expected 11 retained reciprocal controls, plan={len(planned)}, actual={len(crossids)}')
    for r in read_csv(RESULTS/'shortlist.csv'):
        cert=combined.get(r['id'],{});eps=number(r.get('epsilon'));maxerr=number(cert.get('max_E_G'))
        passed=cert.get('status')=='ok' and number(cert.get('n_configs'))==9 and maxerr is not None and maxerr<=eps
        need(truth(r.get('feasible'))==passed,'precision_acceptance',f'{r["id"]}: feasible flag is not only max-nine full error')
    results['cross_plan']=plan
    return results,combined


def timing_audit(validated):
    plan=read_json(RESULTS/'timing_plan.json');context=read_json(RESULTS/'timing_context.json')
    final=read_csv(RESULTS/'final_timing.csv');raw=read_csv(RESULTS/'timing_raw.csv')
    byid={r['id']:r for r in final};planned={r['id'] for r in plan.get('candidates',[])}
    need(bool(byid) and len(byid)==len(final) and set(byid)==planned,'formal_timing','Formal plan/summary candidate IDs differ','incomplete')
    need(context.get('batch')==BATCH and number(context.get('rounds'))==5 and number(context.get('reps'))==7
         and number(context.get('threads'))==1 and number(context.get('n_configs'))==9,'formal_timing','Common 5x7x9 one-thread context differs')
    pools=context.get('threadpools',[])
    need(isinstance(pools,list) and all(number(row.get('num_threads'))==1 for row in pools),
         'formal_timing','A discovered native thread pool does not report one thread')
    for name in CORE:
        need(context.get('source_sha256',{}).get(name)==sha(HERE/name),'core_provenance',f'{name} differs from formal timing source snapshot')
    need(not truth(plan.get('PSD_checks_or_bounds_in_acceptance')),'formal_timing','Formal plan applies a bound/PSD acceptance constraint')
    prescreen=read_csv(RESULTS/'serial_prescreen_raw.csv');timed=read_csv(RESULTS/'timed_candidates.csv')
    pool={id_:r for id_,r in validated.items() if number(r.get('max_E_G')) is not None and number(r['max_E_G'])<=max(EPSILONS)}
    prescreen_groups=defaultdict(list)
    for row in prescreen:prescreen_groups[row.get('id')].append(row)
    need(set(prescreen_groups)==set(pool),'prescreen','Serial prescreen must include every freshly validated <=1e-3 candidate')
    ranks={};bad_prescreen=[]
    for id_,samples in prescreen_groups.items():
        keys=Counter((number(r.get('round')),number(r.get('rep'))) for r in samples)
        if (len(samples)!=9 or set(keys)!={(float(a),float(b)) for a in range(3) for b in range(3)}
                or not all(v==1 for v in keys.values()) or any(number(r.get('seconds')) is None or number(r.get('seconds'))<=0
                  or number(r.get('threads'))!=1 or r.get('batch')!='serial_prescreen_only' or r.get('config')!='uniform_N256_s2_rho0.75' for r in samples)):
            bad_prescreen.append(id_);continue
        ranks[id_]=statistics.median(number(r['seconds']) for r in samples)
    expected_timed=set()
    for eps in EPSILONS:
        for method in METHODS:
            feasible=sorted((r for id_,r in pool.items() if id_ in ranks and r.get('method')==method and number(r['max_E_G'])<=eps),key=lambda r:ranks[r['id']])
            expected_timed.update(r['id'] for r in feasible[:2])
    expected_timed.update(r['id'] for r in read_csv(RESULTS/'cross_validation_summary.csv')
                          if r.get('status')=='ok' and number(r.get('max_E_G')) is not None and number(r['max_E_G'])<=max(EPSILONS))
    need(not bad_prescreen and {r['id'] for r in timed}==expected_timed,'prescreen',f'Fresh top2/cross shortlist differs; malformed prescreen={bad_prescreen}')
    for row in timed:
        need(equal(number(row.get('prescreen_median_s')),ranks.get(row['id'])),'prescreen',f'{row["id"]}: rank does not use new serial samples')
    groups=defaultdict(list)
    for r in raw:groups[r['id']].append(r)
    need(set(groups)==set(byid),'formal_timing','Raw/summary candidate IDs differ')
    expected={(cfg,float(rnd),float(rep)) for cfg in CONFIGS for rnd in range(5) for rep in range(7)}
    bad=[]
    for id_,row in byid.items():
        samples=groups.get(id_,[]);count=Counter((r.get('config'),number(r.get('round')),number(r.get('rep'))) for r in samples)
        if (len(samples)!=315 or set(count)!=expected or not all(n==1 for n in count.values())
                or any(number(r.get('seconds')) is None or number(r.get('seconds'))<=0 or r.get('batch')!=BATCH
                       or number(r.get('threads'))!=1 or r.get('phase')!='matvec' for r in samples)):
            bad.append(dict(id=id_,reason='not all 315 unique positive same-batch samples'));continue
        med=statistics.median(statistics.median(number(r['seconds']) for r in samples if r['config']==cfg) for cfg in CONFIGS)
        if not equal(med,number(row.get('median_matvec_s'))):bad.append(dict(id=id_,reason='reported median differs from fresh raw samples'))
        if row.get('method')=='direct-dense':
            if number(row.get('max_E_G'))!=0 or any(number(row.get(k)) is not None for k in ('median_table_setup_s','median_config_setup_s')):
                bad.append(dict(id=id_,reason='cached direct matrix action must not fabricate assembly/setup costs'))
            need(bool(row.get('error_role')) and bool(row.get('setup_status')),'direct_dense','Reference-self error/setup limitations not labeled')
        else:
            cert=validated.get(id_,{})
            if number(cert.get('n_configs'))!=9 or number(row.get('max_E_G'))!=number(cert.get('max_E_G')):
                bad.append(dict(id=id_,reason='timed candidate lacks own fresh complete precision certificate'))
        if row.get('timing_batch')!=BATCH:bad.append(dict(id=id_,reason='summary uses another batch'))
    need(not bad,'formal_timing',f'{len(bad)} raw/precision timing inconsistencies')
    for filename,with_config in (('table_setup_raw.csv',False),('config_setup_raw.csv',True)):
        setup=read_csv(RESULTS/filename);sg=defaultdict(list)
        for r in setup:sg[(r.get('id'),r.get('config') if with_config else None)].append(r)
        ids={id_ for id_,r in byid.items() if r.get('method') in METHODS}
        expected_keys={(id_,cfg if with_config else None) for id_ in ids for cfg in (CONFIGS if with_config else (None,))}
        malformed=[key for key,rs in sg.items() if len(rs)!=3 or {number(r.get('rep')) for r in rs}!={0.,1.,2.}
                   or any(number(r.get('seconds')) is None or number(r.get('seconds'))<0 or r.get('batch')!=BATCH for r in rs)]
        need(set(sg)==expected_keys and not malformed,'setup',f'{filename}: incomplete fresh three-replicate groups')
        for id_ in ids:
            times=([statistics.median(number(r['seconds']) for r in setup if r['id']==id_ and r.get('config')==cfg) for cfg in CONFIGS]
                   if with_config else [number(r['seconds']) for r in setup if r['id']==id_])
            field='median_config_setup_s' if with_config else 'median_table_setup_s'
            need(bool(times) and all(v is not None for v in times) and equal(statistics.median(times),number(byid[id_].get(field))),
                 'setup',f'{id_}: {field} differs from fresh samples')
    snapshots=read_json(RESULTS/'system_load.json')
    need(isinstance(snapshots,list) and len(snapshots)==10 and all(len(row.get('loadavg',[]))==3
         and bool(row.get('top_processes')) for row in snapshots),'formal_timing','Missing before/each-config process/load snapshots')
    environment=read_json(RESULTS/'runtime_environment.json')
    thread_keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
    post_pools=environment.get('enumerable_native_pools',[])
    verified=bool(post_pools) and all(number(row.get('num_threads'))==1 for row in post_pools)
    need(number(environment.get('requested_threads'))==1
         and all(str(environment.get('thread_environment',{}).get(key))=='1' for key in thread_keys)
         and bool(environment.get('numpy_config')) and bool(environment.get('numpy')),
         'runtime_environment','Post-timing same-protocol environment/library provenance incomplete')
    need(truth(environment.get('native_thread_count_independently_verified'))==verified
         and (bool(post_pools) or 'not independently enumerated' in str(environment.get('scope'))),
         'runtime_environment','Native thread enumeration limitation is mislabeled')
    return dict(candidates=len(final),samples=len(raw),expected_samples=len(final)*315,inconsistent=bad,
                prescreen_candidates=len(prescreen_groups),prescreen_samples=len(prescreen),prescreen_malformed=bad_prescreen,
                prescreen_selected_candidates=len(timed),context=context,plan=plan,system_load=snapshots,runtime_environment=environment,
                native_threadpool_discovery='All discovered pools report one thread.' if pools else
                    'threadpool_info discovered no inspectable native pools; protocol thread environment requests one, but native pool size is not independently enumerated.'),byid


def best_and_cost(final):
    best=read_csv(RESULTS/'best_by_epsilon.csv');expected={(method,eps) for method in (*METHODS,'direct-dense') for eps in EPSILONS}
    count=Counter((r.get('method'),number(r.get('epsilon'))) for r in best)
    need(len(best)==12 and set(count)==expected and all(v==1 for v in count.values()),'best','Expected all 12 method/epsilon availability rows')
    indexed={};bad=[]
    for row in best:
        method=row.get('method');eps=number(row.get('epsilon'));choices=[r for r in final.values() if r.get('method')==method and number(r.get('max_E_G')) is not None and number(r['max_E_G'])<=eps]
        winner=min(choices,key=lambda r:number(r.get('median_matvec_s'))) if choices else None
        if truth(row.get('available')):
            if winner is None or row.get('id')!=winner['id'] or number(row.get('max_E_G'))!=number(winner.get('max_E_G')):
                bad.append(dict(method=method,epsilon=eps,reason='not the fastest measured own-error feasible common-batch candidate'))
            if row.get('timing_batch')!=BATCH:bad.append(dict(method=method,epsilon=eps,reason='not common formal batch'))
            indexed[(method,eps)]=row
        elif winner is not None or number(row.get('max_E_G')) is not None or number(row.get('median_matvec_s')) is not None:
            bad.append(dict(method=method,epsilon=eps,reason='unavailable row fabricated values or omitted a feasible measured point'))
    need(not bad,'best',f'{len(bad)} precision/cost selection inconsistencies')
    amort=read_csv(RESULTS/'amortization.csv');cost_bad=[]
    for row in amort:
        b=indexed.get((row.get('method'),number(row.get('epsilon'))),{})
        if not b:cost_bad.append(dict(id=row.get('id'),reason='amortized candidate is not available best'));continue
        table=number(b.get('median_table_setup_s'));setup=number(b.get('median_config_setup_s'));action=number(b.get('median_matvec_s'))
        if row.get('method')=='direct-dense':
            if truth(row.get('available')) or number(row.get('total_step_s')) is not None:cost_bad.append(dict(id=row.get('id'),reason='unmeasured direct assembly amortization must stay unavailable'))
        else:
            expected_cost=table/number(row.get('table_reuse_steps'))+setup+number(row.get('matvecs_per_step'))*action
            if not truth(row.get('available')) or not equal(expected_cost,number(row.get('total_step_s'))):cost_bad.append(dict(id=row.get('id'),reason='setup/amortization formula differs'))
    need(len(amort)==16*len(indexed) and not cost_bad,'amortization',f'Amortization rows={len(amort)}, mismatches={cost_bad}')
    ratios=read_csv(RESULTS/'paired_speed_ratios.csv');r_bad=[]
    expected_ratios={(eps,'PPPM-grid-PSD',reference) for eps in EPSILONS
                     for reference in ('PPPM-original','B2-positive','direct-dense')
                     if ('PPPM-grid-PSD',eps) in indexed and (reference,eps) in indexed}
    ratio_keys=Counter((number(r.get('epsilon')),r.get('method'),r.get('reference_method')) for r in ratios)
    need(set(ratio_keys)==expected_ratios and all(n==1 for n in ratio_keys.values()),'paired_ratios','Missing/duplicate eligible Grid/comparator speed ratios')
    raw=read_csv(RESULTS/'timing_raw.csv');blocks=defaultdict(list)
    for row in raw:
        value=number(row.get('seconds'))
        if value is not None:blocks[(row.get('id'),row.get('config'),number(row.get('round')))].append(value)
    for row in ratios:
        a=indexed.get((row.get('method'),number(row.get('epsilon'))),{});b=indexed.get((row.get('reference_method'),number(row.get('epsilon'))),{})
        if row.get('batch')!=BATCH or row.get('id')!=a.get('id') or row.get('reference_id')!=b.get('id') or number(row.get('blocks'))!=45 or number(row.get('resamples'))!=5000:
            r_bad.append(row.get('epsilon'))
        low=number(row.get('ratio_ci95_low'));high=number(row.get('ratio_ci95_high'))
        if low is None or high is None or not 0<low<=high:r_bad.append(row.get('epsilon'))
        paired=[statistics.median(blocks[(a.get('id'),cfg,float(rnd))])/statistics.median(blocks[(b.get('id'),cfg,float(rnd))])
                for cfg in CONFIGS for rnd in range(5) if blocks[(a.get('id'),cfg,float(rnd))] and blocks[(b.get('id'),cfg,float(rnd))]]
        if len(paired)!=45 or not equal(statistics.median(paired),number(row.get('ratio_median_paired_blocks'))):r_bad.append(row.get('epsilon'))
    need(not r_bad,'paired_ratios','Ratio source/batch/interval inconsistencies')
    return dict(best_rows=len(best),unavailable=[dict(method=r.get('method'),epsilon=number(r.get('epsilon')),status=r.get('status')) for r in best if not truth(r.get('available'))],
                selection_mismatches=bad,amortization_rows=len(amort),cost_mismatches=cost_bad,paired_ratios=ratios),indexed


def components(best):
    rows=read_csv(RESULTS/'components_raw.csv');ids={r['id'] for r in best.values() if r.get('method') in METHODS};groups=defaultdict(list)
    for row in rows:groups[row.get('id')].append(row)
    phases={'near','fft_influence','spread_gather','mesh','total'};bad=[]
    expected={(cfg,phase,float(rep)) for cfg in CONFIGS for phase in phases for rep in range(9)}
    for id_ in ids:
        count=Counter((r.get('config'),r.get('phase'),number(r.get('rep'))) for r in groups.get(id_,[]))
        if set(count)!=expected or not all(v==1 for v in count.values()) or any(number(r.get('seconds')) is None or number(r.get('seconds'))<0 or r.get('batch')!='isolated_component_diagnostics' for r in groups.get(id_,[])):
            bad.append(id_)
    need(set(groups)==ids and not bad,'components',f'Components incomplete for {bad}; separate medians are not added to infer full action')
    return dict(rows=len(rows),candidates=len(ids),incomplete=bad)


def scaling():
    rows=read_csv(RESULTS/'scaling.csv',required=False)
    refs=[]
    for N in (64,128):
        npz=HERE/'scaling_references'/f'N{N}_uniform_s2.npz';meta_path=npz.with_suffix('.json')
        meta=read_json(meta_path,required=False)
        if meta:
            need(npz.exists() and number(meta.get('N'))==N and number(meta.get('kappa'))==.2 and number(meta.get('rho'))==.75
                 and number(meta.get('amplitude'))==1 and number(meta.get('cert_bound_rel')) is not None and number(meta['cert_bound_rel'])<=1e-10,
                 'scaling_reference',f'N={N}: fresh reference physical/certificate mismatch')
            need('not mixed' in str(meta.get('setup_timing_role')),'scaling_reference','Concurrent reference generation must not be matched assembly/setup timing')
            refs.append(dict(N=N,npz_sha256=sha(npz) if npz.exists() else None,metadata=meta))
    if not rows:return dict(status='optional scaling not yet run',fresh_references=refs)
    raw=read_csv(RESULTS/'scaling_raw.csv');bad=[]
    for row in rows:
        if not truth(row.get('available')):continue
        N=number(row.get('N'));method=row.get('method')
        samples=[r for r in raw if number(r.get('N'))==N and r.get('method')==method]
        keys=Counter((number(r.get('round')),number(r.get('rep'))) for r in samples)
        expected={(float(rnd),float(rep)) for rnd in range(3) for rep in range(7)}
        if (set(keys)!=expected or not all(n==1 for n in keys.values()) or number(row.get('n_configs'))!=1
                or number(row.get('max_E_G')) is None or number(row['max_E_G'])>1e-4
                or any(r.get('batch')!='isolated_N_scaling_single_config' or number(r.get('threads'))!=1 or number(r.get('seconds')) is None for r in samples)):
            bad.append(dict(N=N,method=method))
        elif not equal(statistics.median(number(r['seconds']) for r in samples),number(row.get('median_matvec_s'))):bad.append(dict(N=N,method=method,reason='median differs'))
    need(not bad,'scaling',f'Single-configuration scaling inconsistencies: {bad}')
    return dict(status='single nested uniform configuration per N, separate timing batch; not a nine-configuration scaling certificate',
                rows=len(rows),raw_samples=len(raw),fresh_references=refs,inconsistent=bad)


def plots_and_docs():
    names=('fig1_best_tested_parameters','fig2_accuracy_cost_pareto','fig3_matched_error_and_runtime',
           'fig4_setup_and_matvec_components','fig5_dynamics_step_amortized_cost')
    folder=HERE/'figures';status=read_json(folder/'plot_status.json')
    generated={row.get('figure') if isinstance(row,dict) else row for row in status.get('generated',[])}
    skipped={row.get('figure') if isinstance(row,dict) else row for row in status.get('skipped',[])}
    need(set(names)<=generated and not (set(names)&skipped) and not status.get('failed'),
         'plots','A required plot stage was not generated or a plot failed; optional figures may be explicitly skipped')
    actual=(set(names)|generated|{p.stem for p in folder.glob('fig*.png')}|{p.stem for p in folder.glob('fig*.svg')})-{None}
    result={}
    for name in sorted(actual):
        png,svg,data=folder/f'{name}.png',folder/f'{name}.svg',folder/'plot_data'/f'{name}.csv'
        exists=all(p.exists() and p.stat().st_size for p in (png,svg,data));need(exists,'plots',f'{name}: PNG/SVG/CSV incomplete','incomplete')
        if not exists:continue
        raw=png.read_bytes();size=struct.unpack('>II',raw[16:24]) if len(raw)>=24 and raw[:8]==b'\x89PNG\r\n\x1a\n' else (0,0)
        text=svg.read_text();points=read_csv(data);need(min(size)>0 and '<svg' in text and '</svg>' in text and bool(points),'plots',f'{name}: invalid images/empty plotted source')
        result[name]=dict(pixels=size,source_rows=len(points),png_sha256=sha(png),svg_sha256=sha(svg),csv_sha256=sha(data))
    for name in ('README.md','report.md'):
        path=HERE/name;need(path.exists() and path.stat().st_size,'documents',f'{name} missing','incomplete')
    report=HERE/'report.md';need(report.exists() and bool(re.search(r'[\u4e00-\u9fff]',report.read_text() if report.exists() else '')),'documents','Expected Chinese report','incomplete')
    return dict(figures=result,status=status)


def main():
    old=preservation();theory=code_and_theory();validation,certificates=validation_audit()
    measured,final=timing_audit(certificates);selected,best=best_and_cost(final)
    decomposition=components(best);scale=scaling();plots=plots_and_docs()
    excluded={RESULTS/'audit_final.json',RESULTS/'seal_manifest.json'}
    artifacts={str(path.relative_to(HERE)):sha(path) for path in sorted(HERE.rglob('*')) if path.is_file()
               and path not in excluded and '__pycache__' not in path.parts and path.suffix in ('.py','.csv','.json','.png','.svg','.md','.log','.npz')}
    result=dict(timestamp=datetime.now(ZoneInfo('America/Detroit')).isoformat(),status='pass' if not ISSUES else 'partial_fail',ok=not ISSUES,issues=ISSUES,
                checks=dict(preservation=old,four_step_and_structure=theory,validation=validation,formal_timing=measured,
                            best_and_cost=selected,components=decomposition,scaling=scale,plots=plots),artifact_sha256=artifacts,
                scope='Fresh finite local search and max-nine empirical precision; structural PSD proof is independent of configurations. No global optimum or uniform accuracy certificate is asserted.')
    RESULTS.mkdir(exist_ok=True);target=RESULTS/'audit_final.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    # The seal references the audit; the audit never references the seal. Reports
    # consume status only, avoiding any SHA cycle during report/seal reruns.
    seal=dict(status=result['status'],audit_file='results/audit_final.json',audit_sha256=sha(target),
              artifact_sha256=artifacts,core_source_sha256={name:sha(HERE/name) for name in CORE},
              initial_snapshot_sha256=sha(HERE/'source_snapshot.json'),base_head=old['git']['base_head'],
              current_head=old['git']['current_head'],policy='HEAD may advance inside the new directory; original file/reference invariants are preserved.')
    (RESULTS/'seal_manifest.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],issues=len(ISSUES),output=str(target)),ensure_ascii=False))
    if ISSUES:sys.exit(1)


if __name__=='__main__':main()
