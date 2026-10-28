"""Independent Stage 0/1 evidence audit; run only after numerical timing ends.

This reader does not regenerate trajectories or alter old evidence. Optional
--replay reexecutes the six first production steps locally and compares them
with saved arrays. It must never run concurrently with the timing pilot.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import ast
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
from evidence import HERE,RESULTS,INPUTS,LOGS,BASE,sha,array_sha,save,read_csv
from model import lj_force,rng_sha,AMPLITUDE,KAPPA,L_BOX

REPO=HERE.parents[2]
PREFIX=str(HERE.relative_to(REPO))+'/'


def read_json(path):return json.loads(Path(path).read_text())


def command(*args):
    return subprocess.check_output(list(args),cwd=REPO,text=True).strip()


def close(actual,expected,rtol=1e-11,atol=1e-12):
    return bool(np.allclose(actual,expected,rtol=rtol,atol=atol,equal_nan=False))


class Audit:
    def __init__(self):self.checks=[]
    def check(self,name,passed,detail=None):
        self.checks.append(dict(name=name,passed=bool(passed),detail=detail))
        print(('PASS ' if passed else 'FAIL ')+name,flush=True)
    def guarded(self,name,fn):
        try:return fn()
        except Exception as exc:
            self.check(name,False,dict(exception=repr(exc)))
            return None


def baseline_checks(a):
    snapshot=read_json(HERE/'baseline_snapshot.json');hashes=snapshot['tracked_file_sha256']
    tracked=command('git','ls-tree','-r','--name-only',BASE).splitlines()
    a.check('exact_registered_PR9_base_and_all_1759_tracked_files',snapshot['base_sha']==BASE and len(hashes)==1759 and set(tracked)==set(hashes),
            dict(base=BASE,snapshot_count=len(hashes),git_tree_count=len(tracked)))
    differences=[name for name,value in hashes.items() if not (REPO/name).is_file() or sha(REPO/name)!=value]
    a.check('all_PR3_to_PR9_baseline_file_SHA256_unchanged',not differences,differences)
    changed=command('git','diff',BASE,'--name-only').splitlines()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=REPO).decode().split('\0')
    outside=[p for p in changed+untracked if p and not p.startswith(PREFIX)]
    a.check('all_tracked_and_untracked_changes_confined_new_pilot_directory',not outside,outside)
    branch=command('git','branch','--show-current')
    a.check('requested_local_working_branch',branch=='codex/baoab-ab-stage01',branch)
    environment=read_json(RESULTS/'environment_initial.json')
    a.check('local_Mac_only_environment_and_Stage0_1_scope',platform.system()=='Darwin' and environment['execution']=='local Mac only'
            and environment['stages']==[0,1] and environment['machine']=='arm64',
            dict(current_platform=platform.platform(),saved_environment=environment))
    failures=[]
    for path in HERE.glob('*.py'):
        try:ast.parse(path.read_text(),filename=str(path))
        except SyntaxError as exc:failures.append(str(exc))
    a.check('all_new_python_sources_parse',not failures,failures)
    for name,minimum in [('baseline_regressions.log',54),('new_tests_final.log',27)]:
        path=LOGS/name
        if not path.exists() and name=='new_tests_final.log':path=LOGS/'new_tests.log'
        text=path.read_text() if path.exists() else ''
        matches=re.findall(r'Ran (\d+) tests? in ([0-9.]+)s',text)
        passed=bool(matches) and int(matches[-1][0])>=minimum and text.rstrip().endswith('OK')
        a.check(name+'_passed',passed,dict(path=str(path.relative_to(HERE)),count=int(matches[-1][0]) if matches else None,minimum=minimum))
    params=read_json(RESULTS/'spatial_selection.json')['params']
    a.check('new_physical_kernel_and_box_not_historical_kappa02',params['kappa']==.7 and close(params['c'],AMPLITUDE,rtol=1e-15,atol=0)
            and close(params['L'],L_BOX,rtol=1e-15),params)


def initial_checks(a):
    seeds=[];q_hashes=[];p_hashes=[];prep_durations=[]
    for idx in range(3):
        meta=read_json(RESULTS/f'initial_{idx}.json');rows=read_csv(RESULTS/f'initial_{idx}_preparation.csv')
        path=INPUTS/f'initial_{idx}.npz'
        with np.load(path,allow_pickle=False) as d:data={k:d[k] for k in d.files}
        seeds.append(meta['seed']);q_hashes.append(array_sha(data['q']));p_hashes.append(array_sha(data['p']))
        prep_durations.append(meta['total_wall_s'])
        a.check(f'initial{idx}_6000_actual_scalar_Langevin_steps_time30',meta['steps']==6000 and meta['duration']==30.
                and meta['dt']==.005 and len(rows)==6000 and [r['step'] for r in rows]==list(range(1,6001))
                and close([r['time'] for r in rows],np.arange(1,6001)*.005),
                dict(seed=meta['seed'],steps=len(rows),duration=meta['duration'],actual_wall_s=meta['total_wall_s']))
        valid_hashes=(sha(path)==meta['file_sha256'] and array_sha(data['q'])==meta['q_sha256']
                      and array_sha(data['q_unwrapped'])==meta['q_unwrapped_sha256'] and array_sha(data['p'])==meta['p_sha256'])
        a.check(f'initial{idx}_saved_input_SHA_and_physics',valid_hashes and meta['N']==256 and meta['rho']==.75
                and meta['temperature']==1. and meta['mass']==1. and meta['scalar_gamma']==1.
                and close(data['L'],L_BOX) and np.linalg.norm(data['p'].sum(0))<1e-10)
        # Reconstruct the scalar-preparation random stream without rerunning dynamics.
        rng=np.random.default_rng(meta['seed']);before=rng_sha(rng);p0=rng.standard_normal((256,3));p0-=p0.mean(0)
        stream=hashlib.sha256()
        for _ in range(6000):stream.update(rng.standard_normal((256,3)).tobytes())
        a.check(f'initial{idx}_independent_raw_Gaussian_stream_reconstruction',before==meta['rng_state_before_sha256']
                and rng_sha(rng)==meta['rng_state_after_sha256'] and stream.hexdigest()==meta['raw_gaussian_stream_sha256']
                and np.array_equal(p0,data['initial_p']))
        finite=all(np.isfinite(data[k]).all() for k in ('q','q_unwrapped','p'))
        finite=finite and all(all(np.isfinite(r[k]) for k in ('Tkin','U_per_particle','min_pair_distance','momentum_norm','wall_s')) for r in rows)
        a.check(f'initial{idx}_finite_preparation_and_separate_positive_wall',finite and meta['total_wall_s']>0
                and min(r['min_pair_distance'] for r in rows)>0 and max(r['momentum_norm'] for r in rows)<1e-8)
    a.check('three_independent_seeds_and_distinct_pre_equilibrated_states',len(set(seeds))==3 and len(set(q_hashes))==3
            and len(set(p_hashes))==3,dict(seeds=seeds,preparation_wall_s=prep_durations,
            canonical_scope='30-time-unit COM-constrained Langevin pre-equilibration protocol; no proof of exact canonical equilibration'))


def audit_trajectory(a,label,steps,params,ranks):
    meta=read_json(RESULTS/f'{label}.json');rows=read_csv(RESULTS/f'{label}_steps.csv')
    evidence=read_json(RESULTS/f'{label}_rng.json');path=INPUTS/f'{label}.npz'
    with np.load(path,allow_pickle=False) as d:data={k:d[k] for k in d.files}
    q,p,mid=data['q_unwrapped'],data['p'],data['q_mid'];L=float(data['L']);method=meta['method']
    errors=[]
    if q.shape!=(steps+1,256,3) or p.shape!=q.shape or mid.shape!=(steps,256,3):errors.append('trajectory array dimensions')
    if len(rows)!=steps or len(evidence['states'])!=steps:errors.append('step/state counts')
    if [r['step'] for r in rows]!=list(range(1,steps+1)):errors.append('ordered step indices')
    if not all(np.isfinite(x).all() for x in (q,p,mid,data['q'])):errors.append('nonfinite arrays')
    if not np.array_equal(np.mod(q,L),data['q']):errors.append('wrapped/unwrapped mismatch')
    if meta['steps']!=steps or meta['dt']!=.005 or not close(meta['simulated_time'],steps*.005):errors.append('duration metadata')
    if sha(path)!=meta['file_sha256']:errors.append('trajectory file SHA')
    for k,value in [('start_q',q[0]),('start_p',p[0]),('end_q',q[-1]),('end_p',p[-1])]:
        if meta[k+'_sha256']!=array_sha(value):errors.append(k+' SHA')
    with np.load(INPUTS/f"initial_{meta['initial_index']}.npz",allow_pickle=False) as d:
        if not np.array_equal(q[0],d['q_unwrapped']) or not np.array_equal(p[0],d['p']):errors.append('shared initial state')
    a.check(label+'_saved_trajectory_input_and_complete_step_evidence',not errors,errors)
    errors=[];rng=np.random.default_rng(meta['seed']);tokens=[];build_indices=[];table_hashes=set()
    max_qmid_error=0.;max_second_drift_error=0.;max_momentum=0.;max_phase_residual=0.
    for j,row in enumerate(rows):
        where=f'step{j+1}'
        if evidence['states'][j]['step']!=j+1:errors.append(where+': RNG step order')
        if evidence['states'][j]['before']!=rng.bit_generator.state:errors.append(where+': saved RNG beforestate')
        if row['rng_state_before_sha256']!=rng_sha(rng):errors.append(where+': RNG before SHA')
        z=rng.standard_normal((256,3));ef=er=None
        if method=='A':
            edge_count=int(row['edge_count']);ef=rng.standard_normal((edge_count,3));er=rng.standard_normal((edge_count,3))
            if row['eta_forward_sha256']!=array_sha(ef) or row['eta_reverse_sha256']!=array_sha(er):errors.append(where+': independent edge Gaussian hashes')
            if row['coefficient_cache_initial_size']!=0 or row['coefficient_first_prepare_hit'] is not False:errors.append(where+': coefficient cache reuse')
            if row['active_backend']!='compiled':errors.append(where+': actual compiled backend')
            if row['t_edge_eigh_s']<=0 or row['t_coefficient_setup_s']<=0:errors.append(where+': missing actual edge/coeff cost')
        else:
            if row['eta_forward_sha256'] is not None or row['eta_reverse_sha256'] is not None:errors.append(where+': unexpected B edge noise')
            if row['edge_count']!=0 or row['active_backend']!='none':errors.append(where+': B extra edge setup')
        if row['z_sha256']!=array_sha(z):errors.append(where+': long/full Gaussian SHA')
        if evidence['states'][j]['after']!=rng.bit_generator.state or row['rng_state_after_sha256']!=rng_sha(rng):errors.append(where+': RNG afterstate')
        for name,value in [('q_before',q[j]),('q_after',q[j+1]),('p_before',p[j]),('p_after',p[j+1]),('q_mid',mid[j])]:
            if row[name+'_sha256']!=array_sha(value):errors.append(where+': '+name+' SHA')
        f0,_,_=lj_force(np.mod(q[j],L),L);ph=p[j]+.5*.005*f0
        qm_unwrapped=q[j]+.5*.005*ph
        qm=np.mod(qm_unwrapped,L)
        qerr=float(np.max(np.abs(qm-mid[j])));max_qmid_error=max(max_qmid_error,qerr)
        if not close(qm,mid[j],rtol=0,atol=2e-13):errors.append(where+': BAOAB actual qmid')
        f1,potential,separation=lj_force(np.mod(q[j+1],L),L)
        po=p[j+1]-.5*.005*f1;predicted=q[j]+.5*.005*ph+.5*.005*po
        drift_error=float(np.max(np.abs(predicted-q[j+1])));max_second_drift_error=max(max_second_drift_error,drift_error)
        if not close(predicted,q[j+1],rtol=0,atol=2e-13):errors.append(where+': BAOAB second drift/kick')
        obs=dict(Tkin=float(np.sum((p[j+1]-p[j+1].mean(0))**2)/(3*256-3)),U_per_particle=potential/256,
                 min_pair_distance=separation,momentum_norm=float(np.linalg.norm(p[j+1].sum(0))),
                 q_motion_rms=float(np.sqrt(np.mean((q[j+1]-q[j])**2))))
        for key,value in obs.items():
            if not close(row[key],value):errors.append(where+': actual observable '+key)
        max_momentum=max(max_momentum,obs['momentum_norm'])
        if obs['q_motion_rms']<=0 or separation<=0:errors.append(where+': no valid particle motion')
        if row['q_dependent_setup_rebuilt'] is not True or row['fft_parameter_table_reused'] is not True:errors.append(where+': q/table lifecycle')
        if row['r_d']!=ranks[method]['r_d'] or row['r_n']!=ranks[method]['r_n'] or row['substeps']!=1:errors.append(where+': changed selected rank/substeps')
        if row['r_d_used']>row['r_d'] or row['r_n_used']>row['r_n']:errors.append(where+': invalid actual rank')
        if row['negative_ritz_count']!=0:errors.append(where+': negative Ritz encountered')
        if row['dt']!=.005 or not close(row['time'],(j+1)*.005):errors.append(where+': registered dt/time')
        phases=['t_LJ_forces_s','t_dynamic_setup_s','t_rng_s','t_ou_wall_s','t_baoab_other_s']
        phase_sum=sum(row[k] for k in phases);residual=abs(phase_sum-row['t_baoab_s'])
        max_phase_residual=max(max_phase_residual,residual)
        if residual>1e-12 or min(row[k] for k in phases)<-1e-10:errors.append(where+': disjoint total wall accounting')
        if not close(row['t_LJ_forces_s'],row['t_LJ_force_before_s']+row['t_LJ_force_after_s'],atol=1e-14):errors.append(where+': LJ cost')
        if not close(row['t_fft_s'],row['t_setup_fft_s']+row['t_ou_fft_s'],atol=1e-14):errors.append(where+': nested FFT cost')
        if row['t_operator_setup_s']>row['t_dynamic_setup_s']+1e-10 or row['t_edge_eigh_s']>row['t_edge_constructor_s']+1e-10:errors.append(where+': nested setup clock')
        if row['t_lanczos_damp_s']+row['t_lanczos_noise_s']>row['t_ou_wall_s']+1e-10:errors.append(where+': nested Lanczos clocks')
        if row['t_pair_forward_s']+row['t_pair_reverse_s']>row['t_ou_wall_s']+1e-10:errors.append(where+': nested pair clocks')
        if method=='A' and (row['t_pair_forward_s']<=0 or row['t_pair_reverse_s']<=0):errors.append(where+': missing actual pair sweeps')
        if row['t_reorth_s']+row['t_ritz_eigh_s']>row['t_lanczos_damp_s']+row['t_lanczos_noise_s']+1e-10:errors.append(where+': nested reorth/eigh clocks')
        if row['rss_bytes']<=0 or row['peak_rss_bytes']<=0:errors.append(where+': missing memory evidence')
        tokens.append(row['q_mid_sha256']);build_indices.append(row['operator_build_index']);table_hashes.add(row['table_sha256'])
    if len(set(tokens))!=steps:errors.append('fresh q tokens not unique each step')
    if build_indices!=list(range(int(build_indices[0]),int(build_indices[0])+steps)):errors.append('operator build indices not increasing once per step')
    if len(table_hashes)!=1 or meta['parameter_table_sha256'] not in table_hashes:errors.append('parameter table changed')
    if max_momentum>1e-8:errors.append('COM drift >1e-8')
    means=np.mean([r['t_baoab_s'] for r in rows]);medians=np.median([r['t_baoab_s'] for r in rows]);total=sum(r['t_baoab_s'] for r in rows)
    if not close(means,meta['full_BAOAB_mean_s'],atol=1e-14) or not close(medians,meta['full_BAOAB_median_s'],atol=1e-14) or not close(total,meta['BAOAB_compute_wall_s'],atol=1e-12):errors.append('saved wall summaries')
    if meta['loop_wall_including_evidence_s']+1e-9<total:errors.append('outer loop wall smaller than compute')
    a.check(label+'_independent_RNG_BAOAB_lifecycle_observations_and_timing',not errors,
            dict(errors=errors[:100],error_count=len(errors),steps=steps,max_qmid_abs_error=max_qmid_error,
                 max_second_drift_abs_error=max_second_drift_error,max_COM_norm=max_momentum,
                 max_major_phase_residual_s=max_phase_residual,unique_q_tokens=len(set(tokens)),
                 full_BAOAB_mean_s=float(means),full_BAOAB_median_s=float(medians),compute_wall_s=float(total),
                 timing_interpretation='FFT/eigh/reorth are nested diagnostic subdivisions; excluded from disjoint major phase sum'))
    return rows,meta


def trajectory_checks(a):
    params=read_json(RESULTS/'spatial_selection.json')['params'];ranks=read_json(RESULTS/'ranks_final.json')
    prod=[];timing=[];labels=[]
    for i in range(3):
        for method in ('A','B'):
            label=f'production_{i}_{method}';rows,meta=audit_trajectory(a,label,200,params,ranks)
            prod.extend(rows);labels.append(label)
    for r in range(6):
        for method in ('A','B'):
            rows,meta=audit_trajectory(a,f'timing_round{r}_{method}',40,params,ranks)
            timing.extend(rows)
    pdone=read_json(RESULTS/'production_done.json');tdone=read_json(RESULTS/'timing_done.json')
    a.check('exactly_1200_production_plus_480_separate_true_moving_timing_steps',len(prod)==1200 and len(timing)==480
            and pdone['actual_steps']==1200 and tdone['actual_steps']==480 and tdone['rounds']==6
            and tdone['steps_per_method_per_round']==40,
            dict(actual_production_steps=len(prod),actual_timing_steps=len(timing),production_labels=labels))
    seeds=[read_json(RESULTS/f'{label}.json')['seed'] for label in labels]
    a.check('six_production_trajectory_streams_use_distinct_actual_seeds',len(set(seeds))==6,seeds)
    table=read_json(RESULTS/'production_table.json')
    a.check('single_parameter_table_reused_and_1200_q_dependent_builds',table['reused_across_steps']
            and len({r['table_sha256'] for r in prod})==1 and all(r['table_sha256']==table['sha256'] for r in prod)
            and [r['operator_build_index'] for r in prod]==list(range(1,1201))
            and table['wall_s']>0,table)
    aggregate=read_csv(RESULTS/'production_steps.csv')
    keys=('run','step','q_before_sha256','q_after_sha256','p_before_sha256','p_after_sha256','q_mid_sha256','t_baoab_s')
    a.check('aggregate_production_CSV_matches_each_saved_trajectory',len(aggregate)==len(prod) and
            all(all(x[k]==y[k] for k in keys) for x,y in zip(aggregate,prod)))
    aggregate=read_csv(RESULTS/'timing_steps.csv');index={(r['run'],r['step']):r for r in timing}
    a.check('aggregate_timing_CSV_matches_each_saved_trajectory',len(aggregate)==len(timing) and
            all(all(r[k]==index[(r['run'],r['step'])][k] for k in keys) for r in aggregate))
    rounds=read_csv(RESULTS/'timing_rounds.csv');order_rng=np.random.default_rng(tdone['order_seed']);issues=[]
    for idx,row in enumerate(rounds):
        order=list(order_rng.permutation(['A','B']))
        if row['round']!=idx or row['order']!=','.join(order):issues.append(dict(round=idx,error='interleaved order'))
        A=read_json(RESULTS/f'timing_round{idx}_A.json');B=read_json(RESULTS/f'timing_round{idx}_B.json')
        if A['start_q_sha256']!=B['start_q_sha256'] or A['start_p_sha256']!=B['start_p_sha256']:issues.append(dict(round=idx,error='same start control'))
        ratio=A['full_BAOAB_mean_s']/B['full_BAOAB_mean_s']
        if not close(row['A_over_B'],ratio,atol=1e-14):issues.append(dict(round=idx,error='ratio'))
    a.check('six_interleaved_rounds_reconstruct_order_control_and_actual_wall_ratios',len(rounds)==6 and not issues,issues)


def spatial_checks(a):
    selection=read_json(RESULTS/'spatial_selection.json')
    paths=[RESULTS/f'initial_{i}_spatial.json' for i in range(3)]
    paths += [RESULTS/f'production_{i}_{m}_visited_spatial.json' for i in range(3) for m in ('A','B')]
    issues=[];errors=[]
    for path in paths:
        r=read_json(path);d=r['diagnostics'];ref=r['reference'];params=r['params']
        if not r['strict_budget_met'] or r['conservative_relative_error_bound']>2e-7 or not r['structural_checks_passed']:issues.append(dict(file=path.name,error='strict spatial/structure budget'))
        if params!=selection['params']:issues.append(dict(file=path.name,error='common spatial params'))
        if ref['kappa']!=.7 or ref['c']!=AMPLITUDE or ref['R_max']<45 or ref['certified_tail_relative']>2e-9:issues.append(dict(file=path.name,error='independent physical direct reference'))
        if len(ref['convergence'])<3 or ref['convergence'][-2]['difference_to_final_relative']>2e-9:issues.append(dict(file=path.name,error='direct annulus convergence'))
        if d['self_particle_edges']!=0 or not d['physical_k0_preserved_before_projection']:issues.append(dict(file=path.name,error='self edges/k0'))
        direct=path.with_suffix('').parent/(path.with_suffix('').name+'_direct.npz')
        with np.load(direct,allow_pickle=False) as data:
            if array_sha(data['G'])!=ref['matrix_sha256'] or array_sha(np.mod(data['q'],float(data['L'])))!=ref['q_sha256']:issues.append(dict(file=path.name,error='direct reference hashes'))
        if path.name.startswith('initial_'):
            i=int(path.name.split('_')[1])
            with np.load(INPUTS/f'initial_{i}.npz',allow_pickle=False) as data:q=data['q']
        else:
            label=path.name.replace('_visited_spatial.json','')
            with np.load(INPUTS/f'{label}.npz',allow_pickle=False) as data:q=data['q_mid'][-1]
        if array_sha(np.mod(q,params['L']))!=r['q_sha256']:issues.append(dict(file=path.name,error='actual sampled geometry'))
        errors.append(dict(file=path.name,relative_error_bound=r['conservative_relative_error_bound'],reference_tail=ref['certified_tail_relative']))
    a.check('three_actual_starts_and_six_actual_visited_O_states_meet_strict_spatial_and_structure_budgets',not issues,
            dict(issues=issues,measured_bounds=errors))
    stage0=read_json(RESULTS/'stage0_done.json');visited=read_json(RESULTS/'visited_done.json')
    a.check('completed_Stage0_and_six_visited_state_markers',stage0['completed'] and stage0['strict_spatial_all_starts']
            and stage0['initial_states']==3 and visited['sampled_states']==6 and visited['at_actual_q_mid'])


def rank_checks(a):
    ranks=read_json(RESULTS/'ranks_final.json');tags=[f'initial_{i}_final' for i in range(3)]
    tags += [f'production_{i}_{m}_visited' for i in range(3) for m in ('A','B')]
    issues=[];max_error=0.;rows_total=0
    for tag in tags:
        record=read_json(RESULTS/f'action_validation_{tag}.json');rows=read_csv(RESULTS/f'action_checks_{tag}.csv')
        with np.load(RESULTS/f'action_outputs_{tag}.npz',allow_pickle=False) as data:
            if array_sha(data['q'])!=record['q_sha256'] or array_sha(data['p'])!=record['p_sha256']:issues.append(dict(tag=tag,error='actual probe state hashes'))
            for row in rows:
                prefix=f"{row['method']}__{row['kind']}__{row['probe']}__"
                probe,exact,actual=(data[prefix+k] for k in ('probe','exact','actual'))
                err=float(np.linalg.norm(actual-exact)/max(np.linalg.norm(exact),np.finfo(float).tiny))
                if not close(err,row['relative_error'],rtol=1e-10,atol=1e-16) or err>2e-8 or not row['passed']:issues.append(dict(tag=tag,error='recomputed Krylov probe error',probe=row['probe']))
                for key,arr in [('probe',probe),('exact',exact),('actual',actual)]:
                    if array_sha(arr)!=row[key+'_sha256']:issues.append(dict(tag=tag,error='probe output hash'))
                key='r_d' if row['kind']=='damping' else 'r_n'
                if row['rank_requested']!=ranks[row['method']][key] or row['rank_used']>row['rank_requested']:issues.append(dict(tag=tag,error='actual method-specific rank'))
                if row['negative_ritz_count']!=0:issues.append(dict(tag=tag,error='negative Ritz'))
                max_error=max(max_error,err);rows_total+=1
            if not record['all_probes_passed'] or record['max_relative_action_error']>2e-8:issues.append(dict(tag=tag,error='action marker'))
            if not {'slow','fast','random','actual_momentum','actual_gaussian'}.issubset({r['probe'] for r in rows}):issues.append(dict(tag=tag,error='missing registered directions'))
            if not any(str(r['probe']).startswith('lowk_') and '_L' in r['probe'] for r in rows) or not any(str(r['probe']).startswith('lowk_') and '_T' in r['probe'] for r in rows):issues.append(dict(tag=tag,error='missing low-k L/T'))
            if tag.startswith('initial_'):
                if len(record['splitting'])!=3 or [r['substeps'] for r in record['splitting']]!=[1,2,4]:issues.append(dict(tag=tag,error='splitting convergence evidence separate'))
            else:
                label=tag.replace('_visited','');step=read_csv(RESULTS/f'{label}_steps.csv')[-1]
                with np.load(INPUTS/f'{label}.npz',allow_pickle=False) as trajectory:
                    q0,p0,L=trajectory['q_unwrapped'][-2],trajectory['p'][-2],float(trajectory['L'])
                    force,_,_=lj_force(np.mod(q0,L),L);ph=p0+.5*.005*force
                    if not close(data['p'].ravel(),ph.ravel(),rtol=0,atol=1e-13):issues.append(dict(tag=tag,error='actual O halfkick momentum'))
                    if not np.array_equal(data['q'],trajectory['q_mid'][-1]):issues.append(dict(tag=tag,error='actual O midpoint q'))
                supplied=record['supplied_noise_sha256'];method=step['method']
                if supplied.get('z')!=step['z_sha256']:issues.append(dict(tag=tag,error='actual long/full Gaussian'))
                if method=='A' and (supplied.get('ef')!=step['eta_forward_sha256'] or supplied.get('er')!=step['eta_reverse_sha256']):issues.append(dict(tag=tag,error='actual independent A edge Gaussian'))
    a.check('all_saved_dense_vs_actual_Krylov_outputs_recompute_at_strict2e8_threshold',not issues,
            dict(issues=issues[:100],issue_count=len(issues),actual_probe_rows=rows_total,max_recomputed_relative_error=max_error,
                 selected_ranks=ranks,scope='finite saved states/probes; no uniform-rank or full Gibbs proof'))


def artifacts_checks(a):
    figures=[];issues=[]
    for png in sorted((HERE/'figures').glob('*.png')):
        svg=png.with_suffix('.svg')
        if png.read_bytes()[:8]!=b'\x89PNG\r\n\x1a\n':issues.append(str(png.name))
        if not svg.exists():issues.append('missing '+svg.name)
        else:
            try:
                if not ET.parse(svg).getroot().tag.endswith('svg'):issues.append('invalid '+svg.name)
            except ET.ParseError:issues.append('unparseable '+svg.name)
        figures.append(dict(png=png.name,svg=svg.name,png_sha256=sha(png),svg_sha256=sha(svg) if svg.exists() else None))
    a.check('two_or_more_actual_PNG_SVG_figure_pairs',len(figures)>=2 and not issues,dict(figures=figures,issues=issues))
    report=HERE/'report.md';readme=HERE/'README.md'
    a.check('Chinese_report_and_reproduction_commands_saved',report.exists() and report.stat().st_size>500
            and bool(re.search('[\u4e00-\u9fff]',report.read_text())) and readme.exists() and 'python3' in readme.read_text())


def replay_checks(a):
    from dynamics import PilotSystem
    params=read_json(RESULTS/'spatial_selection.json')['params'];ranks=read_json(RESULTS/'ranks_final.json')
    system=PilotSystem(params);issues=[];records=[]
    for idx in range(3):
        for method in ('A','B'):
            label=f'production_{idx}_{method}';meta=read_json(RESULTS/f'{label}.json')
            with np.load(INPUTS/f'{label}.npz',allow_pickle=False) as d:q,p,q1,p1=(d['q_unwrapped'][0].copy(),d['p'][0].copy(),d['q_unwrapped'][1].copy(),d['p'][1].copy())
            qr,pr,row=system.step(q,p,method,**ranks[method],rng=np.random.default_rng(meta['seed']),dt=.005)
            eq=float(np.max(np.abs(qr-q1)));ep=float(np.max(np.abs(pr-p1)))
            if not close(qr,q1,rtol=1e-12,atol=1e-12) or not close(pr,p1,rtol=1e-12,atol=1e-12):issues.append(label)
            records.append(dict(run=label,max_q_abs_error=eq,max_p_abs_error=ep,
                                q_bitwise_equal=np.array_equal(qr,q1),p_bitwise_equal=np.array_equal(pr,p1)))
    a.check('independent_local_replay_six_first_production_steps_after_timing',not issues,dict(issues=issues,replay=records))


def manifest():
    # Exclude circular/self-changing audit files and regenerate-able caches.
    out={}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(HERE)
        if any(x in ('__pycache__','.build','build') for x in rel.parts):continue
        if path.name=='independent_audit.json' or path.name.startswith('independent_audit') and path.suffix=='.log':continue
        if path.suffix=='.tmp':continue
        out[str(rel)]=sha(path)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    if not (RESULTS/'timing_done.json').exists():
        raise SystemExit('Refusing audit before independent moving-particle timing is complete')
    audit=Audit()
    for name,fn in [('baseline',baseline_checks),('initials',initial_checks),('trajectories',trajectory_checks),
                    ('spatial',spatial_checks),('ranks',rank_checks),('artifacts',artifacts_checks)]:
        audit.guarded(name,lambda fn=fn:fn(audit))
    if args.replay:audit.guarded('independent_step_replay',lambda:replay_checks(audit))
    result=dict(status='PASS' if audit.checks and all(x['passed'] for x in audit.checks) else 'FAIL',
        timestamp_utc=datetime.now(timezone.utc).isoformat(),auditor='independent read/reconstruction module',
        branch=command('git','branch','--show-current'),base_sha=BASE,checks=audit.checks,
        actual_execution='local Mac; audit launched only after timing_done; no remote numerical work',
        stage_scope=[0,1],production_steps_required=1200,extra_timing_steps_required=480,
        replay_executed=bool(args.replay),evidence_sha256=manifest(),
        manifest_exclusions='audit JSON and its changing redirected log, pycache, regenerable compiled caches, temporary files',
        interpretation='Saved evidence and finite-state/probe checks; no long-time Gibbs proof, no final equal-dynamics A/B ranking; Stage2/3 not executed')
    save(RESULTS/'independent_audit.json',result)
    print(json.dumps(dict(status=result['status'],checks=len(audit.checks),failures=[x['name'] for x in audit.checks if not x['passed']]),indent=2),flush=True)
    raise SystemExit(0 if result['status']=='PASS' else 1)
