"""Independent reconstruction of matched frozen-accuracy and moving-cost evidence.

Run after every experiment and plot/report write is complete. This script
never changes prior evidence or launches trajectories during timing. Optional
--replay repeats only the first saved step of each matched run after timing.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from scipy import linalg,stats
from matched_common import HERE,REPO,BASE,LEGACY,RESULTS,INPUTS,LOGS,PARAMS,TARGETS,sha,array_sha,save,read_csv,js
from model import lj_force,rng_sha

PREFIX=str(HERE.relative_to(REPO))+'/'

def close(a,b,rtol=1e-10,atol=1e-12):
    return bool(np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=False))

def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()

def digest_hashes(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()

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
    snapshot=js(HERE/'baseline_snapshot.json');hashes=snapshot['tracked_file_sha256']
    tracked=git('ls-tree','-r','--name-only',BASE).splitlines()
    a.check('registered_PR10_base_and_complete1979_file_snapshot',snapshot['base_sha']==BASE
        and len(hashes)==1979 and set(hashes)==set(tracked),dict(base=BASE,snapshot_files=len(hashes),git_files=len(tracked)))
    changed=[n for n,h in hashes.items() if not (REPO/n).is_file() or sha(REPO/n)!=h]
    a.check('all_PR3_to_PR10_tracked_file_hashes_unchanged',not changed,changed)
    paths=git('diff',BASE,'--name-only').splitlines()
    paths+=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=REPO).decode().split('\0')
    outside=[n for n in paths if n and not n.startswith(PREFIX)]
    a.check('all_new_changes_confined_independent_experiment_directory',not outside,outside)
    a.check('local_Mac_environment_and_independent_requested_branch',platform.system()=='Darwin'
        and git('branch','--show-current')=='codex/baoab-ab-matched-accuracy',
        dict(platform=platform.platform(),branch=git('branch','--show-current')))
    parse=[]
    for path in HERE.glob('*.py'):
        try:ast.parse(path.read_text(),filename=str(path))
        except SyntaxError as e:parse.append(str(e))
    a.check('all_new_sources_parse',not parse,parse)
    counts={}
    for name,minimum in [('baseline_regressions.log',81),('test_matched_dynamics.log',10),
                         ('test_frozen_accuracy.log',6),('test_stochastic_validation.log',5),
                         ('new_tests_final.log',25)]:
        path=LOGS/name;text=path.read_text() if path.exists() else ''
        found=re.findall(r'Ran (\d+) tests? in ([0-9.]+)s',text)
        passed=bool(found) and int(found[-1][0])>=minimum and text.rstrip().endswith('OK')
        counts[name]=int(found[-1][0]) if found else None
        a.check(name+'_passed',passed,dict(count=counts[name],minimum=minimum))
    a.check('fixed_declared_kernel_spatial_parameters_and_dt',PARAMS['N']==256 and PARAMS['rho']==.75
        and PARAMS['kappa']==.7 and close(PARAMS['c'],.5*1.3*np.exp(.7*1.3),rtol=1e-15,atol=0)
        and close(PARAMS['L'],(256/.75)**(1/3),rtol=1e-15)
        and all(PARAMS[k]==v for k,v in dict(xi=.65,sR=4.7,sF=4.7,M=40,p=10).items()),PARAMS)

def domain_checks(a):
    domain=js(RESULTS/'domain.json');records=domain['states'];issues=[]
    initial=[];visited=[]
    for r in records:
        path=HERE/r['file'];source=REPO/r['source']
        if sha(path)!=r['file_sha256'] or sha(source)!=r['source_sha256']:issues.append(r['label']+': source/file SHA')
        with np.load(path,allow_pickle=False) as d:q,p=d['q'],d['p']
        if q.shape!=(256,3) or p.shape!=q.shape or not np.isfinite(q).all() or not np.isfinite(p).all():issues.append(r['label']+': invalid arrays')
        if array_sha(q)!=r['q_sha256'] or array_sha(p)!=r['p_sha256']:issues.append(r['label']+': input array SHA')
        if r['kind']=='initial':
            initial.append(r)
            with np.load(source,allow_pickle=False) as d:
                if not np.array_equal(q,d['q']) or not np.array_equal(p,d['p']):issues.append(r['label']+': canonical start mismatch')
        else:
            visited.append(r)
            with np.load(source,allow_pickle=False) as d:
                index=int(r['step'])-1
                ph=d['p'][index]+.0025*lj_force(np.mod(d['q_unwrapped'][index],PARAMS['L']),PARAMS['L'])[0]
                if not np.array_equal(q,d['q_mid'][index]) or not close(p,ph,rtol=0,atol=1e-13):issues.append(r['label']+': actual midpoint/halfkick input')
    scan=read_csv(RESULTS/'prior_midpoint_distance_scan.csv')
    if len(scan)!=1200 or not close(min(r['min_pair_distance'] for r in scan),domain['global_min_midpoint_distance']):issues.append('minimum-distance scan')
    a.check('reuse_three_distinct_hash_checked_canonical_starts_and_six_actual_midpoints_plus_extreme',not issues
        and len(initial)==3 and len({r['preparation_seed'] for r in initial})==3 and len(visited)>=6
        and len(records)>=9 and domain['preparation_reused'] and domain['canonical_preparation_steps_run']==0,
        dict(issues=issues,registered_states=len(records),initial_seeds=[r['preparation_seed'] for r in initial],
             scanned_actual_midpoints=len(scan),validation_scope='finite declared saved geometries; not a uniform moving-trajectory operator bound'))

def accuracy_checks(a):
    folder=RESULTS/'accuracy';selection=js(folder/'frozen_selection.json')
    rows=read_csv(folder/'frozen_errors.csv');domains=selection['states'];issues=[];recomputed=[]
    wanted={int(t['k']) for t in selection['targets'] if t['k'] is not None}
    # Recompute selected norms and the immediately preceding tested candidate.
    tested=sorted(map(int,selection['ks']))
    wanted|={tested[tested.index(k)-1] for k in list(wanted) if tested.index(k)>0}
    for domain in domains:
        refpath=folder/domain['reference_file']
        if sha(refpath)!=domain['reference_file_sha256']:issues.append(domain['label']+': dense ref file SHA')
        with np.load(refpath,allow_pickle=False) as d:
            data={k:d[k] for k in d.files}
        Q=data['Q'];RB=data['RB_reduced'];CB=data['CB_reduced'];dimension=3*len(data['q'])-3
        if Q.shape!=(768,765) or not close(Q.T@Q,np.eye(765),atol=2e-13):issues.append(domain['label']+': full COM basis')
        if np.linalg.norm(Q.T@np.tile(np.eye(3),(256,1)))>2e-12:issues.append(domain['label']+': COM basis null directions')
        if not close(data['GH'],data['GS']+data['GL'],atol=2e-12):issues.append(domain['label']+': full=near+long')
        U=data['full_U'];e=data['full_eig'];V=Q.T@U
        if not close(data['GH']@U,U*e,rtol=2e-11,atol=2e-11):issues.append(domain['label']+': exact full graph eigensystem')
        if not close(RB,(V*np.exp(-.005*e))@V.T,atol=3e-12) or not close(CB,(V*(-np.expm1(-.01*e)))@V.T,atol=3e-12):issues.append(domain['label']+': independent exact B mean/covariance')
        if array_sha(data['q'])!=domain['q_sha256'] or array_sha(data['p'])!=domain['p_sha256']:issues.append(domain['label']+': geometry SHA')
        if array_sha(data['GH'])!=domain['GH_sha256'] or array_sha(data['GL'])!=domain['GL_sha256']:issues.append(domain['label']+': graph SHA')
        mean_den=float(np.max(np.exp(-.005*data['full_eig'])))
        cov_den=float(np.max(-np.expm1(-.01*data['full_eig'])))
        for row in [r for r in rows if r['state']==domain['label']]:
            path=folder/row['maps_file']
            if sha(path)!=row['maps_file_sha256']:issues.append(path.name+': map file SHA')
            with np.load(path,allow_pickle=False) as m:
                A,C=m['RA_reduced'],m['CA_reduced']
                if A.shape!=(dimension,dimension) or C.shape!=A.shape:issues.append(path.name+': full COM matrix dimensions')
                if array_sha(A)!=row['RA_reduced_sha256'] or array_sha(C)!=row['CA_reduced_sha256']:issues.append(path.name+': actual reduced matrix SHA')
                if array_sha(RB)!=row['RB_reduced_sha256'] or array_sha(CB)!=row['CB_reduced_sha256']:issues.append(path.name+': exact B matrix SHA')
                if not close(C,np.eye(dimension)-A@A.T,atol=2e-12):issues.append(path.name+': ideal covariance identity')
                ma=float(np.max(np.abs(m['mean_difference_spectrum'])));ca=float(np.max(np.abs(m['cov_difference_spectrum'])))
                if not close(ma,row['mean_absolute_norm'],rtol=1e-8,atol=2e-14) or not close(ca,row['cov_absolute_norm'],rtol=1e-8,atol=2e-14):issues.append(path.name+': saved complete norm spectra')
                if int(row['k']) in wanted:
                    # Independent full SVD, including any antisymmetric roundoff.
                    ma=float(linalg.svdvals(A-RB,check_finite=False)[0]);ca=float(linalg.svdvals(C-CB,check_finite=False)[0])
                    em=ma/max(mean_den,1e-300);ec=ca/max(cov_den,1e-300)
                    if not close(em,row['e_mean'],rtol=3e-4,atol=3e-13) or not close(ec,row['e_cov'],rtol=3e-4,atol=3e-13):issues.append(path.name+': independent SVD norm')
                    recomputed.append(dict(state=domain['label'],k=int(row['k']),full_COM_dimension=dimension,e_mean=em,e_cov=ec,norm='independent full svdvals'))
                if not close(row['mean_denominator'],mean_den,rtol=1e-13) or not close(row['cov_denominator'],cov_den,rtol=1e-13):issues.append(path.name+': declared normalization denominator')
                if row['norm_scope']!='full 3N-3 COM subspace, not sampled probes':issues.append(path.name+': norm scope')
    for target in selection['targets']:
        eps=target['epsilon_frozen'];passing=[]
        for k in tested:
            batch=[r for r in rows if int(r['k'])==k]
            if len(batch)==len(domains) and all(max(r['e_mean'],r['e_cov'])<=eps for r in batch):passing.append(k)
        expected=min(passing) if passing else None
        if target['k']!=expected:issues.append('target minimum actually tested k')
    a.check('full_COM_mean_covariance_operator_norms_and_minimum_actual_k_certification',not issues
        and len(rows)==len(domains)*len(tested) and len(domains)>=9,
        dict(issues=issues[:100],issue_count=len(issues),actual_maps=len(rows),states=len(domains),tested_ks=tested,
            independently_recomputed_selected_and_preceding_norms=recomputed,targets=selection['targets'],
            scope='ideal frozen operators on registered domain only; finite Lanczos and dynamic timing checked separately'))

def stochastic_checks(a):
    folder=RESULTS/'stochastic';selection=js(folder/'stochastic_selection.json');issues=[];paths=0;actions=0
    for path in sorted(folder.glob('path_check_*.json')):
        record=js(path);rawpath=folder/path.name.replace('path_check_','path_outputs_').replace('.json','.npz')
        with np.load(rawpath,allow_pickle=False) as d:raw={k:d[k] for k in d.files}
        method,k=record['method'],int(record['k']);rng=np.random.default_rng(record['seed'])
        if rng.bit_generator.state!=record['rng_state_before']:issues.append(path.name+': initial RNG')
        refpath=RESULTS/'accuracy'/f"frozen_refs_{record['state']}.npz"
        with np.load(refpath,allow_pickle=False) as ref:E=len(ref['edge_i'])
        for l,sub in enumerate(record['substeps']):
            z=rng.standard_normal(raw['p'].shape);ef=er=None
            if method=='A':ef=rng.standard_normal((E,3));er=rng.standard_normal((E,3))
            if array_sha(z)!=sub['z_sha256'] or (method=='A' and (array_sha(ef)!=sub['ef_sha256'] or array_sha(er)!=sub['er_sha256'])):issues.append(path.name+': per-substep independent Gaussian SHA')
            previous=raw['p'] if l==0 else raw['actual_substeps'][l-1]
            if array_sha(previous)!=sub['p_input_sha256'] or array_sha(raw['actual_substeps'][l])!=sub['p_actual_sha256'] or array_sha(raw['ideal_substeps'][l])!=sub['p_ideal_sha256']:issues.append(path.name+': actual intermediate path SHA')
            if sub['sub_dt']!=.005/k or sub['negative_ritz_count']!=0:issues.append(path.name+': actual inner dt/Ritz')
        if rng.bit_generator.state!=record['rng_state_after']:issues.append(path.name+': final RNG')
        err=float(np.linalg.norm(raw['actual']-raw['ideal'])/max(np.linalg.norm(raw['ideal']),np.finfo(float).tiny))
        if not close(err,record['coupled_relative_error'],rtol=1e-8,atol=1e-16) or err>record['full_path_budget'] or not record['passed']:issues.append(path.name+': composed own-reference path error')
        for row in record['action_checks']:
            prefix=f"sub{row['substep']}_{row['kind']}_"
            inp,exact,finite=(raw[prefix+n] for n in ('input','exact','finite'))
            err=float(np.linalg.norm(finite-exact)/max(np.linalg.norm(exact),np.finfo(float).tiny))
            if err>row['budget'] or not close(err,row['relative_error'],rtol=1e-8,atol=1e-16):issues.append(path.name+': actual action error')
            for key,arr in [('input',inp),('exact',exact),('actual',finite)]:
                if array_sha(arr)!=row[key+'_sha256']:issues.append(path.name+': action raw SHA')
            actions+=1
        paths+=1
    a.check('finite_Lanczos_composed_paths_actual_substep_inputs_and_independent_noise_reconstruct',not issues
        and selection['all_checks_passed'] and not selection['failures'] and paths>0,
        dict(issues=issues[:100],issue_count=len(issues),paths=paths,actual_action_checks=actions,
            selected_ranks=selection['by_k'],scope='own-method coupled paths, never raw A versus B output distance'))

def mc_checks(a):
    folder=RESULTS/'stochastic';marker=js(folder/'mc_done.json');rows=read_csv(folder/'mc_directional.csv')
    state=marker['representative_state'];issues=[];checked=0
    from stochastic_validation import SEED
    with np.load(RESULTS/'accuracy'/f'frozen_refs_{state}.npz',allow_pickle=False) as d:ref={k:d[k] for k in d.files}
    Q,U,eig=ref['Q'],ref['full_U'],ref['full_eig'];N=len(ref['q']);dimension=3*N
    positive=np.flatnonzero(eig>max(float(eig[-1]),1.)*1e-12);first=int(positive[0]) if len(positive) else 0
    rng=np.random.default_rng(SEED+700000);random=rng.normal(size=dimension).reshape(-1,3);random=(random-random.mean(0)).ravel();random/=np.linalg.norm(random)
    probes=dict(slow=U[:,first],fast=U[:,-1],random=random,COM=np.tile([1.,0.,0.],N)/np.sqrt(N))
    for item in marker['manifest']:
        rawpath=HERE/item['file']
        with np.load(rawpath,allow_pickle=False) as d:data={k:d[k] for k in d.files}
        actual,ideal=data['actual'],data['ideal'];n=len(actual);method,k,phase=item['method'],int(item['k']),item['phase']
        if array_sha(actual)!=item['actual_sha256'] or array_sha(ideal)!=item['ideal_sha256']:issues.append(rawpath.name+': sample SHA')
        stream=js(folder/f'mc_streams_{state}_{method}_k{k}_{phase}.json')
        for j,r in enumerate(stream):
            # ou_core.project preserves the generated (N,3) shape in both
            # phases. NPZ sample rows are flattened only for storage; the
            # per-sample SHA intentionally includes the original shape.
            shape=(N,3)
            if int(data['seed'][j])!=r['seed'] or array_sha(data['p_inputs'][j].reshape(shape))!=r['input_momentum_sha256']:
                issues.append(rawpath.name+': sample seed/input SHA')
            if array_sha(actual[j].reshape(shape))!=r['actual_sha256'] or array_sha(ideal[j].reshape(shape))!=r['ideal_sha256']:
                issues.append(rawpath.name+': actual/own-reference sample SHA')
        if method=='B':R,C=ref['RB'],ref['CB']
        else:
            with np.load(RESULTS/'accuracy'/f'frozen_maps_{state}_k{k}.npz',allow_pickle=False) as d:
                R=np.eye(dimension)-Q@Q.T+Q@d['RA_reduced']@Q.T;C=Q@d['CA_reduced']@Q.T
        for row in [r for r in rows if r['method']==method and int(r['k'])==k and r['phase']==phase]:
            v=probes[row['direction']];x=actual@v;y=ideal@v;center=x-x.mean();critical=float(stats.t.ppf(.975,n-1))
            target_mean=float(v@R@ref['p'].ravel()) if phase=='conditional' else 0.
            target_var=float(v@C@v) if phase=='conditional' else float(v@(Q@Q.T)@v)
            if row['direction']=='COM':target_mean=float(v@ref['p'].ravel()) if phase=='conditional' else 0.;target_var=0.
            mean=float(x.mean());mse=float(x.std(ddof=1)/np.sqrt(n));var=float(x.var(ddof=1));vse=float((center**2).std(ddof=1)/np.sqrt(n))
            pm=x-y;ps=(x-target_mean)**2-(y-target_mean)**2
            values=dict(mean=mean,mean_se=mse,mean_ci_low=mean-critical*mse,mean_ci_high=mean+critical*mse,
                variance=var,variance_se=vse,variance_ci_low=max(0.,var-critical*vse),variance_ci_high=var+critical*vse,
                fourth_moment=float((center**4).mean()),fourth_se=float((center**4).std(ddof=1)/np.sqrt(n)),
                ideal_mean=target_mean,ideal_variance=target_var,
                paired_mean_error=float(pm.mean()),paired_mean_se=float(pm.std(ddof=1)/np.sqrt(n)),
                paired_second_error=float(ps.mean()),paired_second_se=float(ps.std(ddof=1)/np.sqrt(n)))
            values.update(paired_mean_ci_low=values['paired_mean_error']-critical*values['paired_mean_se'],
                paired_mean_ci_high=values['paired_mean_error']+critical*values['paired_mean_se'],
                paired_second_ci_low=values['paired_second_error']-critical*values['paired_second_se'],
                paired_second_ci_high=values['paired_second_error']+critical*values['paired_second_se'])
            if row['n_samples']!=n or any(not close(row[key],value,rtol=1e-8,atol=2e-12) for key,value in values.items()):issues.append(rawpath.name+': raw MC moments/SE/CI '+row['direction'])
            checked+=1
    a.check('actual_directional_MC_raw_moments_SE_and_pointwise_CIs_recompute',not issues and checked==len(rows)
        and marker['samples_per_method_phase']>=4,
        dict(issues=issues[:100],issue_count=len(issues),actual_direction_rows=checked,
            actual_samples_per_method_phase=marker['samples_per_method_phase'],
            scope='representative finite directional MC; wide pointwise CIs, no covariance-operator or Gibbs proof'))

def audit_run(a,label,expected_steps):
    meta=js(RESULTS/f'{label}.json');rows=read_csv(RESULTS/f'{label}_steps.csv');sub=read_csv(RESULTS/f'{label}_substeps.csv')
    states=js(RESULTS/f'{label}_rng.json');path=INPUTS/f'{label}.npz';method,k=meta['method'],int(meta['k']);issues=[]
    with np.load(path,allow_pickle=False) as d:data={name:d[name] for name in d.files}
    q,p,mid,L=data['q_unwrapped'],data['p'],data['q_mid'],float(data['L'])
    if q.shape!=(expected_steps+1,256,3) or p.shape!=q.shape or mid.shape!=(expected_steps,256,3):issues.append('trajectory array dimensions')
    if len(rows)!=expected_steps or len(sub)!=expected_steps*k or len(states['states'])!=expected_steps:issues.append('outer/inner/state counts')
    if method=='B' and k!=1:issues.append('B incorrectly repeated')
    if not all(np.isfinite(x).all() for x in (q,p,mid,data['q'])) or not np.array_equal(np.mod(q,L),data['q']):issues.append('finite/wrapped state')
    if sha(path)!=meta['file_sha256']:issues.append('trajectory file SHA')
    for name,arr in [('start_q',q[0]),('start_p',p[0]),('end_q',q[-1]),('end_p',p[-1])]:
        if array_sha(arr)!=meta[name+'_sha256']:issues.append(name+' SHA')
    original=LEGACY/'inputs'/f"initial_{meta['initial_index']}.npz"
    with np.load(original,allow_pickle=False) as d:
        if not np.array_equal(q[0],d['q_unwrapped']) or not np.array_equal(p[0],d['p']):issues.append('same canonical start control')
    rng=np.random.default_rng(meta['seed']);max_mid=0.;max_drift=0.;max_com=0.;max_phase=0.;tokens=[]
    for j,row in enumerate(rows):
        where=f'step{j+1}'
        rr=[r for r in sub if int(r['outer_step'])==j+1]
        if len(rr)!=k or [r['substep'] for r in rr]!=list(range(1,k+1)):issues.append(where+': inner indexing')
        if states['states'][j]['before']!=rng.bit_generator.state or row['rng_state_before_sha256']!=rng_sha(rng):issues.append(where+': RNG before')
        hashes=[]
        for inner in rr:
            before=rng_sha(rng);z=rng.standard_normal((256,3));ef=er=None
            if method=='A':ef=rng.standard_normal((int(row['edge_count']),3));er=rng.standard_normal((int(row['edge_count']),3))
            h=dict(substep=int(inner['substep']),z_sha256=array_sha(z),eta_forward_sha256=array_sha(ef) if ef is not None else '',eta_reverse_sha256=array_sha(er) if er is not None else '')
            for name,value in h.items():
                if inner[name]!=(None if value=='' else value):issues.append(where+': independent inner Gaussian '+name)
            if inner['rng_state_before_sha256']!=before or inner['rng_state_after_sha256']!=rng_sha(rng):issues.append(where+': inner RNG state')
            if inner['dt_inner']!=.005/k or inner['dt_outer']!=.005 or inner['substeps']!=k or inner['q_mid_sha256']!=row['q_mid_sha256']:issues.append(where+': inner frozen interval/graph')
            if inner['r_d']!=row['r_d'] or inner['r_n']!=row['r_n'] or inner['negative_ritz_count']!=0:issues.append(where+': inner actual ranks/Ritz')
            hashes.append(h)
        if digest_hashes(hashes)!=row['gaussian_substeps_sha256']:issues.append(where+': ordered Gaussian sequence SHA')
        if states['states'][j]['after']!=rng.bit_generator.state or row['rng_state_after_sha256']!=rng_sha(rng):issues.append(where+': RNG after')
        for name,arr in [('q_before',q[j]),('q_after',q[j+1]),('p_before',p[j]),('p_after',p[j+1]),('q_mid',mid[j])]:
            if array_sha(arr)!=row[name+'_sha256']:issues.append(where+': '+name+' SHA')
        force0=lj_force(np.mod(q[j],L),L)[0];ph=p[j]+.0025*force0;qm=q[j]+.0025*ph
        err=float(np.max(np.abs(np.mod(qm,L)-mid[j])));max_mid=max(max_mid,err)
        if err>2e-13:issues.append(where+': actual midpoint')
        force1,U,separation=lj_force(np.mod(q[j+1],L),L);po=p[j+1]-.0025*force1
        err=float(np.max(np.abs(qm+.0025*po-q[j+1])));max_drift=max(max_drift,err)
        if err>2e-13:issues.append(where+': second drift/kick')
        obs=dict(Tkin=float(np.sum((p[j+1]-p[j+1].mean(0))**2)/(3*256-3)),U_per_particle=U/256,
            min_pair_distance=separation,momentum_norm=float(np.linalg.norm(p[j+1].sum(0))),
            q_motion_rms=float(np.sqrt(np.mean((q[j+1]-q[j])**2))))
        if any(not close(row[name],value) for name,value in obs.items()):issues.append(where+': actual observables')
        if obs['q_motion_rms']<=0 or separation<=0 or obs['momentum_norm']>1e-8:issues.append(where+': movement/sanity/COM')
        max_com=max(max_com,obs['momentum_norm'])
        if row['graph_builds_this_step']!=1 or row['q_dependent_setup_rebuilt'] is not True or row['fft_parameter_table_reused'] is not True:issues.append(where+': one graph per midpoint')
        if row['ou_calls']!=k or row['long_ou_calls']!=(k if method=='A' else 0) or row['full_ou_calls']!=int(method=='B'):issues.append(where+': actual OU call count')
        if method=='A':
            if row['coefficient_tau']!=.005/(2*k) or row['coefficient_sets_this_step']!=1 or row['edge_eigh_builds_this_step']!=1:issues.append(where+': A setup once/inner tau')
            if row['coefficient_cache_initial_size']!=0 or row['coefficient_first_prepare_hit'] is not False or row['coefficient_cache_size']!=1:issues.append(where+': coefficient invalidation')
            if row['coefficient_prepare_calls']!=2*k or row['coefficient_cache_hits']!=2*k or row['active_backend']!='compiled':issues.append(where+': actual C sweeps/cache hits')
        elif row['edge_count']!=0 or row['coefficient_sets_this_step']!=0:issues.append(where+': extra B setup')
        major=sum(row[name] for name in ('t_dynamic_setup_s','t_LJ_forces_s','t_rng_s','t_ou_wall_s','t_baoab_other_s'))
        residual=abs(major-row['t_baoab_s']);max_phase=max(max_phase,residual)
        if residual>1e-12 or row['t_baoab_other_s']<-1e-9:issues.append(where+': disjoint phase sum')
        for outer,inside in [('t_ou_wall_s','t_ou_wall_s'),('t_rng_s','t_rng_s'),('t_pair_forward_s','t_pair_forward_s'),
                             ('t_pair_reverse_s','t_pair_reverse_s'),('t_lanczos_damp_s','t_lanczos_damp_s'),('t_lanczos_noise_s','t_lanczos_noise_s'),
                             ('t_reorth_s','t_reorth_s'),('t_ritz_eigh_s','t_ritz_eigh_s'),('t_ou_fft_s','t_fft_s'),('matvec_count','matvec_count')]:
            if not close(row[outer],sum(r[inside] for r in rr),rtol=1e-12,atol=1e-12):issues.append(where+': aggregate '+outer)
        if row['r_d_used']>row['r_d'] or row['r_n_used']>row['r_n'] or row['rss_bytes']<=0 or row['peak_rss_bytes']<=0:issues.append(where+': actual rank/memory')
        tokens.append(row['q_mid_sha256'])
    if len(set(tokens))!=expected_steps:issues.append('nonunique moving midpoint')
    if not close(meta['full_BAOAB_mean_s'],np.mean([r['t_baoab_s'] for r in rows]),atol=1e-14) or not close(meta['compute_wall_s'],sum(r['t_baoab_s'] for r in rows),atol=1e-12):issues.append('actual full wall summaries')
    a.check(label+'_RNG_complete_subcycling_moving_BAOAB_and_clock_reconstruction',not issues,
        dict(issues=issues[:100],issue_count=len(issues),outer_steps=len(rows),inner_calls=len(sub),method=method,k=k,
            max_qmid_error=max_mid,max_second_drift_error=max_drift,max_COM_norm=max_com,max_phase_residual=max_phase))
    return rows,sub,meta

def timing_checks(a):
    marker=js(RESULTS/'timing_done.json');curve=js(RESULTS/'cost_curve_done.json')
    selection=js(RESULTS/'accuracy/frozen_selection.json');ranks=js(RESULTS/'stochastic/stochastic_selection.json')
    actual=[];inner=[];metas=[]
    for item in marker['runs']:
        if item.get('status')=='accuracy_not_met':continue
        rr,ss,meta=audit_run(a,item['run'],20);actual.extend(rr);inner.extend(ss);metas.append(meta)
        target=next(t for t in selection['targets'] if t['epsilon_frozen']==meta['epsilon_frozen'])
        expected_k=int(target['k']) if meta['method']=='A' else 1
        if meta['k']!=expected_k or meta['ranks']!=ranks['by_k'][str(target['k'])][meta['method']]:
            a.check(meta['run']+'_selected_k_and_ranks_enabled',False,meta)
    extra=[];extra_inner=[]
    for item in curve['runs']:
        rr,ss,meta=audit_run(a,item['run'],3);extra.extend(rr);extra_inner.extend(ss)
    a.check('actual720_matched_dynamic_steps_plus54_nonselected_curve_controls',len(actual)==720
        and len(extra)==54 and marker['actual_moving_steps']==720 and curve['actual_extra_moving_steps']==54
        and len(inner)==marker['actual_inner_O_calls'] and len(extra_inner)==curve['actual_inner_O_calls'],
        dict(matched_steps=len(actual),matched_inner_calls=len(inner),extra_curve_steps=len(extra),extra_inner_calls=len(extra_inner)))
    aggregate=read_csv(RESULTS/'timing_steps.csv');aggregate_inner=read_csv(RESULTS/'timing_substeps.csv')
    key=('run','step','q_mid_sha256','gaussian_substeps_sha256','k','t_baoab_s')
    a.check('aggregate_raw_timing_CSV_matches_per_run_actual_evidence',len(aggregate)==len(actual)
        and all(all(x[n]==y[n] for n in key) for x,y in zip(aggregate,actual)) and len(aggregate_inner)==len(inner))
    rounds=read_csv(RESULTS/'timing_rounds.csv');issues=[]
    for target_index,target in enumerate(sorted(selection['targets'],key=lambda t:t['epsilon_frozen'])):
        order_rng=np.random.default_rng(marker['seed']+target_index)
        for r in range(6):
            order=','.join(order_rng.permutation(['A','B']))
            row=next(x for x in rounds if x['epsilon_frozen']==target['epsilon_frozen'] and x['round']==r)
            aa=next(m for m in metas if m['epsilon_frozen']==target['epsilon_frozen'] and m['round']==r and m['method']=='A')
            bb=next(m for m in metas if m['epsilon_frozen']==target['epsilon_frozen'] and m['round']==r and m['method']=='B')
            if row['order']!=order or aa['initial_index']!=r%3 or bb['initial_index']!=r%3:issues.append('randomized paired round control')
            if not close(row['A_mean_s'],aa['full_BAOAB_mean_s'],atol=1e-14) or not close(row['B_mean_s'],bb['full_BAOAB_mean_s'],atol=1e-14):issues.append('raw round mean')
            if not close(row['A_over_B'],aa['full_BAOAB_mean_s']/bb['full_BAOAB_mean_s'],atol=1e-14):issues.append('actual paired cost ratio')
    a.check('same_start_interleaved_round_order_and_actual_paired_ratios',not issues and len(rounds)==18,issues)
    table=js(RESULTS/'timing_table.json')
    a.check('single_shared_parameter_table_and_every_outer_graph_rebuilt',table['parameter_table_reused'] and table['graph_rebuilt_each_step']
        and table['physics']==PARAMS and all(r['table_sha256']==table['sha256'] for r in actual)
        and [r['operator_build_index'] for r in actual]==list(range(1,len(actual)+1)),table)

def endpoint_checks(a):
    marker=js(RESULTS/'post_timing_validation.json');records=marker['records'];issues=[];recomputed=[]
    selection=js(RESULTS/'accuracy/frozen_selection.json')
    Q=np.kron(linalg.helmert(256,full=False).T,np.eye(3))
    for record in records:
        label=record['run'];path=HERE/record['file']
        if sha(path)!=record['file_sha256']:issues.append(label+': endpoint file SHA')
        with np.load(path,allow_pickle=False) as d:data={k:d[k] for k in d.files}
        with np.load(INPUTS/f'{label}.npz',allow_pickle=False) as d:
            q0=d['q_unwrapped'][-2];p0=d['p'][-2]
            ph=p0+.0025*lj_force(np.mod(q0,PARAMS['L']),PARAMS['L'])[0]
            if not np.array_equal(data['q'],d['q_mid'][-1]) or not close(data['p_half'],ph,rtol=0,atol=1e-13):issues.append(label+': actual new O state')
        for key,field in [('q','q_mid_sha256'),('p_half','p_half_sha256'),('GH','GH_sha256'),('GL','GL_sha256')]:
            if array_sha(data[key])!=record[field]:issues.append(label+': endpoint '+key+' SHA')
        target=next(r for r in selection['targets'] if r['epsilon_frozen']==record['epsilon_frozen'])
        if record['k']!=target['k'] or record['step']!=20 or record['dimension_COM']!=765:issues.append(label+': selected k/domain')
        G=Q.T@data['GH']@Q;e,V=linalg.eigh((G+G.T)/2,check_finite=False)
        if e.min()<-256*np.finfo(float).eps*max(1.,float(np.abs(e).max())):issues.append(label+': materially negative full graph eigenvalue')
        e=np.maximum(e,0.)
        R=(V*np.exp(-.005*e))@V.T;C=(V*(-np.expm1(-.01*e)))@V.T
        A,CA=data['RA_reduced'],data['CA_reduced']
        if array_sha(A)!=record['RA_reduced_sha256'] or array_sha(CA)!=record['CA_reduced_sha256']:issues.append(label+': endpoint A maps SHA')
        if not close(CA,np.eye(765)-A@A.T,atol=2e-12):issues.append(label+': ideal A covariance relation')
        em=float(linalg.svdvals(A-R,check_finite=False)[0])/max(float(np.exp(-.005*e[0])),1e-300)
        ec=float(linalg.svdvals(CA-C,check_finite=False)[0])/max(float(np.max(-np.expm1(-.01*e))),1e-300)
        if not close(em,record['e_mean'],rtol=3e-4,atol=4e-13) or not close(ec,record['e_cov'],rtol=3e-4,atol=4e-13):issues.append(label+': independent new endpoint norms')
        actual_pass=max(em,ec)<=record['epsilon_frozen']
        if actual_pass!=record['passed']:issues.append(label+': endpoint accuracy classification')
        recomputed.append(dict(run=label,k=int(record['k']),epsilon_frozen=record['epsilon_frozen'],
            e_mean=em,e_cov=ec,passed=actual_pass,method='fresh GH reduced eigensolve and full SVD'))
    if marker['all_passed']!=all(r['passed'] for r in records):issues.append('endpoint summary classification')
    a.check('18_actual_new_endpoint_full_COM_norms_reconstruct_with_honest_classification',not issues
        and marker['sampled_new_midpoints']==18 and len(records)==18,
        dict(issues=issues,recomputed=recomputed,scope='18 sampled new midpoints; no uniform accuracy assertion for unsampled steps'))

def analysis_checks(a):
    result=js(RESULTS/'analysis.json');rounds=read_csv(RESULTS/'timing_rounds.csv')
    steps=read_csv(RESULTS/'timing_steps.csv');extra=read_csv(RESULTS/'cost_curve_extra_steps.csv')
    endpoints=js(RESULTS/'post_timing_validation.json')['records'];issues=[];checks=[]
    rng=np.random.default_rng(result['bootstrap_seed'])
    with np.load(RESULTS/'paired_bootstrap_samples.npz',allow_pickle=False) as saved:
        for target in sorted(result['targets'],key=lambda r:r['epsilon_frozen']):
            if target['k'] is None:
                if target['verdict']!='accuracy_not_met':issues.append('unmet target classified as comparable')
                continue
            eps=target['epsilon_frozen'];block=[r for r in rounds if r['epsilon_frozen']==eps]
            ratios=np.asarray([r['A_over_B'] for r in block]);indices=rng.integers(0,len(block),size=(20000,len(block)))
            bootstrap=np.exp(np.log(ratios[indices]).mean(axis=1));ratio=float(np.exp(np.log(ratios).mean()))
            lo,hi=map(float,np.quantile(bootstrap,[.025,.975]))
            if not np.array_equal(bootstrap,saved[f'eps{eps:g}']):issues.append('raw whole-paired-round bootstrap draws')
            if not close(target['ratio'],ratio,rtol=1e-14,atol=1e-14) or not close(target['ratio_ci_low'],lo,rtol=1e-14,atol=1e-14) or not close(target['ratio_ci_high'],hi,rtol=1e-14,atol=1e-14):issues.append('paired geometric-mean ratio/percentile CI')
            ep=[r for r in endpoints if r['epsilon_frozen']==eps]
            passed=bool(ep) and all(r['passed'] for r in ep)
            verdict='accuracy_not_met' if not passed else ('A_faster' if hi<1 else ('B_faster' if lo>1 else 'INCONCLUSIVE'))
            if target['verdict']!=verdict or target['sampled_endpoint_accuracy_passed']!=passed:issues.append('honest frozen-accuracy-matched conclusion')
            for method in ('A','B'):
                rr=[r for r in steps if r['epsilon_frozen']==eps and r['method']==method]
                if target[method+'_steps']!=len(rr) or not close(target[method+'_mean_ms'],1000*np.mean([r['t_baoab_s'] for r in rr]),atol=1e-10):issues.append('actual mean complete BAOAB cost')
                for field in ['t_dynamic_setup_s','t_LJ_forces_s','t_rng_s','t_ou_wall_s','t_pair_forward_s',
                              't_pair_reverse_s','t_lanczos_damp_s','t_lanczos_noise_s','t_fft_s','t_reorth_s',
                              't_edge_eigh_s','t_coefficient_setup_s','t_baoab_other_s']:
                    if not close(target[method+'_'+field+'_mean_ms'],1000*np.mean([r[field] for r in rr]),atol=1e-10):issues.append('actual '+method+' phase mean '+field)
            checks.append(dict(epsilon_frozen=eps,k=target['k'],ratio=ratio,CI_low=lo,CI_high=hi,
                verdict=verdict,whole_paired_round_blocks=len(block),bootstrap_replicates=len(bootstrap)))
    for record in result['cost_curve']:
        rr=[r for r in steps+extra if r['method']=='A' and r['k']==record['k']]
        if record['actual_steps']!=len(rr) or not close(record['full_mean_ms'],1000*np.mean([r['t_baoab_s'] for r in rr]),atol=1e-10):issues.append('cost curve from actual saved same-run data')
    a.check('paired_round_bootstrap_replicates_CIs_decisions_and_actual_cost_curves_recompute',not issues
        and result['old_PR10_timings_used_as_background_only'] is True and result['actual_matched_moving_steps']==720,
        dict(issues=issues,targets=checks,scope='pointwise conditional timing uncertainty over six local paired rounds; not physical-observable equivalence or a global hardware guarantee'))

def artifacts_checks(a):
    pairs=[];issues=[]
    for path in sorted((HERE/'figures').glob('*.png')):
        svg=path.with_suffix('.svg')
        if path.read_bytes()[:8]!=b'\x89PNG\r\n\x1a\n' or not svg.exists():issues.append(path.name)
        else:
            try:
                if not ET.parse(svg).getroot().tag.endswith('svg'):issues.append(svg.name)
            except ET.ParseError:issues.append(svg.name)
        pairs.append(path.name)
    a.check('three_or_more_actual_PNG_and_SVG_figure_pairs',len(pairs)>=3 and not issues,dict(png=pairs,issues=issues))
    report=HERE/'report.md';readme=HERE/'README.md'
    a.check('Chinese_report_and_reproduction_commands_saved',report.exists() and report.stat().st_size>1000
        and bool(re.search('[\u4e00-\u9fff]',report.read_text())) and readme.exists() and 'python3' in readme.read_text())

def replay_checks(a):
    from matched_dynamics import MatchedSystem
    marker=js(RESULTS/'timing_done.json');system=MatchedSystem(PARAMS);issues=[];records=[]
    # First round of each tier/method: all selected k plus each corresponding B.
    for meta in [m for m in marker['runs'] if m.get('round')==0]:
        with np.load(INPUTS/f"{meta['run']}.npz",allow_pickle=False) as d:
            q,p,q1,p1=(d['q_unwrapped'][0].copy(),d['p'][0].copy(),d['q_unwrapped'][1].copy(),d['p'][1].copy())
        qr,pr,row=system.step(q,p,meta['method'],**meta['ranks'],rng=np.random.default_rng(meta['seed']),dt=.005,k=int(meta['k']))
        eq=float(np.max(np.abs(qr-q1)));ep=float(np.max(np.abs(pr-p1)))
        if not close(qr,q1,rtol=1e-12,atol=1e-12) or not close(pr,p1,rtol=1e-12,atol=1e-12):issues.append(meta['run'])
        records.append(dict(run=meta['run'],k=meta['k'],method=meta['method'],max_q_abs_error=eq,max_p_abs_error=ep,
            q_bitwise_equal=np.array_equal(qr,q1),p_bitwise_equal=np.array_equal(pr,p1)))
    a.check('independent_local_firststep_replay_selected_k_and_B_after_timing',not issues and len(records)==6,dict(issues=issues,records=records))

def manifest():
    hashes={}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(HERE)
        if any(p in ('__pycache__','.build','build') for p in rel.parts) or path.suffix=='.tmp':continue
        if path.name=='independent_audit.json' or (path.name.startswith('independent_audit') and path.suffix=='.log'):continue
        hashes[str(rel)]=sha(path)
    return hashes

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    for needed in ('timing_done.json','cost_curve_done.json','stochastic/mc_done.json','post_timing_validation.json','analysis.json'):
        if not (RESULTS/needed).exists():raise SystemExit('Refusing audit before completed '+needed)
    a=Audit()
    for name,fn in [('baseline',baseline_checks),('domain',domain_checks),('accuracy',accuracy_checks),
                    ('stochastic',stochastic_checks),('MC',mc_checks),('timing',timing_checks),
                    ('new_endpoints',endpoint_checks),('paired_analysis',analysis_checks),('artifacts',artifacts_checks)]:
        a.guarded(name,lambda fn=fn:fn(a))
    if args.replay:a.guarded('firststep_replay',lambda:replay_checks(a))
    result=dict(status='PASS' if a.checks and all(r['passed'] for r in a.checks) else 'FAIL',
        timestamp_utc=datetime.now(timezone.utc).isoformat(),base_sha=BASE,branch=git('branch','--show-current'),
        actual_execution='local Mac only; audit guarded until timing,curve and MC markers exist',
        checks=a.checks,replay_executed=bool(args.replay),evidence_sha256=manifest(),
        manifest_exclusions='audit JSON and changing redirected audit log, bytecode/build caches and temporary files',
        scope='finite-domain ideal frozen-operator accuracy plus actual finite-Lanczos paths, finite MC and moving-particle cost; not equal physical-observable dynamics')
    save(RESULTS/'independent_audit.json',result)
    print(json.dumps(dict(status=result['status'],checks=len(a.checks),failures=[r['name'] for r in a.checks if not r['passed']]),indent=2),flush=True)
    raise SystemExit(0 if result['status']=='PASS' else 1)
