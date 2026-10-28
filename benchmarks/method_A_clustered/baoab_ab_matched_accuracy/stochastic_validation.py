"""Finite-Krylov and Monte Carlo evidence for the matched frozen OU pilot.

Ideal transition operator norms live in frozen_accuracy.py.  This module
checks actual vector-dependent Lanczos actions and each method against its
OWN coupled ideal transition.  Finite directions and MC do not certify an
operator norm or a physical-observable/Gibbs error bound.
"""
from __future__ import annotations
from matched_common import HERE,RESULTS,PARAMS,TARGETS,load_states,array_sha,save,write_csv,js
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import stats
from ou_core import lanczos_action,project,ou_a,ou_b
from protocol import DenseOU
from rank_validation import directions

SEED=202610280903
DT=.005
RANKS=(3,5,8,12,14,16,20,24,32,48,64,96,128,192,256,512,765)

def relative(actual,exact):
    return float(np.linalg.norm(np.asarray(actual)-np.asarray(exact))/max(np.linalg.norm(exact),np.finfo(float).tiny))

def target_records(selection):
    """Accept the frozen-accuracy result's explicit target records."""
    raw=selection.get('targets',selection.get('selections',selection.get('selected',[])))
    if isinstance(raw,dict):
        raw=[dict(v,epsilon_frozen=float(key)) if isinstance(v,dict) else dict(epsilon_frozen=float(key),k=v) for key,v in raw.items()]
    out=[]
    for r in raw:
        eps=float(r.get('epsilon_frozen',r.get('target',r.get('epsilon',0))))
        k=r.get('k',r.get('selected_k',r.get('substeps')))
        if k is not None and r.get('status','').lower() not in ('accuracy_not_met','not_met'):
            out.append(dict(epsilon_frozen=eps,k=int(k),ideal_selection=r))
    if not out:raise ValueError('No actually certified frozen-accuracy target/k records')
    return out

def prepare(q,params):
    from frozen_accuracy import prepare as make
    return make(q,params)

def prepare_state(state,params):
    label=state.get('label',state.get('name','state'))
    ref=RESULTS/'accuracy'/f'frozen_refs_{label}.npz'
    if ref.exists():
        from frozen_accuracy import load_prepared
        return load_prepared(ref,params)
    return prepare(state['q'],params)

def action_of(prepared,method):
    if 'actions' in prepared:return prepared['actions'][method]
    if method=='B':return prepared['op'].matvec
    from spatial_validation import long_action
    return lambda v:long_action(prepared['op'],v)

def exact_action(ref,v,dt,kind):
    v=np.asarray(v,float);x=project(v).ravel()
    values=np.exp(-dt*ref.eig) if kind=='damping' else np.sqrt(-np.expm1(-2*dt*ref.eig))
    return (ref.U@(values*(ref.U.T@x))).reshape(v.shape)

def budget_for(epsilon,k):
    return min(2e-8,float(epsilon)/(20*int(k)))

def validate_probes(prepared,q,p,method,k,kind,rank,budget,seed):
    rng=np.random.default_rng(int(seed));ref=prepared['full_ref' if method=='B' else 'long_ref']
    z=rng.standard_normal(np.asarray(p).shape)
    probe_p=np.asarray(p).ravel()
    if method=='A':
        ef=rng.standard_normal((len(prepared['edges'].i),3))
        probe_p=prepared['edges'].sweep(probe_p,ef,DT/(2*k))[0]
    probes=directions(q,ref,probe_p,z)
    action=action_of(prepared,method);h=DT if method=='B' else DT/k
    rows=[];arrays={}
    for name,v in probes.items():
        expected=exact_action(ref,v,h,kind);actual,diag=lanczos_action(action,v,rank,h,kind)
        error=relative(actual,expected)
        rows.append(dict(method=method,k=int(k),kind=kind,probe=name,rank_requested=int(rank),
            rank_used=diag['rank_used'],relative_error=error,budget=budget,passed=error<=budget,
            matvec_count=diag['matvec_count'],negative_ritz_count=diag['negative_ritz_count'],
            clipped_negative_count=diag['clipped_negative_count'],ritz_min_raw=diag['ritz_min_raw'],
            t_action_s=diag['t_total_s'],probe_sha256=array_sha(v),exact_sha256=array_sha(expected),actual_sha256=array_sha(actual)))
        prefix=f'{method}_k{k}_{kind}_{name}'
        arrays[prefix+'_probe']=v;arrays[prefix+'_exact']=expected;arrays[prefix+'_actual']=actual
    return rows,arrays

def path_check(prepared,p,method,k,r_d,r_n,seed,action_budget=None):
    """Actual FFT/Lanczos path vs same-method exact-long/full coupled path.

    All draws are newly generated for each substep.  Raw input/output arrays,
    PCG64 seed/recipe and every Gaussian hash make the path replayable without
    archiving k*E*3 edge draws repeatedly.  Pair sweeps use the previously
    independently tested exact 3x3 C implementation on both coupled paths.
    """
    if method not in ('A','B') or int(k)!=k or k not in (1,2,4,8,16,32,64):raise ValueError('Invalid registered method/k')
    if method=='B' and k!=1:raise ValueError('B is one full-dt OU call')
    rng=np.random.default_rng(int(seed));shape=np.asarray(p).shape
    actual=np.asarray(p,float).copy();ideal=actual.copy();sub=[];actions=[]
    raw=dict(p=np.asarray(p).copy());damping_inputs=[];actual_outputs=[];ideal_outputs=[]
    ref=prepared['full_ref' if method=='B' else 'long_ref'];h=DT if method=='B' else DT/k
    R,F,_=ref.matrices(h);loops=1 if method=='B' else int(k)
    seed_state_before=json.loads(json.dumps(rng.bit_generator.state));total_matvec=0;max_action=0.
    for l in range(loops):
        z=rng.standard_normal(shape);ef=er=None
        p_in=actual.copy()
        if method=='A':
            ef=rng.standard_normal((len(prepared['edges'].i),3));er=rng.standard_normal((len(prepared['edges'].i),3))
            damp_input=prepared['edges'].sweep(actual,ef,h/2)[0]
            ideal_forward=prepared['edges'].sweep(ideal,ef,h/2)[0]
            actual,diag=ou_a(action_of(prepared,'A'),actual,z,ef,er,prepared['edges'],h,r_d,r_n)
            ideal_middle=(R@ideal_forward.ravel()+F@z.ravel()).reshape(shape)
            ideal=prepared['edges'].sweep(ideal_middle,er,h/2,reverse=True)[0]
        else:
            damp_input=actual.copy()
            actual,diag=ou_b(action_of(prepared,'B'),actual,z,h,r_d,r_n)
            ideal=(R@ideal.ravel()+F@z.ravel()).reshape(shape)
        if action_budget is not None:
            for kind,vector,rank in [('damping',damp_input,r_d),('noise',z,r_n)]:
                expected=exact_action(ref,vector,h,kind)
                finite,dd=lanczos_action(action_of(prepared,method),vector,rank,h,kind)
                err=relative(finite,expected);max_action=max(max_action,err)
                actions.append(dict(method=method,k=k,substep=l,kind=kind,relative_error=err,
                    budget=action_budget,passed=err<=action_budget,rank_used=dd['rank_used'],
                    negative_ritz_count=dd['negative_ritz_count'],matvec_count=dd['matvec_count'],
                    input_sha256=array_sha(vector),exact_sha256=array_sha(expected),actual_sha256=array_sha(finite)))
                raw[f'sub{l}_{kind}_input']=vector.copy();raw[f'sub{l}_{kind}_exact']=expected;raw[f'sub{l}_{kind}_finite']=finite
        total_matvec+=diag['matvec_count']
        row=dict(substep=l,k=k,method=method,sub_dt=h,z_sha256=array_sha(z),
            ef_sha256=array_sha(ef) if ef is not None else None,er_sha256=array_sha(er) if er is not None else None,
            p_input_sha256=array_sha(p_in),p_actual_sha256=array_sha(actual),p_ideal_sha256=array_sha(ideal),
            coupled_relative_error=relative(actual,ideal),matvec_count=diag['matvec_count'],
            damp_rank_used=diag['damp_rank_used'],noise_rank_used=diag['noise_rank_used'],
            negative_ritz_count=diag['negative_ritz_count'],t_ou_s=diag['step_wall_s'])
        sub.append(row);damping_inputs.append(damp_input.copy());actual_outputs.append(actual.copy());ideal_outputs.append(ideal.copy())
    raw.update(actual=actual.copy(),ideal=ideal.copy(),damping_inputs=np.asarray(damping_inputs),
               actual_substeps=np.asarray(actual_outputs),ideal_substeps=np.asarray(ideal_outputs))
    summary=dict(method=method,k=k,r_d=r_d,r_n=r_n,seed=int(seed),bit_generator=type(rng.bit_generator).__name__,
        rng_state_before=seed_state_before,rng_state_after=json.loads(json.dumps(rng.bit_generator.state)),
        noise_recipe='for each substep: standard_normal(p.shape), then A standard_normal((E,3)) forward and reverse; no extra sqrt(dt) factor',
        coupled_relative_error=relative(actual,ideal),max_actual_input_action_error=max_action,
        actual_matvec_count=total_matvec,negative_ritz_count=sum(r['negative_ritz_count'] for r in sub),
        COM_error=float(np.linalg.norm(actual.reshape(-1,3).sum(0)-np.asarray(p).reshape(-1,3).sum(0))),
        substeps=sub,action_checks=actions,
        interpretation='Same-method finite-Lanczos vs coupled own ideal reference only; not raw A/B path bias')
    return summary,raw

def _select_one(prepared,q,p,method,k,budget,seed,state):
    chosen={};allrows=[];arrays={};dimension=np.asarray(p).size-3
    for kind,key in [('damping','r_d'),('noise','r_n')]:
        for rank in sorted(set(min(r,dimension) for r in RANKS if r<=max(dimension,3))):
            rr,aa=validate_probes(prepared,q,p,method,k,kind,rank,budget,seed)
            for r in rr:r['state']=state;r['phase']='rank_selection'
            allrows.extend(rr)
            print('rank',state,method,'k',k,kind,rank,'max',max(r['relative_error'] for r in rr),'budget',budget,flush=True)
            if all(r['passed'] for r in rr):chosen[key]=rank;arrays.update(aa);break
        else:raise RuntimeError(f'No rank passed {state} {method} k{k} {kind}')
    return chosen,allrows,arrays

def select_and_validate(states,params,selection,out,mc_samples=0):
    """Run all-domain selection, then recheck common ranks and whole paths."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    marker=out/'stochastic_selection.json'
    if marker.exists():raise FileExistsError('Preserve completed stochastic selection')
    targets=target_records(selection);selected_ks=sorted({r['k'] for r in targets})
    ks=sorted(set(selected_ks)|{1,2,4,8,16,32})
    eps_by_k={k:min((r['epsilon_frozen'] for r in targets if r['k']==k),default=min(r['epsilon_frozen'] for r in targets)) for k in ks}
    action_budgets={k:budget_for(eps_by_k[k],k) for k in ks}
    path_budgets={k:min(2e-8,eps_by_k[k]/10) for k in ks}
    slack_by_k={}
    for r in targets:
        ideal=r['ideal_selection'];slack=r['epsilon_frozen']-max(float(ideal.get('max_e_mean',0)),float(ideal.get('max_e_cov',0)))
        if slack<=0:raise ValueError('Selected ideal transition has no positive threshold slack')
        k=r['k'];slack_by_k[k]=min(slack_by_k.get(k,slack),slack)
        action_budgets[k]=min(action_budgets[k],slack/(40*k))
        path_budgets[k]=min(path_budgets[k],slack/8)
    B_budget=min(budget_for(r['epsilon_frozen'],1) for r in targets)
    globalA={str(k):dict(r_d=0,r_n=0) for k in ks};globalB=dict(r_d=0,r_n=0)
    all_attempts=[];state_selections=[];started=perf_counter()
    for index,state in enumerate(states):
        label=state.get('label',state.get('name',str(index)));q,p=np.asarray(state['q']),np.asarray(state['p'])
        prepared=prepare_state(state,params);arr=dict(q=q,p=p)
        b,rr,aa=_select_one(prepared,q,p,'B',1,B_budget,SEED+index*1000,label)
        all_attempts.extend(rr);arr.update(aa)
        for key in globalB:globalB[key]=max(globalB[key],b[key])
        perA={}
        for k in ks:
            if k not in selected_ks and index!=0:continue
            a,rr,aa=_select_one(prepared,q,p,'A',k,action_budgets[k],SEED+index*1000+k,label)
            all_attempts.extend(rr);arr.update(aa);perA[str(k)]=a
            for key in a:globalA[str(k)][key]=max(globalA[str(k)][key],a[key])
        state_selections.append(dict(state=label,A=perA,B=b))
        np.savez_compressed(out/f'rank_selection_outputs_{label}.npz',**arr)
        write_csv(out/'rank_attempts.csv',all_attempts)
        save(out/f'rank_selection_{label}.json',state_selections[-1])
    actual=[];failures=[]
    for index,state in enumerate(states):
        label=state.get('label',state.get('name',str(index)));q,p=np.asarray(state['q']),np.asarray(state['p'])
        prepared=prepare_state(state,params)
        for method,k in [('B',1)]+[('A',k) for k in ks]:
            if method=='A' and k not in selected_ks and index!=0:continue
            ranks=globalB if method=='B' else globalA[str(k)]
            eps=min(r['epsilon_frozen'] for r in targets) if method=='B' else eps_by_k[k]
            budget=B_budget if method=='B' else action_budgets[k]
            seed=SEED+100000+index*1000+k*10+(method=='B')
            result,raw=path_check(prepared,p,method,k,**ranks,seed=seed,action_budget=budget)
            result.update(state=label,epsilon_frozen=eps,per_action_budget=budget,
                full_path_budget=min(2e-8,eps/10) if method=='B' else path_budgets[k])
            passed=(result['max_actual_input_action_error']<=budget and result['coupled_relative_error']<=result['full_path_budget'] and result['negative_ritz_count']==0)
            result['passed']=passed;actual.append(result)
            if not passed:failures.append(dict(state=label,method=method,k=k,error=result['coupled_relative_error'],action=result['max_actual_input_action_error']))
            np.savez_compressed(out/f'path_outputs_{label}_{method}_k{k}.npz',q=q,**raw)
            save(out/f'path_check_{label}_{method}_k{k}.json',result)
            print('path',label,method,k,result['coupled_relative_error'],'passed',passed,flush=True)
    by_k={str(k):dict(A=globalA[str(k)],B=globalB,per_action_budget=action_budgets[k],
        full_path_budget=path_budgets[k],epsilon_frozen=eps_by_k[k],ideal_threshold_slack=slack_by_k.get(k),
        validation_domain=('all_registered_states' if k in selected_ks else 'representative_state_cost_curve_only'),
        validation_states=([s.get('label',s.get('name')) for s in states] if k in selected_ks else [states[0].get('label',states[0].get('name'))]),
        matched_accuracy_tier=k in selected_ks) for k in ks}
    result=dict(targets=[dict(r,A=globalA[str(r['k'])],B=globalB) for r in targets],by_k=by_k,
        B=globalB,B_action_budget=B_budget,states=[s.get('label',s.get('name')) for s in states],
        all_checks_passed=not failures,failures=failures,elapsed_s=perf_counter()-started,
        max_coupled_path_relative_error=max(r['coupled_relative_error'] for r in actual),
        max_actual_input_action_error=max(r['max_actual_input_action_error'] for r in actual),
        interpretation='Ideal full-operator norm certification is separate; finite-Lanczos evidence covers saved directions and actual coupled paths only')
    save(marker,result)
    if failures:raise RuntimeError('Actual full composed paths fail budget; see stochastic_selection failures before timing')
    if mc_samples:run_mc(states[0],params,result,out,n=mc_samples)
    return result

def directional_statistics(actual,ideal,mean,variance,temperature=1.):
    """Pointwise approximate moments and paired discrepancy SE/CI."""
    actual=np.asarray(actual,float);ideal=np.asarray(ideal,float);n=len(actual)
    if n<4:raise ValueError('Require at least four independent samples for MC uncertainty')
    critical=float(stats.t.ppf(.975,n-1));avg=float(actual.mean());center=actual-avg
    var=float(actual.var(ddof=1));mean_se=float(actual.std(ddof=1)/np.sqrt(n))
    variance_se=float((center**2).std(ddof=1)/np.sqrt(n));fourth=center**4
    paired=actual-ideal;paired_second=(actual-mean)**2-(ideal-mean)**2
    pm=float(paired.mean());pmse=float(paired.std(ddof=1)/np.sqrt(n))
    ps=float(paired_second.mean());psse=float(paired_second.std(ddof=1)/np.sqrt(n))
    return dict(n_samples=n,mean=avg,mean_se=mean_se,mean_ci_low=avg-critical*mean_se,mean_ci_high=avg+critical*mean_se,
        variance=var,variance_se=variance_se,variance_ci_low=max(0.,var-critical*variance_se),variance_ci_high=var+critical*variance_se,
        fourth_moment=float(fourth.mean()),fourth_se=float(fourth.std(ddof=1)/np.sqrt(n)),ideal_mean=float(mean),ideal_variance=float(variance),
        paired_mean_error=pm,paired_mean_se=pmse,paired_mean_ci_low=pm-critical*pmse,paired_mean_ci_high=pm+critical*pmse,
        paired_second_error=ps,paired_second_se=psse,paired_second_ci_low=ps-critical*psse,paired_second_ci_high=ps+critical*psse,
        variance_target_in_pointwise_CI=bool(max(0.,var-critical*variance_se)<=variance<=var+critical*variance_se),
        CI_scope='pointwise approximate95% Student-t/plug-in moment intervals; no multiplicity correction; paired second moment about known ideal mean includes finite-mean bias squared')

def verify_common_probes(states,params,ranks,out):
    """Recheck every registered direction using final cross-state common ranks.

    Taking a maximum selected rank is not used as an untested assumption of
    monotone finite-Lanczos error.  This separate pass checks the exact ranks
    that timing will execute, including nonselected cost-curve ranks only on
    their explicitly smaller representative validation domain.
    """
    out=Path(out);marker=out/'common_probe_done.json'
    if marker.exists():raise FileExistsError('Preserve completed common-rank probe evidence')
    allrows=[];issues=[];started=perf_counter()
    for index,state in enumerate(states):
        label=state.get('label',state.get('name',str(index)));prepared=prepare_state(state,params)
        arr=dict(q=state['q'],p=state['p'])
        methods=[('B',1,ranks['B'],ranks['B_action_budget'])]
        for key,item in sorted(ranks['by_k'].items(),key=lambda kv:int(kv[0])):
            if label in item['validation_states']:methods.append(('A',int(key),item['A'],item['per_action_budget']))
        for method,k,rr,budget in methods:
            for kind,key in [('damping','r_d'),('noise','r_n')]:
                rows,arrays=validate_probes(prepared,state['q'],state['p'],method,k,kind,rr[key],budget,SEED+index*1000+k)
                for row in rows:row.update(state=label,phase='final_common_rank_validation')
                allrows.extend(rows);arr.update(arrays)
                if not all(row['passed'] for row in rows):issues.extend([dict(state=label,method=method,k=k,kind=kind,probe=row['probe'],error=row['relative_error']) for row in rows if not row['passed']])
        np.savez_compressed(out/f'common_probe_outputs_{label}.npz',**arr)
        write_csv(out/'common_probe_checks.csv',allrows)
        print('Common final-rank probes',label,'rows',len(allrows),flush=True)
    result=dict(passed=not issues,issues=issues,rows=len(allrows),max_relative_error=max(row['relative_error'] for row in allrows),elapsed_s=perf_counter()-started)
    save(marker,result)
    if issues:raise RuntimeError('Final common ranks fail registered probes')
    return result

def run_mc(state,params,ranks,out,n=16):
    """Actual finite-Lanczos directional conditional/equilibrium MC.

    Uses one declared representative state and modest actual sample counts;
    raw samples retain wide uncertainty rather than implying strict operator
    certification.  Conditional ideal mean/cov come from analytic A(k)/B;
    equilibrium target is fixed-COM Maxwell with directional variance one.
    """
    from protocol import ideal_a_mean
    out=Path(out);marker=out/'mc_done.json'
    if marker.exists():raise FileExistsError('Preserve completed MC evidence')
    if not ranks['all_checks_passed']:raise RuntimeError('Select and validate actual ranks before MC')
    prepared=prepare_state(state,params);label=state.get('label',state.get('name','representative'))
    RB,_,CB=prepared['full_ref'].matrices(DT);rows=[];manifest=[];started=perf_counter()
    selected_ks={r['k'] for r in ranks['targets']}
    methods=[('B',1,ranks['B'])]+[('A',int(k),v['A']) for k,v in sorted(ranks['by_k'].items(),key=lambda kv:int(kv[0])) if int(k) in selected_ks]
    probes=prepared['full_ref'].probes(np.random.default_rng(SEED+700000));probes={k:v for k,v in probes.items() if k in ('slow','fast','random','COM')}
    for method,k,rr in methods:
        R=RB if method=='B' else ideal_a_mean(prepared['edges'],prepared['GL'],DT,k)
        C=CB if method=='B' else prepared['full_ref'].Pi-R@prepared['full_ref'].Pi@R.T
        for phase in ('conditional','equilibrium'):
            actual=[];ideal=[];p_inputs=[];sample_records=[]
            for sample in range(int(n)):
                seed=SEED+800000+k*10000+(method=='B')*1000000+(phase=='equilibrium')*1000+sample
                if phase=='equilibrium':
                    p=project(np.random.default_rng(seed+9000000).standard_normal(np.asarray(state['p']).shape))
                else:p=np.asarray(state['p']).copy()
                result,raw=path_check(prepared,p,method,k,**rr,seed=seed)
                actual.append(raw['actual'].ravel());ideal.append(raw['ideal'].ravel());p_inputs.append(np.asarray(p).ravel())
                sample_records.append(dict(sample=sample,seed=seed,input_momentum_sha256=array_sha(p),
                    actual_sha256=array_sha(raw['actual']),ideal_sha256=array_sha(raw['ideal']),
                    coupled_relative_error=result['coupled_relative_error'],noise_substeps=result['substeps'],
                    rng_state_before=result['rng_state_before'],rng_state_after=result['rng_state_after']))
            actual=np.asarray(actual);ideal=np.asarray(ideal);p_inputs=np.asarray(p_inputs)
            path=out/f'mc_samples_{label}_{method}_k{k}_{phase}.npz'
            np.savez_compressed(path,actual=actual,ideal=ideal,p_inputs=p_inputs,seed=np.array([r['seed'] for r in sample_records],dtype=np.uint64),q=state['q'])
            save(out/f'mc_streams_{label}_{method}_k{k}_{phase}.json',sample_records)
            for direction,v in probes.items():
                expected_mean=float(v@R@np.asarray(state['p']).ravel()) if phase=='conditional' else 0.
                expected_variance=float(v@C@v) if phase=='conditional' else float(v@prepared['full_ref'].Pi@v)
                if direction=='COM':
                    expected_mean=float(v@np.asarray(state['p']).ravel()) if phase=='conditional' else 0.
                    expected_variance=0.
                stat=directional_statistics(actual@v,ideal@v,expected_mean,expected_variance)
                rows.append(dict(state=label,method=method,k=k,phase=phase,direction=direction,**rr,**stat))
            manifest.append(dict(state=label,method=method,k=k,phase=phase,n_samples=n,file=str(path.relative_to(HERE)),
                actual_sha256=array_sha(actual),ideal_sha256=array_sha(ideal),max_coupled_relative=max(r['coupled_relative_error'] for r in sample_records)))
            write_csv(out/'mc_directional.csv',rows)
            print('MC',label,method,k,phase,'samples',n,flush=True)
    result=dict(representative_state=label,samples_per_method_phase=n,rows=len(rows),methods=[dict(method=m,k=k,**rr) for m,k,rr in methods],
        manifest=manifest,elapsed_s=perf_counter()-started,
        scope='local finite directional MC with wide pointwise uncertainty; no strict covariance-operator or Gibbs proof; no raw A/B path-distance bias comparison')
    save(marker,result);return result

def verify_production_wrapper(state,params,ranks,out):
    """Cross-check the actual timing wrapper against the validated actual path."""
    from matched_dynamics import MatchedSystem
    out=Path(out);marker=out/'wrapper_validation.json'
    if marker.exists():raise FileExistsError('Preserve completed wrapper evidence')
    prepared=prepare_state(state,params);system=MatchedSystem(params,retain_noise=True)
    label=state.get('label',state.get('name','representative'))
    selected_ks=sorted({int(t['k']) for t in ranks['targets']})
    methods=[('B',1,ranks['B'])]+[('A',k,ranks['by_k'][str(k)]['A']) for k in selected_ks]
    rows=[];raw=dict(q=np.asarray(state['q']),p=np.asarray(state['p']))
    for method,k,rr in methods:
        seed=SEED+990000+k*10+(method=='B');rng=np.random.default_rng(seed)
        checked,path_raw=path_check(prepared,state['p'],method,k,**rr,seed=seed)
        built=system.build(state['q'],method,DT,k)
        actual,diag=system.frozen_ou(built,state['p'],method,**rr,rng=rng,dt=DT,k=k)
        helper=path_raw['actual'];error=relative(actual,helper)
        hash_errors=[]
        for previous,production in zip(checked['substeps'],system.last_gaussian_hashes):
            if previous['z_sha256']!=production['z_sha256']:hash_errors.append('z')
            if method=='A' and (previous['ef_sha256']!=production['eta_forward_sha256'] or previous['er_sha256']!=production['eta_reverse_sha256']):hash_errors.append('ef/er')
        passed=bool(np.allclose(actual,helper,rtol=1e-13,atol=1e-13) and not hash_errors and
            rng.bit_generator.state==checked['rng_state_after'] and diag['matvec_count']==checked['actual_matvec_count'])
        rows.append(dict(state=label,method=method,k=k,**rr,seed=seed,passed=passed,
            bitwise_equal=np.array_equal(actual,helper),relative_error=error,
            max_abs_error=float(np.max(np.abs(actual-helper))),gaussian_hash_errors=hash_errors,
            actual_matvec_count=diag['matvec_count'],graph_builds=1,
            coefficient_tau=built.metrics['coefficient_tau'],
            wrapper_actual_sha256=array_sha(actual),validated_actual_sha256=array_sha(helper),
            own_dense_ideal_sha256=array_sha(path_raw['ideal']),
            wrapper_substep_hashes=system.last_gaussian_hashes))
        raw[f'{method}_k{k}_wrapper']=actual;raw[f'{method}_k{k}_validated_actual']=helper;raw[f'{method}_k{k}_own_ideal']=path_raw['ideal']
        print('Production wrapper crosscheck',method,k,'bitwise',rows[-1]['bitwise_equal'],'passed',passed,flush=True)
    np.savez_compressed(out/'wrapper_validation_outputs.npz',**raw)
    result=dict(passed=all(row['passed'] for row in rows),state=label,checks=rows,
        scope='production frozen_ou wrapper vs separately implemented validated actual ou_a/ou_b path, same p/q/draws/ranks')
    save(marker,result)
    if not result['passed']:raise RuntimeError('Production wrapper differs from validated actual path')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['select','common','mc','wrapper']);parser.add_argument('--selection',default=str(RESULTS/'accuracy/frozen_selection.json'))
    parser.add_argument('--samples',type=int,default=16);args=parser.parse_args()
    states=load_states()
    if args.action=='select':select_and_validate(states,PARAMS,js(args.selection),RESULTS/'stochastic')
    elif args.action=='common':verify_common_probes(states,PARAMS,js(RESULTS/'stochastic/stochastic_selection.json'),RESULTS/'stochastic')
    elif args.action=='wrapper':verify_production_wrapper(states[0],PARAMS,js(RESULTS/'stochastic/stochastic_selection.json'),RESULTS/'stochastic')
    else:run_mc(states[0],PARAMS,js(RESULTS/'stochastic/stochastic_selection.json'),RESULTS/'stochastic',n=args.samples)
