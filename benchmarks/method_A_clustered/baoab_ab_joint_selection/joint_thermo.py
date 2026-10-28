"""Physical-reference OU certification and fresh finite-Krylov selection.

Every ideal transition is compared with the SAME independently summed
physical periodic Gamma_ref, as well as with its own discrete Gamma_h.
Finite Lanczos paths are a separate sampled test, not a covariance operator.
"""
from __future__ import annotations
from joint_common import *
from time import perf_counter
from scipy import linalg
from frozen_accuracy import ideal_mean,exact_operator_norm
from stochastic_validation import validate_probes,path_check,exact_action,action_of
from rank_validation import directions
from ou_core import lanczos_action
from matched_dynamics import MatchedSystem

KS=(1,2,4,8,16)
RANKS=tuple(range(1,17))+(20,24,32,48,64,96,128,192,256,512,765)
SMALL=1e-300
SEED=202610281104

def norm2(A):return exact_operator_norm(np.asarray(A,float))[0]

def map_errors(R,C,Rh,Ch,Rref,Cref,tail_absolute=0.,dt=DT):
    """Complete COM-space mean/cov norms with a conservative physical tail.

    On the common COM subspace all friction operators are PSD. Duhamel gives
    ||exp(-h G)-exp(-h Gref)|| <= h||G-Gref||. Contractivity gives a
    covariance remainder <=2h||G-Gref||. Denominators use the corresponding
    lower bound for the unknown infinite-lattice exact reference norm.
    """
    R,C,Rh,Ch,Rref,Cref=map(lambda x:np.asarray(x,float),(R,C,Rh,Ch,Rref,Cref))
    rd=norm2(Rref);cd=norm2(Cref);dr=float(dt)*float(tail_absolute);dc=2*dr
    lower_r=max(rd-dr,0.);lower_c=max(cd-dc,0.)
    mr,md,ms=exact_operator_norm(R-Rref);cr,ccd,cs=exact_operator_norm(C-Cref)
    ownmr=norm2(R-Rh);owncr=norm2(C-Ch)
    space_m=norm2(Rh-Rref);space_c=norm2(Ch-Cref)
    result=dict(physical_mean_absolute=mr,physical_cov_absolute=cr,
        physical_mean_reference_norm=rd,physical_cov_reference_norm=cd,
        physical_mean_tail_allowance=dr,physical_cov_tail_allowance=dc,
        physical_mean_denominator_lower=lower_r,physical_cov_denominator_lower=lower_c,
        denominator_small=SMALL,physical_mean_relative=mr/max(rd,SMALL),
        physical_cov_relative=cr/max(cd,SMALL),
        physical_mean_bound=(mr+dr)/max(lower_r,SMALL),
        physical_cov_bound=(cr+dc)/max(lower_c,SMALL),
        reference_denominators_resolved=bool(lower_r>SMALL and lower_c>SMALL),
        own_mean_absolute=ownmr,own_cov_absolute=owncr,
        own_mean_relative=ownmr/max(norm2(Rh),SMALL),own_cov_relative=owncr/max(norm2(Ch),SMALL),
        spatial_transition_mean_relative=space_m/max(rd,SMALL),
        spatial_transition_cov_relative=space_c/max(cd,SMALL),
        physical_mean_norm_method=md['method'],physical_cov_norm_method=ccd['method'],
        norm_scope='full 3N-3 COM operator 2-norm; not sampled probes',
        tail_rule='Duhamel h*deltaGamma for R, 2h*deltaGamma for C; subtract remainder from reference denominator')
    return result,dict(physical_mean_difference_spectrum=ms,physical_cov_difference_spectrum=cs)

def _physical_data(prepared):
    if '_joint_map_data' not in prepared:
        Q=prepared['Q'];Rh,_,Ch=prepared['full_ref'].matrices(DT)
        Rp,_,Cp=prepared['physical_ref'].matrices(DT)
        prepared['_joint_map_data']=(Q,Q.T@Rh@Q,Q.T@Ch@Q,Q.T@Rp@Q,Q.T@Cp@Q)
    return prepared['_joint_map_data']

def compare(prepared,method,k=1,dt=DT):
    if dt!=DT:raise ValueError('Registered joint-selection dt is .005')
    if method=='B' and k!=1:raise ValueError('B is a single full OU call')
    Q,Rh,Ch,Rp,Cp=_physical_data(prepared)
    if method=='B':R,C=Rh,Ch
    elif method=='A':
        full=ideal_mean(prepared,int(k),dt);R=Q.T@full@Q;C=np.eye(len(R))-R@R.T
    else:raise ValueError('Unknown thermostat')
    meta=prepared['physical_meta'];tail=float(meta['tail_absolute'])
    rec,spectra=map_errors(R,C,Rh,Ch,Rp,Cp,tail,dt)
    # This check is independent of the OU exponential error split.
    gn=float(meta['norm2']);gdiff=norm2(Q.T@(prepared['GH']-prepared['physical_G'])@Q)
    spatial_bound=(gdiff+tail)/max(gn-tail,SMALL)
    rec.update(method=method,k=int(k),dt=dt,dimension_COM=len(R),
        spatial_error_raw=gdiff/max(gn,SMALL),spatial_error_bound=spatial_bound,
        spatial_passed=spatial_bound<=2e-7,physical_reference_tail_absolute=tail,
        RA_reduced_sha256=array_sha(R),CA_reduced_sha256=array_sha(C),
        physical_reference_source=meta.get('source_file'),physical_reference_source_sha256=meta.get('source_sha256'))
    return rec,dict(RA_reduced=R,CA_reduced=C,**spectra)

def _sid(setting):return str(setting.get('setting_id',setting.get('spatial_id',setting.get('id'))))

def run_thermo(states,settings,out,ks=KS):
    """Adaptive ideal certification; stop only after all states pass tight tier."""
    from joint_spatial import prepare_setting
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    marker=out/'thermo_selection.json'
    if marker.exists():raise FileExistsError('Preserve completed ideal physical certification')
    rows=[];settings_out=[];configs=[];start=perf_counter();executed={}
    for setting in settings:
        sid=_sid(setting);params=setting['params'];prepared=[]
        if not setting.get('spatial_passed',False):raise ValueError('Spatial candidate is not certified')
        for state in states:
            label=state['label'];pp=prepare_setting(state,params);prepared.append(pp)
            ref_file=out/f'reference_{sid}_{label}.npz'
            np.savez_compressed(ref_file,q=state['q'],p=state['p'],GH=pp['GH'],GL=pp['GL'])
            save(out/f'reference_{sid}_{label}.json',dict(setting_id=sid,state=label,
                q_sha256=array_sha(state['q']),GH_sha256=array_sha(pp['GH']),GL_sha256=array_sha(pp['GL']),
                file=str(ref_file.relative_to(HERE)),file_sha256=sha(ref_file),physical_reference=pp['physical_meta']))
        def evaluate(method,k):
            for state,pp in zip(states,prepared):
                label=state['label'];rec,raw=compare(pp,method,k)
                rec.update(setting_id=sid,state=label,q_sha256=array_sha(state['q']),
                    lambda_max=float(pp['full_ref'].eig[-1]),
                    splitting_coefficient_mean=rec['own_mean_relative']*k*k,
                    splitting_coefficient_cov=rec['own_cov_relative']*k*k)
                previous=[r for r in rows if r['setting_id']==sid and r['state']==label and r['method']==method]
                if method=='A' and previous:
                    prior=previous[-1]
                    for name in ('own_mean_relative','own_cov_relative','physical_mean_bound','physical_cov_bound'):
                        if rec[name]>0 and prior[name]>0:
                            rec['observed_slope_'+name]=float(np.log(rec[name]/prior[name])/np.log(k/prior['k']))
                path=out/f'maps_{sid}_{label}_{method}_k{k}.npz'
                np.savez_compressed(path,**raw)
                rec.update(maps_file=str(path.relative_to(HERE)),maps_file_sha256=sha(path))
                rows.append(rec);write_csv(out/'ideal_errors.csv',rows)
                print('ideal',sid,label,method,k,'physical mean/cov',rec['physical_mean_bound'],rec['physical_cov_bound'],flush=True)
        evaluate('B',1);actual=[];stopped=False
        for k in ks:
            evaluate('A',int(k));actual.append(int(k))
            batch=[r for r in rows if r['setting_id']==sid and r['method']=='A' and r['k']==k]
            if len(batch)==len(states) and all(r['spatial_passed'] and r['reference_denominators_resolved'] and
                    max(r['physical_mean_bound'],r['physical_cov_bound'])<=min(TIERS) for r in batch):
                stopped=True;break
        executed[sid]=actual;targets=[]
        for eps in TIERS:
            b=[r for r in rows if r['setting_id']==sid and r['method']=='B']
            bp=len(b)==len(states) and all(r['spatial_passed'] and r['reference_denominators_resolved'] and
                max(r['physical_mean_bound'],r['physical_cov_bound'])<=eps for r in b)
            accepted=[]
            for k in actual:
                rr=[r for r in rows if r['setting_id']==sid and r['method']=='A' and r['k']==k]
                if len(rr)==len(states) and all(r['spatial_passed'] and r['reference_denominators_resolved'] and
                    max(r['physical_mean_bound'],r['physical_cov_bound'])<=eps for r in rr):accepted.append(int(k))
            chosen=min(accepted) if accepted else None
            targets.append(dict(epsilon_frozen=eps,A_k=chosen,A_accepted_ks=accepted,B_passed=bp))
            for method,k,passed in [('A',chosen,chosen is not None),('B',1,bp)]:
                if not passed:continue
                rr=[r for r in rows if r['setting_id']==sid and r['method']==method and r['k']==k]
                configs.append(dict(config_id=f'{sid}_{method}_eps{eps:g}',spatial_id=sid,method=method,
                    epsilon_frozen=eps,params=params,k=int(k),r_d=None,r_n=None,
                    certificate=dict(spatial_passed=True,ideal_physical_passed=True,finite_passed=False,
                        validation_states=[s['label'] for s in states],
                        max_spatial_error=max(r['spatial_error_bound'] for r in rr),
                        max_ideal_physical_mean=max(r['physical_mean_bound'] for r in rr),
                        max_ideal_physical_cov=max(r['physical_cov_bound'] for r in rr),
                        max_own_Gh_mean=max(r['own_mean_relative'] for r in rr),
                        max_own_Gh_cov=max(r['own_cov_relative'] for r in rr))))
        settings_out.append(dict(setting_id=sid,params=params,targets=targets,
            executed_ks=actual,not_executed_ks=[int(k) for k in ks if k not in actual],
            stop_reason='all registered states passed the tight required physical tier' if stopped else 'registered k list exhausted',
            spatial_error=setting.get('max_spatial_error'),validation_states=[s['label'] for s in states]))
        save(out/'thermo_progress.json',dict(settings=settings_out,configs=configs,executed_ks_by_setting=executed,rows=len(rows)))
    result=dict(settings=settings_out,configs=configs,rows=len(rows),registered_ks=list(ks),executed_ks_by_setting=executed,tiers=list(TIERS),
        elapsed_s=perf_counter()-start,reference='common independent periodic physical full-Gamma exact OU',
        interpretation='ideal full-COM operator certification over finite registered states; finite Lanczos requires separate sampled checks')
    save(marker,result);return result

def _ordered_probes(prepared,state,method,k,seed):
    rng=np.random.default_rng(seed);ref=prepared['long_ref'] if method=='A' else prepared['full_ref']
    z=rng.standard_normal(np.asarray(state['p']).shape);p=np.asarray(state['p'])
    if method=='A':
        ef=rng.standard_normal((len(prepared['edges'].i),3))
        p=prepared['edges'].sweep(p,ef,DT/(2*k))[0]
    dd=directions(state['q'],ref,p,z)
    # Reject bad ranks quickly on stochastic/actual directions; eigenvector probes alone are too easy.
    names=['random','actual_gaussian','actual_momentum']+[n for n in dd if n not in ('random','actual_gaussian','actual_momentum')]
    return ref,{n:dd[n] for n in names if n in dd}

def _probe_rank(prepared,state,method,k,rank,kind,budget,seed,early=True):
    ref,probes=_ordered_probes(prepared,state,method,k,seed);h=DT/k if method=='A' else DT
    action=action_of(prepared,method);rows=[];raw={}
    for name,v in probes.items():
        expected=exact_action(ref,v,h,kind);actual,diag=lanczos_action(action,v,rank,h,kind)
        error=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),np.finfo(float).tiny))
        prefix=f'{kind}__{name}'
        rows.append(dict(state=state['label'],method=method,k=k,kind=kind,probe=name,
            rank_requested=rank,rank_used=int(diag['rank_used']),relative_error=error,budget=budget,
            passed=error<=budget,matvec_count=diag['matvec_count'],negative_ritz_count=diag['negative_ritz_count'],
            clipped_negative_count=diag['clipped_negative_count'],ritz_min_raw=diag['ritz_min_raw'],
            probe_sha256=array_sha(v),exact_sha256=array_sha(expected),actual_sha256=array_sha(actual)))
        raw[prefix+'__probe']=v;raw[prefix+'__exact']=expected;raw[prefix+'__actual']=actual
        if early and error>budget:break
    return rows,raw

def select_ranks(states,settings,thermo,out):
    """Fresh rank search and recheck for every actual spatial/method/tier config."""
    from joint_spatial import prepare_setting
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    marker=out/'rank_selection.json'
    if marker.exists():raise FileExistsError('Preserve completed finite certification')
    settings_by_id={_sid(s):s for s in settings};attempts=[];configs=[];start=perf_counter()
    # Reuse only genuinely identical budgets; a tighter tier must not inflate the loose-tier rank.
    groups={}
    for c in thermo['configs']:
        eps=c['epsilon_frozen'];k=c['k']
        slack=eps-max(c['certificate']['max_ideal_physical_mean'],c['certificate']['max_ideal_physical_cov'])
        if slack<=0:raise ValueError('Ideal candidate has no positive physical threshold slack')
        action_budget=min(2e-8,eps/(20*k),slack/(40*k));path_budget=min(2e-8,eps/10,slack/8)
        groups.setdefault((c['spatial_id'],c['method'],k,action_budget,path_budget),[]).append(c)
    for gi,((sid,method,k,action_budget,path_budget),cc) in enumerate(groups.items()):
        params=settings_by_id[sid]['params'];eps=min(c['epsilon_frozen'] for c in cc)
        import hashlib
        tag=hashlib.sha256(f'{action_budget:.17g},{path_budget:.17g}'.encode()).hexdigest()[:8]
        group_id=f'{sid}_{method}_k{k}_b{tag}'
        global_rank=dict(r_d=0,r_n=0);prepared=[]
        for index,state in enumerate(states):
            pp=prepare_setting(state,params);prepared.append(pp)
            for kind,key in [('damping','r_d'),('noise','r_n')]:
                dimension=np.asarray(state['p']).size-3
                for rank in sorted({min(r,dimension) for r in RANKS}):
                    rr,raw=_probe_rank(pp,state,method,k,rank,kind,action_budget,SEED+gi*10000+index*100,True)
                    for r in rr:r.update(spatial_id=sid,phase='rank_selection')
                    attempts.extend(rr);write_csv(out/'rank_attempts.csv',attempts)
                    print('rank',sid,state['label'],method,k,kind,rank,'max',max(r['relative_error'] for r in rr),flush=True)
                    if all(r['passed'] for r in rr):global_rank[key]=max(global_rank[key],rank);break
                else:raise RuntimeError('No actual tested rank passes')
        failures=[];max_probe=0.;max_path=0.;max_actual=0.;checks=[]
        for index,(state,pp) in enumerate(zip(states,prepared)):
            raw=dict(q=state['q'],p=state['p'])
            for kind,key in [('damping','r_d'),('noise','r_n')]:
                rr,values=_probe_rank(pp,state,method,k,global_rank[key],kind,action_budget,SEED+gi*10000+index*100,False)
                for r in rr:r.update(spatial_id=sid,phase='final_common_rank')
                checks.extend(rr);raw.update(values);max_probe=max(max_probe,max(r['relative_error'] for r in rr))
                if not all(r['passed'] for r in rr):failures.append('common registered probes '+state['label'])
            np.savez_compressed(out/f'common_{group_id}_{state["label"]}.npz',**raw)
            result,pathraw=path_check(pp,state['p'],method,k,**global_rank,seed=SEED+500000+gi*10000+index*100,action_budget=action_budget)
            result.update(spatial_id=sid,state=state['label'],per_action_budget=action_budget,full_path_budget=path_budget)
            result['passed']=result['max_actual_input_action_error']<=action_budget and result['coupled_relative_error']<=path_budget and result['negative_ritz_count']==0
            if not result['passed']:failures.append('own composed path '+state['label'])
            max_path=max(max_path,result['coupled_relative_error']);max_actual=max(max_actual,result['max_actual_input_action_error'])
            np.savez_compressed(out/f'path_{group_id}_{state["label"]}.npz',q=state['q'],**pathraw)
            save(out/f'path_{group_id}_{state["label"]}.json',result)
        write_csv(out/f'common_probes_{group_id}.csv',checks)
        # Cross-check the actual production wrapper, including full k and RNG draw order.
        system=MatchedSystem(params);state=states[0];seed=SEED+800000+gi
        built=system.build(state['q'],method,DT,k)
        actual,diag=system.frozen_ou(built,state['p'],method,**global_rank,rng=np.random.default_rng(seed),dt=DT,k=k)
        path,raw=path_check(prepared[0],state['p'],method,k,**global_rank,seed=seed)
        wrapper_equal=bool(np.array_equal(actual,raw['actual']))
        if not wrapper_equal:failures.append('production wrapper mismatch')
        np.savez_compressed(out/f'wrapper_{group_id}.npz',q=state['q'],p=state['p'],production=actual,validated_actual=raw['actual'],own_ideal=raw['ideal'])
        summary=dict(group_id=group_id,spatial_id=sid,method=method,k=k,**global_rank,per_action_budget=action_budget,
            full_path_budget=path_budget,max_registered_probe_error=max_probe,max_actual_input_action_error=max_actual,
            max_coupled_path_error=max_path,wrapper_bitwise_equal=wrapper_equal,passed=not failures,failures=failures,
            validation_states=[s['label'] for s in states],
            limitation='sampled finite-Lanczos actions and own-ideal paths; not a uniform nonlinear covariance operator bound')
        save(out/f'finite_{group_id}.json',summary)
        for c in cc:
            c=dict(c);c.update(global_rank)
            c['certificate']=dict(c['certificate'],finite_validation_group=group_id,finite_passed=not failures,per_action_budget=action_budget,
                full_path_budget=path_budget,max_probe_error=max_probe,max_actual_input_action_error=max_actual,
                max_finite_path_error=max_path,wrapper_bitwise_equal=wrapper_equal)
            configs.append(c)
        if failures:raise RuntimeError('Finite validation failed; see preserved raw records')
    result=dict(configs=configs,all_checks_passed=all(c['certificate']['finite_passed'] for c in configs),
        groups=len(groups),elapsed_s=perf_counter()-start,
        scope='fresh per-spatial-setting rank selection; physical ideal certification and finite sampled evidence separately')
    save(marker,result);return result

def check_config_state(config,state,out=None,*,prepared=None):
    """Full physical/spatial check for an expanded or actual visited geometry."""
    from joint_spatial import prepare_setting
    if prepared is None:prepared=prepare_setting(state,config['params'])
    rec,raw=compare(prepared,config['method'],config['k'])
    rec.update(config_id=config['config_id'],state=state['label'],epsilon_frozen=config['epsilon_frozen'],
        q_sha256=array_sha(state['q']),p_sha256=array_sha(state['p']),
        ideal_physical_passed=rec['reference_denominators_resolved'] and
            max(rec['physical_mean_bound'],rec['physical_cov_bound'])<=config['epsilon_frozen'])
    rec['passed']=rec['spatial_passed'] and rec['ideal_physical_passed']
    if out is not None:
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        path=out/f'check_{config["config_id"]}_{state["label"]}.npz'
        np.savez_compressed(path,q=state['q'],p=state['p'],GH=prepared['GH'],GL=prepared['GL'],**raw)
        rec.update(file=str(path.relative_to(HERE)),file_sha256=sha(path),physical_reference=prepared['physical_meta'])
        save(path.with_suffix('.json'),rec)
    return rec

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['ideal','ranks'])
    args=parser.parse_args();states=certification_states()
    settings=js(RESULTS/'spatial/spatial_selection.json')['qualified_settings']
    if args.action=='ideal':run_thermo(states,settings,RESULTS/'thermo')
    else:select_ranks(states,settings,js(RESULTS/'thermo/thermo_selection.json'),RESULTS/'thermo')
