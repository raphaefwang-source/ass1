"""Adaptive space search against an independent complete physical lattice sum.

The production Grid PSD code is imported unchanged. Probe bounds only reject
or prioritize candidates; every qualified setting passes a complete765-D COM
spectral test on the registered multi-geometry domain, including a rigorous
allowance for the physical reference's infinite lattice remainder.
"""
from __future__ import annotations
from joint_common import *
from time import perf_counter
from scipy import linalg
from spatial_validation import build_mesh,build_operator,dense_parts,near_edges,long_action,cutoff,physical_tail_bound,direct_reference
from compiled_edge import CompiledEdgeOU
from protocol import DenseOU

SPATIAL_TOL=2e-7
_PHYSICAL={}
_PHYSICAL_DENSE={}

def norm2(A):
    ev=linalg.eigvalsh((np.asarray(A)+np.asarray(A).T)/2,check_finite=False)
    return float(np.max(np.abs(ev),initial=0.))

def spatial_bound(error_abs,reference_norm,tail_abs):
    denominator=float(reference_norm)-float(tail_abs)
    if denominator<=0:raise ValueError('Reference norm lower bound must be positive')
    return (float(error_abs)+float(tail_abs))/denominator

def probe_rejection_lower_bound(action_difference,reference_norm,tail_abs):
    """A unit-probe lower bound on the true relative physical matrix error."""
    return max(0.,float(action_difference)-float(tail_abs))/(float(reference_norm)+float(tail_abs))

def physical_reference(state,out=None):
    q=np.mod(np.asarray(state['q'],float),BASE_PARAMS['L']);signature=array_sha(q)
    if signature in _PHYSICAL:return _PHYSICAL[signature]
    candidates=list((PILOT/'results').glob('*spatial_direct.json'))
    candidates += list((PILOT/'results').glob('selection_direct.json'))
    prefix=None;old=None
    for path in candidates:
        record=js(path)
        if record.get('q_sha256')==signature:
            if record.get('kappa')!=BASE_PARAMS['kappa'] or record.get('c')!=BASE_PARAMS['c']:
                raise RuntimeError('Physical direct reference kernel mismatch')
            if record.get('R_max',0)<45.:raise RuntimeError('Direct radius insufficient')
            prefix=path.with_suffix('');old=record;break
    if prefix is None:
        folder=Path(out).resolve() if out is not None else RESULTS/'physical_refs'
        folder.mkdir(parents=True,exist_ok=True)
        prefix=folder/('direct_'+signature[:16])
        G,old=direct_reference(q,BASE_PARAMS['L'],prefix,R=45.)
    else:
        with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as d:
            G=d['G'].copy()
            if not np.array_equal(np.mod(d['q'],BASE_PARAMS['L']),q):raise RuntimeError('Physical reference geometry mismatch')
    if array_sha(G)!=old['matrix_sha256']:raise RuntimeError('Physical direct matrix SHA mismatch')
    if old['L']!=BASE_PARAMS['L']:raise RuntimeError('Physical direct box mismatch')
    tail=physical_tail_bound(len(q),BASE_PARAMS['L'],old['R_max'])
    # Metadata allowance is independently recomputed from the unchanged analytic bound.
    tail=max(tail,float(old['certified_tail_relative'])*float(old['norm2']))
    meta=dict(q_sha256=signature,matrix_sha256=array_sha(G),norm2=float(old['norm2']),
        tail_absolute=tail,tail_relative=tail/float(old['norm2']),
        source_file=str(prefix.with_suffix('.npz').relative_to(REPO)),
        source_sha256=sha(prefix.with_suffix('.npz')),
        source_metadata_file=str(prefix.with_suffix('.json').relative_to(REPO)),
        source_metadata_sha256=sha(prefix.with_suffix('.json')),
        kappa=BASE_PARAMS['kappa'],c=BASE_PARAMS['c'],L=BASE_PARAMS['L'],R_max=old['R_max'],
        reference='unsplit c exp(-kappa*r)/r full periodic direct lattice graph',
        direct_convergence=old['convergence'],self_particle_edges_excluded=True,
        reused_readonly_reference=prefix.is_relative_to(PILOT))
    _PHYSICAL[signature]=(G,meta)
    return G,meta

def prepare_setting(state,params,physical_out=None,mesh=None):
    q=np.asarray(state['q'],float)
    if mesh is None:mesh=build_mesh(params,float(params['L']))
    op=build_operator(q,params,mesh=mesh);GS,GL,GH=dense_parts(op)
    edges=CompiledEdgeOU(near_edges(op),n_particles=len(q),configuration_token=array_sha(q))
    if edges.backend!='compiled':raise RuntimeError('Compiled Pair OU backend required')
    Gref,meta=physical_reference(state,physical_out)
    if meta['q_sha256'] not in _PHYSICAL_DENSE:_PHYSICAL_DENSE[meta['q_sha256']]=DenseOU(Gref)
    physical=_PHYSICAL_DENSE[meta['q_sha256']]
    full,long=DenseOU(GH),DenseOU(GL)
    return dict(op=op,edges=edges,GS=GS,GL=GL,GH=GH,full_ref=full,long_ref=long,
        physical_ref=physical,physical_G=Gref,physical_meta=meta,Q=physical.Q,params=dict(params))

def make_params(xi,sR,sF,M,p):
    return dict(BASE_PARAMS,xi=float(xi),sR=float(sR),sF=float(sF),M=int(M),p=int(p))

def setting_id(params):
    payload={k:params[k] for k in ('xi','sR','sF','M','p')}
    return 'sp_'+hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:10]

def mesh_metadata(mesh,params,op=None):
    rc,kc,shell=cutoff(params,params['L']);d=mesh.diagnostics
    record=dict(rc=rc,kc=kc,kc_shell=shell,eta=kc*mesh.h/np.pi,h=mesh.h,
        modes=int(mesh.original.n_modes),excluded_modes=int(mesh.n_excluded),W2min=float(mesh.W2min),
        projection_active=bool(d['negative_blocks']),negative_blocks=int(d['negative_blocks']),
        significant_negative_blocks=int(d['significant_negative_blocks']),
        correction_relative_fro=float(d['correction_relative_fro']),
        correction_max_spectral=float(d['correction_max_spectral']),
        table_min_eigenvalue_before=float(d['min_eig_before']),
        physical_k0_retained=bool(np.linalg.norm(mesh.zero_mode_before)>0),
        zero_mode_before=mesh.zero_mode_before.tolist(),zero_mode_after=mesh.zero_mode.tolist())
    if op is not None:record.update(images=int(op.n_images),near_unique_edges=len(near_edges(op)),self_edges=int(np.sum(op.i==op.j)))
    return record

def cheap_screen(state,params):
    Gref,ref=physical_reference(state)
    started=perf_counter();t=perf_counter();mesh=build_mesh(params,params['L']);table=perf_counter()-t
    t=perf_counter();op=build_operator(state['q'],params,mesh=mesh);setup=perf_counter()-t
    rng=np.random.default_rng(2026102801201)
    probes=[rng.normal(size=3*len(state['q'])) for _ in range(3)]
    probes.append(np.asarray(state['p']).ravel())
    errors=[];walls=[]
    for v in probes:
        v=v/np.linalg.norm(v);t=perf_counter();actual=op.matvec(v);walls.append(perf_counter()-t)
        errors.append(float(np.linalg.norm(actual-Gref@v)))
    lower=max(probe_rejection_lower_bound(e,ref['norm2'],ref['tail_absolute']) for e in errors)
    return dict(setting_id=setting_id(params),**params,**mesh_metadata(mesh,params,op),
        sampled_physical_rejection_lower_bound=lower,cheap_screen_rejected=lower>SPATIAL_TOL,
        sampled_action_error_abs_max=max(errors),reference_norm=ref['norm2'],reference_tail_absolute=ref['tail_absolute'],
        t_table_setup_s=table,t_q_operator_setup_s=setup,t_full_action_mean_s=float(np.mean(walls)),
        screen_wall_s=perf_counter()-started,
        priority_proxy_s=setup+17*float(np.mean(walls)),
        priority_scope='setup+17full-actions screening proxy only; not dynamic BAOAB timing or winner',
        status='rejected_by_physical_probe_lower_bound' if lower>SPATIAL_TOL else 'eligible_for_fullnorm_screen')

def certify_state(state,params,out,save_matrix=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sid=setting_id(params);label=state['label'];Gref,ref=physical_reference(state)
    started=perf_counter();mesh=build_mesh(params,params['L']);op=build_operator(state['q'],params,mesh=mesh)
    GS,GL,GH=dense_parts(op);Q=np.kron(linalg.helmert(len(state['q']),full=False).T,np.eye(3))
    difference=Q.T@(GH-Gref)@Q
    error=norm2(difference);bound=spatial_bound(error,ref['norm2'],ref['tail_absolute'])
    reduced=Q.T@GH@Q;long=Q.T@GL@Q
    eig=linalg.eigvalsh((reduced+reduced.T)/2,check_finite=False)
    lev=linalg.eigvalsh((long+long.T)/2,check_finite=False)
    scale=max(float(np.max(np.abs(eig))),1.);roundoff=256*np.finfo(float).eps*scale
    rng=np.random.default_rng(2026102801202);actions=[];parts=[];raw={}
    for i in range(2):
        v=rng.normal(size=len(GH));expected=GH@v;actual=op.matvec(v)
        split=GS@v+long_action(op,v)
        actions.append(float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)))
        parts.append(float(np.linalg.norm(actual-split)/np.linalg.norm(expected)))
        raw[f'probe{i}']=v;raw[f'fft{i}']=actual;raw[f'dense{i}']=expected
    trans=np.tile(np.eye(3),(len(state['q']),1))
    com=float(np.linalg.norm(GH@trans)/max(float(eig[-1]),1.))
    symmetry=float(np.linalg.norm(GH-GH.T)/np.linalg.norm(GH))
    info=mesh_metadata(mesh,params,op)
    structural=bool(symmetry<=1e-12 and com<=1e-11 and eig[0]>=-roundoff and lev[0]>=-roundoff
        and max(actions)<=1e-10 and max(parts)<=1e-10 and info['self_edges']==0 and info['physical_k0_retained'])
    record=dict(setting_id=sid,state=label,params=dict(params),**info,
        spatial_error_absolute_norm=error,spatial_error_relative_without_tail=error/ref['norm2'],
        spatial_error_conservative_bound=bound,spatial_budget=SPATIAL_TOL,
        spatial_passed=bound<=SPATIAL_TOL,structural_checks_passed=structural,
        dimension_COM=len(GH)-3,full_operator_norm_computed=True,
        min_eigenvalue_COM=float(eig[0]),max_eigenvalue_COM=float(eig[-1]),
        long_min_eigenvalue_COM=float(lev[0]),relative_symmetry=symmetry,COM_null_relative=com,
        FFT_action_relative_max=max(actions),full_near_long_relative_max=max(parts),
        q_sha256=array_sha(state['q']),GH_sha256=array_sha(GH),reference=ref,
        wall_s=perf_counter()-started)
    if save_matrix:
        file=out/f'spatial_GH_{sid}_{label}.npz'
        np.savez_compressed(file,GH=GH,q=state['q'],difference_eigenvalues=linalg.eigvalsh((difference+difference.T)/2,check_finite=False),**raw)
        record.update(matrix_file=str(file.relative_to(HERE)),matrix_file_sha256=sha(file))
    save(out/f'spatial_{sid}_{label}.json',record)
    return record

def candidate_round_one():
    """Budgeted joint families, not a Cartesian scan or one-xi experiment."""
    values=[dict(BASE_PARAMS)]
    for xi in (.45,.55,.65,.75,.85,1.):
        values.append(make_params(xi,4.5,4.1,32,6))
        values.append(make_params(xi,4.5,4.7,40,8))
        values.append(make_params(xi,4.7,5.0,48,10))
    return values

def adaptive_refinements(rows):
    """Respond to promising or rejected families with resolution/rc changes."""
    candidates=[]
    # Refine each physical splitting scale, preserving independent near/far knobs.
    for xi in (.45,.55,.65,.75,.85,1.):
        rr=[r for r in rows if r.get('xi')==xi and r.get('status')!='illegal_or_failed']
        promising=any(not r.get('cheap_screen_rejected',True) for r in rr)
        if promising:
            candidates.extend([make_params(xi,4.1,4.7,32,8),make_params(xi,4.7,4.1,32,8),
                               make_params(xi,4.7,4.7,40,6)])
        else:
            candidates.extend([make_params(xi,5.2,5.0,64,10),make_params(xi,4.7,5.2,64,12)])
    return candidates

def run_search(states,out=None,max_full=9,max_final=4):
    out=RESULTS/'spatial' if out is None else Path(out);out.mkdir(parents=True,exist_ok=True)
    marker=out/'spatial_selection.json'
    if marker.exists():raise FileExistsError('Preserve completed space search')
    initial=next(s for s in states if s['label']=='initial_0')
    cert_names={'initial_0','initial_1','initial_2','visited_global_min_distance','visited_1_B_200'}
    cert=[s for s in states if s['label'] in cert_names]
    rows=[];seen=set();decisions=[];started=perf_counter()
    for round_index in (0,1):
        candidates=candidate_round_one() if round_index==0 else adaptive_refinements(rows)
        decisions.append(dict(round=round_index,candidates=[{k:p[k] for k in ('xi','sR','sF','M','p')} for p in candidates],
            rule='baseline and3jointfamilies/xi' if round_index==0 else 'if any cheap-eligible family: lowercost asymmetric cutoff/resolution alternatives; otherwise refine M/p and cutoffs'))
        for par in candidates:
            sid=setting_id(par)
            if sid in seen:continue
            seen.add(sid);print('Spatial cheap',round_index,sid,par,flush=True)
            try:row=cheap_screen(initial,par)
            except Exception as exc:row=dict(setting_id=sid,**par,status='illegal_or_failed',error=repr(exc),cheap_screen_rejected=True)
            row['search_round']=round_index;rows.append(row)
            write_csv(out/'cheap_screen.csv',rows);save(out/'adaptive_decisions.json',decisions)
            print('Spatial cheap result',sid,row.get('sampled_physical_rejection_lower_bound'),row['status'],flush=True)
    eligible=[r for r in rows if not r.get('cheap_screen_rejected',True)]
    baseline=setting_id(BASE_PARAMS)
    order=[]
    if any(r['setting_id']==baseline for r in eligible):order.append(next(r for r in eligible if r['setting_id']==baseline))
    # Preserve xi diversity, then prioritize actual observed cheap proxies.
    for xi in (.45,.55,.65,.75,.85,1.):
        rr=sorted([r for r in eligible if r['xi']==xi and r['setting_id']!=baseline],key=lambda r:r['priority_proxy_s'])
        if rr:order.append(rr[0])
    for row in sorted(eligible,key=lambda r:r['priority_proxy_s']):
        if row['setting_id'] not in {r['setting_id'] for r in order}:order.append(row)
    full=[];full_pass=[]
    for row in order[:max_full]:
        par={k:row[k] for k in BASE_PARAMS};rec=certify_state(initial,par,out,save_matrix=True)
        full.append(rec)
        if rec['spatial_passed'] and rec['structural_checks_passed']:full_pass.append((row,rec))
        print('Spatial full initial',row['setting_id'],rec['spatial_error_conservative_bound'],'passed',rec['spatial_passed'] and rec['structural_checks_passed'],flush=True)
        write_csv(out/'full_spatial_checks.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in full])
    # Keep old control and distinct promising splitting scales; actual BAOAB decides performance.
    proposed=[]
    base=next((x for x in full_pass if x[0]['setting_id']==baseline),None)
    if base:proposed.append(base)
    alternatives=sorted([x for x in full_pass if x[0]['setting_id']!=baseline],key=lambda x:x[0]['priority_proxy_s'])
    # A cheaper resolution at the same xi must not be excluded merely because
    # the old control already occupies that xi. Preserve the proxy leader,
    # then add splitting-scale diversity for subsequent actual BAOAB tests.
    if alternatives:proposed.append(alternatives[0])
    for x in alternatives:
        if x[0]['xi'] not in {y[0]['xi'] for y in proposed}:proposed.append(x)
        if len(proposed)>=max_final:break
    for x in alternatives:
        if len(proposed)>=max_final:break
        if x[0]['setting_id'] not in {y[0]['setting_id'] for y in proposed}:proposed.append(x)
    qualified=[]
    for row,first in proposed:
        par=first['params'];checks=[first]
        for state in cert:
            if state['label']==initial['label']:continue
            rec=certify_state(state,par,out,save_matrix=True);checks.append(rec);full.append(rec)
            print('Spatial full domain',row['setting_id'],state['label'],rec['spatial_error_conservative_bound'],flush=True)
        good=all(r['spatial_passed'] and r['structural_checks_passed'] for r in checks)
        if good:qualified.append(dict(setting_id=row['setting_id'],params=par,spatial_passed=True,
            max_spatial_error=max(r['spatial_error_conservative_bound'] for r in checks),validation_states=[s['label'] for s in cert],
            priority_proxy_s=row['priority_proxy_s'],proxy_not_performance_winner=True,
            baseline_control=row['setting_id']==baseline))
        else:print('Spatial finalist rejected multigeometry',row['setting_id'],flush=True)
    write_csv(out/'full_spatial_checks.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in full])
    result=dict(qualified_settings=qualified,candidates_screened=len(rows),initial_full_candidates=min(max_full,len(order)),
        actual_full_operator_checks=len(full),registered_validation_states=[s['label'] for s in cert],
        spatial_budget=SPATIAL_TOL,baseline_setting_id=baseline,adaptive_decisions=decisions,
        max_unique_candidates=60,full_candidate_budget=max_full,finalist_budget=max_final,
        local_budget_reduction='Small adaptive search, max9 initial full candidates and4 final settings; not global optimization',
        search_elapsed_s=perf_counter()-started,
        no_qualified_speed_claim='Actual dynamic BAOAB benchmark and thermostat constraints decide method-specific leaders later')
    save(marker,result)
    if len(qualified)<2:print('WARNING actualqualified settings fewerthan2',len(qualified),flush=True)
    print('Qualified settings',[(r['setting_id'],r['params'],r['max_spatial_error']) for r in qualified],flush=True)
    return result

def extend_resolution_leader(states,out=None):
    """Preserve the initial run and add the omitted same-xi resolution leader.

    This follow-up responds to the observed screen results. It does not use a
    dynamic-time winner label; it only broadens eligible settings before any
    actual BAOAB head-to-head timing.
    """
    import shutil
    out=RESULTS/'spatial' if out is None else Path(out)
    marker=out/'spatial_selection.json';result=js(marker)
    original=out/'spatial_selection_initial.json'
    if not original.exists():shutil.copyfile(marker,original)
    current={r['setting_id'] for r in result['qualified_settings']}
    rows=read_csv(out/'cheap_screen.csv')
    eligible=[]
    for row in rows:
        if row.get('cheap_screen_rejected') or row['setting_id'] in current:continue
        check=out/f"spatial_{row['setting_id']}_initial_0.json"
        if check.exists():
            rec=js(check)
            if rec['spatial_passed'] and rec['structural_checks_passed']:eligible.append((row,rec))
    if not eligible:return result
    row,first=min(eligible,key=lambda x:x[0]['priority_proxy_s'])
    labels=result['registered_validation_states'];checks=[];newrows=[]
    for state in states:
        if state['label'] not in labels:continue
        path=out/f"spatial_{row['setting_id']}_{state['label']}.json"
        if path.exists():rec=js(path)
        else:
            rec=certify_state(state,first['params'],out,save_matrix=True)
            newrows.append({k:v for k,v in rec.items() if not isinstance(v,(dict,list))})
        checks.append(rec)
        print('Adaptive resolution leader',row['setting_id'],state['label'],rec['spatial_error_conservative_bound'],flush=True)
    good=all(r['spatial_passed'] and r['structural_checks_passed'] for r in checks)
    result['adaptive_resolution_followup']=dict(setting_id=row['setting_id'],
        reason='Preserve cheapest observed setup/action proxy despite samexi as baseline; then qualify onall5states before actualdynamicoptimization',
        passed=good,actual_new_full_operator_checks=len(newrows))
    if good:
        result['qualified_settings'].append(dict(setting_id=row['setting_id'],params=first['params'],spatial_passed=True,
            max_spatial_error=max(r['spatial_error_conservative_bound'] for r in checks),validation_states=labels,
            priority_proxy_s=row['priority_proxy_s'],proxy_not_performance_winner=True,baseline_control=False))
    result['actual_full_operator_checks']+=len(newrows);result['finalist_budget']=4
    result['local_budget_reduction']='Adaptive local search max9 initial full candidates and4 qualified settings; not global optimization'
    full=read_csv(out/'full_spatial_checks.csv');write_csv(out/'full_spatial_checks.csv',full+newrows)
    save(marker,result)
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--extend-resolution-leader',action='store_true');args=parser.parse_args()
    result=extend_resolution_leader(load_states()) if args.extend_resolution_leader else run_search(load_states())
    print(json.dumps(result['qualified_settings'],indent=2),flush=True)
