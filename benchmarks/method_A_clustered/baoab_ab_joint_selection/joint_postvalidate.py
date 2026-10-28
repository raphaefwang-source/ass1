"""Post-head physical/transition certification on actual timed midpoints.

Run only after ALL dynamic timing has ended. Each unique head comparison
contributes the final sampled midpoint of all three initial states and both
methods. Duplicate comparison groups explicitly reuse the identical original
group's validation, never fabricate extra trajectories or independent samples.
"""
from __future__ import annotations
from joint_common import *
from time import perf_counter
from model import lj_force,rng_sha
from joint_spatial import prepare_setting
from joint_thermo import check_config_state
from stochastic_validation import action_of,exact_action
from ou_core import ou_a,ou_b,lanczos_action


def select_endpoints(heads):
    """Last actual round for each unique(group,method,initial) control."""
    groups={g['group_id']:g for g in heads['groups']};runs=heads['runs'];chosen={}
    for run in runs:
        gid=run['group_id']
        if gid not in groups or groups[gid].get('reuse_group_id'):
            raise ValueError('Actual head run must belong to one unique comparison')
        key=(gid,run['method'],int(run['initial_index']))
        if key not in chosen or int(run['round'])>int(chosen[key]['round']):chosen[key]=run
    records=[]
    for group in heads['groups']:
        if group.get('reuse_group_id'):continue
        for i in (0,1,2):
            for method in ('A','B'):
                key=(group['group_id'],method,i)
                if key not in chosen:raise ValueError('Need all three starts and both methods in each unique head group')
                records.append(chosen[key])
    return records


def reconstruct_draws(config,p_shape,rng_record,inner_rows,edge_count):
    """Recreate every actual inner Gaussian from the saved BEFORE PCG64 state."""
    k=int(config['k']);method=config['method'];issues=[]
    if method=='B' and k!=1:raise ValueError('B must use a single full step')
    if len(inner_rows)!=k:raise ValueError('Incomplete saved inner-step evidence')
    rng=np.random.default_rng();rng.bit_generator.state=rng_record['before']
    draws=[];rows=[]
    for l,row in enumerate(inner_rows):
        if int(row['substep'])!=l+1:raise ValueError('Inner Gaussian order mismatch')
        before=rng_sha(rng);z=rng.standard_normal(p_shape);ef=er=None
        if method=='A':
            ef=rng.standard_normal((int(edge_count),3));er=rng.standard_normal((int(edge_count),3))
        actual=dict(substep=l+1,z_sha256=array_sha(z),
            eta_forward_sha256=array_sha(ef) if ef is not None else None,
            eta_reverse_sha256=array_sha(er) if er is not None else None,
            rng_state_before_sha256=before,rng_state_after_sha256=rng_sha(rng))
        for name,value in actual.items():
            if row[name]!=value:issues.append(f'inner{l+1}: '+name)
        draws.append(dict(z=z,ef=ef,er=er));rows.append(actual)
    if rng.bit_generator.state!=rng_record['after']:issues.append('outer RNG afterstate')
    if issues:raise RuntimeError('Actual saved Gaussian reconstruction mismatch: '+str(issues))
    return draws,rows


def finite_actual_noise_check(prepared,config,p_half,draws,po_saved):
    """Actual finite actions versus own-method coupled dense reference.

    Full physical mean/covariance norms are certified separately. This checks
    the ACTUAL timed momentum and all actual independent noise arrays, rather
    than interpreting unrelated A/B sample distances as an algorithm bias.
    """
    method=config['method'];k=int(config['k']);rd=int(config['r_d']);rn=int(config['r_n'])
    if len(draws)!=k:raise ValueError('Exactly one independent draw dictionary per inner step required')
    action=action_of(prepared,method);ref=prepared['long_ref' if method=='A' else 'full_ref']
    h=DT/k if method=='A' else DT;R,F,_=ref.matrices(h)
    actual=np.asarray(p_half,float).copy();ideal=actual.copy();shape=actual.shape
    rows=[];path_rows=[];raw=dict(p_half=actual.copy());max_action=0.;negative=0
    action_budget=float(config['certificate']['per_action_budget'])
    path_budget=float(config['certificate']['full_path_budget'])
    for l,draw in enumerate(draws):
        z=draw['z'];ef=draw['ef'];er=draw['er']
        if method=='A':
            damp_input=prepared['edges'].sweep(actual,ef,h/2)[0]
            ideal_forward=prepared['edges'].sweep(ideal,ef,h/2)[0]
        else:damp_input=actual.copy();ideal_forward=ideal.copy()
        for kind,vector,rank in [('damping',damp_input,rd),('noise',z,rn)]:
            expected=exact_action(ref,vector,h,kind)
            finite,diag=lanczos_action(action,vector,rank,h,kind)
            error=float(np.linalg.norm(finite-expected)/max(np.linalg.norm(expected),np.finfo(float).tiny))
            max_action=max(max_action,error)
            rows.append(dict(substep=l+1,kind=kind,method=method,k=k,rank_requested=rank,
                rank_used=diag['rank_used'],relative_error=error,budget=action_budget,passed=error<=action_budget,
                negative_ritz_count=diag['negative_ritz_count'],clipped_negative_count=diag['clipped_negative_count'],
                matvec_count=diag['matvec_count'],input_sha256=array_sha(vector),
                exact_sha256=array_sha(expected),actual_sha256=array_sha(finite)))
            raw[f'sub{l+1}_{kind}_input']=vector.copy();raw[f'sub{l+1}_{kind}_exact']=expected
            raw[f'sub{l+1}_{kind}_finite']=finite
        if method=='A':
            actual,diag=ou_a(action,actual,z,ef,er,prepared['edges'],h,rd,rn)
            middle=(R@ideal_forward.ravel()+F@z.ravel()).reshape(shape)
            ideal=prepared['edges'].sweep(middle,er,h/2,reverse=True)[0]
        else:
            actual,diag=ou_b(action,actual,z,DT,rd,rn)
            ideal=(R@ideal_forward.ravel()+F@z.ravel()).reshape(shape)
        negative+=int(diag['negative_ritz_count'])
        raw[f'sub{l+1}_actual']=actual.copy();raw[f'sub{l+1}_own_ideal']=ideal.copy()
        path_rows.append(dict(substep=l+1,dt_inner=h,k=k,method=method,
            actual_sha256=array_sha(actual),own_ideal_sha256=array_sha(ideal),
            coupled_relative_error=float(np.linalg.norm(actual-ideal)/max(np.linalg.norm(ideal),np.finfo(float).tiny)),
            matvec_count=diag['matvec_count'],r_d_used=diag['damp_rank_used'],r_n_used=diag['noise_rank_used'],
            negative_ritz_count=diag['negative_ritz_count']))
    error=float(np.linalg.norm(actual-ideal)/max(np.linalg.norm(ideal),np.finfo(float).tiny))
    replay_abs=float(np.max(np.abs(actual-po_saved)))
    replay_relative=float(np.linalg.norm(actual-po_saved)/max(np.linalg.norm(po_saved),np.finfo(float).tiny))
    # The saved O output is reconstructed by subtracting the second force kick;
    # that floating point inverse need not be bitwise identical to p_after_O.
    timed_output_match=bool(np.allclose(actual,po_saved,rtol=1e-12,atol=1e-12))
    com=float(np.linalg.norm(actual.reshape(-1,3).sum(0)-np.asarray(p_half).reshape(-1,3).sum(0)))
    passed=bool(all(r['passed'] for r in rows) and error<=path_budget and negative==0 and timed_output_match and com<=1e-9)
    record=dict(method=method,k=k,r_d=rd,r_n=rn,actual_noise_reconstructed=True,
        max_actual_input_action_error=max_action,per_action_budget=action_budget,
        coupled_own_ideal_path_error=error,full_path_budget=path_budget,
        timed_O_output_match=timed_output_match,timed_O_output_replay_abs_error=replay_abs,
        timed_O_output_replay_relative_error=replay_relative,
        negative_ritz_count=negative,COM_error=com,finite_passed=passed,
        action_checks=rows,substep_paths=path_rows,
        scope='actual timed inner inputs/noise, finite own-ideal path; no nonlinear covariance operator or Gibbs guarantee')
    raw.update(actual_O=actual,own_ideal_O=ideal,timed_O_reconstructed=po_saved)
    return record,raw


def validate_endpoint(run,out):
    label=run['run'];config=run['config'];path=HERE/run['file']
    if sha(path)!=run['file_sha256']:raise RuntimeError('Timed trajectory file SHA changed')
    rows=read_csv(RESULTS/f'{label}_steps.csv');row=rows[-1];step=int(row['step'])
    inner=[r for r in read_csv(RESULTS/f'{label}_substeps.csv') if int(r['outer_step'])==step]
    rng_file=RESULTS/f'{label}_rng.json';rng_data=js(rng_file)['states'][-1]
    with np.load(path,allow_pickle=False) as d:
        q=d['q_mid'][-1].copy();q0=d['q_unwrapped'][-2].copy();q1=d['q_unwrapped'][-1].copy()
        p0=d['p'][-2].copy();p1=d['p'][-1].copy();L=float(d['L'])
    if array_sha(q)!=row['q_mid_sha256']:raise RuntimeError('Actual midpoint hash mismatch')
    p_half=p0+.5*DT*lj_force(np.mod(q0,L),L)[0]
    po_saved=p1-.5*DT*lj_force(np.mod(q1,L),L)[0]
    state=dict(label='endpoint_'+label,q=q,p=p_half,
        metadata=dict(source=str(path.relative_to(REPO)),source_sha256=sha(path),step=step))
    prepared=prepare_setting(state,config['params'])
    cert=check_config_state(config,state,out=out,prepared=prepared)
    draws,noise_rows=reconstruct_draws(config,p_half.shape,rng_data,inner,int(row['edge_count']))
    finite,raw=finite_actual_noise_check(prepared,config,p_half,draws,po_saved)
    finite_path=Path(out)/f'finite_{label}.npz'
    np.savez_compressed(finite_path,q=q,**raw)
    finite_record=Path(out)/f'finite_{label}.json'
    finite.update(run=label,config_id=config['config_id'],q_mid_sha256=array_sha(q),p_half_sha256=array_sha(p_half),
        file=str(finite_path.relative_to(HERE)),file_sha256=sha(finite_path),
        source_rng_file=str(rng_file.relative_to(HERE)),source_rng_file_sha256=sha(rng_file),
        actual_noise_substeps=noise_rows)
    save(finite_record,finite)
    cert.update(kind='head_endpoint',run=label,group_id=run['group_id'],initial_index=run['initial_index'],
        round=run['round'],step=step,method=config['method'],spatial_id=config['spatial_id'],
        k=config['k'],r_d=config['r_d'],r_n=config['r_n'],
        actual_input_p_half_sha256=array_sha(p_half),timed_source_file=run['file'],timed_source_file_sha256=sha(path),
        finite=finite,finite_record_file=str(finite_record.relative_to(HERE)),
        physical_ideal_passed=cert['ideal_physical_passed'],
        passed=bool(cert['passed'] and finite['finite_passed']))
    return cert


def run_postvalidation():
    marker=RESULTS/'post_validation.json'
    if marker.exists():raise FileExistsError('Preserve completed post-head validation')
    # All dynamic batches must have ended before independent reference work.
    heads=js(RESULTS/'heads_done.json')
    if not (RESULTS/'exploratory_done.json').exists():raise RuntimeError('Need completed exploratory batch')
    source_files=[HERE/'joint_postvalidate.py',HERE/'joint_spatial.py',HERE/'joint_thermo.py',HERE/'joint_common.py',
        MATCHED/'matched_dynamics.py',PILOT/'model.py']
    source_hashes={str(p.relative_to(REPO)):sha(p) for p in source_files}
    selected=select_endpoints(heads);out=RESULTS/'post_certificates';out.mkdir(parents=True,exist_ok=True)
    records=[];scalar=[];finite_rows=[];started=perf_counter()
    for run in selected:
        rec=validate_endpoint(run,out);records.append(rec)
        scalar.append({k:v for k,v in rec.items() if not isinstance(v,(dict,list))})
        finite_rows.extend([dict(r,run=run['run'],group_id=run['group_id'],
            config_id=run['config_id'],epsilon_frozen=run['epsilon_frozen']) for r in rec['finite']['action_checks']])
        write_csv(RESULTS/'post_validation.csv',scalar);write_csv(RESULTS/'post_finite_checks.csv',finite_rows)
        save(RESULTS/'post_validation_progress.json',dict(records=records,completed=len(records),required=len(selected)))
        print('Post endpoint',run['run'],'space',rec['spatial_error_bound'],
            'physical mean/cov',rec['physical_mean_bound'],rec['physical_cov_bound'],
            'actual path',rec['finite']['coupled_own_ideal_path_error'],'passed',rec['passed'],flush=True)
    groups=[]
    for group in heads['groups']:
        source=group.get('reuse_group_id',group['group_id'])
        rr=[r for r in records if r['group_id']==source]
        if len(rr)!=6:raise RuntimeError('Every comparison needs six actual new endpoint checks or explicit identical-group reuse')
        groups.append(dict(group_id=group['group_id'],source_group_id=source,
            comparison_type=group['comparison_type'],epsilon_frozen=group['epsilon_frozen'],
            reused=source!=group['group_id'],sampled_midpoints=len(rr),runs=[r['run'] for r in rr],
            spatial_passed=all(r['spatial_passed'] for r in rr),
            physical_ideal_passed=all(r['ideal_physical_passed'] for r in rr),
            finite_actual_noise_passed=all(r['finite']['finite_passed'] for r in rr),
            passed=all(r['passed'] for r in rr),
            max_spatial_error=max(r['spatial_error_bound'] for r in rr),
            max_physical_mean_bound=max(r['physical_mean_bound'] for r in rr),
            max_physical_cov_bound=max(r['physical_cov_bound'] for r in rr),
            max_actual_finite_path_error=max(r['finite']['coupled_own_ideal_path_error'] for r in rr)))
    result=dict(records=records,groups=groups,actual_sampled_midpoints=len(records),
        unique_geometries=len({r['q_sha256'] for r in records}),
        all_passed=all(r['passed'] for r in records),elapsed_s=perf_counter()-started,
        actual_noise_replayed=True,execution_source_sha256=source_hashes,
        scope='Six actual sampled midpoints per unique head comparison; full physical COM norms plus actual-input/noise finite paths, not an all-trajectory or equal physical-observable guarantee')
    save(marker,result);return result


if __name__=='__main__':
    result=run_postvalidation();print('Post validation complete',result['actual_sampled_midpoints'],'all_passed',result['all_passed'],flush=True)
