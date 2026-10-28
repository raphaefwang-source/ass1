"""Actual matched-accuracy dynamic timing; every O loop uses selected k."""
from matched_common import *
from matched_dynamics import MatchedSystem
import argparse
import copy
import os
import resource
import time
import psutil

TIMING_SEED=202610281020

def protocol():
    accuracy=js(RESULTS/'accuracy/frozen_selection.json')
    stochastic=js(RESULTS/'stochastic/stochastic_selection.json')
    if not stochastic['all_checks_passed']:raise RuntimeError('Stochastic validation must pass before any timing')
    for marker in ('common_probe_done.json','mc_done.json','wrapper_validation.json'):
        if not (RESULTS/'stochastic'/marker).exists():raise RuntimeError('Complete common-rank and MC validation before timing')
    if not js(RESULTS/'stochastic/common_probe_done.json')['passed']:
        raise RuntimeError('Common final-rank probes failed')
    if not js(RESULTS/'stochastic/wrapper_validation.json')['passed']:
        raise RuntimeError('Actual production wrapper failed independent path crosscheck')
    return accuracy,stochastic

def initial(i):
    record=next(r for r in js(RESULTS/'domain.json')['states'] if r['label']==f'initial_{i}')
    source=REPO/record['source']
    if sha(source)!=record['source_sha256']:raise RuntimeError('PR10 canonical start changed')
    with np.load(source,allow_pickle=False) as d:return d['q_unwrapped'].copy(),d['p'].copy()

def run(system,label,initial_index,method,k,ranks,steps,seed,epsilon=None,round_index=None,order_position=None):
    if (RESULTS/f'{label}.json').exists():raise FileExistsError('Preserve completed run '+label)
    q,p=initial(initial_index);rng=np.random.default_rng(int(seed))
    qh=[q.copy()];ph=[p.copy()];mid=[];rows=[];inner=[];states=[]
    load_before=os.getloadavg();started=time.perf_counter()
    for j in range(steps):
        before=copy.deepcopy(rng.bit_generator.state)
        q,p,row=system.step(q,p,method,**ranks,rng=rng,dt=.005,k=k)
        row.update(run=label,step=j+1,time=(j+1)*.005,initial_index=initial_index,
            seed=int(seed),epsilon_frozen=epsilon,round=round_index,order_position=order_position,
            rss_bytes=psutil.Process().memory_info().rss,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rows.append(row)
        inner.extend([dict(r,run=label,outer_step=j+1,epsilon_frozen=epsilon,
                           round=round_index,initial_index=initial_index) for r in system.last_substep_rows])
        states.append(dict(step=j+1,before=before,after=copy.deepcopy(rng.bit_generator.state)))
        qh.append(q.copy());ph.append(p.copy());mid.append(system.last_built.q_mid.copy())
    loop_wall=time.perf_counter()-started
    file=INPUTS/f'{label}.npz'
    np.savez_compressed(file,q_unwrapped=np.asarray(qh),q=np.mod(qh,system.L),p=np.asarray(ph),q_mid=np.asarray(mid),L=system.L)
    write_csv(RESULTS/f'{label}_steps.csv',rows)
    write_csv(RESULTS/f'{label}_substeps.csv',inner)
    save(RESULTS/f'{label}_rng.json',dict(seed=int(seed),bit_generator='PCG64',states=states,
        recipe='Each A inner substep draws z,ef,er independently in this order; B draws only z'))
    summary=dict(run=label,method=method,k=k,ranks=ranks,steps=steps,dt=.005,
        epsilon_frozen=epsilon,initial_index=initial_index,seed=int(seed),round=round_index,order_position=order_position,
        file=str(file.relative_to(HERE)),file_sha256=sha(file),
        start_q_sha256=array_sha(qh[0]),start_p_sha256=array_sha(ph[0]),
        end_q_sha256=array_sha(q),end_p_sha256=array_sha(p),
        full_BAOAB_mean_s=float(np.mean([r['t_baoab_s'] for r in rows])),
        full_BAOAB_median_s=float(np.median([r['t_baoab_s'] for r in rows])),
        compute_wall_s=sum(r['t_baoab_s'] for r in rows),loop_wall_s=loop_wall,
        dynamic_setup_mean_s=float(np.mean([r['t_dynamic_setup_s'] for r in rows])),
        inner_calls=len(inner),max_COM_norm=max(r['momentum_norm'] for r in rows),
        min_pair_distance=min(r['min_pair_distance'] for r in rows),
        max_peak_rss_bytes=max(r['peak_rss_bytes'] for r in rows),
        table_sha256=system.table_sha256,load_before=load_before,load_after=os.getloadavg(),
        scope='true moving-particle BAOAB; one fresh graph per outer midpoint; selected k active')
    save(RESULTS/f'{label}.json',summary)
    print(label,'steps',steps,'k',k,'mean_BAOAB_ms',1000*summary['full_BAOAB_mean_s'],flush=True)
    return rows,inner,summary

def timing():
    if (RESULTS/'timing_done.json').exists():raise FileExistsError('Preserve completed timing')
    accuracy,stochastic=protocol()
    targets=sorted(accuracy['targets'],key=lambda t:t['epsilon_frozen'])
    system=MatchedSystem(PARAMS)
    save(RESULTS/'timing_table.json',dict(sha256=system.table_sha256,wall_s=system.t_parameter_table_setup_s,
        physics=PARAMS,graph_rebuilt_each_step=True,parameter_table_reused=True))
    rr=[];inner=[];rounds=[];summaries=[]
    for target_index,target in enumerate(targets):
        eps=float(target['epsilon_frozen']);k=target['k']
        if k is None:
            summaries.append(dict(epsilon_frozen=eps,status='accuracy_not_met',k=None));continue
        rank=stochastic['by_k'][str(k)]
        order_rng=np.random.default_rng(TIMING_SEED+target_index)
        for r in range(6):
            order=list(order_rng.permutation(['A','B']));pair={}
            for position,method in enumerate(order):
                label=f'matched_t{target_index}_round{r}_{method}'
                rows,sub,summary=run(system,label,r%3,method,int(k) if method=='A' else 1,
                    rank[method],20,TIMING_SEED+10000+target_index*1000+r*10+(method=='B'),
                    epsilon=eps,round_index=r,order_position=position)
                rr.extend(rows);inner.extend(sub);pair[method]=summary;summaries.append(summary)
            rounds.append(dict(epsilon_frozen=eps,k=int(k),round=r,initial_index=r%3,order=','.join(order),
                A_mean_s=pair['A']['full_BAOAB_mean_s'],B_mean_s=pair['B']['full_BAOAB_mean_s'],
                A_over_B=pair['A']['full_BAOAB_mean_s']/pair['B']['full_BAOAB_mean_s']))
    write_csv(RESULTS/'timing_steps.csv',rr);write_csv(RESULTS/'timing_substeps.csv',inner)
    write_csv(RESULTS/'timing_rounds.csv',rounds)
    save(RESULTS/'timing_done.json',dict(actual_moving_steps=len(rr),actual_inner_O_calls=len(inner),
        rounds_per_target=6,steps_per_method_per_round=20,targets=targets,runs=summaries,
        seed=TIMING_SEED,precision_order='strict first',thread_request='native one thread',
        control='Each A/B round restarts same PR10 initial q,p; each independently advances20 actual moving steps; starts rotate0/1/2 twice',
        scope='frozen-transition-accuracy-matched dynamic cost, not equal physical-observable accuracy'))

def cost_curve():
    if (RESULTS/'cost_curve_done.json').exists():raise FileExistsError('Preserve completed curve')
    accuracy,stochastic=protocol();system=MatchedSystem(PARAMS)
    selected={int(t['k']) for t in accuracy['targets'] if t['k'] is not None}
    rows=[];inner=[];summaries=[]
    # Matched tiers already supply120 genuine A steps/k. Other k receive3starts×3 moving steps.
    for kstr,rank in sorted(stochastic['by_k'].items(),key=lambda kv:int(kv[0])):
        k=int(kstr)
        if k in selected:continue
        for i in range(3):
            for method in (('A','B') if i%2==0 else ('B','A')):
                label=f'curve_k{k}_start{i}_{method}'
                rr,ss,meta=run(system,label,i,method,k if method=='A' else 1,rank[method],3,
                    TIMING_SEED+100000+k*100+i*10+(method=='B'))
                rows.extend(rr);inner.extend(ss);summaries.append(meta)
    write_csv(RESULTS/'cost_curve_extra_steps.csv',rows);write_csv(RESULTS/'cost_curve_extra_substeps.csv',inner)
    save(RESULTS/'cost_curve_done.json',dict(actual_extra_moving_steps=len(rows),actual_inner_O_calls=len(inner),
        selected_k_from_matched_batch=sorted(selected),runs=summaries,
        scope='nonselected k cost-curve points are short9 A steps plus9 B control steps/k; not independently matched accuracy tiers'))

def post_validate():
    """Selected-k fullnorm checks on18 actual new timing endpoint midpoints."""
    from frozen_accuracy import prepare,compare_maps
    from spatial_validation import build_mesh
    from model import lj_force
    if (RESULTS/'post_timing_validation.json').exists():raise FileExistsError('Preserve endpoint validation')
    done=js(RESULTS/'timing_done.json');mesh=build_mesh(PARAMS,PARAMS['L']);records=[]
    for index,target in enumerate(done['targets']):
        if target['k'] is None:continue
        for r in (3,4,5):
            for method in ('A','B'):
                label=f'matched_t{index}_round{r}_{method}'
                with np.load(INPUTS/f'{label}.npz',allow_pickle=False) as d:
                    q=d['q_mid'][-1].copy();q0=d['q_unwrapped'][-2].copy()
                    p=d['p'][-2].copy()+.0025*lj_force(np.mod(q0,PARAMS['L']),PARAMS['L'])[0]
                prepared=prepare(q,PARAMS,mesh=mesh)
                rec,maps=compare_maps(prepared,int(target['k']))
                path=RESULTS/f'endpoint_{label}.npz'
                # GH/GL plus deterministic Helmert basis reproduce RB/CB independently.
                np.savez_compressed(path,q=q,p_half=p,GH=prepared['GH'],GL=prepared['GL'],**maps)
                rec.update(run=label,method=method,step=20,epsilon_frozen=target['epsilon_frozen'],
                    q_mid_sha256=array_sha(q),p_half_sha256=array_sha(p),
                    GH_sha256=array_sha(prepared['GH']),GL_sha256=array_sha(prepared['GL']),
                    file=str(path.relative_to(HERE)),file_sha256=sha(path),
                    passed=max(rec['e_mean'],rec['e_cov'])<=target['epsilon_frozen'])
                records.append(rec)
                write_csv(RESULTS/'post_timing_errors.csv',records)
                print('Endpoint',label,'k',target['k'],'mean',rec['e_mean'],'cov',rec['e_cov'],'passed',rec['passed'],flush=True)
    save(RESULTS/'post_timing_validation.json',dict(records=records,sampled_new_midpoints=len(records),
        all_passed=all(r['passed'] for r in records),
        scope='18 actual new endpoint midpoints, full765D norms; sampled-state evidence not a uniform trajectory guarantee'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['timing','cost_curve','post_validate'])
    args=parser.parse_args();globals()[args.action]()
