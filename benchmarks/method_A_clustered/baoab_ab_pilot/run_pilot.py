"""Stage 0/1 local experiment driver. No later-stage study is implemented."""
from evidence import HERE,RESULTS,INPUTS,LOGS,save,write_csv,array_sha,sha
import argparse
import copy
import json
import os
import resource
import time
import numpy as np
import psutil
from model import make_initial,lj_force
from dynamics import PilotSystem

INITIAL_SEEDS=(202610280101,202610280102,202610280103)
TRAJECTORY_SEEDS={'A':202610280201,'B':202610280301}
TIMING_SEED=202610280401

def load_json(path):return json.loads(path.read_text())
def initial(index):
    path=INPUTS/f'initial_{index}.npz'
    with np.load(path,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}
def settings():
    return load_json(RESULTS/'spatial_selection.json')['params'],load_json(RESULTS/'ranks_final.json')

def initialize():
    for i,seed in enumerate(INITIAL_SEEDS):
        path=INPUTS/f'initial_{i}.npz'
        if path.exists():
            if not (RESULTS/f'initial_{i}.json').exists():raise RuntimeError('Incomplete initial evidence')
            print('Preserving completed start',i,flush=True);continue
        arrays,rows,meta=make_initial(seed,steps=6000,dt=.005)
        np.savez_compressed(path,**arrays)
        meta.update(file=str(path.relative_to(HERE)),file_sha256=sha(path))
        write_csv(RESULTS/f'initial_{i}_preparation.csv',rows)
        save(RESULTS/f'initial_{i}.json',meta)
        print('Prepared',i,'Tkin',meta['final_Tkin'],'wall',meta['total_wall_s'],flush=True)

def stage0():
    from spatial_validation import validate_state
    from rank_validation import select_ranks,validate_actions
    params=load_json(RESULTS/'spatial_selection.json')['params']
    selections=[];spatial=[]
    for i in range(3):
        d=initial(i)
        sp=validate_state(d['q'],params,RESULTS/f'initial_{i}_spatial.json')
        if not sp['exploratory_budget_met']:raise RuntimeError('Initial spatial accuracy not met')
        spatial.append(sp)
        rank=select_ranks(d['q'],d['p'],params,RESULTS,tag=f'initial_{i}')
        selections.append(rank)
        print('Initial Stage0',i,'spatial',sp['conservative_relative_error_bound'],'ranks',rank,flush=True)
    ranks=dict(A={k:max(x['A'][k] for x in selections) for k in ('r_d','r_n')},
               B={k:max(x['B'][k] for x in selections) for k in ('r_d','r_n')},
               threshold=2e-8,substeps=1,
               selection='maximum of separately verified ranks over 3 independent pre-equilibrated starts')
    checks=[]
    for i in range(3):
        d=initial(i)
        checks.append(validate_actions(d['q'],d['p'],params,ranks,RESULTS,tag=f'initial_{i}_final',splitting=True))
    save(RESULTS/'ranks_final.json',ranks)
    save(RESULTS/'stage0_done.json',dict(completed=True,strict_spatial_all_starts=all(x['strict_budget_met'] for x in spatial),
        initial_states=3,ranks=ranks,checks=checks,stage=0))

def run_trajectory(system,index,method,steps,seed,label):
    if steps not in (40,200):raise ValueError('Only Stage1 40/200-step trajectories are permitted')
    params,ranks=settings();d=initial(index)
    q,p=d['q_unwrapped'].copy(),d['p'].copy()
    rng=np.random.default_rng(int(seed))
    qh=[q.copy()];ph=[p.copy()];mid=[];rows=[];rng_states=[]
    t=time.perf_counter();load_before=os.getloadavg()
    for j in range(steps):
        state_before=copy.deepcopy(rng.bit_generator.state)
        q,p,row=system.step(q,p,method,**ranks[method],rng=rng,dt=.005)
        row.update(step=j+1,time=(j+1)*.005,initial_index=index,seed=int(seed),run=label,
                   rss_bytes=psutil.Process().memory_info().rss,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rows.append(row);qh.append(q.copy());ph.append(p.copy());mid.append(system.last_built.q_mid.copy())
        rng_states.append(dict(step=j+1,before=state_before,after=copy.deepcopy(rng.bit_generator.state)))
        if (j+1)%40==0:print(label,j+1,'/',steps,'T',row['Tkin'],'wall mean',np.mean([r['t_baoab_s'] for r in rows]),flush=True)
    loop_wall=time.perf_counter()-t
    path=INPUTS/f'{label}.npz'
    np.savez_compressed(path,q_unwrapped=np.asarray(qh),q=np.mod(qh,system.L),p=np.asarray(ph),q_mid=np.asarray(mid),L=system.L)
    write_csv(RESULTS/f'{label}_steps.csv',rows)
    save(RESULTS/f'{label}_rng.json',dict(seed=int(seed),bit_generator=type(rng.bit_generator).__name__,states=rng_states,
        recipe='one PCG64 stream; each A step draws z_long, eta_forward, eta_reverse in order; B draws z_full',
        independent_methods=True))
    summary=dict(label=label,method=method,initial_index=index,seed=int(seed),steps=steps,dt=.005,
                 simulated_time=steps*.005,file=str(path.relative_to(HERE)),file_sha256=sha(path),
                 start_q_sha256=array_sha(qh[0]),start_p_sha256=array_sha(ph[0]),
                 end_q_sha256=array_sha(q),end_p_sha256=array_sha(p),
                 full_BAOAB_mean_s=float(np.mean([r['t_baoab_s'] for r in rows])),
                 full_BAOAB_median_s=float(np.median([r['t_baoab_s'] for r in rows])),
                 loop_wall_including_evidence_s=loop_wall,
                 BAOAB_compute_wall_s=float(sum(r['t_baoab_s'] for r in rows)),
                 max_momentum_norm=max(r['momentum_norm'] for r in rows),
                 min_pair_distance=min(r['min_pair_distance'] for r in rows),
                 max_rss_bytes=max(r['peak_rss_bytes'] for r in rows),
                 load_before=load_before,load_after=os.getloadavg(),
                 ranks=ranks[method],parameter_table_sha256=system.table_sha256,
                 scope='true moving-particle BAOAB; setup rebuilt every q_mid; exploratory Stage1 pilot')
    save(RESULTS/f'{label}.json',summary)
    return rows,summary

def trajectories():
    params,ranks=settings()
    if not load_json(RESULTS/'stage0_done.json')['completed']:raise RuntimeError('Stage0 required')
    system=PilotSystem(params);allrows=[];summaries=[]
    save(RESULTS/'production_table.json',dict(wall_s=system.t_parameter_table_setup_s,sha256=system.table_sha256,
        reused_across_steps=True,scope='parameter-only; outside production step; fresh q data each step'))
    for index in range(3):
        for method in ('A','B'):
            label=f'production_{index}_{method}'
            if (RESULTS/f'{label}.json').exists():raise RuntimeError('Completed production evidence exists; refusing overwrite')
            rows,summary=run_trajectory(system,index,method,200,TRAJECTORY_SEEDS[method]+index,label)
            allrows.extend(rows);summaries.append(summary)
    write_csv(RESULTS/'production_steps.csv',allrows)
    save(RESULTS/'production_done.json',dict(stage=1,actual_steps=len(allrows),trajectories=summaries,
        parameter_table_setup_s=system.t_parameter_table_setup_s))

def timing():
    params,ranks=settings()
    if (RESULTS/'timing_done.json').exists():raise RuntimeError('Completed paired timing exists')
    system=PilotSystem(params);order_rng=np.random.default_rng(TIMING_SEED)
    rows=[];rounds=[]
    for round_index in range(6):
        order=list(order_rng.permutation(['A','B']));per=[]
        for position,method in enumerate(order):
            label=f'timing_round{round_index}_{method}'
            rr,ss=run_trajectory(system,round_index%3,method,40,TIMING_SEED+1000+round_index*10+(method=='B'),label)
            for r in rr:r.update(round=round_index,order_position=position)
            rows.extend(rr);per.append(ss)
        by={x['method']:x for x in per}
        rounds.append(dict(round=round_index,initial_index=round_index%3,order=','.join(order),
            A_mean_s=by['A']['full_BAOAB_mean_s'],B_mean_s=by['B']['full_BAOAB_mean_s'],
            A_median_s=by['A']['full_BAOAB_median_s'],B_median_s=by['B']['full_BAOAB_median_s'],
            A_over_B=by['A']['full_BAOAB_mean_s']/by['B']['full_BAOAB_mean_s']))
    write_csv(RESULTS/'timing_steps.csv',rows);write_csv(RESULTS/'timing_rounds.csv',rounds)
    save(RESULTS/'timing_done.json',dict(stage=1,actual_steps=len(rows),rounds=6,steps_per_method_per_round=40,
        order_seed=TIMING_SEED,parameter_table_setup_s=system.t_parameter_table_setup_s,
        parameter_table_sha256=system.table_sha256,
        control='Each round A/B restart from same q,p start; independent correct internal noise; both advance 40 true moving steps',
        scope='exploratory dynamic timing only; no equal-dynamics ranking'))

def visited():
    from spatial_validation import validate_state
    from rank_validation import validate_actions
    params,ranks=settings();records=[];samples=[]
    for i in range(3):
        for method in ('A','B'):
            label=f'production_{i}_{method}'
            with np.load(INPUTS/f'{label}.npz',allow_pickle=False) as d:
                # O actually acts at q_mid, not at the end-of-step positions.
                q=d['q_mid'][-1].copy()
                q_before=d['q_unwrapped'][-2].copy()
                p=d['p'][-2].copy()+.0025*lj_force(np.mod(q_before,params['L']),params['L'])[0]
            from evidence import read_csv
            row=read_csv(RESULTS/f'{label}_steps.csv')[-1]
            state=load_json(RESULTS/f'{label}_rng.json')['states'][-1]['before']
            rng=np.random.default_rng();rng.bit_generator.state=state
            noise={'z':rng.standard_normal(p.shape)}
            if method=='A':
                noise['ef']=rng.standard_normal((int(row['edge_count']),3))
                noise['er']=rng.standard_normal((int(row['edge_count']),3))
            for name,field in [('z','z_sha256'),('ef','eta_forward_sha256'),('er','eta_reverse_sha256')]:
                if name in noise and array_sha(noise[name])!=row[field]:raise RuntimeError('Actual Gaussian replay mismatch')
            sp=validate_state(q,params,RESULTS/f'{label}_visited_spatial.json')
            if not sp['exploratory_budget_met']:raise RuntimeError('Visited state spatial failure')
            kr=validate_actions(q,p,params,ranks,RESULTS,tag=f'{label}_visited',splitting=False,noise=noise)
            records.append(dict(run=label,spatial=sp,rank_validation=kr))
            samples.append(dict(run=label,step=200,method=method,initial_index=i,
                q_mid_sha256=row['q_mid_sha256'],
                conservative_spatial_relative_error_bound=sp['conservative_relative_error_bound'],
                strict_spatial_budget_met=sp['strict_budget_met'],
                max_Krylov_relative_error=kr['max_relative_action_error'],
                actual_O_input_and_noise_replayed=True))
            print('Validated visited state',label,'spatial',sp['conservative_relative_error_bound'],flush=True)
    write_csv(RESULTS/'spatial_visited_samples.csv',samples)
    save(RESULTS/'visited_done.json',dict(sampled_states=6,at_actual_q_mid=True,records=records))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['initialize','stage0','trajectories','timing','visited'])
    args=parser.parse_args();globals()[args.action]()
