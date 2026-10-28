"""Actual dynamic BAOAB candidate screening and paired joint-selection timing.

The unchanged PR11 MatchedSystem performs all physical steps. This driver
only selects certified configurations, reuses legal parameter-only tables,
records actual moving trajectories, and keeps exploratory versus head batches
distinct. No dense/reference validation is hidden inside a timed BAOAB step.
"""
from joint_common import *
from matched_dynamics import MatchedSystem
import argparse
import copy
import re
import resource
import time
import psutil

SEED=202610282101
ALLOWED_K=(1,2,4,8,16,32)
PHYSICS_KEYS=('N','rho','L','kappa','c')
SPATIAL_KEYS=('xi','sR','sF','M','p')


def safe_label(value):
    value=str(value)
    if not value or not re.fullmatch(r'[A-Za-z0-9_.-]+',value):
        raise ValueError('Use a safe alphanumeric configuration/run identifier')
    return value


def normalize_config(raw):
    """Reject uncertified or physically incomparable settings before timing."""
    config=copy.deepcopy(raw)
    method=config.get('method')
    if method not in ('A','B'):raise ValueError('Configuration method must be A or B')
    eps=float(config['epsilon_frozen'])
    if eps not in TIERS:raise ValueError('Only the two registered frozen-accuracy tiers are enabled')
    params=dict(BASE_PARAMS,**config['params'])
    for key in PHYSICS_KEYS:
        if not np.isclose(params[key],BASE_PARAMS[key],rtol=0,atol=1e-14):
            raise ValueError('Joint selection cannot change physical '+key)
    for key in SPATIAL_KEYS:
        if not np.isfinite(params[key]) or params[key]<=0:raise ValueError('Invalid spatial '+key)
    for key in ('M','p'):
        if int(params[key])!=params[key]:raise ValueError('M and p must be integers')
        params[key]=int(params[key])
    if isinstance(config.get('k',1),bool):raise ValueError('Substeps must be an integer, not boolean')
    k=int(config.get('k',1))
    if k!=config.get('k',1) or k not in ALLOWED_K or (method=='B' and k!=1):
        raise ValueError('A uses registered complete substeps; B stays one full step')
    ranks={}
    for key in ('r_d','r_n'):
        value=config.get(key,config.get('ranks',{}).get(key))
        if isinstance(value,bool) or value is None or int(value)!=value or value<1:
            raise ValueError('Actual independently validated ranks are required')
        ranks[key]=int(value)
    cert=config.get('certificate',{})
    for key in ('spatial_passed','ideal_physical_passed','finite_passed'):
        if cert.get(key) is not True:raise ValueError('Missing successful certificate '+key)
    if float(config.get('dt',DT))!=DT:raise ValueError('This pilot fixes outer dt=.005')
    sid=safe_label(config['spatial_id'])
    cid=safe_label(config.get('config_id',f'{sid}_{method}_e{eps:g}_k{k}'))
    config.update(config_id=cid,spatial_id=sid,method=method,epsilon_frozen=eps,
                  params=params,k=k,dt=DT,**ranks)
    return config


def configuration_key(config):
    """Identity of actual running physics/ranks, independent of display labels."""
    c=normalize_config(config)
    data={k:c[k] for k in ('method','epsilon_frozen','params','k','r_d','r_n','dt')}
    return json.dumps(data,sort_keys=True,separators=(',',':'))


def source_initial(index):
    if int(index)!=index or index not in (0,1,2):raise ValueError('Require one of three preserved starts')
    path=PILOT/'inputs'/f'initial_{index}.npz'
    record=js(PILOT/'results'/f'initial_{index}.json')
    if sha(path)!=record['file_sha256']:raise RuntimeError('Preserved canonical input hash changed')
    with np.load(path,allow_pickle=False) as d:
        q,p=d['q_unwrapped'].copy(),d['p'].copy()
    if q.shape!=(256,3) or p.shape!=q.shape:raise RuntimeError('Unexpected N256 initial dimensions')
    return q,p,dict(file=str(path.relative_to(REPO)),file_sha256=sha(path),
                    seed=record['seed'],q_sha256=array_sha(q),p_sha256=array_sha(p))


class SystemCache:
    """Cache only parameter-only FFT/kernel tables via unchanged system objects."""
    def __init__(self,factory=MatchedSystem):
        self.factory=factory;self.systems={};self.tables=[]
    def get(self,config):
        params=config['params']
        key=json.dumps(params,sort_keys=True,separators=(',',':'))
        if key not in self.systems:
            system=self.factory(params)
            self.systems[key]=system
            self.tables.append(dict(table_id=len(self.tables)+1,params=params,
                parameter_table_sha256=system.table_sha256,
                actual_table_setup_s=system.t_parameter_table_setup_s,
                scope='one-time parameter-only construction, excluded from moving step wall'))
        return self.systems[key]


def load_configs(path=None):
    path=Path(path or RESULTS/'thermo/rank_selection.json')
    data=js(path)
    if data.get('all_passed',data.get('all_checks_passed',True)) is not True:
        raise RuntimeError('Joint rank selection is not complete and successful')
    raw=data.get('configs',[])
    if not raw:raise RuntimeError('No complete certified configuration records')
    configs=[]
    for item in raw:
        c=normalize_config(item)
        c['configuration_certificate_file']=str(path.relative_to(HERE)) if path.is_relative_to(HERE) else str(path)
        c['configuration_certificate_file_sha256']=sha(path)
        configs.append(c)
    if len({c['config_id'] for c in configs})!=len(configs):raise ValueError('Duplicate config identifiers')
    return configs


def run_candidate(config,initial_index,steps,seed,label,*,cache=None,batch='exploratory',
                  round_index=None,order_position=None,group_id=None):
    config=normalize_config(config);label=safe_label(label)
    if steps not in (3,4,5,20,40):raise ValueError('Only registered short moving pilots are enabled')
    if (RESULTS/f'{label}.json').exists():raise FileExistsError('Preserve completed run '+label)
    cache=cache if cache is not None else SystemCache()
    system=cache.get(config);q,p,initial_meta=source_initial(initial_index)
    qh=[q.copy()];ph=[p.copy()];mid=[];rows=[];sub=[];states=[]
    rng=np.random.default_rng(int(seed));load_before=os.getloadavg();started=time.perf_counter()
    for j in range(steps):
        before=copy.deepcopy(rng.bit_generator.state)
        q,p,row=system.step(q,p,config['method'],config['r_d'],config['r_n'],rng,
                            dt=DT,k=config['k'])
        row.update(run=label,batch=batch,step=j+1,time=(j+1)*DT,
            config_id=config['config_id'],spatial_id=config['spatial_id'],
            epsilon_frozen=config['epsilon_frozen'],initial_index=initial_index,
            seed=int(seed),round=round_index,order_position=order_position,group_id=group_id,
            rss_bytes=psutil.Process().memory_info().rss,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            wall_per_simulated_time=row['t_baoab_s']/DT,
            force_finite_checked_in_LJ_kernel=True,
            **{key:config['params'][key] for key in SPATIAL_KEYS})
        rows.append(row)
        sub.extend([dict(r,run=label,batch=batch,outer_step=j+1,
            config_id=config['config_id'],spatial_id=config['spatial_id'],
            epsilon_frozen=config['epsilon_frozen'],initial_index=initial_index,
            round=round_index,group_id=group_id) for r in system.last_substep_rows])
        states.append(dict(step=j+1,before=before,after=copy.deepcopy(rng.bit_generator.state)))
        qh.append(q.copy());ph.append(p.copy());mid.append(system.last_built.q_mid.copy())
    loop_wall=time.perf_counter()-started
    file=INPUTS/f'{label}.npz'
    np.savez_compressed(file,q_unwrapped=np.asarray(qh),q=np.mod(qh,system.L),
                        p=np.asarray(ph),q_mid=np.asarray(mid),L=system.L)
    write_csv(RESULTS/f'{label}_steps.csv',rows);write_csv(RESULTS/f'{label}_substeps.csv',sub)
    save(RESULTS/f'{label}_rng.json',dict(seed=int(seed),bit_generator='PCG64',states=states,
        recipe='Every complete A inner step draws z,ef,er independently; B draws z once'))
    summary=dict(run=label,batch=batch,config=config,method=config['method'],
        spatial_id=config['spatial_id'],config_id=config['config_id'],params=config['params'],
        k=config['k'],r_d=config['r_d'],r_n=config['r_n'],epsilon_frozen=config['epsilon_frozen'],
        initial_index=initial_index,initial_source=initial_meta,steps=steps,dt=DT,
        simulated_time=steps*DT,seed=int(seed),round=round_index,order_position=order_position,group_id=group_id,
        file=str(file.relative_to(HERE)),file_sha256=sha(file),
        start_q_sha256=array_sha(qh[0]),start_p_sha256=array_sha(ph[0]),
        end_q_sha256=array_sha(q),end_p_sha256=array_sha(p),
        full_BAOAB_mean_s=float(np.mean([r['t_baoab_s'] for r in rows])),
        full_BAOAB_median_s=float(np.median([r['t_baoab_s'] for r in rows])),
        wall_per_simulated_time=float(sum(r['t_baoab_s'] for r in rows)/(steps*DT)),
        compute_wall_s=sum(r['t_baoab_s'] for r in rows),loop_wall_including_evidence_s=loop_wall,
        dynamic_setup_mean_s=float(np.mean([r['t_dynamic_setup_s'] for r in rows])),
        O_mean_s=float(np.mean([r['t_ou_wall_s'] for r in rows])),inner_calls=len(sub),
        min_pair_distance=min(r['min_pair_distance'] for r in rows),
        max_COM_norm=max(r['momentum_norm'] for r in rows),
        peak_rss_bytes=max(r['peak_rss_bytes'] for r in rows),
        table_sha256=system.table_sha256,load_before=load_before,load_after=os.getloadavg(),
        scope='actual short moving BAOAB, fresh midpoint graph per outer step, selected params/ranks/k active')
    save(RESULTS/f'{label}.json',summary)
    print(label,config['method'],config['spatial_id'],'eps',config['epsilon_frozen'],
          'k',config['k'],'steps',steps,'mean_ms',1000*summary['full_BAOAB_mean_s'],flush=True)
    return rows,sub,summary


def rank_exploratory(configs,runs):
    """Select independently by measured total wall/physical time, never matvec time."""
    rankings=[];winners=[]
    for eps in TIERS:
        for method in ('A','B'):
            ranked=[]
            for c in configs:
                if c['epsilon_frozen']!=eps or c['method']!=method:continue
                rr=[r for r in runs if r['config_id']==c['config_id']]
                if len(rr)!=3 or {r['initial_index'] for r in rr}!={0,1,2}:
                    raise RuntimeError('Each exploratory candidate needs all three real starts')
                score=sum(r['compute_wall_s'] for r in rr)/sum(r['simulated_time'] for r in rr)
                ranked.append(dict(config_id=c['config_id'],spatial_id=c['spatial_id'],method=method,
                    epsilon_frozen=eps,measured_wall_per_simulated_time=score,
                    measured_step_mean_s=score*DT,actual_steps=sum(r['steps'] for r in rr),
                    k=c['k'],r_d=c['r_d'],r_n=c['r_n'],params=c['params'],certificate=c['certificate']))
            ranked.sort(key=lambda r:(r['measured_wall_per_simulated_time'],r['config_id']))
            for place,r in enumerate(ranked,1):r['rank']=place;rankings.append(r)
            if ranked:winners.append(dict(epsilon_frozen=eps,method=method,config_id=ranked[0]['config_id'],
                score=ranked[0]['measured_wall_per_simulated_time'],selection='best measured among certified local candidates; not global optimum'))
    return rankings,winners


def exploratory(configs=None,steps=3):
    if (RESULTS/'exploratory_done.json').exists():raise FileExistsError('Preserve completed exploratory batch')
    configs=[normalize_config(c) for c in (configs if configs is not None else load_configs())]
    if steps not in (3,4,5):raise ValueError('Exploratory steps must be3/4/5')
    cache=SystemCache();allrows=[];allsub=[];runs=[]
    for i in range(3):
        # Interleave A/B within precision/setting; rotate direction across starts.
        order=sorted(configs,key=lambda c:(c['epsilon_frozen'],c['spatial_id'],c['method']))
        if i%2:order=list(reversed(order))
        for position,c in enumerate(order):
            label=f'explore_{c["config_id"]}_start{i}'
            rr,ss,meta=run_candidate(c,i,steps,SEED+i*10000+position,
                label,cache=cache,batch='exploratory',order_position=position)
            allrows.extend(rr);allsub.extend(ss);runs.append(meta)
            save(RESULTS/'exploratory_tables.json',cache.tables)
    rankings,winners=rank_exploratory(configs,runs)
    write_csv(RESULTS/'exploratory_steps.csv',allrows);write_csv(RESULTS/'exploratory_substeps.csv',allsub)
    write_csv(RESULTS/'exploratory_ranking.csv',rankings)
    result=dict(configs=configs,runs=runs,rankings=rankings,winners=winners,
        actual_moving_steps=len(allrows),actual_inner_calls=len(allsub),steps_per_start=steps,
        starts=[0,1,2],canonical_preparation_steps_run=0,
        selection_objective='measured complete BAOAB wall / simulated physical time at fixed dt=.005',
        scope='short exploratory candidate ranking, no global optimum or equal physical-observable accuracy claim')
    save(RESULTS/'exploratory_done.json',result);return result


def make_groups(configs,exploration,baseline_spatial_id):
    """Build independent optimum and common-baseline groups with honest dedup."""
    by_id={c['config_id']:normalize_config(c) for c in configs};groups=[];seen={}
    for eps in TIERS:
        pair={}
        for method in ('A','B'):
            win=next((w for w in exploration['winners'] if w['epsilon_frozen']==eps and w['method']==method),None)
            if win is None:raise RuntimeError('Both methods need a qualified measured winner per tier')
            pair[method]=by_id[win['config_id']]
        proposals=[('independent_optimum',pair)]
        baseline={m:next((c for c in by_id.values() if c['epsilon_frozen']==eps and c['method']==m
                  and c['spatial_id']==baseline_spatial_id),None) for m in ('A','B')}
        if all(baseline.values()):proposals.append(('controlled_common_baseline',baseline))
        else:raise RuntimeError('Qualified common baseline A/B configuration missing')
        for kind,pp in proposals:
            group_id=safe_label(f'{kind}_e{eps:g}')
            identity=(configuration_key(pp['A']),configuration_key(pp['B']))
            group=dict(group_id=group_id,comparison_type=kind,epsilon_frozen=eps,A=pp['A'],B=pp['B'])
            if identity in seen:
                group.update(reuse_group_id=seen[identity],reuse_reason='identical A/B running configurations in this new experiment; no duplicate timing')
            else:seen[identity]=group_id
            groups.append(group)
    return groups


def paired_schedule(group,rounds,seed):
    if rounds not in (3,6):raise ValueError('Head rounds are3 or6')
    rng=np.random.default_rng(int(seed));first=['A','B','A'] if rounds==3 else ['A']*3+['B']*3
    first=list(rng.permutation(first))
    return [dict(round=r,initial_index=r%3,order=[first[r],'B' if first[r]=='A' else 'A']) for r in range(rounds)]


def head_to_head(groups,rounds=6,steps=20):
    if (RESULTS/'heads_done.json').exists():raise FileExistsError('Preserve completed head-to-head batch')
    if steps not in (20,40):raise ValueError('Head steps must be20 or40')
    cache=SystemCache();rows=[];sub=[];runs=[];blocks=[];completed=set()
    for index,raw in enumerate(groups):
        group=copy.deepcopy(raw);gid=safe_label(group['group_id']);eps=float(group['epsilon_frozen'])
        group['A']=normalize_config(group['A']);group['B']=normalize_config(group['B'])
        if any(group[m]['method']!=m or group[m]['epsilon_frozen']!=eps for m in ('A','B')):
            raise ValueError('Group method labels and accuracy tiers must match actual configurations')
        if group.get('reuse_group_id'):
            if group['reuse_group_id'] not in completed:raise RuntimeError('Reuse must reference an earlier completed identical group')
            source=next(g for g in groups if g['group_id']==group['reuse_group_id'])
            if any(configuration_key(group[m])!=configuration_key(source[m]) for m in ('A','B')):
                raise RuntimeError('Cannot reuse timing from different configurations')
            completed.add(gid);continue
        schedule=paired_schedule(group,rounds,SEED+200000+index)
        for slot in schedule:
            pair={};r=slot['round'];i=slot['initial_index']
            for position,method in enumerate(slot['order']):
                c=group[method];label=f'head_g{index}_round{r}_{method}'
                rr,ss,meta=run_candidate(c,i,steps,SEED+300000+index*1000+r*10+(method=='B'),label,
                    cache=cache,batch='head_to_head',round_index=r,order_position=position,group_id=gid)
                rows.extend(rr);sub.extend(ss);runs.append(meta);pair[method]=meta
                save(RESULTS/'heads_tables.json',cache.tables)
            blocks.append(dict(group_id=gid,comparison_type=group['comparison_type'],epsilon_frozen=eps,
                round=r,initial_index=i,order=','.join(slot['order']),
                A_config_id=group['A']['config_id'],B_config_id=group['B']['config_id'],
                A_spatial_id=group['A']['spatial_id'],B_spatial_id=group['B']['spatial_id'],
                A_mean_s=pair['A']['full_BAOAB_mean_s'],B_mean_s=pair['B']['full_BAOAB_mean_s'],
                A_wall_per_simulated_time=pair['A']['wall_per_simulated_time'],
                B_wall_per_simulated_time=pair['B']['wall_per_simulated_time'],
                A_over_B=pair['A']['full_BAOAB_mean_s']/pair['B']['full_BAOAB_mean_s']))
        completed.add(gid)
    write_csv(RESULTS/'heads_steps.csv',rows);write_csv(RESULTS/'heads_substeps.csv',sub)
    write_csv(RESULTS/'heads_rounds.csv',blocks)
    result=dict(groups=groups,runs=runs,actual_moving_steps=len(rows),actual_inner_calls=len(sub),
        rounds_per_unique_group=rounds,steps_per_method_round=steps,seed=SEED,
        table_registry=cache.tables,
        scope='new same-machine paired dynamic batches; independent candidate optima versus shared-space controlled baseline; finite validation domain only')
    save(RESULTS/'heads_done.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['explore','heads','groups'])
    parser.add_argument('--configs',default=str(RESULTS/'thermo/rank_selection.json'))
    parser.add_argument('--groups',default=str(RESULTS/'head_groups.json'))
    parser.add_argument('--baseline-id',default='baseline');parser.add_argument('--steps',type=int)
    parser.add_argument('--rounds',type=int,default=6);args=parser.parse_args()
    if args.action=='explore':exploratory(load_configs(args.configs),steps=args.steps or 3)
    elif args.action=='groups':
        groups=make_groups(load_configs(args.configs),js(RESULTS/'exploratory_done.json'),args.baseline_id)
        save(Path(args.groups),dict(groups=groups));print('Saved',len(groups),'declared comparison groups')
    else:head_to_head(js(Path(args.groups))['groups'],rounds=args.rounds,steps=args.steps or 20)
