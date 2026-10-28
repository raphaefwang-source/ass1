"""Actual local joint-selection ranking, paired uncertainty and Pareto data."""
from joint_common import *

SEED=202610282199
REPLICATES=20000
PHASES=('t_dynamic_setup_s','t_LJ_forces_s','t_rng_s','t_ou_wall_s','t_baoab_other_s',
        't_edge_eigh_s','t_coefficient_setup_s','t_pair_forward_s','t_pair_reverse_s',
        't_lanczos_damp_s','t_lanczos_noise_s','t_fft_s','t_reorth_s')

def pareto_rows(exploration):
    rows=[]
    for r in exploration['rankings']:
        c=r['certificate'];row=dict(r)
        row.update(spatial_error=float(c['max_spatial_error']),
            frozen_error=max(float(c['max_ideal_physical_mean']),float(c['max_ideal_physical_cov'])),
            finite_path_error=float(c['max_finite_path_error']),
            actual_cost=float(r['measured_wall_per_simulated_time']))
        rows.append(row)
    for r in rows:
        peers=[x for x in rows if x['method']==r['method'] and x['epsilon_frozen']==r['epsilon_frozen'] and x['config_id']!=r['config_id']]
        dims=('actual_cost','spatial_error','frozen_error')
        dominating=[x['config_id'] for x in peers if all(x[k]<=r[k] for k in dims) and any(x[k]<r[k] for k in dims)]
        r.update(pareto=not dominating,dominated_by=','.join(dominating),
            pareto_scope='empirical three-objective frontier among tested exploratory settings;9steps/config;not statistically certified/global')
    return rows

def main():
    heads=js(RESULTS/'heads_done.json');post=js(RESULTS/'post_validation.json')
    explore=js(RESULTS/'exploratory_done.json');rounds=read_csv(RESULTS/'heads_rounds.csv');steps=read_csv(RESULTS/'heads_steps.csv')
    post_by={r['group_id']:r for r in post['groups']}
    comparisons=[];by_gid={};samples={}
    for group_index,group in enumerate(heads['groups']):
        gid=group['group_id'];source=group.get('reuse_group_id',gid)
        p=post_by[gid]
        if source!=gid:
            rec=dict(by_gid[source],group_id=gid,source_group_id=source,
                comparison_type=group['comparison_type'],reused=True)
            samples[gid]=samples[source].copy()
        else:
            rr=[r for r in rounds if r['group_id']==gid]
            if len(rr)!=heads['rounds_per_unique_group']:raise RuntimeError('Incomplete paired comparison')
            group_seed=SEED+group_index;rng=np.random.default_rng(group_seed)
            ratios=np.array([r['A_over_B'] for r in rr]);indices=rng.integers(0,len(rr),size=(REPLICATES,len(rr)))
            bs=np.exp(np.log(ratios[indices]).mean(axis=1));samples[gid]=bs
            lo,hi=np.quantile(bs,[.025,.975]);ratio=float(np.exp(np.log(ratios).mean()))
            rec=dict(group_id=gid,source_group_id=gid,comparison_type=group['comparison_type'],
                epsilon_frozen=group['epsilon_frozen'],reused=False,
                ratio=ratio,A_over_B=ratio,ratio_ci_low=float(lo),ratio_ci_high=float(hi),CI_low=float(lo),CI_high=float(hi),
                ratio_statistic='geometric mean of whole paired round mean A/B wall ratios',
                bootstrap_seed=group_seed,bootstrap_group_index=group_index,bootstrap_replicates=REPLICATES,paired_rounds=len(rr),
                bootstrap_unit='whole paired20-step A/B round;three preserved starts;native-load blocks',
                CI_scope='approximate95%percentile interval conditional on tested local blocks, not physical-observable confidence',
                A_config=group['A'],B_config=group['B'])
            for method in ('A','B'):
                ss=[s for s in steps if s['group_id']==gid and s['method']==method]
                rec[method+'_mean_ms']=float(1000*np.mean([s['t_baoab_s'] for s in ss]))
                rec[method+'_median_ms']=float(1000*np.median([s['t_baoab_s'] for s in ss]))
                rec[method+'_wall_per_simulated_time']=rec[method+'_mean_ms']/1000/DT
                rec[method+'_actual_steps']=len(ss)
                for field in PHASES:rec[method+'_'+field+'_mean_ms']=float(1000*np.mean([s[field] for s in ss]))
        rec.update(accuracy_passed=bool(p['passed']),post_certificate=p,
            verdict='accuracy_not_met' if not p['passed'] else ('A_faster' if rec['ratio_ci_high']<1 else ('B_faster' if rec['ratio_ci_low']>1 else 'INCONCLUSIVE')))
        comparisons.append(rec);by_gid[gid]=rec
    np.savez_compressed(RESULTS/'paired_bootstrap_samples.npz',**samples)
    scalar=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in comparisons]
    write_csv(RESULTS/'comparison_summary.csv',scalar)
    pareto=pareto_rows(explore)
    write_csv(RESULTS/'pareto.csv',[{k:v for k,v in r.items() if not isinstance(v,(list,dict))}|r['params'] for r in pareto])
    allsteps=steps+read_csv(RESULTS/'exploratory_steps.csv')
    save(RESULTS/'analysis.json',dict(comparisons=comparisons,pareto=pareto,
        actual_head_moving_steps=len(steps),actual_exploratory_moving_steps=len(allsteps)-len(steps),
        max_COM_norm=max(s['momentum_norm'] for s in allsteps),min_pair_distance=min(s['min_pair_distance'] for s in allsteps),
        peak_rss_MiB=max(s['peak_rss_bytes'] for s in allsteps)/1024**2,
        bootstrap_seed=SEED,bootstrap_seed_rule='base+declared group index; reused identical groups copy original samples',bootstrap_replicates=REPLICATES,required_tiers=list(TIERS),
        optional_2e8_stress_executed=False,canonical_preparation_steps_run=0,
        search_scope='best measured among four locally tested certified settings;not global optimum',
        scientific_scope='physical-Gamma frozen-accuracy-constrained movingBAOAB cost;full equal physical-observable accuracy unfinished',
        old_PR10_PR11_batches_used_in_new_ratios=False))
    print([(r['comparison_type'],r['epsilon_frozen'],r['verdict'],r['ratio']) for r in comparisons])

if __name__=='__main__':main()
