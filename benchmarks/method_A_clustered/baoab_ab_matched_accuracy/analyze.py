"""Paired-round bootstrap of actual dynamic wall costs; no old-batch ratios."""
from matched_common import *

def main():
    steps=read_csv(RESULTS/'timing_steps.csv');rounds=read_csv(RESULTS/'timing_rounds.csv')
    timing=js(RESULTS/'timing_done.json');selection=js(RESULTS/'accuracy/frozen_selection.json')
    ranks=js(RESULTS/'stochastic/stochastic_selection.json')
    endpoint=js(RESULTS/'post_timing_validation.json')
    rng=np.random.default_rng(202610281021);targets=[];raw={}
    for target in sorted(selection['targets'],key=lambda t:t['epsilon_frozen']):
        eps=float(target['epsilon_frozen']);k=target['k']
        if k is None:
            targets.append(dict(epsilon_frozen=eps,k=None,verdict='accuracy_not_met'));continue
        block=[r for r in rounds if r['epsilon_frozen']==eps]
        if len(block)!=6:raise RuntimeError('Need six complete actual paired rounds')
        ratios=np.array([r['A_over_B'] for r in block]);indices=rng.integers(0,len(block),size=(20000,len(block)))
        bootstrap=np.exp(np.log(ratios[indices]).mean(axis=1));low,high=np.quantile(bootstrap,[.025,.975])
        ratio=float(np.exp(np.log(ratios).mean()));raw[f'eps{eps:g}']=bootstrap
        end=[r for r in endpoint['records'] if r['epsilon_frozen']==eps]
        accuracy_passed=bool(end) and all(r['passed'] for r in end)
        verdict='accuracy_not_met' if not accuracy_passed else ('A_faster' if high<1 else ('B_faster' if low>1 else 'INCONCLUSIVE'))
        record=dict(epsilon_frozen=eps,k=int(k),ratio=ratio,ratio_ci_low=float(low),ratio_ci_high=float(high),
            ratio_statistic='geometric mean of6 paired round mean A/B ratios',verdict=verdict,
            bootstrap_sample_unit='one whole paired20-step A/B round;6 blocks,3 starts eachappearingtwice',
            bootstrap_replicates=20000,CI_scope='approximate95% percentile paired-round bootstrap conditional on these local blocks; not physical-observable CI',
            ideal_max_e_mean=target['max_e_mean'],ideal_max_e_cov=target['max_e_cov'],
            post_max_e_mean=max(r['e_mean'] for r in end),post_max_e_cov=max(r['e_cov'] for r in end),
            sampled_endpoint_accuracy_passed=accuracy_passed,A_ranks=ranks['by_k'][str(k)]['A'],B_ranks=ranks['by_k'][str(k)]['B'])
        for method in ('A','B'):
            rr=[r for r in steps if r['method']==method and r['epsilon_frozen']==eps]
            record[method+'_mean_ms']=float(1000*np.mean([r['t_baoab_s'] for r in rr]))
            record[method+'_median_ms']=float(1000*np.median([r['t_baoab_s'] for r in rr]))
            record[method+'_steps']=len(rr)
            for field in ['t_dynamic_setup_s','t_LJ_forces_s','t_rng_s','t_ou_wall_s','t_pair_forward_s',
                    't_pair_reverse_s','t_lanczos_damp_s','t_lanczos_noise_s','t_fft_s','t_reorth_s',
                    't_edge_eigh_s','t_coefficient_setup_s','t_baoab_other_s']:
                record[method+'_'+field+'_mean_ms']=float(1000*np.mean([r[field] for r in rr]))
        record['pooled_mean_ratio']=record['A_mean_ms']/record['B_mean_ms']
        targets.append(record)
    np.savez_compressed(RESULTS/'paired_bootstrap_samples.npz',**raw)
    write_csv(RESULTS/'matched_summary.csv',targets)
    extra=read_csv(RESULTS/'cost_curve_extra_steps.csv');curve=[]
    for k in (1,2,4,8,16,32):
        rows=[r for r in steps+extra if r['method']=='A' and r['k']==k]
        curve.append(dict(k=k,actual_steps=len(rows),full_mean_ms=1000*np.mean([r['t_baoab_s'] for r in rows]),
            full_median_ms=1000*np.median([r['t_baoab_s'] for r in rows]),
            ranks_d=','.join(map(str,sorted({int(r['r_d']) for r in rows}))),
            ranks_n=','.join(map(str,sorted({int(r['r_n']) for r in rows}))),
            data_scope='matched target120Asteps' if k in {t['k'] for t in targets} else 'shortcostcurve9Asteps; one-statefinitevalidation'))
    write_csv(RESULTS/'cost_curve_summary.csv',curve)
    save(RESULTS/'analysis.json',dict(targets=targets,cost_curve=curve,
        actual_matched_moving_steps=len(steps),actual_extra_costcurve_steps=len(extra),
        max_COM_norm=max(r['momentum_norm'] for r in steps+extra),
        min_pair_distance=min(r['min_pair_distance'] for r in steps+extra),
        peak_rss_MiB=max(r['peak_rss_bytes'] for r in steps+extra)/1024**2,
        bootstrap_seed=202610281021,registered_domain_states=10,new_endpoint_samples=endpoint['sampled_new_midpoints'],
        scope='frozen-transition-accuracy-matched dynamic cost over finite registered/sampledstates; completeBAOABequalphysicalobservableaccuracyunfinished',
        old_PR10_timings_used_as_background_only=True))
    print('Actual results',[(r['epsilon_frozen'],r['k'],r['verdict'],r.get('ratio')) for r in targets])

if __name__=='__main__':main()
