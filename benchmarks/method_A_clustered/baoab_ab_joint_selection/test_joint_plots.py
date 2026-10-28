"""Small evidence-transform tests; no plotting, FFT, or direct reference run."""
import unittest
from joint_plots import candidate_rows,comparison_rows,phase_rows,aggregate_ideal


class PlotEvidenceTests(unittest.TestCase):
    def ranking(self,cid,method='A',eps=1e-6,cost=10.,space=1e-8,frozen=1e-7):
        return dict(config_id=cid,spatial_id=cid,method=method,epsilon_frozen=eps,
            measured_wall_per_simulated_time=cost,k=2,r_d=3,r_n=8,actual_steps=9,
            params=dict(xi=.65,sR=4.5,sF=4.1,M=32,p=6),
            certificate=dict(max_spatial_error=space,max_ideal_physical_mean=frozen/2,max_ideal_physical_cov=frozen))
    def test_pareto_requires_three_objectives_and_same_method_tier(self):
        rr=[self.ranking('base'),self.ranking('dominated',cost=11),
            self.ranking('tradeoff',cost=12,space=1e-9),
            self.ranking('other_method',method='B',cost=1,space=1e-10,frozen=1e-10),
            self.ranking('other_tier',eps=1e-4,cost=1,space=1e-10,frozen=1e-10)]
        rows=candidate_rows(dict(exploration=dict(rankings=rr),pareto=[dict(config_id=r['config_id']) for r in rr]))
        front={r['config_id'] for r in rows if r['empirical_pareto']}
        self.assertEqual(front,{'base','tradeoff','other_method','other_tier'})
    def test_missing_pareto_registry_cannot_silently_add_unmeasured_point(self):
        with self.assertRaises(ValueError):candidate_rows(dict(exploration=dict(rankings=[self.ranking('x')]),pareto=[]))
    def test_phase_costs_add_disjoint_outer_clocks_including_rng_once(self):
        b=dict(heads=dict(groups=[dict(group_id='g',comparison_type='independent_optimum',epsilon_frozen=1e-6)]),
            head_steps=[dict(group_id='g',method=m,t_baoab_s=.010,t_dynamic_setup_s=.003,
                t_ou_loop_s=.005,t_LJ_forces_s=.001,t_rng_s=.0002,t_fft_s=.002) for m in ('A','B')])
        for row in phase_rows(b):
            self.assertAlmostEqual(row['integration_other_ms'],1)
            self.assertAlmostEqual(sum(row[k] for k in ('setup_ms','OU_including_RNG_ms','LJ_ms','integration_other_ms')),row['total_ms'])
    def test_full_norm_figure_aggregates_worst_state_and_only_executed_k(self):
        rows=[]
        keys=('physical_mean_bound','physical_cov_bound','own_mean_relative','own_cov_relative',
              'spatial_transition_mean_relative','spatial_transition_cov_relative')
        for k,state,value in [(1,'a',1e-5),(1,'b',2e-5),(2,'a',3e-6),(2,'b',4e-6)]:
            rows.append(dict(setting_id='s',method='A',k=k,state=state,**{name:value for name in keys}))
        out=aggregate_ideal(dict(ideal_rows=rows))
        self.assertEqual([r['k'] for r in out],[1,2]);self.assertEqual([r['states'] for r in out],[2,2])
        self.assertEqual([r['physical_cov_bound'] for r in out],[2e-5,4e-6])


if __name__=='__main__':unittest.main()
