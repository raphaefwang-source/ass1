"""Driver tests: certification gates, measured objectives, controls and evidence."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import joint_driver as D


def config(method='A',sid='baseline',eps=1e-4,M=40):
    return dict(config_id=f'{sid}_{method}_e{eps:g}',spatial_id=sid,method=method,
        epsilon_frozen=eps,params=dict(D.BASE_PARAMS,M=M),k=4 if method=='A' and eps==1e-6 else 1,
        r_d=3 if method=='A' else 5,r_n=8 if method=='A' else 12,
        certificate=dict(spatial_passed=True,ideal_physical_passed=True,finite_passed=True))


class FakeSystem:
    """Synthetic IO fixture only; never produces an experimental timing result."""
    def __init__(self,params):
        self.params=params;self.L=params['L'];self.table_sha256='synthetic_table_'+str(params['M'])
        self.t_parameter_table_setup_s=.125;self.builds=0
    def step(self,q,p,method,r_d,r_n,rng,dt,k):
        self.builds+=1;noise=rng.standard_normal(p.shape);noise-=noise.mean(0)
        p1=p+.001*noise;q1=q+dt*p1
        self.last_built=type('Midpoint',(),{'q_mid':np.mod(q+.5*dt*p,self.L)})()
        self.last_substep_rows=[dict(substep=l+1,method=method,k=k) for l in range(k)]
        return q1,p1,dict(method=method,k=k,r_d=r_d,r_n=r_n,t_baoab_s=.025,
            t_dynamic_setup_s=.010,t_ou_wall_s=.008,momentum_norm=float(np.linalg.norm(p1.sum(0))),
            min_pair_distance=1.,operator_build_index=self.builds)


class JointDriverTests(unittest.TestCase):
    def test_certification_is_required_before_measuring(self):
        for name in ('spatial_passed','ideal_physical_passed','finite_passed'):
            c=config();c['certificate'][name]=False
            with self.assertRaises(ValueError):D.normalize_config(c)

    def test_wrong_physics_or_B_subcycling_cannot_enter_ranking(self):
        c=config();c['params']['kappa']=.2
        with self.assertRaises(ValueError):D.normalize_config(c)
        c=config('B');c['k']=4
        with self.assertRaises(ValueError):D.normalize_config(c)
        c=config();c['dt']=.0025
        with self.assertRaises(ValueError):D.normalize_config(c)
        c=config();c['k']=True
        with self.assertRaises(ValueError):D.normalize_config(c)

    def test_parameter_cache_shared_only_for_identical_spatial_settings(self):
        cache=D.SystemCache(FakeSystem)
        a=D.normalize_config(config());b=D.normalize_config(config('B'))
        self.assertIs(cache.get(a),cache.get(b))
        self.assertIsNot(cache.get(a),cache.get(D.normalize_config(config(sid='small',M=32))))
        self.assertEqual(len(cache.tables),2)

    def test_objective_uses_full_wall_per_simulated_time_not_cached_components(self):
        configs=[D.normalize_config(config(m,s,e)) for e in D.TIERS for m in ('A','B') for s in ('baseline','other')]
        runs=[]
        for c in configs:
            # Different winning spaces for A and B; expensive fake setup is irrelevant to score.
            cost=(1. if c['method']=='A' and c['spatial_id']=='other' else
                  2. if c['method']=='B' and c['spatial_id']=='baseline' else 4.)
            for i in range(3):
                runs.append(dict(config_id=c['config_id'],initial_index=i,compute_wall_s=cost,
                    simulated_time=.015,steps=3,dynamic_setup_mean_s=999.))
        ranks,winners=D.rank_exploratory(configs,runs)
        self.assertEqual(len(ranks),8)
        for w in winners:
            self.assertIn('other_A' if w['method']=='A' else 'baseline_B',w['config_id'])
            self.assertAlmostEqual(w['score'],(1. if w['method']=='A' else 2.)/.015)
        bad=runs[:-1]
        with self.assertRaises(RuntimeError):D.rank_exploratory(configs,bad)

    def test_identical_independent_and_controlled_groups_reuse_new_batch(self):
        configs=[D.normalize_config(config(m,eps=e)) for e in D.TIERS for m in ('A','B')]
        winners=[dict(epsilon_frozen=c['epsilon_frozen'],method=c['method'],config_id=c['config_id']) for c in configs]
        groups=D.make_groups(configs,dict(winners=winners),'baseline')
        self.assertEqual(len(groups),4)
        self.assertEqual(sum('reuse_group_id' in g for g in groups),2)
        self.assertTrue(all(g['reuse_group_id'].startswith('independent_optimum') for g in groups if 'reuse_group_id' in g))
        # Qualification tiers are distinct experiment groups even where one
        # method happens to have identical actual running parameters.
        loose=D.normalize_config(config('B',eps=1e-4));tight=D.normalize_config(config('B',eps=1e-6))
        self.assertNotEqual(D.configuration_key(loose),D.configuration_key(tight))

    def test_schedule_covers_three_starts_twice_and_balances_order(self):
        schedule=D.paired_schedule({},6,230)
        self.assertEqual([s['initial_index'] for s in schedule],[0,1,2,0,1,2])
        self.assertEqual(sum(s['order'][0]=='A' for s in schedule),3)
        self.assertTrue(all(set(s['order'])=={'A','B'} for s in schedule))
        self.assertEqual(schedule,D.paired_schedule({},6,230))

    def test_run_archives_actual_substeps_and_normalizes_physical_time(self):
        q=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        p=np.arange(12.).reshape(4,3)/100;p-=p.mean(0)
        with tempfile.TemporaryDirectory() as td:
            here=Path(td);(here/'inputs').mkdir();(here/'results').mkdir()
            patches=dict(HERE=here,INPUTS=here/'inputs',RESULTS=here/'results')
            with mock.patch.multiple(D,**patches),mock.patch.object(D,'source_initial',return_value=(q.copy(),p.copy(),dict(seed=3))):
                rows,sub,summary=D.run_candidate(config(eps=1e-6),0,3,912,'synthetic_run',cache=D.SystemCache(FakeSystem))
                self.assertEqual(len(rows),3);self.assertEqual(len(sub),12)
                self.assertEqual(summary['inner_calls'],12)
                self.assertAlmostEqual(summary['simulated_time'],.015)
                self.assertAlmostEqual(summary['wall_per_simulated_time'],5.)
                self.assertEqual(summary['k'],4)
                with np.load(here/'inputs/synthetic_run.npz',allow_pickle=False) as d:
                    self.assertEqual(d['q_unwrapped'].shape,(4,4,3))
                    self.assertEqual(d['q_mid'].shape,(3,4,3))
                    self.assertGreater(np.linalg.norm(d['q_unwrapped'][-1]-d['q_unwrapped'][0]),0)
                self.assertEqual(D.js(here/'results/synthetic_run_rng.json')['seed'],912)
                with self.assertRaises(FileExistsError):D.run_candidate(config(),0,3,912,'synthetic_run',cache=D.SystemCache(FakeSystem))

    def test_preserved_initial_data_is_hash_checked_and_copied(self):
        q,p,meta=D.source_initial(0)
        self.assertEqual(q.shape,(256,3));self.assertEqual(p.shape,q.shape)
        self.assertLess(np.linalg.norm(p.sum(0)),1e-10)
        self.assertEqual(D.array_sha(q),meta['q_sha256'])


if __name__=='__main__':unittest.main()
