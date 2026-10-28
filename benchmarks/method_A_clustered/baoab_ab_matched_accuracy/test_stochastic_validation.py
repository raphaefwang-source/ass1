"""Small independent distribution/path checks for the new stochastic audit."""
import unittest
import numpy as np
from scipy import linalg
import stochastic_validation as S
from compiled_edge import CompiledEdgeOU
from protocol import DenseOU

def graph(edges,N):
    out=np.zeros((N,3,N,3))
    for i,j,K in edges:
        out[i,:,i,:]+=K;out[j,:,j,:]+=K;out[i,:,j,:]-=K;out[j,:,i,:]-=K
    return out.reshape(3*N,3*N)

def pair_reference(edges,p,eta,tau,reverse=False):
    p=np.asarray(p).reshape(-1,3).copy();order=list(range(len(edges)))
    if reverse:order.reverse()
    for e in order:
        i,j,K=edges[e];w,U=linalg.eigh(K)
        r=p[i]-p[j]
        rp=U@(np.exp(-2*tau*w)*(U.T@r)+np.sqrt(2*-np.expm1(-4*tau*w))*(U.T@eta[e]))
        increment=.5*(rp-r);p[i]+=increment;p[j]-=increment
    return p

class StochasticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=[(0,1,np.array([[2.,.2,0],[.2,.7,.1],[0,.1,.8]])),
                     (1,2,np.array([[.8,-.1,.1],[-.1,1.4,0],[.1,0,.5]]))]
        gl=graph([(0,2,np.diag([.6,.9,1.2])),(0,1,.2*np.eye(3)),(1,2,.1*np.eye(3))],3)
        gh=gl+graph(cls.records,3)
        cls.prepared=dict(edges=CompiledEdgeOU(cls.records,n_particles=3),GL=gl,GH=gh,
            full_ref=DenseOU(gh),long_ref=DenseOU(gl),actions={'A':lambda v:gl@v,'B':lambda v:gh@v})
        cls.p=np.random.default_rng(93).normal(size=(3,3))

    def test_substep_budgets_and_schema(self):
        self.assertEqual(S.budget_for(1e-4,1),2e-8)
        self.assertEqual(S.budget_for(2e-8,16),2e-8/320)
        records=S.target_records(dict(targets=[dict(epsilon_frozen=1e-4,k=1,status='passed'),dict(epsilon_frozen=2e-8,k=16,status='passed')]))
        self.assertEqual([r['k'] for r in records],[1,16])

    def test_full_dimension_paths_match_each_own_ideal(self):
        for method,k in [('B',1),('A',1),('A',2),('A',4)]:
            result,raw=S.path_check(self.prepared,self.p,method,k,6,6,seed=832,action_budget=2e-10)
            self.assertLess(result['coupled_relative_error'],2e-14)
            self.assertLess(result['max_actual_input_action_error'],2e-14)
            self.assertLess(result['COM_error'],1e-14)
            self.assertEqual(len(result['substeps']),k)
            self.assertTrue(all(r['passed'] for r in result['action_checks']))

    def test_multistep_noise_order_matches_literal_pair_dense_reference(self):
        seed=524;k=4;rng=np.random.default_rng(seed);expected=self.p.copy()
        R,F,_=self.prepared['long_ref'].matrices(S.DT/k)
        for _ in range(k):
            z=rng.standard_normal(self.p.shape);ef=rng.standard_normal((2,3));er=rng.standard_normal((2,3))
            expected=pair_reference(self.records,expected,ef,S.DT/(2*k))
            expected=(R@expected.ravel()+F@z.ravel()).reshape(self.p.shape)
            expected=pair_reference(self.records,expected,er,S.DT/(2*k),True)
        result,raw=S.path_check(self.prepared,self.p,'A',k,6,6,seed)
        np.testing.assert_allclose(raw['actual'],expected,atol=2e-14,rtol=2e-14)
        hashes=[r[key] for r in result['substeps'] for key in ('z_sha256','ef_sha256','er_sha256')]
        self.assertEqual(len(hashes),len(set(hashes)))
        self.assertEqual(result['rng_state_after'],rng.bit_generator.state)

    def test_B_rejects_artificial_substeps(self):
        with self.assertRaises(ValueError):S.path_check(self.prepared,self.p,'B',2,6,6,1)

    def test_mc_statistics_are_sample_based_and_paired(self):
        values=np.array([-2.,-1.,0.,1.,2.])
        actual=S.directional_statistics(values,values,0.,2.)
        self.assertEqual(actual['variance'],2.5)
        self.assertEqual(actual['mean'],0.)
        self.assertEqual(actual['paired_mean_error'],0.)
        self.assertEqual(actual['paired_mean_se'],0.)
        self.assertEqual(actual['paired_second_error'],0.)
        self.assertEqual(actual['paired_second_se'],0.)
        self.assertLess(actual['variance_ci_low'],2.5)
        self.assertGreater(actual['variance_ci_high'],2.5)
        with self.assertRaises(ValueError):S.directional_statistics(values[:3],values[:3],0.,1.)

if __name__=='__main__':unittest.main()
