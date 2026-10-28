"""Independent small-system checks of complete A substeps and moving BAOAB."""
import unittest
from unittest import mock
import numpy as np
from scipy import linalg
from matched_dynamics import MatchedSystem,ALLOWED_SUBSTEPS
from dynamics import PilotSystem
from model import lj_force,array_sha
from ou_core import ou_a
from spatial_validation import dense_parts


def literal_pair_sweep(p,edges,eta,tau,reverse=False):
    """Direct relative-momentum OU formula, independent of EdgeOU code."""
    out=np.asarray(p).reshape(-1,3).copy()
    order=range(len(edges.i)-1,-1,-1) if reverse else range(len(edges.i))
    for e in order:
        i,j=int(edges.i[e]),int(edges.j[e]);K=edges.K[e]
        lam,U=linalg.eigh(K)
        if lam.min()<-1e-12:raise ValueError('Indefinite synthetic edge')
        lam=np.maximum(lam,0.)
        r=out[i]-out[j]
        damp=linalg.expm(-2*tau*K)@r
        noise=(U*np.sqrt(2*-np.expm1(-4*tau*lam)))@U.T@eta[e]
        delta=.5*(damp+noise-r)
        out[i]+=delta;out[j]-=delta
    return out.reshape(np.asarray(p).shape)


def literal_frozen_a(p,built,draws,dt,k):
    _,GL,_=dense_parts(built.op)
    N=len(GL)//3
    Q=np.kron(linalg.helmert(N,full=False).T,np.eye(3))
    reduced=Q.T@GL@Q
    lam,V=linalg.eigh((reduced+reduced.T)/2)
    U=Q@V
    if lam.min()<-1e-11:raise ValueError('Indefinite long matrix')
    lam=np.maximum(lam,0.)
    h=dt/k
    R=np.eye(3*N)-Q@Q.T+Q@linalg.expm(-h*reduced)@Q.T
    F=(U*np.sqrt(-np.expm1(-2*h*lam)))@U.T
    current=np.asarray(p).copy()
    for d in draws:
        current=literal_pair_sweep(current,built.edges,d['ef'],h/2)
        current=(R@current.ravel()+F@d['z'].ravel()).reshape(current.shape)
        current=literal_pair_sweep(current,built.edges,d['er'],h/2,True)
    return current


class MatchedDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params=dict(N=4,L=8.,rho=.75,xi=.85,sR=4.1,sF=2.,M=12,p=3)
        cls.q=np.array([[.15,.2,.3],[1.4,.3,.35],[2.8,1.1,.6],[4.,3.,2.]])
        cls.p=np.random.default_rng(712).normal(size=(4,3));cls.p-=cls.p.mean(0)

    def system(self,**kwargs):return MatchedSystem(self.params,L=8.,**kwargs)

    def test_k1_bitwise_legacy_moving_A_and_B(self):
        for method in ('A','B'):
            old=PilotSystem(self.params,L=8.);new=self.system()
            qo,po,_=old.step(self.q,self.p,method,9,9,np.random.default_rng(902))
            qn,pn,row=new.step(self.q,self.p,method,9,9,np.random.default_rng(902),k=1)
            np.testing.assert_array_equal(qn,qo);np.testing.assert_array_equal(pn,po)
            self.assertEqual(row['ou_calls'],1)

    def test_k1_bitwise_reused_fixed_graph_ou(self):
        system=self.system(retain_noise=True);built=system.build(self.q,'A',k=1)
        actual,_=system.frozen_ou(built,self.p,'A',9,9,np.random.default_rng(44),k=1)
        d=system.last_noise_arrays[0]
        expected,_=ou_a(built.long_action,self.p,d['z'],d['ef'],d['er'],built.edges,.005,9,9)
        np.testing.assert_array_equal(actual,expected)

    def test_k_greater_one_literal_dense_path(self):
        # Multiple periodic images give well-conditioned full-rank edge blocks.
        # Independent eigensolvers need not agree to machine precision on the
        # square root of positive roundoff at a rank-one edge's null modes.
        params=dict(self.params,L=3.)
        q=np.array([[0,0,0],[0,1.5,1.5],[1.5,0,1.5],[1.5,1.5,0.]])
        for k in (2,4,8):
            system=MatchedSystem(params,L=3.,retain_noise=True);built=system.build(q,'A',k=k)
            self.assertGreater(np.linalg.eigvalsh(built.edges.K).min(),1e-5)
            actual,diag=system.frozen_ou(built,self.p,'A',9,9,np.random.default_rng(381+k),k=k)
            exact=literal_frozen_a(self.p,built,system.last_noise_arrays,.005,k)
            np.testing.assert_allclose(actual,exact,rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(actual.sum(0),self.p.sum(0),atol=2e-13)
            self.assertEqual(diag['ou_calls'],k)
            self.assertEqual(diag['long_ou_calls'],k)
            self.assertEqual(diag['full_ou_calls'],0)

    def test_all_registered_k_setup_once_and_coefficient_tau(self):
        system=self.system()
        for k in ALLOWED_SUBSTEPS:
            before=system.builds
            _,_,row=system.step(self.q,self.p,'A',9,9,np.random.default_rng(151+k),k=k)
            self.assertEqual(system.builds,before+1)
            self.assertEqual(row['graph_builds_this_step'],1)
            self.assertEqual(row['edge_eigh_builds_this_step'],1)
            self.assertEqual(row['coefficient_sets_this_step'],1)
            self.assertEqual(row['coefficient_prepare_calls'],2*k)
            self.assertEqual(row['coefficient_cache_hits'],2*k)
            self.assertEqual(row['coefficient_cache_size'],1)
            self.assertEqual(row['coefficient_tau'],.005/(2*k))
            self.assertEqual(set(system.last_built.edges._compiled_factors),{(.005/(2*k),1.,1.)})
            self.assertTrue(all(r['dt_inner']==.005/k for r in system.last_substep_rows))

    def test_independent_draw_indexing_replay_no_extra_scaling(self):
        k=4;system=self.system(retain_noise=True);built=system.build(self.q,'A',k=k)
        rng=np.random.default_rng(811)
        actual,_=system.frozen_ou(built,self.p,'A',9,9,rng,k=k)
        saved=system.last_noise_arrays;hashes=system.last_gaussian_hashes.copy()
        independent=np.random.default_rng(811)
        expected=[]
        for _ in range(k):
            expected.append(dict(z=independent.standard_normal(self.p.shape),
                ef=independent.standard_normal((len(built.edges.i),3)),
                er=independent.standard_normal((len(built.edges.i),3))))
        for i,d in enumerate(expected):
            for field in ('z','ef','er'):np.testing.assert_array_equal(saved[i][field],d[field])
        self.assertEqual(rng.bit_generator.state,independent.bit_generator.state)
        values=[v for row in hashes for key,v in row.items() if key.endswith('sha256')]
        self.assertEqual(len(values),len(set(values)))
        replay_rng=np.random.default_rng(913);before=replay_rng.bit_generator.state
        replay,_=system.frozen_ou(built,self.p,'A',9,9,replay_rng,k=k,noise_draws=expected)
        np.testing.assert_array_equal(replay,actual)
        self.assertEqual(before,replay_rng.bit_generator.state)

    def test_B_remains_one_full_step(self):
        system=self.system();before=system.builds
        _,_,row=system.step(self.q,self.p,'B',9,9,np.random.default_rng(53),k=1)
        self.assertEqual(row['full_ou_calls'],1);self.assertEqual(row['long_ou_calls'],0)
        self.assertEqual(system.builds,before+1)
        self.assertEqual(row['coefficient_sets_this_step'],0)
        with self.assertRaises(ValueError):system.step(self.q,self.p,'B',9,9,np.random.default_rng(53),k=4)

    def test_fresh_q_graph_and_same_A_B_Gamma(self):
        system=self.system()
        a=system.build(self.q,'A',k=4);b=system.build(self.q,'B')
        for field in ('i','j','d','w','Dm','DS'):
            np.testing.assert_array_equal(getattr(a.op,field),getattr(b.op,field))
        changed=self.q.copy();changed[0]+=[.02,.01,0.]
        other=system.build(changed,'A',k=4)
        self.assertNotEqual(a.token,other.token)
        self.assertIsNot(a.op,other.op);self.assertIsNot(a.edges,other.edges)
        self.assertIs(a.op.mesh,other.op.mesh)
        self.assertFalse(other.metrics['coefficient_first_prepare_hit'])

    def test_outer_BAOAB_midpoint_unchanged_by_subcycling(self):
        dt=.005;system=self.system()
        f0=lj_force(self.q,8.)[0];ph=self.p+.5*dt*f0
        expected_mid=self.q+.5*dt*ph
        q,p,row=system.step(self.q,self.p,'A',9,9,np.random.default_rng(97),k=4)
        np.testing.assert_array_equal(system.last_built.q_mid,np.mod(expected_mid,8.))
        po=p-.5*dt*lj_force(np.mod(q,8.),8.)[0]
        np.testing.assert_allclose(q,expected_mid+.5*dt*po,atol=1e-14)
        self.assertGreater(row['q_motion_rms'],0.)
        self.assertLess(row['momentum_norm'],1e-12)

    def test_complete_wall_and_nested_phase_accounting(self):
        system=self.system()
        _,_,row=system.step(self.q,self.p,'A',9,9,np.random.default_rng(17),k=4)
        major=sum(row[k] for k in ('t_dynamic_setup_s','t_LJ_forces_s','t_rng_s','t_ou_wall_s','t_baoab_other_s'))
        self.assertAlmostEqual(major,row['t_baoab_s'],places=14)
        self.assertGreaterEqual(row['t_baoab_other_s'],-1e-10)
        self.assertAlmostEqual(sum(r['t_ou_wall_s'] for r in system.last_substep_rows),row['t_ou_wall_s'])
        self.assertEqual(sum(r['matvec_count'] for r in system.last_substep_rows),row['matvec_count'])
        self.assertEqual(sum(r['fft_calls'] for r in system.last_substep_rows),row['ou_fft_calls'])
        self.assertEqual(len(system.last_substep_rows),4)
        self.assertEqual(row['negative_ritz_count'],0)

    def test_reject_unregistered_k_and_mismatched_setup(self):
        system=self.system()
        for k in (0,3,True,128):
            with self.assertRaises(ValueError):system.build(self.q,'A',k=k)
        built=system.build(self.q,'A',k=4)
        with self.assertRaises(ValueError):system.frozen_ou(built,self.p,'A',9,9,np.random.default_rng(1),k=2)


if __name__=='__main__':unittest.main(verbosity=2)
