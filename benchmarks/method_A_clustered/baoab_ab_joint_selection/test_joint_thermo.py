"""Independent matrix checks for the joint physical-OU error contract."""
import unittest
import numpy as np
from scipy import linalg
from joint_thermo import map_errors,compare,norm2
from protocol import DenseOU
from ou_core import EdgeOU

class PhysicalMapTests(unittest.TestCase):
    def test_physical_reference_distinct_from_own_Gh(self):
        R=np.diag([.95,.9]);C=np.eye(2)-R@R.T
        Rp=np.diag([.951,.902]);Cp=np.eye(2)-Rp@Rp.T
        rec,_=map_errors(R,C,R,C,Rp,Cp)
        self.assertEqual(rec['own_mean_relative'],0.)
        self.assertEqual(rec['own_cov_relative'],0.)
        self.assertAlmostEqual(rec['physical_mean_relative'],np.linalg.norm(R-Rp,2)/np.linalg.norm(Rp,2),places=14)
        self.assertAlmostEqual(rec['physical_cov_relative'],np.linalg.norm(C-Cp,2)/np.linalg.norm(Cp,2),places=14)
        self.assertGreater(rec['physical_cov_bound'],0.)

    def test_common_invariant_covariance_is_not_conditional_agreement(self):
        a=np.diag([.8,.7]);b=np.diag([.9,.85]);I=np.eye(2)
        ca=I-a@a.T;cb=I-b@b.T
        np.testing.assert_allclose(a@a.T+ca,I)
        np.testing.assert_allclose(b@b.T+cb,I)
        rec,_=map_errors(a,ca,b,cb,b,cb)
        self.assertGreater(rec['physical_mean_bound'],.1)
        self.assertGreater(rec['physical_cov_bound'],.5)

    def test_Duhamel_tail_allowance_covers_actual_PSD_perturbation(self):
        G=np.array([[1.2,.2],[.2,2.]])
        delta=np.array([[.03,.01],[.01,.02]])
        dt=.005;R=linalg.expm(-dt*G);C=np.eye(2)-R@R.T
        Rt=linalg.expm(-dt*(G+delta));Ct=np.eye(2)-Rt@Rt.T
        # An arbitrary numerical approximation, then compare against unknown true G+delta.
        Ra=linalg.expm(-dt*(G+.5*delta));Ca=np.eye(2)-Ra@Ra.T
        rec,_=map_errors(Ra,Ca,Ra,Ca,R,C,np.linalg.norm(delta,2),dt)
        actual_mean=np.linalg.norm(Ra-Rt,2)/np.linalg.norm(Rt,2)
        actual_cov=np.linalg.norm(Ca-Ct,2)/np.linalg.norm(Ct,2)
        self.assertLessEqual(actual_mean,rec['physical_mean_bound'])
        self.assertLessEqual(actual_cov,rec['physical_cov_bound'])
        self.assertAlmostEqual(rec['physical_mean_tail_allowance'],dt*np.linalg.norm(delta,2))
        self.assertAlmostEqual(rec['physical_cov_tail_allowance'],2*dt*np.linalg.norm(delta,2))

    def test_unresolved_zero_covariance_denominator_is_explicit(self):
        I=np.eye(2);Z=np.zeros((2,2))
        rec,_=map_errors(I,Z,I,Z,I,Z)
        self.assertFalse(rec['reference_denominators_resolved'])
        self.assertEqual(rec['physical_cov_denominator_lower'],0.)
        self.assertEqual(rec['denominator_small'],1e-300)

    def test_full_operator_norm_handles_nonsymmetric_difference(self):
        A=np.array([[1.,3.],[-.4,.2]])
        self.assertAlmostEqual(norm2(A),linalg.svdvals(A)[0],places=14)

    def test_B_own_discrete_zero_but_physical_error_retained(self):
        N=3;K=np.array([[1.,.1,0],[.1,.8,.05],[0,.05,.6]])
        edges=EdgeOU([(0,1,K),(1,2,.6*K),(0,2,.3*K)],n_particles=N)
        G=np.zeros((9,9))
        for i,j,k in zip(edges.i,edges.j,edges.K):
            a=slice(3*i,3*i+3);b=slice(3*j,3*j+3)
            G[a,a]+=k;G[b,b]+=k;G[a,b]-=k;G[b,a]-=k
        Gp=1.00001*G
        full=DenseOU(G);physical=DenseOU(Gp)
        pp=dict(Q=full.Q,full_ref=full,long_ref=DenseOU(.4*G),edges=edges,GH=G,physical_G=Gp,
                physical_ref=physical,physical_meta=dict(tail_absolute=0.,norm2=np.linalg.norm(Gp,2)))
        rec,raw=compare(pp,'B',1)
        self.assertEqual(rec['dimension_COM'],6)
        self.assertEqual(rec['own_mean_relative'],0.)
        self.assertEqual(rec['own_cov_relative'],0.)
        self.assertGreater(rec['physical_mean_bound'],0.)
        self.assertGreater(rec['physical_cov_bound'],0.)
        self.assertFalse(rec['spatial_passed'])
        np.testing.assert_allclose(raw['CA_reduced'],np.eye(6)-raw['RA_reduced']@raw['RA_reduced'].T,atol=2e-15)

if __name__=='__main__':unittest.main()
