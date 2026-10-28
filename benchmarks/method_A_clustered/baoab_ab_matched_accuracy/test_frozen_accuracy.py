"""Independent small-system checks of full COM frozen operator references."""
import unittest
import numpy as np
from scipy import linalg
import frozen_accuracy as F
from ou_core import EdgeOU

def graph(edges,N):
    G=np.zeros((3*N,3*N))
    for i,j,K in edges:
        ii=slice(3*i,3*i+3);jj=slice(3*j,3*j+3)
        G[ii,ii]+=K;G[jj,jj]+=K;G[ii,jj]-=K;G[jj,ii]-=K
    return G

def exact_edge_block(i,j,K,tau,N):
    vals,U=linalg.eigh(K)
    D=.5*(U*np.expm1(-2*tau*vals))@U.T
    S=.5*(U*np.sqrt(2*-np.expm1(-4*tau*vals)))@U.T
    R=np.eye(3*N);B=np.zeros((3*N,3))
    ii=slice(3*i,3*i+3);jj=slice(3*j,3*j+3)
    R[ii,ii]+=D;R[jj,jj]+=D;R[ii,jj]-=D;R[jj,ii]-=D
    B[ii]=S;B[jj]=-S
    return R,B

class FrozenAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.N=3
        near=[(0,1,np.array([[1.4,.2,.1],[.2,.7,.0],[.1,.0,.5]])),
              (1,2,np.array([[.6,.1,.0],[.1,1.2,.2],[.0,.2,.9]]))]
        self.edges=EdgeOU(near,n_particles=3)
        long=[(0,2,np.array([[.8,.1,.0],[.1,.7,.1],[.0,.1,.5]])),
              (0,1,.3*np.eye(3)),(1,2,.2*np.eye(3))]
        self.GL=graph(long,3);self.GH=graph(near,3)+self.GL
        self.prepared=dict(edges=self.edges,GS=self.GH-self.GL,GL=self.GL,GH=self.GH,
            full_ref=F.DenseOU(self.GH),long_ref=F.DenseOU(self.GL))
        self.prepared['Q']=self.prepared['full_ref'].Q

    def literal_transition_and_covariance(self,k,dt):
        dim=9;R=np.eye(dim);C=np.zeros((dim,dim));t=dt/k
        rl,_,cl=self.prepared['long_ref'].matrices(t)
        for _ in range(k):
            for e in range(len(self.edges.i)):
                a,b=exact_edge_block(int(self.edges.i[e]),int(self.edges.j[e]),self.edges.K[e],t/2,3)
                R=a@R;C=a@C@a.T+b@b.T
            R=rl@R;C=rl@C@rl.T+cl
            for e in reversed(range(len(self.edges.i))):
                a,b=exact_edge_block(int(self.edges.i[e]),int(self.edges.j[e]),self.edges.K[e],t/2,3)
                R=a@R;C=a@C@a.T+b@b.T
        return R,C

    def test_ordered_mean_matches_literal_full_edge_matrices(self):
        for k in (1,2,4,8):
            actual=F.ideal_mean(self.prepared,k,.2)
            expected,_=self.literal_transition_and_covariance(k,.2)
            np.testing.assert_allclose(actual,expected,rtol=3e-14,atol=5e-15)

    def test_covariance_identity_matches_independent_Gaussian_columns(self):
        Pi=self.prepared['full_ref'].Pi
        for k in (1,2,4):
            R,C=self.literal_transition_and_covariance(k,.2)
            np.testing.assert_allclose(C,Pi-R@Pi@R.T,rtol=1e-13,atol=1e-14)
            self.assertGreaterEqual(float(linalg.eigvalsh(C).min()),-1e-14)
            self.assertLess(np.linalg.norm(C@np.tile(np.eye(3),(3,1))),3e-14)

    def test_normalizations_exclude_COM_unit_modes(self):
        rec,arrays=F.compare_maps(self.prepared,2,.2)
        full=self.prepared['full_ref'];Q=full.Q
        rb,_,cb=full.matrices(.2)
        self.assertLess(rec['mean_denominator'],1.)
        self.assertAlmostEqual(rec['mean_denominator'],np.linalg.norm(Q.T@rb@Q,2),places=13)
        self.assertAlmostEqual(rec['cov_denominator'],np.linalg.norm(Q.T@cb@Q,2),places=13)
        self.assertAlmostEqual(rec['e_mean'],np.linalg.norm(arrays['RA_reduced']-Q.T@rb@Q,2)/rec['mean_denominator'],places=12)
        self.assertEqual(rec['dimension_COM'],6)

    def test_expected_convergence_observed_not_assumed(self):
        errors=[F.compare_maps(self.prepared,k,.1)[0]['e_cov'] for k in (1,2,4)]
        self.assertTrue(errors[0]>errors[1]>errors[2]>0)
        self.assertLess(abs(np.log2(errors[1]/errors[0])+2),.03)
        self.assertLess(abs(np.log2(errors[2]/errors[1])+2),.03)

    def test_nonsymmetric_norm_uses_full_SVD(self):
        value,diag,_=F.exact_operator_norm(np.array([[0.,3.],[0.,0.]]))
        self.assertEqual(value,3.)
        self.assertEqual(diag['method'],'full svdvals')

    def test_positive_integer_substeps(self):
        for k in (0,-1,1.5,True):
            with self.assertRaises(ValueError):F.ideal_mean(self.prepared,k)

if __name__=='__main__':unittest.main()
