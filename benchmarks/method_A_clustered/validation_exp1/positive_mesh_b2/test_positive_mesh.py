"""Structural regression tests; PSD follows from proof.md, not these tests."""
import unittest
import experiment as X
from experiment import np
from positive_mesh import PositiveMesh,PositiveGamma,positive_bspline

class PositiveMeshTests(unittest.TestCase):
    def setUp(self):
        self.L=3.
        self.K=X.E.kernel(.7,.85,0.)
        self.q=np.random.default_rng(143).uniform(0,3,(5,3))

    def test_spline_nonnegativity_partition(self):
        for p in range(1,10):
            m=PositiveMesh(3,self.K,10,p,sT=4)
            g,w=m.stencil(np.r_[self.q,[[0,0,0],[3,3,3],[m.h/2,m.h/2,m.h/2]]])
            self.assertGreaterEqual(w.min(),0)
            np.testing.assert_allclose(w.sum(2),1,atol=5e-16)

    def test_fft_dense_odd_even_and_wrap(self):
        for M in (5,6):
            for p in (1,2,3):
                m=PositiveMesh(3,self.K,M,p,sT=4)
                op=PositiveGamma(self.q,3.,self.K,4.,m)
                G,GL=X.dense(op)
                rng=np.random.default_rng(M+p)
                v=rng.normal(size=15)
                np.testing.assert_allclose(op.matvec(v),G@v,rtol=1e-12,atol=2e-13)
                np.testing.assert_allclose(op.apply_staged(v),op.apply(v),rtol=1e-13,atol=2e-13)
                op2=PositiveGamma(self.q+np.array([3,-6,9]),3.,self.K,4.,m)
                np.testing.assert_allclose(op2.matvec(v),op.matvec(v),rtol=1e-12,atol=2e-13)
                self.assertGreater(np.linalg.eigvalsh(G).min(),-1e-12)
                np.testing.assert_allclose(G,G.T,atol=1e-13)

    def test_table_psd_inversion_and_zero_mode(self):
        for M in (5,6):
            m=PositiveMesh(3,self.K,M,3,sT=4,quad_order=2)
            tab=np.moveaxis(m.table,0,-1)
            full=tab[...,np.array([[0,3,4],[3,1,5],[4,5,2]])]
            self.assertGreaterEqual(np.linalg.eigvalsh(full).min(),-1e-14)
            neg=(-np.arange(M))%M
            np.testing.assert_allclose(m.table,m.table[:,neg][:,:,neg][:,:,:,neg],atol=1e-15)
            np.testing.assert_allclose(m.zero_mode,m.table.sum((1,2,3)),rtol=2e-15,atol=1e-14)
            self.assertGreater(m.zero_mode[:3].min(),0)

    def test_self_response_cancels(self):
        m=PositiveMesh(3,self.K,6,3,sT=4)
        # Includes the full interpolated self, not only T(0).
        op=PositiveGamma(self.q[:1],3.,self.K,4.,m)
        np.testing.assert_allclose(op.apply([[1.,2.,3.]]),0,atol=5e-15)
        op=PositiveGamma(self.q,3.,self.K,4.,m)
        for a in range(3):
            v=np.zeros((5,3));v[:,a]=1
            np.testing.assert_allclose(op.apply(v),0,atol=1e-14)

    def test_full_basis_and_zero_amplitude(self):
        m=PositiveMesh(3,self.K,6,3,sT=4)
        op=PositiveGamma(self.q,3.,self.K,4.,m)
        G,_=X.dense(op)
        np.testing.assert_allclose(op.dense_by_basis(),G,atol=2e-13,rtol=1e-12)
        zero=X.E.ThetaKernel(.7,.85,0.,amp=0.)
        m0=PositiveMesh(3,zero,6,3,sT=4)
        op0=PositiveGamma(self.q,3.,zero,4.,m0)
        np.testing.assert_array_equal(op0.matvec(np.ones(15)),np.zeros(15))

    def test_radial_kernel_api(self):
        from radial_kernels import RadialEwald
        k=RadialEwald('A',kappa=.7,xi=.85,amp=1.)
        a=PositiveMesh(3,self.K,6,2,sT=4)
        b=PositiveMesh(3,k,6,2,sT=4)
        np.testing.assert_array_equal(a.table,b.table)
        v=np.random.default_rng(14).normal(size=(5,3))
        pa=PositiveGamma(self.q,3.,self.K,4.,a)
        pb=PositiveGamma(self.q,3.,k,4.,b)
        np.testing.assert_allclose(pa.apply(v),pb.apply(v),atol=2e-13,rtol=1e-13)
        with self.assertRaises(ValueError):
            PositiveMesh(3,RadialEwald('B',.7,.85),6,2,sT=4)

    def test_independent_image_integral(self):
        m=PositiveMesh(3,self.K,5,3,sT=3.5)
        # Brute finite 3D images and the original independent s-integral.
        R=m.R
        for ix in ((0,0,0),(1,2,0),(4,3,2)):
            t=np.array(ix)*m.h
            t-=3*np.round(t/3)
            n=np.arange(-3,4)
            r=t+3*np.stack(np.meshgrid(n,n,n,indexing='ij'),-1).reshape(-1,3)
            r=r[np.all(np.abs(r)<=R,axis=1)]
            rr=np.linalg.norm(r,axis=1)
            f=np.zeros_like(rr)
            keep=rr>0
            f[keep]=self.K.theta_L(rr[keep])/rr[keep]**2
            direct=np.einsum('n,na,nb->ab',f,r,r)
            stored=m.table[(slice(None),)+ix][np.array([[0,3,4],[3,1,5],[4,5,2]])]
            np.testing.assert_allclose(direct,stored,atol=1e-12,rtol=2e-12)

    def test_bad_inputs(self):
        for M,p,sT in ((1,1,4),(5,6,4),(5,0,4),(5,2,-1),(5,2,np.nan)):
            with self.assertRaises(ValueError):PositiveMesh(3,self.K,M,p,sT)
        m=PositiveMesh(3,self.K,5,2,4)
        with self.assertRaises(ValueError):PositiveGamma(np.zeros((2,3)),3.,self.K,4.,m)
        with self.assertRaises(ValueError):PositiveGamma(self.q,3.,self.K,-1,m)

if __name__=='__main__':
    with X.threadpool_limits(1):unittest.main()
