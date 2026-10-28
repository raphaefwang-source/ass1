"""Stage 0 tests for NEW physics around the unchanged PR7 Grid PSD operator."""
import unittest
import numpy as np
import spatial_validation as S


class SpatialPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L=(8/.75)**(1/3)
        cls.q=np.array([[.19,.33,.11],[1.22,.28,.17],[.29,1.34,.25],[.14,.18,1.38],
                        [1.23,1.31,.18],[1.21,.18,1.29],[.22,1.24,1.22],[1.32,1.38,1.28]])%cls.L
        cls.params=dict(xi=.85,sR=4.1,sF=4.1,M=24,p=8,L=cls.L)
        cls.mesh=S.build_mesh(cls.params,cls.L)
        cls.op=S.build_operator(cls.q,cls.params,cls.mesh)
        cls.GS,cls.GL,cls.G=S.dense_parts(cls.op)

    def test_physical_normalization(self):
        self.assertEqual(self.op.K.kap,.7)
        self.assertEqual(self.op.K.amp,.5*1.3*np.exp(.7*1.3))
        self.assertAlmostEqual(float(self.op.K.theta_full(np.array([1.3]))[0]),.5,places=14)

    def test_full_fft_and_long_dense_actions(self):
        v=np.random.default_rng(1001).normal(size=24)
        np.testing.assert_allclose(self.op.matvec(v),self.G@v,rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(S.long_action(self.op,v),self.GL@v,rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(self.op.matvec(v),self.GS@v+S.long_action(self.op,v),rtol=2e-12,atol=2e-12)

    def test_symmetry_PSD_COM(self):
        for G in (self.GS,self.GL,self.G):
            np.testing.assert_allclose(G,G.T,rtol=0,atol=2e-12)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(G)[0]),-2e-12)
            for axis in range(3):
                v=np.zeros((8,3));v[:,axis]=1
                np.testing.assert_allclose(G@v.ravel(),0,atol=2e-12)

    def test_complete_image_tensors_no_self_edges(self):
        self.assertFalse(np.any(self.op.i==self.op.j))
        edges=S.near_edges(self.op)
        self.assertTrue(any(np.linalg.matrix_rank(e[2],tol=1e-12)==3 for e in edges))
        self.assertGreater(self.op.n_images,len(edges))
        for i,j,K in edges:
            m=(self.op.i==i)&(self.op.j==j)
            expected=np.einsum('n,na,nb->ab',self.op.w[m],self.op.d[m],self.op.d[m])
            np.testing.assert_allclose(K,expected,rtol=1e-13,atol=1e-13)

    def test_same_geometry_A_B_graph_equality(self):
        other=S.build_operator(self.q.copy(),self.params,self.mesh)
        np.testing.assert_array_equal(self.op.i,other.i)
        np.testing.assert_array_equal(self.op.j,other.j)
        np.testing.assert_array_equal(self.op.D,other.D)
        np.testing.assert_array_equal(self.op.idx,other.idx)
        np.testing.assert_array_equal(self.op.wt,other.wt)

    def test_changed_geometry_rebuild_and_parameter_table_reuse(self):
        q=self.q.copy();q[0]+=[.012,.008,-.006]
        other=S.build_operator(q,self.params,self.mesh)
        self.assertIs(other.mesh,self.op.mesh)
        self.assertIs(other.K,self.op.K)
        self.assertFalse(np.array_equal(other.wt,self.op.wt))
        self.assertFalse(np.array_equal(other.Dm,self.op.Dm))
        self.assertFalse(np.array_equal(other.d,self.op.d))
        np.testing.assert_array_equal(self.op.x,self.q)

    def test_physical_fourier_zero_mode_is_retained(self):
        A,B=self.op.K.AB_scaled(np.array([0.]))
        expected=np.array([A[0],A[0],A[0],0.,0.,0.])
        self.assertGreater(float(A[0]),0.)
        np.testing.assert_allclose(self.mesh.zero_mode_before,expected,rtol=1e-13,atol=1e-13)

    def test_mesh_self_cancellation(self):
        # Mesh W_ii may be nonzero but never becomes a friction self edge.
        one=S.build_operator(self.q[:1],dict(self.params,L=self.L),self.mesh)
        self.assertEqual(len(one.i),0)
        for v in (np.array([.1,.2,.3]),np.ones(3)):
            np.testing.assert_allclose(one.matvec(v),0,atol=2e-12)

    def test_periodicity_of_dense_operator_with_mesh_translation(self):
        shifted=S.build_operator(self.q+np.array([self.L,0,-self.L]),self.params,self.mesh)
        _,_,G=S.dense_parts(shifted)
        np.testing.assert_allclose(G,self.G,rtol=2e-12,atol=2e-12)

    def test_analytic_direct_tail_bound_decreases(self):
        bounds=[S.physical_tail_bound(256,(256/.75)**(1/3),r) for r in (30,35,40,45)]
        self.assertTrue(all(a>b>0 for a,b in zip(bounds,bounds[1:])))
        self.assertLess(bounds[-1],1e-9)

    def test_reference_literal_physics_and_no_self(self):
        q=self.q[:2];L=self.L
        G,meta=S.direct_reference(q,L,R=18.)
        i,j=np.triu_indices(2,1);d=q[0]-q[1]
        extent=int(np.ceil(18/L+.5));K=np.zeros((3,3))
        for a in range(-extent,extent+1):
            for b in range(-extent,extent+1):
                for c in range(-extent,extent+1):
                    rvec=d+L*np.array([a,b,c]);r=np.linalg.norm(rvec)
                    if 0<r<=18:
                        K+=S.AMPLITUDE*np.exp(-.7*r)/r**3*np.outer(rvec,rvec)
        expected=np.block([[K,-K],[-K,K]])
        np.testing.assert_allclose(G,expected,rtol=2e-13,atol=2e-13)
        self.assertEqual(meta['c'],S.AMPLITUDE)
        self.assertEqual(meta['kappa'],.7)
        self.assertTrue(meta['excludes_self_particle_edges'])


if __name__=='__main__':unittest.main()
