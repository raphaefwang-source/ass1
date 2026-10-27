"""Independent structural / normalization tests for the new grid projection."""
import unittest

from grid_psd import ProjectedMesh, GridPSDGamma, project_table, COMPONENT_INDEX
from grid_psd import np, PAIRS6, Mesh, mesh_for, _inverse
import ewald_exp_lib as E
from threadpoolctl import threadpool_limits


class GridPSDTests(unittest.TestCase):
    def setUp(self):
        self.L = 5.
        self.K = E.kernel(2., 1.3, 0.)
        self.q = np.random.default_rng(431).uniform(0, self.L, (4, 3))

    def make(self, M=8, p=5, project=True):
        kc = 2*np.pi/self.L*(M//2-1)
        return ProjectedMesh.from_parameters(self.L,self.K,kc,M,p,project)

    def test_exact_block_clipping_no_negative_threshold(self):
        table = np.zeros((6,3,3,3))
        table[0] = -1.
        table[1] = 2.
        table[2] = -1e-25
        after, delta, info = project_table(table)
        np.testing.assert_array_equal(after[0], 0.)
        np.testing.assert_array_equal(after[1], 2.)
        np.testing.assert_array_equal(after[2], 0.)
        np.testing.assert_array_equal(delta[2], 1e-25)
        self.assertEqual(info['negative_eigenvalues'], 54)
        self.assertEqual(info['negative_blocks'], 27)
        np.testing.assert_array_equal(after,_inverse(after))

    def test_rotated_spectrum_and_nearest_projection(self):
        rng = np.random.default_rng(93)
        U, _ = np.linalg.qr(rng.normal(size=(3,3)))
        A = U @ np.diag([-3.,.5,2.]) @ U.T
        table = np.array([np.full((3,3,3),A[a,b]) for a,b in PAIRS6])
        after, delta, _ = project_table(table)
        target = U @ np.diag([0.,.5,2.]) @ U.T
        np.testing.assert_allclose(after[:,0,0,0][COMPONENT_INDEX],target,atol=2e-15)
        self.assertGreaterEqual(np.linalg.eigvalsh(delta[:,0,0,0][COMPONENT_INDEX]).min(),-1e-15)

    def test_production_audit_switch_preserves_bitwise_operator(self):
        for M in (7,8):
            audited = self.make(M,5)
            production = ProjectedMesh(audited.original,audit_eigenvalues=False)
            np.testing.assert_array_equal(production.table,audited.table)
            np.testing.assert_array_equal(production.delta_table,audited.delta_table)
            np.testing.assert_array_equal(production.Gr,audited.Gr)
            self.assertTrue(audited.diagnostics['diagnostic_eigenvalues_evaluated'])
            self.assertFalse(production.diagnostics['diagnostic_eigenvalues_evaluated'])
            self.assertIsNone(production.diagnostics['min_eig_after'])
            self.assertIsNone(production.diagnostics['min_eig_raw'])
            self.assertIsNone(production.diagnostics['correction_spectral_q99'])
            self.assertEqual(production.diagnostics['t_eigenvalue_diagnostics_s'],0.)
            self.assertEqual(production.diagnostics['negative_blocks'],audited.diagnostics['negative_blocks'])
            idx,wt = production.flat(self.q)
            v = np.random.default_rng(M).normal(size=(4,3))
            np.testing.assert_array_equal(production.apply(idx,wt,v),audited.apply(idx,wt,v))
        raw = audited.table_before
        ta,da,_ = project_table(raw)
        tp,dp,info = project_table(raw,audit_eigenvalues=False)
        np.testing.assert_array_equal(ta,tp)
        np.testing.assert_array_equal(da,dp)
        self.assertIsNone(info['min_eig_after'])

    def test_positive_stencils_wrap_and_half_ties(self):
        for M in (9,10):
            for p in range(1,10):
                m = self.make(M,p)
                x = np.r_[self.q,[[0,0,0],[5,5,5],[m.h/2]*3,[-5,10,0]]]
                _, w = m.stencil(x)
                self.assertGreaterEqual(float(w.min()),0.)
                np.testing.assert_allclose(w.sum(axis=2),1.,atol=5e-16)

    def test_original_normalization_control_odd_even(self):
        for M in (7,8):
            m = self.make(M,3,False)
            idx, wt = m.flat(self.q)
            oi, ow = m.original.flat(self.q)
            np.testing.assert_array_equal(idx,oi)
            np.testing.assert_allclose(wt,ow,atol=5e-16)
            v = np.random.default_rng(M).normal(size=(4,3))
            np.testing.assert_allclose(m.apply(idx,wt,v),m.original.apply(oi,ow,v),atol=2e-15,rtol=2e-13)
            actual = np.fft.irfftn(m.Gr,s=(M,)*3,axes=(1,2,3))/m.h**3
            np.testing.assert_allclose(actual,m.table,atol=5e-16)
            np.testing.assert_array_equal(m.Gr,m.original.Gr)

    def test_all_modes_including_odd_even_nyquist(self):
        for M in (7,8):
            m = self.make(M,5)
            self.assertGreater(m.diagnostics['negative_blocks'],0)
            self.assertGreater(m.diagnostics['new_fourier_modes_rfft'],0)
            W, imaginary = m.pair_matrix(self.q)
            self.assertEqual(imaginary,0.)
            np.testing.assert_allclose(W,m.direct_pair_matrix(self.q),atol=2e-15,rtol=2e-13)
            # Independent FULL complex DFT, with the even-grid Nyquist planes.
            idx, wt = m.flat(self.q)
            grid = np.stack(np.unravel_index(idx,(M,)*3),axis=-1)
            freq = np.rint(np.fft.fftfreq(M)*M).astype(int)
            mv = np.stack(np.meshgrid(freq,freq,freq,indexing='ij'),axis=-1).reshape(-1,3)
            S = np.sum(wt[:,:,None]*np.exp(-2j*np.pi*np.einsum('nsa,ka->nsk',grid,mv)/M),axis=1)
            G = np.fft.fftn(m.table,axes=(1,2,3)).reshape(6,-1)
            for c,(a,b) in enumerate(PAIRS6):
                z = (S.conj()*G[c]) @ S.T / M**3
                np.testing.assert_allclose(W[:,:,a,b],z.real,atol=2e-15,rtol=2e-13)
                self.assertLess(float(np.abs(z.imag).max()),2e-15)
            expected = m.original.Gr + np.fft.rfftn(m.delta_table,axes=(1,2,3)).real*m.h**3
            np.testing.assert_allclose(m.Gr,expected,atol=0.,rtol=0.)
            np.testing.assert_allclose(m.zero_mode,m.h**3*m.table.sum(axis=(1,2,3)),atol=2e-13)

    def test_dense_fft_actions_new_degree_and_psd(self):
        for M in (7,8):
            for p in (1,3,5):
                m = self.make(M,p)
                W,_ = m.pair_matrix(self.q)
                op = GridPSDGamma(self.q,self.L,self.K,2.,m)
                G = E.real_dense(op)+E.lap_full(W)
                np.testing.assert_allclose(op.Dm,W.sum(axis=1),atol=2e-15,rtol=2e-13)
                np.testing.assert_allclose(G,op.dense_by_basis(),atol=2e-15,rtol=2e-13)
                np.testing.assert_allclose(G,G.T,atol=2e-15)
                self.assertGreaterEqual(float(np.linalg.eigvalsh(G).min()),-2e-14)
                self.assertGreaterEqual(float(np.linalg.eigvalsh(W).min()),-2e-14)
                self.assertGreaterEqual(m.diagnostics['min_eig_after'],-2e-14)
                if m.diagnostics['negative_blocks']:
                    self.assertGreater(float(np.linalg.norm(op.Dm-m.original.degree(op.idx,op.wt))),1e-9)
                v = np.random.default_rng(M+p).normal(size=(4,3))
                np.testing.assert_allclose(op.apply(v),op.apply_staged(v),atol=2e-15,rtol=2e-13)
                for c in range(3):
                    trans = np.zeros((4,3)); trans[:,c] = 1.
                    np.testing.assert_allclose(op.apply(trans),0.,atol=2e-15)

    def test_self_cancellation_and_periodic_positions(self):
        m = self.make(8,5)
        op = GridPSDGamma(self.q[:1],self.L,self.K,2.,m)
        np.testing.assert_allclose(op.apply([[1.,2.,3.]]),0.,atol=2e-15)
        op = GridPSDGamma(self.q,self.L,self.K,2.,m)
        shifted = GridPSDGamma(self.q+[5,-10,15],self.L,self.K,2.,m)
        v = np.random.default_rng(91).normal(size=(4,3))
        np.testing.assert_allclose(op.apply(v),shifted.apply(v),atol=2e-15,rtol=2e-13)
        np.testing.assert_array_equal(m.table,_inverse(m.table))

    def test_zero_amplitude_and_illegal_inputs(self):
        zero = E.ThetaKernel(2.,1.3,0.,amp=0.)
        m = ProjectedMesh.from_parameters(self.L,zero,2.,8,3)
        op = GridPSDGamma(self.q,self.L,zero,2.,m)
        np.testing.assert_array_equal(op.apply(np.ones((4,3))),0.)
        self.assertEqual(m.diagnostics['active_correction_offsets'],0)
        for bad in (np.zeros((3,2,2,2)),np.full((6,2,2,2),np.nan)):
            with self.assertRaises(ValueError):project_table(bad)
        with self.assertRaises(TypeError):ProjectedMesh(object())
        m = self.make()
        for q,rc in ((np.zeros((2,3)),2.),(self.q,-1.),(np.full((4,3),np.nan),2.)):
            with self.assertRaises(ValueError):GridPSDGamma(q,self.L,self.K,rc,m)
        with self.assertRaises(ValueError):m.pair_matrix(self.q,0)


if __name__ == '__main__':
    with threadpool_limits(1):
        unittest.main()
