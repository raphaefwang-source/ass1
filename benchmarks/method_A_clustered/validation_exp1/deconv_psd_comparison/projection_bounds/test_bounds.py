"""Independent correction-only bound tests; no changes to the PPPM method."""
import unittest
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from bounds import CorrectionBounds
from grid_psd import ProjectedMesh, COMPONENT_INDEX, PAIRS6
from positive_mesh import PositiveMesh
import ewald_exp_lib as E
from threadpoolctl import threadpool_limits


class SyntheticMesh:
    """Known physical correction tables, independent of the projection code."""
    stencil = PositiveMesh.stencil
    flat = PositiveMesh.flat

    def __init__(self,table,delta,L=5.,p=1):
        self.M = table.shape[1]
        self.L,self.p,self.h = L,p,L/self.M
        self.projected = True
        self.delta_table = table
        self.diagnostics = dict(max_negative_part_spectral=delta,
            negative_blocks=int(np.any(table!=0,axis=0).sum()),
            fraction_negative_blocks=float(np.any(table!=0,axis=0).mean()),
            correction_fro=float(np.sqrt(np.sum(np.array([1,1,1,2,2,2])[:,None,None,None]*table**2))),
            min_eig_before=-delta,inversion_residual_before=0.)


def constant_table(M,A):
    return np.array([np.full((M,M,M),A[a,b]) for a,b in PAIRS6])


class CorrectionBoundsTests(unittest.TestCase):
    def setUp(self):
        self.L = 5.
        self.q = np.random.default_rng(921).uniform(0,self.L,(4,3))
        self.K = E.kernel(2.,1.3,0.)

    def projected(self,M=8,p=3):
        kc = 2*np.pi/self.L*(M//2-1)
        return ProjectedMesh.from_parameters(self.L,self.K,kc,M,p,audit_eigenvalues=False)

    def assert_bounds(self,cb,q):
        r,D = cb.bounds(q)
        W = cb.pair_matrix(q)
        G = E.lap_full(W)
        actual = float(np.linalg.norm(G,2))
        scale = max(actual,cb.delta,np.finfo(float).tiny)
        for key in ('B_simple','B_degree','B'):
            self.assertGreaterEqual(r[key]+2e-13*scale,actual,(key,r,actual))
        self.assertEqual(r['B'],min(r['B_simple'],r['B_degree']))
        np.testing.assert_allclose(D,W.sum(axis=1),atol=3e-14*scale,rtol=3e-13)
        return r,D,W,G

    def test_simple_bound_is_tight_for_every_N_at_least_two(self):
        direction = np.array([1.,2.,-3.]);direction /= np.linalg.norm(direction)
        delta = .73
        for N in (2,3,7):
            C = constant_table(5,delta*np.outer(direction,direction))
            cb = CorrectionBounds(SyntheticMesh(C,delta,p=3))
            q = np.random.default_rng(N).uniform(0,self.L,(N,3))
            r,D,W,G = self.assert_bounds(cb,q)
            actual = np.linalg.norm(G,2)
            np.testing.assert_allclose(actual,N*delta,rtol=3e-15,atol=3e-15)
            np.testing.assert_allclose(r['B'],actual,rtol=3e-15,atol=3e-15)
            np.testing.assert_allclose(r['B_degree'],2*N*delta,rtol=3e-15,atol=3e-15)

    def test_degree_bound_is_tight_and_improves_simple(self):
        M,delta = 7,.4
        C = np.zeros((6,M,M,M))
        C[:3,1,0,0] = delta
        C[:3,-1,0,0] = delta
        cb = CorrectionBounds(SyntheticMesh(C,delta,L=7.,p=1))
        q = np.array([[0.,0,0],[1.,0,0],[3.,0,0]])
        r,D,W,G = self.assert_bounds(cb,q)
        np.testing.assert_allclose(np.linalg.norm(G,2),2*delta,atol=2e-15)
        np.testing.assert_allclose(r['B_degree'],2*delta,atol=2e-15)
        self.assertLess(r['B_degree'],r['B_simple'])
        self.assertEqual(r['B'],r['B_degree'])

    def test_odd_even_bounds_fft_sparse_direct_and_basis(self):
        for M in (7,8):
            for p in (1,2,3):
                cb = CorrectionBounds(self.projected(M,p),audit=True)
                self.assertTrue(cb.diagnostics['active'])
                r,D,W,G = self.assert_bounds(cb,self.q)
                direct = cb.direct_pair_matrix(self.q)
                np.testing.assert_allclose(W,direct,atol=2e-16,rtol=3e-13)
                idx,wt = cb.flat(self.q)
                np.testing.assert_allclose(cb.degree(idx,wt),D,atol=0.,rtol=0.)
                for j in range(4):
                    for b in range(3):
                        v = np.zeros((4,3));v[j,b] = 1.
                        np.testing.assert_allclose(cb.apply(idx,wt,v),W[:,j,:,b],atol=2e-16,rtol=3e-13)
                        np.testing.assert_allclose(cb.matvec(self.q,v,D),G[:,3*j+b],atol=2e-16,rtol=3e-13)
                self.assertGreaterEqual(np.linalg.eigvalsh(W).min(),-2e-16)
                self.assertGreaterEqual(np.linalg.eigvalsh(G).min(),-2e-16)

    def test_full_modes_nyquist_and_physical_units(self):
        for M in (5,6):
            cb = CorrectionBounds(self.projected(M,3),audit=True)
            W = cb.pair_matrix(self.q)
            idx,wt = cb.flat(self.q)
            grid = np.stack(np.unravel_index(idx,(M,)*3),axis=-1)
            freq = np.rint(np.fft.fftfreq(M)*M).astype(int)
            modes = np.stack(np.meshgrid(freq,freq,freq,indexing='ij'),axis=-1).reshape(-1,3)
            S = np.sum(wt[:,:,None]*np.exp(-2j*np.pi*np.einsum('nsa,ka->nsk',grid,modes)/M),axis=1)
            G = np.fft.fftn(cb.C,axes=(1,2,3)).reshape(6,-1)
            for c,(a,b) in enumerate(PAIRS6):
                z = (S.conj()*G[c]) @ S.T / M**3
                np.testing.assert_allclose(z.real,W[:,:,a,b],atol=2e-16,rtol=2e-13)
                self.assertLess(float(np.abs(z.imag).max()),2e-16)
            np.testing.assert_allclose(cb.zero_mode,cb.C.sum(axis=(1,2,3)),atol=3e-15,rtol=3e-15)

    def test_singleton_self_cancels_and_three_translations(self):
        cb = CorrectionBounds(self.projected(),audit=True)
        r,D = cb.bounds(self.q[:1])
        self.assertTrue(r['singleton_exact_zero'])
        self.assertEqual(r['B_simple'],0.)
        self.assertEqual(r['B'],0.)
        np.testing.assert_allclose(cb.matvec(self.q[:1],[1.,2.,3.],D),0.,atol=2e-16)
        r,D = cb.bounds(self.q)
        for e in np.eye(3):
            np.testing.assert_allclose(cb.matvec(self.q,np.tile(e,(4,1)),D),0.,atol=2e-16)
        shifted = self.q+[5,-10,15]
        v = np.random.default_rng(2).normal(size=12)
        np.testing.assert_allclose(cb.matvec(self.q,v),cb.matvec(shifted,v),atol=2e-16,rtol=3e-13)

    def test_zero_correction_has_true_zero_bounds_and_response(self):
        zero = E.ThetaKernel(2.,1.3,0.,amp=0.)
        mesh = ProjectedMesh.from_parameters(self.L,zero,2.,8,3,audit_eigenvalues=False)
        cb = CorrectionBounds(mesh,audit=True)
        r,D,W,G = self.assert_bounds(cb,self.q)
        self.assertFalse(r['active'])
        self.assertEqual(r['delta'],0.)
        for k in ('B_simple','B_degree','B'):
            self.assertEqual(r[k],0.)
        np.testing.assert_array_equal(D,0.)
        np.testing.assert_array_equal(W,0.)
        np.testing.assert_array_equal(cb.matvec(self.q,np.ones(12)),0.)

    def test_tiny_correction_not_dropped_or_inflated(self):
        tiny = 1e-200
        C = constant_table(5,tiny*np.eye(3))
        cb = CorrectionBounds(SyntheticMesh(C,tiny,p=2))
        r,D = cb.bounds(self.q)
        self.assertTrue(r['active'])
        self.assertEqual(len(cb.active_offsets),125)
        self.assertEqual(r['delta'],tiny)
        self.assertEqual(r['B_simple'],4*tiny)
        self.assertGreater(float(cb.pair_matrix(self.q).max()),0.)

    def test_psd_inversion_roundtrip_and_mesh_unchanged(self):
        mesh = self.projected()
        snapshots = {k:(v.copy(),v.flags.writeable) for k,v in mesh.__dict__.items() if isinstance(v,np.ndarray)}
        keys_before = set(mesh.__dict__)
        cb = CorrectionBounds(mesh,audit=True)
        self.assertFalse(cb.C.flags.writeable)
        self.assertGreaterEqual(cb.diagnostics['correction_float_min_eigenvalue'],-2e-16)
        self.assertEqual(cb.diagnostics['correction_inversion_max_component_defect'],0.)
        self.assertLess(cb.diagnostics['correction_fft_roundtrip_max_component_defect'],2e-16)
        cb.bounds(self.q);cb.pair_matrix(self.q);cb.matvec(self.q,np.ones(12))
        self.assertEqual(set(mesh.__dict__),keys_before)
        for k,(original,writeable) in snapshots.items():
            np.testing.assert_array_equal(mesh.__dict__[k],original)
            self.assertEqual(mesh.__dict__[k].flags.writeable,writeable)

    def test_degree_bound_never_calls_pair_matrix(self):
        cb = CorrectionBounds(self.projected())
        def forbidden(*args,**kwargs):
            raise AssertionError('A degree bound must not form dense particle pair/graph matrices')
        cb.pair_matrix = forbidden
        cb.direct_pair_matrix = forbidden
        r,D = cb.bounds(self.q)
        self.assertEqual(D.shape,(4,3,3))
        self.assertGreater(r['B'],0.)

    def test_invalid_inputs(self):
        cb = CorrectionBounds(self.projected())
        for bad in (np.empty((0,3)),np.full((2,3),np.nan),np.zeros((3,2))):
            with self.assertRaises(ValueError):cb.bounds(bad)
        idx,wt = cb.flat(self.q)
        with self.assertRaises(ValueError):cb.degree(idx,-wt)
        with self.assertRaises(ValueError):cb.degree(idx,wt*.5)
        with self.assertRaises(ValueError):cb.pair_matrix(self.q,0)
        unprojected = ProjectedMesh(cb.mesh.original,project=False,audit_eigenvalues=False)
        with self.assertRaises(ValueError):CorrectionBounds(unprojected)


if __name__=='__main__':
    started = time.perf_counter()
    with threadpool_limits(1):
        program = unittest.main(exit=False)
    result = program.result
    here = Path(__file__).resolve().parent
    (here/'results').mkdir(exist_ok=True)
    (here/'results'/'tests_summary.json').write_text(json.dumps(dict(
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        ok=result.wasSuccessful(),elapsed_s=time.perf_counter()-started,
        numpy=np.__version__,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (here/'bounds.py',Path(__file__))}),indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
