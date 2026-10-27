"""Independent self-excluded bound, periodic support and cancellation tests."""
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

from offdiag_bounds import OffdiagBounds, CorrectionBounds, np
from grid_psd import ProjectedMesh, PAIRS6, COMPONENT_INDEX
from positive_mesh import PositiveMesh
import ewald_exp_lib as E
from threadpoolctl import threadpool_limits


class SyntheticMesh:
    stencil = PositiveMesh.stencil
    flat = PositiveMesh.flat
    def __init__(self,C,p=1,L=None):
        self.delta_table = C
        self.M = C.shape[1]
        self.L = float(L if L is not None else self.M)
        self.h,self.p,self.projected = self.L/self.M,p,True
        eig = np.linalg.eigvalsh(np.moveaxis(C,0,-1)[...,COMPONENT_INDEX])
        self.diagnostics = dict(max_negative_part_spectral=float(eig.max()),
            negative_blocks=int(np.any(C!=0,axis=0).sum()),
            fraction_negative_blocks=float(np.any(C!=0,axis=0).mean()),
            correction_fro=float(np.linalg.norm(C)),min_eig_before=-float(eig.max()),
            inversion_residual_before=0.)


class RepeatingStencilMesh(SyntheticMesh):
    def stencil(self,x):
        # Repeated wrapped index 0 and a wrap alias in a three-point stencil.
        g = np.broadcast_to(np.array([0,3,-1]),(len(x),3,3)).copy()
        w = np.broadcast_to(np.array([.2,.3,.5]),g.shape).copy()
        return g,w


def isolated_origin(M=7,delta=.7):
    C = np.zeros((6,M,M,M));C[:3,0,0,0] = delta
    return C


def direct_off(cb,q):
    W = cb.pair_matrix(q).copy()
    W[np.arange(len(q)),np.arange(len(q))] = 0.
    return W.sum(axis=1),E.lap_full(W)


class OffdiagTests(unittest.TestCase):
    def setUp(self):
        self.q = np.random.default_rng(701).uniform(0,5,(4,3))
        self.K = E.kernel(2.,1.3,0.)

    def projected(self,M=8,p=3):
        return ProjectedMesh.from_parameters(5.,self.K,2*np.pi/5*(M//2-1),M,p,audit_eigenvalues=False)

    def assert_degree_and_bound(self,ob,q,stable='auto'):
        r,D = ob.bounds(q,stable=stable)
        true,G = direct_off(ob.cb,q)
        scale = max(float(np.abs(true).max()),ob.cb.delta,1e-300)
        np.testing.assert_allclose(D,true,atol=3e-14*scale,rtol=3e-13)
        actual = float(np.linalg.norm(G,2))
        self.assertGreaterEqual(r['B_off']+3e-13*scale,actual)
        self.assertGreaterEqual(r['B_improved']+3e-13*scale,actual)
        self.assertLessEqual(r['B_improved'],r['B_old'])
        self.assertEqual(r['B_improved'],min(r['B_simple'],r['B_total'],r['B_off']))
        json.dumps(r,allow_nan=False)
        return r,D,true,G

    def test_odd_even_p1_and_multiple_orders(self):
        for M in (7,8):
            for p in (1,2,3):
                cb = CorrectionBounds(self.projected(M,p))
                ob = OffdiagBounds(cb)
                r,D,true,G = self.assert_degree_and_bound(ob,self.q)
                W = cb.direct_pair_matrix(self.q)
                np.testing.assert_allclose(ob.last_C_self,W[np.arange(4),np.arange(4)],atol=2e-16,rtol=3e-13)
                stable_D,_ = ob.stable_off_degree(self.q)
                np.testing.assert_allclose(stable_D,true,atol=2e-16,rtol=3e-13)

    def test_wrap_alias_M_less_than_2p_minus_1(self):
        M,p = 5,5
        C = isolated_origin(M)
        cb = CorrectionBounds(SyntheticMesh(C,p=p,L=5.))
        ob = OffdiagBounds(cb)
        self.assert_degree_and_bound(ob,self.q,stable='always')
        direct = cb.direct_pair_matrix(self.q)
        np.testing.assert_allclose(ob.self_tensors(self.q),direct[np.arange(4),np.arange(4)],atol=2e-16,rtol=3e-13)
        self.assertLess(ob.last_self_profile['self_autocorrelation_partition_max_defect'],3e-15)

    def test_repeated_stencil_indices_and_modular_aliases(self):
        C = isolated_origin(3)
        cb = CorrectionBounds(RepeatingStencilMesh(C,p=3,L=3.))
        ob = OffdiagBounds(cb)
        q = self.q[:3]%3
        self.assert_degree_and_bound(ob,q,stable='always')
        W = cb.direct_pair_matrix(q)
        np.testing.assert_allclose(ob.last_C_self,W[np.arange(3),np.arange(3)],atol=2e-16,rtol=3e-13)

    def test_automatic_stability_for_tiny_offdiagonal_mass(self):
        cb = CorrectionBounds(SyntheticMesh(isolated_origin(),p=2))
        ob = OffdiagBounds(cb)
        q = np.array([[0.,0,0],[1.-1e-12,0,0]])
        r,D = ob.bounds(q)
        true,G = direct_off(cb,q)
        self.assertTrue(r['cancellation_unreliable_raw'])
        self.assertTrue(r['stable_fallback_used'])
        self.assertEqual(r['degreeoff_method'],'positive_local_occupancy')
        np.testing.assert_allclose(D,true,atol=1e-27,rtol=3e-15)
        self.assertGreater(r['raw_stable_offdiag_relative_difference'],1e-7)
        self.assertGreaterEqual(r['B_improved']+1e-26,np.linalg.norm(G,2))
        self.assertLess(r['stable_positive_mass_coefficient_sum'],1e-10)

    def test_exact_isolated_self_only_zero_not_a_floor(self):
        cb = CorrectionBounds(SyntheticMesh(isolated_origin(),p=1))
        ob = OffdiagBounds(cb)
        q = np.array([[0.,0,0],[2.,0,0],[4.,0,0]])
        r,D = ob.bounds(q)
        np.testing.assert_array_equal(D,0.)
        self.assertEqual(r['B_off'],0.)
        self.assertEqual(r['B_improved'],0.)
        self.assertTrue(r['stable_fallback_used'])
        self.assertTrue(r['stable_true_zero_from_occupancy'])
        self.assertEqual(r['stable_nonzero_mass_coefficients'],0)

    def test_singleton_self_cancellation(self):
        cb = CorrectionBounds(self.projected())
        ob = OffdiagBounds(cb)
        r,D = ob.bounds(self.q[:1])
        np.testing.assert_array_equal(D,0.)
        self.assertEqual(r['B_off'],0.)
        self.assertEqual(r['B_improved'],0.)
        self.assertEqual(r['degreeoff_method'],'singleton_exact_no_other_particles')
        np.testing.assert_allclose(cb.matvec(self.q[:1],[1.,2.,3.]),0.,atol=2e-16)

    def test_known_tight_off_bound_removes_large_self(self):
        # Two particles have edge delta*I and additional self alpha*I.
        C = isolated_origin(7,delta=10.)
        C[:3,1,0,0] = C[:3,-1,0,0] = .4
        cb = CorrectionBounds(SyntheticMesh(C,p=1))
        ob = OffdiagBounds(cb)
        q = np.array([[0.,0,0],[1.,0,0],[3.,0,0]])
        r,D,true,G = self.assert_degree_and_bound(ob,q)
        np.testing.assert_allclose(r['B_off'],.8,atol=8e-15)
        np.testing.assert_allclose(np.linalg.norm(G,2),.8,atol=2e-15)
        self.assertLess(r['B_improved'],r['B_old']/20)

    def test_zero_correction_and_json_null_ratios(self):
        cb = CorrectionBounds(SyntheticMesh(np.zeros((6,5,5,5)),p=2))
        ob = OffdiagBounds(cb)
        r,D = ob.bounds(self.q)
        np.testing.assert_array_equal(D,0.)
        self.assertIsNone(r['self_degree_trace_fraction'])
        self.assertIsNone(r['degreeoff_total_ratio'])
        self.assertIsNone(r['cancellation_index_raw'])
        self.assertFalse(r['stable_fallback_used'])
        self.assertEqual(r['B_improved'],0.)
        json.dumps(r,allow_nan=False)

    def test_fallback_budget_marks_unavailable_without_clipping(self):
        cb = CorrectionBounds(SyntheticMesh(isolated_origin(),p=2))
        ob = OffdiagBounds(cb,max_stable_queries=1)
        q = np.array([[0.,0,0],[1.-1e-12,0,0]])
        r,D = ob.bounds(q)
        self.assertTrue(r['stable_fallback_unavailable'])
        self.assertTrue(r['cancellation_unreliable_raw'])
        self.assertFalse(r['stable_fallback_used'])
        self.assertFalse(r['numerical_bound_available'])
        self.assertIsNone(r['B_off'])
        self.assertIsNone(r['B_improved'])
        np.testing.assert_array_equal(D,ob.last_D_off_raw)
        r,D = ob.bounds(q,stable='always')
        self.assertTrue(r['stable_fallback_used'])
        self.assertTrue(r['numerical_bound_available'])
        true,_ = direct_off(cb,q)
        np.testing.assert_allclose(D,true,atol=1e-27,rtol=3e-15)

    def test_no_new_fft_no_particle_dense_and_no_mutation(self):
        mesh = self.projected()
        cb = CorrectionBounds(mesh)
        original = {k:v.copy() for k,v in mesh.__dict__.items() if isinstance(v,np.ndarray)}
        keys = set(mesh.__dict__)
        old_hat = cb.C_hat
        ob = OffdiagBounds(cb)
        def forbidden(*args,**kwargs):
            raise AssertionError('The offdiag bound cannot form a particle pair matrix')
        cb.pair_matrix = cb.direct_pair_matrix = forbidden
        with patch('numpy.fft.rfftn',wraps=np.fft.rfftn) as transform:
            ob.bounds(self.q,stable='always')
            # One density FFT inherited from Dtotal, no correction-table rebuild.
            self.assertEqual(transform.call_count,1)
        self.assertIs(cb.C_hat,old_hat)
        self.assertEqual(set(mesh.__dict__),keys)
        for k,v in original.items():np.testing.assert_array_equal(mesh.__dict__[k],v)

    def test_zero_mode_translation_periodic_positions(self):
        cb = CorrectionBounds(self.projected())
        ob = OffdiagBounds(cb)
        r,D = ob.bounds(self.q,stable='always')
        r2,D2 = ob.bounds(self.q+[5.,-10.,15.],stable='always')
        np.testing.assert_allclose(D,D2,atol=2e-16,rtol=3e-13)
        np.testing.assert_allclose(cb.zero_mode,cb.C.sum(axis=(1,2,3)),atol=3e-15,rtol=3e-15)
        for e in np.eye(3):
            np.testing.assert_allclose(cb.matvec(self.q,np.tile(e,(4,1))),0.,atol=2e-16)

    def test_bad_mode_and_reused_total_degree(self):
        cb = CorrectionBounds(self.projected())
        ob = OffdiagBounds(cb)
        old,total = cb.bounds(self.q)
        r,D = ob.bounds(self.q,D_total=total,total_result=old)
        direct,_ = direct_off(cb,self.q)
        np.testing.assert_allclose(D,direct,atol=2e-16,rtol=3e-13)
        with self.assertRaises(ValueError):ob.bounds(self.q,stable='floor')
        with self.assertRaises(ValueError):ob.bounds(self.q,D_total=np.zeros((1,3,3)))


if __name__=='__main__':
    started = time.perf_counter()
    with threadpool_limits(1):program = unittest.main(exit=False)
    result = program.result
    here = Path(__file__).resolve().parent
    (here/'results').mkdir(exist_ok=True)
    (here/'results'/'tests_summary.json').write_text(json.dumps(dict(tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),ok=result.wasSuccessful(),
        elapsed_s=time.perf_counter()-started,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (here/'offdiag_bounds.py',Path(__file__))}),indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
