"""Pathwise and mathematical checks for the compiled sequential edge sweeps."""
import ctypes
from pathlib import Path
import tempfile
import unittest
import numpy as np

from compiled_edge import CompiledEdgeOU, HERE, FLAGS, build_library
from ou_core import EdgeOU, ou_a


def example_edges():
    # Shared vertices and noncommuting full matrices make order observable.
    A=np.array([[1.,.2,.3],[.4,1.3,-.1],[.2,.5,.8]])
    B=np.array([[1.2,-.3,.1],[.1,.6,.4],[.2,.3,1.5]])
    return [(0,1,A@A.T),(1,2,B@B.T),(2,3,np.diag([0.,.7,2.])),
            (0,3,np.zeros((3,3))),(0,2,np.outer([1.,2.,3.],[1.,2.,3.]))]


class TestCompiledEdge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled=CompiledEdgeOU(example_edges(),n_particles=4,configuration_token='initial-q')
        if cls.compiled.backend!='compiled':
            raise AssertionError(f'C backend did not compile: {cls.compiled.compile_info}')

    def setUp(self):
        self.rng=np.random.default_rng(2026102708)

    def test_true_compilation_and_provenance(self):
        info=self.compiled.compile_info
        self.assertIsInstance(self.compiled,EdgeOU)
        for key in ('compiler_command','compiler_version','source_sha256','library_sha256',
                    'library_path','original_cold_compile_s','compiled_now','t_compile_s','t_load_s'):
            self.assertIn(key,info)
        self.assertIn('-O3',info['compiler_command'])
        self.assertIn('-ffp-contract=off',info['compiler_command'])
        self.assertIn('-fno-fast-math',info['compiler_command'])
        self.assertNotIn('-ffast-math',info['compiler_command'])
        self.assertFalse(info['fast_math'])
        self.assertFalse(info['edge_parallelism'])
        self.assertEqual(len(info['source_sha256']),64)
        self.assertEqual(len(info['library_sha256']),64)

    def test_pathwise_forward_reverse_full_rank_deficient_zero_and_small_tau(self):
        old=EdgeOU(example_edges(),n_particles=4)
        p=self.rng.normal(size=(4,3));eta=self.rng.normal(size=(5,3))
        for tau in (0.,1e-20,1e-12,.01,.8):
            for reverse in (False,True):
                for mass,theta in ((1.,1.),(2.3,.7),(1.7,0.)):
                    with self.subTest(tau=tau,reverse=reverse,mass=mass,theta=theta):
                        actual,diag=self.compiled.sweep(p,eta,tau,reverse,mass,theta)
                        truth,_=old.sweep(p,eta,tau,reverse,mass,theta)
                        np.testing.assert_allclose(actual,truth,atol=8e-14,rtol=3e-14)
                        np.testing.assert_allclose(actual.sum(0),p.sum(0),atol=3e-15)
                        self.assertEqual(diag['pair_updates'],5)
                        self.assertEqual(diag['backend'],'compiled')

    def test_single_pair_mean_covariance_and_maxwell_stationarity(self):
        K=np.array([[2.,.3,0.],[.3,.8,0.],[0.,0.,0.]])
        edge=CompiledEdgeOU([(0,1,K)])
        mass,theta,tau=2.,.7,.13
        R=np.column_stack([edge.sweep(np.eye(6)[:,i],np.zeros((1,3)),tau,mass=mass,theta=theta)[0]
                           for i in range(6)])
        F=np.column_stack([edge.sweep(np.zeros(6),np.eye(3)[i:i+1],tau,mass=mass,theta=theta)[0]
                           for i in range(3)])
        lam,U=np.linalg.eigh(K)
        L=(U*np.exp(-2*tau*lam/mass))@U.T
        exactR=.5*np.block([[np.eye(3)+L,np.eye(3)-L],[np.eye(3)-L,np.eye(3)+L]])
        V=(U*(2*mass*theta*-np.expm1(-4*tau*lam/mass)))@U.T
        exactC=.25*np.block([[V,-V],[-V,V]])
        np.testing.assert_allclose(R,exactR,atol=1e-15)
        np.testing.assert_allclose(F@F.T,exactC,atol=1e-15)
        np.testing.assert_allclose(mass*theta*R@R.T+F@F.T,mass*theta*np.eye(6),atol=2e-15)

    def test_zero_operator_exact_identity_and_zero_mode_no_noise(self):
        zero=CompiledEdgeOU([(0,1,np.zeros((3,3)))])
        p=self.rng.normal(size=6);eta=self.rng.normal(size=(1,3))
        out,_=zero.sweep(p,eta,1.)
        np.testing.assert_array_equal(out,p)
        diagonal=CompiledEdgeOU([(0,1,np.diag([1.,0.,0.]))])
        out,_=diagonal.sweep(p,eta,.5)
        np.testing.assert_array_equal(out.reshape(2,3)[:,1:],p.reshape(2,3)[:,1:])

    def test_zero_tau_is_bitwise_identity(self):
        p=self.rng.normal(size=12);eta=self.rng.normal(size=(5,3))
        out,_=self.compiled.sweep(p,eta,0.)
        np.testing.assert_array_equal(out,p)

    def test_expm1_small_tau_noise_remains_nonzero(self):
        edge=CompiledEdgeOU([(0,1,np.eye(3))])
        eta=np.array([[1.,2.,3.]])
        output,_=edge.sweep(np.zeros(6),eta,1e-24)
        expected=np.sqrt(2e-24)*eta.ravel()
        np.testing.assert_allclose(output[:3],expected,rtol=2e-15,atol=1e-39)
        np.testing.assert_array_equal(output[3:],-output[:3])

    def test_order_and_original_noise_indexing_are_observable(self):
        p=self.rng.normal(size=12);eta=self.rng.normal(size=(5,3))
        forward,_=self.compiled.sweep(p,eta,.2)
        reverse,_=self.compiled.sweep(p,eta,.2,True)
        self.assertGreater(np.linalg.norm(forward-reverse),.01)
        reversed_edges=CompiledEdgeOU(list(reversed(example_edges())))
        manual_reverse,_=reversed_edges.sweep(p,eta[::-1],.2)
        np.testing.assert_allclose(reverse,manual_reverse,atol=5e-15)
        wrong_noise,_=reversed_edges.sweep(p,eta,.2)
        self.assertGreater(np.linalg.norm(reverse-wrong_noise),.01)

    def test_repeated_path_and_ou_a_subclass_dispatch(self):
        old=EdgeOU(example_edges());fast=CompiledEdgeOU(example_edges())
        p=self.rng.normal(size=12);pold=p.copy();pnew=p.copy()
        A=self.rng.normal(size=(12,12));Pi=np.eye(12)-np.tile(np.eye(3),(4,4))/4
        G=Pi@(A@A.T)@Pi
        for _ in range(12):
            z=self.rng.normal(size=12);ef=self.rng.normal(size=(5,3));er=self.rng.normal(size=(5,3))
            pold,_=ou_a(lambda v:G@v,pold,z,ef,er,old,.03,9,9)
            pnew,diag=ou_a(lambda v:G@v,pnew,z,ef,er,fast,.03,9,9)
        np.testing.assert_allclose(pnew,pold,atol=5e-14,rtol=2e-14)
        self.assertEqual(diag['pair_updates'],10)
        self.assertGreater(fast.coefficient_cache_size,0)

    def test_preparation_cache_all_physical_parameters(self):
        edge=CompiledEdgeOU(example_edges())
        first=edge.prepare(.2)
        self.assertFalse(first['coefficient_cache_hit'])
        self.assertTrue(edge.prepare(.2)['coefficient_cache_hit'])
        for args in ((.1,1.,1.),(.2,2.,1.),(.2,1.,2.)):
            self.assertFalse(edge.prepare(*args)['coefficient_cache_hit'])
            self.assertTrue(edge.prepare(*args)['coefficient_cache_hit'])
        self.assertEqual(edge.coefficient_cache_size,4)

    def test_graph_is_immutable_and_rebuild_invalidates(self):
        K=np.diag([1.,2.,3.]);records=[(0,1,K)]
        edge=CompiledEdgeOU(records,configuration_token='q1')
        edge.prepare(.1)
        oldhash=edge.graph_sha256
        K[0,0]=5.
        self.assertEqual(edge.K[0,0,0],1.)
        for name in ('i','j','K','U','eigenvalues'):
            with self.assertRaises(ValueError):
                getattr(edge,name).flags.writeable=True
            with self.assertRaises(AttributeError):
                setattr(edge,name,np.zeros(1))
        replacement=edge.rebuild(records,configuration_token='q2')
        self.assertEqual(replacement.coefficient_cache_size,0)
        self.assertNotEqual(oldhash,replacement.graph_sha256)
        self.assertEqual(edge.coefficient_cache_size,1)
        # Excite exactly the changed eigenmode; avoid accidental noise/drift cancellation.
        p=np.array([1.,0.,0.,-1.,0.,0.]);eta=np.zeros((1,3))
        self.assertGreater(np.linalg.norm(edge.sweep(p,eta,.1)[0]-replacement.sweep(p,eta,.1)[0]),.01)
        with self.assertRaises(ValueError):edge.rebuild(records,configuration_token='q1')
        with self.assertRaises(ValueError):edge.rebuild(records,configuration_token=None)

    def test_noncontiguous_inputs_are_copied_without_mutation(self):
        large=self.rng.normal(size=(4,6));p=large[:,::2]
        eta_large=self.rng.normal(size=(5,6));eta=eta_large[:,::2]
        pcopy=p.copy();ecopy=eta.copy()
        output,_=self.compiled.sweep(p,eta,.1)
        self.assertEqual(output.shape,p.shape)
        np.testing.assert_array_equal(p,pcopy)
        np.testing.assert_array_equal(eta,ecopy)
        self.assertFalse(np.shares_memory(output,p))

    def test_empty_graph(self):
        empty=CompiledEdgeOU([],n_particles=4)
        p=self.rng.normal(size=(4,3))
        for reverse in (False,True):
            actual,diag=empty.sweep(p,np.empty((0,3)),.2,reverse)
            np.testing.assert_array_equal(actual,p)
            self.assertEqual(diag['pair_updates'],0)

    def test_validation(self):
        p=np.zeros(12);eta=np.zeros((5,3))
        for invalid in (np.zeros(11),np.zeros((4,4)),np.full(12,np.nan),np.empty(0)):
            with self.assertRaises(ValueError):self.compiled.sweep(invalid,eta,.1)
        for invalid in (np.zeros((5,2)),np.zeros(15),np.full((5,3),np.inf)):
            with self.assertRaises(ValueError):self.compiled.sweep(p,invalid,.1)
        with self.assertRaises(ValueError):self.compiled.sweep(np.zeros(6),eta,.1)
        with self.assertRaises(ValueError):self.compiled.sweep(p,eta,.1,reverse=1)
        for params in ((-.1,1.,1.),(.1,0.,1.),(.1,1.,-1.),(np.inf,1.,1.)):
            with self.assertRaises(ValueError):self.compiled.prepare(*params)
        with self.assertRaises(ValueError):CompiledEdgeOU([(0.5,1,np.eye(3))])
        with self.assertRaises(ValueError):CompiledEdgeOU([(0,0,np.eye(3))])
        with self.assertRaises(ValueError):CompiledEdgeOU([(0,1,np.eye(3))],n_particles=1)
        with self.assertRaises(ValueError):CompiledEdgeOU([(0,1,-np.eye(3))])

    def test_explicit_original_fallback_and_failed_compile_fallback(self):
        python=CompiledEdgeOU(example_edges(),backend='python')
        failed=CompiledEdgeOU(example_edges(),compiler=str(HERE/'missing-clang'))
        self.assertEqual(python.backend,'python')
        self.assertEqual(failed.backend,'python')
        self.assertIn('fallback_reason',failed.compile_info)
        p=self.rng.normal(size=12);eta=self.rng.normal(size=(5,3))
        expected,_=EdgeOU(example_edges()).sweep(p,eta,.1)
        np.testing.assert_array_equal(python.sweep(p,eta,.1)[0],expected)
        np.testing.assert_array_equal(failed.sweep(p,eta,.1)[0],expected)

    def test_c_kernel_rejects_invalid_graph_before_mutation(self):
        edge=self.compiled
        i=np.array([0,99],dtype=np.int64);j=np.array([1,2],dtype=np.int64)
        D=np.ones((2,3,3));S=np.zeros_like(D);p=np.arange(12,dtype=float);old=p.copy();eta=np.zeros((2,3))
        pointer=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        status=edge._library.pairou_sweep(4,2,i.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    j.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),pointer(D),pointer(S),pointer(p),pointer(eta),0)
        self.assertEqual(status,2)
        np.testing.assert_array_equal(p,old)

    def test_isolated_cold_compile_then_cache_load(self):
        with tempfile.TemporaryDirectory(prefix='unit-cold-',dir=HERE/'.build') as directory:
            _,cold=build_library(directory)
            _,warm=build_library(directory)
            self.assertTrue(cold['compiled_now'])
            self.assertGreater(cold['t_compile_s'],0.)
            self.assertFalse(warm['compiled_now'])
            self.assertEqual(warm['t_compile_s'],0.)
            self.assertEqual(cold['source_sha256'],warm['source_sha256'])
            self.assertEqual(cold['library_sha256'],warm['library_sha256'])
        with self.assertRaises(ValueError):build_library(HERE.parent/'outside-cache')


if __name__=='__main__':
    unittest.main()
