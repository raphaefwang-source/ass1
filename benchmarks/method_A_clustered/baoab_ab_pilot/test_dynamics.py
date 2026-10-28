"""Local conservative-force, BAOAB sequencing, rebuild, and RNG checks."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import unittest
from unittest import mock
import numpy as np
from model import lj_force, switch, fcc_configuration, make_initial, L_BOX, AMPLITUDE
from dynamics import PilotSystem, FFTMeter


class ConservativeTests(unittest.TestCase):
    def test_force_finite_difference_including_switch(self):
        for r in (1.1, 1.6, 2.1, 2.3, 2.49, 2.51):
            q = np.array([[0.,0.,0.],[r,0.,0.]])
            f,_,_ = lj_force(q, 8.)
            h = 2e-6
            qp, qm = q.copy(), q.copy()
            qp[0,0] += h; qm[0,0] -= h
            fd = -(lj_force(qp,8.)[1]-lj_force(qm,8.)[1])/(2*h)
            self.assertAlmostEqual(f[0,0],fd,delta=2e-7*max(1.,abs(fd)))
            np.testing.assert_allclose(f.sum(axis=0),0.,atol=1e-13)

    def test_switch_continuity(self):
        for point in (2.,2.5):
            values, derivatives = switch(np.array([point-1e-7,point,point+1e-7]))
            self.assertLess(np.ptp(values),1e-12)
            self.assertLess(np.max(np.abs(derivatives)),1e-10)
        q = np.array([[0.,0.,0.],[2.5,0.,0.]])
        f,u,_ = lj_force(q,8.)
        self.assertEqual(u,0.)
        np.testing.assert_array_equal(f,0.)

    def test_periodic_force_and_translation(self):
        q = np.array([[.1,.2,.3],[7.2,.3,.4],[3.,4.,2.]])
        f,u,r = lj_force(q,8.)
        other = q + np.array([[8.,0.,-8.],[0.,16.,0.],[-8.,0.,8.]])
        ff,uu,rr = lj_force(other,8.)
        np.testing.assert_allclose(f,ff,atol=2e-12)
        self.assertAlmostEqual(u,uu,places=10)
        self.assertAlmostEqual(r,rr,places=12)

    def test_fcc_physical_constants(self):
        q,L = fcc_configuration()
        self.assertEqual(q.shape,(256,3))
        self.assertAlmostEqual(256/L**3,.75)
        self.assertAlmostEqual(L,L_BOX)
        self.assertAlmostEqual(AMPLITUDE,.5*1.3*np.exp(.7*1.3))
        self.assertGreater(lj_force(q,L)[2],1.)

    def test_preparation_distinct_seed_and_com(self):
        one,rows,meta = make_initial(914,steps=4,N=4)
        two,_,_ = make_initial(915,steps=4,N=4)
        self.assertEqual(len(rows),4)
        np.testing.assert_allclose(one['p'].sum(axis=0),0.,atol=1e-13)
        self.assertFalse(np.array_equal(one['q'],two['q']))
        self.assertEqual(meta['steps'],4)
        np.testing.assert_allclose(one['q'],np.mod(one['q_unwrapped'],float(one['L'])))


class MovingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = dict(N=4,L=8.,rho=.75,xi=.85,sR=4.1,sF=2.,M=12,p=3)
        cls.system = PilotSystem(cls.params,L=8.)
        cls.q = np.array([[.15,.2,.3],[1.4,.3,.35],[2.8,1.1,.6],[4.,3.,2.]])
        cls.p = np.random.default_rng(712).normal(size=(4,3))
        cls.p -= cls.p.mean(axis=0)

    def test_ab_identical_graph_and_no_self(self):
        a = self.system.build(self.q,'A')
        b = self.system.build(self.q,'B')
        self.assertIs(a.op.mesh,b.op.mesh)
        for name in ('i','j','d','w','Dm','DS'):
            np.testing.assert_array_equal(getattr(a.op,name),getattr(b.op,name))
        self.assertFalse(np.any(a.edges.i==a.edges.j))
        self.assertEqual(a.edges.backend,'compiled')
        self.assertEqual(a.metrics['coefficient_cache_initial_size'],0)
        self.assertFalse(a.metrics['coefficient_first_prepare_hit'])

    def test_changed_positions_rebuild_assignments_and_eigen(self):
        a = self.system.build(self.q,'A')
        changed = self.q.copy(); changed[0] += [.08,.03,0.]
        b = self.system.build(changed,'A')
        self.assertNotEqual(a.token,b.token)
        self.assertIsNot(a.op,b.op)
        self.assertIsNot(a.edges,b.edges)
        self.assertFalse(np.array_equal(a.op.wt,b.op.wt))
        self.assertFalse(np.array_equal(a.edges.K,b.edges.K))
        self.assertFalse(np.array_equal(a.edges.eigenvalues,b.edges.eigenvalues))
        self.assertIs(a.op.mesh,b.op.mesh)
        self.assertFalse(b.metrics['coefficient_first_prepare_hit'])

    def test_actual_moving_step_com_rng_and_timing_accounting(self):
        for method in ('A','B'):
            q,p,row = self.system.step(self.q,self.p,method,9,9,np.random.default_rng(902),dt=.005)
            self.assertTrue(np.isfinite(q).all() and np.isfinite(p).all())
            self.assertGreater(np.linalg.norm(q-self.q),0.)
            self.assertLess(np.linalg.norm(p.sum(axis=0)-self.p.sum(axis=0)),1e-12)
            self.assertGreater(row['t_baoab_s'],row['t_dynamic_setup_s'])
            self.assertGreaterEqual(row['t_baoab_other_s'],-1e-9)
            self.assertEqual(row['negative_ritz_count'],0)
            self.assertTrue(row['q_dependent_setup_rebuilt'])
            if method=='A':
                self.assertEqual(len({row['z_sha256'],row['eta_forward_sha256'],row['eta_reverse_sha256']}),3)

    def test_baoab_sequence_at_midpoint_with_identity_ou(self):
        dt = .005
        def identity_ou(action,p,z,dt,rd,rn,mass,theta):
            # Reuse a zero-time call solely to provide the real diagnostic schema.
            from ou_core import ou_b
            return ou_b(action,p,z,0.,rd,rn,mass,0.)
        f0 = lj_force(self.q,8.)[0]
        ph = self.p+.5*dt*f0
        qm = self.q+.5*dt*ph
        expected_q = qm+.5*dt*ph
        expected_p = ph+.5*dt*lj_force(np.mod(expected_q,8.),8.)[0]
        with mock.patch('dynamics.ou_b',side_effect=identity_ou):
            q,p,_ = self.system.step(self.q,self.p,'B',9,9,np.random.default_rng(919),dt)
        np.testing.assert_allclose(q,expected_q,atol=1e-14)
        np.testing.assert_allclose(p,expected_p,atol=1e-14)
        np.testing.assert_array_equal(self.system.last_built.q_mid,np.mod(qm,8.))

    def test_wrapped_unwrapped_separate(self):
        q0 = self.q+np.array([8.,-16.,24.])
        q,p,row = self.system.step(q0,self.p,'B',9,9,np.random.default_rng(311))
        self.assertTrue(np.any((q<0)|(q>8)))
        self.assertTrue(np.all((self.system.last_built.q_mid>=0)&(self.system.last_built.q_mid<8)))

    def test_fft_meter_preserves_transform(self):
        x = np.random.default_rng(3).normal(size=(4,4,4))
        expected = np.fft.rfftn(x)
        with FFTMeter() as meter:
            result = np.fft.rfftn(x)
        np.testing.assert_array_equal(result,expected)
        self.assertEqual(meter.calls,1)
        self.assertGreater(meter.elapsed,0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
