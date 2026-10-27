"""Small independent dense checks for the reusable frozen-OU kernels."""
import unittest
import numpy as np
from scipy.linalg import null_space

from ou_core import EdgeOU, PSDViolation, lanczos_action, ou_a, ou_b, project


def basis(N):
    C = np.tile(np.eye(3), (N, 1))
    return null_space(C.T)


def matrix_function(G, dt, kind, mass=1., theta=1.):
    B = basis(len(G)//3)
    eigenvalues, U = np.linalg.eigh(B.T @ G @ B)
    eigenvalues = np.maximum(eigenvalues, 0.)
    if kind == 'damp':
        values = np.exp(-dt * eigenvalues/mass)
    else:
        values = np.sqrt(mass*theta * -np.expm1(-2*dt*eigenvalues/mass))
    return (B @ U * values) @ (B @ U).T


class TestLanczos(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1729)
        self.N = 5
        self.B = basis(self.N)
        A = self.rng.normal(size=(12, 12))
        self.G = self.B @ (A @ A.T + .04*np.eye(12)) @ self.B.T

    def test_dense_both_functions_mass_and_temperature(self):
        v = self.rng.normal(size=(self.N, 3))
        for kind in ('damp', 'noise'):
            out, d = lanczos_action(lambda v: self.G@v, v, 100, .21, kind, mass=2.3, theta=.7)
            truth = matrix_function(self.G, .21, kind, 2.3, .7) @ v.ravel()
            np.testing.assert_allclose(out.ravel(), truth, atol=2e-13, rtol=2e-12)
            self.assertLess(d['orthogonality_error'], 2e-13)
            self.assertEqual(d['rank_used'], 12)
            self.assertEqual(d['matvec_count'], d['rank_used'])
            self.assertEqual(d['reorth_passes'], 2*d['rank_used'])
            self.assertGreater(d['t_reorth_s'], 0)

    def test_finite_rank_is_reported_and_not_declared_converged(self):
        v = self.rng.normal(size=15)
        out, d = lanczos_action(lambda v: self.G@v, v, 2, .2, 'noise')
        truth = matrix_function(self.G, .2, 'noise')@v
        self.assertGreater(np.linalg.norm(out-truth)/np.linalg.norm(truth), 1e-3)
        self.assertEqual(d['rank_requested'], 2)
        self.assertEqual(d['rank_used'], 2)
        self.assertFalse(d['happy_breakdown'])
        self.assertGreater(d['beta_tail'], 0)

    def test_projection_and_com(self):
        v = self.rng.normal(size=(5, 3))
        np.testing.assert_allclose(project(v).sum(0), 0, atol=1e-15)
        np.testing.assert_allclose(project(project(v)), project(v), atol=3e-16)
        com = np.tile([2., -1., .3], (5, 1))
        for kind in ('damp', 'noise'):
            out, _ = lanczos_action(lambda v:self.G@v, com, 12, .4, kind)
            np.testing.assert_allclose(out, 0, atol=1e-15)
        z = self.rng.normal(size=15)
        out, d = ou_b(lambda v:self.G@v, v, z, .2, 12, 12)
        np.testing.assert_allclose(out.sum(0), v.sum(0), atol=2e-15)
        self.assertIn('t_full_lanczos_damp', d)

    def test_zero_operator_happy_breakdown(self):
        v = self.rng.normal(size=15)
        damp, dd = lanczos_action(np.zeros_like, v, 10, 1., 'damp')
        noise, nd = lanczos_action(np.zeros_like, v, 10, 1., 'noise')
        np.testing.assert_allclose(damp, project(v), atol=3e-16)
        np.testing.assert_array_equal(noise, 0.)
        self.assertTrue(dd['happy_breakdown'])
        self.assertEqual(dd['rank_used'], 1)
        self.assertEqual(nd['ritz_min_raw'], 0.)

    def test_exact_additional_zero_mode(self):
        # A relative x mode is in the nullspace; no regularization may alter it.
        v = np.array([1.,0.,0.,-1.,0.,0.])
        K = np.diag([0., 1., 2.])
        G = np.block([[K, -K], [-K, K]])
        damp, _ = lanczos_action(lambda v:G@v, v, 3, .4, 'damp')
        noise, _ = lanczos_action(lambda v:G@v, v, 3, .4, 'noise')
        np.testing.assert_array_equal(damp, v)
        np.testing.assert_array_equal(noise, 0.)

    def test_negative_rejects_both_functions_and_clips_roundoff(self):
        v = project(self.rng.normal(size=15))
        for kind in ('damp', 'noise'):
            with self.assertRaises(PSDViolation) as caught:
                lanczos_action(lambda v:-v, v, 4, .1, kind)
            self.assertLess(caught.exception.diagnostics['ritz_min_raw'], -.99)
            out, d = lanczos_action(lambda v:-1e-16*v, v, 4, .1, kind)
            self.assertGreater(d['clipped_negative_count'], 0)
            np.testing.assert_allclose(out, v if kind=='damp' else 0., atol=1e-15)

    def test_expm1_retains_tiny_time_noise(self):
        v = project(self.rng.normal(size=15))
        out, _ = lanczos_action(lambda v:2*v, v, 4, 1e-20, 'noise')
        np.testing.assert_allclose(out, 2e-10*v, atol=1e-25, rtol=3e-15)
        self.assertGreater(np.linalg.norm(out), 0)

    def test_individual_ranks_and_dense_step(self):
        p, z = self.rng.normal(size=(2, 15))
        out, d = ou_b(lambda v:self.G@v, p, z, .1, 3, 5)
        self.assertEqual(d['damp_rank_used'], 3)
        self.assertEqual(d['noise_rank_used'], 5)
        out, _ = ou_b(lambda v:self.G@v, p, z, .1, 12, 12)
        truth = (p-project(p)) + matrix_function(self.G, .1, 'damp')@p + matrix_function(self.G, .1, 'noise')@z
        np.testing.assert_allclose(out, truth, atol=3e-14, rtol=1e-12)


class TestEdgesAndSplitting(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(37)
        # Noncommuting, full PSD edge matrices; no rank-one shortcut is valid.
        A = self.rng.normal(size=(3, 3))
        B = self.rng.normal(size=(3, 3))
        self.edges = EdgeOU([(0,1,A@A.T), (1,2,B@B.T)])
        self.N = 3
        self.Pi = basis(3)@basis(3).T
        C = self.rng.normal(size=(6,6))
        BB = basis(3)
        self.GL = BB@(C@C.T)@BB.T

    def linear_sweep(self, tau, reverse=False):
        E = np.eye(9)
        return np.column_stack([self.edges.mean_sweep(E[:,j],tau,reverse) for j in range(9)])

    def noise_sweep(self, tau, reverse=False):
        E = np.eye(6)
        return np.column_stack([self.edges.sweep(np.zeros(9),E[:,j].reshape(2,3),tau,reverse)[0] for j in range(6)])

    def test_one_edge_exact_mean_covariance_and_zero_mode(self):
        K = np.array([[2., .3, 0.], [.3, 1., 0.], [0., 0., 0.]])
        edge = EdgeOU([0],[1],[K])
        G = np.block([[K,-K],[-K,K]])
        R = np.column_stack([edge.mean_sweep(np.eye(6)[:,j],.3) for j in range(6)])
        F = np.column_stack([edge.sweep(np.zeros(6),np.eye(3)[j:j+1],.3)[0] for j in range(3)])
        exactR = matrix_function(G,.3,'damp') + np.eye(6)-basis(2)@basis(2).T
        exactF = matrix_function(G,.3,'noise')
        np.testing.assert_allclose(R,exactR,atol=2e-15)
        np.testing.assert_allclose(F@F.T,exactF@exactF.T,atol=2e-15)
        np.testing.assert_allclose(R@R.T+F@F.T,np.eye(6),atol=2e-15)
        self.assertEqual(edge.eigenvalues[0,0],0.)
        self.assertGreater(edge.t_setup,0.)

    def test_forward_reverse_differ_and_noise_is_indexed_original_edge(self):
        p=self.rng.normal(size=9)
        eta=self.rng.normal(size=(2,3))
        forward,_=self.edges.sweep(p,eta,.2)
        reverse,_=self.edges.sweep(p,eta,.2,True)
        self.assertGreater(np.linalg.norm(forward-reverse),1e-3)
        # Independent direct reference, update in reverse but retain eta[e].
        q=p.reshape(3,3).copy()
        for e in (1,0):
            i,j=self.edges.i[e],self.edges.j[e]
            lam,U=np.linalg.eigh(self.edges.K[e])
            a=np.exp(-.4*lam)
            b=np.sqrt(2*-np.expm1(-.8*lam))
            r=q[i]-q[j]
            rn=U@(a*(U.T@r)+b*(U.T@eta[e]))
            increment=(rn-r)/2
            q[i]+=increment;q[j]-=increment
        np.testing.assert_allclose(reverse,q.ravel(),atol=3e-15)

    def test_independent_split_conditional_covariance_and_stationarity(self):
        dt=.27;h=dt/2
        L=matrix_function(self.GL,dt,'damp') + np.eye(9)-self.Pi
        FL=matrix_function(self.GL,dt,'noise')
        S1=self.linear_sweep(h);S2=self.linear_sweep(h,True)
        B1=self.noise_sweep(h);B2=self.noise_sweep(h,True)
        RA=S2@L@S1
        # Independently propagate the three independent noise covariances.
        CA=S2@L@B1@B1.T@L.T@S2.T + S2@FL@FL.T@S2.T + B2@B2.T
        np.testing.assert_allclose(CA,self.Pi-RA@self.Pi@RA.T,atol=4e-15)
        np.testing.assert_allclose(RA@self.Pi@RA.T+CA,self.Pi,atol=4e-15)
        p=self.rng.normal(size=9);zl=self.rng.normal(size=9)
        ef=self.rng.normal(size=(2,3));er=self.rng.normal(size=(2,3))
        out,d=ou_a(lambda v:self.GL@v,p,zl,ef,er,self.edges,dt,6,6)
        truth=RA@p + S2@L@B1@ef.ravel() + S2@FL@zl + B2@er.ravel()
        np.testing.assert_allclose(out,truth,atol=4e-15)
        np.testing.assert_allclose(out.reshape(3,3).sum(0),p.reshape(3,3).sum(0),atol=8e-16)
        self.assertEqual(d['pair_updates'],4)
        self.assertIn('t_long_lanczos_noise',d)
        # Reusing the first sweep noise introduces cross terms and is wrong.
        reused=(S2@L@B1+B2)@(S2@L@B1+B2).T + S2@FL@FL.T@S2.T
        self.assertGreater(np.linalg.norm(reused-CA),1e-2)

    def test_monte_carlo_pair_stationary_mean_and_covariance(self):
        # This samples actual supplied independent normals, not assembled columns.
        edge=EdgeOU([(0,1,self.edges.K[0])])
        M=16000
        values=np.empty((M,3))
        for k in range(M):
            p=self.rng.normal(size=(2,3))
            out,_=edge.sweep(p,self.rng.normal(size=(1,3)),.2)
            values[k]=(out[0]-out[1])/np.sqrt(2)
        self.assertLess(np.max(np.abs(values.mean(0))),.03)
        np.testing.assert_allclose(np.cov(values,rowvar=False),np.eye(3),atol=.035)

    def test_invalid_and_roundoff_negative_edges(self):
        with self.assertRaises(PSDViolation):
            EdgeOU([(0,1,np.diag([-1e-3,1.,2.]))])
        edge=EdgeOU([(0,1,np.diag([-1e-16,1.,2.]))])
        self.assertEqual(edge.spectrum_diagnostics['clipped_negative_count'],1)
        self.assertEqual(edge.eigenvalues[0,0],0.)
        with self.assertRaises(ValueError):
            EdgeOU([(0,1,np.array([[1.,1.,0.],[0.,1.,0.],[0.,0.,1.]]))])


if __name__ == '__main__':
    unittest.main()
