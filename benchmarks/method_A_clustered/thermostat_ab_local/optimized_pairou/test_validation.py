"""Independent small-system checks of the public paired-validation helpers."""
from types import SimpleNamespace
import unittest

import bench_common as C
import validate as V
np, P, O = C.np, C.P, C.O
from compiled_edge import CompiledEdgeOU


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.N = 3
        self.edges = O.EdgeOU([
            (0, 1, np.array([[1.4, .2, 0], [.2, .6, .1], [0, .1, .9]])),
            (1, 2, np.array([[.5, .1, .2], [.1, 1.2, 0], [.2, 0, .8]])),
        ], n_particles=3)
        long_edges = [
            (0, 2, np.array([[.8, -.1, .1], [-.1, .7, 0], [.1, 0, 1.1]])),
            (0, 1, .2*np.eye(3)), (1, 2, .3*np.eye(3)),
        ]
        self.GL = C.S.edges_dense(long_edges, self.N)
        self.G = self.GL+C.S.edges_dense(list(zip(self.edges.i, self.edges.j, self.edges.K)), self.N)
        self.system = SimpleNamespace(full_action=lambda x:self.G@x,
                                      long_action=lambda x:self.GL@x)
        self.full, self.long = P.DenseOU(self.G), P.DenseOU(self.GL)

    def test_full_rank_shared_paths_match_independent_dense_reference(self):
        rng = np.random.default_rng(872)
        p = rng.normal(size=9)+np.tile([.2, -.1, .3], 3)
        dt, k = .2, 2
        noise = V.draws(rng, k, len(self.edges.i), 9)
        compiled = CompiledEdgeOU(list(zip(self.edges.i, self.edges.j, self.edges.K)),
            n_particles=3, configuration_token='synthetic_validation', backend='compiled')
        self.assertEqual(compiled.backend, 'compiled')
        reference = self.long.matrices(dt/k)[:2]
        ideal = V.ideal_step(self.edges, p, dt, k, noise, 'A', reference)
        for engine in (self.edges, compiled):
            actual, diag = V.finite_step(self.system, engine, p, dt, 6, 6, k, noise, 'A')
            np.testing.assert_allclose(actual, ideal, atol=8e-15, rtol=8e-15)
            self.assertEqual(diag['actual_full_actions'], 0)
            self.assertGreater(diag['actual_long_actions'], 0)
            self.assertLess(V.com_error(actual, p), 4e-15)
        actual, _ = V.finite_step(self.system, self.edges, p, dt, 6, 6, 1, noise, 'B')
        ideal = V.ideal_step(self.edges, p, dt, 1, noise, 'B', self.full.matrices(dt)[:2])
        np.testing.assert_allclose(actual, ideal, atol=8e-15, rtol=8e-15)

    def test_ideal_A_noise_columns_give_invariance_covariance(self):
        dt, k = .3, 2
        RA = V.ideal_mean(self.edges, self.long, dt, k)
        target = self.full.Pi-RA@self.full.Pi@RA.T
        covariance = np.zeros_like(target)
        noise = V.zero_draws(k, len(self.edges.i), 9)
        reference = self.long.matrices(dt/k)[:2]
        for key in ('z', 'ef', 'er'):
            for coordinate in np.ndindex(noise[key].shape):
                noise[key][coordinate] = 1.
                column = V.ideal_step(self.edges, np.zeros(9), dt, k, noise, 'A', reference)
                covariance += np.outer(column, column)
                noise[key][coordinate] = 0.
        np.testing.assert_allclose(covariance, target, atol=2e-15, rtol=2e-14)
        mean, _ = V.finite_step(self.system, self.edges, np.arange(9.), dt,
            6, 6, k, V.zero_draws(k, len(self.edges.i), 9), 'A')
        np.testing.assert_allclose(mean, RA@np.arange(9.), atol=5e-15, rtol=2e-14)

    def test_directional_statistics_use_sample_variance(self):
        actual = V.scalar_statistics(np.array([-2., -1., 0., 1., 2.]))
        self.assertEqual(actual['mean'], 0.)
        self.assertEqual(actual['variance'], 2.5)
        self.assertEqual(actual['fourth_moment'], 6.8)
        self.assertLess(actual['variance_ci_low'], actual['variance'])
        self.assertGreater(actual['variance_ci_high'], actual['variance'])


if __name__ == '__main__':
    unittest.main()
