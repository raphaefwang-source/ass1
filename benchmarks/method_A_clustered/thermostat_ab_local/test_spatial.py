"""Structural tests for the production Grid PSD frozen spatial bridge."""
import unittest

import spatial as S
np = S.np
from neighbor_images import brute_force_images


class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small = S.make_system(N=2, family='uniform', seed=2)
        cls.medium = S.make_system(N=16, family='strong', seed=3)

    def test_all_images_aggregate_to_full_rank_pair(self):
        system = self.small
        op = system.op
        self.assertGreater(op.n_images, len(system.edges))
        self.assertGreater(system.metadata['max_images_per_edge'], 1)
        self.assertEqual(len(system.edges), 1)
        i, j, K = system.edges[0]
        self.assertEqual((i, j), (0, 1))
        self.assertGreater(np.linalg.eigvalsh(K).min(), 0.)
        # An independent brute-force image enumeration also checks that no
        # minimum-image-only graph has replaced production's interaction set.
        images = brute_force_images(system.q, system.L, op.rc)
        r = np.linalg.norm(images.d, axis=1)
        weights = op.K.theta_S(r)/r**2
        expected = np.einsum('p,pa,pb->ab', weights, images.d, images.d)
        np.testing.assert_allclose(K, expected, rtol=3e-13, atol=1e-14)

    def test_dense_assembly_matches_production_actions(self):
        rng = np.random.default_rng(642)
        for system in (self.small, self.medium):
            for _ in range(3):
                v = rng.normal(size=3*len(system.q))
                np.testing.assert_allclose(system.full_action(v), system.full_dense@v,
                                           rtol=2e-12, atol=2e-12)
                np.testing.assert_allclose(system.long_action(v), system.long_dense@v,
                                           rtol=2e-12, atol=2e-12)
                X = v.reshape(-1, 3)
                near = (np.einsum('nab,nb->na', system.op.DS, X)-system.op.real_H(X)).ravel()
                np.testing.assert_allclose(near, system.near_dense@v, rtol=2e-12, atol=2e-12)

    def test_long_matrix_matches_literal_stencil_contraction(self):
        system = self.small
        W = system.op.mesh.direct_pair_matrix(system.q)
        independent = S.pair_graph_dense(W)
        np.testing.assert_allclose(system.long_dense, independent, rtol=2e-12, atol=2e-12)

    def test_self_interactions_cancel_exactly_in_graph_conversion(self):
        rng = np.random.default_rng(82)
        W = rng.normal(size=(2, 2, 3, 3))
        original = S.pair_graph_dense(W)
        for i in range(2):
            W[i, i] += 13*np.eye(3)
        np.testing.assert_allclose(S.pair_graph_dense(W), original, atol=2e-15)

    def test_symmetry_psd_and_three_center_of_mass_modes(self):
        for system in (self.small, self.medium):
            T = np.tile(np.eye(3), (len(system.q), 1))
            for G in (system.near_dense, system.long_dense, system.full_dense):
                scale = max(np.linalg.norm(G, 2), 1.)
                np.testing.assert_allclose(G, G.T, atol=1e-12*scale)
                self.assertGreaterEqual(np.linalg.eigvalsh((G+G.T)/2).min(), -1e-12*scale)
                self.assertLess(np.linalg.norm(G@T), 1e-12*scale)
            for k in range(3):
                self.assertLess(np.linalg.norm(system.full_action(T[:, k])), 1e-12)
                self.assertLess(np.linalg.norm(system.long_action(T[:, k])), 1e-12)

    def test_legal_scaled_mesh_and_saved_configuration_hash(self):
        a = self.small
        b = S.make_system(N=2, family='uniform', seed=2)
        self.assertTrue(a.metadata['legal_grid'])
        self.assertLessEqual(a.metadata['eta'], .95)
        self.assertEqual(a.metadata['position_sha256'], b.metadata['position_sha256'])
        np.testing.assert_array_equal(a.q, b.q)

    def test_streamed_unsplit_sum_matches_independent_existing_direct(self):
        a = self.small
        R = 4.3
        actual, count = S.direct_unsplit_dense(a.q, a.L, R, budget=100)
        i, j, acc, counts = S.E.DP.annulus_tensors(a.q, a.L, [R], g=S.E.g_phys(.2))
        expected = S.E.DP.tensors_to_dense(i, j, acc[0], len(a.q))
        self.assertEqual(count, counts.sum())
        np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)

    def test_analytic_tail_dominates_discarded_annulus(self):
        a = self.small
        coarse, _ = S.direct_unsplit_dense(a.q, a.L, 8.)
        fine, _ = S.direct_unsplit_dense(a.q, a.L, 12.)
        actual = S.P.norm2(fine-coarse)
        self.assertLessEqual(actual, S.physical_tail_bound(len(a.q), a.L, 8.))
        self.assertLess(S.physical_tail_bound(len(a.q), a.L, 120.),
                        S.physical_tail_bound(len(a.q), a.L, 100.))


if __name__ == '__main__':
    unittest.main()
