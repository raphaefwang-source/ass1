"""Checks for the full periodic-image friction option (LatticeFriction)."""
import json
import unittest
import numpy as np
import prl_toy_models as model


def direct_lattice(dr, box, kernel, **kw):
    """Brute-force image sum with the same truncation radius, no Chebyshev fit."""
    ref = model.LatticeFriction(box, kernel=kernel, **kw)
    shifts = model.lattice_shifts(ref.box, ref.r_tail+0.5*np.linalg.norm(ref.box))
    return model.image_sum_tensors(dr, shifts, kernel=kernel, **kw)


class LatticeFrictionTests(unittest.TestCase):
    def test_chebyshev_far_field_matches_direct_image_sum(self):
        rng = np.random.default_rng(11)
        for box in (5.5, np.array([5.5, 6.1, 7.0])):
            q = rng.uniform(0, 1, (20, 3))*box
            _, _, dr, _ = model.pair_geometry(q, box)
            for kernel in ('A', 'B'):
                fast = model.LatticeFriction(box, kernel=kernel).pair_tensors(dr)
                ref = direct_lattice(dr, box, kernel)
                np.testing.assert_allclose(fast, ref, rtol=0, atol=1e-11*np.abs(ref).max())

    def test_truncation_converges(self):
        rng = np.random.default_rng(12)
        _, _, dr, _ = model.pair_geometry(rng.uniform(0, 5.5, (8, 3)), 5.5)
        for kernel in ('A', 'B'):
            values = [model.image_sum_tensors(dr, model.lattice_shifts(5.5, reach), kernel=kernel)
                      for reach in (20., 30., 40., 50.)]
            steps = [np.abs(b-a).max() for a, b in zip(values, values[1:])]
            self.assertTrue(steps[0] > steps[1] > steps[2])
            self.assertLess(steps[2], 1e-8)

    def test_gamma_structure_and_frozen_fdt(self):
        rng = np.random.default_rng(13)
        q = rng.uniform(0, 5.5, (10, 3))
        n = len(q)
        translations = np.kron(np.ones((n, 1)), np.eye(3))
        proj = np.eye(3*n)-translations@translations.T/n
        for kernel in ('A', 'B'):
            G = model.LatticeFriction(5.5, kernel=kernel).matrix(q)
            np.testing.assert_allclose(G, G.T, atol=1e-13)
            np.testing.assert_allclose(G@translations, 0, atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(G).min(), -1e-11)
            vec, damp, noise = model.thermostat_factors(G, .005, 1., .7)
            R, S = (vec*damp)@vec.T, (vec*noise)@vec.T
            np.testing.assert_allclose(R@(.7*proj)@R.T+proj@S@S.T@proj, .7*proj, atol=1e-12)

    def test_short_range_limit_equals_minimum_image(self):
        rng = np.random.default_rng(14)
        q = rng.uniform(0, 2, (12, 3))  # all components |dr| <= 2 in a 5.5 box
        for kernel in ('A', 'B'):
            kw = dict(kernel=kernel, kappa=10.)
            lattice = model.LatticeFriction(5.5, **kw).matrix(q)
            minimum = model.friction_matrix(q, 5.5, **kw)
            np.testing.assert_allclose(lattice, minimum, rtol=0, atol=1e-8*np.abs(minimum).max())

    def test_zero_fourier_mode_is_retained(self):
        # Cell average of the periodic pair tensor equals its k=0 coefficient ghat(0)/(3V) I.
        m = 36
        x = (np.arange(m)+0.5)/m-0.5
        grid = np.stack(np.meshgrid(x, x, x, indexing='ij'), -1).reshape(-1, 3)
        box = np.array([5.5, 6.1, 7.0])
        for kernel in ('A', 'B'):
            lat = model.LatticeFriction(box, kernel=kernel)
            mean = lat.pair_tensors(grid*box).mean(axis=0)
            np.testing.assert_allclose(mean, lat.zero_mode*np.eye(3), rtol=0, atol=5e-3*lat.zero_mode)
            self.assertGreater(lat.zero_mode, 0.05)

    def test_short_lattice_trajectory_records_convention(self):
        tr = model.simulate('lj', 'B', n=27, burn=0, steps=8, stride=4, friction_images='lattice')
        meta = json.loads(str(tr['metadata']))
        self.assertEqual(meta['friction_record']['images'], 'lattice')
        self.assertEqual(meta['sample_interval'], 0.02)
        self.assertTrue(meta['V_kind'].startswith('velocity'))
        self.assertTrue(np.all(np.isfinite(tr['Q'])) and np.all(np.isfinite(tr['V'])))
        self.assertLess(tr['diagnostics'][:, 4].max(), 1e-11)


if __name__ == '__main__':
    unittest.main(verbosity=2)
