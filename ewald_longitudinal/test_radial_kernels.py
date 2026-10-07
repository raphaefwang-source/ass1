#!/usr/bin/env python3
"""Unit checks for radial_kernels.py and toy_dynamics.py (about 1-2 min).

    python3 test_radial_kernels.py
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import radial_kernels as rk  # noqa: E402
import toy_dynamics as td  # noqa: E402
import prl_toy_models as v2  # noqa: E402
from verify_ewald_longitudinal import EwaldLongitudinal  # noqa: E402


class KernelIdentities(unittest.TestCase):
    def test_law_A_unit_amplitude_is_the_project_kernel(self):
        K0, KA = EwaldLongitudinal(0.7, 0.7), rk.RadialEwald("A", 0.7, 0.7)
        k, r = np.linspace(0, 8, 17), np.linspace(0.3, 9, 25)
        np.testing.assert_array_equal(np.array(KA.AB_scaled(k)), np.array(K0.AB_scaled(k)))
        np.testing.assert_array_equal(KA.theta_S_scaled_closed(r), K0.theta_S_scaled_closed(r))

    def test_toy_normalisation(self):
        for law in rk.LAWS:
            K = rk.toy_kernel(law, 0.5, 0.7, 1.3, 0.7)
            self.assertAlmostEqual(float(K.theta_full(np.array([1.3]))[0]), 0.5, places=14)
            r = np.linspace(0.4, 6, 9)
            np.testing.assert_allclose(K.theta_full(r), v2.radial_friction(r, law, 0.5, 0.7, 1.3), rtol=1e-14)

    def test_split_closed_form_and_k0(self):
        r = np.linspace(0.2, 9, 40)
        for law in rk.LAWS:
            for xi in (0.5, 0.7, 1.0):
                K = rk.RadialEwald(law, 0.7, xi, 1.7)
                np.testing.assert_allclose(K.theta_S_scaled_closed(r), K.theta_S_scaled_quad(r), rtol=1e-12)
                np.testing.assert_allclose(K.theta_S(r) + K.theta_L(r), K.theta_full(r), rtol=1e-12)
                x, w = np.polynomial.legendre.leggauss(300)
                rr, wr = 25 * (x + 1), 25 * w
                AS0 = 4 * np.pi / 3 * np.sum(wr * rr ** 2 * K.theta_S(rr))
                AL0 = K.AB_scaled(np.array([0.0]))[0][0]
                self.assertAlmostEqual((AS0 + AL0) / (K.ghat0() / 3), 1.0, places=11)
                self.assertGreater(AL0, 0.0)


class FastOperator(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(3).uniform(0, 5.5, (24, 3))

    def test_structure_and_reference_agreement(self):
        N = len(self.x)
        T = np.kron(np.ones((N, 1)), np.eye(3))
        for law in rk.LAWS:
            fast = rk.FastFriction(5.5, rk.toy_kernel(law, 0.5, 0.7, 1.3, 0.7), 4.10, 0.7, 7)
            self.assertTrue(fast.record()["k0_retained"])
            op = fast.gamma_h(self.x)
            G = op.dense()
            X = np.random.default_rng(4).normal(size=3 * N)
            np.testing.assert_allclose(op.matvec(X), G @ X, rtol=0, atol=1e-13 * np.abs(G @ X).max())
            np.testing.assert_allclose(G, G.T, atol=1e-13 * np.abs(G).max())
            np.testing.assert_allclose(G @ T, 0, atol=1e-12 * np.abs(G).max())
            ref = v2.LatticeFriction(5.5, kernel=law).matrix(self.x)
            self.assertLess(np.linalg.norm(G - ref, 2) / np.linalg.norm(ref, 2), 3e-7)
            self.assertGreater(np.linalg.eigvalsh(G)[3], 0.0)


class Enumeration(unittest.TestCase):
    """Cost-optimisation replacements must reproduce the old enumerations exactly (same pairs, same forces)."""

    def test_tree_pair_search_matches_image_enumeration(self):
        rng = np.random.default_rng(11)
        for N, L in ((40, 5.5), (120, 7.7)):
            x = rng.uniform(0, L, (N, 3))
            for rc in (0.3 * L, 0.49 * L, 0.51 * L, 1.07 * L, 1.6 * L):
                a, b = rk.real_pairs(x, L, rc), rk.real_pairs_tree(x, L, rc)
                ka, kb = (np.lexsort((*np.round(c[2], 9).T[::-1], c[1], c[0])) for c in (a, b))
                self.assertEqual(len(ka), len(kb))
                np.testing.assert_array_equal(a[0][ka], b[0][kb])
                np.testing.assert_array_equal(a[1][ka], b[1][kb])
                np.testing.assert_allclose(a[2][ka], b[2][kb], rtol=0, atol=1e-12)
        fo = rk.FastFriction(5.5, rk.toy_kernel("B", 0.5, 0.7, 1.3, 0.7), 4.10, 0.7, 7, pair_search="images")
        fn = rk.FastFriction(5.5, rk.toy_kernel("B", 0.5, 0.7, 1.3, 0.7), 4.10, 0.7, 7, pair_search="tree")
        x = rng.uniform(0, 5.5, (24, 3))
        Go, Gn = fo.gamma_h(x).dense(), fn.gamma_h(x).dense()
        np.testing.assert_allclose(Gn, Go, rtol=0, atol=1e-14 * np.abs(Go).max())

    def test_neighbour_list_force_matches_all_pairs(self):
        ck = HERE / "toy_models/v2_results/restart_checkpoints/double_well_burn140/lattice/double_well_A/seed_101/restart.npz"
        with np.load(ck) as c:
            q64 = c["Q"]
        rng = np.random.default_rng(12)
        for pot in ("lj", "double_well"):
            for q, L in ((q64, 5.5), (q64 + rng.normal(0, 0.05, q64.shape) + 5.5 * rng.integers(-2, 3, q64.shape), 5.5)):
                fa, ua, ra = v2.conservative_force(q, L, pot)
                fn, un, rn = td.conservative_force_neighbor(q, L, pot)
                np.testing.assert_allclose(fn, fa, rtol=0, atol=1e-13 * np.abs(fa).max())
                self.assertAlmostEqual(un, ua, delta=1e-12 * abs(ua))
                self.assertEqual(rn, ra)


class Dynamics(unittest.TestCase):
    def test_noise_function_uses_explicit_temperature(self):
        f = td.f_noise(0.01, 0.7, 2.0)
        lam = np.array([0.0, 1.0, 30.0])
        np.testing.assert_allclose(f(lam) ** 2, 0.7 * 2.0 * (1 - np.exp(-2 * 0.01 * lam / 2.0)), rtol=1e-14)

    def test_coupled_methods_and_momentum(self):
        ck = HERE / "toy_models/v2_results/restart_checkpoints/lj_burn140/lattice/lj_A/seed_101/restart.npz"
        with np.load(ck) as c:
            q0, p0 = c["Q"], c["p"] + np.array([0.3, -0.2, 0.1])        # nonzero total momentum is copied exactly
        out = {}
        for m in td.METHODS:
            th = td.Thermostat(m, 5.5, "A", 0.5, 0.7, 1.3, 0.005, 0.7, 1.0)
            out[m] = td.run("lj", th, q0, p0, steps=12, stride=4, seed=9)
            P = out[m]["p_final"].sum(axis=0)
            np.testing.assert_allclose(P, p0.sum(axis=0), atol=1e-11)
            meta = json.loads(str(out[m]["metadata"]))
            self.assertEqual(meta["thermostat"]["kT"], 0.7)
        self.assertLess(np.abs(out["pppm_dense"]["p_final"] - out["reference"]["p_final"]).max(), 1e-6)
        self.assertLess(np.abs(out["pppm_lanczos"]["p_final"] - out["pppm_dense"]["p_final"]).max(), 1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
