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
