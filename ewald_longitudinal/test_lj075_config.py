#!/usr/bin/env python3
"""
Regression tests for the per-configuration state point (toy_configs.STATE_POINTS) and the lj075 configuration.

    python3 test_lj075_config.py
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toy_configs as tc  # noqa: E402

NEW = "lj_rho0.75_kT1.0_costopt"
# operator_hash values of the configurations that existed before the state-point change (must never change)
OLD_HASHES = {("costopt_hiacc", "A", 64): "6767bb4ee9b295c9", ("costopt_hiacc", "A", 256): "71b4e02fad3e21e3",
              ("costopt_hiacc", "A", 512): "576e1743b9bc5600", ("costopt_hiacc", "B", 64): "a6e6767daa7d3d2a",
              ("costopt_hiacc", "B", 256): "c20d60ff948161d2", ("costopt_hiacc", "B", 512): "2fd68258a2bc0a0f",
              ("baseline_hiacc", "A", 64): "7b8ab5b20cf73932", ("baseline_hiacc", "A", 256): "d997a193299046ee",
              ("baseline_hiacc", "A", 512): "8182d14b16344c95", ("baseline_hiacc", "B", 64): "771e10d21cc4d407",
              ("baseline_hiacc", "B", 256): "569356ca49f4d868", ("baseline_hiacc", "B", 512): "bcf3708e280c1918"}


def runner(*argv):
    return subprocess.run([sys.executable, str(HERE / "toy_run.py"), *argv], capture_output=True, text=True,
                          cwd=HERE, timeout=600)


class Config(unittest.TestCase):
    def test_old_hashes_unchanged(self):
        for (name, law, N), h in OLD_HASHES.items():
            r = tc.resolve(name, law, N, "double_well")
            self.assertEqual(tc.operator_hash(r), h, (name, law, N))
            self.assertAlmostEqual(r["model"]["kT"], 0.7)
            self.assertAlmostEqual(r["L"], (N / (64 / 5.5 ** 3)) ** (1 / 3))
            self.assertEqual(r["dt"], 0.005)

    def test_new_state_point(self):
        r = tc.resolve(NEW, "A", 256, "lj", dt=0.005, allow_unverified=True)
        self.assertEqual(r["model"]["kT"], 1.0)
        self.assertEqual(r["model"]["density"], 0.75)
        self.assertAlmostEqual(r["L"], (256 / 0.75) ** (1 / 3), places=12)
        self.assertNotIn(tc.operator_hash(r), set(OLD_HASHES.values()))
        for k in ("gamma", "kappa", "r_ref", "mass", "epsilon", "sigma", "r_on", "r_cut"):
            self.assertEqual(r["model"][k], tc.MODEL[k])

    def test_new_config_refusals(self):
        with self.assertRaises(tc.ConfigError):
            tc.resolve(NEW, "A", 512, "lj", dt=0.005, allow_unverified=True)
        with self.assertRaises(tc.ConfigError):
            tc.resolve(NEW, "A", 256, "double_well", dt=0.005, allow_unverified=True)
        if tc.validated_dt(NEW, "A") is None:
            with self.assertRaises(tc.ConfigError):
                tc.resolve(NEW, "A", 256, "lj")
        with self.assertRaises(tc.ConfigError):
            tc.resolve(NEW, "A", 256, "lj", dt=0.0123)

    def test_runner_refuses_old_states_and_records_new(self):
        with tempfile.TemporaryDirectory() as d:
            base = ["--config", NEW, "--potential", "lj", "--kernel", "B", "--N", "256", "--seed", "3",
                    "--production-steps", "3", "--allow-unverified", "--dt", "0.005", "--monitor-every-steps", "0"]
            for init in ("v2_checkpoint", "replicate64"):
                res = runner(*base, "--init", init, "--out", f"{d}/{init}")
                self.assertEqual(res.returncode, 1, res.stderr)
                self.assertIn("toy_v2 state", res.stderr)
            np.savez(f"{d}/old.npz", q=np.zeros((256, 3)), p=np.zeros((256, 3)), L=8.73071, kT=0.7)
            res = runner(*base, "--from-state", f"{d}/old.npz", "--out", f"{d}/fs")
            self.assertEqual(res.returncode, 1)
            self.assertIn("this configuration has L", res.stderr)
            np.savez(f"{d}/nometa.npz", q=np.zeros((256, 3)), p=np.zeros((256, 3)))
            res = runner(*base, "--from-state", f"{d}/nometa.npz", "--out", f"{d}/fs2")
            self.assertEqual(res.returncode, 1)
            L = (256 / 0.75) ** (1 / 3)
            np.savez(f"{d}/wrongkT.npz", q=np.zeros((256, 3)), p=np.zeros((256, 3)), L=L, kT=0.7)
            res = runner(*base, "--from-state", f"{d}/wrongkT.npz", "--out", f"{d}/fs3")
            self.assertEqual(res.returncode, 1)
            self.assertIn("kT 0.7", res.stderr)
            os.makedirs(f"{d}/conf")
            Path(f"{d}/conf/config.json").write_text(json.dumps(dict(resolved=dict(
                L=L, N=256, config=NEW, model=dict(kT=1.0, density=0.75)))))
            np.savez(f"{d}/conf/conflict.npz", q=np.zeros((256, 3)), p=np.zeros((256, 3)), L=8.73071, kT=0.7)
            res = runner(*base, "--from-state", f"{d}/conf/conflict.npz", "--out", f"{d}/fs4")
            self.assertEqual(res.returncode, 1)
            self.assertIn("file keys", res.stderr)
            np.savez(f"{d}/badkey.npz", q=np.zeros((256, 3)), p=np.zeros((256, 3)), L=np.array([L]), kT=1.0)
            res = runner(*base, "--from-state", f"{d}/badkey.npz", "--out", f"{d}/fs5")
            self.assertEqual(res.returncode, 1)
            self.assertIn("not a scalar", res.stderr)
            res = runner(*base, "--out", f"{d}/auto")                           # auto init: fcc at the new L
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(json.loads(Path(f"{d}/auto/config.json").read_text())["init"]["method"], "fcc")
            res = runner(*base, "--init", "rsa", "--out", f"{d}/ok")
            self.assertEqual(res.returncode, 0, res.stderr)
            cfg = json.loads(Path(f"{d}/ok/config.json").read_text())
            self.assertEqual(cfg["state_point"], "lj_rho0.75_kT1.0")
            self.assertEqual(cfg["resolved"]["model"]["kT"], 1.0)
            self.assertAlmostEqual(cfg["resolved"]["L"], (256 / 0.75) ** (1 / 3), places=12)
            self.assertEqual(cfg["init"]["method"], "rsa")
            self.assertIn("--config", cfg["command"])
            st = json.loads(Path(f"{d}/ok/status.json").read_text())
            self.assertGreater(st["segments"][-1]["cpu_s"], 0)
            with np.load(f"{d}/ok/init_state.npz") as z:
                q = z["q"]
            dd = q[:, None] - q[None]
            L = cfg["resolved"]["L"]
            dd -= L * np.round(dd / L)
            r = np.linalg.norm(dd, axis=-1)
            np.fill_diagonal(r, np.inf)
            self.assertGreaterEqual(r.min(), 0.9 - 1e-12)


class Physics(unittest.TestCase):
    def test_lj_force_finite_difference_inside_switch(self):
        """Pair distances inside [r_on, r_cut] = [2, 2.5]: the force must include S'(r) u0(r)."""
        sys.path.insert(0, str(HERE / "toy_models"))
        import toy_dynamics as td
        L = (256 / 0.75) ** (1 / 3)
        rng = np.random.default_rng(4)
        q = np.array([[1.0, 1.0, 1.0], [3.1, 1.2, 1.0], [1.3, 3.25, 1.4], [4.0, 3.6, 3.2]])
        F, U, _ = td.conservative_force_neighbor(q, L, "lj")
        d = q[:, None] - q[None]
        r = np.linalg.norm(d - L * np.round(d / L), axis=-1)[np.triu_indices(4, 1)]
        self.assertTrue(np.any((r > 2.0) & (r < 2.5)), r)
        h = 1e-6
        num = np.zeros_like(q)
        for i in range(4):
            for c in range(3):
                qp, qm = q.copy(), q.copy()
                qp[i, c] += h
                qm[i, c] -= h
                num[i, c] = -(td.conservative_force_neighbor(qp, L, "lj")[1]
                              - td.conservative_force_neighbor(qm, L, "lj")[1]) / (2 * h)
        self.assertLess(np.abs(F - num).max(), 1e-7 * max(1.0, np.abs(F).max()))
        _ = rng

    def test_kT_propagates_to_noise_and_coupling(self):
        """kT = 1 would hide a beta = 1 bug: check E|S z|^2 and the coupled O-step at kT = 1 and 2."""
        sys.path.insert(0, str(HERE / "toy_models"))
        import toy_dynamics as td
        import lj075_common as C
        import lj075_coupled as LC
        from test_lanczos_fdt import DenseRef, proj
        out = {}
        for kT in (1.0, 2.0):
            r = tc.resolve(NEW, "B", 256, "lj", dt=0.01, allow_unverified=True)
            r["model"] = dict(r["model"], kT=kT)
            q = C.fcc_positions(256, r["L"], 0.1, np.random.default_rng(1))
            th = td.Thermostat("pppm_lanczos", r["L"], "B", 0.5, 0.7, 1.3, 0.01, kT, 1.0, pppm=r["pppm"],
                               rank_noise=r["rank_noise"], rank_damp=r["rank_damp"], pair_search="tree")
            D = DenseRef(th.fast.gamma_h(q).dense(), 256)
            z = proj(np.random.default_rng(2).standard_normal(768))
            p_new, _ = th.o_step(q, np.zeros((256, 3)), z.reshape(256, 3))
            ref = D.apply(lambda l: np.sqrt(kT * -np.expm1(-0.02 * l)), z)
            self.assertLess(np.linalg.norm(p_new.ravel() - ref) / np.linalg.norm(ref), 1e-8)
            out[kT] = np.linalg.norm(p_new)
            st = LC.selftest(r, q)
            self.assertLess(st["dense_two_fine_vs_one_coarse"], 1e-13)
            self.assertLess(st["lanczos_two_fine_vs_one_coarse"], 1e-9)
        self.assertAlmostEqual(out[2.0] / out[1.0], np.sqrt(2.0), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
