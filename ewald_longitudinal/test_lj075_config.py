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
        if tc.validated_dt(NEW) is None:
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
