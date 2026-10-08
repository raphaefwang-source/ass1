#!/usr/bin/env python3
"""Checks of the production runner toy_run.py (about 1-2 min, N = 64 / 256 / 512, a few dozen steps).

    python3 test_toy_run.py

- configuration resolution: explicit sets, N-dependent ranks, refusal outside the verified range;
- the runner reproduces toy_dynamics.run bitwise: new configuration (tree pairs, neighbour forces) and the original
  path (image pairs, all-pair forces);
- continuous run == the same run split into segments and resumed (bitwise q, p, diagnostics except wall time, frames),
  including a stale chunk left by a kill between chunk and checkpoint writes;
- a signal stops the run cleanly (exit 75); resuming completes it with the same result;
- overwrite protection; resuming a complete run is a no-op; resuming with different arguments is refused;
- initial-state records (replicate64 / fcc are marked not equilibrated).
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import toy_configs as tc  # noqa: E402
import toy_dynamics as td  # noqa: E402
import toy_run as tr  # noqa: E402

CK = HERE / "toy_models/v2_results/restart_checkpoints/double_well_burn140/lattice/double_well_{law}/seed_101/restart.npz"


def run(*argv):
    return tr.main([*map(str, argv), "--quiet"])


def same(a, b):
    """Bitwise equality of two loaded runs except the wall-time column."""
    keep = [i for i, c in enumerate(tr.DIAG_COLS) if c != "step_wall_s"]
    return (np.array_equal(a["diag"][:, keep], b["diag"][:, keep], equal_nan=True)      # step 0 has no Ritz values
            and np.array_equal(a["frame_step"], b["frame_step"]) and np.array_equal(a["Q"], b["Q"])
            and np.array_equal(a["V"], b["V"]) and np.array_equal(a["monitor"], b["monitor"], equal_nan=True))


class Config(unittest.TestCase):
    def test_resolution_and_refusals(self):
        r = tc.resolve("costopt_hiacc", "A", 512, "double_well")
        self.assertEqual((r["rank_noise"], r["rank_damp"], r["pair_search"], r["force_method"]), (40, 5, "tree", "neighbor"))
        self.assertEqual((tc.resolve("costopt_hiacc", "B", 256, "lj")["rank_noise"],
                          tc.resolve("costopt_hiacc", "B", 512, "lj")["rank_damp"]), (16, 8))
        r = tc.resolve("costopt_hiacc", "A", 128, "lj")                      # between verified sizes
        self.assertFalse(r["rank_provenance"]["verified_at_this_N"])
        self.assertTrue(r["warnings"])
        b = tc.resolve("baseline_hiacc", "B", 512, "lj")
        self.assertEqual((b["rank_noise"], b["rank_damp"], b["pair_search"], b["force_method"]), (40, 16, "images", "allpairs"))
        for kw in (dict(N=1728), dict(N=4096), dict(N=512, dt=0.004), dict(N=512, rank_override=(8, 2))):
            N = kw.pop("N")
            with self.assertRaises(tc.ConfigError):
                tc.resolve("costopt_hiacc", "A", N, "lj", **kw)


class MatchesToyDynamics(unittest.TestCase):
    def test_bitwise_against_toy_dynamics_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, law in (("costopt_hiacc", "B"), ("baseline_hiacc", "A")):
                out = Path(tmp) / name
                self.assertEqual(run("--config", name, "--potential", "double_well", "--kernel", law, "--N", 64,
                                     "--seed", 5, "--init-source-seed", 101, "--production-steps", 12, "--save-every-steps", 4, "--out", out), 0)
                R = tr.load_run(out)
                r = tc.resolve(name, law, 64, "double_well")
                m = r["model"]
                th = td.Thermostat("pppm_lanczos", r["L"], law, m["gamma"], m["kappa"], m["r_ref"], r["dt"], m["kT"],
                                   m["mass"], pppm=r["pppm"], rank_noise=r["rank_noise"], rank_damp=r["rank_damp"],
                                   pair_search=r["pair_search"])
                with np.load(str(CK).format(law=law)) as c:
                    q0, p0 = c["Q"], c["p"]
                ref = td.run("double_well", th, q0, p0, steps=12, stride=4, seed=[5, 1], force_method=r["force_method"])
                with np.load(out / "checkpoint.npz") as c:
                    self.assertTrue(np.array_equal(c["q"], ref["q_final"]) and np.array_equal(c["p"], ref["p_final"]), name)
                self.assertTrue(np.array_equal(R["Q"], ref["Q"]) and np.array_equal(R["V"], ref["V"]), name)
                np.testing.assert_array_equal(R["diag"][::4, 1:7], ref["diagnostics"])    # t, T, U/N, K/N, |P|, rmin


class Restart(unittest.TestCase):
    def test_split_resume_equals_continuous(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = ["--config", "costopt_hiacc", "--potential", "double_well", "--kernel", "A", "--N", 64, "--seed", 9, "--init-source-seed", 101,
                    "--burn-in-steps", 10, "--production-steps", 30, "--save-every-steps", 3, "--checkpoint-every-steps", 7]
            a, b = Path(tmp) / "continuous", Path(tmp) / "split"
            self.assertEqual(run(*base, "--out", a), 0)
            self.assertEqual(run(*base, "--out", b, "--max-segment-steps", 13), tr.EXIT_INCOMPLETE)
            st = json.loads((b / "status.json").read_text())
            self.assertEqual((st["state"], st["step"]), ("incomplete", 13))
            # a kill between chunk and checkpoint writes leaves a chunk past the checkpoint
            shutil.copy(b / "chunks" / "chunk_000000013.npz", b / "chunks" / "chunk_000000020.npz")
            self.assertEqual(run("--resume", "--out", b, "--max-segment-steps", 11), tr.EXIT_INCOMPLETE)
            self.assertTrue(list((b / "chunks" / "stale").rglob("chunk_000000020.npz")))
            self.assertEqual(run("--resume", "--out", b), 0)
            A, B = tr.load_run(a), tr.load_run(b)
            self.assertTrue(same(A, B))
            self.assertEqual(len(A["diag"]), 41)
            np.testing.assert_array_equal(A["diag"][:, 0], np.arange(41))
            for f in ("checkpoint.npz", "checkpoint_burnin_end.npz"):
                with np.load(a / f) as x, np.load(b / f) as y:
                    self.assertTrue(np.array_equal(x["q"], y["q"]) and np.array_equal(x["p"], y["p"])
                                    and str(x["rng_state"]) == str(y["rng_state"]), f)
            self.assertEqual(len(json.loads((b / "status.json").read_text())["segments"]), 3)

    def test_signal_stop_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = ["--config", "costopt_hiacc", "--potential", "lj", "--kernel", "B", "--N", 64, "--seed", 3, "--init-source-seed", 101,
                    "--production-steps", 400, "--save-every-steps", 10, "--checkpoint-every-steps", 50]
            a, b = Path(tmp) / "continuous", Path(tmp) / "signalled"
            self.assertEqual(run(*base, "--out", a), 0)
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            proc = subprocess.Popen([sys.executable, str(HERE / "toy_run.py"), *map(str, base), "--out", str(b), "--quiet"],
                                    env=env)
            t0 = time.time()
            while not list((b / "chunks").glob("chunk_00000005*.npz")) and time.time() - t0 < 120:
                time.sleep(0.05)
            proc.send_signal(signal.SIGUSR1)
            self.assertEqual(proc.wait(timeout=120), tr.EXIT_INCOMPLETE)
            st = json.loads((b / "status.json").read_text())
            self.assertEqual(st["state"], "incomplete")
            self.assertIn("SIGUSR1", st["segments"][-1]["stop_reason"])
            self.assertLess(st["step"], 400)
            self.assertEqual(run("--resume", "--out", b), 0)
            self.assertTrue(same(tr.load_run(a), tr.load_run(b)))


class Protection(unittest.TestCase):
    def test_overwrite_and_resume_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r"
            base = ["--config", "costopt_hiacc", "--potential", "double_well", "--kernel", "B", "--N", 64, "--seed", 1, "--init-source-seed", 101,
                    "--production-steps", 5, "--out", out]
            self.assertEqual(run(*base), 0)
            snap = {f: f.stat().st_mtime_ns for f in out.rglob("*") if f.is_file()}
            self.assertEqual(run(*base), tr.EXIT_USAGE)                         # no --resume: refuse
            self.assertEqual(run("--resume", "--out", out), 0)                  # complete: no-op
            self.assertEqual(run(*base, "--resume"), 0)                         # same arguments: no-op
            self.assertEqual(run("--resume", "--out", out, "--N", 512), tr.EXIT_USAGE)
            self.assertEqual(run("--resume", "--out", out, "--seed", 2), tr.EXIT_USAGE)
            self.assertEqual(snap, {f: f.stat().st_mtime_ns for f in out.rglob("*") if f.is_file()})
            (Path(tmp) / "x").mkdir()
            (Path(tmp) / "x" / "junk").write_text("x")
            self.assertEqual(run(*base[:-1], Path(tmp) / "x"), tr.EXIT_USAGE)   # non-empty foreign directory


class InitialStates(unittest.TestCase):
    def test_constructed_states_are_recorded_and_not_equilibrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            for N, method, extra in ((512, "replicate64", {"copies": 8}), (256, "fcc", {})):
                out = Path(tmp) / method
                self.assertEqual(run("--config", "costopt_hiacc", "--potential", "double_well", "--kernel", "B", "--N", N,
                                     "--seed", 101, "--production-steps", 1, "--out", out), 0)
                cfg = json.loads((out / "config.json").read_text())
                self.assertEqual(cfg["init"]["method"], method)
                self.assertFalse(cfg["init"]["equilibrated"])
                self.assertIn("NOT equilibrated", cfg["init"]["note"])
                for k, v in extra.items():
                    self.assertEqual(cfg["init"][k], v)
                with np.load(out / "init_state.npz") as d:
                    self.assertEqual(d["q"].shape, (N, 3))
                    self.assertLess(np.abs(d["p"].sum(axis=0)).max(), 1e-10)
                self.assertGreater(tr.load_run(out)["diag"][:, 6].min(), 0.6)        # stop threshold is 0.45
                self.assertTrue(cfg["verification"]["verified"], cfg["verification"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
