#!/usr/bin/env python3
"""Restart consistency on this machine: a continuous run against the same run stopped and resumed in separate
processes. Positions, momenta, RNG state, diagnostics (all but the wall-time column), frames and accuracy-monitor
rows must agree bitwise.

    python3 hpc/restart_check.py --work DIR [--N 64 --kernel A --steps 60 --segments 25 20] [--signal]

--segments: lengths of the stopped segments (--max-segment-steps); the last resume runs to the end.
--signal: stop the first segment with SIGTERM after its first checkpoint instead of a step limit.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import toy_run as tr  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--config", default="costopt_hiacc")
    ap.add_argument("--potential", default="double_well")
    ap.add_argument("--kernel", default="A")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--burn-in-steps", type=int, default=10)
    ap.add_argument("--segments", type=int, nargs="+", default=[25, 20])
    ap.add_argument("--signal", action="store_true")
    args = ap.parse_args()
    work = Path(args.work)
    if work.exists() and any(work.iterdir()):
        sys.exit(f"{work} is not empty")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    base = [sys.executable, str(HERE / "toy_run.py"), "--config", args.config, "--potential", args.potential,
            "--kernel", args.kernel, "--N", str(args.N), "--seed", str(args.seed), "--burn-in-steps",
            str(args.burn_in_steps), "--production-steps", str(args.steps - args.burn_in_steps),
            "--save-every-steps", "5", "--checkpoint-every-steps", "8", "--monitor-every-steps", "10", "--quiet"]
    a, b = work / "continuous", work / "resumed"
    codes = [subprocess.run(base + ["--out", str(a)], env=env).returncode]
    if args.signal:
        proc = subprocess.Popen(base + ["--out", str(b)], env=env)
        t0 = time.time()
        while not list((b / "chunks").glob("chunk_00000000[1-9]*.npz")) and time.time() - t0 < 600:
            time.sleep(0.05)
        proc.send_signal(signal.SIGTERM)
        codes.append(proc.wait())
    else:
        for i, k in enumerate(args.segments):
            codes.append(subprocess.run(base + ["--out", str(b), "--max-segment-steps", str(k)]
                                        + (["--resume"] if i else []), env=env).returncode)
    codes.append(subprocess.run(base + ["--out", str(b), "--resume"], env=env).returncode)
    A, B = tr.load_run(a), tr.load_run(b)
    keep = [i for i, c in enumerate(tr.DIAG_COLS) if c != "step_wall_s"]
    res = dict(exit_codes=codes, expected="continuous 0, stopped segments 75, final resume 0",
               segments=len(B["status"].get("segments", [])))
    with np.load(a / "checkpoint.npz") as x, np.load(b / "checkpoint.npz") as y:
        res["q_p_bitwise"] = bool(np.array_equal(x["q"], y["q"]) and np.array_equal(x["p"], y["p"]))
        res["rng_state_equal"] = str(x["rng_state"]) == str(y["rng_state"])
    res["diag_bitwise_except_wall"] = bool(np.array_equal(A["diag"][:, keep], B["diag"][:, keep], equal_nan=True))
    res["frames_bitwise"] = bool(np.array_equal(A["Q"], B["Q"]) and np.array_equal(A["V"], B["V"])
                                 and np.array_equal(A["frame_step"], B["frame_step"]))
    res["monitor_bitwise"] = bool(np.array_equal(A["monitor"], B["monitor"], equal_nan=True))
    res["steps"] = int(A["diag"][-1, 0])
    res["passed"] = (codes[0] == 0 and codes[-1] == 0 and all(c == tr.EXIT_INCOMPLETE for c in codes[1:-1])
                     and all(res[k] for k in ("q_p_bitwise", "rng_state_equal", "diag_bitwise_except_wall",
                                              "frames_bitwise", "monitor_bitwise")))
    (work / "restart_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    print("RESTART CHECK", "PASS" if res["passed"] else "FAIL")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
