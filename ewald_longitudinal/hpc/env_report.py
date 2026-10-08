#!/usr/bin/env python3
"""Print a JSON description of the compute environment (CPU, cores visible, Python, NumPy/SciPy, BLAS, threads)."""
import json
import os
import platform
import socket
import sys

import numpy as np
import scipy

info = dict(host=socket.gethostname(), python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
            platform=platform.platform(), machine=platform.machine(),
            cores_visible=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
            threads={v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            slurm={k: v for k, v in os.environ.items() if k.startswith("SLURM_") and k in (
                "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
                "SLURM_MEM_PER_CPU", "SLURM_JOB_PARTITION", "SLURM_CLUSTER_NAME", "SLURM_JOB_NODELIST")})
try:
    with open("/proc/cpuinfo") as fh:
        info["cpu_model"] = next(ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name"))
except (OSError, StopIteration):
    info["cpu_model"] = platform.processor()
try:
    cfg = np.show_config(mode="dicts")
    info["blas"] = cfg.get("Build Dependencies", {}).get("blas", {})
except Exception as exc:                      # older NumPy
    info["blas"] = f"unavailable ({exc})"
print(json.dumps(info, indent=1, default=str))
