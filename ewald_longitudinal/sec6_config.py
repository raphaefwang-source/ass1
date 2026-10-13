"""
Loader for sec6_config.json, the shared configuration of the Section 6 experiments, plus provenance records.

Every Section 6 script takes its model constants, spatial parameter sets, production dt and Lanczos ranks from here,
and stamps its outputs with provenance() so that a stored result can be matched against the code version, the
parameters and this configuration before it is reused (see reusable()).
"""
import datetime
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "sec6_config.json"
RESULTS = HERE / "sec6_results"


def load():
    return json.loads(CONFIG.read_text())


RANK_KEYS = ("noise_rank", "damping_rank", "ranks_source")
RULE_KEYS = ("rank_margin_factor", "rank_robustness_tau", "rank_rule")      # selection post-processing only


def config_hash(cfg=None, include_ranks=False):
    """Hash of the sections that determine numerical results. The production ranks are excluded unless requested,
    so that the rank-selection stage, which writes them, does not invalidate its own inputs."""
    cfg = json.loads(json.dumps(cfg or load()))
    if not include_ranks:
        for k in RANK_KEYS:
            cfg["production"].pop(k, None)
        for k in RULE_KEYS:
            cfg["budgets"].pop(k, None)
    blob = json.dumps({k: cfg[k] for k in ("model", "spatial_sets", "production", "budgets")}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def model():
    return load()["model"]


def spatial(name=None):
    cfg = load()
    name = name or cfg["production"]["spatial_set"]
    p = cfg["spatial_sets"][name]
    return dict(name=name, xi=p["xi"], s=p["s"], eta=p["eta"], p=p["p"], nominal_tol=p["nominal_tol"])


def production():
    cfg = load()
    out = dict(cfg["production"])
    out["spatial"] = spatial(out["spatial_set"])
    out["fine_spatial"] = spatial(out["fine_spatial_set"])
    return out


def budgets():
    return load()["budgets"]


def box_length(N):
    return (N / model()["density"]) ** (1.0 / 3.0)


def set_ranks(noise_rank, damping_rank, source):
    """Write the selected production ranks (used only by the rank-selection stage)."""
    cfg = load()
    cfg["production"]["noise_rank"] = int(noise_rank)
    cfg["production"]["damping_rank"] = int(damping_rank)
    cfg["production"]["ranks_source"] = source
    CONFIG.write_text(json.dumps(cfg, indent=1) + "\n")


def git_state():
    try:
        rev = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        lines = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--untracked-files=no", "--", "."],
                               capture_output=True, text=True).stdout.splitlines()
        files = [ln[3:] for ln in lines if ln.strip()]
        return dict(commit=rev, dirty=bool(files), dirty_files=files[:20])
    except Exception as exc:                                     # noqa: BLE001
        return dict(error=str(exc))


CORE_FILES = ("verify_ewald_longitudinal.py", "verify_periodic_graph_psd.py", "test_parameter_selection.py",
              "test_equal_accuracy_xi.py", "test_lanczos_fdt.py", "sec6_common.py")


def code_hash(files=()):
    """sha256 over the source files that determine the numbers (core files plus the calling script)."""
    h = hashlib.sha256()
    for f in sorted(set(CORE_FILES) | set(files)):
        fp = HERE / f
        h.update(f.encode())
        h.update(fp.read_bytes() if fp.exists() else b"<missing>")
    return h.hexdigest()[:16]


def provenance(script=None, include_ranks=False, **extra):
    import scipy
    files = (script,) if script else ()
    return dict(time=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), git=git_state(),
                config_hash=config_hash(include_ranks=include_ranks), include_ranks=include_ranks,
                ranks=(load()["production"]["noise_rank"], load()["production"]["damping_rank"]), code_hash=code_hash(files), code_files=sorted(set(CORE_FILES) | set(files)),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, host=platform.node(),
                cpu=platform.processor() or platform.machine(),
                threads={v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                **extra)


def reusable(record, script=None):
    """A stored result may be reused only if it carries the current config hash and the current code hash (the
    core numerical files plus the producing script). Returns (ok, reason)."""
    prov = record.get("provenance", {})
    if prov.get("config_hash") != config_hash(include_ranks=prov.get("include_ranks", False)):
        return False, "config hash differs"
    if prov.get("code_hash") != code_hash((script,) if script else ()):
        return False, "code hash differs"
    return True, "config and code hashes match"
