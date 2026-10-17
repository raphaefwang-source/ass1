"""
Shared pieces of the Method A clustered-particle benchmark.

State point and kernel are the validated LJ + kernel A study of the manuscript (manuscript/main.tex, Sec. 5.1;
ewald_longitudinal/lj075_common.STATE; toy_configs "lj_rho0.75_kT1.0_A_prod"):
    K(r) = c exp(-kappa r)/r * r r^T/r^2,  c = gamma r_ref exp(kappa r_ref),  gamma = 0.5, kappa = 0.7, r_ref = 1.3
    density rho = 0.75, full periodic lattice sum, k = 0 retained, no minimum image.
Method A is the existing Ewald split at s = xi^2 (radial_kernels.RadialEwald, law A) with the existing PPPM
(test_parameter_selection.Mesh: B-spline order p spread, FFT, influence Khat_L/|W_hat|^2 on |k| <= kc, IFFT, adjoint
gather). Nothing in the kernel, the split or the solver is changed here.
Reference: v2.LatticeFriction (near images summed directly, far images by a tensor Chebyshev fit), the model matrix
used by lj075_verify.py.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EWALD = REPO / "ewald_longitudinal"
for _p in (EWALD, EWALD / "toy_models", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import prl_toy_models as v2  # noqa: E402
import radial_kernels as rk  # noqa: E402
import toy_configs as tc  # noqa: E402
from lj075_common import STATE  # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CONFIGS = HERE / "configs"
RAW = HERE / "raw"                     # dense matrices and other large caches (git-ignored)

LAW = "A"
GAMMA, KAPPA, R_REF = STATE["gamma"], STATE["kappa"], STATE["r_ref"]
DENSITY = STATE["rho"]
AMP = rk.amplitude(LAW, GAMMA, KAPPA, R_REF)
D_MIN = 0.9                            # RSA d_min of the lj075 canonical starts (manuscript Table 1)
EPS_TARGET = 2e-7                      # relative spectral operator budget of law A (manuscript Sec. 3.5)
P_ORDER = 7                            # assignment order fixed for the theory-guided selection
BASE_CONFIG = "lj_rho0.75_kT1.0_A_prod"
BASELINE = dict(tc.CONFIGS[BASE_CONFIG]["pppm"][LAW])          # xi 0.85, s 4.1, eta 0.6779, p 8


def box_length(N):
    return (N / DENSITY) ** (1.0 / 3.0)


_KERN = {}


def kernel(xi):
    """Law-A kernel object (theta_S, AB_scaled with amplitude) for Ewald parameter xi."""
    key = round(float(xi), 12)
    if key not in _KERN:
        _KERN[key] = rk.toy_kernel(LAW, GAMMA, KAPPA, R_REF, float(xi))
    return _KERN[key]


def lattice(L):
    return v2.LatticeFriction(L, kernel=LAW, gamma=GAMMA, kappa=KAPPA, r_ref=R_REF)


def lattice_matrix(lat, q, chunk=8192):
    """lat.matrix(q) with the Chebyshev far field evaluated in chunks (as lj075_verify.lattice_matrix)."""
    i, j, dr, _ = v2.pair_geometry(q, lat.box)
    K = np.concatenate([lat.pair_tensors(dr[c:c + chunk]) for c in range(0, len(dr), chunk)])
    return v2.assemble_friction(i, j, K, len(q))


def reference(q, L, rel_tol=None):
    """Dense full-periodic Gamma_ref (3N x 3N) and the LatticeFriction object."""
    lat = lattice(L) if rel_tol is None else v2.LatticeFriction(L, kernel=LAW, gamma=GAMMA, kappa=KAPPA,
                                                                 r_ref=R_REF, rel_tol=rel_tol)
    return lattice_matrix(lat, np.mod(q, L)), lat


def zero_mode_matrix(lat, N):
    c0 = lat.zero_mode
    return c0 * np.kron(N * np.eye(N) - np.ones((N, N)), np.eye(3))


# ----------------------------------------------------------------------------- timing
def median_time(fn, reps, warm=3):
    for _ in range(warm):
        fn()
    t = np.empty(reps)
    for r in range(reps):
        t0 = time.perf_counter()
        fn()
        t[r] = time.perf_counter() - t0
    return t


def tstats(t):
    t = np.asarray(t, float)
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    return dict(median=float(med), q1=float(q1), q3=float(q3), min=float(t.min()), max=float(t.max()),
                n=int(t.size), iqr_rel=float((q3 - q1) / med) if med > 0 else float("nan"))


# ----------------------------------------------------------------------------- provenance / io
def git_state():
    try:
        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout
        return dict(commit=rev, dirty=bool(st.strip()))
    except OSError as exc:
        return dict(error=str(exc))


def code_hash():
    h = hashlib.sha256()
    for f in sorted(HERE.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    for f in ("radial_kernels.py", "test_lanczos_fdt.py", "test_parameter_selection.py", "test_equal_accuracy_xi.py",
              "verify_ewald_longitudinal.py", "toy_models/prl_toy_models.py"):
        h.update(f.encode())
        h.update((EWALD / f).read_bytes())
    return h.hexdigest()[:16]


def provenance(**extra):
    import scipy
    try:
        cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "")
    except OSError:
        cpu = platform.processor()
    return dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), command=" ".join([Path(sys.executable).name] + sys.argv),
                git=git_state(), code_hash=code_hash(), python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, host=platform.node(), cpu=cpu, n_cpu=os.cpu_count(),
                threads={v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                **extra)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable) + "\n")
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_csv(path, rows):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def log_command():
    """Append the current command line to results/commands.log (the record of every executed command)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "commands.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}  cd {HERE.relative_to(REPO)} && "
                 f"{' '.join(['python3'] + sys.argv)}\n")


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()
