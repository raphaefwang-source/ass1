#!/usr/bin/env python3
"""
Shared tools for the LJ validation at rho* = 0.75, kT = 1.0 (configuration family lj075_*).

State point (fixed for this study; no temperature or density scan):
    m = sigma = epsilon = 1, rho* = 0.75, kT = 1.0, N = 256 (L = (N/rho)^(1/3) = 6.98864...);
    smooth-switched LJ of toy_models/prl_toy_models.pair_potential (r_on = 2, r_cut = 2.5, C2 quintic switch,
    force includes the switch derivative S'(r) u0(r));
    friction kernels A and B with gamma = 0.5, kappa = 0.7, r_ref = 1.3; full periodic friction, k = 0 retained;
    damping and noise from the same Gamma_h.

Contents:
    canonical pre-equilibration (scalar-friction Langevin with the conservative forces only: positions are canonical
    whatever the friction model), RSA random start, virial pressure, RDF, small-k S(k), VACF, MSD (FFT, all time
    origins), integrated autocorrelation time / block averages, provenance helpers (git, cpu time, command).
Every function takes the state point explicitly (L, kT, m); nothing defaults to the old toy state point.
"""
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import prl_toy_models as v2  # noqa: E402
import toy_dynamics as td  # noqa: E402

OUT = HERE / "lj075_results"
RAW = OUT / "raw"                       # large per-run data (git-ignored); summaries live in OUT
STATE = dict(rho=0.75, kT=1.0, mass=1.0, epsilon=1.0, sigma=1.0, r_on=2.0, r_cut=2.5, N=256,
             gamma=0.5, kappa=0.7, r_ref=1.3)


def box_length(N=STATE["N"], rho=STATE["rho"]):
    return (N / rho) ** (1 / 3)


# ----------------------------------------------------------------------------- provenance
def provenance(extra=None):
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=HERE,
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        sha, dirty = None, None
    import hashlib
    mods = sorted(HERE.glob("lj075_*.py")) + [HERE / f for f in ("toy_configs.py", "toy_run.py", "toy_dynamics.py",
                                                                 "radial_kernels.py", "test_lanczos_fdt.py",
                                                                 "test_true_dynamics.py", "test_equal_accuracy_xi.py",
                                                                 "toy_models/prl_toy_models.py")]
    hashes = {m.name: hashlib.sha256(m.read_bytes()).hexdigest()[:16] for m in mods if m.exists()}
    try:
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "*.py"], cwd=HERE,
                                   capture_output=True, text=True).stdout.split()
    except OSError:
        untracked = None
    rec = dict(command=" ".join([sys.executable] + sys.argv), cwd=str(Path.cwd()), git_commit=sha, git_dirty=dirty,
               untracked_py=untracked, script_sha256_16=hashes,
               host=platform.node(), python=platform.python_version(), numpy=np.__version__,
               threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
               started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if extra:
        rec.update(extra)
    return rec


def cpu_seconds():
    """CPU seconds of this process and its finished children (user + system)."""
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable) + "\n")
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


# ----------------------------------------------------------------------------- initial structures
def fcc_positions(N, L, jitter, rng):
    n = round((N / 4) ** (1 / 3))
    if 4 * n ** 3 != N:
        raise ValueError(f"fcc needs N = 4 n^3, got {N}")
    g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    q = ((g[:, None, :] + basis[None]).reshape(-1, 3) + 0.25) * (L / n)
    return (q + rng.uniform(-jitter, jitter, q.shape)) % L


def rsa_positions(N, L, dmin, rng, max_tries=10 ** 6):
    """Random sequential addition with minimum pair distance dmin (periodic)."""
    q = np.empty((N, 3))
    k, tries = 0, 0
    while k < N:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("RSA failed")
        x = rng.uniform(0, L, 3)
        d = q[:k] - x
        d -= L * np.round(d / L)
        if k == 0 or np.einsum("pa,pa->p", d, d).min() >= dmin * dmin:
            q[k] = x
            k += 1
    return q


def maxwell(rng, N, kT, m):
    p = rng.normal(0.0, np.sqrt(m * kT), (N, 3))
    return p - p.mean(axis=0)


def langevin_prepare(q, p, L, kT, m, t, dt, gamma, rng, record_every=20):
    """Scalar-friction Langevin BAOAB (exact OU step) with the neighbour-list LJ forces only. Positions sample the
    canonical distribution of the conservative potential whatever the friction model; used only to prepare
    starting structures. Returns q, p and an energy trace."""
    c1 = np.exp(-gamma * dt / m)
    c2 = np.sqrt(m * kT * (1 - c1 ** 2))
    F, U, rmin = td.conservative_force_neighbor(q, L, "lj")
    trace = []
    n = int(round(t / dt))
    for s in range(n):
        p = p + 0.5 * dt * F
        q = q + 0.5 * dt / m * p
        p = c1 * p + c2 * rng.standard_normal(q.shape)
        q = q + 0.5 * dt / m * p
        F, U, rmin = td.conservative_force_neighbor(q, L, "lj")
        p = p + 0.5 * dt * F
        if rmin < 0.45 or not np.all(np.isfinite(p)):
            raise RuntimeError("Langevin preparation: close encounter")
        if s % record_every == 0:
            trace.append((s * dt, U / len(q), np.sum((p - p.mean(0)) ** 2) / (m * 3 * (len(q) - 1))))
    p = p - p.mean(axis=0)
    return q % L, p, np.array(trace)


# ----------------------------------------------------------------------------- instantaneous observables
def pair_list(q, L, rc):
    x = np.mod(q, L)
    x[x >= L] = 0.0
    pr = cKDTree(x, boxsize=L).query_pairs(rc, output_type="ndarray")
    d = q[pr[:, 0]] - q[pr[:, 1]]
    d -= L * np.round(d / L)
    return pr[:, 0], pr[:, 1], d, np.linalg.norm(d, axis=1)


def virial(q, L, r_on=2.0, r_cut=2.5):
    """W = sum_{i<j} r_ij . F_ij = -sum r du/dr (switched LJ, switch derivative included)."""
    _, _, _, r = pair_list(q, L, r_cut)
    _, du = v2.pair_potential(r, "lj", r_on=r_on, r_cut=r_cut)
    return float(-(r * du).sum())


def kinetic_temperature(p, m):
    N = len(p)
    return float(np.sum((p - p.mean(axis=0)) ** 2) / (m * 3 * (N - 1)))


def pressure(q, p, L, m, W=None):
    """P = (2 K' + W) / (3 V), K' the kinetic energy relative to the centre of mass."""
    W = virial(q, L) if W is None else W
    K = 0.5 * np.sum((p - p.mean(axis=0)) ** 2) / m
    return float((2 * K + W) / (3 * L ** 3))


# ----------------------------------------------------------------------------- structure
def rdf_counts(Q, L, edges):
    """Histogram of pair distances summed over frames Q (F, N, 3); r < edges[-1] <= L/2."""
    h = np.zeros(len(edges) - 1)
    for q in Q:
        _, _, _, r = pair_list(q, L, edges[-1])
        h += np.histogram(r, edges)[0]
    return h


def rdf_from_counts(h, n_frames, N, L, edges):
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    ideal = n_frames * N * (N - 1) / 2 / L ** 3 * shell
    return h / ideal


def kvectors_small(L, nmax=2):
    """All k = 2 pi n / L with 1 <= |n|^2 <= nmax^2 (one of each +/- pair)."""
    g = np.arange(-nmax, nmax + 1)
    n = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    n2 = (n ** 2).sum(1)
    keep = (n2 >= 1) & (n2 <= nmax ** 2)
    n = n[keep]
    first = np.array([tuple(v) > tuple(-v) for v in n])
    n = n[first]
    return n, 2 * np.pi / L * n


def structure_factor(Q, L, nmax=2):
    """S(k) per frame for the small-k shells: returns |n|^2 values and array (F, n_shells)."""
    n, k = kvectors_small(L, nmax)
    n2 = (n ** 2).sum(1)
    shells = np.unique(n2)
    rho_k = np.exp(1j * np.einsum("fia,ka->fki", Q, k)).sum(-1)          # (F, K)
    s = np.abs(rho_k) ** 2 / Q.shape[1]
    return shells, np.stack([s[:, n2 == sh].mean(1) for sh in shells], axis=1)


# ----------------------------------------------------------------------------- dynamics
def _fft_acf(x, nlag):
    """sum over trailing axes of <x(t0+l) x(t0)> averaged over all origins, l < nlag. x: (T, ...)."""
    T = x.shape[0]
    n = 1 << int(np.ceil(np.log2(2 * T)))
    F = np.fft.rfft(x.reshape(T, -1), n=n, axis=0)
    c = np.fft.irfft((F * F.conj()).sum(axis=1), n=n)[:nlag]
    return c / (T - np.arange(nlag))


def vacf(V, nlag):
    """<v_i(t) . v_i(0)> averaged over particles and origins (velocities relative to the COM)."""
    V = V - V.mean(axis=1, keepdims=True)
    return _fft_acf(V, nlag) / V.shape[1]


def msd(Q, nlag):
    """Mean-square displacement over all origins (FFT algorithm), COM motion removed. Q unwrapped (T, N, 3)."""
    Q = Q - Q.mean(axis=1, keepdims=True)
    T = Q.shape[0]
    nlag = min(nlag, T)
    C = np.concatenate([[0.0], np.cumsum((Q ** 2).sum(axis=(1, 2)))])      # C[n] = sum_{j<n} |r(j)|^2
    m = np.arange(nlag)
    S1 = (C[T] - C[m] + C[T - m]) / (T - m)       # <|r(t0+m)|^2 + |r(t0)|^2> over origins
    S2 = _fft_acf(Q, nlag)                         # <r(t0+m) . r(t0)> over origins
    return (S1 - 2 * S2) / Q.shape[1]


# ----------------------------------------------------------------------------- statistics
def integrated_act(x, c=5.0):
    """Integrated autocorrelation time tau_int (in samples, tau_int = 1/2 + sum rho) with Sokal's automatic window
    M >= c tau_int. Returns tau_int, window, statistical inefficiency g = 2 tau_int."""
    x = np.asarray(x, float) - np.mean(x)
    T = len(x)
    if T < 8 or np.allclose(x, 0):
        return 0.5, 0, 1.0
    rho = _fft_acf(x, T) / _fft_acf(x, 1)[0]
    tau = 0.5
    for M in range(1, T):
        tau = 0.5 + rho[1:M + 1].sum()
        if M >= c * tau:
            break
    return float(max(tau, 0.5)), int(M), float(2 * max(tau, 0.5))


def mean_se(x):
    """Mean and autocorrelation-corrected standard error of a single time series."""
    x = np.asarray(x, float)
    tau, _, g = integrated_act(x)
    return float(x.mean()), float(x.std(ddof=1) * np.sqrt(g / len(x))), tau


def block_se(x, n_blocks=8):
    """Standard error from n_blocks contiguous blocks (cross-check of mean_se)."""
    x = np.asarray(x, float)
    b = len(x) // n_blocks
    m = x[:b * n_blocks].reshape(n_blocks, b).mean(1)
    return float(m.std(ddof=1) / np.sqrt(n_blocks))


def t_ci(values, level=0.95):
    """Mean and two-sided t confidence interval over independent replicas."""
    from scipy import stats
    v = np.asarray(values, float)
    n = len(v)
    m = v.mean()
    if n < 2:
        return float(m), float("nan"), float("nan")
    h = stats.t.ppf(0.5 + level / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return float(m), float(m - h), float(m + h)
