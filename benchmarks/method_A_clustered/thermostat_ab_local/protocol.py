"""Frozen thermostat experiment protocol; no historical output is written."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import scipy
from scipy import linalg
from threadpoolctl import threadpool_info

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
INPUTS = HERE / 'inputs'
BASE = '76ceb517fc24f60dfdf6d9d2b4af7a2109a07e54'
SEED = 2026102707
TOL = 1e-3
KRYLOV_TOL = 1e-5
ALPHAS = (.1, .5, 1., 2.)

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')
    temp.replace(path)

def write(name, rows):
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    temp.replace(path)

def read(name):
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v == '': r[k] = None
            elif v in ('True', 'False'): r[k] = v == 'True'
            else:
                try: r[k] = float(v)
                except ValueError: pass
    return rows

def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def norm2(A):
    A = (A + A.T) / 2
    ev = linalg.eigvalsh(A, check_finite=False)
    return float(max(abs(ev[0]), abs(ev[-1])))

def project(v):
    v = np.asarray(v).reshape(-1, 3)
    return (v - v.mean(axis=0)).ravel()

class DenseOU:
    """Independent eigensolve on Range(Pi), including exactly three COM modes."""
    def __init__(self, G):
        self.N = len(G) // 3
        self.Q = np.kron(linalg.helmert(self.N, full=False).T, np.eye(3))
        self.Pi = self.Q @ self.Q.T
        self.T = np.eye(3 * self.N) - self.Pi
        reduced = self.Q.T @ G @ self.Q
        self.eig, v = linalg.eigh((reduced + reduced.T) / 2, check_finite=False)
        self.raw_min = float(self.eig.min())
        tol = 256 * np.finfo(float).eps * max(float(np.abs(self.eig).max()), 1.)
        if self.raw_min < -tol:
            raise ValueError(f'Dense reference is materially indefinite: {self.raw_min}')
        self.eig = np.maximum(self.eig, 0.)
        self.U = self.Q @ v
        self.lmax = float(self.eig[-1])

    def matrices(self, dt):
        R = self.T + (self.U * np.exp(-dt * self.eig)) @ self.U.T
        F = (self.U * np.sqrt(-np.expm1(-2 * dt * self.eig))) @ self.U.T
        C = (self.U * (-np.expm1(-2 * dt * self.eig))) @ self.U.T
        return R, F, C

    def probes(self, rng):
        positive = np.flatnonzero(self.eig > max(self.lmax, 1.) * 1e-12)
        first = int(positive[0]) if len(positive) else 0
        random = project(rng.normal(size=3 * self.N))
        random /= np.linalg.norm(random)
        com = np.tile([1., 0., 0.], self.N) / np.sqrt(self.N)
        return dict(slow=self.U[:, first].copy(), fast=self.U[:, -1].copy(),
                    random=random, COM=com)

def ideal_a_mean(edges, long_dense, dt, substeps=1):
    """Literal ordered edge product and long dense exponential, independent of OU code."""
    N = len(long_dense) // 3
    short_step = dt / substeps
    F = np.eye(3 * N).reshape(N, 3, 3 * N)
    iterable = zip(edges.i, edges.j, edges.K) if hasattr(edges, 'i') else edges
    for i, j, K in iterable:
        e, U = linalg.eigh(K)
        if e.min() < -1e-10: raise ValueError('Indefinite edge')
        Dminus = (U * np.expm1(-short_step * np.maximum(e, 0.))) @ U.T
        delta = .5 * Dminus @ (F[int(i)] - F[int(j)])
        F[int(i)] += delta
        F[int(j)] -= delta
    F = F.reshape(3 * N, 3 * N)
    Rlong = DenseOU(long_dense).matrices(short_step)[0]
    Rone = F.T @ Rlong @ F
    return np.linalg.matrix_power(Rone, substeps)

def load_snapshot(label):
    try:
        processes = subprocess.check_output(['ps', '-Ao', 'pid,pcpu,comm', '-r'], text=True).splitlines()[:16]
    except Exception as exc:
        processes = [str(exc)]
    return dict(label=label, unix=time.time(), load=os.getloadavg(), processes=processes)

def environment():
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                machine=platform.machine(), numpy=np.__version__, scipy=scipy.__version__,
                native_pools=threadpool_info(),
                native_thread_verification='threadpoolctl sees no Accelerate pool; native count unverified' if not threadpool_info() else 'see native_pools',
                threads_requested={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')},
                base_sha=BASE, global_seed=SEED, mass=1., theta=1., rho=.75,
                dynamical_tolerance=TOL, krylov_tolerance=KRYLOV_TOL,
                timing_scope='fresh same-process interleaved matrix-free frozen steps; setup separate',
                stop_line='frozen thermostat only; no BAOAB/LJ trajectories or HPC')
