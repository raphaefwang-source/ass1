"""Compiled sequential pair OU sweeps with immutable graph/coefficients.

Build artifacts live only under this new experiment directory's .build/.
The original EdgeOU remains available via backend='python'; compilation failure
also returns that explicit fallback and preserves its reason in compile_info.
No OpenMP, edge coloring, edge permutation, or fast-math transformations are used.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0,str(PARENT))
from ou_core import EdgeOU, _parameters, _vector


FLAGS = ('-O3','-std=c11','-fPIC','-fno-fast-math','-ffp-contract=off')
_LOADED = {}
_COMPILER_VERSIONS = {}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _immutable(array, dtype=None):
    """A bytes-backed array cannot have its writeable flag re-enabled."""
    source = np.ascontiguousarray(array,dtype=dtype)
    return np.frombuffer(source.tobytes(),dtype=source.dtype).reshape(source.shape)


def build_library(build_dir=None, compiler=None, force=False):
    """Build/load the scalar C kernel and return (CDLL, complete provenance)."""
    cache = Path(build_dir).resolve() if build_dir is not None else HERE/'.build'
    if not cache.is_relative_to(HERE):
        raise ValueError('compiled artifacts must remain inside optimized_pairou/')
    cache.mkdir(parents=True,exist_ok=True)
    compiler = compiler or shutil.which('clang')
    if not compiler:
        raise RuntimeError('clang is unavailable')
    compiler = str(Path(compiler).resolve())
    compiler_stat = Path(compiler).stat()
    compiler_key = (compiler,compiler_stat.st_mtime_ns,compiler_stat.st_size)
    if compiler_key not in _COMPILER_VERSIONS:
        version_call = subprocess.run([compiler,'--version'],capture_output=True,text=True,check=True,timeout=15)
        _COMPILER_VERSIONS[compiler_key] = version_call.stdout.strip()
    version = _COMPILER_VERSIONS[compiler_key]
    source = HERE/'edge_sweep.c'
    source_sha = _sha(source)
    link_flag = '-dynamiclib' if sys.platform=='darwin' else '-shared'
    signature = dict(source_sha256=source_sha,compiler=compiler,compiler_version=version,
                     flags=list(FLAGS)+( [link_flag] ),platform=platform.platform(),machine=platform.machine())
    key = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:24]
    suffix = '.dylib' if sys.platform=='darwin' else '.so'
    library = cache/f'edge_sweep_{key}{suffix}'
    sidecar = cache/f'edge_sweep_{key}.json'
    compiled_now = force or not library.exists() or not sidecar.exists()
    t_compile = 0.
    if compiled_now:
        descriptor,temporary = tempfile.mkstemp(prefix='compile_',suffix=suffix,dir=cache)
        os.close(descriptor)
        command = [compiler,*FLAGS,link_flag,str(source),'-o',temporary]
        t = perf_counter()
        try:
            completed = subprocess.run(command,capture_output=True,text=True,check=True,timeout=120)
            t_compile = perf_counter()-t
            os.replace(temporary,library)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
        provenance = dict(signature,compiler_command=command,library_path=str(library),
                          library_sha256=_sha(library),original_cold_compile_s=t_compile,
                          compiler_stdout=completed.stdout,compiler_stderr=completed.stderr)
        sidecar.write_text(json.dumps(provenance,indent=2)+'\n')
    else:
        provenance = json.loads(sidecar.read_text())
        if provenance['library_sha256'] != _sha(library):
            raise RuntimeError('compiled library hash differs from its provenance')
    t = perf_counter()
    # Retain loaded objects for ctypes function-pointer lifetime and avoid reloads.
    load_key = str(library)
    if load_key not in _LOADED:
        lib = ctypes.CDLL(str(library))
        lib.pairou_sweep.argtypes = [ctypes.c_int64,ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_int]
        lib.pairou_sweep.restype = ctypes.c_int
        _LOADED[load_key] = lib
    return _LOADED[load_key],dict(provenance,compiled_now=compiled_now,
                                 t_compile_s=t_compile,t_load_s=perf_counter()-t,
                                 fast_math=False,fp_contract=False,edge_parallelism=False)


def compile_backend(build_dir=None,force_rebuild=False):
    """Explicit cold/warm compiler entry point; return JSON-ready provenance."""
    _,info = build_library(build_dir=build_dir,force=force_rebuild)
    return info


class CompiledEdgeOU(EdgeOU):
    """Compatible EdgeOU subclass; graph changes require a new instance.

    .prepare(tau,mass,theta) caches the full 3x3 D and S increment matrices.
    Every key includes all three parameters.  Internal graph and eigensystem
    arrays are genuinely immutable, not merely numpy arrays with a resettable
    write-protection flag.  .rebuild(..., configuration_token=...) creates a
    fresh instance, recomputes eigensystems, and discards coefficient caches.
    """
    _PROTECTED = frozenset(('i','j','K','U','eigenvalues','n_particles','configuration_token'))

    def __setattr__(self,name,value):
        if getattr(self,'_locked',False) and name in self._PROTECTED:
            raise AttributeError(f'{name} is immutable; rebuild for a changed graph/configuration')
        super().__setattr__(name,value)

    def __init__(self,edges,j=None,K=None,n_particles=None,*,configuration_token=None,
                 backend='compiled',build_dir=None,compiler=None):
        if backend not in ('compiled','python'):
            raise ValueError("backend must be 'compiled' or 'python'")
        if n_particles is not None and (isinstance(n_particles,bool) or int(n_particles)!=n_particles or n_particles<1):
            raise ValueError('n_particles must be a positive integer')
        if j is None and K is None:
            records = list(edges)
            endpoints = [e[:2] for e in records]
            edges = records
        else:
            endpoints = [edges,j]
        try:
            endpoints_array = np.asarray(endpoints,dtype=float)
        except (ValueError,TypeError) as exc:
            raise ValueError('edge indices must be integers') from exc
        if not np.isfinite(endpoints_array).all() or np.any(endpoints_array != np.floor(endpoints_array)):
            raise ValueError('edge indices must be finite integers')
        super().__init__(edges,j,K,n_particles=n_particles)
        for name in ('i','j','K','U','eigenvalues'):
            setattr(self,name,_immutable(getattr(self,name)))
        self.configuration_token = None if configuration_token is None else str(configuration_token)
        self.graph_sha256 = hashlib.sha256(self.i.tobytes()+self.j.tobytes()+self.K.tobytes()).hexdigest()
        self._compiled_factors = {}
        self._build_dir,self._compiler,self._requested_backend = build_dir,compiler,backend
        self.backend = backend
        self._library = None
        self.compile_info = dict(requested_backend=backend,compiled_now=False,t_compile_s=0.,t_load_s=0.)
        if backend=='compiled':
            try:
                self._library,info = build_library(build_dir,compiler)
                self.compile_info.update(info)
            except (OSError,subprocess.SubprocessError,RuntimeError,ValueError) as exc:
                self.backend = 'python'
                self.compile_info['fallback_reason'] = str(exc)
        self.compile_info['active_backend'] = self.backend
        self._locked = True

    @property
    def coefficient_cache_size(self):
        return len(self._compiled_factors) if self.backend=='compiled' else len(self._factors)

    def prepare(self,tau,mass=1.,theta=1.):
        start = perf_counter()
        _parameters(tau,mass,theta)
        key = (float(tau),float(mass),float(theta))
        cache = self._compiled_factors if self.backend=='compiled' else self._factors
        hit = key in cache
        if not hit:
            if self.backend=='python':
                super()._coefficients(tau,mass,theta)
            else:
                damping_increment = .5*np.expm1(-(2*tau/mass)*self.eigenvalues)
                noise_increment = .5*np.sqrt(2*mass*theta * -np.expm1(-(4*tau/mass)*self.eigenvalues))
                D = np.einsum('eik,ek,ejk->eij',self.U,damping_increment,self.U)
                S = np.einsum('eik,ek,ejk->eij',self.U,noise_increment,self.U)
                self._compiled_factors[key] = (_immutable(D),_immutable(S))
        return dict(coefficient_cache_hit=hit,t_coefficient_setup_s=perf_counter()-start,
                    coefficient_cache_size=len(cache),tau=float(tau),mass=float(mass),theta=float(theta),
                    graph_sha256=self.graph_sha256,configuration_token=self.configuration_token,
                    backend=self.backend)

    def sweep(self,p,eta,tau,reverse=False,mass=1.,theta=1.):
        start = perf_counter()
        original,q = _vector(p)
        if len(q)<self.n_particles:
            raise ValueError('p has too few particles for these edges')
        eta = np.asarray(eta,dtype=float)
        if eta.shape!=(len(self.i),3) or not np.isfinite(eta).all():
            raise ValueError('eta must be finite with shape (number of edges,3)')
        if not isinstance(reverse,(bool,np.bool_)):
            raise ValueError('reverse must be boolean')
        prepared = self.prepare(tau,mass,theta)
        if self.backend=='python':
            output,diag = super().sweep(p,eta,tau,reverse,mass,theta)
            diag.update(prepared,t_pairOU=perf_counter()-start,t_kernel_s=diag['t_pairOU'])
            return output,diag
        q = np.array(q,dtype=np.float64,order='C',copy=True)
        eta = np.ascontiguousarray(eta,dtype=np.float64)
        D,S = self._compiled_factors[(float(tau),float(mass),float(theta))]
        pointer = lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        t = perf_counter()
        status = self._library.pairou_sweep(len(q),len(self.i),
            self.i.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            self.j.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            pointer(D),pointer(S),pointer(q),pointer(eta),int(reverse))
        kernel_time = perf_counter()-t
        if status:
            raise RuntimeError(f'compiled pair sweep rejected input, status={status}')
        return q.reshape(original.shape),dict(prepared,pair_updates=len(self.i),
                    t_pairOU=perf_counter()-start,t_kernel_s=kernel_time)

    def rebuild(self,edges,*,configuration_token,n_particles=None):
        token = None if configuration_token is None else str(configuration_token)
        if token is None or token==self.configuration_token:
            raise ValueError('rebuild requires an explicit new configuration token')
        if n_particles is None and self.n_particles>0:
            n_particles=self.n_particles
        return type(self)(edges,n_particles=n_particles,configuration_token=token,
                          backend=self._requested_backend,build_dir=self._build_dir,compiler=self._compiler)
