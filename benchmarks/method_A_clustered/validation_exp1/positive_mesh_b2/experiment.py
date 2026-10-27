"""Shared B2 experiments: same particles, physics, direct references and probes."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ass1_b2_mpl')
import hashlib
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ewald_exp_lib as E
import common as C
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits, threadpool_info
from positive_mesh import PositiveMesh, PositiveGamma

RESULTS, FIGURES = HERE/'results', HERE/'figures'
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
BASE = dict(kappa=.7, rho=.75, xi=.85, sR=6., sT=36., M=80, p=9)
MODERATE = dict(BASE, sR=4., sT=24., M=40, p=7)
OLD = dict(kappa=.2, rho=.75, theta=0., xi=1.25, sR=2.8, sF=2.8, M=16, p=5)
TOL = 1e-4

def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)

def jsave(path, obj):
    C.write_json(path, obj)

def write(path, rows):
    C.write_csv(path, rows)

def read(path):
    return E.read_csv(path) if Path(path).exists() else []

def ident(par):
    return '_'.join(f'{k}{par[k]:g}' for k in ('xi','sR','sT','M','p'))

def parameters(xi, sR, M, p, sT=36., kappa=.2, rho=.75):
    return dict(kappa=kappa, rho=rho, xi=float(xi), sR=float(sR), sT=float(sT), M=int(M), p=int(p))

def configurations(rho=.75):
    pairs = [('uniform',2),('strong',3)] + [(f,s) for f in E.FAMS for s in E.SEEDS
                                          if (f,s) not in [('uniform',2),('strong',3)]]
    return [E.config(f,s,rho) for f,s in pairs]

def reference_path(cfg,kappa):
    old=E.ref_path(cfg,kappa)
    return old if old.exists() else C.RAW/'positive_mesh_b2_refs'/old.name

@lru_cache(12)
def reference(family, seed, rho, kappa):
    cfg = E.config(family, seed, rho)
    path = reference_path(cfg, kappa)
    if not path.exists():
        raise FileNotFoundError(f'Run run_experiments.py refs first: {path}')
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta.get('kappa',kappa)==kappa and meta.get('c',1.)==1. and meta.get('config',cfg['name'])==cfg['name']
    assert meta.get('position_hash', meta.get('hash',cfg['hash']))==cfg['hash']
    assert meta['source'] in ('unsplit direct periodic annulus_tensors','direct_periodic.annulus_tensors only')
    with np.load(path) as z:
        G = z['G_D']
    return G, E.norm2_sym(G)[0]

def ref(cfg,kappa):
    return reference(cfg['family'],cfg['seed'],cfg['rho'],kappa)

@lru_cache(12)
def probes(family,seed,rho,kappa):
    G,_=reference(family,seed,rho,kappa)
    return E.probes_for(G)[0]

@lru_cache(1)
def projector(N):
    return E.projector_basis(N)

@lru_cache(4)
def short_dense(family,seed,rho,kappa,xi,sR):
    cfg=E.config(family,seed,rho)
    K=E.kernel(kappa,xi,0.)
    # Diagnostic far-short reference only: stream images to avoid allocating
    # millions of image records at small xi. Production neighbor/action stays
    # CachedGammaH, unchanged. Original closed Candidate-1 profile is reused.
    pi,pj,acc,_=E.DP.annulus_tensors(cfg['q'],cfg['L'],[sR/xi],budget=1<<18,g=K.theta_S)
    return E.DP.tensors_to_dense(pi,pj,acc[0],cfg['N'])

@lru_cache(2)
def mesh_for(L,kappa,xi,M,sT,quad_order=8):
    return PositiveMesh(L,E.kernel(kappa,xi,0.),M,1,sT,quad_order=quad_order)

def build(cfg,par,mesh=None,quad_order=8):
    K = E.kernel(par['kappa'],par['xi'],0.)
    mesh = mesh or mesh_for(cfg['L'],par['kappa'],par['xi'],par['M'],par['sT'],quad_order).with_order(par['p'])
    return PositiveGamma(cfg['q'],cfg['L'],K,par['sR']/par['xi'],mesh)

def dense(op):
    pi,pj = np.triu_indices(op.N,1)
    GL = E.DP.tensors_to_dense(pi,pj,op.mesh.pair_tensors(op.x),op.N)
    return E.real_dense(op) + GL, GL

def norm(A):
    return E.norm2_sym(A)[0]

def evaluate(cfg,par,mesh=None,structure=False,actions=False):
    G,nref = ref(cfg,par['kappa'])
    op = build(cfg,par,mesh)
    Gh,GLh = dense(op)
    row = dict(config=cfg['name'],family=cfg['family'],seed=cfg['seed'],N=cfg['N'],L=cfg['L'],
               **par, E_G=norm(Gh-G)/nref,n_images=op.n_images,h=op.mesh.h,
               n_quad=len(op.mesh.qw),t_mesh_build_s=op.mesh.t_setup,t_setup_s=op.t_setup)
    if structure:
        Q,T=projector(op.N)
        lam=linalg.eigvalsh(Q.T@Gh@Q,subset_by_index=[0,0])[0]
        row.update(lam_min_h=lam,sym_resid=float(np.linalg.norm(Gh-Gh.T)/np.linalg.norm(Gh)),
                   mom_opnorm_max=float(np.linalg.norm(Gh@T)/nref))
    if actions:
        pr = probes(cfg['family'],cfg['seed'],cfg['rho'],par['kappa'])
        Xm = np.stack([p['v'] for p in pr if p['probe_class']=='gaussian'],1)
        Y=np.stack([op.matvec(Xm[:,k]) for k in range(Xm.shape[1])],1)
        row.update(E_GX=float(np.linalg.norm(Y-G@Xm)/np.linalg.norm(G@Xm)),
                   mf_vs_dense=float(np.linalg.norm(Y-Gh@Xm)/np.linalg.norm(Y)))
        T=np.tile(np.eye(3),(op.N,1))/np.sqrt(op.N)
        row['mf_null']=max(np.linalg.norm(op.matvec(T[:,k])) for k in range(3))/nref
    return row,op,Gh,GLh
