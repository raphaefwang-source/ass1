"""Experiment-only helpers; every physical operator delegates to existing code."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ass1_theory_mpl')
import hashlib
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import ewald_exp_lib as E
import common as C
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info

KAPPA=.2
TOL=1e-4
RESULTS=HERE/'results'
FIGURES=HERE/'figures'
for _p in (RESULTS,FIGURES): _p.mkdir(exist_ok=True)

def log(message):
    print(time.strftime('%H:%M:%S'),message,flush=True)

def jsave(path,value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=str)+'\n')
    os.replace(tmp,path)

def write(path,rows):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    C.write_csv(tmp,rows)
    tmp.write_bytes(tmp.read_bytes().replace(b'\r\n',b'\n'))
    os.replace(tmp,path)

def read(path):
    path=Path(path)
    return E.read_csv(path) if path.exists() else []

@lru_cache(None)
def configs():
    return tuple(E.config(f,s,.75) for f,s in
                 [('uniform',2),('strong',3)]+[(f,s) for f in E.FAMS for s in E.SEEDS
                                             if (f,s) not in [('uniform',2),('strong',3)]])

def pars(xi,sR,sF,M,p):
    return dict(kappa=KAPPA,rho=.75,theta=0.,xi=float(xi),sR=float(sR),sF=float(sF),M=int(M),p=int(p))

OLD=pars(1.2,3.1,3.1,20,5)

def ident(par):
    return f"xi{par['xi']:g}_R{par['sR']:g}_F{par['sF']:g}_M{par['M']}_p{par['p']}"

def fromrow(r):
    return pars(r['xi'],r['sR'],r['sF_req'],r['M'],r['p'])

@lru_cache(None)
def reference(name):
    cfg=next(c for c in configs() if c['name']==name)
    path=E.ref_path(cfg,KAPPA)
    meta=json.loads(path.with_suffix('.json').read_text())
    assert meta['kappa']==KAPPA and meta['c']==1. and meta['config']==name
    assert meta['position_hash']==cfg['hash'] and meta['source']=='unsplit direct periodic annulus_tensors'
    with np.load(path) as z: G=z['G_D']
    assert G.shape==(768,768)
    return G

@lru_cache(None)
def normref(name):
    return E.norm2_sym(reference(name))[0]

@lru_cache(None)
def probes(name):
    return E.probes_for(reference(name))[0]

def build(cfg,par,meshes=None):
    assert par['kappa']==KAPPA and par['theta']==0.
    K=E.kernel(KAPPA,par['xi'],0.)
    assert K.kap==KAPPA and K.amp==1.
    rc,kc,shell=E.cutoffs(par,cfg['L'])
    mesh,info=(meshes or E.MeshCache()).get(cfg['L'],K,kc,par['M'],par['p'])
    return (E.ThetaOp(cfg['q'],cfg['L'],K,rc,mesh) if info['valid'] else None),info

def evaluate(cfg,par,**extra):
    G=reference(cfg['name'])
    r=E.evaluate(cfg,G,G,probes(cfg['name']),par,E.MeshCache(),timing=False,
                 dense=True,meshfree=False,extra=extra)
    return {k:v for k,v in r.items() if 'vsLF' not in k and not k.startswith('ref_floor')}

def physical_audit():
    r=np.geomspace(.05,100,150)
    out=[]
    for xi in (.45,.7,1.,1.2,1.6,2.):
        K=E.kernel(KAPPA,xi,0.)
        full=E.g_phys(KAPPA)(r)
        rel=float(np.max(np.abs(K.theta_full(r)-full)/full))
        split=float(np.max(np.abs(K.theta_S(r)+K.theta_L(r)-full)/full))
        assert rel<1e-14 and split<1e-10 and K.amp==1. and K.kap==KAPPA
        out.append(dict(xi=xi,kappa=K.kap,c=K.amp,full_profile_rel=rel,split_rel=split))
    jsave(RESULTS/'physical_kernel_audit.json',out)
    return out

def provenance(**extra):
    return dict(E.provenance(),kappa=KAPPA,N=256,rho=.75,c=1.,theta=0.,target=TOL,
                threadpool_info=threadpool_info(),reference='unsplit direct periodic Gamma_D only',
                **extra)
