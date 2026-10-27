"""Immutable-physics Experiment 2 protocol; all outputs stay in this directory."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key]='1'
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import csv,hashlib,json,math
from pathlib import Path
import sys,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import study as S
np=S.np;E=S.E;C=S.C
METHODS=S.METHODS
EPSILONS=(1e-3,1e-4,1e-5)
RESULTS=HERE/'results';RESULTS.mkdir(exist_ok=True)
ETA_MAX=.95
MAX_M=128


def scalar(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=scalar)+'\n')
    temp.replace(path)


def read(path):
    path=Path(path)
    if not path.exists():return []
    with path.open(newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in r.items():
            if v=='':r[k]=None
            elif v in ('True','False'):r[k]=v=='True'
            else:
                try:r[k]=float(v)
                except ValueError:pass
    return rows


def write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with temp.open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=fields);out.writeheader();out.writerows(rows)
    temp.replace(path)


def log(message):print(time.strftime('%H:%M:%S'),message,flush=True)
def configurations():return S.configurations(.75)
def parameters(row):
    return S.parameters(**{k:row[k] for k in ('xi','sR','sF','sT','M','p','kappa','rho') if row.get(k) is not None})
def ident(method,par):return S.ident(method,parameters(par))
def candidate(method,par,**meta):
    p=parameters(par)
    return dict(meta,method=method,id=ident(method,p),**p)


def legal_metadata(method,mesh,par):
    original=getattr(mesh,'original',mesh)
    result=dict(h=float(mesh.h),assignment_order=int(mesh.p),M=int(mesh.M),
                p_legal=bool(1<=mesh.p<=mesh.M),kernel_amplitude=1.,
                projection_constructed=bool(method=='PPPM-grid-PSD'),
                bound_used_as_constraint=False,PSD_eigenvalue_used_as_constraint=False)
    if method!='B2-positive':
        eta=float(mesh.kc*mesh.h/np.pi)
        result.update(kc=float(mesh.kc),sF_effective=float(mesh.kc/(2*par['xi'])),
                      eta=eta,eta_max=ETA_MAX,eta_safe=bool(eta<=ETA_MAX),
                      n_modes=int(original.n_modes),n_excluded=int(original.n_excluded),
                      W2min=float(original.W2min),influence_denominator_threshold=1e-10,
                      mesh_legal=bool(eta<1 and original.n_excluded==0 and original.W2min>1e-10))
    else:result.update(kc=None,sF_effective=None,eta=None,eta_max=None,eta_safe=None,
                       mesh_legal=bool(1<=mesh.p<=mesh.M),positive_mesh_support_radius=float(mesh.R))
    return result


def evaluate(cand,cfg,proxy=True,reference=None,mesh=None):
    """Full spectral operator error. No degree bound/PSD test is a constraint."""
    started=time.perf_counter();method=cand['method'];par=parameters(cand)
    row=dict(cand,config=cfg['name'],family=cfg['family'],seed=cfg['seed'],N=cfg['N'],L=cfg['L'],
             position_hash=cfg['hash'],amplitude=1.,status='failed')
    try:
        G,nref=S.ref(cfg,par['kappa']) if reference is None else reference
        mesh=S.table(method,cfg['L'],par) if mesh is None else mesh
        metadata=legal_metadata(method,mesh,par)
        if not metadata['mesh_legal'] or (method!='B2-positive' and not metadata['eta_safe']):
            raise ValueError('FFT/shell/denominator safety failed: '+str(metadata))
        op=S.build(method,cfg,par,mesh);GL=S.long_dense(method,mesh,cfg['q']);GS=E.real_dense(op)
        error=S.norm(GS+GL-G)/nref
        row.update(metadata,status='ok',E_G=error,E_Gamma=error,normGammaD=float(nref),
                   rc=par['sR']/par['xi'],n_images=op.n_images,
                   t_config_setup_s=float(op.t_setup),**S.diagnostics(mesh))
        if method=='PPPM-grid-PSD':
            old=S.long_dense('PPPM-original',mesh.original,cfg['q'])
            row.update(E_original=S.norm(GS+old-G)/nref,E_proj=S.norm(GL-old)/nref,
                       active=bool(mesh.diagnostics['negative_blocks']),
                       negative_blocks=int(mesh.diagnostics['negative_blocks']))
        else:row.update(E_original=error if method=='PPPM-original' else None,E_proj=None,active=None)
        if proxy:
            vector=np.random.default_rng(822).normal(size=(cfg['N'],3));vector-=vector.mean(axis=0)
            samples=C.median_time(lambda:op.apply(vector),3,warm=1)
            row['t_screen_s']=float(np.median(samples))
            row['proxy_timing_role']='search hint only; final ranking uses fresh serial same-batch timing'
    except (ValueError,MemoryError,FloatingPointError) as exc:
        row.update(status='failed',failure=str(exc),E_G=None)
    row['elapsed_s']=time.perf_counter()-started
    return row


def screen_one(cand):return evaluate(cand,configurations()[0])


def validate_one(cand):
    par=parameters(cand)
    try:mesh=S.table(cand['method'],configurations()[0]['L'],par)
    except (ValueError,MemoryError,FloatingPointError):mesh=None
    return [evaluate(cand,cfg,proxy=False,mesh=mesh) for cfg in configurations()]


def evaluate_batch(candidates,workers=2):
    with ProcessPoolExecutor(max_workers=workers) as pool:return list(pool.map(screen_one,candidates))


def validate_batch(candidates,workers=2):
    with ProcessPoolExecutor(max_workers=workers) as pool:return [row for batch in pool.map(validate_one,candidates) for row in batch]


def summarize_validation(rows):
    groups={}
    for r in rows:groups.setdefault(r['id'],[]).append(r)
    result=[]
    for id_,group in groups.items():
        distinct={r['config']:r for r in group};group=list(distinct.values());first=group[0]
        ok=len(group)==9 and all(r['status']=='ok' and r.get('E_G') is not None for r in group)
        r={k:v for k,v in first.items() if k not in ('config','family','seed','position_hash','E_G','E_Gamma','E_original','E_proj')}
        r.update(id=id_,n_configs=len(group),status='ok' if ok else 'failed',
                 max_E_G=max(x['E_G'] for x in group) if ok else None,
                 max_E_original=max((x.get('E_original') or 0) for x in group) if ok and first['method']!='B2-positive' else None,
                 max_E_proj=max((x.get('E_proj') or 0) for x in group) if ok and first['method']=='PPPM-grid-PSD' else None)
        result.append(r)
    return result
