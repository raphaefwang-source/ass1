"""Shared three-method study. Historical modules and references are read-only."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ass1_gridpsd_mpl')
import sys, time, json, hashlib, traceback
from pathlib import Path
from functools import lru_cache
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'positive_mesh_b2'))
import experiment as B
from experiment import E,C,np,threadpool_limits,threadpool_info
from scipy import linalg
from exp1_dense_reference import mode_pair_matrix_fast
from test_parameter_selection import lap_full
from positive_mesh import PositiveMesh
sys.path.insert(0,str(HERE))
RESULTS=HERE/'results'; RESULTS.mkdir(exist_ok=True)
METHODS=('PPPM-original','B2-positive','PPPM-grid-PSD')
read=B.read; write=B.write; jsave=B.jsave; log=B.log
norm=B.norm; configurations=B.configurations; ref=B.ref

def parameters(xi=1.25,sR=2.8,M=16,p=5,sF=2.8,sT=36.,kappa=.2,rho=.75):
    return dict(xi=float(xi),sR=float(sR),M=int(M),p=int(p),sF=float(sF),sT=float(sT),kappa=float(kappa),rho=float(rho))

def ident(method,p):
    keys=('xi','sR','sT' if method=='B2-positive' else 'sF','M','p')
    return method+'_'+'_'.join(f'{k}{p[k]:g}' for k in keys)

def table(method,L,par):
    K=E.kernel(par['kappa'],par['xi'],0.)
    if method=='B2-positive':
        return PositiveMesh(L,K,par['M'],par['p'],par['sT'])
    _,kc,_=E.cutoffs(par,L)
    mesh,info=E.MeshCache().get(L,K,kc,par['M'],par['p'])
    if mesh is None: raise ValueError('invalid original mesh: '+str(info))
    if method=='PPPM-grid-PSD':
        from grid_psd import ProjectedMesh
        return ProjectedMesh(mesh,audit_eigenvalues=False)
    return mesh

def build(method,cfg,par,mesh=None):
    mesh=mesh if mesh is not None else table(method,cfg['L'],par)
    K=E.kernel(par['kappa'],par['xi'],0.)
    if method=='B2-positive': return B.build(cfg,par,mesh)
    if method=='PPPM-grid-PSD':
        from grid_psd import GridPSDGamma
        return GridPSDGamma(cfg['q'],cfg['L'],K,par['sR']/par['xi'],mesh)
    return E.ThetaOp(cfg['q'],cfg['L'],K,par['sR']/par['xi'],mesh)

def long_dense(method,mesh,x):
    if method=='B2-positive':
        i,j=np.triu_indices(len(x),1)
        return E.DP.tensors_to_dense(i,j,mesh.pair_tensors(x),len(x))
    if method=='PPPM-grid-PSD':
        W=mesh.pair_matrix(x)
        if isinstance(W,tuple): W=W[0]
    else: W=mode_pair_matrix_fast(mesh,x)
    return lap_full(W)

def diagnostics(mesh):
    d=getattr(mesh,'diagnostics',{})
    d=d() if callable(d) else dict(d)
    for src,dst in [('min_eig_before','min_raw_eig'),('fraction_negative_blocks','neg_fraction'),
                    ('correction_max_spectral','max_correction')]:
        if src in d:d[dst]=d[src]
    return d

def measure(method,cfg,par,structure=False,actions=False,mesh_only=False,cached=None):
    start=time.perf_counter(); G,nref=ref(cfg,par['kappa'])
    if cached is None:
        mesh=table(method,cfg['L'],par)
        GL=long_dense(method,mesh,cfg['q'])
    else: mesh,GL=cached
    op=build(method,cfg,par,mesh)
    GS=E.real_dense(op); Gh=GS+GL
    r=dict(method=method,id=ident(method,par),config=cfg['name'],family=cfg['family'],seed=cfg['seed'],
           N=cfg['N'],L=cfg['L'],h=mesh.h,**par,E_G=norm(Gh-G)/nref,n_images=op.n_images,
           t_setup_s=op.t_setup,**diagnostics(mesh))
    if method=='B2-positive':r.update(R_T=mesh.R)
    else:r.update(kc=mesh.kc,sF_eff=mesh.kc/(2*par['xi']))
    if method=='PPPM-grid-PSD':
        original=getattr(mesh,'original',getattr(mesh,'old',None))
        if original is None: original=table('PPPM-original',cfg['L'],par)
        GLold=long_dense('PPPM-original',original,cfg['q'])
        r.update(E_proj=norm(GL-GLold)/nref,E_original=norm(GS+GLold-G)/nref)
    if structure:
        Q,T=B.projector(op.N)
        r.update(lam_min_h=float(linalg.eigvalsh(Q.T@Gh@Q,subset_by_index=[0,0])[0]),
                 lam_min_long=float(linalg.eigvalsh(Q.T@GL@Q,subset_by_index=[0,0])[0]),
                 sym_resid=float(np.linalg.norm(Gh-Gh.T)/np.linalg.norm(Gh)),
                 mom_opnorm_max=float(np.linalg.norm(Gh@T)/nref))
        if method=='PPPM-grid-PSD':
            r.update(lam_min_long_before=float(linalg.eigvalsh(Q.T@GLold@Q,subset_by_index=[0,0])[0]),
                     lam_min_h_before=float(linalg.eigvalsh(Q.T@(GS+GLold)@Q,subset_by_index=[0,0])[0]))
    if actions:
        pr=B.probes(cfg['family'],cfg['seed'],cfg['rho'],par['kappa'])
        Xm=np.stack([p['v'] for p in pr if p['probe_class']=='gaussian'],1)
        Y=np.stack([op.matvec(Xm[:,k]) for k in range(Xm.shape[1])],1)
        r.update(E_GX=float(np.linalg.norm(Y-G@Xm)/np.linalg.norm(G@Xm)),
                 mf_vs_dense=float(np.linalg.norm(Y-Gh@Xm)/np.linalg.norm(Y)))
        T=np.tile(np.eye(3),(op.N,1))/np.sqrt(op.N)
        r['mf_null']=max(np.linalg.norm(op.matvec(T[:,k])) for k in range(3))/nref
    if mesh_only:
        GSfar=B.short_dense(cfg['family'],cfg['seed'],cfg['rho'],par['kappa'],par['xi'],8.)
        r['E_G_mesh_only']=norm(GL-(G-GSfar))/nref
        r['E_mesh']=r['E_G_mesh_only'];r['E_R']=norm(GS-GSfar)/nref
    r['elapsed_s']=time.perf_counter()-start
    return r,op,Gh,GL

def attempt(method,cfg,par,**kwargs):
    try:return measure(method,cfg,par,**kwargs)
    except (ValueError,MemoryError,FloatingPointError) as exc:
        return dict(method=method,id=ident(method,par),config=cfg['name'],**par,status='failed',failure=str(exc)),None,None,None
