"""Stage 0 physical spatial certification; production algorithms are PR7--9 imports.

No historical output is written. The direct reference sums the UNSPLIT physical
kernel over all retained lattice images, with an analytic bound on the infinite
remainder. Neither a mesh nor the split radial kernel enters this reference.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
HERE = Path(__file__).resolve().parent
CLUSTER = HERE.parent
ROOT = CLUSTER.parents[1]
for _path in (ROOT/'ewald_longitudinal', CLUSTER, CLUSTER/'validation_exp1',
              CLUSTER/'validation_exp1/deconv_psd_comparison'):
    if str(_path) not in sys.path: sys.path.insert(0,str(_path))
import numpy as np
from scipy import linalg, special
from radial_kernels import RadialEwald
from grid_psd import ProjectedMesh, GridPSDGamma
import direct_periodic as DP

KAPPA = .7
AMPLITUDE = float(.5*1.3*np.exp(.7*1.3))
STRICT_SPATIAL_TOL = 2e-7
EXPLORATORY_SPATIAL_TOL = 1e-4


def physical_parameters(N=256):
    return dict(N=int(N),rho=.75,L=float((N/.75)**(1/3)),kappa=KAPPA,c=AMPLITUDE)


def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def save_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def encode(v):
        if isinstance(v,np.generic): return v.item()
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,Path): return str(v)
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,default=encode)+'\n')


def norm2(A):
    ev=linalg.eigvalsh((A+A.T)*.5,check_finite=False)
    return float(max(abs(ev[0]),abs(ev[-1])))


def cutoff(params,L):
    # Identical reciprocal shell convention to existing Ewald implementation.
    from ewald_exp_lib import snap_kc
    kc,shell=snap_kc(2*float(params['xi'])*float(params['sF']),L)
    return float(params['sR'])/float(params['xi']),kc,shell


def build_mesh(params,L):
    if not np.isclose(float(params.get('kappa',KAPPA)),KAPPA,rtol=0,atol=0):
        raise ValueError('Pilot requires kappa=.7')
    if not np.isclose(float(params.get('c',AMPLITUDE)),AMPLITUDE,rtol=1e-14):
        raise ValueError('Pilot requires exact gamma*r_ref*exp(kappa*r_ref) amplitude')
    K=RadialEwald('A',KAPPA,float(params['xi']),amp=AMPLITUDE)
    _,kc,_=cutoff(params,L)
    M,p=int(params['M']),int(params['p'])
    if not 1<=p<=M or kc*(L/M)/np.pi>=1:
        raise ValueError('Illegal mesh p/M or Nyquist cutoff')
    mesh=ProjectedMesh.from_parameters(float(L),K,kc,M,p,audit_eigenvalues=False)
    if mesh.n_excluded or mesh.W2min<=1e-10: raise ValueError('Illegal deconvolved mesh')
    return mesh


def build_operator(q,params,mesh=None):
    q=np.asarray(q,float)
    L=float(params.get('L',(len(q)/.75)**(1/3))) if mesh is None else float(mesh.L)
    if mesh is None: mesh=build_mesh(params,L)
    rc,_,_=cutoff(params,L)
    return GridPSDGamma(q,L,mesh.K,rc,mesh)


def near_edges(op):
    keys=op.i.astype(np.int64)*op.N+op.j
    unique,inverse=np.unique(keys,return_inverse=True)
    tensors=np.zeros((len(unique),3,3))
    for a in range(3):
        for b in range(a,3):
            tensors[:,a,b]=tensors[:,b,a]=np.bincount(inverse,op.w*op.d[:,a]*op.d[:,b],minlength=len(unique))
    return [(int(k//op.N),int(k%op.N),K) for k,K in zip(unique,tensors)]


def edges_dense(edges,N):
    blocks=np.zeros((N,N,3,3))
    for i,j,K in edges:
        blocks[i,i]+=K;blocks[j,j]+=K;blocks[i,j]-=K;blocks[j,i]-=K
    return blocks.transpose(0,2,1,3).reshape(3*N,3*N)


def pair_graph_dense(W):
    W=np.asarray(W,float);N=len(W);blocks=-W.copy();i=np.arange(N)
    blocks[i,i]+=W.sum(axis=1)
    return blocks.transpose(0,2,1,3).reshape(3*N,3*N)


def dense_parts(op):
    GS=edges_dense(near_edges(op),op.N)
    W=op.mesh.pair_matrix(op.x)
    if isinstance(W,tuple): W=W[0]
    GL=pair_graph_dense(W)
    return GS,GL,GS+GL


def long_action(op,v):
    X=np.asarray(v,float).reshape(op.N,3)
    return (np.einsum('nab,nb->na',op.Dm,X)-op.mesh.apply(op.idx,op.wt,X)).ravel()


def physical_tail_bound(N,L,R):
    """Graph norm bound 2 max degree from disjoint lattice-cube counting.

    n(r) <= (N-1)*4pi/3*(r+sqrt(3)L/2)^3/L^3. Integrate this
    against -d[c exp(-kr)/r], dropping the nonpositive boundary term.
    """
    a=np.sqrt(3)*L/2;k=KAPPA;e=np.exp(-k*R)
    integral=e*(R*R+2*R/k+2/k**2)
    integral+=(3*k*a+1)*e*(R/k+1/k**2)
    integral+=(3*k*a*a+3*a)*e/k
    integral+=3*a*a*special.exp1(k*R)+a**3*e/R
    return float(2*(N-1)*4*np.pi/3/L**3*integral*AMPLITUDE)


def direct_reference(q,L=None,output=None,R=45.):
    q=np.asarray(q,float);N=len(q);L=float((N/.75)**(1/3)) if L is None else float(L)
    signature=array_sha(np.mod(q,L))
    prefix=None if output is None else Path(output)
    if prefix is not None and prefix.with_suffix('.npz').exists() and prefix.with_suffix('.json').exists():
        meta=json.loads(prefix.with_suffix('.json').read_text())
        if meta['q_sha256']!=signature or meta['kappa']!=KAPPA or meta['c']!=AMPLITUDE or meta['R_max']<R:
            raise ValueError('Reference cache physics, radius, or geometry mismatch')
        with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as d: G=d['G'].copy()
        return G,meta
    t=perf_counter();radii=np.array([.75*R,.9*R,R])
    g=lambda r: AMPLITUDE*np.exp(-KAPPA*np.asarray(r))/np.asarray(r)
    pi,pj,acc,count=DP.annulus_tensors(q,L,radii,budget=1<<18,g=g)
    cum=np.cumsum(acc,axis=0);G=DP.tensors_to_dense(pi,pj,cum[-1],N);nrm=norm2(G)
    convergence=[]
    for idx,r in enumerate(radii):
        partial=DP.tensors_to_dense(pi,pj,cum[idx],N)
        convergence.append(dict(R=float(r),images=int(count[:idx+1].sum()),
            difference_to_final_relative=norm2(partial-G)/nrm,
            certified_tail_absolute=physical_tail_bound(N,L,r),
            certified_tail_relative=physical_tail_bound(N,L,r)/nrm))
    meta=dict(N=N,L=L,kappa=KAPPA,c=AMPLITUDE,R_max=float(R),q_sha256=signature,
        matrix_sha256=array_sha(G),norm2=nrm,reference='unsplit full periodic direct lattice sum; no Ewald/mesh',
        excludes_self_particle_edges=True,total_images=int(count.sum()),convergence=convergence,
        certified_tail_relative=physical_tail_bound(N,L,R)/nrm,wall_s=perf_counter()-t)
    if prefix is not None:
        prefix.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(prefix.with_suffix('.npz'),G=G,q=q,L=L,radii=radii,
                            annulus_tensors=acc,pi=pi,pj=pj,counts=count)
        save_json(prefix.with_suffix('.json'),meta)
    return G,meta


def operator_diagnostics(op,GS,GL,GH):
    nrm=norm2(GH);N=op.N; rng=np.random.default_rng(2026102801)
    errors=[];part_errors=[]
    for _ in range(3):
        v=rng.normal(size=3*N)
        errors.append(float(np.linalg.norm(op.matvec(v)-GH@v)/np.linalg.norm(GH@v)))
        part_errors.append(float(np.linalg.norm(op.matvec(v)-GS@v-long_action(op,v))/np.linalg.norm(GH@v)))
    trans=np.tile(np.eye(3),(N,1));spectrum=linalg.eigvalsh((GH+GH.T)/2,check_finite=False)
    long_ev=linalg.eigvalsh((GL+GL.T)/2,check_finite=False)
    rc,kc,shell=cutoff(dict(xi=op.K.xi,sR=op.rc*op.K.xi,sF=op.mesh.kc/(2*op.K.xi)),op.L)
    return dict(relative_symmetry=float(np.linalg.norm(GH-GH.T)/np.linalg.norm(GH)),
        COM_null_relative=float(np.linalg.norm(GH@trans)/nrm),min_eigenvalue=float(spectrum[0]),
        long_min_eigenvalue=float(long_ev[0]),largest_eigenvalue=float(spectrum[-1]),
        FFT_action_relative_max=max(errors),full_near_long_relative_max=max(part_errors),
        self_particle_edges=int(np.sum(op.i==op.j)),images=int(op.n_images),
        unique_near_edges=len(near_edges(op)),rc=rc,kc=kc,shell=shell,
        full_rank_near_edges=sum(np.linalg.matrix_rank(e[2],tol=1e-12)==3 for e in near_edges(op)),
        projection_active=bool(op.mesh.diagnostics['negative_blocks']),
        physical_k0_preserved_before_projection=bool(np.linalg.norm(op.mesh.zero_mode_before)>0),
        zero_mode_before=op.mesh.zero_mode_before.tolist(),zero_mode_after=op.mesh.zero_mode.tolist(),
        mesh_self_terms_cancel='full W_ii appears in both degree and convolution and cancels',
        M=op.mesh.M,p=op.mesh.p,kappa=op.K.kap,c=op.K.amp)


def validate_state(q,params,output=None,reference_prefix=None):
    q=np.asarray(q,float);L=float(params.get('L',(len(q)/.75)**(1/3)))
    if reference_prefix is None and output is not None:
        reference_prefix=Path(output).with_suffix('')
        reference_prefix=reference_prefix.parent/(reference_prefix.name+'_direct')
    G,ref=direct_reference(q,L,reference_prefix)
    t=perf_counter();mesh=build_mesh(params,L);table_wall=perf_counter()-t
    t=perf_counter();op=build_operator(q,params,mesh);GS,GL,GH=dense_parts(op)
    raw_error=norm2(GH-G)/ref['norm2'];tail=ref['certified_tail_relative']
    record=dict(params=dict(params,L=L,kappa=KAPPA,c=AMPLITUDE),q_sha256=array_sha(np.mod(q,L)),
        relative_spectral_error=raw_error,reference_tail_relative=tail,
        conservative_relative_error_bound=(raw_error+tail)/(1-tail),
        strict_budget_met=bool((raw_error+tail)/(1-tail)<=STRICT_SPATIAL_TOL),
        exploratory_budget_met=bool((raw_error+tail)/(1-tail)<=EXPLORATORY_SPATIAL_TOL),
        reference=ref,diagnostics=operator_diagnostics(op,GS,GL,GH),
        table_setup_s=table_wall,operator_dense_validation_s=perf_counter()-t)
    d=record['diagnostics'];roundoff=256*np.finfo(float).eps*max(1.,d['largest_eigenvalue'])
    record['structural_checks_passed']=bool(d['relative_symmetry']<=1e-12 and d['COM_null_relative']<=1e-11
        and d['min_eigenvalue']>=-roundoff and d['long_min_eigenvalue']>=-roundoff
        and d['FFT_action_relative_max']<=1e-10 and d['full_near_long_relative_max']<=1e-10
        and d['self_particle_edges']==0 and d['physical_k0_preserved_before_projection'])
    if output is not None: save_json(output,record)
    return record


def select_spatial(q,output_dir):
    """Small feasibility sequence, not a performance optimization search."""
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    L=float((len(q)/.75)**(1/3));G,ref=direct_reference(q,L,output_dir/'selection_direct')
    candidates=[dict(xi=.85,sR=4.1,sF=4.1,M=24,p=8),
                dict(xi=.85,sR=4.7,sF=4.7,M=40,p=10),
                dict(xi=.65,sR=4.7,sF=4.7,M=40,p=10),
                dict(xi=.85,sR=5.2,sF=5.2,M=60,p=12)]
    rows=[];selected=None
    for idx,par in enumerate(candidates):
        par.update(kappa=KAPPA,c=AMPLITUDE,L=L)
        print('spatial candidate',idx,par,flush=True)
        try:
            rec=validate_state(q,par,output_dir/f'spatial_candidate_{idx}.json',output_dir/'selection_direct')
            row=dict(index=idx,**par,relative_spectral_error=rec['relative_spectral_error'],
                conservative_relative_error_bound=rec['conservative_relative_error_bound'],
                strict_budget_met=rec['strict_budget_met'],exploratory_budget_met=rec['exploratory_budget_met'],
                structural_checks_passed=rec['structural_checks_passed'],
                table_setup_s=rec['table_setup_s'],operator_dense_validation_s=rec['operator_dense_validation_s'])
            rows.append(row);print('spatial result',row,flush=True)
            if rec['strict_budget_met'] and rec['structural_checks_passed']:
                selected=rec;break
        except Exception as exc:
            rows.append(dict(index=idx,**par,error=repr(exc)))
            print('spatial candidate error',repr(exc),flush=True)
    if selected is None:
        ok=[r for r in rows if r.get('exploratory_budget_met') and r.get('structural_checks_passed')]
        if not ok:
            save_json(output_dir/'spatial_selection.json',dict(status='NO_VALID_SETTING',candidates=rows,params=None))
            raise RuntimeError('No actually verified <=1e-4 spatial setting')
        best=min(ok,key=lambda r:r['conservative_relative_error_bound'])
        selected=json.loads((output_dir/f"spatial_candidate_{best['index']}.json").read_text())
    status='STRICT_BUDGET_MET' if selected['strict_budget_met'] else 'strict Overleaf budget NOT MET; exploratory-only'
    result=dict(status=status,params=selected['params'],selected_validation=selected,
                candidates=rows,strict_tolerance=STRICT_SPATIAL_TOL,
                exploratory_tolerance=EXPLORATORY_SPATIAL_TOL,
                scope='small spatial feasibility sequence; no fastest-parameter production search')
    save_json(output_dir/'spatial_selection.json',result)
    with (output_dir/'spatial_candidates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output',default=str(HERE/'results'))
    p.add_argument('--params',help='selected JSON or plain parameter dictionary for validation only')
    args=p.parse_args()
    if args.input:
        with np.load(args.input,allow_pickle=False) as d:q=d['q'].copy()
    else:
        # A noncanonical geometry is allowed only for the initial Stage0 feasibility test.
        q=np.load(CLUSTER/'validation_exp1/configs/uniform_N256_s2.npz')['q']
    if args.params:
        params=json.loads(Path(args.params).read_text());params=params.get('params',params)
        result=validate_state(q,params,args.output)
        print(json.dumps({k:result[k] for k in ('relative_spectral_error','conservative_relative_error_bound',
            'strict_budget_met','exploratory_budget_met','structural_checks_passed')},indent=2),flush=True)
    else:
        select_spatial(q,args.output)
