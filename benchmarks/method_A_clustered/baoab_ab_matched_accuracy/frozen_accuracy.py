"""Full COM-subspace frozen transition operator accuracy on the local Mac.

The dense graph is independently assembled from spatial pair tensors. The
ideal A mean is a literal ordered product of exact near-edge means and a dense
long exponential. Ideal conditional covariance follows from exact independent
OU substeps preserving Maxwell covariance; a separate small-system test builds
every Gaussian column and verifies that identity. Finite Lanczos is not used
to construct a purported fixed covariance here.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key]='1'
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from scipy import linalg

HERE=Path(__file__).resolve().parent
PILOT=HERE.parent/'baoab_ab_pilot'
THERMOSTAT=HERE.parent/'thermostat_ab_local'
for _folder in (THERMOSTAT,THERMOSTAT/'optimized_pairou',PILOT):
    if str(_folder) not in sys.path:sys.path.insert(0,str(_folder))
from spatial_validation import build_mesh,build_operator,dense_parts,near_edges
from compiled_edge import CompiledEdgeOU
from protocol import DenseOU

DT=.005
TARGETS=(1e-4,1e-6,2e-8)
DEFAULT_KS=(1,2,4,8,16,32)
SMALL=1e-300

def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def encode(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,default=encode)+'\n')

def save_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def prepare(q,params,mesh=None):
    """Build actual FFT actions plus independent dense full/long references."""
    q=np.asarray(q,float)
    if mesh is None:mesh=build_mesh(params,float(params.get('L',(len(q)/.75)**(1/3))))
    op=build_operator(q,params,mesh=mesh)
    GS,GL,GH=dense_parts(op)
    edges=CompiledEdgeOU(near_edges(op),n_particles=len(q),configuration_token=array_sha(q))
    if edges.backend!='compiled':raise RuntimeError('Actual C pair backend required')
    full_ref,long_ref=DenseOU(GH),DenseOU(GL)
    return dict(op=op,edges=edges,GS=GS,GL=GL,GH=GH,
                full_ref=full_ref,long_ref=long_ref,Q=full_ref.Q)

def _saved_reference(N,Q,eig,U):
    ref=object.__new__(DenseOU)
    ref.N=N;ref.Q=Q;ref.Pi=Q@Q.T;ref.T=np.eye(3*N)-ref.Pi
    ref.eig=eig;ref.U=U;ref.lmax=float(eig[-1]);ref.raw_min=float(eig[0])
    return ref

def load_prepared(path,params,mesh=None):
    """Reuse saved independent eigensolves while rebuilding actual FFT actions."""
    with np.load(path,allow_pickle=False) as data:d={k:data[k].copy() for k in data.files}
    q=d['q'];N=len(q)
    op=build_operator(q,params,mesh=mesh)
    edges=CompiledEdgeOU(near_edges(op),n_particles=N,configuration_token=array_sha(q))
    full_ref=_saved_reference(N,d['Q'],d['full_eig'],d['full_U'])
    long_ref=_saved_reference(N,d['Q'],d['long_eig'],d['long_U'])
    # A new actual spatial action must agree with the independently saved graph.
    probe=np.random.default_rng(202610280510).normal(size=3*N)
    if np.linalg.norm(op.matvec(probe)-d['GH']@probe)>2e-11*max(np.linalg.norm(d['GH']@probe),1.):
        raise RuntimeError('Saved dense graph does not match rebuilt physical FFT operator')
    return dict(op=op,edges=edges,GS=d['GS'],GL=d['GL'],GH=d['GH'],
                full_ref=full_ref,long_ref=long_ref,Q=d['Q'])

def ordered_near_mean(edges,short_step):
    """Literal dense forward near product; no C sweep or OU implementation call."""
    N=edges.n_particles;dimension=3*N
    F=np.eye(dimension).reshape(N,3,dimension)
    # tau=short_step/2; the relative mode contracts by exp(-2*tau*K).
    D=.5*np.einsum('eia,ea,eja->eij',edges.U,
                  np.expm1(-float(short_step)*edges.eigenvalues),edges.U)
    for e,(i,j) in enumerate(zip(edges.i,edges.j)):
        delta=D[e]@(F[int(i)]-F[int(j)])
        F[int(i)]+=delta;F[int(j)]-=delta
    return F.reshape(dimension,dimension)

def ideal_mean(prepared,k,dt=DT):
    if isinstance(k,bool) or int(k)!=k or k<1:raise ValueError('k must be positive integer')
    F=ordered_near_mean(prepared['edges'],dt/int(k))
    Rlong=prepared['long_ref'].matrices(dt/int(k))[0]
    one=F.T@Rlong@F
    return np.linalg.matrix_power(one,int(k))

def exact_operator_norm(matrix):
    """Full spectral norm, with an explicit symmetry test and SVD fallback."""
    matrix=np.asarray(matrix,float)
    asym=float(np.linalg.norm(matrix-matrix.T,'fro'))
    scale=max(float(np.linalg.norm(matrix,'fro')),np.finfo(float).tiny)
    if asym<=2e-12*scale+2e-14:
        eig=linalg.eigvalsh((matrix+matrix.T)*.5,check_finite=False)
        return float(np.max(np.abs(eig),initial=0.)),dict(method='full symmetric eigvalsh',
            symmetry_residual_fro=asym,symmetry_relative_fro=asym/scale),eig
    singular=linalg.svdvals(matrix,check_finite=False)
    return float(singular[0]),dict(method='full svdvals',symmetry_residual_fro=asym,
        symmetry_relative_fro=asym/scale),singular

def compare_maps(prepared,k,dt=DT):
    """Full COM-subspace normalized mean/covariance operator errors."""
    t=perf_counter();Q=prepared['Q'];full=prepared['full_ref']
    RB,_,CB=full.matrices(dt)
    b=Q.T@RB@Q;c=Q.T@CB@Q
    RA=ideal_mean(prepared,k,dt);a=Q.T@RA@Q
    # Analytic ideal linear substep identity. It is not finite-Lanczos covariance.
    ca=np.eye(a.shape[0])-a@a.T
    mean_abs,md,me=exact_operator_norm(a-b)
    cov_abs,cd,ce=exact_operator_norm(ca-c)
    mean_den=float(np.exp(-dt*full.eig[0]))
    cov_den=float(np.max(-np.expm1(-2*dt*full.eig)))
    record=dict(k=int(k),dt=float(dt),dimension_full=3*full.N,dimension_COM=3*full.N-3,
        mean_absolute_norm=mean_abs,cov_absolute_norm=cov_abs,
        mean_denominator=mean_den,cov_denominator=cov_den,
        denominator_small=SMALL,mean_denominator_used=max(mean_den,SMALL),cov_denominator_used=max(cov_den,SMALL),
        near_zero_mean_denominator=mean_den<SMALL,near_zero_cov_denominator=cov_den<SMALL,
        e_mean=mean_abs/max(mean_den,SMALL),e_cov=cov_abs/max(cov_den,SMALL),
        mean_norm_method=md['method'],cov_norm_method=cd['method'],
        mean_difference_symmetry_fro=md['symmetry_residual_fro'],
        cov_difference_symmetry_fro=cd['symmetry_residual_fro'],
        RA_full_COM_residual=float(np.linalg.norm(RA@np.tile(np.eye(3),(full.N,1))-np.tile(np.eye(3),(full.N,1)))),
        RA_reduced_sha256=array_sha(a),CA_reduced_sha256=array_sha(ca),
        RB_reduced_sha256=array_sha(b),CB_reduced_sha256=array_sha(c),
        elapsed_s=perf_counter()-t,
        covariance_interpretation='ideal exact independent linear OU substeps only; CA=I-RA*RA.T on COM; not finite-Lanczos covariance',
        norm_scope='full 3N-3 COM subspace, not sampled probes')
    return record,dict(RA_reduced=a,CA_reduced=ca,
        mean_difference_spectrum=me,cov_difference_spectrum=ce)

def _ref_arrays(prepared,q,p):
    RB,_,CB=prepared['full_ref'].matrices(DT)
    return dict(q=q,p=p,GS=prepared['GS'],GL=prepared['GL'],GH=prepared['GH'],Q=prepared['Q'],
        full_eig=prepared['full_ref'].eig,full_U=prepared['full_ref'].U,
        long_eig=prepared['long_ref'].eig,long_U=prepared['long_ref'].U,
        RB=RB,CB=CB,RB_reduced=prepared['Q'].T@RB@prepared['Q'],CB_reduced=prepared['Q'].T@CB@prepared['Q'],
        edge_i=prepared['edges'].i,edge_j=prepared['edges'].j,edge_K=prepared['edges'].K,
        edge_U=prepared['edges'].U,edge_eigenvalues=prepared['edges'].eigenvalues)

def run_accuracy(states,params,out,ks=DEFAULT_KS):
    """Run every listed k on each supplied geometry before selecting any target."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    marker=out/'frozen_selection.json'
    if marker.exists():raise FileExistsError('Preserve completed frozen selection evidence')
    rows=[];domains=[];started=perf_counter()
    mesh=build_mesh(params,float(params['L']))
    for state in states:
        label=str(state.get('label',state.get('name')))
        if label in ('None','') or '/' in label:raise ValueError('A safe state label is required')
        q=np.asarray(state['q'],float);p=np.asarray(state.get('p',np.zeros_like(q)),float)
        print('prepare full COM reference',label,flush=True)
        t=perf_counter();prepared=prepare(q,params,mesh)
        ref_path=out/f'frozen_refs_{label}.npz'
        np.savez_compressed(ref_path,**_ref_arrays(prepared,q,p))
        domain=dict(label=label,N=len(q),q_sha256=array_sha(q),p_sha256=array_sha(p),
            GH_sha256=array_sha(prepared['GH']),GL_sha256=array_sha(prepared['GL']),
            reference_file=ref_path.name,reference_file_sha256=file_sha(ref_path),
            dimension_COM=3*len(q)-3,lambda_min=float(prepared['full_ref'].eig[0]),
            lambda_max=float(prepared['full_ref'].eig[-1]),
            graph_build_and_reference_s=perf_counter()-t,metadata=state.get('metadata',{}))
        domains.append(domain)
        previous=None
        for k in ks:
            record,arrays=compare_maps(prepared,int(k))
            record.update(state=label,q_sha256=domain['q_sha256'],GH_sha256=domain['GH_sha256'])
            if previous is not None:
                record['observed_slope_mean']=float(np.log(record['e_mean']/previous['e_mean'])/np.log(k/previous['k']))
                record['observed_slope_cov']=float(np.log(record['e_cov']/previous['e_cov'])/np.log(k/previous['k']))
            path=out/f'frozen_maps_{label}_k{k}.npz'
            np.savez_compressed(path,**arrays)
            record['maps_file']=path.name;record['maps_file_sha256']=file_sha(path)
            rows.append(record);previous=record
            save_csv(out/'frozen_errors.csv',rows)
            print(label,'k',k,'mean',record['e_mean'],'cov',record['e_cov'],flush=True)
        save_json(out/f'frozen_domain_{label}.json',domain)
    selection=[]
    for target in TARGETS:
        accepted=[]
        for k in ks:
            batch=[r for r in rows if r['k']==k]
            if len(batch)==len(states) and all(max(r['e_mean'],r['e_cov'])<=target for r in batch):accepted.append(int(k))
        selected=min(accepted) if accepted else None
        rr=[r for r in rows if r['k']==selected] if selected is not None else []
        selection.append(dict(epsilon_frozen=target,k=selected,
            status='ideal_operator_accuracy_met' if selected is not None else 'accuracy_not_met',
            max_e_mean=max((r['e_mean'] for r in rr),default=None),max_e_cov=max((r['e_cov'] for r in rr),default=None),
            actually_tested_passing_ks=accepted,
            selection_scope='minimum actually tested k passing both full COM operator norms on every registered geometry'))
    result=dict(targets=selection,states=domains,ks=list(map(int,ks)),params=params,dt=DT,
        denominator_rules=dict(mean='||RB Pi||2=max(exp(-dt*lambda)) on full COM subspace',
            covariance='||CB||2=max(-expm1(-2dt*lambda))',small=SMALL),
        elapsed_s=perf_counter()-started,
        limitation='Ideal frozen transition operators only. Finite Lanczos composition, Monte Carlo and moving-state timing require separate actual validation; no full BAOAB physical-observable equivalence claim.')
    save_json(marker,result);save_json(out/'frozen_domains.json',domains)
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--states',required=True);parser.add_argument('--params',required=True)
    parser.add_argument('--output',default=str(HERE/'results'));args=parser.parse_args()
    state_index=json.loads(Path(args.states).read_text())
    states=[]
    for row in state_index:
        with np.load(row['file'],allow_pickle=False) as d:
            q=d[row.get('q_key','q')];p=d[row.get('p_key','p')]
            if 'index' in row:q=q[int(row['index'])];p=p[int(row.get('p_index',row['index']))]
        states.append(dict(label=row['label'],q=q,p=p,metadata=row))
    params=json.loads(Path(args.params).read_text());params=params.get('params',params)
    result=run_accuracy(states,params,args.output)
    print(json.dumps(result['targets'],indent=2),flush=True)
