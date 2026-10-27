"""Checkpointed accuracy sweeps, preserving all historical artifacts."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import study as X
from study import np,E,B
import importlib.util
spec=importlib.util.spec_from_file_location('old_accuracy',X.HERE.parent/'accuracy_sweeps'/'run_accuracy_sweeps.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)

def import_history():
    rows=[]
    for method,path in [('PPPM-original',X.HERE.parent/'accuracy_sweeps'/'results'/'sweeps.csv'),
                        ('B2-positive',X.HERE.parent/'positive_mesh_b2'/'results'/'sweeps.csv')]:
        for r in X.read(path):
            r.update(method=method,source='historical_readonly',source_path=str(path.relative_to(X.C.REPO)))
            if method=='PPPM-original':r['sF']=r.get('sF_req')
            rows.append(r)
    return rows

def group_worker(job):
    (f,s,rho,kappa),items=job
    name=f'{f}_s{s}_rho{rho:g}_kap{kappa:g}'
    path=X.RESULTS/'sweep_parts'/f'{name}.csv'
    path.parent.mkdir(exist_ok=True)
    rows=X.read(path);done={(r['sweep'],r['setting'],r['value']) for r in rows}
    cfg=E.config(f,s,rho);last_key=None;cached=None
    with X.threadpool_limits(1):
        for sw,st,v,par in items:
            if (sw,st,v) in done:continue
            key=tuple(par[k] for k in ('xi','sF','M','p','kappa','rho'))
            if key!=last_key:
                cached=None
                mesh=X.table('PPPM-grid-PSD',cfg['L'],par)
                cached=(mesh,X.long_dense('PPPM-grid-PSD',mesh,cfg['q']))
                last_key=key
            row,_,_,_=X.measure('PPPM-grid-PSD',cfg,par,structure=True,actions=True,mesh_only=True,cached=cached)
            rows.append(dict(sweep=sw,setting=st,value=v,source='new_run',**row))
            X.write(path,rows)
    X.log(f'grid-PSD sweeps {name}: {len(rows)} rows')
    return rows

def sweeps(workers):
    pts=A.sweep_points(); groups={}
    for sw,st,v,p in pts:
        p=X.parameters(**{k:p[k] for k in ('xi','sR','sF','M','p','kappa','rho')})
        for f in E.FAMS:
            for s in E.SEEDS:groups.setdefault((f,s,p['rho'],p['kappa']),[]).append((sw,st,v,p))
    rows=import_history()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(group_worker,j) for j in groups.items()]
        for future in as_completed(futures):
            rows+=future.result();X.write(X.RESULTS/'sweeps.csv',rows)
    X.jsave(X.RESULTS/'sweep_provenance.json',E.provenance(workers=workers,rows=len(rows),
             note='Historical accuracy rows read-only, new projected path evaluated on identical references.'))

def convergence_worker(job):
    f,s=job;cfg=E.config(f,s,.75)
    path=X.RESULTS/'convergence_parts'/f'{f}_s{s}.csv';path.parent.mkdir(exist_ok=True)
    rows=X.read(path);done={(r['method'],r['M'],r['p']) for r in rows}
    with X.threadpool_limits(1):
        for p in (2,3,5,7,9):
            for M in ((24,27,30,32,40,48,64,80) if p>=5 else (24,32,40,48,64,80)):
                par=X.parameters(xi=.85,sR=8.,sF=6.,M=M,p=p,kappa=.7)
                for method in X.METHODS:
                    if (method,M,p) in done:continue
                    row,_,_,_=X.measure(method,cfg,par,mesh_only=True)
                    rows.append(row);X.write(path,rows)
    X.log(f'convergence {f} s{s}: {len(rows)} rows')
    return rows

def convergence(workers):
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(convergence_worker,(f,s)) for f in E.FAMS for s in E.SEEDS]
        for fut in as_completed(futures):rows+=fut.result();X.write(X.RESULTS/'convergence.csv',rows)
    fits=[]
    for method in X.METHODS:
        for f in E.FAMS:
            for p in (2,3,5,7,9):
                rr=[r for r in rows if r['method']==method and r['family']==f and r['p']==p]
                # Exclude the independently bounded direct-reference/roundoff floor,
                # and fit only consecutive decreasing errors above 1e-11.
                vv=[]
                for M in sorted({r['M'] for r in rr}):
                    er=float(np.median([r['E_mesh'] for r in rr if r['M']==M]))
                    if er>1e-11:vv.append((M,er))
                vv=vv[-4:]
                if len(vv)>=3 and all(vv[i+1][1]<vv[i][1] for i in range(len(vv)-1)):
                    q=-np.polyfit(np.log([v[0] for v in vv]),np.log([v[1] for v in vv]),1)[0]
                    chosen=[r for r in rr if vv[0][0]<=r['M']<=vv[-1][0]]
                    fits.append(dict(method=method,family=f,p=p,Mlo=vv[0][0],Mhi=vv[-1][0],order=q,n=len(vv),floor=1e-11,
                        max_E_proj=max(r.get('E_proj') or 0. for r in chosen),
                        inactive_rows=sum((r.get('neg_fraction') or 0.)==0 for r in chosen)))
    X.write(X.RESULTS/'mesh_rate_fits.csv',fits)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['sweeps','convergence']);ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    (sweeps if args.stage=='sweeps' else convergence)(args.workers)
