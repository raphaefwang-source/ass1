"""Experiment 1 B2 accuracy sweeps; resume by physical-state group."""
import argparse
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import experiment as X
from experiment import E,C,np
from positive_mesh import table_tail_bound

M_BASE=(24,25,27,30,32,36,40,45,48,50,54,60,64,72,75,80)
M_MOD=(16,18,20,24,25,27,30,32,36,40,45,48,54,60,64,72,80)

def points():
    out=[]
    for s in np.arange(2,6.50001,.25):out.append(('sR','baseline',float(s),dict(X.BASE,sR=float(s))))
    for s in range(2,39,2):out.append(('sT','baseline',s,dict(X.BASE,sT=float(s))))
    for name,B,ms in (('baseline',X.BASE,M_BASE),('moderate',X.MODERATE,M_MOD)):
        for key,grid in [('M',ms),('p',range(3,10)),('rho',(.3,.5,.75,1.)),('kappa',(.2,.5,.7,1.,2.))]:
            for v in grid:out.append((key,name,v,dict(B,**{key:v})))
    return out

def groups():
    out={}
    for sw,st,v,par in points():
        for f in E.FAMS:
            for s in E.SEEDS:
                out.setdefault((f,s,par['rho'],par['kappa']),[]).append((sw,st,v,par))
    return out

def one(job):
    key,items=job
    f,s,rho,kappa=key
    name=f'{f}_s{s}_rho{rho:g}_kap{kappa:g}'
    path=X.RESULTS/'sweeps_parts'/f'{name}.csv'
    old=X.read(path)
    if len(old)==len(items):return old
    cfg=E.config(f,s,rho)
    G,nref=X.ref(cfg,kappa)
    rows=[];done={}
    with X.threadpool_limits(1):
        for sw,st,v,par in items:
            pk=tuple(sorted(par.items()))
            if pk not in done:
                row,op,Gh,GLh=X.evaluate(cfg,par,structure=True,actions=True)
                GSfar=X.short_dense(f,s,rho,kappa,par['xi'],8.)
                row['E_G_mesh_only']=X.norm(GLh-(G-GSfar))/nref
                row['E_R']=X.norm(E.real_dense(op)-GSfar)/nref
                row['table_image_bound_rel']=2*(cfg['N']-1)*table_tail_bound(cfg['L'],kappa,op.mesh.R)/nref
                done[pk]=row
            rows.append(dict(sweep=sw,setting=st,value=v,**done[pk]))
    X.write(path,rows)
    X.log(f'sweeps {name}: {len(rows)} rows')
    return rows

def sweeps(workers,available_only=False):
    todo=[]
    for key,items in groups().items():
        f,s,rho,k=key
        if available_only and not X.reference_path(E.config(f,s,rho),k).exists():continue
        todo.append((key,items))
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rs in pool.map(one,todo):
            rows+=rs
            X.write(X.RESULTS/'sweeps.csv',rows)
    X.log(f'sweeps total {len(rows)} / 927')
    X.jsave(X.RESULTS/'sweep_provenance.json',E.provenance(workers=workers,rows=len(rows),threadpool=X.threadpool_info(),
                                                        note='accuracy only; final performance measured serially separately'))

def fits():
    rows=X.read(X.RESULTS/'sweeps.csv');out=[]
    for st in ('baseline','moderate'):
        for fam in E.FAMS:
            for metric in ('E_G_mesh_only','E_G','E_GX'):
                rs=[r for r in rows if r['sweep']=='M' and r['setting']==st and r['family']==fam]
                vals=sorted({r['M'] for r in rs})
                if len(vals)<4:continue
                for subset,xx in [('full',vals),('finest_half',vals[len(vals)//2:])]:
                    yy=[np.median([r[metric] for r in rs if r['M']==m]) for m in xx]
                    slope,intercept=np.polyfit(np.log(xx),np.log(yy),1)
                    out.append(dict(setting=st,family=fam,metric=metric,fit_range=subset,Mlo=min(xx),Mhi=max(xx),
                                    order=-slope,coefficient=np.exp(intercept),n=len(xx)))
    X.write(X.RESULTS/'mesh_rate_fits.csv',out)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stages',nargs='+',choices=['sweeps','fits'])
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--available-only',action='store_true')
    args=ap.parse_args()
    for stage in args.stages:
        if stage=='sweeps':sweeps(args.workers,args.available_only)
        if stage=='fits':fits()

if __name__=='__main__':main()
