"""Reuse keyed unsplit references; build only missing physical state points."""
import argparse
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import experiment as X
from experiment import E,C,np,time

def jobs():
    return sorted({(f,s,.75,k) for f in E.FAMS for s in E.SEEDS for k in (.2,.5,.7,1.,2.)}
                  | {(f,s,r,.7) for f in E.FAMS for s in E.SEEDS for r in (.3,.5,.75,1.)})

def one(job):
    f,s,rho,kappa=job
    cfg=E.config(f,s,rho)
    path=X.reference_path(cfg,kappa)
    historical=next(r for r in X.read(X.HERE.parent/'accuracy_sweeps/results/references.csv')
                    if r['config']==cfg['name'] and r['kappa']==kappa)
    with X.threadpool_limits(1):
        if not path.exists():
            R=E.choose_RD(256,cfg['L'],kappa,historical['norm2'])
            edges=R*np.array([.6,.8,.9,.95,1.])
            t=time.perf_counter()
            pi,pj,acc,count=E.DP.annulus_tensors(cfg['q'],cfg['L'],edges,budget=1<<20,g=E.g_phys(kappa))
            cum=np.cumsum(acc,0)
            G=E.DP.tensors_to_dense(pi,pj,cum[-1],256)
            norm=X.norm(G)
            conv=np.array([[rad,X.norm(E.DP.tensors_to_dense(pi,pj,cum[k],256)-G)/norm,
                            2*E.DP.tail_bound_per_particle(256,cfg['L'],rad,g=E.g_phys(kappa))/norm]
                           for k,rad in enumerate(edges)])
            assert conv[-1,2]<=1e-13 and abs(norm/historical['norm2']-1)<1e-10
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.tmp.npz')
            np.savez(tmp,G_D=G,conv=conv)
            tmp.replace(path)
            meta=dict(config=cfg['name'],kappa=kappa,c=1.,rho=rho,N=256,L=cfg['L'],position_hash=cfg['hash'],
                      source='unsplit direct periodic annulus_tensors',R_D=R,t_direct_s=time.perf_counter()-t,
                      norm2=norm,cert_bound_rel=float(conv[-1,2]),n_images=int(count.sum()))
            X.jsave(path.with_suffix('.json'),meta)
            X.log(f'reference built {cfg["name"]} kap={kappa} R={R}')
        G,norm=X.ref(cfg,kappa)
        meta=json.loads(path.with_suffix('.json').read_text())
        assert abs(norm/historical['norm2']-1)<1e-10
        return dict(meta,config=cfg['name'],family=f,seed=s,rho=rho,kappa=kappa,position_hash=cfg['hash'],
                    file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(one,jobs()):
            rows.append(r)
            X.write(X.RESULTS/'references.csv',rows)
    X.jsave(X.RESULTS/'reference_provenance.json',X.E.provenance(workers=args.workers,source='unsplit direct periodic only',
                                                             reference_count=len(rows)))
    X.log(f'all {len(rows)} independent direct references ready')

if __name__=='__main__':main()
