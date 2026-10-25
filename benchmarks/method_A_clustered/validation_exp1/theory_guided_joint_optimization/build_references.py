#!/usr/bin/env python3
"""Build only the missing nine kappa=.2 c=1 independent unsplit references."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import time
from pathlib import Path
import experiment as X
from experiment import E,C,np,threadpool_limits

def one(name):
    cfg=next(c for c in X.configs() if c['name']==name)
    path=E.ref_path(cfg,X.KAPPA)
    previous=next(r for r in E.read_csv(X.HERE.parent/'accuracy_sweeps/results/references.csv')
                  if r['config']==name and r['kappa']==X.KAPPA and r['rho']==.75)
    with threadpool_limits(limits=1):
        if path.exists():
            X.reference(name)  # Reject a mismatched or unaudited cache.
            meta=json.loads(path.with_suffix('.json').read_text())
            X.log(f'cache valid {name}')
        else:
            R=E.choose_RD(256,cfg['L'],X.KAPPA,previous['norm2'])
            assert R==190.
            edges=np.round(R*np.array([.4,.5,.6,.7,.8,.9,.95,1.]),6)
            X.log(f'direct start {name} R={R:g}; historical time={previous["t_direct_s"]:.1f}s')
            start=time.perf_counter()
            i,j,acc,count=E.DP.annulus_tensors(cfg['q'],cfg['L'],edges,budget=1<<20,g=E.g_phys(X.KAPPA))
            elapsed=time.perf_counter()-start
            cum=np.cumsum(acc,axis=0)
            G=E.DP.tensors_to_dense(i,j,cum[-1],256)
            norm=E.norm2_sym(G)[0]
            conv=np.array([[radius,E.norm2_sym(E.DP.tensors_to_dense(i,j,cum[k],256)-G)[0]/norm,
                            2*E.DP.tail_bound_per_particle(256,cfg['L'],radius,g=E.g_phys(X.KAPPA))/norm]
                           for k,radius in enumerate(edges)])
            assert conv[-1,2]<=E.REF_CERT_TOL
            assert abs(norm/previous['norm2']-1)<1e-11
            path.parent.mkdir(exist_ok=True,parents=True)
            tmp=path.with_suffix('.tmp.npz')
            np.savez(tmp,G_D=G,conv=conv)
            tmp.replace(path)
            meta=dict(config=name,kappa=X.KAPPA,c=1.,rho=.75,N=256,L=cfg['L'],position_hash=cfg['hash'],
                      source='unsplit direct periodic annulus_tensors',R_D=R,t_direct_s=elapsed,
                      norm2=norm,cert_bound_rel=float(conv[-1,2]),n_images=int(count.sum()),
                      historical_norm_relative_difference=norm/previous['norm2']-1,
                      historical_images=previous['n_images_direct'])
            X.jsave(path.with_suffix('.json'),meta)
            X.log(f'direct done {name} time={elapsed:.2f}s cert={conv[-1,2]:.3g}')
        return dict(meta,cache_file=path.name,cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    X.physical_audit()
    X.jsave(X.RESULTS/'reference_provenance.json',X.provenance(
        workers=args.workers,each_worker_threads=1,
        assessment='No kappa=.2 cache found in checked project copies; historical cost ~430s/config, R=190.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(one,c['name']) for c in X.configs()]
        for f in as_completed(fs):
            rows.append(f.result())
            X.write(X.RESULTS/'references.csv',rows)
    assert len(rows)==9
    X.log('all nine independent kappa=.2 references complete')

if __name__=='__main__': main()
