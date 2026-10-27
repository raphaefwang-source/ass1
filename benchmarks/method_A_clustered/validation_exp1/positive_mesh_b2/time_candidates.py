"""Common serial timing of validated B2 candidates and original PPPM.

Run only after all reference, sweep and search processes have finished.
"""
import argparse
from pathlib import Path
import hashlib
import json
import platform
import random
import subprocess
import experiment as X
from experiment import np,E,C,time
from positive_mesh import PositiveMesh
from exp1_dense_reference import mode_pair_matrix_fast
from test_parameter_selection import lap_full

ROUNDS,REPS=5,7

def original(cfg,mesh=None):
    K=E.kernel(.2,1.25,0.)
    rc,kc,_=E.cutoffs(X.OLD,cfg['L'])
    if mesh is None:
        mesh,info=E.MeshCache().get(cfg['L'],K,kc,16,5)
        assert info['valid']
    return E.ThetaOp(cfg['q'],cfg['L'],K,rc,mesh)

def build(cfg,r,mesh=None):
    if r['method']=='PPPM':return original(cfg,mesh)
    return X.build(cfg,{k:r[k] for k in ('xi','sR','sT','M','p','kappa','rho')},mesh)

def parameters():
    good=[dict(r,method='B2') for r in X.read(X.RESULTS/'validation_summary.csv') if r['feasible']]
    # Re-rank every fully validated feasible point after all accuracy workers
    # finish. This removes the load difference between coarse and local search.
    cfg=X.configurations()[0];raw=[];rng=np.random.default_rng(61026)
    v=rng.normal(size=(256,3))
    for rnd in range(3):
        for ix in rng.permutation(len(good)):
            r=good[ix];op=build(cfg,r)
            for rep,dt in enumerate(C.median_time(lambda:op.apply(v),3,warm=3)):
                raw.append(dict(id=r['id'],round=rnd,rep=rep,seconds=dt))
        X.write(X.RESULTS/'serial_prescreen_raw.csv',raw)
        X.log(f'serial re-ranking round {rnd+1}/3, {len(good)} feasible candidates')
    rank={r['id']:float(np.median([v['seconds'] for v in raw if v['id']==r['id']])) for r in good}
    good.sort(key=lambda r:rank[r['id']])
    # Long common timing covers the 12 fastest serial incumbents and best of
    # each order. Other measured points remain in the search and rank data.
    selected=good[:12]
    for p in sorted({r['p'] for r in good}):selected.append(next(r for r in good if r['p']==p))
    good=list({r['id']:r for r in selected}.values())
    good.append(dict(X.OLD,id='original_PPPM',method='PPPM',sT=0.))
    return good

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rounds',type=int,default=ROUNDS);ap.add_argument('--reps',type=int,default=REPS)
    args=ap.parse_args()
    if (X.RESULTS/'final_timing.csv').exists():
        raise RuntimeError('Timing already exists. Archive the timing outputs before starting a fresh pass.')
    with X.threadpool_limits(1):pars=parameters()
    rng=np.random.default_rng(20261026)
    started=time.time();raw=[];setup=[];table=[];baseline=[]
    with X.threadpool_limits(1):
        # Warm common kernel primitives before measuring setup for either method.
        for r in pars:E.kernel(.2,r['xi'],0.)
        order=rng.permutation(len(pars))
        cfg0=X.configurations()[0]
        for ix in order:
            r=pars[ix]
            for rep in range(3):
                t=time.perf_counter()
                if r['method']=='PPPM':
                    K=E.kernel(.2,1.25,0.)
                    _,kc,_=E.cutoffs(X.OLD,cfg0['L'])
                    mesh,info=E.MeshCache().get(cfg0['L'],K,kc,16,5)
                    assert info['valid']
                    elapsed=time.perf_counter()-t
                else:
                    mesh=PositiveMesh(cfg0['L'],E.kernel(.2,r['xi'],0.),r['M'],r['p'],r['sT'])
                    elapsed=time.perf_counter()-t
                table.append(dict(id=r['id'],method=r['method'],rep=rep,seconds=elapsed))
        X.write(X.RESULTS/'table_setup_raw.csv',table)
        for cfg in X.configurations():
            # Same physical independent reference for baseline validation.
            old=original(cfg);G,nref=X.ref(cfg,.2)
            Gold=E.real_dense(old)+lap_full(mode_pair_matrix_fast(old.mesh,cfg['q']))
            baseline.append(dict(config=cfg['name'],E_G=X.norm(Gold-G)/nref))
            v=rng.normal(size=(256,3));v-=v.mean(0)
            for rnd in range(args.rounds):
                for ix in rng.permutation(len(pars)):
                    r=pars[ix];op=build(cfg,r)
                    for _ in range(3):op.apply(v)
                    for rep in range(args.reps):
                        op.apply(v)  # one warmup immediately before each sample
                        t=time.perf_counter_ns();op.apply(v);dt=(time.perf_counter_ns()-t)*1e-9
                        raw.append(dict(id=r['id'],method=r['method'],config=cfg['name'],round=rnd,rep=rep,seconds=dt))
                X.jsave(X.RESULTS/'timing_progress.json',dict(config=cfg['name'],round=rnd+1,
                          rounds=args.rounds,samples=len(raw),elapsed_s=time.time()-started))
            for ix in rng.permutation(len(pars)):
                r=pars[ix];op=build(cfg,r)
                for rep in range(3):
                    t=time.perf_counter();newop=build(cfg,r,op.mesh);dt=time.perf_counter()-t
                    setup.append(dict(id=r['id'],method=r['method'],config=cfg['name'],rep=rep,seconds=dt))
            X.write(X.RESULTS/'final_timing_raw.csv',raw);X.write(X.RESULTS/'config_setup_raw.csv',setup)
            X.write(X.RESULTS/'original_pppm_revalidation.csv',baseline)
            X.log(f'common serial timing {cfg["name"]} complete, elapsed {time.time()-started:.1f}s')
        summary=[]
        for r in pars:
            ts=[v for v in raw if v['id']==r['id']]
            cms=[np.median([v['seconds'] for v in ts if v['config']==c['name']]) for c in X.configurations()]
            sts=[np.median([v['seconds'] for v in setup if v['id']==r['id'] and v['config']==c['name']]) for c in X.configurations()]
            iqrs=[C.tstats([v['seconds'] for v in ts if v['config']==c['name']])['iqr_rel'] for c in X.configurations()]
            summary.append(dict(r,median_matvec_s=float(np.median(cms)),min_config_median_s=float(min(cms)),
                                max_config_median_s=float(max(cms)),max_relative_iqr=float(max(iqrs)),
                                median_config_setup_s=float(np.median(sts)),
                                median_table_setup_s=float(np.median([v['seconds'] for v in table if v['id']==r['id']])),
                                n_samples_per_config=args.rounds*args.reps))
        summary.sort(key=lambda r:r['median_matvec_s']);X.write(X.RESULTS/'final_timing.csv',summary)
        best=next(r for r in summary if r['method']=='B2')
        # Component timings use the same methods, after common timing has ended.
        components=[]
        for r in (best,next(r for r in summary if r['method']=='PPPM')):
            for cfg in X.configurations():
                op=build(cfg,r);v=rng.normal(size=(256,3))
                grids=op.spread(v);phi=op.fft_influence(grids)
                functions={
                    'real':lambda:np.einsum('nab,nb->na',op.DS,v)-op.real_H(v),
                    'spread_gather':lambda:(op.spread(v),op.gather(phi)),
                    'fft_influence':lambda:op.fft_influence(grids),
                    'mesh':lambda:np.einsum('nab,nb->na',op.Dm,v)-op.mesh.apply(op.idx,op.wt,v)}
                np.testing.assert_allclose(op.apply_staged(v),op.apply(v),atol=1e-11,rtol=1e-12)
                for phase,fn in functions.items():
                    for rep,dt in enumerate(C.median_time(fn,15,warm=3)):
                        components.append(dict(id=r['id'],method=r['method'],config=cfg['name'],phase=phase,rep=rep,seconds=dt))
        X.write(X.RESULTS/'components_raw.csv',components)
        kernels=[]
        for xi in sorted({r['xi'] for r in pars}):
            for rep in range(3):
                t=time.perf_counter();K=E.ThetaKernel(.2,xi,0.)
                kernels.append(dict(xi=xi,rep=rep,seconds=time.perf_counter()-t))
        X.write(X.RESULTS/'common_kernel_setup_raw.csv',kernels)
        # Paired block bootstrap, conditioned on the measured candidate choice.
        oldid='original_PPPM';blocks=[]
        for cfg in X.configurations():
            for rnd in range(args.rounds):
                b=np.median([r['seconds'] for r in raw if r['id']==best['id'] and r['config']==cfg['name'] and r['round']==rnd])
                o=np.median([r['seconds'] for r in raw if r['id']==oldid and r['config']==cfg['name'] and r['round']==rnd])
                blocks.append(b/o)
        ratios=[np.median(rng.choice(blocks,len(blocks),replace=True)) for _ in range(5000)]
        X.jsave(X.RESULTS/'timing_protocol.json',dict(rounds=args.rounds,reps=args.reps,threads=1,
            started_unix=started,elapsed_s=time.time()-started,randomization='candidate blocks randomly ordered each round',
            warmup='3 per block, 1 before each sample',objective='median over nine configuration medians',
            b2_over_pppm_block_bootstrap_95=np.percentile(ratios,[2.5,97.5]).tolist(),
            note='conditional on candidate selection; other system load is not controlled',
            setup='3 fresh configuration builds per state; 3 fresh parameter tables; kernel primitives warmed',
            provenance=E.provenance(platform=platform.platform(),threadpool=X.threadpool_info())))
        X.jsave(X.RESULTS/'best_configuration.json',best)
        X.log(f'best measured B2 {best["id"]}: {1000*best["median_matvec_s"]:.3f} ms')

if __name__=='__main__':main()
