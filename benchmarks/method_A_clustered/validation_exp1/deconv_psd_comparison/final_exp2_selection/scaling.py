"""Optional N scaling with NEW independent unsplit references, never old writes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,time
from pathlib import Path
import protocol as P
np=P.np;S=P.S;E=P.E
REFS=P.HERE/'scaling_references'


def configuration(N):
    base=E.config('uniform',2,.75)
    if N==256:return base
    L=(N/.75)**(1/3)
    q=np.mod(base['q'][:N]*(L/base['L']),L)
    return dict(name=f'uniform_N{N}_s2_nested_rho0.75',family='uniform',seed=2,rho=.75,N=N,L=L,q=q,
        hash=hashlib.sha256(np.concatenate((q.ravel(),[L])).tobytes()).hexdigest()[:12],
        d_min=E.min_periodic_separation(q,L),generation='nested subset of committed uniform_s2, box-scaled at fixed density; one configuration only')


def reference(N):
    cfg=configuration(N)
    if N==256:return cfg,S.ref(cfg,.2)
    path=REFS/f'N{N}_uniform_s2.npz'
    meta=json.loads(path.with_suffix('.json').read_text())
    if meta['position_hash']!=cfg['hash']:raise RuntimeError('Scaling reference geometry mismatch')
    with np.load(path) as z:G=z['G_D'].copy()
    return cfg,(G,meta['norm2'])


def compute_reference(N):
    cfg=configuration(N);REFS.mkdir(exist_ok=True)
    path=REFS/f'N{N}_uniform_s2.npz'
    if path.exists():
        _,(_,norm)=reference(N);P.log(f'reused N={N} reference norm={norm:g}');return
    R=190.;start=time.perf_counter()
    P.log(f'independent unsplit reference N={N}, R={R:g} started')
    pi,pj,acc,count=E.DP.annulus_tensors(cfg['q'],cfg['L'],[R],g=E.g_phys(.2),budget=1<<18)
    G=E.DP.tensors_to_dense(pi,pj,acc[0],N)
    elapsed=time.perf_counter()-start;norm=S.norm(G)
    bound=float(2*E.DP.tail_bound_per_particle(N,cfg['L'],R,g=E.g_phys(.2))/norm)
    if bound>1e-10:raise RuntimeError('Scaling reference tail exceeds requested 1e-10')
    temp=path.with_suffix('.tmp.npz');np.savez(temp,G_D=G,q=cfg['q'],L=cfg['L']);temp.replace(path)
    P.save(path.with_suffix('.json'),dict(N=N,L=cfg['L'],rho=.75,kappa=.2,amplitude=1.,
        position_hash=cfg['hash'],generation=cfg['generation'],R_D=R,norm2=norm,
        cert_bound_rel=bound,t_direct_s=elapsed,n_images=int(np.sum(count)),
        source='fresh independent unsplit direct periodic annulus_tensors; own scaling_references directory',
        setup_timing_role='reference-generation wall time, concurrent accuracy phase; not mixed into formal same-batch setup comparisons'))
    P.log(f'new reference N={N}: {elapsed:.1f}s, tail/norm={bound:.3g}')


def smooth_at_least(n):
    n=max(2,int(np.ceil(n)))
    while True:
        a=n
        for factor in (2,3,5):
            while a%factor==0:a//=factor
        if a==1:return n
        n+=1


def run():
    target=P.RESULTS/'scaling.csv'
    if target.exists():raise FileExistsError('Preserve existing scaling batch before repeating')
    best=[r for r in P.read(P.RESULTS/'best_by_epsilon.csv') if r.get('available') and r.get('epsilon')==1e-4 and r['method'] in P.METHODS]
    rng=np.random.default_rng(20261028);rows=[];raw=[]
    for N in (64,128,256):
        cfg,ref=reference(N);G,nref=ref;ops={};states={}
        for b in best:
            par=P.parameters(b);par['M']=smooth_at_least(max(par['p'],par['M']*cfg['L']/configuration(256)['L']))
            status=None
            for trial in range(4):
                cand=P.candidate(b['method'],par,epsilon=1e-4,stage='N-scaling',origin='rescale best main-target h then increase M if needed')
                check=P.evaluate(cand,cfg,proxy=False,reference=ref)
                if check['status']=='ok' and check['E_G']<=1e-4:
                    mesh=S.table(b['method'],cfg['L'],par);ops[b['method']]=S.build(b['method'],cfg,par,mesh)
                    states[b['method']]=check;status='ok';break
                par['M']=smooth_at_least(par['M']+1)
                if par['M']>P.MAX_M:break
            if status!='ok':
                rows.append(dict(method=b['method'],N=N,epsilon=1e-4,available=False,status='no feasible h-rescaled candidate in four mesh refinements',max_E_G=check.get('E_G')))
        vector=rng.normal(size=(N,3));vector-=vector.mean(0);flat=vector.ravel()
        methods=list(ops)+['direct-dense']
        for rnd in range(3):
            for method in rng.permutation(methods):
                fn=(lambda:G@flat) if method=='direct-dense' else (lambda op=ops[method]:op.apply(vector))
                for _ in range(3):fn()
                for rep in range(7):
                    fn();t=time.perf_counter_ns();fn();seconds=(time.perf_counter_ns()-t)*1e-9
                    raw.append(dict(N=N,method=method,config=cfg['name'],round=rnd,rep=rep,seconds=seconds,batch='isolated_N_scaling_single_config',threads=1))
        for method in methods:
            rr=[r['seconds'] for r in raw if r['N']==N and r['method']==method]
            base=states.get(method,{})
            rows.append(dict(base,method=method,N=N,epsilon=1e-4,available=True,status='single nested uniform configuration empirical pass',
                             max_E_G=base.get('E_G',0.),median_matvec_s=float(np.median(rr)),
                             q25_matvec_s=float(np.quantile(rr,.25)),q75_matvec_s=float(np.quantile(rr,.75)),n_configs=1,
                             direct_dense_error_role='zero against its own independent numerical reference' if method=='direct-dense' else None))
        P.write(P.RESULTS/'scaling_raw.csv',raw);P.write(target,rows)
        P.log(f'N={N} scaling measured')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('refs','run'));a=parser.parse_args()
    if a.stage=='refs':
        with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(compute_reference,(64,128)))
    else:run()


if __name__=='__main__':main()
