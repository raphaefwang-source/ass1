"""Resumable theory-informed measured search, complete operator norm constraint.

Timing here is a search heuristic only. run_timing.py ranks feasible points
after all accuracy workers end. No physical parameter changes are allowed.
"""
import argparse,itertools,random
import study as X
from study import np,E

def pfrom(r):return X.parameters(**{k:r[k] for k in ('xi','sR','M','p','sF','sT','kappa','rho') if k in r and r[k] is not None})

def seeds(method):
    if method=='B2-positive':
        return [X.parameters(xi=.35,sR=3.1,M=72,p=2),X.parameters(xi=.325,sR=3.2,M=64,p=2),
                X.parameters(xi=.375,sR=3.3,M=75,p=2)]
    return [X.parameters(),X.parameters(xi=1.,sR=3.,sF=3.,M=20,p=5),
            X.parameters(xi=.7,sR=3.2,sF=3.5,M=24,p=5)]

def coarse(method):
    pts=seeds(method)
    if method=='B2-positive':
        # Positive averaging observed q~2; allow xi/mesh tradeoffs about old best.
        for xi,M in ((.25,48),(.3,60),(.35,72),(.4,80),(.45,90),(.5,100)):
            for p,sR in itertools.product((2,3),(2.9,3.2,3.5)):
                pts.append(X.parameters(xi=xi,sR=sR,M=M,p=p))
        pts += [X.parameters(xi=.35,sR=3.1,M=72,p=2,sT=t) for t in (16,24,30,42)]
    else:
        # Finite sR/sF Gaussian tails motivate scales sqrt(log(1/epsilon)).
        # Grid-PSD must directly measure correction; this grid imposes no h^p law.
        for xi,M in ((.35,16),(.6,16),(.85,16),(1.,16),(1.25,16),(1.5,20),(1.75,24)):
            for p,s in itertools.product((3,5,7),(2.8,3.3)):
                pts.append(X.parameters(xi=xi,sR=s,sF=s,M=M,p=p))
    # More accurate measured points for optional accuracy/cost tiers.
    if method=='B2-positive':
        pts += [X.parameters(xi=.25,sR=3.8,M=96,p=2),X.parameters(xi=.2,sR=4.,M=128,p=2)]
    else:
        pts += [X.parameters(xi=1.,sR=s,sF=s,M=M,p=p) for s,M,p in ((3.8,24,5),(4.2,32,7),(4.8,40,9))]
    return pts

def theoretical_record():
    conv=X.read(X.RESULTS/'mesh_rate_fits.csv')
    proj=[r for r in X.read(X.RESULTS/'convergence.csv') if r['method']=='PPPM-grid-PSD']
    X.jsave(X.RESULTS/'search_theory.json',dict(
        epsilon=1e-4,fixed_physics=dict(N=256,rho=.75,kappa=.2,c=1),
        sR_initial=float(np.sqrt(np.log(1e4))),
        original='Gaussian cutoff tails and measured high-order mesh slopes; inspect actual complete E_G.',
        B2='Measured positive-assignment second-order bias, initialized at historical best.',
        projected='E_total <= E_original + E_proj (triangle inequality); projection active region measured explicitly. No assumed h^p or h^2 error law. Coarse xi/M/p grid, then measured constrained local and boundary searches.',
        mesh_rate_fits=conv,projection_calibration_rows=len(proj),
        validation='screen uniform seed2, then all nine; criterion max actual spectral operator norm E_G <= epsilon',
        prior_budgets='Historical original extensive optimization and B2 355 candidates retained as initial knowledge. This follow-up gives similar new independent candidate budgets; exact counts saved.',
        resources='local hardware only, M<=128; invalid points retained; no timeout censoring of successful points'))

def screen(method,pts,stage):
    path=X.RESULTS/'search.csv';rows=X.read(path);done={r['id'] for r in rows}
    cfg=X.configurations()[0]
    with X.threadpool_limits(1):
        for par in pts:
            id_=X.ident(method,par)
            if id_ in done:continue
            r,op,_,_=X.attempt(method,cfg,par)
            r.update(stage=stage,status='ok' if op is not None else r['status'])
            if op is not None:
                v=np.random.default_rng(822).normal(size=(256,3))
                r['t_screen_s']=float(np.median(X.C.median_time(lambda:op.apply(v),3,warm=1)))
                r['screen_pass']=r['E_G']<=1e-4
            rows.append(r);done.add(id_);X.write(path,rows)
            X.log(f'{stage} {id_}: '+(f'E={r["E_G"]:.3g}' if op is not None else r['failure']))
    return [r for r in rows if r['method']==method]

def validate(candidates):
    path=X.RESULTS/'validation.csv';rows=X.read(path);done={(r['id'],r['config']) for r in rows}
    with X.threadpool_limits(1):
        for cand in candidates:
            method=cand['method'];par=pfrom(cand);id_=cand['id']
            for cfg in X.configurations():
                if (id_,cfg['name']) in done:continue
                r,op,_,_=X.attempt(method,cfg,par,structure=True,actions=True)
                r['status']='ok' if op is not None else r['status']
                rows.append(r);done.add((id_,cfg['name']));X.write(path,rows)
            rr=[r for r in rows if r['id']==id_]
            X.log(f'validate {id_}: maxE={max(r.get("E_G",float("inf")) or float("inf") for r in rr):.3g}')
    summary=[]
    screened={r['id']:r for r in X.read(X.RESULTS/'search.csv')}
    for id_ in sorted({r['id'] for r in rows}):
        rr=[r for r in rows if r['id']==id_]
        if len(rr)!=9:continue
        r=rr[0];valid=all(v.get('status')=='ok' for v in rr)
        summary.append(dict(method=r['method'],id=id_,**pfrom(r),max_E_G=max(v['E_G'] for v in rr) if valid else None,
            t_screen_s=screened.get(id_,{}).get('t_screen_s'),
            max_E_GX=max(v['E_GX'] for v in rr) if valid else None,n_configs=9,status='ok' if valid else 'failed',
            feasible=valid and max(v['E_G'] for v in rr)<=1e-4,
            max_E_proj=max(v.get('E_proj') or 0 for v in rr),
            min_eigenvalue=min(v.get('lam_min_h') or 0 for v in rr),
            min_long_eigenvalue=min(v.get('lam_min_long') or 0 for v in rr)))
    X.write(X.RESULTS/'validation_summary.csv',summary)
    return summary

def select(rows,n=10):
    good=[r for r in rows if r.get('status')=='ok' and r['E_G']<=1e-3]
    main=sorted([r for r in good if r['E_G']<=1e-4],key=lambda r:r['t_screen_s'])[:n]
    for eps in (1e-3,1e-5,1e-6):
        tier=[r for r in good if r['E_G']<=eps]
        main+=sorted(tier,key=lambda r:r['t_screen_s'])[:2]
    return list({r['id']:r for r in main}.values())

def local(method,summary,boundary=False):
    search={r['id']:r for r in X.read(X.RESULTS/'search.csv')}
    feasible=[r for r in summary if r['method']==method and r.get('feasible')]
    feasible.sort(key=lambda r:search[r['id']]['t_screen_s'])
    if not feasible:return []
    pts=[]
    for r in feasible[:2]:
        par=pfrom(r)
        changes=({'xi':round(r['xi']+a,5)} for a in (-.05,.05))
        for change in changes:pts.append(dict(par,**change))
        for a in (-.1,.1,-.2,.2):pts.append(dict(par,sR=round(r['sR']+a,5)))
        for M in (r['M']-2,r['M']+2,r['M']-4,r['M']+4):
            if M>=r['p']:pts.append(dict(par,M=M))
        for p in (max(2,r['p']-1),r['p']+1):pts.append(dict(par,p=p))
        key='sT' if method=='B2-positive' else 'sF'
        for a in ((-6,6) if key=='sT' else (-.2,.2)):pts.append(dict(par,**{key:r[key]+a}))
        if boundary:
            for dx,ds,dm in itertools.product((-.025,.025),(-.05,.05),(-1,1)):
                pts.append(dict(par,xi=round(r['xi']+dx,5),sR=round(r['sR']+ds,5),M=r['M']+dm))
    return pts

def main(stage,method):
    theoretical_record()
    if stage=='coarse':rows=screen(method,coarse(method),stage)
    else:rows=screen(method,local(method,X.read(X.RESULTS/'validation_summary.csv'),stage=='boundary'),stage)
    validate(select(rows,10 if stage=='coarse' else 14))
    rows=X.read(X.RESULTS/'search.csv')
    X.jsave(X.RESULTS/'search_budget.json',{m:dict(independent_candidates=sum(r['method']==m for r in rows),
        failures=sum(r['method']==m and r['status']!='ok' for r in rows),
        stages={s:sum(r['method']==m and r['stage']==s for r in rows) for s in ('coarse','local','boundary')}) for m in X.METHODS})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['coarse','local','boundary']);ap.add_argument('--method',choices=X.METHODS)
    a=ap.parse_args()
    for method in ([a.method] if a.method else X.METHODS):main(a.stage,method)
