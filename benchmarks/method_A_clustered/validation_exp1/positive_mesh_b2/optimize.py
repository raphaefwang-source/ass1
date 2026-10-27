"""Theory initialization -> measured B2 bias -> constrained timed local search."""
import argparse
from functools import lru_cache
import experiment as X
from experiment import np,E,C,time
from test_equal_accuracy_xi import next_smooth

XIS=(.2,.3,.4,.5,.6,.7,.85,1.,1.25)

def calibrated_orders():
    return [(xi,p) for xi in XIS for p in (2,3)] + [(xi,4) for xi in (.3,.5,.7)]

def measure(cfg,par,mesh_error=False):
    start=time.perf_counter()
    row,op,Gh,GLh=X.evaluate(cfg,par)
    G,nref=X.ref(cfg,.2)
    if mesh_error:
        GSfar=X.short_dense(cfg['family'],cfg['seed'],.75,.2,par['xi'],8.)
        row['E_M']=X.norm(GLh-(G-GSfar))/nref
        row['E_R']=X.norm(E.real_dense(op)-GSfar)/nref
    v=np.random.default_rng(1001).normal(size=(256,3))
    tt=C.median_time(lambda:op.apply(v),3,warm=1)
    row.update(t_screen_s=float(np.median(tt)),elapsed_s=time.perf_counter()-start)
    return row

def calibrate():
    cfg=X.configurations()[0]
    path=X.RESULTS/'calibration.csv'
    rows=X.read(path);done={(r['xi'],r['M'],r['p']) for r in rows}
    for xi,p in calibrated_orders()+[(.3,1),(.6,1),(.85,1)]:
        for M in (32,64):
            if (xi,M,p) in done:continue
            r=measure(cfg,X.parameters(xi,3.2,M,p),mesh_error=True)
            rows.append(r);X.write(path,rows)
            X.log(f'calibrate xi={xi} p={p} M={M}: EM={r["E_M"]:.3g}')

def initial_points():
    cal=X.read(X.RESULTS/'calibration.csv')
    fits=X.read(X.RESULTS/'mesh_rate_fits.csv')
    if not fits:raise RuntimeError('Complete Experiment 1 and its fits before parameter search')
    measured_q=float(np.median([r['order'] for r in fits if r['fit_range']=='finest_half' and r['metric']=='E_G_mesh_only']))
    init=[];pts=[]
    for xi,p in calibrated_orders():
        a,b=sorted([r for r in cal if r['xi']==xi and r['p']==p],key=lambda r:r['M'])
        q=np.log(a['E_M']/b['E_M'])/np.log(b['M']/a['M'])
        # p=2 phase variation is calibrated separately; p>=3 uses Exp 1 order.
        q_used=q if p==2 else measured_q
        Mpred=b['M']*(b['E_M']/(.65*X.TOL))**(1/q_used)
        init.append(dict(xi=xi,p=p,sR_theory=np.sqrt(np.log(1/X.TOL)),measured_mesh_order=q,
                         exp1_mesh_order=measured_q,order_used=q_used,predicted_M=Mpred,
                         model='E_M(M)=E_M(64)*(64/M)^q; predictor only, no permanent error exclusions'))
        for factor in (.82,1.,1.2):
            M=next_smooth(max(p,8,int(np.ceil(factor*Mpred))))
            if M>160:continue
            for sR in (2.6,2.8,3.,3.3):pts.append(X.parameters(xi,sR,M,p))
    X.write(X.RESULTS/'theory_initialization.csv',init)
    return pts

def screen(points,stage):
    path=X.RESULTS/'search.csv';rows=X.read(path)
    done={r['id'] for r in rows};cfg=X.configurations()[0]
    for par in points:
        id_=X.ident(par)
        if id_ in done:continue
        r=measure(cfg,par)
        r.update(id=id_,stage=stage,screen_pass=bool(r['E_G']<=X.TOL))
        rows.append(r);done.add(id_);X.write(path,rows)
        X.log(f'{stage} {id_}: E={r["E_G"]:.3g} time={1000*r["t_screen_s"]:.2f}ms')
    return rows

def validation_candidates(rows,limit=18):
    valid=[r for r in rows if r['screen_pass']]
    valid.sort(key=lambda r:r['t_screen_s'])
    selected=valid[:limit]
    # Cover p and xi slices, not only transient screen timing winners.
    for key in ('p','xi'):
        for v in sorted({r[key] for r in valid}):
            selected.append(next(r for r in valid if r[key]==v))
    return list({r['id']:r for r in selected}.values())

def par_from(r):
    return X.parameters(r['xi'],r['sR'],r['M'],r['p'],r['sT'])

def validate(candidates):
    path=X.RESULTS/'validation.csv';rows=X.read(path)
    done={(r['id'],r['config']) for r in rows}
    for cand in candidates:
        id_=cand['id'];par=par_from(cand)
        for cfg in X.configurations():
            if (id_,cfg['name']) in done:continue
            r,op,Gh,GLh=X.evaluate(cfg,par,structure=True,actions=True)
            rows.append(dict(id=id_,**r));done.add((id_,cfg['name']));X.write(path,rows)
        rs=[r for r in rows if r['id']==id_]
        X.log(f'validate {id_}: maxE={max(r["E_G"] for r in rs):.3g}, n={len(rs)}')
    summary=[]
    for id_ in sorted({r['id'] for r in rows}):
        rs=[r for r in rows if r['id']==id_]
        if len(rs)!=9:continue
        r=rs[0]
        summary.append(dict(id=id_,**par_from(r),max_E_G=max(v['E_G'] for v in rs),
                            max_E_GX=max(v['E_GX'] for v in rs),min_eigenvalue=min(v['lam_min_h'] for v in rs),
                            feasible=max(v['E_G'] for v in rs)<=X.TOL,n_configs=9))
    X.write(X.RESULTS/'validation_summary.csv',summary)
    return summary

def local_points(summary):
    rows=X.read(X.RESULTS/'search.csv');td={r['id']:r['t_screen_s'] for r in rows}
    incumbents=sorted([r for r in summary if r['feasible']],key=lambda r:td[r['id']])[:3]
    pts=[]
    for r in incumbents:
        for xi in (r['xi']-.05,r['xi'],r['xi']+.05):
            for sR in (r['sR']-.1,r['sR'],r['sR']+.1):
                for fac in (.88,1.,1.10):
                    M=next_smooth(max(8,int(np.ceil(r['M']*fac))))
                    for p in (2,3):pts.append(X.parameters(round(xi,4),round(sR,4),M,p))
    return pts

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['calibrate','search','local','boundary','validate'])
    args=ap.parse_args()
    with X.threadpool_limits(1):
        if args.stage=='calibrate':calibrate()
        elif args.stage=='search':
            rows=screen(initial_points(),'theory_coarse')
            validate(validation_candidates(rows))
        elif args.stage=='local':
            summary=X.read(X.RESULTS/'validation_summary.csv')
            rows=screen(local_points(summary),'local')
            validate(validation_candidates(rows,24))
        elif args.stage=='boundary':
            pts=[X.parameters(.3,s,54,2) for s in (3.35,3.4,3.5)]
            pts += [X.parameters(.325,s,m,2) for s in (3.2,3.3) for m in (60,64)]
            pts += [X.parameters(.375,s,m,2) for s in (3.2,3.3) for m in (72,75)]
            pts += [X.parameters(.35,s,64,2) for s in (3.3,3.4)]
            pts += [X.parameters(.275,s,54,2) for s in (3.3,3.4)]
            rows=screen(pts,'boundary')
            validate([r for r in rows if r['stage']=='boundary' and r['screen_pass']])
        else:validate(validation_candidates(X.read(X.RESULTS/'search.csv'),24))

if __name__=='__main__':main()
