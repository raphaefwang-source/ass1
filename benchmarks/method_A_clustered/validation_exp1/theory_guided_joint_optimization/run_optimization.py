#!/usr/bin/env python3
"""Resumable, measured coarse-to-fine five-parameter experiment at kappa=.2."""
import argparse
import hashlib
from collections import OrderedDict
import itertools
import time
import experiment as X
from experiment import E,C,np,threadpool_limits

XI=(.6,.8,1.,1.2,1.4,1.6,1.8,2.)
MESH=(8,10,12,14,16,18,20,22,24,27,30,32,36,40,48)
ORDERS=(3,4,5,6,7)

class Matrices:
    def __init__(self):
        self.meshes=E.MeshCache()
        self.caches={n:OrderedDict() for n in ('S','H','F')}

    def get(self,name,key,fn):
        cache=self.caches[name]
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        value=fn()
        cache[key]=value
        if len(cache)>24: cache.popitem(last=False)
        return value

    def build(self,cfg,par):
        if len(self.meshes.m)>24: self.meshes.m.clear()
        return X.build(cfg,par,self.meshes)

    def components(self,cfg,par,op,meshfree=False):
        name=cfg['name']; xi=par['xi']; kc=op.mesh.kc
        GS=self.get('S',(name,xi,par['sR']),lambda:E.real_dense(op))
        GH=self.get('H',(name,xi,kc,par['M'],par['p']),
                    lambda:E.lap_full(E.mode_pair_matrix_fast(op.mesh,cfg['q'])))
        GF=(self.get('F',(name,xi,kc),lambda:E.fourier_dense(cfg['q'],cfg['L'],
                 E.kernel(.2,xi,0.),kc)) if meshfree else None)
        return GS,GH,GF

def profile_mesh(op,V):
    """Timestamp the identical Mesh.apply expressions; used only for diagnostics.

    FFT includes influence multiplication and inverse transforms. Spread/gather
    excludes the cached degree action, which belongs to the measured mesh total.
    """
    mesh=op.mesh; M=mesh.M; idx=op.idx; wt=op.wt
    t0=time.perf_counter()
    density=[np.bincount(idx.ravel(),(wt*V[:,c,None]).ravel(),M**3).reshape((M,)*3) for c in range(3)]
    t1=time.perf_counter()
    hats=[np.fft.rfftn(rho) for rho in density]
    phi=mesh._conv(hats,range(3))
    t2=time.perf_counter()
    value=np.stack([(phi[d].ravel()[idx]*wt).sum(axis=1) for d in range(3)],axis=1)/mesh.h**3
    t3=time.perf_counter()
    return value,dict(t_spread_s=t1-t0,t_fft_s=t2-t1,t_gather_s=t3-t2,
                      t_spread_gather_s=t1-t0+t3-t2,t_profile_mesh_s=t3-t0)

def quick_times(op,reps=5):
    V=np.random.default_rng(100).standard_normal((256,3))
    ts=C.median_time(lambda:op.apply(V),reps,warm=3)
    return dict(t_screen_s=float(np.median(ts)),t_screen_min_s=float(np.min(ts)),
                t_screen_iqr_rel=C.tstats(ts)['iqr_rel'])

class Search:
    def __init__(self):
        self.rows=X.read(X.HERE/'joint_candidates.csv')
        self.byid={r['candidate_id']:r for r in self.rows}
        self.validation=X.read(X.RESULTS/'screen_validation.csv')
        self.matrices=Matrices()

    def save(self):
        X.write(X.HERE/'joint_candidates.csv',self.rows)
        X.write(X.RESULTS/'screen_validation.csv',self.validation)

    def batch(self,parameters,stage):
        new=0
        for par in parameters:
            cid=X.ident(par)
            if cid in self.byid:
                tags=set(self.byid[cid]['stages'].split(';')); tags.add(stage)
                self.byid[cid]['stages']=';'.join(sorted(tags))
                continue
            cfg=X.configs()[0]
            rc,kc,shell=E.cutoffs(par,cfg['L'])
            row=dict(candidate_id=cid,kappa=.2,c=1.,N=256,rho=.75,theta=0.,xi=par['xi'],
                     sR=par['sR'],sF_req=par['sF'],sF_eff=kc/(2*par['xi']),rc=rc,kc=kc,
                     kc_shell_n=shell,M=par['M'],p=par['p'],stages=stage,
                     error_scope=cfg['name'],n_configs=0,feasible=False)
            op,info=self.matrices.build(cfg,par)
            row.update(mesh_valid=info['valid'],mesh_reason=info['reason'],eta=info['eta'],
                       n_modes=info.get('n_modes'))
            if op is None:
                row['status']='invalid mesh'
            else:
                GS,GH,_=self.matrices.components(cfg,par,op)
                err=E.norm2_sym(GS+GH-X.reference(cfg['name']))[0]/X.normref(cfg['name'])
                row.update(E_G_design=err,n_images_design=op.n_images,**quick_times(op))
                row['screen_pass']=err<=X.TOL
                row['status']='design pass; nine-config validation pending' if err<=X.TOL else 'design error exceeds target'
            self.rows.append(row); self.byid[cid]=row; new+=1
            if new%10==0: self.save()
            if new%40==0: X.log(f'{stage}: {new} new requests, total={len(self.rows)}')
        self.save()
        X.log(f'{stage}: complete, {new} new requests, total={len(self.rows)}')

    def valid_design(self,equal=False):
        return [r for r in self.rows if r.get('screen_pass') and
                (not equal or abs(r['sR']-r['sF_req'])<1e-10)]

    def validate(self,rows,force=False,decompose=False):
        for index,row in enumerate(rows):
            par=X.fromrow(row); cid=row['candidate_id']
            done={v['config']:v for v in self.validation if v['candidate_id']==cid}
            if not force and any(v['E_G']>X.TOL for v in done.values()): continue
            for cfg in X.configs():
                prior=done.get(cfg['name'])
                if prior and (not decompose or prior.get('E_M') is not None): continue
                op,info=self.matrices.build(cfg,par)
                assert info['valid']
                GS,GH,GF=self.matrices.components(cfg,par,op,meshfree=decompose)
                norm=X.normref(cfg['name']); G=X.reference(cfg['name'])
                err=E.norm2_sym(GS+GH-G)[0]/norm
                v=dict(candidate_id=cid,config=cfg['name'],family=cfg['family'],seed=cfg['seed'],
                       E_G=err,n_images=op.n_images,**quick_times(op,3))
                if decompose:
                    v.update(E_cutoff=E.norm2_sym(GS+GF-G)[0]/norm,E_M=E.norm2_sym(GH-GF)[0]/norm)
                if prior: self.validation.remove(prior)
                self.validation.append(v); done[cfg['name']]=v
                if err>X.TOL and not force: break
            vs=list(done.values())
            full=len(vs)==9
            row.update(n_configs=len(vs),E_G_max_tested=max(v['E_G'] for v in vs),
                       feasible=full and max(v['E_G'] for v in vs)<=X.TOL,
                       t_nine_screen_s=float(np.median([v['t_screen_s'] for v in vs])) if full else None,
                       error_scope='maximum of nine configurations' if full else 'partial nine-config rejection')
            row['status']='nine-config feasible' if row['feasible'] else 'nine-config error rejection'
            if decompose:
                row.update(E_cutoff_max=max(v['E_cutoff'] for v in vs),E_M_max=max(v['E_M'] for v in vs))
            self.save()
            X.log(f'validate {index+1}/{len(rows)} {cid} Emax={row["E_G_max_tested"]:.6g} n={len(vs)} feasible={row["feasible"]}')

    def best(self,equal=False):
        choices=[r for r in self.rows if r.get('feasible') and
                 (not equal or abs(r['sR']-r['sF_req'])<1e-10)]
        return min(choices,key=lambda r:r['t_nine_screen_s'])

    def refresh_times(self):
        # Coarse error computations may overlap independent reference building.
        # Rerank every survivor after all nine references exist and workers stop.
        rows=self.valid_design();rng=np.random.default_rng(20261026)
        for i in rng.permutation(len(rows)):
            row=rows[i];op,_=self.matrices.build(X.configs()[0],X.fromrow(row))
            row.setdefault('initial_t_screen_s',row['t_screen_s'])
            row.update(quick_times(op,9),screen_times_refreshed=True)
        self.save();X.log(f'refreshed {len(rows)} design survivor timings')

    def shortlist(self):
        good=sorted(self.valid_design(),key=lambda r:r['t_screen_s'])
        chosen={r['candidate_id']:r for r in good[:40]}
        for field in ('xi','p','M'):
            for value in sorted(set(r[field] for r in good)):
                group=[r for r in good if r[field]==value]
                chosen.update({r['candidate_id']:r for r in group[:3]})
        equal=sorted(self.valid_design(True),key=lambda r:r['t_screen_s'])
        for xi in sorted(set(r['xi'] for r in equal)):
            chosen.update({r['candidate_id']:r for r in [v for v in equal if v['xi']==xi][:3]})
        old=self.byid[X.ident(X.OLD)]; chosen[old['candidate_id']]=old
        return sorted(chosen.values(),key=lambda r:r.get('t_screen_s') or float('inf'))

def coarse(search):
    search.batch([X.OLD],'historical-parameter-control')
    search.batch((X.pars(xi,s,s,M,p) for xi,s,p,M in itertools.product(
        XI,(2.2,2.6,3.,3.4),ORDERS,(12,16,20,24,32))),'coarse-equal')

def cutoffs(search):
    requests=[]
    for xi in XI:
        good=sorted([r for r in search.valid_design() if r['xi']==xi],key=lambda r:r['t_screen_s'])
        if not good:
            good=sorted([r for r in search.rows if r['xi']==xi and r.get('E_G_design') is not None],
                        key=lambda r:r['E_G_design'])
        # Cover every assignment order, rather than letting initial timing noise
        # select only one or two p families for the independent-cutoff plane.
        pairs=[]
        for p in ORDERS:
            order=[r for r in good if r['p']==p]
            if not order:
                order=sorted([r for r in search.rows if r['xi']==xi and r['p']==p and r.get('E_G_design') is not None],
                             key=lambda r:r['E_G_design'])
            if order:pairs.append((order[0]['M'],p))
        for M,p in pairs:
            requests += [X.pars(xi,a,b,M,p) for a,b in itertools.product((2.2,2.5,2.8,3.1,3.4),repeat=2)]
    search.batch(requests,'independent-cutoffs')

def refine(search):
    # Actual measured incumbents, not the theoretical sums, guide local search.
    for iteration in range(2):
        good=sorted(search.valid_design(),key=lambda r:r['t_screen_s'])
        seeds=[]
        for equal in (False,True):
            for xi in sorted(set(r['xi'] for r in good)):
                group=[r for r in good if r['xi']==xi and (not equal or r['sR']==r['sF_req'])]
                seeds+=group[:1]
        requests=[]
        for row in seeds:
            v=dict(xi=row['xi'],sR=row['sR'],sF=row['sF_req'],M=row['M'],p=row['p'])
            for name,values in (('xi',[round(v['xi']-.1,4),round(v['xi']+.1,4)]),
                                ('sR',[round(v['sR']-.1,4),round(v['sR']+.1,4)]),
                                ('sF',[round(v['sF']-.1,4),round(v['sF']+.1,4)]),
                                ('M',[m for m in MESH if abs(m-v['M'])<=4]),
                                ('p',[p for p in ORDERS if abs(p-v['p'])<=1])):
                for value in values:
                    vv=dict(v); vv[name]=value
                    if vv['xi']>=.3 and min(vv['sR'],vv['sF'])>=1.6: requests.append(X.pars(**vv))
            # Equal-cutoff control is refined under its own constraint.
            for s in (round(v['sR']-.1,4),round(v['sR']+.1,4)):
                requests.append(X.pars(v['xi'],s,s,v['M'],v['p']))
        # A local cutoff plane and M/p plane catch coupled moves missed by coordinates.
        for row in good[:3]:
            for a,b in itertools.product((-.1,0.,.1),repeat=2):
                requests.append(X.pars(row['xi'],round(row['sR']+a,4),round(row['sF_req']+b,4),row['M'],row['p']))
            for M,p in itertools.product([m for m in MESH if abs(m-row['M'])<=6],ORDERS):
                requests.append(X.pars(row['xi'],row['sR'],row['sF_req'],M,p))
        search.batch(requests,f'local-refinement-{iteration+1}')

def expansion(search):
    # Broaden a best design lying at a sampled boundary, and explicitly compare
    # both sides of the current fastest xi. Repeated invocations are resumable.
    good=sorted(search.valid_design(),key=lambda r:r['t_screen_s'])
    requests=[]
    for equal in (False,True):
        row=next(r for r in good if not equal or r['sR']==r['sF_req'])
        for delta in (-.2,-.1,.1,.2):
            xi=round(row['xi']+delta,4)
            if xi<.3: continue
            for M,p in itertools.product([m for m in MESH if abs(m-row['M'])<=6],ORDERS):
                requests.append(X.pars(xi,row['sR'],row['sF_req'],M,p))
        for field in ('sR','sF_req'):
            values=[r[field] for r in good]
            if row[field] in (min(values),max(values)):
                direction=-1 if row[field]==min(values) else 1
                for delta in (.2,.4):
                    sr=row['sR']; sf=row['sF_req']
                    if field=='sR': sr=round(sr+direction*delta,4)
                    else: sf=round(sf+direction*delta,4)
                    if equal: sr=sf=round(row['sR']+direction*delta,4)
                    if min(sr,sf)>=1.2: requests.append(X.pars(row['xi'],sr,sf,row['M'],row['p']))
    search.batch(requests,'boundary-and-xi-recheck')

def verified_refine(search):
    """Local coupled moves about the actual nine-configuration incumbents."""
    requests=[]
    for equal in (False,True):
        center=search.best(equal);bp=X.fromrow(center)
        for xi in [round(bp['xi']+d,4) for d in (-.2,-.1,-.05,0.,.05,.1,.2)]:
            if xi<.3:continue
            for M,p in itertools.product([m for m in MESH if abs(m-bp['M'])<=4],ORDERS):
                requests.append(X.pars(xi,bp['sR'],bp['sF'],M,p))
        for dr,df in itertools.product((-.1,-.05,0.,.05,.1),repeat=2):
            if equal and dr!=df:continue
            for M,p in ((bp['M'],bp['p']),
                        (max(m for m in MESH if m<=bp['M']),max(3,bp['p']-1))):
                requests.append(X.pars(bp['xi'],round(bp['sR']+dr,4),round(bp['sF']+df,4),M,p))
    search.batch(requests,'nine-config-guided-refinement')
    search.refresh_times();search.validate(search.shortlist())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stages',nargs='+',choices=('coarse','cutoffs','refine','expand','validate','verified','all'))
    args=ap.parse_args()
    # No reference rebuilding here; the physical-key audit is mandatory.
    stages=('coarse','cutoffs','refine','expand','validate','verified') if 'all' in args.stages else args.stages
    for cfg in (X.configs() if any(s in stages for s in ('validate','verified')) else X.configs()[:1]): X.reference(cfg['name'])
    invocation=X.provenance(stages=args.stages,scope='adaptive discrete search',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in X.HERE.glob('*.py')})
    X.jsave(X.RESULTS/'search_provenance.json',invocation)
    X.jsave(X.RESULTS/f"search_provenance_{time.strftime('%Y%m%d_%H%M%S')}.json",invocation)
    E.log_cmd(X.RESULTS)
    search=Search()
    with threadpool_limits(limits=1):
        if 'validate' in stages and search.rows:search.refresh_times()
        for stage in stages:
            X.log(f'START {stage}')
            if stage=='validate':
                search.refresh_times()
                search.validate(search.shortlist())
            elif stage=='verified':verified_refine(search)
            elif stage=='expand':expansion(search)
            else: globals()[stage](search)
            X.log(f'DONE {stage}')

if __name__=='__main__': main()
