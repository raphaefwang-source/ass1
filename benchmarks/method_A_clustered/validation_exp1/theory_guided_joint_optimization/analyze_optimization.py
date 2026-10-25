#!/usr/bin/env python3
"""Measured sensitivities, same-shell error separation, and interleaved finalists."""
import argparse
import hashlib
import itertools
import json
import shutil
import time
from functools import lru_cache
import experiment as X
from experiment import E,C,np,threadpool_limits
import run_optimization as R

PARAMS=('xi','sR','sF_req','M','p')

def plan_study(search):
    prior=json.loads((X.RESULTS/'study_plan.json').read_text()) if (X.RESULTS/'study_plan.json').exists() else None
    baseline=prior['baseline'] if prior else search.best(equal=True)
    bp=X.fromrow(baseline)
    free=prior['free_incumbent'] if prior else search.best()
    mp=[X.pars(bp['xi'],bp['sR'],bp['sF'],M,p) for M,p in itertools.product(R.MESH,R.ORDERS)]
    search.batch(mp,'mesh-tradeoff')
    # The independent cutoff heatmap uses the free incumbent, with xi/M/p fixed.
    fp=X.fromrow(free)
    rs=[round(fp['sR']+v,4) for v in (-.4,-.3,-.2,-.1,0.,.1,.2,.3,.4)]
    fs=[round(fp['sF']+v,4) for v in (-.4,-.3,-.2,-.1,0.,.1,.2,.3,.4)]
    plane=[X.pars(fp['xi'],a,b,fp['M'],fp['p']) for a,b in itertools.product(rs,fs)]
    search.batch(plane,'cutoff-heatmap')
    values=dict(xi=sorted(set([.6,.8,1.,1.2,1.4,1.6,1.8,2.,bp['xi']]+[
                    round(bp['xi']+d,4) for d in (-.2,-.1,.1,.2)])),
                sR=sorted(set([2.,2.2,2.4,2.6,2.8,3.,3.2,3.4,3.6,bp['sR']])),
                sF=sorted(set([2.,2.2,2.4,2.6,2.8,3.,3.2,3.4,3.6,bp['sF']])),
                M=list(R.MESH),p=list(R.ORDERS))
    oat=[]
    for field,choices in values.items():
        for value in choices:
            p=dict(bp);p[field]=value
            par=X.pars(**{k:p[k] for k in ('xi','sR','sF','M','p')})
            oat.append(dict(parameter=field,value=value,candidate_id=X.ident(par)))
            search.batch([par],f'OAT-{field}')
    X.jsave(X.RESULTS/'study_plan.json',dict(baseline=baseline,free_incumbent=free,
             mesh_ids=[X.ident(p) for p in mp],heatmap_ids=[X.ident(p) for p in plane],OAT=oat))
    # Deliberately complete all nine even for an infeasible sensitivity setting.
    ids=set([X.ident(p) for p in mp+plane]+[r['candidate_id'] for r in oat])
    rows=[search.byid[c] for c in sorted(ids) if search.byid[c]['mesh_valid']]
    search.validate(rows,force=True,decompose=True)

def measure_diagnostics(search,refresh=False):
    plan=json.loads((X.RESULTS/'study_plan.json').read_text())
    ids=set(plan['mesh_ids']+plan['heatmap_ids']+[r['candidate_id'] for r in plan['OAT']])
    ids=sorted(cid for cid in ids if search.byid[cid]['mesh_valid'])
    if refresh and (X.RESULTS/'diagnostic_timing.csv').exists():
        archive=X.RESULTS/f"diagnostic_pass_{time.strftime('%Y%m%d_%H%M%S')}"
        archive.mkdir()
        for name in ('diagnostic_timing.csv','diagnostic_timing_raw.csv','diagnostic_summary.csv','diagnostic_protocol.json'):
            path=X.RESULTS/name
            if path.exists():shutil.move(path,archive/name)
        for name in ('mesh_tradeoff.csv','parameter_sensitivity.csv'):
            if (X.HERE/name).exists():shutil.copy2(X.HERE/name,archive/name)
    raw=X.read(X.RESULTS/'diagnostic_timing.csv')
    samples=X.read(X.RESULTS/'diagnostic_timing_raw.csv')
    done={(r['candidate_id'],r['config']) for r in raw}
    rng=np.random.default_rng(20261024)
    started=time.time(); checkpoints=[]
    for cfg in X.configs():
        V=np.random.default_rng(101).standard_normal((256,3))
        pending=[cid for cid in ids if (cid,cfg['name']) not in done]
        ops={};functions={}
        for cid in pending:
            op,info=search.matrices.build(cfg,X.fromrow(search.byid[cid]))
            assert info['valid']
            v,parts=R.profile_mesh(op,V)
            assert np.linalg.norm(v-op.mesh.apply(op.idx,op.wt,V))/np.linalg.norm(v)<1e-12
            split=np.einsum('nab,nb->na',op.DS,V)-op.real_H(V)+np.einsum('nab,nb->na',op.Dm,V)-v
            assert np.linalg.norm(split-op.apply(V))/np.linalg.norm(split)<1e-12
            ops[cid]=op
            functions[cid]=dict(complete=lambda op=op:op.apply(V),
                real=lambda op=op:np.einsum('nab,nb->na',op.DS,V)-op.real_H(V),
                mesh=lambda op=op:np.einsum('nab,nb->na',op.Dm,V)-op.mesh.apply(op.idx,op.wt,V),
                profile=lambda op=op:R.profile_mesh(op,V))
        for rd in range(3):
            for phase in rng.permutation(('complete','real','mesh','profile')):
                for cid in rng.permutation(pending):
                    for _ in range(3):functions[cid][phase]()
                for rep in range(5):
                    for cid in rng.permutation(pending):
                        fn=functions[cid][phase];fn()
                        if phase=='profile':
                            _,parts=fn()
                            sample=dict(candidate_id=cid,config=cfg['name'],round=rd,rep=rep,phase=phase,**parts)
                        else:
                            t0=time.perf_counter();fn();dt=time.perf_counter()-t0
                            sample=dict(candidate_id=cid,config=cfg['name'],round=rd,rep=rep,phase=phase,t_s=dt)
                        samples.append(sample)
        for cid in pending:
            op=ops[cid];result={}
            subset=[r for r in samples if r['candidate_id']==cid and r['config']==cfg['name']]
            for phase in ('complete','real','mesh'):
                stats=C.tstats([r['t_s'] for r in subset if r['phase']==phase])
                result[f't_{phase}_s']=stats['median'];result[f't_{phase}_iqr_rel']=stats['iqr_rel']
            profiles=[r for r in subset if r['phase']=='profile']
            result.update({k:float(np.median([r[k] for r in profiles])) for k in
                           ('t_spread_s','t_gather_s','t_spread_gather_s','t_fft_s','t_profile_mesh_s')})
            raw.append(dict(candidate_id=cid,config=cfg['name'],n_images=op.n_images,n_modes=op.mesh.n_modes,
                            component_reps=15,**result))
        X.write(X.RESULTS/'diagnostic_timing.csv',raw)
        X.write(X.RESULTS/'diagnostic_timing_raw.csv',samples)
        checkpoints.append(dict(config=cfg['name'],elapsed_s=time.time()-started))
        X.jsave(X.RESULTS/'diagnostic_protocol.json',dict(rounds=3,reps=5,threads=1,
            candidate_order='random every repetition',phase_order='random each round',
            started_unix=started,elapsed_s=time.time()-started,checkpoints=checkpoints))
        X.log(f'diagnostic phase timing complete: {cfg["name"]}')
    summaries={}
    for cid in ids:
        row=dict(search.byid[cid]);vs=[r for r in search.validation if r['candidate_id']==cid]
        ts=[r for r in raw if r['candidate_id']==cid]
        assert len(vs)==len(ts)==9
        row.update(E_G_max=max(v['E_G'] for v in vs),E_cutoff_max=max(v['E_cutoff'] for v in vs),
                   E_M_max=max(v['E_M'] for v in vs),
                   **{k:float(np.median([r[k] for r in ts])) for k in
                      ('t_complete_s','t_real_s','t_mesh_s','t_spread_gather_s','t_fft_s','t_spread_s','t_gather_s','n_images')})
        summaries[cid]=row
    def output(ids):
        return [summaries[c] if c in summaries else search.byid[c] for c in ids]
    X.write(X.HERE/'mesh_tradeoff.csv',output(plan['mesh_ids']))
    X.write(X.RESULTS/'cutoff_heatmap.csv',output(plan['heatmap_ids']))
    oat=[]
    base=summaries[plan['baseline']['candidate_id']]
    for record in plan['OAT']:
        row=dict(record,**{k:v for k,v in (summaries.get(record['candidate_id']) or search.byid[record['candidate_id']]).items()
                          if k!='candidate_id'})
        row['candidate_id']=record['candidate_id']
        field=record['parameter']; source='sF_req' if field=='sF' else field
        row['parameter_relative_change']=record['value']/base[source]-1
        if row.get('t_complete_s'):
            row['complete_time_change_s']=row['t_complete_s']-base['t_complete_s']
            row['complete_time_relative_change']=row['t_complete_s']/base['t_complete_s']-1
        oat.append(row)
    X.write(X.HERE/'parameter_sensitivity.csv',oat)
    X.write(X.RESULTS/'diagnostic_summary.csv',list(summaries.values()))
    # M/p and cutoff-plane results are part of the adaptive search too.
    search.save()
    return summaries

@lru_cache(16)
def far_split(name,xi):
    cfg=next(c for c in X.configs() if c['name']==name)
    K=E.kernel(.2,xi,0.)
    # Use an ordinary valid low-cost mesh only to construct ThetaOp's real part.
    par=X.pars(xi,8.,2.6,48,3)
    op,info=X.build(cfg,par)
    assert info['valid']
    GS=E.real_dense(op)
    kc,_=E.snap_kc(2*xi*7.,cfg['L'])
    GF=E.fourier_dense(cfg['q'],cfg['L'],K,kc)
    residual=E.norm2_sym(GS+GF-X.reference(name))[0]/X.normref(name)
    assert residual<1e-10,residual
    return GS,GF,residual,kc

def decompose(search,ids=None,all_configs=False):
    if ids is None:
        best=search.best();equal=search.best(True)
        ids=[X.ident(X.OLD),best['candidate_id'],equal['candidate_id']]
        diagnostics=X.read(X.HERE/'mesh_tradeoff.csv')
        feasible=[r for r in diagnostics if r.get('feasible')]
        if feasible:
            ids += [min(feasible,key=lambda r:r['t_complete_s'])['candidate_id'],
                    min(feasible,key=lambda r:r['E_M_max'])['candidate_id']]
        # Add the full order sweep at the study baseline's mesh, including failures.
        plan=json.loads((X.RESULTS/'study_plan.json').read_text())
        ids += [r['candidate_id'] for r in plan['OAT'] if r['parameter']=='p']
    ids=list(dict.fromkeys(ids))
    out=X.read(X.HERE/'error_decomposition.csv')
    done={(r['candidate_id'],r['config']) for r in out}
    for cid in ids:
        row=search.byid[cid]
        if not row['mesh_valid']: continue
        par=X.fromrow(row)
        # Nine configs for old/equal/free controls; representative uniform2 and
        # strong3 for surrounding order/resolution diagnostics.
        primary={X.ident(X.OLD),search.best()['candidate_id'],search.best(True)['candidate_id']}
        cfgs=X.configs() if all_configs or cid in primary else X.configs()[:2]
        for cfg in cfgs:
            if (cid,cfg['name']) in done: continue
            op,_=search.matrices.build(cfg,par)
            GS,GH,GF=search.matrices.components(cfg,par,op,meshfree=True)
            Sfar,Ffar,residual,kfar=far_split(cfg['name'],par['xi'])
            G=X.reference(cfg['name']); norm=X.normref(cfg['name'])
            dR=GS-Sfar;dF=GF-Ffar;dM=GH-GF
            normed=lambda A:E.norm2_sym(A)[0]/norm
            ER,EF,EM,EC,EG=[normed(A) for A in (dR,dF,dM,GS+GF-G,GS+GH-G)]
            rowout=dict(candidate_id=cid,config=cfg['name'],kappa=.2,xi=par['xi'],sR=par['sR'],sF_req=par['sF'],
                        sF_eff=op.mesh.kc/(2*par['xi']),M=par['M'],p=par['p'],E_R=ER,E_F=EF,E_M=EM,
                        E_cutoff=EC,E_G=EG,triangle_sum=ER+EF+EM,
                        cutoff_triangle_ratio=EC/(ER+EF),full_triangle_ratio=EG/(ER+EF+EM),
                        E_mesh_to_cutoff=EM/EC,far_split_residual=residual,sR_far=8.,sF_far=kfar/(2*par['xi']),
                        telescope_residual=normed((dR+dF+dM)-(GS+GH-G)))
            assert rowout['telescope_residual']<1e-10
            out.append(rowout);X.write(X.HERE/'error_decomposition.csv',out)
        X.log(f'error decomposition {cid}')

def select_finalists(search):
    feasible=sorted([r for r in search.rows if r.get('feasible')],key=lambda r:r['t_nine_screen_s'])
    chosen={r['candidate_id']:r for r in feasible[:8]}
    prior=X.RESULTS/'previous_measured_incumbents.json'
    if prior.exists():
        for cid in json.loads(prior.read_text()):
            if search.byid[cid].get('feasible'):chosen[cid]=search.byid[cid]
    for field in PARAMS:
        for value in sorted(set(r[field] for r in feasible)):
            group=[r for r in feasible if r[field]==value]
            # xi/sR/sF grids become dense locally; include nearby competitive
            # points, while always retaining every order and small mesh family.
            if field in ('xi','p','M') or group[0]['t_nine_screen_s']<=1.35*feasible[0]['t_nine_screen_s']:
                chosen[group[0]['candidate_id']]=group[0]
    equal=search.best(True);chosen[equal['candidate_id']]=equal
    old=search.byid[X.ident(X.OLD)];chosen[old['candidate_id']]=old
    trades=[r for r in X.read(X.HERE/'mesh_tradeoff.csv') if r.get('feasible')]
    strategies={}
    if trades:
        B=min(trades,key=lambda r:r['t_complete_s'])
        strict=[r for r in trades if r['E_M_max']<=.1*r['E_cutoff_max']]
        A=min(strict,key=lambda r:r['t_complete_s']) if strict else min(trades,key=lambda r:r['E_M_max'])
        for label,row in [('A',A),('B',B)]:
            chosen[row['candidate_id']]=search.byid[row['candidate_id']]
            strategies[label]=row['candidate_id']
    diagnostic=[r for r in X.read(X.RESULTS/'diagnostic_summary.csv') if r.get('feasible')]
    if diagnostic:
        balance=min(diagnostic,key=lambda r:abs(np.log(r['t_real_s']/r['t_mesh_s'])))
        chosen[balance['candidate_id']]=search.byid[balance['candidate_id']]
        X.jsave(X.RESULTS/'balance_control.json',balance)
    # Nearby nominal sF requests can snap to the same shell. Time each physical
    # operator once, retaining the old/equal-cutoff identity when applicable.
    merged={};aliases={}
    for row in [old,equal]+list(chosen.values()):
        signature=tuple(row[k] for k in ('xi','sR','kc_shell_n','M','p'))
        if signature in merged:
            aliases[row['candidate_id']]=merged[signature]['candidate_id']
        else:merged[signature]=row
    strategies={label:aliases.get(cid,cid) for label,cid in strategies.items()}
    X.jsave(X.RESULTS/'strategy_controls.json',strategies)
    X.jsave(X.RESULTS/'final_candidate_aliases.json',aliases)
    return list(merged.values())

def timing(search,rounds=9,reps=11):
    started=time.time();checkpoints=[]
    selected=select_finalists(search)
    # Full existing E.evaluate structural validation on every timed finalist.
    val=X.read(X.RESULTS/'final_validation.csv')
    done={(v['candidate_id'],v['config']) for v in val}
    for row in selected:
        cid=row['candidate_id']; par=X.fromrow(row)
        for cfg in X.configs():
            if (cid,cfg['name']) not in done:
                v=X.evaluate(cfg,par,candidate_id=cid)
                assert v['mesh_valid'] and v['lam_min_h']>0 and v['sym_resid']<1e-12
                assert v['mom_opnorm_max']<1e-12 and v['mf_vs_dense']<1e-10
                if cid!=X.ident(X.OLD): assert v['E_G']<=X.TOL
                val.append(v);X.write(X.RESULTS/'final_validation.csv',val)
        X.log(f'full structure validation {cid}')
    path=X.RESULTS/'final_timing_raw.csv'
    if path.exists():
        archive=X.RESULTS/f"timing_pass_{time.strftime('%Y%m%d_%H%M%S')}"
        archive.mkdir()
        for name in ('final_timing_raw.csv','final_timing.csv','final_setup_raw.csv','timing_protocol.json','analysis_provenance.json'):
            if (X.RESULTS/name).exists():shutil.copy2(X.RESULTS/name,archive/name)
    raw=[];out=[];setup=[];rng=np.random.default_rng(20261025)
    for cfg in X.configs():
        V=np.random.default_rng(101).standard_normal((256,3))
        ops={}
        for row in selected:
            cid=row['candidate_id'];par=X.fromrow(row);K=E.kernel(.2,par['xi'],0.)
            rc,kc,_=E.cutoffs(par,cfg['L'])
            for rep in range(3):
                t0=time.perf_counter();mesh,info=E.MeshCache().get(cfg['L'],K,kc,par['M'],par['p']);t1=time.perf_counter()
                op=E.ThetaOp(cfg['q'],cfg['L'],K,rc,mesh);t2=time.perf_counter()
                setup.append(dict(candidate_id=cid,config=cfg['name'],rep=rep,
                                  t_mesh_setup_s=t1-t0,t_particle_setup_s=t2-t1,t_setup_s=t2-t0))
            ops[cid]=op
        functions={cid:dict(complete=lambda op=op:op.apply(V),
                            real=lambda op=op:np.einsum('nab,nb->na',op.DS,V)-op.real_H(V),
                            mesh=lambda op=op:np.einsum('nab,nb->na',op.Dm,V)-op.mesh.apply(op.idx,op.wt,V),
                            profile=lambda op=op:R.profile_mesh(op,V)) for cid,op in ops.items()}
        for cid,fn in functions.items():
            pv,_=fn['profile']();op=ops[cid]
            assert np.linalg.norm(pv-op.mesh.apply(op.idx,op.wt,V))/np.linalg.norm(pv)<1e-12
            assert np.linalg.norm(fn['real']()+fn['mesh']()-fn['complete']())/np.linalg.norm(fn['complete']())<1e-12
        for rd in range(rounds):
            for phase in rng.permutation(('complete','real','mesh','profile')):
                for cid in rng.permutation(list(ops)):
                    for _ in range(3):functions[cid][phase]()
                for rep in range(reps):
                    for cid in rng.permutation(list(ops)):
                        fn=functions[cid][phase];fn()
                        if phase=='profile':
                            _,parts=fn()
                            rowraw=dict(candidate_id=cid,config=cfg['name'],round=rd,rep=rep,phase=phase,**parts)
                        else:
                            t0=time.perf_counter();fn();dt=time.perf_counter()-t0
                            rowraw=dict(candidate_id=cid,config=cfg['name'],round=rd,rep=rep,phase=phase,t_s=dt)
                        raw.append(rowraw)
        for row in selected:
            cid=row['candidate_id'];summary=dict(candidate_id=cid,config=cfg['name'],rounds=rounds,reps=reps,
                                                n_samples=rounds*reps,n_images=ops[cid].n_images)
            for phase in ('complete','real','mesh'):
                ts=[r['t_s'] for r in raw if r['candidate_id']==cid and r['config']==cfg['name'] and r['phase']==phase]
                stats=C.tstats(ts)
                summary.update({f't_{phase}_s':stats['median'],f't_{phase}_iqr_rel':stats['iqr_rel']})
            profiles=[r for r in raw if r['candidate_id']==cid and r['config']==cfg['name'] and r['phase']=='profile']
            for key in ('t_spread_s','t_gather_s','t_spread_gather_s','t_fft_s','t_profile_mesh_s'):
                summary[key]=float(np.median([r[key] for r in profiles]))
            for key in ('t_mesh_setup_s','t_particle_setup_s','t_setup_s'):
                summary[key]=float(np.median([r[key] for r in setup if r['candidate_id']==cid and r['config']==cfg['name']]))
            out.append(summary)
        X.write(X.RESULTS/'final_timing_raw.csv',raw)
        X.write(X.RESULTS/'final_timing.csv',out)
        X.write(X.RESULTS/'final_setup_raw.csv',setup)
        checkpoints.append(dict(config=cfg['name'],elapsed_s=time.time()-started))
        X.log(f'final interleaved timing {cfg["name"]}: {len(selected)} designs, {rounds*reps} samples/phase')
    X.jsave(X.RESULTS/'timing_protocol.json',dict(rounds=rounds,reps=reps,threads=1,
            started_unix=started,elapsed_s=time.time()-started,checkpoints=checkpoints,
            selected_ids=[r['candidate_id'] for r in selected],reference_workers_running=False,
            warmup_per_phase_block=3,warmup_before_each_sample=1,
            candidate_order='random every repetition',phase_order='random each round',
            profile='identical Mesh.apply expressions, FFT includes influence and inverse, degree excluded from spread/gather'))
    X.log('final common timing complete')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stages',nargs='+',choices=('study','diagnostics','decompose','timing','all'))
    ap.add_argument('--rounds',type=int,default=9);ap.add_argument('--reps',type=int,default=11)
    ap.add_argument('--refresh-diagnostics',action='store_true')
    args=ap.parse_args();E.log_cmd(X.RESULTS)
    if (X.RESULTS/'analysis_provenance.json').exists():
        shutil.copy2(X.RESULTS/'analysis_provenance.json',X.RESULTS/f"analysis_provenance_{time.strftime('%Y%m%d_%H%M%S')}.json")
    X.jsave(X.RESULTS/'analysis_provenance.json',X.provenance(stages=args.stages,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in X.HERE.glob('*.py')}))
    search=R.Search()
    with threadpool_limits(limits=1):
        for stage in (('study','diagnostics','decompose','timing') if 'all' in args.stages else args.stages):
            X.log(f'START {stage}')
            if stage=='study':plan_study(search)
            elif stage=='diagnostics':measure_diagnostics(search,args.refresh_diagnostics)
            elif stage=='decompose':decompose(search)
            else:timing(search,args.rounds,args.reps)
            X.log(f'DONE {stage}')

if __name__=='__main__': main()
