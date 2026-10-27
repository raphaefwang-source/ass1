"""Fresh common serial timing, after all accuracy/reference workers have exited."""
import argparse,gc,hashlib,json,time
from collections import Counter
from pathlib import Path
import protocol as P
from run_timing import system_snapshot,memory_record,table_profile
np=P.np;S=P.S;E=P.E
EPS=P.EPSILONS
RNG_SEED=20261029
BATCH='final_four_step_common_serial'


def pool_rows():
    rows=P.read(P.RESULTS/'validation_summary.csv')+P.read(P.RESULTS/'cross_validation_summary.csv')
    return list({r['id']:r for r in rows if r.get('status')=='ok' and int(r.get('n_configs') or 0)==9 and r.get('max_E_G') is not None}.values())


def hint(row):return float(row.get('t_screen_s') or float('inf'))


def cross():
    target=P.RESULTS/'cross_validation.csv'
    rows=P.read(P.RESULTS/'validation_summary.csv');candidates={}
    for eps in EPS:
        for source,destination in [('PPPM-grid-PSD','PPPM-original'),('PPPM-original','PPPM-grid-PSD')]:
            own=sorted([r for r in rows if r['method']==source and r['max_E_G']<=eps],key=hint)
            if not own:continue
            for r in own[:2]:
                c=P.candidate(destination,P.parameters(r),epsilon=eps,stage='cross-method-control',
                              origin='freshly validated '+source+' top-two proxy candidate; independent acceptance for destination')
                candidates[c['id']]=c
    raw=P.read(target)
    completed={r['id'] for r in P.summarize_validation(raw) if r['n_configs']==9}
    pending=[c for c in candidates.values() if c['id'] not in completed]
    P.save(P.RESULTS/'cross_plan.json',dict(candidates=list(candidates.values()),
        policy='top two fresh validated proxy candidates per source method/epsilon; reciprocal independent validation; existing rows retained'))
    raw.extend(P.validate_batch(pending,workers=2) if pending else [])
    P.write(target,raw);P.write(P.RESULTS/'cross_validation_summary.csv',P.summarize_validation(raw))
    P.log('cross-method controls '+str(len(candidates)))


def prescreen():
    target=P.RESULTS/'serial_prescreen_raw.csv'
    if target.exists():P.log('preserved existing serial prescreen');return
    rows=[r for r in pool_rows() if r['max_E_G']<=max(EPS)]
    cfg=P.configurations()[0];rng=np.random.default_rng(RNG_SEED);v=rng.normal(size=(256,3));v-=v.mean(0)
    raw=[]
    for rnd in range(3):
        for ix in rng.permutation(len(rows)):
            row=rows[ix];op=S.build(row['method'],cfg,P.parameters(row))
            for _ in range(3):op.apply(v)
            for rep in range(3):
                op.apply(v);start=time.perf_counter_ns();op.apply(v);dt=(time.perf_counter_ns()-start)*1e-9
                raw.append(dict(id=row['id'],method=row['method'],config=cfg['name'],round=rnd,rep=rep,seconds=dt,
                                phase='matvec',batch='serial_prescreen_only',threads=1))
            del op
        P.log(f'common serial prescreen {rnd+1}/3, {len(rows)} candidates')
    P.write(target,raw)
    ranks={r['id']:float(np.median([x['seconds'] for x in raw if x['id']==r['id']])) for r in rows}
    selected={}
    for eps in EPS:
        for method in P.METHODS:
            feasible=sorted([r for r in rows if r['method']==method and r['max_E_G']<=eps],key=lambda r:ranks[r['id']])
            for r in feasible[:2]:selected[r['id']]=dict(r,prescreen_median_s=ranks[r['id']],selection='top2 per epsilon/method in new serial prescreen')
    crossids={r['id'] for r in P.read(P.RESULTS/'cross_validation_summary.csv') if r.get('status')=='ok' and r['max_E_G']<=max(EPS)}
    for r in rows:
        if r['id'] in crossids:selected[r['id']]=dict(r,prescreen_median_s=ranks[r['id']],selection='feasible cross-method control')
    P.write(P.RESULTS/'timed_candidates.csv',list(selected.values()))


def median_config(rows,id_,field='seconds'):
    own=[r for r in rows if r['id']==id_]
    return float(np.median([np.median([r[field] for r in own if r['config']==cfg['name']]) for cfg in P.configurations()]))


def formal(rounds=5,reps=7):
    if (P.RESULTS/'final_timing.csv').exists():P.log('preserved existing final timing');return
    rows=P.read(P.RESULTS/'timed_candidates.csv')
    if not rows:raise RuntimeError('Run serial prescreen after fresh nine-config validation')
    rows.append(dict(id='direct-dense',method='direct-dense',n_configs=9,max_E_G=0.,
                     selection='same independent numerical reference matrix, cached dense action',
                     error_role='zero against itself; independent reference truncation remains documented',
                     setup_status='fresh direct assembly not timed; cached matrix binding is not assembly'))
    P.save(P.RESULTS/'timing_plan.json',dict(candidates=rows,rounds=rounds,reps=reps,threads=1,batch=BATCH,seed=RNG_SEED,
        objective='median of nine configuration medians; fresh common batch only',
        direct_dense_setup='reference matrices reused from immutable independent caches; full assembly setup and amortized full direct cost unavailable',
        init_eta_max=P.ETA_MAX,PSD_checks_or_bounds_in_acceptance=False))
    rng=np.random.default_rng(RNG_SEED)
    raw=P.read(P.RESULTS/'timing_raw.csv');tables=P.read(P.RESULTS/'table_setup_raw.csv');setups=P.read(P.RESULTS/'config_setup_raw.csv')
    memories=P.read(P.RESULTS/'memory.csv');snapshots=[system_snapshot('before_common_formal_batch')]
    cfg0=P.configurations()[0]
    for row in rows:
        if row['method'] in P.METHODS:E.kernel(.2,row['xi'],0.)
    for ix in rng.permutation(len(rows)):
        row=rows[ix]
        if row['method']=='direct-dense':continue
        for rep in range(3):
            if any(r['id']==row['id'] and r['rep']==rep for r in tables):continue
            gc.collect();t=time.perf_counter_ns();mesh=S.table(row['method'],cfg0['L'],P.parameters(row));dt=(time.perf_counter_ns()-t)*1e-9
            tables.append(dict(id=row['id'],method=row['method'],rep=rep,seconds=dt,batch=BATCH,**table_profile(mesh)));del mesh
        P.write(P.RESULTS/'table_setup_raw.csv',tables)
    # A parameter table is shared over the nine configurations of this same box.
    meshes={r['id']:S.table(r['method'],cfg0['L'],P.parameters(r)) for r in rows if r['method'] in P.METHODS}
    for cfg in P.configurations():
        v=rng.normal(size=(256,3));v-=v.mean(0);flat=v.ravel();GD,_=S.ref(cfg,.2)
        for rnd in range(rounds):
            counts=Counter((r['id'],r['config'],r['round']) for r in raw)
            completed={key for key,count in counts.items() if count==reps}
            for ix in rng.permutation(len(rows)):
                row=rows[ix]
                if (row['id'],cfg['name'],rnd) in completed:continue
                if row['method']=='direct-dense':fn=lambda:GD@flat;op=None
                else:
                    op=S.build(row['method'],cfg,P.parameters(row),meshes[row['id']]);fn=lambda:op.apply(v)
                for _ in range(3):fn()
                for rep in range(reps):
                    fn();t=time.perf_counter_ns();fn();dt=(time.perf_counter_ns()-t)*1e-9
                    raw.append(dict(id=row['id'],method=row['method'],config=cfg['name'],family=cfg['family'],seed=cfg['seed'],
                                    round=rnd,rep=rep,phase='matvec',seconds=dt,batch=BATCH,threads=1))
                del op
            P.write(P.RESULTS/'timing_raw.csv',raw)
            P.save(P.RESULTS/'timing_progress.json',dict(config=cfg['name'],round=rnd,samples=len(raw),batch=BATCH))
        for ix in rng.permutation(len(rows)):
            row=rows[ix]
            if row['method']=='direct-dense':continue
            for rep in range(3):
                if any(r['id']==row['id'] and r['config']==cfg['name'] and r['rep']==rep for r in setups):continue
                t=time.perf_counter_ns();op=S.build(row['method'],cfg,P.parameters(row),meshes[row['id']]);dt=(time.perf_counter_ns()-t)*1e-9
                setups.append(dict(id=row['id'],method=row['method'],config=cfg['name'],rep=rep,seconds=dt,
                                   n_images=op.n_images,batch=BATCH,**{'part_'+k+'_s':v for k,v in op.t_setup_parts.items()}))
                if rep==0:memories.append(dict(id=row['id'],method=row['method'],config=cfg['name'],**memory_record(op)))
                del op
        P.write(P.RESULTS/'config_setup_raw.csv',setups);P.write(P.RESULTS/'memory.csv',memories)
        snapshots.append(system_snapshot(cfg['name']));P.save(P.RESULTS/'system_load.json',snapshots)
        P.log(f'common formal {cfg["name"]}, {len(rows)} candidates')
    summaries=[]
    for row in rows:
        own=[r for r in raw if r['id']==row['id']]
        config_medians=[float(np.median([r['seconds'] for r in own if r['config']==cfg['name']])) for cfg in P.configurations()]
        q25,q75=np.quantile([r['seconds'] for r in own],[.25,.75])
        direct=row['method']=='direct-dense'
        summaries.append(dict(row,median_matvec_s=float(np.median(config_medians)),
            min_config_median_s=min(config_medians),max_config_median_s=max(config_medians),
            q25_matvec_s=float(q25),q75_matvec_s=float(q75),n_samples_per_config=rounds*reps,
            median_table_setup_s=None if direct else float(np.median([r['seconds'] for r in tables if r['id']==row['id']])),
            median_config_setup_s=None if direct else median_config(setups,row['id']),
            timing_batch=BATCH,available=True))
    P.write(P.RESULTS/'final_timing.csv',summaries)
    P.save(P.RESULTS/'timing_context.json',dict(batch=BATCH,rounds=rounds,reps=reps,n_configs=9,threads=1,seed=RNG_SEED,
        threadpools=S.threadpool_info(),load_average=list(__import__('os').getloadavg()),
        warmup='3 per candidate/config/round and one immediately before every sample',
        setup='3 fresh tables per candidate, 3 fresh config builds per configuration; common kernel primitives warm',
        direct_dense='cached independent numerical Gamma_D; full assembly cost not measured, setup/amortization unavailable',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.HERE.glob('*.py')}))
    summarize()


def summarize():
    rows=P.read(P.RESULTS/'final_timing.csv');eligible=pool_rows();best=[];amortized=[]
    for eps in EPS:
        for method in (*P.METHODS,'direct-dense'):
            feasible=sorted([r for r in rows if r['method']==method and r['max_E_G']<=eps],key=lambda r:r['median_matvec_s'])
            if feasible:
                row=dict(feasible[0],epsilon=eps,available=True,status='fastest common-batch timed, fully nine-configuration empirically feasible candidate',
                         n_validated_feasible=sum(r['method']==method and r['max_E_G']<=eps for r in eligible),n_timed_feasible=len(feasible))
            else:row=dict(epsilon=eps,method=method,available=False,status='no feasible candidate in completed finite search/timing budget',max_E_G=None,median_matvec_s=None)
            best.append(row)
            if row['available']:
                for reuse in (1,10,100,1000):
                    for actions in (1,10,100,1000):
                        table=row.get('median_table_setup_s');setup=row.get('median_config_setup_s')
                        total=table/reuse+setup+actions*row['median_matvec_s'] if table is not None and setup is not None else None
                        amortized.append(dict(method=method,epsilon=eps,id=row['id'],table_reuse_steps=reuse,matvecs_per_step=actions,
                            total_step_s=total,available=total is not None,cost_model='table/reuse_steps + fresh configuration setup + actions*cached full action',
                            status='computed cost model, not a measured trajectory' if total is not None else 'direct assembly setup unavailable; no synthetic total'))
    P.write(P.RESULTS/'best_by_epsilon.csv',best);P.write(P.RESULTS/'amortization.csv',amortized)
    raw=P.read(P.RESULTS/'timing_raw.csv');rng=np.random.default_rng(8231);ratios=[]
    for eps in EPS:
        selected={r['method']:r for r in best if r['epsilon']==eps and r['available']}
        for method,refmethod in [('PPPM-grid-PSD','PPPM-original'),('PPPM-grid-PSD','B2-positive'),('PPPM-grid-PSD','direct-dense')]:
            if method not in selected or refmethod not in selected:continue
            a,b=selected[method],selected[refmethod];blocks=[]
            for cfg in P.configurations():
                for rnd in sorted({r['round'] for r in raw}):
                    va=[r['seconds'] for r in raw if r['id']==a['id'] and r['config']==cfg['name'] and r['round']==rnd]
                    vb=[r['seconds'] for r in raw if r['id']==b['id'] and r['config']==cfg['name'] and r['round']==rnd]
                    if va and vb:blocks.append(float(np.median(va)/np.median(vb)))
            boot=np.array([np.median(rng.choice(blocks,len(blocks),replace=True)) for _ in range(5000)])
            low,high=np.quantile(boot,[.025,.975])
            ratios.append(dict(epsilon=eps,method=method,reference_method=refmethod,id=a['id'],reference_id=b['id'],
                ratio_median_paired_blocks=float(np.median(blocks)),ratio_ci95_low=float(low),ratio_ci95_high=float(high),
                blocks=len(blocks),resamples=5000,batch=BATCH,conditional_on_measured_candidate_selection=True))
    P.write(P.RESULTS/'paired_speed_ratios.csv',ratios)


def components():
    if (P.RESULTS/'components_raw.csv').exists():P.log('preserved existing components');return
    best=P.read(P.RESULTS/'best_by_epsilon.csv');points={r['id']:r for r in best if r.get('available') and r['method'] in P.METHODS}
    rng=np.random.default_rng(61129);raw=[]
    for cfg in P.configurations():
        v=rng.normal(size=(256,3));v-=v.mean(0)
        for row in points.values():
            op=S.build(row['method'],cfg,P.parameters(row));grids=op.spread(v);phi=op.fft_influence(grids)
            np.testing.assert_allclose(op.apply_staged(v),op.apply(v),rtol=1e-12,atol=1e-10)
            functions=dict(near=lambda:np.einsum('nab,nb->na',op.DS,v)-op.real_H(v),
                fft_influence=lambda:op.fft_influence(grids),spread_gather=lambda:(op.spread(v),op.gather(phi)),
                mesh=lambda:np.einsum('nab,nb->na',op.Dm,v)-op.mesh.apply(op.idx,op.wt,v),total=lambda:op.apply(v))
            for phase in rng.permutation(list(functions)):
                for rep,dt in enumerate(P.C.median_time(functions[phase],9,warm=3)):
                    raw.append(dict(id=row['id'],method=row['method'],config=cfg['name'],phase=phase,rep=rep,seconds=float(dt),
                                    batch='isolated_component_diagnostics',timing_note='separately timed component medians need not sum to full action'))
            del op,grids,phi
        P.write(P.RESULTS/'components_raw.csv',raw)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('cross','prescreen','formal','summarize','components','all'))
    parser.add_argument('--rounds',type=int,default=5);parser.add_argument('--reps',type=int,default=7);a=parser.parse_args()
    if a.stage in ('cross','all'):cross()
    if a.stage in ('prescreen','all'):prescreen()
    if a.stage in ('formal','all'):formal(a.rounds,a.reps)
    if a.stage=='summarize':summarize()
    if a.stage in ('components','all'):components()


if __name__=='__main__':main()
