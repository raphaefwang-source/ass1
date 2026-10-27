"""Fresh diagnostic profile of the unchanged Python A implementation."""
import cProfile
import pstats
import time
import bench_common as C
import run_benchmarks as old_run

def main():
    target=C.RESULTS/'profile_baseline.json'
    if target.exists():raise FileExistsError(target)
    meta=next(r for r in C.parent_rows('structure.csv') if r['N']==256 and r['family']=='uniform' and r['seed']==2 and r['epsilon']==1e-4)
    sel=next(r for r in C.parent_rows('selection.csv') if r['config']==meta['config'] and r['method']=='A' and r['alpha']==1.)
    system=C.make_system(meta);saved=C.load_saved(meta);edges=C.O.EdgeOU(system.edges)
    def step(seed):
        return old_run.sample_step(system,edges,saved['p'],sel['dt'],'A',int(sel['r_d']),int(sel['r_n']),int(sel['substeps']),C.np.random.default_rng(seed))
    step(C.SEED)
    rows=[]
    for i in range(7):
        start=time.perf_counter();out,diag=step(C.SEED+i+1);elapsed=time.perf_counter()-start
        rows.append(dict(config=meta['config'],N=256,rep=i,dt=sel['dt'],r_d=sel['r_d'],r_n=sel['r_n'],substeps=sel['substeps'],
                         t_total_step=elapsed,**{key:value for key,value in diag.items() if isinstance(value,(str,int,float,bool)) and key not in ('substeps','dt','r_d','r_n')}))
    profile=cProfile.Profile();profile.enable()
    for i in range(5):step(C.SEED+100+i)
    profile.disable();profile.dump_stats(str(C.LOGS/'baseline.prof'))
    stats=pstats.Stats(profile)
    functions=[]
    for (filename,line,name),(primitive,calls,self_time,cumulative,callers) in stats.stats.items():
        functions.append(dict(file=filename,line=line,function=name,primitive_calls=primitive,calls=calls,self_s=self_time,cumulative_s=cumulative))
    functions.sort(key=lambda r:r['cumulative_s'],reverse=True)
    C.write('profile_baseline_functions.csv',functions)
    C.write('profile_baseline_raw.csv',rows)
    fraction=float(C.np.median([(r['t_pairOU_forward']+r['t_pairOU_reverse'])/r['t_total_step'] for r in rows]))
    C.save(target,dict(config=meta['config'],N=256,edge_count=len(edges.i),n_images=system.op.n_images,
        unprofiled_samples=len(rows),profiled_steps=5,near_fraction_median=fraction,
        median_total_s=float(C.np.median([r['t_total_step'] for r in rows])),
        original_implementation=str(C.PARENT/'ou_core.py'),source_sha256=C.sha(C.PARENT/'ou_core.py'),
        selection_source_sha256=C.sha(C.PARENT/'results/selection.csv'),baseline_input_sha256=C.sha(C.baseline_input(meta)),
        role='fresh pre-optimization diagnostic profile only; final speedups use three-way formal batch',
        profiler='cProfile; cumulative phases overlap, do not sum nested function times',load=C.load_snapshot('baseline_profile')))
    print(dict(config=meta['config'],near_fraction_median=fraction,samples=len(rows)),flush=True)

if __name__=='__main__':main()
