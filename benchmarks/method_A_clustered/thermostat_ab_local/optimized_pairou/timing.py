"""Fresh randomized three-way frozen-step timings, after all accuracy stages."""
import argparse
import hashlib
import json
import time
import bench_common as C
import validate as V
from compiled_edge import CompiledEdgeOU
np=C.np
METHODS=('A-python','A-compiled','B')

def same_graph_hash(edge):
    return hashlib.sha256(edge.i.tobytes()+edge.j.tobytes()+edge.K.tobytes()).hexdigest()

def execute(system, engine, p, selection, label, seed):
    """Include actual needed Gaussian generation, all kernels and diagnostics."""
    start=time.perf_counter()
    rng=np.random.default_rng(seed)
    method='B' if label=='B' else 'A'
    k=int(selection['substeps']) if method=='A' else 1
    noise=V.draws(rng,k,len(engine.i) if method=='A' else 0,p.size)
    random_done=time.perf_counter()
    output,diag=V.finite_step(system,engine,p,float(selection['dt']),
        int(selection['r_d']),int(selection['r_n']),k,noise,method)
    elapsed=time.perf_counter()-start
    return output,diag,noise,elapsed,random_done-start

def setup(system_meta, dt, k, rep):
    system=C.make_system(system_meta)
    t=time.perf_counter();original=C.O.EdgeOU(system.edges,n_particles=len(system.q));py_ctor=time.perf_counter()-t
    t=time.perf_counter();compiled=CompiledEdgeOU(system.edges,n_particles=len(system.q),
        configuration_token=system.metadata['position_sha256']);c_ctor=time.perf_counter()-t
    if compiled.backend!='compiled':raise RuntimeError('Formal optimized timing requires an actual compiled backend')
    tau=dt/(2*k)
    t=time.perf_counter();original._coefficients(tau,1.,1.);py_coeff=time.perf_counter()-t
    t=time.perf_counter();prepared=compiled.prepare(tau,1.,1.);c_coeff=time.perf_counter()-t
    row=dict(V.metadata(system_meta),rep=rep,dt=dt,substeps=k,tau=tau,
        t_table_setup=system.metadata['t_table_setup'],t_config_setup=system.metadata['t_config_setup'],
        t_edge_build=system.metadata['t_near_edge_build'],
        t_python_edge_eigh=original.t_setup,t_compiled_edge_eigh=compiled.t_setup,
        t_python_edge_constructor=py_ctor,t_compiled_edge_constructor=c_ctor,
        t_python_coeff_setup=py_coeff,t_coeff_setup=c_coeff,
        coefficient_cache_hit=prepared['coefficient_cache_hit'],active_backend=compiled.backend,
        graph_sha256=compiled.graph_sha256,compiler_source_sha256=compiled.compile_info['source_sha256'],
        batch='pairou_threeway_local_20261027',
        scope='fresh production table/config/image-summed edge/eigh/cache coefficients; process start/common radial cache/dense verification excluded')
    base=row['t_table_setup']+row['t_config_setup']
    row.update(B_setup=base,A_python_setup=base+row['t_edge_build']+py_ctor+py_coeff,
               A_compiled_setup=base+row['t_edge_build']+c_ctor+c_coeff)
    return row

def main(args):
    if (C.RESULTS/'timing_done.json').exists():raise FileExistsError('Preserve completed timing evidence')
    if not all((C.RESULTS/(stage+'_done.json')).exists() for stage in ('accuracy','mc')):
        raise RuntimeError('Accuracy and MC must finish before formal timing')
    C.save('environment_timing.json',C.environment())
    cases=C.read('structure.csv')
    rows=[];setup_rows=[];loads=[C.load_snapshot('before')]
    started=time.perf_counter();order_rng=np.random.default_rng(C.SEED+30000)
    C.save('timing_protocol.json',dict(methods=METHODS,systems=len(cases),rounds=args.rounds,reps=args.reps,
        setup_reps=args.setup_reps,alpha=1.,primary_epsilon=1e-4,
        batch='pairou_threeway_local_20261027',global_seed=C.SEED,
        Gaussian_seed_recipe='2026102708+1000000+case_index*100000+round*1000+rep',
        timing_scope='seeded RNG construction, all needed Gaussian draws, unchanged OU/Krylov/FFT/reorth plus full near sweeps; diagnostics/hash serialization outside wall',
        paired_random_inputs='each method resets identical sample seed; both A use identical z,forward,reverse; B generates only its needed z',
        warmup='one full step per method after precomputing coefficients; compilation cost separate',
        selection_source_sha256=C.sha(C.PARENT/'results/selection.csv'),
        no_historical_timing_mix=True,
        inference='paired repeated samples share config/round; uncertainty should resample round/config blocks, not claim independent repeated hardware states'))
    for index,meta in enumerate(cases):
        C.log(f'timing {index+1}/{len(cases)} {meta["config"]}')
        system,saved,hashes=V.load_system(meta)
        original,compiled=V.engines(system)
        sa=next(r for r in V.selected_rows(meta['config'],1.) if r['method']=='A')
        sb=next(r for r in V.selected_rows(meta['config'],1.) if r['method']=='B')
        dt=float(sa['dt']);k=int(sa['substeps']);tau=dt/(2*k)
        original._coefficients(tau,1.,1.);compiled.prepare(tau,1.,1.)
        graph=same_graph_hash(original)
        if graph!=compiled.graph_sha256:raise RuntimeError('Different near graphs in formal timing')
        D,S=compiled._compiled_factors[(tau,1.,1.)]
        coeff_sha=hashlib.sha256(D.tobytes()+S.tobytes()).hexdigest()
        for rep in range(args.setup_reps):setup_rows.append(setup(meta,dt,k,rep))
        C.write('setup_raw.csv',setup_rows)
        engines={'A-python':original,'A-compiled':compiled,'B':original}
        selections={'A-python':sa,'A-compiled':sa,'B':sb}
        for label in METHODS:execute(system,engines[label],saved['p'],selections[label],label,C.SEED+900000+index)
        for rnd in range(args.rounds):
            for rep in range(args.reps):
                sample_seed=C.SEED+1000000+index*100000+rnd*1000+rep
                paired={}
                for position,label in enumerate(order_rng.permutation(METHODS)):
                    selection=selections[label];engine=engines[label]
                    output,diag,noise,elapsed,rng_time=execute(system,engine,saved['p'],selection,label,sample_seed)
                    full=int(diag['actual_full_actions']);long=int(diag['actual_long_actions'])
                    if full+long!=diag['matvec_count']:raise RuntimeError('Actual friction call count disagrees with Lanczos')
                    row=dict(V.metadata(meta),method=label,alpha=1.,dt=dt,r_d=selection['r_d'],r_n=selection['r_n'],
                        substeps=selection['substeps'],round=rnd,rep=rep,order_position=position,gaussian_seed=sample_seed,
                        t_total_step=elapsed,t_rng=rng_time,
                        t_near_forward=diag.get('t_pairOU_forward',0.),t_near_reverse=diag.get('t_pairOU_reverse',0.),
                        t_long_damp=diag.get('t_long_lanczos_damp',0.),t_long_noise=diag.get('t_long_lanczos_noise',0.),
                        t_full_damp=diag.get('t_full_lanczos_damp',0.),t_full_noise=diag.get('t_full_lanczos_noise',0.),
                        t_full_B=elapsed if label=='B' else 0.,t_reorth=diag['t_reorth'],t_ritz_eigh=diag['t_ritz_eigh'],
                        matvec_count=diag['matvec_count'],actual_full_actions=full,actual_long_actions=long,FFT=6*(full+long),
                        near_actions=full,near_image_actions=full*system.op.n_images,
                        near_pair_updates=diag.get('pair_updates',0),edge_count=len(original.i),
                        active_backend=compiled.backend if label=='A-compiled' else 'python-reference',
                        coefficient_cache_size=compiled.coefficient_cache_size if label=='A-compiled' else None,
                        graph_sha256=graph,coefficient_sha256=coeff_sha if label=='A-compiled' else None,
                        gaussian_sha256=V.gaussian_hash(noise),first_z_sha256=C.array_sha(noise['z'][0]),
                        p_sha256=hashes['p_sha256'],baseline_input_sha256=hashes['baseline_input_sha256'],
                        compiler_source_sha256=compiled.compile_info['source_sha256'],
                        batch='pairou_threeway_local_20261027',gaussian_generation_included=True,
                        compiler_warmed=True,threads_requested=1)
                    rows.append(row);paired[label]=(output,row)
                if paired['A-python'][1]['gaussian_sha256']!=paired['A-compiled'][1]['gaussian_sha256']:
                    raise RuntimeError('A Gaussian inputs differ in paired timing')
                if len({p[1]['first_z_sha256'] for p in paired.values()})!=1:
                    raise RuntimeError('First full/long Gaussian differs across methods')
                err=V.relative(paired['A-compiled'][0],paired['A-python'][0])
                if err>1e-11:raise RuntimeError('Paired formal-step path exceeds numerical-equivalence tolerance')
                for _,r in paired.values():r['paired_A_path_error']=err
        C.write('timing_raw.csv',rows);loads.append(C.load_snapshot(meta['config']));C.save('system_load.json',loads)
    C.save('timing_done.json',dict(elapsed_s=time.perf_counter()-started,rows=len(rows),systems=len(cases),
        rounds=args.rounds,reps=args.reps,setup_rows=len(setup_rows),methods=METHODS))
    print(dict(rows=len(rows),systems=len(cases),elapsed_s=time.perf_counter()-started),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rounds',type=int,default=5);p.add_argument('--reps',type=int,default=7)
    p.add_argument('--setup-reps',type=int,default=3);main(p.parse_args())
