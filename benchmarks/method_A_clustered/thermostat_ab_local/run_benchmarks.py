"""Actual local frozen-OU experiments. Stages run serially; raw evidence is saved."""
import argparse
import time
import traceback
import protocol as P
np=P.np
from scipy import linalg, stats
import spatial as S
from ou_core import EdgeOU, lanczos_action, ou_a, ou_b

def config_id(N, family, seed, epsilon):
    return f'N{N}_{family}_s{seed}_eps{epsilon:g}'

def plan():
    cases=[(N,'uniform',2,1e-4) for N in (2,8)]
    cases += [(N,f,s,1e-4) for N in (16,64,256) for f in ('uniform','moderate','strong') for s in (1,2,3)]
    cases += [(128,f,2,1e-4) for f in ('uniform','moderate','strong')]
    cases += [(256,'uniform',2,e) for e in (1e-3,1e-5)]
    return cases

def rowmeta(N,family,seed,epsilon):
    return dict(config=config_id(N,family,seed,epsilon),N=N,family=family,seed=seed,epsilon=epsilon)

def make(meta):
    return S.make_system(N=int(meta['N']),family=meta['family'],seed=int(meta['seed']),epsilon=float(meta['epsilon']))

def edge_object(system):
    return EdgeOU(system.edges)

def sample_step(system, edges, p, dt, method, rd, rn, substeps, rng, noise=True, reference=None):
    action_calls=0
    def counted_action(v):
        nonlocal action_calls
        action_calls+=1
        return system.full_action(v) if method=='B' else system.long_action(v)
    if method=='B':
        z=rng.normal(size=p.size) if noise else np.zeros(p.size)
        out,diag=ou_b(counted_action,p,z,dt,rd,rn)
        diag.update(actual_full_actions=action_calls,actual_long_actions=0)
        if reference is not None:
            ideal=reference[0]@p+reference[1]@z
            diag.update(ideal_output=ideal,coupled_path_relative_error=float(np.linalg.norm(out-ideal)/max(np.linalg.norm(ideal),1e-300)))
        return out,diag
    v=p.copy(); ideal=p.copy(); all_diags=[]
    start=time.perf_counter()
    for _ in range(substeps):
        z=rng.normal(size=p.size) if noise else np.zeros(p.size)
        ef=rng.normal(size=(len(edges.i),3)) if noise else np.zeros((len(edges.i),3))
        er=rng.normal(size=(len(edges.i),3)) if noise else np.zeros((len(edges.i),3))
        v,d=ou_a(counted_action,v,z,ef,er,edges,dt/substeps,rd,rn)
        if reference is not None:
            ideal=edges.sweep(ideal,ef,dt/(2*substeps))[0]
            ideal=reference[0]@ideal+reference[1]@z
            ideal=edges.sweep(ideal,er,dt/(2*substeps),reverse=True)[0]
        all_diags.append(d)
    merged={}
    for d in all_diags:
        for k,vv in d.items():
            if isinstance(vv,(int,float,np.number)) and not isinstance(vv,bool):
                merged[k]=merged.get(k,0)+vv
    merged['step_wall_s']=time.perf_counter()-start
    merged['substeps']=substeps
    merged.update(actual_full_actions=0,actual_long_actions=action_calls)
    if reference is not None:
        merged.update(ideal_output=ideal,coupled_path_relative_error=float(np.linalg.norm(v-ideal)/max(np.linalg.norm(ideal),1e-300)))
    return v,merged

def path(action,dense,dt,probes,meta,operator,alpha,rows,tolerance=P.KRYLOV_TOL):
    ref=P.DenseOU(dense); R,F,C=ref.matrices(dt)
    ranks=sorted(set(min(3*int(meta['N'])-3,r) for r in (2,4,6,8,12,16,24,32,48,64,96,128)))
    chosen={}; chosen_error={}
    for kind,target in [('damp',R),('noise',F)]:
        previous={}; best=None
        for rank in ranks:
            maximum=0.
            for name,v in probes.items():
                if name=='COM':continue
                out,diag=lanczos_action(action,v,rank,dt,kind)
                truth=target@v; error=float(np.linalg.norm(out-truth)/max(np.linalg.norm(truth),1e-300))
                change=float(np.linalg.norm(out-previous[name])/max(np.linalg.norm(out),1e-300)) if name in previous else None
                rows.append(dict(meta,operator=operator,alpha=alpha,dt=dt,kind=kind,probe=name,rank=rank,
                                 error=error,step_change=change,tolerance=tolerance,
                                 **{'diag_'+k:v for k,v in diag.items() if isinstance(v,(str,int,float,bool,np.number))}))
                previous[name]=out
                maximum=max(maximum,error)
            if maximum<=tolerance:
                best=rank;chosen_error[kind]=maximum;break
        chosen[kind]=best
    return chosen,chosen_error

def accuracy(args):
    if (P.RESULTS/'accuracy_done.json').exists():raise FileExistsError('Accuracy evidence already exists')
    P.save(P.RESULTS/'environment_accuracy.json',P.environment())
    P.save(P.RESULTS/'protocol.json',dict(cases=[rowmeta(*c) for c in plan()],alphas=P.ALPHAS,
        global_seed=P.SEED,spatial_parameters='PR7 Grid PSD best tested; same discrete graph in A and B',
        rank_probes=['slow','fast','random','input_p','input_z'],rank_tolerance=P.KRYLOV_TOL,dynamical_tolerance=P.TOL,
        mc_cases='uniform seed2 N2/8/16/64/128/256 plus moderate/strong N16 seed2',
        mc_samples_small=args.mc_small,mc_samples_medium=args.mc_medium,
        mc_alpha=1.,CI='pointwise Student-t interval on independent squared centered directional samples; asymptotic delta-method for sample variance',
        selection='smallest tested separate damp/noise ranks passing five saved probes; A splitting smallest tested substeps passing dense mean and covariance error',
        coverage_limits='rank limits 128, splitting substeps<=32; no universal rank guarantee; no trajectory test',
        covariance_note='finite vector-dependent Lanczos is nonlinear: no column-assembly covariance assertion'))
    structures=[];rankrows=[];splits=[];selections=[];setups=[];failures=[];modes=[];verifications=[]
    start_all=time.perf_counter()
    for index,case in enumerate(plan()):
        meta=rowmeta(*case);P.log(f"accuracy {index+1}/{len(plan())}: {meta['config']}")
        try:
            system=make(meta); edges=edge_object(system)
            G=system.full_dense; GL=system.long_dense; GS=system.near_dense
            full=P.DenseOU(G);long=P.DenseOU(GL)
            rng=np.random.default_rng(P.SEED+index)
            probes=full.probes(rng)
            p=P.project(rng.normal(size=3*case[0]))+np.tile([.2,-.1,.3],case[0]);z=rng.normal(size=p.size)
            selection_probes=dict(probes,input_p=P.project(p)/np.linalg.norm(P.project(p)),input_z=P.project(z)/np.linalg.norm(P.project(z)))
            q=system.q
            P.INPUTS.mkdir(exist_ok=True)
            np.savez_compressed(P.INPUTS/(meta['config']+'.npz'),q=q,L=system.L,G=G,GS=GS,GL=GL,
                p=p,z=z,slow=probes['slow'],fast=probes['fast'],random=probes['random'])
            structure=dict(meta,**system.metadata)
            tests=[P.project(rng.normal(size=p.size)) for _ in range(3)]
            structure.update(lambda_min=full.raw_min,lambda_max=full.lmax,
                near_min=P.DenseOU(GS).raw_min,long_min=long.raw_min,
                mf_full_error=max(float(np.linalg.norm(system.full_action(v)-G@v)/max(np.linalg.norm(G@v),1e-300)) for v in tests),
                mf_long_error=max(float(np.linalg.norm(system.long_action(v)-GL@v)/max(np.linalg.norm(GL@v),1e-300)) for v in tests),
                momentum_residual=float(np.linalg.norm(G@np.tile(np.eye(3),(case[0],1)))/max(np.linalg.norm(G),1e-300)),
                symmetry_residual=float(np.linalg.norm(G-G.T)/max(np.linalg.norm(G),1e-300)),
                edge_count=len(edges.i),edge_eigh_time_s=getattr(edges,'t_setup',None),
                input_sha256=P.sha(P.INPUTS/(meta['config']+'.npz')))
            if hasattr(S,'independent_reference'):
                reference=S.independent_reference(system)
                if isinstance(reference,tuple):
                    GD, info=reference; structure.update(info)
                    structure['spatial_error']=P.norm2(G-GD)/P.norm2(GD)
            for alpha in P.ALPHAS:
                dt=alpha/full.lmax;R,F,C=full.matrices(dt)
                identity=P.norm2(R@full.Pi@R.T+C-full.Pi)
                structure['dense_fdt_residual']=max(structure.get('dense_fdt_residual',0.),identity)
                ranksB,errorsB=path(lambda v:G@v,G,dt,selection_probes,meta,'full',alpha,rankrows)
                split_errors=[]
                for k in (1,2,4,8,16,32):
                    RA=P.ideal_a_mean(edges,GL,dt,k)
                    CA=full.Pi-RA@full.Pi@RA.T
                    me=P.norm2(RA-R);ce=P.norm2(CA-C)
                    prev=split_errors[-1]['mean_operator_error'] if split_errors else None
                    order=float(np.log2(prev/me)) if prev and me>1e-14 else None
                    sr=dict(meta,alpha=alpha,dt=dt,substeps=k,mean_operator_error=me,cov_operator_error=ce,observed_order=order,
                            ideal_invariance_residual=P.norm2(RA@full.Pi@RA.T+CA-full.Pi))
                    splits.append(sr);split_errors.append(sr)
                    if k>=8 and max(me,ce)<=P.TOL:break
                feasible=next((r for r in split_errors if max(r['mean_operator_error'],r['cov_operator_error'])<=P.TOL),None)
                k=int(feasible['substeps']) if feasible else 32
                ranksA,errorsA=path(lambda v:GL@v,GL,dt/k,selection_probes,meta,'long',alpha,rankrows,tolerance=P.KRYLOV_TOL/k)
                for method,ranks,errs,steps,split in [('B',ranksB,errorsB,1,0.),('A',ranksA,errorsA,k,max(feasible['mean_operator_error'],feasible['cov_operator_error']) if feasible else None)]:
                    available=all(v is not None for v in ranks.values()) and split is not None
                    select=dict(meta,alpha=alpha,dt=dt,method=method,r_d=ranks['damp'],r_n=ranks['noise'],substeps=steps,
                                krylov_damp_error=errs.get('damp'),krylov_noise_error=errs.get('noise'),split_error=split,
                                tolerance=P.TOL,available=available,rank_domain='five registered directions, dense same-matrix actions')
                    selections.append(select)
                    if not available:continue
                    action=system.full_action if method=='B' else system.long_action
                    reference=full if method=='B' else long
                    RR,FF,_=reference.matrices(dt/steps)
                    for kind,target,v in [('damp',RR,P.project(p)),('noise',FF,P.project(z))]:
                        out,d=lanczos_action(action,v,int(ranks[kind]),dt/steps,kind)
                        denseout,_=lanczos_action(lambda vv: (G if method=='B' else GL)@vv,v,int(ranks[kind]),dt/steps,kind)
                        verifications.append(dict(meta,alpha=alpha,method=method,kind=kind,
                            rank=ranks[kind],matrixfree_vs_dense_lanczos=float(np.linalg.norm(out-denseout)/max(np.linalg.norm(denseout),1e-300)),
                            exact_action_error=float(np.linalg.norm(out-target@v)/max(np.linalg.norm(target@v),1e-300))))
                    RA=P.ideal_a_mean(edges,GL,dt,steps) if method=='A' else R
                    for name,v in probes.items():
                        out,d=sample_step(system,edges,v,dt,method,int(ranks['damp']),int(ranks['noise']),steps,rng,noise=False)
                        modes.append(dict(meta,alpha=alpha,dt=dt,method=method,direction=name,
                            exact_decay=float(v@R@v),ideal_method_decay=float(v@RA@v),
                            finite_decay=float(v@out),
                            finite_mean_action_error=float(np.linalg.norm(out-RA@v)/max(np.linalg.norm(RA@v),1e-300)),
                            full_exact_mean_action_error=float(np.linalg.norm(out-R@v)),
                            COM_error=float(np.linalg.norm(out.reshape(-1,3).sum(0)-v.reshape(-1,3).sum(0)))))
            structures.append(structure)
            setups.append(dict(meta,**{k:v for k,v in system.metadata.items() if k.startswith('t_')},t_near_edge_eigh=getattr(edges,'t_setup',None)))
        except Exception as exc:
            traceback.print_exc();failures.append(dict(meta,error=str(exc)))
        for filename,rows in [('structure.csv',structures),('rank_errors.csv',rankrows),('splitting.csv',splits),('selection.csv',selections),('setup.csv',setups),('mode_decay.csv',modes),('matrixfree_checks.csv',verifications)]:P.write(filename,rows)
        P.save(P.RESULTS/'failures.json',failures)
    P.save(P.RESULTS/'accuracy_done.json',dict(elapsed_s=time.perf_counter()-start_all,systems=len(structures),failures=len(failures)))

def chosen_cases():
    return [r for r in P.read('structure.csv') if r['epsilon']==1e-4 and r['seed']==2 and
            (r['family']=='uniform' or int(r['N'])==16)]

def mc(args):
    if (P.RESULTS/'mc_done.json').exists():raise FileExistsError('MC evidence already exists')
    rows=[];fullcov=[];start_all=time.perf_counter()
    for index,meta in enumerate(chosen_cases()):
        N=int(meta['N']);system=make(meta);edges=edge_object(system);dense=P.DenseOU(system.full_dense)
        rng=np.random.default_rng(P.SEED+10000+index)
        saved=np.load(P.INPUTS/(meta['config']+'.npz'));p=saved['p'];dt=1./dense.lmax
        probes={name:saved[name].copy() for name in ('slow','fast','random')}
        probes['COM']=np.tile([1.,0.,0.],N)/np.sqrt(N)
        R,F,C=dense.matrices(dt)
        sample_count=args.mc_small if N<=16 else args.mc_medium
        sels=[r for r in P.read('selection.csv') if r['config']==meta['config'] and r['alpha']==1. and r['available']]
        for select in sels:
            method=select['method'];k=int(select['substeps']);rd=int(select['r_d']);rn=int(select['r_n'])
            RA=P.ideal_a_mean(edges,system.long_dense,dt,k) if method=='A' else R
            CA=dense.Pi-RA@dense.Pi@RA.T
            local_reference=P.DenseOU(system.long_dense).matrices(dt/k)[:2] if method=='A' else (R,F)
            for phase in ('conditional','equilibrium'):
                P.log(f"MC {meta['config']} {method} {phase} samples={sample_count}")
                samples=[];ideal_samples=[];path_errors=[];max_com=0.
                for sample in range(sample_count):
                    # Shared physical input and first full/long Gaussian across A/B;
                    # edge noises and later internal substeps remain independent draws.
                    sample_seed=P.SEED+100000+index*100000+(10000 if phase=='equilibrium' else 0)+sample
                    sample_rng=np.random.default_rng(sample_seed)
                    p0=p if phase=='conditional' else P.project(sample_rng.normal(size=3*N))+np.tile([.2,-.1,.3],N)
                    out,diagnostic=sample_step(system,edges,p0,dt,method,rd,rn,k,sample_rng,reference=local_reference)
                    max_com=max(max_com,float(np.linalg.norm(out.reshape(N,3).sum(0)-p0.reshape(N,3).sum(0))))
                    samples.append(out)
                    ideal_samples.append(diagnostic['ideal_output']);path_errors.append(diagnostic['coupled_path_relative_error'])
                samples=np.array(samples);ideal_samples=np.array(ideal_samples)
                np.savez_compressed(P.INPUTS/f"mc_{meta['config']}_{method}_{phase}.npz",samples=samples,ideal_samples=ideal_samples,path_errors=path_errors)
                centered=samples-samples.mean(axis=0)
                if N<=16:
                    empirical=centered.T@centered/(sample_count-1)
                    target=CA if phase=='conditional' else dense.Pi
                    fullcov.append(dict(config=meta['config'],N=N,method=method,phase=phase,n_samples=sample_count,
                        spectral_error=P.norm2(empirical-target),fro_relative_error=float(np.linalg.norm(empirical-target)/max(np.linalg.norm(target),1e-300)),
                        interpretation='finite-sample empirical error, not finite-Lanczos covariance operator error'))
                for direction,v in probes.items():
                    x=samples@v;mu=float(x.mean());var=float(x.var(ddof=1));center=x-mu
                    idealx=ideal_samples@v
                    idealmean=float(v@(RA@p if phase=='conditional' else np.tile([.2,-.1,.3],N)))
                    paired_second=(x-idealmean)**2-(idealx-idealmean)**2
                    paired_mean=x-idealx
                    # Squared centered draws estimate sample-variance uncertainty; no Gaussian assumption.
                    squared=center**2;se=float(squared.std(ddof=1)/np.sqrt(sample_count))
                    tcrit=float(stats.t.ppf(.975,sample_count-1));fourth=center**4
                    target=float(v@(C if phase=='conditional' else dense.Pi)@v)
                    ideal=float(v@(CA if phase=='conditional' else dense.Pi)@v)
                    rows.append(dict(config=meta['config'],N=N,family=meta['family'],seed=meta['seed'],method=method,
                        direction=direction,phase=phase,alpha=1.,dt=dt,r_d=rd,r_n=rn,substeps=k,n_samples=sample_count,
                        mean=mu,mean_se=float(x.std(ddof=1)/np.sqrt(sample_count)),
                        exact_mean=float(v@(R@p if phase=='conditional' else np.tile([.2,-.1,.3],N))),
                        ideal_method_mean=float(v@(RA@p if phase=='conditional' else np.tile([.2,-.1,.3],N))),
                        variance=var,variance_se=se,ci_low=max(0.,var-tcrit*se),ci_high=var+tcrit*se,
                        exact_variance=target,ideal_method_variance=ideal,
                        fourth_moment=float(fourth.mean()),fourth_se=float(fourth.std(ddof=1)/np.sqrt(sample_count)),
                        exact_gaussian_fourth=3*target**2,com_max=max_com,
                        paired_mean_error=float(paired_mean.mean()),paired_mean_se=float(paired_mean.std(ddof=1)/np.sqrt(sample_count)),
                        paired_second_error=float(paired_second.mean()),paired_second_se=float(paired_second.std(ddof=1)/np.sqrt(sample_count)),
                        coupled_path_error_max=max(path_errors),
                        paired_reference='same noise ideal dense full OU' if method=='B' else 'same independent edge noises and exact dense long OU; ideal splitting',
                        sample_seed_recipe='2026102707+100000+case_index*100000+equilibrium*10000+sample',
                        CI_role='pointwise approximate 95% sample uncertainty, no multiple-comparison correction'))
                P.write('mc.csv',rows);P.write('mc_fullcov.csv',fullcov)
    P.save(P.RESULTS/'mc_done.json',dict(elapsed_s=time.perf_counter()-start_all,rows=len(rows)))

def timing(args):
    if (P.RESULTS/'timing_done.json').exists():raise FileExistsError('Timing evidence already exists')
    rows=[];setup_rows=[];loads=[P.load_snapshot('before')];start_all=time.perf_counter()
    P.save(P.RESULTS/'environment_timing.json',P.environment())
    # Every primary 16/64/256 nine-configuration case; 2/8/128 seed2 representatives.
    structures=[r for r in P.read('structure.csv') if r['epsilon']==1e-4]
    rng=np.random.default_rng(P.SEED+20000)
    for meta in structures:
        system=make(meta);edges=edge_object(system);dense=P.DenseOU(system.full_dense)
        saved=np.load(P.INPUTS/(meta['config']+'.npz'));p=saved['p'];dt=1./dense.lmax
        sels=[r for r in P.read('selection.csv') if r['config']==meta['config'] and r['alpha']==1. and r['available']]
        selected_A=next((r for r in sels if r['method']=='A'),None)
        factor_substeps=int(selected_A['substeps']) if selected_A else 1
        if not np.array_equal(system.q,saved['q']) or not np.array_equal(system.full_dense,saved['G']):
            raise RuntimeError('Formal timing configuration/matrix differs from accuracy evidence')
        for setup_rep in range(3):
            fresh=make(meta);fresh_edges=edge_object(fresh)
            t=time.perf_counter();fresh_edges._coefficients(dt/(2*factor_substeps),1.,1.)
            factor_setup=time.perf_counter()-t
            setup_rows.append(dict(config=meta['config'],N=meta['N'],family=meta['family'],seed=meta['seed'],rep=setup_rep,
                t_table_setup=fresh.metadata['t_table_setup'],t_config_setup=fresh.metadata['t_config_setup'],
                t_near_edge_build=fresh.metadata.get('t_near_edge_build'),t_near_edge_eigh=fresh_edges.t_setup,
                t_edge_dt_factors=factor_setup,batch='thermostat_local_interleaved_20261027',
                independent_dense_assembly_excluded=True,edge_factor_tau=dt/(2*factor_substeps),
                setup_scope='fresh production table/config/edge phases; common radial kernel cache/process start and validation excluded',
                A_fresh_setup=fresh.metadata['t_table_setup']+fresh.metadata['t_config_setup']+fresh.metadata.get('t_near_edge_build',0.)+fresh_edges.t_setup+factor_setup,
                B_fresh_setup=fresh.metadata['t_table_setup']+fresh.metadata['t_config_setup']))
        P.write('setup_timing_raw.csv',setup_rows)
        if len(sels)!=2:continue
        P.log(f"timing {meta['config']}")
        for sel in sels:
            sample_step(system,edges,p,dt,sel['method'],int(sel['r_d']),int(sel['r_n']),int(sel['substeps']),rng)
        for rnd in range(args.rounds):
            for rep in range(args.reps):
                for order in rng.permutation(2):
                    sel=sels[int(order)]
                    t=time.perf_counter()
                    out,diag=sample_step(system,edges,p,dt,sel['method'],int(sel['r_d']),int(sel['r_n']),int(sel['substeps']),rng)
                    elapsed=time.perf_counter()-t
                    numeric={key:val for key,val in diag.items() if isinstance(val,(str,float,int,np.number)) and key not in ('substeps','method','dt','r_d','r_n')}
                    actual_calls=int(diag.get('actual_full_actions',0))+int(diag.get('actual_long_actions',0))
                    if actual_calls!=diag['matvec_count']:raise RuntimeError('Measured friction call count disagrees with Lanczos')
                    numeric.update(FFT=6*actual_calls,
                                   near_actions=int(diag.get('actual_full_actions',0)),
                                   near_image_actions=int(diag.get('actual_full_actions',0))*system.op.n_images,
                                   FFT_definition='three forward rFFT and three inverse rFFT per friction action')
                    rows.append(dict(config=meta['config'],N=meta['N'],family=meta['family'],seed=meta['seed'],
                        alpha=1.,dt=dt,method=sel['method'],round=rnd,rep=rep,r_d=sel['r_d'],r_n=sel['r_n'],
                        substeps=sel['substeps'],t_total_frozen_step=elapsed,batch='thermostat_local_interleaved_20261027',
                        gaussian_generation_included=True,threads_requested=1,**numeric))
        loads.append(P.load_snapshot(meta['config']));P.write('timing_raw.csv',rows);P.save(P.RESULTS/'system_load.json',loads)
    P.save(P.RESULTS/'timing_done.json',dict(elapsed_s=time.perf_counter()-start_all,rows=len(rows),rounds=args.rounds,reps=args.reps))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['accuracy','mc','timing']);p.add_argument('--mc-small',type=int,default=512)
    p.add_argument('--mc-medium',type=int,default=128);p.add_argument('--rounds',type=int,default=3);p.add_argument('--reps',type=int,default=5)
    a=p.parse_args();globals()[a.stage](a)
