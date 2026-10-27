"""Post-MC rank refinement for the one observed N=2, B, alpha=1 failure.

Run once after the initial MC stage, before formal timing:
    python3 -B refine_small_case.py

Initial evidence is archived with SHA256 hashes before any final evidence is
replaced.  The production MC function is reused in an isolated staging folder
with its original case_index=0 seeds.  Only N2/B/alpha1 is refined to r_n=3.
"""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import json
import shutil
import protocol as P
import run_benchmarks as benchmark
from ou_core import lanczos_action
import stopping_audit
np = P.np


CONFIG = 'N2_uniform_s2_eps0.0001'


def selected(row):
    return row['config']==CONFIG and row['method']=='B' and row.get('alpha',1.)==1.


def main():
    start = perf_counter()
    if not (P.RESULTS/'mc_done.json').exists():
        raise RuntimeError('Initial MC must finish before a rank refinement')
    pilot = P.HERE/'pilot_rank_refinement'
    if pilot.exists():
        raise FileExistsError('Refinement archive exists; inspect it instead of overwriting evidence')
    selection = P.read('selection.csv')
    initial = next(row for row in selection if selected(row))
    if int(initial['r_n']) != 2 or int(initial['r_d']) != 3:
        raise RuntimeError('Expected the actual initial N2/B rank pair (3,2)')
    phases = ('conditional','equilibrium')
    initial_path_errors = {}
    for phase in phases:
        with np.load(P.INPUTS/f'mc_{CONFIG}_B_{phase}.npz') as source:
            initial_path_errors[phase] = float(np.max(source['path_errors']))
    trigger = max(initial_path_errors.values())
    if trigger <= P.KRYLOV_TOL:
        raise RuntimeError('No actual new-Gaussian path failure triggers this refinement')

    # Preserve initial and derived evidence before any final-file mutation.
    pilot.mkdir()
    archive_files = [P.RESULTS/name for name in (
        'selection.csv','rank_errors.csv','matrixfree_checks.csv','mc.csv','mc_fullcov.csv','mc_done.json',
        'stopping_summary.json','stopping_diagnostics.csv','stopping_diagnostics_all_probes.csv','coupling_checks.csv')]
    archive_files += [P.INPUTS/f'mc_{CONFIG}_B_{phase}.npz' for phase in phases]
    archive_hashes = {}
    for source in archive_files:
        relative = source.relative_to(P.HERE)
        target = pilot/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        archive_hashes[str(relative)] = P.sha(source)
        if P.sha(target) != archive_hashes[str(relative)]:
            raise RuntimeError(f'Archive hash mismatch: {relative}')
    P.save(pilot/'initial_evidence_manifest.json',dict(
        role='unmodified initial five-probe selected-rank and MC evidence',sha256=archive_hashes,
        trigger='N2/B new Gaussian coupled ideal-OU path exceeds 1e-5 despite five registered probes passing',
        trigger_path_errors=initial_path_errors,old_r_d=3,old_r_n=2,new_r_d=3,new_r_n=3))

    with np.load(P.INPUTS/(CONFIG+'.npz')) as saved:
        G,p,z = saved['G'],saved['p'],saved['z']
        probes = {name:saved[name].copy() for name in ('slow','fast','random')}
        probes.update(input_p=P.project(p)/np.linalg.norm(P.project(p)),
                      input_z=P.project(z)/np.linalg.norm(P.project(z)))
    meta = next(row for row in benchmark.chosen_cases() if row['config']==CONFIG)
    system = benchmark.make(meta)
    np.testing.assert_array_equal(system.full_dense,G)
    dense = P.DenseOU(G)
    dt = 1./dense.lmax
    _,F,_ = dense.matrices(dt)
    rankrows,mfrows = P.read('rank_errors.csv'),P.read('matrixfree_checks.csv')
    new_errors = []
    base = {key:meta[key] for key in ('config','N','family','seed','epsilon')}
    for name,v in probes.items():
        old,_ = lanczos_action(lambda x:G@x,v,2,dt,'noise')
        new,diag = lanczos_action(lambda x:G@x,v,3,dt,'noise')
        mf,_ = lanczos_action(system.full_action,v,3,dt,'noise')
        truth = F@v
        error = float(np.linalg.norm(new-truth)/np.linalg.norm(truth))
        new_errors.append(error)
        rankrows.append(dict(base,operator='full',alpha=1.,dt=dt,kind='noise',probe=name,rank=3,
                             error=error,step_change=float(np.linalg.norm(new-old)/np.linalg.norm(new)),
                             tolerance=P.KRYLOV_TOL,refinement_role='post-MC extra Gaussian failure; full projected dimension',
                             **{'diag_'+key:value for key,value in diag.items()
                                if isinstance(value,(str,int,float,bool,np.number))}))
        mfrows.append(dict(base,alpha=1.,method='B',kind='noise',rank=3,probe=name,
                           matrixfree_vs_dense_lanczos=float(np.linalg.norm(mf-new)/np.linalg.norm(new)),
                           exact_action_error=float(np.linalg.norm(mf-truth)/np.linalg.norm(truth)),
                           verification_role='post-MC refined rank; initial rank2 rows retained'))
    refined = dict(initial,r_n=3,krylov_noise_error=max(new_errors),
                   rank_domain='five registered directions plus post-MC full projected dimension refinement',
                   rank_refinement='N2/B/alpha1 extra Gaussian path failure; see rank_refinement.csv')

    # Reuse the exact production MC statistics and Gaussian recipe in isolation.
    original_results,original_inputs,original_cases = P.RESULTS,P.INPUTS,benchmark.chosen_cases
    stage = pilot/'refined_run'
    stage_results,stage_inputs = stage/'results',stage/'inputs'
    stage_results.mkdir(parents=True)
    stage_inputs.mkdir()
    shutil.copy2(original_inputs/(CONFIG+'.npz'),stage_inputs/(CONFIG+'.npz'))
    try:
        P.RESULTS,P.INPUTS = stage_results,stage_inputs
        P.write('selection.csv',[refined])
        benchmark.chosen_cases = lambda:[meta]
        benchmark.mc(SimpleNamespace(mc_small=512,mc_medium=128))
        new_mc,new_cov = P.read('mc.csv'),P.read('mc_fullcov.csv')
        stage_done = json.loads((stage_results/'mc_done.json').read_text())
    finally:
        P.RESULTS,P.INPUTS,benchmark.chosen_cases = original_results,original_inputs,original_cases
    if len(new_mc)!=8 or len(new_cov)!=2:
        raise RuntimeError('Unexpected N2/B-only staged MC coverage')
    refined_path_errors = {phase:max(row['coupled_path_error_max'] for row in new_mc if row['phase']==phase)
                           for phase in phases}
    if max(refined_path_errors.values()) > 1e-12:
        raise RuntimeError('Full projected-rank refinement did not reach roundoff error')

    # Only now publish the concrete, completely verified replacement evidence.
    for index,row in enumerate(selection):
        if selected(row):
            selection[index] = refined
    P.write('selection.csv',selection)
    P.write('rank_errors.csv',rankrows)
    P.write('matrixfree_checks.csv',mfrows)
    old_mc,old_cov = P.read('mc.csv'),P.read('mc_fullcov.csv')
    final_mc = [row for row in old_mc if not selected(row)] + new_mc
    final_cov = [row for row in old_cov if not selected(row)] + new_cov
    if len(final_mc)!=len(old_mc) or len(final_cov)!=len(old_cov):
        raise RuntimeError('Refinement unexpectedly changed total MC coverage')
    P.write('mc.csv',final_mc)
    P.write('mc_fullcov.csv',final_cov)
    for phase in phases:
        filename = f'mc_{CONFIG}_B_{phase}.npz'
        shutil.copy2(stage_inputs/filename,P.INPUTS/filename)
    initial_done = json.loads((pilot/'results'/'mc_done.json').read_text())
    P.save(P.RESULTS/'mc_done.json',dict(
        elapsed_s=initial_done['elapsed_s']+stage_done['elapsed_s'],rows=len(final_mc),
        initial_batch_elapsed_s=initial_done['elapsed_s'],refinement_MC_elapsed_s=stage_done['elapsed_s'],
        refinement='one N2/B alpha1 rank2->3 noise rerun, same 512 per-phase Gaussian seeds; initial evidence archived',
        refinement_archive='pilot_rank_refinement/initial_evidence_manifest.json',
        full_batch_rerun=False))
    P.write('rank_refinement.csv',[dict(config=CONFIG,N=2,method='B',alpha=1.,dt=dt,
             old_r_d=3,old_r_n=2,new_r_d=3,new_r_n=3,trigger_threshold=P.KRYLOV_TOL,
             old_registered_noise_error=initial['krylov_noise_error'],new_registered_noise_error=max(new_errors),
             old_conditional_path_max=initial_path_errors['conditional'],old_equilibrium_path_max=initial_path_errors['equilibrium'],
             new_conditional_path_max=refined_path_errors['conditional'],new_equilibrium_path_max=refined_path_errors['equilibrium'],
             reason='out-of-register Gaussian path failed 1e-5; use exact projected dimension for this N2 case',
             same_G_same_q_same_Gaussian_seeds=True,MC_samples_per_phase=512,
             refinement_MC_elapsed_s=stage_done['elapsed_s'],initial_archive='pilot_rank_refinement/')])
    stopping_audit.main()
    final_hashes = {name:P.sha(P.RESULTS/name) for name in (
        'selection.csv','rank_errors.csv','matrixfree_checks.csv','mc.csv','mc_fullcov.csv','mc_done.json',
        'stopping_summary.json','coupling_checks.csv','rank_refinement.csv')}
    summary = dict(config=CONFIG,only_method='B',only_alpha=1.,old_noise_rank=2,new_noise_rank=3,
                   old_r_n=2,new_r_n=3,trigger=trigger,
                   pilot_path='pilot_rank_refinement/',
                   SHA_manifest='pilot_rank_refinement/initial_evidence_manifest.json',
                   reproduce='python3 -B refine_small_case.py',
                   initial_path_errors=initial_path_errors,refined_path_errors=refined_path_errors,
                   final_all_MC_path_error_max=max(row['coupled_path_error_max'] for row in final_mc),
                   new_registered_noise_error_max=max(new_errors),elapsed_s=perf_counter()-start,
                   initial_MC_elapsed_s=initial_done['elapsed_s'],refinement_MC_elapsed_s=stage_done['elapsed_s'],
                   preserved_initial_evidence=archive_hashes,final_result_sha256=final_hashes,
                   interpretation='post-MC rank refinement is disclosed; initial five-probe guarantee failed to cover all Gaussian draws',
                   inference_limit='a finite sample audit of remaining cases is empirical, not a universal error guarantee')
    P.save(P.RESULTS/'rank_refinement_summary.json',summary)
    P.save(P.RESULTS/'rank_refinement.json',summary)
    print(summary,flush=True)


if __name__=='__main__':
    main()
