"""Seal new evidence without modifying or rerunning the PR 8 baseline."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import ast
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import stats
from report_generator import HERE, PARENT, RESULTS, METHODS, read, js


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def array_sha(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def main():
    checks = []
    def check(name, passed, detail=None):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))
    repo = Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=HERE,text=True).strip())
    snapshot = json.loads((HERE/'source_snapshot.json').read_text())
    hashes = snapshot['tracked_file_sha256']
    changed = [name for name, expected in hashes.items() if not (repo/name).is_file() or sha(repo/name) != expected]
    check('all_PR8_tracked_baseline_hashes_preserved', not changed, dict(files=len(hashes), differences=changed))
    prefix = str(HERE.relative_to(repo))+'/'
    diff = subprocess.check_output(['git','diff','--name-only',snapshot['base_sha']],cwd=repo,text=True).splitlines()
    check('tracked_changes_only_in_new_directory', all(path.startswith(prefix) for path in diff), diff)
    errors = []
    for path in HERE.glob('*.py'):
        try: ast.parse(path.read_text(),filename=str(path))
        except SyntaxError as exc: errors.append(str(exc))
    check('new_Python_sources_parse', not errors, errors)
    for name in ('accuracy_done.json','mc_done.json','timing_done.json','compilation.json','profile_baseline.json'):
        check('completed_'+name, (RESULTS/name).is_file())
    for name, minimum in [('baseline_tests.log',26),('compiled_tests.log',16),('test_validation.log',3)]:
        path = HERE/'logs'/name
        text = path.read_text() if path.is_file() else ''
        matches = re.findall(r'Ran (\d+) tests? in ([\d.]+)s', text)
        check(name+'_passed', bool(matches) and int(matches[-1][0]) >= minimum and text.rstrip().endswith('OK'),
              dict(count=int(matches[-1][0]) if matches else None, minimum=minimum))
    compilation = js('compilation.json', {})
    cold = compilation.get('cold', {})
    check('actual_compiler_source_and_strict_float_flags',
          cold.get('source_sha256') == sha(HERE/'edge_sweep.c') and cold.get('compiled_now') and
          cold.get('fast_math') is False and cold.get('fp_contract') is False and cold.get('edge_parallelism') is False and
          '-fno-fast-math' in cold.get('flags', []) and '-ffp-contract=off' in cold.get('flags', []),
          dict(compiler=cold.get('compiler'), flags=cold.get('flags'), cold_compile_s=cold.get('t_compile_s')))
    library = Path(cold.get('library_path') or '')
    check('built_library_matches_recorded_hash', library.is_file() and sha(library) == cold.get('library_sha256'))
    cached = compilation.get('fresh_process_cache_load', {})
    check('cold_and_fresh_process_cache_load_separate', cold.get('t_compile_s',0)>0 and cached.get('compiled_now') is False and
          cached.get('t_compile_s') == 0 and compilation.get('fresh_process_total_s',0)>0)
    startup=js('cold_process_start.json',{});startup_backend=startup.get('backend',{})
    cold_library=Path(startup_backend.get('library_path') or '')
    check('genuine_fresh_process_empty_cache_startup_recorded', startup.get('fresh_process_wall_s',0)>0 and
          startup_backend.get('compiled_now') is True and startup_backend.get('t_compile_s',0)>0 and
          startup_backend.get('source_sha256')==sha(HERE/'edge_sweep.c') and cold_library.is_file() and
          sha(cold_library)==startup_backend.get('library_sha256'))
    cache=js('cache_validation.json',{});cache_rows=read('cache_validation.csv')
    fixture=Path(cache.get('raw_fixture') or '')
    check('physical_graph_parameter_cache_invalidation', cache.get('passed') is True and len(cache_rows)==62 and
          cache.get('parameter_combinations')==30 and cache.get('initial_rebuilt_cache_size')==0 and
          cache.get('original_graph_sha256')!=cache.get('changed_graph_sha256') and cache.get('original_graph_and_cache_unchanged') and
          cache.get('first_miss_then_repeat_hit') and fixture.is_file() and sha(fixture)==cache.get('raw_fixture_sha256') and
          all(r['path_relative_error']<=1e-11 and r['COM_error']<=1e-9 and r['first_prepare_miss'] and r['repeated_prepare_hit'] and
              (r.get('coefficient_definition_max_abs_error') or 0.)<=1e-12 for r in cache_rows),cache)
    profile = js('profile_baseline.json', {})
    check('fresh_profiler_uses_unchanged_Python_reference',
          profile.get('source_sha256') == sha(PARENT/'ou_core.py') and
          profile.get('selection_source_sha256') == sha(PARENT/'results/selection.csv') and
          profile.get('unprofiled_samples',0)>0 and (HERE/'logs/baseline.prof').is_file())
    structures = read('structure.csv')
    baseline_structures = read('structure.csv', PARENT/'results')
    check('all_baseline_34_configurations_recomputed', {r['config'] for r in structures} == {r['config'] for r in baseline_structures},
          dict(expected=len(baseline_structures),actual=len(structures)))
    issues = []
    for row in structures:
        name = row['config']+'.npz'
        current, base = HERE/'inputs'/name, PARENT/'inputs'/name
        if not current.is_file() or not base.is_file():
            issues.append(dict(config=row['config'], error='missing saved input')); continue
        with np.load(current,allow_pickle=False) as a, np.load(base,allow_pickle=False) as b:
            for key in ('q','G','GS','GL','L','p','z','slow','fast','random'):
                if not np.array_equal(a[key],b[key]): issues.append(dict(config=row['config'],field=key,error='PR8 array changed'))
                if key+'_sha256' in row and row[key+'_sha256'] != array_sha(a[key]): issues.append(dict(config=row['config'],field=key,error='array hash mismatch'))
            N = int(row['N']); assembled = np.zeros((3*N,3*N))
            for i,j,K in zip(a['edge_i'],a['edge_j'],a['edge_K']):
                i,j = int(i),int(j)
                assembled[3*i:3*i+3,3*i:3*i+3] += K; assembled[3*j:3*j+3,3*j:3*j+3] += K
                assembled[3*i:3*i+3,3*j:3*j+3] -= K; assembled[3*j:3*j+3,3*i:3*i+3] -= K
            if not np.allclose(assembled,a['GS'],rtol=1e-13,atol=1e-13): issues.append(dict(config=row['config'],error='edges do not assemble saved near graph'))
            for field,key in [('edge_i_sha256','edge_i'),('edge_j_sha256','edge_j'),('edge_K_sha256','edge_K')]:
                if row[field] != array_sha(a[key]): issues.append(dict(config=row['config'],field=field,error='edge hash mismatch'))
        if row['baseline_input_sha256'] != sha(base): issues.append(dict(config=row['config'],error='baseline input SHA mismatch'))
    check('bitwise_PR8_graph_inputs_and_identical_ordered_near_edges', not issues, issues)
    selections = {(r['config'],r['alpha'],r['method']):r for r in read('selection.csv',PARENT/'results')}
    accuracy = read('accuracy.csv'); issues=[]
    for row in accuracy:
        method = 'B' if row['method'] == 'B' else 'A'
        sel = selections.get((row['config'],row['alpha'],method))
        if not sel or any(row[key] != sel[key] for key in ('r_d','r_n','substeps','dt')): issues.append(dict(config=row['config'],method=row['method'],error='accuracy budget/rank changed'))
        if row['COM_error'] > 1e-9 or row['dense_fdt_residual'] > 1e-10: issues.append(dict(config=row['config'],error='COM/dense FDT residual'))
        if max(row['forward_original_vs_compiled'],row['reverse_original_vs_compiled'],row.get('original_vs_compiled_path') or 0.) > 1e-11:
            issues.append(dict(config=row['config'],error='compiled path changed'))
        draws = HERE/'inputs'/f'draws_{row["config"]}_alpha{row["alpha"]:g}.npz'
        if not draws.is_file() or sha(draws) != row['draws_file_sha256']: issues.append(dict(config=row['config'],error='Gaussian file SHA mismatch'))
        elif row['method'] == 'A-compiled':
            h=hashlib.sha256()
            with np.load(draws,allow_pickle=False) as saved:
                for key in ('z','ef','er'): h.update(key.encode());h.update(np.ascontiguousarray(saved[key]).tobytes())
            if h.hexdigest() != row['gaussian_sha256']: issues.append(dict(config=row['config'],error='Gaussian arrays SHA mismatch'))
    check('same_registered_dt_ranks_substeps_Gaussians_and_ordered_paths', not issues, issues)
    grouped = {}
    for row in accuracy: grouped.setdefault((row['config'],row['alpha']),[]).append(row)
    check('three_accuracy_methods_every_case_alpha', len(grouped)==4*len(structures) and all(sorted(r['method'] for r in rr)==sorted(METHODS) for rr in grouped.values()))
    checks_rows = read('selected_rank_checks.csv')
    check('same_selected_matrix_function_accuracy', bool(checks_rows) and all(r['error']<=1e-5+1e-10 and r['ritz_min_raw']>=-1e-10 for r in checks_rows))
    mc=read('mc.csv'); pairwise=read('mc_pairwise.csv')
    check('fresh_compiled_vs_Python_MC_paths', bool(pairwise) and all(r['compiled_original_relative_max']<=1e-11 for r in pairwise),
          dict(case_phases=len(pairwise),maximum=max((r['compiled_original_relative_max'] for r in pairwise),default=None)))
    mc_issues=[]
    for item in js('mc_inputs_manifest.json',[]):
        path=Path(item['file'])
        if not path.is_file() or sha(path)!=item['sha256']:
            mc_issues.append(dict(config=item['config'],phase=item['phase'],error='MC raw file SHA mismatch'));continue
        with np.load(path,allow_pickle=False) as saved, np.load(HERE/'inputs'/(item['config']+'.npz'),allow_pickle=False) as base:
            for row in [r for r in mc if r['config']==item['config'] and r['phase']==item['phase']]:
                field={'A-python':'A_python','A-compiled':'A_compiled','B':'B'}[row['method']]
                vector=np.tile([1.,0.,0.],int(row['N']))/np.sqrt(row['N']) if row['direction']=='COM' else base[row['direction']]
                x=saved[field]@vector
                n=len(x);center=x-x.mean();critical=float(stats.t.ppf(.975,n-1))
                variance=float(x.var(ddof=1));mean_se=float(x.std(ddof=1)/np.sqrt(n))
                variance_se=float((center**2).std(ddof=1)/np.sqrt(n));fourth=center**4
                ideal=saved['ideal_B' if row['method']=='B' else 'ideal_A']@vector
                paired_mean=x-ideal;paired_second=(x-row['ideal_mean'])**2-(ideal-row['ideal_mean'])**2
                values=dict(mean=float(x.mean()),variance=variance,fourth_moment=float(fourth.mean()),mean_se=mean_se,
                    variance_se=variance_se,fourth_se=float(fourth.std(ddof=1)/np.sqrt(n)),
                    variance_ci_low=max(0.,variance-critical*variance_se),variance_ci_high=variance+critical*variance_se,
                    paired_mean_error=float(paired_mean.mean()),paired_mean_se=float(paired_mean.std(ddof=1)/np.sqrt(n)),
                    paired_second_error=float(paired_second.mean()),paired_second_se=float(paired_second.std(ddof=1)/np.sqrt(n)))
                for key,value in values.items():
                    if abs(value-row[key])>1e-11*max(1.,abs(value)): mc_issues.append(dict(config=row['config'],field=key,error='raw sample statistic mismatch'))
                if row['COM_max']>1e-9 or row['coupled_ideal_path_max']>1e-5: mc_issues.append(dict(config=row['config'],error='observed MC path/COM guard'))
    check('fresh_MC_raw_samples_reproduce_directional_statistics', bool(mc) and not mc_issues, mc_issues)
    done=js('mc_done.json',{})
    check('MC_recomputed_all_three_methods', done.get('all_outputs_newly_computed') is True and done.get('baseline_MC_results_used') is False and done.get('rows')==len(mc) and len(mc)==192)
    timing=read('timing_raw.csv'); issues=[]
    triples={}
    structure_by_config={r['config']:r for r in structures}
    for row in timing:
        sel=selections.get((row['config'],row['alpha'],'B' if row['method']=='B' else 'A'))
        if not sel or any(row[k]!=sel[k] for k in ('r_d','r_n','substeps','dt')): issues.append(dict(config=row['config'],error='timed rank/dt changed'))
        if row.get('t_total_step',0)<=0: issues.append(dict(config=row['config'],error='invalid total wall time'))
        full,long=row['actual_full_actions'],row['actual_long_actions']
        if full+long!=row['matvec_count'] or row['FFT']!=6*(full+long) or row['near_actions']!=full or row['near_image_actions']!=full*structure_by_config[row['config']]['n_images']:
            issues.append(dict(config=row['config'],error='actual friction/FFT/near counts disagree'))
        if not row['gaussian_generation_included'] or not row['compiler_warmed'] or row['compiler_source_sha256']!=sha(HERE/'edge_sweep.c'):
            issues.append(dict(config=row['config'],error='incorrect timed scope/compiler source'))
        if row['method']=='A-compiled' and (row['active_backend']!='compiled' or row['paired_A_path_error']>1e-11):
            issues.append(dict(config=row['config'],error='optimized backend/path invalid'))
        if row['p_sha256']!=structure_by_config[row['config']]['p_sha256'] or row['baseline_input_sha256']!=structure_by_config[row['config']]['baseline_input_sha256']:
            issues.append(dict(config=row['config'],error='timing changed saved physical input'))
        triples.setdefault((row['config'],row['round'],row['rep']),[]).append(row)
    for key,rr in triples.items():
        if sorted(r['method'] for r in rr)!=sorted(METHODS): issues.append(dict(group=key,error='unpaired three-way batch'))
        else:
            methods={r['method']:r for r in rr}
            if len({r['graph_sha256'] for r in rr})!=1 or len({r['first_z_sha256'] for r in rr})!=1 or methods['A-python']['gaussian_sha256']!=methods['A-compiled']['gaussian_sha256']:
                issues.append(dict(group=key,error='paired graph/Gaussian mismatch'))
            if {r['order_position'] for r in rr}!={0,1,2}: issues.append(dict(group=key,error='invalid three-way order positions'))
            for label in ('A-compiled','B'):
                row=methods[label];rng=np.random.default_rng(int(row['gaussian_seed']))
                E=int(row['edge_count']) if label=='A-compiled' else 0
                z=rng.normal(size=(int(row['substeps']),3*int(row['N'])))
                ef=rng.normal(size=(int(row['substeps']),E,3));er=rng.normal(size=ef.shape)
                h=hashlib.sha256()
                for field,array in (('z',z),('ef',ef),('er',er)):h.update(field.encode());h.update(np.ascontiguousarray(array).tobytes())
                if h.hexdigest()!=row['gaussian_sha256'] or array_sha(z[0])!=row['first_z_sha256']:
                    issues.append(dict(group=key,error='timed Gaussian recipe replay mismatch'))
    check('new_batch_three_way_timing_same_accuracy', bool(timing) and not issues, issues)
    protocol=js('timing_protocol.json',{});done=js('timing_done.json',{})
    check('complete_new_batch_without_historical_timing_mix', len(timing)==len(structures)*protocol.get('rounds',0)*protocol.get('reps',0)*3 and
          done.get('rows')==len(timing) and len({r['batch'] for r in timing})==1 and protocol.get('no_historical_timing_mix') is True and
          protocol.get('selection_source_sha256')==sha(PARENT/'results/selection.csv'))
    setup=read('setup_raw.csv');setup_issues=[]
    for row in setup:
        base=row['t_table_setup']+row['t_config_setup']
        expected=dict(B_setup=base,A_python_setup=base+row['t_edge_build']+row['t_python_edge_constructor']+row['t_python_coeff_setup'],
                      A_compiled_setup=base+row['t_edge_build']+row['t_compiled_edge_constructor']+row['t_coeff_setup'])
        if any(abs(row[k]-v)>1e-12 for k,v in expected.items()) or row['coefficient_cache_hit'] or row['active_backend']!='compiled':
            setup_issues.append(dict(config=row['config'],rep=row['rep'],error='fresh setup phases/cache/backend mismatch'))
    check('separate_fresh_setup_no_double_counted_eigh', not setup_issues and len(setup)==len(structures)*protocol.get('setup_reps',0),setup_issues)
    amortization_issues=[]
    for row in read('setup_step_costs.csv'):
        sr=[r for r in setup if r['config']==row['config']]
        a_setup=float(np.median([r['A_compiled_setup'] for r in sr]));b_setup=float(np.median([r['B_setup'] for r in sr]))
        a_step=float(np.median([r['t_total_step'] for r in timing if r['config']==row['config'] and r['method']=='A-compiled']))
        b_step=float(np.median([r['t_total_step'] for r in timing if r['config']==row['config'] and r['method']=='B']))
        expected=dict(A_compiled_setup=a_setup,B_setup=b_setup,A_compiled_step=a_step,B_step=b_step,
            A_setup_plus_one_step=a_setup+a_step,B_setup_plus_one_step=b_setup+b_step,
            A_over_B_setup_plus_one_step=(a_setup+a_step)/(b_setup+b_step))
        crossing=max(1,int(np.floor((a_setup-b_setup)/(b_step-a_step)))+1) if b_step>a_step else None
        if any(abs(row[k]-v)>1e-12 for k,v in expected.items()) or row['minimum_reused_steps_to_beat_B']!=crossing:
            amortization_issues.append(dict(config=row['config'],error='setup+K-step accounting mismatch'))
    check('setup_plus_step_accounting_recomputed', not amortization_issues and len(read('setup_step_costs.csv'))==len(structures),amortization_issues)
    analysis=js('analysis_protocol.json',{})
    check('paired_CI_uses_declared_config_round_blocks', analysis.get('source_sha256')==sha(RESULTS/'timing_raw.csv') and
          analysis.get('bootstrap_draws')==4000 and 'round' in analysis.get('resampling','') and 'configurations' in analysis.get('resampling',''))
    ratio_issues=[]
    definitions={'near_python_over_compiled':('A-python','A-compiled','near'),
                 'total_python_over_compiled':('A-python','A-compiled','total'),
                 'total_compiled_over_B':('A-compiled','B','total'),'total_python_over_B':('A-python','B','total')}
    for row in read('paired_round_ratios.csv'):
        numerator,denominator,metric=definitions[row['comparison']];values=[]
        for (config,rnd,rep),rr in triples.items():
            if config!=row['config'] or rnd!=row['round']:continue
            lookup={r['method']:r for r in rr}
            cost=lambda r:r['t_total_step'] if metric=='total' else r['t_near_forward']+r['t_near_reverse']
            values.append(cost(lookup[numerator])/cost(lookup[denominator]))
        if not values or abs(float(np.median(values))-row['ratio'])>1e-12:ratio_issues.append(dict(config=row['config'],round=row['round'],error='paired round ratio mismatch'))
    for row in read('paired_aggregate_ratios.csv'):
        values=[r['ratio'] for r in read('paired_config_ratios.csv') if r['N']==row['N'] and r['epsilon']==row['epsilon'] and r['comparison']==row['comparison']]
        if not values or abs(float(np.median(values))-row['ratio'])>1e-12 or not row['ci_low']<=row['ratio']<=row['ci_high']:
            ratio_issues.append(dict(N=row['N'],epsilon=row['epsilon'],error='aggregate point/CI mismatch'))
    check('paired_ratio_points_recomputed_from_raw_triplets', not ratio_issues,ratio_issues)
    for stem in ('near_sweep_speedup','three_way_total_time','three_way_phases'):
        check('PNG_SVG_'+stem, all((HERE/'figures'/(stem+'.'+ext)).is_file() for ext in ('png','svg')))
    manifest={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and
              '.build' not in p.parts and 'build' not in p.parts and '__pycache__' not in p.parts and p.name!='audit.json' and
              (p.suffix in ('.py','.c','.csv','.json','.log','.md','.npz','.png','.svg','.prof') or p.name in ('.gitattributes','.gitignore'))}
    failed=[r['name'] for r in checks if not r['passed']]
    result=dict(status='FAIL' if failed else 'PASS',timestamp_utc=datetime.now(timezone.utc).isoformat(),
                checks=checks,failed_checks=failed,baseline_sha=snapshot['base_sha'],baseline_files=len(hashes),evidence_sha256=manifest,
                scope='compiled ordered near sweeps only; finite MC and paired timing inference remain empirical')
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],checks=len(checks),failed=failed,baseline_files=len(hashes)),ensure_ascii=False))
    return bool(failed)


if __name__=='__main__': sys.exit(main())
