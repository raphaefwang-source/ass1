"""Read-only provenance and theory audit for the final four-step Exp. 2 run.

Only final_exp2_selection is written. Every prior tracked file, old result file,
and independent reference/sidecar is frozen before new work. Historical timings
are catalogued as historical hints, never substituted for the new common pass.
"""
from collections import Counter,defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent
VALIDATION=PREVIOUS.parent
REPO=HERE.parents[4]
NAMES=(
 ('deconv_psd_comparison/results/search.csv','search',''),
 ('deconv_psd_comparison/results/validation_summary.csv','validation_summary',''),
 ('deconv_psd_comparison/results/validation.csv','validation',''),
 ('deconv_psd_comparison/results/final_timing.csv','timing',''),
 ('deconv_psd_comparison/results/primary_timing/final_timing.csv','timing',''),
 ('positive_mesh_b2/results/search.csv','search','B2-positive'),
 ('positive_mesh_b2/results/validation_summary.csv','validation_summary','B2-positive'),
 ('positive_mesh_b2/results/validation.csv','validation','B2-positive'),
 ('positive_mesh_b2/results/final_timing.csv','timing','B2-positive'),
 ('theory_guided_joint_optimization/joint_candidates.csv','search','PPPM-original'),
 ('theory_guided_joint_optimization/results/final_validation.csv','validation','PPPM-original'),
 ('theory_guided_joint_optimization/results/final_timing.csv','timing','PPPM-original'),
 ('theory_guided_joint_optimization/results/mesh_scale_calibration.csv','mesh_calibration','PPPM-original'))
THEORY_FILES=(
 'ewald_longitudinal/test_parameter_selection.py',
 'ewald_longitudinal/verify_ewald_longitudinal.py',
 'benchmarks/method_A_clustered/parameter_optimizer.py',
 'benchmarks/method_A_clustered/validation_exp1/run_exp1.py',
 'benchmarks/method_A_clustered/validation_exp1/theory_guided_joint_optimization/stage1_theory.md',
 'benchmarks/method_A_clustered/validation_exp1/theory_guided_joint_optimization/theory.py',
 'benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/grid_psd.py')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def number(value):
    try:
        value=float(value);return value if math.isfinite(value) else None
    except (TypeError,ValueError):return None
def first(row,*keys):return next((number(row.get(k)) for k in keys if number(row.get(k)) is not None),None)
def truth(value):return value is True or str(value).lower() in ('true','1')
def git(*args):return subprocess.run(['git',*args],cwd=REPO,check=True,capture_output=True,text=True).stdout
def save_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def read_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))
def save_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def snapshot():
    excluded=HERE.relative_to(REPO)
    tracked=[name for name in git('ls-files','-z').split('\0') if name and not Path(name).is_relative_to(excluded)]
    if len(tracked)!=1048:raise RuntimeError(f'Expected 1048 prior tracked files, found {len(tracked)}')
    tracked_hash={name:sha(REPO/name) for name in tracked}
    # Include ignored old logs/checkpoints as well as tracked result files.
    old_results={str(p.relative_to(REPO)):sha(p) for p in sorted(VALIDATION.rglob('*'))
                 if p.is_file() and not p.is_relative_to(HERE) and 'results' in p.relative_to(VALIDATION).parts
                 and '__pycache__' not in p.parts}
    old=json.loads((PREVIOUS/'projection_bounds/offdiag_degree/source_snapshot.json').read_text())
    refs=old['independent_references']
    if len(refs)!=72:raise RuntimeError('Expected 72 direct-reference/sidecar records')
    for row in refs:
        for pathkey,hashkey in (('path','sha256'),('metadata_path','metadata_sha256')):
            if sha(REPO/row[pathkey])!=row[hashkey]:raise RuntimeError(f'Changed independent reference: {row[pathkey]}')
    source={str((VALIDATION/path).relative_to(REPO)):sha(VALIDATION/path) for path,_,_ in NAMES}
    value=dict(created_at=datetime.now(ZoneInfo('America/Detroit')).isoformat(),
               git=dict(head=git('rev-parse','HEAD').strip(),branch=git('branch','--show-current').strip(),status=git('status','--porcelain').strip()),
               excluded_new_directory=str(excluded),tracked_file_sha256=tracked_hash,
               old_results_sha256=old_results,independent_references=refs,source_csv_sha256=source,
               theory_source_sha256={name:sha(REPO/name) for name in THEORY_FILES},
               new_reference_policy='Any new N-scaling reference/configuration must live under final_exp2_selection; existing raw caches are read-only.')
    target=HERE/'source_snapshot.json'
    if target.exists():
        old=json.loads(target.read_text())
        for key in ('tracked_file_sha256','old_results_sha256','independent_references','source_csv_sha256','theory_source_sha256'):
            if old[key]!=value[key]:raise RuntimeError(f'Prior inputs changed after the initial final-Exp2 snapshot: {key}')
        return old
    save_json(target,value);return value


def canonical(method,default):
    return {'PPPM':'PPPM-original','B2':'B2-positive','original':'PPPM-original'}.get(method,method) or default


def main():
    initial=snapshot();catalog=[];summaries={};lookup={}
    loaded=[]
    for path,kind,default in NAMES:
        rows=read_csv(VALIDATION/path);loaded.append((path,kind,default,rows))
        if path.endswith('joint_candidates.csv'):
            lookup={r['candidate_id']:r for r in rows}
    for path,kind,default,rows in loaded:
        retained=0
        for index,raw in enumerate(rows,1):
            row=dict(raw)
            if path.startswith('theory_guided_joint_optimization') and raw.get('candidate_id') in lookup:
                row=dict(lookup[raw['candidate_id']],**raw)
            method=canonical(row.get('method',''),default)
            if method not in ('PPPM-original','PPPM-grid-PSD','B2-positive'):continue
            if first(row,'kappa') not in (None,.2) or first(row,'rho') not in (None,.75) or first(row,'N') not in (None,256.):continue
            par={key:first(row,key) for key in ('xi','sR','M','p')}
            par['sF']=first(row,'sF','sF_req');par['sT']=first(row,'sT')
            if any(par[key] is None for key in ('xi','sR','M','p')):continue
            if method!='B2-positive' and par['sF'] is None:continue
            if method=='B2-positive' and par['sT'] is None:par['sT']=36.
            key=method+'_'+'_'.join(k+format(par[k],'.12g') for k in ('xi','sR','sT' if method=='B2-positive' else 'sF','M','p'))
            nconfigs=first(row,'n_configs');config=row.get('config') or None
            maximum=first(row,'max_E_G','E_G_max_tested','E_G_max');error=first(row,'E_G','E_G_design')
            scope=('complete maximum over nine configurations' if nconfigs==9 and maximum is not None else
                   ('one configuration' if config or error is not None else 'historical metadata/timing only'))
            entry=dict(source_path=path,source_data_row=index,source_kind=kind,
                source_row_id=f'{path}:{index}',method=method,source_candidate_id=row.get('id',row.get('candidate_id')),
                parameter_key=key,config=config,family=row.get('family') or None,seed=first(row,'seed'),
                N=256,rho=.75,kappa=.2,amplitude=1.,**par,
                E_G=error,max_E_G=maximum,n_configs=nconfigs,error_scope=scope,
                nine_configuration_certificate=bool(nconfigs==9 and maximum is not None),
                mesh_valid=row.get('mesh_valid'),mesh_reason=row.get('mesh_reason'),eta=first(row,'eta','eta_actual'),
                E_proj=first(row,'E_proj','max_E_proj'),E_original=first(row,'E_original'),
                t_screen_s=first(row,'t_screen_s','t_nine_screen_s'),
                historical_matvec_s=first(row,'median_matvec_s','t_complete_s','t_matvec_s'),
                historical_setup_s=first(row,'median_config_setup_s','t_particle_setup_s','t_setup_s'),
                status=row.get('status') or None,stages=row.get('stages',row.get('stage')),
                use_policy='Prior seed/control or empirical calibration only; construct each new candidate own Grid PSD mesh and independently validate full E_G.')
            catalog.append(entry);retained+=1
        summaries[path]=dict(source_rows=len(rows),catalog_rows=retained,sha256=sha(VALIDATION/path))
    groups=defaultdict(list)
    for row in catalog:groups[row['parameter_key']].append(row)
    candidate_summary=[]
    for key,rs in sorted(groups.items()):
        cert=[r for r in rs if r['nine_configuration_certificate']]
        candidate_summary.append(dict(parameter_key=key,method=rs[0]['method'],**{k:rs[0][k] for k in ('xi','sR','sF','sT','M','p')},
                                      source_rows=len(rs),sources=sorted({r['source_path'] for r in rs}),
                                      prior_nine_config_max_E_G=min((r['max_E_G'] for r in cert),default=None)))
    save_csv(HERE/'source_catalog.csv',catalog)
    save_csv(HERE/'prior_candidate_catalog.csv',candidate_summary)
    mesh_source=(REPO/THEORY_FILES[0]).read_text()
    threshold=float(re.search(r'^INFL_THR\s*=\s*([\deE.+-]+)',mesh_source,re.M).group(1))
    prefactors=[];V=256/.75
    for xi in (.35,.6,.85,1.,1.025,1.25,1.5,1.75):
        z=.2/(2*xi);H=math.exp(-z*z)-math.sqrt(math.pi)*z*math.erfc(z)
        CS=4*math.sqrt(math.pi)*H/xi**2;CF=4*xi*H/math.pi
        prefactors.append(dict(xi=xi,kappa=.2,V=V,z=z,H_auxiliary=H,CS=CS,CF=CF,
                               CS_relative_predictor=CS*.2**2/(4*math.pi),
                               CF_relative_predictor=CF*V*.2**2/(4*math.pi)))
    theory=dict(actual_existing_legality=dict(eta_rule='eta_actual = kc*(L/M)/pi < 1 strictly',
                 fixed_eta_max_below_one_in_existing_code=None,
                 nyquist_rule='floor(kc*L/(2*pi)+1e-12) < M/2',
                 influence_denominator_threshold=threshold,denominator_rule='Every retained |W_hat|^2 > INFL_THR; n_excluded must be zero',
                 complete_shell_rule='n_modes+n_excluded equals independent reciprocal-lattice shell count',
                 assignment_legality='1 <= p <= M, enforced by ProjectedMesh',
                 source='validation_exp1/run_exp1.py:check_mesh and ewald_longitudinal/test_parameter_selection.py:Mesh'),
        initialization_model=dict(H='exp(-z^2)-sqrt(pi)*z*erfc(z), z=kappa/(2*xi); H is not mesh spacing',
             CS='4 sqrt(pi) H / xi^2',CF='4 xi H/pi',CS_relative='CS*kappa^2/(4*pi)',CF_relative='CF*V*kappa^2/(4*pi)',
             tail='C*s*exp(-s^2); solve on large-s branch using s_lambert; sqrt(log(C/epsilon)) is an initialization approximation',
             prefactors_kind='Continuum/mean-field predictor, not certified periodic operator error or acceptance constraint'),
        alias_mesh_models=dict(heuristic='q(eta)^p, q=eta/(2-eta); O((xi*h)^p) with a locally calibrated C_M',
             complete_mode_bound='alpha(k)=((1+2*zeta(p))^3-1)*max_a(|k_a|h/(2*pi-|k_a|h))^p for p>1',
             limitation='Original deconvolved mesh predictor only; active real-table PSD projection has no inherited h^p certificate. p need not monotonically improve projected full error.'),
        prefactor_values_at_target_physics=prefactors,
        normalization='Use same-physics independent Gamma_D norm; the kappa=.7/nonunit-amplitude defaults in common/parameter_optimizer are not the target physics.',
        selection_rule='Only mesh legality plus actual max-nine full E_G <= epsilon controls precision feasibility. No projection bound or matrix PSD eigenvalue test is a hard search constraint.',
        source_sha256=initial['theory_source_sha256'])
    save_json(HERE/'theory_source_audit.json',theory)
    output=dict(created_at=datetime.now(ZoneInfo('America/Detroit')).isoformat(),snapshot='source_snapshot.json',
                preservation=dict(old_tracked_files=len(initial['tracked_file_sha256']),old_result_files=len(initial['old_results_sha256']),
                                  independent_references=72,reference_sidecars=72),
                sources=summaries,catalog_rows=len(catalog),unique_prior_candidate_parameters=len(groups),
                method_rows=dict(Counter(r['method'] for r in catalog)),
                unique_candidates_by_method=dict(Counter(r['method'] for r in candidate_summary)),
                theory_audit='theory_source_audit.json',
                policy='Historical records are read-only hints/baselines. New formal rankings require a fresh matched same-machine/same-thread timing batch.')
    save_json(HERE/'source_catalog.json',output)
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
