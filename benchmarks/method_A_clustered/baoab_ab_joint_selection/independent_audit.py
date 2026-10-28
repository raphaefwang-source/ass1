"""Independent reconstruction of the finite local joint-selection evidence.

Run only after numerical timing, endpoint validation, analysis and figures are
complete. This module never changes historical or numerical evidence. Optional
--replay reexecutes first moving steps only after every timing batch has ended.
"""
from joint_common import *
import argparse
import ast
import hashlib
import platform
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime,timezone
from scipy import linalg,special
from model import lj_force,rng_sha

PREFIX=str(HERE.relative_to(REPO))+'/'
SMALL=1e-300
_REFS={}
_REDUCED={}


def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
def close(x,y,rtol=2e-8,atol=2e-12):return bool(np.allclose(x,y,rtol=rtol,atol=atol,equal_nan=False))
def norm2(A):return float(linalg.svdvals(np.asarray(A,float),check_finite=False)[0])
def qbasis(N):return np.kron(linalg.helmert(N,full=False).T,np.eye(3))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class Audit:
    def __init__(self):self.checks=[]
    def check(self,name,passed,detail=None):
        self.checks.append(dict(name=name,passed=bool(passed),detail=detail))
        print(('PASS ' if passed else 'FAIL ')+name,flush=True)
    def guard(self,name,fn):
        try:return fn()
        except Exception as exc:self.check(name,False,dict(exception=repr(exc)))


def tail_bound(N,L,R):
    # Independently evaluate the positive disjoint-lattice-cube tail allowance.
    a=np.sqrt(3)*L/2;k=.7;c=float(.5*1.3*np.exp(.7*1.3));e=np.exp(-k*R)
    integral=e*(R*R+2*R/k+2/k**2)
    integral+=(3*k*a+1)*e*(R/k+1/k**2)
    integral+=(3*k*a*a+3*a)*e/k
    integral+=3*a*a*special.exp1(k*R)+a**3*e/R
    return float(2*(N-1)*4*np.pi/3/L**3*integral*c)


def assemble_annulus_graph(annulus,pi,pj,N):
    """Assemble archived direct pair tensors without production helpers."""
    tensors=np.cumsum(np.asarray(annulus),axis=0)[-1]
    if tensors.ndim==2 and tensors.shape[-1]==6:
        # Direct reference archives xx,yy,zz,xy,xz,yz, not full3x3 blocks.
        tensors=tensors[:,np.array([[0,3,4],[3,1,5],[4,5,2]])]
    if tensors.shape!=(len(pi),3,3):raise ValueError('Unrecognized direct annulus tensor schema')
    if np.any(pi==pj):raise ValueError('Physical direct source contains self-particle edges')
    blocks=np.zeros((N,N,3,3))
    np.add.at(blocks,(pi,pi),tensors);np.add.at(blocks,(pj,pj),tensors)
    blocks[pi,pj]-=tensors;blocks[pj,pi]-=tensors
    return blocks.transpose(0,2,1,3).reshape(3*N,3*N)


def physical(meta):
    path=REPO/meta['source_file'];key=str(path)
    if key in _REFS:return _REFS[key]
    if sha(path)!=meta['source_sha256']:raise ValueError('Physical direct source file SHA differs')
    mpath=REPO/meta['source_metadata_file']
    if sha(mpath)!=meta['source_metadata_sha256']:raise ValueError('Physical reference metadata SHA differs')
    original=js(mpath)
    with np.load(path,allow_pickle=False) as data:d={k:data[k] for k in data.files}
    G=d['G'];q=np.mod(d['q'],BASE_PARAMS['L']);N=len(q)
    if array_sha(G)!=meta['matrix_sha256'] or array_sha(q)!=meta['q_sha256']:
        raise ValueError('Physical direct matrix or geometry SHA differs')
    if meta['kappa']!=.7 or not close(meta['c'],BASE_PARAMS['c'],atol=0,rtol=1e-15) or meta['R_max']<45:
        raise ValueError('Physical reference kernel/radius mismatch')
    analytic=tail_bound(N,BASE_PARAMS['L'],meta['R_max'])
    if meta['tail_absolute']+1e-18<analytic:raise ValueError('Physical direct tail allowance is not conservative')
    if not close(norm2(G),meta['norm2'],rtol=2e-12,atol=1e-12):raise ValueError('Direct graph norm differs')
    if all(k in d for k in ('annulus_tensors','pi','pj')):
        assembled=assemble_annulus_graph(d['annulus_tensors'],d['pi'],d['pj'],N)
        if not close(assembled,G,rtol=2e-13,atol=2e-13):raise ValueError('Direct annulus tensors do not assemble saved G')
    _REFS[key]=(G,q,meta);return _REFS[key]


def exact_reduced(G,Q):
    reduced=Q.T@G@Q;lam,V=linalg.eigh((reduced+reduced.T)/2,check_finite=False)
    tol=256*np.finfo(float).eps*max(float(np.max(np.abs(lam))),1.)
    if lam[0]<-tol:raise ValueError('Materially negative exact-reference spectrum')
    lam=np.maximum(lam,0.)
    R=(V*np.exp(-DT*lam))@V.T
    C=(V*(-np.expm1(-2*DT*lam)))@V.T
    return R,C,lam


def exact_action_from_matrix(G,vector,h,kind):
    """Independent COM eigendecomposition and matrix-function action."""
    key=array_sha(G)
    if key not in _REDUCED:
        Q=qbasis(len(G)//3);reduced=Q.T@G@Q
        lam,V=linalg.eigh((reduced+reduced.T)/2,check_finite=False)
        tol=256*np.finfo(float).eps*max(float(np.max(np.abs(lam))),1.)
        if lam[0]<-tol:raise ValueError('Materially negative discrete-reference spectrum')
        _REDUCED[key]=(Q@V,np.maximum(lam,0.))
    U,lam=_REDUCED[key];v=np.asarray(vector,float);x=v.reshape(-1,3)
    x=(x-x.mean(0)).ravel()
    f=np.exp(-h*lam) if kind=='damping' else np.sqrt(-np.expm1(-2*h*lam))
    return (U@(f*(U.T@x))).reshape(v.shape)


def baseline_checks(a):
    snapshot=js(HERE/'baseline_snapshot.json');hashes=snapshot['tracked_file_sha256']
    tracked=git('ls-tree','-r','--name-only',BASE).splitlines()
    a.check('registered_PR11_base_and_complete_2586_file_snapshot',snapshot['base_sha']==BASE
            and len(hashes)==2586 and set(hashes)==set(tracked),dict(base=BASE,count=len(hashes)))
    changed=[n for n,h in hashes.items() if not (REPO/n).is_file() or sha(REPO/n)!=h]
    a.check('all_PR3_to_PR11_tracked_source_and_evidence_bytes_unchanged',not changed,changed)
    paths=git('diff',BASE,'--name-only').splitlines()
    paths += subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=REPO).decode().split('\0')
    outside=[p for p in paths if p and not p.startswith(PREFIX)]
    a.check('changes_confined_new_joint_selection_directory',not outside,outside)
    env=js(RESULTS/'environment_initial.json')
    a.check('local_Mac_requested_branch_and_fixed_two_tier_scope',platform.system()=='Darwin'
        and git('branch','--show-current')=='codex/baoab-ab-joint-selection'
        and env['machine']=='arm64' and list(TIERS)==[1e-4,1e-6],env)
    errors=[]
    for p in HERE.glob('*.py'):
        try:ast.parse(p.read_text(),filename=str(p))
        except SyntaxError as exc:errors.append(str(exc))
    a.check('all_new_Python_sources_parse',not errors,errors)
    for name,minimum in [('baseline_regressions.log',106),('test_joint_driver.log',8),
            ('test_joint_spatial.log',1),('test_joint_thermo.log',1),('test_joint_postvalidate.log',4),
            ('test_joint_plots.log',4),('test_independent_audit.log',3)]:
        p=LOGS/name;text=p.read_text() if p.exists() else '';m=re.findall(r'Ran (\d+) tests? in ([0-9.]+)s',text)
        a.check(name+'_passed',bool(m) and int(m[-1][0])>=minimum and text.rstrip().endswith('OK'),
            dict(count=int(m[-1][0]) if m else None,minimum=minimum))


def provenance_checks(a):
    data=js(RESULTS/'backend_provenance.json');backend=data['backend'];issues=[]
    source=THERMOSTAT/'optimized_pairou/edge_sweep.c'
    if Path(data['source_path'])!=source or sha(source)!=backend['source_sha256']:issues.append('Unchanged reused C source SHA')
    if sha(backend['library_path'])!=backend['library_sha256']:issues.append('Actual warm C binary SHA')
    if backend['compiled_now'] or backend['fast_math'] or backend['fp_contract'] or backend['edge_parallelism']:issues.append('Unexpected compilation/algorithm transformation')
    if any(v!='1' for v in data['native_thread_requests'].values()):issues.append('Single native-thread requests not recorded')
    a.check('unchanged_compiled_PairOU_source_binary_and_post_timing_native_runtime_provenance',not issues,
        dict(issues=issues,library_path=backend['library_path'],source_sha256=backend['source_sha256'],
             native_threadpool_observations=data['threadpoolctl'],scope=data['native_count_scope']))


def source_checks(a):
    domain=js(RESULTS/'domain.json');states=load_states();issues=[]
    names={s['label'] for s in states}
    if not {'initial_0','initial_1','initial_2','visited_global_min_distance','visited_1_B_200'}.issubset(names):issues.append('Missing actual initial/extreme geometry')
    if domain.get('canonical_preparation_steps_run')!=0:issues.append('Unexpected repeated canonical preparation')
    seeds=[];qsha=[]
    for i in range(3):
        p=PILOT/'inputs'/f'initial_{i}.npz';meta=js(PILOT/'results'/f'initial_{i}.json')
        with np.load(p,allow_pickle=False) as d:
            if sha(p)!=meta['file_sha256'] or array_sha(d['q'])!=meta['q_sha256'] or array_sha(d['p'])!=meta['p_sha256']:issues.append('Initial source SHA '+str(i))
            if np.linalg.norm(d['p'].sum(0))>1e-10:issues.append('Initial COM '+str(i))
            qsha.append(array_sha(d['q']));seeds.append(meta['seed'])
    if len(set(seeds))!=3 or len(set(qsha))!=3:issues.append('Initial independence provenance')
    a.check('three_distinct_hash_checked_starts_reused_without_preparation_and_actual_extreme_domain',not issues,
        dict(issues=issues,seeds=seeds,registered_states=sorted(names),new_preparation_steps=0))


def spatial_checks(a):
    folder=RESULTS/'spatial';selection=js(folder/'spatial_selection.json')
    cheap=read_csv(folder/'cheap_screen.csv');issues=[]
    if len({r['setting_id'] for r in cheap})!=len(cheap) or len(cheap)>100:issues.append('Candidate uniqueness/budget')
    if len({r['xi'] for r in cheap if r.get('xi') is not None})<3:issues.append('No actual xi diversity')
    if not any(r.get('sR')!=r.get('sF') for r in cheap):issues.append('No asymmetric cutoffs explored')
    if selection['baseline_setting_id'] not in {r['setting_id'] for r in cheap}:issues.append('Old feasible control omitted')
    for r in cheap:
        if r.get('status')=='illegal_or_failed' and not r.get('error'):issues.append('Unrecorded rejection cause')
    a.check('adaptive_multivariable_search_with_failures_and_old_feasible_control_saved',not issues,
        dict(issues=issues,candidate_count=len(cheap),unique_xi=sorted({r['xi'] for r in cheap if r.get('xi') is not None}),
        candidate_scope='cheap action lower bounds are screening evidence, not full operator passes'))
    records=[];issues=[];max_diff=0.
    for path in sorted(folder.glob('spatial_sp_*.json')):
        rec=js(path)
        if 'matrix_file' not in rec:continue
        file=HERE/rec['matrix_file']
        if sha(file)!=rec['matrix_file_sha256']:issues.append(path.name+': matrix file SHA')
        with np.load(file,allow_pickle=False) as data:d={k:data[k] for k in data.files}
        Gref,qref,meta=physical(rec['reference']);GH=d['GH'];Q=qbasis(len(d['q']))
        if array_sha(d['q'])!=rec['q_sha256'] or array_sha(GH)!=rec['GH_sha256']:issues.append(path.name+': raw state/matrix SHA')
        if not np.array_equal(np.mod(d['q'],BASE_PARAMS['L']),qref):issues.append(path.name+': physical geometry')
        delta=norm2(Q.T@(GH-Gref)@Q);bound=(delta+meta['tail_absolute'])/(meta['norm2']-meta['tail_absolute'])
        max_diff=max(max_diff,abs(bound-rec['spatial_error_conservative_bound']))
        if not close(bound,rec['spatial_error_conservative_bound'],atol=2e-12):issues.append(path.name+': conservative spatial norm')
        if bool(bound<=2e-7)!=rec['spatial_passed'] or rec['dimension_COM']!=765 or not rec['full_operator_norm_computed']:issues.append(path.name+': fullnorm pass flag')
        for i in (0,1):
            if not close(GH@d[f'probe{i}'],d[f'dense{i}'],rtol=2e-12,atol=2e-12):issues.append(path.name+': dense probe')
            if not close(d[f'fft{i}'],d[f'dense{i}'],rtol=2e-10,atol=2e-11):issues.append(path.name+': FFT action')
        if rec['structural_checks_passed'] and (rec['self_edges']!=0 or not rec['physical_k0_retained']
            or rec['excluded_modes']!=0 or rec['COM_null_relative']>1e-11 or rec['relative_symmetry']>1e-12):issues.append(path.name+': physical structure')
        records.append(rec)
    for setting in selection['qualified_settings']:
        rr=[r for r in records if r['setting_id']==setting['setting_id']]
        if not set(setting['validation_states']).issubset({r['state'] for r in rr}):issues.append('Incomplete setting domain '+setting['setting_id'])
        if not all(r['spatial_passed'] and r['structural_checks_passed'] for r in rr):issues.append('Unqualified setting '+setting['setting_id'])
    a.check('full_765D_physical_spatial_norms_tail_allowances_and_FFT_structure_reconstruct',not issues
        and len(selection['qualified_settings'])>=2,
        dict(issues=issues[:100],issue_count=len(issues),full_operator_records=len(records),qualified=len(selection['qualified_settings']),
             max_bound_recompute_difference=max_diff,unique_physical_direct_sources=len(_REFS)))


def thermo_checks(a):
    folder=RESULTS/'thermo';selection=js(folder/'thermo_selection.json');rows=read_csv(folder/'ideal_errors.csv')
    refs={};issues=[];checked=[]
    for row in rows:
        sid,label=row['setting_id'],row['state'];key=(sid,label)
        if key not in refs:
            meta=js(folder/f'reference_{sid}_{label}.json');path=HERE/meta['file']
            if sha(path)!=meta['file_sha256']:issues.append('Discrete reference file SHA '+sid+label)
            with np.load(path,allow_pickle=False) as data:raw={k:data[k] for k in data.files}
            if array_sha(raw['GH'])!=meta['GH_sha256'] or array_sha(raw['GL'])!=meta['GL_sha256']:issues.append('Discrete raw matrix SHA '+sid+label)
            Gp,_,physical_meta=physical(meta['physical_reference']);Q=qbasis(len(raw['q']))
            Rp,Cp,_=exact_reduced(Gp,Q);Rh,Ch,_=exact_reduced(raw['GH'],Q)
            refs[key]=(raw,Q,Rp,Cp,Rh,Ch,physical_meta)
        raw,Q,Rp,Cp,Rh,Ch,meta=refs[key]
        file=HERE/row['maps_file']
        if sha(file)!=row['maps_file_sha256']:issues.append(file.name+': map file SHA')
        with np.load(file,allow_pickle=False) as data:R,C=data['RA_reduced'],data['CA_reduced']
        if R.shape!=(765,765) or C.shape!=R.shape:issues.append(file.name+': full COM dimension')
        if array_sha(R)!=row['RA_reduced_sha256'] or array_sha(C)!=row['CA_reduced_sha256']:issues.append(file.name+': raw map SHA')
        if not close(C,np.eye(765)-R@R.T,rtol=2e-10,atol=2e-12):issues.append(file.name+': ideal exact linear covariance identity')
        mr=norm2(R-Rp);cr=norm2(C-Cp);rd=norm2(Rp);cd=norm2(Cp)
        dr=DT*meta['tail_absolute'];dc=2*dr
        mb=(mr+dr)/max(rd-dr,SMALL);cb=(cr+dc)/max(cd-dc,SMALL)
        values=dict(physical_mean_absolute=mr,physical_cov_absolute=cr,
            physical_mean_reference_norm=rd,physical_cov_reference_norm=cd,
            physical_mean_tail_allowance=dr,physical_cov_tail_allowance=dc,
            physical_mean_bound=mb,physical_cov_bound=cb,
            own_mean_relative=norm2(R-Rh)/max(norm2(Rh),SMALL),own_cov_relative=norm2(C-Ch)/max(norm2(Ch),SMALL),
            spatial_transition_mean_relative=norm2(Rh-Rp)/max(rd,SMALL),spatial_transition_cov_relative=norm2(Ch-Cp)/max(cd,SMALL))
        if any(not close(row[n],v,rtol=2e-7,atol=2e-11) for n,v in values.items()):issues.append(file.name+': physical/own/space transition decomposition')
        checked.append(dict(row,independent_physical_mean_bound=mb,independent_physical_cov_bound=cb))
    for setting in selection['settings']:
        sid=setting['setting_id'];labels=setting['validation_states']
        for target in setting['targets']:
            eps=target['epsilon_frozen'];accepted=[]
            ks=setting.get('executed_ks',selection.get('executed_ks_by_setting',{}).get(sid,selection.get('ks',[])))
            for k in ks:
                rr=[r for r in checked if r['setting_id']==sid and r['method']=='A' and r['k']==k]
                if len(rr)==len(labels) and all(max(r['independent_physical_mean_bound'],r['independent_physical_cov_bound'])<=eps and r['spatial_passed'] for r in rr):accepted.append(int(k))
            if target['A_k']!=(min(accepted) if accepted else None):issues.append(sid+': minimum actually tested physical passing k')
            bb=[r for r in checked if r['setting_id']==sid and r['method']=='B']
            passed=len(bb)==len(labels) and all(max(r['independent_physical_mean_bound'],r['independent_physical_cov_bound'])<=eps and r['spatial_passed'] for r in bb)
            if target['B_passed']!=passed:issues.append(sid+': B common physical reference flag')
    a.check('ideal_mean_covariance_full_COM_norms_against_common_physical_and_own_Gamma_reconstruct',not issues,
        dict(issues=issues[:100],issue_count=len(issues),actual_map_checks=len(checked),reference='independent unsplit periodic full physical Gamma exact OU',
             scope='finite registered geometry domain; no physical-observable or uniform nonlinear finite-Lanczos claim'))


def finite_checks(a):
    folder=RESULTS/'thermo';selected=js(folder/'rank_selection.json');issues=[];paths=0;actions=0;wrappers=0
    expected_paths=0;expected_actions=0;seen_groups=set()
    attempts=read_csv(folder/'rank_attempts.csv')
    # Independently reconstruct stored own-reference paths/actions and all stochastic draws.
    from spatial_validation import build_mesh,build_operator,near_edges
    params_by_id={c['spatial_id']:c['params'] for c in selected['configs']};meshes={};edges_cache={};matrices={}
    def matrix(sid,state,method):
        key=(sid,state,method)
        if key not in matrices:
            with np.load(folder/f'reference_{sid}_{state}.npz',allow_pickle=False) as data:
                matrices[key]=data['GL' if method=='A' else 'GH'].copy()
        return matrices[key]
    for file in sorted(folder.glob('path_sp_*.json')):
        record=js(file);sid=record['spatial_id'];state=record['state'];method=record['method'];k=int(record['k'])
        npz=file.with_suffix('.npz')
        with np.load(npz,allow_pickle=False) as data:d={n:data[n] for n in data.files}
        if sid not in meshes:meshes[sid]=build_mesh(params_by_id[sid],BASE_PARAMS['L'])
        key=(sid,array_sha(d['q']))
        if key not in edges_cache:
            op=build_operator(d['q'],params_by_id[sid],meshes[sid]);edges_cache[key]=len(near_edges(op))
        E=edges_cache[key];rng=np.random.default_rng(record['seed'])
        if rng.bit_generator.state!=record['rng_state_before']:issues.append(file.name+': RNG before')
        for i,row in enumerate(record['substeps']):
            z=rng.standard_normal(d['p'].shape);ef=er=None
            if method=='A':ef=rng.standard_normal((E,3));er=rng.standard_normal((E,3))
            for name,arr in [('z',z),('ef',ef),('er',er)]:
                expected=array_sha(arr) if arr is not None else None
                if row[name+'_sha256']!=expected:issues.append(file.name+': independent substep Gaussian '+name)
            if row['substep']!=i or row['sub_dt']!=DT/k or row['negative_ritz_count']!=0:issues.append(file.name+': complete inner OU contract')
            if array_sha(d['actual_substeps'][i])!=row['p_actual_sha256'] or array_sha(d['ideal_substeps'][i])!=row['p_ideal_sha256']:issues.append(file.name+': per-inner raw output SHA')
        if rng.bit_generator.state!=record['rng_state_after']:issues.append(file.name+': RNG after')
        error=np.linalg.norm(d['actual']-d['ideal'])/max(np.linalg.norm(d['ideal']),np.finfo(float).tiny)
        if not close(error,record['coupled_relative_error'],rtol=2e-8,atol=1e-16) or error>record['full_path_budget'] or not record['passed']:issues.append(file.name+': complete own-ideal composed path')
        for row in record['action_checks']:
            prefix=f"sub{row['substep']}_{row['kind']}_";inp,exact,finite=(d[prefix+n] for n in ('input','exact','finite'))
            err=np.linalg.norm(finite-exact)/max(np.linalg.norm(exact),np.finfo(float).tiny)
            if err>row['budget'] or not close(err,row['relative_error'],rtol=2e-8,atol=1e-16):issues.append(file.name+': actual input matrix-function error')
            independently_exact=exact_action_from_matrix(matrix(sid,state,method),inp,DT/k,row['kind'])
            if not close(independently_exact,exact,rtol=2e-11,atol=2e-12):issues.append(file.name+': independent own-Gamma matrix-function action')
            for name,arr in [('input',inp),('exact',exact),('actual',finite)]:
                if array_sha(arr)!=row[name+'_sha256']:issues.append(file.name+': actual action SHA')
            actions+=1
        paths+=1
    for file in sorted(folder.glob('finite_sp_*.json')):
        record=js(file)
        if not record['passed'] or record['failures'] or not record['wrapper_bitwise_equal']:issues.append(file.name+': finite certificate')
        gid=record.get('group_id',f"{record['spatial_id']}_{record['method']}_k{record['k']}")
        seen_groups.add(gid);expected_paths+=len(record['validation_states']);expected_actions+=2*record['k']*len(record['validation_states'])
        for kind,key in [('damping','r_d'),('noise','r_n')]:
            minima=[]
            for state in record['validation_states']:
                rr=[r for r in attempts if r['spatial_id']==record['spatial_id'] and r['method']==record['method']
                    and r['k']==record['k'] and r['state']==state and r['kind']==kind and r['budget']==record['per_action_budget']]
                tested=sorted({r['rank_requested'] for r in rr})
                passing=[rank for rank in tested if all(r['passed'] for r in rr if r['rank_requested']==rank)]
                if not passing:issues.append(file.name+': no actual rank search pass '+state+kind)
                else:minima.append(min(passing))
            if len(minima)!=len(record['validation_states']) or max(minima,default=0)!=record[key]:issues.append(file.name+': fresh selected common rank '+kind)
        npz=folder/f"wrapper_{gid}.npz"
        with np.load(npz,allow_pickle=False) as d:
            if not np.array_equal(d['production'],d['validated_actual']):issues.append(file.name+': actual production wrapper mismatch')
        csv=folder/f"common_probes_{gid}.csv";common={}
        for row in read_csv(csv):
            if not row['passed'] or row['relative_error']>record['per_action_budget'] or row['negative_ritz_count']!=0:issues.append(file.name+': final common rank guard')
            state=row['state']
            if state not in common:
                with np.load(folder/f'common_{gid}_{state}.npz',allow_pickle=False) as data:common[state]={k:data[k] for k in data.files}
            prefix=f"{row['kind']}__{row['probe']}__"
            inp,exact,finite=(common[state][prefix+n] for n in ('probe','exact','actual'))
            err=float(np.linalg.norm(finite-exact)/max(np.linalg.norm(exact),np.finfo(float).tiny))
            if not close(err,row['relative_error'],rtol=2e-8,atol=1e-16):issues.append(file.name+': registered probe raw relative error')
            independently_exact=exact_action_from_matrix(matrix(record['spatial_id'],state,record['method']),inp,DT/record['k'],row['kind'])
            if not close(independently_exact,exact,rtol=2e-11,atol=2e-12):issues.append(file.name+': independent registered probe exact action')
            for name,arr in [('probe',inp),('exact',exact),('actual',finite)]:
                if array_sha(arr)!=row[name+'_sha256']:issues.append(file.name+': registered probe raw SHA')
        wrappers+=1
    for c in selected['configs']:
        if not all(c['certificate'].get(k) is True for k in ('spatial_passed','ideal_physical_passed','finite_passed')):issues.append('Incomplete running configuration '+c['config_id'])
        if c['certificate']['finite_validation_group'] not in seen_groups:issues.append('Unrecorded finite validation group '+c['config_id'])
    a.check('fresh_per_spatial_setting_ranks_actual_composed_paths_Gaussians_and_production_wrappers',not issues and selected['all_checks_passed']
        and wrappers==selected['groups'] and wrappers>0 and paths==expected_paths and actions==expected_actions,
        dict(issues=issues[:100],issue_count=len(issues),paths=paths,actual_input_actions=actions,production_wrappers=wrappers,
        scope='finite directions and coupled own-ideal paths only, not a fixed nonlinear noise covariance or raw A/B bias'))


def audit_run(label):
    meta=js(RESULTS/f'{label}.json');rows=read_csv(RESULTS/f'{label}_steps.csv');inner=read_csv(RESULTS/f'{label}_substeps.csv')
    state=js(RESULTS/f'{label}_rng.json');file=HERE/meta['file'];issues=[]
    with np.load(file,allow_pickle=False) as data:d={n:data[n] for n in data.files}
    q,p,mid=d['q_unwrapped'],d['p'],d['q_mid'];steps=meta['steps'];method=meta['method'];k=int(meta['k'])
    if q.shape!=(steps+1,256,3) or p.shape!=q.shape or mid.shape!=(steps,256,3):issues.append('dimensions')
    if len(rows)!=steps or len(inner)!=steps*k or len(state['states'])!=steps:issues.append('outer/inner counts')
    if method=='B' and k!=1:issues.append('B repeated OU')
    if sha(file)!=meta['file_sha256'] or not all(np.isfinite(x).all() for x in (q,p,mid)):issues.append('saved finite trajectory/SHA')
    if not np.array_equal(np.mod(q,BASE_PARAMS['L']),d['q']):issues.append('wrapped/unwrapped separation')
    for name,arr in [('start_q',q[0]),('start_p',p[0]),('end_q',q[-1]),('end_p',p[-1])]:
        if array_sha(arr)!=meta[name+'_sha256']:issues.append(name+' hash')
    with np.load(PILOT/'inputs'/f"initial_{meta['initial_index']}.npz",allow_pickle=False) as data:
        if not np.array_equal(q[0],data['q_unwrapped']) or not np.array_equal(p[0],data['p']):issues.append('common initial control')
    rng=np.random.default_rng(meta['seed']);tokens=[];maxmid=0.;maxkick=0.;maxcom=0.;maxforce=0.
    for j,row in enumerate(rows):
        rr=[r for r in inner if r['outer_step']==j+1];where=f'step{j+1}'
        if len(rr)!=k or [r['substep'] for r in rr]!=list(range(1,k+1)):issues.append(where+': inner order')
        if state['states'][j]['before']!=rng.bit_generator.state or row['rng_state_before_sha256']!=rng_sha(rng):issues.append(where+': RNG before')
        hashes=[]
        for sub in rr:
            before=rng_sha(rng);z=rng.standard_normal((256,3));ef=er=None
            if method=='A':ef=rng.standard_normal((int(row['edge_count']),3));er=rng.standard_normal((int(row['edge_count']),3))
            hd=dict(substep=int(sub['substep']),z_sha256=array_sha(z),eta_forward_sha256=array_sha(ef) if ef is not None else '',eta_reverse_sha256=array_sha(er) if er is not None else '')
            for name,value in hd.items():
                if sub[name]!=(None if value=='' else value):issues.append(where+': independent Gaussian '+name)
            if sub['rng_state_before_sha256']!=before or sub['rng_state_after_sha256']!=rng_sha(rng):issues.append(where+': inner RNG states')
            if sub['dt_inner']!=DT/k or sub['dt_outer']!=DT or sub['negative_ritz_count']!=0:issues.append(where+': complete inner dt/Ritz')
            if sub['q_mid_sha256']!=row['q_mid_sha256'] or sub['r_d']!=row['r_d'] or sub['r_n']!=row['r_n']:issues.append(where+': frozen graph/ranks')
            hashes.append(hd)
        if digest(hashes)!=row['gaussian_substeps_sha256']:issues.append(where+': ordered Gaussian content hash')
        if state['states'][j]['after']!=rng.bit_generator.state or row['rng_state_after_sha256']!=rng_sha(rng):issues.append(where+': RNG after')
        for name,arr in [('q_before',q[j]),('q_after',q[j+1]),('p_before',p[j]),('p_after',p[j+1]),('q_mid',mid[j])]:
            if array_sha(arr)!=row[name+'_sha256']:issues.append(where+': raw state '+name)
        force0=lj_force(np.mod(q[j],BASE_PARAMS['L']),BASE_PARAMS['L'])[0]
        ph=p[j]+.5*DT*force0;qm=q[j]+.5*DT*ph
        em=float(np.max(np.abs(np.mod(qm,BASE_PARAMS['L'])-mid[j])));maxmid=max(maxmid,em)
        force1,U,minr=lj_force(np.mod(q[j+1],BASE_PARAMS['L']),BASE_PARAMS['L']);po=p[j+1]-.5*DT*force1
        ek=float(np.max(np.abs(qm+.5*DT*po-q[j+1])));maxkick=max(maxkick,ek)
        maxforce=max(maxforce,float(np.max(np.abs(force0))),float(np.max(np.abs(force1))))
        if em>2e-13 or ek>2e-13:issues.append(where+': actual BAOAB midpoint/drift/kick')
        obs=dict(Tkin=float(np.sum((p[j+1]-p[j+1].mean(0))**2)/(3*256-3)),U_per_particle=U/256,min_pair_distance=minr,
            momentum_norm=float(np.linalg.norm(p[j+1].sum(0))),q_motion_rms=float(np.sqrt(np.mean((q[j+1]-q[j])**2))))
        if any(not close(row[n],v) for n,v in obs.items()) or obs['q_motion_rms']<=0:issues.append(where+': actual moving observables')
        maxcom=max(maxcom,obs['momentum_norm'])
        if row['graph_builds_this_step']!=1 or not row['q_dependent_setup_rebuilt'] or not row['fft_parameter_table_reused']:issues.append(where+': q lifecycle')
        if row['k']!=k or row['substeps']!=k or row['ou_calls']!=k or row['dt']!=DT:issues.append(where+': running k/dt')
        if method=='A':
            if row['long_ou_calls']!=k or row['full_ou_calls']!=0 or row['coefficient_tau']!=DT/(2*k):issues.append(where+': full subcycling')
            if row['edge_eigh_builds_this_step']!=1 or row['coefficient_sets_this_step']!=1:issues.append(where+': repeated setup')
            if row['coefficient_cache_initial_size']!=0 or row['coefficient_first_prepare_hit'] is not False or row['coefficient_cache_hits']!=2*k or row['active_backend']!='compiled':issues.append(where+': actual coefficient rebuild/compiled backend')
        elif row['full_ou_calls']!=1 or row['long_ou_calls']!=0:issues.append(where+': full B count')
        if row['r_d']!=meta['r_d'] or row['r_n']!=meta['r_n'] or row['negative_ritz_count']!=0:issues.append(where+': active ranks/Ritz')
        for key in ('xi','sR','sF','M','p'):
            if row[key]!=meta['params'][key]:issues.append(where+': active spatial '+key)
        major=sum(row[key] for key in ('t_dynamic_setup_s','t_LJ_forces_s','t_rng_s','t_ou_wall_s','t_baoab_other_s'))
        if abs(major-row['t_baoab_s'])>1e-11 or min(row[key] for key in ('t_dynamic_setup_s','t_LJ_forces_s','t_rng_s','t_ou_wall_s','t_baoab_other_s'))<-1e-10:issues.append(where+': disjoint fullwall phases')
        if not close(row['wall_per_simulated_time'],row['t_baoab_s']/DT,rtol=1e-12,atol=1e-12):issues.append(where+': physical-time objective')
        for field,subfield in [('t_rng_s','t_rng_s'),('t_ou_wall_s','t_ou_wall_s'),('t_pair_forward_s','t_pair_forward_s'),('t_pair_reverse_s','t_pair_reverse_s'),('t_lanczos_damp_s','t_lanczos_damp_s'),('t_lanczos_noise_s','t_lanczos_noise_s'),('t_reorth_s','t_reorth_s')]:
            if not close(row[field],sum(s[subfield] for s in rr),rtol=2e-12,atol=1e-12):issues.append(where+': all inner clocks '+field)
        if row['t_edge_eigh_s']>row['t_edge_constructor_s']+1e-10 or row['t_lanczos_damp_s']+row['t_lanczos_noise_s']>row['t_ou_wall_s']+1e-10:issues.append(where+': nested clocks')
        if row['table_sha256']!=meta['table_sha256'] or row['peak_rss_bytes']<=0:issues.append(where+': table/memory')
        tokens.append(row['q_mid_sha256'])
    if len(set(tokens))!=steps or maxcom>1e-8:issues.append('moving midpoint uniqueness/COM')
    total=sum(r['t_baoab_s'] for r in rows)
    if not close(total,meta['compute_wall_s'],rtol=1e-12,atol=1e-12) or not close(total/(steps*DT),meta['wall_per_simulated_time'],rtol=1e-12,atol=1e-12):issues.append('actual objective summary')
    if meta['loop_wall_including_evidence_s']+1e-9<total:issues.append('outer loop wall smaller thancompute')
    return rows,inner,meta,dict(issues=issues[:100],issue_count=len(issues),max_qmid_error=maxmid,max_second_drift_error=maxkick,max_COM_norm=maxcom,max_force_component=maxforce)


def moving_checks(a):
    explore=js(RESULTS/'exploratory_done.json');heads=js(RESULTS/'heads_done.json');allrows=[];allinner=[];runs={}
    for item in explore['runs']+heads['runs']:
        rr,ss,meta,detail=audit_run(item['run']);allrows.extend(rr);allinner.extend(ss);runs[item['run']]=meta
        a.check(item['run']+'_actual_RNG_subcycling_fresh_graph_BAOAB_and_clocks',not detail['issues'],detail)
    if sum(r['steps'] for r in explore['runs'])!=explore['actual_moving_steps']:raise ValueError('Exploratory actual count')
    if sum(r['steps'] for r in heads['runs'])!=heads['actual_moving_steps']:raise ValueError('Head actual count')
    from joint_driver import rank_exploratory,configuration_key,paired_schedule,SEED as SEED_DRIVER
    rankings,winners=rank_exploratory(explore['configs'],explore['runs'])
    issues=[]
    if [(w['config_id'],w['score']) for w in winners]!=[(w['config_id'],w['score']) for w in explore['winners']]:issues.append('Measured independent optimum selection')
    by_id={c['config_id']:c for c in explore['configs']}
    for group in heads['groups']:
        gid=group['group_id']
        if group['comparison_type']=='controlled_common_baseline':
            baseline=js(RESULTS/'spatial/spatial_selection.json')['baseline_setting_id']
            if group['A']['spatial_id']!=baseline or group['B']['spatial_id']!=baseline:issues.append('Uncontrolled old common baseline '+gid)
        if group['comparison_type']=='independent_optimum':
            for m in ('A','B'):
                winner=next(w for w in winners if w['method']==m and w['epsilon_frozen']==group['epsilon_frozen'])
                if group[m]['config_id']!=winner['config_id']:issues.append('Independent measured head selection '+gid)
        if group.get('reuse_group_id'):
            source=next(g for g in heads['groups'] if g['group_id']==group['reuse_group_id'])
            if any(configuration_key(group[m])!=configuration_key(source[m]) for m in ('A','B')):issues.append('Improper batch reuse '+gid)
            continue
        rr=[r for r in heads['runs'] if r['group_id']==gid]
        if len(rr)!=2*heads['rounds_per_unique_group']:issues.append('Complete paired rounds '+gid)
        index=heads['groups'].index(group);schedule=paired_schedule(group,heads['rounds_per_unique_group'],SEED_DRIVER+200000+index)
        for slot in schedule:
            pair={m:next(r for r in rr if r['method']==m and r['round']==slot['round']) for m in ('A','B')}
            if pair['A']['initial_source']!=pair['B']['initial_source']:issues.append('Unpaired initial source '+gid)
            if sorted(pair,key=lambda m:pair[m]['order_position'])!=slot['order']:issues.append('Declared balanced interleave order '+gid)
    a.check('complete_exploratory_and_paired_head_counts_actual_objective_and_independent_vs_controlled_selection',not issues,
        dict(issues=issues,exploratory_steps=explore['actual_moving_steps'],head_steps=heads['actual_moving_steps'],inner_calls=len(allinner),
             comparison_groups=len(heads['groups']),reuse_groups=sum(bool(g.get('reuse_group_id')) for g in heads['groups']),
             objective='complete actual wall / simulated physical time at dt=.005; not cached matvec cost or global optimum'))


def endpoint_checks(a):
    path=RESULTS/'post_validation.json'
    if not path.exists():raise ValueError('Final expanded-domain and actual-endpoint validation marker missing')
    data=js(path);heads=js(RESULTS/'heads_done.json');issues=[];count=0;actual_passes={}
    required_sources={str(p.relative_to(REPO)) for p in [HERE/'joint_postvalidate.py',HERE/'joint_spatial.py',HERE/'joint_thermo.py',HERE/'joint_common.py',MATCHED/'matched_dynamics.py',PILOT/'model.py']}
    execution_sources=data['execution_source_sha256']
    if set(execution_sources)!=required_sources or any(sha(REPO/name)!=value for name,value in execution_sources.items()):issues.append('Post-validation execution source identity changed/incomplete')
    by_run={r['run']:r for r in heads['runs']};chosen={}
    for run in heads['runs']:
        key=(run['group_id'],run['method'],run['initial_index'])
        if key not in chosen or run['round']>chosen[key]['round']:chosen[key]=run
    expected_runs={r['run'] for r in chosen.values()}
    if {r['run'] for r in data['records']}!=expected_runs:issues.append('Missing or unregistered final sampled actual midpoint')
    for rec in data['records']:
        file=HERE/rec['file']
        if sha(file)!=rec['file_sha256']:issues.append(file.name+': endpoint file SHA')
        with np.load(file,allow_pickle=False) as raw:d={k:raw[k] for k in raw.files}
        G,_,meta=physical(rec['physical_reference']);Q=qbasis(len(d['q']));Rp,Cp,_=exact_reduced(G,Q)
        R,C=d['RA_reduced'],d['CA_reduced'];dr=DT*meta['tail_absolute'];dc=2*dr
        mean=(norm2(R-Rp)+dr)/max(norm2(Rp)-dr,SMALL);cov=(norm2(C-Cp)+dc)/max(norm2(Cp)-dc,SMALL)
        spatial=(norm2(Q.T@(d['GH']-G)@Q)+meta['tail_absolute'])/(meta['norm2']-meta['tail_absolute'])
        ideal_passed=max(mean,cov)<=rec['epsilon_frozen'];spatial_passed=spatial<=2e-7
        if not close(mean,rec['physical_mean_bound'],rtol=2e-7,atol=2e-11) or not close(cov,rec['physical_cov_bound'],rtol=2e-7,atol=2e-11) or not close(spatial,rec['spatial_error_bound'],rtol=2e-7,atol=2e-11):issues.append(file.name+': endpoint physical transition/spatial norm')
        if bool(rec['ideal_physical_passed'])!=ideal_passed or bool(rec['spatial_passed'])!=spatial_passed:issues.append(file.name+': independent endpoint ideal/space classification')
        if rec.get('run') is not None:
            run=by_run[rec['run']];config=run['config'];j=int(rec['step'])-1
            if j+1!=run['steps'] or run['round']!=rec['round'] or run['group_id']!=rec['group_id'] or run['initial_index']!=rec['initial_index']:issues.append(file.name+': actual last-round endpoint selection')
            if rec['timed_source_file']!=run['file'] or rec['timed_source_file_sha256']!=run['file_sha256']:issues.append(file.name+': source trajectory provenance')
            with np.load(INPUTS/f"{rec['run']}.npz",allow_pickle=False) as trajectory:
                q0,p0=trajectory['q_unwrapped'][j],trajectory['p'][j]
                ph=p0+.5*DT*lj_force(np.mod(q0,BASE_PARAMS['L']),BASE_PARAMS['L'])[0]
                if not np.array_equal(d['q'],trajectory['q_mid'][j]) or not close(d['p'],ph,rtol=0,atol=1e-13):issues.append(file.name+': actual timed midpoint/halfkick')
                po=trajectory['p'][j+1]-.5*DT*lj_force(np.mod(trajectory['q_unwrapped'][j+1],BASE_PARAMS['L']),BASE_PARAMS['L'])[0]
            finite=rec['finite'];finite_file=HERE/finite['file']
            if js(HERE/rec['finite_record_file'])!=finite or sha(finite_file)!=finite['file_sha256']:issues.append(file.name+': finite raw record provenance')
            with np.load(finite_file,allow_pickle=False) as raw:fd={k:raw[k] for k in raw.files}
            if not np.array_equal(fd['q'],d['q']) or not np.array_equal(fd['p_half'],ph):issues.append(file.name+': finite actual input geometry/momentum')
            if not close(fd['timed_O_reconstructed'],po,rtol=0,atol=1e-13):issues.append(file.name+': saved timed O inverse second kick')
            replay_abs=float(np.max(np.abs(fd['actual_O']-po)))
            replay_relative=float(np.linalg.norm(fd['actual_O']-po)/max(np.linalg.norm(po),np.finfo(float).tiny))
            coupled=float(np.linalg.norm(fd['actual_O']-fd['own_ideal_O'])/max(np.linalg.norm(fd['own_ideal_O']),np.finfo(float).tiny))
            com=float(np.linalg.norm(fd['actual_O'].sum(0)-ph.sum(0)))
            if not close(replay_abs,finite['timed_O_output_replay_abs_error'],rtol=2e-8,atol=1e-16) or not close(replay_relative,finite['timed_O_output_replay_relative_error'],rtol=2e-8,atol=1e-16) or not close(coupled,finite['coupled_own_ideal_path_error'],rtol=2e-8,atol=1e-16) or not close(com,finite['COM_error'],rtol=2e-8,atol=1e-16):issues.append(file.name+': actual coupled/replay/COM metrics')
            k=int(config['k']);method=config['method'];h=DT/k;action_errors=[]
            rng_file=HERE/finite['source_rng_file']
            if sha(rng_file)!=finite['source_rng_file_sha256']:issues.append(file.name+': actual RNG file SHA')
            rng_record=js(rng_file)['states'][j];rng=np.random.default_rng();rng.bit_generator.state=rng_record['before']
            outer=read_csv(RESULTS/f"{run['run']}_steps.csv")[j]
            saved_inner=[r for r in read_csv(RESULTS/f"{run['run']}_substeps.csv") if r['outer_step']==j+1]
            if len(finite['actual_noise_substeps'])!=k or len(finite['substep_paths'])!=k or len(finite['action_checks'])!=2*k:issues.append(file.name+': complete actual inner evidence count')
            for l in range(1,k+1):
                z=rng.standard_normal(ph.shape);ef=er=None
                if method=='A':ef=rng.standard_normal((int(outer['edge_count']),3));er=rng.standard_normal((int(outer['edge_count']),3))
                noise=finite['actual_noise_substeps'][l-1];saved=saved_inner[l-1]
                for name,arr in [('z',z),('eta_forward',ef),('eta_reverse',er)]:
                    value=array_sha(arr) if arr is not None else None
                    if noise[name+'_sha256']!=value or saved[name+'_sha256']!=value:issues.append(file.name+': actual independent endpoint noise '+name)
                if not np.array_equal(fd[f'sub{l}_noise_input'],z):issues.append(file.name+': actual saved Lanczos noise vector')
                sub=finite['substep_paths'][l-1]
                if sub['substep']!=l or sub['dt_inner']!=h or sub['r_d_used']>config['r_d'] or sub['r_n_used']>config['r_n'] or sub['negative_ritz_count']!=0:issues.append(file.name+': endpoint actual path k/rank/Ritz')
                for name,suffix in [('actual','actual'),('own_ideal','own_ideal')]:
                    if array_sha(fd[f'sub{l}_{suffix}'])!=sub[name+'_sha256']:issues.append(file.name+': endpoint per-inner output SHA')
            if rng.bit_generator.state!=rng_record['after']:issues.append(file.name+': complete actual endpoint RNG state')
            for action in finite['action_checks']:
                prefix=f"sub{action['substep']}_{action['kind']}_";inp,exact,actual=(fd[prefix+n] for n in ('input','exact','finite'))
                error=float(np.linalg.norm(actual-exact)/max(np.linalg.norm(exact),np.finfo(float).tiny));action_errors.append(error)
                independent=exact_action_from_matrix(d['GL' if method=='A' else 'GH'],inp,h,action['kind'])
                if not close(independent,exact,rtol=2e-11,atol=2e-12) or not close(error,action['relative_error'],rtol=2e-8,atol=1e-16):issues.append(file.name+': independent endpoint exact action/error')
                rank=config['r_d' if action['kind']=='damping' else 'r_n']
                if action['rank_requested']!=rank or action['budget']!=config['certificate']['per_action_budget'] or bool(action['passed'])!=(error<=action['budget']):issues.append(file.name+': actual endpoint selected rank/budget')
                for name,arr in [('input',inp),('exact',exact),('actual',actual)]:
                    if array_sha(arr)!=action[name+'_sha256']:issues.append(file.name+': endpoint action raw SHA')
            if not np.array_equal(fd[f'sub{k}_actual'],fd['actual_O']) or not np.array_equal(fd[f'sub{k}_own_ideal'],fd['own_ideal_O']):issues.append(file.name+': complete endpoint path output')
            finite_passed=max(action_errors)<=config['certificate']['per_action_budget'] and coupled<=config['certificate']['full_path_budget'] and finite['negative_ritz_count']==0 and close(fd['actual_O'],po,rtol=1e-12,atol=1e-12) and com<=1e-9
            if finite['finite_passed']!=finite_passed or not close(max(action_errors),finite['max_actual_input_action_error'],rtol=2e-8,atol=1e-16):issues.append(file.name+': endpoint finite path classification')
            passed=ideal_passed and spatial_passed and finite_passed
            if bool(rec['passed'])!=passed:issues.append(file.name+': physical/finite combined classification')
            actual_passes[rec['run']]=passed
        count+=1
    for group in data['groups']:
        source=group['source_group_id'];rr=[r for r in data['records'] if r['group_id']==source]
        if len(rr)!=6 or set(group['runs'])!={r['run'] for r in rr} or {(r['method'],r['initial_index']) for r in rr}!={(m,i) for m in ('A','B') for i in range(3)}:issues.append(group['group_id']+': six actual endpoint coverage')
        if group['passed']!=all(actual_passes[r['run']] for r in rr):issues.append(group['group_id']+': independent accuracy flag')
    if data['all_passed']!=all(actual_passes.values()) or data['actual_sampled_midpoints']!=count:issues.append('Post-validation overall declared classification/count')
    a.check('expanded_and_actual_new_midpoint_errors_against_common_physical_reference_reconstruct',not issues and count>0,
        dict(issues=issues,actual_records=count,declared_all_passed=data['all_passed'],scope='sampled registered and actual q_mid, not a uniform trajectory guarantee'))


def statistical_checks(a):
    analysis=js(RESULTS/'analysis.json');records=analysis.get('comparisons',analysis.get('groups',analysis.get('targets',[])))
    blocks=read_csv(RESULTS/'heads_rounds.csv');issues=[];checked=0;samples={}
    heads=js(RESULTS/'heads_done.json');group_indices={g['group_id']:i for i,g in enumerate(heads['groups'])}
    post={g['group_id']:g for g in js(RESULTS/'post_validation.json')['groups']}
    steps=read_csv(RESULTS/'heads_steps.csv');groups={g['group_id']:g for g in heads['groups']}
    from joint_driver import configuration_key
    with np.load(RESULTS/'paired_bootstrap_samples.npz',allow_pickle=False) as data:saved_samples={k:data[k] for k in data.files}
    for rec in records:
        gid=rec['group_id'];source=rec.get('source_group_id') or rec.get('reuse_group_id') or gid
        group=groups[gid]
        if source!=(group.get('reuse_group_id') or gid):issues.append(gid+': declared identical-group timing source')
        if rec['accuracy_passed']!=post[gid]['passed'] or rec['post_certificate']!=post[gid]:issues.append(gid+': accuracy gate differs from actual physical/finite post certificate')
        rr=[r for r in blocks if r['group_id']==source]
        ratios=np.array([r['A_over_B'] for r in rr]);value=float(np.exp(np.log(ratios).mean()))
        if not close(value,rec['ratio'],rtol=1e-12,atol=1e-12):issues.append(gid+': paired round ratio')
        if 'bootstrap_seed' in rec:
            seed=int(rec['bootstrap_seed']);n=int(rec.get('bootstrap_replicates',20000))
            if rec.get('bootstrap_group_index')!=group_indices[source] or seed!=analysis['bootstrap_seed']+group_indices[source]:issues.append(gid+': declared independent group bootstrap seed')
            if source not in samples:
                rng=np.random.default_rng(seed)
                ix=rng.integers(0,len(rr),size=(n,len(rr)));samples[source]=np.exp(np.log(ratios[ix]).mean(1))
            bs=samples[source];low,high=np.quantile(bs,[.025,.975])
            if not close(low,rec['ratio_ci_low'],rtol=1e-12,atol=1e-12) or not close(high,rec['ratio_ci_high'],rtol=1e-12,atol=1e-12):issues.append(gid+': paired block bootstrap')
            if gid not in saved_samples or not np.array_equal(saved_samples[gid],bs):issues.append(gid+': complete paired bootstrap sample archive')
        else:issues.append(gid+': missing explicit per-group bootstrap seed')
        if rec.get('accuracy_passed',True):expected='A_faster' if rec['ratio_ci_high']<1 else ('B_faster' if rec['ratio_ci_low']>1 else 'INCONCLUSIVE')
        else:expected='accuracy_not_met'
        if rec['verdict']!=expected:issues.append(gid+': honest uncertainty/accuracy verdict')
        for m in ('A','B'):
            actual=1000*np.mean([r[m+'_mean_s'] for r in rr])
            if not close(actual,rec[m+'_mean_ms'],rtol=1e-12,atol=1e-10):issues.append(gid+': actual full dynamic mean '+m)
            ss=[r for r in steps if r['group_id']==source and r['method']==m]
            if configuration_key(rec[m+'_config'])!=configuration_key(group[m]) or rec[m+'_actual_steps']!=len(ss):issues.append(gid+': actual selected configuration/count '+m)
            if not close(rec[m+'_wall_per_simulated_time'],actual/1000/DT,rtol=1e-12,atol=1e-12) or not close(rec[m+'_median_ms'],1000*np.median([r['t_baoab_s'] for r in ss]),rtol=1e-12,atol=1e-10):issues.append(gid+': complete dynamic objective/median '+m)
        checked+=1
    a.check('paired_round_means_bootstrap_uncertainty_and_accuracy_conditioned_verdicts_recompute',not issues and checked>=4,
        dict(issues=issues,comparisons=checked,scope='whole paired round is sample unit; finite local state/timing blocks only'))


def artifact_checks(a):
    issues=[];pairs=[]
    for png in sorted((HERE/'figures').glob('*.png')):
        svg=png.with_suffix('.svg')
        if png.read_bytes()[:8]!=b'\x89PNG\r\n\x1a\n' or not svg.exists():issues.append(png.name)
        else:
            try:
                if not ET.parse(svg).getroot().tag.endswith('svg'):issues.append(svg.name)
            except ET.ParseError:issues.append(svg.name)
        pairs.append(png.name)
    a.check('three_or_more_actual_PNG_SVG_pairs_and_joint_Pareto_performance_artifacts',not issues and len(pairs)>=3,dict(figures=pairs,issues=issues))
    figure=js(HERE/'figures/figure_manifest.json');issues=[]
    for name,value in figure['source_sha256'].items():
        if sha(HERE/name)!=value:issues.append('Figure source SHA '+name)
    required={'accuracy_cost_pareto','independent_controlled_baoab','joint_parameter_sensitivity','frozen_splitting_vs_k','head_phase_costs'}
    for rec in figure['figures']:
        if sha(HERE/rec['rows_file'])!=rec['rows_sha256']:issues.append('Plotted raw rows SHA '+rec['name'])
        for file in rec['files']:
            if sha(HERE/file['file'])!=file['sha256']:issues.append('Figure pixels/vector SHA '+file['file'])
    if not required.issubset({r['name'] for r in figure['figures']}) or figure['historical_timings_used'] is not False:issues.append('Required actual experiment figures or input scope')
    report_sources=js(RESULTS/'report_sources.json')
    if sha(HERE/report_sources['report_file'])!=report_sources['report_sha256'] or report_sources['historical_timings_used'] is not False:issues.append('Generated report SHA/input scope')
    for name,value in report_sources['source_sha256'].items():
        if sha(HERE/name)!=value:issues.append('Generated report source SHA '+name)
    a.check('all_figure_plotted_rows_and_generated_report_source_SHA_chains_match',not issues,dict(issues=issues))
    report=HERE/'report.md';readme=HERE/'README.md'
    a.check('Chinese_report_and_reproducible_commands_with_limited_scientific_scope_saved',report.exists() and readme.exists()
        and bool(re.search('[\u4e00-\u9fff]',report.read_text())) and 'python3' in readme.read_text())


def replay_checks(a):
    from matched_dynamics import MatchedSystem
    done=js(RESULTS/'heads_done.json');issues=[];records=[]
    # One actual first step per unique running configuration after timing has ended.
    seen=set()
    for item in done['runs']:
        config=item['config'];key=(item['config_id'],item['initial_index'])
        if key in seen:continue
        seen.add(key)
        if item['round']!=0:continue
        with np.load(HERE/item['file'],allow_pickle=False) as data:
            q,p,q1,p1=(data['q_unwrapped'][0].copy(),data['p'][0].copy(),data['q_unwrapped'][1].copy(),data['p'][1].copy())
        system=MatchedSystem(config['params'])
        qr,pr,_=system.step(q,p,config['method'],config['r_d'],config['r_n'],np.random.default_rng(item['seed']),dt=DT,k=config['k'])
        error_q=float(np.max(np.abs(qr-q1)));error_p=float(np.max(np.abs(pr-p1)))
        if not close(qr,q1,rtol=1e-12,atol=1e-12) or not close(pr,p1,rtol=1e-12,atol=1e-12):issues.append(item['run'])
        records.append(dict(run=item['run'],max_q_abs_error=error_q,max_p_abs_error=error_p,
            q_bitwise=np.array_equal(qr,q1),p_bitwise=np.array_equal(pr,p1)))
    a.check('independent_local_firststep_replay_of_selected_head_configurations',not issues and bool(records),dict(issues=issues,replays=records))


def manifest():
    result={}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(HERE)
        if any(x in ('__pycache__','.build','build') for x in rel.parts) or path.suffix=='.tmp':continue
        if path.name=='independent_audit.json' or (path.name.startswith('independent_audit') and path.suffix=='.log'):continue
        result[str(rel)]=sha(path)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    if not (RESULTS/'heads_done.json').exists():raise SystemExit('Refusing audit before all numerical head timing is complete')
    from joint_driver import SEED as SEED_DRIVER
    audit=Audit()
    for name,fn in [('baseline',baseline_checks),('provenance',provenance_checks),('sources',source_checks),('spatial',spatial_checks),('thermo',thermo_checks),
            ('finite',finite_checks),('moving',moving_checks),('endpoint',endpoint_checks),('statistics',statistical_checks),('artifacts',artifact_checks)]:
        audit.guard(name,lambda fn=fn:fn(audit))
    if args.replay:audit.guard('replay',lambda:replay_checks(audit))
    result=dict(status='PASS' if audit.checks and all(c['passed'] for c in audit.checks) else 'FAIL',
        timestamp_utc=datetime.now(timezone.utc).isoformat(),checks=audit.checks,base_sha=BASE,
        branch=git('branch','--show-current'),replay_executed=bool(args.replay),
        execution='local Mac only; audit after dynamic timings; no HPC/SSH/cloud numerical compute',
        scope='finite adaptive parameter candidate study and sampled physical frozen-accuracy constrained dynamic cost; no global optimum or equal physical-observable guarantee',
        evidence_sha256=manifest())
    save(RESULTS/'independent_audit.json',result)
    print(json.dumps(dict(status=result['status'],checks=len(audit.checks),failures=[c['name'] for c in audit.checks if not c['passed']]),indent=2),flush=True)
    raise SystemExit(result['status']!='PASS')
