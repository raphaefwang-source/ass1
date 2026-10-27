"""Four-step, theory-initialized Experiment 2 search; immutable old methods.

Only full error against the independent reference controls empirical precision.
Alias/cutoff predictions initialize candidates, while projection diagnostics,
degree bounds and PSD eigenvalue tests are never acceptance constraints.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import study as S
np = S.np
from test_parameter_selection import prefactors, s_lambert, INFL_THR

METHODS = ('PPPM-original','B2-positive','PPPM-grid-PSD')
EPSILONS = (1e-3,1e-4,1e-5)
PARAM_KEYS = ('xi','sR','sF','sT','M','p','kappa','rho')
XI_SEEDS = (.55,.8,1.05,1.3,1.6,1.9)
ETA_MAX = .95
MAX_M = 128
L0 = (256/.75)**(1/3)


def smooth_sizes(maximum=MAX_M):
    out = []
    for M in range(2,int(maximum)+1):
        remainder = M
        for factor in (2,3,5):
            while remainder%factor==0:remainder//=factor
        if remainder==1:out.append(M)
    return out


def next_smooth(value,maximum=MAX_M):
    return next((M for M in smooth_sizes(maximum) if M>=value),None)


def previous_smooth(value):
    return next((M for M in reversed(smooth_sizes()) if M<value),None)


def parfrom(row):
    return S.parameters(**{k:row[k] for k in PARAM_KEYS if row.get(k) is not None})


@lru_cache(maxsize=128)
def _modes(kc,L):
    return S.E.mesh_modes(kc,L)[0]


def legal_metadata(method,par,L=L0,eta_max=ETA_MAX,max_M=MAX_M):
    """Pure shell/denominator legality: no kernel/mesh/operator construction."""
    p,M = par['p'],par['M']
    info = dict(search_mesh_legal=False,eta_max=eta_max,influence_threshold=INFL_THR,
                resource_max_M=max_M)
    if not all(math.isfinite(float(par[k])) for k in PARAM_KEYS):
        return dict(info,search_mesh_reason='nonfinite parameter')
    if int(p)!=p or int(M)!=M or not 1<=p<=M or M<2:
        return dict(info,search_mesh_reason='integer M>=2 and 1<=p<=M required')
    if M>max_M:
        return dict(info,search_mesh_reason='resource maximum M exceeded',resource_unavailable=True)
    if min(par[k] for k in ('xi','sR','sF','sT','kappa','rho'))<=0:
        return dict(info,search_mesh_reason='positive finite physical/cutoff parameters required')
    if method=='B2-positive':
        return dict(info,search_mesh_legal=True,search_mesh_reason='',eta_prediction=None,
                    W2min_prediction=None,positive_mesh_only=True)
    try:rc,kc,shell = S.E.cutoffs(par,L)
    except ValueError as exc:return dict(info,search_mesh_reason=str(exc))
    eta = kc*L/(M*np.pi)
    mv = _modes(float(kc),float(L))
    mc = int(np.abs(mv).max())
    W2 = np.prod(np.sinc(mv/M)**(2*p),axis=1)
    minimum = float(W2.min())
    excluded = int(np.sum(W2<=INFL_THR))
    valid = mc<M/2 and eta<=eta_max and eta<1 and excluded==0
    return dict(info,search_mesh_legal=bool(valid),search_mesh_reason='' if valid else
                'Nyquist, explicit eta safety, or influence denominator stability failed',
                eta_prediction=float(eta),W2min_prediction=minimum,n_excluded_prediction=excluded,
                kc_prediction=float(kc),rc_prediction=float(rc),shell_prediction=int(shell),
                sF_effective_prediction=float(kc/(2*par['xi'])))


def minimum_legal_M(par,L=L0,eta_max=ETA_MAX,max_M=MAX_M):
    for M in smooth_sizes(max_M):
        candidate = dict(par,M=M)
        if legal_metadata('PPPM-grid-PSD',candidate,L,eta_max,max_M)['search_mesh_legal']:
            return M
    return None


def cutoff_initialization(epsilon,xi,L=L0,kappa=.2):
    """Existing continuum/mean-field tail predictor, never a certified bound."""
    P = prefactors(kappa,xi,L**3)
    sR = s_lambert(epsilon/3,P['CS_rel'])
    sF = s_lambert(epsilon/3,P['CF_rel'])
    if not np.isfinite(sR) or not np.isfinite(sF):
        raise ValueError('The existing large-s Lambert branch is unavailable')
    return dict(theory_CS_relative=float(P['CS_rel']),theory_CF_relative=float(P['CF_rel']),
                theory_sR=float(sR),theory_sF=float(sF),cutoff_error_budget=float(epsilon/3),
                cutoff_model='existing C*s*exp(-s^2) continuum/mean-field initialization, not certificate')


def _read_csv(path):
    if not Path(path).exists():return []
    with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in r.items():
            if v=='':r[k]=None
            elif v in ('True','False'):r[k]=v=='True'
            else:
                try:r[k]=float(v)
                except (TypeError,ValueError):pass
    return rows


def alias_calibration(rows=None):
    """Same-kappa historical empirical scale, used only to initialize q^p.

    Complete old errors include tails: this is deliberately labeled a predictor,
    not an alias-only fit or an acceptance bound.  Low-p/minimal-M probes remain
    in the new experiment so an overconservative predictor cannot fix the answer.
    """
    rows = _read_csv(HERE/'source_catalog.csv') if rows is None else rows
    samples = {}
    for r in rows:
        if r.get('method')!='PPPM-original' or r.get('status')!='ok' or r.get('E_G') is None:
            continue
        if r.get('kappa')!=.2 or r.get('rho')!=.75 or not 3<=r.get('p',0)<=5:
            continue
        error = float(r['E_G'])
        if not 1e-6<=error<=1e-2:continue
        par = parfrom(r)
        info = legal_metadata('PPPM-original',par)
        eta = info.get('eta_prediction')
        if eta is None or not .45<=eta<=.95:continue
        q = eta/(2-eta)
        ratio = error/q**par['p']
        key = tuple(par[k] for k in PARAM_KEYS)
        samples[key] = ratio
    scale = float(np.median(list(samples.values()))) if samples else 1e-3
    return dict(alias_C=scale,n_unique_calibration_points=len(samples),target_kappa=.2,target_rho=.75,
        source='same-physics prior Original complete errors and q(eta)^p; initialization only',
        model='C_alias*(eta/(2-eta))^p',fallback_C=1e-3,certified=False,
        limitation='contains cutoff errors; neither alias-only estimate nor certified projected error')


def theory_candidates(epsilon,alias_C,xis=XI_SEEDS,L=L0):
    """Four-step main seeds and measured low-p probes at minimum safe M."""
    out = []
    for seed_number,xi in enumerate(xis):
        tails = cutoff_initialization(epsilon,xi,L)
        base = S.parameters(xi=xi,sR=tails['theory_sR'],sF=tails['theory_sF'],M=2,p=2)
        M0 = minimum_legal_M(base,L)
        if M0 is None:continue
        chosen = None
        for M in smooth_sizes():
            if M<M0:continue
            for p in range(2,min(9,M)+1):
                par = dict(base,M=M,p=p)
                legal = legal_metadata('PPPM-grid-PSD',par,L)
                if not legal['search_mesh_legal']:continue
                eta = legal['eta_prediction'];q=eta/(2-eta)
                if alias_C*q**p<=epsilon/3:
                    chosen=(par,legal,q);break
            if chosen:break
        common = dict(epsilon=float(epsilon),stage='seeds',origin='four_step_theory',
            theory_seed_id=f'eps{epsilon:g}_xi{xi:g}',theory_xi_index=seed_number,
            initial_minimum_legal_M=M0,alias_C=float(alias_C),alias_error_budget=float(epsilon/3),
            predicted_error_is_acceptance_constraint=False,bound_used_as_constraint=False,
            PSD_eigenvalue_used_as_constraint=False,**tails)
        if chosen:
            par,legal,q = chosen
            out.append(dict(common,**par,**legal,seed_role='smallest_predicted_reasonable_p',
                alias_q=float(q),alias_prediction=float(alias_C*q**par['p'])))
        for p in (3,5,7):
            if p>M0:continue
            par = dict(base,M=M0,p=p)
            legal = legal_metadata('PPPM-grid-PSD',par,L)
            if not legal['search_mesh_legal']:continue
            eta=legal['eta_prediction'];q=eta/(2-eta)
            out.append(dict(common,**par,**legal,seed_role='independent_low_order_probe',
                alias_q=float(q),alias_prediction=float(alias_C*q**p)))
    return out


def proxy_seconds(row):
    value = row.get('t_screen_s')
    return float(value) if value is not None and math.isfinite(value) and value>0 else float('inf')


def successful(row):
    return row.get('status')=='ok' and row.get('E_G') is not None and math.isfinite(row['E_G'])


def search_centers(rows,epsilon,number=2):
    good = [r for r in rows if successful(r)]
    feasible = sorted([r for r in good if r['E_G']<=epsilon],key=proxy_seconds)
    borderline = sorted([r for r in good if epsilon<r['E_G']<=5*epsilon],key=proxy_seconds)
    accurate = sorted(good,key=lambda r:r['E_G'])
    selected = []
    for row in feasible[:1]+borderline[:1]+feasible[1:]+accurate:
        if row['id'] not in {r['id'] for r in selected}:selected.append(row)
        if len(selected)>=number:break
    return selected


def joint_neighbors(center,epsilon,stage='local'):
    """Independent and genuinely joint changes of all five search variables."""
    par=parfrom(center);M,p=par['M'],par['p']
    lower=previous_smooth(M)
    smaller=next_smooth(max(p,M*.8))
    larger=next_smooth(M*1.15)
    changes = [
        ('joint_relax',dict(M=smaller or M,p=max(2,p-2),sR=max(1.,par['sR']-.3),sF=max(1.,par['sF']-.3))),
        ('M_down',dict(M=lower or M)),('p_down',dict(p=max(2,p-1))),
        ('xi_down',dict(xi=round(par['xi']*.925,6))),('sR_down',dict(sR=max(1.,par['sR']-.2))),
        ('sF_down',dict(sF=max(1.,par['sF']-.2))),('M_up',dict(M=larger or M)),
        ('p_up',dict(p=min(M,p+1))),('xi_up',dict(xi=round(par['xi']*1.075,6))),
        ('sR_up',dict(sR=par['sR']+.2)),('sF_up',dict(sF=par['sF']+.2)),
        ('joint_tighten',dict(M=larger or M,p=min(M,p+1),sR=par['sR']+.2,sF=par['sF']+.2)),
        ('low_p_3',dict(p=min(M,3))),('low_p_5',dict(p=min(M,5))),
    ]
    if stage=='boundary':
        changes=[
            ('boundary_M',dict(M=lower or M)),('boundary_p',dict(p=max(2,p-1))),
            ('boundary_xi_down',dict(xi=round(par['xi']-.025,6))),
            ('boundary_xi_up',dict(xi=round(par['xi']+.025,6))),
            ('boundary_joint',dict(M=lower or M,sR=max(1.,par['sR']-.05),sF=max(1.,par['sF']-.05))),
            ('boundary_tails',dict(sR=max(1.,par['sR']-.1),sF=max(1.,par['sF']-.1))),
        ]
    out=[]
    for reason,change in changes:
        candidate=dict(par,**change)
        out.append(dict(candidate,epsilon=float(epsilon),stage=stage,origin='joint_measured_neighborhood',
            parent_id=center['id'],neighbor_reason=reason,theory_seed_id=center.get('theory_seed_id'),
            bound_used_as_constraint=False,PSD_eigenvalue_used_as_constraint=False,
            **legal_metadata(center['method'],candidate)))
    return out


def _prior_performance():
    performance={}
    for row in _read_csv(HERE/'source_catalog.csv'):
        key=row.get('parameter_key')
        for field in ('historical_matvec_s','t_screen_s'):
            value=row.get(field)
            if key and value and value>0:performance[key]=min(performance.get(key,float('inf')),value)
    return performance


def comparator_candidates(method,epsilon):
    prior=_read_csv(HERE/'prior_candidate_catalog.csv');perf=_prior_performance()
    feasible=[r for r in prior if r['method']==method and r.get('prior_nine_config_max_E_G') is not None
              and r['prior_nine_config_max_E_G']<=epsilon]
    feasible.sort(key=lambda r:perf.get(r['parameter_key'],float('inf')))
    selected=feasible[:2]
    if feasible:
        accurate=min(feasible,key=lambda r:r['prior_nine_config_max_E_G'])
        if accurate not in selected:selected.append(accurate)
    out=[]
    for row in selected:
        par=parfrom(row)
        out.append(dict(par,epsilon=float(epsilon),stage='seeds',origin='historical_initialization_fresh_recheck',
            historical_parameter_key=row['parameter_key'],historical_nine_max_E_G=row['prior_nine_config_max_E_G'],
            historical_error_reused_as_acceptance=False,**legal_metadata(method,par)))
    if method=='B2-positive':
        # Existing measured second-order positive-mesh bias initializes each
        # accuracy tier; actual own-candidate full error decides acceptance.
        xis=(.35,.5) if epsilon>=1e-3 else ((.25,.35) if epsilon>=1e-4 else (.15,.2))
        for xi in xis:
            target_M=64*(xi/.325)*math.sqrt(1e-4/epsilon)
            M=next_smooth(target_M)
            if M is None:M=MAX_M+1
            tails=cutoff_initialization(epsilon,xi)
            par=S.parameters(xi=xi,sR=tails['theory_sR'],M=M,p=2,sF=tails['theory_sF'])
            out.append(dict(par,epsilon=float(epsilon),stage='seeds',origin='measured_B2_second_order_initialization',
                B2_bias_model='same-physics measured error scales approximately (xi/M)^2, only initialization',
                **tails,**legal_metadata(method,par)))
    return out


def shortlist_rows(rows,epsilon,maximum_fast=3):
    """Screen shortlist; no projection bound or PSD eigenvalue is consulted."""
    good=[r for r in rows if successful(r)]
    passing=sorted([r for r in good if r['E_G']<=epsilon],key=proxy_seconds)
    selected=passing[:maximum_fast]
    if good:
        most_accurate=min(good,key=lambda r:r['E_G'])
        if most_accurate not in selected:selected.append(most_accurate)
    return selected


def nine_pass(summary,epsilon):
    error=summary.get('max_E_G')
    return bool(summary.get('status')=='ok' and summary.get('n_configs')==9 and
                error is not None and math.isfinite(error) and error<=epsilon)


def _protocol():
    import protocol
    return protocol


def _record_candidates(points,method):
    P=_protocol()
    existing=P.read(P.RESULTS/'candidates.csv');byid={r['id']:r for r in existing}
    memberships=P.read(P.RESULTS/'candidate_memberships.csv')
    membership_keys={(r['id'],r['epsilon'],r['stage']) for r in memberships}
    new=[]
    for point in points:
        meta={k:v for k,v in point.items() if k not in PARAM_KEYS}
        cand=P.candidate(method,point,**meta)
        if cand['id'] not in byid:byid[cand['id']]=cand;new.append(cand)
        mk=(cand['id'],cand['epsilon'],cand['stage'])
        if mk not in membership_keys:
            memberships.append(dict(id=cand['id'],method=method,epsilon=cand['epsilon'],stage=cand['stage'],
                origin=cand.get('origin'),theory_seed_id=cand.get('theory_seed_id'),parent_id=cand.get('parent_id')))
            membership_keys.add(mk)
    P.write(P.RESULTS/'candidates.csv',list(byid.values()))
    P.write(P.RESULTS/'candidate_memberships.csv',memberships)
    return new


def evaluate_candidates(candidates,workers):
    P=_protocol();path=P.RESULTS/'search.csv';rows=P.read(path);done={r['id'] for r in rows}
    pending=[c for c in candidates if c['id'] not in done]
    invalid=[dict(c,status='invalid_mesh',failure=c.get('search_mesh_reason'),E_G=None) for c in pending
             if not c.get('search_mesh_legal',True)]
    pending=[c for c in pending if c.get('search_mesh_legal',True)]
    rows.extend(invalid);P.write(path,rows)
    if pending:
        P.log(f'screen batch: {len(pending)} legal candidates, workers={workers}; proxy only')
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(P.screen_one,c):c for c in pending}
            for future in as_completed(futures):
                measured=future.result();rows.append(measured);P.write(path,rows)
                P.log(f'{measured["stage"]} {measured["id"]}: status={measured["status"]}, E_G={measured.get("E_G")}')
    budget(P)
    return rows


def _rows_for(method,epsilon):
    P=_protocol();members=P.read(P.RESULTS/'candidate_memberships.csv')
    ids={r['id'] for r in members if r['method']==method and r['epsilon']==epsilon}
    return [r for r in P.read(P.RESULTS/'search.csv') if r['id'] in ids]


def budget(P=None):
    P=P or _protocol();rows=P.read(P.RESULTS/'search.csv');members=P.read(P.RESULTS/'candidate_memberships.csv')
    P.save(P.RESULTS/'search_budget.json',dict(
        unique_fresh_candidates=len(rows),candidate_epsilon_stage_memberships=len(members),
        eta_max=ETA_MAX,resource_max_M=MAX_M,
        status_counts={method:{status:sum(r['method']==method and r['status']==status for r in rows)
            for status in sorted({r['status'] for r in rows})} for method in METHODS},
        precision_constraint='fresh max-nine complete E_G <= epsilon only; no bounds or PSD eigenvalue constraint',
        budgets={method:{str(eps):dict(unique_candidates=len({r['id'] for r in members if
            r['method']==method and r['epsilon']==eps}),stages={stage:sum(r['method']==method and
            r['epsilon']==eps and r['stage']==stage for r in members) for stage in ('seeds','local','boundary')})
            for eps in EPSILONS} for method in METHODS}))


def seeds(epsilons,workers):
    P=_protocol();calibration=alias_calibration();P.save(P.RESULTS/'alias_calibration.json',calibration)
    all_new=[];theory=[]
    for epsilon in epsilons:
        points=theory_candidates(epsilon,calibration['alias_C']);theory.extend(points)
        all_new.extend(_record_candidates(points,'PPPM-grid-PSD'))
        for method in ('PPPM-original','B2-positive'):
            all_new.extend(_record_candidates(comparator_candidates(method,epsilon),method))
    P.write(P.RESULTS/'theory_initialization.csv',theory)
    # Include all planned ids on resume, so a prior interrupted screen stage is
    # recomputed rather than treating a recorded candidate as already measured.
    planned=[r for r in P.read(P.RESULTS/'candidates.csv') if r['stage']=='seeds']
    return evaluate_candidates(planned,workers)


def neighborhoods(stage,epsilons,workers,max_local=24,max_boundary=12):
    P=_protocol();planned=[]
    recorded_memberships=P.read(P.RESULTS/'candidate_memberships.csv')
    for epsilon in epsilons:
        for method in METHODS:
            if any(r['method']==method and r['epsilon']==epsilon and r['stage']==stage
                   for r in recorded_memberships):continue
            rows=_rows_for(method,epsilon)
            centers=search_centers(rows,epsilon,2)
            lists=[joint_neighbors(center,epsilon,stage) for center in centers]
            points=[]
            # Round robin prevents the first center from consuming the budget.
            for i in range(max((len(x) for x in lists),default=0)):
                for candidates in lists:
                    if i<len(candidates):points.append(candidates[i])
            limit=max_local if stage=='local' else max_boundary
            if method!='PPPM-grid-PSD':limit=min(limit,8 if stage=='local' else 4)
            points=points[:limit]
            _record_candidates(points,method)
    planned=[r for r in P.read(P.RESULTS/'candidates.csv') if r['stage']==stage]
    return evaluate_candidates(planned,workers)


def validate(epsilons,workers):
    P=_protocol();wanted={};membership=[]
    for epsilon in epsilons:
        for method in METHODS:
            for row in shortlist_rows(_rows_for(method,epsilon),epsilon):
                wanted[row['id']]=row
                membership.append(dict(id=row['id'],method=method,epsilon=float(epsilon),
                    shortlist_reason='fast screen passes plus lowest measured screen error'))
    P.write(P.RESULTS/'validation_memberships.csv',membership)
    old=P.read(P.RESULTS/'validation.csv')
    counts={id_:len({r['config'] for r in old if r['id']==id_}) for id_ in wanted}
    pending=[row for id_,row in wanted.items() if counts.get(id_,0)!=9]
    if pending:
        P.log(f'fresh nine-configuration validation: {len(pending)} candidates; no PSD/bound hard filter')
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(P.validate_one,c):c for c in pending}
            for future in as_completed(futures):
                measured=future.result();replace_ids={r['id'] for r in measured}
                old=[r for r in old if r['id'] not in replace_ids]+measured
                P.write(P.RESULTS/'validation.csv',old)
                errors=[r.get('E_G') for r in measured if r.get('E_G') is not None]
                P.log(f'validate {measured[0]["id"]}: configs={len(measured)}, max_E_G={max(errors) if errors else None}')
    summary=P.summarize_validation(old)
    P.write(P.RESULTS/'validation_summary.csv',summary)
    byid={r['id']:r for r in summary};expanded=[]
    for member in membership:
        r=byid.get(member['id'])
        if r is None:continue
        row=dict(r,epsilon=member['epsilon'],shortlist_reason=member['shortlist_reason'])
        row['feasible']=nine_pass(r,member['epsilon'])
        row['selection_status']='empirical_nine_config_pass' if row['feasible'] else 'failed_or_not_met'
        expanded.append(row)
    P.write(P.RESULTS/'shortlist.csv',expanded)
    availability=[]
    for epsilon in epsilons:
        for method in METHODS:
            passing=[r for r in expanded if r['epsilon']==epsilon and r['method']==method and r['feasible']]
            availability.append(dict(method=method,epsilon=epsilon,feasible_shortlist_candidates=len(passing),
                status='available_for_fresh_serial_timing' if passing else 'N/A_no_nine_config_pass_within_budget'))
    P.write(P.RESULTS/'availability.csv',availability)
    return expanded


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('seeds','local','boundary','validate','all'))
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--max-local',type=int,default=24)
    parser.add_argument('--max-boundary',type=int,default=12)
    parser.add_argument('--epsilon',type=float,action='append')
    args=parser.parse_args()
    if not 1<=args.workers<=2:raise ValueError('Use one or two search workers')
    epsilons=tuple(args.epsilon or EPSILONS)
    for epsilon in epsilons:
        if epsilon not in EPSILONS:raise ValueError('This budget preregisters only 1e-3,1e-4,1e-5')
    P=_protocol()
    protocol=dict(epsilons=epsilons,xi_seeds=XI_SEEDS,eta_max=ETA_MAX,max_M=MAX_M,
        max_local=args.max_local,max_boundary=args.max_boundary,workers=args.workers,
        search_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        optional_epsilon_1e6='not run in this preregistered mandatory-three-tier budget',
        note='Local discrete tested optima only; no claim of global optimality. Historical errors seed but do not pass candidates.')
    preset_path=P.RESULTS/'search_protocol.json'
    if preset_path.exists() and json.loads(preset_path.read_text())!=json.loads(json.dumps(protocol)):
        raise RuntimeError('Search protocol changed: preserve existing results before changing the budget')
    P.save(preset_path,protocol)
    if args.stage in ('seeds','all'):seeds(epsilons,args.workers)
    if args.stage in ('local','all'):neighborhoods('local',epsilons,args.workers,args.max_local,args.max_boundary)
    if args.stage in ('boundary','all'):neighborhoods('boundary',epsilons,args.workers,args.max_local,args.max_boundary)
    if args.stage in ('validate','all'):validate(epsilons,args.workers)
    budget(P)
    P.log('Search stage complete; worker pools closed. Final same-batch serial timing is a separate step.')


if __name__=='__main__':main()
