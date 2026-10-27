"""Targeted OFFLINE stability audit; never replaces the primary batch/budget.

Revisit small-N audits with relative degree error >=1e-8 or unavailable auto
bounds. Explicitly force positive occupancy beyond the default work budget,
recording both its accuracy and extra cost instead of hiding unavailable rows.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_offdiag as R
np=R.np


def main():
    target=R.RESULTS/'smallN_stability_cases.csv'
    if target.exists():
        print('Preserved existing targeted stability audit; no samples overwritten')
        return
    allrows=R.read(R.RESULTS/'bounds.csv')
    unique={(r['parameter_id'],r['config']):r for r in allrows}
    candidates=[r for r in unique.values() if (r.get('audit_small_error_relative_true_correction_norm') or 0)>=1e-8
                or r.get('audit_small_stable_fallback_unavailable')]
    groups={}
    for r in candidates:groups.setdefault(r['parameter_id'],[]).append(r)
    output=[]
    for pid,rows in groups.items():
        mesh=R.S.table('PPPM-grid-PSD',rows[0]['L'],R.parameters(rows[0]))
        cb=R.CorrectionBounds(mesh,audit=False);ob=R.OffdiagBounds(cb)
        for row in rows:
            cfg=R.S.E.config(row['family'],int(row['seed']),row['rho'])
            picks=np.random.default_rng(1973+int(row['seed'])).choice(len(cfg['q']),4,replace=False)
            q=cfg['q'][picks]
            auto_start=time.perf_counter();auto,Dauto=ob.bounds(q,stable='auto')
            auto_seconds=time.perf_counter()-auto_start
            W=cb.pair_matrix(q);index=np.arange(len(q));W[index,index]=0.
            Ddirect=W.sum(axis=1);norm=R.S.norm(R.S.lap_full(W))
            forced_start=time.perf_counter();Dstable,profile=ob.stable_off_degree(q)
            forced_seconds=time.perf_counter()-forced_start
            stable_eig=np.linalg.eigvalsh(Dstable)
            Boff=float(2*stable_eig[:,-1].max())
            Bimproved=min(auto['B_simple'],auto['B_total'],Boff)
            err_auto=R.spectral_max(Dauto-Ddirect)
            err_stable=R.spectral_max(Dstable-Ddirect)
            record={k:row[k] for k in ('parameter_id','case_id','config','family','seed','M','p','rho','kappa','xi','sF')}
            record.update(audit_N=4,true_correction_norm=norm,
                primary_small_norm=row['audit_small_true_correction_norm'],
                primary_small_error_ratio=row.get('audit_small_error_relative_true_correction_norm'),
                auto_numerical_bound_available=auto['numerical_bound_available'],
                auto_stable_fallback_unavailable=auto['stable_fallback_unavailable'],
                auto_B_off=auto['B_off'],auto_B_improved=auto['B_improved'],
                auto_error_max_spectral=err_auto,auto_error_relative_true_norm=err_auto/norm if norm>0 else None,
                auto_setup_seconds=auto_seconds,forced_positive_setup_seconds=forced_seconds,
                forced_B_off=Boff,forced_B_improved=Bimproved,
                forced_min_eigenvalue=float(stable_eig[:,0].min()),
                forced_error_max_spectral=err_stable,
                forced_error_relative_true_norm=err_stable/norm if norm>0 else None,
                forced_strict_norm_violation_abs=max(norm-Bimproved,0.),
                auto_strict_norm_violation_abs=max(norm-auto['B_improved'],0.) if auto['B_improved'] is not None else None,
                default_query_budget=ob.max_stable_queries,
                forced_exceeds_default_query_budget=profile['stable_queries']>ob.max_stable_queries,
                primary_records_unchanged=True,**profile)
            output.append(record)
            print(json.dumps(dict(parameter_id=pid,config=row['config'],auto_available=auto['numerical_bound_available'],
                forced_relative_error=record['forced_error_relative_true_norm'],seconds=forced_seconds)),flush=True)
    R.write(target,output)
    R.save(R.RESULTS/'smallN_stability_summary.json',dict(
        selected_table_configurations=len(output),selection_relative_degree_error_threshold=1e-8,
        selection_role='Offline diagnostic only; never a mathematical bound floor or a production selection/rejection criterion',
        auto_unavailable=sum(not r['auto_numerical_bound_available'] for r in output),
        forced_exceeds_default_budget=sum(r['forced_exceeds_default_query_budget'] for r in output),
        max_forced_relative_error=max((r['forced_error_relative_true_norm'] or 0) for r in output),
        max_forced_strict_norm_violation_abs=max(r['forced_strict_norm_violation_abs'] for r in output),
        forced_setup_seconds=[r['forced_positive_setup_seconds'] for r in output],
        algorithm_sha256=hashlib.sha256((HERE/'offdiag_bounds.py').read_bytes()).hexdigest(),
        primary_results_modified=False,
        policy='Explicit extra-cost offline stable audit; the original 3 auto-unavailable rows remain unavailable in the primary batch. Float64 diagnostics, not interval certificates.'))


if __name__=='__main__':main()
