"""Supplemental uncensored dense-action stopping audit; never a speed benchmark.

Run after accuracy: python3 -B uncensored_stopping.py
This leaves rank_errors.csv and every selected benchmark rank unchanged.
"""
from collections import defaultdict
from time import perf_counter
import protocol as P
from ou_core import lanczos_action
from stopping_audit import _decision, counts
np = P.np


TOLERANCES = (1e-6,1e-8,1e-10)
TESTED_RANKS = (1,2,3,4,6,8,12)


def summarize(rows):
    """Count both correct and false decisions, including insufficient-rank rows."""
    comparable = [r for r in rows if r['step_change'] is not None]
    fails = [r for r in comparable if r['error'] > r['tolerance']]
    return dict(counts=counts(rows), comparable_rows=len(comparable),
                false_positive_opportunities=len(fails),
                observed_false_positive_count=sum(r['decision']=='false_positive' for r in comparable),
                assessment='empirical only; observed false positives disprove certification, absence does not prove it')


def main():
    if not (P.RESULTS/'accuracy_done.json').exists():
        raise RuntimeError('Accuracy evidence must finish before this supplemental audit')
    start = perf_counter()
    original_rank_hash = P.sha(P.RESULTS/'rank_errors.csv')
    structures = P.read('structure.csv')
    rows, both = [], []
    input_hashes = {}
    for N in (2,8,16,64,128,256):
        meta = next(r for r in structures if int(r['N'])==N and r['family']=='uniform'
                    and int(r['seed'])==2 and r['epsilon']==1e-4)
        config = meta['config']
        P.log(f'uncensored dense-action stopping audit {config}')
        filename = P.INPUTS/(config+'.npz')
        input_hashes[config] = P.sha(filename)
        with np.load(filename) as saved:
            G,GL = saved['G'],saved['GL']
            probes = dict(saved_random=P.project(saved['random']),saved_z=P.project(saved['z']))
        full = P.DenseOU(G)
        ranks = sorted(set(min(3*N-3,r) for r in TESTED_RANKS))
        for operator,matrix in [('full',G),('long',GL)]:
            reference = full if operator=='full' else P.DenseOU(matrix)
            for alpha in (1.,2.):
                dt = alpha/full.lmax
                R,F,_ = reference.matrices(dt)
                for kind,target in [('damp',R),('noise',F)]:
                    previous = {}
                    for rank in ranks:
                        step_rows = defaultdict(list)
                        for probe,v in probes.items():
                            out,diag = lanczos_action(lambda x:matrix@x,v,rank,dt,kind)
                            exact = target@v
                            error = float(np.linalg.norm(out-exact)/max(np.linalg.norm(exact),1e-300))
                            change = (float(np.linalg.norm(out-previous[probe])/max(np.linalg.norm(out),1e-300))
                                      if probe in previous else None)
                            previous[probe] = out
                            for tolerance in TOLERANCES:
                                row = dict(config=config,N=N,family='uniform',seed=2,epsilon=1e-4,
                                           operator=operator,alpha=alpha,alpha_definition='dt*lambda_max(full)',
                                           dt=dt,kind=kind,probe=probe,rank=rank,rank_used=diag['rank_used'],
                                           error=error,step_change=change,tolerance=tolerance,
                                           decision=_decision(change,error,tolerance),
                                           ritz_min_raw=diag['ritz_min_raw'],ritz_max_raw=diag['ritz_max_raw'],
                                           negative_ritz_count=diag['negative_ritz_count'],
                                           clipped_negative_count=diag['clipped_negative_count'],
                                           action_source='saved dense same-matrix multiplication; no FFT/timing benchmark',
                                           selection_role='supplemental uncensored diagnostic audit; no selected ranks changed')
                                rows.append(row)
                                step_rows[tolerance].append(row)
                        for tolerance,members in step_rows.items():
                            changes = [r['step_change'] for r in members]
                            change = max(changes) if all(c is not None for c in changes) else None
                            error = max(r['error'] for r in members)
                            both.append(dict(config=config,N=N,operator=operator,alpha=alpha,dt=dt,
                                             kind=kind,rank=rank,tolerance=tolerance,probe_count=2,
                                             error=error,step_change=change,
                                             decision=_decision(change,error,tolerance)))
        P.write('uncensored_stopping.csv',rows)
        P.write('uncensored_stopping_both_probes.csv',both)
    summary = dict(source='saved input G and GL; independent DenseOU reference',
                   timing_scope='diagnostic only; dense actions are not additional FFT benchmark measurements',
                   source_rank_errors_sha256_before=original_rank_hash,
                   source_rank_errors_sha256_after=P.sha(P.RESULTS/'rank_errors.csv'),
                   input_sha256=input_hashes,N=[2,8,16,64,128,256],alphas=[1.,2.],
                   alpha_definition='dt*lambda_max(full); long uses the same physical dt',
                   probes=['saved_random','saved_z'],ranks_requested=TESTED_RANKS,
                   ranks_note='cap each rank at 3N-3 and deduplicate; no early stopping',
                   elapsed_s=perf_counter()-start,
                   per_direction={str(t):summarize([r for r in rows if r['tolerance']==t]) for t in TOLERANCES},
                   both_registered_probes={str(t):summarize([r for r in both if r['tolerance']==t]) for t in TOLERANCES},
                   false_positive_examples=[r for r in rows if r['decision']=='false_positive'][:30],
                   limitations=[
                       'Finite observed configurations, time scales, ranks, and directions do not certify an error bound.',
                       'This supplemental sweep does not change the production accuracy selection or its tolerance.',
                       'Successive tested ranks have nonuniform increments.',
                       'Noise action error is vector dependent and is not a covariance-operator error bound.',
                       'A small successive-rank change alone cannot be used as a general stopping certificate.'
                   ])
    if summary['source_rank_errors_sha256_before'] != summary['source_rank_errors_sha256_after']:
        raise RuntimeError('Production rank evidence changed during supplemental audit')
    P.save(P.RESULTS/'uncensored_stopping_summary.json',summary)
    print({key:summary[key] for key in ('elapsed_s','per_direction','both_registered_probes')},flush=True)


if __name__ == '__main__':
    main()
