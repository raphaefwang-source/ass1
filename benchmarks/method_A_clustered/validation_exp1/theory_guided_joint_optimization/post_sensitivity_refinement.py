#!/usr/bin/env python3
"""Use measured M/p costs to test coupled cutoff/xi moves after sensitivities."""
import itertools
import json
import experiment as X
from experiment import E,np,threadpool_limits
import run_optimization as R

def main():
    E.log_cmd(X.RESULTS);search=R.Search()
    # Preserve the previous common-timing winners so a later short timing
    # shortlist cannot discard them due to noise.
    measured=X.read(X.HERE/'best_candidates.csv')
    if not measured:raise RuntimeError('Run the first common finalist timing before this refinement')
    measured.sort(key=lambda r:r['t_complete_s'])
    keep={r['candidate_id'] for r in measured[:8]}
    for p in R.ORDERS:
        group=[r for r in measured if r['p']==p]
        if group:keep.add(group[0]['candidate_id'])
    X.jsave(X.RESULTS/'previous_measured_incumbents.json',sorted(keep))
    center=measured[0]
    pairs={3:(16,18,20,22,24,27),4:(12,14,16,18,20,22),5:(12,14,16,18,20),
           6:(10,12,14,16),7:(10,12,14)}
    requests=[]
    for p in R.ORDERS:
        group=[r for r in measured if r['p']==p]
        bestp=group[0] if group else center
        xis=sorted(set([round(center['xi']+d,4) for d in (-.1,0.,.1)]+[bestp['xi']]))
        for xi,M,dr,df in itertools.product(xis,pairs[p],(-.2,0.,.2),(-.3,0.,.3)):
            sr=round(center['sR']+dr,4);sf=round(center['sF_req']+df,4)
            if xi>0 and min(sr,sf)>1:
                requests.append(X.pars(xi,sr,sf,M,p))
    # Also give the equality-constrained control the same local opportunities.
    equal=next(r for r in measured if abs(r['sR']-r['sF_req'])<1e-10)
    for xi,s,M,p in itertools.product([round(equal['xi']+d,4) for d in (-.1,0.,.1)],
                                     [round(equal['sR']+d,4) for d in (-.2,-.1,0.,.1,.2)],
                                     [m for m in R.MESH if abs(m-equal['M'])<=4],R.ORDERS):
        requests.append(X.pars(xi,s,s,M,p))
    X.jsave(X.RESULTS/'post_sensitivity_plan.json',dict(center=center,equal_center=equal,
             mesh_pairs=pairs,n_requests=len(requests),reason='test whether lower p needs a larger M and wider cutoffs',
             provenance=X.provenance()))
    with threadpool_limits(limits=1):
        search.batch(requests,'post-sensitivity-coupled-refinement')
        search.refresh_times();search.validate(search.shortlist())
    X.log('post-sensitivity joint refinement complete')

if __name__=='__main__':main()
