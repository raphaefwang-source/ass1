"""Construction convergence and preservation audit, independent of timing."""
import hashlib
import json
import experiment as X
from experiment import np,E,C
from positive_mesh import PositiveMesh,table_tail_bound

def main():
    rows=[]
    with X.threadpool_limits(1):
        for kap,xi,M in ((.2,.4,32),(.2,1.25,16),(.7,.85,40),(2.,.85,24)):
            L=(256/.75)**(1/3);K=E.kernel(kap,xi,0.)
            m=PositiveMesh(L,K,M,3,36.,quad_order=8)
            fine=PositiveMesh(L,K,M,3,36.,quad_order=16)
            delta=float(np.max(np.sum(np.abs(m.table-fine.table),axis=0)))
            ix=np.array([[0,3,4],[3,1,5],[4,5,2]])
            tt=np.moveaxis(m.table,0,-1)[...,ix]
            ff=np.moveaxis(m.Gr,0,-1)[...,ix]
            eigmin=float(np.linalg.eigvalsh(tt).min())
            fmin=float(np.linalg.eigvalsh(ff).min())
            A,_=K.AB(np.array([0.]))
            nimg=(2*np.floor(m.R/L)+3)**3
            blo=K.amp*nimg*3*m.R**2*4*m.s_lower**1.5/(3*np.sqrt(np.pi))*np.exp(-kap**2/(4*m.s_lower))
            rows.append(dict(kappa=kap,xi=xi,M=M,n_quad=len(m.qw),table_min_eig=eigmin,fft_min_eig=fmin,
                             max_component_sum_quad8_vs16=delta,omitted_small_scale_bound=blo,
                             omitted_image_bound=table_tail_bound(L,kap,m.R),
                             zero_mode_rel_error=float(np.max(np.abs(m.zero_mode[:3]*m.h**3/A[0]-1)))))
    X.write(X.RESULTS/'construction_audit.csv',rows)
    hist=json.loads((X.RESULTS/'historical_artifact_hashes.json').read_text())
    modified=[p for p,h in hist.items() if not (C.REPO/p).exists() or hashlib.sha256((C.REPO/p).read_bytes()).hexdigest()!=h]
    assert not modified,modified
    X.jsave(X.RESULTS/'audit_checks.json',dict(historical_files_checked=len(hist),historical_modified=modified,
                                             construction_rows=len(rows),structural_proof='proof.md',
                                             eigenvalues_are_regression_only=True,
                                             provenance=E.provenance()))
    X.log(f'audit: {len(hist)} historical files unchanged; construction checked')

if __name__=='__main__':main()
