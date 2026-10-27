"""Separate projection-active transitions from measured inactive convergence."""
import study as X
from study import np

def main():
    rows=X.read(X.RESULTS/'convergence.csv');local=[];inactive=[];floor=1e-11
    for method in X.METHODS:
        for family in X.E.FAMS:
            for p in (2,3,5,7,9):
                rr=[r for r in rows if r['method']==method and r['family']==family and r['p']==p]
                points=[]
                for M in sorted({r['M'] for r in rr}):
                    aa=[r for r in rr if r['M']==M]
                    points.append(dict(M=M,E=float(np.median([r['E_mesh'] for r in aa])),
                        active=any((r.get('negative_blocks') or 0)>0 for r in aa),
                        max_E_proj=max(r.get('E_proj') or 0 for r in aa)))
                for a,b in zip(points,points[1:]):
                    valid=min(a['E'],b['E'])>floor
                    regime='inactive' if not a['active'] and not b['active'] else 'active' if a['active'] and b['active'] else 'transition'
                    local.append(dict(method=method,family=family,p=p,Mlo=a['M'],Mhi=b['M'],
                        E_lo=a['E'],E_hi=b['E'],active_lo=a['active'],active_hi=b['active'],
                        local_order=float(np.log(a['E']/b['E'])/np.log(b['M']/a['M'])) if valid else None,
                        above_fit_floor=valid,floor=floor,regime=regime,
                        interpretation='finite-interval slope only; transitions are not asymptotic accuracy certificates'))
                if method!='PPPM-grid-PSD':continue
                vv=[v for v in points if not v['active'] and v['E']>floor][-4:]
                valid=len(vv)>=3 and all(vv[i+1]['E']<vv[i]['E'] for i in range(len(vv)-1))
                inactive.append(dict(method=method,family=family,p=p,n=len(vv),Mlo=vv[0]['M'] if vv else None,
                    Mhi=vv[-1]['M'] if vv else None,order=-np.polyfit(np.log([v['M'] for v in vv]),np.log([v['E'] for v in vv]),1)[0] if valid else None,
                    available=valid,floor=floor,status='measured projection-inactive interval' if valid else 'N/A: fewer than three decreasing inactive points above floor'))
    X.write(X.RESULTS/'mesh_local_slopes.csv',local)
    X.write(X.RESULTS/'mesh_inactive_rate_fits.csv',inactive)

if __name__=='__main__':main()
