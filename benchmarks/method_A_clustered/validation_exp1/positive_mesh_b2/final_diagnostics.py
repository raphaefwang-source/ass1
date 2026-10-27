"""Final independent long-range limit and telescoping error decomposition."""
import json
import experiment as X
from experiment import E,np

def main():
    best=json.loads((X.RESULTS/'best_configuration.json').read_text())
    par={k:best[k] for k in ('xi','sR','sT','M','p','kappa','rho')}
    rows=[]
    with X.threadpool_limits(1):
        for cfg in X.configurations():
            G,nref=X.ref(cfg,.2)
            row,op,Gh,GLh=X.evaluate(cfg,par,structure=True,actions=True)
            GS=E.real_dense(op)
            GSfar=X.short_dense(cfg['family'],cfg['seed'],.75,.2,par['xi'],8.)
            K=E.kernel(.2,par['xi'],0.)
            GF=E.fourier_dense(cfg['q'],cfg['L'],K,2*par['xi']*7.)
            bigger=dict(par,sT=48.)
            fine=X.build(cfg,bigger,quad_order=16)
            _,GLfine=X.dense(fine)
            bigger8=X.build(cfg,bigger,quad_order=8)
            _,GLbigger=X.dense(bigger8)
            dR=GS-GSfar;dM=GLfine-GF;dT=GLh-GLbigger;dQ=GLbigger-GLfine
            residual=GSfar+GF-G
            row.update(E_R=X.norm(dR)/nref,E_interp=X.norm(dM)/nref,E_table_images=X.norm(dT)/nref,
                       E_table_quad=X.norm(dQ)/nref,far_split_residual=X.norm(residual)/nref,
                       E_mesh_including_table=X.norm(GLh-GF)/nref,
                       telescope_residual=X.norm((Gh-G)-(dR+dM+dT+dQ+residual))/nref)
            rows.append(row);X.write(X.RESULTS/'error_decomposition.csv',rows)
    X.log('final error decomposition complete')

if __name__=='__main__':main()
