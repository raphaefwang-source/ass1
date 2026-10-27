"""Join parameter-only supplemental table diagnostics into new-method CSVs."""
import study as X
KEYS=('rho','kappa','xi','sF','M','p')

def key(r):return tuple(round(float(r[k]),10) for k in KEYS)

def main():
    diag={key(r):r for r in X.read(X.RESULTS/'table_diagnostics.csv')}
    joined={}
    for filename in ('sweeps.csv','convergence.csv'):
        rows=X.read(X.RESULTS/filename);count=0
        for r in rows:
            if r['method']!='B2-positive' and r.get('sF') is not None:
                _,kc,_=X.E.cutoffs(r,r['L'])
                r.update(kc=kc,sF_eff=kc/(2*r['xi']))
            if r['method']!='PPPM-grid-PSD' or key(r) not in diag:continue
            d=diag[key(r)]
            for k,v in d.items():
                if k not in KEYS and (k.startswith(('min_eig','negative','significant','fraction','correction','inversion','zero_','new_fourier','t_')) or k in ('total_blocks','active_correction_offsets')):
                    r[k]=v
            r['min_raw_eig']=d['min_eig_before']
            r['neg_fraction']=d['fraction_negative_blocks']
            r['max_correction']=d['correction_max_spectral']
            count+=1
        if rows:X.write(X.RESULTS/filename,rows)
        joined[filename]=count
    X.jsave(X.RESULTS/'table_diagnostic_join.json',joined)

if __name__=='__main__':main()
