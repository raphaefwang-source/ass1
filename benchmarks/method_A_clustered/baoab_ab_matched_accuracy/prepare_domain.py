"""Hash-check and reuse PR10 starts and actual saved midpoint geometries."""
from matched_common import *
from model import lj_force

def main():
    if (RESULTS/'domain.json').exists():raise FileExistsError('Preserve completed domain')
    records=[]
    def add(label,q,p,metadata):
        path=INPUTS/f'domain_{label}.npz'
        np.savez_compressed(path,q=q,p=p,L=PARAMS['L'])
        records.append(dict(label=label,file=str(path.relative_to(HERE)),file_sha256=sha(path),
            q_sha256=array_sha(q),p_sha256=array_sha(p),**metadata))
    for i in range(3):
        source=LEGACY/'inputs'/f'initial_{i}.npz';meta=js(LEGACY/'results'/f'initial_{i}.json')
        if sha(source)!=meta['file_sha256']:raise RuntimeError('PR10 initial hash mismatch')
        with np.load(source,allow_pickle=False) as d:q,p=d['q'].copy(),d['p'].copy()
        add(f'initial_{i}',q,p,dict(kind='initial',initial_index=i,source=str(source.relative_to(REPO)),
            source_sha256=sha(source),preparation_seed=meta['seed'],preparation_duration=meta['duration']))
    minimum=None;scan=[]
    for i in range(3):
        for method in ('A','B'):
            label=f'production_{i}_{method}';source=LEGACY/'inputs'/f'{label}.npz'
            meta=js(LEGACY/'results'/f'{label}.json')
            if sha(source)!=meta['file_sha256']:raise RuntimeError('PR10 production hash mismatch')
            with np.load(source,allow_pickle=False) as d:
                for j,qmid in enumerate(d['q_mid']):
                    separation=lj_force(qmid,PARAMS['L'])[2]
                    scan.append(dict(run=label,initial_index=i,method=method,step=j+1,min_pair_distance=separation,
                        q_mid_sha256=array_sha(qmid)))
                    if minimum is None or separation<minimum['separation']:
                        ph=d['p'][j]+.0025*lj_force(np.mod(d['q_unwrapped'][j],PARAMS['L']),PARAMS['L'])[0]
                        minimum=dict(separation=separation,q=qmid.copy(),p=ph.copy(),run=label,step=j+1,
                            initial_index=i,method=method,source=str(source.relative_to(REPO)),source_sha256=sha(source))
                ph=d['p'][-2]+.0025*lj_force(np.mod(d['q_unwrapped'][-2],PARAMS['L']),PARAMS['L'])[0]
                add(f'visited_{i}_{method}_200',d['q_mid'][-1].copy(),ph,
                    dict(kind='visited_midpoint',initial_index=i,method=method,step=200,
                         source=str(source.relative_to(REPO)),source_sha256=sha(source),
                         min_pair_distance=lj_force(d['q_mid'][-1],PARAMS['L'])[2]))
    if not any(r['q_sha256']==array_sha(minimum['q']) for r in records):
        add('visited_global_min_distance',minimum['q'],minimum['p'],
            {k:v for k,v in minimum.items() if k not in ('q','p','separation')}|
            dict(kind='visited_global_min_midpoint_distance',min_pair_distance=minimum['separation']))
    write_csv(RESULTS/'prior_midpoint_distance_scan.csv',scan)
    save(RESULTS/'domain.json',dict(base_sha=BASE,params=PARAMS,states=records,
        scanned_actual_midpoints=len(scan),global_min_midpoint_distance=minimum['separation'],
        extrema_scope='Minimum separation over all 1200 saved PR10 midpoints; lambda_max extrema will be measured over the dense validation domain',
        preparation_reused=True,canonical_preparation_steps_run=0))
    print('Saved',len(records),'hash-checked domain states; scanned',len(scan),'actual midpoints; minr',minimum['separation'])

if __name__=='__main__':main()
