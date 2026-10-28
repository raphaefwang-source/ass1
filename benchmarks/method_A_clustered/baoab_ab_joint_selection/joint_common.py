"""Local joint-selection protocol; historical artifacts are readonly sources."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
MATCHED=HERE.parent/'baoab_ab_matched_accuracy';PILOT=HERE.parent/'baoab_ab_pilot'
THERMOSTAT=HERE.parent/'thermostat_ab_local'
RESULTS=HERE/'results';INPUTS=HERE/'inputs';LOGS=HERE/'logs'
BASE='891994a02b611833848839b5de9880f9dabc7a1d'
BASE_PARAMS=dict(N=256,rho=.75,L=(256/.75)**(1/3),kappa=.7,c=float(.5*1.3*np.exp(.7*1.3)),xi=.65,sR=4.7,sF=4.7,M=40,p=10)
PARAMS=BASE_PARAMS
TIERS=(1e-4,1e-6)
DT=.005
for path in (MATCHED,PILOT,THERMOSTAT,THERMOSTAT/'optimized_pairou'):
    if str(path) not in sys.path:sys.path.append(str(path))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()
def js(path):return json.loads(Path(path).read_text())
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,default=encode)+'\n');temp.replace(path)
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row));temp=path.with_suffix('.csv.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    temp.replace(path)
def read_csv(path):
    with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        for k,v in row.items():
            if v=='':row[k]=None
            elif v in ('True','False'):row[k]=v=='True'
            else:
                try:row[k]=float(v)
                except ValueError:pass
    return rows
def load_states():
    old=js(MATCHED/'results/domain.json');states=[]
    for r in old['states']:
        path=MATCHED/r['file']
        if sha(path)!=r['file_sha256']:raise RuntimeError('PR11 domain source hash mismatch')
        with np.load(path,allow_pickle=False) as d:q,p=d['q'].copy(),d['p'].copy()
        states.append(dict(label=r['label'],q=q,p=p,metadata=dict(r,source=str(path.relative_to(REPO)),source_sha256=sha(path))))
    return states
def certification_states():
    names={'initial_0','initial_1','initial_2','visited_global_min_distance','visited_1_B_200'}
    return [s for s in load_states() if s['label'] in names]
def register_domain():
    states=load_states()
    save(RESULTS/'domain.json',dict(states=[dict(s['metadata'],label=s['label'],q_sha256=array_sha(s['q']),p_sha256=array_sha(s['p'])) for s in states],
        initial_full_validation_labels=[s['label'] for s in certification_states()],
        physical=BASE_PARAMS,canonical_preparation_reused=True,canonical_preparation_steps_run=0,
        required_tiers=list(TIERS),optional_stress_2e8_executed=False))
    return states
