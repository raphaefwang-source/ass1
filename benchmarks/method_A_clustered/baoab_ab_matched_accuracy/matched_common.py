"""Protocol and local evidence for the independent PR10 follow-up."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
LEGACY=HERE.parent/'baoab_ab_pilot'
THERMOSTAT=HERE.parent/'thermostat_ab_local'
BASE='84174ae0d11e3f5826cda000c688d147a1b56241'
RESULTS=HERE/'results';INPUTS=HERE/'inputs';LOGS=HERE/'logs'
for path in (LEGACY,THERMOSTAT,THERMOSTAT/'optimized_pairou'):
    if str(path) not in sys.path:sys.path.append(str(path))
PARAMS=dict(N=256,rho=.75,L=(256/.75)**(1/3),kappa=.7,
    c=float(.5*1.3*np.exp(.7*1.3)),xi=.65,sR=4.7,sF=4.7,M=40,p=10)
TARGETS=(1e-4,1e-6,2e-8)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def array_sha(x):
    x=np.ascontiguousarray(x)
    return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,default=encode)+'\n');tmp.replace(path)

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    tmp.replace(path)

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

def js(path):return json.loads(Path(path).read_text())

def load_states():
    records=js(RESULTS/'domain.json')['states'];out=[]
    for r in records:
        path=HERE/r['file']
        if sha(path)!=r['file_sha256']:raise RuntimeError('Saved domain file hash mismatch')
        with np.load(path,allow_pickle=False) as d:
            out.append(dict(label=r['label'],q=d['q'].copy(),p=d['p'].copy(),metadata=r))
    return out
