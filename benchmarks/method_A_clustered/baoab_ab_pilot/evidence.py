"""Local-only pilot evidence, with atomic writes and content hashes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE/'results'
INPUTS = HERE/'inputs'
LOGS = HERE/'logs'
BASE = 'd6a8d208bcdc30731079715dc4c8e925b4fb2f8b'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,default=convert)+'\n')
    tmp.replace(path)

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)

def read_csv(path):
    with Path(path).open(newline='') as f: rows=list(csv.DictReader(f))
    for row in rows:
        for k,v in row.items():
            if v in ('True','False'):row[k]=v=='True'
            elif v=='':row[k]=None
            else:
                try:row[k]=float(v)
                except ValueError:pass
    return rows
