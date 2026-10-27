"""IO and read-only PR8 bridge; all new evidence stays below this directory."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
RESULTS=HERE/'results';INPUTS=HERE/'inputs';LOGS=HERE/'logs'
BASE='bce73847f2968c27e1df5e2ff4be57393c4d2ddd'
SEED=2026102708
sys.path.insert(0,str(PARENT))
import protocol as P
import spatial as S
import ou_core as O
np=P.np

def save(path,data):
    P.save(Path(path) if Path(path).is_absolute() else RESULTS/path,data)

def read(path):
    return P.read(Path(path).resolve() if Path(path).is_absolute() else RESULTS/path)

def write(path,rows):
    P.write(Path(path).resolve() if Path(path).is_absolute() else RESULTS/path,rows)

def parent_rows(name):return P.read(PARENT/'results'/name)
def sha(path):return P.sha(path)
def array_sha(array):
    array=np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape,array.dtype.str)).encode()+array.tobytes()).hexdigest()
def log(message):P.log(message)
def make_system(meta):
    before=P.INPUTS
    try:
        P.INPUTS=INPUTS/'geometry'
        return S.make_system(N=int(meta['N']),family=meta['family'],seed=int(meta['seed']),epsilon=float(meta['epsilon']))
    finally:P.INPUTS=before
def baseline_input(meta):return PARENT/'inputs'/(meta['config']+'.npz')
def load_saved(meta):
    with np.load(baseline_input(meta),allow_pickle=False) as data:return {key:data[key].copy() for key in data.files}
def environment():
    data=P.environment()
    versions={}
    for name in ('numba','llvmlite'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    try:
        import numba
        numba_status=dict(available=True,version=numba.__version__)
    except Exception as exc:numba_status=dict(available=False,error=repr(exc))
    data.update(base_sha=BASE,global_seed=SEED,package_versions=versions,numba_status=numba_status,
                processor=platform.processor(),compiled_backend='C sequential scalar kernel via ctypes; no fast-math',
                compiler=subprocess.check_output(['cc','--version'],text=True).strip())
    return data
def load_snapshot(label):return P.load_snapshot(label)
