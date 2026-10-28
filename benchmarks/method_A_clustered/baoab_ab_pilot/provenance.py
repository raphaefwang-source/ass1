"""Capture the reused C backend and native runtime without rebuilding it."""
from evidence import HERE,RESULTS,save,sha
from dynamics import THERMOSTAT
from compiled_edge import compile_backend
from pathlib import Path
import os
import platform
import psutil
from threadpoolctl import threadpool_info

def main():
    info=compile_backend()
    # This is a warm-cache provenance read, not a newly measured cold compile.
    if info['compiled_now']:raise RuntimeError('Expected reused PR9 compiled binary; inspect unexpected rebuild')
    source=THERMOSTAT/'optimized_pairou/edge_sweep.c'
    if sha(source)!=info['source_sha256'] or sha(info['library_path'])!=info['library_sha256']:
        raise RuntimeError('Compiled backend provenance mismatch')
    save(RESULTS/'backend_provenance.json',dict(backend=info,
        observation='post-timing read of unchanged PR9 warm compiled backend; no new cold-compile claim',
        platform=platform.platform(),source_path=str(source),
        native_thread_requests={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')},
        threadpoolctl=threadpool_info(),native_count_scope='Accelerate native pool count may be unobservable',
        load_average=os.getloadavg(),physical_memory_bytes=psutil.virtual_memory().total))
    print('Existing C library and source SHA verified; actual compile provenance saved')

if __name__=='__main__':main()
