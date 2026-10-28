"""Post-timing read of the unchanged PR9 compiled backend and native runtime."""
from matched_common import *
from compiled_edge import compile_backend
import platform
from threadpoolctl import threadpool_info

def main():
    info=compile_backend()
    if info['compiled_now']:raise RuntimeError('Inspect unexpected backend rebuild; expected unchanged warm PR9 binary')
    source=THERMOSTAT/'optimized_pairou/edge_sweep.c'
    if sha(source)!=info['source_sha256'] or sha(info['library_path'])!=info['library_sha256']:
        raise RuntimeError('Compiled binary/source provenance mismatch')
    save(RESULTS/'backend_provenance.json',dict(backend=info,source_path=str(source),
        observation='warm-cache post-timing provenance read; no new cold-compile timing claim',
        platform=platform.platform(),native_thread_requests={k:os.environ.get(k) for k in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')},
        threadpoolctl=threadpool_info(),native_count_scope='Accelerate pool counts may be unobservable',
        load_average=os.getloadavg()))
    print('Actual reused C binary/source SHA verified')

if __name__=='__main__':main()
