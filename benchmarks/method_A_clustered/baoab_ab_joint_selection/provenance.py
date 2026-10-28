"""After all timings: read the reused C backend and local native runtime."""
from joint_common import *
from compiled_edge import compile_backend
import platform
from threadpoolctl import threadpool_info


def main():
    if not (RESULTS/'heads_done.json').exists():raise RuntimeError('Wait for all formal timings')
    info=compile_backend()
    if info['compiled_now']:raise RuntimeError('Unexpected rebuild: inspect before making a warm-backend claim')
    source=THERMOSTAT/'optimized_pairou/edge_sweep.c'
    if sha(source)!=info['source_sha256'] or sha(info['library_path'])!=info['library_sha256']:
        raise RuntimeError('Backend source or library SHA mismatch')
    save(RESULTS/'backend_provenance.json',dict(backend=info,source_path=str(source),
        observation='warm-cache post-timing provenance read; no new cold-compile timing claim',
        platform=platform.platform(),native_thread_requests={key:os.environ.get(key) for key in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')},
        threadpoolctl=threadpool_info(),native_count_scope='Accelerate pool counts may be unobservable',
        load_average=os.getloadavg()))
    print('Verified unchanged reused C source/binary SHA and recorded native runtime')


if __name__=='__main__':main()
