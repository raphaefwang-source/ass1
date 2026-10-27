"""Record native-library provenance after serial numerical timings finish."""
import os
import platform
import protocol as P


def main():
    keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
    pools=P.S.threadpool_info()
    P.save(P.RESULTS/'runtime_environment.json',dict(
        platform=platform.platform(),python=platform.python_version(),
        numpy=P.np.__version__,numpy_config=P.np.show_config(mode='dicts'),
        requested_threads=1,thread_environment={key:os.environ.get(key) for key in keys},
        enumerable_native_pools=pools,
        native_thread_count_independently_verified=bool(pools) and all(p['num_threads']==1 for p in pools),
        scope='Post-timing provenance under the identical import-time environment protocol; '
              'an empty pool list means native thread count was not independently enumerated. '
              'No extra performance samples or physical changes.'))


if __name__=='__main__':main()
