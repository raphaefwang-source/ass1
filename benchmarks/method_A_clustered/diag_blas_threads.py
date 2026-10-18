#!/usr/bin/env python3
"""
Diagnostic for the run-1 import-order bug (numpy was imported before common.py set the thread variables, so both
OpenBLAS pools had 4 threads during the first calibration and screening). Times the same PPPM actions with the BLAS
pools limited to 1 and to 4 threads, interleaved, in one process. Result: results_psearch/run1_blas_pool4/blas_thread_ab.json.
The run-1 timings are NOT used for any reported result; this only documents whether they were affected.

    python3 diag_blas_threads.py
"""
import common as C
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from clustered_configurations import load
from cached_operator import CachedGammaH, mesh_for
from parameter_optimizer import shell_kc
from run_psearch_dense512 import x_inputs, action_fns, st, RESULTS_PSEARCH

CANDS = ((9, 0.855, 9.412788, 40, 4.886861), (10, 0.885, 9.705759, 36, 4.678719), (7, 0.66, 7.100047, 48, 6.341893))


def main():
    C.RESULTS = RESULTS_PSEARCH
    C.log_command()
    x, L = load("strong_N512_s1")
    x = np.mod(x, L)
    N = len(x)
    Xs = x_inputs(N)
    fns = {}
    for p, xi, kc, M, rc in CANDS:
        K = C.kernel(xi)
        op = CachedGammaH(x, L, K, rc, mesh_for(L, K, shell_kc(kc, L), M, p))
        for impl, fn in action_fns(op).items():
            fns[f"p{p}_{impl}"] = fn
    rng = np.random.default_rng([20261018, 77])
    t = {(k, n): [] for k in fns for n in (1, 4)}
    for w in range(3):
        for fn in fns.values():
            fn(Xs[w])
    import time
    for r in range(60):
        X = Xs[r % len(Xs)]
        order = [(k, n) for k in fns for n in (1, 4)]
        for o in rng.permutation(len(order)):
            k, n = order[o]
            with threadpool_limits(limits=n):
                t0 = time.perf_counter()
                fns[k](X)
                t[(k, n)].append(time.perf_counter() - t0)
    out = {f"{k}|blas{n}": st(np.array(v)) for (k, n), v in t.items()}
    ratio = {k: out[f"{k}|blas4"]["median_ms"] / out[f"{k}|blas1"]["median_ms"] for k in fns}
    C.write_json(RESULTS_PSEARCH / "run1_blas_pool4" / "blas_thread_ab.json",
                 dict(stats=out, ratio_blas4_over_blas1=ratio, rounds=60, provenance=C.provenance(blas=threadpool_info())))
    C.log(f"BLAS 4 / BLAS 1 action time ratio: {ratio}")


if __name__ == "__main__":
    main()
