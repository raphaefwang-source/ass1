#!/usr/bin/env python3
"""Controlled supplementary xi scan: fixed mesh and all nine configurations.

This is a preregistered parameter scan, not a selection from search candidates.
Only xi changes; sR/sF are fixed dimensionless cutoffs, so rc and kc change as
their definitions require.  All writes stay in this new comparison directory.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import time

import study as X
from study import E

XI = (.35,.6,.85,1.,1.25)
FIXED = dict(N=256,rho=.75,kappa=.2,c=1.,M=20,p=5,sR=3.,sF=3.,sT=36.)
PARTS = X.RESULTS/'xi_sweep_parts'


def _key(row):
    return row['method'],float(row['xi'])


def worker(job):
    family,seed = job
    cfg = E.config(family,seed,FIXED['rho'])
    path = PARTS/f'{family}_s{seed}.csv'
    rows = X.read(path)
    done = {_key(r) for r in rows}
    with X.threadpool_limits(1):
        for xi in XI:
            par = X.parameters(xi=xi,**{k:FIXED[k] for k in ('rho','kappa','M','p','sR','sF','sT')})
            for method in X.METHODS:
                if (method,xi) in done:
                    continue
                row,_,_,_ = X.measure(method,cfg,par,structure=True,actions=True,mesh_only=False)
                assert row['N'] == FIXED['N'] and all(math.isfinite(row[k]) for k in ('E_G','E_GX'))
                rows.append(dict(sweep='xi',setting='controlled_fixed_mesh',value=xi,
                                 source='new_controlled_xi_scan',amplitude_c=FIXED['c'],**row))
                done.add((method,xi))
                X.write(path,rows)
                X.log(f'xi scan {family} s{seed}: xi={xi:g} {method}, E_G={row["E_G"]:.6g}, {len(rows)}/15')
    assert len(rows) == 15 and len(done) == 15
    return rows


def main(workers=2):
    if not 1 <= workers <= 2:
        raise ValueError('This supplementary scan allows at most two worker processes')
    PARTS.mkdir(exist_ok=True)
    preset = dict(experiment='Controlled supplementary xi scan',xi=list(XI),fixed=FIXED,
                  families=list(E.FAMS),seeds=list(E.SEEDS),methods=list(X.METHODS),
                  expected_rows=135,structure=True,actions=True,mesh_only=False,
                  note='All three methods share fixed M20,p5,sR3,sF3; B2 sT36. No search or parameter retuning.')
    preset_path = X.RESULTS/'xi_sweep_preset.json'
    if preset_path.exists() and json.loads(preset_path.read_text()) != preset:
        raise RuntimeError('Existing xi scan preset differs; preserve it and use a separate output directory')
    if not preset_path.exists():
        X.jsave(preset_path,preset)
    start = time.perf_counter()
    rows = []
    X.log(f'controlled xi sweep: {len(XI)} xi, 3 methods, 9 configurations, workers={workers}')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(worker,(family,seed)) for family in E.FAMS for seed in E.SEEDS]
        for future in as_completed(jobs):
            rows.extend(future.result())
    rows.sort(key=lambda r:(r['family'],r['seed'],r['xi'],r['method']))
    assert len(rows) == 135 and len({(r['family'],r['seed'],*_key(r)) for r in rows}) == 135
    X.write(X.RESULTS/'xi_sweep.csv',rows)
    provenance = E.provenance(workers=workers,rows=len(rows),elapsed_s=time.perf_counter()-start,
        preset_sha256=hashlib.sha256(preset_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Controlled xi supplementary scan; historical results read-only; all executor workers joined before final output.')
    X.jsave(X.RESULTS/'xi_sweep_provenance.json',provenance)
    X.log(f'controlled xi sweep complete: {len(rows)} rows, elapsed {time.perf_counter()-start:.1f}s; workers exited')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    main(args.workers)
