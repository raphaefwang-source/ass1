#!/usr/bin/env python3
"""Cross-validate each method at the other methods' exact winning parameters.

The original-at-grid-winner case is processed first.  Search's short timing
heuristic is retained by screen(); no final/common timing files are touched.
Every legal candidate is validated on all nine configurations, independently
of its screen pass/fail decision.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import study as X
import run_search as S

PARAM_KEYS = ('xi','sR','sF','sT','M','p','kappa','rho')
STAGE = 'cross_winner'


def update_budget():
    rows = X.read(X.RESULTS/'search.csv')
    stages = sorted({r['stage'] for r in rows} | {STAGE})
    X.jsave(X.RESULTS/'search_budget.json',{
        method:dict(independent_candidates=sum(r['method']==method for r in rows),
                    failures=sum(r['method']==method and r['status']!='ok' for r in rows),
                    stages={stage:sum(r['method']==method and r['stage']==stage for r in rows) for stage in stages})
        for method in X.METHODS})


def summarize(manifest,snapshot_sha,start):
    search = {r['id']:r for r in X.read(X.RESULTS/'search.csv')}
    validation = {r['id']:r for r in X.read(X.RESULTS/'validation_summary.csv')}
    rows = []
    for cand in manifest:
        screened = search.get(cand['id'],{})
        validated = validation.get(cand['id'],{})
        row = dict(source_method=cand['source_method'],method=cand['method'],id=cand['id'],
                   source_winner_id=cand['source_winner_id'],**{k:cand[k] for k in PARAM_KEYS},
                   search_preexisting=cand['search_preexisting'],
                   screen_stage=screened.get('stage'),screen_E_G=screened.get('E_G'),
                   t_screen_s=screened.get('t_screen_s'),screen_pass=screened.get('screen_pass'),
                   max_E_G=validated.get('max_E_G'),max_E_GX=validated.get('max_E_GX'),
                   max_E_proj=validated.get('max_E_proj'),n_configs=validated.get('n_configs',0),
                   status=validated.get('status','pending'),feasible=validated.get('feasible',False),
                   source_winner_median_matvec_s=cand['source_winner_median_matvec_s'])
        rows.append(row)
    X.write(X.RESULTS/'cross_winner_summary.csv',rows)
    X.jsave(X.RESULTS/'cross_winner_summary.json',dict(
        stage=STAGE,source_best_configuration_sha256=snapshot_sha,
        elapsed_s=time.perf_counter()-start,rows=rows,
        note='Exact eight-parameter cross-over; all nine configurations validated even when the screen fails. '
             't_screen_s is only the existing search heuristic; no new final/common timing was run.'))
    return rows


def main():
    archived = X.RESULTS/'primary_timing'/'best_configuration.json'
    path = archived if archived.exists() else X.RESULTS/'best_configuration.json'
    source_text = path.read_bytes()
    best = json.loads(source_text)
    source_sha = hashlib.sha256(source_text).hexdigest()
    snapshot_path = X.RESULTS/'cross_winner_source_snapshot.json'
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        if snapshot['source_sha256'] != source_sha:
            raise RuntimeError('Winning parameters changed since this cross-validation snapshot; preserve prior results')
    else:
        X.jsave(snapshot_path,dict(source_path=str(path),source_sha256=source_sha,best=best))
    existing = {r['id'] for r in X.read(X.RESULTS/'search.csv')}
    pairs = [(method,source) for method in X.METHODS for source in X.METHODS if method != source]
    pairs.sort(key=lambda z:0 if z==('PPPM-original','PPPM-grid-PSD') else 1)
    manifest = []
    for method,source in pairs:
        par = X.parameters(**{k:best[source][k] for k in PARAM_KEYS})
        ident = X.ident(method,par)
        manifest.append(dict(method=method,source_method=source,id=ident,**par,
                             source_winner_id=best[source]['id'],search_preexisting=ident in existing,
                             source_winner_median_matvec_s=best[source].get('median_matvec_s')))
    start = time.perf_counter()
    summarize(manifest,source_sha,start)
    for cand in manifest:
        par = X.parameters(**{k:cand[k] for k in PARAM_KEYS})
        if not cand['search_preexisting']:
            S.screen(cand['method'],[par],STAGE)
        # No screen_pass filter: this is the exact requested cross candidate.
        S.validate([dict(method=cand['method'],id=cand['id'],**par)])
        update_budget()
        rows = summarize(manifest,source_sha,start)
        result = next(r for r in rows if r['id']==cand['id'])
        if result['n_configs'] != 9:
            raise AssertionError(f'Incomplete cross validation: {result}')
        X.log(f'cross winner {cand["method"]} at {cand["source_method"]}: '
              f'max_E_G={result["max_E_G"]}, feasible={result["feasible"]}, '
              f't_screen_s={result["t_screen_s"]}')
    update_budget()
    rows = summarize(manifest,source_sha,start)
    X.log(f'cross winner COMPLETE: {len(rows)} exact cross candidates, '
          f'{sum(r["feasible"] for r in rows)} feasible, elapsed={time.perf_counter()-start:.1f}s; no final timing run')


if __name__ == '__main__':
    with X.threadpool_limits(1):
        main()
