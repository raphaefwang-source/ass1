#!/usr/bin/env python3
"""Offline eigendiagnostics for the final best projected table, no retiming.

Measured validation errors, particle eigenvalues and configuration setup times
are preserved.  All table diagnostic timings in the merged file are explicitly
prefixed offline_audit_; they are not production setup measurements.
"""
import hashlib
import json
from pathlib import Path
import time

import study as X
from grid_psd import ProjectedMesh

PARAM_KEYS = ('xi','sR','sF','sT','M','p','kappa','rho')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    best_path = X.RESULTS/'best_configuration.json'
    validation_path = X.RESULTS/'validation.csv'
    input_hashes = {str(p):sha(p) for p in (best_path,validation_path)}
    best = json.loads(best_path.read_text())['PPPM-grid-PSD']
    par = X.parameters(**{k:best[k] for k in PARAM_KEYS})
    cfg = X.E.config('uniform',1,par['rho'])
    original = X.table('PPPM-original',cfg['L'],par)
    projected = ProjectedMesh(original,audit_eigenvalues=True)
    diagnostics = X.diagnostics(projected)
    table_row = dict(method='PPPM-grid-PSD',id=best['id'],**par,L=cfg['L'],h=projected.h,
        diagnostic_source='fresh_offline_table_audit',
        timing_scope='Every t_* in this row is offline audit/table construction, not production setup.',
        **diagnostics)
    X.write(X.RESULTS/'best_table_diagnostics.csv',[table_row])

    selected = [r for r in X.read(validation_path) if r['id']==best['id']]
    assert len(selected)==9 and len({r['config'] for r in selected})==9
    merged = []
    for measured in selected:
        row = dict(measured)
        protected = {k:measured[k] for k in measured if
                     k.startswith(('E_','lam_min_')) or k=='t_setup_s'}
        for key,value in diagnostics.items():
            row[f'offline_audit_{key}' if key.startswith('t_') else key] = value
        row['diagnostic_source'] = 'fresh_offline_table_audit'
        row['offline_audit_timing_scope'] = 'offline_audit_t_* are audit-only; original t_setup_s is preserved.'
        assert all(row[k]==v for k,v in protected.items())
        assert row['diagnostic_eigenvalues_evaluated'] is True
        assert row['min_eig_after'] is not None
        merged.append(row)
    X.write(X.RESULTS/'best_projection_diagnostics.csv',merged)
    assert all(sha(path)==digest for path,digest in input_hashes.items())
    X.jsave(X.RESULTS/'best_table_audit_provenance.json',X.E.provenance(
        stage='best_table_offline_audit',workers=1,audit_threads=1,rows=1,merged_validation_rows=9,
        best_id=best['id'],parameters=par,input_sha256=input_hashes,
        source_sha256={p.name:sha(p) for p in (Path(__file__),X.HERE/'grid_psd.py',X.HERE/'study.py')},
        elapsed_s=time.perf_counter()-start,
        note='Fresh original mesh plus audit_eigenvalues=True projection. Offline timings explicitly labeled. '
             'No changes to best_configuration.json, validation.csv, or any timing artifact.'))
    X.log(f'best offline table audit complete: min_eig_before={diagnostics["min_eig_before"]:.9g}, '
          f'min_eig_after={diagnostics["min_eig_after"]:.9g}, '
          f'negative_blocks={diagnostics["negative_blocks"]}, 9 measured rows preserved')


if __name__ == '__main__':
    with X.threadpool_limits(1):
        main()
