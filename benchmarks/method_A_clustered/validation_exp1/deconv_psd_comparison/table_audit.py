#!/usr/bin/env python3
"""Serial, resumable parameter-table diagnostics and independent small-N audit.

Only new-directory artifacts are written. No references or old results change.
Run: python3 .../deconv_psd_comparison/table_audit.py
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from grid_psd import ProjectedMesh, np, Mesh
import ewald_exp_lib as E
import common as C
from threadpoolctl import threadpool_limits, threadpool_info
sys.path.insert(0,str(HERE.parent/'accuracy_sweeps'))
from run_accuracy_sweeps import sweep_points

RESULTS = HERE/'results'


def ident(p):
    return '_'.join(f'{k}{p[k]:.12g}' for k in ('rho','kappa','xi','sF','M','p'))


def read(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def representative(p):
    base = p['kappa'] == .7 and p['rho'] == .75 and p['xi'] == .85
    return (base and p['M'] in (24,25,40,80) and
            ((p['sF'] == 6. and p['p'] == 9) or (p['sF'] == 4. and p['p'] == 7))) or (
        p['kappa'] == 2. or p['rho'] == .3)


def direct_checks(mesh):
    """N3 literal stencil contraction vs FFT, basis, degree, translation."""
    start = time.perf_counter()
    q = np.random.default_rng(93812).uniform(0,mesh.L,(3,3))
    W = mesh.direct_pair_matrix(q)
    fast,_ = mesh.pair_matrix(q)
    idx,wt = mesh.flat(q)
    basis = np.empty((3,3,3,3))
    for j in range(3):
        for b in range(3):
            x = np.zeros((3,3)); x[j,b] = 1.
            basis[:,j,:,b] = mesh.apply(idx,wt,x)
    degree = mesh.degree(idx,wt)
    G = E.lap_full(W)
    v = np.random.default_rng(1729).normal(size=(3,3))
    graph = mesh.graph_action(idx,wt,v,degree)
    scale = max(float(np.abs(W).max()),np.finfo(float).tiny)
    graph_scale = max(float(np.linalg.norm(G)),np.finfo(float).tiny)
    trans = max(float(np.linalg.norm(mesh.graph_action(idx,wt,np.tile(e,(3,1)),degree)))
                for e in np.eye(3))
    rec = dict(direct_checked=True,direct_N=3,
        direct_vs_sparse_max_abs=float(np.abs(W-fast).max()),
        direct_vs_fft_basis_max_abs=float(np.abs(W-basis).max()),
        direct_vs_fft_basis_max_relative=float(np.abs(W-basis).max()/scale),
        direct_degree_max_abs=float(np.abs(degree-W.sum(axis=1)).max()),
        direct_graph_max_abs=float(np.abs(graph.ravel()-G@v.ravel()).max()),
        direct_graph_relative=float(np.linalg.norm(graph.ravel()-G@v.ravel())/max(np.linalg.norm(G@v.ravel()),1e-300)),
        direct_symmetry_relative=float(np.linalg.norm(G-G.T)/graph_scale),
        direct_translation_relative=trans/graph_scale,
        direct_pair_min_eig=float(np.linalg.eigvalsh(W).min()),
        direct_graph_min_eig=float(np.linalg.eigvalsh(G).min()),
        t_direct_check_s=time.perf_counter()-start)
    if rec['direct_vs_fft_basis_max_relative'] > 2e-11 or rec['direct_graph_relative'] > 2e-11:
        raise AssertionError(f'Independent stencil audit failed: {rec}')
    return rec


def weights_audit():
    rows = []
    for p in (1,2,3,5,7,9,11):
        maxima = np.zeros(3)
        for family in E.FAMS:
            for seed in E.SEEDS:
                cfg = E.config(family,seed,.75)
                K = E.kernel(.7,.85,0.)
                original,info = E.check_mesh(cfg['L'],K,2.,40,p)
                if original is None:
                    raise ValueError(info)
                # No table construction is needed to compare assignment rules.
                positive = ProjectedMesh.__new__(ProjectedMesh)
                positive.__dict__.update(original.__dict__)
                _,a = original.stencil(cfg['q']); _,b = positive.stencil(cfg['q'])
                oi,ow = original.flat(cfg['q']); pi,pw = positive.flat(cfg['q'])
                v = np.random.default_rng(seed).normal(size=(256,3))
                old = original.apply(oi,ow,v); new = positive.apply(pi,pw,v)
                maxima = np.maximum(maxima,[np.max(np.abs(a-b)),np.max(np.abs(ow-pw)),
                    np.linalg.norm(old-new)/max(np.linalg.norm(old),1e-300)])
        rows.append(dict(p=p,M=40,kappa=.7,xi=.85,kc=2.,rho=.75,configurations=9,
                         axis_weight_max_abs=float(maxima[0]),product_weight_max_abs=float(maxima[1]),
                         apply_max_relative=float(maxima[2])))
    C.write_csv(RESULTS/'assignment_weights_audit.csv',rows)
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    unique = {}
    for sweep,setting,value,p in sweep_points():
        key = ident(p)
        if key not in unique:
            unique[key] = [p,[]]
        unique[key][1].append(f'{sweep}:{setting}:{value:g}')
    out = RESULTS/'table_diagnostics.csv'
    rows = read(out)
    done = {r['parameter_key'] for r in rows if r.get('status') in ('complete','invalid')}
    start = time.perf_counter()
    print(f'{len(unique)} distinct tables, {len(done)} already complete; serial single-thread audit',flush=True)
    for number,(key,(p,members)) in enumerate(unique.items(),1):
        if key in done:
            continue
        row = dict(parameter_key=key,method='PPPM-grid-PSD',sweep_members=';'.join(members),
                   **{k:p[k] for k in ('rho','kappa','xi','sF','M','p')},direct_checked=False)
        cfg = E.config('uniform',1,p['rho'])
        K = E.kernel(p['kappa'],p['xi'],0.)
        _,kc,shell = E.cutoffs(p,cfg['L'])
        original,info = E.check_mesh(cfg['L'],K,kc,p['M'],p['p'])
        row.update(L=cfg['L'],h=cfg['L']/p['M'],kc=kc,sF_eff=kc/(2*p['xi']),kc_shell_n=shell,
                   mesh_valid=bool(info['valid']))
        if original is None:
            row.update(status='invalid',failure=info['reason'])
        else:
            projected = ProjectedMesh(original)
            row.update(status='complete',**projected.diagnostics)
            if representative(p):
                row.update(direct_checks(projected))
            del projected,original
        rows.append(row)
        C.write_csv(out,rows)
        gc.collect()
        print(f'{number}/{len(unique)} {key}: {row["status"]}, negative={row.get("negative_blocks","NA")}, '
              f'direct={row["direct_checked"]}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
    assignment = weights_audit()
    valid = [r for r in rows if r.get('status') == 'complete']
    direct = [r for r in valid if str(r.get('direct_checked')) in ('True','true','1')]
    summary = dict(command='python3 '+str(Path(__file__).resolve()),workers=1,threads=1,
                   elapsed_s=time.perf_counter()-start,distinct_tables=len(unique),
                   complete_tables=len(valid),invalid_tables=len(rows)-len(valid),direct_checked_tables=len(direct),
                   max_direct_vs_fft_relative=max(float(r['direct_vs_fft_basis_max_relative']) for r in direct),
                   max_direct_graph_relative=max(float(r['direct_graph_relative']) for r in direct),
                   max_assignment_apply_relative=max(r['apply_max_relative'] for r in assignment),
                   python=platform.python_version(),numpy=np.__version__,threadpools=threadpool_info(),
                   source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  (HERE/'grid_psd.py',HERE/'table_audit.py',HERE/'test_grid_psd.py')})
    C.write_json(RESULTS/'table_audit_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('threadpools','source_sha256')},indent=2),flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
