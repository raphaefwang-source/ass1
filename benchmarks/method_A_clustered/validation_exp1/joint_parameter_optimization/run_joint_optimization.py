#!/usr/bin/env python3
"""Candidate 1 only, 1e-4 joint (s, xi, M, p) experiment.

No production operator is changed. Search assembly uses the same E.real_dense,
mode_pair_matrix_fast and lap_full as E.evaluate; finalists use E.evaluate.
Run from any directory: python3 run_joint_optimization.py all --rebuild-necessary-refs
Stages are resumable; references are never regenerated if a matching cache exists.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ass1_joint_mpl")

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ewald_exp_lib as E
import common as C
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
TOL = 1e-4
S = (3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0)
XI = (0.85, 0.70, 1.00, 1.15)  # first round at 0.85
MS = (16, 18, 20, 22, 24, 27, 30, 32)
PS = (3, 4, 5)
OLD = dict(E.BASE, xi=0.85, sR=3.2, sF=3.1, M=18, p=4)
REFS, PROBES = {}, {}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def write(name, rows):
    path = RESULTS / name
    C.write_csv(path, rows)
    # Keep repository CSV line endings at LF; numerical text is unchanged.
    path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))


def read(name):
    p = RESULTS / name
    return E.read_csv(p) if p.exists() else []


def jsave(name, obj):
    C.write_json(RESULTS / name, obj)


def configs():
    # uniform s2 is an early rejection configuration, never a feasibility proxy.
    order = [("uniform", 2), ("strong", 1)]
    order += [(f, sd) for f in E.FAMS for sd in E.SEEDS if (f, sd) not in order]
    return [E.config(f, sd, 0.75) for f, sd in order]


def pars(s, xi, M, p):
    return dict(E.BASE, sR=float(s), sF=float(s), xi=float(xi), M=int(M), p=int(p), theta=0.0)


def ident(par):
    return f"xi{par['xi']:g}_s{par['sR']:g}_sf{par['sF']:g}_M{par['M']}_p{par['p']}"


def fromrow(r):
    return pars(r['s'], r['xi'], r['M'], r['p'])


def loadref(cfg):
    key = cfg['name']
    if key not in REFS:
        with np.load(E.ref_path(cfg, 0.7)) as z:
            G = z['G_D']  # exclusively direct periodic sum
        REFS[key] = G
    return REFS[key]


def probes(cfg):
    if cfg['name'] not in PROBES:
        PROBES[cfg['name']] = E.probes_for(loadref(cfg))[0]
    return PROBES[cfg['name']]


def stage_refs(args):
    """Build only nine missing kappa=.7, rho=.75 direct references, no LF."""
    old = E.read_csv(HERE.parent / 'accuracy_sweeps/results/references.csv')
    old = {r['config']: r for r in old if r['kappa'] == 0.7 and r['rho'] == 0.75}
    out, convrows = [], []
    for cfg in configs():
        path = E.ref_path(cfg, 0.7)
        prev = old[cfg['name']]
        if not path.exists():
            if not args.rebuild_necessary_refs:
                raise RuntimeError(f'Missing {path}; assess cost then use --rebuild-necessary-refs')
            # Existing per-config norm estimates select the radius only. They do
            # not supply the numerical reference or any new error measurement.
            R = E.choose_RD(256, cfg['L'], 0.7, prev['norm2'])
            assert R == 55.0
            edges = np.round(R * np.array([.4, .5, .6, .7, .8, .9, .95, 1.]), 6)
            log(f"direct ref {cfg['name']}: R={R}, historical direct time={prev['t_direct_s']:.1f}s")
            t0 = time.perf_counter()
            i, j, acc, count = E.DP.annulus_tensors(cfg['q'], cfg['L'], edges,
                                                   budget=1 << 20, g=E.g_phys(0.7))
            dt = time.perf_counter() - t0
            cum = np.cumsum(acc, axis=0)
            G = E.DP.tensors_to_dense(i, j, cum[-1], 256)
            norm = E.norm2_sym(G)[0]
            conv = np.array([[Rk, E.norm2_sym(E.DP.tensors_to_dense(i, j, cum[k], 256)-G)[0]/norm,
                              2*E.DP.tail_bound_per_particle(256, cfg['L'], Rk, g=E.g_phys(.7))/norm]
                             for k, Rk in enumerate(edges)])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, G_D=G, conv=conv)
            meta = dict(source='direct_periodic.annulus_tensors only', R_D=R, t_direct_s=dt,
                        n_images=int(count.sum()), norm2=norm, cert_bound_rel=float(conv[-1, 2]))
            C.write_json(path.with_suffix('.json'), meta)
            log(f"ref complete {dt:.2f}s, norm={norm:.12g}, cert={conv[-1,2]:.3g}")
        else:
            with np.load(path) as z:
                G = z['G_D']
                conv = z['conv']
            meta = C.read_json(path.with_suffix('.json'), {})
        norm = E.norm2_sym(G)[0]
        R = float(conv[-1, 0])
        cert = 2*E.DP.tail_bound_per_particle(256, cfg['L'], R, g=E.g_phys(.7))/norm
        assert cert <= E.REF_CERT_TOL
        assert abs(norm / prev['norm2'] - 1) < 1e-11
        out.append(dict(**E.config_record(cfg), kappa=.7, c=1., cache_file=path.name,
                        cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), R_D=R,
                        cert_bound_rel=cert, norm2=norm,
                        norm_relative_difference_to_exp1=norm / prev['norm2'] - 1,
                        t_direct_s=meta.get('t_direct_s'), source='Gamma_D direct periodic sum'))
        convrows += [dict(config=cfg['name'], R=r[0], diff_to_RD_rel=r[1], cert_bound_rel=r[2]) for r in conv]
        REFS[cfg['name']] = G
        write('references.csv', out)
        write('reference_convergence.csv', convrows)


def evaluate(cfg, par, meshes, **extra):
    G = loadref(cfg)
    # E.evaluate's legacy interface requires a GLF argument. Passing G twice
    # keeps every computation relative to Gamma_D; discard compatibility fields.
    r = E.evaluate(cfg, G, G, probes(cfg), par, meshes, timing=True, dense=True, extra=extra)
    return {k: v for k, v in r.items() if 'vsLF' not in k and not k.startswith('ref_floor')}


def stage_small(args):
    rows = []
    meshes = E.MeshCache()
    for p in (3, 4):
        for cfg in configs():
            r = evaluate(cfg, pars(3.2, .85, 18, p), meshes, candidate_id=f'small_p{p}')
            rows.append(r)
            write('small_test.csv', rows)
        rs = [r for r in rows if r['p'] == p]
        log(f"small p={p}: max E_G={max(r['E_G'] for r in rs):.8g}, median matvec={np.median([r['t_matvec_s'] for r in rs])*1e3:.4f}ms")


def build(cfg, par, meshes):
    K = E.kernel(.7, par['xi'], 0.)
    rc, kc, _ = E.cutoffs(par, cfg['L'])
    mesh, info = meshes.get(cfg['L'], K, kc, par['M'], par['p'])
    if mesh is None:
        return None, info
    return E.ThetaOp(cfg['q'], cfg['L'], K, rc, mesh), info


def stage_search(args):
    """Screen exact design spectral error; time every survivor on all nine.

    No stopping at the first feasible M: larger FFT-friendly sizes can be faster.
    Within each xi, candidates are then validated in measured runtime order.
    """
    rows = read('search.csv')
    done = {r['candidate_id'] for r in rows}
    design = configs()[0]
    G = loadref(design)
    norm = E.norm2_sym(G)[0]
    X = np.random.default_rng(11).standard_normal((256, 3))
    # Persistent raw timings permit auditing each screening measurement.
    timing = read('search_timing.csv')
    for xi in XI:
        for s in S:
            meshes = E.MeshCache()
            short = None
            for p in PS:
                for M in MS:
                    par = pars(s, xi, M, p)
                    cid = ident(par)
                    if cid in done:
                        continue
                    op, info = build(design, par, meshes)
                    r = dict(candidate_id=cid, round=1 if xi == .85 else 2, s=s, xi=xi, M=M, p=p,
                             mesh_valid=info['valid'], mesh_reason=info['reason'],
                             sF_eff=E.cutoffs(par, design['L'])[1]/(2*xi),
                             eta=info.get('eta'), n_modes=info.get('n_modes'))
                    if op is not None:
                        if short is None:
                            short = E.real_dense(op)
                        Gh = short + E.lap_full(E.mode_pair_matrix_fast(op.mesh, op.x))
                        err = E.norm2_sym(Gh - G)[0]/norm
                        ts = C.median_time(lambda: op.apply(X), 7, warm=3)
                        r.update(E_G_design=err, design_config=design['name'],
                                 t_design_s=float(np.median(ts)), n_images_design=op.n_images)
                        for rep, t in enumerate(ts):
                            timing.append(dict(candidate_id=cid, config=design['name'], rep=rep,
                                               t_matvec_s=float(t), phase='design'))
                        r['screen_pass'] = err <= TOL
                        if err <= TOL:
                            medians = []
                            for cfg in configs():
                                op9, _ = build(cfg, par, meshes)
                                ts9 = C.median_time(lambda: op9.apply(X), 7, warm=3)
                                medians.append(float(np.median(ts9)))
                                for rep, t in enumerate(ts9):
                                    timing.append(dict(candidate_id=cid, config=cfg['name'], rep=rep,
                                                       t_matvec_s=float(t), phase='nine_config_screen'))
                            r['t_nine_median_s'] = float(np.median(medians))
                        else:
                            r['status'] = 'infeasible: exact design error exceeds tolerance'
                    else:
                        r['status'] = 'invalid mesh'
                    rows.append(r)
                    done.add(cid)
                write('search.csv', rows)
                write('search_timing.csv', timing)
            sel = [r for r in rows if r['xi'] == xi and r['s'] == s]
            log(f"search xi={xi:g} s={s:g}: {sum(bool(r.get('screen_pass')) for r in sel)} / {len(sel)} pass design")


def stage_validate(args):
    """Validate fastest ranks, retaining at least three feasible per xi.

    Continue through every screen survivor within 20% of the first feasible's
    nine-config runtime, in addition to the three-feasible minimum. This guards
    against noisy short timing ranks without treating M or p as a cost proxy.
    """
    search = read('search.csv')
    rows = read('validation.csv')
    summaries = read('validation_summary.csv')
    done = {r['candidate_id'] for r in summaries}
    for xi in XI:
        ordered = sorted([r for r in search if r['xi'] == xi and r.get('screen_pass')],
                         key=lambda r: r['t_nine_median_s'])
        first_time = None
        nfeas = 0
        for rank, r in enumerate(ordered, 1):
            if first_time is not None and nfeas >= 3 and r['t_nine_median_s'] > 1.20*first_time:
                break
            cid, par = r['candidate_id'], fromrow(r)
            if cid not in done:
                meshes = E.MeshCache()
                rs = []
                for cfg in configs():
                    v = evaluate(cfg, par, meshes, candidate_id=cid, timing_rank_within_xi=rank)
                    rs.append(v)
                    # Exact failure is sufficient to discard a candidate; never
                    # interpret partial validation as a nine-config maximum.
                    if v.get('E_G', float('inf')) > TOL:
                        break
                rows += rs
                full = len(rs) == 9
                feasible = full and all(v.get('mesh_valid') and v['E_G'] <= TOL and
                                        v['lam_min_h'] > 0 and v['sym_resid'] < 1e-12 and
                                        v['mom_opnorm_max'] < 1e-12 and v['mf_vs_dense'] < 1e-10 for v in rs)
                summary = dict(candidate_id=cid, s=r['s'], xi=xi, M=r['M'], p=r['p'],
                               n_configs=len(rs), fully_validated=full, feasible=feasible,
                               E_G_max_tested=max(v['E_G'] for v in rs),
                               t_search_nine_median_s=r['t_nine_median_s'],
                               timing_rank_within_xi=rank)
                summaries.append(summary)
                done.add(cid)
                write('validation.csv', rows)
                write('validation_summary.csv', summaries)
                log(f"validate {cid}: {len(rs)} configs, max={summary['E_G_max_tested']:.5g}, feasible={feasible}")
            summary = next(v for v in summaries if v['candidate_id'] == cid)
            if summary['feasible']:
                nfeas += 1
                if first_time is None:
                    first_time = r['t_nine_median_s']
    # Benchmark is kept despite unequal requested cutoffs, and checked anew.
    cid = 'old_baseline'
    if cid not in done:
        meshes = E.MeshCache()
        rs = [evaluate(cfg, OLD, meshes, candidate_id=cid) for cfg in configs()]
        rows += rs
        write('validation.csv', rows)


def stage_timing(args):
    """Serial randomized interleaved timing of all fully feasible finalists.

    7 rounds x 7 samples after 3 warmups per block, same X for all operators;
    each round randomizes operator order. Three separate setup repeats record
    mesh build and particle-dependent setup, excluding parameter kernel creation.
    """
    candidates = {r['candidate_id']: fromrow(r) for r in read('validation_summary.csv') if r['feasible']}
    candidates['old_baseline'] = OLD
    raw, setup, out = [], [], []
    rng = np.random.default_rng(20261022)
    for cfg in configs():
        X = np.random.default_rng(11).standard_normal((256, 3))
        ops = {}
        for cid, par in candidates.items():
            K = E.kernel(.7, par['xi'], 0.)
            rc, kc, _ = E.cutoffs(par, cfg['L'])
            for rep in range(3):
                t0 = time.perf_counter()
                mesh, info = E.check_mesh(cfg['L'], K, kc, par['M'], par['p'])
                t1 = time.perf_counter()
                op = E.ThetaOp(cfg['q'], cfg['L'], K, rc, mesh)
                t2 = time.perf_counter()
                setup.append(dict(candidate_id=cid, config=cfg['name'], rep=rep,
                                  t_mesh_s=t1-t0, t_particle_setup_s=t2-t1, t_total_setup_s=t2-t0))
            ops[cid] = op
        for rd in range(7):
            for cid in rng.permutation(list(ops)):
                op = ops[cid]
                ts = C.median_time(lambda: op.apply(X), 7, warm=3)
                for rep, t in enumerate(ts):
                    raw.append(dict(candidate_id=str(cid), config=cfg['name'], round=rd, rep=rep,
                                    t_matvec_s=float(t)))
        for cid, op in ops.items():
            ts = [r['t_matvec_s'] for r in raw if r['candidate_id'] == cid and r['config'] == cfg['name']]
            ss = [r for r in setup if r['candidate_id'] == cid and r['config'] == cfg['name']]
            parts = {}
            # Same matrix-free action components, timed separately for explanation.
            for label, fn in (('real_H', lambda: op.real_H(X)),
                              ('mesh_H', lambda: op.mesh.apply(op.idx, op.wt, X))):
                parts[f't_{label}_s'] = float(np.median(C.median_time(fn, 21, warm=3)))
            out.append(dict(candidate_id=cid, config=cfg['name'], family=cfg['family'], seed=cfg['seed'],
                            t_matvec_s=C.tstats(ts)['median'], t_matvec_iqr_rel=C.tstats(ts)['iqr_rel'],
                            t_matvec_min_s=min(ts), t_matvec_max_s=max(ts), n_samples=len(ts),
                            t_mesh_setup_s=float(np.median([r['t_mesh_s'] for r in ss])),
                            t_particle_setup_s=float(np.median([r['t_particle_setup_s'] for r in ss])),
                            t_total_setup_s=float(np.median([r['t_total_setup_s'] for r in ss])),
                            n_images=op.n_images, n_modes=op.mesh.n_modes, **parts))
        write('final_timing_raw.csv', raw)
        write('final_setup_raw.csv', setup)
        write('final_timing.csv', out)
        log(f"interleaved timing {cfg['name']}: {len(ops)} candidates")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stages', nargs='+', choices=('refs', 'small', 'search', 'validate', 'timing', 'all'))
    ap.add_argument('--rebuild-necessary-refs', action='store_true')
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    E.log_cmd(RESULTS)
    with threadpool_limits(limits=1):
        jsave('provenance.json', dict(E.provenance(), threadpool_info=threadpool_info(),
                                     search_grid=dict(s=S, xi=XI, M=MS, p=PS),
                                     reference='Gamma_D only', target=TOL,
                                     objective='median over nine per-configuration median matvec times',
                                     platform=platform.platform()))
        stages = ['refs', 'small', 'search', 'validate', 'timing'] if 'all' in args.stages else args.stages
        for stage in stages:
            log(f'START {stage}')
            globals()[f'stage_{stage}'](args)
            log(f'DONE {stage}')


if __name__ == '__main__':
    main()
