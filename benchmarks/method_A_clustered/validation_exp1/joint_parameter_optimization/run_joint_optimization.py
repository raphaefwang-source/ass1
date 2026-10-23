#!/usr/bin/env python3
"""Candidate 1 only, 1e-4 joint (s, xi, M, p) experiment.

No production operator is changed. Search assembly uses the same E.real_dense,
mode_pair_matrix_fast and lap_full as E.evaluate; finalists use E.evaluate.
Run from any directory: python3 run_joint_optimization.py all --rebuild-necessary-refs
Stages are resumable; references are never regenerated if a matching cache exists.
Use --xi-extension --output-dir PATH for the wider p=3..7, xi=1.15..1.6
experiment. Historical results/ and figures/ are never used for new output.
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
BASELINE_ID = 'old_baseline'
REFS, PROBES = {}, {}


def configure(args):
    """Configure the same stages for an isolated, resumable extension run."""
    global RESULTS, FIGURES, S, XI, MS, PS, OLD, BASELINE_ID
    if args.xi_extension and args.output_dir is None:
        raise ValueError('--xi-extension requires --output-dir to preserve historical results')
    if args.output_dir:
        root = Path(args.output_dir).expanduser().resolve()
        if root == HERE or any(root.is_relative_to(HERE/p) for p in ('results','figures')):
            raise ValueError('Use a separate output directory, not the historical experiment directory')
        RESULTS, FIGURES = root / 'results', root / 'figures'
    if args.xi_extension:
        XI = (1.15, 1.2, 1.3, 1.4, 1.5, 1.6)
        S = (2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0)
        MS = (16, 18, 20, 22, 24, 27, 30, 32, 36, 40, 45, 48, 54, 60, 64)
        PS = (3, 4, 5, 6, 7)
        OLD = pars(3.1, 1.15, 20, 5)
        BASELINE_ID = 'previous_xi115'
    for attr, global_name, cast in (('xis', 'XI', float), ('s_values', 'S', float),
                                    ('mesh_sizes', 'MS', int), ('orders', 'PS', int)):
        value = getattr(args, attr, None)
        if value is not None:
            globals()[global_name] = tuple(cast(v) for v in value)
    if any(p < 3 or p > 7 for p in PS) or min(MS) < 2 or min(XI) <= 0 or min(S) <= 0:
        raise ValueError('Invalid grid')


def add_options(ap):
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--xi-extension', action='store_true')
    ap.add_argument('--xis', type=float, nargs='+')
    ap.add_argument('--s-values', type=float, nargs='+')
    ap.add_argument('--mesh-sizes', type=int, nargs='+')
    ap.add_argument('--orders', type=int, nargs='+')
    ap.add_argument('--runtime-prune-factor', type=float, default=1.6)
    ap.add_argument('--validation-band', type=float, default=None)
    ap.add_argument('--finalists-per-xi', type=int, default=8)
    ap.add_argument('--rounds', type=int, default=9)
    ap.add_argument('--reps', type=int, default=11)
    ap.add_argument('--recheck-pruned', action='store_true',
                    help='Revisit prior runtime-pruned rows, preserving the first attempts')
    ap.add_argument('--include-balance-controls', action='store_true',
                    help='Time the previous xi=.85 and 1.0 designs as fixed diagnostic controls')
    ap.add_argument('--timing-top-measured', type=int, default=0,
                    help='Refine this many previously timed designs per xi, plus fixed controls')
    ap.add_argument('--refresh-shortlist-times', action='store_true',
                    help='Remeasure all error-screen survivors before ranking the validation shortlist')


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def write(name, rows):
    path = RESULTS / name
    tmp = path.with_name(path.name+'.tmp')
    C.write_csv(tmp, rows)
    # Keep repository CSV line endings at LF; numerical text is unchanged.
    tmp.write_bytes(tmp.read_bytes().replace(b'\r\n', b'\n'))
    os.replace(tmp,path)


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
    if args.xi_extension:
        for r in rows:
            r.setdefault('attempt_id', 0)
    if args.xi_extension and args.recheck_pruned:
        prior = [r for r in rows if r.get('status') == 'runtime pruned before dense error']
        write('search_prior_attempts.csv', read('search_prior_attempts.csv') + prior)
        rows = [r for r in rows if r.get('status') != 'runtime pruned before dense error']
        log(f'Rechecking {len(prior)} runtime-pruned points with factor {args.runtime_prune_factor:g}')
    done = {r['candidate_id'] for r in rows}
    design = configs()[0]
    G = loadref(design)
    norm = E.norm2_sym(G)[0]
    X = np.random.default_rng(11).standard_normal((256, 3))
    # Persistent raw timings permit auditing each screening measurement.
    timing = read('search_timing.csv')
    attempts = {}
    for t in timing:
        if t['phase'] == 'design':
            t.setdefault('attempt_id', 0)
            attempts[t['candidate_id']] = max(attempts.get(t['candidate_id'], -1), t['attempt_id'])
    for xi in XI:
        incumbent = min((r['t_design_s'] for r in rows if r['xi'] == xi and r.get('screen_pass')),
                        default=float('inf'))
        for s in S:
            if all(ident(pars(s,xi,M,p)) in done for p in PS for M in MS):
                log(f'resume xi={xi:g} s={s:g}: all {len(PS)*len(MS)} points already recorded')
                continue
            meshes = E.MeshCache()
            short = None
            for p in PS:
                for M in MS:
                    par = pars(s, xi, M, p)
                    cid = ident(par)
                    if cid in done:
                        continue
                    if args.xi_extension:
                        meshes.m.clear()  # retain the existing MeshCache without accumulating large grids
                    op, info = build(design, par, meshes)
                    r = dict(candidate_id=cid, round=1 if xi == .85 else 2, s=s, xi=xi, M=M, p=p,
                             attempt_id=attempts.get(cid, -1)+1,
                             mesh_valid=info['valid'], mesh_reason=info['reason'],
                             sF_eff=E.cutoffs(par, design['L'])[1]/(2*xi),
                             eta=info.get('eta'), n_modes=info.get('n_modes'))
                    if op is not None:
                        ts = C.median_time(lambda: op.apply(X), 5 if args.xi_extension else 7, warm=3)
                        r.update(t_design_s=float(np.median(ts)), n_images_design=op.n_images,
                                 t_design_min_s=float(np.min(ts)),
                                 design_config=design['name'])
                        for rep, t in enumerate(ts):
                            timing.append(dict(candidate_id=cid, config=design['name'], rep=rep,
                                               attempt_id=r['attempt_id'],
                                               t_matvec_s=float(t), phase='design'))
                        # Cost pruning is based on measured matvec, with a wide
                        # guard band; it is not an assumed monotone M/p cost law.
                        if args.xi_extension and r['t_design_min_s'] > args.runtime_prune_factor*incumbent:
                            r.update(status='runtime pruned before dense error',
                                     runtime_prune_incumbent_s=incumbent, screen_pass=False)
                            rows.append(r)
                            done.add(cid)
                            continue
                        if short is None:
                            short = E.real_dense(op)
                        Gh = short + E.lap_full(E.mode_pair_matrix_fast(op.mesh, op.x))
                        err = E.norm2_sym(Gh - G)[0]/norm
                        r['E_G_design'] = err
                        r['screen_pass'] = err <= TOL
                        if err <= TOL:
                            incumbent = min(incumbent, r['t_design_s'])
                        if err <= TOL and not args.xi_extension:
                            medians = []
                            for cfg in configs():
                                op9, _ = build(cfg, par, meshes)
                                ts9 = C.median_time(lambda: op9.apply(X), 7, warm=3)
                                medians.append(float(np.median(ts9)))
                                for rep, t in enumerate(ts9):
                                    timing.append(dict(candidate_id=cid, config=cfg['name'], rep=rep,
                                                       t_matvec_s=float(t), phase='nine_config_screen'))
                            r['t_nine_median_s'] = float(np.median(medians))
                        elif err > TOL:
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
        if args.xi_extension:
            # Refine a broad timing shortlist on all nine before costly full
            # E.evaluate validation. All candidate grids have been built/timed.
            design_order = [r for r in search if r['xi'] == xi and r.get('screen_pass')]
            timing = read('search_timing.csv')
            X = np.random.default_rng(11).standard_normal((256, 3))
            timing_pass = time.strftime('%Y%m%d_%H%M%S')
            if args.refresh_shortlist_times:
                rng = np.random.default_rng(20261023)
                for idx in rng.permutation(len(design_order)):
                    r = design_order[idx]
                    op, _ = build(configs()[0], fromrow(r), E.MeshCache())
                    ts = C.median_time(lambda:op.apply(X),5,warm=3)
                    r['t_design_ranking_s'] = float(np.median(ts))
                    for rep,t in enumerate(ts):
                        timing.append(dict(candidate_id=r['candidate_id'],config=configs()[0]['name'],
                                           rep=rep,phase='design_refresh',timing_pass=timing_pass,t_matvec_s=float(t)))
            design_order.sort(key=lambda r:r.get('t_design_ranking_s') or r['t_design_s'])
            if not design_order:
                log(f'xi={xi:g}: no design survivor')
                continue
            ranking_time = lambda r:r.get('t_design_ranking_s') or r['t_design_s']
            shortlist = [r for rank, r in enumerate(design_order)
                         if rank < 32 or ranking_time(r) <= 1.4*ranking_time(design_order[0])]
            for r in shortlist:
                if r.get('t_nine_median_s') is not None and not args.refresh_shortlist_times:
                    continue
                par, meshes, medians = fromrow(r), E.MeshCache(), []
                for cfg in configs():
                    op, _ = build(cfg, par, meshes)
                    ts = C.median_time(lambda: op.apply(X), 7, warm=3)
                    medians.append(float(np.median(ts)))
                    for rep, t in enumerate(ts):
                        timing.append(dict(candidate_id=r['candidate_id'], config=cfg['name'], rep=rep,
                                           t_matvec_s=float(t), phase='nine_config_screen',timing_pass=timing_pass))
                r['t_nine_median_s'] = float(np.median(medians))
                r['nine_timing_pass'] = timing_pass
                write('search.csv', search)
            write('search_timing.csv', timing)
            if args.refresh_shortlist_times:
                shortlisted = {r['candidate_id'] for r in shortlist}
                for r in search:
                    if r['xi'] == xi and r['candidate_id'] not in shortlisted:
                        r['t_nine_median_s'] = None
                byid = {r['candidate_id']:r for r in shortlist}
                for v in summaries:
                    if v['xi'] == xi:
                        v['t_search_nine_median_s'] = (byid[v['candidate_id']]['t_nine_median_s']
                                                      if v['candidate_id'] in byid else None)
                write('search.csv',search)
                write('validation_summary.csv',summaries)
            log(f'xi={xi:g}: nine-config timing shortlist {len(shortlist)} of {len(design_order)} survivors')
        ordered = sorted([r for r in search if r['xi'] == xi and r.get('screen_pass')],
                         key=lambda r: r.get('t_nine_median_s') or float('inf'))
        if args.xi_extension:
            ordered = [r for r in ordered if r.get('t_nine_median_s') is not None]
        first_time = None
        nfeas = 0
        for rank, r in enumerate(ordered, 1):
            band = args.validation_band or (1.12 if args.xi_extension else 1.20)
            if first_time is not None and nfeas >= 3 and r['t_nine_median_s'] > band*first_time:
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
                               t_validation_median_s=float(np.median([v['t_matvec_s'] for v in rs])),
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
    cid = BASELINE_ID
    if not any(v['candidate_id'] == cid for v in rows):
        meshes = E.MeshCache()
        rs = [evaluate(cfg, OLD, meshes, candidate_id=cid) for cfg in configs()]
        rows += rs
        write('validation.csv', rows)
    if args.xi_extension and args.include_balance_controls:
        for xi, s, M, p in ((.85, 3.1, 16, 5), (1., 3.3, 16, 5)):
            cid = f'balance_control_xi{xi:g}'
            if any(v['candidate_id'] == cid for v in rows):
                continue
            meshes, par = E.MeshCache(), pars(s, xi, M, p)
            rs = [evaluate(cfg, par, meshes, candidate_id=cid) for cfg in configs()]
            assert all(v['E_G'] <= TOL and v['lam_min_h'] > 0 for v in rs)
            rows += rs
            summaries.append(dict(candidate_id=cid, s=s, xi=xi, M=M, p=p, n_configs=9,
                                  fully_validated=True, feasible=True, diagnostic_control=True,
                                  E_G_max_tested=max(v['E_G'] for v in rs)))
            write('validation.csv', rows)
            write('validation_summary.csv', summaries)


def stage_timing(args):
    """Serial randomized interleaved timing of all fully feasible finalists.

    7 rounds x 7 samples after 3 warmups per block, same X for all operators;
    each round randomizes operator order. Three separate setup repeats record
    mesh build and particle-dependent setup, excluding parameter kernel creation.
    Extension mode times complete, real and mesh actions, randomizing candidates
    at every repetition (default 9 x 11); a later pass may refine measured ranks.
    """
    candidates = {r['candidate_id']: fromrow(r) for r in read('validation_summary.csv') if r['feasible']}
    if args.xi_extension:
        summaries = [r for r in read('validation_summary.csv') if r['feasible']]
        selected = set()
        if args.timing_top_measured:
            previous = read('final_timing.csv')
            if not previous:
                raise ValueError('--timing-top-measured requires a completed timing pass')
            for xi in XI:
                ss = [r for r in summaries if r['xi'] == xi and
                      any(t['candidate_id']==r['candidate_id'] for t in previous)]
                key = lambda r: np.median([t['t_matvec_s'] for t in previous if t['candidate_id']==r['candidate_id']])
                selected.update(r['candidate_id'] for r in sorted(ss,key=key)[:args.timing_top_measured])
        for xi in XI:
            ss = [r for r in summaries if r['xi'] == xi]
            for key in ('t_search_nine_median_s', 't_validation_median_s'):
                selected.update(r['candidate_id'] for r in sorted(ss, key=lambda r:r.get(key) or float('inf'))[:args.finalists_per_xi])
        candidates = {cid:par for cid,par in candidates.items() if cid in selected}
        candidates.update({r['candidate_id']:fromrow(r) for r in summaries if r.get('diagnostic_control')})
    candidates = {cid:par for cid,par in candidates.items() if ident(par) != ident(OLD)}
    candidates[BASELINE_ID] = OLD
    # Preserve a prior timing pass when extending the same new experiment.
    if args.xi_extension and (RESULTS/'final_timing.csv').exists():
        import shutil
        archive = RESULTS / f"timing_pass_{time.strftime('%Y%m%d_%H%M%S')}"
        archive.mkdir()
        for name in ('final_timing.csv', 'final_timing_raw.csv', 'final_setup_raw.csv'):
            shutil.copy2(RESULTS/name, archive/name)
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
        rounds, reps = (args.rounds, args.reps) if args.xi_extension else (7, 7)
        if args.xi_extension:
            functions = {cid:dict(
                matvec=lambda op=op:op.apply(X),
                real=lambda op=op:np.einsum('nab,nb->na',op.DS,X)-op.real_H(X),
                mesh=lambda op=op:np.einsum('nab,nb->na',op.Dm,X)-op.mesh.apply(op.idx,op.wt,X))
                         for cid,op in ops.items()}
            for fn in functions.values():
                Ysplit = fn['real']()+fn['mesh']()
                assert np.linalg.norm(Ysplit-fn['matvec']())/np.linalg.norm(Ysplit) < 1e-12
            # Interleave candidates at every individual repetition, so a whole
            # block of one operator cannot systematically benefit from drift.
            for rd in range(rounds):
                for phase in rng.permutation(('matvec','real','mesh')):
                    for cid in rng.permutation(list(ops)):
                        C.median_time(functions[cid][phase],0,warm=3)
                    for rep in range(reps):
                        for cid in rng.permutation(list(ops)):
                            t = C.median_time(functions[cid][phase],1,warm=1)[0]
                            raw.append(dict(candidate_id=str(cid),config=cfg['name'],round=rd,rep=rep,
                                            phase=str(phase),t_matvec_s=float(t)))
        else:
            for rd in range(rounds):
                for cid in rng.permutation(list(ops)):
                    ts = C.median_time(lambda:ops[cid].apply(X),reps,warm=3)
                    for rep,t in enumerate(ts):
                        raw.append(dict(candidate_id=str(cid),config=cfg['name'],round=rd,rep=rep,
                                        phase='matvec',t_matvec_s=float(t)))
        for cid, op in ops.items():
            ts = [r['t_matvec_s'] for r in raw if r['candidate_id'] == cid and r['config'] == cfg['name'] and r['phase'] == 'matvec']
            ss = [r for r in setup if r['candidate_id'] == cid and r['config'] == cfg['name']]
            parts = {}
            # Same matrix-free action components, timed separately for explanation.
            for label, fn in (('real_H', lambda: op.real_H(X)),
                              ('mesh_H', lambda: op.mesh.apply(op.idx, op.wt, X))):
                parts[f't_{label}_s'] = float(np.median(C.median_time(fn, 21, warm=3)))
            if args.xi_extension:
                for phase in ('real', 'mesh'):
                    tt = [r['t_matvec_s'] for r in raw if r['candidate_id'] == cid and
                          r['config'] == cfg['name'] and r['phase'] == phase]
                    st = C.tstats(tt)
                    parts.update({f't_{phase}_s':st['median'], f't_{phase}_iqr_rel':st['iqr_rel']})
            out.append(dict(candidate_id=cid, config=cfg['name'], family=cfg['family'], seed=cfg['seed'],
                            t_matvec_s=C.tstats(ts)['median'], t_matvec_iqr_rel=C.tstats(ts)['iqr_rel'],
                            t_matvec_min_s=min(ts), t_matvec_max_s=max(ts), n_samples=len(ts),
                            rounds=rounds, reps=reps,
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
    add_options(ap)
    args = ap.parse_args()
    configure(args)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.xi_extension:
        manifest = RESULTS.parent/'historical_artifact_hashes.json'
        if not manifest.exists():
            protected = [p for folder in ('results','figures') for p in (HERE/folder).rglob('*') if p.is_file()]
            protected += [HERE/n for n in ('report.md','run.log') if (HERE/n).exists()]
            C.write_json(manifest,{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})
    E.log_cmd(RESULTS)
    with threadpool_limits(limits=1):
        if args.xi_extension and (RESULTS/'provenance.json').exists():
            previous = json.loads((RESULTS/'provenance.json').read_text())
            jsave(f"provenance_before_{time.strftime('%Y%m%d_%H%M%S')}.json", previous)
        jsave('provenance.json', dict(E.provenance(), threadpool_info=threadpool_info(),
                                     search_grid=dict(s=S, xi=XI, M=MS, p=PS),
                                     xi_extension=args.xi_extension, baseline_id=BASELINE_ID,
                                     baseline_parameters=OLD, options=vars(args),
                                     runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                     reference='Gamma_D only', target=TOL,
                                     objective='median over nine per-configuration median matvec times',
                                     platform=platform.platform()))
        stages = (['refs', 'search', 'validate', 'timing'] if args.xi_extension else
                  ['refs', 'small', 'search', 'validate', 'timing']) if 'all' in args.stages else args.stages
        for stage in stages:
            log(f'START {stage}')
            globals()[f'stage_{stage}'](args)
            log(f'DONE {stage}')


if __name__ == '__main__':
    main()
