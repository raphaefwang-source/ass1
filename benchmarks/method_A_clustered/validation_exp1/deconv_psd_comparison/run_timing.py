"""Common serial three-method timing; run after all accuracy workers finish.

Selection is restricted to complete nine-configuration validation. Historical
timings guide shortlisting only; all reported comparisons use this common pass.
Existing final outputs are never overwritten. ``--dry-run`` only saves a plan.
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import study as X
from study import np, E, C

EPSILONS = (1e-3, 1e-4, 1e-5, 1e-6)
PARAMETERS = ('xi', 'sR', 'sF', 'sT', 'M', 'p', 'kappa', 'rho')


def par_from(row):
    return X.parameters(**{k: row[k] for k in PARAMETERS if k in row and row[k] is not None})


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def proxy_cost(row):
    """A shortlist hint only: no historical cost enters final ranking."""
    for key in ('t_screen_s', 't_nine_screen_s', 'median_matvec_s',
                't_complete_s', 't_matvec_s'):
        if finite(row.get(key)) and float(row[key]) > 0:
            return float(row[key])
    p = par_from(row)
    # Same rough work model for all methods when no measured hint is available.
    return (p['sR']/p['xi'])**3 * 1e-5 + p['M']**3 * math.log(max(p['M'], 2)) * 1e-8 + p['p']**3 * 1e-5


def candidates():
    rows = X.read(X.RESULTS/'validation_summary.csv')
    search = {(r.get('method'), r.get('id')): r for r in X.read(X.RESULTS/'search.csv')}
    valid = []
    for row in rows:
        if (row.get('method') not in X.METHODS or int(row.get('n_configs', 0)) != 9
                or not finite(row.get('max_E_G'))):
            continue
        par = par_from(row)
        if par['kappa'] != .2 or par['rho'] != .75:
            continue
        row = dict(row, **par)
        row.setdefault('id', X.ident(row['method'], par))
        hint = search.get((row['method'], row['id']), {})
        for key in ('t_screen_s', 't_nine_screen_s', 'median_matvec_s', 't_complete_s', 't_matvec_s'):
            if not finite(row.get(key)) and finite(hint.get(key)):
                row[key] = hint[key]
        valid.append(row)
    unique = {(row['method'], row['id']): row for row in valid}
    valid = list(unique.values())
    shortlist, frontier_hints = [], []
    for method in X.METHODS:
        feasible = sorted((r for r in valid if r['method'] == method and r['max_E_G'] <= 1e-4),
                          key=proxy_cost)
        if not feasible:
            raise RuntimeError(f'{method}: no fully nine-configuration validated candidate at 1e-4')
        # Every fully validated main-target feasible candidate receives the
        # same serial prescreen, removing concurrency bias in search timings.
        shortlist.extend(feasible)
        for eps in EPSILONS:
            feasible_eps = sorted((r for r in valid if r['method'] == method and r['max_E_G'] <= eps),
                                  key=proxy_cost)
            if feasible_eps:
                frontier_hints.append(dict(feasible_eps[0], frontier_hint_epsilon=eps))
    return shortlist, frontier_hints, valid


def system_snapshot(label):
    try:
        out = subprocess.run(['ps', '-axo', 'pid,ppid,pcpu,pmem,comm'],
                             check=True, capture_output=True, text=True).stdout.splitlines()
        proc = sorted(out[1:], key=lambda line: float(line.split(None, 4)[2]), reverse=True)
        processes = '\n'.join(out[:1]+proc[:15])
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        processes = str(exc)
    return dict(label=label, unix=time.time(), loadavg=list(os.getloadavg()),
                process_rss_max_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                top_processes=processes)


def array_allocations(obj, excluded=()):
    """Count each owning ndarray allocation once, including shared/view arrays."""
    seen, allocations = set(), {}
    excluded = set(excluded)

    def visit(value):
        if id(value) in seen or id(value) in excluded:
            return
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            owner = value
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            allocations[id(owner)] = int(owner.nbytes)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key not in ('K', 'kernel'):
                    visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        elif (hasattr(value, '__dict__') and not inspect.isroutine(value)
              and not inspect.isclass(value) and not inspect.ismodule(value)):
            visit(vars(value))

    visit(obj)
    return allocations


def memory_record(op):
    table = array_allocations(op.mesh)
    config = array_allocations(op, excluded=(id(op.mesh),))
    config = {key: val for key, val in config.items() if key not in table}
    M = op.mesh.M
    real_grid_bytes = M**3 * 8
    complex_grid_bytes = M*M*(M//2+1)*16
    # Explicit estimate, not a measured peak: three spread fields, three
    # gathered fields, three rFFT sources and one influence work array.
    fft_estimate = 6*real_grid_bytes + 4*complex_grid_bytes
    near_estimate = op.n_images*(2*3+2)*8 + op.N*3*8*4
    return dict(table_array_bytes=sum(table.values()), table_unique_arrays=len(table),
                config_array_bytes=sum(config.values()), config_unique_arrays=len(config),
                resident_array_bytes=sum(table.values())+sum(config.values()),
                fft_temporary_estimated_bytes=fft_estimate,
                near_temporary_estimated_bytes=near_estimate,
                temporary_estimated_bytes=fft_estimate+near_estimate,
                estimated_peak_array_bytes=sum(table.values())+sum(config.values())+fft_estimate+near_estimate,
                memory_estimate_kind='unique owning NumPy allocations; temporary sizes are conservative work-array estimates',
                process_rss_max_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def table_profile(mesh):
    result = {}
    for obj, prefix in ((mesh, ''), (getattr(mesh, 'original', None), 'original_')):
        if obj is None:
            continue
        for key, val in vars(obj).items():
            if key.startswith('t_') and finite(val):
                result[prefix+key] = float(val)
        for name in ('timings', 'timing', 'setup_timings'):
            profile = getattr(obj, name, None)
            if isinstance(profile, dict):
                result.update({prefix+k: float(v) for k, v in profile.items() if finite(v)})
        diagnostics = getattr(obj, 'diagnostics', {})
        diagnostics = diagnostics() if callable(diagnostics) else diagnostics
        if isinstance(diagnostics, dict):
            result.update({prefix+k: float(v) for k, v in diagnostics.items()
                           if k.startswith('t_') and finite(v)})
    return result


def prescreen(rows, rng, outdir):
    cfg = E.config('uniform', 2, .75)
    vector = rng.normal(size=(cfg['N'], 3)); vector -= vector.mean(0)
    raw = []
    for rnd in range(3):
        for ix in rng.permutation(len(rows)):
            row = rows[ix]
            op = X.build(row['method'], cfg, par_from(row))
            for _ in range(3):
                op.apply(vector)
            for rep in range(3):
                op.apply(vector)
                start = time.perf_counter_ns(); op.apply(vector)
                dt = (time.perf_counter_ns()-start)*1e-9
                raw.append(dict(id=row['id'], method=row['method'], config=cfg['name'],
                                round=rnd, rep=rep, seconds=dt))
            del op
        X.write(outdir/'serial_prescreen_raw.csv', raw)
        X.log(f'common timing prescreen round {rnd+1}/3')
    rank = {row['id']: float(np.median([r['seconds'] for r in raw if r['id'] == row['id']])) for row in rows}
    selected = []
    for method in X.METHODS:
        own = sorted((r for r in rows if r['method'] == method), key=lambda r: rank[r['id']])
        selected.extend(dict(r, prescreen_median_s=rank[r['id']], selection='1e-4 common serial prescreen top 2')
                        for r in own[:2])
    return selected


def bootstrap(raw, best, rng):
    result = []
    pairs = [('B2-positive', 'PPPM-original'),
             ('PPPM-grid-PSD', 'PPPM-original'),
             ('PPPM-grid-PSD', 'B2-positive')]
    for method, reference_method in pairs:
        row, reference = best.get(method), best.get(reference_method)
        if row is None or reference is None:
            continue
        blocks = []
        for cfg in X.configurations():
            rounds = sorted({r['round'] for r in raw if r['id'] == row['id'] and r['config'] == cfg['name']})
            for rnd in rounds:
                own = [r['seconds'] for r in raw if r['id'] == row['id'] and r['config'] == cfg['name'] and r['round'] == rnd]
                ref = [r['seconds'] for r in raw if r['id'] == reference['id'] and r['config'] == cfg['name'] and r['round'] == rnd]
                if own and ref:
                    blocks.append(float(np.median(own)/np.median(ref)))
        if not blocks:
            continue
        values = [float(np.median(rng.choice(blocks, len(blocks), replace=True))) for _ in range(5000)]
        low, high = np.percentile(values, [2.5, 97.5])
        result.append(dict(method=method, reference_method=reference_method, id=row['id'],
                           reference_id=reference['id'], ratio_median_paired_blocks=float(np.median(blocks)),
                           ratio_ci95_low=float(low), ratio_ci95_high=float(high), blocks=len(blocks),
                           bootstrap_resamples=5000,
                           note='configuration/round paired blocks; conditional on measured candidate selection'))
    return result


def run(args):
    outdir = Path(args.output).resolve() if args.output else X.RESULTS
    outdir.mkdir(parents=True, exist_ok=True)
    for name in ('final_timing.csv', 'final_timing_raw.csv', 'serial_prescreen_raw.csv', 'timing_protocol.json'):
        if (outdir/name).exists() and not args.dry_run:
            raise RuntimeError(f'{outdir/name} already exists; use a fresh --output directory to preserve this pass')
    shortlist, hints, validated = candidates()
    if args.main_confirmation:
        primary=json.loads((X.RESULTS/'primary_timing'/'best_configuration.json').read_text())
        ids={r['id'] for r in primary.values()}
        ids.add(X.ident('PPPM-original',X.parameters()))
        ids.add(X.ident('B2-positive',X.parameters(xi=.35,sR=3.1,M=72,p=2)))
        ids.add(X.ident('PPPM-original',par_from(primary['PPPM-grid-PSD'])))
        shortlist=[r for r in validated if r['id'] in ids and r['max_E_G']<=1e-4]
        hints=[]
        if {r['method'] for r in shortlist}!=set(X.METHODS):
            raise RuntimeError('Confirmation requires all three methods')
    snapshot = system_snapshot('before_common_serial_timing')
    plan = dict(shortlist=shortlist, frontier_hints=hints, eligible_validated_rows=len(validated),
                rounds=args.rounds, reps=args.reps, n_configs=9, objective='median of nine configuration medians',
                threads=1, seed=args.seed, initial_system=snapshot)
    X.jsave(outdir/'timing_plan.json', plan)
    if args.dry_run:
        X.log(f'timing dry-run: {len(shortlist)} prescreen candidates, {len(hints)} frontier hints')
        return
    rng = np.random.default_rng(args.seed)
    started = time.time()
    raw, setup, tables, memories, snapshots = [], [], [], [], [snapshot]
    component_rows = []
    with X.threadpool_limits(1):
        # Kernel setup is common warm state and is excluded from parameter-table
        # setup for every method. Its separate raw timings are recorded below.
        for row in shortlist+hints:
            E.kernel(row['kappa'], row['xi'], 0.)
        selected = ([dict(r,selection='main confirmation: primary winner, historical anchor, or cross-method winner') for r in shortlist]
                    if args.main_confirmation else prescreen(shortlist, rng, outdir))
        for hint in hints:
            if hint['id'] not in {r['id'] for r in selected}:
                selected.append(dict(hint, selection=f'frontier hint epsilon={hint["frontier_hint_epsilon"]:g}'))
        X.write(outdir/'timed_candidates.csv', selected)
        cfg0 = E.config('uniform', 2, .75)
        for ix in rng.permutation(len(selected)):
            row = selected[ix]
            for rep in range(3):
                gc.collect()
                start = time.perf_counter_ns()
                mesh = X.table(row['method'], cfg0['L'], par_from(row))
                elapsed = (time.perf_counter_ns()-start)*1e-9
                tables.append(dict(id=row['id'], method=row['method'], rep=rep, seconds=elapsed,
                                   **table_profile(mesh)))
                del mesh
            X.write(outdir/'table_setup_raw.csv', tables)
        for cfg in X.configurations():
            vector = rng.normal(size=(cfg['N'], 3)); vector -= vector.mean(0)
            # Fixed parameter tables are shared across rounds for each candidate;
            # configuration setup and complete action remain independently timed.
            meshes = {row['id']: X.table(row['method'], cfg['L'], par_from(row)) for row in selected}
            for rnd in range(args.rounds):
                for ix in rng.permutation(len(selected)):
                    row = selected[ix]
                    op = X.build(row['method'], cfg, par_from(row), meshes[row['id']])
                    for _ in range(3):
                        op.apply(vector)
                    for rep in range(args.reps):
                        op.apply(vector)
                        start = time.perf_counter_ns(); op.apply(vector)
                        elapsed = (time.perf_counter_ns()-start)*1e-9
                        raw.append(dict(id=row['id'], method=row['method'], config=cfg['name'],
                                        round=rnd, rep=rep, seconds=elapsed))
                    del op
                snapshots.append(system_snapshot(f'{cfg["name"]}_round{rnd+1}'))
                X.write(outdir/'final_timing_raw.csv', raw)
                X.jsave(outdir/'timing_progress.json', dict(config=cfg['name'], round=rnd+1,
                          rounds=args.rounds, samples=len(raw), elapsed_s=time.time()-started))
            for ix in rng.permutation(len(selected)):
                row = selected[ix]
                for rep in range(3):
                    start = time.perf_counter_ns()
                    op = X.build(row['method'], cfg, par_from(row), meshes[row['id']])
                    elapsed = (time.perf_counter_ns()-start)*1e-9
                    setup.append(dict(id=row['id'], method=row['method'], config=cfg['name'],
                                      rep=rep, seconds=elapsed, n_images=op.n_images,
                                      **{f'part_{key}_s': val for key, val in op.t_setup_parts.items()}))
                    if rep == 0:
                        memories.append(dict(id=row['id'], method=row['method'], config=cfg['name'],
                                             **memory_record(op)))
                    del op
            X.write(outdir/'config_setup_raw.csv', setup)
            X.write(outdir/'memory.csv', memories)
            X.jsave(outdir/'system_load.json', snapshots)
            meshes.clear(); gc.collect()
            X.log(f'common timing {cfg["name"]}: {len(selected)} candidates, {time.time()-started:.1f} s elapsed')
        summary = []
        for row in selected:
            own = [r for r in raw if r['id'] == row['id']]
            config_medians = [float(np.median([r['seconds'] for r in own if r['config'] == cfg['name']]))
                              for cfg in X.configurations()]
            config_setup = [float(np.median([r['seconds'] for r in setup if r['id'] == row['id'] and r['config'] == cfg['name']]))
                            for cfg in X.configurations()]
            iqrs = [C.tstats([r['seconds'] for r in own if r['config'] == cfg['name']])['iqr_rel']
                    for cfg in X.configurations()]
            mem = [r for r in memories if r['id'] == row['id']]
            summary.append(dict(row, median_matvec_s=float(np.median(config_medians)),
                                min_config_median_s=min(config_medians), max_config_median_s=max(config_medians),
                                max_relative_iqr=max(iqrs), median_config_setup_s=float(np.median(config_setup)),
                                median_table_setup_s=float(np.median([r['seconds'] for r in tables if r['id'] == row['id']])),
                                n_samples_per_config=args.rounds*args.reps,
                                table_array_bytes=max(r['table_array_bytes'] for r in mem),
                                median_config_array_bytes=float(np.median([r['config_array_bytes'] for r in mem])),
                                max_config_array_bytes=max(r['config_array_bytes'] for r in mem),
                                max_temporary_estimated_bytes=max(r['temporary_estimated_bytes'] for r in mem),
                                max_estimated_peak_array_bytes=max(r['estimated_peak_array_bytes'] for r in mem)))
        summary.sort(key=lambda r: r['median_matvec_s'])
        X.write(outdir/'final_timing.csv', summary)
        best = {method: next(r for r in summary if r['method'] == method and r['max_E_G'] <= 1e-4)
                for method in X.METHODS}
        X.jsave(outdir/'best_configuration.json', best)
        X.write(outdir/'best_configuration.csv', list(best.values()))
        frontier = []
        for method in X.METHODS:
            for eps in EPSILONS:
                feasible = [r for r in summary if r['method'] == method and r['max_E_G'] <= eps]
                certified = [r for r in validated if r['method'] == method and r['max_E_G'] <= eps]
                if feasible:
                    frontier.append(dict(feasible[0], epsilon=eps, available=True,
                                         status='fastest in common serial timed, fully nine-configuration validated candidates',
                                         n_validated_feasible=len(certified), n_timed_feasible=len(feasible)))
                else:
                    frontier.append(dict(method=method, epsilon=eps, available=False, id=None,
                                         median_matvec_s=None, max_E_G=None,
                                         n_validated_feasible=len(certified), n_timed_feasible=0,
                                         status='no fully nine-configuration validated feasible candidate'))
        X.write(outdir/'accuracy_time_frontier.csv', frontier)
        X.write(outdir/'frontier.csv', frontier)
        ratios = bootstrap(raw, best, rng)
        X.write(outdir/'paired_speed_ratios.csv', ratios)
        # Components are diagnostic and follow the complete-action pass; their
        # separately measured medians need not sum to the complete-action median.
        for cfg in X.configurations():
            vector = rng.normal(size=(cfg['N'], 3)); vector -= vector.mean(0)
            for method in rng.permutation(X.METHODS):
                row = best[method]; op = X.build(method, cfg, par_from(row))
                grids = op.spread(vector); phi = op.fft_influence(grids)
                np.testing.assert_allclose(op.apply_staged(vector), op.apply(vector), rtol=1e-12, atol=1e-10)
                functions = dict(
                    near=lambda: np.einsum('nab,nb->na', op.DS, vector)-op.real_H(vector),
                    spread_gather=lambda: (op.spread(vector), op.gather(phi)),
                    fft_influence=lambda: op.fft_influence(grids),
                    mesh=lambda: np.einsum('nab,nb->na', op.Dm, vector)-op.mesh.apply(op.idx, op.wt, vector),
                    total=lambda: op.apply(vector))
                for phase in rng.permutation(list(functions)):
                    for rep, dt in enumerate(C.median_time(functions[phase], 15, warm=3)):
                        component_rows.append(dict(id=row['id'], method=method, config=cfg['name'],
                                                   phase=phase, rep=rep, seconds=float(dt)))
                del op, grids, phi
            X.write(outdir/'components_raw.csv', component_rows)
        kernel_rows = []
        for xi in sorted({r['xi'] for r in selected}):
            for rep in range(3):
                start = time.perf_counter_ns(); kernel = E.ThetaKernel(.2, xi, 0.)
                kernel_rows.append(dict(xi=xi, rep=rep, seconds=(time.perf_counter_ns()-start)*1e-9))
                del kernel
        X.write(outdir/'common_kernel_setup_raw.csv', kernel_rows)
        snapshots.append(system_snapshot('after_all_timing'))
        X.jsave(outdir/'system_load.json', snapshots)
        X.jsave(outdir/'timing_protocol.json', dict(
            rounds=args.rounds, reps=args.reps, n_configs=9, threads=1, seed=args.seed,
            started_unix=started, elapsed_s=time.time()-started,
            randomization='all method/candidate blocks randomized within each configuration and round',
            warmup='3 per candidate block; 1 immediately before every sample',
            objective='median over nine configuration medians',
            prescreen=('all selected main confirmation finalists retained; primary 79-candidate common serial prescreen preserved in primary_timing/' if args.main_confirmation
                       else 'all nine-configuration main-target feasible candidates per method; uniform seed2; 3 rounds x 3 samples; retain 2 per method'),
            timing_batch='main_confirmation' if args.main_confirmation else 'primary',
            frontier='one proxy-fastest validated hint per method and epsilon added when absent; final ranking uses common timing only',
            setup='3 fresh parameter tables and 3 fresh configuration builds per state; common kernel primitives warmed',
            memory='unique owning ndarray allocations, views and shared original-mesh arrays counted once; temporary estimate is not measured peak RSS',
            load_limitation='Other system workloads remain active and are neither terminated nor controlled; see system_load.json, including chrome_crashpad when present.',
            paired_ratios=ratios, environment_threads={k: os.environ.get(k) for k in (
                'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')},
            provenance=E.provenance(platform=platform.platform(), threadpool=X.threadpool_info()),
            source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(X.HERE.glob('*.py'))}))
        X.log('common three-method timing finished')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--reps', type=int, default=7)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--output', help='fresh output directory; default is this study results/')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--main-confirmation',action='store_true',help='Confirm primary winners, historical anchors, and Original at grid-PSD winner parameters; preserve primary_timing first')
    args = parser.parse_args()
    if args.rounds < 1 or args.reps < 1:
        parser.error('--rounds and --reps must be positive')
    run(args)


if __name__ == '__main__':
    main()
