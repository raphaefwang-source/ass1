"""Fresh accuracy and Monte Carlo validation of the compiled pair-OU loop.

All baseline inputs and ranks are read only.  New results, generated Gaussian
inputs, raw samples and their provenance stay under optimized_pairou.
"""
import argparse
import hashlib
import time

import bench_common as C
np, P, O = C.np, C.P, C.O
from scipy import linalg, stats
from compiled_edge import CompiledEdgeOU


def draws(rng, k, E, dim):
    """Generate independent long, forward and reverse standard Gaussians."""
    return dict(z=rng.normal(size=(k, dim)), ef=rng.normal(size=(k, E, 3)),
                er=rng.normal(size=(k, E, 3)))


def zero_draws(k, E, dim):
    return dict(z=np.zeros((k, dim)), ef=np.zeros((k, E, 3)),
                er=np.zeros((k, E, 3)))


def finite_step(system, edges, p, dt, rd, rn, k, noise, method):
    """Apply original/compiled A or unchanged B to already generated draws.

    ``method`` is A or B.  ``edges`` selects the Python or compiled A engine.
    Gaussian generation is outside this function; callers time it separately.
    """
    start = time.perf_counter()
    calls = 0

    def action(v):
        nonlocal calls
        calls += 1
        return system.full_action(v) if method == 'B' else system.long_action(v)

    if method == 'B':
        out, diag = O.ou_b(action, p, noise['z'][0], dt, rd, rn)
        diag.update(actual_full_actions=calls, actual_long_actions=0)
        diag['step_wall_s'] = time.perf_counter()-start
        return out, diag
    if method != 'A':
        raise ValueError('method must be A or B')
    out, merged = np.asarray(p).copy(), {}
    for step in range(k):
        out, diag = O.ou_a(action, out, noise['z'][step], noise['ef'][step],
                          noise['er'][step], edges, dt/k, rd, rn)
        for key, value in diag.items():
            if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                merged[key] = merged.get(key, 0)+value
    merged.update(actual_full_actions=0, actual_long_actions=calls, substeps=k,
                  step_wall_s=time.perf_counter()-start)
    return out, merged


def ideal_step(edges, p, dt, k, noise, method, reference):
    """Same draws with exact dense full B or dense long inside ideal A."""
    R, F = reference
    if method == 'B':
        return R@p+F@noise['z'][0]
    if method != 'A':
        raise ValueError('method must be A or B')
    out = np.asarray(p).copy()
    for step in range(k):
        out = edges.sweep(out, noise['ef'][step], dt/(2*k))[0]
        out = R@out+F@noise['z'][step]
        out = edges.sweep(out, noise['er'][step], dt/(2*k), reverse=True)[0]
    return out


def ideal_mean(edges, long_reference, dt, k):
    """Independent dense ordered half-sweep products and dense long damping."""
    N = long_reference.N
    forward = np.eye(3*N).reshape(N, 3, 3*N)
    h = dt/k
    for i, j, K in zip(edges.i, edges.j, edges.K):
        eig, U = linalg.eigh(K, check_finite=False)
        if eig.min() < -1e-10:
            raise ValueError('Materially indefinite near edge')
        Dminus = (U*np.expm1(-h*np.maximum(eig, 0.)))@U.T
        delta = .5*Dminus@(forward[i]-forward[j])
        forward[i] += delta
        forward[j] -= delta
    forward = forward.reshape(3*N, 3*N)
    Rlong = long_reference.matrices(h)[0]
    one = forward.T@Rlong@forward
    return np.linalg.matrix_power(one, k)


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300))


def com_error(a, b):
    return float(np.linalg.norm(np.asarray(a).reshape(-1, 3).sum(0)-
                                np.asarray(b).reshape(-1, 3).sum(0)))


def gaussian_hash(noise):
    h = hashlib.sha256()
    for key in ('z', 'ef', 'er'):
        h.update(key.encode())
        h.update(np.ascontiguousarray(noise[key]).tobytes())
    return h.hexdigest()


def metadata(row):
    return {key: row[key] for key in ('config', 'N', 'family', 'seed', 'epsilon')}


def selected_rows(config, alpha=None):
    return [row for row in C.parent_rows('selection.csv')
            if row['config'] == config and row['available'] and
            (alpha is None or row['alpha'] == alpha)]


def engines(system):
    original = O.EdgeOU(system.edges, n_particles=len(system.q))
    compiled = CompiledEdgeOU(system.edges, n_particles=len(system.q),
        configuration_token=system.metadata['position_sha256'], backend='compiled')
    if compiled.backend != 'compiled':
        raise RuntimeError(f'Compiled validation cannot use a fallback: {compiled.compile_info}')
    np.testing.assert_array_equal(original.i, compiled.i)
    np.testing.assert_array_equal(original.j, compiled.j)
    np.testing.assert_array_equal(original.K, compiled.K)
    return original, compiled


def load_system(meta):
    system, saved = C.make_system(meta), C.load_saved(meta)
    hashes = {}
    for attribute, key in (('q', 'q'), ('full_dense', 'G'),
                           ('near_dense', 'GS'), ('long_dense', 'GL')):
        actual = getattr(system, attribute)
        np.testing.assert_array_equal(actual, saved[key])
        hashes[key+'_sha256'] = C.array_sha(actual)
    if system.L != float(saved['L']):
        raise RuntimeError('Reconstructed box differs from PR 8 input')
    hashes.update(baseline_input_sha256=C.sha(C.baseline_input(meta)),
                  p_sha256=C.array_sha(saved['p']), z_sha256=C.array_sha(saved['z']))
    return system, saved, hashes


def accuracy(args):
    if (C.RESULTS/'accuracy_done.json').exists():
        raise FileExistsError('Preserve completed accuracy evidence before rerunning')
    C.INPUTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    C.save('environment_accuracy.json', C.environment())
    structures, rows, splits, rank_checks, modes = [], [], [], [], []
    cases = C.parent_rows('structure.csv')
    if args.limit:
        cases = cases[:args.limit]
    C.save('validation_protocol.json', dict(
        cases=len(cases), baseline_selected_rows=len(C.parent_rows('selection.csv')),
        global_seed=C.SEED, alphas=list(P.ALPHAS),
        methods=['A-python', 'A-compiled', 'B'],
        baseline_role='read-only positions, p/z, exact discrete graph and final selected ranks; all new numerical outputs recomputed',
        compiled_tolerance=1e-11, dynamical_tolerance=P.TOL,
        MC_samples_small=args.mc_small, MC_samples_medium=args.mc_medium,
        MC_seeds='2026102708 + 100000 + case_index*100000 + equilibrium*10000 + sample',
        MC_reference='exact same independent draws in ideal dense full B or ideal split A with dense long OU',
        MC_second_interpretation='Difference of second moments around known ideal mean; variance difference plus squared finite-mean bias, not a covariance certificate',
        inference_limit='Finite registered probes and samples; no universal finite-rank guarantee'))
    for index, case in enumerate(cases):
        meta = metadata(case)
        C.log(f'accuracy {index+1}/{len(cases)} {meta["config"]}')
        system, saved, hashes = load_system(meta)
        original, compiled = engines(system)
        full, long = P.DenseOU(system.full_dense), P.DenseOU(system.long_dense)
        base = dict(meta, **hashes, graph_identical=True,
            active_backend=compiled.backend,
            compiled_library_sha256=compiled.compile_info.get('library_sha256'),
            edges=len(original.i), n_images=system.op.n_images,
            edge_i_sha256=C.array_sha(original.i), edge_j_sha256=C.array_sha(original.j),
            edge_K_sha256=C.array_sha(original.K), near_min=P.DenseOU(system.near_dense).raw_min,
            long_min=long.raw_min, full_min=full.raw_min, lambda_max=full.lmax)
        structures.append(base)
        np.savez_compressed(C.INPUTS/(meta['config']+'.npz'),
            **saved, edge_i=original.i, edge_j=original.j, edge_K=original.K)
        probes = {name:saved[name].copy() for name in ('slow', 'fast', 'random')}
        probes['COM'] = np.tile([1., 0., 0.], int(meta['N']))/np.sqrt(int(meta['N']))
        for aindex, alpha in enumerate(P.ALPHAS):
            selections = {r['method']:r for r in selected_rows(meta['config'], alpha)}
            if set(selections) != {'A', 'B'}:
                raise RuntimeError('Expected available A and B baseline ranks')
            sa, sb = selections['A'], selections['B']
            dt, k = float(sa['dt']), int(sa['substeps'])
            if not np.isclose(dt*full.lmax, alpha, rtol=3e-14, atol=1e-15):
                raise RuntimeError('Physical dt does not match reconstructed eigenvalue')
            R, F, covariance = full.matrices(dt)
            local = long.matrices(dt/k)[:2]
            RA = ideal_mean(original, long, dt, k)
            CA = full.Pi-RA@full.Pi@RA.T
            invariance = P.norm2(R@full.Pi@R.T+covariance-full.Pi)
            split_invariance = P.norm2(RA@full.Pi@RA.T+CA-full.Pi)
            previous = None
            for steps in (k, 2*k, 4*k):
                mean = RA if steps == k else ideal_mean(original, long, dt, steps)
                cov = full.Pi-mean@full.Pi@mean.T
                error = P.norm2(mean-R)
                order = float(np.log2(previous/error)) if previous is not None and error > 1e-14 else None
                splits.append(dict(meta, alpha=alpha, dt=dt, substeps=steps,
                    selected_substeps=k, mean_operator_error=error,
                    cov_operator_error=P.norm2(cov-covariance), observed_order=order))
                previous = error
            seed = C.SEED+1000+index*100+aindex
            noise = draws(np.random.default_rng(seed), k, len(original.i), len(saved['p']))
            # The frozen benchmark's saved first Gaussian is identical to PR 8.
            noise['z'][0] = saved['z']
            path = C.INPUTS/f'draws_{meta["config"]}_alpha{alpha:g}.npz'
            np.savez_compressed(path, **noise, seed=seed)
            forward_py = original.sweep(saved['p'], noise['ef'][0], dt/(2*k))[0]
            forward_c = compiled.sweep(saved['p'], noise['ef'][0], dt/(2*k))[0]
            reverse_py = original.sweep(forward_py, noise['er'][0], dt/(2*k), reverse=True)[0]
            reverse_c = compiled.sweep(forward_py, noise['er'][0], dt/(2*k), reverse=True)[0]
            original_out = None
            for label, engine, method, selection in (
                    ('A-python', original, 'A', sa), ('A-compiled', compiled, 'A', sa),
                    ('B', original, 'B', sb)):
                rd, rn = int(selection['r_d']), int(selection['r_n'])
                steps = k if method == 'A' else 1
                exact_mean = RA if method == 'A' else R
                reference = local if method == 'A' else (R, F)
                actual, diag = finite_step(system, engine, saved['p'], dt, rd, rn, steps, noise, method)
                ideal = ideal_step(original, saved['p'], dt, steps, noise, method, reference)
                mean, _ = finite_step(system, engine, saved['p'], dt, rd, rn, steps,
                                     zero_draws(steps, len(original.i), len(saved['p'])), method)
                if label == 'A-python':
                    original_out = actual
                row = dict(meta, alpha=alpha, dt=dt, method=label, r_d=rd, r_n=rn,
                    substeps=steps, gaussian_sha256=gaussian_hash(noise), draws_file_sha256=C.sha(path),
                    same_first_saved_z=True,
                    coupled_ideal_path_error=relative(actual, ideal),
                    mean_action_error=relative(mean, exact_mean@saved['p']),
                    full_exact_mean_action_error=relative(mean, R@saved['p']),
                    COM_error=com_error(actual, saved['p']), dense_fdt_residual=invariance,
                    ideal_A_invariance_residual=split_invariance,
                    split_mean_operator_error=P.norm2(RA-R),
                    split_cov_operator_error=P.norm2(CA-covariance),
                    forward_original_vs_compiled=relative(forward_c, forward_py),
                    reverse_original_vs_compiled=relative(reverse_c, reverse_py),
                    original_vs_compiled_path=relative(actual, original_out) if label == 'A-compiled' else None,
                    actual_full_actions=diag['actual_full_actions'], actual_long_actions=diag['actual_long_actions'])
                rows.append(row)
                for name, vector in probes.items():
                    output, _ = finite_step(system, engine, vector, dt, rd, rn, steps,
                        zero_draws(steps, len(original.i), len(vector)), method)
                    modes.append(dict(meta, alpha=alpha, dt=dt, method=label, direction=name,
                        exact_decay=float(vector@R@vector), ideal_decay=float(vector@exact_mean@vector),
                        actual_decay=float(vector@output),
                        mean_action_error=relative(output, exact_mean@vector),
                        COM_error=com_error(output, vector)))
                if label == 'A-compiled' and max(row['original_vs_compiled_path'],
                        row['forward_original_vs_compiled'], row['reverse_original_vs_compiled']) > 1e-11:
                    raise RuntimeError('Compiled pair sweep changed the ordered numerical path')
            for method, selection, action, dense_reference, h in (
                    ('A', sa, system.long_action, long, dt/k),
                    ('B', sb, system.full_action, full, dt)):
                RR, FF, _ = dense_reference.matrices(h)
                for kind, rank, target, vector in (
                        ('damp', int(selection['r_d']), RR, saved['p']),
                        ('noise', int(selection['r_n']), FF, saved['z'])):
                    projected = P.project(vector)
                    actual, diag = O.lanczos_action(action, projected, rank, h, kind)
                    rank_checks.append(dict(meta, alpha=alpha, dt=dt, method=method,
                        kind=kind, rank=rank, error=relative(actual, target@projected),
                        rank_used=diag['rank_used'], ritz_min_raw=diag['ritz_min_raw'],
                        negative_ritz_count=diag['negative_ritz_count']))
        for filename, data in (('structure.csv', structures), ('accuracy.csv', rows),
                ('splitting.csv', splits), ('selected_rank_checks.csv', rank_checks), ('mode_decay.csv', modes)):
            C.write(filename, data)
    C.save('accuracy_done.json', dict(elapsed_s=time.perf_counter()-started,
        systems=len(structures), rows=len(rows), splitting_rows=len(splits),
        selected_rank_rows=len(rank_checks), mode_rows=len(modes),
        compiled_path_max=max(r['original_vs_compiled_path'] or 0. for r in rows),
        compiled_sweep_max=max(max(r['forward_original_vs_compiled'], r['reverse_original_vs_compiled']) for r in rows),
        baseline_graphs_identical=all(r['graph_identical'] for r in structures)))


def scalar_statistics(x):
    n = len(x)
    mean = float(x.mean())
    centered = x-mean
    variance = float(x.var(ddof=1))
    mean_se = float(x.std(ddof=1)/np.sqrt(n))
    variance_se = float((centered**2).std(ddof=1)/np.sqrt(n))
    fourth = centered**4
    fourth_mean = float(fourth.mean())
    fourth_se = float(fourth.std(ddof=1)/np.sqrt(n))
    critical = float(stats.t.ppf(.975, n-1))
    return dict(mean=mean, mean_se=mean_se, mean_ci_low=mean-critical*mean_se,
        mean_ci_high=mean+critical*mean_se, variance=variance, variance_se=variance_se,
        variance_ci_low=max(0., variance-critical*variance_se),
        variance_ci_high=variance+critical*variance_se,
        fourth_moment=fourth_mean, fourth_se=fourth_se,
        fourth_ci_low=max(0., fourth_mean-critical*fourth_se),
        fourth_ci_high=fourth_mean+critical*fourth_se)


def mc(args):
    if (C.RESULTS/'mc_done.json').exists():
        raise FileExistsError('Preserve completed MC evidence before rerunning')
    started = time.perf_counter()
    cases = [r for r in C.parent_rows('structure.csv') if r['epsilon'] == 1e-4 and
             r['seed'] == 2 and (r['family'] == 'uniform' or int(r['N']) == 16)]
    if args.limit:
        cases = cases[:args.limit]
    rows, fullcov, pairrows, manifests = [], [], [], []
    for index, case in enumerate(cases):
        meta = metadata(case)
        system, saved, hashes = load_system(meta)
        original, compiled = engines(system)
        full, long = P.DenseOU(system.full_dense), P.DenseOU(system.long_dense)
        selection = {r['method']:r for r in selected_rows(meta['config'], 1.)}
        if set(selection) != {'A', 'B'}:
            raise RuntimeError('Missing final baseline alpha=1 ranks')
        dt, k = float(selection['A']['dt']), int(selection['A']['substeps'])
        R, F, covariance = full.matrices(dt)
        RA = ideal_mean(original, long, dt, k)
        CA = full.Pi-RA@full.Pi@RA.T
        local = long.matrices(dt/k)[:2]
        N = int(meta['N'])
        count = args.mc_small if N <= 16 else args.mc_medium
        probes = {name:saved[name].copy() for name in ('slow', 'fast', 'random')}
        probes['COM'] = np.tile([1., 0., 0.], N)/np.sqrt(N)
        for phase in ('conditional', 'equilibrium'):
            C.log(f'MC {index+1}/{len(cases)} {meta["config"]} {phase} n={count}')
            samples = {name:[] for name in ('A-python', 'A-compiled', 'B')}
            ideals = {'A':[], 'B':[]}
            paths = {name:[] for name in samples}
            seeds, draw_hashes, starts = [], [], []
            max_com = {name:0. for name in samples}
            for sample in range(count):
                seed = C.SEED+100000+index*100000+(10000 if phase == 'equilibrium' else 0)+sample
                rng = np.random.default_rng(seed)
                p = saved['p'].copy() if phase == 'conditional' else (
                    P.project(rng.normal(size=3*N))+np.tile([.2, -.1, .3], N))
                noise = draws(rng, k, len(original.i), 3*N)
                seeds.append(seed)
                draw_hashes.append(gaussian_hash(noise))
                starts.append(p)
                idealA = ideal_step(original, p, dt, k, noise, 'A', local)
                idealB = ideal_step(original, p, dt, 1, noise, 'B', (R, F))
                ideals['A'].append(idealA)
                ideals['B'].append(idealB)
                for label, engine, method in (('A-python', original, 'A'),
                        ('A-compiled', compiled, 'A'), ('B', original, 'B')):
                    sel = selection[method]
                    actual, _ = finite_step(system, engine, p, dt, int(sel['r_d']),
                        int(sel['r_n']), k if method == 'A' else 1, noise, method)
                    samples[label].append(actual)
                    paths[label].append(relative(actual, idealA if method == 'A' else idealB))
                    max_com[label] = max(max_com[label], com_error(actual, p))
            samples = {name:np.asarray(value) for name, value in samples.items()}
            ideals = {name:np.asarray(value) for name, value in ideals.items()}
            path = C.INPUTS/f'mc_{meta["config"]}_{phase}.npz'
            np.savez_compressed(path, A_python=samples['A-python'], A_compiled=samples['A-compiled'],
                B=samples['B'], ideal_A=ideals['A'], ideal_B=ideals['B'],
                p=np.asarray(starts), seeds=np.asarray(seeds, dtype=np.int64),
                gaussian_sha256=np.asarray(draw_hashes),
                path_A_python=np.asarray(paths['A-python']), path_A_compiled=np.asarray(paths['A-compiled']),
                path_B=np.asarray(paths['B']))
            manifests.append(dict(meta, phase=phase, samples=count, file=str(path),
                sha256=C.sha(path), baseline_input_sha256=hashes['baseline_input_sha256'],
                seed_recipe='2026102708+100000+case_index*100000+equilibrium*10000+sample',
                gaussian_recipe='numpy.default_rng(seed); equilibrium p draw first if needed; draws z[k,d], ef[k,E,3], er[k,E,3] in that order',
                raw_Gaussian_preservation='per-sample seed, exact draw recipe and SHA256; raw output samples and p retained'))
            max_difference = float(np.max(np.linalg.norm(samples['A-compiled']-samples['A-python'], axis=1)))
            pairrows.append(dict(meta, phase=phase, n_samples=count,
                compiled_original_sample_max_abs=float(np.max(np.abs(samples['A-compiled']-samples['A-python']))),
                compiled_original_sample_max_norm=max_difference,
                compiled_original_relative_max=max(relative(a, b) for a, b in zip(samples['A-compiled'], samples['A-python']))))
            if pairrows[-1]['compiled_original_relative_max'] > 1e-11:
                raise RuntimeError('Compiled MC differs from original ordered samples')
            for label, sample_array in samples.items():
                method = 'B' if label == 'B' else 'A'
                sel = selection[method]
                exactmean = RA@saved['p'] if phase == 'conditional' and method == 'A' else (
                    R@saved['p'] if phase == 'conditional' else np.tile([.2, -.1, .3], N))
                ideal_cov = (CA if method == 'A' else covariance) if phase == 'conditional' else full.Pi
                full_cov = covariance if phase == 'conditional' else full.Pi
                if N <= 16:
                    centered = sample_array-sample_array.mean(axis=0)
                    empirical = centered.T@centered/(count-1)
                    fullcov.append(dict(meta, method=label, phase=phase, n_samples=count,
                        spectral_error_vs_ideal=P.norm2(empirical-ideal_cov),
                        spectral_error_vs_exact_full=P.norm2(empirical-full_cov),
                        interpretation='finite-sample empirical covariance error; includes sampling uncertainty'))
                for direction, vector in probes.items():
                    x, ix = sample_array@vector, ideals[method]@vector
                    mu = float(vector@exactmean)
                    paired_mean = x-ix
                    paired_second = (x-mu)**2-(ix-mu)**2
                    ideal_variance = float(vector@ideal_cov@vector)
                    exact_variance = float(vector@full_cov@vector)
                    rows.append(dict(meta, method=label, phase=phase, direction=direction,
                        alpha=1., dt=dt, r_d=int(sel['r_d']), r_n=int(sel['r_n']),
                        substeps=k if method == 'A' else 1, n_samples=count,
                        **scalar_statistics(x), ideal_mean=mu,
                        exact_full_mean=float(vector@(R@saved['p'] if phase == 'conditional' else np.tile([.2, -.1, .3], N))),
                        ideal_variance=ideal_variance, exact_full_variance=exact_variance,
                        ideal_gaussian_fourth=3*ideal_variance**2,
                        exact_full_gaussian_fourth=3*exact_variance**2,
                        paired_mean_error=float(paired_mean.mean()),
                        paired_mean_se=float(paired_mean.std(ddof=1)/np.sqrt(count)),
                        paired_second_error=float(paired_second.mean()),
                        paired_second_se=float(paired_second.std(ddof=1)/np.sqrt(count)),
                        paired_second_role='about known ideal mean: variance difference plus squared finite-mean bias',
                        coupled_ideal_path_max=max(paths[label]), COM_max=max_com[label],
                        confidence_role='pointwise approximate 95% t intervals; variance/fourth use plug-in centered moments; no multiplicity correction'))
            for filename, values in (('mc.csv', rows), ('mc_fullcov.csv', fullcov), ('mc_pairwise.csv', pairrows)):
                C.write(filename, values)
            C.save('mc_inputs_manifest.json', manifests)
    C.save('mc_done.json', dict(elapsed_s=time.perf_counter()-started, cases=len(cases),
        rows=len(rows), fullcov_rows=len(fullcov), paired_case_phases=len(pairrows),
        maximum_compiled_original_relative=max(r['compiled_original_relative_max'] for r in pairrows),
        maximum_coupled_ideal_path=max(r['coupled_ideal_path_max'] for r in rows),
        mc_samples_small=args.mc_small, mc_samples_medium=args.mc_medium,
        all_outputs_newly_computed=True, baseline_MC_results_used=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('accuracy', 'mc', 'all'))
    parser.add_argument('--mc-small', type=int, default=512)
    parser.add_argument('--mc-medium', type=int, default=128)
    parser.add_argument('--limit', type=int, default=0,
                        help='smoke-test prefix; use an isolated output directory for pilots')
    args = parser.parse_args()
    if args.stage in ('accuracy', 'all'):
        accuracy(args)
    if args.stage in ('mc', 'all'):
        mc(args)
