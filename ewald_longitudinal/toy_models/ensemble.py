#!/usr/bin/env python3
"""Independent-seed ensembles of the toy models and A/B comparison figures.

Error bars are standard errors across independent seeds (separate initial
positions, initial momenta and noise streams). Time origins inside one
trajectory are correlated and are NOT treated as independent samples.
Toy application only; no claim about the PRL system or real MD.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import prl_toy_models as model

VCCF_EDGES = {'double_well': [0., 1.36, 1.90, 2.75], 'lj': [0., 1.55, 2.40, 2.75]}
COLORS = {'A': '#2563eb', 'B': '#dc7627'}
STACKED = ('vacf', 'vccf_normalized', 'hydro_L', 'hydro_T', 'van_hove_distinct', 'rdf', 'msd')


def mean_squared_displacement(Q, starts, lags):
    disp = Q[starts[None, :]+lags[:, None]]-Q[starts][None]
    return np.einsum('lsnc,lsnc->l', disp, disp)/(len(starts)*Q.shape[1])


def run_one(job):
    potential, kernel, images, seed, opts, root = job
    folder = Path(root)/images/f'{potential}_{kernel}'/f'seed_{seed}'
    if (folder/'observables.npz').exists():
        return json.loads((folder/'seed_summary.json').read_text())
    folder.mkdir(parents=True, exist_ok=True)
    sim = {k: opts[k] for k in ('n', 'box', 'dt', 'burn', 'steps', 'stride', 'temperature',
                                'gamma', 'kappa', 'r_ref')}
    prior_time, noise_seed, start = 0., seed, None
    if opts.get('continue_from'):
        # Restart from the last saved frame of an earlier ensemble with a fresh noise stream.
        source = Path(opts['continue_from'])/images/f'{potential}_{kernel}'/f'seed_{seed}'
        with np.load(source/'trajectory.npz', allow_pickle=False) as old:
            meta = json.loads(str(old['metadata']))
            start = (old['Q'][-1], meta['mass']*old['V'][-1])  # saved V is velocity, not momentum
        prior_time = meta.get('total_prior_time', 0.)+meta['burn_in_time']+meta['production_time']
        noise_seed = seed+opts['continue_seed_offset']
    tr = model.simulate(potential, kernel, seed=noise_seed, friction_images=images,
                        initial_state=start, **sim)
    meta = json.loads(str(tr['metadata']))
    meta.update(total_prior_time=prior_time, ensemble_seed=seed,
                continued_from=str(opts.get('continue_from')))
    tr['metadata'] = np.array(json.dumps(meta))
    np.savez_compressed(folder/'trajectory.npz', **tr)
    frame = opts['stride']*opts['dt']
    max_lag = int(round(opts['max_lag_time']/frame))
    obs = model.compute_observables(tr['Q'], tr['V'], tr['t'], tr['box'], max_lag=max_lag,
                                    n_lags=max_lag+1, origins=opts['origins'],
                                    vccf_edges=VCCF_EDGES[potential])
    obs['msd'] = mean_squared_displacement(tr['Q'], obs['time_origins'], obs['lag_frames'])
    np.savez_compressed(folder/'observables.npz', **obs)
    d = tr['diagnostics']
    prod = d[d[:, 0] >= opts['burn']*opts['dt']]
    half = len(prod)//2
    t, vacf = obs['lag_time'], obs['vacf']
    fit = t >= t[-1]/2
    r = obs['radial_centers']
    peak = np.argmax(obs['rdf'])
    summary = dict(
        potential=potential, kernel=kernel, images=images, seed=seed, noise_seed=noise_seed,
        effective_burn_in_time=prior_time+opts['burn']*opts['dt'],
        friction_record=json.loads(str(tr['metadata']))['friction_record'],
        T_mean=prod[:, 1].mean(), U_per_particle=prod[:, 2].mean(),
        U_second_minus_first_half=prod[half:, 2].mean()-prod[:half, 2].mean(),
        T_second_minus_first_half=prod[half:, 1].mean()-prod[:half, 1].mean(),
        max_total_momentum=prod[:, 4].max(), min_distance=d[:, 5].min(),
        D_green_kubo=float(obs['mean_squared_speed'])/3*np.trapezoid(vacf, t),
        D_msd=np.polyfit(t[fit], obs['msd'][fit], 1)[0]/6,
        vacf_min=vacf.min(), vacf_min_time=t[np.argmin(vacf)],
        tau_T=np.trapezoid(obs['hydro_T'].mean(axis=1), t),
        rdf_peak=obs['rdf'][peak], rdf_peak_r=r[peak])
    summary = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in summary.items()}
    (folder/'seed_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def aggregate(root, images, potential, kernel, seeds):
    folder = Path(root)/images/f'{potential}_{kernel}'
    runs = [np.load(folder/f'seed_{s}'/'observables.npz') for s in seeds]
    first = runs[0]
    stack = {key: np.stack([np.asarray(o[key], float) for o in runs]) for key in STACKED}
    for key in ('hydro_L', 'hydro_T'):
        stack[key+'_mode_mean'] = stack[key].mean(axis=-1)  # per seed first, then across seeds
    n = len(seeds)
    out = {key+'_mean': v.mean(axis=0) for key, v in stack.items()}
    out.update({key+'_sem': v.std(axis=0, ddof=1)/np.sqrt(n) for key, v in stack.items()})
    for key in ('lag_time', 'vccf_edges', 'radial_centers', 'radial_edges', 'modes'):
        out[key] = first[key]
    out['vccf_pair_origin_counts'] = np.stack([o['vccf_pair_origin_counts'] for o in runs])
    out['seeds'] = np.asarray(seeds)
    np.savez_compressed(folder/'ensemble_observables.npz', **out)
    return out


def band(ax, t, mean, sem, **kw):
    line, = ax.plot(t, mean, **kw)
    ax.fill_between(t, mean-sem, mean+sem, color=line.get_color(), alpha=.22, lw=0)


def plot_ab(ens, potential, images, path, n_seeds, rule=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), layout='constrained')
    for kernel in ('A', 'B'):
        e, c = ens[kernel], COLORS[kernel]
        t = e['lag_time']
        band(axes[0, 0], t, e['vacf_mean'], e['vacf_sem'], color=c, label=kernel)
        edges = e['vccf_edges']
        for b, ls in zip(range(len(edges)-1), ('-', '--', ':')):
            band(axes[0, 1], t, e['vccf_normalized_mean'][:, b], e['vccf_normalized_sem'][:, b],
                 color=c, ls=ls, label=f'{kernel}: {edges[b]:.2f} <= r0 < {edges[b+1]:.2f}')
        band(axes[0, 2], t, e['hydro_L_mode_mean_mean'], e['hydro_L_mode_mean_sem'], color=c, label=kernel)
        band(axes[1, 0], t, e['hydro_T_mode_mean_mean'], e['hydro_T_mode_mean_sem'], color=c, label=kernel)
        r = e['radial_centers']
        band(axes[1, 1], r, e['rdf_mean'], e['rdf_sem'], color=c, label=kernel)
        for target, ls in zip((0.1, 0.5, 2.0), ('-', '--', ':')):
            k = int(np.argmin(np.abs(t-target)))
            band(axes[1, 2], r, e['van_hove_distinct_mean'][k], e['van_hove_distinct_sem'][k],
                 color=c, ls=ls, label=f'{kernel}, t={t[k]:.2f}')
    labels = [('Lag time', 'Normalized VACF', 'VACF'),
              ('Lag time', 'VCCF / <|v|^2>', 'VCCF grouped by r_ij(t0)'),
              ('Lag time', 'C_L(k,t)', 'Longitudinal current, |k|=2pi/L (3 modes averaged)'),
              ('Lag time', 'C_T(k,t)', 'Transverse current, |k|=2pi/L (3 modes averaged)'),
              ('Distance r', 'g(r)', 'RDF = g_d(r,0)'),
              ('Minimum-image distance r', 'g_d(r,t)', 'Distinct van Hove (finite-N normalization)')]
    for ax, (xl, yl, title) in zip(axes.flat, labels):
        ax.set(xlabel=xl, ylabel=yl, title=title)
        ax.legend(fontsize=7)
    for ax in axes.flat[:4]:
        ax.axhline(0, color='.6', lw=.6)
        ax.set_xlim(0, 1.5)
    for ax in axes.flat[4:]:
        ax.axhline(1, color='.6', lw=.6)
    rule = rule or {'minimum': 'minimum-image friction', 'lattice': 'full periodic-image friction (k=0 retained)'}[images]
    fig.suptitle(f'{potential}: kernel A vs B, {rule}\nMean of {n_seeds} independent seeds; bands = +/-1 SEM '
                 'across seeds. Toy model, not a PRL reproduction or MD validation.', fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_conventions(ens_by_images, potential, path, n_seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
    for images, ls in (('minimum', '-'), ('lattice', '--')):
        if images not in ens_by_images:
            continue
        for kernel in ('A', 'B'):
            e = ens_by_images[images][kernel]
            t = e['lag_time']
            for ax, key in zip(axes, ('vacf', 'hydro_L_mode_mean', 'hydro_T_mode_mean')):
                band(ax, t, e[key+'_mean'], e[key+'_sem'], color=COLORS[kernel], ls=ls,
                     label=f'{kernel}, {images}')
    for ax, title in zip(axes, ('VACF', 'C_L, |k|=2pi/L', 'C_T, |k|=2pi/L')):
        ax.set(xlabel='Lag time', ylabel='Normalized correlation', title=title, xlim=(0, 1.0))
        ax.axhline(0, color='.6', lw=.6)
        ax.legend(fontsize=8)
    fig.suptitle(f'{potential}: two different friction MODELS (minimum-image vs full periodic images). '
                 f'{n_seeds} seeds, +/-1 SEM.\nThe gap is a model difference, not an Ewald error.', fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def table(summaries, keys):
    rows = []
    groups = {}
    for s in summaries:
        groups.setdefault((s['images'], s['potential'], s['kernel']), []).append(s)
    for (images, potential, kernel), runs in sorted(groups.items()):
        row = dict(images=images, potential=potential, kernel=kernel, n_seeds=len(runs),
                   seeds=sorted(r['seed'] for r in runs))
        for key in keys:
            v = np.array([r[key] for r in runs])
            row[key] = dict(mean=float(v.mean()), sem=float(v.std(ddof=1)/np.sqrt(len(v))))
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--potentials', nargs='+', choices=['double_well', 'lj'], default=['double_well', 'lj'])
    p.add_argument('--kernels', nargs='+', choices=['A', 'B'], default=['A', 'B'])
    p.add_argument('--images', nargs='+', choices=['minimum', 'lattice'], default=['minimum'])
    p.add_argument('--seeds', type=int, nargs='+', default=[101, 102, 103, 104, 105])
    p.add_argument('--n', type=int, default=64); p.add_argument('--box', type=float, default=5.5)
    p.add_argument('--dt', type=float, default=.005); p.add_argument('--stride', type=int, default=4)
    p.add_argument('--burn', type=int, default=8000); p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--temperature', type=float, default=.7)
    p.add_argument('--gamma', type=float, default=.5); p.add_argument('--kappa', type=float, default=.7)
    p.add_argument('--r-ref', type=float, default=1.3)
    p.add_argument('--max-lag-time', type=float, default=3.0)
    p.add_argument('--origins', type=int, default=300)
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--out', default='ensemble_results')
    p.add_argument('--continue-from', default=None,
                   help='Earlier ensemble root; each seed restarts from its last saved frame.')
    p.add_argument('--continue-seed-offset', type=int, default=1000)
    args = p.parse_args()
    if len(args.seeds) < 2:
        p.error('Need at least two independent seeds for a standard error.')
    opts = vars(args)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(pot, ker, img, seed, opts, str(root)) for img in args.images for pot in args.potentials
            for ker in args.kernels for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        summaries = []
        for s in pool.map(run_one, jobs):
            summaries.append(s)
            print(f"done {s['images']}/{s['potential']}_{s['kernel']} seed {s['seed']}: "
                  f"T={s['T_mean']:.3f} U/N={s['U_per_particle']:.3f} D_GK={s['D_green_kubo']:.4f}", flush=True)
    ens = {img: {pot: {ker: aggregate(root, img, pot, ker, args.seeds) for ker in args.kernels}
                 for pot in args.potentials} for img in args.images}
    for pot in args.potentials:
        for img in args.images:
            if set(args.kernels) == {'A', 'B'}:
                plot_ab(ens[img][pot], pot, img, root/f'ab_{pot}_{img}.png', len(args.seeds))
        if len(args.images) == 2 and set(args.kernels) == {'A', 'B'}:
            plot_conventions({img: ens[img][pot] for img in args.images}, pot,
                             root/f'conventions_{pot}.png', len(args.seeds))
    keys = ['T_mean', 'U_per_particle', 'U_second_minus_first_half', 'T_second_minus_first_half',
            'D_green_kubo', 'D_msd', 'vacf_min', 'vacf_min_time', 'tau_T', 'rdf_peak', 'rdf_peak_r',
            'max_total_momentum', 'min_distance']
    report = dict(parameters={k: v for k, v in opts.items() if k not in ('workers', 'out')},
                  vccf_edges=VCCF_EDGES, sample_interval=args.stride*args.dt,
                  burn_in_time=args.burn*args.dt, production_time=args.steps*args.dt,
                  effective_burn_in_time=sorted({s['effective_burn_in_time'] for s in summaries}),
                  error_bars='standard error of the mean across independent seeds (ddof=1)',
                  table=table(summaries, keys))
    (root/'ensemble_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Saved ensemble results to {root}', flush=True)


if __name__ == '__main__':
    main()
