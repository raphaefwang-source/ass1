"""Publication PNG/SVG figures from the saved same-batch evidence only."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_generator import HERE, read, timing_summary

FIGURES = HERE / 'figures'
COLORS = {'A': '#0072B2', 'B': '#D55E00', 'slow': '#0072B2', 'fast': '#D55E00', 'random': '#009E73',
          'input_p': '#CC79A7', 'input_z': '#E69F00',
          'uniform': '#0072B2', 'moderate': '#D55E00', 'strong': '#009E73'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'savefig.dpi': 300, 'axes.grid': True,
                     'grid.alpha': .18, 'figure.constrained_layout.use': True})


def save(fig, stem):
    FIGURES.mkdir(exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(FIGURES / (stem + '.' + ext), bbox_inches='tight')
    plt.close(fig)


def representative(rows):
    candidates = [r for r in rows if r.get('family') == 'uniform' and r.get('seed') == 2 and r.get('epsilon') == 1e-4]
    candidates = candidates or rows
    return max(candidates, key=lambda r: r['N'])['config'] if candidates else None


def rank_plot():
    rows = read('rank_errors.csv')
    config = representative(rows)
    if not config:
        return
    rows = [r for r in rows if r['config'] == config and r['alpha'] == 1.]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey=True)
    for axis, (operator, kind) in zip(axes.flat, [('full', 'damp'), ('full', 'noise'), ('long', 'damp'), ('long', 'noise')]):
        subset = [r for r in rows if r['operator'] == operator and r['kind'] == kind]
        for direction in ('slow', 'fast', 'random', 'input_p', 'input_z'):
            rr = sorted([r for r in subset if r['probe'] == direction], key=lambda r: r['rank'])
            if rr:
                axis.semilogy([r['rank'] for r in rr], [max(r['error'], 1e-16) for r in rr],
                              'o-', ms=4, color=COLORS[direction], label=direction)
        if subset:
            axis.axhline(subset[0].get('tolerance', 1e-5), color='.35', ls='--', lw=1, label='selection threshold')
            axis.set_title(f'{operator} operator: {kind}; tau={subset[0]["dt"]:.3g}')
        axis.set_xlabel('Lanczos rank (requested)')
        if subset:
            axis.set_xticks(sorted({int(r['rank']) for r in subset}))
        axis.set_ylabel('Relative action error')
        axis.set_ylim(5e-17, 2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'Deterministic matrix-function action errors — {config}, alpha=1\nDense same-discretization target; registered probe directions')
    save(fig, 'rank_vs_error')


def fdt_plot():
    rows = [r for r in read('mc.csv') if r.get('family') == 'uniform' and r.get('seed') == 2 and r['direction'] != 'COM']
    if not rows:
        return
    Ns = sorted({int(r['N']) for r in rows})
    keys = [(N, direction) for N in Ns for direction in ('slow', 'fast', 'random')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axis, (phase, method) in zip(axes.flat, [('conditional', 'A'), ('conditional', 'B'), ('equilibrium', 'A'), ('equilibrium', 'B')]):
        subset = [r for r in rows if r['phase'] == phase and r['method'] == method]
        for i, (N, direction) in enumerate(keys):
            rr = [r for r in subset if r['N'] == N and r['direction'] == direction]
            if not rr:
                continue
            row = rr[0]
            target = row['exact_variance']
            if target <= 1e-14:
                continue
            y = row['variance'] / target
            axis.errorbar(i, y, yerr=[[max(0., y - row['ci_low'] / target)], [max(0., row['ci_high'] / target - y)]],
                          color=COLORS[direction], fmt='o', ms=4, capsize=2)
            axis.plot(i, row['ideal_method_variance'] / target, marker='_', ms=10, color='black')
        axis.axhline(1., color='.35', ls='--', lw=1)
        axis.set_title(f'{phase}; method {method}')
        axis.set_ylabel('Variance / exact-full target')
        axis.set_xticks(range(len(keys)), [f'{N}\n{d[0].upper()}' for N, d in keys], fontsize=8)
        axis.set_xlabel('N and direction (S=slow, F=fast, R=random)')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker='o', color=COLORS[d], ls='', label=d) for d in ('slow', 'fast', 'random')] +
                       [Line2D([], [], marker='_', color='black', ls='', label='ideal method target'),
                        Line2D([], [], ls='--', color='.35', label='exact-full target')],
               loc='outside lower center', ncol=5, frameon=False, fontsize=9)
    fig.suptitle('Actual finite-Krylov OU samples; pointwise approximate 95% variance intervals\nIndependent draws; no simultaneous-coverage or exact-FDT claim')
    save(fig, 'directional_fdt')


def splitting_plot():
    rows = [r for r in read('splitting.csv') if r.get('epsilon') == 1e-4 and r.get('seed') == 2 and r['alpha'] == 1.]
    if not rows:
        return
    N = max(r['N'] for r in rows)
    rows = [r for r in rows if r['N'] == N]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, metric in zip(axes, ('mean_operator_error', 'cov_operator_error')):
        for family in ('uniform', 'moderate', 'strong'):
            rr = sorted([r for r in rows if r['family'] == family], key=lambda r: r['substeps'])
            if rr:
                axis.loglog([1/r['substeps'] for r in rr], [max(r[metric], 1e-16) for r in rr],
                            'o-', label=family, color=COLORS[family], ms=4)
        axis.axhline(1e-3, color='.35', ls='--', lw=1, label='selection tolerance')
        axis.set_xlabel('Internal step / physical dt (= 1/k)')
        steps = sorted({int(r['substeps']) for r in rows})
        axis.set_xticks([1/k for k in steps], ['1' if k == 1 else f'1/{k}' for k in steps])
        axis.minorticks_off()
        axis.set_ylabel('Spectral norm error vs exact full OU')
        axis.set_title('Conditional mean' if metric.startswith('mean') else 'Conditional covariance')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Observed ideal-A splitting convergence — N={int(N)}, seed 2, alpha=1\nExact invariant substeps; finite Krylov and MC error excluded')
    save(fig, 'splitting_convergence')


def timing_plot():
    summary, paired = timing_summary()
    if not summary:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for method in ('A', 'B'):
        Ns = sorted({int(r['N']) for r in summary if r['method'] == method})
        med, low, high = [], [], []
        for N in Ns:
            values = np.array([r['median_total_s'] * 1000 for r in summary if r['N'] == N and r['method'] == method])
            med.append(np.median(values)); low.append(values.min()); high.append(values.max())
        med, low, high = map(np.array, (med, low, high))
        axes[0].errorbar(Ns, med, yerr=np.vstack((med-low, high-med)), fmt='o-', capsize=3,
                         color=COLORS[method], label=f'Method {method}')
    axes[0].set_xscale('log', base=2); axes[0].set_yscale('log')
    Ns = sorted({int(r['N']) for r in summary})
    axes[0].set_xticks(Ns, [str(N) for N in Ns])
    axes[0].set_xlabel('Particles N'); axes[0].set_ylabel('Total frozen step wall time (ms)')
    axes[0].legend()
    for family in ('uniform', 'moderate', 'strong'):
        rr = [r for r in paired if r['family'] == family]
        axes[1].scatter([r['N'] for r in rr], [r['A_over_B'] for r in rr], label=family,
                        color=COLORS[family], alpha=.75, s=28)
    axes[1].axhline(1., color='.35', ls='--', lw=1)
    axes[1].set_xscale('log', base=2); axes[1].set_yscale('log')
    axes[1].set_xticks(Ns, [str(N) for N in Ns])
    axes[1].set_xlabel('Particles N'); axes[1].set_ylabel('A / B total time (per-configuration medians)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Fresh interleaved matrix-free timings, alpha=1, epsilon=1e-4\nSeparate rank selection + ideal splitting tolerance; bars show configuration range')
    save(fig, 'equal_accuracy_total_time')


def phase_plot():
    rows = read('timing_raw.csv')
    if not rows:
        return
    phases = [('t_pairOU_forward', 'near pair OU: forward', '#56B4E9'),
              ('t_pairOU_reverse', 'near pair OU: reverse', '#0072B2'),
              ('t_long_lanczos_damp', 'long Lanczos: damping', '#009E73'),
              ('t_long_lanczos_noise', 'long Lanczos: noise', '#E69F00'),
              ('t_full_lanczos_damp', 'full Lanczos: damping', '#D55E00'),
              ('t_full_lanczos_noise', 'full Lanczos: noise', '#CC79A7'),
              ('other', 'RNG, projection and driver overhead', '#999999')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    legend_handles = {}
    for axis, method in zip(axes, ('A', 'B')):
        observations = []
        for N in (16, 64, 256):
            subset = [row for row in rows if row['N'] == N and row['method'] == method]
            configs = sorted({row['config'] for row in subset})
            medians = []
            for config in configs:
                rr = sorted([row for row in subset if row['config'] == config], key=lambda r: r['t_total_frozen_step'])
                medians.append(rr[len(rr)//2])
            if medians:
                medians.sort(key=lambda r: r['t_total_frozen_step'])
                observations.append(medians[len(medians)//2])
        bottom = np.zeros(len(observations))
        x = np.arange(len(observations))
        for key, label, color in phases:
            values = []
            for row in observations:
                if key == 'other':
                    value = row['t_total_frozen_step'] - sum(row.get(k, 0.) or 0. for k, _, _ in phases[:-1])
                    if value < -1e-8:
                        raise ValueError('Overlapping phase costs exceed observed total; cannot stack')
                    values.append(max(0., value)*1000)
                else:
                    values.append((row.get(key, 0.) or 0.)*1000)
            if any(values):
                bars = axis.bar(x, values, bottom=bottom, color=color, label=label, width=.65)
                legend_handles[label] = bars[0]
                bottom += values
        axis.set_xticks(x, [int(row['N']) for row in observations])
        axis.set_xlabel('Particles N')
        axis.set_ylabel('Observed total step and its phases (ms)')
        axis.set_title(f'Method {method}')
        for xx, total in zip(x, bottom):
            axis.annotate(f'{total:.3g}', (xx, total), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8)
        axis.margins(y=.15)
    fig.legend(list(legend_handles.values()), list(legend_handles), loc='outside lower center', ncol=3, frameon=False, fontsize=8)
    fig.suptitle('Observed phase costs at the median configuration and median repetition\nNon-overlapping phases only; FFT and reorthogonalization are included in Lanczos')
    save(fig, 'total_cost_phases')


if __name__ == '__main__':
    rank_plot(); fdt_plot(); splitting_plot(); timing_plot(); phase_plot()
    print('Wrote available evidence figures (PNG + SVG)')
