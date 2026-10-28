"""Plot only saved moving-particle evidence; no numerical experiment is run."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
FIGURES = HERE / 'figures'


def read(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key,value in row.items():
            try:
                row[key] = float(value)
            except (ValueError,TypeError):
                pass
    return rows


def phase(row, kind):
    """Disjoint subdivisions; nested FFT/eigh/reorth times are never added."""
    if kind == 'step':
        names = ['Dynamic q setup','LJ forces','Gaussian RNG','OU update','Other BAOAB']
        values = [row['t_dynamic_setup_s'],row['t_LJ_forces_s'],row['t_rng_s'],
                  row['t_ou_wall_s'],row['t_baoab_other_s']]
    else:
        known = sum(row[key] for key in ('t_operator_setup_s','t_edge_aggregate_s',
                                         't_edge_constructor_s','t_coefficient_setup_s'))
        op_known = sum(row[key] for key in ('t_near_neighbor_images_s','t_near_weights_degree_s',
                                           't_mesh_assignment_s','t_mesh_degree_s'))
        names = ['Near image enumeration','Near weights / degree','Mesh assignment','Mesh degree',
                 'Operator overhead','Aggregate 3x3 edges','Edge constructor (includes eigh)',
                 'Fresh coefficients','Other setup']
        values = [row['t_near_neighbor_images_s'],row['t_near_weights_degree_s'],
                  row['t_mesh_assignment_s'],row['t_mesh_degree_s'],
                  row['t_operator_setup_s']-op_known,row['t_edge_aggregate_s'],
                  row['t_edge_constructor_s'],row['t_coefficient_setup_s'],
                  row['t_dynamic_setup_s']-known]
    if min(values) < -1e-7:
        raise ValueError('Timing phases overlap or do not match total: '+str(values))
    return names,np.asarray(values)*1000


def save(fig, stem):
    fig.savefig(FIGURES/f'{stem}.png',dpi=180,bbox_inches='tight')
    fig.savefig(FIGURES/f'{stem}.svg',bbox_inches='tight')
    plt.close(fig)


def make_plots():
    FIGURES.mkdir(exist_ok=True)
    formal = RESULTS/'timing_steps.csv'
    source = formal if formal.exists() else RESULTS/'production_steps.csv'
    rows = read(source)
    if not rows:
        raise ValueError('No actual moving-particle timing rows')
    methods = ['A','B']
    groups = [[row for row in rows if row['method']==method] for method in methods]
    nrounds = len(set(row.get('round',row.get('initial_index')) for row in rows))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    plotted = []
    colors = plt.get_cmap('tab20').colors
    for kind,stem,title,total_key in [
            ('setup','dynamic_setup','Fresh position-dependent setup at every BAOAB midpoint','t_dynamic_setup_s'),
            ('step','baoab_total_time','Complete moving-particle BAOAB cost (N=256, dt=0.005)','t_baoab_s')]:
        fig,ax = plt.subplots(figsize=(9.6,5.7))
        phases = [np.array([phase(row,kind)[1] for row in group]) for group in groups]
        names = phase(rows[0],kind)[0]
        means = np.array([values.mean(axis=0) for values in phases])
        bottom = np.zeros(2)
        for j,name in enumerate(names):
            ax.bar(np.arange(2),means[:,j],bottom=bottom,label=name,color=colors[j*2 % len(colors)],
                   edgecolor='white',linewidth=.4,width=.6)
            for i,method in enumerate(methods):
                plotted.append(dict(figure=stem,method=method,phase=name,mean_ms=means[i,j],steps=len(groups[i])))
            bottom += means[:,j]
        total_means = [np.mean([row[total_key] for row in group])*1000 for group in groups]
        block_means = []
        for group in groups:
            blocks = sorted(set(row.get('round',row.get('initial_index')) for row in group))
            block_means.append([np.mean([row[total_key] for row in group
                if row.get('round',row.get('initial_index'))==block])*1000 for block in blocks])
        lower = np.array(total_means)-np.array([min(values) for values in block_means])
        upper = np.array([max(values) for values in block_means])-np.array(total_means)
        # All round blocks have equal length in the requested 6x40 protocol.
        ax.errorbar(np.arange(2),total_means,yerr=np.maximum(0.,np.array([lower,upper])),
                    fmt='none',ecolor='#202020',capsize=5,lw=1.3,label='Round-mean range (not CI)')
        for i,total in enumerate(total_means):
            ax.annotate(f'{total:.2f} ms',xy=(i,max(block_means[i])),xytext=(0,9),
                        textcoords='offset points',ha='center',weight='bold')
        ax.set_xticks([0,1],['A: compiled pair + long OU','B: full OU'])
        ax.set_ylabel('Measured wall time per moving step (ms)')
        ax.set_title(title,loc='left',pad=15,weight='bold')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.set_ylim(0,max(max(values) for values in block_means)*1.2)
        ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,fontsize=9)
        fig.text(.09,.01,f'Actual saved data: {nrounds} blocks, {len(groups[0])} A steps + {len(groups[1])} B steps. '
                 'Parameter tables reused; q-dependent data rebuilt every step.',fontsize=8,color='#444444')
        fig.tight_layout(rect=(0,.04,1,1))
        save(fig,stem)

    production = RESULTS/'production_steps.csv'
    if production.exists():
        trajectories = read(production)
        fig,axes = plt.subplots(2,2,figsize=(11.5,7.5),sharex=True)
        items = [('Tkin','Kinetic temperature'),('U_per_particle','Potential energy per particle'),
                 ('min_pair_distance','Minimum pair distance'),('momentum_norm','Total momentum norm')]
        seed_colors = ['#2166ac','#b2182b','#4d9221']
        for ax,(key,label) in zip(axes.flat,items):
            for index in sorted(set(int(row['initial_index']) for row in trajectories)):
                for method in methods:
                    group = [row for row in trajectories if row['method']==method and int(row['initial_index'])==index]
                    ax.plot([row['time'] for row in group],[row[key] for row in group],
                            color=seed_colors[index%3],ls='-' if method=='A' else '--',lw=1.1,
                            label=f'Start {index+1}, {method}')
            if key=='Tkin':
                ax.axhline(1.,color='gray',lw=.7,alpha=.7)
            if key=='momentum_norm':
                ax.set_yscale('symlog',linthresh=1e-14)
            ax.set_ylabel(label);ax.grid(alpha=.17)
        for ax in axes[-1]:
            ax.set_xlabel('Simulated time')
        axes[0,0].legend(ncol=2,frameon=False,fontsize=8)
        fig.suptitle('Stage 1 moving trajectories: correctness / performance pilot only',x=.08,ha='left',weight='bold')
        fig.tight_layout(rect=(0,0,1,.96))
        save(fig,'trajectory_observables')
    with (RESULTS/'figure_plotted_phases.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(plotted[0]));writer.writeheader();writer.writerows(plotted)
    manifest = dict(source=str(source.relative_to(HERE)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    rows=len(rows),scope='Saved true moving-particle steps; no frozen replay',
                    error_bars='Range of round/block mean times, not confidence intervals',
                    accounting='Setup/LJ/RNG/OU/other disjoint. FFT/eigh/reorth are nested and not added twice.',
                    figures=[str(p.relative_to(HERE)) for p in sorted(FIGURES.glob('*')) if p.suffix in ('.png','.svg')])
    (RESULTS/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    make_plots()
