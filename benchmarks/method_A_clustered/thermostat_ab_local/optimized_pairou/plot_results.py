"""Saved fresh-batch figures; run only after formal timing completes."""
import os
os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_generator import HERE, RESULTS, METHODS, read

COLORS={'A-python':'#0072B2','A-compiled':'#009E73','B':'#D55E00'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','savefig.dpi':300,'axes.grid':True,'grid.alpha':.18,
                     'figure.constrained_layout.use':True})


def save(fig,stem):
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    for ext in ('png','svg'): fig.savefig(folder/(stem+'.'+ext),bbox_inches='tight')
    plt.close(fig)


def primary(name): return [r for r in read(name) if r['epsilon']==1e-4]


def time_lines(axis,rows,methods,metric):
    for method in methods:
        Ns=sorted({int(r['N']) for r in rows if r['method']==method})
        middle,low,high=[],[],[]
        for N in Ns:
            values=np.array([r['median_'+metric]*1000 for r in rows if r['method']==method and r['N']==N])
            middle.append(np.median(values));low.append(values.min());high.append(values.max())
        middle,low,high=map(np.array,(middle,low,high))
        axis.errorbar(Ns,middle,yerr=np.vstack((middle-low,high-middle)),fmt='o-',capsize=3,color=COLORS[method],label=method)
    Ns=sorted({int(r['N']) for r in rows})
    axis.set_xscale('log',base=2);axis.set_yscale('log');axis.set_xticks(Ns,[str(n) for n in Ns])
    axis.set_xlabel('Particles N');axis.legend(fontsize=8)


def ratio_line(axis,rows,comparison,label):
    rr=sorted([r for r in rows if r['comparison']==comparison],key=lambda r:r['N'])
    Ns=[int(r['N']) for r in rr];middle=np.array([r['ratio'] for r in rr])
    low=np.array([r['ci_low'] for r in rr]);high=np.array([r['ci_high'] for r in rr])
    axis.errorbar(Ns,middle,yerr=np.vstack((middle-low,high-middle)),fmt='o-',capsize=3,color=COLORS['A-compiled'])
    axis.axhline(1.,ls='--',color='.35',lw=1)
    axis.set_xscale('log',base=2);axis.set_yscale('log');axis.set_xticks(Ns,[str(n) for n in Ns])
    axis.set_xlabel('Particles N');axis.set_ylabel(label)


def main():
    summaries=primary('timing_summary.csv');ratios=primary('paired_aggregate_ratios.csv')
    if not summaries or not ratios: raise RuntimeError('Run report_generator.py after timing before plotting')
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    time_lines(axes[0],summaries,('A-python','A-compiled'),'t_near_total')
    axes[0].set_ylabel('Both near sweeps: median wall time (ms)')
    ratio_line(axes[1],ratios,'near_python_over_compiled','Paired Python / compiled near cost')
    fig.suptitle('Sequential pair OU acceleration, same edges and Gaussian draws\nFresh alpha=1, epsilon=1e-4; left: configuration range, right: block-bootstrap 95% CI')
    save(fig,'near_sweep_speedup')
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    time_lines(axes[0],summaries,METHODS,'t_total_step')
    axes[0].set_ylabel('Total frozen step: median wall time (ms)')
    ratio_line(axes[1],ratios,'total_compiled_over_B','Paired compiled A / full B total cost')
    fig.suptitle('Three-way equal-accuracy frozen-step comparison\nFresh batch only; unchanged PR8 dt/ranks/splitting; ratio < 1 favors compiled A')
    save(fig,'three_way_total_time')
    raw=primary('timing_raw.csv')
    phases=[('t_rng','RNG','#999999'),('t_near_forward','near forward','#56B4E9'),
            ('t_near_reverse','near reverse','#0072B2'),('t_long_damp','long damping','#009E73'),
            ('t_long_noise','long noise','#E69F00'),('t_full_damp','full damping','#D55E00'),
            ('t_full_noise','full noise','#CC79A7'),('other','other step overhead','#CCCCCC')]
    fig,axes=plt.subplots(1,3,figsize=(12,5));legend={}
    for axis,method in zip(axes,METHODS):
        chosen=[]
        for N in (16,64,256):
            subset=[r for r in raw if r['method']==method and r['N']==N]
            medians=[]
            for config in sorted({r['config'] for r in subset}):
                rr=sorted([r for r in subset if r['config']==config],key=lambda r:r['t_total_step'])
                medians.append(rr[len(rr)//2])
            medians.sort(key=lambda r:r['t_total_step']);chosen.append(medians[len(medians)//2])
        x=np.arange(len(chosen));bottom=np.zeros(len(chosen))
        for key,label,color in phases:
            values=[]
            for row in chosen:
                value=row[key] if key!='other' else row['t_total_step']-sum(row[k] for k,_,_ in phases[:-1])
                if value < -1e-8: raise ValueError('Phase times overlap; cannot stack')
                values.append(max(value,0.)*1000)
            if any(values):
                bars=axis.bar(x,values,bottom=bottom,width=.7,color=color,label=label);legend[label]=bars[0]
                bottom+=values
        axis.set_xticks(x,[int(r['N']) for r in chosen]);axis.set_xlabel('Particles N');axis.set_ylabel('Observed step cost (ms)')
        axis.set_title(method);axis.margins(y=.18)
        for xx,yy in zip(x,bottom):axis.annotate(f'{yy:.3g}',(xx,yy),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    fig.legend(list(legend.values()),list(legend),loc='outside lower center',ncol=4,frameon=False,fontsize=8)
    fig.suptitle('Non-overlapping phases of an actual median-config / median-repeat step\nFFT, reorthogonalization and Ritz eigensolve are included in Lanczos; not added again')
    save(fig,'three_way_phases')
    print('Wrote three PNG + SVG figures')


if __name__=='__main__':main()
