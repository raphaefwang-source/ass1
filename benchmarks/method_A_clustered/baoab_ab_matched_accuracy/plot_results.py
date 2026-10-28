"""Plot saved matched-frozen-accuracy evidence without running experiments.

Only this experiment's fresh timing files enter costs and ratios. PR10's
historical k=1 batch is deliberately not loaded. Full norm accuracy comes from
all registered geometries; curve-only timing points retain their smaller n.
"""
from __future__ import annotations
from matched_common import HERE,RESULTS,read_csv,write_csv,save,sha,js
from collections import defaultdict
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURES=HERE/'figures'
PHASES=[('Dynamic q setup','t_dynamic_setup_s'),('LJ forces','t_LJ_forces_s'),
        ('Gaussian RNG','t_rng_s'),('Complete OU','t_ou_wall_s'),('Other BAOAB','t_baoab_other_s')]
TARGET_COLORS={1e-4:'#228833',1e-6:'#ee7733',2e-8:'#aa3377'}

def phase_vector(row):
    """Disjoint complete-step phases; nested FFT/eigh/reorth are excluded."""
    values=np.asarray([float(row[key]) for _,key in PHASES])
    if not np.isfinite(values).all() or values.min() < -1e-9:
        raise ValueError('Invalid or overlapping disjoint phase clocks')
    if not np.isclose(values.sum(),float(row['t_baoab_s']),rtol=1e-10,atol=1e-10):
        raise ValueError('Major phase sum does not equal complete moving BAOAB wall')
    return values*1000

def curve_k(row,target_k):
    """B keeps k=1; its x coordinate identifies its corresponding A control."""
    eps=row.get('epsilon_frozen')
    if eps is not None:
        return int(target_k[float(eps)])
    match=re.fullmatch(r'curve_k(\d+)_start\d+_[AB]',str(row['run']))
    if not match:raise ValueError('Unregistered cost-curve run '+str(row['run']))
    return int(match.group(1))

def block_summary(rows):
    if not rows:raise ValueError('Cannot summarize absent actual timing rows')
    blocks=defaultdict(list)
    for row in rows:
        key=row.get('round')
        if key is None:key=('start',int(row['initial_index']))
        blocks[str(key)].append(float(row['t_baoab_s'])*1000)
    bmeans=np.asarray([np.mean(v) for v in blocks.values()])
    return dict(n_steps=len(rows),n_blocks=len(blocks),
        mean_ms=float(np.mean([float(r['t_baoab_s'])*1000 for r in rows])),
        block_mean_min_ms=float(bmeans.min()),block_mean_max_ms=float(bmeans.max()))

def accuracy_summary(rows):
    out=[]
    for k in sorted({int(r['k']) for r in rows}):
        group=[r for r in rows if int(r['k'])==k]
        record=dict(k=k,n_geometries=len(group))
        for kind in ('mean','cov'):
            values=[float(r['e_'+kind]) for r in group]
            record['e_'+kind+'_min']=min(values);record['e_'+kind+'_max']=max(values)
        out.append(record)
    return out

def analysis_records(data):
    if isinstance(data,list):return data
    for key in ('targets','results','comparisons'):
        if isinstance(data.get(key),list):return data[key]
    raise ValueError('analysis.json must contain target/comparison records')

def save_figure(fig,stem):
    png=FIGURES/(stem+'.png');svg=FIGURES/(stem+'.svg')
    fig.savefig(png,dpi=180,bbox_inches='tight')
    fig.savefig(svg,bbox_inches='tight')
    plt.close(fig)
    return [png,svg]

def plot_accuracy(selection,errors):
    summaries=accuracy_summary(errors)
    if {r['k'] for r in summaries}!={int(k) for k in selection['ks']} or any(r['n_geometries']!=len(selection['states']) for r in summaries):
        raise ValueError('Incomplete registered full-COM accuracy grid')
    write_csv(FIGURES/'frozen_accuracy_plotted.csv',summaries)
    x=np.asarray([r['k'] for r in summaries])
    fig,axes=plt.subplots(1,2,figsize=(12,5.1))
    for ax,kind,title,color in zip(axes,('mean','cov'),('Conditional mean operator','Conditional covariance operator'),('#0077bb','#cc3311')):
        lo=np.asarray([r['e_'+kind+'_min'] for r in summaries]);hi=np.asarray([r['e_'+kind+'_max'] for r in summaries])
        ax.fill_between(x,lo,hi,color=color,alpha=.18,label='Registered-geometry range')
        ax.plot(x,hi,'o-',color=color,lw=2,label='Maximum actual full-COM error')
        ax.plot(x,hi[0]/x**2,'--',color='#888888',lw=1.1,label=r'$k^{-2}$ guide (not a pass prediction)')
        for target in selection['targets']:
            eps=float(target['epsilon_frozen'])
            ax.axhline(eps,color=TARGET_COLORS[eps],ls=':',lw=1.5,label=f'Target {eps:g}')
        ax.set_xscale('log',base=2);ax.set_yscale('log')
        ax.set_xticks(x,[str(int(k)) for k in x]);ax.set_xlabel('A internal splitting substeps k')
        ax.set_ylabel(r'Normalized operator 2-norm error');ax.set_title(title)
        ax.grid(True,which='both',alpha=.2)
        ax.set_ylim(bottom=min(float(lo.min())*.65,2e-10),top=2e-4)
    axes[0].legend(loc='lower left',fontsize=8,frameon=False)
    for target in selection['targets']:
        if target['k'] is None:continue
        k=int(target['k']);eps=float(target['epsilon_frozen']);row=next(r for r in summaries if r['k']==k)
        axes[1].scatter([k],[row['e_cov_max']],s=110,facecolors='none',edgecolors=TARGET_COLORS[eps],lw=2,zorder=6)
    fig.suptitle('Actual ideal frozen-OU operator errors vs internal substeps',fontweight='bold')
    fig.subplots_adjust(bottom=.19,top=.83,wspace=.29)
    fig.text(.5,.04,f"Full {selection['states'][0]['dimension_COM']}-D COM subspace; {len(selection['states'])} saved geometries; dt=0.005. All k shown were computed.",ha='center',fontsize=9,color='#444444')
    return save_figure(fig,'frozen_accuracy_vs_k')

def collect_curve(selection,main_rows,extra_rows):
    target_k={float(t['epsilon_frozen']):int(t['k']) for t in selection['targets'] if t['k'] is not None}
    groups=defaultdict(list)
    for source,rows in [('matched tiers',main_rows),('short curve controls',extra_rows)]:
        for row in rows:
            k=curve_k(row,target_k)
            if row['method']=='A' and int(row['k'])!=k:raise ValueError('Actual A timing k differs from registered target')
            if row['method']=='B' and int(row['k'])!=1:raise ValueError('B must remain one full OU call')
            groups[(k,row['method'],source)].append(row)
    records=[]
    for (k,method,source),rows in sorted(groups.items()):
        record=dict(k=k,method=method,source=source,**block_summary(rows))
        phases=np.mean([phase_vector(r) for r in rows],axis=0)
        for (label,key),value in zip(PHASES,phases):record[key+'_mean_ms']=float(value)
        records.append(record)
    return records,groups

def plot_cost_curve(selection,main_rows,extra_rows):
    records,groups=collect_curve(selection,main_rows,extra_rows)
    write_csv(FIGURES/'dynamic_cost_curve_plotted.csv',records)
    fig,ax=plt.subplots(figsize=(10.8,5.6))
    colors={'A':'#0077bb','B':'#cc3311'}
    for method in ('A','B'):
        rr=sorted([r for r in records if r['method']==method],key=lambda r:r['k'])
        ax.plot([r['k'] for r in rr],[r['mean_ms'] for r in rr],color=colors[method],lw=1.5,
                label='A(k): full moving BAOAB' if method=='A' else 'B: single full OU control for each A k')
        for r in rr:
            x=r['k'];y=r['mean_ms'];selected=r['source']=='matched tiers'
            ax.vlines(x,r['block_mean_min_ms'],r['block_mean_max_ms'],color=colors[method],lw=1.3)
            ax.scatter([x],[y],s=55,marker='o' if method=='A' else 's',
                facecolors=colors[method] if selected else 'white',edgecolors=colors[method],lw=1.6,zorder=5)
            if method=='A':ax.annotate(f"n={r['n_steps']}",(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xscale('log',base=2)
    ks=sorted({r['k'] for r in records});ax.set_xticks(ks,[str(k) for k in ks])
    ax.set_xlabel('A internal substeps k (B remains one full OU call)')
    ax.set_ylabel('Complete moving BAOAB wall time per outer step (ms)')
    ax.set_title('Fresh dynamic cost vs k, including one q-dependent setup per outer step',fontweight='bold')
    ax.grid(True,axis='y',alpha=.2);ax.legend(loc='upper left',frameon=False,fontsize=9)
    ax.set_ylim(bottom=0,top=max(r['block_mean_max_ms'] for r in records)*1.14)
    fig.subplots_adjust(bottom=.22,top=.85)
    selected_counts=sorted({r['n_steps'] for r in records if r['source']=='matched tiers'})
    extra_counts=sorted({r['n_steps'] for r in records if r['source']=='short curve controls'})
    sc='/'.join(map(str,selected_counts));ec='/'.join(map(str,extra_counts))
    fig.text(.5,.07,f'Filled: matched-tier batches ({sc} steps/method/k). Open: short curve controls ({ec} steps/method/k).',ha='center',fontsize=9)
    fig.text(.5,.025,'Whiskers: range of round/start block means, not CI. Every point is actual moving-state timing; no historical PR10 ratio.',ha='center',fontsize=8.5,color='#444444')
    return save_figure(fig,'dynamic_cost_vs_k'),records,groups

def plot_matched(selection,main_rows,rounds,analysis):
    records=analysis_records(analysis)
    targets=sorted([t for t in selection['targets'] if t['k'] is not None],key=lambda t:float(t['epsilon_frozen']),reverse=True)
    fig,(ax,ratio_ax)=plt.subplots(1,2,figsize=(13,5.8),gridspec_kw={'width_ratios':[1.7,1.]})
    colors=['#4477aa','#ee7733','#228833','#cc6677','#bbbbbb']
    plotted=[];width=.34
    for ti,target in enumerate(targets):
        eps=float(target['epsilon_frozen']);k=int(target['k'])
        for mi,method in enumerate(('A','B')):
            rr=[r for r in main_rows if r['method']==method and float(r['epsilon_frozen'])==eps]
            phases=np.mean([phase_vector(r) for r in rr],axis=0);summary=block_summary(rr)
            x=ti+(mi-.5)*width;bottom=0.
            for pi,((label,key),value) in enumerate(zip(PHASES,phases)):
                ax.bar(x,value,width*.94,bottom=bottom,color=colors[pi],label=label if ti==0 and mi==0 else None)
                bottom+=value
            ax.vlines(x,summary['block_mean_min_ms'],summary['block_mean_max_ms'],color='#222222',lw=1.3)
            ax.plot([x-.035,x+.035],[summary['block_mean_min_ms']]*2,color='#222222',lw=1.3)
            ax.plot([x-.035,x+.035],[summary['block_mean_max_ms']]*2,color='#222222',lw=1.3)
            ax.annotate(f"{method}\n{bottom:.1f}",(x,max(bottom,summary['block_mean_max_ms'])),
                xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
            plotted.append(dict(epsilon_frozen=eps,k=k,method=method,**summary,
                **{key+'_mean_ms':float(value) for (_,key),value in zip(PHASES,phases)}))
        record=next(r for r in records if float(r['epsilon_frozen'])==eps)
        ratio=float(record['ratio']);lo=float(record['ratio_ci_low']);hi=float(record['ratio_ci_high'])
        if not (np.isfinite([ratio,lo,hi]).all() and min(ratio,lo,hi)>0 and lo<=hi):
            raise ValueError('Invalid paired-round uncertainty record')
        ratio_ax.hlines(ti,lo,hi,color='#0077bb',lw=2)
        ratio_ax.plot(ratio,ti,'o',color='#0077bb')
        ratio_ax.annotate(f"{ratio:.3g} [{lo:.3g}, {hi:.3g}]\n{record['verdict']}",(hi,ti),
            xytext=(7,0),textcoords='offset points',va='center',fontsize=8)
    ax.set_xticks(range(len(targets)),[f"eps={float(t['epsilon_frozen']):g}\nA k={int(t['k'])}" for t in targets])
    ax.set_ylabel('Complete moving BAOAB wall time per step (ms)')
    ax.set_title('Measured complete-step phases')
    ax.grid(True,axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.set_ylim(0,max(r['block_mean_max_ms'] for r in plotted)*1.20)
    ax.legend(frameon=False,fontsize=8,loc='upper left')
    ratio_ax.axvline(1.,color='#555555',ls='--',lw=1.)
    ratio_ax.set_yticks(range(len(targets)),[f"{float(t['epsilon_frozen']):g}\nk={int(t['k'])}" for t in targets])
    ratio_ax.invert_yaxis();ratio_ax.set_xlabel('Paired complete-step time ratio A/B')
    ratio_ax.set_title('Paired-round ratio and uncertainty')
    ratio_ax.grid(True,axis='x',alpha=.2)
    hi_max=max(float(r['ratio_ci_high']) for r in records)
    ratio_ax.set_xlim(0,hi_max*1.55)
    fig.suptitle('Frozen-transition-accuracy-matched dynamic cost',fontweight='bold')
    fig.subplots_adjust(bottom=.24,top=.82,wspace=.33,right=.98)
    fig.text(.5,.09,'Left whiskers: round-mean range (not CI). Right intervals: paired-round uncertainty from analysis.json.',ha='center',fontsize=9)
    fig.text(.5,.04,'Fresh same-experiment batches only. Includes dynamic setup, LJ, RNG and all selected A substeps; not equal physical-observable accuracy.',ha='center',fontsize=8.6,color='#444444')
    write_csv(FIGURES/'matched_dynamic_total_plotted.csv',plotted)
    return save_figure(fig,'matched_accuracy_dynamic_total')

def plot_dynamic_phases(records):
    rr=sorted([r for r in records if r['method']=='A'],key=lambda r:r['k'])
    fig,ax=plt.subplots(figsize=(10,5.6));bottom=np.zeros(len(rr));colors=['#4477aa','#ee7733','#228833','#cc6677','#bbbbbb']
    for j,(label,key) in enumerate(PHASES):
        values=np.asarray([r[key+'_mean_ms'] for r in rr]);ax.bar(np.arange(len(rr)),values,bottom=bottom,color=colors[j],label=label);bottom+=values
    ax.set_xticks(range(len(rr)),[f"k={r['k']}\nn={r['n_steps']}" for r in rr])
    ax.set_ylabel('Complete moving BAOAB wall time per A step (ms)')
    ax.set_title('One dynamic setup plus k complete A OU substeps: actual phase costs',fontweight='bold')
    ax.grid(True,axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False,loc='upper left',fontsize=9)
    fig.subplots_adjust(bottom=.20,top=.85)
    fig.text(.5,.04,'Disjoint major phases only. FFT, edge eigensolve and reorthogonalization are nested, never added twice.',ha='center',fontsize=8.5,color='#444444')
    return save_figure(fig,'dynamic_phases_vs_k')

def make_plots():
    FIGURES.mkdir(parents=True,exist_ok=True)
    sources=[RESULTS/'accuracy/frozen_selection.json',RESULTS/'accuracy/frozen_errors.csv',
             RESULTS/'timing_steps.csv',RESULTS/'timing_rounds.csv',RESULTS/'cost_curve_extra_steps.csv',RESULTS/'analysis.json']
    missing=[str(p) for p in sources if not p.is_file()]
    if missing:raise FileNotFoundError('Actual completed evidence required: '+', '.join(missing))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none'})
    selection=js(sources[0]);errors=read_csv(sources[1]);main=read_csv(sources[2]);rounds=read_csv(sources[3]);extra=read_csv(sources[4]);analysis=js(sources[5])
    figures=plot_accuracy(selection,errors)
    pair,curve,groups=plot_cost_curve(selection,main,extra);figures.extend(pair)
    figures.extend(plot_matched(selection,main,rounds,analysis));figures.extend(plot_dynamic_phases(curve))
    plotted=list(FIGURES.glob('*_plotted.csv'))
    manifest=dict(input_sha256={str(p.relative_to(HERE)):sha(p) for p in sources},
        figures=[dict(file=str(p.relative_to(HERE)),sha256=sha(p)) for p in figures],
        plotted_csv_sha256={str(p.relative_to(HERE)):sha(p) for p in plotted},
        scope='Actual full-COM norm accuracy and fresh fully dynamic timing; PR10 historical batch excluded',
        cost_curve_counts=[{k:r[k] for k in ('k','method','source','n_steps','n_blocks')} for r in curve],
        uncertainty='Cost-curve and total-bar whiskers show block-mean range, not CI; paired A/B interval comes directly from analysis.json',
        phase_accounting='setup/LJ/RNG/OU/other disjoint; FFT/eigh/reorth nested and not added twice')
    save(FIGURES/'figure_manifest.json',manifest)
    print('Saved',len(figures)//2,'PNG/SVG pairs and source/plot data SHA manifest',flush=True)

if __name__=='__main__':make_plots()
