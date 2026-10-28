"""Figures from this run's saved local evidence; no numerical experiments.

Run after post_validation and joint_analyze. All plotted rows, source SHAs,
and PNG/SVG hashes are saved next to the figures. Historical timings are not
inputs. Candidate sensitivity is a confounded joint-family comparison.
"""
from __future__ import annotations
from joint_common import *
from collections import defaultdict

FIGURES=HERE/'figures'


def load_bundle():
    paths=dict(analysis=RESULTS/'analysis.json',exploration=RESULTS/'exploratory_done.json',
        heads=RESULTS/'heads_done.json',post=RESULTS/'post_validation.json',
        spatial=RESULTS/'spatial/spatial_selection.json',thermo=RESULTS/'thermo/thermo_selection.json',
        ranks=RESULTS/'thermo/rank_selection.json')
    bundle={k:js(p) for k,p in paths.items()}
    bundle['ideal_rows']=read_csv(RESULTS/'thermo/ideal_errors.csv')
    bundle['head_steps']=read_csv(RESULTS/'heads_steps.csv')
    bundle['sources']={str(p.relative_to(HERE)):sha(p) for p in paths.values()}
    for p in [RESULTS/'thermo/ideal_errors.csv',RESULTS/'heads_steps.csv',RESULTS/'pareto.csv']:
        bundle['sources'][str(p.relative_to(HERE))]=sha(p)
    bundle['pareto']=read_csv(RESULTS/'pareto.csv')
    return bundle


def field(row,*names):
    for name in names:
        if name in row and row[name] is not None:return row[name]
    raise KeyError('Missing field, expected one of '+str(names))


def comparison_rows(bundle):
    """Normalize the analysis table, retaining original group/config identity."""
    groups={g['group_id']:g for g in bundle['heads']['groups']};rows=[]
    for item in bundle['analysis']['comparisons']:
        row=dict(item);group=groups[row['group_id']]
        for m in ('A','B'):
            row[m+'_config']=group[m]
            row[m+'_mean_ms']=float(field(row,m+'_mean_ms',m+'_BAOAB_mean_ms',m+'_full_BAOAB_mean_ms'))
            row[m+'_wall_per_simulated_time']=float(field(row,m+'_wall_per_simulated_time'))
        row['A_over_B']=float(field(row,'A_over_B','ratio','ratio_A_over_B'))
        row['CI_low']=float(field(row,'CI_low','ci_low','ratio_ci_low'))
        row['CI_high']=float(field(row,'CI_high','ci_high','ratio_ci_high'))
        row['comparison_type']=group['comparison_type'];row['epsilon_frozen']=group['epsilon_frozen']
        row['reuse_group_id']=group.get('reuse_group_id')
        rows.append(row)
    return rows


def candidate_rows(bundle):
    """Measured 3D Pareto coordinates, partitioned within method/accuracy tier."""
    rows=[];registry={str(r.get('config_id')):r for r in bundle['pareto']}
    for ranking in bundle['exploration']['rankings']:
        c=ranking['certificate'];p=ranking['params']
        rec=dict(config_id=ranking['config_id'],spatial_id=ranking['spatial_id'],
            method=ranking['method'],epsilon_frozen=ranking['epsilon_frozen'],
            cost=ranking['measured_wall_per_simulated_time'],
            spatial_error=c['max_spatial_error'],
            frozen_error=max(c['max_ideal_physical_mean'],c['max_ideal_physical_cov']),
            k=ranking['k'],r_d=ranking['r_d'],r_n=ranking['r_n'],actual_steps=ranking['actual_steps'],
            xi=p['xi'],sR=p['sR'],sF=p['sF'],M=p['M'],p=p['p'])
        if rec['config_id'] not in registry:raise ValueError('Pareto registry omits measured candidate '+rec['config_id'])
        rows.append(rec)
    for rec in rows:
        peers=[r for r in rows if (r['method'],r['epsilon_frozen'])==(rec['method'],rec['epsilon_frozen'])]
        keys=('cost','spatial_error','frozen_error')
        rec['empirical_pareto']=not any(all(r[k]<=rec[k] for k in keys) and any(r[k]<rec[k] for k in keys) for r in peers)
    return rows


def aggregate_ideal(bundle):
    groups=defaultdict(list)
    for row in bundle['ideal_rows']:groups[(row['setting_id'],row['method'],int(row['k']))].append(row)
    result=[]
    for (sid,method,k),rr in sorted(groups.items()):
        result.append(dict(setting_id=sid,method=method,k=k,states=len(rr),
            **{name:max(r[name] for r in rr) for name in ('physical_mean_bound','physical_cov_bound',
                'own_mean_relative','own_cov_relative','spatial_transition_mean_relative','spatial_transition_cov_relative')}))
    return result


def phase_rows(bundle):
    groups={g['group_id']:g for g in bundle['heads']['groups']};out=[]
    for gid,group in groups.items():
        source=group.get('reuse_group_id',gid)
        for method in ('A','B'):
            rr=[r for r in bundle['head_steps'] if r['group_id']==source and r['method']==method]
            if not rr:raise ValueError('No actual dynamic rows for '+gid+' '+method)
            avg=lambda name:float(np.mean([r[name] for r in rr]))*1000
            total=avg('t_baoab_s');setup=avg('t_dynamic_setup_s');ou=avg('t_ou_loop_s');lj=avg('t_LJ_forces_s')
            overhead=total-setup-ou-lj
            # Tiny clock roundoff can make a residual negative by a few ns.
            if overhead < -1e-5:raise ValueError('Overlapping dynamic phase clocks')
            out.append(dict(group_id=gid,source_group_id=source,comparison_type=group['comparison_type'],
                epsilon_frozen=group['epsilon_frozen'],method=method,steps=len(rr),
                setup_ms=setup,OU_including_RNG_ms=ou,LJ_ms=lj,integration_other_ms=max(overhead,0),total_ms=total))
    return out


def short_setting(params):return f"xi={params['xi']:g}; M{int(params['M'])}/p{int(params['p'])}"
def group_label(row):return ('Independent' if row['comparison_type']=='independent_optimum' else 'Shared baseline')+'\n'+f"eps={row['epsilon_frozen']:g}"


def generate_figures(bundle=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    bundle=load_bundle() if bundle is None else bundle;FIGURES.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'figure.dpi':130,'savefig.dpi':180,'svg.fonttype':'none'})
    registry=[]
    def emit(fig,name,rows,caption):
        if fig.get_layout_engine() is None:fig.tight_layout()
        files=[]
        for suffix in ('png','svg'):
            p=FIGURES/f'{name}.{suffix}';fig.savefig(p,bbox_inches='tight');files.append(dict(file=str(p.relative_to(HERE)),sha256=sha(p)))
        csv_path=FIGURES/f'{name}_plotted.csv';write_csv(csv_path,rows)
        registry.append(dict(name=name,files=files,rows_file=str(csv_path.relative_to(HERE)),rows_sha256=sha(csv_path),caption=caption))
        plt.close(fig)
    candidates=candidate_rows(bundle);comparisons=comparison_rows(bundle);ideal=aggregate_ideal(bundle)
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True,layout='constrained')
    all_errors=[r['frozen_error'] for r in candidates];norm=LogNorm(vmin=min(all_errors),vmax=max(all_errors))
    for ax,(eps,method) in zip(axes.flat,[(e,m) for e in TIERS for m in ('A','B')]):
        rr=[r for r in candidates if r['epsilon_frozen']==eps and r['method']==method]
        scatter=ax.scatter([r['spatial_error'] for r in rr],[r['cost'] for r in rr],c=[r['frozen_error'] for r in rr],norm=norm,cmap='viridis',s=100)
        for r in rr:
            if r['empirical_pareto']:ax.scatter(r['spatial_error'],r['cost'],s=220,facecolors='none',edgecolors='black',linewidths=1.2)
            offset=(4,-12) if r['xi']==.65 and int(r['M'])==32 else (4,6)
            ax.annotate(f"{r['xi']:g}/M{int(r['M'])}/p{int(r['p'])}",(r['spatial_error'],r['cost']),xytext=offset,textcoords='offset points',fontsize=7)
        ax.axvline(2e-7,color='firebrick',ls='--',lw=1);ax.set_xscale('log');ax.set_yscale('log')
        ax.margins(y=.14);ax.set_title(f'{method}: eps={eps:g}');ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Physical spatial error upper bound')
    for ax in axes[:,0]:ax.set_ylabel('Full BAOAB wall / simulated time')
    fig.colorbar(scatter,ax=axes.ravel().tolist(),label='Physical frozen max(mean, covariance)',pad=.02,shrink=.85)
    emit(fig,'accuracy_cost_pareto',candidates,'Exploratory measured candidates; circles are the empirical 3-objective front within method/tier. No CI or global-optimum claim.')
    fig,axes=plt.subplots(1,2,figsize=(12,5));x=np.arange(len(comparisons));w=.36
    for method,color,offset in [('A','#2878B5',-w/2),('B','#D55E00',w/2)]:
        axes[0].bar(x+offset,[r[method+'_mean_ms'] for r in comparisons],width=w,label=method,color=color)
    axes[0].set_xticks(x,[group_label(r) for r in comparisons]);axes[0].set_ylabel('Complete moving BAOAB ms / step');axes[0].legend()
    for j,r in enumerate(comparisons):
        axes[1].errorbar(r['A_over_B'],j,xerr=[[r['A_over_B']-r['CI_low']],[r['CI_high']-r['A_over_B']]],fmt='o',color='#333333',capsize=5)
        axes[1].annotate(str(r['verdict']),(r['CI_high'],j),xytext=(7,0),textcoords='offset points',va='center',fontsize=8)
    axes[1].axvline(1,color='firebrick',ls='--');axes[1].set_yticks(x,[group_label(r) for r in comparisons]);axes[1].invert_yaxis()
    axes[1].set_xlabel('Paired full BAOAB A/B ratio, 95% bootstrap CI')
    emit(fig,'independent_controlled_baoab',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in comparisons],
        'New local paired batches only; A/B>1 means B has lower wall time. Post-validation gates verdicts; starts/rounds define the sampled population.')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,param in zip(axes.flat,('xi','M','p','k')):
        for method,color in [('A','#2878B5'),('B','#D55E00')]:
            for eps,marker in [(TIERS[0],'o'),(TIERS[1],'^')]:
                rr=[r for r in candidates if r['method']==method and r['epsilon_frozen']==eps]
                ax.scatter([r[param] for r in rr],[r['cost'] for r in rr],s=80,c=color,marker=marker,label=f'{method}, eps={eps:g}',alpha=.85)
        ax.set_xlabel(param);ax.set_ylabel('Measured full BAOAB wall / simulated time');ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Joint candidate families: other parameters and ranks change too; not causal one-factor sensitivity',fontsize=10)
    emit(fig,'joint_parameter_sensitivity',candidates,'Confounded jointly varied families. Each point uses actual short moving BAOAB; complete setting identities are in the plotted CSV.')
    settings={s['setting_id']:s['params'] for s in bundle['spatial']['qualified_settings']}
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for sid,params in settings.items():
        rr=sorted([r for r in ideal if r['setting_id']==sid and r['method']=='A'],key=lambda r:r['k'])
        for ax,kind in zip(axes,('mean','cov')):
            ax.loglog([r['k'] for r in rr],[r[f'physical_{kind}_bound'] for r in rr],'-o',label=short_setting(params))
            ax.loglog([r['k'] for r in rr],[max(r[f'own_{kind}_relative'],1e-18) for r in rr],':',alpha=.65)
    for ax,kind in zip(axes,('mean','covariance')):
        for eps in TIERS:ax.axhline(eps,c='grey',ls='--',lw=.8)
        ax.set_xlabel('Actually tested complete A substeps k');ax.set_ylabel(f'Max {kind} COM operator error');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Solid: common physical reference bound; dotted: own discrete Gamma_h splitting error',fontsize=10)
    emit(fig,'frozen_splitting_vs_k',ideal,'Full 765D COM norms, maximum over five registered geometries. Only actually evaluated k are shown; early stopping is explicit in thermo_selection.json.')
    phases=phase_rows(bundle);fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(phases));bottom=np.zeros(len(phases))
    for key,label,color in [('setup_ms','q-dependent setup','#4C78A8'),('OU_including_RNG_ms','complete O loop incl. RNG','#F58518'),('LJ_ms','two LJ forces','#54A24B'),('integration_other_ms','integration/other residual','#B279A2')]:
        values=np.array([r[key] for r in phases]);ax.bar(x,values,bottom=bottom,label=label,color=color);bottom+=values
    ax.set_xticks(x,[group_label(r)+' '+r['method'] for r in phases],fontsize=8);ax.set_ylabel('Complete moving BAOAB mean ms / step');ax.legend(fontsize=8)
    emit(fig,'head_phase_costs',phases,'Disjoint outer clocks only. FFT, pair, eigensolve and reorthogonalization subclocks are nested diagnostics, never added again.')
    manifest=dict(figures=registry,source_sha256=bundle['sources'],historical_timings_used=False)
    save(FIGURES/'figure_manifest.json',manifest);return manifest


if __name__=='__main__':
    result=generate_figures();print('Wrote',len(result['figures']),'PNG+SVG figure pairs from saved local results')
