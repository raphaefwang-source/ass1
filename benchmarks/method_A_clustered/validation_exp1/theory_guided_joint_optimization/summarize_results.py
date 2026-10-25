#!/usr/bin/env python3
"""Generate requested plots and Discussion from persisted numerical data only."""
import hashlib
import itertools
import json
import platform
from pathlib import Path
import experiment as X
from experiment import E,C,np,threadpool_limits
import run_optimization as R
import analyze_optimization as A
import theory as B
from scipy.optimize import nnls
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def summary(search):
    timing=X.read(X.RESULTS/'final_timing.csv')
    validation=X.read(X.RESULTS/'final_validation.csv')
    out=[]
    for cid in sorted(set(t['candidate_id'] for t in timing)):
        ts=[t for t in timing if t['candidate_id']==cid]
        vs=[v for v in validation if v['candidate_id']==cid]
        assert len(ts)==len(vs)==9
        row=dict(search.byid[cid])
        row.update(E_G_max=max(v['E_G'] for v in vs),
                   feasible=max(v['E_G'] for v in vs)<=X.TOL,
                   lam_min_h=min(v['lam_min_h'] for v in vs),symmetry_max=max(v['sym_resid'] for v in vs),
                   translation_null_max=max(v['mom_opnorm_max'] for v in vs),
                   matrixfree_dense_max=max(v['mf_vs_dense'] for v in vs))
        assert row['lam_min_h']>0 and row['symmetry_max']<1e-12
        assert row['translation_null_max']<1e-12 and row['matrixfree_dense_max']<1e-10
        for key in ts[0]:
            if key.startswith('t_'): row[key]=float(np.median([t[key] for t in ts]))
        row['max_complete_iqr_rel']=max(t['t_complete_iqr_rel'] for t in ts)
        row['n_images']=float(np.median([t['n_images'] for t in ts]))
        row['n_samples_per_configuration_phase']=ts[0]['n_samples']
        out.append(row)
    out.sort(key=lambda r:r['t_complete_s'])
    X.write(X.HERE/'best_candidates.csv',[r for r in out if r['feasible']])
    X.write(X.RESULTS/'timed_controls_and_candidates.csv',out)
    free=next(r for r in out if r['feasible'])
    equal=next(r for r in out if r['feasible'] and abs(r['sR']-r['sF_req'])<1e-10)
    old=next(r for r in out if r['candidate_id']==X.ident(X.OLD))
    # Final A/B ranks use the common timing, rather than retaining a winner
    # selected by the earlier short diagnostic timing blocks.
    controls=json.loads((X.RESULTS/'strategy_controls.json').read_text())
    if not (X.RESULTS/'strategy_controls_before_common_timing.json').exists():
        X.jsave(X.RESULTS/'strategy_controls_before_common_timing.json',controls)
    plan=json.loads((X.RESULTS/'study_plan.json').read_text());base=plan['baseline']
    trades=X.read(X.HERE/'mesh_tradeoff.csv')
    same=[r for r in out if r['feasible'] and all(r[k]==base[k] for k in ('xi','sR','kc_shell_n'))]
    strict=[]
    for row in same:
        diagnostic=next(r for r in trades if r['M']==row['M'] and r['p']==row['p'])
        if diagnostic['E_M_max']<=.1*diagnostic['E_cutoff_max']:strict.append(row)
    assert same and strict
    controls={'A':min(strict,key=lambda r:r['t_complete_s'])['candidate_id'],
              'B':min(same,key=lambda r:r['t_complete_s'])['candidate_id']}
    X.jsave(X.RESULTS/'strategy_controls.json',controls)
    A.decompose(search,[r['candidate_id'] for r in (free,equal,old)],all_configs=True)
    reopt=[]
    for field in A.PARAMS:
        for value in sorted(set(r[field] for r in out if r['feasible'])):
            best=min([r for r in out if r['feasible'] and r[field]==value],key=lambda r:r['t_complete_s'])
            reopt.append(dict(parameter=field,value=value,candidate_id=best['candidate_id'],
                              t_complete_s=best['t_complete_s'],E_G_max=best['E_G_max'],
                              xi=best['xi'],sR=best['sR'],sF_req=best['sF_req'],M=best['M'],p=best['p']))
    X.write(X.RESULTS/'reoptimized_sensitivity.csv',reopt)
    return out,free,equal,old

def bounds(rows):
    out=[]
    for row in rows:
        par=X.fromrow(row)
        for cfg in X.configs():
            norm=X.normref(cfg['name']);G=X.reference(cfg['name'])
            lower=B.P.lambda_lower_bound(G,256)
            assert 0<lower<=norm*(1+1e-13)
            rst=B.P.RealSpaceTail(cfg['q'],cfg['L'],7/par['xi'])
            with B.physical_bound_kernel():
                prep=rst.prepare(par['xi']);fs=B.P.FourierShells(par['xi'],cfg['L'])
                rc,kc,_=E.cutoffs(par,cfg['L'])
                br=rst.eps_S(prep,rc)/norm;bf=fs.eps_F(kc,.75)/norm
                bm=B.P.eps_mesh(fs,kc,par['M'],par['p'],.75)/norm
                eq,relative=B.P.eps_quad_estimate(par['xi'],cfg['L'],kc,par['M'],par['p'],.75,fs)
            out.append(dict(candidate_id=row['candidate_id'],config=cfg['name'],B_R=br,B_F=bf,B_M=bm,
                            B_sum=br+bf+bm,quadrature_estimate=eq/norm,coefficient_relative_max=relative,
                            norm2=norm,lambda_lower_bound=lower,lower_bound_normalization_factor=norm/lower))
    X.write(X.RESULTS/'final_theory_bounds.csv',out)

def check_far_limits(rows):
    out=[];seen=set()
    for row in rows:
        xi=row['xi']
        for cfg in X.configs()[:2]:
            key=(cfg['name'],xi)
            if key in seen:continue
            seen.add(key)
            S,F,residual,kfar=A.far_split(cfg['name'],xi)
            op,info=X.build(cfg,X.pars(xi,9.,2.6,48,3));assert info['valid']
            S2=E.real_dense(op)
            k2,_=E.snap_kc(2*xi*7.5,cfg['L'])
            F2=E.fourier_dense(cfg['q'],cfg['L'],E.kernel(.2,xi,0.),k2)
            norm=X.normref(cfg['name'])
            dr=E.norm2_sym(S2-S)[0]/norm;df=E.norm2_sym(F2-F)[0]/norm
            assert max(dr,df)<1e-12
            out.append(dict(config=cfg['name'],xi=xi,sR_far_from=8.,sR_far_to=9.,
                            sF_far_from=kfar/(2*xi),sF_far_to=k2/(2*xi),
                            real_far_refinement_change=dr,fourier_far_refinement_change=df,
                            split_reference_residual=residual))
    X.write(X.RESULTS/'far_limit_convergence.csv',out)

def speed_bootstrap(new,baseline):
    raw=X.read(X.RESULTS/'final_timing_raw.csv')
    protocol=json.loads((X.RESULTS/'timing_protocol.json').read_text())
    rd,reps=protocol['rounds'],protocol['reps']
    def tensor(cid):
        ar=np.empty((9,rd,reps));seen=set()
        names=[c['name'] for c in X.configs()]
        for r in raw:
            if r['candidate_id']==cid and r['phase']=='complete':
                key=(names.index(r['config']),r['round'],r['rep'])
                assert key not in seen;seen.add(key);ar[key]=r['t_s']
        assert len(seen)==9*rd*reps
        return ar
    newar,oldar=tensor(new['candidate_id']),tensor(baseline['candidate_id'])
    rng=np.random.default_rng(20261027);n=5000
    ci=rng.integers(0,9,(n,9));ri=rng.integers(0,rd,(n,9,rd))
    idx=(ci[:,:,None,None],ri[:,:,:,None],np.arange(reps)[None,None,None,:])
    newmed=np.median(newar[idx],axis=(2,3));oldmed=np.median(oldar[idx],axis=(2,3))
    ratios=np.median(oldmed,axis=1)/np.median(newmed,axis=1)
    return dict(speedup=baseline['t_complete_s']/new['t_complete_s'],
                time_reduction_fraction=1-new['t_complete_s']/baseline['t_complete_s'],
                paired_configuration_and_round_bootstrap_95pct=np.percentile(ratios,[2.5,97.5]).tolist(),
                bootstrap_fraction_speedup_le_one=float(np.mean(ratios<=1)),bootstrap_repetitions=n,
                limitation='resamples configurations and paired round blocks; conditional on this machine/run')

def cost_model(search):
    data=X.read(X.RESULTS/'diagnostic_timing.csv');out=[]
    for phase in ('real','spread_gather','fft'):
        x=[];y=[]
        for r in data:
            row=search.byid[r['candidate_id']]
            feature={'real':r['n_images'],'spread_gather':256*row['p']**3,
                     'fft':row['M']**3*np.log(row['M']**3)}[phase]
            x.append(feature);y.append(r[f't_{phase}_s'])
        x=np.asarray(x);y=np.asarray(y);scale=np.median(x)
        coef,_=nnls(np.c_[x/scale,np.ones(len(x))],y)
        pred=coef[0]*x/scale+coef[1]
        out.append(dict(phase=phase,coefficient_s_per_work=coef[0]/scale,constant_s=coef[1],
                        R_squared=1-float(np.sum((pred-y)**2)/np.sum((y-y.mean())**2)),
                        median_absolute_relative_residual=float(np.median(np.abs(pred-y)/y)),n=len(y)))
    X.write(X.RESULTS/'cost_model.csv',out)
    return out

def heatmap(rows,xkey,ykey,filename,title):
    xs=sorted(set(r[xkey] for r in rows));ys=sorted(set(r[ykey] for r in rows))
    errors=np.full((len(ys),len(xs)),np.nan);times=errors.copy()
    for r in rows:
        if r.get('E_G_max') is not None:
            i,j=ys.index(r[ykey]),xs.index(r[xkey]);errors[i,j]=r['E_G_max'];times[i,j]=r['t_complete_s']*1e3
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,Z,label,norm in ((axes[0],errors,'Maximum spectral error over nine',LogNorm()),
                            (axes[1],times,'Complete matvec (ms)',None)):
        im=ax.imshow(Z,origin='lower',aspect='auto',norm=norm,cmap='viridis')
        fig.colorbar(im,ax=ax,label=label,shrink=.8)
        ax.set(xticks=range(len(xs)),xticklabels=[f'{v:g}' for v in xs],yticks=range(len(ys)),
               yticklabels=[f'{v:g}' for v in ys],xlabel=xkey,ylabel=ykey)
        ax.tick_params(axis='x',labelsize=8)
        if np.nanmin(errors)<X.TOL<np.nanmax(errors):
            ax.contour(np.ma.masked_invalid(errors),levels=[X.TOL],colors='white',linewidths=1.4)
        for i,j in itertools.product(range(len(ys)),range(len(xs))):
            if np.isnan(errors[i,j]):ax.text(j,i,'-',ha='center',va='center',color='gray',fontsize=8)
            elif errors[i,j]>X.TOL and ax is axes[1]:ax.plot(j,i,'x',c='white',ms=4)
    axes[0].set_title('Accuracy; white contour = 1e-4')
    axes[1].set_title('Runtime; crosses exceed tolerance')
    fig.suptitle(title)
    fig.savefig(X.FIGURES/filename,dpi=190);plt.close(fig)

def plots(out,free,equal,old):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    plane=X.read(X.RESULTS/'cutoff_heatmap.csv');mesh=X.read(X.HERE/'mesh_tradeoff.csv')
    plan=json.loads((X.RESULTS/'study_plan.json').read_text());fp=plan['free_incumbent'];bp=plan['baseline']
    heatmap(plane,'sF_req','sR','fig1_cutoff_heatmaps.png',
            f"Independent cutoffs: xi={fp['xi']:g}, M={fp['M']}, p={fp['p']}")
    heatmap(mesh,'p','M','fig2_mesh_order_heatmaps.png',
            f"Fixed cutoffs: xi={bp['xi']:g}, sR={bp['sR']:g}, sF={bp['sF_req']:g}")
    oat=X.read(X.HERE/'parameter_sensitivity.csv');reopt=X.read(X.RESULTS/'reoptimized_sensitivity.csv')
    fig,axes=plt.subplots(2,5,figsize=(17,7),layout='constrained')
    for i,field in enumerate(('xi','sR','sF','M','p')):
        rr=sorted([r for r in oat if r['parameter']==field and r.get('t_complete_s') is not None],key=lambda r:r['value'])
        xx=[r['value'] for r in rr]
        for key,label,color in (('t_complete_s','Complete','black'),('t_real_s','Real','C1'),
                                ('t_spread_gather_s','Spread/gather','C0'),('t_fft_s','FFT + influence','C2')):
            axes[0,i].plot(xx,[r[key]*1e3 for r in rr],'o-',ms=3,lw=1.2,label=label,color=color)
        rp=[r for r in reopt if r['parameter']==('sF_req' if field=='sF' else field)]
        axes[0,i].plot([r['value'] for r in rp],[r['t_complete_s']*1e3 for r in rp],':',color='C4',label='Reoptimized, tested finalists')
        for key,label,color in (('E_G_max','Full error','black'),('E_cutoff_max','Cutoff','C1'),('E_M_max','Mesh','C0')):
            axes[1,i].semilogy(xx,[r[key] for r in rr],'o-',ms=3,lw=1.2,label=label,color=color)
        axes[1,i].axhline(X.TOL,c='red',ls='--',lw=1)
        for ax in axes[:,i]:ax.grid(alpha=.15);ax.set_xlabel(field)
        axes[0,i].set_title(field)
        axes[0,i].set_yscale('log')
    axes[0,0].set_ylabel('Measured time (ms, log scale)');axes[1,0].set_ylabel('Maximum error over nine')
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    fig.suptitle(f"OAT about feasible baseline: xi={bp['xi']:g}, sR=sF={bp['sR']:g}, M={bp['M']}, p={bp['p']}")
    fig.savefig(X.FIGURES/'fig3_parameter_sensitivity.png',dpi=190);plt.close(fig)
    controls=json.loads((X.RESULTS/'strategy_controls.json').read_text())
    byid={r['candidate_id']:r for r in out}
    selected=[('Old parameters',old),('Equal cutoffs',equal),('Free optimum',free)]
    selected += [(f'Strategy {label}',byid[cid]) for label,cid in controls.items()]
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');xx=np.arange(len(selected));bottom=np.zeros(len(selected))
    for key,label,color in (('t_real_s','Real-space action','C1'),('t_spread_gather_s','Spread + gather','C0'),('t_fft_s','FFT + influence','C2')):
        yy=np.array([r[key]*1e3 for _,r in selected]);ax.bar(xx,yy,bottom=bottom,label=label,color=color);bottom+=yy
    actual=[r['t_complete_s']*1e3 for _,r in selected]
    ax.scatter(xx,actual,c='black',marker='D',s=35,label='Actual complete matvec',zorder=4)
    for x,y in zip(xx,actual):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    ax.set(xticks=xx,xticklabels=[f"{name}\nM={r['M']}, p={r['p']}" for name,r in selected],ylabel='Time (ms)',
           title='Final common timing: component medians and complete action')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15);ax.margins(y=.22)
    fig.savefig(X.FIGURES/'fig4_cost_breakdown.png',dpi=190);plt.close(fig)

def audit(search):
    manifest=json.loads((X.HERE/'historical_artifact_hashes.json').read_text())
    assert all(hashlib.sha256((C.REPO/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    refs=X.read(X.RESULTS/'references.csv');assert len(refs)==9
    executed=json.loads((X.RESULTS/'analysis_provenance.json').read_text())['source_sha256']
    for name in ('experiment.py','run_optimization.py','analyze_optimization.py'):
        assert executed[name]==hashlib.sha256((X.HERE/name).read_bytes()).hexdigest()
    for r in refs:
        assert r['kappa']==.2 and r['c']==1. and r['cert_bound_rel']<=1e-13
        path=E.REF_DIR/r['cache_file'];assert 'kap0.2_' in path.name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['cache_sha256']
    raw=X.read(X.RESULTS/'final_timing_raw.csv');timing=X.read(X.RESULTS/'final_timing.csv')
    for t in timing:
        rs=[r for r in raw if r['candidate_id']==t['candidate_id'] and r['config']==t['config']]
        for phase in ('complete','real','mesh','profile'):
            ts=[r for r in rs if r['phase']==phase]
            assert len(ts)==t['rounds']*t['reps']==t['n_samples']
            assert {(r['round'],r['rep']) for r in ts}==set(itertools.product(range(t['rounds']),range(t['reps'])))
            if phase!='profile': assert np.median([r['t_s'] for r in ts])==t[f't_{phase}_s']
    X.jsave(X.RESULTS/'audit_checks.json',dict(historical_tracked_files_unchanged=len(manifest),
             references_kappa=.2,references_c=1.,n_references=9,reference_hashes_verified=True,
             reference_certification_tolerance=1e-13,reference_tails_within_tolerance=True,
             maximum_reference_certified_tail=max(r['cert_bound_rel'] for r in refs),raw_timing_medians_verified=True,
             executed_numerical_sources_verified=True,
             n_candidates=len(search.rows),n_configurations=9,n_final_candidate_configuration_rows=len(timing),n_timing_samples=len(raw),
             production_implementation_unchanged=True,
             source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in X.HERE.glob('*.py')}))

def report(search,out,free,equal,old,speed,eqspeed,models):
    decomposition=X.read(X.HERE/'error_decomposition.csv')
    dec=[r for r in decomposition if r['candidate_id']==free['candidate_id']]
    assert len(dec)==9
    emax={key:max(r[key] for r in dec) for key in ('E_R','E_F','E_M','E_cutoff','E_G','far_split_residual')}
    mesh=X.read(X.HERE/'mesh_tradeoff.csv');oat=X.read(X.HERE/'parameter_sensitivity.csv')
    control_ids=json.loads((X.RESULTS/'strategy_controls.json').read_text());byid={r['candidate_id']:r for r in out}
    diagnostics={r['candidate_id']:r for r in X.read(X.RESULTS/'diagnostic_summary.csv')}
    Arow=byid[control_ids['A']];Brow=byid[control_ids['B']]
    def physical_diagnostic(row):
        return next(d for d in diagnostics.values() if all(d[k]==row[k] for k in ('xi','sR','kc_shell_n','M','p')))
    diagA=physical_diagnostic(Arow);diagB=physical_diagnostic(Brow)
    strategy_speed=Arow['t_complete_s']/Brow['t_complete_s']
    mesh_reduction=1-Brow['t_mesh_s']/Arow['t_mesh_s']
    ratios=[r['E_mesh_to_cutoff'] for r in dec]
    components={'real':free['t_real_s'],'spread/gather':free['t_spread_gather_s'],'FFT+influence':free['t_fft_s']}
    bottleneck=max(components,key=components.get)
    oldparts={'real':old['t_real_s'],'spread/gather':old['t_spread_gather_s'],'FFT+influence':old['t_fft_s']}
    oldbottleneck=max(oldparts,key=oldparts.get)
    balanced=min([r for r in out if r['feasible']],key=lambda r:abs(np.log(r['t_real_s']/r['t_mesh_s'])))
    ci=speed['paired_configuration_and_round_bootstrap_95pct']
    eci=eqspeed['paired_configuration_and_round_bootstrap_95pct']
    environment=json.loads((X.RESULTS/'analysis_provenance.json').read_text())
    protocol=json.loads((X.RESULTS/'timing_protocol.json').read_text())
    significant=ci[0]>1
    bestparams=lambda r:f"(ξ,sR,sF_req,M,p)=({r['xi']:g},{r['sR']:g},{r['sF_req']:g},{r['M']},{r['p']})"
    lines=['# κ=0.2：理论指导、精度约束的联合参数优化','',
           f"本次最快 tested-grid configuration 为 **{bestparams(free)}**；实际 sF_eff={free['sF_eff']:.8g}，最大 EΓ={free['E_G_max']:.9g}，cached complete matvec={free['t_complete_s']*1e3:.6f} ms。这是自适应离散搜索中的实测最优，未证明全局最优。",'',
           '固定 Candidate 1（θ=0）、N=256、ρ=0.75、κ=0.2、c=1；使用原九组粒子位置。所有比较使用新生成的 κ=0.2 独立 unsplit direct periodic Γ_D。κ=0.7 的旧矩阵和历史 1.730 ms/8.65e-5 未用于本次精度或加速比。','',
           '## 三个主要对照','',
           '| 对照 | ξ | sR | sF_req / effective | M | p | max EΓ | real ms | mesh ms | complete ms | setup ms |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for label,r in [('旧参数在κ=.2重测',old),('强制sR=sF',equal),('五参数独立',free)]:
        lines.append(f"| {label} | {r['xi']:g} | {r['sR']:g} | {r['sF_req']:g} / {r['sF_eff']:.6f} | {r['M']} | {r['p']} | {r['E_G_max']:.8g} | {r['t_real_s']*1e3:.6f} | {r['t_mesh_s']*1e3:.6f} | {r['t_complete_s']*1e3:.6f} | {r['t_setup_s']*1e3:.6f} |")
    lines += ['',f"旧参数在本次物理条件下{'满足' if old['feasible'] else '不满足'}精度。自由搜索相对旧参数加速 {speed['speedup']:.5f}×，耗时变化 {-speed['time_reduction_fraction']*100:+.3f}%；相对本次等 cutoff 最佳配置加速 {eqspeed['speedup']:.5f}×。",'',
              '## Reference、核和理论初始化','',
              '现有本地副本中未发现 κ=.2 缓存，因此只生成九份必要 reference。每份调用原 direct_periodic.annulus_tensors，以 g(r)=exp(-.2r)/r 直接累加到 R_D=190，并保存八个半径的收敛记录和认证尾界≤1e-13。三进程构建，各进程单线程；所有构建任务结束后才进行最终计时。新旧物理状态的 cache key 不同，加载时核对 κ、c、位置 hash 和来源。历史 κ=.2 的范数仅用于选取求和半径和事后交叉核对，没有替代矩阵数据。', '',
              '理论公式、条件及 Landau/Fast DPD 文稿适用范围见 stage1_theory.md。Landau 原文控制自由空间系数误差，其结论不能直接转化为周期谱算子证书。本次没有固定三项预算，也没有用保守理论界永久排除候选。', '',
              '生产 Fourier 系数与加密积分、短程闭式与积分的数值一致性见 coefficient_audit.json。最终候选的 BR/BF/BM、真实 Γ_D 归一化及对角块下界见 final_theory_bounds.csv；quadrature 项是数值估计。','',
              '## 实际误差分解','',
              '| 最佳候选，九组分别取最大值 | relative spectral norm |','|---|---|']
    for key,value in emax.items():lines.append(f'| {key} | {value:.9g} |')
    lines += ['', 'ER 使用 sR_far=8 的直接短程算子；EF 使用 sF_far≈7 的独立 mesh-free Fourier 算子作为无限 cutoff 的数值近似。无限极限的分割重构残差已记录并检查<1e-10。EM 严格比较同一 rc、kc 和同一 reciprocal shells 上的 PPPM 与 mesh-free 算子。Ecutoff 则直接比较该 mesh-free 总算子与独立 Γ_D。', '',
              'error_decomposition.csv 同时保存矩阵误差相加的 telescope residual、ER+EF+EM 与真实总误差的比值。不同配置的最大值可能出现在不同配置上；谱范数的和不等于总误差，不能通过相减标量误差来推断某个分量。','',
              '## 降低 M、p：两种策略','',
              '以下两点固定同一个 ξ、sR、sF，只改变 M、p。Strategy A 在实测可行点中要求 max EM≤0.1 max Ecutoff；Strategy B 采用该 M/p 平面在最终共同计时中最快的已测可行点。最终性能均来自共同交错计时。','',
              '| 策略 | M | p | max Ecutoff | max EM | max EΓ | spread/gather ms | FFT ms | mesh ms | complete ms |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for label,r,d in [('A',Arow,diagA),('B',Brow,diagB)]:
        lines.append(f"| {label} | {r['M']} | {r['p']} | {d['E_cutoff_max']:.8g} | {d['E_M_max']:.8g} | {r['E_G_max']:.8g} | {r['t_spread_gather_s']*1e3:.6f} | {r['t_fft_s']*1e3:.6f} | {r['t_mesh_s']*1e3:.6f} | {r['t_complete_s']*1e3:.6f} |")
    lines += ['',f"B 相对 A 的 mesh 耗时降低 {mesh_reduction*100:.3f}%，完整加速 {strategy_speed:.5f}×；总误差增加 {Brow['E_G_max']-Arow['E_G_max']:+.8g}。这些数字才是固定 cutoff 的 mesh 分辨率收益，不能把五参数全部调整后的速度差都归因于 M/p。",'',
              '各 p 的最小已测可行 M 与最快 M：','',
              '| p | 最小已测可行 M | 最快已测 M | max EΓ | complete ms（诊断批次） |', '|---|---|---|---|---|']
    for p in R.ORDERS:
        group=[r for r in mesh if r['p']==p and r.get('feasible')]
        if group:
            r=min(group,key=lambda r:r['t_complete_s'])
            lines.append(f"| {p} | {min(v['M'] for v in group)} | {r['M']} | {r['E_G_max']:.8g} | {r['t_complete_s']*1e3:.6f} |")
        else:lines.append(f'| {p} | 无 | 无 | - | - |')
    plateau=[]
    for p in R.ORDERS:
        group=sorted([r for r in mesh if r['p']==p and r.get('E_G_max') is not None],key=lambda r:r['M'])
        flat=[r['M'] for r in group if r['E_M_max']<=.1*r['E_cutoff_max'] and abs(r['E_G_max']/r['E_cutoff_max']-1)<=.1]
        important=[r['M'] for r in group if r['E_M_max']>=.2*r['E_cutoff_max']]
        failed=[r['M'] for r in group if r['E_G_max']>X.TOL]
        plateau.append(dict(p=p,plateau_M_values=';'.join(str(v) for v in flat),
                            mesh_significant_M_values=';'.join(str(v) for v in important),
                            accuracy_failed_M_values=';'.join(str(v) for v in failed)))
    X.write(X.RESULTS/'plateau_analysis.csv',plateau)
    lines += ['', 'Plateau 诊断采用显式阈值：EM≤0.1 Ecutoff 且 |EΓ/Ecutoff−1|≤10%；EM≥0.2 Ecutoff 标记为开始显著。这里使用各量九组最大值的比值作为平面摘要，逐配置数据另存。','',
              '| p | cutoff plateau 的 M | mesh 显著影响区的 M | 精度超标的 M |','|---|---|---|---|']
    for r in plateau:
        lines.append(f"| {r['p']} | {r['plateau_M_values'] or '无'} | {r['mesh_significant_M_values'] or '无'} | {r['accuracy_failed_M_values'] or '无'} |")
    lines += ['', '允许 ξ、sR、sF 重新调整后的 order 对照如下。“最小 M”来自全部九组验证通过的搜索点；“最快”来自最终共同计时，二者不是同一目标。','',
              '| p | 全搜索最小已测可行 M | 共同计时最快 (ξ,sR,sF_req,M) | max EΓ | complete ms |',
              '|---|---|---|---|---|']
    order_summary=[]
    for p in R.ORDERS:
        feasible=[r for r in search.rows if r.get('feasible') and r['p']==p]
        measured=[r for r in out if r['feasible'] and r['p']==p]
        if feasible and measured:
            r=min(measured,key=lambda r:r['t_complete_s']);minimum=min(v['M'] for v in feasible)
            order_summary.append(dict(r,smallest_observed_feasible_M=minimum))
            lines.append(f"| {p} | {minimum} | ({r['xi']:g},{r['sR']:g},{r['sF_req']:g},{r['M']}) | {r['E_G_max']:.8g} | {r['t_complete_s']*1e3:.6f} |")
    X.write(X.RESULTS/'reoptimized_order_tradeoff.csv',order_summary)
    lines += ['', '上表限定固定 baseline 的 cutoff 平面，最小 M 是已测试离散值，不是所有整数或所有 cutoff 的理论最小值。M/p heatmap 与数据包含不合法和误差超标的点，未将它们选择性删除。高分辨率 EΓ 是否趋近 Ecutoff、低分辨率 EM 是否开始占据误差余量，可直接从图 2/3 及 mesh_tradeoff.csv 检查。','',
              '## OAT 与重新优化的敏感性','',
              'OAT 保持 baseline 的其余四个参数固定；每个合法点完整检查九组误差，计量 real、spread/gather、FFT+influence、mesh total 和 complete。parameter_sensitivity.csv 保存参数相对变化、节省毫秒和相对耗时变化。Reoptimized sensitivity 使用最终共同计时的全部可行候选，固定一个参数值后取其余参数已测组合中的最低成本；它不是每个连续参数值的全局重优化。两类曲线在图 3 中分别显示。','',
              '| 参数 | OAT可行点的最佳相对耗时改善 | 对应绝对节省 ms | 测试范围 |', '|---|---|---|---|']
    sensitivity=[]
    for field in ('xi','sR','sF','M','p'):
        group=[r for r in oat if r['parameter']==field and r.get('feasible') and r.get('complete_time_relative_change') is not None]
        if group:
            r=min(group,key=lambda r:r['complete_time_relative_change'])
            improvement=-r['complete_time_relative_change'];saving=-r['complete_time_change_s']
            times=[v['t_complete_s'] for v in group]
            baseline_time=r['t_complete_s']-r['complete_time_change_s']
            sensitivity.append(dict(parameter=field,improvement=improvement,saving_s=saving,
                                    feasible_cost_span_s=max(times)-min(times),
                                    feasible_cost_span_relative=(max(times)-min(times))/baseline_time))
            vals=[r['value'] for r in oat if r['parameter']==field]
            lines.append(f"| {field} | {improvement*100:.3f}% | {saving*1e3:.6f} | {min(vals):g}–{max(vals):g} |")
    most=max(sensitivity,key=lambda r:r['improvement'])
    widest=max(sensitivity,key=lambda r:r['feasible_cost_span_relative'])
    X.write(X.RESULTS/'relative_cost_sensitivity.csv',sensitivity)
    lines += ['',f"在本次指定 OAT 范围内，最大可行降时来自 {most['parameter']}；这不是对不同单位的数值导数排名，采样跨度和离散合法性会影响结论。此处不足1%的名义改善小于本轮计时波动，不能据此认定 ξ 微调产生确定加速。",'',
              '各 OAT 曲线合法端点的工作量与耗时变化如下。端点可能精度超标，误差列明确保留；上面的“最佳改善”表只使用满足精度的点。','',
              '| 参数 | 合法端点 | 近场 images | retained modes | real ms | spread/gather ms | FFT ms | max EΓ |',
              '|---|---|---|---|---|---|---|---|']
    effects=[]
    for field in ('xi','sR','sF','M','p'):
        group=sorted([r for r in oat if r['parameter']==field and r.get('t_complete_s') is not None],key=lambda r:r['value'])
        for tag,r in [('low',group[0]),('high',group[-1])]:
            effects.append(dict(r,endpoint=tag))
            lines.append(f"| {field} | {tag}={r['value']:g} | {r['n_images']:.0f} | {r['n_modes']} | {r['t_real_s']*1e3:.5f} | {r['t_spread_gather_s']*1e3:.5f} | {r['t_fft_s']*1e3:.5f} | {r['E_G_max']:.7g} |")
    X.write(X.RESULTS/'parameter_effect_endpoints.csv',effects)
    lines += ['', 'ξ 的 OAT 保持 M、p 固定：rc=sR/ξ 缩短、retained modes 增加，但 FFT 仍在同一完整 M³ 网格上执行，不能据此声称其 FFT 工作量按 ξ³ 增加。重新优化的 ξ 曲线才包含为满足精度/Nyquist 而提高 M/p 的阶梯成本。sR 直接改变 images 和 real 时间；sF 改变误差、modes 及最小合法 M，在固定 M/p 的完整网格 FFT 中不产生对应的稀疏 FFT 加速。M 改变网格工作量而固定 p³ stencil 大小；p 改变 spread/gather 工作量及 transfer error，而固定 M 下 FFT 尺寸不变。实际变化量以上表和完整 OAT 数据为准。','',
              '## 成本模型与瓶颈','',
              '以非负最小二乘拟合 TR≈CR Npairs+C0R、Tspread/gather≈CP Np³+C0P、TFFT≈CF M³log(M³)+C0F。FFT 计时包含 influence 乘法和 inverse FFT。模型只解释本实现的实测成本，不能代替目标函数。','',
              '| phase | 秒/工作量系数 | 固定秒数 | R² | 相对残差中位数 |','|---|---|---|---|---|']
    for m in models:lines.append(f"| {m['phase']} | {m['coefficient_s_per_work']:.7g} | {m['constant_s']:.7g} | {m['R_squared']:.4f} | {m['median_absolute_relative_residual']:.4f} |")
    lines += ['',f"最快候选：real={free['t_real_s']*1e3:.6f} ms，spread/gather={free['t_spread_gather_s']*1e3:.6f} ms，FFT+influence={free['t_fft_s']*1e3:.6f} ms，mesh total={free['t_mesh_s']*1e3:.6f} ms。real/mesh={free['t_real_s']/free['t_mesh_s']:.4f}，主要被测阶段为 {bottleneck}。",'',
              f"旧参数的最大被测阶段为 {oldbottleneck}，优化后为 {bottleneck}。最终计时中 real/mesh 最接近1的可行点是 {bestparams(balanced)}，比值 {balanced['t_real_s']/balanced['t_mesh_s']:.4f}，完整 {balanced['t_complete_s']*1e3:.6f} ms，为最快时间的 {balanced['t_complete_s']/free['t_complete_s']:.4f} 倍。",'',
              '组件的独立中位数不要求精确相加。real 与 mesh total 各含对应的缓存 degree 乘法；spread/gather 和 FFT 不含 degree。用于组件计时的表达式逐项复用原 Mesh.apply，并与原 mesh.apply 校验；优化指标始终是未经改写的 op.apply cached action。', '',
              '重新优化后的 ξ 切片（每个 ξ 只取最终计时中最快的已测可行组合）：','',
              '| ξ | sR | sF_req | M | p | real ms | mesh ms | complete ms |',
              '|---|---|---|---|---|---|---|---|']
    xi_summary=[]
    for xi in sorted(set(r['xi'] for r in out if r['feasible'])):
        r=min([r for r in out if r['feasible'] and r['xi']==xi],key=lambda r:r['t_complete_s'])
        xi_summary.append(r)
        lines.append(f"| {xi:g} | {r['sR']:g} | {r['sF_req']:g} | {r['M']} | {r['p']} | {r['t_real_s']*1e3:.6f} | {r['t_mesh_s']*1e3:.6f} | {r['t_complete_s']*1e3:.6f} |")
    X.write(X.RESULTS/'best_runtime_by_xi.csv',xi_summary)
    xi_limits=(min(r['xi'] for r in search.rows),max(r['xi'] for r in search.rows))
    lines += ['',f"实际搜索 ξ 范围 {xi_limits[0]:g}–{xi_limits[1]:g}。最佳 ξ={free['xi']:g} {'处于边界，需要继续外扩' if free['xi'] in xi_limits else '位于已搜索范围内部'}；这些非穷举切片可显示右侧成本回升或阶梯变化，但不证明连续问题的全局最小值。",'',
              '## 搜索、验证与计时方法','',
              f"搜索共记录 {len(search.rows)} 个不同请求，其中 {sum(bool(r.get('feasible')) for r in search.rows)} 个通过全部九组精度；最终 {len(out)} 个物理候选进入共同长计时。报告的最快点取自最终共同计时中的已验证可行候选。顺序为 equal-cutoff coarse slices、独立 sR/sF 平面、局部五参数与 mesh/order 耦合细化、边界/ξ 复查、九组验证和 sensitivity 平面。所有实际参数、shell、合法性及排除原因在 joint_candidates.csv。理论界从未作为永久排除条件。单配置精度超标可直接排除全九组目标；通过单配置者必须另行检查九组，不能将初筛通过误称可行。",'',
              '单配置初筛可与 reference 生成重叠，初筛时间仅用于临时引导。九份 reference 完成后，按随机顺序重测全部初筛通过者再排名。敏感性分析后又进行了五参数耦合细化，随后连续重测 diagnostics 和最终候选。首次批次有长时间间隔，其原始数据和首版报告保留在 results/ 的 timing_pass、diagnostic_pass 和 first_summary 目录；结论使用最后一批共同交错计时。最终连续批次用 caffeinate -i 避免 idle sleep，不排除系统其他负载。setup 单独独立构建三次，不含 reference、稠密验证或参数核首次初始化。', '',
              'OAT、M/p 和 cutoff 平面的诊断计时采用3轮×5次，候选每次重复随机排序，并保存逐样本数据。最终候选采用下述更长的计时；两批用于不同分析，不能直接把两个批次的绝对时间相减当作收益。diagnostic_protocol.json 和 timing_protocol.json 保存开始时间、总时长和每配置检查点。', '',
              f"最终每候选每配置 {free['n_samples_per_configuration_phase']} 个样本（9轮×11次），候选每次重复随机排序，phase每轮随机排序；每块先warmup3次、每个样本前再warmup1次。单线程环境变量和 threadpool_limits(1) 一致。目标统计量是九个配置各自中位数的中位数。最大相对 IQR={free['max_complete_iqr_rel']*100:.2f}%。",'',
              f"本轮环境：{platform.platform()}，CPU {environment['cpu']}、逻辑 CPU {environment['n_cpu']}；Python {environment['python']}、NumPy {environment['numpy']}、SciPy {environment['scipy']}。最终验证、setup 和共同计时总计 {protocol.get('elapsed_s',float('nan'))/60:.2f} 分钟；每配置检查点另存 protocol。OMP、OPENBLAS、MKL、VECLIB、NUMEXPR、BLIS 均限制为1线程。",'',
              f"自由候选相对旧参数的配对配置/轮次 block-bootstrap 95% 加速区间为 [{ci[0]:.5f},{ci[1]:.5f}]；{'区间完全大于1，支持本次环境中的加速。' if significant else '区间包含1，尚不足以将提升与本次重复计时波动明确区分。'} 该区间只描述本机本轮，是候选选择后的条件估计，未作多重比较校正，不保证其他机器或系统负载。",'',
              f"最佳候选九组均通过 mesh validity、PSD（平移零空间正交补上的最小λ={free['lam_min_h']:.9g}）、symmetry≤{free['symmetry_max']:.3g}、translation nullspace≤{free['translation_null_max']:.3g}、matrix-free/dense≤{free['matrixfree_dense_max']:.3g}。",'',
              '## 图与复现','',
              '![sR-sF](figures/fig1_cutoff_heatmaps.png)','',
              '![M-p](figures/fig2_mesh_order_heatmaps.png)','',
              '![Sensitivity](figures/fig3_parameter_sensitivity.png)','',
              '![Cost breakdown](figures/fig4_cost_breakdown.png)','',
              '```sh',
              'cd benchmarks/method_A_clustered/validation_exp1/theory_guided_joint_optimization',
              'python3 build_references.py --workers 3',
              'python3 theory.py',
              'python3 run_optimization.py all',
              'python3 analyze_optimization.py all',
              'python3 summarize_results.py',
              'python3 post_sensitivity_refinement.py',
              'caffeinate -i python3 analyze_optimization.py diagnostics timing --refresh-diagnostics',
              'python3 summarize_results.py',
              '```','',
              '已有 reference 和候选自动恢复；从头运行时应先等 reference 构建命令完成。caffeinate 是 macOS 的防 idle sleep 前缀，其他平台可省略。重复最终 timing 会将上次本实验的计时归档，不覆盖历史 Experiment 1/2、joint optimization、xi extension。环境、git版本、Python/NumPy/SciPy、BLAS线程和源文件hash见 results/ 中 provenance 与 audit_checks；命令及进度日志一并保存。','',
              '## Discussion','',
              f"**Q1. 谁主导误差？** 最快点九组 max Ecutoff={emax['E_cutoff']:.8g}、max EM={emax['E_M']:.8g}，逐配置 EM/Ecutoff 范围 [{min(ratios):.4g},{max(ratios):.4g}]。"
              +('cutoff 分量明显较大。' if max(ratios)<.2 else ('mesh 分量明显较大。' if min(ratios)>5 else 'mesh 不能忽略，误差分量处于可比较的尺度。'))
              +' ER/EF/EM 的矩阵方向及抵消决定真实 EΓ，不能把范数直接相加当作等式。','',
              f"**Q2. 之前 M、p 是否过大？** 旧参数 M=20,p=5 在 κ=.2 重测为 {old['t_complete_s']*1e3:.6f} ms；本次最优 M={free['M']},p={free['p']}，完整 {free['t_complete_s']*1e3:.6f} ms。网格点数为旧参数的 {(free['M']/20)**3:.4f} 倍，assignment 工作量 Np³ 为旧参数的 {(free['p']/5)**3:.4f} 倍；它们是工作量比，不是直接的时间比。固定 cutoff 的 A→B mesh 降时为 {mesh_reduction*100:.3f}%，完整加速 {strategy_speed:.5f}×。是否过度分辨率应以这些本次数据判断，不能把 κ=.7 的选参认作错误。较小 M/p 的精度失败、合法性限制和每个 p 的最小已测 M 均保留在图2和CSV。",'',
              f"**Q3. 哪个参数最敏感？** 在本次合法且满足精度的 OAT 采样范围内，成本跨度最大的是 {widest['parameter']}（跨度 {widest['feasible_cost_span_s']*1e3:.6f} ms，相对 baseline {widest['feasible_cost_span_relative']*100:.3f}%）；从 baseline 出发，{most['parameter']} 可带来最大实测相对降时（{most['improvement']*100:.3f}%，{most['saving_s']*1e3:.6f} ms）。两种排名回答不同问题，且依赖所测范围。重新优化曲线还允许其余参数变化；两者及成本模型需同时看，不能比较原始数值导数。",'',
              f"**Q4. 最优是否允许 mesh error 接近 cutoff error？** 最优逐配置 EM/Ecutoff 为 {min(ratios):.4g}–{max(ratios):.4g}；A/B 对照将误差变化与纯 mesh 速度收益分开。允许 mesh 消耗余量是否有益由上表实测决定，没有人为要求三项误差相等。",'',
              f"**Q5. 放开 sR=sF 更快吗？** 自由/等cutoff 的完整时间分别为 {free['t_complete_s']*1e3:.6f}/{equal['t_complete_s']*1e3:.6f} ms，加速比 {eqspeed['speedup']:.5f}×，配对区间 [{eci[0]:.5f},{eci[1]:.5f}]；参数分别为 {bestparams(free)} 与 {bestparams(equal)}。"
              +('自由搜索的第一名恰好也是等 cutoff 配置，因此两列是同一次测量；[1,1] 并非两个独立算法等速的统计证据。' if free['candidate_id']==equal['candidate_id'] else ('本轮可分辨自由搜索的收益。' if eci[0]>1 else '本轮没有分辨出超过波动的自由搜索收益。'))+' 这是已测候选中的对比。','',
              f"**Q6. 主要瓶颈和下一步？** 最快点主要被测阶段是 {bottleneck}，应优先优化该阶段的实现/数据移动，再以实际 complete action 复核收益。real/mesh={free['t_real_s']/free['t_mesh_s']:.4f}，两部分相等不是最优必要条件。",'',
              f"**Q7. 理论尺度覆盖实际区域吗？** sqrt(log(1/ε))≈3.035，而最优 sR={free['sR']:g}、sF_eff={free['sF_eff']:.6f}。该尺度指导了包含更小 cutoff 的宽初始区域；严格保守界与实际谱误差并不等价。κ=.2 的大零模提高了全算子归一化范数，离散shell、配置几何、aliasing与矩阵抵消还会改变实际最优位置。最终 BR/BF/BM 及 quadrature 估计另表给出，不混作实测误差。",'',
              f"**Q8. 改善超过计时波动吗？** 相对旧参数加速 {speed['speedup']:.5f}×，配对配置/轮次 bootstrap 95% 区间 [{ci[0]:.5f},{ci[1]:.5f}]。"
              +('本轮区间完全大于1，支持实测加速。' if significant else '本轮未能明确分辨加速与波动。')
              +' 单个最小时间不是结论依据；全部重复样本、IQR和环境均已保存。']
    (X.HERE/'report.md').write_text('\n'.join(lines)+'\n')
    X.jsave(X.RESULTS/'best_configuration.json',dict(best_free=free,best_equal=equal,old_parameters_remeasured=old,
             speedup_over_old=speed,speedup_over_equal=eqspeed,error_decomposition_maxima=emax,
             strategy_A=Arow,strategy_B=Brow,main_component=bottleneck,
             scope='fastest tested adaptive-grid configuration; no global optimality proof'))

def main():
    search=R.Search()
    with threadpool_limits(limits=1):
        out,free,equal,old=summary(search)
        selected=list({r['candidate_id']:r for r in (free,equal,old)}.values())
        bounds(selected)
        check_far_limits(selected)
        speed=speed_bootstrap(free,old);eqspeed=speed_bootstrap(free,equal)
        models=cost_model(search);plots(out,free,equal,old)
        report(search,out,free,equal,old,speed,eqspeed,models);audit(search)
        X.log(f"COMPLETE {free['candidate_id']} {free['t_complete_s']*1e3:.6f} ms E={free['E_G_max']:.9g}")

if __name__=='__main__': main()
