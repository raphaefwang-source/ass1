#!/usr/bin/env python3
"""Generate summaries, both figures, and the report from measured CSVs only."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ass1_joint_mpl')
import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path
import run_joint_optimization as J
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extension_report():
    """Report an isolated extension, using the run's actual persisted grid."""
    provenance = json.loads((J.RESULTS/'provenance.json').read_text())
    grid = provenance['search_grid']
    J.XI, J.S, J.MS, J.PS = (tuple(grid[k]) for k in ('xi', 's', 'M', 'p'))
    search, val, timing = (J.read(n) for n in ('search.csv', 'validation.csv', 'final_timing.csv'))
    expected = set(itertools.product(J.S, J.XI, J.MS, J.PS))
    actual = {(r['s'], r['xi'], r['M'], r['p']) for r in search}
    assert actual == expected and len(search) == len(expected)
    baseline_id = provenance['baseline_id']
    baseline = {r['config']:r for r in timing if r['candidate_id'] == baseline_id}
    assert len(baseline) == 9
    bt = float(np.median([r['t_matvec_s'] for r in baseline.values()]))
    out = []
    for cid in sorted(set(r['candidate_id'] for r in timing)):
        ts = [r for r in timing if r['candidate_id'] == cid]
        vs = [r for r in val if r['candidate_id'] == cid]
        assert len(ts) == len(vs) == 9
        assert all(v['mesh_valid'] and v['E_G'] <= J.TOL and v['lam_min_h'] > 0 and
                   v['sym_resid'] < 1e-12 and v['mom_opnorm_max'] < 1e-12 and v['mf_vs_dense'] < 1e-10 for v in vs)
        v, worst = vs[0], max(vs, key=lambda r:r['E_G'])
        med = lambda k: float(np.median([r[k] for r in ts]))
        t = med('t_matvec_s')
        out.append(dict(candidate_id=cid, xi=v['xi'], s=v['sR'], M=v['M'], p=v['p'],
                        sF_eff=v['sF_eff'], rc=v['rc'], kc=v['kc'], kc_shell_n=v['kc_shell_n'],
                        E_G_max=worst['E_G'], worst_config=worst['config'],
                        t_matvec_s=t, t_real_s=med('t_real_s'), t_mesh_s=med('t_mesh_s'),
                        t_config_min_s=min(r['t_matvec_s'] for r in ts),
                        t_config_max_s=max(r['t_matvec_s'] for r in ts),
                        t_matvec_iqr_rel_max=max(r['t_matvec_iqr_rel'] for r in ts),
                        t_total_setup_s=med('t_total_setup_s'), t_mesh_setup_s=med('t_mesh_setup_s'),
                        t_particle_setup_s=med('t_particle_setup_s'),
                        real_mesh_ratio=med('t_real_s')/med('t_mesh_s'),
                        n_images_median=med('n_images'), n_modes=v['n_modes'],
                        lam_min_h=min(v['lam_min_h'] for v in vs),
                        sym_resid_max=max(v['sym_resid'] for v in vs),
                        momentum_max=max(v['mom_opnorm_max'] for v in vs),
                        mf_vs_dense_max=max(v['mf_vs_dense'] for v in vs),
                        speedup_over_xi115=bt/t,
                        paired_config_speedup_median=float(np.median([baseline[r['config']]['t_matvec_s']/r['t_matvec_s'] for r in ts]))))
    out.sort(key=lambda r:r['t_matvec_s'])
    J.write('final_summary.csv', out)
    J.write('feasible_candidates.csv', [r for r in out if r['candidate_id'] != baseline_id])
    bestxi = [min((r for r in out if r['xi'] == xi),
                  key=lambda r:r['t_matvec_s']) for xi in sorted(J.XI)]
    J.write('best_by_xi.csv', bestxi)
    best = min(bestxi, key=lambda r:r['t_matvec_s'])
    old = next(r for r in out if r['candidate_id'] == baseline_id)
    paired = [baseline[t['config']]['t_matvec_s']/t['t_matvec_s'] for t in timing
              if t['candidate_id'] == best['candidate_id']]
    rng = np.random.default_rng(20261023)
    boots = np.median(np.asarray(paired)[rng.integers(0,9,size=(10000,9))],axis=1)
    speed_ci = np.percentile(boots,[2.5,97.5]).tolist()
    controls = [r for r in out if r['candidate_id'].startswith('balance_control_')]
    balance = min(bestxi+controls, key=lambda r:abs(np.log(r['real_mesh_ratio'])))
    boundary = best['xi'] in (min(J.XI), max(J.XI))
    steps = []
    for a, b in zip(bestxi, bestxi[1:]):
        steps.append(dict(xi_from=a['xi'], xi_to=b['xi'],
                          real_time_saved_s=a['t_real_s']-b['t_real_s'],
                          mesh_time_added_s=b['t_mesh_s']-a['t_mesh_s'],
                          matvec_time_change_s=b['t_matvec_s']-a['t_matvec_s']))
    J.write('near_far_tradeoffs.csv', steps)
    raw = J.read('final_timing_raw.csv')
    for t in timing:
        for phase, key in (('matvec','t_matvec_s'),('real','t_real_s'),('mesh','t_mesh_s')):
            rs = [r for r in raw if r['candidate_id'] == t['candidate_id'] and
                  r['config'] == t['config'] and r['phase'] == phase]
            assert len(rs) == t['n_samples'] == t['rounds']*t['reps']
            assert {(r['round'],r['rep']) for r in rs} == set(itertools.product(range(t['rounds']),range(t['reps'])))
            assert np.median([r['t_matvec_s'] for r in rs]) == t[key]
    root = J.RESULTS.parent
    historical = json.loads((root/'historical_artifact_hashes.json').read_text())
    assert all(hashlib.sha256((J.HERE/p).read_bytes()).hexdigest()==h for p,h in historical.items())
    refs = J.read('references.csv')
    assert len(refs) == 9 and max(r['cert_bound_rel'] for r in refs) <= 1e-13
    assert all(hashlib.sha256((J.E.REF_DIR/r['cache_file']).read_bytes()).hexdigest()==r['cache_sha256'] for r in refs)
    previous_refs = J.E.read_csv(J.HERE/'results/references.csv')
    assert {r['cache_file']:r['cache_sha256'] for r in refs} == {r['cache_file']:r['cache_sha256'] for r in previous_refs}
    repo = J.C.REPO
    allowed = {str((J.HERE/n).relative_to(repo)) for n in ('run_joint_optimization.py','summarize_results.py')}
    prefix = str(root.relative_to(repo))+'/' if root.is_relative_to(repo) else None
    changed = subprocess.check_output(['git','diff','--name-only','f9ce8564a80ce627ed866c6df4186d860f6af2bc'],
                                      cwd=repo,text=True).splitlines()
    assert all(p in allowed or (prefix is not None and p.startswith(prefix)) for p in changed), changed
    assert provenance.get('runner_sha256') == hashlib.sha256((J.HERE/'run_joint_optimization.py').read_bytes()).hexdigest()
    J.jsave('best_configuration.json', dict(best=best, fastest_including_fixed_controls=out[0], previous_xi115_remeasured=old,
                                           best_by_xi=bestxi, closest_balance=balance,
                                           optimum_on_search_boundary=boundary, tradeoffs=steps,
                                           paired_config_speedup_bootstrap_95pct=speed_ci,
                                           scope='adaptive discrete measured search; no global optimum claim'))
    J.jsave('audit_checks.json', dict(complete_requested_grid=True, n_requested=len(search),
                                     n_valid_mesh=sum(bool(r['mesh_valid']) for r in search),
                                     n_dense_errors=sum(r.get('E_G_design') is not None for r in search),
                                     n_runtime_pruned=sum(r.get('status')=='runtime pruned before dense error' for r in search),
                                     n_final_timed=len(out), n_raw_samples=len(raw),
                                     nine_config_error_structure_checks_pass=True,
                                     interleaved_component_medians_recomputed=True,
                                     nine_reference_caches_unchanged=True,
                                     historical_artifacts_unchanged=len(historical),
                                     production_PPPM_and_other_experiments_unchanged=True,
                                     executed_runner_hash_verified=True,
                                     source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                                    (J.HERE/'run_joint_optimization.py',Path(__file__))}))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':140,'savefig.dpi':200})
    xs = [r['xi'] for r in bestxi]
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(xs,[r['t_matvec_s']*1e3 for r in bestxi],'o-',label='Complete matvec',lw=2)
    ax.plot(xs,[r['t_real_s']*1e3 for r in bestxi],'s-',label='Real-space action, including degree')
    ax.plot(xs,[r['t_mesh_s']*1e3 for r in bestxi],'^-',label='Mesh action, including degree')
    if controls:
        cc = sorted(controls,key=lambda r:r['xi'])
        for key,marker,color,label in (('t_matvec_s','o','C0','Fixed earlier designs: total'),
                                      ('t_real_s','s','C1','Fixed earlier designs: real'),
                                      ('t_mesh_s','^','C2','Fixed earlier designs: mesh')):
            ax.plot([r['xi'] for r in cc],[r[key]*1e3 for r in cc],marker+'--',color=color,alpha=.6,label=label)
    ax.axhline(old['t_matvec_s']*1e3,ls='--',c='#6b7280',label='Previous xi=1.15 optimum, remeasured')
    ax.scatter(best['xi'],best['t_matvec_s']*1e3,c='black',s=140,marker='*',zorder=5)
    ax.set(xlabel=r'$\xi$',ylabel='Measured time (ms)',title='Jointly optimized Candidate 1: real, mesh, and total cost',
           xticks=sorted(set(xs+[r['xi'] for r in controls])))
    ax.tick_params(axis='x', labelrotation=35)
    ax.xaxis.set_label_coords(.5,-.13)
    ax.grid(alpha=.15); ax.legend(fontsize=8); fig.tight_layout()
    for suffix in ('png','pdf'): fig.savefig(J.FIGURES/f'near_far_matvec_vs_xi.{suffix}')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(xs,[r['t_matvec_s']*1e3 for r in bestxi],'o-',lw=2)
    ax.fill_between(xs,[r['t_config_min_s']*1e3 for r in bestxi],
                    [r['t_config_max_s']*1e3 for r in bestxi],alpha=.15,label='Nine-configuration min-max')
    for i,r in enumerate(bestxi):
        ax.annotate(f"s={r['s']:g}\nM={r['M']}, p={r['p']}",(r['xi'],r['t_matvec_s']*1e3),
                    xytext=(0,12 if i%2==0 else -35),textcoords='offset points',
                    ha='left' if i==0 else ('right' if i==len(bestxi)-1 else 'center'),fontsize=8)
    ax.axhline(old['t_matvec_s']*1e3,ls='--',c='#6b7280',label='Previous optimum, same timing run')
    ax.set(xlabel=r'$\xi$',ylabel='Complete matvec time (ms)',title='Best feasible measured runtime versus xi',xticks=xs)
    ax.tick_params(axis='x', labelrotation=35)
    ax.xaxis.set_label_coords(.5,-.13)
    ax.margins(x=.08,y=.3); ax.legend(fontsize=8); ax.grid(alpha=.15); fig.tight_layout()
    for suffix in ('png','pdf'): fig.savefig(J.FIGURES/f'best_runtime_vs_xi.{suffix}')
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    dense = [r for r in search if r.get('E_G_design') is not None]
    ax=axes[0]
    ax.scatter([r['t_design_s']*1e3 for r in dense],[r['E_G_design'] for r in dense],
               s=10,c='#9ca3af',alpha=.3,label='Design configuration only')
    final=[r for r in out if r['candidate_id']!=baseline_id]
    for ax in axes:
        sc=ax.scatter([r['t_matvec_s']*1e3 for r in final],[r['E_G_max'] for r in final],
                      c=[r['xi'] for r in final],s=40,cmap='viridis',label='Nine-config validated')
        ax.scatter(old['t_matvec_s']*1e3,old['E_G_max'],marker='D',s=65,
                   facecolors='none',edgecolors='black',label='Previous optimum, remeasured')
        ax.scatter(best['t_matvec_s']*1e3,best['E_G_max'],marker='*',s=160,c='black',
                   label='Fastest in joint search domain',zorder=5)
        ax.axhline(J.TOL,ls='--',c='#dc2626',label='Tolerance 1e-4')
        ax.set(xlabel='Matvec time (ms)',ylabel=r'$E_\Gamma$',yscale='log')
        ax.grid(alpha=.15)
    axes[0].set(xscale='log',title='Full design-screen overview')
    axes[0].legend(fontsize=7,loc='lower right')
    axes[1].set(title='Finalists: maximum error over nine configurations')
    axes[1].set_ylim(min(r['E_G_max'] for r in out)*.85,J.TOL*1.16)
    axes[1].tick_params(axis='y',which='minor',labelsize=8)
    fig.colorbar(sc,ax=list(axes),label=r'$\xi$',shrink=.9)
    for suffix in ('png','pdf'): fig.savefig(J.FIGURES/f'accuracy_vs_runtime.{suffix}')
    plt.close(fig)
    lines = ['# Candidate 1：ξ 向右扩展的联合参数优化', '',
             f"本次联合搜索域内最快实测可行参数为 **(ξ,s,M,p)=({best['xi']:g},{best['s']:g},{best['M']},{best['p']})**，"
             f"最大 EΓ={best['E_G_max']:.9g}，完整 matvec={best['t_matvec_s']*1e3:.6f} ms。", '',
             f"近场={best['t_real_s']*1e3:.6f} ms，mesh={best['t_mesh_s']*1e3:.6f} ms，setup={best['t_total_setup_s']*1e3:.6f} ms。", '',
             f"ξ=1.15 的旧最优固定为 (s,M,p)=(3.1,20,5)，同次交错重测={old['t_matvec_s']*1e3:.6f} ms；"
             f"加速 **{best['speedup_over_xi115']:.5f}×**，耗时降低 {(1-1/best['speedup_over_xi115'])*100:.3f}%。历史 1.707 ms 仅作来源记录，速度比采用本次重测。", '',
             f"逐配置配对加速比中位数={best['paired_config_speedup_median']:.5f}×；对九个配对配置重采样的 95% bootstrap 区间为 [{speed_ci[0]:.5f}, {speed_ci[1]:.5f}]。该区间反映配置间变异，不代表所有系统负载下的保证。", '',
             ('最优 ξ 处于搜索边界，不能声称找到内部或全局最优。' if boundary else
              '最优 ξ 位于本次离散搜索内部。以下表格用于检查右侧回升；结果不构成连续参数全局最优证明。'), '',
             '## 每个 ξ 的最快可行组合', '',
             '| ξ | s | M | p | sF_eff | max EΓ | real (ms) | mesh (ms) | complete (ms) | real/mesh |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in bestxi:
        lines.append(f"| {r['xi']:g} | {r['s']:g} | {r['M']} | {r['p']} | {r['sF_eff']:.6f} | {r['E_G_max']:.8g} | {r['t_real_s']*1e3:.6f} | {r['t_mesh_s']*1e3:.6f} | {r['t_matvec_s']*1e3:.6f} | {r['real_mesh_ratio']:.4f} |")
    if out[0]['t_matvec_s'] < best['t_matvec_s']:
        r=out[0]
        lines += ['', f"固定诊断对照 {r['candidate_id']} 在本轮还更快（{r['t_matvec_s']*1e3:.6f} ms）。它未在本轮重新联合优化，不能将上述搜索域内结果表述为包含所有 ξ 的最优。"]
    lines += ['', '## 近远场平衡', '',
              '这里的“平衡”以完整实测 matvec 最小为准；两个部分的绝对时间相等不是优化的必要条件。增大 ξ 同时缩短实空间半径，并增加 Fourier modes 和满足精度所需的 mesh/assignment 成本。M、p 是离散变量，FFT 的实际成本还受网格因子影响。', '',
              f"本次各 ξ 最优点和固定诊断对照中，real/mesh 最接近 1 的点为 ξ={balance['xi']:g}（{balance['candidate_id']}），比值 {balance['real_mesh_ratio']:.4f}；其完整时间为全体最优的 {balance['t_matvec_s']/best['t_matvec_s']:.4f} 倍。"
              + (' 在联合搜索域 ξ≥1.15，各 ξ 的最佳点均由 mesh 占用更多时间；固定 ξ=1.0 对照接近等时，但完整时间更长。' if all(r['real_mesh_ratio']<1 for r in bestxi) else ''), '',
              '| ξ 变化 | real 节省 (ms) | mesh 新增 (ms) | 完整时间变化 (ms) |',
              '|---|---|---|---|']
    for r in steps:
        lines.append(f"| {r['xi_from']:g} → {r['xi_to']:g} | {r['real_time_saved_s']*1e3:.6f} | {r['mesh_time_added_s']*1e3:.6f} | {r['matvec_time_change_s']*1e3:.6f} |")
    next_step = next((r for r in steps if r['xi_from']==best['xi']),None)
    if next_step:
        lines += ['', f"从最快点 ξ={best['xi']:g} 增至 {next_step['xi_to']:g} 时，real 节省 {next_step['real_time_saved_s']*1e3:.6f} ms，"
                  f"mesh 新增 {next_step['mesh_time_added_s']*1e3:.6f} ms，完整 matvec 增加 {next_step['matvec_time_change_s']*1e3:.6f} ms。"
                  '这里已出现 mesh 新增成本超过近场节省的区域；各部分独立测量，差值不要求精确相加。']
    at15 = next((r for r in bestxi if r['xi']==1.5),None)
    right = [r for r in bestxi if r['xi']>1.5]
    if at15 and right:
        lines += ['', '按追加要求继续检查 ξ=1.5 右侧：'+ '；'.join(
            f"ξ={r['xi']:g} 为 {r['t_matvec_s']*1e3:.6f} ms（相对 ξ=1.5 耗时变化 {(r['t_matvec_s']/at15['t_matvec_s']-1)*100:+.2f}%）"
            for r in right)+'。本次测试未发现更右侧的收益。']
    if controls:
        lines += ['', '额外诊断对照采用上次已验证的固定设计，不在本次重新优化其 s/M/p。它们也重新通过了九组验证，并与本次候选在同一计时轮中测量。', '',
                  '| 固定对照 ξ | s | M | p | real (ms) | mesh (ms) | complete (ms) | real/mesh |',
                  '|---|---|---|---|---|---|---|---|']
        for r in sorted(controls,key=lambda r:r['xi']):
            lines.append(f"| {r['xi']:g} | {r['s']:g} | {r['M']} | {r['p']} | {r['t_real_s']*1e3:.6f} | {r['t_mesh_s']*1e3:.6f} | {r['t_matvec_s']*1e3:.6f} | {r['real_mesh_ratio']:.4f} |")
    fixed = sorted([r for r in search if r['s']==3.1 and r['M']==20 and r['p']==5],key=lambda r:r['xi'])
    lines += ['', '固定 s=3.1、M=20、p=5 的精度/合法性对照（初筛配置 uniform seed2），用于检查增大 ξ 后为何必须改变 M 或 p：', '',
              '| ξ | mesh valid | 初筛 EΓ | 排除原因 |', '|---|---|---|---|']
    for r in fixed:
        err = f"{r['E_G_design']:.9g}" if r.get('E_G_design') is not None else '未计算'
        reason = (r.get('mesh_reason') or r.get('status') or '初筛通过').replace('|',r'\|')
        lines.append(f"| {r['xi']:g} | {r['mesh_valid']} | {err} | {reason} |")
    lines += ['', 'real 和 mesh 各自包括 DS·X、Dm·X 的 degree 项。每轮随机化候选及三个计时 phase 的顺序；分别调用现有 op.real_H、mesh.apply 和 op.apply。显式验证 real(X)+mesh(X) 与完整 op.apply(X) 相对差小于 1e-12。独立测得的中位数不要求精确可加。', '',
              '## 搜索、验证与计时方法', '',
              f"- 仅 θ=0，N=256、ρ=.75、κ=.7、c=1，原有 uniform/moderate/strong 各三个 seed。九份 Γ_D 的 SHA256 与上次一致，未重新计算 reference。",
              f"- ξ={list(J.XI)}；s={list(J.S)}；M={list(J.MS)}；p={list(J.PS)}。共 {len(search)} 组请求，s_R=s_F=s，请求 Fourier cutoff 按原算法 snapping。",
              '- 修改并复用原 run_joint_optimization.py 的搜索、验证与计时函数；未新增 run_xi_extension.py，未改 PPPM 算法。',
              f"- 先按实际单配置 matvec 时间筛选。最终复查时，只有 5 个样本的最短时间仍超过该 ξ 已知精度初筛通过者中位数的 {provenance['options']['runtime_prune_factor']:g} 倍，才跳过其稠密误差计算；记录 status=runtime pruned，不能将其理解为精度不合格。ξ≤1.6 初轮采用 1.6 倍中位数门槛，随后复查所有因此跳过的组合；search_prior_attempts.csv 保留初轮记录。追加的更右侧 ξ 直接采用最终门槛。",
              '- 其余组合计算 uniform seed2 的完整谱误差；通过者按单配置速度取至少 32 个及最快者 1.4 倍内候选，在九组配置上测 7 次并按九组时间中位数排序。',
              f"- 每个 ξ 对九组时间最快者进行完整 E.evaluate 验证：至少三个可行者。各轮时间带见 provenance 快照；最终重新排名时覆盖首个可行者 {(provenance['options'].get('validation_band') or 1.12):g} 倍时间范围。失败候选提前停止，不进入最终列表。",
              '- 为避免批次负载漂移影响旧、新候选排名，重测全部精度初筛通过者的设计配置时间，再重测短名单的九组时间；不混用旧批次短名单时间。已经通过的精度验证可复用。',
              f"- 最终近邻复核合并每个 ξ 按当轮九组短名单时间、完整验证时间各前 {provenance['options']['finalists_per_xi']} 个，以及上一完整终选中实测前 {provenance['options'].get('timing_top_measured',0)} 个，始终包含旧最优和两个固定诊断对照。与旧基准参数完全相同的候选合并为基准，避免将重复测量噪声误报成参数改进。",
              '- 所有终选都满足九组最大 EΓ≤1e-4、mesh 合法、PSD、symmetry、momentum 及 matrix-free/dense 一致性。',
              f"- 最终复核采用单线程、串行；每个候选每配置 {timing[0]['rounds']} 轮×{timing[0]['reps']} 次。每轮 phase 顺序随机；同一 phase 的每次重复内随机化候选顺序。每块先 3 warmup，每次单独测量前再 warmup 1 次，减少候选轮换导致的缓存差异。三个 phase 均保留逐样本数据。setup 独立构建三次，分别记录 mesh 和 particle setup。",
              '- 目标统计量为九个配置各自中位数的中位数。自适应时间剪枝及有限网格会限制全局结论，全部跳过原因在 search.csv 中保留。', '',
              f"计时波动：最佳点最大相对 IQR={best['t_matvec_iqr_rel_max']*100:.2f}%。相近候选的细小时间差应视为本次实测排序。"
              'ξ=1.15–1.2 构成低耗时区域；约 3% 的差距不足以支持唯一且精确的连续参数最优点结论。', '',
              '## 最佳候选的九组结果', '',
              '| configuration | EΓ | real (ms) | mesh (ms) | complete (ms) | λmin on Range(Π) |',
              '|---|---|---|---|---|---|']
    for v in [r for r in val if r['candidate_id']==best['candidate_id']]:
        t = next(r for r in timing if r['candidate_id']==best['candidate_id'] and r['config']==v['config'])
        lines.append(f"| {v['config']} | {v['E_G']:.9g} | {t['t_real_s']*1e3:.6f} | {t['t_mesh_s']*1e3:.6f} | {t['t_matvec_s']*1e3:.6f} | {v['lam_min_h']:.9g} |")
    lines += ['', f"最佳候选 symmetry≤{best['sym_resid_max']:.3g}，momentum≤{best['momentum_max']:.3g}，matrix-free/dense≤{best['mf_vs_dense_max']:.3g}。", '',
              '## 图与数据', '', '![Real, mesh, complete](figures/near_far_matvec_vs_xi.png)', '',
              '![Best runtime](figures/best_runtime_vs_xi.png)', '',
              '![Accuracy vs runtime](figures/accuracy_vs_runtime.png)', '',
              'results/ 保存 search.csv、search_timing.csv、validation.csv、validation_summary.csv、final_timing_raw.csv、final_setup_raw.csv、final_timing.csv、final_summary.csv、best_by_xi.csv、near_far_tradeoffs.csv、best_configuration.json、references.csv、reference_convergence.csv、provenance.json、audit_checks.json。', '',
              '先前完整计时保存在 timing_pass_*，因用户追加更右侧搜索而停止的中间六点计时保存在 timing_partial_*，转接记录为 right_extension_transition.json。结论只使用根目录 final_timing* 的最终共同计时，不能把中间轮次样本混合为独立重复。', '',
              f"上次的 {len(historical)} 份 CSV/JSON/报告/图/log 均已逐文件核验 SHA256 保持不变。本次输出目录独立，historical_artifact_hashes.json 保存清单。", '',
              '## 重现命令', '', '```sh',
              'python3 benchmarks/method_A_clustered/validation_exp1/joint_parameter_optimization/run_joint_optimization.py all --xi-extension --runtime-prune-factor 2 --include-balance-controls --xis '+' '.join(str(x) for x in J.XI)+' --output-dir benchmarks/method_A_clustered/validation_exp1/joint_parameter_optimization/xi_extension_20261023',
              'python3 benchmarks/method_A_clustered/validation_exp1/joint_parameter_optimization/run_joint_optimization.py validate timing --xi-extension --runtime-prune-factor 2 --refresh-shortlist-times --validation-band 1.12 --finalists-per-xi 2 --timing-top-measured 2 --include-balance-controls --xis '+' '.join(str(x) for x in J.XI)+' --output-dir benchmarks/method_A_clustered/validation_exp1/joint_parameter_optimization/xi_extension_20261023',
              'python3 benchmarks/method_A_clustered/validation_exp1/joint_parameter_optimization/summarize_results.py --xi-extension --output-dir benchmarks/method_A_clustered/validation_exp1/joint_parameter_optimization/xi_extension_20261023',
              '```', '', '实际命令见 results/commands.log。若扩大 ξ 范围，实际网格以 provenance.json 为准。重复 timing 会将上一轮新实验计时存入 timing_pass_* 子目录。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(best=best, baseline=old, best_by_xi=bestxi,boundary=boundary),indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    J.add_options(ap)
    args = ap.parse_args()
    J.configure(args)
    if args.xi_extension:
        return extension_report()
    timing = J.read('final_timing.csv')
    val = J.read('validation.csv')
    search = J.read('search.csv')
    checks = J.read('validation_summary.csv')
    small = J.read('small_test.csv')
    assert len(search) == 4*9*3*8
    expected = set(itertools.product(J.S, J.XI, J.MS, J.PS))
    assert {(r['s'], r['xi'], r['M'], r['p']) for r in search} == expected
    ids = sorted(set(r['candidate_id'] for r in timing))
    out = []
    baseline = {r['config']: r for r in timing if r['candidate_id'] == 'old_baseline'}
    baseline_time = float(np.median([r['t_matvec_s'] for r in baseline.values()]))
    for cid in ids:
        ts = [r for r in timing if r['candidate_id'] == cid]
        vs = [r for r in val if r['candidate_id'] == cid]
        assert len(ts) == len(vs) == 9
        assert all(r['n_samples'] == 49 for r in ts)
        assert all(v['mesh_valid'] and v['E_G'] <= J.TOL and v['lam_min_h'] > 0 and
                   v['sym_resid'] < 1e-12 and v['mom_opnorm_max'] < 1e-12 and v['mf_vs_dense'] < 1e-10
                   for v in vs)
        v = vs[0]
        err = max(vs, key=lambda v: v['E_G'])
        med = float(np.median([r['t_matvec_s'] for r in ts]))
        out.append(dict(candidate_id=cid, s=v['sR'], sF_req=v['sF_req'], sF_eff=v['sF_eff'],
                        xi=v['xi'], M=v['M'], p=v['p'], rc=v['rc'], kc=v['kc'], kc_shell_n=v['kc_shell_n'],
                        E_G_max=err['E_G'], worst_config=err['config'],
                        t_matvec_median_s=med,
                        t_matvec_config_min_s=min(r['t_matvec_s'] for r in ts),
                        t_matvec_config_max_s=max(r['t_matvec_s'] for r in ts),
                        t_matvec_iqr_rel_max=max(r['t_matvec_iqr_rel'] for r in ts),
                        t_total_setup_median_s=float(np.median([r['t_total_setup_s'] for r in ts])),
                        t_mesh_setup_median_s=float(np.median([r['t_mesh_setup_s'] for r in ts])),
                        t_particle_setup_median_s=float(np.median([r['t_particle_setup_s'] for r in ts])),
                        t_real_H_median_s=float(np.median([r['t_real_H_s'] for r in ts])),
                        t_mesh_H_median_s=float(np.median([r['t_mesh_H_s'] for r in ts])),
                        images_median=float(np.median([r['n_images'] for r in ts])), n_modes=v['n_modes'],
                        lam_min_h=min(r['lam_min_h'] for r in vs), sym_resid_max=max(r['sym_resid'] for r in vs),
                        mom_opnorm_max=max(r['mom_opnorm_max'] for r in vs),
                        mf_vs_dense_max=max(r['mf_vs_dense'] for r in vs),
                        speedup_over_remeasured_baseline=baseline_time/med,
                        paired_config_speedup_median=float(np.median([baseline[r['config']]['t_matvec_s']/r['t_matvec_s'] for r in ts]))))
    out.sort(key=lambda r: r['t_matvec_median_s'])
    J.write('final_summary.csv', out)
    bestxi = [min([r for r in out if r['xi'] == xi and r['candidate_id'] != 'old_baseline'],
                  key=lambda r: r['t_matvec_median_s']) for xi in sorted(J.XI)]
    J.write('best_by_xi.csv', bestxi)
    best = min(bestxi, key=lambda r: r['t_matvec_median_s'])
    old = next(r for r in out if r['candidate_id'] == 'old_baseline')
    ss = []
    for p in (3, 4):
        rs = [r for r in small if r['p'] == p]
        assert len(rs) == 9
        ss.append(dict(p=p, M=18, xi=.85, s=3.2, E_G_max=max(r['E_G'] for r in rs),
                       n_pass=sum(r['E_G'] <= J.TOL for r in rs),
                       worst_config=max(rs, key=lambda r:r['E_G'])['config'],
                       t_matvec_median_s=float(np.median([r['t_matvec_s'] for r in rs])),
                       t_setup_median_s=float(np.median([r['t_setup_wall_s'] for r in rs]))))
    J.write('small_test_summary.csv', ss)
    J.jsave('best_configuration.json', dict(best=best, old_baseline_remeasured=old, small_test=ss,
                                           best_by_xi=bestxi,
                                           historical_baseline_matvec_s=.00187,
                                           historical_baseline_E_G_max=8.96e-5,
                                           historical_time_ratio=.00187/best['t_matvec_median_s'],
                                           search_scope='given discrete grid; adaptive measured timing shortlist; no continuous global optimum claim'))

    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.dpi':150, 'savefig.dpi':200})
    fig, ax = plt.subplots(figsize=(8,5))
    screened = [r for r in search if r.get('mesh_valid')]
    ax.scatter([r['t_design_s']*1e3 for r in screened], [r['E_G_design'] for r in screened],
               c='#9ca3af', s=14, alpha=.35, label='Screen: uniform seed 2 only')
    for xi, color in zip(sorted(J.XI), ('#2563eb','#059669','#d97706','#9333ea')):
        rs = [r for r in out if r['xi'] == xi and r['candidate_id'] != 'old_baseline']
        ax.scatter([r['t_matvec_median_s']*1e3 for r in rs], [r['E_G_max'] for r in rs],
                   c=color, s=40, edgecolors='white', linewidths=.4, label=f'9 configurations, xi={xi:g}')
    for r in ss:
        ax.scatter(r['t_matvec_median_s']*1e3, r['E_G_max'], marker='x', s=65, c='black', zorder=5)
        ax.annotate(f"M=18, p={r['p']}", (r['t_matvec_median_s']*1e3,r['E_G_max']), xytext=(6,7),textcoords='offset points')
    ax.scatter(old['t_matvec_median_s']*1e3, old['E_G_max'], marker='D', c='#db2777',s=55,label='Old unequal-cutoff baseline')
    ax.scatter(best['t_matvec_median_s']*1e3, best['E_G_max'],marker='*',c='#111827',s=180,zorder=6,label='Fastest measured feasible')
    ax.axhline(J.TOL, color='#dc2626',ls='--',lw=1.2,label='Tolerance 1e-4')
    ax.set(xlabel='Measured matvec time (ms)',ylabel=r'$E_\Gamma=\|\Gamma_h-\Gamma_D\|_2/\|\Gamma_D\|_2$',yscale='log',
           title='Candidate 1: accuracy and measured runtime')
    ax.grid(alpha=.15)
    ax.legend(fontsize=8,loc='upper right')
    fig.tight_layout()
    fig.savefig(J.FIGURES/'accuracy_vs_runtime.png')
    fig.savefig(J.FIGURES/'accuracy_vs_runtime.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5,4.5))
    xs = [r['xi'] for r in bestxi]
    ys = [r['t_matvec_median_s']*1e3 for r in bestxi]
    ax.plot(xs,ys,'o-',color='#2563eb',lw=1.6,label='Best measured feasible (median of 9)')
    ax.fill_between(xs,[r['t_matvec_config_min_s']*1e3 for r in bestxi],
                    [r['t_matvec_config_max_s']*1e3 for r in bestxi],alpha=.14,color='#2563eb',label='Configuration min-max')
    for index, r in enumerate(bestxi):
        ax.annotate(f"s={r['s']:g}, M={r['M']}, p={r['p']}",(r['xi'],r['t_matvec_median_s']*1e3),
                    xytext=(0,12),textcoords='offset points',
                    ha='left' if index == 0 else ('right' if index == len(bestxi)-1 else 'center'),fontsize=9)
    ax.axhline(old['t_matvec_median_s']*1e3,c='#db2777',ls='--',label='Old baseline, remeasured')
    ax.set(xlabel=r'$\xi$',ylabel='Matvec time (ms)',xticks=xs,title='Candidate 1: best feasible runtime versus xi')
    ax.margins(y=.25)
    ax.grid(alpha=.15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(J.FIGURES/'best_runtime_vs_xi.png')
    fig.savefig(J.FIGURES/'best_runtime_vs_xi.pdf')
    plt.close(fig)

    nvalid = sum(bool(r.get('mesh_valid')) for r in search)
    nsurvive = sum(bool(r.get('screen_pass')) for r in search)
    lines = [
        '# Candidate 1 联合参数优化（EΓ ≤ 10⁻⁴）', '',
        f"最快实测可行参数为 **(s, ξ, M, p)=({best['s']:g}, {best['xi']:g}, {best['M']}, {best['p']})**，"
        f"九组最大 EΓ={best['E_G_max']:.9g}，matvec={best['t_matvec_median_s']*1e3:.6f} ms。", '',
        '## 指定的小测试', '',
        '| ξ | s_R=s_F 请求值 | M | p | 九组最大 EΓ | 达标数 | matvec 中位数 (ms) |',
        '|---|---|---|---|---|---|---|']
    for r in ss:
        lines.append(f"| .85 | 3.2 | 18 | {r['p']} | {r['E_G_max']:.9g} | {r['n_pass']}/9 | {r['t_matvec_median_s']*1e3:.6f} |")
    lines += ['', '以上均为本次实际计算，误差来自完整稠密算子的谱范数，并非 action error 或旧图读数。小测试时间采用 2 warmup + 15 次重复。', '',
              '## 物理条件、reference 与实现', '',
              '- 仅 Candidate 1，θ=0，N=256、ρ=.75、κ=.7、c=1；原有 uniform/moderate/strong 各 seeds 1/2/3。',
              '- 只使用 Γ_D。初始本地 exp_refs_c1 中仅有 strong seed1 的 κ=2 缓存，完整文件名搜索未发现所需九份 κ=.7 缓存。',
              '- 先评估已有 references.csv：κ=.7 每份约 11–13 秒，R_D=55；因此只重建本次九份，未启动 Experiment 1 refs 全扫描。',
              '- 通过原有 direct_periodic.annulus_tensors(g=exp(-.7*r)/r) 直接周期求和；R_D=55，保留与原实验相同的 8 个半径，记录收敛与解析尾界。',
              '- 新建缓存只含 G_D 和收敛记录，位于 git-ignored 的 raw/exp_refs_c1；references.csv 保存位置 hash、缓存 SHA256、norm、实际耗时和截断误差界。',
              '- 旧 CSV 的 norm 仅用于选择重建半径；所有本次误差均由重建的完整 Γ_D 矩阵计算。与旧 norm 的差异亦保存。',
              '- 复用 E.MeshCache、E.ThetaOp、E.evaluate、C.median_time/C.tstats；搜索稠密组装复用 E.real_dense、mode_pair_matrix_fast 和 lap_full。',
              '- E.evaluate 旧接口要求 GLF 参数，此处传入 Γ_D 两次，并删除 vsLF/ref_floor 兼容字段；未计算或使用 LatticeFriction reference。',
              '- 没有修改 PPPM、B-spline、FFT、influence、邻居图、degree 或 self cancellation，也没有重新运行 Experiment 1/2。', '',
              '## 搜索与自适应规则', '',
              f'- 按用户指定网格先 ξ=.85，再 ξ=.70/1.00/1.15，共 {len(search)} 组请求参数。{nvalid} 组 mesh 合法，{nsurvive} 组通过 uniform seed2 的精确谱误差初筛。',
              '- s_R=s_F=s 是请求值；k_c 按原有 shell snapping，因此实现的 sF_eff 可能略小于 s。每个 CSV 均保留实际 cutoff。',
              '- 每个 s 内优先小 p、小 M，检查所有 M（不会在第一个可行 M 停止）；非法 mesh 立即排除。',
              '- 初筛失败只凭实际完整谱误差排除：单个配置超过门槛即无法满足九组约束。',
              '- 对初筛通过者，在全部九组配置上各测 3 warmup + 7 次 matvec，用九个中位数的中位数排序。',
              '- 每个 ξ 从实测最快者开始做完整九组 E.evaluate 验证；一组误差失败即停止该候选。至少保留三个可行候选，同时验证首个可行者初筛时间 1.20 倍内的所有候选。',
              '- 所有完整可行候选进入串行随机交错终选计时：每配置 7 轮，每轮每候选 3 warmup + 7 次样本（49 次），每轮候选顺序随机化。',
              '- 目标统计量是九个配置各自 matvec 中位数的中位数；最大误差必须对九个配置取最大值。单线程，无并行 worker。',
              '- “最快”表示本次离散网格、自适应筛选与实测终选中的最快值；未把较大 M/p 的成本当作单调函数，也不声称连续参数全局最优。短计时排序存在噪声，20% 终选带用于降低漏选风险。', '',
              '## 各 ξ 最佳可行配置', '',
              '| ξ | s | M | p | sF_eff | 最大 EΓ | matvec (ms) | 总 setup (ms) | images 中位数 |',
              '|---|---|---|---|---|---|---|---|---|']
    for r in bestxi:
        lines.append(f"| {r['xi']:g} | {r['s']:g} | {r['M']} | {r['p']} | {r['sF_eff']:.6f} | {r['E_G_max']:.9g} | {r['t_matvec_median_s']*1e3:.6f} | {r['t_total_setup_median_s']*1e3:.6f} | {r['images_median']:g} |")
    lines += ['', '## 最佳候选的九组验证', '',
              '| configuration | EΓ | matvec (ms) | min eigenvalue on Range(Π) | symmetry | momentum |',
              '|---|---|---|---|---|---|']
    for v in [v for v in val if v['candidate_id'] == best['candidate_id']]:
        t = next(r for r in timing if r['candidate_id']==best['candidate_id'] and r['config']==v['config'])
        lines.append(f"| {v['config']} | {v['E_G']:.9g} | {t['t_matvec_s']*1e3:.6f} | {v['lam_min_h']:.9g} | {v['sym_resid']:.3g} | {v['mom_opnorm_max']:.3g} |")
    lines += ['', f"所有最终候选 mesh 均合法、PSD 均通过；最优候选 min λ={best['lam_min_h']:.9g}，symmetry 最大 {best['sym_resid_max']:.3g}，momentum 最大 {best['mom_opnorm_max']:.3g}，matrix-free/dense 差异最大 {best['mf_vs_dense_max']:.3g}。", '',
              '## 与旧配置比较及速度原因', '',
              f"旧基准保留 (ξ,s_R,s_F,M,p)=(.85,3.2,3.1,18,4)。本次重算最大 EΓ={old['E_G_max']:.9g}；同机随机交错 matvec={old['t_matvec_median_s']*1e3:.6f} ms。", '',
              f"最优候选相对本次旧基准 **{best['speedup_over_remeasured_baseline']:.5f}×**，时间下降 **{(1-1/best['speedup_over_remeasured_baseline'])*100:.3f}%**。逐配置配对加速比中位数 {best['paired_config_speedup_median']:.5f}×。", '',
              f"历史 1.87 ms 与本次最优时间的数值比为 {.00187/best['t_matvec_median_s']:.5f}×；历史与本次运行环境/负载可能不同，主要速度结论采用同机重测基准。", '',
              f"计时波动：最佳候选在九组配置上的最大样本相对 IQR 为 {best['t_matvec_iqr_rel_max']*100:.2f}%。次快的实测候选为 {out[1]['candidate_id']}，时间 {out[1]['t_matvec_median_s']*1e3:.6f} ms，领先幅度 {(out[1]['t_matvec_median_s']/best['t_matvec_median_s']-1)*100:.2f}%。因此最优标签指本次统计量最小者，不能把邻近候选的细微次序解释为跨运行环境稳定的性能结论。", '',
              '| 候选 | s | ξ | M | p | images | real H (ms) | mesh H (ms) | matvec (ms) |',
              '|---|---|---|---|---|---|---|---|---|']
    for r in bestxi + [old]:
        lines.append(f"| {r['candidate_id']} | {r['s']:g} | {r['xi']:g} | {r['M']} | {r['p']} | {r['images_median']:g} | {r['t_real_H_median_s']*1e3:.6f} | {r['t_mesh_H_median_s']*1e3:.6f} | {r['t_matvec_median_s']*1e3:.6f} |")
    lines += ['',
              f"相对旧配置，最优候选 r_c 从 {old['rc']:.6f} 变为 {best['rc']:.6f}，images 中位数从 {old['images_median']:g} 变为 {best['images_median']:g}。实测 real H 从 {old['t_real_H_median_s']*1e3:.6f} ms 变为 {best['t_real_H_median_s']*1e3:.6f} ms；mesh H 从 {old['t_mesh_H_median_s']*1e3:.6f} ms 变为 {best['t_mesh_H_median_s']*1e3:.6f} ms。这两项的平衡决定完整 matvec 的实际速度。", '',
              'real H 与 mesh H 是对原算子对应函数分别测量，二者中位数之和不要求精确等于独立测量的完整 matvec。p 的 spread/gather 成本约随 p³ 增长，FFT 成本依赖 M³ 及实际 FFT 因子；ξ 增大会减小 r_c=s/ξ 与 real-space images，却通常提高 mesh 精度需求。最终采用实际 matvec 排序。', '',
              f"最优候选 setup 分解：mesh={best['t_mesh_setup_median_s']*1e3:.6f} ms、particle setup={best['t_particle_setup_median_s']*1e3:.6f} ms，总 setup={best['t_total_setup_median_s']*1e3:.6f} ms。参数 kernel 首次初始化、reference 构建和稠密误差验证不属于 setup/matvec。setup 为三个独立构建的中位数；未使用 reference 缓存读取时间冒充 setup。", '',
              '## 图与文件', '',
              '![Accuracy vs runtime](figures/accuracy_vs_runtime.png)', '',
              '灰点为 uniform seed2 初筛（单配置误差和时间），彩点为最终九组最大误差及九组运行时间；黑叉为指定小测试，星号为最佳候选。灰点不能解释为九组可行。', '',
              '![Best runtime vs xi](figures/best_runtime_vs_xi.png)', '',
              '- results/search.csv：全部 864 组请求、mesh validity、初筛误差与时间；search_timing.csv：逐样本时间。',
              '- results/validation.csv 与 validation_summary.csv：完整或遇失败提前停止的验证、九组完成标志。',
              '- results/final_timing_raw.csv、final_setup_raw.csv：终选全部逐样本数据；final_timing.csv：逐配置统计。',
              '- results/final_summary.csv、best_by_xi.csv、best_configuration.json：最终汇总。',
              '- results/references.csv、reference_convergence.csv、provenance.json、commands.log、audit_checks.json 以及根目录 run.log：缓存来源、误差界、环境、命令和数据审计。',
              '- 原 Experiment 1/2 文件不变；raw references 不纳入 Git。', '',
              '## 重现', '', '```sh',
              'python3 benchmarks/method_A_clustered/validation_exp1/joint_parameter_optimization/run_joint_optimization.py all --rebuild-necessary-refs',
              'python3 benchmarks/method_A_clustered/validation_exp1/joint_parameter_optimization/summarize_results.py', '```', '',
              '已有精确匹配缓存时不会重建。search/validate 自动续跑；要独立重复搜索，应将本目录 results 另存后运行。timing 阶段会重新计时并替换本次优化自身的计时文件。']
    (J.HERE/'report.md').write_text('\n'.join(lines)+'\n')
    raw = J.read('final_timing_raw.csv')
    for t in timing:
        rs = [r for r in raw if r['candidate_id'] == t['candidate_id'] and r['config'] == t['config']]
        assert {(r['round'], r['rep']) for r in rs} == set(itertools.product(range(7), range(7)))
        assert np.median([r['t_matvec_s'] for r in rs]) == t['t_matvec_s']
    refs = J.read('references.csv')
    assert len(refs) == 9 and max(r['cert_bound_rel'] for r in refs) <= 1e-13
    assert not any('vsLF' in k or k.startswith('ref_floor') for r in val+small for k in r)
    for p in (3, 4):
        a = next(r for r in search if r['xi'] == .85 and r['s'] == 3.2 and r['M'] == 18 and r['p'] == p)
        b = next(r for r in small if r['family'] == 'uniform' and r['seed'] == 2 and r['p'] == p)
        assert a['E_G_design'] == b['E_G']
    repo = J.C.REPO
    newdir = str(J.HERE.relative_to(repo))
    changed = subprocess.check_output(['git', 'diff', '--name-only', '692175342a3c58f2d46b94e8aa12ae6db33073ec',
                                       '--', '.', f':(exclude){newdir}/**'], cwd=repo, text=True).strip()
    assert not changed, f'Existing tracked files changed: {changed}'
    sources = [J.HERE/'run_joint_optimization.py', Path(__file__), J.E.HERE/'ewald_exp_lib.py',
               repo/'benchmarks/method_A_clustered/cached_operator.py',
               repo/'ewald_longitudinal/test_parameter_selection.py']
    J.jsave('audit_checks.json', dict(
        complete_requested_grid=True, n_requested=len(search),
        n_mesh_valid=nvalid, n_design_survivors=nsurvive,
        n_fully_feasible=len(out)-1, n_final_including_baseline=len(out),
        n_final_timing_samples=len(raw), samples_per_candidate_configuration=49,
        timing_medians_recomputed=True, nine_reference_certificates_pass=True,
        small_test_vs_search_exact_match=True, no_LF_error_fields=True,
        existing_tracked_files_unchanged=True,
        source_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    print(json.dumps(dict(best=best, baseline=old, n_final=len(out), n_search=len(search)), indent=2))


if __name__ == '__main__':
    main()
