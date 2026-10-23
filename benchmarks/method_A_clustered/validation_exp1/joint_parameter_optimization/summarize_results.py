#!/usr/bin/env python3
"""Generate summaries, both figures, and the report from measured CSVs only."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ass1_joint_mpl')
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


def main():
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
