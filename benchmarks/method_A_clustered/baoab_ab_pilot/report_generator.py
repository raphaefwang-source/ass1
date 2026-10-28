"""Summarize actual Stage0/1 evidence; never supplies unrun results."""
from evidence import HERE,RESULTS,read_csv,save
from pathlib import Path
import json
import numpy as np

def js(name):return json.loads((RESULTS/name).read_text())
def fmt(x):return f'{x:.6g}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])

def main():
    prep=[js(f'initial_{i}.json') for i in range(3)]
    stage=js('stage0_done.json');sel=js('spatial_selection.json');par=sel['params'];ranks=js('ranks_final.json')
    production=read_csv(RESULTS/'production_steps.csv');timing=read_csv(RESULTS/'timing_steps.csv')
    rounds=read_csv(RESULTS/'timing_rounds.csv');visited=js('visited_done.json')
    # Means answer full end-to-end work/step; variability across rounds is a range, not CI.
    summaries={}
    for method in ('A','B'):
        rr=[r for r in timing if r['method']==method]
        pp=[r for r in production if r['method']==method]
        summaries[method]=dict(n_timing=len(rr),n_production=len(pp),
            full_mean_ms=1000*np.mean([r['t_baoab_s'] for r in rr]),
            full_median_ms=1000*np.median([r['t_baoab_s'] for r in rr]),
            production_mean_ms=1000*np.mean([r['t_baoab_s'] for r in pp]),
            round_mean_min_ms=1000*min(r[method+'_mean_s'] for r in rounds),
            round_mean_max_ms=1000*max(r[method+'_mean_s'] for r in rounds),
            dynamic_setup_ms=1000*np.mean([r['t_dynamic_setup_s'] for r in rr]))
    ratios=[r['A_over_B'] for r in rounds]
    spatial=[js(f'initial_{i}_spatial.json') for i in range(3)]+[r['spatial'] for r in visited['records']]
    action=[js(f'action_validation_initial_{i}_final.json') for i in range(3)]+[r['rank_validation'] for r in visited['records']]
    max_spatial=max(r['conservative_relative_error_bound'] for r in spatial)
    max_action=max(r['max_relative_action_error'] for r in action)
    max_com=max(r['momentum_norm'] for r in production+timing)
    peak=max(r['peak_rss_bytes'] for r in production+timing)/1024**2
    phase_fields=[('LJ 两次 force','t_LJ_forces_s'),('q 相关动态 setup','t_dynamic_setup_s'),
        ('RNG','t_rng_s'),('完整 O','t_ou_wall_s'),('其它 BAOAB','t_baoab_other_s')]
    phase_rows=[]
    for label,key in phase_fields:
        phase_rows.append([label]+[fmt(1000*np.mean([r[key] for r in timing if r['method']==m])) for m in ('A','B')])
    setup_fields=[('近场邻居/全镜像','t_near_neighbor_images_s'),('近场权重/degree','t_near_weights_degree_s'),
        ('mesh assignment','t_mesh_assignment_s'),('mesh degree','t_mesh_degree_s'),
        ('A 边张量聚合','t_edge_aggregate_s'),('A edge constructor（含 eigh）','t_edge_constructor_s'),
        ('A coefficients','t_coefficient_setup_s')]
    setup_rows=[[label]+[fmt(1000*np.mean([r[key] for r in timing if r['method']==m])) for m in ('A','B')] for label,key in setup_fields]
    split=[]
    for i in range(3):
        for r in read_csv(RESULTS/f'splitting_initial_{i}_final.csv'):
            split.append([i,int(r['substeps']),fmt(r['mean_operator_error']),fmt(r['covariance_relative_error'])])
    a=summaries['A'];b=summaries['B']
    prep_wall=sum(r['total_wall_s'] for r in prep)
    # Hypothetical planning extrapolation only, no later-stage experiments launched.
    one_dt_s=16*(10/.005)*(a['full_mean_ms']+b['full_mean_ms'])/1000
    four_dt_s=15*one_dt_s
    run_files=list((HERE/'inputs').glob('production_*.npz'))+list((HERE/'inputs').glob('timing_round*.npz'))
    saved_trajectory_bytes=sum(p.stat().st_size for p in run_files)
    estimate_four_dt_storage_GiB=saved_trajectory_bytes/(len(production)+len(timing))*960000/1024**3
    save(RESULTS/'summary.json',dict(A=a,B=b,round_ratio_mean=float(np.mean(ratios)),
        round_ratio_min=min(ratios),round_ratio_max=max(ratios),spatial_max=max_spatial,
        action_max=max_action,COM_max=max_com,peak_rss_MiB=peak,preparation_wall_s=prep_wall,
        planning_only_estimate=dict(one_dt_16starts_seconds=one_dt_s,
            hypothetical_four_halved_dt_seconds=four_dt_s,
            hypothetical_four_dt_trajectory_storage_GiB=estimate_four_dt_storage_GiB,executed=False)))
    text=f'''# 运动粒子 BAOAB A/B：本地 Stage 0 + Stage 1 报告

基线为 PR #9 `d6a8d208bcdc30731079715dc4c8e925b4fb2f8b`，工作分支 `codex/baoab-ab-stage01`。本次只在本机 Apple M2 / 8 GiB / macOS 27.0.1 执行。完整环境见 `environment_initial.json`；未执行 Stage 2/3、HPC、SSH、远程计算或正式长期统计。

本次实际完成 **{len(production)} 个生产 BAOAB 步**（3 个独立初态 × A/B × 200），另完成 **{len(timing)} 个独立动态计时步**（6 interleaved rounds × A/B ×40）。所有粒子真实移动，每步在新的 q_mid 构建算子。下表来自同一次独立 interleaved batch，单位 ms/完整 BAOAB 步，包含两次 LJ force、全动态 setup、RNG、完整 OU 和漂移/kick：

{table(['方法','完整 wall 均值','每步中位数','6轮均值范围','动态 setup 均值','生产轨迹均值'],[[m,fmt(summaries[m]['full_mean_ms']),fmt(summaries[m]['full_median_ms']),f"{fmt(summaries[m]['round_mean_min_ms'])}–{fmt(summaries[m]['round_mean_max_ms'])}",fmt(summaries[m]['dynamic_setup_ms']),fmt(summaries[m]['production_mean_ms'])] for m in ('A','B')])}

6 个 round 的 A/B 平均成本比均值为 {fmt(np.mean(ratios))}，范围 [{fmt(min(ratios))}, {fmt(max(ratios))}]。该范围是实际 round 变化，不是置信区间。该比率仅描述本机短 pilot 的动态性能，**不是等动力学精度排名，也不作 A/B 最终 winner 结论**。

## 物理模型与实际积分

N=256，rho=.75，L={fmt(par['L'])}，m=sigma=epsilon_LJ=kBT=1。Candidate 1/S1L 使用 kappa=.7 与 c=.5×1.3×exp(.7×1.3)={par['c']:.16g}。gamma=.5 是核幅度定义，不是预平衡 scalar Langevin 的 gamma=1。LJ 使用 r_on=2、r_cut=2.5 的 quintic switch，force 同时包含 switch derivative。只有保守 LJ 使用 minimum image；摩擦保留所有 cutoff 内周期镜像，聚合同一 i<j 的完整3×3张量，没有把 self-image 变成自边。

每步执行 p_half=p+dt F(q)/2；q_mid=q+dt p_half/2；在 q_mid 更新 Gamma_h 并冻结整个 O_dt；q_next=q_mid+dt p_after/2；p_next=p_after+dt F(q_next)/2。dt=.005，每条200步轨迹总时长1。wrapped q 与 unwrapped q 分开保存。

复用既有 GridPSDGamma、ProjectedMesh、full/long Lanczos 和 PR9 顺序 C Pair OU。A/B 共享完全相同物理 kernel、盒子与参数表；只有 A 的 splitting 与 B 的 full OU 不同。每步重建近场镜像边、assignment/degree，A 额外重建 eigensystem 和 coefficients。物理 FFT 表跨步共享，位置相关数据不跨步复用。

## Stage 0：空间、算子和 splitting 分开验证

历史候选 xi=.85,sR=sF4.1,M24,p8 在新物理核下空间误差约1.20e-2，未通过。较细的 .85/M40/p10 仍约2.21e-3。实际选定 xi={par['xi']},sR=sF={par['sR']},M={par['M']},p={par['p']}，未进行 fastest-parameter 搜索。

独立参考直接累加 unsplit c exp(-.7r)/r 的完整周期图，截断 R=45，记录 R33.75/40.5/45 annulus 收敛和解析无限尾界；没有使用 mesh/Ewald split 构造 reference。三个初态和六个实际访问 q_mid 的最大保守相对谱误差为 **{max_spatial:.4g}**，均满足2e-7。每个几何都保存独立矩阵/镜像计数/尾界、q SHA 和结构诊断；初始可行性几何另存。物理 Fourier k=0 保留；完整 mesh self cancellation 与近场无自边通过测试。实际 projection_active=False 表示本组选定参数无需负块修正，算法仍使用既有 Grid PSD 实现。

实际选择的秩：A/long r_d={ranks['A']['r_d']},r_n={ranks['A']['r_n']}；B/full r_d={ranks['B']['r_d']},r_n={ranks['B']['r_n']}。两方法分别验证后恰好相同，并非强制同秩。slow/fast/random/actual momentum/actual Gaussian/主轴低波数 L/T 方向均对照 dense 矩阵函数；检查实际 FFT actions，最大相对 action 误差 **{max_action:.4g}**，阈值2e-8。六个访问状态复原真实 p_half 和该生产步 Gaussian（A 包含真实 near-forward 输入）。有限方向和采样状态验证不构成全域 rank 误差界。

A 保持1个内部 splitting step。下表是理想线性 splitting 相对 full exact OU 的误差；该误差不同于上面的 finite Krylov 误差：

{table(['初态','内部步数（实际采用1）','mean operator absolute error','covariance relative error'],split)}

**单个 splitting step 并未达到相对 full exact OU 的合并2e-8误差预算。**例如初态0的 mean operator absolute error约2.39e-7，covariance relative error约4.05e-6。因此严格空间与Krylov检查通过不代表完整A算子已满足等动力学精度；保留substep1仅用于本轮pilot，后续正式精度选择尚未执行。

负 Ritz 使用原有 scale-aware roundoff 规则；非微小负值立即抛出 PSDViolation。固定 frozen O 的理想 Maxwell/FDT 恒等式不作为全 BAOAB Gibbs 证明；本次未完成长期平衡、时间步等价性或动力学收敛认证。

## 独立初态和短轨迹

每个初态从 FCC 和独立 Maxwell 动量开始，用不同 seed 的 COM-constrained scalar Langevin+LJ 预平衡6000步×.005=30时间单位。下面为真实生成记录，准备时间单列，不计入生产单步：

{table(['初态','seed','准备wall(s)','末态Tkin','末态U/N','最小距离'],[[i,r['seed'],fmt(r['total_wall_s']),fmt(r['final_Tkin']),fmt(r['final_U_per_particle']),fmt(r['final_min_pair_distance'])] for i,r in enumerate(prep)])}

准备总 wall={fmt(prep_wall)}s。这是实际预平衡协议，未证明有限时间内精确 canonical mixing。A/B 对每个初态使用完全相同起始 q,p；内部 Gaussian 流彼此独立且可由保存的 PCG64 state 重放。每步记录 Tkin、U/N、总动量、最小距离、位移、真实使用秩、projection 状态和 q/输入/输出/RNG SHA。原始 q_unwrapped/q/p/q_mid 全部保留在 inputs NPZ。

全部生产与计时步最大总动量范数={max_com:.4g}；最小粒子距离={min(r['min_pair_distance'] for r in production+timing):.6g}；最大进程 peak RSS={peak:.5g} MiB。温度/势能随短轨迹的实际变化见 trajectory_observables，不能据此宣称全程 canonical 或正式动力学一致。

## 动态成本分解

以下为 interleaved batch 的 disjoint 大阶段均值，单位 ms：

{table(['阶段','A','B'],phase_rows)}

本次A的q相关setup占完整wall约 {100*a['dynamic_setup_ms']/a['full_mean_ms']:.3g}%；B的完整OU占约 {100*np.mean([r['t_ou_wall_s'] for r in timing if r['method']=='B'])/(b['full_mean_ms']/1000):.3g}%。这些是正式研究资源规划时必须计入的实际成本；仅看已缓存Pair OU sweep无法代表整体吞吐。A构造内含每步完整3×3边eigh/不可变数组拷贝，B每次full matvec还包括近场action。保留本次实现和参数，没有启动生产级联合优化。

动态 setup 子阶段均值如下；其余 Python 构造/包装/哈希开销包含在完整 setup，未遗漏进完整 BAOAB wall：

{table(['setup子阶段','A','B'],setup_rows)}

![动态setup分解](figures/dynamic_setup.png)

![完整运动粒子BAOAB时间](figures/baoab_total_time.png)

![短轨迹实际观测](figures/trajectory_observables.png)

对应SVG矢量文件保存在同名 figures/*.svg，原始绘图CSV和SHA manifest一并保存。

edge eigh 属于 edge constructor；setup FFT 属于 mesh/setup；OU FFT、reorth、Ritz eigensolve 属于完整 OU。CSV 保存这些嵌套诊断，**不可重复相加**。完整 compute wall 在第二次 momentum kick 后结束；构图q token hash以及force返回势能/最小距离已包含。额外证据哈希、诊断汇总与序列化另记；完整 run loop wall也保存。物理表的一次性构建 wall另列于 production_table.json/timing_done.json，初态准备和空间/rank审计也不计入单步。

独立计时的每个 round 都让 A/B 从同一初态 q,p 开始，分别真实推进40步；方法顺序由固定独立 RNG 随机化。每轮重新开始不复用相同 q 的 frozen/replay O 时钟。环境请求所有 BLAS/native线程为1；macOS Accelerate 的实际 native线程数不能被 threadpoolctl 完整观测，不能把环境变量当作独立硬件测量。计时期间没有其它本任务数值进程。

## 验证、复现、资源规划与停止线

复用实现回归54项、新实现测试27项通过；测试与初次失败修正日志均保留。独立 evidence 审计见 independent_audit.json，核对全部基线 SHA、原始轨迹、真实 q_mid、RNG重放、rank/spatial证据与时间分相。三张 PNG+SVG：dynamic_setup、baoab_total_time、trajectory_observables。生成图所用实际CSV与manifest一并保存。

基于本机上述平均成本，仅估算16初态×A/B×T=10 在单个dt=.005约需 {one_dt_s/3600:.3g} wall小时（不含准备/验证/IO）。若四个dt假设为 .005,.0025,.00125,.000625，共960000个A/B步骤，同成本/步线性外推约 {four_dt_s/3600:.3g} 小时；更小dt的rank/系统负载可能改变，不能当作实测。按本次NPZ实际压缩字节/步外推，全量轨迹约 {estimate_four_dt_storage_GiB:.3g} GiB，另需CSV/RNG/参考矩阵；未压缩四组 q/q_unwrapped/p/q_mid 为24576 bytes/step、约22 GiB。串行单轨迹可参考本次实测peak RSS，16条并发不可按同一峰值当总内存。HPC性能、MPI/FFT scaling未测，不能预测加速数字。**这些只用于规划，未启动任何正式16-replica或远程作业。**

本次停止于 Stage 0 + Stage 1。dt=.005 未被认证为已收敛时间步；尚未执行16-replica四步长、T=10、8/11条长平衡、T=100、VACF/MSD/RDF正式CI或1%等价性研究，未做生产参数联合优化或最终winner选择。
'''
    (HERE/'report.md').write_text(text)
    print('Wrote report.md and summary.json',flush=True)

if __name__=='__main__':main()
