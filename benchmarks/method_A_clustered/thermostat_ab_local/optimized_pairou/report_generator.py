"""Describe only this fresh three-way experiment's saved evidence."""
from pathlib import Path
import csv
import json
import math
import re
import statistics
import hashlib

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
RESULTS = HERE/'results'
METHODS = ('A-python', 'A-compiled', 'B')


def read(name, directory=RESULTS):
    path = Path(directory)/name
    if not path.is_file():
        return []
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value == '': row[key] = None
            elif value in ('True', 'False'): row[key] = value == 'True'
            else:
                try: row[key] = float(value)
                except (TypeError, ValueError): pass
    return rows


def js(name, default=None):
    path = RESULTS/name
    return json.loads(path.read_text()) if path.is_file() else default


def num(value, digits=4):
    return f'{value:.{digits}g}' if isinstance(value, (float, int)) and math.isfinite(value) else 'N/A'


def maximum(rows, key):
    return max((r[key] for r in rows if isinstance(r.get(key), (float, int))), default=None)


def table(headers, rows):
    escape = lambda x: str(x).replace('|', r'\|').replace('\n', '<br>')
    return '\n'.join(['| '+' | '.join(map(escape, headers))+' |',
                      '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(escape, row))+' |' for row in rows])


def write(name, rows):
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def med(values):
    values = list(values)
    return statistics.median(values) if values else None


def timing_groups():
    grouped = {}
    for row in read('timing_raw.csv'):
        grouped.setdefault((row['config'], row['method']), []).append(row)
    output = []
    for (config, method), rows in sorted(grouped.items()):
        first = rows[0]
        result = {key:first.get(key) for key in ('config','N','family','seed','epsilon','method','alpha','dt','r_d','r_n','substeps','batch')}
        result['n_samples'] = len(rows)
        near=[r['t_near_forward']+r['t_near_reverse'] for r in rows]
        result.update(median_t_near_total=med(near),min_t_near_total=min(near),max_t_near_total=max(near))
        for key in first:
            if key.startswith('t_') or key in ('FFT', 'matvec_count', 'near_actions', 'pair_updates'):
                values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
                if values:
                    result['median_'+key] = med(values)
                    result['min_'+key] = min(values)
                    result['max_'+key] = max(values)
        output.append(result)
    write('timing_summary.csv', output)
    return output


def paired_analysis():
    """Hierarchical bootstrap of paired config/round blocks, never iid reps."""
    import numpy as np
    raw=read('timing_raw.csv'); matched={}
    for row in raw: matched.setdefault((row['config'],row['round'],row['rep']),{})[row['method']]=row
    definitions=[('near_python_over_compiled','A-python','A-compiled','near'),
                 ('total_python_over_compiled','A-python','A-compiled','total'),
                 ('total_compiled_over_B','A-compiled','B','total'),
                 ('total_python_over_B','A-python','B','total')]
    blocks={}
    for (config,rnd,rep), methods in matched.items():
        if set(methods)!=set(METHODS): raise ValueError('Incomplete paired timing triplet')
        for comparison,numerator,denominator,metric in definitions:
            cost=lambda r:r['t_total_step'] if metric=='total' else r['t_near_forward']+r['t_near_reverse']
            value=cost(methods[numerator])/cost(methods[denominator])
            blocks.setdefault((config,rnd,comparison),[]).append(value)
    metadata={r['config']:r for r in raw}
    round_rows=[]
    for (config,rnd,comparison),values in sorted(blocks.items()):
        meta=metadata[config]
        round_rows.append(dict(config=config,N=meta['N'],family=meta['family'],epsilon=meta['epsilon'],round=rnd,
            comparison=comparison,paired_reps=len(values),ratio=med(values),rep_min=min(values),rep_max=max(values)))
    write('paired_round_ratios.csv',round_rows)
    rng=np.random.default_rng(202610270809); B=4000
    per_config=[]
    for config in sorted(metadata):
        meta=metadata[config]
        for comparison,_,_,_ in definitions:
            values=np.array([r['ratio'] for r in round_rows if r['config']==config and r['comparison']==comparison])
            draws=values[rng.integers(0,len(values),size=(B,len(values)))]
            low,high=np.quantile(np.median(draws,axis=1),[.025,.975])
            per_config.append(dict(config=config,N=meta['N'],family=meta['family'],epsilon=meta['epsilon'],comparison=comparison,
                ratio=float(np.median(values)),ci_low=float(low),ci_high=float(high),rounds=len(values),
                round_min=float(values.min()),round_max=float(values.max()),bootstrap_draws=B))
    write('paired_config_ratios.csv',per_config)
    aggregate=[]
    for N,epsilon in sorted({(r['N'],r['epsilon']) for r in raw}):
        configs=sorted({r['config'] for r in raw if r['N']==N and r['epsilon']==epsilon})
        for comparison,_,_,_ in definitions:
            matrix=np.array([[r['ratio'] for r in round_rows if r['config']==config and r['comparison']==comparison] for config in configs])
            config_indices=rng.integers(0,len(configs),size=(B,len(configs)))
            round_indices=rng.integers(0,matrix.shape[1],size=(B,len(configs),matrix.shape[1]))
            samples=matrix[config_indices[:,:,None],round_indices]
            estimates=np.median(np.median(samples,axis=2),axis=1)
            low,high=np.quantile(estimates,[.025,.975])
            point=np.median(matrix,axis=1)
            aggregate.append(dict(N=N,epsilon=epsilon,comparison=comparison,configurations=len(configs),rounds=matrix.shape[1],
                ratio=float(np.median(point)),ci_low=float(low),ci_high=float(high),
                configuration_min=float(point.min()),configuration_max=float(point.max()),bootstrap_draws=B))
    write('paired_aggregate_ratios.csv',aggregate)
    protocol=dict(seed=202610270809,bootstrap_draws=B,ci='approximate percentile 95%',
        estimator='paired per-repetition cost ratios -> median within config/round -> median across rounds -> median across configurations',
        resampling='for each draw resample configurations with replacement, then resample entire round-ratio blocks within each selected configuration; repetitions are not iid bootstrap units',
        per_config_CI='resample whole round-ratio blocks only',variation='configuration_min/max and round_min/max describe observed variation separately',
        inference_limit='conditional local sample and observed frozen configurations; does not certify future hardware states or a population of systems',
        source_sha256=hashlib.sha256((RESULTS/'timing_raw.csv').read_bytes()).hexdigest())
    (RESULTS/'analysis_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    return round_rows,per_config,aggregate


def main():
    groups=timing_groups();rounds,ratios,aggregate=paired_analysis()
    structures=read('structure.csv');accuracy=read('accuracy.csv');mc=read('mc.csv')
    compilation=js('compilation.json',{});profile=js('profile_baseline.json',{})
    environment=js('environment_timing.json',js('environment.json',{}));audit=js('audit.json',{})
    done=js('timing_done.json',{});setup=read('setup_raw.csv')
    costs=[];speed=[]
    for N in (2,8,16,64,128,256):
        rr=[r for r in groups if r['N']==N and r['epsilon']==1e-4]
        values=[med(r['median_t_total_step']*1000 for r in rr if r['method']==m) for m in METHODS]
        costs.append([N,len(rr)//3]+[num(v) for v in values])
        lookup={r['comparison']:r for r in aggregate if r['N']==N and r['epsilon']==1e-4}
        cells=[]
        for key in ('near_python_over_compiled','total_python_over_compiled','total_compiled_over_B'):
            r=lookup.get(key,{})
            cells.append(f"{num(r.get('ratio'))} [{num(r.get('ci_low'))}, {num(r.get('ci_high'))}]")
        speed.append([N]+cells)
    cold=compilation.get('cold',{});cached=compilation.get('fresh_process_cache_load',{})
    coverage=[]
    for N in (2,8,16,64,128,256):
        rr=[r for r in structures if r['N']==N and r['epsilon']==1e-4]
        coverage.append([N,len(rr),', '.join(sorted({r['family'] for r in rr})),len({r['config'] for r in mc if r['N']==N})])
    cirows=[r for r in mc if r['direction']!='COM']
    ciinside=sum(r['variance_ci_low']<=r['ideal_variance']<=r['variance_ci_high'] for r in cirows)
    tests=[]
    for name in ('baseline_tests.log','compiled_tests.log','test_validation.log'):
        path=HERE/'logs'/name;text=path.read_text() if path.exists() else ''
        matches=re.findall(r'Ran (\d+) tests? in ([\d.]+)s',text)
        tests.append([name,matches[-1][0] if matches else 'N/A','OK' if text.rstrip().endswith('OK') else 'N/A'])
    setup_table=[]
    amortization=[]
    group_index={(r['config'],r['method']):r for r in groups}
    for config in sorted({r['config'] for r in setup}):
        rr=[r for r in setup if r['config']==config]
        compiled_setup=med(r['A_compiled_setup'] for r in rr);b_setup=med(r['B_setup'] for r in rr)
        a=group_index[(config,'A-compiled')];b=group_index[(config,'B')]
        a_step=a['median_t_total_step'];b_step=b['median_t_total_step'];savings=b_step-a_step
        crossing=max(1,math.floor((compiled_setup-b_setup)/savings)+1) if savings>0 else None
        amortization.append(dict(config=config,N=a['N'],epsilon=a['epsilon'],A_compiled_setup=compiled_setup,B_setup=b_setup,
            A_compiled_step=a_step,B_step=b_step,A_setup_plus_one_step=compiled_setup+a_step,B_setup_plus_one_step=b_setup+b_step,
            A_over_B_setup_plus_one_step=(compiled_setup+a_step)/(b_setup+b_step),minimum_reused_steps_to_beat_B=crossing))
    write('setup_step_costs.csv',amortization)
    amortization_table=[]
    for N in (16,64,128,256):
        rr=[r for r in amortization if r['N']==N and r['epsilon']==1e-4]
        ks=[r['minimum_reused_steps_to_beat_B'] for r in rr if r['minimum_reused_steps_to_beat_B'] is not None]
        amortization_table.append([N,num(med(r['A_setup_plus_one_step']*1000 for r in rr)),num(med(r['B_setup_plus_one_step']*1000 for r in rr)),
            num(med(r['A_over_B_setup_plus_one_step'] for r in rr)),f"{num(med(ks))} [{min(ks)},{max(ks)}]" if ks else 'N/A',f'{len(ks)}/{len(rr)}'])
    for N in (16,64,128,256):
        rr=[r for r in setup if r['N']==N and r['epsilon']==1e-4]
        setup_table.append([N,len(rr)]+[num(med(r[key]*1000 for r in rr)) for key in ('A_python_setup','A_compiled_setup','B_setup','t_coeff_setup')])
    extra=[]
    for epsilon in (1e-3,1e-5):
        rr=[r for r in groups if r['N']==256 and r['epsilon']==epsilon]
        extra.append([epsilon]+[num(med(r['median_t_total_step']*1000 for r in rr if r['method']==method)) for method in METHODS])
    target=next((r for r in aggregate if r['N']==256 and r['epsilon']==1e-4 and r['comparison']=='total_compiled_over_B'),{})
    acceleration=next((r for r in aggregate if r['N']==256 and r['epsilon']==1e-4 and r['comparison']=='total_python_over_compiled'),{})
    target_times={method:med(r['median_t_total_step']*1000 for r in groups if r['N']==256 and r['epsilon']==1e-4 and r['method']==method) for method in METHODS}
    if target.get('ci_high',float('inf'))<1:
        verdict='优化 A 在这九个配置和本次运行中快于 B，paired ratio 的近似区间完全低于 1。'
    elif target.get('ci_low',0)>1:
        verdict='优化 A 在这九个配置和本次运行中仍慢于 B，paired ratio 的近似区间完全高于 1。'
    else:
        verdict='优化 A/B 的 paired ratio 区间跨过 1，本次数据不能稳定宣称其中一方更快。'
    compiled256=[r for r in read('timing_raw.csv') if r['N']==256 and r['epsilon']==1e-4 and r['method']=='A-compiled']
    long_share=med((r['t_long_damp']+r['t_long_noise'])/r['t_total_step'] for r in compiled256)
    startup=js('cold_process_start.json',{})
    text=f'''# Combination A 顺序 Pair OU 编译优化：本地报告

本实验以 PR #8 head `bce73847f2968c27e1df5e2ff4be57393c4d2ddd` 为只读基线。只改变 A 的 near-edge sweep 实现，全部 q/GammaS/GammaL/Gamma、Gaussian 输入、r_d/r_n、dt 与 splitting substeps 均与对应数值对照相同。B 与 Grid PSD 源码未改。下述成本只取本目录 fresh three-way batch，不与旧 PR #8 的 79.8/11.6 ms 混算速度比。

主要结果：N256、epsilon=1e-4 的配置中位总步从新测 Python A 的 {num(target_times.get('A-python'))} ms 降到 compiled A 的 {num(target_times.get('A-compiled'))} ms；同批 B 为 {num(target_times.get('B'))} ms。paired Python/compiled 整步加速 {num(acceleration.get('ratio'))}，compiled/B={num(target.get('ratio'))}，95% block-bootstrap CI [{num(target.get('ci_low'))}, {num(target.get('ci_high'))}]。{verdict}

## 先定位瓶颈，再保持算法实现编译

新跑的预优化诊断 profile 在 N256 uniform seed2、{profile.get('edge_count','N/A')} 条有序聚合边上测量 {profile.get('unprofiled_samples','N/A')} 次未插桩整步，near 双 sweep 占总时间中位比例 {num((profile.get('near_fraction_median') or 0)*100)}%，总步中位数 {num((profile.get('median_total_s') or 0)*1000)} ms。另用 cProfile 记录 {profile.get('profiled_steps','N/A')} 步，原始 `.prof`、函数表和 load 见 logs/results。嵌套 cumulative profiler 时间相互重叠，不能相加；该阶段只确认优化对象，不是正式速度比的分母。

完整 3x3 image-summed PSD 边的相对 OU 可写作 `dp=D r+S eta`，`D=0.5 U expm1(-2 tau lambda/m) U^T`，`S=0.5 U sqrt(2m theta[-expm1(-4 tau lambda/m)]) U^T`；再顺序执行 `p_i+=dp; p_j-=dp`。这与原式 `0.5(r_plus-r)` 同义，避免 `exp()-I` 消减。D/S 在 tau、m、theta 已知时缓存，循环内只做 3x3 action 与顺序粒子更新。C 循环严格保留 forward/reverse 顺序与原边索引，不并行或重排共享粒子边、不假设 rank-one、不复用本应独立的 Gaussian。

graph/边张量/eigensystem 不可变，位置或图改变要求显式 rebuild；每个实例的 coefficient cache 区分 tau/m/theta。参数更新、同形状图更新、零/小 tau、空边、rank-deficient/full-rank、输入不变性和 Python fallback 均由新增测试覆盖。

另以实际 N16 uniform seed2 的生产图做 {js('cache_validation.json',{}).get('parameter_combinations','N/A')} 种 tau/温度/质量组合、双方向，共 {js('cache_validation.json',{}).get('rows','N/A')} 行 cache 验证；先 miss 后 repeat hit，并实际扰动 q、重建 GridPSDGamma/multi-image edges、使用新 token 清空并重建 coefficients，同时验证旧图/旧缓存不变。原始 fixture/路径与参数 SHA 见 `cache_validation.*`；最大路径差 {num(js('cache_validation.json',{}).get('path_relative_error_max'))}。零 tau 的 D=S=0 在不同参数下具有相同系数哈希是合法情形，不能误判为缓存未更新。

## 实际编译环境与启动成本

实际后端为顺序 C/ctypes。Numba import 检查为 `{environment.get('numba_status',{})}`；package versions 为 `{environment.get('package_versions',{})}`。这是实际环境结果，未为 Numba 改装 NumPy。compiler 为 `{cold.get('compiler','N/A')}`，flags 为 `{cold.get('flags',[])}`，关闭 fast-math/FP contraction，无 OpenMP；完整 compiler version/command、源与库 SHA 见 `compilation.json`。原 Python EdgeOU 可选择为 reference/fallback，正式 optimized 样本要求 active_backend=compiled。

真实 cold compile subprocess={num(cold.get('t_compile_s'))} s；首次 CDLL load={num(cold.get('t_load_s'))} s；整个 cold entry={num(compilation.get('cold_entry_total_s'))} s。独立新进程使用现有库的 cache load={num(cached.get('t_load_s'))} s，整个该 Python 进程（含 imports）={num(compilation.get('fresh_process_total_s'))} s。这些不是 steady-state 步成本。

另在空二进制 cache 中启动全新 Python 进程，真实首次进程启动总 wall 为 {num(startup.get('fresh_process_wall_s'))} s（`cold_process_start.json`）；该次包含 imports、compiler lookup、编译和 load，排除物理 table/config setup。它与前面的 backend-entry/缓存进程时间分别记录，未从其它阶段时间相加伪造首次启动。

fresh configuration setup 每配置 {js('timing_protocol.json',{}).get('setup_reps','N/A')} 次，单位 ms：

{table(['N','setup记录','A Python setup','A compiled setup','B setup','compiled coeff setup'],setup_table)}

setup 包含真实 table/config/image-summed edges、constructor/eigh/immutable copies 和 coefficients；compiled constructor 的 warm-library检查/load 也计入。constructor 已包含 eigh，不能再把其诊断 eigh 时间重复加到总 setup。共同 radial-kernel cache、进程启动和验证用 dense 组装排除；不声称全进程冷启动。改变 q 后需要重新构图/eigh/coeff，不能把旧 frozen setup 当作免费。steady step 只使用预热后的已缓存配置。

若每次 q 更新都需要 fresh setup，必须核算 `setup + K*step`。下表将**同一配置**的 setup 中位数与 step 中位数相加，再在配置间取中位数；这是分相成本核算，未声称另做了连续端到端计时。K 为同一 frozen 图需要复用多少步后 compiled A 才便宜于 B 的点估计，未计入一次性 cold compile/启动，且不是置信保证：

{table(['N','A setup+1步 ms','B setup+1步 ms','配对成本比 A/B','复用步数K中位数 [范围]','可摊销配置数'],amortization_table)}

因此已缓存 frozen step 的优势不能直接解释为位置每步变化的生产轨迹优势。`setup_step_costs.csv` 保留每配置核算；若 step 中位数没有优势，K 显示 N/A。

## 实际精度与不变量覆盖

{table(['N','epsilon=1e-4配置数','族','MC配置数'],coverage)}

实际 fresh accuracy 为 {len(structures)} 个配置、{len(accuracy)} 行三方法/alpha 对照，alpha=0.1/0.5/1/2。N16/64/256 保留各九配置，N128 每族 seed2；另有 N256 uniform seed2 的 epsilon=1e-3/1e-5。所有 saved q/G/GS/GL/p/z 与 PR8 bitwise identity，原始边序和完整 image-summed K 一致；沿用 PR8 最终秩，含 N2/B 已修正的 noise rank3，没有重新调低误差预算。

compiled/Python A 全步最大相对路径差 {num(maximum(accuracy,'original_vs_compiled_path'))}，near forward/reverse 最大差 {num(max(maximum(accuracy,'forward_original_vs_compiled') or 0,maximum(accuracy,'reverse_original_vs_compiled') or 0))}；COM 最大改变量 {num(maximum(accuracy,'COM_error'))}。exact dense FDT 代数 residual 最大 {num(maximum(accuracy,'dense_fdt_residual'))}。相同 ordered ideal-A 均值/covariance 与 full exact B 的 splitting 误差和 dt/2 收敛重新计算于 `splitting.csv`，优化没有删除这一动力学误差。

`accuracy.csv`/`mode_decay.csv` 的 mean_action_error 是关闭全部噪声后的确定性 damping action，**不是**实际 finite-Lanczos A 的随机条件均值。有限 Krylov damping 映射依赖输入，near 噪声经非线性中间 action 传播可能产生 Jensen 效应；实际条件 mean 只能由 MC sample mean/paired mean 及 SE 检查。只有理想线性 Gaussian A 的 R_A/C_A 具有上述精确均值/covariance 解释。

理想 A 的各 exact 子步保持固定 COM 的 Maxwell covariance Pi，组合 covariance 为 `Pi-R_A Pi R_A^T`；该代数表达式的 residual 只是构造一致性，不是 finite-Krylov 精确 FDT。有限 Lanczos 基随输入 Gaussian 改变，不是固定线性 F，不能逐列组装后把 F F^T 称作算法的精确 covariance。

fresh MC 保存 {len(mc)} 行方向统计，conditional/equilibrium、slow/fast/random/COM 三方法共用同源独立 draws。原始样本、理想同噪声参考、种子与 Gaussian SHA/recipe 均保存。编译/原 Python 最大相对样本差 {num(maximum(read('mc_pairwise.csv'),'compiled_original_relative_max'))}；finite-Krylov 对理想同噪声路径最大相对差 {num(maximum(mc,'coupled_ideal_path_max'))}，COM 最大 {num(maximum(mc,'COM_max'))}。非 COM 理想方向方差落入逐点近似 95% CI 为 {ciinside}/{len(cirows)}；该计数只描述样本，未作多重修正，不能作为严格 FDT proof。mean/variance/fourth moments、SE/CI、paired mean/second-moment discrepancy 见 MC CSV；二阶 paired 差围绕已知理想均值，包含均值偏差平方，不能当作 covariance 算子界。

这些输出全部重新执行，A 两实现使用同源噪声；本次 global seed 与旧 PR8 仅相差 1，逐样本种子集合可能大量重叠，因此不声称相对于旧 PR8 是统计独立的复现实验，也不合并两次样本数。当前 SE/CI 只使用本次输出，在本次 run 内不同 sample 独立。

A-python/A-compiled 的同源近似相同输出也不构成额外独立统计重复；192 行是方法/phase/方向汇总行数，不是 192 次独立 FDT 实验。

{table(['测试日志','测试数','结果'],tests)}

初次新增测试日志 `compiled_tests_initial.log` 仍保留：一个随机输入的 graph-change 测试把输出差必须大于 0.01 当作任意阈值，实际差为 0.0008888。最终测试改为直接激发被改变的本征方向并关闭该项噪声，排除漂移/噪声偶然抵消，随后 16 项通过；这次调整针对测试 fixture，没有改变编译 kernel 或精度预算。

## fresh 三方完整总成本

正式 batch 共 {done.get('systems','N/A')} 配置、{done.get('rounds','N/A')} rounds × {done.get('reps','N/A')} paired reps ×3方法，{done.get('rows','N/A')} 行。随机顺序交错所有方法；每个 sample 各方法重设相同种子，两个 A 使用相同 z/前后边 Gaussian，B 仅生成需要的 z；全部 RNG 构造/实际所需 Gaussian、近场、FFT、Krylov、重正交计入整步。hash/序列化及 paired path comparison 在计时之外。每配置先取总时间中位数，再对同 N 的配置取中位数（ms）：

{table(['N','配置数','A Python','A compiled','B full'],costs)}

下表为 paired ratio 与近似 bootstrap 95% CI。每次 repetition 先做同 sample 成本比，再在 config/round 中取中位数，再在 round、配置间取中位数。CI 用4000次分层 bootstrap：配置作为 block 重采样，每个抽中配置再整体重采样其 round-ratio blocks；没有把相邻 reps 假定为独立硬件状态。配置 min/max、round min/max 另存，不能把变化范围当作 CI。N2/8 单配置的 CI 只反映该配置的 round 变化。

各方法配置中位时间与配对比率的取中位顺序不同，不能把上表两个时间直接相除来替代 paired ratio。

{table(['N','near Python/compiled','total Python/compiled','total compiled/B'],speed)}

Python/compiled 大于1表示编译优化更快；compiled/B 小于1表示优化 A 在本地测试域比 B 更快。CI 条件于这些 frozen 配置和本次运行，不保证其它硬件、未来状态或生产轨迹。所有 ratio 来自新 batch，附加 epsilon 不混入主表：

{table(['N256 epsilon','A Python ms','A compiled ms','B full ms'],extra)}

`t_reorth`/`t_ritz_eigh`/FFT 均已包含在 Lanczos 阶段，`t_full_B` 已包含整步，禁止再与分相重复相加。阶段图只堆叠 RNG、near sweeps、long/full damping/noise 与实测总时间的剩余项。

## 图、限制与审计

三张 PNG+SVG：`near_sweep_speedup`、`three_way_total_time`、`three_way_phases`。前两张展示实际 near/完整步骤改进及区间，分相图解释剩余成本；绝对时间误差条为配置范围，ratio 误差条为上述 bootstrap CI。

环境 Python {str(environment.get('python','N/A')).splitlines()[0]}，NumPy {environment.get('numpy','N/A')}，SciPy {environment.get('scipy','N/A')}，{environment.get('platform','N/A')}。请求单线程，实际 native pool 可验证性为 `{environment.get('native_thread_verification','N/A')}`；系统负载见 `system_load.json`。CPU/ctypes/Python 驱动与有限本地覆盖构成本结论边界。

accuracy耗时 {num(js('accuracy_done.json',{}).get('elapsed_s'))} s，MC {num(js('mc_done.json',{}).get('elapsed_s'))} s，timing {num(done.get('elapsed_s'))} s。审计状态 {audit.get('status','N/A')}；基线全部 tracked SHA、源/库/输入/Gaussian/cache/旧秩关联、原始样本统计与 evidence manifest 见 `audit.json`。

本次没有 HPC 或长轨迹结果。是否需要 HPC 取决于更大 N、并行 FFT 与轨迹目标；单凭 frozen 本地速度比不足以决定。复现见 [README.md](README.md)，使用 source-only 新目录保留所有现有证据。

原 near 双 sweep 的主要瓶颈可以通过移除 Python per-edge 循环解决；优化后 N256 A 的 long damping/noise 已占整步中位比例 {num(long_share*100 if long_share is not None else None)}%，剩余 FFT/Krylov、RNG 与驱动成本仍存在，所以不能把整个方法的成本都归结为 Python overhead。本次 frozen 验收已在本地完成，不需要 HPC 才能得出上述结论。若下一步目标是更大 N 或生产轨迹吞吐，则 HPC benchmark 有价值，且必须重新测并行 FFT、配置更新及积分器整体成本；本实验不预设其速度结果。
'''
    (HERE/'report.md').write_text(text)
    print(f'Report generated: {len(structures)} systems, {len(mc)} MC rows, {len(groups)} timing groups')


if __name__ == '__main__':
    main()
