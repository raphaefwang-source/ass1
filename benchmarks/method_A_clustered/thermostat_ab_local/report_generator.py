"""Summarize only saved local evidence; never fills unrun experiments with values."""
from pathlib import Path
import csv
import json
import math
import re
import statistics

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'


def read(name):
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value == '':
                row[key] = None
            elif value in ('True', 'False'):
                row[key] = value == 'True'
            else:
                try:
                    row[key] = float(value)
                except (ValueError, TypeError):
                    pass
    return rows


def json_read(name, default=None):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else default


def number(value, digits=3):
    return f'{value:.{digits}g}' if isinstance(value, (float, int)) and math.isfinite(value) else 'N/A'


def maximum(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return max(values) if values else None


def write_csv(name, rows):
    RESULTS.mkdir(exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (RESULTS / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def timing_summary():
    grouped = {}
    for row in read('timing_raw.csv'):
        grouped.setdefault((row['config'], row['method']), []).append(row)
    summary = []
    for (config, method), rows in sorted(grouped.items()):
        times = [row['t_total_frozen_step'] for row in rows]
        first = rows[0]
        item = {key: first.get(key) for key in ('config', 'N', 'family', 'seed', 'method', 'alpha', 'dt', 'r_d', 'r_n', 'substeps', 'batch')}
        item.update(n_repetitions=len(times), median_total_s=statistics.median(times),
                    min_total_s=min(times), max_total_s=max(times))
        for key in first:
            if key.startswith('t_') or key.endswith('_s') or 'matvec' in key or key in ('fft_calls', 'near_actions'):
                values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
                if values:
                    item['median_' + key] = statistics.median(values)
        summary.append(item)
    write_csv('timing_summary.csv', summary)
    paired = []
    for config in sorted({row['config'] for row in summary}):
        methods = {row['method']: row for row in summary if row['config'] == config}
        if set(methods) == {'A', 'B'}:
            a, b = methods['A'], methods['B']
            paired.append(dict(config=config, N=a['N'], family=a['family'], seed=a['seed'],
                               A_median_s=a['median_total_s'], B_median_s=b['median_total_s'],
                               A_over_B=a['median_total_s'] / b['median_total_s']))
    write_csv('timing_comparison.csv', paired)
    return summary, paired


def table(headers, rows):
    def cell(value):
        return str(value).replace('|', r'\|').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows])


def main():
    structures, selection = read('structure.csv'), read('selection.csv')
    splitting, checks, mc = read('splitting.csv'), read('matrixfree_checks.csv'), read('mc.csv')
    summary, paired = timing_summary()
    protocol = json_read('protocol.json', {})
    accuracy_done, mc_done, timing_done = (json_read(name, {}) for name in ('accuracy_done.json', 'mc_done.json', 'timing_done.json'))
    audit = json_read('audit.json', {})
    stopping = json_read('stopping_summary.json', {})
    uncensored = json_read('uncensored_stopping_summary.json', {})
    refinement = json_read('rank_refinement_summary.json', {})
    stop_rows = read('stopping_diagnostics.csv')
    stop_over_tolerance = sum(r.get('step_change') is not None and r['error'] > r['tolerance'] for r in stop_rows)
    coupling = read('coupling_checks.csv')
    env = json_read('environment_timing.json', json_read('environment_accuracy.json', {}))
    failures = json_read('failures.json', [])
    test_log = HERE / 'logs' / 'tests.log'
    test_matches = re.findall(r'Ran (\d+) tests? in ([\d.]+)s', test_log.read_text()) if test_log.exists() else []
    tests_text = (f"{test_matches[-1][0]} 项单元/空间测试通过，耗时 {test_matches[-1][1]} s（见 logs/tests.log）" if test_matches and test_log.read_text().rstrip().endswith('OK') else '测试日志 N/A 或未通过，见 audit')
    coverage = []
    for N in (2, 8, 16, 64, 128, 256):
        ss = [r for r in structures if r['N'] == N and r.get('epsilon') == 1e-4]
        sels = [r for r in selection if r['N'] == N and r.get('epsilon') == 1e-4]
        coverage.append([N, len(ss), ', '.join(sorted({r['family'] for r in ss})) or 'N/A',
                         f"{sum(bool(r.get('available')) for r in sels)}/{len(sels)}" if sels else 'N/A',
                         len({r['config'] for r in mc if r['N'] == N}),
                         len({r['config'] for r in summary if r['N'] == N})])
    valid_variance = [r for r in mc if r['direction'] != 'COM']
    ci_covered = sum(r['ci_low'] <= r['ideal_method_variance'] <= r['ci_high'] for r in valid_variance)
    outside = [r for r in valid_variance if not r['ci_low'] <= r['ideal_method_variance'] <= r['ci_high']]
    moments_rows = []
    for phase in ('conditional', 'equilibrium'):
        for method in ('A', 'B'):
            rows = [r for r in valid_variance if r['phase'] == phase and r['method'] == method]
            mean_z = [abs(r['mean']-r['ideal_method_mean'])/r['mean_se'] for r in rows if r['mean_se'] > 1e-14]
            fourth_ratio = [r['fourth_moment']/(3*r['ideal_method_variance']**2) for r in rows if r['ideal_method_variance'] > 1e-14]
            fourth_z = [abs(r['fourth_moment']-3*r['ideal_method_variance']**2)/r['fourth_se'] for r in rows if r['fourth_se'] > 1e-14]
            moments_rows.append([phase, method, len(rows), number(max(mean_z) if mean_z else None),
                                 number(statistics.median(fourth_ratio) if fourth_ratio else None),
                                 number(max(fourth_z) if fourth_z else None)])
    coupled_rows = []
    for phase in ('conditional', 'equilibrium'):
        for method in ('A', 'B'):
            rows = [r for r in valid_variance if r['phase'] == phase and r['method'] == method]
            coupled_rows.append([phase, method, number(max((abs(r['paired_mean_error']) for r in rows if isinstance(r.get('paired_mean_error'), (int, float))), default=None)),
                                 number(max((abs(r['paired_second_error']) for r in rows if isinstance(r.get('paired_second_error'), (int, float))), default=None)),
                                 number(maximum(rows, 'coupled_path_error_max'))])
    selected = [r for r in selection if r.get('available')]
    orders = [r['observed_order'] for r in splitting if isinstance(r.get('observed_order'), (int, float)) and r['mean_operator_error'] > 1e-11]
    spatial = [r for r in structures if isinstance(r.get('spatial_error'), (int, float))]
    spatial_bad = [r for r in spatial if r['spatial_error'] > r['epsilon']]
    spatial_bounds = [dict(config=r['config'], epsilon=r['epsilon'], measured_relative_error=r['spatial_error'],
                           reference_tail_bound_rel=r['reference_tail_bound_rel'],
                           physical_relative_error_upper=(r['spatial_error']+r['reference_tail_bound_rel'])/(1-r['reference_tail_bound_rel']))
                      for r in spatial if isinstance(r.get('reference_tail_bound_rel'), (int, float)) and 0 <= r['reference_tail_bound_rel'] < 1]
    write_csv('spatial_accuracy_bounds.csv', spatial_bounds)
    ratio_rows = []
    setup_rows = []
    setup_data = read('setup_timing_raw.csv')
    for N in (2, 8, 16, 64, 128, 256):
        rows = [r for r in paired if r['N'] == N]
        ratio_rows.append([N, len(rows), number(statistics.median([r['A_median_s'] for r in rows]) * 1000) if rows else 'N/A',
                           number(statistics.median([r['B_median_s'] for r in rows]) * 1000) if rows else 'N/A',
                           number(statistics.median([r['A_over_B'] for r in rows])) if rows else 'N/A'])
        fresh = [r for r in setup_data if r['N'] == N]
        setup_rows.append([N, len(fresh), number(statistics.median([r['A_fresh_setup'] for r in fresh])*1000) if fresh else 'N/A',
                           number(statistics.median([r['B_fresh_setup'] for r in fresh])*1000) if fresh else 'N/A'])
    unavailable = [r for r in selection if not r.get('available')]
    stop_counts = [[scope, values.get('true_positive', 0), values.get('true_negative', 0),
                    values.get('false_positive', 0), values.get('false_negative', 0), values.get('no_previous_rank', 0)]
                   for scope, values in [('每个方向', stopping.get('per_direction', {})),
                                         ('全部注册方向同时', stopping.get('all_registered_directions', {}))]]
    if not stopping:
        stop_counts = [['N/A'] * 6]
    uncensored_counts = [[tol, item['comparable_rows'], item['false_positive_opportunities'],
                          item['observed_false_positive_count'], item['counts'].get('false_negative', 0)]
                         for tol, item in uncensored.get('per_direction', {}).items()] or [['N/A'] * 5]
    refinement_note = ('N/A：未保存采样后秩修正。' if not refinement else
        f"初始五注册方向均通过时，{refinement['config']}、B、alpha={refinement['only_alpha']:g} 的 noise rank={refinement['old_noise_rank']} 在额外 Gaussian MC 路径上出现最大相对差 {number(max(refinement['initial_path_errors'].values()))}，触发 1e-5 的额外 observed-path guard。这说明有限注册方向不能保证其它 Gaussian 输入。随后仅将该 case 的 noise rank 从 {refinement['old_noise_rank']} 增至 {refinement['new_noise_rank']}（完整投影维度 3），damping rank=3 保持不变，同一个 G/q 与相同每 phase 512 个 Gaussian seeds 重算。修正后该 case 最大路径相对差为 {number(max(refinement['refined_path_errors'].values()))}；最终全 MC 最大为 {number(refinement.get('final_all_MC_path_error_max'))}。初始选择、rank/action 数据、MC CSV 与该 case 的原始样本均保存在 `{refinement['pilot_path']}`，哈希见 `{refinement['SHA_manifest']}`，没有隐藏初次未通过结果。复现命令 `{refinement['reproduce']}`，明细见 `rank_refinement.csv`。其它 case 的有限样本通过仍然是经验性证据。")
    text = f'''# Frozen thermostat A/B 本地实验报告

只读基线为 PR #7 的 `76ceb517fc24f60dfdf6d9d2b4af7a2109a07e54`。本目录只研究 Candidate 1 / S1L；A 和 B 共用逐项相同的已投影离散 `Gamma_h = Gamma_h^S + Gamma_h^L`。本报告由保存的 CSV/JSON 自动生成；未运行项显示 N/A。

## 实际覆盖与结果

已保存 {len(structures)} 个冻结配置、{len(selection)} 组方法/步长选择，其中 {len(selected)} 组在声明的注册方向测试域可用，{len(unavailable)} 组不可用。accuracy 阶段记录 {len(failures)} 个配置异常。{tests_text}。基准没有运行长时间 LJ、BAOAB 或 HPC。

{table(['N', 'epsilon=1e-4 配置数', '配置族', '可用方法/alpha 组合', 'MC 配置数', 'fresh timing 配置数'], coverage)}

`alpha = dt * lambda_max / m` 测试 {protocol.get('alphas', 'N/A')}，`m=theta=1`。额外 epsilon=1e-3/1e-5 仅为 N=256 uniform seed 2 的精度覆盖；计时只用 epsilon=1e-4。MC 只在 alpha=1 的列明配置运行。N=128 的族覆盖为每族 seed 2，未运行九配置；其它未列入计划的覆盖均为 N/A。

## 三种证据必须分开解释

1. **Exact dense 代数恒等式**：在 `Range(Pi)` 独立对角化同一个离散 Gamma，令 `R=exp(-dt Gamma)`、`F=sqrt(I-exp(-2dt Gamma)) Pi`，检查 `R Pi R^T+F F^T=Pi`。全数据最大 dense FDT residual 为 {number(maximum(structures, 'dense_fdt_residual'))}；Gamma 对称残差 {number(maximum(structures, 'symmetry_residual'))}，COM 零模残差 {number(maximum(structures, 'momentum_residual'))}。这是该离散 exact OU 的代数证据。
2. **有限 Krylov 确定性 action**：damping 与 noise 分别选秩，在 {', '.join(protocol.get('rank_probes', ['slow', 'fast', 'random']))} 注册方向上与相同矩阵的 dense function action 比较。选秩误差预算通常为 B 的 `1e-5`，A 每个内部步的 `1e-5/k`，A 的理想 splitting mean/covariance 谱范数误差需不超过 `1e-3`。matrix-free 与 dense Lanczos action 最大相对差为 {number(maximum(checks, 'matrixfree_vs_dense_lanczos'))}；保存 p/z 上的 matrix-free exact action 最大误差为 {number(maximum(checks, 'exact_action_error'))}。这是有限方向和已采样输入证据，**不是对所有输入成立的算子误差界或有限秩精确 FDT 证明**。秩扫描的 step-change 是辅助诊断，选秩依据为真实 dense-reference 误差；极值本征向量常会提早终止，random 方向仍必需。
3. **有限样本统计证据**：MC 对实际 finite-Lanczos 随机步独立采样，已保存 {len(mc)} 行方向统计。每个 variance 的误差线是独立样本中心化平方的近似、逐点 95% Student-t 区间，未作多重比较校正，也不是严格协方差界。排除 COM 后，理想方法方差落入区间为 {ci_covered}/{len(valid_variance)}，区间外为 {len(outside)}；这只描述本次样本，不作为全局通过/失败标准。MC 最大 COM 改变量为 {number(maximum(mc, 'com_max'))}。四阶矩与 SE 见 `results/mc.csv`；小 N 完整经验协方差谱/Frobenius 误差见 `mc_fullcov.csv`，也包含有限样本误差。

有限 Lanczos 的基 `Q_r(z)` 随输入 z 改变，近似 `f_r(Gamma;z)` 一般不是固定线性算子；不能逐列组装一个 F 后用 `F F^T` 当作该随机算法的精确 covariance。本目录没有这样做。MC 中的 exact target 为 B/full-Gamma 的 OU；ideal-method target 为理想 A 或 B 的参考，二者在 A 的 conditional phase 可能不同。A/B 同一 sample 重设相同种子，共用物理输入与首个 long/full Gaussian；前后边噪声及后续内部子步仍为独立抽样。不同 sample 独立；本报告的单方法 SE 不因耦合被人为缩小。

下面用理想方法的方向 mean 和 variance 计算统计量。第四矩分母为 Gaussian 参考 `3*ideal_variance^2`，mean/第四矩的标准化偏差均用各自保存的样本 SE；最大值会受到多重观察与有限样本影响，不构成假设检验。条件均值的确定性误差另见 `mode_decay.csv`，不与这里的 MC SE 混合。

{table(['phase', '方法', '非 COM 方向数', '最大 mean 偏差/SE', '第四矩/理想 Gaussian 参考中位数', '最大第四矩偏差/SE'], moments_rows)}

为把有限 Krylov 误差从宽 MC 区间中分离，另对每个实际样本推进同噪声的 exact-dense 参考。B 参考为 exact full OU；A 参考为相同独立前后边噪声和 exact long OU 的理想 splitting。保存实际/理想样本对及逐路径相对误差。下面是非 COM 方向的最大绝对 paired mean/second-moment 样本差，以及全向量最大路径相对误差；second moment 围绕理想方法的条件/平衡均值中心化，不能等同严格 covariance 算子误差。各 paired 差的独立样本 SE 也保存在 `mc.csv`。

{table(['phase', '方法', '最大 |paired mean 差|', '最大 |paired second-moment 差|', '最大同噪声路径相对误差'], coupled_rows)}

### 采样后发现与保留的秩修正

{refinement_note}

### 经过实测的 stopping diagnostic 与正反向同源检验

`stopping_audit.py` 把相邻**已测试秩**的 step-change 与相同 tolerance 下的 dense true error 对照。秩增量不均匀，且扫描在首次 dense 通过时终止，不能据此预测未运行的 heuristic stopping 成本。false positive 指 step-change 已小而真实误差仍超标，false negative 则反之。下表只描述已运行记录，正式选秩始终使用 dense-reference error：

{table(['判断域', 'true positive', 'true negative', 'false positive', 'false negative', '无前一秩'], stop_counts)}

正式 rank evidence（含披露的补充 rank3 检查）中，具有前一已测试秩且真实误差仍超 tolerance 的记录为 {stop_over_tolerance}。如果该数量为零，false-positive 风险在这组 first-dense-pass 扫描中**没有被评估**，零 false positive 不能解释为 stopping 规则安全。

另做独立、不中途停止的 dense-action diagnostic sweep：六个 N 的 uniform seed 2，saved random/z，full/long 的 damping/noise，alpha=1/2，测试秩 1/2/3/4/6/8/12（截至 3N-3 并去重），附加更严格 tolerance。该实验不使用 FFT 计时，也不替换正式选秩证据。结果如下；这些相关方向/秩记录的零 false positive 仍不能构成普遍安全证书。

{table(['补充 tolerance', '有前一秩的方向记录', '真实超标的可检验记录', '观察到 false positive', 'false negative'], uncensored_counts)}

`coupling_checks.csv` 另在 N=2/8/16/64/128/256 uniform seed 2 保存共 {len(coupling)} 个方法路径检验：将 p 与所有输入 Gaussian 同时变号，最大正反向和的相对范数为 {number(maximum(coupling, 'signflip_relative'))}；与独立实现的同噪声 exact B / ideal A 路径最大相对差为 {number(maximum(coupling, 'coupled_exact_relative'))}。A 的独立参考重新做 3x3 edge eigensolve 与 dense long OU，未调用生产 pair sweeps。变号检验说明本次路径的奇对称性，不能证明有限 Krylov 映射是线性算子，也不能证明其精确 covariance。

## Combination A 的数学说明与 splitting

每个无向近场边的 K 是生产近场中所有保留周期镜像的对称 3x3 总和。逐边对相对动量执行 exact OU，半步前扫与半步后扫使用独立噪声；中间 Lanczos 只调用 long action。任一理想子步保持固定 COM 的 Maxwell covariance Pi。若前扫均值矩阵为 E，中间 long 均值为 L，则一完整子步均值 `R_A=E^T L E`，k 个子步取相应幂。独立子步噪声的 covariance 可由递推 `C_next=R_j C R_j^T+C_j` 得到，并望远镜化为 `Pi-R_A Pi R_A^T`。因此理想 A 保持 frozen Maxwell，但其条件均值/covariance 一般不同于 full-Gamma exact OU。

`splitting.csv` 中的 `ideal_invariance_residual` 使用上述代数表达式构造 covariance，仅是表达式一致性检查；独立子步 covariance 的验证由测试承担。有限 Krylov 的随机算法不自动继承该理想结论。随着内部 `dt/k` 减半，本次非数值地板区间记录的相邻 mean-error 观察阶范围为 {number(min(orders) if orders else None)} 至 {number(max(orders) if orders else None)}，不单凭对称结构宣称二阶；完整分配置、alpha 和相邻阶见 CSV。

## 空间离散与公平性

保存的 q、p、z、Gamma、GammaS、GammaL 及方向位于 `inputs/`。full/long matrix-free 与 dense action 相对误差最大分别为 {number(maximum(structures, 'mf_full_error'))} / {number(maximum(structures, 'mf_long_error'))}。空间 independent-reference 检查实际覆盖 {len(spatial)} 个配置；相对截断 direct reference 的最大谱误差 {number(maximum(spatial, 'spatial_error'))}，超过各自 epsilon 的配置数为 {len(spatial_bad)}。reference 的最大相对 tail bound 为 {number(maximum(spatial, 'reference_tail_bound_rel'))}；若记测量误差 e、相对尾界 b，则对完整周期物理矩阵的保守相对误差上界为 `(e+b)/(1-b)`，本次最大为 {number(maximum(spatial_bounds, 'physical_relative_error_upper'))}（分配置见 `spatial_accuracy_bounds.csv`）。尾界是实数算术下的解析截断界，没有声称浮点求和已获 interval certification；推导与独立积分检查见 `reference_tail_audit.json`。空间误差与热浴时间离散/Krylov 误差分开报告；A/B 的成本不会使用不同空间矩阵。

空间参数与合法小 N 网格详见 `structure.csv`，基线 N=256、epsilon=1e-4 使用 xi=1.24875、sR=sF=2.9、M=20、p=4；严格保留周期多镜像、真实 k=0 与 self cancellation，不添加 delta_h Pi、不在 Fourier multiplier 上开平方。

## 同一 fresh batch 的总成本

以下每个配置先取其重复总步耗时中位数，再在同 N 的配置间取中位数；A/B 比为逐配置 ratio 的中位数。ratio > 1 说明本地 A 更慢。图中的区间表示配置间范围，不是置信区间。只有 alpha=1 且两方法选秩均可用的配置进入此表；“同等精度”限于前述注册方向/理想 splitting 的筛选与额外验证域，未建立所有随机输入的统一误差界。

{table(['N', '配对配置数', 'A 总步 ms', 'B 总步 ms', 'A/B'], ratio_rows)}

本地分相数据表明，A 的两个逐边 OU sweep 占其主要成本；当前 `EdgeOU.sweep` 使用 Python 逐边循环。因此这里的 A/B 时间结论适用于本实现与本地环境，不能据此断言编译优化后的 pair kernel 或生产轨迹也有相同速度比。

总步 wall time 包含 Gaussian 生成、边前后扫、Lanczos 的 damping/noise、matrix-free Gamma action 与重正交；分相成本和 matvec/FFT/near 计数保留在 `timing_raw.csv` 与 `timing_summary.csv`。full/long action 为运行时实际计数，并与 Lanczos matvec_count 核对；每次 friction action 有三个正向、三个反向 FFT，near image action 仅计入 B 的 full action。未混用历史 PR #7 cached Gamma-v 计时。各方法的 damping/noise 秩独立选取；A 的内部子步数也计入成本。fresh timing 原始行数 {len(read('timing_raw.csv'))}，轮数/每轮重复数为 {timing_done.get('rounds', 'N/A')}/{timing_done.get('reps', 'N/A')}。

fresh setup 每配置另做 3 次，以下为同 N 的实际记录中位数（ms）。B 包含 table/config setup；A 另包含 image-summed 边构建、3x3 eigendecomposition 和 dt factors。用于验证的独立 dense 组装、共同 radial-kernel cache 建立与进程启动均排除；这不是全进程冷启动计时。原始记录为 `setup_timing_raw.csv`，accuracy 构建耗时另存 `setup.csv`；两者均不混入已缓存配置的单步时间。计入所测配置 setup 的 K 步成本为各方法 `fresh_setup + K * frozen_step`。

{table(['N', 'fresh setup 记录数', 'A setup ms', 'B setup ms'], setup_rows)}

环境：Python {str(env.get('python', 'N/A')).splitlines()[0]}；NumPy {env.get('numpy', 'N/A')}；SciPy {env.get('scipy', 'N/A')}；{env.get('platform', 'N/A')}。请求各 native pool 单线程；实际池见 `environment_timing.json`：`{env.get('native_thread_verification', 'N/A')}`。系统负载及活跃进程快照见 `system_load.json`，本地 wall cost 不能外推为 HPC 或生产轨迹成本。

## 图、耗时、审计与限制

五张图均来自本目录 CSV，同时输出 PNG 与 SVG：

- `figures/rank_vs_error.*`：full/long 分开、damping/noise 分开，注册方向的真实 action 相对误差。
- `figures/directional_fdt.*`：实际 finite-Krylov MC 方差及逐点 CI，相对 exact-full target 归一化，同时显示 ideal-method target；COM 在数值审计单列。
- `figures/splitting_convergence.*`：真实观察的内部步长减半误差，mean/covariance 分开。
- `figures/equal_accuracy_total_time.*`：fresh、交错、总 frozen step wall cost，及逐配置 A/B ratio。
- `figures/total_cost_phases.*`：N=16/64/256 各方法总耗时中位配置中、重复总耗时中位数对应的实际一步，展示不重叠的 pairOU/Lanczos 分相；其它项为总时间减分相之和。FFT 与重正交已经包含在 Lanczos 中，不重复相加。

阶段实际墙钟：accuracy={number(accuracy_done.get('elapsed_s'))} s；MC={number(mc_done.get('elapsed_s'))} s；timing={number(timing_done.get('elapsed_s'))} s。审计状态：{audit.get('status', 'N/A（请运行 audit.py）')}；检查细项、baseline 全部受追踪文件哈希及 evidence manifest 见 `results/audit.json`。异常列表见 `failures.json`；不可用选择见 `selection.csv`。观察域有限、有限 MC 精度与未验证的原生线程池均是明确限制；没有未运行的轨迹、HPC 结果或有限 Lanczos 精确 FDT 声明。

复现命令见 [README.md](README.md)。脚本拒绝覆盖已完成阶段；完整复现应在独立基线 clone 的相同相对路径中只复制本 PR 新脚本及说明文件，再从 source-only 目录生成新 evidence，保留现有结果。
'''
    (HERE / 'report.md').write_text(text)
    print(f'Wrote report.md from {len(structures)} systems, {len(mc)} MC rows, {len(summary)} timing groups')


if __name__ == '__main__':
    main()
