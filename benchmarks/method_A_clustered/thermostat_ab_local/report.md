# Frozen thermostat A/B 本地实验报告

只读基线为 PR #7 的 `76ceb517fc24f60dfdf6d9d2b4af7a2109a07e54`。本目录只研究 Candidate 1 / S1L；A 和 B 共用逐项相同的已投影离散 `Gamma_h = Gamma_h^S + Gamma_h^L`。本报告由保存的 CSV/JSON 自动生成；未运行项显示 N/A。

## 实际覆盖与结果

已保存 34 个冻结配置、272 组方法/步长选择，其中 272 组在声明的注册方向测试域可用，0 组不可用。accuracy 阶段记录 0 个配置异常。26 项单元/空间测试通过，耗时 0.332 s（见 logs/tests.log）。基准没有运行长时间 LJ、BAOAB 或 HPC。

| N | epsilon=1e-4 配置数 | 配置族 | 可用方法/alpha 组合 | MC 配置数 | fresh timing 配置数 |
| --- | --- | --- | --- | --- | --- |
| 2 | 1 | uniform | 8/8 | 1 | 1 |
| 8 | 1 | uniform | 8/8 | 1 | 1 |
| 16 | 9 | moderate, strong, uniform | 72/72 | 3 | 9 |
| 64 | 9 | moderate, strong, uniform | 72/72 | 1 | 9 |
| 128 | 3 | moderate, strong, uniform | 24/24 | 1 | 3 |
| 256 | 9 | moderate, strong, uniform | 72/72 | 1 | 9 |

`alpha = dt * lambda_max / m` 测试 [0.1, 0.5, 1.0, 2.0]，`m=theta=1`。额外 epsilon=1e-3/1e-5 仅为 N=256 uniform seed 2 的精度覆盖；计时只用 epsilon=1e-4。MC 只在 alpha=1 的列明配置运行。N=128 的族覆盖为每族 seed 2，未运行九配置；其它未列入计划的覆盖均为 N/A。

## 三种证据必须分开解释

1. **Exact dense 代数恒等式**：在 `Range(Pi)` 独立对角化同一个离散 Gamma，令 `R=exp(-dt Gamma)`、`F=sqrt(I-exp(-2dt Gamma)) Pi`，检查 `R Pi R^T+F F^T=Pi`。全数据最大 dense FDT residual 为 1.07e-12；Gamma 对称残差 8.59e-18，COM 零模残差 4.82e-16。这是该离散 exact OU 的代数证据。
2. **有限 Krylov 确定性 action**：damping 与 noise 分别选秩，在 slow, fast, random, input_p, input_z 注册方向上与相同矩阵的 dense function action 比较。选秩误差预算通常为 B 的 `1e-5`，A 每个内部步的 `1e-5/k`，A 的理想 splitting mean/covariance 谱范数误差需不超过 `1e-3`。matrix-free 与 dense Lanczos action 最大相对差为 1.81e-15；保存 p/z 上的 matrix-free exact action 最大误差为 9.53e-06。这是有限方向和已采样输入证据，**不是对所有输入成立的算子误差界或有限秩精确 FDT 证明**。秩扫描的 step-change 是辅助诊断，选秩依据为真实 dense-reference 误差；极值本征向量常会提早终止，random 方向仍必需。
3. **有限样本统计证据**：MC 对实际 finite-Lanczos 随机步独立采样，已保存 128 行方向统计。每个 variance 的误差线是独立样本中心化平方的近似、逐点 95% Student-t 区间，未作多重比较校正，也不是严格协方差界。排除 COM 后，理想方法方差落入区间为 91/96，区间外为 5；这只描述本次样本，不作为全局通过/失败标准。MC 最大 COM 改变量为 7.22e-14。四阶矩与 SE 见 `results/mc.csv`；小 N 完整经验协方差谱/Frobenius 误差见 `mc_fullcov.csv`，也包含有限样本误差。

有限 Lanczos 的基 `Q_r(z)` 随输入 z 改变，近似 `f_r(Gamma;z)` 一般不是固定线性算子；不能逐列组装一个 F 后用 `F F^T` 当作该随机算法的精确 covariance。本目录没有这样做。MC 中的 exact target 为 B/full-Gamma 的 OU；ideal-method target 为理想 A 或 B 的参考，二者在 A 的 conditional phase 可能不同。A/B 同一 sample 重设相同种子，共用物理输入与首个 long/full Gaussian；前后边噪声及后续内部子步仍为独立抽样。不同 sample 独立；本报告的单方法 SE 不因耦合被人为缩小。

下面用理想方法的方向 mean 和 variance 计算统计量。第四矩分母为 Gaussian 参考 `3*ideal_variance^2`，mean/第四矩的标准化偏差均用各自保存的样本 SE；最大值会受到多重观察与有限样本影响，不构成假设检验。条件均值的确定性误差另见 `mode_decay.csv`，不与这里的 MC SE 混合。

| phase | 方法 | 非 COM 方向数 | 最大 mean 偏差/SE | 第四矩/理想 Gaussian 参考中位数 | 最大第四矩偏差/SE |
| --- | --- | --- | --- | --- | --- |
| conditional | A | 24 | 2.14 | 0.969 | 3.73 |
| conditional | B | 24 | 2.08 | 0.916 | 3.89 |
| equilibrium | A | 24 | 2.77 | 0.96 | 3.41 |
| equilibrium | B | 24 | 3.07 | 0.96 | 3.59 |

为把有限 Krylov 误差从宽 MC 区间中分离，另对每个实际样本推进同噪声的 exact-dense 参考。B 参考为 exact full OU；A 参考为相同独立前后边噪声和 exact long OU 的理想 splitting。保存实际/理想样本对及逐路径相对误差。下面是非 COM 方向的最大绝对 paired mean/second-moment 样本差，以及全向量最大路径相对误差；second moment 围绕理想方法的条件/平衡均值中心化，不能等同严格 covariance 算子误差。各 paired 差的独立样本 SE 也保存在 `mc.csv`。

| phase | 方法 | 最大 \|paired mean 差\| | 最大 \|paired second-moment 差\| | 最大同噪声路径相对误差 |
| --- | --- | --- | --- | --- |
| conditional | A | 2.25e-07 | 2.93e-06 | 2.9e-07 |
| conditional | B | 2.62e-07 | 3.75e-06 | 3.52e-07 |
| equilibrium | A | 2.09e-07 | 2.9e-06 | 4.23e-07 |
| equilibrium | B | 2.54e-07 | 3.71e-06 | 3.39e-07 |

### 采样后发现与保留的秩修正

初始五注册方向均通过时，N2_uniform_s2_eps0.0001、B、alpha=1 的 noise rank=2 在额外 Gaussian MC 路径上出现最大相对差 1.96e-05，触发 1e-5 的额外 observed-path guard。这说明有限注册方向不能保证其它 Gaussian 输入。随后仅将该 case 的 noise rank 从 2 增至 3（完整投影维度 3），damping rank=3 保持不变，同一个 G/q 与相同每 phase 512 个 Gaussian seeds 重算。修正后该 case 最大路径相对差为 2.56e-15；最终全 MC 最大为 4.23e-07。初始选择、rank/action 数据、MC CSV 与该 case 的原始样本均保存在 `pilot_rank_refinement/`，哈希见 `pilot_rank_refinement/initial_evidence_manifest.json`，没有隐藏初次未通过结果。复现命令 `python3 -B refine_small_case.py`，明细见 `rank_refinement.csv`。其它 case 的有限样本通过仍然是经验性证据。

### 经过实测的 stopping diagnostic 与正反向同源检验

`stopping_audit.py` 把相邻**已测试秩**的 step-change 与相同 tolerance 下的 dense true error 对照。秩增量不均匀，且扫描在首次 dense 通过时终止，不能据此预测未运行的 heuristic stopping 成本。false positive 指 step-change 已小而真实误差仍超标，false negative 则反之。下表只描述已运行记录，正式选秩始终使用 dense-reference error：

| 判断域 | true positive | true negative | false positive | false negative | 无前一秩 |
| --- | --- | --- | --- | --- | --- |
| 每个方向 | 640 | 0 | 0 | 1685 | 2720 |
| 全部注册方向同时 | 1 | 0 | 0 | 464 | 544 |

正式 rank evidence（含披露的补充 rank3 检查）中，具有前一已测试秩且真实误差仍超 tolerance 的记录为 0。如果该数量为零，false-positive 风险在这组 first-dense-pass 扫描中**没有被评估**，零 false positive 不能解释为 stopping 规则安全。

另做独立、不中途停止的 dense-action diagnostic sweep：六个 N 的 uniform seed 2，saved random/z，full/long 的 damping/noise，alpha=1/2，测试秩 1/2/3/4/6/8/12（截至 3N-3 并去重），附加更严格 tolerance。该实验不使用 FFT 计时，也不替换正式选秩证据。结果如下；这些相关方向/秩记录的零 false positive 仍不能构成普遍安全证书。

| 补充 tolerance | 有前一秩的方向记录 | 真实超标的可检验记录 | 观察到 false positive | false negative |
| --- | --- | --- | --- | --- |
| 1e-06 | 512 | 148 | 0 | 96 |
| 1e-08 | 512 | 237 | 0 | 96 |
| 1e-10 | 512 | 278 | 0 | 96 |

`coupling_checks.csv` 另在 N=2/8/16/64/128/256 uniform seed 2 保存共 12 个方法路径检验：将 p 与所有输入 Gaussian 同时变号，最大正反向和的相对范数为 0；与独立实现的同噪声 exact B / ideal A 路径最大相对差为 2.59e-07。A 的独立参考重新做 3x3 edge eigensolve 与 dense long OU，未调用生产 pair sweeps。变号检验说明本次路径的奇对称性，不能证明有限 Krylov 映射是线性算子，也不能证明其精确 covariance。

## Combination A 的数学说明与 splitting

每个无向近场边的 K 是生产近场中所有保留周期镜像的对称 3x3 总和。逐边对相对动量执行 exact OU，半步前扫与半步后扫使用独立噪声；中间 Lanczos 只调用 long action。任一理想子步保持固定 COM 的 Maxwell covariance Pi。若前扫均值矩阵为 E，中间 long 均值为 L，则一完整子步均值 `R_A=E^T L E`，k 个子步取相应幂。独立子步噪声的 covariance 可由递推 `C_next=R_j C R_j^T+C_j` 得到，并望远镜化为 `Pi-R_A Pi R_A^T`。因此理想 A 保持 frozen Maxwell，但其条件均值/covariance 一般不同于 full-Gamma exact OU。

`splitting.csv` 中的 `ideal_invariance_residual` 使用上述代数表达式构造 covariance，仅是表达式一致性检查；独立子步 covariance 的验证由测试承担。有限 Krylov 的随机算法不自动继承该理想结论。随着内部 `dt/k` 减半，本次非数值地板区间记录的相邻 mean-error 观察阶范围为 2 至 2，不单凭对称结构宣称二阶；完整分配置、alpha 和相邻阶见 CSV。

## 空间离散与公平性

保存的 q、p、z、Gamma、GammaS、GammaL 及方向位于 `inputs/`。full/long matrix-free 与 dense action 相对误差最大分别为 6.57e-16 / 6.85e-16。空间 independent-reference 检查实际覆盖 34 个配置；相对截断 direct reference 的最大谱误差 0.000791，超过各自 epsilon 的配置数为 0。reference 的最大相对 tail bound 为 1.92e-09；若记测量误差 e、相对尾界 b，则对完整周期物理矩阵的保守相对误差上界为 `(e+b)/(1-b)`，本次最大为 0.000791（分配置见 `spatial_accuracy_bounds.csv`）。尾界是实数算术下的解析截断界，没有声称浮点求和已获 interval certification；推导与独立积分检查见 `reference_tail_audit.json`。空间误差与热浴时间离散/Krylov 误差分开报告；A/B 的成本不会使用不同空间矩阵。

空间参数与合法小 N 网格详见 `structure.csv`，基线 N=256、epsilon=1e-4 使用 xi=1.24875、sR=sF=2.9、M=20、p=4；严格保留周期多镜像、真实 k=0 与 self cancellation，不添加 delta_h Pi、不在 Fourier multiplier 上开平方。

## 同一 fresh batch 的总成本

以下每个配置先取其重复总步耗时中位数，再在同 N 的配置间取中位数；A/B 比为逐配置 ratio 的中位数。ratio > 1 说明本地 A 更慢。图中的区间表示配置间范围，不是置信区间。只有 alpha=1 且两方法选秩均可用的配置进入此表；“同等精度”限于前述注册方向/理想 splitting 的筛选与额外验证域，未建立所有随机输入的统一误差界。

| N | 配对配置数 | A 总步 ms | B 总步 ms | A/B |
| --- | --- | --- | --- | --- |
| 2 | 1 | 0.984 | 1.45 | 0.68 |
| 8 | 1 | 2.78 | 2.61 | 1.06 |
| 16 | 9 | 4.09 | 2.71 | 1.49 |
| 64 | 9 | 21 | 5.53 | 3.79 |
| 128 | 3 | 40.6 | 7.24 | 5.57 |
| 256 | 9 | 79.8 | 11.6 | 6.94 |

本地分相数据表明，A 的两个逐边 OU sweep 占其主要成本；当前 `EdgeOU.sweep` 使用 Python 逐边循环。因此这里的 A/B 时间结论适用于本实现与本地环境，不能据此断言编译优化后的 pair kernel 或生产轨迹也有相同速度比。

总步 wall time 包含 Gaussian 生成、边前后扫、Lanczos 的 damping/noise、matrix-free Gamma action 与重正交；分相成本和 matvec/FFT/near 计数保留在 `timing_raw.csv` 与 `timing_summary.csv`。full/long action 为运行时实际计数，并与 Lanczos matvec_count 核对；每次 friction action 有三个正向、三个反向 FFT，near image action 仅计入 B 的 full action。未混用历史 PR #7 cached Gamma-v 计时。各方法的 damping/noise 秩独立选取；A 的内部子步数也计入成本。fresh timing 原始行数 960，轮数/每轮重复数为 3/5。

fresh setup 每配置另做 3 次，以下为同 N 的实际记录中位数（ms）。B 包含 table/config setup；A 另包含 image-summed 边构建、3x3 eigendecomposition 和 dt factors。用于验证的独立 dense 组装、共同 radial-kernel cache 建立与进程启动均排除；这不是全进程冷启动计时。原始记录为 `setup_timing_raw.csv`，accuracy 构建耗时另存 `setup.csv`；两者均不混入已缓存配置的单步时间。计入所测配置 setup 的 K 步成本为各方法 `fresh_setup + K * frozen_step`。

| N | fresh setup 记录数 | A setup ms | B setup ms |
| --- | --- | --- | --- |
| 2 | 3 | 1.26 | 1.16 |
| 8 | 3 | 2.56 | 2.38 |
| 16 | 27 | 2.85 | 2.4 |
| 64 | 27 | 11.8 | 8.71 |
| 128 | 9 | 17.5 | 11.2 |
| 256 | 27 | 34 | 20.2 |

环境：Python 3.11.5 (v3.11.5:cce6ba91b3, Aug 24 2023, 10:50:31) [Clang 13.0.0 (clang-1300.0.29.30)]；NumPy 2.4.6；SciPy 1.17.1；macOS-27.0.1-arm64-arm-64bit。请求各 native pool 单线程；实际池见 `environment_timing.json`：`threadpoolctl sees no Accelerate pool; native count unverified`。系统负载及活跃进程快照见 `system_load.json`，本地 wall cost 不能外推为 HPC 或生产轨迹成本。

## 图、耗时、审计与限制

五张图均来自本目录 CSV，同时输出 PNG 与 SVG：

- `figures/rank_vs_error.*`：full/long 分开、damping/noise 分开，注册方向的真实 action 相对误差。
- `figures/directional_fdt.*`：实际 finite-Krylov MC 方差及逐点 CI，相对 exact-full target 归一化，同时显示 ideal-method target；COM 在数值审计单列。
- `figures/splitting_convergence.*`：真实观察的内部步长减半误差，mean/covariance 分开。
- `figures/equal_accuracy_total_time.*`：fresh、交错、总 frozen step wall cost，及逐配置 A/B ratio。
- `figures/total_cost_phases.*`：N=16/64/256 各方法总耗时中位配置中、重复总耗时中位数对应的实际一步，展示不重叠的 pairOU/Lanczos 分相；其它项为总时间减分相之和。FFT 与重正交已经包含在 Lanczos 中，不重复相加。

阶段实际墙钟：accuracy=741 s；MC=116 s；timing=32.5 s。审计状态：PASS；检查细项、baseline 全部受追踪文件哈希及 evidence manifest 见 `results/audit.json`。异常列表见 `failures.json`；不可用选择见 `selection.csv`。观察域有限、有限 MC 精度与未验证的原生线程池均是明确限制；没有未运行的轨迹、HPC 结果或有限 Lanczos 精确 FDT 声明。

复现命令见 [README.md](README.md)。脚本拒绝覆盖已完成阶段；完整复现应在独立基线 clone 的相同相对路径中只复制本 PR 新脚本及说明文件，再从 source-only 目录生成新 evidence，保留现有结果。
