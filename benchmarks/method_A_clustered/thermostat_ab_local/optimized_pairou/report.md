# Combination A 顺序 Pair OU 编译优化：本地报告

本实验以 PR #8 head `bce73847f2968c27e1df5e2ff4be57393c4d2ddd` 为只读基线。只改变 A 的 near-edge sweep 实现，全部 q/GammaS/GammaL/Gamma、Gaussian 输入、r_d/r_n、dt 与 splitting substeps 均与对应数值对照相同。B 与 Grid PSD 源码未改。下述成本只取本目录 fresh three-way batch，不与旧 PR #8 的 79.8/11.6 ms 混算速度比。

主要结果：N256、epsilon=1e-4 的配置中位总步从新测 Python A 的 88.4 ms 降到 compiled A 的 10.59 ms；同批 B 为 13.35 ms。paired Python/compiled 整步加速 8.213，compiled/B=0.7996，95% block-bootstrap CI [0.7875, 0.8406]。优化 A 在这九个配置和本次运行中快于 B，paired ratio 的近似区间完全低于 1。

## 先定位瓶颈，再保持算法实现编译

新跑的预优化诊断 profile 在 N256 uniform seed2、4948 条有序聚合边上测量 7 次未插桩整步，near 双 sweep 占总时间中位比例 85.42%，总步中位数 102.1 ms。另用 cProfile 记录 5 步，原始 `.prof`、函数表和 load 见 logs/results。嵌套 cumulative profiler 时间相互重叠，不能相加；该阶段只确认优化对象，不是正式速度比的分母。

完整 3x3 image-summed PSD 边的相对 OU 可写作 `dp=D r+S eta`，`D=0.5 U expm1(-2 tau lambda/m) U^T`，`S=0.5 U sqrt(2m theta[-expm1(-4 tau lambda/m)]) U^T`；再顺序执行 `p_i+=dp; p_j-=dp`。这与原式 `0.5(r_plus-r)` 同义，避免 `exp()-I` 消减。D/S 在 tau、m、theta 已知时缓存，循环内只做 3x3 action 与顺序粒子更新。C 循环严格保留 forward/reverse 顺序与原边索引，不并行或重排共享粒子边、不假设 rank-one、不复用本应独立的 Gaussian。

graph/边张量/eigensystem 不可变，位置或图改变要求显式 rebuild；每个实例的 coefficient cache 区分 tau/m/theta。参数更新、同形状图更新、零/小 tau、空边、rank-deficient/full-rank、输入不变性和 Python fallback 均由新增测试覆盖。

另以实际 N16 uniform seed2 的生产图做 30 种 tau/温度/质量组合、双方向，共 62 行 cache 验证；先 miss 后 repeat hit，并实际扰动 q、重建 GridPSDGamma/multi-image edges、使用新 token 清空并重建 coefficients，同时验证旧图/旧缓存不变。原始 fixture/路径与参数 SHA 见 `cache_validation.*`；最大路径差 1.402e-15。零 tau 的 D=S=0 在不同参数下具有相同系数哈希是合法情形，不能误判为缓存未更新。

## 实际编译环境与启动成本

实际后端为顺序 C/ctypes。Numba import 检查为 `{'available': False, 'error': "ImportError('Numba needs NumPy 2.3 or less. Got NumPy 2.4.')"}`；package versions 为 `{'numba': '0.62.1', 'llvmlite': '0.45.1'}`。这是实际环境结果，未为 Numba 改装 NumPy。compiler 为 `/usr/bin/clang`，flags 为 `['-O3', '-std=c11', '-fPIC', '-fno-fast-math', '-ffp-contract=off', '-dynamiclib']`，关闭 fast-math/FP contraction，无 OpenMP；完整 compiler version/command、源与库 SHA 见 `compilation.json`。原 Python EdgeOU 可选择为 reference/fallback，正式 optimized 样本要求 active_backend=compiled。

真实 cold compile subprocess=0.4065 s；首次 CDLL load=0.5539 s；整个 cold entry=1.128 s。独立新进程使用现有库的 cache load=0.001738 s，整个该 Python 进程（含 imports）=0.4491 s。这些不是 steady-state 步成本。

另在空二进制 cache 中启动全新 Python 进程，真实首次进程启动总 wall 为 1.191 s（`cold_process_start.json`）；该次包含 imports、compiler lookup、编译和 load，排除物理 table/config setup。它与前面的 backend-entry/缓存进程时间分别记录，未从其它阶段时间相加伪造首次启动。

fresh configuration setup 每配置 3 次，单位 ms：

| N | setup记录 | A Python setup | A compiled setup | B setup | compiled coeff setup |
| --- | --- | --- | --- | --- | --- |
| 16 | 27 | 3.088 | 4.176 | 2.638 | 0.1276 |
| 64 | 27 | 12.28 | 14.11 | 9.398 | 1.112 |
| 128 | 9 | 19.62 | 24.84 | 12.56 | 2.608 |
| 256 | 27 | 37.02 | 45.01 | 22 | 5.308 |

setup 包含真实 table/config/image-summed edges、constructor/eigh/immutable copies 和 coefficients；compiled constructor 的 warm-library检查/load 也计入。constructor 已包含 eigh，不能再把其诊断 eigh 时间重复加到总 setup。共同 radial-kernel cache、进程启动和验证用 dense 组装排除；不声称全进程冷启动。改变 q 后需要重新构图/eigh/coeff，不能把旧 frozen setup 当作免费。steady step 只使用预热后的已缓存配置。

若每次 q 更新都需要 fresh setup，必须核算 `setup + K*step`。下表将**同一配置**的 setup 中位数与 step 中位数相加，再在配置间取中位数；这是分相成本核算，未声称另做了连续端到端计时。K 为同一 frozen 图需要复用多少步后 compiled A 才便宜于 B 的点估计，未计入一次性 cold compile/启动，且不是置信保证：

| N | A setup+1步 ms | B setup+1步 ms | 配对成本比 A/B | 复用步数K中位数 [范围] | 可摊销配置数 |
| --- | --- | --- | --- | --- | --- |
| 16 | 7.956 | 6.5 | 1.161 | 4.5 [3,10] | 8/9 |
| 64 | 18.85 | 14.66 | 1.304 | 9 [5,20] | 9/9 |
| 128 | 33.91 | 22.34 | 1.486 | 9 [8,34] | 3/3 |
| 256 | 55.56 | 36.93 | 1.554 | 9 [6,16] | 9/9 |

因此已缓存 frozen step 的优势不能直接解释为位置每步变化的生产轨迹优势。`setup_step_costs.csv` 保留每配置核算；若 step 中位数没有优势，K 显示 N/A。

## 实际精度与不变量覆盖

| N | epsilon=1e-4配置数 | 族 | MC配置数 |
| --- | --- | --- | --- |
| 2 | 1 | uniform | 1 |
| 8 | 1 | uniform | 1 |
| 16 | 9 | moderate, strong, uniform | 3 |
| 64 | 9 | moderate, strong, uniform | 1 |
| 128 | 3 | moderate, strong, uniform | 1 |
| 256 | 9 | moderate, strong, uniform | 1 |

实际 fresh accuracy 为 34 个配置、408 行三方法/alpha 对照，alpha=0.1/0.5/1/2。N16/64/256 保留各九配置，N128 每族 seed2；另有 N256 uniform seed2 的 epsilon=1e-3/1e-5。所有 saved q/G/GS/GL/p/z 与 PR8 bitwise identity，原始边序和完整 image-summed K 一致；沿用 PR8 最终秩，含 N2/B 已修正的 noise rank3，没有重新调低误差预算。

compiled/Python A 全步最大相对路径差 3.86e-15，near forward/reverse 最大差 2.157e-15；COM 最大改变量 7.255e-14。exact dense FDT 代数 residual 最大 1.069e-12。相同 ordered ideal-A 均值/covariance 与 full exact B 的 splitting 误差和 dt/2 收敛重新计算于 `splitting.csv`，优化没有删除这一动力学误差。

`accuracy.csv`/`mode_decay.csv` 的 mean_action_error 是关闭全部噪声后的确定性 damping action，**不是**实际 finite-Lanczos A 的随机条件均值。有限 Krylov damping 映射依赖输入，near 噪声经非线性中间 action 传播可能产生 Jensen 效应；实际条件 mean 只能由 MC sample mean/paired mean 及 SE 检查。只有理想线性 Gaussian A 的 R_A/C_A 具有上述精确均值/covariance 解释。

理想 A 的各 exact 子步保持固定 COM 的 Maxwell covariance Pi，组合 covariance 为 `Pi-R_A Pi R_A^T`；该代数表达式的 residual 只是构造一致性，不是 finite-Krylov 精确 FDT。有限 Lanczos 基随输入 Gaussian 改变，不是固定线性 F，不能逐列组装后把 F F^T 称作算法的精确 covariance。

fresh MC 保存 192 行方向统计，conditional/equilibrium、slow/fast/random/COM 三方法共用同源独立 draws。原始样本、理想同噪声参考、种子与 Gaussian SHA/recipe 均保存。编译/原 Python 最大相对样本差 2.389e-15；finite-Krylov 对理想同噪声路径最大相对差 4.229e-07，COM 最大 7.22e-14。非 COM 理想方向方差落入逐点近似 95% CI 为 138/144；该计数只描述样本，未作多重修正，不能作为严格 FDT proof。mean/variance/fourth moments、SE/CI、paired mean/second-moment discrepancy 见 MC CSV；二阶 paired 差围绕已知理想均值，包含均值偏差平方，不能当作 covariance 算子界。

这些输出全部重新执行，A 两实现使用同源噪声；本次 global seed 与旧 PR8 仅相差 1，逐样本种子集合可能大量重叠，因此不声称相对于旧 PR8 是统计独立的复现实验，也不合并两次样本数。当前 SE/CI 只使用本次输出，在本次 run 内不同 sample 独立。

A-python/A-compiled 的同源近似相同输出也不构成额外独立统计重复；192 行是方法/phase/方向汇总行数，不是 192 次独立 FDT 实验。

| 测试日志 | 测试数 | 结果 |
| --- | --- | --- |
| baseline_tests.log | 26 | OK |
| compiled_tests.log | 16 | OK |
| test_validation.log | 3 | OK |

初次新增测试日志 `compiled_tests_initial.log` 仍保留：一个随机输入的 graph-change 测试把输出差必须大于 0.01 当作任意阈值，实际差为 0.0008888。最终测试改为直接激发被改变的本征方向并关闭该项噪声，排除漂移/噪声偶然抵消，随后 16 项通过；这次调整针对测试 fixture，没有改变编译 kernel 或精度预算。

## fresh 三方完整总成本

正式 batch 共 34 配置、5 rounds × 7 paired reps ×3方法，3570 行。随机顺序交错所有方法；每个 sample 各方法重设相同种子，两个 A 使用相同 z/前后边 Gaussian，B 仅生成需要的 z；全部 RNG 构造/实际所需 Gaussian、近场、FFT、Krylov、重正交计入整步。hash/序列化及 paired path comparison 在计时之外。每配置先取总时间中位数，再对同 N 的配置取中位数（ms）：

| N | 配置数 | A Python | A compiled | B full |
| --- | --- | --- | --- | --- |
| 2 | 1 | 1.295 | 1.302 | 2.225 |
| 8 | 1 | 2.775 | 2.347 | 2.535 |
| 16 | 9 | 4.707 | 2.995 | 2.987 |
| 64 | 9 | 20.8 | 4.668 | 5.212 |
| 128 | 3 | 55.62 | 9.072 | 10.77 |
| 256 | 9 | 88.4 | 10.59 | 13.35 |

下表为 paired ratio 与近似 bootstrap 95% CI。每次 repetition 先做同 sample 成本比，再在 config/round 中取中位数，再在 round、配置间取中位数。CI 用4000次分层 bootstrap：配置作为 block 重采样，每个抽中配置再整体重采样其 round-ratio blocks；没有把相邻 reps 假定为独立硬件状态。配置 min/max、round min/max 另存，不能把变化范围当作 CI。N2/8 单配置的 CI 只反映该配置的 round 变化。

各方法配置中位时间与配对比率的取中位顺序不同，不能把上表两个时间直接相除来替代 paired ratio。

| N | near Python/compiled | total Python/compiled | total compiled/B |
| --- | --- | --- | --- |
| 2 | 0.6262 [0.618, 0.7169] | 0.9948 [0.8254, 1.055] | 0.67 [0.5672, 0.7243] |
| 8 | 7.834 [6.028, 8.213] | 1.163 [1.132, 1.185] | 0.9348 [0.9168, 0.9491] |
| 16 | 29.67 [26.7, 31.28] | 1.605 [1.562, 1.621] | 0.9057 [0.8929, 0.9373] |
| 64 | 172.1 [164.3, 174.9] | 4.325 [4.261, 4.525] | 0.888 [0.8777, 0.8987] |
| 128 | 241.4 [223, 256.1] | 6.304 [5.281, 6.584] | 0.8316 [0.7664, 0.8837] |
| 256 | 293 [283.6, 306.7] | 8.213 [7.894, 8.413] | 0.7996 [0.7875, 0.8406] |

Python/compiled 大于1表示编译优化更快；compiled/B 小于1表示优化 A 在本地测试域比 B 更快。CI 条件于这些 frozen 配置和本次运行，不保证其它硬件、未来状态或生产轨迹。所有 ratio 来自新 batch，附加 epsilon 不混入主表：

| N256 epsilon | A Python ms | A compiled ms | B full ms |
| --- | --- | --- | --- |
| 0.001 | 134 | 5.486 | 9.75 |
| 1e-05 | 194.4 | 18.08 | 25.69 |

`t_reorth`/`t_ritz_eigh`/FFT 均已包含在 Lanczos 阶段，`t_full_B` 已包含整步，禁止再与分相重复相加。阶段图只堆叠 RNG、near sweeps、long/full damping/noise 与实测总时间的剩余项。

## 图、限制与审计

三张 PNG+SVG：`near_sweep_speedup`、`three_way_total_time`、`three_way_phases`。前两张展示实际 near/完整步骤改进及区间，分相图解释剩余成本；绝对时间误差条为配置范围，ratio 误差条为上述 bootstrap CI。

环境 Python 3.11.5 (v3.11.5:cce6ba91b3, Aug 24 2023, 10:50:31) [Clang 13.0.0 (clang-1300.0.29.30)]，NumPy 2.4.6，SciPy 1.17.1，macOS-27.0.1-arm64-arm-64bit。请求单线程，实际 native pool 可验证性为 `threadpoolctl sees no Accelerate pool; native count unverified`；系统负载见 `system_load.json`。CPU/ctypes/Python 驱动与有限本地覆盖构成本结论边界。

accuracy耗时 235.7 s，MC 184 s，timing 96.36 s。审计状态 PASS；基线全部 tracked SHA、源/库/输入/Gaussian/cache/旧秩关联、原始样本统计与 evidence manifest 见 `audit.json`。

本次没有 HPC 或长轨迹结果。是否需要 HPC 取决于更大 N、并行 FFT 与轨迹目标；单凭 frozen 本地速度比不足以决定。复现见 [README.md](README.md)，使用 source-only 新目录保留所有现有证据。

原 near 双 sweep 的主要瓶颈可以通过移除 Python per-edge 循环解决；优化后 N256 A 的 long damping/noise 已占整步中位比例 93.75%，剩余 FFT/Krylov、RNG 与驱动成本仍存在，所以不能把整个方法的成本都归结为 Python overhead。本次 frozen 验收已在本地完成，不需要 HPC 才能得出上述结论。若下一步目标是更大 N 或生产轨迹吞吐，则 HPC benchmark 有价值，且必须重新测并行 FFT、配置更新及积分器整体成本；本实验不预设其速度结果。
