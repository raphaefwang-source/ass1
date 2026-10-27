# 顺序 Pair OU 编译优化

只读基线为 PR #8 head `bce73847f2968c27e1df5e2ff4be57393c4d2ddd`。本目录只优化 Combination A 的 near-edge OU 顺序 sweep，原 Python sweep、B full-Gamma Lanczos 与 Grid PSD 物理图均保留。完整要求见 [requirements_github.md](requirements_github.md)。不运行 HPC、LJ、BAOAB 或 C/D/E。

对于一条完整 3x3 PSD 边，原相对动量 OU 更新可写成 `dp=D r+S eta`，其中 `D=0.5 U expm1(-2 tau lambda/m) U^T`、`S=0.5 U sqrt(2m theta[-expm1(-4 tau lambda/m)]) U^T`。然后严格顺序执行 `p_i+=dp; p_j-=dp`。这只是缓存矩阵系数和编译循环，没有交换共享粒子边的顺序，也没有把满秩 image-summed tensor 近似为 rank-one。

`compiled_edge.py` 在不可变 graph/eigensystem 上缓存 `(tau,mass,theta)` 对应的 D/S；配置或边变化需要显式 rebuild。`edge_sweep.c` 为无 OpenMP 的顺序 C kernel，编译关闭 fast-math 与浮点乘加收缩。原 `EdgeOU.sweep` 保留为 numerical reference/fallback；实际选择的 backend、编译失败原因、NumPy/SciPy/Numba 状态和 compiler/flags 均记录到新 evidence，不能把 fallback 当作已编译性能结果。

## 可复现原则

在独立的新 clone 检出上面的基线 SHA，在相同相对路径 `benchmarks/method_A_clustered/thermostat_ab_local/optimized_pairou/` 新建 source-only 目录，只复制本 PR 的新 `*.py`、`*.c`、本 README、`requirements_github.md`、`source_snapshot.json`。不要复制本次生成的 results、inputs、logs、figures、`build` 或 `.build`。已有完成标记保护真实 evidence。所有新增结果都写入本目录；基线输入、最终选秩、原图及其旧结果保持只读。

本地依赖 Python 3、NumPy、SciPy、Matplotlib、threadpoolctl 和可用 C compiler（默认查找 clang）。当前机器是否支持 Numba 由实际 import 检查决定；编译替代路径不是假定环境。计时前先完成 profiler、单元/回归测试和精度/MC，计时期间不并行执行审计或其它数值工作。cold compile、fresh-process cache load、table/config/edge/eigh/coeff setup、steady frozen step 分开记录。

在 source-only 新目录执行（先从仓库根目录切入）：

```bash
cd benchmarks/method_A_clustered/thermostat_ab_local/optimized_pairou
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PYTHONPATH=.. python3 -B -m unittest -v test_ou_core test_spatial test_experiment > logs/baseline_tests.log 2>&1
python3 -B -m unittest -v test_compiled_edge > logs/compiled_tests.log 2>&1
python3 -B -m unittest -v test_validation > logs/test_validation.log 2>&1
python3 -B profile_baseline.py > logs/profile_baseline.log 2>&1
python3 -B compile_probe.py > logs/compile.log 2>&1
python3 -B cold_start_probe.py > logs/cold_start.log 2>&1
python3 -B validate.py accuracy --mc-small 512 --mc-medium 128 > logs/validate_accuracy.log 2>&1
python3 -B cache_validation.py > logs/cache_validation.log 2>&1
python3 -B validate.py mc --mc-small 512 --mc-medium 128 > logs/validate_mc.log 2>&1
python3 -B timing.py --rounds 5 --reps 7 --setup-reps 3 > logs/timing.log 2>&1
python3 -B report_generator.py > logs/report.log 2>&1
python3 -B plot_results.py > logs/plots.log 2>&1
python3 -B audit.py
python3 -B report_generator.py > logs/report.log 2>&1
python3 -B audit.py
```

最后一次 audit 必须晚于图、报告和日志的最后写入；直接打印 audit 输出，不重定向至本目录内正在变化的 audit.log。编译后的 build/`.build` 是本地 cache，不需要复制到新机器；实际二进制 SHA 在编译 provenance 中记录，审计本地核对，但不把可再生二进制纳入 source/evidence manifest。原始 cold/profile/formal wall cost 会受机器状态影响，不能要求跨机器重现相同耗时。

## 证据解释

Exact dense OU 的 FDT 恒等式、理想 A 的 frozen Maxwell 保持、有限 Lanczos 的 action/采样误差，以及相对 full-Gamma exact OU 的 splitting 动力学误差分开报告。有限 Krylov 噪声 action 随 Gaussian 输入改变，不能逐列拼成一个固定 F 后自称获得精确 covariance。MC CI 是逐点近似样本区间，没有多重比较修正。

所有三方速度比只用本目录 fresh interleaved batch，包含同声明范围的 Gaussian 生成与完整 FFT/Krylov/reorth；旧 PR #8 的计时只作为问题背景。优化前后 A 共用 frozen q/p、Gaussian arrays、r_d/r_n 和内部 splitting substeps，不能通过放松精度换速度。paired ratio 的样本单位与 CI 算法会在报告和原始统计中明确列出；配置间变化与计时不确定性分开展示。

`accuracy.csv`/`mode_decay.csv` 的 mean_action_error 是零噪声的确定性 action，不是非线性有限 Krylov A 的实际随机条件均值。实际条件均值采用 `mc.csv` 的 sample/paired mean 与 SE；理想线性 A 的均值/covariance 与有限 Krylov 的随机证据分开。

主要文件：`structure.csv`/inputs 保存与 PR8 相同的物理数组及边；`accuracy.csv`、`selected_rank_checks.csv`、`splitting.csv` 保存新数值对照；`mc_inputs_manifest.json` 和原始 NPZ 关联种子、Gaussian SHA 与三方实际/理想样本；`cache_validation.*` 保存真实配置的参数与 q 更新检查；`compilation.json`、profile CSV/.prof 保存启动/瓶颈证据；`timing_raw.csv`、`setup_raw.csv` 为同批次原始记录；`paired_*ratios.csv` 和 `analysis_protocol.json` 公开 config/round block bootstrap；三张 PNG+SVG 与中文 `report.md` 总结结果。`results/audit.json` 验证全基线 SHA、同图/同噪声/同秩、真实编译、原始样本统计与最终 evidence manifest。
