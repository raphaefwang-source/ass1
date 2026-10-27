# Frozen thermostat: Combination A / B

本目录以 PR #7 head `76ceb517fc24f60dfdf6d9d2b4af7a2109a07e54` 为只读基线，实施 Candidate 1 / S1L 的 frozen thermostat Task 1–3。原始要求保存在 [requirements_github.md](requirements_github.md)。只写本目录；不重写旧结果、不跑 HPC、长时间 BAOAB 或 LJ 轨迹。

`ou_core.py` 实现完整 Gamma 的 two-call Lanczos OU（B）、生产近场多镜像边的 exact 3x3 OU 与 long-only Lanczos 对称组合（A）。`spatial.py` 读取生产 Grid PSD 参数、构造同一个近/远场离散图及独立 dense reference。`protocol.py` 固定质量/温度/种子、注册方向、误差预算和环境记录。`run_benchmarks.py` 依次运行 accuracy、MC、fresh timing；`plot_results.py`、`audit.py` 与 `report_generator.py` 仅消费实际 evidence。

## 环境与复现

依赖 Python 3、NumPy、SciPy、Matplotlib、threadpoolctl。使用项目当前可用的 Python 环境；实际版本、路径、系统与可识别的 native pool 保存在 `results/environment_*.json`。原生线程数通过环境请求为 1，未识别的 pool 不声称已验证。计时阶段应独占本次实验计算进程，避免同时跑测试/图像生成/审计。

完整重算请在**单独的新 clone** 检出只读基线 `76ceb517fc24f60dfdf6d9d2b4af7a2109a07e54`，在同一相对路径 `benchmarks/method_A_clustered/thermostat_ab_local/` 新建目录，只复制本 PR 新目录中的 `*.py`、`requirements_github.md`、`source_snapshot.json` 与 `README.md`。不要复制本次生成的 `results/`、`inputs/`、`references/`、`logs/` 或 `figures/`；脚本会从基线数据重建这些 evidence。直接 checkout 本 PR 会带上已完成标记，不能当作空白重算目录；脚本保护现有 evidence，不覆盖已完成阶段。

在这个 source-only 新目录中执行：

```bash
cd benchmarks/method_A_clustered/thermostat_ab_local
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p logs
python3 -B -m unittest discover -s . -p 'test_*.py' -v > logs/tests.log 2>&1
python3 -B reference_tail_audit.py > logs/reference_tail_audit.log 2>&1
python3 -B run_benchmarks.py accuracy --mc-small 512 --mc-medium 128 > logs/accuracy.log 2>&1
python3 -B stopping_audit.py > logs/stopping.log 2>&1
python3 -B run_benchmarks.py mc --mc-small 512 --mc-medium 128 > logs/mc.log 2>&1
python3 -B refine_small_case.py > logs/rank_refinement.log 2>&1
python3 -B uncensored_stopping.py > logs/uncensored_stopping.log 2>&1
python3 -B run_benchmarks.py timing --rounds 3 --reps 5 > logs/timing.log 2>&1
python3 -B plot_results.py > logs/plots.log 2>&1
python3 -B audit.py
python3 -B report_generator.py > logs/report.log 2>&1
python3 -B audit.py
```

本次实际测试命令从仓库根目录执行 `python3 -B -m unittest discover -s benchmarks/method_A_clustered/thermostat_ab_local -p 'test_*.py' -v`，原始输出保存在 `logs/tests.log`。

`audit.py --allow-incomplete` 可在明确记录未运行覆盖时生成审计；它不会用推测数字填补 N/A，也不会免除已存在结果的真实性/确定性检查。报告生成后再次执行 `audit.py` 可刷新 evidence 哈希；报告引用审计状态，不依赖哈希自循环。

## 数据与解释

- `inputs/` 保存 q、同一 Gamma/GammaS/GammaL、p/z、注册方向，以及实际 MC 样本；配置来源与几何哈希保存在相邻 JSON。
- `results/structure.csv` 保存空间参数、谱、matrix-free/dense action 对比、空间精度与 exact dense FDT 代数残差。
- `rank_errors.csv` 分开记录 damping/noise 的 rank/error、step-change 与实际 Lanczos 诊断；秩选择依据 dense-reference 真实误差，step-change 只是辅助指标。
- `stopping_diagnostics*.csv`/`stopping_summary.json` 对比已测试秩变化与 dense true error，记录诊断 false positive/negative；`coupling_checks.csv` 保存正反向同源变号及独立同噪声 exact/ideal 路径对照，不把奇对称性当作线性或 covariance 证明。
- `splitting.csv` 是理想 exact 子步 A 与 exact full OU 的条件均值/covariance 误差；`mode_decay.csv` 是实际有限 Krylov 无噪声步的方向衰减与 COM 检查。
- `selection.csv` 分别选择 A/B damping/noise 秩及 A 内部子步数。该筛选仅覆盖注册输入域，没有宣称全输入统一误差界。
- `mc.csv` 的逐点近似 95% CI/SE 只表达有限样本不确定性；保存同噪声实际/理想 dense 样本对及 paired mean/second-moment 差，单独量化采样输入上的 finite-Krylov 误差。有限 Lanczos 的基依赖 Gaussian 输入，不能把逐列近似组装成固定 F 后自称得到算法的精确 covariance。
- `refine_small_case.py` 公开复现采样后发现的 N2/B alpha1 rank2 noise 额外 Gaussian 路径未通过及 rank3 完整投影维度修正；原始 evidence/样本保存在 `pilot_rank_refinement/`（带 SHA manifest），仅该 case 使用相同种子重算。该脚本同时刷新正式 stopping/coupling 证据，之后刷新补充 uncensored sweep。
- `timing_raw.csv` 是新 batch、同配置交错 A/B 的完整 frozen step 总 wall cost，包含随机数生成与实际 action 计数；`setup_timing_raw.csv` 为每配置三次 fresh setup 分相计时（排除验证用 dense 组装），`setup.csv` 为 accuracy 构建耗时；`timing_summary.csv`/`timing_comparison.csv` 汇总真实重复，不混用历史 cached-Gamma-v 时间。
- `figures/` 包含五张论文用 PNG+SVG（四项主要比较加不重叠的总步分相图）；[report.md](report.md) 为中文数学、证据、实际覆盖、耗时与限制报告。`results/audit.json` 检查原基线全部 tracked 文件哈希、保存矩阵分解、选秩证据与计时关联，并记录 evidence 哈希。

Exact dense 代数恒等式、有限 Krylov action 误差、有限样本 Monte Carlo CI 是三层不同证据。理想 A 保持固定 COM 的 Maxwell 分布，同时具有非对易 splitting 的动力学误差；有限 Krylov 不自动获得精确 FDT。
