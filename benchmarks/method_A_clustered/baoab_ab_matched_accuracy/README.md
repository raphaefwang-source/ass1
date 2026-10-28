# 本地 A/B 冻结 OU 精度匹配与动态 BAOAB 成本

基线：PR #10 `84174ae0d11e3f5826cda000c688d147a1b56241`；工作分支 `codex/baoab-ab-matched-accuracy`。完整要求和停止线保存在 [requirements_github.md](requirements_github.md)。结果解读见 [report.md](report.md)。

本轮固定旧LJ模型和Grid PSD参数；只增加A的完整内部OU子步，先在完整765维COM子空间实际验证三个冻结精度档，再进行真实运动粒子A/B动态计时。旧PR3–10的代码、输入与结果保持只读。三个旧独立预平衡q/p通过SHA复用，不重新运行canonical preparation。

## 环境和 source-only 复现

只在本地Mac执行。依赖Python 3.11、NumPy、SciPy、Matplotlib、threadpoolctl、psutil及可用clang；复用PR9 C sequential 3×3 pair OU，编译关闭fast-math和FP contraction。原Numba与本机NumPy不兼容时不用于性能结果。请求native单线程，Accelerate实际线程数若不可观测会明确记录。

请在新的基线clone中检出上述SHA并创建独立分支，在同一相对路径建立source-only目录。仅复制本目录的 `*.py`、README、requirements、`.gitattributes`、`.gitignore`、`baseline_snapshot.json`，不要复制本次inputs/results/logs/figures或缓存。已有完成marker保护真实证据，不能直接覆盖现有完成实验。

从仓库根目录切入本目录后，顺序执行；计时期间不得并行运行本任务其它数值工作：

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
mkdir -p logs results inputs figures
python3 -B capture_environment.py > logs/environment.log 2>&1
python3 -B prepare_domain.py > logs/prepare_domain.log 2>&1
python3 -B run_regressions.py > logs/baseline_regressions.log 2>&1
python3 -B -m unittest -v test_matched_dynamics test_frozen_accuracy test_stochastic_validation test_plot_results > logs/new_tests_final.log 2>&1
python3 -B - <<'PY' > logs/frozen_accuracy.log 2>&1
from matched_common import load_states, PARAMS, RESULTS
from frozen_accuracy import run_accuracy
run_accuracy(load_states(), PARAMS, RESULTS/'accuracy')
PY
python3 -B stochastic_validation.py select > logs/stochastic_selection.log 2>&1
python3 -B stochastic_validation.py common > logs/stochastic_common.log 2>&1
python3 -B stochastic_validation.py mc --samples 16 > logs/stochastic_mc.log 2>&1
python3 -B stochastic_validation.py wrapper > logs/stochastic_wrapper.log 2>&1
python3 -B run_experiment.py timing > logs/dynamic_timing.log 2>&1
python3 -B run_experiment.py cost_curve > logs/cost_curve.log 2>&1
python3 -B run_experiment.py post_validate > logs/post_timing_validation.log 2>&1
python3 -B provenance.py > logs/backend_provenance.log 2>&1
python3 -B analyze.py > logs/analysis.log 2>&1
python3 -B plot_results.py > logs/plots.log 2>&1
python3 -B report_generator.py > logs/report.log 2>&1
python3 -B independent_audit.py --replay > logs/independent_audit.log 2>&1
```

环境捕获在实验前执行；capture_environment会保存当前Mac/分支/依赖/编译环境。baseline_snapshot核对从基线继承的tracked文件，不包含本轮新文件。C库是可再生缓存；源码、实际backend/库SHA或运行provenance按记录核对，不能将Python fallback冒充已编译性能。

## 文件和解释边界

- `results/domain.json`、`inputs/domain_*.npz`：3个旧独立初态、6个旧轨迹q_mid、扫描1200个已保存中点的最小距离态及来源SHA。
- `results/accuracy/`：全部实际k=1/2/4/8/16/32的独立稠密refs、完整COM均值/协方差maps、范数/分母、斜率、SHA与最小实测通过k。
- `results/stochastic/`：每个k实际rank选择，最终共用rank重新检查、逐子步实际输入、同方法理想参考、独立Gaussian/RNG证据；代表态conditional/equilibrium各16样本的方向moments/SE/CI。
- `timing_steps.csv`、`timing_substeps.csv`、`timing_rounds.csv`：每档6个配对round×A/B各20个真正moving外步；所有选定k在运行时实际启用，q相关setup每外步只执行一次。
- `cost_curve_extra_steps.csv`：非选中k=2/8/32各3初态×A/B各3步，仅为小样本成本曲线；实际验证域和样本数与匹配档分开。
- `post_timing_validation.json`：新轨迹18个实际末步q_mid的完整COM误差检查。`analysis.json`和`paired_bootstrap_samples.npz`保存新batch配对round统计与原始bootstrap样本。
- `figures/`：4组PNG+SVG、绘图CSV与SHA manifest；`report.md`为中文报告，`independent_audit.json`为最终证据审计。

完整BAOAB wall包含两次LJ force、fresh q-dependent setup、所有k子步、RNG与外层drift/kick。构图token hash及force同时返回的能量/最小距离在wall内，额外证据哈希、诊断汇总/磁盘IO在wall外单列；one-time参数表独立记录。FFT、edge eigh和reorth属于父阶段内部，不可重复相加。

严格区分：旧空间误差预算2e-7、旧Lanczos action目标2e-8、本轮人为设置的冻结transition stress阈值2e-8。完整理想COM operator-norm pass不被改写成finite非线性Lanczos全域保证；16样本方向MC区间较宽，不能证明2e-8级covariance精度。A自然随机路径与B不同结构噪声的raw output距离不用于偏差判定。

PR10旧A=397.96ms/B=515.77ms只作背景，所有本轮ratio使用新的同batch实际动态时间。当前停止线是有限本地matched-frozen-accuracy moving pilot；不自动运行HPC/SSH、长轨迹、16 replica正式跨dt研究、Grid PSD联合生产优化或C/D/E。完整BAOAB等物理观测精度A/B选型仍需下一阶段独立验证。
