# 本地 BAOAB A/B：Stage 0 + Stage 1

基线：PR #9 `d6a8d208bcdc30731079715dc4c8e925b4fb2f8b`；分支 `codex/baoab-ab-stage01`。
完整执行要求与停止线见 [requirements_github.md](requirements_github.md)。本目录仅包含本地正确性验证、三个独立预平衡初态、短移动粒子轨迹和动态计时。不包含 Stage 2/3 的执行程序。

执行环境为 Apple M2、8 GiB、macOS 27.0.1、Python 3.11.5、NumPy 2.4.6、SciPy 1.17.1；实际环境和基线文件 SHA 保存在 `results/environment_initial.json` 与 `baseline_snapshot.json`。复用原 Grid PSD、full/long Lanczos 和 C 顺序 Pair OU；不改写历史源码或结果。Numba 与本机 NumPy 不兼容，使用 PR #9 已编译 C 后端。

所有命令在本目录运行，禁止与正式计时同时执行其它数值工作。请求 native 单线程环境：

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
mkdir -p logs results inputs figures
python3 -B capture_environment.py > logs/environment.log 2>&1
python3 -B run_regressions.py > logs/baseline_regressions.log 2>&1
python3 -B -m unittest -v test_dynamics test_spatial_pilot test_rank_validation > logs/new_tests_final.log 2>&1
python3 -B spatial_validation.py > logs/spatial_selection.log 2>&1
python3 -B run_pilot.py initialize > logs/initial_preparation.log 2>&1
python3 -B run_pilot.py stage0 > logs/stage0_actual_starts.log 2>&1
python3 -B run_pilot.py trajectories > logs/production.log 2>&1
python3 -B run_pilot.py timing > logs/interleaved_timing.log 2>&1
python3 -B run_pilot.py visited > logs/visited_validation.log 2>&1
python3 -B provenance.py > logs/backend_provenance.log 2>&1
python3 -B plot_results.py > logs/plots.log 2>&1
python3 -B report_generator.py > logs/report.log 2>&1
python3 -B independent_audit.py --replay > logs/independent_audit.log 2>&1
```

新的实际运行命令、测试日志、参数选择、初态生成、空间 direct reference、Krylov 验证、每步轨迹/相位时钟与随机数证据均保存在本目录。启动/参数表、预平衡、验证、生产步与额外独立动态计时分开计费。

原始 evidence 已存在时请在独立的新目录复现，避免覆盖本次完成的数据。由 `baseline_snapshot.json` 记录的基线 tracked 文件必须保持哈希不变。跨步只允许复用物理 FFT/kernel 表，不能跨步复用位置相关 graph、assignment、degree 或边系数。

在新的 baseline clone 创建同名 source-only 目录，仅复制本目录源码、README、requirements 和 baseline_snapshot，不复制 inputs/results/logs/figures。初态从 FCC 经独立 scalar Langevin+LJ 实际生成；6000×.005=30 的有限预平衡不等于精确 canonical mixing 证明。默认 drive 只允许200步生产或40步计时，防止误启动后续正式实验。

完整 BAOAB wall 在第二次 momentum kick 后结束，包含两次保守 force、全部 q-dependent setup、RNG、漂移/kick 和完整 OU。构图所需 q token hash、force 同时返回的势能/最小距离也在计时内；额外证据哈希、诊断汇总与磁盘写入在计时外并单列。CSV 的 FFT、edge eigh、reorth 和 Ritz eigen 是嵌套子阶段，不得与父阶段重复相加。所有原始 step时钟在 results 下，图仅由这些实际数据生成。
