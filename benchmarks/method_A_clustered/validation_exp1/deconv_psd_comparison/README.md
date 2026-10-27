# PPPM 三方案 PSD-preserving 比较

本目录基于 B2 提交 `128aaf1cd8bf39c69d2bf2db9619e478fefc0c73`，独立实现实空间网格 3×3 PSD 投影。完整要求存于 [requirements_github.md](requirements_github.md)，结构证明与精度限定见 [proof.md](proof.md)，中文实验结论见 [report.md](report.md)。旧源码、CSV、图片和 reference 只读保留。

方法稳定名称：`PPPM-original`、`B2-positive`、`PPPM-grid-PSD`。物理振幅 c=1、Candidate 1 split、完整周期镜像、相同配置和独立 unsplit direct references 不变；选参的 N=256、rho=.75、kappa=.2 不变。新 sF 是投影前 Fourier 截断，投影后保留全部网格模态与实际零频；B2 的 sT=kappa R_T 是不同物理表窗口。

在仓库根目录执行（Python 3.11，NumPy、SciPy、Matplotlib、threadpoolctl）：

```sh
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/test_grid_psd.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/run_accuracy.py sweeps --workers 2
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/run_accuracy.py convergence --workers 2
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/run_xi_sweep.py --workers 2
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/run_search.py coarse
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/run_search.py local
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/run_search.py boundary
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/table_audit.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/merge_diagnostics.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/analyze_regimes.py
# 待上面的 accuracy/search 任务全部结束后，独立串行计时：
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/run_timing.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/cross_winner.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/archive_primary_timing.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/run_timing.py --main-confirmation
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/merge_frontiers.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/best_table_audit.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/plot_results.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/report_generator.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/finalize_audit.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/report_generator.py
```

也可使用 `run_all.py --workers 2` 顺序执行，或通过 `--from-stage convergence` 等从已保存 checkpoint 继续。sweep 按物理状态/配置保存到 results/sweep_parts；convergence 按配置保存；搜索每个候选/每次九配置验证均保存，不必重复已完成点。不要同时启动两个写同一 CSV 的搜索进程。新的计时批次用 `run_timing.py --output /独立新目录`；脚本拒绝覆盖旧批次。

`results/sweeps.csv` 包括两份经 SHA 核验的历史精度数据和本次新方案；历史运行时间不用于本次速度比。`results/convergence.csv` 用独立 reference 减 sR=8 的短程极限定义统一 mesh-only 误差。拟合排除 <=1e-11 的 reference/roundoff 区，并在 CSV 保存区间；跨投影活跃切换区的斜率只解释为有限区间实测值。`results/search.csv` 保留失败的 Nyquist/非法候选，`validation.csv` 的完整谱范数而非 probes 决定可行性。Gaussian probes 仅作为另一个 action 指标。

`results/xi_sweep.csv` 是额外 135 行受控 ξ 实验：固定 κ=.2、rho=.75、M20、p5、sR=sF=3、B2 sT=36，取 ξ=.35,.6,.85,1,1.25；三方法×九配置。此图研究固定其余数值参数时的 ξ 精度关系，不用分别优化后的参数替代受控比较。

正式性能目标为完整 cached near+far action。统一单计算线程、随机候选顺序、5 rounds×7 samples×9 configs；分别保存 table setup、configuration setup、near、spread/gather、FFT、完整 action 与数组内存。不同组件独立计时中位数无需相加。计时报告记录系统外部负载；未控制系统其他应用，因此结果是该共同批次条件下的测量。reference/dense assembly 不混入 matvec。

第一轮 15 候选、4725 个正式样本及全部原始文件完整保存在 `results/primary_timing/`，归档 SHA 在 `primary_archive_manifest.json`。交叉验证发现 Original 也能使用新方案 winner 的 M15 参数，因此主 1e-4 最终比较使用独立第二轮确认批：三方法 primary winners、两条历史最优 anchors 和 Original@grid-winner，共六候选，仍执行5×7×9。主档用第二轮，其他 epsilon 用第一轮；`frontier.csv` 的 timing_batch/source 明示出处，同一 epsilon 内三个方法来自同一批次。不能把不同批次的参数差或搜索路径差误称为投影本身带来的加速。

所有图有 PNG、SVG 和 `figures/plot_data` 源 CSV，可仅重跑 `plot_results.py`。`audit_initial.json` 包含开始时所有 tracked 文件与全部使用 reference 的 SHA-256；结束审计核查它们未变。
