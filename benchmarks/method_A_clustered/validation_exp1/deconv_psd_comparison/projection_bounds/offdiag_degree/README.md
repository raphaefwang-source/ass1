# Self-excluded degree bound：增量验证

本目录基于 PR #5 的既有 projection correction，仅收紧依赖配置的 degree bound。完整要求见 [requirements_github.md](requirements_github.md)，精确算术证明与数值路径见 [proof.md](proof.md)，中文结果见 [report.md](report.md)。旧 Grid PSD、Ewald splitting、物理核、PPPM、cached matvec、Lanczos/OU、已有 CSV 和已发布图不改动。

```text
D_total_i = sum_j C_ij
C_ii      = tensor-product stencil autocorrelation contraction with C
D_off_i   = sum_(j!=i) C_ij = D_total_i-C_ii
B_off     = 2*max_i lambda_max(D_off_i)
B_old     = min(N*delta, 2*Dtotal_max)
B_improved= min(N*delta, B_total, B_off)
||Delta Gamma||_2 <= B_improved <= B_old     （精确算术）
```

correction-only 物理规范仍是 `C_hat=FFT(C)`，不混入额外 h³ 因子，zero mode/self cancellation 保持。普通路径用逐轴 autocorrelation 求 self tensor，避免每粒子 p⁶ 双重 3D stencil 循环；几乎 self-only 时必须检查 `D_total-C_ii` 的浮点相消，必要时采用明示成本的稳定 fallback。新图表不得用 raw subtraction 补充数值不可用的 `B_off/B_improved`，不得静默 clipping/floor、膨胀 bound 或伪称区间证书。

仓库根目录执行（Python/NumPy/SciPy/Matplotlib/threadpoolctl 与既有实验一致）：

```bash
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/offdiag_degree/audit_sources.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/offdiag_degree/test_offdiag_bounds.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/offdiag_degree/run_offdiag.py compute --workers 2
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/offdiag_degree/run_offdiag.py merge
# compute 与独立核查全部结束后，仅测本轮 bound setup：
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/offdiag_degree/run_offdiag.py timing --repeats 9
# 额外离线 small-N 稳定核查；显式强制路径，可超默认查询预算并耗时数十秒：
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/offdiag_degree/audit_small_stability.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/offdiag_degree/plot_results.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/offdiag_degree/report_generator.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/offdiag_degree/final_audit.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/offdiag_degree/report_generator.py
# report 刷新为 audit pass 后，再封存最终产物 hash；此后不再改报告：
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/offdiag_degree/final_audit.py
```

`run_offdiag.py all --workers 2 --repeats 9` 可顺序执行数值阶段；按参数表 checkpoint 恢复，不应同时启动两个写同一结果的进程。已有 `timing_raw.csv` 且 repeats 相同时，恢复会保留原 raw/context，只重建 derived summary；不同 repeats 会拒绝覆写，须先另名保留旧批次。离线 `audit_small_stability.py` 也保留已有结果；它不属于默认 auto 路径，不修改三项 auto unavailable 或生产预算。旧来源 hash 固化于新 `source_snapshot.json`；`source_catalog.csv/json` 和 `focus_catalog.csv` 保留 case_id、parameter_id、原误差、原界与 true correction norm 的来源，供逐行复核。

恢复已实跑：`results/resume.log` 保存 141 个 cached 参数表复用记录，`results/timing_resume_check.json` 验证 timing raw/context SHA 未改变。报告 sources manifest 将终审 JSON 作为 status-only dependency，不固定其 SHA，避免 audit hash 报告、报告又 hash audit 的循环。

本轮覆盖原 1692 个来源行、141 表/1269 table-config correction；其中 PR #5 的 true norm 复用约为 477 个实际 active dense group 与 729 个 literal-zero group，63 个 wide active norm 留空，最终准确计数以保存的分析/审计为准。逻辑完整算子 case 与 table-config correction 分开统计；不同 short cutoff 可共享同一 correction，不能当作独立 dense 计算。fresh offdiagonal 核查应先将 `W_ii=0` 再求和，以独立验证 autocorrelation self 和相消处理，不用两个大 degree 的相减当真值。

新增文件：

- `results/bounds.csv`：保留旧字段并增加 `B_off/B_improved`、normalized bounds、数值相消/fallback/可用性标记、self trace 与 off/total degree ratio；原界另有 previous 字段作审计。不可用值与 wide norm 保持空白。
- `results/cases_summary.csv`：winner、active bad case、self-dominated outlier；误差与两个 triangle envelopes 都是相对于独立数值 reference 的经验比较。
- `results/timing_raw.csv`、`timing_context.json`：仅本轮独立 bound-only setup。参数 stage 在同 repeat 跨九配置共享时去重；配置 stage 汇总构型/repeat 变化，中位数与 IQR 不构成旧 matvec 速度比。
- `results/smallN_stability_cases.csv`、`smallN_stability_summary.json`：额外针对 auto 相对误差≥1e-8 或 auto unavailable 的六 case 离线核查，显式记录强制 positive 路径的误差、预算越限与额外秒级成本；主 CSV 和默认 unavailable 保持原样。
- `results/analysis_summary.json`：source/逻辑 case/table-config 双口径覆盖、true norm tightness、相消诊断与 unavailable、selected-case 的 1e-4 预算通过数及 setup IQR。
- `figures/`：至少四组 PNG/SVG 和逐图 `plot_data/*.csv`，`plot_status.json` 明示缺失/失败。图 1 比较 true correction norm 与五个界；图 2 为 old/new tightness、self trace fraction 与 off/total degree ratio；图 3 显示 winner/bad case 的九配置两个 empirical envelopes；图 4 只展示新的 setup 批次。

`E_old_measured+B_improved_rel` 是 sufficient/conservative 的经验预算，**不是 certified total error**。目前没有独立认证的 `epsilon_old^cert`，所以 theory-certified candidates unavailable；独立 reference 实测通过的 empirical-pass candidate 不能因为 loose bound 失败自动淘汰。PSD 结构保证对任意合法参数和位置成立，误差界不在每次 Lanczos/cached matvec 中计算。本轮不开发高阶新离散、不实现 M15 special case，也不拼接旧计时宣称加速。
