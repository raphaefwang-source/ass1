# Grid PSD Projection：修正误差上界验证

本轮只在此独立目录计算、核查和可视化既有投影修正的上界。旧 Grid PSD 投影、参数搜索、OU、cached matvec、旧 CSV/图/reference 均不改动。完整任务见 [requirements_github.md](requirements_github.md)，数学条件与证明见 [proof.md](proof.md)，中文实测分析见 [report.md](report.md)。

对物理实空间 correction 表 `C_d=(-S_d)_+`，定义 `delta=max_d||C_d||₂`。在非负、partition-of-unity 的同权 stencil 下，精确算术满足：

```text
B_simple = N*delta
B_degree = 2*max_i lambda_max(D_i^C)
||Delta Gamma||₂ <= B = min(B_simple,B_degree).
```

`D_i^C` 直接从 correction 表与粒子 stencil 求得，包含 self；graph action 的 self response 代数抵消。数学实数证书与 float64 FFT/eigensolve 的诊断分开。所有 `E_old/E_new` 都是相对于独立数值 reference 的经验误差，`E_old_measured+B/||Gamma_D||₂` 只是该数值参考下的 triangle envelope。没有独立认证的旧离散误差界，因此 `certified_total_bound` 始终 unavailable。

在仓库根目录运行（Python、NumPy、SciPy、Matplotlib、threadpoolctl 与原实验相同）：

```bash
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/audit_sources.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/test_bounds.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/run_bounds.py compute --workers 2
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/run_bounds.py merge
# compute workers 全部退出后，仅计本轮新增 bound setup：
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/run_bounds.py timing --repeats 9
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/plot_results.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/report_generator.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/final_audit.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/report_generator.py
```

`compute` 按唯一参数表写入 `results/parts/` 并检查来源摘要，已完成 checkpoint 可恢复；不要同时运行两个写同一产物的进程。`merge` 只合并已完成的新 checkpoint，不重算或覆盖旧研究结果。141 表全部命中 cache 的实际恢复检查保存为 `results/resume.log`。`timing`/`all` 若发现已有 `timing_raw.csv` 且 repeats 相同，会保留原 raw/context，仅重建 case 的派生中位数；repeats 不同则抛出 `FileExistsError`，要求先保留或改名旧批次。它不会静默覆盖计时证据。

`source_catalog.csv/json` 保留原数据集、来源行号、稳定 `data_id/case_id`、参数与 reference 信息；`source_snapshot.json` 与最终 `results/audit_final.json` 核查历史文件未变。`bounds.csv` 允许同一真实 case 被多个来源引用，报告按 case_id 去重；不能把最终 aggregate timing 行当作单构型误差。

核心输出：

- `results/bounds.csv`：两个界、合并界、完整经验误差、匹配的 mesh-only 误差、真实零、负谱、对称化与比较 residual、roundoff violations。宽活跃表若未单独算 correction-only dense norm，其 norm/tightness 留空；历史完整新旧算子差另有来源字段，不冒充 true norm。
- `results/cases_summary.csv`：主 1e-4 winner 与九配置 active bad case；`timing_raw.csv` 和 `timing_context.json` 只测新增 correction-bound 参数/配置 setup。参数表和 correction FFT 的同一次样本跨九配置共享，不计作九个独立样本。
- `results/verification_summary.json`、`analysis_summary.json`：覆盖、true dense tightness、经验预算通过数、浮点超界、参数扫描与 selected-case 的中位数/IQR。
- `figures/`：PNG、SVG、逐图 `plot_data/*.csv` 与 `plot_status.json`。可仅重跑绘图脚本，不执行数值实验。

图 1a 完整 operator error 与六条 correction 曲线；图 1b 单独标记 mesh-only error。图 2 仅在独立核查的非零 correction norm 上计算 tightness，并显示 delta/负块比例。图 3 区分 xi/sF/p/rho/kappa 的受控 preset 与活跃性；图 3b 补充真实 moderate M/p/rho/kappa，历史没有 moderate sF 扫描，不能虚构。图 4 比较 winner/active bad case 与新增 setup，误差棒为原始样本 q25–q75；图 5 显示逐配置 empirical triangle envelope 对 1e-4 预算的保守性。七组图的真零使用明确的显示 floor，实际数值与零 mask 保留在 CSV；不跨投影开关区拟合或宣称 12/26 阶收敛。

本轮不开发新 PSD 高阶离散，不做 M15 special case，不与旧计时拼接速度比，也不把 empirical envelope 写成 certified total error。
