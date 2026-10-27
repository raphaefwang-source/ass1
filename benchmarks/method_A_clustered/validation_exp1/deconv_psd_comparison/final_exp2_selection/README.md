# Final Experiment 2：Grid PSD 四步联合选参

本目录基于 `56b9e1f`，工作分支 `codex/final-exp2-gridpsd-selection`。完整任务见 [requirements_github.md](requirements_github.md)。所有新代码、原始数据、图和报告均留在此目录；不修改物理核、Grid PSD 投影、Ewald split、OU/Lanczos、旧 cached matvec 或 PR #3–#6 证据。

固定主基准 N=256、rho=.75、kappa=.2、amplitude=1，使用 uniform/moderate/strong × 3 seed 九配置的周期 independent direct reference。目标 epsilon=1e-3、1e-4、1e-5，1e-6 及 N 扩展按实际预算报告 N/A。最终结果是局部搜索的 **best tested**，不是全局最优。

四步为：tail prefactor 指导初始 sR/sF；按 `kc=2*xi*sF`、`eta=kc*(L/M)/pi<=eta_max<1` 与完整 Fourier shell/influence-denominator 稳定性选 M；按 alias/mesh 规律选初始最小合理 p 并满足 `1<=p<=M`；在 `(xi,sR,sF,M,p)` 邻域联合搜索并以完整 `max_9 E_Gamma<=epsilon` 验证。prefactor 仅是初始化模型，详见 [theory_source_audit.json](theory_source_audit.json)。Original 最佳参数可作 baseline/种子，不固定 Grid PSD 的选择。

已有 `snap_kc` 将 nominal `2*xi*sF` 映射到完整 reciprocal shell。图表报告actual kc和effective sF=`actual kc/(2*xi)`，eta用actual kc；不把nominal sF与snapped cutoff之间的差异当数据错误。B2有独立sT定义，不从其占位sF推导Fourier kc/eta。

“最小合理p”仅指初始化模型与有限实测域，不能等同最低合法p。数学合法域仍1<=p<=M；本轮从p>=2、初始化cap9开始，p1未测而非数学排除或实测不合格。zeta(p) majorant仅p>1有限，p2为首个连续cardinal B-spline；经验alias_C来自p3..5的完整旧误差，不提供p2/p1保证。最终报告动态列实际测得p集合，不声称遍历全部合法域。

本轮不把 B_off、Ndelta、degree bounds 或 full-matrix eigenvalue PSD 检查作为硬筛。九配置 empirical-pass 不等同于任意配置统一精度证明；PSD 仍来自原 Grid PSD 结构，没有 `epsilon_old^cert` 时完整精度认证 unavailable。每候选构造自身投影 table，但缓存后的 Gamma-v 不重复投影或计算 bound。

搜索、完整算子误差验证全部结束后，先完成同机同线程隔离重复 timing，再绘图，避免渲染 CPU 干扰正式时钟。仓库根目录执行：

```bash
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/audit_sources.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/test_selection.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/search.py all --workers 2 --max-local 24 --max-boundary 12
# 可选 N 扩展：在正式计时前生成独立 N=64/128 reference：
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/scaling.py refs
# 以下必须等所有 accuracy/reference worker pool 退出；cross仍是准确性阶段：
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/timing.py cross
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/timing.py prescreen
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/timing.py formal --rounds 5 --reps 7
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/timing.py components
# 完成正式主 timing、components并生成 best_by_epsilon.csv 后：
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/scaling.py run
# 所有正式timing结束后才执行独立结构诊断（不参与选参）：
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/structural_audit.py
# 计时后同一导入环境的provenance，不新增性能样本：
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/runtime_environment.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/plot_results.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/report_generator.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/final_audit.py
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/report_generator.py
# audit pass写入report后再封存；最后不再改report：
python3 -B benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/final_exp2_selection/final_audit.py
```

`plot_results.py` 只读本轮 CSV，缺失数据在 `figures/plot_status.json` 明确跳过。各图均保存 PNG、SVG、逐图 `figures/plot_data/*.csv`，保留 candidate id、method、timing batch、实际误差与可用状态。

正式排名用同一 common batch 的 5 round×7 repeat×9配置，中央统计量为九配置计时中位数的中位数；q25–q75保留原始样本分布，不是CI。表setup每候选新建3次，configuration setup每配置新建3次。near/FFT/spread/mesh分解另为独立诊断batch，每阶段每配置9样本，不混入正式full-action排名。cross-method controls 将Grid/Original各自proxy winner的参数交叉构造、独立九配置验证，减少搜索路径不同造成的假优势。direct dense只计已缓存独立reference的G@v；fresh dense assembly setup未测，因此setup与动态步总成本N/A。

正式计时只停止本实验自己的 search/reference/render workers；外部 chrome_crashpad 等进程持续CPU负载，不操作或终止其它应用。不能宣称机器空闲或硬件完全隔离。`results/system_load.json`记录每配置的load average和top15进程；同批随机交错、单线程与paired区间减轻顺序偏差，但无法消除外部竞争/调频等影响。cross_plan采用每source/epsilon top2 reciprocal controls，保留旧核查并追加独立验证，正式计时所有可行controls。

“单线程”指导入前六类线程环境变量（含VECLIB_MAXIMUM_THREADS）请求1且raw/context记录threads=1；Mac Accelerate未被threadpoolctl枚举，threadpools=[]，不能称工具已独立验证实际native线程数。所有方法同进程同设置。正式排名的ratio-of-nine-config-medians与45个configuration×round paired block比值的中位数是不同估计量，中位数不可与除法交换；bootstrap区间条件于已选候选，不包含有限搜索的不确定性。

- 图 1：Grid PSD 的最佳已测 sR、sF、xi、kc、rc、M、p、eta 随 epsilon。
- 图 2：正式已计时候选的精度–成本 Pareto；只连接同方法/同批次且达到当前目标的可行点，超标点为空心，不用 screening 时间冒充正式时钟。
- 图 3：Grid PSD、Original、B2、direct dense 的同精度 error/runtime；各用自己的可行/已优化参数，未测或不达标显示 N/A。
- 图 4：table/config setup、cached full matvec、near/spread/FFT 分解；重叠阶段不能相加。
- 图 5：每动力学步总成本的透明线性摊销，多个 matvec 次数、table 每步重建/复用100步；这是实测阶段中位数推算，不是新增轨迹实测。
- 图 6（若实测）：单个 nested uniform_s2 构型的 N 扩展，N64/128 新 reference、N256 沿用主 reference；从主已选参数按箱长重标定 M 并至多四次 mesh refinement。它不等同于九配置/每N独立全优化，不外推未测 N 或声称渐近全局最优。
- 补图 7（有逐配置诊断时）：同参数投影前后误差、Eproj及负网格块fraction；它不替代独立优化Original的method comparison，活跃性不作hard筛。

摊销公式为 `table_setup/table_reuse_steps + config_setup + n_matvec*cached_matvec`。假设每步位置改变，做一次 config setup；不假设额外邻居表跨步复用。box/kappa/xi/cutoff/M/p 改变会使 table 缓存失效。`results/amortization.csv` 由timing脚本保存；报告另存 `results/report_amortization.csv`，图源另存 `results/plot_amortization.csv`，都不覆写正式时样。

[report.md](report.md) 由 `report_generator.py` 动态读取已保存 CSV/JSON，包含四步初始化与候选范围、参数表、每配置完整误差和投影诊断、same-batch-only 时钟比、setup/摊销、测试与未完成项。`results/analysis_summary.json` 保存机器可读汇总，`report.sources.json` 固定所读数据/脚本 SHA。终审 JSON 只作 status-only dependency，避免 report/audit 互相 hash 的循环；刷新 report 后应最后一次 audit 封存且不再改 report。
