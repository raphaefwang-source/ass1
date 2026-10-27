Source: https://github.com/raphaefwang-source/ass1/pull/8#issuecomment-6455563048

@codex 请**实际实现并运行** PR #8 的 **Combination A Pairwise OU 性能优化**，重点解决本地 N=256 的 Python 逐边 sweep 瓶颈，并重新进行同精度 A/B 对比。不要只写方案；按下述验收标准出代码、真实计时/精度证据、报告和新 draft PR。

### 当前实测症结（来自 PR #8，不要当成新结果）
- PR #8 head `bce73847f2968c27e1df5e2ff4be57393c4d2ddd`，`benchmarks/method_A_clustered/thermostat_ab_local/ou_core.py` 的 `EdgeOU.sweep` 当前按 edge 用 Python for 循环，循环内每边两次小矩阵乘、临时数组及粒子更新。
- N=256、ε=1e-4、alpha=1，A 79.8ms vs B 11.6ms（跨9配置中位数）；代表配置两次 near sweep 约31.95+32.49ms，占 A 约88%。此次必须先针对 near sweeps 优化，公平验证之后再判断 A 算法快慢。
- B 的 full-Γ Lanczos、Grid PSD 核和旧实验数据是对照，不要改动。

### 核心实现要求
1. 以 **PR #8 head 为只读基线**，先检查实际仓库 HEAD/AGENTS.md/依赖。创建**独立新分支和 draft PR**，例如 `codex/optimized-pairwise-ou`，新工作/结果放入 `thermostat_ab_local/optimized_pairou/` 或清晰独立目录；不要覆盖 PR #3–#8 的原始 CSV/PNG/报告和已测轨迹，不 merge/force-push，必要时用最小适配包装/新增模块而不是悄悄改旧逻辑。
2. 首选**真正编译的顺序边 sweep kernel**（Numba njit 若本地环境兼容，或其他可用的编译途径；无相应依赖时提供可运行 fallback 并明确说明），避免 Python per-edge loop。先 profiling 确认占比，别为了优化引入不可复现的新环境假设。
3. 每个固定 frozen q 构建原来**完全相同**的无向、按既定顺序排列的 image-summed near edges；**多周期镜像全部聚合，完整3×3 PSD K_e，不得假设 rank-one**。缓存各 edge 3×3 的 eigensystem，且在 `tau=dt/2, m,theta` 已知时**预计算**:
   - `D_e = (1/2)*U diag(expm1(-2*tau*lambda/m)) U^T`（避免 `exp()-I` 消减），
   - `S_e = (1/2)*U diag(sqrt(2*m*theta*(-expm1(-4*tau*lambda/m)))) U^T`.
   sweep 内仅对 `r=p_i-p_j` 计算 `dp=D_e@r+S_e@eta_e`，然后 `p_i+=dp; p_j-=dp`，显式无 per-edge eigh/expm/小矩阵构建；写清楚为何与原相对动量 exact OU 同义。参数、位置或边张量改变时使缓存失效；系数建立时间计入 setup。浮点误差、zero eigenvalue、small tau、rank-deficient/full-rank、空边集合均测试。
4. **严格保留原算法动力学而不是换一个算法：** forward 按原 order、reverse 严格逆序；共享粒子必须按顺序更新，**禁止**未经独立理论+误差验证地把共享边并行向量化/重排；`eta_forward`、`eta_reverse`、`z_long` 独立，优化前后对同一事先生成的 Gaussian arrays 及同一 frozen (q,p) 作 pathwise paired 比较。不能为了快削减 edge、重新定义 cutoff、跳过 long noise、改变 dt、重用独立噪声、把自项/zero Fourier mode 修改或额外加 `delta Pi`。使用原 `GammaS+GammaL` 及 Grid PSD。
5. 保留可选择的原 `EdgeOU.sweep` 作为 numerical reference/fallback。若 JIT 首次编译成本大，**分别**报告 cold compile、cache/table/edge/coeff setup、steady-state sweep；公平比较 A_baseline, A_optimized, B_full。编译前热身，不把首次 JIT 计入已缓存步耗时，但必须披露单次首次启动与位置变化引起的真实 per-step setup。报实际运行时 NumPy/SciPy/Numba 版本、原生线程池可验证性、硬件/负载。
6. 在完全相同系统、物理参数、选秩和 `alpha=dt λ_max/m` 上重跑 PR #8 既有本地测试（N=2/8/16/64/128/256；重点九配置 N=16/64/256；ε=1e-4，另 ε=1e-3/1e-5 能力范围内）。**先精度后计时**：单边均值、方差和动量；严格与 Python sweep **同源逐路径向量误差**；A/full-Gamma dense conditional mean/cov、理想分裂误差及 dt/2 收敛；组合 OU 的 FDT 方向 Monte Carlo checks、COM 与 near/long graph 完全一致。至少对一组配置验证 cached coefficients 用多种 dt/temperature 的更新一致性。区分理想 A 的 frozen Maxwell 保持、有限 Lanczos MC empirical FDT 与相对于 exact-full-B 的 splitting dynamical error，禁止因变快而放松精度预算。
7. **重跑同批次交错计时**：A_python vs A_compiled/optimized vs B，使用相同已验证 r_d/r_n 和内部分裂 substeps，所有 Gaussian 生成与 FFT/Krylov + reorth 都算入整步 wall；分别报告 `t_near_forward,t_near_reverse,t_long_damp,t_long_noise,t_full_B,t_coeff_setup,t_edge_eigh,t_config_setup,t_total_step`，注明真实 sample count / median / variation / paired ratio。若新的 optimized A 胜过 B，继续要求同一 frozen reference 下的实际等精度成立；否则报告真实差距，不人工改 baseline。**禁止将 PR #7/#8 不同 timing batch 混为 speedup。**
8. 提交真实源代码、单元/回归测试、原始 CSV/JSON、性能 profiler、至少两张 PNG/SVG（near sweep 加速/分相成本，完整 A/B equal-accuracy comparison）、中文 report.md、复现命令与独立 audit。保留原先 26 测试全部通过，并增补同源路径和缓存失效测试；若环境缺乏 Numba/编译器或工作量受限，明确说明实际能做到的性能及未运行部分，绝不编造数字。

**停止线**：本次只做 A 的 near-edge OU performance engineering + 本地 Task3 重测，不跑 HPC、长时间 LJ、BAOAB，不开发 C/D/E，不修改二项 Grid PSD 算子或物理模型。最终说明：A 的成本能降到多少、是否优于 B、是否仅有 Python overhead 限制，以及后续 HPC 是否有必要。
