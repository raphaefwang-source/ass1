来源：https://github.com/raphaefwang-source/ass1/pull/6#issuecomment-6447774395

## Codex 下一轮：按最终四步选参规则重跑 Experiment 2（Grid PSD）
请在 Mac Codex 实际实现、运行实验、出图并提交独立 draft PR；不要只做计划或文字分析。以本 PR #6 (`codex/offdiag-degree-bounds`) 为基线，先检查实际 HEAD / AGENTS.md / 已有结果，创建新工作分支和独立实验目录，**不要覆盖旧数据、不要改物理核、Grid PSD 投影算法、OU/Lanczos 或历史代码**。

**目标：第二项 friction operator 的最终等精度选参验证。** 固定主要基准 N=256, ρ=0.75, κ=0.2, amplitude=1，沿用九个 uniform/moderate/strong × 3 seed 配置、原周期 direct reference、同一 FFT 归一化、真实 k=0 和完整 self cancellation。目标 ε∈{1e-3,1e-4,1e-5}，若资源允许再 ε=1e-6；任何超预算情况明确写 N/A 而不杜撰结果。

严格按照以下**四步选参**重做：
1. 给定 ε，以 `s_R≈s_F≈sqrt(log(C/ε))` 做理论指导的**初始化**（不是硬约束，C/prefactor 按既有 Ewald 误差模型；`r_c=s_R/ξ, k_c=2ξ s_F`）。
2. 对每个 ξ/kc 起点，根据已确立的 FFT/Nyquist safety `k_c h/π≤η_max<1`、influence denominator stability 和 `h=L/M` 选初始最小合法 M（可按 FFT-friendly sizes；报告实际 M、η）。
3. 根据已有 alias/mesh 误差规律与合法性 `1≤p≤M` 选最小合理 p；M、p 可相互微调，不要误把 p 越大一定越好。
4. 以以上参数为种子，在附近**联合搜索** `(ξ,s_R,s_F,M,p)`，每组候选都构造自己的 **Grid PSD projection**，完整算子统一做 `E_Γ(q)=||Γ_h^{GridPSD}(q)-Γ_D(q)||₂/||Γ_D(q)||₂`，以 `max_{9 configs} E_Γ≤ε` 为实际等精度通过条件，目标最小化可重复测得的 cached full Γv matvec 时间，并**单独**报告 table/config setup 与按每个动力学步摊销的总成本。原 PPPM 最佳参数只能做 baseline，**不能直接固定为 Grid PSD 的最优参数**。不得把 B_off / Nδ / degree bounds / full-matrix eigenvalue PSD checks 加入选参的硬约束；PSD 由原 Grid PSD 结构保证（仍做独立结构性代码检查）。

交付：按 ε 的 `(s_R,s_F,ξ,k_c,r_c,M,p)` 最佳已测试参数表、每配置 E_Γ、投影前后误差/投影活跃诊断（只作解释）、cached matvec/near/FFT/setup 分解、同机同线程重复 timing；与 Original PPPM、B2、direct dense 的**同精度**结果比较（各自用可行/已优化参数，标清未达标或未测）；图：参数随 ε、精度–成本 Pareto、误差与成本对比、随 N 扩展（若可负担）。输出 PNG/SVG、原始 CSV、中文 report、复现命令、单元/审计结果。禁止声称局部搜索是全局最优；区分实测九配置合格与任意配置的统一精度证明，不把 B_off 的经验 envelope 当作 certified total error。保留既有 PR #3-#6 结果/证据，不覆盖、不 force-push、不 merge。

**重点：本次不是重做 B_off 分析，也不是开发新 PSD 算法；就是按最终选参规则重新跑第二项，并判断 Grid PSD 在等精度下能否保留原 PPPM 的速度优势。**
