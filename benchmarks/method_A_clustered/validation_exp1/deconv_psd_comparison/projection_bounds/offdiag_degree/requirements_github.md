来源：https://github.com/raphaefwang-source/ass1/pull/5#issuecomment-6435753178

## Codex follow-up: self-excluded degree bound（增量绘图 + 选参含义）

请实际实现、运行与出图。本任务是在现有 **PR #5** `codex/projection-correction-bounds` （已验证 `Nδ` / `2 D_total`）上的小幅增量；请先确认 Git HEAD、AGENTS.md 和是否有人新推送。**不要改** Grid PSD Projection 本体、Ewald splitting、old PPPM、物理核、Lanczos/OU、已有数据、cached matvec 和已发布图。可以在 PR #5 上安全追加 commits，或者创建以 PR #5 为 base 的后续 draft PR；不要覆盖旧文件、force push 或 merge。新增材料请放独立 `projection_bounds/offdiag_degree/`。

### 数学目标（任意合法参数/任意粒子位置、精确算术）
现有 `C_d = (-S_d)_+ \succeq0`，`C_{ij}=Σ_{g,l} w_{ig} C_{g-l} w_{jl}\succeq0`，`\Delta\Gamma = \operatorname{Laplacian}(C_{ij})\succeq0`。原界
`B_simple=Nδ`, `B_total=2 max_i λmax(D_i^{total})`，其中 `D_i^{total}=Σ_j C_ij` 含 self。
新增 **self-excluded** degree
`C_ii=Σ_{g,l} w_ig C_{g-l} w_il`，
`D_i^{off}=D_i^{total}-C_ii=Σ_{j≠i} C_ij\succeq0`，
`B_off=2 max_i λmax(D_i^{off})`，
`B_improved=min(Nδ, B_total, B_off)=min(Nδ,B_off)`（精确算术）。
严格证明 `||\Delta\Gamma||_2<=B_improved<=B_old`，证明 self cancellation；保持 correction-only 物理归一化和 zero mode。注意这个界依赖配置，但 PSD 保证本身不依赖配置。

### 实现：避免 N² 和 catastrophic cancellation
优先直接从 **stencil autocorrelation** 求 `C_ii`，而不是对每个粒子 p^6 双重 3D stencil 循环或 3N dense 矩阵。对于 tensor-product spline，逐轴
`a_{i,α}(d)=Σ_{u,v} w_{i,α,u} w_{i,α,v} 1[(g_{i,α,u}-g_{i,α,v}) mod M=d]`，
`C_ii=Σ_{d=(d1,d2,d3)} C_d a_{i,x}(d1)a_{i,y}(d2)a_{i,z}(d3)`；
只访问实际 autocorrelation support，注意周期 wrap、M 小于 2p-1、p=1、奇偶 M、stencil 重复 index、6×3x3 对称编码及浮点求和。目标额外成本 O(N p²+N (2p-1)^3)（固定 p），不形成粒子对稠密图；已有 `D_total` correction-only FFT 不变。
`D_off=D_total-C_ii` 解析成立，但在几乎 self-only 的例子可能发生严重浮点相消；**必须**逐例做精度审计（独立 direct offdiagonal pair sum 的小 N offline 核对、对称性、min eig、error relative to true correction norm）并标记 machine-scale 不可靠结果。若数值无法稳健，提出/实现稳定的替代并明示额外成本；禁止无说明地 clip 特征值、set floor 或夸大为区间数值证书。不要混用 `C_hat=FFT(C)` 和 `h³` normalization。

### 出图与分析
复用 PR #5 的 141 parameter tables/旧来源数据；至少完成全部 324 条 convergence 和主 winner（xi=1.025,sR=2.85,sF=3,M=15,p=5）、active bad case（kappa=.7,xi=.85,sR=4,sF=4,M=16,p=7）、**self-dominated outlier**（kappa=.7,xi=.85,sR=8,sF=6,M=64,p=9）。根据资源再覆盖 sweeps。
1. 同轴对比 `||ΔΓ||/||Γ_D||`、`Nδ/||Γ_D||`、`2 D_total,max/||Γ_D||`、`2 D_off,max/||Γ_D||`、`B_old/||Γ_D||`、`B_improved/||Γ_D||`，按 p=2/3/5/7/9、M 分图；active/inactive & exact zero 单独标记。
2. 显示 `B_old/||ΔΓ||` vs `B_improved/||ΔΓ||`（仅 true correction-only dense norm，不用历史 full-difference 替代）、`self_degree_trace_fraction` 与 `max λ(Doff)/max λ(Dtotal)`。
3. 单独图展示 winner 与 bad case：`E_old_measured`、`E_new_measured`、`E_proj`、old empirical envelope、new empirical envelope、目标 1e-4；列出 `envelope<=1e-4` 的配置 9 个中有多少，及实际 pass。不要将 `E_old_measured+B_improved_rel` 称为 certified total error，除非另有 rigorously certified epsilon_old。解释原 5/9 是否改善。
4. 新增 bound-only setup 性能（parameter and per-config），不得与 PR #4 的旧 cached matvec 时序拼接宣称提速。

PNG + SVG + per-plot source CSV + 中文 report/proof + unit tests + independent audit + reproducible commands。不得只画图而不检查真值与数值相消；所有 failed/unavailable 明确报告，不伪造结果。

### 研究解释（请在报告强调）
这个上界是 **sufficient / conservative**；它不通过 **不能推出实际误差不满足**。选参应区分“theory-certified candidates”（需要完整 `ε_old^cert+B_improved`，目前 ε_old^cert 未就绪）与“independent-reference empirical-pass candidates”，不能因 loose bound 自动淘汰后者。PSD 对所有合法参数已由结构保证，bound 只控制离散误差，不在每次 Lanczos matvec 中计算。我们的目标是在等精度验证基础上挑成本最低的参数，而不是强制所有实测达标候选都通过保守 bound。
