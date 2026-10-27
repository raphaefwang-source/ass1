## Codex follow-up: Grid PSD Projection 的严格误差上界与可视化（先画图，不开发新方案）

请在本地 Codex Mac app 中按此任务实际实现并运行，**不是只给 review**。起点：PR #4、分支 `codex/deconv-grid-psd-comparison`、当前 head `1b9997a4efd10d592ae9cff2c50784c8f83bbd28`。先检查实际 HEAD/AGENTS.md 和是否有别人后续修改，勿覆盖旧实验/CSV/图。新工作放独立目录 `benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds/`；如有新提交，独立分支和 draft PR，勿 merge/force-push。**这轮仅做 bounds 的验证、绘图和简短分析，不要改 Grid PSD 投影、选参、OU 或实现复杂 SDP。**

### 1. 严格定义并实现两个 PSD projection correction bound

在精确算术中，先取 (S_d=[T_d^{raw}+T_{-d}^{raw}]/2)（六个分量保证每块对称），令 (C_d=( - S_d)_+ = T_d^+ - S_d \succeq0)，(\delta=\max_d\|C_d\|_2=\max_d[-\lambda_{min}(S_d)]_+\)。按当前 Mesh 中 `IFFT(G)/h^3` 的物理表规范及现有 `grid_psd.py` 定义。

设非负且总和为 1 的 stencil 权重 (w_{ig})，(C_{ij}=\sum_{g,l}w_{ig}C_{g-l}w_{jl}\succeq0)，(ΔΓ=\operatorname{Laplacian}(C_{ij})\succeq0)。于是**对任意粒子配置**有
`B_simple=N*delta`
（N>=2；由 sum_{i<j} ||v_i-v_j||² = N sum_i||v_i||² - ||sum_i v_i||²）。
算更紧、依赖当前配置但不需形成 3N×3N 稠密矩阵的
`n_g=sum_j w_jg`，
`D_i^C=sum_g w_ig sum_l C_(g-l) n_l`（可以包含 self，应保持 PSD），
`B_degree=2*max_i lambda_max(D_i^C)`，
`B=min(B_simple,B_degree)`。
精确算术严格满足 `||Delta Gamma||_2 <= B`。优先直接从 correction table / correction influence 算 degree；不要从两个大的完整 degree 相减带来不必要的灾难性相消。需要说明浮点 FFT 的舍入误差，数学证书与实数 float64 诊断分开。如原始 raw/inversion 对称化引入微小改动，记录 defect 与比较残差，别把 `C_d` 的上界直接错用于非对称 raw 算子。

### 2. 在已有数据上核对，并按需要补算，不要虚构

复用 PR #4 的 `results/convergence.csv`、`validation.csv`、`sweeps.csv`、`xi_sweep.csv`、`best_projection_diagnostics.csv`、`final_timing.csv`、共享独立 direct reference，优先覆盖 `convergence` 的 p=2,3,5,7,9 / 多 M / 九配置，以及 Experiment 2 主 1e-4 winner、投影明显活跃的非 winner。已有 `E_proj=||Gamma_gridPSD-Gamma_original||_2/||Gamma_D||_2` 仍以完整矩阵或独立 direct stencil 检验为准（N=256 可用 dense 离线核查）；真正的 `||Delta Gamma||` 也可以通过 correction-only graph 单独稠密组装交叉验证。

检验每条数据 `E_proj <= B/||Gamma_D||`（允许另行报告 machine-scale roundoff violations 而不是篡改 bounds）；分别保存 `delta, B_simple, B_degree, B, E_proj, normGammaD, active, negative_blocks, correction Fro, min_raw_eig, E_old, E_new`，和 bound tightness (B/||Delta Gamma||_2)，真实 0 单独标记。

**非常重要：** (E_old) 若来自独立数值 reference 的实测比较，只能画作“empirical error”，不能冒充严格 `epsilon_old`。可画 `E_old_measured + B/||Gamma_D||` 作为相对于该数值参考的三角不等式 envelope，并准确命名；如果要写完全 certified total bound，必须已有另行认证的 `epsilon_old`，否则标 unavailable。不要声称投影活跃即高阶，也不要把 mixed-transition 的 12/26 的 formal slopes 作为收敛阶。

### 3. 至少画三组论文级图（PNG + SVG + source CSV + plot script）

1. **核心图：** 同一 (p) 下随 (M)（或 (h)）的曲线 (E_old), (E_new), (E_proj), (B_simple/||Gamma_D||), (B_degree/||Gamma_D||), (B/||Gamma_D||)，分 p 子图（建议 p=3,5,7,9），标明 active/inactive 区和 zero floor；log-y 并清楚标注 mesh-only 与 complete operator，不要混为一谈。
2. **上界紧度与负块图：** (B/||Delta Gamma||_2)（大于等于 1），(\delta)，负块数/比例，横轴 M，按 p 或参数族分面，显示上界是否紧到有选参价值。
3. **参数与成本：** 已有不同 xi/sF/p/rho/kappa 的 active/inactive 下误差/界对比，以及主 1e-4 winner 与典型 active bad case 的 (E_proj)、(E_new)、bound 和 setup 开销。新增 bound 的 setup 成本单独计时，保留原 cached matvec 不变；不要与旧计时混批声称加速。

所有图坐标和相同数据 ID 保持一致，包含九配置 seed/family 范围并说明聚合方式。若组合过密可用多面板而不是强塞同一图。输出中文 `report.md`、数学 `proof.md`、`README.md` 和可恢复命令、原始数据、单元测试。对 `N\delta`、`2Dmax`、合并 bound、self cancellation、奇偶 M、零模、symmetric/PSD correction 做独立测试，最终报告明确记录 pass / failures。

### 4. 本轮科研目的
我们要回答：在投影活跃时，上界有多保守？什么时候足以给实际参数选择提供误差预算？同时保持“任意合法参数 PSD”的结构结论。不开发依赖 M=15 的 special-case 修补。下一步我们会单独研究从结构上构造保 PSD 的高阶离散，而不是这轮提前实现新方法。

请完成代码、真正绘图、测试、提交和 PR 后，在该 PR 中给出关键图、CSV、运行命令及结论。
