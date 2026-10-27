Source: https://github.com/raphaefwang-source/ass1/pull/7#issuecomment-6451080392

@codex implement **Fast DPD 第三项的 Task 1 + Task 2 + Task 3，并实际在可用计算环境跑完本地小规模测试、出图和报告**。这不是 code review，也不是只生成计划；请开发代码、运行测试/实验，给出可复现的真实结果。**先不使用 HPC，不跑长时间 LJ 生产轨迹。**

## 范围、版本和不变量
- 目标 repo：`raphaefwang-source/ass1`，以当前 PR #7 的 head `76ceb517fc24f60dfdf6d9d2b4af7a2109a07e54`（`codex/final-exp2-gridpsd-selection`）作为只读基线；先检查最新 HEAD、`AGENTS.md`、原代码与环境。请新建独立分支/目录 `benchmarks/method_A_clustered/thermostat_ab_local/`，开独立 **draft PR**，不能改写 PR #7 或 PR #3–#6 的结果，不能 force push、merge、覆盖旧数据。
- 只研究 **Candidate 1 / S1L** 物理核。复用当前 `deconv_psd_comparison/grid_psd.py` 和 PR #7 各目标 `epsilon=1e-3,1e-4,1e-5` 的已测最优 **Grid PSD** 空间参数；首要测试以 `N=256,rho=.75,kappa=.2,amplitude=1,epsilon=1e-4,xi=1.24875,sR=2.9,sF=2.9,M=20,p=4` 作为空间基线，可按同密度盒子尺度为 N=2/8/16/64/128 构造各自合法网格并独立确认精度。优先复用九配置 uniform/moderate/strong × 3 seeds。所有 A/B 实验对**完全相同的离散矩阵** `Gamma_h=Gamma_h^S+Gamma_h^L`、同一配置及物理条件，避免使用不同 spatial discretization 伪造 A/B 速度差。周期 near 要保留多镜像、真实 k=0、完整 self cancellation。
- 历史代码可参考 Git 历史 commit `27789f6cc032509337f3951ed42b1eb6135ee576` 中 `ewald_longitudinal/test_lanczos_fdt.py`、`test_true_dynamics.py`、`toy_dynamics.py`、`sec6_common.py`、`sec6_dynamics.py`（旧版本可能已被当前 cleanup 移除）。只按模块复用，经测试适配新版 Grid PSD，不要直接将旧 Γ/PARAMS 混入。
- 数学依据为用户的 22 页 technical draft §5–§7 / §11.2–11.3 / §13.3。冻结热浴（`theta=kBT`）：
  `p_plus = p_bar + exp(-dt*Gamma_h/m)*(p_minus-p_bar) + [m*theta*(I-exp(-2dt*Gamma_h/m))]^(1/2)*Pi*z`，
  `Pi=(I-11^T/N) kron I3`。噪声与 drift 必须使用同一个**已投影的** Gamma，保留 COM，禁止仅剪 noise Ritz 而不更新 damping；有限 Lanczos 不能自称精确 FDT。

## Task 1 — 实现/迁移 Combination B（完整 Γh Lanczos–OU）
1. 将已有 full-graph two-call Lanczos/damping+noise 适配为直接调用新的 Grid PSD 完整 `Gamma_h v`，输入 frozen q,p,独立 Gaussian; 支持显式 momentum projection、robust expm1、正交化、阶数可控、negative Ritz numerical diagnostics。不能额外施加 `delta_h Pi`，不能在 Fourier multiplier 上平方根。
2. N=2/8/16/64（资源允许 N=128/256）用**同一个离散 Gamma_h** 独立 assemble Dense Reference，精确 eigendecomposition 比较 `exp(-dt Gamma/m)p`、`f(Gamma)z`、完整 OU 的条件均值/条件 covariance、动量零模以及方向协方差。分别报告 noise rank 与 damping rank 的误差、曲线及经过测试的 stopping diagnostics；固定 seed、正反向同源耦合减少 MC 误差。
3. 对 frozen OU 的 FDT 检验严格区分 **exact dense algebraic identity**、finite-Lanczos error 与 finite-sample Monte Carlo SE/CI；不凭单个 mean temperature 宣称 FDT 完全验证。

## Task 2 — 实现 Combination A（近场逐对精确 OU + 远场 ΓL Lanczos–OU）
1. 从生产近场摩擦图读取每个无向 pair `i<j` 的有效 `K_e^S = sum_all_retained_periodic_images K_S(q_i-q_j+nL) \succeq0`（用已有非负 cutoff/taper），严格与 B 里的 `Gamma_h^S` **逐项相同**。若多镜像求和导致满秩，用对称 3x3 eigendecomposition，不要假设所有边 rank-one；缓存边分解，但 configuration 改变要重建。
2. 对相对动量 `r=p_i-p_j` 的长度 `tau` 的 exact edge OU：
   `r_plus=exp(-2*tau*K_e/m)r + [2*m*theta*(I-exp(-4*tau*K_e/m))]^(1/2)*eta_e`；
   `p_i += (r_plus-r)/2; p_j -= (r_plus-r)/2`。用 `-expm1` 评价微小 noise variance，零特征方向噪声为零，验证单边条件均值、协方差、总动量与 Maxwell relative variance `2m theta I`。
3. 一次 frozen A 步采用对称前向半步 `e1..eE (dt/2)`，中间长程 `Gamma_h^L` 的 two-call Lanczos OU `dt`，反向半步 `eE..e1 (dt/2)`；所有 edge OU 调用与 long Gaussian 彼此独立，禁止 reuse 前后边 sweep 噪声。远场 Lanczos **仅**调用 `Gamma_h^L v`，near 不进入 Krylov；保证两组相同的总离散 graph，并明确 frozen Gibbs/Maxwell invariance 只对理想子步成立，A 有非对易 dynamical splitting error。

## Task 3 — 本地、公平的 A/B 准确性与成本比较（不做 HPC）
- 小 N 多种 frozen q，优先 `N=16,64,128,256` 和已有九配置（不可负担的范围明确 N/A）。保存相同 q、同一 `Gamma^S/Gamma^L`、可耦合的输入 p/z，物理 dt 及 dimensionless `dt*lambda_max/m`（如 0.1/0.5/1/2）。
- 对比 (i) 无噪声条件均值/非平衡慢模态与快模态衰减 vs dense **完整 Γ** exact OU；(ii) conditional covariance、定向方差、零模、Maxwell equilibrium 与更高 moments（MC uncertainty 独立报告）；(iii) A 的 `dt,dt/2,dt/4` internal splitting error 对比 B exact-frozen，验证观察到的收敛而非凭形式对称性直接宣称阶数；(iv) 分别调 `r_d,r_n` 到同等 Krylov/FDT/dynamical tolerance，不能强制相同 rank 以作公平速度比较。
- 计算成本：`t_table_setup,t_config_setup,t_near_edge_eigh,t_pairOU_forward,t_pairOU_reverse,t_long_lanczos_noise,t_long_lanczos_damp,t_full_lanczos_noise,t_full_lanczos_damp,matvec_counts,FFT,near_actions,reorth,t_total_frozen_step`。固定 same batch 交错重复、单线程请求/披露实际 native pool 与负载。报告 **等冻结动力学精度**下总 wall cost，不仅是 FFT matvec。不要把历史 PR #7 的 cached Γv 计时与新 batch 混合声称加速。
- 若条件 covariance MC 完整谱矩阵太贵，可采用 dense small-N 完整检查，以及中等 N 少量预注册方向（slow/fast/random）和明确的 confidence intervals。拒绝把有限样本通过当作严格 FDT proof。禁止伪造未运行的结果。
- 输出：独立脚本与测试、随机种子/配置及环境 provenance、实际日志、CSV/JSON、论文可用 PNG+SVG（matrix function rank-vs-error、frozen FDT directional variance、splitting dt convergence、equal-accuracy A/B total time）、中文 `report.md`、数学/误差说明、复现命令、最终 audit。提交独立分支与 Draft PR。
- **Stop line:** Task 1/2/3 到 frozen thermostat + small local benchmarks 为止；不跑长时间 BAOAB、LJ 多轨迹、HPC，别改 Candidate 2/C/D/E 或 PSD projection 算法。任务很重，可先确保正确性测试再追加覆盖，资源不够时明确标注未完成项，不可用猜测数字替代。

请真的开始执行 Task 1–3；完成后在新 PR 中总结通过/失败、耗时和证据链接。
