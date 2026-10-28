@codex 请**实际实现并运行**基于 PR #10 的下一项有限范围实验：**Combination A 与 B 的“matched frozen-OU accuracy + fully dynamic BAOAB wall cost”对比**。此次目标是让 A 通过增加内部 splitting substeps（以及必要的 Lanczos rank）达到与完整 B 相同的已声明**冻结 OU 精度**，然后实测 A 是否仍比 B 快。不要只给 k^-2 外推或计划；**先验证精度再计时**，保留原始结果并报告真实胜负与不确定性。

### 基线、实验范围及物理参数
- Repo `raphaefwang-source/ass1`，从 PR #10 当前 head `84174ae0d11e3f5826cda000c688d147a1b56241`、branch `codex/baoab-ab-stage01` 创建**独立新分支 + draft PR + 全新独立实验目录**，例如 `benchmarks/method_A_clustered/baoab_ab_matched_accuracy/`。检查现有 HEAD/AGENTS.md；所有 PR #3–#10 已有代码、数据、CSV/NPZ、图、测试和 evidence SHA 保持不变，绝不覆盖或 force-push/merge。
- **严格保持 PR #10 Overleaf LJ 物理设置：** `N=256, rho=.75, L=(256/.75)^(1/3), m=sigma=epsilon_LJ=kBT=1, gamma=.5, kappa=.7, r_ref=1.3, c=.5*1.3*exp(.7*1.3)`；smooth switched LJ `r_on=2,r_cut=2.5` 并保留 switch derivative；全部摩擦周期镜像、物理 k=0、self cancellation、COM=0。固定通过前轮验证的**同一套 Grid PSD 数值参数** `xi=.65,sR=sF=4.7,M=40,p=10`，不重新搜索 Γv 最优值；固定 initial `dt=.005`，A/B 完全相同的离散 `GammaS+GammaL`、相同 q_mid。
- 复用 PR #9 C sequential 3x3 pair OU、PR #8 B/full 与 A/long 两-call Lanczos、PR #10 `dynamics.py` 的真正移动粒子 BAOAB、3 个已保存并可 hash-check 的独立预平衡 q/p；除非必要，不重复成本很高的 canonical preparation。此次**不 HPC，不跑 Stage2 16-replica×4-dt、长 LJ 链，也不宣称完成完整动力学等价性**。

### 1. 先修复 A 动力学接口：真正可变 substeps
- 检查 PR #10 的 `PilotSystem.step` 是否硬编码 `substeps=1`。实现可配置 `k=1,2,4,8,16,32`，需要时继续到64（须记录额外成本）。**每个 BAOAB 中点 q_mid 只构建一次 Gamma_h**，在同一个冻结图上重复 k 次对称小 OU：
  `for l in range(k): near_forward(dt/(2k), independent eta_f[l]); long_OU(dt/k, independent z_l[l]); near_reverse(dt/(2k), independent eta_r[l])`；
  每个子步 3 组新的独立 Gaussian，噪声大小不手工按 sqrt(dt) 再重复缩放（exact OU 本身已经包含 dt/k）；在完整 h 内保存总动量，保持相同有序 image-summed pair tensor、同一已投影 long ΓL 与外层 BAOAB。严禁把“只把 pair sweeps 细分、long OU 仍整步”冒充完整 A 子循环。
- A coefficient cache key 区分 `dt/(2k),mass,kBT`；每步位置更新使图/eigh/coeff 重建，FFT/kernel table 位置不变部分只做合法复用。报 setup 1次 vs k次、k 倍 long Lanczos、边OU/RNG 的实际细项；不漏计/重复计。B 保持单次 full-Γ OU，不能为了公平强迫 B 有 k 个 OU 子步。
- 需验证 k=1 与 PR #10 完全相同且固定 q、同源 p/Gaussian 下与 PR #8/9 的实现吻合，k>1 用独立 reference 验证 Gaussian indexing/COM。

### 2. 精度：区分严格“冻结 OU 精度匹配”和“完整动力学精度匹配”
- 在相同 q 的完整离散 Γ=ΓS+ΓL、固定 `dt=.005` 下，**目标参考是 B 的 independent dense exact full-Γ OU**。对理想 A(k) 单次完整冻步的均值映射 `R_A(k)`、条件协方差 `C_A(k)`，以及 B exact `R_B=exp(-dt*Γ/m)`、`C_B=m*kBT*(Π-R_B Π R_B^T)`，用可审计的范数误差分别计量。均值与协方差要分别报告，不能把 A 的 ideal FDT invariance 误当作 full-B conditional covariance 一致。采用写清分母的 normalized operator 2-norm：`e_mean=||R_A-R_B||_2 / max(||R_B Π||_2,small)`、`e_cov=||C_A-C_B||_2 / max(||C_B||_2,small)`，在 COM 子空间比较；记录绝对范数、相对范数和近零处理规则。有限 Lanczos 对这些理想映射的非线性输入相关误差另外量化，不可用逐列近似 F 拼成固定 covariance 后声称 exact FDT。
- 注册**三档精度目标**：`epsilon_frozen=1e-4,1e-6,2e-8`，对每档选最小经验证通过的 k；严档2e-8是本轮 stress test / 与此前 action budget 同数量级的严格 frozen comparison，**不是 Overleaf 完整 BAOAB 物理观测误差必需达到的条件**。不因某档未达标而伪称 A 失效；记录 N/A。
- 覆盖 PR #10 三个独立初始态以及真实 trajectory 中至少 6 个保存的 q_mid，优先加上 minimum separation/λmax 极值访问态。检验 k=1,2,4,8,16,32/64 误差对 k 的斜率与观察的约 k^-2 是否成立；不凭拟合外推代替真实通过。可在小 N 做 independent full-matrix dense ground truth，同时 N256 尽可能对完整 COM 子空间独立计算准确算子误差；若 CPU/内存导致只能做探针而不能完整谱范数，则明确标签 sampled-probe，而不能宣称严格 operator-norm pass。
- 各个 A 子步的 `r_d^A,r_n^A` 应独立选/验证；B `r_d^B,r_n^B` 单独验证。在实际 p、Gaussians、slow/fast/random/低 k 波模等方向对 dense two matrix functions 及实际完整随机步路径对照；有限 rank budget 必须随 k 的组合误差重新检查，不能因 PR10 k=1 使用14/5 就直接套用到 k=16。必要时增 rank，并为每档保存实际计数、matvec、误差、FDT directional covariance MC SE/CI；使用独立的真实 OU 噪声（各 substeps），正反近场噪声不重用。B 的原有 action rank 验证也要在同批次复查。
- **禁止**把 A 同一自然随机路径与 B 不同结构 Gaussian 的 raw output 距离当作算法偏差；比较解析/独立 dense 条件分布、真实 Monte Carlo 的 means/cov/SE，以及各方法自己的精确基准。A 理想 frozen Maxwell 保持不等于 full-B transition kernel 一致，也不等于运动粒子 Gibbs 时间误差已验证。
- 特别区分 `2e-7` 空间精度约束、`2e-8` 以前对应的 Lanczos action 目标、此次人为设置的 frozen transition `2e-8` 压力测试阈值，三者绝不混淆。

### 3. 真正重新计时，回答“同冻结 OU 精度时 A 还快不快？”
- **对每一个精度档**先确定满足该档的 k 和所需 r_d/r_n；随后在同样真实移动的 `N=256,dt=.005` LJ 轨迹下进行 **A(k) vs B** 的同批次 interleaved wall-time 实测，不能只按 PR10 的 154.7ms × k 做预测。
- 在同一 3 个预平衡初态和相同机器、线程请求下进行适量重复，例如每个 k 和 B 至少各 `3 starts × 20–40 moving steps`，并交错/轮换顺序和记录 wall load；资源足够可做原稿式 `6 rounds × 40 steps`，但若严档 k16+ 很贵可缩小样本、明确 CI 范围。选择的 A k、额外 rank、B rank、所有独立随机流都在实际 running O-step 中启用；不能先验证高 k，再偷偷用 k=1 做计时。需要保存每步/每round真实 `q_mid, dt, selected k, r_n/r_d, full BAOAB wall, q-dependent setup, per k long OU, pair sweeps, FFT/reorth, RNG, LJ force, peak memory`。
- 报每档 A/B 的 paired time ratio 及合理 uncertainty（配对round/bootstrap及样本单位），并单列所有 k 与原 k1 的成本曲线。真实状态温度/力/动量的短轨迹 sanity 必须过关。**优先完整跑出至少一组严档 k 与 B 的 actual dynamic timing**；若已看到速度劣势也不得用估计数字替代实测。
- 在同一 dt、相同空间算子、同一已声明冻步误差阈值下，明确结论 `A_faster/B_faster/INCONCLUSIVE`；如果某档 A 无法满足严档，则 `accuracy_not_met`，不把未达标配置纳入平等比较。所有结果均限定 **frozen-transition-accuracy-matched dynamic cost**，绝不偷换成已验证的 **equal physical-observable accuracy**。
- 若本地余量允许，可选 `dt=.0025` 做很短的 moving-particle probe，以了解 k/精度/速度关系，但**不要启动 Stage 2 正式跨dt 16-replica动力学收敛**。最终 equal dynamical accuracy（VACF/MSD/pressure/RDF 95% CI）仍须下一阶段 HPC。

### 4. 交付与停止线
- 源码、Regression+new tests、每条原始 CSV/JSON/NPZ、seed、dense refs 与验证 domain、rank/substep 选择记录、pilot outputs、CPU/per-phase setup + step raw wall、3张以上PNG/SVG（`e_mean/e_cov vs k`，`cost vs k`，`matched-accuracy A/B dynamic total`）、中文报告、复现命令及独立 audit；新分支/新 draft PR。准确列明 k=1 时 PR #10 A=397.96ms,B=515.77ms **只是旧 batch 背景**, 不能和新 batch 混算加速比。
- **只做本地 matched-accuracy moving-particle pilot，不 HPC/SSH、长轨迹、16 replica、联合 Grid PSD 参数最优搜索、C/D/E。** 完成后先给 `A 是否在同冻结 OU 精度下仍快于 B` 的可复现证据，然后写明“完整 BAOAB 等动力学精度 A/B 选型仍未完结”及下一阶段建议。

**请实际开始执行，而不只是回复任务计划；如果当前 @codex GitHub 环境未连接，明确提示用户在已配置的本地 Codex session 中执行这条任务，不要声称已运行。**
