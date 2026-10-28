@codex **请从 PR #11 实际开展新一轮本地 A/B 联合选参实验，而不是继续只调 A 的 OU 子步数 k。** 目标：在同一 Overleaf LJ 物理模型和相同准确性约束下，**分别优化 Combination A 与 B 的空间 Ewald/Grid PSD、Lanczos、A 的 substeps**，以真正移动粒子的 **完整 BAOAB wall time / simulated physical time** 为性能目标，判断 PR #11 中“严格冻结精度下 B 更快”的结果是否只是固定参数所致。必须实现、实跑、保存原始数据；不能只生成方案或速度预测。

## 基线 / 交付原则
- Repo `raphaefwang-source/ass1`，从当前 PR #11 的 `codex/baoab-ab-matched-accuracy` / SHA `891994a02b611833848839b5de9880f9dabc7a1d` 起步；先检查实际 HEAD、AGENTS.md、PR #7 参数选取、PR #10 moving BAOAB、PR #11 complete subcycling 和其原始结果。
- 新建**独立分支 + 独立文件夹**（例如 `benchmarks/method_A_clustered/baoab_ab_joint_selection/`）与独立 Draft PR。绝不覆盖、编辑或误删 PR #3–#11 历史源码、实验/图/CSV/NPZ/manifest；不 merge、不 force push。保存 upstream SHA、测试和可复现命令。仅在**当前本地 Mac** 实跑；**不要提交 HPC/SSH/云端数值计算**。
- 固定物理：`N=256,rho=.75,L=(N/rho)^(1/3),m=sigma=epsilon_LJ=kBT=1,kappa=.7,gamma=.5,r_ref=1.3,c=.5*1.3*exp(.7*1.3)`；3D 周期 Candidate1/S1L `K(x)=c exp(-kappa r)/r (xx^T/r²)`，真 `k=0`、完整周期摩擦镜像与 self cancellation、COM 动量约束。保守 LJ: `r_on=2,r_cut=2.5` quintic switched + switch derivative，LJ minimum image only。不要使用 PR #7 在 `kappa=.2,c=1` 下的最优数字充当本物理参数下的合格解。
- PR #10 旧基线 `xi=.65,sR=sF=4.7,M=40,p=10,dt=.005` 是**已验证可行点**，但不是全局最优；保留为对照。Gamma 随 q 变，必须在每个 BAOAB 的 q_mid 更新近边、assignment、degree、A 的 eigensystem/coefficient；位置无关 FFT/kernel table 可以合法复用。A 内部 k 个 `near_forward(h/(2k))→long_OU(h/k)→near_reverse(h/(2k))` 使用独立 Gaussian，每个外步只 setup 一次；B 一次 full-Gamma OU。近场多镜像产生一般满秩3×3边，不能强制 rank-one。

## 实验目标：两个不同层次，严格区分
1. **局部可执行目标：** 同 `dt=.005` 和指定冻结 OU 精度档，找各自最快的**实际动态 BAOAB**组合。这是本轮真正可完成的目标。
2. **最终科学目标：** 同样的 VACF/MSD/RDF/pressure/温度等物理观测误差下，允许 A/B 各自优化 `dt` 的 `wall time / simulated physical time`。**本轮不得把第一项直接叫做最终 equal-dynamical-accuracy winner**；不启动正式 16-replica×4dt 或长轨迹 HPC。可短测 `dt=.0025` 作为灵敏度记录，但不作完整物理精度认证。

## 第一阶段：空间候选联合搜索，必须独立精度筛选
- 优化变量 **`xi,sR=xi*r_c,sF=k_c/(2xi),M,p`**，不要强制 `sR=sF`，也不要只扫描一个 `xi`。建议从 `xi={.45,.55,.65,.75,.85,1.0}`、`sR,sF≈4.1–5.2`、`M≈32–64`、`p≈6–12` 自适应启动，**不是笛卡尔暴力全扫**；基于 truncation/Nyquist、误差/成本趋势按需扩展，记录失败的合法候选及失败原因。PR10 可行点必须包含。
- 所有候选采用完全相同的 Candidate 1 Grid PSD 路径；逐 setting 构造正确的投影表并记录 `projection_active`、eigenclip/修正影响、PSD/COM，以及近远场/mesh 全算子分解。无需人为把 `xi` 变化解释为“必然更少 splitting”；必须实际测，包括 near-edge 间不对易带来的误差。
- **共同空间硬约束：** 对与真实周期 unsplit `Gamma_ref` 独立对照，至少在 3 个以前实际预平衡初态及代表性 visited q_mid 通过保守 `||Gamma_h-Gamma_ref||2/||Gamma_ref||2 <= 2e-7`，对数值 direct radius/tail error 做已验证的 conservative allowance；不许拿对旧 splitting 参考或仅 Γv 少数方向作为通过证据。先对便宜 action/小样本误差做筛选，再对 finalist 在更多已注册几何做完整 765 维 COM 算子核查；空间不通过的候选不能参与等精度 speed 表。若可行候选稀少，明确记录并自适应扩展，不用不合格点凑数量。
- 必须区分 `r_c=sR/xi` 导致的多镜像近边成本、mesh `M,p` 与 Nyquist 约束；合法参数不同可能导致 A/B 最优点不同，允许**独立选参**，但都必须满足同一个真实物理核与空间预算。既输出共同空间设置的 controlled A/B baseline，也输出独立选参结果。
- 搜索规模实际可控：先自适应 cheap screening（例如不超过约60–100个独特候选），只对约8–15个有希望的候选进行昂贵的多配置 full-operator 验证，优先挑选 3–5 个可靠空间设置进入真实动态 benchmark；可因本地预算缩小并保留清晰筛选 provenance。不得悄然把大量 dense eigensolves、matrix assembly、验证计入正式每步 BAOAB 时间。

## 第二阶段：每个空间候选联合调 thermostat / 精度（不是只调k）
- **A** 候选联合 `(xi,sR,sF,M,p,k,r_d,r_n)`，`k∈{1,2,4,8,16}`（仅确实必要时 32）；**B** 联合 `(xi,sR,sF,M,p,r_d,r_n)`，不需要 A 型 splitting k。对每个候选的实际 `GammaS(q),GammaL(q)` 分别重新选择有限 Lanczos damping/noise ranks，不能将 PR11 的 `3/8` 与 `5/12` 直接搬用；尽可能寻找满足精度的最小秩。
- **空间约束 + thermo 精度是两回事。** 注册至少 `epsilon_frozen={1e-4,1e-6}` 两档用于真实 equal-frozen-transition-accuracy 选参，`2e-8` 可作为严格 stress tier，但若过贵不可阻塞更有意义的两档。冻结参考应是**共同的独立物理周期 full-Γref exact OU**，同时记录对**各自离散 Γh 的 exact OU**误差，明确拆分空间、split、有限 Lanczos 三类贡献。对 A 计算有序 pair+long 完整理想均值/条件协方差，B 用 full-Γ exact；在 COM 子空间用明确定义的规范化 2-范数，需分别过 mean 与 covariance（不是只过温度/Maxwell steady state），并在至少 3 初态与此前 visited 极值态验证。对于最终最优候选再扩充几何与 actual-input/rng probe；任何声称 full 765维 operator norm pass 的 setting 必须真的计算 full norm，而非 sampled-probe。
- 记录 A 在不同 `xi` 下的实际 splitting coefficient/误差及需要的最小 `k`，明确边间与近远场不对易；也测试 **k=1 是否通过**某档而不是默认 k=4/16。可能存在空间精度通过但 split 不通过的候选，作为失败点保留。
- 每个冻步子循环各使用**独立** near-forward/long/reverse Gaussian，避免重复 noise；用真实随机输入、慢/快谱向量、低波数 L/T 等验证 Lanczos 两个矩阵函数、组合路径、负 Ritz（非微小负值 abort）、动量与路径。不把有限 Lanczos 的输入相关 Gaussian 映射假设为固定线性噪声因子来宣称 exact FDT；严格档若无法用有限MC检验算子级 2e-8，则将其作为独立理想矩阵 benchmark + finite-path evidence，真实统计局限如实列出。
- **不要把 `2e-8` 作为 BAOAB 物理可接受误差的强制要求。** 2e-7 为物理空间 Γ 预算，曾用2e-8为 Lanczos action guard，PR11 2e-8 frozen transition 是压力测试，意义不同。允许 A 与 B 具有不同的合格空间误差，报告 **真实测得的误差值**，最好以 accuracy-cost Pareto 前沿而不是只比较同一个宽松阈值。

## 第三阶段：实跑真正 moving BAOAB end-to-end 性能，选出候选 Pareto
- 基于同一 PR10 已保存并经 SHA 验证的 3 个预平衡 canonical q,p，`dt=.005`，同一物理场与保守势，在同一机器/线程请求下分别进行 A 与 B 的**独立最优选参**、同参数 controlled comparison（区分两种表）。每条实际推进 q,p，不可 freeze/replay 当作 dynamic。
- 对筛选后各精度档的 A/B 头部组合做 interleaved paired timing，例如每组 `3 starts × A/B ×20` 真正移动步骤，最终头部可扩展到 6 paired rounds（各20–40步），使用交错顺序与完整分相时钟；如果候选很多先做较短 exploratory timing，最终 head-to-head 才做可靠配对bootstrap。**实际 cost = 动态每步 q graph/setup + eigensolve/coefficients + near sweeps + long/full Lanczos/FFT + RNG + LJ kicks/drifts + measured integration overhead**；一次性 table/preparation/独立 reference/绘图IO 另列，不漏、不重复加嵌套FFT。
- 不允许把 PR10/11不同批次旧计时直接当成本轮的 ratio；不能以 cached `Gamma v` 最快取代完整 BAOAB 最快。每个 winner 记录 `xi,sR,sF,M,p,k(if A),r_d,r_n,spatial_error,ideal_frozen_mean/cov_error,finite_path_error,T_setup,T_O,T_BAOAB,peakRSS,paired A/B ratio+CI`，原始 CSV、每步 q_mid hash/RNG seed、环境、CPU线程可观测限制全部保存。测速需显式防止模型 q 相关 setup 被意外 cache。
- 输出每档 **A/B 独立选参最优** 和 **共同空间参数的 controlled** 两份 ranking；同时输出主要参数 sensitivity 和 2D/多变量 Pareto（速度-空间误差-冻结误差）。清楚解释 A 是否可以通过换 `xi` 减少所需 k、以及下降的 k 是否足以抵消 A 的 near setup 额外成本。若抽样/CI重叠，标 INCONCLUSIVE，不宣布确定胜负。
- 本轮只需要本地可复现小规模联合优化和真实 A/B Pilot；如资源开销太大，完成 **至少两种实际合格的空间候选**（若找不到须明确报告）和两种 precision tiers 的筛选与顶级动态对比，清晰报告哪些探索未执行。不可为了追求完整搜索虚构实验，也不可未经证明称“全局最快”。

## 交付与停止线
- 新代码、测试/回归、参数与误差扫描 CSV、被拒候选原因、完整 COM mean/cov 验证、rank/子步选择、真实动态/paired timing 原始数据、资源/峰值内存、中文 `report.md` 与复现 README、性能/精度/Pareto 图 PNG+SVG、独立 SHA/审计，以及单独 draft PR。明确本轮结论仅局限于 frozen-accuracy-constrained **moving-particle BAOAB speed**；论文最终的 VACF/MSD/RDF/pressure 95%CI、允许不同 `dt` 下物理等精度和 HPC 正式实验**尚未验证，且本次不要启动**。
- 如果 GitHub 上 `@codex` 由于没有配置 Cloud environment 无法触发实际运行，请明确提示使用已经配置的**本地 Codex session** 执行此任务；**不要声称 Codex 已执行或在后台计算**。
