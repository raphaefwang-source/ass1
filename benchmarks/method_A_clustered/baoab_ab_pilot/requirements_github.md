@codex **执行范围已缩减：本次只实现并实跑 Stage 0 + Stage 1（本地正确性验证 + moving-particle BAOAB pilot）；严格禁止启动 Stage 2/3、HPC、16-replica 四步长正式研究和长轨迹。此版本完全替代本评论原来的执行要求。**

## 目标和当前基线
- 从 PR #9 `codex/optimized-pairwise-ou` 的 head `d6a8d208bcdc30731079715dc4c8e925b4fb2f8b` 开始，检查 AGENTS.md/当前 HEAD。复用 PR #8 完整 Γh Lanczos–OU (Combination B) 和 PR #9 已编译顺序 Pairwise + long Lanczos–OU (Combination A)；旧的 `ewald_longitudinal/lj075_*` 和历史 commit `27789f6cc032509337f3951ed42b1eb6135ee576` 只能作为 LJ/BAOAB 模型参考，不能混用旧摩擦算子。
- 新建独立分支、新目录例如 `benchmarks/method_A_clustered/baoab_ab_pilot/` 和 Draft PR；不得覆盖 PR #3–#9 任一原始测试/结果，不 merge，不 force push。实际完成的测试/轨迹/时钟须保存原始证据和 SHA；资源不足标明 N/A、不填预测数字。
- **本轮不做 A/B 最终胜负结论**。只回答：在真实运动粒子下两种方法能否正确跑、每步动态重新构建数据有多贵、A/B end-to-end 每步成本各是多少、正式实验的瓶颈和资源量。

## 物理设置（Overleaf 主稿 §7）
- `N=256, rho=0.75, L=(256/.75)^(1/3)≈6.98864`，3D 周期，`m=sigma=epsilon_LJ=kBT=1`，初始总动量为零。
- Candidate 1 S1L 摩擦核：`K(x)=c*exp(-kappa*r)/r * (xx^T/r^2)`；`gamma=0.5,kappa=0.7,r_ref=1.3,c=gamma*r_ref*exp(kappa*r_ref)`（计算精确表达式，约1.6148）。不得用历史 PR7/9 的 `kappa=.2,c=1` 替代。
- LJ 保守势：`u(r)=4*(r^-12-r^-6)*S(r)`, `S=1-10z^3+15z^4-6z^5, z=clip((r-2)/(.5),0,1)`; `r_on=2,r_cut=2.5`。force 必须包含 switch derivative；minimum image 仅用于 LJ conservative forces，摩擦保留全部贡献周期镜像；保留物理 Fourier k=0、完整 mesh self cancellation、不将 i=j 自镜像变成 graph 边。
- 注意 Overleaf 的 `Kernel A` 指物理核，非 thermostats 的 Combination A。

## Stage 0 — 先实现、测试并通过
1. 实现 Overleaf 技术文档 §13.1 Eq.(66–70) 的实际 BAOAB：`p_half=p-dt/2 gradU(q)`; `q_mid=q+dt/(2m)*p_half`; 在 `q_mid` 构建 `Gamma_h(q_mid)`，于整个 `O_dt` 冻结 q；`q_next=q_mid+dt/(2m)*p_after`; `p_next=p_after-dt/2 gradU(q_next)`。周期包裹与 unwrapped q 分离；禁止把 q_mid 上的算子沿用到下一时间步。
2. Grid PSD 采用**同一套**对 A/B 完全相同的 `Gamma_h(q)=GammaS+GammaL`。Overleaf 旧 PPPM 起点 `xi=.85,s_R=s_F=4.1,M=24,p=8` 仅是候选，**不是**新 Grid PSD 已认证参数；请在 `kappa=.7,c≈1.6148` 下用独立完整周期 direct reference 于典型 q 检验原稿空间预算 `relative spectral error <=2e-7`、对称/PSD/COM、FFT action 和 full/near/long 一致性。优先找到合格可行设置，而不是完成大规模缓存 Gamma-v 最快参数搜索。若本地实际无合格设置，明确 `strict Overleaf budget NOT MET`，可另以实际验证的 `1e-4` 空间误差开展标注为 **exploratory-only** 的 pilot；不可冒称严格成功、不能进入正式精度断言。
3. FFT/kernel 参数表（只依赖物理参数和盒子）跨步复用；q 相关近场边、long degree/assignment、A 的 3×3 edge eigensystem 和 coefficients 在每步 q_mid 正确更新。由于旧参数 `r_c>L/2` 可能有多周期镜像，通用完整 3×3 张量不可简化成 rank-one。
4. 势/力有限差分、switch continuity、periodic forces、动量守恒、无虚构 self edges、给定同一 q 的 A/B 完整 graph equality、密集矩阵函数对照、A 近前/近后/long 独立 Gaussian、finite Lanczos noise/damp 真实误差、每步数据正确失效/重建与无 NaN 等测试。负 Ritz 出现须中止并诊断；不默默 clip 非微小负值。不把固定冻结 OU FDT 当作全程 Gibbs 证明。
5. 本地实际选择 `r_noise, r_damp`：旧论文 B 14/5 只作初值，分别针对新 B/full 和 A/long 在实际 q 和 `dt=.005` 验证 slow/fast/random/actual-input/低波数 L/T 等方向，优先执行严格 action 阈值 `<=2e-8`（原稿空间预算的十分之一），如不能做到报告误差/增加 rank。A 从1个内部 splitting step 开始，精度不足时显式报告 splitting 与 Krylov 的不同误差；不要强制两方法相同 rank。

## Stage 1 — 仅跑本地短轨迹 / 性能 pilot
- 参数 `N=256, rho=.75, dt=.005`；**3 个相互独立、带不同随机 seed 的 canonical/pre-equilibrated 初态**，初始化按 Overleaf §7：FCC 或 RSA，使用 scalar Langevin+LJ 保守力预平衡约30–50时间单位（若已有经过证实的 canonical starts 可有证据地复用），总动量设为0；初态准备时间**单列**，不可算进 BAOAB 生产单步。
- 对同一每个初态，**A 和 B 各跑 200 个真正移动粒子、逐步 q_mid 更新的 BAOAB 步**（共 `3 initial states × 2 methods × 200 steps = 1200 production steps`，若没有完成，请给真实数目）；时间总长每轨迹为1，这是**性能/正确性 pilot**，不是 Overleaf 正式 16-replica、T=10 的动力学收敛研究。
- 对相同初态、同一物理核与共同 spatial 参数，两方法分别使用各自已验证的 noise/damp rank；Gaussian 源必须保证各方法内部正确独立。记录至少 `Tkin,U/N,total momentum,min_pair_distance,q motion,real r_d/r_n,projection_active,spatial accuracy on sampled visited states`、异常/失败；保留每方法实际种子、环境及随机序列审计。
- **逐步分相 + 完整 wall time**：LJ forces、FFT table 是否复用、near 邻居/全镜像边 setup、mesh config/degree/assignment、A 每步 edge eigendecomposition、A coefficient cache setup、compiled pair forward/reverse sweeps、long/full Lanczos damp/noise、FFT、重正交和 RNG、完整 BAOAB wall；peak memory。绝不把 cached frozen OU/跨 batch timing 当成动态性能，不能把共享 setup 重复相加。
- 额外依据 Overleaf 原 §7.3 尝试**同一机器一线程、6 interleaved rounds × 40 真正 moving steps** 的 A/B 同批计时；如果预算有限可降低并明示。此独立 paired timing 不得复用相同 q 的 freeze/replay 结果冒充动态，每轮 A/B 使用可比的初态状态或经解释的控制，完整包括 q-dependent setup。表面 `T_BAOAB_A / T_BAOAB_B` 是探索性性能差异，**不是最终等动力学精度排名**。

## 本轮明确禁止
- **禁止 Stage 2**：不运行 `16 independent canonical starts × four dt × A/B × T_end=10`；不跑 Overleaf 8条长平衡、11条 T=100 长链，亦不开展 VACF/MSD/RDF 的正式时间步 1% 等价性/CI 判定。
- **禁止 Stage 3**：不作生产级 BAOAB 联合参数优化、最终 A/B winner、额外 HPC 提交、MPI scaling、C/D/E 开发；不要使用 `dt=.005` 作为已严格验证时间步。
- 不补填未运行的任务；明确区分 Stage0 空间/算子精度、Stage1 轨迹正确性、Stage1 动态性能，和 **未完成的** 长期统计/同动力学精度选择。

## 本轮交付/停止线
新源码、实际 tests/logs、3×A/B ×200步 trajectory evidence、RNG/输入哈希、每步和分相时间 CSV、初态生成记录、中文 `report.md`、复现命令、动态 setup / 整步 A/B 两张以上 PNG+SVG、独立审计、独立 **draft PR**。完成 Stage 0+Stage 1 就立即停止；报告最真实的 pilot 发现、误差局限和正式 16-replica+HPC 的资源预算（只估算，不执行）。
