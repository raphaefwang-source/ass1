@codex 请实际实现并运行下面的后续研究任务，不只是 review 或给建议：增加“去卷积 PPPM 的实空间网格张量 PSD 投影”方案，重做精度扫描和约束选参实验，并对三个方案统一绘图、同机计时比较。

## 0. 范围、基线和交付

以本 PR 的 B2 版本为起点（当前 head `128aaf1cd8bf39c69d2bf2db9619e478fefc0c73`；原 PPPM 基线 `99b416fb8a5eba3d4130f62a39110f6f00197188`）。先检查适用的 AGENTS.md 和实际 HEAD；若已有后续工作，保留并避免覆盖。优先从本 PR head 创建独立分支 `codex/deconv-grid-psd-comparison`，提交新实现、测试、原始数据、图片和中文报告，开 draft PR；不要 merge、force-push 或改写本 PR/默认分支。保持原 PPPM 和 B2 的历史源码、结果、图、reference 缓存不变，新输出放独立目录，例如 `benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/`。缺失的 reference 可独立重建并保存 provenance，不得用待测算法自身作 reference。

三种方法使用稳定名字，避免与 draft 的 Candidate 2 splitting 或 OU Combination A/B 混淆：
- `PPPM-original`：原去卷积 PPPM，作为高精度/速度基准，不宣称任意配置下已有结构 PSD 证明。
- `B2-positive`：现有直接正实空间网格核 + 非负分配，不去卷积。
- `PPPM-grid-PSD`：本任务的新方案，原去卷积网格核在**实空间**逐 3x3 张量 PSD 投影，再 FFT 回去使用。

物理核、Candidate 1 split、amplitude、周期模型、short-range 实现、位置、seed、reference 和误差定义保持一致。这次不做 OU、VACF/VCCF，也不引入全局 delta*P 修正、Fourier 特征值 clipping 或改变物理 kappa 来换速度。不要预设新方案一定高阶或一定快；实验应能得出否定结果。

## 1. 新方案的准确实现

阅读 `ewald_longitudinal/test_parameter_selection.py` 的 Mesh、`benchmarks/method_A_clustered/cached_operator.py`、`positive_mesh.py` 及 B2 的 proof/experiment/timing 脚本。在新 adapter/module 中复用旧路径，避免污染历史实现。

旧频域 influence 是 `G_old(k)=Khat_L(k)/|What_h(k)|^2`，先按原 kc 截断与安全分母规则生成。对称实空间表记为 T。按当前代码中 gather 和 degree 的 `/h^3` 归一化，应当使用：

```
T_raw = irfftn(G_old, s=(M,M,M), spatial_axes) / h**3
S[d] = (T_raw[d] + T_raw[d].T + T_raw[-d] + T_raw[-d].T)/4
lam,Q = eigh(S[d])
T_plus[d] = (Q * maximum(lam,0)) @ Q.T
T_plus[-d] = T_plus[d]
G_plus = h**3 * rfftn(T_plus, spatial_axes)
```

这里每个 d 对应一个实对称 3x3 矩阵，`-d` 是 modular offset。六个独立 tensor components 与 FFT axes 按实际数组布局正确处理。允许选择等价的统一归一化，但必须写清并用“不投影时 round-trip 恢复原算子”的测试验证，不能混用 B2 的无 `/h^3` convention。

重要约束：
1. 投影的是实空间 T，不是 Fourier 乘子；不是逐元素 max(T,0)，也不是整个 3N 矩阵的 eigendecomposition。
2. 所有负 eigenvalues 按定义截为零，零 eigenvalues 保留零；不能添加任意正 floor 或凭经验选择大 diagonal shift。数值容差用于 diagnostics，不偷换数学算子。
3. 相反位移使用同一修复表，保持每块对称和全局 inversion symmetry；奇偶 M 和 Nyquist 均测试。
4. 投影后保留 G_plus 的全部 mesh modes，包括新产生的 modes；不得再施加旧 kc mask，也不得再次去卷积。sF 此时只描述投影前的 Fourier 截断参数。
5. 保留修复表自身产生的零频，不能置零或强制改回解析零频；不能把 T_plus[0] 置零。
6. 用新表重新构造 degree，和 vector response 使用完全相同的离散响应，代数抵消整个 interpolated self response。不能复用旧 degree。
7. spread/gather 使用相同非负、partition-of-unity 的 B-spline 权重。可用 B2 的稳定正权 recurrence，但必须与 old 权重验证一致至舍入误差，说明任何非微小差异。原代码的 p=1 边界问题不能带进新方案。
8. 投影只在参数表 setup 执行；fixed-configuration matvec 不重复投影。参数表可复用，位置改变只重建配置相关数据。

提供结构证明：T_plus[d] PSD + 非负权重 => A_ij^+=sum_g,l w_ig T_plus[g-l] w_jl PSD；配合对称 degree-minus-weight => Gamma_Lh PSD、translation modes 为零。证明是精确算术结论；float64 重建/FFT 的舍入误差单列，不以采样 eigenvalues 代替证明。

## 2. 先验证实现，再运行全实验

测试至少包括：未投影 round-trip 与旧 Mesh pair/action/degree 一致；投影在本来 PSD 的表上是 identity；已知 indefinite 3x3 例子；d/-d、奇偶网格和 Nyquist；非负/归一化 stencil 和 periodic boundaries；保留零频与完整 self cancellation；直接 stencil contraction 的粒子边/图与 FFT matvec 一致；完整算子对称、translation-null、long-only PSD；零 amplitude 与非法输入；旧 PPPM/B2 回归及历史文件 hash。

注意：原 Mesh 的 pair_matrix/mode list 使用原有限 Fourier 支撑，投影后可能新增 modes，不能原样沿用它当新算子的 dense truth。新方案的 dense/direct-stencil 验证必须覆盖实际完整修复表/全部 modes，且与 FFT 路径独立。

## 3. Experiment 1：精度随参数变化，三方比较

沿用既有 accuracy sweeps 的三种配置族 uniform/moderate/strong、seed 1/2/3 和独立 unsplit periodic direct reference。复用所有可核对 hash 的旧准确性数据，不必为了复用而覆盖旧文件。

复现旧六组参数研究：sR、远场/表截断参数、M、p、rho、kappa；按原 baseline/moderate 设置运行新方案。共享参数的比较固定同样的其余参数、核和配置。区分扫描物理参数与优化数值参数；选参实验的 rho/kappa 始终固定。

主误差继续是 `E_G=||Gamma_h-Gamma_D||_2/||Gamma_D||_2`，不是用少量 random probes 代替 operator norm；另报原 Gaussian probes 的 E_GX。对收敛阶研究单独分离 mesh-only 误差，使 short cutoff、Fourier tail、reference/table construction error 不掩盖拟合；不把 cutoff plateau、roundoff floor 或某个恰好 cancellation 的点当作阶数。

额外记录新方案诊断：
- T_raw/S 的最小 eigenvalue，negative tensor/eigenvalue 的数量与比例，区分显著负数与 roundoff 尺度。
- symmetrization defect、各块 negative-part 谱范数、`max_d ||T_plus-S||_2`，必要时其空间分布/分位数。
- 粒子算子实际投影改变量 `E_proj=||Gamma_gridPSD-Gamma_original||_2/||Gamma_D||_2`，和原始误差、修复后总误差并排记录。
- Gamma_Lh 与 complete Gamma_h 的 restricted min eigenvalue、symmetry/translation residual、FFT/direct-stencil discrepancy。
- 每个 p 的 M-refinement/log-log slope（例如 p=2,3,5,7,9，按已有合法参数域；不强行拟合 floor）。检查是否始终二阶、出现 pre-asymptotic 区，或投影最终不活跃而恢复高阶；仅对实测范围陈述结论。

固定模型的完整周期核正定并不意味着当前最快网格已进入无投影区。不要把“最终渐近可能高阶”写成已测或任意参数的高阶证书，也不要将高精度原算子的误差直接当成网格表投影量的上界。

## 4. Experiment 2：目标精度约束下分别选参、找最快已测值

主对照保持原 N=256、rho=.75、kappa=.2、c=1、九个配置，目标 epsilon=1e-4。每个方法分别优化它合法的参数，而不把同一组参数强塞给三者：
- 原 PPPM/new grid-PSD：xi,sR,sF,M,p；new sF 是 pre-projection cutoff，post-projection 使用全部 modes。
- B2：xi,sR,sT=kappa*R_T,M,p，仍使用正核构造。

先借助 Experiment 1 的实测规律提出候选，再直接检查实际 E_G，最后计时排序。新方案的误差模型要显式包含投影误差/活跃区，不能直接假定 O(h^p)，也不要不加检查套用 B2 的二阶模型。

复用历史 best 作为起点，并对新方法做 coarse -> constrained local -> feasibility-boundary 搜索。对三种方法给相近搜索预算/明确报告不等的预算；同机局部复查原 PPPM/B2 best 附近，保证没有仅让新方法获得更充分调参。记录搜索域、独立候选数、失败/超时/内存不足点、筛选顺序、九配置验证和最终 best。最终目标为 complete cached graph-action（包括 near+far）的中位数，不是只计 FFT。只称“已测最优”，不称全局最优。

主 1e-4 完成后，资源允许时增加 epsilon=1e-3,1e-5,1e-6 的 accuracy-cost frontier；报告实际达到的档位。对达不到或资源超限的点明确标 N/A/未达标和边界，不放宽误差、不删失败点、不把外推时间画成实测值。长期任务应有 checkpoint 和可恢复命令；不使用未授权的付费计算资源。

## 5. 公平计时与 setup

旧的 1.268ms/2.094ms/65.630ms 只能作历史说明，不能与新硬件/新负载数据直接拼出速度比。最终三方法必须同一硬件、同批、单计算线程（记录环境），待所有 accuracy/reference/search 并发进程结束后计时。候选顺序随机化，warmup、多个 round 和重复样本，保存逐样本 raw CSV 和 IQR/区间。最好沿用原 5 rounds x 7 samples x 9 configs 协议。

分别报告：parameter-table setup（新方案含旧 influence、IFFT、symmetrization、batched eigh+projection、FFT）、configuration setup、near action、spread/gather、FFT/influence、complete cached matvec、内存。给出 `T_table+T_config+n_apply*T_matvec` 的摊销成本，说明位置改变后哪些缓存失效。不要将独立 reference/dense validation 时间混进 matvec，也不要把 setup 隐藏掉。PSD theorem 与数值精度只对所测配置通过的事实分开。

## 6. 绘图要求：三种方案都比较，不只是两张互不相干的报告

输出 PNG+SVG 或 PDF、所有绘图源 CSV、可单独重画的脚本和中文报告，method 命名/颜色/线型跨图一致，尽量延续现有科研图风格，配置族可分面以免曲线拥挤。至少包括：
1. 三方法 E_G 的参数扫描对比；E_GX 另图。共享横轴 sR/M/p/rho/kappa 可直接叠加或对齐分面。
2. 截断参数 panel 必须区分 pre-projection/原始 sF 与 B2 的 sT=kappa*R_T，不将它们数值相等当成同等截断；使用标清各自定义的分面，或改用物理可比的误差/成本坐标。
3. 固定物理设置下的 log-log mesh-error vs h（或 M），多个 p 的实测 slope，三方法对比，标明拟合区间和 floor。
4. 新方案的 projection correction vs M/p、负块比例/负特征值大小；与总误差的关系。投影前后 PSD 指标分清 table 与 full graph。
5. 三方法 error-vs-complete-runtime Pareto scatter/frontier，标出 epsilon=1e-4 及各自 fastest validated 参数；不把不可行点连接成可行前沿。
6. 各已完成 epsilon 的 fastest validated time 和所选 xi/sR/M/p；方法专用 cutoff 分开画。未完成档位显式留空。
7. 主精度档三方 complete-time/setup/component 对比；给出相对原 PPPM 与 B2 的同批速度比。

最终回答研究问题：新方案是否严格保证 long graph PSD；投影量有多大；实际阶数是否保留及适用范围；同精度最佳已测时间相比两条基线如何；新增 setup/内存代价；是否值得替代基础 B2。即使变慢或降阶，也如实保留结果。报告未测/未完成项，不声称已证明高阶或全局最优。完成后在 PR 中回报新分支/提交/图表路径与执行命令。
