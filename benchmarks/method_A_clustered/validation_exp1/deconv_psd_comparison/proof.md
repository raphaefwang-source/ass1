# Deconvolved PPPM + 实空间网格 PSD 投影：定义、结构证明与精度限制

本目录的新方案在原始高阶去卷积 PPPM 的实空间网格张量上作逐块谱投影。它不投影 Fourier multiplier，不投影粒子矩阵，也不向粒子矩阵加对角平移。短程项仍是既有 Candidate 1 / law A，物理振幅、κ、周期镜像和全图拉普拉斯定义均保持原模型。

## 离散算子与 FFT 归一化

原始 `Mesh.Gr` 存放的是去卷积后的 Fourier 核系数，含原来的 spherical Fourier 截断、保留的 zero mode 与原来的 influence threshold。其 `apply` 和 `degree` 在 IFFT/gather 后除以 `h³`。因此物理实空间张量是

\[
 T_d=h^{-3}\operatorname{IFFT}(G)_d,
 \qquad d\in\mathbb Z_M^3.
\]

首先采用 `T_d ← (T_d+T_{−d})/2` 保证周期反演对称。对每个实对称 3×3 block，设 `T_d=Q_d diag(λ_d) Q_dᵀ`，定义

\[
 T_d^+=Q_d\operatorname{diag}(\max(\lambda_d,0))Q_d^T,
 \qquad \Delta T_d=Q_d\operatorname{diag}(\max(-\lambda_d,0))Q_d^T.
\]

代码直接累积负谱的修正 `ΔT`，没有用阈值忽略较小负特征值；无需修改的 block 保留原值。这避免对所有正 block 做特征分解重建后产生遍布网格的消去舍入误差。`ΔT` 与其反演作正平均，保持 PSD 与反演对称。新系数为

\[
 G^+=G+h^3\operatorname{FFT}(\Delta T)
     =h^3\operatorname{FFT}(T^+) .
\]

最后一个等式在实数精确运算下成立；浮点下差异受已有 FFT 与反演平均的舍入误差控制。特别地，如果不存在任何负特征值，则 `Gr` 与原始数组逐位相同。

**FFT 后保留完整 rFFT 网格。** 投影会产生原球形频带之外的新模态，不能再用旧 `kc` mask 截断。新的 zero mode 由修正后的实空间表决定，不恢复为投影前的解析值。此后继续使用原始 `/h³` gather 规范；不能与 B2 的物理表规范混用。

## 对任意粒子配置的结构 PSD

令 `w_i(g)≥0` 是 B2 已有正 B-spline recurrence 给出的张量积权重，并使用相同 spread 与 adjoint gather。每维权重归一化，`p=1` 的半网格 tie 显式处理。粒子长程边张量是

\[
 W_{ij}^+=\sum_{g,h}w_i(g)w_j(h)T_{g-h}^+.
\]

这是 PSD 张量的非负线性组合，所以每条 `W_ij^+ ⪰ 0`。网格表的反演对称给出 `W_ji^+=W_ij^+`，而张量本身对称。既有短程权重 `K_S(r)=θ_S(r) rrᵀ/r²` 同样 PSD；保留任何成对镜像集合也保持此性质。总算子的二次型是

\[
 X^T\Gamma_h^+ X
 =\sum_{i<j}(X_i-X_j)^T
  (K_{S,ij}+W_{ij}^+)(X_i-X_j)\ge0.
\]

这是一项与具体九组测试无关的结构结论；数值测试只验证实现。必须以新 `G^+` 重新计算完整 mesh degree `D_i^+=Σ_j W_ij^+`。沿用旧 degree 会破坏上述差分形式、平移零模与 PSD 证明。`CachedGammaH` 的 setup 会通过新 mesh 自动重算 degree；短程与原始作用代码无需改写。

degree 和 H 都包含完整的插值 self response `W_ii^+`，二者相消，不能只删除网格原点 `T_0`。三种一致平移场位于零空间；单粒子系统的完整 action 也为零。浮点实现中的 PSD 与零模偏差应按机器精度及算子尺度评估，不能声称浮点特征值逐位非负。

## 不遗漏新模态的高效稠密核查

原始有限 mode sum 加以下修正，等于使用全部新 Fourier 模态的结果：

\[
 \Delta W_{ij}=\sum_{d:\Delta T_d\ne0}\Delta T_d
 \prod_{a=1}^{3}C_{ij,a}(d_a),
 \quad C_{ij,a}(r)=\sum_g w_{i,a}(g)w_{j,a}(g-r).
\]

实现通过一维紧 stencil 的 sparse×dense 乘积计算相关函数，再分块累积所有非零 `ΔT`。没有修正阈值、频率丢弃、低秩近似或粒子基向量重放。这也避免了 `O(N²M³)` 的稠密全 Fourier mode 求和。`direct_pair_matrix` 则直接进行两组完整三维 stencil 的双重收缩，独立于 mode sum、相关函数算法和 FFT action，仅用于小系统核查。

## PSD 不能推出高阶精度

逐 block 投影是到 PSD 锥的 Frobenius 最近点投影，但**这不证明它保留原 PPPM 的高阶精度**。原始去卷积表含抵消 assignment smoothing 的各阶修正；对近秩一的纵向实空间核，二阶逆 smoothing 修正可能出现负横向谱。剪掉这部分修正后，spread/gather 的二阶偏差可能重新出现。

可以给出一个局部连续展开，解释二阶误差的来源。忽略 alias 和 Fourier 截断，在固定光滑位置 `r>0`，B-spline 的 spread/gather transfer 为

\[
 |\widehat W(k)|^2=1-\frac{p h^2}{12}|k|^2+O(h^4),
 \quad T_{\rm dec}=K-a\Delta K+O(h^4),\quad a=\frac{p h^2}{12}.
\]

对孤立径向纵向张量 `K=f(r)nnᵀ`，有

\[
 \Delta K=(f''+2f'/r-6f/r^2)nn^T+(2f/r^2)I.
\]

所以原逆 smoothing 修正有两个横向特征值 `−2af/r²+O(h⁴)`；当 `f>0` 时它们为负。PSD 投影删掉此横向修正，即向 `T_dec` 加入 `2af/r²(I−nnᵀ)`。再 spread/gather 后得到

\[
 K+\frac{p h^2 f(r)}{6r^2}(I-nn^T)+O(h^4).
\]

这说明对 rank-one/近 rank-one 的局部核，剪谱可留下明确的 `O(h²)` 横向偏差；提高 p 还会增加这个首项的系数。此展开是机制分析，未将周期镜像、alias 和有限 `kc` 忽略项冒充本实验的严格误差界。

对于固定有限周期盒和 κ，所有周期镜像的和可能使理想长程网格表处处严格正定。如果足够细的网格下原去卷积表已全部 PSD，则投影从该点起成为恒等映射，高阶精度恢复。这一渐近区间可能远超实际最快参数所在范围。因此，报告同时给出实际 `M` 扫描斜率、mesh-only 误差、投影前后误差、修正量、负 block 数和修正完全消失的位置；不能用结构证明代替收敛实验。

同理，“修正张量本身 PSD”说明投影后的图拉普拉斯在 Loewner 顺序上增加，并不说明它更接近物理 reference。应以同一份独立 unsplit periodic direct sum reference 评估三种方法。

## 测试与审计范围

`python3 benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/test_grid_psd.py`

九项测试包含：已知旋转谱及任意小负特征值、正权重及 half-grid ties、奇偶网格的原 `/h³` 归一化、含 Nyquist 平面的完整复 DFT 对照、独立三维 direct stencil、新频率保留、degree 重算、FFT/staged/dense/basis action 一致性、单粒子 self cancellation、平移零模、周期换位、粒子边和完整算子 PSD、零振幅与非法输入。结构证明适用于非负振幅、κ>0、ξ>0、合法有限网格和有限截断；重合的不同粒子不是奇异完整物理核的合法输入。

`audit_eigenvalues=True` 是默认审计模式，额外计算投影前 raw 与投影后表的特征值，以及修正量分位数。正式 production setup 可传 `audit_eigenvalues=False` 跳过这些独立核查，保留算法必需的完整 `eigh` 与谱剪裁。未实际计算的 `min_eig_after`、`min_eig_raw` 和分位数标为 `None`，并记录 `diagnostic_eigenvalues_evaluated=False`；不能将缺失值视为零。新增测试验证两种模式的 `table`、`delta_table`、`Gr` 和完整 action **逐位相同**，因此此计时优化不改变数学算子。

典型单线程开发核查（N=256，κ=.7，ξ=.85，sF=6）：M40,p7 有1447个负 block，最小投影前特征值约−0.00290519，完整 dense pair matrix 约0.53秒；M80,p9 无负 block，最小特征值约0.000950235，dense pair matrix 约0.13秒。这里只是算法开发时的耗时量级，正式公平计时以统一串行实验原始样本为准。

正 recurrence 与旧 `Mesh` assignment 的量化对照使用全部九组 N256 配置、M40、κ=.7、ξ=.85、kc=2。最大差异如下；每个 apply 使用同种子随机向量，比较 `project=False` 与原 `Mesh`。精确 half-grid tie 不属于这九组随机配置；p1 tie 是已有原 stencil 的边界歧义，采用 B2 显式最近格点规则。

| p | 一维权重 max abs 差 | 三维权重 max abs 差 | apply 最大相对差 |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 |
| 3 | 1.1102e−16 | 1.6653e−16 | 2.9095e−16 |
| 5 | 2.2204e−16 | 1.3878e−16 | 4.9194e−16 |
| 7 | 2.2204e−16 | 1.1102e−16 | 4.5572e−16 |
| 9 | 2.2204e−16 | 8.3267e−17 | 4.8934e−16 |
| 11 | 2.7756e−16 | 6.9389e−17 | 5.2863e−16 |

`table_audit.py` 独立遍历六组扫描中77个不同的 `(rho,kappa,xi,sF,M,p)` 网格表，全部构建成功；12个代表参数（含 M24/25/40/80、κ=2、ρ=.3）进一步作 N3 literal 三维 stencil 核查。结果保存在 `results/table_diagnostics.csv` 和 `results/table_audit_summary.json`：direct stencil 与 FFT basis 最大相对差3.853e−13，graph action 最大相对差3.583e−13，所有表投影后最小特征值的最小值为−9.608e−17（谱剪裁和相消的浮点舍入），raw 与反演平均表最小特征值差不超过6.08e−18。独立正权重对照另存 `results/assignment_weights_audit.csv`。

77表中，κ=.7、ρ=.75、ξ=.85 的 baseline `sF=6,p9` 在 M72/75/80 已无负 block，M64仅19个；moderate `sF=4,p7` 从已测 M60 开始投影恒等。这里的“开始”仅指给定离散扫描点，并不声称所有更大整数网格均已逐一验证。
