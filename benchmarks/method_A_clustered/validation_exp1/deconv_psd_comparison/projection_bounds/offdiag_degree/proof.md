# Self-excluded degree bound：证明、计算与浮点限制

本增量仅研究已有 PSD projection correction 的误差界，不修改 `grid_psd.py`、Ewald splitting、物理核、参数选择、cached matvec、OU 或历史数据。输入是 PR #5 已有的 `CorrectionBounds`，其物理修正表 `C_d=(-S_d)_+⪰0`、非负 stencil 和 `C_hat=FFT(C)` 均直接复用。

## 精确算术中的改进界

沿用

\[
 C_{ij}=\sum_{g,l}w_{ig}C_{g-l}w_{jl}\succeq0,
 \quad C_{ji}=C_{ij}=C_{ij}^T,
 \quad\delta=\max_d\|C_d\|_2.
\]

权重非负、总和为一，故 `0⪯C_ij⪯δI`。含 self 的 degree 与 self 张量分别是

\[
 D_i^{total}=\sum_j C_{ij},\qquad C_{ii}\succeq0.
\]

定义

\[
 D_i^{off}=D_i^{total}-C_{ii}=\sum_{j\ne i}C_{ij}\succeq0.
\]

对任意粒子场 `v`，修正图拉普拉斯满足

\[
 v^T\Delta\Gamma v
 =\sum_{i<j}(v_i-v_j)^TC_{ij}(v_i-v_j).
\]

利用 PSD 张量的 `||a−b||_C²≤2||a||_C²+2||b||_C²`，得到

\[
 v^T\Delta\Gamma v\le2\sum_i v_i^TD_i^{off}v_i
 \le2\max_i\lambda_{max}(D_i^{off})\|v\|^2.
\]

因此

\[
 \boxed{\|\Delta\Gamma\|_2\le B_{off}
 =2\max_i\lambda_{max}(D_i^{off}).}
\]

由于 `D_i^total−D_i^off=C_ii⪰0`，有 `B_off≤B_total`。结合 PR #5 已证明的 `B_simple=Nδ`，对 `N≥2`，

\[
 \boxed{\|\Delta\Gamma\|_2\le B_{improved}
 =\min(N\delta,B_{total},B_{off})
 =\min(N\delta,B_{off})\le B_{old}.}
\]

其中 `B_old=min(Nδ,B_total)`。实现保留三项最小值以及两项等价式；浮点下两种表达可能存在末位差异，不能把该差异冒充定理失效或作数值修补。

## Self cancellation、零模与 tight case

完整响应可写成

\[
 (\Delta\Gamma v)_i=D_i^{total}v_i-\sum_j C_{ij}v_j
 =D_i^{off}v_i-\sum_{j\ne i}C_{ij}v_j.
\]

这里消去的是完整插值 self tensor `C_ii`，不是仅删除网格原点 `C_0`。一致平移场仍为零模；`N=1` 时没有其他粒子，`D_off=0` 和修正图严格为零。代码对这个空边集合返回结构上的精确零，不是人为误差 floor。

去除 self 可严格且显著改善旧界。例：`M=7,p=1`，三个粒子位于网格坐标 0、1、3；取 `C_0=αI`、`C_{±1,0,0}=βI`、其余零。只有前两个粒子之间存在修正边，故 `||ΔΓ||=2β=B_off`。较大的 `α` 只增加 self degree，不改变图拉普拉斯；旧 `B_total` 却随 `α` 增长。测试取 `α=10,β=.4`，旧合并界20.8，新界及真实范数均为.8。这是 sharp 构造，不是声称所有 PPPM 参数都能获得同样改善。

## Tensor-product stencil 的 self autocorrelation

对每粒子、每轴，计算

\[
 a_{i,\alpha}(d)=\sum_{u,v}
 w_{i,\alpha,u}w_{i,\alpha,v}
 \mathbf1[(g_{i,\alpha,u}-g_{i,\alpha,v})\bmod M=d].
\]

三维 tensor-product 结构给出

\[
 C_{ii}=\sum_d C_d a_{i,x}(d_x)a_{i,y}(d_y)a_{i,z}(d_z).
\]

`self_tensors` 用周期 index 的 `add.at` 聚合每轴全部 `p²` 项，所以 wrap、`M<2p−1`、`p=1`、以及重复 grid index 均被正确合并。对既有连续 cardinal stencil，每轴实际差分 support 至多 `min(M,2p−1)`；代码只构造该实际 support 的 Cartesian product，并从物理 `C_table` gather 相应 block，**不扫描完整 correction active set**。标准复杂度为

\[
 O(Np^2+N\min(M,2p-1)^3),
\]

没有 `p^6` 双重三维 stencil 循环，没有粒子 pair matrix。测试中的自定义非连续重复 stencil 仍按其实际周期 support 精确计算；其 support 大小应按实际值报告，而不是套用连续 cardinal stencil 的复杂度公式。

## 普通相减与严重相消诊断

普通路径使用原 correction-only FFT 的 `D_total`，再计算 `Draw=D_total−C_self`。当 self trace fraction 接近一时，两个大数相减可能丢失真正的 off degree。首阶误差尺度与

\[
 u\bigl(\|D^{total}\|+\|C_{self}\|\bigr)
\]

有关，而真实 `D_off` 可能远小于这两个量。实现记录 raw 最小特征值、对称性、相消 ratio、估计 roundoff 指示量及与稳定结果的差。这些是**诊断 heuristic，不是严格数值误差界或区间证书**。

默认 `auto` 在相消 ratio 大于 `1e8`、非零 self 被 raw 相减消成零，或 raw 负谱超出 heuristic roundoff 尺度时启用稳定路径。该阈值只决定计算哪一种等价求和，不裁剪特征值、不改 `C`、不改变数学 bound，也不设置误差 floor。真实零分母的 ratio 使用 `None` 和明确 flag，绝不通过 `tiny` 改变数学量来获得可序列化数字。

## 无相消的 local occupancy 替代

稳定路径不形成 N×N pair matrix，也不为每个粒子执行 FFT。它执行以下正权局部求和：

1. 将 tensor-stencil entries 按 `(grid cell,particle owner)` 排序，正相加聚合重复 index。
2. 对每个 occupied cell，按 owner 的正权列表计算 prefix/suffix sums。对 owner `i`，other mass 是 `prefix_before_i+suffix_after_i`；从不取 `total−self`。若 `i` 不在 source cell，直接取该 cell 的正 total。
3. 对每个非零 `C_d` 和目标粒子的每个 occupied stencil cell `g`，查找 `h=g−d`，只累计其他 owner 的 mass，乘目标正权，再累计 `C_d`。

精确算术中，这直接计算

\[
 D_i^{off}=\sum_g w_{ig}\sum_{j\ne i,l}C_{g-l}w_{jl},
\]

所以与 raw 相减公式数学等价；浮点下避免了大 self 项的灾难性消去。prefix/suffix 只对其他 owner 的非负 mass 求和；它不消除一般 float64 累加、tensor component 相加或 `C` 本身的特征分解舍入，仍不能声称 interval-certified。

设去重的 cell-owner entries 数为 `E≤Np³`，全 correction active offset 数为 `K`。额外成本约为排序 `O(E log E)`、正 prefix/suffix 与 `O(KE)` 次局部查询；查询用排序 index 的二分查找。内存为 `O(E+N)`，不保留粒子 pair graph。self-dominated outlier 只有19个 active offset，因此 `N256,p9` 的上限约354万次查询；宽表应按实际 `K` 明示更高成本。

默认自动路径采用5000万次查询上限控制离线资源。如果严重相消且该预算不足，代码**不**将不可靠 raw 结果作为新界：`B_off/B_improved=None`、`numerical_bound_available=False`，保留 `B_off_raw/B_improved_raw`、raw tensors 与 failure reason。`stable='always'` 可显式要求完整稳定求和而越过预算；`stable='never'` 用于展示 raw 审计，不可靠时同样标 unavailable。没有丢弃小 correction 或悄悄截短 active support。

## 接口、独立审计与成本口径

`OffdiagBounds(existing_cb)` 直接引用已有 correction Fourier 表，不重建 `C_hat`。主接口 `bounds(q)->(scalars,Doff)` 返回旧界、新界、raw 版本、trace fraction、最大 degree 特征值比、相消标记与稳定算法成本。只读审计可用 `last_D_total/last_C_self/last_D_off_raw/last_D_off_stable/last_D_off`；`self_tensors(q)` 和 `stable_off_degree(q)` 可单独调用。

`t_offdiag_setup_s` 是已有 `D_total` 之后新增的 self/autocorrelation、raw 谱与必要稳定求和成本；`t_existing_total_bound_setup_s` 与 `t_offdiag_with_total_setup_s` 单独保存。`t_offdiag_parameter_setup_s` 不包括已由 `CorrectionBounds` 建好的 correction FFT。提供 `D_total,total_result` 可复用外部已有 degree。任何新增 bound 成本都不进入 production matvec，也不与 PR #4 的旧计时拼接声称提速。

独立审计应直接组装 correction-only `C_ij`，**先把 `W_ii` 设为零再按行相加**，以免验证代码再次发生 total-minus-self 相消。小 N literal double-stencil 仅用于验证；真实图范数也应来自 correction-only offdiagonal graph，不能用历史 full-difference norm 替代。报告应保留 raw/stable 相对这个真实范数的误差、symmetry、最小特征值及 failed/unavailable。

12项单元测试覆盖：奇偶网格、p1及多 p、wrap 和 `M<2p−1`、重复 index、literal self 与 off pair sum、tiny other mass 的稳定求和、真实 self-only 零图、N1、已知 tight 改善构造、inactive/JSON null、预算不足显式 unavailable、禁止新 FFT/稠密粒子图、旧 mesh 不变、零模/周期位置及外部 degree 复用。结果保存为 `results/test_offdiag_bounds.log` 和 `results/tests_summary.json`。

## 选参含义

`B_improved` 仍是 sufficient/conservative 的 projection correction 上界。它不通过，不能推出实际误差不满足。当前的 `E_old_measured+B_improved/||Γ_D||` 只是相对于独立数值 reference 的 triangle envelope；`epsilon_old^cert` 尚未就绪时，不是 certified total error。PSD 的结构保证对所有合法参数成立，不依赖某个配置的 bound 是否紧。应区分另需完整认证预算的 theory-certified candidates 与 independently measured empirical-pass candidates，不因保守 bound 宽松自动淘汰实测达标者。
