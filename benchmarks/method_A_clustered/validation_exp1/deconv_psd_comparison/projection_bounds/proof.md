# PSD 投影修正的两个严格误差界

本轮只核查已有投影的修正量和上界。`bounds.py` 不更改投影、粒子算子、参数选择、cached matvec 或任何历史结果。下面的严格结论以精确实数算术、非负且 partition-of-unity 的既有 stencil 为前提；后文单独说明 float64 计算的限制。

## 定义与物理规范

令已有去卷积网格的原始物理表为

\[
 T_d^{raw}=h^{-3}\operatorname{IFFT}(G)_d,
 \qquad S_d=\tfrac12(T_d^{raw}+T_{-d}^{raw}).
\]

六个存储分量表示每个实对称 3×3 block。现有 `grid_psd.py` 已定义

\[
 C_d=(-S_d)_+=Q_d\operatorname{diag}([ -\lambda_d]_+)Q_d^T\succeq0,
 \quad T_d^+=S_d+C_d,
 \quad \delta=\max_d\|C_d\|_2
             =\max_d[-\lambda_{min}(S_d)]_+.
\]

对反演成对的修正作正平均不会破坏 PSD；在精确算术中谱函数的唯一性给出 `C_d=C_{−d}`，因此该平均不改变理想修正。

`CorrectionBounds` 只读取既有 `ProjectedMesh.delta_table`，即上述**物理量纲**的 `C`。其 Fourier 表取 `C_hat=FFT(C)`，不再乘 `h³`，gather 也不再除 `h³`。这与原始 `Mesh.Gr` 采用 Fourier 核量纲的约定等价，不能把两套规范混用。保留完整 rFFT 网格，包括投影产生的旧截断球外频率与实际 zero mode。

设 `w_ig≥0`，且 `Σ_g w_ig=1`，定义粒子边张量

\[
 C_{ij}=\sum_{g,l}w_{ig}C_{g-l}w_{jl}.
\]

由于每个 block 满足 `0⪯C_d⪯δI`，非负插值及总权重为一给出

\[
 0\preceq C_{ij}\preceq\delta I,\qquad C_{ji}=C_{ij}=C_{ij}^T.
\]

修正图拉普拉斯的二次型为

\[
 v^T\Delta\Gamma v
 =\sum_{i<j}(v_i-v_j)^T C_{ij}(v_i-v_j)\ge0.
\]

这里 self terms 不出现在差分式中；实现可以在 `D` 与 `H` 中同时包含完整插值的 `C_ii`，然后相消。

## 简单界：Nδ，且普遍常数不可改善

对 `N≥2`，利用完整图的恒等式，

\[
\begin{aligned}
 v^T\Delta\Gamma v
 &\le\delta\sum_{i<j}\|v_i-v_j\|^2\\
 &=\delta\bigl(N\sum_i\|v_i\|^2-\|\sum_i v_i\|^2\bigr)
 \le N\delta\|v\|^2.
\end{aligned}
\]

因此

\[
 \boxed{\|\Delta\Gamma\|_2\le B_{simple}=N\delta.}
\]

这个 `N` 系数对允许的 correction table 类是 sharp 的：取所有位移上的 `C_d=δuuᵀ`，其中 `u` 是单位向量，则任何合法 stencil 都给出 `C_ij=δuuᵀ`。对沿 `u` 且粒子平均为零的非零向量，有 `ΔΓv=Nδv`，从而范数恰为 `Nδ`。这不是声称每个实际 PPPM 表都达到等号，而是证明仅凭 `δ`、非负 stencil 和粒子数无法普遍减小这个常数。

`N=1` 的图没有不同粒子边，故修正算子严格为零。本实现单独返回 `B_simple=0`，并标注 `singleton_exact_zero=True`；仍按原定义计算含 self 的 degree 及 `B_degree`。这不对一般配置设置任何误差 floor。

## 配置相关 degree 界：不需要粒子稠密矩阵

令

\[
 n_l=\sum_jw_{jl},\qquad
 D_i^C=\sum_j C_{ij}
 =\sum_g w_{ig}\sum_l C_{g-l}n_l.
\]

`D_i^C` 是 PSD 张量之和，包括 self 也保持 PSD。对任意 PSD `C`，有

\[
 (a-b)^TC(a-b)\le2a^TCa+2b^TCb.
\]

因此

\[
\begin{aligned}
 v^T\Delta\Gamma v
 &\le2\sum_i v_i^T\bigl(\sum_{j\ne i}C_{ij}\bigr)v_i\\
 &\le2\sum_i v_i^TD_i^Cv_i
 \le2\max_i\lambda_{max}(D_i^C)\|v\|^2.
\end{aligned}
\]

得到

\[
 \boxed{\|\Delta\Gamma\|_2\le B_{degree}
       =2\max_i\lambda_{max}(D_i^C).}
\]

这个证明只依赖每条边 PSD，并不需要整个 pair-response block matrix `H=(C_ij)` PSD。含 self 的 degree 有时偏保守，但不会让界失效。

两种上界均成立，所以

\[
 \boxed{\|\Delta\Gamma\|_2\le B=\min(B_{simple},B_{degree}).}
\]

degree 界可以严格优于 `Nδ`，甚至 tight。例如三个最近格点粒子处于一维网格坐标 0、1、3，`M=7,p=1`；只让 `C_{±1,0,0}=δI`，其余 block 为零。只有前两个粒子之间有修正边，第三个孤立，`||ΔΓ||=2δ=B_degree<3δ=B_simple`。

## 实现、复杂度及独立检查

参数相关 setup 只为 `C` 构建六个 Fourier 分量，成本 `O(M³log M)`，可供相同参数的多个配置复用。每个配置的 bound 只需 density spread、一次 density FFT、六次 correction convolution/IFFT/gather，以及 `N` 个 3×3 特征值计算；assignment 工作量 `O(Np³)`。`bounds(x)` 从不形成 `3N×3N` 矩阵。

`pair_matrix(x)` 仅供离线验证，利用每个非零 correction offset 的一维 stencil correlation 精确收缩。它不计算两个大算子的差，也不删除任何微小 correction。独立 `direct_pair_matrix(x)` 则显式双重三维 stencil 收缩；测试再用完整 complex DFT（包含偶数网格 Nyquist 平面）、correction-only FFT basis response 和 graph action 交叉验证。

计时分开记录：

- `t_correction_fft_setup_s`：新增 correction FFT，参数相关；
- `t_correction_parameter_setup_s`：新增 bound 参数 setup 总时间，不含投影构建或可选审计；
- `t_bound_configuration_setup_s`：当前配置的 stencil、correction degree、3×3 谱及 bound；
- `t_correction_audit_s`：可选离线浮点审计，不计入前两项。

这些成本不替换历史 projection setup、原 cached matvec 或共同串行最终计时。

## 浮点误差与原始 E_proj 的适用范围

上述是精确算术定理。当前实现使用 float64 FFT 与特征分解，没有 interval arithmetic、定向舍入或向上取整证书，因此**计算出的 float64 数字本身不是经过舍入认证的区间上界**。没有通过增加 floor、裁掉微小 correction 或调大 bound 来隐藏数值违例。

`audit=True` 单独报告 correction 的最小/最大浮点特征值、相对 `δ` 的谱最大值差、反演缺陷、FFT roundtrip 残差、raw→symmetrized table 缺陷，以及存储的新旧 influence 差与 `FFT(C)` 的残差。degree 的最小浮点特征值、stencil partition defect 也独立记录。非常小的负特征值或机器量级 bound violation 必须按其原值报告。

严格 `C` 上界描述同一非负 stencil 下、对称化基准表 `S` 到 `S+C` 的修正。已有 `E_proj` 是两个完整 float64 离散算子之差；raw 反演舍入、旧/新 assignment 的末位归一化差异、FFT 和影响函数相加舍入，均可能使它与 correction-only 图范数有机器量级差别。不能把 `C` 的界未经比较直接套给一个非对称 raw 算子。runner 应保存 correction-only 范数与历史 full-difference `E_proj` 的比较残差，并独立列出数值违例。

真实 `C=0`、严格 inactive 和计算出的 full-difference 浮点残差分开标记；tightness `B/||ΔΓ||` 在真实零修正时不定义，不以人为 floor 代替。

## 经验误差 envelope 不是 certified 总误差

若 `Γ_D` 是现有独立数值 reference，`E_old=||Γ_old−Γ_D||/||Γ_D||` 是实测误差，那么三角不等式给出相对于该数值 reference 的

\[
 E_{new}\le E_{old,measured}+B/\|\Gamma_D\|_2
\]

（仍须把上述 float64 残差分开处理）。这条曲线应命名为“empirical-reference triangle envelope”。`E_old,measured` 不是另行认证的 `epsilon_old`，不能据此声称完全 certified total bound；若没有已有的独立认证，应标 `unavailable`。complete-operator 与 mesh-only 误差也必须分别命名，不能混画后解释为同一个量。

本轮只比较 active 区上界的保守程度与误差预算意义，不增加方法、SDP、special-case 修补或新的参数选择法；active/inactive 混合过渡段的形式斜率也不被解释为收敛阶。
