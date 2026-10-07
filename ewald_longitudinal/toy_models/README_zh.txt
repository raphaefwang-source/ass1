Double well / Lennard-Jones + longitudinal friction: CPU reference kit
=====================================================================

用途
这是独立编写的小规模参考程序，用于把“保守势 + A/B 摩擦与匹配噪声 + 轨迹 + 四类物理量”跑通。
它不是 PRL 原作者代码，也没有复现原论文的 DeePCG 势、多体记忆模型或 MD 数据。
两种保守势都依赖粒子间距离。double well 不是把每个粒子束缚在固定位置的外场势。

安装与运行（Python 3.10+，NumPy，Matplotlib）
  python -m pip install -r requirements.txt
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_physics.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python prl_toy_models.py demo --kernels A B

上面的完整默认示例：N=64，L=5.5，m=1，kBT=0.7，dt=0.005，burn=800，production=1600 步，
每4步保存一帧。生产轨迹长度为8，是调试与初步观察用的短轨迹。不能据此声称平衡、收敛或物理优越性。
可增加 --burn、--steps，并用多个 --seed 独立运行；热化长度应由观测量的稳定性判断。
默认只有 A；--kernels A B 会运行两个势与两个 kernel 的四种组合。

力学模型与周期约定
  dq = p/m dt
  dp = F(q) dt - Gamma(q) p/m dt + sqrt(2 kBT Gamma(q)) dW
  F = -grad sum_{i<j} u(r_ij)
  K_ij = g(r_ij) rhat_ij rhat_ij^T
  Gamma_ii = sum_{j!=i} K_ij; Gamma_ij = -K_ij.

这里所有相互作用都使用立方盒的 minimum-image 距离。摩擦包括所有粒子对，没有径向截断。
这是一种明确的有限盒测试模型；它不同于完整周期镜像和，也没有使用 Ewald/PPPM。
与现有 Ewald 代码对比时，必须先统一这两种周期定义，不能把二者差异当作算法误差。
未对 Kernel A 在 r=0 作软化；排斥势阻止近距离重叠。零距离会报错，过近碰撞会中止并提示缩小 dt。

保守势（s=r/sigma；默认 epsilon=sigma=1）
  LJ: u0 = 4 epsilon (s^-12 - s^-6).
  DW: u0 = epsilon [0.1 s^-12
           - exp(-(s-1.12)^2/(2*0.12^2))
           - exp(-(s-1.60)^2/(2*0.12^2))].
DW 是短程排斥核心加两个吸引 Gaussian 势阱；两个实际极小值略偏离1.12和1.60。
两种势均乘相同的 C2 quintic switch：r<=2sigma 时 S=1，r>=2.5sigma 时 S=0；
中间令 x=(r/sigma-2)/0.5，S=1-10x^3+15x^4-6x^5。
力包含 S'(r)u0(r) 项，不能仅把未截断力乘 S。
需要 L>5sigma，以使势的截止距离小于半盒长。

A/B 径向摩擦（默认 gamma=0.5, kappa=0.7, r_ref=1.3）
  g_A(r)=gamma*(r_ref/r)*exp[-kappa*(r-r_ref)].
  g_B(r)=gamma*(r/r_ref)*exp[-kappa*(r-r_ref)].
等价于 A exp(-kappa r)/r 和 B r exp(-kappa r)。
此处规定 g_A(r_ref)=g_B(r_ref)=gamma，仅统一一个参考距离的摩擦强度。
这不等于两者总摩擦相同，也不是用 MD 做过参数拟合。

时间推进
使用 BAOAB。冻结 q 的 O 步对完整 Gamma 做 eigh，计算：
  R=exp(-dt Gamma/m)
  S=[m kBT (I-exp(-2dt Gamma/m))]^(1/2).
总动量分量原样复制，噪声和涨落速度投影到零总动量子空间。
对冻结位置的完整矩阵函数满足有限时间 FDT。BAOAB 整个有限步长模拟仍有时间离散误差。
只截除舍入量级的负特征值，显著负特征值直接报错。没有静默裁剪保守力。
Gamma 的构造为 O(N^2)，每步 dense eigh 为 O(N^3)。本包仅是小规模基准，不能用于宣称快速算法复杂度。

接入你自己的轨迹
保存一个不含 pickle 的 .npz，字段为：
  Q: (saved_frames, N, 3) 位置；可以是连续位置或周期折回位置。
  V: (saved_frames, N, 3) 速度，不是动量。
  t: (saved_frames,) 均匀采样的实际物理时间。
  box: 正的标量或长度为3的盒长。
  粒子编号必须跨帧保持一致。这里只处理固定正交盒、三维、等质量体系。

  python prl_toy_models.py analyze your_trajectory.npz --out analysis
  python prl_toy_models.py analyze your_trajectory.npz --vccf-edges 0 1 1.5 2 2.5

--max-lag 的单位是已保存的帧数，不是积分步数；默认取轨迹帧数的一半。
--remove-com 用一个常数平均质心速度变换到共动参考系，同时修正位置和速度。
它不会逐帧减去一个随时间变化的漂移；后者可能掩盖动量不守恒。

物理量定义
  VACF(t)=<v_i(t0+t) dot v_i(t0)> / <|v_i(t0)|^2>。
  VCCF(t;bin)=<v_i(t0) dot v_j(t0+t) | r_ij(t0) in bin, i!=j>。
VCCF 对每个时间起点按初始距离分组，保留相同 pair 随延迟时间的相关；原始值单位为速度平方。
同时输出 vccf_normalized（除以相同起点的平均速度平方）和每个 bin 的 pair-origin 数。
空 bin 返回 NaN，不能用零替换缺失数据。

  u(k,t)=N^-1 sum_j v_j(t) exp(i k dot q_j(t))，k=2pi*(nx/Lx,ny/Ly,nz/Lz)。
  u_L=khat dot u，u_T=u-u_L khat。
  C_L(t)=Re< u_L(t0+t) conj(u_L(t0)) > / <|u_L(t0)|^2>。
  C_T(t)=Re< u_T(t0+t) dot conj(u_T(t0)) >/2，再除以自身零时值。
默认输出三个最小波矢 (1,0,0),(0,1,0),(0,0,1)，不假设不同模态独立。
这里的 L/T 是观测方向，不是在摩擦 kernel 中额外加入 transverse channel。

  distinct van Hove 使用 i!=j，按 |minimum_image(q_i(t0+t)-q_j(t0))| 分箱。
  g_d(r,t)=Vbox * counts/[n_origins*N*(N-1)*shell_volume]。
  shell_volume=4pi/3*(r_right^3-r_left^3)，仅绘到最短盒长的一半。
这是无量纲、有限 N 归一化的 distinct 函数，满足 g_d(r,0)=同一定义的 RDF，
独立均匀粒子的期望基线为1。传统 G_d 除以 rho 的定义可能有 (N-1)/N 因子，比较前须对齐。
不包括 i=j 的 self van Hove，也没有把普通位移直方图误当 distinct 函数。

每组输出
trajectory.npz：生产轨迹和完整参数。
observables.npz：所有统计量、距离 bin、波矢、时间起点与 pair 数。
correlations.csv、diagnostics.csv：可直接作图的表。
vacf.png、vccf.png、hydrodynamic_modes.png、van_hove.png、rdf.png。
overview.png：势形状、试跑 VACF 和温度概览。
summary.json：参数与数值诊断。
单条轨迹的多个时间起点有相关性，本程序不把它们当独立样本制造置信区间。
正式报告需要独立种子或合理分块的误差估计，并检查步长、记录间隔、轨迹长度与体系规模。

验证
test_physics.py 检查保守力有限差分、两个势阱、截断连续性、成对作用力抵消、
Gamma 耗散恒等式/正定性/平移零模、冻结 O 步 FDT、动量守恒、
静态相关、周期绕回不变性、共动变换和理想气体 distinct 归一化。
这些检验验证实现与定义，不证明 toy 模型描述了 PRL 的星形聚合物熔体。

参考应用与观测量来源
Liyao Lyu and Huan Lei, Phys. Rev. Lett. 131, 177301 (2023).
https://doi.org/10.1103/PhysRevLett.131.177301
本包的两种 toy 势、参数和实现由本次讨论独立选择，不来自该论文。
