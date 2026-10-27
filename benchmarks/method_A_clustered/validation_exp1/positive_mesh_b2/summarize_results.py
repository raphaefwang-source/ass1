"""Generate the report from measured data, and check experimental completeness."""
import hashlib
import json
from collections import Counter
import experiment as X
from experiment import np,E,C
from scipy.optimize import nnls

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)

def maximum(rows,key):return max(r[key] for r in rows)

def main():
    rows=X.read(X.RESULTS/'sweeps.csv');fits=X.read(X.RESULTS/'mesh_rate_fits.csv')
    search=X.read(X.RESULTS/'search.csv');valid=X.read(X.RESULTS/'validation.csv')
    summary=X.read(X.RESULTS/'validation_summary.csv');times=X.read(X.RESULTS/'final_timing.csv')
    decomp=X.read(X.RESULTS/'error_decomposition.csv');refs=X.read(X.RESULTS/'references.csv')
    assert len(rows)==927 and len(refs)==72 and len(decomp)==9
    counts=Counter((r['sweep'],r['setting'],r['value']) for r in rows)
    assert len(counts)==103 and all(v==9 for v in counts.values())
    assert min(r['lam_min_h'] for r in rows)>0
    best=json.loads((X.RESULTS/'best_configuration.json').read_text())
    old=next(r for r in times if r['method']=='PPPM')
    baseline=X.read(X.RESULTS/'original_pppm_revalidation.csv')
    protocol=json.loads((X.RESULTS/'timing_protocol.json').read_text())
    audit=json.loads((X.RESULTS/'audit_checks.json').read_text())
    construction=X.read(X.RESULTS/'construction_audit.csv')
    assert best['max_E_G']<=X.TOL and len(baseline)==9
    hist=json.loads((X.RESULTS/'historical_artifact_hashes.json').read_text())
    assert all(hashlib.sha256((C.REPO/p).read_bytes()).hexdigest()==h for p,h in hist.items())
    br=[r for r in valid if r['id']==best['id']]
    # A cost guide for this implementation; screen timings are explicitly noisy.
    AA=np.array([[r['n_images'],r['M']**3*np.log(r['M']**3),256*r['p']**3,1] for r in search],float)
    yy=np.array([r['t_screen_s'] for r in search]);scale=np.maximum(AA.max(0),1)
    coeff=nnls(AA/scale,yy)[0]/scale
    pred=AA@coeff
    cost=dict(zip(('seconds_per_image','seconds_per_fft_work','seconds_per_stencil_work','fixed_seconds'),coeff))
    cost.update(R2=float(1-((yy-pred)**2).sum()/((yy-yy.mean())**2).sum()),
                note='screen timings overlapped accuracy work; guide only, never the final objective')
    X.jsave(X.RESULTS/'screen_cost_model.json',cost)
    ms=lambda x:f'{x*1000:.6f}'
    sci=lambda x:f'{x:.8g}'
    bms=best['median_matvec_s'];oms=old['median_matvec_s']
    base=[r for r in rows if r['sweep']=='M' and r['setting']=='baseline' and r['value']==80]
    oldrows=X.read(X.HERE.parent/'accuracy_sweeps/results/sweeps.csv')
    oldbase=[r for r in oldrows if r['sweep']=='M' and r['setting']=='baseline' and r['value']==80]
    finefits=[r for r in fits if r['fit_range']=='finest_half' and r['metric']=='E_G_mesh_only']
    rates=table(['setting','family','M range','measured order q'],[
        [r['setting'],r['family'],f"{r['Mlo']}–{r['Mhi']}",f"{r['order']:.5f}"] for r in finefits])
    maincompare=table(['method','xi','sR / rc','table or Fourier cutoff','M','p','max EΓ','cached ms','config setup ms','table setup ms'],[
        ['B2',best['xi'],f"{best['sR']:g} / {best['sR']/best['xi']:.7g}",f"κ R_T={best['sT']:g}",best['M'],best['p'],sci(best['max_E_G']),ms(bms),ms(best['median_config_setup_s']),ms(best['median_table_setup_s'])],
        ['original PPPM',1.25,'2.8 / 2.24','sF_req=2.8',16,5,sci(maximum(baseline,'E_G')),ms(oms),ms(old['median_config_setup_s']),ms(old['median_table_setup_s'])]])
    comps=X.read(X.RESULTS/'components_raw.csv')
    comp_rows=[]
    for method in ('B2','PPPM'):
        vals=[]
        for phase in ('real','spread_gather','fft_influence','mesh'):
            vals.append(float(np.median([np.median([r['seconds'] for r in comps if r['method']==method and r['phase']==phase and r['config']==cfg['name']]) for cfg in X.configurations()])))
        comp_rows.append([method]+[ms(v) for v in vals])
    top=sorted([r for r in times if r['method']=='B2'],key=lambda r:r['median_matvec_s'])[:10]
    top_table=table(['xi','sR','M','p','max EΓ','common serial ms'],[[r['xi'],r['sR'],r['M'],r['p'],sci(r['max_E_G']),ms(r['median_matvec_s'])] for r in top])
    fields=('E_R','E_interp','E_table_images','E_table_quad','E_mesh_including_table','far_split_residual','telescope_residual')
    dtable=table(['error component (max over nine)','relative spectral norm'],[[k,sci(maximum(decomp,k))] for k in fields])
    cal=X.read(X.RESULTS/'calibration.csv')
    qp=[]
    for xi,p in sorted({(r['xi'],r['p']) for r in cal}):
        a,b=sorted([r for r in cal if r['xi']==xi and r['p']==p],key=lambda r:r['M'])
        qp.append((p,float(np.log(a['E_M']/b['E_M'])/np.log(2))))
    q2=[q for p,q in qp if p==2];q3=[q for p,q in qp if p==3]
    endpoints=[]
    for sw in ('sR','sT','M','p','rho','kappa'):
        rr=[r for r in rows if r['sweep']==sw and r['setting']=='baseline']
        xx=sorted({r['value'] for r in rr})
        ee=[max(r['E_G'] for r in rr if r['value']==v) for v in xx]
        endpoints.append([sw,xx[0],sci(ee[0]),xx[-1],sci(ee[-1])])
    endpoint_table=table(['baseline sweep','low value','max EΓ at low','high value','max EΓ at high'],endpoints)
    kr=[r for r in rows if r['sweep']=='kappa' and r['setting']=='baseline']
    kpeak=max(sorted({r['value'] for r in kr}),key=lambda v:max(r['E_G'] for r in kr if r['value']==v))
    report=f'''# B2 positive-mesh PPPM：结构 PSD 与两组重跑实验

本次最快的**完整九组验证、共同串行计时**的 B2 参数为 `(xi,sR,κR_T,M,p)=({best['xi']:g},{best['sR']:g},{best['sT']:g},{best['M']},{best['p']})`，`rc={best['sR']/best['xi']:.9g}`。最大 `EΓ={best['max_E_G']:.9g} <= 1e-4`，cached complete matvec 为 **{ms(bms)} ms**。这是自适应离散搜索中的已测最优，未证明连续参数问题的全局最优。

同硬件、同批次原始 PPPM 重测为 **{ms(oms)} ms**，B2/PPPM 为 **{bms/oms:.5f} 倍耗时**。用户给定的 1.268417 ms 仅列作历史基准；本次速度比使用共同重测，未拿跨批次时间作严格加速比。

## 1. 数学保证与实现

完整证明、误差边界、浮点假设见 [proof.md](proof.md)。证明核心是每个有限网格张量都是非负系数的 `rrᵀ` 之和；非负 B-spline 的对称 spread/gather 又给出 PSD 粒子边 `Lij=Σ w_ig T_(g-h) w_jh`。因此

`XᵀΓhX = 1/2 Σ_(i≠j) (Xi-Xj)ᵀ(Sij+Lij)(Xi-Xj) >= 0`。

这适用于任意合法粒子配置和参数，不依赖 eigenvalue tests。Candidate 1 的物理核、splitting 和 `CachedGammaH` short-range 路径保持原样。表由正 Laplace 求积、完整的有限物理镜像窗口和均值/方差的平方和构造；不做 deconvolution、Fourier eigenvalue clipping 或 diagonal repair。FFT 保留实际表的 zero mode；degree 和 vector field 使用同一离散响应，抵消完整的插值 self response（含所有自粒子镜像）。不能只扣 `K_L(0)` 或把表的原点置零。

表、正插值、图的 PSD 是精确算术结构结论。float64 与 FFT 的舍入残差另行测量，不宣称机器数对任意趋零输入都有逐位非负能量。求积加密差是数值估计，不能充作严格 quadrature bound；这些精度限定不影响有限正构造的 PSD 证明。实现需要有限可表示的中间量和足够内存；`kappa=0`、重合粒子等不合法输入不在原奇异物理模型内。

## 2. Experiment 1：六组 accuracy sweeps

`N=256,c=1,rho=.75,kappa=.7,xi=.85`，原 uniform / moderate / strong 三族、seed 1/2/3；共 **927 行、103 个参数点、每点九组**。密度扫描对原位置同比缩放。`sR=2:0.25:6.5`、M、`p=3:9`、`rho=.3,.5,.75,1`、`kappa=.2,.5,.7,1,2` 均沿用原扫描。baseline 为 `(sR,κR_T,M,p)=(6,36,80,9)`，moderate 为 `(4,24,40,7)`。

新 panel (b) 用 **`κ R_T=2,4,...,38`** 替换旧 `sF`。`R_T` 是物理镜像立方体的半宽，不是 Fourier cutoff，也不是最小镜像半径。其尾部按 screened physical tail 控制；旧 Fourier 尾部按 Gaussian `exp(-sF²)` 控制，数值不能直接等同。所有 FFT 模式始终保留。`κR_T=36` 的保守表尾界约为谱归一化后的 `1e-12` 量级，远小于此实验的插值误差；增大表窗口不能消除 smoothing bias。

![B2 operator six sweeps](figures/fig1_operator_six_sweeps.png)

![B2 action six sweeps](figures/fig2_action_six_sweeps.png)

颜色和 marker 沿用原三族，实线/实心与虚线/空心分别是 baseline/moderate，线为三 seed 中位数、色带为 min–max。算子误差统一为 `||Γh-ΓD||₂/||ΓD||₂`；action 误差使用同一组十个 Gaussian probes 的真实 FFT matvec。原六联图保留在 [原始 accuracy_sweeps](../accuracy_sweeps/report.md)。直接对比如下：

{endpoint_table}

提高 sR 或表半宽只在 cutoff 主导区降低误差，随后达到固定 M/p 的 smoothing plateau；继续加大 cutoff 不会恢复原 PPPM 的高精度。固定 M 时，提高密度使 L 和 h 变小，本次受控缩放扫描中误差下降；物理 κL 和参考算子范数也同时改变，不能把这当作所有构型的一般密度定律。κ 扫描的相对误差并不单调，baseline 的最大值出现在 κ={kpeak:g}；改变 κ 既改变核的 long/short 比例，也改变归一化范数。选参阶段始终固定 κ=.2，没有借改变物理 screening 降低成本。

![B2 versus original PPPM](figures/fig3_original_pppm_comparison.png)

同样的 `xi=.85,M=80,p=9` baseline，B2 最大 EΓ 为 `{maximum(base,'E_G'):.8g}`，原 PPPM 为 `{maximum(oldbase,'E_G'):.8g}`。提高 p 在 B2 中会增加正插值的 smoothing bias；不能沿用 deconvolved PPPM 的 `O(h^p)` 模型。完整六联扫描的高分辨率 mesh-only 拟合为：

{rates}

用于 κ=.2 选参的额外校准中，p=2 的 M32→64 阶数范围为 `{min(q2):.4f}–{max(q2):.4f}`，p=3 为 `{min(q3):.4f}–{max(q3):.4f}`；p=1 仅接近一阶。高 p 不恢复高阶精度。mesh-only 比较使用 `ΓD-ΓS_far`，不把完整算子的 cutoff plateau 误当作 mesh 收敛阶数。

所有 927 行的 `lambda_min`（平移正交补）最小为 `{min(r['lam_min_h'] for r in rows):.8g}`；最大 symmetry 残差 `{maximum(rows,'sym_resid'):.3g}`，matrix-free/direct-stencil dense 差 `{maximum(rows,'mf_vs_dense'):.3g}`，平移 matvec 残差 `{maximum(rows,'mf_null'):.3g}`。这些是回归验证，结构证明在前。

## 3. Reference 与网格核构造误差

72 份独立 `unsplit direct periodic annulus_tensors` reference 按 `(kappa,c,rho,L,position hash)` 校验并记录 SHA-256。已有 κ=.2/.7 缓存只读复用，其余状态点补建；没有使用 B2 自身或 truncation-matched Ewald 当 reference。每份 direct sum 使用原 physical `exp(-kappa*r)/r`，认证相对尾界不超过 `1e-13`。κ=.2 的 `R_D=190`；所有 reference 生成结束后才开始最终计时。历史 reference 的范数只用于选择求和半径和独立校核，没有替代矩阵。

正表在 GL8 与 GL16 间的最大 component-sum 差为 `{maximum(construction,'max_component_sum_quad8_vs16'):.3g}`，遗漏小尺度区间的解析上界不超过 `{maximum(construction,'omitted_small_scale_bound'):.3g}`（表张量绝对误差）。uniform image-tail 解析界、Nyquist/反射处理与误差传递 `||δΓ||<=2(N-1) max||δT||` 见证明。construction audit 同时显示 Fourier multiplier 最小特征值为负；这是允许的，不能对它逐模开协方差平方根。

## 4. Experiment 2：理论初始化、实测约束与局部成本优化

固定 `kappa=.2,rho=.75,c=1,N=256`。先用 `sqrt(log(1/epsilon))≈3.035` 初始化 short cutoff 尺度；以 Experiment 1 已完成的 M 扫描和 Taylor 展开确定二阶规律，再在 κ=.2 对 p=2/3/4 校准 mesh 系数和实际阶数。初始化模型为 `E_M(M)=E_M(64)*(64/M)^q`，仅用于建议 M。完整九组结果随后确认该规律；初始扫描时使用已完成的 uniform 族拟合。没有把拟合预测视作严格误差证书。

粗搜索 xi 为 `.2,.3,.4,.5,.6,.7,.85,1,1.25`，主要测试 p=2/3，p=4 在 .3/.5/.7 切片；M 取预测值的 .82/1/1.2 附近 FFT-friendly 值，测试上限 160，short cutoff 初始为 2.6/2.8/3/3.3。p=1 在 M32/64 的校准保留，误差和一阶规律使其不进入此次有限成本搜索。局部步骤在实测可行 incumbent 周围改变 xi±.05、sR±.1、M×.88/1/1.10 并检查 p2/3。随后专门检查 M54 附近的 cutoff 可行性边界，以及 xi=.275/.325/.375、M60/64/72/75 的相邻耦合点。搜索域、未测点和参数离散性限制了“最优”的含义。

共记录 **{len(search)} 个不同候选**；先在 uniform seed 2 排除完整 EΓ 已超标者，再验证 **{len(summary)} 个候选的全部九组**，其中 **{sum(bool(r['feasible']) for r in summary)} 个满足 max EΓ<=1e-4**。临时三次 matvec 中位数指导局部成本选择；其间可能有 accuracy 工作，不能用该批的绝对时间作最终速度结论。所有 accuracy 工作结束后，对每个九组可行候选先在同一配置共同串行重测三轮×三次，消除粗/细搜索期间负载不同带来的排名偏差；再对领先的十二个候选及每个 order 的最佳对照做完整九组共同长计时。全部搜索值、失败点和 validation 行均保留。`screen_cost_model.json` 另存 `Nimages,M³log M³,Np³` 的非负成本拟合，仅作噪声下的解释性预测。

{top_table}

## 5. 同硬件比较、setup cost 与实际代价

{maincompare}

两者均为原 cached complete action，包括 short-range、degree 乘法和 mesh response。setup 分为参数表（可跨同参数配置复用）与配置相关 geometry/stencil/degree；两项各做三次 fresh 构建，中间不含 reference 或 dense validation。共同 kernel primitive 初始化单独记录在 `common_kernel_setup_raw.csv`。用户历史最优的 EΓ=8.51354478e-5、1.268417 ms 与新重测分别列出，历史计时未被覆盖。

{table(['method','real ms','spread+gather ms','FFT+influence ms','mesh incl degree ms'],comp_rows)}

分阶段中位数不要求精确相加；最终目标始终是完整 action。B2 最优的 near image 数中位数为 `{int(np.median([r['n_images'] for r in br]))}`，原最优历史约 4822；网格点数比为 `{(best['M']/16)**3:.5f}`，assignment 工作量比为 `{(best['p']/5)**3:.5f}`。降低 p 减轻 spread/gather，但二阶正插值偏差迫使更小 xi / 更大网格，增加 near 与 FFT 成本。

本轮每候选每配置 `{protocol['rounds']}×{protocol['reps']}={protocol['rounds']*protocol['reps']}` 样本，单计算线程，candidate blocks 每轮随机排列，3 次 block warmup、每样本前 1 次 warmup。目标统计量为九个配置中位数的中位数。B2 最大相对 IQR 为 `{best['max_relative_iqr']:.3%}`，原 PPPM 为 `{old['max_relative_iqr']:.3%}`。配对 configuration/round block 的 B2/PPPM 耗时比 bootstrap 95% 区间为 `{protocol['b2_over_pppm_block_bootstrap_95']}`；它条件于已选候选，未校正多重选参。系统其他工作负载不能保证为零；原始重复样本全部提供。

![Parameter selection and cost](figures/fig4_optimization_cost.png)

最终误差分解使用同一 split 的 far-short `sR=8` 和独立 mesh-free Fourier `sF=7` 极限交叉校核；表半宽加至 κR_T=48、求积加至 GL16 以分别分离表/插值误差。以下各项为各配置最大值，范数不能直接按标量相减，各最大值也可能来自不同配置。

{dtable}

**结论：B2 在这九组、同一物理模型上达到了 1e-4；代价显著。** 最优已测完整 matvec 比共同重测原 PPPM 慢 `{bms/oms:.4f}` 倍，配置 setup 与参数表成本也已列明。对本次 N=256、以极快 repeated matvec 为主要目标的场景，这不能称为低成本替代；其收益是对任意合法配置的结构 PSD 证书。没有放宽精度或更改 κ 来换速度。数值精度仍只对已验证配置和已测参数作结论；结构 PSD 则不局限于这些样本。

## 6. 保留与复现

原始实现及全部 tracked artifacts 的 **{audit['historical_files_checked']} 份 SHA-256 全部未变**。B2 源码、结果与图位于独立目录。数据表、绘图中位数/范围、输入 references SHA、命令日志、参数搜索失败点及计时原始样本均保存。复现命令见 [README.md](README.md)。测试覆盖非负 B-spline、奇偶网格、周期边界、zero mode、self cancellation、直接镜像积分、FFT/direct-stencil 作用一致性及非法输入。

环境：`{protocol['provenance']['platform']}`，Python `{protocol['provenance']['python']}`、NumPy `{protocol['provenance']['numpy']}`、SciPy `{protocol['provenance']['scipy']}`；最终 serial timing 用时 `{protocol['elapsed_s']/60:.2f}` 分钟。详细线程、源文件 hash 与 provenance 在 results 中。
'''
    (X.HERE/'report.md').write_text(report)
    code={str(p.relative_to(C.REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(X.HERE.glob('*.py'))}
    source=C.HERE/'positive_mesh.py';code[str(source.relative_to(C.REPO))]=hashlib.sha256(source.read_bytes()).hexdigest()
    X.jsave(X.RESULTS/'final_audit.json',dict(sweep_rows=len(rows),parameter_points=len(counts),references=len(refs),
        search_candidates=len(search),validated_candidates=len(summary),best=best,source_sha256=code,
        historical_modified=[],all_sweep_translation_complement_eigenvalues_positive=True,
        proof_not_based_on_eigenvalues=True,accuracy_threshold=X.TOL))
    X.log('report generated; completeness and historical checks pass')

if __name__=='__main__':main()
