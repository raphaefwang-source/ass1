# 本地 BAOAB A/B 联合选参

本目录从 PR11 `codex/baoab-ab-matched-accuracy` 的 `891994a02b611833848839b5de9880f9dabc7a1d` 起步。工作分支 `codex/baoab-ab-joint-selection`。任务原文保存在 [requirements_github.md](requirements_github.md)。所有结果由当前本地 Mac 计算，历史 PR3–11 文件只读保留。

目标是在 `N=256,kappa=.7,c=.5*1.3*exp(.7*1.3),rho=.75,dt=.005`、共同 `2e-7` 空间 Γ 预算及 `epsilon_frozen={1e-4,1e-6}` 下，独立优化 A/B 联合空间/thermostat 参数，按真正移动粒子的完整 BAOAB wall / simulated physical time 排名。完整说明和结果见 [report.md](report.md)。这不是物理观测量等精度或全局最优认证；可选 `2e-8` frozen stress、长轨迹、HPC 与不同 dt 优化没有执行。

## 环境与保存目录

在当前 checkout 的本目录运行。Python 3.11，NumPy、SciPy、Matplotlib、psutil、threadpoolctl，以及 Apple Clang 用于复用既有 compiled Pairwise OU；本次实际版本在 `results/environment_initial.json`。`joint_common.py` 在数值库导入前将线程环境请求设为 1，实际可观测约束另留环境/资源记录。

运行前确认 `pwd`、`git branch --show-current`、`git rev-parse HEAD`，检查可用依赖并保存初始环境。不要在正式动态计时期间并行执行 dense/reference、测试或绘图工作。所有命令均本地运行，不需要 SSH、HPC 或远程 compute。

- `results/spatial/`：34 个候选 screening、拒绝原因、自适应选择、25 次 full COM 空间核查、4 个合格设置。
- `results/physical_refs/`：实际新增几何的独立 R45 direct 参考与保守尾界；已有物理参考通过源路径/SHA 只读复用。
- `results/thermo/`：五态完整 physical/own-Gamma_h OU norm、实际 executed k、最小已测通过 k、独立 damping/noise 秩搜索、finite path 与 wrapper 数组。
- `inputs/`：实际 moving trajectories；`results/*_steps.csv`、`*_substeps.csv`、`*_rng.json` 保存各外/内步时钟、q_mid/噪声哈希、before/after RNG 状态和资源。
- `results/post_certificates/`：正式 head 后实际 midpoint 的完整物理空间/OU、真实输入/噪声 finite 路径证据。
- `figures/`：PNG/SVG、plotted CSV 和 SHA manifest；`logs/`：实际命令日志。
- `baseline_snapshot.json`、`results/independent_audit.json`：上游 tracked 文件不变与本轮数据独立审计/全文件证据清单。

## 原样复现顺序

已完成结果有覆盖保护。原样结果应先审计；重新数值运行请在从同一 upstream SHA 建立的新 checkout 中，只复制本目录源码、测试、任务与环境入口，保留旧目录为只读依赖，并为新输出建立空目录。不要删除本次实际结果或向历史实验目录写入。重跑会产生新的时钟和 SHA，旧速度数字不作为重跑比值输入。

```sh
cd benchmarks/method_A_clustered/baoab_ab_joint_selection
mkdir -p inputs results logs figures
python3 -B capture_environment.py > logs/capture_environment.log 2>&1
python3 -B run_regressions.py > logs/baseline_regressions.log 2>&1
python3 -B -m unittest -v test_joint_spatial > logs/test_joint_spatial.log 2>&1
python3 -B -m unittest -v test_joint_thermo > logs/test_joint_thermo.log 2>&1
python3 -B -m unittest -v test_joint_driver > logs/test_joint_driver.log 2>&1
python3 -B -m unittest -v test_joint_postvalidate > logs/test_joint_postvalidate.log 2>&1
python3 -B -m unittest -v test_joint_plots > logs/test_joint_plots.log 2>&1
python3 -B -m unittest -v test_independent_audit > logs/test_independent_audit.log 2>&1
python3 -B joint_spatial.py > logs/joint_spatial.log 2>&1
python3 -B joint_thermo.py ideal > logs/joint_thermo_ideal.log 2>&1
python3 -B joint_thermo.py ranks > logs/joint_thermo_ranks.log 2>&1
python3 -B joint_driver.py explore --steps 3 > logs/exploratory.log 2>&1
python3 -B joint_driver.py groups --baseline-id sp_b4884e6cd3 > logs/head_groups.log 2>&1
python3 -B joint_driver.py heads --rounds 6 --steps 20 > logs/heads.log 2>&1
python3 -B provenance.py > logs/backend_provenance.log 2>&1
python3 -B joint_postvalidate.py > logs/post_validation.log 2>&1
python3 -B joint_analyze.py > logs/analysis.log 2>&1
python3 -B joint_plots.py > logs/plots.log 2>&1
python3 -B report_generator.py > logs/report.log 2>&1
python3 -B independent_audit.py --replay > logs/independent_audit.log 2>&1
```

本次空间搜索先保存三 finalist，又以 `joint_spatial.py --extend-resolution-leader` 对同 xi 的低分辨率点补充四态完整核查。当前默认源码已把该点优先保留纳入四 finalist 选择；历史执行顺序、旧三点选择和补充原因也保存在结果/日志，默认干净重跑无需重复补充命令。

## 精度与统计解释

空间通过证据是五个注册几何上的完整 765 维 COM 算子范数，并含 direct tail 的保守 allowance；cheap 方向下界只可拒绝，不能给通过证书。独立 physical-reference mean/covariance 分别过门限，同时记录 own-Gamma_h splitting 和空间 transition 分量。五态为三初态、旧 visited 全局最小距离与旧最大 lambda_max 态，不能把只读登记的旧十态域称为候选十态全验证。

每 setting 自适应测 k=1、2、4…，严格必需档通过后停止；未执行 k 显式列出。每配置重新选取有限 Lanczos 两种秩，使用实际输入、谱/低波数方向、完整独立 Gaussian 子循环及 production wrapper 检查。有限路径误差不等于输入相关非线性噪声映射的统一 covariance 算子误差。

Exploratory 排名每配置三初态各三步，选出各方法/档实际完整成本最低点；正式组区分独立选参和共同旧基线，六配对轮每方法各二十步，A-first/B-first 平衡。全套正式计时结束后才执行 direct/dense 末端验证，取每初态最后一轮 A/B 实际 midpoint，重放保存的真实 PCG64 噪声并检查实际 action/路径。相同实际配置组显式复用本轮批次，不伪造独立样本。

`analysis.json` 和 `pareto.csv` 只使用本轮新时钟；CI 按配对轮 bootstrap，空间/冻结/finite 末端门控失败时不作合格 winner 结论。Pareto 为同方法/档内成本—空间误差—冻结误差的短程经验前沿；灵敏度图是多参数同时变化的候选族，不是单因素因果实验。最终仍需另行验证 VACF/MSD/RDF/pressure/温度等物理观测误差并允许各方法优化 dt，才能讨论物理等精度 winner。
