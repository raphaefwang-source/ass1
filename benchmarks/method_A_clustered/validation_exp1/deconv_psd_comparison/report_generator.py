"""Generate the Chinese research report using saved observations only.

This never runs an experiment, imports its numerical implementation, or reads a
historical timing as a new measurement. Unavailable results stay explicitly
unavailable. Re-run after checkpoints finish to refresh report.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
METHODS = ("PPPM-original", "B2-positive", "PPPM-grid-PSD")
LABEL = {"PPPM-original": "Original PPPM", "B2-positive": "B2 Positive Mesh",
         "PPPM-grid-PSD": "Deconvolved PPPM + PSD Projection"}
INPUTS = []


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def read(path):
    if not path.exists():
        return []
    INPUTS.append(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, val in list(row.items()):
            if key in ("id", "reference_id", "method", "reference_method", "family", "config", "sweep", "setting",
                       "source", "source_path", "stage", "status", "failure", "phase", "selection", "note", "timing_batch"):
                continue
            num = number(val)
            if num is not None:
                row[key] = num
    return rows


def read_json(path):
    if not path.exists():
        return {}
    INPUTS.append(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"read_error": "JSON checkpoint not complete"}


def med(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    n = len(values)
    return values[n//2] if n % 2 else .5*(values[n//2-1]+values[n//2])


def fmt(value, digits=4):
    value = number(value)
    return "N/A" if value is None else f"{value:.{digits}g}"


def ms(value):
    value = number(value)
    return "N/A" if value is None else fmt(value*1e3)


def mib(value):
    value = number(value)
    return "N/A" if value is None else fmt(value/(1024**2))


def escaped(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    if not rows:
        return "尚无已保存的观测数据。\n"
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output += ["| " + " | ".join(escaped(v) for v in row) + " |" for row in rows]
    return "\n".join(output) + "\n"


def max_field(rows, field):
    vals = [number(r.get(field)) for r in rows]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def min_field(rows, field):
    vals = [number(r.get(field)) for r in rows]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def range_text(rows, key):
    a, b = min_field(rows, key), max_field(rows, key)
    return "N/A" if a is None else (fmt(a) if a == b else f"{fmt(a)}–{fmt(b)}")


def mismatch_counts(value, prefix=""):
    output = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if "mismatch" in key.lower() and isinstance(child, list):
                output.append([name, len(child)])
            elif isinstance(child, dict):
                output.extend(mismatch_counts(child, name))
    return output


def fit_regime(fit, convergence):
    if fit.get("method") != "PPPM-grid-PSD":
        return "不适用"
    selected = [r for r in convergence if r.get("method") == fit.get("method")
                and r.get("family") == fit.get("family") and r.get("p") == fit.get("p")
                and number(r.get("M")) is not None
                and fit.get("Mlo", math.inf) <= r["M"] <= fit.get("Mhi", -math.inf)]
    if not selected or any(number(r.get("negative_blocks", r.get("neg_fraction"))) is None for r in selected):
        return "未记录"
    active = sum((number(r.get("negative_blocks", r.get("neg_fraction"))) or 0) > 0 for r in selected)
    return f"{'inactive' if active == 0 else 'active' if active == len(selected) else 'mixed'} ({active}/{len(selected)} active)"


def same_parameters(left, right, keys=("xi", "sR", "sF", "M", "p")):
    return all(number(left.get(k)) is not None and number(right.get(k)) is not None
               and abs(number(left[k])-number(right[k])) <= 1e-10 for k in keys)


def image_section(figures, name, caption, link_dir=None):
    path = figures / f"{name}.png"
    if path.exists():
        # Report lives beside figures by default; absolute override is also supported.
        link = ((link_dir / f"{name}.png") if link_dir is not None else path).as_posix()
        return f"\n![{caption}]({link})\n\n{caption}。"
    return f"\n图 `{name}`：尚未生成。"


def quality_status(tables, plots):
    sweeps = tables["sweeps"]
    conv = tables["convergence"]
    counts = {m: sum(r.get("method") == m for r in sweeps) for m in METHODS}
    expected = 927
    sweep_done = all(counts[m] == expected for m in METHODS)
    # The prescribed grid has 6 M for p=2,3 and 8 M for p=5,7,9, for 9 configurations.
    expected_conv = 3 * 9 * (2*6 + 3*8)
    conv_done = len(conv) == expected_conv
    validated = [r for r in tables["validation_summary"] if number(r.get("n_configs")) == 9
                 and number(r.get("max_E_G")) is not None and r["max_E_G"] <= 1e-4]
    timing_done = all(any(r.get("method") == m and number(r.get("max_E_G")) is not None
                         and r["max_E_G"] <= 1e-4 for r in tables["best_configuration"])
                      for m in METHODS)
    return dict(sweep_done=sweep_done, sweep_counts=counts, expected_sweeps_per_method=expected,
                convergence_done=conv_done, convergence_rows=len(conv), expected_convergence_rows=expected_conv,
                validated_main_methods=sorted({r.get("method") for r in validated}),
                timing_done=timing_done, figures_generated=len(plots.get("generated", [])),
                figures_skipped=plots.get("skipped", []), controlled_xi_rows=len(tables.get("xi_sweep", [])),
                controlled_xi_done=len(tables.get("xi_sweep", [])) == 135)


def generate(args):
    names = ("sweeps", "convergence", "mesh_rate_fits", "mesh_local_slopes", "mesh_inactive_rate_fits", "search", "validation", "validation_summary",
             "final_timing", "best_configuration", "components_raw", "frontier", "paired_speed_ratios",
             "memory", "table_setup_raw", "config_setup_raw", "final_timing_raw", "table_diagnostics", "assignment_weights_audit", "xi_sweep", "cross_winner_summary", "best_projection_diagnostics")
    tables = {name: read(args.results / f"{name}.csv") for name in names}
    tables["best_configuration"] = [r for r in tables["best_configuration"]
            if number(r.get("n_configs")) == 9 and number(r.get("max_E_G")) is not None
            and r["max_E_G"] <= 1e-4 and number(r.get("median_matvec_s")) is not None
            and r["median_matvec_s"] > 0]
    if not tables["frontier"]:
        tables["frontier"] = read(args.results / "accuracy_time_frontier.csv")
    if not tables["best_configuration"] and tables["final_timing"]:
        # Restrict fallback to the same saved common timing table.
        tables["best_configuration"] = [min(selected, key=lambda r: r["median_matvec_s"])
             for method in METHODS
             if (selected := [r for r in tables["final_timing"] if r.get("method") == method
                              and number(r.get("max_E_G")) is not None and r["max_E_G"] <= 1e-4
                              and number(r.get("n_configs")) == 9
                              and number(r.get("median_matvec_s")) is not None])]
    audit = read_json(HERE / "audit_initial.json")
    initial_state = read_json(HERE / 'initial_state.json')
    final_audit = read_json(HERE / "audit_final.json") or read_json(args.results / "audit_final.json")
    protocol = read_json(args.results / "timing_protocol.json")
    primary_dir = args.results / "primary_timing"
    primary_protocol = read_json(primary_dir / "timing_protocol.json")
    primary_best = read(primary_dir / "best_configuration.csv")
    primary_timing = read(primary_dir / "final_timing.csv")
    primary_raw = read(primary_dir / "final_timing_raw.csv")
    if primary_dir.exists():
        # Retain provenance for both batches without mixing raw samples.
        INPUTS.extend(path for path in primary_dir.glob("*") if path.is_file())
    protocol_complete = (number(protocol.get("threads")) == 1 and number(protocol.get("n_configs")) == 9
                         and (number(protocol.get("rounds")) or 0) >= 1 and (number(protocol.get("reps")) or 0) >= 1)
    theory = read_json(args.results / "search_theory.json")
    table_audit = read_json(args.results / "table_audit_summary.json")
    budgets = read_json(args.results / "search_budget.json")
    plots = read_json(args.figures / "plot_status.json")
    status = quality_status(tables, plots)
    status["protocol_complete"] = protocol_complete
    best = {r["method"]: r for r in tables["best_configuration"]}
    lines = []
    add = lines.append
    now = datetime.now(ZoneInfo("America/Detroit")).isoformat(timespec="seconds")
    add("# 去卷积 PPPM 的实空间网格张量 PSD 投影：三方法实验报告\n")
    add(f"生成时间：{now}（America/Detroit）。本报告由 `report_generator.py` 读取现有 CSV 自动生成，"
        "未完成的计算保留为未完成，所有数值均来自已保存观测。")
    add("完整要求见 [GitHub 评论](https://github.com/raphaefwang-source/ass1/pull/3#issuecomment-6431287227)"
        "及本目录 `requirements_github.md`。")
    add("\n## 当前交付状态\n")
    rows = []
    for m in METHODS:
        rows.append([m, status["sweep_counts"][m], status["expected_sweeps_per_method"],
                     sum(r.get("method") == m for r in tables["convergence"]),
                     sum(r.get("method") == m for r in tables["search"]),
                     "完成" if m in best else "未完成"])
    add(table(["方法", "六扫记录", "预期六扫记录", "独立网格收敛记录", "新搜索候选", "主档同批计时"], rows))
    add(f"网格收敛记录总计 {status['convergence_rows']}/{status['expected_convergence_rows']}；"
        f"补充受控 xi 记录 {status['controlled_xi_rows']}/135；图表已保存 {status['figures_generated']} 组。"
        "只有三方法六扫、独立收敛、九配置验证与同批计时均完成后，主实验才完整。")
    if not (status["sweep_done"] and status["convergence_done"] and status["timing_done"] and protocol_complete
            and status["controlled_xi_done"]):
        add("**当前仍为检查点报告，包含尚未完成项；不能视为已经完成全部实验。**")
    add("\n## 1. 模型、基线和历史数据保护\n")
    add("三个方案采用相同 Candidate-1 / law-A 分裂、原 amplitude 与周期核、短程图 action、位置与 seed。"
        "`PPPM-original` 保留旧高阶去卷积；`B2-positive` 使用无去卷积正实空间表与非负分配；"
        "`PPPM-grid-PSD` 对原去卷积实空间表逐 3×3 张量做 PSD 投影。没有更改物理 kappa，"
        "没有 Fourier clipping、全局 delta*P 修正或 OU 实验。")
    git = audit.get("git", {})
    add(f"初始分支 `{initial_state.get('initial_branch',git.get('branch', '未记录'))}`；初始 HEAD `{git.get('commit', '未记录')}`。"
        f"独立新分支为 `{initial_state.get('new_branch',git.get('branch', '未记录'))}`。"
        "新代码、CSV、图与报告均位于独立 `deconv_psd_comparison/`；历史源码、结果与 reference 只读复用。")
    manifest_rows = []
    for key, label in [("original_artifact_manifest", "原始历史产物"), ("b2_source_manifest", "B2 源文件"),
                       ("original_reference_manifest", "原始 reference")]:
        record = audit.get(key, {})
        manifest_rows.append([label, record.get("checked", "N/A"), len(record.get("mismatches", [])) if record else "N/A"])
    refs = audit.get("references", {})
    manifest_rows.append(["独立 direct reference", refs.get("count", "N/A"), len(refs.get("mismatches", [])) if refs else "N/A"])
    add(table(["初始核对项", "文件数", "不匹配数"], manifest_rows))
    if final_audit:
        add("结束时审计文件已存在，详见 `audit_final.json`；具体逐文件 hash 记录为判断历史内容是否变化的依据。")
        add(table(["结束时 hash 检查项", "不匹配数"], mismatch_counts(final_audit)))
    else:
        add("结束时历史文件 hash 复核：尚未保存 `audit_final.json`，需在全部任务结束后补齐。")
    add(f"Direct reference 为独立 unsplit periodic annulus sum；初始审计中最大相对证书界为 "
        f"{fmt(refs.get('max_cert_bound_rel'))}。reference 位置 hash、物理参数及来源在 `audit_initial.json` 中逐一核对。")
    env = audit.get("environment", {})
    add(f"环境：{env.get('platform', '未记录')}；Python {env.get('python', 'N/A')}；"
        f"NumPy {env.get('numpy', 'N/A')}；SciPy {env.get('scipy', 'N/A')}；"
        f"逻辑 CPU {env.get('cpu_count', 'N/A')}；物理内存 {mib(env.get('memory_bytes'))} MiB。")
    add("\n## 2. 实现及 PSD 证明\n")
    add("旧 influence `G_old=Khat_L/|What_h|²` 仍沿用原 kc、Nyquist 与安全分母规则。"
        "adapter `grid_psd.py` 使用物理张量 `T_raw=irfftn(G_old)/h³`；"
        "六个分量构成对称 3×3 块，先对 d 与 −d 求平均，再把**全部负特征值**截到零。"
        "修正按 negative part 直接构造，避免两个近似相等的重建张量相减造成污染；"
        "diagnostics 的 roundoff threshold 只用于统计，未用于丢弃修正。")
    add("`G_plus=G_old+h³*rfftn(T_plus−S)` 与 `h³*rfftn(T_plus)` 在精确算术中等价。"
        "当修正为零时保留原 influence，避免无意义 FFT round-trip 舍入；保留投影新产生的所有 modes、"
        "Nyquist 面与零频，不重施 kc mask、不再次去卷积、不把实空间原点置零。"
        "degree 使用新 influence 与同一套非负 B-spline spread/gather 重新构造；"
        "完整 interpolated self response 在 degree-minus-response 中代数抵消。")
    add("令 `A_ij⁺=Σ_(g,l) w_ig T_plus[g−l] w_jl`。因每个 `T_plus[d]` 是 PSD 且 "
        "`w_ig,w_jl≥0`，所以每个 `A_ij⁺` 是 PSD；因 `T_plus[d]=T_plus[−d]=T_plus[d]ᵀ`，"
        "边权满足 `A_ij⁺=A_ji⁺`。于是对任意粒子向量 v，")
    add("\n```text\nvᵀ Γ_Lh⁺ v = (1/2) Σ_(i,j) (v_i−v_j)ᵀ A_ij⁺ (v_i−v_j) ≥ 0.\n```\n")
    add("因此新长程 graph 在精确算术中严格 PSD，并保留三维 translation nullspace。"
        "Candidate-1 短程边核也为 PSD，完整短加长 graph 同样具有该结构。"
        "这是结构证明；float64 的表重建、degree 与 FFT 的舍入误差单列报告，"
        "不把有限配置上的最小特征值采样当成证明。")
    add("完整归一化、结构证明与近 rank-one 核二阶偏差机制分析见 [proof.md](proof.md)。"
        "该机制分析忽略 alias 与有限截断，仅解释可能的误差来源，不充当本实验的任意参数高阶或二阶定理。")
    add("原与新使用数学上同一 B-spline 分配；新方案采用 B2 稳定正 recurrence 和 partition normalization。"
        "因此即使表投影完全不活跃，新旧粒子算子在浮点中仍可能相差舍入量；"
        "`E_proj≈机器精度` 不应解读为数学投影活跃。参数表随 L、核、xi、sF、M、p 变化而重建；"
        "同一参数表可跨位置配置复用；位置改变需重建 stencil、degree 与短程邻居缓存。")
    add("\n## 3. 独立验证\n")
    add("`test_grid_psd.py` 覆盖已知 indefinite/PSD 块、微小负特征值不丢弃、"
        "无投影 round-trip 与旧 action、奇偶网格/Nyquist、非负与归一化 stencil、周期边界与 p=1 半格点、"
        "零 amplitude 与非法输入、全 mode 的 DFT/独立 direct-stencil contraction、"
        "新 degree、完整 action 的对称/translation-null、单粒子 self cancellation。"
        "投影后 dense 真值使用旧有限 mode sum **加完整实空间稀疏修正**，"
        "不把旧 mode list 当成修复后算子的全部真值；小网格测试另用 literal stencil 与完整 DFT 独立交叉核对。")
    test_paths = sorted(set(args.results.glob("*test*.log")) | set(HERE.glob("*test*.log")))
    if test_paths:
        test_rows = []
        for path in test_paths:
            INPUTS.append(path)
            text = path.read_text(encoding="utf-8", errors="replace")
            count = re.findall(r"Ran\s+(\d+)\s+tests?", text)
            result = "通过" if re.search(r"^OK\s*$", text, re.M) and not re.search(r"^FAILED", text, re.M) else "需核对日志"
            test_rows.append([path.name, count[-1] if count else "N/A", result])
        add(table(["测试日志", "测试数", "结果"], test_rows))
    else:
        add("测试源码已存在，但本报告尚未找到本次测试日志；不能据此声称测试已通过。")
    if table_audit:
        add(f"独立 table audit 完成 {table_audit.get('complete_tables', 'N/A')}/"
            f"{table_audit.get('distinct_tables', 'N/A')} 个不同参数表，非法表 {table_audit.get('invalid_tables', 'N/A')}；"
            f"其中 {table_audit.get('direct_checked_tables', 'N/A')} 个参数表用小 N literal 3D stencil 对照。"
            f"最大 direct/FFT-basis 相对差 {fmt(table_audit.get('max_direct_vs_fft_relative'))}，"
            f"最大 direct graph action 相对差 {fmt(table_audit.get('max_direct_graph_relative'))}。"
            "逐表、零频、新 mode、负谱与浮点重建残差见 `results/table_diagnostics.csv`；"
            "这些校验耗时不作为公平 matvec 时间。")
        add(f"全部 audit 表的最小 post-clipping eigenvalue 为 "
            f"{fmt(min_field(tables['table_diagnostics'], 'min_eig_after'))}；"
            "谱剪裁后的微小负值按算子尺度评估为浮点重建误差，数学 PSD 结论来自证明。")
    weights = tables["assignment_weights_audit"]
    if weights:
        add(table(["p", "配置数", "一维权重最大差", "三维乘积权重最大差", "无投影 apply 最大相对差"], [
            [fmt(r.get("p")), fmt(r.get("configurations")), fmt(r.get("axis_weight_max_abs")),
             fmt(r.get("product_weight_max_abs")), fmt(r.get("apply_max_relative"))] for r in weights]))
        add("上表是稳定非负 recurrence 与旧 Mesh assignment 的实际量化对照；"
            "p1 精确半格点的历史 tie 歧义单独用测试覆盖，不从随机配置的一致性推定。")
    validations = tables["validation"]
    add(table(["方法", "完整验证记录", "最小 restricted λ(full)", "最小 restricted λ(long)",
               "最大 symmetry residual", "最大 translation residual", "最大 FFT/dense discrepancy"], [
        [m, len(rr), fmt(min_field(rr, "lam_min_h")), fmt(min_field(rr, "lam_min_long")),
         fmt(max_field(rr, "sym_resid")), fmt(max_field(rr, "mom_opnorm_max")), fmt(max_field(rr, "mf_vs_dense"))]
        for m in METHODS if (rr := [r for r in validations if r.get("method") == m])]))
    add("restricted λ 在去除 translation 的子空间上计算；此表的未达标候选也保留，"
        "不只筛选成功点。计时只使用完成九配置验证的可行候选。")
    add("\n## 4. Experiment 1：六组参数与高阶精度\n")
    add("沿用 uniform/moderate/strong × seed 1/2/3。主误差为完整矩阵 "
        "`E_G=||Γ_h−Γ_D||₂/||Γ_D||₂`，以对称矩阵极端特征值计算 spectral norm；"
        "`E_GX` 使用原 10 个 Gaussian probes，作为 action 补充而非主约束的替代。"
        "rho/kappa 扫描改变物理设置用于研究规律；选参实验始终固定物理设置。")
    sweeps = tables["sweeps"]
    add(table(["方法", "sR", "sF（原/投影前）", "sT（B2）", "M", "p", "rho", "kappa"], [
        [m] + [sum(r.get("method") == m and r.get("sweep") == sw for r in sweeps)
               for sw in ("sR", "sF", "sT", "M", "p", "rho", "kappa")] for m in METHODS]))
    add("sR=xi*r_c；原/新 sF=k_c/(2xi)，Fourier cutoff 实际按 reciprocal shell 向下吸附；"
        "新方案中的 sF 只限定投影前 influence。B2 的 sT=kappa*R_T 是实空间 image cube half-window，"
        "与 sF 具有不同定义，不因数值相等就视为相同截断。图中将两者分面。"
        "baseline/moderate 与历史扫描保持相同定义；历史结果的 source/path/hash 见导入记录及初始审计。")
    add(image_section(args.figures, "fig1_operator_six_sweeps", "图 1：三方法六组 E_G 参数扫描，颜色统一，带状区域为 seed min–max", args.figure_links))
    add(image_section(args.figures, "fig2_action_six_sweeps", "图 2：对应 Gaussian action 误差 E_GX", args.figure_links))
    add("\n### 受控 xi 扫描：与独立选参分开\n")
    xi_rows = tables["xi_sweep"]
    add("补充扫描固定 N=256、rho=.75、kappa=.2、c=1、M=20、p=5、sR=3；"
        "原/新 sF=3，B2 sT=36。三方法均取 xi∈{.35,.6,.85,1,1.25}，"
        "每个 xi 覆盖三个配置族 × 三个 seed，共预期 135 条记录。"
        "这组曲线用于隔离共同数值设置下 xi 的影响，"
        "与 Experiment 2 中各方法独立优化 xi/M/p/截断参数的最快配置分开解读。"
        "固定的是无量纲 sR/sF，因此 xi 改变时派生物理截断 rc=sR/xi、kc≈2*xi*sF 也随之改变；"
        "B2 的 image-table 窗口仍为 sT/kappa。")
    add(table(["方法", "已保存/预期记录", "xi 范围", "E_G 范围", "E_GX 范围", "max E_proj"], [
        [method, f"{len(rr)}/45", range_text(rr, "xi"), range_text(rr, "E_G"), range_text(rr, "E_GX"),
         fmt(max_field(rr, "E_proj"))] for method in METHODS
        if (rr := [r for r in xi_rows if r.get("method") == method])]))
    add(image_section(args.figures, "fig9_xi_controlled_sweep", "图 9：受控 xi 扫描，E_G/E_GX × 配置族分面；曲线为 seed 中位数与 min–max", args.figure_links))
    add("原始逐配置观测保存在 `results/xi_sweep.csv`，逐图聚合源在 "
        "`figures/plot_data/fig9_xi_controlled_sweep.csv`。共同参数下的单个最小误差 xi "
        "不自动等于各方法相同误差约束下最快的 xi。")
    projected = [r for r in sweeps + tables["convergence"] + tables["validation"]
                 if r.get("method") == "PPPM-grid-PSD"]
    tagged = [("六扫", [r for r in sweeps if r.get("method") == "PPPM-grid-PSD"]),
              ("独立网格收敛", [r for r in tables["convergence"] if r.get("method") == "PPPM-grid-PSD"]),
              ("九配置候选验证", [r for r in validations if r.get("method") == "PPPM-grid-PSD"])]
    diagnostic_rows = []
    for tag, rr in tagged:
        active = [r for r in rr if (number(r.get("negative_blocks")) or 0) > 0 or (number(r.get("neg_fraction")) or 0) > 0]
        significant = [r for r in rr if (number(r.get("significant_negative_blocks")) or 0) > 0]
        diagnostic_rows.append([tag, len(rr), len(active), len(significant), fmt(max_field(rr, "E_proj")),
             fmt(max_field(rr, "correction_relative_fro")), fmt(max_field(rr, "correction_max_spectral")),
             fmt(min_field(rr, "min_raw_eig")), fmt(min_field(rr, "min_eig_after"))])
    add(table(["投影数据范围", "记录数", "活跃记录", "显著负块记录*", "max E_proj", "max 相对表修正 Fro",
               "max 单块修正谱范数", "min λ(raw)", "min λ(after)"], diagnostic_rows))
    add("*显著性统计仅在保存了 `significant_negative_blocks` 的记录上有意义，阈值为 "
        "64*eps_float64*max|λ|；所有 λ<0 的修正仍全部施加。旧检查点缺该字段时，上表的显著记录数不能当作无显著负块的证据。")
    if projected:
        active = [r for r in projected if (number(r.get("negative_blocks")) or 0) > 0 or (number(r.get("neg_fraction")) or 0) > 0]
        if not active:
            add("已保存范围中未观察到活跃表投影；此范围的新旧误差接近只能说明投影在这些参数处是 identity，"
                "不能由此推出任意粗网格或最快选参处保阶。必须另查搜索/验证中的活跃区。")
        else:
            add(f"已保存范围有 {len(active)}/{len(projected)} 条活跃记录；活跃区 M 为 {range_text(active, 'M')}，"
                f"p 为 {range_text(active, 'p')}，xi 为 {range_text(active, 'xi')}。这些范围是混合多个设置的包络，"
                "不代表其中每一点都活跃。")
        ratios = [r["E_G"]/r["E_original"] for r in projected if number(r.get("E_G")) is not None
                  and number(r.get("E_original")) is not None and r["E_original"] > 1e-11]
        if ratios:
            add(f"排除原始误差≤1e-11的 floor 附近点后，修复后/原始总误差比在已保存记录中为 "
                f"{fmt(min(ratios))}–{fmt(max(ratios))}（中位数 {fmt(med(ratios))}）。"
                "PSD projection 保证结构，不保证误差一定下降；两个误差仍按相同 Γ_D 实测。")
    add("\n### 独立网格收敛与阶数\n")
    add("独立 refinement 固定 rho=.75、kappa=.7、xi=.85、sR=8、sF=6、B2 sT=36；"
        "p∈{2,3,5,7,9}，M 依合法 Nyquist 域取值。定义 mesh-only error 为 "
        "`||Γ_Lh−(Γ_D−Γ_S(sR=8))||₂/||Γ_D||₂`，GSfar 独立从短程核计算。"
        "已采用大截断隔离短程与 Fourier tail；若这些残差在某范围仍占主导，该范围不得解释为网格渐近阶。"
        "拟合对每个 method/family 先对 seed 取中位数，剔除 E_mesh≤1e-11，"
        "取剩余最后四个 M，至少三个点且要求严格下降；无合法拟合时留空。"
        "1e-11 是拟合排除阈值，不能据此声称 direct reference 的实际误差恰等于该值。")
    fits = tables["mesh_rate_fits"]
    add(table(["方法", "配置族", "p", "M 拟合区间", "点数", "观测 q", "fit内 max E_proj", "投影区域"], [
        [r.get("method"), r.get("family"), fmt(r.get("p")), f"{fmt(r.get('Mlo'))}–{fmt(r.get('Mhi'))}",
         fmt(r.get("n")), fmt(r.get("order")), fmt(r.get("max_E_proj")), fit_regime(r, tables["convergence"])]
        for r in sorted(fits, key=lambda r: (str(r.get("method")), str(r.get("family")), r.get("p", 0)))]))
    fit_group = defaultdict(list)
    for row in fits:
        if number(row.get("order")) is not None:
            fit_group[(row.get("method"), row.get("p"))].append(row)
    if fit_group:
        add("按 p 汇总配置族的 formal slope 范围如下；这是上述有限 M 区间的形式斜率，mixed-transition 不代表收敛阶。")
        add(table(["方法", "p", "formal q 的配置族范围", "中位 q", "参与配置族数"], [
            [m, fmt(p), range_text(rr, "order"), fmt(med([r["order"] for r in rr])), len(rr)]
            for (m,p), rr in sorted(fit_group.items())]))
    else:
        add("尚无合法阶数拟合，当前不能回答投影在独立 refinement 中是否保留原高阶。")
    add(image_section(args.figures, "fig3_mesh_convergence", "图 3：三方法多 p mesh-only 收敛；粗线为已记录拟合区间", args.figure_links))
    add("**mixed-transition 拟合不是收敛阶证书。** 若某个区间横跨投影从活跃到不活跃、"
        "负块修正迅速消失或误差抵消，formal log–log slope 可能非常大；这不表示获得超高阶。"
        "`mesh_rate_fits.csv` 保留该区间斜率以便审计，图中 mixed active/inactive 点为空心。"
        "局部相邻斜率与 above-floor 标志另见 `results/mesh_local_slopes.csv`；"
        "只有投影全不活跃、至少三个 above-floor 网格点的恢复区拟合另见 `results/mesh_inactive_rate_fits.csv`。"
        "若 p7/p9 在不活跃区迅速达到 reference/roundoff floor、只剩不足三个可用点，恢复阶必须记为 N/A，不能用过渡区大斜率补齐。")
    inactive_fits=tables['mesh_inactive_rate_fits']
    if inactive_fits:
        add(table(['配置族','p','无投影拟合区间 M','有效点数','恢复区实测 q','状态'],[
            [r.get('family'),fmt(r.get('p')),f"{fmt(r.get('Mlo'))}–{fmt(r.get('Mhi'))}",
             fmt(r.get('n')),fmt(r.get('order')),r.get('status')]
            for r in inactive_fits]))
    inactive_fits = tables["mesh_inactive_rate_fits"]
    if inactive_fits:
        preferred = ["method", "family", "p", "Mlo", "Mhi", "order", "n", "floor", "status"]
        columns = [k for k in preferred if any(k in r for r in inactive_fits)]
        add(table(columns, [[fmt(r[k]) if number(r.get(k)) is not None else r.get(k, "N/A") for k in columns]
                            for r in inactive_fits]))
    else:
        add("目前尚无单独不活跃恢复区拟合；不以混合区的 formal slope 替代它。")
    add(image_section(args.figures, "fig3b_observed_mesh_orders", "图 3b：各 p 的记录区间 formal slope；mixed-transition 不是收敛阶", args.figure_links))
    add(image_section(args.figures, "fig4_projection_error_and_correction", "图 4：投影前后误差、实际粒子算子修正、负块比例与块修正；空心倒三角表示真零", args.figure_links))
    add("对数图中的 exact zero 使用单独的显示 floor，逐图 `plot_data/*.csv` 同时保留真实零与显示 floor。"
        "投影不活跃区的高阶、投影活跃区的有限区间斜率与任意参数高阶证书须分别陈述。")
    add("\n## 5. Experiment 2：固定误差约束下的实测选参\n")
    add("固定 N=256、rho=.75、kappa=.2、c=1，九个配置，主目标 epsilon=1e-4。"
        "三方法分别优化合法数值参数：原/新为 xi,sR,sF,M,p，B2 为 xi,sR,sT,M,p。"
        "采用历史 best 起点及 coarse→local→feasibility-boundary 搜索；"
        "short/Fourier Gaussian tail 以 sqrt(log(1/epsilon)) 提供量级，"
        "B2 参考实测正分配偏差，new 依据 `E_total≤E_original+E_proj` 逐点检查 correction；"
        "未先验强加新方案 h^p 或 h²。搜索的 t_screen 在 accuracy 任务并发负载下获得，"
        "只用于候选排序，不能与最终 serial 时间混用。")
    search = tables["search"]
    budget_rows = []
    for method in METHODS:
        rr = [r for r in search if r.get("method") == method]
        b = budgets.get(method, {})
        budget_rows.append([method, len({r.get("id") for r in rr}),
             sum(r.get("status") != "ok" for r in rr),
             sum(r.get("stage") == "coarse" for r in rr), sum(r.get("stage") == "local" for r in rr),
             sum(r.get("stage") == "boundary" for r in rr),
             sum(r.get("stage") not in ("coarse", "local", "boundary") for r in rr),
             sum(r.get("method") == method and number(r.get("n_configs")) == 9 for r in tables["validation_summary"])])
    add(table(["方法", "独立新候选", "失败候选", "coarse", "local", "boundary", "cross/其它", "完成九配置验证候选"], budget_rows))
    add(table(["方法", "xi 搜索包络", "sR 搜索包络", "专用 cutoff 包络", "M 搜索包络", "p 搜索包络"], [
        [method, range_text(rr, "xi"), range_text(rr, "sR"),
         range_text(rr, "sT" if method == "B2-positive" else "sF"), range_text(rr, "M"), range_text(rr, "p")]
        for method in METHODS if (rr := [r for r in search if r.get("method") == method])]))
    add("包络不是穷举笛卡尔积；完整独立候选、失败原因和顺序见 `results/search.csv`。"
        "历史搜索预算与新预算不同，旧最佳仅作为初始知识；具体阶段计数如上。"
        "有限候选、局部与边界检查产生的是**最快已测值**，不构成全局最优证明。")
    failures = [r for r in search if r.get("status") != "ok"]
    if failures:
        add(table(["方法", "候选 id", "失败原因"], [[r.get("method"), r.get("id"), r.get("failure", r.get("status"))] for r in failures]))
    else:
        add("当前已保存 search.csv 中未记录失败候选；这只描述已执行点，不代表搜索域内不存在非法或资源超限点。")
    add("候选先在 uniform seed2 用完整 E_G 筛选；之后检查所有九配置，约束为 "
        "`max_configuration E_G≤epsilon`，不会以 probe 误差或平均误差放宽约束。"
        "数据文件区分 screen-pass、nine-config feasible 与 final common timed。")
    add("\n### 主 1e-4 档最快已测参数与同批时间\n")
    if primary_protocol or primary_best:
        add("**两轮计时与候选交叉复核。** 第一轮 primary timing 的全部原始样本、参数表/配置 setup、组件、"
            "内存及协议完整保存在 `results/primary_timing/`。发现新方案 winner 的数值参数此前未被 Original "
            "同样尝试后，增加 `Original @ grid-PSD winner` 的九配置准确性验证，并将三方法 primary best、"
            "原/B2 历史 anchor 与交叉候选纳入第二轮主 1e-4 confirmation。"
            "`results/` 根目录的 final_timing/best_configuration/components/paired ratios 是最终主档确认批次；"
            "本报告所有主档速度比、Pareto、setup/组件与摊销只读取这一轮。"
            "交叉复核让 Original 也获得相同的候选参数，避免把搜索路径差异当作投影算法本身的速度优势。")
        batch_rows = []
        for label, purpose, records, raw, saved_protocol in [
                ("primary", "secondary epsilon 与第一轮候选记录", primary_timing, primary_raw, primary_protocol),
                ("confirmation", "最终主 1e-4 比较", tables["final_timing"], tables["final_timing_raw"], protocol)]:
            rounds, reps, configs = (number(saved_protocol.get(k)) for k in ("rounds", "reps", "n_configs"))
            expected = len(records)*int(rounds*reps*configs) if None not in (rounds,reps,configs) else None
            batch_rows.append([label, purpose, len(records),
                    f"{fmt(rounds)}×{fmt(reps)}×{fmt(configs)}", len(raw), fmt(expected)])
        add(table(["批次", "用途", "已计时候选数", "rounds×samples×configs", "原始正式样本行", "协议预期行"], batch_rows))
        add("完整协议为 primary 15 候选 ×5×7×9=4725 条，随后主档 confirmation 6 候选 ×5×7×9=1890 条；"
            "上表读取实际保存数，在检查点时可能尚未齐全。primary 发现的新方案较快，"
            "不能直接归因于投影，因为 Original 在第一轮尚未覆盖该 winner 参数；最终主判定使用交叉后的第二轮。")
        primary_grid = next((r for r in primary_best if r.get("method") == "PPPM-grid-PSD"), None)
        if primary_grid:
            cross_rows = []
            for method in ("PPPM-original", "PPPM-grid-PSD"):
                certified = [r for r in tables["validation_summary"] if r.get("method") == method
                             and same_parameters(r, primary_grid) and number(r.get("n_configs")) == 9]
                timed = [r for r in tables["final_timing"] if r.get("method") == method and same_parameters(r, primary_grid)]
                chosen = timed[0] if timed else (certified[0] if certified else {})
                cross_rows.append([method, f"xi={fmt(primary_grid.get('xi'))}, sR={fmt(primary_grid.get('sR'))}, "
                                   f"sF={fmt(primary_grid.get('sF'))}, M={fmt(primary_grid.get('M'))}, p={fmt(primary_grid.get('p'))}",
                                   fmt(chosen.get("n_configs")), fmt(chosen.get("max_E_G")),
                                   ms(chosen.get("median_matvec_s")), "最终主档确认批" if timed else "尚未在最终批计时"])
            add(table(["同参数交叉方法", "来自 primary grid winner 的参数", "验证配置数", "max E_G",
                       "最终批 matvec / ms", "状态"], cross_rows))
    else:
        add("主档以 `results/` 根目录的统一计时为准；若另有 primary/confirmation 批次，"
            "需保留各自原始文件及 timing_batch 标签，不能拼接不同批次来形成主档速度比。")
    cross = tables["cross_winner_summary"]
    if cross:
        add(table(["参数来源 winner", "接受交叉参数的方法", "九配置数", "max E_G", "max E_proj", "主 1e-4 可行性"], [
            [r.get("source_method"), r.get("method"), fmt(r.get("n_configs")), fmt(r.get("max_E_G")),
             fmt(r.get("max_E_proj")), "达标" if number(r.get("max_E_G")) is not None and r["max_E_G"] <= 1e-4 else "未达标"]
            for r in cross]))
        add("完整六个交叉候选及失败筛选点也完成九配置验证，见 `results/cross_winner_summary.csv`；"
            "该表的 source_winner 旧耗时与 t_screen 仅保留追溯信息，本报告没有将它们用作最终速度比。")
    add(table(["方法", "xi", "sR", "专用 cutoff", "M", "p", "max E_G", "max E_proj", "完整 matvec / ms", "最大相对 IQR"], [
        [method, fmt(row.get("xi")), fmt(row.get("sR")),
         ("sT="+fmt(row.get("sT"))) if method == "B2-positive" else ("sF="+fmt(row.get("sF"))),
         fmt(row.get("M")), fmt(row.get("p")), fmt(row.get("max_E_G")), fmt(row.get("max_E_proj")), ms(row.get("median_matvec_s")),
         fmt(row.get("max_relative_iqr"))] for method in METHODS if (row := best.get(method))]))
    if len(best) != 3 or not protocol_complete:
        add("同批完整计时/协议尚未齐全，不给出三方法的最终速度结论。历史 1.268/2.094/65.630 ms "
            "不参与本报告任何新速度比。")
    else:
        add(f"上述时间来自同一 serial timing pass：{protocol.get('rounds', 'N/A')} rounds × "
            f"{protocol.get('reps', 'N/A')} samples × {protocol.get('n_configs', 'N/A')} configs，"
            f"单计算线程；目标统计量为 `{protocol.get('objective', '未记录')}`。"
            "方法/候选 block 随机化，warmup 与逐样本原始时间见 `timing_protocol.json`、`final_timing_raw.csv`。")
        ratio_rows = []
        for method in METHODS:
            if method not in best:
                continue
            for ref_method in ("PPPM-original", "B2-positive"):
                if method == ref_method or ref_method not in best:
                    continue
                tm, tr = number(best[method].get("median_matvec_s")), number(best[ref_method].get("median_matvec_s"))
                if tm and tr:
                    ratio_rows.append([method, ref_method, fmt(tm/tr), fmt(tr/tm)])
        add(table(["方法", "参照", "方法耗时 / 参照耗时", "参照耗时 / 方法耗时（加速倍数）"], ratio_rows))
        add("这些比值仅由同批、相同最大误差约束下各自最佳已测配置计算；没有拼接旧硬件或旧负载的时间。")
    if "PPPM-grid-PSD" in best:
        row = best["PPPM-grid-PSD"]
        own = [r for r in validations if r.get("method") == "PPPM-grid-PSD" and str(r.get("id")) == str(row.get("id"))]
        offline=[r for r in tables['best_projection_diagnostics'] if str(r.get('id'))==str(row.get('id'))]
        if len(offline)==9:own=offline
        active = sum((number(r.get("negative_blocks", r.get("neg_fraction"))) or 0) > 0 for r in own)
        add(f"新方案主档最快已测候选 `{row.get('id')}`：九配置验证记录 {len(own)} 条，"
            f"其中表投影活跃 {active} 条，max E_proj={fmt(max_field(own, 'E_proj'))}，"
            f"raw 最小特征值={fmt(min_field(own, 'min_raw_eig'))}，"
            f"post 最小特征值={fmt(min_field(own, 'min_eig_after'))}。"
            "这把最快区的投影状态与高精度 baseline 的状态分开检验。")
        if offline:
            add("最快表的 raw/post 实际谱诊断另由 `best_table_audit.py` 离线计算，见 "
                "`results/best_table_diagnostics.csv` 与 `best_projection_diagnostics.csv`；"
                "其中 offline_audit_t_* 不计入生产 setup，既有 E_G、E_proj 与 graph eigenvalue 不变。")
    paired = tables["paired_speed_ratios"]
    if paired:
        add(table(["方法", "参照方法", "paired 耗时比", "95% bootstrap 区间", "配置-round blocks"], [
            [r.get("method"), r.get("reference_method"), fmt(r.get("ratio_median_paired_blocks")),
             f"[{fmt(r.get('ratio_ci95_low'))}, {fmt(r.get('ratio_ci95_high'))}]", fmt(r.get("blocks"))]
            for r in paired]))
        add("paired ratio 是方法耗时/参照耗时；bootstrap 以同配置-round 的中位数比为 block，"
            "区间条件于已经完成的候选选择，不包含全搜索选择偏差。")
        matched=[r for r in paired if r.get('method')=='PPPM-grid-PSD' and r.get('reference_method')=='PPPM-original']
        if matched:
            r=matched[0];lo=number(r.get('ratio_ci95_low'));hi=number(r.get('ratio_ci95_high'))
            if lo is not None and hi is not None and lo<=1<=hi:
                add(f"**与 Original 的速度判断：** 配对耗时比为 {fmt(r.get('ratio_median_paired_blocks'))}，"
                    f"95% 区间 [{fmt(lo)}, {fmt(hi)}] 覆盖 1。两者在本模型与匹配参数下的缓存作用成本相近；"
                    "不能用九配置中位数的中位数之比宣称投影更快。两个统计量聚合顺序不同，"
                    "加上实测 IQR 与外部负载，可能给出不同点估计；所有原始样本均保留。")
    add(image_section(args.figures, "fig5_error_vs_time_pareto", "图 5：筛选误差-时间与九配置验证最大误差-时间；只连接主 1e-4 可行的实测非支配点", args.figure_links))
    add("\n### 额外精度档 frontier\n")
    frontier = tables["frontier"]
    frontier_rows = []
    for eps in (1e-3, 1e-4, 1e-5, 1e-6):
        for method in METHODS:
            rr = [r for r in frontier if r.get("method") == method and number(r.get("epsilon")) == eps]
            valid = [r for r in rr if number(r.get("median_matvec_s")) is not None
                     and number(r.get("max_E_G")) is not None and r["max_E_G"] <= eps]
            if valid:
                row = min(valid, key=lambda r: r["median_matvec_s"])
                frontier_rows.append([f"{eps:.0e}", method, ms(row.get("median_matvec_s")), fmt(row.get("max_E_G")),
                     f"xi={fmt(row.get('xi'))},sR={fmt(row.get('sR'))},M={fmt(row.get('M'))},p={fmt(row.get('p'))}", "已测达标"])
            else:
                why = rr[0].get("status", "尚无同批计时结果") if rr else "尚无同批计时结果"
                frontier_rows.append([f"{eps:.0e}", method, "N/A", "N/A", "N/A", why])
    # Append the batch explicitly to each tier; merged frontier rows may come
    # from separate complete passes, while each target must remain same-batch.
    for display_row in frontier_rows:
        eps, method = float(display_row[0]), display_row[1]
        source = next((r for r in frontier if r.get("method") == method and number(r.get("epsilon")) == eps), {})
        display_row.append(source.get("timing_batch", "未标注"))
    add(table(["目标 epsilon", "方法", "最快完整 matvec / ms", "max E_G", "参数摘要", "状态", "timing_batch"], frontier_rows))
    add("N/A 不插值、不外推时间、不放宽目标误差；有验证但未计时也不能算已经完成该档。")
    if primary_protocol or primary_best:
        add("合并 frontier 的主 1e-4 行来自最终 confirmation，1e-3/1e-5/1e-6 行沿用完整归档的 primary timing；"
            "每一 epsilon 下三方法取同一批次。不同 epsilon 可能来自不同轮，图 6/7 标明 timing_batch；"
            "只在同 epsilon、同 batch 内计算速度比。主 Pareto 不从 secondary frontier 追加点，"
            "也不把 primary 与 confirmation 的 raw samples 混池。secondary 高成本 M128 等候选未在 confirmation 重跑。")
    for eps in sorted({number(r.get("epsilon")) for r in frontier if number(r.get("epsilon")) is not None}):
        batches = {str(r.get("timing_batch", "未标注")) for r in frontier if number(r.get("epsilon")) == eps}
        if len(batches) > 1:
            add(f"**批次检查异常：epsilon={eps:g} 包含多个 batch {sorted(batches)}；图中该档速度比留空，不作跨批比较。**")
    add(image_section(args.figures, "fig6_optimal_speed_frontier", "图 6：各精度档的最快已测时间与同批速度比", args.figure_links))
    add(image_section(args.figures, "fig7_optimal_parameters", "图 7：各档最快已测参数；专用 cutoff 不混同定义", args.figure_links))
    add("\n## 6. Setup、组件、内存与摊销\n")
    add(table(["方法", "table setup / ms", "config setup / ms", "matvec / ms", "表数组 / MiB",
               "最大配置数组 / MiB", "最大临时数组估算 / MiB", "最大数组峰值估算 / MiB"], [
        [m, ms(r.get("median_table_setup_s")), ms(r.get("median_config_setup_s")), ms(r.get("median_matvec_s")),
         mib(r.get("table_array_bytes")), mib(r.get("max_config_array_bytes")),
         mib(r.get("max_temporary_estimated_bytes")), mib(r.get("max_estimated_peak_array_bytes"))]
        for m in METHODS if (r := best.get(m))]))
    add("参数表 setup 包含 new 的旧 influence、IFFT、d/−d symmetrization、投影必须的 batched eigh、negative-part projection 与修正 FFT；"
        "kernel primitive 预热条件对三者一致，另有 `common_kernel_setup_raw.csv`。"
        "reference、dense validation 与 eigen diagnostics 不进入 cached matvec。"
        "正式生产路径使用 `ProjectedMesh(audit_eigenvalues=False)`，关闭额外 raw/post eigvalsh 与分位数审计；"
        "投影必须的 eigh 与全部负谱修正仍执行。该路径与审计路径的 table、delta、Gr、degree/action "
        "逐位一致性另有测试。77-table 的独立审计默认打开额外检查，其耗时不属于正式 setup。"
        "下表以本次 `table_setup_raw.csv` 的实际阶段字段核实诊断是否执行；不因历史审计耗时推断正式 setup。")
    table_raw = tables["table_setup_raw"]
    setup_fields = [("t_original_influence_s", "旧 influence"), ("t_table_ifft_s", "原表 IFFT"),
                    ("t_table_symmetrize_s", "d/−d 对称化"), ("t_block_eigh_s", "投影所需 batched eigh"),
                    ("t_block_projection_s", "negative-part/反演平均"), ("t_projected_fft_s", "修正 FFT"),
                    ("t_eigenvalue_diagnostics_s", "额外 raw/post eigen diagnostics")]
    grid_row = best.get("PPPM-grid-PSD")
    grid_raw = [r for r in table_raw if r.get("method") == "PPPM-grid-PSD"
                and (grid_row is None or str(r.get("id")) == str(grid_row.get("id")))]
    if grid_raw:
        add(table(["新方案参数表阶段", "中位数 / ms"], [
            [label, ms(med([number(r.get(field)) for r in grid_raw]))] for field, label in setup_fields]))
        diagnostic_values = [number(r.get("t_eigenvalue_diagnostics_s")) for r in grid_raw]
        diagnostic_values = [v for v in diagnostic_values if v is not None]
        corrected = [r["seconds"] - r["t_eigenvalue_diagnostics_s"] for r in grid_raw
                     if number(r.get("seconds")) is not None
                     and number(r.get("t_eigenvalue_diagnostics_s")) is not None]
        if diagnostic_values and all(v == 0 for v in diagnostic_values):
            add("本轮所有已记录参数表样本 `t_eigenvalue_diagnostics_s=0`：正式 setup 已排除可选 raw/post eigen 审计，"
                "上表总 setup 就是已实现生产路径的实测成本，包含投影所需 eigh；"
                "无须再扣除 77-table 独立审计或旧检查点的审计耗时。matvec 从不重复投影。")
        elif corrected:
            add(f"逐样本以 measured total table setup 减去额外 raw/post eigen diagnostics，再取中位数，"
                f"得到派生的 algorithm-only table setup≈{ms(med(corrected))} ms。"
                "这是相同样本阶段成本相减的诊断估算，不是另一次精简 production 实现的独立实测，"
                "也不替代上表包含全部实际开销的总 setup；matvec 从不重复投影或这些诊断。")
    components = tables["components_raw"]
    add(table(["方法", "组件", "样本数", "独立计时中位数 / ms", "min–max / ms"], [
        [method, phase, len(rr), ms(med([r["seconds"] for r in rr])),
         f"{ms(min_field(rr, 'seconds'))}–{ms(max_field(rr, 'seconds'))}"]
        for method in METHODS
        for phase in sorted({str(r.get("phase")) for r in components if r.get("method") == method})
        if (rr := [r for r in components if r.get("method") == method and str(r.get("phase")) == phase
                   and number(r.get("seconds")) is not None
                   and (method not in best or str(r.get("id")) == str(best[method].get("id")))])]))
    add("near、spread/gather、FFT/influence、mesh、total 是分别计时的诊断，其中 mesh 与子阶段有包含关系；"
        "独立中位数不相加。内存按 unique owning NumPy allocations 计数，view/shared original influence 不重复算；"
        "临时与峰值为数组规模估算，不等同于单次实测 peak RSS。")
    amortized = []
    for method in METHODS:
        row = best.get(method)
        if not row:
            continue
        tt, tc, tm = (number(row.get(k)) for k in ("median_table_setup_s", "median_config_setup_s", "median_matvec_s"))
        if None not in (tt, tc, tm):
            amortized.append([method] + [ms(tt+tc+n*tm) for n in (1,10,100,1000,10000)])
    add(table(["方法", "1 apply 总成本 / ms", "10 applies / ms", "100 applies / ms", "1000 applies / ms", "10000 applies / ms"], amortized))
    amortization_rows = []
    for method in METHODS:
        row = best.get(method)
        if not row:
            continue
        tt, tc, tm = (number(row.get(k)) for k in ("median_table_setup_s", "median_config_setup_s", "median_matvec_s"))
        if None in (tt, tc, tm):
            continue
        for n in (1,10,100,1000,10000):
            amortization_rows.append(dict(method=method,id=row.get("id"),n_apply=n,table_s=tt,config_s=tc,
                    matvec_s=tm,total_s=tt+tc+n*tm,kind="linear amortization from measured phase medians; not independently timed"))
    amortization_path = args.results / "amortization.csv"
    if amortization_rows:
        with amortization_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(amortization_rows[0]))
            writer.writeheader(); writer.writerows(amortization_rows)
    add("摊销公式 `T_table+T_config+n_apply*T_matvec` 假定参数与位置固定；"
        "这是实测阶段成本的线性摊销推算，不是新增完整循环实测；数值另存 `results/amortization.csv`。"
        "若位置改变，配置 setup 失效并需重新支付；box/kappa/xi/M/投影前 cutoff 改变时，参数表也失效。"
        "Original 与新方案的去卷积还依赖 p；B2 的正表独立于 p，可通过 with_order 复用，但 assignment 与 degree 须重建。")
    add(image_section(args.figures, "fig8_setup_and_cached_cost", "图 8：主 1e-4 档三方 complete action 与 setup 比较", args.figure_links))
    add(image_section(args.figures, "fig8b_matvec_components", "图 8b：主精度档各组件独立计时", args.figure_links))
    if protocol.get("load_limitation"):
        add(f"运行负载限制：{protocol['load_limitation']}。逐轮系统负载和高 CPU 进程在 `results/system_load.json`；"
            "候选随机化与 paired blocks 有助于控制漂移，但不能消除外部负载不确定性。")
    add("\n## 7. 研究问题的证据边界\n")
    add("1. **新方案是否保证 long graph PSD？** 在精确算术、非负同权 spread/gather、对称新表与匹配 degree 下保证；"
        "上文给出结构证明。float64 的具体 residual 和 min eigenvalue 以已保存验证表为准。")
    add(f"2. **投影量多大？** 当前保存范围 max E_proj={fmt(max_field(projected, 'E_proj'))}；"
        f"max 相对表 Fro 修正={fmt(max_field(projected, 'correction_relative_fro'))}。"
        "须结合 active/significant-negative 记录和 before/after 误差判断，不能把原误差当投影量上界。")
    if fit_group:
        facts = []
        for p in sorted({r.get("p") for r in fits if r.get("method") == "PPPM-grid-PSD"}):
            new_fits = fit_group.get(("PPPM-grid-PSD", p), [])
            old_fits = fit_group.get(("PPPM-original", p), [])
            if not new_fits:
                continue
            regimes = sorted({fit_regime(r, tables["convergence"]).split(" ")[0] for r in new_fits})
            facts.append(f"p={fmt(p)}：新 q={range_text(new_fits, 'order')}、原 q={range_text(old_fits, 'order')}，"
                         f"新 fit 区域={','.join(regimes)}")
        add("3. **是否保阶？** " + "；".join(facts) + "。"
            "上述 q 中 mixed-transition 值只是形式斜率，不能当作渐近收敛阶；"
            "投影 inactive 区间可与原 PPPM 对照，active 区间需单独看 correction。"
            "本实验不证明任意参数下始终高阶，也不把所有方案先验归为二阶。")
    else:
        add("3. **是否保阶？** 独立收敛拟合尚未完成，当前不能给出该结论。")
    if len(best) == 3 and protocol_complete:
        original_t = number(best["PPPM-original"].get("median_matvec_s"))
        b2_t = number(best["B2-positive"].get("median_matvec_s"))
        new_t = number(best["PPPM-grid-PSD"].get("median_matvec_s"))
        if original_t and b2_t and new_t:
            add(f"4. **同精度最快已测时间？** 新方案 {ms(new_t)} ms，Original {ms(original_t)} ms，B2 {ms(b2_t)} ms；"
                f"新/Original 耗时比 {fmt(new_t/original_t)}，新/B2 耗时比 {fmt(new_t/b2_t)}。"
                "这里只比较同批主 1e-4 约束下的最快已测配置。")
            if primary_protocol or primary_best:
                add("primary 轮次中新方案相对 Original 的时间差受到搜索路径不同影响，"
                    "不作为 PSD 投影本身加速的证据。Original @ grid winner 已通过独立九配置约束验证，"
                    "其准确性与同批时间以上述交叉复核表及最终确认表为准。")
            add(f"5. **是否值得替代 B2？** 在本次主档 cached action 上，"
                f"{'新方案更快' if new_t < b2_t else 'B2 更快或相当'}；"
                "具体是否替代还取决于上表 setup/内存和位置更新频率，"
                "此结论限于所测模型、精度和硬件，不扩展为所有工作负载的推荐。")
    else:
        add("4. **同精度最快时间与是否替代 B2？** 三方法同批结果不齐，暂不作速度或替代结论；"
            "旧历史时间不能填补缺失的新测值。")
    add("\n## 8. 未完成项与可恢复命令\n")
    pending = []
    if not status["sweep_done"]:
        pending.append("六组扫描尚未全部达到三方法各 927 记录。")
    if not status["convergence_done"]:
        pending.append(f"独立 mesh refinement 尚未达到 {status['expected_convergence_rows']} 条完整记录。")
    if len(tables["xi_sweep"]) != 135:
        pending.append(f"补充受控 xi 扫描尚未达到 135 条完整记录（当前 {len(tables['xi_sweep'])}）。")
    if not status["timing_done"] or not protocol_complete:
        pending.append("三方法主档完整九配置验证/同批计时及协议尚未齐全。")
    if not final_audit:
        pending.append("需要全部任务结束后的历史文件 hash 复核。")
    elif any(count for _, count in mismatch_counts(final_audit)):
        pending.append("结束时历史文件 hash 核对存在不匹配，需逐项解释或修复；不能声称历史内容全部未变。")
    if not test_paths:
        pending.append("本次自动测试日志尚未保存。")
    for item in status["figures_skipped"]:
        pending.append(f"绘图跳过 {item.get('figure')}: {item.get('reason')}")
    if status["figures_generated"] < 10:
        pending.append("完整的十组图及逐图数据尚未全部生成。")
    elif status["figures_generated"] < 11:
        pending.append("十组主图已保存，补充受控 xi 图仍未全部生成；完整交付应包含十一组图。")
    if pending:
        add("\n".join("- " + p for p in pending))
    else:
        add("主实验的上述可检查数据项已齐备；额外精度档仍按 frontier 中 N/A 状态逐项说明。")
    add("以下命令均从仓库根目录运行；accuracy/search 先完成并退出，再启动最终同批计时。"
        "accuracy 与 search 带检查点跳过已有点；计时若已有 final 输出会拒绝覆盖，"
        "复跑请使用新的 `--output` 目录并明确让绘图/报告读取该轮数据。")
    prefix = "benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison"
    add("\n```bash\n"
        f"python3 {prefix}/test_grid_psd.py 2>&1 | tee {prefix}/results/tests_grid_psd.log\n"
        f"python3 {prefix}/run_accuracy.py sweeps --workers 2\n"
        f"python3 {prefix}/run_accuracy.py convergence --workers 2\n"
        f"python3 {prefix}/run_xi_sweep.py --workers 2\n"
        f"python3 {prefix}/run_search.py coarse\n"
        f"python3 {prefix}/run_search.py local\n"
        f"python3 {prefix}/run_search.py boundary\n"
        f"python3 {prefix}/table_audit.py\n"
        f"python3 {prefix}/merge_diagnostics.py\n"
        f"python3 {prefix}/analyze_regimes.py\n"
        "# 等待 accuracy/reference/search workers 全部结束后：\n"
        f"python3 {prefix}/run_timing.py --rounds 5 --reps 7\n"
        f"python3 {prefix}/cross_winner.py\n"
        f"python3 {prefix}/archive_primary_timing.py\n"
        f"python3 {prefix}/run_timing.py --main-confirmation --rounds 5 --reps 7\n"
        f"python3 {prefix}/merge_frontiers.py\n"
        f"python3 {prefix}/plot_results.py\n"
        f"python3 {prefix}/report_generator.py\n"
        f"python3 {prefix}/finalize_audit.py\n"
        f"python3 {prefix}/report_generator.py\n"
        "```\n")
    add("顺序执行和恢复也可使用 `run_all.py --workers 2`；该入口会跳过已保存 archive 与已完成 confirmation，"
        "避免再次移动归档或覆盖完整计时批次。阶段恢复参数见 README。")
    add("结果索引：`results/sweeps.csv`、`convergence.csv`、`mesh_rate_fits.csv`、`search.csv`、"
        "`xi_sweep.csv`（共同参数下的 xi 受控补充扫描）、"
        "`validation.csv`/`validation_summary.csv`、`final_timing_raw.csv`/`final_timing.csv`、"
        "`best_configuration.csv`、`frontier.csv`（按误差档合并；两批原始 frontier 位于 `accuracy_time_frontier.csv` 与 `primary_timing/accuracy_time_frontier.csv`）、"
        "`paired_speed_ratios.csv`、`table_setup_raw.csv`、`config_setup_raw.csv`、`components_raw.csv`、"
        "`memory.csv`、`timing_protocol.json`。`figures/plot_data/` 是逐图绘制统计源 CSV，"
        "`plot_status.json` 保留缺数据/跳图原因。")
    add("本报告未替换全局最优证明、任意参数高阶证书或跨模型结论；"
        "没有把未测结果、不可行点或外推时间写成实测。")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    snapshot = dict(generated=now, report=str(args.output), results=str(args.results), status=status,
                    source_files=[dict(path=str(path), size=path.stat().st_size,
                                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                  for path in sorted(set(INPUTS)) if path.exists()])
    args.output.with_suffix(".sources.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(report=str(args.output), status=status), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--figures", type=Path, default=HERE / "figures")
    parser.add_argument("--output", type=Path, default=HERE / "report.md")
    args = parser.parse_args()
    try:
        args.figure_links = args.figures.relative_to(args.output.parent)
    except ValueError:
        args.figure_links = args.figures.resolve()
    # Existing-files checks use the actual output directory, not relative links.
    generate(args)


if __name__ == "__main__":
    main()
