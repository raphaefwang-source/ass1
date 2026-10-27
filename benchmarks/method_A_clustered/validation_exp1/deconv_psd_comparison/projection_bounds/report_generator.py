"""Generate a short Chinese bounds report and machine-readable analysis.

Only this new directory's outputs/catalog are read. Numerical-reference errors
remain empirical. There is no certified total-error bound, method modification,
cached-matvec re-timing, or historical timing speed ratio in this report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
INPUTS = []
PARAMETERS = ("N", "rho", "kappa", "xi", "sR", "sF", "M", "p")
TEXT = {"dataset", "source_csv", "case_id", "data_id", "parameter_id", "table_key", "config", "family", "id",
        "source_row_id", "correction_norm_source", "selected_label", "label", "case_label", "role", "phase",
        "method", "sweep", "setting", "source_sweep", "source_setting", "batch", "status"}


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def truth(value):
    return value is True or str(value).strip().lower() in ("true", "1", "1.0", "yes")


def read(path):
    if not path.exists():
        return []
    INPUTS.append(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
            if key not in TEXT and number(value) is not None:
                row[key] = number(value)
        row.setdefault("case_id", row.get("data_id", ""))
        row.setdefault("parameter_id", row.get("table_key", ""))
        if not row.get("dataset") and row.get("source_csv"):
            row["dataset"] = Path(str(row["source_csv"])).stem
    return rows


def read_json(path):
    if not path.exists():
        return {}
    INPUTS.append(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"read_error": "Incomplete JSON checkpoint"}


def quantile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    index = (len(values)-1)*q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo]*(hi-index)+values[hi]*(index-lo) if lo != hi else values[lo]


def summary(values):
    values = [number(v) for v in values]
    values = [v for v in values if v is not None]
    return dict(n=len(values), min=min(values) if values else None, q25=quantile(values,.25),
                median=quantile(values,.5), q75=quantile(values,.75), max=max(values) if values else None)


def field_summary(rows, field):
    return summary([r.get(field) for r in rows])


def maximum(rows, field):
    return field_summary(rows, field)["max"]


def minimum(rows, field):
    return field_summary(rows, field)["min"]


def fmt(value, digits=5):
    val = number(value)
    return "N/A" if val is None else f"{val:.{digits}g}"


def milliseconds(value):
    return fmt(number(value)*1e3) if number(value) is not None else "N/A"


def md_table(headers, rows):
    if not rows:
        return "尚无已保存数据。\n"
    esc = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output += ["| " + " | ".join(esc(v) for v in row) + " |" for row in rows]
    return "\n".join(output)


def unique(rows):
    seen = {}
    for row in rows:
        key = row.get("case_id") or tuple(row.get(k) for k in ("config",)+PARAMETERS)
        seen.setdefault(key, row)
    return list(seen.values())


def verified(row):
    explicit = row.get("correction_norm_verified", row.get("correction_only_verified"))
    if explicit not in (None, ""):
        return truth(explicit)
    source = str(row.get("correction_norm_source", "")).lower().replace("-", "_")
    if any(s in source for s in ("proxy", "historical", "empirical_eproj", "reused_e_proj")):
        return False
    return "correction" in source and any(s in source for s in ("dense", "direct", "eigval", "only"))


def active(row):
    return truth(row["active"]) if row.get("active") not in (None, "") else (number(row.get("negative_blocks")) or 0)>0


def selected_label(row):
    return str(next((row[k] for k in ("selected_label", "case_label", "label", "role") if row.get(k)), "unlabelled"))


def labelled_groups(bounds, cases):
    groups = defaultdict(list)
    for row in cases:
        matches = [r for r in bounds if row.get("case_id") and r.get("case_id") == row["case_id"]]
        if not matches and row.get("parameter_id"):
            matches = [r for r in bounds if r.get("parameter_id") == row["parameter_id"]]
        groups[selected_label(row)].extend(matches or [row])
    return {label: unique(rows) for label, rows in groups.items()}


def timing_samples(raw, group, label, phase):
    ids = {r.get("case_id") for r in group}; pars = {r.get("parameter_id") for r in group}
    rows = [r for r in raw if r.get("phase") == phase and number(r.get("seconds")) is not None
            and (selected_label(r) == label or (r.get("case_id") and r["case_id"] in ids)
                 or (r.get("parameter_id") and r["parameter_id"] in pars))]
    seen = {}
    for i, row in enumerate(rows):
        shared = truth(row.get("parameter_sample_shared_across_configs")) or phase in ("bound_parameter_setup", "correction_fft")
        key = (row.get("parameter_id"), row.get("repeat"), phase) if shared else ("row", i)
        seen.setdefault(key, row)
    return list(seen.values())


def group_analysis(rows):
    dense = [r for r in rows if verified(r) and number(r.get("correction_norm")) is not None and r["correction_norm"]>0]
    ratios = [r["B"]/r["correction_norm"] for r in dense if number(r.get("B")) is not None]
    machine_floor = 512*2.220446049250313e-16
    resolved = [r for r in dense if number(r.get("normGammaD")) and r["correction_norm"]/r["normGammaD"]>machine_floor]
    resolved_ratios = [r["B"]/r["correction_norm"] for r in resolved if number(r.get("B")) is not None]
    return dict(rows=len(rows), configurations=len({r.get("config") for r in rows}),
        active=sum(active(r) for r in rows), inactive=sum(not active(r) for r in rows),
        literal_zero=sum(truth(r.get("exact_zero")) for r in rows),
        verified_positive_norms=len(dense), uncomputed_active_norms=sum(active(r) and number(r.get("correction_norm")) is None for r in rows),
        norm_sources=dict(Counter(str(r.get("correction_norm_source")) for r in rows)), tightness=summary(ratios),
        correction_relative_machine_floor=machine_floor, positive_norms_above_machine_floor=len(resolved),
        positive_norms_at_or_below_machine_floor=len(dense)-len(resolved), tightness_above_machine_floor=summary(resolved_ratios),
        **{key:field_summary(rows,key) for key in ("E_old","E_new","E_proj","B_rel","B_simple_rel","B_degree_rel",
                    "empirical_envelope","delta","negative_fraction","negative_blocks","correction_fro","min_raw_eig")},
        E_proj_float_violations=sum((number(r.get("violation_abs")) or 0)>0 for r in rows),
        roundoff_violations=sum(truth(r.get("roundoff_violation")) for r in rows),
        material_violations=sum(truth(r.get("material_violation")) for r in rows),
        max_E_proj_violation=maximum(rows,"violation_abs"),
        max_correction_only_bound_violation=maximum(rows,"correction_bound_violation_abs"),
        max_symmetrization_defect=maximum(rows,"symmetrization_defect"),
        max_norm_comparison_residual=maximum(rows,"norm_comparison_residual"),
        empirical_budget_epsilon=1e-4,
        empirical_budget_pass=sum(number(r.get("empirical_envelope")) is not None and r["empirical_envelope"]<=1e-4 for r in rows),
        empirical_new_error_pass=sum(number(r.get("E_new")) is not None and r["E_new"]<=1e-4 for r in rows))


def figure(args, name, caption):
    path = args.figures/f"{name}.png"
    if not path.exists():
        return f"图 `{name}` 尚未生成。"
    try:
        link = path.relative_to(args.output.parent).as_posix()
    except ValueError:
        link = path.resolve().as_posix()
    return f"![{caption}]({link})\n\n{caption}。"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--figures", type=Path, default=HERE/"figures")
    parser.add_argument("--output", type=Path, default=HERE/"report.md")
    args = parser.parse_args()
    bounds, raw, cases = (read(args.results/f"{name}.csv") for name in ("bounds","timing_raw","cases_summary"))
    catalog = read_json(HERE/"source_catalog.json")
    verification = read_json(args.results/"verification_summary.json")
    timing_context = read_json(args.results/"timing_context.json")
    outlier = read_json(args.results/"tightness_outlier_diagnostic.json")
    plots = read_json(args.figures/"plot_status.json")
    final_audit = read_json(HERE/"audit_final.json") or read_json(HERE/"final_audit.json") or read_json(args.results/"audit_final.json")
    distinct = unique(bounds)
    correction_groups = {}
    for row in distinct:
        correction_groups.setdefault((row.get("parameter_id"),row.get("config")),row)
    distinct_corrections = list(correction_groups.values())
    grouped = labelled_groups(bounds,cases)
    all_phases = sorted({str(r.get("phase")) for r in raw})
    analysis = dict(created_at=datetime.now(ZoneInfo("America/Detroit")).isoformat(timespec="seconds"),
                    source_rows=len(bounds),unique_cases=len(distinct),global_summary=group_analysis(distinct),
                    source_row_summary=group_analysis(bounds),
                    unique_table_configuration_corrections=len(distinct_corrections),
                    distinct_correction_group_summary=group_analysis(distinct_corrections),
                    tightness_outlier_diagnostic=outlier,
                    certified_total_bound_available=False,datasets={},table_groups=[],selected_cases={},parameter_scans={})
    for dataset in sorted({str(r.get("dataset")) for r in bounds}):
        rows = [r for r in bounds if str(r.get("dataset"))==dataset]
        analysis["datasets"][dataset] = dict(source_rows=len(rows),**group_analysis(unique(rows)))
    table_groups = defaultdict(list)
    for row in distinct:
        table_groups[str(row.get("parameter_id"))].append(row)
    for parameter_id, rows in sorted(table_groups.items()):
        analysis["table_groups"].append(dict(parameter_id=parameter_id,
                parameters={key:rows[0].get(key) for key in PARAMETERS if key!="sR"}, **group_analysis(rows)))
    for label, rows in grouped.items():
        record = group_analysis(rows)
        record["parameters"] = {key:rows[0].get(key) for key in PARAMETERS} if rows else {}
        record["case_ids"] = [r.get("case_id") for r in rows]
        record["timing"] = {}
        for phase in all_phases:
            samples = timing_samples(raw,rows,label,phase)
            record["timing"][phase] = dict(summary([r["seconds"] for r in samples]),
                parameter_sample_shared_across_configs=phase in ("bound_parameter_setup","correction_fft"))
        analysis["selected_cases"][label] = record
    for parameter in ("xi","sF","p","rho","kappa"):
        if parameter=="xi":
            rows=[r for r in bounds if r.get("dataset")=="xi_sweep"]
        else:
            rows=[r for r in bounds if r.get("dataset")=="sweeps" and r.get("sweep",r.get("source_sweep"))==parameter
                  and r.get("setting",r.get("source_setting"))=="baseline"]
        values=[]
        for value in sorted({r.get(parameter) for r in rows if number(r.get(parameter)) is not None}):
            values.append(dict(value=value,**group_analysis(unique([r for r in rows if r.get(parameter)==value]))))
        analysis["parameter_scans"][parameter] = values
    analysis["moderate_parameter_scans"]={}
    for parameter in ("M","p","rho","kappa"):
        rows=[r for r in bounds if r.get("dataset")=="sweeps" and r.get("sweep",r.get("source_sweep"))==parameter
              and r.get("setting",r.get("source_setting"))=="moderate"]
        analysis["moderate_parameter_scans"][parameter]=[
            dict(value=value,**group_analysis(unique([r for r in rows if r.get(parameter)==value])))
            for value in sorted({r.get(parameter) for r in rows if number(r.get(parameter)) is not None})]
    args.results.mkdir(parents=True,exist_ok=True)
    (args.results/"analysis_summary.json").write_text(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    text=[]; add=text.append
    add("# Grid PSD 投影修正上界：验证与误差预算\n")
    add(f"生成时间：{analysis['created_at']}。本轮只验证既有投影修正的两个上界、绘图与分析；"
        "不更改旧 Grid PSD 方法、参数搜索、cached matvec 或旧结果，不开发新离散方案或 SDP。"
        "任务全文见 [requirements_github.md](requirements_github.md)。")
    add("## 1. 数学界与数值比较的不同含义\n")
    add("完整证明、适用条件、self cancellation 和浮点诊断界限见 [proof.md](proof.md)。"
        "使用既有物理实空间规范 `T_raw=IFFT(G_old)/h³`，再令 `S_d=(T_raw[d]+T_raw[-d])/2` 得到反演对称化表；"
        "`C_d=(-S_d)_+`，`delta=max_d||C_d||₂`。非负且总和为 1 的同权 stencil "
        "给出 PSD 边修正 `C_ij=Σ w_ig C_(g-l) w_jl`。实数精确算术中，")
    add("```text\nB_simple = N*delta                         (N>=2)\n"
        "D_i^C = sum_g w_ig sum_l C_(g-l) n_l       (包含 self)\n"
        "B_degree = 2*max_i lambda_max(D_i^C)\n"
        "||Delta Gamma||_2 <= B = min(B_simple,B_degree).\n```\n")
    add("`N*delta` 对任意粒子配置成立；`2Dmax` 依赖当前配置，但只需 correction-only FFT 与 N 个 3×3 degree，"
        "不用形成 3N×3N 稠密矩阵。degree 直接来自 C，不从两个大完整 degree 相减。"
        "self 可保留在 degree 以使上界成立，graph action 中完整 self response 代数抵消。"
        "这是精确算术数学证书；本轮 float64 FFT、特征分解、对称化 defect 与 norm residual 是独立数值诊断，"
        "未经有向舍入/区间算术认证，不能把保存的浮点数字当作逐位严格的机器证书。")
    add("**所有 E_old/E_new 都是对独立数值 direct reference 的 empirical error。**"
        "`E_proj` 是历史完整新旧算子的数值差，可能含旧权重/对称化/FFT 的舍入差。"
        "真正 correction-only graph 范数优先独立 dense all-offset 核查；宽活跃表未核查则留空。"
        "`E_old_measured+B/||Gamma_D||₂` 仅叫相对于该数值参考的 triangle envelope，"
        "不是 certified total bound。另行认证的 epsilon_old 不可用，本轮 `certified_total_bound` 始终 unavailable。")
    if any(r.get("certified_total_bound") not in (None,"") for r in bounds):
        add("**输入字段检查：发现非空 certified_total_bound；本报告不采纳它为总误差证书，需独立核查其来源。**")
    add("## 2. 覆盖范围与浮点检查\n")
    add(f"保存 {len(bounds)} 条来源记录、{len(distinct)} 个唯一构型算子、{len(table_groups)} 个唯一参数表；"
        "同一 case_id 出现在多个数据集时，逻辑完整算子统计按 case_id 去重。"
        f"真正重算的 table/config correction 有 {len(distinct_corrections)} 组；不同 sR 的完整算子可共享同一 C/degree/GC，"
        "不能把它们当作独立 correction 计算。原始来源行与参数标识在 bounds.csv 保留。")
    add(md_table(["数据集","来源行","唯一 case","active/inactive","true dense 正范数","活跃宽表未核查"], [
        [dataset,record["source_rows"],record["rows"],f"{record['active']}/{record['inactive']}",record["verified_positive_norms"],record["uncomputed_active_norms"]]
        for dataset,record in analysis["datasets"].items()]))
    g=analysis["global_summary"]
    independent=analysis["distinct_correction_group_summary"]
    add(md_table(["correction 覆盖口径","活跃","true dense 正范数","literal zero","宽活跃表未核查"], [
        ["逻辑完整算子 case（含不同 sR）",g["active"],g["verified_positive_norms"],g["literal_zero"],g["uncomputed_active_norms"]],
        ["实际 table/config correction 组",independent["active"],independent["verified_positive_norms"],independent["literal_zero"],independent["uncomputed_active_norms"]]]))
    add(md_table(["检查量（唯一 case）","观测"], [
        ["literal-zero 修正",g["literal_zero"]], ["true dense 正范数覆盖",g["verified_positive_norms"]],
        ["宽活跃表只复用历史完整差、无 true norm",g["uncomputed_active_norms"]],
        ["E_proj>B_rel 浮点诊断 / roundoff / material",f"{g['E_proj_float_violations']}/{g['roundoff_violations']}/{g['material_violations']}"],
        ["最大 E_proj 比较超界（相对）",fmt(g["max_E_proj_violation"])],
        ["true correction-only 最大超界（绝对）",fmt(g["max_correction_only_bound_violation"])],
        ["最大对称化 defect（绝对表分量）",fmt(g["max_symmetrization_defect"])],
        ["最大 correction norm 比较残差（绝对）",fmt(g["max_norm_comparison_residual"])]]))
    add("roundoff/material 分类使用运行摘要中保存的判定尺度，不通过膨胀 B、截掉 E_proj 或强制 tightness≥1 来修饰结果。"
        f"本次保存尺度为 {fmt(verification.get('roundoff_classification_relative_scale'))}。"
        "C 完全为零时，B 与真实 correction graph 均为真零；历史 E_proj 仍可在机器舍入量级非零。"
        "图中真零使用空心倒三角和单独显示 floor，源 CSV 同时保留零 mask 与显示坐标；"
        "零修正的 tightness=0/0 未定义，未用无穷大或假定的 1 代替。")
    add(f"按原始来源行统计，E_proj 比较超界 {fmt(verification.get('strict_float_E_proj_violations'))} 条，"
        f"其中 roundoff {fmt(verification.get('roundoff_E_proj_violations'))}、material {fmt(verification.get('material_E_proj_violations'))}；"
        "上表按唯一逻辑 case 去重，两个计数口径不代表不同的失败结论。")
    add("## 3. 上界保守性与参数规律\n")
    add(f"对具有 true dense correction-only 正范数核查结果的逻辑完整算子 case，合并上界 tightness "
        f"`B/||Delta Gamma||₂` 范围 {fmt(g['tightness']['min'])}–{fmt(g['tightness']['max'])}，"
        f"中位数 {fmt(g['tightness']['median'])}（{g['tightness']['n']} 逻辑 case，包含共享 correction 的不同 sR）。"
        "宽表历史 E_proj 只作完整算子比较，不进入这项严格定义的 tightness 统计。")
    add(f"按实际独立 table/config correction 分组后，正范数核查 {independent['verified_positive_norms']} 组，"
        f"tightness 中位数 {fmt(independent['tightness']['median'])}、范围 "
        f"{fmt(independent['tightness']['min'])}–{fmt(independent['tightness']['max'])}。"
        f"补充使用相对 correction norm > {fmt(independent['correction_relative_machine_floor'])} 的明确显示尺度，"
        f"保留 {independent['positive_norms_above_machine_floor']} 组，"
        f"tightness 中位数 {fmt(independent['tightness_above_machine_floor']['median'])}、范围 "
        f"{fmt(independent['tightness_above_machine_floor']['min'])}–{fmt(independent['tightness_above_machine_floor']['max'])}；"
        f"另有 {independent['positive_norms_at_or_below_machine_floor']} 组图修正范数位于该尺度以下。"
        "这是额外的诊断分层，不删除 C、不改 B、也不是新的数学容差。")
    finite=[r for r in distinct_corrections if verified(r) and number(r.get("tightness")) is not None]
    if finite:
        extreme=max(finite,key=lambda r:r["tightness"])
        analysis["maximum_tightness_case"]={key:extreme.get(key) for key in ("case_id","parameter_id","config","M","p","xi","rho","kappa","delta","B","correction_norm","normGammaD","tightness","negative_blocks")}
        relative=extreme["correction_norm"]/extreme["normGammaD"] if number(extreme.get("normGammaD")) else None
        add(f"最大 ratio 的代表是 `{extreme.get('config')}`、M={fmt(extreme.get('M'))}、p={fmt(extreme.get('p'))}："
            f"delta={fmt(extreme.get('delta'))}、负块 {fmt(extreme.get('negative_blocks'))}，"
            f"B={fmt(extreme.get('B'))}，真实图修正范数={fmt(extreme.get('correction_norm'))}，"
            f"相对图修正={fmt(relative)}。这里 C 的表尺度本身并非机器微小量；"
            "图修正很小而 degree 含 self，可使上界极保守；不能把这种 below-roundoff 的巨大 ratio 当作有效选参预算精度。")
    if outlier:
        fraction=number(outlier.get("self_degree_trace_fraction"))
        add(f"独立离线审计 `results/tightness_outlier_diagnostic.json` 复算同一 correction graph norm "
            f"{fmt(outlier.get('correction_norm_recomputed'))}，与保存值一致；self degree 的 trace 占比为 "
            f"{fmt(fraction*100 if fraction is not None else None,12)}%，self 与 total degree 最大特征值分别为 "
            f"{fmt(outlier.get('self_degree_max_eigenvalue'),12)} / {fmt(outlier.get('total_degree_max_eigenvalue_direct'),12)}。"
            f"先将 W_ii 置零再直接求和的非 self degree 最大特征值只有 {fmt(outlier.get('offdiagonal_degree_max_eigenvalue'))}，"
            f"非 self / total 最大 degree 比 {fmt(outlier.get('offdiagonal_vs_total_max_degree_ratio'))}。"
            "因此这个极端保守性确由含 self 的 degree 上界主导，且图修正分母落在完整算子比较的浮点尺度；"
            "不是 tiny active C，也不是用两个大 degree 相减造成的灾难性相消。该离线解释未修改任何 bound 或生产方法。")
    add(figure(args,"fig1a_complete_error_bounds","图 1a：同 p 随 M 的完整经验误差、投影比较量及三个 correction bound"))
    add(figure(args,"fig1b_mesh_error_bounds","图 1b：独立 mesh-only 经验误差；correction 与界仍以同一 ||Gamma_D||₂ 归一化"))
    add(figure(args,"fig2_tightness_delta_negative_blocks","图 2：真实 correction-only tightness、delta 与负块比例，包含 p2/3/5/7/9"))
    add(figure(args,"fig3_parameter_bounds_activity","图 3：受控 xi/sF/p/rho/kappa 扫描的经验误差、界与投影活跃性"))
    add(figure(args,"fig3b_parameter_bounds_activity_moderate","图 3b：补充 moderate 的 M/p/rho/kappa 受控扫描，保留明显活跃情形"))
    add("曲线以九配置（uniform/moderate/strong × 三 seed）的中位数与 min–max 表示，"
        "逐图 CSV 保留参与 case_id/parameter_id；顶部色块表示 active/mixed/inactive。"
        "xi panel 使用其独立共享参数 preset，其余优先 baseline controlled sweep；跨 panel 不是同一物理设置的速度比较。"
        "本轮不拟合收敛阶：投影开关过渡处原先 12/26 等 formal slopes 不作为高阶证书。")
    add("原历史数据没有 controlled moderate sF 扫描，因此不虚构该 panel；其 sF 扫描只在 baseline 图中展示，"
        "moderate 补图用真实存在的 M/p/rho/kappa 四组。独立 xi preset 的全不活跃结果如实保留，"
        "不通过跨 M 混合验证候选来制造 xi 活跃曲线。")
    add("## 4. Winner、active bad case 与经验误差预算\n")
    add(md_table(["case组","配置数","max E_old","max E_new","max E_proj","max B_rel","max 经验 envelope","envelope≤1e-4","E_new≤1e-4","tightness中位/范围"], [
        [label,record["rows"],fmt(record["E_old"]["max"]),fmt(record["E_new"]["max"]),fmt(record["E_proj"]["max"]),
         fmt(record["B_rel"]["max"]),fmt(record["empirical_envelope"]["max"]),
         f"{record['empirical_budget_pass']}/{record['rows']}",f"{record['empirical_new_error_pass']}/{record['rows']}",
         f"{fmt(record['tightness']['median'])} / [{fmt(record['tightness']['min'])},{fmt(record['tightness']['max'])}]"]
        for label,record in analysis["selected_cases"].items()]))
    for label,record in analysis["selected_cases"].items():
        par=record["parameters"]
        add(f"`{label}` 参数："+", ".join(f"{key}={fmt(par.get(key))}" for key in ("N","rho","kappa","xi","sR","sF","M","p"))+"。")
        if record["rows"]:
            add(f"该组 {record['empirical_budget_pass']}/{record['rows']} 个配置满足 measured E_old+B_rel≤1e-4，"
                f"而实际 E_new 有 {record['empirical_new_error_pass']}/{record['rows']} 个达标。"
                "这直接量化上界提供预算的范围与保守性：经验 envelope 未通过不能反推算法实际误差未达标；"
                "通过也只对该数值 reference 的经验预算成立，不能升级为 certified total error。")
    add(figure(args,"fig4_winner_activecase_and_bound_cost","图 4：winner/active bad case 的经验误差、correction bound 与本轮增量 setup"))
    add(figure(args,"fig5_empirical_budget_envelope","图 5：逐配置经验 triangle envelope 与 1e-4 预算，显示保守性而非总误差证书"))
    add("## 5. 新增成本、测试和结论边界\n")
    timing_rows=[]
    for label,record in analysis["selected_cases"].items():
        for phase,values in record["timing"].items():
            if values["n"]:
                timing_rows.append([label,phase,values["n"],milliseconds(values["median"]),
                    f"[{milliseconds(values['q25'])},{milliseconds(values['q75'])}]",
                    "同一 repeat 跨配置共享，只计一次" if values["parameter_sample_shared_across_configs"] else "包含构型与重复变化"])
    add(md_table(["case组","新增阶段","独立样本数","中位 / ms","IQR / ms","统计范围"],timing_rows))
    add("这些时间来自本轮 isolated_incremental_bound_setup，保留旧 cached matvec 完全不变。"
        "parameter setup/correction FFT 的同一次参数表样本可供九个构型共享，不能算作九个独立 FFT 样本。"
        "configuration setup、stencil、density FFT、degree convolutions 与小 degree eigensolve 单独列出；"
        "阶段包含关系与独立中位数不相加。不与 PR #4 的旧 matvec/setup 计时拼接速度比，"
        "也不据此声称新方法或加速。负载与线程上下文见 `results/timing_context.json`。")
    add("运行过程中曾在合并输出阶段遇到 JSON 序列化问题；修复后仅重新执行 merge/postprocess，"
        "完整参数表 checkpoint 保留，最终成功摘要见 merge.log 与 verification_summary.json，计时见 timing.log。"
        "保留初次 run.log 作为过程证据，不把早期输出异常解释为数学上界或单元测试失败。")
    logs=sorted(set(args.results.glob("*test*.log"))|set(HERE.glob("*test*.log")))
    test_rows=[]
    for path in logs:
        INPUTS.append(path); content=path.read_text(encoding="utf-8",errors="replace")
        counts=re.findall(r"Ran\s+(\d+)\s+tests?",content)
        passed=bool(re.search(r"^OK\s*$",content,re.M)) and not bool(re.search(r"^FAILED",content,re.M))
        test_rows.append([path.name,counts[-1] if counts else "未识别", "通过" if passed else "需检查日志"])
    add(md_table(["测试日志","测试数","结果"],test_rows))
    add("本次独立测试覆盖了 N*delta、2Dmax、合并 bound、self cancellation、奇偶 M、零模、"
        "对称/PSD correction 与独立 correction-only action；实际 pass/failure 以上表日志为准。"
        "数学结构与任意合法参数 PSD 保证延续既有证明，不由有限样本 eigenvalues 代替。")
    pending=[]
    if not bounds:pending.append("尚无 bounds.csv。")
    expected=catalog.get("catalog_grid_source_rows")
    if number(expected) is not None:
        # The catalog may contain a final-timing aggregate without a configuration.
        add(f"来源 catalog 记录 {fmt(expected)} 行，最终 bounds 只处理有真实构型的条目；"
            "aggregate timing 行不作为单构型误差数据。最终覆盖与分组见 verification_summary.json 和 analysis_summary.json。")
    if len(plots.get("generated",[]))<7:pending.append("七组图（含 mesh-only、moderate 和经验预算补图）尚未全部保存。")
    if not raw:pending.append("新增 bound setup 的独立计时尚未保存。")
    if not test_rows:pending.append("测试日志尚未保存，不能声称测试已通过。")
    if not final_audit:pending.append("结束时旧文件 hash 审计尚未保存。")
    elif final_audit.get("ok") is False or final_audit.get("status") in ("partial_fail","fail","failed"):
        pending.append("现存最终审计仍记录未完成项；图/报告保存后需重跑 final_audit.py。")
    add("尚未完成/覆盖限制：" + ("；".join(pending) if pending else "已保存主流程产物，仍保留如下覆盖限制。"))
    add(f"活跃宽表中 {g['uncomputed_active_norms']} 个 case 未构造 true correction-only dense norm，"
        "对应 tightness 留空；不得用 historical full-difference norm 冒充。"
        "所有 total certified bounds unavailable。degree bound 的预算价值限于本次模型和已测设置；"
        "本轮不推出保 PSD 高阶新方案、不修改投影，也不作速度替代结论。")
    if final_audit:
        add(f"结束审计文件已保存，状态 `{final_audit.get('status','未注明')}`、issues={len(final_audit.get('issues',[]))}；旧源码、CSV、图和 reference "
            "未改动的具体逐文件 hash/pass 结果见该审计文件。")
    resume=args.results/"resume.log"
    if resume.exists():
        INPUTS.append(resume)
        content=resume.read_text(encoding="utf-8",errors="replace")
        add(f"checkpoint 恢复已实跑，`results/resume.log` 中记录 cached {content.count('cached')} 次；"
            "这是复用既有参数表结果的恢复检查，不是重新执行同一批计算。")
    add("## 复现\n\n在仓库根目录运行，compute 按参数表 checkpoint 恢复；仅补本轮目录的产物：")
    prefix="benchmarks/method_A_clustered/validation_exp1/deconv_psd_comparison/projection_bounds"
    add("```bash\n"+f"python3 -B {prefix}/audit_sources.py\n"
        +f"python3 -B {prefix}/test_bounds.py\n"
        +f"python3 -B {prefix}/run_bounds.py compute --workers 2\n"
        +f"python3 -B {prefix}/run_bounds.py merge\n"
        +f"python3 -B {prefix}/run_bounds.py timing --repeats 9\n"
        +f"python3 -B {prefix}/plot_results.py\n"
        +f"python3 -B {prefix}/report_generator.py\n"
        +"```\n")
    add("入口、最终审计命令与输出索引见 [README.md](README.md)。"
        "本报告的逐数据集/参数表/selected case/参数扫描分析保存为 `results/analysis_summary.json`；"
        "所有图附 PNG、SVG 与 `figures/plot_data/` 中的源 CSV。")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text("\n\n".join(text)+"\n",encoding="utf-8")
    # Include any late-added extreme-case diagnostics in the saved analysis.
    (args.results/"analysis_summary.json").write_text(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    manifest=dict(created_at=analysis["created_at"],report=str(args.output),
            inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
                    for p in sorted(set(INPUTS)) if p.exists() and p.name not in ("audit_final.json","final_audit.json")],
            audit_status_consulted=dict(status=final_audit.get("status","unavailable"),
                    ok=final_audit.get("ok"),issues=len(final_audit.get("issues",[])),
                    dependency_kind="status-only; no fixed SHA to avoid a report/audit artifact-hash cycle"))
    args.output.with_suffix(".sources.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(report=str(args.output),source_rows=len(bounds),unique_cases=len(distinct),pending=pending),ensure_ascii=False))


if __name__=="__main__":main()
