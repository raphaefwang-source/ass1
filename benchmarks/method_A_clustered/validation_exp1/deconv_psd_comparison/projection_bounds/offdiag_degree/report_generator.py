"""Chinese off-diagonal-degree report and analysis JSON from new saved data.

This does not import or change the published PPPM method, matvec, prior CSVs,
or prior plots. Empirical envelopes are not certified total-error bounds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("offdiag_local_plots",HERE/"plot_results.py")
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
INPUTS=[]
num=P.num;flag=P.flag;first=P.first;verified=P.verified


def read(path):
    if path.exists():INPUTS.append(path)
    return P.read(path) if path.exists() else []


def jsread(path):
    if not path.exists():return {}
    INPUTS.append(path)
    try:return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:return {"read_error":"incomplete JSON checkpoint"}


def q(values,p):
    values=sorted(v for v in values if v is not None)
    if not values:return None
    at=(len(values)-1)*p;lo,hi=math.floor(at),math.ceil(at)
    return values[lo] if lo==hi else values[lo]*(hi-at)+values[hi]*(at-lo)


def stats(values):
    values=[num(v) for v in values];values=[v for v in values if v is not None]
    return dict(n=len(values),min=min(values) if values else None,q25=q(values,.25),median=q(values,.5),q75=q(values,.75),max=max(values) if values else None)


def fs(rows,key):return stats([r.get(key) for r in rows])


def fmt(value,digits=6):
    value=num(value);return "N/A" if value is None else f"{value:.{digits}g}"


def ms(value):return fmt(num(value)*1e3) if num(value) is not None else "N/A"


def table(headers,rows):
    if not rows:return "尚无已保存数据。"
    esc=lambda v:str(v).replace("|","\\|").replace("\n"," ")
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]+
                     ["| "+" | ".join(esc(v) for v in row)+" |" for row in rows])


def isactive(row):return flag(row.get("active")) or (num(row.get("negative_blocks")) or 0)>0


def unreliable(row):return flag(first(row,"cancellation_unreliable_raw","cancellation_unreliable","subtraction_unreliable","cancellation_flag"))


def fallback(row):return flag(first(row,"stable_fallback_used","fallback_used"))


def summary(rows,epsilon=1e-4):
    positive=[r for r in rows if verified(r) and (num(r.get("correction_norm")) or 0)>0]
    floor=512*2.220446049250313e-16
    resolved=[r for r in positive if (num(r.get("true_correction_rel")) or 0)>floor]
    good=lambda key:sum(num(r.get(key)) is not None and r[key]<=epsilon for r in rows)
    output=dict(rows=len(rows),n_configs=len({r.get("config") for r in rows}),
        active=sum(isactive(r) for r in rows),inactive=sum(not isactive(r) for r in rows),
        literal_zero=sum(flag(r.get("exact_zero")) or num(r.get("correction_norm"))==0 for r in rows),
        true_positive_norms=len(positive),unavailable_active_norms=sum(isactive(r) and num(r.get("correction_norm")) is None for r in rows),
        correction_norm_sources=dict(Counter(str(r.get("correction_norm_source")) for r in rows)),
        raw_cancellation_flagged=sum(unreliable(r) for r in rows),stable_fallback_used=sum(fallback(r) for r in rows),
        stable_fallback_unavailable=sum(flag(r.get("stable_fallback_unavailable")) for r in rows),
        improved_bound_unavailable=sum(num(r.get("B_improved")) is None for r in rows),
        improved_envelope_available=sum(num(r.get("new_empirical_envelope")) is not None for r in rows),
        degreeoff_methods=dict(Counter(str(first(r,"degreeoff_method","degree_off_method","offdiag_method")) for r in rows)),
        epsilon=epsilon,old_empirical_budget_pass=good("old_empirical_envelope"),new_empirical_budget_pass=good("new_empirical_envelope"),
        measured_new_error_pass=good("E_new"),
        measured_pass_new_envelope_fail=sum(num(r.get("E_new")) is not None and r["E_new"]<=epsilon and
            num(r.get("new_empirical_envelope")) is not None and r["new_empirical_envelope"]>epsilon for r in rows),
        old_tightness=fs(positive,"B_old_tightness"),improved_tightness=fs(positive,"B_improved_tightness"),
        relative_machine_display_floor=floor,positive_norms_above_display_floor=len(resolved),
        old_tightness_above_display_floor=fs(resolved,"B_old_tightness"),
        improved_tightness_above_display_floor=fs(resolved,"B_improved_tightness"),
        improved_exceeds_old=stats([r["B_improved"]-r["B_old"] for r in rows if num(r.get("B_improved")) is not None and num(r.get("B_old")) is not None and r["B_improved"]>r["B_old"]]),
        true_correction_exceeds_improved=stats([r["correction_norm"]-r["B_improved"] for r in positive if num(r.get("B_improved")) is not None and r["correction_norm"]>r["B_improved"]]))
    for key in ("E_old","E_new","E_proj","B_simple_rel","B_total_rel","B_off_rel","B_old_rel","B_improved_rel",
                "old_empirical_envelope","new_empirical_envelope","self_degree_trace_fraction","degree_off_total_ratio"):
        output[key]=fs(rows,key)
    diagnostics={key for r in rows for key in r if any(t in key.lower() for t in ("residual","violation","min_eig","error","difference","cancellation","roundoff"))}
    output["numeric_diagnostics"]={key:fs(rows,key) for key in sorted(diagnostics) if fs(rows,key)["n"]}
    return output


def timing_stats(raw,group,case):
    out={};ids={r.get("case_id") for r in group};pars={r.get("parameter_id") for r in group}
    for phase in sorted({str(r.get("phase")) for r in raw}):
        samples=[r for r in raw if r.get("phase")==phase and num(r.get("seconds")) is not None and
            (P.label(r)==case or (r.get("case_id") and r["case_id"] in ids) or (r.get("parameter_id") and r["parameter_id"] in pars))]
        samples=P.independent_samples(samples,phase)
        out[phase]=dict(stats([r["seconds"] for r in samples]),shared_parameter_sample="parameter" in phase or phase=="correction_fft")
    return out


def image(args,name,caption):
    path=args.figures/f"{name}.png"
    if not path.exists():return f"图 `{name}` 尚未生成。"
    try:link=path.relative_to(args.output.parent).as_posix()
    except ValueError:link=path.resolve().as_posix()
    return f"![{caption}]({link})\n\n{caption}。"


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--results",type=Path,default=HERE/"results")
    parser.add_argument("--figures",type=Path,default=HERE/"figures");parser.add_argument("--output",type=Path,default=HERE/"report.md")
    args=parser.parse_args()
    bounds=read(args.results/"bounds.csv")
    if not bounds:bounds=read(args.results/"offdiag_bounds.csv")
    cases=read(args.results/"cases_summary.csv");raw=read(args.results/"timing_raw.csv")
    small_stability=read(args.results/"smallN_stability_cases.csv")
    small_stability_summary=jsread(args.results/"smallN_stability_summary.json")
    catalog=jsread(HERE/"source_catalog.json");checks=jsread(args.results/"verification_summary.json")
    numerical=jsread(args.results/"numerical_audit_summary.json");context=jsread(args.results/"timing_context.json")
    plots=jsread(args.figures/"plot_status.json");tests=jsread(args.results/"tests_summary.json")
    resume=jsread(args.results/"timing_resume_check.json")
    resume_log=args.results/"resume.log"
    cached_tables=None
    if resume_log.exists():
        INPUTS.append(resume_log)
        cached_tables=len(re.findall(r"^\d+/\d+.*'cached'",resume_log.read_text(encoding="utf-8"),re.M))
    final=jsread(args.results/"audit_final.json") or jsread(HERE/"audit_final.json")
    distinct=P.unique(bounds);bycorrection={}
    for row in distinct:bycorrection.setdefault((row.get("parameter_id"),row.get("config")),row)
    correction=list(bycorrection.values());groups=P.selected(bounds,cases)
    analysis=dict(created_at=datetime.now(ZoneInfo("America/Detroit")).isoformat(timespec="seconds"),
        source_rows=len(bounds),logical_complete_cases=len(distinct),table_configuration_groups=len(correction),
        parameter_tables=len({r.get("parameter_id") for r in distinct}),source_row_summary=summary(bounds),
        logical_case_summary=summary(distinct),correction_group_summary=summary(correction),datasets={},selected_cases={},
        verification_summary=checks,numerical_audit_summary=numerical,
        resume_verification=dict(cached_parameter_tables=cached_tables,timing=resume),
        small_N_offline_forced_stability=dict(saved_summary=small_stability_summary,rows=small_stability,
            n=len(small_stability),auto_available=sum(flag(r.get("auto_numerical_bound_available")) for r in small_stability),
            auto_unavailable=sum(not flag(r.get("auto_numerical_bound_available")) for r in small_stability),
            forced_exceeds_default_budget=sum(flag(r.get("forced_exceeds_default_query_budget")) for r in small_stability),
            auto_error_relative_true_norm=fs(small_stability,"auto_error_relative_true_norm"),
            forced_error_relative_true_norm=fs(small_stability,"forced_error_relative_true_norm"),
            forced_error_max_spectral=fs(small_stability,"forced_error_max_spectral"),
            forced_strict_norm_violation_abs=fs(small_stability,"forced_strict_norm_violation_abs"),
            forced_positive_setup_seconds=fs(small_stability,"forced_positive_setup_seconds"),
            role="offline forced positive audit only; default auto-unavailable rows and primary batch remain unchanged"),
        theory_certified_total_bound_available=False,theory_certified_candidates="unavailable: no certified epsilon_old",
        candidate_rule="bound failure is not a rejection rule for an independent-reference empirical-pass candidate")
    small_checked=[r for r in correction if flag(r.get("audit_small_checked"))]
    small_unavailable=[r for r in small_checked if flag(r.get("audit_small_stable_fallback_unavailable")) or
        (r.get("audit_small_bound_numerical_available") not in (None,"") and not flag(r["audit_small_bound_numerical_available"]))]
    small_available=[r for r in small_checked if r not in small_unavailable and
        (flag(r.get("audit_small_bound_numerical_available")) or flag(r.get("audit_small_literal_zero")))]
    analysis["small_N_audit_availability"]=dict(checked=len(small_checked),available=len(small_available),
        unavailable=len(small_unavailable),unknown=len(small_checked)-len(small_available)-len(small_unavailable),
        unavailable_case_ids=[r.get("case_id") for r in small_unavailable],
        unavailable_parameters=[{k:r.get(k) for k in ("parameter_id","config","M","p","kappa","rho","active_correction_offsets")} for r in small_unavailable],
        available_error_relative_true_norm=fs(small_available,"audit_small_error_relative_true_correction_norm"),
        available_absolute_spectral_error=fs(small_available,"audit_small_off_error_max_spectral"),
        unavailable_raw_error_relative_true_norm=fs(small_unavailable,"audit_small_raw_error_relative_true_correction_norm"),
        unavailable_returned_error_relative_true_norm=fs(small_unavailable,"audit_small_error_relative_true_correction_norm"))
    for dataset in sorted({str(r.get("dataset")) for r in bounds}):
        rr=[r for r in bounds if str(r.get("dataset"))==dataset];analysis["datasets"][dataset]=dict(source_rows=len(rr),**summary(P.unique(rr)))
    for case,group in groups.items():
        analysis["selected_cases"][case]=dict(summary(group),parameters={k:group[0].get(k) for k in P.PARAMETERS} if group else {},
            case_ids=[r.get("case_id") for r in group],timing=timing_stats(raw,group,case))
    args.results.mkdir(parents=True,exist_ok=True)
    (args.results/"analysis_summary.json").write_text(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    text=[];add=text.append;g=analysis["correction_group_summary"]
    add("# Self-excluded degree bound：验证、预算与数值相消\n")
    add(f"生成时间：{analysis['created_at']}。本轮仅在 `offdiag_degree/` 增量验证既有 correction 的 self-excluded degree bound；"
        "不改变 Grid PSD、Ewald split、物理核、旧 PPPM、cached matvec、Lanczos/OU 或已发布文件。"
        "完整任务见 [requirements_github.md](requirements_github.md)。")
    add("## 1. 数学结论与浮点实现\n")
    add("证明与适用条件见 [proof.md](proof.md)。沿用 correction-only 物理规范 "
        "`C_hat=FFT(C)`，不额外乘/除 h³，保留 zero mode。因非负 stencil 和 `C_d` PSD，所有 `C_ij` PSD，"
        "`D_total=Σ_j C_ij`，`D_off=Σ_(j≠i) C_ij=D_total−C_ii`。精确算术下，")
    add("```text\nB_simple   = N*delta\nB_total    = 2*max_i lambda_max(D_total_i)\n"
        "B_off      = 2*max_i lambda_max(D_off_i)\nB_old      = min(B_simple,B_total)\n"
        "B_improved = min(B_simple,B_total,B_off) = min(B_simple,B_off)\n"
        "||Delta Gamma||_2 <= B_improved <= B_old.\n```\n")
    add("self response 在 degree-minus-weight graph 中代数抵消；去除 self 只收紧误差界，不改变 graph 或 PSD 结构。"
        "`C_ii` 用逐轴 stencil autocorrelation 与真实模 M support 求和，不默认 p^6 双重三维循环或稠密粒子图。"
        "解析式 `D_off=D_total−C_ii` 在 self 主导时会相消；原始相减、数值标记、稳定 fallback 和独立先置 Wii=0 再求和的审计必须分开。"
        "任何额外 fallback 成本均计入本轮 setup，不以静默 clipping/floor 伪造 PSD 数值证书。"
        "精确算术定理不等同于 float64 FFT/eigensolve 的有向舍入或区间证书。")
    add("## 2. 覆盖、真值与相消检查\n")
    add(f"保存 {len(bounds)} 条来源记录、{len(distinct)} 个逻辑完整算子 case、{len(correction)} 个 table/config correction、"
        f"{analysis['parameter_tables']} 个参数表。不同 sR 可共享 correction，不把复用范数当作独立计算。")
    add(table(["口径","记录","active/inactive","true正范数","literal zero","活跃wide norm unavailable"],[
        [name,r["rows"],f"{r['active']}/{r['inactive']}",r["true_positive_norms"],r["literal_zero"],r["unavailable_active_norms"]]
        for name,r in [("逻辑完整算子",analysis["logical_case_summary"]),("实际table/config",g)]]))
    add(table(["数值检查（table/config）","观测"],[
        ["fresh small-N 核查",checks.get("small_N_checked_table_configurations","N/A")],
        ["small-N 明确 unavailable（非主N256结果）",checks.get("small_N_unavailable_bounds",analysis["small_N_audit_availability"]["unavailable"])],
        ["full-N focus 核查（含 literal zero）",checks.get("full_N_checked_table_configurations","N/A")],
        ["两粒子 literal 双stencil 核查",checks.get("literal_N2_checked_tables","N/A")],
        ["raw subtraction 相消标记",g["raw_cancellation_flagged"]],["stable fallback 使用",g["stable_fallback_used"]],
        ["stable fallback unavailable",g["stable_fallback_unavailable"]],["改进界 unavailable",g["improved_bound_unavailable"]],
        ["B_improved>B_old 浮点诊断条数",g["improved_exceeds_old"]["n"]],
        ["true correction norm>B_improved 诊断条数",g["true_correction_exceeds_improved"]["n"]],
        ["最大 true norm 超界（已核查部分，绝对）",fmt(checks.get("max_true_norm_violation_abs",g["true_correction_exceeds_improved"]["max"]))]]))
    add("这里只用 true correction-only dense norm 或 literal-zero 真值；旧 full-operator `E_proj` 不替代 true norm。"
        "wide norm 空值、失败和不可靠结果保留；B_off/B_improved unavailable 时留空，不用 raw subtraction 补空。数值诊断细项、fresh direct Doff 审计、对称性/min eigenvalue 与 residual "
        "见 `results/verification_summary.json`、独立数值审计及下列实际字段；不能仅凭图片宣布相消安全。")
    available=g["numeric_diagnostics"]
    selected_diagnostics=("audit_full_raw_error_relative_true_correction_norm","audit_full_error_relative_true_correction_norm",
        "audit_small_raw_error_relative_true_correction_norm","audit_small_error_relative_true_correction_norm",
        "raw_offdiag_min_eigenvalue","offdiag_min_eigenvalue","cancellation_index_raw","raw_relative_roundoff_indicator",
        "B_total_source_residual_abs","B_old_source_residual_abs","audit_literal_pair_max_component_residual")
    add(table(["关键实际数值诊断字段","n","min","median","max"],[
        [key,available[key]["n"],fmt(available[key]["min"]),fmt(available[key]["median"]),fmt(available[key]["max"])]
        for key in selected_diagnostics if key in available]))
    add("完整数值诊断字段范围保存在 analysis_summary.json，未因图表简化而删除失败/不可用记录。"
        "full-N 核查总数包含可直接证明为零的 correction；active dense 与 literal-zero 的口径分开保留。")
    sn=analysis["small_N_audit_availability"]
    add(f"**small-N 与主 N=256 的可用性分开判断。** small-N 检查 {sn['checked']} 组，"
        f"可用 {sn['available']}、明确 unavailable {sn['unavailable']}、未注明 {sn['unknown']}。"
        "wide correction support 触发稳定查询预算上限时，auto 路径只保留 raw 诊断并明确不提供数值 bound；"
        "这些行不是稳定路径成功，也不冒充主 N=256 的 bound 失败。"
        f"available small-N 的 error/true correction norm 最大 {fmt(sn['available_error_relative_true_norm']['max'])}，"
        f"对应集合的绝对谱误差最大 {fmt(sn['available_absolute_spectral_error']['max'])}；"
        f"unavailable small-N 的 raw 相对误差最大 {fmt(sn['unavailable_raw_error_relative_true_norm']['max'])}，"
        "保留该大值及 unavailable 标志，不用它描述可用稳定路径的误差。完整字段表中的 small-N 全体最大值因此可能非常大。"
        "这些都是浮点精度诊断，不是区间证书。")
    if sn["unavailable_parameters"]:
        add(table(["small-N unavailable 参数表","配置","M","p","kappa","rho","active offsets"],[
            [r.get("parameter_id"),r.get("config"),fmt(r.get("M")),fmt(r.get("p")),fmt(r.get("kappa")),fmt(r.get("rho")),fmt(r.get("active_correction_offsets"))]
            for r in sn["unavailable_parameters"]]))
    supplemental=analysis["small_N_offline_forced_stability"]
    if small_stability:
        add(f"另保存 **{len(small_stability)} 个 targeted small-N 离线稳定核查**：选择 auto error/true norm≥1e-8 或 auto bound unavailable 的 case，"
            "显式强制 positive-occupancy 路径，允许超出默认查询预算。它不修改默认阈值、主 CSV 或主实验的 unavailable 状态，"
            "不是新的默认生产路径；计时也不与主 setup 批次合并。"
            f"其中原 auto unavailable {supplemental['auto_unavailable']} 个，强制路径超默认预算 {supplemental['forced_exceeds_default_budget']} 个。"
            f"强制后最大 error/true norm 为 {fmt(supplemental['forced_error_relative_true_norm']['max'])}，"
            f"最大绝对谱误差 {fmt(supplemental['forced_error_max_spectral']['max'])}；"
            f"额外单次强制 setup 为 [{fmt(supplemental['forced_positive_setup_seconds']['min'])},"
            f"{fmt(supplemental['forced_positive_setup_seconds']['max'])}] **秒**。")
        add(table(["参数表/配置（N=4子集）","auto bound可用","auto error/true norm","强制 error/true norm","额外强制setup/秒","超默认预算"],[
            [str(r.get("parameter_id"))+" / "+str(r.get("config")),flag(r.get("auto_numerical_bound_available")),
             fmt(r.get("auto_error_relative_true_norm")),fmt(r.get("forced_error_relative_true_norm")),
             fmt(r.get("forced_positive_setup_seconds")),flag(r.get("forced_exceeds_default_query_budget"))]
            for r in small_stability]))
        add(f"强制路径保存的最大严格 `true norm−B_improved` 正部为 "
            f"{fmt(supplemental['forced_strict_norm_violation_abs']['max'])}（浮点诊断，未用 floor/inflation 消去）。"
            "即使误差落在机器精度尺度，也不等同于有向舍入数值证书。特别是 wide case 的数十秒成本只属于该额外离线核查，"
            "不能称免费修复，也不能据此将默认 auto 的三项 unavailable 改成 passed。详见 "
            "`results/smallN_stability_cases.csv`、`smallN_stability_summary.json` 与 `audit_small_stability.py`。")
    add("实际采用的 degree-off 路径计数：`"+json.dumps(g["degreeoff_methods"],ensure_ascii=False)+"`。"
        "普通 autocorrelation 路径与稳定 fallback 的复杂度/触发条件以实现和 proof 为准；没有将 fallback 成本隐藏到旧 matvec 中。")
    add("## 3. 界的收紧与 self 主导区\n")
    add(table(["实际table/config正范数组","n","tightness min","median","max"],[
        ["old",g["old_tightness"]["n"],fmt(g["old_tightness"]["min"]),fmt(g["old_tightness"]["median"]),fmt(g["old_tightness"]["max"])],
        ["improved",g["improved_tightness"]["n"],fmt(g["improved_tightness"]["min"]),fmt(g["improved_tightness"]["median"]),fmt(g["improved_tightness"]["max"])]]))
    add(f"按相对 correction norm>{fmt(g['relative_machine_display_floor'])} 的诊断显示尺度另有 "
        f"{g['positive_norms_above_display_floor']} 组：old/improved tightness 中位数 "
        f"{fmt(g['old_tightness_above_display_floor']['median'])}/{fmt(g['improved_tightness_above_display_floor']['median'])}。"
        "这只分层解释 below-roundoff 的 ratio，不删除任何 C、不改 bound、不在 ratio 图强制取 max(1,ratio)。"
        "真实零修正的 tightness 和 zero/zero degree ratio 未定义；图中真零使用明确 floor/空心倒三角，CSV 保留真实数值。")
    add(image(args,"fig1_correction_and_degree_bounds","图 1：p2/3/5/7/9 随 M 的 true correction norm 与五个界，统一 ||Gamma_D||₂ 归一化"))
    add(image(args,"fig2_tightness_self_and_offdegree","图 2：old/improved tightness、self trace fraction 与 off/total 最大degree比"))
    add("曲线是 uniform/moderate/strong × 三 seed 的中位数与 min–max，逐图 CSV 保留 case_id/parameter_id。"
        "不在投影开关或 mixed-transition 上解释 12/26 等 formal slopes 为收敛阶。"
        "self-dominated outlier 的改善既需看真实 correction 图，也需看相消标记与 fresh direct offdiagonal 核查；"
        "不能把机器尺度上很大的 ratio 改善推广成所有实际精度场景。")
    add("## 4. Winner 与 active bad case：经验预算是否改善\n")
    add(table(["case组","n","max E_new","max old envelope","max new envelope","old pass","new pass","new envelope可用","实际 E_new pass"],[
        [case,r["rows"],fmt(r["E_new"]["max"]),fmt(r["old_empirical_envelope"]["max"]),fmt(r["new_empirical_envelope"]["max"]),
         f"{r['old_empirical_budget_pass']}/{r['rows']}",f"{r['new_empirical_budget_pass']}/{r['rows']}",r["improved_envelope_available"],f"{r['measured_new_error_pass']}/{r['rows']}"]
        for case,r in analysis["selected_cases"].items()]))
    for case,r in analysis["selected_cases"].items():
        add(f"`{case}` 参数："+", ".join(f"{k}={fmt(r['parameters'].get(k))}" for k in ("rho","kappa","xi","sR","sF","M","p"))+"。"
            f"经验 envelope 通过数由 {r['old_empirical_budget_pass']}/{r['rows']} 变为 {r['new_empirical_budget_pass']}/{r['rows']}；"
            f"独立 reference 的实际新误差通过数为 {r['measured_new_error_pass']}/{r['rows']}。")
    add(image(args,"fig3_winner_badcase_empirical_budgets","图 3：逐配置 empirical errors、两个 triangle envelope 与 1e-4；直接显示原 5/9 是否改善"))
    add("**界是 sufficient/conservative。** `E_old_measured+B_improved_rel` 仍只是相对于独立数值参考的经验 envelope，"
        "不是 certified total error。`epsilon_old^cert` 当前 unavailable，theory-certified candidates 无法据此认证；"
        "独立 reference 已实测达标的 empirical-pass candidates 不能因保守 bound 未通过自动淘汰。"
        "PSD 对任意合法参数/粒子位置的结构保证不依赖这个配置相关误差预算，"
        "bound 不在每次 cached/Lanczos matvec 中计算；选参目标仍应在等精度验证基础上挑成本最低者。")
    add("## 5. 新增 setup、测试与复现\n")
    cost=[]
    for case,r in analysis["selected_cases"].items():
        for phase,v in r["timing"].items():
            if v["n"]:cost.append([case,phase,v["n"],fmt(v["median"]),f"[{fmt(v['q25'])},{fmt(v['q75'])}]",
                "跨配置共享，按repeat去重" if v["shared_parameter_sample"] else "配置与repeat聚合"])
    add(table(["case组","新批次阶段","独立样本数","median/秒","q25–q75/秒","范围"],cost))
    for case,r in analysis["selected_cases"].items():
        slow=r["timing"].get("stable_fallback",{})
        if slow.get("n"):
            add(f"`{case}` 的 stable fallback 在独立计时批中的中位数为 {fmt(slow.get('median'))} **秒**，"
                f"q25–q75 为 [{fmt(slow.get('q25'))},{fmt(slow.get('q75'))}] 秒，"
                f"保存样本范围 [{fmt(slow.get('min'))},{fmt(slow.get('max'))}] 秒。"
                "稳定慢路径是明确的额外代价，不能以 bound 更紧就宣称计算免费。")
    add(image(args,"fig4_incremental_bound_setup_cost","图 4：新的 bound-only setup 独立批次，以秒和 log 轴显示慢 fallback，中位数与 q25–q75"))
    add("参数表样本跨配置共享时只计一次；配置样本包含位置和 repeat 变化。"
        "full-bound configuration、self autocorrelation、fallback 与 offdiag incremental 可有包含关系，独立中位数不相加。"
        "仅比较本轮新增 bound setup；不拼接 PR #4/5 的旧 cached matvec 时间宣称提速，不改变旧 action。")
    if cached_tables is not None or resume:
        add(f"恢复检查另存 `results/resume.log`：{cached_tables if cached_tables is not None else 'N/A'} 个参数表复用 cached checkpoint；"
            f"`results/timing_resume_check.json` 记录 raw timing/context 的 SHA 在恢复前后保持不变："
            f"`{resume.get('raw_timing_and_context_unchanged','unavailable')}`。恢复只重建 derived summary，不重测或静默覆写旧批次。")
    logs=sorted(set(args.results.glob("*test*.log"))|set(HERE.glob("*test*.log")));test_rows=[]
    for path in logs:
        INPUTS.append(path);content=path.read_text(encoding="utf-8",errors="replace");counts=re.findall(r"Ran\s+(\d+)\s+tests?",content)
        passed=bool(re.search(r"^OK\s*$",content,re.M)) and not bool(re.search(r"^FAILED",content,re.M))
        test_rows.append([path.name,counts[-1] if counts else "未识别", "通过" if passed else "需检查日志"])
    if tests and not test_rows:test_rows.append(["tests_summary.json",tests.get("tests_run","N/A"),"通过" if tests.get("ok") else "需检查"])
    add(table(["测试证据","测试数","结果"],test_rows))
    add("测试与独立审计覆盖 autocorrelation support 的周期 wrap、重复 indices、p=1、M<2p−1、奇偶网格、"
        "zero mode/self cancellation，以及不用大 degree 相减的 direct offdiagonal 对照。实际 pass/failure 与覆盖以保存日志/审计为准。")
    pending=[]
    if not bounds:pending.append("尚无新 bounds 数据。")
    if len(plots.get("generated",[]))<4:pending.append("四组图尚未齐全。")
    if not raw:pending.append("新增 setup 独立计时尚未保存。")
    if not small_stability or not small_stability_summary:pending.append("targeted small-N 额外离线稳定审计尚未保存完整。")
    if not tests and not logs:pending.append("测试结果尚未保存。")
    if not final:pending.append("最终独立审计尚未保存。")
    elif final.get("ok") is False or final.get("status") in ("partial_fail","fail","failed"):pending.append("现存终审仍有未完成项，需重跑。")
    add("未完成/覆盖限制："+("；".join(pending) if pending else "主流程产物已保存。")+
        f"活跃 wide norm 未核查 {g['unavailable_active_norms']} 组仍留空；certified total bound 永远标 unavailable，除非另有独立 epsilon_old^cert。")
    add(f"最终审计状态 `{final.get('status','unavailable')}`，issues={len(final.get('issues',[]))}；"
        "旧已发布文件、source catalog 与 reference 的保全以最终 hash 审计为准。")
    add("恢复命令与文件索引见 [README.md](README.md)。`results/analysis_summary.json` 包括双口径覆盖、数值相消/true-bound诊断、"
        "selected-case 预算与 setup IQR；各图 source CSV 和 `plot_status.json` 可独立复核。")
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text("\n\n".join(text)+"\n",encoding="utf-8")
    INPUTS.extend(path for path in (HERE/"proof.md",HERE/"requirements_github.md",HERE/"README.md",
        HERE/"plot_results.py",Path(__file__),HERE/"offdiag_bounds.py",HERE/"run_offdiag.py",HERE/"audit_small_stability.py") if path.exists())
    manifest=dict(created_at=analysis["created_at"],report=str(args.output),
        inputs=[dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(set(INPUTS)) if path.exists() and path.name not in ("audit_final.json","final_audit.json")],
        audit_status_consulted=dict(status=final.get("status","unavailable"),issues=len(final.get("issues",[])),dependency_kind="status-only; avoid report/audit SHA cycle"))
    args.output.with_suffix(".sources.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(report=str(args.output),source_rows=len(bounds),pending=pending),ensure_ascii=False))


if __name__=="__main__":main()
