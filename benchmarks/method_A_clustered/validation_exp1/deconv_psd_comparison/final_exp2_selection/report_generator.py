"""Generate the Chinese final Experiment 2 report from this run's saved data.

Unavailable targets stay unavailable. Empirical nine-configuration validation,
structural PSD, and a certified uniform error bound are distinct statements.
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
spec=importlib.util.spec_from_file_location("final_exp2_local_plots",HERE/"plot_results.py")
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
INPUTS=[]
num=P.num;flag=P.flag;first=P.first


def jsread(path):
    if not path.exists():return {}
    INPUTS.append(path)
    try:return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:return {"read_error":"incomplete checkpoint"}


def json_first(root,names):
    for name in names:
        if (root/name).exists():return jsread(root/name)
    return {}


def fmt(value,digits=6):
    v=num(value);return "N/A" if v is None else f"{v:.{digits}g}"


def stats(values):
    values=sorted(v for v in values if num(v) is not None)
    q25,median,q75=P.quantiles(values)
    return dict(n=len(values),min=values[0] if values else None,q25=q25,median=median,q75=q75,max=values[-1] if values else None)


def table(headers,rows):
    if not rows:return "尚无已保存数据。"
    esc=lambda v:str(v).replace("|","\\|").replace("\n"," ")
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]+
        ["| "+" | ".join(esc(v) for v in row)+" |" for row in rows])


def config_rows(data,row):
    return P.unique([r for r in data["validation"] if r.get("method")==row.get("method") and r.get("id")==row.get("id")],("method","id","config"))


def measured_best(data,epsilon):
    return {m:P.best_row(data["best"],m,epsilon) for m in P.METHODS}


def ratio(a,b,epsilon):
    if not a or not b or not P.feasible(a,epsilon) or not P.feasible(b,epsilon):return dict(available=False,reason="one method not measured/feasible")
    if not a.get("batch") or not b.get("batch") or a["batch"]!=b["batch"]:return dict(available=False,reason="different or unspecified timing batch")
    return dict(available=True,ratio_medians=a["median_matvec_s"]/b["median_matvec_s"],batch=a["batch"],
        interpretation="ratio of saved stage medians in one batch; not a paired confidence interval")


def image(args,name,caption):
    path=args.figures/f"{name}.png"
    if not path.exists():return f"图 `{name}` 尚未生成（见 plot_status.json）。"
    try:link=path.relative_to(args.output.parent).as_posix()
    except ValueError:link=path.resolve().as_posix()
    return f"![{caption}]({link})\n\n{caption}。"


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--results",type=Path,default=HERE/"results")
    parser.add_argument("--figures",type=Path,default=HERE/"figures");parser.add_argument("--output",type=Path,default=HERE/"report.md")
    args=parser.parse_args();data=P.inputs(args.results);INPUTS.extend(data["paths"])
    protocol=json_first(args.results,("protocol.json","search_protocol.json","experiment_context.json"))
    theory=jsread(HERE/"theory_source_audit.json")
    alias=jsread(args.results/"alias_calibration.json");budget=jsread(args.results/"search_budget.json")
    context=json_first(args.results,("timing_context.json","run_context.json"))
    runtime=jsread(args.results/"runtime_environment.json")
    cross_plan=jsread(args.results/"cross_plan.json");system_load=jsread(args.results/"system_load.json")
    structural=jsread(HERE/"structural_audit.json")
    checks=json_first(args.results,("verification_summary.json","checks_summary.json"))
    tests=json_first(args.results,("tests_summary.json","test_summary.json"))
    audit=json_first(args.results,("audit_final.json","final_audit.json")) or json_first(HERE,("audit_final.json","final_audit.json"))
    plots=jsread(args.figures/"plot_status.json")
    scaling_refs=[dict(file=str(path),**jsread(path)) for path in sorted((HERE/"scaling_references").glob("*.json"))]
    epsilons=list(P.EPSILONS)
    analysis=dict(created_at=datetime.now(ZoneInfo("America/Detroit")).isoformat(timespec="seconds"),
        scope="final four-step Grid PSD local joint selection; best tested, not global optimum",protocol=protocol,theory_source_audit=theory,alias_calibration=alias,search_budget=budget,timing_context=context,runtime_environment=runtime,
        cross_method_plan=cross_plan,system_load_snapshots=system_load,
        timing_isolation_scope="our own accuracy/reference/render workers are stopped; external processes including chrome_crashpad remain uncontrolled",
        checks=checks,independent_structural_audit=structural,input_rows={k:len(v) for k,v in data.items() if k!="paths"},targets={},candidate_groups={},
        structural_psd="guaranteed by unchanged nonnegative assignment and blockwise PSD construction in exact arithmetic; independent code checks retained",
        empirical_accuracy="max of nine saved full Gamma operator errors against periodic numerical direct reference",
        certified_uniform_total_accuracy="unavailable: no certified epsilon_old; empirical error is not a proof for arbitrary configurations",
        selection_constraints="no B_off/Ndelta/degree bounds/full operator eigenvalue PSD hard screen",amortization=[],
        audit_status_consulted=dict(status=audit.get("status","unavailable"),issues=len(audit.get("issues",[])),dependency_kind="status-only; no SHA dependency"))
    for e in epsilons:
        selected=measured_best(data,e);target={"epsilon":e,"methods":{},"speed_ratios":{}}
        for method,row in selected.items():
            entry=dict(available=False,reason="no target-wise measured result")
            if row:
                cfg=config_rows(data,row);entry=dict(row=row,available=P.feasible(row,e),n_saved_config_errors=len(cfg),configuration_errors=cfg,
                    measured_error_summary=stats([r.get("E_G") for r in cfg]),
                    projection_before_error_summary=stats([r.get("E_original") for r in cfg]),
                    projection_difference_summary=stats([r.get("E_proj") for r in cfg]),
                    projection_active=sum(flag(r.get("active")) or (num(r.get("negative_blocks")) or 0)>0 for r in cfg),
                    stages={phase:P.phase_stats(row,data["raw"],phase) for phase in ("table_setup","config_setup","matvec","near","spread_gather","fft_influence","mesh")})
                if not entry["available"]:entry["reason"]=str(first(row,"reason","status") or "error above target, unavailable, or isolated time not measured")
                stages=entry["stages"]
                if entry["available"] and all(stages[k]["median_s"] is not None for k in ("table_setup","config_setup","matvec")):
                    for reuse in (1,10,100,1000):
                        for napply in (1,10,100,1000):
                            analysis["amortization"].append(dict(epsilon=e,method=method,id=row.get("id"),batch=row.get("batch"),
                                table_reuse_steps=reuse,matvecs_per_step=napply,
                                measured_table_setup_s=stages["table_setup"]["median_s"],measured_config_setup_s=stages["config_setup"]["median_s"],measured_matvec_s=stages["matvec"]["median_s"],
                                derived_step_cost_s=stages["table_setup"]["median_s"]/reuse+stages["config_setup"]["median_s"]+napply*stages["matvec"]["median_s"],
                                role="linear model of measured medians; not measured dynamics trajectory"))
            target["methods"][method]=entry
        for ref in P.METHODS[1:]:target["speed_ratios"]["Grid/"+ref]=ratio(selected["PPPM-grid-PSD"],selected[ref],e)
        analysis["targets"][f"{e:.0e}"]=target
    paired_original=[r for r in data["ratios"] if r.get("method")=="PPPM-grid-PSD" and P.method_name(first(r,"reference_method","reference"))=="PPPM-original"]
    paired_b2=[r for r in data["ratios"] if r.get("method")=="PPPM-grid-PSD" and P.method_name(first(r,"reference_method","reference"))=="B2-positive"]
    paired_assessment=[]
    for r in paired_original:
        low=num(first(r,"ratio_ci95_low","ci95_low"));high=num(first(r,"ratio_ci95_high","ci95_high"))
        paired_assessment.append(dict(epsilon=r.get("epsilon"),median_paired_ratio=first(r,"ratio_median_paired_blocks","ratio_median","ratio"),low=low,high=high,
            includes_one=low is not None and high is not None and low<=1<=high))
    analysis["paired_Original_assessment"]=paired_assessment
    grouped=defaultdict(list);byid={r.get("id"):r for r in data["candidates"]}
    memberships=data["memberships"] or data["candidates"]
    for member in memberships:
        r=dict(byid.get(member.get("id"),{}),**member)
        grouped[(str(r.get("method")),str(r.get("epsilon")),str(r.get("stage")))].append(r)
    for key,rr in grouped.items():analysis["candidate_groups"][" / ".join(key)]=dict(rows=len(rr),unique_ids=len({r.get("id") for r in rr}),statuses=dict(Counter(str(r.get("status")) for r in rr)))
    analysis["scaling"]=data["scaling"];analysis["scaling_reference_metadata"]=scaling_refs;analysis["paired_ratio_source_rows"]=data["ratios"]
    analysis["availability_rows"]=data["availability"]
    analysis["validated_grid_projection_effect"]=P.validated_grid_projection(data)
    tested_p=sorted({num(r.get("p")) for r in data["candidates"] if num(r.get("p")) is not None})
    analysis["p_search_scope"]=dict(mathematical_legality="1 <= p <= M",actual_candidate_p=tested_p,p1_tested=1 in tested_p,
        initialization_scope="p>=2 and initial model cap 9; finite measured local/boundary neighborhood",
        reason="zeta(p) majorant finite only for p>1; p2 first continuous cardinal B-spline; p1 untested, not mathematically rejected",
        calibration_scope="empirical alias_C fitted from prior p3..5 complete errors; no p1/p2 guarantee")
    analysis["selection_hard_constraint_flag_violations"]=sum(flag(r.get("bound_used_as_constraint")) or flag(r.get("PSD_eigenvalue_used_as_constraint")) for r in data["candidates"])
    args.results.mkdir(parents=True,exist_ok=True)
    (args.results/"analysis_summary.json").write_text(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    P.write(args.results/"report_amortization.csv",analysis["amortization"])
    content=[];add=content.append
    add("# Experiment 2：最终四步 Grid PSD 联合选参\n")
    add(f"生成时间：{analysis['created_at']}。本轮独立目录只重跑最终选参验证，保留 PR #3–#6 的既有结果。"
        "主基准固定 N=256、rho=0.75、kappa=0.2、amplitude=1；九配置为 uniform/moderate/strong × 三 seed。"
        "需求原文见 [requirements_github.md](requirements_github.md)，复现和产物索引见 [README.md](README.md)。"
        "下文的最优一律指 **本轮搜索范围内、已测试候选中的最佳**，不声称全局最优。")
    grid_summary=[]
    for key,target in analysis["targets"].items():
        g=target["methods"]["PPPM-grid-PSD"];r=g.get("row",{})
        grid_summary.append([key,"实测可行" if g["available"] else "N/A",fmt(r.get("max_E_G")),fmt(r.get("median_matvec_s")),g.get("n_saved_config_errors",0),str(r.get("id") or g.get("reason"))])
    add(table(["目标 epsilon","Grid PSD状态","max E_Gamma","cached matvec/秒","已保存逐配置误差数","候选/限制"],grid_summary))
    if paired_assessment and all(r["includes_one"] for r in paired_assessment):
        lead_grid=[t["methods"]["PPPM-grid-PSD"] for t in analysis["targets"].values() if t["methods"]["PPPM-grid-PSD"]["available"]]
        inactive_lead="全部三个 winner 的投影 inactive；不能据此推出 active 投影保持高阶精度。" if len(lead_grid)==3 and all(num(r["row"].get("negative_blocks"))==0 for r in lead_grid) else ""
        add("**主结论：本轮三个精度档的 Grid/Original paired timing 区间均跨 1，Grid PSD 与 Original 的 cached full-action 速度相近，不能宣称 Grid PSD 更快。** "
            +inactive_lead+"它在本轮已测参数与九配置上保留了 Original 的同量级 cached 速度；这个结论限于当前有限搜索、计时设置和外部负载条件，"
            "不等于所有参数/构型或全局最优的速度结论。")
    if paired_b2:
        add("Grid 相对同精度 B2 的 cached-action 加速（取 paired Grid/B2 ratio 的倒数）："+
            "；".join(f"epsilon={fmt(r.get('epsilon'))}，{fmt(1/num(first(r,'ratio_median_paired_blocks','ratio_median','ratio')))}×" for r in paired_b2 if (num(first(r,"ratio_median_paired_blocks","ratio_median","ratio")) or 0)>0)+
            "。这是 cached-action 比较，不能替代包含 setup 的每步总成本。")
    dense_entries=[t["methods"]["direct-dense"] for t in analysis["targets"].values() if t["methods"]["direct-dense"]["available"]]
    if dense_entries:
        add(f"N=256 的已缓存 direct dense G@v 本轮为 {fmt(dense_entries[0]['row'].get('median_matvec_s'))} 秒，"
            "在该静态 cached-action 比较中更便宜；fresh 周期 dense assembly setup 未测，故其动态步总成本 N/A，不能据此比较完整动力学总成本。")
    ratio_text=[]
    for key,t in analysis["targets"].items():
        rr=t["speed_ratios"]["Grid/PPPM-original"]
        ratio_text.append(f"epsilon={key}："+(f"同批次 Grid/Original cached-time 中位数比 {fmt(rr['ratio_medians'])}" if rr["available"] else "速度比较 N/A（"+rr["reason"]+"）"))
    add("以选择目标统计量直接相除（**ratio of nine-configuration medians**）得到："+"；".join(ratio_text)+"。"
        "该比值的分母/分子均需各自通过相同目标误差且来自同一 timing batch；中位数比不自行构成显著性或全局优势证明。")
    add("## 1. 四步规则与实际搜索范围\n")
    add("1. 给定 epsilon，以既有 Ewald tail 模型指导 `s_R≈s_F≈sqrt(log(C/epsilon))` 初始化；prefactor C 及模型常数以本轮 protocol/initialization 原始记录为准。"
        "初始化只是种子，邻域联合搜索可调整两个 cutoff。\n"
        "2. 每个 xi/kc 起点先得 nominal `k_c=2*xi*s_F`，沿用已有 `snap_kc` 映射到完整 reciprocal shell；用实际 shell kc、`h=L/M`、`eta=kc_actual*h/pi<=eta_max<1` 与 influence denominator 稳定性选最小合理合法 M，同时考虑 FFT-friendly sizes。\n"
        "3. 根据 alias/mesh 误差规律从合法 `1<=p<=M` 选最小合理 p，允许 M,p 相互微调；p 增大并不保证误差单调减小。\n"
        "4. 邻域联合搜索 `(xi,s_R,s_F,M,p)`，每个候选构造自己的 Grid PSD table；以完整算子的 `max_9 E_Gamma<=epsilon` 为经验可行条件，在可重复的 cached full Gamma-v 时间中选择。"
        "Original 最佳参数可作为 baseline/种子，但不是 Grid PSD 最优参数的固定约束。")
    add("不把 `B_off`、`N*delta`、degree bounds 或 full-matrix PSD eigenvalue 检查加入误差选参硬约束；"
        "本轮不重复 B_off 分析，也不开发新 PSD 离散。PSD 来自既有非负 assignment 与逐网格块 PSD 构造的结构；独立结构性代码检查仍保留。"
        "经验预算是否松紧不决定本轮 empirical-pass 候选的去留。")
    add("**最低合理 p 不等于最低合法 p。** 数学合法域仍为 `1<=p<=M`；本轮初始化模型与有限局部预算从 p>=2 开始、"
        "初始化 cap=9，p=1 未测，不能据此说 p=1 数学非法或实测不合格。"
        "现有含 zeta(p) 的 majorant 仅在 p>1 有限，p=2 是首个连续 cardinal B-spline；"
        "alias_C 经验校准来自 p=3..5 的完整旧误差，亦不是 p=2/p=1 的保证。"
        f"本轮候选文件记录的实际 p 集合为 `{', '.join(fmt(v) for v in tested_p) if tested_p else '尚无数据'}`；"
        "所谓最小合理值只在这组初始化预测与有限已测域内成立，不是对全合法域的最小 p 或全局最优证明。")
    if protocol:add("已保存 protocol：\n```json\n"+json.dumps(protocol,ensure_ascii=False,indent=2)+"\n```")
    if theory:
        add("理论源代码与 target physics 的 prefactor 审计见 `theory_source_audit.json`。具体 tail 初始化使用 "
            "`C*s*exp(-s²)` 的大 s 分支（Lambert 解），`sqrt(log(C/epsilon))` 只是其近似；"
            "`H=exp(-z²)-sqrt(pi)*z*erfc(z), z=kappa/(2*xi)` 是辅助函数而非网格间距。"
            "这里使用 kappa=.2、amplitude=1 的 prefactor，不沿用旧 optimizer 的 kappa=.7/nonunit-amplitude 默认值。"
            "alias predictor `q(eta)^p` / original mesh 的局部 h^p 规律只指导初始 p，不是 active Grid PSD 的高阶或完整误差证书。")
        add(table(["xi","target kappa","C_S相对预测系数","C_F相对预测系数"],[
            [fmt(r.get("xi")),fmt(r.get("kappa")),fmt(r.get("CS_relative_predictor")),fmt(r.get("CF_relative_predictor"))]
            for r in theory.get("prefactor_values_at_target_physics",[])]))
    if data["seeds"]:
        add(table(["method","epsilon","stage","xi","sR","sF","M","p","eta"],[
            [r.get("method"),fmt(r.get("epsilon")),r.get("stage"),*[fmt(r.get(k)) for k in ("xi","sR","sF","M","p","eta")]] for r in data["seeds"]]))
    if alias:add("本轮 alias 初始化校准（只用于种子，含旧完整误差的 cutoff 成分，不是 alias-only bound）：\n```json\n"+json.dumps(alias,ensure_ascii=False,indent=2)+"\n```")
    add(table(["候选分组 method/epsilon/stage","记录","独立候选","状态计数"],[
        [key,r["rows"],r["unique_ids"],json.dumps(r["statuses"],ensure_ascii=False)] for key,r in analysis["candidate_groups"].items()]))
    add("完整搜索轨迹、被测但未通过的点、超预算/未测状态以原始 CSV 为准。筛选时钟只用于搜索启发，不能替代正式隔离 timing 的 full matvec 时钟。"
        "局部范围与可承受预算限制了结论；未找到可行候选不能推出该方法在该 epsilon 不存在可行参数。")
    add(f"记录中 `bound_used_as_constraint` 或 `PSD_eigenvalue_used_as_constraint` 为真的候选数："
        f"{analysis['selection_hard_constraint_flag_violations']}。候选按 epsilon/stage 的归属来自 memberships，"
        "同一参数 id 可服务多个精度目标，不把重复归属当成独立新测。实际独立候选数和预算以 `search_budget.json` 为准。")
    add("## 2. 按目标的最佳已测参数与九配置误差\n")
    parameter_rows=[]
    for key,t in analysis["targets"].items():
        for method,entry in t["methods"].items():
            r=entry.get("row",{})
            parameter_rows.append([key,P.SHORT[method],"可行" if entry["available"] else "N/A",fmt(r.get("sR")),
                fmt(r.get("sF")) if method in ("PPPM-grid-PSD","PPPM-original") else "N/A",fmt(r.get("sF_effective")) if method in ("PPPM-grid-PSD","PPPM-original") else "N/A",fmt(r.get("sT")) if method=="B2-positive" else "N/A",
                *[fmt(r.get(k)) for k in ("xi","k_c","r_c","M","p","eta")],str(r.get("batch") or "未注明")])
    add(table(["epsilon","方法","状态","sR","nominal sF","effective sF","B2 sT","xi","actual kc","rc","M","p","eta","timing batch"],parameter_rows))
    add("B2 的 positive-mesh 卷积截断 sT 与 Original/Grid PSD 的 Fourier shell cutoff sF 含义不同；"
        "不把占位 sF 推导为 B2 的 kc/eta。direct dense 不使用网格参数。"
        "表中的 actual kc 取实际完整 reciprocal shell，effective sF=kc_actual/(2*xi)；输入 nominal sF 可能因既有 shell snapping 与它略有差别，eta始终用实际kc。")
    safety=[]
    for key,t in analysis["targets"].items():
        entry=t["methods"]["PPPM-grid-PSD"];r=entry.get("row",{})
        if entry["available"]:safety.append([key,fmt(r.get("M")),fmt(r.get("p")),fmt(r.get("eta")),fmt(r.get("eta_max")),fmt(r.get("W2min")),fmt(r.get("influence_denominator_threshold")),fmt(r.get("n_excluded")),fmt(r.get("n_modes"))])
    add(table(["Grid目标","M","p","actual eta","本轮eta_max","实际min W²","denominator阈值","excluded modes","retained modes"],safety))
    add("本轮 protocol 选 eta_max=.95 作为小于1的操作安全余量；它与既有严格Nyquist/完整shell和influence denominator合法性检查一起使用，"
        "不是新的完整误差上界，也不声称.95是所有精度/硬件下的最优margin。")
    add(image(args,"fig1_best_tested_parameters","图 1：Grid PSD 随目标 epsilon 的最佳已测参数，实际 eta 与两个物理 cutoff 一并显示"))
    validation_table=[]
    for key,t in analysis["targets"].items():
        entry=t["methods"]["PPPM-grid-PSD"]
        for r in entry.get("configuration_errors",[]):validation_table.append([key,r.get("config"),fmt(r.get("E_G")),fmt(r.get("E_original")),fmt(r.get("E_proj")),
            str(first(r,"active","negative_blocks") if first(r,"active","negative_blocks") is not None else "N/A")])
    add(table(["epsilon","配置","Grid E_Gamma","同参数投影前误差","E_proj","投影活跃诊断"],validation_table))
    winning_grid=[t["methods"]["PPPM-grid-PSD"] for t in analysis["targets"].values() if t["methods"]["PPPM-grid-PSD"]["available"]]
    if winning_grid and all(num(r["row"].get("negative_blocks"))==0 for r in winning_grid):
        add("本轮三档已选 Grid table 的 negative_blocks 均为0，所选点的 PSD 投影 inactive，"
            "投影前后完整算子差仅保留实际 CSV 中的浮点数值。它解释了这些局部点与 Original 的精度/速度接近，"
            "不能据此推广为 active 投影始终保留高阶精度，亦不能将速度接近归因于投影带来加速。")
    add(table(["Grid目标","获选stage","获选origin","theory seed/parent（有则列）"],[
        [key,entry["row"].get("stage"),entry["row"].get("origin"),first(entry["row"],"theory_seed_id","parent_id") or "N/A"]
        for key,t in analysis["targets"].items() for entry in [t["methods"]["PPPM-grid-PSD"]] if entry["available"]]))
    add("同参数的投影前后误差用于解释投影影响，不能把其中的 Original 点等同于独立调参后的 Original baseline。"
        "`E_Gamma=||Gamma_h(q)-Gamma_D(q)||_2/||Gamma_D(q)||_2` 使用完整周期算子，保留真实 k=0 与完整 self cancellation；"
        "near/mesh-only 指标不能替代它。PSD/负块/修正量只作解释性诊断，不构成额外 hard filter。"
        "negative grid blocks 是参数表级诊断：固定箱长/参数的九构型共享同一 table，不能把重复列出的活跃性当成九次独立参数表实验；"
        "E_proj 则反映各自位置下的完整算子差。")
    if (args.figures/"fig7_projection_effect_diagnostics.png").exists():add(image(args,"fig7_projection_effect_diagnostics","补图 7：最佳已测 Grid 参数的逐配置投影前后误差与 raw grid block 活跃性，仅作解释性诊断"))
    projection_validated=[r for r in analysis["validated_grid_projection_effect"] if r["nine_configuration_error_complete"]]
    projection_active=[r for r in projection_validated if r["negative_blocks"]>0]
    if projection_validated:
        add(f"为避免只展示 inactive winner，另汇总全部 **{len(projection_validated)} 个完整九配置验证 Grid 候选**，"
            f"其中 {len(projection_active)} 个参数表投影 active；这不是全部单配置 screening 候选集合。"
            "按各候选逐配置完整误差求 max9 before/after/Eproj，保留离散候选索引与参数/id，不把这些点解释为收敛扫描。")
        add(table(["active验证候选","xi","sR","sF","M","p","负网格块数","max9 E_before","max9 E_after","max9 Eproj"],[
            [r.get("id"),*[fmt(r.get(k)) for k in ("xi","sR","sF","M","p","negative_blocks","max9_E_before","max9_E_after","max9_E_proj")]] for r in projection_active]))
        add("这些 active 点展示真实投影可能改变误差，但不能仅从本轮少量 active 候选推断任意参数高阶精度。"
            "既有 PR #4 的 active-projection 精度扫描仍完整保留，本轮不替代或覆盖它。")
    if (args.figures/"fig7b_validated_grid_projection_effect.png").exists():add(image(args,"fig7b_validated_grid_projection_effect","补图 7b：全部九配置验证 Grid 候选的 max9 投影前后误差与 Eproj，标出 active 参数表和三个最终winner"))
    add("## 3. 同精度成本比较与局部 Pareto\n")
    if context:
        scalar_context=[(key,str(value)) for key,value in context.items() if not isinstance(value,(dict,list))]
        add(table(["正式 timing context 字段","保存值"],scalar_context))
        add("完整线程、随机顺序、warmup/repeat 与计时环境记录见 `results/timing_context.json`（或本轮等价 context 文件）；分析 JSON 保留全文。")
        if context.get("threadpools")==[]:
            add("线程设置的证据限于导入前 OMP/OPENBLAS/MKL/VECLIB/NUMEXPR/BLIS 六类环境变量请求1，协议与 raw/context 的 threads=1。"
                "Mac Accelerate 下 `threadpool_info()` 返回空列表，native pool 未能由 threadpoolctl 枚举，实际 native 线程数没有被独立验证。"
                "各方法在同一进程、相同设置下运行；本文的单线程应理解为按单线程配置/请求运行，而非工具已经实测确认所有 native pool 均为1。")
    if runtime:
        add(table(["运行环境来源","保存值"],[
            ["platform",runtime.get("platform")],["Python/NumPy",str(runtime.get("python"))+" / "+str(runtime.get("numpy"))],
            ["线程环境请求",json.dumps(runtime.get("thread_environment",{}),ensure_ascii=False)],
            ["可枚举native pools",json.dumps(runtime.get("enumerable_native_pools",[]))],
            ["native线程数独立验证",runtime.get("native_thread_count_independently_verified")]]))
        add("`runtime_environment.json` 是计时后的同一导入环境 provenance，保留 NumPy 的 Accelerate BLAS/LAPACK 构建信息；未新增性能样本，也不回填未验证native线程数。")
    add("**计时环境限制：仅隔离本实验自身的 search/reference/render worker，不能称机器空闲或硬件完全隔离。** "
        "本机存在 chrome_crashpad 等外部进程持续 CPU 负载；未操作或终止其它应用。正式候选采用同批随机交错、单线程与 paired timing/区间，"
        "这些措施降低顺序偏差，却不能消除外部系统负载、温度或调频的不确定性。`system_load.json` 保存每配置与起始时的 load average 和 top15 进程证据。")
    if isinstance(system_load,list):
        add(table(["计时阶段","1/5/15分钟load average","保存的top CPU进程（原始top15另见JSON）"],[
            [r.get("label"),", ".join(fmt(v) for v in r.get("loadavg",[])),str(r.get("top_processes","")).splitlines()[1] if len(str(r.get("top_processes","")).splitlines())>1 else "N/A"]
            for r in system_load]))
    comparison=[]
    for key,t in analysis["targets"].items():
        for method,entry in t["methods"].items():
            r=entry.get("row",{});st=entry.get("stages",{}).get("matvec",{})
            comparison.append([key,P.SHORT[method],"可行" if entry["available"] else "N/A",fmt(r.get("max_E_G")),fmt(r.get("n_configs")),
                fmt(r.get("median_matvec_s")) if entry["available"] else "N/A",f"[{fmt(st.get('q25_s'))},{fmt(st.get('q75_s'))}]" if entry["available"] else "N/A",
                fmt(r.get("n_validated_feasible")),fmt(r.get("n_timed_feasible")),r.get("batch") or entry.get("reason")])
    add(table(["epsilon","方法","状态","max E_Gamma","验证配置数","cached matvec/秒","q25–q75/秒","已验证可行数","正式已计时可行数","batch/限制"],comparison))
    speed=[]
    for key,t in analysis["targets"].items():
        for label,r in t["speed_ratios"].items():speed.append([key,label,fmt(r.get("ratio_medians")),r.get("batch") or r.get("reason")])
    add(table(["epsilon","同批次中位数成本比","ratio（<1表示Grid更快）","batch/不可比较原因"],speed))
    if data["ratios"]:
        add(table(["epsilon","方法/参照","paired ratio","CI95 low","CI95 high","batch/配对单位"],[
            [fmt(r.get("epsilon")),str(r.get("method"))+" / "+str(first(r,"reference_method","reference")),
             fmt(first(r,"ratio_median_paired_blocks","ratio_median","ratio")),fmt(first(r,"ratio_ci95_low","ci95_low")),fmt(first(r,"ratio_ci95_high","ci95_high")),
             str(first(r,"batch","pairing_unit","pairing") or "见原始CSV")] for r in data["ratios"]]))
        add("原始 paired/bootstrap 比值与置信区间完整保留于本轮 speed-ratio CSV 和 analysis_summary.json；"
            "配对单位、批次和区间解释以文件字段及 timing protocol 为准，不能用独立中位数比冒充 paired interval。"
            "该 bootstrap 区间条件于已选候选及当前计时样本，不包含有限搜索/候选选择的不确定性，也不证明全局速度优势。")
        add("两种比值是不同估计量：正式排名先对各配置的35次样本取中位数，再对九个配置中位数取中位数；"
            "上表的 ratio-of-medians 是两个这样的目标统计量相除。paired ratio 则在每个配置×round 的7样本块内分别取中位数后相除，"
            "最后取45个块比值的中位数。中位数与除法不可交换，因此如约1.04与约.988可以同时正确；不能混用它们的点估计与区间。")
    add(image(args,"fig2_accuracy_cost_pareto","图 2：正式已计时候选的精度–成本关系，仅连接同方法同批次且符合当前目标的 Pareto 点"))
    add(image(args,"fig3_matched_error_and_runtime","图 3：Grid PSD、Original、B2、direct dense 的同目标误差与 cached full matvec 比较，未达标/未测显示 N/A"))
    add("各方法使用自己的可行/已优化参数，不能以不同误差水平的旧时钟拼接速度比。"
        "不同 epsilon 若来自不同 timing batch，只在各自批次内比较；筛选期间竞争负载下的时间不作正式速度证据。"
        "即使同机同线程，接近的时钟也应结合重复样本分布与已保存区间判断，不强行宣称稳定提速。"
        "正式时钟池从新 serial prescreen 的每方法/epsilon top2 及可行 cross-controls 组成；"
        "未纳入正式池的已验证候选不能被称为已作同等重复计时，因此最佳结论限于该正式池。")
    if data["cross_validation"]:
        counts=Counter(r.get("method") for r in data["cross_validation"])
        add(f"cross-method control 另保存 {len(data['cross_validation'])} 条逐配置验证（方法记录数：{dict(counts)}），"
            "将 Grid/Original 各自每 epsilon 的 top2 proxy 候选参数交叉构造并独立检验，再将全部可行 controls 纳入正式计时；计划保存在 `cross_plan.json`。"
            "这样减少仅因搜索路径不同造成的表观优势；交叉点仍只代表已测控制，不能证明全局最优。")
    add("## 4. Setup、matvec 分解与动力学步总成本\n")
    costs=[]
    for key,t in analysis["targets"].items():
        for m,entry in t["methods"].items():
            if not entry["available"]:continue
            for phase,st in entry.get("stages",{}).items():
                if st["median_s"] is not None:costs.append([key,P.SHORT[m],phase,fmt(st["median_s"]),f"[{fmt(st['q25_s'])},{fmt(st['q75_s'])}]",st["n"],st.get("phase_batches")])
    add(table(["epsilon","方法","阶段","median/秒","q25–q75/秒","原始样本数","phase batch"],costs))
    add(image(args,"fig4_setup_and_matvec_components","图 4：table/config setup、cached full matvec 与 near/spread/FFT 分解，所有阶段以秒显示"))
    add("Grid PSD 投影在 table 构造阶段执行；缓存后的 Gamma-v 不重新投影、不重新计算 bound。"
        "table 与 config setup 分开测量：位置改变使 config setup 失效；box、kappa、xi、cutoff、M 或 p 改变使 table 失效。"
        "matvec 的 mesh/FFT/spread 等阶段可能包含彼此，不能把重叠阶段独立中位数直接相加。参数表跨配置共享的样本需按实际重复单位去重。")
    add("正式 full-action 的目标统计量是九个配置各自计时中位数的中位数；保存的 q25–q75 来自原始重复样本，"
        "是样本分布范围而非置信区间。near/mesh/spread/FFT 分解另在独立 component-diagnostics batch 测量，"
        "其 pooled median 只用于解释成本，不回填或改变正式 full-action 排名。direct dense 使用已缓存独立 reference 的 G@v；"
        "完整周期 dense assembly setup 未隔离计时，因此 direct table/config setup 与动态步总成本均 N/A，不能用矩阵绑定/copy 的耗时代替 assembly。")
    add("每动力学步的摊销只作透明的线性成本模型：\n```text\n"
        "derived_cost_per_step = measured_table_setup / table_reuse_steps\n"
        "                      + measured_config_setup\n"
        "                      + matvecs_per_step * measured_cached_full_matvec\n```\n"
        "这里每步更新位置并做一次 config setup，不假设额外邻居表跨步复用。"
        "正式 `results/amortization.csv` 与报告另存的 `results/report_amortization.csv` 保留 table 复用 1/10/100/1000 步与每步 1/10/100/1000 次 matvec 的组合；"
        "这是实测阶段中位数的线性摊销，不能冒充新测得的完整动力学轨迹或随机/Lanczos 总时间。")
    add(image(args,"fig5_dynamics_step_amortized_cost","图 5：多个 matvec 次数下的每步推算总成本，对比 table 每步重建与复用 100 步"))
    add("## 5. 随 N 扩展、检查与结论边界\n")
    if data["scaling"]:
        add("N 扩展是固定密度下的单个 nested uniform_s2 构型（N=64/128 新生成 unsplit direct references，N=256 沿用主基准 reference）。"
            "从主 epsilon=1e-4 的已测最佳参数按箱长重标定 M 以近似保持 h，必要时最多四次增大 M；"
            "没有在每个 N 完整重复五维联合搜索。故它既不是九配置扩展验证，也不代表每 N 的优化最优。"
            "reference 生成的并发 wall time 不拼入正式 setup 比较。")
        if scaling_refs:add(table(["新reference N","rho","kappa","position hash","unsplit cutoff RD","reference tail/norm"],[
            [fmt(r.get("N")),fmt(r.get("rho")),fmt(r.get("kappa")),r.get("position_hash"),fmt(r.get("R_D")),fmt(r.get("cert_bound_rel"))] for r in scaling_refs]))
        add(table(["N","方法","epsilon","max E_Gamma","cached matvec/秒","M","p","batch"],[
            [fmt(r.get("N")),P.SHORT.get(r.get("method"),r.get("method")),fmt(r.get("epsilon")),fmt(r.get("max_E_G")),fmt(r.get("median_matvec_s")),fmt(r.get("M")),fmt(r.get("p")),r.get("batch")]
            for r in data["scaling"]]))
        add(image(args,"fig6_particle_number_scaling","图 6：单个 nested uniform 构型的实测 N 扩展，64/128 为新 reference，256 沿用主 reference"))
    else:add("N scaling 本轮尚无已保存数据，记 N/A；不外推未运行 N 或杜撰计算时间。")
    test_rows=[]
    for path in sorted(args.results.glob("*test*.log")):
        INPUTS.append(path);s=path.read_text(encoding="utf-8",errors="replace");n=re.findall(r"Ran\s+(\d+)\s+tests?",s)
        test_rows.append([path.name,n[-1] if n else "见原日志","通过" if re.search(r"^OK\s*$",s,re.M) and not re.search(r"^FAILED",s,re.M) else "检查原日志"])
    if tests:test_rows.append(["tests_summary.json",first(tests,"tests_run","n_tests","tests"),"通过" if flag(first(tests,"ok","passed")) else "见原记录"])
    add(table(["测试证据","测试数","状态"],test_rows))
    if checks:add("数值/结构检查摘要：\n```json\n"+json.dumps(checks,ensure_ascii=False,indent=2)+"\n```")
    if structural:
        records=structural.get("records",[])
        add(f"独立 `structural_audit.json` 状态 `{structural.get('status','unavailable')}`，"
            f"issues={len(structural.get('issues',[]))}。它在全部正式计时结束后单独运行，不参与搜索接受/淘汰："
            "odd/even 两组各自新建 projection table，以 literal pair contraction 对照 FFT、完整degree与full action，"
            "检查真实 k=0/其投影改变量、完整 singleton self cancellation、平移零模、对称性、PSD能量。"
            "独立小N eigensolve/随机能量只是float64结构回归诊断，不是选参hard filter，也不是任意配置统一精度认证。")
        add(table(["M/parity","N","新建自身投影","checks通过/总数","失败项"],[
            [f"{fmt(r.get('M'))}/{r.get('parity')}",fmt(r.get("N")),r.get("freshly_constructed_own_projection"),
             f"{sum(bool(v) for v in r.get('checks',{}).values())}/{len(r.get('checks',{}))}",
             ",".join(k for k,v in r.get("checks",{}).items() if not v) or "无"] for r in records]))
        metrics=("stencil_min_weight","stencil_partition_defect","physical_ifft_table_max_defect","fft_pair_vs_literal_max_defect",
            "sparse_laplacian_vs_literal_max_defect","full_degree_vs_including_self_literal_max_defect","full_action_vs_literal_max_defect",
            "zero_mode_max_defect","zero_mode_projection_shift_max_defect","retained_zero_mode_norm","singleton_self_response_norm",
            "singleton_full_self_cancellation_residual","translation_residual_max","symmetry_residual","min_projected_table_eigenvalue",
            "min_full_graph_eigenvalue","min_centered_energy_rayleigh")
        add(table(["独立结构量"]+[f"M={fmt(r.get('M'))} {r.get('parity')}" for r in records],[[metric]+[fmt(r.get(metric)) for r in records] for metric in metrics]))
    pending=[]
    if not data["best"]:pending.append("最佳已测参数表尚未保存")
    if not data["formal_raw"]:pending.append("正式重复 timing raw 尚未保存")
    if not data["validation"]:pending.append("逐配置完整算子误差尚未保存")
    if len(plots.get("generated",[]))<5:pending.append("五组主图未全部生成")
    if not structural or structural.get("status")!="pass":pending.append("独立odd/even结构诊断尚未通过/保存")
    if not audit or audit.get("status")!="pass":pending.append("最终独立审计尚未通过/保存")
    add("未完成项："+("；".join(pending) if pending else "主流程所列产物已保存，终审通过")+"。"
        "epsilon=1e-6 和 N 扩展按实际预算报告；N/A 不表示数学不可行。最终审计状态 "
        f"`{audit.get('status','unavailable')}`，issues={len(audit.get('issues',[]))}；旧结果保全以 hash 审计为准。")
    add("本轮区分三种结论：九配置实测 `max E_Gamma<=epsilon` 是 empirical-pass；任意合法参数/位置的 PSD 来自既有结构；"
        "任意配置的统一完整精度认证目前 unavailable，尤其缺少独立 `epsilon_old^cert`，不能从 empirical envelope 升级得到。"
        "最终判断限于已测候选、构型、机器与批次；不把局部搜索称全局最优，不以保守 bound 未过淘汰实测合格候选。")
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text("\n\n".join(content)+"\n",encoding="utf-8")
    INPUTS.extend(p for p in (HERE/"requirements_github.md",HERE/"README.md",HERE/"plot_results.py",Path(__file__),HERE/"structural_audit.py",HERE/"runtime_environment.py") if p.exists())
    manifest=dict(created_at=analysis["created_at"],report=str(args.output),
        inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(INPUTS)) if p.exists() and p.name not in ("audit_final.json","final_audit.json")],
        audit_status_consulted=analysis["audit_status_consulted"])
    args.output.with_suffix(".sources.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(report=str(args.output),pending=pending),ensure_ascii=False))


if __name__=="__main__":main()
