"""New off-diagonal-degree figures from this directory's saved CSVs only.

Old published figures/results and the PPPM implementation are never imported or
modified. True zeros, missing norms, and numerical-audit flags remain explicit.
Every exported PNG/SVG has an aggregate source CSV with original case IDs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

HERE=Path(__file__).resolve().parent
STATUS={"generated":[],"skipped":[],"notes":[]}
PARAMETERS=("N","rho","kappa","xi","sR","sF","M","p")
TEXT={"dataset","source_csv","case_id","data_id","parameter_id","table_key","config","family","id",
      "selected_label","label","case_label","role","phase","method","correction_norm_source",
      "offdiag_method","degreeoff_method","degree_off_method","status","batch","setting","sweep","focus_group"}
COL={"correction":"#df7b24","simple":"#9467bd","total":"#c9474f","off":"#218c64",
     "old":"#2874b2","improved":"#222222"}


def num(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (ValueError,TypeError):return None


def flag(value):
    return value is True or str(value).strip().lower() in ("true","1","1.0","yes")


def first(row,*keys):
    for key in keys:
        if row.get(key) not in (None,""):return row[key]
    return None


def verified(row):
    explicit=first(row,"correction_norm_verified","correction_only_verified")
    if explicit is not None:return flag(explicit)
    source=str(row.get("correction_norm_source","")).lower().replace("-","_")
    if any(v in source for v in ("proxy","historical","empirical_eproj","reused_e_proj")):return False
    return "correction" in source and any(v in source for v in ("dense","direct","eigval","only"))


def canonical(row):
    aliases={"case_id":("data_id",),"parameter_id":("table_key",),
        "E_old":("E_old_measured","E_original"),"E_new":("E_new_measured",),
        "B_total":("B_degree","B_total_degree"),"B_off":("B_offdiag","B_off_degree"),
        "B_old":("old_bound",),"B_improved":("B_new","improved_bound"),
        "self_degree_trace_fraction":("self_trace_fraction",),
        "degree_off_total_ratio":("degreeoff_total_ratio","off_to_total_degree_ratio","max_degree_off_total_ratio"),
        "old_empirical_envelope":("empirical_envelope_old","old_envelope","empirical_envelope"),
        "new_empirical_envelope":("empirical_envelope_new","new_envelope","improved_empirical_envelope")}
    for key,names in aliases.items():
        if row.get(key) in (None,""):
            row[key]=first(row,*names)
    if not row.get("dataset") and row.get("source_csv"):row["dataset"]=Path(str(row["source_csv"])).stem
    if num(row.get("B_simple")) is None and num(row.get("delta")) is not None and num(row.get("N")) is not None:
        row["B_simple"]=row["N"]*row["delta"]
    if num(row.get("B_old")) is None and all(num(row.get(k)) is not None for k in ("B_simple","B_total")):
        row["B_old"]=min(row["B_simple"],row["B_total"])
    # Missing improved bounds can mean numerical unavailability. Never replace
    # an unavailable production result with raw subtraction or a derived min.
    norm=num(row.get("normGammaD"))
    for key in ("B_simple","B_total","B_off","B_old","B_improved"):
        if num(row.get(key+"_rel")) is None and num(row.get(key)) is not None and norm and norm>0:
            row[key+"_rel"]=row[key]/norm
    if norm and norm>0 and num(row.get("correction_norm")) is not None:
        row["true_correction_rel"]=row["correction_norm"]/norm
    else:row["true_correction_rel"]=None
    if verified(row) and (num(row.get("correction_norm")) or 0)>0:
        for key in ("B_old","B_improved"):
            if num(row.get(key)) is not None:row[key+"_tightness"]=row[key]/row["correction_norm"]
    for field,bound in (("old_empirical_envelope","B_old_rel"),("new_empirical_envelope","B_improved_rel")):
        if num(row.get(field)) is None and num(row.get("E_old")) is not None and num(row.get(bound)) is not None:
            row[field]=row["E_old"]+row[bound]
    return row


def read(path):
    if not path.exists():
        STATUS["notes"].append(f"Missing optional input: {path.name}");return []
    with path.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    for row in rows:
        for key,value in list(row.items()):
            if key not in TEXT and num(value) is not None:row[key]=num(value)
        canonical(row)
    return rows


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ["no_data"]
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def save(fig,name,rows,args):
    for ext in ("png","svg"):fig.savefig(args.output/f"{name}.{ext}",dpi=args.dpi,bbox_inches="tight",pad_inches=.06)
    plt.close(fig);write_csv(args.output/"plot_data"/f"{name}.csv",rows)
    STATUS["generated"].append(dict(figure=name,plotted_records=len(rows)))


def skip(name,reason):STATUS["skipped"].append(dict(figure=name,reason=reason))


def unique(rows):
    seen={}
    for row in rows:seen.setdefault(row.get("case_id") or tuple(row.get(k) for k in ("config",)+PARAMETERS),row)
    return list(seen.values())


def label(row):return str(first(row,"selected_label","case_label","label","role","focus_group") or "unlabelled")


def selected(rows,cases):
    groups=defaultdict(list)
    for case in cases:
        matches=[r for r in rows if case.get("case_id") and r.get("case_id")==case["case_id"]]
        if not matches and case.get("parameter_id"):
            matches=[r for r in rows if r.get("parameter_id")==case["parameter_id"]]
        groups[label(case)].extend(matches or [case])
    return {k:unique(v) for k,v in groups.items()}


def floor_for(rows,fields):
    vals=[num(r.get(k)) for r in rows for k in fields];vals=[v for v in vals if v is not None and v>0]
    return min(vals)*.25 if vals else 1e-18


def stats(rows,xkey,ykey,floor):
    groups=defaultdict(list)
    for row in rows:
        if num(row.get(xkey)) is not None and num(row.get(ykey)) is not None:groups[row[xkey]].append(row)
    out=[]
    for x,group in sorted(groups.items()):
        vals=[r[ykey] for r in group];med=float(np.median(vals))
        out.append(dict(x=x,median=med,min=min(vals),max=max(vals),n=len(vals),n_configs=len({r.get("config") for r in group}),
            n_zero=sum(v==0 for v in vals),n_negative=sum(v<0 for v in vals),plot_median=floor if med==0 else med,
            zero_display_floor=floor,case_ids=";".join(sorted({str(r.get("case_id","")) for r in group})),
            parameter_ids=";".join(sorted({str(r.get("parameter_id","")) for r in group}))))
    return out


def curve(ax,points,color,ls="-",marker="o",legend=None,logy=True):
    if not points:return
    x=np.array([p["x"] for p in points]);y=np.array([p["median"] for p in points]);floor=points[0]["zero_display_floor"]
    lo=np.array([p["min"] for p in points]);hi=np.array([p["max"] for p in points])
    if logy:
        # Negative/failed values are not silently converted to a positive floor.
        ax.fill_between(x,np.where(lo==0,floor,np.where(lo>0,lo,np.nan)),np.where(hi==0,floor,np.where(hi>0,hi,np.nan)),color=color,alpha=.09,lw=0)
        ax.plot(x,np.where(y>0,y,np.nan),color=color,ls=ls,marker=marker,label=legend,mfc="white" if ls!="-" else color)
        z=y==0
        if z.any():ax.scatter(x[z],np.full(z.sum(),floor),marker="v",s=28,facecolors="white",edgecolors=color,zorder=6)
    else:
        ax.fill_between(x,lo,hi,color=color,alpha=.09,lw=0);ax.plot(x,y,color=color,ls=ls,marker=marker,label=legend)


def append(out,points,**meta):out.extend(dict(meta,**p) for p in points)


def blank(ax,text="Saved data unavailable"):ax.text(.5,.5,text,transform=ax.transAxes,ha="center",va="center",fontsize=8,color="#777777")


def mesh_axis(ax,rows,max_ticks=5):
    """Use actual integer M values; log minor labels can overlap narrow ranges."""
    values=sorted({r["M"] for r in rows if num(r.get("M")) is not None and r["M"]>0})
    ax.set_xscale("log")
    if not values:return
    preferred=[v for v in (24,32,48,64,80) if v in values]
    picks=sorted({round(v) for v in np.linspace(0,len(values)-1,min(max_ticks,len(values)))})
    ticks=preferred if len(preferred)>=3 else [values[i] for i in picks]
    ax.xaxis.set_major_locator(FixedLocator(ticks));ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x",labelbottom=True)
    ax.set_xlim(values[0]/1.05,values[-1]*1.05)


def activity(ax,rows,out,**meta):
    groups=defaultdict(list)
    for row in rows:
        if num(row.get("M")) is not None:groups[row["M"]].append(row)
    for M,group in sorted(groups.items()):
        n=sum(flag(r.get("active")) or (num(r.get("negative_blocks")) or 0)>0 for r in group)
        state="inactive" if n==0 else "active" if n==len(group) else "mixed"
        color={"inactive":"#a9b9c6","active":"#c9474f","mixed":"#e3a527"}[state]
        ax.scatter([M],[.98],transform=ax.get_xaxis_transform(),marker="s",s=22,c=color,zorder=7)
        bad=sum(flag(first(r,"cancellation_unreliable_raw","cancellation_unreliable","subtraction_unreliable","cancellation_flag")) for r in group)
        if bad:ax.text(M,.92,"!",transform=ax.get_xaxis_transform(),ha="center",va="center",color="#5a398d",fontweight="bold",fontsize=10,zorder=7)
        out.append(dict(record_type="activity",M=M,n=len(group),active_count=n,regime=state,raw_subtraction_flagged=bad,**meta))


METRICS=[("true_correction_rel","correction","-","o",r"true $\|G_C\|_2/\|\Gamma_D\|_2$"),
    ("B_simple_rel","simple",":","D",r"$N\delta/\|\Gamma_D\|_2$"),
    ("B_total_rel","total","-.","x",r"$B_{\rm total}/\|\Gamma_D\|_2$"),
    ("B_off_rel","off","-","s",r"$B_{\rm off}/\|\Gamma_D\|_2$"),
    ("B_old_rel","old","--","^",r"$B_{\rm old}/\|\Gamma_D\|_2$"),
    ("B_improved_rel","improved","--","+",r"$B_{\rm improved}/\|\Gamma_D\|_2$")]


def convergence(rows,args):
    name="fig1_correction_and_degree_bounds"
    conv=unique([r for r in rows if r.get("dataset")=="convergence"])
    ps=[p for p in (2,3,5,7,9) if any(r.get("p")==p for r in conv)]
    if not ps:skip(name,"No convergence rows");return
    fig,axes=plt.subplots(2,3,figsize=(12.5,7.2),sharex=True,sharey=True);out=[]
    floor=floor_for(conv,[key for key,*_ in METRICS])
    for ax,p in zip(axes.ravel(),ps):
        group=[r for r in conv if r.get("p")==p]
        for key,color,ls,marker,legend in METRICS:
            measured=[r for r in group if verified(r) or flag(r.get("exact_zero"))] if key=="true_correction_rel" else group
            points=stats(measured,"M",key,floor);curve(ax,points,COL[color],ls,marker,legend)
            append(out,points,p=p,metric=key,scope="correction graph only")
        activity(ax,group,out,p=p)
        mesh_axis(ax,conv);ax.set_yscale("log");ax.set_xlabel("Mesh size M");ax.set_ylabel("Relative correction / sufficient bound");ax.set_title(f"p={p}")
    axes[1,2].axis("off")
    handles=[Line2D([],[],color=COL[c],ls=ls,marker=mk,label=lab) for _,c,ls,mk,lab in METRICS]
    handles += [Line2D([],[],ls="",marker="v",mfc="white",mec="black",label="exact zero (recorded display floor)"),
                Line2D([],[],ls="",marker="$!$",color="#5a398d",label="raw cancellation strip (! at axes y=0.92)")]
    axes[1,2].legend(handles=handles,loc="center",fontsize=8)
    fig.text(.5,.025,"Medians and min–max across 9 configurations. Top strip: active/red, mixed/amber, inactive/gray squares; purple ! marks raw cancellation.\n"
        "True correction norms only; unavailable wide-table norms stay blank. No convergence order is inferred across projection transitions.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.08,1,1));save(fig,name,out,args)


def tightness(rows,args):
    name="fig2_tightness_self_and_offdegree"
    conv=unique([r for r in rows if r.get("dataset")=="convergence"])
    ps=[p for p in (2,3,5,7,9) if any(r.get("p")==p for r in conv)]
    if not ps:skip(name,"No convergence rows");return
    fig,axes=plt.subplots(3,len(ps),figsize=(3*len(ps),8.4),squeeze=False,sharex="col",sharey="row");out=[]
    for ci,p in enumerate(ps):
        group=[r for r in conv if r.get("p")==p]
        true=[r for r in group if verified(r) and (num(r.get("correction_norm")) or 0)>0]
        for key,color,ls in (("B_old_tightness","old","--"),("B_improved_tightness","improved","-")):
            points=stats(true,"M",key,1e-18);curve(axes[0,ci],points,COL[color],ls,"o")
            append(out,points,p=p,metric=key,norm_scope="true correction-only; zero ratio undefined")
        axes[0,ci].axhline(1,color="#777777",ls=":",lw=.8);axes[0,ci].set_yscale("log");axes[0,ci].set_title(f"p={p}")
        for ri,key,col,logy in ((1,"self_degree_trace_fraction","total",False),(2,"degree_off_total_ratio","off",True)):
            floor=floor_for(group,[key]);points=stats(group,"M",key,floor);curve(axes[ri,ci],points,COL[col],logy=logy)
            append(out,points,p=p,metric=key)
            if logy:axes[ri,ci].set_yscale("log")
            if not points:blank(axes[ri,ci])
        mesh_axis(axes[2,ci],conv);axes[2,ci].set_xlabel("Mesh size M")
    axes[0,0].set_ylabel("Bound / true correction norm")
    axes[1,0].set_ylabel("Self / total degree trace fraction")
    axes[2,0].set_ylabel(r"$\max\lambda(D_{off})/\max\lambda(D_{total})$")
    fig.legend(handles=[Line2D([],[],color=COL["old"],ls="--",label="old bound tightness"),
        Line2D([],[],color=COL["improved"],label="self-excluded bound tightness")],loc="lower center",ncol=2)
    fig.text(.5,.065,"No clipping of ratios to 1. Below-full-operator-floor correction norms can produce extreme ratios; all positive verified norms are retained.\n"
        "Zero-correction tightness and zero/zero degree ratios are undefined, not fabricated values. Bands: configuration min–max.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.14,1,1));save(fig,name,out,args)


def budgets(rows,cases,args):
    name="fig3_winner_badcase_empirical_budgets"
    groups={k:v for k,v in selected(rows,cases).items() if "outlier" not in k.lower()}
    if not groups:skip(name,"No selected winner/bad-case rows");return
    fig,axes=plt.subplots(1,len(groups),figsize=(5.8*len(groups),5.1),squeeze=False);out=[]
    order={"uniform":0,"moderate":1,"strong":2}
    metrics=[("E_old","#2874b2","--","o","Old empirical error"),("E_new","#218c64","-","s","New empirical error"),
        ("E_proj","#df7b24",":","x","E_proj comparison"),
        ("old_empirical_envelope","#9467bd","-.","^","Old empirical envelope"),
        ("new_empirical_envelope","#222222","-","+","Self-excluded empirical envelope")]
    for ax,(case,group) in zip(axes[0],groups.items()):
        group=sorted(group,key=lambda r:(order.get(str(r.get("family")),99),num(r.get("seed")) or 0))
        prepared=[dict(r,index=i) for i,r in enumerate(group)];floor=floor_for(group,[key for key,*_ in metrics])
        for key,color,ls,mk,legend in metrics:
            points=stats(prepared,"index",key,floor);curve(ax,points,color,ls,mk,legend);append(out,points,selected_label=case,metric=key,scope="empirical; not certified total")
        old=sum(num(r.get("old_empirical_envelope")) is not None and r["old_empirical_envelope"]<=1e-4 for r in group)
        new=sum(num(r.get("new_empirical_envelope")) is not None and r["new_empirical_envelope"]<=1e-4 for r in group)
        actual=sum(num(r.get("E_new")) is not None and r["E_new"]<=1e-4 for r in group)
        ax.axhline(1e-4,color="#777777",ls=":",lw=.9,label="target 1e-4")
        ax.set_title(f"{case}\nEnvelope pass old/new: {old}/{len(group)} → {new}/{len(group)}; measured new pass: {actual}/{len(group)}")
        ax.set_xticks(range(len(group)));ax.set_xticklabels([f"{str(r.get('family','?'))[0].upper()}{num(r.get('seed')):g}" if num(r.get("seed")) is not None else "?" for r in group])
        ax.set_yscale("log");ax.set_xlabel("Configuration (family initial + seed)");ax.set_ylabel("Relative empirical error / envelope")
    handles,labels=axes[0,-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.105),ncol=3,fontsize=7)
    fig.text(.5,.02,"Envelopes use measured E_old against the numerical direct reference; no certified old-error bound is available.\n"
        "These are sufficient/conservative budgets: envelope failure must not automatically reject an independently measured empirical-pass candidate.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.22,1,1));save(fig,name,out,args)


def independent_samples(samples,phase):
    seen={}
    for i,row in enumerate(samples):
        shared=flag(row.get("parameter_sample_shared_across_configs")) or "parameter" in phase or phase=="correction_fft"
        key=(row.get("parameter_id"),row.get("repeat"),phase) if shared else ("row",i)
        seen.setdefault(key,row)
    return list(seen.values())


def costs(rows,cases,raw,args):
    name="fig4_incremental_bound_setup_cost"
    groups=selected(rows,cases)
    if not groups or not raw:skip(name,"No selected-case new bound timing");return
    phases=sorted({str(r.get("phase")) for r in raw if num(r.get("seconds")) is not None})
    fig,ax=plt.subplots(figsize=(max(10,len(groups)*3),5.2));out=[];zero_positions=[]
    positive=[r["seconds"] for r in raw if num(r.get("seconds")) is not None and r["seconds"]>0]
    display_floor=min(positive)*.25 if positive else 1e-9
    for pi,phase in enumerate(phases):
        for gi,(case,group) in enumerate(groups.items()):
            ids={r.get("case_id") for r in group};pars={r.get("parameter_id") for r in group}
            samples=[r for r in raw if r.get("phase")==phase and num(r.get("seconds")) is not None and
                     (label(r)==case or (r.get("case_id") and r["case_id"] in ids) or (r.get("parameter_id") and r["parameter_id"] in pars))]
            samples=independent_samples(samples,phase)
            if not samples:continue
            vals=[r["seconds"] for r in samples];median=float(np.median(vals));q25,q75=np.quantile(vals,[.25,.75])
            x=gi+(pi-(len(phases)-1)/2)*.8/max(1,len(phases));color=plt.get_cmap("tab10")(pi%10)
            if median>0:
                ax.bar(x,median,width=.8/max(1,len(phases)),color=color)
                ax.errorbar(x,median,yerr=[[median-max(float(q25),display_floor)],[q75-median]],fmt="none",ecolor="#333333",lw=.7,capsize=2)
                if median>=1 and "fallback" in phase:ax.text(x,median*1.07,f"{median:.2g} s",ha="center",va="bottom",fontsize=6)
            else:zero_positions.append((x,color))
            out.append(dict(selected_label=case,phase=phase,median_s=median,q25_s=float(q25),q75_s=float(q75),
                            median_ms=median*1e3,q25_ms=float(q25)*1e3,q75_ms=float(q75)*1e3,n=len(vals),
                            true_zero_median=median==0,zero_display_floor_s=display_floor,
                            sample_scope="new isolated bound-only setup; shared parameter repeats deduplicated"))
    for x,color in zero_positions:ax.scatter([x],[display_floor],marker="v",s=28,facecolors="white",edgecolors=color,zorder=6)
    ax.set_xticks(range(len(groups)));ax.set_xticklabels(list(groups),fontsize=8)
    ax.set_yscale("log");ax.set_ylabel("New bound-only setup (seconds; log scale)")
    ax.set_title("Isolated setup batch; slow fallback costs remain visible in seconds")
    ax.legend(handles=[plt.Rectangle((0,0),1,1,color=plt.get_cmap("tab10")(i%10),label=phase) for i,phase in enumerate(phases)],ncol=min(4,len(phases)),fontsize=6.5)
    fig.text(.5,.025,"Configuration samples aggregate saved configurations and repeats; shared parameter samples count once per repeat.\n"
        "Phases may overlap; medians must not be added. Zero stage medians: open triangles at recorded floor.\n"
        "Slow fallback is additional work, not free. No old cached-matvec timings or speed ratios are used.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.13,1,1));save(fig,name,out,args)


def style():
    plt.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix",
        "font.size":8.5,"axes.titlesize":9,"axes.titleweight":"bold","axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
        "legend.fontsize":6.5,"axes.linewidth":.7,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,
        "axes.grid":True,"grid.color":"#dddddd","grid.linewidth":.5,"legend.frameon":True,"legend.fancybox":False,
        "lines.linewidth":1.2,"lines.markersize":3.5,"svg.fonttype":"none"})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--results",type=Path,default=HERE/"results")
    parser.add_argument("--output",type=Path,default=HERE/"figures");parser.add_argument("--dpi",type=int,default=240)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);style()
    bounds=read(args.results/"bounds.csv")
    if not bounds:bounds=read(args.results/"offdiag_bounds.csv")
    cases=read(args.results/"cases_summary.csv");raw=read(args.results/"timing_raw.csv")
    jobs=[("convergence",lambda:convergence(bounds,args)),("tightness",lambda:tightness(bounds,args)),
          ("budgets",lambda:budgets(bounds,cases,args)),("costs",lambda:costs(bounds,cases,raw,args))]
    for name,job in jobs:
        try:job()
        except (ValueError,KeyError,TypeError,IndexError) as exc:plt.close("all");skip(name,f"{type(exc).__name__}: {exc}")
    STATUS.update(input_rows=len(bounds),dataset_rows=dict(Counter(str(r.get("dataset")) for r in bounds)))
    (args.output/"plot_status.json").write_text(json.dumps(STATUS,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(STATUS,ensure_ascii=False))


if __name__=="__main__":main()
