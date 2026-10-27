"""Saved-data figures for the final four-step Experiment 2 selection.

Only this experiment directory is written. No physics, PSD projection, old
results, or cached matvec code is imported. Screening clocks never substitute
for the common serial full-operator timing batch. Each figure has source CSV.
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
from matplotlib.ticker import NullLocator

HERE=Path(__file__).resolve().parent
EPSILONS=(1e-3,1e-4,1e-5,1e-6)
METHODS=("PPPM-grid-PSD","PPPM-original","B2-positive","direct-dense")
COLORS={"PPPM-grid-PSD":"#b23a48","PPPM-original":"#2874b2","B2-positive":"#218c64","direct-dense":"#777777"}
SHORT={"PPPM-grid-PSD":"Grid PSD","PPPM-original":"Original","B2-positive":"B2","direct-dense":"Direct dense"}
TEXT={"method","id","candidate_id","config","family","phase","batch","timing_batch","stage","status","reason","selection_status","source","source_csv"}
STATUS={"generated":[],"skipped":[],"notes":[]}


def num(value):
    try:
        v=float(value);return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None


def flag(value):return value is True or str(value).strip().lower() in ("true","1","1.0","yes","pass","passed")


def first(row,*keys):
    for key in keys:
        if row.get(key) not in (None,""):return row[key]
    return None


def method_name(value):
    v=str(value or "");key=v.lower().replace("_","-")
    if "grid" in key and "psd" in key:return "PPPM-grid-PSD"
    if "original" in key or key in ("pppm","original-pppm"):return "PPPM-original"
    if "b2" in key or "positive-mesh" in key:return "B2-positive"
    if "dense" in key or key in ("direct","reference"):return "direct-dense"
    return v


def canonical(row):
    aliases={"id":("candidate_id","candidate","parameter_id"),"epsilon":("eps","target_error","target_epsilon"),
        "method":("scheme",),"batch":("timing_batch","batch_id"),"max_E_G":("max_error","max_EGamma","max_E_gamma","max_E_Gamma"),
        "E_G":("E_Gamma","E_gamma","E_new"),"E_original":("E_old","E_before_projection"),
        "median_matvec_s":("median_total_s","median_cached_matvec_s","matvec_median_s","median_runtime_s"),
        "median_table_setup_s":("table_setup_median_s","median_parameter_setup_s"),
        "median_config_setup_s":("config_setup_median_s","median_configuration_setup_s"),
        "xi":("ξ",),"sR":("s_R",),"sF":("s_F",),"eta":("nyquist_eta","eta_actual","eta_prediction"),
        "k_c":("kc","kcut","kc_prediction"),"r_c":("rc","rcut","rc_prediction"),"N":("n_particles",),"n_configs":("validation_configs","n_validation_configs"),
        "q25_matvec_s":("matvec_q25_s","q25_total_s"),"q75_matvec_s":("matvec_q75_s","q75_total_s")}
    for key,other in aliases.items():
        if row.get(key) in (None,""):row[key]=first(row,*other)
    row["method"]=method_name(row.get("method"))
    xi=num(row.get("xi"));sr=num(row.get("sR"));sf=num(row.get("sF"))
    if xi and sr is not None:
        row["r_c_computed"]=sr/xi
        if num(row.get("r_c")) is None:row["r_c"]=sr/xi
    spectral=row["method"] in ("PPPM-grid-PSD","PPPM-original")
    if spectral and xi and sf is not None:
        row["k_c_nominal"]=2*xi*sf
        if num(row.get("k_c")) is None:row["k_c"]=2*xi*sf
    if spectral and xi and num(row.get("k_c")) is not None and num(row.get("sF_effective")) is None:row["sF_effective"]=row["k_c"]/(2*xi)
    L=num(row.get("L"));M=num(row.get("M"));kc=num(row.get("k_c"))
    if spectral and L and M and kc is not None:
        row["eta_computed"]=kc*L/(M*math.pi)
        if num(row.get("eta")) is None:row["eta"]=row["eta_computed"]
    return row


def read(path):
    if not path.exists():return []
    with path.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
    for row in rows:
        for key,value in list(row.items()):
            if key not in TEXT and num(value) is not None:row[key]=num(value)
        canonical(row)
    return rows


def load_first(root,names):
    for name in names:
        path=root/name
        if path.exists():return read(path),path
    return [],None


def inputs(root):
    groups={"best":("best_by_epsilon.csv","frontier.csv","best_parameters.csv","accuracy_time_frontier.csv"),
        "timing":("final_timing.csv","timing_summary.csv","timing_candidates.csv"),
        "raw":("timing_raw.csv",),"components":("components_raw.csv",),"tables":("table_setup_raw.csv",),"setups":("config_setup_raw.csv",),
        "candidates":("search.csv","candidates.csv","search_raw.csv"),
        "memberships":("candidate_memberships.csv",),"shortlist":("shortlist.csv",),"availability":("availability.csv",),
        "validation":("validation_raw.csv","configuration_errors.csv","validation.csv"),"cross_validation":("cross_validation.csv",),
        "scaling":("scaling.csv","scaling_summary.csv","N_scaling.csv"),"scaling_raw":("scaling_raw.csv",),
        "ratios":("paired_speed_ratios.csv","speed_ratios.csv"),"seeds":("theory_initialization.csv","initialization.csv","seeds.csv","initial_candidates.csv")}
    result={};paths=[]
    for key,names in groups.items():
        result[key],path=load_first(root,names)
        if path:paths.append(path)
    for row in result["seeds"]:
        if not row.get("method"):row["method"]="PPPM-grid-PSD"
    result["formal_raw"]=list(result["raw"])
    for key,phase in (("tables","table_setup"),("setups","config_setup")):
        for row in result[key]:
            if not row.get("phase"):row["phase"]=phase
    result["raw"]=result["raw"]+result["tables"]+result["setups"]+result["components"]
    result["validation_primary"]=list(result["validation"])
    # pool_rows in the producer keeps a later cross revalidation for duplicate
    # parameter IDs; mirror that provenance when displaying its config errors.
    result["validation"]=unique(result["cross_validation"]+result["validation"],("method","id","config"))
    for row in result["scaling"]:
        matching=[r for r in result["scaling_raw"] if r.get("method")==row.get("method") and r.get("N")==row.get("N")]
        batches={r.get("batch") for r in matching if r.get("batch")}
        if not row.get("batch") and len(batches)==1:row["batch"]=next(iter(batches));row["batch_source"]="matching N/method scaling_raw.csv"
        row["n_timing_samples"]=len(matching)
    result["paths"]=paths;return result


def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ["no_data"]
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def unique(rows,keys=("method","id","batch")):
    found={}
    for row in rows:found.setdefault(tuple(row.get(k) for k in keys),row)
    return list(found.values())


def eps_match(a,b):return num(a) is not None and math.isclose(float(a),float(b),rel_tol=1e-7,abs_tol=0)


def feasible(row,epsilon):
    if row.get("available") not in (None,"") and not flag(row["available"]):return False
    error=num(row.get("max_E_G"));runtime=num(row.get("median_matvec_s"))
    return error is not None and error<=epsilon and runtime is not None and runtime>0


def best_row(rows,method,epsilon):
    candidates=[r for r in rows if r.get("method")==method and eps_match(r.get("epsilon"),epsilon)]
    passing=[r for r in candidates if feasible(r,epsilon)]
    return min(passing,key=lambda r:r["median_matvec_s"]) if passing else (candidates[0] if candidates else None)


def display_eps(rows):
    requested=[e for e in EPSILONS[:3]]
    if any(eps_match(r.get("epsilon"),1e-6) and feasible(r,1e-6) for r in rows):requested.append(1e-6)
    return requested


def quantiles(values):
    values=[v for v in values if num(v) is not None]
    return (None,None,None) if not values else tuple(float(v) for v in np.quantile(values,[.25,.5,.75]))


PHASES={"matvec":("total","matvec","cached_matvec","full_matvec"),"table_setup":("table_setup","parameter_setup"),
    "config_setup":("config_setup","configuration_setup"),"near":("near","near_field"),
    "spread_gather":("spread_gather","spread/gather"),"fft_influence":("fft_influence","fft","fourier"),"mesh":("mesh",)}


def phase_records(raw,row,phase):
    phases=PHASES.get(phase,(phase,));samples=[];seen=set()
    for r in raw:
        if r.get("method")!=row.get("method") or r.get("id")!=row.get("id") or r.get("phase") not in phases:continue
        if row.get("batch") and r.get("batch") and r["batch"]!=row["batch"] and phase not in ("near","spread_gather","fft_influence","mesh"):continue
        value=num(r.get("seconds"))
        if value is None:continue
        if flag(r.get("parameter_sample_shared_across_configs")):
            key=(r.get("method"),r.get("id"),r.get("repeat"),r.get("batch"),r.get("phase"))
            if key in seen:continue
            seen.add(key)
        samples.append(r)
    return samples


def phase_samples(raw,row,phase):return [r["seconds"] for r in phase_records(raw,row,phase)]


def phase_stats(row,raw,phase):
    q25,median,q75=quantiles(phase_samples(raw,row,phase))
    field="matvec" if phase=="matvec" else phase
    saved=num(row.get("median_"+field+"_s"))
    if saved is not None:
        median=saved;q25=num(row.get("q25_"+field+"_s")) if num(row.get("q25_"+field+"_s")) is not None else q25
        q75=num(row.get("q75_"+field+"_s")) if num(row.get("q75_"+field+"_s")) is not None else q75
    records=phase_records(raw,row,phase)
    return dict(median_s=median,q25_s=q25,q75_s=q75,n=len(records),
        phase_batches=";".join(sorted({str(r.get("batch")) for r in records})),
        median_source="saved summary; formal objective is median of configuration medians" if saved is not None else "pooled diagnostic stage samples",
        interval_source="q25/q75 of saved raw samples or explicit summary fields; not a confidence interval")


def save(fig,name,rows,args):
    for ext in ("png","svg"):fig.savefig(args.output/f"{name}.{ext}",dpi=args.dpi,bbox_inches="tight",pad_inches=.08)
    plt.close(fig);write(args.output/"plot_data"/f"{name}.csv",rows)
    STATUS["generated"].append(dict(figure=name,records=len(rows)))


def skip(name,reason):STATUS["skipped"].append(dict(figure=name,reason=reason))


def floor(values,default=1e-18):
    values=[v for v in values if num(v) is not None and v>0];return min(values)*.25 if values else default


def parameters(data,args):
    name="fig1_best_tested_parameters";rows=data["best"]
    eps=display_eps(rows);chosen=[best_row(rows,"PPPM-grid-PSD",e) for e in eps]
    good=[r for r,e in zip(chosen,eps) if r and feasible(r,e)]
    if not good:skip(name,"No measured feasible Grid PSD best-tested rows");return
    metrics=[("sR",r"$s_R$"),("sF",r"Nominal $s_F$"),("xi",r"$\xi$"),("k_c",r"Actual shell $k_c$"),
        ("r_c",r"$r_c=s_R/\xi$"),("M","Mesh M"),("p","Spline p"),("eta",r"$\eta=k_c h/\pi$")]
    fig,axes=plt.subplots(2,4,figsize=(12,6.3));out=[]
    for ax,(key,title) in zip(axes.ravel(),metrics):
        valid=[r for r in good if num(r.get(key)) is not None]
        ax.plot([r["epsilon"] for r in valid],[r[key] for r in valid],"o-",color=COLORS["PPPM-grid-PSD"])
        ax.margins(y=.17)
        for i,r in enumerate(valid):
            align="left" if i==0 else "right" if i==len(valid)-1 else "center"
            dx=3 if align=="left" else -3 if align=="right" else 0
            ax.annotate(f"{r[key]:.4g}",(r["epsilon"],r[key]),xytext=(dx,5),textcoords="offset points",ha=align,fontsize=7)
        ax.set_xscale("log");ax.invert_xaxis();ax.set_xticks(eps,labels=[f"{e:.0e}" for e in eps]);ax.set_title(title);ax.set_xlabel("Target epsilon")
        for r in valid:out.append(dict(r,metric=key,value=r[key],selection_scope="local best tested; not global optimum"))
    missing=[f"{e:.0e}" for r,e in zip(chosen,eps) if not r or not feasible(r,e)]
    fig.text(.5,.02,"Grid PSD local best tested after independent-reference validation; each target is a separate feasibility condition.\n"+
        ("Unavailable targets: "+", ".join(missing) if missing else "Initialization is a search seed; parameters are jointly adjusted, not fixed to Original PPPM."),ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.09,1,1));save(fig,name,out,args)


def pareto_points(rows):
    out=[];best=math.inf
    for r in sorted(rows,key=lambda r:(r["median_matvec_s"],r["max_E_G"])):
        if r["max_E_G"]<best:out.append(r);best=r["max_E_G"]
    return out


def accuracy_cost(data,args):
    name="fig2_accuracy_cost_pareto";rows=unique(data["timing"])
    rows=[r for r in rows if num(r.get("median_matvec_s")) is not None and r["median_matvec_s"]>0 and num(r.get("max_E_G")) is not None]
    if not rows:skip(name,"No common-batch timed candidates with max full-operator errors");return
    eps=display_eps(data["best"]);fig,axes=plt.subplots(1,len(eps),figsize=(4.3*len(eps),4.3),squeeze=False);out=[]
    errorfloor=floor([r["max_E_G"] for r in rows])
    for ax,e in zip(axes[0],eps):
        for method in METHODS:
            group=[r for r in rows if r.get("method")==method];color=COLORS[method]
            for r in group:
                passed=feasible(r,e);y=r["max_E_G"] or errorfloor
                ax.scatter([r["median_matvec_s"]],[y],s=20,marker="v" if r["max_E_G"]==0 else "o",edgecolors=color,facecolors=color if passed and r["max_E_G"]>0 else "white")
                out.append(dict(r,epsilon=e,record_type="timed candidate",feasible_at_target=passed,zero_display_floor=errorfloor))
            batches=defaultdict(list)
            for r in group:
                if feasible(r,e):batches[str(r.get("batch") or "unspecified")].append(r)
            for batch,rr in batches.items():
                frontier=pareto_points(rr);ax.plot([r["median_matvec_s"] for r in frontier],[r["max_E_G"] or errorfloor for r in frontier],color=color,lw=.9)
                out.extend(dict(r,epsilon=e,record_type="feasible Pareto within timing batch",zero_display_floor=errorfloor) for r in frontier)
            selected=best_row(data["best"],method,e)
            if selected and feasible(selected,e):
                if selected["max_E_G"]==0:ax.scatter([selected["median_matvec_s"]],[errorfloor],marker="v",s=40,facecolors="white",edgecolors=color,zorder=6)
                else:ax.scatter([selected["median_matvec_s"]],[selected["max_E_G"]],marker="*",s=95,color=color,zorder=6)
                out.append(dict(selected,epsilon=e,record_type="reported best tested marker",zero_display_floor=errorfloor))
        ax.axhline(e,color="#777777",ls=":");ax.set_xscale("log");ax.set_yscale("log");ax.set_title(f"Target {e:.0e}");ax.set_xlabel("Cached full Gamma-v time / s");ax.set_ylabel("max over configurations E_Gamma")
    fig.legend(handles=[Line2D([],[],color=COLORS[m],marker="o",label=SHORT[m]) for m in METHODS],loc="lower center",ncol=4)
    fig.text(.5,.075,"Common serial full matvec clocks only; our accuracy/render workers are stopped, while external system load remains uncontrolled.\nLines join feasible same-method/same-batch points; open points exceed target (or mark exact zero), and stars show best tested. No global optimum claim.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.17,1,1));save(fig,name,out,args)


def comparison(data,args):
    name="fig3_matched_error_and_runtime";rows=data["best"]
    if not rows:skip(name,"No target-wise measured comparison summary");return
    eps=display_eps(rows);fig,axes=plt.subplots(2,len(eps),figsize=(4.3*len(eps),6.8),squeeze=False);out=[]
    for ci,e in enumerate(eps):
        selected=[best_row(rows,m,e) for m in METHODS]
        errorfloor=floor([num(r.get("max_E_G")) for r in selected if r]);timefloor=floor([num(r.get("median_matvec_s")) for r in selected if r],1e-8)
        for i,(m,r) in enumerate(zip(METHODS,selected)):
            available=bool(r and feasible(r,e));color=COLORS[m]
            if r and num(r.get("max_E_G")) is not None:
                error=r["max_E_G"]
                axes[0,ci].scatter([i],[error or errorfloor],s=36,marker="v" if error==0 else "o",edgecolors=color,facecolors=color if available and error>0 else "white")
            if available:
                st=phase_stats(r,data["raw"],"matvec");median=st["median_s"]
                axes[1,ci].bar(i,median,color=color,width=.6)
                if st["q25_s"] is not None and st["q75_s"] is not None:axes[1,ci].errorbar(i,median,yerr=[[max(0,median-st["q25_s"])],[max(0,st["q75_s"]-median)]],fmt="none",ecolor="#333333",capsize=2)
                axes[1,ci].annotate(f"{median*1e3:.3g} ms",(i,median),xytext=(0,5),textcoords="offset points",ha="center",fontsize=6)
            else:axes[1,ci].text(i,.06,"N/A",transform=axes[1,ci].get_xaxis_transform(),ha="center",fontsize=8,color=color)
            out.append(dict(r or {},epsilon=e,method=m,matched_feasible=available,not_available_reason="" if available else "not measured or no validated feasible row",error_display_floor=errorfloor,time_display_floor=timefloor))
        axes[0,ci].axhline(e,color="#777777",ls=":");axes[0,ci].set_title(f"Target {e:.0e}")
        for ax in axes[:,ci]:ax.set_xticks(range(4),labels=[SHORT[m] for m in METHODS],rotation=15);ax.set_yscale("log")
        axes[0,ci].set_ylabel("max configuration E_Gamma");axes[1,ci].set_ylabel("Feasible cached full matvec / s")
    fig.text(.5,.02,"Each method uses its own feasible best-tested parameters. Infeasible/unmeasured runtime comparisons remain N/A.\nDirect-reference zero errors are open triangles at the recorded display floor. Error bars: q25–q75 when saved; timing batches remain in source CSV.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.09,1,1));save(fig,name,out,args)


def setup_components(data,args):
    name="fig4_setup_and_matvec_components";best=data["best"]
    if not best:skip(name,"No measured best-tested rows");return
    eps=display_eps(best);fig,axes=plt.subplots(2,len(eps),figsize=(4.4*len(eps),7),squeeze=False);out=[]
    phasesets=(("table_setup","config_setup","matvec"),("near","spread_gather","fft_influence","mesh"))
    any_data=False
    for ci,e in enumerate(eps):
        for ri,phases in enumerate(phasesets):
            ax=axes[ri,ci];zero=[]
            for mi,m in enumerate(METHODS):
                r=best_row(best,m,e)
                if not r or not feasible(r,e):ax.text(mi,.04,"N/A",transform=ax.get_xaxis_transform(),ha="center",fontsize=7);continue
                nstage=0
                for pi,phase in enumerate(phases):
                    st=phase_stats(r,data["raw"],phase);value=st["median_s"]
                    if value is None:continue
                    nstage+=1
                    any_data=True;x=mi+(pi-(len(phases)-1)/2)*.8/len(phases)
                    if value>0:
                        ax.bar(x,value,width=.8/len(phases),color=COLORS[m],alpha=1-.18*pi,hatch=("","//","..","xx")[pi])
                        if st["q25_s"] is not None and st["q75_s"] is not None:ax.errorbar(x,value,yerr=[[max(0,value-st["q25_s"])],[max(0,st["q75_s"]-value)]],fmt="none",ecolor="#333333",lw=.7,capsize=1.5)
                    else:zero.append(x)
                    out.append(dict(r,epsilon=e,phase=phase,**st,record_type="measured setup or component; phases may overlap"))
                if not nstage:ax.text(mi,.04,"N/A",transform=ax.get_xaxis_transform(),ha="center",fontsize=7)
            values=[v["median_s"] for v in out if eps_match(v["epsilon"],e) and v["phase"] in phases]
            f=floor(values,1e-9)
            for x in zero:ax.scatter([x],[f],marker="v",facecolors="white",edgecolors="black",s=22)
            ax.set_yscale("log");ax.set_xticks(range(4),labels=[SHORT[m] for m in METHODS],rotation=15);ax.set_ylabel("Measured stage median / s")
            ax.set_xlim(-.6,3.6)
            ax.set_title(f"Target {e:.0e}: "+("setup / full matvec" if ri==0 else "matvec components"))
            ax.legend(handles=[plt.Rectangle((0,0),1,1,color="#bbbbbb",hatch=("","//","..","xx")[pi],alpha=1-.18*pi,label=phase) for pi,phase in enumerate(phases)],fontsize=6,loc="best")
    if not any_data:plt.close(fig);skip(name,"No saved setup or component clocks");return
    fig.text(.5,.02,"Setup/full action use the common formal batch; components use a separate serial diagnostic batch. External system load remains uncontrolled.\nSaved q25–q75 are sample ranges, not CI. Components overlap. Direct dense table/config assembly and component breakdown are N/A, not zero.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.09,1,1));save(fig,name,out,args)


def amortization(data,args):
    name="fig5_dynamics_step_amortized_cost";best=data["best"]
    if not best:skip(name,"No feasible measured stage costs");return
    eps=display_eps(best);fig,axes=plt.subplots(1,len(eps),figsize=(4.3*len(eps),4.4),squeeze=False);out=[];any_data=False
    applies=(1,10,100,1000)
    for ax,e in zip(axes[0],eps):
        for m in METHODS:
            r=best_row(best,m,e)
            if not r or not feasible(r,e):continue
            stages={key:phase_stats(r,data["raw"],key)["median_s"] for key in ("table_setup","config_setup","matvec")}
            if any(v is None for v in stages.values()):continue
            any_data=True
            for reuse,ls in ((1,"--"),(100,"-")):
                costs=[stages["table_setup"]/reuse+stages["config_setup"]+n*stages["matvec"] for n in applies]
                ax.plot(applies,costs,color=COLORS[m],ls=ls,marker="o",label=SHORT[m]+f"; table/{reuse} steps")
                out.extend(dict(r,epsilon=e,matvecs_per_step=n,table_reuse_steps=reuse,derived_step_cost_s=cost,**{key+"_s":v for key,v in stages.items()},
                    formula="table_setup_s/table_reuse_steps + config_setup_s + matvecs_per_step*matvec_s",measurement_role="linear amortization of measured stage medians; not a new trajectory measurement") for n,cost in zip(applies,costs))
        ax.set_xscale("log");ax.set_yscale("log");ax.set_xticks(applies,labels=[str(n) for n in applies]);ax.set_title(f"Target {e:.0e}");ax.set_xlabel("Cached matvecs per dynamics step");ax.set_ylabel("Derived cost per step / s")
    if not any_data:plt.close(fig);skip(name,"Missing measured table/config/matvec stage medians");return
    handles=[Line2D([],[],color=COLORS[m],label=SHORT[m]) for m in METHODS]+[Line2D([],[],color="black",ls="--",label="table rebuilt each step"),Line2D([],[],color="black",ls="-",label="table amortized over 100 steps")]
    fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=7)
    fig.text(.5,.12,"Derived stage-median model, not a measured trajectory; positions refresh each step (one config setup), with no neighbor-list reuse.\nA changed box/kappa/xi/cutoff/M/p invalidates table reuse. Direct dense total cost is N/A because fresh assembly is not timed.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.25,1,1));save(fig,name,out,args);write(args.results/"plot_amortization.csv",out)


def scaling(data,args):
    name="fig6_particle_number_scaling";rows=data["scaling"]
    if not rows:skip(name,"N scaling not measured within budget");return
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.1));out=[];grouped=defaultdict(list)
    global_error_floor=floor([r.get("max_E_G") for r in rows])
    for r in rows:
        if num(r.get("N")) is not None:grouped[(r.get("method"),r.get("epsilon"),r.get("batch"))].append(r)
    for (m,e,batch),rr in grouped.items():
        rr=sorted(rr,key=lambda r:r["N"]);valid=[r for r in rr if num(r.get("median_matvec_s")) is not None and num(r.get("max_E_G")) is not None]
        if not valid:continue
        label=SHORT.get(m,m)+f", eps={num(e):.0e}" if num(e) is not None else SHORT.get(m,m)
        color=COLORS.get(m,"#555555");axes[0].plot([r["N"] for r in valid],[r["median_matvec_s"] for r in valid],"o-",label=label,color=color)
        errfloor=global_error_floor
        for r in valid:axes[1].scatter([r["N"]],[r["max_E_G"] or errfloor],marker="v" if r["max_E_G"]==0 else "o",edgecolors=color,facecolors=color if num(e) and feasible(r,e) and r["max_E_G"]>0 else "white")
        out.extend(dict(r,error_display_floor=errfloor,scaling_role="single nested uniform configuration; h-rescaled parameters, at most four M refinements; no per-N optimum claim") for r in valid)
    if not out:plt.close(fig);skip(name,"Scaling rows lack measured error/runtime");return
    ns=sorted({r["N"] for r in out})
    for ax in axes:ax.set_xscale("log");ax.set_yscale("log");ax.set_xticks(ns,labels=[f"{n:g}" for n in ns]);ax.xaxis.set_minor_locator(NullLocator());ax.set_xlabel("Particle count N")
    for e in sorted({r["epsilon"] for r in out if num(r.get("epsilon")) is not None}):axes[1].axhline(e,color="#888888",ls=":",lw=.8)
    axes[0].set_ylabel("Cached full matvec / s");axes[1].set_ylabel("Single-configuration E_Gamma");axes[0].legend(fontsize=6)
    fig.text(.5,.02,"One nested uniform configuration per N; new N=64/128 references and the existing main N=256 reference.\nParameters preserve approximate h and allow limited M refinements; no nine-configuration or per-N optimization claim. Open markers: infeasible or exact zero.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.12,1,1));save(fig,name,out,args)


def projection_effect(data,args):
    name="fig7_projection_effect_diagnostics";best=data["best"];eps=display_eps(best)
    groups=[]
    for e in eps:
        r=best_row(best,"PPPM-grid-PSD",e)
        if not r or not feasible(r,e):groups.append([]);continue
        group=unique([v for v in data["validation"] if v.get("method")=="PPPM-grid-PSD" and v.get("id")==r.get("id")],("method","id","config"))
        groups.append(group)
    if not any(groups):skip(name,"No selected Grid PSD per-configuration projection diagnostics");return
    fig,axes=plt.subplots(2,len(eps),figsize=(4.4*len(eps),6.5),squeeze=False);out=[];order={"uniform":0,"moderate":1,"strong":2}
    for ci,(e,group) in enumerate(zip(eps,groups)):
        rr=sorted(group,key=lambda r:(order.get(str(r.get("family")),99),num(r.get("seed")) or 0));f=floor([num(r.get(k)) for r in rr for k in ("E_G","E_original","E_proj")])
        if not rr:
            for ax in axes[:,ci]:ax.text(.5,.5,"N/A",transform=ax.transAxes,ha="center")
        for key,color,ls,label in (("E_original",COLORS["PPPM-original"],"--","Same-parameter before projection"),("E_G",COLORS["PPPM-grid-PSD"],"-","Grid PSD after projection"),("E_proj","#df7b24",":","Operator projection difference")):
            x=[i for i,r in enumerate(rr) if num(r.get(key)) is not None];values=[rr[i][key] for i in x]
            axes[0,ci].plot(x,[v if v>0 else math.nan for v in values],ls=ls,color=color,marker="o",label=label)
            for i,v in zip(x,values):
                if v==0:axes[0,ci].scatter([i],[f],marker="v",facecolors="white",edgecolors=color,s=22)
        axes[0,ci].axhline(e,color="#777777",ls=":");axes[0,ci].set_yscale("log");axes[0,ci].set_title(f"Target {e:.0e}");axes[0,ci].set_ylabel("Relative full-operator error")
        active=0
        for i,r in enumerate(rr):
            count=num(r.get("negative_blocks"));M=num(r.get("M"));fraction=count/(M**3) if count is not None and M else num(first(r,"negative_fraction","fraction_negative_blocks"))
            if flag(r.get("active")) or (count or 0)>0:active+=1
            if fraction is not None:axes[1,ci].bar(i,fraction,color=COLORS["PPPM-grid-PSD"],width=.65)
            out.append(dict(r,epsilon=e,negative_block_fraction_computed=fraction,error_display_floor=f,diagnostic_role="explanation only; not a bound or PSD-eigenvalue hard constraint"))
        axes[1,ci].set_title(f"Projection active in {active}/{len(rr)} configurations");axes[1,ci].set_ylabel("Fraction of negative raw grid blocks");axes[1,ci].set_ylim(bottom=0)
        ticks=[f"{str(r.get('family','?'))[:1].upper()}{num(r.get('seed')):g}" if num(r.get("seed")) is not None else "?" for r in rr]
        for ax in axes[:,ci]:ax.set_xticks(range(len(rr)),labels=ticks);ax.set_xlabel("Configuration (family initial + seed)")
    handles,labels=next((ax.get_legend_handles_labels() for ax in axes[0] if ax.get_legend_handles_labels()[0]),([],[]));fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=7)
    fig.text(.5,.075,"Before/after use identical parameters, not the independently optimized Original baseline. Negative-block activity is explanatory only.\nLiteral zero errors use open triangles at the source-recorded display floor; machine-scale full-operator differences are not a uniform accuracy certificate.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.16,1,1));save(fig,name,out,args)


def validated_grid_projection(data):
    grouped=defaultdict(list)
    for r in data["validation"]:
        if r.get("method")=="PPPM-grid-PSD":grouped[r.get("id")].append(r)
    selected=defaultdict(list)
    for r in data["best"]:
        if r.get("method")=="PPPM-grid-PSD" and num(r.get("epsilon")) is not None and feasible(r,r["epsilon"]):selected[r.get("id")].append(r["epsilon"])
    out=[]
    for id_,rows in grouped.items():
        rr=unique(rows,("method","id","config"));r=rr[0]
        complete=len(rr)==9 and all(num(v.get("E_G")) is not None and num(v.get("E_original")) is not None and num(v.get("E_proj")) is not None for v in rr)
        result={k:r.get(k) for k in ("method","id","xi","sR","sF","sF_effective","k_c","r_c","M","p","eta","stage","origin")}
        result.update(n_configs=len(rr),nine_configuration_error_complete=complete,
            max9_E_before=max(v["E_original"] for v in rr) if complete else None,
            max9_E_after=max(v["E_G"] for v in rr) if complete else None,
            max9_E_proj=max(v["E_proj"] for v in rr) if complete else None,
            negative_blocks=max((num(v.get("negative_blocks")) or 0) for v in rr),
            selected_epsilons=";".join(f"{e:.0e}" for e in sorted(set(selected[id_]),reverse=True)),
            config_ids=";".join(sorted(str(v.get("config")) for v in rr)),
            validation_scope="all fresh nine-configuration validated Grid PSD candidates, not all screened search candidates")
        out.append(result)
    return sorted(out,key=lambda r:(num(r.get("xi")) or 0,num(r.get("M")) or 0,num(r.get("p")) or 0,num(r.get("sR")) or 0,num(r.get("sF")) or 0))


def validated_projection_effect(data,args):
    name="fig7b_validated_grid_projection_effect";rows=[r for r in validated_grid_projection(data) if r["nine_configuration_error_complete"]]
    if not rows:skip(name,"No complete nine-configuration Grid PSD projection comparisons");return
    fig,axes=plt.subplots(2,1,figsize=(12.8,6.7),sharex=True);out=[]
    f=floor([r["max9_E_proj"] for r in rows]);x=list(range(1,len(rows)+1))
    for key,color,mk,label in (("max9_E_before",COLORS["PPPM-original"],"o","Before projection, max over 9"),("max9_E_after",COLORS["PPPM-grid-PSD"],"s","After Grid PSD, max over 9")):
        axes[0].scatter(x,[r[key] for r in rows],marker=mk,color=color,s=30,facecolors="white" if mk=="o" else color,label=label)
    for i,r in zip(x,rows):
        value=r["max9_E_proj"]
        axes[1].scatter([i],[value or f],color="#df7b24",facecolors="white" if value==0 else "#df7b24",marker="v" if value==0 else "o",s=28)
        if r["negative_blocks"]>0:
            for ax in axes:ax.axvspan(i-.35,i+.35,color="#e8a04e",alpha=.16,zorder=0)
            axes[1].annotate(f"active: {r['negative_blocks']:g} raw block(s)",(i,value or f),xytext=(8,-16),textcoords="offset points",fontsize=7)
        if r["selected_epsilons"]:
            axes[0].scatter([i],[r["max9_E_after"]],marker="*",s=110,edgecolors="black",facecolors=COLORS["PPPM-grid-PSD"],zorder=5)
            axes[0].annotate(r["selected_epsilons"],(i,r["max9_E_after"]),xytext=(3,8),textcoords="offset points",fontsize=7)
        out.append(dict(r,candidate_index=i,projection_zero_display_floor=f))
    for e in EPSILONS[:3]:axes[0].axhline(e,color="#999999",ls=":",lw=.7)
    for ax in axes:ax.set_yscale("log")
    axes[0].set_ylabel("max9 relative full-operator error");axes[0].set_title(f"All {len(rows)} complete nine-configuration Grid PSD validation candidates");axes[0].legend(fontsize=7)
    axes[1].set_ylabel("max9 projection operator difference");axes[1].set_xticks(x,labels=[f"G{i:02d}" for i in x]);axes[1].set_xlabel("Discrete validated candidate index (parameters/id in source CSV)")
    fig.text(.5,.02,"Shaded columns have active projection; stars identify final selected targets. Candidate indices are not a continuous parameter or convergence study.\nThis is the nine-configuration validation set, not the full single-configuration search. Literal zero differences use open triangles at a recorded display floor.",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.1,1,1));save(fig,name,out,args)


def style():
    plt.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix","font.size":8,
        "axes.titlesize":9,"axes.titleweight":"bold","axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"axes.linewidth":.7,
        "xtick.direction":"in","ytick.direction":"in","axes.grid":True,"grid.color":"#dddddd","grid.linewidth":.5,
        "legend.frameon":True,"legend.fancybox":False,"lines.linewidth":1.2,"lines.markersize":4,"svg.fonttype":"none"})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--results",type=Path,default=HERE/"results")
    parser.add_argument("--output",type=Path,default=HERE/"figures");parser.add_argument("--dpi",type=int,default=240)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);style();data=inputs(args.results)
    jobs=(("parameters",parameters),("accuracy_cost",accuracy_cost),("comparison",comparison),("setup_components",setup_components),("amortization",amortization),("scaling",scaling),("projection_effect",projection_effect),("validated_projection_effect",validated_projection_effect))
    for label,job in jobs:
        try:job(data,args)
        except (ValueError,KeyError,TypeError,IndexError) as exc:plt.close("all");skip(label,f"{type(exc).__name__}: {exc}")
    STATUS.update(input_files=[str(p) for p in data["paths"]],input_rows={k:len(v) for k,v in data.items() if k!="paths"})
    (args.output/"plot_status.json").write_text(json.dumps(STATUS,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(STATUS,ensure_ascii=False))


if __name__=="__main__":main()
