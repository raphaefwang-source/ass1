#!/usr/bin/env python3
"""
Markdown tables for lj075_results/REPORT.md, generated from the result JSON files only (no hand-copied numbers).
Writes lj075_results/summary_tables.md. Sections whose source file is missing are skipped and listed at the end.

    python3 lj075_summary.py [--law A]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lj075_common as C  # noqa: E402

OUT = C.OUT


def load(name):
    f = OUT / name
    return json.loads(f.read_text()) if f.exists() else None


def fmt(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{100 * x:+.{nd - 1}f}%" if pct else f"{x:.{nd}g}"


def ci(c, pct=True, nd=3):
    """c = (mean, lo, hi) or dict with rel_diff / abs_diff and ci95."""
    if isinstance(c, dict):
        m = c.get("rel_diff", c.get("abs_diff"))
        lo, hi = c["ci95"]
    else:
        m, lo, hi = c
    if pct:
        return f"{100 * m:+.{nd - 1}f}% [{100 * lo:+.{nd - 1}f}, {100 * hi:+.{nd - 1}f}]"
    return f"{m:+.{nd}g} [{lo:+.{nd}g}, {hi:+.{nd}g}]"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def sec_operator(law, md, missing):
    rows = []
    for f in sorted(OUT.glob(f"verify_*_{law}*.json")):
        v = json.loads(f.read_text())
        per = v["per_state"]
        op = [s["op_err"] for s in per.values()]
        ok = [s["op_err_vs_k_nonzero_part"] for s in per.values()]
        cc = v.get("chosen_checks", {})
        mx = lambda k: max((cc[n][d].get(k, 0) for n in cc for d in cc[n]), default=float("nan"))  # noqa: E731
        inc = v.get("increment_normalised", {})
        pp = v["pppm"]
        rows.append([f.name, len(per), f"{pp['xi']}, {pp['s']}, {pp['eta']:.4g}, {pp['p']} (M {v['mesh']['M']})",
                     f"{min(op):.2e} – {max(op):.2e}", f"{max(ok):.2e}",
                     f"{max(s['op_err_low12_eigs'] for s in per.values()):.1e}",
                     f"{max(s['sym_resid'] for s in per.values()):.0e} / "
                     f"{max(max(s['null_resid'], s['null_resid_ref']) for s in per.values()):.0e} / "
                     f"{min(min(s['lam_min'], s['lam_min_ref']) for s in per.values()):.3f}",
                     "{noise}/{damp}/{couple}".format(**v["chosen_ranks"]) if "chosen_ranks" in v else "–",
                     f"{mx('ostep_krylov'):.1e} / {inc.get('ostep_krylov_increment_max', float('nan')):.1e}",
                     f"{mx('fdt_lanczos'):.1e}", f"{mx('coupling_input_krylov'):.1e}",
                     "PASS" if v.get("passed") else "FAIL", f"{v.get('cpu_s', float('nan')) / 3600:.2f}"])
    if not rows:
        missing.append("verification")
        return
    md.append("### Operator and Lanczos verification (static, per state set)\n")
    md.append(table(["file", "states", "PPPM ξ, s, η*, p", "ε_op (full norm) min – max", "ε_op vs k≠0 part (max)",
                     "low-12 eig err", "sym / null / λ_min", "ranks n/d/c", "O-step Krylov (vec / incr)",
                     "FDT Lanczos", "coupling Krylov", "verdict", "core-h"], rows))
    md.append("")


def sec_opsens(law, md, missing):
    d = load("operator_sensitivity.json")
    if not d or law not in d:
        missing.append("operator_sensitivity")
        return
    r = d[law]
    md.append(f"### Physical effect of the operator error (production vs tight PPPM, lock-step, same noise; "
              f"n = {r['n']})\n")
    keys = list(r["per_replica"][0].keys())
    rows = [[k] + [f"{p[k]:+.2e}" for p in r["per_replica"]] + [f"{r['max_abs'][k]:.2e}"] for k in keys]
    md.append(table(["quantity (rel.; pressure abs.)"] + [f"rep {i + 1}" for i in range(r["n"])] + ["max |diff|"],
                    rows))
    md.append("\nPathwise RMS position difference: " + "; ".join(
        ", ".join(f"{k} {v:.1e}" for k, v in p["dq"].items()) for p in r["paths"]) + "\n")


def sec_dt(law, md, missing):
    d = load("dt_study.json")
    if not d or law not in d:
        missing.append("dt_study")
        return
    for tag, r in d[law].items():
        md.append(f"### dt study, law {law}: group {tag} (n = {r['n_replicas']} replicas {r['replicas']}, "
                  f"reference {r['reference']}, {r['cpu_core_hours']:.2f} core-h)\n")
        comps = list(r["comparisons"])
        rows = []
        lab = {"T_kin": "T_kin (rel., tol 1%)", "U_per_N": "U/N (rel., tol 1%)"}
        for k in ("T_kin", "U_per_N"):
            rows.append([lab[k]] + [f"{ci(r['comparisons'][c][k])} **{r['comparisons'][c][k]['verdict']}**"
                                    for c in comps])
        rows.append(["pressure (abs., tol 0.0075)"] + [
            f"{ci(r['comparisons'][c]['pressure'], pct=False)} **{r['comparisons'][c]['pressure']['verdict']}**"
            for c in comps])
        rows.append(["RDF E_g: debiased [lower, upper] (tol 1%)"] + [
            f"{fmt(r['comparisons'][c]['rdf']['E_g_debiased'], pct=True)} [{fmt(r['comparisons'][c]['rdf']['E_g_lower95'], pct=True)}, "
            f"{fmt(r['comparisons'][c]['rdf']['E_g_upper95'], pct=True)}] **{r['comparisons'][c]['rdf']['verdict']}**"
            for c in comps])
        for k, nm in (("first_peak_height", "RDF first-peak height"), ("first_peak_position", "RDF first-peak position")):
            rows.append([nm + " (rel.)"] + [f"{ci(r['comparisons'][c]['rdf'][k])} **{r['comparisons'][c]['rdf'][k]['verdict']}**"
                                             for c in comps])
        rows.append([f"VACF E_V = max|ΔC|/C(0), t ≤ {r['comparisons'][comps[0]]['vacf']['t_max']}: [lower, upper]"] + [
            f"[{fmt(r['comparisons'][c]['vacf']['E_V_lower95'], pct=True)}, {fmt(r['comparisons'][c]['vacf']['E_V_upper95'], pct=True)}]"
            f" (noise floor {fmt(r['comparisons'][c]['vacf']['noise_floor_median'], pct=True)}) **{r['comparisons'][c]['vacf']['verdict']}**"
            for c in comps])
        for j, x in enumerate(r["comparisons"][comps[0]]["msd"]):
            rows.append([f"MSD(t = {x['t']:g}) (rel.)"] + [
                f"{ci(r['comparisons'][c]['msd'][j])} **{r['comparisons'][c]['msd'][j]['verdict']}**" for c in comps])
        rows.append(["D_GK to t_max (information)"] + [ci(r["comparisons"][c]["D_GK_info_short"]) for c in comps])
        rows.append(["operator/Lanczos on-line ≤ budget/10"] + [str(r["comparisons"][c]["operator_ok"]) for c in comps])
        rows.append(["all criteria PASS"] + [str(r["comparisons"][c]["all_pass"]) for c in comps])
        md.append(table(["criterion: mean difference [95% CI]"] + [c.replace("_vs_", " vs ") for c in comps], rows))
        md.append("")
        lm = r["level_means"]
        md.append("Level means (95% t-CI over replicas): " + "; ".join(
            f"dt {k}: T {lm[k]['T_kin'][0]:.4f}, U/N {lm[k]['U_per_N'][0]:.4f}, P {lm[k]['pressure'][0]:.3f}"
            for k in lm) + "\n")
        mon = r["lanczos_monitor_max"]
        md.append("On-line Lanczos error estimates on actual inputs (max over replicas; damp / noise / coupling): " +
                  "; ".join(f"dt {k}: " + " / ".join(fmt(x, 2) for x in v) for k, v in mon.items()) +
                  f" (budget/10 = {r['operator_budget'] / 10:.0e})\n")
        st = {c: r["comparisons"][c]["stationarity"] for c in comps}
        md.append("Stationarity of the paired differences (late-minus-early window, 95% CI): " + "; ".join(
            f"{c}: " + ", ".join(f"{k} {ci(v['late_minus_early'], pct=(v['unit'] != 'absolute'))}" for k, v in s.items())
            for c, s in st.items()) + "\n")


def sec_equil(law, md, missing):
    e = load("equilibration.json")
    if not e or law not in e:
        missing.append("equilibration")
        return
    r = e[law]
    md.append(f"### Equilibration, law {law}: {r['n_runs']} runs, start types {r['start_types']}, dt {r['dt']}, "
              f"{r['cpu_core_hours']:.2f} core-h\n")
    b = r["burn_in"]
    obs = list(next(iter(b["by_start_type"].values())).keys())
    rows = [[g] + [f"{v[o]:g}" for o in obs] for g, v in b["by_start_type"].items()]
    md.append(f"Burn-in rule: {b['rule']}. Estimate: **{b['estimate']:g}** time units.\n")
    md.append(table(["start type"] + obs, rows))
    md.append("")
    rows = [[o, f"{v['mean']:.5g}", f"{v['block_sd']:.3g}", f"{v['halves_paired_t_p']:.2f}"] for o, v in b["pool"].items()]
    md.append(table(["observable", "pool mean (t ≥ 25)", "2-unit block sd", "pool halves paired-t p"], rows))
    md.append("")
    mi = b["mser_info"]
    md.append("MSER truncation (information): " + "; ".join(
        f"{n}: " + ", ".join(f"{o} {v['t']:g}{'*' if v['at_boundary'] else ''}" for o, v in x.items())
        for n, x in mi.items()) + " (* = at the d = n/2 boundary, not determined)\n")
    a = r["ic_dependence_after_burn_in"]
    rows = [[o, ", ".join(f"{g} {m:.5g}" for g, m in v["group_means"].items()), fmt(v["F"]), fmt(v["p"]),
             str(v["dependence_detected"]), f"{v['between_run_sd']:.3g}", f"{v['se_single_run_median']:.3g}",
             f"{np.median(list(v['tau_int'].values())):.3g}", str(all(v["tau_reliable"].values()))]
            for o, v in a.items()]
    md.append(table(["observable", "start-type means", "ANOVA F", "p", "dependence (p<0.01)", "between-run sd",
                     "single-run SE (median)", "τ_int median", "τ reliable (T ≥ 50τ)"], rows))
    md.append("")
    s = r["sampling_requirements"]
    rows = [[k, f"{v['target_half_width']:.3g}", f"{v['s_run']:.3g}", f"{v['run_length']:g}", f"{v['n_runs']}",
             f"{v['total_time_needed']:.0f}"] for k, v in s.items() if k != "D"]
    md.append(table(["observable", "target 95% half-width", "sd of run means", "run length", "runs",
                     "total production time needed"], rows))
    md.append("")
    for lab, v in s.get("D", {}).items():
        md.append(f"- {lab} at {v['at']}: mean {v['mean']:.4g}, sd over runs {v['s_run']:.3g} ({100 * v['s_run'] / v['mean']:.1f}%), "
                  f"run length {v['run_length']:g}; replicas needed at this length for ±5% (95%): "
                  f"{v['replicas_needed_at_this_length']}; total time {v['total_time_needed']:.0f}. {v['note']}")
    md.append("")
    dfu = r["diffusion"]
    rows = [[k, ci(v, pct=False)] for k, v in dfu["D_GK"].items()]
    md.append(table(["GK upper limit τ", "D_GK(τ) mean [95% CI]"], rows))
    md.append("")
    rows = [[p["tau"], ci((p["rel_change"], *p["ci95"])), p.get("status", "plateau" if p["plateau"] else "no")]
            for p in dfu["gk_plateau"]]
    rows += [[p["windows"], ci((p["rel_change"], *p["ci95"])), p.get("status", "plateau" if p["plateau"] else "no")]
             for p in dfu["msd_plateau"]]
    md.append(table(["step (GK τ or MSD window)", "relative change [95% CI]", "status"], rows))
    md.append(f"\nMSD windows: " + "; ".join(f"[{k}] {ci(v, pct=False)}" for k, v in dfu["D_MSD"].items()) +
              f". Long-time D determined: **{dfu['long_time_D_determined']}**.\n")


def sec_sampling(law, md, missing):
    s = load("sampling.json")
    if not s or law not in s:
        missing.append("sampling")
        return
    r = s[law]
    md.append(f"### Save interval, law {law} (dt {r['dt']}, burn-in {r['burn_in']}, {len(r['runs'])} runs)\n")
    ref = r["every_step_reference"]
    md.append(f"Every-step reference: τ½ {ci(ref['tau_half'], pct=False)}; D_GK(T): " +
              ", ".join(f"T={k} {ci(v, pct=False)}" for k, v in ref["gk"].items()) + "\n")
    rows = []
    for st, v in r["by_stride"].items():
        q, p = v["quadrature"], v["production"]
        rows.append([st, f"{v['save_interval']:g}", f"{100 * v['max_run_gk_bias']:.3f}%", q["lags_before_half"],
                     ci(q["tau_half"]), ", ".join(f"{k}: {ci(x)}" for k, x in p["msd"].items()), ci(p["D_msd"]),
                     "PASS" if v["passes_vacf_gk_rule"] else "FAIL", "PASS" if v["passes_msd_rule"] else "FAIL"])
    md.append(table(["stride", "Δ", "max run GK quadrature bias", "lags before C/C0 = ½", "τ½ rel.",
                     "production-like MSD(t) rel.", "D_MSD[2,8] rel.", "VACF/GK rule", "MSD rule"], rows))
    md.append(f"\nLargest passing stride: {r['largest_passing']} (MSD only: {r['largest_passing_msd_only']}); "
              f"VACF resolved at every step: {r['vacf_resolved_at_every_step']}.\n")


def sec_longD(law, md, missing):
    d = load("longD.json")
    if not d or law not in d:
        missing.append("longD")
        return
    r = d[law]
    md.append(f"### Long-time diffusion, law {law}: levels {r['levels']}, n = {r['n_replicas']} "
              f"(replicas {r['replicas']}), T0 = {d['T0']}, {r['cpu_core_hours']:.2f} core-h\n")
    for dt, v in r["per_level"].items():
        rows = [[k, ci(x, pct=False)] for k, x in v["D_MSD"].items()]
        md.append(f"dt {dt}: T_kin {ci(v['T_kin'], pct=False, nd=5)}")
        md.append(table(["MSD window", "D_MSD [95% CI]"], rows))
        rows = [[p["windows"], ci((p["rel_change"], *p["ci95"])), p["status"]] for p in v["plateau"]]
        md.append(table(["step", "relative change [95% CI]", "status"], rows))
        md.append(f"\nLong-time D determined: **{v['long_time_D_determined']}**" +
                  (f", D = {ci(v['D_long'], pct=False)} (window {v['D_long_window']})" if v["D_long"] else "") + "\n")
    for k, v in r["dt_comparison"].items():
        rows = [[w, ci(x)] for w, x in v["rel_diff"].items()]
        md.append(f"{k}: free-particle prior x coth x − 1 (vs continuum) for dt: "
                  f"{', '.join(f'{100 * x:.3f}%' for x in v['prior_xcothx_minus_1_vs_continuum'])}; reference: "
                  f"{', '.join(f'{100 * x:.3f}%' for x in v['prior_ref'])}")
        md.append(table(["window", "D_dt / D_ref − 1 [95% CI]"], rows))
        md.append("")


def sec_cost(law, md):
    rows = []
    for f in sorted(OUT.glob(f"verify_*_{law}*.json")):
        v = json.loads(f.read_text())
        rows.append([f"static verification {f.stem}", f"{v.get('cpu_s', 0) / 3600:.3f}"])
    ps = load("prepared_states.json") or {}
    rows.append(["state preparation (all prepared states, both laws' inputs)",
                 f"{sum(v.get('cpu_s', 0) for k, v in ps.items() if not k.startswith('_') and '_x' not in k) / 3600:.3f}"])
    for name, key in (("dt_study.json", "dt study"), ("longD.json", "long-D")):
        d = load(name)
        if d and law in d:
            if name == "dt_study.json":
                for tag, r in d[law].items():
                    rows.append([f"{key} {tag}", f"{r['cpu_core_hours']:.3f}"])
            else:
                rows.append([key, f"{d[law]['cpu_core_hours']:.3f}"])
    e = load("equilibration.json")
    if e and law in e:
        rows.append(["equilibration / production-runner runs", f"{e[law]['cpu_core_hours']:.3f}"])
    osd = C.RAW / "operator_sensitivity"
    cpu = sum(json.loads(f.read_text())["cpu_s"] for f in osd.glob(f"{law}_rep*.json")) if osd.exists() else 0
    rows.append(["operator sensitivity runs", f"{cpu / 3600:.3f}"])
    pc = load("pppm_cost.json")
    if pc and law in pc.get("laws", {}):
        m = pc["laws"][law]
        md.append(f"### Production-step cost (dt {pc['dt']}, single thread, {pc['rounds']} interleaved rounds × "
                  f"{pc['steps_per_round']} steps)\n")
        md.append(table(["PPPM set", "PPPM", "ranks n/d", "verification", "CPU ms/step median [min, max]",
                         "rel. to production", "core-h per time unit"],
                        [[k, json.dumps(v["pppm"]), f"{v['ranks']['noise']}/{v['ranks']['damp']}",
                          (f"{v['verification']['file']} passed={v['verification']['passed']}, max ε_op "
                           f"{v['verification'].get('op_err_max', float('nan')):.2e}") if v["verification"].get("file")
                          else "none", f"{v['cpu_ms_per_step_median']:.1f} [{v['cpu_ms_per_step_range'][0]:.1f}, "
                                       f"{v['cpu_ms_per_step_range'][1]:.1f}]",
                          f"{v['relative_to_production']:.3f}", f"{v['core_hours_per_time_unit']:.4f}"]
                         for k, v in m.items()]))
        md.append("")
    md.append(f"### Core-hours, law {law} (process CPU time recorded by each script)\n")
    tot = sum(float(r[1]) for r in rows)
    md.append(table(["component", "core-h"], rows + [["**total recorded**", f"**{tot:.2f}**"]]))
    md.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--law", default="A")
    args = ap.parse_args()
    law = args.law
    md, missing = [f"# lj075 summary tables, law {law} (generated by lj075_summary.py; do not edit)\n"], []
    for f in (sec_operator, sec_opsens, sec_dt, sec_longD, sec_equil, sec_sampling):
        f(law, md, missing)
    sec_cost(law, md)
    if missing:
        md.append("Missing sources: " + ", ".join(missing) + "\n")
    (OUT / "summary_tables.md").write_text("\n".join(md) + "\n")
    print(f"wrote {OUT / 'summary_tables.md'}; missing: {missing}")


if __name__ == "__main__":
    main()
