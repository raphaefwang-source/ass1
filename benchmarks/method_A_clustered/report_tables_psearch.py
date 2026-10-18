#!/usr/bin/env python3
"""
Markdown tables for report_psearch_direct512.md, generated from the result files (no number is typed by hand).
Writes results_psearch/report_tables.md.

    python3 report_tables_psearch.py
"""
import common as C
import numpy as np  # noqa: F401

from run_psearch_dense512 import RESULTS_PSEARCH, CONFIGS, path
from run_direct512 import out_path, QS
from make_figures_psearch import load_stage1, per_p_rows

TAU = C.EPS_TARGET


def f(v, fmt):
    return "–" if v is None else format(v, fmt)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def stage1(md):
    cm = C.read_json(RESULTS_PSEARCH / "cost_model_p.json")["configs"]
    md.append("## Stage 1 tables\n")
    md.append("### Per-p cost model (N = 512, seconds -> ms)\n")
    rows = []
    for name in CONFIGS:
        m = cm[name]["models"]
        for p in range(4, 12):
            a, b = m["cached"]["per_p"][str(p)], m["pairdiff"]["per_p"][str(p)]
            ia, ib = m["cached"]["per_p_info"][str(p)], m["pairdiff"]["per_p_info"][str(p)]
            rows.append([name, str(p), f(1e3 * a["S"], ".3f"), f(1e9 * a["G"], ".3f"), f(ia["C_P_equiv"] * 1e9, ".2f"),
                         f(1e3 * a["C_0"], ".3f"), f(1e3 * b["C_0"], ".3f"),
                         f(max(abs(e) for e in ia["full_rel_err"]), ".2f"),
                         f(max(abs(e) for e in ib["full_rel_err"]), ".2f")])
    md.append(table(["config", "p", "S_p [ms]", "G_p [ns/M^3]", "C_P(p) = S_p/(N p^3) [ns]", "C_0 cached [ms]",
                     "C_0 pairdiff [ms]", "max full-action fit err cached", "pairdiff"], rows))
    md.append("\nShared per configuration: " + "; ".join(
        f"{n}: C_R cached {cm[n]['models']['cached']['C_R'] * 1e9:.2f} ns, pairdiff "
        f"{cm[n]['models']['pairdiff']['C_R'] * 1e9:.2f} ns per image; C_F {cm[n]['models']['cached']['C_F'] * 1e9:.3f} ns"
        f" per M^3 ln M^3; ranking model: {cm[n]['rank_impl']}" for n in CONFIGS) + "\n")
    for name in CONFIGS:
        sel, bench, opt = load_stage1(name)
        rows = per_p_rows(name, sel, bench, opt)
        lam = sel["lam_lb"]
        md.append(f"### {name}: model optimum of every p (certified errors absolute; cert_rel = sum / Lambda_lb, "
                  f"Lambda_lb = {lam:.4f}, ||Gamma_ref||_2 = {sel['norm2']:.4f})\n")
        md.append(table(["p", "xi", "r_c", "k_c", "M", "eta", "n_img", "eps_S", "eps_F", "eps_mesh (HZ)",
                         "eps_quad (est.)", "cert_rel", "measured E_op", "T_pred [ms]", "T_action cached / pairdiff [ms]",
                         "T_setup cached / pairdiff [ms]"],
                        [[str(r["p"]), f(r["model_xi"], ".3f"), f(r["model_rc"], ".4f"), f(r["model_kc"], ".4f"),
                          str(r["model_M"]), f(r["model_eta"], ".3f"), f(r["model_n_img"], ".0f"),
                          f(r["model_eps_S"], ".2e"), f(r["model_eps_F"], ".2e"), f(r["model_eps_mesh"], ".2e"),
                          f(r["model_eps_quad_est"], ".1e"), f(r["model_cert_rel"], ".4e"), f(r["model_E_op"], ".3e"),
                          f(r["model_T_pred_ms"], ".2f"),
                          f(r["model_T_action_cached_ms"], ".2f") + " / " + f(r["model_T_action_pairdiff_ms"], ".2f"),
                          f(r["model_T_setup_cached_ms"], ".0f") + " / " + f(r["model_T_setup_pairdiff_ms"], ".0f")]
                         for r in rows]))
        md.append(f"\n### {name}: measured fastest passing candidate of every p (fastest in the confirmation run; screening for a p not confirmed)\n")
        md.append(table(["p", "role", "xi", "r_c", "k_c", "M", "eta", "n_img", "cert_rel", "E_op", "impl",
                         "T_pred [ms]", "screen [ms]", "confirm median [q1, q3] [ms]", "T_setup [ms]"],
                        [[str(r["p"]), r["best_roles"], f(r["best_xi"], ".4f"), f(r["best_rc"], ".4f"),
                          f(r["best_kc"], ".4f"), str(r["best_M"]), f(r["best_eta"], ".3f"), str(r["best_n_img"]),
                          f(r["best_cert_rel"], ".4e"), f(r["best_E_op"], ".3e"), r["best_impl"],
                          f(r["best_T_pred_ms"], ".2f"), f(r["best_T_action_screen_ms"], ".2f"),
                          (f(r["best_T_action_confirm_ms"], ".2f") + f" [{f(r['best_T_action_confirm_q1_ms'], '.2f')}, "
                           f"{f(r['best_T_action_confirm_q3_ms'], '.2f')}]") if r["best_T_action_confirm_ms"] else "–",
                          f(r["best_T_setup_ms"], ".0f")] for r in rows]))
        md.append(f"\n### {name}: confirmation run ({opt['rounds']} interleaved rounds, both implementations)\n")
        md.append(table(["candidate", "impl", "median [ms]", "q1", "q3", "IQR/median", "T_pred [ms]", "E_op",
                         "cert_rel"],
                        [[r["key"], r["impl"], f(r["T_action_median_ms"], ".3f"), f(r["T_action_q1_ms"], ".3f"),
                          f(r["T_action_q3_ms"], ".3f"), f(r["T_action_iqr_rel"], ".3f"), f(r["T_pred_ms"], ".2f"),
                          f(r["E_op"], ".3e"), f(r["cert_rel"], ".4e")] for r in opt["confirm_rows"]]))
        b = opt["best"]
        md.append(f"\n**Measured optimum {name}:** {b['key']} with the {b['impl']} action: {b['T_action_median_ms']:.3f} ms "
                  f"[{b['T_action_q1_ms']:.3f}, {b['T_action_q3_ms']:.3f}], E_op {b['E_op']:.3e}, cert_rel "
                  f"{b['cert_rel']:.4e}; p at search boundary: {opt['boundary_p']}.\n")
        cands = bench["candidates"]
        npass = sum(c["E_op"] <= TAU for c in cands)
        dev = max(c["cached_vs_pairdiff_rel"] for c in cands)
        md.append(f"Candidates: {len(cands)}; E_op <= tau: {npass}/{len(cands)}; max E_op "
                  f"{max(c['E_op'] for c in cands):.3e}; max cached-vs-pairdiff action difference {dev:.1e}; "
                  f"max E_op/cert_rel x Lambda_lb/||Gamma_ref|| "
                  f"{max(c['E_op'] / (c['cert_rel_max'] * sel['lam_lb'] / sel['norm2']) for c in cands):.3f}.\n")


def stage2(md):
    md.append("## Stage 2 tables\n")
    for name in CONFIGS:
        if not out_path("stage2", name).exists():
            continue
        cv = C.read_json(out_path("convergence", name))
        d = C.read_json(out_path("stage2", name))
        md.append(f"### {name}\n")
        md.append(f"Direct truncation: R_meas = {cv['R_meas']} (E_ref {cv['row_meas']['E_ref']:.3e}, E_conv "
                  f"{cv['row_meas']['E_conv']:.3e}, n_img {cv['row_meas']['n_img']}), R_cert = {cv['R_cert']} (bound "
                  f"{cv['row_cert']['bound_rel_lam']:.3e}, n_img {cv['row_cert']['n_img']}); "
                  f"Gamma_D(R_max = {cv['R_max']}) vs Gamma_ref: {cv['E_ref_at_Rmax']:.2e}.\n")
        pp = d["pppm"]
        md.append(f"PPPM: p = {pp['p']}, xi = {pp['xi']:.4f}, r_c = {pp['rc']:.4f}, k_c = {pp['kc']:.4f}, M = {pp['M']}, "
                  f"eta = {pp['eta']:.3f}, action implementation {pp['impl']}.\n")
        acc = d["accuracy"]
        lab = {"direct_images_Rmeas": "direct image-resolved, R_meas", "direct_images_Rcert": "direct image-resolved, R_cert",
               "direct_pairs_Rmeas": "direct pair-summed, R_meas", "direct_pairs_Rcert": "direct pair-summed, R_cert",
               "dense_phys_Rmeas": "dense physical Gamma_D(R_meas), GEMV", "pppm": "PPPM (Stage 1 optimum)"}
        akey = {"direct_images_Rmeas": "direct_images_Rmeas", "direct_images_Rcert": "direct_images_Rcert",
                "direct_pairs_Rmeas": "direct_pairs_Rmeas", "direct_pairs_Rcert": "direct_pairs_Rcert",
                "dense_phys_Rmeas": "dense_phys_GEMV", "pppm": f"pppm_{pp['impl']}"}
        eop = {"direct_pairs_Rmeas": d["E_op"]["dense_phys_Rmeas"], "direct_images_Rmeas": d["E_op"]["dense_phys_Rmeas"],
               "dense_phys_Rmeas": d["E_op"]["dense_phys_Rmeas"], "direct_pairs_Rcert": d["E_op"]["dense_phys_Rcert"],
               "direct_images_Rcert": d["E_op"]["dense_phys_Rcert"], "pppm": d["E_op"]["pppm_Gamma_h"]}
        st_ = {**d["setups"]}
        sk = {"direct_images_Rmeas": "direct_images_Rmeas", "direct_images_Rcert": "direct_images_Rcert",
              "direct_pairs_Rmeas": "direct_pairs_Rmeas", "direct_pairs_Rcert": "direct_pairs_Rcert",
              "dense_phys_Rmeas": "dense_phys_Rmeas", "pppm": f"pppm_setup_{pp['impl']}"}
        rows = []
        for k, v in d["summary"].items():
            s = st_[sk[k]]
            a = (d["actions_fast"].get(akey[k]) or d["actions_slow"][akey[k]])
            rows.append([lab[k], f"{s['median_ms']:.1f} [{s['q1_ms']:.1f}, {s['q3_ms']:.1f}] (n={s['n']})",
                         f"{a['median_ms']:.3f} [{a['q1_ms']:.3f}, {a['q3_ms']:.3f}] (n={a['n']})",
                         f"{v['action_speedup_vs_pppm']:.3f}", f"{v['setup_ratio_vs_pppm']:.2f}",
                         f"{eop[k]:.3e}", f"{acc[akey[k]]['max_err_opnorm']:.2e}",
                         f"{acc[akey[k]]['translation_null_opnorm']:.1e}"])
        md.append(table(["method", "setup median [q1, q3] [ms]", "action median [q1, q3] [ms]",
                         "T_action / T_action(PPPM)", "T_setup / T_setup(PPPM)", "E_op vs Gamma_ref",
                         "max ||A X - Gamma_ref X||/(||Gamma_ref|| ||X||)", "translation null"], rows))
        md.append("\nEnd-to-end per configuration of positions, T_setup + q T_action [ms] (measured medians):\n")
        md.append(table(["method"] + [f"q = {q}" for q in QS] + ["ratio to PPPM at q = 1 / 80", "break-even q vs PPPM"],
                        [[lab[k]] + [f"{v['total_ms'][str(q)]:.1f}" for q in QS]
                         + [f"{v['total_ratio_vs_pppm']['1']:.2f} / {v['total_ratio_vs_pppm']['80']:.2f}",
                            f(v["break_even_q_vs_pppm"], ".1f")] for k, v in d["summary"].items()]))
        o = d
        md.append(f"\nOne-time (not in setup or action): PPPM mesh + influence function "
                  f"{o['once_pppm_mesh_influence']['median_ms']:.1f} ms (n={o['once_pppm_mesh_influence']['n']}); "
                  f"Gamma_ref generation (LatticeFriction, rel_tol 1e-13) {o['once_reference_generation']['median_ms']:.0f} ms "
                  f"(n={o['once_reference_generation']['n']}; regenerated vs stored max abs diff "
                  f"{o['reference_regenerated_vs_raw_maxabs']:.1e}). Dense GEMV with Gamma_ref itself: "
                  f"{o['actions_fast']['dense_ref_GEMV']['median_ms']:.3f} ms. Images: {o['n_images']}. Identity: "
                  f"{ {k: (f'{v:.1e}' if isinstance(v, float) else v) for k, v in o['identity'].items()} }.\n")


def supplementary(md):
    rows = []
    for name in CONFIGS:
        p = path("dense512", name)
        if not p.exists():
            continue
        d = C.read_json(p)
        e = d["end_to_end"]
        idc = d["identity"]
        rows.append([name, f"{d['builds']['dense_project']['median_ms']:.0f}", f"{d['builds']['dense_fast']['median_ms']:.0f}",
                     f"{d['T_dense_basis']['median_ms']:.0f}", f"{e['T_GEMV_ms']:.3f}",
                     f"{d['builds']['setup_' + d['production_impl']]['median_ms']:.1f}", f"{e['T_action_ms']:.2f}",
                     f"{e['ratio_GEMV_over_action']:.3f}", f(e["builds"]["dense_project"]["break_even_q"], ".1f"),
                     f(e["builds"]["dense_fast"]["break_even_q"], ".1f"),
                     f"{max(idc['max_rel_random'].values()):.1e}", f"{idc['basis_vs_project_rel2']:.1e}",
                     f"{idc['fast_vs_project_rel2']:.1e}", f"{idc['E_op_project']:.3e}"])
    if rows:
        md.append("## Supplementary: dense discrete Gamma_h (the same PPPM operator), issue #2 item\n")
        md.append(table(["config", "T_build project GammaH.dense [ms]", "T_build vectorised [ms]", "T_build basis [ms]",
                         "T_GEMV(Gamma_h) [ms]", "T_PPPM_setup [ms]", "T_PPPM_action [ms]", "T_GEMV/T_action",
                         "break-even q (project build)", "break-even q (vectorised)", "max rel dense vs matrix-free",
                         "basis vs project", "vectorised vs project", "E_op"], rows))


def reproducibility():
    """Machine, versions, thread pools, seeds, code hashes and commands of this round in one file."""
    from run_direct512 import RESULTS_DIRECT
    prov = {}
    for name in CONFIGS:
        for kind, p in (("cost_model", RESULTS_PSEARCH / "cost_model_p.json"), ("confirm", path("optimum", name, "main")),
                        ("dense_gamma_h", path("dense512", name)), ("direct_converge", out_path("convergence", name)),
                        ("direct_bench", out_path("stage2", name))):
            if p.exists():
                prov[f"{kind}:{name}" if kind != "cost_model" else kind] = C.read_json(p)["provenance"]
    cmds = []
    for p in (RESULTS_PSEARCH / "commands.log", RESULTS_DIRECT / "commands.log"):
        if p.exists():
            cmds += p.read_text().splitlines()
    C.write_json(RESULTS_PSEARCH / "reproducibility_psearch.json",
                 dict(seeds=dict(x_inputs="default_rng([20261018, 512, 7])", interleaving="default_rng([20261018, 3, s])",
                                 probes="default_rng([20261018, 13, N])", alias_test="default_rng([20261018, 21])",
                                 setup_order_stage2="default_rng([20261018, 41, N])"),
                      commands=cmds, provenance_by_stage=prov, now=C.provenance()))


def main():
    reproducibility()
    md = ["# Generated tables (report_tables_psearch.py)\n"]
    stage1(md)
    stage2(md)
    supplementary(md)
    (RESULTS_PSEARCH / "report_tables.md").write_text("\n".join(md) + "\n")
    C.log(f"wrote {RESULTS_PSEARCH / 'report_tables.md'}")


if __name__ == "__main__":
    main()
