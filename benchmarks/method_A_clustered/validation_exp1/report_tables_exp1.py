#!/usr/bin/env python3
"""Markdown tables for report_exp1.md, generated from results/ (python3 report_tables_exp1.py > results/tables_exp1.md)."""
import json

import numpy as np

from plot_exp1 import load, RES, FAMS


def e(x, d=2):
    return "–" if x is None else f"{x:.{d}e}"


def table_a():
    rows = load("split_configs.csv")
    radial = load("split_radial.csv")
    mp_rows = load("split_mpmath.csv")
    xis = sorted({r["xi"] for r in rows})
    out = ["| ξ | max rel. Frobenius (uniform / strong / moderate) | max pair rel. (all 9) | max abs. entry | "
           "radial sweep max (r ≥ 1e-6) | mpmath θ_S (ξr ≤ 7), rel. self | mpmath θ_L, rel. self | naive θ−θ_S at r = 1e-4 |",
           "|---|---|---|---|---|---|---|---|"]
    for xi in xis:
        fro = [max(r["rel_fro"] for r in rows if r["xi"] == xi and r["family"] == f) for f in FAMS]
        pt = max(r["max_rel_pointwise"] for r in rows if r["xi"] == xi)
        ab = max(r["max_abs"] for r in rows if r["xi"] == xi)
        rad = max(r["rel_pointwise"] for r in radial if r["xi"] == xi)
        mS = max(q["err_theta_S_rel_self"] for q in mp_rows if q["xi"] == xi and q["xi"] * q["r"] <= 7.0)
        mL = max(q["err_theta_L_rel_self"] for q in mp_rows if q["xi"] == xi)
        nv = [q["err_naive_L_rel_self"] for q in mp_rows if q["xi"] == xi and abs(q["r"] - 1e-4) < 1e-9][0]
        out.append(f"| {xi} | {' / '.join(e(v) for v in fro)} | {e(pt)} | {e(ab)} | {e(rad)} | {e(mS)} | {e(mL)} | {e(nv)} |")
    n_pairs = sorted({(r["config"], r["n_pairs"]) for r in rows})
    out.append("")
    out.append("Pair images per configuration (|d| ≤ 8): " + ", ".join(f"{c} {n}" for c, n in n_pairs) + ".")
    return "\n".join(out)


def table_b():
    fits = load("shortrange_fits.csv")
    rows = load("shortrange.csv")
    summ = json.loads((RES / "shortrange_summary.json").read_text())
    out = ["| configuration | fit interval s | points | c (all) | c (s ≥ 3) | E_S at s = 4.1 / 4.2 | floor vs Γ_ref | floor vs Γ_D55 | "
           "bound/E (fit range) | Fourier tail of Γ_L^ex (certified, rel.) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for f in fits:
        mine = [r for r in rows if r["config"] == f["config"]]
        e41 = [r["E_S"] for r in mine if abs(r["s"] - 4.1) < 1e-9][0]
        e42 = [r["E_S"] for r in mine if abs(r["s"] - 4.2) < 1e-9][0]
        f55 = float(np.median([r["E_S_vs_D55"] for r in mine[-5:]]))
        out.append(f"| {f['config']} | [{f['s_lo']:.1f}, {f['s_hi']:.1f}] | {f['n_points']} | {f['c']:.3f} | {f['c_s_ge_3']:.3f} | "
                   f"{e(e41)} / {e(e42)} | {e(f['floor'])} | {e(f55)} | {f['ratio_bound_over_E_min']:.2f}–"
                   f"{f['ratio_bound_over_E_max']:.2f} | {e(f['eps_F_exact_rel'], 1)} |")
    out.append("")
    out.append("| family | c (median curve, s ∈ [2.0, 5.3]) | c (median curve, s ≥ 3) | median floor |")
    out.append("|---|---|---|---|")
    for fam, v in summ["family_fits"].items():
        out.append(f"| {fam} | {v['c_median_curve']:.3f} | {v['c_median_curve_s_ge_3']:.3f} | {e(v['floor_median'])} |")
    trunc = max(abs(r["E_S"] - r["E_trunc_exact"]) / r["E_trunc_exact"] for r in rows if r["E_trunc_exact"] > 1e-12)
    out.append("")
    out.append(f"Largest relative difference between E_S and the exact-arithmetic tail ||Γ_S^full − Γ_S^rc||_2/||Γ_ref||_2 "
               f"over all points with tail > 1e-12: {trunc:.1e}. Same-neighbour steps: {summ['n_flat_steps_total']}.")
    return "\n".join(out)


def table_c_validity():
    mesh = load("action_mesh.csv")
    meta = json.loads((RES / "action_probes.json").read_text())["configs"]
    out = ["| N | cutoffs | r_c | k_c used (outermost shell n) | modes retained = shell count | valid M | invalid M (reason) | "
           "min \\|W_hat\\|^2 | excluded modes |", "|---|---|---|---|---|---|---|---|---|"]
    for N in (128, 256, 512):
        for st in ("production", "strict"):
            sub = [m for m in mesh if m["N"] == N and m["setting"] == st]
            v = [m for m in sub if m["valid"]]
            inv = sorted({(m["M"], m["reason"]) for m in sub if not m["valid"]})
            name = sub[0]["config"]
            ms = meta[name]["settings"][st]
            out.append(f"| {N} | {st} | {sub[0]['rc']:.10g} | {sub[0]['kc']:.12g} (n = {ms['n_shell']}) | "
                       f"{v[0]['n_modes']} = {v[0]['n_modes_expected']} | {v[0]['M']}–{v[-1]['M']} | "
                       f"{'; '.join(f'{M}: ' + r.replace('|', chr(92) + '|') for M, r in inv) or '—'} | {e(min(m['W2min'] for m in v), 1)} | "
                       f"{max(m['n_excluded'] for m in v)} |")
    mm = meta["strong_N512_s1"]["settings"]["production"]
    out.append("")
    out.append(f"N = 512 production: retained modes with the full-precision k_c = {mm['n_modes']}; with k_c truncated to "
               f"9.3585 (old code) = {mm['n_modes_if_kc_rounded_to_9p3585']} (shell n = 172 dropped).")
    return "\n".join(out)


def table_c_main():
    rows = load("action_errors.csv")
    mesh = load("action_mesh.csv")
    out = ["| cutoffs | M | Gaussian E_complete (min–max of 10) | slow spectral E_complete (max of 3) | fast E_complete | "
           "fixed-cutoff value E_trunc, Gaussian / slow (max) | dense E_op |", "|---|---|---|---|---|---|---|"]
    for st in ("production", "strict"):
        sub = [r for r in rows if r["config"] == "strong_N512_s1" and r["setting"] == st]
        for M in sorted({r["M"] for r in sub}):
            q = [r for r in sub if r["M"] == M]
            g = [r for r in q if r["probe_class"] == "gaussian"]
            sl = [r for r in q if r["probe_class"] == "spectral_slow"]
            fa = [r for r in q if r["probe_class"] == "spectral_fast"]
            m = [x for x in mesh if x["config"] == "strong_N512_s1" and x["setting"] == st and x["M"] == M][0]
            out.append(f"| {st}{' (production M)' if st == 'production' and M == 40 else ''} | {M} | "
                       f"{e(min(r['E_complete'] for r in g))}–{e(max(r['E_complete'] for r in g))} | "
                       f"{e(max(r['E_complete'] for r in sl))} | {e(fa[0]['E_complete'])} | "
                       f"{e(max(r['E_trunc'] for r in g))} / {e(max(r['E_trunc'] for r in sl))} | {e(m['E_op'])} |")
    return "\n".join(out)


def table_c_small():
    rows = load("action_errors.csv")
    fits = load("action_rate_fits.csv")
    out = ["| N | family | cutoffs | M (smallest / middle / largest valid) | Gaussian E_complete | slow E_complete | "
           "fast E_complete | LSQ rate, Gaussian E_complete |", "|---|---|---|---|---|---|---|---|"]
    for N in (128, 256, 512):
        for fam in ("uniform", "moderate", "strong"):
            for st in ("production", "strict"):
                sub = [r for r in rows if r["N"] == N and r["family"] == fam and r["setting"] == st]
                Ms = sorted({r["M"] for r in sub})
                sel = (Ms[0], Ms[len(Ms) // 2], Ms[-1])
                f = lambda cls, k: " / ".join(e(max(r[k] for r in sub if r["probe_class"] == cls and r["M"] == M), 1)  # noqa: E731
                                              for M in sel)
                fr = [q for q in fits if q["N"] == N and q["family"] == fam and q["setting"] == st and q["probe_class"] == "gaussian"
                      and q["quantity"].startswith("E_complete")]
                out.append(f"| {N} | {fam} | {st} | {' / '.join(map(str, sel))} | {f('gaussian', 'E_complete')} | "
                           f"{f('spectral_slow', 'E_complete')} | {f('spectral_fast', 'E_complete')} | "
                           f"{fr[0]['rate_lsq']:.2f}{' (flat: fixed-cutoff value)' if st == 'production' else ''} |")
    out.append("")
    out.append("Maxima over the 10 Gaussian probes (and the 3 seeds for N = 128, 256); N = 512 has one configuration per family.")
    return "\n".join(out)


def table_c_rates():
    out = ["| N | cutoffs | family | probe class | quantity | M range | points | LSQ rate | R^2 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for f in load("action_rate_fits.csv"):
        if not f["quantity"].startswith("E_complete") or f["setting"] != "strict":
            continue
        out.append(f"| {f['N']} | {f['setting']} | {f['family']} | {f['probe_class']} | {f['quantity']} | {f['M_lo']}–{f['M_hi']} | "
                   f"{f['n_points']} | {f['rate_lsq']:.2f} | {f['r2']:.4f} |")
    out.append("")
    out.append("p study (N = 512 strongly clustered, strict cutoffs, complete error, maximum over tested probes):")
    out.append("")
    out.append("| cutoffs | p | probe class | M range | points | LSQ rate | R^2 |")
    out.append("|---|---|---|---|---|---|---|")
    for f in load("action_p_rate_fits.csv"):
        if f["quantity"] != "E_complete" or f["setting"] != "strict":
            continue
        out.append(f"| {f['setting']} | {f['p']} | {f['probe_class']} | {f['M_lo']}–{f['M_hi']} | {f['n_points']} | "
                   f"{f['rate_lsq']:.2f} | {f['r2']:.4f} |")
    return "\n".join(out)


def table_c_checks():
    meta = json.loads((RES / "action_probes.json").read_text())["configs"]
    summ = json.loads((RES / "action_summary.json").read_text())
    tr = load("action_translation.csv")
    out = ["| N | cutoffs | translation ‖Γ_h T‖/‖Γ_ref‖, pair-difference | cached D − H | Γ_ref itself |",
           "|---|---|---|---|---|"]
    for N in (128, 256, 512):
        for st in ("production", "strict"):
            sub = [r for r in tr if r["N"] == N and r["setting"] == st]
            out.append(f"| {N} | {st} | {e(max(r['opnorm_pairdiff'] for r in sub))} | {e(max(r['opnorm_cached'] for r in sub))} | "
                       f"{e(max(r['opnorm_ref'] for r in sub))} |")
    m = meta["strong_N512_s1"]
    out.append("")
    out.append("Spectral probes of strong_N512_s1 (eigenvectors of sym(Γ_ref)):")
    out.append("")
    out.append("| probe | eigenvalue | ‖Γ_ref u − λu‖/λ | relative gap to neighbours | ‖Tᵀu‖ |")
    out.append("|---|---|---|---|---|")
    for pr in m["probes"]:
        if pr["probe_class"].startswith("spectral"):
            out.append(f"| {pr['probe_id']} | {pr['eigenvalue']:.6f} | {e(pr['eigen_residual_rel'], 1)} | "
                       f"{pr['rel_gap_to_neighbours']:.4f} | {e(pr['translation_component'], 1)} |")
    sl = m["strict_lattice_check"]
    out.append("")
    out.append(f"Translation eigenvalues of sym(Γ_ref) (N = 512): {', '.join(e(v, 1) for v in m['spectral']['translation_eigenvalues'])}; "
               f"λ_min(positive) = {m['spectral']['lambda_min_positive']:.6f}, λ_max = {m['spectral']['lambda_max']:.6f}.")
    out.append(f"Stricter LatticeFriction (rel_tol 1e-15, degree 20, near_cells 3) vs Γ_ref: {e(sl['op_rel'])} (operator), "
               f"{e(sl['action_rel_max_nonnull'])} (largest relative action difference over the non-null probes).")
    out.append(f"Checks over all {summ['n_valid']} valid (configuration, cutoffs, M) cases: dense Γ_h vs matrix-free action "
               f"≤ {e(summ['max_dense_vs_matrix_free'])}; symmetry defect ≤ {e(summ['max_sym_defect_h'])}; negative eigenvalue on "
               f"Range Π: {summ['any_negative']}; cached vs pair-difference ≤ {e(summ['max_cached_vs_pairdiff'])}.")
    return "\n".join(out)


def table_refs():
    r = json.loads((RES / "references.json").read_text())["rows"]
    out = ["| configuration | L | ||Γ_ref||_2 | Λ_lb | λ_min on Range Π | ||Γ_D55 − Γ_ref||_2/||Γ_ref||_2 | D55 tail bound (rel.) | "
           "||Γ_ref T||/λ_max |", "|---|---|---|---|---|---|---|---|"]
    for k in sorted(r):
        v = r[k]
        out.append(f"| {k} | {v['L']:.4f} | {v['norm2']:.4f} | {v['lam_lb']:.4f} | {v['lam_min_RangePi']:.4f} | "
                   f"{e(v['D55_vs_ref'])} | {e(v['D55_tail_bound_rel'], 1)} | {e(v['null'], 1)} |")
    return "\n".join(out)


if __name__ == "__main__":
    for title, fn in (("References", table_refs), ("(b) split identity", table_a), ("(c) short-range cutoff", table_b),
                      ("(d) mesh validity and retained modes", table_c_validity),
                      ("(d) N = 512 strongly clustered (optimisation case), every M", table_c_main),
                      ("(d) per family and N (all probe classes)", table_c_small), ("(d) fitted rates (strict cutoffs, complete error)", table_c_rates),
                      ("(d) translation, spectral probes and structural checks", table_c_checks)):
        print(f"### {title}\n")
        print(fn())
        print()
