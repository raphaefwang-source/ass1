#!/usr/bin/env python3
"""Markdown tables for report_exp1.md, generated from results/ (python3 report_tables_exp1.py > results/tables_exp1.md)."""
import json

import numpy as np

from plot_exp1 import load, RES, FAMS


def e(x, d=2):
    return "–" if x is None else f"{x:.{d}e}"


def table_a():
    rows = load("a_split_configs.csv")
    radial = load("a_split_radial.csv")
    mp_rows = load("a_split_mpmath.csv")
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
    fits = load("b_shortrange_fits.csv")
    rows = load("b_shortrange.csv")
    summ = json.loads((RES / "b_shortrange_summary.json").read_text())
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


def table_c():
    rows = load("c_degree.csv")
    fits = load("c_degree_rate_fits.csv")
    out = ["| configuration | cutoffs | p | valid M | E_D complete (min M → max M) | E_D mesh (min M → max M) | fixed-cutoff floor | "
           "LSQ mesh rate (M range) | first M with complete ≤ 1.1 floor |",
           "|---|---|---|---|---|---|---|---|---|"]
    for f in fits:
        mine = [r for r in rows if r["config"] == f["config"] and r["p"] == f["p"] and r["setting"] == f["setting"] and r["valid"]]
        fl = mine[0]["E_D_floor"]
        reach = [r["M"] for r in mine if r["E_D_complete"] <= 1.1 * fl]
        out.append(f"| {f['config']} | {f['setting']} | {f['p']} | {mine[0]['M']}–{mine[-1]['M']} | {e(mine[0]['E_D_complete'])} → "
                   f"{e(mine[-1]['E_D_complete'])} | {e(mine[0]['E_D_mesh'])} → {e(mine[-1]['E_D_mesh'])} | {e(fl)} | "
                   f"{f['rate_mesh_lsq']:.2f} ({f['M_lo']}–{f['M_hi']}) | {reach[0] if reach else '—'} |")
    return "\n".join(out)


def table_d():
    rows = [r for r in load("d_action.csv") if r["valid"]]
    out = ["| N | cutoffs | M | E_op (dense) | Gaussian (max of 5) | spectral min | spectral max | slow (max of 4) | "
           "λ_min(Range Π) rel. dev. |",
           "|---|---|---|---|---|---|---|---|---|"]
    for N in (128, 256):
        for st in ("production", "strict"):
            Ms = sorted({r["M"] for r in rows if r["N"] == N and r["setting"] == st})
            for M in (Ms[0], Ms[len(Ms) // 2], Ms[-1]):
                s = [r for r in rows if r["N"] == N and r["setting"] == st and r["M"] == M]
                f = lambda k: max(r[k] for r in s)          # noqa: E731
                dl = max(abs(r["lam_min_h_RangePi"] - r["lam_min_ref_RangePi"]) / r["lam_min_ref_RangePi"] for r in s)
                out.append(f"| {N} | {st} | {M} | {e(f('E_op'))} | {e(f('E_gauss_max'))} | {e(f('E_spec_min'))} | "
                           f"{e(f('E_spec_max'))} | {e(f('E_slow_max'))} | {e(dl)} |")
    out.append("")
    out.append("Maxima over the 9 configurations (3 families × 3 seeds) at each (N, cutoffs, M).")
    return "\n".join(out)


def table_d_rates():
    fits = load("d_action_rate_fits.csv")
    out = ["| N | family | quantity | M range | LSQ rate |", "|---|---|---|---|---|"]
    for f in fits:
        out.append(f"| {f['N']} | {f['family']} | {f['quantity']} | {f['M_lo']}–{f['M_hi']} | {f['rate_lsq']:.2f} |")
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
    for title, fn in (("References", table_refs), ("(a) split identity", table_a), ("(b) short-range", table_b),
                      ("(c) degree", table_c), ("(d) action", table_d), ("(d) action rates (strict)", table_d_rates)):
        print(f"### {title}\n")
        print(fn())
        print()
