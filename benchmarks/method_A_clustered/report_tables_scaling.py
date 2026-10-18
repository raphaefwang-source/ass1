#!/usr/bin/env python3
"""
Markdown tables and CSV summaries of the scaling benchmark (run_scaling.py), generated from the result files.
Writes results_scaling/report_tables.md, results_scaling/scaling_summary.csv, results_scaling/empirical_exponents.json
and results_scaling/reproducibility_scaling.json.

Empirical exponents are FINITE-RANGE fits of log T vs log N over the measured N (local slopes between consecutive N,
and least squares over all N and over N >= 512); they are not asymptotic statements.

    python3 report_tables_scaling.py
"""
import common as C
import numpy as np

from run_scaling import RESULTS, NS, FAMS, name_of, path, QS, TAU
from direct_periodic import shifts_for

MiB = 2 ** 20


def f(v, fmt):
    return "–" if v is None else format(v, fmt)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def load(fam, N):
    name = name_of(fam, N)
    if not path("scale", name).exists():
        return None
    d = dict(name=name, scale=C.read_json(path("scale", name)), direct=C.read_json(path("direct", name)),
             verify=C.read_json(path("verify", name)), sel=C.read_json(path("selection", name)),
             ref=C.read_json(path("reference", name)), conf=C.read_json(path("confirm", name)))
    d["mem"] = C.read_json(path("memory", name)) if path("memory", name).exists() else None
    return d


def fit(Ns, T):
    x, y = np.log(np.asarray(Ns, float)), np.log(np.asarray(T, float))
    if len(x) < 2:
        return None
    A = np.stack([x, np.ones_like(x)], 1)
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0])


def exponents(rows):
    out = {}
    for fam in FAMS:
        rs = [r for r in rows if r["fam"] == fam]
        if len(rs) < 2:
            continue
        Ns = [r["N"] for r in rs]
        for k in ("T_action_direct_ms", "T_action_pppm_ms", "T_setup_direct_ms", "T_setup_pppm_ms"):
            T = [r[k] for r in rs]
            local = [float(np.log(T[i + 1] / T[i]) / np.log(Ns[i + 1] / Ns[i])) for i in range(len(rs) - 1)]
            big = [(n, t) for n, t in zip(Ns, T) if n >= 512]
            out[f"{fam}:{k}"] = dict(N=Ns, local_slopes=local, lsq_all=fit(Ns, T),
                                     lsq_N_ge_512=fit(*zip(*big)) if len(big) >= 2 else None)
    return out


def main():
    md = ["# Generated tables (report_tables_scaling.py)\n"]
    rows = []
    for fam in FAMS:
        for N in NS:
            d = load(fam, N)
            if d is None:
                continue
            s, dr, sm = d["scale"], d["direct"], d["scale"]["summary"]
            pp = s["pppm"]
            v = d["verify"]["rows"][d["verify"]["optimum"]["key"]]
            cc = d["conf"]["candidates"][d["verify"]["optimum"]["key"]]
            n_shifts = int(len(shifts_for(s["L"], dr["R_d"])))
            row = dict(fam=fam, N=N, L=s["L"], p=pp["p"], xi=pp["xi"], rc=pp["rc"], kc=pp["kc"], M=pp["M"], eta=pp["eta"],
                       M3_over_N=pp["M"] ** 3 / N, real_interactions_per_particle=pp["n_img"] / N, impl=pp["impl"],
                       pppm_cert_rel_lam=pp["cert_rel"], pppm_E_op=pp["E_op"], pppm_E_op_method=v["E_op_method"],
                       pppm_E_op_arpack=v.get("E_op_arpack"), pppm_E_op_dense=v.get("E_op_dense"),
                       R_d=dr["R_d"], direct_bound_rel_lam=dr["bound_rel_lam"], direct_E_op=dr["E_ref"],
                       direct_E_op_method=dr["E_ref_method"], direct_E_conv=dr.get("E_conv_dense", dr.get("E_conv_arpack")),
                       direct_refcheck=dr.get("E_refcheck_dense", dr.get("E_refcheck_arpack")),
                       direct_images=dr["n_images"], direct_images_per_particle=dr["n_images"] / N,
                       pairs=dr["n_pairs"], norm2_ref=d["ref"]["norm2"], lam_lb=d["ref"]["lam_lb"],
                       direct_n_shifts=n_shifts, direct_setup_evaluations=n_shifts * dr["n_pairs"],
                       share_S=cc["share_S"], share_F=cc["share_F"], share_mesh=cc["share_mesh"],
                       share_quad=cc["share_quad"], h=s["L"] / pp["M"],
                       T_setup_direct_ms=sm["T_setup_direct_ms"], T_action_direct_ms=sm["T_action_direct_ms"],
                       T_setup_pppm_ms=sm["T_setup_pppm_ms"], T_action_pppm_ms=sm["T_action_pppm_ms"],
                       action_ratio=sm["action_ratio_direct_over_pppm"], setup_ratio=sm["setup_ratio_direct_over_pppm"],
                       break_even_q=sm["break_even_q"],
                       **{f"total_direct_q{q}_ms": sm["total_direct_ms"][str(q)] for q in QS},
                       **{f"total_pppm_q{q}_ms": sm["total_pppm_ms"][str(q)] for q in QS},
                       **{f"total_ratio_q{q}": sm["total_ratio_direct_over_pppm"][str(q)] for q in QS},
                       once_direct_geometry_ms=s["once_direct_geometry"]["median_ms"],
                       once_pppm_mesh_ms=s["once_pppm_mesh"]["median_ms"],
                       rounds_setup=s["rounds_setup"], rounds_action=s["rounds_action"],
                       err10X_direct=s["action_err_opnorm_10X"]["direct"],
                       err10X_pppm=s["action_err_opnorm_10X"][f"pppm_{pp['impl']}"])
            a = s["actions"]
            for k in ("direct", "pppm_cached", "pppm_pairdiff"):
                row.update({f"{k}_action_q1_ms": a[k]["q1_ms"], f"{k}_action_q3_ms": a[k]["q3_ms"],
                            f"{k}_action_median_ms": a[k]["median_ms"],
                            f"{k}_setup_median_ms": s["setups"][k]["median_ms"],
                            f"{k}_setup_q1_ms": s["setups"][k]["q1_ms"], f"{k}_setup_q3_ms": s["setups"][k]["q3_ms"]})
            if d["mem"]:
                row.update({f"mem_{k}_MiB": v_ / MiB for k, v_ in d["mem"].items() if isinstance(v_, int) and k != "N"})
            rows.append(row)
    C.write_csv(RESULTS / "scaling_summary.csv", rows)
    ex = exponents(rows)
    C.write_json(RESULTS / "empirical_exponents.json", dict(exponents=ex, note="finite-range log-log fits of medians"))

    for fam in FAMS:
        rs = [r for r in rows if r["fam"] == fam]
        if not rs:
            continue
        md.append(f"## {fam}: PPPM optimum per N (measured-fastest, certified, verified)\n")
        md.append(table(["N", "L", "p", "ξ", "r_c", "k_c", "M", "h = L/M", "η", "M³/N", "real-space interactions / particle",
                         "action", "certified budget shares S / F / mesh", "cert bound / Λ_lb", "measured E_op (method)"],
                        [[str(r["N"]), f(r["L"], ".3f"), str(r["p"]), f(r["xi"], ".3f"), f(r["rc"], ".4f"),
                          f(r["kc"], ".4f"), str(r["M"]), f(r["h"], ".4f"), f(r["eta"], ".3f"), f(r["M3_over_N"], ".1f"),
                          f(r["real_interactions_per_particle"], ".1f"), r["impl"],
                          f"{r['share_S']:.2f} / {r['share_F']:.1e} / {r['share_mesh']:.2f}", f(r["pppm_cert_rel_lam"], ".4e"),
                          f"{r['pppm_E_op']:.3e} ({'dense' if r['pppm_E_op_method'].startswith('dense') else 'ARPACK'})"]
                         for r in rs]))
        md.append(f"\n## {fam}: Direct cutoff and accuracy per N\n")
        md.append(table(["N", "R_d (certified)", "R_d / L", "bound / Λ_lb", "measured E_op vs Γ_ref (method)",
                         "‖Γ_D(36) − Γ_D(R_d)‖/‖Γ_ref‖", "Γ_D(36) vs Γ_ref", "images / particle", "pairs",
                         "lattice shifts", "setup evaluations (pairs × shifts)"],
                        [[str(r["N"]), f(r["R_d"], ".3f"), f(r["R_d"] / r["L"], ".2f"), f(r["direct_bound_rel_lam"], ".4e"),
                          f"{r['direct_E_op']:.3e} ({r['direct_E_op_method']})", f(r["direct_E_conv"], ".3e"),
                          f(r["direct_refcheck"], ".1e"), f(r["direct_images_per_particle"], ".0f"), str(r["pairs"]),
                          str(r["direct_n_shifts"]), f"{r['direct_setup_evaluations']:.3e}"]
                         for r in rs]))
        md.append(f"\n## {fam}: setup and single action (medians [q1, q3], ms; interleaved)\n")
        md.append(table(["N", "Direct setup", "PPPM setup", "Direct action", "PPPM action", "action Direct/PPPM",
                         "setup Direct/PPPM", "rounds setup / action"],
                        [[str(r["N"]),
                          f"{r['direct_setup_median_ms']:.1f} [{r['direct_setup_q1_ms']:.1f}, {r['direct_setup_q3_ms']:.1f}]",
                          f"{r['T_setup_pppm_ms']:.1f} [{r['pppm_' + r['impl'] + '_setup_q1_ms']:.1f}, {r['pppm_' + r['impl'] + '_setup_q3_ms']:.1f}]",
                          f"{r['direct_action_median_ms']:.3f} [{r['direct_action_q1_ms']:.3f}, {r['direct_action_q3_ms']:.3f}]",
                          f"{r['T_action_pppm_ms']:.3f} [{r['pppm_' + r['impl'] + '_action_q1_ms']:.3f}, {r['pppm_' + r['impl'] + '_action_q3_ms']:.3f}]",
                          f(r["action_ratio"], ".3f"), f(r["setup_ratio"], ".1f"), f"{r['rounds_setup']} / {r['rounds_action']}"]
                         for r in rs]))
        md.append(f"\n## {fam}: T_total(q) = T_setup + q T_action [ms] and ratio Direct/PPPM (> 1: PPPM faster)\n")
        md.append(table(["N"] + [f"q = {q}: Direct / PPPM (ratio)" for q in QS] + ["break-even q"],
                        [[str(r["N"])] + [f"{r[f'total_direct_q{q}_ms']:.1f} / {r[f'total_pppm_q{q}_ms']:.1f} "
                                          f"({r[f'total_ratio_q{q}']:.2f})" for q in QS] + [f(r["break_even_q"], ".1f")]
                         for r in rs]))
        if any("mem_direct_persistent_MiB" in r for r in rs):
            md.append(f"\n## {fam}: memory [MiB] (tracemalloc peaks above baseline; persistent = array nbytes)\n")
            md.append(table(["N", "Direct persistent", "Direct setup peak", "Direct action peak", "Direct one-time geometry",
                             "PPPM persistent (pairdiff / cached)", "PPPM mesh (one-time)", "PPPM setup peak",
                             "PPPM action peak"],
                            [[str(r["N"]), f(r.get("mem_direct_persistent_MiB"), ".1f"), f(r.get("mem_direct_setup_peak_MiB"), ".1f"),
                              f(r.get("mem_direct_action_peak_MiB"), ".1f"), f(r.get("mem_direct_geometry_bytes_MiB"), ".1f"),
                              f"{f(r.get('mem_pppm_pairdiff_persistent_MiB'), '.1f')} / {f(r.get('mem_pppm_cached_persistent_MiB'), '.1f')}",
                              f(r.get("mem_pppm_mesh_bytes_MiB"), ".1f"), f(r.get(f"mem_pppm_{r['impl']}_setup_peak_MiB"), ".1f"),
                              f(r.get(f"mem_pppm_{r['impl']}_action_peak_MiB"), ".1f")] for r in rs if "mem_direct_persistent_MiB" in r]))
        md.append("\n")
    md.append("## Finite-range empirical exponents (log-log slopes of the medians)\n")
    md.append(table(["series", "local slopes between consecutive N", "LSQ all N", "LSQ N ≥ 512"],
                    [[k, ", ".join(f"{s:.2f}" for s in v["local_slopes"]), f(v["lsq_all"], ".2f"), f(v["lsq_N_ge_512"], ".2f")]
                     for k, v in ex.items()]))
    (RESULTS / "report_tables.md").write_text("\n".join(md) + "\n")
    prov = {}
    for fam in FAMS:
        for N in NS:
            name = name_of(fam, N)
            for kind in ("selection", "confirm", "verify", "direct", "scale", "memory"):
                if path(kind, name).exists():
                    prov[f"{kind}:{name}"] = C.read_json(path(kind, name)).get("provenance")
        for N in NS:
            if path("cost_model", f"N{N}").exists():
                prov[f"cost_model:N{N}"] = C.read_json(path("cost_model", f"N{N}"))["provenance"]
    cmds = (RESULTS / "commands.log").read_text().splitlines() if (RESULTS / "commands.log").exists() else []
    C.write_json(RESULTS / "reproducibility_scaling.json",
                 dict(seeds=dict(configurations="default_rng([20261017, N, 1, attempt]) (clustered_configurations)",
                                 x_inputs="default_rng([20261018, 512, 7]): 10 standard-normal (N, 3) arrays (run_psearch_dense512.x_inputs)",
                                 interleaving="default_rng([20261018, 3, s])", arpack_v0="default_rng([20261018, 31, n, k])"),
                      commands=cmds, provenance_by_stage=prov, now=C.provenance()))
    C.log(f"wrote {RESULTS / 'report_tables.md'} ({len(rows)} configurations)")


if __name__ == "__main__":
    main()
