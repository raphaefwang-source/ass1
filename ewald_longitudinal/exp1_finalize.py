#!/usr/bin/env python3
"""
Experiment 1 close-out: output integrity and the measured scope of each spatial parameter set.

Writes
  exp1_results/integrity.json             row counts against the designed stages, NaN/Inf scan, log exit status,
                                          absence of |k| rounding in khat_AB, post-fix Fourier-sum precision floor
  exp1_results/parameter_set_status.json  per set (T5/T7/T9): configurations tested, pass/fail against the nominal
  exp1_results/parameter_set_status.csv   relative operator tolerance, worst errors (operator and probe actions),
                                          every failing configuration

Reads only exp1_results/*.csv and the source of test_parameter_selection.py; computes nothing new.
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

import sec6_config

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp1_results"


def rd(name):
    with open(OUT / name) as fh:
        return list(csv.DictReader(fh))


def f(x):
    return float(x)


def integrity():
    cfg = rd("configs.csv")
    n_cfg = len(cfg)
    sets = {s: sum(r["set"] == s for r in cfg) for s in sorted({r["set"] for r in cfg})}
    n_lm = sets.get("legacy", 0) + sets.get("main", 0)
    n_lmc = n_lm + sets.get("check", 0)
    n_box = sets.get("box_fixedN", 0) + sets.get("box_fixedn", 0)
    probes_per_config = 16 + 18 + 3
    expected = {
        "configs.csv": n_cfg,
        "reference_ladder.csv": 7 * n_cfg,
        "split_operator.csv": 4 * n_lm,
        "cutoffs.csv": n_lmc * 2 * (25 + 26),
        "mesh.csv": None,
        "mesh_decomposition.csv": 5 * 2 * 2,
        "action.csv": 3 * n_lmc + n_box,
        "action_probes.csv": (3 * n_lmc + n_box) * probes_per_config,
        "fourier_precision.csv": 6,
    }
    files = {}
    for fn, exp in expected.items():
        rows = rd(fn)
        bad = sum(v.strip().lower() in ("nan", "inf", "-inf") for r in rows for v in r.values() if v is not None)
        files[fn] = dict(rows=len(rows), expected=exp, ok=(exp is None or len(rows) == exp), nan_inf_cells=bad)
    mesh = rd("mesh.csv")
    files["mesh.csv"]["note"] = "designed size depends on the admissible meshes per box; checked per scan below"
    scans = {}
    for r in mesh:
        k = (r["set"], r["s_F"][:4], r["p"])
        scans[k] = scans.get(k, 0) + 1
    files["mesh.csv"]["rows_per_(set,s_F,p)"] = {"|".join(k): v for k, v in sorted(scans.items())}
    log = (OUT / "run_2026-10-13_exact_k.log").read_text()
    src = (HERE / "test_parameter_selection.py").read_text()
    m = re.search(r"def khat_AB\(K, kn\):(.*?)\n\n", src, re.S)
    body = m.group(1) if m else ""
    prec = rd("fourier_precision.csv")
    keys = ("floor_rounded12", "floor_float64", "floor_int_shell", "floor_refined_coeffs", "floor_longdouble",
            "float64_vs_longdouble", "GL_images_R40_vs_R45")
    out = dict(
        provenance=sec6_config.provenance("exp1_finalize.py"),
        files=files,
        all_row_counts_ok=all(v["ok"] for v in files.values()),
        nan_inf_cells_total=sum(v["nan_inf_cells"] for v in files.values()),
        rerun_log=dict(file="exp1_results/run_2026-10-13_exact_k.log", exit_zero="exit 0" in log,
                       stages_done=re.findall(r"stage (\w+) done", log),
                       error_lines=[ln for ln in log.splitlines() if re.search(r"Traceback|Error|FAILED", ln)]),
        khat_AB=dict(source_has_round=bool(re.search(r"np\.round\(\s*kn", body)),
                     dedup_line=[ln.strip() for ln in body.splitlines() if "np.unique" in ln]),
        fourier_precision_after_fix={k: [min(f(r[k]) for r in prec), max(f(r[k]) for r in prec)] for k in keys},
        fourier_precision_note="Mesh-free Fourier sum against the real-space image sum Gamma_L (relative to "
                               "||Gamma_ref||_2), case A, lattice_N128_s1, uniform_N128_s1 at xi = 0.7 and 1. "
                               "floor_rounded12 reproduces the removed rounded-|k| routine; the former ~1e-13 "
                               "plateau is NOT a float64 summation floor.",
    )
    (OUT / "integrity.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def parameter_set_status():
    S = json.loads((OUT / "summary.json").read_text())
    act = rd("action.csv")
    probes = rd("action_probes.csv")
    ref = {r["name"]: f(r["norm2"]) for r in rd("reference_ladder.csv") if r.get("norm2")}
    spatial = sec6_config.load()["spatial_sets"]
    status, rows = {}, []
    for pn, sp in spatial.items():
        tol = sp["nominal_tol"]
        g = [r for r in act if r["params"] == pn]
        fails = [r for r in g if f(r["err_spec_rel"]) > tol]
        pmax = {}
        for r in probes:
            if r["params"] == pn and r["probe"] in ("gauss", "long", "trans"):
                pmax[(r["name"], r["probe"])] = max(pmax.get((r["name"], r["probe"]), 0.0), f(r["rel_err"]))
        by_set = {}
        for st in sorted({r["set"] for r in g}):
            gs = [r for r in g if r["set"] == st]
            by_set[st] = dict(n=len(gs), n_fail=sum(f(r["err_spec_rel"]) > tol for r in gs),
                              err_rel_max=max(f(r["err_spec_rel"]) for r in gs),
                              err_abs_max=max(f(r["err_spec_rel"]) * ref[r["name"]] for r in gs))
        by_family = {}
        for fam in sorted({r["family"] for r in g}):
            gs = [r for r in g if r["family"] == fam]
            by_family[fam] = dict(n=len(gs), n_fail=sum(f(r["err_spec_rel"]) > tol for r in gs),
                                  err_rel_range=[min(f(r["err_spec_rel"]) for r in gs),
                                                 max(f(r["err_spec_rel"]) for r in gs)])
        ok_names = [r["name"] for r in g if f(r["err_spec_rel"]) <= tol]
        status[pn] = dict(
            spatial=dict(xi=sp["xi"], s=sp["s"], eta_N=sp["eta"], p=sp["p"]), nominal_tol=tol,
            accepted_on=["A", "B"], accepted_note="accepted on cases A and B only; not a guarantee elsewhere",
            n_tested=len(g), n_pass=len(g) - len(fails), n_fail=len(fails),
            err_rel_max=max(f(r["err_spec_rel"]) for r in g), err_rel_min=min(f(r["err_spec_rel"]) for r in g),
            err_abs_range=[min(f(r["err_spec_rel"]) * ref[r["name"]] for r in g),
                           max(f(r["err_spec_rel"]) * ref[r["name"]] for r in g)],
            failures=[dict(name=r["name"], set=r["set"], family=r["family"], N=int(r["N"]), L=f(r["L"]),
                           err_rel=f(r["err_spec_rel"]), ratio_to_tol=f(r["err_spec_rel"]) / tol,
                           norm2_ref=ref[r["name"]]) for r in sorted(fails, key=lambda r: -f(r["err_spec_rel"]))],
            by_set=by_set, by_family=by_family,
            probe_rel_action_max={k: max(v for (n, kk), v in pmax.items() if kk == k) for k in ("gauss", "long", "trans")},
            probe_rel_action_max_where_operator_ok={k: max([v for (n, kk), v in pmax.items() if kk == k and n in ok_names],
                                                           default=None) for k in ("gauss", "long", "trans")},
            n_operator_ok_with_trans_probe_over_tol=sum(pmax.get((n, "trans"), 0) > tol for n in ok_names),
            scope=("relative operator error <= nominal tolerance on %d of %d tested static configurations "
                   "(rho = 30/216 unless the set is legacy B or box_fixedN); fails on: %s"
                   % (len(g) - len(fails), len(g), ", ".join(r["name"] for r in fails) or "none")),
        )
        for r in g:
            rows.append(dict(set_name=pn, nominal_tol=tol, config=r["name"], config_set=r["set"], family=r["family"],
                             N=r["N"], L=r["L"], norm2_ref=ref[r["name"]], err_rel=r["err_spec_rel"],
                             err_abs=f(r["err_spec_rel"]) * ref[r["name"]],
                             status="pass" if f(r["err_spec_rel"]) <= tol else "fail",
                             gauss_rel_max=pmax.get((r["name"], "gauss")), long_rel_max=pmax.get((r["name"], "long")),
                             trans_rel_max=pmax.get((r["name"], "trans"))))
    out = dict(provenance=sec6_config.provenance("exp1_finalize.py"), source="exp1_results/action.csv, "
               "action_probes.csv, reference_ladder.csv (rerun with unrounded |k|, run_2026-10-13_exact_k.log)",
               measure="relative operator error ||Gamma_h - Gamma_ref||_2 / ||Gamma_ref||_2 against the nominal "
                       "tolerance; probe columns are relative action errors ||(Gamma_h-Gamma_ref)X||/||Gamma_ref X||",
               production_set=sec6_config.load()["production"]["spatial_set"], sets=status)
    (OUT / "parameter_set_status.json").write_text(json.dumps(out, indent=1) + "\n")
    with open(OUT / "parameter_set_status.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out


if __name__ == "__main__":
    a = integrity()
    print(json.dumps({k: a[k] for k in ("all_row_counts_ok", "nan_inf_cells_total", "rerun_log", "khat_AB",
                                         "fourier_precision_after_fix")}, indent=1))
    b = parameter_set_status()
    for pn, v in b["sets"].items():
        print(pn, v["n_pass"], "/", v["n_tested"], "max %.3e" % v["err_rel_max"], [x["name"] for x in v["failures"]])
