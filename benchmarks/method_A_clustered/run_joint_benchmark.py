#!/usr/bin/env python3
"""
Joint cost-aware error-budget allocation for Method A (p = 7, tau = 2e-7): selection, timing and dense validation.
Outputs go to results_joint/ (the earlier fixed-split results in results/ are read, never overwritten).

Stages:
  calibrate  isolated timings -> cost model of this session (results_joint/cost_model.json); the VM was rebooted, so
             every timing compared below (baseline included) is measured again in this session
  select     approaches A (fixed 33 % split, original C(p) mesh bound), A_hz (fixed split, Hurwitz-zeta bound;
             diagnostic), B (joint, original bound), C (joint, Hurwitz-zeta bound): best (kc, M, rc) for every xi of the
             0.025 grid, then a 0.005 refinement around each approach's best xi; top-6 (kc, M) alternatives at xi_opt
  bench      measured cached action: baseline and the four optima for every configuration; for N = 256 also the
             per-xi optima of B and C at xi_opt +- 0.05, 0.1, 0.2 and their (kc, M) alternatives at xi_opt; coarse xi
             scans (0.1 steps) of A, B and C on strong_N256_s1 and uniform_N256_s1; N = 512 neighbours +- 0.1
  errors     E_op, structure and probe actions of every measured set against the dense full-periodic Gamma_ref;
             measured error components and basis-vector assembly at the B and C optima (N <= 256)

    python3 run_joint_benchmark.py --stage calibrate select bench errors
"""
import argparse
import csv
import time

import numpy as np

import common as C
import run_clustered_benchmark as rcb
from clustered_configurations import load, all_names
from cached_operator import baseline_mesh
from parameter_optimizer import RealSpaceTail, CostModel, A_MAX, R_PAD, XI_GRID, shell_kc, FourierShells, eps_mesh, \
    lambda_lower_bound
from joint_optimizer import evaluate_xi, finalize, RESULTS_JOINT, OLD_RESULTS, APPROACHES, VARIANT, TAU, mesh_bound_one
from error_validation import (reference_cached, reference_summary, probes, validate_parameter_set, Components)

PRIMARY = ("strong_N256_s1", "uniform_N256_s1")
NEIGH = (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2)
N_ALT = 3


def family_of(name):
    return name.split("_N")[0]


def N_of(name):
    return int(name.split("_N")[1].split("_")[0])


def names_for(args):
    return [n for n in all_names(Ns=tuple(args.N)) if N_of(n) in args.N]


def groups(names):
    g = {}
    for n in names:
        g.setdefault((family_of(n), N_of(n)), []).append(n)
    return g


def cost_model():
    return CostModel.from_dict(C.read_json(RESULTS_JOINT / "cost_model.json")["action"])


def refs():
    return C.read_json(OLD_RESULTS / "references.json")["configs"]


# ============================================================================ calibrate
def stage_calibrate(args):
    rcb.stage_calibrate(args)                    # C.RESULTS points to results_joint/


# ============================================================================ select
def stage_select(args):
    cm = cost_model()
    R = refs()
    out = C.read_json(RESULTS_JOINT / "joint_selection.json", {"groups": {}}) if args.resume else {"groups": {}}
    flat = []
    for (fam, N), names in sorted(groups(names_for(args)).items(), key=lambda kv: (kv[0][1], kv[0][0])):
        key = f"{fam}_N{N}"
        if key in out["groups"]:
            continue
        t0 = time.perf_counter()
        seeds = []
        for name in names:
            x, L = load(name)
            seeds.append(dict(name=name, rst=RealSpaceTail(x, L, (A_MAX + R_PAD) / XI_GRID.min()),
                              lam_lb=R[name]["lam_lb"], norm2=R[name]["norm2"]))
        cache = {}

        def ev(xi):
            xi = round(float(xi), 4)
            if xi not in cache:
                res, tab, curves = evaluate_xi(xi, seeds, L, N, cm)
                cache[xi] = res
            return cache[xi]

        for xi in XI_GRID:
            ev(xi)
        g = dict(family=fam, N=N, L=L, configs=names, approaches={})
        for ap in APPROACHES:
            coarse = {x_: r[ap]["best"] for x_, r in cache.items() if r.get(ap)}
            xb = min(coarse, key=lambda x_: coarse[x_]["T_pred"])
            for xi in np.round(np.arange(xb - 0.025, xb + 0.02501, 0.005), 4):
                if XI_GRID.min() <= xi <= XI_GRID.max():
                    ev(xi)
            allx = {x_: r[ap]["best"] for x_, r in cache.items() if r.get(ap)}
            xo = min(allx, key=lambda x_: allx[x_]["T_pred"])
            # (kc, M) alternatives at xi_opt, finalised with exact rc
            res, tab, curves = evaluate_xi(xo, seeds, L, N, cm)
            alts = []
            for t in res[ap]["top"][1:1 + N_ALT]:
                alts.append(finalize(xo, tab, curves, seeds, t["i"], t["j"], np.array(t["B"]), np.array(t["g"]),
                                     VARIANT[ap], cm, N, 7, TAU))
            g["approaches"][ap] = dict(opt=allx[xo], alternatives=alts,
                                       per_xi={f"{x_:.4f}": allx[x_] for x_ in sorted(allx)},
                                       n_candidates_at_opt=res[ap]["n_candidates"])
            for x_ in sorted(allx):
                r = dict(allx[x_])
                flat.append(dict(group=key, family=fam, N=N, approach=ap, is_opt=bool(x_ == xo),
                                 **{k: v for k, v in r.items() if not isinstance(v, list)},
                                 cert_rel_max_=r["cert_rel_max"]))
        out["groups"][key] = g
        o = {ap: g["approaches"][ap]["opt"] for ap in APPROACHES}
        C.log(f"select {key}: " + " | ".join(f"{ap} xi {o[ap]['xi']:.3f} kc {o[ap]['kc']:.3f} M {o[ap]['M']} rc "
                                             f"{o[ap]['rc']:.3f} T {1e3 * o[ap]['T_pred']:.2f}" for ap in APPROACHES)
              + f" ({time.perf_counter() - t0:.0f} s)")
        out.update(tau=TAU, p=7, provenance=C.provenance())
        C.write_json(RESULTS_JOINT / "joint_selection.json", out)
        del seeds
    C.write_json(RESULTS_JOINT / "joint_selection.json", out)
    if flat:
        C.write_csv(RESULTS_JOINT / "joint_selection_grid.csv", flat)


# ============================================================================ bench
def bench_plan(name, g):
    """List of (approach, role, record) to time for one configuration."""
    N = N_of(name)
    plan = [("baseline", "baseline", None)]
    for ap in APPROACHES:
        plan.append((ap, "opt", g["approaches"][ap]["opt"]))
    if N == 256 or (N == 512):
        for ap in ("B", "C"):
            per = g["approaches"][ap]["per_xi"]
            xo = g["approaches"][ap]["opt"]["xi"]
            grid = np.array([float(k) for k in per])
            offs = NEIGH if N == 256 else (-0.1, 0.1)
            for o in offs:
                k = grid[np.argmin(np.abs(grid - (xo + o)))]
                if abs(k - xo) > 1e-9:
                    plan.append((ap, f"xi_neighbour_{o:+.2f}", per[f"{k:.4f}"]))
            if N == 256:
                for t, alt in enumerate(g["approaches"][ap]["alternatives"]):
                    plan.append((ap, f"alternative_{t + 1}", alt))
    if name in PRIMARY:
        for ap in ("A", "B", "C"):
            per = g["approaches"][ap]["per_xi"]
            for xs in np.round(np.arange(0.4, 1.6001, 0.1), 4):
                if f"{xs:.4f}" in per:
                    plan.append((ap, "scan", per[f"{xs:.4f}"]))
    seen, out = set(), []
    for ap, role, rec in plan:
        k = (ap, None) if rec is None else (VARIANT.get(ap), round(rec["xi"], 4), round(rec["kc"], 9), rec["M"],
                                            round(rec["rc"], 9))
        if k in seen:
            continue
        seen.add(k)
        out.append((ap, role, rec))
    return out


def stage_bench(args):
    cm = cost_model()
    sel = C.read_json(RESULTS_JOINT / "joint_selection.json")["groups"]
    rows = C.read_json(RESULTS_JOINT / "bench.json", {"rows": []})["rows"] if args.resume else []
    done = {(r["config"], r["approach"], r["role"], round(r["xi"], 4)) for r in rows}
    for name in names_for(args):
        x, L = load(name)
        N = len(x)
        g = sel[f"{family_of(name)}_N{N}"]
        for ap, role, rec in bench_plan(name, g):
            xi = C.BASELINE["xi"] if rec is None else rec["xi"]
            if (name, ap, role, round(xi, 4)) in done:
                continue
            comps = role in ("opt", "baseline")
            if rec is None:
                b = C.BASELINE
                K = C.kernel(b["xi"])
                rc, kc = b["s"] / b["xi"], 2 * b["xi"] * b["s"]
                mesh = baseline_mesh(L, K, kc, b["eta"], b["p"])
                row, op = rcb.measure_point(x, L, b["xi"], rc, kc, mesh.M, b["p"], mesh=mesh, components=comps)
                cert = None
            else:
                kc = shell_kc(rec["kc"], L)
                row, op = rcb.measure_point(x, L, rec["xi"], rec["rc"], kc, rec["M"], rec["p"], components=comps)
                assert op.mesh.mc == int(np.floor(kc * L / (2 * np.pi) + 1e-12)) and \
                    np.isclose(np.max(np.linalg.norm(op.mesh.kv, axis=1)), kc, rtol=1e-12), "outer shell lost"
                cert = rec["cert_rel_max"]
            parts = cm.parts(row["n_img"], N, row["p"], row["M"])
            row.update(config=name, family=family_of(name), approach=ap, role=role, cert_rel_group=cert,
                       T_pred_s=float(sum(parts.values())), **{f"T_pred_{k}_s": float(v) for k, v in parts.items()})
            rows.append(row)
            C.log(f"bench {name} {ap:8s} {role:18s} xi={row['xi']:.3f} kc={row['kc']:.3f} M={row['M']} rc={row['rc']:.3f} "
                  f"n_img={row['n_img']}: {1e3 * row['t_action_s']:.2f} ms (pred {1e3 * row['T_pred_s']:.2f}, "
                  f"IQR {100 * row['t_action_iqr_rel']:.0f}%)")
            C.write_json(RESULTS_JOINT / "bench.json", dict(rows=rows, provenance=C.provenance()))
    C.write_json(RESULTS_JOINT / "bench.json", dict(rows=rows, provenance=C.provenance()))


# ============================================================================ errors
def stage_errors(args):
    bench = C.read_json(RESULTS_JOINT / "bench.json")["rows"]
    out = C.read_json(RESULTS_JOINT / "errors.json", {"rows": [], "probes": []}) if args.resume else \
        {"rows": [], "probes": []}
    done = {(r["config"], r["approach"], r["role"], round(r["xi"], 4)) for r in out["rows"]}
    for name in names_for(args):
        x, L = load(name)
        x = np.mod(x, L)
        N = len(x)
        G = reference_cached(name, x, L)
        rs, Q = reference_summary(G, N, C.lattice(L))
        lam = lambda_lower_bound(G, N)
        P = probes(x, L, N, G, Q, np.random.default_rng([20261017, 11, N, sum(map(ord, name))]))
        for r in [b for b in bench if b["config"] == name]:
            if (name, r["approach"], r["role"], round(r["xi"], 4)) in done:
                continue
            t0 = time.perf_counter()
            is_opt = r["role"] == "opt"
            basis = is_opt and r["approach"] in ("B", "C") and N <= 256
            kw = dict(xi=r["xi"], rc=r["rc"], kc=r["kc"], M=r["M"], p=r["p"])
            rec, act = validate_parameter_set(name, x, L, G, rs, Q, P, basis=basis, **kw)
            rec.update(config=name, family=family_of(name), N=N, approach=r["approach"], role=r["role"], **kw,
                       norm2_ref=rs["norm2"], lam_lb=lam, eps_target=TAU, passes=bool(rec["E_op"] <= TAU))
            if is_opt and r["approach"] in APPROACHES and N <= 256:
                cp = Components(x, L, G, r["xi"])
                rec.update(cp.measured(r["rc"], r["kc"], r["M"], r["p"]))
                fs = FourierShells(r["xi"], L)
                prep = cp.rst.prepare(r["xi"])
                rec.update(cert_S_abs=cp.rst.eps_S(prep, r["rc"]), cert_F_abs=fs.eps_F(r["kc"], N / L ** 3),
                           cert_mesh_orig_abs=mesh_bound_one(fs, r["kc"], r["M"], r["p"], N / L ** 3, "orig"),
                           cert_mesh_hz_abs=mesh_bound_one(fs, r["kc"], r["M"], r["p"], N / L ** 3, "hz"),
                           cert_mesh_check_existing=eps_mesh(fs, r["kc"], r["M"], r["p"], N / L ** 3))
            out["rows"].append(rec)
            out["probes"] += [dict(config=name, approach=r["approach"], role=r["role"], xi=r["xi"], probe=k, **v)
                              for k, v in act.items()]
            C.log(f"errors {name} {r['approach']:8s} {r['role']:18s} xi={r['xi']:.3f} M={r['M']}: E_op={rec['E_op']:.3e} "
                  f"lam_min {rec['lam_min_h']:.4f}/{rec['lam_min_ref']:.4f}"
                  + (f" basis {rec['basis_vs_independent']:.1e}" if basis else "")
                  + (f" meas R/F/M {rec['meas_real_rel']:.2e}/{rec['meas_fourier_rel']:.2e}/{rec['meas_mesh_rel']:.2e}"
                     if "meas_real_rel" in rec else "") + f" ({time.perf_counter() - t0:.1f} s)")
            C.write_json(RESULTS_JOINT / "errors.json", dict(out, provenance=C.provenance()))
    C.write_json(RESULTS_JOINT / "errors.json", dict(out, provenance=C.provenance()))


STAGES = dict(calibrate=stage_calibrate, select=stage_select, bench=stage_bench, errors=stage_errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", required=True, choices=list(STAGES))
    ap.add_argument("--N", type=int, nargs="+", default=[64, 256, 512])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    C.RESULTS = RESULTS_JOINT                      # every output of this round goes to results_joint/
    RESULTS_JOINT.mkdir(parents=True, exist_ok=True)
    C.log_command()
    for st in args.stage:
        t0 = time.perf_counter()
        STAGES[st](args)
        C.log(f"stage {st} done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
