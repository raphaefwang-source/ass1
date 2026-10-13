#!/usr/bin/env python3
"""
Production rank selection from the Lanczos study (reads sec6_results/lanczos/raw/{small,large}_cases.json written by
sec6_lanczos.py; computes nothing new).

Rule (sec6_config.json budgets.rank_rule): the production rank is the smallest r such that the sample-maximum relative
action error over every tested case with an accepted reference stays <= budget / rank_margin_factor for all computed
ranks >= r, both at the production dt and at the fixed-tau diagnostic tau = rank_robustness_tau. Minimal passing
ranks (no margin) are reported alongside. Large-N references are accepted for a tolerance only if their non-adjacent
self-differences (S-20, S-40) are <= reference_margin * tolerance; otherwise the case is UNRESOLVED. Errors are
sample maxima over the tested cases, not bounds.

    python3 sec6_rank_select.py [--write-config]
"""
import argparse
import json
import time

import sec6_config
from sec6_lanczos import RAW, OUT, required_rank

SCRIPT = "sec6_rank_select.py"


def per_case(rec, scale, key, tol, margin):
    """(case id, N, required rank or None, reference status) for every case."""
    rows = []
    for c in rec["cases"]:
        if scale == "small":
            rows.append((f"{c['name']}|v{c['vec_index']}", c["N"], required_rank(c["err"][key], tol, c["S"]), "dense"))
        else:
            chk = c["ref_check"][key]
            ok = max(chk["S-20"], chk["S-40"]) <= margin * tol
            r = required_rank(c["err"][key], tol, c["S"] - 40) if ok else None
            rows.append((f"N{c['N']}|pos{c['pos_index']}|v{c['vec_index']}", c["N"], r,
                         "accepted" if ok else "UNRESOLVED"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-config", action="store_true")
    args = ap.parse_args()
    B = sec6_config.budgets()
    margin, fac, tau_r = B["reference_margin"], B["rank_margin_factor"], B["rank_robustness_tau"]
    small = json.loads((RAW / "small_cases.json").read_text())
    large = json.loads((RAW / "large_cases.json").read_text())
    status = {}
    for name, rec in (("small", small), ("large", large)):
        ok, why = sec6_config.reusable(rec, "sec6_lanczos.py")
        status[name] = dict(current=ok, reason=why, n_cases=len(rec["cases"]))
        if not ok:
            raise SystemExit(f"{name} cases are not current ({why}); rerun sec6_lanczos.py")
    budgets = [("noise", "prod", B["noise_rel_action"]), ("damp", "prod", B["damping_rel_action"])]
    budgets += [("noise", "prod", t) for t in B["noise_high_accuracy"]]
    budgets += [("noise", f"tau{tau_r:g}", B["noise_rel_action"]), ("damp", f"tau{tau_r:g}", B["damping_rel_action"])]
    out = dict(provenance=sec6_config.provenance(SCRIPT), inputs=status, rule=B["rank_rule"],
               margin_factor=fac, robustness_tau=tau_r, budgets={})
    for fname, dlab, tol in budgets:
        key = f"{fname}|{dlab}"
        for lab, t in (("minimal", tol), ("with_margin", tol / fac)):
            rows = per_case(small, "small", key, t, margin) + per_case(large, "large", key, t, margin)
            unresolved = [r[0] for r in rows if r[3] == "UNRESOLVED"]
            not_reached = [r[0] for r in rows if r[2] is None and r[3] != "UNRESOLVED"]
            by_N = {}
            for cid, N, r, st in rows:
                by_N.setdefault(N, []).append(r)
            out["budgets"][f"{key}|{tol:g}|{lab}"] = dict(
                function=fname, dt=dlab, tol=tol, threshold=t, n_cases=len(rows), unresolved=unresolved,
                not_reached=not_reached,
                max_required=(max(r[2] for r in rows if r[2] is not None) if not unresolved and not not_reached else None),
                max_required_by_N={str(N): (max(v) if all(x is not None for x in v) else None)
                                   for N, v in sorted(by_N.items())},
                per_case={r[0]: r[2] for r in rows})
    choice = {}
    for fname, tol in (("noise", B["noise_rel_action"]), ("damp", B["damping_rel_action"])):
        a = out["budgets"][f"{fname}|prod|{tol:g}|with_margin"]["max_required"]
        b = out["budgets"][f"{fname}|tau{tau_r:g}|{tol:g}|with_margin"]["max_required"]
        choice[fname] = dict(minimal_at_prod_dt=out["budgets"][f"{fname}|prod|{tol:g}|minimal"]["max_required"],
                             with_margin_at_prod_dt=a, with_margin_at_tau=b,
                             chosen=max(a, b) if a is not None and b is not None else None)
    out["production"] = dict(noise_rank=choice["noise"]["chosen"], damping_rank=choice["damp"]["chosen"], detail=choice)
    # sample maxima at the chosen ranks, per N, production dt and tau diagnostic
    at = {}
    for fname in ("noise", "damp"):
        r = choice[fname]["chosen"]
        for dlab in ("prod", f"tau{tau_r:g}"):
            key = f"{fname}|{dlab}"
            for scale, rec in (("small", small), ("large", large)):
                for c in rec["cases"]:
                    if r is not None and r <= (c["S"] if scale == "small" else c["S"] - 40):
                        k2 = f"{key}|N{c['N']}"
                        at[k2] = max(at.get(k2, 0.0), c["err"][key][r - 1])
    out["sample_max_at_chosen"] = at
    out["high_accuracy_noise_ranks"] = {f"{t:g}": out["budgets"][f"noise|prod|{t:g}|minimal"]["max_required_by_N"]
                                        for t in B["noise_high_accuracy"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rank_selection.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["production"], indent=1))
    for k, v in out["budgets"].items():
        print(k, v["max_required"], v["max_required_by_N"], "unresolved", len(v["unresolved"]), "not reached",
              len(v["not_reached"]))
    if args.write_config:
        p = out["production"]
        if p["noise_rank"] is None or p["damping_rank"] is None:
            raise SystemExit("not writing ranks: unresolved or not reached cases remain")
        sec6_config.set_ranks(p["noise_rank"], p["damping_rank"],
                              f"sec6_results/lanczos/rank_selection.json ({time.strftime('%Y-%m-%dT%H:%M:%S')}, "
                              f"rule budgets.rank_rule)")
        print("wrote production ranks to sec6_config.json")


if __name__ == "__main__":
    main()
