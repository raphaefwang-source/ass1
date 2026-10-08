#!/usr/bin/env python3
"""Acceptance checks of runs written by toy_run.py (pilot or production).

    python3 hpc/check_run.py RUN_DIR [RUN_DIR ...] [--expect-config costopt_hiacc] [--json out.json]

Exit code 1 if any check fails. The temperature check is a sanity bound, not an equilibration test.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import toy_configs as tc  # noqa: E402
import toy_run as tr  # noqa: E402


def check(d, expect_config=None, allow_incomplete=False):
    R = tr.load_run(d)
    cfg, st = R["config"], R["status"]
    r, plan = cfg["resolved"], cfg["plan"]
    D, M = R["diag"], R["monitor"]
    col = {c: i for i, c in enumerate(R["cols"])}
    total = plan["burn_steps"] + plan["prod_steps"]
    fresh = tc.resolve(r["config"], r["law"], r["N"], r["potential"], dt=r["dt"],
                       rank_override=None if r["rank_provenance"]["rule"] != "user override" else
                       (r["rank_noise"], r["rank_damp"]), allow_unverified=bool(r["warnings"]))
    kT, m, n = r["model"]["kT"], r["model"]["mass"], r["N"]
    P = D[:, col["P_norm"]]
    T = D[:, col["T_kin"]]
    steps_done = int(st.get("step", 0))
    expect_frames = (plan["prod_steps"] // plan["save_every"] + 1) if steps_done == total else None
    c = {}
    c["status_complete"] = st.get("state") == "complete" or (allow_incomplete and st.get("state") == "incomplete")
    if expect_config:
        c["config_name"] = r["config"] == expect_config
    c["params_match_registry"] = all(fresh[k] == r[k] for k in ("pppm", "pair_search", "force_method", "rank_noise",
                                                               "rank_damp", "model", "dt"))
    if r["config"] == "costopt_hiacc":
        c["tree_pairs_and_neighbour_forces"] = r["pair_search"] == "tree" and r["force_method"] == "neighbor"
    c["static_verification_for_these_parameters"] = bool(cfg["verification"].get("verified"))
    c["steps_contiguous"] = np.array_equal(D[:, col["step"]], np.arange(steps_done + 1))
    c["finite"] = bool(np.all(np.isfinite(D[:, :col["step_wall_s"] + 1])))
    c["momentum_conserved"] = float(np.abs(P - P[0]).max()) <= 1e-9 * np.sqrt(n * m * kT)
    c["rmin_above_stop"] = float(D[:, col["rmin"]].min()) > r["model"]["stop_rmin"]
    c["temperature_sanity_20pct"] = abs(float(T[1:].mean()) / kT - 1) < 0.2 if len(T) > 1 else False
    c["ritz_positive"] = float(np.nanmin(D[:, col["ritz_min"]])) > 0 if len(D) > 1 else False
    if expect_frames is not None:
        c["frames_complete"] = len(R["frame_step"]) == expect_frames
    if len(M):
        c["monitor_error_within_budget"] = float(M[:, 1:5].max()) <= r["operator_budget"]
    c["no_accuracy_warnings"] = not st.get("accuracy_warnings")
    walls = D[6:, col["step_wall_s"]] if len(D) > 6 else D[1:, col["step_wall_s"]]
    info = dict(run=str(d), config=r["config"], potential=r["potential"], kernel=r["law"], N=n,
                ranks=f"{r['rank_noise']}/{r['rank_damp']}", pair_search=r["pair_search"], force=r["force_method"],
                mesh_M=cfg["operator_record"]["M"], steps=steps_done, total_steps=total,
                init=cfg["init"]["method"], init_equilibrated=cfg["init"]["equilibrated"],
                step_ms_median=1e3 * float(np.median(walls)) if len(walls) else None,
                step_ms_p90=1e3 * float(np.percentile(walls, 90)) if len(walls) else None,
                peak_rss_mb=st.get("peak_rss_mb"), T_mean=float(T[1:].mean()) if len(T) > 1 else None,
                U_per_N_last=float(D[-1, col["U_per_N"]]), rmin_min=float(D[:, col["rmin"]].min()),
                P_drift_max=float(np.abs(P - P[0]).max()),
                ritz_range=[float(np.nanmin(D[:, col["ritz_min"]])), float(np.nanmax(D[:, col["ritz_max"]]))],
                monitor_err_max=float(M[:, 1:5].max()) if len(M) else None,
                S_kmin_max=float(M[:, 7].max()) if len(M) else None,
                frames=len(R["frame_step"]), segments=len(st.get("segments") or []),
                verification=cfg["verification"].get("key"), code=cfg["code"]["git"],
                host=cfg["code"]["host"], checks=c, passed=all(c.values()))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--expect-config")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()
    res = [check(Path(d), args.expect_config, args.allow_incomplete) for d in args.runs]
    for x in res:
        print(f"{x['run']}: {'PASS' if x['passed'] else 'FAIL'}  {x['config']} {x['potential']} {x['kernel']} N={x['N']} "
              f"ranks {x['ranks']} M {x['mesh_M']} steps {x['steps']}/{x['total_steps']} | step {x['step_ms_median']:.1f} ms "
              f"(p90 {x['step_ms_p90']:.1f}) peak RSS {x['peak_rss_mb']:.0f} MB | T {x['T_mean']:.4f} rmin {x['rmin_min']:.3f} "
              f"|dP| {x['P_drift_max']:.1e} Ritz [{x['ritz_range'][0]:.3f}, {x['ritz_range'][1]:.1f}]"
              + (f" monitor err {x['monitor_err_max']:.1e} S(kmin) {x['S_kmin_max']:.1f}" if x["monitor_err_max"] is not None else ""))
        for k, v in x["checks"].items():
            if not v:
                print(f"    FAILED: {k}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=float) + "\n")
    sys.exit(0 if all(x["passed"] for x in res) else 1)


if __name__ == "__main__":
    main()
