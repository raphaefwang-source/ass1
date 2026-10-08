#!/usr/bin/env python3
"""Resource estimate for production tasks from pilot runs (one core per trajectory).

    python3 hpc/estimate_resources.py --pilot PILOT_DIR/double_well_A_N512_s101 PILOT_DIR/double_well_B_N512_s101 \\
        --tasklist hpc/tasks/production_N256_N512.tsv --margin 1.5 [--max-job-hours 24] [--max-parallel 20]

For every (N, kernel) of the task list:
- run time per task = steps x median pilot step time;
- --time = that x margin, rounded up to 15 min, plus 10 min for start-up and checkpoints;
- --mem = max(1 GB, 2 x pilot peak RSS), rounded up to 0.5 GB;
- disk = frames, per-step diagnostics and checkpoints.

An N without a pilot (e.g. 256) is scaled from the pilot at another N of the same kernel by the ratio of the local
benchmark step times (cost_optimization_results/timing.json, same machine for both N); the output labels this.

Queue waiting time cannot be estimated from a pilot (site load, fair share, priorities); only run time and core-hours
are given. Wall-clock for the whole array = ceil(tasks / max_parallel) x run time, excluding queue waits.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import toy_run as tr  # noqa: E402

LOCAL_TAG = {("costopt_hiacc", "A"): "A_2e-7", ("costopt_hiacc", "B"): "B_5e-8",
             ("baseline_hiacc", "A"): "baseline_images", ("baseline_hiacc", "B"): "baseline_images"}


def pilot_stats(d):
    R = tr.load_run(d)
    cfg, D = R["config"], R["diag"]
    r = cfg["resolved"]
    walls = D[6:, tr.DIAG_COLS.index("step_wall_s")]
    ck = Path(d) / "checkpoint.npz"
    return dict(run=str(d), config=r["config"], kernel=r["law"], N=r["N"], dt=r["dt"], steps=len(walls),
                step_s_median=float(np.median(walls)), step_s_p90=float(np.percentile(walls, 90)),
                peak_rss_mb=float(R["status"].get("peak_rss_mb") or 0), checkpoint_bytes=ck.stat().st_size,
                frame_bytes=2 * r["N"] * 3 * 8, diag_row_bytes=len(tr.DIAG_COLS) * 8,
                host=cfg["code"]["host"], monitor_every=cfg["runtime"].get("monitor_every_steps"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", nargs="+", required=True)
    ap.add_argument("--tasklist")
    ap.add_argument("--burn-in", type=float, default=140.0, help="used when no task list is given")
    ap.add_argument("--production", type=float, default=60.0)
    ap.add_argument("--save-every", type=float, default=0.1)
    ap.add_argument("--N", type=int, nargs="+", default=[256, 512], help="used when no task list is given")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--max-job-hours", type=float, help="site wall-time limit per job (splits long tasks)")
    ap.add_argument("--max-parallel", type=int, default=20)
    ap.add_argument("--local-timing", default=str(HERE / "cost_optimization_results" / "timing.json"))
    ap.add_argument("--json")
    args = ap.parse_args()
    pil = [pilot_stats(d) for d in args.pilot]
    local = json.loads(Path(args.local_timing).read_text()) if Path(args.local_timing).exists() else []
    if args.tasklist:
        with open(args.tasklist) as fh:
            tasks = list(csv.DictReader(fh, delimiter="\t"))
    else:
        tasks = [dict(kernel=p["kernel"], N=N, config=p["config"], burn_in=args.burn_in, production=args.production,
                      save_every=args.save_every) for p in pil for N in args.N]
    groups = {}
    for t in tasks:
        groups.setdefault((t["config"], t["kernel"], int(t["N"]), float(t["burn_in"]), float(t["production"]),
                           float(t["save_every"])), []).append(t)
    rows = []
    for (cfgname, ker, N, burn, prod, save), ts in sorted(groups.items()):
        same = [p for p in pil if p["config"] == cfgname and p["kernel"] == ker]
        if not same:
            rows.append(dict(config=cfgname, kernel=ker, N=N, tasks=len(ts), note="no pilot for this config/kernel"))
            continue
        exact = [p for p in same if p["N"] == N]
        p = exact[0] if exact else same[0]
        step_s, source = p["step_s_median"], f"pilot {p['run']}"
        rss = p["peak_rss_mb"]
        if not exact:
            tag = LOCAL_TAG.get((cfgname, ker))
            t_loc = {r["N"]: r["step_median"] for r in local if r.get("tag") == tag and r["law"] == ker}
            if N in t_loc and p["N"] in t_loc:
                step_s *= t_loc[N] / t_loc[p["N"]]
                source = (f"scaled from pilot N={p['N']} by the local benchmark ratio "
                          f"{t_loc[N] / t_loc[p['N']]:.3f} (timing.json tag {tag}); not a pilot at this N")
                rss = max(rss, 0.0)                      # upper bound: the larger pilot's peak RSS
            else:
                rows.append(dict(config=cfgname, kernel=ker, N=N, tasks=len(ts), note="no pilot and no local benchmark"))
                continue
        dt = p["dt"]
        steps = int(round((burn + prod) / dt))
        run_h = steps * step_s / 3600
        req_h = math.ceil(run_h * args.margin * 4) / 4 + 1 / 6
        segs = 1 if not args.max_job_hours or req_h <= args.max_job_hours else math.ceil(req_h / (args.max_job_hours - 1 / 6))
        per_job_h = req_h if segs == 1 else args.max_job_hours
        mem_gb = max(1.0, math.ceil(2 * rss / 1024 * 2) / 2)
        frames = int(round(prod / save)) + 1
        disk_mb = (frames * 2 * N * 3 * 8 + (steps + 1) * len(tr.DIAG_COLS) * 8 + 3 * p["checkpoint_bytes"]
                   * (N / p["N"])) / 2 ** 20
        rows.append(dict(config=cfgname, kernel=ker, N=N, tasks=len(ts), steps_per_task=steps, step_ms=1e3 * step_s,
                         step_time_source=source, run_hours_per_task=run_h, sbatch_time_hours=per_job_h,
                         sbatch_time=f"{int(per_job_h):02d}:{int(round(per_job_h % 1 * 60)):02d}:00",
                         jobs_per_task=segs, sbatch_mem_gb=mem_gb, pilot_peak_rss_mb=rss, disk_mb_per_task=disk_mb,
                         core_hours=len(ts) * run_h, core_hours_with_margin=len(ts) * run_h * args.margin,
                         wall_hours_if_parallel=math.ceil(len(ts) / args.max_parallel) * run_h * segs ** 0))
    tot = dict(tasks=sum(r["tasks"] for r in rows), core_hours=sum(r.get("core_hours", 0) for r in rows),
               core_hours_with_margin=sum(r.get("core_hours_with_margin", 0) for r in rows),
               disk_gb=sum(r.get("disk_mb_per_task", 0) * r["tasks"] for r in rows) / 1024,
               margin=args.margin, max_parallel=args.max_parallel,
               queue_wait="not estimable from a pilot (site load, fair share, priorities)")
    out = dict(pilots=pil, groups=rows, totals=tot)
    for r in rows:
        if "note" in r:
            print(f"{r['config']} {r['kernel']} N={r['N']}: {r['tasks']} tasks - {r['note']}")
            continue
        print(f"{r['config']} {r['kernel']} N={r['N']}: {r['tasks']} tasks x {r['steps_per_task']} steps, {r['step_ms']:.1f} ms/step "
              f"-> {r['run_hours_per_task']:.2f} h run per task; sbatch --time={r['sbatch_time']} (margin {args.margin}, "
              f"{r['jobs_per_task']} job(s) per task) --mem={r['sbatch_mem_gb']:g}G; {r['core_hours']:.1f} core-h "
              f"({r['core_hours_with_margin']:.1f} with margin); disk {r['disk_mb_per_task']:.0f} MB/task\n    step time: {r['step_time_source']}")
    print(f"TOTAL {tot['tasks']} tasks: {tot['core_hours']:.1f} core-h run time ({tot['core_hours_with_margin']:.1f} with "
          f"margin), disk {tot['disk_gb']:.2f} GB; queue wait: {tot['queue_wait']}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
