#!/usr/bin/env python3
"""Write a task list (TSV) for the Slurm array: one row per potential x kernel x N x seed.

    python3 hpc/make_tasklist.py --out hpc/tasks/production_N256_N512.tsv            # defaults below

Columns: task_id potential kernel N seed config burn_in production save_every out_rel
out_rel is relative to $RUN_ROOT (from hpc/cluster.env), so the list does not depend on the cluster paths.
Defaults: double_well and lj, kernels A and B, N 256 and 512, seeds 101-105, config costopt_hiacc, burn-in t = 140,
production t = 60, frames (q, v) every step (t = 0.005; see vacf_sampling_results/REPORT.md). The burn-in length is
an initial run plan, not an equilibration guarantee.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import toy_configs as tc  # noqa: E402

COLS = ("task_id", "potential", "kernel", "N", "seed", "config", "burn_in", "production", "save_every", "out_rel")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--potentials", nargs="+", default=list(tc.POTENTIALS))
    ap.add_argument("--kernels", nargs="+", default=list(tc.KERNELS))
    ap.add_argument("--N", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    ap.add_argument("--config", default="costopt_hiacc", choices=sorted(tc.CONFIGS))
    ap.add_argument("--burn-in", type=float, default=140.0)
    ap.add_argument("--production", type=float, default=60.0)
    ap.add_argument("--save-every", type=float, default=0.005,
                    help="frame interval (default every step, 0.005: law B's VACF halves in ~3.4 steps; "
                         "vacf_sampling_results/REPORT.md)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if tc.CONFIGS[args.config].get("state_point", "toy_v2") != "toy_v2":
        given = {o for o in ("--burn-in", "--production", "--save-every", "--seeds") if any(
            x == o or x.startswith(o + "=") for x in sys.argv)}
        if given != {"--burn-in", "--production", "--save-every", "--seeds"}:
            raise SystemExit(f"{args.config}: the defaults (burn-in 140, production 60, save 0.005, seeds 101-105) "
                             "belong to the toy_v2 state point; pass --burn-in, --production, --save-every and --seeds "
                             "explicitly (see lj075_results/REPORT.md)")
    for N in args.N:                                                  # contiguous ids per (N, kernel) group
        for ker in args.kernels:
            for pot in args.potentials:
                r = tc.resolve(args.config, ker, N, pot)              # refuses unverified combinations
                if r["warnings"]:
                    print(f"note N={N} {ker}: " + "; ".join(r["warnings"]))
                for seed in args.seeds:
                    rows.append(dict(potential=pot, kernel=ker, N=N, seed=seed, config=args.config,
                                     burn_in=args.burn_in, production=args.production, save_every=args.save_every,
                                     out_rel=f"{args.config}/{pot}_{ker}_N{N}_s{seed}"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLS)
        for i, r in enumerate(rows, 1):
            w.writerow([i] + [r[c] for c in COLS[1:]])
    groups = {}
    for i, r in enumerate(rows, 1):
        groups.setdefault((r["N"], r["kernel"]), []).append(i)
    print(f"{out}: {len(rows)} tasks")
    for (N, ker), ids in sorted(groups.items()):
        print(f"  N={N} kernel {ker}: task ids {compress(ids)}")


def compress(ids):
    """[1,2,3,7,8] -> '1-3,7-8' (Slurm --array syntax)."""
    out, start, prev = [], None, None
    for i in sorted(ids):
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = i
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(out)


if __name__ == "__main__":
    main()
