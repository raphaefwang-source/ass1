#!/usr/bin/env python3
"""Status of every task of a task list (reads $RUN_ROOT/<out_rel>/status.json).

    python3 hpc/status.py hpc/tasks/production_N256_N512.tsv --run-root $RUN_ROOT

Prints one line per task and the Slurm --array strings of tasks still to (re)submit: not started, incomplete
(stopped cleanly, resumable) and failed (needs inspection; not resubmitted automatically).
"""
import argparse
import csv
import json
from pathlib import Path

from make_tasklist import compress


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasklist")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    by = {}
    with open(args.tasklist) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            d = Path(args.run_root) / row["out_rel"]
            st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
            state = st.get("state", "not_started")
            if state == "running":
                state = "running_or_killed"            # a SIGKILL leaves "running"; resume handles both
            by.setdefault(state, []).append(int(row["task_id"]))
            if not args.quiet:
                seg = (st.get("segments") or [{}])[-1]
                warn = len(st.get("accuracy_warnings") or [])
                print(f"{row['task_id']:>4} {row['out_rel']:45s} {state:18s} step {st.get('step', 0)}/{st.get('total_steps', '?')}"
                      f" segments {len(st.get('segments') or [])} peak RSS {st.get('peak_rss_mb', 0):.0f} MB"
                      f"{'' if not seg.get('stop_reason') else ' last: ' + seg['stop_reason']}"
                      f"{'' if not warn else f' ACCURACY WARNINGS {warn}'}")
    print({k: len(v) for k, v in by.items()})
    todo = sorted(by.get("not_started", []) + by.get("incomplete", []) + by.get("running_or_killed", []))
    if todo:
        print(f"resubmit (not started / incomplete / killed): --array={compress(todo)}")
    if by.get("failed"):
        print(f"failed, inspect before any resubmission (toy_run.py --resume --retry-failed): {compress(by['failed'])}")


if __name__ == "__main__":
    main()
