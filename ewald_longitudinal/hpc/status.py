#!/usr/bin/env python3
"""Status of every task of a task list, and the --array string of the tasks that are safe to (re)submit.

    python3 hpc/status.py hpc/tasks/production.tsv --run-root $RUN_ROOT [--job-name toy_production] [--no-slurm]

Evidence used for each task:
- **Slurm:** if `squeue` works, the user's queued jobs with the array's job name (default `toy_<tasklist stem>`, the
  name hpc/submit.sh gives). A task listed there (PENDING, RUNNING, COMPLETING, ...) is active.
- **Directory lock:** `.run.lock` in the run directory (run_lock.probe). "held" means a live process owns the run
  (active). A free lock alone is not proof of termination: on file systems with node-local locks a remote holder is
  invisible.
- **status.json:** the state the run last wrote.

| category | condition | resubmitted? |
|---|---|---|
| active | in the Slurm queue, or lock held | no |
| complete | state complete | no |
| failed | state failed | no; inspect, then `toy_run.py --resume --retry-failed` |
| resumable | state incomplete (stopped cleanly) | yes |
| resumable_unclean | state running, Slurm reachable and the task not queued, lock free (process gone without a clean exit) | yes |
| unknown | state running and Slurm not reachable (or the lock cannot be probed) | no: check with sacct / squeue first |
| not_started | no status.json and not queued | yes |

Without Slurm, "running" is never taken as terminated.
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_lock  # noqa: E402
from make_tasklist import compress  # noqa: E402


def slurm_active_tasks(job_name, squeue="squeue"):
    """{array task id: state} of this user's queued array tasks with job_name; None if Slurm cannot be queried."""
    if shutil.which(squeue) is None:
        return None
    try:
        out = subprocess.run([squeue, "--me", "-r", "-h", "-n", job_name, "-o", "%i|%T"], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    act = {}
    for line in out.stdout.splitlines():
        m = re.match(r"\s*(\d+)_(\d+)\|(\S+)", line)
        if m:
            act[int(m.group(2))] = m.group(3)
    return act


def classify(row, run_root, slurm):
    d = Path(run_root) / row["out_rel"]
    st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
    state = st.get("state", "not_started")
    lock = run_lock.probe(d) if d.exists() else "absent"
    tid = int(row["task_id"])
    queued = slurm is not None and tid in slurm
    if queued or lock == "held":
        cat = "active"
    elif state == "complete":
        cat = "complete"
    elif state == "failed":
        cat = "failed"
    elif state == "incomplete":
        cat = "resumable"
    elif state == "running":
        cat = "resumable_unclean" if (slurm is not None and lock in ("free", "absent")) else "unknown"
    else:
        cat = "not_started"
    return dict(task_id=tid, out_rel=row["out_rel"], category=cat, state=state, lock=lock,
                slurm_state=(slurm or {}).get(tid), step=st.get("step", 0), total=st.get("total_steps"),
                segments=len(st.get("segments") or []), peak_rss_mb=st.get("peak_rss_mb"),
                last_stop=(st.get("segments") or [{}])[-1].get("stop_reason"),
                accuracy_warnings=len(st.get("accuracy_warnings") or []))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasklist")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--job-name", help="Slurm job name of the array (default toy_<tasklist stem>)")
    ap.add_argument("--no-slurm", action="store_true", help="do not query Slurm (then 'running' stays unknown)")
    ap.add_argument("--squeue", default="squeue")
    ap.add_argument("--json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    job = args.job_name or f"toy_{Path(args.tasklist).stem}"
    slurm = None if args.no_slurm else slurm_active_tasks(job, args.squeue)
    with open(args.tasklist) as fh:
        rows = [classify(r, args.run_root, slurm) for r in csv.DictReader(fh, delimiter="\t")]
    print(f"Slurm: {'queried, job name ' + job + f', {len(slurm)} queued task(s)' if slurm is not None else 'not queried / unavailable'}")
    if not args.quiet:
        for r in rows:
            print(f"{r['task_id']:>4} {r['out_rel']:45s} {r['category']:18s} state {r['state']:11s} lock {r['lock']:7s}"
                  f" slurm {r['slurm_state'] or '-':10s} step {r['step']}/{r['total'] or '?'} seg {r['segments']}"
                  + (f" last: {r['last_stop']}" if r["last_stop"] else "")
                  + (f" ACCURACY WARNINGS {r['accuracy_warnings']}" if r["accuracy_warnings"] else ""))
    by = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r["task_id"])
    print({k: len(v) for k, v in sorted(by.items())})
    todo = sorted(by.get("not_started", []) + by.get("resumable", []) + by.get("resumable_unclean", []))
    print(f"safe to (re)submit: --array={compress(todo)}" if todo else "nothing to (re)submit")
    if by.get("unknown"):
        print(f"unknown (status 'running' but Slurm not queried; NOT resubmitted, check sacct/squeue): "
              f"{compress(by['unknown'])}")
    if by.get("failed"):
        print(f"failed, inspect first (toy_run.py --resume --retry-failed): {compress(by['failed'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(slurm_queried=slurm is not None, job_name=job, tasks=rows,
                                                   resubmit=todo), indent=1) + "\n")
    return dict(rows=rows, todo=todo, by=by, slurm=slurm)


if __name__ == "__main__":
    main()
