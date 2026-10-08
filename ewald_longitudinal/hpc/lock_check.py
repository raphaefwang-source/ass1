#!/usr/bin/env python3
"""Check that the run-directory lock works on the file system where runs are written (e.g. $RUN_ROOT).

    python3 hpc/lock_check.py --dir $RUN_ROOT [--json out.json]

The check runs four steps in a scratch subdirectory:
1. A holder process takes the lock.
2. A second process must be refused.
3. The probe must report "held".
4. After SIGKILL of the holder, the lock must be free and acquirable.

It also prints the file-system type and mount options. Lustre needs "flock" for locks that are coherent across nodes;
"localflock" gives node-local locks only and cannot be detected from a single node. NFS mounted "nolock" makes locks
local. Exit code 1 on failure.
"""
import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_lock  # noqa: E402


def mount_info(path):
    try:
        out = subprocess.run(["findmnt", "-T", str(path), "-n", "-o", "TARGET,FSTYPE,OPTIONS"], capture_output=True,
                             text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            target, fstype, opts = out.stdout.split(None, 2)
            return dict(target=target, fstype=fstype, options=opts.strip())
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    best = None
    p = str(Path(path).resolve())
    for line in Path("/proc/mounts").read_text().splitlines():
        dev, target, fstype, opts = line.split()[:4]
        if p == target or p.startswith(target.rstrip("/") + "/"):
            if best is None or len(target) > len(best["target"]):
                best = dict(target=target, fstype=fstype, options=opts)
    return best or {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    base = Path(args.dir)
    base.mkdir(parents=True, exist_ok=True)
    work = base / f".lockcheck_{socket.gethostname()}_{os.getpid()}"
    work.mkdir()
    res = dict(dir=str(base.resolve()), host=socket.gethostname(), mount=mount_info(base))
    holder = subprocess.Popen([sys.executable, "-c", "import sys, time; sys.path.insert(0, sys.argv[1]); import run_lock;"
                               " l = run_lock.RunLock(sys.argv[2]); ok = l.acquire(10); print('held' if ok else 'no',"
                               " flush=True); time.sleep(600)", str(HERE), str(work)], stdout=subprocess.PIPE, text=True)
    try:
        res["holder_acquired"] = holder.stdout.readline().strip() == "held"
        try:
            res["second_refused"] = run_lock.RunLock(work).acquire(wait_s=1.0) is False
        except run_lock.LockUnsupported as exc:
            res["second_refused"] = False
            res["error"] = str(exc)
        res["probe_while_held"] = run_lock.probe(work)
        holder.send_signal(signal.SIGKILL)
        holder.communicate(timeout=60)
        time.sleep(0.2)
        res["probe_after_kill"] = run_lock.probe(work)
        lk = run_lock.RunLock(work)
        res["acquire_after_kill"] = lk.acquire(wait_s=5.0)
        lk.release()
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.communicate()
        shutil.rmtree(work, ignore_errors=True)
    opts = res["mount"].get("options", "")
    fstype = res["mount"].get("fstype", "")
    warn = []
    if fstype == "lustre" and "localflock" in opts:
        warn.append("Lustre localflock: locks are node-local; two jobs on different nodes would not see each other")
    elif fstype == "lustre" and "flock" not in opts.split(","):
        warn.append("Lustre without 'flock' in the mount options: cross-node lock coherence not guaranteed")
    if fstype.startswith("nfs") and "nolock" in opts:
        warn.append("NFS 'nolock': locks are local to each client")
    res["warnings"] = warn
    res["passed"] = (res.get("holder_acquired") and res.get("second_refused") and res.get("probe_while_held") == "held"
                     and res.get("probe_after_kill") == "free" and res.get("acquire_after_kill")) is True
    print(json.dumps(res, indent=1))
    print("LOCK CHECK", "PASS" if res["passed"] else "FAIL", "(with warnings)" if warn else "")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
