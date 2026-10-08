#!/usr/bin/env python3
"""
Exclusive per-run-directory lock for toy_run.py (no NumPy import; also used by hpc/status.py).

The lock is a POSIX record lock (fcntl.lockf) on <run dir>/.run.lock.
- **Held for the process lifetime.** The runner takes it before reading, resuming or modifying any run state and keeps
  it until the process exits. The kernel releases it however the process ends (normal exit, exception, SIGKILL, node
  loss on file systems whose lock manager recovers). Liveness is never inferred from the file's existence or
  contents. The holder writes host / PID / job id into the file only so that a refused process can say who holds it.
- **POSIX semantics:** the lock belongs to the process. Closing ANY descriptor of the lock file in the holder releases
  it, so the holder never reopens the file.
- **Cluster file systems:**
  - Coherent across nodes on local disks, NFSv4, GPFS/Spectrum Scale and Lustre mounted with "-o flock".
  - Lustre mounted with "-o localflock" gives node-local locks only.
  - Lustre without either option refuses locks (ENOSYS); this module then raises LockUnsupported instead of
    running unlocked. hpc/lock_check.py tests the real RUN_ROOT.
"""
import errno
import fcntl
import json
import os
import socket
import time
from pathlib import Path

LOCK_NAME = ".run.lock"
_BUSY = (errno.EACCES, errno.EAGAIN)


class LockUnsupported(OSError):
    pass


class RunLock:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / LOCK_NAME
        self.fd = None

    def acquire(self, wait_s=30.0, poll_s=0.25):
        """True when the lock is held; False if another process held it for the whole wait (nothing written)."""
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        t0 = time.monotonic()
        while True:
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno in _BUSY:
                    if time.monotonic() - t0 >= wait_s:
                        os.close(fd)                     # this process held no lock on it
                        return False
                    time.sleep(poll_s)
                    continue
                os.close(fd)
                raise LockUnsupported(exc.errno, f"file locking not supported on {self.path.parent} "
                                                 f"({os.strerror(exc.errno)})") from exc
        info = dict(host=socket.gethostname(), pid=os.getpid(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
                    since=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        os.ftruncate(fd, 0)
        os.pwrite(fd, (json.dumps(info) + "\n").encode(), 0)
        os.fsync(fd)
        self.fd = fd
        return True

    def release(self):
        if self.fd is not None:
            try:
                fcntl.lockf(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None


def holder_info(run_dir):
    """Last holder's note (informational only; not a liveness test). Never call from the holding process."""
    try:
        return json.loads((Path(run_dir) / LOCK_NAME).read_text() or "{}")
    except (OSError, ValueError):
        return {}


def probe(run_dir):
    """'held', 'free', 'absent' (no lock file) or 'unsupported'.

    'free' briefly takes and drops the lock: a runner starting at that instant retries (toy_run --lock-wait), so the
    probe cannot make it fail. Never call from a process that holds the lock (POSIX: closing releases it)."""
    path = Path(run_dir) / LOCK_NAME
    if not path.exists():
        return "absent"
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return "absent"
    try:
        fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.lockf(fd, fcntl.LOCK_UN)
        return "free"
    except OSError as exc:
        return "held" if exc.errno in _BUSY else "unsupported"
    finally:
        os.close(fd)
