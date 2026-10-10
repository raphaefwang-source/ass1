#!/usr/bin/env python3
"""
LJ law A, HPC short pilot: dense full-periodic reference O-step vs fast PPPM-Lanczos O-step, dt 0.005 and 0.01,
each to t = 1 from one shared canonical state. Checks the cluster workflow and measures cost; no physics claim.

    python3 lj075_hpc_pilot.py case --case fast_dt0.005 --root DIR    # one case (one Slurm job, hpc/lj075_pilot_case.sbatch)
    python3 lj075_hpc_pilot.py analyze --root DIR --out OUT           # after the four cases (hpc/lj075_pilot_analyze.sbatch)
    python3 lj075_hpc_pilot.py all --root DIR --out OUT               # local: the four cases in sequence, then analyze

Fixed experiment (nothing here is a command-line option):
- model: toy_configs "lj_rho0.75_kT1.0_A_prod" (LJ, law A, N 256, rho* 0.75, kT 1, full periodic friction with the
  k = 0 mode retained), PPPM xi 0.85, s 4.1, eta 0.6779, p 8 (mesh M 24), Lanczos ranks noise 14 / damping 5;
- cases: dense_dt0.005, fast_dt0.005, dense_dt0.01, fast_dt0.01. "dense" = toy_run.py --method reference: the full
  periodic lattice-sum Gamma (v2.LatticeFriction; not the PPPM Gamma_h) with dense eigendecomposition matrix functions.
  "fast" = toy_run.py --method pppm_lanczos (production). Same conservative force (toy_dynamics.conservative_force_neighbor)
  and the same runner in all four;
- initial state: lj075_results/hpc_pilot/init_state_eqA_lgv_s305_t50.npz (end of the law A equilibration run
  eq_A_lgv_s305, t = 50 after a Langevin-prepared canonical start), identical for all four cases;
- noise: at each dt, fast and dense use the same seed, so the runner's stream default_rng([seed, 1]) gives both the
  same standard-normal input at every step (checked: identical RNG state at the end). The two step sizes use different
  seeds and are NOT coupled: same seeds would not couple paths of different dt, and the verified OU coupling
  (lj075_coupled.py) makes the dt = 0.01 input depend on Gamma, which would break the identical fast / dense inputs.
  No cross-dt pathwise comparison is made;
- every step: positions (unwrapped), velocities and diagnostics saved; accuracy monitor every 10 steps; checkpoints
  every 25 steps counted from each segment's start (the cadence is passed to every segment) and at every stop.

Per case (subcommand "case"), in DIR/<case>/:
  run/      the trajectory, in three segments: (1) stopped by SIGUSR1 sent after its first checkpoint (exit 75 expected),
            (2) resumed with a 30-step segment limit (exit 75) while a duplicate resume on the same directory must be
            refused by the lock (exit 3), (3) resumed to t = 1 (exit 0);
  twin/     a continuous run of the same case up to 10 steps past the second restart: positions, momenta, diagnostics
            (except wall time), frames and monitor rows must agree bitwise with run/ (checkpoint save and restore);
  env_report.json, lock_check.json, case_summary.json (exit codes, process wall / CPU / peak RSS per phase).

"analyze" reads the four cases and writes OUT/pilot_results.json, pilot_table.md, pilot_report.md, per-step
differences (pilot_differences.npz) and figures. It labels the run LOCAL (not an HPC result) unless every segment
carries a Slurm job id.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import run_lock  # noqa: E402
import toy_configs as tc  # noqa: E402
import toy_run as tr  # noqa: E402

CONFIG = "lj_rho0.75_kT1.0_A_prod"
LAW, N, POT = "A", 256, "lj"
T_END = 1.0
INIT = HERE / "lj075_results" / "hpc_pilot" / "init_state_eqA_lgv_s305_t50.npz"
SEEDS = {0.005: 7005, 0.01: 7010}
CASES = {
    "dense_dt0.005": dict(method="reference", dt=0.005),
    "fast_dt0.005": dict(method="pppm_lanczos", dt=0.005),
    "dense_dt0.01": dict(method="reference", dt=0.01),
    "fast_dt0.01": dict(method="pppm_lanczos", dt=0.01),
}
PAIRS = {0.005: ("fast_dt0.005", "dense_dt0.005"), 0.01: ("fast_dt0.01", "dense_dt0.01")}
CHECKPOINT_EVERY, MONITOR_EVERY, SEGMENT2_STEPS, TWIN_EXTRA = 25, 10, 30, 10
CADENCE = ["--checkpoint-every-steps", str(CHECKPOINT_EVERY), "--checkpoint-every-minutes", "600"]  # every segment:
# toy_run.py --resume takes the checkpoint cadence from its own command line (default 2000 steps), not config.json
ENV1 = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
STOP = {"signal": None, "child": None}


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def toy_args(case, out, steps=None):
    c = CASES[case]
    a = ["--config", CONFIG, "--potential", POT, "--kernel", LAW, "--N", str(N), "--seed", str(SEEDS[c["dt"]]),
         "--dt", str(c["dt"]), "--allow-unverified", "--save-every-steps", "1",
         *CADENCE, "--monitor-every-steps", str(MONITOR_EVERY),
         "--from-state", str(INIT), "--method", c["method"], "--out", str(out), "--quiet"]
    return a + (["--production-steps", str(steps)] if steps is not None else ["--production", str(T_END)])


def _child_usage():
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime, ru.ru_maxrss / 1024


def _forward(sig, _frame):
    """Slurm's USR1 / TERM to the batch shell is forwarded here: pass it to the running toy_run (it checkpoints and
    exits 75) and start no further phase."""
    STOP["signal"] = signal.Signals(sig).name
    ch = STOP["child"]
    if ch is not None and ch.poll() is None:
        ch.send_signal(signal.SIGUSR1)


def _phase(rec, name, argv, expect, wait_until=None, on_running=None):
    """Run one toy_run.py process; record exit code, process wall and CPU. wait_until(proc) -> True sends SIGUSR1."""
    if STOP["signal"]:
        rec["phases"].append(dict(phase=name, skipped=f"external {STOP['signal']}"))
        return None
    cpu0, _ = _child_usage()
    t0 = time.monotonic()
    proc = subprocess.Popen([sys.executable, "-u", str(HERE / "toy_run.py"), *argv], env=ENV1)
    STOP["child"] = proc
    extra = {}
    if wait_until is not None:
        while proc.poll() is None and not wait_until():
            time.sleep(0.05)
        if proc.poll() is None:
            proc.send_signal(signal.SIGUSR1)
            extra["sigusr1_sent_after_s"] = time.monotonic() - t0
    if on_running is not None:
        extra.update(on_running(proc))
    rc = proc.wait()
    STOP["child"] = None
    cpu1, rss = _child_usage()
    rec["phases"].append(dict(phase=name, exit_code=rc, expected_exit_code=expect, process_wall_s=time.monotonic() - t0,
                              process_cpu_s=cpu1 - cpu0, children_peak_rss_mb_so_far=rss, **extra))
    return rc


def run_case(case, root):
    if case not in CASES:
        sys.exit(f"unknown case {case}; choose from {sorted(CASES)}")
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, _forward)
    d = Path(root) / case
    run, twin = d / "run", d / "twin"
    if run.exists() or twin.exists():
        sys.exit(f"{d} already holds this case; use a new --root")
    d.mkdir(parents=True)
    t0 = time.monotonic()
    rec = dict(case=case, **CASES[case], seed=SEEDS[CASES[case]["dt"]], root=str(Path(root).resolve()), start=_now(),
               slurm_job_id=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
               threads={v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
               init_state=str(INIT), init_sha256=hashlib.sha256(INIT.read_bytes()).hexdigest(), phases=[])
    (d / "env_report.json").write_text(subprocess.run([sys.executable, str(HERE / "hpc" / "env_report.py")], env=ENV1,
                                                      capture_output=True, text=True).stdout)
    rec["lock_check_exit_code"] = subprocess.run(
        [sys.executable, str(HERE / "hpc" / "lock_check.py"), "--dir", str(d), "--json", str(d / "lock_check.json")],
        env=ENV1, capture_output=True, text=True).returncode

    def first_checkpoint():
        return any(int(c.stem.split("_")[1]) >= CHECKPOINT_EVERY for c in (run / "chunks").glob("chunk_*.npz")) \
            if (run / "chunks").exists() else False

    def duplicate(proc):
        t = time.monotonic()
        while proc.poll() is None and run_lock.probe(run) != "held" and time.monotonic() - t < 120:
            time.sleep(0.05)
        if proc.poll() is not None:
            return dict(duplicate_exit_code=None, duplicate_note="segment 2 ended before the duplicate could start")
        r = subprocess.run([sys.executable, str(HERE / "toy_run.py"), "--resume", "--out", str(run), "--lock-wait", "0",
                            *CADENCE, "--quiet"], env=ENV1, capture_output=True, text=True)
        return dict(duplicate_exit_code=r.returncode, duplicate_expected_exit_code=tr.EXIT_LOCKED,
                    duplicate_stderr=r.stderr.strip()[-300:], segment2_running_after_duplicate=proc.poll() is None)

    _phase(rec, "segment1_signal_stop", toy_args(case, run), tr.EXIT_INCOMPLETE, wait_until=first_checkpoint)
    _phase(rec, "segment2_step_limit", ["--resume", "--out", str(run), "--max-segment-steps", str(SEGMENT2_STEPS),
                                        *CADENCE, "--quiet"], tr.EXIT_INCOMPLETE, on_running=duplicate)
    for attempt in range(4):                                  # a stray external signal would stop it with 75
        rc = _phase(rec, f"segment3_to_end_{attempt}", ["--resume", "--out", str(run), *CADENCE, "--quiet"],
                    tr.EXIT_COMPLETE)
        if rc != tr.EXIT_INCOMPLETE:
            break
    st = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
    seg = st.get("segments", [])
    if len(seg) >= 2 and not STOP["signal"]:
        n_twin = int(seg[1]["end_step"]) + TWIN_EXTRA
        rc_twin = _phase(rec, "twin_continuous", toy_args(case, twin, steps=n_twin), tr.EXIT_COMPLETE)
        if (twin / "checkpoint.npz").exists():
            rec["restart"] = compare_restart(run, twin, [int(s["end_step"]) for s in seg[:2]], n_twin, rc_twin)
        else:
            rec["restart"] = dict(passed=False, reason=f"twin not run (exit {rc_twin})")
    else:
        rec["restart"] = dict(passed=False, reason=f"{len(seg)} segment(s), external stop {STOP['signal']}")
    rec.update(end=_now(), case_wall_s=time.monotonic() - t0, run_state=st.get("state"),
               children_cpu_s=_child_usage()[0], children_peak_rss_mb=_child_usage()[1], external_stop=STOP["signal"])
    (d / "case_summary.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    ok = (st.get("state") == "complete" and all(p.get("exit_code") == p.get("expected_exit_code")
                                                for p in rec["phases"] if "exit_code" in p)
          and rec.get("restart", {}).get("passed", False) and rec["lock_check_exit_code"] == 0
          and any(p.get("duplicate_exit_code") == tr.EXIT_LOCKED for p in rec["phases"]))
    print(f"case {case}: {'OK' if ok else 'NOT OK'} (see {d / 'case_summary.json'})", flush=True)
    return 0 if ok else 1


def compare_restart(run, twin, restart_steps, n_twin, rc_twin):
    """Bitwise: the twin's continuous trajectory against the restarted run over the twin's steps. Passes only if the
    twin completed (exit 0) past both restart points."""
    A, B = tr.load_run(twin), tr.load_run(run)
    n = len(A["diag"])
    keep = [i for i, c in enumerate(tr.DIAG_COLS) if c != "step_wall_s"]
    mA, mB = A["monitor"], B["monitor"][B["monitor"][:, 0] < n] if len(B["monitor"]) else B["monitor"]
    with np.load(twin / "checkpoint.npz") as x:
        qT, pT, sT = x["q"], x["p"], int(x["step"])
    res = dict(restart_steps=restart_steps, twin_steps=sT,
               diag_bitwise_except_wall=bool(np.array_equal(A["diag"][:, keep], B["diag"][:n, keep], equal_nan=True)),
               frames_bitwise=bool(np.array_equal(A["Q"], B["Q"][:n]) and np.array_equal(A["V"], B["V"][:n])
                                   and np.array_equal(A["frame_step"], B["frame_step"][:n])),
               monitor_bitwise=bool(np.array_equal(mA, mB, equal_nan=True)),
               q_p_at_twin_end_bitwise=bool(np.array_equal(qT, B["Q"][sT]) and np.array_equal(pT, B["V"][sT])),
               twin_exit_code=rc_twin, twin_complete=rc_twin == tr.EXIT_COMPLETE and sT == n_twin,
               covers_both_restarts=bool(sT > max(restart_steps)),
               note="mass 1: the saved velocity V = p / m equals p bitwise")
    res["passed"] = all(v for k, v in res.items() if isinstance(v, bool))
    return res


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------
def _fit_growth(t, y, tmin=0.1):
    """Exponential growth rate of y(t) from a least-squares fit of log y on t >= tmin (y > 0)."""
    m = (t >= tmin) & (y > 0)
    if m.sum() < 3:
        return None
    A = np.vstack([t[m], np.ones(m.sum())]).T
    coef, *_ = np.linalg.lstsq(A, np.log(y[m]), rcond=None)
    return float(coef[0])


def _power(t, y, tmin=0.1):
    """Exponent a of y ~ t^a (log-log least squares on t >= tmin, y > 0): about 0.5 for accumulated round-off (random
    walk), 1 for a systematic drift."""
    m = (t >= tmin) & (y > 0)
    if m.sum() < 3:
        return None
    return float(np.polyfit(np.log(t[m]), np.log(y[m]), 1)[0])


def lanczos_check(R, case):
    """At every monitor step of a fast run: the monitor's error estimate, the actual Lanczos error against dense
    matrix functions of the same Gamma_h, and the PPPM operator error against the full periodic Gamma."""
    import toy_dynamics as td
    from test_lanczos_fdt import DenseRef, proj
    from test_true_dynamics import lanczos_apply
    cfg = R["config"]
    r = cfg["resolved"]
    m = r["model"]
    th = tr.build_thermostat(r)
    ref = td.Thermostat("reference", r["L"], r["law"], m["gamma"], m["kappa"], m["r_ref"], r["dt"], m["kT"], m["mass"])
    seed, L = cfg["plan"]["seed"], r["L"]
    rows = []
    for mon in R["monitor"]:
        s = int(mon[0])
        q, p = R["Q"][s], R["V"][s] * m["mass"]
        op = th.fast.gamma_h(q)
        Dh = DenseRef(op.dense(), N)
        Dr = DenseRef(ref.lat.matrix(q % L), N)
        z = proj(np.random.default_rng([seed, 3, s]).standard_normal(3 * N))       # the monitor's own input
        pp = proj(p.ravel())
        out = dict(step=s, t=s * r["dt"], est_noise=float(mon[1]), est_damp=float(mon[3]))
        for key, v, f, rank in (("noise", z, th.fn, th.rank_noise), ("damp", pp, th.fd, th.rank_damp)):
            exact_h = Dh.apply(f, v)
            out[f"true_{key}"] = float(np.linalg.norm(proj(lanczos_apply(op, v, rank, f)[0]) - exact_h)
                                       / np.linalg.norm(exact_h))
            exact_r = Dr.apply(f, v)
            out[f"operator_{key}"] = float(np.linalg.norm(exact_h - exact_r) / np.linalg.norm(exact_r))
        Gh, Gr = Dh.matrix(lambda x: x), Dr.matrix(lambda x: x)
        out["operator_frobenius"] = float(np.linalg.norm(Gh - Gr) / np.linalg.norm(Gr))
        out.update(lam_min_h=float(Dh.lam[0]), lam_min_ref=float(Dr.lam[0]), lam_max_h=float(Dh.lam[-1]),
                   lam_max_ref=float(Dr.lam[-1]))
        rows.append(out)
    return rows


def _segments_cost(st):
    seg = st.get("segments", [])
    keys = ("import_wall_s", "init_wall_s", "init_cpu_s", "wall_s", "cpu_s", "step_wall_s_sum", "monitor_wall_s",
            "output_wall_s")
    tot = {k: float(sum((s.get(k) or 0.0) for s in seg)) for k in keys}
    tot["steps_run"] = int(sum(s.get("steps_run") or 0 for s in seg))
    tot["segments"] = len(seg)
    tot["peak_rss_mb"] = float(max([s.get("peak_rss_mb") or 0 for s in seg] + [0]))
    tot["os_threads"] = sorted({s.get("os_threads") for s in seg if s.get("os_threads") is not None})
    tot["slurm_job_ids"] = sorted({str(s.get("slurm_job_id")) for s in seg})
    tot["hosts"] = sorted({s.get("host") for s in seg})
    tot["first_segment_init_wall_s"] = float(seg[0].get("init_wall_s") or 0) if seg else None
    tot["first_segment_import_wall_s"] = float(seg[0].get("import_wall_s") or 0) if seg else None
    return tot


def analyze(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t_an = time.monotonic()
    R, S, ENV, LOCK, CHUNKS = {}, {}, {}, {}, {}
    missing, failed = [], {}
    for c in CASES:
        d = root / c
        if not (d / "run" / "config.json").exists():
            missing.append(c)
            if d.exists():
                cs = json.loads((d / "case_summary.json").read_text()) if (d / "case_summary.json").exists() else {}
                lk = json.loads((d / "lock_check.json").read_text()) if (d / "lock_check.json").exists() else {}
                failed[c] = dict(state="no run directory state", step=None, stop_reason=None, failure_diag=[],
                                 phases=cs.get("phases"), lock_check=dict(passed=lk.get("passed"),
                                                                          warnings=lk.get("warnings"), mount=lk.get("mount")))
            continue
        X = tr.load_run(d / "run")
        S_c = json.loads((d / "case_summary.json").read_text()) if (d / "case_summary.json").exists() else {}
        if len(X["diag"]) < 2 or len(X["frame_step"]) < 2:
            seg = X["status"].get("segments") or [{}]
            missing.append(c)
            failed[c] = dict(state=X["status"].get("state"), step=X["status"].get("step"),
                             stop_reason=seg[-1].get("stop_reason"),
                             failure_diag=[p.name for p in (d / "run").glob("failure_diag_*.npz")],
                             phases=S_c.get("phases"))
            continue
        R[c], S[c] = X, S_c
        ENV[c] = json.loads((d / "env_report.json").read_text()) if (d / "env_report.json").exists() else {}
        LOCK[c] = json.loads((d / "lock_check.json").read_text()) if (d / "lock_check.json").exists() else {}
        CHUNKS[c] = sorted(int(p.stem.split("_")[1]) for p in (d / "run" / "chunks").glob("chunk_*.npz"))
    with np.load(INIT) as z:
        q0, p0 = z["q"], z["p"]
        init_prov = json.loads(str(z["provenance"]))
    res = dict(created=_now(), root=str(root.resolve()), missing_cases=missing, failed_cases=failed,
               init_state=str(INIT.relative_to(HERE)),
               init_provenance=init_prov, cases={}, pairs={}, checks={})
    git = {c: R[c]["config"]["code"]["git"] for c in R}
    on_slurm = {c: all(j not in ("None", "") for j in _segments_cost(R[c]["status"])["slurm_job_ids"]) for c in R}
    res["environment"] = dict(
        label="HPC (Slurm)" if R and all(on_slurm.values()) else "LOCAL (no Slurm job id): not an HPC result",
        on_slurm=on_slurm, git=git,
        hardware={c: dict(host=ENV[c].get("host"), cpu_model=ENV[c].get("cpu_model"),
                          cores_visible=ENV[c].get("cores_visible"), platform=ENV[c].get("platform"),
                          python=ENV[c].get("python"), numpy=ENV[c].get("numpy"), scipy=ENV[c].get("scipy"),
                          slurm=ENV[c].get("slurm"), blas=(ENV[c].get("blas") or {}).get("name")
                          if isinstance(ENV[c].get("blas"), dict) else ENV[c].get("blas")) for c in R})
    chk = res["checks"]
    diffs = {}
    for c, X in R.items():
        cfg, st = X["config"], X["status"]
        r, plan = cfg["resolved"], cfg["plan"]
        D, cols = X["diag"], {k: i for i, k in enumerate(tr.DIAG_COLS)}
        n_exp = int(round(T_END / r["dt"]))
        t = D[:, cols["t"]]
        m = r["model"]["mass"]
        P = m * X["V"].sum(axis=1)                                       # (frames, 3) total momentum
        dP = np.linalg.norm(P - p0.sum(axis=0)[None], axis=1)
        half = len(dP) // 2
        alpha = _power(X["frame_step"] * r["dt"], dP)
        # round-off scale of one step's total momentum: eps * sum_i |p_i| (median over frames); a random walk of n
        # steps gives about sqrt(n) times that
        eps_sum = float(np.finfo(float).eps * np.median(np.abs(X["V"] * m).sum(axis=(1, 2))))
        walk = eps_sum * np.sqrt(n_exp)
        if dP.max() <= walk:
            p_verdict = (f"no drift detectable: max |ΔP| = {dP.max() / eps_sum:.1f} ε Σ|p|, within the round-off of "
                         f"forming Σp (random-walk scale over {n_exp} steps: {np.sqrt(n_exp):.0f} ε Σ|p|)")
        else:
            p_verdict = (f"above the round-off scale: max |ΔP| = {dP.max() / walk:.1f} × ε Σ|p| √n; second / first "
                         f"half max = {dP[half:].max() / max(dP[:half].max(), 1e-300):.1f}; growth exponent "
                         f"{_f(alpha, '.2f')}")
        segs = st.get("segments", [])
        contiguous = (bool(segs) and int(segs[0].get("start_step", -1)) == 0
                      and all(int(b.get("start_step", -1)) == int(a.get("end_step", -2)) for a, b in zip(segs, segs[1:]))
                      and sum(int(x.get("steps_run") or 0) for x in segs) == n_exp)
        # toy_run checkpoints every CHECKPOINT_EVERY steps counted from the segment start, and at the segment end
        want = sorted({0} | {k for x in segs for k in range(int(x["start_step"]) + CHECKPOINT_EVERY, int(x["end_step"]),
                                                              CHECKPOINT_EVERY)} | {int(x["end_step"]) for x in segs})
        finite = bool(np.all(np.isfinite(X["Q"])) and np.all(np.isfinite(X["V"]))
                      and np.all(np.isfinite(D[:, :cols["step_wall_s"] + 1])) and np.all(np.isfinite(D[1:, 8:10])))
        mon = X["monitor"]
        cost = _segments_cost(st)
        # cost statistics without segment 2: the duplicate lock test runs during it and, on a one-CPU allocation,
        # shares the core with it
        segs_all = st.get("segments", [])
        keep = [x for i, x in enumerate(segs_all) if i != 1] or segs_all
        step_no = D[:, cols["step"]]
        in_keep = np.zeros(len(D), bool)
        for x in keep:
            in_keep |= (step_no > int(x.get("start_step", 0))) & (step_no <= int(x.get("end_step", -1)))
        walls = D[in_keep, cols["step_wall_s"]]
        if len(walls) == 0:
            walls = D[1:, cols["step_wall_s"]]
        steps_k = max(sum(int(x.get("steps_run") or 0) for x in keep), 1)
        sum_k = {k: float(sum(x.get(k) or 0.0 for x in keep)) for k in ("step_wall_s_sum", "monitor_wall_s",
                                                                        "output_wall_s")}
        steps = max(cost["steps_run"], 1)
        with np.load(root / c / "run" / "checkpoint.npz") as ck:
            rng_final = str(ck["rng_state"])
        with np.load(root / c / "run" / "init_state.npz") as z:
            same_init = bool(np.array_equal(z["q"], q0) and np.array_equal(z["p"], p0))
        e = dict(method=r.get("thermostat_method", "pppm_lanczos"), dt=r["dt"], seed=plan["seed"], config=r["config"],
                 law=r["law"], N=r["N"], L=r["L"], model=r["model"], force_method=r["force_method"],
                 pair_search=r["pair_search"], pppm=r["pppm"], operator_record=cfg["operator_record"],
                 ranks=[r["rank_noise"], r["rank_damp"]] if CASES[c]["method"] == "pppm_lanczos" else None,
                 mesh_M=cfg["operator_record"].get("M"), threads=cfg["code"]["threads"], git=cfg["code"]["git"],
                 state=st.get("state"), steps=int(st.get("step", 0)), steps_expected=n_exp,
                 frames=len(X["frame_step"]), frames_contiguous=bool(np.array_equal(X["frame_step"], np.arange(n_exp + 1))),
                 init_identical_to_shared_state=same_init, finite=finite,
                 momentum=dict(P0=p0.sum(axis=0).tolist(), max_dev=float(dP.max()), end_dev=float(dP[-1]),
                               max_dev_first_half=float(dP[:half].max()), max_dev_second_half=float(dP[half:].max()),
                               growth_exponent=alpha if dP.max() > walk else None, roundoff_per_step=eps_sum,
                               verdict=p_verdict, within_roundoff_walk=bool(dP.max() <= walk),
                               max_dev_over_roundoff_walk=float(dP.max() / (eps_sum * np.sqrt(n_exp))),
                               linear_extrapolation_t55_rel=float(dP.max() / T_END * 55 / np.sqrt(N * m * r["model"]["kT"])),
                               scale_sqrt_NmkT=float(np.sqrt(N * m * r["model"]["kT"])),
                               rel_max_dev=float(dP.max() / np.sqrt(N * m * r["model"]["kT"]))),
                 T_kin=dict(t0=float(D[0, cols["T_kin"]]), mean=float(D[1:, cols["T_kin"]].mean()),
                            min=float(D[:, cols["T_kin"]].min()), max=float(D[:, cols["T_kin"]].max()),
                            end=float(D[-1, cols["T_kin"]])),
                 U_per_N=dict(t0=float(D[0, cols["U_per_N"]]), mean=float(D[1:, cols["U_per_N"]].mean()),
                              min=float(D[:, cols["U_per_N"]].min()), max=float(D[:, cols["U_per_N"]].max()),
                              end=float(D[-1, cols["U_per_N"]])),
                 rmin=float(D[:, cols["rmin"]].min()),
                 ritz=[float(np.nanmin(D[:, cols["ritz_min"]])), float(np.nanmax(D[:, cols["ritz_max"]]))],
                 monitor=dict(points=len(mon), steps=[int(x) for x in mon[:, 0]] if len(mon) else [],
                              max_err_est=float(np.nanmax(mon[:, 1:5])) if len(mon) and CASES[c]["method"] == "pppm_lanczos" else None,
                              budget=r["operator_budget"], S_kmin_max=float(mon[:, 7].max()) if len(mon) else None),
                 accuracy_warnings=st.get("accuracy_warnings") or [],
                 segments_contiguous=contiguous, chunk_ends=CHUNKS[c], chunk_ends_expected=want,
                 lock_check=dict(passed=LOCK[c].get("passed"), warnings=LOCK[c].get("warnings"),
                                 mount=LOCK[c].get("mount")),
                 hardware=res["environment"]["hardware"][c],
                 rng_state_final_sha=hashlib.sha256(rng_final.encode()).hexdigest()[:16],
                 cost=dict(**cost, step_s_median=float(np.median(walls)), step_s_p90=float(np.percentile(walls, 90)),
                           step_s_max=float(walls.max()), step_s_mean=float(walls.mean()),
                           init_s=cost["first_segment_init_wall_s"], import_s=cost["first_segment_import_wall_s"],
                           steps_in_cost_statistics=int(len(walls)),
                           output_s_per_step=sum_k["output_wall_s"] / steps_k, monitor_s_per_point=(
                               cost["monitor_wall_s"] / max(len(mon), 1)),
                           full_s_per_step=(sum_k["step_wall_s_sum"] + sum_k["monitor_wall_s"] + sum_k["output_wall_s"])
                           / steps_k,
                           run_cpu_s=cost["init_cpu_s"] + cost["cpu_s"],
                           run_wall_s=sum((s.get("phase") or "").startswith("segment") and s.get("process_wall_s") or 0
                                          for s in S[c].get("phases", [])) or None,
                           frame_bytes=int(X["Q"][0].nbytes + X["V"][0].nbytes),
                           raw_bytes_on_disk=int(sum(f.stat().st_size for f in (root / c / "run").rglob("*") if f.is_file()))),
                 case_summary=S[c])
        res["cases"][c] = e
        diffs[c] = dict(t=X["frame_step"] * r["dt"], dP=dP, T=D[:, cols["T_kin"]], U=D[:, cols["U_per_N"]])
    # fast vs dense at the same dt
    npz = {}
    for dt, (cf, cd) in PAIRS.items():
        if cf not in R or cd not in R:
            continue
        F, Dd = R[cf], R[cd]
        n = min(len(F["Q"]), len(Dd["Q"]))
        t = F["frame_step"][:n] * dt
        dq = np.linalg.norm(F["Q"][:n] - Dd["Q"][:n], axis=2)            # (frames, N)
        dv = np.linalg.norm(F["V"][:n] - Dd["V"][:n], axis=2)
        vrms = np.sqrt(np.mean(np.sum(Dd["V"][:n] ** 2, axis=2), axis=1))
        dq_rms, dv_rms = np.sqrt(np.mean(dq ** 2, axis=1)), np.sqrt(np.mean(dv ** 2, axis=1))
        dT = F["diag"][:n, 2] - Dd["diag"][:n, 2]
        dU = F["diag"][:n, 3] - Dd["diag"][:n, 3]
        at = {}
        for tt in (dt, 0.05, 0.1, 0.25, 0.5, 1.0):
            k = int(round(tt / dt))
            if k < n:
                at[f"{tt:g}"] = dict(dq_rms=float(dq_rms[k]), dq_max=float(dq[k].max()), dv_rms=float(dv_rms[k]),
                                     dv_max=float(dv[k].max()), dv_rel=float(dv_rms[k] / vrms[k]),
                                     dT=float(dT[k]), dU_per_N=float(dU[k]))
        with np.load(root / cf / "run" / "checkpoint.npz") as a, np.load(root / cd / "run" / "checkpoint.npz") as b:
            same_rng = str(a["rng_state"]) == str(b["rng_state"])
        res["pairs"][f"{dt:g}"] = dict(
            fast=cf, dense=cd, frames=n, same_seed=R[cf]["config"]["plan"]["seed"] == R[cd]["config"]["plan"]["seed"],
            same_final_rng_state=same_rng, identical_at_t0=bool(dq[0].max() == 0 and dv[0].max() == 0),
            same_force_method=R[cf]["config"]["resolved"]["force_method"] == R[cd]["config"]["resolved"]["force_method"],
            at=at, growth_rate_dq=_fit_growth(t, dq_rms), growth_rate_dv=_fit_growth(t, dv_rms),
            max_dq=float(dq.max()), max_dv=float(dv.max()), max_abs_dT=float(np.abs(dT).max()),
            max_abs_dU_per_N=float(np.abs(dU).max()))
        npz.update({f"t_{dt:g}": t, f"dq_rms_{dt:g}": dq_rms, f"dq_max_{dt:g}": dq.max(axis=1),
                    f"dv_rms_{dt:g}": dv_rms, f"dv_max_{dt:g}": dv.max(axis=1), f"dT_{dt:g}": dT, f"dU_{dt:g}": dU})
    for c in R:
        npz.update({f"dP_{c}": diffs[c]["dP"], f"T_{c}": diffs[c]["T"], f"U_{c}": diffs[c]["U"],
                    f"tt_{c}": diffs[c]["t"]})
    np.savez_compressed(out / "pilot_differences.npz", **npz)
    # Lanczos: monitor estimate vs actual error, and PPPM operator vs full periodic Gamma, on the fast runs
    t_lc = time.monotonic()
    res["lanczos_check"] = {c: lanczos_check(R[c], c) for c in R if CASES[c]["method"] == "pppm_lanczos"}
    res["lanczos_check_wall_s"] = time.monotonic() - t_lc
    # checks
    E = res["cases"]
    for c, e in E.items():
        cs = e["case_summary"]
        ph = [p for p in cs.get("phases", []) if "exit_code" in p]
        chk[c] = dict(
            complete=e["state"] == "complete" and e["steps"] == e["steps_expected"],
            every_step_frames=e["frames_contiguous"] and e["frames"] == e["steps_expected"] + 1,
            shared_initial_state=e["init_identical_to_shared_state"],
            finite=e["finite"],
            momentum_within_roundoff=e["momentum"]["within_roundoff_walk"],
            segments_contiguous=e["segments_contiguous"],
            checkpoint_cadence_in_all_segments=e["chunk_ends"] == e["chunk_ends_expected"],
            monitor_points_sampled=(e["monitor"]["points"] >= e["steps_expected"] // MONITOR_EVERY + 1),
            monitor_within_budget=(e["monitor"]["max_err_est"] is None or e["monitor"]["max_err_est"] <= e["monitor"]["budget"]),
            no_accuracy_warnings=not e["accuracy_warnings"],
            exit_codes_as_expected=bool(ph) and all(p["exit_code"] == p["expected_exit_code"] for p in ph),
            signal_stop_exercised=any(p["phase"].startswith("segment1") and p.get("exit_code") == tr.EXIT_INCOMPLETE
                                      and "sigusr1_sent_after_s" in p for p in ph),
            duplicate_refused_by_lock=any(p.get("duplicate_exit_code") == tr.EXIT_LOCKED for p in ph),
            restart_bitwise=bool(cs.get("restart", {}).get("passed")),
            lock_check=cs.get("lock_check_exit_code") == 0,
            lock_check_without_warnings=not e["lock_check"]["warnings"],
            single_thread_env=all(v == "1" for v in e["threads"].values()),
            single_os_thread=e["cost"]["os_threads"] == [1])
        if e["method"] == "pppm_lanczos":
            lc = res["lanczos_check"][c]
            chk[c]["true_lanczos_error_within_budget"] = bool(lc) and max(max(x["true_noise"], x["true_damp"]) for x in lc) <= e["monitor"]["budget"]
    for dt, pr in res["pairs"].items():
        chk[f"pair_{dt}"] = dict(same_seed=pr["same_seed"], same_final_rng_state=pr["same_final_rng_state"],
                                 identical_at_t0=pr["identical_at_t0"], same_force_method=pr["same_force_method"])
    seeds = {E[c]["dt"]: E[c]["seed"] for c in E}
    chk["cross_dt"] = dict(different_seeds_no_coupling_claimed=len(set(seeds.values())) == len(seeds))
    chk["all_passed"] = all(all(v.values()) for k, v in chk.items() if isinstance(v, dict)) and not missing
    res["analysis_wall_s"] = time.monotonic() - t_an
    res["estimates"] = estimates(E)
    (out / "pilot_results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    figures(npz, res, out)
    write_tables(res, out)
    write_report(res, out)
    print(f"analysis written to {out} (all checks passed: {chk['all_passed']})", flush=True)
    return 0 if chk["all_passed"] else 1


def estimates(E):
    """Cost per physical time unit from the measured per-step costs (this machine; re-measure on the cluster)."""
    est = {}
    for c, e in E.items():
        co, dt = e["cost"], e["dt"]
        spt = 1 / dt                                             # steps per time unit
        mon_prod = co["monitor_s_per_point"] * spt / 500 if e["method"] == "pppm_lanczos" else 0.0
        est[c] = dict(steps_per_time_unit=spt, step_s_mean=co["step_s_mean"],
                      integration_s_per_time_unit=co["step_s_mean"] * spt,
                      output_s_per_time_unit_every_step_frames=co["output_s_per_step"] * spt,
                      monitor_s_per_time_unit_every_500_steps=mon_prod,
                      core_s_per_time_unit=co["step_s_mean"] * spt + co["output_s_per_step"] * spt + mon_prod,
                      storage_mb_per_time_unit_every_step_frames=co["frame_bytes"] * spt / 1e6,
                      init_s=co["init_s"], peak_rss_mb=co["peak_rss_mb"])
        est[c]["core_h_per_100_time_units"] = est[c]["core_s_per_time_unit"] * 100 / 3600
        # REPORT.md plan: >= 10 replicas x (5 burn-in + 50 production) time units
        est[c]["core_h_10x55"] = est[c]["core_s_per_time_unit"] * 550 / 3600
        est[c]["wall_h_one_replica_55"] = (est[c]["core_s_per_time_unit"] * 55 + (co["init_s"] or 0)) / 3600
    return est


def figures(npz, res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for dt, sty in ((0.005, "-"), (0.01, "--")):
        k = f"{dt:g}"
        if f"t_{k}" not in npz:
            continue
        t = npz[f"t_{k}"]
        ax[0].semilogy(t[1:], npz[f"dq_rms_{k}"][1:], sty, label=f"rms, dt {k}")
        ax[0].semilogy(t[1:], npz[f"dq_max_{k}"][1:], sty, alpha=0.5, label=f"max, dt {k}")
        ax[1].semilogy(t[1:], npz[f"dv_rms_{k}"][1:], sty, label=f"rms, dt {k}")
        ax[1].semilogy(t[1:], npz[f"dv_max_{k}"][1:], sty, alpha=0.5, label=f"max, dt {k}")
    ax[0].set(xlabel="t", ylabel="|q_fast - q_dense| per particle", title="positions, fast vs dense (same noise)")
    ax[1].set(xlabel="t", ylabel="|v_fast - v_dense| per particle", title="velocities, fast vs dense (same noise)")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle(f"LJ law A HPC pilot, one initial state, t = 1 ({res['environment']['label']})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "pilot_fast_vs_dense.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for c in CASES:
        if f"tt_{c}" not in npz:
            continue
        t = npz[f"tt_{c}"]
        ls = "-" if "fast" in c else ":"
        ax[0].plot(t, npz[f"T_{c}"], ls, label=c)
        ax[1].plot(t, npz[f"U_{c}"], ls, label=c)
        ax[2].semilogy(t[1:], np.maximum(npz[f"dP_{c}"][1:], 1e-18), ls, label=c)
    ax[0].set(xlabel="t", ylabel="T_kin")
    ax[1].set(xlabel="t", ylabel="U / N")
    ax[2].set(xlabel="t", ylabel="|P(t) - P(0)|")
    for a in ax:
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pilot_diagnostics.png", dpi=110)
    plt.close(fig)


def _f(x, fmt=".3g"):
    return "–" if x is None else format(x, fmt)


def _mem_request(rss_mb):
    """--mem advice from a measured peak RSS: 1.5x plus 256 MB, rounded up to 0.5 GB, at least 1 GB."""
    return max(1.0, np.ceil((1.5 * rss_mb + 256) / 512) / 2)


def write_tables(res, out):
    E, est = res["cases"], res["estimates"]
    L = ["| | " + " | ".join(E) + " |", "|---|" + "---|" * len(E)]

    def row(name, fn):
        L.append(f"| {name} | " + " | ".join(fn(c, E[c]) for c in E) + " |")

    def lc_max(c, key):
        rows = res["lanczos_check"].get(c) or []
        return f"{max(x[key] for x in rows):.1e}" if rows else "–"
    row("O-step operator", lambda c, e: "full periodic Γ (lattice sum, k=0 kept), dense eigendecomposition"
        if e["method"] == "reference" else f"PPPM Γ_h (M {e['mesh_M']}, k=0 kept), Lanczos {e['ranks'][0]}/{e['ranks'][1]}")
    row("dt / steps / frames", lambda c, e: f"{e['dt']:g} / {e['steps']} / {e['frames']}")
    row("seed (noise stream)", lambda c, e: f"{e['seed']}")
    row("host / CPU", lambda c, e: f"{e['hardware'].get('host')} / {e['hardware'].get('cpu_model')}")
    row("state / all values finite", lambda c, e: f"{e['state']} / {'yes' if e['finite'] else 'NO'}")
    row("max \\|P(t)−P(0)\\| (1st half, 2nd half)", lambda c, e: f"{e['momentum']['max_dev']:.1e} "
        f"({e['momentum']['max_dev_first_half']:.1e}, {e['momentum']['max_dev_second_half']:.1e})")
    row("max \\|P−P0\\| / (ε Σ\\|p\\|); drift", lambda c, e: f"{e['momentum']['max_dev'] / e['momentum']['roundoff_per_step']:.1f}; "
        + ("none detectable (round-off)" if e["momentum"]["within_roundoff_walk"] else "**above round-off**"))
    row("T_kin: t=0 / mean / min–max", lambda c, e: f"{e['T_kin']['t0']:.4f} / {e['T_kin']['mean']:.4f} / "
        f"{e['T_kin']['min']:.3f}–{e['T_kin']['max']:.3f}")
    row("U/N: t=0 / mean / end", lambda c, e: f"{e['U_per_N']['t0']:.4f} / {e['U_per_N']['mean']:.4f} / "
        f"{e['U_per_N']['end']:.4f}")
    row("min pair distance", lambda c, e: f"{e['rmin']:.3f}")
    row("spectrum (Ritz) range", lambda c, e: f"{e['ritz'][0]:.3f} – {e['ritz'][1]:.2f}")
    row("monitor points / max error estimate (budget)", lambda c, e: f"{e['monitor']['points']} / "
        + (f"{e['monitor']['max_err_est']:.1e} ({e['monitor']['budget']:.0e})" if e["monitor"]["max_err_est"] is not None
           else "n/a (exact matrix functions)"))
    row("actual Lanczos error, max (noise, damping)", lambda c, e: "n/a" if e["method"] == "reference" else
        f"{lc_max(c, 'true_noise')}, {lc_max(c, 'true_damp')}")
    row("init (process start → first step) [s]", lambda c, e: f"{(e['cost']['import_s'] or 0) + (e['cost']['init_s'] or 0):.2f} "
        f"(imports {(e['cost']['import_s'] or 0):.2f})")
    row("integration step: median / p90 / max [s]", lambda c, e: f"{e['cost']['step_s_median']:.3f} / "
        f"{e['cost']['step_s_p90']:.3f} / {e['cost']['step_s_max']:.3f}")
    row("monitor per point [s]", lambda c, e: f"{e['cost']['monitor_s_per_point']:.3f}")
    row(f"output per step (frame every step, checkpoint every {CHECKPOINT_EVERY} steps) [s]",
        lambda c, e: f"{e['cost']['output_s_per_step']:.4f}")
    row("full per step (integration + monitor + output) [s]", lambda c, e: f"{e['cost']['full_s_per_step']:.3f}")
    row("run CPU [s] / run process wall [s]", lambda c, e: f"{e['cost']['run_cpu_s']:.1f} / {_f(e['cost']['run_wall_s'], '.1f')}")
    row("peak RSS [MB]", lambda c, e: f"{e['cost']['peak_rss_mb']:.0f}")
    row("segments / checkpoints at / restart bitwise / duplicate refused", lambda c, e: f"{e['cost']['segments']} / "
        f"{e['chunk_ends']} / {'yes' if res['checks'][c]['restart_bitwise'] else 'NO'} / "
        f"{'yes' if res['checks'][c]['duplicate_refused_by_lock'] else 'NO'}")
    row("threads (env / OS threads)", lambda c, e: f"{','.join(sorted(set(e['threads'].values())))} / {e['cost']['os_threads']}")
    row("raw output on disk [MB]", lambda c, e: f"{e['cost']['raw_bytes_on_disk'] / 1e6:.1f}")
    row("est. core-s per time unit (prod. monitor cadence)", lambda c, e: f"{est[c]['core_s_per_time_unit']:.0f}")
    t1 = "\n".join(L) + "\n"
    P = ["| dt | t | Δq rms | Δq max | Δv rms | Δv max | Δv rms / v rms | ΔT_kin | ΔU/N |", "|---|---|---|---|---|---|---|---|---|"]
    for dt, pr in res["pairs"].items():
        for tt, a in pr["at"].items():
            P.append(f"| {dt} | {tt} | {a['dq_rms']:.2e} | {a['dq_max']:.2e} | {a['dv_rms']:.2e} | {a['dv_max']:.2e} | "
                     f"{a['dv_rel']:.2e} | {a['dT']:+.1e} | {a['dU_per_N']:+.1e} |")
    t2 = "\n".join(P) + "\n"
    (out / "pilot_table.md").write_text("# HPC pilot: four cases\n\n" + res["environment"]["label"] + "\n\n" + t1
                                        + "\n## Fast minus dense, same dt, same initial state and noise\n\n" + t2)
    res["_tables"] = (t1, t2)


def write_report(res, out):
    E, chk, est, env = res["cases"], res["checks"], res["estimates"], res["environment"]
    t1, t2 = res.pop("_tables")
    local = not env["label"].startswith("HPC")
    lines = ["# LJ law A: HPC short pilot (dense reference vs fast PPPM-Lanczos, dt 0.005 / 0.01, t = 1)", "",
             f"**Where this ran: {env['label']}.**" + (
                 " No Slurm job id is attached to any segment. Every number below is from this machine, not from the "
                 "cluster; the cluster run is still to be done (commands in Section 9)." if local else ""), "",
             "Scope: one shared initial state, t = 1 per case. This pilot checks that the runs, the comparison and the "
             "restart machinery work and measures cost. It does not show statistical equivalence of the dynamics or "
             "long-time equilibrium.", ""]
    if res["failed_cases"]:
        lines += ["**Cases without analysable output** (fewer than two frames):", ""]
        for c, x in res["failed_cases"].items():
            lines.append(f"- {c}: state {x['state']} at step {x['step']}, stop reason {x['stop_reason']}, "
                         f"failure diagnostics {x['failure_diag']}; phases {[(p.get('phase'), p.get('exit_code')) for p in (x['phases'] or [])]}"
                         + (f"; lock check {x['lock_check']}" if x.get("lock_check") else ""))
        lines.append("")
    if not E:
        lines += [f"No case has analysable output; missing: {res['missing_cases']}."]
        (out / "pilot_report.md").write_text("\n".join(lines) + "\n")
        return
    any_e = next(iter(E.values()))
    fast_e = next((e for e in E.values() if e["method"] == "pppm_lanczos"), None)
    hw = sorted({(h.get("host"), h.get("cpu_model"), h.get("cores_visible"), h.get("platform"), h.get("python"),
                  h.get("numpy"), h.get("scipy"), str(h.get("blas"))) for h in env["hardware"].values()},
                key=str)
    commits = sorted({f"{g.get('commit')} (dirty={g.get('dirty')})" for g in env["git"].values()})
    lc_all = [x for v in res["lanczos_check"].values() for x in v]
    lines += [
        "## 1. Setup", "",
        f"- Model: `{any_e['config']}`, LJ, law {any_e['law']}, N {any_e['N']}, ρ* {any_e['model']['density']}, "
        f"kT {any_e['model']['kT']}, L {any_e['L']:.6f}, γ {any_e['model']['gamma']}, κ {any_e['model']['kappa']}, "
        f"r_ref {any_e['model']['r_ref']}; full periodic friction, k = 0 mode kept in both operators.",
        (f"- Fast: PPPM ξ {fast_e['pppm']['xi']}, s {fast_e['pppm']['s']}, η {fast_e['pppm']['eta']:.4f}, "
         f"p {fast_e['pppm']['p']} → mesh M {fast_e['mesh_M']}; Lanczos ranks noise/damping {fast_e['ranks']}; one "
         "Γ_h(q_half) for damping and noise." if fast_e else "- Fast: no fast case analysed."),
        "- Dense: full periodic lattice-sum Γ (`v2.LatticeFriction`: 33 near + 988 Chebyshev far images, tail "
        "≈ 1e-11 per pair, k = 0 coefficient retained), exact matrix functions by eigendecomposition on Range(Π) "
        "(`DenseRef`). Not an eigendecomposition of the PPPM Γ_h.",
        "- Both: BAOAB with the finite-time FDT O-step, the same runner (`toy_run.py`, `--method`), the same conservative "
        f"force (`{any_e['force_method']}` = `toy_dynamics.conservative_force_neighbor`).",
        f"- Initial state: `{res['init_state']}` = end of law A run `eq_A_lgv_s305` at t = 50 (Langevin-prepared "
        f"canonical start; REPORT.md §4: canonical starts need no burn-in). T_kin of this snapshot "
        f"{res['init_provenance']['T_kin']:.3f}, U/N {res['init_provenance']['U_per_N']:.4f}. Identical in all "
        "cases (checked bitwise).",
        "- Noise: fast and dense at the same dt share the seed, so they receive the same standard-normal input at "
        "every step (checked: identical final RNG state, identical state at t = 0). dt 0.005 and dt 0.01 use "
        f"different seeds ({SEEDS[0.005]}, {SEEDS[0.01]}) and are **not coupled**; no cross-dt pathwise comparison is "
        "made. (A shared seed would not couple different step sizes. The verified OU coupling of `lj075_coupled.py` "
        "makes the dt 0.01 input depend on Γ, so fast and dense at 0.01 would no longer get identical inputs.)",
        "- Output: positions (unwrapped), velocities and diagnostics every step; accuracy monitor every "
        f"{MONITOR_EVERY} steps; checkpoint every {CHECKPOINT_EVERY} steps counted from each segment's start, and at "
        "every stop (`--checkpoint-every-steps` is passed to every segment, since `toy_run.py --resume` takes the "
        "cadence from its command line).",
        f"- Threads: OPENBLAS/OMP/MKL_NUM_THREADS = 1; OS threads per process at segment end: "
        f"{sorted({tuple(e['cost']['os_threads']) for e in E.values()})}."]
    for h in hw:
        lines.append(f"- Hardware: host {h[0]}, {h[1]}, {h[2]} cores visible, {h[3]}; Python {h[4]}, NumPy {h[5]}, "
                     f"SciPy {h[6]}, BLAS {h[7]}.")
    if len({(h[0], h[1]) for h in hw}) > 1:
        lines.append("- **The cases ran on different hosts or CPU models**: cost ratios between cases mix hardware.")
    lines += [f"- Code: git {', '.join(commits)}.", "", "## 2. Workflow checks", "",
              "| check | " + " | ".join(k for k in chk if isinstance(chk[k], dict)) + " |"]
    keys = []
    for k, v in chk.items():
        if isinstance(v, dict):
            keys += [x for x in v if x not in keys]
    cols = [k for k in chk if isinstance(chk[k], dict)]
    lines.append("|---|" + "---|" * len(cols))
    for x in keys:
        lines.append(f"| {x} | " + " | ".join(("PASS" if chk[c][x] else "**FAIL**") if x in chk[c] else "" for c in cols) + " |")
    lw = {c: e["lock_check"] for c, e in E.items() if e["lock_check"].get("warnings")}
    lines += ["", f"All checks passed: **{chk['all_passed']}**." + (f" Missing cases: {res['missing_cases']}."
                                                                   if res["missing_cases"] else ""), "",
              "`lock_check` runs `hpc/lock_check.py` on the case directory; `duplicate_refused_by_lock` starts a second "
              "`--resume` on the same node while a segment runs. Neither shows that locks exclude processes on *other* "
              "nodes (Lustre `localflock`, NFS `nolock`); `lock_check.py` warns about such mounts"
              + (": " + "; ".join(f"{c}: {v['warnings']} (mount {v['mount']})" for c, v in lw.items()) if lw else
                 ", and it gave no warning here") + ".", "",
              "## 3. Four cases", "", t1, "",
              "Init = process start to the first step (imports, configuration, initial state, operator set-up, initial "
              "force), first segment. Integration step = one BAOAB step without monitor or output. Output = chunk and "
              "checkpoint files, status and log writes. Step, output and full per-step costs leave out segment 2, "
              "during which the duplicate lock test runs (on a one-CPU allocation it shares the core). Run CPU / wall: "
              "the three run segments (the twin restart check is extra)."]
    wide = [f"{c} (p90 / median {e['cost']['step_s_p90'] / e['cost']['step_s_median']:.1f})" for c, e in E.items()
            if e["cost"]["step_s_p90"] > 1.5 * e["cost"]["step_s_median"]]
    if wide:
        lines.append(f"Integration step times vary widely in {', '.join(wide)}; the cost estimates use the mean step. "
                     "The cause was not investigated (the dense step builds and diagonalises a 768 × 768 matrix with "
                     "large temporaries).")
    lines += ["", "## 4. Fast vs dense at the same dt", "", t2, ""]
    for dt, pr in res["pairs"].items():
        a1 = pr["at"].get("1", {})
        lines.append(f"- dt {dt}: identical at t = 0 ({pr['identical_at_t0']}); max over t ≤ 1: |Δq| {pr['max_dq']:.1e}, "
                     f"|Δv| {pr['max_dv']:.1e}, |ΔT_kin| {pr['max_abs_dT']:.1e}, |ΔU/N| {pr['max_abs_dU_per_N']:.1e}; "
                     f"at t = 1 the rms velocity difference is {_f(a1.get('dv_rel'), '.1e')} of the rms velocity; "
                     f"fitted exponential growth rate of the rms difference for t ≥ 0.1: Δq {_f(pr['growth_rate_dq'], '.2f')}, "
                     f"Δv {_f(pr['growth_rate_dv'], '.2f')} per time unit.")
    if lc_all:
        op = [x["operator_frobenius"] for x in lc_all]
        opn, opd = [x["operator_noise"] for x in lc_all], [x["operator_damp"] for x in lc_all]
        ln, ld = [x["true_noise"] for x in lc_all], [x["true_damp"] for x in lc_all]
        ratio = min(min(opn) / max(max(ln), 1e-300), min(opd) / max(max(ld), 1e-300))
        lines += ["", "The fast and dense O-steps differ by two approximations, measured at the fast runs' monitor "
                  "configurations and inputs (Section 6):",
                  f"- PPPM Γ_h vs full periodic Γ: {min(op):.1e}–{max(op):.1e} (relative Frobenius norm on Range(Π)); "
                  f"in the O-step matrix functions noise {min(opn):.1e}–{max(opn):.1e}, damping {min(opd):.1e}–{max(opd):.1e};",
                  f"- Lanczos truncation (ranks {fast_e['ranks'] if fast_e else '–'}) against exact f(Γ_h): noise ≤ "
                  f"{max(ln):.1e}, damping ≤ {max(ld):.1e}.",
                  (f"The operator term is the larger one by at least a factor {ratio:.0f} in both functions, so the "
                   "path difference comes mainly from the PPPM operator approximation." if ratio > 10 else
                   f"The two terms are within a factor {ratio:.1f} of each other in at least one function; the path "
                   "difference cannot be attributed to one of them."),
                  "", "The differences grow over t ≤ 1 (`pilot_fast_vs_dense.png`); the fitted exponential rates "
                  "above summarise the growth over t ∈ [0.1, 1] only and are not Lyapunov exponents. One initial state "
                  "and one noise path per dt: these numbers describe this path pair only and are not a statistical "
                  "comparison of the two methods."]
    lines += ["", "## 5. Momentum, temperature, potential energy", ""]
    for c, e in E.items():
        mo = e["momentum"]
        lines.append(f"- {c}: max |P(t) − P(0)| = {mo['max_dev']:.1e} (relative to √(NmkT) = {mo['scale_sqrt_NmkT']:.0f}: "
                     f"{mo['rel_max_dev']:.1e}); first / second half {mo['max_dev_first_half']:.1e} / "
                     f"{mo['max_dev_second_half']:.1e}; ε Σ|p| = {mo['roundoff_per_step']:.1e}. **{mo['verdict'][0].upper() + mo['verdict'][1:]}.** "
                     f"T_kin mean {e['T_kin']['mean']:.4f} (range {e['T_kin']['min']:.3f}–{e['T_kin']['max']:.3f}), "
                     f"U/N mean {e['U_per_N']['mean']:.4f}; all values finite: {e['finite']}.")
    p0n = float(np.linalg.norm(any_e["momentum"]["P0"]))
    lines += ["", f"|P(0)| = {p0n:.1e} (round-off of the source run). The O-step carries the mean momentum exactly and "
              "projects the friction update, and the pair forces cancel in pairs, so P changes only by floating-point "
              "round-off. ε Σ|p| is the round-off of forming Σp once; a deviation within ε Σ|p| √n (n steps) cannot be "
              "told apart from round-off, and no drift is detectable then. A systematic error of 2e-14 per step (0.15 ε Σ|p|) "
              "would exceed that scale within t = 1 at both step sizes. T_kin over t = 1 from one state is a single correlated sample; "
              "its mean is not a temperature test.", "", "## 6. Lanczos error monitor (fast runs)", ""]
    for c, rows in res["lanczos_check"].items():
        if not rows:
            continue
        lines.append(f"- {c}: {len(rows)} monitor points at steps {[x['step'] for x in rows]}. Monitor estimate "
                     f"|f_r − f_(r+e)| max: noise {max(x['est_noise'] for x in rows):.1e}, damping "
                     f"{max(x['est_damp'] for x in rows):.1e}. Actual error against dense f(Γ_h) at the same "
                     f"configurations and inputs: noise {max(x['true_noise'] for x in rows):.1e}, damping "
                     f"{max(x['true_damp'] for x in rows):.1e}. Budget {E[c]['monitor']['budget']:.0e}. "
                     f"Smallest eigenvalue of Γ_h / Γ: {min(x['lam_min_h'] for x in rows):.3f} / "
                     f"{min(x['lam_min_ref'] for x in rows):.3f}.")
    lines += ["", "The monitor points include steps after both restarts. The dense runs have no Lanczos step; their "
              "monitor rows hold S(k_min) only.", "", "## 7. Checkpoint save and restore", ""]
    for c, e in E.items():
        cs = e["case_summary"]
        rs = cs.get("restart", {})
        ph = {p["phase"]: p for p in cs.get("phases", []) if "exit_code" in p}
        s1 = ph.get("segment1_signal_stop", {})
        s2 = ph.get("segment2_step_limit", {})
        rst = rs.get("restart_steps") or ["?", "?"]
        lines.append(f"- {c}: segment 1 stopped by SIGUSR1 → exit {s1.get('exit_code')} at step {rst[0]}; segment 2 "
                     f"({SEGMENT2_STEPS}-step limit) exit {s2.get('exit_code')} at step {rst[1]}, duplicate resume during "
                     f"it exit {s2.get('duplicate_exit_code')} (3 = refused by the lock); final exit "
                     f"{[p['exit_code'] for k, p in ph.items() if k.startswith('segment3')]}; checkpoints at "
                     f"{e['chunk_ends']} (expected {e['chunk_ends_expected']}). Continuous twin to step "
                     f"{rs.get('twin_steps')} (exit {rs.get('twin_exit_code')}, past both restarts: "
                     f"{rs.get('covers_both_restarts')}): diagnostics {rs.get('diag_bitwise_except_wall')}, frames "
                     f"{rs.get('frames_bitwise')}, monitor rows {rs.get('monitor_bitwise')}, q/p "
                     f"{rs.get('q_p_at_twin_end_bitwise')} (bitwise)." + (f" {rs['reason']}" if rs.get("reason") else ""))
    lines += ["", "## 8. Cost and resource recommendation", "",
              "Cost per physical time unit, from the measured mean integration step, the measured output cost with a "
              "frame every step, and the monitor at the production cadence (every 500 steps):", "",
              "| case | steps / time unit | step [s] | core-s / time unit | storage MB / time unit (every-step frames) | "
              "core-h per 100 time units | core-h, 10 replicas × 55 | peak RSS [MB] | --mem |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c, x in est.items():
        lines.append(f"| {c} | {x['steps_per_time_unit']:.0f} | {x['step_s_mean']:.3f} | {x['core_s_per_time_unit']:.0f} | "
                     f"{x['storage_mb_per_time_unit_every_step_frames']:.1f} | {x['core_h_per_100_time_units']:.2f} | "
                     f"{x['core_h_10x55']:.1f} | {x['peak_rss_mb']:.0f} | {_mem_request(x['peak_rss_mb']):g}G |")
    f5, f10, d5 = est.get("fast_dt0.005"), est.get("fast_dt0.01"), est.get("dense_dt0.005")
    if f5:
        lines += ["", "Recommendation for the formal law A runs (fast method only"
                  + (f"; the dense reference costs about {d5['core_s_per_time_unit'] / f5['core_s_per_time_unit']:.0f}× "
                     "more per time unit at dt 0.005 and is for spot checks" if d5 else "") + "):", "",
                  f"- one replica = one single-core task, OPENBLAS/OMP/MKL threads 1; 5 + 50 time units at dt 0.005 "
                  f"≈ {f5['wall_h_one_replica_55']:.2f} h of compute on this CPU. Request `--time` ≈ 1.5 × the "
                  "cluster-measured estimate + 10 min, with `--signal=B:USR1@600` and `--resume` so a job stopped at the "
                  "limit continues;",
                  f"- `--mem={_mem_request(f5['peak_rss_mb']):g}G` for the fast runs (1.5 × peak RSS "
                  f"{f5['peak_rss_mb']:.0f} MB + 256 MB, rounded up)"
                  + (f"; `--mem={_mem_request(d5['peak_rss_mb']):g}G` for the dense reference (peak RSS "
                     f"{d5['peak_rss_mb']:.0f} MB, lattice set-up)" if d5 else "") + ";",
                  f"- storage with every-step frames: {f5['storage_mb_per_time_unit_every_step_frames']:.1f} MB per "
                  f"time unit at dt 0.005 ({f5['storage_mb_per_time_unit_every_step_frames'] * 55:.0f} MB per replica "
                  "of 55);",
                  f"- 10 replicas × 55 time units: ≈ {f5['core_h_10x55']:.1f} core-h at dt 0.005"
                  + (f", ≈ {f10['core_h_10x55']:.1f} core-h at dt 0.01" if f10 else "") +
                  " (dt 0.01 is not validated for production; REPORT_observable_dt.md);",
                  "- these numbers come from " + ("this local machine; the per-step time on the cluster's CPU must "
                                                  "replace them before sizing the array (rerun this pilot there)."
                                                  if local else "the cluster nodes of this pilot.")]
    lines += ["", "## 9. Running it on the cluster", "",
              "```bash",
              "cp hpc/cluster.env.example hpc/cluster.env   # fill in every FILL_ME",
              "hpc/submit.sh lj075-pilot --dry-run          # prints the five sbatch commands",
              "hpc/submit.sh lj075-pilot                    # 4 single-core case jobs + 1 analysis job (afterany)",
              "squeue -u $USER; sacct -j <ids> --format=JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,NodeList",
              "```", "",
              "The analysis job writes `$RUN_ROOT/lj075_pilot/<tag>/report/` (this report, `pilot_table.md`, "
              "`pilot_results.json`, figures); copy that directory back into `lj075_results/hpc_pilot/` for review.", "",
              "## 10. What this pilot does not show", "",
              "- No statistical equivalence of fast and dense dynamics and no dt conclusion: one initial state, one noise "
              "path per dt, t = 1.",
              "- No long-time equilibrium or stationarity: T_kin and U/N over t = 1 are single correlated samples.",
              "- dt 0.005 and dt 0.01 are not pathwise coupled; nothing here compares them pathwise.",
              "- Locking across nodes: the duplicate test runs on one node (see Section 2).",
              "- " + ("Nothing about the cluster: scheduler signals, the cluster file system's locking, its Python/BLAS "
                      "build and its step times are tested only when these jobs run there." if local else
                      "Single cluster run; node-to-node speed variation is not sampled."), "",
              f"Analysis wall time {res['analysis_wall_s']:.0f} s (Lanczos/operator check {res['lanczos_check_wall_s']:.0f} s)."]
    if (out / "sacct.txt").exists():
        lines += ["", "## Slurm accounting of the case jobs (sacct.txt)", "", "```",
                  (out / "sacct.txt").read_text().strip(), "```"]
    (out / "pilot_report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("case")
    a.add_argument("--case", required=True, choices=sorted(CASES))
    a.add_argument("--root", required=True)
    b = sub.add_parser("analyze")
    b.add_argument("--root", required=True)
    b.add_argument("--out", required=True)
    c = sub.add_parser("all")
    c.add_argument("--root", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "case":
        return run_case(args.case, args.root)
    if args.cmd == "analyze":
        return analyze(args.root, args.out)
    rcs = [run_case(k, args.root) for k in CASES]
    return max(rcs + [analyze(args.root, args.out)])


if __name__ == "__main__":
    sys.exit(main())
