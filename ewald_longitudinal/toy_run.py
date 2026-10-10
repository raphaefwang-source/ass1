#!/usr/bin/env python3
"""
Unified, restartable runner for the toy friction dynamics (local and HPC entry point).

    python3 toy_run.py --config costopt_hiacc --potential double_well --kernel A --N 512 --seed 101 \\
        --burn-in 140 --production 60 --out RUNS/double_well_A_N512_s101
    python3 toy_run.py --resume --out RUNS/double_well_A_N512_s101          # continue from the last checkpoint

Model and step:
- the step is that of toy_dynamics.run, B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2): finite-time FDT O-step of
  toy_dynamics.Thermostat("pppm_lanczos"), with damping and noise from the same Gamma_h;
- --method reference replaces only the O-step operator: the full periodic lattice-sum Gamma (v2.LatticeFriction,
  k = 0 retained) with exact (dense eigendecomposition) matrix functions, toy_dynamics.Thermostat("reference").
  Same conservative force, same noise stream: with the same --seed and initial state, a reference run and a
  pppm_lanczos run receive identical standard-normal inputs at every step. Cost about 2 s per step at N = 256;
  meant for short comparisons, not production;
- one standard-normal (N, 3) draw per step. With the same state and noise stream, the trajectory is bitwise that of
  toy_dynamics.run (test_toy_run.py);
- every parameter comes from a named configuration in toy_configs.py. The resolved set is printed at start-up and
  stored in config.json; nothing falls back to an old default.

Initial states (recorded in config.json; none of the constructed ones is marked equilibrated):
- v2_checkpoint  N = 64 only: the v2 restart state (after its burn-in, with the v2 dense lattice thermostat);
- replicate64    N = 64 k^3: k^3 periodic copies of a v2 N = 64 state, uniform jitter, fresh Maxwell momenta;
- fcc / sc       N = 4 n^3 / n^3: lattice at the configuration's density, uniform jitter, fresh Maxwell momenta;
- rsa            random sequential addition with minimum pair distance --init-dmin, fresh Maxwell momenta;
- --from-state   q, p from an .npz file (e.g. checkpoint_burnin_end.npz of another run). The source must carry its
                 state point (a run directory's config.json next to it, or keys L / kT / N in the file) and it must
                 match this configuration; v2_checkpoint and replicate64 exist only for the toy_v2 state point.

Output directory:
  config.json                  resolved parameters, provenance, init record, code version, hashes (written once)
  status.json                  state (running / incomplete / complete / failed), counters, segments, peak RSS
  run.log                      human-readable log (appended)
  init_state.npz               initial q, p
  checkpoint.npz               latest state: q, p, step, RNG state, physics hash (atomic replace)
  checkpoint_burnin_end.npz    state at the end of burn-in (kept)
  chunks/chunk_<end step>.npz  per checkpoint interval (start, end]: per-step diagnostics (DIAG_COLS), saved
                               frames (unwrapped Q, velocity V, step) and accuracy-monitor rows (MON_COLS);
                               chunk_000000000 holds the initial state.
                               On resume, chunks ending past the checkpoint are moved to chunks/stale/

Timing (status.json segments): init_wall_s (process entry to the first step, without the initial output),
step_wall_s_sum (integration steps), monitor_wall_s, output_wall_s (chunks, checkpoints, status, log), wall_s, cpu_s,
peak_rss_mb, os_threads (threads of the process at the end of the segment).

Accuracy monitor (every --monitor-every-steps, own RNG keyed by step, so the trajectory is unchanged; pppm_lanczos
only, the reference method has no Lanczos approximation and records S(k_min) alone):
- Lanczos error estimates of the noise sqrt-action and the damping action at the configured ranks:
  |f_r - f_(r+8)| / |f_(r+8)| (damping: r+4), overall and in the k = 2 pi / L plane-wave components;
- extreme Ritz values;
- max_k S(k) over the three k = 2 pi / L vectors: long-wavelength density fluctuations (clustering or phase
  separation change the friction spectrum and the rank requirement).
Warnings go to run.log and status.json when an estimate exceeds the operator budget or the smallest Ritz value falls
below the verified spectrum. They are estimates, not proofs.

Concurrency: an exclusive lock on <out>/.run.lock (run_lock.py) is taken before any run state is read, resumed or
modified and held until the process exits (the kernel releases it on any exit, including SIGKILL). A second process
for the same directory waits --lock-wait seconds, then exits 3 without changing anything.

Exit codes:
- 0   complete, including an already complete run;
- 3   the run directory is locked by another live process (nothing changed);
- 75  stopped early but resumable (signal, wall limit, segment limit);
- 2   physical or numerical failure (the last checkpoint is kept);
- 1   usage or configuration error.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                     # before NumPy is imported; recorded in config.json

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_T_IMPORT = time.monotonic()                           # before NumPy / SciPy imports
_T_MAIN = [None]

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import scipy  # noqa: E402
import run_lock  # noqa: E402
import toy_configs as tc  # noqa: E402
import toy_dynamics as td  # noqa: E402
from test_lanczos_fdt import lanczos, proj, tridiag  # noqa: E402

V2_CKPT = HERE / "toy_models" / "v2_results" / "restart_checkpoints"
VERIFICATION_DEFAULT = HERE / "cost_optimization_results" / "production_config_verification.json"
DIAG_COLS = ("step", "t", "T_kin", "U_per_N", "KE_per_N", "P_norm", "rmin", "step_wall_s", "ritz_min", "ritz_max")
MON_COLS = ("step", "noise_err_est", "noise_err_est_kmin", "damp_err_est", "damp_err_est_kmin", "ritz_min_mon",
            "ritz_max_mon", "S_kmin_max")
MON_EXTRA = (8, 4)                  # extra Lanczos steps of the error estimate (noise, damping)
EXIT_COMPLETE, EXIT_USAGE, EXIT_FAILED, EXIT_LOCKED, EXIT_INCOMPLETE = 0, 1, 2, 3, 75
METHODS = ("pppm_lanczos", "reference")
STOP = {"signal": None}


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_savez(path, **arrays):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _git_info():
    try:
        head = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return dict(commit=head, dirty=bool(dirty))
    except Exception as exc:                                          # git absent on the node is not fatal
        return dict(commit=None, dirty=None, error=str(exc))


def _steps(t, k, dt, what, required=False):
    """Time t or step count k -> integer steps (exactly one of them, or neither -> 0 unless required)."""
    if t is not None and k is not None:
        raise tc.ConfigError(f"give {what} as time or as steps, not both")
    if k is not None:
        return int(k)
    if t is None:
        if required:
            raise tc.ConfigError(f"{what} is required")
        return 0
    n = int(round(t / dt))
    if abs(n * dt - t) > 1e-9 * max(1.0, t):
        raise tc.ConfigError(f"{what} = {t} is not a multiple of dt = {dt}")
    return n


# ----------------------------------------------------------------------------
# initial states
# ----------------------------------------------------------------------------
def _maxwell(rng, N, kT, m):
    p = rng.normal(0.0, np.sqrt(m * kT), (N, 3))
    return p - p.mean(axis=0)


def _lattice(kind, N, L):
    if kind == "fcc":
        n = round((N / 4) ** (1 / 3))
        if 4 * n ** 3 != N:
            raise tc.ConfigError(f"fcc needs N = 4 n^3 (N = {N})")
        g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
        basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        return ((g[:, None, :] + basis[None]).reshape(-1, 3) + 0.25) * (L / n)
    n = round(N ** (1 / 3))
    if n ** 3 != N:
        raise tc.ConfigError(f"sc needs N = n^3 (N = {N})")
    g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    return (g + 0.5) * (L / n)


def initial_state(r, args):
    """(q, p, record). Constructed states are never marked equilibrated."""
    N, L, law, pot = r["N"], r["L"], r["law"], r["potential"]
    kT, m = r["model"]["kT"], r["model"]["mass"]
    rng = np.random.default_rng([args.seed, 2])
    method = args.init or "auto"
    if args.from_state:
        method = "from_state"
    elif method == "auto" and r["model"] != tc.STATE_POINTS["toy_v2"]:
        method = "fcc" if 4 * round((N / 4) ** (1 / 3)) ** 3 == N else "sc"
    elif method == "auto":
        k = round((N / 64) ** (1 / 3))
        if N == 64:
            method = "v2_checkpoint"
        elif 64 * k ** 3 == N:
            method = "replicate64"
        elif 4 * round((N / 4) ** (1 / 3)) ** 3 == N:
            method = "fcc"
        else:
            method = "sc"
    rec = dict(method=method, rng=f"numpy default_rng([seed, 2]) (seed {args.seed})", equilibrated=False)
    if method == "from_state":
        f = Path(args.from_state)
        sources = {}
        with np.load(f) as d:
            q, p = np.array(d["q"], float), np.array(d["p"], float)
            own = {}
            for k in ("L", "kT", "N"):
                if k in d.files:
                    if np.ndim(d[k]) != 0 or not np.issubdtype(d[k].dtype, np.number):
                        raise tc.ConfigError(f"--from-state {f}: key {k} is not a scalar number")
                    own[k] = float(d[k])
            if own:
                sources["file keys"] = own
            if "config" in d.files:                              # toy_run checkpoints embed their configuration
                er = json.loads(str(d["config"]))["resolved"]
                sources["embedded config"] = dict(L=er["L"], kT=er["model"]["kT"], N=er["N"])
        if q.shape != (N, 3):
            raise tc.ConfigError(f"--from-state has shape {q.shape}, expected ({N}, 3)")
        src_cfg = f.parent / "config.json"
        if src_cfg.exists():
            sr = json.loads(src_cfg.read_text())["resolved"]
            sources["adjacent config.json"] = dict(L=sr["L"], kT=sr["model"]["kT"], N=sr["N"],
                                                   source_config=sr["config"])
        if not any({"L", "kT"} <= set(v) for v in sources.values()):
            raise tc.ConfigError(f"--from-state {f}: no state-point record (keys L, kT in the file, an embedded config "
                                 "or a config.json next to it); refusing a state of unknown box and temperature")
        for name, v in sources.items():                        # every record present must agree with the target
            if "L" in v and abs(v["L"] - L) > 1e-6 * L or "kT" in v and abs(v["kT"] - kT) > 1e-9:
                raise tc.ConfigError(f"--from-state {f}: {name} gives L {v.get('L')}, kT {v.get('kT')}; this "
                                     f"configuration has L {L}, kT {kT}")
        meta = dict(sources)
        kin = float(np.sum((p - p.mean(0)) ** 2) / (m * 3 * (N - 1)))
        rec.update(source=str(f.resolve()), source_sha256=_sha256(f), source_state_point=meta,
                   source_kinetic_temperature=kin,
                   note="state supplied by the user; its equilibration status is the user's responsibility")
        return q, p, rec
    if method in ("v2_checkpoint", "replicate64") and r["model"] != tc.STATE_POINTS["toy_v2"]:
        raise tc.ConfigError(f"{method} is a toy_v2 state (L 5.5 per 64 particles, kT 0.7); not usable for "
                             f"{r['config']}")
    if method in ("v2_checkpoint", "replicate64"):
        src_seed = args.init_source_seed if args.init_source_seed is not None else args.seed
        f = V2_CKPT / f"{pot}_burn140" / "lattice" / f"{pot}_{law}" / f"seed_{src_seed}" / "restart.npz"
        if not f.exists():
            raise tc.ConfigError(f"no v2 checkpoint {f} (available source seeds 101-105); set --init-source-seed")
        with np.load(f) as d:
            q64, p64 = np.array(d["Q"], float), np.array(d["p"], float)
            meta = json.loads(str(d["metadata"]))
        src = dict(source=str(f.relative_to(HERE)), source_sha256=_sha256(f), source_seed=src_seed,
                   source_history=dict(burn_in_time=meta.get("burn_in_time"), total_prior_time=meta.get("total_prior_time"),
                                       production_time=meta.get("production_time"), thermostat="v2 dense lattice friction",
                                       box=meta.get("box"), n=meta.get("n")))
        if method == "v2_checkpoint":
            if N != 64:
                raise tc.ConfigError("v2_checkpoint is the N = 64 state; use replicate64 for N = 64 k^3")
            rec.update(src, note="v2 state after its burn-in under the v2 dense lattice thermostat; not "
                                 "re-equilibrated with this operator", equilibrated=False)
            return q64, p64, rec
        k = round((N / 64) ** (1 / 3))
        if 64 * k ** 3 != N:
            raise tc.ConfigError(f"replicate64 needs N = 64 k^3 (N = {N})")
        jit = 0.02 if args.init_jitter is None else args.init_jitter
        shifts = np.stack(np.meshgrid(*[np.arange(k)] * 3, indexing="ij"), -1).reshape(-1, 3) * 5.5
        q = ((q64 % 5.5)[None] + shifts[:, None]).reshape(-1, 3) + rng.uniform(-jit, jit, (N, 3))
        p = _maxwell(rng, N, kT, m)
        rec.update(src, copies=k ** 3, jitter_uniform_halfwidth=jit, momenta="fresh Maxwell at kT, total momentum 0",
                   note=f"{k}x{k}x{k} periodic copies of one N = 64 state: correlated, NOT equilibrated at N = {N}; "
                        "needs burn-in")
        return q % L, p, rec
    if method in ("fcc", "sc"):
        jit = 0.05 if args.init_jitter is None else args.init_jitter
        q = _lattice(method, N, L) + rng.uniform(-jit, jit, (N, 3))
        p = _maxwell(rng, N, kT, m)
        rec.update(jitter_uniform_halfwidth=jit, momenta="fresh Maxwell at kT, total momentum 0",
                   note=f"{method} lattice at density {r['model']['density']:.6f}: NOT equilibrated; needs burn-in")
        return q % L, p, rec
    if method == "rsa":
        dmin = 0.9 if args.init_dmin is None else args.init_dmin
        q = np.empty((N, 3))
        k, tries = 0, 0
        while k < N:
            tries += 1
            if tries > 10 ** 7:
                raise tc.ConfigError(f"rsa: could not place {N} particles with dmin {dmin}")
            x = rng.uniform(0, L, 3)
            d = q[:k] - x
            d -= L * np.round(d / L)
            if k == 0 or np.einsum("pa,pa->p", d, d).min() >= dmin * dmin:
                q[k] = x
                k += 1
        p = _maxwell(rng, N, kT, m)
        rec.update(min_distance=dmin, attempts=tries, momenta="fresh Maxwell at kT, total momentum 0",
                   note="random sequential addition: NOT equilibrated; needs burn-in")
        return q, p, rec
    raise tc.ConfigError(f"unknown init method {method}")


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------
def verification_status(r):
    key = f"{r['config']}_{r['law']}_N{r['N']}"
    if r.get("thermostat_method", "pppm_lanczos") == "reference":
        return dict(key=key, verified=None, not_applicable=True,
                    reason="reference method: full periodic lattice-sum Gamma with dense matrix functions (no PPPM, "
                           "no Lanczos); the PPPM/Lanczos verification does not apply")
    vf = tc.CONFIGS[r["config"]].get("verification_file")
    VERIFICATION = HERE / vf if vf else VERIFICATION_DEFAULT
    if not VERIFICATION.exists():
        return dict(key=key, verified=False, reason="no verification file")
    db = json.loads(VERIFICATION.read_text())
    if f"{key}_dt{r['dt']:g}" in db:                     # per-dt entries (lj075 configurations)
        key = f"{key}_dt{r['dt']:g}"
    e = db.get(key)
    if e is None:
        return dict(key=key, verified=False, reason="no verification entry for this config / law / N")
    if e.get("operator_hash") != tc.operator_hash(r):
        return dict(key=key, verified=False, reason="verification entry is for different parameters")
    lam = [x for x in (e.get("lam_min"), e.get("ritz_min")) if x is not None]
    return dict(key=key, verified=bool(e["passed"]), operator_hash=e["operator_hash"],
                operator_error=e.get("op_err", e.get("gv_rows_err")), budget=e["budget"],
                failed_checks=[k for k, v in e["checks"].items() if not v],
                lambda_min_verified=min(lam) if lam else None, states=list(e.get("per_state", {})))


def _os_threads():
    try:
        with open("/proc/self/status") as fh:
            return int(next(ln.split()[1] for ln in fh if ln.startswith("Threads:")))
    except (OSError, StopIteration, ValueError):
        return None


def build_thermostat(r):
    m = r["model"]
    if r.get("thermostat_method", "pppm_lanczos") == "reference":
        th = td.Thermostat("reference", r["L"], r["law"], m["gamma"], m["kappa"], m["r_ref"], r["dt"], m["kT"],
                           m["mass"])
        rec = th.record["operator"]
        if not (rec["images"] == "lattice" and th.lat.kwargs == dict(kernel=r["law"], gamma=m["gamma"],
                                                                      kappa=m["kappa"], r_ref=m["r_ref"])):
            raise RuntimeError(f"reference thermostat built with {th.lat.kwargs}, {rec}")
        return th
    th = td.Thermostat("pppm_lanczos", r["L"], r["law"], m["gamma"], m["kappa"], m["r_ref"], r["dt"], m["kT"],
                       m["mass"], pppm=r["pppm"], rank_noise=r["rank_noise"], rank_damp=r["rank_damp"],
                       pair_search=r["pair_search"])
    f = th.fast                                                      # what was actually built
    actual = dict(pair_search=f.pair_search, xi=f.K.xi, s=f.s, eta=f.eta, p=f.p, rank_noise=th.rank_noise,
                  rank_damp=th.rank_damp)
    want = dict(pair_search=r["pair_search"], rank_noise=r["rank_noise"], rank_damp=r["rank_damp"], **r["pppm"])
    if actual != want:
        raise RuntimeError(f"thermostat built with {actual}, configuration asked for {want}")
    return th


class Runner:
    def __init__(self, out, cfg, quiet=False):
        self.out, self.cfg, self.quiet = out, cfg, quiet
        r = cfg["resolved"]
        self.r = r
        self.n, self.L, self.dt = r["N"], r["L"], r["dt"]
        self.m = r["model"]["mass"]
        self.pot = r["potential"]
        mp = r["model"]
        self.fkw = dict(epsilon=mp["epsilon"], sigma=mp["sigma"], r_on=mp["r_on"], r_cut=mp["r_cut"])
        self.force_fn = td.FORCES[r["force_method"]]
        self.stop_rmin = mp["stop_rmin"]
        plan = cfg["plan"]
        self.burn, self.total = plan["burn_steps"], plan["burn_steps"] + plan["prod_steps"]
        self.save_every, self.save_burnin = plan["save_every"], plan["save_burnin"]
        self.method = r.get("thermostat_method", "pppm_lanczos")
        self.th = build_thermostat(r)
        self.monitor_every = cfg["runtime"].get("monitor_every_steps") or 0
        self.t_out = 0.0
        self.n_warn = 0
        (out / "chunks").mkdir(parents=True, exist_ok=True)

    def log(self, msg):
        line = f"[{_now()}] {msg}"
        if not self.quiet:
            print(line, flush=True)
        with open(self.out / "run.log", "a") as fh:
            fh.write(line + "\n")

    def frame_due(self, step):
        if step >= self.burn:
            return (step - self.burn) % self.save_every == 0
        return self.save_burnin and step % self.save_every == 0

    def diag_row(self, step, q, p, energy, rmin, wall, info):
        n, m = self.n, self.m
        kin = np.sum((p - p.mean(axis=0)) ** 2) / (2 * m)
        return [step, step * self.dt, 2 * kin / (3 * (n - 1)), energy / n, kin / n, np.linalg.norm(p.sum(axis=0)),
                rmin, wall, info.get("lam_min", np.nan), info.get("lam_max", np.nan)]

    def monitor(self, q, p, step):
        """Accuracy monitor row (MON_COLS); see the module docstring. Does not touch the noise RNG."""
        x = q % self.L
        if self.method == "reference":                            # exact matrix functions: nothing to estimate
            S = np.abs(np.exp(1j * 2 * np.pi / self.L * x).sum(axis=0)) ** 2 / self.n
            return [step] + [np.nan] * 6 + [float(S.max())]
        op = self.th.fast.gamma_h(q)
        rng = np.random.default_rng([self.cfg["plan"]["seed"], 3, int(step)])
        ph = np.exp(1j * 2 * np.pi / self.L * x)
        row = [step]
        ritz = [np.inf, 0.0]
        for v, fun, r, e in ((proj(rng.standard_normal(3 * self.n)), self.th.fn, self.th.rank_noise, MON_EXTRA[0]),
                             (proj(p.ravel()), self.th.fd, self.th.rank_damp, MON_EXTRA[1])):
            Lz = lanczos(op, v, r + e)
            f = {}
            for k in (r, r + e):
                s = min(k, Lz["s"])
                th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], s))
                ritz = [min(ritz[0], th[0]), max(ritz[1], th[-1])]
                f[k] = proj(Lz["nz"] * (Lz["Q"][:s].T @ (U @ (fun(th) * U[0]))))
            d, ref = f[r] - f[r + e], f[r + e]
            Jd, Jr = ph.T @ d.reshape(self.n, 3), ph.T @ ref.reshape(self.n, 3)
            row += [np.linalg.norm(d) / np.linalg.norm(ref),
                    float(np.max(np.linalg.norm(Jd, axis=1) / np.linalg.norm(Jr, axis=1)))]
        S = np.abs(np.exp(1j * 2 * np.pi / self.L * x).sum(axis=0)) ** 2 / self.n
        return row + [ritz[0], ritz[1], float(S.max())]

    def check_monitor(self, row):
        budget = self.r["operator_budget"]
        lam_ver = self.cfg["verification"].get("lambda_min_verified")
        msgs = []
        if max(row[1:5]) > budget:
            msgs.append(f"Lanczos error estimate {max(row[1:5]):.1e} > budget {budget:.0e} at step {int(row[0])}")
        if lam_ver is not None and row[5] < 0.9 * lam_ver:
            msgs.append(f"smallest Ritz value {row[5]:.3f} below the verified spectrum (lambda_min {lam_ver:.3f}; "
                        f"Lanczos error estimate here {max(row[1:5]):.1e}, budget {budget:.0e}) "
                        f"at step {int(row[0])}")
        if msgs:
            self.n_warn += 1
            if self.n_warn <= 20:
                for msg in msgs:
                    self.log("WARNING " + msg)
                st = json.loads((self.out / "status.json").read_text())
                self.write_status(accuracy_warnings=(st.get("accuracy_warnings") or []) + msgs)

    def write_status(self, **kw):
        t0 = time.perf_counter()
        st = json.loads((self.out / "status.json").read_text()) if (self.out / "status.json").exists() else {}
        st.update(kw, updated=_now(), total_steps=self.total, burn_steps=self.burn,
                  physics_hash=self.cfg["physics_hash"])
        _atomic_write_text(self.out / "status.json", json.dumps(st, indent=1, default=float) + "\n")
        self.t_out += time.perf_counter() - t0
        return st

    def checkpoint(self, q, p, step, rng, chunk_start, rows, frames, mon):
        """Chunk (start, step] first, then the checkpoint (both atomic): a kill in between only leaves a chunk ending
        past the checkpoint, which the next resume moves to stale/ and recomputes."""
        t0 = time.perf_counter()
        D = np.array(rows, float).reshape(-1, len(DIAG_COLS))
        fs = np.array([f[0] for f in frames], np.int64)
        Q = np.array([f[1] for f in frames], float).reshape(-1, self.n, 3)
        V = np.array([f[2] for f in frames], float).reshape(-1, self.n, 3)
        _atomic_savez(self.out / "chunks" / f"chunk_{step:09d}.npz", diag=D, diag_cols=np.array(DIAG_COLS),
                      frame_step=fs, Q=Q, V=V, monitor=np.array(mon, float).reshape(-1, len(MON_COLS)),
                      monitor_cols=np.array(MON_COLS), start=np.int64(chunk_start), end=np.int64(step))
        state = dict(q=q, p=p, step=np.int64(step), rng_state=np.array(json.dumps(rng.bit_generator.state)),
                     physics_hash=np.array(self.cfg["physics_hash"]), config=np.array(json.dumps(self.cfg, default=float)))
        _atomic_savez(self.out / "checkpoint.npz", **state)
        if step == self.burn and self.burn > 0:
            _atomic_savez(self.out / "checkpoint_burnin_end.npz", **state)
        self.t_out += time.perf_counter() - t0

    def run(self, q, p, rng, step, fresh, max_wall_s=None, max_segment_steps=None, previous_state=None):
        cpu0 = time.process_time()
        seg = dict(host=socket.gethostname(), pid=os.getpid(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                   slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"), start_step=int(step), start=_now(),
                   previous_state=previous_state)
        st = self.write_status(state="running", step=int(step))
        segments = st.get("segments", [])
        t_seg = time.monotonic()
        force, energy, rmin = self.force_fn(q, self.L, self.pot, **self.fkw)
        t_entry = _T_MAIN[0] if _T_MAIN[0] is not None else t_seg
        seg.update(import_wall_s=(t_entry - _T_IMPORT) if _T_MAIN[0] is not None else None,
                   init_wall_s=time.monotonic() - t_entry - self.t_out, init_cpu_s=time.process_time())
        self.t_out = 0.0
        t_mon = 0.0
        rows, frames, walls, mon = [], [], [], []
        chunk_start = step
        if fresh:
            rows.append(self.diag_row(step, q, p, energy, rmin, 0.0, {}))
            if self.frame_due(step):
                frames.append((step, q.copy(), p / self.m))
            if self.monitor_every:
                tm = time.perf_counter()
                mon.append(self.monitor(q, p, step))
                self.check_monitor(mon[-1])
                t_mon += time.perf_counter() - tm
            self.checkpoint(q, p, step, rng, chunk_start, rows, frames, mon)     # resumable from step 0
            rows, frames, mon = [], [], []
        last_ckpt_step, last_ckpt_wall = step, time.monotonic()
        stop_reason, rc = None, EXIT_COMPLETE
        dt, m = self.dt, self.m
        try:
            while step < self.total:
                t0 = time.perf_counter()
                xi = rng.standard_normal((self.n, 3))                 # same draw order as toy_dynamics.run
                p = p + 0.5 * dt * force
                q = q + 0.5 * dt / m * p
                p, info = self.th.o_step(q, p, xi)
                q = q + 0.5 * dt / m * p
                force, energy, rmin = self.force_fn(q, self.L, self.pot, **self.fkw)
                p = p + 0.5 * dt * force
                wall = time.perf_counter() - t0
                step += 1
                if not np.all(np.isfinite(p)) or rmin < self.stop_rmin:
                    raise RuntimeError(f"unstable close encounter at step {step}: rmin {rmin:.3f} < "
                                       f"{self.stop_rmin} or non-finite momenta (no force clipping is applied)")
                rows.append(self.diag_row(step, q, p, energy, rmin, wall, info))
                walls.append(wall)
                if self.frame_due(step):
                    frames.append((step, q.copy(), p / m))
                if self.monitor_every and step % self.monitor_every == 0:
                    tm = time.perf_counter()
                    mon.append(self.monitor(q, p, step))
                    self.check_monitor(mon[-1])
                    t_mon += time.perf_counter() - tm
                elapsed = time.monotonic() - t_seg
                if STOP["signal"]:
                    stop_reason = f"signal {STOP['signal']}"
                elif max_wall_s is not None and elapsed >= max_wall_s:
                    stop_reason = f"wall limit {max_wall_s:.0f} s"
                elif max_segment_steps is not None and step - seg["start_step"] >= max_segment_steps:
                    stop_reason = f"segment step limit {max_segment_steps}"
                due = (step - last_ckpt_step >= self.cfg["runtime"]["checkpoint_every_steps"]
                       or time.monotonic() - last_ckpt_wall >= 60 * self.cfg["runtime"]["checkpoint_every_minutes"]
                       or step == self.burn or step == self.total)
                if due or stop_reason:
                    self.checkpoint(q, p, step, rng, chunk_start, rows, frames, mon)
                    tl = time.perf_counter()
                    D = np.array(rows)
                    self.log(f"checkpoint step {step}/{self.total} ({'burn-in' if step <= self.burn else 'production'}) "
                             f"T {D[:, 2].mean():.4f} U/N {D[-1, 3]:+.4f} |P| {D[:, 5].max():.1e} rmin {D[:, 6].min():.3f} "
                             f"step {1e3 * np.median(D[:, 7]):.1f} ms, Ritz [{np.nanmin(D[:, 8]):.3f}, {np.nanmax(D[:, 9]):.1f}]"
                             + (f"; monitor err {max(max(x[1:5]) for x in mon):.1e} S(kmin) {mon[-1][7]:.1f}" if mon else ""))
                    self.t_out += time.perf_counter() - tl
                    st = self.write_status(state="running", step=int(step), last_checkpoint_step=int(step),
                                           peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
                    chunk_start, rows, frames, mon = step, [], [], []
                    last_ckpt_step, last_ckpt_wall = step, time.monotonic()
                    if stop_reason and step < self.total:
                        rc = EXIT_INCOMPLETE
                        break
        except Exception as exc:
            if rows:
                D = np.array(rows, float)
                _atomic_savez(self.out / f"failure_diag_{chunk_start:09d}.npz", diag=D, diag_cols=np.array(DIAG_COLS))
            stop_reason, rc = f"failure: {type(exc).__name__}: {exc}", EXIT_FAILED
            self.log(stop_reason + f" (last checkpoint kept at step {last_ckpt_step})")
        seg.update(end_step=int(last_ckpt_step), end=_now(), wall_s=time.monotonic() - t_seg,
                   cpu_s=time.process_time() - cpu0,
                   stop_reason=stop_reason or "complete",
                   median_step_wall_s=float(np.median(walls)) if walls else None,
                   steps_run=len(walls), step_wall_s_sum=float(np.sum(walls)), monitor_wall_s=t_mon,
                   output_wall_s=self.t_out, method=self.method, os_threads=_os_threads(),
                   peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        state = {EXIT_COMPLETE: "complete", EXIT_INCOMPLETE: "incomplete", EXIT_FAILED: "failed"}[rc]
        peak = max([s.get("peak_rss_mb") or 0 for s in segments] + [seg["peak_rss_mb"]])
        self.write_status(state=state, step=int(last_ckpt_step), last_checkpoint_step=int(last_ckpt_step),
                          segments=segments + [seg], peak_rss_mb=peak, exit_code=rc)
        self.log(f"segment end: {state} at step {last_ckpt_step}/{self.total} ({seg['stop_reason']}); exit {rc}")
        return rc


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=sorted(tc.CONFIGS))
    ap.add_argument("--potential", choices=tc.POTENTIALS)
    ap.add_argument("--kernel", choices=tc.KERNELS)
    ap.add_argument("--N", type=int)
    ap.add_argument("--seed", type=int, help="noise stream default_rng([seed, 1]); init stream default_rng([seed, 2])")
    ap.add_argument("--dt", type=float)
    ap.add_argument("--burn-in", type=float, help="burn-in time (an initial plan, not an equilibration guarantee)")
    ap.add_argument("--burn-in-steps", type=int)
    ap.add_argument("--production", type=float, help="production time")
    ap.add_argument("--production-steps", type=int)
    ap.add_argument("--save-every", type=float,
                    help="raw-frame interval in time (default: every step; vacf_sampling_results/REPORT.md)")
    ap.add_argument("--save-every-steps", type=int)
    ap.add_argument("--save-burnin", action="store_true", help="also save frames during burn-in")
    ap.add_argument("--init", default=None, choices=["auto", "v2_checkpoint", "replicate64", "fcc", "sc", "rsa"])
    ap.add_argument("--init-dmin", type=float, help="rsa minimum pair distance (default 0.9)")
    ap.add_argument("--init-source-seed", type=int, help="v2 checkpoint seed (101-105); default: --seed")
    ap.add_argument("--init-jitter", type=float)
    ap.add_argument("--from-state", help=".npz with q, p arrays (overrides --init)")
    ap.add_argument("--ranks", type=int, nargs=2, metavar=("NOISE", "DAMP"), help="override (unverified)")
    ap.add_argument("--method", choices=METHODS,
                    help="O-step operator: pppm_lanczos (default, production) or reference (full periodic "
                         "lattice-sum Gamma, dense matrix functions; short comparisons only)")
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument("--checkpoint-every-steps", type=int, default=2000)
    ap.add_argument("--checkpoint-every-minutes", type=float, default=15.0)
    ap.add_argument("--monitor-every-steps", type=int, default=500, help="accuracy monitor interval (0: off)")
    ap.add_argument("--max-wall-hours", type=float, help="stop cleanly (checkpoint, exit 75) after this wall time")
    ap.add_argument("--max-segment-steps", type=int, help="stop cleanly after this many steps in this invocation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="continue an existing run (starts fresh if none exists)")
    ap.add_argument("--retry-failed", action="store_true", help="resume a run whose last segment failed")
    ap.add_argument("--lock-wait", type=float, default=30.0,
                    help="seconds to wait for the run-directory lock before exiting 3 (default 30)")
    ap.add_argument("--unsafe-no-lock", action="store_true",
                    help="run without the directory lock (only where the file system cannot lock and no second "
                         "process can start)")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def new_config(args):
    need = [k for k in ("config", "potential", "kernel", "N", "seed") if getattr(args, k) is None]
    if need:
        raise tc.ConfigError("missing for a new run: " + ", ".join("--" + k for k in need))
    dt = tc.validated_dt(args.config, args.kernel) if args.dt is None else args.dt
    if dt is None:
        raise tc.ConfigError(f"{args.config} has no validated dt yet; pass --dt (with --allow-unverified)")
    r = tc.resolve(args.config, args.kernel, args.N, args.potential, dt=dt, rank_override=args.ranks,
                   allow_unverified=args.allow_unverified)
    r["thermostat_method"] = args.method or "pppm_lanczos"
    burn = _steps(args.burn_in, args.burn_in_steps, dt, "burn-in")
    prod = _steps(args.production, args.production_steps, dt, "production", required=True)
    se = _steps(args.save_every, args.save_every_steps if (args.save_every is not None or args.save_every_steps
                                                           is not None) else 1, dt, "save interval")
    if se < 1 or prod < 1:
        raise tc.ConfigError("production and save interval must be >= 1 step")
    init = "from_state" if args.from_state else (args.init or "auto")
    plan = dict(seed=args.seed, burn_steps=burn, prod_steps=prod, save_every=se, save_burnin=bool(args.save_burnin),
                init=init, init_source_seed=args.init_source_seed, init_jitter=args.init_jitter,
                from_state=args.from_state, burn_in_time=burn * dt, production_time=prod * dt,
                save_interval_time=se * dt,
                burn_in_note="initial run plan, not an equilibration guarantee; check the diagnostics")
    return r, plan


def check_resume_args(args, cfg):
    """On --resume, any physics or plan argument that is given must equal the stored value."""
    r, plan = cfg["resolved"], cfg["plan"]
    dt = r["dt"]
    stored = dict(config=r["config"], potential=r["potential"], kernel=r["law"], N=r["N"], seed=plan["seed"], dt=dt)
    diff = [f"{k}: given {getattr(args, k)!r}, stored {v!r}" for k, v in stored.items()
            if getattr(args, k) is not None and getattr(args, k) != v]
    for what, t, k, key in (("burn-in", args.burn_in, args.burn_in_steps, "burn_steps"),
                            ("production", args.production, args.production_steps, "prod_steps"),
                            ("save interval", args.save_every, args.save_every_steps, "save_every")):
        if t is not None or k is not None:
            if _steps(t, k, dt, what) != plan[key]:
                diff.append(f"{what}: given {_steps(t, k, dt, what)} steps, stored {plan[key]}")
    if args.method is not None and args.method != r.get("thermostat_method", "pppm_lanczos"):
        diff.append(f"method: given {args.method}, stored {r.get('thermostat_method', 'pppm_lanczos')}")
    if args.ranks is not None and list(args.ranks) != [r["rank_noise"], r["rank_damp"]]:
        diff.append(f"ranks: given {args.ranks}, stored {[r['rank_noise'], r['rank_damp']]}")
    if diff:
        raise tc.ConfigError("--resume with arguments that differ from config.json: " + "; ".join(diff))


def load_run(out):
    """Concatenate the chunks of a run (by end step): dict(diag, cols, frame_step, Q, V, config, status)."""
    out = Path(out)
    files = sorted((out / "chunks").glob("chunk_*.npz"), key=lambda f: int(f.stem.split("_")[1]))
    D, fs, Q, V, M = [], [], [], [], []
    for f in files:
        with np.load(f) as c:
            D.append(c["diag"])
            M.append(c["monitor"] if "monitor" in c.files else np.zeros((0, len(MON_COLS))))
            fs.append(c["frame_step"])
            Q.append(c["Q"])
            V.append(c["V"])
    cfg = json.loads((out / "config.json").read_text())
    st = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    n = cfg["resolved"]["N"]
    return dict(diag=np.concatenate(D) if D else np.zeros((0, len(DIAG_COLS))), cols=list(DIAG_COLS),
                frame_step=np.concatenate(fs) if fs else np.zeros(0, np.int64),
                Q=np.concatenate(Q) if Q else np.zeros((0, n, 3)), V=np.concatenate(V) if V else np.zeros((0, n, 3)),
                monitor=np.concatenate(M) if M else np.zeros((0, len(MON_COLS))), monitor_cols=list(MON_COLS),
                config=cfg, status=st)


def main(argv=None):
    _T_MAIN[0] = time.monotonic()
    args = parse(argv)
    STOP["signal"] = None
    out = Path(args.out)
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, lambda s, f: STOP.__setitem__("signal", signal.Signals(s).name))
    lock = None
    try:
        out.mkdir(parents=True, exist_ok=True)              # the lock file lives in the run directory
        if not args.unsafe_no_lock:
            lock = run_lock.RunLock(out)
            try:
                got = lock.acquire(args.lock_wait)
            except run_lock.LockUnsupported as exc:
                lock = None
                raise tc.ConfigError(f"{exc}; refusing to run without a lock. Use a lock-capable directory (see "
                                     "hpc/lock_check.py) or --unsafe-no-lock if no second process can start")
            if not got:
                lock = None
                print(f"{out}: locked by another live process (holder note: {run_lock.holder_info(out)}); this "
                      f"process changes nothing and exits {EXIT_LOCKED}", file=sys.stderr, flush=True)
                return EXIT_LOCKED
        fresh = not (out / "config.json").exists()
        if not fresh and not args.resume:
            raise tc.ConfigError(f"{out} already holds a run; pass --resume to continue it or choose a new --out")
        if fresh:
            if any(x.name != run_lock.LOCK_NAME for x in out.iterdir()):
                raise tc.ConfigError(f"{out} is not empty but has no config.json; refusing to write into it")
            r, plan = new_config(args)
            out.mkdir(parents=True, exist_ok=True)
            q, p, init_rec = initial_state(r, args)
            cfg = dict(resolved=r, plan=plan, init=init_rec, physics_hash=tc.physics_hash(r),
                       operator_hash=tc.operator_hash(r), verification=verification_status(r),
                       runtime=dict(checkpoint_every_steps=args.checkpoint_every_steps,
                                    checkpoint_every_minutes=args.checkpoint_every_minutes,
                                    monitor_every_steps=args.monitor_every_steps),
                       noise_rng=f"numpy default_rng([seed, 1]) = PCG64 (seed {args.seed})",
                       command=[sys.executable] + list(sys.argv if argv is None else ["toy_run.py"] + list(argv)),
                       state_point=tc.CONFIGS[r["config"]].get("state_point", "toy_v2"),
                       code=dict(git=_git_info(), python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__, host=socket.gethostname(), created=_now(),
                                 threads={v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                                          "MKL_NUM_THREADS")}),
                       diag_cols=list(DIAG_COLS))
            runner = Runner(out, cfg, args.quiet)
            cfg["operator_record"] = runner.th.record["operator"]
            _atomic_write_text(out / "config.json", json.dumps(cfg, indent=1, default=float) + "\n")
            _atomic_savez(out / "init_state.npz", q=q, p=p)
            rng, step = np.random.default_rng([args.seed, 1]), 0
        else:
            cfg = json.loads((out / "config.json").read_text())
            check_resume_args(args, cfg)
            st = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
            if st.get("state") == "complete":
                print(f"{out}: already complete ({st.get('step')} steps); nothing to do", flush=True)
                return EXIT_COMPLETE
            unclean = st.get("state") == "running"            # the lock is ours, so that process is gone
            if st.get("state") == "failed" and not args.retry_failed:
                raise tc.ConfigError(f"{out} failed earlier ({st.get('segments', [{}])[-1].get('stop_reason')}); the "
                                     "same state and noise would fail again. Inspect it; --retry-failed to try anyway")
            cfg["runtime"].update(checkpoint_every_steps=args.checkpoint_every_steps,
                                  checkpoint_every_minutes=args.checkpoint_every_minutes)   # monitor cadence is kept
            with np.load(out / "checkpoint.npz") as ck:
                if str(ck["physics_hash"]) != cfg["physics_hash"]:
                    raise tc.ConfigError("checkpoint physics hash differs from config.json")
                q, p, step = ck["q"].copy(), ck["p"].copy(), int(ck["step"])
                rng = np.random.default_rng()
                rng.bit_generator.state = json.loads(str(ck["rng_state"]))
            stale = [c for c in (out / "chunks").glob("chunk_*.npz") if int(c.stem.split("_")[1]) > step]
            if stale:
                sd = out / "chunks" / "stale" / datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                sd.mkdir(parents=True)
                for c in stale:
                    shutil.move(str(c), sd / c.name)
            if cfg["plan"]["burn_steps"] + cfg["plan"]["prod_steps"] <= step:
                raise tc.ConfigError("checkpoint is at the end but status is not complete; inspect status.json")
            runner = Runner(out, cfg, args.quiet)
            if unclean:
                runner.log("previous segment ended without a clean shutdown (status was 'running'); resuming from "
                           f"the last checkpoint at step {step}")
            if stale:
                runner.log(f"moved {len(stale)} chunk(s) past the checkpoint to {sd.relative_to(out)}")
        r = cfg["resolved"]
        if fresh or not args.quiet:
            v = cfg["verification"]
            lines = [f"run {out} ({'new' if fresh else 'resume at step %d' % step})",
                     f"  config {r['config']}: {r['description']}",
                     f"  potential {r['potential']}, kernel {r['law']}, N {r['N']}, L {r['L']:.6f}, density "
                     f"{r['model']['density']:.6f}, dt {r['dt']}, kT {r['model']['kT']}, gamma {r['model']['gamma']}, "
                     f"kappa {r['model']['kappa']}, r_ref {r['model']['r_ref']}",
                     f"  O-step method {r.get('thermostat_method', 'pppm_lanczos')}"]
            orec = cfg["operator_record"]
            if r.get("thermostat_method", "pppm_lanczos") == "reference":
                lines += [f"  full periodic lattice-sum Gamma: {orec['near_images']} near + {orec['far_images']} far "
                          f"images (Chebyshev degree {orec['chebyshev_degree']}), tail estimate "
                          f"{orec['tail_estimate_per_pair']:.1e} per pair, k=0 mode retained "
                          f"(ghat0/(3V) = {orec['zero_mode_per_pair']:.4f} per pair); dense matrix functions",
                          f"  conservative force {r['force_method']}",
                          f"  static verification: not applicable ({v.get('reason')})"]
            else:
                lines += [f"  PPPM xi {r['pppm']['xi']} s {r['pppm']['s']} eta {r['pppm']['eta']} p {r['pppm']['p']} -> "
                          f"M {orec['M']}, eta_actual {orec['eta_actual']:.4f}, rc {orec['rc']:.3f}, kc {orec['kc']:.3f}, "
                          f"k0 retained {orec['k0_retained']}",
                          f"  pair search {r['pair_search']}, conservative force {r['force_method']}",
                          f"  Lanczos ranks noise {r['rank_noise']} / damping {r['rank_damp']} "
                          f"({r['rank_provenance']['rule']}; verified at this N: "
                          f"{r['rank_provenance']['verified_at_this_N']})",
                          f"  static verification {v['key']}: {'PASS' if v['verified'] else 'NOT VERIFIED'}"
                          + (f" (operator error {v['operator_error']:.2e} <= {v['budget']:.0e})" if v.get("verified")
                             else f" ({v.get('reason', v.get('failed_checks'))})")]
            lines += [
                     f"  plan: burn-in {cfg['plan']['burn_steps']} steps (t {cfg['plan']['burn_in_time']}), production "
                     f"{cfg['plan']['prod_steps']} steps (t {cfg['plan']['production_time']}), frames every "
                     f"{cfg['plan']['save_every']} steps; init {cfg['init']['method']} (equilibrated: "
                     f"{cfg['init']['equilibrated']})",
                     f"  hashes: physics {cfg['physics_hash']}, operator {cfg['operator_hash']}; code "
                     f"{cfg['code']['git'].get('commit')} dirty={cfg['code']['git'].get('dirty')}; threads "
                     f"{cfg['code']['threads']}"]
            lines += [f"  WARNING: {w}" for w in r["warnings"]]
            for ln in lines:
                runner.log(ln)
        mw = None if args.max_wall_hours is None else 3600 * args.max_wall_hours
        return runner.run(q, p, rng, step, fresh, max_wall_s=mw, max_segment_steps=args.max_segment_steps,
                          previous_state=None if fresh else st.get("state"))
    except tc.ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr, flush=True)
        return EXIT_USAGE
    finally:
        if lock is not None:
            lock.release()


if __name__ == "__main__":
    sys.exit(main())
