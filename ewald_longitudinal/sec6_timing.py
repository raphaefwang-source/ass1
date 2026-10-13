#!/usr/bin/env python3
"""
Section 6, full-step equal-accuracy timing (former experiment 10).

One complete ideal-gas step A(dt/2) O(dt) A(dt/2) at the production settings of sec6_config.json (spatial set,
dt, noise and damping ranks), timed component by component on one machine, one process, one thread:
    q_half                  position half update
    setup: pairs            neighbour pairs within r_c (cKDTree, periodic)
    setup: weights          real-space pair weights theta_S(r) e^{-xi^2 r^2}/r^2
    setup: stencil          B-spline stencil indices and weights (mesh.flat)
    setup: degree           PPPM degree term (spread of the constant field + 6 inverse FFTs)
    damping / noise         Lanczos: Gamma_h actions (real-space part and PPPM part timed separately),
                            reorthogonalisation, tridiagonal eigenproblem + function evaluation, other vector work
    update                  projection, mean momentum, position update
One-time preprocessing (mesh, influence function) is listed separately and excluded from the step.
Warm-up steps are run first; then repeated timed steps, reported as median and spread. Gamma_h calls are counted.

Baseline at small and medium N: the same periodic model by direct real-space summation of the unsplit kernel over
all images |r + nL| <= R_d (pair list from a periodic replica tree), with the same Lanczos, ranks and dt. R_d is
chosen so that its relative operator error on the Experiment 1 configurations is comparable to that of the
production PPPM set (stage `direct_accuracy`). The dense eigendecomposition cost is listed separately (`dense`).

    python3 sec6_timing.py direct_accuracy
    python3 sec6_timing.py time [--N ...] [--method pppm direct]
    python3 sec6_timing.py dense
    python3 sec6_timing.py report
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sec6_config  # noqa: E402
from sec6_common import Box, f_noise, f_damp, tridiag, proj, MASS, KAPPA  # noqa: E402
from test_lanczos_fdt import GammaH, real_pairs  # noqa: E402
from sec6_dynamics import initial_state, noise_stream  # noqa: E402

SCRIPT = "sec6_timing.py"
OUT = sec6_config.RESULTS / "timing"
N_TIME = (256, 512, 1024, 2048, 4000, 6000, 8000, 11000, 16000)
N_DIRECT = (256, 512, 1024, 2048, 4000)
WARMUP, REPEATS = 2, 5
STATE_TRAJ = 900                                     # initial-state index used for timing (not used elsewhere)


def cpu_model():
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


# ----------------------------------------------------------------------------
# timed operators
# ----------------------------------------------------------------------------
class Timer(dict):
    def add(self, k, t):
        self[k] = self.get(k, 0.0) + t


class TimedPPPM:
    """Same arithmetic as test_lanczos_fdt.GammaH.matvec, with the real-space and PPPM parts timed separately."""

    def __init__(self, box, x, tm):
        t0 = time.perf_counter()
        self.i, self.j, self.d = real_pairs(x, box.L, box.rc)
        tm.add("setup_pairs", time.perf_counter() - t0)
        t0 = time.perf_counter()
        r = np.linalg.norm(self.d, axis=1)
        self.w = box.K.theta_S_scaled_closed(r) * np.exp(-(box.K.xi * r) ** 2) / r ** 2
        tm.add("setup_weights", time.perf_counter() - t0)
        t0 = time.perf_counter()
        self.mesh = box.mesh
        self.idx, self.wt = box.mesh.flat(x)
        tm.add("setup_stencil", time.perf_counter() - t0)
        t0 = time.perf_counter()
        self.D = box.mesh.degree(self.idx, self.wt)
        tm.add("setup_degree", time.perf_counter() - t0)
        self.N, self.tm, self.n_actions = len(x), tm, 0
        self.n_pairs = len(self.i)

    def matvec(self, v):
        t0 = time.perf_counter()
        X = v.reshape(self.N, 3)
        f = (self.w * np.einsum("pa,pa->p", self.d, X[self.i] - X[self.j]))[:, None] * self.d
        out = np.stack([np.bincount(self.i, f[:, c], self.N) - np.bincount(self.j, f[:, c], self.N)
                        for c in range(3)], axis=1)
        t1 = time.perf_counter()
        out += self.mesh.graph_action(self.idx, self.wt, X, self.D)
        t2 = time.perf_counter()
        self.tm.add("action_real", t1 - t0)
        self.tm.add("action_mesh", t2 - t1)
        self.n_actions += 1
        return out.ravel()


def direct_pairs(x, L, R):
    """Directed (target j, source i, displacement d = x_j - x_i - nL) over all images with 0 < |d| <= R."""
    N = len(x)
    nimg = int(np.ceil(R / L))
    sh = np.arange(-nimg, nimg + 1)
    S = np.stack(np.meshgrid(sh, sh, sh, indexing="ij"), axis=-1).reshape(-1, 3) * L
    ext = (x[None, :, :] + S[:, None, :]).reshape(-1, 3)
    src = np.tile(np.arange(N), len(S))
    tree_e, tree_o = cKDTree(ext), cKDTree(x)
    sdm = tree_o.sparse_distance_matrix(tree_e, R, output_type="ndarray")
    keep = sdm["v"] > 0
    j, e = sdm["i"][keep], sdm["j"][keep]
    return j, src[e], x[j] - ext[e]


class TimedDirect:
    """Direct real-space sum of the unsplit kernel K(r) = e^{-kappa r}/r r r^T/r^2 over |r + nL| <= R_d."""

    def __init__(self, box, x, tm, R):
        t0 = time.perf_counter()
        self.j, self.i, self.d = direct_pairs(x, box.L, R)
        tm.add("setup_pairs", time.perf_counter() - t0)
        t0 = time.perf_counter()
        r = np.linalg.norm(self.d, axis=1)
        self.w = np.exp(-KAPPA * r) / r ** 3
        tm.add("setup_weights", time.perf_counter() - t0)
        self.N, self.tm, self.n_actions, self.n_pairs = len(x), tm, 0, len(self.j)

    def matvec(self, v):
        t0 = time.perf_counter()
        X = v.reshape(self.N, 3)
        f = (self.w * np.einsum("pa,pa->p", self.d, X[self.j] - X[self.i]))[:, None] * self.d
        out = np.stack([np.bincount(self.j, f[:, c], self.N) for c in range(3)], axis=1)
        self.tm.add("action_real", time.perf_counter() - t0)
        self.n_actions += 1
        return out.ravel()


def lanczos_timed(op, z, smax, tm, prefix):
    """Arithmetic identical to test_lanczos_fdt.lanczos (full two-pass reorthogonalisation, Pi after each action)."""
    n = z.size
    smax = min(smax, n)
    Q = np.zeros((smax + 1, n))
    al, be = np.zeros(smax), np.zeros(smax)
    t0 = time.perf_counter()
    nz = np.linalg.norm(z)
    Q[0] = z / nz
    s_done = smax
    t_other = time.perf_counter() - t0
    for k in range(smax):
        w = op.matvec(Q[k])
        t0 = time.perf_counter()
        w = proj(w)
        al[k] = Q[k] @ w
        w = w - al[k] * Q[k] - (be[k - 1] * Q[k - 1] if k > 0 else 0.0)
        t1 = time.perf_counter()
        for _ in range(2):
            w -= Q[:k + 1].T @ (Q[:k + 1] @ w)
        t2 = time.perf_counter()
        w = proj(w)
        be[k] = np.linalg.norm(w)
        brk = be[k] <= 1e-13 * abs(al[:k + 1]).max()
        if not brk:
            Q[k + 1] = w / be[k]
        t3 = time.perf_counter()
        t_other += (t1 - t0) + (t3 - t2)
        tm.add(f"{prefix}_reorth", t2 - t1)
        if brk:
            s_done = k + 1
            break
    tm.add(f"{prefix}_vector", t_other)
    return dict(Q=Q[:s_done + 1], al=al[:s_done], be=be[:s_done], nz=nz, s=s_done)


def apply_fun(Lz, fun, tm, prefix):
    t0 = time.perf_counter()
    s = Lz["s"]
    th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], s))
    c = U @ (fun(th) * U[0])
    t1 = time.perf_counter()
    out = proj(Lz["nz"] * (Lz["Q"][:s].T @ c))
    t2 = time.perf_counter()
    tm.add(f"{prefix}_tridiag_feval", t1 - t0)
    tm.add(f"{prefix}_combine", t2 - t1)
    return out, float(th[0]), float(th[-1])


def timed_step(box, q, p, xi, dt, rn, rd, method, R_d=None):
    tm = Timer()
    N = box.N
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    qh = (q + 0.5 * dt / MASS * p) % box.L
    tm.add("q_half", time.perf_counter() - t0)
    op = TimedPPPM(box, qh, tm) if method == "pppm" else TimedDirect(box, qh, tm, R_d)
    t0 = time.perf_counter()
    pm = p.mean(axis=0)
    pp = proj(p.ravel())
    zz = proj(xi.ravel())
    tm.add("update", time.perf_counter() - t0)
    Ld = lanczos_timed(op, pp, min(rd, 3 * N - 3), tm, "damping")
    damp, dmin, dmax = apply_fun(Ld, f_damp(dt), tm, "damping")
    Ln = lanczos_timed(op, zz, min(rn, 3 * N - 3), tm, "noise")
    noise, nmin, nmax = apply_fun(Ln, f_noise(dt), tm, "noise")
    t0 = time.perf_counter()
    p_new = pm[None, :] + (proj(damp) + proj(noise)).reshape(N, 3)
    q_new = (qh + 0.5 * dt / MASS * p_new) % box.L
    tm.add("update", time.perf_counter() - t0)
    total = time.perf_counter() - t_all
    tm["total"] = total
    tm["accounted"] = sum(v for k, v in tm.items() if k not in ("total",))
    return q_new, p_new, dict(times=dict(tm), n_actions=op.n_actions, n_pairs=op.n_pairs,
                              ritz_min=min(dmin, nmin), ritz_max=max(dmax, nmax))


# ----------------------------------------------------------------------------
def stage_direct_accuracy(args):
    """Relative operator error of the direct real-space sum at R_d against the Experiment 1 references."""
    sys.path.insert(0, str(HERE))
    import exp1_operator_action as E
    from verify_periodic_graph_psd import laplacian
    cfgs = E.load_configs(sets=("legacy", "main", "check"))
    radii = (15.0, 16.0, 17.0, 18.0, 19.0, 20.0)
    rows = []
    for c in cfgs:
        ref = E.load_ref(c["name"])
        pi, pj, d = E.pair_geometry(c["x"], c["L"])
        lad = E.lattice_ladder(d, c["L"], radii, E.theta_full)
        for R, Kp in zip(radii, lad):
            G = laplacian(Kp, pi, pj, c["N"])
            rows.append(dict(name=c["name"], set=c["set"], family=c["family"], N=c["N"], R_d=R,
                             err_rel=E.specnorm_sym(G - ref["G"]) / float(ref["norm2"])))
        print(c["name"], ["%.1e" % r["err_rel"] for r in rows[-len(radii):]], flush=True)
    t7 = [r for r in csv.DictReader(open(HERE / "exp1_results" / "action.csv"))
          if r["params"] == sec6_config.production()["spatial_set"] and r["set"] in ("legacy", "main", "check")]
    t7max = max(float(r["err_spec_rel"]) for r in t7)
    chosen = None
    for R in radii:
        if max(r["err_rel"] for r in rows if r["R_d"] == R) <= t7max:
            chosen = R
            break
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "direct_accuracy.json").write_text(json.dumps(dict(
        provenance=sec6_config.provenance(SCRIPT), production_pppm_err_max_same_configs=t7max, R_d=chosen,
        rule="smallest R_d whose largest relative operator error over the 14 legacy/main/check configurations does "
             "not exceed that of the production PPPM set on the same configurations", rows=rows), indent=1) + "\n")
    print("chosen R_d", chosen, "production PPPM max", t7max)


def stage_time(args):
    pr = sec6_config.production()
    rn, rd = pr["noise_rank"], pr["damping_rank"]
    if rn is None or rd is None:
        raise SystemExit("production ranks not set")
    dt = pr["dt"]
    acc = json.loads((OUT / "direct_accuracy.json").read_text()) if (OUT / "direct_accuracy.json").exists() else {}
    R_d = acc.get("R_d")
    rows = []
    f = OUT / "timing_raw.json"
    old = json.loads(f.read_text())["rows"] if f.exists() and not args.fresh else []
    for N in args.N or N_TIME:
        for method in args.method:
            if method == "direct" and (N not in N_DIRECT or R_d is None):
                continue
            t0 = time.perf_counter()
            box = Box(N, pr["spatial_set"]) if method == "pppm" else Box(N, pr["spatial_set"])
            t_pre = time.perf_counter() - t0
            q, p = initial_state(N, box.L, STATE_TRAJ)
            xis = noise_stream(N, STATE_TRAJ, WARMUP + REPEATS)
            steps = []
            for n in range(WARMUP + REPEATS):
                q, p, info = timed_step(box, q, p, xis[n], dt, rn, rd, method, R_d)
                if n >= WARMUP:
                    steps.append(info)
            tot = np.array([s["times"]["total"] for s in steps])
            comp = {k: float(np.median([s["times"].get(k, 0.0) for s in steps])) for k in steps[0]["times"]}
            row = dict(N=N, method=method, L=box.L, box=box.describe(), R_d=R_d if method == "direct" else None,
                       noise_rank=rn, damping_rank=rd, dt=dt, warmup=WARMUP, repeats=REPEATS,
                       total_median=float(np.median(tot)), total_min=float(tot.min()), total_max=float(tot.max()),
                       spread_rel=float((tot.max() - tot.min()) / np.median(tot)), components_median=comp,
                       n_actions=[s["n_actions"] for s in steps], n_pairs=int(np.median([s["n_pairs"] for s in steps])),
                       ritz_min=min(s["ritz_min"] for s in steps), ritz_max=max(s["ritz_max"] for s in steps),
                       t_preprocess=t_pre, per_N_logN=float(np.median(tot) / (N * np.log(N))),
                       machine=dict(cpu=cpu_model(), host=platform.node(), threads=os.environ.get("OMP_NUM_THREADS")))
            rows.append(row)
            print(f"[{time.strftime('%H:%M:%S')}] {method} N={N}: {row['total_median']:.3f}s "
                  f"(spread {row['spread_rel']:.2f}) actions {row['n_actions'][0]} pairs {row['n_pairs']} "
                  f"pre {t_pre:.1f}s", flush=True)
            keep = [r for r in old if not (r["N"] == N and r["method"] == method)]
            old = keep
            OUT.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(dict(provenance=sec6_config.provenance(SCRIPT, True), rows=old + rows), indent=1)
                         + "\n")


def stage_dense(args):
    """Cost of the dense route at small N (assembly of Gamma_h by the mode-space pair matrix + eigendecomposition)."""
    from sec6_common import DenseRef
    pr = sec6_config.production()
    rows = []
    for N in (64, 128, 256, 512, 1024):
        box = Box(N, pr["spatial_set"])
        q, p = initial_state(N, box.L, STATE_TRAJ)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            op = box.gamma(q)
            G = op.dense()
            t1 = time.perf_counter()
            DenseRef(G, N)
            t2 = time.perf_counter()
            ts.append((t1 - t0, t2 - t1))
        rows.append(dict(N=N, assemble_median=float(np.median([a for a, b in ts])),
                         eig_median=float(np.median([b for a, b in ts]))))
        print(rows[-1], flush=True)
    (OUT / "dense_cost.json").write_text(json.dumps(dict(provenance=sec6_config.provenance(SCRIPT), rows=rows,
                                                         machine=dict(cpu=cpu_model())), indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["direct_accuracy", "time", "dense", "report"])
    ap.add_argument("--N", type=int, nargs="*")
    ap.add_argument("--method", nargs="*", default=["pppm", "direct"])
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    {"direct_accuracy": stage_direct_accuracy, "time": stage_time, "dense": stage_dense,
     "report": lambda a: __import__("sec6_timing_report").main()}[args.stage](args)


if __name__ == "__main__":
    main()
