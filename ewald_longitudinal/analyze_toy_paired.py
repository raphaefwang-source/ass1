#!/usr/bin/env python3
"""
Seed-paired comparison of the fast operator (PPPM + Lanczos 40/16) with the coupled dense reference for the
production_t60 runs of test_toy_dynamics.py.

Pairing: run i of both methods starts from the same state and uses the same Gaussian stream (seed i + 3000), so
the analysis is done on d_i = fast_i - reference_i even after the paths have separated (they are not treated as
independent samples). With n = 5 seeds:
    mean d,  SEM = std(d_i, ddof=1)/sqrt(n),  Student-t 95% CI = mean d +/- t_{0.975, n-1} SEM,
    relative difference = mean d / |mean(reference_i)|,  paired t statistic = mean d / SEM (df = n - 1),
    sign count (how many d_i > 0) and the exact two-sided sign-test p-value.
A non-significant result is NOT a statement of equivalence; the CI says which differences remain compatible.

Outputs (toy_dynamics_results/production_t60/paired/):
    paired_summary.csv   full-run quantities (D Green-Kubo and D MSD at max lag 3, T, U/N, tau_T, RDF peak, VACF min)
    lag_scan.csv         D_GK and D_MSD for max lag 0.5, 1, 2, 3, plus Green-Kubo contributions of lag segments
    window_summary.csv   four t = 15 windows of each run: D_GK, D_MSD, T, U/N, and a per-seed trend across windows
    operator_along_trajectories.csv   Gamma_h vs dense reference on saved frames of both methods (raw files needed)
    status.json          which stages ran; raw-dependent stages are skipped (explicitly) when raw files are missing
Cache use: paired_summary and lag_scan read only toy_dynamics_results/production_t60/observables/ (per-seed
observables) and observables/run_diagnostics.json (per-run T and U means from the raw diagnostics). These are built
from the raw files when available and committed, so these two stages run without raw trajectories.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
TAG = "production_t60"
RES = HERE / "toy_dynamics_results" / TAG
OBS = RES / "observables"
RAW = HERE / "toy_dynamics_raw" / TAG
MANIFEST = HERE / "toy_dynamics_results" / "raw_trajectory_manifest.json"
OUT = RES / "paired"
CASES = ("lj_A", "lj_B", "double_well_A", "double_well_B")
METHODS = ("reference", "pppm_lanczos")
SEEDS = (101, 102, 103, 104, 105)
LAGS = (0.5, 1.0, 2.0, 3.0)
GK_SEGMENTS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0))
N_WINDOWS = 4


# ----------------------------------------------------------------------------
# Paired statistics
# ----------------------------------------------------------------------------
def paired(ref, fast):
    ref, fast = np.asarray(ref, float), np.asarray(fast, float)
    d = fast - ref
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    sem = sd / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    tstat = mean / sem if sem > 0 else np.nan
    npos = int((d > 0).sum())
    nzero = int((d == 0).sum())
    sign_p = stats.binomtest(npos, n - nzero, 0.5).pvalue if n - nzero > 0 else np.nan
    mref = ref.mean()
    aref = abs(mref)                                   # relative to |reference| so the sign is that of d (U < 0)
    return dict(n=n, mean_ref=mref, mean_fast=fast.mean(), mean_diff=mean, sd_diff=sd, sem_diff=sem, t_crit_95=tc,
                ci95_low=mean - tc * sem, ci95_high=mean + tc * sem, rel_mean_diff=mean / aref,
                rel_ci95_low=(mean - tc * sem) / aref, rel_ci95_high=(mean + tc * sem) / aref,
                paired_t=tstat, p_two_sided=2 * stats.t.sf(abs(tstat), n - 1) if np.isfinite(tstat) else np.nan,
                n_diff_positive=npos, n_diff_zero=nzero, sign_test_p=sign_p,
                **{f"ref_s{s}": r for s, r in zip(SEEDS, ref)}, **{f"fast_s{s}": f for s, f in zip(SEEDS, fast)},
                **{f"diff_s{s}": x for s, x in zip(SEEDS, d)})


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


# ----------------------------------------------------------------------------
# Estimators (same definitions as toy_models/ensemble.py and test_toy_dynamics.run_scalars)
# ----------------------------------------------------------------------------
def d_green_kubo(t, vacf_norm, v2, tmax):
    k = int(round(tmax / (t[1] - t[0])))
    return v2 / 3 * np.trapezoid(vacf_norm[:k + 1], t[:k + 1])


def d_msd(t, msd, tmax):
    k = int(round(tmax / (t[1] - t[0])))
    sel = (t[:k + 1] >= tmax / 2)
    return np.polyfit(t[:k + 1][sel], msd[:k + 1][sel], 1)[0] / 6


def gk_segment(t, vacf_norm, v2, a, b):
    sel = (t >= a - 1e-12) & (t <= b + 1e-12)
    return v2 / 3 * np.trapezoid(vacf_norm[sel], t[sel])


def load_obs(case, method, seed):
    f = OBS / case / f"{method}_seed{seed}.npz"
    if not f.exists():
        raise SystemExit(f"Missing cached observables {f}. Rebuild with test_toy_dynamics.py --stage report "
                         f"--tag {TAG} (requires the raw trajectories).")
    with np.load(f) as z:
        return {k: z[k] for k in z.files}


def raw_file(case, method, seed):
    return RAW / case / f"{method}_seed{seed}.npz"


def build_diag_cache():
    """Per-run T and U means from the raw diagnostics (needs raw files); written once and committed."""
    path = OBS / "run_diagnostics.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    changed = False
    for case in CASES:
        for m in METHODS:
            for s in SEEDS:
                key = f"{case}/{m}_seed{s}"
                if key in cache:
                    continue
                f = raw_file(case, m, s)
                if not f.exists():
                    raise SystemExit(f"{path} lacks {key} and the raw file {f} is missing; cannot build the cache.")
                with np.load(f, allow_pickle=False) as tr:
                    d = tr["diagnostics"]
                half = len(d) // 2
                cache[key] = dict(T_mean=float(d[:, 1].mean()), U_per_particle=float(d[:, 2].mean()),
                                  U_first_half=float(d[:half, 2].mean()), U_second_half=float(d[half:, 2].mean()),
                                  frames=int(len(d)), source=str(f.relative_to(HERE)))
                changed = True
    if changed:
        path.write_text(json.dumps(cache, indent=1) + "\n")
    return cache


def load_diag_cache():
    path = OBS / "run_diagnostics.json"
    if path.exists():
        cache = json.loads(path.read_text())
        if all(f"{c}/{m}_seed{s}" in cache for c in CASES for m in METHODS for s in SEEDS):
            return cache
    return build_diag_cache()


# ----------------------------------------------------------------------------
# Stage 1 and 2: from cache only
# ----------------------------------------------------------------------------
def full_run_quantities(case, method, seed, diag):
    o = load_obs(case, method, seed)
    t, v2 = o["lag_time"], float(o["mean_squared_speed"])
    dg = diag[f"{case}/{method}_seed{seed}"]
    return dict(D_green_kubo=d_green_kubo(t, o["vacf"], v2, 3.0), D_msd=d_msd(t, o["msd"], 3.0),
                T_mean=dg["T_mean"], U_per_particle=dg["U_per_particle"],
                tau_T=float(np.trapezoid(o["hydro_T"].mean(axis=1), t)), rdf_peak=float(o["rdf"].max()),
                vacf_min=float(o["vacf"].min()))


def stage_paired(diag):
    rows = []
    for case in CASES:
        q = {m: [full_run_quantities(case, m, s, diag) for s in SEEDS] for m in METHODS}
        for name in q["reference"][0]:
            r = dict(case=case, quantity=name, max_lag=3.0 if name in ("D_green_kubo", "D_msd", "tau_T") else "")
            r.update(paired([x[name] for x in q["reference"]], [x[name] for x in q["pppm_lanczos"]]))
            rows.append(r)
    write_csv(OUT / "paired_summary.csv", rows)
    return rows


def stage_lag_scan():
    rows = []
    for case in CASES:
        obs = {m: [load_obs(case, m, s) for s in SEEDS] for m in METHODS}
        for tmax in LAGS:
            for name, fn in (("D_green_kubo", lambda o: d_green_kubo(o["lag_time"], o["vacf"], float(o["mean_squared_speed"]), tmax)),
                             ("D_msd", lambda o: d_msd(o["lag_time"], o["msd"], tmax))):
                r = dict(case=case, quantity=name, max_lag=tmax)
                r.update(paired([fn(o) for o in obs["reference"]], [fn(o) for o in obs["pppm_lanczos"]]))
                rows.append(r)
        for a, b in GK_SEGMENTS:
            fn = lambda o: gk_segment(o["lag_time"], o["vacf"], float(o["mean_squared_speed"]), a, b)
            r = dict(case=case, quantity=f"GK_contribution_{a:g}_to_{b:g}", max_lag=b)
            r.update(paired([fn(o) for o in obs["reference"]], [fn(o) for o in obs["pppm_lanczos"]]))
            rows.append(r)
    write_csv(OUT / "lag_scan.csv", rows)
    return rows


# ----------------------------------------------------------------------------
# Stage 3: time windows (raw trajectories required)
# ----------------------------------------------------------------------------
def fft_corr_sum(x, nlag):
    """sum_k x(k) x(k+m) over the time axis 0 for m = 0..nlag, summed over all other axes (zero-padded FFT)."""
    T = x.shape[0]
    n = 1 << int(np.ceil(np.log2(2 * T)))
    f = np.fft.rfft(x, n=n, axis=0)
    c = np.fft.irfft(f * f.conj(), n=n, axis=0)[:nlag + 1]
    return c.reshape(nlag + 1, -1).sum(axis=1)


def vacf_msd_window(Q, V, dt, nlag):
    """All-origin estimators in one window: <v(0).v(t)> and MSD(t), averaged over origins and particles."""
    T, N = V.shape[0], V.shape[1]
    cnt = (T - np.arange(nlag + 1)) * N
    cvv = fft_corr_sum(V, nlag) / cnt
    x2 = (Q ** 2).reshape(T, -1).sum(axis=1)                      # sum over particles and components
    cs = np.concatenate([[0.0], np.cumsum(x2)])
    m = np.arange(nlag + 1)
    s1 = (cs[T - m] + (cs[T] - cs[m]))                             # sum_k x(k)^2 + x(k+m)^2 over k < T - m
    msd = (s1 - 2 * fft_corr_sum(Q, nlag)) / cnt
    t = dt * m
    d_gk = np.trapezoid(cvv, t) / 3
    sel = t >= t[-1] / 2
    d_ms = np.polyfit(t[sel], msd[sel], 1)[0] / 6
    return d_gk, d_ms


def rel(f):
    try:
        return str(f.relative_to(HERE))
    except ValueError:
        return str(f)


def raw_ok():
    files = [raw_file(c, m, s) for c in CASES for m in METHODS for s in SEEDS]
    missing = [rel(f) for f in files if not f.exists()]
    if missing:
        return False, dict(reason="raw trajectories missing", missing=missing)
    man = {r["path"]: r["sha256"] for r in json.loads(MANIFEST.read_text())["runs"]}
    key = lambda f: str(Path("toy_dynamics_raw") / TAG / f.relative_to(RAW))
    bad = [rel(f) for f in files if man.get(key(f)) != hashlib.sha256(f.read_bytes()).hexdigest()]
    if bad:
        return False, dict(reason="raw trajectories do not match the sha256 manifest", mismatched=bad)
    return True, dict(files=len(files), verified_against=str(MANIFEST.relative_to(HERE)))


def stage_windows(max_lag=3.0):
    rows, per = [], {}
    for case in CASES:
        for m in METHODS:
            for s in SEEDS:
                with np.load(raw_file(case, m, s), allow_pickle=False) as tr:
                    Q, V, t, dg = tr["Q"], tr["V"], tr["t"], tr["diagnostics"]
                dt = float(t[1] - t[0])
                nlag = int(round(max_lag / dt))
                edges = np.linspace(0, len(t) - 1, N_WINDOWS + 1).round().astype(int)
                edges[-1] = len(t)
                for w in range(N_WINDOWS):
                    a, b = edges[w], edges[w + 1]
                    d_gk, d_ms = vacf_msd_window(Q[a:b], V[a:b], dt, nlag)
                    per[(case, m, s, w)] = dict(D_green_kubo=d_gk, D_msd=d_ms, T_mean=float(dg[a:b, 1].mean()),
                                                U_per_particle=float(dg[a:b, 2].mean()), t_start=float(t[a]),
                                                t_end=float(t[b - 1]))
                d_gk, d_ms = vacf_msd_window(Q, V, dt, nlag)
                per[(case, m, s, "full")] = dict(D_green_kubo=d_gk, D_msd=d_ms, T_mean=float(dg[:, 1].mean()),
                                                 U_per_particle=float(dg[:, 2].mean()), t_start=float(t[0]),
                                                 t_end=float(t[-1]))
    for case in CASES:
        for name in ("D_green_kubo", "D_msd", "T_mean", "U_per_particle"):
            for w in list(range(N_WINDOWS)) + ["full"]:
                ref = [per[(case, "reference", s, w)][name] for s in SEEDS]
                fast = [per[(case, "pppm_lanczos", s, w)][name] for s in SEEDS]
                x0 = per[(case, "reference", SEEDS[0], w)]
                r = dict(case=case, quantity=name, window=w, t_start=x0["t_start"], t_end=x0["t_end"],
                         estimator="all-origin FFT, max lag 3.0" if name.startswith("D_") else "frame mean")
                r.update(paired(ref, fast))
                # seed-to-seed SEM of each method on its own (drift of the quantity across windows)
                r["sem_ref_over_seeds"] = float(np.std(ref, ddof=1) / np.sqrt(len(ref)))
                r["sem_fast_over_seeds"] = float(np.std(fast, ddof=1) / np.sqrt(len(fast)))
                rows.append(r)
            # trend across windows: per-seed least-squares slope (per window) of d_i, then paired-t over seeds
            idx = np.arange(N_WINDOWS)
            for label, getter in (("diff", lambda s, w: per[(case, "pppm_lanczos", s, w)][name] - per[(case, "reference", s, w)][name]),
                                  ("reference", lambda s, w: per[(case, "reference", s, w)][name]),
                                  ("fast", lambda s, w: per[(case, "pppm_lanczos", s, w)][name])):
                slopes = np.array([np.polyfit(idx, [getter(s, w) for w in idx], 1)[0] for s in SEEDS])
                sem = slopes.std(ddof=1) / np.sqrt(len(slopes))
                tc = stats.t.ppf(0.975, len(slopes) - 1)
                rows.append(dict(case=case, quantity=name, window=f"slope_per_window_of_{label}",
                                 estimator="per-seed LS slope over windows 0-3, then mean over seeds",
                                 n=len(slopes), mean_diff=slopes.mean(), sem_diff=sem, t_crit_95=tc,
                                 ci95_low=slopes.mean() - tc * sem, ci95_high=slopes.mean() + tc * sem,
                                 paired_t=slopes.mean() / sem, p_two_sided=2 * stats.t.sf(abs(slopes.mean() / sem), len(slopes) - 1),
                                 **{f"diff_s{s}": v for s, v in zip(SEEDS, slopes)}))
    write_csv(OUT / "window_summary.csv", rows)
    return rows


# ----------------------------------------------------------------------------
# Stage 4: operator check on configurations visited by the dynamics (raw trajectories required)
# ----------------------------------------------------------------------------
def stage_operator_along_trajectories(every=5.0):
    """Gamma_h (fast PPPM, production set) vs the dense full-periodic reference on saved frames of both methods."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE / "toy_models"))
    import prl_toy_models as v2
    import radial_kernels as rk
    from test_lanczos_fdt import DenseRef
    rows = []
    for case in CASES:
        pot, law = case.rsplit("_", 1)
        lat = v2.LatticeFriction(5.5, kernel=law, gamma=0.5, kappa=0.7, r_ref=1.3)
        fast = rk.FastFriction(5.5, rk.toy_kernel(law, 0.5, 0.7, 1.3, 0.7), 4.10, 0.7, 7)
        errs, lmin_ref, lmin_h = [], [], []
        for m in METHODS:
            for s in SEEDS:
                with np.load(raw_file(case, m, s), allow_pickle=False) as tr:
                    Q, t = tr["Q"], tr["t"]
                for k in np.arange(0, len(t), int(round(every / (t[1] - t[0])))):
                    x = Q[k] % 5.5
                    G, Gh = lat.matrix(x), fast.gamma_h(x).dense()
                    errs.append(np.linalg.norm(Gh - G, 2) / np.linalg.norm(G, 2))
                    lmin_ref.append(DenseRef(G, len(x)).lam[0])
                    lmin_h.append(DenseRef(Gh, len(x)).lam[0])
        errs = np.array(errs)
        rows.append(dict(case=case, n_configurations=len(errs), frames_every=every,
                         rel_spectral_err_median=float(np.median(errs)), rel_spectral_err_max=float(errs.max()),
                         restricted_lam_min_ref_min=float(min(lmin_ref)), restricted_lam_min_fast_min=float(min(lmin_h)),
                         max_abs_diff_restricted_lam_min=float(np.max(np.abs(np.array(lmin_ref) - np.array(lmin_h))))))
    write_csv(OUT / "operator_along_trajectories.csv", rows)
    return rows


def main():
    global RAW
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-windows", action="store_true", help="run only the cache-based stages")
    ap.add_argument("--raw-dir", default=None, help=f"raw trajectory directory (default {RAW.relative_to(HERE)})")
    args = ap.parse_args()
    if args.raw_dir is not None:
        RAW = Path(args.raw_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    status = dict(tag=TAG, seeds=list(SEEDS), pairing="d_i = pppm_lanczos_i - reference_i (same start, same noise)")
    diag = load_diag_cache()
    stage_paired(diag)
    stage_lag_scan()
    status["paired_summary"] = status["lag_scan"] = "computed from cached observables (no raw files needed)"
    ok, info = (False, dict(reason="--skip-windows given")) if args.skip_windows else raw_ok()
    if ok:
        stage_windows()
        status["window_summary"] = dict(state="computed from raw trajectories", **info)
        stage_operator_along_trajectories()
        status["operator_along_trajectories"] = dict(state="computed from raw trajectories", **info)
    else:
        state = "SKIPPED: " + info["reason"]
        if (OUT / "window_summary.csv").exists():
            state += " (existing window_summary.csv was left unchanged; it is from an earlier run with raw files)"
        status["window_summary"] = dict(state=state, **info)
        status["operator_along_trajectories"] = dict(state="SKIPPED: " + info["reason"])
        print(f"Window check {state}", file=sys.stderr)
    (OUT / "status.json").write_text(json.dumps(status, indent=1) + "\n")
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
