#!/usr/bin/env python3
"""
Section 6, Lanczos accuracy and rank selection (former experiments 4, 5 and 8).

Matrix functions of the O-step, on the same Gamma_h, the same input vector and the same dt:
    noise    eta = f_dt(Gamma_h) Pi z,   f_dt(l) = sqrt((m/beta) * (-expm1(-2 dt l/m)))
    damping  d   = exp(-dt Gamma_h/m) Pi v
Direct-start Lanczos with full reorthogonalisation and Pi after every literal Gamma_h action (test_lanczos_fdt.lanczos,
unchanged). One recursion per (Gamma_h, start vector) gives every rank s = 1..S and every function and dt; ranks are
never recomputed separately. Both functions are evaluated on each recursion (in the dynamics the damping vector is the
projected momentum, which in equilibrium has the same N(0, Pi/beta) law as Pi z).

Stages
  small   Experiment 1 configurations (cases A and B, uniform / perturbed lattice / clustered N = 128 x 3 seeds,
          N = 256 x 3 families), production Gamma_h (spatial set T7). Reference: dense eigendecomposition of the same
          Gamma_h on Range(Pi); every rank compared with it.
  large   Fixed density N scan (sec6_config n_scan, 512 ... 16000). Reference: the same recursion continued to S_ref,
          accepted per tolerance only if ||eta_S - eta_{S-20}|| and ||eta_S - eta_{S-40}|| <= margin * tol
          (non-adjacent checks); otherwise the case is UNRESOLVED for that tolerance. Position seeds and Gaussian-vector
          seeds are separate; representative N carry several independent position configurations.
  select  Rank requirements per budget, sample maxima, failures and reference status; writes rank_selection.json and,
          with --write-config, the production ranks into sec6_config.json.
  report  Tables and figures.

dt: the production dt of sec6_config (primary criterion). The fixed-tau scan dt_tau = tau m / lambda_max is kept only
as a diagnostic (lambda_max exact for small cases, the top Ritz value -- an ESTIMATE -- for large ones).
Errors are sample maxima over the tested cases, not bounds. Negative Ritz values are recorded, never clipped.

    python3 sec6_lanczos.py small --workers 4
    python3 sec6_lanczos.py large --workers 4
    python3 sec6_lanczos.py select [--write-config]
    python3 sec6_lanczos.py report
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sec6_config  # noqa: E402
from sec6_common import Box, f_noise, f_damp, lanczos, tridiag, proj, DenseRef, MASS  # noqa: E402

OUT = sec6_config.RESULTS / "lanczos"
RAW = OUT / "raw"
SCRIPT = "sec6_lanczos.py"
SEED_POS, SEED_VEC = 20261014, 20261015
TAUS = (0.2, 0.5, 2.0, 5.0)                        # diagnostic fixed-tau scan
N_VEC_SMALL = 4                                    # Gaussian vectors per small configuration
LARGE_POS = {512: 1, 1024: 1, 2048: 1, 4000: 3, 8000: 1, 16000: 3}   # independent position configurations per N
N_VEC_LARGE = 3                                    # Gaussian vectors per position configuration
S_REF = {512: 260, 1024: 300, 2048: 320, 4000: 340, 8000: 360, 16000: 380}
SMALL_SMAX = 300


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def functions(dt_prod, lam_max):
    """Functions evaluated on each recursion: production dt (primary) and the fixed-tau diagnostic."""
    funs = {"noise|prod": f_noise(dt_prod), "damp|prod": f_damp(dt_prod)}
    for tau in TAUS:
        dt = tau * MASS / lam_max
        funs[f"noise|tau{tau:g}"] = f_noise(dt)
        funs[f"damp|tau{tau:g}"] = f_damp(dt)
    return funs


def coeffs_all_ranks(Lz, funs, smax):
    """Coefficient vectors c_s(f) = U f(Theta) U^T e1 (length s) for s = 1..smax, and Ritz data per rank."""
    C = {k: [] for k in funs}
    ritz_min, ritz_max, nneg, negvals = [], [], [], []
    for s in range(1, smax + 1):
        th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], s))
        ritz_min.append(float(th[0]))
        ritz_max.append(float(th[-1]))
        nneg.append(int((th < 0).sum()))
        if (th < 0).any():
            negvals.append(dict(s=s, values=[float(t) for t in th[th < 0]]))
        for k, fn in funs.items():
            with np.errstate(invalid="ignore"):
                fv = fn(th)
            C[k].append(U @ (fv * U[0]))
    return C, dict(ritz_min=ritz_min, ritz_max=ritz_max, n_negative=nneg, negative_values=negvals[:50])


def rel_err_coeff(c_s, c_ref):
    """||Q_s c_s - Q_S c_ref|| / ||Q_S c_ref|| in coefficient space (Q orthonormal; orthogonality is recorded)."""
    d = c_ref.copy()
    d[:len(c_s)] -= c_s
    return float(np.linalg.norm(d) / np.linalg.norm(c_ref))


# ----------------------------------------------------------------------------
# small: dense references on the Experiment 1 configurations
# ----------------------------------------------------------------------------
def small_configs():
    rows = list(csv.DictReader(open(HERE / "exp1_results" / "configs.csv")))
    X = np.load(HERE / "exp1_results" / "configs.npz")
    keep = [r for r in rows if r["set"] in ("legacy", "main", "check")]
    return [dict(name=r["name"], set=r["set"], family=r["family"], N=int(r["N"]), L=float(r["L"]),
                 pos_seed=r["seed"], x=X[r["name"]]) for r in keep]


def _small_one(job):
    cfg, k = job
    t0 = time.perf_counter()
    prod = sec6_config.production()
    N, L, x = cfg["N"], cfg["L"], cfg["x"] % cfg["L"]
    box = Box(N, prod["spatial_set"], L=L)
    op = box.gamma(x)
    G = op.dense()
    ref = DenseRef(G, N)
    dt = prod["dt"]
    vseed = [SEED_VEC, 1, N, zlib.crc32(cfg["name"].encode()), k]
    rng = np.random.default_rng(vseed)
    z = proj(rng.normal(size=3 * N))
    smax = min(SMALL_SMAX, 3 * N - 3)
    Lz = lanczos(op, z, smax)
    S = Lz["s"]
    funs = functions(dt, ref.lam_max)
    C, ritz = coeffs_all_ranks(Lz, funs, S)
    Q = Lz["Q"][:S]
    orth = float(np.linalg.norm(Q @ Q.T - np.eye(S), 2))
    out = dict(name=cfg["name"], set=cfg["set"], family=cfg["family"], N=N, L=L, pos_seed=cfg["pos_seed"],
               vec_index=k, vec_seed=vseed, S=S, orth=orth,
               lam_min=ref.lam_star, lam_max=ref.lam_max, lam_source="dense (exact for this Gamma_h)",
               tau_prod=dt * ref.lam_max / MASS, ritz=ritz, box=box.describe(), err={})
    # dense literal-vs-assembled check of the operator used
    v = proj(np.random.default_rng(5).normal(size=3 * N))
    out["literal_vs_dense"] = float(np.linalg.norm(proj(op.matvec(v)) - proj(G @ v)) / np.linalg.norm(proj(G @ v)))
    for key, fn in funs.items():
        eta_ref = ref.apply(fn, z)
        nref = np.linalg.norm(eta_ref)
        errs = []
        for s in range(1, S + 1):
            eta = proj(Lz["nz"] * (Q[:s].T @ C[key][s - 1]))
            errs.append(float(np.linalg.norm(eta - eta_ref) / nref) if np.all(np.isfinite(eta)) else float("nan"))
        out["err"][key] = errs
    out["t_s"] = time.perf_counter() - t0
    return out


def stage_small(args):
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(c, k) for c in small_configs() for k in range(N_VEC_SMALL)]
    res = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(_small_one, jobs):
            res.append(r)
            e = r["err"]
            print(f"[{now()}] small {r['name']} v{r['vec_index']}: S={r['S']} tau_prod={r['tau_prod']:.3f} "
                  f"noise r40 {e['noise|prod'][min(39, r['S'] - 1)]:.2e} damp r16 {e['damp|prod'][min(15, r['S'] - 1)]:.2e} "
                  f"neg {sum(r['ritz']['n_negative'])} {r['t_s']:.0f}s", flush=True)
    rec = dict(provenance=sec6_config.provenance(SCRIPT, stage="small", n_vec=N_VEC_SMALL, taus=TAUS,
                                                 seed_vec=SEED_VEC), cases=res)
    (RAW / "small_cases.json").write_text(json.dumps(rec) + "\n")


# ----------------------------------------------------------------------------
# large: fixed-density N scan with self-checked high-order references
# ----------------------------------------------------------------------------
def _large_one(job):
    N, ipos, k = job
    t0 = time.perf_counter()
    prod = sec6_config.production()
    box = Box(N, prod["spatial_set"])
    x = np.random.default_rng([SEED_POS, N, ipos]).uniform(0.0, box.L, size=(N, 3))
    op = box.gamma(x)
    z = proj(np.random.default_rng([SEED_VEC, 2, N, ipos, k]).normal(size=3 * N))
    t1 = time.perf_counter()
    Lz = lanczos(op, z, S_REF[N])
    t_lan = time.perf_counter() - t1
    S = Lz["s"]
    lam_max_est = float(np.linalg.eigvalsh(tridiag(Lz["al"], Lz["be"], S))[-1])
    funs = functions(prod["dt"], lam_max_est)
    C, ritz = coeffs_all_ranks(Lz, funs, S)
    Q = Lz["Q"][:S]
    orth = float(np.linalg.norm(Q @ Q.T - np.eye(S), 2))
    out = dict(N=N, L=box.L, pos_index=ipos, pos_seed=[SEED_POS, N, ipos], vec_index=k,
               vec_seed=[SEED_VEC, 2, N, ipos, k], S=S, orth=orth, lam_max_est=lam_max_est,
               lam_min_est=ritz["ritz_min"][-1], lam_source="Ritz values of the reference recursion (ESTIMATE)",
               tau_prod_est=prod["dt"] * lam_max_est / MASS, ritz=ritz, box=box.describe(),
               n_actions=op.n_actions, t_lanczos_s=t_lan, err={}, ref_check={})
    for key in funs:
        c_ref = C[key][S - 1]
        out["err"][key] = [rel_err_coeff(C[key][s - 1], c_ref) for s in range(1, S + 1)]
        # non-adjacent self-checks of the reference, in vector space
        e_ref = Lz["nz"] * (Q.T @ c_ref)
        chk = {}
        for back in (1, 20, 40):
            e_b = Lz["nz"] * (Q[:S - back].T @ C[key][S - back - 1])
            chk[f"S-{back}"] = float(np.linalg.norm(e_b - e_ref) / np.linalg.norm(e_ref))
        out["ref_check"][key] = chk
    # coefficient-space errors against vector-space errors at a few ranks (orthogonality guard)
    dev = 0.0
    for s in (16, 40, 64, 120):
        if s < S:
            for key in ("noise|prod", "damp|prod"):
                c_ref = C[key][S - 1]
                e_ref = Q.T @ c_ref
                e_s = Q[:s].T @ C[key][s - 1]
                v_err = np.linalg.norm(e_s - e_ref) / np.linalg.norm(e_ref)
                dev = max(dev, abs(v_err - out["err"][key][s - 1]) / max(v_err, 1e-300))
    out["coeff_vs_vector_rel_dev"] = float(dev)
    out["t_s"] = time.perf_counter() - t0
    return out


def stage_large(args):
    RAW.mkdir(parents=True, exist_ok=True)
    Ns = [n for n in sec6_config.load()["n_scan"] if n in LARGE_POS and (not args.N or n in args.N)]
    jobs = [(N, ip, k) for N in sorted(Ns, reverse=True) for ip in range(LARGE_POS[N]) for k in range(N_VEC_LARGE)]
    f = RAW / "large_cases.json"
    old = json.loads(f.read_text()) if f.exists() else {"cases": []}
    done = {(c["N"], c["pos_index"], c["vec_index"]) for c in old["cases"]}
    ok, why = sec6_config.reusable(old, SCRIPT) if old["cases"] else (False, "empty")
    if not ok:
        done, old = set(), {"cases": []}
    jobs = [j for j in jobs if j not in done]
    print(f"[{now()}] large: {len(jobs)} jobs ({len(done)} reusable: {why})", flush=True)
    res = list(old["cases"])
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(_large_one, jobs):
            res.append(r)
            e = r["err"]
            print(f"[{now()}] large N={r['N']} pos{r['pos_index']} v{r['vec_index']}: S={r['S']} "
                  f"tau_est={r['tau_prod_est']:.2f} lam_max_est={r['lam_max_est']:.2f} noise r40 {e['noise|prod'][39]:.2e} "
                  f"damp r16 {e['damp|prod'][15]:.2e} ref(S-20) noise {r['ref_check']['noise|prod']['S-20']:.1e} "
                  f"neg {sum(r['ritz']['n_negative'])} {r['t_s']:.0f}s", flush=True)
            rec = dict(provenance=sec6_config.provenance(SCRIPT, stage="large", taus=TAUS, s_ref=S_REF,
                                                         large_pos=LARGE_POS, n_vec=N_VEC_LARGE), cases=res)
            f.write_text(json.dumps(rec) + "\n")


# ----------------------------------------------------------------------------
# select
# ----------------------------------------------------------------------------
def required_rank(err, tol, upto):
    """Smallest r such that err[s] <= tol for every computed s in [r, upto] (1-based). None if never."""
    e = np.asarray(err[:upto], float)
    ok = np.isfinite(e) & (e <= tol)
    if not ok[-1]:
        return None
    bad = np.where(~ok)[0]
    return int(bad[-1] + 2) if bad.size else 1


def stage_select(args):
    cfgd = sec6_config.load()
    B = cfgd["budgets"]
    margin = B["reference_margin"]
    small = json.loads((RAW / "small_cases.json").read_text())
    large = json.loads((RAW / "large_cases.json").read_text())
    for rec, name in ((small, "small"), (large, "large")):
        ok, why = sec6_config.reusable(rec, SCRIPT)
        if not ok:
            print(f"WARNING: {name} cases not current ({why})")
    budgets = [("noise|prod", B["noise_rel_action"], "production noise"),
               ("damp|prod", B["damping_rel_action"], "production damping")]
    budgets += [("noise|prod", t, "noise high accuracy") for t in B["noise_high_accuracy"]]
    sel = dict(provenance=sec6_config.provenance(SCRIPT, stage="select"), budgets={}, cases_small=len(small["cases"]),
               cases_large=len(large["cases"]))
    for key, tol, label in budgets:
        rows, unresolved = [], []
        for c in small["cases"]:
            r = required_rank(c["err"][key], tol, c["S"])
            rows.append(dict(scale="small", case=f"{c['name']}|v{c['vec_index']}", N=c["N"], family=c["family"],
                             required=r, tau=c["tau_prod"], S=c["S"],
                             note=None if r is not None else "not reached within S (dense reference)"))
        for c in large["cases"]:
            chk = c["ref_check"][key]
            ref_ok = max(chk["S-20"], chk["S-40"]) <= margin * tol
            upto = c["S"] - 40                     # ranks below S-40 only: the reference must be clearly better
            r = required_rank(c["err"][key], tol, upto) if ref_ok else None
            row = dict(scale="large", case=f"N{c['N']}|pos{c['pos_index']}|v{c['vec_index']}", N=c["N"],
                       family="uniform", required=r, tau=c["tau_prod_est"], S=c["S"], ref_check=chk, ref_ok=ref_ok)
            if not ref_ok:
                row["note"] = "UNRESOLVED: reference self-difference above margin*tol"
                unresolved.append(row["case"])
            elif r is None:
                row["note"] = f"not reached up to S-40 = {upto}"
            rows.append(row)
        req = [r["required"] for r in rows if r["required"] is not None]
        failed = [r["case"] for r in rows if r["required"] is None and r.get("ref_ok", True)]
        rmax = max(req) if req and not failed else None
        by_N = {}
        for r in rows:
            by_N.setdefault(r["N"], []).append(r["required"])
        sel["budgets"][f"{key}|{tol:g}"] = dict(
            function=key, tol=tol, label=label, n_cases=len(rows), n_unresolved=len(unresolved), unresolved=unresolved,
            n_not_reached=len(failed), not_reached=failed, max_required=rmax,
            max_required_by_N={str(n): (max(v) if all(x is not None for x in v) else None) for n, v in sorted(by_N.items())},
            rows=rows)
    # sample-maximum errors at candidate ranks (production dt)
    cand = sorted({8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160})
    tab = {}
    for key in ("noise|prod", "damp|prod"):
        for scale, rec in (("small", small), ("large", large)):
            for c in rec["cases"]:
                n = c["N"]
                for r in cand:
                    upto = c["S"] if scale == "small" else c["S"] - 40
                    if r <= upto:
                        v = c["err"][key][r - 1]
                        k2 = f"{key}|N{n}|r{r}"
                        tab[k2] = max(tab.get(k2, 0.0), v)
    sel["sample_max_at_candidates"] = tab
    # production choice: smallest rank meeting the production budget in every resolved case, at the production dt
    pn = sel["budgets"][f"noise|prod|{B['noise_rel_action']:g}"]
    pd = sel["budgets"][f"damp|prod|{B['damping_rel_action']:g}"]
    sel["production"] = dict(noise_rank=pn["max_required"], damping_rank=pd["max_required"],
                             rule="smallest rank r such that, in every tested case with an accepted reference, the "
                                  "relative action error stays <= budget for all computed ranks >= r (production dt); "
                                  "sample maxima, not bounds",
                             noise_unresolved=pn["unresolved"], damping_unresolved=pd["unresolved"],
                             noise_not_reached=pn["not_reached"], damping_not_reached=pd["not_reached"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rank_selection.json").write_text(json.dumps(sel, indent=1) + "\n")
    print(json.dumps(sel["production"], indent=1))
    for k, v in sel["budgets"].items():
        print(k, "max required", v["max_required"], "by N", v["max_required_by_N"], "unresolved", v["n_unresolved"],
              "not reached", v["n_not_reached"])
    if args.write_config:
        p = sel["production"]
        if p["noise_rank"] is None or p["damping_rank"] is None or p["noise_unresolved"] or p["damping_unresolved"]:
            print("not writing ranks: unresolved or not reached cases remain")
        else:
            sec6_config.set_ranks(p["noise_rank"], p["damping_rank"],
                                  "sec6_results/lanczos/rank_selection.json (" + now() + ")")
            print("wrote production ranks to sec6_config.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["small", "large", "select", "report"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--N", type=int, nargs="*")
    ap.add_argument("--write-config", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    print(f"[{now()}] stage {args.stage}", flush=True)
    if args.stage == "report":
        import sec6_lanczos_report
        sec6_lanczos_report.main()
    else:
        {"small": stage_small, "large": stage_large, "select": stage_select}[args.stage](args)
    print(f"[{now()}] stage {args.stage} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
