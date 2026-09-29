#!/usr/bin/env python3
"""
Direct-start versus Landau-style range-start Lanczos for the finite-time discrete-FDT noise.

    target      eta = f_dt(Gamma_h) z,   z = Pi xi,
                f_dt(l) = sqrt(beta^{-1} m [-expm1(-2 dt l/m)]),  beta = m = 1
    direct      q1 = z/||z||,   eta_r = ||z|| Q_r f_dt(T_r) e1
    range       b = Pi Gamma_h z,  q1 = b/||b||,  eta_r = ||b|| Q_r g_dt(T_r) e1,  g_dt(l) = f_dt(l)/l
    debug       f = sqrt, g = 1/sqrt  (Gamma_h^{-1/2} Gamma_h z = Gamma_h^{1/2} z on Range(Gamma_h))

Same Gamma_h (literal PPPM action, Pi after every action), same configurations, same Gaussian
vectors and same scaling seeds as test_lanczos_fdt.py.  No Ritz value is clipped or thresholded
in the main experiment; every Ritz value is checked before g is evaluated.  The Landau-style
threshold (Ritz <= 1e-12 ||T_r|| -> g := 0) is run afterwards as a separate ablation only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_ewald_longitudinal import set_style, INK2  # noqa: E402
from test_lanczos_fdt import (GammaH, PARAMS, lanczos, tridiag, proj, DenseRef, required_rank, estimators,  # noqa: E402
                              MASS, BETA)

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
TOLS = [1e-4, 1e-6, 1e-8, 1e-10]
TAU = 0.5                                   # dt lambda_max / m (as in the completed covariance/scaling runs)
EXTRA = [("E1", 30, 6.0, 1), ("E2", 30, 6.0, 2), ("E3", 40, 10.0, 3), ("E4", 40, 10.0, 4), ("E5", 50, 8.0, 5),
         ("E6", 50, 12.0, 6), ("E7", 20, 5.0, 8), ("E8", 60, 9.0, 9), ("E9", 36, 7.0, 10), ("E10", 45, 11.0, 11)]


def funcs(dt):
    f = lambda l: np.sqrt(MASS / BETA * (-np.expm1(-2 * dt * l / MASS)))
    return {"f_dt": (f, lambda l: f(l) / l), "sqrt": (np.sqrt, lambda l: 1.0 / np.sqrt(l))}


class RitzStop(Exception):
    pass


def sweep(L, fun, scale, landau_tau=None):
    """eta_r for r = 1..S from a Lanczos run L (start vector norm L['nz']); every Ritz value is inspected first."""
    S, Q, nz = L["s"], L["Q"], L["nz"]
    etas, rmin, gmax, disc = [], [], [], []
    stop = None
    for r in range(1, S + 1):
        th, U = np.linalg.eigh(tridiag(L["al"], L["be"], r))
        rmin.append(th[0])
        if th[0] <= 0 and landau_tau is None:
            stop = (r, float(th[0]), float(th[0] / scale))
            break
        if landau_tau is not None:
            tau = landau_tau * np.abs(th).max()
            keep = th > tau
            gv = np.zeros_like(th)
            gv[keep] = fun(th[keep])
            disc.append(float((U[0, ~keep] ** 2).sum()))
        else:
            gv = fun(th)
        if not np.all(np.isfinite(gv)):
            stop = (r, float(th[0]), float(th[0] / scale))
            break
        gmax.append(float(np.abs(gv).max()))
        etas.append(proj(nz * (Q[:r].T @ (U @ (gv * U[0])))))
    return dict(etas=np.array(etas), ritz_min=np.array(rmin), gmax=np.array(gmax), discarded=np.array(disc), stop=stop)


def pair_run(op, z, smax, dense=None):
    """Direct and range Lanczos bases for the same z; the range start costs one extra Gamma_h action."""
    n0 = op.n_actions
    t0 = time.perf_counter()
    Ld = lanczos(op, z, smax, dense=dense)
    Ld["t"] = time.perf_counter() - t0
    Ld["actions"] = op.n_actions - n0
    n0 = op.n_actions
    t0 = time.perf_counter()
    b = proj(op.matvec(z))
    Lr = lanczos(op, b, smax, dense=dense)
    Lr["t"] = time.perf_counter() - t0
    Lr["actions"] = op.n_actions - n0
    Lr["b_offPi"] = float(np.linalg.norm(op.matvec(z) - b) / np.linalg.norm(b))   # (I - Pi) Gamma_h z before projection
    return Ld, Lr, b


def stop_rules(err, d1, d2, tol):
    true = required_rank(err, tol)
    out = {}
    for name, ok in (("d <= tol/10", d1 <= tol / 10), ("d2 <= tol", d2 <= tol), ("d <= tol", d1 <= tol)):
        hit = np.where(ok)[0]
        if hit.size and true:
            s = int(hit[0] + 1)
            out[name] = dict(stop=s, true=true, err=float(err[s - 1]), false=bool(err[s - 1] > tol), over=s / true)
    return out


# ----------------------------------------------------------------------------
def dense_study(rng, say, nvec_main=32, nvec_extra=16):
    """Replays the random-number sequence of test_lanczos_fdt.py so that z are the same vectors."""
    rows, per_vec, weights, fractions, ablation = [], [], [], [], []
    configs = [("A", 30, 6.0, 7, nvec_main), ("B", 40, 10.0, 13, nvec_main)] + [(n, N, L, s, nvec_extra) for n, N, L, s in EXTRA]
    for name, N, L, seed, nvec in configs:
        x = np.random.default_rng(seed).uniform(0, L, size=(N, 3))
        op = GammaH(x, L, **PARAMS)
        G = op.dense()
        rng.normal(size=(5, 3 * N))                       # same draw as the literal-vs-dense check before
        ref = DenseRef(G, N)
        dt = TAU * MASS / ref.lam_max
        F = funcs(dt)
        for v in range(nvec):
            z = proj(rng.normal(size=3 * N))
            Ld, Lr, b = pair_run(op, z, 3 * N, dense=G if name in ("A", "B") else None)
            c = ref.U.T @ z
            for fn, (f, g) in F.items():
                if fn == "sqrt" and name not in ("A", "B"):
                    continue
                ex = ref.apply(f, z)
                sd, sr = sweep(Ld, f, ref.lam_max), sweep(Lr, g, ref.lam_max)
                for meth, sw, Lz in (("direct", sd, Ld), ("range", sr, Lr)):
                    if sw["stop"] is not None:
                        say(f"   RITZ STOP {name} v{v} {fn} {meth}: rank {sw['stop'][0]}, theta {sw['stop'][1]:.3e} "
                            f"(theta/lambda_max {sw['stop'][2]:.1e})")
                    err = np.linalg.norm(sw["etas"] - ex, axis=1) / np.linalg.norm(ex)
                    d1, d2 = estimators(sw["etas"])
                    e = sw["etas"]
                    per_vec.append(dict(config=name, vec=v, fn=fn, method=meth, err=err, d1=d1, d2=d2,
                                        ritz_min=sw["ritz_min"], gmax=sw["gmax"], stop=sw["stop"],
                                        offPi=np.linalg.norm(e - np.array([proj(q) for q in e]), axis=1) / np.linalg.norm(e, axis=1),
                                        mom=np.linalg.norm(e.reshape(len(e), N, 3).sum(axis=1), axis=1) / np.linalg.norm(e, axis=1),
                                        b_offPi=Lr["b_offPi"] if meth == "range" else 0.0,
                                        final_err=float(err.min()), actions_per_rank=0 if meth == "direct" else 1))
                if fn == "f_dt" and name in ("A", "B"):
                    # Landau-style threshold ablation (range start only; tau = 1e-12 ||T_r||)
                    st = sweep(Lr, g, ref.lam_max, landau_tau=1e-12)
                    k = min(len(st["etas"]), len(sr["etas"]))
                    ablation.append(dict(config=name, vec=v, discarded_max=float(st["discarded"].max()),
                                         action_bias_max=float((np.linalg.norm(st["etas"][:k] - sr["etas"][:k], axis=1)
                                                                / np.linalg.norm(sr["etas"][:k], axis=1)).max()),
                                         eta_thr=st["etas"][-1], eta_nothr=sr["etas"][-1],
                                         rank_thr_1e8=required_rank(np.linalg.norm(st["etas"] - ex, axis=1) / np.linalg.norm(ex), 1e-8)))
            if v < nvec:
                fz = F["f_dt"][0](ref.lam)
                w_d = c ** 2 / (c ** 2).sum()
                w_r = (ref.lam * c) ** 2 / ((ref.lam * c) ** 2).sum()
                tgt = (fz * c) ** 2 / ((fz * c) ** 2).sum()
                weights.append(dict(config=name, vec=v, lam=ref.lam, w_direct=w_d, w_range=w_r, w_target=tgt))
                fr = {}
                for lab, thr in (("<2lam*", 2 * ref.lam_star), ("<5lam*", 5 * ref.lam_star), ("<10lam*", 10 * ref.lam_star),
                                 ("<0.01lam_max", 0.01 * ref.lam_max), ("<0.05lam_max", 0.05 * ref.lam_max)):
                    m = ref.lam < thr
                    fr[lab] = (float(tgt[m].sum()), float(w_d[m].sum()), float(w_r[m].sum()), int(m.sum()))
                fractions.append(dict(config=name, vec=v, **fr))
        row = dict(config=name, N=N, L=L, lam_star=ref.lam_star, lam_max=ref.lam_max, cond=ref.lam_max / ref.lam_star)
        for fn in ("f_dt", "sqrt"):
            for meth in ("direct", "range"):
                rr = [q for q in per_vec if q["config"] == name and q["fn"] == fn and q["method"] == meth]
                for tol in TOLS:
                    ranks = [required_rank(q["err"], tol) for q in rr]
                    ranks = np.array([r for r in ranks if r is not None], float)
                    if ranks.size:
                        row[f"{fn}_{meth}_{tol:g}"] = dict(median=float(np.median(ranks)), p90=float(np.percentile(ranks, 90)),
                                                           p99=float(np.percentile(ranks, 99)), max=int(ranks.max()),
                                                           n=int(ranks.size), n_total=len(rr))
        rows.append(row)
        g8 = lambda m: row.get(f"f_dt_{m}_1e-08", {}).get("median")
        say(f"  {name}: N={N} L={L:g} cond={row['cond']:.1f}  f_dt rank(1e-8) median direct {g8('direct')} "
            f"range {g8('range')}  (Gamma_h actions: direct r, range r+1)")
    return rows, per_vec, weights, fractions, ablation


def scaling_study(Ns, smax, say, n_vec=2, tols=(1e-6, 1e-8)):
    dens = 30 / 216.0
    out = []
    for N in Ns:
        seed = 1000 + N
        L = (N / dens) ** (1 / 3)
        x = np.random.default_rng(seed).uniform(0, L, size=(N, 3))
        op = GammaH(x, L, **PARAMS)
        rng = np.random.default_rng(seed + 1)
        for v in range(n_vec):
            z = proj(rng.normal(size=3 * N))
            Ld, Lr, b = pair_run(op, z, smax)
            thd = np.linalg.eigvalsh(tridiag(Ld["al"], Ld["be"], Ld["s"]))
            lam_max = float(thd[-1])
            dt = TAU * MASS / lam_max                       # same dt definition as the completed scaling run
            f, g = funcs(dt)["f_dt"]
            t0 = time.perf_counter()
            sd = sweep(Ld, f, lam_max)
            t_fd = time.perf_counter() - t0
            t0 = time.perf_counter()
            sr = sweep(Lr, g, lam_max)
            t_fr = time.perf_counter() - t0
            ed, er = sd["etas"], sr["etas"]
            d_d, _ = estimators(ed)
            d_r, d2_r = estimators(er)
            ref = ed[-1]
            agree = float(np.linalg.norm(er[-1] - ed[-1]) / np.linalg.norm(ref))
            conv_d = float(d_d[-1])
            conv_r = float(d_r[-1])
            err_d = np.linalg.norm(ed - ref, axis=1) / np.linalg.norm(ref)
            err_r = np.linalg.norm(er - ref, axis=1) / np.linalg.norm(ref)
            # Gauss-quadrature estimate of the low-mode share of ||f(Gamma_h) z||^2 (converged direct run)
            th, U = np.linalg.eigh(tridiag(Ld["al"], Ld["be"], Ld["s"]))
            wq = U[0] ** 2 * f(th) ** 2
            wq /= wq.sum()
            row = dict(N=N, L=L, vec=v, M=op.mesh.M, lam_max=lam_max, ritz_min=float(th[0]), cond=lam_max / th[0],
                       s_direct=Ld["s"], s_range=Lr["s"], conv_direct=conv_d, conv_range=conv_r, agree=agree,
                       reference_ok=bool(conv_d <= 1e-12 and agree <= 1e-11),
                       ritz_min_range=float(sr["ritz_min"].min()), gmax_range=float(sr["gmax"].max()),
                       stop_range=sr["stop"], t_action=op.t_actions / op.n_actions,
                       low_share_5ritzmin=float(wq[th < 5 * th[0]].sum()), low_share_001lmax=float(wq[th < 0.01 * lam_max].sum()),
                       mom_range=float(np.linalg.norm(er[-1].reshape(N, 3).sum(0)) / np.linalg.norm(er[-1])),
                       b_offPi=Lr["b_offPi"], t_feval_direct_full=t_fd, t_feval_range_full=t_fr)
            for tol in tols:
                row[f"r_direct_{tol:g}"] = required_rank(err_d[:-5], tol)
                row[f"r_range_{tol:g}"] = required_rank(err_r[:-5], tol)
                hit = np.where(d_r <= tol / 10)[0]
                row[f"rule_range_{tol:g}"] = int(hit[0] + 1) if hit.size else None
                row[f"err_rule_range_{tol:g}"] = float(err_r[hit[0]]) if hit.size else None
                hit = np.where(d_d <= tol / 10)[0]
                row[f"rule_direct_{tol:g}"] = int(hit[0] + 1) if hit.size else None
                row[f"err_rule_direct_{tol:g}"] = float(err_d[hit[0]]) if hit.size else None
            # production-style timed solves with the d <= tol/10 rule (fresh runs, rank-by-rank stopping)
            for tol in tols:
                for meth in ("direct", "range"):
                    T = timed_solve(op, z, meth, f if meth == "direct" else g, tol, smax)
                    T["err"] = float(np.linalg.norm(T["eta"] - ref) / np.linalg.norm(ref))
                    del T["eta"]
                    row[f"timed_{meth}_{tol:g}"] = T
            out.append(row)
            say(f"   N={N} v{v}: ref ok {row['reference_ok']} (conv {conv_d:.1e}, agree {agree:.1e}); "
                f"rank 1e-6 d/r {row['r_direct_1e-06']}/{row['r_range_1e-06']}, 1e-8 d/r {row['r_direct_1e-08']}/"
                f"{row['r_range_1e-08']}; timed 1e-8: direct {row['timed_direct_1e-08']['t_total']:.2f}s "
                f"({row['timed_direct_1e-08']['actions']} act) range {row['timed_range_1e-08']['t_total']:.2f}s "
                f"({row['timed_range_1e-08']['actions']} act)")
    return out


def timed_solve(op, z, method, fun, tol, smax):
    """Rank-by-rank Lanczos with the stopping rule d_r <= tol/10; counts every Gamma_h action (incl. b = Gamma_h z)."""
    n0, ta0 = op.n_actions, op.t_actions
    t_start = time.perf_counter()
    v0 = proj(op.matvec(z)) if method == "range" else z
    n = v0.size
    nv = np.linalg.norm(v0)
    Q = np.zeros((smax + 1, n))
    al, be = np.zeros(smax), np.zeros(smax)
    Q[0] = v0 / nv
    t_re = t_f = 0.0
    prev = None
    eta = None
    r_stop = None
    for k in range(smax):
        w = proj(op.matvec(Q[k]))
        al[k] = Q[k] @ w
        w = w - al[k] * Q[k] - (be[k - 1] * Q[k - 1] if k else 0.0)
        t0 = time.perf_counter()
        for _ in range(2):
            w -= Q[:k + 1].T @ (Q[:k + 1] @ w)
        w = proj(w)
        be[k] = np.linalg.norm(w)
        Q[k + 1] = w / be[k]
        t_re += time.perf_counter() - t0
        t0 = time.perf_counter()
        th, U = np.linalg.eigh(tridiag(al, be, k + 1))
        if th[0] <= 0:
            raise RuntimeError(f"non-positive Ritz value {th[0]} in timed {method} solve")
        eta = proj(nv * (Q[:k + 1].T @ (U @ (fun(th) * U[0]))))
        t_f += time.perf_counter() - t0
        if prev is not None and np.linalg.norm(eta - prev) / np.linalg.norm(eta) <= tol / 10:
            r_stop = k + 1
            break
        prev = eta
    t_tot = time.perf_counter() - t_start
    return dict(eta=eta, rank=r_stop, actions=op.n_actions - n0, t_actions=op.t_actions - ta0, t_reorth=t_re,
                t_feval=t_f, t_total=t_tot)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scaling-N", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4000])
    ap.add_argument("--smax", type=int, default=400)
    args = ap.parse_args()
    out = HERE / "lanczos_range_start_results"
    out.mkdir(exist_ok=True)
    log = []
    say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(t) for t in a)))
    T0 = time.perf_counter()
    rng = np.random.default_rng(2026)        # same generator/sequence as test_lanczos_fdt.py

    say("== dense validation: A, B (32 vectors) and E1..E10 (16 vectors) ==")
    rows, per_vec, weights, fractions, ablation = dense_study(rng, say)
    prev = {r["config"]: r["rank_median_1e-08"] for r in json.loads((HERE / "lanczos_fdt_results" / "gap_dependence.json").read_text())}
    consistency = {r["config"]: (r["f_dt_direct_1e-08"]["median"], prev.get(r["config"])) for r in rows}
    say(f"  direct-start consistency with previous run (median rank 1e-8, now vs before): {consistency}")

    say("\n== fixed-density scaling (same seeds as test_lanczos_fdt.py) ==")
    scal = scaling_study(args.scaling_N, args.smax, say)

    # ---- aggregate checks -----------------------------------------------------------
    rng_ = [q for q in per_vec if q["method"] == "range"]
    checks = dict(
        final_err_max_direct=max(q["final_err"] for q in per_vec if q["method"] == "direct"),
        final_err_max_range=max(q["final_err"] for q in rng_),
        ritz_stops=[(q["config"], q["vec"], q["fn"], q["method"], q["stop"]) for q in per_vec if q["stop"] is not None],
        min_ritz_over_lam_star_range={r["config"]: float(min(q["ritz_min"].min() for q in rng_ if q["config"] == r["config"])
                                                         / r["lam_star"]) for r in rows},
        gmax_range_over_g_at_lam_star={},
        offPi_max_range=max(float(q["offPi"].max()) for q in rng_),
        mom_max_range=max(float(q["mom"].max()) for q in rng_),
        b_offPi_max=max(q["b_offPi"] for q in rng_),
        direct_consistency=consistency)
    rules = {}
    for meth in ("direct", "range"):
        for tol in TOLS:
            for rule in ("d <= tol/10", "d2 <= tol", "d <= tol"):
                res = [stop_rules(q["err"], q["d1"], q["d2"], tol).get(rule) for q in per_vec
                       if q["method"] == meth and q["fn"] == "f_dt"]
                res = [r for r in res if r]
                if res:
                    rules[f"{meth} {tol:g} {rule}"] = dict(false_rate=float(np.mean([r["false"] for r in res])),
                                                           over_median=float(np.median([r["over"] for r in res])),
                                                           over_max=float(max(r["over"] for r in res)),
                                                           over_min=float(min(r["over"] for r in res)),
                                                           err_max=float(max(r["err"] for r in res)), n=len(res))
    abl = dict(discarded_max=max(a["discarded_max"] for a in ablation),
               action_bias_max=max(a["action_bias_max"] for a in ablation),
               rank_change_1e8=[(a["config"], a["rank_thr_1e8"]) for a in ablation[:3]])
    for name in ("A", "B"):
        aa = [a for a in ablation if a["config"] == name]
        Ct = sum(np.outer(a["eta_thr"], a["eta_thr"]) for a in aa) / len(aa)
        Cn = sum(np.outer(a["eta_nothr"], a["eta_nothr"]) for a in aa) / len(aa)
        abl[f"cov_bias_{name}"] = float(np.linalg.norm(Ct - Cn) / np.linalg.norm(Cn))
        abl[f"rank_thr_vs_nothr_1e8_{name}"] = (float(np.median([a["rank_thr_1e8"] for a in aa])),
                                                next(r for r in rows if r["config"] == name)["f_dt_range_1e-08"]["median"])

    # ---- power-law fits ----------------------------------------------------------------
    Ns = np.array(sorted({r["N"] for r in scal}), float)
    fits = {}
    for key in ("r_direct_1e-06", "r_range_1e-06", "r_direct_1e-08", "r_range_1e-08"):
        y = np.array([np.mean([r[key] for r in scal if r["N"] == n]) for n in Ns])
        fits[key] = dict(alpha=float(np.polyfit(np.log(Ns), np.log(y), 1)[0]), values=y.tolist())
    tfit = {}
    for meth in ("direct", "range"):
        for tol in (1e-6, 1e-8):
            y = np.array([np.mean([r[f"timed_{meth}_{tol:g}"]["t_total"] for r in scal if r["N"] == n]) for n in Ns])
            tfit[f"{meth}_{tol:g}"] = dict(alpha=float(np.polyfit(np.log(Ns), np.log(y), 1)[0]), values=y.tolist())

    # ---- save -------------------------------------------------------------------------
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, np.generic):
            return o.item()
        return o
    for name, obj in (("dense_ranks", rows), ("fractions", fractions), ("checks", checks), ("stopping_rules", rules),
                      ("landau_threshold_ablation", abl), ("scaling", scal), ("fits", dict(rank=fits, time=tfit))):
        (out / f"{name}.json").write_text(json.dumps(clean(obj), indent=1) + "\n")
    (out / "per_vector_compact.json").write_text(json.dumps(clean([{k: q[k] for k in ("config", "vec", "fn", "method", "err",
                                                                                          "d1", "d2", "ritz_min", "gmax")}
                                                                   for q in per_vec if q["vec"] < 3]), indent=0) + "\n")

    plots(out, rows, per_vec, weights, fractions, scal, fits)
    rep = summary(rows, fractions, checks, rules, abl, scal, fits, tfit)
    say("\n" + rep)
    (out / "summary.md").write_text(rep + "\n")
    say(f"\nwall time {time.perf_counter() - T0:.0f}s")
    (out / "run_log.txt").write_text("\n".join(log) + "\n")


# ----------------------------------------------------------------------------
def plots(out, rows, per_vec, weights, fractions, scal, fits):
    set_style()
    info = {r["config"]: r for r in rows}

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, name in zip(axs, ("A", "B")):
        for ci, (fn, meth, st) in enumerate((("f_dt", "direct", "-"), ("f_dt", "range", "-"), ("sqrt", "direct", "--"),
                                              ("sqrt", "range", "--"))):
            rr = [q for q in per_vec if q["config"] == name and q["fn"] == fn and q["method"] == meth]
            S = min(len(q["err"]) for q in rr)
            E = np.array([q["err"][:S] for q in rr])
            s = np.arange(1, S + 1)
            ax.semilogy(s, np.median(E, 0), st, color=PAL[ci], label=f"{fn}, {meth} start (median)")
            if fn == "f_dt":
                ax.fill_between(s, E.min(0), E.max(0), color=PAL[ci], alpha=0.15, lw=0)
        ax.set(xlabel="Lanczos rank r", ylabel="relative action error", ylim=(1e-16, 2),
               title=f"1. config {name}: direct vs range start (band = min..max)")
        ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "plot01_error_vs_rank.png"), plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, name in zip(axs, ("A", "B")):
        r = info[name]
        for ci, meth in enumerate(("direct", "range")):
            ax.semilogx(TOLS, [r[f"f_dt_{meth}_{t:g}"]["median"] for t in TOLS], "o-", color=PAL[ci], label=f"{meth}: median")
            ax.semilogx(TOLS, [r[f"f_dt_{meth}_{t:g}"]["max"] for t in TOLS], "^:", color=PAL[ci], ms=4, label=f"{meth}: max")
        ax.invert_xaxis()
        ax.set(xlabel="tolerance", ylabel="required rank", title=f"2. config {name}: f_dt rank vs tolerance")
        ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plot02_rank_vs_tol.png"), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ci, meth in enumerate(("direct", "range")):
        for mk, tol in (("o", 1e-6), ("s", 1e-8), ("^", 1e-10)):
            ok = [r for r in rows if f"f_dt_{meth}_{tol:g}" in r]
            ax.plot([np.log10(r["cond"]) for r in ok], [r[f"f_dt_{meth}_{tol:g}"]["median"] for r in ok], mk,
                    color=PAL[ci], mfc="none" if meth == "range" else PAL[ci], label=f"{meth}, tol {tol:g}")
    ax.set(xlabel="log10(lambda_max / lambda_*)", ylabel="median required rank (f_dt)",
           title="3. Required rank vs condition ratio (12 dense configurations)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(), fig.savefig(out / "plot03_rank_vs_cond.png"), plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, name in zip(axs, ("A", "B")):
        ww = [w for w in weights if w["config"] == name]
        lam = ww[0]["lam"]
        for key, ci, lab in (("w_direct", 0, "|c_j|^2 (direct start)"), ("w_range", 1, "lambda_j^2 |c_j|^2 (range start)"),
                             ("w_target", 2, "f_dt(lambda_j)^2 |c_j|^2 (target norm)")):
            ax.loglog(lam, np.mean([w[key] for w in ww], 0), ".", ms=5, color=PAL[ci], label=lab)
        ax.axvline(info[name]["lam_star"], color=INK2, lw=0.8)
        ax.set(xlabel="eigenvalue lambda_j", ylabel="normalised spectral weight (mean over vectors)",
               title=f"4. config {name}: spectral weight of the start vectors")
        ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "plot04_spectral_weight.png"), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labs = ["<2lam*", "<5lam*", "<10lam*", "<0.01lam_max", "<0.05lam_max"]
    for ci, name in enumerate(("A", "B")):
        ff = [f for f in fractions if f["config"] == name]
        ax.semilogy(range(len(labs)), [max(np.mean([f[l][0] for f in ff]), 1e-18) for l in labs], "o-", color=PAL[ci],
                    label=f"{name}: share of ||f_dt(Gamma) z||^2")
        ax.semilogy(range(len(labs)), [max(np.mean([f[l][1] for f in ff]), 1e-18) for l in labs], "s--", color=PAL[ci],
                    lw=1, label=f"{name}: share of ||z||^2 (direct start)")
    ax.set_xticks(range(len(labs)), labs)
    ax.set(ylabel="fraction (mean over vectors)", title="5. Low-eigenvalue contribution to the target noise norm")
    ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "plot05_low_mode_share.png"), plt.close(fig)

    Ns = sorted({r["N"] for r in scal})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ci, (meth, tol) in enumerate((("direct", 1e-6), ("range", 1e-6), ("direct", 1e-8), ("range", 1e-8))):
        key = f"r_{meth}_{tol:g}"
        ax.loglog(Ns, fits[key]["values"], "o-" if meth == "direct" else "s--", color=PAL[ci],
                  label=f"{meth}, tol {tol:g}: alpha = {fits[key]['alpha']:.2f}")
    ax.set(xlabel="N", ylabel="required rank", title="6. Required rank vs N (fixed density)")
    ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plot06_rank_vs_N.png"), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for ci, tol in enumerate((1e-6, 1e-8)):
        ax.semilogx(Ns, np.array(fits[f"r_range_{tol:g}"]["values"]) / np.array(fits[f"r_direct_{tol:g}"]["values"]),
                    "o-", color=PAL[ci], label=f"tol {tol:g}")
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set(xlabel="N", ylabel="rank_range / rank_direct", title="7. Range/direct rank ratio vs N")
    ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plot07_rank_ratio_vs_N.png"), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ci, (meth, tol) in enumerate((("direct", 1e-6), ("range", 1e-6), ("direct", 1e-8), ("range", 1e-8))):
        y = [np.mean([r[f"timed_{meth}_{tol:g}"]["t_total"] for r in scal if r["N"] == n]) for n in Ns]
        ax.loglog(Ns, y, "o-" if meth == "direct" else "s--", color=PAL[ci], label=f"{meth}, tol {tol:g} (d <= tol/10)")
    ax.set(xlabel="N", ylabel="complete root-action time [s]",
           title="8. Root-action wall time vs N (all Gamma_h actions incl. b)")
    ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plot08_time_vs_N.png"), plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, name in zip(axs, ("A", "B")):
        for ci, meth in enumerate(("direct", "range")):
            rr = [q for q in per_vec if q["config"] == name and q["fn"] == "f_dt" and q["method"] == meth]
            xs = np.concatenate([q["err"][2:] for q in rr])
            ys = np.concatenate([q["d1"][2:] for q in rr])
            m = (xs > 1e-16) & np.isfinite(ys) & (ys > 1e-17)
            ax.loglog(xs[m], ys[m], ".", ms=2, alpha=0.5, color=PAL[ci], label=f"d_r, {meth} start")
        ax.plot([1e-16, 1], [1e-16, 1], color=INK2, lw=0.8, label="y = x")
        ax.plot([1e-15, 1], [1e-16, 0.1], color=INK2, lw=0.8, ls="--", label="y = x/10")
        ax.set(xlabel="true action error", ylabel="d_r", title=f"9. config {name}: successive-rank estimator")
        ax.legend(fontsize=7, markerscale=4)
    fig.tight_layout(), fig.savefig(out / "plot09_estimator.png"), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for ci, name in enumerate(("A", "B")):
        rr = [q for q in per_vec if q["config"] == name and q["fn"] == "f_dt" and q["method"] == "range"]
        S = min(len(q["mom"]) for q in rr)
        ax.semilogy(np.arange(1, S + 1), np.max([q["mom"][:S] for q in rr], 0), color=PAL[ci],
                    label=f"{name}: |total momentum| / ||eta_r|| (max)")
        ax.semilogy(np.arange(1, S + 1), np.maximum(np.max([q["offPi"][:S] for q in rr], 0), 1e-18), "--", color=PAL[ci],
                    label=f"{name}: ||(I - Pi) eta_r|| / ||eta_r|| (max)")
        ax.axhline(max(q["b_offPi"] for q in rr), color=PAL[ci], lw=0.8, ls=":",
                   label=f"{name}: ||(I - Pi) Gamma_h z|| / ||b|| (before projection)")
    ax.set(xlabel="rank r", ylabel="relative residual", ylim=(1e-18, 1e-12), title="10. Range start: momentum / nullspace residual")
    ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "plot10_momentum.png"), plt.close(fig)


def summary(rows, fractions, checks, rules, abl, scal, fits, tfit):
    L = ["# Direct-start vs range-start Lanczos: summary", "",
         f"Gamma_h: {PARAMS}; f_dt with dt lambda_max/m = {TAU}; beta = m = 1. Same z as test_lanczos_fdt.py.", "",
         "## Dense configurations (f_dt; median/p90/p99/max required rank)", "",
         "| config | lam_* | lam_max | cond | tol | direct | range | range/direct (median) |", "|---|---|---|---|---|---|---|---|"]
    fmt = lambda d: f"{d['median']:.0f}/{d['p90']:.0f}/{d['p99']:.0f}/{d['max']}" if d else "n/a"
    for r in sorted(rows, key=lambda r: r["cond"]):
        for tol in TOLS:
            d, g = r.get(f"f_dt_direct_{tol:g}"), r.get(f"f_dt_range_{tol:g}")
            L.append(f"| {r['config']} | {r['lam_star']:.3e} | {r['lam_max']:.3f} | {r['cond']:.1f} | {tol:g} | {fmt(d)} | "
                     f"{fmt(g)} | {g['median'] / d['median']:.2f} |" if d and g else
                     f"| {r['config']} | | | | {tol:g} | {fmt(d)} | {fmt(g)} | |")
    L += ["", "sqrt debug test (A, B):"]
    for name in ("A", "B"):
        r = next(r for r in rows if r["config"] == name)
        L.append(f"- {name}: " + ", ".join(f"tol {t:g}: direct {fmt(r.get(f'sqrt_direct_{t:g}'))}, range "
                                            f"{fmt(r.get(f'sqrt_range_{t:g}'))}" for t in TOLS))
    L += ["", "## Low-mode share (mean over vectors): (share of ||f_dt z||^2, share of ||z||^2, share of ||b||^2, #modes)", ""]
    for name in ("A", "B"):
        ff = [f for f in fractions if f["config"] == name]
        L.append(f"- {name}: " + "; ".join(f"{lab}: {np.mean([f[lab][0] for f in ff]):.2e}, {np.mean([f[lab][1] for f in ff]):.2e}, "
                                            f"{np.mean([f[lab][2] for f in ff]):.2e}, {ff[0][lab][3]}"
                                            for lab in ("<2lam*", "<5lam*", "<10lam*", "<0.01lam_max", "<0.05lam_max")))
    L += ["", "## Checks", ""] + [f"- {k}: {v}" for k, v in checks.items()]
    L += ["", "## Stopping rules (f_dt, dense, all configurations)", ""]
    for k, v in rules.items():
        L.append(f"- {k}: false-stop {v['false_rate']:.3f}, over-solve median {v['over_median']:.2f} "
                 f"(range {v['over_min']:.2f}-{v['over_max']:.2f}), max error at stop {v['err_max']:.1e} (n={v['n']})")
    L += ["", "## Landau-style threshold ablation (tau = 1e-12 ||T_r||, range start)", ""] + [f"- {k}: {v}" for k, v in abl.items()]
    L += ["", "## Scaling (fixed density, same seeds)", "",
          "| N | vec | cond est | ref ok | r_d(1e-6) | r_r(1e-6) | r_d(1e-8) | r_r(1e-8) | rule r_r(1e-8) (err) | "
          "direct 1e-8: actions / t [s] | range 1e-8: actions / t [s] | low share <5 ritz_min |", "|" + "---|" * 12]
    for r in scal:
        td, tr = r["timed_direct_1e-08"], r["timed_range_1e-08"]
        L.append(f"| {r['N']} | {r['vec']} | {r['cond']:.0f} | {r['reference_ok']} | {r['r_direct_1e-06']} | {r['r_range_1e-06']} | "
                 f"{r['r_direct_1e-08']} | {r['r_range_1e-08']} | {r['rule_range_1e-08']} ({r['err_rule_range_1e-08']:.1e}) | "
                 f"{td['actions']} / {td['t_total']:.2f} | {tr['actions']} / {tr['t_total']:.2f} | {r['low_share_5ritzmin']:.1e} |")
    L += ["", "Power-law fits r ~ N^alpha: " + ", ".join(f"{k}: {v['alpha']:.3f}" for k, v in fits.items()),
          "Time fits t ~ N^alpha (timed solves, d <= tol/10): " + ", ".join(f"{k}: {v['alpha']:.3f}" for k, v in tfit.items())]
    return "\n".join(L)


if __name__ == "__main__":
    main()
