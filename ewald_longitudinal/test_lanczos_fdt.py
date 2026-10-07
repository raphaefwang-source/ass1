#!/usr/bin/env python3
"""
Validation of the Lanczos matrix-function stage of the finite-time discrete-FDT thermostat.

Frozen positions q, particle graph operator Gamma_h (real-space K_S sum + PPPM long-range part,
degree-minus-weight, parameters calibrated in test_equal_accuracy_xi.py: xi = 0.7, s = 4.10,
eta_N = 0.7, p = 7, actual operator error ~1e-7 in configurations A and B).

    R_dt = exp(-dt Gamma_h/m),  eta = f_dt(Gamma_h) Pi xi,
    f_dt(l) = sqrt(beta^{-1} m [1 - exp(-2 dt l/m)]),  f_dt(0) = 0,
    debugging function g(l) = sqrt(l),  Euler: f_E(l) = sqrt(2 dt l/beta - dt^2 l^2/(beta m)).
    Pi = (I_N - 11^T/N) kron I_3.

Lanczos (full reorthogonalisation), started from q1 = z/||z||, z = Pi xi; every action is the literal
Gamma_h action (real-space pair sum + spread/FFT/G/IFFT/gather) followed by Pi.
    eta_s = ||z|| Q_s f(T_s) e1   (then projected with Pi).
Dense references use the eigendecomposition of Gamma_h restricted to Range(Pi).

Model unchanged; no Q projector; no clipping of Khat_L or of Ritz values; no stabilisation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_ewald_longitudinal import set_style, INK2  # noqa: E402
from verify_periodic_graph_psd import image_vectors  # noqa: E402
from test_parameter_selection import Config, ewald, lap_full  # noqa: E402
from test_equal_accuracy_xi import make_mesh  # noqa: E402

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
PARAMS = dict(xi=0.7, s=4.100, eta=0.7, p=7)       # accepted at tol 1e-7 for xi = 0.7 in both A and B
BETA, MASS = 1.0, 1.0
TAUS = [0.05, 0.2, 0.5, 1.0, 2.0]                   # dt * lambda_max / m
TOLS = [1e-4, 1e-6, 1e-8, 1e-10]
EPS = np.finfo(float).eps


class NegativeRitz(Exception):
    pass


# ----------------------------------------------------------------------------
# Scalar functions (evaluated on Ritz values; negative arguments are reported, never clipped)
# ----------------------------------------------------------------------------
def check_theta(theta, lam_scale):
    tmin = float(theta.min())
    if tmin < 0:
        raise NegativeRitz(f"Ritz value {tmin:.3e} (lambda_max {lam_scale:.3e}, ratio {tmin / lam_scale:.2e})")


def make_funcs(dt):
    return {
        "f_dt": lambda t: np.sqrt(MASS / BETA * (-np.expm1(-2 * dt * t / MASS))),
        "sqrt": lambda t: np.sqrt(t),
        # factored form dt l (2 - dt l/m)/beta; a negative argument gives NaN and is reported, never clipped
        "f_E": lambda t: np.sqrt(dt * t * (2.0 - dt * t / MASS) / BETA),
    }


# ----------------------------------------------------------------------------
# Gamma_h: literal action and independent dense assembly
# ----------------------------------------------------------------------------
def real_pairs(x, L, rc):
    N = len(x)
    if rc < L / 2:
        pr = cKDTree(x % L, boxsize=L).query_pairs(rc, output_type="ndarray")
        i, j = pr[:, 0], pr[:, 1]
        d = x[i] - x[j]
        d -= L * np.round(d / L)
        return i, j, d
    pi, pj = np.triu_indices(N, 1)
    d0 = x[pi] - x[pj]
    d0 -= L * np.round(d0 / L)
    dd = d0[:, None, :] + image_vectors(L, rc)[None]
    keep = np.linalg.norm(dd, axis=-1) <= rc
    return (np.broadcast_to(pi[:, None], keep.shape)[keep], np.broadcast_to(pj[:, None], keep.shape)[keep], dd[keep])


class GammaH:
    def __init__(self, x, L, xi, s, eta, p, kappa=1.0):
        t0 = time.perf_counter()
        self.x, self.L, self.N = x, L, len(x)
        K = ewald(kappa, xi)
        self.rc, self.kc = s / xi, 2 * xi * s
        self.i, self.j, self.d = real_pairs(x, L, self.rc)
        r = np.linalg.norm(self.d, axis=1)
        self.w = K.theta_S_scaled_closed(r) * np.exp(-(xi * r) ** 2) / r ** 2
        self.mesh = make_mesh(L, K, self.kc, eta, p)
        self.idx, self.wt = self.mesh.flat(x)
        self.D = self.mesh.degree(self.idx, self.wt)
        self.t_setup = time.perf_counter() - t0
        self.n_actions, self.t_actions = 0, 0.0

    def matvec(self, v):
        t0 = time.perf_counter()
        X = v.reshape(self.N, 3)
        f = (self.w * np.einsum("pa,pa->p", self.d, X[self.i] - X[self.j]))[:, None] * self.d
        out = np.stack([np.bincount(self.i, f[:, c], self.N) - np.bincount(self.j, f[:, c], self.N)
                        for c in range(3)], axis=1)
        out += self.mesh.graph_action(self.idx, self.wt, X, self.D)
        self.n_actions += 1
        self.t_actions += time.perf_counter() - t0
        return out.ravel()

    def dense(self):
        """Independent assembly: real-space pair blocks + mode-space PPPM pair matrix (no literal FFT path)."""
        N = self.N
        Bk = self.w[:, None, None] * self.d[:, :, None] * self.d[:, None, :]
        G = np.zeros((N, 3, N, 3))
        a = np.arange(3)
        for (p_, q_, sgn) in ((self.i, self.j, -1), (self.j, self.i, -1), (self.i, self.i, 1), (self.j, self.j, 1)):
            np.add.at(G, (p_[:, None, None], a[None, :, None], q_[:, None, None], a[None, None, :]), sgn * Bk)
        W, _ = self.mesh.pair_matrix(self.x)
        return G.reshape(3 * N, 3 * N) + lap_full(W)


def proj(v):
    X = v.reshape(-1, 3)
    return (X - X.mean(axis=0)).ravel()


def proj_b(V, N):
    X = V.reshape(V.shape[0], N, 3)
    return (X - X.mean(axis=1, keepdims=True)).reshape(V.shape)


# ----------------------------------------------------------------------------
# Lanczos with full reorthogonalisation
# ----------------------------------------------------------------------------
def lanczos(op, z, smax, dense=None, conv_tol=0.0):
    n = z.size
    smax = min(smax, n)
    Q = np.zeros((smax + 1, n))
    AQ = np.zeros((smax, n))
    al, be = np.zeros(smax), np.zeros(smax)
    nz = np.linalg.norm(z)
    Q[0] = z / nz
    t_re, act_dev = 0.0, 0.0
    s_done = smax
    for k in range(smax):
        w = proj(op.matvec(Q[k]))
        if dense is not None:
            ref = proj(dense @ Q[k])
            act_dev = max(act_dev, np.linalg.norm(w - ref) / np.linalg.norm(ref))
        AQ[k] = w
        al[k] = Q[k] @ w
        w = w - al[k] * Q[k] - (be[k - 1] * Q[k - 1] if k > 0 else 0.0)
        t0 = time.perf_counter()
        for _ in range(2):
            w -= Q[:k + 1].T @ (Q[:k + 1] @ w)
        w = proj(w)
        t_re += time.perf_counter() - t0
        be[k] = np.linalg.norm(w)
        if be[k] <= 1e-13 * abs(al[:k + 1]).max():       # invariant subspace (e.g. full dimension of Range(Pi))
            s_done = k + 1
            break
        Q[k + 1] = w / be[k]
    return dict(Q=Q[:s_done + 1], AQ=AQ[:s_done], al=al[:s_done], be=be[:s_done], nz=nz, s=s_done, t_reorth=t_re,
                action_dev=act_dev)


def tridiag(al, be, s):
    return np.diag(al[:s]) + np.diag(be[:s - 1], 1) + np.diag(be[:s - 1], -1)


def rank_sweep(L, fun, lam_scale):
    """eta_s for s = 1..S (projected), Ritz extrema, generalised residual beta_s |e_s^T f(T_s) e1| ||z||."""
    S, Q, nz = L["s"], L["Q"], L["nz"]
    etas, rmin, rmax, gres, neg = [], [], [], [], None
    for s in range(1, S + 1):
        th, U = np.linalg.eigh(tridiag(L["al"], L["be"], s))
        try:
            check_theta(th, lam_scale)
        except NegativeRitz as e:
            neg = (s, str(e))
            break
        fv = fun(th)
        if not np.all(np.isfinite(fv)):
            neg = (s, f"f not finite at Ritz values {th[~np.isfinite(fv)]} (lambda_max {lam_scale:.6e})")
            break
        c = U @ (fv * U[0])
        eta = proj(nz * (Q[:s].T @ c))
        etas.append(eta)
        rmin.append(th[0])
        rmax.append(th[-1])
        gres.append(nz * L["be"][s - 1] * abs(c[-1]) / max(np.linalg.norm(eta), 1e-300))
    return dict(etas=np.array(etas), ritz_min=np.array(rmin), ritz_max=np.array(rmax), gres=np.array(gres), neg=neg)


def identity_checks(L, dense):
    s, Q, AQ = L["s"], L["Q"], L["AQ"]
    Qs = Q[:s]
    T = tridiag(L["al"], L["be"], s)
    out = dict(orth=float(np.linalg.norm(Qs @ Qs.T - np.eye(s), 2)))
    Rres = AQ.T - Qs.T @ T
    if s < Q.shape[0]:
        Rres[:, -1] -= L["be"][s - 1] * Q[s]
    out["recurrence_resid"] = float(np.linalg.norm(Rres, 2) / np.linalg.norm(T, 2))
    out["T_vs_QtAQ"] = float(np.linalg.norm(T - Qs @ AQ.T, 2) / np.linalg.norm(T, 2))
    if dense is not None:
        out["T_vs_QtGQ_dense"] = float(np.linalg.norm(T - Qs @ dense @ Qs.T, 2) / np.linalg.norm(T, 2))
    return out


# ----------------------------------------------------------------------------
# Dense references on Range(Pi)
# ----------------------------------------------------------------------------
class DenseRef:
    def __init__(self, G, N):
        T = np.zeros((3 * N, 3))
        for c in range(3):
            T[c::3, c] = 1 / np.sqrt(N)
        self.P = null_space(T.T)                      # orthonormal basis of Range(Pi)
        Gs = 0.5 * (G + G.T)
        self.lam, V = np.linalg.eigh(self.P.T @ Gs @ self.P)
        self.U = self.P @ V                           # eigenvectors of Gamma_h on Range(Pi)
        self.full_eigs = np.linalg.eigvalsh(Gs)
        self.lam_star, self.lam_max = float(self.lam[0]), float(self.lam[-1])

    def apply(self, fun, z):
        return self.U @ (fun(self.lam) * (self.U.T @ z))

    def matrix(self, fun):
        return (self.U * fun(self.lam)) @ self.U.T


# ----------------------------------------------------------------------------
def required_rank(err, tol):
    """First rank after which the error stays <= tol (1-based); None if never."""
    ok = err <= tol
    if not ok[-1]:
        return None
    bad = np.where(~ok)[0]
    return int(bad[-1] + 2) if bad.size else 1


def first_rank(err, tol):
    idx = np.where(err <= tol)[0]
    return int(idx[0] + 1) if idx.size else None


RULES = {
    "d1 <= tol": lambda d1, d2, gr, tol: d1 <= tol,
    "d2 <= tol": lambda d1, d2, gr, tol: d2 <= tol,
    "d1 <= tol twice": lambda d1, d2, gr, tol: (d1 <= tol) & (np.r_[np.inf, d1[:-1]] <= tol),
    "gres <= tol": lambda d1, d2, gr, tol: gr <= tol,
    "d1 <= tol/10": lambda d1, d2, gr, tol: d1 <= tol / 10,
}


def estimators(etas):
    nrm = np.linalg.norm(etas, axis=1)
    d1 = np.r_[np.inf, np.linalg.norm(np.diff(etas, axis=0), axis=1) / nrm[1:]]
    d2 = np.r_[np.inf, np.inf, np.linalg.norm(etas[2:] - etas[:-2], axis=1) / nrm[2:]]
    return d1, d2


# ----------------------------------------------------------------------------
# Dense-configuration validation
# ----------------------------------------------------------------------------
def validate_config(cfg_name, N, L, seed, n_vec, rng, say, full=True):
    x = np.random.default_rng(seed).uniform(0, L, size=(N, 3))
    op = GammaH(x, L, **PARAMS)
    G = op.dense()
    # literal action vs independent dense assembly on random vectors
    V = rng.normal(size=(5, 3 * N))
    lit_dev = max(np.linalg.norm(op.matvec(v) - G @ v) / np.linalg.norm(G @ v) for v in V)
    ref = DenseRef(G, N)
    info = dict(config=cfg_name, N=N, L=L, seed=seed, lam_star=ref.lam_star, lam_max=ref.lam_max,
                cond=ref.lam_max / ref.lam_star, literal_vs_dense=float(lit_dev),
                sym=float(np.linalg.norm(G - G.T) / np.linalg.norm(G)),
                full_min_eig=float(ref.full_eigs[0]), full_max_eig=float(ref.full_eigs[-1]),
                M=op.mesh.M, rc=op.rc, kc=op.kc)
    runs = []
    for v in range(n_vec):
        z = proj(rng.normal(size=3 * N))
        Lz = lanczos(op, z, 3 * N, dense=G if full else None)
        idc = identity_checks(Lz, G)
        for tau in (TAUS + [1.9] if full else [0.5]):
            dt = tau * MASS / ref.lam_max
            funcs = make_funcs(dt)
            for fname in (("f_dt", "sqrt", "f_E") if full else ("f_dt",)):
                if fname == "sqrt" and tau != 0.5:
                    continue                              # g does not depend on dt
                if tau == 1.9 and fname != "f_E":
                    continue                              # 1.9: strictly stable Euler case next to the boundary 2
                sw = rank_sweep(Lz, funcs[fname], ref.lam_max)
                if sw["neg"] is not None and v == 0:
                    say(f"   f NOT EVALUABLE: {cfg_name} vec {v} {fname} tau {tau}: rank {sw['neg'][0]}: {sw['neg'][1]}")
                ex = ref.apply(funcs[fname], z)
                etas = sw["etas"]
                err = np.linalg.norm(etas - ex, axis=1) / np.linalg.norm(ex)
                d1, d2 = estimators(etas)
                mom = np.linalg.norm(etas.reshape(len(etas), N, 3).sum(axis=1), axis=1) / np.linalg.norm(etas, axis=1)
                offP = np.linalg.norm(etas - np.array([proj(e) for e in etas]), axis=1) / np.linalg.norm(etas, axis=1)
                runs.append(dict(config=cfg_name, vec=v, tau=tau, fn=fname, err=err, d1=d1, d2=d2, gres=sw["gres"],
                                 ritz_min=sw["ritz_min"], ritz_max=sw["ritz_max"], mom=mom, offPi=offP,
                                 neg=sw["neg"], S=Lz["s"], **idc, action_dev=Lz["action_dev"]))
    return info, runs, op, G, ref


def rank_stats(runs, tol):
    r = [required_rank(q["err"], tol) for q in runs]
    r = [v for v in r if v is not None]
    if not r:
        return None
    r = np.array(r)
    return dict(median=float(np.median(r)), max=int(r.max()), p90=float(np.percentile(r, 90)),
                p99=float(np.percentile(r, 99)), n=len(r), n_total=len(runs))


def rule_stats(runs, tol):
    out = {}
    for name, rule in RULES.items():
        fs, over, act = 0, [], []
        for q in runs:
            hit = np.where(rule(q["d1"], q["d2"], q["gres"], tol))[0]
            true = required_rank(q["err"], tol)
            if hit.size == 0 or true is None:
                continue
            s_stop = int(hit[0] + 1)
            e = q["err"][s_stop - 1]
            fs += e > tol
            over.append(s_stop / true)
            act.append(e)
        if act:
            out[name] = dict(false_stop_rate=fs / len(act), over_median=float(np.median(over)), over_max=float(max(over)),
                             over_min=float(min(over)), err_at_stop_median=float(np.median(act)),
                             err_at_stop_max=float(max(act)), n=len(act))
    return out


# ----------------------------------------------------------------------------
# Covariance with batched Lanczos (dense Gamma_h matrix, verified equal to the literal action)
# ----------------------------------------------------------------------------
def batched_lanczos(G, Z, S, N):
    B, n = Z.shape
    nz = np.linalg.norm(Z, axis=1)
    Q = np.zeros((B, S + 1, n))
    al, be = np.zeros((B, S)), np.zeros((B, S))
    Q[:, 0] = Z / nz[:, None]
    for k in range(S):
        w = proj_b(Q[:, k] @ G, N)
        al[:, k] = np.einsum("bn,bn->b", Q[:, k], w)
        w -= al[:, k, None] * Q[:, k]
        if k:
            w -= be[:, k - 1, None] * Q[:, k - 1]
        for _ in range(2):
            w -= np.einsum("bk,bkn->bn", np.einsum("bkn,bn->bk", Q[:, :k + 1], w), Q[:, :k + 1])
        w = proj_b(w, N)
        be[:, k] = np.linalg.norm(w, axis=1)
        Q[:, k + 1] = w / be[:, k, None]
    return Q, al, be, nz


def batched_etas(Q, al, be, nz, fun, S, N, lam_scale, ranks=None):
    B, _, n = Q.shape
    ranks = list(range(1, S + 1)) if ranks is None else sorted(ranks)
    out = {}
    rmin = np.inf
    for s in ranks:
        T = np.zeros((B, s, s))
        ii = np.arange(s)
        T[:, ii, ii] = al[:, :s]
        T[:, ii[:-1], ii[1:]] = be[:, :s - 1]
        T[:, ii[1:], ii[:-1]] = be[:, :s - 1]
        th, U = np.linalg.eigh(T)
        rmin = min(rmin, th[:, 0].min())
        check_theta(th, lam_scale)
        c = np.einsum("bij,bj->bi", U, fun(th) * U[:, 0, :])
        out[s] = proj_b(nz[:, None] * np.einsum("bi,bin->bn", c, Q[:, :s]), N)
    return out, rmin


def covariance_study(name, op, G, ref, N, tau, n_samples, S, rng, say, n_literal=1000):
    dt = tau * MASS / ref.lam_max
    fun = make_funcs(dt)["f_dt"]
    C_exact = ref.matrix(lambda l: fun(l) ** 2)
    R = ref.matrix(lambda l: np.exp(-dt * l / MASS)) + (np.eye(3 * N) - ref.P @ ref.P.T)
    Pi = ref.P @ ref.P.T
    Iden = np.eye(3 * N)
    fdt_exact_full = float(np.linalg.norm(R @ (MASS / BETA * Iden) @ R.T + C_exact - MASS / BETA * Iden) /
                           np.linalg.norm(MASS / BETA * Iden))
    fdt_exact_Pi = float(np.linalg.norm(Pi @ (R @ (MASS / BETA * Pi) @ R.T + C_exact) @ Pi - MASS / BETA * Pi) /
                         np.linalg.norm(MASS / BETA * Pi))
    ranks = [r for r in (4, 8, 12, 16, 24, 32, 48, 64, 80, 100) if r <= S]
    acc = {("rank", r): np.zeros((3 * N, 3 * N)) for r in ranks}
    acc.update({("tol", t): np.zeros((3 * N, 3 * N)) for t in TOLS})
    CD = np.zeros((3 * N, 3 * N))
    Cmax_L, Cmax_D = np.zeros((3 * N, 3 * N)), np.zeros((3 * N, 3 * N))
    mom_max, stop_ranks = 0.0, {t: [] for t in TOLS}
    snaps = []
    done, chunk = 0, 1000
    rmin_all = np.inf
    t0 = time.perf_counter()
    targets = [m for m in (1000, 10000, 100000) if m <= n_samples]
    while done < n_samples:
        B = min(chunk, n_samples - done)
        Xi = rng.normal(size=(B, 3 * N))
        Z = proj_b(Xi, N)
        Q, al, be, nz = batched_lanczos(G, Z, S, N)
        rule_mode = done < 10000               # per-sample stopping rules need every rank: first 1e4 samples only
        Ed_ = batched_etas(Q, al, be, nz, fun, S, N, ref.lam_max, ranks=None if rule_mode else ranks)
        Edict, rmin = Ed_
        rmin_all = min(rmin_all, rmin)
        Ed = (ref.U @ (fun(ref.lam)[:, None] * (ref.U.T @ Z.T))).T          # dense root, same xi
        CD += Ed.T @ Ed
        for r in ranks:
            acc[("rank", r)] += Edict[r].T @ Edict[r]
        done += B
        if not rule_mode:
            if done in targets:
                nC = np.linalg.norm(C_exact)
                row = dict(config=name, tau=tau, M=done, mc_error=float(np.linalg.norm(CD / done - C_exact) / nC),
                           rule_samples=10000)
                for r in ranks:
                    C = acc[("rank", r)]
                    row[f"krylov_rank_{r:g}"] = float(np.linalg.norm(C / done - CD / done) / nC)
                    row[f"total_rank_{r:g}"] = float(np.linalg.norm(C / done - C_exact) / nC)
                snaps.append(row)
                say(f"   {name} tau={tau} M={done}: MC {row['mc_error']:.2e}, Krylov(rank {ranks[-1]}) "
                    f"{row[f'krylov_rank_{ranks[-1]}']:.2e} ({time.perf_counter() - t0:.0f}s)")
            continue
        done -= B
        E = np.stack([Edict[s] for s in range(1, S + 1)], axis=1)
        nrm = np.linalg.norm(E, axis=2)
        d1 = np.concatenate([np.full((B, 1), np.inf), np.linalg.norm(np.diff(E, axis=1), axis=2) / nrm[:, 1:]], axis=1)
        for t in TOLS:
            ok = (d1 <= t) & (np.concatenate([np.full((B, 1), np.inf), d1[:, :-1]], axis=1) <= t)   # rule: twice
            s_stop = np.where(ok.any(axis=1), ok.argmax(axis=1), S - 1)
            Es = E[np.arange(B), s_stop]
            acc[("tol", t)] += Es.T @ Es
            stop_ranks[t] += (s_stop + 1).tolist()
            if t == 1e-8:
                mom_max = max(mom_max, float((np.linalg.norm(Es.reshape(B, N, 3).sum(axis=1), axis=1)
                                              / np.linalg.norm(Es, axis=1)).max()))
                # conditional Maxwell step: p0 ~ N(0, beta^{-1} m Pi), p1 = R p0 + eta
                P0 = np.sqrt(MASS / BETA) * proj_b(rng.normal(size=(B, 3 * N)), N)
                P1L = P0 @ R.T + Es
                P1D = P0 @ R.T + Ed
                Cmax_L += P1L.T @ P1L
                Cmax_D += P1D.T @ P1D
        done += B
        if done in targets:
            nC = np.linalg.norm(C_exact)
            row = dict(config=name, tau=tau, M=done, mc_error=float(np.linalg.norm(CD / done - C_exact) / nC))
            for (kind, v), C in acc.items():
                row[f"krylov_{kind}_{v:g}"] = float(np.linalg.norm(C / done - CD / done) / nC)
                row[f"total_{kind}_{v:g}"] = float(np.linalg.norm(C / done - C_exact) / nC)
            CL8 = acc[("tol", 1e-8)] / done
            row["fdt_dev_lanczos"] = float(np.linalg.norm(R @ (MASS / BETA * Pi) @ R.T + CL8 - MASS / BETA * Pi) /
                                           np.linalg.norm(MASS / BETA * Pi))
            row["fdt_dev_dense_same"] = float(np.linalg.norm(R @ (MASS / BETA * Pi) @ R.T + CD / done - MASS / BETA * Pi) /
                                              np.linalg.norm(MASS / BETA * Pi))
            row["maxwell_dev_lanczos"] = float(np.linalg.norm(Cmax_L / done - MASS / BETA * Pi) / np.linalg.norm(MASS / BETA * Pi))
            row["maxwell_dev_dense_same"] = float(np.linalg.norm(Cmax_D / done - MASS / BETA * Pi) / np.linalg.norm(MASS / BETA * Pi))
            row["maxwell_krylov"] = float(np.linalg.norm(Cmax_L / done - Cmax_D / done) / np.linalg.norm(MASS / BETA * Pi))
            row["stop_rank_median"] = {f"{t:g}": float(np.median(stop_ranks[t])) for t in TOLS}
            row["stop_rank_max"] = {f"{t:g}": int(max(stop_ranks[t])) for t in TOLS}
            snaps.append(row)
            say(f"   {name} tau={tau} M={done}: MC {row['mc_error']:.2e}, Krylov(tol 1e-8) {row['krylov_tol_1e-08']:.2e}, "
                f"Krylov(tol 1e-4) {row['krylov_tol_0.0001']:.2e} ({time.perf_counter() - t0:.0f}s)")
    # literal-action check of the batched path on the first samples
    lit = {}
    rs = np.random.default_rng(99)
    Xi = rs.normal(size=(n_literal, 3 * N))
    Z = proj_b(Xi, N)
    Q, al, be, nz = batched_lanczos(G, Z, S, N)
    s_fix = min(S, 48)
    Edict, _ = batched_etas(Q, al, be, nz, fun, S, N, ref.lam_max, ranks=[s_fix])
    CLb = Edict[s_fix].T @ Edict[s_fix] / n_literal
    CLl = np.zeros_like(CLb)
    for b in range(n_literal):
        Lz = lanczos(op, Z[b], s_fix)
        th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], s_fix))
        e = proj(Lz["nz"] * (Lz["Q"][:s_fix].T @ (U @ (fun(th) * U[0]))))
        CLl += np.outer(e, e) / n_literal
    lit = dict(rank=s_fix, n=n_literal, literal_vs_batched_dense=float(np.linalg.norm(CLl - CLb) / np.linalg.norm(C_exact)))
    return dict(config=name, tau=tau, fdt_exact_full=fdt_exact_full, fdt_exact_Pi=fdt_exact_Pi, ritz_min=float(rmin_all),
                mom_max=mom_max, snaps=snaps, ranks=ranks, literal_check=lit)


# ----------------------------------------------------------------------------
# System-size scaling (no dense reference)
# ----------------------------------------------------------------------------
def scaling_run(N, density, seed, n_vec, smax, say):
    L = (N / density) ** (1 / 3)
    x = np.random.default_rng(seed).uniform(0, L, size=(N, 3))
    op = GammaH(x, L, **PARAMS)
    rows = []
    rng = np.random.default_rng(seed + 1)
    for v in range(n_vec):
        z = proj(rng.normal(size=3 * N))
        op.n_actions, op.t_actions = 0, 0.0
        t0 = time.perf_counter()
        Lz = lanczos(op, z, smax)
        t_lan = time.perf_counter() - t0
        lam_max = None
        th_all = np.linalg.eigvalsh(tridiag(Lz["al"], Lz["be"], Lz["s"]))
        lam_max, ritz_min = float(th_all[-1]), float(th_all[0])
        idc = identity_checks(Lz, None)
        out = dict(N=N, L=L, vec=v, M=op.mesh.M, n_pairs=int(len(op.w)), s_run=Lz["s"], lam_max_est=lam_max,
                   ritz_min_final=ritz_min, cond_est=lam_max / ritz_min, t_setup=op.t_setup,
                   t_action_per=op.t_actions / max(op.n_actions, 1), t_actions=op.t_actions, t_reorth=Lz["t_reorth"],
                   t_lanczos=t_lan, **idc)
        for tau in (0.5, 2.0):
            dt = tau * MASS / lam_max
            t1 = time.perf_counter()
            sw = rank_sweep(Lz, make_funcs(dt)["f_dt"], lam_max)
            t_f = time.perf_counter() - t1
            etas = sw["etas"]
            refv = etas[-1]
            err = np.linalg.norm(etas - refv, axis=1) / np.linalg.norm(refv)
            d1, d2 = estimators(etas)
            for tol in (1e-6, 1e-8):
                rr = required_rank(err[:-5], tol) if len(err) > 5 else None
                hit = np.where((d1 <= tol) & (np.r_[np.inf, d1[:-1]] <= tol))[0]
                out[f"rank_true_tau{tau}_{tol:g}"] = rr
                out[f"rank_rule_tau{tau}_{tol:g}"] = int(hit[0] + 1) if hit.size else None
                out[f"err_at_rule_tau{tau}_{tol:g}"] = float(err[hit[0]]) if hit.size else None
            out[f"final_d1_tau{tau}"] = float(d1[-1])
            out[f"t_feval_tau{tau}"] = t_f
            out[f"mom_tau{tau}"] = float(np.linalg.norm(refv.reshape(N, 3).sum(axis=0)) / np.linalg.norm(refv))
        rows.append(out)
        say(f"   N={N} L={L:.2f} M={op.mesh.M} vec {v}: s_run={Lz['s']}, lam_max~{lam_max:.3f}, ritz_min {ritz_min:.2e}, "
            f"rank(1e-6/1e-8, tau .5) true {out['rank_true_tau0.5_1e-06']}/{out['rank_true_tau0.5_1e-08']} "
            f"rule {out['rank_rule_tau0.5_1e-06']}/{out['rank_rule_tau0.5_1e-08']}; t_action {out['t_action_per'] * 1e3:.1f}ms "
            f"x {Lz['s']}, reorth {Lz['t_reorth']:.2f}s, Lanczos {t_lan:.2f}s")
    return rows


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nvec", type=int, default=32)
    ap.add_argument("--cov-samples", type=int, default=100000)
    ap.add_argument("--scaling-N", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4000])
    ap.add_argument("--smax", type=int, default=400)
    args = ap.parse_args()
    out = HERE / "lanczos_fdt_results"
    out.mkdir(exist_ok=True)
    log = []
    say = lambda *a: (print(*a, flush=True), log.append(" ".join(str(t) for t in a)))
    rng = np.random.default_rng(2026)
    T0 = time.perf_counter()

    # ---- dense configurations A and B ---------------------------------------
    say("== dense validation (A, B) ==")
    infos, runs, keep = [], [], {}
    for name, N, L, seed in (("A", 30, 6.0, 7), ("B", 40, 10.0, 13)):
        info, rr, op, G, ref = validate_config(name, N, L, seed, args.nvec, rng, say)
        infos.append(info)
        runs += rr
        keep[name] = (op, G, ref, N)
        say(f"  {name}: lambda_* {info['lam_star']:.4e}, lambda_max {info['lam_max']:.4e}, cond {info['cond']:.1f}, "
            f"literal-vs-dense {info['literal_vs_dense']:.1e}, full-space min eig {info['full_min_eig']:.1e}")

    # ---- gap dependence (additional dense configurations) --------------------
    say("\n== spectral-gap dependence ==")
    extra = [("E1", 30, 6.0, 1), ("E2", 30, 6.0, 2), ("E3", 40, 10.0, 3), ("E4", 40, 10.0, 4), ("E5", 50, 8.0, 5),
             ("E6", 50, 12.0, 6), ("E7", 20, 5.0, 8), ("E8", 60, 9.0, 9), ("E9", 36, 7.0, 10), ("E10", 45, 11.0, 11)]
    gap_rows = []
    for name, N, L, seed in [("A", 30, 6.0, 7), ("B", 40, 10.0, 13)] + extra:
        if name in keep:
            info = next(i for i in infos if i["config"] == name)
            rr = [q for q in runs if q["config"] == name and q["fn"] == "f_dt" and q["tau"] == 0.5]
        else:
            info, rr, _, _, _ = validate_config(name, N, L, seed, 16, rng, say, full=False)
        row = dict(config=name, N=N, L=L, lam_star=info["lam_star"], lam_max=info["lam_max"], cond=info["cond"])
        for tol in TOLS:
            st = rank_stats(rr, tol)
            row[f"rank_median_{tol:g}"] = st["median"] if st else None
            row[f"rank_max_{tol:g}"] = st["max"] if st else None
        row["dim"] = 3 * N - 3
        gap_rows.append(row)
        say(f"  {name}: N={N} L={L:g} cond={info['cond']:.1f}  median rank 1e-6/1e-8/1e-10: "
            f"{row['rank_median_1e-06']}/{row['rank_median_1e-08']}/{row['rank_median_1e-10']} (dim {3 * N - 3})")

    # ---- statistics tables ----------------------------------------------------
    stat_rows = []
    for name in ("A", "B"):
        for fn in ("f_dt", "sqrt", "f_E"):
            for tau in TAUS + [1.9]:
                rr = [q for q in runs if q["config"] == name and q["fn"] == fn and q["tau"] == tau]
                if not rr:
                    continue
                for tol in TOLS:
                    st = rank_stats(rr, tol)
                    rs = rule_stats(rr, tol)
                    stat_rows.append(dict(config=name, fn=fn, tau=tau, tol=tol, ranks=st, rules=rs))
    negs = [q for q in runs if q["neg"] is not None]
    checks = dict(
        orth_max=max(q["orth"] for q in runs), recurrence_max=max(q["recurrence_resid"] for q in runs),
        T_vs_QtAQ_max=max(q["T_vs_QtAQ"] for q in runs), T_vs_QtGQ_max=max(q["T_vs_QtGQ_dense"] for q in runs),
        action_dev_max=max(q["action_dev"] for q in runs), mom_max=max(float(q["mom"].max()) for q in runs),
        offPi_max=max(float(q["offPi"].max()) for q in runs),
        n_runs_f_not_evaluable=len(negs),
        f_not_evaluable_cases=sorted({(q["config"], q["fn"], q["tau"]) for q in negs}),
        n_runs_negative_ritz=sum(q["neg"][1].startswith("Ritz value") for q in negs),
        ritz_min_over_lam_star={n: float(min(q["ritz_min"].min() for q in runs if q["config"] == n) /
                                         next(i["lam_star"] for i in infos if i["config"] == n)) for n in ("A", "B")},
        ritz_max_over_lam_max={n: float(max(q["ritz_max"].max() for q in runs if q["config"] == n) /
                                         next(i["lam_max"] for i in infos if i["config"] == n)) for n in ("A", "B")},
        worst_final_err_evaluable_runs=max(float(q["err"].min()) for q in runs if q["neg"] is None))
    say("\n== Lanczos identity / momentum checks ==")
    for k, v in checks.items():
        say(f"  {k}: {v}")

    # ---- covariance -------------------------------------------------------------
    say("\n== discrete-FDT covariance (tau = dt lambda_max/m = 0.5) ==")
    cov = []
    for name in ("A", "B"):
        op, G, ref, N = keep[name]
        S = min(3 * N - 3 - 2, 100)
        cov.append(covariance_study(name, op, G, ref, N, 0.5, args.cov_samples, S, rng, say))
        c = cov[-1]
        say(f"  {name}: exact FDT identity {c['fdt_exact_full']:.1e} (full) {c['fdt_exact_Pi']:.1e} (Range Pi); "
            f"literal-vs-batched covariance {c['literal_check']['literal_vs_batched_dense']:.1e}; "
            f"momentum max {c['mom_max']:.1e}; min Ritz {c['ritz_min']:.3e}")

    # ---- system-size scaling ----------------------------------------------------
    say("\n== system-size scaling (density of A, same spatial parameters) ==")
    scal = []
    dens = 30 / 216.0
    for N in args.scaling_N:
        scal += scaling_run(N, dens, 1000 + N, 2, args.smax, say)

    # ---- save -------------------------------------------------------------------
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, float) and not np.isfinite(o):
            return str(o)
        return o

    (out / "configs.json").write_text(json.dumps(clean(infos), indent=1) + "\n")
    (out / "rank_and_rule_stats.json").write_text(json.dumps(clean(stat_rows), indent=1) + "\n")
    (out / "gap_dependence.json").write_text(json.dumps(clean(gap_rows), indent=1) + "\n")
    (out / "checks.json").write_text(json.dumps(clean(checks), indent=1) + "\n")
    (out / "covariance.json").write_text(json.dumps(clean(cov), indent=1) + "\n")
    (out / "scaling.json").write_text(json.dumps(clean(scal), indent=1) + "\n")
    (out / "runs_compact.json").write_text(json.dumps(clean([{k: q[k] for k in ("config", "vec", "tau", "fn", "err", "d1",
                                                                                  "d2", "gres", "ritz_min", "ritz_max",
                                                                                  "mom", "offPi")} for q in runs
                                                             if q["vec"] < 4]), indent=0) + "\n")

    # ---- plots --------------------------------------------------------------------
    set_style()
    info_by = {i["config"]: i for i in infos}

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, name in zip(axs, ("A", "B")):
        for ci, tau in enumerate(TAUS):
            rr = [q for q in runs if q["config"] == name and q["fn"] == "f_dt" and q["tau"] == tau]
            S = min(len(q["err"]) for q in rr)
            E = np.array([q["err"][:S] for q in rr])
            s = np.arange(1, S + 1)
            ax.semilogy(s, np.median(E, 0), color=PAL[ci], label=f"f_dt, dt lam_max/m = {tau}")
            ax.fill_between(s, E.min(0), E.max(0), color=PAL[ci], alpha=0.15, lw=0)
        rr = [q for q in runs if q["config"] == name and q["fn"] == "sqrt"]
        S = min(len(q["err"]) for q in rr)
        ax.semilogy(np.arange(1, S + 1), np.median([q["err"][:S] for q in rr], 0), "--", color=PAL[6], label="sqrt (debug)")
        ax.set(xlabel="Lanczos rank s", ylabel="relative action error", ylim=(1e-16, 2),
               title=f"1. config {name}: action error (median, band = min..max over {args.nvec} vectors)")
        ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "plot01_error_vs_rank.png"), plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, name in zip(axs, ("A", "B")):
        for ci, tau in enumerate(TAUS):
            st = [r for r in stat_rows if r["config"] == name and r["fn"] == "f_dt" and r["tau"] == tau]
            ax.semilogx([r["tol"] for r in st], [r["ranks"]["median"] for r in st], "o-", color=PAL[ci],
                        label=f"tau = {tau}: median")
            ax.semilogx([r["tol"] for r in st], [r["ranks"]["max"] for r in st], "^:", color=PAL[ci], ms=4)
        ax.axhline(3 * keep[name][3] - 3, color=INK2, lw=0.8, label="dim Range(Pi)")
        ax.invert_xaxis()
        ax.set(xlabel="tolerance", ylabel="required rank (median; dotted = max)", title=f"2. config {name}: rank vs tolerance")
        ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "plot02_rank_vs_tol.png"), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ci, tol in enumerate((1e-6, 1e-8, 1e-10)):
        ok = [g for g in gap_rows if g[f"rank_median_{tol:g}"] is not None]
        ax.plot([np.log10(g["cond"]) for g in ok], [g[f"rank_median_{tol:g}"] for g in ok], "o", color=PAL[ci],
                label=f"tol {tol:g}")
        xx = np.array([np.log10(g["cond"]) for g in ok])
        yy = np.array([g[f"rank_median_{tol:g}"] for g in ok])
        if len(ok) > 2:
            cc = np.corrcoef(np.sqrt(10 ** xx), yy)[0, 1]
            ax.text(0.02, 0.95 - 0.06 * ci, f"tol {tol:g}: corr(rank, sqrt(cond)) = {cc:.2f}", transform=ax.transAxes,
                    color=INK2, fontsize=8)
    ax.set(xlabel="log10(lambda_max / lambda_*)", ylabel="median required rank (f_dt, tau = 0.5)",
           title="3. Required rank vs spectral condition ratio (12 dense configurations)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(), fig.savefig(out / "plot03_rank_vs_cond.png"), plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, name in zip(axs, ("A", "B")):
        rr = [q for q in runs if q["config"] == name and q["fn"] == "f_dt" and q["tau"] == 0.5]
        for key, col, lab in (("d1", PAL[0], "d_s (successive)"), ("d2", PAL[1], "d2_s (two-step)"),
                              ("gres", PAL[2], "generalised residual")):
            xs = np.concatenate([q["err"][2:] for q in rr])
            ys = np.concatenate([q[key][2:] for q in rr])
            m = (xs > 1e-16) & (ys > 1e-17) & np.isfinite(ys)
            ax.loglog(xs[m], ys[m], ".", ms=2, color=col, alpha=0.5, label=lab)
        ax.plot([1e-16, 1], [1e-16, 1], color=INK2, lw=0.8, label="y = x")
        ax.set(xlabel="true action error", ylabel="estimator", title=f"4. config {name}: estimators vs true error (tau 0.5)")
        ax.legend(fontsize=7, markerscale=4)
    fig.tight_layout(), fig.savefig(out / "plot04_estimators.png"), plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, name in zip(axs, ("A", "B")):
        rr = [q for q in runs if q["config"] == name and q["fn"] == "f_dt" and q["tau"] == 0.5]
        for q in rr[:8]:
            s = np.arange(1, len(q["ritz_min"]) + 1)
            ax.semilogy(s, q["ritz_min"], color=PAL[0], lw=0.8)
            ax.semilogy(s, q["ritz_max"], color=PAL[1], lw=0.8)
        ax.axhline(info_by[name]["lam_star"], color=PAL[0], ls="--", lw=1, label="lambda_* (dense)")
        ax.axhline(info_by[name]["lam_max"], color=PAL[1], ls="--", lw=1, label="lambda_max (dense)")
        ax.set(xlabel="rank s", ylabel="Ritz value", title=f"5. config {name}: smallest / largest Ritz value (8 vectors)")
        ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plot05_ritz_extrema.png"), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for ci, name in enumerate(("A", "B")):
        rr = [q for q in runs if q["config"] == name and q["fn"] == "f_dt" and q["tau"] == 0.5]
        S = min(len(q["mom"]) for q in rr)
        ax.semilogy(np.arange(1, S + 1), np.max([q["mom"][:S] for q in rr], 0), color=PAL[ci],
                    label=f"{name}: |total momentum| / ||eta_s|| (max)")
        ax.semilogy(np.arange(1, S + 1), np.maximum(np.max([q["offPi"][:S] for q in rr], 0), 1e-18), "--", color=PAL[ci],
                    label=f"{name}: ||(I - Pi) eta_s|| / ||eta_s|| (max)")
    ax.set(xlabel="rank s", ylabel="relative residual", title="6. Translation / momentum residual vs rank", ylim=(1e-18, 1e-12))
    ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "plot06_momentum.png"), plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, c in zip(axs, cov):
        last = c["snaps"][-1]
        rk = c["ranks"]
        ax.semilogy(rk, [last[f"krylov_rank_{r:g}"] for r in rk], "o-", color=PAL[0], label="Krylov part (vs dense, same samples)")
        ax.semilogy(rk, [last[f"total_rank_{r:g}"] for r in rk], "s--", color=PAL[1], label="total (vs C_exact)")
        ax.axhline(last["mc_error"], color=INK2, lw=0.8, label=f"Monte Carlo floor (M = {last['M']})")
        ax.set(xlabel="fixed Lanczos rank", ylabel="||C_hat - C_ref||_F / ||C_exact||_F",
               title=f"7. config {c['config']}: covariance error vs rank")
        ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "plot07_cov_vs_rank.png"), plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, c in zip(axs, cov):
        Ms = [sn["M"] for sn in c["snaps"]]
        for ci, t in enumerate(TOLS):
            sn_ = [sn for sn in c["snaps"] if f"krylov_tol_{t:g}" in sn]
            ax.loglog([sn["M"] for sn in sn_], [max(sn[f"krylov_tol_{t:g}"], 1e-18) for sn in sn_], "o-", color=PAL[ci],
                      label=f"Krylov part, stop tol {t:g}")
        ax.loglog(Ms, [sn[f"krylov_rank_{c['ranks'][-1]}"] for sn in c["snaps"]], "x:", color=PAL[5],
                  label=f"Krylov part, fixed rank {c['ranks'][-1]}")
        ax.loglog(Ms, [sn["mc_error"] for sn in c["snaps"]], "k--", lw=1, label="Monte Carlo (dense, same samples)")
        ax.set(xlabel="number of samples", ylabel="relative covariance error",
               title=f"8. config {c['config']}: Krylov vs Monte Carlo error")
        ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "plot08_krylov_vs_mc.png"), plt.close(fig)

    Ns = sorted({r["N"] for r in scal})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ci, (tau, tol) in enumerate(((0.5, 1e-6), (0.5, 1e-8), (2.0, 1e-6), (2.0, 1e-8))):
        ax.plot(Ns, [np.nanmax([r[f"rank_true_tau{tau}_{tol:g}"] or np.nan for r in scal if r["N"] == n]) for n in Ns],
                "o-", color=PAL[ci], label=f"tau {tau}, tol {tol:g} (vs converged Lanczos)")
    ax.set_xscale("log")
    ax.set(xlabel="N", ylabel="required rank (max over vectors)", title="9. Required Lanczos rank vs N (fixed density)")
    ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plot09_rank_vs_N.png"), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, col, lab in (("t_action_per", PAL[0], "one Gamma_h action"), ("t_actions", PAL[1], "all actions (full run)"),
                          ("t_reorth", PAL[2], "reorthogonalisation (full run)"), ("t_lanczos", PAL[3], "Lanczos total (full run)")):
        ax.loglog(Ns, [np.mean([r[key] for r in scal if r["N"] == n]) for n in Ns], "o-", color=col, label=lab)
    nn = np.array(Ns, float)
    ax.loglog(nn, 1e-5 * nn * np.log(nn), color=INK2, lw=0.8, label="~ N log N (reference slope)")
    ax.set(xlabel="N", ylabel="wall time [s]", title="10. Wall time vs N (run length s_run shown in table)")
    ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(out / "plot10_time_vs_N.png"), plt.close(fig)

    # ---- report -------------------------------------------------------------------
    rep = report(infos, stat_rows, gap_rows, checks, cov, scal, negs)
    say("\n" + rep)
    (out / "summary.md").write_text(rep + "\n")
    say(f"\nwall time {time.perf_counter() - T0:.0f}s")
    (out / "run_log.txt").write_text("\n".join(log) + "\n")


def report(infos, stat_rows, gap_rows, checks, cov, scal, negs):
    L = ["# Lanczos discrete-FDT thermostat: summary", ""]
    L.append(f"Gamma_h parameters: {PARAMS} (accepted at operator tol 1e-7 in A and B); beta = m = 1.")
    for i in infos:
        L.append(f"- config {i['config']}: N={i['N']} L={i['L']:g}, lambda_* = {i['lam_star']:.4e}, lambda_max = "
                 f"{i['lam_max']:.4e}, lambda_max/lambda_* = {i['cond']:.1f}; literal vs dense action {i['literal_vs_dense']:.1e}")
    L += ["", "## Required rank (sustained, f_dt), median / 90% / 99% / max over vectors", "",
          "| config | fn | dt lam_max/m | 1e-4 | 1e-6 | 1e-8 | 1e-10 |", "|---|---|---|---|---|---|---|"]
    for name in ("A", "B"):
        for fn in ("f_dt", "sqrt", "f_E"):
            for tau in TAUS + [1.9]:
                rr = [r for r in stat_rows if r["config"] == name and r["fn"] == fn and r["tau"] == tau]
                if not rr:
                    continue
                cells = [(f"{r['ranks']['median']:.0f}/{r['ranks']['p90']:.0f}/{r['ranks']['p99']:.0f}/{r['ranks']['max']}"
                          if r["ranks"] else "n/a") for r in rr]
                L.append(f"| {name} | {fn} | {tau} | " + " | ".join(cells) + " |")
    L += ["", "## Stopping rules (f_dt, all tau pooled): false-stop rate, over-solving s_stop/s_true, error at stop", ""]
    for name in ("A", "B"):
        for tol in TOLS:
            rr = [r for r in stat_rows if r["config"] == name and r["fn"] == "f_dt" and r["tol"] == tol]
            for rule in RULES:
                vals = [r["rules"][rule] for r in rr if rule in r["rules"]]
                if not vals:
                    continue
                n = sum(v["n"] for v in vals)
                fs = sum(v["false_stop_rate"] * v["n"] for v in vals) / n
                L.append(f"- {name} tol {tol:g} [{rule}]: false-stop {fs:.3f}, over-solve median "
                         f"{np.median([v['over_median'] for v in vals]):.2f} (range {min(v['over_min'] for v in vals):.2f}-"
                         f"{max(v['over_max'] for v in vals):.2f}), error at stop median "
                         f"{np.median([v['err_at_stop_median'] for v in vals]):.1e}, max {max(v['err_at_stop_max'] for v in vals):.1e}")
    L += ["", "## Gap dependence (f_dt, tau = 0.5, median rank)", "", "| config | N | L | cond | 1e-6 | 1e-8 | 1e-10 | dim |",
          "|---|---|---|---|---|---|---|---|"]
    for g in sorted(gap_rows, key=lambda g: g["cond"]):
        L.append(f"| {g['config']} | {g['N']} | {g['L']:g} | {g['cond']:.1f} | {g['rank_median_1e-06']} | "
                 f"{g['rank_median_1e-08']} | {g['rank_median_1e-10']} | {g['dim']} |")
    L += ["", "## Checks", ""] + [f"- {k}: {v}" for k, v in checks.items()]
    L += ["", "## Covariance (tau = 0.5)", ""]
    for c in cov:
        L.append(f"- config {c['config']}: exact FDT identity residual {c['fdt_exact_full']:.1e} (full), "
                 f"{c['fdt_exact_Pi']:.1e} (Range Pi); literal-action vs batched covariance "
                 f"{c['literal_check']['literal_vs_batched_dense']:.1e} ({c['literal_check']['n']} samples, rank "
                 f"{c['literal_check']['rank']}); min Ritz {c['ritz_min']:.3e}; max |momentum|/||eta|| {c['mom_max']:.1e}")
        for sn in c["snaps"]:
            if "krylov_tol_1e-08" not in sn:
                L.append(f"  - M = {sn['M']} (fixed ranks only): Monte Carlo {sn['mc_error']:.2e}; Krylov part at rank "
                         + ", ".join(f"{r}: {sn[f'krylov_rank_{r}']:.1e}" for r in c["ranks"]))
                continue
            L.append(f"  - M = {sn['M']}: Monte Carlo {sn['mc_error']:.2e}; Krylov part at stop tol 1e-4/1e-6/1e-8/1e-10: "
                     f"{sn['krylov_tol_0.0001']:.1e}/{sn['krylov_tol_1e-06']:.1e}/{sn['krylov_tol_1e-08']:.1e}/"
                     f"{sn['krylov_tol_1e-10']:.1e}; FDT deviation Lanczos {sn['fdt_dev_lanczos']:.2e} vs dense-same "
                     f"{sn['fdt_dev_dense_same']:.2e}; conditional Maxwell deviation Lanczos {sn['maxwell_dev_lanczos']:.2e} vs "
                     f"dense-same {sn['maxwell_dev_dense_same']:.2e} (Krylov part {sn['maxwell_krylov']:.1e}); stop ranks "
                     f"median {sn['stop_rank_median']}")
    L += ["", "## System-size scaling (density 30/216, same spatial parameters)", "",
          "| N | M | s_run | lam_max est | min Ritz | cond est | rank 1e-6 (tau .5) | rank 1e-8 (tau .5) | rank 1e-8 (tau 2) | "
          "rule rank 1e-8 | t_action [ms] | t_actions [s] | t_reorth [s] | t_Lanczos [s] |", "|" + "---|" * 14]
    for r in scal:
        L.append(f"| {r['N']} | {r['M']} | {r['s_run']} | {r['lam_max_est']:.3f} | {r['ritz_min_final']:.2e} | "
                 f"{r['cond_est']:.0f} | {r['rank_true_tau0.5_1e-06']} | {r['rank_true_tau0.5_1e-08']} | "
                 f"{r['rank_true_tau2.0_1e-08']} | {r['rank_rule_tau0.5_1e-08']} | {r['t_action_per'] * 1e3:.2f} | "
                 f"{r['t_actions']:.2f} | {r['t_reorth']:.2f} | {r['t_lanczos']:.2f} |")
    if negs:
        L.append(f"\nf NOT EVALUABLE in {len(negs)} runs (no negative Ritz value among them: "
                 f"{sum(q['neg'][1].startswith('Ritz value') for q in negs)} negative-Ritz cases): "
                 + "; ".join(f"{q['config']} {q['fn']} dt lam_max/m={q['tau']}: rank {q['neg'][0]}, {q['neg'][1]}"
                             for q in negs[:6]) + (" ..." if len(negs) > 6 else ""))
    return "\n".join(L)


if __name__ == "__main__":
    main()
