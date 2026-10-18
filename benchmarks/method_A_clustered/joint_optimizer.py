"""
Joint cost-aware error-budget allocation for Method A, and the mode-dependent Hurwitz-zeta alias bound.

Problem (p fixed):   min over (xi, kc, M, rc) of T_pred = C_R N_img(rc) + C_P N p^3 + C_F M^3 log M^3 + C_0
                     subject to  eps_S(rc) + eps_F(kc) + eps_mesh(kc, M) + eps_quad(kc, M) <= tau * Lambda_lb
for every configuration of a clustering level (Lambda_lb = max_i lambda_max(Gamma_ref,ii) <= ||Gamma_ref||_2, so the
relative bound is never understated). For each candidate (xi, kc, M):
  1. eps_F (reciprocal-lattice tail), eps_mesh (a priori alias bound) and eps_quad (numerical estimate of the
     coefficient-quadrature error, NOT a certificate; ~1e-15 relative) are evaluated;
  2. the remaining real-space budget is B = tau Lambda_lb - eps_F - eps_mesh - eps_quad (per configuration);
  3. rc is the smallest radius with the certified, configuration-specific eps_S(rc) <= B;
  4. N_img(rc) is the exact full-periodic image-interaction count (i < j, all lattice shifts) of each configuration;
  5. the candidate with the smallest predicted time is selected.
kc runs over reciprocal-lattice shell radii (the discrete cutoff is exact), M over 2,3,5-smooth sizes with all retained
modes below Nyquist (m_c < M/2, eta = kc h/pi < 1). Candidates whose eps_F or eps_mesh is already below
NEGL * tau * Lambda are the last ones considered along kc and M (a finer mesh or a larger cutoff cannot pay off).

Mesh bounds (both rigorous for the existing B-spline PPPM, see test_alias_bound.py; x_a = |m_a|/M = |k_a| h/(2 pi)):
  "orig"  alpha(k) = C(p) max_a (x_a/(1 - x_a))^p,  C(p) = (1 + 2 zeta(p))^3 - 1
  "hz"    alpha(k) = prod_a (1 + sigma_p(x_a)) - 1,  sigma_p(x) = |x|^p [zeta(p, 1 + |x|) + zeta(p, 1 - |x|)]
          (Hurwitz zeta; sigma_p(x) = sum_{m != 0} |W_hat(k + 2 pi m/h)/W_hat(k)| exactly, so it is the tightest bound of
          this per-axis triangle-inequality form)
  eps_mesh = 2 rho sum_{|k| <= kc} alpha (2 + alpha) ||Khat_L(k)||_2   in both cases.

Approaches: "A" fixed 33/33/33/1 split with the orig bound (the previous benchmark), "A_hz" the same split with the hz
bound (diagnostic, separates the two effects), "B" joint with orig, "C" joint with hz.

Search over p (run_psearch_dense512.py): approach "C" evaluated separately for every order p with a per-p cost model
(PCostModel, recalibrated per p and per action implementation); the eps_S curves of one xi are shared by all p
(they do not depend on p), and only the hz mesh variant is tabulated. The defaults below reproduce results_joint/.
"""
import numpy as np
from scipy import special

import common as C
from parameter_optimizer import (FourierShells, mesh_modes, smooth_numbers, C_p, A_MAX, XI_GRID, SPLIT)
from verify_ewald_longitudinal import graded_rule

RESULTS_JOINT = C.HERE / "results_joint"
FIGURES_JOINT = C.HERE / "figures_joint"
OLD_RESULTS = C.HERE / "results"
TAU = C.EPS_TARGET
NEGL = 1e-4                      # eps below NEGL * tau * Lambda: a larger kc or M cannot pay off any more
M_CAP = 400
A_GRID = (3.0, A_MAX)            # a = xi rc range of the eps_S curve (a_tol is ~4.0-4.2 at this budget)
G_RC = 700                       # rc grid points (spacing 0.09 %); the selected rc is refined exactly by bisection
APPROACHES = ("A", "A_hz", "B", "C")
LABEL = {"A": "fixed 33 % split, original C(p) mesh bound", "A_hz": "fixed split, Hurwitz-zeta bound (diagnostic)",
         "B": "joint optimisation, original C(p) mesh bound", "C": "joint optimisation, Hurwitz-zeta mesh bound"}
VARIANT = {"A": "orig", "A_hz": "hz", "B": "orig", "C": "hz"}


# ----------------------------------------------------------------------------- alias bounds
def sigma_hz(x, p):
    """sum_{m != 0} |x/(x + m)|^p = |x|^p [zeta(p, 1 + |x|) + zeta(p, 1 - |x|)] for |x| < 1 (0 at x = 0)."""
    ax = np.abs(np.asarray(x, float))
    out = np.zeros_like(ax)
    nz = ax > 0
    out[nz] = ax[nz] ** p * (special.zeta(p, 1 + ax[nz]) + special.zeta(p, 1 - ax[nz]))
    return out


def alpha_hz(mv, M, p):
    return np.prod(1 + sigma_hz(np.abs(mv) / M, p), axis=1) - 1


def alpha_orig(mv, M, p):
    x = np.abs(mv) / M
    return C_p(p) * (x / (1 - x)).max(axis=1) ** p


def mesh_bound_one(fs, kc, M, p, rho, variant):
    """Single evaluation of the a priori mesh bound (for tests)."""
    mv, n2, mc = mesh_modes(kc, fs.L)
    if mc >= M / 2:
        return np.inf
    a = alpha_hz(mv, M, p) if variant == "hz" else alpha_orig(mv, M, p)
    return 2 * rho * float(np.sum(a * (2 + a) * fs.knorm_of_n[n2]))


# ----------------------------------------------------------------------------- per-xi tables
class XiTables:
    """eps_F on shells and eps_mesh, eps_quad on the (shell, M) grid for one xi and box (configuration independent)."""

    def __init__(self, xi, L, N, p, lam_min, tau=TAU, variants=("orig", "hz"), M_cap=M_CAP):
        self.xi, self.L, self.N, self.p = float(xi), float(L), N, p
        self.rho = N / L ** 3
        fs = self.fs = FourierShells(xi, L)
        tgt = tau * lam_min
        tail = 2 * self.rho * fs.tail_after
        ok = np.nonzero(tail <= tgt)[0]
        if ok.size == 0:
            raise ValueError("no feasible Fourier cutoff")
        first = int(ok[0])
        negl = np.nonzero(tail <= NEGL * tgt)[0]
        last = int(negl[0]) if negl.size else len(tail) - 1
        self.shell_n = fs.n[first:last + 1]
        self.kc = fs.k[first:last + 1]
        self.eps_F = tail[first:last + 1]
        self.mc = np.floor(self.kc * L / (2 * np.pi) + 1e-12).astype(int)
        mv, n2, mc_max = mesh_modes(self.kc[-1], L)
        o = np.argsort(n2, kind="stable")
        mv, n2 = mv[o], n2[o]
        self.pos = np.searchsorted(n2, self.shell_n, side="right")          # modes with |m|^2 <= shell n
        kn = fs.knorm_of_n[n2]
        dkn = self._quad_dknorm(n2)
        self.Ms = np.array(smooth_numbers(2 * self.mc[0] + 1, M_cap))
        nS, nM = len(self.kc), len(self.Ms)
        self.eps_mesh = {v: np.full((nS, nM), np.inf) for v in variants}
        self.eps_quad = {v: np.full((nS, nM), np.inf) for v in variants}
        am = np.abs(mv)
        for jM, M in enumerate(self.Ms):
            valid = (self.mc < M / 2) & (self.kc * (L / M) / np.pi < 1)
            if not valid.any():
                continue
            mmax = am.max()
            sig = sigma_hz(np.arange(mmax + 1) / M, p)
            for v in variants:
                if v == "hz":
                    a = np.prod(1 + sig[am], axis=1) - 1
                else:
                    x = am / M
                    a = C_p(p) * (x / np.maximum(1 - x, 1e-300)).max(axis=1) ** p
                W2 = np.prod(np.sinc(am / M) ** (2 * p), axis=1)
                term = np.where(W2 <= 1e-10, 1.0, a * (2 + a)) * kn
                cm = np.concatenate([[0.0], np.cumsum(term)])
                cq = np.concatenate([[0.0], np.cumsum((1 + a) ** 2 * dkn)])
                self.eps_mesh[v][valid, jM] = 2 * self.rho * cm[self.pos[valid]]
                self.eps_quad[v][valid, jM] = 2 * self.rho * cq[self.pos[valid]]

    def _quad_dknorm(self, n2):
        """|production - refined| Khat_L norm per mode (refined rule of exp1_operator_action.RefinedEwald)."""
        K = self.fs.K
        nu = np.unique(n2)
        k = 2 * np.pi / self.L * np.sqrt(nu)
        base = K.base
        saved = base.tau_rule
        try:
            base.tau_rule = graded_rule(100.0, 1e-12, 2 * 46, n=30)
            Ar, Br = K.AB_scaled(k)
        finally:
            base.tau_rule = saved
        Ap, Bp = K.AB_scaled(k)
        dmp = np.exp(-k ** 2 / (4 * self.xi ** 2))
        dA, dB = (Ap - Ar) * dmp, (Bp - Br) * dmp
        d = np.maximum(np.abs(dA), np.abs(dA + dB * k ** 2))
        out = np.zeros(nu.max() + 1)
        out[nu] = d
        return out[n2]


class RealSpaceCurve:
    """Certified eps_S(rc) of one configuration and xi on an rc grid (upper envelope, non-increasing), with exact
    bisection refinement against RealSpaceTail.eps_S."""

    def __init__(self, rst, xi, G=G_RC):
        self.rst, self.xi = rst, float(xi)
        self.prep = rst.prepare(xi)
        self.rc = np.geomspace(A_GRID[0] / xi, A_GRID[1] / xi, G)
        m = self.prep["m"]
        N = rst.N
        idx = np.searchsorted(rst.r2[:m], self.rc ** 2, side="right")       # images with index >= idx are discarded
        gi = np.searchsorted(idx, np.arange(m), side="right")                 # image -> first grid point that drops it
        nb = (G + 1) * N
        D = np.empty((G + 1, N, 6))
        cs = self.prep["comps"]
        for c in range(6):
            D[:, :, c] = (np.bincount(gi * N + rst.i[:m], cs[:, c], nb)
                          + np.bincount(gi * N + rst.j[:m], cs[:, c], nb)).reshape(G + 1, N)
        # image with gi = g is retained for grid points >= g and discarded for grid points < g
        Dt = np.cumsum(D[::-1], axis=0)[::-1][1:]                              # sum over gi > g  -> discarded at g
        T = Dt[:, :, [0, 3, 4, 3, 1, 5, 4, 5, 2]].reshape(G * N, 3, 3)
        lam = np.linalg.eigvalsh(T)[:, -1].reshape(G, N).max(axis=1)
        self.eps = 2 * (lam + self.prep["tail"])
        self.env = np.maximum.accumulate(self.eps[::-1])[::-1]                 # upper envelope, non-increasing

    def lookup(self, B):
        """Index of the smallest grid rc with envelope <= B (G means infeasible)."""
        return np.searchsorted(-self.env, -np.asarray(B), side="left")

    def refine(self, B, g):
        """Exact smallest admissible rc in (rc[g-1], rc[g]] by bisection on the exact bound."""
        while self.rst.eps_S(self.prep, self.rc[g]) > B:     # the direct sum can differ from the grid sum by
            g += 1                                              # round-off at a near-tie: step to the next grid point
            if g >= len(self.rc):
                raise RuntimeError("no admissible rc on the grid")
        hi = self.rc[g]
        lo = self.rc[g - 1] if g > 0 else 0.5 * hi
        while hi - lo > 1e-7 * hi:
            mid = 0.5 * (lo + hi)
            if self.rst.eps_S(self.prep, mid) <= B:
                hi = mid
            else:
                lo = mid
        return hi


# ----------------------------------------------------------------------------- per-p cost model
class PCostModel:
    """Cost model recalibrated per assignment order p (one CostModel-like table per p):
        T(p) = C_R N_img + [S_p + G_p M^3] + C_F M^3 log M^3 + C_0(p)
    spread/gather is measured per p at the benchmark N (S_p) with its mesh-size dependence G_p M^3 (grid allocation and
    zeroing in the scatter); C_0(p) holds the component intercepts and the per-p glue of the full action."""

    def __init__(self, C_R, C_F, per_p, **extra):
        self.C_R, self.C_F = float(C_R), float(C_F)
        self.per_p = {int(k): dict(v) for k, v in per_p.items()}
        self.extra = extra

    def parts(self, n_img, N, p, M):
        q = self.per_p[int(p)]
        M = np.asarray(M, float) if np.ndim(M) else float(M)
        return dict(real=self.C_R * n_img, spread_gather=q["S"] + q["G"] * M ** 3,
                    fft=self.C_F * M ** 3 * np.log(M ** 3), overhead=q["C_0"])

    def predict(self, n_img, N, p, M):
        return float(sum(self.parts(n_img, N, p, M).values()))

    def as_dict(self):
        return dict(C_R=self.C_R, C_F=self.C_F, per_p={str(k): v for k, v in self.per_p.items()}, **self.extra)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        return cls(d.pop("C_R"), d.pop("C_F"), d.pop("per_p"), **d)


def predict_vec(cm, nimg, N, p, Ms):
    """sum of the model parts (same operation order as the inline formula of the previous round)."""
    return sum(cm.parts(nimg, N, p, Ms).values())


# ----------------------------------------------------------------------------- one xi, all approaches
def evaluate_xi(xi, seeds, L, N, cm, p=C.P_ORDER, tau=TAU, topk=6, approaches=APPROACHES, curves=None,
                variants=("orig", "hz"), M_cap=M_CAP):
    """seeds: list of dict(name, rst, lam_lb). Returns {approach: dict(best=<exact record>, top=[light records])}
    and the shared tables (for finalising other candidates of the same xi later). curves (the eps_S curves of this xi,
    independent of p) can be passed in to share them between orders."""
    lam = np.array([s["lam_lb"] for s in seeds])
    tab = XiTables(xi, L, N, p, lam.min(), tau, variants=variants, M_cap=M_cap)
    if curves is None:
        curves = [RealSpaceCurve(s["rst"], xi) for s in seeds]
    out = {}
    for ap in approaches:
        v = VARIANT[ap]
        eM, eQ = tab.eps_mesh[v], tab.eps_quad[v]
        if ap in ("A", "A_hz"):
            tF, tM = SPLIT["F"] * tau * lam.min(), SPLIT["mesh"] * tau * lam.min()
            i = int(np.argmax(tab.eps_F <= tF))
            okM = np.nonzero(eM[i] <= tM)[0]
            if okM.size == 0:
                out[ap] = None
                continue
            ii, jj = np.array([i]), np.array([int(okM[0])])
            Bs = (SPLIT["S"] * tau * lam)[:, None]
        else:
            ok = np.isfinite(eM) & (tau * lam.min() - tab.eps_F[:, None] - eM - eQ > 0)
            ii, jj = np.nonzero(ok)
            if ii.size == 0:
                out[ap] = None
                continue
            Bs = tau * lam[:, None] - (tab.eps_F[ii] + eM[ii, jj] + eQ[ii, jj])[None, :]
        gs = np.stack([c.lookup(Bs[k]) for k, c in enumerate(curves)])      # (n_seed, n_cand)
        feas = np.all(gs < G_RC, axis=0) & np.all(Bs > 0, axis=0)
        if not feas.any():
            out[ap] = None
            continue
        rc_c = curves[0].rc[np.minimum(gs, G_RC - 1)].max(axis=0)
        nimg = np.mean([np.searchsorted(sd["rst"].r2, rc_c ** 2, side="right") for sd in seeds], axis=0)
        Ms = tab.Ms[jj].astype(float)
        T = np.where(feas, predict_vec(cm, nimg, N, p, Ms), np.inf)
        order = [o for o in np.argsort(T)[:topk] if np.isfinite(T[o])]
        top = [dict(xi=float(xi), kc=float(tab.kc[ii[o]]), M=int(tab.Ms[jj[o]]), rc_grid=float(rc_c[o]),
                    T_pred_grid=float(T[o]), i=int(ii[o]), j=int(jj[o]), B=Bs[:, o].tolist(), g=gs[:, o].tolist())
               for o in order]
        o = order[0]
        best = finalize(xi, tab, curves, seeds, int(ii[o]), int(jj[o]), Bs[:, o], gs[:, o], v, cm, N, p, tau)
        out[ap] = dict(best=best, top=top, n_candidates=int(feas.sum()))
    return out, tab, curves


def finalize(xi, tab, curves, seeds, i, j, B, g, variant, cm, N, p, tau):
    """Exact rc (bisection; max over configurations) and the certified error decomposition of one candidate."""
    rc = max(c.refine(B[k], int(g[k])) for k, c in enumerate(curves))
    kc, M = float(tab.kc[i]), int(tab.Ms[j])
    nimg = [int(np.searchsorted(sd["rst"].r2, rc * rc, side="right")) for sd in seeds]
    nuni = [sd["rst"].n_unique(rc) for sd in seeds]
    eS = [c.rst.eps_S(c.prep, rc) for c in curves]
    eF, eM, eQ = float(tab.eps_F[i]), float(tab.eps_mesh[variant][i, j]), float(tab.eps_quad[variant][i, j])
    lam = [sd["lam_lb"] for sd in seeds]
    tot = [(eS[k] + eF + eM + eQ) / lam[k] for k in range(len(seeds))]
    parts = cm.parts(np.mean(nimg), N, p, M)
    lmin = tau * min(lam)
    return dict(xi=float(xi), kc=kc, M=M, p=p, rc=rc, a=xi * rc, b=kc / (2 * xi), eta=kc * (tab.L / M) / np.pi,
                mc=int(tab.mc[i]), shell_n=int(tab.shell_n[i]), variant=variant,
                eps_F_abs=eF, eps_mesh_abs=eM, eps_quad_abs_est=eQ, eps_S_abs_per_seed=eS, lam_lb_per_seed=lam,
                cert_rel_per_seed=tot, cert_rel_max=max(tot), feasible=bool(max(tot) <= tau * (1 + 1e-12)),
                share_S=float(max(eS) / lmin), share_F=eF / lmin, share_mesh=eM / lmin, share_quad=eQ / lmin,
                n_img_per_seed=nimg, n_img_mean=float(np.mean(nimg)), n_unique_mean=float(np.mean(nuni)),
                T_pred=float(sum(parts.values())), **{f"T_pred_{k}": float(v) for k, v in parts.items()})
