"""
Accuracy of Method A against the independent dense FULL-PERIODIC reference Gamma_ref (v2.LatticeFriction, k = 0
retained, no minimum image), and the measured error components.

Measured quantities (all against Gamma_ref, never against another approximation):
    E_op = ||Gamma_h - Gamma_ref||_2 / ||Gamma_ref||_2             (primary constraint, <= 2e-7)
    E_op_k0 = ||Gamma_h - Gamma_ref||_2 / ||Gamma_ref - k=0 term||_2
    E_fro, symmetry residual, translation-nullspace residual ||Gamma_h T||/(||Gamma_h|| ||T||),
    lambda_min of Gamma_h and Gamma_ref on Range(Pi) (zero-momentum subspace),
    action errors on fixed probe vectors X: absolute ||dGamma X||, operator-normalised ||dGamma X||/(||Gamma_ref|| ||X||),
    and ||dGamma X||/||Gamma_ref X|| (unstable near the null space: reported, not used as a criterion).
Gamma_h is assembled for testing only: by applying the cached matrix-free action to the 3N standard basis vectors
(dense_by_basis) and, independently, by the project's mode-space assembly (GammaH.dense).

Components (exact decomposition, Gamma_S from all images up to R_far = 7/xi plus an analytic tail <= 1e-15):
    real    ||Gamma_S^{rc} - Gamma_S||_2 = ||Lap(K_S images with rc < r <= R_far)||_2
    Fourier ||Gamma_L^{kc} - Gamma_L||_2,   Gamma_L = Gamma_ref - Gamma_S,   Gamma_L^{kc} the mesh-free lattice sum
    mesh    ||Gamma_L^{mesh} - Gamma_L^{kc}||_2
The empirical equal-accuracy search (a DIAGNOSTIC, separate from the certified selection) chooses, per xi, the
smallest rc, the smallest lattice-shell kc and the smallest smooth M whose MEASURED components are <= 0.33 eps
||Gamma_ref||_2 each, so the measured total is <= 0.99 eps by the triangle inequality.
"""
import numpy as np
from scipy import linalg

import common as C
from cached_operator import CachedGammaH, mesh_for
from parameter_optimizer import RealSpaceTail, mesh_modes, smooth_numbers, SPLIT, A_MAX, R_PAD
from test_parameter_selection import lap_full

PAIRS6 = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
REF_REL_TOL = 1e-13         # LatticeFriction far-sum truncation (default 1e-10 changes Gamma_ref by ~1e-11 relative)


# ----------------------------------------------------------------------------- reference
def reference_cached(name, x, L):
    """Gamma_ref from raw/<name>_Gref.npy (regenerated deterministically if missing)."""
    C.RAW.mkdir(parents=True, exist_ok=True)
    f = C.RAW / f"{name}_Gref.npy"
    if f.exists():
        return np.load(f)
    G, _ = C.reference(x, L, rel_tol=REF_REL_TOL)
    np.save(f, G)
    return G


def reference_summary(G, N, lat):
    T = np.kron(np.ones((N, 1)), np.eye(3)) / np.sqrt(N)
    Q = linalg.null_space(T.T)
    lam = linalg.eigvalsh(Q.T @ (0.5 * (G + G.T)) @ Q)
    nrm = float(np.linalg.norm(G, 2))
    return dict(norm2=nrm, lam_min_PiRange=float(lam[0]), lam_max=float(lam[-1]),
                norm2_k_nonzero=float(np.linalg.norm(G - C.zero_mode_matrix(lat, N), 2)),
                k0_shift=float(N * lat.zero_mode), sym=float(np.linalg.norm(G - G.T) / np.linalg.norm(G)),
                null=float(np.abs(G @ T).max() / np.abs(G).max())), Q


# ----------------------------------------------------------------------------- probes
def probes(x, L, N, Gref, Q, rng):
    """Fixed probe vectors (same for every parameter set of one configuration)."""
    P = {}
    for k in range(3):
        P[f"gauss{k}"] = rng.standard_normal(3 * N)
    for k in range(3):
        g = rng.standard_normal((N, 3))
        P[f"zeromean{k}"] = (g - g.mean(0)).ravel()
    kmin = 2 * np.pi / L
    for a in range(3):
        for f, ph in (("cos", np.cos(kmin * x[:, a])), ("sin", np.sin(kmin * x[:, a]))):
            for b in range(3):
                X = np.zeros((N, 3))
                X[:, b] = ph
                X -= X.mean(0)
                P[f"{'long' if a == b else 'trans'}_k{'xyz'[a]}_{f}_e{'xyz'[b]}"] = X.ravel()
    lam, U = linalg.eigh(Q.T @ (0.5 * (Gref + Gref.T)) @ Q, subset_by_index=[0, 5])
    for k in range(3):
        v = Q @ (U @ rng.standard_normal(U.shape[1]))
        v /= np.linalg.norm(v)
        z = rng.standard_normal(3 * N)
        z = (z.reshape(N, 3) - z.reshape(N, 3).mean(0)).ravel()
        P[f"slow{k}"] = v + 0.1 * z / np.linalg.norm(z)
    T = np.zeros((N, 3))
    T[:, 0] = 1.0
    P["translation_x"] = T.ravel()
    return P


def probe_class(name):
    for c in ("gauss", "zeromean", "long", "translation", "trans", "slow"):    # "translation" before "trans"
        if name.startswith(c):
            return c
    return name


# ----------------------------------------------------------------------------- metrics
def operator_metrics(Gh, Gref, ref_sum, Q, P, N):
    diff = Gh - Gref
    nd = float(np.linalg.norm(diff, 2))
    T = np.kron(np.ones((N, 1)), np.eye(3)) / np.sqrt(N)
    lam_h = linalg.eigvalsh(Q.T @ (0.5 * (Gh + Gh.T)) @ Q, subset_by_index=[0, 0])[0]
    out = dict(E_op=nd / ref_sum["norm2"], E_op_abs=nd, E_op_k0=nd / ref_sum["norm2_k_nonzero"],
               E_fro=float(np.linalg.norm(diff) / np.linalg.norm(Gref)),
               sym_resid=float(np.linalg.norm(Gh - Gh.T) / np.linalg.norm(Gh)),
               null_resid=float(np.linalg.norm(Gh @ T) / np.linalg.norm(Gh, 2)),
               lam_min_h=float(lam_h), lam_min_ref=ref_sum["lam_min_PiRange"])
    act = {}
    for name, X in P.items():
        dx = diff @ X
        gx = np.linalg.norm(Gref @ X)
        act[name] = dict(abs=float(np.linalg.norm(dx)),
                         op_normalised=float(np.linalg.norm(dx) / (ref_sum["norm2"] * np.linalg.norm(X))),
                         rel=float(np.linalg.norm(dx) / gx) if gx > 1e-12 * ref_sum["norm2"] * np.linalg.norm(X) else None)
    for c in ("gauss", "zeromean", "long", "trans", "slow", "translation"):
        sel = [v for k, v in act.items() if probe_class(k) == c]
        if sel:
            out[f"act_opnorm_max_{c}"] = max(v["op_normalised"] for v in sel)
            out[f"act_abs_max_{c}"] = max(v["abs"] for v in sel)
            rels = [v["rel"] for v in sel if v["rel"] is not None]
            out[f"act_rel_max_{c}"] = max(rels) if rels else None
    return out, act


def dense_independent(op):
    """Project's independent assembly of the same operator (real-space blocks + mode-space PPPM pair matrix)."""
    return op.existing_gammah().dense()


# ----------------------------------------------------------------------------- components
class Components:
    """Exact Gamma_S / Gamma_L split of one configuration for one xi, and the measured component errors."""

    def __init__(self, x, L, Gref, xi, rst=None):
        self.x, self.L, self.N, self.xi = np.mod(x, L), float(L), len(x), float(xi)
        self.K = C.kernel(xi)
        self.rst = rst or RealSpaceTail(self.x, L, (A_MAX + R_PAD) / xi)
        self.R_far = min((A_MAX + R_PAD) / xi, self.rst.R_max)
        m = self.rst.n_img(self.R_far)
        self.r_all = np.sqrt(self.rst.r2[:m])
        self.w_all = self.K.theta_S_scaled_closed(self.r_all) * np.exp(-(xi * self.r_all) ** 2) / self.r_all ** 2
        self.m_far = m
        self.GS = self.lap_images(0, m)
        self.GL = Gref - self.GS
        self.Gref = Gref
        self.norm_ref = float(np.linalg.norm(Gref, 2))

    def lap_images(self, lo, hi):
        """Dense Laplacian of the K_S images with sorted index in [lo, hi)."""
        N = self.N
        i, j, d, w = self.rst.i[lo:hi], self.rst.j[lo:hi], self.rst.d[lo:hi], self.w_all[lo:hi]
        flat = i.astype(np.int64) * N + j
        Wsum = np.empty((N, N, 3, 3))
        for a, b in PAIRS6:
            Wsum[:, :, a, b] = Wsum[:, :, b, a] = np.bincount(flat, w * d[:, a] * d[:, b], N * N).reshape(N, N)
        Wsum = Wsum + Wsum.transpose(1, 0, 2, 3)
        return lap_full(Wsum)

    def real_error(self, rc):
        lo = self.rst.n_img(rc)
        if lo >= self.m_far:
            return 0.0
        Gt = self.lap_images(lo, self.m_far)
        return float(linalg.eigvalsh(0.5 * (Gt + Gt.T), subset_by_index=[3 * self.N - 1, 3 * self.N - 1])[0])

    def fourier_W(self, kc, mv=None):
        """Mesh-free truncated lattice Fourier pair matrix (1/V) sum_{|k| <= kc} e^{ik.(q_i - q_j)} Khat_L(k)."""
        if mv is None:
            mv, _, _ = mesh_modes(kc, self.L)
        kv = 2 * np.pi / self.L * mv
        kn = np.linalg.norm(kv, axis=1)
        ku, inv = np.unique(kn, return_inverse=True)
        A, B = self.K.AB_scaled(ku)
        dmp = np.exp(-ku ** 2 / (4 * self.xi ** 2))
        A, B = (A * dmp)[inv], (B * dmp)[inv]
        E = np.exp(-1j * self.x @ kv.T)
        W = np.empty((self.N, self.N, 3, 3))
        V = self.L ** 3
        for a, b in PAIRS6:
            g = A * (a == b) + B * kv[:, a] * kv[:, b]
            Z = ((E.conj() * g) @ E.T).real / V
            W[:, :, a, b] = W[:, :, b, a] = Z
        return W

    def fourier_error(self, kc):
        GF = lap_full(self.fourier_W(kc))
        return float(np.linalg.norm(GF - self.GL, 2)), GF

    def mesh_error(self, kc, M, p, GF=None):
        if GF is None:
            GF = lap_full(self.fourier_W(kc))
        mesh = mesh_for(self.L, self.K, kc, M, p)
        W, imag = mesh.pair_matrix(self.x)
        return float(np.linalg.norm(lap_full(W) - GF, 2)), mesh.n_excluded

    def measured(self, rc, kc, M, p):
        eR = self.real_error(rc)
        eF, GF = self.fourier_error(kc)
        eM, n_excl = self.mesh_error(kc, M, p, GF)
        return dict(meas_real_abs=eR, meas_fourier_abs=eF, meas_mesh_abs=eM, meas_real_rel=eR / self.norm_ref,
                    meas_fourier_rel=eF / self.norm_ref, meas_mesh_rel=eM / self.norm_ref, n_excluded=n_excl,
                    meas_tail_beyond_Rfar_bound=2 * self.rst.analytic_tail(self.xi, self.R_far))

    # -------------------------------------------------------------------------- empirical diagnostic search
    def empirical(self, eps, p, rc_hi, kc_hi, split=SPLIT, M_cap=200):
        tgt = {k: split[k] * eps * self.norm_ref for k in ("S", "F", "mesh")}
        # real space: bisection on rc (the discarded Laplacian is PSD and shrinks monotonically with rc)
        lo, hi = 0.4 * rc_hi, rc_hi
        if self.real_error(hi) > tgt["S"]:
            hi = A_MAX / self.xi
        while hi - lo > 1e-4 * hi:
            mid = 0.5 * (lo + hi)
            if self.real_error(mid) <= tgt["S"]:
                hi = mid
            else:
                lo = mid
        rc = hi
        # Fourier: smallest lattice shell, descending from kc_hi
        mv, n2, _ = mesh_modes(kc_hi, self.L)
        shells = np.unique(n2)[::-1]
        kc_ok = 2 * np.pi / self.L * np.sqrt(shells[0])
        eF0, _ = self.fourier_error(kc_ok)
        if eF0 > tgt["F"]:
            raise RuntimeError("certified kc does not pass the measured Fourier budget")
        for n in shells[1:]:
            kc = 2 * np.pi / self.L * np.sqrt(n)
            eF, _ = self.fourier_error(kc)
            if eF > tgt["F"]:
                break
            kc_ok = kc
        kc = kc_ok
        eF, GF = self.fourier_error(kc)
        # mesh: smallest smooth M (ascending) that passes, confirmed by the next smooth M passing too
        mc = int(np.floor(kc * self.L / (2 * np.pi) + 1e-12))
        cands = [M for M in smooth_numbers(2 * mc + 1, M_cap) if kc * (self.L / M) / np.pi < 1]
        M_ok, trace = None, []
        for t, M in enumerate(cands):
            eM, _ = self.mesh_error(kc, M, p, GF)
            trace.append((M, eM))
            if eM <= tgt["mesh"]:
                nxt = cands[t + 1] if t + 1 < len(cands) else None
                if nxt is not None:
                    eN, _ = self.mesh_error(kc, nxt, p, GF)
                    trace.append((nxt, eN))
                    if eN > tgt["mesh"]:
                        continue
                M_ok = M
                break
        return dict(xi=self.xi, rc_emp=rc, kc_emp=kc, M_emp=M_ok, a_emp=self.xi * rc, b_emp=kc / (2 * self.xi),
                    meas_real_rel=self.real_error(rc) / self.norm_ref, meas_fourier_rel=eF / self.norm_ref,
                    meas_mesh_rel=(dict(trace)[M_ok] / self.norm_ref) if M_ok else None, mesh_trace=trace)


def validate_parameter_set(name, x, L, Gref, ref_sum, Q, P, xi, rc, kc, M, p, basis=False):
    """E_op and structure of one parameter set; optional basis-vector assembly check."""
    N = len(x)
    op = CachedGammaH(x, L, C.kernel(xi), rc, mesh_for(L, C.kernel(xi), kc, M, p))
    Gh = dense_independent(op)
    rec, act = operator_metrics(Gh, Gref, ref_sum, Q, P, N)
    rng = np.random.default_rng([20261017, 99])
    Xs = [rng.standard_normal((N, 3)) for _ in range(3)]
    rec["cached_vs_dense_max"] = max(float(np.linalg.norm(op.apply(X).ravel() - Gh @ X.ravel())
                                           / np.linalg.norm(Gh @ X.ravel())) for X in Xs)
    if basis:
        Gb = op.dense_by_basis()
        rec["basis_vs_independent"] = float(np.linalg.norm(Gb - Gh, 2) / np.linalg.norm(Gh, 2))
        rec["E_op_basis"] = float(np.linalg.norm(Gb - Gref, 2) / ref_sum["norm2"])
    rec.update(n_images=op.n_images, n_excluded=op.mesh.n_excluded, n_modes=op.mesh.n_modes)
    return rec, act
