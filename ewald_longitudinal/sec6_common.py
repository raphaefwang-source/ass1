"""
Shared numerical pieces of the Section 6 experiments, parameterised only through sec6_config.json.

- Box(N, set_name): periodic cube of edge L = (N/rho)^(1/3) with the PPPM mesh/influence function built once;
  Box.gamma(x) returns the literal Gamma_h operator (real-space pair sum + spread/FFT/influence/IFFT/gather + degree
  term) of test_lanczos_fdt.GammaH for positions x.
- f_noise(dt), f_damp(dt): the O-step matrix functions, noise via expm1.
- lanczos / tridiag / proj / DenseRef are the existing, unchanged implementations of test_lanczos_fdt.
- krylov_apply_all(L, funs): every rank s = 1..S of one Lanczos recursion for several functions, with the Ritz
  extrema and any negative Ritz values recorded (never clipped).
"""
import numpy as np

import sec6_config
from test_lanczos_fdt import GammaH, lanczos, tridiag, proj, DenseRef, real_pairs  # noqa: F401
from test_parameter_selection import ewald
from test_equal_accuracy_xi import make_mesh

_M = sec6_config.model()
KAPPA, BETA, MASS, DENSITY = _M["kappa"], _M["beta"], _M["mass"], _M["density"]


class Box:
    def __init__(self, N, set_name=None, L=None):
        sp = sec6_config.spatial(set_name)
        self.set_name, self.params = sp["name"], sp
        self.N = N
        self.L = float(L) if L is not None else sec6_config.box_length(N)
        self.K = ewald(KAPPA, sp["xi"])
        self.rc, self.kc = sp["s"] / sp["xi"], 2 * sp["xi"] * sp["s"]
        self.mesh = make_mesh(self.L, self.K, self.kc, sp["eta"], sp["p"])

    def describe(self):
        m = self.mesh
        return dict(N=self.N, L=self.L, density=self.N / self.L ** 3, set=self.set_name, xi=self.params["xi"],
                    s=self.params["s"], eta_N=self.params["eta"], p=self.params["p"], rc=self.rc, kc=self.kc, M=m.M,
                    h=m.h, eta_actual=m.eta_actual, n_modes=m.n_modes, n_excluded=m.n_excluded)

    def gamma(self, x):
        op = GammaH.__new__(GammaH)
        op.x, op.L, op.N, op.rc, op.kc = x, self.L, len(x), self.rc, self.kc
        op.i, op.j, op.d = real_pairs(x, self.L, self.rc)
        r = np.linalg.norm(op.d, axis=1)
        op.w = self.K.theta_S_scaled_closed(r) * np.exp(-(self.K.xi * r) ** 2) / r ** 2
        op.mesh = self.mesh
        op.idx, op.wt = self.mesh.flat(x)
        op.D = self.mesh.degree(op.idx, op.wt)
        op.n_actions, op.t_actions = 0, 0.0
        return op


def f_noise(dt):
    return lambda lam: np.sqrt(MASS / BETA * (-np.expm1(-2.0 * dt * lam / MASS)))


def f_damp(dt):
    return lambda lam: np.exp(-dt * lam / MASS)


def krylov_apply_all(Lz, funs, ranks=None):
    """For one Lanczos recursion Lz (from test_lanczos_fdt.lanczos), return for each requested rank s the projected
    results nz * Q_s^T f(T_s) e1 of every function in funs (dict name -> callable), plus Ritz extrema.
    Negative Ritz values are recorded; the functions are still evaluated (f_noise gives NaN for theta < 0 and the
    NaN is kept, so a negative Ritz value can never pass silently)."""
    S, Q, nz = Lz["s"], Lz["Q"], Lz["nz"]
    ranks = [s for s in (ranks or range(1, S + 1)) if s <= S]
    out = {name: {} for name in funs}
    ritz = {}
    for s in ranks:
        th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], s))
        ritz[s] = dict(min=float(th[0]), max=float(th[-1]), n_negative=int((th < 0).sum()),
                       negative=[float(t) for t in th[th < 0]])
        for name, fun in funs.items():
            with np.errstate(invalid="ignore"):
                fv = fun(th)
            c = U @ (fv * U[0])
            out[name][s] = proj(nz * (Q[:s].T @ c))
    return out, ritz


def uniform_positions(N, L, rng):
    return rng.uniform(0.0, L, size=(N, 3))
