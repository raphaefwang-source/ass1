#!/usr/bin/env python3
"""
Cross-time-step coupled trajectories for the dt study (lj075 family).

Integrator (unchanged; toy_dynamics.run): B(h/2) A(h/2) O(h) A(h/2) B(h/2), with the O-step
    p <- p_mean + R_h Pi p + S_h Pi xi,   R_h = exp(-h Gamma/m),  S_h = sqrt(kT m (1 - exp(-2 h Gamma/m))),
Gamma = Gamma_h(q_half) used for BOTH damping and noise.

Coupling across step sizes (nested, exact-OU weights). Level 0 has the smallest step h0 and draws xi ~ N(0, I) from
its own Generator. Level k (step H = 2^k h0) takes its standard-normal input from two consecutive inputs xi1, xi2 of
level k-1 (step h = H/2):
    xi_c = a(Gamma_c) xi1 + b(Gamma_c) xi2,   a = e^{-x}/sqrt(1 + e^{-2x}),  b = 1/sqrt(1 + e^{-2x}),  x = h lambda/m,
with Gamma_c = Gamma_h(q_half) of the level-k trajectory itself (predictable: known before xi1, xi2 enter).
Properties:
  * a^2 + b^2 = 1 and a, b commute, so xi_c | past ~ N(0, I) exactly: every level has exactly the law of an
    uncoupled run with its own step (up to the Lanczos error of a, b, verified separately).
  * With Gamma frozen and no conservative motion, S_H xi_c = R_h S_h xi1 + S_h xi2, i.e. one coarse O-step equals
    two fine O-steps driven by xi1, xi2 exactly (test: lj075_coupled.py --selftest).
  * At lambda -> 0, a = b = 1/sqrt(2): the Brownian-increment sum.
Using the same seed for every level would NOT couple the paths; this construction does.

Cost: level k > 0 needs two extra Lanczos applications (a xi1, b xi2) per step.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import toy_dynamics as td  # noqa: E402
from test_lanczos_fdt import DenseRef, proj  # noqa: E402
from test_true_dynamics import lanczos_apply  # noqa: E402


def coupling_funcs(h, m):
    """a, b for combining two step-h inputs into one step-2h input."""
    def a(lam):
        e = np.exp(-h * lam / m)
        return e / np.sqrt(1 + e * e)

    def b(lam):
        e = np.exp(-h * lam / m)
        return 1 / np.sqrt(1 + e * e)
    return a, b


class Level:
    """One trajectory at step dt, sharing physics with the other levels; holds its own state."""

    def __init__(self, r, dt, q0, p0, rank_couple=None, monitor_every=0, monitor_extra=(8, 4)):
        mdl = r["model"]
        self.dt, self.m, self.kT, self.L = float(dt), mdl["mass"], mdl["kT"], r["L"]
        self.th = td.Thermostat("pppm_lanczos", r["L"], r["law"], mdl["gamma"], mdl["kappa"], mdl["r_ref"], dt,
                                mdl["kT"], mdl["mass"], pppm=r["pppm"], rank_noise=r["rank_noise"],
                                rank_damp=r["rank_damp"], pair_search=r["pair_search"])
        self.fa, self.fb = coupling_funcs(dt / 2, self.m)
        self.rank_c = int(rank_couple or r["rank_noise"])
        self.q, self.p = np.array(q0, float), np.array(p0, float)
        self.F, self.U, self.rmin = td.conservative_force_neighbor(self.q, self.L, "lj")
        self.nstep = 0
        self.stop_rmin = mdl.get("stop_rmin", 0.45)
        self.monitor_every, self.monitor_extra = monitor_every, monitor_extra
        self.monitor_rows = []
        self.lam = [np.inf, -np.inf]

    def couple(self, op, xi1, xi2):
        a1, (l1, h1) = lanczos_apply(op, xi1, self.rank_c, self.fa)
        b2, (l2, h2) = lanczos_apply(op, xi2, self.rank_c, self.fb)
        return a1 + b2, (min(l1, l2), max(h1, h2))

    def step(self, xi=None, pair=None):
        """One step. xi: projected standard-normal (3N,) input (level 0), or pair = (xi1, xi2) from the level below.
        Returns the effective standard-normal input used (for the level above)."""
        dt, m, th = self.dt, self.m, self.th
        self.p = self.p + 0.5 * dt * self.F
        self.q = self.q + 0.5 * dt / m * self.p
        op = th.fast.gamma_h(self.q)                                   # one Gamma_h for damping and noise
        if pair is not None:
            xi, (lo, hi) = self.couple(op, pair[0], pair[1])
            self.lam = [min(self.lam[0], lo), max(self.lam[1], hi)]
        N = len(self.p)
        pm = self.p.mean(axis=0)
        pp = proj(self.p.ravel())
        damp, (dlo, dhi) = lanczos_apply(op, pp, th.rank_damp, th.fd)
        noise, (nlo, nhi) = lanczos_apply(op, xi, th.rank_noise, th.fn)
        self.lam = [min(self.lam[0], dlo, nlo), max(self.lam[1], dhi, nhi)]
        if self.monitor_every and self.nstep % self.monitor_every == 0:
            self.monitor_rows.append(self.monitor(op, pp, xi, damp, noise, pair))
        self.p = pm[None, :] + (proj(damp) + proj(noise)).reshape(N, 3)
        self.q = self.q + 0.5 * dt / m * self.p
        self.F, self.U, self.rmin = td.conservative_force_neighbor(self.q, self.L, "lj")
        self.p = self.p + 0.5 * dt * self.F
        self.nstep += 1
        if not np.all(np.isfinite(self.p)) or self.rmin < self.stop_rmin:
            raise RuntimeError(f"unstable close encounter at dt = {dt} (rmin {self.rmin:.3f}); no clipping applied")
        return xi

    def monitor(self, op, pp, xi, damp, noise, pair):
        """Lanczos error estimates on the ACTUAL inputs of this step: |f_r - f_(r+e)| / |f_(r+e)|."""
        en, ed = self.monitor_extra
        th = self.th
        dref, _ = lanczos_apply(op, pp, th.rank_damp + ed, th.fd)
        nref, _ = lanczos_apply(op, xi, th.rank_noise + en, th.fn)
        row = [self.nstep * self.dt, float(np.linalg.norm(damp - dref) / np.linalg.norm(dref)),
               float(np.linalg.norm(noise - nref) / np.linalg.norm(nref))]
        if pair is not None:
            c_ref = (lanczos_apply(op, pair[0], self.rank_c + en, self.fa)[0]
                     + lanczos_apply(op, pair[1], self.rank_c + en, self.fb)[0])
            c_now, _ = self.couple(op, pair[0], pair[1])
            row.append(float(np.linalg.norm(c_now - c_ref) / np.linalg.norm(c_ref)))
        else:
            row.append(np.nan)
        return row


def selftest(r, q, seed=3):
    """Frozen Gamma, no conservative motion: one coarse O-step with the coupled input equals two fine O-steps.
    Dense reference (exact matrix functions) and Lanczos at the configured ranks."""
    N, L, m, kT = len(q), r["L"], r["model"]["mass"], r["model"]["kT"]
    h = 0.005
    lvl_f = Level(r, h, q, np.zeros_like(q))
    lvl_c = Level(r, 2 * h, q, np.zeros_like(q))
    op = lvl_f.th.fast.gamma_h(q)
    D = DenseRef(op.dense(), N)
    rng = np.random.default_rng(seed)
    p0 = proj(rng.normal(0, np.sqrt(kT * m), 3 * N))
    x1, x2 = proj(rng.standard_normal(3 * N)), proj(rng.standard_normal(3 * N))
    fd = lambda dt: (lambda l: np.exp(-dt * l / m))                                   # noqa: E731
    fn = lambda dt: (lambda l: np.sqrt(kT * m * -np.expm1(-2 * dt * l / m)))           # noqa: E731
    a, b = coupling_funcs(h, m)
    # dense
    fine = D.apply(fd(h), D.apply(fd(h), p0) + D.apply(fn(h), x1)) + D.apply(fn(h), x2)
    xc = D.apply(a, x1) + D.apply(b, x2)
    coarse = D.apply(fd(2 * h), p0) + D.apply(fn(2 * h), xc)
    dense_err = float(np.linalg.norm(fine - coarse) / np.linalg.norm(fine))
    # law: a^2 + b^2 = 1 on the spectrum
    lam = D.lam
    law_err = float(np.max(np.abs(a(lam) ** 2 + b(lam) ** 2 - 1)))
    # Lanczos at configured ranks (production routine)
    xc_l, _ = lvl_c.couple(op, x1, x2)
    co_l = lanczos_apply(op, p0, r["rank_damp"], fd(2 * h))[0] + lanczos_apply(op, xc_l, r["rank_noise"], fn(2 * h))[0]
    lanczos_err = float(np.linalg.norm(fine - co_l) / np.linalg.norm(fine))
    couple_err = float(np.linalg.norm(xc_l - xc) / np.linalg.norm(xc))
    # covariance of xi_c (Monte Carlo over many inputs) at the dense level: E[xc xc^T] = I on Range(Pi)
    return dict(dense_two_fine_vs_one_coarse=dense_err, a2_plus_b2_minus_1=law_err,
                lanczos_two_fine_vs_one_coarse=lanczos_err, lanczos_coupling_input_err=couple_err,
                lam_min=float(lam[0]), lam_max=float(lam[-1]))


def run_chain(r, dts, q0, p0, t_end, seed, save_dt=None, on_save=None, on_step=None, monitor_every=0,
              rank_couple=None, log=None):
    """Integrate all levels (ascending dts, each twice the previous) from the same (q0, p0) to t_end in lock-step.
    on_step(level_index, level) is called after every step of every level and once for the initial state;
    on_save(level_index, t, level) at every multiple of save_dt (a multiple of the largest dt).
    Returns per-level wall times, steps, Ritz ranges and Lanczos monitor rows."""
    K = len(dts)
    for k in range(1, K):
        if abs(dts[k] - 2 * dts[k - 1]) > 1e-15:
            raise ValueError("dts must double")
    H = dts[-1]
    n_macro = int(round(t_end / H))
    save_every_macro = int(round((save_dt or H) / H))
    if abs(save_every_macro * H - (save_dt or H)) > 1e-12 or abs(n_macro * H - t_end) > 1e-9:
        raise ValueError("t_end and save_dt must be multiples of the largest dt")
    levels = [Level(r, dt, q0, p0, rank_couple=rank_couple, monitor_every=monitor_every) for dt in dts]
    rng = np.random.default_rng(seed)
    N = len(q0)
    wall = np.zeros(K)
    for k, lv in enumerate(levels):
        if on_save:
            on_save(k, 0.0, lv)
        if on_step:
            on_step(k, lv)
    t0 = time.perf_counter()
    for M in range(n_macro):
        inputs = None
        for k, lv in enumerate(levels):
            n_sub = 2 ** (K - 1 - k)
            out = []
            tw = time.perf_counter()
            for s in range(n_sub):
                if k == 0:
                    xi = proj(rng.standard_normal(3 * N))
                    out.append(lv.step(xi=xi))
                else:
                    out.append(lv.step(pair=(inputs[2 * s], inputs[2 * s + 1])))
                if on_step:
                    on_step(k, lv)
            wall[k] += time.perf_counter() - tw
            inputs = out
        if on_save and (M + 1) % save_every_macro == 0:
            t = (M + 1) * H
            for k, lv in enumerate(levels):
                on_save(k, t, lv)
        if log and (M + 1) % max(1, n_macro // 20) == 0:
            log(f"  t = {(M + 1) * H:.3f}/{t_end} ({time.perf_counter() - t0:.0f} s)")
    return dict(wall_per_level=wall.tolist(), monitor=[lv.monitor_rows for lv in levels],
                lam=[lv.lam for lv in levels], steps=[lv.nstep for lv in levels])


if __name__ == "__main__":
    import json
    import lj075_common as C
    import toy_configs as tc
    if "--selftest" in sys.argv:
        out = {}
        for law in ("A", "B"):
            r = tc.resolve(sys.argv[sys.argv.index("--config") + 1], law, 256, "lj", dt=0.01, allow_unverified=True)
            rng = np.random.default_rng(5)
            q = C.fcc_positions(256, r["L"], 0.1, rng)
            out[law] = selftest(r, q)
            print(law, out[law], flush=True)
        print(json.dumps(out))
