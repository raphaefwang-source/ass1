#!/usr/bin/env python3
"""
Static comparison of the fast PPPM friction operator Gamma_h with dense full-periodic references, for the two
radial laws A and B of the toy application (radial_kernels.py), on small fixed configurations.

Operators (all pure longitudinal, K = g(r) rhat rhat^T, same gamma, kappa, r_ref, box and amplitude):
    Gamma_h     project PPPM: real-space K_S sum (|r| <= rc) + B-spline PPPM K_L (|k| <= kc, k = 0 included),
                literal GammaH.matvec and its independent dense assembly GammaH.dense()
    Gamma_ref1  v2 reference (toy_models/prl_toy_models.LatticeFriction): real-space lattice sum of the
                unsplit kernel over all images (Chebyshev far field), k = 0 retained, no Ewald split
    Gamma_ref2  project-style direct image sum of the unsplit kernel (verify_periodic_graph_psd.periodic_kernel)
Checks per configuration and parameter set:
    relative spectral / Frobenius / action error of Gamma_h; literal action vs dense assembly; symmetry;
    translation (total-momentum) null space; restricted smallest eigenvalue (PSD) with the Weyl certificate;
    exact split Gamma_S(xi) + Gamma_L(xi) = Gamma_ref for several xi (split implementation);
    error budget eps_S (real cutoff rc), eps_F (Fourier cutoff kc), eps_alias + eps_transfer (mesh),
    reported separately for each xi so that a xi dependence is attributed to its actual source;
    the k = 0 term: present in the mesh, its size, and what removing it would do (diagnostic only);
    friction and noise from the SAME Gamma_h: dense R = exp(-dt Gamma_h), S = f_dt(Gamma_h) satisfy the frozen-q
    FDT identity, and production-rank Lanczos (noise 40, damping 16) reproduces the dense actions.
Run:  python3 test_radial_kernel_static.py      (about 10 min, single thread)
Output: radial_kernel_static_results/ (summary.json, REPORT.md, figures)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import prl_toy_models as v2  # noqa: E402
import radial_kernels as rk  # noqa: E402
from verify_periodic_graph_psd import periodic_kernel, laplacian  # noqa: E402
from test_parameter_selection import Mesh, lap_full, lattice_modes, khat_AB, fourier_pairs  # noqa: E402
from test_lanczos_fdt import DenseRef, proj  # noqa: E402
from test_true_dynamics import lanczos_apply  # noqa: E402

OUT = HERE / "radial_kernel_static_results"
CKPT = HERE / "toy_models" / "v2_results" / "restart_checkpoints"
TOY = dict(gamma=0.5, kappa=0.7, r_ref=1.3)
PARAM_SETS = {"production": dict(xi=0.7, s=4.10, eta=0.7, p=7),     # project production set (tol 1e-7 at kappa=1)
              "tight": dict(xi=0.7, s=4.628, eta=0.6, p=8)}         # project tight set (tol 1e-9 at kappa=1)
XI_SCAN = [0.5, 0.6, 0.7, 0.85, 1.0]
REF_RCUT = {"A": 50.0, "B": 65.0}         # direct-image reference radius (tail < 1e-15 relative at kappa = 0.7)
DT, MASS, KT = 0.005, 1.0, 0.7
COL = {"A": "#2563eb", "B": "#dc7627"}    # A/B colours of the toy figures


class Case:
    def __init__(self, name, x, L, law, gamma, kappa, r_ref):
        self.name, self.x, self.L, self.law = name, np.asarray(x, float) % L, float(L), law
        self.fric = dict(kernel=law, gamma=gamma, kappa=kappa, r_ref=r_ref)
        self.N = len(self.x)
        self.pi, self.pj = np.triu_indices(self.N, 1)
        d = self.x[self.pi] - self.x[self.pj]
        self.d = d - L * np.round(d / L)
        self.amp = rk.amplitude(law, gamma, kappa, r_ref)
        self.kappa = kappa
        t0 = time.perf_counter()
        self.lat = v2.LatticeFriction(L, **self.fric)
        self.G1 = self.lat.matrix(self.x)
        prof = rk.RadialEwald(law, kappa, 1.0, self.amp)
        self.G2 = laplacian(periodic_kernel(prof, self.d, "full", L, REF_RCUT[law]), self.pi, self.pj, self.N)
        self.ref_agree = float(np.abs(self.G1 - self.G2).max() / np.abs(self.G2).max())
        self.G = self.G2
        self.norm2 = float(np.linalg.norm(self.G, 2))
        T = np.zeros((3 * self.N, 3))
        for c in range(3):
            T[c::3, c] = 1 / np.sqrt(self.N)
        self.T, self.Q = T, null_space(T.T)
        self.lam_star = self.restricted_min(self.G)
        self.t_ref = time.perf_counter() - t0

    def restricted_min(self, G):
        return float(np.linalg.eigvalsh(self.Q.T @ (0.5 * (G + G.T)) @ self.Q)[0])

    def lap(self, Kp):
        return laplacian(Kp, self.pi, self.pj, self.N)


def evaluate(case, xi, s, eta, p, rng, budget=True, dynamics=False):
    K = rk.RadialEwald(case.law, case.kappa, xi, case.amp)
    fast = rk.FastFriction(case.L, K, s, eta, p)
    op = fast.gamma_h(case.x)
    Gh = op.dense()
    nG = case.norm2
    D = Gh - case.G
    R = dict(case=case.name, N=case.N, L=case.L, **fast.record())
    R["err_2"] = float(np.linalg.norm(D, 2)) / nG
    R["err_F"] = float(np.linalg.norm(D) / np.linalg.norm(case.G))
    X = rng.normal(size=(50, 3 * case.N))
    act = np.linalg.norm(X @ D.T, axis=1) / np.linalg.norm(X @ case.G.T, axis=1)
    R["action_err_median"], R["action_err_max"] = float(np.median(act)), float(act.max())
    lit = np.array([op.matvec(v) for v in X[:5]])
    R["literal_vs_dense"] = float(np.linalg.norm(lit - X[:5] @ Gh.T) / np.linalg.norm(X[:5] @ Gh.T))
    nGh = float(np.linalg.norm(Gh, 2))
    R["sym_resid"] = float(np.linalg.norm(Gh - Gh.T) / np.linalg.norm(Gh))
    R["trans_resid"] = float(max(np.linalg.norm(Gh @ case.T[:, c]) for c in range(3)) / nGh)
    R["trans_resid_literal"] = float(max(np.linalg.norm(op.matvec(case.T[:, c])) for c in range(3)) / nGh)
    R["lam_star_ref"], R["lam_h"] = case.lam_star, case.restricted_min(Gh)
    R["lam_max_h"] = float(np.linalg.eigvalsh(0.5 * (Gh + Gh.T))[-1])
    R["certified_psd"] = bool(R["err_2"] * nG < case.lam_star)
    R["psd_observed"] = bool(R["lam_h"] >= -1e-12 * nGh)
    if budget:
        prof = rk.RadialEwald(case.law, case.kappa, xi, case.amp)
        GS = case.lap(periodic_kernel(prof, case.d, "S", case.L, REF_RCUT[case.law]))
        GL = case.lap(periodic_kernel(prof, case.d, "L", case.L, REF_RCUT[case.law]))
        GS_rc = case.lap(periodic_kernel(prof, case.d, "S", case.L, fast.rc))
        mv, kv, _ = lattice_modes(case.L, fast.kc)
        A, B = khat_AB(K, np.linalg.norm(kv, axis=1))
        GL_kc = case.lap(fourier_pairs(case.d, kv, A, B, case.L ** 3))
        W, _ = fast.mesh.pair_matrix(case.x)
        GL_mesh = lap_full(W)
        GAL = case.lap(fast.mesh.alias_TI(case.d))
        nrm = lambda M_: float(np.linalg.norm(M_, 2)) / nG
        R["split_exact_err"] = float(np.abs(GS + GL - case.G).max() / np.abs(case.G).max())
        R["eps_S"], R["eps_F"] = nrm(GS_rc - GS), nrm(GL_kc - GL)
        R["eps_alias"], R["eps_transfer"] = nrm(GAL), nrm(GL_mesh - GL_kc - GAL)
        R["eps_mesh"] = nrm(GL_mesh - GL_kc)
        R["real_space_matches_GammaH"] = float(np.abs((Gh - GL_mesh) - GS_rc).max() / np.abs(GS_rc).max())
        # k = 0 term of the long-range part: (1/V) Khat_L(0) per pair (isotropic); diagnostic of its size only
        A0 = float(K.AB_scaled(np.array([0.0]))[0][0])
        R["k0_coefficient_per_pair"] = A0 / case.L ** 3
        U0 = A0 / case.L ** 3 * np.kron(case.N * np.eye(case.N) - np.ones((case.N, case.N)), np.eye(3))
        R["k0_share_of_Gamma"] = nrm(U0)
        R["lam_min_if_k0_removed"] = case.restricted_min(Gh - U0)
    if dynamics:
        R.update(fdt_checks(case, op, Gh, rng))
    return R


def f_noise(dt, kT=None, m=None):
    """f_dt(l) = sqrt(kT m [1 - exp(-2 dt l/m)]). NOTE: test_true_dynamics.f_noise hard-codes beta = 1;
    the toy runs at kT = 0.7, so the temperature must be passed explicitly."""
    kT, m = KT if kT is None else kT, MASS if m is None else m
    return lambda l: np.sqrt(kT * m * -np.expm1(-2 * dt * l / m))


def f_damp(dt, m=None):
    m = MASS if m is None else m
    return lambda l: np.exp(-dt * l / m)


def fdt_checks(case, op, Gh, rng):
    """Friction and noise both come from this one Gamma_h; dense frozen-q FDT and Lanczos production ranks."""
    N = case.N
    ref = DenseRef(Gh, N)
    lam, U = ref.lam, ref.U
    Rm = U @ np.diag(np.exp(-DT * lam / MASS)) @ U.T
    Sm = U @ np.diag(np.sqrt(MASS * KT * -np.expm1(-2 * DT * lam / MASS))) @ U.T
    Pi = np.kron(np.eye(N) - np.ones((N, N)) / N, np.eye(3))
    C = MASS * KT * Pi
    out = dict(dt=DT, dt_lam_max=float(DT * lam.max() / MASS),
               fdt_identity=float(np.abs(Rm @ C @ Rm.T + Sm @ Sm.T - C).max() / (MASS * KT)))
    # the same Gamma_h ingredients against the reference operator
    refG = DenseRef(case.G, N)
    Sref = refG.U @ np.diag(np.sqrt(MASS * KT * -np.expm1(-2 * DT * refG.lam / MASS))) @ refG.U.T
    Rref = refG.U @ np.diag(np.exp(-DT * refG.lam / MASS)) @ refG.U.T
    out["S_h_vs_S_ref"] = float(np.linalg.norm(Sm - Sref, 2) / np.linalg.norm(Sref, 2))
    out["R_h_vs_R_ref"] = float(np.linalg.norm(Rm - Rref, 2))
    errs_n, errs_d = [], []
    for _ in range(4):
        z = proj(rng.normal(size=3 * N))
        pp = proj(rng.normal(size=3 * N) * np.sqrt(MASS * KT))
        nl, _ = lanczos_apply(op, z, 40, f_noise(DT))
        dl, _ = lanczos_apply(op, pp, 16, f_damp(DT))
        errs_n.append(np.linalg.norm(nl - Sm @ z) / np.linalg.norm(Sm @ z))
        errs_d.append(np.linalg.norm(dl - Rm @ pp) / np.linalg.norm(Rm @ pp))
    out["lanczos_noise40_rel_err_max"] = float(max(errs_n))
    out["lanczos_damp16_rel_err_max"] = float(max(errs_d))
    return out


def load_cases():
    cases = []
    rng = np.random.default_rng(20261007)
    for law in rk.LAWS:
        for pot in ("lj", "double_well"):
            f = CKPT / "lj_burn140" if pot == "lj" else CKPT / "double_well_burn140"
            # v2 final states of the full-lattice runs (effective burn-in 140 + production 60)
            with np.load(next((f / "lattice" / f"{pot}_{law}").glob("seed_101/restart.npz"))) as ck:
                cases.append(Case(f"{pot}_{law}_v2final", ck["Q"], float(ck["box"][0]), law, **TOY))
        cases.append(Case(f"uniform_{law}", rng.uniform(0, 5.5, (64, 3)), 5.5, law, **TOY))
    return cases


def figures(rows, scan):
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, law in zip(axes, rk.LAWS):
        sub = [r for r in scan if r["law"] == law]
        names = sorted({r["case"] for r in sub})
        for key, mk, lab in (("err_2", "o", "total ||dGamma||/||Gamma||"), ("eps_S", "s", "real cutoff eps_S"),
                             ("eps_F", "^", "Fourier cutoff eps_F"), ("eps_mesh", "D", "mesh (alias+transfer)"),
                             ("split_exact_err", "x", "exact split S+L vs ref (max entry)")):
            for n_i, name in enumerate(names):
                rr = sorted([r for r in sub if r["case"] == name], key=lambda r: r["xi"])
                ax.semilogy([r["xi"] for r in rr], [max(r[key], 1e-17) for r in rr], marker=mk, ls="-" if n_i == 0 else ":",
                            color=COL[law], alpha=1 - 0.25 * n_i, label=lab if n_i == 0 else None)
        ax.set(xlabel="Ewald splitting parameter xi (s = 4.10, eta = 0.7, p = 7 fixed)", ylabel="relative spectral error",
               title=f"law {law}: error components vs xi (each config one line style)")
        ax.legend(fontsize=7)
    fig.savefig(OUT / "fig1_error_budget_vs_xi.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    labels = [f"{r['case']}\n{r['set']}" for r in rows]
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r["err_2"] for r in rows], 0.4, color=[COL[r["law"]] for r in rows], label="spectral")
    ax.bar(xs + 0.2, [r["action_err_max"] for r in rows], 0.4, color=[COL[r["law"]] for r in rows], alpha=.45, label="max action")
    ax.set_yscale("log")
    ax.set_xticks(xs, labels, rotation=60, fontsize=6)
    ax.set(ylabel="relative error vs full-periodic reference", title="Gamma_h vs dense full-periodic reference (blue A, orange B)")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "fig2_error_by_config.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="first configuration per law only, no xi scan")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    t0 = time.perf_counter()
    cases = load_cases()
    if args.quick:
        cases = [c for c in cases if c.name.startswith("lj_")]
    refs = [dict(case=c.name, law=c.law, N=c.N, L=c.L, amplitude=c.amp, ref_v2_vs_direct=c.ref_agree,
                 norm2=c.norm2, lam_star=c.lam_star, t_ref=c.t_ref) for c in cases]
    for r in refs:
        print(f"[ref] {r['case']}: v2 lattice vs direct image sum {r['ref_v2_vs_direct']:.2e}, "
              f"||Gamma||={r['norm2']:.3f}, lam*={r['lam_star']:.4f}", flush=True)
    rows = []
    for c in cases:
        for name, ps in PARAM_SETS.items():
            R = evaluate(c, **ps, rng=rng, budget=True, dynamics=(name == "production"))
            R["set"] = name
            rows.append(R)
            print(f"[{name}] {c.name}: err2={R['err_2']:.2e} action_max={R['action_err_max']:.2e} "
                  f"eps_S={R['eps_S']:.1e} eps_F={R['eps_F']:.1e} mesh={R['eps_mesh']:.1e} "
                  f"lam_h={R['lam_h']:.4f} (ref {R['lam_star_ref']:.4f}) k0={R['k0_coefficient_per_pair']:.4f}", flush=True)
    scan = []
    if not args.quick:
        for c in cases:
            for xi in XI_SCAN:
                R = evaluate(c, xi=xi, s=4.10, eta=0.7, p=7, rng=rng, budget=True)
                scan.append(R)
                print(f"[xi={xi}] {c.name}: err2={R['err_2']:.2e} S={R['eps_S']:.1e} F={R['eps_F']:.1e} "
                      f"mesh={R['eps_mesh']:.1e} split={R['split_exact_err']:.1e}", flush=True)
        figures(rows, scan)
    summary = dict(toy=TOY, param_sets=PARAM_SETS, xi_scan=XI_SCAN, dt=DT, kT=KT, mass=MASS,
                   references=refs, rows=rows, xi_scan_rows=scan, wall_time=time.perf_counter() - t0)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(f"done in {summary['wall_time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
