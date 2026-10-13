#!/usr/bin/env python3
"""
Experiment 1 (manuscript Section 6.1): complete particle-mesh operator action on static configurations.

Model (unchanged): periodic graph Laplacian of the screened pure-longitudinal kernel
    K(r) = exp(-kappa r)/r P(r),  P = r r^T / r^2,  kappa = 1,
    (Gamma X)_i = sum_j sum_n K(x_i - x_j + nL)(X_i - X_j)   (all periodic images, k = 0 retained),
split at s = xi^2 into K = K^S + K^L; Gamma_h = Gamma_S^{rc} (real-space pairs) + Gamma_L^{mesh} (B-spline order p
spread, FFT, influence K^L(k)/|W(k)|^2 on |k| <= kc, IFFT, adjoint gather).

Reference: Gamma_ref is assembled from the UNSPLIT kernel exp(-kappa r)/r P by a direct periodic lattice sum over all
images with |x_i - x_j + nL| <= R (no minimum image, no Ewald split, k = 0 included because nothing is removed). The
image radius R is increased in a ladder and the change is reported against the errors under test.

Stages (results in exp1_results/, dense matrices cached in exp1_results/raw/, git-ignored):
  configs     configurations and their geometry: configs.csv, configs.npz
  reference   image-radius ladder of Gamma_ref and exact Gamma_S, Gamma_L (xi in XI_SWEEP): reference_ladder.csv
  quadrature  accuracy of the numerical kernel coefficients (theta_L, theta_S, A, B) under refined rules
  split       K = K^S + K^L on nonzero pair images and a radial grid; operator-level split identity
  realspace   real-space cutoff sweep (Gamma_S^rc vs exact Gamma_S)
  fourier     Fourier cutoff sweep (exact lattice sum |k| <= kc vs exact Gamma_L)
  mesh        mesh spacing h and B-spline order p with random grid shifts (Gamma_L^mesh vs Gamma_L^kc);
              off-grid alias decomposition D_h + H (alpha_i^* + alpha_j + alpha_i^* alpha_j), literal spread check
  action      production path GammaH.matvec (literal spread-FFT-influence-gather): ||Gamma_h - Gamma_ref||_2, probes,
              symmetry, translation null space, lambda_min on T_perp, error budget, at the accepted xi = 0.7 sets
  all         every stage in order

    python3 exp1_operator_action.py all --workers 4
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.linalg import null_space  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_ewald_longitudinal import EwaldLongitudinal, graded_rule, rho  # noqa: E402
from verify_periodic_graph_psd import Profiles, image_vectors, laplacian  # noqa: E402
from test_parameter_selection import Mesh, lap_full, lattice_modes, khat_AB  # noqa: E402
from test_mesh_error_decomposition import decomposition, literal_W, mesh_at  # noqa: E402
from test_equal_accuracy_xi import next_smooth  # noqa: E402
from test_lanczos_fdt import GammaH  # noqa: E402

OUT = HERE / "exp1_results"
RAW = OUT / "raw"
KAPPA = 1.0
EXP_SEED = 20261013
R_LADDER = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0)
R_REF = R_LADDER[-1]
XI_SWEEP = (0.7, 1.0)                          # component sweeps; 0.7 = production
XI_SPLIT = (0.5, 0.7, 1.0, 1.4, 2.0)
N_A = 30 / 6.0 ** 3                            # number density of case A
# accepted xi = 0.7 parameter sets (equal_accuracy_xi_results/accepted_best.csv, tol 1e-5 / 1e-7 / 1e-9)
PARAM_SETS = {"T5": dict(xi=0.7, s=3.493719027845567, eta=0.8, p=6),
              "T7": dict(xi=0.7, s=4.100151562017953, eta=0.7, p=7),
              "T9": dict(xi=0.7, s=4.627787054036341, eta=0.6, p=8)}
MESH_P = (3, 4, 5, 6, 7, 8)
MESH_SHIFTS = 8
I3 = np.eye(3)


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def git_rev():
    try:
        rev = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
        return dict(commit=rev, dirty=dirty)
    except Exception as exc:                     # noqa: BLE001
        return dict(error=str(exc))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def log_run(stage, extra=None):
    """Append the command, time, commit and environment of a stage to exp1_results/manifest.json."""
    f = OUT / "manifest.json"
    man = json.loads(f.read_text()) if f.exists() else {"runs": []}
    man["runs"].append(dict(stage=stage, time=now(), command=[sys.executable] + sys.argv, git=git_rev(),
                            python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                            host=platform.node(), threads={v: os.environ.get(v) for v in
                                                           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                            "MKL_NUM_THREADS")}, **(extra or {})))
    f.write_text(json.dumps(man, indent=1) + "\n")


# ----------------------------------------------------------------------------
# configurations
# ----------------------------------------------------------------------------
def min_image(d, L):
    return d - L * np.round(d / L)


def pair_geometry(x, L):
    N = len(x)
    pi, pj = np.triu_indices(N, 1)
    d = min_image(x[pi] - x[pj], L)
    return pi, pj, d


def lattice_points(N, L):
    """Simple cubic, bcc or fcc points filling the box with N = n^3, 2 n^3 or 4 n^3."""
    for kind, basis in (("sc", [[0, 0, 0]]), ("bcc", [[0, 0, 0], [.5, .5, .5]]),
                        ("fcc", [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])):
        n = round((N / len(basis)) ** (1 / 3))
        if len(basis) * n ** 3 == N:
            a = L / n
            g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
            pts = ((g[:, None, :] + np.asarray(basis)[None]) * a).reshape(-1, 3)
            dnn = {"sc": a, "bcc": a * np.sqrt(3) / 2, "fcc": a / np.sqrt(2)}[kind]
            return pts, kind, a, dnn
    raise ValueError(f"no cubic lattice with N = {N}")


def make_config(spec):
    """Positions for one configuration spec; every random draw uses default_rng([EXP_SEED, family code, N, seed])."""
    fam, N, L, seed = spec["family"], spec["N"], spec["L"], spec["seed"]
    gen = {}
    if fam in ("caseA", "caseB"):                   # the original draws (verify_periodic_graph_psd / Config)
        x = np.random.default_rng(seed).uniform(0, L, size=(N, 3))
        gen = dict(method="uniform i.i.d. in the box, no minimum distance", rng=f"default_rng({seed})")
    else:
        code = {"uniform": 1, "lattice": 2, "cluster": 3}[fam]
        rng = np.random.default_rng([EXP_SEED, code, N, seed])
        if fam == "uniform":
            x = rng.uniform(0, L, size=(N, 3))
            gen = dict(method="uniform i.i.d. in the box, no minimum distance")
        elif fam == "lattice":
            pts, kind, a, dnn = lattice_points(N, L)
            sig = 0.10 * dnn
            x = (pts + rng.normal(0, sig, size=pts.shape) + rng.uniform(0, L, size=3)) % L
            gen = dict(method=f"{kind} lattice + isotropic Gaussian displacement + random global shift",
                       lattice=kind, cell=a, d_nn=dnn, sigma=sig, sigma_over_dnn=0.10)
        else:
            n_cl, sig_c, dmin = N // 16, 0.5, 0.25
            centres = rng.uniform(0, L, size=(n_cl, 3))
            x = np.empty((N, 3))
            tries = 0
            for i in range(N):
                c = centres[i % n_cl]
                while True:
                    tries += 1
                    y = (c + rng.normal(0, sig_c, 3)) % L
                    if i == 0 or np.linalg.norm(min_image(x[:i] - y, L), axis=1).min() >= dmin:
                        break
                    if tries > 10 ** 6:
                        raise RuntimeError("cluster rejection did not converge")
                x[i] = y
            gen = dict(method="Gaussian clusters around uniform centres, rejection of pairs closer than d_min",
                       n_clusters=n_cl, per_cluster=N // n_cl, sigma_c=sig_c, d_min_imposed=dmin,
                       draws=tries)
        gen["rng"] = f"default_rng([{EXP_SEED}, {code}, {N}, {seed}])"
    pi, pj, d = pair_geometry(x, L)
    r = np.linalg.norm(d, axis=1)
    D = np.full((N, N), np.inf)
    D[pi, pj] = D[pj, pi] = r
    nn = D.min(axis=1)
    geo = dict(n=N / L ** 3, kappaL=KAPPA * L, spacing=(L ** 3 / N) ** (1 / 3), d_min=float(r.min()),
               nn_mean=float(nn.mean()), nn_min=float(nn.min()), pairs_below_0p5=int((r < 0.5).sum()))
    return x, gen, geo


def config_specs():
    specs = [dict(name="A", family="caseA", set="legacy", N=30, L=6.0, seed=7),
             dict(name="B", family="caseB", set="legacy", N=40, L=10.0, seed=13)]
    L128 = (128 / N_A) ** (1 / 3)
    L256 = (256 / N_A) ** (1 / 3)
    for fam in ("uniform", "lattice", "cluster"):
        for s in (1, 2, 3):
            specs.append(dict(name=f"{fam}_N128_s{s}", family=fam, set="main", N=128, L=L128, seed=s))
        specs.append(dict(name=f"{fam}_N256_s1", family=fam, set="check", N=256, L=L256, seed=1))
    for L in (4.0, 6.0, 8.0, 10.0, 12.0, 16.0):                       # fixed N: density changes with L
        for s in (1, 2, 3):
            specs.append(dict(name=f"fixN64_L{L:g}_s{s}", family="uniform", set="box_fixedN", N=64, L=L, seed=s))
    for L in (6.0, 8.0, 10.0, 12.0):                                  # fixed density n_A: N changes with L
        N = int(round(N_A * L ** 3))
        for s in (1, 2, 3):
            specs.append(dict(name=f"fixn_L{L:g}_N{N}_s{s}", family="uniform", set="box_fixedn", N=N, L=L, seed=s))
    return specs


def stage_configs(args):
    rows, arrays = [], {}
    for sp in config_specs():
        x, gen, geo = make_config(sp)
        arrays[sp["name"]] = x
        rows.append(dict(**sp, **geo, generator=json.dumps(gen)))
    write_csv(OUT / "configs.csv", rows)
    np.savez_compressed(OUT / "configs.npz", **arrays)
    log_run("configs", dict(n_configs=len(rows)))
    print(f"{len(rows)} configurations -> {OUT / 'configs.csv'}")


def load_configs(sets=None, names=None):
    rows = read_csv(OUT / "configs.csv")
    X = np.load(OUT / "configs.npz")
    out = []
    for r in rows:
        if sets and r["set"] not in sets:
            continue
        if names and r["name"] not in names:
            continue
        out.append(dict(name=r["name"], family=r["family"], set=r["set"], N=int(r["N"]), L=float(r["L"]),
                        seed=int(r["seed"]), x=X[r["name"]]))
    return out


# ----------------------------------------------------------------------------
# dense reference: periodic lattice sum of the unsplit kernel, image-radius ladder
# ----------------------------------------------------------------------------
def theta_full(r):
    return np.exp(-KAPPA * r) / r


def lattice_ladder(d, L, radii, theta, chunk=None):
    """sum_n theta(|d+nL|)/|d+nL|^2 (d+nL)(d+nL)^T over |d+nL| <= R for every R in radii (ascending).
    Shells (R_{k-1}, R_k] are summed separately and accumulated, so every R uses the same images."""
    radii = np.asarray(radii, float)
    chunk = chunk or max(8, int(1.0e6 / max(len(d), 1)))
    nL = image_vectors(L, radii[-1])
    shells = np.zeros((len(radii), len(d), 3, 3))
    edges = np.concatenate(([0.0], radii))
    for c0 in range(0, len(nL), chunk):
        r = d[:, None, :] + nL[None, c0:c0 + chunk, :]
        rr = np.linalg.norm(r, axis=-1)
        ok = (rr > 0) & (rr <= radii[-1])
        w = np.zeros_like(rr)
        w[ok] = theta(rr[ok]) / rr[ok] ** 2
        b = np.searchsorted(edges, rr, side="left") - 1          # shell index: edges[b] < rr <= edges[b+1]
        for k in range(len(radii)):
            m = ok & (b == k)
            if m.any():
                wk = np.where(m, w, 0.0)
                shells[k] += np.einsum("mc,mci,mcj->mij", wk, r, r)
    return np.cumsum(shells, axis=0)


def specnorm_sym(M):
    return float(np.abs(np.linalg.eigvalsh(0.5 * (M + M.T))).max())


def specnorm(M):
    return float(np.linalg.norm(M, 2))


def ref_path(name):
    return RAW / f"ref_{name}.npz"


def _reference_one(cfg):
    t0 = time.perf_counter()
    x, L, N = cfg["x"], cfg["L"], cfg["N"]
    pi, pj, d = pair_geometry(x, L)
    lad = lattice_ladder(d, L, R_LADDER, theta_full)
    Gs = [laplacian(Kp, pi, pj, N) for Kp in lad]
    G = Gs[-1]
    nG = specnorm_sym(G)
    rows = []
    for R, Gk, Kp in zip(R_LADDER, Gs, lad):
        rows.append(dict(name=cfg["name"], set=cfg["set"], family=cfg["family"], N=N, L=L, R=R,
                         spec_rel_vs_Rmax=specnorm_sym(Gk - G) / nG,
                         max_rel_vs_Rmax=float(np.abs(Gk - G).max() / np.abs(G).max()),
                         tail_bound_exp=float(np.exp(-KAPPA * R))))
    for a, b in zip(rows[:-1], rows[1:]):
        a["spec_rel_to_next"] = specnorm_sym(Gs[R_LADDER.index(a["R"])] - Gs[R_LADDER.index(b["R"])]) / nG
    store = dict(G=G, pi=pi, pj=pj, norm2=nG)
    for xi in XI_SWEEP:
        prof = Profiles(EwaldLongitudinal(KAPPA, xi))
        KS = lattice_ladder(d, L, (R_REF,), lambda r: prof.theta(r, "S"))[0]
        KL = lattice_ladder(d, L, (R_REF,), lambda r: prof.theta(r, "L"))[0]
        store[f"KS_{xi:g}"], store[f"KL_{xi:g}"] = KS, KL
        rows[-1][f"split_images_rel_xi{xi:g}"] = float(np.abs(KS + KL - lad[-1]).max() / np.abs(lad[-1]).max())
    T = np.zeros((3 * N, 3))
    for c in range(3):
        T[c::3, c] = 1 / np.sqrt(N)
    Q = null_space(T.T)
    lam = np.linalg.eigvalsh(Q.T @ (0.5 * (G + G.T)) @ Q)
    store["lam_star"] = lam[0]
    rows[-1].update(norm2=nG, lam_star=float(lam[0]), lam_max_Tperp=float(lam[-1]),
                    trans_resid=float(max(np.linalg.norm(G @ T[:, c]) for c in range(3)) / nG),
                    sym_resid=float(np.linalg.norm(G - G.T) / np.linalg.norm(G)), t_s=time.perf_counter() - t0)
    np.savez(ref_path(cfg["name"]), **store)
    return rows


def stage_reference(args):
    RAW.mkdir(parents=True, exist_ok=True)
    cfgs = [c for c in load_configs() if args.force or not ref_path(c["name"]).exists()]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(_reference_one, cfgs):
            rows += res
            lad = " ".join("%.1e" % r["spec_rel_vs_Rmax"] for r in res[:-1])
            print(f"  reference {res[0]['name']}: ladder {lad}, {res[-1]['t_s']:.0f} s", flush=True)
    old = read_csv(OUT / "reference_ladder.csv") if (OUT / "reference_ladder.csv").exists() and not args.force else []
    done = {r["name"] for r in rows}
    write_csv(OUT / "reference_ladder.csv", [r for r in old if r["name"] not in done] + rows)
    log_run("reference", dict(radii=R_LADDER, n=len(cfgs)))


def load_ref(name):
    with np.load(ref_path(name)) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------
# numerical kernel coefficients: quadrature accuracy
# ----------------------------------------------------------------------------
class RefinedEwald(EwaldLongitudinal):
    """Same s-integrals with every Gauss-Legendre rule refined (twice the panels, 30 nodes per panel)."""

    def __init__(self, kappa, xi):
        super().__init__(kappa, xi)
        xi2 = self.xi ** 2
        self.s_low = graded_rule(xi2, xi2 * 2.0 ** -60, 2 * 97, n=30)
        self.rho_low = rho(self.s_low[0], self.kap)
        self.t_rule = graded_rule(80.0, 1e-14, 2 * 56, n=30)
        self.tau_rule = graded_rule(100.0, 1e-12, 2 * 46, n=30)


def stage_quadrature(args):
    rows = []
    r = np.geomspace(1e-3, 20.0, 300)
    for xi in XI_SPLIT:
        K, Kr = EwaldLongitudinal(KAPPA, xi), RefinedEwald(KAPPA, xi)
        kmax = 2 * xi * 7.0
        k = np.concatenate(([0.0], np.geomspace(1e-3, kmax, 300)))
        A, B = K.AB_scaled(k)
        Ar, Br = Kr.AB_scaled(k)
        lam = np.abs(A + B * k ** 2)
        scale = np.maximum(np.abs(A), lam)                         # ||Khat_L||_2 (scaled)
        thL, thLr = K.theta_L(r), Kr.theta_L(r)
        thSq, thSc = K.theta_S_scaled(r), K.theta_S_scaled_closed(r)
        rows.append(dict(xi=xi, k_max=kmax,
                         AB_rel_vs_refined=float(max(np.abs(A - Ar).max(), (np.abs(B - Br) * k ** 2).max())
                                                 / scale.max()),
                         AB_rel_pointwise_max=float(np.max(np.maximum(np.abs(A - Ar), np.abs(B - Br) * k ** 2)
                                                           / scale)),
                         thetaL_rel_vs_refined=float(np.max(np.abs(thL - thLr) / theta_full(r))),
                         thetaS_quad_vs_closed=float(np.max(np.abs(thSq - thSc) / thSc))))
        print(f"  xi {xi}: {rows[-1]}", flush=True)
    write_csv(OUT / "quadrature.csv", rows)
    log_run("quadrature")


# ----------------------------------------------------------------------------
# split identity
# ----------------------------------------------------------------------------
def pair_images(x, L, rmax):
    """All pair images i < j, n with 0 < |x_i - x_j + nL| <= rmax: (i, j, displacement)."""
    pi, pj, d = pair_geometry(x, L)
    nL = image_vectors(L, rmax)
    dd = d[:, None, :] + nL[None]
    rr = np.linalg.norm(dd, axis=-1)
    keep = (rr <= rmax) & (rr > 0)
    I = np.broadcast_to(pi[:, None], keep.shape)[keep]
    J = np.broadcast_to(pj[:, None], keep.shape)[keep]
    return I, J, dd[keep], rr[keep]


def fourier_ladder_pairs(d, L, kc_list, A_of_k, chunk=None):
    """Exact truncated lattice sums (1/V) sum_{|k| <= kc} Khat(k) cos(k.d) for ascending kc, k = 0 included."""
    kc_list = np.asarray(kc_list, float)
    chunk = chunk or max(64, int(4.0e6 / max(len(d), 1)))
    mv, kv, _ = lattice_modes(L, kc_list[-1])
    kn = np.linalg.norm(kv, axis=1)
    order = np.argsort(kn, kind="stable")
    kv, kn = kv[order], kn[order]
    A, B = A_of_k(kn)
    V = L ** 3
    out = np.zeros((len(kc_list), len(d), 9))
    acc = np.zeros((len(d), 9))
    nmodes = []
    pos = 0
    for t, kc in enumerate(kc_list):
        end = int(np.searchsorted(kn, kc * (1 + 1e-12), side="right"))
        for c0 in range(pos, end, chunk):
            c1 = min(c0 + chunk, end)
            k = kv[c0:c1]
            C = np.cos(d @ k.T)
            kk = (k[:, :, None] * k[:, None, :]).reshape(-1, 9)
            acc += np.outer(C @ A[c0:c1], I3.ravel()) + (C * B[c0:c1]) @ kk
        pos = end
        out[t] = acc / V
        nmodes.append(end)
    return out.reshape(len(kc_list), len(d), 3, 3), nmodes


def stage_split(args):
    cfgs = load_configs(sets=("legacy", "main"))
    rows_pair, rows_op = [], []
    rgrid = np.geomspace(1e-3, 20.0, 400)
    allr = np.concatenate([pair_images(c["x"], c["L"], 12.0)[3] for c in cfgs])
    for xi in XI_SPLIT:
        K = EwaldLongitudinal(KAPPA, xi)
        for label, r in (("pair images (all configs, r <= 12)", allr), ("radial grid 1e-3..20", rgrid)):
            thS = K.theta_S_scaled_closed(r) * np.exp(-(xi * r) ** 2)
            thL = K.theta_L(r)                                       # s-integral, no subtraction
            err = np.abs(thS + thL - theta_full(r)) / theta_full(r)
            rows_pair.append(dict(xi=xi, points=label, n=len(r), r_min=float(r.min()), r_max=float(r.max()),
                                  max_rel=float(err.max()), median_rel=float(np.median(err)),
                                  r_at_max=float(r[np.argmax(err)])))
            print(f"  split xi {xi} {label}: max {err.max():.2e}", flush=True)
    # operator level: Gamma_S (images, closed theta_S) + Gamma_L (exact lattice Fourier sum, kc = 2 xi 6.5)
    for c in cfgs:
        ref = load_ref(c["name"])
        pi, pj, d = pair_geometry(c["x"], c["L"])
        for xi in (0.5, 0.7, 1.0, 1.4):
            K = EwaldLongitudinal(KAPPA, xi)
            prof = Profiles(K)
            KS = lattice_ladder(d, c["L"], (R_REF,), lambda r: prof.theta(r, "S"))[0]
            KF, nm = fourier_ladder_pairs(d, c["L"], [2 * xi * 6.5], lambda kn: khat_AB(K, kn))
            G = laplacian(KS + KF[0], pi, pj, c["N"])
            rows_op.append(dict(name=c["name"], xi=xi, kc=2 * xi * 6.5, n_modes=nm[0],
                                spec_rel=specnorm_sym(G - ref["G"]) / float(ref["norm2"]),
                                max_rel=float(np.abs(G - ref["G"]).max() / np.abs(ref["G"]).max())))
            print(f"  operator split {c['name']} xi {xi}: {rows_op[-1]['spec_rel']:.2e}", flush=True)
    write_csv(OUT / "split_pairs.csv", rows_pair)
    write_csv(OUT / "split_operator.csv", rows_op)
    log_run("split")


# ----------------------------------------------------------------------------
# real-space and Fourier cutoff sweeps
# ----------------------------------------------------------------------------
X_RC = np.round(np.arange(0.5, 6.51, 0.25), 4)          # xi rc
Y_KC = np.round(np.arange(0.25, 6.51, 0.25), 4)         # kc / (2 xi)


def _cutoff_one(cfg):
    ref = load_ref(cfg["name"])
    x, L, N = cfg["x"], cfg["L"], cfg["N"]
    pi, pj, d = pair_geometry(x, L)
    nG = float(ref["norm2"])
    rows = []
    for xi in XI_SWEEP:
        K = EwaldLongitudinal(KAPPA, xi)
        GS = laplacian(ref[f"KS_{xi:g}"], pi, pj, N)
        GL = laplacian(ref[f"KL_{xi:g}"], pi, pj, N)
        # real space: pair images sorted by distance, accumulated per pair
        I, J, dd, rr = pair_images(x, L, X_RC[-1] / xi)
        pidx = I * N - I * (I + 1) // 2 + (J - I - 1)            # index of (i, j) in triu order
        w = K.theta_S_scaled_closed(rr) * np.exp(-(xi * rr) ** 2) / rr ** 2
        blk = w[:, None, None] * dd[:, :, None] * dd[:, None, :]
        order = np.argsort(rr, kind="stable")
        rr_s, pidx_s, blk_s = rr[order], pidx[order], blk[order]
        acc = np.zeros((len(d), 3, 3))
        pos, prev_n = 0, -1
        for xr in X_RC:
            end = int(np.searchsorted(rr_s, xr / xi, side="right"))
            np.add.at(acc, pidx_s[pos:end], blk_s[pos:end])
            pos = end
            e = specnorm_sym(laplacian(acc, pi, pj, N) - GS) / nG
            rows.append(dict(name=cfg["name"], set=cfg["set"], family=cfg["family"], N=N, L=L, xi=xi, part="real",
                             x=float(xr), cutoff=float(xr / xi), err=e, count=end, no_new=bool(end == prev_n),
                             exp_guide=float(np.exp(-xr ** 2))))
            prev_n = end
        # Fourier: exact lattice sums, sorted |k| shells
        KF, nm = fourier_ladder_pairs(d, L, 2 * xi * Y_KC, lambda kn: khat_AB(K, kn))
        prev_n = -1
        for y, Kp, n in zip(Y_KC, KF, nm):
            e = specnorm_sym(laplacian(Kp, pi, pj, N) - GL) / nG
            rows.append(dict(name=cfg["name"], set=cfg["set"], family=cfg["family"], N=N, L=L, xi=xi,
                             part="fourier", x=float(y), cutoff=float(2 * xi * y), err=e, count=n,
                             no_new=bool(n == prev_n), exp_guide=float(np.exp(-y ** 2))))
            prev_n = n
    return rows


def stage_cutoffs(args):
    cfgs = load_configs(sets=("legacy", "main", "check"))
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(_cutoff_one, cfgs):
            rows += res
            print(f"  cutoffs {res[0]['name']} done", flush=True)
    write_csv(OUT / "cutoffs.csv", rows)
    log_run("cutoffs", dict(x_rc=X_RC.tolist(), y_kc=Y_KC.tolist()))


# ----------------------------------------------------------------------------
# mesh: h and p at fixed kc, random global grid shifts; alias decomposition
# ----------------------------------------------------------------------------
def mesh_grid(L, kc, n=9, factor=4.0):
    """FFT-friendly M from the smallest admissible (eta_actual < 1, retained modes below Nyquist) to ~factor x."""
    M0 = int(np.floor(kc * L / np.pi)) + 1
    Ms = sorted({next_smooth(int(np.ceil(m))) for m in np.geomspace(M0, factor * M0, n)})
    mc = int(np.floor(kc * L / (2 * np.pi) + 1e-12))
    return [M for M in Ms if mc < M / 2 and kc * (L / M) / np.pi < 1]


def _mesh_one(job):
    cfg, xi, s_F, ps, n_shift, n_M = job
    ref = load_ref(cfg["name"])
    x, L, N = cfg["x"], cfg["L"], cfg["N"]
    pi, pj, d = pair_geometry(x, L)
    nG = float(ref["norm2"])
    K = EwaldLongitudinal(KAPPA, xi)
    kc = 2 * xi * s_F
    KF, nm = fourier_ladder_pairs(d, L, [kc], lambda kn: khat_AB(K, kn))
    GLkc = laplacian(KF[0], pi, pj, N)
    GL = laplacian(ref[f"KL_{xi:g}"], pi, pj, N)
    eF = specnorm_sym(GLkc - GL) / nG
    rng = np.random.default_rng([EXP_SEED, 7, N, cfg["seed"], int(round(100 * s_F))])
    rows = []
    for M in mesh_grid(L, kc, n=n_M):
        shifts = rng.uniform(0, L / M, size=(n_shift, 3))
        shifts[0] = 0.0                                        # shift 0 = the configuration as generated
        for p in ps:
            mesh = mesh_at(L, K, kc, M, p)
            errs = []
            for u in shifts:
                W, imag = mesh.pair_matrix((x + u) % L)
                errs.append(specnorm_sym(lap_full(W) - GLkc) / nG)
            errs = np.array(errs)
            rows.append(dict(name=cfg["name"], set=cfg["set"], family=cfg["family"], N=N, L=L, xi=xi, s_F=s_F,
                             kc=kc, M=M, h=L / M, xi_h=xi * L / M, eta=kc * (L / M) / np.pi, p=p,
                             n_modes=mesh.n_modes, n_excluded=mesh.n_excluded, shifts=n_shift,
                             err_rms=float(np.sqrt(np.mean(errs ** 2))), err_max=float(errs.max()),
                             err_min=float(errs.min()), err_shift0=float(errs[0]), eps_F=eF))
    return rows


def _decomp_one(job):
    cfg, xi, s_F, M, p = job
    x, L, N = cfg["x"], cfg["L"], cfg["N"]
    pi, pj, d = pair_geometry(x, L)
    ref = load_ref(cfg["name"])
    nG = float(ref["norm2"])
    K = EwaldLongitudinal(KAPPA, xi)
    kc = 2 * xi * s_F
    mesh = mesh_at(L, K, kc, M, p)
    W, _ = mesh.pair_matrix(x)
    KF, _ = fourier_ladder_pairs(d, L, [kc], lambda kn: khat_AB(K, kn))
    Fk = np.zeros((N, N, 3, 3))
    Fk[pi, pj] = KF[0]
    Fk[pj, pi] = KF[0]
    dec = decomposition(mesh, x)
    off = ~np.eye(N, dtype=bool)
    E = (W - Fk)
    recon = dec["ED"] + dec["ELin"] + dec["EQuad"]
    lap = lambda T: lap_full(np.where(off[:, :, None, None], T, 0.0))      # noqa: E731
    tot = specnorm_sym(lap(E))
    row = dict(name=cfg["name"], family=cfg["family"], N=N, L=L, xi=xi, kc=kc, M=M, p=p, eta=kc * (L / M) / np.pi,
               total_rel=tot / nG,
               identity_resid_rel_to_error=float(np.abs((E - recon)[off]).max() / np.abs(E[off]).max()),
               identity_resid_rel_to_K=float(np.abs((E - recon)[off]).max() / np.abs(Fk[off]).max()),
               linear_over_total=specnorm_sym(lap(dec["ELin"])) / tot,
               quadratic_over_total=specnorm_sym(lap(dec["EQuad"])) / tot,
               Dh_over_total=specnorm_sym(lap(dec["ED"])) / tot, D_rel=dec["D_rel"], alpha_max=dec["alpha_max"],
               imag_rel=float(dec["imag"]))
    if N <= 128:                                              # literal spread/FFT/G/IFFT/gather, one source at a time
        Wl = literal_W(mesh, x)
        row["literal_vs_modespace"] = float(np.abs(Wl - W).max() / np.abs(W).max())
    return row


def stage_mesh(args):
    jobs = []
    cfgs = load_configs(sets=("legacy", "main"))
    for c in cfgs:                                            # production kc (s = 4.1), all p, 8 shifts
        jobs.append((c, 0.7, PARAM_SETS["T7"]["s"], MESH_P, MESH_SHIFTS, 9))
    for c in [c for c in cfgs if c["name"].endswith("_s1") or c["set"] == "legacy"]:
        jobs.append((c, 0.7, 5.5, (4, 6, 8), MESH_SHIFTS, 7))   # converged Fourier cutoff (e^{-s^2} ~ 7e-14)
    for c in load_configs(sets=("check",)):                   # N = 256, production kc, p 5 and 7, 4 shifts
        jobs.append((c, 0.7, PARAM_SETS["T7"]["s"], (5, 7), 4, 6))
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(_mesh_one, jobs):
            rows += res
            print(f"  mesh {res[0]['name']} s_F {res[0]['s_F']:.2f}: {len(res)} rows", flush=True)
    write_csv(OUT / "mesh.csv", rows)
    djobs = []
    for c in [c for c in cfgs if c["name"].endswith("_s1") or c["set"] == "legacy"]:
        kc = 2 * 0.7 * PARAM_SETS["T7"]["s"]
        Ms = mesh_grid(c["L"], kc, n=9)
        for M in (Ms[0], Ms[len(Ms) // 2]):
            for p in (4, 7):
                djobs.append((c, 0.7, PARAM_SETS["T7"]["s"], M, p))
    drows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(_decomp_one, djobs):
            drows.append(r)
            print(f"  decomposition {r['name']} M {r['M']} p {r['p']}: identity {r['identity_resid_rel_to_error']:.1e},"
                  f" linear/total {r['linear_over_total']:.3f}", flush=True)
    write_csv(OUT / "mesh_decomposition.csv", drows)
    log_run("mesh", dict(shifts=MESH_SHIFTS, p=MESH_P))


# ----------------------------------------------------------------------------
# production path: Gamma_h X
# ----------------------------------------------------------------------------
def probes(x, L, N, rng, n_gauss=16):
    """Mean-free Gaussian vectors; lowest nonzero lattice wave number along x, y, z, longitudinal and transverse,
    cos and sin, projected onto T_perp; the three translation modes."""
    P = []
    for i in range(n_gauss):
        g = rng.normal(size=(N, 3))
        P.append(("gauss", i, (g - g.mean(0)).ravel()))
    k = 2 * np.pi / L
    for a in range(3):
        for f, ph in (("cos", np.cos(k * x[:, a])), ("sin", np.sin(k * x[:, a]))):
            for b in range(3):
                X = np.zeros((N, 3))
                X[:, b] = ph
                X -= X.mean(0)
                P.append(("long" if a == b else "trans", f"k{'xyz'[a]}_{f}_e{'xyz'[b]}", X.ravel()))
    for a in range(3):
        X = np.zeros((N, 3))
        X[:, a] = 1.0
        P.append(("translation", "xyz"[a], X.ravel()))
    return P


def _action_one(job):
    cfg, pname = job
    prm = PARAM_SETS[pname]
    t0 = time.perf_counter()
    x, L, N = cfg["x"], cfg["L"], cfg["N"]
    ref = load_ref(cfg["name"])
    Gref, nG = ref["G"], float(ref["norm2"])
    xi = prm["xi"]
    op = GammaH(x % L, L, xi, prm["s"], prm["eta"], prm["p"], kappa=KAPPA)
    t_setup = time.perf_counter() - t0
    E = np.eye(3 * N)
    t1 = time.perf_counter()
    Glit = np.stack([op.matvec(E[:, j]) for j in range(3 * N)], axis=1)       # literal production matvec columns
    t_mat = (time.perf_counter() - t1) / (3 * N)
    Gms = op.dense()
    nGh = specnorm(Glit)
    D = Glit - Gref
    row = dict(name=cfg["name"], set=cfg["set"], family=cfg["family"], N=N, L=L, params=pname, **prm,
               rc=op.rc, kc=op.kc, M=op.mesh.M, h=op.mesh.h, eta_actual=op.mesh.eta_actual, n_modes=op.mesh.n_modes,
               n_excluded=op.mesh.n_excluded, k0_retained=bool(np.any(np.all(op.mesh.mv == 0, axis=1))),
               real_pair_images=len(op.i), err_spec_rel=specnorm(D) / nG, err_fro_rel=float(np.linalg.norm(D) /
                                                                                               np.linalg.norm(Gref)),
               literal_vs_modespace=float(np.abs(Glit - Gms).max() / np.abs(Gms).max()),
               sym_resid=float(np.linalg.norm(Glit - Glit.T) / np.linalg.norm(Glit)),
               t_setup_s=t_setup, t_matvec_s=t_mat)
    # probes through the actual matvec (not the assembled matrix)
    rng = np.random.default_rng([EXP_SEED, 11, N, cfg["seed"]])
    prow = []
    for kind, lab, X in probes(x, L, N, rng):
        y_h, y_r = op.matvec(X), Gref @ X
        nX, nR = np.linalg.norm(X), np.linalg.norm(y_r)
        prow.append(dict(name=cfg["name"], set=cfg["set"], family=cfg["family"], N=N, L=L, params=pname,
                         probe=kind, label=str(lab), ref_action_opnorm=nR / (nG * nX),
                         rel_err=(np.linalg.norm(y_h - y_r) / nR) if nR > 0 else np.nan,
                         opnorm_err=np.linalg.norm(y_h - y_r) / (nG * nX),
                         h_resid_opnorm=np.linalg.norm(y_h) / (nGh * nX)))
    for kind in ("gauss", "long", "trans"):
        sub = [p_ for p_ in prow if p_["probe"] == kind]
        row[f"{kind}_rel_max"] = max(p_["rel_err"] for p_ in sub)
        row[f"{kind}_rel_median"] = float(np.median([p_["rel_err"] for p_ in sub]))
        row[f"{kind}_refaction_min"] = min(p_["ref_action_opnorm"] for p_ in sub)
    tr = [p_ for p_ in prow if p_["probe"] == "translation"]
    row["trans_resid_h_opnorm"] = max(p_["h_resid_opnorm"] for p_ in tr)
    row["trans_resid_ref_opnorm"] = max(p_["ref_action_opnorm"] for p_ in tr)
    row["trans_err_abs_opnorm"] = max(p_["opnorm_err"] for p_ in tr)
    # T_perp spectrum
    T = np.zeros((3 * N, 3))
    for c in range(3):
        T[c::3, c] = 1 / np.sqrt(N)
    Q = null_space(T.T)
    lam_h = np.linalg.eigvalsh(Q.T @ (0.5 * (Glit + Glit.T)) @ Q)
    row.update(lam_star_ref=float(ref["lam_star"]), lam_star_h=float(lam_h[0]), lam_max_h=float(lam_h[-1]),
               weyl_margin=float(ref["lam_star"] - specnorm(D)))
    # error budget: real cutoff, Fourier cutoff, mesh (exact definitions, xi = 0.7 exact pieces from the reference)
    pi, pj, d = pair_geometry(x, L)
    K = EwaldLongitudinal(KAPPA, xi)
    I, J, dd, rr = pair_images(x, L, op.rc)
    pidx = I * N - I * (I + 1) // 2 + (J - I - 1)
    w = K.theta_S_scaled_closed(rr) * np.exp(-(xi * rr) ** 2) / rr ** 2
    KSrc = np.zeros((len(d), 3, 3))
    np.add.at(KSrc, pidx, w[:, None, None] * dd[:, :, None] * dd[:, None, :])
    KF, _ = fourier_ladder_pairs(d, L, [op.kc], lambda kn: khat_AB(K, kn))
    GS, GL = laplacian(ref[f"KS_{xi:g}"], pi, pj, N), laplacian(ref[f"KL_{xi:g}"], pi, pj, N)
    GSrc, GLkc = laplacian(KSrc, pi, pj, N), laplacian(KF[0], pi, pj, N)
    W, _ = op.mesh.pair_matrix(x % L)
    GLm = lap_full(W)
    row.update(eps_S=specnorm_sym(GSrc - GS) / nG, eps_F=specnorm_sym(GLkc - GL) / nG,
               eps_mesh=specnorm_sym(GLm - GLkc) / nG,
               real_matches_production=float(np.abs((GSrc + GLm) - Gms).max() / np.abs(Gms).max()),
               t_total_s=time.perf_counter() - t0)
    return row, prow


def stage_action(args):
    jobs = [(c, p) for c in load_configs() for p in (PARAM_SETS if c["set"] in ("legacy", "main", "check") else ("T7",))]
    rows, prows = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for row, prow in ex.map(_action_one, jobs):
            rows.append(row)
            prows += prow
            print(f"  action {row['name']} {row['params']}: spec {row['err_spec_rel']:.2e}, gauss max "
                  f"{row['gauss_rel_max']:.2e}, trans {row['trans_resid_h_opnorm']:.1e}, "
                  f"lam* {row['lam_star_h']:.4f}/{row['lam_star_ref']:.4f}", flush=True)
    write_csv(OUT / "action.csv", rows)
    write_csv(OUT / "action_probes.csv", prows)
    log_run("action", dict(params=PARAM_SETS))


STAGES = dict(configs=stage_configs, reference=stage_reference, quadrature=stage_quadrature, split=stage_split,
              cutoffs=stage_cutoffs, mesh=stage_mesh, action=stage_action)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for st in (list(STAGES) if args.stage == "all" else [args.stage]):
        t0 = time.perf_counter()
        print(f"[{now()}] stage {st}", flush=True)
        STAGES[st](args)
        print(f"[{now()}] stage {st} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
