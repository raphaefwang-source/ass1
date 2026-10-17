#!/usr/bin/env python3
"""
Particle configurations for the clustered-particle benchmark (LJ + A state point: rho = 0.75, L = (N/rho)^(1/3)).

Families (same N, L, mean density, kernel and random seeds; only the cluster widths and centres differ):
    uniform    uniform proposal (sigma = infinity)
    moderate   3 Gaussian clouds, requested per-axis width sigma = 0.20 L
    strong     2 Gaussian clouds, requested per-axis width sigma = 0.08 L      (primary stress test)
    lj_equilibrated   the committed equilibrated LJ + A state (lj075_results/hpc_pilot/init_state_eqA_lgv_s305_t50.npz,
                      t = 50 of law-A dynamics, N = 256 only): physically motivated validation set
Excluded: fft_vs_direct_results/states/*_liquid_clustered.npz (double-well toy state point, rho = 64/5.5^3, kT = 0.7),
which is a different state point and must not be mixed with the LJ parameters.

Minimum separation d_min = 0.9 under PERIODIC distances (the RSA d_min of the lj075 canonical starts; the
equilibrated LJ state has r_min = 0.924). Generation: positions are drawn from the (wrapped) Gaussian mixture, then
overlaps are removed by pairwise periodic push-apart (each pair closer than 1.01 d_min is moved apart along its
minimum-image direction) until every periodic pair distance is >= d_min. A requested width that is too narrow for
the particles to fit is thereby widened; if the push-apart does not converge, the width itself is relaxed by a
factor 1.15 and the draw repeated. Overlapping particles are never accepted and the kernel is not modified. The
requested width, the width actually drawn from, and the realised RMS spread of each cloud are recorded.

    python3 clustered_configurations.py [--N 64 256 512]     # writes configs/*.npz, results/configurations.csv
"""
import argparse

import numpy as np
from scipy.spatial import cKDTree

import common as C
from neighbor_images import wrap, kdtree_images

FAMILIES = {
    "uniform": dict(code=0, sigma=None, centres=None),
    "moderate": dict(code=1, sigma=0.20, centres=((0.20, 0.20, 0.25), (0.70, 0.35, 0.70), (0.35, 0.75, 0.80))),
    "strong": dict(code=2, sigma=0.08, centres=((0.25, 0.25, 0.25), (0.75, 0.75, 0.75))),
}
SEEDS = {64: (1, 2, 3), 256: (1, 2), 512: (1,)}
LJ_STATE = C.EWALD / "lj075_results" / "hpc_pilot" / "init_state_eqA_lgv_s305_t50.npz"
EXCLUDED = ["ewald_longitudinal/fft_vs_direct_results/states/N*_liquid_clustered.npz (double-well liquid at the toy "
            "state point rho = 64/5.5^3, kT = 0.7; not the LJ rho = 0.75 state point)"]
R_NEIGH = 1.5           # first-shell neighbour radius (LJ units)
R_BASE = 4.824          # baseline real-space cutoff s/xi = 4.1/0.85
HIST_EDGES = np.linspace(0.0, 10.0, 201)


def min_periodic_separation(x, L):
    d, _ = cKDTree(wrap(x, L), boxsize=L).query(wrap(x, L), k=2)
    return float(d[:, 1].min())


def push_apart(x, L, dmin, rng, target_factor=1.01, max_iter=200000, step_max=0.02):
    """Pairwise periodic overlap removal (overdamped soft-sphere relaxation); returns (x, iterations, converged).

    The per-particle displacement of one sweep is capped at step_max * dmin. Without the cap a particle inside a
    dense cloud, overlapping dozens of neighbours, is thrown far out in a single sweep and the cloud disperses
    to an almost uniform state; with it the cloud expands quasi-statically until the overlaps are gone."""
    target = target_factor * dmin
    x = wrap(x, L)
    for it in range(max_iter):
        pr = cKDTree(x, boxsize=L).query_pairs(target, output_type="ndarray")
        if pr.size == 0:
            return x, it, True
        i, j = pr[:, 0], pr[:, 1]
        d = x[i] - x[j]
        d -= L * np.round(d / L)
        r = np.linalg.norm(d, axis=1)
        if r.min() >= dmin and it > 0 and min_periodic_separation(x, L) >= dmin:
            return x, it, True
        tiny = r < 1e-9
        if tiny.any():
            d[tiny] = rng.standard_normal((int(tiny.sum()), 3)) * 1e-6
            r[tiny] = np.linalg.norm(d[tiny], axis=1)
        push = (0.5 * (target - r) / r)[:, None] * d
        disp = np.zeros_like(x)
        np.add.at(disp, i, push)
        np.add.at(disp, j, -push)
        nd = np.linalg.norm(disp, axis=1)
        big = nd > step_max * dmin
        disp[big] *= (step_max * dmin / nd[big])[:, None]
        x = wrap(x + disp, L)
    return x, max_iter, False


def generate(family, N, seed, dmin=C.D_MIN):
    spec = FAMILIES[family]
    L = C.box_length(N)
    rec = dict(family=family, N=N, seed=seed, L=L, density=N / L ** 3, d_min_required=dmin,
               rng_seed=[20261017, N, seed])
    for attempt in range(40):
        rng = np.random.default_rng([20261017, N, seed, attempt])
        if spec["sigma"] is None:
            x0 = rng.uniform(0, L, (N, 3))
            lab = np.zeros(N, int)
            sig = None
        else:
            cen = np.array(spec["centres"]) * L
            lab = rng.permutation(np.arange(N) % len(cen))
            sig = spec["sigma"] * L * 1.15 ** attempt
            x0 = cen[lab] + sig * rng.standard_normal((N, 3))
        x, it, ok = push_apart(x0, L, dmin, rng)
        if ok and min_periodic_separation(x, L) >= dmin:
            break
    else:
        raise RuntimeError(f"{family} N={N}: no feasible configuration")
    rec.update(attempt=attempt, push_iterations=it, sigma_requested=None if sig is None else spec["sigma"] * L,
               sigma_requested_over_L=spec["sigma"], sigma_drawn=sig, sigma_relaxed=bool(attempt > 0),
               centres_frac=spec["centres"], labels=lab)
    if spec["sigma"] is not None:
        cen = np.array(spec["centres"]) * L
        dc = x - cen[lab]
        dc -= L * np.round(dc / L)
        rec["sigma_realised"] = [float(np.sqrt(np.mean(dc[lab == c] ** 2))) for c in range(len(cen))]
        rec["sigma_realised_over_requested"] = float(np.mean(rec["sigma_realised"]) / (spec["sigma"] * L))
    return x, rec


def structure_stats(x, L):
    """Pair-distance histogram (all periodic images up to 10), minimum separation, neighbour counts, S(k) small k."""
    N = len(x)
    il = kdtree_images(x, L, HIST_EDGES[-1])
    r = np.linalg.norm(il.d, axis=1)
    hist = np.histogram(r, HIST_EDGES)[0]
    shell = 4 / 3 * np.pi * (HIST_EDGES[1:] ** 3 - HIST_EDGES[:-1] ** 3)
    g = hist / (N * (N - 1) / 2 / L ** 3 * shell)
    cnt = {}
    for rad in (R_NEIGH, R_BASE):
        k = r <= rad
        cnt[rad] = np.bincount(il.i[k], minlength=N) + np.bincount(il.j[k], minlength=N)
    n = np.array([[a, b, c] for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3)
                  if 0 < a * a + b * b + c * c <= 4])
    kv = 2 * np.pi / L * n
    Sk = np.abs(np.exp(1j * x @ kv.T).sum(0)) ** 2 / N
    return dict(hist=hist, g=g, nb_first=cnt[R_NEIGH], nb_rc=cnt[R_BASE], Sk_small=Sk), dict(
        d_min_periodic=min_periodic_separation(x, L),
        n_pairs_below_1p0=int(np.sum(r < 1.0)),
        nb_first_mean=float(cnt[R_NEIGH].mean()), nb_first_std=float(cnt[R_NEIGH].std()),
        nb_first_min=int(cnt[R_NEIGH].min()), nb_first_max=int(cnt[R_NEIGH].max()),
        nb_rc_mean=float(cnt[R_BASE].mean()), nb_rc_std=float(cnt[R_BASE].std()),
        nb_rc_min=int(cnt[R_BASE].min()), nb_rc_max=int(cnt[R_BASE].max()),
        nb_rc_homogeneous=float((N - 1) / L ** 3 * 4 / 3 * np.pi * R_BASE ** 3),
        cv_nb_first=float(cnt[R_NEIGH].std() / cnt[R_NEIGH].mean()),
        S_k_max_small=float(Sk.max()), images_within_rc_base=int(np.sum(r <= R_BASE)),
        unique_pairs_within_rc_base=int(np.unique(il.i[r <= R_BASE] * N + il.j[r <= R_BASE]).size))


def save(name, x, rec, hist):
    C.CONFIGS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.CONFIGS / f"{name}.npz", q=x, L=rec["L"], labels=rec.get("labels", np.zeros(len(x), int)),
                        hist=hist["hist"], g=hist["g"], hist_edges=HIST_EDGES, nb_first=hist["nb_first"],
                        nb_rc=hist["nb_rc"], Sk_small=hist["Sk_small"])


def config_name(family, N, seed):
    return f"{family}_N{N}_s{seed}"


def load(name):
    with np.load(C.CONFIGS / f"{name}.npz") as d:
        return d["q"].copy(), float(d["L"])


def all_names(Ns=(64, 256, 512), families=("uniform", "moderate", "strong"), lj=True):
    out = [config_name(f, N, s) for N in Ns for f in families for s in SEEDS[N]]
    if lj and 256 in Ns:
        out.append("lj_equilibrated_N256_s305")
    return out


def main():
    C.log_command()
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[64, 256, 512])
    args = ap.parse_args()
    rows = []
    for N in args.N:
        for fam in FAMILIES:
            for seed in SEEDS[N]:
                x, rec = generate(fam, N, seed)
                hist, st = structure_stats(x, rec["L"])
                name = config_name(fam, N, seed)
                save(name, x, rec, hist)
                row = {k: v for k, v in rec.items() if k not in ("labels",)}
                row.update(st, name=name)
                rows.append(row)
                C.log(f"{name}: L={rec['L']:.4f} attempt={rec['attempt']} iters={rec['push_iterations']} "
                      f"dmin={st['d_min_periodic']:.4f} nb1.5 {st['nb_first_mean']:.2f}+-{st['nb_first_std']:.2f} "
                      f"[{st['nb_first_min']},{st['nb_first_max']}] nb_rc {st['nb_rc_mean']:.1f}+-{st['nb_rc_std']:.1f} "
                      f"S(k)max {st['S_k_max_small']:.1f} sigma_real/req {rec.get('sigma_realised_over_requested')}")
    if 256 in args.N:
        with np.load(LJ_STATE) as d:
            q, L = np.asarray(d["q"], float), float(d["L"])
        assert abs(L - C.box_length(256)) < 1e-12
        x = wrap(q, L)
        rec = dict(family="lj_equilibrated", N=256, seed=305, L=L, density=256 / L ** 3, d_min_required=C.D_MIN,
                   source=str(LJ_STATE.relative_to(C.REPO)), attempt=0, sigma_requested=None)
        hist, st = structure_stats(x, L)
        if st["d_min_periodic"] < C.D_MIN:
            raise RuntimeError("equilibrated LJ state violates d_min")
        name = "lj_equilibrated_N256_s305"
        save(name, x, rec, hist)
        rows.append(dict(rec, **st, name=name))
        C.log(f"{name}: dmin={st['d_min_periodic']:.4f} nb1.5 {st['nb_first_mean']:.2f}+-{st['nb_first_std']:.2f} "
              f"nb_rc {st['nb_rc_mean']:.1f}+-{st['nb_rc_std']:.1f} S(k)max {st['S_k_max_small']:.1f}")
    C.write_csv(C.RESULTS / "configurations.csv", rows)
    C.write_json(C.RESULTS / "configurations.json", dict(rows=rows, families=FAMILIES, excluded=EXCLUDED,
                                                         d_min=C.D_MIN, provenance=C.provenance()))


if __name__ == "__main__":
    main()
