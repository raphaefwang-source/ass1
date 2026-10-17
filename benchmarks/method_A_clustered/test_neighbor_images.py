#!/usr/bin/env python3
"""
Validation of the periodic KD-tree neighbour search and of the FULL-PERIODIC image multiplicity.

1. Exact set test. For uniform, clustered and boundary-placed configurations and rc/L from 0.2 to 2.3 (both
   rc < L/2 and rc > L/2, including rc > sqrt(3) L/2 where every pair is a candidate), the retained interaction set
   {(i, j, n): i < j, |q_i - q_j + nL| <= rc} of kdtree_images (shift "tight" and "cube") must equal the brute-force
   enumeration (all pairs x all shifts |n_a| <= ceil(rc/L) + 2) exactly. Also compared: the project's existing
   routines test_lanczos_fdt.real_pairs (min-image KD-tree for rc < L/2, all pairs x image vectors otherwise) and
   radial_kernels.real_pairs_tree (KD-tree against replicas for rc >= L/2).
2. Multiplicity. rc < L/2: at most one image per pair. rc > L/2: several; the unique-pair count of
   query_pairs(rc) and the image-interaction count are recorded with their ratio. Self images are counted
   and confirmed absent.
3. Action. Gamma_S^{rc} X from the KD-tree list vs the brute-force list vs a dense brute-force assembly, in the
   D - H form and the existing pair-difference form; full Gamma_h X (cached D - H operator) vs the existing
   FastFriction/GammaH path with image-enumerated pairs, at the LJ + A state point.

    python3 test_neighbor_images.py            # ~1 min; writes results/neighbor_validation.{json,csv}
"""
import time

import numpy as np

import common as C
from neighbor_images import (kdtree_images, brute_force_images, from_displacements, self_image_count, wrap,
                             candidate_pairs)
from cached_operator import CachedGammaH, mesh_for, pair_difference_action, short_range_weights
from test_lanczos_fdt import real_pairs
import radial_kernels as rk

RC_OVER_L = (0.2, 0.35, 0.49, 0.5, 0.51, 0.7, 0.866, 0.9, 1.2, 1.6, 2.3)


def test_configs(rng):
    out = []
    for N in (8, 27, 64, 128):
        out.append((f"uniform_N{N}", rng.uniform(0, 1, (N, 3))))
        c = np.array([[0.1, 0.1, 0.1], [0.6, 0.55, 0.9]])
        lab = rng.integers(0, 2, N)
        out.append((f"clustered_N{N}", wrap(c[lab] + 0.06 * rng.standard_normal((N, 3)), 1.0)))
    # particles on and next to the box faces, a pair across the periodic boundary, an exactly-at-cutoff pair
    xb = rng.uniform(0, 1, (24, 3))
    xb[:6, 0] = 0.0
    xb[6:12, 1] = 1.0 - 1e-15
    xb[12] = [0.01, 0.5, 0.5]
    xb[13] = [0.99, 0.5, 0.5]
    xb[14] = [0.0, 0.0, 0.0]
    xb[15] = [0.25, 0.0, 0.0]        # |d| = 0.25 exactly in binary: on the cutoff for rc/L = 0.25
    out.append(("boundary_N24", wrap(xb, 1.0)))
    return out


def compare_keys(a, b):
    ka, kb = a.keys(), b.keys()
    if len(ka) != len(kb):
        sa, sb = set(map(tuple, ka.tolist())), set(map(tuple, kb.tolist()))
        return False, len(sa - sb), len(sb - sa)
    eq = bool(np.all(ka == kb))
    return eq, (0 if eq else -1), (0 if eq else -1)


def margin_to_cutoff(il):
    r = np.linalg.norm(il.d, axis=1)
    return float(np.min(np.abs(r - il.rc))) if len(r) else float("nan")


def set_tests(rows, say):
    rng = np.random.default_rng([20261017, 1])
    n_fail = 0
    cfgs = test_configs(rng)
    for name, x in cfgs:
        rcs = RC_OVER_L + ((0.25,) if name.startswith("boundary") else ())
        for rcl in rcs:
            L = 1.0
            rc = rcl * L
            bf = brute_force_images(x, L, rc, extra=1)
            kt = kdtree_images(x, L, rc, shift="tight")
            kc = kdtree_images(x, L, rc, shift="cube")
            kn = kdtree_images(x, L, rc, shift="tight", use_tree=False)
            ok_t, miss_t, extra_t = compare_keys(kt, bf)
            ok_c, _, _ = compare_keys(kc, bf)
            ok_n, _, _ = compare_keys(kn, bf)
            same_d = bool(np.array_equal(kt.d[np.lexsort((kt.n[:, 2], kt.n[:, 1], kt.n[:, 0], kt.j, kt.i))],
                                         bf.d[np.lexsort((bf.n[:, 2], bf.n[:, 1], bf.n[:, 0], bf.j, bf.i))])) if ok_t else False
            # existing project routines
            i0, j0, d0 = real_pairs(x, L, rc)
            ex = from_displacements(x, L, i0, j0, d0, rc, "real_pairs")
            ok_ex, m_ex, e_ex = compare_keys(ex, bf)
            i1, j1, d1 = rk.real_pairs_tree(x, L, rc)
            et = from_displacements(x, L, i1, j1, d1, rc, "real_pairs_tree")
            ok_et, m_et, e_et = compare_keys(et, bf)
            # candidate completeness: every brute-force pair must be a KD-tree candidate at rc
            ci, cj = candidate_pairs(wrap(x, L), L, rc)
            cand = set(zip(ci.tolist(), cj.tolist()))
            bf_pairs = set(zip(bf.i.tolist(), bf.j.tolist()))
            st = bf.stats()
            row = dict(test="set", config=name, N=len(x), L=L, rc=rc, rc_over_L=rcl, regime="rc<L/2" if rc < L / 2 else "rc>=L/2",
                       n_candidates_kdtree=int(len(ci)), n_unique_pairs=st["n_unique_pairs"],
                       n_image_interactions=st["n_image_interactions"], images_per_pair=st["images_per_pair"],
                       max_images_per_pair=st["max_images_per_pair"], self_images_per_particle=self_image_count(L, rc),
                       self_images_in_list=int(np.sum(kt.i == kt.j)),
                       kdtree_tight_eq_brute=ok_t, kdtree_cube_eq_brute=ok_c, notree_eq_brute=ok_n,
                       displacements_bitwise_equal=same_d, candidates_complete=bf_pairs <= cand,
                       existing_real_pairs_eq=ok_ex, existing_real_pairs_tree_eq=ok_et,
                       existing_missing=(m_ex, m_et), existing_extra=(e_ex, e_et),
                       min_margin_to_cutoff=margin_to_cutoff(bf))
            fail = not (ok_t and ok_c and ok_n and same_d and row["candidates_complete"]
                        and row["self_images_in_list"] == 0 and (rc >= L / 2 or st["max_images_per_pair"] <= 1))
            row["pass"] = not fail
            n_fail += fail
            rows.append(row)
        say(f"  {name}: {len(rcs)} cutoffs, all exact = {all(r['pass'] for r in rows if r['config'] == name)}")
    return n_fail


def action_tests(rows, say):
    """Physical units: LJ + A state point, xi = 0.85 kernel, N = 64 and 256, rc < L/2 and rc > L/2."""
    n_fail = 0
    rng = np.random.default_rng([20261017, 2])
    K = C.kernel(0.85)
    for N, rcs in ((64, (1.5, 2.6, 4.824)), (256, (3.0, 4.824, 6.0))):
        L = C.box_length(N)
        x = rsa(N, L, C.D_MIN, rng)     # LJ-like minimum separation, so no accidental near-collision dominates
        X = rng.standard_normal((N, 3))
        for rc in rcs:
            kt = kdtree_images(x, L, rc)
            bf = brute_force_images(x, L, rc)
            wk, wb = short_range_weights(K, kt.d), short_range_weights(K, bf.d)
            a_k = pair_difference_action(kt.i, kt.j, kt.d, wk, X, N)
            a_b = pair_difference_action(bf.i, bf.j, bf.d, wb, X, N)
            # dense brute-force Gamma_S^{rc} (blocks summed image by image)
            G = np.zeros((N, 3, N, 3))
            Kp = wb[:, None, None] * bf.d[:, :, None] * bf.d[:, None, :]
            for (p_, q_, sg) in ((bf.i, bf.j, -1.0), (bf.j, bf.i, -1.0), (bf.i, bf.i, 1.0), (bf.j, bf.j, 1.0)):
                np.add.at(G, (p_[:, None, None], np.arange(3)[None, :, None], q_[:, None, None],
                              np.arange(3)[None, None, :]), sg * Kp)
            a_dense = (G.reshape(3 * N, 3 * N) @ X.ravel()).reshape(N, 3)
            # D - H form on the KD-tree list
            DS = np.zeros((N, 3, 3))
            KpK = wk[:, None, None] * kt.d[:, :, None] * kt.d[:, None, :]
            np.add.at(DS, kt.i, KpK)
            np.add.at(DS, kt.j, KpK)
            wd = wk[:, None] * kt.d
            H = np.zeros((N, 3))
            np.add.at(H, kt.i, np.einsum("pa,pa->p", kt.d, X[kt.j])[:, None] * wd)
            np.add.at(H, kt.j, np.einsum("pa,pa->p", kt.d, X[kt.i])[:, None] * wd)
            a_DH = np.einsum("nab,nb->na", DS, X) - H
            # minimum-image-only (one image per pair), to show what is lost for rc > L/2
            i0, j0 = candidate_pairs(wrap(x, L), L, rc)
            dmi = x[i0] - x[j0]
            dmi -= L * np.round(dmi / L)
            keep = np.linalg.norm(dmi, axis=1) <= rc
            a_mi = pair_difference_action(i0[keep], j0[keep], dmi[keep], short_range_weights(K, dmi[keep]), X, N)
            sc = np.linalg.norm(a_dense)
            row = dict(test="action", config=f"rsa_N{N}", N=N, L=L, rc=rc, rc_over_L=rc / L,
                       regime="rc<L/2" if rc < L / 2 else "rc>=L/2", n_unique_pairs=bf.unique_pairs(),
                       n_image_interactions=bf.n_images, images_per_pair=bf.n_images / max(bf.unique_pairs(), 1),
                       sets_equal=compare_keys(kt, bf)[0],
                       rel_diff_kdtree_vs_brute=float(np.linalg.norm(a_k - a_b) / sc),
                       rel_diff_kdtree_vs_dense=float(np.linalg.norm(a_k - a_dense) / sc),
                       rel_diff_DH_vs_pairdiff=float(np.linalg.norm(a_DH - a_k) / sc),
                       rel_diff_min_image_only=float(np.linalg.norm(a_mi - a_dense) / sc))
            row["pass"] = bool(row["sets_equal"] and row["rel_diff_kdtree_vs_brute"] < 1e-13
                               and row["rel_diff_kdtree_vs_dense"] < 1e-13 and row["rel_diff_DH_vs_pairdiff"] < 1e-13)
            n_fail += not row["pass"]
            rows.append(row)
            say(f"  N={N} rc={rc:.3f} (rc/L={rc / L:.2f}): images {bf.n_images}, unique pairs {bf.unique_pairs()}, "
                f"kdtree-brute {row['rel_diff_kdtree_vs_brute']:.1e}, D-H vs pair-diff {row['rel_diff_DH_vs_pairdiff']:.1e}, "
                f"min-image-only error {row['rel_diff_min_image_only']:.2e}")
    return n_fail


def rsa(N, L, dmin, rng):
    from lj075_common import rsa_positions
    return rsa_positions(N, L, dmin, rng)


def full_operator_tests(rows, say):
    """Cached D - H Gamma_h (KD-tree images) vs the existing FastFriction GammaH (image enumeration), same mesh."""
    n_fail = 0
    rng = np.random.default_rng([20261017, 3])
    b = C.BASELINE
    for N in (64, 256):
        L = C.box_length(N)
        x = rsa(N, L, C.D_MIN, rng)
        K = C.kernel(b["xi"])
        ff = rk.FastFriction(L, K, b["s"], b["eta"], b["p"], pair_search="images")
        old = ff.gamma_h(x)
        new = CachedGammaH(x, L, K, ff.rc, ff.mesh)
        Xs = [rng.standard_normal((N, 3)) for _ in range(4)]
        diffs = [float(np.linalg.norm(new.apply(X) - old.matvec(X.ravel()).reshape(N, 3))
                       / np.linalg.norm(old.matvec(X.ravel()))) for X in Xs]
        dstag = [float(np.linalg.norm(new.apply_staged(X) - new.apply(X)) / np.linalg.norm(new.apply(X))) for X in Xs]
        row = dict(test="full_operator", config=f"rsa_N{N}", N=N, L=L, rc=ff.rc, rc_over_L=ff.rc / L,
                   regime="rc<L/2" if ff.rc < L / 2 else "rc>=L/2", n_image_interactions=new.n_images,
                   n_image_interactions_existing=int(len(old.i)),
                   max_rel_diff_cached_vs_existing=max(diffs), max_rel_diff_staged_vs_cached=max(dstag))
        row["pass"] = bool(row["n_image_interactions"] == row["n_image_interactions_existing"] and max(diffs) < 1e-13
                           and max(dstag) < 1e-14)
        n_fail += not row["pass"]
        rows.append(row)
        say(f"  N={N}: cached D-H vs existing GammaH {max(diffs):.1e}, staged vs cached {max(dstag):.1e}, "
            f"images {new.n_images} = {len(old.i)}")
    return n_fail


def timing_tests(rows, say):
    rng = np.random.default_rng([20261017, 4])
    for N in (256, 512):
        L = C.box_length(N)
        x = rsa(N, L, C.D_MIN, rng)
        for rc in (3.0, 4.824):
            t = {}
            for name, fn in (("kdtree_tight", lambda: kdtree_images(x, L, rc)),
                             ("kdtree_cube", lambda: kdtree_images(x, L, rc, shift="cube")),
                             ("existing_real_pairs", lambda: real_pairs(x, L, rc)),
                             ("existing_real_pairs_tree", lambda: rk.real_pairs_tree(x, L, rc))):
                ts = C.median_time(fn, reps=5, warm=1)
                t[name] = float(np.median(ts))
            rows.append(dict(test="timing", config=f"rsa_N{N}", N=N, L=L, rc=rc, rc_over_L=rc / L,
                             regime="rc<L/2" if rc < L / 2 else "rc>=L/2",
                             **{f"t_{k}_ms": 1e3 * v for k, v in t.items()}, **{"pass": True}))
            say(f"  N={N} rc={rc}: " + ", ".join(f"{k} {1e3 * v:.2f} ms" for k, v in t.items()))


def main():
    C.log_command()
    t0 = time.perf_counter()
    rows = []
    say = C.log
    say("1. exact (i, j, n) set tests: KD-tree + shift enumeration vs brute force")
    f1 = set_tests(rows, say)
    say("2-3. short-range action at the LJ + A state point")
    f2 = action_tests(rows, say)
    say("3b. full cached Gamma_h vs existing GammaH")
    f3 = full_operator_tests(rows, say)
    say("4. enumeration timing")
    timing_tests(rows, say)
    sets = [r for r in rows if r["test"] == "set"]
    summary = dict(
        n_set_tests=len(sets), n_set_fail=f1, n_action_fail=f2, n_full_operator_fail=f3,
        all_pass=(f1 + f2 + f3) == 0,
        existing_real_pairs_all_equal=all(r["existing_real_pairs_eq"] for r in sets),
        existing_real_pairs_tree_all_equal=all(r["existing_real_pairs_tree_eq"] for r in sets),
        max_images_per_pair_below_half=max(r["max_images_per_pair"] for r in sets if r["regime"] == "rc<L/2"),
        max_images_per_pair_above_half=max(r["max_images_per_pair"] for r in sets if r["regime"] == "rc>=L/2"),
        wall_s=time.perf_counter() - t0)
    C.write_json(C.RESULTS / "neighbor_validation.json", dict(summary=summary, rows=rows, provenance=C.provenance()))
    C.write_csv(C.RESULTS / "neighbor_validation.csv", rows)
    say(f"summary: {summary}")
    if not summary["all_pass"]:
        raise SystemExit("neighbor validation FAILED")


if __name__ == "__main__":
    main()
