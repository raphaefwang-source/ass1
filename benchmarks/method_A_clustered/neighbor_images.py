"""
Full-periodic real-space image enumeration for the short-range part Gamma_S^{rc} of Method A.

Retained interaction set (one entry per unordered particle pair and lattice shift, no self images):

    I(rc) = {(i, j, n) : 0 <= i < j < N,  n in Z^3,  |q_i - q_j + n L| <= rc},   d_ijn = q_i - q_j + n L.

A periodic KD-tree (scipy cKDTree(..., boxsize=L)) answers query_pairs(r) with the unique unordered pairs whose
MINIMUM-IMAGE distance is <= r. That is the candidate set: if any image of (i, j) lies within rc, the minimum
image does too, so no pair is missed. It is not the interaction set: for rc > L/2 one pair can have several images
inside rc, and every one of them is a separate term of the full-periodic kernel sum. kdtree_images() therefore
enumerates, for every candidate pair, all lattice shifts n with |d0 + nL| <= rc, d0 = x_i - x_j (x wrapped to
[0, L)^3, so d0 in (-L, L)^3):

    shift="cube"   all n with |n_a| <= ceil(rc/L) + 1 (simple safe range; ceil(rc/L) already suffices because
                   |d0_a + n_a L| <= rc and |d0_a| < L give |n_a| < rc/L + 1)
    shift="tight"  per pair and axis n_a in [ceil((-rc - d0_a)/L), floor((rc - d0_a)/L)], the exact range

Both apply the identical floating-point test |d0 + n L|^2 <= rc^2 (d0_a + n_a L, then the squared sum), so they
and brute_force_images() return bit-identical sets. Self images (i = j, n != 0) multiply X_i - X_i = 0 in the
graph-Laplacian pair-difference action and are never generated.
"""
import numpy as np
from scipy.spatial import cKDTree

# candidate radius inflation: the KD-tree computes distances its own way, so a pair whose minimum image lies
# within one ulp of rc could be dropped by it and kept by the exact test. The inflated candidate set is a superset;
# the exact test below decides membership.
CAND_INFLATE = 1e-12


class ImageList:
    """(i, j, n, d) of the retained interactions, plus the unique-pair bookkeeping."""

    def __init__(self, i, j, n, d, rc, L, n_candidates=None, method=""):
        self.i, self.j, self.n, self.d = i, j, n, d
        self.rc, self.L, self.method = float(rc), float(L), method
        self.n_candidates = n_candidates

    @property
    def n_images(self):
        return int(len(self.i))

    def unique_pairs(self):
        return np.unique(self.i.astype(np.int64) * (1 << 32) + self.j.astype(np.int64)).size

    def multiplicity(self):
        """Number of retained images per pair that has at least one."""
        key = self.i.astype(np.int64) * (1 << 32) + self.j.astype(np.int64)
        _, cnt = np.unique(key, return_counts=True)
        return cnt

    def keys(self):
        """Sorted structured array of (i, j, n0, n1, n2) for exact set comparison."""
        rec = np.empty(self.n_images, dtype=[("i", "i8"), ("j", "i8"), ("n0", "i8"), ("n1", "i8"), ("n2", "i8")])
        rec["i"], rec["j"] = self.i, self.j
        rec["n0"], rec["n1"], rec["n2"] = self.n[:, 0], self.n[:, 1], self.n[:, 2]
        return np.sort(rec, order=("i", "j", "n0", "n1", "n2"))

    def stats(self):
        mult = self.multiplicity()
        nu = int(mult.size)
        return dict(rc=self.rc, L=self.L, rc_over_L=self.rc / self.L, method=self.method,
                    n_candidates=self.n_candidates, n_unique_pairs=nu, n_image_interactions=self.n_images,
                    images_per_pair=self.n_images / nu if nu else float("nan"),
                    max_images_per_pair=int(mult.max()) if nu else 0)


def wrap(x, L):
    x = np.mod(np.asarray(x, float), L)
    x[x >= L] = 0.0                     # np.mod can return L for tiny negative inputs; cKDTree requires [0, L)
    return x


def candidate_pairs(x, L, r, use_tree=True):
    """Unique unordered pairs (i < j) with minimum-image distance <= r (periodic KD-tree)."""
    N = len(x)
    if not use_tree:
        i, j = np.triu_indices(N, 1)
        return i.astype(np.intp), j.astype(np.intp)
    pr = cKDTree(x, boxsize=L).query_pairs(r, output_type="ndarray")
    if pr.size == 0:
        return np.zeros(0, np.intp), np.zeros(0, np.intp)
    i, j = pr[:, 0].astype(np.intp), pr[:, 1].astype(np.intp)
    sw = i > j
    i[sw], j[sw] = j[sw], i[sw]
    return i, j


def _keep(d0, n, L, rc2):
    d = d0 + n * L
    r2 = np.einsum("pa,pa->p", d, d)
    keep = r2 <= rc2
    return d[keep], keep


def kdtree_images(x, L, rc, shift="tight", use_tree=True, chunk=1 << 22):
    """All retained (i, j, n) from periodic KD-tree candidates + explicit lattice-shift enumeration."""
    x = wrap(x, L)
    rc = float(rc)
    i, j = candidate_pairs(x, L, rc * (1 + CAND_INFLATE), use_tree=use_tree)
    n_cand = len(i)
    d0 = x[i] - x[j]
    rc2 = rc * rc
    if shift == "cube":
        m = int(np.ceil(rc / L)) + 1
        g = np.arange(-m, m + 1)
        S = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
        out_i, out_j, out_n, out_d = [], [], [], []
        step = max(1, chunk // len(S))
        for c0 in range(0, n_cand, step):
            dd = d0[c0:c0 + step, None, :] + S[None, :, :] * L
            r2 = np.einsum("psa,psa->ps", dd, dd)
            p, s = np.nonzero(r2 <= rc2)
            out_i.append(i[c0:c0 + step][p]); out_j.append(j[c0:c0 + step][p])
            out_n.append(S[s]); out_d.append(dd[p, s])
        cat = (lambda a, shp: np.concatenate(a) if a else np.zeros(shp))
        return ImageList(cat(out_i, (0,)).astype(np.intp), cat(out_j, (0,)).astype(np.intp),
                         cat(out_n, (0, 3)).astype(np.int64), cat(out_d, (0, 3)), rc, L, n_cand, "kdtree+" + shift)
    if shift != "tight":
        raise ValueError(shift)
    if rc < L / 2:
        # two images of one pair within rc would be 2 rc < L apart, but distinct images are >= L apart: the only
        # possible image is the minimum image n = -rint(d0/L) (a component with |d0_a| = L/2 exceeds rc anyway)
        n = -np.rint(d0 / L).astype(np.int64)
        d, keep = _keep(d0, n, L, rc2)
        return ImageList(i[keep], j[keep], n[keep], d, rc, L, n_cand, "kdtree+tight")
    lo = np.ceil((-rc - d0) / L).astype(np.int64)
    hi = np.floor((rc - d0) / L).astype(np.int64)
    cnt = np.clip(hi - lo + 1, 0, None)
    tot = cnt.prod(axis=1)
    csum = np.cumsum(tot)
    out = [[], [], [], []]
    c0 = 0
    while c0 < n_cand:                      # candidate blocks of about `chunk` shift tests
        base = csum[c0] - tot[c0]
        c1_ = int(np.searchsorted(csum, base + chunk, side="right"))
        c1_ = max(c1_, c0 + 1)
        sl = slice(c0, c1_)
        rep = np.repeat(np.arange(c0, c1_), tot[sl])
        off = np.repeat(csum[sl] - tot[sl], tot[sl])
        t = np.arange(base, base + rep.size, dtype=np.int64) - off
        c1, c2 = cnt[rep, 1], cnt[rep, 2]
        n = lo[rep] + np.stack([t // (c1 * c2), (t // c2) % c1, t % c2], axis=1)
        d, keep = _keep(d0[rep], n, L, rc2)
        out[0].append(i[rep][keep]); out[1].append(j[rep][keep]); out[2].append(n[keep]); out[3].append(d)
        c0 = c1_
    if not out[0]:
        return ImageList(np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros((0, 3), np.int64), np.zeros((0, 3)),
                         rc, L, n_cand, "kdtree+tight")
    return ImageList(*(np.concatenate(o) for o in out), rc, L, n_cand, "kdtree+tight")


def brute_force_images(x, L, rc, extra=1, chunk=1 << 22):
    """Reference: every pair i < j and every shift |n_a| <= ceil(rc/L) + 1 + extra (no KD-tree, no pruning)."""
    x = wrap(x, L)
    N = len(x)
    i, j = np.triu_indices(N, 1)
    d0 = x[i] - x[j]
    m = int(np.ceil(rc / L)) + 1 + extra
    g = np.arange(-m, m + 1)
    S = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    rc2 = float(rc) ** 2
    out = [[], [], [], []]
    step = max(1, chunk // len(S))
    for c0 in range(0, len(i), step):
        dd = d0[c0:c0 + step, None, :] + S[None, :, :] * L
        r2 = np.einsum("psa,psa->ps", dd, dd)
        p, s = np.nonzero(r2 <= rc2)
        out[0].append(i[c0:c0 + step][p]); out[1].append(j[c0:c0 + step][p])
        out[2].append(S[s]); out[3].append(dd[p, s])
    return ImageList(np.concatenate(out[0]).astype(np.intp), np.concatenate(out[1]).astype(np.intp),
                     np.concatenate(out[2]).astype(np.int64), np.concatenate(out[3]), rc, L, len(i), "brute_force")


def self_image_count(L, rc):
    """Number of self images (i = i, n != 0) within rc per particle: excluded from the pair-difference action."""
    m = int(np.ceil(rc / L)) + 1
    g = np.arange(-m, m + 1)
    S = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3) * L
    r = np.linalg.norm(S, axis=1)
    return int(((r > 0) & (r <= rc)).sum())


def from_displacements(x, L, i, j, d, rc, method):
    """ImageList from an (i, j, d) triple of another routine; n is recovered as round((d - (x_i - x_j)) / L)."""
    x = wrap(x, L)
    i, j, d = np.asarray(i, np.intp), np.asarray(j, np.intp), np.asarray(d, float)
    sw = i > j
    i, j, d = np.where(sw, j, i), np.where(sw, i, j), np.where(sw[:, None], -d, d)
    n = np.rint((d - (x[i] - x[j])) / L).astype(np.int64)
    return ImageList(i, j, n, d, rc, L, None, method)


def image_distances_upto(x, L, R):
    """All retained interactions up to radius R (sorted by distance), for cumulative counts N_img(rc), rc <= R."""
    il = kdtree_images(x, L, R)
    r = np.linalg.norm(il.d, axis=1)
    o = np.argsort(r, kind="stable")
    return il.i[o], il.j[o], il.d[o], r[o]
