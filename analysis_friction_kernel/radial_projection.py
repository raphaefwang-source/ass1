"""Pair geometry, longitudinal/transverse projections and radial binning with honest uncertainties.

Sign convention (Lyu-Lei, PRL 131, 177301): Markovian memory K(Q,t) = -Gamma(Q) delta(t) with Gamma PSD and
Gamma_ii = -sum_{j!=i} Gamma_ij (momentum conservation). The PAIR friction tensor is therefore
    G_ij := -Gamma_ij  (i != j),
so that a DPD-like pair friction gamma(r) P_L gives G_ij = +gamma(r) P_L. Everything below works with G_ij:
    gamma_L = rhat^T G_ij rhat,    gamma_T = (tr G_ij - gamma_L) / 2."""
import numpy as np


def pairs(q, L, rmax):
    """Unordered minimum-image pairs i<j with |r_ij| < rmax. Returns i, j, d (r_i - r_j), r. Vectorised O(N^2)."""
    i, j = np.triu_indices(len(q), 1)
    d = q[i] - q[j]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=1)
    k = r < rmax
    return i[k], j[k], d[k], r[k]


def project_blocks(G, d, r):
    """G: (P,3,3) pair friction tensors, d: (P,3) separation vectors. Returns gamma_L, gamma_T, antisymmetric norm."""
    e = d / r[:, None]
    gl = np.einsum("pa,pab,pb->p", e, G, e)
    gt = 0.5 * (np.einsum("paa->p", G) - gl)
    return gl, gt


def bin_index(r, edges):
    b = np.searchsorted(edges, r, side="right") - 1
    return np.where((b >= 0) & (b < len(edges) - 1), b, -1)


def group_bin_stats(r, y, groups, edges):
    """Bin y by r. 'groups' labels statistically independent units (configurations / trajectory blocks).
    Returns per-bin: n, mean, std (conditional scatter at fixed r), naive SE, group-level SE (used for weights),
    per-group bin sums and counts (for bootstrap) and the pair-distance sample per bin (for exact bin-averaged
    model predictions)."""
    nb = len(edges) - 1
    b = bin_index(r, edges)
    k = b >= 0
    r, y, g, b = r[k], y[k], groups[k], b[k]
    G = int(groups.max()) + 1
    n = np.bincount(b, minlength=nb).astype(float)
    s1 = np.bincount(b, y, nb)
    s2 = np.bincount(b, y * y, nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0) * n / np.maximum(n - 1, 1))
        gs = np.zeros((G, nb))
        gc = np.zeros((G, nb))
        np.add.at(gs, (g, b), y)
        np.add.at(gc, (g, b), 1.0)
        gm = gs / gc
        ok = gc > 0
        G_eff = ok.sum(0)
        gvar = np.nansum(np.where(ok, (gm - mean) ** 2, 0), axis=0) / np.maximum(G_eff - 1, 1)
        se_group = np.sqrt(gvar / np.maximum(G_eff, 1))
        rbar = np.bincount(b, r, nb) / n
    return dict(edges=edges, n=n, mean=mean, std=std, se_naive=std / np.sqrt(n), se_group=se_group,
                rbar=rbar, group_sum=gs, group_cnt=gc, r_samples=r, b_samples=b)
