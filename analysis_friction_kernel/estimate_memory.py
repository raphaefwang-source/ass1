"""Estimators of pair friction / memory from the three possible data types.

CASE 1  friction tensor Gamma (3M x 3M) available per configuration     -> pair_friction_from_tensor
CASE 2  constrained-dynamics unresolved forces dF_i(t) with the CG
        coordinates held fixed (Lyu-Lei K_MZ = <(e^{Rt} Q L P)(Q L P)^T>) -> pair_memory_from_constrained_forces
CASE 3  ordinary positions/velocities only: the MZ memory is NOT identified. markov_drift_regression only
        estimates the friction of an ASSUMED Markovian pairwise model (and requires the conservative force to be
        known and removed); it is a model-fit diagnostic, not an MZ estimate.

Sign: pair quantities are reported for G_ij = -Gamma_ij (see radial_projection.py), i.e.
    K_pair(r,t) = -beta < dF_i(t) dF_j(0)^T >  (i != j),   G_eff(r; T) = int_0^T K_pair(r,t) dt  (one-sided,
Green-Kubo: Gamma = beta int_0^inf <dF(t) dF(0)^T> dt)."""
import numpy as np

from radial_projection import pairs, project_blocks


# ----------------------------------------------------------------------------- CASE 1
def pair_friction_from_tensor(Gamma, q, L, rmax):
    """Gamma: (3M,3M) or (M,3,M,3). Returns pair arrays r, gamma_L, gamma_T, d and the symmetry defect."""
    M = len(q)
    G4 = Gamma.reshape(M, 3, M, 3)
    i, j, d, r = pairs(q, L, rmax)
    B = -G4[i, :, j, :]                                    # pair friction tensors G_ij
    asym = np.linalg.norm(B - np.transpose(G4[j, :, i, :], (0, 2, 1)) * -1, axis=(1, 2))
    gl, gt = project_blocks(0.5 * (B + np.transpose(B, (0, 2, 1))), d, r)
    return dict(r=r, gL=gl, gT=gt, d=d, i=i, j=j, sym_defect=float(asym.max() / max(np.abs(B).max(), 1e-300)))


# ----------------------------------------------------------------------------- CASE 2
def _xcorr_fft(a, b, nlag):
    """Symmetrised cross-correlation 0.5(<a(t+l) b(t)> + <b(t+l) a(t)>), l = 0..nlag-1, rows are series."""
    T = a.shape[-1]
    n = 1 << int(np.ceil(np.log2(2 * T)))
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    c = np.fft.irfft(A * B.conj(), n)
    pos, neg = c[..., :nlag], c[..., (n - np.arange(nlag)) % n]
    return 0.5 * (pos + neg) / (T - np.arange(nlag))


def pair_memory_from_constrained_forces(dF, q, L, beta, dt, edges, nlag, max_pairs_per_bin=300, rng=None,
                                        chunk=256):
    """dF: (T, M, 3) unresolved forces F - <F | Q> from constrained dynamics at fixed CG configuration q.
    Returns, for a subsample of pairs (at most max_pairs_per_bin per radial bin):
        r, K_L(t), K_T(t) per pair (pair memory, sign as in module doc), plus split-half time integrals
        (to separate estimation noise from genuine conditional scatter)."""
    rng = np.random.default_rng(0) if rng is None else rng
    dF = dF - dF.mean(axis=0)
    i, j, d, r = pairs(q, L, edges[-1])
    b = np.searchsorted(edges, r, side="right") - 1
    pick = np.concatenate([rng.permutation(np.nonzero(b == k)[0])[:max_pairs_per_bin] for k in range(len(edges) - 1)])
    i, j, d, r = i[pick], j[pick], d[pick], r[pick]
    e = d / r[:, None]
    T = dF.shape[0]
    h = T // 2
    KL = np.empty((len(r), nlag))
    KT = np.empty((len(r), nlag))
    halves = np.empty((2, 2, len(r)))                       # (half, L/T, pair) time integrals up to nlag
    w = np.full(nlag, dt)
    w[0] = 0.5 * dt
    for s in range(0, len(r), chunk):
        sl = slice(s, s + chunk)
        Fi = dF[:, i[sl], :].transpose(1, 2, 0)              # (P,3,T)
        Fj = dF[:, j[sl], :].transpose(1, 2, 0)
        ai = np.einsum("pat,pa->pt", Fi, e[sl])
        aj = np.einsum("pat,pa->pt", Fj, e[sl])
        for part, (t0, t1) in enumerate(((0, T), (0, h), (h, T))):
            cl = -beta * _xcorr_fft(ai[:, t0:t1], aj[:, t0:t1], nlag)
            ctr = -beta * _xcorr_fft(Fi[..., t0:t1], Fj[..., t0:t1], nlag).sum(axis=1)
            ct = 0.5 * (ctr - cl)
            if part == 0:
                KL[sl], KT[sl] = cl, ct
            else:
                halves[part - 1, 0, sl] = cl @ w
                halves[part - 1, 1, sl] = ct @ w
    return dict(r=r, KL=KL, KT=KT, t=np.arange(nlag) * dt, halves=halves, i=i, j=j)


def cumulative_integral(K, dt):
    """G(T) = int_0^T K dt (trapezoid) along the last axis."""
    c = np.cumsum(K, axis=-1) * dt - 0.5 * dt * (K[..., :1] + K)
    return c


# ----------------------------------------------------------------------------- CASE 3 diagnostic
def markov_drift_regression(Qh, V, dP, L, dt, edges, block_len=100):
    """Least-squares fit of a Markovian PAIRWISE friction model
        dP_i = -dt * sum_j [gL(r_ij) P_L + gT(r_ij) P_T] (v_i - v_j) + noise
    with piecewise-constant gL, gT on 'edges', using configurations Qh at which the friction acts.
    Only valid if (a) the dynamics is Markovian with pairwise friction and (b) conservative forces have been
    removed from dP. Returns normal-equation pieces per time block (for block bootstrap)."""
    nb = len(edges) - 1
    T, M, _ = V.shape
    nblk = int(np.ceil(T / block_len))
    XtX = np.zeros((nblk, 2 * nb, 2 * nb))
    Xty = np.zeros((nblk, 2 * nb))
    rs, bs, blk = [], [], []
    for t in range(T):
        i, j, d, r = pairs(Qh[t], L, edges[-1])
        b = np.searchsorted(edges, r, side="right") - 1
        e = d / r[:, None]
        dv = V[t, i] - V[t, j]
        fl = (e * np.einsum("pa,pa->p", e, dv)[:, None])           # P_L dv
        ft = dv - fl                                               # P_T dv
        X = np.zeros((M, 3, 2 * nb))
        for col0, f in ((0, fl), (nb, ft)):
            for c in range(3):
                X[:, c, col0:col0 + nb] += (np.bincount(i * nb + b, -dt * f[:, c], M * nb)
                                            - np.bincount(j * nb + b, -dt * f[:, c], M * nb)).reshape(M, nb)
        X = X.reshape(3 * M, 2 * nb)
        k = t // block_len
        XtX[k] += X.T @ X
        Xty[k] += X.T @ dP[t].ravel()
        if t % 25 == 0:
            rs.append(r)
            bs.append(b)
    return dict(XtX=XtX, Xty=Xty, r_samples=np.concatenate(rs), b_samples=np.concatenate(bs))


def solve_blocks(XtX, Xty, idx=None):
    idx = np.arange(len(XtX)) if idx is None else idx
    return np.linalg.lstsq(XtX[idx].sum(0), Xty[idx].sum(0), rcond=None)[0]
