"""Atom -> molecule (CG bead) mapping: centre-of-mass positions, total momenta, total forces (vectorised)."""
import numpy as np


def unwrap_molecules(x, mol, L):
    """Make each molecule whole: minimum-image every atom relative to the first atom of its molecule."""
    _, first = np.unique(mol, return_index=True)
    idx = np.searchsorted(np.unique(mol), mol)
    ref = x[..., first[idx], :]
    d = x - ref
    d -= L * np.round(d / L)
    return ref + d


def com_map(x, mol, mass, L, v=None, f=None):
    """x: (..., n_atoms, 3). Returns Q (..., M, 3) wrapped into the box, P = sum m v, F = sum f per molecule."""
    ids, idx = np.unique(mol, return_inverse=True)
    M = len(ids)
    xu = unwrap_molecules(x, mol, L)
    mtot = np.bincount(idx, mass, M)

    def seg_sum(a):
        flat = a.reshape(-1, a.shape[-2], 3)
        out = np.stack([np.stack([np.bincount(idx, fr[:, c], M) for c in range(3)], axis=1) for fr in flat])
        return out.reshape(a.shape[:-2] + (M, 3))

    Q = seg_sum(xu * mass[:, None]) / mtot[:, None]
    out = dict(Q=Q % L, M=mtot)
    if v is not None:
        out["P"] = seg_sum(v * mass[:, None])
    if f is not None:
        out["F"] = seg_sum(f)
    return out
