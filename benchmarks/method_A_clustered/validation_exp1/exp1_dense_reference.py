"""Independent mode-space matrix cross-check for Experiment 1.
Extracted unmodified from historical p-search script.
"""
import numpy as np
from error_validation import PAIRS6

def mode_pair_matrix_fast(mesh, x):
    """Mesh.pair_matrix in real arithmetic: the +-k terms of the mode sum are complex conjugates (real stencil
    weights, even G_h), so W = V^-1 [S_0 S_0^T G_h(0) + 2 sum_{k in half} (Re S Re S^T + Im S Im S^T) G_h(k)]."""
    s = mesh.stencil_dft(x)
    o, mv = mesh.mc, mesh.mv
    half = (mv[:, 2] > 0) | ((mv[:, 2] == 0) & (mv[:, 1] > 0)) | ((mv[:, 2] == 0) & (mv[:, 1] == 0) & (mv[:, 0] > 0))
    zero = np.all(mv == 0, axis=1)
    assert 2 * int(half.sum()) + int(zero.sum()) == len(mv)
    S = s[:, 0, mv[half, 0] + o] * s[:, 1, mv[half, 1] + o] * s[:, 2, mv[half, 2] + o]
    A = np.concatenate([S.real, S.imag], axis=1)
    Gh = mesh.Gm[half]
    N = len(x)
    W = np.empty((N, N, 3, 3))
    S0 = (s[:, 0, o] * s[:, 1, o] * s[:, 2, o]).real if zero.any() else None
    for a, b in PAIRS6:
        g = np.concatenate([Gh[:, a, b], Gh[:, a, b]])
        Z = (A * g) @ A.T * (2.0 / mesh.V)
        if S0 is not None:
            Z += np.outer(S0, S0) * (mesh.Gm[zero][0, a, b] / mesh.V)
        W[:, :, a, b] = W[:, :, b, a] = Z
    return W
