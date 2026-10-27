"""Real-space PSD projection of the ORIGINAL deconvolved PPPM mesh.

All production convolution methods retain Mesh's convention: Gr has the
original Fourier-kernel units, and gather divides by h**3.  ``table`` and
``delta_table`` contain physical real-space tensors.  The dense evaluator
adds the exact sparse real-space correction to the original mode sum; it
therefore includes EVERY Fourier mode produced by the projection.
"""
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import common  # noqa: F401,E402 (historical import paths / thread defaults)
import numpy as np  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from test_parameter_selection import Mesh  # noqa: E402
from positive_mesh import PositiveMesh, PAIRS6  # noqa: E402
from cached_operator import CachedGammaH, mesh_for  # noqa: E402
from exp1_dense_reference import mode_pair_matrix_fast  # noqa: E402

COMPONENT_INDEX = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])


def _inverse(table):
    neg = (-np.arange(table.shape[1])) % table.shape[1]
    return table[:, neg][:, :, neg][:, :, :, neg]


def project_table(table, audit_eigenvalues=True):
    """Project symmetric 3x3 blocks, preserving inversion by positive averaging.

    The correction is assembled as -Q diag(min(lambda,0)) Q.T rather than by
    subtracting two almost equal reconstructed tables.  No negative eigenvalue
    or nonzero correction is discarded with a numerical threshold.

    ``audit_eigenvalues=False`` preserves the projected arrays exactly while
    skipping independent post/raw eigvalsh checks and correction quantiles.
    Uncomputed diagnostic values are None, rather than claimed exact zeros.
    """
    arr = np.asarray(table, float)
    if arr.ndim != 4 or arr.shape[0] != 6 or len(set(arr.shape[1:])) != 1 or not np.all(np.isfinite(arr)):
        raise ValueError('Require a finite (6,M,M,M) tensor table')
    t_start = time.perf_counter()
    sym = .5 * (arr + _inverse(arr))
    full = np.moveaxis(sym, 0, -1)[..., COMPONENT_INDEX]
    t_sym = time.perf_counter() - t_start
    t_start = time.perf_counter()
    eig, vec = np.linalg.eigh(full)
    t_eigh = time.perf_counter() - t_start
    t_start = time.perf_counter()
    negative = np.any(eig < 0, axis=-1)
    delta = np.zeros_like(sym)
    if negative.any():
        en, vn = eig[negative], vec[negative]
        correction = np.einsum('...ac,...c,...bc->...ab', vn, np.maximum(-en, 0), vn)
        for c, (a, b) in enumerate(PAIRS6):
            delta[c][negative] = correction[:, a, b]
        delta = .5 * (delta + _inverse(delta))
    projected = sym + delta
    t_clip = time.perf_counter() - t_start
    if audit_eigenvalues:
        t_start = time.perf_counter()
        post = np.linalg.eigvalsh(np.moveaxis(projected, 0, -1)[..., COMPONENT_INDEX])
        raw_eig = np.linalg.eigvalsh(np.moveaxis(arr, 0, -1)[..., COMPONENT_INDEX])
        min_post, min_raw = float(post.min()), float(raw_eig.min())
        t_eig_diagnostics = time.perf_counter() - t_start
    else:
        # Only the first eigh is required by the algorithm.  Do not report a
        # calculated post-projection eigenvalue when this audit was skipped.
        min_post = min_raw = None
        t_eig_diagnostics = 0.
    frob_weights = np.array([1, 1, 1, 2, 2, 2])[:, None, None, None]
    original_norm = float(np.sqrt(np.sum(frob_weights * sym**2)))
    correction_norm = float(np.sqrt(np.sum(frob_weights * delta**2)))
    spectral_correction = np.maximum(-eig[...,0],0.)
    eig_scale = float(np.max(np.abs(eig)))
    significant_threshold = 64*np.finfo(float).eps*eig_scale
    diagnostics = dict(
        min_eig_before=float(eig.min()), min_eig_after=min_post,
        min_eig_raw=min_raw, min_eig_symmetrized=float(eig.min()),
        diagnostic_eigenvalues_evaluated=bool(audit_eigenvalues),
        max_negative_part_spectral=float(spectral_correction.max()),
        correction_max_spectral=float(spectral_correction.max()),
        significant_negative_threshold=significant_threshold,
        significant_negative_eigenvalues=int((eig < -significant_threshold).sum()),
        significant_negative_blocks=int((eig[...,0] < -significant_threshold).sum()),
        negative_eigenvalues=int((eig < 0).sum()), negative_blocks=int(negative.sum()),
        total_blocks=int(np.prod(arr.shape[1:])), fraction_negative_blocks=float(negative.mean()),
        correction_fro=correction_norm, table_fro=original_norm,
        correction_relative_fro=correction_norm / max(original_norm, np.finfo(float).tiny),
        correction_max_abs=float(np.max(np.abs(delta))),
        correction_max_block_fro=float(np.sqrt(np.sum(frob_weights * delta**2, axis=0)).max()),
        inversion_residual_before=float(np.max(np.abs(arr-_inverse(arr)))),
        inversion_residual_after=float(np.max(np.abs(projected-_inverse(projected)))),
        t_table_symmetrize_s=t_sym, t_block_eigh_s=t_eigh,
        t_block_projection_s=t_clip, t_eigenvalue_diagnostics_s=t_eig_diagnostics,
    )
    for q in (25,50,90,99):
        diagnostics[f'correction_spectral_q{q}'] = (
            float(np.percentile(spectral_correction,q)) if audit_eigenvalues else None)
        diagnostics[f'correction_active_spectral_q{q}'] = (
            (float(np.percentile(spectral_correction[negative],q)) if negative.any() else 0.)
            if audit_eigenvalues else None)
    return projected, delta, diagnostics


class ProjectedMesh(Mesh):
    """Wrap an existing legal original Mesh; project its physical tensor table.

    ``project=False`` is an independent normalization/control path.  The
    B-spline implementation uses the existing B2 nonnegative recurrence and
    partition normalization, including the exact p=1 half-grid tie rule.
    Deconvolution still depends on p: rebuild this object when p changes.
    For production setup timing, audit_eigenvalues=False omits extra validation
    eigenvalue solves and quantiles; the mandatory projection eigh is identical.
    """

    stencil = PositiveMesh.stencil
    flat = PositiveMesh.flat

    def __init__(self, original, project=True, audit_eigenvalues=True):
        start = time.perf_counter()
        if not isinstance(original, Mesh) or not hasattr(original, 'Gr'):
            raise TypeError('Require an existing original Mesh with its rFFT grid')
        if not 1 <= int(original.p) <= original.M:
            raise ValueError('Require 1<=p<=M')
        self.__dict__.update(original.__dict__)
        self.original = original
        self.projected = bool(project)
        t_start = time.perf_counter()
        raw = np.fft.irfftn(original.Gr, s=(self.M,)*3, axes=(1, 2, 3)) / self.h**3
        t_ifft = time.perf_counter() - t_start
        projected, delta, info = project_table(raw, audit_eigenvalues=audit_eigenvalues)
        self.table_before = .5*(raw + _inverse(raw))
        self.table = projected if project else self.table_before
        self.delta_table = delta if project else np.zeros_like(raw)
        # Retain original coefficients exactly if projection is the identity.
        # Adding the correction in Fourier space is algebraically FFT(T+)*h^3.
        self.Gr = np.array(original.Gr, copy=True)
        t_start = time.perf_counter()
        if project and np.any(delta):
            transformed = np.fft.rfftn(delta, axes=(1, 2, 3)) * self.h**3
            self.Gr += transformed.real
            info['delta_fft_imag_max'] = float(np.abs(transformed.imag).max())
        else:
            info['delta_fft_imag_max'] = 0.
        t_fft = time.perf_counter() - t_start
        if not project:
            info.update(min_eig_after=info['min_eig_before'] if audit_eigenvalues else None, correction_fro=0.,
                        correction_relative_fro=0., correction_max_abs=0., correction_max_block_fro=0.)
        self.active_offsets = np.argwhere(np.any(self.delta_table != 0., axis=0))
        self.active_values = self.delta_table[(slice(None),) + tuple(self.active_offsets.T)].T
        self.zero_mode_before = np.array(original.Gr[:, 0, 0, 0], copy=True)
        self.zero_mode = self.Gr[:, 0, 0, 0].copy()
        self.diagnostics = info
        info.update(active_correction_offsets=int(len(self.active_offsets)),
                    zero_mode_shift_norm=float(np.linalg.norm(self.zero_mode-self.zero_mode_before)),
                    projection_enabled=bool(project), t_original_influence_s=float(original.t_setup),
                    t_table_ifft_s=t_ifft, t_projected_fft_s=t_fft)
        for c, name in enumerate(('xx','yy','zz','xy','xz','yz')):
            info[f'zero_before_{name}'] = float(self.zero_mode_before[c])
            info[f'zero_after_{name}'] = float(self.zero_mode[c])
        mask = np.zeros(self.Gr.shape[1:], bool)
        half = self.mv[:, 2] >= 0
        mask[self.mv[half,0] % self.M, self.mv[half,1] % self.M, self.mv[half,2]] = True
        info['new_fourier_modes_rfft'] = int(np.any(self.Gr[:,~mask] != 0., axis=0).sum())
        info['new_fourier_max_abs'] = float(np.max(np.abs(self.Gr[:,~mask]), initial=0.))
        self.t_projection_setup = time.perf_counter() - start
        self.t_setup = original.t_setup + self.t_projection_setup
        info.update(t_projection_adapter_total_s=self.t_projection_setup,
                    t_parameter_setup_total_s=self.t_setup)

    @classmethod
    def from_parameters(cls, L, K, kc, M, p, project=True, audit_eigenvalues=True):
        return cls(mesh_for(L, K, kc, M, p), project=project, audit_eigenvalues=audit_eigenvalues)

    def _axis_correlations(self, x):
        """C_d[a,i,j] = sum_g w_i,d[g] w_j,d[g-a], upper triangle.

        Compact 1D sparse stencils make this inexpensive.  It is independent
        of the 3D table and has no Fourier-band limitation.
        """
        g, w = self.stencil(x)
        N, _, p = g.shape
        tri = np.triu_indices(N)
        rows = np.repeat(np.arange(N), p)
        corr, inv = [], []
        for d in range(3):
            S = csr_matrix((w[:,d].ravel(), (rows, (g[:,d] % self.M).ravel())), shape=(N,self.M))
            dense = S.toarray()
            offsets, inverse = np.unique(self.active_offsets[:,d], return_inverse=True)
            C = np.empty((len(offsets), len(tri[0])))
            for k, offset in enumerate(offsets):
                z = S @ np.ascontiguousarray(np.roll(dense, int(offset), axis=1).T)
                C[k] = z[tri]
            corr.append(C)
            inv.append(inverse)
        return tri, corr, inv

    def pair_matrix(self, x, correction_chunk=32):
        """Exact dense W_ij, including all projected modes and all self terms.

        An original retained-mode sum plus literal sparse real-table stencil
        correction avoids a costly N^2*M^3 full-mode dense Fourier sum.  Every
        nonzero offset is retained, including tiny negative-eigenvalue repairs.
        """
        if int(correction_chunk) != correction_chunk or correction_chunk < 1:
            raise ValueError('Require a positive correction chunk')
        W = mode_pair_matrix_fast(self, np.asarray(x, float))
        if not len(self.active_offsets):
            return W, 0.
        tri, corr, inv = self._axis_correlations(x)
        out = np.zeros((6, len(tri[0])))
        for begin in range(0, len(self.active_offsets), int(correction_chunk)):
            end = min(begin+int(correction_chunk), len(self.active_offsets))
            C = corr[0][inv[0][begin:end]].copy()
            C *= corr[1][inv[1][begin:end]]
            C *= corr[2][inv[2][begin:end]]
            out += self.active_values[begin:end].T @ C
        for c, (a, b) in enumerate(PAIRS6):
            W[tri[0],tri[1],a,b] += out[c]
            if a != b:
                W[tri[0],tri[1],b,a] += out[c]
            # global inversion of the table guarantees W_ji = W_ij.
            W[tri[1],tri[0],a,b] = W[tri[0],tri[1],a,b]
            W[tri[1],tri[0],b,a] = W[tri[0],tri[1],b,a]
        return W, 0.

    def direct_pair_matrix(self, x):
        """Independent literal double 3D stencil contraction; small-N test only."""
        idx, wt = self.flat(x)
        grid = np.stack(np.unravel_index(idx, (self.M,)*3), axis=-1)
        N = len(x)
        W = np.empty((N,N,3,3))
        for i in range(N):
            for j in range(N):
                d = (grid[i,:,None,:] - grid[j,None,:,:]) % self.M
                values = self.table[:,d[...,0],d[...,1],d[...,2]]
                result = np.einsum('a,b,cab->c', wt[i], wt[j], values)
                W[i,j] = result[COMPONENT_INDEX]
        return W


class GridPSDGamma(CachedGammaH):
    """Existing short algorithm and original h^-3 convention, NEW mesh degree."""
    def __init__(self, x, L, K, rc, mesh, images=None):
        if not isinstance(mesh, ProjectedMesh) or mesh.L != L or mesh.K is not K:
            raise ValueError('Require the projected mesh of this box and kernel')
        theta = getattr(K, 'theta', 0.)
        if (not callable(theta) and theta != 0.) or getattr(K, 'law', 'A') != 'A':
            raise ValueError('PSD theorem requires the unchanged Candidate-1 law A split')
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.all(np.isfinite(x)):
            raise ValueError('Require finite nonempty positions of shape (N,3)')
        if not np.isfinite(rc) or rc <= 0 or len(np.unique(np.mod(x,L),axis=0)) != len(x):
            raise ValueError('Require finite rc>0 and distinct periodic particles')
        self.iso = False
        super().__init__(x, L, K, rc, mesh, images)
        if not np.all(np.isfinite(self.D)) or not np.all(np.isfinite(self.w)) or np.any(self.w < 0):
            raise FloatingPointError('Nonfinite degree or invalid short weights')


__all__ = ['ProjectedMesh','GridPSDGamma','project_table','COMPONENT_INDEX']
