"""Correction-only projection bounds; the projected PPPM method is unchanged.

C = mesh.delta_table is a PHYSICAL real-space tensor table.  Consequently
C_hat = FFT(C) convolves assignment masses directly, without an extra h^-3.
No new-old degree or operator subtraction is used to obtain the correction.
"""
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from grid_psd import ProjectedMesh, COMPONENT_INDEX, PAIRS6, np  # noqa: E402


class CorrectionBounds:
    """Reuse an existing projected mesh, without changing any of its arrays.

    ``bounds(x) -> (scalars,D)`` forms only N symmetric 3x3 degrees.  The exact
    arithmetic theorem is N*delta and 2*max_i lambda_max(D_i), with their minimum.
    N=1 has the separate exact zero correction graph.  Float diagnostics are
    reported separately; no tolerance, spectral floor or bound inflation is used.

    A mesh protocol with the same physical delta_table, positive stencil/flat,
    and max_negative_part_spectral diagnostic is accepted for independent tests.
    """

    def __init__(self,mesh,audit=False):
        start = time.perf_counter()
        if getattr(mesh,'projected',True) is not True:
            raise ValueError('Require the correction of ProjectedMesh(project=True)')
        self.mesh = mesh
        self.M,self.L,self.h,self.p = int(mesh.M),float(mesh.L),float(mesh.h),int(mesh.p)
        if self.M<2 or not 1<=self.p<=self.M:
            raise ValueError('Require M>=2 and 1<=p<=M')
        C = np.asarray(mesh.delta_table)
        if C.shape!=(6,self.M,self.M,self.M) or C.dtype.kind!='f' or not np.all(np.isfinite(C)):
            raise ValueError('Require a finite real physical (6,M,M,M) correction table')
        # A read-only view protects this API without changing the mesh's flags.
        self.C = C.view()
        self.C.flags.writeable = False
        self.table = self.C
        info = mesh.diagnostics
        self.delta = float(info['max_negative_part_spectral'])
        if not np.isfinite(self.delta) or self.delta<0:
            raise ValueError('Require a finite nonnegative negative-part spectral maximum')
        self.active_offsets = np.argwhere(np.any(self.C!=0.,axis=0))
        self.active_values = self.C[(slice(None),)+tuple(self.active_offsets.T)].T
        t_fft = time.perf_counter()
        self.C_hat = np.fft.rfftn(self.C,axes=(1,2,3))
        self.t_correction_fft_setup_s = time.perf_counter()-t_fft
        self.zero_mode = self.C_hat[:,0,0,0].copy()
        self.t_correction_parameter_setup_s = time.perf_counter()-start
        self.diagnostics = dict(
            delta=self.delta,active=bool(len(self.active_offsets)),
            negative_blocks=info.get('negative_blocks'),
            fraction_negative_blocks=info.get('fraction_negative_blocks'),
            active_correction_offsets=int(len(self.active_offsets)),
            correction_fro=info.get('correction_fro'),
            min_raw_eig=info.get('min_eig_before'),
            raw_inversion_max_component_defect=info.get('inversion_residual_before'),
            correction_fft_imag_max=float(np.abs(self.C_hat.imag).max()),
            t_correction_fft_setup_s=self.t_correction_fft_setup_s,
            t_correction_parameter_setup_s=self.t_correction_parameter_setup_s,
            correction_table_convention='physical C; C_hat=FFT(C); no h^-3 gather factor',
            correction_audit_evaluated=bool(audit))
        self.t_correction_parameter_setup_s = time.perf_counter()-start
        self.diagnostics['t_correction_parameter_setup_s'] = self.t_correction_parameter_setup_s
        t_audit = time.perf_counter()
        if audit:
            self.diagnostics.update(self.audit_table())
        self.diagnostics['t_correction_audit_s'] = time.perf_counter()-t_audit if audit else 0.

    def stencil(self,x):
        return self.mesh.stencil(x)

    def flat(self,x):
        return self.mesh.flat(x)

    def _positions(self,x):
        x = np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=3 or not len(x) or not np.all(np.isfinite(x)):
            raise ValueError('Require finite nonempty positions (N,3)')
        return x

    def _stencil_inputs(self,x_or_idx,wt=None):
        if wt is None:
            return self.flat(self._positions(x_or_idx))
        idx = np.asarray(x_or_idx)
        wt = np.asarray(wt,float)
        if idx.ndim!=2 or idx.shape!=wt.shape or not len(idx) or idx.dtype.kind not in 'iu':
            raise ValueError('Require matching nonempty integer idx and real wt arrays')
        if np.any(idx<0) or np.any(idx>=self.M**3) or not np.all(np.isfinite(wt)) or np.any(wt<0):
            raise ValueError('Require in-range indices and finite nonnegative weights')
        if np.max(np.abs(wt.sum(axis=1)-1.))>128*np.finfo(float).eps*max(1,wt.shape[1]):
            raise ValueError('Stencil weights must partition unity, up to floating roundoff')
        return idx,wt

    def _degree(self,idx,wt):
        t0 = time.perf_counter()
        density = np.bincount(idx.ravel(),wt.ravel(),self.M**3).reshape((self.M,)*3)
        density_hat = np.fft.rfftn(density)
        t_density = time.perf_counter()-t0
        t0 = time.perf_counter()
        D = np.empty((len(idx),3,3))
        for c,(a,b) in enumerate(PAIRS6):
            z = np.fft.irfftn(self.C_hat[c]*density_hat,s=(self.M,)*3,axes=(0,1,2))
            D[:,a,b] = D[:,b,a] = (z.ravel()[idx]*wt).sum(axis=1)
        return D,dict(t_bound_density_fft_s=t_density,
                      t_bound_degree_convolutions_s=time.perf_counter()-t0)

    def degree(self,x_or_idx,wt=None):
        """Correction-only D_i, including self; accept x or (idx,wt)."""
        idx,wt = self._stencil_inputs(x_or_idx,wt)
        return self._degree(idx,wt)[0]

    def bounds(self,x):
        """Return scalar bounds and N x 3 x 3 degree, never a 3N matrix."""
        t0 = time.perf_counter()
        x = self._positions(x)
        idx,wt = self.flat(x)
        t_stencil = time.perf_counter()-t0
        D,profile = self._degree(idx,wt)
        te = time.perf_counter()
        eig = np.linalg.eigvalsh(D)
        t_eigh = time.perf_counter()-te
        N = len(x)
        degree_max = float(eig[:,-1].max())
        B_simple = N*self.delta if N>=2 else 0.
        B_degree = 2*degree_max
        result = dict(self.diagnostics,N=N,B_simple=float(B_simple),B_degree=float(B_degree),
                      B=float(min(B_simple,B_degree)),singleton_exact_zero=bool(N==1),
                      degree_max_eigenvalue=degree_max,
                      degree_float_min_eigenvalue=float(eig[:,0].min()),
                      stencil_partition_max_defect=float(np.abs(wt.sum(axis=1)-1.).max()),
                      t_bound_stencil_s=t_stencil,t_bound_degree_eigh_s=t_eigh,**profile)
        result['t_bound_configuration_setup_s'] = time.perf_counter()-t0
        return result,D

    def apply(self,idx,wt,X):
        """Correction pair response H^C(X), using C only and physical units."""
        idx,wt = self._stencil_inputs(idx,wt)
        X = np.asarray(X,float).reshape(len(idx),3)
        src = [np.fft.rfftn(np.bincount(idx.ravel(),(wt*X[:,c,None]).ravel(),self.M**3)
                            .reshape((self.M,)*3)) for c in range(3)]
        phi = [np.fft.irfftn(sum(self.C_hat[PAIRS6.index((min(a,b),max(a,b)))]*src[b]
                                 for b in range(3)),s=(self.M,)*3,axes=(0,1,2)) for a in range(3)]
        return np.stack([(z.ravel()[idx]*wt).sum(axis=1) for z in phi],axis=1)

    def graph_action(self,idx,wt,X,D=None):
        idx,wt = self._stencil_inputs(idx,wt)
        X = np.asarray(X,float).reshape(len(idx),3)
        if D is None:
            D = self.degree(idx,wt)
        return np.einsum('nab,nb->na',D,X)-self.apply(idx,wt,X)

    def matvec(self,x,v,D=None):
        """Flattened correction graph action, for independent offline checks."""
        idx,wt = self.flat(self._positions(x))
        return self.graph_action(idx,wt,v,D).ravel()

    def pair_matrix(self,x,correction_chunk=32):
        """All C_ij from every nonzero offset; no full-operator subtraction."""
        x = self._positions(x)
        if int(correction_chunk)!=correction_chunk or correction_chunk<1:
            raise ValueError('Require a positive integer correction chunk')
        N = len(x)
        W = np.zeros((N,N,3,3))
        if not len(self.active_offsets):
            return W
        tri,corr,inv = ProjectedMesh._axis_correlations(self,x)
        out = np.zeros((6,len(tri[0])))
        for begin in range(0,len(self.active_offsets),int(correction_chunk)):
            end = min(begin+int(correction_chunk),len(self.active_offsets))
            weights = corr[0][inv[0][begin:end]].copy()
            weights *= corr[1][inv[1][begin:end]]
            weights *= corr[2][inv[2][begin:end]]
            out += self.active_values[begin:end].T @ weights
        for c,(a,b) in enumerate(PAIRS6):
            W[tri[0],tri[1],a,b] = W[tri[0],tri[1],b,a] = out[c]
            W[tri[1],tri[0],a,b] = W[tri[1],tri[0],b,a] = out[c]
        return W

    def direct_pair_matrix(self,x):
        """Independent literal double 3D stencil contraction, small N only."""
        x = self._positions(x)
        idx,wt = self.flat(x)
        grid = np.stack(np.unravel_index(idx,(self.M,)*3),axis=-1)
        W = np.empty((len(x),len(x),3,3))
        for i in range(len(x)):
            for j in range(len(x)):
                d = (grid[i,:,None,:]-grid[j,None,:,:])%self.M
                values = self.C[:,d[...,0],d[...,1],d[...,2]]
                comp = np.einsum('a,b,cab->c',wt[i],wt[j],values)
                W[i,j] = comp[COMPONENT_INDEX]
        return W

    def audit_table(self):
        """Float64 diagnostics, not interval-certified error margins."""
        full = np.moveaxis(self.C,0,-1)[...,COMPONENT_INDEX]
        eig = np.linalg.eigvalsh(full)
        neg = (-np.arange(self.M))%self.M
        reflected = self.C[:,neg][:,:,neg][:,:,:,neg]
        roundtrip = np.fft.irfftn(self.C_hat,s=(self.M,)*3,axes=(1,2,3))
        result = dict(correction_float_min_eigenvalue=float(eig.min()),
                      correction_float_max_eigenvalue=float(eig.max()),
                      correction_float_delta_excess=float(eig.max()-self.delta),
                      correction_inversion_max_component_defect=float(np.abs(self.C-reflected).max()),
                      correction_fft_roundtrip_max_component_defect=float(np.abs(roundtrip-self.C).max()))
        if hasattr(self.mesh,'original') and hasattr(self.mesh,'table_before'):
            raw = np.fft.irfftn(self.mesh.original.Gr,s=(self.M,)*3,axes=(1,2,3))/self.h**3
            difference = raw-self.mesh.table_before
            result['raw_to_sym_max_component_defect'] = float(np.abs(difference).max())
            result['raw_to_sym_max_block_fro_defect'] = float(np.sqrt(
                np.sum(np.array([1,1,1,2,2,2])[:,None,None,None]*difference**2,axis=0)).max())
            influence_difference = (self.mesh.Gr-self.mesh.original.Gr)/self.h**3
            result['stored_influence_minus_C_hat_max_defect'] = float(np.abs(influence_difference-self.C_hat).max())
        return result


__all__ = ['CorrectionBounds']
