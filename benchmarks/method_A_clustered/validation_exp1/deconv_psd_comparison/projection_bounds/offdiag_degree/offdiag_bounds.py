"""Self-excluded correction degree bounds; no production operator is changed.

Normal path: tensor-stencil autocorrelation -> C_ii, then Dtotal-C_ii.
Stable fallback: positive local grid occupancy sums of OTHER particles, with
prefix/suffix addition and no subtraction of nearly equal densities/degrees.
"""
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from bounds import CorrectionBounds, np, PAIRS6, COMPONENT_INDEX  # noqa: E402


def _ratio(numerator,denominator):
    if denominator==0:
        return None
    value = float(numerator/denominator)
    return value if np.isfinite(value) else None


class OffdiagBounds:
    def __init__(self,correction_bounds,cancellation_switch_ratio=1e8,max_stable_queries=50_000_000):
        start = time.perf_counter()
        if not isinstance(correction_bounds,CorrectionBounds):
            raise TypeError('Require an existing CorrectionBounds object')
        if not np.isfinite(cancellation_switch_ratio) or cancellation_switch_ratio<=1:
            raise ValueError('Require a finite cancellation switch ratio >1')
        if max_stable_queries is not None and max_stable_queries<1:
            raise ValueError('Require max_stable_queries>=1 or None')
        self.cb = correction_bounds
        self.mesh = correction_bounds.mesh
        self.M,self.p = correction_bounds.M,correction_bounds.p
        self.cancellation_switch_ratio = float(cancellation_switch_ratio)
        self.max_stable_queries = max_stable_queries
        self.t_offdiag_parameter_setup_s = time.perf_counter()-start
        self.last_D_total = self.last_C_self = self.last_D_off_raw = None
        self.last_D_off_stable = self.last_D_off = None
        self.last_self_profile = {}

    def self_tensors(self,x):
        """C_ii using O(N*p^2+N*min(M,2p-1)^3) autocorrelation support.

        Periodic offsets are accumulated with add.at: wrap aliases and repeated
        stencil indices are combined, never assumed to be distinct.
        """
        started = time.perf_counter()
        x = self.cb._positions(x)
        g,w = self.mesh.stencil(x)
        N = len(x)
        t_axis = time.perf_counter()
        difference = (g[:,:,:,None]-g[:,:,None,:])%self.M
        product = w[:,:,:,None]*w[:,:,None,:]
        autocorr = np.zeros((N,3,self.M))
        np.add.at(autocorr,(np.arange(N)[:,None,None,None],np.arange(3)[None,:,None,None],difference),product)
        t_axis = time.perf_counter()-t_axis
        t_contract = time.perf_counter()
        support = [np.flatnonzero(np.any(autocorr[:,a,:]!=0.,axis=0)) for a in range(3)]
        offsets = np.stack(np.meshgrid(*support,indexing='ij'),axis=-1).reshape(-1,3)
        # Visit only the actual autocorrelation support, independently of how
        # many nonzero corrections the full M^3 table contains.
        values = self.cb.C[(slice(None),)+tuple(offsets.T)].T
        nonzero = np.any(values!=0.,axis=1)
        support_offsets_visited = len(offsets)
        offsets,values = offsets[nonzero],values[nonzero]
        comp = np.zeros((N,6))
        for begin in range(0,len(offsets),256):
            d = offsets[begin:begin+256]
            weights = autocorr[:,0,d[:,0]]*autocorr[:,1,d[:,1]]*autocorr[:,2,d[:,2]]
            comp += weights @ values[begin:begin+256]
        result = comp[:,COMPONENT_INDEX]
        self.last_self_profile = dict(t_self_axis_autocorrelation_s=t_axis,
            t_self_tensor_contraction_s=time.perf_counter()-t_contract,
            self_support_correction_offsets=int(len(offsets)),
            self_autocorrelation_offsets_visited=int(support_offsets_visited),
            self_autocorrelation_partition_max_defect=float(np.abs(autocorr.sum(axis=2)-1.).max()),
            t_self_autocorrelation_s=time.perf_counter()-started)
        return result

    def _occupancy(self,idx,wt):
        """Positive cell+owner aggregation and subtraction-free excluded mass."""
        N = len(idx)
        owners = np.repeat(np.arange(N),idx.shape[1])
        flat_idx,flat_wt = idx.ravel(),wt.ravel()
        nonzero = flat_wt!=0.
        keys = flat_idx[nonzero].astype(np.int64)*N+owners[nonzero]
        data = flat_wt[nonzero]
        order = np.argsort(keys,kind='stable')
        keys,data = keys[order],data[order]
        starts = np.r_[0,np.flatnonzero(np.diff(keys))+1]
        data = np.add.reduceat(data,starts)
        keys = keys[starts]
        cells,owners = keys//N,keys%N
        group_starts = np.r_[0,np.flatnonzero(np.diff(cells))+1]
        group_ends = np.r_[group_starts[1:],len(cells)]
        unique_cells = cells[group_starts]
        totals = data[group_starts].copy()
        excluded = np.zeros_like(data)
        for group in np.flatnonzero(group_ends-group_starts>1):
            lo,hi = group_starts[group],group_ends[group]
            block = data[lo:hi]
            prefix = np.cumsum(block)
            suffix = np.cumsum(block[::-1])[::-1]
            totals[group] = prefix[-1]
            excluded[lo:hi] = np.r_[0.,prefix[:-1]]+np.r_[suffix[1:],0.]
        return keys,cells,owners,data,unique_cells,totals,excluded

    def stable_off_degree(self,x):
        """Direct other-particle mass on local grid cells, no N^2 matrix.

        Visits Kactive times the nonzero tensor-stencil entries.  No C offset
        or small positive weight is discarded; every sum defining other mass
        uses only other owners, so a dominant self mass is never subtracted.
        """
        started = time.perf_counter()
        x = self.cb._positions(x)
        idx,wt = self.mesh.flat(x)
        N = len(x)
        t_occupancy = time.perf_counter()
        keys,cells,owners,mass,unique_cells,totals,excluded = self._occupancy(idx,wt)
        t_occupancy = time.perf_counter()-t_occupancy
        xyz = np.stack(np.unravel_index(cells,(self.M,)*3),axis=1)
        components = np.zeros((N,6))
        nonzero_coefficients = 0
        positive_weight_sum = 0.
        t_contraction = time.perf_counter()
        for offset,value in zip(self.cb.active_offsets,self.cb.active_values):
            source_xyz = (xyz-offset)%self.M
            source_cells = (source_xyz[:,0]*self.M+source_xyz[:,1])*self.M+source_xyz[:,2]
            cell_pos = np.searchsorted(unique_cells,source_cells)
            exists = cell_pos<len(unique_cells)
            exists[exists] &= unique_cells[cell_pos[exists]]==source_cells[exists]
            other_mass = np.zeros(len(cells))
            other_mass[exists] = totals[cell_pos[exists]]
            owner_keys = source_cells*N+owners
            owner_pos = np.searchsorted(keys,owner_keys)
            same_owner_exists = owner_pos<len(keys)
            same_owner_exists[same_owner_exists] &= keys[owner_pos[same_owner_exists]]==owner_keys[same_owner_exists]
            other_mass[same_owner_exists] = excluded[owner_pos[same_owner_exists]]
            coefficients = np.bincount(owners,mass*other_mass,N)
            nonzero_coefficients += int(np.count_nonzero(coefficients))
            positive_weight_sum += float(coefficients.sum())
            components += coefficients[:,None]*value[None,:]
        D = components[:,COMPONENT_INDEX]
        profile = dict(t_stable_occupancy_s=t_occupancy,
            t_stable_positive_contraction_s=time.perf_counter()-t_contraction,
            t_stable_offdiag_setup_s=time.perf_counter()-started,
            stable_occupied_grid_cells=int(len(unique_cells)),stable_cell_owner_entries=int(len(keys)),
            stable_queries=int(len(keys)*len(self.cb.active_offsets)),
            stable_nonzero_mass_coefficients=nonzero_coefficients,
            stable_positive_mass_coefficient_sum=positive_weight_sum,
            stable_true_zero_from_occupancy=bool(nonzero_coefficients==0),
            stable_accumulation='positive other-owner prefix/suffix; no total-minus-self')
        return D,profile

    def bounds(self,x,stable='auto',D_total=None,total_result=None):
        """Return (scalars,Doff); retained tensors are available as last_* attrs.

        stable='never' exposes raw subtraction, 'always' forces the positive
        occupancy path, and 'auto' switches at the stated diagnostic criterion.
        The switch ratio is an algorithm choice, never an error floor or clip.
        """
        wall_start = time.perf_counter()
        if stable not in ('auto','always','never'):
            raise ValueError('stable must be auto, always or never')
        x = self.cb._positions(x)
        N = len(x)
        t_total = time.perf_counter()
        if D_total is None:
            old,D_total = self.cb.bounds(x)
        else:
            D_total = np.asarray(D_total,float)
            if D_total.shape!=(N,3,3) or not np.all(np.isfinite(D_total)):
                raise ValueError('Require finite D_total of shape (N,3,3)')
            old = dict(total_result or self.cb.diagnostics)
        existing_total_setup = time.perf_counter()-t_total
        incremental_start = time.perf_counter()
        C_self = self.self_tensors(x)
        raw = D_total-C_self
        total_eig = np.linalg.eigvalsh(D_total)
        self_eig = np.linalg.eigvalsh(C_self)
        raw_eig = np.linalg.eigvalsh(raw)
        total_max,self_max = float(total_eig[:,-1].max()),float(self_eig[:,-1].max())
        raw_norm = float(np.abs(raw_eig).max())
        subtraction_scale = float(np.abs(total_eig).max()+np.abs(self_eig).max())
        cancellation_index = _ratio(subtraction_scale,raw_norm)
        ratio_overflow = bool(raw_norm>0 and subtraction_scale>0 and cancellation_index is None)
        raw_zero_with_self = bool(raw_norm==0 and subtraction_scale>0)
        estimate = np.finfo(float).eps*(8+np.ceil(np.log2(self.M**3))+3*self.p)*subtraction_scale
        significant_negative = bool(float(raw_eig.min()) < -estimate)
        cancellation_unreliable = bool(subtraction_scale>0 and (
            raw_zero_with_self or ratio_overflow or (cancellation_index is not None and cancellation_index>self.cancellation_switch_ratio)
            or significant_negative))
        reasons = []
        if raw_zero_with_self:reasons.append('raw_zero_with_nonzero_self')
        if ratio_overflow:reasons.append('cancellation_ratio_overflow')
        if cancellation_index is not None and cancellation_index>self.cancellation_switch_ratio:
            reasons.append('cancellation_ratio_exceeds_switch')
        if significant_negative:reasons.append('raw_negative_exceeds_heuristic_roundoff')
        if stable=='always':reasons.append('explicit_always')
        need_stable = stable=='always' or (stable=='auto' and cancellation_unreliable)
        queries_estimate = int(N*self.p**3*len(self.cb.active_offsets))
        stable_unavailable = bool(need_stable and stable!='always' and self.max_stable_queries is not None
                                  and queries_estimate>self.max_stable_queries)
        stable_D = None
        profile = dict(t_stable_offdiag_setup_s=0.,stable_queries_estimate=queries_estimate)
        fallback_used = False
        method = 'total_minus_autocorrelation_self'
        if N==1:
            D_off = np.zeros_like(raw)
            method = 'singleton_exact_no_other_particles'
            profile['stable_true_zero_from_occupancy'] = True
            stable_unavailable = False
        elif need_stable and not stable_unavailable:
            stable_D,stable_profile = self.stable_off_degree(x)
            profile.update(stable_profile)
            D_off = stable_D
            fallback_used = True
            method = 'positive_local_occupancy'
        else:
            D_off = raw
            if stable_unavailable:reasons.append('stable_query_budget_exceeded')
        chosen_eig = np.linalg.eigvalsh(D_off)
        off_max = float(chosen_eig[:,-1].max())
        B_simple = N*self.cb.delta if N>=2 else 0.
        B_total,B_off_raw,B_off = 2*total_max,2*float(raw_eig[:,-1].max()),2*off_max
        B_old = min(B_simple,B_total)
        numerical_bound_available = bool(N==1 or fallback_used or not cancellation_unreliable)
        trace_total,trace_self = float(np.trace(D_total,axis1=1,axis2=2).sum()),float(np.trace(C_self,axis1=1,axis2=2).sum())
        result = dict(old,N=N,delta=self.cb.delta,B_simple=float(B_simple),B_total=float(B_total),
            B_old=float(B_old),B_off_raw=float(B_off_raw),
            B_improved_raw=float(min(B_simple,B_total,B_off_raw)),
            B_off=float(B_off) if numerical_bound_available else None,
            B_improved=float(min(B_simple,B_total,B_off)) if numerical_bound_available else None,
            B_improved_equivalent=float(min(B_simple,B_off)) if numerical_bound_available else None,
            numerical_bound_available=numerical_bound_available,
            self_degree_trace_fraction=_ratio(trace_self,trace_total),
            self_degree_trace_sum=trace_self,total_degree_trace_sum=trace_total,
            degreeoff_total_ratio=_ratio(off_max,total_max),
            total_max_eigenvalue=total_max,self_max_eigenvalue=self_max,offdiag_max_eigenvalue=off_max,
            raw_offdiag_min_eigenvalue=float(raw_eig.min()),offdiag_min_eigenvalue=float(chosen_eig.min()),
            raw_offdiag_symmetry_defect=float(np.abs(raw-raw.swapaxes(1,2)).max()),
            offdiag_symmetry_defect=float(np.abs(D_off-D_off.swapaxes(1,2)).max()),
            cancellation_index_raw=cancellation_index,cancellation_switch_ratio=self.cancellation_switch_ratio,
            cancellation_ratio_overflow=ratio_overflow,
            raw_zero_with_nonzero_self=raw_zero_with_self,
            cancellation_unreliable_raw=cancellation_unreliable,
            raw_roundoff_scale_heuristic=float(estimate),
            raw_relative_roundoff_indicator=_ratio(estimate,raw_norm),
            stable_fallback_used=fallback_used,stable_fallback_unavailable=stable_unavailable,
            stable_switch_reason=';'.join(reasons),degreeoff_method=method,
            bound_float_interval_certified=False,
            t_offdiag_parameter_setup_s=self.t_offdiag_parameter_setup_s,
            t_existing_total_bound_setup_s=existing_total_setup,
            **self.last_self_profile,**profile)
        if stable_D is not None:
            difference_eig = np.linalg.eigvalsh(stable_D-raw)
            result['raw_stable_offdiag_max_spectral_difference'] = float(np.abs(difference_eig).max())
            result['raw_stable_offdiag_relative_difference'] = _ratio(
                float(np.abs(difference_eig).max()),float(np.abs(chosen_eig).max()))
        else:
            result['raw_stable_offdiag_max_spectral_difference'] = None
            result['raw_stable_offdiag_relative_difference'] = None
        result['t_offdiag_setup_s'] = time.perf_counter()-incremental_start
        result['t_offdiag_with_total_setup_s'] = time.perf_counter()-wall_start
        self.last_D_total,self.last_C_self,self.last_D_off_raw = D_total,C_self,raw
        self.last_D_off_stable,self.last_D_off = stable_D,D_off
        return result,D_off


__all__ = ['OffdiagBounds']
