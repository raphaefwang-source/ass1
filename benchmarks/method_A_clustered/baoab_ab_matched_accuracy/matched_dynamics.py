"""True moving BAOAB with complete repeated A OU substeps on one frozen graph.

The PR10 model, graph construction and PR8/9 OU implementations are imported
unchanged. Only parameter-only FFT tables persist between outer steps. Timing
contains real q-dependent setup once and every inner RNG/OU call. Additional
noise hashes are finalized after the outer second kick, outside compute wall.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import hashlib
import json
import sys
from time import perf_counter
import numpy as np

HERE=Path(__file__).resolve().parent
LEGACY=HERE.parent/'baoab_ab_pilot'
if str(LEGACY) not in sys.path:sys.path.insert(0,str(LEGACY))
from dynamics import PilotSystem, FFTMeter
from model import MASS,TEMPERATURE,array_sha,rng_sha,lj_force
from ou_core import ou_a,ou_b

ALLOWED_SUBSTEPS=(1,2,4,8,16,32,64)


def _registered_k(method,k):
    if method not in ('A','B'):raise ValueError('method must be A or B')
    if isinstance(k,bool) or int(k)!=k or k not in ALLOWED_SUBSTEPS:
        raise ValueError('k must be one of 1,2,4,8,16,32,64')
    if method=='B' and k!=1:raise ValueError('B remains one full-dt OU call')
    return int(k)


def _sha_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class MatchedSystem(PilotSystem):
    """One graph per midpoint, k complete symmetric A substeps or one B step."""
    def __init__(self,params,L=None,*,retain_noise=False):
        super().__init__(params,L=L)
        self.retain_noise=bool(retain_noise)
        self.last_substep_rows=[]
        self.last_gaussian_hashes=[]
        self.last_noise_arrays=[]
        self._pending_noise=[]

    def build(self,q,method='A',dt=.005,k=1):
        k=_registered_k(method,k)
        # Legacy build prepares exactly one coefficient set, at half its dt.
        # Supplying the inner dt avoids creating an unused full-step cache key.
        built=super().build(q,method,dt/k if method=='A' else dt)
        built.metrics.update(setup_substeps=k,outer_dt=float(dt),
            coefficient_tau=float(dt/(2*k)) if method=='A' else 0.,
            graph_builds_this_step=1,
            edge_eigh_builds_this_step=int(method=='A'),
            coefficient_sets_this_step=int(method=='A'))
        return built

    def _finalize_noise_evidence(self):
        hashes=[]
        for row,draw in zip(self.last_substep_rows,self._pending_noise):
            values=dict(substep=row['substep'],z_sha256=array_sha(draw['z']),
                eta_forward_sha256=array_sha(draw['ef']) if draw['ef'] is not None else '',
                eta_reverse_sha256=array_sha(draw['er']) if draw['er'] is not None else '')
            row.update(values);hashes.append(values)
        self.last_gaussian_hashes=hashes
        self.last_noise_arrays=self._pending_noise if self.retain_noise else []
        self._pending_noise=[]
        return _sha_json(hashes)

    def frozen_ou(self,built,p,method,r_d,r_n,rng,dt=.005,k=1,*,
                  noise_draws=None,finalize_evidence=True):
        """Advance p at one frozen q using the actual production OU loop.

        Returns ``(p_after, aggregate_diagnostics)``. ``noise_draws``, if
        supplied, is a list of k dictionaries with z/ef/er for A (only z for B)
        and consumes no RNG. A production caller supplies an actual Generator.
        Gaussian shape/order is exactly PR10 for k=1. ``last_substep_rows``
        preserves every inner call's real clocks, counts, Ritz and draw SHA.
        """
        k=_registered_k(method,k)
        if built.metrics.get('setup_substeps',k)!=k:
            raise ValueError('Frozen graph coefficients were prepared for another k')
        if noise_draws is not None and len(noise_draws)!=k:
            raise ValueError('Require exactly one independent draw dictionary per substep')
        if noise_draws is None and rng is None:raise ValueError('Need RNG or supplied draws')
        current=np.asarray(p,dtype=float)
        self.last_substep_rows=[];self.last_gaussian_hashes=[];self.last_noise_arrays=[];self._pending_noise=[]
        prefix='long' if method=='A' else 'full'
        inner_dt=float(dt)/k
        totals={name:0. for name in ('t_rng_s','t_pairOU_forward','t_pairOU_reverse',
            't_reorth','t_ritz_eigh','step_wall_s','t_fft_s')}
        totals.update({f't_{prefix}_lanczos_damp':0.,f't_{prefix}_lanczos_noise':0.,
            'matvec_count':0,'damp_matvec_count':0,'noise_matvec_count':0,
            'negative_ritz_count':0,'clipped_negative_count':0,'ou_fft_calls':0})
        ranks_d=[];ranks_n=[];ritz_d=[];ritz_n=[];prepare_calls=[]
        original_prepare=None
        if method=='A':
            if built.edges.backend!='compiled':raise RuntimeError('Matched A requires compiled Pair OU')
            original_prepare=built.edges.prepare
            def observed_prepare(tau,mass=1.,theta=1.):
                info=original_prepare(tau,mass,theta)
                prepare_calls.append(info)
                return info
            built.edges.prepare=observed_prepare
        loop_start=perf_counter()
        try:
            for index in range(k):
                state_before=rng_sha(rng) if noise_draws is None else ''
                t=perf_counter()
                if noise_draws is None:
                    z=rng.standard_normal(current.shape)
                    ef=er=None
                    if method=='A':
                        ef=rng.standard_normal((len(built.edges.i),3))
                        er=rng.standard_normal((len(built.edges.i),3))
                else:
                    draw=noise_draws[index]
                    z=np.asarray(draw['z'],dtype=float)
                    ef=np.asarray(draw['ef'],dtype=float) if method=='A' else None
                    er=np.asarray(draw['er'],dtype=float) if method=='A' else None
                t_rng=perf_counter()-t if noise_draws is None else 0.
                if z.shape!=current.shape or not np.isfinite(z).all():raise ValueError('Invalid long/full Gaussian')
                if method=='A' and (ef.shape!=(len(built.edges.i),3) or er.shape!=ef.shape
                        or not np.isfinite(ef).all() or not np.isfinite(er).all()):
                    raise ValueError('Invalid independent near Gaussian arrays')
                with FFTMeter() as meter:
                    if method=='A':
                        current,diag=ou_a(built.long_action,current,z,ef,er,built.edges,
                            inner_dt,int(r_d),int(r_n),MASS,TEMPERATURE)
                    else:
                        current,diag=ou_b(built.full_action,current,z,float(dt),
                            int(r_d),int(r_n),MASS,TEMPERATURE)
                row=dict(substep=index+1,substeps=k,method=method,dt_outer=float(dt),
                    dt_inner=inner_dt,coefficient_tau=inner_dt/2 if method=='A' else 0.,
                    q_mid_sha256=built.token,r_d=int(r_d),r_n=int(r_n),
                    r_d_used=int(diag['damp_rank_used']),r_n_used=int(diag['noise_rank_used']),
                    t_rng_s=t_rng,t_ou_wall_s=diag['step_wall_s'],
                    t_pair_forward_s=diag.get('t_pairOU_forward',0.),
                    t_pair_reverse_s=diag.get('t_pairOU_reverse',0.),
                    t_lanczos_damp_s=diag[f't_{prefix}_lanczos_damp'],
                    t_lanczos_noise_s=diag[f't_{prefix}_lanczos_noise'],
                    t_reorth_s=diag['t_reorth'],t_ritz_eigh_s=diag['t_ritz_eigh'],
                    t_fft_s=meter.elapsed,fft_calls=meter.calls,
                    matvec_count=int(diag['matvec_count']),
                    damp_matvec_count=int(diag['damp_matvec_count']),
                    noise_matvec_count=int(diag['noise_matvec_count']),
                    negative_ritz_count=int(diag['negative_ritz_count']),
                    clipped_negative_count=int(diag['clipped_negative_count']),
                    ritz_min_damp=diag['damping']['ritz_min_raw'],
                    ritz_min_noise=diag['noise']['ritz_min_raw'],
                    rng_state_before_sha256=state_before,
                    rng_state_after_sha256=rng_sha(rng) if noise_draws is None else '',
                    noise_source='PCG64 generator' if noise_draws is None else 'provided independent draws')
                self.last_substep_rows.append(row)
                self._pending_noise.append(dict(z=z,ef=ef,er=er))
                totals['t_rng_s']+=t_rng
                for key in ('t_pairOU_forward','t_pairOU_reverse','t_reorth','t_ritz_eigh','step_wall_s',
                            'matvec_count','damp_matvec_count','noise_matvec_count',
                            'negative_ritz_count','clipped_negative_count'):
                    totals[key]+=diag.get(key,0)
                totals[f't_{prefix}_lanczos_damp']+=diag[f't_{prefix}_lanczos_damp']
                totals[f't_{prefix}_lanczos_noise']+=diag[f't_{prefix}_lanczos_noise']
                totals['t_fft_s']+=meter.elapsed;totals['ou_fft_calls']+=meter.calls
                ranks_d.append(diag['damp_rank_used']);ranks_n.append(diag['noise_rank_used'])
                if diag['damping']['ritz_min_raw'] is not None:ritz_d.append(diag['damping']['ritz_min_raw'])
                if diag['noise']['ritz_min_raw'] is not None:ritz_n.append(diag['noise']['ritz_min_raw'])
        finally:
            if original_prepare is not None:del built.edges.prepare
        totals.update(t_ou_loop_s=perf_counter()-loop_start,substeps=k,ou_calls=k,
            long_ou_calls=k if method=='A' else 0,full_ou_calls=int(method=='B'),
            damp_rank_used=max(ranks_d),noise_rank_used=max(ranks_n),
            damp_rank_used_min=min(ranks_d),noise_rank_used_min=min(ranks_n),
            ritz_min_damp=min(ritz_d) if ritz_d else None,
            ritz_min_noise=min(ritz_n) if ritz_n else None,
            coefficient_prepare_calls=len(prepare_calls),
            coefficient_cache_hits=sum(int(d['coefficient_cache_hit']) for d in prepare_calls),
            coefficient_cache_size=built.edges.coefficient_cache_size if method=='A' else 0)
        if method=='A' and (len(prepare_calls)!=2*k or
                any(d['tau']!=inner_dt/2 or not d['coefficient_cache_hit'] for d in prepare_calls)):
            raise RuntimeError('A inner coefficient key/hit contract failed')
        totals['t_ou_loop_overhead_s']=totals['t_ou_loop_s']-totals['t_rng_s']-totals['step_wall_s']
        if finalize_evidence:totals['gaussian_substeps_sha256']=self._finalize_noise_evidence()
        return current,totals

    def step(self,q_unwrapped,p,method,r_d,r_n,rng,dt=.005,k=1):
        """One outer BAOAB step, with q frozen only inside its complete O loop."""
        k=_registered_k(method,k)
        if not np.isfinite(dt) or dt<=0:raise ValueError('dt must be positive')
        q0=np.asarray(q_unwrapped,dtype=float);p0=np.asarray(p,dtype=float)
        rng_before=rng_sha(rng)
        start=perf_counter();t=perf_counter()
        force0,_,_=lj_force(np.mod(q0,self.L),self.L);t_force0=perf_counter()-t
        ph=p0+.5*dt*force0
        qm=q0+.5*dt*ph/MASS
        built=self.build(np.mod(qm,self.L),method,dt,k)
        po,diag=self.frozen_ou(built,ph,method,r_d,r_n,rng,dt,k,finalize_evidence=False)
        q1=qm+.5*dt*po/MASS
        t=perf_counter();force1,potential,minimum=lj_force(np.mod(q1,self.L),self.L);t_force1=perf_counter()-t
        p1=po+.5*dt*force1
        t_baoab=perf_counter()-start
        diagnostic_start=perf_counter()
        if not np.isfinite(q1).all() or not np.isfinite(p1).all():raise FloatingPointError('Nonfinite moving BAOAB state')
        gaussian_hash=self._finalize_noise_evidence()
        prefix='long' if method=='A' else 'full'
        row=dict(built.metrics,method=method,dt=float(dt),k=k,substeps=k,
            r_d=int(r_d),r_n=int(r_n),r_d_used=diag['damp_rank_used'],r_n_used=diag['noise_rank_used'],
            r_d_used_min=diag['damp_rank_used_min'],r_n_used_min=diag['noise_rank_used_min'],
            t_LJ_force_before_s=t_force0,t_LJ_force_after_s=t_force1,t_LJ_forces_s=t_force0+t_force1,
            t_rng_s=diag['t_rng_s'],t_pair_forward_s=diag['t_pairOU_forward'],
            t_pair_reverse_s=diag['t_pairOU_reverse'],
            t_lanczos_damp_s=diag[f't_{prefix}_lanczos_damp'],
            t_lanczos_noise_s=diag[f't_{prefix}_lanczos_noise'],
            t_reorth_s=diag['t_reorth'],t_ritz_eigh_s=diag['t_ritz_eigh'],
            t_ou_wall_s=diag['step_wall_s'],t_ou_loop_s=diag['t_ou_loop_s'],
            t_ou_loop_overhead_s=diag['t_ou_loop_overhead_s'],
            t_ou_fft_s=diag['t_fft_s'],t_fft_s=diag['t_fft_s']+built.metrics['t_setup_fft_s'],
            ou_fft_calls=diag['ou_fft_calls'],ou_calls=diag['ou_calls'],
            long_ou_calls=diag['long_ou_calls'],full_ou_calls=diag['full_ou_calls'],
            coefficient_prepare_calls=diag['coefficient_prepare_calls'],
            coefficient_cache_hits=diag['coefficient_cache_hits'],coefficient_cache_size=diag['coefficient_cache_size'],
            t_baoab_s=t_baoab,t_baoab_wall_s=t_baoab,
            negative_ritz_count=diag['negative_ritz_count'],
            clipped_roundoff_ritz_count=diag['clipped_negative_count'],
            ritz_min_damp=diag['ritz_min_damp'],ritz_min_noise=diag['ritz_min_noise'],
            matvec_count=diag['matvec_count'],damp_matvec_count=diag['damp_matvec_count'],
            noise_matvec_count=diag['noise_matvec_count'],
            Tkin=float(np.sum((p1-p1.mean(axis=0))**2)/(3*len(p1)-3)),
            U_per_particle=potential/len(p1),min_pair_distance=minimum,
            momentum_x=float(p1[:,0].sum()),momentum_y=float(p1[:,1].sum()),momentum_z=float(p1[:,2].sum()),
            momentum_norm=float(np.linalg.norm(p1.sum(axis=0))),
            q_motion_rms=float(np.sqrt(np.mean((q1-q0)**2))),
            q_mid_sha256=built.token,q_before_sha256=array_sha(q0),q_after_sha256=array_sha(q1),
            p_before_sha256=array_sha(p0),p_after_sha256=array_sha(p1),
            rng_state_before_sha256=rng_before,rng_state_after_sha256=rng_sha(rng),
            gaussian_substeps_sha256=gaussian_hash,
            q_dependent_setup_rebuilt=True,table_sha256=self.table_sha256)
        row['t_baoab_other_s']=t_baoab-(row['t_dynamic_setup_s']+row['t_LJ_forces_s']+row['t_rng_s']+row['t_ou_wall_s'])
        row['t_evidence_diagnostics_s']=perf_counter()-diagnostic_start
        return q1,p1,row
