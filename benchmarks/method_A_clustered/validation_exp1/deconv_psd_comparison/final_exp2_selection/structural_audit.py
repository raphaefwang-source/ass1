"""Independent small-N structural diagnostics; run AFTER formal timing ends.

This audit never screens search candidates or changes a physical method. It
checks literal stencil contractions against FFT actions, including every self
response and the real zero mode, on separate odd/even projected mesh objects.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import protocol as P

np=P.np


def main():
    started=time.perf_counter();records=[];issues=[]
    rng=np.random.default_rng(6447774395)
    base=P.configurations()[0]
    with P.S.threadpool_limits(1):
        for M,xi,sF in ((15,1.025,3.),(16,1.25,2.8)):
            par=P.S.parameters(M=M,p=5,xi=xi,sR=2.85,sF=sF)
            q=base['q'][:6].copy();N=len(q)
            cfg=dict(base,q=q,N=N)
            mesh=P.S.table('PPPM-grid-PSD',base['L'],par)
            second=P.S.table('PPPM-grid-PSD',base['L'],par)
            fresh=bool(mesh is not second and mesh.original is not second.original
                       and mesh.M==M and mesh.p==5 and mesh.original.M==M and mesh.original.p==5)
            del second
            idx,wt=mesh.flat(q)
            literal=mesh.direct_pair_matrix(q)
            sparse=P.S.long_dense('PPPM-grid-PSD',mesh,q)
            lap_literal=P.S.lap_full(literal)
            literal_pair=literal.transpose(0,2,1,3).reshape(3*N,3*N)
            fft_pair=np.empty_like(literal_pair)
            for c in range(3*N):
                vector=np.zeros((N,3));vector.ravel()[c]=1.
                fft_pair[:,c]=mesh.apply(idx,wt,vector).ravel()
            degree=mesh.degree(idx,wt)
            expected_degree=literal.sum(axis=1)
            roundtrip=np.fft.irfftn(mesh.Gr,s=(M,)*3,axes=(1,2,3))/mesh.h**3
            expected_zero=mesh.table.sum(axis=(1,2,3))*mesh.h**3
            zero_shift=(mesh.delta_table.sum(axis=(1,2,3))*mesh.h**3)
            op=P.S.build('PPPM-grid-PSD',cfg,par,mesh)
            full=P.E.real_dense(op)+lap_literal
            basis=np.empty_like(full)
            for c in range(3*N):
                vector=np.zeros((N,3));vector.ravel()[c]=1.
                basis[:,c]=op.apply(vector).ravel()
            table_eigs=np.linalg.eigvalsh(np.moveaxis(mesh.table,0,-1)[...,np.array([[0,3,4],[3,1,5],[4,5,2]])])
            scale=max(1.,float(np.linalg.norm(full,2)))
            translations=[]
            for axis in range(3):
                vector=np.zeros((N,3));vector[:,axis]=1.
                translations.append(float(np.linalg.norm(op.apply(vector))/scale))
            energies=[]
            for _ in range(32):
                vector=rng.normal(size=(N,3));vector-=vector.mean(axis=0)
                energies.append(float(vector.ravel()@(full@vector.ravel())/np.sum(vector**2)))
            singleton=dict(base,q=q[:1].copy(),N=1)
            one=P.S.build('PPPM-grid-PSD',singleton,par,mesh)
            one_vector=rng.normal(size=(1,3))
            singleton_action=float(np.linalg.norm(one.apply(one_vector)))
            one_idx,one_wt=mesh.flat(q[:1])
            singleton_response=float(np.linalg.norm(mesh.apply(one_idx,one_wt,one_vector)))
            result=dict(M=M,parity='odd' if M%2 else 'even',N=N,xi=xi,sF=sF,p=5,
                freshly_constructed_own_projection=fresh,
                active_blocks=int(mesh.diagnostics['negative_blocks']),
                stencil_partition_defect=float(np.abs(wt.sum(axis=1)-1).max()),stencil_min_weight=float(wt.min()),
                physical_ifft_table_max_defect=float(np.max(np.abs(roundtrip-mesh.table))),
                fft_pair_vs_literal_max_defect=float(np.max(np.abs(fft_pair-literal_pair))),
                sparse_laplacian_vs_literal_max_defect=float(np.max(np.abs(sparse-lap_literal))),
                full_degree_vs_including_self_literal_max_defect=float(np.max(np.abs(degree-expected_degree))),
                full_action_vs_literal_max_defect=float(np.max(np.abs(basis-full))),
                zero_mode_max_defect=float(np.max(np.abs(mesh.Gr[:,0,0,0]-expected_zero))),
                zero_mode_projection_shift_max_defect=float(np.max(np.abs(mesh.Gr[:,0,0,0]-mesh.original.Gr[:,0,0,0]-zero_shift))),
                retained_zero_mode_norm=float(np.linalg.norm(mesh.Gr[:,0,0,0])),
                singleton_self_response_norm=singleton_response,singleton_full_self_cancellation_residual=singleton_action,
                translation_residual_max=max(translations),symmetry_residual=float(np.linalg.norm(full-full.T)/scale),
                min_projected_table_eigenvalue=float(table_eigs.min()),
                min_full_graph_eigenvalue=float(np.linalg.eigvalsh(full).min()),
                min_centered_energy_rayleigh=min(energies),norm_full=scale)
            checks=dict(fresh_projection=fresh,nonnegative_stencil=result['stencil_min_weight']>=0,
                partition=result['stencil_partition_defect']<2e-14,
                normalization=result['physical_ifft_table_max_defect']<2e-12,
                literal_fft=result['fft_pair_vs_literal_max_defect']<2e-11,
                literal_sparse=result['sparse_laplacian_vs_literal_max_defect']<2e-11,
                complete_degree=result['full_degree_vs_including_self_literal_max_defect']<2e-11,
                complete_action=result['full_action_vs_literal_max_defect']<2e-11,
                actual_zero_mode=result['retained_zero_mode_norm']>0 and result['zero_mode_max_defect']<2e-10,
                zero_shift=result['zero_mode_projection_shift_max_defect']<2e-10,
                complete_self=singleton_response>0 and singleton_action<2e-11,
                translations=result['translation_residual_max']<2e-12,
                symmetry=result['symmetry_residual']<2e-12,
                table_psd=result['min_projected_table_eigenvalue']>=-2e-12,
                graph_psd=result['min_full_graph_eigenvalue']>=-2e-11*scale,
                sampled_psd_energy=result['min_centered_energy_rayleigh']>=-2e-11*scale)
            result['checks']=checks;records.append(result)
            issues.extend(dict(M=M,check=key) for key,value in checks.items() if not value)
    output=dict(ok=not issues,status='pass' if not issues else 'fail',issues=issues,records=records,
        elapsed_s=time.perf_counter()-started,
        scope='Independent diagnostic only, never a parameter-selection constraint. Float64 regression checks support the unchanged structural proof; they are not an arbitrary-configuration accuracy certificate.',
        bound_used_as_constraint=False,PSD_eigenvalue_used_as_constraint=False,
        source_sha256={name:hashlib.sha256((P.HERE/name).read_bytes()).hexdigest() for name in ('protocol.py','search.py','timing.py')})
    P.save(P.HERE/'structural_audit.json',output)
    print(json.dumps(dict(status=output['status'],issues=issues,elapsed_s=output['elapsed_s']),indent=2))
    if issues:sys.exit(1)


if __name__=='__main__':main()
