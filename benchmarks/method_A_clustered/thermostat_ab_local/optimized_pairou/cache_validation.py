"""Physical N16 cache/parameter/configuration validation, with raw provenance.

Run before formal timing: python3 -B cache_validation.py
All inputs and evidence are written below optimized_pairou; PR8 stays read only.
"""
from time import perf_counter
import bench_common as C
from compiled_edge import CompiledEdgeOU
np,P,O = C.np,C.P,C.O


def coefficients_sha(pair):
    import hashlib
    digest=hashlib.sha256()
    for array in pair:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def expected_coefficients(original,tau,mass,theta):
    # Independent definition from the unchanged Python engine's eigensystem.
    drift=.5*np.expm1(-2*tau*original.eigenvalues/mass)
    noise=.5*np.sqrt(2*mass*theta*-np.expm1(-4*tau*original.eigenvalues/mass))
    return tuple(np.einsum('eik,ek,ejk->eij',original.U,values,original.U)
                 for values in (drift,noise))


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))


def main():
    if (C.RESULTS/'cache_validation.json').exists():
        raise FileExistsError('Physical cache validation evidence already exists')
    started=perf_counter()
    meta=next(r for r in C.parent_rows('structure.csv') if int(r['N'])==16 and
              r['family']=='uniform' and int(r['seed'])==2 and r['epsilon']==1e-4)
    system=C.make_system(meta)
    original=O.EdgeOU(system.edges,n_particles=16)
    token=C.S.position_hash(system.q,system.L)
    compiled=CompiledEdgeOU(system.edges,n_particles=16,configuration_token=token)
    if compiled.backend!='compiled':
        raise RuntimeError(f'Physical validation requires compiled backend: {compiled.compile_info}')
    baseline_graph_sha=compiled.graph_sha256
    dt=1./P.DenseOU(system.full_dense).lmax
    taus=(0.,1e-16,.25*dt,.5*dt,dt)
    thetas=(0.,.3,1.7)
    masses=(.7,1.)
    seed=C.SEED+800000
    rng=np.random.default_rng(seed)
    p=rng.normal(size=(16,3))
    eta=rng.normal(size=(len(original.i),3))
    rows=[]
    previous_key=previous_sha=None
    for tau in taus:
        for theta in thetas:
            for mass in masses:
                key=(float(tau),float(mass),float(theta))
                first=compiled.prepare(tau,mass,theta)
                second=compiled.prepare(tau,mass,theta)
                if first['coefficient_cache_hit'] or not second['coefficient_cache_hit']:
                    raise RuntimeError('Fresh key/repeat did not produce miss/hit')
                stored=compiled._compiled_factors[key]
                expected=expected_coefficients(original,tau,mass,theta)
                residual=max(float(np.max(np.abs(a-b),initial=0.)) for a,b in zip(stored,expected))
                sha,expected_sha=coefficients_sha(stored),coefficients_sha(expected)
                if residual>2e-14:
                    raise RuntimeError('Cached D/S does not match physical parameters')
                for reverse in (False,True):
                    actual,diag=compiled.sweep(p,eta,tau,reverse,mass,theta)
                    truth,_=original.sweep(p,eta,tau,reverse,mass,theta)
                    error=relative(actual,truth)
                    if error>1e-11:
                        raise RuntimeError(f'Physical compiled sweep differs: {error}')
                    rows.append(dict(config=meta['config'],fixture='original_q',tau=tau,theta=theta,mass=mass,
                        reverse=reverse,edges=len(original.i),dt=dt,first_prepare_miss=not first['coefficient_cache_hit'],
                        repeated_prepare_hit=second['coefficient_cache_hit'],sweep_prepare_hit=diag['coefficient_cache_hit'],
                        coefficient_cache_size=compiled.coefficient_cache_size,
                        coefficient_sha256=sha,expected_coefficient_sha256=expected_sha,
                        coefficient_definition_max_abs_error=residual,
                        parameter_key_changed=previous_key is not None and previous_key!=key,
                        coefficients_changed=previous_sha is not None and previous_sha!=sha,
                        coefficient_identity_note='zero tau gives D=S=0 for every mass/theta; equal hashes are expected',
                        path_relative_error=error,path_absolute_error=float(np.linalg.norm(actual-truth)),
                        COM_error=float(np.linalg.norm(actual.sum(0)-p.sum(0))),
                        graph_sha256=compiled.graph_sha256,position_sha256=token))
                previous_key,previous_sha=key,sha

    # Perturb actual positions, then rebuild the production geometry using exactly
    # the same box, physical kernel, cutoff and projected mesh table.
    changed_q=system.q.copy()
    perturbation=1e-4*system.L*np.array([.3,-.2,.1])
    changed_q[0]=np.mod(changed_q[0]+perturbation,system.L)
    changed_token=C.S.position_hash(changed_q,system.L)
    changed_op=C.S.GridPSDGamma(changed_q,system.L,system.op.K,system.op.rc,system.op.mesh)
    changed_edges=C.S.near_edges(changed_op)
    changed_original=O.EdgeOU(changed_edges,n_particles=16)
    rebuilt=compiled.rebuild(changed_edges,configuration_token=changed_token)
    initial_rebuilt_cache_size=rebuilt.coefficient_cache_size
    if initial_rebuilt_cache_size!=0 or rebuilt.graph_sha256==baseline_graph_sha:
        raise RuntimeError('Changed production geometry did not invalidate the coefficient cache')
    changed_eta=eta.copy() if len(changed_original.i)==len(original.i) else rng.normal(size=(len(changed_original.i),3))
    tau,mass,theta=.5*dt,1.,1.7
    key=(float(tau),float(mass),float(theta))
    original_coefficient_sha=coefficients_sha(compiled._compiled_factors[key])
    first=rebuilt.prepare(tau,mass,theta)
    second=rebuilt.prepare(tau,mass,theta)
    changed_sha=coefficients_sha(rebuilt._compiled_factors[key])
    if first['coefficient_cache_hit'] or not second['coefficient_cache_hit'] or changed_sha==original_coefficient_sha:
        raise RuntimeError('Rebuilt geometry failed to recompute parameter coefficients')
    for reverse in (False,True):
        actual,diag=rebuilt.sweep(p,changed_eta,tau,reverse,mass,theta)
        truth,_=changed_original.sweep(p,changed_eta,tau,reverse,mass,theta)
        error=relative(actual,truth)
        if error>1e-11:
            raise RuntimeError('Changed-configuration production sweep differs')
        rows.append(dict(config=meta['config'],fixture='perturbed_q',tau=tau,theta=theta,mass=mass,
            reverse=reverse,edges=len(changed_original.i),dt=dt,
            first_prepare_miss=not first['coefficient_cache_hit'],repeated_prepare_hit=second['coefficient_cache_hit'],
            sweep_prepare_hit=diag['coefficient_cache_hit'],coefficient_cache_size=rebuilt.coefficient_cache_size,
            coefficient_sha256=changed_sha,old_coefficient_sha256=original_coefficient_sha,
            initial_rebuilt_cache_size=initial_rebuilt_cache_size,coefficients_changed=True,
            path_relative_error=error,path_absolute_error=float(np.linalg.norm(actual-truth)),
            COM_error=float(np.linalg.norm(actual.sum(0)-p.sum(0))),
            graph_sha256=rebuilt.graph_sha256,position_sha256=changed_token))
    old_graph_unchanged=(compiled.graph_sha256==baseline_graph_sha and
        coefficients_sha(compiled._compiled_factors[key])==original_coefficient_sha)
    if not old_graph_unchanged or compiled.coefficient_cache_size!=30:
        raise RuntimeError('Rebuild mutated the old compiled graph/cache')
    immutable_checks={}
    for name in ('i','j','K','U','eigenvalues'):
        try:
            getattr(compiled,name).flags.writeable=True
        except ValueError:
            immutable_checks[name]=True
        else:
            raise RuntimeError(f'{name} mutation protection failed')

    C.INPUTS.mkdir(parents=True,exist_ok=True)
    fixture=C.INPUTS/'cache_validation_fixture.npz'
    np.savez_compressed(fixture,q=system.q,q_perturbed=changed_q,L=system.L,
        perturbation=perturbation,p=p,eta=eta,eta_perturbed=changed_eta,
        edge_i=original.i,edge_j=original.j,edge_K=original.K,
        changed_edge_i=changed_original.i,changed_edge_j=changed_original.j,changed_edge_K=changed_original.K,
        tau=np.array(taus),theta=np.array(thetas),mass=np.array(masses),seed=seed,dt=dt)
    C.write('cache_validation.csv',rows)
    C.save('cache_validation.json',dict(
        passed=True,elapsed_s=perf_counter()-started,config=meta['config'],rows=len(rows),parameter_combinations=30,
        original_graph_cache_size=compiled.coefficient_cache_size,rebuilt_graph_cache_size=rebuilt.coefficient_cache_size,
        initial_rebuilt_cache_size=initial_rebuilt_cache_size,
        path_relative_error_max=max(r['path_relative_error'] for r in rows),
        path_absolute_error_max=max(r['path_absolute_error'] for r in rows),
        COM_error_max=max(r['COM_error'] for r in rows),path_budget=1e-11,
        first_miss_then_repeat_hit=all(r['first_prepare_miss'] and r['repeated_prepare_hit'] and r['sweep_prepare_hit'] for r in rows),
        coefficient_definition_max_abs_error=max(r.get('coefficient_definition_max_abs_error',0.) for r in rows),
        zero_tau_identity_hash_note='D and S are both zero, so coefficient hashes legitimately agree across mass/theta keys',
        raw_fixture=str(fixture),raw_fixture_sha256=C.sha(fixture),seed=seed,
        original_position_sha256=token,changed_position_sha256=changed_token,
        original_graph_sha256=baseline_graph_sha,changed_graph_sha256=rebuilt.graph_sha256,
        original_edge_K_sha256=C.array_sha(original.K),changed_edge_K_sha256=C.array_sha(changed_original.K),
        original_graph_and_cache_unchanged=old_graph_unchanged,immutable_checks=immutable_checks,
        production_geometry='new GridPSDGamma(q_perturbed,L,old.K,old.rc,old.mesh); all near images re-extracted',
        physical_metadata=system.metadata,reproduce='python3 -B cache_validation.py',
        inference_limit='finite physical parameter/configuration checks; no trajectory or universal floating-point bound'))
    print(dict(passed=True,rows=len(rows),path_relative_error_max=max(r['path_relative_error'] for r in rows),
               old_graph_unchanged=old_graph_unchanged,elapsed_s=perf_counter()-started),flush=True)


if __name__=='__main__':
    main()
