"""Audit tested rank-change diagnostics against independent dense action error.

This does not select benchmark ranks or claim an error bound.  The accuracy
driver selects ranks from dense errors; this utility retrospectively asks
whether a successive-tested-rank change would have made a correct decision.
The rank spacing is nonuniform and paths stop at their first dense pass.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import protocol as P
np = P.np


FIELDS = ('config', 'operator', 'alpha', 'dt', 'kind')


def _number(value):
    return None if value is None or value == '' else float(value)


def _decision(change, error, tolerance):
    if change is None:
        return 'no_previous_rank'
    diagnostic_pass, exact_pass = change <= tolerance, error <= tolerance
    if diagnostic_pass and not exact_pass:
        return 'false_positive'
    if not diagnostic_pass and exact_pass:
        return 'false_negative'
    return 'true_positive' if diagnostic_pass else 'true_negative'


def evaluate(rows):
    """Return per-direction and all-registered-direction decisions as data."""
    individual = []
    groups = defaultdict(list)
    for row in rows:
        error, change, tolerance = (_number(row[k]) for k in ('error', 'step_change', 'tolerance'))
        record = {k: row[k] for k in FIELDS}
        record.update(probe=row['probe'], rank=int(float(row['rank'])), error=error,
                      step_change=change, tolerance=tolerance,
                      decision=_decision(change, error, tolerance))
        individual.append(record)
        groups[tuple(record[k] for k in FIELDS) + (record['rank'],)].append(record)

    aggregate = []
    for key, members in groups.items():
        record = dict(zip(FIELDS + ('rank',), key))
        changes = [r['step_change'] for r in members]
        error = max(r['error'] for r in members)
        change = max(changes) if all(c is not None for c in changes) else None
        tolerance = min(r['tolerance'] for r in members)
        record.update(probe_count=len(members), max_error=error, max_step_change=change,
                      tolerance=tolerance, decision=_decision(change, error, tolerance))
        aggregate.append(record)
    return individual, aggregate


def counts(rows):
    output = dict.fromkeys(('no_previous_rank','true_positive','true_negative',
                            'false_positive','false_negative'),0)
    for row in rows:
        output[row['decision']] += 1
    return dict(output)


class GaussianSource:
    """First draw is saved z; later edge/substep draws are seeded normals."""
    def __init__(self, first_z, seed):
        self.first_z = np.asarray(first_z).copy()
        self.rng = np.random.default_rng(seed)
        self.first = True

    def normal(self, size):
        if self.first:
            self.first = False
            if np.prod(size) != self.first_z.size:
                raise ValueError('first Gaussian draw must match saved z')
            return self.first_z.reshape(size).copy()
        return self.rng.normal(size=size)


class NegRNG:
    """Sign-reverse every Gaussian without changing draw order or RNG state."""
    def __init__(self, source):
        self.source = source

    def normal(self, size):
        return -self.source.normal(size)


def ideal_a_path(edges, long_dense, p, dt, substeps, source):
    """Independent eigensolve path with the same literal Gaussian draw order.

    This intentionally does not call the production pair sweeps or Lanczos.
    Edge normals are interpreted in Cartesian coordinates, as in EdgeOU.
    """
    tau = dt/substeps
    Rlong, Flong, _ = P.DenseOU(long_dense).matrices(tau)
    lam, U = np.linalg.eigh(edges.K)
    lam = np.maximum(lam, 0.)
    # Pair sweeps have length tau/2, hence exp(-tau K) in relative momentum.
    damping = np.exp(-tau*lam)
    noise = np.sqrt(2 * -np.expm1(-2*tau*lam))

    def sweep(vector, eta, reverse):
        q = vector.reshape(-1,3).copy()
        order = range(len(edges.i)-1,-1,-1) if reverse else range(len(edges.i))
        for e in order:
            i,j = edges.i[e],edges.j[e]
            relative = q[i]-q[j]
            updated = U[e] @ (damping[e]*(U[e].T@relative) + noise[e]*(U[e].T@eta[e]))
            delta = .5*(updated-relative)
            q[i] += delta
            q[j] -= delta
        return q.ravel()

    output = np.asarray(p).ravel().copy()
    for _ in range(substeps):
        z = source.normal(size=output.size)
        ef = source.normal(size=(len(edges.i),3))
        er = source.normal(size=(len(edges.i),3))
        output = sweep(output,ef,False)
        output = Rlong@output + Flong@z
        output = sweep(output,er,True)
    return output


def coupling_checks():
    """Six physical production checks, each with coupled +, -, and exact paths."""
    import run_benchmarks as benchmark
    selections = P.read('selection.csv')
    structures = P.read('structure.csv')
    rows = []
    for N in (2,8,16,64,128,256):
        meta = next((r for r in structures if int(r['N'])==N and r['family']=='uniform'
                     and int(r['seed'])==2 and r['epsilon']==1e-4),None)
        if meta is None:
            rows.append(dict(N=N,status='configuration_unavailable'))
            continue
        P.log(f'coupled +/- production check {meta["config"]}')
        saved = np.load(P.INPUTS/(meta['config']+'.npz'))
        p,z = saved['p'],saved['z']
        system = benchmark.make(meta)
        # Enforce that a post-run source change cannot silently swap operators.
        np.testing.assert_array_equal(system.q,saved['q'])
        np.testing.assert_allclose(system.full_dense,saved['G'],rtol=0,atol=0)
        edges = benchmark.edge_object(system)
        dense = P.DenseOU(saved['G'])
        dt = 1./dense.lmax
        R,F,_ = dense.matrices(dt)
        for method in ('A','B'):
            selected = next((r for r in selections if r['config']==meta['config']
                             and r['alpha']==1. and r['method']==method and r['available']),None)
            if selected is None:
                rows.append(dict(N=N,config=meta['config'],method=method,status='rank_or_split_unavailable'))
                continue
            seed = P.SEED+50000+N
            r_d,r_n,k = (int(selected[key]) for key in ('r_d','r_n','substeps'))
            plus,diag = benchmark.sample_step(system,edges,p,dt,method,r_d,r_n,k,GaussianSource(z,seed))
            minus,_ = benchmark.sample_step(system,edges,-p,dt,method,r_d,r_n,k,NegRNG(GaussianSource(z,seed)))
            if method=='B':
                exact = R@p + F@z
                reference = 'independent full dense exact OU, same saved p/z'
            else:
                exact = ideal_a_path(edges,saved['GL'],p,dt,k,GaussianSource(z,seed))
                reference = 'independent ideal A same-noise path, full 3x3 edge eigensolves and dense long OU'
            row = dict(N=N,config=meta['config'],method=method,status='run',alpha=1.,dt=dt,
                       r_d=r_d,r_n=r_n,substeps=k,gaussian_seed=seed,
                       signflip_absolute=float(np.linalg.norm(plus+minus)),
                       signflip_relative=float(np.linalg.norm(plus+minus)/max(np.linalg.norm(plus),1e-300)),
                       coupled_exact_absolute=float(np.linalg.norm(plus-exact)),
                       coupled_exact_relative=float(np.linalg.norm(plus-exact)/max(np.linalg.norm(exact),1e-300)),
                       plus_COM_error=float(np.linalg.norm(plus.reshape(N,3).sum(0)-p.reshape(N,3).sum(0))),
                       exact_reference=reference,
                       interpretation='one saved coupled path tests actual finite action error; no universal covariance/FDT bound',
                       source_scope='same p; first z saved; all later edge/substep normals seeded independently; minus flips every supplied Gaussian',
                       mc_relationship='coupled exact path reduces stochastic comparison noise; MC CIs still use independent 128/512 samples')
            rows.append(row)
        P.write('coupling_checks.csv',rows)
    return rows


def main(allow_partial=False,skip_coupling=False):
    if not allow_partial and not (P.RESULTS/'accuracy_done.json').exists():
        raise RuntimeError('Wait until accuracy_done.json exists, or explicitly use --allow-partial')
    rows = P.read('rank_errors.csv')
    if not rows:
        raise RuntimeError('No rank_errors.csv evidence is available')
    individual, aggregate = evaluate(rows)
    P.write('stopping_diagnostics.csv', individual)
    P.write('stopping_diagnostics_all_probes.csv', aggregate)
    false_positives = [r for r in individual if r['decision'] == 'false_positive']
    comparable = [r for r in individual if r['step_change'] is not None]
    above_tolerance = [r for r in comparable if r['error'] > r['tolerance']]
    summary = dict(
        source='rank_errors.csv', source_sha256=P.sha(P.RESULTS/'rank_errors.csv'),
        complete_accuracy_marker=(P.RESULTS/'accuracy_done.json').exists(),
        per_direction=counts(individual), all_registered_directions=counts(aggregate),
        diagnostic_comparable_rows=len(comparable),
        false_positive_opportunities=len(above_tolerance),
        false_positive_assessment='unassessed: no rank-change row has dense error above tolerance' if not above_tolerance
                                  else 'empirical on observed above-tolerance rows only; not a certificate',
        false_positive_examples=false_positives[:20],
        definition='diagnostic pass: ||y_r-y_previous_tested_rank||/||y_r|| <= the same action tolerance; true pass: independent dense action error <= tolerance',
        benchmark_rank_selection='independent dense errors on five saved registered vectors; rank-change diagnostic does not select ranks',
        limitations=[
            'Only tested ranks, vectors, configurations, and time scales are covered; no general error certificate.',
            'Paths stop at the first dense passing rank, so further diagnostic-selected ranks and costs are unobserved.',
            'Rank increments are nonuniform; these are successive tested ranks, not necessarily r and r-1.',
            'A finite vector-dependent Lanczos noise action is nonlinear; these action diagnostics do not establish a covariance operator bound.',
            'No false positives in the observed rows would be empirical evidence only, not a proof that the diagnostic is safe.'
        ])
    P.save(P.RESULTS/'stopping_summary.json', summary)
    print(summary)
    if not skip_coupling:
        if not summary['complete_accuracy_marker']:
            print('Skipping coupled production checks until accuracy completes.')
        else:
            coupling_checks()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--skip-coupling', action='store_true')
    args=parser.parse_args()
    main(args.allow_partial,args.skip_coupling)
