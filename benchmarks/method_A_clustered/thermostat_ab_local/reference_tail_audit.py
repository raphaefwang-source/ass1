"""Reproduce the physical-reference tail proof, quadrature checks and hashes.

Run from this directory with ``python3 reference_tail_audit.py``.  The first
run preserves the original exploratory JSON before regenerating the audit.
No historical source or result is modified.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import protocol as P
import spatial as S
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUTPUT = HERE/'results'/'reference_tail_audit.json'
PILOT = OUTPUT.with_name('pilot_reference_tail_audit.json')
SNAPSHOT = HERE/'source_snapshot.json'

PROOF = (
    'For each particle, images of each other particle form a shifted L-lattice. '
    'Disjoint cubes of side L centered at images inside B(r) fit into B(r+a), '
    'a=sqrt(3)L/2. Their count n(r) is <= C(r)=(N-1)*4*pi*(r+a)^3/(3*L^3). '
    'Integration by parts gives sum_{distance>R}g(distance)=-g(R)n(R)+'
    'integral_R^infinity n(r)*(-gprime(r))dr <= '
    'integral_R^infinity C(r)*(-gprime(r))dr. '
    'Omitting self images is exact for a graph action. The omitted PSD graph '
    'norm is bounded by twice its maximum weighted degree, then by twice '
    'its maximum degree trace. physical_tail_bound evaluates the resulting '
    'analytic integral for g=exp(-.2*r)/r. Radius uses a physical minimum-image '
    'graph norm lower bound, so the norm ratio criterion is conservative.'
)
LOWER_NORM_PROOF = (
    'The minimum-image physical graph is a PSD summand of the full unsplit '
    'graph. Its maximum eigenvalue is a lower bound on the full graph norm '
    'and on each direct partial sum containing all minimum images.'
)
KERNEL_CORRESPONDENCE = (
    'E.kernel(.2,xi,0) creates RadialEwald(law=A,kappa=.2,amp=1); '
    'theta_full(r)=exp(-.2*r)/r. direct_unsplit_dense independently evaluates '
    'exp(-.2*r)*d*d.T/r^3. theta=0 removes every isotropic shift. '
    'Neither path uses common.py LJ amplitude or kappa=.7.'
)
FLOATING_POINT_SCOPE = (
    'Analytic truncation bound in real arithmetic; summation and evaluation '
    'floating point errors are not claimed as interval-certified. '
    'Unit tests compare independent sums and graph assemblies.'
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numerical_checks():
    checks = []
    for N, R in ((2, 8.), (16, 40.), (64, 140.)):
        L = (N/.75)**(1/3)
        a, k = S.np.sqrt(3)*L/2, .2

        def integrand(r):
            count = (N-1)*4*S.np.pi/(3*L**3)*(r+a)**3
            minus_gprime = S.np.exp(-k*r)*(k/r+1/r**2)
            return 2*count*minus_gprime

        numerical, error = quad(integrand, R, S.np.inf,
                                epsabs=1e-18, epsrel=1e-12)
        analytic = S.physical_tail_bound(N, L, R)
        discrepancy = abs(analytic-numerical)/analytic
        checks.append(dict(N=N, L=L, R=R, analytic_bound=analytic,
            independent_quadrature=numerical, quadrature_error_estimate=error,
            relative_discrepancy=discrepancy))
    return checks


def historical_hash_checks():
    snapshot = json.loads(SNAPSHOT.read_text())
    tracked = snapshot['tracked_file_sha256']
    differences = []
    for filename, expected in sorted(tracked.items()):
        path = REPO/filename
        actual = sha(path) if path.exists() else None
        if actual != expected:
            differences.append(dict(path=filename, expected_sha256=expected,
                                    actual_sha256=actual))
    return snapshot, len(tracked), differences


def main():
    if OUTPUT.exists() and not PILOT.exists():
        shutil.copyfile(OUTPUT, PILOT)
    checks = numerical_checks()
    snapshot, count, differences = historical_hash_checks()
    passed = not differences and all(c['relative_discrepancy'] < 1e-11 for c in checks)
    audit = dict(
        proof=PROOF, numerical_checks=checks, lower_norm_proof=LOWER_NORM_PROOF,
        kernel_correspondence=KERNEL_CORRESPONDENCE,
        historical_snapshot_checks=count, historical_snapshot_differences=differences,
        floating_point_scope=FLOATING_POINT_SCOPE,
        status='pass' if passed else 'fail',
        numerical_relative_tolerance=1e-11,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        command='python3 reference_tail_audit.py',
        script_sha256=sha(__file__), snapshot_sha256=sha(SNAPSHOT),
        spatial_source_sha256=sha(HERE/'spatial.py'),
        baseline_sha=snapshot['base_sha'],
        original_pilot_evidence=str(PILOT.relative_to(HERE)),
        original_pilot_sha256=sha(PILOT) if PILOT.exists() else None,
    )
    P.save(OUTPUT, audit)
    print(json.dumps(dict(status=audit['status'], output=str(OUTPUT),
        baseline_sha=audit['baseline_sha'], historical_snapshot_checks=count,
        historical_snapshot_differences=differences,
        numerical_checks=checks, numerical_relative_tolerance=1e-11), indent=2))
    if not passed:
        raise SystemExit('Reference-tail audit failed; inspect generated JSON')


if __name__ == '__main__':
    main()
