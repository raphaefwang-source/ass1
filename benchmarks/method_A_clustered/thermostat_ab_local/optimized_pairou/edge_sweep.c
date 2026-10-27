/* Sequential exact pair OU increment kernel.  No edge reordering or threading.
 * D[e] = 0.5 U expm1(-2 tau lambda/m) U^T
 * S[e] = 0.5 U sqrt(2 m theta [-expm1(-4 tau lambda/m)]) U^T.
 * Noise is indexed by ORIGINAL edge id even during a reverse sweep.
 * Compile with -O3 -fno-fast-math -ffp-contract=off; see compiled_edge.py.
 */
#include <stdint.h>

int pairou_sweep(const int64_t n_particles, const int64_t n_edges,
                 const int64_t *ii, const int64_t *jj,
                 const double *D, const double *S,
                 double *p, const double *eta, const int reverse) {
    if (n_particles < 1 || n_edges < 0) return 1;
    /* Validate the whole graph before mutating p, even if called outside Python. */
    for (int64_t e = 0; e < n_edges; ++e) {
        if (ii[e] < 0 || jj[e] < 0 || ii[e] >= n_particles ||
            jj[e] >= n_particles || ii[e] == jj[e]) return 2;
    }
    for (int64_t k = 0; k < n_edges; ++k) {
        const int64_t e = reverse ? n_edges - 1 - k : k;
        const int64_t i = 3 * ii[e], j = 3 * jj[e];
        const int64_t a = 9 * e, b = 3 * e;
        /* All relative components must be read before either endpoint changes. */
        const double r0 = p[i] - p[j];
        const double r1 = p[i+1] - p[j+1];
        const double r2 = p[i+2] - p[j+2];
        const double z0 = eta[b], z1 = eta[b+1], z2 = eta[b+2];
        const double d0 = (D[a]*r0 + D[a+1]*r1 + D[a+2]*r2)
                        + (S[a]*z0 + S[a+1]*z1 + S[a+2]*z2);
        const double d1 = (D[a+3]*r0 + D[a+4]*r1 + D[a+5]*r2)
                        + (S[a+3]*z0 + S[a+4]*z1 + S[a+5]*z2);
        const double d2 = (D[a+6]*r0 + D[a+7]*r1 + D[a+8]*r2)
                        + (S[a+6]*z0 + S[a+7]*z1 + S[a+8]*z2);
        p[i] += d0;     p[j] -= d0;
        p[i+1] += d1;   p[j+1] -= d1;
        p[i+2] += d2;   p[j+2] -= d2;
    }
    return 0;
}
