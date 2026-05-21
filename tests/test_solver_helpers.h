#ifndef TEST_SOLVER_HELPERS_H
#define TEST_SOLVER_HELPERS_H

#include "sparse_matrix.h"

#include <math.h>
#include <stdlib.h>

/* Cluster-scoped helper layer for solver/integration tests.
 *
 * Sprint 37 Day 5 starts by consolidating the repeated L2 residual
 * calculations that were drifting across iterative/preconditioner and
 * integration tests.  Keep this header narrow and explicit instead of
 * growing a broad generic test framework.
 */

static inline double tf_vec_norm2(const double *v, idx_t n) {
    double sum = 0.0;
    for (idx_t i = 0; i < n; i++)
        sum += v[i] * v[i];
    return sqrt(sum);
}

static inline double tf_relative_residual_l2(const SparseMatrix *A, const double *b,
                                             const double *x, idx_t n, double alloc_fail_sentinel) {
    if (n == 0)
        return 0.0;

    double *r = calloc((size_t)n, sizeof(double));
    if (!r)
        return alloc_fail_sentinel;

    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - r[i];

    double rnorm = tf_vec_norm2(r, n);
    double bnorm = tf_vec_norm2(b, n);
    free(r);
    return (bnorm > 0.0) ? rnorm / bnorm : rnorm;
}

static inline double tf_block_relative_residual_l2(const SparseMatrix *A, const double *B,
                                                   const double *X, idx_t n, idx_t nrhs,
                                                   double alloc_fail_sentinel) {
    if (n == 0 || nrhs == 0)
        return 0.0;

    double *Y = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    if (!Y)
        return alloc_fail_sentinel;

    sparse_matvec_block(A, X, nrhs, Y);
    double worst = 0.0;
    for (idx_t k = 0; k < nrhs; k++) {
        double rnorm_sq = 0.0;
        double bnorm_sq = 0.0;
        for (idx_t i = 0; i < n; i++) {
            double ri = B[i + k * n] - Y[i + k * n];
            rnorm_sq += ri * ri;
            bnorm_sq += B[i + k * n] * B[i + k * n];
        }
        double rnorm = sqrt(rnorm_sq);
        double bnorm = sqrt(bnorm_sq);
        double rel = (bnorm > 0.0) ? rnorm / bnorm : rnorm;
        if (rel > worst)
            worst = rel;
    }
    free(Y);
    return worst;
}

#endif
