#ifndef SPARSE_SVD_INTERNAL_H
#define SPARSE_SVD_INTERNAL_H

/**
 * @file sparse_svd_internal.h
 * @brief Internal SVD helpers shared by the retained full-SVD path, the
 * partial-SVD backend, and selected proof/benchmark surfaces.
 *
 * Not part of the public API. Test code may include this header to access
 * internal functions; library consumers should not depend on these symbols.
 */

#include "sparse_types.h"

/**
 * @brief Full bidiagonal SVD: iterate QR steps until all superdiagonal entries
 * converge to zero.
 */
sparse_err_t bidiag_svd_iterate(double *diag, double *superdiag, idx_t k, double *U, idx_t m,
                                double *V, idx_t n, idx_t max_iter, double tol);

#endif /* SPARSE_SVD_INTERNAL_H */
