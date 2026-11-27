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

#include "sparse_svd.h"
#include <stddef.h>

typedef struct {
    size_t bidiag_work_peak_bytes;
    size_t retained_factor_bytes;
    size_t reconstruction_scratch_peak_bytes;
    size_t sparse_shell_map_bytes;
    size_t sparse_node_slab_bytes;
    size_t bidiag_work_allocations;
    size_t retained_factor_allocations;
    size_t reconstruction_scratch_allocations;
    size_t sparse_shell_map_allocations;
    size_t sparse_node_slab_allocations;
    idx_t output_nnz;
    int valid;
} sparse_svd_lowrank_metrics_t;

/** Reset/read the thread-local low-rank accounting probe. Internal only. */
void sparse_svd_lowrank_metrics_reset(void);
void sparse_svd_lowrank_metrics_get(sparse_svd_lowrank_metrics_t *metrics);

/**
 * @brief Full bidiagonal SVD: iterate QR steps until all superdiagonal entries
 * converge to zero.
 */
sparse_err_t bidiag_svd_iterate(double *diag, double *superdiag, idx_t k, double *U, idx_t m,
                                double *V, idx_t n, idx_t max_iter, double tol);

/**
 * @brief Reconstruct a sparse rank-k approximation directly from economy-SVD
 * factors without an m*n accumulator.
 *
 * This is an internal proof surface used by the public low-rank wrapper and
 * selected numerical/ownership tests. @p effective_drop_tol must already be
 * resolved to a finite nonnegative threshold. On every failure, @p out is
 * left NULL and any partially constructed sparse matrix is destroyed.
 */
sparse_err_t sparse_svd_lowrank_outer_product(const sparse_svd_t *svd, idx_t rank_k,
                                              double effective_drop_tol, idx_t m, idx_t n,
                                              SparseMatrix **out);

#endif /* SPARSE_SVD_INTERNAL_H */
