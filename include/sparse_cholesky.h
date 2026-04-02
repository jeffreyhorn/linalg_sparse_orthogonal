#ifndef SPARSE_CHOLESKY_H
#define SPARSE_CHOLESKY_H

/**
 * @file sparse_cholesky.h
 * @brief Cholesky decomposition for sparse symmetric positive-definite matrices.
 *
 * Provides in-place Cholesky factorization A = L*L^T where L is lower
 * triangular. Exploits symmetry: only the lower triangle is stored after
 * factorization. No pivoting is needed for SPD matrices.
 *
 * **Usage pattern:**
 * @code
 *   SparseMatrix *A = sparse_create(n, n);
 *   // ... populate A with SPD entries ...
 *   SparseMatrix *L = sparse_copy(A);
 *   sparse_cholesky_factor(L);
 *   sparse_cholesky_solve(L, b, x);   // solve A*x = b
 *
 *   // With fill-reducing reordering:
 *   SparseMatrix *L2 = sparse_copy(A);
 *   sparse_cholesky_opts_t opts = { SPARSE_REORDER_AMD };
 *   sparse_cholesky_factor_opts(L2, &opts);
 *   sparse_cholesky_solve(L2, b, x);  // reorder/unpermute automatic
 * @endcode
 */

#include "sparse_matrix.h"

/**
 * @brief Options for Cholesky factorization with optional fill-reducing reordering.
 */
typedef struct {
    sparse_reorder_t reorder; /**< Fill-reducing reordering (NONE, RCM, or AMD) */
} sparse_cholesky_opts_t;

/**
 * @brief Compute the Cholesky factorization of a sparse SPD matrix in-place.
 *
 * Computes A = L*L^T where L is lower triangular. The lower triangle of mat
 * is overwritten with L; the upper triangle entries are removed. No pivoting
 * is performed (SPD matrices have all-positive pivots).
 *
 * Fill-in entries with |value| < SPARSE_DROP_TOL * L(k,k) are dropped to
 * control memory growth.
 *
 * @param mat  The SPD matrix to factor (modified in-place). Must be square
 *             and symmetric. After factorization, contains L in the lower triangle.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if mat is NULL.
 * @return SPARSE_ERR_SHAPE if mat is not square.
 * @return SPARSE_ERR_NOT_SPD if the matrix is not symmetric (checked first)
 *         or a non-positive pivot is encountered during factorization.
 *
 * @par Thread safety: Mutates mat. Not safe to call concurrently on the same matrix.
 *               Safe to call concurrently on different matrices.
 */
sparse_err_t sparse_cholesky_factor(SparseMatrix *mat);

/**
 * @brief Compute Cholesky factorization with options including fill-reducing reordering.
 *
 * If opts->reorder != SPARSE_REORDER_NONE, the matrix is symmetrically
 * permuted before factorization. The reordering permutation is stored in
 * the matrix so that sparse_cholesky_solve() can automatically unpermute.
 *
 * @param mat   The SPD matrix to factor (modified in-place).
 * @param opts  Factorization options.
 * @return SPARSE_OK on success, or an error code.
 */
sparse_err_t sparse_cholesky_factor_opts(SparseMatrix *mat, const sparse_cholesky_opts_t *opts);

/**
 * @brief Solve A*x = b using a Cholesky-factored matrix.
 *
 * Given L from sparse_cholesky_factor(), solves:
 *   L * y = b    (forward substitution)
 *   L^T * x = y  (backward substitution)
 *
 * If the matrix was factored with reordering, b is automatically permuted
 * and x is automatically unpermuted.
 *
 * @param mat  A matrix that has been factored by sparse_cholesky_factor().
 * @param b    Right-hand side vector of length n.
 * @param x    Solution vector of length n (overwritten). May alias b.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 *
 * @par Thread safety: Read-only on mat. Safe to call concurrently on the same
 *               factored matrix with different b/x vectors.
 */
sparse_err_t sparse_cholesky_solve(const SparseMatrix *mat, const double *b, double *x);

#endif /* SPARSE_CHOLESKY_H */
