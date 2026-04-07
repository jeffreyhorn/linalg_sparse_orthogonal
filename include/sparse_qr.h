#ifndef SPARSE_QR_H
#define SPARSE_QR_H

/**
 * @file sparse_qr.h
 * @brief Sparse QR factorization with column pivoting.
 *
 * Provides column-pivoted QR factorization A*P = Q*R using Householder
 * reflections, with least-squares solving, rank estimation, and null-space
 * extraction.
 *
 * Q is stored implicitly as a sequence of Householder reflectors
 * (v_k, beta_k) for memory efficiency. Use sparse_qr_apply_q() to
 * apply Q or Q^T to vectors without forming Q explicitly.
 *
 * **Usage pattern:**
 * @code
 *   SparseMatrix *A = ...;  // m×n matrix
 *   sparse_qr_t qr;
 *   sparse_qr_factor(A, &qr);
 *
 *   // Least-squares solve: min ||Ax - b||
 *   double *x = malloc(n * sizeof(double));
 *   double residual;
 *   sparse_qr_solve(&qr, b, x, &residual);
 *
 *   // Rank estimation
 *   idx_t rank = sparse_qr_rank(&qr, 1e-12);
 *
 *   sparse_qr_free(&qr);
 * @endcode
 */

#include "sparse_matrix.h"

/**
 * @brief Options for QR factorization.
 */
typedef struct {
    sparse_reorder_t reorder; /**< Column reordering before QR (default: NONE) */
    int economy;              /**< When nonzero and m > n, compute economy (thin) QR:
                                   form_q produces m×n instead of m×m. Has no effect
                                   when m <= n (Q is already m×m = m×k where k=min(m,n)).
                                   (default: 0 = full QR) */
    int sparse_mode;          /**< When nonzero, use column-by-column Householder
                                   application instead of O(m*n) dense workspace.
                                   Uses O(m) working memory per column. Slower but
                                   scales to larger matrices. (default: 0) */
} sparse_qr_opts_t;

/**
 * @brief QR factorization data.
 *
 * Stores R (upper triangular), Householder reflectors (v, beta) for Q,
 * column permutation, and rank information.
 *
 * Callers must call sparse_qr_free() before reusing a sparse_qr_t for
 * a new factorization; the factor functions overwrite the struct without
 * freeing prior contents. sparse_qr_free() is safe on a zeroed struct.
 */
typedef struct {
    SparseMatrix *R;    /**< Upper triangular factor (min(m,n) × n after permutation) */
    double *betas;      /**< Householder scalars beta_k, length min(m,n) */
    double **v_vectors; /**< Householder vectors v_k, each length m-k (stored from diagonal down) */
    idx_t *col_perm;    /**< Column permutation: col_perm[k] = original column index */
    idx_t m;            /**< Number of rows of original A */
    idx_t n;            /**< Number of columns of original A */
    idx_t rank;         /**< Numerical rank (set during factorization) */
    int economy;        /**< Nonzero if economy (thin Q) was requested.
                             A thin Q is only formed when m > n; when m <= n
                             this flag has no effect on the shape of Q. */
} sparse_qr_t;

/**
 * @brief Compute column-pivoted QR factorization: A*P = Q*R.
 *
 * @pre A must have identity permutations (not previously factored, pivoted,
 *      or reordered).  A is not modified.
 *
 * @param A   The matrix to factor (not modified). May be rectangular (m×n).
 * @param qr  Output: QR factors. Must be freed with sparse_qr_free().
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or qr is NULL.
 * @return SPARSE_ERR_BADARG if A has non-identity permutations.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_qr_factor(const SparseMatrix *A, sparse_qr_t *qr);

/**
 * @brief Compute QR factorization with options (e.g., fill-reducing reordering).
 *
 * @param A    The matrix to factor (not modified). Must have identity permutations.
 * @param opts Factorization options (NULL for defaults).
 * @param qr   Output: QR factors.
 * @return SPARSE_OK on success, or an error code.
 * @return SPARSE_ERR_BADARG if A has non-identity permutations.
 */
sparse_err_t sparse_qr_factor_opts(const SparseMatrix *A, const sparse_qr_opts_t *opts,
                                   sparse_qr_t *qr);

/**
 * @brief Apply Q or Q^T to a vector.
 *
 * Computes y = Q*x or y = Q^T*x using the stored Householder reflectors,
 * without forming Q explicitly.
 *
 * @param qr        The QR factorization.
 * @param transpose 0 for Q*x, 1 for Q^T*x.
 * @param x         Input vector of length m.
 * @param y         Output vector of length m (may alias x for in-place).
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_qr_apply_q(const sparse_qr_t *qr, int transpose, const double *x, double *y);

/**
 * @brief Explicitly form the Q matrix (for testing/diagnostics).
 *
 * For full QR (economy=0): forms Q as a dense m×m orthogonal matrix.
 * Caller allocates m*m doubles.
 *
 * For economy QR (economy=1): forms the thin Q as a dense m×k matrix
 * with orthonormal columns, where k = min(m, n). Caller allocates m*k doubles.
 *
 * @param qr  The QR factorization.
 * @param Q   Output: dense matrix in column-major order.
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_qr_form_q(const sparse_qr_t *qr, double *Q);

/**
 * @brief Solve the least-squares problem min ||Ax - b||_2.
 *
 * For overdetermined systems (m > n), computes a least-squares solution.
 * For underdetermined or rank-deficient systems, computes a basic
 * least-squares solution by solving for the numerically determined
 * (rank) components and setting the remaining free components to zero
 * in the column-permuted coordinate system. The resulting solution is
 * not, in general, the minimum-norm least-squares solution.
 *
 * @param qr       The QR factorization of A.
 * @param b        Right-hand side vector of length m.
 * @param x        Output: solution vector of length n.
 * @param residual Output: residual norm ||b - Ax||_2 (may be NULL).
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_qr_solve(const sparse_qr_t *qr, const double *b, double *x, double *residual);

/**
 * @brief Iterative refinement for QR least-squares solutions.
 *
 * Improves an existing QR solution by repeatedly computing the residual
 * r = b - A*x and solving for a correction via the existing QR factorization.
 * Useful for reducing the residual on ill-conditioned systems.
 *
 * @param qr         The QR factorization of A.
 * @param A          The original matrix (for computing residuals).
 * @param b          Right-hand side vector of length m.
 * @param x          On entry: initial solution (from sparse_qr_solve).
 *                   On exit: refined solution. Length n.
 * @param max_refine Maximum number of refinement iterations. 0 = just compute residual.
 * @param residual   Output: final residual norm ||b - Ax||_2 (may be NULL).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any required argument is NULL.
 * @return SPARSE_ERR_SHAPE if A dimensions don't match the QR factorization.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_qr_refine(const sparse_qr_t *qr, const SparseMatrix *A, const double *b,
                              double *x, idx_t max_refine, double *residual);

/**
 * @brief Estimate numerical rank from QR factorization.
 *
 * Counts the number of R diagonal entries exceeding tol * |R(0,0)|.
 *
 * @param qr  The QR factorization.
 * @param tol Tolerance (0 for default: eps * max(m,n) * |R(0,0)|).
 * @return The estimated numerical rank.
 */
idx_t sparse_qr_rank(const sparse_qr_t *qr, double tol);

/**
 * @brief Extract null-space basis vectors.
 *
 * Returns basis vectors for the (right) null space of A. The basis is
 * constructed from the columns associated with zero/small diagonal entries
 * of the R factor, taking into account the stored column permutation, so
 * that each output vector x satisfies A*x ≈ 0 in the original column
 * ordering.
 *
 * @param qr       The QR factorization.
 * @param tol      Tolerance for rank determination (same as sparse_qr_rank).
 * @param basis    Output: null-space basis vectors (n × null_dim, column-major),
 *                 expressed in the original column ordering. Caller allocates
 *                 n * (n - rank) doubles.
 * @param null_dim Output: null-space dimension (n - rank).
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_qr_nullspace(const sparse_qr_t *qr, double tol, double *basis, idx_t *null_dim);

/**
 * @brief Free QR factorization data.
 *
 * @param qr  The QR factors to free. Safe to call on a zeroed struct.
 */
void sparse_qr_free(sparse_qr_t *qr);

#endif /* SPARSE_QR_H */
