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
} sparse_qr_opts_t;

/**
 * @brief QR factorization data.
 *
 * Stores R (upper triangular), Householder reflectors (v, beta) for Q,
 * column permutation, and rank information.
 */
typedef struct {
    SparseMatrix *R;    /**< Upper triangular factor (min(m,n) × n after permutation) */
    double *betas;      /**< Householder scalars beta_k, length min(m,n) */
    double **v_vectors; /**< Householder vectors v_k, each length m-k (stored from diagonal down) */
    idx_t *col_perm;    /**< Column permutation: col_perm[k] = original column index */
    idx_t m;            /**< Number of rows of original A */
    idx_t n;            /**< Number of columns of original A */
    idx_t rank;         /**< Numerical rank (set during factorization) */
} sparse_qr_t;

/**
 * @brief Compute column-pivoted QR factorization: A*P = Q*R.
 *
 * @param A   The matrix to factor (not modified). May be rectangular (m×n).
 * @param qr  Output: QR factors. Must be freed with sparse_qr_free().
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or qr is NULL.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_qr_factor(const SparseMatrix *A, sparse_qr_t *qr);

/**
 * @brief Compute QR factorization with options (e.g., fill-reducing reordering).
 *
 * @param A    The matrix to factor (not modified).
 * @param opts Factorization options (NULL for defaults).
 * @param qr   Output: QR factors.
 * @return SPARSE_OK on success, or an error code.
 */
sparse_err_t sparse_qr_factor_opts(const SparseMatrix *A, const sparse_qr_opts_t *opts,
                                   sparse_qr_t *qr);

/**
 * @brief Apply Q or Q^T to a vector.
 *
 * Computes y = Q*x or y = Q^T*x using the stored Householder reflectors,
 * without forming Q explicitly.
 *
 * @param qr     The QR factorization.
 * @param side   0 for Q*x, 1 for Q^T*x.
 * @param x      Input vector of length m.
 * @param y      Output vector of length m (may alias x for in-place).
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_qr_apply_q(const sparse_qr_t *qr, int transpose, const double *x, double *y);

/**
 * @brief Explicitly form the Q matrix (for testing/diagnostics).
 *
 * Forms Q as a dense m×m matrix by applying Q to columns of I.
 * Not recommended for large matrices.
 *
 * @param qr  The QR factorization.
 * @param Q   Output: dense m×m matrix in column-major order. Caller allocates m*m doubles.
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_qr_form_q(const sparse_qr_t *qr, double *Q);

/**
 * @brief Solve the least-squares problem min ||Ax - b||_2.
 *
 * For overdetermined systems (m > n), computes the minimum-norm solution.
 * For rank-deficient systems, solves for the rank components and sets
 * remaining to zero.
 *
 * @param qr       The QR factorization of A.
 * @param b        Right-hand side vector of length m.
 * @param x        Output: solution vector of length n.
 * @param residual Output: residual norm ||b - Ax||_2 (may be NULL).
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_qr_solve(const sparse_qr_t *qr, const double *b, double *x, double *residual);

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
 * Returns basis vectors for the null space of A (columns of V corresponding
 * to zero/small R diagonals).
 *
 * @param qr       The QR factorization.
 * @param tol      Tolerance for rank determination (same as sparse_qr_rank).
 * @param basis    Output: null-space basis vectors (n × null_dim, column-major).
 *                 Caller allocates n * (n - rank) doubles.
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
