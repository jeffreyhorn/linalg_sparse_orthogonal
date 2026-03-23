#ifndef SPARSE_LU_H
#define SPARSE_LU_H

/**
 * @file sparse_lu.h
 * @brief LU decomposition and linear system solving for sparse matrices.
 *
 * Provides in-place LU factorization with row and column pivoting
 * (P·A·Q = L·U), forward/backward substitution, and iterative refinement.
 *
 * The factorization stores L and U in the same matrix: L occupies the
 * strictly lower triangle (with an implicit unit diagonal), and U occupies
 * the upper triangle (including the diagonal). The permutations P and Q
 * are stored in the matrix's row_perm and col_perm arrays.
 *
 * **Usage pattern:**
 * @code
 *   SparseMatrix *A  = sparse_create(n, n);
 *   // ... populate A ...
 *   SparseMatrix *LU = sparse_copy(A);          // preserve original
 *   sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
 *   sparse_lu_solve(LU, b, x);                  // solve A*x = b
 *   sparse_lu_refine(A, LU, b, x, 5, 1e-15);   // optional refinement
 *
 *   // Check conditioning:
 *   double cond;
 *   sparse_lu_condest(A, LU, &cond);
 *   if (cond > 1e12) fprintf(stderr, "Warning: ill-conditioned (cond ~%.1e)\n", cond);
 * @endcode
 */

#include "sparse_matrix.h"

/**
 * @brief Options for LU factorization with optional fill-reducing reordering.
 *
 * Use with sparse_lu_factor_opts(). The reordering permutation is computed
 * and applied automatically; sparse_lu_solve() will detect it and unpermute
 * the solution transparently.
 *
 * @code
 *   sparse_lu_opts_t opts = { SPARSE_PIVOT_PARTIAL, SPARSE_REORDER_AMD, 1e-12 };
 *   sparse_lu_factor_opts(A, &opts);
 *   sparse_lu_solve(A, b, x);  // reorder/unpermute handled automatically
 * @endcode
 */
typedef struct {
    sparse_pivot_t  pivot;    /**< Pivoting strategy */
    sparse_reorder_t reorder; /**< Fill-reducing reordering (NONE, RCM, or AMD) */
    double          tol;      /**< Pivot tolerance */
} sparse_lu_opts_t;

/**
 * @brief Compute LU factorization with options including fill-reducing reordering.
 *
 * If opts->reorder != SPARSE_REORDER_NONE, the matrix is symmetrically
 * permuted before factorization. The reordering permutation is stored in
 * the matrix so that sparse_lu_solve() can automatically unpermute the
 * solution.
 *
 * @param mat   The matrix to factor (modified in-place: reordered, then factored).
 * @param opts  Factorization options.
 * @return SPARSE_OK on success, or an error code.
 */
sparse_err_t sparse_lu_factor_opts(SparseMatrix *mat,
                                   const sparse_lu_opts_t *opts);

/**
 * @brief Compute the LU factorization of a sparse matrix in-place.
 *
 * Performs Gaussian elimination with the chosen pivoting strategy. After
 * factorization, the matrix stores both L (below diagonal, unit diagonal
 * implied) and U (on and above diagonal). Permutations are recorded in the
 * matrix's internal row_perm, col_perm, and their inverses.
 *
 * Fill-in entries with |value| < SPARSE_DROP_TOL * |pivot| are dropped
 * to control memory growth.
 *
 * @param mat    The matrix to factor (modified in-place). Must be square.
 * @param pivot  Pivoting strategy:
 *               - @c SPARSE_PIVOT_COMPLETE — search entire remaining submatrix
 *                 for the largest element. Better stability, O(n^2) per step.
 *               - @c SPARSE_PIVOT_PARTIAL — search only the pivot column.
 *                 Faster (O(n) per step), Q remains identity.
 * @param tol    Pivot tolerance. If the best pivot candidate has |value| < tol,
 *               the matrix is declared singular.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if mat is NULL.
 * @return SPARSE_ERR_SHAPE if the matrix is not square.
 * @return SPARSE_ERR_SINGULAR if a zero (or below-tolerance) pivot is encountered.
 * @return SPARSE_ERR_ALLOC if memory allocation fails during fill-in.
 */
sparse_err_t sparse_lu_factor(SparseMatrix *mat, sparse_pivot_t pivot,
                              double tol);

/**
 * @brief Solve A*x = b using a previously factored matrix.
 *
 * Chains: row permutation → forward substitution (L) → backward
 * substitution (U) → inverse column permutation to produce x.
 *
 * @param mat  A matrix that has been factored by sparse_lu_factor().
 * @param b    Right-hand side vector of length n.
 * @param x    Solution vector of length n (overwritten). May alias b.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if any argument is NULL.
 */
sparse_err_t sparse_lu_solve(const SparseMatrix *mat,
                             const double *b, double *x);

/* ═══════════════════════════════════════════════════════════════════════════
 * Condition number estimation
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Estimate the 1-norm condition number of A from its LU factors.
 *
 * Uses Hager's algorithm (Hager 1984, refined by Higham 2000) to estimate
 * ||A^{-1}||_1 without forming the inverse. The condition estimate is:
 *
 *     condest = ||A||_1 * ||A^{-1}||_1_estimate
 *
 * **Algorithm outline (Hager/Higham 1-norm estimator):**
 *  1. Start with x = [1/n, ..., 1/n]
 *  2. Solve A*w = x  (using existing LU factors)
 *  3. Set xi = sign(w)
 *  4. Solve A^T*z = xi  (transpose solve)
 *  5. If ||z||_inf <= z^T * w, converged: ||A^{-1}||_1 ~ ||w||_1
 *  6. Otherwise, set x = e_j where j = argmax|z_j|, go to step 2
 *  7. Limit to 5 iterations to bound cost
 *
 * **Design decisions:**
 *  - Requires a transpose solve (A^T * z = b). This is implemented internally
 *    as sparse_lu_solve_transpose(), which solves Q^T U^T L^T P^T x = b
 *    using the same LU factors stored in the matrix.
 *  - The 1-norm ||A||_1 (max column sum) is computed from the original matrix.
 *    Since LU factorization overwrites A, the caller must pass the original
 *    matrix separately (via mat_orig), or we compute ||A||_1 from ||L||_1*||U||_1
 *    as an upper bound. We take the mat_orig approach for accuracy.
 *  - Returns SPARSE_ERR_BADARG if the matrix has not been factored (no
 *    row_perm allocated, or factor_norm not set).
 *
 * @param mat_orig  The original (unfactored) matrix A. Used to compute ||A||_1.
 * @param mat_lu    The LU-factored matrix (from sparse_lu_factor).
 * @param[out] condest  Pointer to receive the condition number estimate.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_BADARG if mat_lu has not been factored.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 */
sparse_err_t sparse_lu_condest(const SparseMatrix *mat_orig,
                               const SparseMatrix *mat_lu,
                               double *condest);

/**
 * @brief Solve A^T * x = b using a previously factored matrix.
 *
 * Given the factorization P*A*Q = L*U, solves:
 *     A^T * x = b
 * which is equivalent to:
 *     Q^T * U^T * L^T * P^T * x = b
 *
 * Steps:
 *  1. Apply Q permutation: qb[i] = b[col_perm[i]]
 *  2. Forward-substitute with U^T (lower triangular solve)
 *  3. Backward-substitute with L^T (upper triangular solve, unit diagonal)
 *  4. Apply P^{-1} permutation: x[i] = result[inv_row_perm[i]]
 *
 * @param mat  A matrix that has been factored by sparse_lu_factor().
 * @param b    Right-hand side vector of length n.
 * @param x    Solution vector of length n (overwritten). May alias b.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if any argument is NULL.
 */
sparse_err_t sparse_lu_solve_transpose(const SparseMatrix *mat,
                                       const double *b, double *x);

/* ═══════════════════════════════════════════════════════════════════════════
 * Individual solver phases (exposed for testing and advanced use)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Apply row permutation: pb[i] = b[row_perm[i]].
 *
 * Reorders the right-hand side vector according to the row permutation P.
 *
 * @param mat  Factored matrix (provides row_perm).
 * @param b    Input vector of length n.
 * @param pb   Output permuted vector of length n (overwritten).
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_apply_row_perm(const SparseMatrix *mat,
                                   const double *b, double *pb);

/**
 * @brief Apply inverse column permutation: x[i] = z[inv_col_perm[i]].
 *
 * Recovers the solution in the original column ordering after backward
 * substitution.
 *
 * @param mat  Factored matrix (provides inv_col_perm).
 * @param z    Input vector of length n (from backward substitution).
 * @param x    Output solution vector of length n (overwritten).
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_apply_inv_col_perm(const SparseMatrix *mat,
                                       const double *z, double *x);

/**
 * @brief Forward substitution: solve L*y = pb.
 *
 * L is the unit lower triangular factor stored in the strictly lower
 * triangle of the factored matrix.
 *
 * @param mat  Factored matrix.
 * @param pb   Permuted right-hand side (length n).
 * @param y    Output vector (length n, overwritten).
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_forward_sub(const SparseMatrix *mat,
                                const double *pb, double *y);

/**
 * @brief Backward substitution: solve U*z = y.
 *
 * U is the upper triangular factor stored on and above the diagonal
 * of the factored matrix.
 *
 * @param mat  Factored matrix.
 * @param y    Input vector from forward substitution (length n).
 * @param z    Output vector (length n, overwritten).
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_backward_sub(const SparseMatrix *mat,
                                 const double *y, double *z);

/* ═══════════════════════════════════════════════════════════════════════════
 * Iterative refinement
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Iterative refinement to improve solution accuracy.
 *
 * Computes the residual r = b - A*x using the original matrix, solves
 * A*d = r using the LU factorization, and updates x += d. Repeats until
 * the relative residual ||r|| / ||b|| drops below tol or max_iters is reached.
 *
 * @param mat_orig   The original (unfactored) matrix A.
 * @param mat_lu     The LU-factored matrix (from sparse_lu_factor).
 * @param b          Right-hand side vector of length n.
 * @param x          Solution vector of length n (modified in-place).
 * @param max_iters  Maximum number of refinement iterations.
 * @param tol        Convergence tolerance on relative residual.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if any argument is NULL.
 */
sparse_err_t sparse_lu_refine(const SparseMatrix *mat_orig,
                              const SparseMatrix *mat_lu,
                              const double *b, double *x,
                              int max_iters, double tol);

#endif /* SPARSE_LU_H */
