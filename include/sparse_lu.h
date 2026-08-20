#ifndef SPARSE_LU_H
#define SPARSE_LU_H

/**
 * @file sparse_lu.h
 * @brief LU decomposition and linear system solving for sparse matrices.
 *
 * Provides in-place LU factorization with row and column pivoting
 * (P·A·Q = L·U), forward/backward substitution, and iterative refinement.
 *
 * LU factorization overwrites the caller-owned matrix with the factors:
 * L occupies the strictly lower triangle with an implicit unit diagonal, and
 * U occupies the upper triangle including the diagonal. Row and column
 * permutations are stored on the matrix and used by the solve routines.
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
 *
 * For stable-pattern repeated runs, prefer the shared direct lifecycle in
 * `sparse_analysis.h`: analyze once, factor/refactor numerically, solve, then
 * free the analysis and factor objects. The APIs in this header are the
 * one-shot LU path for callers that factor a fresh matrix or a copy.
 */

#include "sparse_matrix.h"

/* Options */

/**
 * @brief Options for LU factorization with optional fill-reducing reordering.
 *
 * Use with sparse_lu_factor_opts(). The reordering permutation is computed
 * and applied automatically; sparse_lu_solve() will detect it and unpermute
 * the solution transparently.
 *
 * @code
 *   sparse_lu_opts_t opts = {
 *       .pivot = SPARSE_PIVOT_PARTIAL,
 *       .reorder = SPARSE_REORDER_AMD,
 *       .tol = 1e-12,
 *   };
 *   sparse_lu_factor_opts(A, &opts);
 *   sparse_lu_solve(A, b, x);  // reorder/unpermute handled automatically
 * @endcode
 */
typedef struct {
    sparse_pivot_t pivot;     /**< Pivoting strategy. */
    sparse_reorder_t reorder; /**< Optional fill-reducing reordering (NONE, RCM,
                                   AMD, or ND). See sparse_reorder.h for
                                   reorder-mode details. */
    double tol;               /**< Absolute pivot tolerance used during elimination. */
    /** Optional progress/cancellation callback.
     *
     *  The callback is invoked at the top of each column-elimination
     *  iteration with `phase = "lu_factor"`, `step = k`, and `total = n`.
     *  Return 0 to continue. Returning non-zero cancels the factorization and
     *  returns `SPARSE_ERR_CANCELLED`.
     *
     *  Cancellation after completed steps may leave the one-shot matrix
     *  partially eliminated. Cancellation at step 0 restores the no-reorder
     *  compatibility state before returning. Reordered callback paths factor a
     *  temporary working copy and publish it back only on success, so a
     *  cancelled reordered call leaves the caller-owned matrix in its original
     *  coordinate space. NULL disables callbacks. This trailing field preserves
     *  designated-initializer compatibility. */
    sparse_progress_cb_t progress_cb;
    /** Opaque context pointer passed unchanged to `progress_cb`; ignored when
     *  `progress_cb == NULL`. */
    void *progress_user;
} sparse_lu_opts_t;

/* Factorization */

/**
 * @brief Compute LU factorization with options including fill-reducing reordering.
 *
 * This is a one-shot entry point. Call it on a fresh matrix, or on a fresh
 * `sparse_copy()` of the original coefficients, not on a matrix that has
 * already been factored, pivoted, or reordered by an earlier direct-solver
 * call. For stable-pattern repeated runs, use the shared direct lifecycle in
 * `sparse_analysis.h`.
 *
 * If opts->reorder != SPARSE_REORDER_NONE, the matrix is symmetrically
 * permuted before factorization. The reordering permutation is stored in
 * the matrix so that sparse_lu_solve() can automatically unpermute the
 * solution. Invalid pivot or reorder enums are rejected before reorder or
 * factor mutation begins. Some reordered paths factor a temporary working copy
 * and publish it back to @p mat only on success, so failed or cancelled calls
 * do not strand the caller matrix in an intermediate reordered state.
 *
 * @param mat   The matrix to factor; modified in-place on success.
 * @param opts  Factorization options. Must be non-NULL.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if @p mat or @p opts is NULL.
 * @return SPARSE_ERR_SHAPE if @p mat is not square.
 * @return SPARSE_ERR_BADARG if @p opts->pivot or @p opts->reorder is invalid,
 *         or if @p mat is already factored/pivoted/reordered.
 * @return SPARSE_ERR_SINGULAR if factorization encounters a below-tolerance pivot.
 * @return SPARSE_ERR_ALLOC if allocation fails.
 * @return SPARSE_ERR_CANCELLED if @p opts->progress_cb cancels factorization.
 */
sparse_err_t sparse_lu_factor_opts(SparseMatrix *mat, const sparse_lu_opts_t *opts);

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
 * For repeated solves on matrices that keep the same sparsity pattern but
 * change values, use the shared analyze/factor/refactor path in
 * `sparse_analysis.h` instead of repeatedly re-entering this one-shot API.
 *
 * @pre mat must still be in its original row/column state. This one-shot
 *      entry point rejects matrices that have already been factored, pivoted,
 *      or reordered. Use a fresh matrix or `sparse_copy()` of the original.
 * @pre mat must not be needed after factorization; use sparse_copy() first to
 *      preserve the original. The matrix is overwritten with L and U.
 *
 * @param mat    The matrix to factor (modified in-place). Must be square.
 * @param pivot  Pivoting strategy:
 *               - @c SPARSE_PIVOT_COMPLETE — search entire remaining submatrix
 *                 for the largest element. Better stability, O(n^2) per step.
 *               - @c SPARSE_PIVOT_PARTIAL — search only the pivot column.
 *                 Faster (O(n) per step), Q remains identity.
 * @param tol    Pivot tolerance. If the best pivot candidate has |value| < tol,
 *               the matrix is declared singular.
 *
 * @note **Tolerance semantics:** The @p tol parameter is an absolute pivot
 *       threshold used during elimination. Separately, the backward
 *       substitution (solve) phase uses a norm-relative singularity check:
 *       a diagonal U(i,i) is rejected if |U(i,i)| < SPARSE_DROP_TOL ×
 *       ||A||_inf, where ||A||_inf was cached at factorization time.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if mat is NULL.
 * @return SPARSE_ERR_SHAPE if the matrix is not square.
 * @return SPARSE_ERR_BADARG if @p pivot is invalid, or if @p mat has already
 *         been factored, pivoted, or reordered.
 * @return SPARSE_ERR_SINGULAR if a zero (or below-tolerance) pivot is encountered.
 * @return SPARSE_ERR_ALLOC if memory allocation fails during fill-in.
 *
 * @par Thread safety: Mutates mat. Not safe to call concurrently on the same matrix.
 *               Safe to call concurrently on different matrices.
 */
sparse_err_t sparse_lu_factor(SparseMatrix *mat, sparse_pivot_t pivot, double tol);

/* Solves */

/**
 * @brief Solve A*x = b using a previously factored matrix.
 *
 * Chains: row permutation → forward substitution (L) → backward
 * substitution (U) → inverse column permutation to produce x.
 *
 * @param mat  A matrix with LU factors produced by sparse_lu_factor() or
 *             sparse_lu_factor_opts().
 * @param b    Right-hand side vector of length n.
 * @param x    Solution vector of length n; overwritten and may alias @p b.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_BADARG if mat has not been factored.
 * @return SPARSE_ERR_ALLOC if temporary workspace allocation fails.
 * @return SPARSE_ERR_SINGULAR if a near-zero U diagonal is encountered.
 *
 * @par Thread safety: Read-only on mat. Safe to call concurrently on the same
 *               factored matrix with different b/x vectors.
 */
sparse_err_t sparse_lu_solve(const SparseMatrix *mat, const double *b, double *x);

/**
 * @brief Solve A*X = B for multiple right-hand side vectors simultaneously.
 *
 * Performs forward and backward substitution for nrhs vectors at once,
 * amortizing sparse pattern traversal across all RHS vectors for better
 * cache efficiency.
 *
 * @param mat   A matrix with LU factors produced by sparse_lu_factor() or
 *              sparse_lu_factor_opts().
 * @param B     Right-hand side matrix of size n × nrhs (column-major:
 *              B[i + n*k] = B(i,k)). Not modified. Must be non-NULL even
 *              if @p nrhs is 0.
 * @param nrhs  Number of right-hand side vectors. If 0, this function is a
 *              no-op and returns SPARSE_OK, but all pointer arguments must
 *              still be non-NULL.
 * @param X     Solution matrix of size n × nrhs (column-major, overwritten).
 *              Must be non-NULL even if @p nrhs is 0.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any pointer is NULL, including when @p nrhs is 0.
 * @return SPARSE_ERR_BADARG if mat has not been factored, or if @p nrhs is negative.
 * @return SPARSE_ERR_SINGULAR if a zero diagonal in U is encountered.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 */
sparse_err_t sparse_lu_solve_block(const SparseMatrix *mat, const double *B, idx_t nrhs, double *X);

/* Conditioning and transpose solves */

/**
 * @brief Estimate the 1-norm condition number of A from its LU factors.
 *
 * Uses a bounded Hager/Higham-style 1-norm estimator to estimate
 * ||A^{-1}||_1 without forming the inverse. The reported estimate is:
 *
 *     condest = ||A||_1 * ||A^{-1}||_1_estimate
 *
 * The estimator repeatedly solves with the existing LU factors and the
 * transposed system through sparse_lu_solve_transpose(). Because LU
 * factorization overwrites A, callers must pass the original unfactored
 * matrix separately through @p mat_orig so the implementation can compute
 * ||A||_1 from the original coefficients.
 *
 * @param mat_orig  The original (unfactored) matrix A. Used to compute ||A||_1.
 * @param mat_lu    The LU-factored matrix from sparse_lu_factor() or
 *                  sparse_lu_factor_opts().
 * @param[out] condest  Pointer to receive the condition number estimate.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_SHAPE if matrix dimensions are incompatible.
 * @return SPARSE_ERR_BADARG if mat_lu has not been factored.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 */
sparse_err_t sparse_lu_condest(const SparseMatrix *mat_orig, const SparseMatrix *mat_lu,
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
 * @param mat  A matrix with LU factors produced by sparse_lu_factor() or
 *             sparse_lu_factor_opts().
 * @param b    Right-hand side vector of length n.
 * @param x    Solution vector of length n; overwritten and may alias @p b.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_BADARG if mat has not been factored.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 * @return SPARSE_ERR_SINGULAR if a near-zero pivot is encountered during U^T solve.
 */
sparse_err_t sparse_lu_solve_transpose(const SparseMatrix *mat, const double *b, double *x);

/* Advanced solver phases */

/**
 * @brief Apply row permutation: pb[i] = b[row_perm[i]].
 *
 * Reorders the right-hand side vector according to the row permutation P.
 *
 * @param mat  Factored matrix that provides row_perm.
 * @param b    Input vector of length n.
 * @param pb   Output permuted vector of length n (overwritten).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 */
sparse_err_t sparse_apply_row_perm(const SparseMatrix *mat, const double *b, double *pb);

/**
 * @brief Apply inverse column permutation: x[i] = z[inv_col_perm[i]].
 *
 * Recovers the solution in the original column ordering after backward
 * substitution.
 *
 * @param mat  Factored matrix that provides inv_col_perm.
 * @param z    Input vector of length n (from backward substitution).
 * @param x    Output solution vector of length n (overwritten).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 */
sparse_err_t sparse_apply_inv_col_perm(const SparseMatrix *mat, const double *z, double *x);

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
 * @return SPARSE_ERR_NULL if any argument is NULL.
 */
sparse_err_t sparse_forward_sub(const SparseMatrix *mat, const double *pb, double *y);

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
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_SINGULAR if a near-zero U diagonal is encountered.
 */
sparse_err_t sparse_backward_sub(const SparseMatrix *mat, const double *y, double *z);

/* Refinement */

/**
 * @brief Iterative refinement to improve solution accuracy.
 *
 * Computes the residual r = b - A*x using the original matrix, solves
 * A*d = r using the LU factorization, and updates x += d. Repeats until
 * the relative residual ||r|| / ||b|| drops below tol or max_iters is reached.
 *
 * @param mat_orig   The original (unfactored) matrix A.
 * @param mat_lu     The LU-factored matrix from sparse_lu_factor().
 * @param b          Right-hand side vector of length n.
 * @param x          Solution vector of length n (modified in-place).
 * @param max_iters  Maximum number of refinement iterations.
 * @param tol        Convergence tolerance on relative residual.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_BADARG if @p mat_lu has not been factored.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 * @return Any error propagated from sparse_lu_solve().
 */
sparse_err_t sparse_lu_refine(const SparseMatrix *mat_orig, const SparseMatrix *mat_lu,
                              const double *b, double *x, int max_iters, double tol);

#endif /* SPARSE_LU_H */
