#ifndef SPARSE_QR_H
#define SPARSE_QR_H

/**
 * @file sparse_qr.h
 * @brief Sparse QR factorization with column pivoting.
 *
 * Provides column-pivoted QR factorization A*P = Q*R using Householder
 * reflections, with least-squares solving, rank estimation, and null-space
 * extraction. Q is stored implicitly as Householder reflectors; use
 * sparse_qr_apply_q() to apply Q or Q^T without forming Q explicitly.
 *
 * Start with `examples/README.md` and `docs/solver_selection.md` for runnable
 * workflow guidance and evidence boundaries. This header owns the API-local
 * contracts: identity-permutation preconditions, borrowed inputs,
 * caller-owned output buffers, rank/residual diagnostics, and
 * sparse_qr_free() cleanup for factor objects.
 */

#include "sparse_matrix.h"

/* Options and factor object */

/**
 * @brief Options for QR factorization.
 */
typedef struct {
    sparse_reorder_t reorder; /**< Column reordering before QR (default: NONE).
                                   SPARSE_REORDER_COLAMD is recommended for
                                   unsymmetric matrices — operates directly on
                                   A's column structure without forming A^T*A.
                                   SPARSE_REORDER_AMD / _RCM / _ND are also
                                   accepted: they form A^T*A and reorder
                                   symmetrically; ND is best on 2D / 3D PDE
                                   meshes. */
    int economy;              /**< When nonzero and m > n, compute economy (thin) QR:
                                   form_q produces m×n instead of m×m. Has no effect
                                   when m <= n (Q is already m×m = m×k where k=min(m,n)).
                                   (default: 0 = full QR) */
    int sparse_mode;          /**< When nonzero, use column-by-column Householder
                                   application instead of O(m*n) dense workspace.
                                   Uses O(m) working memory per active column and
                                   the same public factorization contract as the
                                   default path. (default: 0) */
    /** Optional progress / cancellation callback.  Invoked at the top of each
     *  Householder column-elimination iteration with `phase = "qr_factor"`,
     *  `step = k`, `total = min(m, n)`.  Return 0 to continue; non-zero
     *  cancels with `SPARSE_ERR_CANCELLED` after freeing intermediate state.
     *  NULL (default) disables callback work.  Trailing field for
     *  designated-init compatibility. */
    sparse_progress_cb_t progress_cb;
    /** Opaque context pointer passed through unchanged to
     *  `progress_cb`.  Ignored when `progress_cb == NULL`. */
    void *progress_user;
} sparse_qr_opts_t;

/**
 * @brief QR factorization data.
 *
 * Stores R (upper triangular), Householder reflectors (v, beta) for Q,
 * column permutation, and rank information.
 *
 * Callers own the sparse_qr_t object itself. Successful factorization stores
 * owned factor data inside the object, and callers must release that data with
 * sparse_qr_free(). Call sparse_qr_free() before reusing a populated
 * sparse_qr_t for a new factorization; factor functions overwrite the struct
 * without freeing prior contents. sparse_qr_free() is safe on a zeroed struct.
 */
typedef struct {
    SparseMatrix *R;             /**< Upper triangular factor (min(m,n) × n after permutation) */
    sparse_scalar_t *betas;      /**< Householder scalars beta_k, length min(m,n) */
    sparse_scalar_t **v_vectors; /**< Householder vectors v_k, each length m-k
                                      (stored from diagonal down) */
    idx_t *col_perm;             /**< Column permutation: col_perm[k] = original column index */
    idx_t m;                     /**< Number of rows of original A */
    idx_t n;                     /**< Number of columns of original A */
    idx_t rank;                  /**< Numerical rank (set during factorization) */
    int economy;                 /**< Nonzero if economy (thin Q) was requested.
                                      A thin Q is only formed when m > n; when m <= n
                                      this flag has no effect on the shape of Q. */
} sparse_qr_t;

/* Factorization and lifecycle */

/**
 * @brief Compute column-pivoted QR factorization: A*P = Q*R.
 *
 * @pre A must have identity permutations (not previously factored, pivoted,
 *      or reordered).  A is not modified.
 *
 * @param A   Borrowed matrix to factor (not modified). May be rectangular (m×n).
 * @param qr  Output factor object. Must be freed with sparse_qr_free() after success.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or qr is NULL.
 * @return SPARSE_ERR_BADARG if A has non-identity permutations.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_qr_factor(const SparseMatrix *A, sparse_qr_t *qr);

/**
 * @brief Compute QR factorization with options (e.g., fill-reducing reordering).
 *
 * @param A    Borrowed matrix to factor (not modified). Must have identity permutations.
 * @param opts Factorization options (NULL for defaults).
 * @param qr   Output factor object. Must be freed with sparse_qr_free() after success.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or qr is NULL.
 * @return SPARSE_ERR_BADARG if A has non-identity permutations.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 * @return SPARSE_ERR_CANCELLED if the progress callback returns nonzero.
 */
sparse_err_t sparse_qr_factor_opts(const SparseMatrix *A, const sparse_qr_opts_t *opts,
                                   sparse_qr_t *qr);

/**
 * @brief Free QR factorization data.
 *
 * @param qr  QR factors to free. NULL and zeroed structs are safe.
 */
void sparse_qr_free(sparse_qr_t *qr);

/* Q operations */

/**
 * @brief Apply Q or Q^T to a vector.
 *
 * Computes y = Q*x or y = Q^T*x using the stored Householder reflectors,
 * without forming Q explicitly.
 *
 * @param qr        Borrowed QR factorization.
 * @param transpose 0 for Q*x, 1 for Q^T*x.
 * @param x         Input vector of length m.
 * @param y         Caller-owned output vector of length m (may alias x for in-place).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if qr, x, or y is NULL.
 */
sparse_err_t sparse_qr_apply_q(const sparse_qr_t *qr, int transpose, const sparse_scalar_t *x,
                               sparse_scalar_t *y);

/**
 * @brief Explicitly form the Q matrix (for testing/diagnostics).
 *
 * For full QR (economy=0): forms Q as a dense m×m orthogonal matrix.
 * Caller allocates m*m dense scalars.
 *
 * For economy QR (economy=1): forms the thin Q as a dense m×k matrix
 * with orthonormal columns, where k = min(m, n). Caller allocates m*k dense
 * scalars.
 *
 * @param qr  Borrowed QR factorization.
 * @param Q   Caller-owned dense output matrix in column-major order.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if qr or Q is NULL.
 * @return SPARSE_ERR_ALLOC if output-size arithmetic overflows.
 */
sparse_err_t sparse_qr_form_q(const sparse_qr_t *qr, sparse_scalar_t *Q);

/* Solve operations */

/**
 * @brief Solve the least-squares problem min ||Ax - b||_2.
 *
 * **Overdetermined (m > n):** Computes the least-squares solution that
 * minimizes ||Ax - b||_2 via back-substitution in R.
 *
 * **Square (m == n):** Computes the direct solution A*x = b.
 *
 * **Underdetermined (m < n):** Computes a basic solution by solving for
 * the rank leading components and setting remaining free components to
 * zero in the column-permuted coordinate system. This is NOT the
 * minimum-norm solution. For the minimum 2-norm solution, use
 * sparse_qr_solve_minnorm() instead.
 *
 * @note For rank-deficient systems, components corresponding to
 *       near-zero R diagonals are set to zero. Use sparse_qr_rank()
 *       or sparse_qr_rank_info() to inspect the effective rank, and
 *       sparse_qr_diag_r() for manual threshold selection.
 *
 * @param qr       Borrowed QR factorization of A.
 * @param b        Right-hand side vector of length m.
 * @param x        Caller-owned output solution vector of length n.
 * @param residual Optional caller-owned output for ||b - Ax||_2 (may be NULL).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if qr, b, or x is NULL, or if qr does not contain a valid factorization.
 * @return SPARSE_ERR_ALLOC if temporary workspace allocation fails.
 *
 * @see sparse_qr_solve_minnorm for minimum-norm underdetermined solutions.
 * @see sparse_qr_rank_info for rank diagnostics.
 */
sparse_err_t sparse_qr_solve(const sparse_qr_t *qr, const sparse_scalar_t *b, sparse_scalar_t *x,
                             sparse_scalar_t *residual);

/**
 * @brief Iterative refinement for QR least-squares solutions.
 *
 * Improves an existing QR solution by repeatedly computing the residual
 * r = b - A*x and solving for a correction via the existing QR factorization.
 * Useful for reducing the residual on ill-conditioned systems.
 *
 * @param qr         Borrowed QR factorization of A.
 * @param A          Borrowed original matrix (for computing residuals).
 * @param b          Right-hand side vector of length m.
 * @param x          Caller-owned solution vector. On entry: initial solution
 *                   (from sparse_qr_solve).
 *                   On exit: refined solution. Length n.
 * @param max_refine Maximum number of refinement iterations. 0 = just compute residual.
 * @param residual   Optional caller-owned output for final ||b - Ax||_2 (may be NULL).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any required argument is NULL.
 * @return SPARSE_ERR_SHAPE if A dimensions don't match the QR factorization.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_qr_refine(const sparse_qr_t *qr, const SparseMatrix *A,
                              const sparse_scalar_t *b, sparse_scalar_t *x, idx_t max_refine,
                              sparse_scalar_t *residual);

/**
 * @brief Compute the minimum 2-norm solution for underdetermined systems.
 *
 * For an underdetermined system A*x = b where m < n, computes the
 * solution x with minimum ||x||_2 among all solutions. Uses QR
 * factorization of A^T: factor A^T = Q*R*P^T, then solve via
 * x = Q * R^{-T} * P^T * b.
 *
 * For overdetermined systems (m >= n), falls back to standard
 * least-squares via QR.
 *
 * This routine builds temporary QR factorizations internally and releases
 * them before returning. If opts is non-NULL, its factorization options,
 * including progress cancellation, apply to those internal factorizations.
 *
 * @param A     Borrowed m×n matrix (not modified).
 * @param b     Right-hand side vector of length m.
 * @param x     Caller-owned solution vector of length n (overwritten).
 * @param opts  QR options (reordering, etc.), or NULL for defaults.
 *
 * @return SPARSE_OK on success. Near-zero R diagonals are handled by
 *         zeroing the corresponding components (not treated as an error).
 * @return SPARSE_ERR_NULL if A, b, or x is NULL.
 * @return SPARSE_ERR_BADARG if A has non-identity row/col permutations.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 * @return SPARSE_ERR_CANCELLED if opts provides a progress callback that cancels.
 *
 * @see sparse_qr_solve for overdetermined least-squares.
 */
sparse_err_t sparse_qr_solve_minnorm(const SparseMatrix *A, const sparse_scalar_t *b,
                                     sparse_scalar_t *x, const sparse_qr_opts_t *opts);

/**
 * @brief Iterative refinement for minimum-norm solutions.
 *
 * Given an initial minimum-norm solution x (from sparse_qr_solve_minnorm),
 * improves accuracy by repeatedly computing the residual r = b - A*x and
 * solving for a minimum-norm correction dx. Stops when the residual stops
 * decreasing or max_refine iterations are reached.
 *
 * @note Each refinement iteration calls sparse_qr_solve_minnorm(), which
 * rebuilds A^T and computes a full QR factorization. This makes refinement
 * O(max_refine * cost(QR(A^T))). For large problems, keep max_refine small
 * (1-3 iterations typically suffice).
 *
 * The correction solve uses sparse_qr_solve_minnorm(), so opts has the same
 * internal-factorization meaning and cancellation behavior here.
 *
 * @param A           Borrowed m×n matrix (not modified).
 * @param b           Right-hand side vector of length m.
 * @param x           Caller-owned solution vector of length n (modified in-place).
 * @param max_refine  Maximum number of refinement iterations.
 * @param residual    Optional caller-owned output for final ||b - A*x||_2.
 * @param opts        QR options for the correction solves, or NULL.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A, b, or x is NULL.
 * @return SPARSE_ERR_BADARG if A has non-identity row/col permutations
 *         (propagated from sparse_qr_solve_minnorm).
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 * @return SPARSE_ERR_CANCELLED if opts provides a progress callback that cancels.
 */
sparse_err_t sparse_qr_refine_minnorm(const SparseMatrix *A, const sparse_scalar_t *b,
                                      sparse_scalar_t *x, idx_t max_refine,
                                      sparse_scalar_t *residual, const sparse_qr_opts_t *opts);

/* Rank, nullspace, and diagnostics */

/**
 * @brief Estimate numerical rank from QR factorization.
 *
 * Counts the leading R diagonal entries that exceed the effective absolute
 * threshold. Column pivoting orders the diagonal entries used by this
 * diagnostic, but this is a QR-local rank estimate, not a global rank policy.
 *
 * When tol > 0, the effective absolute threshold is tol * |R(0,0)|.
 * When tol <= 0, the default threshold is eps * max(m,n) * |R(0,0)|, where
 * eps is machine epsilon.
 *
 * Use sparse_qr_diag_r() to inspect the R diagonal directly for
 * manual threshold selection.
 *
 * @param qr  Borrowed QR factorization.
 * @param tol Relative tolerance (0 or negative for default).
 * @return The estimated numerical rank.
 *
 * @see sparse_qr_diag_r for R diagonal extraction.
 * @see sparse_qr_rank_info for comprehensive rank diagnostics.
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
 * @param qr       Borrowed QR factorization.
 * @param tol      Tolerance for rank determination (same as sparse_qr_rank).
 * @param basis    Optional caller-owned output for null-space basis vectors
 *                 (n × null_dim, column-major), expressed in the original
 *                 column ordering. When non-NULL and null_dim > 0, caller
 *                 allocates n * (n - rank) dense scalars.
 * @param null_dim Caller-owned output for null-space dimension (n - rank).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if qr or null_dim is NULL, or if qr has no factor data.
 * @return SPARSE_ERR_ALLOC if temporary workspace allocation fails.
 */
sparse_err_t sparse_qr_nullspace(const sparse_qr_t *qr, double tol, sparse_scalar_t *basis,
                                 idx_t *null_dim);

/**
 * @brief Extract the diagonal of the R factor.
 *
 * Writes R(i,i) for i = 0..min(m,n)-1 into diag[], in factorization
 * order (after column pivoting). Useful for manual rank determination
 * and condition estimation.
 *
 * @param qr    Borrowed QR factorization. Must contain a valid R factor.
 * @param diag  Caller-owned output array of length min(m,n).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if qr or diag is NULL.
 * @return SPARSE_ERR_BADARG if qr does not contain a valid factorization
 *         (qr->R is NULL).
 */
sparse_err_t sparse_qr_diag_r(const sparse_qr_t *qr, sparse_scalar_t *diag);

/**
 * @brief Rank diagnostics from a QR factorization.
 */
typedef struct {
    idx_t rank;            /**< Numerical rank (R diagonals above effective threshold) */
    idx_t k;               /**< min(m, n) — number of R diagonal entries */
    sparse_scalar_t r_max; /**< Largest |R(i,i)| */
    sparse_scalar_t r_min; /**< Smallest |R(i,i)| among the first rank entries */
    double condest;        /**< Quick R-diagonal condition estimate: r_max / r_min */
    int near_deficient;    /**< 1 if r_min / r_max < 1e-8 (near rank-deficient) */
} sparse_qr_rank_info_t;

/**
 * @brief Compute rank diagnostics from a QR factorization.
 *
 * Analyzes the R diagonal to determine numerical rank, condition
 * estimate, and whether the matrix is near rank-deficient.
 *
 * When tol > 0, the effective absolute threshold is tol * |R(0,0)|.
 * When tol <= 0, the default threshold is eps * max(m,n) * |R(0,0)|, where
 * eps is machine epsilon.
 *
 * @note The rank computed here may differ from qr->rank (set during
 * factorization with a different internal threshold). Use this function
 * for post-factorization rank analysis; qr->rank controls which
 * components sparse_qr_solve() sets to zero.
 *
 * Use tol = 0 for the default threshold. For noisy data or known-rank
 * problems, callers may choose a problem-specific tolerance from the R
 * diagonal; see `docs/solver_selection.md#qr-evidence-boundary` for the
 * bounded QR evidence interpretation.
 *
 * @param qr    Borrowed QR factorization.
 * @param tol   Relative rank tolerance (0 or negative for default).
 * @param info  Caller-owned output rank diagnostics.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if qr or info is NULL.
 * @return SPARSE_ERR_BADARG if qr does not contain a valid factorization
 *         (qr->R is NULL).
 */
sparse_err_t sparse_qr_rank_info(const sparse_qr_t *qr, double tol, sparse_qr_rank_info_t *info);

/**
 * @brief Quick condition number estimate from R diagonal.
 *
 * Returns |R(0,0)| / |R(k-1,k-1)| where k = rank. This is a rough
 * R-diagonal diagnostic estimate, not a full condition-number guarantee.
 *
 * @param qr  Borrowed QR factorization.
 * @return Finite condition estimate (>= 1.0) in the normal case.
 * @return INFINITY if the smallest R diagonal in the rank-determined
 *         block is zero (numerically singular).
 * @return -1.0 if qr is NULL, unfactored, or rank is 0.
 */
double sparse_qr_condest(const sparse_qr_t *qr);

#endif /* SPARSE_QR_H */
