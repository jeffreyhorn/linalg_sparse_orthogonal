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
    sparse_reorder_t reorder; /**< Column reordering before QR (default: NONE).
                                   SPARSE_REORDER_COLAMD is recommended for
                                   unsymmetric matrices — operates directly on
                                   A's column structure without forming A^T*A. */
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
 * @param qr       The QR factorization of A.
 * @param b        Right-hand side vector of length m.
 * @param x        Output: solution vector of length n.
 * @param residual Output: residual norm ||b - Ax||_2 (may be NULL).
 * @return SPARSE_OK on success.
 *
 * @see sparse_qr_solve_minnorm for minimum-norm underdetermined solutions.
 * @see sparse_qr_rank_info for rank diagnostics.
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
 * The R diagonals are in decreasing order (due to column pivoting),
 * so the rank is the number of leading diagonals above the threshold.
 *
 * Use sparse_qr_diag_r() to inspect the R diagonal directly for
 * manual threshold selection.
 *
 * @param qr  The QR factorization.
 * @param tol Tolerance (0 for default: eps * max(m,n) * |R(0,0)|).
 *            The absolute threshold is tol * |R(0,0)|.
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

/* ═══════════════════════════════════════════════════════════════════════════
 * Rank-revealing diagnostics
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Extract the diagonal of the R factor.
 *
 * Writes R(i,i) for i = 0..min(m,n)-1 into diag[], in factorization
 * order (after column pivoting). Useful for manual rank determination
 * and condition estimation.
 *
 * @param qr    The QR factorization.
 * @param diag  Output array of length min(m,n). Must be pre-allocated.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if qr or diag is NULL.
 */
sparse_err_t sparse_qr_diag_r(const sparse_qr_t *qr, double *diag);

/**
 * @brief Rank diagnostics from a QR factorization.
 */
typedef struct {
    idx_t rank;         /**< Numerical rank (R diagonals above threshold) */
    idx_t k;            /**< min(m, n) — number of R diagonal entries */
    double r_max;       /**< Largest |R(i,i)| */
    double r_min;       /**< Smallest |R(i,i)| among the first rank entries */
    double condest;     /**< Quick condition estimate: r_max / r_min */
    int near_deficient; /**< 1 if r_min / r_max < 1e-8 (near rank-deficient) */
} sparse_qr_rank_info_t;

/**
 * @brief Compute rank diagnostics from a QR factorization.
 *
 * Analyzes the R diagonal to determine numerical rank, condition
 * estimate, and whether the matrix is near rank-deficient.
 *
 * The rank tolerance is: tol * |R(0,0)|. If tol <= 0, a default of
 * eps * max(m,n) is used (where eps ≈ 2.2e-16).
 *
 * **Threshold selection guidance:**
 * - For well-conditioned problems: tol = 0 (automatic) works well
 * - For noisy data: use tol ≈ noise_level / |R(0,0)|
 * - For problems with known rank: set tol between the rank-th and
 *   (rank+1)-th singular value ratios
 * - Machine epsilon (≈2.2e-16) times max(m,n) is a safe default
 *
 * @param qr    The QR factorization.
 * @param tol   Rank tolerance (0 for default: eps * max(m,n)).
 * @param info  Output rank diagnostics.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if qr or info is NULL.
 */
sparse_err_t sparse_qr_rank_info(const sparse_qr_t *qr, double tol, sparse_qr_rank_info_t *info);

/**
 * @brief Quick condition number estimate from R diagonal.
 *
 * Returns |R(0,0)| / |R(k-1,k-1)| where k = rank. This is a rough
 * estimate of the condition number (exact for diagonal matrices,
 * within a factor of sqrt(n) for general matrices).
 *
 * @param qr  The QR factorization.
 * @return Condition estimate, or -1.0 if qr is NULL or rank is 0.
 */
double sparse_qr_condest(const sparse_qr_t *qr);

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
 * @param A     The m×n matrix (not modified).
 * @param b     Right-hand side vector of length m.
 * @param x     Solution vector of length n (overwritten).
 * @param opts  QR options (reordering, etc.), or NULL for defaults.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A, b, or x is NULL.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 * @return SPARSE_ERR_SINGULAR if R has a near-zero diagonal.
 *
 * @see sparse_qr_solve for overdetermined least-squares.
 */
sparse_err_t sparse_qr_solve_minnorm(const SparseMatrix *A, const double *b, double *x,
                                     const sparse_qr_opts_t *opts);

/**
 * @brief Iterative refinement for minimum-norm solutions.
 *
 * Given an initial minimum-norm solution x (from sparse_qr_solve_minnorm),
 * improves accuracy by repeatedly computing the residual r = b - A*x and
 * solving for a minimum-norm correction dx. Stops when the residual stops
 * decreasing or max_refine iterations are reached.
 *
 * @param A           The m×n matrix (not modified).
 * @param b           Right-hand side vector of length m.
 * @param x           Solution vector of length n (modified in-place).
 * @param max_refine  Maximum number of refinement iterations.
 * @param residual    If non-NULL, receives the final ||b - A*x||_2.
 * @param opts        QR options for the correction solves, or NULL.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A, b, or x is NULL.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_qr_refine_minnorm(const SparseMatrix *A, const double *b, double *x,
                                      idx_t max_refine, double *residual,
                                      const sparse_qr_opts_t *opts);

#endif /* SPARSE_QR_H */
