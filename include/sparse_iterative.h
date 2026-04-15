#ifndef SPARSE_ITERATIVE_H
#define SPARSE_ITERATIVE_H

/**
 * @file sparse_iterative.h
 * @brief Krylov subspace iterative solvers for sparse linear systems.
 *
 * Provides Conjugate Gradient (CG) for symmetric positive-definite systems
 * and restarted GMRES for general (unsymmetric) systems, both with optional
 * preconditioning via a user-supplied callback.
 *
 * **CG usage pattern:**
 * @code
 *   SparseMatrix *A = sparse_create(n, n);
 *   // ... populate A with SPD entries ...
 *   double *b = ..., *x = calloc(n, sizeof(double));
 *   sparse_iter_opts_t opts = { .max_iter = 1000, .tol = 1e-10 };
 *   sparse_iter_result_t result;
 *   sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);
 *   printf("converged in %d iterations, residual = %e\n",
 *          result.iterations, result.residual_norm);
 * @endcode
 *
 * **GMRES usage pattern:**
 * @code
 *   sparse_gmres_opts_t opts = { .max_iter = 500, .restart = 30, .tol = 1e-10 };
 *   sparse_iter_result_t result;
 *   sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);
 * @endcode
 *
 * **Preconditioned solve:**
 * @code
 *   // ILU preconditioner (see sparse_ilu.h)
 *   sparse_ilu_t ilu;
 *   sparse_ilu_factor(A, &ilu);
 *   sparse_solve_gmres(A, b, x, &opts, sparse_ilu_precond, &ilu, &result);
 * @endcode
 */

#include "sparse_matrix.h"

/**
 * @par Breakdown behavior summary
 *
 * | Solver   | Condition               | Detection                | Response |
 * |----------|-------------------------|--------------------------|--------------------------------|
 * | CG       | p^T*A*p ≈ 0             | Threshold on |p^T*A*p|   | Stop, set breakdown=1 | | CG |
 * r^T*z ≈ 0               | Threshold on |r^T*z|     | Stop, set breakdown=1          | | GMRES |
 * H(j+1,j) ≈ 0            | Threshold on H(j+1,j)   | Lucky breakdown: extract exact | |          |
 * (lucky breakdown)        |                          | solution, set breakdown=1,     | | | | |
 * converged=1                    | | MINRES   | beta_{k+1} ≈ 0          | Threshold on beta_new |
 * Lanczos breakdown: Krylov      | |          | (Lanczos breakdown)      | | exhausted, set
 * breakdown=1     | | MINRES   | gamma ≈ 0               | Threshold on gamma       | QR breakdown,
 * set breakdown=1  | | BiCGSTAB | rho = r_hat^T*r ≈ 0     | Threshold on |rho|       | Stop, set
 * breakdown=1          | | BiCGSTAB | r_hat^T*v ≈ 0           | Threshold on |r_hat^T*v| | Stop,
 * set breakdown=1          | | BiCGSTAB | t^T*t ≈ 0               | Threshold on t^T*t       |
 * Stop, set breakdown=1          | | BiCGSTAB | omega ≈ 0 (near-zero)   | |omega| < 1e-15*|alpha|
 * | Accept half-step, restart      |
 *
 * All threshold checks use sparse_rel_tol(0, DROP_TOL) ≈ DBL_MIN*100.
 * For GMRES lucky breakdown, breakdown=1 AND converged=1 indicates success.
 * For all other breakdowns, breakdown=1 AND converged=0 indicates failure.
 */

/**
 * @brief Progress information passed to the verbose callback.
 *
 * Populated by the solver at each iteration (or restart boundary for GMRES)
 * and passed to the user callback if one is provided.
 */
typedef struct {
    idx_t iteration;      /**< Current iteration number (0-based) */
    double residual_norm; /**< Current relative residual norm ||r||/||b|| */
    const char *solver;   /**< Solver name ("CG", "GMRES", "MINRES", "BiCGSTAB") */
} sparse_iter_progress_t;

/**
 * @brief Verbose callback type for custom progress reporting.
 *
 * If set in the options struct, the solver calls this function at each
 * iteration instead of printing to stderr. When NULL and verbose is set,
 * the default stderr printing is used.
 *
 * @param progress  Current iteration progress information.
 * @param ctx       User-supplied context pointer.
 */
typedef void (*sparse_iter_callback_fn)(const sparse_iter_progress_t *progress, void *ctx);

/**
 * @brief Options for the Conjugate Gradient solver.
 *
 * Pass NULL to sparse_solve_cg() to use defaults:
 * max_iter = 1000, tol = 1e-10, verbose = 0.
 */
typedef struct {
    idx_t max_iter; /**< Maximum number of CG iterations (default: 1000) */
    double tol;     /**< Convergence tolerance on relative residual ||r||/||b|| (default: 1e-10) */
    int verbose;    /**< If nonzero, print iteration log to stderr (default: 0) */
    idx_t stagnation_window;          /**< Stagnation detection window size. If > 0, the solver
                                           tracks the last N residual norms and declares stagnation
                                           if max/min in the window differ by less than 1%.
                                           0 = disabled (default). Typical value: 10-20. */
    double *residual_history;         /**< Caller-allocated array for per-iteration residual norms.
                                           If non-NULL, the solver stores ||r_k||/||b|| at index k.
                                           NULL = no recording (default). */
    idx_t residual_history_len;       /**< Capacity of the residual_history array. The solver
                                           writes at most this many entries. */
    sparse_iter_callback_fn callback; /**< Verbose callback. If non-NULL, called each iteration
                                           instead of fprintf(stderr). NULL = use default verbose
                                           behavior (default). */
    void *callback_ctx;               /**< Context pointer passed to callback. */
} sparse_iter_opts_t;

/**
 * @brief Preconditioning side for GMRES.
 */
typedef enum {
    SPARSE_PRECOND_LEFT = 0,  /**< Left preconditioning: solve M^{-1}Ax = M^{-1}b */
    SPARSE_PRECOND_RIGHT = 1, /**< Right preconditioning: solve AM^{-1}y = b, x = M^{-1}y */
} sparse_precond_side_t;

/**
 * @brief Options for the GMRES solver.
 *
 * Pass NULL to sparse_solve_gmres() to use defaults:
 * max_iter = 1000, restart = 30, tol = 1e-10, verbose = 0, precond_side = LEFT.
 */
typedef struct {
    idx_t max_iter; /**< Maximum total number of GMRES iterations (default: 1000) */
    idx_t restart;  /**< Restart parameter k for GMRES(k) (default: 30) */
    double tol;     /**< Convergence tolerance on relative residual (default: 1e-10) */
    int verbose;    /**< If nonzero, print iteration log to stderr (default: 0) */
    sparse_precond_side_t precond_side; /**< Left or right preconditioning (default: LEFT) */
    idx_t stagnation_window;            /**< Stagnation detection window size (across restarts).
                                             0 = disabled (default). See sparse_iter_opts_t. */
    double *residual_history;           /**< See sparse_iter_opts_t::residual_history. */
    idx_t residual_history_len;         /**< See sparse_iter_opts_t::residual_history_len. */
    sparse_iter_callback_fn callback;   /**< See sparse_iter_opts_t::callback. */
    void *callback_ctx;                 /**< See sparse_iter_opts_t::callback_ctx. */
} sparse_gmres_opts_t;

/**
 * @brief Result information from an iterative solve.
 *
 * Populated by sparse_solve_cg() and sparse_solve_gmres() on return.
 * Pass NULL if result information is not needed.
 */
typedef struct {
    idx_t iterations;             /**< Number of iterations performed */
    double residual_norm;         /**< Final true relative residual norm ||b - A*x|| / ||b|| */
    int converged;                /**< Nonzero if solver converged within tolerance */
    int stagnated;                /**< Nonzero if stagnation was detected (residual stopped
                                       decreasing over the stagnation window). Only set when
                                       stagnation_window > 0 in opts. */
    idx_t residual_history_count; /**< Number of entries written to residual_history.
                                       0 if residual_history was NULL. */
    int breakdown;                /**< Nonzero if a solver breakdown was detected.
                                       For CG: p^T*A*p = 0 or r^T*z = 0.
                                       For GMRES: lucky breakdown (Krylov subspace
                                       contains exact solution — converged=1 in this case).
                                       For MINRES: Lanczos breakdown (beta = 0).
                                       For BiCGSTAB: rho=0, omega=0, or r_hat^T*v=0. */
} sparse_iter_result_t;

/**
 * @brief Preconditioner callback type.
 *
 * A preconditioner approximates the solve M*z = r, where M approximates A.
 * Given an input vector r, the callback writes z = M^{-1}*r into the output
 * vector z. Both r and z have length n (the matrix dimension).
 *
 * @param ctx   User-supplied context (e.g., a factored preconditioner struct).
 * @param n     Vector length (matrix dimension).
 * @param r     Input vector (residual).
 * @param z     Output vector (preconditioned residual).
 * @return SPARSE_OK on success, or an error code on failure.
 */
typedef sparse_err_t (*sparse_precond_fn)(const void *ctx, idx_t n, const double *r, double *z);

/**
 * @brief Solve A*x = b using the Preconditioned Conjugate Gradient method.
 *
 * CG is applicable only to symmetric positive-definite (SPD) matrices.
 * The input x is used as the initial guess (pass a zero vector for no guess).
 *
 * Algorithm: standard preconditioned CG with relative residual convergence
 * test ||r_k|| / ||b|| < tol.
 *
 * @param A           The SPD coefficient matrix (not modified). Must be square.
 * @param b           Right-hand side vector of length n.
 * @param x           On entry, initial guess; on exit, approximate solution.
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for no preconditioning).
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result      Output: iteration count, residual, convergence flag (may be NULL).
 * @return SPARSE_OK if converged within tolerance.
 * @return SPARSE_ERR_NOT_CONVERGED if max_iter exceeded without convergence.
 * @return SPARSE_ERR_NULL if A, b, or x is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_BADARG if opts has negative max_iter or tol.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 *
 * @par Thread safety: Read-only on A. Safe to call concurrently on the same matrix
 *               with different b/x vectors.
 */
sparse_err_t sparse_solve_cg(const SparseMatrix *A, const double *b, double *x,
                             const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                             const void *precond_ctx, sparse_iter_result_t *result);

/**
 * @brief Solve A*x = b using the restarted GMRES(k) method.
 *
 * GMRES is applicable to general (possibly unsymmetric) square matrices.
 * Uses the Arnoldi process with Givens rotations for the Hessenberg
 * least-squares problem. Supports left and right preconditioning via
 * opts->precond_side (default: left). With right preconditioning, the
 * GMRES residual norm equals the true residual ||b - Ax|| (no gap).
 *
 * The input x is used as the initial guess (pass a zero vector for no guess).
 *
 * @param A           The coefficient matrix (not modified). Must be square.
 * @param b           Right-hand side vector of length n.
 * @param x           On entry, initial guess; on exit, approximate solution.
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for none). Used for both left
 *                    and right preconditioning, controlled by opts->precond_side.
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result      Output: iteration count, residual, convergence flag (may be NULL).
 * @return SPARSE_OK if converged within tolerance.
 * @return SPARSE_ERR_NOT_CONVERGED if max_iter exceeded without convergence.
 * @return SPARSE_ERR_NULL if A, b, or x is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_BADARG if opts has negative max_iter or tol, or restart <= 0.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails or overflows.
 *
 * @par Thread safety: Read-only on A. Safe to call concurrently on the same matrix
 *               with different b/x vectors.
 */
sparse_err_t sparse_solve_gmres(const SparseMatrix *A, const double *b, double *x,
                                const sparse_gmres_opts_t *opts, sparse_precond_fn precond,
                                const void *precond_ctx, sparse_iter_result_t *result);

/**
 * @brief Solve A*X = B for multiple RHS using block Conjugate Gradient.
 *
 * Runs CG simultaneously for all nrhs vectors with per-column convergence
 * tracking. Each column converges independently, and once a column's
 * residual drops below tolerance its per-column updates stop. The shared
 * SpMV amortizes matrix traversal.
 *
 * @param A           SPD coefficient matrix (not modified).
 * @param B           RHS matrix, n × nrhs column-major.
 * @param nrhs        Number of RHS vectors.
 * @param X           Solution matrix, n × nrhs column-major (initial guess on entry).
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for none). Applied per-column.
 * @param precond_ctx Context pointer passed to precond.
 * @param result      Output: iterations = max across columns, residual = max across columns.
 * @return SPARSE_OK if all columns converged.
 * @return SPARSE_ERR_NULL if A, B, or X is NULL.
 * @return SPARSE_ERR_BADARG if @p nrhs is negative or opts has invalid values.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails or n*nrhs overflows.
 * @return SPARSE_ERR_NOT_CONVERGED if any column did not converge.
 * @return Other error codes may be propagated from the preconditioner callback.
 */
sparse_err_t sparse_cg_solve_block(const SparseMatrix *A, const double *B, idx_t nrhs, double *X,
                                   const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                                   const void *precond_ctx, sparse_iter_result_t *result);

/**
 * @brief Solve A*X = B for multiple RHS using per-column GMRES.
 *
 * Runs restarted GMRES independently for each column and aggregates
 * convergence reporting across columns. This routine does not perform
 * a shared block-iteration scheme that skips converged columns during
 * later iterations; instead, it solves each RHS separately using the
 * existing single-RHS GMRES path. Supports preconditioning via callback.
 *
 * @param A           General (possibly unsymmetric) coefficient matrix.
 * @param B           RHS matrix, n × nrhs column-major.
 * @param nrhs        Number of RHS vectors.
 * @param X           Solution matrix, n × nrhs column-major (initial guess on entry).
 * @param opts        GMRES options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for none). Applied per-column.
 * @param precond_ctx Context pointer passed to precond.
 * @param result      Output: iterations = max across columns, residual = max across columns.
 * @return SPARSE_OK if all columns converged.
 * @return SPARSE_ERR_NULL if A, B, or X is NULL.
 * @return SPARSE_ERR_BADARG if @p nrhs is negative.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_ALLOC if n*nrhs overflows size_t.
 * @return SPARSE_ERR_NOT_CONVERGED if any column did not converge (but no
 *         hard error occurred). Hard errors from individual column solves
 *         (e.g., SPARSE_ERR_ALLOC) take priority over NOT_CONVERGED.
 */
sparse_err_t sparse_gmres_solve_block(const SparseMatrix *A, const double *B, idx_t nrhs, double *X,
                                      const sparse_gmres_opts_t *opts, sparse_precond_fn precond,
                                      const void *precond_ctx, sparse_iter_result_t *result);

/* ═══════════════════════════════════════════════════════════════════════
 * MINRES — Minimum Residual method for symmetric systems
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Solve A*x = b using the Preconditioned MINRES method.
 *
 * MINRES is applicable to symmetric (possibly indefinite) matrices.
 * It minimizes the 2-norm of the residual ||b - A*x|| over the Krylov
 * subspace using a Lanczos tridiagonalization with implicit QR via
 * Givens rotations. Unlike CG, MINRES does not require A to be
 * positive-definite; unlike GMRES, it exploits symmetry to use only
 * short recurrences (O(n) storage, no restart needed).
 *
 * The residual norm decreases monotonically at every iteration (a
 * guarantee that CG and GMRES(k) do not provide).
 *
 * The input x is used as the initial guess (pass a zero vector for no guess).
 *
 * @note **Symmetry requirement:** A must be symmetric. If A is not
 *       symmetric, the behavior is undefined (the Lanczos recurrence
 *       assumes symmetry). For non-symmetric systems, use GMRES instead.
 *
 * @note **Preconditioner requirement:** If a preconditioner is supplied,
 *       it must be symmetric positive-definite (SPD). The preconditioned
 *       MINRES algorithm uses the M-inner product, which requires M to
 *       define a valid inner product (i.e., M must be SPD).
 *
 * @param A           The symmetric coefficient matrix (not modified). Must be square.
 * @param b           Right-hand side vector of length n.
 * @param x           On entry, initial guess; on exit, approximate solution.
 * @param opts        Solver options (NULL for defaults: max_iter=1000, tol=1e-10).
 * @param precond     Preconditioner callback (NULL for no preconditioning).
 *                    Must be SPD if provided.
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result      Output: iteration count, residual, convergence flag (may be NULL).
 * @return SPARSE_OK if converged within tolerance.
 * @return SPARSE_ERR_NOT_CONVERGED if max_iter exceeded without convergence.
 * @return SPARSE_ERR_NULL if A, b, or x is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_BADARG if opts has negative max_iter or tol, or if a
 *         provided preconditioner is non-SPD (r^T M^{-1} r < 0) or degenerate.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 *
 * @par Thread safety: Read-only on A. Safe to call concurrently on the same matrix
 *               with different b/x vectors.
 */
sparse_err_t sparse_solve_minres(const SparseMatrix *A, const double *b, double *x,
                                 const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                                 const void *precond_ctx, sparse_iter_result_t *result);

/**
 * @brief Solve A*X = B for multiple RHS using per-column MINRES.
 *
 * Runs MINRES independently for each column and aggregates convergence
 * reporting across columns. Each column converges independently.
 *
 * @note **Symmetry requirement:** A must be symmetric. For non-symmetric
 *       systems, use sparse_gmres_solve_block() instead.
 *
 * @note **Preconditioner requirement:** If a preconditioner is supplied,
 *       it must be symmetric positive-definite.
 *
 * @param A           Symmetric coefficient matrix (not modified).
 * @param B           RHS matrix, n × nrhs column-major.
 * @param nrhs        Number of RHS vectors.
 * @param X           Solution matrix, n × nrhs column-major (initial guess on entry).
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for none). Must be SPD if provided.
 * @param precond_ctx Context pointer passed to precond.
 * @param result      Output: iterations = max across columns, residual = max across columns.
 * @return SPARSE_OK if all columns converged.
 * @return SPARSE_ERR_NULL if A, B, or X is NULL.
 * @return SPARSE_ERR_BADARG if @p nrhs is negative or opts has invalid values.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails or n*nrhs overflows.
 * @return SPARSE_ERR_NOT_CONVERGED if any column did not converge.
 */
sparse_err_t sparse_minres_solve_block(const SparseMatrix *A, const double *B, idx_t nrhs,
                                       double *X, const sparse_iter_opts_t *opts,
                                       sparse_precond_fn precond, const void *precond_ctx,
                                       sparse_iter_result_t *result);

/* ═══════════════════════════════════════════════════════════════════════
 * BiCGSTAB — Bi-Conjugate Gradient Stabilized for nonsymmetric systems
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Solve A*x = b using the BiCGSTAB method (Van der Vorst, 1992).
 *
 * BiCGSTAB is applicable to general nonsymmetric square matrices. It
 * combines the BiCG two-sided Lanczos approach with a polynomial
 * stabilization step, producing smoother convergence than CGS without
 * requiring A^T. Each iteration requires two matrix-vector products.
 *
 * BiCGSTAB is a good choice when:
 * - The matrix is nonsymmetric (CG and MINRES require symmetry).
 * - Restarted GMRES stalls due to information loss at restarts.
 * - Memory is limited (BiCGSTAB uses O(n) storage vs O(n*k) for GMRES(k)).
 *
 * For symmetric positive-definite systems, CG is preferred. For symmetric
 * indefinite systems, MINRES is preferred. For general nonsymmetric systems
 * where robustness matters more than storage, GMRES may be better.
 *
 * The input x is used as the initial guess (pass a zero vector for no guess).
 *
 * @param A           The coefficient matrix (not modified). Must be square.
 * @param b           Right-hand side vector of length n.
 * @param x           On entry, initial guess; on exit, approximate solution.
 * @param opts        Solver options (NULL for defaults: max_iter=1000, tol=1e-10).
 * @param precond     Preconditioner callback (NULL for no preconditioning).
 *                    Left preconditioning only: solves M*z = r.
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result      Output: iteration count, residual, convergence flag (may be NULL).
 * @return SPARSE_OK if converged within tolerance.
 * @return SPARSE_ERR_NOT_CONVERGED if max_iter exceeded without convergence.
 * @return SPARSE_ERR_NULL if A, b, or x is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_BADARG if opts has negative max_iter or tol.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 * @return SPARSE_ERR_NUMERIC if NaN or Inf is produced during iteration.
 *
 * @par Thread safety: Read-only on A. Safe to call concurrently on the same matrix
 *               with different b/x vectors.
 */
sparse_err_t sparse_solve_bicgstab(const SparseMatrix *A, const double *b, double *x,
                                   const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                                   const void *precond_ctx, sparse_iter_result_t *result);

/**
 * @brief Solve A*X = B for multiple RHS using per-column BiCGSTAB.
 *
 * Runs BiCGSTAB independently for each column and aggregates convergence
 * reporting across columns. Each column converges independently.
 *
 * @param A           General (possibly unsymmetric) coefficient matrix.
 * @param B           RHS matrix, n × nrhs column-major.
 * @param nrhs        Number of RHS vectors.
 * @param X           Solution matrix, n × nrhs column-major (initial guess on entry).
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for none). Applied per-column.
 * @param precond_ctx Context pointer passed to precond.
 * @param result      Output: iterations = max across columns, residual = max across columns.
 * @return SPARSE_OK if all columns converged.
 * @return SPARSE_ERR_NULL if A, B, or X is NULL.
 * @return SPARSE_ERR_BADARG if @p nrhs is negative or opts has invalid values.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails or n*nrhs overflows.
 * @return SPARSE_ERR_NOT_CONVERGED if any column did not converge.
 * @return SPARSE_ERR_NUMERIC if NaN or Inf is produced during iteration.
 * @return Other error codes may be propagated from the preconditioner callback.
 */
sparse_err_t sparse_bicgstab_solve_block(const SparseMatrix *A, const double *B, idx_t nrhs,
                                         double *X, const sparse_iter_opts_t *opts,
                                         sparse_precond_fn precond, const void *precond_ctx,
                                         sparse_iter_result_t *result);

/* ═══════════════════════════════════════════════════════════════════════
 * Matrix-free iterative solvers
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Matrix-free matrix-vector product callback.
 *
 * Computes y = A*x for an implicit linear operator. The operator is
 * defined by the context pointer (e.g., a struct containing the operator
 * parameters).
 *
 * @param ctx  User-supplied context (e.g., operator parameters).
 * @param n    Vector length (operator dimension — square operator assumed).
 * @param x    Input vector of length n.
 * @param y    Output vector of length n (overwritten with A*x).
 * @return SPARSE_OK on success, or an error code on failure.
 */
typedef sparse_err_t (*sparse_matvec_fn)(const void *ctx, idx_t n, const double *x, double *y);

/**
 * @brief Solve A*x = b using matrix-free Conjugate Gradient.
 *
 * Same algorithm as sparse_solve_cg() but the matrix-vector product A*x
 * is provided via a callback instead of an explicit SparseMatrix.
 *
 * @param matvec     Callback computing y = A*x. Must not be NULL.
 * @param matvec_ctx Context pointer passed to matvec callback.
 * @param n          System dimension (A is n×n).
 * @param b          Right-hand side vector of length n.
 * @param x          On entry, initial guess; on exit, approximate solution.
 * @param opts       Solver options (NULL for defaults).
 * @param precond    Preconditioner callback (NULL for none).
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result     Output: iteration count, residual, convergence flag (may be NULL).
 * @return SPARSE_OK on convergence, SPARSE_ERR_NOT_CONVERGED otherwise.
 * @return SPARSE_ERR_NULL if matvec, b, or x is NULL.
 * @return SPARSE_ERR_BADARG if n < 0 or opts has invalid fields.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 * @return Any error returned by the matvec or precond callbacks.
 */
sparse_err_t sparse_solve_cg_mf(sparse_matvec_fn matvec, const void *matvec_ctx, idx_t n,
                                const double *b, double *x, const sparse_iter_opts_t *opts,
                                sparse_precond_fn precond, const void *precond_ctx,
                                sparse_iter_result_t *result);

/**
 * @brief Solve A*x = b using matrix-free restarted GMRES(k).
 *
 * Same algorithm as sparse_solve_gmres() but the matrix-vector product A*x
 * is provided via a callback instead of an explicit SparseMatrix.
 *
 * @param matvec     Callback computing y = A*x. Must not be NULL.
 * @param matvec_ctx Context pointer passed to matvec callback.
 * @param n          System dimension (A is n×n).
 * @param b          Right-hand side vector of length n.
 * @param x          On entry, initial guess; on exit, approximate solution.
 * @param opts       Solver options (NULL for defaults).
 * @param precond    Preconditioner callback (NULL for none).
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result     Output: iteration count, residual, convergence flag (may be NULL).
 * @return SPARSE_OK on convergence, SPARSE_ERR_NOT_CONVERGED otherwise.
 * @return SPARSE_ERR_NULL if matvec, b, or x is NULL.
 * @return SPARSE_ERR_BADARG if n < 0, restart <= 0, or opts has invalid fields.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 * @return Any error returned by the matvec or precond callbacks.
 */
sparse_err_t sparse_solve_gmres_mf(sparse_matvec_fn matvec, const void *matvec_ctx, idx_t n,
                                   const double *b, double *x, const sparse_gmres_opts_t *opts,
                                   sparse_precond_fn precond, const void *precond_ctx,
                                   sparse_iter_result_t *result);

/**
 * @brief Solve A*x = b using matrix-free BiCGSTAB.
 *
 * Same algorithm as sparse_solve_bicgstab() but the matrix-vector product A*x
 * is provided via a callback instead of an explicit SparseMatrix.
 *
 * @param matvec     Callback computing y = A*x. Must not be NULL.
 * @param matvec_ctx Context pointer passed to matvec callback.
 * @param n          System dimension (A is n×n).
 * @param b          Right-hand side vector of length n.
 * @param x          On entry, initial guess; on exit, approximate solution.
 * @param opts       Solver options (NULL for defaults).
 * @param precond    Preconditioner callback (NULL for none).
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result     Output: iteration count, residual, convergence flag (may be NULL).
 * @return SPARSE_OK on convergence, SPARSE_ERR_NOT_CONVERGED otherwise.
 * @return SPARSE_ERR_NULL if matvec, b, or x is NULL.
 * @return SPARSE_ERR_BADARG if n < 0 or opts has invalid fields.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 * @return SPARSE_ERR_NUMERIC if NaN or Inf is produced during iteration.
 * @return Any error returned by the matvec or precond callbacks.
 */
sparse_err_t sparse_solve_bicgstab_mf(sparse_matvec_fn matvec, const void *matvec_ctx, idx_t n,
                                      const double *b, double *x, const sparse_iter_opts_t *opts,
                                      sparse_precond_fn precond, const void *precond_ctx,
                                      sparse_iter_result_t *result);

#endif /* SPARSE_ITERATIVE_H */
