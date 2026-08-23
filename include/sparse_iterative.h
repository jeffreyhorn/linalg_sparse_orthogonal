#ifndef SPARSE_ITERATIVE_H
#define SPARSE_ITERATIVE_H

/**
 * @file sparse_iterative.h
 * @brief Krylov subspace iterative solvers for sparse linear systems.
 *
 * Provides CG for symmetric positive-definite systems, GMRES and BiCGSTAB for
 * general nonsymmetric systems, and MINRES for symmetric indefinite systems.
 * One-shot solve functions are the normal starting point. Explicit repeated-run
 * handles are available for CG, GMRES, and MINRES when the problem dimension is
 * stable and workspace reuse matters.
 *
 * Preconditioners are caller-supplied callbacks and contexts. Solver calls
 * borrow those pointers only for the duration of a solve; callers own the
 * callback context and any factor/preconditioner object it references. Match
 * each preconditioner to the solver assumptions, inspect
 * `sparse_iter_result_t` for convergence and residual diagnostics, and use
 * `docs/solver_selection.md`, `docs/tutorial.md`, `docs/cookbook.md`, and
 * `examples/README.md` for the public workflow path before using this header
 * as the exact declaration and option/result reference.
 */

#include "sparse_matrix.h"

/* ═══════════════════════════════════════════════════════════════════════
 * Shared callbacks, options, and result types
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @par Breakdown behavior summary
 *
 * `result.breakdown` records Krylov breakdown conditions such as zero CG
 * denominators, GMRES lucky breakdown, MINRES Lanczos/QR breakdown, and
 * BiCGSTAB rho/omega/t-vector breakdown. GMRES lucky breakdown is reported as
 * both `breakdown=1` and `converged=1`; other breakdown paths report
 * `breakdown=1` with `converged=0`. Threshold checks use
 * `sparse_rel_tol(0, DROP_TOL)`.
 */

/**
 * @brief Progress information passed to the verbose callback.
 *
 * Populated by the solver at each iteration (or restart boundary for GMRES)
 * and passed to the user callback if one is provided. The pointer is borrowed
 * for the callback invocation only; do not store it after the callback
 * returns.
 */
typedef struct {
    idx_t iteration;               /**< Current iteration number (0-based) */
    sparse_scalar_t residual_norm; /**< Current relative residual norm ||r||/||b|| */
    const char *solver;            /**< Solver name ("CG", "GMRES", "MINRES", "BiCGSTAB") */
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
 * A zero-initialized explicit struct is not identical to NULL defaults:
 * `max_iter = 0` requests a zero-iteration budget, `tol = 0` requires a
 * zero relative residual, and optional callback/history fields remain off
 * unless the caller supplies them.
 */
typedef struct {
    idx_t max_iter; /**< Maximum number of CG iterations (default: 1000) */
    sparse_scalar_t
        tol;     /**< Convergence tolerance on relative residual ||r||/||b|| (default: 1e-10) */
    int verbose; /**< If nonzero, print iteration log to stderr (default: 0) */
    idx_t stagnation_window;           /**< Stagnation detection window size. If > 0, the solver
                                            tracks the last N residual norms and declares stagnation
                                            if max/min in the window differ by less than 1%.
                                            0 = disabled (default). Typical value: 10-20. */
    sparse_scalar_t *residual_history; /**< Caller-owned array for per-iteration residual norms.
                                            If non-NULL, the solver writes ||r_k||/||b|| at index k
                                            but does not allocate, retain, or free the buffer.
                                            On non-convergence or cancellation, only the first
                                            result.residual_history_count entries are meaningful.
                                            NULL = no recording (default). */
    idx_t residual_history_len;        /**< Capacity of the residual_history array. The solver
                                            writes at most this many entries. */
    sparse_iter_callback_fn callback;  /**< Verbose callback. If non-NULL, called each iteration
                                            instead of fprintf(stderr). NULL = use default verbose
                                            behavior (default). */
    void *callback_ctx;                /**< Context pointer passed to callback. */
    /** Optional progress / cancellation callback. Invoked at the top of
     *  each solver iteration with
     *  `phase = "cg" / "minres" / "bicgstab"` (matching the solver
     *  selected by the opts struct), `step = iter`, `total = max_iter`.
     *  Return 0 to continue; non-zero cancels — `SPARSE_ERR_CANCELLED`
     *  propagates.  NULL (default) disables.  Distinct from
     *  `callback` above: `callback` is verbose logging only (void
     *  return), `progress_cb` adds cancellation. Trailing field for
     *  designated-init back-compat. */
    sparse_progress_cb_t progress_cb;
    /** Opaque context pointer passed through unchanged to
     *  `progress_cb`.  Ignored when `progress_cb == NULL`. */
    void *progress_user;
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
 * A zero-initialized explicit struct is not identical to NULL defaults:
 * `restart = 0` is invalid, `max_iter = 0` requests only the initial-residual
 * check, and optional callback/history fields remain off unless supplied.
 */
typedef struct {
    idx_t max_iter;      /**< Maximum total number of GMRES iterations (default: 1000) */
    idx_t restart;       /**< Restart parameter k for GMRES(k) (default: 30) */
    sparse_scalar_t tol; /**< Convergence tolerance on relative residual (default: 1e-10) */
    int verbose;         /**< If nonzero, print iteration log to stderr (default: 0) */
    sparse_precond_side_t precond_side; /**< Left or right preconditioning (default: LEFT) */
    idx_t stagnation_window;            /**< Stagnation detection window size (across restarts).
                                             0 = disabled (default). See sparse_iter_opts_t. */
    sparse_scalar_t *residual_history;  /**< See sparse_iter_opts_t::residual_history. */
    idx_t residual_history_len;         /**< See sparse_iter_opts_t::residual_history_len. */
    sparse_iter_callback_fn callback;   /**< See sparse_iter_opts_t::callback. */
    void *callback_ctx;                 /**< See sparse_iter_opts_t::callback_ctx. */
    /** Progress / cancel callback —
     *  semantics identical to `sparse_iter_opts_t::progress_cb`.
     *  Emitted per GMRES inner Arnoldi iteration with `phase =
     *  "gmres"`, `step = total_iter`, `total = max_iter`.  Trailing
     *  field for designated-init back-compat. */
    sparse_progress_cb_t progress_cb;
    void *progress_user; /**< User context passed to progress_cb. */
} sparse_gmres_opts_t;

/**
 * @brief Result information from an iterative solve.
 *
 * Populated by solve functions on return. The struct storage is caller-owned;
 * the library writes scalar fields only and does not allocate nested result
 * buffers. Pass NULL if result information is not needed.
 * Interpret fields only together with the function return code: successful
 * convergence, non-convergence, stagnation, and breakdown use the same struct
 * but have different meanings for the approximation in `x`/`X`.
 * `residual_history_count` is 0 when no residual history buffer was supplied
 * or no entries were recorded.
 */
typedef struct {
    idx_t iterations;              /**< Number of iterations performed */
    sparse_scalar_t residual_norm; /**< Final true relative residual norm ||b - A*x|| / ||b|| */
    int converged;                 /**< Nonzero if solver converged within tolerance */
    int stagnated;                 /**< Nonzero if stagnation was detected (residual stopped
                                        decreasing over the stagnation window). Only set when
                                        stagnation_window > 0 in opts. */
    idx_t residual_history_count;  /**< Number of entries written to residual_history. */
    int breakdown;                 /**< Nonzero if a solver breakdown was detected.
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
 * vector z. Both r and z have length n (the matrix dimension). The solver
 * owns neither buffer; it supplies temporary borrowed storage for the
 * invocation. The callback must not retain r or z after returning.
 *
 * @param ctx   User-supplied context (e.g., a factored preconditioner struct).
 * @param n     Vector length (matrix dimension).
 * @param r     Input vector (residual).
 * @param z     Output vector (preconditioned residual). The callback should
 *              fully write z on SPARSE_OK; on callback error, solver behavior
 *              is limited to propagating that error.
 * @return SPARSE_OK on success, or an error code on failure.
 */
typedef sparse_err_t (*sparse_precond_fn)(const void *ctx, idx_t n, const sparse_scalar_t *r,
                                          sparse_scalar_t *z);

/* ═══════════════════════════════════════════════════════════════════════
 * Explicit repeated-run lifecycle handles
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Reusable handle for repeated iterative solves on stable-dimension
 *        problems.
 *
 * The one-shot public entries (`sparse_solve_cg()`,
 * `sparse_solve_gmres()`, `sparse_solve_minres()`, and related wrappers)
 * remain first-class and fully supported. This handle exposes the explicit
 * repeated-run lifecycle for callers that want to preserve workspace
 * capacity across solves while keeping the existing option/result contracts.
 *
 * The layout is intentionally opaque at the public level: zero-initialize
 * the struct (`{0}`) or call sparse_iter_handle_init() before first use,
 * then use the prepare / solve / free helpers below. The caller owns the
 * handle object, while the library owns any internal workspace reachable
 * through it. Reuse may preserve allocation capacity, but it does not
 * preserve prior Krylov state, residual history contents, or convergence
 * status as a numerical feature.
 *
 * sparse_iter_handle_free() is safe on NULL, on a zeroed struct, and after
 * a prior free. Invalid prepare arguments return SPARSE_ERR_BADARG before
 * publishing internal handle state. Allocation failures during the maintained
 * repeated-run iterative prepare/growth paths leave either an empty handle or
 * the previously usable handle capacity intact; callers should still treat
 * SPARSE_ERR_ALLOC as a failed prepare/solve and retry only after handling the
 * error.
 */
typedef struct {
    void *internal_state; /**< Opaque internal repeated-run workspace owner.
                               Treat as private implementation state. */
} sparse_iter_handle_t;

/**
 * @brief Initialize an iterative repeated-run handle.
 *
 * Equivalent to assigning `{0}`. Safe to call on an already-zeroed handle.
 *
 * @param handle  Handle to initialize. NULL is ignored.
 */
void sparse_iter_handle_init(sparse_iter_handle_t *handle);

/**
 * @brief Release all memory owned by an iterative repeated-run handle.
 *
 * Safe to call on a zeroed struct; after freeing, the handle is reset to
 * the zero state.
 *
 * @param handle  Handle to free. NULL is ignored.
 */
void sparse_iter_handle_free(sparse_iter_handle_t *handle);

/**
 * @brief Prepare an iterative handle for repeated CG solves of dimension n.
 *
 * This is the explicit repeated-run setup path for CG callers. Successful
 * prepare preserves reusable capacity for later `sparse_solve_cg_with_handle()`
 * calls on the same or smaller dimension. Re-preparing may grow capacity and
 * discards any prior numerical iteration state.
 *
 * @param handle  Reusable handle to prepare.
 * @param n       Problem dimension.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if handle is NULL.
 * @return SPARSE_ERR_BADARG if n is less than 1.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 */
sparse_err_t sparse_iter_handle_prepare_cg(sparse_iter_handle_t *handle, idx_t n);

/**
 * @brief Prepare an iterative handle for repeated GMRES(k) solves.
 *
 * Successful prepare preserves reusable capacity for later
 * `sparse_solve_gmres_with_handle()` calls with the same or smaller
 * dimension and restart parameter. Re-preparing may grow capacity and
 * discards any prior numerical iteration state.
 *
 * @param handle   Reusable handle to prepare.
 * @param n        Problem dimension.
 * @param restart  GMRES restart parameter.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if handle is NULL.
 * @return SPARSE_ERR_BADARG if n is less than 1 or restart is invalid.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails or overflows.
 */
sparse_err_t sparse_iter_handle_prepare_gmres(sparse_iter_handle_t *handle, idx_t n, idx_t restart);

/**
 * @brief Prepare an iterative handle for repeated MINRES solves.
 *
 * Successful prepare preserves reusable capacity for later
 * `sparse_solve_minres_with_handle()` calls with the same or smaller
 * dimension. Re-preparing may grow capacity and discards any prior
 * numerical iteration state.
 *
 * MINRES may be run with or without a preconditioner on the same prepared
 * handle. Reuse preserves allocation capacity only; it does not preserve
 * prior Lanczos state, recurrence state, or convergence history.
 *
 * @param handle  Reusable handle to prepare.
 * @param n       Problem dimension.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if handle is NULL.
 * @return SPARSE_ERR_BADARG if n is less than 1.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails or overflows.
 */
sparse_err_t sparse_iter_handle_prepare_minres(sparse_iter_handle_t *handle, idx_t n);

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
 * @param x           On entry, initial guess; on SPARSE_OK or
 *                    SPARSE_ERR_NOT_CONVERGED, approximate solution.
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for no preconditioning).
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result      Output: iteration count, residual, convergence flag
 *                    (may be NULL). Populated on SPARSE_OK and
 *                    SPARSE_ERR_NOT_CONVERGED; on validation, allocation,
 *                    cancellation, or callback errors, fields are
 *                    best-effort/unspecified unless documented otherwise.
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
sparse_err_t sparse_solve_cg(const SparseMatrix *A, const sparse_scalar_t *b, sparse_scalar_t *x,
                             const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                             const void *precond_ctx, sparse_iter_result_t *result);

/**
 * @brief Solve A*x = b using CG with an explicit reusable handle.
 *
 * This has the same numerical contract as sparse_solve_cg(), but reuses a
 * caller-owned handle across repeated solves. Callers may explicitly prepare
 * the handle via sparse_iter_handle_prepare_cg() first; if the handle is
 * zeroed or underprepared, the implementation may grow its internal workspace
 * on demand.
 *
 * @param A           The SPD coefficient matrix (not modified). Must be square.
 * @param b           Right-hand side vector of length n.
 * @param x           On entry, initial guess; on SPARSE_OK or
 *                    SPARSE_ERR_NOT_CONVERGED, approximate solution.
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for no preconditioning).
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result      Output: iteration count, residual, convergence flag
 *                    (may be NULL). Same population rules as
 *                    sparse_solve_cg().
 * @param handle      Reusable handle. Must be non-NULL.
 * @return Same error contract as sparse_solve_cg(), plus SPARSE_ERR_NULL when
 *         handle is NULL.
 */
sparse_err_t sparse_solve_cg_with_handle(const SparseMatrix *A, const sparse_scalar_t *b,
                                         sparse_scalar_t *x, const sparse_iter_opts_t *opts,
                                         sparse_precond_fn precond, const void *precond_ctx,
                                         sparse_iter_result_t *result,
                                         sparse_iter_handle_t *handle);

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
 * @param x           On entry, initial guess; on SPARSE_OK or
 *                    SPARSE_ERR_NOT_CONVERGED, approximate solution.
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for none). Used for both left
 *                    and right preconditioning, controlled by opts->precond_side.
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result      Output: iteration count, residual, convergence flag
 *                    (may be NULL). Populated on SPARSE_OK and
 *                    SPARSE_ERR_NOT_CONVERGED; on validation, allocation,
 *                    cancellation, or callback errors, fields are
 *                    best-effort/unspecified unless documented otherwise.
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
sparse_err_t sparse_solve_gmres(const SparseMatrix *A, const sparse_scalar_t *b, sparse_scalar_t *x,
                                const sparse_gmres_opts_t *opts, sparse_precond_fn precond,
                                const void *precond_ctx, sparse_iter_result_t *result);

/**
 * @brief Solve A*x = b using restarted GMRES(k) with an explicit reusable
 *        handle.
 *
 * This has the same numerical contract as sparse_solve_gmres(), but reuses a
 * caller-owned handle across repeated solves. Callers may explicitly prepare
 * the handle via sparse_iter_handle_prepare_gmres() first; if the handle is
 * zeroed or underprepared, the implementation may grow its internal workspace
 * on demand.
 *
 * @param A           The coefficient matrix (not modified). Must be square.
 * @param b           Right-hand side vector of length n.
 * @param x           On entry, initial guess; on SPARSE_OK or
 *                    SPARSE_ERR_NOT_CONVERGED, approximate solution.
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for none). Used for both left
 *                    and right preconditioning, controlled by opts->precond_side.
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result      Output: iteration count, residual, convergence flag
 *                    (may be NULL). Same population rules as
 *                    sparse_solve_gmres().
 * @param handle      Reusable handle. Must be non-NULL.
 * @return Same error contract as sparse_solve_gmres(), plus SPARSE_ERR_NULL
 *         when handle is NULL.
 */
sparse_err_t sparse_solve_gmres_with_handle(const SparseMatrix *A, const sparse_scalar_t *b,
                                            sparse_scalar_t *x, const sparse_gmres_opts_t *opts,
                                            sparse_precond_fn precond, const void *precond_ctx,
                                            sparse_iter_result_t *result,
                                            sparse_iter_handle_t *handle);

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
 * @param X           Solution matrix, n × nrhs column-major. On entry, initial
 *                    guesses; on SPARSE_OK or SPARSE_ERR_NOT_CONVERGED,
 *                    per-column approximate solutions.
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for none). Applied per-column.
 * @param precond_ctx Context pointer passed to precond.
 * @param result      Output: iterations = max across columns, residual = max
 *                    across columns (may be NULL). Populated after per-column
 *                    solves complete; on hard errors, fields are
 *                    best-effort/unspecified.
 * @return SPARSE_OK if all columns converged.
 * @return SPARSE_ERR_NULL if A, B, or X is NULL.
 * @return SPARSE_ERR_BADARG if @p nrhs is negative or opts has invalid values.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails or n*nrhs overflows.
 * @return SPARSE_ERR_NOT_CONVERGED if any column did not converge.
 * @return Other error codes may be propagated from the preconditioner callback.
 */
sparse_err_t sparse_cg_solve_block(const SparseMatrix *A, const sparse_scalar_t *B, idx_t nrhs,
                                   sparse_scalar_t *X, const sparse_iter_opts_t *opts,
                                   sparse_precond_fn precond, const void *precond_ctx,
                                   sparse_iter_result_t *result);

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
 * @param X           Solution matrix, n × nrhs column-major. On entry, initial
 *                    guesses; on SPARSE_OK or SPARSE_ERR_NOT_CONVERGED,
 *                    per-column approximate solutions.
 * @param opts        GMRES options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for none). Applied per-column.
 * @param precond_ctx Context pointer passed to precond.
 * @param result      Output: iterations = max across columns, residual = max
 *                    across columns (may be NULL). Populated after per-column
 *                    solves complete; on hard errors, fields are
 *                    best-effort/unspecified.
 * @return SPARSE_OK if all columns converged.
 * @return SPARSE_ERR_NULL if A, B, or X is NULL.
 * @return SPARSE_ERR_BADARG if @p nrhs is negative.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_ALLOC if n*nrhs overflows size_t.
 * @return SPARSE_ERR_NOT_CONVERGED if any column did not converge (but no
 *         hard error occurred). Hard errors from individual column solves
 *         (e.g., SPARSE_ERR_ALLOC) take priority over NOT_CONVERGED.
 */
sparse_err_t sparse_gmres_solve_block(const SparseMatrix *A, const sparse_scalar_t *B, idx_t nrhs,
                                      sparse_scalar_t *X, const sparse_gmres_opts_t *opts,
                                      sparse_precond_fn precond, const void *precond_ctx,
                                      sparse_iter_result_t *result);

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
 * @param x           On entry, initial guess; on SPARSE_OK or
 *                    SPARSE_ERR_NOT_CONVERGED, approximate solution.
 * @param opts        Solver options (NULL for defaults: max_iter=1000, tol=1e-10).
 * @param precond     Preconditioner callback (NULL for no preconditioning).
 *                    Must be SPD if provided.
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result      Output: iteration count, residual, convergence flag
 *                    (may be NULL). Populated on SPARSE_OK and
 *                    SPARSE_ERR_NOT_CONVERGED; on validation, allocation,
 *                    cancellation, or callback errors, fields are
 *                    best-effort/unspecified unless documented otherwise.
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
sparse_err_t sparse_solve_minres(const SparseMatrix *A, const sparse_scalar_t *b,
                                 sparse_scalar_t *x, const sparse_iter_opts_t *opts,
                                 sparse_precond_fn precond, const void *precond_ctx,
                                 sparse_iter_result_t *result);

/**
 * @brief Solve A*x = b using MINRES with an explicit reusable handle.
 *
 * This has the same numerical contract as sparse_solve_minres(), but reuses a
 * caller-owned handle across repeated solves. Callers may explicitly prepare
 * the handle via sparse_iter_handle_prepare_minres() first; if the handle is
 * zeroed or underprepared, the implementation may grow its internal workspace
 * on demand.
 *
 * @param A           The symmetric coefficient matrix (not modified). Must be square.
 * @param b           Right-hand side vector of length n.
 * @param x           On entry, initial guess; on SPARSE_OK or
 *                    SPARSE_ERR_NOT_CONVERGED, approximate solution.
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for no preconditioning).
 *                    Must be SPD if provided.
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result      Output: iteration count, residual, convergence flag
 *                    (may be NULL). Same population rules as
 *                    sparse_solve_minres().
 * @param handle      Reusable handle. Must be non-NULL.
 * @return Same error contract as sparse_solve_minres(), plus SPARSE_ERR_NULL
 *         when handle is NULL.
 */
sparse_err_t sparse_solve_minres_with_handle(const SparseMatrix *A, const sparse_scalar_t *b,
                                             sparse_scalar_t *x, const sparse_iter_opts_t *opts,
                                             sparse_precond_fn precond, const void *precond_ctx,
                                             sparse_iter_result_t *result,
                                             sparse_iter_handle_t *handle);

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
 * @param X           Solution matrix, n × nrhs column-major. On entry, initial
 *                    guesses; on SPARSE_OK or SPARSE_ERR_NOT_CONVERGED,
 *                    per-column approximate solutions.
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for none). Must be SPD if provided.
 * @param precond_ctx Context pointer passed to precond.
 * @param result      Output: iterations = max across columns, residual = max
 *                    across columns (may be NULL). Populated after per-column
 *                    solves complete; on hard errors, fields are
 *                    best-effort/unspecified.
 * @return SPARSE_OK if all columns converged.
 * @return SPARSE_ERR_NULL if A, B, or X is NULL.
 * @return SPARSE_ERR_BADARG if @p nrhs is negative or opts has invalid values.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails or n*nrhs overflows.
 * @return SPARSE_ERR_NOT_CONVERGED if any column did not converge.
 */
sparse_err_t sparse_minres_solve_block(const SparseMatrix *A, const sparse_scalar_t *B, idx_t nrhs,
                                       sparse_scalar_t *X, const sparse_iter_opts_t *opts,
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
 * @param x           On entry, initial guess; on SPARSE_OK or
 *                    SPARSE_ERR_NOT_CONVERGED, approximate solution.
 * @param opts        Solver options (NULL for defaults: max_iter=1000, tol=1e-10).
 * @param precond     Preconditioner callback (NULL for no preconditioning).
 *                    Left preconditioning only: solves M*z = r.
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result      Output: iteration count, residual, convergence flag
 *                    (may be NULL). Populated on SPARSE_OK and
 *                    SPARSE_ERR_NOT_CONVERGED; on validation, allocation,
 *                    cancellation, numeric, or callback errors, fields are
 *                    best-effort/unspecified unless documented otherwise.
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
sparse_err_t sparse_solve_bicgstab(const SparseMatrix *A, const sparse_scalar_t *b,
                                   sparse_scalar_t *x, const sparse_iter_opts_t *opts,
                                   sparse_precond_fn precond, const void *precond_ctx,
                                   sparse_iter_result_t *result);

/**
 * @brief Solve A*X = B for multiple RHS using per-column BiCGSTAB.
 *
 * Runs BiCGSTAB independently for each column and aggregates convergence
 * reporting across columns. Each column converges independently.
 *
 * @param A           General (possibly unsymmetric) coefficient matrix.
 * @param B           RHS matrix, n × nrhs column-major.
 * @param nrhs        Number of RHS vectors.
 * @param X           Solution matrix, n × nrhs column-major. On entry, initial
 *                    guesses; on SPARSE_OK or SPARSE_ERR_NOT_CONVERGED,
 *                    per-column approximate solutions.
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback (NULL for none). Applied per-column.
 * @param precond_ctx Context pointer passed to precond.
 * @param result      Output: iterations = max across columns, residual = max
 *                    across columns (may be NULL). Populated after block
 *                    iteration completes; on hard errors, fields are
 *                    best-effort/unspecified.
 * @return SPARSE_OK if all columns converged.
 * @return SPARSE_ERR_NULL if A, B, or X is NULL.
 * @return SPARSE_ERR_BADARG if @p nrhs is negative or opts has invalid values.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails or n*nrhs overflows.
 * @return SPARSE_ERR_NOT_CONVERGED if any column did not converge.
 * @return SPARSE_ERR_NUMERIC if NaN or Inf is produced during iteration.
 * @return Other error codes may be propagated from the preconditioner callback.
 */
sparse_err_t sparse_bicgstab_solve_block(const SparseMatrix *A, const sparse_scalar_t *B,
                                         idx_t nrhs, sparse_scalar_t *X,
                                         const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                                         const void *precond_ctx, sparse_iter_result_t *result);

/* ═══════════════════════════════════════════════════════════════════════
 * Matrix-free iterative solvers
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Matrix-free matrix-vector product callback.
 *
 * Computes y = A*x for an implicit linear operator. The operator is
 * defined by the caller-owned context pointer (e.g., a struct containing the
 * operator parameters). The solver borrows the context for the duration of
 * each callback invocation and does not retain or free it.
 *
 * @param ctx  User-supplied context (e.g., operator parameters).
 * @param n    Vector length (operator dimension — square operator assumed).
 * @param x    Input vector of length n.
 * @param y    Output vector of length n (overwritten with A*x on SPARSE_OK).
 * @return SPARSE_OK on success, or an error code on failure.
 */
typedef sparse_err_t (*sparse_matvec_fn)(const void *ctx, idx_t n, const sparse_scalar_t *x,
                                         sparse_scalar_t *y);

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
 * @param x          On entry, initial guess; on SPARSE_OK or
 *                   SPARSE_ERR_NOT_CONVERGED, approximate solution.
 * @param opts       Solver options (NULL for defaults).
 * @param precond    Preconditioner callback (NULL for none).
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result     Output: iteration count, residual, convergence flag
 *                   (may be NULL). Populated on SPARSE_OK and
 *                   SPARSE_ERR_NOT_CONVERGED; on validation, allocation, or
 *                   callback errors, fields are best-effort/unspecified.
 * @return SPARSE_OK on convergence, SPARSE_ERR_NOT_CONVERGED otherwise.
 * @return SPARSE_ERR_NULL if matvec, b, or x is NULL.
 * @return SPARSE_ERR_BADARG if n < 0 or opts has invalid fields.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 * @return Any error returned by the matvec or precond callbacks.
 */
sparse_err_t sparse_solve_cg_mf(sparse_matvec_fn matvec, const void *matvec_ctx, idx_t n,
                                const sparse_scalar_t *b, sparse_scalar_t *x,
                                const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                                const void *precond_ctx, sparse_iter_result_t *result);

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
 * @param x          On entry, initial guess; on SPARSE_OK or
 *                   SPARSE_ERR_NOT_CONVERGED, approximate solution.
 * @param opts       Solver options (NULL for defaults).
 * @param precond    Preconditioner callback (NULL for none).
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result     Output: iteration count, residual, convergence flag
 *                   (may be NULL). Populated on SPARSE_OK and
 *                   SPARSE_ERR_NOT_CONVERGED; on validation, allocation, or
 *                   callback errors, fields are best-effort/unspecified.
 * @return SPARSE_OK on convergence, SPARSE_ERR_NOT_CONVERGED otherwise.
 * @return SPARSE_ERR_NULL if matvec, b, or x is NULL.
 * @return SPARSE_ERR_BADARG if n < 0, restart <= 0, or opts has invalid fields.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 * @return Any error returned by the matvec or precond callbacks.
 */
sparse_err_t sparse_solve_gmres_mf(sparse_matvec_fn matvec, const void *matvec_ctx, idx_t n,
                                   const sparse_scalar_t *b, sparse_scalar_t *x,
                                   const sparse_gmres_opts_t *opts, sparse_precond_fn precond,
                                   const void *precond_ctx, sparse_iter_result_t *result);

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
 * @param x          On entry, initial guess; on SPARSE_OK or
 *                   SPARSE_ERR_NOT_CONVERGED, approximate solution.
 * @param opts       Solver options (NULL for defaults).
 * @param precond    Preconditioner callback (NULL for none).
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result     Output: iteration count, residual, convergence flag
 *                   (may be NULL). Populated on SPARSE_OK and
 *                   SPARSE_ERR_NOT_CONVERGED; on validation, allocation,
 *                   numeric, or callback errors, fields are
 *                   best-effort/unspecified.
 * @return SPARSE_OK on convergence, SPARSE_ERR_NOT_CONVERGED otherwise.
 * @return SPARSE_ERR_NULL if matvec, b, or x is NULL.
 * @return SPARSE_ERR_BADARG if n < 0 or opts has invalid fields.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 * @return SPARSE_ERR_NUMERIC if NaN or Inf is produced during iteration.
 * @return Any error returned by the matvec or precond callbacks.
 */
sparse_err_t sparse_solve_bicgstab_mf(sparse_matvec_fn matvec, const void *matvec_ctx, idx_t n,
                                      const sparse_scalar_t *b, sparse_scalar_t *x,
                                      const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                                      const void *precond_ctx, sparse_iter_result_t *result);

#endif /* SPARSE_ITERATIVE_H */
