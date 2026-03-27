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
 * @brief Options for the Conjugate Gradient solver.
 *
 * Pass NULL to sparse_solve_cg() to use defaults:
 * max_iter = 1000, tol = 1e-10, verbose = 0.
 */
typedef struct {
    idx_t  max_iter;   /**< Maximum number of CG iterations (default: 1000) */
    double tol;        /**< Convergence tolerance on relative residual ||r||/||b|| (default: 1e-10) */
    int    verbose;    /**< If nonzero, print iteration log to stderr (default: 0) */
} sparse_iter_opts_t;

/**
 * @brief Options for the GMRES solver.
 *
 * Pass NULL to sparse_solve_gmres() to use defaults:
 * max_iter = 1000, restart = 30, tol = 1e-10, verbose = 0.
 */
typedef struct {
    idx_t  max_iter;   /**< Maximum total number of GMRES iterations (default: 1000) */
    idx_t  restart;    /**< Restart parameter k for GMRES(k) (default: 30) */
    double tol;        /**< Convergence tolerance on relative residual ||r||/||b|| (default: 1e-10) */
    int    verbose;    /**< If nonzero, print iteration log to stderr (default: 0) */
} sparse_gmres_opts_t;

/**
 * @brief Result information from an iterative solve.
 *
 * Populated by sparse_solve_cg() and sparse_solve_gmres() on return.
 * Pass NULL if result information is not needed.
 */
typedef struct {
    idx_t  iterations;     /**< Number of iterations performed */
    double residual_norm;  /**< Final true relative residual norm ||b - A*x|| / ||b|| */
    int    converged;      /**< Nonzero if solver converged within tolerance */
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
typedef sparse_err_t (*sparse_precond_fn)(const void *ctx, idx_t n,
                                          const double *r, double *z);

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
 * @threadsafety Read-only on A. Safe to call concurrently on the same matrix
 *               with different b/x vectors.
 */
sparse_err_t sparse_solve_cg(const SparseMatrix *A,
                              const double *b, double *x,
                              const sparse_iter_opts_t *opts,
                              sparse_precond_fn precond,
                              const void *precond_ctx,
                              sparse_iter_result_t *result);

/**
 * @brief Solve A*x = b using the restarted GMRES(k) method.
 *
 * GMRES is applicable to general (possibly unsymmetric) square matrices.
 * Uses the Arnoldi process with Givens rotations for the Hessenberg
 * least-squares problem. Supports left preconditioning.
 *
 * The input x is used as the initial guess (pass a zero vector for no guess).
 *
 * @param A           The coefficient matrix (not modified). Must be square.
 * @param b           Right-hand side vector of length n.
 * @param x           On entry, initial guess; on exit, approximate solution.
 * @param opts        Solver options (NULL for defaults).
 * @param precond     Preconditioner callback for left preconditioning (NULL for none).
 * @param precond_ctx Context pointer passed to precond callback.
 * @param result      Output: iteration count, residual, convergence flag (may be NULL).
 * @return SPARSE_OK if converged within tolerance.
 * @return SPARSE_ERR_NOT_CONVERGED if max_iter exceeded without convergence.
 * @return SPARSE_ERR_NULL if A, b, or x is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_BADARG if opts has negative max_iter or tol, or restart <= 0.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails or overflows.
 *
 * @threadsafety Read-only on A. Safe to call concurrently on the same matrix
 *               with different b/x vectors.
 */
sparse_err_t sparse_solve_gmres(const SparseMatrix *A,
                                 const double *b, double *x,
                                 const sparse_gmres_opts_t *opts,
                                 sparse_precond_fn precond,
                                 const void *precond_ctx,
                                 sparse_iter_result_t *result);

#endif /* SPARSE_ITERATIVE_H */
