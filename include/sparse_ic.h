#ifndef SPARSE_IC_H
#define SPARSE_IC_H

/**
 * @file sparse_ic.h
 * @brief Incomplete Cholesky IC(0) preconditioner for iterative solvers.
 *
 * Provides IC(0) factorization: an approximate Cholesky decomposition that
 * preserves the sparsity pattern of the lower triangle of A. The resulting
 * factor satisfies L*L^T ≈ A and can be used as a preconditioner for Krylov
 * solvers such as CG and MINRES.
 *
 * IC(0) is the symmetric analogue of ILU(0): where ILU(0) produces L and U
 * factors for general matrices, IC(0) produces a single lower triangular L
 * such that L*L^T approximates a symmetric positive-definite (SPD) matrix.
 * Because it preserves symmetry, IC(0) is intended for SPD systems and is a
 * natural preconditioner for CG.
 *
 * The factorization reuses @ref sparse_ilu_t for storage: @c L stores the
 * lower factor and @c U stores L^T. Use sparse_ic_precond() when passing the
 * factor to solver APIs that accept @ref sparse_precond_fn callbacks.
 *
 * **Usage pattern:**
 * @code
 *   SparseMatrix *A = ...;  // SPD coefficient matrix
 *   sparse_ilu_t ic;
 *   sparse_ic_factor(A, &ic);
 *
 *   // Use as preconditioner with CG:
 *   sparse_solve_cg(A, b, x, &opts, sparse_ic_precond, &ic, &result);
 *
 *   sparse_ic_free(&ic);
 * @endcode
 */

#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_matrix.h"

/**
 * @brief Compute the IC(0) factorization of a symmetric positive-definite matrix.
 *
 * IC(0) computes an approximate lower triangular factor L such that L*L^T ≈ A,
 * where the sparsity pattern of L is the lower triangle of A (no fill-in
 * beyond the original nonzero positions).
 *
 * The original matrix A is not modified. The output object is reset on entry,
 * so sparse_ic_free() is safe after an error return.
 *
 * @note **SPD requirement:** IC(0) requires the input matrix to be symmetric
 *       positive-definite. If A is indefinite or non-symmetric, the
 *       factorization will fail with SPARSE_ERR_NOT_SPD. For indefinite
 *       symmetric systems, use ILU(0) or ILUT instead.
 *
 * @note **Tolerance semantics:** The factorization computes and caches
 *       ||A||_inf in ic->factor_norm. Singularity detection uses a
 *       norm-relative threshold: a diagonal entry is rejected if
 *       L(k,k)^2 < SPARSE_DROP_TOL × ||A||_inf.
 *
 * @pre A must be symmetric positive-definite.
 * @pre A must be square.
 *
 * @param A   The SPD matrix to factor (not modified). Must have identity
 *            row/column state.
 * @param ic  Output: IC(0) factors stored in sparse_ilu_t. L holds the lower
 *            triangular factor and U holds L^T. Must be freed with
 *            sparse_ic_free() after success.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or ic is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_BADARG if A has been previously factored or has
 *         non-identity permutations.
 * @return SPARSE_ERR_NOT_SPD if A is not symmetric or a non-positive diagonal
 *         is encountered during factorization (breakdown).
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_ic_factor(const SparseMatrix *A, sparse_ilu_t *ic);

/**
 * @brief Apply the IC(0) preconditioner: solve L*L^T*z = r.
 *
 * Performs forward substitution (L*y = r) followed by backward substitution
 * (L^T*z = y). z is overwritten and may alias r.
 *
 * @param ic  The IC(0) factors from sparse_ic_factor().
 * @param r   Input vector (right-hand side) of length n.
 * @param z   Output vector (preconditioned result) of length n.
 * @return SPARSE_OK on success (including the n==0 no-op case).
 * @return SPARSE_ERR_NULL if any argument is NULL, or if ic->L/ic->U are
 *         NULL when n > 0 (e.g., factorization was not performed or failed).
 * @return SPARSE_ERR_SINGULAR if an L diagonal entry is zero or near-zero.
 */
sparse_err_t sparse_ic_solve(const sparse_ilu_t *ic, const double *r, double *z);

/**
 * @brief Preconditioner callback compatible with sparse_precond_fn.
 *
 * Wraps sparse_ic_solve() for use with sparse_solve_cg(),
 * sparse_solve_minres(), and other APIs that accept sparse_precond_fn. Pass a
 * pointer to a sparse_ilu_t returned by sparse_ic_factor() as the context.
 *
 * @param ctx  Pointer to a sparse_ilu_t (cast from const void*).
 * @param n    Vector length (must match ic->n).
 * @param r    Input vector.
 * @param z    Output vector.
 * @return SPARSE_OK on success, or an error code.
 * @return SPARSE_ERR_NULL if ctx, r, or z is NULL.
 * @return SPARSE_ERR_SHAPE if n does not match the factor dimension.
 */
sparse_err_t sparse_ic_precond(const void *ctx, idx_t n, const double *r, double *z);

/**
 * @brief Free the IC(0) factorization data.
 *
 * Delegates to sparse_ilu_free(). NULL and zeroed structs are safe.
 *
 * @param ic  The IC(0) factors to free.
 */
void sparse_ic_free(sparse_ilu_t *ic);

#endif /* SPARSE_IC_H */
