#ifndef SPARSE_ILU_H
#define SPARSE_ILU_H

/**
 * @file sparse_ilu.h
 * @brief Incomplete LU (ILU(0)) preconditioner for iterative solvers.
 *
 * Provides ILU(0) factorization: an approximate LU decomposition that
 * preserves the sparsity pattern of the original matrix (no fill-in beyond
 * positions where A has nonzeros). The resulting L and U factors are used
 * as a preconditioner for Krylov solvers (CG, GMRES).
 *
 * **Usage pattern:**
 * @code
 *   SparseMatrix *A = ...;  // coefficient matrix
 *   sparse_ilu_t ilu;
 *   sparse_ilu_factor(A, &ilu);
 *
 *   // Use as preconditioner with GMRES:
 *   sparse_solve_gmres(A, b, x, &opts, sparse_ilu_precond, &ilu, &result);
 *
 *   sparse_ilu_free(&ilu);
 * @endcode
 */

#include "sparse_iterative.h"
#include "sparse_matrix.h"

/**
 * @brief ILU factorization data.
 *
 * Stores the L and U factors from incomplete LU factorization.
 * L is unit lower triangular, U is upper triangular with diagonal.
 * The perm field is reserved for future use (e.g., ILUT with row
 * pivoting) and is currently always NULL.
 */
typedef struct {
    SparseMatrix *L; /**< Unit lower triangular factor */
    SparseMatrix *U; /**< Upper triangular factor (includes diagonal) */
    idx_t n;         /**< Matrix dimension */
    idx_t *perm;     /**< Row permutation (NULL if no pivoting). perm[i] = original row. */
} sparse_ilu_t;

/**
 * @brief Compute the ILU(0) factorization of a sparse matrix.
 *
 * ILU(0) computes approximate L and U factors such that L*U ≈ A, where
 * the sparsity pattern of L+U is a subset of the sparsity pattern of A.
 * No fill-in is allowed beyond the original nonzero positions.
 *
 * The original matrix A is not modified.
 *
 * @note The factorization operates on the physical storage of A and requires
 *       identity permutations (i.e., an unfactored matrix).  Passing a matrix
 *       whose permutation arrays have been modified by a prior factorization
 *       will be rejected with SPARSE_ERR_BADARG.
 *
 * @param A    The matrix to factor (not modified). Must be square.
 * @param ilu  Output: ILU(0) factors. Must be freed with sparse_ilu_free().
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or ilu is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_BADARG if A has non-identity permutations (e.g., after LU pivoting).
 * @return SPARSE_ERR_SINGULAR if a zero pivot is encountered.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_ilu_factor(const SparseMatrix *A, sparse_ilu_t *ilu);

/**
 * @brief Apply the ILU(0) preconditioner: solve L*U*z = r.
 *
 * Performs forward substitution (L*y = r) followed by backward
 * substitution (U*z = y).
 *
 * @param ilu  The ILU(0) factors from sparse_ilu_factor().
 * @param r    Input vector (right-hand side) of length n.
 * @param z    Output vector (preconditioned result) of length n.
 * @return SPARSE_OK on success (including the n==0 no-op case, where L/U
 *         may be NULL).
 * @return SPARSE_ERR_NULL if any argument is NULL, or if ilu->L/ilu->U are
 *         NULL when n > 0 (e.g., factorization was not performed or failed).
 * @return SPARSE_ERR_SINGULAR if a U diagonal pivot is zero or near-zero.
 */
sparse_err_t sparse_ilu_solve(const sparse_ilu_t *ilu, const double *r, double *z);

/**
 * @brief Free the ILU(0) factorization data.
 *
 * @param ilu  The ILU(0) factors to free. Safe to call on a zeroed struct.
 */
void sparse_ilu_free(sparse_ilu_t *ilu);

/**
 * @brief Preconditioner callback compatible with sparse_precond_fn.
 *
 * Wraps sparse_ilu_solve() for use as a preconditioner callback with
 * sparse_solve_cg() and sparse_solve_gmres(). Pass a pointer to a
 * sparse_ilu_t as the context.
 *
 * @param ctx  Pointer to a sparse_ilu_t (cast from const void*).
 * @param n    Vector length (must match ilu->n).
 * @param r    Input vector.
 * @param z    Output vector.
 * @return SPARSE_OK on success, or an error code.
 */
sparse_err_t sparse_ilu_precond(const void *ctx, idx_t n, const double *r, double *z);

/* ═══════════════════════════════════════════════════════════════════════
 * ILUT — Incomplete LU with Threshold dropping
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Options for ILUT factorization.
 *
 * Controls the amount of fill-in allowed beyond the original sparsity pattern.
 */
typedef struct {
    double tol;     /**< Drop tolerance: entries with |value| < tol * ||row|| are dropped
                         (default: 1e-3) */
    idx_t max_fill; /**< Maximum number of fill entries per row in L and U (default: 10) */
} sparse_ilut_opts_t;

/**
 * @brief Compute the ILUT factorization of a sparse matrix.
 *
 * ILUT computes approximate L and U factors using dual drop rules:
 * 1. Entries with |value| < tol * ||row_i|| are dropped.
 * 2. At most max_fill largest entries are kept per row in L and U.
 *
 * Unlike ILU(0), ILUT allows controlled fill-in and can handle matrices
 * with structurally zero diagonals (e.g., west0067) by using diagonal
 * modification: when a pivot is too small, a nonzero value is inserted
 * on the diagonal to stabilize the factorization. No row permutation
 * is performed (the @c perm field of @c sparse_ilu_t is unused/NULL).
 *
 * The original matrix A is not modified.
 *
 * @note The factorization operates on the physical storage of A and requires
 *       identity permutations.  Passing a matrix whose permutation arrays
 *       have been modified by a prior factorization will be rejected with
 *       SPARSE_ERR_BADARG.
 *
 * @param A    The matrix to factor (not modified). Must be square.
 * @param opts Factorization options (NULL for defaults: tol=1e-3, max_fill=10).
 * @param ilu  Output: ILUT factors. Must be freed with sparse_ilu_free().
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or ilu is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_BADARG if A has non-identity permutations or opts has
 *         invalid values (tol < 0 or max_fill < 0).
 * @return SPARSE_ERR_SINGULAR if a zero pivot is encountered after diagonal modification.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_ilut_factor(const SparseMatrix *A, const sparse_ilut_opts_t *opts,
                                sparse_ilu_t *ilu);

/**
 * @brief Preconditioner callback for ILUT, compatible with sparse_precond_fn.
 *
 * Identical to sparse_ilu_precond() — ILUT produces the same sparse_ilu_t
 * output format.  This is a convenience alias for clarity.
 *
 * @param ctx  Pointer to a sparse_ilu_t (cast from const void*).
 * @param n    Vector length (must match ilu->n).
 * @param r    Input vector.
 * @param z    Output vector.
 * @return SPARSE_OK on success, or an error code.
 */
sparse_err_t sparse_ilut_precond(const void *ctx, idx_t n, const double *r, double *z);

#endif /* SPARSE_ILU_H */
