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

#include "sparse_matrix.h"
#include "sparse_iterative.h"

/**
 * @brief ILU(0) factorization data.
 *
 * Stores the L and U factors from incomplete LU factorization.
 * L is unit lower triangular, U is upper triangular with diagonal.
 */
typedef struct {
    SparseMatrix *L;   /**< Unit lower triangular factor */
    SparseMatrix *U;   /**< Upper triangular factor (includes diagonal) */
    idx_t n;           /**< Matrix dimension */
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
 * @param A    The matrix to factor (not modified). Must be square.
 * @param ilu  Output: ILU(0) factors. Must be freed with sparse_ilu_free().
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or ilu is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
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
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 */
sparse_err_t sparse_ilu_solve(const sparse_ilu_t *ilu,
                               const double *r, double *z);

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
sparse_err_t sparse_ilu_precond(const void *ctx, idx_t n,
                                 const double *r, double *z);

#endif /* SPARSE_ILU_H */
