#ifndef SPARSE_LU_H
#define SPARSE_LU_H

#include "sparse_matrix.h"

/*
 * LU decomposition with complete pivoting: P * A * Q = L * U
 *
 * The factorization is performed in-place: after calling sparse_lu_factor,
 * the matrix stores L (below diagonal, unit diagonal not stored) and
 * U (on and above diagonal). The permutations P and Q are stored in the
 * matrix's row_perm and col_perm arrays.
 *
 * tol: pivot tolerance. If the largest pivot candidate is < tol,
 *      the matrix is declared singular and SPARSE_ERR_SINGULAR is returned.
 */
sparse_err_t sparse_lu_factor(SparseMatrix *mat, double tol);

/*
 * Solve A*x = b using a previously factored matrix.
 *
 * mat must have been factored by sparse_lu_factor.
 * b and x are vectors of length n. x may alias b.
 */
sparse_err_t sparse_lu_solve(const SparseMatrix *mat,
                             const double *b, double *x);

/*
 * Individual solver phases (exposed for testing/advanced use)
 */

/* pb[i] = b[row_perm[i]] */
sparse_err_t sparse_apply_row_perm(const SparseMatrix *mat,
                                   const double *b, double *pb);

/* x[i] = z[col_perm[i]] */
sparse_err_t sparse_apply_inv_col_perm(const SparseMatrix *mat,
                                       const double *z, double *x);

/* Solve L*y = pb (unit lower triangular) */
sparse_err_t sparse_forward_sub(const SparseMatrix *mat,
                                const double *pb, double *y);

/* Solve U*z = y (upper triangular) */
sparse_err_t sparse_backward_sub(const SparseMatrix *mat,
                                 const double *y, double *z);

/*
 * Iterative refinement: improve solution accuracy.
 *
 * mat_orig: the original (unfactored) matrix A
 * mat_lu:   the LU-factored matrix
 * b:        right-hand side
 * x:        solution (modified in-place)
 * max_iters: maximum refinement iterations
 * tol:       stop when ||r|| / ||b|| < tol
 */
sparse_err_t sparse_lu_refine(const SparseMatrix *mat_orig,
                              const SparseMatrix *mat_lu,
                              const double *b, double *x,
                              int max_iters, double tol);

#endif /* SPARSE_LU_H */
