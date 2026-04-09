#ifndef SPARSE_LDLT_H
#define SPARSE_LDLT_H

/**
 * @file sparse_ldlt.h
 * @brief Sparse LDL^T factorization for symmetric indefinite matrices.
 *
 * Provides LDL^T factorization with Bunch-Kaufman symmetric pivoting:
 * P*A*P^T = L*D*L^T, where L is unit lower triangular, D is block-diagonal
 * with 1x1 and 2x2 blocks, and P is a symmetric permutation.
 *
 * Bunch-Kaufman pivoting handles symmetric indefinite matrices (KKT systems,
 * saddle-point problems, constrained optimization) by choosing 1x1 or 2x2
 * pivots to maintain bounded element growth without requiring symmetry-breaking
 * row pivoting.
 *
 * **Usage pattern:**
 * @code
 *   SparseMatrix *A = ...;  // symmetric indefinite matrix
 *   sparse_ldlt_t ldlt;
 *   sparse_ldlt_factor(A, &ldlt);
 *   sparse_ldlt_solve(&ldlt, b, x);   // solve A*x = b
 *
 *   // With fill-reducing reordering:
 *   sparse_ldlt_opts_t opts = { .reorder = SPARSE_REORDER_AMD };
 *   sparse_ldlt_t ldlt2;
 *   sparse_ldlt_factor_opts(A, &opts, &ldlt2);
 *   sparse_ldlt_solve(&ldlt2, b, x);  // reorder/unpermute automatic
 *
 *   // Inertia (count of +, -, 0 eigenvalues):
 *   idx_t pos, neg, zero;
 *   sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero);
 *
 *   sparse_ldlt_free(&ldlt);
 *   sparse_ldlt_free(&ldlt2);
 * @endcode
 */

#include "sparse_matrix.h"

/**
 * @brief LDL^T factorization data.
 *
 * Stores the L factor (unit lower triangular), D (block-diagonal with
 * 1x1 and 2x2 blocks), the symmetric pivot permutation, and pivot block
 * size information.
 *
 * For 1x1 pivots at step k: D[k] holds the scalar pivot, D_offdiag[k] = 0,
 * pivot_size[k] = 1.
 *
 * For 2x2 pivots at steps k and k+1: D[k] and D[k+1] hold the diagonal
 * of the 2x2 block, D_offdiag[k] holds the off-diagonal entry,
 * pivot_size[k] = pivot_size[k+1] = 2.
 *
 * Callers must call sparse_ldlt_free() before reusing a sparse_ldlt_t for
 * a new factorization; the factor functions overwrite the struct without
 * freeing prior contents. sparse_ldlt_free() is safe on a zeroed struct.
 */
typedef struct {
    SparseMatrix *L;    /**< Unit lower triangular factor */
    double *D;          /**< Diagonal of D (length n). For 2x2 pivot at k,
                             D[k] and D[k+1] are the 2x2 block diagonal. */
    double *D_offdiag;  /**< Off-diagonal of 2x2 pivot blocks (length n).
                             Zero for 1x1 pivots. For 2x2 pivot at k,
                             D_offdiag[k] = D(k, k+1) = D(k+1, k). */
    int *pivot_size;    /**< Pivot block size at each step (length n).
                             1 for a 1x1 pivot, 2 for a 2x2 pivot. */
    idx_t *perm;        /**< Symmetric permutation (length n).
                             perm[i] = original row/column index.
                             NULL if no reordering was applied. */
    idx_t n;            /**< Matrix dimension */
    double factor_norm; /**< ||A||_inf at factorization time, for relative tolerance */
} sparse_ldlt_t;

/**
 * @brief Options for LDL^T factorization.
 */
typedef struct {
    sparse_reorder_t reorder; /**< Fill-reducing reordering (NONE, RCM, or AMD) */
    double tol;               /**< Pivot tolerance for singularity detection.
                                   0 for default (SPARSE_DROP_TOL). */
} sparse_ldlt_opts_t;

/**
 * @brief Compute the LDL^T factorization of a symmetric matrix.
 *
 * Computes P*A*P^T = L*D*L^T using Bunch-Kaufman symmetric pivoting.
 * L is unit lower triangular (stored as a new SparseMatrix), D is
 * block-diagonal with 1x1 and 2x2 blocks.  The original matrix A is
 * not modified.
 *
 * @note **Tolerance semantics:** The factorization computes and caches
 *       ||A||_inf in ldlt->factor_norm.  Singularity detection uses
 *       norm-relative thresholds: a 1x1 pivot is rejected if
 *       |D[k]| < SPARSE_DROP_TOL * ||A||_inf; a 2x2 pivot block is
 *       rejected if its determinant is near zero relative to ||A||_inf^2.
 *
 * @pre A must be symmetric.  Symmetry is checked at entry.
 * @pre A must have identity permutations (not previously factored or
 *      reordered).  Use a fresh matrix or sparse_copy() of the original.
 *
 * @param A     The symmetric matrix to factor (not modified). Must be square.
 * @param ldlt  Output: LDL^T factors. Must be freed with sparse_ldlt_free().
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or ldlt is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_NOT_SPD if A is not symmetric (name reused for symmetry check).
 * @return SPARSE_ERR_BADARG if A has non-identity permutations.
 * @return SPARSE_ERR_SINGULAR if a singular pivot is encountered.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_ldlt_factor(const SparseMatrix *A, sparse_ldlt_t *ldlt);

/**
 * @brief Compute LDL^T factorization with options including fill-reducing reordering.
 *
 * If opts->reorder != SPARSE_REORDER_NONE, the matrix is symmetrically
 * permuted before factorization.  The reordering permutation is composed
 * with the Bunch-Kaufman pivot permutation and stored so that
 * sparse_ldlt_solve() can automatically unpermute the solution.
 *
 * @param A     The symmetric matrix to factor (not modified). Must be square.
 * @param opts  Factorization options (NULL for defaults: no reordering,
 *              default tolerance).
 * @param ldlt  Output: LDL^T factors. Must be freed with sparse_ldlt_free().
 * @return SPARSE_OK on success, or an error code (see sparse_ldlt_factor()).
 */
sparse_err_t sparse_ldlt_factor_opts(const SparseMatrix *A, const sparse_ldlt_opts_t *opts,
                                     sparse_ldlt_t *ldlt);

/**
 * @brief Solve A*x = b using a previously computed LDL^T factorization.
 *
 * Given P*A*P^T = L*D*L^T, solves:
 *   1. Apply permutation: b_p = P * b
 *   2. Forward substitution: L * y = b_p
 *   3. Diagonal solve: D * z = y  (1x1 and 2x2 blocks)
 *   4. Backward substitution: L^T * w = z
 *   5. Apply inverse permutation: x = P^T * w
 *
 * @param ldlt  The LDL^T factorization from sparse_ldlt_factor().
 * @param b     Right-hand side vector of length n.
 * @param x     Solution vector of length n (overwritten). May alias b.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_BADARG if ldlt has not been factored.
 * @return SPARSE_ERR_SINGULAR if a zero D block is encountered during solve.
 *
 * @par Thread safety: Read-only on ldlt. Safe to call concurrently on the
 *               same factorization with different b/x vectors.
 */
sparse_err_t sparse_ldlt_solve(const sparse_ldlt_t *ldlt, const double *b, double *x);

/**
 * @brief Free the LDL^T factorization data.
 *
 * @param ldlt  The factorization to free. Safe to call on a zeroed struct.
 */
void sparse_ldlt_free(sparse_ldlt_t *ldlt);

/**
 * @brief Compute the inertia of A from its LDL^T factorization.
 *
 * The inertia is the triple (n_pos, n_neg, n_zero) counting the number
 * of positive, negative, and zero eigenvalues of A.  This is determined
 * from the signs of the D blocks:
 *   - 1x1 block D[k] > 0 → one positive eigenvalue
 *   - 1x1 block D[k] < 0 → one negative eigenvalue
 *   - 1x1 block D[k] = 0 → one zero eigenvalue
 *   - 2x2 block with one positive and one negative eigenvalue
 *     (det < 0 → one of each)
 *
 * @param ldlt   The LDL^T factorization.
 * @param n_pos  Output: number of positive eigenvalues. May be NULL.
 * @param n_neg  Output: number of negative eigenvalues. May be NULL.
 * @param n_zero Output: number of zero eigenvalues. May be NULL.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if ldlt is NULL.
 */
sparse_err_t sparse_ldlt_inertia(const sparse_ldlt_t *ldlt, idx_t *n_pos, idx_t *n_neg,
                                 idx_t *n_zero);

/**
 * @brief Iterative refinement to improve LDL^T solution accuracy.
 *
 * Computes the residual r = b - A*x using the original matrix, solves
 * A*d = r using the LDL^T factorization, and updates x += d.
 *
 * @param A          The original (unfactored) symmetric matrix.
 * @param ldlt       The LDL^T factorization (from sparse_ldlt_factor).
 * @param b          Right-hand side vector of length n.
 * @param x          Solution vector of length n (modified in-place).
 * @param max_iters  Maximum number of refinement iterations.
 * @param tol        Convergence tolerance on relative residual.
 * @return SPARSE_OK on success, SPARSE_ERR_NULL if any argument is NULL.
 */
sparse_err_t sparse_ldlt_refine(const SparseMatrix *A, const sparse_ldlt_t *ldlt, const double *b,
                                double *x, int max_iters, double tol);

/**
 * @brief Estimate the 1-norm condition number of A from its LDL^T factors.
 *
 * Uses Hager's algorithm to estimate ||A^{-1}||_1 without forming the
 * inverse. Since A is symmetric, A^T = A and the same factorization is
 * used for both forward and transpose solves.
 *
 * @param A       The original (unfactored) symmetric matrix.
 * @param ldlt    The LDL^T factorization.
 * @param condest Output: condition number estimate.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 */
sparse_err_t sparse_ldlt_condest(const SparseMatrix *A, const sparse_ldlt_t *ldlt, double *condest);

#endif /* SPARSE_LDLT_H */
