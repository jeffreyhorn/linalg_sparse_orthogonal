#ifndef SPARSE_BIDIAG_H
#define SPARSE_BIDIAG_H

/**
 * @file sparse_bidiag.h
 * @brief Householder bidiagonalization for SVD preprocessing.
 *
 * Reduces a sparse m×n matrix A to upper bidiagonal form B = U^T * A * V
 * using alternating left and right Householder reflections. U and V are
 * stored implicitly as Householder vector sequences.
 *
 * This is the first phase of the Golub-Kahan SVD algorithm.
 */

#include "sparse_matrix.h"

/**
 * @brief Bidiagonal factorization data.
 *
 * Stores the bidiagonal matrix B (diagonal + superdiagonal) and the
 * left/right Householder reflector sequences for U and V.
 *
 * Callers must call sparse_bidiag_free() before reusing.
 */
typedef struct {
    double *diag;      /**< Diagonal entries of B, length min(m,n) */
    double *superdiag; /**< Superdiagonal entries of B, length min(m,n)-1 */
    double **u_vecs;   /**< Left Householder vectors, u_vecs[k] has length m-k */
    double *u_betas;   /**< Left Householder scalars, length min(m,n) */
    double **v_vecs;   /**< Right Householder vectors, v_vecs[k] has length n-k-1 */
    double *v_betas;   /**< Right Householder scalars, length min(m,n)-1 */
    idx_t m;           /**< Number of rows of original A */
    idx_t n;           /**< Number of columns of original A */
    int transposed;    /**< Nonzero if A was transposed internally (m < n case).
                            When set, u_vecs/u_betas are for A^T's left reflectors
                            (length n-i) and v_vecs/v_betas are for A^T's right
                            reflectors (length m-i-1). SVD code must swap U↔V. */
} sparse_bidiag_t;

/**
 * @brief Compute Householder bidiagonalization: A = U * B * V^T.
 *
 * Reduces A to upper bidiagonal form using alternating left and right
 * Householder reflections. The result satisfies A = U * B * V^T where
 * U is m×m orthogonal, V is n×n orthogonal, and B is m×n with nonzeros
 * only on the diagonal and first superdiagonal.
 *
 * Supports both tall/square (m >= n) and wide (m < n) matrices. For
 * m < n, the matrix is transposed internally and U/V are swapped.
 * Non-identity row/column permutations are rejected with SPARSE_ERR_BADARG.
 *
 * @param A      The matrix to factor (not modified). Must have identity permutations.
 * @param bidiag Output: bidiagonal factors. Must be freed with sparse_bidiag_free().
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or bidiag is NULL.
 * @return SPARSE_ERR_BADARG if A has non-identity permutations.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_bidiag_factor(const SparseMatrix *A, sparse_bidiag_t *bidiag);

/**
 * @brief Free bidiagonal factorization data.
 *
 * @param bidiag The bidiag factors to free. Safe to call on a zeroed struct.
 */
void sparse_bidiag_free(sparse_bidiag_t *bidiag);

#endif /* SPARSE_BIDIAG_H */
