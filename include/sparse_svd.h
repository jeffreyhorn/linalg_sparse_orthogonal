#ifndef SPARSE_SVD_H
#define SPARSE_SVD_H

/**
 * @file sparse_svd.h
 * @brief Sparse Singular Value Decomposition (SVD).
 *
 * Computes A = U * diag(sigma) * V^T via Golub-Kahan bidiagonalization
 * followed by implicit QR iteration on the bidiagonal.
 *
 * **Usage pattern:**
 * @code
 *   SparseMatrix *A = ...;  // m×n matrix
 *   sparse_svd_opts_t opts = { .compute_uv = 1, .economy = 1 };
 *   sparse_svd_t svd;
 *   sparse_svd_compute(A, &opts, &svd);
 *
 *   // svd.sigma[0..k-1] are singular values in descending order
 *   // svd.U is m×k column-major, svd.Vt is k×n column-major
 *
 *   sparse_svd_free(&svd);
 * @endcode
 */

#include "sparse_bidiag.h"
#include "sparse_matrix.h"

/**
 * @brief SVD computation options.
 */
typedef struct {
    int compute_uv; /**< If nonzero, compute U and V^T (default: 0 = singular values only) */
    int economy;    /**< If nonzero and compute_uv, produce thin U (m×k) and V^T (k×n)
                         where k = min(m,n). Otherwise full U (m×m), V^T (n×n). (default: 0) */
    idx_t max_iter; /**< Maximum QR iterations (0 for default: 30*k) */
    double tol;     /**< Convergence tolerance for superdiagonal entries (0 for default: 1e-14) */
} sparse_svd_opts_t;

/**
 * @brief SVD result data.
 *
 * Stores singular values and optionally the left/right singular vectors.
 * Callers must call sparse_svd_free() before reusing.
 */
typedef struct {
    double *sigma; /**< Singular values in descending order, length k = min(m,n) */
    double *U;     /**< Left singular vectors (column-major). NULL if compute_uv=0.
                        Economy: m×k. Full: m×m. */
    double *Vt;    /**< Right singular vectors transposed (column-major). NULL if compute_uv=0.
                        Economy: k×n. Full: n×n. */
    idx_t m;       /**< Number of rows of original A */
    idx_t n;       /**< Number of columns of original A */
    idx_t k;       /**< min(m,n) — number of singular values */
    int economy;   /**< Nonzero if economy (thin) SVD was computed */
} sparse_svd_t;

/**
 * @brief Compute SVD of a sparse matrix: A = U * diag(sigma) * V^T.
 *
 * @param A    The matrix to decompose (not modified). Must have identity permutations.
 * @param opts Options (NULL for defaults: singular values only).
 * @param svd  Output: SVD result. Must be freed with sparse_svd_free().
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or svd is NULL.
 * @return SPARSE_ERR_BADARG if A has non-identity permutations.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 * @return SPARSE_ERR_NOT_CONVERGED if QR iteration fails to converge.
 */
sparse_err_t sparse_svd_compute(const SparseMatrix *A, const sparse_svd_opts_t *opts,
                                sparse_svd_t *svd);

/**
 * @brief Free SVD result data.
 *
 * @param svd  The SVD result to free. Safe to call on a zeroed struct.
 */
void sparse_svd_free(sparse_svd_t *svd);

/**
 * @brief Extract explicit U and V matrices from a bidiagonal factorization.
 *
 * Applies the stored Householder reflectors to form dense U (m×k) and V (n×k)
 * matrices from a sparse_bidiag_t factorization.
 *
 * @param bd       The bidiagonal factorization.
 * @param U        Output: m×k column-major matrix (caller allocates m*k doubles). May be NULL.
 * @param V        Output: n×k column-major matrix (caller allocates n*k doubles). May be NULL.
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_svd_extract_uv(const sparse_bidiag_t *bd, double *U, double *V);

#endif /* SPARSE_SVD_H */
