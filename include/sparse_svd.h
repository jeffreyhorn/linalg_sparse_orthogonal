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
    int compute_uv; /**< If nonzero, compute U and V^T (default: 0 = singular values only).
                         Requires economy=1 when set; full SVD (economy=0) is not implemented. */
    int economy;    /**< Must be nonzero when compute_uv is set (only economy/thin SVD is
                         implemented). Produces thin U (m×k) and V^T (k×n) where k = min(m,n). */
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
 * @return SPARSE_ERR_BADARG if A has non-identity permutations, or if
 *         compute_uv is set without economy (full SVD not implemented).
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

/**
 * @brief Compute the k largest singular values via Lanczos bidiagonalization.
 *
 * More efficient than full SVD when k << min(m,n). Uses iterative
 * Lanczos bidiagonalization to build a small k×k bidiagonal, then
 * applies the bidiagonal SVD iteration to extract singular values.
 *
 * This routine returns singular values only. The U and V^T fields in
 * @p svd are left NULL; opts->compute_uv and opts->economy are ignored.
 *
 * @param A    The matrix (not modified). Must have identity permutations.
 * @param k    Number of singular values to compute.
 * @param opts Options (NULL for defaults). Only max_iter and tol are used;
 *             singular vectors are not computed.
 * @param svd  Output: partial SVD result (sigma has k entries; U and V^T are NULL).
 *             Must be freed with sparse_svd_free().
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or svd is NULL.
 * @return SPARSE_ERR_BADARG if k <= 0 or k > min(m,n).
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_svd_partial(const SparseMatrix *A, idx_t k, const sparse_svd_opts_t *opts,
                                sparse_svd_t *svd);

/* ═══════════════════════════════════════════════════════════════════════
 * SVD applications
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Estimate the numerical rank of a matrix via SVD.
 *
 * Counts singular values above a tolerance threshold.
 * Default tolerance: eps * max(m,n) * sigma_max, where eps = 2.2e-16.
 *
 * @param A    The matrix (not modified).
 * @param tol  Tolerance (0 for default). Singular values <= tol are treated as zero.
 * @param rank Output: the numerical rank.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or rank is NULL.
 * @return SPARSE_ERR_BADARG if A has non-identity permutations.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 * @return SPARSE_ERR_NOT_CONVERGED if SVD iteration fails to converge.
 */
sparse_err_t sparse_svd_rank(const SparseMatrix *A, double tol, idx_t *rank);

/**
 * @brief Compute the Moore-Penrose pseudoinverse via SVD.
 *
 * Returns A^+ = V * Sigma^+ * U^T as a dense column-major matrix.
 * Sigma^+ inverts singular values above tolerance and zeros the rest.
 *
 * @param A     The matrix (not modified).
 * @param tol   Tolerance for rank determination (0 for default).
 * @param pinv  Output: dense n×m column-major array (caller must free).
 *              Set to NULL on failure.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or pinv is NULL.
 * @return SPARSE_ERR_BADARG if A has non-identity permutations.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 * @return SPARSE_ERR_NOT_CONVERGED if SVD iteration fails to converge.
 */
sparse_err_t sparse_pinv(const SparseMatrix *A, double tol, double **pinv);

/**
 * @brief Compute the best rank-k approximation via truncated SVD.
 *
 * Returns A_k = U_k * Sigma_k * V_k^T as a dense column-major matrix,
 * which is the closest rank-k matrix to A in Frobenius norm.
 *
 * @param A       The matrix (not modified).
 * @param rank_k  Desired rank (must be 1..min(m,n)).
 * @param lowrank Output: dense m×n column-major array (caller must free).
 *                Set to NULL on failure.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or lowrank is NULL.
 * @return SPARSE_ERR_BADARG if rank_k is out of range or A has non-identity permutations.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 * @return SPARSE_ERR_NOT_CONVERGED if SVD iteration fails to converge.
 */
sparse_err_t sparse_svd_lowrank(const SparseMatrix *A, idx_t rank_k, double **lowrank);

#endif /* SPARSE_SVD_H */
