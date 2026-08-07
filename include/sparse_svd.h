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
 *   sparse_svd_opts_t opts = {
 *       .compute_uv = 1,
 *       .economy = 1,  // thin/economy U and V^T
 *   };
 *   sparse_svd_t svd;
 *   sparse_svd_compute(A, &opts, &svd);
 *
 *   // svd.sigma[0..k-1] are singular values in descending order
 *   // svd.U is m×k column-major, svd.Vt is k×n column-major
 *   // Set opts.economy = 0 to request full U (m×m) and V^T (n×n)
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
    int compute_uv; /**< If nonzero, compute U and V^T (default: 0 = singular values only). */
    int economy;    /**< When compute_uv is set: nonzero produces thin U (m×k col-major,
                         leading dim m) and V^T (k×n col-major, leading dim k) where
                         k = min(m,n); zero produces full U (m×m col-major, leading dim m)
                         and V^T (n×n col-major, leading dim n) — the padded columns/rows
                         past index k are orthonormal completions of the basis (MGS over
                         canonical unit vectors). */
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
 * @pre A must have identity permutations (not previously factored or reordered).
 *      A is not modified.
 *
 * If @p opts is NULL, this routine computes singular values only. When
 * `opts->compute_uv` is nonzero, `opts->economy = 1` returns thin/economy
 * factors (`U` is m×k, `V^T` is k×n) and `opts->economy = 0` returns full
 * orthonormal factors (`U` is m×m, `V^T` is n×n), where k = min(m,n).
 *
 * @param A    The matrix to decompose (not modified).
 * @param opts Options (NULL for defaults: singular values only).
 * @param svd  Output: SVD result. Must be freed with sparse_svd_free().
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or svd is NULL.
 * @return SPARSE_ERR_BADARG if A has non-identity permutations, or if
 *         opts requests a negative max_iter / tol.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 * @return SPARSE_ERR_NOT_CONVERGED if QR iteration fails to converge.
 *
 * @note Uses the zero-diagonal chase (Golub & Van Loan §8.6.2) for
 *       rank-deficient bidiagonals with near-zero diagonal entries.
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
 * When `opts->compute_uv` is set with `opts->economy = 1`, approximate
 * thin left and right singular vectors are recovered from the Lanczos basis.
 * The vectors satisfy A*v_i ≈ sigma_i * u_i for the top-k triplets.
 * A maintained Sprint 140 corpus fixture covers one generated 8x6 diagonal
 * case with clustered/repeated leading singular values, top-3 subspace
 * projectors, triplet residuals, orthogonality, default-budget success, and
 * tight-budget fail-closed behavior. That fixture-local evidence is not a
 * broad repeated-spectrum, external-library parity, performance, or
 * partial-result guarantee.
 * Partial SVD does not support the full-U / full-V^T mode.
 *
 * @param A    The matrix (not modified). Must have identity permutations.
 * @param k    Number of singular values to compute.
 * @param opts Options (NULL for defaults). `max_iter` and `tol` tune the
 *             Lanczos / bidiagonal iteration. Set `compute_uv = 1` together
 *             with `economy = 1` to recover approximate thin singular vectors;
 *             `compute_uv = 1` with `economy = 0` is rejected.
 * @param svd  Output: partial SVD result. `sigma` has k entries. When
 *             `opts->compute_uv && opts->economy`, `U` is m×k and `V^T` is
 *             k×n; otherwise `U` and `V^T` are NULL. Must be freed with
 *             sparse_svd_free().
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or svd is NULL.
 * @return SPARSE_ERR_BADARG if k <= 0 or k > min(m,n), if A has non-identity
 *         permutations, if opts has negative max_iter/tol, or if
 *         `opts->compute_uv` is set without `opts->economy = 1`.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 * @return SPARSE_ERR_NOT_CONVERGED if bidiagonal SVD iteration fails.
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
 * @param tol  Tolerance (0 for default, negative rejected). Singular values <= tol
 *             are treated as zero.
 * @param rank Output: the numerical rank.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or rank is NULL.
 * @return SPARSE_ERR_BADARG if A has non-identity permutations or tol < 0.
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
 * @param tol   Tolerance for rank determination (0 for default, negative rejected).
 * @param pinv  Output: dense n×m column-major array (caller must free).
 *              Set to NULL on failure.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or pinv is NULL.
 * @return SPARSE_ERR_BADARG if A has non-identity permutations or tol < 0.
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

/**
 * @brief Compute the best rank-k approximation as a sparse matrix.
 *
 * Returns A_k = U_k * Sigma_k * V_k^T as a SparseMatrix, dropping entries
 * whose absolute value is below @p drop_tol. The final sparse output uses
 * less memory than the dense array from sparse_svd_lowrank() when the
 * low-rank approximation is itself sparse.
 *
 * @note Internally allocates a temporary m*n dense accumulator during
 *       construction. Peak memory is comparable to sparse_svd_lowrank().
 *
 * @note Set environment variable SPARSE_SVD_LOWRANK_OUTER=on to route through
 *       an alternative per-cell outer-product accumulator that avoids the
 *       m*n dense intermediate. Output is bit-identical (same per-cell sum
 *       order, same drop_tol cutoff). The env-on path trades the dense
 *       intermediate for O(nnz_result) sparse-insert overhead -- it wins
 *       large memory reductions on min(m,n) >> rank_k fixtures (e.g.
 *       ~76-88 % rss reduction on bcsstk14) with neutral wall (SVD compute
 *       dominates either way). Default off preserves the dense-intermediate path;
 *       opt in for memory-constrained workloads.
 *
 * @param A        The matrix (not modified).
 * @param rank_k   Desired rank (must be 1..min(m,n)).
 * @param drop_tol Entries with |value| < drop_tol are dropped. If <= 0,
 *                 uses default: eps * sigma_1.
 * @param result   Output: sparse m×n matrix (caller must free with sparse_free()).
 *                 Set to NULL on failure.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or result is NULL.
 * @return SPARSE_ERR_BADARG if rank_k is out of range or A has non-identity permutations.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 * @return SPARSE_ERR_NOT_CONVERGED if SVD iteration fails to converge.
 */
sparse_err_t sparse_svd_lowrank_sparse(const SparseMatrix *A, idx_t rank_k, double drop_tol,
                                       SparseMatrix **result);

/**
 * @brief Estimate the 2-norm condition number of a matrix via SVD.
 *
 * Computes cond(A) = sigma_max / sigma_min using the full SVD.
 * Returns INFINITY for singular matrices (sigma_min below tolerance).
 *
 * @param A    The matrix (not modified). Must have identity permutations
 *             (i.e., must not have been previously factored in-place).
 * @param err  Output: error code (SPARSE_OK on success). May be NULL.
 *             Set to SPARSE_ERR_BADARG if A has non-identity permutations.
 * @return The condition number, or INFINITY if A is singular.
 *         Returns INFINITY and sets *err on failure.
 */
double sparse_cond(const SparseMatrix *A, sparse_err_t *err);

#endif /* SPARSE_SVD_H */
