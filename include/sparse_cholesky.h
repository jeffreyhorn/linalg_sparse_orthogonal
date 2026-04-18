#ifndef SPARSE_CHOLESKY_H
#define SPARSE_CHOLESKY_H

/**
 * @file sparse_cholesky.h
 * @brief Cholesky decomposition for sparse symmetric positive-definite matrices.
 *
 * Provides in-place Cholesky factorization A = L*L^T where L is lower
 * triangular. Exploits symmetry: only the lower triangle is stored after
 * factorization. No pivoting is needed for SPD matrices.
 *
 * **Usage pattern:**
 * @code
 *   SparseMatrix *A = sparse_create(n, n);
 *   // ... populate A with SPD entries ...
 *   SparseMatrix *L = sparse_copy(A);
 *   sparse_cholesky_factor(L);
 *   sparse_cholesky_solve(L, b, x);   // solve A*x = b
 *
 *   // With fill-reducing reordering:
 *   SparseMatrix *L2 = sparse_copy(A);
 *   sparse_cholesky_opts_t opts = { SPARSE_REORDER_AMD };
 *   sparse_cholesky_factor_opts(L2, &opts);
 *   sparse_cholesky_solve(L2, b, x);  // reorder/unpermute automatic
 * @endcode
 */

#include "sparse_matrix.h"

/**
 * @brief Options for Cholesky factorization with optional fill-reducing reordering.
 *
 * @note **Selecting the CSC numeric backend.** For larger SPD systems
 * (`n >= SPARSE_CSC_THRESHOLD`, defined in `sparse_matrix.h`), the CSC
 * working-format kernel from Sprint 17 delivers a measured factor-time
 * speedup on structural-mechanics problems such as bcsstk04 (n=132,
 * nnz=3648).  For current reported numbers see
 * `benchmarks/bench_chol_csc.c` and
 * `docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md` — measurements vary
 * with machine and run, so the benchmark output is the source of truth
 * rather than a baked-in figure in this header.  Today the CSC kernel
 * is reached via the internal `chol_csc_factor` /
 * `chol_csc_factor_solve` APIs (declared in
 * `src/sparse_chol_csc_internal.h`); a transparent threshold-based
 * dispatch through `sparse_cholesky_factor_opts` is tracked as
 * follow-up work.
 */
typedef struct {
    sparse_reorder_t reorder; /**< Fill-reducing reordering (NONE, RCM, or AMD) */
} sparse_cholesky_opts_t;

/**
 * @brief Compute the Cholesky factorization of a sparse SPD matrix in-place.
 *
 * Computes A = L*L^T where L is lower triangular. The lower triangle of mat
 * is overwritten with L; the upper triangle entries are removed. No pivoting
 * is performed (SPD matrices have all-positive pivots).
 *
 * Fill-in entries with |value| < SPARSE_DROP_TOL * L(k,k) are dropped to
 * control memory growth.
 *
 * @note **Tolerance semantics:** Factorization computes and caches ||A||_inf.
 *       The solve phase checks each L(i,i) against a norm-relative threshold:
 *       |L(i,i)| < SPARSE_DROP_TOL × sqrt(||A||_inf).  The square root is
 *       used because Cholesky factors scale as sqrt(A).  This prevents
 *       false-singular detection on uniformly small SPD matrices.
 *
 * @pre mat must be symmetric positive-definite.  Symmetry is checked;
 *      positive-definiteness is verified during factorization (non-positive
 *      pivot → SPARSE_ERR_NOT_SPD).
 * @pre mat must not be needed after factorization — use sparse_copy() first
 *      to preserve the original.  The upper triangle is removed and the
 *      lower triangle is overwritten with L.
 *
 * @param mat  The SPD matrix to factor (modified in-place). Must be square
 *             and symmetric. After factorization, contains L in the lower triangle.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if mat is NULL.
 * @return SPARSE_ERR_SHAPE if mat is not square.
 * @return SPARSE_ERR_NOT_SPD if the matrix is not symmetric (checked first)
 *         or a non-positive pivot is encountered during factorization.
 *
 * @par Thread safety: Mutates mat. Not safe to call concurrently on the same matrix.
 *               Safe to call concurrently on different matrices.
 */
sparse_err_t sparse_cholesky_factor(SparseMatrix *mat);

/**
 * @brief Compute Cholesky factorization with options including fill-reducing reordering.
 *
 * If opts->reorder != SPARSE_REORDER_NONE, the matrix is symmetrically
 * permuted before factorization. The reordering permutation is stored in
 * the matrix so that sparse_cholesky_solve() can automatically unpermute.
 *
 * @param mat   The SPD matrix to factor (modified in-place).
 * @param opts  Factorization options.
 * @return SPARSE_OK on success, or an error code.
 */
sparse_err_t sparse_cholesky_factor_opts(SparseMatrix *mat, const sparse_cholesky_opts_t *opts);

/**
 * @brief Solve A*x = b using a Cholesky-factored matrix.
 *
 * Given L from sparse_cholesky_factor(), solves:
 *   L * y = b    (forward substitution)
 *   L^T * x = y  (backward substitution)
 *
 * If the matrix was factored with reordering, b is automatically permuted
 * and x is automatically unpermuted.
 *
 * @param mat  A matrix that has been factored by sparse_cholesky_factor().
 * @param b    Right-hand side vector of length n.
 * @param x    Solution vector of length n (overwritten). May alias b.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_BADARG if mat has not been factored.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 *
 * @par Thread safety: Read-only on mat. Safe to call concurrently on the same
 *               factored matrix with different b/x vectors.
 */
sparse_err_t sparse_cholesky_solve(const SparseMatrix *mat, const double *b, double *x);

#endif /* SPARSE_CHOLESKY_H */
