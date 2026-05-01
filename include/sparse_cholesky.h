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
 * @brief Cholesky numeric backend selector.
 *
 * Since Sprint 18 Day 11, `sparse_cholesky_factor_opts` dispatches
 * between the linked-list kernel and the CSC working-format kernel
 * based on matrix size — but callers can force either path via
 * `sparse_cholesky_opts_t::backend`.
 *
 * - `SPARSE_CHOL_BACKEND_AUTO` (default, zero-initialised): use the
 *   CSC backend when `mat->rows >= SPARSE_CSC_THRESHOLD`, otherwise
 *   the linked-list backend.
 * - `SPARSE_CHOL_BACKEND_LINKED_LIST`: always use the linked-list
 *   kernel regardless of dimension.
 * - `SPARSE_CHOL_BACKEND_CSC`: always use the CSC kernel (including
 *   `chol_csc_writeback_to_sparse` to transplant the result back
 *   into `mat`).
 */
typedef enum {
    SPARSE_CHOL_BACKEND_AUTO = 0,
    SPARSE_CHOL_BACKEND_LINKED_LIST = 1,
    SPARSE_CHOL_BACKEND_CSC = 2,
} sparse_chol_backend_t;

/**
 * @brief Options for Cholesky factorization with optional fill-reducing reordering.
 *
 * @warning **ABI break in v2.0.0.**  Sprint 18 Day 11 added the
 * `backend` and `used_csc_path` fields to this struct, changing its
 * size (and therefore layout) relative to the v1.x version shipped
 * through Sprint 17.  The library's `VERSION` bumped to `2.0.0` to
 * signal the break.  Source-level compatibility is preserved:
 * zero-initialising `sparse_cholesky_opts_t` still yields the
 * expected defaults (AUTO backend, no telemetry), so pre-Sprint-18
 * caller code compiles and links against v2.x without edits.
 * Pre-compiled downstream binaries linked against v1.x must be
 * recompiled against v2.x — stack-allocating the old (4-byte)
 * struct would cause the new library to read past its end.
 *
 * @note **Transparent CSC dispatch.**  `sparse_cholesky_factor_opts`
 * routes to the CSC supernodal kernel whenever `mat->rows >=
 * SPARSE_CSC_THRESHOLD` (defined in `sparse_matrix.h`, default 100)
 * and `backend == SPARSE_CHOL_BACKEND_AUTO`.  The dispatch runs
 * `sparse_analyze` → `chol_csc_from_sparse_with_analysis` →
 * `chol_csc_eliminate_supernodal` → `chol_csc_writeback_to_sparse`
 * internally, so callers receive the standard `SparseMatrix` result
 * format regardless of which backend ran.  For smaller matrices the
 * linked-list kernel runs instead.  Set `backend` explicitly to
 * force one path on the same binary — benchmarks and tests use
 * `SPARSE_CHOL_BACKEND_CSC` and `SPARSE_CHOL_BACKEND_LINKED_LIST` to
 * exercise both sides.  For current speedups see
 * `benchmarks/bench_chol_csc.c` and
 * `docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md` — measurements vary
 * with machine and run, so the benchmark output is the source of
 * truth rather than a baked-in figure.
 *
 * The optional `used_csc_path` output pointer is set to 1 when the
 * CSC backend ran and 0 when the linked-list backend ran.  Pass NULL
 * if the caller does not need this telemetry.
 */
typedef struct {
    sparse_reorder_t reorder;      /**< Fill-reducing reordering (NONE, RCM, AMD, or ND —
                                        ND is best on 2D / 3D PDE meshes, see
                                        sparse_reorder.h) */
    sparse_chol_backend_t backend; /**< AUTO dispatches by size; LINKED_LIST / CSC force a path */
    int *used_csc_path;            /**< Optional output: set to 1 if CSC ran, 0 if linked-list */
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
