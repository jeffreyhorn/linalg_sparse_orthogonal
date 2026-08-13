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
 * Bunch-Kaufman pivoting handles symmetric indefinite matrices such as KKT and
 * saddle-point systems by choosing 1x1 or 2x2 pivots without symmetry-breaking
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
 *
 * This header owns one-shot LDL^T factors and solves. Use the
 * `sparse_analysis.h` analyze/factor/refactor path when one symbolic analysis
 * should be reused across repeated direct solves.
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
 * Factor functions overwrite this struct without freeing existing contents.
 * Call sparse_ldlt_free() before reusing a populated object. A zeroed
 * sparse_ldlt_t is valid to pass to sparse_ldlt_free().
 *
 * This owned factor object is separate from the repeated-run direct lifecycle
 * in `sparse_analysis.h`.
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
    idx_t *perm;        /**< Overall symmetric permutation used by the factorization
                             (length n), mapping factorization order to original
                             row/column indices: perm[i] = original row/column index.
                             Includes any fill-reducing reordering composed with
                             Bunch-Kaufman pivoting; may be the identity permutation
                             but is generally non-NULL after successful factorization. */
    idx_t n;            /**< Matrix dimension */
    double factor_norm; /**< ||A||_inf at factorization time, for relative tolerance */
    double tol;         /**< Effective pivot/drop tolerance used during factorization.
                             Solve, refine, and condest use this same tolerance for
                             consistency with factorization's singularity criteria. */
} sparse_ldlt_t;

/**
 * @brief LDL^T numeric backend selector.
 *
 * `sparse_ldlt_factor_opts` can select the linked-list or CSC LDL^T numeric
 * path. Leave the option at its zero-initialized default
 * (`SPARSE_LDLT_BACKEND_AUTO`) for size-based dispatch, or force a path for
 * focused benchmarks and regression tests.
 *
 * - `SPARSE_LDLT_BACKEND_AUTO` (default, zero-initialised): use the
 *   CSC supernodal backend when `A->rows >= SPARSE_CSC_THRESHOLD`,
 *   otherwise the linked-list backend.
 * - `SPARSE_LDLT_BACKEND_LINKED_LIST`: always use the linked-list
 *   kernel regardless of dimension.
 * - `SPARSE_LDLT_BACKEND_CSC`: always use the CSC pipeline
 *   (`sparse_analyze` / scalar pre-pass resolution →
 *   `ldlt_csc_factor_with_resolved_analysis` -> CSC-to-`sparse_ldlt_t`
 *   writeback). Once selected, the CSC pipeline may finish through either the
 *   batched supernodal completion or the resolved scalar pre-pass fallback.
 *   The empty-matrix edge case (`n == 0`) still routes to the linked-list path.
 */
typedef enum {
    SPARSE_LDLT_BACKEND_AUTO = 0,
    SPARSE_LDLT_BACKEND_LINKED_LIST = 1,
    SPARSE_LDLT_BACKEND_CSC = 2,
} sparse_ldlt_backend_t;

/**
 * @brief Options for LDL^T factorization.
 *
 * @warning **Source rebuild required for v2.1.0 options layout.** The
 * `backend` and `used_csc_path` fields were added after the original
 * reorder/tolerance fields. Positional initializers such as
 * `{SPARSE_REORDER_AMD, 0.0}` still compile because trailing fields
 * zero-initialize, but downstream objects compiled against the older struct
 * layout must be rebuilt.
 *
 * @note **Backend telemetry.** See `sparse_ldlt_backend_t` for per-value
 * semantics. `used_csc_path` is optional; pass NULL when the caller does not
 * need telemetry. When non-NULL, it is set to 1 if the CSC pipeline was
 * selected and 0 if the linked-list path ran. A forced CSC request still
 * reports 0 for `n == 0` because empty matrices use the linked-list no-op
 * path. "CSC selected" includes both the batched supernodal completion and
 * the resolved scalar-prepass fallback.
 */
typedef struct {
    sparse_reorder_t reorder;      /**< Fill-reducing reordering (NONE, RCM, AMD, or ND —
                                        ND is best on 2D / 3D PDE meshes, see
                                        sparse_reorder.h) */
    double tol;                    /**< Pivot tolerance for singularity detection and
                                        fill-in drop threshold. 0 or negative for the
                                        compile-time default (SPARSE_DROP_TOL).
                                        Stored in ldlt->tol and reused by solve, refine,
                                        and condest for consistency with factorization.
                                        Backend caveat: the linked-list backend threads
                                        `tol` into both the factorization drop / pivot
                                        checks and the recorded `ldlt->tol`, while the
                                        CSC backend currently enforces an internal
                                        `SPARSE_DROP_TOL` floor inside elimination and
                                        only propagates `max(tol, SPARSE_DROP_TOL)` into
                                        `ldlt->tol`.  A caller-supplied `tol > SPARSE_DROP_TOL`
                                        therefore tightens the solve-time singularity
                                        check under CSC but does not change the drops
                                        the CSC kernels apply during factorization. */
    sparse_ldlt_backend_t backend; /**< AUTO dispatches by size; LINKED_LIST / CSC force a path */
    int *used_csc_path;            /**< Optional output: 1 if the CSC pipeline was selected
                                        (including the structural fallback to the scalar
                                        pre-pass factor), 0 if the linked-list kernel ran. */
    /** Optional progress / cancellation callback. Invoked by the linked-list
     *  backend at each Bunch-Kaufman pivot with phase `"ldlt_factor"`,
     *  `step = k`, and `total = n`; k advances by 1 or 2 depending on pivot
     *  block size. Return 0 to continue. A non-zero return cancels the
     *  factorization, frees the partial `ldlt` output, and returns
     *  `SPARSE_ERR_CANCELLED`. LDL^T factorization does not modify the input
     *  matrix. NULL disables callbacks. The CSC backend currently emits no
     *  progress events. */
    sparse_progress_cb_t progress_cb;
    /** Opaque context pointer passed through unchanged to
     *  `progress_cb`.  Ignored when `progress_cb == NULL`. */
    void *progress_user;
} sparse_ldlt_opts_t;

/**
 * @brief Compute the LDL^T factorization of a symmetric matrix.
 *
 * Computes P*A*P^T = L*D*L^T using Bunch-Kaufman symmetric pivoting.
 * L is unit lower triangular (stored as a new SparseMatrix), D is
 * block-diagonal with 1x1 and 2x2 blocks.  The original matrix A is
 * not modified. The output object is reset on entry, so sparse_ldlt_free() is
 * safe after an error return.
 *
 * @note **Tolerance semantics:** The factorization computes and caches
 *       ||A||_inf in ldlt->factor_norm. Singularity detection uses
 *       relative thresholds based on the effective tolerance `tol`
 *       (caller-specified `opts->tol` when positive, otherwise
 *       `SPARSE_DROP_TOL`): a 1x1 pivot is rejected if
 *       |D[k]| < tol * ||A||_inf; a 2x2 pivot block with entries d11,
 *       d22, and d21 is rejected if its determinant is near zero
 *       relative to the local block scale
 *       (|d11| + |d22| + |d21|), rather than relative to ||A||_inf^2.
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
 * @param opts  Factorization options. NULL uses defaults: no reordering,
 *              default tolerance, AUTO backend, no telemetry, no callback.
 * @param ldlt  Output: LDL^T factors. Reset on entry and must be freed with
 *              sparse_ldlt_free() after success.
 * @return SPARSE_OK on success, or an error code (see sparse_ldlt_factor()).
 * @return SPARSE_ERR_CANCELLED if the linked-list progress callback cancels.
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
 * @return SPARSE_ERR_ALLOC if temporary solve workspace allocation fails.
 * @return SPARSE_ERR_SINGULAR if a zero D block is encountered during solve.
 *
 * @par Thread safety: Read-only on ldlt. Safe to call concurrently on the
 *               same factorization with different b/x vectors.
 */
sparse_err_t sparse_ldlt_solve(const sparse_ldlt_t *ldlt, const double *b, double *x);

/**
 * @brief Free the LDL^T factorization data.
 *
 * @param ldlt  The factorization to free. NULL and zeroed structs are safe.
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
 * @return SPARSE_ERR_BADARG if ldlt has not been factored.
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
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any pointer argument is NULL.
 * @return SPARSE_ERR_ALLOC if temporary workspace allocation fails.
 * @return SPARSE_ERR_BADARG or SPARSE_ERR_SINGULAR if the factorization solve
 *         fails during a refinement step.
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
 * @return SPARSE_ERR_ALLOC if temporary estimator workspace allocation fails.
 * @return SPARSE_ERR_BADARG or SPARSE_ERR_SINGULAR if the factorization solve
 *         fails during estimation.
 */
sparse_err_t sparse_ldlt_condest(const SparseMatrix *A, const sparse_ldlt_t *ldlt, double *condest);

#endif /* SPARSE_LDLT_H */
