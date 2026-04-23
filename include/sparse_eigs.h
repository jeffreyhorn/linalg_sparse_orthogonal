#ifndef SPARSE_EIGS_H
#define SPARSE_EIGS_H

/**
 * @file sparse_eigs.h
 * @brief Sparse symmetric eigensolvers (Sprint 20).
 *
 * Provides `sparse_eigs_sym()` for computing k extreme or near-sigma
 * eigenpairs of a symmetric sparse matrix A.  The default backend is
 * a thick-restart Lanczos iteration (Sprint 20 Days 8-11); shift-
 * invert mode for interior eigenvalues (Sprint 20 Day 12) reuses the
 * LDL^T factorization plumbing added in Days 4-6 via
 * `sparse_ldlt_factor_opts`.
 *
 * **Usage pattern:**
 * @code
 *   SparseMatrix *A = ...;  // symmetric matrix
 *   idx_t n = sparse_rows(A);
 *   idx_t k = 5;            // want 5 eigenpairs
 *
 *   double *vals = malloc((size_t)k * sizeof(double));
 *   double *vecs = malloc((size_t)(k * n) * sizeof(double));  // column-major
 *   sparse_eigs_t result = {
 *       .eigenvalues = vals,
 *       .eigenvectors = vecs,
 *   };
 *   sparse_eigs_opts_t opts = {
 *       .which = SPARSE_EIGS_LARGEST,
 *       .compute_vectors = 1,
 *       .tol = 1e-10,
 *   };
 *   sparse_err_t err = sparse_eigs_sym(A, k, &opts, &result);
 *   if (err == SPARSE_OK) {
 *       printf("%zd of %zd eigenpairs converged in %zd iterations\n",
 *              (ptrdiff_t)result.n_converged, (ptrdiff_t)result.n_requested,
 *              (ptrdiff_t)result.iterations);
 *       for (idx_t i = 0; i < result.n_converged; i++)
 *           printf("  lambda[%zd] = %.12e\n", (ptrdiff_t)i, vals[i]);
 *   }
 *   free(vals);
 *   free(vecs);
 * @endcode
 *
 * **Design notes.** The result struct uses caller-owned buffers for
 * the eigenvalue and eigenvector arrays — consistent with the
 * iterative-solver convention in `sparse_iterative.h`
 * (`residual_history` is caller-allocated).  The library writes
 * scalar output fields (`n_requested`, `n_converged`, `iterations`,
 * `residual_norm`) into `sparse_eigs_t` on return.  No library-side
 * allocation means no `sparse_eigs_free` helper is needed — caller
 * frees its own buffers.
 */

#include "sparse_matrix.h"
#include "sparse_types.h"

/**
 * @brief Selector for which portion of the spectrum to return.
 *
 * - `SPARSE_EIGS_LARGEST`: k algebraically largest eigenvalues.
 *   Lanczos converges fastest to extreme eigenvalues, so this (or
 *   SMALLEST) is the default sweet spot for symmetric problems.
 * - `SPARSE_EIGS_SMALLEST`: k algebraically smallest eigenvalues.
 *   Implemented as Lanczos on -A (so Lanczos still converges to
 *   the extreme values).
 * - `SPARSE_EIGS_NEAREST_SIGMA`: k eigenvalues closest (in absolute
 *   value of lambda − sigma) to the shift point `opts->sigma`.
 *   Implemented via shift-invert: Lanczos on (A − sigma·I)^{-1},
 *   whose eigenvalues are 1/(lambda − sigma).  Requires the LDL^T
 *   factorization of (A − sigma·I) — returns `SPARSE_ERR_SINGULAR`
 *   if sigma coincides with (or is numerically too close to) an
 *   eigenvalue of A, in which case the caller should perturb sigma
 *   slightly and retry.
 */
typedef enum {
    SPARSE_EIGS_LARGEST = 0,
    SPARSE_EIGS_SMALLEST = 1,
    SPARSE_EIGS_NEAREST_SIGMA = 2,
} sparse_eigs_which_t;

/**
 * @brief Eigensolver backend selector.
 *
 * - `SPARSE_EIGS_BACKEND_AUTO` (default, zero-initialised): let the
 *   library pick.  Currently always routes to Lanczos; reserved for
 *   future LOBPCG dispatch (Sprint 21 Item 1) when the preconditioned
 *   block path is preferable for specific which/k combinations.
 * - `SPARSE_EIGS_BACKEND_LANCZOS`: thick-restart Lanczos with
 *   optional full reorthogonalization.  The Sprint 20 workhorse.
 */
typedef enum {
    SPARSE_EIGS_BACKEND_AUTO = 0,
    SPARSE_EIGS_BACKEND_LANCZOS = 1,
    /* Sprint 21: SPARSE_EIGS_BACKEND_LOBPCG = 2, */
} sparse_eigs_backend_t;

/**
 * @brief Options for `sparse_eigs_sym()`.
 *
 * Pass NULL to `sparse_eigs_sym()` to use defaults:
 * `which = LARGEST`, `sigma = 0.0`, `max_iterations = 0` (library
 * default), `tol = 0.0` (library default 1e-10), `reorthogonalize =
 * 1`, `compute_vectors = 0`, `backend = AUTO`.
 */
typedef struct {
    /** Which portion of the spectrum to return. */
    sparse_eigs_which_t which;
    /** Shift point for `SPARSE_EIGS_NEAREST_SIGMA`; ignored
     *  otherwise.  Default: 0.0. */
    double sigma;
    /** Maximum outer iterations across thick-restarts.  0 selects
     *  the library default (currently `max(10 * k + 20, 100)`). */
    idx_t max_iterations;
    /** Convergence tolerance on the relative Ritz residual
     *  `||A·v - θ·v|| / (|θ| * ||A||_inf)`.  0 selects the library
     *  default `1e-10`.  Negative values are rejected with
     *  SPARSE_ERR_BADARG. */
    double tol;
    /** Full-reorthogonalization flag.  Nonzero (default) reorthogonalizes
     *  each new Lanczos vector against every prior Lanczos vector,
     *  maintaining `V^T V ≈ I` under finite precision.  Zero disables
     *  reorth (faster per iteration but loses orthogonality on
     *  wide-spectrum matrices — "ghost" eigenvalues may appear;
     *  mainly useful for cheap smoke tests). */
    int reorthogonalize;
    /** Nonzero to also compute eigenvectors; zero (default) returns
     *  eigenvalues only.  When nonzero, `result->eigenvectors` must
     *  be a caller-allocated buffer of length at least `n * k`
     *  (column-major). */
    int compute_vectors;
    /** Backend selector — see `sparse_eigs_backend_t`.  Default
     *  AUTO routes to Lanczos. */
    sparse_eigs_backend_t backend;
} sparse_eigs_opts_t;

/**
 * @brief Result from a call to `sparse_eigs_sym()`.
 *
 * The `eigenvalues` and `eigenvectors` fields are caller-owned
 * buffers that the caller must allocate (to length `>= k` and
 * `>= n * k` respectively) before the call and free after.  The
 * library writes at most `k` entries into each.  The remaining
 * fields are library-written outputs populated on return.
 *
 * **Partial convergence.** When Lanczos hits `max_iterations`
 * without converging all `k` requested pairs, the call returns
 * `SPARSE_ERR_NOT_CONVERGED` and `n_converged` (< k) pairs are
 * filled in.  Unfilled slots retain their pre-call values.
 * Callers that want partial results in the unconverged case
 * should inspect `n_converged` before consuming `eigenvalues[]`.
 */
typedef struct {
    /** Caller-owned buffer of length `>= k`.  The library writes
     *  converged eigenvalues into indices [0, n_converged),
     *  ordered as `which` selects (LARGEST → descending,
     *  SMALLEST → ascending, NEAREST_SIGMA → ascending |lambda − sigma|). */
    double *eigenvalues;
    /** Caller-owned buffer of length `>= n * k` (column-major).
     *  Ignored when `opts->compute_vectors == 0`; must be non-NULL
     *  when compute_vectors is set.  The library writes normalized
     *  eigenvectors into columns [0, n_converged); each column
     *  corresponds to the eigenvalue at the same index. */
    double *eigenvectors;
    /** Output: the `k` passed to `sparse_eigs_sym()`.  Written on
     *  return so the result struct is self-describing. */
    idx_t n_requested;
    /** Output: number of Ritz pairs that met the convergence
     *  tolerance within `max_iterations`.  0 <= n_converged <= k. */
    idx_t n_converged;
    /** Output: total Lanczos iterations across all restarts. */
    idx_t iterations;
    /** Output: maximum relative Ritz residual across the converged
     *  pairs.  Always <= `opts->tol` when the call returns
     *  SPARSE_OK. */
    double residual_norm;
} sparse_eigs_t;

/**
 * @brief Compute k extreme or near-sigma eigenpairs of a symmetric matrix.
 *
 * Uses thick-restart Lanczos (default) with optional full
 * reorthogonalization; shift-invert mode for interior eigenvalues
 * (`opts->which == SPARSE_EIGS_NEAREST_SIGMA`) factors `A - sigma*I`
 * via `sparse_ldlt_factor_opts` and applies its inverse at every
 * Lanczos step.
 *
 * @pre A must be symmetric — `sparse_is_symmetric(A, 1e-12)` is
 *      checked at entry.  A must be square.
 * @pre `1 <= k <= sparse_rows(A)`.
 * @pre `result != NULL`; `result->eigenvalues` must be a
 *      caller-allocated buffer of length `>= k`.
 * @pre When `opts->compute_vectors != 0`, `result->eigenvectors`
 *      must be a caller-allocated buffer of length `>= n * k`
 *      (column-major).
 * @pre When `opts->which == SPARSE_EIGS_NEAREST_SIGMA`, `sigma`
 *      must not exactly coincide with an eigenvalue of A
 *      (otherwise `A - sigma*I` is singular and the shift-invert
 *      factorization fails).  Perturb sigma slightly if this
 *      trips.
 *
 * @param A       Symmetric sparse matrix (not modified).  Must be
 *                square.
 * @param k       Number of eigenpairs to compute (1 <= k <= n).
 * @param opts    Options (NULL for defaults; see `sparse_eigs_opts_t`).
 * @param result  Output: caller-owned eigenvalue / eigenvector
 *                buffers filled in place, plus library-written
 *                scalar outputs.  Must be non-NULL.
 * @return SPARSE_OK if all k pairs converged within tolerance.
 * @return SPARSE_ERR_NOT_CONVERGED if max_iterations was reached
 *         with fewer than k pairs converged.  Partial results in
 *         `result->eigenvalues[0..n_converged)` / `eigenvectors`
 *         are still valid.
 * @return SPARSE_ERR_NULL if A, opts->callback, or required result
 *         buffers are NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_NOT_SPD if A fails the symmetry check
 *         (reused for "not symmetric" per existing convention; see
 *         `sparse_ldlt_factor`).
 * @return SPARSE_ERR_BADARG if `k` is out of range or `opts` values
 *         are invalid (negative tol or max_iterations, unknown
 *         `which` / `backend`).
 * @return SPARSE_ERR_SINGULAR when shift-invert mode factors
 *         `A - sigma*I` and that matrix is (near-)singular.
 * @return SPARSE_ERR_ALLOC if Lanczos workspace allocation fails.
 *
 * @par Thread safety: read-only on A.  Safe to call concurrently
 *              on the same matrix with different result buffers.
 *              Shift-invert mode internally calls
 *              `sparse_ldlt_factor_opts`, which is also read-only
 *              on A.
 */
sparse_err_t sparse_eigs_sym(const SparseMatrix *A, idx_t k, const sparse_eigs_opts_t *opts,
                             sparse_eigs_t *result);

#endif /* SPARSE_EIGS_H */
