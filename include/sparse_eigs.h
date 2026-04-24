#ifndef SPARSE_EIGS_H
#define SPARSE_EIGS_H

/**
 * @file sparse_eigs.h
 * @brief Sparse symmetric eigensolvers (Sprint 20).
 *
 * Provides `sparse_eigs_sym()` for computing k extreme or near-sigma
 * eigenpairs of a symmetric sparse matrix A via Lanczos with full
 * MGS reorthogonalisation and a growing-subspace outer loop (a true
 * Wu/Simon thick-restart backend is planned for Sprint 21).  Three
 * spectrum-selection modes (`SPARSE_EIGS_LARGEST`, `_SMALLEST`,
 * `_NEAREST_SIGMA`); interior eigenvalues are found via shift-invert
 * Lanczos, which composes with the LDL^T dispatch in
 * `sparse_ldlt.h` (see `sparse_ldlt_factor_opts` — the Sprint 20
 * Day 4-6 AUTO / LINKED_LIST / CSC backend selector routes through
 * the supernodal path on `n >= SPARSE_CSC_THRESHOLD`).  The SVD in
 * `sparse_svd.h` is a cousin API: singular values of a rectangular
 * A are related to eigenvalues of A^T·A (cross-checked in the
 * Sprint 20 Day 13 tests).
 *
 * **Usage pattern:**
 * @code
 *   SparseMatrix *A = ...;  // symmetric matrix
 *   idx_t n = sparse_rows(A);
 *   idx_t k = 5;            // want 5 eigenpairs
 *
 *   double *vals = malloc((size_t)k * sizeof(double));
 *   double *vecs = malloc((size_t)k * (size_t)n * sizeof(double));  // column-major
 *   sparse_eigs_t result = {
 *       .eigenvalues = vals,
 *       .eigenvectors = vecs,
 *   };
 *   sparse_eigs_opts_t opts = {
 *       .which = SPARSE_EIGS_LARGEST,
 *       .compute_vectors = 1,
 *       .tol = 1e-10,
 *       .reorthogonalize = 1,  // explicit — designated init zeros
 *                              // unset fields, so set this to 1 for
 *                              // reorth (the library default when
 *                              // opts==NULL).
 *   };
 *   sparse_err_t err = sparse_eigs_sym(A, k, &opts, &result);
 *   if (err == SPARSE_OK) {
 *       printf("%td of %td eigenpairs converged in %td Lanczos iterations\n",
 *              (ptrdiff_t)result.n_converged, (ptrdiff_t)result.n_requested,
 *              (ptrdiff_t)result.iterations);
 *       for (idx_t i = 0; i < result.n_converged; i++)
 *           printf("  lambda[%td] = %.12e\n", (ptrdiff_t)i, vals[i]);
 *   }
 *   free(vals);
 *   free(vecs);
 * @endcode
 *
 * **Convergence.** `sparse_eigs_sym` runs a single growing-m
 * Lanczos sequence starting from a deterministic pseudo-random v0
 * (golden-ratio fractional mixing — reproducible across runs and
 * avoids eigenvector alignment on diagonal fixtures).  The
 * per-retry grow-m strategy strictly extends the Krylov basis, so
 * every pass benefits from prior work.  Convergence is gated on
 * the Wu/Simon residual `|beta_m * y_{m-1, j}| / |theta_j|` of
 * every selected Ritz pair and is reported in
 * `result.residual_norm`.  The residual bounds the eigen-equation
 * relative error of whatever operator Lanczos is running on:
 *   - `LARGEST` / `SMALLEST`: Lanczos runs on `A`, so the bound
 *     applies to `||A v - λ v|| / (|λ| * ||v||)` directly.
 *   - `NEAREST_SIGMA`: Lanczos runs on `(A - sigma·I)^{-1}`, so
 *     `theta_j` is an eigenvalue of the inverse operator and the
 *     reported residual bounds `||(A - sigma·I)^{-1} v - theta v||`
 *     rather than the original-A residual.  The post-processed
 *     eigenvalue `lambda = sigma + 1/theta` is still accurate to
 *     the same per-step tolerance, but callers who want an
 *     original-A residual must recompute `||A v - lambda v||`
 *     themselves from `result.eigenvectors`.
 *
 * **Design notes.** The result struct uses caller-owned buffers for
 * the eigenvalue and eigenvector arrays — consistent with the
 * iterative-solver convention in `sparse_iterative.h`
 * (`residual_history` is caller-allocated).  The library writes
 * scalar output fields (`n_requested`, `n_converged`, `iterations`,
 * `residual_norm`, `used_csc_path_ldlt`) into `sparse_eigs_t` on
 * return.  No library-side allocation means no `sparse_eigs_free`
 * helper is needed — caller frees its own buffers.
 *
 * @see sparse_ldlt.h — factorisation backend used by shift-invert.
 * @see sparse_svd.h — related decomposition for rectangular A.
 * @see docs/algorithm.md — Lanczos theory and implementation notes.
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
 *   Implemented by running Lanczos on `A` and selecting the
 *   algebraically smallest Ritz values from the computed pairs.
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
 * - `SPARSE_EIGS_BACKEND_LANCZOS`: Lanczos with a growing-subspace
 *   outer loop and optional full reorthogonalization.  The Sprint 20
 *   workhorse; true Wu/Simon thick-restart is a Sprint 21 extension.
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
    /** Cap on the Lanczos subspace size (`m`) per Lanczos run.
     *  Also caps the growth across grow-m retries (`m` never
     *  exceeds this value on any run), but does not bound the
     *  cumulative Lanczos iterations across retries — that is
     *  reported as `result->iterations`.  0 selects the library
     *  default (currently `max(10 * k + 20, 100)`). */
    idx_t max_iterations;
    /** Convergence tolerance on the Wu/Simon Ritz-residual bound
     *  `|beta_m * y_{m-1,j}| / max(|theta_j|, scale)` used by
     *  `sparse_eigs_sym()` for each selected Ritz pair (`scale` is
     *  a floor tied to the Lanczos T-norm, so near-zero Ritz values
     *  still get a meaningful relative threshold).  This is the
     *  same residual scaling reported in `result->residual_norm`;
     *  it does not include an explicit `||A||_inf` factor.  0
     *  selects the library default `1e-10`.  Negative values are
     *  rejected with SPARSE_ERR_BADARG. */
    double tol;
    /** Full-reorthogonalization flag.  Nonzero reorthogonalizes each
     *  new Lanczos vector against every prior Lanczos vector,
     *  maintaining `V^T V ≈ I` under finite precision.  Zero
     *  disables reorth (faster per iteration but loses orthogonality
     *  on wide-spectrum matrices — "ghost" eigenvalues may appear;
     *  mainly useful for cheap smoke tests).  The library default is
     *  ON — pass `opts == NULL` to get it, or set this field to 1
     *  explicitly when using designated initialisers (which zero
     *  unset fields).
     *
     *  **Reliability caveat (important):** both the Wu/Simon
     *  convergence gate reported in `result->residual_norm` and the
     *  lifted eigenvectors written to `result->eigenvectors`
     *  (`V·y`) rely on `V` being orthonormal.  When this flag is 0
     *  the Krylov basis drifts from orthonormal, so the residual
     *  bound can be silently optimistic (false convergence on
     *  ghost Ritz pairs) and lifted eigenvectors are not unit norm
     *  nor truly A-eigenvectors.  Callers that want reliable
     *  `result->residual_norm` or `compute_vectors = 1` output
     *  MUST set this field to 1. */
    int reorthogonalize;
    /** Nonzero to also compute eigenvectors; zero (default) returns
     *  eigenvalues only.  When nonzero, `result->eigenvectors` must
     *  be a caller-allocated buffer of length at least `n * k`
     *  (column-major).  See the `reorthogonalize` field above for
     *  the reliability requirement — `compute_vectors = 1` is only
     *  meaningful when `reorthogonalize = 1`. */
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
    /** Output: for shift-invert mode (`which == NEAREST_SIGMA`),
     *  set to 1 when the internal `sparse_ldlt_factor_opts` call
     *  selected the CSC supernodal backend and 0 when it routed to
     *  the linked-list path.  Mirrors the Day 4-6 `used_csc_path`
     *  telemetry on `sparse_ldlt_opts_t`.  Always 0 for
     *  LARGEST / SMALLEST (no LDL^T factor involved).  Sprint 20
     *  Day 13 observability. */
    int used_csc_path_ldlt;
} sparse_eigs_t;

/**
 * @brief Compute k extreme or near-sigma eigenpairs of a symmetric matrix.
 *
 * Uses Lanczos (default) with a growing-subspace outer loop and
 * optional full reorthogonalization; shift-invert mode for interior
 * eigenvalues (`opts->which == SPARSE_EIGS_NEAREST_SIGMA`) factors
 * `A - sigma*I` via `sparse_ldlt_factor_opts` and applies its
 * inverse at every Lanczos step.  A true Wu/Simon thick-restart
 * backend is planned for Sprint 21.
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
 * @return SPARSE_ERR_NULL if A, result, or required result buffers
 *         (`result->eigenvalues`, and `result->eigenvectors` when
 *         eigenvectors are requested) are NULL.
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
