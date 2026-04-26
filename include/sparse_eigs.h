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
 * `residual_norm`, `used_csc_path_ldlt`, `peak_basis_size`,
 * `backend_used`) into `sparse_eigs_t` on return.  No library-side
 * allocation means no `sparse_eigs_free` helper is needed — caller
 * frees its own buffers.
 *
 * @see sparse_ldlt.h — factorisation backend used by shift-invert.
 * @see sparse_svd.h — related decomposition for rectangular A.
 * @see docs/algorithm.md — Lanczos theory and implementation notes.
 */

#include "sparse_iterative.h" /* sparse_precond_fn for LOBPCG (Sprint 21 Day 7) */
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
 *   library pick.  Sprint 21 Day 10 routing decision tree:
 *
 *     if (opts->precond != NULL && n >= SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD
 *         && (opts->block_size == 0 ? k >= 4 : opts->block_size >= 4)):
 *         → SPARSE_EIGS_BACKEND_LOBPCG
 *     else if (n >= SPARSE_EIGS_THICK_RESTART_THRESHOLD):
 *         → SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART
 *     else:
 *         → SPARSE_EIGS_BACKEND_LANCZOS  (grow-m)
 *
 *   The thresholds are tuned on the Sprint 21 bench corpus
 *   (`docs/planning/EPIC_2/SPRINT_21/bench_day*.txt`); tune
 *   further in future sprints when the workload shifts by
 *   overriding `SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD` /
 *   `SPARSE_EIGS_THICK_RESTART_THRESHOLD` at compile time.
 *   `result->backend_used` records AUTO's choice on every call.
 * - `SPARSE_EIGS_BACKEND_LANCZOS`: Lanczos with a growing-subspace
 *   outer loop and optional full reorthogonalization.  The Sprint 20
 *   workhorse.  Peak memory `O(m_cap · n)` across retries.
 * - `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`: True Wu/Simon
 *   thick-restart Lanczos (Sprint 21 Day 3).  Preserves the
 *   converged Ritz subspace in a compact arrowhead basis between
 *   restart phases; peak memory `O((k + m_restart) · n)`
 *   regardless of total iteration count.  Use this for large-n
 *   problems where the grow-m path would blow memory holding V.
 * - `SPARSE_EIGS_BACKEND_LOBPCG` (Sprint 21 Days 7-10):
 *   Knyazev's Locally Optimal Block Preconditioned Conjugate
 *   Gradient (Knyazev 2001).  Iterates a block of `block_size`
 *   approximate eigenvectors X by block Rayleigh-Ritz on the
 *   subspace `[X, W, P]` where W is the (preconditioned) residual
 *   and P is the previous step's search direction.  Plugs the
 *   Sprint 13 IC(0) / LDL^T preconditioners in via `opts->precond`,
 *   composing with the rest of the eigensolver pipeline through the
 *   same `sparse_precond_fn` callback the iterative solvers use.
 *   Best for ill-conditioned SPD problems where a cheap
 *   preconditioner is available; vanilla (`precond == NULL`) LOBPCG
 *   is correct but converges slower than Lanczos on the well-
 *   conditioned corpus — which is why AUTO routes to LOBPCG only
 *   when a preconditioner is actually supplied.  All three `which`
 *   modes (LARGEST / SMALLEST / NEAREST_SIGMA) are supported;
 *   NEAREST_SIGMA composes with the same shift-invert LDL^T
 *   pipeline the Lanczos backends use.
 */
typedef enum {
    SPARSE_EIGS_BACKEND_AUTO = 0,
    SPARSE_EIGS_BACKEND_LANCZOS = 1,
    SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART = 2,
    SPARSE_EIGS_BACKEND_LOBPCG = 3,
} sparse_eigs_backend_t;

/**
 * @brief Crossover threshold for AUTO backend dispatch.
 *
 * When `opts->backend == SPARSE_EIGS_BACKEND_AUTO` and
 * `sparse_rows(A) >= SPARSE_EIGS_THICK_RESTART_THRESHOLD`, the
 * library routes to the bounded-memory thick-restart backend
 * rather than the Sprint 20 grow-m path.  Below the threshold
 * the grow-m path wins because its full-basis Ritz extraction
 * converges in slightly fewer matvecs on small problems and
 * memory isn't a concern — bcsstk04 (n = 132) grow-m holds
 * ~160 KB of V at m_cap = 130, which is cheap on modern
 * machines.  Above the threshold the memory bound matters —
 * bcsstk14 (n = 1806) grow-m at m_cap = 500 holds ~7 MB of V,
 * which grows to ~26 MB if max_iterations = n.  Thick-restart
 * caps peak V at `m_restart + k_locked ≈ 35` columns regardless
 * of total iteration count.
 *
 * Provisional value: 500 (matches the nos4 / bcsstk04 / kkt-150
 * / bcsstk14 measured crossover in the Sprint 21 Day 4 benchmark
 * capture at `docs/planning/EPIC_2/SPRINT_21/bench_day4_restart.txt`).
 * Override at compile time with `-DSPARSE_EIGS_THICK_RESTART_THRESHOLD=N`
 * when profiling on a different corpus.
 */
#ifndef SPARSE_EIGS_THICK_RESTART_THRESHOLD
#define SPARSE_EIGS_THICK_RESTART_THRESHOLD 500
#endif

/**
 * @brief AUTO routing crossover threshold for LOBPCG (Sprint 21 Day 10).
 *
 * When `opts->backend == SPARSE_EIGS_BACKEND_AUTO`,
 * `opts->precond != NULL`, and `sparse_rows(A) >=
 * SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD`, the library routes to
 * LOBPCG.  Below the threshold (or when no preconditioner is
 * supplied) AUTO continues to choose between Lanczos backends per
 * `SPARSE_EIGS_THICK_RESTART_THRESHOLD`.
 *
 * Rationale: LOBPCG's per-iteration cost is `O(block_size · matvec
 * + block_size² · Jacobi)`, so it amortises well only when the
 * preconditioner makes the iteration count tiny relative to
 * Lanczos's per-Ritz-pair work.  Without a preconditioner LOBPCG
 * generally underperforms thick-restart Lanczos on the same n; the
 * AUTO path therefore declines to pick LOBPCG when `precond ==
 * NULL`.
 *
 * Provisional value: 1000 (matches the n-thresholds in PROJECT_PLAN
 * Sprint 21 PLAN.md Day 10 task 3).  Override at compile time with
 * `-DSPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD=N` when profiling on a
 * different corpus.
 */
#ifndef SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD
#define SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD 1000
#endif

/**
 * @brief Options for `sparse_eigs_sym()`.
 *
 * Pass NULL to `sparse_eigs_sym()` to use defaults:
 * `which = LARGEST`, `sigma = 0.0`, `max_iterations = 0` (library
 * default), `tol = 0.0` (library default 1e-10), `reorthogonalize =
 * 1`, `compute_vectors = 0`, `backend = AUTO`, `block_size = 0`
 * (library default `block_size = k` for LOBPCG; ignored for
 * Lanczos), `precond = NULL` / `precond_ctx = NULL` (vanilla
 * LOBPCG; ignored for Lanczos), `lobpcg_soft_lock = 1` (per-column
 * freezing on; ignored for Lanczos).
 *
 * @warning **ABI break in v2.2.0.**  Sprint 21 Days 7-9 added the
 * `block_size`, `precond`, `precond_ctx`, and `lobpcg_soft_lock`
 * fields at the end of this struct, changing its size relative to
 * the v2.1.x version shipped through Sprint 20.  Source-level
 * compatibility is preserved: positional and designated initialisers
 * from v2.1.x continue to compile — the new trailing fields zero-
 * init to the LOBPCG library defaults.  Pre-compiled downstream
 * binaries linked against v2.1.x must be recompiled against v2.2.x
 * because stack-allocating the old struct would cause the new
 * library to read past its end.
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
    /** LOBPCG block size — number of approximate eigenvector columns
     *  iterated together (Sprint 21 Day 7).  Ignored unless
     *  `backend == SPARSE_EIGS_BACKEND_LOBPCG` (or AUTO routes
     *  there).  0 (the designated-init default) selects the library
     *  default `block_size = k`, which is the minimum that produces
     *  k Ritz pairs per Rayleigh-Ritz step.  Larger blocks accelerate
     *  convergence on clustered spectra at the cost of more memory
     *  (peak `O((3 · block_size) · n)`).  Must satisfy
     *  `0 <= block_size` and, when nonzero, `k <= block_size <= n`.
     *  Values of `block_size < k` are rejected with
     *  SPARSE_ERR_BADARG.
     *
     *  Backwards compatibility: this field is trailing in the
     *  struct, so designated-initialiser callers from before
     *  Sprint 21 still compile and get the library default. */
    idx_t block_size;
    /** LOBPCG preconditioner callback (Sprint 21 Days 7-9).  When
     *  non-NULL, applied to each column of the residual block W in
     *  every LOBPCG iteration: `precond(precond_ctx, n, R[:, j],
     *  W[:, j])`.  NULL selects vanilla (unpreconditioned) LOBPCG —
     *  correct but typically slower convergence on ill-conditioned
     *  problems.  See `sparse_iterative.h` for the typedef and
     *  `sparse_ic.h` / `sparse_ldlt.h` for ready-made preconditioner
     *  builders.
     *
     *  Ignored when `backend != SPARSE_EIGS_BACKEND_LOBPCG`. */
    sparse_precond_fn precond;
    /** Opaque context pointer passed through unchanged to the
     *  `precond` callback.  Typically a pointer to a factored
     *  preconditioner struct (e.g. `sparse_ilu_t *` for IC(0),
     *  `sparse_ldlt_t *` for LDL^T).  When `precond == NULL` this
     *  field is ignored — but `precond_ctx != NULL` while
     *  `precond == NULL` is rejected as SPARSE_ERR_BADARG (the
     *  obvious user error of forgetting to set the callback). */
    const void *precond_ctx;
    /** LOBPCG soft-locking flag (Sprint 21 Day 9).  Nonzero enables
     *  per-column convergence freezing: once a Ritz pair's residual
     *  drops below `tol`, the corresponding W (preconditioned
     *  residual) and P (search direction) columns are zeroed for
     *  subsequent iterations.  The orthonormalisation step then
     *  ejects those zero columns, shrinking the effective
     *  Rayleigh-Ritz subspace and saving work on the remaining
     *  unconverged columns.  Locked columns themselves stay in X
     *  (their Ritz pair is preserved by the RR step because X is
     *  in the basis).
     *
     *  Library default is ON when `opts == NULL`.  Designated
     *  initialisers leave this field at 0 (off); set it to 1
     *  explicitly to enable.  Off is correct but slower on
     *  problems where the spectrum has a wide gap between the
     *  bottom-k and the rest — typical of the ill-conditioned
     *  fixtures the Day 9 preconditioning targets.
     *
     *  Ignored when `backend != SPARSE_EIGS_BACKEND_LOBPCG`. */
    int lobpcg_soft_lock;
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
 *
 * @warning **ABI break in v2.2.0.**  Sprint 21 Days 4 and 7 added
 * the `peak_basis_size` and `backend_used` fields at the end of
 * this struct, changing its size relative to the v2.1.x version
 * shipped through Sprint 20.  Source-level compatibility is
 * preserved: positional and designated initialisers from v2.1.x
 * continue to compile.  Pre-compiled downstream binaries linked
 * against v2.1.x must be recompiled against v2.2.x because stack-
 * allocating the old struct would cause the new library to write
 * past its end.
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
    /** Output: peak Lanczos basis size (number of length-n columns
     *  held simultaneously in the dominant allocation) observed
     *  during the run.  Sprint 21 Day 4 telemetry; lets callers
     *  compare the grow-m path's monotonically-growing `m_cap` to
     *  the thick-restart path's bounded `m_restart + k_locked_cap`
     *  peak to verify the memory-savings claim on large-n
     *  problems.  For the grow-m backend this equals `m_cap` (the
     *  largest V actually allocated across retries); for
     *  thick-restart it equals `m_restart + k_locked_cap_used`
     *  (the simultaneously-live `V` basis plus the restart
     *  state's `V_locked` block).  Reported in doubles-times-`n`
     *  units — multiply by `n * sizeof(double)` to get bytes. */
    idx_t peak_basis_size;
    /** Output: which backend the library actually dispatched to.
     *  Mirrors `sparse_ldlt_t::used_csc_path` in spirit — when
     *  `opts->backend == SPARSE_EIGS_BACKEND_AUTO`, AUTO picks one
     *  of the concrete backends per the size threshold (currently
     *  LANCZOS below `SPARSE_EIGS_THICK_RESTART_THRESHOLD` and
     *  LANCZOS_THICK_RESTART above; Sprint 21 Day 10 extends this
     *  to also route LOBPCG when a preconditioner is supplied).
     *  Set on every successful return.  On error returns, treat
     *  this field as unspecified / best-effort telemetry: backend
     *  selection may already have been recorded before a later
     *  failure is detected.  Sprint 21 Day 7 observability — used
     *  by the Day 11 `bench_eigs` driver to log AUTO's choice per
     *  fixture in the CSV output. */
    sparse_eigs_backend_t backend_used;
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
