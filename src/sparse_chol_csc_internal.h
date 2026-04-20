#ifndef SPARSE_CHOL_CSC_INTERNAL_H
#define SPARSE_CHOL_CSC_INTERNAL_H

/**
 * @file sparse_chol_csc_internal.h
 * @brief CSC working format for Cholesky (and LDL^T) numeric factorization.
 *
 * Not part of the public API.  Used by sparse_chol_csc.c (and, in later
 * days of Sprint 17, sparse_ldlt_csc.c).
 *
 * ─── Design: why CSC for Cholesky? ───────────────────────────────────────
 *
 * The CSR working format introduced in Sprint 10 accelerated LU elimination
 * by replacing linked-list pointer chasing with contiguous row arrays.  LU's
 * inner loop is row-oriented (`row[k] -= (row[k][j] / pivot) * row[pivot]`),
 * which maps naturally onto CSR.
 *
 * Cholesky's inner loop is column-oriented.  Both the left-looking and
 * up-looking formulations operate column-by-column: for column j,
 *
 *     cmod(j, k):  L[*,j] -= L[j,k] * L[*,k]    for each k < j with L[j,k] ≠ 0
 *     cdiv(j):    L[j,j]  = sqrt(L[j,j])
 *                 L[i,j] /= L[j,j]              for each i > j with L[i,j] ≠ 0
 *
 * CSC stores columns contiguously, so both cmod (reading column k) and cdiv
 * (scaling column j in-place) are sequential sweeps.  The same layout also
 * serves the triangular solves: forward `L*y = b` is a left-to-right column
 * sweep, and backward `L^T*x = y` is a right-to-left column sweep over the
 * same columns (no transpose needed).
 *
 * The linked-list `SparseMatrix` remains the mutable input format; the
 * pipeline is: linked-list → CSC → eliminate → CSC → linked-list.
 *
 * ─── Storage invariants ──────────────────────────────────────────────────
 *
 * - `col_ptr[0] == 0` and `col_ptr[n] == nnz`.
 * - For each column j, the entries at positions `col_ptr[j] .. col_ptr[j+1]-1`
 *   are sorted ascending by `row_idx`.
 * - Cholesky stores the lower triangle *including the diagonal*.  For column
 *   j, the first stored entry (smallest row index) is the diagonal
 *   `L[j,j]`; all subsequent entries have row_idx > j.
 * - All stored row indices in column j satisfy `row_idx >= j` (strict
 *   lower triangular plus diagonal only).
 *
 * The factor storage convention for Cholesky / LDL^T uses an explicit
 * diagonal entry in each non-empty, factor-ready column — the first
 * entry (smallest row index) is `L[j,j]`, and all subsequent entries
 * have row_idx > j.  This is the shape `chol_csc_eliminate` produces
 * and that every solve path consumes.
 *
 * `chol_csc_validate` enforces exactly the same diagonal-first rule
 * for every non-empty column (structurally zero / empty columns are
 * the only permitted exception — all-zero off-diagonals are valid
 * A-conversion input, and their missing diagonal is diagnosed later
 * by `chol_csc_cdiv` returning `SPARSE_ERR_NOT_SPD`).  In other
 * words: "diagonal-first" is an invariant of every validated
 * non-empty `CholCsc` column, not merely of the final factor-ready
 * layout.
 */

#include "sparse_analysis.h"
#include "sparse_matrix_internal.h"
#include <stdint.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * CholCsc: CSC working format for Cholesky / LDL^T numeric factorization
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * CSC working format for Cholesky elimination.
 *
 * Columns are stored contiguously in `row_idx` / `values`, indexed by
 * `col_ptr`.  The storage carries extra capacity (`capacity >= nnz`) so that
 * fill-in produced during elimination can be absorbed without reallocation
 * for predictable-structure matrices, and with geometric growth otherwise.
 */
typedef struct {
    idx_t n;            /**< Matrix dimension (square n×n). */
    idx_t nnz;          /**< Current number of stored nonzeros. */
    idx_t capacity;     /**< Allocated length of row_idx / values. */
    idx_t *col_ptr;     /**< Column pointers (length n+1). col_ptr[j]..col_ptr[j+1]-1
                             index into row_idx / values for column j. */
    idx_t *row_idx;     /**< Row indices (length capacity), sorted ascending per column. */
    double *values;     /**< Nonzero values (length capacity). */
    double factor_norm; /**< ||A||_inf at conversion time, for relative tolerance. */

    /* Sprint 19 Day 7: set to 1 when `chol_csc_from_sparse_with_analysis`
     * pre-populated the full sym_L pattern.  `chol_csc_gather`'s fast
     * path (Day 6) reads this to skip the O(pattern_count) merge-walk
     * check that confirms every survivor row is in the slot — the
     * sym_L pre-population is itself the proof.  Set to 0 by
     * `chol_csc_from_sparse` (heuristic) so its callers fall back to
     * the merge-walk + slow-path as needed. */
    int sym_L_preallocated;
} CholCsc;

/* ═══════════════════════════════════════════════════════════════════════
 * Lifecycle helpers: alloc / free / grow
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Allocate a CholCsc with the given dimension and initial capacity.
 *
 * Allocates the struct and its three arrays (`col_ptr`, `row_idx`, `values`).
 * On success, `*out` points to a zero-initialized CholCsc with
 *   - `n` set
 *   - `nnz == 0`
 *   - `capacity == initial_nnz` (clamped to at least 1)
 *   - `col_ptr[0..n]` zeroed (so the CSC represents an empty matrix)
 *   - `factor_norm == 0.0` (caller fills this in during conversion)
 *
 * @param n            Matrix dimension (must be >= 0).
 * @param initial_nnz  Initial capacity for row_idx / values (clamped to >= 1).
 * @param[out] out     Receives the allocated CholCsc*.  Set to NULL on error.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if out is NULL.
 * @return SPARSE_ERR_BADARG if n < 0.
 * @return SPARSE_ERR_ALLOC if any allocation fails.
 */
sparse_err_t chol_csc_alloc(idx_t n, idx_t initial_nnz, CholCsc **out);

/**
 * Free a CholCsc and all its arrays.  Safe to call with NULL.
 */
void chol_csc_free(CholCsc *m);

/**
 * Ensure the CholCsc has capacity for at least `needed` nonzeros.
 *
 * If `needed <= m->capacity`, this is a no-op.  Otherwise, `row_idx` and
 * `values` are reallocated to at least `max(needed, 2 * m->capacity)` to
 * keep amortized growth cost linear.  `col_ptr` is not touched.
 *
 * @param m       Target CholCsc (not NULL).
 * @param needed  Minimum capacity required.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if m is NULL.
 * @return SPARSE_ERR_ALLOC if reallocation fails (m is unchanged on failure).
 */
sparse_err_t chol_csc_grow(CholCsc *m, idx_t needed);

/* ═══════════════════════════════════════════════════════════════════════
 * Conversion: linked-list SparseMatrix ↔ CSC working format
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Convert a linked-list SparseMatrix into a CSC working format for Cholesky.
 *
 * Reads `mat` in logical index order (applying the matrix's own
 * inv_row_perm / inv_col_perm) and extracts the lower triangle (including
 * the diagonal).  An optional symmetric fill-reducing permutation `perm`
 * (where `perm[new] = old`, typically from AMD or RCM) is applied on top
 * of the matrix's internal permutations: the CSC stores entries in the
 * NEW (permuted) coordinate system.  Only entries with `new_row >= new_col`
 * are kept.
 *
 * The CSC row indices within each column are sorted ascending, with the
 * diagonal first in each column — the invariant expected by the elimination
 * kernel that will land in Day 4-5.
 *
 * Allocates `ceil(fill_factor * nnz_lower)` capacity, where `nnz_lower` is
 * the count of entries kept after lower-triangle filtering.  `fill_factor`
 * is clamped to [1.0, 20.0].  Day 3 will add a variant that uses symbolic
 * column counts for exact pre-allocation.
 *
 * @param mat          Input matrix (not modified).  Must be square.
 * @param perm         Optional symmetric permutation (length n) with
 *                     `perm[new] = old`.  May be NULL (identity).
 * @param fill_factor  Capacity multiplier for fill-in (clamped to [1, 20]).
 * @param[out] csc_out Receives the allocated CholCsc*.  Caller must free
 *                     with `chol_csc_free()`.  Set to NULL on error.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if `mat` or `csc_out` is NULL.
 * @return SPARSE_ERR_SHAPE if `mat` is not square.
 * @return SPARSE_ERR_BADARG if `perm` contains out-of-range or duplicate
 *         entries (best-effort check).
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t chol_csc_from_sparse(const SparseMatrix *mat, const idx_t *perm, double fill_factor,
                                  CholCsc **csc_out);

/**
 * Convert a CholCsc back into a linked-list SparseMatrix.
 *
 * Creates a new lower-triangular SparseMatrix (identity permutations)
 * from the CSC data.  If `perm` was used during `chol_csc_from_sparse`,
 * pass the same array here to un-permute back to the user's coordinate
 * system: CSC entry at `(new_r, new_c)` is written to
 * `mat[perm[new_r], perm[new_c]]`.  Upper-triangle mirror entries are
 * not written — the result stores only the lower triangle (including
 * diagonal), matching the Cholesky storage convention.
 *
 * @param csc          Input CSC (not modified).
 * @param perm         Optional symmetric permutation (length n) with
 *                     `perm[new] = old`.  May be NULL (identity).
 * @param[out] mat_out Receives the allocated SparseMatrix*.  Caller must
 *                     free with `sparse_free()`.  Set to NULL on error.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if `csc` or `mat_out` is NULL.
 * @return SPARSE_ERR_ALLOC if allocation fails.
 */
sparse_err_t chol_csc_to_sparse(const CholCsc *csc, const idx_t *perm, SparseMatrix **mat_out);

/**
 * Convert a SparseMatrix into a CholCsc using a precomputed Sprint 14
 * symbolic analysis to size storage exactly.
 *
 * This is the "symbolic-aware" allocation path.  Unlike
 * `chol_csc_from_sparse` (which sizes capacity by a heuristic
 * `fill_factor * nnz(A_lower)`), this variant allocates capacity equal
 * to the exact predicted `nnz(L)` from `analysis->sym_L`.  Combined with
 * Day 4-5's elimination kernel, that eliminates capacity-growth churn
 * (no `chol_csc_grow()` calls during elimination).
 *
 * The fill-reducing permutation `analysis->perm` is applied symmetrically
 * (same convention as `chol_csc_from_sparse`'s `perm` argument: `perm[new]
 * = old`).  NULL `analysis->perm` means identity.
 *
 * The CSC `col_ptr` produced here reflects A's current lower-triangle
 * layout (identical to what `chol_csc_from_sparse` would produce with the
 * same `perm`).  The extra capacity `nnz(L) - nnz(A_lower)` is reserved
 * space at the tail of the `values` / `row_idx` arrays for the
 * elimination kernel to absorb fill-in without reallocation.  A future
 * variant may pre-layout per-column slots to sym_L widths; that is a
 * Day 5 concern.
 *
 * @param mat           Input matrix (not modified).  Must be square and
 *                      match `analysis->n`.
 * @param analysis      Precomputed symbolic analysis from `sparse_analyze`.
 *                      Must have `type == SPARSE_FACTOR_CHOLESKY`.
 * @param[out] csc_out  Receives the allocated CholCsc*.  Caller frees with
 *                      `chol_csc_free()`.  Set to NULL on error.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if `mat`, `analysis`, or `csc_out` is NULL.
 * @return SPARSE_ERR_BADARG if `analysis->type != SPARSE_FACTOR_CHOLESKY`.
 * @return SPARSE_ERR_SHAPE if `mat` is not square or `mat->rows != analysis->n`.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t chol_csc_from_sparse_with_analysis(const SparseMatrix *mat,
                                                const sparse_analysis_t *analysis,
                                                CholCsc **csc_out);

/* ═══════════════════════════════════════════════════════════════════════
 * Invariant checking
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Check the structural invariants of a CholCsc.
 *
 * Verifies:
 *  - `col_ptr[0] == 0` and `col_ptr[n] == nnz`
 *  - `col_ptr` is monotonically non-decreasing
 *  - Within each non-empty column: row indices are sorted ascending and
 *    distinct, the first row index equals the column index (the diagonal
 *    is present and stored first), and all row indices are >= column
 *    index (lower-triangular invariant).
 *  - Empty columns are permitted and not flagged (the conversion allows
 *    structurally zero off-diagonal columns; Cholesky itself will detect
 *    and error on a missing diagonal at factor time).
 *
 * Intended for use from tests and from `assert()` in debug builds.  Not
 * a performance-critical path.
 *
 * @param csc  CSC to validate (not NULL).
 * @return SPARSE_OK when all invariants hold.
 * @return SPARSE_ERR_NULL if `csc` is NULL.
 * @return SPARSE_ERR_BADARG if any invariant is violated.
 */
sparse_err_t chol_csc_validate(const CholCsc *csc);

/* ═══════════════════════════════════════════════════════════════════════
 * Elimination: workspace and left-looking column kernel
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Algorithm (left-looking, column-by-column Cholesky)
 *
 * References:
 *   George & Liu, "Computer Solution of Large Sparse Positive Definite
 *                  Systems," Prentice-Hall, 1981 — Ch. 5 (left-looking
 *                  dense column accumulator) and Ch. 7 (compressed column
 *                  storage).
 *   Davis, "Direct Methods for Sparse Linear Systems," SIAM, 2006 — §4.4.
 *
 * For each column j = 0..n-1:
 *   (1) scatter: copy A[*,j]'s lower-triangle entries into a dense
 *       length-n workspace `dense_col`, tracking touched rows in
 *       `dense_pattern` via the `dense_marker` bit vector.
 *   (2) cmod(j,k) — for each k < j with L[j,k] != 0, update column j:
 *       for each stored L[i,k] with i >= j,
 *         dense_col[i] -= L[i,k] * L[j,k].
 *   (3) cdiv — set L[j,j] = sqrt(dense_col[j]); scale L[i,j] = dense_col[i]
 *       / L[j,j] for i > j in the pattern.  A non-positive diagonal
 *       signals NOT_SPD.
 *   (4) gather: write the new L values from the dense workspace back into
 *       column j's CSC slot.
 *   (5) end_column: zero dense_col / dense_marker for every row in the
 *       pattern so column j+1 starts from a clean workspace.
 *
 * Day 4 (this header) ships the scaffolding: workspace, scatter, cdiv,
 * gather, end_column, and a basic cmod that handles columns whose
 * elimination-tree contributors are already represented in column j's
 * CSC slot (no fill-in).  Day 5 completes cmod with fill-in support
 * (append rows to the pattern and grow the column's slot) and extends
 * `chol_csc_eliminate` to orchestrate the full left-looking sweep.
 */

/**
 * Dense workspace for a single column's elimination.
 *
 * `dense_col[i]` is the accumulator for row i of the column being
 * factored.  `dense_pattern[0..pattern_count-1]` lists the touched rows
 * (unordered) and `dense_marker[i] != 0` iff i is in the pattern.  The
 * marker lets scatter/cmod add a row to the pattern in O(1) without
 * scanning.
 *
 * Between columns, `end_column` clears `dense_col[i] = 0` and
 * `dense_marker[i] = 0` for every i in the pattern, in O(pattern_count)
 * rather than O(n), keeping the amortized cost proportional to the
 * eventual nnz(L).
 */
typedef struct {
    idx_t n;              /**< Matrix dimension (matches the CholCsc being factored). */
    double *dense_col;    /**< Length n, row accumulator for the current column. */
    idx_t *dense_pattern; /**< Length n, touched-row list (unordered). */
    int8_t *dense_marker; /**< Length n, 1 iff row is in dense_pattern. */
    idx_t pattern_count;  /**< Number of valid entries in dense_pattern. */
} CholCscWorkspace;

/**
 * Allocate a workspace for a matrix of dimension n.
 *
 * All three arrays are zero-initialised; `pattern_count == 0`.  The
 * caller frees with `chol_csc_workspace_free`.
 *
 * @param n        Matrix dimension (n >= 0).
 * @param[out] out Receives the allocated workspace.  Set to NULL on error.
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_BADARG (n<0), or SPARSE_ERR_ALLOC.
 */
sparse_err_t chol_csc_workspace_alloc(idx_t n, CholCscWorkspace **out);

/** Free a workspace.  Safe to call with NULL. */
void chol_csc_workspace_free(CholCscWorkspace *ws);

/**
 * Scatter column j's current CSC contents into the dense workspace.
 *
 * For each stored entry `(row_idx[p], values[p])` in column j, assigns
 * `dense_col[row_idx[p]] = values[p]` and adds the row to the pattern.
 * Precondition: `pattern_count == 0` and all `dense_marker` / `dense_col`
 * slots for this call's rows are zero (satisfied by `end_column` of the
 * previous column, or a fresh workspace).
 */
void chol_csc_scatter(const CholCsc *csc, idx_t j, CholCscWorkspace *ws);

/**
 * Apply cmod updates from columns k < j to the dense workspace.
 *
 * For each k in [0, j), locates L[j,k] via binary search in column k's
 * row_idx slice; when present, subtracts the rank-1 contribution
 * `L[i,k] * L[j,k]` from `dense_col[i]` for every stored L[i,k] with
 * i >= j.  Any i updated but not already in the pattern is appended
 * (tracked for `end_column` correctness even when the slot is outside
 * column j's CSC range — Day 5 will promote such entries into column j).
 *
 * Expects the CSC to have columns 0..j-1 already factored (their `values`
 * slots hold L).
 */
void chol_csc_cmod(const CholCsc *csc, idx_t j, CholCscWorkspace *ws);

/**
 * Compute L[j,j] = sqrt(dense_col[j]) and scale dense_col[i] /= L[j,j]
 * for every i > j in the pattern.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NOT_SPD if dense_col[j] <= 0 (near-zero or negative
 *         diagonal — the matrix is not SPD).
 * @return SPARSE_ERR_NULL if ws is NULL.
 */
sparse_err_t chol_csc_cdiv(CholCscWorkspace *ws, idx_t j);

/**
 * Gather the factored column from the dense workspace back into the CSC.
 *
 * Sorts `dense_pattern` ascending in place, applies a drop tolerance
 * (any entry with `|value| < drop_tol * |L[j,j]|` is discarded, except
 * the diagonal itself which is always kept), and writes the surviving
 * entries — in sorted order — into column j's CSC slot.  Fill-in rows
 * (those added to the pattern by cmod but not present in the original
 * slot) are materialized by growing the CSC's per-column space in place:
 * columns j+1..n-1 are shifted, and `chol_csc_grow()` is invoked if
 * total capacity would overflow.  When the pattern shrinks below the
 * current slot size (drop tolerance applied), columns j+1..n-1 are
 * shifted left to keep the dense-packing invariant.
 *
 * @param csc       Target CSC — column j is overwritten, columns
 *                  j+1..n may be shifted.
 * @param j         Column index.
 * @param ws        Workspace (non-const: `dense_pattern` is sorted).
 * @param drop_tol  Relative drop tolerance, typically SPARSE_DROP_TOL.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_ALLOC if capacity growth fails.
 */
sparse_err_t chol_csc_gather(CholCsc *csc, idx_t j, CholCscWorkspace *ws, double drop_tol);

/**
 * Reset the dense workspace at the end of a column: zero `dense_col[i]`
 * and `dense_marker[i]` for every i in the pattern, and clear the
 * pattern list.  O(pattern_count).
 */
void chol_csc_end_column(CholCscWorkspace *ws);

/**
 * Perform left-looking Cholesky elimination on a CholCsc.
 *
 * Runs scatter → cmod → cdiv → gather → end_column for each column
 * j = 0..n-1.  On success, the CSC's `values` array contains the
 * Cholesky factor L (lower triangular, including diagonal).
 *
 * Day 4 supports matrices where column j's CSC slot already has room
 * for every nonzero in L[*,j] (no fill-in).  Diagonal, tridiagonal, and
 * most banded SPD matrices fall in this class.  Day 5 extends the
 * kernel with fill-in handling.
 *
 * @param csc  Input CSC (lower triangle of A, permuted as desired).  On
 *             return, contains L such that L*L^T = A (in the permuted
 *             space).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if csc is NULL.
 * @return SPARSE_ERR_NOT_SPD if a non-positive diagonal is encountered.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 */
sparse_err_t chol_csc_eliminate(CholCsc *csc);

/* ═══════════════════════════════════════════════════════════════════════
 * Day 6: Triangular solves on a factored CholCsc
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Cholesky produces A = L*L^T, so solving A*x = b reduces to
 *   (forward)   L*y = b
 *   (backward)  L^T*x = y
 *
 * With L stored in CSC, both sweeps walk columns contiguously:
 *   - Forward: for each column j left-to-right, divide the j-th row of
 *     the partial solution by L[j,j], then subtract L[i,j] * x[j] from
 *     every below-diagonal entry L[i,j].  That is a single pass through
 *     column j's row_idx/values.
 *   - Backward: for each column j right-to-left, accumulate
 *     sum_{i>j} L[i,j] * x[i] (row j of L^T is exactly the below-
 *     diagonal slice of column j of L), subtract from x[j], divide by
 *     L[j,j].  Again one pass through column j's slice.
 *
 * So CSC is naturally suited to both Cholesky sweeps — *no transpose is
 * materialised*, unlike LU where row-oriented forward/backward sweeps
 * want a CSR-style layout.
 *
 * Singularity detection uses `sparse_rel_tol(sqrt(factor_norm),
 * SPARSE_DROP_TOL)` as the threshold, matching `sparse_cholesky.c`.
 * The square-root scaling reflects that L entries grow as sqrt of A's
 * entries in Cholesky.
 */

/**
 * Solve L * L^T * x = b using a factored CholCsc.
 *
 * `b` and `x` may alias; the solve overwrites x in place either way.
 *
 * @param L    Factored CholCsc (output of `chol_csc_eliminate`).
 * @param b    Right-hand side (length L->n).
 * @param x    Solution (length L->n, output).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any pointer is NULL.
 * @return SPARSE_ERR_SINGULAR if a diagonal entry falls below
 *         `sparse_rel_tol(sqrt(L->factor_norm), SPARSE_DROP_TOL)`, or if
 *         a column's diagonal is missing from the CSC.
 */
sparse_err_t chol_csc_solve(const CholCsc *L, const double *b, double *x);

/**
 * Solve L * L^T * x = b with an outer symmetric permutation.
 *
 * Given `perm[new] = old` (the same convention used by
 * `chol_csc_from_sparse_with_analysis`), this function permutes b into
 * the new space, delegates to `chol_csc_solve`, and unpermutes the
 * result back into user coordinates.  A NULL `perm` short-circuits to
 * `chol_csc_solve`.
 *
 * @param L     Factored CholCsc in the permuted space.
 * @param perm  Symmetric permutation (length L->n) or NULL for identity.
 * @param b     RHS in user coordinates.
 * @param x     Solution in user coordinates (may alias b).
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_ALLOC, SPARSE_ERR_SINGULAR.
 */
sparse_err_t chol_csc_solve_perm(const CholCsc *L, const idx_t *perm, const double *b, double *x);

/* ═══════════════════════════════════════════════════════════════════════
 * Day 6: Internal public-API shims — convert + factor (+ solve)
 *
 * These will become the default Cholesky backend in Day 12 once
 * benchmarks confirm the expected speedup over the linked-list path.
 * For now they are internal helpers, exercised only by tests.
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Convert A to CSC (optionally applying `analysis->perm`), run
 * `chol_csc_eliminate`, and return the factored CholCsc.
 *
 * Pass `analysis = NULL` for no reordering and heuristic capacity.
 * Pass a Cholesky analysis (type SPARSE_FACTOR_CHOLESKY) to use the
 * fill-reducing permutation and pre-size capacity to `sym_L.nnz`.
 *
 * @param A           Symmetric positive-definite matrix.
 * @param analysis    Optional symbolic analysis.
 * @param[out] L_out  Receives the factored CholCsc.  Caller frees with
 *                    `chol_csc_free`.  Set to NULL on error.
 * @return SPARSE_OK, or error codes from conversion/elimination.
 */
sparse_err_t chol_csc_factor(const SparseMatrix *A, const sparse_analysis_t *analysis,
                             CholCsc **L_out);

/**
 * Factor A and solve A*x = b in a single call.
 *
 * Equivalent to `chol_csc_factor` followed by `chol_csc_solve_perm` (or
 * `chol_csc_solve` when `analysis` is NULL).  The intermediate factor
 * is freed before return; callers that plan multiple solves against
 * the same matrix should use `chol_csc_factor` directly.
 *
 * @param A         Symmetric positive-definite matrix.
 * @param analysis  Optional symbolic analysis.
 * @param b         RHS in user coordinates (length A->rows).
 * @param x         Solution in user coordinates (length A->rows).
 * @return SPARSE_OK or propagated error.
 */
sparse_err_t chol_csc_factor_solve(const SparseMatrix *A, const sparse_analysis_t *analysis,
                                   const double *b, double *x);

/* ═══════════════════════════════════════════════════════════════════════
 * Day 10: Supernode detection
 * ═══════════════════════════════════════════════════════════════════════
 *
 * A fundamental supernode is a contiguous group of columns j, j+1, ...,
 * j+s-1 in a Cholesky factor L such that every column in the group has
 * the same nonzero structure below the supernode's diagonal block.
 *
 * Equivalent characterisation (Liu, Ng, Peyton, SIMAX 1993) — two
 * adjacent columns j and j+1 share a supernode iff:
 *   (1) L[j+1, j] is the first stored sub-diagonal entry of column j
 *       (so j+1 is the immediate etree parent of j).
 *   (2) Column j+1 stores exactly one fewer entry than column j
 *       (the diagonal of j+1 replaces column j's entry at row j+1).
 *   (3) The remaining row indices of column j+1 match column j's rows
 *       starting two positions after j's diagonal, element-for-element.
 *
 * Conditions (1)-(3) can be tested directly on the sorted CSC
 * col_ptr / row_idx arrays in O(nnz(column j)) per pair, without
 * separately materialising the etree.  The classical etree+colcount
 * characterisation is equivalent.
 *
 * Supernodes of size < min_size are reported as a sequence of
 * scalar columns instead (caller's elimination kernel handles them
 * column-by-column, like Day 5's scatter-gather path).  Day 11 uses
 * the detected supernodes to apply dense BLAS-style kernels to the
 * diagonal blocks and the below-diagonal panels.
 */

/**
 * Detect fundamental supernodes in a Cholesky CSC factor (or symbolic
 * structure sized the same way).
 *
 * Writes parallel start / size arrays describing each supernode that
 * meets `min_size`.  Supernodes of size < `min_size` are skipped (the
 * caller treats their columns scalar-by-scalar).  The arrays are
 * written in ascending column order.
 *
 * @param L             Cholesky CSC with sorted per-column row indices
 *                      and the diagonal stored first in each column.
 * @param min_size      Minimum supernode size to report (typically 4,
 *                      matching Sprint 10's `lu_detect_dense_blocks`
 *                      threshold).  Pass 1 to report every fundamental
 *                      supernode, including singletons.
 * @param super_starts  Caller-allocated array, capacity >= L->n.
 *                      `super_starts[i]` = starting column of supernode i.
 * @param super_sizes   Caller-allocated array parallel to super_starts;
 *                      `super_sizes[i]` = number of columns in supernode i.
 * @param[out] count    Receives the number of supernodes detected.
 * @return SPARSE_OK on success; SPARSE_ERR_NULL on null inputs;
 *         SPARSE_ERR_BADARG if `min_size < 1`.
 */
sparse_err_t chol_csc_detect_supernodes(const CholCsc *L, idx_t min_size, idx_t *super_starts,
                                        idx_t *super_sizes, idx_t *count);

#ifndef NDEBUG
/**
 * Debug dump of a supernode partition.  Prints each supernode's
 * `[start, start+size)` column range to stdout.  Compiled out in
 * release builds (NDEBUG defined).
 */
void chol_csc_dump_supernodes(const idx_t *super_starts, const idx_t *super_sizes, idx_t count);
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Day 11: Dense Cholesky primitives + supernodal-aware elimination entry
 * ═══════════════════════════════════════════════════════════════════════
 *
 * These dense kernels are self-contained and reusable by future
 * supernodal paths.  They work on column-major arrays, matching the
 * convention of Sprint 10's `lu_dense_factor` / `lu_dense_solve`.
 *
 * ─── Current integration status ───────────────────────────────────────
 *
 * Day 11 ships:
 *   (1) `chol_dense_factor` — in-place Cholesky on an n×n column-major
 *       block, with the same non-positive-diagonal → NOT_SPD semantics
 *       as the scalar CSC path.
 *   (2) `chol_dense_solve_lower` — forward substitution for L·x = b on
 *       the output of (1).
 *   (3) `chol_csc_eliminate_supernodal` — an elimination entry point
 *       that detects supernodes with `chol_csc_detect_supernodes` and
 *       then runs the Day 5 scalar kernel.  The detection step is real
 *       and validated; the *batched* supernodal cmod + dense factor +
 *       panel solve is a future-sprint deliverable (cross-supernode
 *       cmod in packed CSC storage requires a workspace layout change
 *       that is beyond the Day 11 scope).
 *
 * Correctness: residuals match the scalar path bit-for-bit on every
 * test matrix today because the integrated path still executes the
 * scalar kernel.  The dense primitives stand on their own for unit
 * testing and future reuse.
 */

/**
 * In-place dense Cholesky on a column-major n×n block.
 *
 * Treats `A[i + j*lda]` as the (i, j) entry of an n×n symmetric
 * positive-definite matrix.  Only the lower triangle is read; the upper
 * triangle is ignored.  On return, the lower triangle (including the
 * diagonal) holds the Cholesky factor L such that A = L·L^T.  The
 * upper triangle is untouched.
 *
 * @param A    Column-major array, must have at least n + (n-1)*lda entries.
 * @param n    Block dimension (n >= 0).
 * @param lda  Leading dimension (stride between columns), lda >= n.
 * @param tol  Relative drop tolerance used for singularity detection:
 *             a diagonal accumulator below
 *             `sparse_rel_tol(||A||_inf_approx, tol)` is rejected with
 *             `SPARSE_ERR_NOT_SPD`.  Pass `tol <= 0` to select the
 *             standard default (`SPARSE_DROP_TOL`); note that this is
 *             still a norm-relative check with a DBL_MIN floor — it
 *             does not request a strict positivity-only test.
 * @return SPARSE_OK, SPARSE_ERR_NULL (A=NULL), SPARSE_ERR_BADARG
 *         (n<0 or lda<n), SPARSE_ERR_NOT_SPD (non-positive / tiny
 *         diagonal encountered).
 */
sparse_err_t chol_dense_factor(double *A, idx_t n, idx_t lda, double tol);

/**
 * Forward substitution on a dense lower-triangular factor.
 *
 * Solves L·x = b in place, where L is the output of
 * `chol_dense_factor`.  b is overwritten with x.
 *
 * @param L    Factored matrix (column-major, only lower triangle is read).
 * @param n    Dimension.
 * @param lda  Leading dimension.
 * @param b    RHS of length n (input), overwritten with the solution.
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_BADARG, SPARSE_ERR_SINGULAR
 *         (if a diagonal entry is zero, which shouldn't happen after a
 *         successful `chol_dense_factor` but is checked defensively).
 */
sparse_err_t chol_dense_solve_lower(const double *L, idx_t n, idx_t lda, double *b);

/**
 * Sprint 19 Day 11: Dense LDL^T factor with Bunch-Kaufman pivoting.
 *
 * Column-major analogue of `sparse_ldlt.c`'s BK kernel, intended for
 * the Sprint 19 Days 12-14 batched supernodal LDL^T path to call per
 * supernode's diagonal block (the same way `chol_dense_factor` serves
 * the Cholesky batched supernodal kernel).
 *
 * Input: `A` is n×n column-major symmetric with BOTH triangles
 * populated so the four-criteria BK scan can read `A[i + r*lda]`
 * directly without symmetry reflection.
 *
 * Output:
 *   - `A` below-diagonal holds unit-L: diagonal set to 1.0, L[k+1, k] = 0
 *     for 2×2 pivots (the coupling lives in `D_offdiag`).  Upper
 *     triangle not preserved.
 *   - `D[k]`: 1×1 pivot scalar, or the (k, k) diagonal of a 2×2 block;
 *     `D[k+1]` for 2×2 holds (k+1, k+1).
 *   - `D_offdiag[k]`: 2×2 block's (k, k+1) off-diagonal; 0 for 1×1.
 *   - `pivot_size[k]`: 1 for 1×1, 2 for both indices of a 2×2 pair.
 *   - `elem_growth_out` (optional, may be NULL): receives the max
 *     |L[i, j]| observed during the factor.
 *
 * @param A              n×n column-major symmetric buffer (both triangles).
 * @param D              Length-n output.
 * @param D_offdiag      Length-n output.
 * @param pivot_size     Length-n output.
 * @param n              Dimension.
 * @param lda            Leading dimension (`lda >= n`).
 * @param tol            Drop / singularity tolerance; <=0 uses SPARSE_DROP_TOL.
 * @param elem_growth_out Optional output for max observed |L|.
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_BADARG, SPARSE_ERR_SINGULAR.
 */
sparse_err_t ldlt_dense_factor(double *A, double *D, double *D_offdiag, idx_t *pivot_size, idx_t n,
                               idx_t lda, double tol, double *elem_growth_out);

/**
 * Supernode-aware elimination entry point.
 *
 * Currently a thin wrapper that detects supernodes and then delegates
 * to the scalar `chol_csc_eliminate`.  The supernode detection lets
 * Day 11 test infrastructure observe that a detected partition exists
 * and falls into reasonable shape without changing numeric behaviour.
 * A future sprint will replace the scalar delegation with a batched
 * dense-kernel path over each supernode's diagonal block and panel.
 *
 * @param csc       Input CSC with A's lower triangle (as produced by
 *                  `chol_csc_from_sparse` / `chol_csc_from_sparse_with_analysis`).
 * @param min_size  Minimum supernode size to consider, e.g. 4.
 * @return SPARSE_OK, or any error from scalar elimination or detection.
 */
sparse_err_t chol_csc_eliminate_supernodal(CholCsc *csc, idx_t min_size);

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 6: supernode extract / writeback plumbing
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The batched supernodal path (Days 7-8) replaces per-column cdiv loops
 * with one `chol_dense_factor` call on the supernode's diagonal block
 * plus one `chol_dense_solve_lower` pass per panel column.  Both dense
 * kernels operate on a column-major buffer, which has to be scattered
 * from packed CSC storage on the way in and gathered back on the way
 * out.  Day 6 ships only the extract/writeback plumbing — with round-
 * trip tests — so the Day 7 diagonal factor and Day 8 panel solve can
 * focus on the numeric logic alone.
 *
 * Dense buffer layout (column-major, leading dimension `lda`):
 *
 *   panel_height = s_size + |below-supernode rows|
 *               = col_ptr[s_start + 1] - col_ptr[s_start]
 *
 *   rows [0, s_size):             diagonal block (s_size × s_size).
 *                                  Column j in the block stores entries
 *                                  at rows [j, s_size) — the lower
 *                                  triangle of the block.  Upper-
 *                                  triangle cells [0, j) are left
 *                                  uninitialised; chol_dense_factor
 *                                  only reads the lower triangle.
 *   rows [s_size, panel_height):  shared below-supernode panel,
 *                                  populated identically in every
 *                                  column of the supernode.
 *
 *   row_map[i] for i in [0, s_size)             = s_start + i  (diag block)
 *   row_map[i] for i in [s_size, panel_height)  = global CSC row of the
 *                                                 i-th below-supernode
 *                                                 entry (sorted ascending
 *                                                 — taken verbatim from
 *                                                 col s_start's row_idx).
 */

/** Panel height of a supernode starting at `s_start`.
 *
 * Equal to `s_size + |below-supernode rows|`.  The value is read
 * directly from `col_ptr[s_start + 1] - col_ptr[s_start]`, which by the
 * fundamental-supernode invariant is the row count of the first column
 * in the supernode. */
static inline idx_t chol_csc_supernode_panel_height(const CholCsc *csc, idx_t s_start) {
    return csc->col_ptr[s_start + 1] - csc->col_ptr[s_start];
}

/**
 * Extract a supernode from CSC storage into a dense column-major buffer.
 *
 * @param csc              Input CSC (not modified).  Must satisfy the
 *                         fundamental-supernode invariant on the
 *                         column range [s_start, s_start + s_size).
 * @param s_start          Starting column of the supernode.
 * @param s_size           Number of columns in the supernode (>= 1,
 *                         s_start + s_size <= csc->n).
 * @param dense            Column-major output buffer; must span at
 *                         least `lda * s_size` doubles.  Values in
 *                         the upper triangle of the diagonal block
 *                         and any padding rows above `panel_height`
 *                         are left untouched.
 * @param lda              Leading dimension between dense columns;
 *                         must satisfy `lda >= panel_height`.  Call
 *                         `chol_csc_supernode_panel_height` first to
 *                         size the buffer.
 * @param row_map          Output map from local row index into the
 *                         dense buffer to the global CSC row index.
 *                         Must have capacity >= panel_height.
 * @param[out] panel_height_out  Receives the panel height (also the
 *                               number of valid entries in row_map).
 * @return SPARSE_OK on success; SPARSE_ERR_NULL on null args;
 *         SPARSE_ERR_BADARG on invalid range or insufficient lda;
 *         SPARSE_ERR_BADARG if a column in the supernode references a
 *         row outside the first column's row_map (violation of the
 *         fundamental-supernode invariant).
 */
sparse_err_t chol_csc_supernode_extract(const CholCsc *csc, idx_t s_start, idx_t s_size,
                                        double *dense, idx_t lda, idx_t *row_map,
                                        idx_t *panel_height_out);

/**
 * Gather a supernode's dense column-major buffer back into CSC storage.
 *
 * The inverse of `chol_csc_supernode_extract`.  For each stored entry
 * in columns [s_start, s_start + s_size), the writeback looks up the
 * entry's CSC row in `row_map` to find its local row in the dense
 * buffer and overwrites `values[p]` with the corresponding cell.
 *
 * The CSC's per-column `row_idx` and `col_ptr` arrays are not changed
 * — the supernode's structural pattern is preserved end-to-end.  This
 * matches the Cholesky invariant that the factored L within a
 * fundamental supernode has the same structure as the pre-factor
 * scatter (no new fill inside a supernode).
 *
 * Below-diagonal entries whose magnitude falls below
 * `drop_tol * |L[j, j]|` are written back as exactly 0.0, mirroring
 * the scalar kernel's `chol_csc_gather` policy so the supernodal path
 * produces factors with the same numerical dropping semantics as the
 * scalar / linked-list paths.  The diagonal is never dropped.  Pass
 * `drop_tol = 0.0` to retain every value verbatim (useful in tests
 * that compare against an exact dense factor).
 *
 * @return SPARSE_OK on success; SPARSE_ERR_NULL on null args;
 *         SPARSE_ERR_BADARG on invalid range, insufficient lda, or a
 *         stored row outside the provided row_map.
 */
sparse_err_t chol_csc_supernode_writeback(CholCsc *csc, idx_t s_start, idx_t s_size,
                                          const double *dense, idx_t lda, const idx_t *row_map,
                                          idx_t panel_height, double drop_tol);

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 7: supernode diagonal block factor
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Given a supernode's extracted dense buffer (from
 * `chol_csc_supernode_extract`), this helper applies left-looking
 * external cmod from every prior column k in [0, s_start) and then
 * runs `chol_dense_factor` on the top s_size × s_size slab.
 *
 * The cmod uses the current CSC contents of col k as the L column
 * (so prior columns must already be factored — either as part of an
 * earlier supernode pass or via the scalar kernel).  The helper does
 * not touch the CSC; the caller decides whether to write back the
 * factored diagonal block (and, after Day 8, the solved panel).
 *
 * After a successful return, `dense[0..s_size-1, 0..s_size-1]` holds
 * the Cholesky factor L of the external-cmod'd diagonal block, and
 * `dense[s_size..panel_height-1, 0..s_size-1]` holds the external-
 * cmod'd panel values (still pre-triangular-solve; Day 8 runs
 * `chol_dense_solve_lower` on this panel).
 */

/**
 * Apply external cmod + dense Cholesky to a supernode's extracted buffer.
 *
 * @param csc           Input CSC (const).  Columns [0, s_start) must
 *                      already be factored (i.e., their stored values
 *                      are L entries, not A entries).
 * @param s_start       Starting column of the supernode.
 * @param s_size        Number of columns in the supernode.
 * @param dense         Column-major buffer, already populated by
 *                      `chol_csc_supernode_extract` with A's values.
 *                      The top s_size × s_size slab is factored in
 *                      place; the panel portion receives external
 *                      cmod but no triangular solve.
 * @param lda           Leading dimension, lda >= panel_height.
 * @param row_map       Row map from the extract call.
 * @param panel_height  Panel height from the extract call.
 * @param tol           Relative tolerance for the dense factor
 *                      (pass 0 or a negative value to select
 *                      `SPARSE_DROP_TOL`).
 * @return SPARSE_OK on success, SPARSE_ERR_NULL on null args,
 *         SPARSE_ERR_BADARG on invalid range or insufficient lda,
 *         SPARSE_ERR_ALLOC if the scratch buffer for L[supernode, k]
 *         cannot be allocated, SPARSE_ERR_NOT_SPD if the dense factor
 *         detects a non-positive-definite diagonal block.
 */
sparse_err_t chol_csc_supernode_eliminate_diag(const CholCsc *csc, idx_t s_start, idx_t s_size,
                                               double *dense, idx_t lda, const idx_t *row_map,
                                               idx_t panel_height, double tol);

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 8: supernode panel triangular solve
 * ═══════════════════════════════════════════════════════════════════════
 *
 * After Day 7's `chol_csc_supernode_eliminate_diag` returns, the
 * dense buffer's diagonal block holds the Cholesky factor L_diag and
 * the panel slab holds the external-cmod'd Schur-complement values
 * (still needing the within-supernode triangular solve).  The
 * below-supernode L entries come out of the triangular solve:
 *
 *   L_panel = rect * L_diag^{-T}
 *
 * Equivalently, for each panel row i, solve
 *   L_diag * x = rect[i, :]^T
 * in place (forward substitution) to obtain x = L_panel[i, :]^T.
 * This is exactly `chol_dense_solve_lower` applied row-by-row.
 */

/**
 * Apply the panel triangular solve for one supernode.
 *
 * @param L_diag     Factored s_size × s_size diagonal block (lower
 *                   triangular), column-major with leading dimension
 *                   `lda_diag`.  Read-only.
 * @param s_size     Supernode size.
 * @param lda_diag   Leading dimension of L_diag.
 * @param panel      Panel slab, `panel_rows × s_size`, column-major
 *                   with leading dimension `lda_panel`.  On entry
 *                   holds the external-cmod'd Schur values; on
 *                   successful return holds the below-supernode L
 *                   entries.
 * @param lda_panel  Leading dimension of the panel (>= panel_rows).
 * @param panel_rows Number of panel rows (= panel_height - s_size
 *                   for a standard supernode extraction).  May be 0
 *                   (no panel — the routine returns SPARSE_OK
 *                   immediately).
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_BADARG,
 *         SPARSE_ERR_ALLOC (scratch vector for the row-by-row solve),
 *         SPARSE_ERR_SINGULAR (propagated from `chol_dense_solve_lower`
 *         if L_diag has a zero diagonal — should not happen after a
 *         successful `chol_dense_factor`).
 */
sparse_err_t chol_csc_supernode_eliminate_panel(const double *L_diag, idx_t s_size, idx_t lda_diag,
                                                double *panel, idx_t lda_panel, idx_t panel_rows);

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 10: CSC → linked-list writeback for transparent dispatch
 * ═══════════════════════════════════════════════════════════════════════
 *
 * After a matrix has been factored via the CSC working-format path
 * (`chol_csc_from_sparse` + `chol_csc_eliminate_supernodal`), callers
 * still want the result to look like what the linked-list Cholesky
 * would have produced — same `factor_norm`, same `reorder_perm`,
 * same `factored` flag, same internal storage shape and L values.
 *
 * `chol_csc_writeback_to_sparse` transplants L from the CSC back into
 * a `SparseMatrix`, then sets every field the scalar factor would
 * have set.  The resulting `SparseMatrix` is indistinguishable from
 * `sparse_cholesky_factor_opts` output on the same input (values
 * within round-off; everything else exact).
 *
 * End-state field checklist (matches `sparse_cholesky_factor_opts`):
 *
 *   mat->row_headers / col_headers    — L entries in the permuted
 *                                        "new" coordinate space, lower
 *                                        triangle only (diagonal + below).
 *   mat->row_perm / col_perm          — identity (scalar path resets
 *                                        these to identity after the
 *                                        symmetric permutation).
 *   mat->inv_row_perm / inv_col_perm  — identity (same reason).
 *   mat->reorder_perm                 — copy of `perm` (caller's
 *                                        fill-reducing permutation;
 *                                        NULL if no reorder was used).
 *   mat->factor_norm                  — L->factor_norm (= ||A||_inf
 *                                        cached at conversion time).
 *   mat->factored                     — 1.
 *   mat->nnz                          — updated to L's nnz.
 *   mat->cached_norm                  — invalidated (-1.0), since the
 *                                        matrix contents changed from
 *                                        A to L.
 */

/**
 * Populate `mat` with the L factor stored in `L`, preserving every
 * field that the linked-list `sparse_cholesky_factor_opts` would have
 * set.
 *
 * @param L     Factored CholCsc (post-elimination).  Must have been
 *              produced from the same matrix (and the same `perm`)
 *              that the caller wants `mat` to end up matching.
 * @param mat   Target `SparseMatrix` (mutable).  Must be
 *              **unfactored** (`mat->factored == 0`) with identity
 *              row_perm / col_perm / inv_row_perm / inv_col_perm.
 *              Its current entries (A's values) are discarded.
 * @param perm  Fill-reducing permutation used at conversion time
 *              (`perm[new] = old`), or NULL when no reorder was used.
 *              A copy is stored into `mat->reorder_perm` — the caller
 *              retains ownership of the input buffer.
 * @return SPARSE_OK, SPARSE_ERR_NULL (L or mat NULL),
 *         SPARSE_ERR_SHAPE (n mismatch),
 *         SPARSE_ERR_BADARG (already factored, or non-identity
 *         row_perm / col_perm), SPARSE_ERR_ALLOC.
 */
sparse_err_t chol_csc_writeback_to_sparse(const CholCsc *L, SparseMatrix *mat, const idx_t *perm);

#endif /* SPARSE_CHOL_CSC_INTERNAL_H */
