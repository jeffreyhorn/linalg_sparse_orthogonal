#ifndef SPARSE_LDLT_CSC_INTERNAL_H
#define SPARSE_LDLT_CSC_INTERNAL_H

/**
 * @file sparse_ldlt_csc_internal.h
 * @brief CSC working format for LDL^T numeric factorization.
 *
 * Not part of the public API.  Used by sparse_ldlt_csc.c.
 *
 * ─── Design: LDL^T on top of the Cholesky CSC ──────────────────────────
 *
 * The CSC working format introduced in Days 1-6 for Cholesky lays out L
 * as sorted-row columns with the diagonal first in each column.  LDL^T
 * reuses exactly that layout for its L factor — unit lower triangular
 * with the diagonal stored as 1.0 for uniformity — and attaches three
 * auxiliary arrays capturing D's block structure:
 *
 *   L ─ unit lower triangular factor, stored as a `CholCsc` with
 *       `L->values[col_ptr[j]]` carrying the (stored) unit 1.0 at the
 *       diagonal.  Before elimination the diagonal position actually
 *       carries A[j,j] (conversion preserves A's values bit-for-bit).
 *       After elimination (Days 8-9) the diagonal holds the stored 1.0
 *       and below-diagonal rows hold the L multipliers.
 *   D          ─ length n, diagonal of D.  1x1 pivot at step k stores the
 *                scalar in D[k].  2x2 pivot at k,k+1 stores the block
 *                diagonal in D[k] and D[k+1].
 *   D_offdiag  ─ length n, off-diagonal of 2x2 pivots.  D_offdiag[k] ==
 *                D(k,k+1) == D(k+1,k) when the block at k,k+1 is 2x2.
 *                Zero for 1x1 pivots.
 *   pivot_size ─ length n, 1 for a 1x1 pivot.  For a 2x2 pivot block at
 *                k, k+1 both `pivot_size[k]` and `pivot_size[k+1]` are 2,
 *                matching `sparse_ldlt_t`.
 *   perm       ─ length n, composed symmetric permutation such that
 *                perm[new] = old.  Covers any fill-reducing permutation
 *                applied at conversion plus the Bunch-Kaufman pivoting
 *                chosen during elimination.  Initialised to identity (or
 *                the caller-supplied fill-reducing perm) at conversion.
 *
 * The field layout deliberately mirrors the linked-list `sparse_ldlt_t`
 * (src/sparse_ldlt.h) so solve / inertia / refinement helpers can be
 * shared or ported verbatim in Days 9+.
 */

#include "sparse_analysis.h"
#include "sparse_chol_csc_internal.h"
#include "sparse_matrix_internal.h"
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * LdltCsc: CSC working format for LDL^T numeric factorization
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    idx_t n;            /**< Matrix dimension (n × n). */
    CholCsc *L;         /**< Unit lower triangular factor (owned). */
    double *D;          /**< Diagonal of D (length n). */
    double *D_offdiag;  /**< 2x2 pivot off-diagonals (length n).  Zero for 1x1. */
    idx_t *pivot_size;  /**< Pivot block size per step (length n).  1 or 2. */
    idx_t *perm;        /**< Composed symmetric perm (length n), perm[new] = old. */
    double factor_norm; /**< ||A||_inf cached at conversion time. */

    /* Sprint 19 Day 8: row-adjacency index for the factored prefix.
     *
     * `ldlt_csc_cmod_unified`'s Phase A (src/sparse_ldlt_csc.c) used to
     * iterate `kp = 0..step_k-1` and binary-search every prior column
     * for `L(col, kp)`, yielding O(step_k · log nnz) per elimination
     * step even when row `col` is very sparse.  The linked-list
     * reference (`acc_schur_col` in src/sparse_ldlt.c) iterates only
     * over the columns with `L(col, j) != 0` via the cross-linked row
     * list on the `SparseMatrix`.
     *
     * `row_adj[r]` is a dynamically-grown array of the prior column
     * indices `kp < r` where `L(r, kp)` was stored during elimination.
     * Populated by `ldlt_csc_row_adj_append` as each column finishes
     * its writeback; Phase A (Day 9) then iterates this list instead
     * of `[0, step_k)`.
     *
     * Option A (per-row dynamic array) chosen over Option B (global
     * CSC transpose) because the per-row footprint is bounded by
     * `n · avg_fill` — roughly 2× the L capacity we already pay for —
     * while keeping writes O(1) amortised with geometric growth.
     * Option B would require updating O(n) row pointers on every
     * column factor. */
    idx_t **row_adj; /**< Per-row lists of prior columns with stored entries (length n); each slot
                        NULL until first append. */
    idx_t
        *row_adj_count; /**< Number of entries currently populated in each row_adj[r] (length n). */
    idx_t *row_adj_cap; /**< Allocated capacity of each row_adj[r] (length n). */
} LdltCsc;

/* ═══════════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Allocate an LdltCsc, including its embedded CholCsc (for L) and the
 * D / D_offdiag / pivot_size / perm arrays.
 *
 * On success, `*out` points to a zero-initialised LdltCsc:
 *   - `L` is allocated via `chol_csc_alloc(n, initial_nnz, ...)`
 *   - `D`, `D_offdiag` zeroed
 *   - `pivot_size[i]` defaults to 1 (stable default for a not-yet-
 *     factored matrix; Day 8 overrides each entry during elimination)
 *   - `perm[i] = i` (identity)
 *   - `n` set; `factor_norm = 0.0`
 *
 * @param n            Matrix dimension (n >= 0).
 * @param initial_nnz  Initial nnz capacity for the embedded L (clamped to >= 1).
 * @param[out] out     Receives the allocated LdltCsc*.  NULL on error.
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_BADARG (n<0), SPARSE_ERR_ALLOC.
 */
sparse_err_t ldlt_csc_alloc(idx_t n, idx_t initial_nnz, LdltCsc **out);

/** Free an LdltCsc and all its arrays.  Safe with NULL. */
void ldlt_csc_free(LdltCsc *m);

/**
 * Append `col` to the row-adjacency list `F->row_adj[row]`, growing
 * the per-row array geometrically (2×) when capacity is hit.
 *
 * Called as each column finishes its writeback during `ldlt_csc_eliminate_native`
 * (Day 9) so `ldlt_csc_cmod_unified`'s Phase A can iterate only the
 * prior columns that carry a stored entry in a given row rather than
 * scanning `[0, step_k)` and binary-searching.  Insertion order
 * preserved; callers append monotonically increasing `col` values
 * during the left-to-right column sweep, so the list stays sorted
 * ascending.
 *
 * @param F    LdltCsc whose row-adjacency index to update.
 * @param row  Row index in [0, F->n).
 * @param col  Prior column index (`col < row` by Cholesky invariant).
 * @return SPARSE_OK, SPARSE_ERR_NULL (F NULL), SPARSE_ERR_BADARG (row
 *         or col out of range), SPARSE_ERR_ALLOC.
 */
sparse_err_t ldlt_csc_row_adj_append(LdltCsc *F, idx_t row, idx_t col);

/**
 * Detect fundamental supernodes of `F->L` that also respect the 2×2
 * pivot boundaries in `F->pivot_size`.
 *
 * Sprint 19 Day 10 builds on `chol_csc_detect_supernodes` (Sprint 17
 * Day 10) with one extra invariant: a 2×2 pivot pair at (k, k+1) —
 * identified by `pivot_size[k] == pivot_size[k+1] == 2` — is atomic.
 * Either both columns are in the same supernode or neither is.
 * Supernodes that would end on the first of a 2×2 pair are either
 * extended by one column (if pattern permits) or truncated to
 * exclude it; 2×2 pairs that can't be kept together through their
 * pattern become scalar-factored columns.
 *
 * Callers feed the returned supernodes into the Days 11-14 batched
 * path (extract / eliminate_diag / eliminate_panel / writeback),
 * which operates atomically on each supernode without splitting a
 * 2×2 pivot.  `pivot_size[]` must already be populated from a prior
 * `ldlt_csc_eliminate_native` run — detection uses it to decide
 * atomicity, not to perform any numeric work.
 *
 * `super_starts[i]` and `super_sizes[i]` describe the i-th detected
 * supernode (columns `[starts[i], starts[i] + sizes[i])`).  Both
 * output arrays must have space for at most `F->n` entries; `*count`
 * receives the number of emitted supernodes (≤ F->n / min_size).
 *
 * @param F            Factored LdltCsc (pivot_size populated).
 * @param min_size     Minimum supernode size to emit (≥ 1).
 * @param super_starts Output (length ≥ F->n).
 * @param super_sizes  Output (length ≥ F->n).
 * @param[out] count   Number of emitted supernodes.
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_BADARG.
 */
sparse_err_t ldlt_csc_detect_supernodes(const LdltCsc *F, idx_t min_size, idx_t *super_starts,
                                        idx_t *super_sizes, idx_t *count);

/* ═══════════════════════════════════════════════════════════════════════
 * Conversion: linked-list SparseMatrix ↔ LdltCsc
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Convert a SparseMatrix into an LdltCsc for LDL^T numeric factorization.
 *
 * Delegates to `chol_csc_from_sparse` for the L factor (lower triangle
 * including diagonal, permuted via `perm`, capacity sized by
 * `fill_factor`).  Allocates and zeroes `D`, `D_offdiag`, and initialises
 * `pivot_size[i] = 1` and `perm[i] = perm_in[i]` (or identity when
 * `perm_in == NULL`).  Caches `factor_norm = ||A||_inf`.
 *
 * The factorization itself (Bunch-Kaufman 1x1/2x2 pivots) runs in
 * Day 8's `ldlt_csc_eliminate`; Day 7 just scaffolds storage.
 *
 * @param mat           Input matrix (not modified).  Must be square.
 * @param perm_in       Optional symmetric fill-reducing permutation
 *                      (`perm_in[new] = old`).  NULL → identity.
 * @param fill_factor   Capacity multiplier for the embedded L (clamped to [1, 20]).
 * @param[out] ldlt_out Receives the allocated LdltCsc*.  Caller frees with
 *                      `ldlt_csc_free`.  NULL on error.
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_SHAPE,
 *         SPARSE_ERR_NOT_SPD (non-symmetric `mat` — LDL^T requires
 *         symmetry, matching `sparse_ldlt_factor`'s contract),
 *         SPARSE_ERR_BADARG (invalid perm), SPARSE_ERR_ALLOC.
 */
sparse_err_t ldlt_csc_from_sparse(const SparseMatrix *mat, const idx_t *perm_in, double fill_factor,
                                  LdltCsc **ldlt_out);

/**
 * Convert the L factor of an LdltCsc back to a linked-list SparseMatrix.
 *
 * Mirrors `chol_csc_to_sparse` — writes only the lower-triangle entries
 * stored in `ldlt->L`.  D / D_offdiag / pivot_size are *not* embedded in
 * the result; the output is strictly the lower-triangle sparse matrix
 * (identity permutations on the returned matrix).  Apply `perm_out` to
 * un-permute back into the user's coordinate system.
 *
 * This helper exists mostly for round-trip tests; once Day 8 runs the
 * full Bunch-Kaufman kernel, a separate path will be needed to write
 * back into a `sparse_ldlt_t` for interop with the existing public API.
 *
 * @param ldlt         Input LdltCsc (not modified).
 * @param perm_out     Optional symmetric permutation (same convention as
 *                     `perm_in` above).  NULL → identity.
 * @param[out] mat_out Receives the allocated SparseMatrix*.
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_ALLOC, SPARSE_ERR_BADARG.
 */
sparse_err_t ldlt_csc_to_sparse(const LdltCsc *ldlt, const idx_t *perm_out, SparseMatrix **mat_out);

/* ═══════════════════════════════════════════════════════════════════════
 * Invariant checking
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Structural sanity check on an LdltCsc.
 *
 * Delegates to `chol_csc_validate` for the embedded L, and verifies:
 *   - `D`, `D_offdiag`, `pivot_size`, `perm` are non-NULL when n > 0.
 *   - Every `pivot_size[i]` is 1 or 2.
 *   - 2x2 pivots cover two consecutive indices i, i+1 (both set to 2).
 *   - `perm[0..n-1]` is a valid permutation of [0, n).
 *
 * @return SPARSE_OK when invariants hold; otherwise SPARSE_ERR_NULL,
 *         SPARSE_ERR_BADARG, or SPARSE_ERR_ALLOC when the internal
 *         `seen` workspace used for the permutation-validity check
 *         cannot be allocated.
 */
sparse_err_t ldlt_csc_validate(const LdltCsc *ldlt);

/* ═══════════════════════════════════════════════════════════════════════
 * Bunch-Kaufman elimination — two kernels share one entry point
 * ═══════════════════════════════════════════════════════════════════════
 *
 * `ldlt_csc_eliminate` is the stable public entry point.  Under the
 * hood it dispatches to one of two implementations:
 *
 *   - **Wrapper** (Sprint 17 Day 8).  Expands the CSC lower triangle
 *     to a full symmetric `SparseMatrix`, calls `sparse_ldlt_factor`,
 *     and unpacks the result.  Correct by construction; slow because
 *     the factor body runs through the linked-list kernel.
 *
 *   - **Native** (Sprint 18).  Column-by-column Bunch-Kaufman directly
 *     on packed CSC storage.  Target: bit-identical output vs wrapper
 *     on every test matrix, with the pointer-chasing overhead of the
 *     linked-list kernel removed.
 *
 * Dispatch is compile-time (`-DLDLT_CSC_USE_NATIVE=1` to flip the
 * default to native) and runtime-overridable via
 * `ldlt_csc_set_kernel_override` so a single test binary can exercise
 * either path on demand during Sprint 18's migration.
 */

/** @name Kernel selection for ldlt_csc_eliminate.
 *
 * Compile-time default controlled by LDLT_CSC_USE_NATIVE (0 = wrapper,
 * 1 = native).  Sprint 18 Day 5 flipped this to native-by-default after
 * the full test corpus (including a 20+ random-matrix cross-check
 * against `sparse_ldlt_factor`) passes bit-identically on both paths.
 * The wrapper stays compiled so tests and benchmarks can exercise it
 * via the runtime override, but no production call site selects it.
 */
#ifndef LDLT_CSC_USE_NATIVE
#define LDLT_CSC_USE_NATIVE 1
#endif

/** Kernel-selection override codes for `ldlt_csc_set_kernel_override`. */
typedef enum {
    LDLT_CSC_KERNEL_DEFAULT = 0, /**< Use the compile-time default (LDLT_CSC_USE_NATIVE). */
    LDLT_CSC_KERNEL_WRAPPER = 1, /**< Force the Sprint 17 wrapper path. */
    LDLT_CSC_KERNEL_NATIVE = 2,  /**< Force the Sprint 18 native kernel. */
} LdltCscKernelOverride;

/**
 * Override the kernel selected by `ldlt_csc_eliminate` for the current
 * process.  Intended for tests and benchmarks that need to exercise
 * both paths on the same binary during the Sprint 18 migration.
 *
 * Thread-safety: not thread-safe.  Callers are expected to set once
 * at test setup and clear at teardown.  Production call sites should
 * never call this — they get whichever kernel the compile-time flag
 * selects.
 *
 * @param mode  One of LDLT_CSC_KERNEL_{DEFAULT, WRAPPER, NATIVE}.
 */
void ldlt_csc_set_kernel_override(LdltCscKernelOverride mode);

/**
 * Read the current kernel override, for diagnostics.  Returns
 * LDLT_CSC_KERNEL_DEFAULT when no override is active.
 */
LdltCscKernelOverride ldlt_csc_get_kernel_override(void);

/**
 * Factor the LdltCsc via Bunch-Kaufman pivoting.  Dispatches to the
 * wrapper or native kernel based on the compile-time default and any
 * runtime override.
 *
 * On entry `F->L` is expected to hold the lower triangle of the (optionally
 * permuted) input A — exactly what `ldlt_csc_from_sparse` produces.
 * On successful return:
 *   - `F->L` is the unit lower-triangular factor (diagonal stored as 1.0),
 *   - `F->D` / `F->D_offdiag` hold the D block-diagonal entries,
 *   - `F->pivot_size[k] ∈ {1, 2}` identifies each pivot's block size
 *     (`pivot_size[k] == pivot_size[k+1] == 2` for a 2x2 block),
 *   - `F->perm` is the composed permutation
 *     `perm_factored[k] = perm_fill_reducing[perm_BK[k]]`,
 *   - `F->factor_norm` is ||A||_inf at factorization time.
 *
 * @param F  Input/output LdltCsc.  Must have been set up by
 *           `ldlt_csc_from_sparse` (or the equivalent `ldlt_csc_alloc`
 *           plus manual population).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if F is NULL.
 * @return SPARSE_ERR_ALLOC on allocation failure.
 * @return SPARSE_ERR_NOT_SPD if A is not symmetric (mirror insertion
 *         during expansion should keep symmetry, but the linked-list
 *         check is applied defensively).
 * @return SPARSE_ERR_SINGULAR if a pivot is numerically singular
 *         (near-zero 1x1 diagonal, or 2x2 block with near-zero
 *         determinant, or L-entry magnitude exceeding the growth bound).
 */
sparse_err_t ldlt_csc_eliminate(LdltCsc *F);

/** Sprint 17 wrapper path — exposed for tests/benchmarks via override. */
sparse_err_t ldlt_csc_eliminate_wrapper(LdltCsc *F);

/** Sprint 18 native path — exposed for tests/benchmarks via override. */
sparse_err_t ldlt_csc_eliminate_native(LdltCsc *F);

/* ═══════════════════════════════════════════════════════════════════════
 * Native-kernel workspace (Sprint 18)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Bunch-Kaufman needs two dense column accumulators per step: one for
 * the current column k, and (conditionally) one for the 2×2 pivot-
 * candidate partner column r.  Each accumulator carries a scatter-
 * gather triple (dense buffer + pattern list + per-row marker) so a
 * single cmod pass over the prior factored columns touches only the
 * rows that end up non-zero.  This mirrors the `CholCscWorkspace`
 * layout from Sprint 17 Day 4, duplicated for the partner column.
 */
typedef struct {
    idx_t n;                /**< Matrix dimension (matches the LdltCsc being factored). */
    double *dense_col;      /**< Length n, accumulator for the current column k. */
    idx_t *dense_pattern;   /**< Length n, touched-row list for column k. */
    int8_t *dense_marker;   /**< Length n, 1 iff row is in `dense_pattern`. */
    idx_t pattern_count;    /**< Valid entries in `dense_pattern`. */
    double *dense_col_r;    /**< Length n, accumulator for partner column r (2×2 candidate). */
    idx_t *dense_pattern_r; /**< Length n, touched-row list for column r. */
    int8_t *dense_marker_r; /**< Length n, 1 iff row is in `dense_pattern_r`. */
    idx_t pattern_count_r;  /**< Valid entries in `dense_pattern_r`. */
} LdltCscWorkspace;

/**
 * Allocate a native-kernel workspace for a matrix of dimension n.
 *
 * All six dense arrays are zero-initialised and `pattern_count` /
 * `pattern_count_r` start at zero.  The caller frees with
 * `ldlt_csc_workspace_free`.
 *
 * @param n        Matrix dimension (n >= 0).
 * @param[out] out Receives the allocated workspace.  Set to NULL on error.
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_BADARG (n<0), or SPARSE_ERR_ALLOC.
 */
sparse_err_t ldlt_csc_workspace_alloc(idx_t n, LdltCscWorkspace **out);

/** Free a native-kernel workspace.  Safe to call with NULL. */
void ldlt_csc_workspace_free(LdltCscWorkspace *ws);

/* ═══════════════════════════════════════════════════════════════════════
 * In-place symmetric swap (Sprint 18 Day 2)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Bunch-Kaufman pivoting forms a symmetric permutation P every time it
 * picks a pivot off the diagonal.  On the symmetric matrix A this is
 * just a relabelling — every stored entry's row/column index passes
 * through σ = (i ↔ j) — but on the packed lower-triangular CSC storage
 * used by `LdltCsc->L`, the same operation touches:
 *
 *   1. Columns 0..i-1: rows i and j may appear in each; their values
 *      swap, or the single present row gets renamed and re-sorted into
 *      the column's sorted row_idx slice.
 *   2. Columns [i..j]: the whole block under rows [i..n) is affected.
 *      Middle-block entries can even move between columns (e.g., an
 *      old (r, i) with r in (i, j) becomes stored at (j, r) in the
 *      new layout because (r, j) after σ falls into the upper triangle
 *      and is symmetric-reflected back to (j, r)).  Total nnz inside
 *      the block is preserved (σ is a bijection on (row, col) pairs
 *      after lower-triangle reflection), so no CSC growth is required.
 *   3. Columns j+1..n-1: unchanged (rows >= c > j never equal i or j).
 *
 * `F->D`, `F->D_offdiag`, `F->pivot_size`, and `F->perm` are also
 * swapped at positions i and j so the LdltCsc stays internally
 * consistent with the relabelling.  Callers that mean to update perm
 * differently (e.g., composing with a fill-reducing perm) can rewrite
 * `F->perm` after the swap.
 */

/**
 * Perform a symmetric row-and-column swap on an LdltCsc in place.
 *
 * The function is self-contained: it operates on the full CSC (not just
 * a BK "active submatrix"), so callers in non-BK contexts (stress
 * tests, ad-hoc permutations) behave correctly.  When called from the
 * BK elimination loop, cols j+1..n-1 touching only cols < i is still
 * the dominant cost, matching the reference `swap_L_rows` in
 * `src/sparse_ldlt.c`.
 *
 * The operation preserves total nnz in `F->L`; no capacity growth is
 * ever needed.
 *
 * @param F  LdltCsc to mutate.  Must be non-NULL with valid storage.
 * @param i  First index to swap; must satisfy 0 <= i < F->n.
 * @param j  Second index to swap; must satisfy 0 <= j < F->n.  When
 *           i == j the call is a no-op and returns SPARSE_OK.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if F or any of its arrays is NULL.
 * @return SPARSE_ERR_BADARG if i or j is out of range.
 * @return SPARSE_ERR_ALLOC if the temporary gather buffers (sized to
 *         the swapped block's nnz) cannot be allocated.
 */
sparse_err_t ldlt_csc_symmetric_swap(LdltCsc *F, idx_t i, idx_t j);

/* ═══════════════════════════════════════════════════════════════════════
 * Day 9: Triangular + block diagonal solve on a factored LdltCsc
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Given P*A*P^T = L*D*L^T, solving A*x = b proceeds in five phases:
 *   (0) apply P to b: y[i] = b[perm[i]]
 *   (1) forward solve L*w = y (L is unit lower triangular)
 *   (2) diagonal solve D*z = w, handling 1x1 and 2x2 pivot blocks
 *         1x1: z[k] = w[k] / D[k]
 *         2x2: det = D[k]*D[k+1] - D_off[k]^2
 *              z[k]   = ( D[k+1]*w[k]   - D_off[k]*w[k+1]) / det
 *              z[k+1] = (-D_off[k]*w[k] + D[k]*w[k+1])    / det
 *   (3) backward solve L^T*v = z (walks L columns right-to-left)
 *   (4) apply P^T to v: x[perm[i]] = v[i]
 *
 * Both triangular sweeps walk the CSC column slice once per column —
 * same CSC-friendly structure as `chol_csc_solve`, but skipping the
 * division/multiplication by a non-unit diagonal (L is unit
 * triangular for LDL^T).
 *
 * Singularity detection mirrors `sparse_ldlt_solve`: `SPARSE_DROP_TOL
 * * ||A||_inf` for 1x1 pivots, block-relative `drop_tol * bscale^2`
 * for 2x2 determinants (bscale = |d11| + |d22| + |d21|).
 */

/**
 * Solve A * x = b using a factored LdltCsc.
 *
 * `b` and `x` may alias.  The solve uses an internal workspace of
 * length 2n, allocated and freed once per call.
 *
 * @param F  Factored LdltCsc (output of `ldlt_csc_eliminate`).
 * @param b  RHS in user coordinates (length F->n).
 * @param x  Solution in user coordinates (length F->n).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any pointer is NULL.
 * @return SPARSE_ERR_BADARG if F's arrays are missing.
 * @return SPARSE_ERR_ALLOC on workspace allocation failure.
 * @return SPARSE_ERR_SINGULAR if a 1x1 pivot is near zero or a 2x2
 *         block determinant is near zero, using the same relative
 *         tolerance as the linked-list `sparse_ldlt_solve`.
 */
sparse_err_t ldlt_csc_solve(const LdltCsc *F, const double *b, double *x);

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 19 Day 12: batched supernodal LDL^T — extract / writeback
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Plumbing that moves an LDL^T supernode's diagonal block + panel
 * between packed CSC (`F->L`) and a dense column-major buffer.
 * Mirrors `chol_csc_supernode_extract` / `chol_csc_supernode_writeback`
 * (Sprint 18 Days 6 / 10) but with two LDL^T-specific deltas:
 *
 *   1. The diagonal of `F->L` carries the unit 1.0 (LDL^T factor is
 *      unit lower triangular).  Writeback's per-column drop threshold
 *      therefore can't use `|L[j, j]|` as the scale the way Cholesky
 *      does (where `L[j, j] = sqrt(pivot)`).  Instead the threshold
 *      uses `|D[j]|` for 1×1 pivots and the 2×2 block magnitude
 *      `|d11| + |d22| + |d21|` for 2×2 pivots — matching the scalar
 *      LDL^T `chol_csc_gather` invocations in
 *      `ldlt_csc_eliminate_native` (`drop_tol` for 1×1,
 *      `drop_tol * bscale` for 2×2).
 *
 *   2. Writeback also distributes the dense block factor's auxiliary
 *      outputs into the LdltCsc: `D_block[j] → F->D[s_start + j]`,
 *      `D_offdiag_block[j] → F->D_offdiag[s_start + j]`,
 *      `pivot_size_block[j] → F->pivot_size[s_start + j]`.  These
 *      arrays are produced by `ldlt_dense_factor` in Day 13's
 *      `eliminate_diag` step; the writeback contract just requires the
 *      caller to pass them through verbatim.
 */

/**
 * Extract a supernode from an LdltCsc into a dense column-major buffer.
 *
 * Mirrors `chol_csc_supernode_extract` exactly, dispatched on the
 * embedded `F->L`.  The buffer layout, `row_map` semantics, and panel-
 * height contract all match the Cholesky helper — see its header for
 * the per-argument detail.  No D / D_offdiag extraction here; those
 * live separately on the LdltCsc and only flow back through writeback.
 *
 * @param F                    Input LdltCsc (not modified).
 * @param s_start              Starting column of the supernode.
 * @param s_size               Number of columns in the supernode.
 * @param dense                Column-major output buffer (>= lda * s_size).
 * @param lda                  Leading dimension; must be >= panel_height.
 * @param row_map              Output (>= panel_height entries).
 * @param[out] panel_height_out  Receives the panel height.
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_BADARG.
 */
sparse_err_t ldlt_csc_supernode_extract(const LdltCsc *F, idx_t s_start, idx_t s_size,
                                        double *dense, idx_t lda, idx_t *row_map,
                                        idx_t *panel_height_out);

/**
 * Gather a supernode's dense column-major buffer back into an LdltCsc.
 *
 * Inverse of `ldlt_csc_supernode_extract`, plus the LDL^T-specific
 * extras: per-column `D_block[j]`, `D_offdiag_block[j]`, and
 * `pivot_size_block[j]` outputs from the dense block factor get copied
 * into `F->D[s_start + j]`, `F->D_offdiag[s_start + j]`, and
 * `F->pivot_size[s_start + j]` respectively.
 *
 * The CSC's per-column `row_idx` and `col_ptr` are not changed — the
 * supernode's structural pattern is preserved end-to-end.  Below-
 * diagonal entries whose magnitude falls below the per-column drop
 * threshold are written back as exactly 0.0; the diagonal is never
 * dropped.  Per-column threshold uses `drop_tol * |D_block[j]|` for
 * 1×1 pivots and `drop_tol * (|d11| + |d22| + |d21|)` for either
 * column of a 2×2 pair.  Pass `drop_tol = 0.0` to retain every value
 * verbatim (used by the round-trip tests).
 *
 * 2×2 pair detection uses `D_offdiag_block`: a column j with
 * `pivot_size_block[j] == 2` is the FIRST of its pair iff
 * `D_offdiag_block[j] != 0`, and the SECOND otherwise.  This mirrors
 * the convention `ldlt_csc_eliminate_native` uses when populating
 * `F->D_offdiag` (set to `d21` on the first, zeroed on the second)
 * and is robust against adjacent 2×2 pairs (where `pivot_size[]`
 * alone wouldn't disambiguate).
 *
 * @param F                  LdltCsc to mutate.
 * @param s_start            Starting column of the supernode.
 * @param s_size             Number of columns in the supernode.
 * @param dense              Column-major input buffer (>= lda * s_size).
 * @param lda                Leading dimension; must be >= panel_height.
 * @param row_map            Local-row → global-row mapping (length panel_height).
 * @param panel_height       Stored-row count of the supernode's first column.
 * @param D_block            Per-column D values (length s_size).
 * @param D_offdiag_block    Per-column D off-diagonals (length s_size).
 * @param pivot_size_block   Per-column pivot sizes (length s_size).
 * @param drop_tol           Drop tolerance; 0.0 disables dropping.
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_BADARG.
 */
sparse_err_t ldlt_csc_supernode_writeback(LdltCsc *F, idx_t s_start, idx_t s_size,
                                          const double *dense, idx_t lda, const idx_t *row_map,
                                          idx_t panel_height, const double *D_block,
                                          const double *D_offdiag_block,
                                          const idx_t *pivot_size_block, double drop_tol);

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 19 Day 13: batched supernodal LDL^T — eliminate_diag /
 *                   eliminate_panel / eliminate_supernodal
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Three helpers complete the batched supernodal flow:
 *
 *   `ldlt_csc_supernode_eliminate_diag`   — apply left-looking external
 *     cmod from prior columns `[0, s_start)` (handling 1×1 and 2×2
 *     prior pivots), then run `ldlt_dense_factor` on the s_size×s_size
 *     diagonal slab.  The cmod expansion mirrors the Phase A + Phase B
 *     terms in `ldlt_csc_cmod_unified` but applied to a dense block
 *     rather than a single column.
 *
 *   `ldlt_csc_supernode_eliminate_panel`  — for each below-supernode
 *     row, two-phase solve `L_diag · D_block · L_panel_row^T = A_row^T`:
 *     (1) forward-sub against the unit lower triangular L_diag, then
 *     (2) divide by the block diagonal D_block (1×1 division for 1×1
 *     pivots, 2×2 inverse for 2×2 pivots).  The Cholesky equivalent
 *     is `chol_csc_supernode_eliminate_panel` (a single forward sub —
 *     no D step because Cholesky's L isn't unit triangular).
 *
 *   `ldlt_csc_eliminate_supernodal`       — top-level entry point that
 *     interleaves the batched path (extract / eliminate_diag /
 *     eliminate_panel / writeback) with the scalar fallback for
 *     non-supernodal columns.  Mirror of `chol_csc_eliminate_supernodal`
 *     (Sprint 18 Day 10).
 *
 * **Pivot-stability assumption.**  `ldlt_dense_factor` performs its
 * own Bunch-Kaufman pivot selection on the extracted diagonal block
 * and may symmetrically swap rows/columns within the block.  Such a
 * swap would invalidate the surrounding CSC's row indices (the panel
 * rows below and the prior columns referencing supernode rows).  The
 * batched path therefore relies on the two-pass refactor model: the
 * first factor (scalar `ldlt_csc_eliminate_native`) resolves all
 * pivot swaps and bakes the post-permutation row order into `F->perm`
 * and `F->L`'s storage; the second-pass batched factor sees a
 * pre-permuted matrix where BK chooses the same pivot pattern without
 * further swaps.  `eliminate_diag` validates this by comparing the
 * dense factor's output `pivot_size_block` against the cached
 * `F->pivot_size[s_start..]`; a mismatch returns `SPARSE_ERR_BADARG`,
 * signalling the caller to fall back to the scalar kernel.
 */

/**
 * Apply external cmod + dense LDL^T factor to a supernode's extracted buffer.
 *
 * @param F                Factored LdltCsc (cached `pivot_size` and
 *                         prior-column factor values; unchanged).
 * @param s_start          Starting column of the supernode.
 * @param s_size           Number of columns in the supernode.
 * @param dense            Column-major buffer (>= lda * s_size).  On
 *                         entry holds the supernode's extracted A
 *                         values (lower triangle of the diagonal block
 *                         plus panel).  On successful return the
 *                         top-s_size diagonal slab holds factored L
 *                         (unit lower triangular); the panel below
 *                         carries `A_panel - external_cmod` ready for
 *                         `eliminate_panel`'s solve.
 * @param lda              Leading dimension; must be >= panel_height.
 * @param row_map          Local-row → global-row map (length panel_height).
 * @param panel_height     Stored-row count of the supernode's first column.
 * @param[out] D_block            Length s_size.  Per-column D values.
 * @param[out] D_offdiag_block    Length s_size.
 * @param[out] pivot_size_block   Length s_size.
 * @param tol              Drop / singularity tolerance for the dense
 *                         factor; <=0 uses SPARSE_DROP_TOL.
 * @return SPARSE_OK on success.  SPARSE_ERR_NULL / SPARSE_ERR_BADARG
 *         on invalid args.  SPARSE_ERR_BADARG if `pivot_size_block`
 *         disagrees with `F->pivot_size[s_start..s_start+s_size)`
 *         (the dense factor made a different BK decision than the
 *         cached scalar pass — caller should fall back).
 *         SPARSE_ERR_SINGULAR on element-growth or near-zero pivot.
 */
sparse_err_t ldlt_csc_supernode_eliminate_diag(const LdltCsc *F, idx_t s_start, idx_t s_size,
                                               double *dense, idx_t lda, const idx_t *row_map,
                                               idx_t panel_height, double *D_block,
                                               double *D_offdiag_block, idx_t *pivot_size_block,
                                               double tol);

/**
 * Solve the panel rows against the factored diagonal block.
 *
 * For each below-supernode row `i` in [0, panel_rows):
 *   (1) forward-sub: y = L_diag^{-1} * panel_row_i^T  (unit lower triangular)
 *   (2) block diagonal: x = D_block^{-1} * y           (1×1 div / 2×2 solve)
 *   (3) panel_row_i ← x
 *
 * @param L_diag             Factored s_size×s_size diagonal block (unit
 *                           lower triangular; column-major; lda_diag).
 * @param D_block            Per-column D values (length s_size).
 * @param D_offdiag_block    Per-column D off-diagonals (length s_size).
 * @param pivot_size_block   Per-column pivot sizes (length s_size).
 * @param s_size             Diagonal block dimension.
 * @param lda_diag           Leading dimension of L_diag (>= s_size).
 * @param panel              Panel buffer (column-major, panel_rows × s_size, lda_panel).
 * @param lda_panel          Leading dimension of panel (>= panel_rows).
 * @param panel_rows         Number of below-supernode rows.
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_BADARG, SPARSE_ERR_SINGULAR.
 */
sparse_err_t ldlt_csc_supernode_eliminate_panel(const double *L_diag, const double *D_block,
                                                const double *D_offdiag_block,
                                                const idx_t *pivot_size_block, idx_t s_size,
                                                idx_t lda_diag, double *panel, idx_t lda_panel,
                                                idx_t panel_rows);

/**
 * Top-level batched supernodal LDL^T entry point.
 *
 * Mirror of `chol_csc_eliminate_supernodal` (Sprint 18 Day 10) for the
 * LDL^T side.  Detects 2×2-aware supernodes via
 * `ldlt_csc_detect_supernodes`, then for each detected supernode of
 * size >= 2 runs the batched path (extract / eliminate_diag /
 * eliminate_panel / writeback).  Singleton supernodes and any column
 * outside detected supernodes fall back to the scalar
 * scatter/cmod/cdiv/gather path from `ldlt_csc_eliminate_native`.
 *
 * Two-pass refactor model: requires `F->pivot_size` to be populated
 * from a prior `ldlt_csc_eliminate_native` run.  On a fresh
 * `LdltCsc` from `ldlt_csc_from_sparse` (where pivot_size defaults to
 * all-1×1) detection emits one big 1×1 supernode and the batched path
 * exercises that — so the entry point is also valid as a first
 * factor when callers know the matrix is essentially Cholesky-like.
 *
 * @param F         Input/output LdltCsc.  See contract above.
 * @param min_size  Minimum supernode size to batch (>= 1).  Below
 *                  `min_size`, columns fall through to the scalar path.
 * @return SPARSE_OK on success.  Same error codes as
 *         `ldlt_csc_eliminate_native` plus `SPARSE_ERR_BADARG` if any
 *         batched supernode's BK decision diverged from cached
 *         `F->pivot_size`.
 */
sparse_err_t ldlt_csc_eliminate_supernodal(LdltCsc *F, idx_t min_size);

#endif /* SPARSE_LDLT_CSC_INTERNAL_H */
