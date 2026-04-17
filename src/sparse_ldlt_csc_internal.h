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
 *   pivot_size ─ length n, 1 for a 1x1 pivot or 2 for the first index of
 *                a 2x2 pivot.  (Both indices of a 2x2 block have value 2,
 *                matching `sparse_ldlt_t`.)
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
 * Day 8: Bunch-Kaufman elimination
 * ═══════════════════════════════════════════════════════════════════════
 *
 * ─── Design: wrap the linked-list kernel, present a CSC interface ─────
 *
 * The linked-list LDL^T factorization in src/sparse_ldlt.c implements
 * Bunch-Kaufman 1x1/2x2 pivot selection with symmetric row-and-column
 * swaps, element-growth guarding, and drop-tolerance pruning — ~500
 * lines of delicate numerical code.  A native CSC implementation must
 * reproduce every one of those decisions to stay bit-identical with the
 * linked-list path, and the cost of getting symmetric swaps right in
 * packed CSC storage is substantial (each 2x2 swap touches both the
 * swapped column's slot and every column that references the swapped
 * row, which in CSC means a scan + memmove across potentially every
 * column to the right).
 *
 * Rather than port that algorithm wholesale — a week of work for no
 * Sprint 17 perf payoff, since the CSC numeric backend's initial
 * deliverable is correctness — Day 8 delegates: expand the embedded L's
 * lower triangle to a full symmetric `SparseMatrix`, call
 * `sparse_ldlt_factor` on it, and copy the resulting L / D / D_offdiag /
 * pivot_size / perm back into the CSC.  The perm composes with the
 * fill-reducing perm stored by Day 7's `ldlt_csc_from_sparse`.  The
 * result is numerically identical to the linked-list path (it IS the
 * linked-list path) and meets Day 8's completion criterion.
 *
 * A native CSC LDL^T kernel with element growth checking, 2x2 swap
 * handling, and supernodal dense blocks is tracked as follow-up work
 * for a future sprint (likely Sprint 18+); the current wrapper keeps
 * the interface stable so that replacement is drop-in.
 */

/**
 * Factor the LdltCsc via Bunch-Kaufman pivoting.
 *
 * On entry `F->L` is expected to hold the lower triangle of the (optionally
 * permuted) input A — exactly what Day 7's `ldlt_csc_from_sparse` produces.
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
 * Day 11: Supernodal kernels for LDL^T — deliberately not implemented
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The plan calls for a conservative integration: "disable supernodal
 * kernel when 2x2 pivot is selected; use scalar path."  Today's LDL^T
 * CSC kernel already delegates the entire factorization to the
 * linked-list `sparse_ldlt_factor` (see `ldlt_csc_eliminate`'s Day 8
 * design block), so the conservative fallback is the default —
 * there is nothing to gate or switch off.
 *
 * A future sprint that introduces a native CSC LDL^T kernel will need
 * to (a) detect supernode boundaries, (b) accumulate the supernode's
 * dense panel, (c) run a dense LDL^T factor (handling 1x1/2x2 pivots),
 * and (d) handle 2x2 pivots that cross a supernode boundary by
 * splitting the supernode.  Until then, `ldlt_csc_eliminate` remains
 * the single entry point.
 */

#endif /* SPARSE_LDLT_CSC_INTERNAL_H */
