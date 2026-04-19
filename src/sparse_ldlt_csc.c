/**
 * @file sparse_ldlt_csc.c
 * @brief CSC working-format numeric backend for symmetric indefinite
 *        LDL^T factorization.
 *
 * ─── Native kernel ────────────────────────────────────────────────────
 *
 * `ldlt_csc_eliminate` runs column-by-column Bunch-Kaufman directly on
 * the packed CSC arrays: four-criteria pivot selection (α = (1 + √17)/8),
 * in-place symmetric row/column swaps, element-growth tracking, and
 * 1×1 / 2×2 block pivots.  The column-sweep structure mirrors
 * `chol_csc_eliminate`, with the Bunch-Kaufman scan/swap/cmod
 * replacing Cholesky's cdiv at each step.  No linked-list round-trip
 * anywhere — the kernel operates on `CholCsc` row/value arrays plus
 * the auxiliary `D`, `D_offdiag`, `pivot_size`, and `perm` slots
 * defined in `LdltCsc`.
 *
 * Factor times (see `benchmarks/bench_ldlt_csc.c` and
 * `docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md`): 2.2–2.5× faster
 * than the linked-list `sparse_ldlt_factor` on bcsstk14 (n = 1806)
 * and s3rmt3m3 (n = 5357), one-shot factor with AMD inside the timed
 * region on both sides.
 *
 * ─── Legacy wrapper path (tests/benchmarks only) ──────────────────────
 *
 * A secondary body, `ldlt_csc_eliminate_wrapper`, remains compiled in
 * so the benchmark and a handful of regression tests can A/B-test the
 * native kernel against the Sprint 17 expand-and-delegate path.  That
 * body builds a full symmetric `SparseMatrix`, calls
 * `sparse_ldlt_factor`, and unpacks the result back into CSC — bit-
 * identical output, ~2× slower on anything larger than bcsstk04.  No
 * production call site reaches it.  Selection between the two is a
 * runtime override (`ldlt_csc_set_kernel_override`); the compile-
 * time `-DLDLT_CSC_USE_NATIVE=0` fallback flips the default for
 * emergency debugging.
 *
 * ─── Storage layout ───────────────────────────────────────────────────
 *
 * `LdltCsc` mirrors `sparse_ldlt_t`'s pivot conventions so the solve
 * code paths can share idioms:
 *
 *   L          — unit lower triangular factor, stored in a `CholCsc`.
 *                Diagonal entries are stored as 1.0 (for CSC invariant
 *                uniformity); below-diagonal entries are the actual L
 *                multipliers.
 *   D          — diagonal of D, length n.  For a 1×1 pivot at step k,
 *                D[k] is the scalar.  For a 2×2 pivot at (k, k+1),
 *                D[k] and D[k+1] hold the 2×2 block's diagonal.
 *   D_offdiag  — off-diagonal of 2×2 pivots, length n.  D_offdiag[k] =
 *                D(k, k+1) = D(k+1, k) when pivot_size[k] == 2.  Zero
 *                for 1×1 pivots.
 *   pivot_size — length n.  1 for a 1×1 pivot at step k, or 2 for both
 *                indices of a 2×2 pair (pivot_size[k] == pivot_size[k+1]
 *                == 2).
 *   perm       — length n, composed symmetric permutation such that
 *                `perm[k]` maps factorization-order index k back to the
 *                user's original row/column index.  Combines any
 *                fill-reducing input permutation with the Bunch-
 *                Kaufman pivot permutation chosen during elimination.
 *
 * ─── Worked solve example ─────────────────────────────────────────────
 *
 * For P·A·P^T = L·D·L^T, solving A·x = b proceeds in five phases:
 *
 *   1. y[i] = b[perm[i]]                        (apply P to b)
 *   2. Solve L·w = y                            (forward on unit L)
 *   3. Solve D·z = w                            (block-diagonal):
 *        1×1 block at k : z[k] = w[k] / D[k]
 *        2×2 block at k : det = D[k]·D[k+1] - D_offdiag[k]²
 *                         z[k]   = ( D[k+1]·w[k]   - D_offdiag[k]·w[k+1]) / det
 *                         z[k+1] = (-D_offdiag[k]·w[k] +      D[k]·w[k+1]) / det
 *   4. Solve L^T·v = z                          (backward on unit L^T)
 *   5. x[perm[i]] = v[i]                        (apply P^T to v)
 *
 * Both triangular sweeps walk the CSC column slice once each, skipping
 * the unit diagonal (same structure as `chol_csc_solve` but without the
 * diagonal division).
 */

#include "sparse_ldlt.h"
#include "sparse_ldlt_csc_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ─── Free ───────────────────────────────────────────────────────────── */

void ldlt_csc_free(LdltCsc *m) {
    if (!m)
        return;
    chol_csc_free(m->L);
    free(m->D);
    free(m->D_offdiag);
    free(m->pivot_size);
    free(m->perm);
    free(m);
}

/* ─── Allocate ───────────────────────────────────────────────────────── */

sparse_err_t ldlt_csc_alloc(idx_t n, idx_t initial_nnz, LdltCsc **out) {
    if (!out)
        return SPARSE_ERR_NULL;
    *out = NULL;
    if (n < 0)
        return SPARSE_ERR_BADARG;

    /* Overflow guards for byte counts (n known non-negative above). */
    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    if ((size_t)n > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;

    LdltCsc *m = calloc(1, sizeof(LdltCsc));
    if (!m)
        return SPARSE_ERR_ALLOC;

    m->n = n;
    m->factor_norm = 0.0;

    sparse_err_t err = chol_csc_alloc(n, initial_nnz, &m->L);
    if (err != SPARSE_OK) {
        ldlt_csc_free(m);
        return err;
    }

    /* Always allocate at least one slot so array pointers are non-NULL
     * even for n == 0 — keeps invariant checks simple. */
    size_t alloc_n = n > 0 ? (size_t)n : 1;
    m->D = calloc(alloc_n, sizeof(double));
    m->D_offdiag = calloc(alloc_n, sizeof(double));
    m->pivot_size = calloc(alloc_n, sizeof(idx_t));
    m->perm = calloc(alloc_n, sizeof(idx_t));
    if (!m->D || !m->D_offdiag || !m->pivot_size || !m->perm) {
        ldlt_csc_free(m);
        return SPARSE_ERR_ALLOC;
    }

    /* Defaults: every step is a 1x1 pivot; perm is the identity.  Both
     * are overwritten during elimination in Day 8. */
    for (idx_t i = 0; i < n; i++) {
        m->pivot_size[i] = 1;
        m->perm[i] = i;
    }

    *out = m;
    return SPARSE_OK;
}

/* ─── Convert SparseMatrix → LdltCsc ─────────────────────────────────── */

sparse_err_t ldlt_csc_from_sparse(const SparseMatrix *mat, const idx_t *perm_in, double fill_factor,
                                  LdltCsc **ldlt_out) {
    if (!ldlt_out)
        return SPARSE_ERR_NULL;
    *ldlt_out = NULL;
    if (!mat)
        return SPARSE_ERR_NULL;
    if (mat->rows != mat->cols)
        return SPARSE_ERR_SHAPE;

    /* Reject factored / non-identity-perm matrices up front — the same
     * precondition `sparse_ldlt_factor` enforces.  Without this, the
     * `sparse_is_symmetric` check below walks physical storage while
     * `chol_csc_from_sparse` later walks logical storage via
     * inv_row_perm / inv_col_perm, so the two views could disagree on
     * a matrix with a non-identity internal permutation. */
    idx_t n = mat->rows;
    if (mat->factored)
        return SPARSE_ERR_BADARG;
    {
        const idx_t *rp = sparse_row_perm(mat);
        const idx_t *cp = sparse_col_perm(mat);
        for (idx_t i = 0; i < n; i++) {
            if ((rp && rp[i] != i) || (cp && cp[i] != i))
                return SPARSE_ERR_BADARG;
        }
    }

    /* LDL^T requires a symmetric input; reject non-symmetric A the same
     * way the linked-list `sparse_ldlt_factor` does, with the shared
     * SPARSE_ERR_NOT_SPD code (that enum also covers "not symmetric"
     * per sparse_ldlt.h's documented contract). */
    if (!sparse_is_symmetric(mat, 1e-12))
        return SPARSE_ERR_NOT_SPD;

    /* Build L via the Cholesky CSC converter; this validates perm_in and
     * caches L->factor_norm.  Errors propagate unchanged. */
    CholCsc *L = NULL;
    sparse_err_t err = chol_csc_from_sparse(mat, perm_in, fill_factor, &L);
    if (err != SPARSE_OK)
        return err;

    LdltCsc *m = calloc(1, sizeof(LdltCsc));
    if (!m) {
        chol_csc_free(L);
        return SPARSE_ERR_ALLOC;
    }
    m->n = n;
    m->L = L;
    m->factor_norm = L->factor_norm;

    size_t alloc_n = n > 0 ? (size_t)n : 1;
    m->D = calloc(alloc_n, sizeof(double));
    m->D_offdiag = calloc(alloc_n, sizeof(double));
    m->pivot_size = calloc(alloc_n, sizeof(idx_t));
    m->perm = calloc(alloc_n, sizeof(idx_t));
    if (!m->D || !m->D_offdiag || !m->pivot_size || !m->perm) {
        ldlt_csc_free(m);
        return SPARSE_ERR_ALLOC;
    }

    /* Initial pivot_size is 1 everywhere; elimination overwrites. */
    for (idx_t i = 0; i < n; i++)
        m->pivot_size[i] = 1;

    /* Initial perm: the caller-supplied fill-reducing permutation if any,
     * else identity.  Bunch-Kaufman pivoting (Day 8) composes further
     * swaps into this array. */
    if (perm_in) {
        for (idx_t i = 0; i < n; i++)
            m->perm[i] = perm_in[i];
    } else {
        for (idx_t i = 0; i < n; i++)
            m->perm[i] = i;
    }

    *ldlt_out = m;
    return SPARSE_OK;
}

/* ─── Convert LdltCsc → SparseMatrix (L lower triangle only) ─────────── */

sparse_err_t ldlt_csc_to_sparse(const LdltCsc *ldlt, const idx_t *perm_out,
                                SparseMatrix **mat_out) {
    if (!ldlt)
        return SPARSE_ERR_NULL;
    return chol_csc_to_sparse(ldlt->L, perm_out, mat_out);
}

/* ─── Invariant checker ─────────────────────────────────────────────── */

sparse_err_t ldlt_csc_validate(const LdltCsc *ldlt) {
    if (!ldlt)
        return SPARSE_ERR_NULL;
    sparse_err_t err = chol_csc_validate(ldlt->L);
    if (err != SPARSE_OK)
        return err;
    if (ldlt->n < 0)
        return SPARSE_ERR_BADARG;
    if (ldlt->n > 0) {
        if (!ldlt->D || !ldlt->D_offdiag || !ldlt->pivot_size || !ldlt->perm)
            return SPARSE_ERR_BADARG;
    }

    /* pivot_size must be 1 or 2, and 2x2 pivots must cover consecutive
     * indices (pivot_size[i] = pivot_size[i+1] = 2). */
    for (idx_t i = 0; i < ldlt->n; i++) {
        idx_t s = ldlt->pivot_size[i];
        if (s != 1 && s != 2)
            return SPARSE_ERR_BADARG;
        if (s == 2) {
            if (i + 1 >= ldlt->n)
                return SPARSE_ERR_BADARG;
            if (ldlt->pivot_size[i + 1] != 2)
                return SPARSE_ERR_BADARG;
            /* Skip the second index of the 2x2 pair — it's been checked. */
            i++;
        }
    }

    /* perm must be a permutation of [0, n): every index appears exactly
     * once.  Use a small bit vector to detect duplicates / out-of-range. */
    if (ldlt->n == 0)
        return SPARSE_OK;
    char *seen = calloc((size_t)ldlt->n, sizeof(char));
    if (!seen)
        return SPARSE_ERR_ALLOC;
    for (idx_t i = 0; i < ldlt->n; i++) {
        idx_t p = ldlt->perm[i];
        if (p < 0 || p >= ldlt->n || seen[p]) {
            free(seen);
            return SPARSE_ERR_BADARG;
        }
        seen[p] = 1;
    }
    free(seen);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 8: Bunch-Kaufman elimination via the linked-list kernel
 * ═══════════════════════════════════════════════════════════════════════ */

/* Expand the lower-triangle CSC into a full symmetric SparseMatrix —
 * every off-diagonal entry is mirrored across the diagonal so that
 * sparse_ldlt_factor's symmetry check passes. */
static sparse_err_t csc_to_full_symmetric_matrix(const CholCsc *csc, SparseMatrix **out) {
    if (!csc || !out)
        return SPARSE_ERR_NULL;
    *out = NULL;
    idx_t n = csc->n;
    if (n <= 0)
        return SPARSE_ERR_BADARG;

    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return SPARSE_ERR_ALLOC;

    for (idx_t j = 0; j < n; j++) {
        for (idx_t p = csc->col_ptr[j]; p < csc->col_ptr[j + 1]; p++) {
            idx_t i = csc->row_idx[p];
            double v = csc->values[p];
            sparse_err_t ierr = sparse_insert(A, i, j, v);
            if (ierr != SPARSE_OK) {
                sparse_free(A);
                return ierr;
            }
            if (i != j) {
                ierr = sparse_insert(A, j, i, v);
                if (ierr != SPARSE_OK) {
                    sparse_free(A);
                    return ierr;
                }
            }
        }
    }

    *out = A;
    return SPARSE_OK;
}

sparse_err_t ldlt_csc_eliminate_wrapper(LdltCsc *F) {
    if (!F)
        return SPARSE_ERR_NULL;
    idx_t n = F->n;
    if (n <= 0)
        return SPARSE_OK;

    /* Validate only the invariants elimination actually needs.  The
     * full `ldlt_csc_validate(F)` call is too strict here because it
     * delegates to `chol_csc_validate(F->L)`, which rejects non-empty
     * CSC columns that don't start with an explicit diagonal entry —
     * but the linked-list `sparse_ldlt_factor` legitimately accepts
     * A's with a structurally-missing diagonal (treats them as zero
     * and either forms a 2x2 BK pivot or returns SPARSE_ERR_SINGULAR
     * later).  So we keep the safety checks that prevent crashes on
     * partially-initialised inputs but drop the diagonal-first
     * requirement and the sorted/distinct-row-index requirement that
     * full validate imposes. */
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_NULL;
    if (F->L->n != n || !F->L->col_ptr)
        return SPARSE_ERR_BADARG;
    if (F->L->col_ptr[0] != 0)
        return SPARSE_ERR_BADARG;
    idx_t l_nnz = F->L->col_ptr[n];
    if (l_nnz < 0)
        return SPARSE_ERR_BADARG;
    /* Validate every col_ptr entry against [0, l_nnz] before any reads of
     * row_idx/values.  Checking monotonicity alone is not enough — a
     * corrupted col_ptr[j] could be negative or exceed l_nnz while still
     * satisfying col_ptr[j] <= col_ptr[j+1], which would let later loops
     * read past the row_idx/values buffers. */
    for (idx_t j = 0; j < n; j++) {
        idx_t col_start = F->L->col_ptr[j];
        idx_t col_end = F->L->col_ptr[j + 1];
        if (col_start < 0 || col_end < 0 || col_start > col_end || col_start > l_nnz ||
            col_end > l_nnz)
            return SPARSE_ERR_BADARG;
    }
    if (l_nnz > 0 && (!F->L->row_idx || !F->L->values))
        return SPARSE_ERR_NULL;
    for (idx_t p = 0; p < l_nnz; p++) {
        if (F->L->row_idx[p] < 0 || F->L->row_idx[p] >= n)
            return SPARSE_ERR_BADARG;
    }
    /* Confirm perm is a real permutation so we can memcpy into perm_in
     * without risk of a stray index later in the factor path. */
    if ((size_t)n > SIZE_MAX / sizeof(unsigned char))
        return SPARSE_ERR_ALLOC;
    unsigned char *seen = calloc((size_t)n, sizeof(unsigned char));
    if (!seen)
        return SPARSE_ERR_ALLOC;
    for (idx_t j = 0; j < n; j++) {
        idx_t pj = F->perm[j];
        if (pj < 0 || pj >= n || seen[pj]) {
            free(seen);
            return SPARSE_ERR_BADARG;
        }
        seen[pj] = 1;
    }
    free(seen);

    /* Save the pre-elimination perm (fill-reducing) so we can compose it
     * with the Bunch-Kaufman perm chosen during factorization.  Guard
     * `n * sizeof(idx_t)` against size_t overflow on 32-bit platforms
     * (or absurdly large n) before computing the byte count. */
    if ((size_t)n > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;
    size_t perm_bytes = (size_t)n * sizeof(idx_t);
    idx_t *perm_in = malloc(perm_bytes);
    if (!perm_in)
        return SPARSE_ERR_ALLOC;
    memcpy(perm_in, F->perm, perm_bytes);

    /* Expand F->L's stored lower triangle to a full symmetric matrix so
     * the linked-list factor's symmetry check passes. */
    SparseMatrix *A_work = NULL;
    sparse_err_t err = csc_to_full_symmetric_matrix(F->L, &A_work);
    if (err != SPARSE_OK) {
        free(perm_in);
        return err;
    }

    /* Run the linked-list LDL^T factorization on the expanded matrix. */
    sparse_ldlt_t ll = {0};
    err = sparse_ldlt_factor(A_work, &ll);
    sparse_free(A_work);
    if (err != SPARSE_OK) {
        free(perm_in);
        return err;
    }

    /* Copy D / D_offdiag / pivot_size verbatim.  pivot_size widens from
     * the linked-list's int to our idx_t — values are 1 or 2 so the
     * conversion is lossless. */
    for (idx_t k = 0; k < n; k++) {
        F->D[k] = ll.D[k];
        F->D_offdiag[k] = ll.D_offdiag[k];
        F->pivot_size[k] = (idx_t)ll.pivot_size[k];
    }

    /* Compose the permutation: the factorization-order index k in the
     * CSC corresponds to index perm_in[ll.perm[k]] in the user's
     * original coordinate system. */
    if (ll.perm) {
        for (idx_t k = 0; k < n; k++)
            F->perm[k] = perm_in[ll.perm[k]];
    } else {
        for (idx_t k = 0; k < n; k++)
            F->perm[k] = perm_in[k];
    }

    /* Mirror ll.factor_norm into F->factor_norm — same quantity
     * (||A||_inf), just copied for consistency with `sparse_ldlt_t`. */
    F->factor_norm = ll.factor_norm;

    /* `sparse_ldlt_factor` initialises ll.L with a full identity
     * diagonal (`sparse_insert(L, i, i, 1.0)` for every i) before the
     * Bunch-Kaufman sweep, so the CSC conversion below can rely on
     * every column already containing its unit diagonal — no extra
     * injection loop is needed here. */

    /* Replace F->L with a CSC built from ll.L.  The linked-list factor
     * is already complete — no further fill-in will be introduced — so
     * allocate the CSC at exact capacity (fill_factor = 1.0) to avoid
     * a spurious 2x over-allocation on large factors. */
    CholCsc *new_L = NULL;
    err = chol_csc_from_sparse(ll.L, NULL, 1.0, &new_L);
    if (err != SPARSE_OK) {
        sparse_ldlt_free(&ll);
        free(perm_in);
        return err;
    }
    chol_csc_free(F->L);
    F->L = new_L;

    sparse_ldlt_free(&ll);
    free(perm_in);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 2: In-place symmetric swap primitive
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Algorithm (see header comment for the why):
 *
 *   Normalise i < j.
 *
 *   Phase A (cols [0, i)):
 *     For each column c, look up rows i and j in c's sorted row_idx
 *     slice.  Four cases:
 *       - Both present: swap values in place.
 *       - Only i: rename row_idx[pos_i] = j and bubble forward.
 *       - Only j: rename row_idx[pos_j] = i and bubble backward.
 *       - Neither: no-op.
 *     Column sizes and col_ptr are unchanged.
 *
 *   Phase B (cols [i, j]):
 *     Gather every entry in the block into (row, col, value) triples,
 *     apply σ to both coordinates, reflect to lower triangle if
 *     needed, bucket by new column and insertion-sort each bucket by
 *     row.  Write the result back into the same CSC slots (total block
 *     nnz is preserved so col_ptr[j+1] stays put — only col_ptr[i+1..j]
 *     shift within the block).
 *
 *   Phase C: swap F->D[i] ↔ F->D[j], F->D_offdiag[i] ↔ F->D_offdiag[j],
 *   F->pivot_size[i] ↔ F->pivot_size[j], F->perm[i] ↔ F->perm[j].
 *
 * Cols (j, n) are untouched: row indices in column c > j are all >= c,
 * so they never equal i or j. */

sparse_err_t ldlt_csc_symmetric_swap(LdltCsc *F, idx_t i, idx_t j) {
    if (!F)
        return SPARSE_ERR_NULL;
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_NULL;
    idx_t n = F->n;
    if (i < 0 || i >= n || j < 0 || j >= n)
        return SPARSE_ERR_BADARG;
    if (i == j)
        return SPARSE_OK;
    if (i > j) {
        idx_t tmp = i;
        i = j;
        j = tmp;
    }

    CholCsc *L = F->L;
    idx_t *col_ptr = L->col_ptr;
    idx_t *row_idx = L->row_idx;
    double *values = L->values;

    /* ── Phase A: cols [0, i) — swap row i ↔ row j per column ──────── */
    for (idx_t c = 0; c < i; c++) {
        idx_t start = col_ptr[c];
        idx_t end = col_ptr[c + 1];
        idx_t pos_i = end;
        idx_t pos_j = end;
        /* Linear scan is fine: columns are typically small (O(nnz/n))
         * and the scan stops as soon as row_idx[p] > j since row_idx
         * is sorted ascending. */
        for (idx_t p = start; p < end; p++) {
            idx_t r = row_idx[p];
            if (r == i) {
                pos_i = p;
            } else if (r == j) {
                pos_j = p;
                break; /* j is the larger target; beyond this, no matches. */
            } else if (r > j) {
                break;
            }
        }
        if (pos_i < end && pos_j < end) {
            /* Both present: swap values; row_idx slots keep their order. */
            double tmp = values[pos_i];
            values[pos_i] = values[pos_j];
            values[pos_j] = tmp;
        } else if (pos_i < end) {
            /* Only i present: rename to j and bubble forward to keep
             * row_idx sorted ascending. */
            double v = values[pos_i];
            idx_t p = pos_i;
            while (p + 1 < end && row_idx[p + 1] < j) {
                row_idx[p] = row_idx[p + 1];
                values[p] = values[p + 1];
                p++;
            }
            row_idx[p] = j;
            values[p] = v;
        } else if (pos_j < end) {
            /* Only j present: rename to i and bubble backward. */
            double v = values[pos_j];
            idx_t p = pos_j;
            while (p > start && row_idx[p - 1] > i) {
                row_idx[p] = row_idx[p - 1];
                values[p] = values[p - 1];
                p--;
            }
            row_idx[p] = i;
            values[p] = v;
        }
    }

    /* ── Phase B: cols [i, j] — gather-permute-scatter ─────────────── */
    idx_t block_start = col_ptr[i];
    idx_t block_end = col_ptr[j + 1];
    idx_t block_nnz = block_end - block_start;

    if (block_nnz > 0) {
        /* Temporary buffers: (row, col, value) triples for the gathered
         * block.  Total nnz is preserved through the permutation, so the
         * same block slot holds the rebuilt content without shifting
         * cols (j, n-1]. */
        if ((size_t)block_nnz > SIZE_MAX / sizeof(idx_t))
            return SPARSE_ERR_ALLOC;
        if ((size_t)block_nnz > SIZE_MAX / sizeof(double))
            return SPARSE_ERR_ALLOC;
        idx_t block_width = j - i + 1;

        idx_t *tmp_rows = malloc((size_t)block_nnz * sizeof(idx_t));
        idx_t *tmp_cols = malloc((size_t)block_nnz * sizeof(idx_t));
        double *tmp_vals = malloc((size_t)block_nnz * sizeof(double));
        idx_t *new_col_count = calloc((size_t)block_width, sizeof(idx_t));
        if (!tmp_rows || !tmp_cols || !tmp_vals || !new_col_count) {
            free(tmp_rows);
            free(tmp_cols);
            free(tmp_vals);
            free(new_col_count);
            return SPARSE_ERR_ALLOC;
        }

        /* Gather: walk cols [i, j], apply σ to (row, col), reflect to
         * lower triangle. */
        idx_t k = 0;
        for (idx_t c = i; c <= j; c++) {
            idx_t cstart = col_ptr[c];
            idx_t cend = col_ptr[c + 1];
            for (idx_t p = cstart; p < cend; p++) {
                idx_t r = row_idx[p];
                double v = values[p];
                idx_t rn = (r == i) ? j : ((r == j) ? i : r);
                idx_t cn = (c == i) ? j : ((c == j) ? i : c);
                /* Lower-triangle reflection by symmetry of the underlying
                 * matrix: (rn, cn) and (cn, rn) hold the same value, so
                 * pick whichever is lower-triangular. */
                if (rn < cn) {
                    idx_t t = rn;
                    rn = cn;
                    cn = t;
                }
                tmp_rows[k] = rn;
                tmp_cols[k] = cn;
                tmp_vals[k] = v;
                /* Invariant: cn ∈ [i, j] after the σ/reflect above, so
                 * `cn - i` ∈ [0, block_width).  The static analyzer
                 * can't prove this through the ternary + conditional
                 * swap chain, so the bound is asserted here for
                 * documentation and the access is NOLINT-suppressed. */
                new_col_count[cn - i]++; // NOLINT(clang-analyzer-security.ArrayBound)
                k++;
            }
        }

        /* Compute the new per-column write cursors within the existing
         * block slot.  Block boundaries (col_ptr[i], col_ptr[j+1]) do
         * not move; only col_ptr[i+1..j] shift within the block. */
        idx_t cursor = block_start;
        idx_t *col_write = new_col_count; /* reuse: turn into write cursors */
        for (idx_t c = 0; c < block_width; c++) {
            idx_t count = new_col_count[c];
            col_ptr[i + c] = cursor;
            col_write[c] = cursor;
            cursor += count;
        }
        /* col_ptr[j + 1] already equals block_end; leave it. */

        /* Scatter: bucket each triple by new column.  `tmp_cols[t]` is
         * written for every t in [0, block_nnz) during the gather loop
         * above (k ends at exactly block_nnz), and `tmp_cols[t]` is
         * always in [i, j] by the σ/reflect invariant — so `c_local` is
         * always a valid index into col_write[0..block_width).  NOLINT
         * suppresses a false positive where the analyser can't see the
         * 1:1 correspondence between the gather and scatter loops. */
        for (idx_t t = 0; t < block_nnz; t++) {
            idx_t c_local =
                tmp_cols[t] - i; // NOLINT(clang-analyzer-core.UndefinedBinaryOperatorResult)
            idx_t pos = col_write[c_local]++;
            row_idx[pos] = tmp_rows[t];
            values[pos] = tmp_vals[t];
        }

        /* Sort each rebuilt column's slot ascending by row.  Insertion
         * sort matches `sort_column_entries` in sparse_chol_csc.c —
         * columns are typically small and nearly sorted after scatter. */
        for (idx_t c = i; c <= j; c++) {
            idx_t cstart = col_ptr[c];
            idx_t cend = (c + 1 <= j) ? col_ptr[c + 1] : block_end;
            for (idx_t p = cstart + 1; p < cend; p++) {
                idx_t key_row = row_idx[p];
                double key_val = values[p];
                idx_t q = p;
                while (q > cstart && row_idx[q - 1] > key_row) {
                    row_idx[q] = row_idx[q - 1];
                    values[q] = values[q - 1];
                    q--;
                }
                row_idx[q] = key_row;
                values[q] = key_val;
            }
        }

        free(tmp_rows);
        free(tmp_cols);
        free(tmp_vals);
        free(new_col_count); /* was aliased as col_write — same buffer. */
    }

    /* ── Phase C: swap auxiliary arrays at positions i and j ──────── */
    {
        double tmp = F->D[i];
        F->D[i] = F->D[j];
        F->D[j] = tmp;
    }
    {
        double tmp = F->D_offdiag[i];
        F->D_offdiag[i] = F->D_offdiag[j];
        F->D_offdiag[j] = tmp;
    }
    {
        idx_t tmp = F->pivot_size[i];
        F->pivot_size[i] = F->pivot_size[j];
        F->pivot_size[j] = tmp;
    }
    {
        idx_t tmp = F->perm[i];
        F->perm[i] = F->perm[j];
        F->perm[j] = tmp;
    }

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18: Native CSC Bunch-Kaufman — scaffolding
 * ═══════════════════════════════════════════════════════════════════════ */

/* ─── Kernel selection & runtime override ──────────────────────────── */

/* Process-scope override.  Default = 0 (LDLT_CSC_KERNEL_DEFAULT) means
 * "use the compile-time default from LDLT_CSC_USE_NATIVE".  Tests and
 * benchmarks set this via `ldlt_csc_set_kernel_override` to exercise a
 * specific path on the current binary. */
static LdltCscKernelOverride g_ldlt_csc_kernel_override = LDLT_CSC_KERNEL_DEFAULT;

void ldlt_csc_set_kernel_override(LdltCscKernelOverride mode) { g_ldlt_csc_kernel_override = mode; }

LdltCscKernelOverride ldlt_csc_get_kernel_override(void) { return g_ldlt_csc_kernel_override; }

sparse_err_t ldlt_csc_eliminate(LdltCsc *F) {
    /* Resolve any DEFAULT override to the compile-time-selected kernel.
     * Writing it as an if + early return (rather than a switch with a
     * separate DEFAULT case) avoids a bugprone-branch-clone lint when
     * LDLT_CSC_USE_NATIVE == 1 makes the DEFAULT and NATIVE bodies
     * identical. */
    LdltCscKernelOverride mode = g_ldlt_csc_kernel_override;
    if (mode == LDLT_CSC_KERNEL_DEFAULT) {
#if LDLT_CSC_USE_NATIVE
        mode = LDLT_CSC_KERNEL_NATIVE;
#else
        mode = LDLT_CSC_KERNEL_WRAPPER;
#endif
    }
    if (mode == LDLT_CSC_KERNEL_NATIVE)
        return ldlt_csc_eliminate_native(F);
    return ldlt_csc_eliminate_wrapper(F);
}

/* ─── Native-kernel workspace lifecycle ────────────────────────────── */

void ldlt_csc_workspace_free(LdltCscWorkspace *ws) {
    if (!ws)
        return;
    free(ws->dense_col);
    free(ws->dense_pattern);
    free(ws->dense_marker);
    free(ws->dense_col_r);
    free(ws->dense_pattern_r);
    free(ws->dense_marker_r);
    free(ws);
}

sparse_err_t ldlt_csc_workspace_alloc(idx_t n, LdltCscWorkspace **out) {
    if (!out)
        return SPARSE_ERR_NULL;
    *out = NULL;
    if (n < 0)
        return SPARSE_ERR_BADARG;

    /* Overflow guards: six length-n arrays. */
    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    if ((size_t)n > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;

    LdltCscWorkspace *ws = calloc(1, sizeof(LdltCscWorkspace));
    if (!ws)
        return SPARSE_ERR_ALLOC;
    ws->n = n;

    /* Allocate at least 1 slot so later subscripts never trip over a
     * null pointer even for n == 0 — matches `chol_csc_workspace_alloc`. */
    size_t alloc_n = n > 0 ? (size_t)n : 1;
    ws->dense_col = calloc(alloc_n, sizeof(double));
    ws->dense_pattern = calloc(alloc_n, sizeof(idx_t));
    ws->dense_marker = calloc(alloc_n, sizeof(int8_t));
    ws->dense_col_r = calloc(alloc_n, sizeof(double));
    ws->dense_pattern_r = calloc(alloc_n, sizeof(idx_t));
    ws->dense_marker_r = calloc(alloc_n, sizeof(int8_t));
    if (!ws->dense_col || !ws->dense_pattern || !ws->dense_marker || !ws->dense_col_r ||
        !ws->dense_pattern_r || !ws->dense_marker_r) {
        ldlt_csc_workspace_free(ws);
        return SPARSE_ERR_ALLOC;
    }

    *out = ws;
    return SPARSE_OK;
}

/* ─── Ported BK pivot scanner (phase 1) ────────────────────────────── */

/* Bunch-Kaufman alpha = (1 + sqrt(17)) / 8 ≈ 0.6404.  Computed once at
 * first call rather than baking in a double literal — matches
 * `sparse_ldlt.c`'s runtime computation, so any future sqrt-precision
 * tweak applies to both kernels. */
static double ldlt_csc_bk_alpha(void) { return (1.0 + sqrt(17.0)) / 8.0; }

/* Scan the dense column accumulator for the largest off-diagonal
 * magnitude at rows i > k.  Returns the magnitude (0.0 if none) and
 * writes the row index of that off-diagonal into *r_out (k when the
 * column has no below-diagonal fill, matching `sparse_ldlt.c`'s
 * sentinel convention).
 *
 * Ported from the inline block at src/sparse_ldlt.c:498-507.  Day 3
 * wires this into `ldlt_csc_eliminate_native`'s column loop.
 */
static double ldlt_csc_bk_scan_offdiag(const double *dense_col, const idx_t *pattern,
                                       idx_t pattern_count, idx_t k, idx_t *r_out) {
    double max_offdiag = 0.0;
    idx_t r = k;
    for (idx_t t = 0; t < pattern_count; t++) {
        idx_t i = pattern[t];
        if (i > k) {
            double mag = fabs(dense_col[i]);
            if (mag > max_offdiag) {
                max_offdiag = mag;
                r = i;
            }
        }
    }
    *r_out = r;
    return max_offdiag;
}

/* ─── Scatter + cmod helpers (Sprint 18 Day 3) ──────────────────── */

/* Scatter the symmetric column `col` at step k into a dense accumulator.
 *
 * At step `step_k`, F->L stores factored L (with unit diagonal) in
 * columns [0, step_k) and A's lower triangle in columns [step_k, n).
 * Scattering "column col" in the symmetric sense means picking up:
 *   - lower-tri stored entries of column col with row >= step_k
 *     (iterate col_ptr[col]..col_ptr[col+1]); and
 *   - reflected upper-tri entries at rows in [step_k, col) — by
 *     symmetry A[col, c] == A[c, col], so for each column c in
 *     [step_k, col) we binary-search for row `col` in c's slice and
 *     place the hit into dense[c].
 *
 * The upper-tri loop is empty when `col == step_k` (primary pivot
 * column k) and non-empty when `col > step_k` (BK phase-2 partner r).
 */
static void ldlt_csc_scatter_symmetric(const CholCsc *L, idx_t col, idx_t step_k, double *dense,
                                       idx_t *pattern, int8_t *marker, idx_t *pattern_count) {
    idx_t cstart = L->col_ptr[col];
    idx_t cend = L->col_ptr[col + 1];
    for (idx_t p = cstart; p < cend; p++) {
        idx_t i = L->row_idx[p];
        if (i < step_k)
            continue;
        dense[i] = L->values[p];
        if (!marker[i]) {
            marker[i] = 1;
            pattern[(*pattern_count)++] = i;
        }
    }
    for (idx_t c = step_k; c < col; c++) {
        idx_t start = L->col_ptr[c];
        idx_t end = L->col_ptr[c + 1];
        idx_t lo = start;
        idx_t hi = end;
        while (lo < hi) {
            idx_t mid = lo + (hi - lo) / 2;
            if (L->row_idx[mid] < col) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo < end && L->row_idx[lo] == col) {
            dense[c] = L->values[lo];
            if (!marker[c]) {
                marker[c] = 1;
                pattern[(*pattern_count)++] = c;
            }
        }
    }
}

/* Binary-search for `target` in the sorted row_idx slice
 * [cstart, cend) of a CSC column.  Returns the L value at that row, or
 * 0.0 if not present. */
static double ldlt_csc_lookup_Lrc(const CholCsc *L, idx_t cstart, idx_t cend, idx_t target) {
    idx_t lo = cstart;
    idx_t hi = cend;
    while (lo < hi) {
        idx_t mid = lo + (hi - lo) / 2;
        if (L->row_idx[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < cend && L->row_idx[lo] == target) {
        return L->values[lo];
    }
    return 0.0;
}

/* Apply cmod contributions from prior factored columns kp in [0, step_k)
 * to the dense accumulator for column `col`.
 *
 * Two passes:
 *
 *   Phase A: for every prior column kp (whether 1×1 or part of a 2×2
 *   block), subtract `L[i, kp] * D[kp] * L[col, kp]` from dense[i] for
 *   every stored row i >= step_k in column kp.  D[kp] here is the
 *   per-column diagonal entry (the block-diagonal element for 2×2
 *   priors, NOT the 2×2 inverse).
 *
 *   Phase B: for every 2×2 block pivot pair (kp, kp+1) with kp < step_k,
 *   add the off-diagonal cross-term
 *     `L[i, kp] * D_off[kp] * L[col, kp+1]
 *      + L[i, kp+1] * D_off[kp] * L[col, kp]`
 *   to the subtraction, matching the expansion of
 *   `L[i, kp:kp+1] · D_block · L[col, kp:kp+1]^T`.
 *
 * Phase A + Phase B together reproduce the reference acc_schur_col
 * semantics in sparse_ldlt.c exactly.  New rows touched by either pass
 * get appended to the pattern via the marker check. */
static void ldlt_csc_cmod_unified(const LdltCsc *F, idx_t col, idx_t step_k, double *dense,
                                  idx_t *pattern, int8_t *marker, idx_t *pattern_count) {
    const CholCsc *L = F->L;

    /* Phase A: per-column diagonal contribution. */
    for (idx_t kp = 0; kp < step_k; kp++) {
        idx_t cstart = L->col_ptr[kp];
        idx_t cend = L->col_ptr[kp + 1];
        double L_col_kp = ldlt_csc_lookup_Lrc(L, cstart, cend, col);
        if (L_col_kp == 0.0)
            continue;
        double factor = F->D[kp] * L_col_kp;
        for (idx_t p = cstart; p < cend; p++) {
            idx_t i = L->row_idx[p];
            if (i < step_k)
                continue;
            if (!marker[i]) {
                marker[i] = 1;
                /* pattern is allocated with n slots and marker-gated
                 * uniqueness bounds *pattern_count by n throughout the
                 * elimination; analyzer can't follow this across the
                 * helper boundary so the access is NOLINT-suppressed. */
                pattern[(*pattern_count)++] = i; // NOLINT(clang-analyzer-security.ArrayBound)
            }
            dense[i] -= L->values[p] * factor;
        }
    }

    /* Phase B: cross-term correction for 2×2 block priors.  Mirrors the
     * reference's acc_schur_col cross-term block (sparse_ldlt.c:285+) —
     * iterate the two block columns whenever d_off is non-zero, even
     * when one of the L_col values is zero, so the pattern grows with
     * every row that will also need the diagonal-contribution bookkeeping.
     * Skipping those iterations silently drops pattern entries that
     * end up needing non-zero Schur values through subsequent phases. */
    idx_t kp = 0;
    while (kp + 1 < step_k) {
        if (F->pivot_size[kp] != 2) {
            kp += 1;
            continue;
        }
        double d_off = F->D_offdiag[kp];
        if (d_off == 0.0) {
            kp += 2;
            continue;
        }
        idx_t cstart_a = L->col_ptr[kp];
        idx_t cend_a = L->col_ptr[kp + 1];
        idx_t cstart_b = L->col_ptr[kp + 1];
        idx_t cend_b = L->col_ptr[kp + 2];
        double L_col_kp = ldlt_csc_lookup_Lrc(L, cstart_a, cend_a, col);
        double L_col_kp1 = ldlt_csc_lookup_Lrc(L, cstart_b, cend_b, col);

        double ct1 = d_off * L_col_kp1;
        double ct2 = d_off * L_col_kp;
        /* dense[i] -= L[i, kp] * ct1  (from kp column) */
        if (ct1 != 0.0) {
            for (idx_t p = cstart_a; p < cend_a; p++) {
                idx_t i = L->row_idx[p];
                if (i < step_k)
                    continue;
                if (!marker[i]) {
                    marker[i] = 1;
                    pattern[(*pattern_count)++] = i; // NOLINT(clang-analyzer-security.ArrayBound)
                }
                dense[i] -= L->values[p] * ct1;
            }
        }
        /* dense[i] -= L[i, kp+1] * ct2  (from kp+1 column) */
        if (ct2 != 0.0) {
            for (idx_t p = cstart_b; p < cend_b; p++) {
                idx_t i = L->row_idx[p];
                if (i < step_k)
                    continue;
                if (!marker[i]) {
                    marker[i] = 1;
                    pattern[(*pattern_count)++] = i; // NOLINT(clang-analyzer-security.ArrayBound)
                }
                dense[i] -= L->values[p] * ct2;
            }
        }
        kp += 2;
    }
}

/* Clear the primary accumulator's touched entries.
 *
 * pattern_count is maintained by the column loop to never exceed the
 * allocation size (ws->n); the marker-gated increments in scatter and
 * cmod ensure uniqueness.  Analyzer can't track that through the helper
 * boundary, hence the NOLINT suppression on the access. */
static void ldlt_csc_clear_dense_col(LdltCscWorkspace *ws) {
    for (idx_t t = 0; t < ws->pattern_count; t++) {
        idx_t i = ws->dense_pattern[t]; // NOLINT(clang-analyzer-security.ArrayBound)
        ws->dense_col[i] = 0.0;
        ws->dense_marker[i] = 0;
    }
    ws->pattern_count = 0;
}

/* Clear the partner accumulator's touched entries (same invariants as
 * `ldlt_csc_clear_dense_col`). */
static void ldlt_csc_clear_dense_col_r(LdltCscWorkspace *ws) {
    for (idx_t t = 0; t < ws->pattern_count_r; t++) {
        idx_t i = ws->dense_pattern_r[t]; // NOLINT(clang-analyzer-security.ArrayBound)
        ws->dense_col_r[i] = 0.0;
        ws->dense_marker_r[i] = 0;
    }
    ws->pattern_count_r = 0;
}

/* ─── Native kernel (Sprint 18 Days 3-4: full 1×1 + 2×2) ─────────── */

/* Sprint 18 Days 3-4 ship a complete Bunch-Kaufman kernel directly on
 * CSC storage: scatter + cmod per column (handling both 1×1 and 2×2
 * priors via `ldlt_csc_cmod_unified`), four-criteria pivot selection
 * with an in-place symmetric swap for criteria 3 and 4, 1×1 divide
 * or 2×2 block factor with element-growth tracking against
 * `growth_bound = 1 / (100 * DROP_TOL)` matching the linked-list
 * reference, and gather through `chol_csc_gather`.
 *
 * F->perm is updated in place via the symmetric-swap helper at each
 * BK swap, so no separate "compose with fill-reducing perm" step is
 * needed — by the end of the loop F->perm holds the composition the
 * wrapper produces via its post-factor unpack.
 */
sparse_err_t ldlt_csc_eliminate_native(LdltCsc *F) {
    if (!F)
        return SPARSE_ERR_NULL;
    idx_t n = F->n;
    if (n <= 0)
        return SPARSE_OK;

    /* Same structural input validation the wrapper performs. */
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_NULL;
    if (F->L->n != n || !F->L->col_ptr)
        return SPARSE_ERR_BADARG;
    if (F->L->col_ptr[0] != 0)
        return SPARSE_ERR_BADARG;
    idx_t l_nnz = F->L->col_ptr[n];
    if (l_nnz < 0)
        return SPARSE_ERR_BADARG;
    for (idx_t j = 0; j < n; j++) {
        idx_t col_start = F->L->col_ptr[j];
        idx_t col_end = F->L->col_ptr[j + 1];
        if (col_start < 0 || col_end < 0 || col_start > col_end || col_start > l_nnz ||
            col_end > l_nnz)
            return SPARSE_ERR_BADARG;
    }
    if (l_nnz > 0 && (!F->L->row_idx || !F->L->values))
        return SPARSE_ERR_NULL;
    for (idx_t p = 0; p < l_nnz; p++) {
        if (F->L->row_idx[p] < 0 || F->L->row_idx[p] >= n)
            return SPARSE_ERR_BADARG;
    }

    LdltCscWorkspace *ws = NULL;
    sparse_err_t err = ldlt_csc_workspace_alloc(n, &ws);
    if (err != SPARSE_OK)
        return err;

    /* Tolerances — match sparse_ldlt.c exactly so native / wrapper
     * decisions stay in lockstep on borderline matrices. */
    const double drop_tol = SPARSE_DROP_TOL;
    const double sing_tol = sparse_rel_tol(F->factor_norm, drop_tol);
    const double alpha_bk = ldlt_csc_bk_alpha();
    const double growth_bound = 1.0 / (100.0 * drop_tol);

    sparse_err_t rc = SPARSE_OK;

    idx_t k = 0;
    while (k < n) {
        /* ── Scatter + cmod for column k ────────────────────────── */
        ldlt_csc_scatter_symmetric(F->L, k, k, ws->dense_col, ws->dense_pattern, ws->dense_marker,
                                   &ws->pattern_count);
        ldlt_csc_cmod_unified(F, k, k, ws->dense_col, ws->dense_pattern, ws->dense_marker,
                              &ws->pattern_count);

        /* Ensure k is in the pattern even when A[k,k] was structurally
         * zero and no cmod contribution landed at row k — the gather
         * step writes the unit diagonal through this entry.
         *
         * pattern_count is <= n by the same marker-gated uniqueness
         * invariant used throughout this kernel; the NOLINT suppresses
         * a false positive where the analyzer can't follow that across
         * the scatter/cmod helpers. */
        if (!ws->dense_marker[k]) {
            ws->dense_marker[k] = 1;
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            ws->dense_pattern[ws->pattern_count++] = k;
        }

        /* ── Phase 1 BK scan ────────────────────────────────────── */
        idx_t r = k;
        double max_offdiag =
            ldlt_csc_bk_scan_offdiag(ws->dense_col, ws->dense_pattern, ws->pattern_count, k, &r);
        double diag_k = ws->dense_col[k];

        /* Decision flag: use_2x2 set by criterion 4 below. */
        int use_2x2 = 0;

        /* ── Phase 2 BK criteria (only when the diagonal is small) ── */
        if (max_offdiag > 0.0 && k + 1 < n && fabs(diag_k) < alpha_bk * max_offdiag) {
            /* Populate the partner accumulator for column r. */
            ldlt_csc_scatter_symmetric(F->L, r, k, ws->dense_col_r, ws->dense_pattern_r,
                                       ws->dense_marker_r, &ws->pattern_count_r);
            ldlt_csc_cmod_unified(F, r, k, ws->dense_col_r, ws->dense_pattern_r, ws->dense_marker_r,
                                  &ws->pattern_count_r);

            /* σ_r = max |col_acc_r[i]| for i >= k, i != r. */
            double sigma_r = 0.0;
            for (idx_t t = 0; t < ws->pattern_count_r; t++) {
                idx_t i = ws->dense_pattern_r[t];
                if (i >= k && i != r) {
                    double m = fabs(ws->dense_col_r[i]);
                    if (m > sigma_r)
                        sigma_r = m;
                }
            }

            if (fabs(diag_k) * sigma_r >= alpha_bk * max_offdiag * max_offdiag) {
                /* Criterion 2: 1×1 at (k,k), no swap.  Discard col r. */
                ldlt_csc_clear_dense_col_r(ws);
            } else if (fabs(ws->dense_col_r[r]) >= alpha_bk * sigma_r) {
                /* Criterion 3: 1×1 at (r,r), swap k ↔ r.
                 *
                 * The swap helper permutes F->L, F->D, F->D_offdiag,
                 * F->pivot_size, and F->perm all at positions k and r.
                 * After the swap, re-seat the primary accumulator from
                 * the partner buffer with σ row relabelling so the
                 * 1×1 apply below operates on the new column k. */
                rc = ldlt_csc_symmetric_swap(F, k, r);
                if (rc != SPARSE_OK)
                    goto cleanup;

                ldlt_csc_clear_dense_col(ws);
                for (idx_t t = 0; t < ws->pattern_count_r; t++) {
                    idx_t i = ws->dense_pattern_r[t];
                    idx_t m = (i == k) ? r : (i == r ? k : i);
                    ws->dense_col[m] = ws->dense_col_r[i];
                    if (!ws->dense_marker[m]) {
                        ws->dense_marker[m] = 1;
                        ws->dense_pattern[ws->pattern_count++] = m;
                    }
                }
                ldlt_csc_clear_dense_col_r(ws);
                diag_k = ws->dense_col[k];
                /* Ensure k is in the pattern after the relabel. */
                if (!ws->dense_marker[k]) {
                    ws->dense_marker[k] = 1;
                    ws->dense_pattern[ws->pattern_count++] = k;
                }
            } else {
                /* Criterion 4: 2×2 pivot at (k, r).  Land the block
                 * factor below, leaving dense_col and dense_col_r
                 * populated for the merge step. */
                use_2x2 = 1;
            }
        }

        if (!use_2x2) {
            /* ── 1×1 apply: singularity, divide, element growth ── */
            if (fabs(diag_k) < sing_tol) {
                rc = SPARSE_ERR_SINGULAR;
                goto cleanup;
            }
            F->D[k] = diag_k;
            F->D_offdiag[k] = 0.0;
            F->pivot_size[k] = 1;

            for (idx_t t = 0; t < ws->pattern_count; t++) {
                idx_t i = ws->dense_pattern[t];
                if (i > k) {
                    double v = ws->dense_col[i] / diag_k;
                    if (fabs(v) > growth_bound) {
                        rc = SPARSE_ERR_SINGULAR;
                        goto cleanup;
                    }
                    ws->dense_col[i] = v;
                }
            }

            /* Stored unit diagonal so the CSC "diagonal-first in each
             * non-empty column" invariant holds after gather. */
            ws->dense_col[k] = 1.0;

            /* ── Gather into F->L column k via chol_csc_gather ── */
            CholCscWorkspace view;
            view.n = ws->n;
            view.dense_col = ws->dense_col;
            view.dense_pattern = ws->dense_pattern;
            view.dense_marker = ws->dense_marker;
            view.pattern_count = ws->pattern_count;
            rc = chol_csc_gather(F->L, k, &view, drop_tol);
            ws->pattern_count = view.pattern_count;
            if (rc != SPARSE_OK)
                goto cleanup;

            ldlt_csc_clear_dense_col(ws);
            k += 1;
            continue;
        }

        /* ── 2×2 block pivot at (k, r) ─────────────────────────── */

        /* Swap r ↔ k+1 so the partner lands adjacent to the pivot.
         * Relabel both accumulators' row storage to match the new
         * index layout: dense_col's value at old row r becomes value
         * at new row k+1, and vice versa; same for dense_col_r. */
        if (r != k + 1) {
            rc = ldlt_csc_symmetric_swap(F, r, k + 1);
            if (rc != SPARSE_OK)
                goto cleanup;

            double tmp_d = ws->dense_col[r];
            ws->dense_col[r] = ws->dense_col[k + 1];
            ws->dense_col[k + 1] = tmp_d;
            int8_t tmp_m = ws->dense_marker[r];
            ws->dense_marker[r] = ws->dense_marker[k + 1];
            ws->dense_marker[k + 1] = tmp_m;
            for (idx_t t = 0; t < ws->pattern_count; t++) {
                if (ws->dense_pattern[t] == r) {
                    ws->dense_pattern[t] = k + 1;
                } else if (ws->dense_pattern[t] == k + 1) {
                    ws->dense_pattern[t] = r;
                }
            }

            tmp_d = ws->dense_col_r[r];
            ws->dense_col_r[r] = ws->dense_col_r[k + 1];
            ws->dense_col_r[k + 1] = tmp_d;
            tmp_m = ws->dense_marker_r[r];
            ws->dense_marker_r[r] = ws->dense_marker_r[k + 1];
            ws->dense_marker_r[k + 1] = tmp_m;
            for (idx_t t = 0; t < ws->pattern_count_r; t++) {
                if (ws->dense_pattern_r[t] == r) {
                    ws->dense_pattern_r[t] = k + 1;
                } else if (ws->dense_pattern_r[t] == k + 1) {
                    ws->dense_pattern_r[t] = r;
                }
            }
        }

        /* 2×2 block read-off + singularity check.
         *
         *   D_block = [[d11, d21], [d21, d22]]
         *   det = d11 * d22 - d21^2
         *   Singular when |det| < tol * bscale^2 (block-relative,
         *   matching sparse_ldlt.c so tiny Schur complements are
         *   compared against the block's own scale, not ||A||_inf).
         */
        double d11 = ws->dense_col[k];
        double d21 = ws->dense_col[k + 1];
        double d22 = ws->dense_col_r[k + 1];
        double det = d11 * d22 - d21 * d21;
        double bscale = fabs(d11) + fabs(d22) + fabs(d21);
        double det_tol = (bscale > 0.0) ? drop_tol * bscale * bscale : sing_tol * sing_tol;
        if (fabs(det) < det_tol) {
            rc = SPARSE_ERR_SINGULAR;
            goto cleanup;
        }
        double inv_det = 1.0 / det;
        double drop_2x2 = (bscale > 0.0) ? drop_tol * bscale : drop_tol;

        F->D[k] = d11;
        F->D[k + 1] = d22;
        F->D_offdiag[k] = d21;
        F->D_offdiag[k + 1] = 0.0;
        F->pivot_size[k] = 2;
        F->pivot_size[k + 1] = 2;

        /* Merge partner's pattern into the primary so the L[i,k] /
         * L[i,k+1] compute loop visits every row touched by either
         * column.  Pattern stays within capacity because each row is
         * appended at most once (marker-gated). */
        for (idx_t t = 0; t < ws->pattern_count_r; t++) {
            idx_t i = ws->dense_pattern_r[t];
            if (!ws->dense_marker[i]) {
                ws->dense_marker[i] = 1;
                ws->dense_pattern[ws->pattern_count++] = i;
            }
        }

        /* Compute L[i, k] and L[i, k+1] for i > k+1 via the 2×2
         * inverse.  Element-growth guard matches the reference's
         * dual-column check.
         *
         * The loop writes to dense_col_r[i] for every i > k+1 in the
         * merged pattern — including rows that were only in the
         * primary's pattern before the merge.  Mark those rows in
         * dense_marker_r / dense_pattern_r so the end-of-column
         * clear reaches them; otherwise the stale L[i, k+1] value
         * persists and poisons cmod at subsequent steps (where
         * dense_col_r for row i would read non-zero before the
         * scatter/cmod even starts). */
        for (idx_t t = 0; t < ws->pattern_count; t++) {
            idx_t i = ws->dense_pattern[t];
            if (i <= k + 1)
                continue;
            double s_ik = ws->dense_col[i];
            double s_ik1 = ws->dense_col_r[i];
            double l_ik = (s_ik * d22 - s_ik1 * d21) * inv_det;
            double l_ik1 = (-s_ik * d21 + s_ik1 * d11) * inv_det;
            if (fabs(l_ik) > growth_bound || fabs(l_ik1) > growth_bound) {
                rc = SPARSE_ERR_SINGULAR;
                goto cleanup;
            }
            ws->dense_col[i] = l_ik;
            ws->dense_col_r[i] = l_ik1;
            if (!ws->dense_marker_r[i]) {
                ws->dense_marker_r[i] = 1;
                ws->dense_pattern_r[ws->pattern_count_r++] = i;
            }
        }

        /* Set up col k for gather: unit diag at k, zero at k+1 (the
         * (k+1, k) cross-pivot entry is not stored — it lives in
         * D_offdiag).  drop_2x2 will filter out the k+1 zero. */
        ws->dense_col[k] = 1.0;
        ws->dense_col[k + 1] = 0.0;

        CholCscWorkspace view_k;
        view_k.n = ws->n;
        view_k.dense_col = ws->dense_col;
        view_k.dense_pattern = ws->dense_pattern;
        view_k.dense_marker = ws->dense_marker;
        view_k.pattern_count = ws->pattern_count;
        rc = chol_csc_gather(F->L, k, &view_k, drop_2x2);
        ws->pattern_count = view_k.pattern_count;
        if (rc != SPARSE_OK)
            goto cleanup;

        /* Set up col k+1 for gather: unit diag at k+1, zero at k
         * (k < k+1 is upper triangle; not stored).  Reuse the merged
         * primary pattern, since it covers every row that also has a
         * non-zero in dense_col_r. */
        ws->dense_col_r[k + 1] = 1.0;
        ws->dense_col_r[k] = 0.0;
        /* Ensure k+1 is in the primary pattern for the gather to
         * write the unit diagonal at (k+1, k+1). */
        if (!ws->dense_marker[k + 1]) {
            ws->dense_marker[k + 1] = 1;
            ws->dense_pattern[ws->pattern_count++] = k + 1;
        }
        /* Ensure k and k+1 are in the partner pattern so the end-of-
         * column clear zeroes the 1.0 / 0.0 we just wrote (otherwise
         * dense_col_r[k] and dense_col_r[k+1] linger stale for
         * subsequent steps).  NOLINT on the append: pattern_count_r is
         * bounded by n through the marker-gated uniqueness invariant,
         * but the analyzer can't prove it across helpers. */
        if (!ws->dense_marker_r[k + 1]) {
            ws->dense_marker_r[k + 1] = 1;
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            ws->dense_pattern_r[ws->pattern_count_r++] = k + 1;
        }
        if (!ws->dense_marker_r[k]) {
            ws->dense_marker_r[k] = 1;
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            ws->dense_pattern_r[ws->pattern_count_r++] = k;
        }

        CholCscWorkspace view_k1;
        view_k1.n = ws->n;
        view_k1.dense_col = ws->dense_col_r;
        view_k1.dense_pattern = ws->dense_pattern;
        view_k1.dense_marker = ws->dense_marker;
        view_k1.pattern_count = ws->pattern_count;
        rc = chol_csc_gather(F->L, k + 1, &view_k1, drop_2x2);
        ws->pattern_count = view_k1.pattern_count;
        if (rc != SPARSE_OK)
            goto cleanup;

        ldlt_csc_clear_dense_col(ws);
        ldlt_csc_clear_dense_col_r(ws);
        k += 2;
    }

cleanup:
    /* Ensure both accumulators are clean if we hit an error mid-loop
     * — subsequent calls reuse the same process-scope workspace is
     * not a concern today (we allocate per-call) but symmetry with
     * the reference's clear_acc-on-error pattern makes ASan runs
     * deterministic across failure branches. */
    ldlt_csc_clear_dense_col(ws);
    ldlt_csc_clear_dense_col_r(ws);
    ldlt_csc_workspace_free(ws);
    return rc;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 9: LDL^T solve — forward / diagonal-block / backward sweeps
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t ldlt_csc_solve(const LdltCsc *F, const double *b, double *x) {
    if (!F || !b || !x)
        return SPARSE_ERR_NULL;
    idx_t n = F->n;
    if (n == 0)
        return SPARSE_OK;
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_BADARG;

    /* Workspace: y holds the permuted RHS → forward-solved vector;
     * z receives the diagonal-solved vector and is then overwritten
     * in place by the backward sweep. */
    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    double *y = malloc((size_t)n * sizeof(double));
    double *z = malloc((size_t)n * sizeof(double));
    if (!y || !z) {
        free(y);
        free(z);
        return SPARSE_ERR_ALLOC;
    }

    /* Tolerance scaling matches sparse_ldlt_solve: 1x1 singularity
     * against ||A||_inf, 2x2 block singularity against the block's own
     * scale (|d11| + |d22| + |d21|)^2 to handle Schur complements
     * whose magnitude has drifted from the original matrix norm. */
    double solve_tol = SPARSE_DROP_TOL;
    double sing_tol = sparse_rel_tol(F->factor_norm, solve_tol);

    /* Phase 0: y[i] = b[perm[i]]  (apply P to b). */
    for (idx_t i = 0; i < n; i++)
        y[i] = b[F->perm[i]];

    /* Phase 1: Forward solve L*w = y.  L is unit lower triangular in
     * CSC: for each column j left-to-right, skip the unit diagonal
     * (first stored entry) and subtract L[i,j] * y[j] from every row
     * i > j present in column j's slot. */
    for (idx_t j = 0; j < n; j++) {
        idx_t start = F->L->col_ptr[j];
        idx_t end = F->L->col_ptr[j + 1];
        if (start == end || F->L->row_idx[start] != j) {
            /* Missing unit diagonal — Day 8's elimination guarantees it,
             * so this indicates a hand-corrupted CSC.  Fail safely. */
            free(y);
            free(z);
            return SPARSE_ERR_BADARG;
        }
        double yj = y[j];
        for (idx_t p = start + 1; p < end; p++) {
            idx_t i = F->L->row_idx[p];
            y[i] -= F->L->values[p] * yj;
        }
    }

    /* Phase 2: Diagonal solve D*z = w (w has overwritten y). */
    for (idx_t k = 0; k < n;) {
        if (F->pivot_size[k] == 1) {
            if (fabs(F->D[k]) < sing_tol) {
                free(y);
                free(z);
                return SPARSE_ERR_SINGULAR;
            }
            z[k] = y[k] / F->D[k];
            k++;
        } else {
            /* 2x2 block: [[d11, d21], [d21, d22]]; inv = 1/det * [[d22, -d21], [-d21, d11]].
             * `ldlt_csc_validate` guarantees pivot_size[k] == 2 implies
             * pivot_size[k+1] == 2, which in turn requires k+1 < n — but
             * that invariant isn't visible to clang-tidy's path analyser,
             * so we reject a malformed trailing 2x2 pivot here rather than
             * indexing past y/z. */
            if (k + 1 >= n) {
                free(y);
                free(z);
                return SPARSE_ERR_BADARG;
            }
            double d11 = F->D[k];
            double d22 = F->D[k + 1];
            double d21 = F->D_offdiag[k];
            double det = d11 * d22 - d21 * d21;
            double bscale = fabs(d11) + fabs(d22) + fabs(d21);
            double det_tol = (bscale > 0.0) ? solve_tol * bscale * bscale : sing_tol * sing_tol;
            if (fabs(det) < det_tol) {
                free(y);
                free(z);
                return SPARSE_ERR_SINGULAR;
            }
            double y_k1 = y[k + 1];
            z[k] = (d22 * y[k] - d21 * y_k1) / det;
            z[k + 1] = (d11 * y_k1 - d21 * y[k]) / det;
            k += 2;
        }
    }

    /* Phase 3: Backward solve L^T*v = z.  For each column j of L
     * right-to-left, the below-diagonal entries are exactly row j of
     * L^T; accumulate sum_{i>j} L[i,j] * z[i] and subtract from z[j]. */
    for (idx_t j = n - 1; j >= 0; j--) {
        idx_t start = F->L->col_ptr[j];
        idx_t end = F->L->col_ptr[j + 1];
        double sum = 0.0;
        for (idx_t p = start + 1; p < end; p++) {
            idx_t i = F->L->row_idx[p];
            sum += F->L->values[p] * z[i];
        }
        z[j] -= sum;
    }

    /* Phase 4: x[perm[i]] = z[i]  (apply P^T to z). */
    for (idx_t i = 0; i < n; i++)
        x[F->perm[i]] = z[i];

    free(y);
    free(z);
    return SPARSE_OK;
}
