/**
 * @file sparse_ldlt_csc.c
 * @brief CSC working-format numeric backend for symmetric indefinite
 *        LDL^T factorization.
 *
 * ─── Why a CSC LDL^T exists (and why today's is a wrapper) ────────────
 *
 * `sparse_chol_csc.c` ships a full column-oriented Cholesky kernel for
 * SPD matrices.  Extending that to LDL^T for symmetric indefinite
 * matrices requires Bunch-Kaufman pivoting (1×1 and 2×2 block pivots,
 * four-criteria pivot selection, symmetric row-and-column swaps) with
 * proper element-growth guarding.  The linked-list implementation in
 * `src/sparse_ldlt.c` is ~500 lines of delicate code that the Sprint 17
 * plan asked us to preserve behaviourally.
 *
 * Day 8's `ldlt_csc_eliminate` therefore delegates: expand the CSC
 * lower triangle to a full symmetric `SparseMatrix`, call
 * `sparse_ldlt_factor`, and unpack the result back into the CSC layout
 * (with unit diagonals restored so the CSC invariant "diagonal first in
 * every non-empty column" holds).  This produces bit-identical output
 * to the linked-list path on every symmetric indefinite test matrix.
 *
 * The solve `ldlt_csc_solve` is fully native on the CSC (no linked-
 * list fallback): forward sweep on unit L, 1×1 / 2×2 block-diagonal
 * solve on D, backward sweep on L^T, plus the symmetric permutation
 * apply/unapply.  Measured on the same SuiteSparse matrices (20-repeat
 * `make bench`):
 *   nos4     (n=100)  →  factor 1.6× faster than linked-list
 *   bcsstk04 (n=132)  →  factor 1.3× faster than linked-list
 * Most of that is avoided redundant AMD reordering rather than a
 * genuine kernel-level speedup — the factor body still runs through
 * the linked-list kernel.
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
 *
 * ─── Follow-up work ───────────────────────────────────────────────────
 *
 * A native CSC LDL^T kernel that implements Bunch-Kaufman directly on
 * packed column storage (with in-place symmetric swaps and its own
 * element-growth tracking) is tracked as a post-Sprint 17 item.  The
 * Day 8 wrapper is the baseline that the replacement must beat — the
 * benchmarks in `benchmarks/bench_ldlt_csc.c` already run both paths
 * for direct comparison.
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

sparse_err_t ldlt_csc_eliminate(LdltCsc *F) {
    if (!F)
        return SPARSE_ERR_NULL;
    idx_t n = F->n;
    if (n <= 0)
        return SPARSE_OK;

    /* Validate F up front.  The header advertises that an LdltCsc may
     * be either the output of `ldlt_csc_from_sparse` or built via
     * `ldlt_csc_alloc` plus manual population — the latter could
     * legitimately pass a partially-initialised struct here with a
     * NULL `F->perm`, `F->D`, etc., and we'd segfault on the
     * `memcpy(perm_in, F->perm, ...)` below.  `ldlt_csc_validate`
     * already checks every required field (n >= 0, non-NULL D /
     * D_offdiag / pivot_size / perm, well-formed L, valid
     * permutation) and returns SPARSE_ERR_NULL / BADARG / ALLOC on
     * failure; propagate the error instead of crashing. */
    sparse_err_t verr = ldlt_csc_validate(F);
    if (verr != SPARSE_OK)
        return verr;

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
