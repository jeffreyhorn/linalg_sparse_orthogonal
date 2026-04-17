/**
 * @file sparse_chol_csc.c
 * @brief CSC working-format numeric backend for Cholesky factorization.
 *
 * ─── Why CSC over the linked-list Cholesky ────────────────────────────
 *
 * The library's original Cholesky kernel (`src/sparse_cholesky.c`)
 * factors A = L·L^T in place on the linked-list `SparseMatrix`.  Each
 * column update walks a sorted linked list of row nodes, chasing
 * pointers; each fill-in insertion allocates a new node from the slab
 * pool and re-links it into both the row and column chains.  For
 * structural matrices where L has significant fill, the pointer
 * chasing and node allocation dominate the runtime.
 *
 * This file ships the Sprint 17 answer: a *column-oriented CSC working
 * format* that mirrors Sprint 10's CSR-for-LU strategy but uses
 * column storage because Cholesky's inner loop (`cmod` / `cdiv`) is
 * fundamentally column-by-column.  Contiguous `row_idx` / `values`
 * arrays replace node-chasing with sequential loads, and the dense
 * scatter-gather workspace absorbs fill-in without per-entry
 * allocation.
 *
 * Measured on SuiteSparse SPD matrices (20-repeat `make bench` runs):
 *   nos4     (n=100, nnz=594)  →  2.6× factor speedup
 *   bcsstk04 (n=132, nnz=3648) →  3.5× factor speedup
 *
 * Residuals match the linked-list path to double-precision round-off.
 *
 * ─── cdiv / cmod walked end-to-end on a tiny example ─────────────────
 *
 * For A = [[4, 2],          L = [[2, 0],
 *          [2, 5]],              [1, 2]]
 *
 * Column 0:
 *   1. scatter column 0 of A → dense_col = {0: 4, 1: 2}
 *      pattern = [0, 1]
 *   2. cmod: no prior columns, skip.
 *   3. cdiv: L[0,0] = sqrt(dense_col[0]) = sqrt(4) = 2
 *            dense_col[1] /= 2  →  dense_col[1] = 1
 *   4. gather: write dense_col[0..1] back to column 0's CSC slot:
 *      col 0 row 0 = 2, col 0 row 1 = 1.
 *
 * Column 1:
 *   1. scatter column 1 of A → dense_col = {1: 5}
 *      pattern = [1]
 *   2. cmod from k = 0: L[1, 0] = 1 (present in column 0). The
 *      contribution is L[i, 0] · L[1, 0] for every stored row i ≥ 1
 *      of column 0. The only such i is 1, so:
 *        dense_col[1] -= L[1, 0] · L[1, 0] = 1 · 1 = 1
 *        dense_col[1] = 5 - 1 = 4
 *   3. cdiv: L[1,1] = sqrt(4) = 2.
 *   4. gather: col 1 row 1 = 2.
 *
 * The resulting CSC stores L = [[2, 0], [1, 2]], which satisfies
 * L·L^T = A exactly.
 *
 * ─── Supernodal extension (Days 10-11) ────────────────────────────────
 *
 * For well-structured SPD matrices, groups of adjacent columns often
 * share the same below-diagonal pattern.  Those *fundamental
 * supernodes* (Liu, Ng, Peyton) can be factored by a single dense
 * Cholesky on the diagonal block plus a dense triangular solve for
 * the panel below — replacing s scalar cdivs with one dense factor.
 *
 * `chol_csc_detect_supernodes` identifies supernodes directly on the
 * sorted CSC arrays (three O(nnz(column)) conditions per pair), and
 * `chol_csc_eliminate_supernodal` is the dispatch entry point.  The
 * dense primitives `chol_dense_factor` and `chol_dense_solve_lower`
 * are shipped as tested helpers; the batched supernodal factor that
 * uses them to replace the scalar kernel per supernode is follow-up
 * work (the current integration runs detection and then delegates to
 * the scalar kernel, producing bit-identical output).
 *
 * ─── Role of Sprint 14 symbolic analysis ──────────────────────────────
 *
 * `sparse_analyze` (Sprint 14) computes the exact nnz(L) via the
 * elimination tree and column counts.  `chol_csc_from_sparse_with_analysis`
 * uses that count to size the CSC's `values` / `row_idx` arrays to
 * the exact predicted capacity — so elimination's fill-in absorbs in
 * place and `chol_csc_grow` is never called during factorization
 * on well-behaved matrices.  Without analysis, the heuristic path
 * (`chol_csc_from_sparse`) uses a `fill_factor` multiplier and
 * geometric growth; still correct, just slightly less tight on
 * memory.
 */

#include "sparse_chol_csc_internal.h"
#include "sparse_matrix.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ─── Free ───────────────────────────────────────────────────────────── */

void chol_csc_free(CholCsc *m) {
    if (!m)
        return;
    free(m->col_ptr);
    free(m->row_idx);
    free(m->values);
    free(m);
}

/* ─── Allocate ───────────────────────────────────────────────────────── */

sparse_err_t chol_csc_alloc(idx_t n, idx_t initial_nnz, CholCsc **out) {
    if (!out)
        return SPARSE_ERR_NULL;
    *out = NULL;
    if (n < 0)
        return SPARSE_ERR_BADARG;

    idx_t cap = initial_nnz;
    if (cap < 1)
        cap = 1;

    /* Overflow guards for byte counts */
    if ((size_t)(n + 1) > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;
    if ((size_t)cap > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    if ((size_t)cap > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;

    CholCsc *m = calloc(1, sizeof(CholCsc));
    if (!m)
        return SPARSE_ERR_ALLOC;

    m->n = n;
    m->nnz = 0;
    m->capacity = cap;
    m->factor_norm = 0.0;

    m->col_ptr = calloc((size_t)(n + 1), sizeof(idx_t));
    /* calloc row_idx / values so fresh storage is deterministic — tools
     * (clang-tidy, UBSan) can reason about it without flagging paths
     * where a column's slot is touched before being written. */
    m->row_idx = calloc((size_t)cap, sizeof(idx_t));
    m->values = calloc((size_t)cap, sizeof(double));

    if (!m->col_ptr || !m->row_idx || !m->values) {
        chol_csc_free(m);
        return SPARSE_ERR_ALLOC;
    }

    *out = m;
    return SPARSE_OK;
}

/* ─── Grow ───────────────────────────────────────────────────────────── */

sparse_err_t chol_csc_grow(CholCsc *m, idx_t needed) {
    if (!m)
        return SPARSE_ERR_NULL;
    if (needed <= m->capacity)
        return SPARSE_OK;
    if (needed > INT32_MAX)
        return SPARSE_ERR_ALLOC;

    /* Geometric growth: at least 2× current capacity, or needed — whichever
     * is larger.  Guard idx_t overflow in the doubling. */
    idx_t new_cap;
    if (m->capacity > INT32_MAX / 2)
        new_cap = INT32_MAX;
    else
        new_cap = m->capacity * 2;
    if (new_cap < needed)
        new_cap = needed;

    if ((size_t)new_cap > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    if ((size_t)new_cap > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;

    /* Transactional growth: allocate and populate both buffers before
     * mutating any field on `m`.  If either allocation fails, `m` is
     * left exactly as the caller passed it — honouring the header
     * contract that "m is unchanged on failure". */
    idx_t *new_row = calloc((size_t)new_cap, sizeof(idx_t));
    if (!new_row)
        return SPARSE_ERR_ALLOC;
    double *new_val = calloc((size_t)new_cap, sizeof(double));
    if (!new_val) {
        free(new_row);
        return SPARSE_ERR_ALLOC;
    }

    /* Copy live entries into the fresh buffers, then commit. */
    if (m->capacity > 0) {
        memcpy(new_row, m->row_idx, (size_t)m->capacity * sizeof(idx_t));
        memcpy(new_val, m->values, (size_t)m->capacity * sizeof(double));
    }
    free(m->row_idx);
    free(m->values);

    m->row_idx = new_row;
    m->values = new_val;
    m->capacity = new_cap;
    return SPARSE_OK;
}

/* ─── Column-wise insertion sort: sort one column's row_idx/values ──── */

/* Columns are typically small (at most O(nnz/n) entries) and nearly
 * sorted, so insertion sort is the right choice here. */
static void sort_column_entries(idx_t *row_idx, double *values, idx_t start, idx_t end) {
    for (idx_t i = start + 1; i < end; i++) {
        idx_t key_row = row_idx[i];
        double key_val = values[i];
        idx_t j = i;
        while (j > start && row_idx[j - 1] > key_row) {
            row_idx[j] = row_idx[j - 1];
            values[j] = values[j - 1];
            j--;
        }
        row_idx[j] = key_row;
        values[j] = key_val;
    }
}

/* ─── Validate a user-supplied symmetric permutation array ──────────── */

/* Best-effort check: all entries in [0, n) and distinct.  Allocates
 * an internal seen[] array, so the caller must distinguish:
 *   SPARSE_OK         — perm is a valid permutation of [0, n)
 *   SPARSE_ERR_BADARG — perm contains an out-of-range or duplicate entry
 *   SPARSE_ERR_ALLOC  — the internal seen[] allocation failed */
static sparse_err_t validate_perm(const idx_t *perm, idx_t n) {
    if (n == 0)
        return SPARSE_OK;
    char *seen = calloc((size_t)n, sizeof(char));
    if (!seen)
        return SPARSE_ERR_ALLOC;
    sparse_err_t err = SPARSE_OK;
    for (idx_t i = 0; i < n; i++) {
        idx_t p = perm[i];
        if (p < 0 || p >= n || seen[p]) {
            err = SPARSE_ERR_BADARG;
            break;
        }
        seen[p] = 1;
    }
    free(seen);
    return err;
}

/* ─── Convert SparseMatrix → CholCsc (lower triangle, permuted space) ── */

/* Shared core for the two public `from_sparse` variants.  If
 * `explicit_capacity > 0`, it overrides the fill_factor-derived capacity;
 * this is how the symbolic-analysis path requests exact sizing. */
static sparse_err_t from_sparse_impl(const SparseMatrix *mat, const idx_t *perm, double fill_factor,
                                     idx_t explicit_capacity, CholCsc **csc_out) {
    if (!csc_out)
        return SPARSE_ERR_NULL;
    *csc_out = NULL;
    if (!mat)
        return SPARSE_ERR_NULL;
    if (mat->rows != mat->cols)
        return SPARSE_ERR_SHAPE;

    idx_t n = mat->rows;

    /* Clamp fill factor to the same range as lu_csr_from_sparse(). */
    if (fill_factor < 1.0)
        fill_factor = 1.0;
    if (fill_factor > 20.0)
        fill_factor = 20.0;

    /* Build inverse of external perm so we can map logical → new in O(1).
     * invperm[old] = new, i.e. invperm[perm[new]] = new. */
    idx_t *invperm = NULL;
    if (perm) {
        sparse_err_t verr = validate_perm(perm, n);
        if (verr != SPARSE_OK)
            return verr;
        invperm = malloc((size_t)n * sizeof(idx_t));
        if (!invperm)
            return SPARSE_ERR_ALLOC;
        for (idx_t new_i = 0; new_i < n; new_i++)
            invperm[perm[new_i]] = new_i;
    }

    /* ─── Pass 1: count entries per column in the NEW (permuted) space ─
     * Only entries with new_row >= new_col contribute (lower triangle). */
    idx_t *col_count = calloc((size_t)n, sizeof(idx_t));
    if (!col_count) {
        free(invperm);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t phys_i = 0; phys_i < n; phys_i++) {
        idx_t log_i = mat->inv_row_perm[phys_i];
        idx_t new_i = invperm ? invperm[log_i] : log_i;
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            idx_t new_j = invperm ? invperm[log_j] : log_j;
            if (new_i >= new_j)
                col_count[new_j]++;
            node = node->right;
        }
    }

    /* Sum: total nonzeros in the lower triangle after permutation. */
    idx_t nnz_lower = 0;
    for (idx_t j = 0; j < n; j++)
        nnz_lower += col_count[j];

    /* Compute capacity.  Symbolic path overrides with an explicit
     * capacity (exact predicted nnz(L)); heuristic path uses fill_factor. */
    idx_t cap;
    if (explicit_capacity > 0) {
        cap = explicit_capacity;
    } else {
        double cap_d = (double)nnz_lower * fill_factor;
        if (cap_d > (double)INT32_MAX)
            cap_d = (double)INT32_MAX;
        cap = (idx_t)cap_d;
    }
    if (cap < nnz_lower)
        cap = nnz_lower;
    if (cap < 1)
        cap = 1;

    /* ─── Allocate the CSC with the computed capacity. ────────────────── */
    CholCsc *csc = NULL;
    sparse_err_t err = chol_csc_alloc(n, cap, &csc);
    if (err != SPARSE_OK) {
        free(col_count);
        free(invperm);
        return err;
    }

    /* Build col_ptr via prefix sum of col_count. */
    csc->col_ptr[0] = 0;
    for (idx_t j = 0; j < n; j++)
        csc->col_ptr[j + 1] = csc->col_ptr[j] + col_count[j];
    csc->nnz = nnz_lower;

    /* Cache ||A||_inf for relative tolerance in solve paths.  Same
     * convention as LuCsr: norm is taken of the original matrix, before
     * permutation. */
    csc->factor_norm = sparse_norminf_const(mat);

    /* ─── Pass 2: scatter entries into columns. ──────────────────────── */
    idx_t *write_pos = malloc((size_t)n * sizeof(idx_t));
    if (!write_pos) {
        chol_csc_free(csc);
        free(col_count);
        free(invperm);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t j = 0; j < n; j++)
        write_pos[j] = csc->col_ptr[j];

    for (idx_t phys_i = 0; phys_i < n; phys_i++) {
        idx_t log_i = mat->inv_row_perm[phys_i];
        idx_t new_i = invperm ? invperm[log_i] : log_i;
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            idx_t new_j = invperm ? invperm[log_j] : log_j;
            if (new_i >= new_j) {
                idx_t pos = write_pos[new_j]++;
                csc->row_idx[pos] = new_i;
                csc->values[pos] = node->value;
            }
            node = node->right;
        }
    }

    free(write_pos);
    free(col_count);
    free(invperm);

    /* ─── Sort row indices ascending within each column ──────────────── */
    for (idx_t j = 0; j < n; j++)
        sort_column_entries(csc->row_idx, csc->values, csc->col_ptr[j], csc->col_ptr[j + 1]);

    *csc_out = csc;
    return SPARSE_OK;
}

/* ─── Public: heuristic (fill_factor) conversion ────────────────────── */

sparse_err_t chol_csc_from_sparse(const SparseMatrix *mat, const idx_t *perm, double fill_factor,
                                  CholCsc **csc_out) {
    return from_sparse_impl(mat, perm, fill_factor, 0, csc_out);
}

/* ─── Public: symbolic-analysis-aware conversion ────────────────────── */

sparse_err_t chol_csc_from_sparse_with_analysis(const SparseMatrix *mat,
                                                const sparse_analysis_t *analysis,
                                                CholCsc **csc_out) {
    if (!csc_out)
        return SPARSE_ERR_NULL;
    *csc_out = NULL;
    if (!mat || !analysis)
        return SPARSE_ERR_NULL;
    if (analysis->type != SPARSE_FACTOR_CHOLESKY)
        return SPARSE_ERR_BADARG;
    if (mat->rows != mat->cols)
        return SPARSE_ERR_SHAPE;
    if (mat->rows != analysis->n)
        return SPARSE_ERR_SHAPE;

    /* Exact predicted nnz(L) = sum of column counts of the symbolic L. */
    idx_t predicted = analysis->sym_L.nnz;
    if (predicted < 0)
        predicted = 0;

    return from_sparse_impl(mat, analysis->perm, 1.0, predicted, csc_out);
}

/* ─── Convert CholCsc → SparseMatrix (linked-list, lower triangle) ──── */

sparse_err_t chol_csc_to_sparse(const CholCsc *csc, const idx_t *perm, SparseMatrix **mat_out) {
    if (!mat_out)
        return SPARSE_ERR_NULL;
    *mat_out = NULL;
    if (!csc)
        return SPARSE_ERR_NULL;

    idx_t n = csc->n;

    if (perm) {
        sparse_err_t verr = validate_perm(perm, n);
        if (verr != SPARSE_OK)
            return verr;
    }

    /* sparse_create requires rows > 0 and cols > 0 — handle n == 0 by
     * returning an error; callers for n == 0 should not round-trip at all.
     * (sparse_create returns NULL for n == 0; we surface that as BADARG.) */
    if (n <= 0)
        return SPARSE_ERR_BADARG;

    SparseMatrix *mat = sparse_create(n, n);
    if (!mat)
        return SPARSE_ERR_ALLOC;

    /* For each CSC entry (new_r, new_c, value), map back to user-space
     * indices (old_r, old_c) = (perm[new_r], perm[new_c]) and insert.
     * Freshly-created matrices have identity permutations, so physical ==
     * logical and sparse_insert takes these directly. */
    for (idx_t j = 0; j < n; j++) {
        for (idx_t p = csc->col_ptr[j]; p < csc->col_ptr[j + 1]; p++) {
            idx_t new_r = csc->row_idx[p];
            double v = csc->values[p];
            idx_t old_r = perm ? perm[new_r] : new_r;
            idx_t old_c = perm ? perm[j] : j;
            sparse_err_t ierr = sparse_insert(mat, old_r, old_c, v);
            if (ierr != SPARSE_OK) {
                sparse_free(mat);
                return ierr;
            }
        }
    }

    *mat_out = mat;
    return SPARSE_OK;
}

/* ─── Invariant checker ─────────────────────────────────────────────── */

sparse_err_t chol_csc_validate(const CholCsc *csc) {
    if (!csc)
        return SPARSE_ERR_NULL;
    if (csc->n < 0)
        return SPARSE_ERR_BADARG;
    if (!csc->col_ptr)
        return SPARSE_ERR_BADARG;
    if (csc->col_ptr[0] != 0)
        return SPARSE_ERR_BADARG;
    if (csc->col_ptr[csc->n] != csc->nnz)
        return SPARSE_ERR_BADARG;

    for (idx_t j = 0; j < csc->n; j++) {
        idx_t start = csc->col_ptr[j];
        idx_t end = csc->col_ptr[j + 1];
        if (end < start)
            return SPARSE_ERR_BADARG;
        if (start > csc->nnz || end > csc->nnz)
            return SPARSE_ERR_BADARG;
        if (start == end)
            continue; /* empty column permitted (structurally zero) */

        /* First entry in a non-empty column must be the diagonal. */
        if (csc->row_idx[start] != j)
            return SPARSE_ERR_BADARG;

        for (idx_t p = start; p < end; p++) {
            idx_t r = csc->row_idx[p];
            if (r < j || r >= csc->n)
                return SPARSE_ERR_BADARG; /* lower triangular bound */
            if (p > start && r <= csc->row_idx[p - 1])
                return SPARSE_ERR_BADARG; /* sorted and distinct */
        }
    }
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 4: Elimination workspace & column kernel scaffolding
 * ═══════════════════════════════════════════════════════════════════════ */

/* ─── Workspace lifecycle ──────────────────────────────────────────── */

void chol_csc_workspace_free(CholCscWorkspace *ws) {
    if (!ws)
        return;
    free(ws->dense_col);
    free(ws->dense_pattern);
    free(ws->dense_marker);
    free(ws);
}

sparse_err_t chol_csc_workspace_alloc(idx_t n, CholCscWorkspace **out) {
    if (!out)
        return SPARSE_ERR_NULL;
    *out = NULL;
    if (n < 0)
        return SPARSE_ERR_BADARG;

    /* Overflow guards: all three arrays indexed by row, length n. */
    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    if ((size_t)n > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;

    CholCscWorkspace *ws = calloc(1, sizeof(CholCscWorkspace));
    if (!ws)
        return SPARSE_ERR_ALLOC;
    ws->n = n;
    ws->pattern_count = 0;

    /* Always allocate at least 1 so later subscripts never trip over a
     * null pointer — cheap, and keeps the struct invariant simple. */
    size_t alloc_n = n > 0 ? (size_t)n : 1;
    ws->dense_col = calloc(alloc_n, sizeof(double));
    ws->dense_pattern = calloc(alloc_n, sizeof(idx_t));
    ws->dense_marker = calloc(alloc_n, sizeof(int8_t));
    if (!ws->dense_col || !ws->dense_pattern || !ws->dense_marker) {
        chol_csc_workspace_free(ws);
        return SPARSE_ERR_ALLOC;
    }

    *out = ws;
    return SPARSE_OK;
}

/* ─── Binary search for a row index in a column's sorted row_idx slice ─ */

/* Returns the position within [start, end) where row_idx == target, or
 * end (out-of-range) when not present.  Uses a standard lower_bound-style
 * search on the sorted row_idx slice. */
static idx_t bsearch_row(const idx_t *row_idx, idx_t start, idx_t end, idx_t target) {
    idx_t lo = start;
    idx_t hi = end;
    while (lo < hi) {
        idx_t mid = lo + (hi - lo) / 2;
        if (row_idx[mid] < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return (lo < end && row_idx[lo] == target) ? lo : end;
}

/* ─── Scatter / cmod / cdiv / gather / end_column ─────────────────── */

void chol_csc_scatter(const CholCsc *csc, idx_t j, CholCscWorkspace *ws) {
    idx_t start = csc->col_ptr[j];
    idx_t end = csc->col_ptr[j + 1];
    for (idx_t p = start; p < end; p++) {
        idx_t i = csc->row_idx[p];
        ws->dense_col[i] = csc->values[p];
        if (!ws->dense_marker[i]) {
            ws->dense_marker[i] = 1;
            ws->dense_pattern[ws->pattern_count++] = i;
        }
    }
}

void chol_csc_cmod(const CholCsc *csc, idx_t j, CholCscWorkspace *ws) {
    /* Basic O(j * avg_col_depth) left-looking sweep: for each k < j,
     * look up L[j,k] by binary search in column k's sorted row_idx; if
     * present, subtract its rank-1 contribution to dense_col[i] for
     * every stored L[i,k] with i >= j.  Day 5 will replace the linear
     * scan of k with an elimination-tree-guided traversal. */
    for (idx_t k = 0; k < j; k++) {
        idx_t start = csc->col_ptr[k];
        idx_t end = csc->col_ptr[k + 1];
        if (start == end)
            continue;
        idx_t p_jk = bsearch_row(csc->row_idx, start, end, j);
        if (p_jk == end)
            continue;

        double l_jk = csc->values[p_jk];
        /* row_idx is sorted, so everything at positions p_jk..end-1 has
         * row >= j — exactly the rows that contribute to column j's
         * accumulator. */
        for (idx_t q = p_jk; q < end; q++) {
            idx_t i = csc->row_idx[q];
            if (!ws->dense_marker[i]) {
                ws->dense_marker[i] = 1;
                /* pattern_count is bounded by n: each row i is gated by
                 * the marker and added at most once. */
                idx_t slot = ws->pattern_count++;
                ws->dense_pattern[slot] = i; // NOLINT(clang-analyzer-security.ArrayBound)
            }
            ws->dense_col[i] -= csc->values[q] * l_jk;
        }
    }
}

sparse_err_t chol_csc_cdiv(CholCscWorkspace *ws, idx_t j) {
    if (!ws)
        return SPARSE_ERR_NULL;
    /* Caller guarantees 0 <= j < ws->n; dense_col is length ws->n. */
    double diag = ws->dense_col[j]; // NOLINT(clang-analyzer-security.ArrayBound)
    if (diag <= 0.0)
        return SPARSE_ERR_NOT_SPD;
    double l_jj = sqrt(diag);
    ws->dense_col[j] = l_jj;
    double inv_l_jj = 1.0 / l_jj;
    for (idx_t idx = 0; idx < ws->pattern_count; idx++) {
        idx_t i = ws->dense_pattern[idx];
        if (i > j)
            ws->dense_col[i] *= inv_l_jj;
    }
    return SPARSE_OK;
}

/* Shift columns (after_col+1)..n-1 by `delta` positions.  Updates
 * col_ptr[after_col+1..n] by `delta` and csc->nnz by `delta`.  For
 * positive delta, grows the CSC first; overlapping memmoves are safe.
 * Negative delta packs columns left (drop-tolerance shrink). */
static sparse_err_t shift_columns_right_of(CholCsc *csc, idx_t after_col, idx_t delta) {
    if (delta == 0)
        return SPARSE_OK;

    idx_t old_total = csc->nnz;
    if (delta > 0) {
        /* Explicit overflow guard: `old_total + delta` is evaluated in
         * idx_t (signed 32-bit), so overflow would be undefined behaviour
         * and could slip past chol_csc_grow's own INT32_MAX check.  Reject
         * growth requests that would overflow before calling in. */
        if (old_total > INT32_MAX - delta)
            return SPARSE_ERR_BADARG;
        sparse_err_t err = chol_csc_grow(csc, old_total + delta);
        if (err != SPARSE_OK)
            return err;
    } else if (delta < -old_total) {
        /* Defensive: the caller's bookkeeping is off — refuse rather
         * than corrupt the CSC.  Equivalent to `old_total + delta < 0`
         * but written without the addition, to avoid the signed-overflow
         * trap in the shrink direction too. */
        return SPARSE_ERR_BADARG;
    }

    idx_t src_start = csc->col_ptr[after_col + 1];
    idx_t src_len = old_total - src_start;
    if (src_len > 0) {
        /* delta >= -(col_ptr[after_col+1] - col_ptr[after_col]) is enforced by
         * the caller (cannot shrink past the left boundary of the previous
         * column), so src_start + delta >= col_ptr[after_col] >= 0. */
        memmove(&csc->row_idx[src_start + delta], // NOLINT(clang-analyzer-security.ArrayBound)
                &csc->row_idx[src_start], (size_t)src_len * sizeof(idx_t));
        memmove(&csc->values[src_start + delta], // NOLINT(clang-analyzer-security.ArrayBound)
                &csc->values[src_start], (size_t)src_len * sizeof(double));
    }

    for (idx_t k = after_col + 1; k <= csc->n; k++)
        csc->col_ptr[k] += delta;
    csc->nnz += delta;
    return SPARSE_OK;
}

/* Ascending comparator for idx_t, used by qsort on dense_pattern. */
static int idx_t_cmp(const void *a, const void *b) {
    idx_t ia = *(const idx_t *)a;
    idx_t ib = *(const idx_t *)b;
    return (ia > ib) - (ia < ib);
}

sparse_err_t chol_csc_gather(CholCsc *csc, idx_t j, CholCscWorkspace *ws, double drop_tol) {
    /* Sort the pattern ascending.  All rows are >= j (scatter and cmod
     * only touch rows in the lower triangle), so after sorting the
     * diagonal j sits first — satisfying the CSC invariant that the
     * diagonal is the first entry in each non-empty column. */
    if (ws->pattern_count > 1)
        qsort(ws->dense_pattern, (size_t)ws->pattern_count, sizeof(idx_t), idx_t_cmp);

    /* Drop threshold relative to the just-computed diagonal magnitude.
     * The diagonal itself is never dropped; sparse_cholesky.c uses the
     * same `SPARSE_DROP_TOL * |L[j,j]|` strategy. */
    double abs_l_jj = fabs(ws->dense_col[j]);
    double threshold = drop_tol * abs_l_jj;

    /* Count survivors so we can resize the column slot in one shot. */
    idx_t keep = 0;
    for (idx_t idx = 0; idx < ws->pattern_count; idx++) {
        idx_t i = ws->dense_pattern[idx];
        if (i == j || fabs(ws->dense_col[i]) >= threshold)
            keep++;
    }

    /* Resize column j's slot: shift later columns to accommodate `keep`
     * entries where the slot currently holds (col_ptr[j+1] - col_ptr[j])
     * entries.  delta > 0 grows, delta < 0 packs. */
    idx_t old_size = csc->col_ptr[j + 1] - csc->col_ptr[j];
    idx_t delta = keep - old_size;
    sparse_err_t err = shift_columns_right_of(csc, j, delta);
    if (err != SPARSE_OK)
        return err;

    /* Write the surviving entries in sorted order into the fresh slot. */
    idx_t p = csc->col_ptr[j];
    for (idx_t idx = 0; idx < ws->pattern_count; idx++) {
        idx_t i = ws->dense_pattern[idx];
        double v = ws->dense_col[i];
        if (i == j || fabs(v) >= threshold) {
            csc->row_idx[p] = i;
            csc->values[p] = v;
            p++;
        }
    }
    return SPARSE_OK;
}

void chol_csc_end_column(CholCscWorkspace *ws) {
    for (idx_t idx = 0; idx < ws->pattern_count; idx++) {
        idx_t i = ws->dense_pattern[idx];
        ws->dense_col[i] = 0.0;
        ws->dense_marker[i] = 0;
    }
    ws->pattern_count = 0;
}

/* ─── Orchestrator ─────────────────────────────────────────────────── */

sparse_err_t chol_csc_eliminate(CholCsc *csc) {
    if (!csc)
        return SPARSE_ERR_NULL;

    CholCscWorkspace *ws = NULL;
    sparse_err_t err = chol_csc_workspace_alloc(csc->n, &ws);
    if (err != SPARSE_OK)
        return err;

    for (idx_t j = 0; j < csc->n; j++) {
        chol_csc_scatter(csc, j, ws);
        chol_csc_cmod(csc, j, ws);
        err = chol_csc_cdiv(ws, j);
        if (err != SPARSE_OK) {
            chol_csc_end_column(ws);
            chol_csc_workspace_free(ws);
            return err;
        }
        err = chol_csc_gather(csc, j, ws, SPARSE_DROP_TOL);
        if (err != SPARSE_OK) {
            chol_csc_end_column(ws);
            chol_csc_workspace_free(ws);
            return err;
        }
        chol_csc_end_column(ws);
    }

    chol_csc_workspace_free(ws);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 6: Triangular solves
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t chol_csc_solve(const CholCsc *L, const double *b, double *x) {
    if (!L || !b || !x)
        return SPARSE_ERR_NULL;
    idx_t n = L->n;
    if (n == 0)
        return SPARSE_OK;

    /* Singularity threshold, scaled to the L factor's magnitude.
     * Cholesky factors grow as sqrt of A's entries, so reference norm
     * is sqrt(||A||_inf) — same convention used by sparse_cholesky.c. */
    double l_norm = L->factor_norm > 0.0 ? sqrt(L->factor_norm) : 0.0;
    double sing_tol = sparse_rel_tol(l_norm, SPARSE_DROP_TOL);

    /* Copy b into x if they don't alias (in-place solve when they do). */
    if (b != x)
        memcpy(x, b, (size_t)n * sizeof(double));

    /* Forward solve L*y = b in a left-to-right column sweep.
     *   x[j] = x[j] / L[j,j]                                (diagonal)
     *   x[i] -= L[i,j] * x[j]  for each stored i > j        (below-diag) */
    for (idx_t j = 0; j < n; j++) {
        idx_t start = L->col_ptr[j];
        idx_t end = L->col_ptr[j + 1];
        if (start == end || L->row_idx[start] != j)
            return SPARSE_ERR_SINGULAR; /* column empty / missing diagonal */
        double l_jj = L->values[start];
        if (fabs(l_jj) < sing_tol)
            return SPARSE_ERR_SINGULAR;
        x[j] /= l_jj;
        for (idx_t p = start + 1; p < end; p++) {
            idx_t i = L->row_idx[p];
            x[i] -= L->values[p] * x[j];
        }
    }

    /* Backward solve L^T*x = y in a right-to-left column sweep.  The
     * below-diagonal slice of column j of L is exactly row j of L^T, so
     *   x[j] -= sum_{i>j} L[i,j] * x[i]
     *   x[j] /= L[j,j]                                                  */
    for (idx_t j = n - 1; j >= 0; j--) {
        idx_t start = L->col_ptr[j];
        idx_t end = L->col_ptr[j + 1];
        double l_jj = L->values[start]; /* already validated in forward sweep */
        for (idx_t p = start + 1; p < end; p++) {
            idx_t i = L->row_idx[p];
            x[j] -= L->values[p] * x[i];
        }
        x[j] /= l_jj;
    }

    return SPARSE_OK;
}

sparse_err_t chol_csc_solve_perm(const CholCsc *L, const idx_t *perm, const double *b, double *x) {
    if (!L || !b || !x)
        return SPARSE_ERR_NULL;
    if (!perm)
        return chol_csc_solve(L, b, x);

    idx_t n = L->n;
    if (n == 0)
        return SPARSE_OK;

    /* tmp holds the permuted RHS and receives the solution in the new
     * (permuted) coordinate system — then we un-permute back to user
     * coordinates in x. */
    double *tmp = malloc((size_t)n * sizeof(double));
    if (!tmp)
        return SPARSE_ERR_ALLOC;

    /* Apply the permutation: tmp[new] = b[perm[new]] = b[old]. */
    for (idx_t new_i = 0; new_i < n; new_i++)
        tmp[new_i] = b[perm[new_i]];

    sparse_err_t err = chol_csc_solve(L, tmp, tmp); /* in-place */
    if (err != SPARSE_OK) {
        free(tmp);
        return err;
    }

    /* Un-permute: x[perm[new]] = tmp[new] (x[old] = tmp[new]). */
    for (idx_t new_i = 0; new_i < n; new_i++)
        x[perm[new_i]] = tmp[new_i];

    free(tmp);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 6: factor / factor+solve shims (internal; default backend Day 12)
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t chol_csc_factor(const SparseMatrix *A, const sparse_analysis_t *analysis,
                             CholCsc **L_out) {
    if (!L_out)
        return SPARSE_ERR_NULL;
    *L_out = NULL;
    if (!A)
        return SPARSE_ERR_NULL;

    CholCsc *L = NULL;
    sparse_err_t err;
    if (analysis) {
        err = chol_csc_from_sparse_with_analysis(A, analysis, &L);
    } else {
        err = chol_csc_from_sparse(A, NULL, 2.0, &L);
    }
    if (err != SPARSE_OK)
        return err;

    err = chol_csc_eliminate(L);
    if (err != SPARSE_OK) {
        chol_csc_free(L);
        return err;
    }

    *L_out = L;
    return SPARSE_OK;
}

sparse_err_t chol_csc_factor_solve(const SparseMatrix *A, const sparse_analysis_t *analysis,
                                   const double *b, double *x) {
    if (!A || !b || !x)
        return SPARSE_ERR_NULL;

    CholCsc *L = NULL;
    sparse_err_t err = chol_csc_factor(A, analysis, &L);
    if (err != SPARSE_OK)
        return err;

    if (analysis)
        err = chol_csc_solve_perm(L, analysis->perm, b, x);
    else
        err = chol_csc_solve(L, b, x);

    chol_csc_free(L);
    return err;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 10: Fundamental supernode detection
 * ═══════════════════════════════════════════════════════════════════════ */

/* Return 1 iff columns `prev` and `curr = prev + 1` belong to the same
 * fundamental supernode of L.  See the header's design block for the
 * three conditions; all three run in O(nnz(column prev)) on sorted
 * CSC storage. */
static int columns_in_same_supernode(const CholCsc *L, idx_t prev) {
    idx_t curr = prev + 1;
    idx_t prev_start = L->col_ptr[prev];
    idx_t prev_end = L->col_ptr[prev + 1];
    idx_t curr_start = L->col_ptr[curr];
    idx_t curr_end = L->col_ptr[curr + 1];

    idx_t prev_size = prev_end - prev_start;
    idx_t curr_size = curr_end - curr_start;

    /* Condition 1: column prev has at least the diagonal plus one
     * sub-diagonal entry, and that sub-diagonal is exactly row `curr`
     * (i.e., L[curr, prev] != 0 and is the first stored below-diag
     * entry of column prev). */
    if (prev_size < 2)
        return 0;
    if (L->row_idx[prev_start + 1] != curr)
        return 0;

    /* Condition 2: column curr has exactly one fewer stored entry. */
    if (curr_size != prev_size - 1)
        return 0;

    /* Condition 3: rows after curr's diagonal match rows after prev's
     * first sub-diagonal, 1-for-1.  prev_start + 0 is prev's diagonal
     * (row = prev), prev_start + 1 is row = curr; so prev's remaining
     * rows start at prev_start + 2.  curr's diagonal is at curr_start
     * (row = curr); its remaining rows start at curr_start + 1. */
    idx_t tail_len = curr_size - 1;
    for (idx_t t = 0; t < tail_len; t++) {
        if (L->row_idx[curr_start + 1 + t] != L->row_idx[prev_start + 2 + t])
            return 0;
    }
    return 1;
}

sparse_err_t chol_csc_detect_supernodes(const CholCsc *L, idx_t min_size, idx_t *super_starts,
                                        idx_t *super_sizes, idx_t *count) {
    if (!L || !super_starts || !super_sizes || !count)
        return SPARSE_ERR_NULL;
    if (min_size < 1)
        return SPARSE_ERR_BADARG;

    idx_t n = L->n;
    idx_t written = 0;
    idx_t j = 0;

    while (j < n) {
        /* Extend from column j as long as consecutive columns satisfy
         * the supernode invariants. */
        idx_t end = j + 1;
        while (end < n && columns_in_same_supernode(L, end - 1))
            end++;

        idx_t size = end - j;
        if (size >= min_size) {
            super_starts[written] = j;
            super_sizes[written] = size;
            written++;
        }
        /* Columns that don't form a large-enough supernode are simply
         * skipped — the caller's elimination treats them scalar-wise. */
        j = end;
    }

    *count = written;
    return SPARSE_OK;
}

#ifndef NDEBUG
#include <stdio.h>
void chol_csc_dump_supernodes(const idx_t *super_starts, const idx_t *super_sizes, idx_t count) {
    printf("chol_csc supernodes: count=%d\n", (int)count);
    for (idx_t i = 0; i < count; i++) {
        idx_t s = super_starts[i];
        idx_t sz = super_sizes[i];
        printf("  [%d]: cols [%d, %d) size=%d\n", (int)i, (int)s, (int)(s + sz), (int)sz);
    }
}
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Day 11: Dense Cholesky primitives
 * ═══════════════════════════════════════════════════════════════════════ */

/* Column-major indexing helper: A[i, j] lives at A[i + j*lda]. */

sparse_err_t chol_dense_factor(double *A, idx_t n, idx_t lda, double tol) {
    if (!A)
        return SPARSE_ERR_NULL;
    if (n < 0 || lda < n)
        return SPARSE_ERR_BADARG;
    if (n == 0)
        return SPARSE_OK;

    /* Approximate reference norm from A's initial diagonal (before any
     * updates) for relative tolerance scaling.  Keeps the kernel
     * self-contained without forcing callers to pass ||A||_inf. */
    double ref_norm = 0.0;
    for (idx_t j = 0; j < n; j++) {
        double d = fabs(A[j + j * lda]);
        if (d > ref_norm)
            ref_norm = d;
    }
    double sing_tol = sparse_rel_tol(ref_norm, tol > 0.0 ? tol : SPARSE_DROP_TOL);

    for (idx_t k = 0; k < n; k++) {
        /* Diagonal accumulator: A[k,k] - sum_{j<k} L[k,j]^2. */
        double s = A[k + k * lda];
        for (idx_t j = 0; j < k; j++) {
            double l_kj = A[k + j * lda];
            s -= l_kj * l_kj;
        }
        if (s < sing_tol)
            return SPARSE_ERR_NOT_SPD;
        double l_kk = sqrt(s);
        A[k + k * lda] = l_kk;
        double inv_l_kk = 1.0 / l_kk;

        /* Below-diagonal column: L[i, k] = (A[i,k] - sum_{j<k} L[i,j]*L[k,j]) / L[k,k]. */
        for (idx_t i = k + 1; i < n; i++) {
            double t = A[i + k * lda];
            for (idx_t j = 0; j < k; j++)
                t -= A[i + j * lda] * A[k + j * lda];
            A[i + k * lda] = t * inv_l_kk;
        }
    }
    return SPARSE_OK;
}

sparse_err_t chol_dense_solve_lower(const double *L, idx_t n, idx_t lda, double *b) {
    if (!L || !b)
        return SPARSE_ERR_NULL;
    if (n < 0 || lda < n)
        return SPARSE_ERR_BADARG;
    if (n == 0)
        return SPARSE_OK;

    /* Forward substitution: for each row i, b[i] -= L[i, j] * b[j] for
     * j < i, then b[i] /= L[i, i]. */
    for (idx_t i = 0; i < n; i++) {
        double sum = b[i];
        for (idx_t j = 0; j < i; j++)
            sum -= L[i + j * lda] * b[j];
        double l_ii = L[i + i * lda];
        if (l_ii == 0.0)
            return SPARSE_ERR_SINGULAR;
        b[i] = sum / l_ii;
    }
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 11: Supernode-aware elimination dispatch
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t chol_csc_eliminate_supernodal(CholCsc *csc, idx_t min_size) {
    if (!csc)
        return SPARSE_ERR_NULL;
    if (min_size < 1)
        return SPARSE_ERR_BADARG;

    /* Supernode detection on the pre-factor CSC: this observes the
     * structural shape of A's lower triangle, which matches L's shape
     * for the test matrices Day 11 targets (dense, block-diagonal).
     * For matrices whose L has strictly more fill than A, the detected
     * partition may underestimate — the scalar fallback below remains
     * correct regardless. */
    idx_t n = csc->n;
    if (n > 0) {
        idx_t *starts = malloc((size_t)n * sizeof(idx_t));
        idx_t *sizes = malloc((size_t)n * sizeof(idx_t));
        if (!starts || !sizes) {
            free(starts);
            free(sizes);
            return SPARSE_ERR_ALLOC;
        }
        idx_t super_count = 0;
        sparse_err_t err = chol_csc_detect_supernodes(csc, min_size, starts, sizes, &super_count);
        free(starts);
        free(sizes);
        if (err != SPARSE_OK)
            return err;
        /* Detection's output informs a future batched dense-kernel
         * path; for now we delegate to the scalar Day 5 kernel, which
         * is already correct and well-tested. */
    }

    return chol_csc_eliminate(csc);
}
