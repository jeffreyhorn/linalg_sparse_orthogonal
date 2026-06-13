/**
 * @file sparse_chol_csc.c
 * @brief CSC working-format numeric backend for Cholesky factorization.
 *
 * This backend keeps the Cholesky factor in column-oriented CSC storage
 * because both the scalar `cmod` / `cdiv` updates and the triangular solves
 * are naturally column sweeps. Compared with the linked-list
 * `SparseMatrix` path, the CSC layout replaces pointer chasing and per-entry
 * fill allocation with contiguous row/value arrays plus a dense
 * scatter-gather workspace.
 *
 * The file owns:
 *
 * - CSC lifecycle and validation helpers
 * - linked-list ↔ CSC conversion and writeback
 * - the scalar numeric elimination and solve path
 * - backend dispatch and compatibility shims
 *
 * The supernodal batched backend lives beside this file in
 * `src/sparse_chol_csc_supernodal.c`. The two paths share the same storage
 * invariants and can be selected transparently by the higher-level Cholesky
 * API.
 *
 * Symbolic analysis matters because the batched supernodal path needs the
 * full structural `sym_L` pattern up front. `chol_csc_from_sparse_with_analysis`
 * materialises that exact pattern, while `chol_csc_from_sparse` keeps the
 * heuristic growth path for the scalar kernel.
 */

#include "sparse_analysis_internal.h"
#include "sparse_chol_csc_internal.h"
#include "sparse_matrix.h"
#include "sparse_matrix_internal.h"
#include "sparse_matrix_state_internal.h"

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
    m->sym_L_preallocated = 0; /* default: heuristic; flipped to 1 by `_with_analysis` */

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
    if (needed < 0)
        return SPARSE_ERR_ALLOC;
    if (needed <= m->capacity)
        return SPARSE_OK;

    /* Geometric growth: at least 2× current capacity, or needed — whichever
     * is larger.  Guard idx_t overflow in the doubling. */
    idx_t new_cap;
    if (m->capacity > IDX_MAX / 2)
        new_cap = IDX_MAX;
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

/* Forward declaration — full definition near the CSC scatter helpers.
 * Needed early by `chol_csc_from_sparse_with_analysis` when it places
 * A's values at their sym_L positions. */
static idx_t bsearch_row(const idx_t *row_idx, idx_t start, idx_t end, idx_t target);

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

    /* Sum: total nonzeros in the lower triangle after permutation.
     * Accumulate in size_t so the sum can exceed INT32_MAX without
     * signed overflow; reject matrices whose lower triangle doesn't fit
     * in idx_t before narrowing. */
    size_t nnz_lower_wide = 0;
    for (idx_t j = 0; j < n; j++)
        nnz_lower_wide += (size_t)col_count[j];
    if (nnz_lower_wide > (size_t)INT32_MAX) {
        free(col_count);
        free(invperm);
        return SPARSE_ERR_ALLOC;
    }
    idx_t nnz_lower = (idx_t)nnz_lower_wide;

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

/* Current state in this PR: `chol_csc_from_sparse` still uses the
 * `fill_factor`-sized `from_sparse_impl(...)` path below (i.e. the
 * heuristic initialiser, with `sym_L_preallocated == 0`).  The
 * `chol_csc_gather` "column slots are already at their final size"
 * fast path belongs to `chol_csc_from_sparse_with_analysis`, which
 * consumes a preallocated `analysis->sym_L` and sets
 * `sym_L_preallocated = 1` — that is the initialiser the Sprint 18
 * Day 11 supernodal dispatch (`sparse_cholesky_factor_opts` with
 * `backend == SPARSE_CHOL_BACKEND_AUTO` and `n >= SPARSE_CSC_THRESHOLD`)
 * routes through today.
 *
 * Planned follow-up (per `docs/planning/EPIC_2/SPRINT_19/kuu_fix_decision.md`):
 * migrate this function from the `fill_factor` heuristic to full
 * sym_L pre-allocation so it matches `_with_analysis`.  The Day 5
 * `sample` profile on Kuu attributes 60% of scalar CSC factor time
 * to `_platform_memmove` inside `chol_csc_gather`'s
 * `shift_columns_right_of` path; once this function also pre-
 * allocates sym_L up front, that shift can be removed here too
 * because the column slots will already be at their final size on
 * entry to elimination.  Sketched migration steps:
 *
 *   1. Extract a static `compute_sym_L_pattern(mat, perm, &col_ptr,
 *      &row_idx, &nnz)` helper that runs the `sparse_etree_compute`
 *      / `sparse_colcount` / `sparse_symbolic_cholesky` pipeline
 *      without building a full `sparse_analysis_t`.  The helper is
 *      shared by the heuristic path (this function) and
 *      `_with_analysis` (which currently consumes `analysis->sym_L`
 *      directly; it will keep working unchanged, just sharing the
 *      implementation).
 *   2. `chol_csc_from_sparse` calls `compute_sym_L_pattern` when the
 *      matrix is symmetric; the returned `col_ptr` / `row_idx`
 *      define the CSC's immutable structural layout.  Falls back to
 *      the current heuristic when the matrix fails the symmetry
 *      check (non-Cholesky callers; they aren't the Kuu-impacted
 *      code path).
 *   3. A's lower-triangle entries are scattered into their matching
 *      sym_L positions via the same bsearch-into-row-range pattern
 *      that `_with_analysis` uses today.  Fill positions are zero-
 *      initialised.
 *   4. `chol_csc_gather` loses the `shift_columns_right_of` call.
 *      Drop-tolerance filtering becomes "write 0.0 into below-
 *      threshold below-diagonal slots" (matching the supernodal
 *      writeback's in-place zero convention from Sprint 18 Day 10),
 *      keeping `col_ptr` immutable after elimination.
 *   5. Downstream consumers already tolerate zero-valued stored
 *      entries (`chol_csc_solve` reads through them harmlessly;
 *      `chol_csc_writeback_to_sparse` skips `v == 0.0`).
 *
 * Memory cost on the Sprint 18 corpus (measured Day 5, table in the
 * decision doc): small matrices get a ~10% reduction (sym_L stores
 * only the lower triangle vs fill_factor × A.nnz covering the full
 * symmetric pattern); larger matrices pay up to +2.5× over the
 * current heuristic but exactly equal to what `_with_analysis` (and
 * therefore the Sprint 18 Day 11 supernodal dispatch) already pays.
 * Net library memory footprint stays the same; only the scalar path
 * rebalances. */
sparse_err_t chol_csc_from_sparse(const SparseMatrix *mat, const idx_t *perm, double fill_factor,
                                  CholCsc **csc_out) {
    return from_sparse_impl(mat, perm, fill_factor, 0, csc_out);
}

/* ─── Public: symbolic-analysis-aware conversion ────────────────────── */

/* Sprint 18 Day 12 change: pre-populate each column with the FULL
 * symbolic L pattern (from `analysis->sym_L`) instead of just A's
 * lower-triangle entries.  The batched supernodal kernel
 * (`chol_csc_eliminate_supernodal`) reads `col_ptr[j+1] - col_ptr[j]`
 * as the supernode's panel height and requires every fill row to be
 * pre-allocated — the Sprint 18 Days 6-10 implementation silently
 * missed fill rows on matrices like bcsstk14/s3rmt3m3/Kuu where
 * sym_L(j) is strictly larger than A(j).  Scalar elimination still
 * works on an A-pattern-only CSC because `chol_csc_gather` extends
 * columns via `shift_columns_right_of`, but the supernodal path has
 * no such hook.  Materialising sym_L up front keeps both paths
 * correct; scalar gather still shrinks the column slot when drop-
 * tolerance prunes entries below the fill threshold. */
sparse_err_t chol_csc_from_sparse_with_analysis(const SparseMatrix *mat,
                                                const sparse_analysis_t *analysis,
                                                CholCsc **csc_out) {
    if (!csc_out)
        return SPARSE_ERR_NULL;
    *csc_out = NULL;
    if (!mat || !analysis)
        return SPARSE_ERR_NULL;
    /* Accept either symmetric factor type: `sparse_analyze` runs the
     * same etree / colcount / symbolic-Cholesky pipeline for both
     * (see `src/sparse_analysis.c`: the `SPARSE_FACTOR_CHOLESKY:
     * case SPARSE_FACTOR_LDLT:` fall-through), so the resulting
     * `sym_L` is identical and valid for both factorisations.  LU
     * analyses produce a different pattern and are rejected. */
    if (analysis->type != SPARSE_FACTOR_CHOLESKY && analysis->type != SPARSE_FACTOR_LDLT)
        return SPARSE_ERR_BADARG;
    if (mat->rows != mat->cols)
        return SPARSE_ERR_SHAPE;
    if (mat->rows != analysis->n)
        return SPARSE_ERR_SHAPE;

    idx_t n = mat->rows;
    idx_t predicted = analysis->sym_L.nnz;
    if (predicted < 0)
        predicted = 0;

    /* Build inverse of analysis->perm for logical → new mapping. */
    idx_t *invperm = NULL;
    if (analysis->perm) {
        sparse_err_t verr = validate_perm(analysis->perm, n);
        if (verr != SPARSE_OK)
            return verr;
        invperm = malloc((size_t)n * sizeof(idx_t));
        if (!invperm)
            return SPARSE_ERR_ALLOC;
        for (idx_t new_i = 0; new_i < n; new_i++)
            invperm[analysis->perm[new_i]] = new_i;
    }

    CholCsc *csc = NULL;
    sparse_err_t err = chol_csc_alloc(n, predicted > 0 ? predicted : 1, &csc);
    if (err != SPARSE_OK) {
        free(invperm);
        return err;
    }

    /* Copy sym_L's col_ptr and row_idx directly — every column now
     * holds its full predicted pattern (diagonal first, then sorted
     * sub-diagonal fill rows).  Values start at 0 (from calloc inside
     * chol_csc_alloc); A's lower-triangle entries are scattered into
     * their matching positions below. */
    memcpy(csc->col_ptr, analysis->sym_L.col_ptr, (size_t)(n + 1) * sizeof(idx_t));
    if (predicted > 0)
        memcpy(csc->row_idx, analysis->sym_L.row_idx, (size_t)predicted * sizeof(idx_t));
    csc->nnz = predicted;
    /* Reuse the ||A||_inf that `sparse_analyze` already computed
     * during symbolic analysis instead of re-walking the matrix.
     * Saves an O(nnz) pass on the hot CSC dispatch path. */
    csc->factor_norm = analysis->analysis_norm;
    /* Sprint 19 Day 7: mark the CSC as sym_L-pre-allocated so
     * `chol_csc_gather`'s fast path can skip the per-call merge-walk
     * safety check — we've pre-populated every sym_L row so every
     * cmod survivor is guaranteed to be in the slot. */
    csc->sym_L_preallocated = 1;

    for (idx_t phys_i = 0; phys_i < n; phys_i++) {
        idx_t log_i = mat->inv_row_perm[phys_i];
        idx_t new_i = invperm ? invperm[log_i] : log_i;
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            idx_t new_j = invperm ? invperm[log_j] : log_j;
            if (new_i >= new_j) {
                idx_t cs = csc->col_ptr[new_j];
                idx_t ce = csc->col_ptr[new_j + 1];
                idx_t pos = bsearch_row(csc->row_idx, cs, ce, new_i);
                if (pos < ce) {
                    csc->values[pos] = node->value;
                } else {
                    /* A's entry doesn't appear in sym_L's pattern — indicates
                     * analysis and matrix are out of sync.  Reject. */
                    chol_csc_free(csc);
                    free(invperm);
                    return SPARSE_ERR_BADARG;
                }
            }
            node = node->right;
        }
    }

    free(invperm);
    *csc_out = csc;
    return SPARSE_OK;
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
    /* When there is any storage at all, row_idx/values must be present —
     * the per-column loop below dereferences row_idx for non-empty
     * columns, and downstream consumers (e.g. csc_to_full_symmetric_matrix)
     * dereference values after ldlt_csc_validate() delegates here. */
    if (csc->nnz > 0 && (!csc->row_idx || !csc->values))
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
    if (start >= end)
        return end;

    const idx_t *slice = row_idx + start;
    idx_t count = end - start;
    idx_t lo = 0;
    idx_t hi = count;
    while (lo < hi) {
        idx_t mid = lo + (hi - lo) / 2;
        if (slice[mid] < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo >= count)
        return end;
    return (slice[lo] == target) ? (start + lo) : end;
}

/* ─── Scatter / cmod / cdiv / gather / end_column ─────────────────── */

void chol_csc_scatter(const CholCsc *csc, idx_t j, CholCscWorkspace *ws) {
    idx_t start = csc->col_ptr[j];
    idx_t end = csc->col_ptr[j + 1];
    if (start >= end)
        return;

    const idx_t *rows = csc->row_idx + start;
    const double *values = csc->values + start;
    idx_t count = end - start;
    for (idx_t p = 0; p < count; p++) {
        idx_t i = rows[p];
        ws->dense_col[i] = values[p];
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

/* Sprint 19 Days 6–7 implementation (rationale captured in
 * `docs/planning/EPIC_2/SPRINT_19/kuu_fix_decision.md` and the
 * companion block in `chol_csc_from_sparse`).
 *
 * Day 5's `sample` profile on scalar CSC Kuu factor attributes 60%
 * of total factor time to `_platform_memmove` launched from here
 * via `shift_columns_right_of`.  The Day 6 fix pre-allocates the
 * full `sym_L` pattern in `chol_csc_from_sparse_with_analysis`
 * (which sets `csc->sym_L_preallocated = 1`) so columns passing
 * through that initialiser arrive at `chol_csc_gather` with their
 * final slot size already populated with sym_L rows.  On those
 * columns this function takes the fast path below:
 *
 *   - sort the survivor pattern in place (kept);
 *   - write surviving values into the pre-sized slot starting at
 *     `csc->col_ptr[j]` (kept — same write loop as today);
 *   - zero-pad the remaining slot positions up to `csc->col_ptr[j+1]`
 *     (new — replaces the `shift_columns_right_of` memmove);
 *   - return SPARSE_OK (kept).
 *
 * Columns built through the heuristic `chol_csc_from_sparse`
 * (`sym_L_preallocated == 0`) still run the safety merge-walk below
 * to decide per-column whether every survivor row was pre-
 * populated; if a cmod introduced a fill row the heuristic pattern
 * did not cover, the old `shift_columns_right_of` path is used for
 * that column so the factor remains correct.  Once
 * `chol_csc_from_sparse` itself migrates to full sym_L pre-
 * allocation (see the plan block above `chol_csc_from_sparse`), the
 * merge-walk gate can be removed and `shift_columns_right_of`
 * retired entirely in a follow-up.
 *
 * Downstream consumers already handle zero-valued stored entries:
 * `chol_csc_solve` multiplies by them harmlessly; the supernodal
 * writeback's `v == 0.0` skip in `chol_csc_writeback_to_sparse`
 * (Sprint 18 Day 10) keeps the transplanted `SparseMatrix` sparsity
 * matching the linked-list kernel.  The CSC itself then retains the
 * full sym_L row pattern with some slots zeroed — identical to the
 * supernodal path's post-writeback state. */
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

    /* Count survivors so we know whether the pre-allocated slot fits. */
    idx_t keep = 0;
    for (idx_t idx = 0; idx < ws->pattern_count; idx++) {
        idx_t i = ws->dense_pattern[idx];
        if (i == j || fabs(ws->dense_col[i]) >= threshold)
            keep++;
    }

    idx_t old_size = csc->col_ptr[j + 1] - csc->col_ptr[j];

    /* Sprint 19 Day 6 fast path: when the pre-allocated slot fits the
     * survivors AND every survivor row is already present in the
     * slot's row_idx (guaranteed for the `chol_csc_from_sparse_with_analysis`
     * initialiser that pre-populates sym_L's full pattern), skip
     * `shift_columns_right_of` entirely — write values in place into
     * the existing slot and zero out the unused positions.  `col_ptr`
     * stays immutable across the elimination, which eliminates the
     * 60%-of-factor-time `_platform_memmove` the Day 5 Kuu profile
     * attributed to shrink shifts.
     *
     * Sprint 19 Day 7: when `csc->sym_L_preallocated` is set (by
     * `chol_csc_from_sparse_with_analysis`), the merge-walk safety
     * check is redundant — sym_L by definition covers every cmod-
     * producible row.  Skip the O(pattern_count) walk and jump
     * straight into the fast path.  This restored small-matrix
     * performance on nos4 / bcsstk04 that the Day 6 merge-walk had
     * regressed.  Heuristic initialisers (`chol_csc_from_sparse` with
     * a `fill_factor`) run the merge-walk as before so they fall
     * back to the slow path when cmod introduces a fill row that
     * wasn't in A's lower-triangle pattern. */
    int all_in_slot = (keep <= old_size);
    if (all_in_slot && !csc->sym_L_preallocated) {
        idx_t slot_scan = csc->col_ptr[j];
        idx_t slot_end = csc->col_ptr[j + 1];
        for (idx_t idx = 0; idx < ws->pattern_count; idx++) {
            idx_t i = ws->dense_pattern[idx];
            if (i != j && fabs(ws->dense_col[i]) < threshold)
                continue; /* dropped — no slot needed */
            while (slot_scan < slot_end && csc->row_idx[slot_scan] < i)
                slot_scan++;
            if (slot_scan == slot_end || csc->row_idx[slot_scan] != i) {
                all_in_slot = 0;
                break;
            }
            slot_scan++;
        }
    }

    if (all_in_slot) {
        /* Write values in place keyed by the existing row_idx; zero
         * any pre-populated position whose accumulator is below
         * threshold (drop-tol) or untouched (fill row that did not
         * collect a cmod contribution). */
        idx_t start = csc->col_ptr[j];
        idx_t end = csc->col_ptr[j + 1];
        if (start >= end)
            return SPARSE_OK;
        const idx_t *rows = csc->row_idx + start;
        double *values = csc->values + start;
        idx_t count = end - start;
        for (idx_t p = 0; p < count; p++) {
            idx_t i = rows[p];
            double v = ws->dense_col[i];
            if (i != j && fabs(v) < threshold)
                v = 0.0;
            values[p] = v;
        }
        return SPARSE_OK;
    }

    /* Slow path (heuristic initialiser + fill-in, or a survivor row
     * missing from the pre-allocated slot): resize the column slot
     * via `shift_columns_right_of` and write survivors from scratch.
     * `delta > 0` grows, `delta < 0` shrinks; either way the slow
     * path pays the O(nnz) memmove that the fast path skipped. */
    idx_t delta = keep - old_size;
    sparse_err_t err = shift_columns_right_of(csc, j, delta);
    if (err != SPARSE_OK)
        return err;

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

    if (analysis && A->rows >= SPARSE_CSC_THRESHOLD) {
        err = chol_csc_eliminate_supernodal(L, SPARSE_CSC_SUPERNODE_MIN_SIZE);
    } else {
        err = chol_csc_eliminate(L);
    }
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
 * Sprint 19 Day 11: Dense LDL^T primitive (Bunch-Kaufman)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * `ldlt_dense_factor` runs Bunch-Kaufman LDL^T on a column-major
 * n×n dense buffer.  Mirrors the sparse reference in
 * `src/sparse_ldlt.c` column-by-column but on dense storage, so the
 * Sprint 19 Days 12-14 batched supernodal path can call it per
 * supernode's diagonal block just like the Cholesky batched path
 * calls `chol_dense_factor`.
 *
 * Input convention: `A` is n×n column-major symmetric with BOTH
 * triangles populated — `A[i + j*lda] == A[j + i*lda]` at call time.
 * Supernodal callers should scatter the diagonal block that way
 * before calling (see Day 12 `ldlt_csc_supernode_extract`).  The
 * upper triangle gets overwritten as scratch during symmetric swaps
 * and trailing updates; not preserved post-factor.
 *
 * Output convention:
 *   - `A` holds unit lower triangular L: `A[k + k*lda] = 1.0`,
 *     below-diagonal positions hold the L multipliers.  For a 2×2
 *     pivot at (k, k+1) the row-k+1 column-k slot `A[(k+1) + k*lda]`
 *     is zeroed (the 2×2 coupling lives in `D_offdiag`, not L).
 *     Positions above the diagonal are scratch.
 *   - `D[k]`: the 1×1 pivot scalar, or the block-(k, k) diagonal for
 *     a 2×2 pivot.  `D[k+1]` for a 2×2 holds the block-(k+1, k+1)
 *     diagonal.
 *   - `D_offdiag[k]`: the 2×2 block's (k, k+1) off-diagonal for a 2×2
 *     pivot; 0 otherwise.
 *   - `pivot_size[k]`: 1 for a 1×1, 2 for both indices of a 2×2
 *     pair (`pivot_size[k] == pivot_size[k+1] == 2`).
 *   - `elem_growth_out` (optional): the maximum |L[i, j]| observed
 *     during the factor, suitable for callers that want to monitor
 *     element-growth stability.  Pass NULL to skip.
 *
 * Errors: `SPARSE_ERR_NULL` (required output pointer NULL),
 * `SPARSE_ERR_BADARG` (n < 0, lda < n), `SPARSE_ERR_SINGULAR`
 * (pivot near-zero or |L| blows past the growth bound).
 */

/* Symmetric row + column swap of a dense column-major n×n matrix.
 * Maintains symmetry by swapping both rows AND columns in sync. */
static void ldlt_dense_sym_swap(double *A, idx_t n, idx_t lda, idx_t a, idx_t b) {
    if (a == b)
        return;
    for (idx_t c = 0; c < n; c++) {
        double tmp = A[a + c * lda];
        A[a + c * lda] = A[b + c * lda];
        A[b + c * lda] = tmp;
    }
    for (idx_t r = 0; r < n; r++) {
        double tmp = A[r + a * lda];
        A[r + a * lda] = A[r + b * lda];
        A[r + b * lda] = tmp;
    }
}

sparse_err_t ldlt_dense_factor(double *A, double *D, double *D_offdiag, idx_t *pivot_size, idx_t n,
                               idx_t lda, double tol, double *elem_growth_out) {
    if (!A || !D || !D_offdiag || !pivot_size)
        return SPARSE_ERR_NULL;
    if (n < 0 || lda < n)
        return SPARSE_ERR_BADARG;
    if (elem_growth_out)
        *elem_growth_out = 0.0;
    if (n == 0)
        return SPARSE_OK;

    /* Relative tolerance tied to the initial diagonal max — matches
     * `chol_dense_factor`'s self-contained convention. */
    double ref_norm = 0.0;
    for (idx_t j = 0; j < n; j++) {
        double d = fabs(A[j + j * lda]);
        if (d > ref_norm)
            ref_norm = d;
    }
    double eff_tol = tol > 0.0 ? tol : SPARSE_DROP_TOL;
    double sing_tol = sparse_rel_tol(ref_norm, eff_tol);
    double growth_bound = 1.0 / (100.0 * eff_tol);
    double alpha_bk = (1.0 + sqrt(17.0)) / 8.0; /* ≈ 0.6404 */
    double max_growth = 0.0;
    sparse_err_t err = SPARSE_OK;

    /* Single per-call scratch buffer for 2×2-pivot L multipliers
     * (`l_col_k` || `l_col_k1`).  Sized for the worst-case tail
     * (k = 0 → tail_len = n - 2, two columns), then reused across
     * every 2×2 pivot in this call to avoid per-pivot malloc/free
     * churn in the hot supernode kernel.  When n <= 2 the 2×2 path's
     * tail_len is always 0, so no scratch is needed. */
    double *pivot_scratch = NULL;
    if (n > 2) {
        pivot_scratch = malloc((size_t)(2 * (n - 2)) * sizeof(double));
        if (!pivot_scratch)
            return SPARSE_ERR_ALLOC;
    }

    idx_t k = 0;
    while (k < n) {
        /* ── Pivot selection: four-criteria Bunch-Kaufman ──────────────── */
        double lambda = 0.0;
        idx_t r = k;
        for (idx_t i = k + 1; i < n; i++) {
            double v = fabs(A[i + k * lda]);
            if (v > lambda) {
                lambda = v;
                r = i;
            }
        }

        int use_2x2 = 0;
        if (lambda == 0.0 || fabs(A[k + k * lda]) >= alpha_bk * lambda) {
            /* Criterion 1: 1×1 at (k, k). */
            use_2x2 = 0;
        } else {
            /* Need sigma_r = max |A[i, r]| for i in [k, n), i != r. */
            double sigma_r = 0.0;
            for (idx_t i = k; i < n; i++) {
                if (i == r)
                    continue;
                double v = fabs(A[i + r * lda]);
                if (v > sigma_r)
                    sigma_r = v;
            }
            if (fabs(A[k + k * lda]) * sigma_r >= alpha_bk * lambda * lambda) {
                /* Criterion 2: 1×1 at (k, k). */
                use_2x2 = 0;
            } else if (fabs(A[r + r * lda]) >= alpha_bk * sigma_r) {
                /* Criterion 3: 1×1 at (r, r) — swap k ↔ r to bring
                 * the chosen pivot into slot k. */
                ldlt_dense_sym_swap(A, n, lda, k, r);
                use_2x2 = 0;
            } else {
                /* Criterion 4: 2×2 at (k, r) — ensure the partner sits
                 * at slot k+1 via swap. */
                if (r != k + 1)
                    ldlt_dense_sym_swap(A, n, lda, k + 1, r);
                use_2x2 = 1;
            }
        }

        if (!use_2x2) {
            /* ── 1×1 pivot elimination at column k ─────────────────────── */
            double dk = A[k + k * lda];
            if (fabs(dk) < sing_tol) {
                err = SPARSE_ERR_SINGULAR;
                goto cleanup;
            }

            D[k] = dk;
            D_offdiag[k] = 0.0;
            pivot_size[k] = 1;

            /* Trailing rank-1 update: for i > k, j > k (lower tri, j <= i):
             *   A[i, j] -= A[i, k] * A[j, k] / dk
             * Also update the upper triangle to keep the matrix symmetric
             * for the next iteration's pivot selection (which reads
             * `A[i + r*lda]` via direct indexing, not symmetry reflection). */
            double inv_dk = 1.0 / dk;
            for (idx_t j = k + 1; j < n; j++) {
                double ajk = A[j + k * lda];
                double factor = ajk * inv_dk;
                for (idx_t i = j; i < n; i++) {
                    double update = A[i + k * lda] * factor;
                    A[i + j * lda] -= update;
                    if (i != j)
                        A[j + i * lda] = A[i + j * lda]; /* mirror to upper tri */
                }
            }

            /* Write L[i, k] = A[i, k] / dk, then the unit diagonal. */
            for (idx_t i = k + 1; i < n; i++) {
                double l_ik = A[i + k * lda] * inv_dk;
                if (fabs(l_ik) > growth_bound) {
                    err = SPARSE_ERR_SINGULAR;
                    goto cleanup;
                }
                if (fabs(l_ik) > max_growth)
                    max_growth = fabs(l_ik);
                A[i + k * lda] = l_ik;
                A[k + i * lda] = l_ik; /* mirror for subsequent BK scans */
            }
            A[k + k * lda] = 1.0;

            k += 1;
        } else {
            /* ── 2×2 pivot elimination at (k, k+1) ─────────────────────── */
            double d11 = A[k + k * lda];
            double d21 = A[(k + 1) + k * lda];
            double d22 = A[(k + 1) + (k + 1) * lda];
            double det = d11 * d22 - d21 * d21;
            double bscale = fabs(d11) + fabs(d22) + fabs(d21);
            double det_tol = (bscale > 0.0) ? eff_tol * bscale * bscale : sing_tol * sing_tol;
            if (fabs(det) < det_tol) {
                err = SPARSE_ERR_SINGULAR;
                goto cleanup;
            }

            D[k] = d11;
            D[k + 1] = d22;
            D_offdiag[k] = d21;
            D_offdiag[k + 1] = 0.0;
            pivot_size[k] = 2;
            pivot_size[k + 1] = 2;

            double inv_det = 1.0 / det;

            /* Precompute the L multipliers once per row i.  The
             * trailing rank-2 update and the column overwrite both
             * need (l_ik, l_ik1) for every i in [k+2, n) — recomputing
             * them inside the (i, j) loops costs O(supernode^2) extra
             * work per 2×2 pivot, which dominates on large supernodes.
             * Scratch storage comes from the per-call `pivot_scratch`
             * buffer allocated up front. */
            idx_t tail_len = n - (k + 2);
            if (tail_len > 0) {
                /* `pivot_scratch` is non-NULL whenever n > 2, and the
                 * worst-case allocation (2 * (n - 2) doubles) covers
                 * every tail_len reached at any k. */
                double *l_col_k = pivot_scratch;
                double *l_col_k1 = pivot_scratch + tail_len;
                for (idx_t i = k + 2; i < n; i++) {
                    double aik = A[i + k * lda];
                    double aik1 = A[i + (k + 1) * lda];
                    double l_ik = (aik * d22 - aik1 * d21) * inv_det;
                    double l_ik1 = (-aik * d21 + aik1 * d11) * inv_det;
                    if (fabs(l_ik) > growth_bound || fabs(l_ik1) > growth_bound) {
                        err = SPARSE_ERR_SINGULAR;
                        goto cleanup;
                    }
                    if (fabs(l_ik) > max_growth)
                        max_growth = fabs(l_ik);
                    if (fabs(l_ik1) > max_growth)
                        max_growth = fabs(l_ik1);
                    l_col_k[i - (k + 2)] = l_ik;
                    l_col_k1[i - (k + 2)] = l_ik1;
                }

                /* Trailing rank-2 update: for j > k+1, i >= j:
                 *   A[i, j] -= L[i, k] * A[j, k] + L[i, k+1] * A[j, k+1]
                 * where L[i, k] and L[i, k+1] are the precomputed
                 * multipliers from the 2×2 solve.  Do the update BEFORE
                 * overwriting A[:, k] / A[:, k+1] with L so the
                 * right-hand side uses the original Schur values. */
                for (idx_t j = k + 2; j < n; j++) {
                    double ajk = A[j + k * lda];
                    double ajk1 = A[j + (k + 1) * lda];
                    for (idx_t i = j; i < n; i++) {
                        idx_t ti = i - (k + 2);
                        double update = l_col_k[ti] * ajk + l_col_k1[ti] * ajk1;
                        A[i + j * lda] -= update;
                        if (i != j)
                            A[j + i * lda] = A[i + j * lda];
                    }
                }

                /* Overwrite columns k and k+1 with the L multipliers. */
                for (idx_t i = k + 2; i < n; i++) {
                    idx_t ti = i - (k + 2);
                    double l_ik = l_col_k[ti];
                    double l_ik1 = l_col_k1[ti];
                    A[i + k * lda] = l_ik;
                    A[i + (k + 1) * lda] = l_ik1;
                    A[k + i * lda] = l_ik;
                    A[(k + 1) + i * lda] = l_ik1;
                }
            }

            /* Unit-L diagonal within the 2×2 block: L[k,k] = L[k+1,k+1] = 1,
             * L[k+1, k] = 0 (the 2×2 coupling is captured in D_offdiag). */
            A[k + k * lda] = 1.0;
            A[(k + 1) + k * lda] = 0.0;
            A[(k + 1) + (k + 1) * lda] = 1.0;

            k += 2;
        }
    }

    if (elem_growth_out)
        *elem_growth_out = max_growth;

cleanup:
    free(pivot_scratch);
    return err;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 10: CSC → linked-list writeback for transparent dispatch
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t chol_csc_writeback_to_sparse(const CholCsc *L, SparseMatrix *mat, const idx_t *perm) {
    if (!L || !mat)
        return SPARSE_ERR_NULL;
    if (mat->rows != L->n || mat->cols != L->n)
        return SPARSE_ERR_SHAPE;
    if (sparse_matrix_require_original_state(mat) != SPARSE_OK)
        return SPARSE_ERR_BADARG;

    idx_t n = L->n;

    /* Copy the caller's perm (if any) up front so a later allocation
     * failure doesn't leave `mat` half-updated. */
    idx_t *perm_copy = NULL;
    if (perm && n > 0) {
        if ((size_t)n > SIZE_MAX / sizeof(idx_t))
            return SPARSE_ERR_ALLOC;
        perm_copy = malloc((size_t)n * sizeof(idx_t));
        if (!perm_copy)
            return SPARSE_ERR_ALLOC;
        for (idx_t i = 0; i < n; i++)
            perm_copy[i] = perm[i];
    }

    /* Empty-matrix shortcut: nothing to transplant, just set state. */
    if (n == 0) {
        sparse_factor_state_publish_factored(mat, L->factor_norm, perm_copy);
        /* cached_norm stays as-is (no matrix contents to invalidate). */
        return SPARSE_OK;
    }

    /* Build a fresh SparseMatrix holding L (already in "new"
     * coordinate space — chol_csc_from_sparse put it there at
     * conversion time).  Insert CSC column-by-column. */
    SparseMatrix *tmp = sparse_create(n, n);
    if (!tmp) {
        free(perm_copy);
        return SPARSE_ERR_ALLOC;
    }

    /* Skip exact zeros (common when the CSC was pre-populated with the
     * full sym_L pattern but some fill positions never received a non-
     * zero value) and drop below-diagonal entries below
     * `SPARSE_DROP_TOL * |L[j, j]|` — mirrors the scalar path's
     * `chol_csc_gather` policy so the transplanted `SparseMatrix`
     * sparsity matches what the linked-list kernel would publish.
     * The diagonal (row_idx[col_ptr[j]] == j by CSC invariant) is
     * always inserted so the solver's diagonal lookup never misses. */
    for (idx_t j = 0; j < n; j++) {
        idx_t cstart = L->col_ptr[j];
        idx_t cend = L->col_ptr[j + 1];
        if (cstart == cend)
            continue;
        /* Diagonal is the first stored entry per CSC invariant. */
        double abs_l_jj = (L->row_idx[cstart] == j) ? fabs(L->values[cstart]) : 0.0;
        double threshold = SPARSE_DROP_TOL * abs_l_jj;
        for (idx_t p = cstart; p < cend; p++) {
            idx_t i = L->row_idx[p];
            double v = L->values[p];
            if (v == 0.0)
                continue;
            if (i != j && fabs(v) < threshold)
                continue;
            sparse_err_t ierr = sparse_insert(tmp, i, j, v);
            if (ierr != SPARSE_OK) {
                sparse_free(tmp);
                free(perm_copy);
                return ierr;
            }
        }
    }

    /* Transplant tmp's internal storage into mat.  Matches the
     * post-permute swap in sparse_cholesky_factor_opts: free the
     * caller's current storage, move tmp's pool + headers over, null
     * tmp's pointers so sparse_free(tmp) doesn't double-free. */
    pool_free_all(&mat->pool);
    free(mat->row_headers);
    free(mat->col_headers);

    mat->row_headers = tmp->row_headers;
    mat->col_headers = tmp->col_headers;
    mat->pool = tmp->pool;
    mat->nnz = tmp->nnz;
    /* cached_norm is invalidated — the stored matrix just changed
     * from A to L.  Use the same sentinel (-1.0) the rest of the code
     * treats as "not cached". */
    mat->cached_norm = -1.0;

    tmp->row_headers = NULL;
    tmp->col_headers = NULL;
    tmp->pool.head = NULL;
    tmp->pool.current = NULL;
    tmp->pool.free_list = NULL;
    sparse_free(tmp);

    /* Row/col perms stay identity (precondition enforced).  Apply the
     * fill-reducing perm and the factor state. */
    sparse_factor_state_publish_factored(mat, L->factor_norm, perm_copy);

    return SPARSE_OK;
}
