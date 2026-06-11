#include "sparse_alloc_internal.h"
#include "sparse_chol_csc_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

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

/* Fully integrated batched supernodal elimination.
 *
 * Walks the CSC column-by-column, dispatching to:
 *   - the batched path for columns that start a supernode of size
 *     >= min_size (per chol_csc_detect_supernodes);
 *   - the scalar cdiv/cmod/gather path for every other column.
 *
 * For each supernode, the batched path runs:
 *   1. chol_csc_supernode_extract            (A → dense buffer)
 *   2. chol_csc_supernode_eliminate_diag     (external cmod + dense
 *                                             Cholesky on the top
 *                                             s_size × s_size slab)
 *   3. chol_csc_supernode_eliminate_panel    (chol_dense_solve_lower
 *                                             row-by-row on the panel)
 *   4. chol_csc_supernode_writeback          (dense buffer → CSC)
 *
 * After writeback, columns [s_start, s_start + s_size) hold the
 * factored L values (diagonal block + panel).  Subsequent scalar
 * columns therefore pick up correct L entries when they cmod from
 * prior supernode columns. */
sparse_err_t chol_csc_eliminate_supernodal(CholCsc *csc, idx_t min_size) {
    if (!csc)
        return SPARSE_ERR_NULL;
    if (min_size < 1)
        return SPARSE_ERR_BADARG;

    idx_t n = csc->n;
    if (n == 0)
        return SPARSE_OK;

    /* Keep the supernodal path aligned with the scalar CSC path on the
     * simplest non-SPD contract: a stored diagonal that is already
     * non-positive must reject immediately instead of slipping through
     * supernode dispatch and producing a bogus successful factor. */
    for (idx_t j = 0; j < n; j++) {
        idx_t start = csc->col_ptr[j];
        if (start < csc->col_ptr[j + 1] && csc->values[start] <= 0.0)
            return SPARSE_ERR_NOT_SPD;
    }

    idx_t *starts = NULL;
    idx_t *sizes = NULL;
    if (sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&starts) != SPARSE_OK ||
        sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&sizes) != SPARSE_OK) {
        free(starts);
        free(sizes);
        return SPARSE_ERR_ALLOC;
    }

    idx_t super_count = 0;
    sparse_err_t err = chol_csc_detect_supernodes(csc, min_size, starts, sizes, &super_count);
    if (err != SPARSE_OK) {
        free(starts);
        free(sizes);
        return err;
    }

    CholCscWorkspace *ws = NULL;
    err = chol_csc_workspace_alloc(n, &ws);
    if (err != SPARSE_OK) {
        free(starts);
        free(sizes);
        return err;
    }

    const double drop_tol = SPARSE_DROP_TOL;
    idx_t super_idx = 0;
    idx_t j = 0;
    while (j < n) {
        /* Skip past singleton detected supernodes: a size-1 "supernode"
         * has no within-supernode factoring to batch, and the batched
         * extract uses A's pre-fill column pattern — which misses
         * fill rows that the scalar gather path handles correctly via
         * its column-shift machinery.  Delegating singletons to the
         * scalar branch keeps min_size == 1 correct on matrices with
         * fill without sacrificing the batched speedup on size >= 2
         * supernodes (where the fundamental-supernode invariant
         * guarantees no new fill lands inside the supernode). */
        if (super_idx < super_count && j == starts[super_idx] && sizes[super_idx] == 1) {
            super_idx++;
            /* Fall through to the scalar branch for this column. */
        }
        if (super_idx < super_count && j == starts[super_idx]) {
            /* Batched supernode at column j (size >= 2). */
            idx_t s_start = j;
            idx_t s_size = sizes[super_idx];
            idx_t panel_height = chol_csc_supernode_panel_height(csc, s_start);

            double *dense = NULL;
            idx_t *row_map = NULL;
            /* By construction `chol_csc_detect_supernodes` only
             * reports supernodes with `s_size >= min_size >= 1`, and
             * `panel_height >= s_size` because the supernode's first
             * column stores its diagonal block plus the panel.  Guard
             * explicitly so the overflow checks below (and `calloc`)
             * can assume both are > 0. */
            if (panel_height < 1 || s_size < 1) {
                err = SPARSE_ERR_BADARG;
                break;
            }
            if ((size_t)panel_height > SIZE_MAX / sizeof(idx_t)) {
                err = SPARSE_ERR_ALLOC;
                break;
            }
            /* Guard the multiplication itself before computing the
             * product: `(size_t)panel_height * (size_t)s_size` can
             * overflow on pathological inputs, and a subsequent
             * `dense_cells > SIZE_MAX / sizeof(double)` check on an
             * already-wrapped product would miss the overflow and
             * `calloc` could under-allocate. */
            if ((size_t)s_size > SIZE_MAX / (size_t)panel_height) {
                err = SPARSE_ERR_ALLOC;
                break;
            }
            size_t dense_cells = (size_t)panel_height * (size_t)s_size;
            if (dense_cells > SIZE_MAX / sizeof(double)) {
                err = SPARSE_ERR_ALLOC;
                break;
            }
            dense = calloc(dense_cells, sizeof(double));
            row_map = malloc((size_t)panel_height * sizeof(idx_t));
            if (!dense || !row_map) {
                free(dense);
                free(row_map);
                err = SPARSE_ERR_ALLOC;
                break;
            }

            idx_t ph_out = 0;
            err = chol_csc_supernode_extract(csc, s_start, s_size, dense, panel_height, row_map,
                                             &ph_out);
            if (err != SPARSE_OK) {
                free(dense);
                free(row_map);
                break;
            }

            err = chol_csc_supernode_eliminate_diag(csc, s_start, s_size, dense, panel_height,
                                                    row_map, panel_height, drop_tol);
            if (err != SPARSE_OK) {
                free(dense);
                free(row_map);
                break;
            }

            /* Panel triangular solve: for each below-supernode row,
             * solve L_diag * x = panel_row to produce L[row, :]. */
            idx_t panel_rows = panel_height - s_size;
            if (panel_rows > 0) {
                err = chol_csc_supernode_eliminate_panel(dense, s_size, panel_height,
                                                         dense + s_size, panel_height, panel_rows);
                if (err != SPARSE_OK) {
                    free(dense);
                    free(row_map);
                    break;
                }
            }

            err = chol_csc_supernode_writeback(csc, s_start, s_size, dense, panel_height, row_map,
                                               panel_height, drop_tol);
            free(dense);
            free(row_map);
            if (err != SPARSE_OK)
                break;

            j += s_size;
            super_idx++;
        } else {
            /* Scalar column: standard scatter → cmod → cdiv → gather. */
            chol_csc_scatter(csc, j, ws);
            chol_csc_cmod(csc, j, ws);
            err = chol_csc_cdiv(ws, j);
            if (err != SPARSE_OK) {
                chol_csc_end_column(ws);
                break;
            }
            err = chol_csc_gather(csc, j, ws, drop_tol);
            if (err != SPARSE_OK) {
                chol_csc_end_column(ws);
                break;
            }
            chol_csc_end_column(ws);
            j++;
        }
    }

    chol_csc_workspace_free(ws);
    free(starts);
    free(sizes);
    return err;
}

/* Binary-search a row_map (sorted ascending) for a target global row.
 * Returns the local index, or `panel_height` when not found. */
static idx_t chol_csc_bsearch_row_map(const idx_t *row_map, idx_t panel_height, idx_t target) {
    idx_t lo = 0;
    idx_t hi = panel_height;
    while (lo < hi) {
        idx_t mid = lo + (hi - lo) / 2;
        if (row_map[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < panel_height && row_map[lo] == target) {
        return lo;
    }
    return panel_height;
}

sparse_err_t chol_csc_supernode_extract(const CholCsc *csc, idx_t s_start, idx_t s_size,
                                        double *dense, idx_t lda, idx_t *row_map,
                                        idx_t *panel_height_out) {
    if (!csc || !dense || !row_map || !panel_height_out)
        return SPARSE_ERR_NULL;
    if (s_start < 0 || s_size < 1 || s_start > csc->n - s_size)
        return SPARSE_ERR_BADARG;

    /* Panel height is the first column's stored-entry count.  By the
     * fundamental-supernode invariant this equals s_size + |S| for the
     * shared below-supernode row set S. */
    idx_t first_start = csc->col_ptr[s_start];
    idx_t panel_height = csc->col_ptr[s_start + 1] - first_start;
    if (lda < panel_height)
        return SPARSE_ERR_BADARG;
    if (panel_height < s_size)
        return SPARSE_ERR_BADARG;

    /* Seed row_map from the first column's sorted row_idx slice.  The
     * first s_size entries must be [s_start, s_start+1, ..., s_start+s_size-1]
     * by the supernode invariant (the diagonal block is stored before
     * the shared panel). */
    for (idx_t i = 0; i < panel_height; i++)
        row_map[i] = csc->row_idx[first_start + i];
    for (idx_t i = 0; i < s_size; i++) {
        if (row_map[i] != s_start + i)
            return SPARSE_ERR_BADARG;
    }
    *panel_height_out = panel_height;

    /* Scatter each column into the dense buffer.  For col s_start + j,
     * each stored (row, value) goes to dense[local_row + j*lda] where
     * local_row is looked up in row_map. */
    for (idx_t j = 0; j < s_size; j++) {
        idx_t c = s_start + j;
        idx_t cstart = csc->col_ptr[c];
        idx_t cend = csc->col_ptr[c + 1];
        for (idx_t p = cstart; p < cend; p++) {
            idx_t row = csc->row_idx[p];
            idx_t local = chol_csc_bsearch_row_map(row_map, panel_height, row);
            if (local >= panel_height)
                return SPARSE_ERR_BADARG;
            dense[local + j * lda] = csc->values[p];
        }
    }

    return SPARSE_OK;
}

sparse_err_t chol_csc_supernode_eliminate_diag(const CholCsc *csc, idx_t s_start, idx_t s_size,
                                               double *dense, idx_t lda, const idx_t *row_map,
                                               idx_t panel_height, double tol) {
    if (!csc || !dense || !row_map)
        return SPARSE_ERR_NULL;
    if (s_start < 0 || s_size < 1 || s_start > csc->n - s_size)
        return SPARSE_ERR_BADARG;
    if (panel_height < s_size || lda < panel_height)
        return SPARSE_ERR_BADARG;

    /* Scratch for L[s_start..s_start+s_size-1, k] values per prior
     * column.  Stored densely — s_size is typically small relative
     * to n, and the external-cmod loop runs once per k. */
    if ((size_t)s_size > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    double *L_col_k = malloc((size_t)s_size * sizeof(double));
    if (!L_col_k)
        return SPARSE_ERR_ALLOC;

    for (idx_t k = 0; k < s_start; k++) {
        idx_t cstart = csc->col_ptr[k];
        idx_t cend = csc->col_ptr[k + 1];
        if (cstart == cend)
            continue;

        /* Harvest L[s_start+j, k] into L_col_k[j].  row_idx within a
         * column is sorted ascending, so stop scanning once we've
         * passed the last supernode row. */
        for (idx_t j = 0; j < s_size; j++)
            L_col_k[j] = 0.0;
        int saw_supernode_row = 0;
        for (idx_t p = cstart; p < cend; p++) {
            idx_t r = csc->row_idx[p];
            if (r < s_start)
                continue;
            if (r >= s_start + s_size)
                break;
            L_col_k[r - s_start] = csc->values[p];
            saw_supernode_row = 1;
        }
        if (!saw_supernode_row)
            continue;

        /* Apply cmod: for each stored (row, k, value) whose row maps
         * into the supernode's local coordinate, subtract
         * value * L_col_k[j] from dense[local + j*lda] for j in
         * [0, s_size).
         *
         * This updates both the diagonal block (local < s_size) and
         * the panel (local >= s_size).  chol_dense_factor only reads
         * the lower triangle of the diagonal block, so writing the
         * upper triangle of that block is harmless — we don't branch
         * on local < j for correctness. */
        for (idx_t p = cstart; p < cend; p++) {
            idx_t r = csc->row_idx[p];
            idx_t local = chol_csc_bsearch_row_map(row_map, panel_height, r);
            if (local >= panel_height)
                continue;
            double v_r_k = csc->values[p];
            for (idx_t j = 0; j < s_size; j++) {
                double ljk = L_col_k[j];
                if (ljk != 0.0)
                    dense[local + j * lda] -= v_r_k * ljk;
            }
        }
    }

    free(L_col_k);

    /* Dense Cholesky factor on the top s_size × s_size diagonal
     * block.  Reads the lower triangle only; writes the factor L back
     * in place over that same region. */
    const chol_dense_kernels_t *kernels = chol_csc_supernodal_dense_kernels();
    if (!kernels || !kernels->factor)
        return SPARSE_ERR_BACKEND_CONTRACT;
    return kernels->factor(dense, s_size, lda, tol);
}

sparse_err_t chol_csc_supernode_eliminate_panel(const double *L_diag, idx_t s_size, idx_t lda_diag,
                                                double *panel, idx_t lda_panel, idx_t panel_rows) {
    if (!L_diag)
        return SPARSE_ERR_NULL;
    if (s_size < 1 || lda_diag < s_size || panel_rows < 0)
        return SPARSE_ERR_BADARG;
    if (panel_rows == 0)
        return SPARSE_OK;
    if (!panel)
        return SPARSE_ERR_NULL;
    if (lda_panel < panel_rows)
        return SPARSE_ERR_BADARG;

    if ((size_t)s_size > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    double *row_buf = malloc((size_t)s_size * sizeof(double));
    if (!row_buf)
        return SPARSE_ERR_ALLOC;

    const chol_dense_kernels_t *kernels = chol_csc_supernodal_dense_kernels();
    if (!kernels || !kernels->solve_lower) {
        free(row_buf);
        return SPARSE_ERR_BACKEND_CONTRACT;
    }

    for (idx_t i = 0; i < panel_rows; i++) {
        for (idx_t j = 0; j < s_size; j++)
            row_buf[j] = panel[i + j * lda_panel];
        sparse_err_t err = kernels->solve_lower(L_diag, s_size, lda_diag, row_buf);
        if (err != SPARSE_OK) {
            free(row_buf);
            return err;
        }
        for (idx_t j = 0; j < s_size; j++)
            panel[i + j * lda_panel] = row_buf[j];
    }

    free(row_buf);
    return SPARSE_OK;
}

sparse_err_t chol_csc_supernode_writeback(CholCsc *csc, idx_t s_start, idx_t s_size,
                                          const double *dense, idx_t lda, const idx_t *row_map,
                                          idx_t panel_height, double drop_tol) {
    if (!csc || !dense || !row_map)
        return SPARSE_ERR_NULL;
    if (s_start < 0 || s_size < 1 || s_start > csc->n - s_size)
        return SPARSE_ERR_BADARG;
    if (panel_height < s_size || lda < panel_height)
        return SPARSE_ERR_BADARG;

    /* Gather: walk each column's stored entries, translate row →
     * local_row via row_map, overwrite values[p] with the dense cell.
     * Apply the same per-column drop rule as `chol_csc_gather`:
     * below-diagonal entries below `drop_tol * |L[j, j]|` get written
     * as 0.0 so downstream consumers (solve, writeback_to_sparse) see
     * matching sparsity to the scalar path. */
    for (idx_t j = 0; j < s_size; j++) {
        idx_t c = s_start + j;
        idx_t cstart = csc->col_ptr[c];
        idx_t cend = csc->col_ptr[c + 1];
        /* Diagonal value is at dense[j + j*lda] after the diag block
         * factor ran.  Used to set the per-column threshold. */
        double abs_l_jj = fabs(dense[j + j * lda]);
        double threshold = drop_tol * abs_l_jj;
        for (idx_t p = cstart; p < cend; p++) {
            idx_t row = csc->row_idx[p];
            idx_t local = chol_csc_bsearch_row_map(row_map, panel_height, row);
            if (local >= panel_height)
                return SPARSE_ERR_BADARG;
            double v = dense[local + j * lda];
            /* Never drop the diagonal (local == j for column s_start+j
             * because row_map's first s_size slots are the supernode
             * rows in order). */
            if (local != j && fabs(v) < threshold)
                v = 0.0;
            csc->values[p] = v;
        }
    }

    return SPARSE_OK;
}
