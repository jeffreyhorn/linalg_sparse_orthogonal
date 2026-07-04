/**
 * @file sparse_ldlt_csc_rowadj.c
 * @brief Row-adjacency support for the LDL^T CSC numeric backend.
 *
 * The row-adjacency index lets the native LDL^T CSC kernel iterate only prior
 * columns with stored entries in a target row, matching the linked-list
 * reference's sparse-row traversal without scanning every previous column.
 */

#include "sparse_ldlt_csc_internal.h"

#include <stdint.h>
#include <stdlib.h>

sparse_err_t ldlt_csc_row_adj_append(LdltCsc *F, idx_t row, idx_t col) {
    if (!F)
        return SPARSE_ERR_NULL;
    if (row < 0 || row >= F->n || col < 0 || col >= F->n)
        return SPARSE_ERR_BADARG;

    idx_t cap = F->row_adj_cap[row];
    idx_t count = F->row_adj_count[row];
    if (count >= cap) {
        /* Geometric growth (2x), starting at 4 for first-touch rows so
         * short row-adjacency lists do not pay a per-append reallocation
         * when the fill pattern is modest. */
        idx_t new_cap = 4;
        if (cap > 0) {
            if (cap > IDX_MAX / 2)
                return SPARSE_ERR_ALLOC;
            new_cap = cap * 2;
        }
        if ((size_t)new_cap > SIZE_MAX / sizeof(idx_t))
            return SPARSE_ERR_ALLOC;
        idx_t *resized = realloc(F->row_adj[row], (size_t)new_cap * sizeof(idx_t));
        if (!resized)
            return SPARSE_ERR_ALLOC;
        F->row_adj[row] = resized;
        F->row_adj_cap[row] = new_cap;
    }
    F->row_adj[row][count] = col;
    F->row_adj_count[row] = count + 1;
    return SPARSE_OK;
}

void ldlt_csc_row_adj_swap_slots(LdltCsc *F, idx_t i, idx_t j) {
    if (!F)
        return;

    /* A symmetric pivot swap renames row i <-> row j in already-factored
     * columns.  The row-adj lists are keyed by row, so swapping the three row
     * slots preserves the index without rebuilding it. */
    if (F->row_adj) {
        idx_t *tmp_ptr = F->row_adj[i];
        F->row_adj[i] = F->row_adj[j];
        F->row_adj[j] = tmp_ptr;
    }
    if (F->row_adj_count) {
        idx_t tmp_cnt = F->row_adj_count[i];
        F->row_adj_count[i] = F->row_adj_count[j];
        F->row_adj_count[j] = tmp_cnt;
    }
    if (F->row_adj_cap) {
        idx_t tmp_cap = F->row_adj_cap[i];
        F->row_adj_cap[i] = F->row_adj_cap[j];
        F->row_adj_cap[j] = tmp_cap;
    }
}

sparse_err_t ldlt_csc_populate_row_adj(LdltCsc *F, idx_t col) {
    idx_t cstart = F->L->col_ptr[col];
    idx_t cend = F->L->col_ptr[col + 1];
    for (idx_t p = cstart; p < cend; p++) {
        idx_t i = F->L->row_idx[p];
        if (i > col) {
            sparse_err_t err = ldlt_csc_row_adj_append(F, i, col);
            if (err != SPARSE_OK)
                return err;
        }
    }
    return SPARSE_OK;
}
