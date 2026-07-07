#include "sparse_alloc_internal.h"
#include "sparse_matrix_internal.h"

#include <stdlib.h>

static int sparse_build_entry_cmp(const void *lhs, const void *rhs) {
    const SparseBuildEntry *a = lhs;
    const SparseBuildEntry *b = rhs;
    if (a->row < b->row)
        return -1;
    if (a->row > b->row)
        return 1;
    if (a->col < b->col)
        return -1;
    if (a->col > b->col)
        return 1;
    if (a->order < b->order)
        return -1;
    if (a->order > b->order)
        return 1;
    return 0;
}

sparse_err_t sparse_matrix_build_from_entries(idx_t rows, idx_t cols, SparseBuildEntry *entries,
                                              idx_t nentries, int entries_sorted,
                                              SparseMatrix **mat_out) {
    Node **row_tails = NULL;
    Node **col_tails = NULL;
    SparseMatrix *mat = NULL;
    sparse_err_t err = SPARSE_OK;

    if (!mat_out)
        return SPARSE_ERR_NULL;
    *mat_out = NULL;
    if (nentries < 0)
        return SPARSE_ERR_ALLOC;
    if (!entries && nentries > 0)
        return SPARSE_ERR_NULL;

    mat = sparse_create(rows, cols);
    if (!mat)
        return SPARSE_ERR_ALLOC;
    if (nentries == 0) {
        *mat_out = mat;
        return SPARSE_OK;
    }

    if (!entries_sorted) {
        size_t qsort_count = 0;
        if (sparse_idx_to_size_checked(nentries, &qsort_count)) {
            sparse_free(mat);
            return SPARSE_ERR_ALLOC;
        }
        qsort(entries, qsort_count, sizeof(*entries), sparse_build_entry_cmp);
    }

    if (sparse_calloc_idx_array(rows, sizeof(Node *), (void **)&row_tails) != SPARSE_OK ||
        sparse_calloc_idx_array(cols, sizeof(Node *), (void **)&col_tails) != SPARSE_OK) {
        err = SPARSE_ERR_ALLOC;
        goto fail;
    }

    for (idx_t pos = 0; pos < nentries;) {
        idx_t row = entries[pos].row;
        idx_t col = entries[pos].col;
        sparse_scalar_t value = entries[pos].value;
        idx_t next = pos + 1;

        while (next < nentries && entries[next].row == row && entries[next].col == col) {
            value = entries[next].value;
            next++;
        }

        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            err = SPARSE_ERR_BOUNDS;
            goto fail;
        }
        if (value != 0.0) {
            Node *node = sparse_matrix_make_node(mat, row, col, value);
            if (!node) {
                err = SPARSE_ERR_ALLOC;
                goto fail;
            }
            if (row_tails[row])
                row_tails[row]->right = node;
            else
                mat->row_headers[row] = node;
            row_tails[row] = node;

            if (col_tails[col])
                col_tails[col]->down = node;
            else
                mat->col_headers[col] = node;
            col_tails[col] = node;
            mat->nnz++;
        }

        pos = next;
    }

    free(row_tails);
    free(col_tails);
    *mat_out = mat;
    return SPARSE_OK;

fail:
    free(row_tails);
    free(col_tails);
    sparse_free(mat);
    return err;
}
