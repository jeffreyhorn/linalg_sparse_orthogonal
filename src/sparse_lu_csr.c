#include "sparse_lu_csr.h"
#include "sparse_matrix_internal.h"

#include <stdlib.h>
#include <string.h>

/* ─── Free ───────────────────────────────────────────────────────────── */

void lu_csr_free(LuCsr *csr) {
    if (!csr)
        return;
    free(csr->row_ptr);
    free(csr->col_idx);
    free(csr->values);
    free(csr);
}

/* ─── Convert SparseMatrix → LuCsr (logical index order) ────────────── */

sparse_err_t lu_csr_from_sparse(const SparseMatrix *mat, double fill_factor, LuCsr **csr_out) {
    if (!csr_out)
        return SPARSE_ERR_NULL;
    *csr_out = NULL;
    if (!mat)
        return SPARSE_ERR_NULL;

    idx_t n = mat->rows;
    if (n != mat->cols)
        return SPARSE_ERR_SHAPE;

    /* Clamp fill factor */
    if (fill_factor < 1.0)
        fill_factor = 1.0;
    if (fill_factor > 20.0)
        fill_factor = 20.0;

    idx_t nnz = mat->nnz;
    idx_t cap = (idx_t)(nnz * fill_factor);
    if (cap < nnz)
        cap = nnz; /* overflow guard */
    if (cap < 1)
        cap = 1;

    LuCsr *csr = malloc(sizeof(LuCsr));
    if (!csr)
        return SPARSE_ERR_ALLOC;

    csr->n = n;
    csr->nnz = nnz;
    csr->capacity = cap;
    csr->row_ptr = malloc((size_t)(n + 1) * sizeof(idx_t));
    csr->col_idx = malloc((size_t)cap * sizeof(idx_t));
    csr->values = malloc((size_t)cap * sizeof(double));

    if (!csr->row_ptr || !csr->col_idx || !csr->values) {
        lu_csr_free(csr);
        return SPARSE_ERR_ALLOC;
    }

    /*
     * First pass: count nonzeros per logical row.
     * We traverse each physical row and map to logical row via inv_row_perm.
     */
    memset(csr->row_ptr, 0, (size_t)(n + 1) * sizeof(idx_t));

    for (idx_t phys_i = 0; phys_i < n; phys_i++) {
        idx_t log_i = mat->inv_row_perm[phys_i];
        Node *node = mat->row_headers[phys_i];
        while (node) {
            (void)mat->inv_col_perm[node->col]; /* col mapping used in 2nd pass */
            csr->row_ptr[log_i + 1]++;
            node = node->right;
        }
    }

    /* Prefix sum to get row pointers */
    for (idx_t i = 0; i < n; i++)
        csr->row_ptr[i + 1] += csr->row_ptr[i];

    /*
     * Second pass: fill col_idx and values in logical order.
     * Use a temporary write-position array (reuse beginning of col_idx
     * would be tricky, so allocate a small temp array).
     */
    idx_t *write_pos = malloc((size_t)n * sizeof(idx_t));
    if (!write_pos) {
        lu_csr_free(csr);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t i = 0; i < n; i++)
        write_pos[i] = csr->row_ptr[i];

    for (idx_t phys_i = 0; phys_i < n; phys_i++) {
        idx_t log_i = mat->inv_row_perm[phys_i];
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            idx_t pos = write_pos[log_i]++;
            csr->col_idx[pos] = log_j;
            csr->values[pos] = node->value;
            node = node->right;
        }
    }
    free(write_pos);

    /*
     * Sort each row by logical column index. The linked-list rows are sorted
     * by physical column, but after applying inv_col_perm the logical columns
     * may be out of order (e.g., after complete pivoting).
     * Use insertion sort since rows are typically short and nearly sorted.
     */
    for (idx_t i = 0; i < n; i++) {
        idx_t start = csr->row_ptr[i];
        idx_t end = csr->row_ptr[i + 1];
        for (idx_t j = start + 1; j < end; j++) {
            idx_t key_col = csr->col_idx[j];
            double key_val = csr->values[j];
            idx_t k = j - 1;
            while (k >= start &&
                   csr->col_idx[k] > key_col) { // NOLINT(clang-analyzer-security.ArrayBound)
                csr->col_idx[k + 1] = csr->col_idx[k];
                csr->values[k + 1] = csr->values[k];
                k--;
            }
            csr->col_idx[k + 1] = key_col;
            csr->values[k + 1] = key_val;
        }
    }

    *csr_out = csr;
    return SPARSE_OK;
}

/* ─── Convert LuCsr → SparseMatrix ──────────────────────────────────── */

sparse_err_t lu_csr_to_sparse(const LuCsr *csr, SparseMatrix **mat_out) {
    if (!mat_out)
        return SPARSE_ERR_NULL;
    *mat_out = NULL;
    if (!csr)
        return SPARSE_ERR_NULL;

    idx_t n = csr->n;
    SparseMatrix *mat = sparse_create(n, n);
    if (!mat)
        return SPARSE_ERR_ALLOC;

    for (idx_t i = 0; i < n; i++) {
        for (idx_t k = csr->row_ptr[i]; k < csr->row_ptr[i + 1]; k++) {
            /* Skip exact zeros (dropped entries) */
            if (csr->values[k] == 0.0)
                continue;
            sparse_err_t err = sparse_insert(mat, i, csr->col_idx[k], csr->values[k]);
            if (err != SPARSE_OK) {
                sparse_free(mat);
                return err;
            }
        }
    }

    *mat_out = mat;
    return SPARSE_OK;
}
