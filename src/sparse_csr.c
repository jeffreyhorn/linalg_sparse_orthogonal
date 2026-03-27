#include "sparse_csr.h"
#include "sparse_matrix_internal.h"

#include <stdlib.h>
#include <string.h>

/* ─── CSR free ───────────────────────────────────────────────────────── */

void sparse_csr_free(SparseCsr *csr) {
    if (!csr)
        return;
    free(csr->row_ptr);
    free(csr->col_idx);
    free(csr->values);
    free(csr);
}

/* ─── CSC free ───────────────────────────────────────────────────────── */

void sparse_csc_free(SparseCsc *csc) {
    if (!csc)
        return;
    free(csc->col_ptr);
    free(csc->row_idx);
    free(csc->values);
    free(csc);
}

/* ─── To CSR ─────────────────────────────────────────────────────────── */

sparse_err_t sparse_to_csr(const SparseMatrix *mat, SparseCsr **csr_out) {
    if (!csr_out)
        return SPARSE_ERR_NULL;
    *csr_out = NULL;
    if (!mat)
        return SPARSE_ERR_NULL;

    idx_t m = mat->rows;
    idx_t nc = mat->cols;
    idx_t nz = mat->nnz;

    SparseCsr *csr = malloc(sizeof(SparseCsr));
    if (!csr)
        return SPARSE_ERR_ALLOC;

    csr->rows = m;
    csr->cols = nc;
    csr->nnz = nz;
    csr->row_ptr = malloc((size_t)(m + 1) * sizeof(idx_t));
    csr->col_idx = malloc((size_t)(nz > 0 ? nz : 1) * sizeof(idx_t));
    csr->values = malloc((size_t)(nz > 0 ? nz : 1) * sizeof(double));

    if (!csr->row_ptr || !csr->col_idx || !csr->values) {
        sparse_csr_free(csr);
        return SPARSE_ERR_ALLOC;
    }

    /* Walk each row's linked list (already sorted by column) */
    idx_t pos = 0;
    for (idx_t i = 0; i < m; i++) {
        csr->row_ptr[i] = pos;
        Node *node = mat->row_headers[i];
        while (node) {
            csr->col_idx[pos] = node->col;
            csr->values[pos] = node->value;
            pos++;
            node = node->right;
        }
    }
    csr->row_ptr[m] = pos;

    *csr_out = csr;
    return SPARSE_OK;
}

/* ─── From CSR ───────────────────────────────────────────────────────── */

sparse_err_t sparse_from_csr(const SparseCsr *csr, SparseMatrix **mat_out) {
    if (!mat_out)
        return SPARSE_ERR_NULL;
    *mat_out = NULL;
    if (!csr)
        return SPARSE_ERR_NULL;

    if (!csr->row_ptr || (!csr->col_idx && csr->nnz > 0) || (!csr->values && csr->nnz > 0))
        return SPARSE_ERR_BADARG;

    idx_t m = csr->rows;
    idx_t nc = csr->cols;

    if (csr->row_ptr[0] != 0)
        return SPARSE_ERR_BADARG;

    /* Validate: row_ptr must be monotonically non-decreasing and in [0, nnz] */
    for (idx_t i = 0; i <= m; i++) {
        if (csr->row_ptr[i] < 0 || csr->row_ptr[i] > csr->nnz)
            return SPARSE_ERR_BADARG;
        if (i > 0 && csr->row_ptr[i] < csr->row_ptr[i - 1])
            return SPARSE_ERR_BADARG;
    }
    if (csr->row_ptr[m] != csr->nnz)
        return SPARSE_ERR_BADARG;

    /* Validate: col_idx in range */
    for (idx_t k = 0; k < csr->nnz; k++) {
        if (csr->col_idx[k] < 0 || csr->col_idx[k] >= nc)
            return SPARSE_ERR_BADARG;
    }

    /* Validate: col_idx within each row must be strictly increasing (no duplicates) */
    for (idx_t i = 0; i < m; i++) {
        for (idx_t k = csr->row_ptr[i] + 1; k < csr->row_ptr[i + 1]; k++) {
            if (csr->col_idx[k] <= csr->col_idx[k - 1])
                return SPARSE_ERR_BADARG;
        }
    }

    SparseMatrix *mat = sparse_create(m, nc);
    if (!mat)
        return SPARSE_ERR_ALLOC;

    for (idx_t i = 0; i < m; i++) {
        for (idx_t k = csr->row_ptr[i]; k < csr->row_ptr[i + 1]; k++) {
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

/* ─── To CSC ─────────────────────────────────────────────────────────── */

sparse_err_t sparse_to_csc(const SparseMatrix *mat, SparseCsc **csc_out) {
    if (!csc_out)
        return SPARSE_ERR_NULL;
    *csc_out = NULL;
    if (!mat)
        return SPARSE_ERR_NULL;

    idx_t m = mat->rows;
    idx_t nc = mat->cols;
    idx_t nz = mat->nnz;

    SparseCsc *csc = malloc(sizeof(SparseCsc));
    if (!csc)
        return SPARSE_ERR_ALLOC;

    csc->rows = m;
    csc->cols = nc;
    csc->nnz = nz;
    csc->col_ptr = malloc((size_t)(nc + 1) * sizeof(idx_t));
    csc->row_idx = malloc((size_t)(nz > 0 ? nz : 1) * sizeof(idx_t));
    csc->values = malloc((size_t)(nz > 0 ? nz : 1) * sizeof(double));

    if (!csc->col_ptr || !csc->row_idx || !csc->values) {
        sparse_csc_free(csc);
        return SPARSE_ERR_ALLOC;
    }

    /* Walk each column's linked list (already sorted by row) */
    idx_t pos = 0;
    for (idx_t j = 0; j < nc; j++) {
        csc->col_ptr[j] = pos;
        Node *node = mat->col_headers[j];
        while (node) {
            csc->row_idx[pos] = node->row;
            csc->values[pos] = node->value;
            pos++;
            node = node->down;
        }
    }
    csc->col_ptr[nc] = pos;

    *csc_out = csc;
    return SPARSE_OK;
}

/* ─── From CSC ───────────────────────────────────────────────────────── */

sparse_err_t sparse_from_csc(const SparseCsc *csc, SparseMatrix **mat_out) {
    if (!mat_out)
        return SPARSE_ERR_NULL;
    *mat_out = NULL;
    if (!csc)
        return SPARSE_ERR_NULL;

    if (!csc->col_ptr || (!csc->row_idx && csc->nnz > 0) || (!csc->values && csc->nnz > 0))
        return SPARSE_ERR_BADARG;

    idx_t m = csc->rows;
    idx_t nc = csc->cols;

    if (csc->col_ptr[0] != 0)
        return SPARSE_ERR_BADARG;

    /* Validate: col_ptr must be monotonically non-decreasing and in [0, nnz] */
    for (idx_t j = 0; j <= nc; j++) {
        if (csc->col_ptr[j] < 0 || csc->col_ptr[j] > csc->nnz)
            return SPARSE_ERR_BADARG;
        if (j > 0 && csc->col_ptr[j] < csc->col_ptr[j - 1])
            return SPARSE_ERR_BADARG;
    }
    if (csc->col_ptr[nc] != csc->nnz)
        return SPARSE_ERR_BADARG;

    /* Validate: row_idx in range */
    for (idx_t k = 0; k < csc->nnz; k++) {
        if (csc->row_idx[k] < 0 || csc->row_idx[k] >= m)
            return SPARSE_ERR_BADARG;
    }

    /* Validate: row_idx within each column must be strictly increasing (no duplicates) */
    for (idx_t j = 0; j < nc; j++) {
        for (idx_t k = csc->col_ptr[j] + 1; k < csc->col_ptr[j + 1]; k++) {
            if (csc->row_idx[k] <= csc->row_idx[k - 1])
                return SPARSE_ERR_BADARG;
        }
    }

    SparseMatrix *mat = sparse_create(m, nc);
    if (!mat)
        return SPARSE_ERR_ALLOC;

    for (idx_t j = 0; j < nc; j++) {
        for (idx_t k = csc->col_ptr[j]; k < csc->col_ptr[j + 1]; k++) {
            sparse_err_t err = sparse_insert(mat, csc->row_idx[k], j, csc->values[k]);
            if (err != SPARSE_OK) {
                sparse_free(mat);
                return err;
            }
        }
    }

    *mat_out = mat;
    return SPARSE_OK;
}
