#include "sparse_lu_csr_internal.h"

#include <stdint.h>
#include <stdlib.h>

sparse_err_t lu_csr_grow(LuCsr *csr, idx_t needed) {
    if (needed < 0)
        return SPARSE_ERR_ALLOC;
    if (needed <= csr->capacity)
        return SPARSE_OK;

    idx_t new_cap;
    if (csr->capacity > IDX_MAX - csr->capacity / 2)
        new_cap = IDX_MAX;
    else
        new_cap = csr->capacity + csr->capacity / 2;
    if (new_cap < needed)
        new_cap = needed;
    if ((size_t)new_cap > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;

    idx_t *new_col = realloc(csr->col_idx, (size_t)new_cap * sizeof(idx_t));
    if (!new_col)
        return SPARSE_ERR_ALLOC;

    double *new_val = realloc(csr->values, (size_t)new_cap * sizeof(double));
    if (!new_val) {
        csr->col_idx = new_col;
        return SPARSE_ERR_ALLOC;
    }

    csr->col_idx = new_col;
    csr->values = new_val;
    csr->capacity = new_cap;
    return SPARSE_OK;
}

sparse_err_t lu_csr_validate(const LuCsr *csr) {
    idx_t n = csr->n;
    if (csr->nnz < 0 || csr->capacity < 0 || csr->nnz > csr->capacity)
        return SPARSE_ERR_BADARG;
    if (!csr->row_ptr || !csr->col_idx || !csr->values)
        return SPARSE_ERR_NULL;
    if (csr->row_ptr[0] != 0)
        return SPARSE_ERR_BADARG;
    for (idx_t i = 0; i < n; i++) {
        if (csr->row_ptr[i] < 0 || csr->row_ptr[i] > csr->row_ptr[i + 1])
            return SPARSE_ERR_BADARG;
    }
    if (csr->row_ptr[n] != csr->nnz)
        return SPARSE_ERR_BADARG;
    for (idx_t p = 0; p < csr->nnz; p++) {
        if (csr->col_idx[p] < 0 || csr->col_idx[p] >= n)
            return SPARSE_ERR_BADARG;
    }
    return SPARSE_OK;
}
