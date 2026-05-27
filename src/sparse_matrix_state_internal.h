#ifndef SPARSE_MATRIX_STATE_INTERNAL_H
#define SPARSE_MATRIX_STATE_INTERNAL_H

#include "sparse_matrix_internal.h"

static inline int sparse_matrix_has_identity_row_col_perms(const SparseMatrix *mat) {
    if (!mat)
        return 0;
    for (idx_t i = 0; i < mat->rows; i++) {
        if (mat->row_perm && mat->row_perm[i] != i)
            return 0;
    }
    for (idx_t i = 0; i < mat->cols; i++) {
        if (mat->col_perm && mat->col_perm[i] != i)
            return 0;
    }
    return 1;
}

static inline int sparse_matrix_has_identity_perms(const SparseMatrix *mat) {
    if (!mat)
        return 0;
    for (idx_t i = 0; i < mat->rows; i++) {
        if ((mat->row_perm && mat->row_perm[i] != i) ||
            (mat->inv_row_perm && mat->inv_row_perm[i] != i))
            return 0;
    }
    for (idx_t i = 0; i < mat->cols; i++) {
        if ((mat->col_perm && mat->col_perm[i] != i) ||
            (mat->inv_col_perm && mat->inv_col_perm[i] != i))
            return 0;
    }
    return 1;
}

static inline sparse_err_t sparse_matrix_require_original_row_col_state(const SparseMatrix *mat) {
    if (!mat)
        return SPARSE_ERR_NULL;
    return (!sparse_factor_state_is_factored(mat) && sparse_matrix_has_identity_row_col_perms(mat))
               ? SPARSE_OK
               : SPARSE_ERR_BADARG;
}

static inline sparse_err_t sparse_matrix_require_original_state(const SparseMatrix *mat) {
    if (!mat)
        return SPARSE_ERR_NULL;
    return (!sparse_factor_state_is_factored(mat) && sparse_matrix_has_identity_perms(mat))
               ? SPARSE_OK
               : SPARSE_ERR_BADARG;
}

static inline sparse_err_t sparse_matrix_require_factored_state(const SparseMatrix *mat) {
    if (!mat)
        return SPARSE_ERR_NULL;
    return sparse_factor_state_is_factored(mat) ? SPARSE_OK : SPARSE_ERR_BADARG;
}

#endif
