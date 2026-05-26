#include "sparse_matrix_internal.h"

#include <stdlib.h>
#include <string.h>

static sparse_err_t sparse_factor_state_bind(SparseMatrix *mat, sparse_factor_state_kind_t kind) {
    if (!mat)
        return SPARSE_ERR_NULL;
    if (!mat->factor_state) {
        mat->factor_state = malloc(sizeof(*mat->factor_state));
        if (!mat->factor_state)
            return SPARSE_ERR_ALLOC;
    }
    memset(mat->factor_state, 0, sizeof(*mat->factor_state));
    mat->factor_state->kind = kind;
    mat->factor_state->prev_factor_norm = mat->factor_norm;
    mat->factor_state->prev_factored = mat->factored;
    switch (kind) {
    case SPARSE_FACTOR_STATE_LU:
        mat->factor_state->state.lu.factor_norm = mat->factor_norm;
        mat->factor_state->state.lu.is_factored = mat->factored;
        break;
    case SPARSE_FACTOR_STATE_CHOLESKY:
        mat->factor_state->state.cholesky.factor_norm = mat->factor_norm;
        mat->factor_state->state.cholesky.is_factored = mat->factored;
        break;
    case SPARSE_FACTOR_STATE_NONE:
    default:
        break;
    }
    return SPARSE_OK;
}

sparse_err_t sparse_factor_state_bind_lu(SparseMatrix *mat) {
    return sparse_factor_state_bind(mat, SPARSE_FACTOR_STATE_LU);
}

sparse_err_t sparse_factor_state_bind_cholesky(SparseMatrix *mat) {
    return sparse_factor_state_bind(mat, SPARSE_FACTOR_STATE_CHOLESKY);
}

sparse_err_t sparse_factor_state_begin_lu(SparseMatrix *mat) {
    sparse_err_t err = sparse_factor_state_bind_lu(mat);
    if (err != SPARSE_OK)
        return err;
    sparse_factor_state_set_factored(mat, 0);
    return SPARSE_OK;
}

sparse_err_t sparse_factor_state_begin_cholesky(SparseMatrix *mat) {
    sparse_err_t err = sparse_factor_state_bind_cholesky(mat);
    if (err != SPARSE_OK)
        return err;
    sparse_factor_state_set_factored(mat, 0);
    return SPARSE_OK;
}

void sparse_factor_state_set_factored(SparseMatrix *mat, int is_factored) {
    if (!mat)
        return;
    mat->factored = is_factored ? 1 : 0;
    if (!mat->factor_state)
        return;
    switch (mat->factor_state->kind) {
    case SPARSE_FACTOR_STATE_LU:
        mat->factor_state->state.lu.is_factored = mat->factored;
        break;
    case SPARSE_FACTOR_STATE_CHOLESKY:
        mat->factor_state->state.cholesky.is_factored = mat->factored;
        break;
    case SPARSE_FACTOR_STATE_NONE:
    default:
        break;
    }
}

void sparse_factor_state_set_factor_norm(SparseMatrix *mat, double factor_norm) {
    if (!mat)
        return;
    mat->factor_norm = factor_norm;
    if (!mat->factor_state)
        return;
    switch (mat->factor_state->kind) {
    case SPARSE_FACTOR_STATE_LU:
        mat->factor_state->state.lu.factor_norm = factor_norm;
        break;
    case SPARSE_FACTOR_STATE_CHOLESKY:
        mat->factor_state->state.cholesky.factor_norm = factor_norm;
        break;
    case SPARSE_FACTOR_STATE_NONE:
    default:
        break;
    }
}

void sparse_factor_state_replace_reorder_perm(SparseMatrix *mat, idx_t *perm) {
    if (!mat) {
        free(perm);
        return;
    }
    free(mat->reorder_perm);
    mat->reorder_perm = perm;
}

void sparse_factor_state_publish_factored(SparseMatrix *mat, double factor_norm, idx_t *perm) {
    if (!mat) {
        free(perm);
        return;
    }
    sparse_factor_state_replace_reorder_perm(mat, perm);
    sparse_factor_state_set_factor_norm(mat, factor_norm);
    sparse_factor_state_set_factored(mat, 1);
}

void sparse_factor_state_restore_compat(SparseMatrix *mat) {
    if (!mat || !mat->factor_state)
        return;
    mat->factor_norm = mat->factor_state->prev_factor_norm;
    mat->factored = mat->factor_state->prev_factored;
    switch (mat->factor_state->kind) {
    case SPARSE_FACTOR_STATE_LU:
        mat->factor_state->state.lu.factor_norm = mat->factor_norm;
        mat->factor_state->state.lu.is_factored = mat->factored;
        break;
    case SPARSE_FACTOR_STATE_CHOLESKY:
        mat->factor_state->state.cholesky.factor_norm = mat->factor_norm;
        mat->factor_state->state.cholesky.is_factored = mat->factored;
        break;
    case SPARSE_FACTOR_STATE_NONE:
    default:
        break;
    }
}

int sparse_factor_state_is_factored(const SparseMatrix *mat) {
    if (!mat || !mat->factor_state)
        return mat ? mat->factored : 0;
    switch (mat->factor_state->kind) {
    case SPARSE_FACTOR_STATE_LU:
        return mat->factor_state->state.lu.is_factored;
    case SPARSE_FACTOR_STATE_CHOLESKY:
        return mat->factor_state->state.cholesky.is_factored;
    case SPARSE_FACTOR_STATE_NONE:
    default:
        return mat->factored;
    }
}

double sparse_factor_state_factor_norm(const SparseMatrix *mat) {
    if (!mat || !mat->factor_state)
        return mat ? mat->factor_norm : -1.0;
    switch (mat->factor_state->kind) {
    case SPARSE_FACTOR_STATE_LU:
        return mat->factor_state->state.lu.factor_norm;
    case SPARSE_FACTOR_STATE_CHOLESKY:
        return mat->factor_state->state.cholesky.factor_norm;
    case SPARSE_FACTOR_STATE_NONE:
    default:
        return mat->factor_norm;
    }
}

void sparse_factor_state_clear(SparseMatrix *mat) {
    if (!mat)
        return;
    free(mat->factor_state);
    mat->factor_state = NULL;
    mat->factored = 0;
    mat->factor_norm = -1.0;
}

sparse_err_t sparse_factor_state_clone(SparseMatrix *dst, const SparseMatrix *src) {
    if (!dst || !src)
        return SPARSE_ERR_NULL;
    if (!src->factor_state) {
        dst->factor_state = NULL;
        return SPARSE_OK;
    }
    dst->factor_state = malloc(sizeof(*dst->factor_state));
    if (!dst->factor_state)
        return SPARSE_ERR_ALLOC;
    memcpy(dst->factor_state, src->factor_state, sizeof(*dst->factor_state));
    return SPARSE_OK;
}
