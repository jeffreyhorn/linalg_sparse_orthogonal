#include "sparse_iterative_workspace_internal.h"

#include "sparse_alloc_internal.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void sparse_iter_workspace_init(sparse_iter_workspace_t *ws) {
    if (ws)
        *ws = (sparse_iter_workspace_t){0};
}

void sparse_iter_workspace_free(sparse_iter_workspace_t *ws) {
    if (!ws)
        return;
    free(ws->double_mem);
    free(ws->int_mem);
    *ws = (sparse_iter_workspace_t){0};
}

static sparse_err_t sparse_iter_workspace_reserve_doubles(sparse_iter_workspace_t *ws,
                                                          size_t count) {
    if (!ws)
        return SPARSE_ERR_NULL;
    if (count <= ws->double_capacity)
        return SPARSE_OK;

    double *new_mem = NULL;
    sparse_err_t err = sparse_calloc_array(count, sizeof(double), (void **)&new_mem);
    if (err != SPARSE_OK)
        return err;

    free(ws->double_mem);
    ws->double_mem = new_mem;
    ws->double_capacity = count;
    return SPARSE_OK;
}

static sparse_err_t sparse_iter_workspace_reserve_ints(sparse_iter_workspace_t *ws, size_t count) {
    if (!ws)
        return SPARSE_ERR_NULL;
    if (count <= ws->int_capacity)
        return SPARSE_OK;

    int *new_mem = NULL;
    sparse_err_t err = sparse_calloc_array(count, sizeof(int), (void **)&new_mem);
    if (err != SPARSE_OK)
        return err;

    free(ws->int_mem);
    ws->int_mem = new_mem;
    ws->int_capacity = count;
    return SPARSE_OK;
}

static void sparse_iter_workspace_zero_doubles(sparse_iter_workspace_t *ws, size_t count) {
    if (ws && ws->double_mem && count > 0)
        memset(ws->double_mem, 0, count * sizeof(double));
}

static void sparse_iter_workspace_zero_ints(sparse_iter_workspace_t *ws, size_t count) {
    if (ws && ws->int_mem && count > 0)
        memset(ws->int_mem, 0, count * sizeof(int));
}

sparse_err_t sparse_iter_workspace_prepare_cg(sparse_iter_workspace_t *ws, idx_t n,
                                              sparse_cg_workspace_view_t *view) {
    if (!ws || !view)
        return SPARSE_ERR_NULL;

    size_t n_size = 0;
    size_t total = 0;
    if (sparse_idx_to_size_checked(n, &n_size) || sparse_size_mul_overflow(n_size, 4, &total))
        return SPARSE_ERR_ALLOC;

    sparse_err_t err = sparse_iter_workspace_reserve_doubles(ws, total);
    if (err != SPARSE_OK)
        return err;

    view->r = ws->double_mem;
    view->z = view->r + n_size;
    view->p = view->z + n_size;
    view->Ap = view->p + n_size;

    ws->n_capacity = n;
    return SPARSE_OK;
}

sparse_err_t sparse_iter_workspace_prepare_gmres(sparse_iter_workspace_t *ws, idx_t n,
                                                 idx_t restart,
                                                 sparse_gmres_workspace_view_t *view) {
    if (!ws || !view)
        return SPARSE_ERR_NULL;
    if (restart <= 0)
        return SPARSE_ERR_BADARG;

    size_t n_size = 0;
    size_t m_size = 0;
    size_t m1_size = 0;
    if (sparse_idx_to_size_checked(n, &n_size) || sparse_idx_to_size_checked(restart, &m_size) ||
        sparse_size_add_overflow(m_size, 1, &m1_size))
        return SPARSE_ERR_ALLOC;

    size_t sz_v = 0;
    size_t sz_h = 0;
    if (sparse_size_mul_overflow(m1_size, n_size, &sz_v) ||
        sparse_size_mul_overflow(m1_size, m_size, &sz_h))
        return SPARSE_ERR_ALLOC;

    size_t total = 0;
    size_t pieces[] = {sz_v, sz_h, m_size, m_size, m1_size, m_size, n_size};
    for (size_t i = 0; i < sizeof(pieces) / sizeof(pieces[0]); i++) {
        if (sparse_size_add_overflow(total, pieces[i], &total))
            return SPARSE_ERR_ALLOC;
    }

    sparse_err_t err = sparse_iter_workspace_reserve_doubles(ws, total);
    if (err != SPARSE_OK)
        return err;

    view->v = ws->double_mem;
    view->h = view->v + sz_v;
    view->cs = view->h + sz_h;
    view->sn = view->cs + m_size;
    view->g = view->sn + m_size;
    view->y = view->g + m1_size;
    view->w = view->y + m_size;

    ws->n_capacity = n;
    ws->restart_capacity = restart;
    return SPARSE_OK;
}

sparse_err_t sparse_iter_workspace_prepare_block_cg(sparse_iter_workspace_t *ws, idx_t n,
                                                    idx_t nrhs,
                                                    sparse_block_cg_workspace_view_t *view) {
    if (!ws || !view)
        return SPARSE_ERR_NULL;
    if (nrhs < 0)
        return SPARSE_ERR_BADARG;

    size_t n_size = 0;
    size_t nrhs_size = 0;
    size_t blk = 0;
    if (sparse_idx_to_size_checked(n, &n_size) || sparse_idx_to_size_checked(nrhs, &nrhs_size) ||
        sparse_size_mul_overflow(n_size, nrhs_size, &blk))
        return SPARSE_ERR_ALLOC;

    size_t total = 0;
    size_t pieces[] = {blk, blk, blk, blk, nrhs_size, nrhs_size, nrhs_size};
    for (size_t i = 0; i < sizeof(pieces) / sizeof(pieces[0]); i++) {
        if (sparse_size_add_overflow(total, pieces[i], &total))
            return SPARSE_ERR_ALLOC;
    }

    sparse_err_t err = sparse_iter_workspace_reserve_doubles(ws, total);
    if (err != SPARSE_OK)
        return err;
    err = sparse_iter_workspace_reserve_ints(ws, nrhs_size);
    if (err != SPARSE_OK)
        return err;

    view->R = ws->double_mem;
    view->Z = view->R + blk;
    view->P = view->Z + blk;
    view->AP = view->P + blk;
    view->bnorms = view->AP + blk;
    view->rz = view->bnorms + nrhs_size;
    view->rnorms = view->rz + nrhs_size;
    view->conv = ws->int_mem;

    sparse_iter_workspace_zero_ints(ws, nrhs_size);

    ws->n_capacity = n;
    ws->nrhs_capacity = nrhs;
    return SPARSE_OK;
}

sparse_err_t sparse_iter_workspace_prepare_minres(sparse_iter_workspace_t *ws, idx_t n,
                                                  int with_precond,
                                                  sparse_minres_workspace_view_t *view) {
    if (!ws || !view)
        return SPARSE_ERR_NULL;

    size_t n_size = 0;
    if (sparse_idx_to_size_checked(n, &n_size))
        return SPARSE_ERR_ALLOC;

    size_t nvecs = with_precond ? 8U : 6U;
    size_t total = 0;
    if (sparse_size_mul_overflow(n_size, nvecs, &total))
        return SPARSE_ERR_ALLOC;

    sparse_err_t err = sparse_iter_workspace_reserve_doubles(ws, total);
    if (err != SPARSE_OK)
        return err;
    sparse_iter_workspace_zero_doubles(ws, total);

    view->v = ws->double_mem;
    view->v_old = view->v + n_size;
    view->w = view->v_old + n_size;
    view->d0 = view->w + n_size;
    view->d1 = view->d0 + n_size;
    view->d2 = view->d1 + n_size;
    if (with_precond) {
        view->z = view->d2 + n_size;
        view->z_tmp = view->z + n_size;
    } else {
        view->z = NULL;
        view->z_tmp = NULL;
    }

    ws->n_capacity = n;
    return SPARSE_OK;
}
