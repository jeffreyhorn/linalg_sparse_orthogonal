#include "sparse_eigs_workspace_internal.h"

#include "sparse_alloc_internal.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void sparse_eigs_workspace_init(sparse_eigs_workspace_t *ws) {
    if (ws)
        *ws = (sparse_eigs_workspace_t){0};
}

void sparse_eigs_workspace_free(sparse_eigs_workspace_t *ws) {
    if (!ws)
        return;
    free(ws->double_mem);
    free(ws->idx_mem);
    free(ws->int_mem);
    *ws = (sparse_eigs_workspace_t){0};
}

static sparse_err_t sparse_eigs_workspace_reserve_doubles(sparse_eigs_workspace_t *ws,
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

static sparse_err_t sparse_eigs_workspace_reserve_idxs(sparse_eigs_workspace_t *ws, size_t count) {
    if (!ws)
        return SPARSE_ERR_NULL;
    if (count <= ws->idx_capacity)
        return SPARSE_OK;

    idx_t *new_mem = NULL;
    sparse_err_t err = sparse_malloc_array(count, sizeof(idx_t), (void **)&new_mem);
    if (err != SPARSE_OK)
        return err;

    free(ws->idx_mem);
    ws->idx_mem = new_mem;
    ws->idx_capacity = count;
    return SPARSE_OK;
}

static sparse_err_t sparse_eigs_workspace_reserve_ints(sparse_eigs_workspace_t *ws, size_t count) {
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

static void sparse_eigs_workspace_zero_doubles(sparse_eigs_workspace_t *ws, size_t count) {
    if (ws && ws->double_mem && count > 0)
        memset(ws->double_mem, 0, count * sizeof(double));
}

static void sparse_eigs_workspace_zero_ints(sparse_eigs_workspace_t *ws, size_t count) {
    if (ws && ws->int_mem && count > 0)
        memset(ws->int_mem, 0, count * sizeof(int));
}

sparse_err_t sparse_eigs_workspace_prepare_growm(sparse_eigs_workspace_t *ws, idx_t n, idx_t m_cap,
                                                 idx_t k,
                                                 sparse_eigs_growm_workspace_view_t *view) {
    if (!ws || !view)
        return SPARSE_ERR_NULL;
    if (n < 1 || m_cap < 1 || k < 1)
        return SPARSE_ERR_BADARG;

    size_t n_size = 0;
    size_t m_cap_size = 0;
    size_t k_size = 0;
    if (sparse_idx_to_size_checked(n, &n_size) || sparse_idx_to_size_checked(m_cap, &m_cap_size) ||
        sparse_idx_to_size_checked(k, &k_size))
        return SPARSE_ERR_ALLOC;

    size_t v_elems = 0;
    size_t y_elems = 0;
    if (sparse_size_mul_overflow(n_size, m_cap_size, &v_elems) ||
        sparse_size_mul_overflow(m_cap_size, m_cap_size, &y_elems))
        return SPARSE_ERR_ALLOC;

    size_t total = 0;
    size_t pieces[] = {v_elems, m_cap_size, m_cap_size, n_size, m_cap_size, m_cap_size, y_elems};
    for (size_t i = 0; i < sizeof(pieces) / sizeof(pieces[0]); i++) {
        if (sparse_size_add_overflow(total, pieces[i], &total))
            return SPARSE_ERR_ALLOC;
    }

    sparse_err_t err = sparse_eigs_workspace_reserve_doubles(ws, total);
    if (err != SPARSE_OK)
        return err;
    err = sparse_eigs_workspace_reserve_idxs(ws, k_size);
    if (err != SPARSE_OK)
        return err;

    sparse_eigs_workspace_zero_doubles(ws, total);

    view->V = ws->double_mem;
    view->alpha = view->V + v_elems;
    view->beta = view->alpha + m_cap_size;
    view->v0 = view->beta + m_cap_size;
    view->theta_long = view->v0 + n_size;
    view->subdiag = view->theta_long + m_cap_size;
    view->Y_long = view->subdiag + m_cap_size;
    view->sel_idx = ws->idx_mem;

    ws->n_capacity = n;
    ws->lanczos_capacity = m_cap;
    ws->block_capacity = k;
    return SPARSE_OK;
}

sparse_err_t
sparse_eigs_workspace_prepare_thick_restart(sparse_eigs_workspace_t *ws, idx_t n, idx_t m_restart,
                                            idx_t k,
                                            sparse_eigs_thick_restart_workspace_view_t *view) {
    if (!ws || !view)
        return SPARSE_ERR_NULL;
    if (n < 1 || m_restart < 1 || k < 1)
        return SPARSE_ERR_BADARG;

    size_t n_size = 0;
    size_t m_restart_size = 0;
    size_t k_size = 0;
    if (sparse_idx_to_size_checked(n, &n_size) ||
        sparse_idx_to_size_checked(m_restart, &m_restart_size) ||
        sparse_idx_to_size_checked(k, &k_size))
        return SPARSE_ERR_ALLOC;

    size_t v_elems = 0;
    size_t K2 = 0;
    size_t vk_elems = 0;
    if (sparse_size_mul_overflow(n_size, m_restart_size, &v_elems) ||
        sparse_size_mul_overflow(m_restart_size, m_restart_size, &K2) ||
        sparse_size_mul_overflow(n_size, k_size, &vk_elems))
        return SPARSE_ERR_ALLOC;

    size_t total = 0;
    size_t pieces[] = {v_elems, m_restart_size, m_restart_size, n_size, n_size, K2, m_restart_size,
                       K2,      k_size,         vk_elems,       k_size, k_size};
    for (size_t i = 0; i < sizeof(pieces) / sizeof(pieces[0]); i++) {
        if (sparse_size_add_overflow(total, pieces[i], &total))
            return SPARSE_ERR_ALLOC;
    }

    sparse_err_t err = sparse_eigs_workspace_reserve_doubles(ws, total);
    if (err != SPARSE_OK)
        return err;
    err = sparse_eigs_workspace_reserve_idxs(ws, k_size);
    if (err != SPARSE_OK)
        return err;

    sparse_eigs_workspace_zero_doubles(ws, total);

    view->V = ws->double_mem;
    view->alpha = view->V + v_elems;
    view->beta = view->alpha + m_restart_size;
    view->v0 = view->beta + m_restart_size;
    view->residual_vec = view->v0 + n_size;
    view->T_arrow = view->residual_vec + n_size;
    view->theta_arrow = view->T_arrow + K2;
    view->Y_arrow = view->theta_arrow + m_restart_size;
    view->sel_idx = ws->idx_mem;
    view->V_locked_tmp = view->Y_arrow + K2;
    view->theta_locked_tmp = view->V_locked_tmp + vk_elems;
    view->beta_coupling_tmp = view->theta_locked_tmp + k_size;

    ws->n_capacity = n;
    ws->restart_capacity = m_restart;
    ws->block_capacity = k;
    return SPARSE_OK;
}

sparse_err_t sparse_eigs_workspace_prepare_lobpcg(sparse_eigs_workspace_t *ws, idx_t n,
                                                  idx_t block_size, int with_p,
                                                  sparse_eigs_lobpcg_workspace_view_t *view) {
    if (!ws || !view)
        return SPARSE_ERR_NULL;
    if (n < 1 || block_size < 1)
        return SPARSE_ERR_BADARG;

    size_t n_size = 0;
    size_t bs_size = 0;
    if (sparse_idx_to_size_checked(n, &n_size) || sparse_idx_to_size_checked(block_size, &bs_size))
        return SPARSE_ERR_ALLOC;

    size_t cap = 0;
    size_t cap_factor = with_p ? 3U : 2U;
    if (sparse_size_mul_overflow(bs_size, cap_factor, &cap))
        return SPARSE_ERR_ALLOC;
    size_t nc = 0;
    size_t cc = 0;
    size_t nb = 0;
    if (sparse_size_mul_overflow(n_size, cap, &nc) || sparse_size_mul_overflow(cap, cap, &cc) ||
        sparse_size_mul_overflow(n_size, bs_size, &nb))
        return SPARSE_ERR_ALLOC;

    size_t total = 0;
    size_t pieces[] = {nc, nc, cc, cc,     cap, nb, with_p ? nb : 0U, nb, with_p ? nb : 0U,
                       nb, nb, nb, bs_size};
    for (size_t i = 0; i < sizeof(pieces) / sizeof(pieces[0]); i++) {
        if (sparse_size_add_overflow(total, pieces[i], &total))
            return SPARSE_ERR_ALLOC;
    }

    sparse_err_t err = sparse_eigs_workspace_reserve_doubles(ws, total);
    if (err != SPARSE_OK)
        return err;
    err = sparse_eigs_workspace_reserve_idxs(ws, bs_size);
    if (err != SPARSE_OK)
        return err;
    err = sparse_eigs_workspace_reserve_ints(ws, bs_size);
    if (err != SPARSE_OK)
        return err;

    sparse_eigs_workspace_zero_doubles(ws, total);
    sparse_eigs_workspace_zero_ints(ws, bs_size);

    view->Q = ws->double_mem;
    view->AQ = view->Q + nc;
    view->G = view->AQ + nc;
    view->Y = view->G + cc;
    view->theta_full = view->Y + cc;
    view->sel_idx = ws->idx_mem;
    view->X_new = view->theta_full + cap;
    if (with_p) {
        view->P_new = view->X_new + nb;
        view->X = view->P_new + nb;
        view->P = view->X + nb;
    } else {
        view->P_new = NULL;
        view->X = view->X_new + nb;
        view->P = NULL;
    }
    view->R = with_p ? (view->P + nb) : (view->X + nb);
    view->W = view->R + nb;
    view->AX = view->W + nb;
    view->theta = view->AX + nb;
    view->converged = ws->int_mem;

    ws->n_capacity = n;
    ws->block_capacity = block_size;
    return SPARSE_OK;
}
