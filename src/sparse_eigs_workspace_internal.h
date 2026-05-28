#ifndef SPARSE_EIGS_WORKSPACE_INTERNAL_H
#define SPARSE_EIGS_WORKSPACE_INTERNAL_H

/*
 * Private header: reusable internal workspace owner + typed view helpers for
 * eigensolvers. Not part of the public API.
 */

#include "sparse_types.h"
#include <stddef.h>

typedef struct {
    double *double_mem;
    idx_t *idx_mem;
    int *int_mem;
    size_t double_capacity;
    size_t idx_capacity;
    size_t int_capacity;
    idx_t n_capacity;
    idx_t lanczos_capacity;
    idx_t restart_capacity;
    idx_t block_capacity;
} sparse_eigs_workspace_t;

typedef struct {
    double *V;
    double *alpha;
    double *beta;
    double *v0;
    double *theta_long;
    double *subdiag;
    double *Y_long;
    idx_t *sel_idx;
} sparse_eigs_growm_workspace_view_t;

typedef struct {
    double *V;
    double *alpha;
    double *beta;
    double *v0;
    double *residual_vec;
    double *T_arrow;
    double *theta_arrow;
    double *Y_arrow;
    idx_t *sel_idx;
    double *V_locked_tmp;
    double *theta_locked_tmp;
    double *beta_coupling_tmp;
} sparse_eigs_thick_restart_workspace_view_t;

typedef struct {
    double *Q;
    double *AQ;
    double *G;
    double *Y;
    double *theta_full;
    idx_t *sel_idx;
    double *X_new;
    double *P_new;
    double *X;
    double *R;
    double *W;
    double *AX;
    double *theta;
    int *converged;
} sparse_eigs_lobpcg_workspace_view_t;

void sparse_eigs_workspace_init(sparse_eigs_workspace_t *ws);
void sparse_eigs_workspace_free(sparse_eigs_workspace_t *ws);

sparse_err_t sparse_eigs_workspace_prepare_growm(sparse_eigs_workspace_t *ws, idx_t n, idx_t m_cap,
                                                 idx_t k, sparse_eigs_growm_workspace_view_t *view);
sparse_err_t
sparse_eigs_workspace_prepare_thick_restart(sparse_eigs_workspace_t *ws, idx_t n, idx_t m_restart,
                                            idx_t k,
                                            sparse_eigs_thick_restart_workspace_view_t *view);
sparse_err_t sparse_eigs_workspace_prepare_lobpcg(sparse_eigs_workspace_t *ws, idx_t n,
                                                  idx_t block_size, int with_p,
                                                  sparse_eigs_lobpcg_workspace_view_t *view);

#endif /* SPARSE_EIGS_WORKSPACE_INTERNAL_H */
