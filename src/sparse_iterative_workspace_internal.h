#ifndef SPARSE_ITERATIVE_WORKSPACE_INTERNAL_H
#define SPARSE_ITERATIVE_WORKSPACE_INTERNAL_H

/*
 * Private header: reusable internal workspace owner + typed view helpers for
 * iterative solvers. Not part of the public API.
 */

#include "sparse_types.h"
#include <stddef.h>

typedef struct {
    double *double_mem;
    size_t double_capacity;
    int *int_mem;
    size_t int_capacity;
    idx_t n_capacity;
    idx_t restart_capacity;
    idx_t nrhs_capacity;
} sparse_iter_workspace_t;

typedef struct {
    double *r;
    double *z;
    double *p;
    double *Ap;
} sparse_cg_workspace_view_t;

typedef struct {
    double *v;
    double *h;
    double *cs;
    double *sn;
    double *g;
    double *y;
    double *w;
} sparse_gmres_workspace_view_t;

typedef struct {
    double *R;
    double *Z;
    double *P;
    double *AP;
    double *bnorms;
    double *rz;
    int *conv;
    double *rnorms;
} sparse_block_cg_workspace_view_t;

typedef struct {
    double *v;
    double *v_old;
    double *w;
    double *d0;
    double *d1;
    double *d2;
    double *z;
    double *z_tmp;
} sparse_minres_workspace_view_t;

void sparse_iter_workspace_init(sparse_iter_workspace_t *ws);
void sparse_iter_workspace_free(sparse_iter_workspace_t *ws);

sparse_err_t sparse_iter_workspace_prepare_cg(sparse_iter_workspace_t *ws, idx_t n,
                                              sparse_cg_workspace_view_t *view);
sparse_err_t sparse_iter_workspace_prepare_gmres(sparse_iter_workspace_t *ws, idx_t n,
                                                 idx_t restart,
                                                 sparse_gmres_workspace_view_t *view);
sparse_err_t sparse_iter_workspace_prepare_block_cg(sparse_iter_workspace_t *ws, idx_t n,
                                                    idx_t nrhs,
                                                    sparse_block_cg_workspace_view_t *view);
sparse_err_t sparse_iter_workspace_prepare_minres(sparse_iter_workspace_t *ws, idx_t n,
                                                  int with_precond,
                                                  sparse_minres_workspace_view_t *view);

#endif /* SPARSE_ITERATIVE_WORKSPACE_INTERNAL_H */
