/* Request POSIX clock_gettime on platforms that gate it behind
 * _POSIX_C_SOURCE; Windows uses timespec_get below instead. */
#if !defined(_WIN32) && (!defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 199309L)
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define _POSIX_C_SOURCE 199309L
#endif

#include "sparse_iterative.h"
#include "sparse_alloc_internal.h"
#include "sparse_bicgstab_internal.h"
#include "sparse_iterative_internal.h"
#include "sparse_matrix_internal.h"
#include "sparse_vector.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double s29_iter_now_s(void) {
    struct timespec ts;
#ifdef _WIN32
    timespec_get(&ts, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Default option values
 * ═══════════════════════════════════════════════════════════════════════ */

static const sparse_iter_opts_t cg_defaults = {
    .max_iter = 1000,
    .tol = 1e-10,
    .verbose = 0,
};

static const sparse_gmres_opts_t gmres_defaults = {
    .max_iter = 1000,
    .restart = 30,
    .tol = 1e-10,
    .verbose = 0,
    .precond_side = SPARSE_PRECOND_LEFT,
};

static sparse_iter_workspace_t *s49_iter_handle_workspace(const sparse_iter_handle_t *handle) {
    return handle ? (sparse_iter_workspace_t *)handle->internal_state : NULL;
}

/* Shared with sparse_iterative_block.c so block CG uses the same defaults. */
const sparse_iter_opts_t *s85_iter_cg_defaults(void) { return &cg_defaults; }

void s85_iter_result_reset(sparse_iter_result_t *result) {
    if (result)
        *result = (sparse_iter_result_t){0};
}

void s85_iter_result_mark_converged(sparse_iter_result_t *result) {
    s85_iter_result_reset(result);
    if (result)
        result->converged = 1;
}

static int s85_iter_handle_trivial_system(idx_t n, const double *b, double *x,
                                          sparse_iter_result_t *result, double *bnorm_out) {
    if (bnorm_out)
        *bnorm_out = 0.0;

    if (n == 0) {
        s85_iter_result_mark_converged(result);
        return 1;
    }

    double bnorm = vec_norm2(b, n);
    if (bnorm_out)
        *bnorm_out = bnorm;
    if (bnorm != 0.0)
        return 0;

    vec_zero(x, n);
    s85_iter_result_mark_converged(result);
    if (result)
        result->residual_norm = 0.0;
    return 1;
}

sparse_err_t s49_iter_handle_ensure(sparse_iter_handle_t *handle,
                                    sparse_iter_workspace_t **workspace_out) {
    if (!handle || !workspace_out)
        return SPARSE_ERR_NULL;

    sparse_iter_workspace_t *workspace = s49_iter_handle_workspace(handle);
    if (!workspace) {
        workspace = NULL;
        sparse_err_t err = sparse_malloc_array(1, sizeof(*workspace), (void **)&workspace);
        if (err != SPARSE_OK)
            return err;
        sparse_iter_workspace_init(workspace);
        handle->internal_state = workspace;
    }

    *workspace_out = workspace;
    return SPARSE_OK;
}

void sparse_iter_handle_init(sparse_iter_handle_t *handle) {
    if (handle)
        *handle = (sparse_iter_handle_t){0};
}

void sparse_iter_handle_free(sparse_iter_handle_t *handle) {
    if (!handle)
        return;
    sparse_iter_workspace_t *workspace = s49_iter_handle_workspace(handle);
    if (workspace) {
        sparse_iter_workspace_free(workspace);
        free(workspace);
    }
    *handle = (sparse_iter_handle_t){0};
}

sparse_err_t sparse_iter_handle_prepare_cg(sparse_iter_handle_t *handle, idx_t n) {
    if (!handle)
        return SPARSE_ERR_NULL;
    if (n < 1)
        return SPARSE_ERR_BADARG;
    sparse_iter_workspace_t *workspace = NULL;
    sparse_err_t err = s49_iter_handle_ensure(handle, &workspace);
    if (err != SPARSE_OK)
        return err;
    sparse_cg_workspace_view_t view;
    return sparse_iter_workspace_prepare_cg(workspace, n, &view);
}

sparse_err_t sparse_iter_handle_prepare_gmres(sparse_iter_handle_t *handle, idx_t n,
                                              idx_t restart) {
    if (!handle)
        return SPARSE_ERR_NULL;
    if (n < 1)
        return SPARSE_ERR_BADARG;
    if (restart <= 0)
        return SPARSE_ERR_BADARG;
    sparse_iter_workspace_t *workspace = NULL;
    sparse_err_t err = s49_iter_handle_ensure(handle, &workspace);
    if (err != SPARSE_OK)
        return err;
    sparse_gmres_workspace_view_t view;
    return sparse_iter_workspace_prepare_gmres(workspace, n, restart, &view);
}

sparse_err_t sparse_iter_handle_prepare_minres(sparse_iter_handle_t *handle, idx_t n) {
    if (!handle)
        return SPARSE_ERR_NULL;
    if (n < 1)
        return SPARSE_ERR_BADARG;
    sparse_iter_workspace_t *workspace = NULL;
    sparse_err_t err = s49_iter_handle_ensure(handle, &workspace);
    if (err != SPARSE_OK)
        return err;
    sparse_minres_workspace_view_t view;
    return sparse_iter_workspace_prepare_minres(workspace, n, /*with_precond=*/1, &view);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Stagnation detection helper
 * ═══════════════════════════════════════════════════════════════════════ */

#define STAG_DEFAULT_TOL 0.01

sparse_err_t sparse_iter_stag_init(stag_tracker_t *st, idx_t window) {
    *st = (stag_tracker_t){0};
    if (window <= 0)
        return SPARSE_OK;
    double *buf = NULL;
    if (sparse_malloc_idx_array(window, sizeof(double), (void **)&buf) != SPARSE_OK)
        return SPARSE_ERR_ALLOC;
    st->buf = buf;
    st->capacity = window;
    st->count = 0;
    st->head = 0;
    st->tol = STAG_DEFAULT_TOL;
    return SPARSE_OK;
}

void sparse_iter_stag_free(stag_tracker_t *st) {
    free(st->buf);
    *st = (stag_tracker_t){0};
}

void sparse_iter_stag_record(stag_tracker_t *st, double residual) {
    if (!st->buf)
        return;
    st->buf[st->head] = residual;
    st->head = (st->head + 1) % st->capacity;
    if (st->count < st->capacity)
        st->count++;
}

int sparse_iter_stag_check(const stag_tracker_t *st) {
    if (!st->buf || st->count < st->capacity)
        return 0;
    double mn = st->buf[0], mx = st->buf[0];
    for (idx_t i = 1; i < st->capacity; i++) {
        if (st->buf[i] < mn)
            mn = st->buf[i];
        if (st->buf[i] > mx)
            mx = st->buf[i];
    }
    if (mn <= 0.0)
        return 0;
    return (mx / mn - 1.0) < st->tol;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Residual history recording helper
 * ═══════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════
 * Conjugate Gradient
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_solve_cg_with_workspace_internal(const SparseMatrix *A, const double *b,
                                                     double *x, const sparse_iter_opts_t *opts,
                                                     sparse_precond_fn precond,
                                                     const void *precond_ctx,
                                                     sparse_iter_result_t *result,
                                                     sparse_iter_workspace_t *workspace) {
    s85_iter_result_reset(result);

    if (!A || !b || !x)
        return SPARSE_ERR_NULL;
    if (!workspace)
        return SPARSE_ERR_NULL;
    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;

    const sparse_iter_opts_t *o = opts ? opts : &cg_defaults;
    if (o->max_iter < 0 || o->tol < 0.0)
        return SPARSE_ERR_BADARG;
    idx_t n = sparse_rows(A);
    if (n > 0 && (!s217_iter_vector_finite(b, n) || !s217_iter_vector_finite(x, n)))
        return SPARSE_ERR_BADARG;
    double bnorm = 0.0;
    if (s85_iter_handle_trivial_system(n, b, x, result, &bnorm))
        return SPARSE_OK;
    if (!isfinite(bnorm)) {
        s217_iter_numeric_result(result, 0, 0);
        return SPARSE_ERR_NUMERIC;
    }

    sparse_cg_workspace_view_t cg_ws;
    if (sparse_iter_workspace_prepare_cg(workspace, n, &cg_ws) != SPARSE_OK)
        return SPARSE_ERR_ALLOC;
    double *r = cg_ws.r;
    double *z = cg_ws.z;
    double *p = cg_ws.p;
    double *Ap = cg_ws.Ap;

    stag_tracker_t stag;
    if (sparse_iter_stag_init(&stag, o->stagnation_window) != SPARSE_OK) {
        return SPARSE_ERR_ALLOC;
    }

    /* r_0 = b - A*x_0 */
    sparse_matvec(A, x, Ap); /* Ap = A*x_0 */
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - Ap[i];

    double rnorm = vec_norm2(r, n);
    double rel_res = s217_iter_relative_residual(rnorm, bnorm);
    double rz = 0.0;

    idx_t iter = 0;
    int converged = 0;
    int stagnated = 0;
    int breakdown = 0;
    reshist_t rh = reshist_make(o->residual_history, o->residual_history_len);
    if (!isfinite(rel_res))
        goto cg_numeric_failure;
    if (o->max_iter > 0 && !(rel_res <= o->tol)) {
        if (precond) {
            sparse_err_t perr = precond(precond_ctx, n, r, z);
            if (perr != SPARSE_OK) {
                sparse_iter_stag_free(&stag);
                return perr;
            }
        } else {
            vec_copy(r, z, n);
        }
        vec_copy(z, p, n);
        rz = vec_dot(r, z, n);
        if (!isfinite(rz))
            goto cg_numeric_failure;
    }

    double cg_phase_start_s = o->progress_cb ? s29_iter_now_s() : 0.0;

    for (iter = 0; iter < o->max_iter; iter++) {
        /* Check convergence */
        if (rel_res <= o->tol) {
            break;
        }

        /* Publish progress at the top of each iteration.  On
         * cancellation, return the latest iterate in x. */
        if (o->progress_cb) {
            sparse_progress_t pp = {
                .phase = "cg",
                .step = iter,
                .total = o->max_iter,
                .elapsed_s = s29_iter_now_s() - cg_phase_start_s,
            };
            if (o->progress_cb(&pp, o->progress_user) != 0) {
                if (result) {
                    result->iterations = iter;
                    result->residual_norm = rel_res;
                }
                sparse_iter_stag_free(&stag);
                return SPARSE_ERR_CANCELLED;
            }
        }

        iter_report(o->callback, o->callback_ctx, o->verbose, "CG", iter, rel_res);

        /* Ap = A*p */
        sparse_matvec(A, p, Ap);

        /* alpha = (r^T * z) / (p^T * Ap) */
        double pAp = vec_dot(p, Ap, n);
        if (!isfinite(pAp))
            goto cg_numeric_failure;
        if (fabs(pAp) < sparse_rel_tol(0, DROP_TOL)) {
            breakdown = 1;
            break;
        }
        double alpha = rz / pAp;
        if (!isfinite(alpha))
            goto cg_numeric_failure;

        /* x_{k+1} = x_k + alpha * p_k */
        vec_axpy(alpha, p, x, n);

        /* r_{k+1} = r_k - alpha * Ap */
        vec_axpy(-alpha, Ap, r, n);

        rnorm = vec_norm2(r, n);
        rel_res = s217_iter_relative_residual(rnorm, bnorm);
        if (!isfinite(rel_res) || !s217_iter_vector_finite(x, n))
            goto cg_numeric_failure;

        /* Record post-update residual and check stagnation */
        reshist_record(&rh, rel_res);
        if (rel_res <= o->tol || iter + 1 == o->max_iter) {
            iter++;
            break;
        }
        sparse_iter_stag_record(&stag, rel_res);
        if (sparse_iter_stag_check(&stag)) {
            stagnated = 1;
            break;
        }

        /* Apply preconditioner: z_{k+1} = M^{-1}*r_{k+1} */
        if (precond) {
            sparse_err_t perr = precond(precond_ctx, n, r, z);
            if (perr != SPARSE_OK) {
                sparse_iter_stag_free(&stag);
                return perr;
            }
        } else {
            vec_copy(r, z, n);
        }

        /* beta = (r_{k+1}^T * z_{k+1}) / (r_k^T * z_k) */
        double rz_new = vec_dot(r, z, n);
        if (!isfinite(rz_new))
            goto cg_numeric_failure;
        if (fabs(rz) < sparse_rel_tol(0, DROP_TOL)) {
            breakdown = 1;
            break;
        }
        double beta = rz_new / rz;
        if (!isfinite(beta))
            goto cg_numeric_failure;

        /* p_{k+1} = z_{k+1} + beta * p_k */
        for (idx_t i = 0; i < n; i++)
            p[i] = z[i] + beta * p[i];

        rz = rz_new;
    }

    /* Verify the reported result against the actual system residual. */
    if (!s217_iter_vector_finite(x, n))
        goto cg_numeric_failure;
    sparse_matvec(A, x, Ap);
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - Ap[i];
    rel_res = s217_iter_relative_residual(vec_norm2(r, n), bnorm);
    if (!isfinite(rel_res))
        goto cg_numeric_failure;
    converged = (rel_res <= o->tol);

    if (result) {
        result->iterations = iter;
        result->residual_norm = rel_res;
        result->converged = converged;
        result->stagnated = stagnated;
        result->breakdown = breakdown;
        result->residual_history_count = rh.count < rh.len ? rh.count : rh.len;
    }

    sparse_iter_stag_free(&stag);
    return converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;

cg_numeric_failure:
    s217_iter_numeric_result(result, iter, rh.count < rh.len ? rh.count : rh.len);
    sparse_iter_stag_free(&stag);
    return SPARSE_ERR_NUMERIC;
}

sparse_err_t sparse_solve_cg(const SparseMatrix *A, const double *b, double *x,
                             const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                             const void *precond_ctx, sparse_iter_result_t *result) {
    sparse_iter_handle_t handle = {0};
    sparse_err_t err =
        sparse_solve_cg_with_handle(A, b, x, opts, precond, precond_ctx, result, &handle);
    sparse_iter_handle_free(&handle);
    return err;
}

sparse_err_t sparse_solve_cg_with_handle(const SparseMatrix *A, const double *b, double *x,
                                         const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                                         const void *precond_ctx, sparse_iter_result_t *result,
                                         sparse_iter_handle_t *handle) {
    sparse_iter_workspace_t *workspace = NULL;
    sparse_err_t err = s49_iter_handle_ensure(handle, &workspace);
    if (err != SPARSE_OK)
        return err;
    return sparse_solve_cg_with_workspace_internal(A, b, x, opts, precond, precond_ctx, result,
                                                   workspace);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Matrix-free CG
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_solve_cg_mf(sparse_matvec_fn matvec, const void *matvec_ctx, idx_t n,
                                const double *b, double *x, const sparse_iter_opts_t *opts,
                                sparse_precond_fn precond, const void *precond_ctx,
                                sparse_iter_result_t *result) {
    s85_iter_result_reset(result);

    if (!matvec || !b || !x)
        return SPARSE_ERR_NULL;

    const sparse_iter_opts_t *o = opts ? opts : &cg_defaults;
    if (o->max_iter < 0 || o->tol < 0.0)
        return SPARSE_ERR_BADARG;
    if (n < 0)
        return SPARSE_ERR_BADARG;
    if (n > 0 && (!s217_iter_vector_finite(b, n) || !s217_iter_vector_finite(x, n)))
        return SPARSE_ERR_BADARG;
    double bnorm = 0.0;
    if (s85_iter_handle_trivial_system(n, b, x, result, &bnorm))
        return SPARSE_OK;
    if (!isfinite(bnorm)) {
        s217_iter_numeric_result(result, 0, 0);
        return SPARSE_ERR_NUMERIC;
    }

    sparse_iter_workspace_t workspace;
    sparse_cg_workspace_view_t cg_ws;
    sparse_iter_workspace_init(&workspace);
    if (sparse_iter_workspace_prepare_cg(&workspace, n, &cg_ws) != SPARSE_OK)
        return SPARSE_ERR_ALLOC;
    double *r = cg_ws.r;
    double *z = cg_ws.z;
    double *p = cg_ws.p;
    double *Ap = cg_ws.Ap;

    /* r_0 = b - A*x_0 */
    sparse_err_t merr = matvec(matvec_ctx, n, x, Ap);
    if (merr != SPARSE_OK) {
        sparse_iter_workspace_free(&workspace);
        return merr;
    }
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - Ap[i];

    double rnorm = vec_norm2(r, n);
    double rel_res = s217_iter_relative_residual(rnorm, bnorm);
    double rz = 0.0;

    stag_tracker_t stag = {0};
    if (sparse_iter_stag_init(&stag, o->stagnation_window) != SPARSE_OK) {
        sparse_iter_workspace_free(&workspace);
        return SPARSE_ERR_ALLOC;
    }

    idx_t iter = 0;
    int converged = 0;
    int stagnated = 0;
    int breakdown = 0;
    reshist_t rh = reshist_make(o->residual_history, o->residual_history_len);
    if (!isfinite(rel_res))
        goto cg_mf_numeric_failure;
    if (o->max_iter > 0 && !(rel_res <= o->tol)) {
        if (precond) {
            sparse_err_t perr = precond(precond_ctx, n, r, z);
            if (perr != SPARSE_OK) {
                sparse_iter_stag_free(&stag);
                sparse_iter_workspace_free(&workspace);
                return perr;
            }
        } else {
            vec_copy(r, z, n);
        }
        vec_copy(z, p, n);
        rz = vec_dot(r, z, n);
        if (!isfinite(rz))
            goto cg_mf_numeric_failure;
    }

    double cg_mf_phase_start_s = o->progress_cb ? s29_iter_now_s() : 0.0;

    for (iter = 0; iter < o->max_iter; iter++) {
        if (rel_res <= o->tol) {
            break;
        }

        if (o->progress_cb) {
            sparse_progress_t pp = {
                .phase = "cg",
                .step = iter,
                .total = o->max_iter,
                .elapsed_s = s29_iter_now_s() - cg_mf_phase_start_s,
            };
            if (o->progress_cb(&pp, o->progress_user) != 0) {
                if (result) {
                    result->iterations = iter;
                    result->residual_norm = rel_res;
                }
                sparse_iter_stag_free(&stag);
                sparse_iter_workspace_free(&workspace);
                return SPARSE_ERR_CANCELLED;
            }
        }

        iter_report(o->callback, o->callback_ctx, o->verbose, "CG", iter, rel_res);

        merr = matvec(matvec_ctx, n, p, Ap);
        if (merr != SPARSE_OK) {
            sparse_iter_stag_free(&stag);
            sparse_iter_workspace_free(&workspace);
            return merr;
        }

        double pAp = vec_dot(p, Ap, n);
        if (!isfinite(pAp))
            goto cg_mf_numeric_failure;
        if (fabs(pAp) < sparse_rel_tol(0, DROP_TOL)) {
            breakdown = 1;
            break;
        }
        double alpha = rz / pAp;
        if (!isfinite(alpha))
            goto cg_mf_numeric_failure;

        vec_axpy(alpha, p, x, n);
        vec_axpy(-alpha, Ap, r, n);
        rnorm = vec_norm2(r, n);
        rel_res = s217_iter_relative_residual(rnorm, bnorm);
        if (!isfinite(rel_res) || !s217_iter_vector_finite(x, n))
            goto cg_mf_numeric_failure;

        reshist_record(&rh, rel_res);
        if (rel_res <= o->tol || iter + 1 == o->max_iter) {
            iter++;
            break;
        }
        sparse_iter_stag_record(&stag, rel_res);
        if (sparse_iter_stag_check(&stag)) {
            stagnated = 1;
            break;
        }

        if (precond) {
            sparse_err_t perr = precond(precond_ctx, n, r, z);
            if (perr != SPARSE_OK) {
                sparse_iter_stag_free(&stag);
                sparse_iter_workspace_free(&workspace);
                return perr;
            }
        } else {
            vec_copy(r, z, n);
        }

        double rz_new = vec_dot(r, z, n);
        if (!isfinite(rz_new))
            goto cg_mf_numeric_failure;
        if (fabs(rz) < sparse_rel_tol(0, DROP_TOL)) {
            breakdown = 1;
            break;
        }
        double beta = rz_new / rz;
        if (!isfinite(beta))
            goto cg_mf_numeric_failure;

        for (idx_t i = 0; i < n; i++)
            p[i] = z[i] + beta * p[i];

        rz = rz_new;
    }

    if (!s217_iter_vector_finite(x, n))
        goto cg_mf_numeric_failure;
    merr = matvec(matvec_ctx, n, x, Ap);
    if (merr != SPARSE_OK) {
        sparse_iter_stag_free(&stag);
        sparse_iter_workspace_free(&workspace);
        return merr;
    }
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - Ap[i];
    rel_res = s217_iter_relative_residual(vec_norm2(r, n), bnorm);
    if (!isfinite(rel_res))
        goto cg_mf_numeric_failure;
    converged = (rel_res <= o->tol);

    if (result) {
        result->iterations = iter;
        result->residual_norm = rel_res;
        result->converged = converged;
        result->stagnated = stagnated;
        result->breakdown = breakdown;
        result->residual_history_count = rh.count < rh.len ? rh.count : rh.len;
    }

    sparse_iter_stag_free(&stag);
    sparse_iter_workspace_free(&workspace);
    return converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;

cg_mf_numeric_failure:
    s217_iter_numeric_result(result, iter, rh.count < rh.len ? rh.count : rh.len);
    sparse_iter_stag_free(&stag);
    sparse_iter_workspace_free(&workspace);
    return SPARSE_ERR_NUMERIC;
}

/* ═══════════════════════════════════════════════════════════════════════
 * GMRES — Restarted GMRES(k) with Arnoldi & Givens rotations
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Hessenberg matrix H is stored column-major in a flat array of size
 * (m+1)*m, where m = restart.  H[i + j*(m+1)] = H(i,j).
 *
 * Arnoldi basis V is stored as m+1 column vectors of length n, flat
 * array of size (m+1)*n.  V[i + j*n] = V_j[i]  (column j, row i).
 */

/* Access macros for column-major storage */
#define H(i, j) h[(size_t)(i) + (size_t)(j) * ((size_t)(m) + 1)]
#define V(col) (&v[(size_t)(col) * (size_t)n])

/* Adapter: wrap SparseMatrix* into sparse_matvec_fn for GMRES */
static sparse_err_t gmres_sparse_matvec_adapter(const void *ctx, idx_t n, const double *x_in,
                                                double *y_out) {
    (void)n;
    const SparseMatrix *A = (const SparseMatrix *)ctx;
    return sparse_matvec(A, x_in, y_out);
}

sparse_err_t sparse_solve_gmres(const SparseMatrix *A, const double *b, double *x,
                                const sparse_gmres_opts_t *opts, sparse_precond_fn precond,
                                const void *precond_ctx, sparse_iter_result_t *result) {
    s85_iter_result_reset(result);

    if (!A || !b || !x)
        return SPARSE_ERR_NULL;
    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;

    sparse_iter_handle_t handle = {0};
    sparse_err_t err =
        sparse_solve_gmres_with_handle(A, b, x, opts, precond, precond_ctx, result, &handle);
    sparse_iter_handle_free(&handle);
    return err;
}

static sparse_err_t sparse_solve_gmres_mf_with_workspace_internal(
    sparse_matvec_fn matvec, const void *matvec_ctx, idx_t n, const double *b, double *x,
    const sparse_gmres_opts_t *opts, sparse_precond_fn precond, const void *precond_ctx,
    sparse_iter_result_t *result, sparse_iter_workspace_t *workspace) {
    s85_iter_result_reset(result);

    if (!matvec || !b || !x)
        return SPARSE_ERR_NULL;
    if (!workspace)
        return SPARSE_ERR_NULL;

    const sparse_gmres_opts_t *o = opts ? opts : &gmres_defaults;
    if (o->max_iter < 0 || o->restart <= 0 || o->tol < 0.0)
        return SPARSE_ERR_BADARG;
    if (o->precond_side != SPARSE_PRECOND_LEFT && o->precond_side != SPARSE_PRECOND_RIGHT)
        return SPARSE_ERR_BADARG;
    if (n < 0)
        return SPARSE_ERR_BADARG;
    idx_t m = o->restart; /* restart parameter */
    int right_precond = (precond && o->precond_side == SPARSE_PRECOND_RIGHT);
    if (n > 0 && (!s217_iter_vector_finite(b, n) || !s217_iter_vector_finite(x, n)))
        return SPARSE_ERR_BADARG;
    double bnorm = 0.0;
    if (s85_iter_handle_trivial_system(n, b, x, result, &bnorm))
        return SPARSE_OK;
    if (!isfinite(bnorm)) {
        s217_iter_numeric_result(result, 0, 0);
        return SPARSE_ERR_NUMERIC;
    }

    sparse_err_t merr;

    /* Fast path for max_iter==0: compute initial residual without
     * allocating the full Arnoldi workspace */
    if (o->max_iter == 0) {
        double *tmp = NULL;
        if (sparse_malloc_idx_array(n, sizeof(double), (void **)&tmp) != SPARSE_OK)
            return SPARSE_ERR_ALLOC;
        merr = matvec(matvec_ctx, n, x, tmp);
        if (merr != SPARSE_OK) {
            free(tmp);
            return merr;
        }
        for (idx_t i = 0; i < n; i++)
            tmp[i] = b[i] - tmp[i];
        double rr = s217_iter_relative_residual(vec_norm2(tmp, n), bnorm);
        free(tmp);
        if (!isfinite(rr)) {
            s217_iter_numeric_result(result, 0, 0);
            return SPARSE_ERR_NUMERIC;
        }
        int conv = (rr <= o->tol);
        if (result) {
            result->iterations = 0;
            result->residual_norm = rr;
            result->converged = conv;
        }
        return conv ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
    }

    /* Clamp restart to min(n, max_iter) to avoid oversized allocations
     * when restart is large but max_iter is small */
    if (m > n)
        m = n;
    if (m > o->max_iter)
        m = o->max_iter;
    /* m >= 1 guaranteed: restart <= 0 rejected above, n >= 1, max_iter >= 1 */

    /* Check initial true residual before allocating the full workspace,
     * so we return cheaply if the initial guess already satisfies tol */
    double initial_rel = NAN;
    {
        double *tmp = NULL;
        if (sparse_malloc_idx_array(n, sizeof(double), (void **)&tmp) != SPARSE_OK)
            return SPARSE_ERR_ALLOC;
        merr = matvec(matvec_ctx, n, x, tmp);
        if (merr != SPARSE_OK) {
            free(tmp);
            return merr;
        }
        for (idx_t i = 0; i < n; i++)
            tmp[i] = b[i] - tmp[i];
        double rr = s217_iter_relative_residual(vec_norm2(tmp, n), bnorm);
        free(tmp);
        if (!isfinite(rr)) {
            s217_iter_numeric_result(result, 0, 0);
            return SPARSE_ERR_NUMERIC;
        }
        initial_rel = rr;
        if (rr <= o->tol) {
            if (result) {
                result->iterations = 0;
                result->residual_norm = rr;
                result->converged = 1;
            }
            return SPARSE_OK;
        }
    }

    sparse_gmres_workspace_view_t gmres_ws;
    if (sparse_iter_workspace_prepare_gmres(workspace, n, m, &gmres_ws) != SPARSE_OK)
        return SPARSE_ERR_ALLOC;
    double *v = gmres_ws.v;
    double *h = gmres_ws.h;
    double *cs = gmres_ws.cs;
    double *sn = gmres_ws.sn;
    double *g = gmres_ws.g;
    double *y = gmres_ws.y;
    double *w = gmres_ws.w;

    size_t m_size = 0;
    size_t m1_size = 0;
    if (sparse_idx_to_size_checked(m, &m_size) || sparse_size_add_overflow(m_size, 1, &m1_size)) {
        return SPARSE_ERR_ALLOC;
    }
    size_t sz_h = 0;
    if (sparse_size_mul_overflow(m1_size, m_size, &sz_h)) {
        return SPARSE_ERR_ALLOC;
    }

    stag_tracker_t stag;
    if (sparse_iter_stag_init(&stag, o->stagnation_window) != SPARSE_OK) {
        sparse_iter_stag_free(&stag);
        return SPARSE_ERR_ALLOC;
    }

    idx_t total_iter = 0;
    int converged = 0;
    int stagnated = 0;
    int breakdown = 0;
    double rel_res = 1.0;
    reshist_t rh = reshist_make(o->residual_history, o->residual_history_len);
    double gmres_phase_start_s = o->progress_cb ? s29_iter_now_s() : 0.0;

    /* Outer restart loop — compute ceil(max_iter/m) in wider type to avoid
     * signed overflow when max_iter is near INT32_MAX */
    idx_t max_restarts = (idx_t)(((int64_t)o->max_iter + m - 1) / m);

    for (idx_t restart = 0; restart < max_restarts && !converged; restart++) {
        /* Compute r = b - A*x */
        merr = matvec(matvec_ctx, n, x, w);
        if (merr != SPARSE_OK) {
            sparse_iter_stag_free(&stag);
            return merr;
        }
        for (idx_t i = 0; i < n; i++)
            V(0)[i] = b[i] - w[i];

        /* Left preconditioning: v_0 = M^{-1} * r (right precond: no change) */
        if (precond && !right_precond) {
            vec_copy(V(0), w, n);
            sparse_err_t perr = precond(precond_ctx, n, w, V(0));
            if (perr != SPARSE_OK) {
                sparse_iter_stag_free(&stag);
                return perr;
            }
        }

        double beta = vec_norm2(V(0), n);
        if (!isfinite(beta))
            goto gmres_numeric_failure;
        if (beta == 0.0) {
            breakdown = 1;
            rel_res = initial_rel;
            break;
        }

        /* Normalize v_0 */
        double inv_beta = 1.0 / beta;
        for (idx_t i = 0; i < n; i++)
            V(0)[i] *= inv_beta;
        if (!s217_iter_vector_finite(V(0), n))
            goto gmres_numeric_failure;

        /* Initialize residual vector g = beta * e_1 */
        for (idx_t i = 0; i <= m; i++)
            g[i] = 0.0;
        g[0] = beta;

        /* Zero out Hessenberg matrix */
        for (size_t i = 0; i < sz_h; i++)
            h[i] = 0.0;

        idx_t j;
        for (j = 0; j < m; j++) {
            if (total_iter >= o->max_iter)
                break;

            /* Publish progress for each inner GMRES iteration. */
            if (o->progress_cb) {
                sparse_progress_t pp = {
                    .phase = "gmres",
                    .step = total_iter,
                    .total = o->max_iter,
                    .elapsed_s = s29_iter_now_s() - gmres_phase_start_s,
                };
                if (o->progress_cb(&pp, o->progress_user) != 0) {
                    if (result) {
                        result->iterations = total_iter;
                    }
                    sparse_iter_stag_free(&stag);
                    return SPARSE_ERR_CANCELLED;
                }
            }

            total_iter++;

            if (right_precond) {
                /* Right preconditioning: w = A * M^{-1} * v_j */
                sparse_err_t perr = precond(precond_ctx, n, V(j), V(j + 1));
                if (perr != SPARSE_OK) {
                    sparse_iter_stag_free(&stag);
                    return perr;
                }
                if (!s217_iter_vector_finite(V(j + 1), n))
                    goto gmres_numeric_failure;
                merr = matvec(matvec_ctx, n, V(j + 1), w);
                if (merr != SPARSE_OK) {
                    sparse_iter_stag_free(&stag);
                    return merr;
                }
            } else {
                /* w = A * v_j */
                merr = matvec(matvec_ctx, n, V(j), w);
                if (merr != SPARSE_OK) {
                    sparse_iter_stag_free(&stag);
                    return merr;
                }

                /* Left preconditioning: w = M^{-1} * A * v_j */
                if (precond) {
                    vec_copy(w, V(j + 1), n);
                    sparse_err_t perr = precond(precond_ctx, n, V(j + 1), w);
                    if (perr != SPARSE_OK) {
                        sparse_iter_stag_free(&stag);
                        return perr;
                    }
                    if (!s217_iter_vector_finite(w, n))
                        goto gmres_numeric_failure;
                }
            }

            /* Arnoldi: modified Gram-Schmidt orthogonalization */
            for (idx_t i = 0; i <= j; i++) {
                H(i, j) = vec_dot(w, V(i), n);
                if (!isfinite(H(i, j)))
                    goto gmres_numeric_failure;
                vec_axpy(-H(i, j), V(i), w, n);
            }
            H(j + 1, j) = vec_norm2(w, n);
            if (!isfinite(H(j + 1, j)))
                goto gmres_numeric_failure;

            /* Check for lucky breakdown (before Givens rotation zeroes H(j+1,j)) */
            int lucky = (H(j + 1, j) < sparse_rel_tol(0, DROP_TOL));
            if (lucky) {
                breakdown = 1;
                vec_zero(V(j + 1), n);
            } else {
                double inv_h = 1.0 / H(j + 1, j);
                for (idx_t i = 0; i < n; i++)
                    V(j + 1)[i] = w[i] * inv_h;
                if (!s217_iter_vector_finite(V(j + 1), n))
                    goto gmres_numeric_failure;
            }

            /* Apply previous Givens rotations to column j of H */
            for (idx_t i = 0; i < j; i++) {
                double tmp = cs[i] * H(i, j) + sn[i] * H(i + 1, j);
                H(i + 1, j) = -sn[i] * H(i, j) + cs[i] * H(i + 1, j);
                H(i, j) = tmp;
            }

            /* Compute new Givens rotation for row j */
            {
                double a = H(j, j);
                double b_val = H(j + 1, j);
                double r = hypot(a, b_val);
                if (!isfinite(r))
                    goto gmres_numeric_failure;
                if (r > 0.0) {
                    cs[j] = a / r;
                    sn[j] = b_val / r;
                } else {
                    cs[j] = 1.0;
                    sn[j] = 0.0;
                }
            }

            /* Apply new Givens rotation to H and g */
            H(j, j) = cs[j] * H(j, j) + sn[j] * H(j + 1, j);
            H(j + 1, j) = 0.0;

            {
                double tmp = cs[j] * g[j] + sn[j] * g[j + 1];
                g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
                g[j] = tmp;
            }

            rel_res = s217_iter_relative_residual(fabs(g[j + 1]), bnorm);
            if (!isfinite(rel_res) || !isfinite(g[j]))
                goto gmres_numeric_failure;

            reshist_record(&rh, rel_res);
            iter_report(o->callback, o->callback_ctx, o->verbose, "GMRES", total_iter - 1, rel_res);

            /* Stop inner Arnoldi loop on preconditioned convergence or
             * lucky breakdown.  Final convergence is decided by the true
             * residual check after x is updated. */
            if (rel_res <= o->tol || lucky) {
                j++; /* include this column in the solution */
                break;
            }
        }

        /* Solve the upper triangular system H * y = g */
        for (idx_t i = j - 1; i >= 0; i--) {
            y[i] = g[i];
            for (idx_t k = i + 1; k < j; k++)
                y[i] -= H(i, k) * y[k];
            if (fabs(H(i, i)) > sparse_rel_tol(0, DROP_TOL))
                y[i] /= H(i, i);
            else
                y[i] = 0.0; /* singular Hessenberg diagonal — treat as zero */
            if (!isfinite(y[i]))
                goto gmres_numeric_failure;
        }

        /* Update solution */
        if (right_precond) {
            /* Right precond: x = x + M^{-1} * (V * y) */
            /* First compute V*y into w, then apply M^{-1} */
            vec_zero(w, n);
            for (idx_t k = 0; k < j; k++)
                vec_axpy(y[k], V(k), w, n);
            sparse_err_t perr = precond(precond_ctx, n, w, V(0)); /* reuse V(0) as temp */
            if (perr != SPARSE_OK) {
                sparse_iter_stag_free(&stag);
                return perr;
            }
            if (!s217_iter_vector_finite(V(0), n))
                goto gmres_numeric_failure;
            vec_axpy(1.0, V(0), x, n);
        } else {
            /* Left precond / unpreconditioned: x = x + V * y */
            for (idx_t k = 0; k < j; k++)
                vec_axpy(y[k], V(k), x, n);
        }
        if (!s217_iter_vector_finite(x, n))
            goto gmres_numeric_failure;

        /* Compute true residual to decide convergence */
        merr = matvec(matvec_ctx, n, x, w);
        if (merr != SPARSE_OK) {
            sparse_iter_stag_free(&stag);
            return merr;
        }
        for (idx_t i = 0; i < n; i++)
            w[i] = b[i] - w[i];
        rel_res = s217_iter_relative_residual(vec_norm2(w, n), bnorm);
        if (!isfinite(rel_res))
            goto gmres_numeric_failure;
        initial_rel = rel_res;

        if (rel_res <= o->tol) {
            converged = 1;
            break;
        }

        /* Stagnation check across restarts */
        sparse_iter_stag_record(&stag, rel_res);
        if (sparse_iter_stag_check(&stag)) {
            stagnated = 1;
            break;
        }
    }

    if (result) {
        result->iterations = total_iter;
        result->residual_norm = rel_res;
        result->converged = converged;
        result->stagnated = stagnated;
        result->breakdown = breakdown;
        result->residual_history_count = rh.count < rh.len ? rh.count : rh.len;
    }

    sparse_iter_stag_free(&stag);
    return converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;

gmres_numeric_failure:
    s217_iter_numeric_result(result, total_iter, rh.count < rh.len ? rh.count : rh.len);
    sparse_iter_stag_free(&stag);
    return SPARSE_ERR_NUMERIC;
}

sparse_err_t sparse_solve_gmres_mf(sparse_matvec_fn matvec, const void *matvec_ctx, idx_t n,
                                   const double *b, double *x, const sparse_gmres_opts_t *opts,
                                   sparse_precond_fn precond, const void *precond_ctx,
                                   sparse_iter_result_t *result) {
    sparse_iter_workspace_t workspace;
    sparse_iter_workspace_init(&workspace);
    sparse_err_t err = sparse_solve_gmres_mf_with_workspace_internal(
        matvec, matvec_ctx, n, b, x, opts, precond, precond_ctx, result, &workspace);
    sparse_iter_workspace_free(&workspace);
    return err;
}

sparse_err_t sparse_solve_gmres_with_workspace_internal(const SparseMatrix *A, const double *b,
                                                        double *x, const sparse_gmres_opts_t *opts,
                                                        sparse_precond_fn precond,
                                                        const void *precond_ctx,
                                                        sparse_iter_result_t *result,
                                                        sparse_iter_workspace_t *workspace) {
    s85_iter_result_reset(result);

    if (!A || !b || !x)
        return SPARSE_ERR_NULL;
    if (!workspace)
        return SPARSE_ERR_NULL;

    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;

    return sparse_solve_gmres_mf_with_workspace_internal(gmres_sparse_matvec_adapter, A,
                                                         sparse_rows(A), b, x, opts, precond,
                                                         precond_ctx, result, workspace);
}

sparse_err_t sparse_solve_gmres_with_handle(const SparseMatrix *A, const double *b, double *x,
                                            const sparse_gmres_opts_t *opts,
                                            sparse_precond_fn precond, const void *precond_ctx,
                                            sparse_iter_result_t *result,
                                            sparse_iter_handle_t *handle) {
    sparse_iter_workspace_t *workspace = NULL;
    sparse_err_t err = s49_iter_handle_ensure(handle, &workspace);
    if (err != SPARSE_OK)
        return err;
    return sparse_solve_gmres_with_workspace_internal(A, b, x, opts, precond, precond_ctx, result,
                                                      workspace);
}

#undef H
#undef V

/* ═══════════════════════════════════════════════════════════════════════
 * BiCGSTAB — Bi-Conjugate Gradient Stabilized
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_solve_bicgstab(const SparseMatrix *A, const double *b, double *x,
                                   const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                                   const void *precond_ctx, sparse_iter_result_t *result) {
    s85_iter_result_reset(result);

    if (!A || !b || !x)
        return SPARSE_ERR_NULL;
    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;

    const sparse_iter_opts_t *o = opts ? opts : &cg_defaults;
    if (o->max_iter < 0 || o->tol < 0.0)
        return SPARSE_ERR_BADARG;
    idx_t n = sparse_rows(A);
    if (n > 0 && (!s217_iter_vector_finite(b, n) || !s217_iter_vector_finite(x, n)))
        return SPARSE_ERR_BADARG;
    double bnorm = 0.0;
    if (s85_iter_handle_trivial_system(n, b, x, result, &bnorm))
        return SPARSE_OK;
    if (!isfinite(bnorm)) {
        s217_iter_numeric_result(result, 0, 0);
        return SPARSE_ERR_NUMERIC;
    }

    /* Allocate workspace */
    bicgstab_workspace_t ws;
    sparse_err_t werr = bicgstab_workspace_alloc(n, precond != NULL, &ws);
    if (werr != SPARSE_OK)
        return werr;

    stag_tracker_t stag;
    if (sparse_iter_stag_init(&stag, o->stagnation_window) != SPARSE_OK) {
        bicgstab_workspace_free(&ws);
        return SPARSE_ERR_ALLOC;
    }

    /* r_0 = b - A*x_0 */
    sparse_matvec(A, x, ws.v); /* use v as temp */
    for (idx_t i = 0; i < n; i++)
        ws.r[i] = b[i] - ws.v[i];

    /* r_hat = r_0 (shadow residual, fixed throughout) */
    vec_copy(ws.r, ws.r_hat, n);

    /* p_0 = r_0 */
    vec_copy(ws.r, ws.p, n);

    double rnorm = vec_norm2(ws.r, n);
    double rel_res = s217_iter_relative_residual(rnorm, bnorm);
    double rho = 0.0;

    idx_t iter = 0;
    int converged = 0;
    int stagnated = 0;
    int breakdown = 0;
    sparse_err_t numeric_err = SPARSE_OK;
    reshist_t rh = reshist_make(o->residual_history, o->residual_history_len);
    if (!isfinite(rel_res)) {
        numeric_err = SPARSE_ERR_NUMERIC;
        goto done;
    }

    /* Check if already converged */
    if (rel_res <= o->tol) {
        converged = 1;
        goto done;
    }
    if (o->max_iter == 0)
        goto done;

    /* Initial rho must be safely nonzero before starting BiCGSTAB recurrences. */
    rho = vec_dot(ws.r_hat, ws.r, n);
    if (!isfinite(rho)) {
        numeric_err = SPARSE_ERR_NUMERIC;
        goto done;
    }
    if (fabs(rho) < sparse_rel_tol(0, DROP_TOL)) {
        breakdown = 1;
        goto done;
    }

    double bicgstab_phase_start_s = o->progress_cb ? s29_iter_now_s() : 0.0;

    for (iter = 0; iter < o->max_iter; iter++) {
        /* Publish progress for each BiCGSTAB iteration. */
        if (o->progress_cb) {
            sparse_progress_t pp = {
                .phase = "bicgstab",
                .step = iter,
                .total = o->max_iter,
                .elapsed_s = s29_iter_now_s() - bicgstab_phase_start_s,
            };
            if (o->progress_cb(&pp, o->progress_user) != 0) {
                if (result) {
                    result->iterations = iter;
                    result->residual_norm = rel_res;
                }
                sparse_iter_stag_free(&stag);
                bicgstab_workspace_free(&ws);
                return SPARSE_ERR_CANCELLED;
            }
        }

        reshist_record(&rh, rel_res);
        iter_report(o->callback, o->callback_ctx, o->verbose, "BiCGSTAB", iter, rel_res);

        /* --- First half-step: BiCG direction --- */

        /* Apply preconditioner to p */
        double *p_eff = ws.p;
        if (precond) {
            sparse_err_t perr = precond(precond_ctx, n, ws.p, ws.p_hat);
            if (perr != SPARSE_OK) {
                sparse_iter_stag_free(&stag);
                bicgstab_workspace_free(&ws);
                return perr;
            }
            p_eff = ws.p_hat;
        }

        /* v = A * p_hat (or A * p if no precond) */
        sparse_matvec(A, p_eff, ws.v);

        /* alpha = rho / (r_hat^T * v) */
        double r_hat_v = vec_dot(ws.r_hat, ws.v, n);
        if (fabs(r_hat_v) < sparse_rel_tol(0, DROP_TOL)) {
            breakdown = 1;
            break;
        }
        double alpha = rho / r_hat_v;

        if (!isfinite(alpha)) {
            numeric_err = SPARSE_ERR_NUMERIC;
            break;
        }

        /* s = r - alpha * v */
        vec_copy(ws.r, ws.s, n);
        vec_axpy(-alpha, ws.v, ws.s, n);

        /* Check if ||s|| is small enough for early termination */
        double snorm = vec_norm2(ws.s, n);
        double srel = s217_iter_relative_residual(snorm, bnorm);
        if (!isfinite(srel)) {
            numeric_err = SPARSE_ERR_NUMERIC;
            break;
        }
        if (srel <= o->tol) {
            /* Accept half-step: x = x + alpha * p_hat */
            vec_axpy(alpha, p_eff, x, n);
            if (!s217_iter_vector_finite(x, n)) {
                numeric_err = SPARSE_ERR_NUMERIC;
                break;
            }
            rel_res = srel;
            converged = 1;
            iter++;
            break;
        }

        /* --- Second half-step: stabilization --- */

        /* Apply preconditioner to s */
        double *s_eff = ws.s;
        if (precond) {
            sparse_err_t perr = precond(precond_ctx, n, ws.s, ws.s_hat);
            if (perr != SPARSE_OK) {
                sparse_iter_stag_free(&stag);
                bicgstab_workspace_free(&ws);
                return perr;
            }
            s_eff = ws.s_hat;
        }

        /* t = A * s_hat (or A * s if no precond) */
        sparse_matvec(A, s_eff, ws.t);

        /* omega = (t^T * s) / (t^T * t) */
        double tt = vec_dot(ws.t, ws.t, n);
        if (tt < sparse_rel_tol(0, DROP_TOL)) {
            breakdown = 1;
            break;
        }
        double omega = vec_dot(ws.t, ws.s, n) / tt;

        if (!isfinite(omega)) {
            numeric_err = SPARSE_ERR_NUMERIC;
            break;
        }

        /* Near-zero omega: stabilization polynomial failed.
         * Accept the half-step x += alpha * p_hat and continue. */
        if (fabs(omega) < 1e-15 * fabs(alpha)) {
            vec_axpy(alpha, p_eff, x, n);
            sparse_matvec(A, x, ws.r);
            for (idx_t i = 0; i < n; i++)
                ws.r[i] = b[i] - ws.r[i];
            rnorm = vec_norm2(ws.r, n);
            rel_res = s217_iter_relative_residual(rnorm, bnorm);
            if (!isfinite(rel_res) || !s217_iter_vector_finite(x, n)) {
                numeric_err = SPARSE_ERR_NUMERIC;
                break;
            }
            vec_copy(ws.r, ws.p, n);
            rho = vec_dot(ws.r_hat, ws.r, n);
            if (fabs(rho) < sparse_rel_tol(0, DROP_TOL)) {
                breakdown = 1;
                break;
            }
            continue;
        }

        /* x = x + alpha * p_hat + omega * s_hat */
        vec_axpy(alpha, p_eff, x, n);
        vec_axpy(omega, s_eff, x, n);

        /* r = s - omega * t */
        vec_copy(ws.s, ws.r, n);
        vec_axpy(-omega, ws.t, ws.r, n);

        rnorm = vec_norm2(ws.r, n);
        rel_res = s217_iter_relative_residual(rnorm, bnorm);
        if (!isfinite(rel_res) || !s217_iter_vector_finite(x, n)) {
            numeric_err = SPARSE_ERR_NUMERIC;
            break;
        }

        /* Check convergence */
        if (rel_res <= o->tol) {
            converged = 1;
            iter++;
            break;
        }

        /* Stagnation check */
        sparse_iter_stag_record(&stag, rel_res);
        if (sparse_iter_stag_check(&stag)) {
            stagnated = 1;
            break;
        }

        /* Prepare for next iteration */
        double rho_new = vec_dot(ws.r_hat, ws.r, n);
        if (fabs(rho_new) < sparse_rel_tol(0, DROP_TOL)) {
            breakdown = 1;
            break;
        }

        double beta = (rho_new / rho) * (alpha / omega);

        if (!isfinite(beta)) {
            numeric_err = SPARSE_ERR_NUMERIC;
            break;
        }

        /* p = r + beta * (p - omega * v) */
        for (idx_t i = 0; i < n; i++)
            ws.p[i] = ws.r[i] + beta * (ws.p[i] - omega * ws.v[i]);

        rho = rho_new;
    }

done:;
    /* Compute true residual ||b - Ax|| / ||b|| for the final report. */
    double true_rel_res = rel_res;
    if (iter > 0 && numeric_err == SPARSE_OK) {
        sparse_matvec(A, x, ws.r);
        for (idx_t i = 0; i < n; i++)
            ws.r[i] = b[i] - ws.r[i];
        true_rel_res = s217_iter_relative_residual(vec_norm2(ws.r, n), bnorm);
        if (true_rel_res > o->tol)
            converged = 0;
    }
    if (!isfinite(true_rel_res) || numeric_err != SPARSE_OK) {
        numeric_err = SPARSE_ERR_NUMERIC;
        converged = 0;
        true_rel_res = NAN;
    }

    if (result) {
        result->iterations = iter;
        result->residual_norm = true_rel_res;
        result->converged = converged;
        result->stagnated = stagnated;
        result->breakdown = breakdown;
        result->residual_history_count = rh.count < rh.len ? rh.count : rh.len;
    }

    sparse_iter_stag_free(&stag);
    bicgstab_workspace_free(&ws);
    if (numeric_err != SPARSE_OK)
        return numeric_err;
    return converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Matrix-free BiCGSTAB
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_solve_bicgstab_mf(sparse_matvec_fn matvec, const void *matvec_ctx, idx_t n,
                                      const double *b, double *x, const sparse_iter_opts_t *opts,
                                      sparse_precond_fn precond, const void *precond_ctx,
                                      sparse_iter_result_t *result) {
    s85_iter_result_reset(result);

    if (!matvec || !b || !x)
        return SPARSE_ERR_NULL;

    const sparse_iter_opts_t *o = opts ? opts : &cg_defaults;
    if (o->max_iter < 0 || o->tol < 0.0)
        return SPARSE_ERR_BADARG;
    if (n < 0)
        return SPARSE_ERR_BADARG;
    if (n > 0 && (!s217_iter_vector_finite(b, n) || !s217_iter_vector_finite(x, n)))
        return SPARSE_ERR_BADARG;
    double bnorm = 0.0;
    if (s85_iter_handle_trivial_system(n, b, x, result, &bnorm))
        return SPARSE_OK;
    if (!isfinite(bnorm)) {
        s217_iter_numeric_result(result, 0, 0);
        return SPARSE_ERR_NUMERIC;
    }

    bicgstab_workspace_t ws;
    sparse_err_t werr = bicgstab_workspace_alloc(n, precond != NULL, &ws);
    if (werr != SPARSE_OK)
        return werr;

    stag_tracker_t stag;
    if (sparse_iter_stag_init(&stag, o->stagnation_window) != SPARSE_OK) {
        sparse_iter_stag_free(&stag);
        bicgstab_workspace_free(&ws);
        return SPARSE_ERR_ALLOC;
    }

    sparse_err_t merr = matvec(matvec_ctx, n, x, ws.v);
    if (merr != SPARSE_OK) {
        sparse_iter_stag_free(&stag);
        bicgstab_workspace_free(&ws);
        return merr;
    }
    for (idx_t i = 0; i < n; i++)
        ws.r[i] = b[i] - ws.v[i];

    vec_copy(ws.r, ws.r_hat, n);
    vec_copy(ws.r, ws.p, n);

    double rnorm = vec_norm2(ws.r, n);
    double rel_res = s217_iter_relative_residual(rnorm, bnorm);
    double rho = 0.0;

    idx_t iter = 0;
    int converged = 0;
    int stagnated = 0;
    int breakdown = 0;
    sparse_err_t numeric_err = SPARSE_OK;
    reshist_t rh = reshist_make(o->residual_history, o->residual_history_len);
    if (!isfinite(rel_res)) {
        numeric_err = SPARSE_ERR_NUMERIC;
        goto done_mf;
    }

    if (rel_res <= o->tol) {
        converged = 1;
        goto done_mf;
    }
    if (o->max_iter == 0)
        goto done_mf;

    rho = vec_dot(ws.r_hat, ws.r, n);
    if (!isfinite(rho)) {
        numeric_err = SPARSE_ERR_NUMERIC;
        goto done_mf;
    }
    if (fabs(rho) < sparse_rel_tol(0, DROP_TOL)) {
        breakdown = 1;
        goto done_mf;
    }

    for (iter = 0; iter < o->max_iter; iter++) {
        reshist_record(&rh, rel_res);
        iter_report(o->callback, o->callback_ctx, o->verbose, "BiCGSTAB", iter, rel_res);

        double *p_eff = ws.p;
        if (precond) {
            sparse_err_t perr = precond(precond_ctx, n, ws.p, ws.p_hat);
            if (perr != SPARSE_OK) {
                sparse_iter_stag_free(&stag);
                bicgstab_workspace_free(&ws);
                return perr;
            }
            p_eff = ws.p_hat;
        }

        merr = matvec(matvec_ctx, n, p_eff, ws.v);
        if (merr != SPARSE_OK) {
            sparse_iter_stag_free(&stag);
            bicgstab_workspace_free(&ws);
            return merr;
        }

        double r_hat_v = vec_dot(ws.r_hat, ws.v, n);
        if (fabs(r_hat_v) < sparse_rel_tol(0, DROP_TOL)) {
            breakdown = 1;
            break;
        }
        double alpha = rho / r_hat_v;
        if (!isfinite(alpha)) {
            numeric_err = SPARSE_ERR_NUMERIC;
            break;
        }

        vec_copy(ws.r, ws.s, n);
        vec_axpy(-alpha, ws.v, ws.s, n);

        double snorm = vec_norm2(ws.s, n);
        double srel = s217_iter_relative_residual(snorm, bnorm);
        if (!isfinite(srel)) {
            numeric_err = SPARSE_ERR_NUMERIC;
            break;
        }
        if (srel <= o->tol) {
            vec_axpy(alpha, p_eff, x, n);
            if (!s217_iter_vector_finite(x, n)) {
                numeric_err = SPARSE_ERR_NUMERIC;
                break;
            }
            rel_res = srel;
            converged = 1;
            iter++;
            break;
        }

        double *s_eff = ws.s;
        if (precond) {
            sparse_err_t perr = precond(precond_ctx, n, ws.s, ws.s_hat);
            if (perr != SPARSE_OK) {
                sparse_iter_stag_free(&stag);
                bicgstab_workspace_free(&ws);
                return perr;
            }
            s_eff = ws.s_hat;
        }

        merr = matvec(matvec_ctx, n, s_eff, ws.t);
        if (merr != SPARSE_OK) {
            sparse_iter_stag_free(&stag);
            bicgstab_workspace_free(&ws);
            return merr;
        }

        double tt = vec_dot(ws.t, ws.t, n);
        if (tt < sparse_rel_tol(0, DROP_TOL)) {
            breakdown = 1;
            break;
        }
        double omega = vec_dot(ws.t, ws.s, n) / tt;
        if (!isfinite(omega)) {
            numeric_err = SPARSE_ERR_NUMERIC;
            break;
        }

        if (fabs(omega) < 1e-15 * fabs(alpha)) {
            vec_axpy(alpha, p_eff, x, n);
            merr = matvec(matvec_ctx, n, x, ws.r);
            if (merr != SPARSE_OK) {
                sparse_iter_stag_free(&stag);
                bicgstab_workspace_free(&ws);
                return merr;
            }
            for (idx_t i = 0; i < n; i++)
                ws.r[i] = b[i] - ws.r[i];
            rnorm = vec_norm2(ws.r, n);
            rel_res = s217_iter_relative_residual(rnorm, bnorm);
            if (!isfinite(rel_res) || !s217_iter_vector_finite(x, n)) {
                numeric_err = SPARSE_ERR_NUMERIC;
                break;
            }
            vec_copy(ws.r, ws.p, n);
            rho = vec_dot(ws.r_hat, ws.r, n);
            if (fabs(rho) < sparse_rel_tol(0, DROP_TOL)) {
                breakdown = 1;
                break;
            }
            continue;
        }

        vec_axpy(alpha, p_eff, x, n);
        vec_axpy(omega, s_eff, x, n);

        vec_copy(ws.s, ws.r, n);
        vec_axpy(-omega, ws.t, ws.r, n);
        rnorm = vec_norm2(ws.r, n);
        rel_res = s217_iter_relative_residual(rnorm, bnorm);
        if (!isfinite(rel_res) || !s217_iter_vector_finite(x, n)) {
            numeric_err = SPARSE_ERR_NUMERIC;
            break;
        }

        if (rel_res <= o->tol) {
            converged = 1;
            iter++;
            break;
        }

        /* Stagnation check */
        sparse_iter_stag_record(&stag, rel_res);
        if (sparse_iter_stag_check(&stag)) {
            stagnated = 1;
            break;
        }

        double rho_new = vec_dot(ws.r_hat, ws.r, n);
        if (fabs(rho_new) < sparse_rel_tol(0, DROP_TOL)) {
            breakdown = 1;
            break;
        }

        double beta = (rho_new / rho) * (alpha / omega);
        if (!isfinite(beta)) {
            numeric_err = SPARSE_ERR_NUMERIC;
            break;
        }

        for (idx_t i = 0; i < n; i++)
            ws.p[i] = ws.r[i] + beta * (ws.p[i] - omega * ws.v[i]);

        rho = rho_new;
    }

done_mf:;
    double true_rel_res = rel_res;
    if (iter > 0 && numeric_err == SPARSE_OK) {
        merr = matvec(matvec_ctx, n, x, ws.r);
        if (merr != SPARSE_OK) {
            sparse_iter_stag_free(&stag);
            bicgstab_workspace_free(&ws);
            return merr;
        }
        for (idx_t i = 0; i < n; i++)
            ws.r[i] = b[i] - ws.r[i];
        true_rel_res = s217_iter_relative_residual(vec_norm2(ws.r, n), bnorm);
        if (true_rel_res > o->tol)
            converged = 0;
    }
    if (!isfinite(true_rel_res) || numeric_err != SPARSE_OK) {
        numeric_err = SPARSE_ERR_NUMERIC;
        converged = 0;
        true_rel_res = NAN;
    }

    if (result) {
        result->iterations = iter;
        result->residual_norm = true_rel_res;
        result->converged = converged;
        result->stagnated = stagnated;
        result->breakdown = breakdown;
        result->residual_history_count = rh.count < rh.len ? rh.count : rh.len;
    }

    sparse_iter_stag_free(&stag);
    bicgstab_workspace_free(&ws);
    if (numeric_err != SPARSE_OK)
        return numeric_err;
    return converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
}
