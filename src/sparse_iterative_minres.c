#include "sparse_iterative_internal.h"
#include "sparse_matrix_internal.h"
#include "sparse_vector.h"
#include <math.h>

static const sparse_iter_opts_t minres_defaults = {
    .max_iter = 1000,
    .tol = 1e-10,
    .verbose = 0,
};

sparse_err_t sparse_solve_minres_with_workspace_internal(const SparseMatrix *A, const double *b,
                                                         double *x, const sparse_iter_opts_t *opts,
                                                         sparse_precond_fn precond,
                                                         const void *precond_ctx,
                                                         sparse_iter_result_t *result,
                                                         sparse_iter_workspace_t *workspace) {
    if (result) {
        result->iterations = 0;
        result->residual_norm = 0.0;
        result->converged = 0;
        result->stagnated = 0;
        result->residual_history_count = 0;
        result->breakdown = 0;
    }

    if (!A || !b || !x)
        return SPARSE_ERR_NULL;
    if (!workspace)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    const sparse_iter_opts_t *o = opts ? opts : &minres_defaults;
    if (o->max_iter < 0 || o->tol < 0.0)
        return SPARSE_ERR_BADARG;

    idx_t n = A->rows;

    if (n == 0) {
        if (result)
            result->converged = 1;
        return SPARSE_OK;
    }
    if (!s217_iter_vector_finite(b, n) || !s217_iter_vector_finite(x, n))
        return SPARSE_ERR_BADARG;

    double bnorm = vec_norm2(b, n);
    if (bnorm == 0.0) {
        vec_zero(x, n);
        if (result) {
            result->converged = 1;
            result->residual_norm = 0.0;
        }
        return SPARSE_OK;
    }
    if (!isfinite(bnorm)) {
        s217_iter_numeric_result(result, 0, 0);
        return SPARSE_ERR_NUMERIC;
    }

    sparse_minres_workspace_view_t minres_ws;
    if (sparse_iter_workspace_prepare_minres(workspace, n, precond != NULL, &minres_ws) !=
        SPARSE_OK)
        return SPARSE_ERR_ALLOC;
    double *v = minres_ws.v;
    double *v_old = minres_ws.v_old;
    double *w = minres_ws.w;
    double *d0 = minres_ws.d0;
    double *d1 = minres_ws.d1;
    double *d2 = minres_ws.d2;
    double *z = minres_ws.z;
    double *z_tmp = minres_ws.z_tmp;

    stag_tracker_t stag;
    if (sparse_iter_stag_init(&stag, o->stagnation_window) != SPARSE_OK)
        return SPARSE_ERR_ALLOC;

    idx_t iter = 0;
    int converged = 0;
    int stagnated = 0;
    int breakdown = 0;
    double true_res_cached = -1.0;
    reshist_t rh = reshist_make(o->residual_history, o->residual_history_len);

    sparse_matvec(A, x, w);
    for (idx_t i = 0; i < n; i++)
        v[i] = b[i] - w[i];

    double r0norm = vec_norm2(v, n);
    double r0rel = s217_iter_relative_residual(r0norm, bnorm);
    if (!isfinite(r0rel))
        goto minres_numeric_failure;
    if (r0rel <= o->tol) {
        if (result) {
            result->converged = 1;
            result->residual_norm = r0rel;
        }
        sparse_iter_stag_free(&stag);
        return SPARSE_OK;
    }

    double beta;
    if (precond) {
        sparse_err_t perr = precond(precond_ctx, n, v, z);
        if (perr != SPARSE_OK) {
            sparse_iter_stag_free(&stag);
            return perr;
        }
        beta = vec_dot(v, z, n);
        if (!isfinite(beta))
            goto minres_numeric_failure;
        if (beta < 0.0) {
            sparse_iter_stag_free(&stag);
            return SPARSE_ERR_BADARG;
        }
        beta = sqrt(beta);
        if (beta <= 0.0) {
            if (result) {
                result->breakdown = 1;
                result->residual_norm = r0rel;
            }
            sparse_iter_stag_free(&stag);
            return SPARSE_ERR_NOT_CONVERGED;
        }
    } else {
        beta = r0norm;
    }

    {
        double inv_beta = 1.0 / beta;
        for (idx_t i = 0; i < n; i++)
            v[i] *= inv_beta;
        if (precond) {
            for (idx_t i = 0; i < n; i++)
                z[i] *= inv_beta;
        }
        if (!s217_iter_vector_finite(v, n) || (precond && !s217_iter_vector_finite(z, n)))
            goto minres_numeric_failure;
    }

    double cs = 1.0, sn = 0.0;
    double cs_old = 1.0, sn_old = 0.0;
    double phi_bar = beta;
    double beta_old = 0.0;

    double minres_phase_start_s = o->progress_cb ? s29_iter_now_s() : 0.0;

    for (iter = 1; iter <= o->max_iter; iter++) {
        if (o->progress_cb) {
            sparse_progress_t pp = {
                .phase = "minres",
                .step = iter - 1,
                .total = o->max_iter,
                .elapsed_s = s29_iter_now_s() - minres_phase_start_s,
            };
            if (o->progress_cb(&pp, o->progress_user) != 0) {
                if (result)
                    result->iterations = iter - 1;
                sparse_iter_stag_free(&stag);
                return SPARSE_ERR_CANCELLED;
            }
        }

        if (precond)
            sparse_matvec(A, z, w);
        else
            sparse_matvec(A, v, w);

        double alpha = precond ? vec_dot(z, w, n) : vec_dot(v, w, n);
        if (!isfinite(alpha))
            goto minres_numeric_failure;

        for (idx_t i = 0; i < n; i++)
            w[i] = w[i] - alpha * v[i] - beta_old * v_old[i];

        double beta_new;
        if (precond) {
            sparse_err_t perr = precond(precond_ctx, n, w, z_tmp);
            if (perr != SPARSE_OK) {
                sparse_iter_stag_free(&stag);
                return perr;
            }
            double inner = vec_dot(w, z_tmp, n);
            if (!isfinite(inner))
                goto minres_numeric_failure;
            if (inner < 0.0) {
                sparse_iter_stag_free(&stag);
                return SPARSE_ERR_BADARG;
            }
            beta_new = sqrt(inner);
        } else {
            beta_new = vec_norm2(w, n);
        }
        if (!isfinite(beta_new))
            goto minres_numeric_failure;

        double eps = sn_old * beta_old;
        double delta_bar = cs_old * beta_old;
        double delta = cs * delta_bar + sn * alpha;
        double gamma_bar = -sn * delta_bar + cs * alpha;
        double gamma = hypot(gamma_bar, beta_new);
        if (!isfinite(gamma))
            goto minres_numeric_failure;

        if (gamma < sparse_rel_tol(0, DROP_TOL)) {
            breakdown = 1;
            iter--;
            break;
        }

        double cs_new = gamma_bar / gamma;
        double sn_new = beta_new / gamma;
        double phi = cs_new * phi_bar;
        phi_bar = -sn_new * phi_bar;
        if (!isfinite(phi) || !isfinite(phi_bar))
            goto minres_numeric_failure;

        {
            const double *dv = precond ? z : v;
            double inv_gamma = 1.0 / gamma;
            for (idx_t i = 0; i < n; i++)
                d0[i] = (dv[i] - eps * d2[i] - delta * d1[i]) * inv_gamma;
        }

        for (idx_t i = 0; i < n; i++)
            x[i] += phi * d0[i];
        if (!s217_iter_vector_finite(x, n))
            goto minres_numeric_failure;

        double relres = s217_iter_relative_residual(fabs(phi_bar), bnorm);
        if (!isfinite(relres))
            goto minres_numeric_failure;

        iter_report(o->callback, o->callback_ctx, o->verbose, "MINRES", iter - 1, relres);
        reshist_record(&rh, relres);

        if (relres <= o->tol) {
            sparse_matvec(A, x, d2);
            for (idx_t i = 0; i < n; i++)
                d2[i] -= b[i];
            double verified_res = s217_iter_relative_residual(vec_norm2(d2, n), bnorm);
            if (!isfinite(verified_res))
                goto minres_numeric_failure;
            if (verified_res <= o->tol) {
                true_res_cached = verified_res;
                break;
            }
        }

        sparse_iter_stag_record(&stag, relres);
        if (sparse_iter_stag_check(&stag)) {
            stagnated = 1;
            break;
        }

        cs_old = cs;
        sn_old = sn;
        cs = cs_new;
        sn = sn_new;

        {
            double *tmp = d2;
            d2 = d1;
            d1 = d0;
            d0 = tmp;
        }

        if (beta_new > sparse_rel_tol(0, DROP_TOL)) {
            double inv_beta = 1.0 / beta_new;
            for (idx_t i = 0; i < n; i++) {
                v_old[i] = v[i];
                v[i] = w[i] * inv_beta;
            }
            if (precond) {
                for (idx_t i = 0; i < n; i++)
                    z[i] = z_tmp[i] * inv_beta;
            }
        } else {
            breakdown = 1;
            break;
        }

        beta_old = beta_new;
    }

    double true_res;
    if (true_res_cached >= 0.0) {
        true_res = true_res_cached;
    } else {
        sparse_matvec(A, x, w);
        for (idx_t i = 0; i < n; i++)
            w[i] -= b[i];
        true_res = s217_iter_relative_residual(vec_norm2(w, n), bnorm);
    }

    if (!isfinite(true_res))
        goto minres_numeric_failure;

    converged = (true_res <= o->tol);

    if (result) {
        result->iterations = iter > o->max_iter ? o->max_iter : iter;
        result->residual_norm = true_res;
        result->converged = converged;
        result->stagnated = stagnated;
        result->breakdown = breakdown;
        result->residual_history_count = rh.count < rh.len ? rh.count : rh.len;
    }

    sparse_iter_stag_free(&stag);
    return converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;

minres_numeric_failure:
    s217_iter_numeric_result(result, iter, rh.count < rh.len ? rh.count : rh.len);
    sparse_iter_stag_free(&stag);
    return SPARSE_ERR_NUMERIC;
}

sparse_err_t sparse_solve_minres(const SparseMatrix *A, const double *b, double *x,
                                 const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                                 const void *precond_ctx, sparse_iter_result_t *result) {
    sparse_iter_workspace_t workspace;
    sparse_iter_workspace_init(&workspace);
    sparse_err_t err = sparse_solve_minres_with_workspace_internal(A, b, x, opts, precond,
                                                                   precond_ctx, result, &workspace);
    sparse_iter_workspace_free(&workspace);
    return err;
}

sparse_err_t sparse_solve_minres_with_handle(const SparseMatrix *A, const double *b, double *x,
                                             const sparse_iter_opts_t *opts,
                                             sparse_precond_fn precond, const void *precond_ctx,
                                             sparse_iter_result_t *result,
                                             sparse_iter_handle_t *handle) {
    sparse_iter_workspace_t *workspace = NULL;
    sparse_err_t err = s49_iter_handle_ensure(handle, &workspace);
    if (err != SPARSE_OK)
        return err;
    return sparse_solve_minres_with_workspace_internal(A, b, x, opts, precond, precond_ctx, result,
                                                       workspace);
}
