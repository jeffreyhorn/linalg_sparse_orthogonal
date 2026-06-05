#include "sparse_eigs_internal.h"

#include <float.h>
#include <math.h>
#include <string.h>

/* Deterministic pseudo-random initial X for LOBPCG.
 * Different columns use distinct irrational shifts so the starting
 * block is reproducible but not trivially collinear. */
static void s21_lobpcg_init_X(double *X, idx_t n, idx_t bs) {
    for (idx_t j = 0; j < bs; j++) {
        double *col = X + (size_t)j * (size_t)n;
        for (idx_t i = 0; i < n; i++) {
            double x = (double)(i + 1) * 0.618033988749895 + (double)(j + 1) * 0.31415926535897932;
            col[i] = 0.3 + (x - floor(x));
        }
    }
}

sparse_err_t s21_lobpcg_orthonormalize_block(double *Q, idx_t n, idx_t block_size_in,
                                             idx_t *block_size_out) {
    if (!Q || !block_size_out)
        return SPARSE_ERR_NULL;
    if (n < 1 || block_size_in < 0)
        return SPARSE_ERR_BADARG;

    if (block_size_in == 0) {
        *block_size_out = 0;
        return SPARSE_OK;
    }

    double scale = 0.0;
    idx_t accepted = 0;

    for (idx_t j = 0; j < block_size_in; j++) {
        double *col_in = Q + (size_t)j * (size_t)n;
        double *col_out = Q + (size_t)accepted * (size_t)n;

        if (accepted != j) {
            for (idx_t i = 0; i < n; i++)
                col_out[i] = col_in[i];
        }

        double sq_in = 0.0;
        for (idx_t i = 0; i < n; i++)
            sq_in += col_out[i] * col_out[i];
        double norm_in = sqrt(sq_in);
        if (norm_in > scale)
            scale = norm_in;

        s21_mgs_reorth(col_out, Q, n, accepted);

        double sq_out = 0.0;
        for (idx_t i = 0; i < n; i++)
            sq_out += col_out[i] * col_out[i];
        double norm_out = sqrt(sq_out);

        double breakdown_tol = scale * 1e-14;
        if (breakdown_tol < DBL_MIN * 100.0)
            breakdown_tol = DBL_MIN * 100.0;

        if (norm_out > breakdown_tol) {
            double inv = 1.0 / norm_out;
            for (idx_t i = 0; i < n; i++)
                col_out[i] *= inv;
            accepted++;
        }
    }

    *block_size_out = accepted;
    return SPARSE_OK;
}

sparse_err_t s21_lobpcg_rr_step(lanczos_op_fn op, const void *ctx, idx_t n, idx_t block_size,
                                sparse_eigs_lobpcg_workspace_view_t *view,
                                sparse_eigs_which_t which, int use_p) {
    if (!op || !view || !view->Q || !view->AQ || !view->G || !view->Y || !view->theta_full ||
        !view->sel_idx || !view->X_new || !view->X || !view->W || !view->theta)
        return SPARSE_ERR_NULL;
    if (n < 1 || block_size < 1)
        return SPARSE_ERR_BADARG;

    double *X = view->X;
    double *W = view->W;
    double *P = use_p ? view->P : NULL;
    double *Q = view->Q;
    double *AQ = view->AQ;
    double *G = view->G;
    double *Y = view->Y;
    double *theta_full = view->theta_full;
    idx_t *sel_idx = view->sel_idx;
    double *X_new = view->X_new;
    double *P_new = use_p ? view->P_new : NULL;
    double *theta_out = view->theta;

    if (use_p && (!P || !P_new))
        return SPARSE_ERR_NULL;

    int has_p = use_p != 0;
    idx_t cap = has_p ? 3 * block_size : 2 * block_size;
    if (cap > n)
        return SPARSE_ERR_BADARG;

    size_t nb = (size_t)n * (size_t)block_size;
    memcpy(Q, X, nb * sizeof(double));
    memcpy(Q + nb, W, nb * sizeof(double));
    if (has_p)
        memcpy(Q + 2 * nb, P, nb * sizeof(double));

    idx_t K_eff = 0;
    sparse_err_t err = s21_lobpcg_orthonormalize_block(Q, n, cap, &K_eff);
    if (err != SPARSE_OK)
        return err;
    if (K_eff < block_size)
        return SPARSE_ERR_BADARG;

    for (idx_t j = 0; j < K_eff; j++) {
        err = op(ctx, n, Q + (size_t)j * (size_t)n, AQ + (size_t)j * (size_t)n);
        if (err != SPARSE_OK)
            return err;
    }

    for (idx_t i = 0; i < K_eff; i++) {
        const double *qi = Q + (size_t)i * (size_t)n;
        for (idx_t j = 0; j < K_eff; j++) {
            const double *aqj = AQ + (size_t)j * (size_t)n;
            double s = 0.0;
            for (idx_t r = 0; r < n; r++)
                s += qi[r] * aqj[r];
            G[(size_t)i + (size_t)j * (size_t)K_eff] = s;
        }
    }
    for (idx_t i = 0; i < K_eff; i++) {
        for (idx_t j = i + 1; j < K_eff; j++) {
            size_t ij = (size_t)i + (size_t)j * (size_t)K_eff;
            size_t ji = (size_t)j + (size_t)i * (size_t)K_eff;
            double avg = 0.5 * (G[ij] + G[ji]);
            G[ij] = avg;
            G[ji] = avg;
        }
    }

    err = s21_dense_sym_jacobi(G, K_eff, theta_full, Y);
    if (err != SPARSE_OK)
        return err;

    idx_t take = s20_select_indices(theta_full, K_eff, which, block_size, sel_idx);
    if (take < block_size)
        return SPARSE_ERR_NOT_CONVERGED;

    double scale_theta = 0.0;
    for (idx_t l = 0; l < K_eff; l++) {
        double a = fabs(theta_full[l]);
        if (a > scale_theta)
            scale_theta = a;
    }
    int gram_singular = 0;
    if (scale_theta > 0.0) {
        double cond_floor = scale_theta * 1e-12;
        for (idx_t l = 0; l < K_eff; l++) {
            if (fabs(theta_full[l]) < cond_floor) {
                gram_singular = 1;
                break;
            }
        }
    }

    for (idx_t j = 0; j < block_size; j++) {
        const double *yj = Y + (size_t)sel_idx[j] * (size_t)K_eff;
        double *xn = X_new + (size_t)j * (size_t)n;
        for (idx_t i = 0; i < n; i++)
            xn[i] = 0.0;
        for (idx_t c = 0; c < K_eff; c++) {
            double yc = yj[c];
            if (yc == 0.0)
                continue;
            const double *qc = Q + (size_t)c * (size_t)n;
            for (idx_t i = 0; i < n; i++)
                xn[i] += yc * qc[i];
        }
    }

    if (has_p) {
        if (gram_singular) {
            memset(P_new, 0, nb * sizeof(double));
        } else {
            for (idx_t j = 0; j < block_size; j++) {
                const double *xn = X_new + (size_t)j * (size_t)n;
                double *pn = P_new + (size_t)j * (size_t)n;
                for (idx_t i = 0; i < n; i++)
                    pn[i] = xn[i];
                for (idx_t l = 0; l < block_size; l++) {
                    const double *xl = X + (size_t)l * (size_t)n;
                    double dot = 0.0;
                    for (idx_t i = 0; i < n; i++)
                        dot += xn[i] * xl[i];
                    for (idx_t i = 0; i < n; i++)
                        pn[i] -= dot * xl[i];
                }
            }
        }
    }

    memcpy(X, X_new, nb * sizeof(double));
    if (has_p)
        memcpy(P, P_new, nb * sizeof(double));
    for (idx_t j = 0; j < block_size; j++)
        theta_out[j] = theta_full[sel_idx[j]];
    return SPARSE_OK;
}

sparse_err_t s21_lobpcg_solve(lanczos_op_fn op, const void *ctx, idx_t n, idx_t k,
                              const sparse_eigs_opts_t *o, double eff_tol, idx_t max_iters,
                              sparse_eigs_t *result, sparse_eigs_workspace_t *workspace) {
    if (!op || !o || !result || !result->eigenvalues)
        return SPARSE_ERR_NULL;
    if (n < 1 || k < 1 || max_iters < 1)
        return SPARSE_ERR_BADARG;

    idx_t bs = (o->block_size > 0) ? o->block_size : k;
    if (bs > n)
        bs = n;
    if (bs < k)
        return SPARSE_ERR_BADARG;

    result->peak_basis_size = 10 * bs;

    sparse_eigs_workspace_t local_ws;
    sparse_eigs_lobpcg_workspace_view_t lobpcg_view;
    sparse_eigs_workspace_t *lobpcg_ws = workspace ? workspace : &local_ws;
    if (!workspace)
        sparse_eigs_workspace_init(lobpcg_ws);

    sparse_err_t rc = SPARSE_ERR_NOT_CONVERGED;
    idx_t total_iters = 0;
    double last_res_rel = 0.0;
    int have_p = 0;

    sparse_err_t ws_err = sparse_eigs_workspace_prepare_lobpcg(lobpcg_ws, n, bs, 1, &lobpcg_view);
    if (ws_err != SPARSE_OK) {
        rc = ws_err;
        goto cleanup;
    }

    double *X = lobpcg_view.X;
    double *R = lobpcg_view.R;
    double *W = lobpcg_view.W;
    double *P = lobpcg_view.P;
    double *AX = lobpcg_view.AX;
    double *theta = lobpcg_view.theta;
    int *converged = lobpcg_view.converged;

    s21_lobpcg_init_X(X, n, bs);
    idx_t bs_eff = 0;
    sparse_err_t err = s21_lobpcg_orthonormalize_block(X, n, bs, &bs_eff);
    if (err != SPARSE_OK) {
        rc = err;
        goto cleanup;
    }
    if (bs_eff < k) {
        rc = SPARSE_ERR_BADARG;
        goto cleanup;
    }
    bs = bs_eff;
    size_t nb_bytes = (size_t)n * (size_t)bs * sizeof(double);

    for (idx_t j = 0; j < bs; j++) {
        err = op(ctx, n, X + (size_t)j * (size_t)n, AX + (size_t)j * (size_t)n);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }
    }
    for (idx_t j = 0; j < bs; j++) {
        const double *xj = X + (size_t)j * (size_t)n;
        const double *axj = AX + (size_t)j * (size_t)n;
        double s = 0.0;
        for (idx_t i = 0; i < n; i++)
            s += xj[i] * axj[i];
        theta[j] = s;
    }

    double lobpcg_phase_start_s = o->progress_cb ? s29_eigs_now_s() : 0.0;
    for (idx_t iter = 0; iter < max_iters; iter++) {
        if (o->progress_cb) {
            sparse_progress_t pp = {
                .phase = "lobpcg",
                .step = iter,
                .total = max_iters,
                .elapsed_s = s29_eigs_now_s() - lobpcg_phase_start_s,
            };
            if (o->progress_cb(&pp, o->progress_user) != 0) {
                rc = SPARSE_ERR_CANCELLED;
                goto cleanup;
            }
        }

        total_iters = iter + 1;

        for (idx_t j = 0; j < bs; j++) {
            const double *xj = X + (size_t)j * (size_t)n;
            const double *axj = AX + (size_t)j * (size_t)n;
            double *rj = R + (size_t)j * (size_t)n;
            double tj = theta[j];
            for (idx_t i = 0; i < n; i++)
                rj[i] = axj[i] - tj * xj[i];
        }

        double scale = 0.0;
        for (idx_t j = 0; j < bs; j++) {
            double a = fabs(theta[j]);
            if (a > scale)
                scale = a;
        }
        double max_res_rel = 0.0;
        idx_t n_locked = 0;
        for (idx_t j = 0; j < bs; j++) {
            const double *rj = R + (size_t)j * (size_t)n;
            double sq = 0.0;
            for (idx_t i = 0; i < n; i++)
                sq += rj[i] * rj[i];
            double r_norm = sqrt(sq);
            double anchor = fabs(theta[j]);
            if (anchor < scale * 1e-12)
                anchor = scale > 0.0 ? scale : 1.0;
            double rel = r_norm / anchor;
            if (rel > max_res_rel)
                max_res_rel = rel;
            if (rel <= eff_tol)
                converged[j] = 1;
            if (converged[j])
                n_locked++;
        }
        last_res_rel = max_res_rel;
        if (max_res_rel <= eff_tol) {
            rc = SPARSE_OK;
            break;
        }

        if (o->precond) {
            for (idx_t j = 0; j < bs; j++) {
                const double *rj = R + (size_t)j * (size_t)n;
                double *wj = W + (size_t)j * (size_t)n;
                err = o->precond(o->precond_ctx, n, rj, wj);
                if (err != SPARSE_OK) {
                    rc = err;
                    goto cleanup;
                }
            }
        } else {
            memcpy(W, R, nb_bytes);
        }

        if (o->lobpcg_soft_lock && n_locked > 0) {
            for (idx_t j = 0; j < bs; j++) {
                if (!converged[j])
                    continue;
                memset(W + (size_t)j * (size_t)n, 0, (size_t)n * sizeof(double));
                if (P)
                    memset(P + (size_t)j * (size_t)n, 0, (size_t)n * sizeof(double));
            }
        }

        err = s21_lobpcg_rr_step(op, ctx, n, bs, &lobpcg_view, o->which, have_p);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }
        have_p = 1;

        for (idx_t j = 0; j < bs; j++) {
            err = op(ctx, n, X + (size_t)j * (size_t)n, AX + (size_t)j * (size_t)n);
            if (err != SPARSE_OK) {
                rc = err;
                goto cleanup;
            }
        }
    }

    idx_t emit = (k < bs) ? k : bs;
    for (idx_t j = 0; j < emit; j++) {
        double t = theta[j];
        result->eigenvalues[j] = (o->which == SPARSE_EIGS_NEAREST_SIGMA) ? (o->sigma + 1.0 / t) : t;
    }
    if (o->compute_vectors) {
        for (idx_t j = 0; j < emit; j++) {
            const double *xj = X + (size_t)j * (size_t)n;
            double *vj = result->eigenvectors + (size_t)j * (size_t)n;
            for (idx_t i = 0; i < n; i++)
                vj[i] = xj[i];
        }
    }
    result->n_converged = emit;
    result->iterations = total_iters;
    result->residual_norm = last_res_rel;

cleanup:
    if (!workspace)
        sparse_eigs_workspace_free(lobpcg_ws);
    return rc;
}
