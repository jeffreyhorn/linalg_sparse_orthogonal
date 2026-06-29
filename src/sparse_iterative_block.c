#include "sparse_alloc_internal.h"
#include "sparse_iterative.h"
#include "sparse_iterative_internal.h"
#include "sparse_matrix_internal.h"
#include "sparse_vector.h"
#include <stdint.h>
#include <stdlib.h>

/*
 * Iterative block-solver implementations. Scalar CG, GMRES, MINRES, and
 * BiCGStab remain in their scalar owners; this file owns the multiple-RHS
 * public entry points and their per-column adapter glue.
 */

/* ═══════════════════════════════════════════════════════════════════════
 * Block Conjugate Gradient (multiple RHS)
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_cg_solve_block(const SparseMatrix *A, const double *B, idx_t nrhs, double *X,
                                   const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                                   const void *precond_ctx, sparse_iter_result_t *result) {
    s85_iter_result_reset(result);

    if (!A || !B || !X)
        return SPARSE_ERR_NULL;
    if (nrhs < 0)
        return SPARSE_ERR_BADARG;
    if (nrhs == 0) {
        s85_iter_result_mark_converged(result);
        return SPARSE_OK;
    }
    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;

    const sparse_iter_opts_t *o = opts ? opts : s85_iter_cg_defaults();
    if (o->max_iter < 0 || o->tol < 0.0)
        return SPARSE_ERR_BADARG;

    idx_t n = sparse_rows(A);
    if (n == 0) {
        s85_iter_result_mark_converged(result);
        return SPARSE_OK;
    }

    /* Upfront overflow guards — must run before any n*k pointer arithmetic */
    size_t n_size = 0;
    size_t nrhs_size = 0;
    size_t blk = 0;
    if (sparse_idx_to_size_checked(n, &n_size) || sparse_idx_to_size_checked(nrhs, &nrhs_size) ||
        sparse_size_mul_overflow(n_size, nrhs_size, &blk))
        return SPARSE_ERR_ALLOC;
    if (blk > (size_t)INT32_MAX)
        return SPARSE_ERR_ALLOC;

    sparse_iter_workspace_t workspace;
    sparse_block_cg_workspace_view_t block_ws;
    sparse_iter_workspace_init(&workspace);
    if (sparse_iter_workspace_prepare_block_cg(&workspace, n, nrhs, &block_ws) != SPARSE_OK)
        return SPARSE_ERR_ALLOC;
    double *R = block_ws.R;
    double *Z = block_ws.Z;
    double *P = block_ws.P;
    double *AP = block_ws.AP;
    double *bnorms = block_ws.bnorms;
    double *rz = block_ws.rz;
    int *conv = block_ws.conv;
    double *rnorms = block_ws.rnorms;

    /* Compute ||B(:,k)|| for each column */
    for (idx_t k = 0; k < nrhs; k++) {
        size_t off = n_size * (size_t)k;
        bnorms[k] = vec_norm2(&B[off], n);
        if (bnorms[k] == 0.0) {
            vec_zero(&X[off], n);
            bnorms[k] = 1.0; /* avoid div-by-zero; already converged */
        }
    }

    /* R = B - A*X (initial residual for all columns) */
    {
        sparse_err_t mv_err = sparse_matvec_block(A, X, nrhs, AP);
        if (mv_err != SPARSE_OK) {
            sparse_iter_workspace_free(&workspace);
            return mv_err;
        }
    }
    for (idx_t k = 0; k < nrhs; k++)
        for (idx_t i = 0; i < n; i++)
            R[i + n * k] = B[i + n * k] - AP[i + n * k];

    /* Apply preconditioner: Z = M^{-1}*R (or Z = R) */
    for (idx_t k = 0; k < nrhs; k++) {
        if (precond) {
            sparse_err_t perr = precond(precond_ctx, n, &R[n * k], &Z[n * k]); // NOLINT
            if (perr != SPARSE_OK) {
                sparse_iter_workspace_free(&workspace);
                return perr;
            }
        } else {
            vec_copy(&R[n * k], &Z[n * k], n); // NOLINT
        }
    }

    /* P = Z, compute rz = R^T*Z per column */
    for (idx_t k = 0; k < nrhs; k++) {
        vec_copy(&Z[n * k], &P[n * k], n);
        rz[k] = vec_dot(&R[n * k], &Z[n * k], n);
        rnorms[k] = vec_norm2(&R[n * k], n);
    }

    idx_t max_iter_done = 0;
    int all_converged = 0;

    for (idx_t iter = 0; iter < o->max_iter; iter++) {
        /* Check convergence for all columns */
        all_converged = 1;
        for (idx_t k = 0; k < nrhs; k++) {
            if (!conv[k] && rnorms[k] / bnorms[k] <= o->tol)
                conv[k] = 1;
            if (!conv[k])
                all_converged = 0;
        }
        if (all_converged)
            break;

        max_iter_done = iter + 1;

        /* AP = A*P (shared SpMV for all columns) */
        {
            sparse_err_t mv_err = sparse_matvec_block(A, P, nrhs, AP);
            if (mv_err != SPARSE_OK) {
                sparse_iter_workspace_free(&workspace);
                return mv_err;
            }
        }

        for (idx_t k = 0; k < nrhs; k++) {
            if (conv[k])
                continue;

            /* alpha = rz / (P^T * AP) */
            double pAp = vec_dot(&P[n * k], &AP[n * k], n);
            if (pAp == 0.0)
                continue; /* breakdown for this column */
            double alpha = rz[k] / pAp;

            /* X(:,k) += alpha * P(:,k) */
            vec_axpy(alpha, &P[n * k], &X[n * k], n);

            /* R(:,k) -= alpha * AP(:,k) */
            vec_axpy(-alpha, &AP[n * k], &R[n * k], n);

            rnorms[k] = vec_norm2(&R[n * k], n);

            /* Z(:,k) = M^{-1} * R(:,k) */
            if (precond) {
                sparse_err_t perr = precond(precond_ctx, n, &R[n * k], &Z[n * k]);
                if (perr != SPARSE_OK) {
                    sparse_iter_workspace_free(&workspace);
                    return perr;
                }
            } else {
                vec_copy(&R[n * k], &Z[n * k], n);
            }

            /* beta = rz_new / rz_old */
            double rz_new = vec_dot(&R[n * k], &Z[n * k], n);
            double beta = (rz[k] != 0.0) ? rz_new / rz[k] : 0.0;

            /* P(:,k) = Z(:,k) + beta * P(:,k) */
            for (idx_t i = 0; i < n; i++)
                P[i + n * k] = Z[i + n * k] + beta * P[i + n * k];

            rz[k] = rz_new;
        }
    }

    /* Final convergence check */
    if (!all_converged) {
        all_converged = 1;
        for (idx_t k = 0; k < nrhs; k++) {
            if (!conv[k] && rnorms[k] / bnorms[k] <= o->tol)
                conv[k] = 1;
            if (!conv[k])
                all_converged = 0;
        }
    }

    if (result) {
        result->iterations = max_iter_done;
        /* Report max residual across columns */
        double max_res = 0.0;
        for (idx_t k = 0; k < nrhs; k++) {
            double rel = rnorms[k] / bnorms[k];
            if (rel > max_res)
                max_res = rel;
        }
        result->residual_norm = max_res;
        result->converged = all_converged;
    }

    sparse_iter_workspace_free(&workspace);
    return all_converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
}

typedef sparse_err_t (*iter_block_column_solver_fn)(const SparseMatrix *A, const double *b,
                                                    double *x, const void *opts,
                                                    sparse_precond_fn precond,
                                                    const void *precond_ctx,
                                                    sparse_iter_result_t *result);

static sparse_err_t solve_block_independent_columns(const SparseMatrix *A, const double *B,
                                                    idx_t nrhs, double *X, idx_t n,
                                                    const void *opts, sparse_precond_fn precond,
                                                    const void *precond_ctx,
                                                    sparse_iter_result_t *result,
                                                    iter_block_column_solver_fn solve_column) {
    idx_t max_iters = 0;
    double max_residual = 0.0;
    int all_converged = 1;
    int any_stagnated = 0;
    int any_breakdown = 0;
    sparse_err_t worst_err = SPARSE_OK;

    for (idx_t k = 0; k < nrhs; k++) {
        size_t off = (size_t)n * (size_t)k;
        sparse_iter_result_t col_result = {0, 0.0, 0, 0, 0, 0};
        sparse_err_t err =
            solve_column(A, &B[off], &X[off], opts, precond, precond_ctx, &col_result);

        if (col_result.iterations > max_iters)
            max_iters = col_result.iterations;
        if (col_result.residual_norm > max_residual)
            max_residual = col_result.residual_norm;
        if (!col_result.converged)
            all_converged = 0;
        if (col_result.stagnated)
            any_stagnated = 1;
        if (col_result.breakdown)
            any_breakdown = 1;

        if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED && worst_err == SPARSE_OK)
            worst_err = err;
    }

    if (result) {
        result->iterations = max_iters;
        result->residual_norm = max_residual;
        result->converged = all_converged;
        result->stagnated = any_stagnated;
        result->breakdown = any_breakdown;
    }

    if (worst_err != SPARSE_OK)
        return worst_err;
    return all_converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
}

static sparse_err_t solve_block_gmres_column(const SparseMatrix *A, const double *b, double *x,
                                             const void *opts, sparse_precond_fn precond,
                                             const void *precond_ctx,
                                             sparse_iter_result_t *result) {
    return sparse_solve_gmres(A, b, x, (const sparse_gmres_opts_t *)opts, precond, precond_ctx,
                              result);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Block GMRES (multiple RHS)
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_gmres_solve_block(const SparseMatrix *A, const double *B, idx_t nrhs, double *X,
                                      const sparse_gmres_opts_t *opts, sparse_precond_fn precond,
                                      const void *precond_ctx, sparse_iter_result_t *result) {
    s85_iter_result_reset(result);

    if (!A || !B || !X)
        return SPARSE_ERR_NULL;
    if (nrhs < 0)
        return SPARSE_ERR_BADARG;
    if (nrhs == 0) {
        s85_iter_result_mark_converged(result);
        return SPARSE_OK;
    }
    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;

    idx_t n = sparse_rows(A);

    /* Overflow guard for per-column offset computation */
    if (n > 0 && (size_t)nrhs > SIZE_MAX / (size_t)n)
        return SPARSE_ERR_ALLOC;
    return solve_block_independent_columns(A, B, nrhs, X, n, opts, precond, precond_ctx, result,
                                           solve_block_gmres_column);
}

static sparse_err_t solve_block_minres_column(const SparseMatrix *A, const double *b, double *x,
                                              const void *opts, sparse_precond_fn precond,
                                              const void *precond_ctx,
                                              sparse_iter_result_t *result) {
    return sparse_solve_minres(A, b, x, (const sparse_iter_opts_t *)opts, precond, precond_ctx,
                               result);
}

sparse_err_t sparse_minres_solve_block(const SparseMatrix *A, const double *B, idx_t nrhs,
                                       double *X, const sparse_iter_opts_t *opts,
                                       sparse_precond_fn precond, const void *precond_ctx,
                                       sparse_iter_result_t *result) {
    s85_iter_result_reset(result);

    if (!A || !B || !X)
        return SPARSE_ERR_NULL;
    if (nrhs < 0)
        return SPARSE_ERR_BADARG;
    if (nrhs == 0) {
        s85_iter_result_mark_converged(result);
        return SPARSE_OK;
    }
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    idx_t n = A->rows;

    /* Overflow check for j*n pointer offsets (guard before computing size_t products) */
    if (n > 0 && (size_t)nrhs > SIZE_MAX / (size_t)n)
        return SPARSE_ERR_ALLOC;

    if (n == 0) {
        s85_iter_result_mark_converged(result);
        return SPARSE_OK;
    }
    return solve_block_independent_columns(A, B, nrhs, X, n, opts, precond, precond_ctx, result,
                                           solve_block_minres_column);
}

static sparse_err_t solve_block_bicgstab_column(const SparseMatrix *A, const double *b, double *x,
                                                const void *opts, sparse_precond_fn precond,
                                                const void *precond_ctx,
                                                sparse_iter_result_t *result) {
    return sparse_solve_bicgstab(A, b, x, (const sparse_iter_opts_t *)opts, precond, precond_ctx,
                                 result);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Block BiCGSTAB — per-column independent solves
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_bicgstab_solve_block(const SparseMatrix *A, const double *B, idx_t nrhs,
                                         double *X, const sparse_iter_opts_t *opts,
                                         sparse_precond_fn precond, const void *precond_ctx,
                                         sparse_iter_result_t *result) {
    s85_iter_result_reset(result);

    if (!A || !B || !X)
        return SPARSE_ERR_NULL;
    if (nrhs < 0)
        return SPARSE_ERR_BADARG;
    if (nrhs == 0) {
        s85_iter_result_mark_converged(result);
        return SPARSE_OK;
    }
    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;

    idx_t n = sparse_rows(A);

    if (n > 0 && (size_t)nrhs > SIZE_MAX / (size_t)n)
        return SPARSE_ERR_ALLOC;

    if (n == 0) {
        s85_iter_result_mark_converged(result);
        return SPARSE_OK;
    }
    return solve_block_independent_columns(A, B, nrhs, X, n, opts, precond, precond_ctx, result,
                                           solve_block_bicgstab_column);
}
