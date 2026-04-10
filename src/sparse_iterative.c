#include "sparse_iterative.h"
#include "sparse_matrix_internal.h"
#include "sparse_vector.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

/* ═══════════════════════════════════════════════════════════════════════
 * Conjugate Gradient
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_solve_cg(const SparseMatrix *A, const double *b, double *x,
                             const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                             const void *precond_ctx, sparse_iter_result_t *result) {
    /* Initialize result to safe defaults before any early return */
    if (result) {
        result->iterations = 0;
        result->residual_norm = 0.0;
        result->converged = 0;
    }

    if (!A || !b || !x)
        return SPARSE_ERR_NULL;
    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;

    const sparse_iter_opts_t *o = opts ? opts : &cg_defaults;
    if (o->max_iter < 0 || o->tol < 0.0)
        return SPARSE_ERR_BADARG;
    idx_t n = sparse_rows(A);

    /* Trivial case: zero-size system */
    if (n == 0) {
        if (result)
            result->converged = 1;
        return SPARSE_OK;
    }

    /* Compute ||b|| for relative residual test */
    double bnorm = vec_norm2(b, n);

    /* If b == 0, solution is x = 0 */
    if (bnorm == 0.0) {
        vec_zero(x, n);
        if (result) {
            result->converged = 1;
            result->residual_norm = 0.0;
        }
        return SPARSE_OK;
    }

    /* Allocate workspace: r, z, p, Ap (4 vectors of length n) */
    if ((size_t)n > SIZE_MAX / (4 * sizeof(double)))
        return SPARSE_ERR_ALLOC;
    double *work = malloc(4 * (size_t)n * sizeof(double));
    if (!work)
        return SPARSE_ERR_ALLOC;
    double *r = work;
    double *z = work + n;
    double *p = work + 2 * n;
    double *Ap = work + 3 * n;

    /* r_0 = b - A*x_0 */
    sparse_matvec(A, x, Ap); /* Ap = A*x_0 */
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - Ap[i];

    /* Apply preconditioner: z_0 = M^{-1}*r_0 (or z_0 = r_0 if none) */
    if (precond) {
        sparse_err_t perr = precond(precond_ctx, n, r, z);
        if (perr != SPARSE_OK) {
            free(work);
            return perr;
        }
    } else {
        vec_copy(r, z, n);
    }

    /* p_0 = z_0 */
    vec_copy(z, p, n);

    double rz = vec_dot(r, z, n); /* r^T * z */
    double rnorm = vec_norm2(r, n);

    idx_t iter = 0;
    int converged = 0;

    for (iter = 0; iter < o->max_iter; iter++) {
        /* Check convergence */
        if (rnorm / bnorm <= o->tol) {
            converged = 1;
            break;
        }

        if (o->verbose) {
            fprintf(stderr, "  CG iter %4d: ||r||/||b|| = %.6e\n", (int)iter, rnorm / bnorm);
        }

        /* Ap = A*p */
        sparse_matvec(A, p, Ap);

        /* alpha = (r^T * z) / (p^T * Ap) */
        double pAp = vec_dot(p, Ap, n);
        if (pAp == 0.0)
            break; /* breakdown */
        double alpha = rz / pAp;

        /* x_{k+1} = x_k + alpha * p_k */
        vec_axpy(alpha, p, x, n);

        /* r_{k+1} = r_k - alpha * Ap */
        vec_axpy(-alpha, Ap, r, n);

        rnorm = vec_norm2(r, n);

        /* Apply preconditioner: z_{k+1} = M^{-1}*r_{k+1} */
        if (precond) {
            sparse_err_t perr = precond(precond_ctx, n, r, z);
            if (perr != SPARSE_OK) {
                free(work);
                return perr;
            }
        } else {
            vec_copy(r, z, n);
        }

        /* beta = (r_{k+1}^T * z_{k+1}) / (r_k^T * z_k) */
        double rz_new = vec_dot(r, z, n);
        double beta = (rz != 0.0) ? rz_new / rz : 0.0;

        /* p_{k+1} = z_{k+1} + beta * p_k */
        for (idx_t i = 0; i < n; i++)
            p[i] = z[i] + beta * p[i];

        rz = rz_new;
    }

    /* Final convergence check if loop exhausted */
    if (!converged && rnorm / bnorm <= o->tol)
        converged = 1;

    if (result) {
        result->iterations = iter;
        result->residual_norm = rnorm / bnorm;
        result->converged = converged;
    }

    free(work);
    return converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Matrix-free CG
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_solve_cg_mf(sparse_matvec_fn matvec, const void *matvec_ctx, idx_t n,
                                const double *b, double *x, const sparse_iter_opts_t *opts,
                                sparse_precond_fn precond, const void *precond_ctx,
                                sparse_iter_result_t *result) {
    if (result) {
        result->iterations = 0;
        result->residual_norm = 0.0;
        result->converged = 0;
    }

    if (!matvec || !b || !x)
        return SPARSE_ERR_NULL;

    const sparse_iter_opts_t *o = opts ? opts : &cg_defaults;
    if (o->max_iter < 0 || o->tol < 0.0)
        return SPARSE_ERR_BADARG;
    if (n < 0)
        return SPARSE_ERR_BADARG;

    if (n == 0) {
        if (result)
            result->converged = 1;
        return SPARSE_OK;
    }

    double bnorm = vec_norm2(b, n);
    if (bnorm == 0.0) {
        vec_zero(x, n);
        if (result) {
            result->converged = 1;
            result->residual_norm = 0.0;
        }
        return SPARSE_OK;
    }

    if ((size_t)n > SIZE_MAX / (4 * sizeof(double)))
        return SPARSE_ERR_ALLOC;
    double *work = malloc(4 * (size_t)n * sizeof(double));
    if (!work)
        return SPARSE_ERR_ALLOC;
    double *r = work;
    double *z = work + n;
    double *p = work + 2 * n;
    double *Ap = work + 3 * n;

    /* r_0 = b - A*x_0 */
    sparse_err_t merr = matvec(matvec_ctx, n, x, Ap);
    if (merr != SPARSE_OK) {
        free(work);
        return merr;
    }
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - Ap[i];

    if (precond) {
        sparse_err_t perr = precond(precond_ctx, n, r, z);
        if (perr != SPARSE_OK) {
            free(work);
            return perr;
        }
    } else {
        vec_copy(r, z, n);
    }

    vec_copy(z, p, n);
    double rz = vec_dot(r, z, n);
    double rnorm = vec_norm2(r, n);

    idx_t iter = 0;
    int converged = 0;

    for (iter = 0; iter < o->max_iter; iter++) {
        if (rnorm / bnorm <= o->tol) {
            converged = 1;
            break;
        }

        merr = matvec(matvec_ctx, n, p, Ap);
        if (merr != SPARSE_OK) {
            free(work);
            return merr;
        }

        double pAp = vec_dot(p, Ap, n);
        if (pAp == 0.0)
            break;
        double alpha = rz / pAp;

        vec_axpy(alpha, p, x, n);
        vec_axpy(-alpha, Ap, r, n);
        rnorm = vec_norm2(r, n);

        if (precond) {
            sparse_err_t perr = precond(precond_ctx, n, r, z);
            if (perr != SPARSE_OK) {
                free(work);
                return perr;
            }
        } else {
            vec_copy(r, z, n);
        }

        double rz_new = vec_dot(r, z, n);
        double beta = (rz != 0.0) ? rz_new / rz : 0.0;

        for (idx_t i = 0; i < n; i++)
            p[i] = z[i] + beta * p[i];

        rz = rz_new;
    }

    if (!converged && rnorm / bnorm <= o->tol)
        converged = 1;

    if (result) {
        result->iterations = iter;
        result->residual_norm = rnorm / bnorm;
        result->converged = converged;
    }

    free(work);
    return converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
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
    if (!A || !b || !x) {
        if (result) {
            result->iterations = 0;
            result->residual_norm = 0.0;
            result->converged = 0;
        }
        return SPARSE_ERR_NULL;
    }
    if (sparse_rows(A) != sparse_cols(A)) {
        if (result) {
            result->iterations = 0;
            result->residual_norm = 0.0;
            result->converged = 0;
        }
        return SPARSE_ERR_SHAPE;
    }

    return sparse_solve_gmres_mf(gmres_sparse_matvec_adapter, A, sparse_rows(A), b, x, opts,
                                 precond, precond_ctx, result);
}

sparse_err_t sparse_solve_gmres_mf(sparse_matvec_fn matvec, const void *matvec_ctx, idx_t n,
                                   const double *b, double *x, const sparse_gmres_opts_t *opts,
                                   sparse_precond_fn precond, const void *precond_ctx,
                                   sparse_iter_result_t *result) {
    /* Initialize result to safe defaults before any early return */
    if (result) {
        result->iterations = 0;
        result->residual_norm = 0.0;
        result->converged = 0;
    }

    if (!matvec || !b || !x)
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

    /* Trivial case */
    if (n == 0) {
        if (result)
            result->converged = 1;
        return SPARSE_OK;
    }

    double bnorm = vec_norm2(b, n);
    if (bnorm == 0.0) {
        vec_zero(x, n);
        if (result) {
            result->converged = 1;
            result->residual_norm = 0.0;
        }
        return SPARSE_OK;
    }

    sparse_err_t merr;

    /* Fast path for max_iter==0: compute initial residual without
     * allocating the full Arnoldi workspace */
    if (o->max_iter == 0) {
        if ((size_t)n > SIZE_MAX / sizeof(double))
            return SPARSE_ERR_ALLOC;
        double *tmp = malloc((size_t)n * sizeof(double));
        if (!tmp)
            return SPARSE_ERR_ALLOC;
        merr = matvec(matvec_ctx, n, x, tmp);
        if (merr != SPARSE_OK) {
            free(tmp);
            return merr;
        }
        for (idx_t i = 0; i < n; i++)
            tmp[i] = b[i] - tmp[i];
        double rr = vec_norm2(tmp, n) / bnorm;
        free(tmp);
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
    {
        if ((size_t)n > SIZE_MAX / sizeof(double))
            return SPARSE_ERR_ALLOC;
        double *tmp = malloc((size_t)n * sizeof(double));
        if (!tmp)
            return SPARSE_ERR_ALLOC;
        merr = matvec(matvec_ctx, n, x, tmp);
        if (merr != SPARSE_OK) {
            free(tmp);
            return merr;
        }
        for (idx_t i = 0; i < n; i++)
            tmp[i] = b[i] - tmp[i];
        double rr = vec_norm2(tmp, n) / bnorm;
        free(tmp);
        if (rr <= o->tol) {
            if (result) {
                result->iterations = 0;
                result->residual_norm = rr;
                result->converged = 1;
            }
            return SPARSE_OK;
        }
    }

    /* Allocate workspace:
     *   v:  (m+1) * n        Arnoldi basis vectors
     *   h:  (m+1) * m        Upper Hessenberg matrix
     *   cs: m                Givens cosines
     *   sn: m                Givens sines
     *   g:  m+1              Residual vector in Hessenberg space
     *   y:  m                Solution of triangular system
     *   w:  n                Temporary vector for matvec / precond
     */
    size_t sz_v = (size_t)(m + 1) * (size_t)n;
    size_t sz_h = (size_t)(m + 1) * (size_t)m;
    size_t sz_cs = (size_t)m;
    size_t sz_sn = (size_t)m;
    size_t sz_g = (size_t)(m + 1);
    size_t sz_y = (size_t)m;
    size_t sz_w = (size_t)n;

    /* Overflow checks for workspace sizing */
    if (n > 0 && sz_v / (size_t)n != (size_t)(m + 1))
        return SPARSE_ERR_ALLOC;
    if (m > 0 && sz_h / (size_t)m != (size_t)(m + 1))
        return SPARSE_ERR_ALLOC;
    size_t total = 0;
    {
        size_t sizes[] = {sz_v, sz_h, sz_cs, sz_sn, sz_g, sz_y, sz_w};
        for (int s = 0; s < 7; s++) {
            if (sizes[s] > SIZE_MAX - total)
                return SPARSE_ERR_ALLOC;
            total += sizes[s];
        }
    }
    if (total > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;

    double *mem = calloc(total, sizeof(double));
    if (!mem)
        return SPARSE_ERR_ALLOC;

    double *v = mem;
    double *h = v + sz_v;
    double *cs = h + sz_h;
    double *sn = cs + sz_cs;
    double *g = sn + sz_sn;
    double *y = g + sz_g;
    double *w = y + sz_y;

    idx_t total_iter = 0;
    int converged = 0;
    double rel_res = 1.0;

    /* Outer restart loop — compute ceil(max_iter/m) in wider type to avoid
     * signed overflow when max_iter is near INT32_MAX */
    idx_t max_restarts = (idx_t)(((int64_t)o->max_iter + m - 1) / m);

    for (idx_t restart = 0; restart < max_restarts && !converged; restart++) {
        /* Compute r = b - A*x */
        merr = matvec(matvec_ctx, n, x, w);
        if (merr != SPARSE_OK) {
            free(mem);
            return merr;
        }
        for (idx_t i = 0; i < n; i++)
            V(0)[i] = b[i] - w[i];

        /* Left preconditioning: v_0 = M^{-1} * r (right precond: no change) */
        if (precond && !right_precond) {
            vec_copy(V(0), w, n);
            sparse_err_t perr = precond(precond_ctx, n, w, V(0));
            if (perr != SPARSE_OK) {
                free(mem);
                return perr;
            }
        }

        double beta = vec_norm2(V(0), n);
        if (beta == 0.0) {
            converged = 1;
            rel_res = 0.0;
            break;
        }

        /* Normalize v_0 */
        double inv_beta = 1.0 / beta;
        for (idx_t i = 0; i < n; i++)
            V(0)[i] *= inv_beta;

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
            total_iter++;

            if (right_precond) {
                /* Right preconditioning: w = A * M^{-1} * v_j */
                sparse_err_t perr = precond(precond_ctx, n, V(j), V(j + 1));
                if (perr != SPARSE_OK) {
                    free(mem);
                    return perr;
                }
                merr = matvec(matvec_ctx, n, V(j + 1), w);
                if (merr != SPARSE_OK) {
                    free(mem);
                    return merr;
                }
            } else {
                /* w = A * v_j */
                merr = matvec(matvec_ctx, n, V(j), w);
                if (merr != SPARSE_OK) {
                    free(mem);
                    return merr;
                }

                /* Left preconditioning: w = M^{-1} * A * v_j */
                if (precond) {
                    vec_copy(w, V(j + 1), n);
                    sparse_err_t perr = precond(precond_ctx, n, V(j + 1), w);
                    if (perr != SPARSE_OK) {
                        free(mem);
                        return perr;
                    }
                }
            }

            /* Arnoldi: modified Gram-Schmidt orthogonalization */
            for (idx_t i = 0; i <= j; i++) {
                H(i, j) = vec_dot(w, V(i), n);
                vec_axpy(-H(i, j), V(i), w, n);
            }
            H(j + 1, j) = vec_norm2(w, n);

            /* Check for lucky breakdown (before Givens rotation zeroes H(j+1,j)) */
            int lucky = (H(j + 1, j) < sparse_rel_tol(0, DROP_TOL));
            if (lucky) {
                vec_zero(V(j + 1), n);
            } else {
                double inv_h = 1.0 / H(j + 1, j);
                for (idx_t i = 0; i < n; i++)
                    V(j + 1)[i] = w[i] * inv_h;
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

            rel_res = fabs(g[j + 1]) / bnorm;

            if (o->verbose) {
                fprintf(stderr, "  GMRES iter %4d: %s||r||/||b|| = %.6e\n", (int)total_iter,
                        precond ? "precond " : "", rel_res);
            }

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
                free(mem);
                return perr;
            }
            vec_axpy(1.0, V(0), x, n);
        } else {
            /* Left precond / unpreconditioned: x = x + V * y */
            for (idx_t k = 0; k < j; k++)
                vec_axpy(y[k], V(k), x, n);
        }

        /* Compute true residual to decide convergence */
        merr = matvec(matvec_ctx, n, x, w);
        if (merr != SPARSE_OK) {
            free(mem);
            return merr;
        }
        for (idx_t i = 0; i < n; i++)
            w[i] = b[i] - w[i];
        rel_res = vec_norm2(w, n) / bnorm;

        if (rel_res <= o->tol) {
            converged = 1;
            break;
        }
    }

    /* converged is set only when the true residual meets tolerance */

    if (result) {
        result->iterations = total_iter;
        result->residual_norm = rel_res;
        result->converged = converged;
    }

    free(mem);
    return converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
}

#undef H
#undef V

/* ═══════════════════════════════════════════════════════════════════════
 * Block Conjugate Gradient (multiple RHS)
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_cg_solve_block(const SparseMatrix *A, const double *B, idx_t nrhs, double *X,
                                   const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                                   const void *precond_ctx, sparse_iter_result_t *result) {
    if (result) {
        result->iterations = 0;
        result->residual_norm = 0.0;
        result->converged = 0;
    }

    if (!A || !B || !X)
        return SPARSE_ERR_NULL;
    if (nrhs < 0)
        return SPARSE_ERR_BADARG;
    if (nrhs == 0) {
        if (result)
            result->converged = 1;
        return SPARSE_OK;
    }
    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;

    const sparse_iter_opts_t *o = opts ? opts : &cg_defaults;
    if (o->max_iter < 0 || o->tol < 0.0)
        return SPARSE_ERR_BADARG;

    idx_t n = sparse_rows(A);
    if (n == 0) {
        if (result)
            result->converged = 1;
        return SPARSE_OK;
    }

    /* Upfront overflow guards — must run before any n*k pointer arithmetic */
    if ((size_t)nrhs > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    if (n > 0 && (size_t)nrhs > SIZE_MAX / (size_t)n)
        return SPARSE_ERR_ALLOC;
    size_t blk = (size_t)n * (size_t)nrhs;
    if (blk > (size_t)INT32_MAX)
        return SPARSE_ERR_ALLOC;

    /* Compute ||B(:,k)|| for each column */
    double *bnorms = malloc((size_t)nrhs * sizeof(double));
    if (!bnorms)
        return SPARSE_ERR_ALLOC;
    for (idx_t k = 0; k < nrhs; k++) {
        size_t off = (size_t)n * (size_t)k;
        bnorms[k] = vec_norm2(&B[off], n);
        if (bnorms[k] == 0.0) {
            vec_zero(&X[off], n);
            bnorms[k] = 1.0; /* avoid div-by-zero; already converged */
        }
    }

    /* Allocate workspace: R, Z, P, AP — each n × nrhs */
    if (blk > SIZE_MAX / sizeof(double)) {
        free(bnorms);
        return SPARSE_ERR_ALLOC;
    }
    double *R = malloc(blk * sizeof(double));
    double *Z = malloc(blk * sizeof(double));
    double *P = malloc(blk * sizeof(double));
    double *AP = malloc(blk * sizeof(double));
    double *rz = malloc((size_t)nrhs * sizeof(double)); /* r^T*z per column */
    int *conv = calloc((size_t)nrhs, sizeof(int));      /* convergence flags */
    double *rnorms = malloc((size_t)nrhs * sizeof(double));
    if (!R || !Z || !P || !AP || !rz || !conv || !rnorms) {
        free(R);
        free(Z);
        free(P);
        free(AP);
        free(rz);
        free(conv);
        free(rnorms);
        free(bnorms);
        return SPARSE_ERR_ALLOC;
    }

    /* R = B - A*X (initial residual for all columns) */
    {
        sparse_err_t mv_err = sparse_matvec_block(A, X, nrhs, AP);
        if (mv_err != SPARSE_OK) {
            free(R);
            free(Z);
            free(P);
            free(AP);
            free(rz);
            free(conv);
            free(rnorms);
            free(bnorms);
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
                free(R);
                free(Z);
                free(P);
                free(AP);
                free(rz);
                free(conv);
                free(rnorms);
                free(bnorms);
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
                free(R);
                free(Z);
                free(P);
                free(AP);
                free(rz);
                free(conv);
                free(rnorms);
                free(bnorms);
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
                    free(R);
                    free(Z);
                    free(P);
                    free(AP);
                    free(rz);
                    free(conv);
                    free(rnorms);
                    free(bnorms);
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

    free(R);
    free(Z);
    free(P);
    free(AP);
    free(rz);
    free(conv);
    free(rnorms);
    free(bnorms);
    return all_converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Block GMRES (multiple RHS)
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_gmres_solve_block(const SparseMatrix *A, const double *B, idx_t nrhs, double *X,
                                      const sparse_gmres_opts_t *opts, sparse_precond_fn precond,
                                      const void *precond_ctx, sparse_iter_result_t *result) {
    if (result) {
        result->iterations = 0;
        result->residual_norm = 0.0;
        result->converged = 0;
    }

    if (!A || !B || !X)
        return SPARSE_ERR_NULL;
    if (nrhs < 0)
        return SPARSE_ERR_BADARG;
    if (nrhs == 0) {
        if (result)
            result->converged = 1;
        return SPARSE_OK;
    }
    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;

    idx_t n = sparse_rows(A);

    /* Overflow guard for per-column offset computation */
    if (n > 0 && (size_t)nrhs > SIZE_MAX / (size_t)n)
        return SPARSE_ERR_ALLOC;

    /* Solve each column independently using the existing GMRES implementation.
     * Per-column convergence tracking: each column gets its own result,
     * and we report the worst case. */
    idx_t max_iters = 0;
    double max_residual = 0.0;
    int all_converged = 1;
    sparse_err_t worst_err = SPARSE_OK;

    for (idx_t k = 0; k < nrhs; k++) {
        size_t off = (size_t)n * (size_t)k;
        sparse_iter_result_t col_result;
        sparse_err_t err =
            sparse_solve_gmres(A, &B[off], &X[off], opts, precond, precond_ctx, &col_result);

        if (col_result.iterations > max_iters)
            max_iters = col_result.iterations;
        if (col_result.residual_norm > max_residual)
            max_residual = col_result.residual_norm;
        if (!col_result.converged)
            all_converged = 0;

        /* Track worst error (but continue processing all columns).
         * Prefer hard errors over SPARSE_ERR_NOT_CONVERGED so real
         * failures are not masked by an earlier convergence failure. */
        if (err != SPARSE_OK && (worst_err == SPARSE_OK || (worst_err == SPARSE_ERR_NOT_CONVERGED &&
                                                            err != SPARSE_ERR_NOT_CONVERGED)))
            worst_err = err;
    }

    if (result) {
        result->iterations = max_iters;
        result->residual_norm = max_residual;
        result->converged = all_converged;
    }

    /* Return NOT_CONVERGED if any column failed, but not other errors
     * (those indicate real failures like NULL/SHAPE/ALLOC) */
    if (worst_err != SPARSE_OK && worst_err != SPARSE_ERR_NOT_CONVERGED)
        return worst_err;
    return all_converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MINRES — Minimum Residual method for symmetric systems
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_solve_minres(const SparseMatrix *A, const double *b, double *x,
                                 const sparse_iter_opts_t *opts, sparse_precond_fn precond,
                                 const void *precond_ctx, sparse_iter_result_t *result) {
    /* Initialize result to safe defaults */
    if (result) {
        result->iterations = 0;
        result->residual_norm = 0.0;
        result->converged = 0;
    }

    if (!A || !b || !x)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    const sparse_iter_opts_t *o = opts ? opts : &cg_defaults;
    if (o->max_iter < 0 || o->tol < 0.0)
        return SPARSE_ERR_BADARG;

    idx_t n = A->rows;

    if (n == 0) {
        if (result)
            result->converged = 1;
        return SPARSE_OK;
    }

    double bnorm = vec_norm2(b, n);
    if (bnorm == 0.0) {
        vec_zero(x, n);
        if (result) {
            result->converged = 1;
            result->residual_norm = 0.0;
        }
        return SPARSE_OK;
    }

    /* Workspace: v, v_old, w, d0, d1, d2 = 6 vectors
     * For preconditioned: +2 vectors (z, z_tmp) = 8 total */
    idx_t nvecs = precond ? 8 : 6;
    if ((size_t)n > SIZE_MAX / ((size_t)nvecs * sizeof(double)))
        return SPARSE_ERR_ALLOC;
    double *work = calloc((size_t)nvecs * (size_t)n, sizeof(double));
    if (!work)
        return SPARSE_ERR_ALLOC;

    double *v = work;                  /* current Lanczos vector */
    double *v_old = work + (size_t)n;  /* previous Lanczos vector */
    double *w = work + 2 * (size_t)n;  /* A*v workspace */
    double *d0 = work + 3 * (size_t)n; /* direction vector d_{k} */
    double *d1 = work + 4 * (size_t)n; /* direction vector d_{k-1} */
    double *d2 = work + 5 * (size_t)n; /* direction vector d_{k-2} */
    double *z = NULL, *z_tmp = NULL;
    if (precond) {
        z = work + 6 * (size_t)n;
        z_tmp = work + 7 * (size_t)n;
    }

    /* r0 = b - A*x0 (store in v temporarily) */
    sparse_matvec(A, x, w);
    for (idx_t i = 0; i < n; i++)
        v[i] = b[i] - w[i];

    /* Compute beta1 = ||r0|| or sqrt(r0^T * M^{-1} * r0) */
    double beta;
    if (precond) {
        sparse_err_t perr = precond(precond_ctx, n, v, z);
        if (perr != SPARSE_OK) {
            free(work);
            return perr;
        }
        beta = vec_dot(v, z, n);
        if (beta < 0.0) {
            free(work);
            return SPARSE_ERR_BADARG; /* M is not SPD */
        }
        beta = sqrt(beta);
    } else {
        beta = vec_norm2(v, n);
    }

    /* Check early convergence using the true Euclidean residual norm.
     * When preconditioned, beta = sqrt(r^T M^{-1} r) which differs from
     * ||r||_2, so always use vec_norm2(v, n) (v holds r0 before normalization). */
    {
        double r0norm = vec_norm2(v, n);
        if (r0norm / bnorm <= o->tol) {
            if (result) {
                result->converged = 1;
                result->residual_norm = r0norm / bnorm;
            }
            free(work);
            return SPARSE_OK;
        }
    }

    /* Normalize: v = r0/beta, z = M^{-1}r0/beta */
    {
        double inv_beta = 1.0 / beta;
        for (idx_t i = 0; i < n; i++)
            v[i] *= inv_beta;
        if (precond) {
            for (idx_t i = 0; i < n; i++)
                z[i] *= inv_beta;
        }
    }

    /* Givens rotation state:
     * cs, sn   = G_{k-1} (initialized as identity)
     * cs_old, sn_old = G_{k-2} (initialized as identity) */
    double cs = 1.0, sn = 0.0;
    double cs_old = 1.0, sn_old = 0.0;

    double phi_bar = beta; /* residual norm estimate */
    double beta_old = 0.0;

    idx_t iter = 0;
    int converged = 0;

    for (iter = 1; iter <= o->max_iter; iter++) {
        /* ── Lanczos step ──────────────────────────────────────────── */
        /* w = A * v_k (unpreconditioned) or A * z_k (preconditioned) */
        if (precond)
            sparse_matvec(A, z, w);
        else
            sparse_matvec(A, v, w);

        /* alpha = <v_k, w> (or <z_k, w> for preconditioned) */
        double alpha;
        if (precond)
            alpha = vec_dot(z, w, n);
        else
            alpha = vec_dot(v, w, n);

        /* w = w - alpha*v - beta_old*v_old (three-term recurrence) */
        for (idx_t i = 0; i < n; i++)
            w[i] = w[i] - alpha * v[i] - beta_old * v_old[i];

        /* Compute beta_new */
        double beta_new;
        if (precond) {
            sparse_err_t perr = precond(precond_ctx, n, w, z_tmp);
            if (perr != SPARSE_OK) {
                free(work);
                return perr;
            }
            double inner = vec_dot(w, z_tmp, n);
            if (inner < 0.0) {
                free(work);
                return SPARSE_ERR_BADARG; /* M not SPD */
            }
            beta_new = sqrt(inner);
        } else {
            beta_new = vec_norm2(w, n);
        }

        /* ── QR update: process column k of the tridiagonal ──────── */
        /* The column has: beta_old at row k-1, alpha at row k,
         * beta_new at row k+1. Apply previous Givens rotations. */

        /* Step 1: Apply G_{k-2} to (row k-2, row k-1) = (0, beta_old) */
        double eps = sn_old * beta_old;
        double delta_bar = cs_old * beta_old;

        /* Step 2: Apply G_{k-1} to (row k-1, row k) = (delta_bar, alpha) */
        double delta = cs * delta_bar + sn * alpha;
        double gamma_bar = -sn * delta_bar + cs * alpha;

        /* Step 3: Compute G_k to zero out beta_new at row k+1 */
        double gamma = sqrt(gamma_bar * gamma_bar + beta_new * beta_new);

        if (gamma == 0.0)
            break; /* breakdown */

        double cs_new = gamma_bar / gamma;
        double sn_new = beta_new / gamma;

        /* Step 4: Update RHS */
        double phi = cs_new * phi_bar;
        phi_bar = -sn_new * phi_bar;

        /* Step 5: Direction vector d_k = (v_k - eps*d_{k-2} - delta*d_{k-1}) / gamma
         * For preconditioned: use z instead of v */
        {
            const double *dv = precond ? z : v;
            double inv_gamma = 1.0 / gamma;
            for (idx_t i = 0; i < n; i++)
                d0[i] = (dv[i] - eps * d2[i] - delta * d1[i]) * inv_gamma;
        }

        /* Step 6: Update solution x = x + phi * d_k */
        for (idx_t i = 0; i < n; i++)
            x[i] += phi * d0[i];

        /* Step 7: Check convergence */
        double relres = fabs(phi_bar) / bnorm;

        if (o->verbose)
            fprintf(stderr, "  MINRES iter %d: relres = %.6e\n", (int)iter, relres);

        if (relres <= o->tol) {
            /* QR estimate says converged — verify with true residual before
             * breaking, since the QR estimate can underestimate in finite
             * precision (especially with preconditioning). */
            sparse_matvec(A, x, w);
            double tr = 0.0;
            for (idx_t i = 0; i < n; i++) {
                double di = w[i] - b[i];
                tr += di * di;
            }
            if (sqrt(tr) / bnorm <= o->tol)
                break;
        }

        /* ── Prepare for next iteration ──────────────────────────── */
        /* Shift Givens rotations */
        cs_old = cs;
        sn_old = sn;
        cs = cs_new;
        sn = sn_new;

        /* Shift direction vectors: d2 ← d1, d1 ← d0 (swap pointers) */
        {
            double *tmp = d2;
            d2 = d1;
            d1 = d0;
            d0 = tmp;
        }

        /* Shift Lanczos vectors and normalize */
        if (beta_new > 0.0) {
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
            /* Lanczos breakdown: Krylov subspace exhausted — solution is exact */
            break;
        }

        beta_old = beta_new;
    }

    /* Compute true residual ||b - A*x|| / ||b|| */
    sparse_matvec(A, x, w);
    double true_res = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double di = w[i] - b[i];
        true_res += di * di;
    }
    true_res = sqrt(true_res) / bnorm;

    /* Final convergence decision based on true residual, not QR estimate */
    converged = (true_res <= o->tol);

    if (result) {
        result->iterations = iter > o->max_iter ? o->max_iter : iter;
        result->residual_norm = true_res;
        result->converged = converged;
    }

    free(work);
    return converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
}

sparse_err_t sparse_minres_solve_block(const SparseMatrix *A, const double *B, idx_t nrhs,
                                       double *X, const sparse_iter_opts_t *opts,
                                       sparse_precond_fn precond, const void *precond_ctx,
                                       sparse_iter_result_t *result) {
    if (result) {
        result->iterations = 0;
        result->residual_norm = 0.0;
        result->converged = 0;
    }

    if (!A || !B || !X)
        return SPARSE_ERR_NULL;
    if (nrhs < 0)
        return SPARSE_ERR_BADARG;
    if (nrhs == 0) {
        if (result)
            result->converged = 1;
        return SPARSE_OK;
    }
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    idx_t n = A->rows;

    /* Overflow check for j*n pointer offsets (guard before computing size_t products) */
    if (n > 0 && (size_t)nrhs > SIZE_MAX / (size_t)n)
        return SPARSE_ERR_ALLOC;

    if (n == 0) {
        if (result)
            result->converged = 1;
        return SPARSE_OK;
    }

    /* Run MINRES independently for each column */
    idx_t max_iters = 0;
    double max_residual = 0.0;
    int all_converged = 1;
    sparse_err_t worst_err = SPARSE_OK;

    for (idx_t j = 0; j < nrhs; j++) {
        const double *bj = B + (size_t)j * (size_t)n;
        double *xj = X + (size_t)j * (size_t)n;
        sparse_iter_result_t col_result = {0, 0.0, 0};

        sparse_err_t err = sparse_solve_minres(A, bj, xj, opts, precond, precond_ctx, &col_result);

        if (col_result.iterations > max_iters)
            max_iters = col_result.iterations;
        if (col_result.residual_norm > max_residual)
            max_residual = col_result.residual_norm;
        if (!col_result.converged)
            all_converged = 0;
        if (err != SPARSE_OK && err != SPARSE_ERR_NOT_CONVERGED)
            worst_err = err;
    }

    if (result) {
        result->iterations = max_iters;
        result->residual_norm = max_residual;
        result->converged = all_converged;
    }

    if (worst_err != SPARSE_OK && worst_err != SPARSE_ERR_NOT_CONVERGED)
        return worst_err;
    return all_converged ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
}
