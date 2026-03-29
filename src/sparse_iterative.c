#include "sparse_iterative.h"
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

sparse_err_t sparse_solve_gmres(const SparseMatrix *A, const double *b, double *x,
                                const sparse_gmres_opts_t *opts, sparse_precond_fn precond,
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

    const sparse_gmres_opts_t *o = opts ? opts : &gmres_defaults;
    if (o->max_iter < 0 || o->restart <= 0 || o->tol < 0.0)
        return SPARSE_ERR_BADARG;
    if (o->precond_side != SPARSE_PRECOND_LEFT && o->precond_side != SPARSE_PRECOND_RIGHT)
        return SPARSE_ERR_BADARG;
    idx_t n = sparse_rows(A);
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

    /* Fast path for max_iter==0: compute initial residual without
     * allocating the full Arnoldi workspace */
    if (o->max_iter == 0) {
        double *tmp = malloc((size_t)n * sizeof(double));
        if (!tmp)
            return SPARSE_ERR_ALLOC;
        sparse_matvec(A, x, tmp);
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
        double *tmp = malloc((size_t)n * sizeof(double));
        if (!tmp)
            return SPARSE_ERR_ALLOC;
        sparse_matvec(A, x, tmp);
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
        sparse_matvec(A, x, w);
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
                sparse_matvec(A, V(j + 1), w);
            } else {
                /* w = A * v_j */
                sparse_matvec(A, V(j), w);

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
            int lucky = (H(j + 1, j) < 1e-30);
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
            if (fabs(H(i, i)) > 1e-30)
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
        sparse_matvec(A, x, w);
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
