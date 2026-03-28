#include "sparse_qr.h"
#include "sparse_matrix_internal.h"
#include "sparse_vector.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Householder reflection helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Compute Householder vector v and scalar beta such that
 * (I - beta*v*v^T)*x = ||x||*e_1.
 *
 * On entry: x[0..len-1] is the input vector.
 * On exit:  v[0..len-1] is the Householder vector (v[0] = 1 implicitly
 *           for the standard form, but we store the full vector).
 *           beta is the scalar such that H = I - beta*v*v^T.
 *
 * Returns beta. If x is zero, returns beta=0 (no reflection needed).
 */
static double householder_compute(const double *x, double *v, idx_t len) {
    if (len <= 0)
        return 0.0;

    /* Copy x into v */
    memcpy(v, x, (size_t)len * sizeof(double));

    double sigma = 0.0;
    for (idx_t i = 1; i < len; i++)
        sigma += v[i] * v[i];

    if (sigma == 0.0 && v[0] >= 0.0) {
        /* x is already a non-negative multiple of e_1 — no reflection needed */
        return 0.0;
    }

    double xnorm = sqrt(v[0] * v[0] + sigma);

    /* v[0] += sign(x[0]) * ||x|| */
    if (v[0] >= 0.0)
        v[0] += xnorm;
    else
        v[0] -= xnorm;

    /* beta = 2 / (v^T * v) */
    double vtv = v[0] * v[0] + sigma;
    if (vtv == 0.0)
        return 0.0;

    return 2.0 / vtv;
}

/**
 * Apply Householder reflection (I - beta*v*v^T) to vector y in-place.
 * y = y - beta * v * (v^T * y)
 *
 * v has length len, y has length len.
 */
static void householder_apply(const double *v, double beta, double *y, idx_t len) {
    if (beta == 0.0)
        return;

    /* Compute v^T * y */
    double vty = 0.0;
    for (idx_t i = 0; i < len; i++)
        vty += v[i] * y[i];

    /* y = y - beta * (v^T * y) * v */
    double scale = beta * vty;
    for (idx_t i = 0; i < len; i++)
        y[i] -= scale * v[i];
}

/* ═══════════════════════════════════════════════════════════════════════
 * QR factorization — stub implementations (Days 5-6 will complete)
 * ═══════════════════════════════════════════════════════════════════════ */

void sparse_qr_free(sparse_qr_t *qr) {
    if (!qr)
        return;
    sparse_free(qr->R);
    qr->R = NULL;
    if (qr->v_vectors) {
        idx_t k = (qr->m < qr->n) ? qr->m : qr->n;
        for (idx_t i = 0; i < k; i++)
            free(qr->v_vectors[i]);
        free(qr->v_vectors);
        qr->v_vectors = NULL;
    }
    free(qr->betas);
    qr->betas = NULL;
    free(qr->col_perm);
    qr->col_perm = NULL;
    qr->m = 0;
    qr->n = 0;
    qr->rank = 0;
}

sparse_err_t sparse_qr_factor(const SparseMatrix *A, sparse_qr_t *qr) {
    return sparse_qr_factor_opts(A, NULL, qr);
}

sparse_err_t sparse_qr_factor_opts(const SparseMatrix *A, const sparse_qr_opts_t *opts,
                                   sparse_qr_t *qr) {
    if (!qr)
        return SPARSE_ERR_NULL;
    memset(qr, 0, sizeof(*qr));
    if (!A)
        return SPARSE_ERR_NULL;

    (void)opts; /* reordering deferred to Day 11 */

    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);
    idx_t k = (m < n) ? m : n; /* min(m, n) */

    qr->m = m;
    qr->n = n;

    if (m == 0 || n == 0)
        return SPARSE_OK;

    /* Allocate dense m×n working matrix (column-major) */
    double *W = calloc((size_t)m * (size_t)n, sizeof(double));
    double *col_norms = malloc((size_t)n * sizeof(double));
    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    double *betas = calloc((size_t)k, sizeof(double));
    double **vecs = calloc((size_t)k, sizeof(double *));

    if (!W || !col_norms || !perm || !betas || !vecs) {
        free(W);
        free(col_norms);
        free(perm);
        free(betas);
        free(vecs);
        return SPARSE_ERR_ALLOC;
    }

    /* Copy A into dense column-major W */
    for (idx_t i = 0; i < m; i++) {
        Node *nd = A->row_headers[i];
        while (nd) {
            W[(size_t)nd->col * (size_t)m + (size_t)i] = nd->value;
            nd = nd->right;
        }
    }

    /* Initialize column permutation and norms */
    for (idx_t j = 0; j < n; j++) {
        perm[j] = j;
        double s = 0.0;
        for (idx_t i = 0; i < m; i++) {
            double v = W[(size_t)j * (size_t)m + (size_t)i];
            s += v * v;
        }
        col_norms[j] = s; /* squared norm */
    }

    /* Dense Householder vector workspace */
    double *hv = malloc((size_t)m * sizeof(double));
    if (!hv) {
        free(W);
        free(col_norms);
        free(perm);
        free(betas);
        free(vecs);
        return SPARSE_ERR_ALLOC;
    }

    idx_t rank = 0;
    double r00 = 0.0; /* |R(0,0)| for rank tolerance */

    for (idx_t step = 0; step < k; step++) {
        /* Column pivoting: find column with largest remaining norm */
        idx_t best = step;
        double best_norm = col_norms[step];
        for (idx_t j = step + 1; j < n; j++) {
            if (col_norms[j] > best_norm) {
                best_norm = col_norms[j];
                best = j;
            }
        }

        /* Swap columns step and best */
        if (best != step) {
            /* Swap in W */
            for (idx_t i = 0; i < m; i++) {
                double tmp = W[(size_t)step * (size_t)m + (size_t)i];
                W[(size_t)step * (size_t)m + (size_t)i] = W[(size_t)best * (size_t)m + (size_t)i];
                W[(size_t)best * (size_t)m + (size_t)i] = tmp;
            }
            /* Swap norms */
            double tmp_n = col_norms[step];
            col_norms[step] = col_norms[best];
            col_norms[best] = tmp_n;
            /* Swap perm */
            idx_t tmp_p = perm[step];
            perm[step] = perm[best];
            perm[best] = tmp_p;
        }

        /* Check for rank deficiency: if remaining norm is tiny, stop */
        idx_t col_len = m - step;
        double *col_ptr = &W[(size_t)step * (size_t)m + (size_t)step];

        if (step == 0) {
            r00 = sqrt(best_norm);
        }
        double rank_tol = 1e-14 * (r00 > 0.0 ? r00 : 1.0) * (double)(m > n ? m : n);
        if (sqrt(best_norm) < rank_tol) {
            break; /* remaining columns are numerically zero */
        }

        /* Compute Householder vector for column step (below diagonal) */
        double beta = householder_compute(col_ptr, hv, col_len);
        betas[step] = beta;

        /* Store Householder vector */
        vecs[step] = malloc((size_t)col_len * sizeof(double));
        if (!vecs[step]) {
            free(hv);
            free(W);
            free(col_norms);
            qr->betas = betas;
            qr->v_vectors = vecs;
            qr->col_perm = perm;
            qr->rank = rank;
            sparse_qr_free(qr);
            return SPARSE_ERR_ALLOC;
        }
        memcpy(vecs[step], hv, (size_t)col_len * sizeof(double));

        /* Apply Householder to column step (produces R diagonal and zeros below) */
        householder_apply(hv, beta, col_ptr, col_len);

        /* Check R diagonal: if tiny after reflection, this column is rank-deficient */
        if (fabs(col_ptr[0]) < rank_tol) {
            break;
        }

        /* Apply Householder to remaining columns step+1..n-1 */
        for (idx_t j = step + 1; j < n; j++) {
            double *cj = &W[(size_t)j * (size_t)m + (size_t)step];
            householder_apply(hv, beta, cj, col_len);
        }

        /* Update column norms (downdate: remove the squared entry at row step) */
        for (idx_t j = step + 1; j < n; j++) {
            double entry = W[(size_t)j * (size_t)m + (size_t)step];
            col_norms[j] -= entry * entry;
            if (col_norms[j] < 0.0)
                col_norms[j] = 0.0;
        }

        rank++;
    }

    free(hv);
    free(col_norms);

    /* Extract R as sparse upper triangular */
    SparseMatrix *R = sparse_create(k, n);
    if (!R) {
        free(W);
        free(perm);
        free(betas);
        for (idx_t i = 0; i < k; i++)
            free(vecs[i]);
        free(vecs);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < rank; i++) {
        for (idx_t j = i; j < n; j++) {
            double val = W[(size_t)j * (size_t)m + (size_t)i];
            if (fabs(val) > 1e-15)
                sparse_insert(R, i, j, val);
        }
    }

    free(W);

    qr->R = R;
    qr->betas = betas;
    qr->v_vectors = vecs;
    qr->col_perm = perm;
    qr->rank = rank;

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Q application
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_qr_apply_q(const sparse_qr_t *qr, int transpose, const double *x, double *y) {
    if (!qr || !x || !y)
        return SPARSE_ERR_NULL;

    idx_t m = qr->m;
    idx_t k = (qr->m < qr->n) ? qr->m : qr->n;

    /* Copy x to y if not aliased */
    if (x != y)
        memcpy(y, x, (size_t)m * sizeof(double));

    if (!qr->v_vectors || !qr->betas)
        return SPARSE_OK;

    if (!transpose) {
        /* Q*x = H_0 * H_1 * ... * H_{k-1} * x (apply in forward order) */
        for (idx_t i = k - 1; i >= 0; i--) {
            if (qr->betas[i] == 0.0)
                continue;
            idx_t len = m - i;
            householder_apply(qr->v_vectors[i], qr->betas[i], &y[i], len);
        }
    } else {
        /* Q^T*x = H_{k-1} * ... * H_1 * H_0 * x (apply in reverse order) */
        for (idx_t i = 0; i < k; i++) {
            if (qr->betas[i] == 0.0)
                continue;
            idx_t len = m - i;
            householder_apply(qr->v_vectors[i], qr->betas[i], &y[i], len);
        }
    }

    return SPARSE_OK;
}

sparse_err_t sparse_qr_form_q(const sparse_qr_t *qr, double *Q) {
    if (!qr || !Q)
        return SPARSE_ERR_NULL;

    idx_t m = qr->m;

    /* Start with identity */
    memset(Q, 0, (size_t)m * (size_t)m * sizeof(double));
    for (idx_t i = 0; i < m; i++)
        Q[(size_t)i * (size_t)m + (size_t)i] = 1.0;

    /* Apply Q to each column of I */
    for (idx_t j = 0; j < m; j++) {
        sparse_qr_apply_q(qr, 0, &Q[(size_t)j * (size_t)m], &Q[(size_t)j * (size_t)m]);
    }

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Least-squares solver: min ||Ax - b||_2
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_qr_solve(const sparse_qr_t *qr, const double *b, double *x, double *residual) {
    if (!qr || !b || !x)
        return SPARSE_ERR_NULL;
    if (!qr->R || !qr->col_perm)
        return SPARSE_ERR_NULL;

    idx_t m = qr->m;
    idx_t n = qr->n;
    idx_t rank = qr->rank;

    /* Allocate workspace: c = Q^T * b (length m) */
    double *c = malloc((size_t)m * sizeof(double));
    if (!c)
        return SPARSE_ERR_ALLOC;

    /* c = Q^T * b */
    sparse_qr_apply_q(qr, 1, b, c);

    /* Compute residual norm: ||c[rank:]||_2 */
    if (residual) {
        double rnorm = 0.0;
        for (idx_t i = rank; i < m; i++)
            rnorm += c[i] * c[i];
        *residual = sqrt(rnorm);
    }

    /* Back-substitute: R[0:rank, 0:rank] * x_p = c[0:rank] */
    double *x_p = calloc((size_t)n, sizeof(double));
    if (!x_p) {
        free(c);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = rank - 1; i >= 0; i--) {
        double sum = 0.0;
        /* Walk row i of R for entries j > i */
        Node *nd = qr->R->row_headers[i];
        double diag = 0.0;
        while (nd) {
            if (nd->col == i) {
                diag = nd->value;
            } else if (nd->col > i && nd->col < n) {
                // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
                sum += nd->value * x_p[nd->col];
            }
            nd = nd->right;
        }
        if (fabs(diag) < 1e-30) {
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            x_p[i] = 0.0; /* rank-deficient: set to zero */
        } else {
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            x_p[i] = (c[i] - sum) / diag;
        }
    }

    /* Apply column permutation: x[col_perm[i]] = x_p[i] */
    for (idx_t i = 0; i < n; i++)
        x[qr->col_perm[i]] = x_p[i];

    free(c);
    free(x_p);
    return SPARSE_OK;
}

idx_t sparse_qr_rank(const sparse_qr_t *qr, double tol) {
    (void)qr;
    (void)tol;
    return 0; /* stub — implemented in Day 10 */
}

// NOLINTNEXTLINE(readability-non-const-parameter)
sparse_err_t sparse_qr_nullspace(const sparse_qr_t *qr, double tol,
                                 double *basis,     // NOLINT(readability-non-const-parameter)
                                 idx_t *null_dim) { // NOLINT(readability-non-const-parameter)
    (void)qr;
    (void)tol;
    (void)basis;
    (void)null_dim;
    return SPARSE_ERR_BADARG; /* stub — implemented in Day 10 */
}
