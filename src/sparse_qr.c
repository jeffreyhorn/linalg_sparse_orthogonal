#include "sparse_qr.h"
#include "sparse_matrix_internal.h"
#include "sparse_reorder.h"
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
 * QR factorization — factorization, solve, rank, and nullspace routines
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
    /* Zero the output struct. Callers must call sparse_qr_free() before
     * reusing a struct that already holds a factorization. */
    memset(qr, 0, sizeof(*qr));
    if (!A)
        return SPARSE_ERR_NULL;

    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);
    idx_t k = (m < n) ? m : n; /* min(m, n) */

    qr->m = m;
    qr->n = n;

    if (m == 0 || n == 0)
        return SPARSE_OK;

    /* Compute optional column reordering (AMD on A^T*A pattern) */
    idx_t *col_reorder = NULL;
    if (opts && opts->reorder != SPARSE_REORDER_NONE && n > 1) {
        /* Build A^T*A pattern: an n×n matrix where (i,j) is nonzero if
         * columns i and j of A share at least one row. */
        SparseMatrix *AtA = sparse_create(n, n);
        if (AtA) {
            for (idx_t row = 0; row < m; row++) {
                /* Collect column indices in this row */
                Node *nd1 = A->row_headers[row];
                while (nd1) {
                    Node *nd2 = nd1->right;
                    while (nd2) {
                        sparse_insert(AtA, nd1->col, nd2->col, 1.0);
                        sparse_insert(AtA, nd2->col, nd1->col, 1.0);
                        nd2 = nd2->right;
                    }
                    /* Diagonal */
                    sparse_insert(AtA, nd1->col, nd1->col, 1.0);
                    nd1 = nd1->right;
                }
            }

            col_reorder = malloc((size_t)n * sizeof(idx_t));
            if (col_reorder) {
                sparse_err_t rerr = SPARSE_ERR_BADARG;
                if (opts->reorder == SPARSE_REORDER_AMD)
                    rerr = sparse_reorder_amd(AtA, col_reorder);
                else if (opts->reorder == SPARSE_REORDER_RCM)
                    rerr = sparse_reorder_rcm(AtA, col_reorder);
                if (rerr != SPARSE_OK) {
                    free(col_reorder);
                    col_reorder = NULL;
                }
            }
            sparse_free(AtA);
        }
    }

    /* Overflow check for dense workspace sizing */
    if (n > 0 && (size_t)m > SIZE_MAX / ((size_t)n * sizeof(double))) {
        free(col_reorder);
        return SPARSE_ERR_ALLOC;
    }

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
        free(col_reorder);
        return SPARSE_ERR_ALLOC;
    }

    /* Build inverse permutation for O(1) lookup if reordering is present.
     * col_reorder[new] = old, so inv_col_reorder[old] = new. */
    idx_t *inv_col_reorder = NULL;
    if (col_reorder) {
        inv_col_reorder = malloc((size_t)n * sizeof(idx_t));
        if (!inv_col_reorder) {
            free(W);
            free(col_norms);
            free(perm);
            free(betas);
            free(vecs);
            free(col_reorder);
            return SPARSE_ERR_ALLOC;
        }
        for (idx_t j = 0; j < n; j++)
            inv_col_reorder[col_reorder[j]] = j;
    }

    /* Copy A into dense column-major W, applying column reorder if present */
    for (idx_t i = 0; i < m; i++) {
        Node *nd = A->row_headers[i];
        while (nd) {
            idx_t dest_col = inv_col_reorder ? inv_col_reorder[nd->col] : nd->col;
            W[(size_t)dest_col * (size_t)m + (size_t)i] = nd->value;
            nd = nd->right;
        }
    }
    free(inv_col_reorder);

    /* Initialize column permutation: track original column indices.
     * If col_reorder is present, perm[j] starts as col_reorder[j] (original col). */
    for (idx_t j = 0; j < n; j++) {
        perm[j] = col_reorder ? col_reorder[j] : j;
        double s = 0.0;
        for (idx_t i = 0; i < m; i++) {
            double v = W[(size_t)j * (size_t)m + (size_t)i];
            s += v * v;
        }
        col_norms[j] = s; /* squared norm */
    }
    free(col_reorder); /* consumed into perm */

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

    idx_t rank = 0;       /* numerical rank (columns with |R(k,k)| above tolerance) */
    idx_t steps_done = 0; /* total reflectors applied (may exceed rank by 1) */
    double r00 = 0.0;     /* |R(0,0)| for rank tolerance */

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

        /* Apply Householder to remaining columns step+1..n-1.
         * This must happen before the rank-deficiency check so that
         * stored reflectors and the workspace are consistent even
         * when we break early. */
        for (idx_t j = step + 1; j < n; j++) {
            double *cj = &W[(size_t)j * (size_t)m + (size_t)step];
            householder_apply(hv, beta, cj, col_len);
        }

        /* Check R diagonal: if tiny after reflection, this column is numerically
         * rank-deficient. Count the step (reflector was applied to all columns)
         * but do not increment rank. */
        steps_done++;
        if (fabs(col_ptr[0]) < rank_tol) {
            break;
        }

        rank++;

        /* Update column norms (downdate: remove the squared entry at row step) */
        for (idx_t j = step + 1; j < n; j++) {
            double entry = W[(size_t)j * (size_t)m + (size_t)step];
            col_norms[j] -= entry * entry;
            if (col_norms[j] < 0.0)
                col_norms[j] = 0.0;
        }
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

    /* Extract R rows for all applied reflectors (steps_done), not just
     * rank — this keeps A*P = Q*R consistent even when the last step
     * detected rank deficiency (its reflector was applied but R(k,k) ≈ 0). */
    for (idx_t i = 0; i < steps_done; i++) {
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
        /* Q*x = H_0 * H_1 * ... * H_{k-1} * x (apply reflectors right-to-left) */
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

    /* Overflow-safe computation of m*m*sizeof(double) */
    size_t m_sz = (size_t)m;
    size_t mm = 0;
    size_t bytes = 0;
    if (__builtin_mul_overflow(m_sz, m_sz, &mm))
        return SPARSE_ERR_ALLOC;
    if (__builtin_mul_overflow(mm, sizeof(double), &bytes))
        return SPARSE_ERR_ALLOC;

    /* Start with identity */
    memset(Q, 0, bytes);
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

/* ═══════════════════════════════════════════════════════════════════════
 * Rank estimation and null-space extraction
 * ═══════════════════════════════════════════════════════════════════════ */

idx_t sparse_qr_rank(const sparse_qr_t *qr, double tol) {
    if (!qr || !qr->R)
        return 0;

    idx_t k = (qr->m < qr->n) ? qr->m : qr->n;
    if (k == 0)
        return 0;

    double r00 = fabs(sparse_get_phys(qr->R, 0, 0));
    if (r00 == 0.0)
        return 0;

    /* Default tolerance: eps * max(m,n) * |R(0,0)| */
    if (tol <= 0.0) {
        double eps = 2.2204460492503131e-16; /* DBL_EPSILON */
        idx_t mn = (qr->m > qr->n) ? qr->m : qr->n;
        tol = eps * (double)mn * r00;
    } else {
        tol = tol * r00;
    }

    idx_t rank = 0;
    for (idx_t i = 0; i < k; i++) {
        if (fabs(sparse_get_phys(qr->R, i, i)) > tol)
            rank++;
        else
            break;
    }
    return rank;
}

sparse_err_t sparse_qr_nullspace(const sparse_qr_t *qr, double tol, double *basis,
                                 idx_t *null_dim) {
    if (!qr || !null_dim)
        return SPARSE_ERR_NULL;

    idx_t n = qr->n;
    idx_t rank = sparse_qr_rank(qr, tol);
    idx_t ndim = n - rank;
    *null_dim = ndim;

    if (ndim == 0 || !basis)
        return SPARSE_OK;

    idx_t m = qr->m;

    /* For each null-space vector j (j = rank..n-1 in permuted space):
     * Solve R[0:rank, 0:rank] * z[0:rank] = -R[0:rank, j]
     * Then the null-space vector in permuted space is [z; e_j]
     * Unpermute to get the vector in original column space.
     */
    for (idx_t j_idx = 0; j_idx < ndim; j_idx++) {
        idx_t j = rank + j_idx;                         /* column index in permuted R */
        double *nv = &basis[(size_t)j_idx * (size_t)n]; /* j_idx-th null vector, length n */

        /* Extract -R[0:rank, j] */
        double *rhs = calloc((size_t)n, sizeof(double));
        if (!rhs)
            return SPARSE_ERR_ALLOC;
        for (idx_t i = 0; i < rank; i++)
            rhs[i] = -sparse_get_phys(qr->R, i, j);

        /* Back-substitute: R[0:rank, 0:rank] * z = rhs */
        for (idx_t i = rank - 1; i >= 0; i--) {
            double sum = 0.0;
            double diag = 0.0;
            Node *nd = qr->R->row_headers[i];
            while (nd) {
                if (nd->col == i)
                    diag = nd->value;
                else if (nd->col > i && nd->col < rank)
                    sum += nd->value * rhs[nd->col];
                nd = nd->right;
            }
            if (fabs(diag) > 1e-30)
                rhs[i] = (rhs[i] - sum) / diag;
            else
                rhs[i] = 0.0;
        }

        /* Form null vector in permuted space: [z_0..z_{rank-1}, 0, ..., 1_j, ..., 0] */
        double *perm_vec = calloc((size_t)n, sizeof(double));
        if (!perm_vec) {
            free(rhs);
            return SPARSE_ERR_ALLOC;
        }
        for (idx_t i = 0; i < rank; i++)
            perm_vec[i] = rhs[i];
        perm_vec[j] = 1.0;

        /* Unpermute: nv[col_perm[i]] = perm_vec[i] */
        for (idx_t i = 0; i < n; i++)
            nv[qr->col_perm[i]] = perm_vec[i];

        free(rhs);
        free(perm_vec);
    }

    (void)m;
    return SPARSE_OK;
}
