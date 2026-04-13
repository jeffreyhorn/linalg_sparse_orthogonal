#include "sparse_qr.h"
#include "sparse_matrix_internal.h"
#include "sparse_reorder.h"
#include "sparse_vector.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Portable overflow-safe multiplication: returns 0 on success, 1 on overflow */
static int size_mul_overflow(size_t a, size_t b, size_t *result) {
    if (a != 0 && b > SIZE_MAX / a)
        return 1;
    *result = a * b;
    return 0;
}

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
 * Sparse column utilities for column-by-column QR
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Extract column `col` of sparse matrix A into a dense vector of length m.
 * The dense vector must be pre-zeroed by the caller.
 * Uses column headers for efficient traversal.
 */
static void sparse_extract_column(const SparseMatrix *A, idx_t col, double *dense) {
    Node *nd = A->col_headers[col];
    while (nd) {
        dense[nd->row] = nd->value;
        nd = nd->down;
    }
}

/**
 * Apply Householder reflection (I - beta*v*v^T) to a dense column vector
 * starting at index `start`. v has length (m - start), dense has length m.
 * Only dense[start..m-1] is modified.
 */
static void householder_apply_to_column(const double *v, double beta, double *dense, idx_t start,
                                        idx_t m) {
    if (beta == 0.0)
        return;
    idx_t len = m - start;
    householder_apply(v, beta, &dense[start], len);
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
    qr->economy = 0;
}

/**
 * Column-by-column QR factorization (sparse_mode=1).
 * Uses O(m) working memory per column instead of O(m*n) dense workspace.
 * Operates on a mutable copy of A, applying Householder reflectors
 * column-by-column and extracting R entries as they are produced.
 */
static sparse_err_t sparse_qr_factor_colwise(const SparseMatrix *A, const sparse_qr_opts_t *opts,
                                             sparse_qr_t *qr) {
    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);
    idx_t k = (m < n) ? m : n;

    qr->m = m;
    qr->n = n;

    if (m == 0 || n == 0)
        return SPARSE_OK;

    /* Work on a mutable copy of A */
    SparseMatrix *W = sparse_copy(A);
    if (!W)
        return SPARSE_ERR_ALLOC;

    /* Optional column reordering */
    idx_t *col_reorder = NULL;
    if (opts && opts->reorder == SPARSE_REORDER_COLAMD && n > 1) {
        col_reorder = malloc((size_t)n * sizeof(idx_t));
        if (col_reorder) {
            if (sparse_reorder_colamd(A, col_reorder) != SPARSE_OK) {
                free(col_reorder);
                col_reorder = NULL;
            }
        }
    } else if (opts && opts->reorder != SPARSE_REORDER_NONE && n > 1) {
        SparseMatrix *AtA = sparse_create(n, n);
        if (AtA) {
            sparse_err_t ins_err = SPARSE_OK;
            for (idx_t row = 0; row < m && ins_err == SPARSE_OK; row++) {
                Node *nd1 = A->row_headers[row];
                while (nd1 && ins_err == SPARSE_OK) {
                    Node *nd2 = nd1->right;
                    while (nd2 && ins_err == SPARSE_OK) {
                        ins_err = sparse_insert(AtA, nd1->col, nd2->col, 1.0);
                        if (ins_err == SPARSE_OK)
                            ins_err = sparse_insert(AtA, nd2->col, nd1->col, 1.0);
                        nd2 = nd2->right;
                    }
                    if (ins_err == SPARSE_OK)
                        ins_err = sparse_insert(AtA, nd1->col, nd1->col, 1.0);
                    nd1 = nd1->right;
                }
            }
            if (ins_err == SPARSE_OK && (size_t)n <= SIZE_MAX / sizeof(idx_t)) {
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
            }
            sparse_free(AtA);
        }
    }

    /* Overflow checks for buffer allocations */
    {
        size_t tmp = 0;
        if (size_mul_overflow((size_t)n, sizeof(idx_t), &tmp) ||
            size_mul_overflow((size_t)m, sizeof(double), &tmp) ||
            size_mul_overflow((size_t)n, sizeof(double), &tmp) ||
            size_mul_overflow((size_t)k, sizeof(double), &tmp) ||
            size_mul_overflow((size_t)k, sizeof(double *), &tmp)) {
            sparse_free(W);
            return SPARSE_ERR_ALLOC;
        }
    }

    /* Allocate outputs */
    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    double *betas = calloc((size_t)k, sizeof(double));
    double **vecs = calloc((size_t)k, sizeof(double *));
    double *col_norms = calloc((size_t)n, sizeof(double));
    double *dense_col = malloc((size_t)m * sizeof(double));  /* pivot column */
    double *dense_col2 = malloc((size_t)m * sizeof(double)); /* reusable work column */
    SparseMatrix *R = NULL;
    sparse_err_t status = SPARSE_OK;

    if (!perm || !betas || !vecs || !col_norms || !dense_col || !dense_col2) {
        status = SPARSE_ERR_ALLOC;
        goto cleanup_colwise;
    }

    /* Apply column reordering to perm and swap columns in W */
    if (col_reorder) {
        /* Build inverse permutation */
        idx_t *inv = malloc((size_t)n * sizeof(idx_t));
        if (!inv) {
            status = SPARSE_ERR_ALLOC;
            goto cleanup_colwise;
        }
        for (idx_t j = 0; j < n; j++)
            inv[col_reorder[j]] = j;

        /* Rebuild W with reordered columns */
        SparseMatrix *W2 = sparse_create(m, n);
        if (!W2) {
            free(inv);
            status = SPARSE_ERR_ALLOC;
            goto cleanup_colwise;
        }
        for (idx_t i = 0; i < m; i++) {
            Node *nd = W->row_headers[i];
            while (nd) {
                sparse_err_t ierr = sparse_insert(W2, i, inv[nd->col], nd->value);
                if (ierr != SPARSE_OK) {
                    free(inv);
                    sparse_free(W2);
                    status = ierr;
                    goto cleanup_colwise;
                }
                nd = nd->right;
            }
        }
        free(inv);
        sparse_free(W);
        W = W2;

        for (idx_t j = 0; j < n; j++)
            perm[j] = col_reorder[j];
        free(col_reorder);
        col_reorder = NULL;
    } else {
        for (idx_t j = 0; j < n; j++)
            perm[j] = j;
    }

    /* Compute initial column norms from W */
    for (idx_t j = 0; j < n; j++) {
        double s = 0.0;
        Node *nd = W->col_headers[j];
        while (nd) {
            s += nd->value * nd->value;
            nd = nd->down;
        }
        col_norms[j] = s;
    }

    idx_t rank = 0;
    double r00 = 0.0;
    R = sparse_create(k, n);
    if (!R) {
        status = SPARSE_ERR_ALLOC;
        goto cleanup_colwise;
    }

    for (idx_t step = 0; step < k; step++) {
        /* Column pivoting */
        idx_t best = step;
        double best_norm = col_norms[step];
        for (idx_t j = step + 1; j < n; j++) {
            if (col_norms[j] > best_norm) {
                best_norm = col_norms[j];
                best = j;
            }
        }

        if (best != step) {
            /* Column pivoting: swap metadata (col_norms, perm) then
             * physically swap columns in W via extract/clear/reinsert
             * using dense_col and dense_col2 as temporary buffers. */
            double tmp_n = col_norms[step];
            col_norms[step] = col_norms[best];
            col_norms[best] = tmp_n;
            idx_t tmp_p = perm[step];
            perm[step] = perm[best];
            perm[best] = tmp_p;

            /* Swap columns in W: extract both, clear, reinsert swapped.
             * Use dense_col for one, dense_col2 for the other. */
            memset(dense_col, 0, (size_t)m * sizeof(double));
            memset(dense_col2, 0, (size_t)m * sizeof(double));
            sparse_extract_column(W, step, dense_col);
            sparse_extract_column(W, best, dense_col2);

            /* Remove old column entries */
            for (idx_t i = 0; i < m; i++) {
                if (fabs(dense_col[i]) > 0) {
                    sparse_err_t ierr = sparse_insert(W, i, step, 0.0);
                    if (ierr != SPARSE_OK) {
                        status = ierr;
                        goto cleanup_colwise;
                    }
                }
                if (fabs(dense_col2[i]) > 0) {
                    sparse_err_t ierr = sparse_insert(W, i, best, 0.0);
                    if (ierr != SPARSE_OK) {
                        status = ierr;
                        goto cleanup_colwise;
                    }
                }
            }
            /* Reinsert swapped — use exact nonzero check (no dropping)
             * so the swap is a true permutation of W columns. */
            for (idx_t i = 0; i < m; i++) {
                if (dense_col2[i] != 0.0) {
                    sparse_err_t ierr = sparse_insert(W, i, step, dense_col2[i]);
                    if (ierr != SPARSE_OK) {
                        status = ierr;
                        goto cleanup_colwise;
                    }
                }
                if (dense_col[i] != 0.0) {
                    sparse_err_t ierr = sparse_insert(W, i, best, dense_col[i]);
                    if (ierr != SPARSE_OK) {
                        status = ierr;
                        goto cleanup_colwise;
                    }
                }
            }
        }

        /* Rank deficiency check on column norm */
        if (step == 0)
            r00 = sqrt(best_norm);
        double rank_tol = 1e-14 * (r00 > 0.0 ? r00 : 1.0) * (double)(m > n ? m : n);
        if (sqrt(best_norm) < rank_tol)
            break;

        /* Extract column step into dense vector */
        memset(dense_col, 0, (size_t)m * sizeof(double));
        sparse_extract_column(W, step, dense_col);

        /* Compute Householder from entries step..m-1 */
        idx_t col_len = m - step;
        double *hv = malloc((size_t)col_len * sizeof(double));
        if (!hv) {
            status = SPARSE_ERR_ALLOC;
            goto cleanup_colwise;
        }
        double beta = householder_compute(&dense_col[step], hv, col_len);
        betas[step] = beta;
        vecs[step] = hv;

        /* Apply Householder to column step */
        householder_apply_to_column(hv, beta, dense_col, step, m);

        /* Write R entries from this column (row step and above-diagonal entries).
         * Drop off-diagonal entries negligible relative to the diagonal.
         * sparse_rel_tol provides a nonzero floor when the diagonal is 0. */
        double r_drop = sparse_rel_tol(fabs(dense_col[step]), DROP_TOL);
        for (idx_t i = 0; i <= step; i++) {
            if (fabs(dense_col[i]) > r_drop) {
                sparse_err_t ierr = sparse_insert(R, i, step, dense_col[i]);
                if (ierr != SPARSE_OK) {
                    status = ierr;
                    goto cleanup_colwise;
                }
            }
        }

        /* Apply Householder to remaining columns step+1..n-1 */
        for (idx_t j = step + 1; j < n; j++) {
            memset(dense_col2, 0, (size_t)m * sizeof(double));
            sparse_extract_column(W, j, dense_col2);
            householder_apply_to_column(hv, beta, dense_col2, step, m);

            /* Write back modified column to W.
             * Clear entries with row >= step by traversing col_headers once
             * (avoids O(m * nnz_in_row) sparse_get_phys scans). */
            {
                Node *nd = W->col_headers[j];
                while (nd) {
                    Node *next = nd->down;
                    if (nd->row >= step) {
                        sparse_err_t ierr = sparse_insert(W, nd->row, j, 0.0);
                        if (ierr != SPARSE_OK) {
                            status = ierr;
                            goto cleanup_colwise;
                        }
                    }
                    nd = next;
                }
            }
            /* Reinsert — exact nonzero check (no dropping) for
             * bitwise-identical results with dense-mode. */
            for (idx_t i = step; i < m; i++) {
                if (dense_col2[i] != 0.0) {
                    sparse_err_t ierr = sparse_insert(W, i, j, dense_col2[i]);
                    if (ierr != SPARSE_OK) {
                        status = ierr;
                        goto cleanup_colwise;
                    }
                }
            }

            /* Update column norm using the value actually stored */
            double entry_step = dense_col2[step];
            col_norms[j] -= entry_step * entry_step;
            if (col_norms[j] < 0.0)
                col_norms[j] = 0.0;
        }

        if (fabs(dense_col[step]) < rank_tol)
            break;
        rank++;
    }

    /* Extract off-diagonal R entries from W: for each completed step,
     * R(step, j) for j > step lives in W(step, j) after Householder application.
     * The pivot column R entries were already inserted during the loop. */
    {
        idx_t steps = rank; /* number of fully completed steps */
        /* If we broke due to rank deficiency after applying reflectors,
         * the last step's R row also needs extraction */
        if (rank < k && vecs && vecs[rank]) {
            /* Reflectors were applied for step `rank` (the one that
             * triggered the break), so include its R row too. */
            steps = rank + 1;
        }
        for (idx_t s = 0; s < steps; s++) {
            /* Traverse row s nonzeros once instead of probing every (s,j).
             * Drop off-diagonal entries negligible relative to the diagonal.
             * sparse_rel_tol provides a nonzero floor when R(s,s) is 0. */
            double r_drop_s = sparse_rel_tol(fabs(sparse_get_phys(R, s, s)), DROP_TOL);
            Node *rnd = W->row_headers[s];
            while (rnd) {
                if (rnd->col > s && fabs(rnd->value) > r_drop_s) {
                    sparse_err_t ierr = sparse_insert(R, s, rnd->col, rnd->value);
                    if (ierr != SPARSE_OK) {
                        status = ierr;
                        goto cleanup_colwise;
                    }
                }
                rnd = rnd->right;
            }
        }
    }

    free(col_norms);
    col_norms = NULL;
    free(dense_col);
    dense_col = NULL;
    free(dense_col2);
    dense_col2 = NULL;
    sparse_free(W);
    W = NULL;

    qr->R = R;
    qr->betas = betas;
    qr->v_vectors = vecs;
    qr->col_perm = perm;
    qr->rank = rank;
    qr->economy = (opts && opts->economy) ? 1 : 0;

    return SPARSE_OK;

cleanup_colwise:
    free(col_norms);
    free(dense_col);
    free(dense_col2);
    free(col_reorder);
    free(perm);
    free(betas);
    if (vecs) {
        for (idx_t i = 0; i < k; i++)
            free(vecs[i]);
        free(vecs);
    }
    sparse_free(R);
    sparse_free(W);
    return status;
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

    /* Reject non-identity permutations (QR factors physical storage) */
    {
        const idx_t *rp = sparse_row_perm(A);
        const idx_t *cp = sparse_col_perm(A);
        idx_t nr = sparse_rows(A);
        idx_t nc = sparse_cols(A);
        if (rp) {
            for (idx_t i = 0; i < nr; i++) {
                if (rp[i] != i)
                    return SPARSE_ERR_BADARG;
            }
        }
        if (cp) {
            for (idx_t i = 0; i < nc; i++) {
                if (cp[i] != i)
                    return SPARSE_ERR_BADARG;
            }
        }
    }

    /* Dispatch to column-by-column path if sparse_mode is enabled */
    if (opts && opts->sparse_mode)
        return sparse_qr_factor_colwise(A, opts, qr);

    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);
    idx_t k = (m < n) ? m : n; /* min(m, n) */

    qr->m = m;
    qr->n = n;

    if (m == 0 || n == 0)
        return SPARSE_OK;

    /* Compute optional column reordering */
    idx_t *col_reorder = NULL;
    if (opts && opts->reorder == SPARSE_REORDER_COLAMD && n > 1) {
        /* COLAMD operates directly on A's structure — no need to form A^T*A */
        col_reorder = malloc((size_t)n * sizeof(idx_t));
        if (col_reorder) {
            if (sparse_reorder_colamd(A, col_reorder) != SPARSE_OK) {
                free(col_reorder);
                col_reorder = NULL;
            }
        }
    } else if (opts && opts->reorder != SPARSE_REORDER_NONE && n > 1) {
        /* AMD/RCM: build A^T*A pattern for the symmetric ordering */
        SparseMatrix *AtA = sparse_create(n, n);
        if (AtA) {
            sparse_err_t ins_err = SPARSE_OK;
            for (idx_t row = 0; row < m && ins_err == SPARSE_OK; row++) {
                /* Collect column indices in this row */
                Node *nd1 = A->row_headers[row];
                while (nd1 && ins_err == SPARSE_OK) {
                    Node *nd2 = nd1->right;
                    while (nd2 && ins_err == SPARSE_OK) {
                        ins_err = sparse_insert(AtA, nd1->col, nd2->col, 1.0);
                        if (ins_err == SPARSE_OK)
                            ins_err = sparse_insert(AtA, nd2->col, nd1->col, 1.0);
                        nd2 = nd2->right;
                    }
                    /* Diagonal */
                    if (ins_err == SPARSE_OK)
                        ins_err = sparse_insert(AtA, nd1->col, nd1->col, 1.0);
                    nd1 = nd1->right;
                }
            }
            if (ins_err != SPARSE_OK) {
                /* Insertion failed; abandon reordering, proceed without it */
                sparse_free(AtA);
                AtA = NULL;
            }

            if (AtA) {
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
    }

    /* Overflow check for dense workspace sizing: m * n * sizeof(double) */
    {
        size_t mn = 0, mn_bytes = 0;
        if (size_mul_overflow((size_t)m, (size_t)n, &mn) ||
            size_mul_overflow(mn, sizeof(double), &mn_bytes)) {
            free(col_reorder);
            return SPARSE_ERR_ALLOC;
        }
    }

    /* Allocate dense m×n working matrix (column-major) */
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
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
    sparse_err_t ins_err = SPARSE_OK;
    for (idx_t i = 0; i < steps_done && ins_err == SPARSE_OK; i++) {
        /* Drop off-diagonal entries negligible relative to the diagonal.
         * sparse_rel_tol provides a nonzero floor when R(i,i) is 0. */
        double r_ii_drop = sparse_rel_tol(fabs(W[(size_t)i * (size_t)m + (size_t)i]), DROP_TOL);
        for (idx_t j = i; j < n && ins_err == SPARSE_OK; j++) {
            double val = W[(size_t)j * (size_t)m + (size_t)i];
            if (fabs(val) > r_ii_drop)
                ins_err = sparse_insert(R, i, j, val);
        }
    }

    free(W);

    if (ins_err != SPARSE_OK) {
        sparse_free(R);
        free(perm);
        free(betas);
        for (idx_t i = 0; i < k; i++)
            free(vecs[i]);
        free(vecs);
        return ins_err;
    }

    qr->R = R;
    qr->betas = betas;
    qr->v_vectors = vecs;
    qr->col_perm = perm;
    qr->rank = rank;
    qr->economy = (opts && opts->economy) ? 1 : 0;

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
    /* Economy QR: form m×n thin Q; full QR: form m×m */
    idx_t ncols = (qr->economy && qr->n < m) ? qr->n : m;

    /* Overflow-safe computation of m*ncols*sizeof(double) */
    size_t total = 0, bytes = 0;
    if (size_mul_overflow((size_t)m, (size_t)ncols, &total) ||
        size_mul_overflow(total, sizeof(double), &bytes))
        return SPARSE_ERR_ALLOC;

    /* Start with first ncols columns of identity (m × ncols) */
    memset(Q, 0, bytes);
    for (idx_t i = 0; i < ncols; i++)
        Q[(size_t)i * (size_t)m + (size_t)i] = 1.0;

    /* Apply Q to each column of the truncated identity.
     * Each column is length m; apply_q works on m-vectors. */
    for (idx_t j = 0; j < ncols; j++) {
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

    double r00_abs = fabs(sparse_get_phys(qr->R, 0, 0));
    double solve_tol = sparse_rel_tol(r00_abs, DROP_TOL);

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
        if (fabs(diag) < solve_tol) {
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
 * Iterative refinement
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_qr_refine(const sparse_qr_t *qr, const SparseMatrix *A, const double *b,
                              double *x, idx_t max_refine, double *residual) {
    if (!qr || !A || !b || !x)
        return SPARSE_ERR_NULL;

    idx_t m = qr->m;
    idx_t n = qr->n;

    /* Validate A matches factorization dimensions */
    if (sparse_rows(A) != m || sparse_cols(A) != n)
        return SPARSE_ERR_SHAPE;

    /* Overflow checks for buffer allocations */
    {
        size_t tmp = 0;
        if (size_mul_overflow((size_t)m, sizeof(double), &tmp) ||
            size_mul_overflow((size_t)n, sizeof(double), &tmp))
            return SPARSE_ERR_ALLOC;
    }

    double *r = malloc((size_t)m * sizeof(double));
    double *dx = malloc((size_t)n * sizeof(double));
    double *Ax = malloc((size_t)m * sizeof(double));
    if (!r || !dx || !Ax) {
        free(r);
        free(dx);
        free(Ax);
        return SPARSE_ERR_ALLOC;
    }

    double prev_rnorm = INFINITY;
    sparse_err_t status = SPARSE_OK;

    for (idx_t iter = 0; iter <= max_refine; iter++) {
        /* Compute residual r = b - A*x */
        sparse_matvec(A, x, Ax);
        double rnorm = 0.0;
        for (idx_t i = 0; i < m; i++) {
            r[i] = b[i] - Ax[i];
            rnorm += r[i] * r[i];
        }
        rnorm = sqrt(rnorm);

        if (residual)
            *residual = rnorm;

        /* When iter reaches max_refine, stop after computing the residual
         * (so max_refine == 0 means: compute residual once and return). */
        if (iter >= max_refine)
            break;

        /* Stop if residual is not decreasing */
        if (rnorm >= prev_rnorm)
            break;
        prev_rnorm = rnorm;

        /* Solve for correction: QR * dx = r */
        sparse_err_t serr = sparse_qr_solve(qr, r, dx, NULL);
        if (serr != SPARSE_OK) {
            status = serr;
            break;
        }

        /* Apply correction and validate: roll back if residual increases */
        for (idx_t i = 0; i < n; i++)
            x[i] += dx[i];

        /* Compute post-update residual */
        sparse_matvec(A, x, Ax);
        double new_rnorm = 0.0;
        for (idx_t i = 0; i < m; i++) {
            double ri = b[i] - Ax[i];
            new_rnorm += ri * ri;
        }
        new_rnorm = sqrt(new_rnorm);

        if (new_rnorm >= rnorm) {
            /* Update made things worse — roll back */
            for (idx_t i = 0; i < n; i++)
                x[i] -= dx[i];
            break;
        }
    }

    free(r);
    free(dx);
    free(Ax);
    return status;
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

/* ═══════════════════════════════════════════════════════════════════════
 * Rank-revealing diagnostics
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_qr_diag_r(const sparse_qr_t *qr, double *diag) {
    if (!qr || !diag)
        return SPARSE_ERR_NULL;
    if (!qr->R)
        return SPARSE_ERR_BADARG;

    idx_t k = (qr->m < qr->n) ? qr->m : qr->n;
    for (idx_t i = 0; i < k; i++)
        diag[i] = sparse_get_phys(qr->R, i, i);

    return SPARSE_OK;
}

sparse_err_t sparse_qr_rank_info(const sparse_qr_t *qr, double tol, sparse_qr_rank_info_t *info) {
    if (!qr || !info)
        return SPARSE_ERR_NULL;
    if (!qr->R)
        return SPARSE_ERR_BADARG;

    memset(info, 0, sizeof(*info));
    idx_t k = (qr->m < qr->n) ? qr->m : qr->n;
    info->k = k;

    if (k == 0)
        return SPARSE_OK;

    /* Compute absolute threshold */
    double r00 = fabs(sparse_get_phys(qr->R, 0, 0));
    double abs_tol;
    if (tol <= 0.0) {
        double eps = 2.2204460492503131e-16;
        idx_t mn = (qr->m > qr->n) ? qr->m : qr->n;
        abs_tol = eps * (double)mn * r00;
    } else {
        abs_tol = tol * r00;
    }

    info->r_max = r00;
    info->r_min = r00;
    info->rank = 0;

    for (idx_t i = 0; i < k; i++) {
        double ri = fabs(sparse_get_phys(qr->R, i, i));
        if (ri > abs_tol) {
            info->rank++;
            if (ri < info->r_min)
                info->r_min = ri;
            if (ri > info->r_max)
                info->r_max = ri;
        } else {
            break;
        }
    }

    if (info->rank > 0 && info->r_min > 0.0) {
        info->condest = info->r_max / info->r_min;
        info->near_deficient = (info->r_min / info->r_max < 1e-8) ? 1 : 0;
    } else {
        info->condest = INFINITY;
        info->near_deficient = 1;
    }

    return SPARSE_OK;
}

double sparse_qr_condest(const sparse_qr_t *qr) {
    if (!qr || !qr->R)
        return -1.0;

    idx_t k = (qr->m < qr->n) ? qr->m : qr->n;
    if (k == 0)
        return -1.0;

    idx_t rank = qr->rank;
    if (rank <= 0)
        return -1.0;

    double r00 = fabs(sparse_get_phys(qr->R, 0, 0));
    double rkk = fabs(sparse_get_phys(qr->R, rank - 1, rank - 1));
    if (rkk == 0.0)
        return INFINITY;

    return r00 / rkk;
}

sparse_err_t sparse_qr_nullspace(const sparse_qr_t *qr, double tol, double *basis,
                                 idx_t *null_dim) {
    if (!qr || !null_dim)
        return SPARSE_ERR_NULL;

    /* Validate factorization data up front — even dimension-only queries
     * need R to compute rank correctly. */
    if (!qr->R || !qr->col_perm)
        return SPARSE_ERR_NULL;

    idx_t n = qr->n;
    idx_t rank = sparse_qr_rank(qr, tol);
    idx_t ndim = n - rank;
    *null_dim = ndim;

    if (ndim == 0 || !basis)
        return SPARSE_OK;

    idx_t m = qr->m;
    double ns_r00 = fabs(sparse_get_phys(qr->R, 0, 0));
    double ns_tol = sparse_rel_tol(ns_r00, DROP_TOL);

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
            if (fabs(diag) > ns_tol)
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

/* ═══════════════════════════════════════════════════════════════════════
 * Minimum-norm least-squares for underdetermined systems
 *
 * For m < n, computes the minimum 2-norm solution to A*x = b.
 *
 * Algorithm:
 *   1. Compute A^T via sparse_transpose
 *   2. Factor A^T with QR: A^T * P = Q * R  (R is m×m upper triangular)
 *   3. Permute b: bp = P^T * b
 *   4. Forward substitute: solve R^T * y = bp
 *   5. Apply Q: x = Q * y
 *
 * The result x has minimum 2-norm among all solutions of A*x = b.
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_qr_solve_minnorm(const SparseMatrix *A, const double *b, double *x,
                                     const sparse_qr_opts_t *opts) {
    if (!A || !b || !x)
        return SPARSE_ERR_NULL;

    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);

    /* Handle empty matrices */
    if (m == 0 || n == 0)
        return SPARSE_OK;

    /* For m >= n, fall back to regular least-squares via QR */
    if (m >= n) {
        sparse_qr_t qr;
        sparse_err_t err = opts ? sparse_qr_factor_opts(A, opts, &qr) : sparse_qr_factor(A, &qr);
        if (err != SPARSE_OK)
            return err;
        err = sparse_qr_solve(&qr, b, x, NULL);
        sparse_qr_free(&qr);
        return err;
    }

    /* Underdetermined: m < n */

    /* Step 1: Transpose A (n×m matrix) */
    SparseMatrix *At = sparse_transpose(A);
    if (!At)
        return SPARSE_ERR_ALLOC;

    /* Step 2: Factor A^T with QR.
     * A^T is n×m with n > m, so this is overdetermined QR.
     * R will be m×m upper triangular. */
    sparse_qr_t qr_t;
    sparse_err_t err = opts ? sparse_qr_factor_opts(At, opts, &qr_t) : sparse_qr_factor(At, &qr_t);
    sparse_free(At);
    if (err != SPARSE_OK)
        return err;

    idx_t rank = qr_t.rank;

    /* Step 3: Permute b via the column permutation of A^T's QR.
     * col_perm[k] = original column of A^T = original row of A.
     * bp[k] = b[col_perm[k]] */
    double *bp = malloc((size_t)m * sizeof(double));
    if (!bp) {
        sparse_qr_free(&qr_t);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t k = 0; k < m; k++)
        bp[k] = b[qr_t.col_perm[k]];

    /* Step 4: Forward substitute: solve R^T * y = bp.
     * R^T is lower triangular (m×m). */
    double *y = calloc((size_t)n, sizeof(double));
    if (!y) {
        free(bp);
        sparse_qr_free(&qr_t);
        return SPARSE_ERR_ALLOC;
    }

    double r00_abs = (rank > 0) ? fabs(sparse_get_phys(qr_t.R, 0, 0)) : 0.0;
    double solve_tol = sparse_rel_tol(r00_abs, DROP_TOL);

    for (idx_t i = 0; i < rank; i++) {
        /* R^T(i,j) = R(j,i) for j <= i.
         * Sum contributions from R^T(i, 0..i-1) * y[0..i-1] */
        double sum = 0.0;
        double diag = 0.0;

        /* Walk column i of R to get R(j,i) for j <= i → R^T(i,j) */
        for (Node *nd = qr_t.R->col_headers[i]; nd; nd = nd->down) {
            idx_t j = nd->row;
            if (j == i) {
                diag = nd->value;
            } else if (j < i) {
                sum += nd->value * y[j]; // NOLINT(clang-analyzer-security.ArrayBound)
            }
        }

        if (fabs(diag) < solve_tol) {
            y[i] = 0.0;
        } else {
            y[i] = (bp[i] - sum) / diag;
        }
    }

    free(bp);

    /* Step 5: Apply Q: x = Q * y */
    sparse_qr_apply_q(&qr_t, 0, y, x);

    free(y);
    sparse_qr_free(&qr_t);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Iterative refinement for minimum-norm solutions
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_qr_refine_minnorm(const SparseMatrix *A, const double *b, double *x,
                                      idx_t max_refine, double *residual,
                                      const sparse_qr_opts_t *opts) {
    if (!A || !b || !x)
        return SPARSE_ERR_NULL;

    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);

    double *r = malloc((size_t)m * sizeof(double));
    double *dx = malloc((size_t)n * sizeof(double));
    double *Ax_buf = malloc((size_t)m * sizeof(double));
    if (!r || !dx || !Ax_buf) {
        free(r);
        free(dx);
        free(Ax_buf);
        return SPARSE_ERR_ALLOC;
    }

    double prev_rnorm = INFINITY;
    sparse_err_t status = SPARSE_OK;

    for (idx_t iter = 0; iter <= max_refine; iter++) {
        /* Compute residual r = b - A*x */
        sparse_matvec(A, x, Ax_buf);
        double rnorm = 0.0;
        for (idx_t i = 0; i < m; i++) {
            r[i] = b[i] - Ax_buf[i];
            rnorm += r[i] * r[i];
        }
        rnorm = sqrt(rnorm);

        if (residual)
            *residual = rnorm;

        if (iter >= max_refine)
            break;

        /* Stop if residual is not decreasing */
        if (rnorm >= prev_rnorm)
            break;
        prev_rnorm = rnorm;

        /* Solve for minimum-norm correction: dx = minnorm(A, r) */
        sparse_err_t serr = sparse_qr_solve_minnorm(A, r, dx, opts);
        if (serr != SPARSE_OK) {
            status = serr;
            break;
        }

        /* Apply correction */
        for (idx_t i = 0; i < n; i++)
            x[i] += dx[i];
    }

    free(r);
    free(dx);
    free(Ax_buf);
    return status;
}
