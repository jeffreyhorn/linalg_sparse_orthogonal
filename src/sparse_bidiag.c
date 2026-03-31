#include "sparse_bidiag.h"
#include "sparse_matrix_internal.h"
#include "sparse_vector.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Reuse Householder helpers — same as in sparse_qr.c */
static double bidiag_householder_compute(const double *x, double *v, idx_t len) {
    if (len <= 0)
        return 0.0;
    memcpy(v, x, (size_t)len * sizeof(double));
    double sigma = 0.0;
    for (idx_t i = 1; i < len; i++)
        sigma += v[i] * v[i];
    if (sigma == 0.0 && v[0] >= 0.0)
        return 0.0;
    double xnorm = sqrt(v[0] * v[0] + sigma);
    if (v[0] >= 0.0)
        v[0] += xnorm;
    else
        v[0] -= xnorm;
    double vtv = v[0] * v[0] + sigma;
    if (vtv == 0.0)
        return 0.0;
    return 2.0 / vtv;
}

static void bidiag_householder_apply(const double *v, double beta, double *y, idx_t len) {
    if (beta == 0.0)
        return;
    double vty = 0.0;
    for (idx_t i = 0; i < len; i++)
        vty += v[i] * y[i];
    double scale = beta * vty;
    for (idx_t i = 0; i < len; i++)
        y[i] -= scale * v[i];
}

void sparse_bidiag_free(sparse_bidiag_t *bidiag) {
    if (!bidiag)
        return;
    free(bidiag->diag);
    free(bidiag->superdiag);
    if (bidiag->u_vecs) {
        idx_t k = (bidiag->m < bidiag->n) ? bidiag->m : bidiag->n;
        for (idx_t i = 0; i < k; i++)
            free(bidiag->u_vecs[i]);
        free(bidiag->u_vecs);
    }
    free(bidiag->u_betas);
    if (bidiag->v_vecs) {
        idx_t k = (bidiag->m < bidiag->n) ? bidiag->m : bidiag->n;
        idx_t nv = (k > 1) ? k - 1 : 0;
        for (idx_t i = 0; i < nv; i++)
            free(bidiag->v_vecs[i]);
        free(bidiag->v_vecs);
    }
    free(bidiag->v_betas);
    memset(bidiag, 0, sizeof(*bidiag));
}

// NOLINTNEXTLINE(misc-no-recursion)
sparse_err_t sparse_bidiag_factor(const SparseMatrix *A, sparse_bidiag_t *bidiag) {
    if (!bidiag)
        return SPARSE_ERR_NULL;
    memset(bidiag, 0, sizeof(*bidiag));
    if (!A)
        return SPARSE_ERR_NULL;

    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);

    /* Reject non-identity permutations */
    {
        const idx_t *rp = sparse_row_perm(A);
        const idx_t *cp = sparse_col_perm(A);
        if (rp) {
            for (idx_t i = 0; i < m; i++)
                if (rp[i] != i)
                    return SPARSE_ERR_BADARG;
        }
        if (cp) {
            for (idx_t i = 0; i < n; i++)
                if (cp[i] != i)
                    return SPARSE_ERR_BADARG;
        }
    }

    /* Wide matrix (m < n): factor A^T (tall) and set transposed flag.
     * If A^T = U_t * B_t * V_t^T, then A = V_t * B_t^T * U_t^T.
     * The reflectors are stored as-is (for A^T); SVD code swaps U↔V. */
    if (m < n) {
        SparseMatrix *At = sparse_transpose(A);
        if (!At)
            return SPARSE_ERR_ALLOC;

        sparse_bidiag_t bd_t;
        sparse_err_t err = sparse_bidiag_factor(At, &bd_t);
        sparse_free(At);
        if (err != SPARSE_OK)
            return err;

        /* Store A^T's factorization with transposed flag */
        bidiag->m = m;
        bidiag->n = n;
        bidiag->diag = bd_t.diag;
        bidiag->superdiag = bd_t.superdiag;
        bidiag->u_vecs = bd_t.u_vecs;
        bidiag->u_betas = bd_t.u_betas;
        bidiag->v_vecs = bd_t.v_vecs;
        bidiag->v_betas = bd_t.v_betas;
        bidiag->transposed = 1;
        return SPARSE_OK;
    }

    idx_t k = n; /* m >= n guaranteed */

    bidiag->m = m;
    bidiag->n = n;

    if (k == 0)
        return SPARSE_OK;

    /* Work on a dense m×n column-major copy (same approach as dense-mode QR).
     * Bidiagonalization is inherently dense due to fill-in from right
     * Householder reflections. */
    /* Overflow-safe check for m * n * sizeof(double) */
    {
        size_t mn = (size_t)m * (size_t)n;
        if (n > 0 && mn / (size_t)n != (size_t)m)
            return SPARSE_ERR_ALLOC;
        if (mn > SIZE_MAX / sizeof(double))
            return SPARSE_ERR_ALLOC;
    }

    double *W = calloc((size_t)m * (size_t)n, sizeof(double));
    if (!W)
        return SPARSE_ERR_ALLOC;

    /* Copy A into W (column-major) */
    for (idx_t i = 0; i < m; i++) {
        Node *nd = A->row_headers[i];
        while (nd) {
            W[(size_t)nd->col * (size_t)m + (size_t)i] = nd->value;
            nd = nd->right;
        }
    }

    /* Overflow checks for output array allocations */
    {
        size_t max_dim = (size_t)(m > n ? m : n);
        if (max_dim > SIZE_MAX / sizeof(double) || (size_t)k > SIZE_MAX / sizeof(double *)) {
            free(W);
            return SPARSE_ERR_ALLOC;
        }
    }

    /* Allocate output arrays */
    double *diag = calloc((size_t)k, sizeof(double));
    double *superdiag = (k > 1) ? calloc((size_t)(k - 1), sizeof(double)) : NULL;
    double **u_vecs = calloc((size_t)k, sizeof(double *));
    double *u_betas = calloc((size_t)k, sizeof(double));
    idx_t nv = (k > 1) ? k - 1 : 0;
    double **v_vecs = (nv > 0) ? calloc((size_t)nv, sizeof(double *)) : NULL;
    double *v_betas = (nv > 0) ? calloc((size_t)nv, sizeof(double)) : NULL;

    if (!diag || !u_vecs || !u_betas || (k > 1 && (!superdiag || !v_vecs || !v_betas))) {
        free(W);
        free(diag);
        free(superdiag);
        free(u_vecs);
        free(u_betas);
        free(v_vecs);
        free(v_betas);
        return SPARSE_ERR_ALLOC;
    }

    idx_t maxdim = (m > n) ? m : n;
    double *hv = malloc((size_t)maxdim * sizeof(double));
    if (!hv) {
        free(W);
        free(diag);
        free(superdiag);
        free(u_vecs);
        free(u_betas);
        free(v_vecs);
        free(v_betas);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t step = 0; step < k; step++) {
        /* --- Left Householder: zero column step below diagonal --- */
        idx_t col_len = m - step;
        double *col_ptr = &W[(size_t)step * (size_t)m + (size_t)step];

        double beta_u = bidiag_householder_compute(col_ptr, hv, col_len);
        u_betas[step] = beta_u;
        u_vecs[step] = malloc((size_t)col_len * sizeof(double));
        if (!u_vecs[step]) {
            free(hv);
            free(W);
            bidiag->diag = diag;
            bidiag->superdiag = superdiag;
            bidiag->u_vecs = u_vecs;
            bidiag->u_betas = u_betas;
            bidiag->v_vecs = v_vecs;
            bidiag->v_betas = v_betas;
            sparse_bidiag_free(bidiag);
            return SPARSE_ERR_ALLOC;
        }
        memcpy(u_vecs[step], hv, (size_t)col_len * sizeof(double));

        /* Apply left Householder to column step */
        bidiag_householder_apply(hv, beta_u, col_ptr, col_len);
        diag[step] = col_ptr[0];

        /* Apply left Householder to remaining columns step+1..n-1 */
        for (idx_t j = step + 1; j < n; j++) {
            double *cj = &W[(size_t)j * (size_t)m + (size_t)step];
            bidiag_householder_apply(hv, beta_u, cj, col_len);
        }

        /* --- Right Householder: zero row step right of superdiagonal --- */
        if (step < k - 1) {
            idx_t row_len = n - step - 1;
            /* Extract row step entries from columns step+1..n-1 */
            double *row_entries = hv; /* reuse buffer */
            for (idx_t j = 0; j < row_len; j++)
                row_entries[j] = W[(size_t)(step + 1 + j) * (size_t)m + (size_t)step];

            double *rv = malloc((size_t)row_len * sizeof(double));
            if (!rv) {
                free(hv);
                free(W);
                bidiag->diag = diag;
                bidiag->superdiag = superdiag;
                bidiag->u_vecs = u_vecs;
                bidiag->u_betas = u_betas;
                bidiag->v_vecs = v_vecs;
                bidiag->v_betas = v_betas;
                sparse_bidiag_free(bidiag);
                return SPARSE_ERR_ALLOC;
            }

            double beta_v = bidiag_householder_compute(row_entries, rv, row_len);
            v_betas[step] = beta_v;
            v_vecs[step] = rv;

            /* Apply right Householder to row step */
            bidiag_householder_apply(rv, beta_v, row_entries, row_len);
            superdiag[step] = row_entries[0];

            /* Write back the modified row entries */
            for (idx_t j = 0; j < row_len; j++)
                W[(size_t)(step + 1 + j) * (size_t)m + (size_t)step] = row_entries[j];

            /* Apply right Householder to rows step+1..m-1 (columns step+1..n-1) */
            for (idx_t i = step + 1; i < m; i++) {
                /* Extract row i entries from columns step+1..n-1 */
                for (idx_t j = 0; j < row_len; j++)
                    hv[j] = W[(size_t)(step + 1 + j) * (size_t)m + (size_t)i];
                bidiag_householder_apply(rv, beta_v, hv, row_len);
                for (idx_t j = 0; j < row_len; j++)
                    W[(size_t)(step + 1 + j) * (size_t)m + (size_t)i] = hv[j];
            }
        }
    }

    free(hv);
    free(W);

    bidiag->diag = diag;
    bidiag->superdiag = superdiag;
    bidiag->u_vecs = u_vecs;
    bidiag->u_betas = u_betas;
    bidiag->v_vecs = v_vecs;
    bidiag->v_betas = v_betas;

    return SPARSE_OK;
}
