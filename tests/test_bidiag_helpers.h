#ifndef TEST_BIDIAG_HELPERS_H
#define TEST_BIDIAG_HELPERS_H

#include "sparse_bidiag.h"
#include "sparse_matrix.h"

#include <math.h>
#include <stdlib.h>

/**
 * Compute ||A - U*B*V^T|| where U/V are applied via Householder sequences
 * and B is bidiagonal. Uses explicit dense reconstruction.
 */
static inline double tf_bidiag_reconstruction_max_error(const SparseMatrix *A,
                                                        const sparse_bidiag_t *bd) {
    /* For transposed case: reflectors are for A^T (n x m).
     * Reconstruct A^T using reflectors, then compare with A transposed. */
    if (bd->transposed) {
        SparseMatrix *At = sparse_transpose(A);
        if (!At)
            return HUGE_VAL;
        /* Create a non-transposed bd view for A^T */
        sparse_bidiag_t bd_t = *bd;
        bd_t.m = bd->n; /* A^T is n x m */
        bd_t.n = bd->m;
        bd_t.transposed = 0;
        double err = tf_bidiag_reconstruction_max_error(At, &bd_t);
        sparse_free(At);
        return err;
    }

    idx_t m = bd->m;
    idx_t n = bd->n;
    idx_t k = (m < n) ? m : n;
    if (k == 0)
        return 0.0;

    /* Form dense B (m x n, upper bidiagonal) */
    double *B = calloc((size_t)m * (size_t)n, sizeof(double));
    if (!B)
        return HUGE_VAL;
    for (idx_t i = 0; i < k; i++)
        B[(size_t)i * (size_t)m + (size_t)i] = bd->diag[i]; /* col-major */
    for (idx_t i = 0; i < k - 1; i++)
        B[(size_t)(i + 1) * (size_t)m + (size_t)i] = bd->superdiag[i];

    /* Apply U to columns of B: U*B, applying left Householder reflectors.
     * U = H_0 * H_1 * ... * H_{k-1}, so U*B = H_0 * H_1 * ... * H_{k-1} * B
     * Apply reflectors right-to-left to each column of B. */
    for (idx_t j = 0; j < n; j++) {
        double *col = &B[(size_t)j * (size_t)m];
        for (idx_t i = k - 1; i >= 0; i--) {
            if (bd->u_betas[i] == 0.0)
                continue;
            idx_t len = m - i;
            /* Apply (I - beta*v*v^T) to col[i..m-1] */
            double vty = 0.0;
            for (idx_t p = 0; p < len; p++)
                vty += bd->u_vecs[i][p] * col[i + p];
            double scale = bd->u_betas[i] * vty;
            for (idx_t p = 0; p < len; p++)
                col[i + p] -= scale * bd->u_vecs[i][p];
        }
    }

    /* Now B holds U*B. Compute (U*B) * V^T by applying right Householder
     * reflectors to rows: for each row, apply V reflectors.
     * V = H_0^R * H_1^R * ... so V^T applies reflectors in reverse.
     * (U*B)*V^T: for each row i, apply reflectors 0, 1, ... to
     * entries columns step+1..n-1. */
    idx_t nv = (k > 1) ? k - 1 : 0;
    if (nv > 0) {
        double *row_buf = malloc((size_t)n * sizeof(double));
        if (!row_buf) {
            free(B);
            return HUGE_VAL;
        }
        for (idx_t i = 0; i < m; i++) {
            /* Extract row i */
            for (idx_t j = 0; j < n; j++)
                row_buf[j] = B[(size_t)j * (size_t)m + (size_t)i];

            /* Apply V^T: reverse order */
            for (idx_t s = nv - 1; s >= 0; s--) {
                if (bd->v_betas[s] == 0.0)
                    continue;
                idx_t len = n - s - 1;
                double vty = 0.0;
                for (idx_t p = 0; p < len; p++)
                    vty += bd->v_vecs[s][p] * row_buf[s + 1 + p];
                double sc = bd->v_betas[s] * vty;
                for (idx_t p = 0; p < len; p++)
                    row_buf[s + 1 + p] -= sc * bd->v_vecs[s][p];
            }

            /* Write back */
            for (idx_t j = 0; j < n; j++)
                B[(size_t)j * (size_t)m + (size_t)i] = row_buf[j];
        }
        free(row_buf);
    }

    /* Now B holds U*B*V^T. Compute ||A - U*B*V^T|| */
    double maxerr = 0.0;
    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = 0; j < n; j++) {
            double a_val = sparse_get_phys(A, i, j);
            double ubvt_val = B[(size_t)j * (size_t)m + (size_t)i];
            double e = fabs(a_val - ubvt_val);
            if (e > maxerr)
                maxerr = e;
        }
    }

    free(B);
    return maxerr;
}

#endif
