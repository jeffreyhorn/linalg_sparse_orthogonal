#include "sparse_svd.h"
#include "sparse_bidiag.h"
#include "sparse_dense.h"
#include "sparse_matrix.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Householder application helper (same as in sparse_qr.c / sparse_bidiag.c)
 * ═══════════════════════════════════════════════════════════════════════ */

static void hh_apply(const double *v, double beta, double *y, idx_t len) {
    if (beta == 0.0)
        return;
    double vty = 0.0;
    for (idx_t i = 0; i < len; i++)
        vty += v[i] * y[i];
    double scale = beta * vty;
    for (idx_t i = 0; i < len; i++)
        y[i] -= scale * v[i];
}

/* ═══════════════════════════════════════════════════════════════════════
 * Extract explicit U and V from bidiagonal factorization
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_svd_extract_uv(const sparse_bidiag_t *bd, double *U, double *V) {
    if (!bd)
        return SPARSE_ERR_NULL;

    idx_t m = bd->m;
    idx_t n = bd->n;
    idx_t k = (m < n) ? m : n;

    /* For transposed factorizations (m < n): the reflectors are for A^T (n×m).
     * U-reflectors have length n-i (left reflectors of A^T → right of A).
     * V-reflectors have length m-i-1 (right reflectors of A^T → left of A).
     * We form U_t and V_t for A^T, then swap: A's U = V_t, A's V = U_t. */
    if (bd->transposed) {
        idx_t mt = n; /* A^T dimensions */
        idx_t nt = m;

        /* Form U_t (n×k) from left reflectors of A^T */
        double *Ut = NULL;
        if (V) { /* A's V = A^T's U_t */
            Ut = calloc((size_t)mt * (size_t)k, sizeof(double));
            if (!Ut)
                return SPARSE_ERR_ALLOC;
            /* Start with I_k embedded in n×k */
            for (idx_t i = 0; i < k; i++)
                Ut[(size_t)i * (size_t)mt + (size_t)i] = 1.0;
            /* Apply U-reflectors (for A^T) right-to-left */
            for (idx_t j = 0; j < k; j++) {
                for (idx_t r = k - 1; r >= 0; r--) {
                    if (bd->u_betas[r] == 0.0)
                        continue;
                    idx_t len = mt - r;
                    hh_apply(bd->u_vecs[r], bd->u_betas[r], &Ut[(size_t)j * (size_t)mt + (size_t)r],
                             len);
                }
            }
            /* Copy Ut (n×k) to V (n×k) — A's V = A^T's U */
            memcpy(V, Ut, (size_t)n * (size_t)k * sizeof(double));
            free(Ut);
        }

        /* Form V_t (m×k) from right reflectors of A^T */
        if (U) { /* A's U = A^T's V_t */
            idx_t nv = (k > 1) ? k - 1 : 0;
            double *Vt = calloc((size_t)nt * (size_t)k, sizeof(double));
            if (!Vt)
                return SPARSE_ERR_ALLOC;
            /* Start with I_k embedded in m×k */
            for (idx_t i = 0; i < k; i++)
                Vt[(size_t)i * (size_t)nt + (size_t)i] = 1.0;
            /* Apply V-reflectors (for A^T) right-to-left to each column */
            for (idx_t j = 0; j < k; j++) {
                for (idx_t r = nv - 1; r >= 0; r--) {
                    if (bd->v_betas[r] == 0.0)
                        continue;
                    idx_t len = nt - r - 1;
                    hh_apply(bd->v_vecs[r], bd->v_betas[r],
                             &Vt[(size_t)j * (size_t)nt + (size_t)(r + 1)], len);
                }
            }
            /* Copy Vt (m×k) to U (m×k) */
            memcpy(U, Vt, (size_t)m * (size_t)k * sizeof(double));
            free(Vt);
        }

        return SPARSE_OK;
    }

    /* Non-transposed case (m >= n): form U and V directly from reflectors */

    /* Form U (m×k): apply left Householder reflectors to columns of I_k */
    if (U) {
        memset(U, 0, (size_t)m * (size_t)k * sizeof(double));
        for (idx_t i = 0; i < k; i++)
            U[(size_t)i * (size_t)m + (size_t)i] = 1.0;

        for (idx_t j = 0; j < k; j++) {
            /* Apply reflectors right-to-left: U = H_0 * H_1 * ... * H_{k-1} */
            for (idx_t r = k - 1; r >= 0; r--) {
                if (bd->u_betas[r] == 0.0)
                    continue;
                idx_t len = m - r;
                hh_apply(bd->u_vecs[r], bd->u_betas[r], &U[(size_t)j * (size_t)m + (size_t)r], len);
            }
        }
    }

    /* Form V (n×k): apply right Householder reflectors to columns of I_k */
    if (V) {
        idx_t nv = (k > 1) ? k - 1 : 0;
        memset(V, 0, (size_t)n * (size_t)k * sizeof(double));
        for (idx_t i = 0; i < k; i++)
            V[(size_t)i * (size_t)n + (size_t)i] = 1.0;

        for (idx_t j = 0; j < k; j++) {
            /* Apply reflectors right-to-left */
            for (idx_t r = nv - 1; r >= 0; r--) {
                if (bd->v_betas[r] == 0.0)
                    continue;
                idx_t len = n - r - 1;
                hh_apply(bd->v_vecs[r], bd->v_betas[r], &V[(size_t)j * (size_t)n + (size_t)(r + 1)],
                         len);
            }
        }
    }

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * SVD free
 * ═══════════════════════════════════════════════════════════════════════ */

void sparse_svd_free(sparse_svd_t *svd) {
    if (!svd)
        return;
    free(svd->sigma);
    free(svd->U);
    free(svd->Vt);
    memset(svd, 0, sizeof(*svd));
}

/* ═══════════════════════════════════════════════════════════════════════
 * Full SVD computation (stub — Days 6-9 will complete)
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_svd_compute(const SparseMatrix *A, const sparse_svd_opts_t *opts,
                                sparse_svd_t *svd) {
    if (!svd)
        return SPARSE_ERR_NULL;
    memset(svd, 0, sizeof(*svd));
    if (!A)
        return SPARSE_ERR_NULL;

    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);
    idx_t k = (m < n) ? m : n;

    svd->m = m;
    svd->n = n;
    svd->k = k;

    if (k == 0)
        return SPARSE_OK;

    /* Step 1: Bidiagonalize */
    sparse_bidiag_t bd;
    sparse_err_t err = sparse_bidiag_factor(A, &bd);
    if (err != SPARSE_OK)
        return err;

    /* Copy bidiagonal entries as initial singular values */
    svd->sigma = malloc((size_t)k * sizeof(double));
    if (!svd->sigma) {
        sparse_bidiag_free(&bd);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t i = 0; i < k; i++)
        svd->sigma[i] = fabs(bd.diag[i]);

    /* Extract U and V if requested */
    int compute_uv = opts ? opts->compute_uv : 0;
    int economy = opts ? opts->economy : 0;
    svd->economy = economy;

    if (compute_uv) {
        idx_t u_cols = economy ? k : m;
        idx_t v_cols = economy ? k : n;
        (void)u_cols;
        (void)v_cols;

        /* For now, extract economy U (m×k) and V (n×k) */
        svd->U = calloc((size_t)m * (size_t)k, sizeof(double));
        svd->Vt = calloc((size_t)k * (size_t)n, sizeof(double));
        if (!svd->U || !svd->Vt) {
            sparse_bidiag_free(&bd);
            sparse_svd_free(svd);
            return SPARSE_ERR_ALLOC;
        }

        /* Extract U (m×k) and V (n×k) from Householder reflectors */
        double *V_tmp = calloc((size_t)n * (size_t)k, sizeof(double));
        if (!V_tmp) {
            sparse_bidiag_free(&bd);
            sparse_svd_free(svd);
            return SPARSE_ERR_ALLOC;
        }

        err = sparse_svd_extract_uv(&bd, svd->U, V_tmp);
        if (err != SPARSE_OK) {
            free(V_tmp);
            sparse_bidiag_free(&bd);
            sparse_svd_free(svd);
            return err;
        }

        /* Transpose V (n×k) to get Vt (k×n) in column-major */
        for (idx_t i = 0; i < k; i++)
            for (idx_t j = 0; j < n; j++)
                svd->Vt[(size_t)j * (size_t)k + (size_t)i] =
                    V_tmp[(size_t)i * (size_t)n + (size_t)j];

        free(V_tmp);
    }

    sparse_bidiag_free(&bd);

    /* TODO (Days 6-9): Apply implicit QR SVD iteration to refine singular
     * values from the bidiagonal. For now, sigma = |diag| is approximate. */

    /* Sort singular values descending (and permute U/Vt columns) */
    for (idx_t i = 0; i < k - 1; i++) {
        idx_t best = i;
        for (idx_t j = i + 1; j < k; j++)
            if (svd->sigma[j] > svd->sigma[best])
                best = j;
        if (best != i) {
            double tmp = svd->sigma[i];
            svd->sigma[i] = svd->sigma[best];
            svd->sigma[best] = tmp;
            if (svd->U) {
                for (idx_t r = 0; r < m; r++) {
                    double t = svd->U[(size_t)i * (size_t)m + (size_t)r];
                    svd->U[(size_t)i * (size_t)m + (size_t)r] =
                        svd->U[(size_t)best * (size_t)m + (size_t)r];
                    svd->U[(size_t)best * (size_t)m + (size_t)r] = t;
                }
            }
            if (svd->Vt) {
                for (idx_t c = 0; c < n; c++) {
                    double t = svd->Vt[(size_t)c * (size_t)k + (size_t)i];
                    svd->Vt[(size_t)c * (size_t)k + (size_t)i] =
                        svd->Vt[(size_t)c * (size_t)k + (size_t)best];
                    svd->Vt[(size_t)c * (size_t)k + (size_t)best] = t;
                }
            }
        }
    }

    return SPARSE_OK;
}
