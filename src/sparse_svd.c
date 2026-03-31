#include "sparse_svd.h"
#include "sparse_bidiag.h"
#include "sparse_dense.h"
#include "sparse_matrix.h"
#include <math.h>
#include <stdint.h>
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
 * Implicit QR SVD step on bidiagonal (Golub-Kahan)
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * One implicit QR SVD step on the unreduced bidiagonal block
 * diag[lo..hi], superdiag[lo..hi-1] using a Wilkinson-like shift
 * derived from the trailing 2×2 of B^T*B.
 *
 * Optionally accumulates left rotations into U (m×k col-major)
 * and right rotations into V (n×k col-major).
 */
void bidiag_svd_step(double *diag, double *superdiag, idx_t lo, idx_t hi, double *U, idx_t m,
                     double *V, idx_t n) {
    /* Compute shift from trailing 2×2 of B^T*B.
     * T = B^T*B trailing 2×2:
     *   T(0,0) = diag[hi-1]^2 + superdiag[hi-2]^2  (if hi-2 >= lo, else diag[hi-1]^2)
     *   T(0,1) = diag[hi-1]*superdiag[hi-1]
     *   T(1,0) = T(0,1)
     *   T(1,1) = diag[hi]^2 + superdiag[hi-1]^2 */
    double d_hm1 = diag[hi - 1];
    double d_h = diag[hi];
    double e_hm1 = superdiag[hi - 1];
    double e_hm2 = (hi - 1 > lo) ? superdiag[hi - 2] : 0.0;

    double t00 = d_hm1 * d_hm1 + e_hm2 * e_hm2;
    double t01 = d_hm1 * e_hm1;
    double t11 = d_h * d_h + e_hm1 * e_hm1;

    /* Wilkinson shift: eigenvalue of trailing 2×2 closer to t11 */
    double l1, l2;
    eigen2x2(t00, t01, t11, &l1, &l2);
    double shift = (fabs(l1 - t11) < fabs(l2 - t11)) ? l1 : l2;

    /* Initial right rotation: zero the bulge in (B^T*B - shift*I) first column.
     * The first column of B^T*B - shift*I has entries:
     *   [diag[lo]^2 - shift, diag[lo]*superdiag[lo], 0, ...] */
    double y = diag[lo] * diag[lo] - shift;
    double z = diag[lo] * superdiag[lo];

    for (idx_t k = lo; k < hi; k++) {
        /* Right Givens: zero z using y → rotates columns k, k+1 */
        double c, s;
        givens_compute(y, z, &c, &s);

        /* Apply right rotation to B (affects columns k, k+1) */
        /* Row k: [diag[k], superdiag[k]] → rotated */
        /* Row k-1 (if k>lo): [0, superdiag[k-1]] already set by previous left rotation */
        if (k > lo) {
            double tmp = superdiag[k - 1];
            superdiag[k - 1] = c * tmp + s * 0.0; /* bulge was zeroed */
            /* No entry at (k-1, k+1) after rotation since B is bidiag */
        }

        double dk = diag[k];
        double ek = superdiag[k];
        diag[k] = c * dk + s * ek;
        superdiag[k] = -s * dk + c * ek;

        /* Bulge appears at (k+1, k): z_new = s * diag[k+1] */
        double dk1 = diag[k + 1];
        double bulge = s * dk1;
        diag[k + 1] = c * dk1;

        /* Accumulate right rotation into V: V_new = V * G */
        if (V) {
            givens_apply_left(c, s, &V[(size_t)k * (size_t)n], &V[(size_t)(k + 1) * (size_t)n], n);
        }

        /* Left Givens: zero bulge at (k+1, k) */
        y = diag[k];
        z = bulge;
        givens_compute(y, z, &c, &s);

        diag[k] = c * y + s * z;
        /* superdiag[k] is updated: mix with the entry at (k, k+1) */
        double ek_new = superdiag[k];
        superdiag[k] = c * ek_new + s * diag[k + 1];
        diag[k + 1] = -s * ek_new + c * diag[k + 1];

        /* Accumulate left rotation into U: U_new = U * G */
        if (U) {
            givens_apply_left(c, s, &U[(size_t)k * (size_t)m], &U[(size_t)(k + 1) * (size_t)m], m);
        }

        /* Prepare for next iteration */
        if (k + 1 < hi) {
            /* New bulge at (k, k+2): comes from superdiag[k+1] */
            y = superdiag[k];
            z = s * superdiag[k + 1];
            superdiag[k + 1] = c * superdiag[k + 1];
        }
    }
}

/**
 * Full bidiagonal SVD: iterate QR steps until all superdiagonal entries
 * converge to zero. Returns singular values in diag (not sorted).
 *
 * @param diag      Diagonal entries (length k). Modified in-place → singular values.
 * @param superdiag Superdiagonal entries (length k-1). Destroyed.
 * @param k         Bidiagonal dimension.
 * @param U         Optional left singular vectors (m×k col-major). Modified in-place.
 * @param m         Row dimension for U.
 * @param V         Optional right singular vectors (n×k col-major). Modified in-place.
 * @param n         Column dimension for V.
 * @param max_iter  Maximum QR iterations (0 → default 30*k).
 * @param tol       Convergence tolerance (0 → default 1e-14).
 * @return SPARSE_OK on success, SPARSE_ERR_NOT_CONVERGED if max_iter reached.
 */
sparse_err_t bidiag_svd_iterate(double *diag, double *superdiag, idx_t k, double *U, idx_t m,
                                double *V, idx_t n, idx_t max_iter, double tol) {
    if (k <= 1) {
        if (k == 1 && diag[0] < 0.0) {
            diag[0] = -diag[0];
            /* Flip sign in U column 0 */
            if (U) {
                for (idx_t i = 0; i < m; i++)
                    U[i] = -U[i];
            }
        }
        return SPARSE_OK;
    }

    if (max_iter <= 0) {
        int64_t def = (int64_t)30 * (int64_t)k;
        max_iter = (def > INT32_MAX) ? INT32_MAX : (idx_t)def;
    }
    if (tol <= 0.0)
        tol = 1e-14;

    idx_t total_iter = 0;
    idx_t hi = k - 1;

    while (hi > 0 && total_iter < max_iter) {
        /* Check for deflation at bottom */
        double off = fabs(superdiag[hi - 1]);
        double dsum = fabs(diag[hi - 1]) + fabs(diag[hi]);
        if (off <= tol * dsum || off < 1e-30) {
            superdiag[hi - 1] = 0.0;
            hi--;
            continue;
        }

        /* Find start of unreduced block */
        idx_t lo = hi - 1;
        while (lo > 0) {
            double off_lo = fabs(superdiag[lo - 1]);
            double ds_lo = fabs(diag[lo - 1]) + fabs(diag[lo]);
            if (off_lo <= tol * ds_lo || off_lo < 1e-30) {
                superdiag[lo - 1] = 0.0;
                break;
            }
            lo--;
        }

        /* Check for zero diagonal entries — need special handling */
        int has_zero_diag = 0;
        for (idx_t i = lo; i <= hi; i++) {
            if (fabs(diag[i]) < 1e-30) {
                has_zero_diag = 1;
                /* Zero out the row by rotating superdiag into adjacent entries */
                if (i < hi && fabs(superdiag[i]) > 1e-30) {
                    double c, s;
                    givens_compute(diag[i + 1], superdiag[i], &c, &s);
                    diag[i + 1] = c * diag[i + 1] + s * superdiag[i];
                    superdiag[i] = 0.0;
                    if (U)
                        givens_apply_left(c, s, &U[(size_t)(i + 1) * (size_t)m],
                                          &U[(size_t)i * (size_t)m], m);
                }
                break;
            }
        }
        if (has_zero_diag)
            continue; /* re-check deflation */

        /* One QR step on block [lo..hi] */
        bidiag_svd_step(diag, superdiag, lo, hi, U, m, V, n);
        total_iter++;
    }

    if (hi > 0)
        return SPARSE_ERR_NOT_CONVERGED;

    /* Make all singular values non-negative */
    for (idx_t i = 0; i < k; i++) {
        if (diag[i] < 0.0) {
            diag[i] = -diag[i];
            if (U) {
                for (idx_t r = 0; r < m; r++)
                    U[(size_t)i * (size_t)m + (size_t)r] = -U[(size_t)i * (size_t)m + (size_t)r];
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

    /* Extract U and V if requested, then run QR iteration on bidiag */
    int compute_uv = opts ? opts->compute_uv : 0;
    int economy = opts ? opts->economy : 0;
    idx_t svd_max_iter = opts ? opts->max_iter : 0;
    double svd_tol = opts ? opts->tol : 0.0;
    svd->economy = economy;

    /* Copy bidiag arrays (QR iteration modifies them in-place) */
    double *bd_diag = malloc((size_t)k * sizeof(double));
    double *bd_super = (k > 1) ? malloc((size_t)(k - 1) * sizeof(double)) : NULL;
    if (!bd_diag || (k > 1 && !bd_super)) {
        free(bd_diag);
        free(bd_super);
        sparse_bidiag_free(&bd);
        return SPARSE_ERR_ALLOC;
    }
    memcpy(bd_diag, bd.diag, (size_t)k * sizeof(double));
    if (k > 1)
        memcpy(bd_super, bd.superdiag, (size_t)(k - 1) * sizeof(double));

    double *U_work = NULL;
    double *V_work = NULL;

    if (compute_uv) {
        /* Extract economy U (m×k) and V (n×k) from Householder reflectors */
        U_work = calloc((size_t)m * (size_t)k, sizeof(double));
        V_work = calloc((size_t)n * (size_t)k, sizeof(double));
        if (!U_work || !V_work) {
            free(bd_diag);
            free(bd_super);
            free(U_work);
            free(V_work);
            sparse_bidiag_free(&bd);
            return SPARSE_ERR_ALLOC;
        }
        err = sparse_svd_extract_uv(&bd, U_work, V_work);
        if (err != SPARSE_OK) {
            free(bd_diag);
            free(bd_super);
            free(U_work);
            free(V_work);
            sparse_bidiag_free(&bd);
            return err;
        }
    }

    sparse_bidiag_free(&bd);

    /* Run implicit QR SVD iteration on the bidiagonal */
    err = bidiag_svd_iterate(bd_diag, bd_super, k, U_work, m, V_work, n, svd_max_iter, svd_tol);
    free(bd_super);

    if (err != SPARSE_OK) {
        free(bd_diag);
        free(U_work);
        free(V_work);
        return err;
    }

    /* Store singular values */
    svd->sigma = bd_diag;

    /* Store U and Vt */
    if (compute_uv) {
        svd->U = U_work;
        /* Transpose V (n×k) to get Vt (k×n) */
        svd->Vt = calloc((size_t)k * (size_t)n, sizeof(double));
        if (!svd->Vt) {
            free(V_work);
            sparse_svd_free(svd);
            return SPARSE_ERR_ALLOC;
        }
        for (idx_t i = 0; i < k; i++)
            for (idx_t j = 0; j < n; j++)
                svd->Vt[(size_t)j * (size_t)k + (size_t)i] =
                    V_work[(size_t)i * (size_t)n + (size_t)j];
        free(V_work);
    }

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
