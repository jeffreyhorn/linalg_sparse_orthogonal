#include "sparse_svd.h"
#include "sparse_bidiag.h"
#include "sparse_dense.h"
#include "sparse_matrix.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/** Return nonzero if a*b overflows size_t; store result in *out. */
static int size_mul_overflow(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a)
        return 1;
    *out = a * b;
    return 0;
}

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
static void bidiag_svd_step(double *diag, double *superdiag, idx_t lo, idx_t hi, double *U, idx_t m,
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
        double c, s;
        givens_compute(y, z, &c, &s);

        /* --- Right rotation G_R on columns k, k+1 of B --- */
        /* B_new = B * G_R where G_R = [c s; -s c] */
        if (k > lo)
            superdiag[k - 1] = c * superdiag[k - 1] + s * z; /* row k-1: both bidiag and bulge */

        double dk = diag[k], ek = superdiag[k], dk1 = diag[k + 1];
        diag[k] = c * dk + s * ek;
        superdiag[k] = -s * dk + c * ek;
        double bulge = s * dk1;
        diag[k + 1] = c * dk1;

        /* V = V * G_R: update columns k, k+1 of V */
        if (V) {
            for (idx_t i = 0; i < n; i++) {
                double vk = V[(size_t)k * (size_t)n + (size_t)i];
                double vk1 = V[(size_t)(k + 1) * (size_t)n + (size_t)i];
                V[(size_t)k * (size_t)n + (size_t)i] = c * vk + s * vk1;
                V[(size_t)(k + 1) * (size_t)n + (size_t)i] = -s * vk + c * vk1;
            }
        }

        /* --- Left rotation G_L on rows k, k+1 of B --- */
        /* B_new = G_L^T * B where G_L = [c s; -s c], G_L^T = [c -s; s c] */
        givens_compute(diag[k], bulge, &c, &s);
        diag[k] = c * diag[k] + s * bulge; /* = hypot */

        double ek2 = superdiag[k], dk12 = diag[k + 1];
        superdiag[k] = c * ek2 + s * dk12;
        diag[k + 1] = -s * ek2 + c * dk12;

        /* U = U * G_L: update columns k, k+1 of U */
        if (U) {
            for (idx_t i = 0; i < m; i++) {
                double uk = U[(size_t)k * (size_t)m + (size_t)i];
                double uk1 = U[(size_t)(k + 1) * (size_t)m + (size_t)i];
                U[(size_t)k * (size_t)m + (size_t)i] = c * uk + s * uk1;
                U[(size_t)(k + 1) * (size_t)m + (size_t)i] = -s * uk + c * uk1;
            }
        }

        /* Prepare for next right rotation */
        if (k + 1 < hi) {
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

    /* Compute an absolute tolerance floor based on the matrix norm */
    double bidiag_norm = 0.0;
    for (idx_t i = 0; i < k; i++) {
        double row_sum = fabs(diag[i]);
        if (i < k - 1)
            row_sum += fabs(superdiag[i]);
        if (row_sum > bidiag_norm)
            bidiag_norm = row_sum;
    }
    double abs_tol = tol * bidiag_norm;
    if (abs_tol < 1e-30)
        abs_tol = 1e-30;

    while (hi > 0 && total_iter < max_iter) {
        /* Check for deflation at bottom */
        double off = fabs(superdiag[hi - 1]);
        double dsum = fabs(diag[hi - 1]) + fabs(diag[hi]);
        if (off <= tol * dsum || off < abs_tol) {
            superdiag[hi - 1] = 0.0;
            hi--;
            continue;
        }

        /* Find start of unreduced block */
        idx_t lo = hi - 1;
        while (lo > 0) {
            double off_lo = fabs(superdiag[lo - 1]);
            double ds_lo = fabs(diag[lo - 1]) + fabs(diag[lo]);
            if (off_lo <= tol * ds_lo || off_lo < abs_tol) {
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

        /* For a 2×2 block, solve directly via 2×2 SVD */
        if (hi - lo == 1) {
            double d0 = diag[lo], e0 = superdiag[lo], d1 = diag[hi];

            if (U || V) {
                /* Full 2×2 SVD: B = U_2 * Sigma * V_2^T.
                 * Step 1: Right rotation (Jacobi on B^T*B) to diagonalize B^T*B.
                 * Step 2: Left rotation to zero the (1,0) entry of B*V_2. */
                double t00 = d0 * d0;
                double t01 = d0 * e0;
                double t11 = e0 * e0 + d1 * d1;
                double diff = t00 - t11;
                double cv, sv;
                if (fabs(t01) < 1e-30) {
                    cv = 1.0;
                    sv = 0.0;
                } else {
                    double tau = diff / (2.0 * t01);
                    double t_val =
                        (tau >= 0) ? 1.0 / (tau + hypot(1.0, tau)) : 1.0 / (tau - hypot(1.0, tau));
                    cv = 1.0 / hypot(1.0, t_val);
                    sv = t_val * cv;
                }
                if (V) {
                    for (idx_t i = 0; i < n; i++) {
                        double v0 = V[(size_t)lo * (size_t)n + (size_t)i];
                        double v1 = V[(size_t)hi * (size_t)n + (size_t)i];
                        V[(size_t)lo * (size_t)n + (size_t)i] = cv * v0 + sv * v1;
                        V[(size_t)hi * (size_t)n + (size_t)i] = -sv * v0 + cv * v1;
                    }
                }
                /* B * V_2 gives a quasi-upper-triangular matrix */
                double bv00 = d0 * cv + e0 * sv;
                double bv10 = d1 * sv;
                double bv01 = -d0 * sv + e0 * cv;
                double bv11 = d1 * cv;
                /* Left rotation to zero bv10 */
                double cu, su;
                givens_compute(bv00, bv10, &cu, &su);
                if (U) {
                    for (idx_t i = 0; i < m; i++) {
                        double u0 = U[(size_t)lo * (size_t)m + (size_t)i];
                        double u1 = U[(size_t)hi * (size_t)m + (size_t)i];
                        U[(size_t)lo * (size_t)m + (size_t)i] = cu * u0 + su * u1;
                        U[(size_t)hi * (size_t)m + (size_t)i] = -su * u0 + cu * u1;
                    }
                }
                /* Singular values from the actual rotation results */
                diag[lo] = cu * bv00 + su * bv10;
                diag[hi] = -su * bv01 + cu * bv11;
            } else {
                /* Singular values only: use eigenvalues of B^T*B */
                double t00 = d0 * d0;
                double t01 = d0 * e0;
                double t11 = e0 * e0 + d1 * d1;
                double l1, l2;
                eigen2x2(t00, t01, t11, &l1, &l2);
                diag[lo] = sqrt(fabs(l2));
                diag[hi] = sqrt(fabs(l1));
            }
            superdiag[lo] = 0.0;
            total_iter++;
            continue;
        }

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
 * Full SVD computation
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

    /* Full (non-economy) SVD with UV is not implemented — only economy mode */
    if (compute_uv && !economy) {
        sparse_bidiag_free(&bd);
        return SPARSE_ERR_BADARG;
    }

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
        size_t sz_u, sz_v;
        if (size_mul_overflow((size_t)m, (size_t)k, &sz_u) ||
            size_mul_overflow((size_t)n, (size_t)k, &sz_v) || sz_u > SIZE_MAX / sizeof(double) ||
            sz_v > SIZE_MAX / sizeof(double)) {
            free(bd_diag);
            free(bd_super);
            sparse_bidiag_free(&bd);
            return SPARSE_ERR_ALLOC;
        }
        U_work = calloc(sz_u, sizeof(double)); // NOLINT(clang-analyzer-optin.portability.UnixAPI)
        V_work = calloc(sz_v, sizeof(double)); // NOLINT(clang-analyzer-optin.portability.UnixAPI)
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
        size_t vt_sz;
        if (size_mul_overflow((size_t)k, (size_t)n, &vt_sz) || vt_sz > SIZE_MAX / sizeof(double)) {
            free(V_work);
            sparse_svd_free(svd);
            return SPARSE_ERR_ALLOC;
        }
        svd->Vt = calloc(vt_sz, sizeof(double));
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

/* ═══════════════════════════════════════════════════════════════════════
 * Partial SVD via Lanczos bidiagonalization
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_svd_partial(const SparseMatrix *A, idx_t kk, const sparse_svd_opts_t *opts,
                                sparse_svd_t *svd) {
    if (!svd)
        return SPARSE_ERR_NULL;
    memset(svd, 0, sizeof(*svd));
    if (!A)
        return SPARSE_ERR_NULL;

    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);
    idx_t kmax = (m < n) ? m : n;

    if (kk <= 0 || kk > kmax)
        return SPARSE_ERR_BADARG;

    /* Reject non-identity permutations (same check as sparse_bidiag_factor) */
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

    svd->m = m;
    svd->n = n;
    svd->k = kk;

    /* Use more Lanczos steps than k for better convergence.
     * Clustered spectra (e.g. stiffness matrices) need a larger subspace. */
    idx_t lanczos_k = 2 * kk + 20;
    if (lanczos_k < kk + 30)
        lanczos_k = kk + 30;
    if (lanczos_k > kmax)
        lanczos_k = kmax;

    /* Transpose for A^T * x operations */
    SparseMatrix *At = sparse_transpose(A);
    if (!At)
        return SPARSE_ERR_ALLOC;

    /* Allocate Lanczos vectors: P (m x lanczos_k) and Q (n x (lanczos_k+1)) */
    size_t sz_p, sz_q;
    if (size_mul_overflow((size_t)m, (size_t)lanczos_k, &sz_p) ||
        size_mul_overflow((size_t)n, (size_t)(lanczos_k + 1), &sz_q) ||
        sz_p > SIZE_MAX / sizeof(double) || sz_q > SIZE_MAX / sizeof(double)) {
        sparse_free(At);
        return SPARSE_ERR_ALLOC;
    }
    double *P = calloc(sz_p, sizeof(double)); // NOLINT(clang-analyzer-optin.portability.UnixAPI)
    double *Q = calloc(sz_q, sizeof(double)); // NOLINT(clang-analyzer-optin.portability.UnixAPI)
    double *alpha = calloc((size_t)lanczos_k, sizeof(double));
    double *beta = calloc((size_t)(lanczos_k + 1), sizeof(double));

    if (!P || !Q || !alpha || !beta) {
        free(P);
        free(Q);
        free(alpha);
        free(beta);
        sparse_free(At);
        return SPARSE_ERR_ALLOC;
    }

    /* Initialize q_0 = [1/sqrt(n), ...] (unit vector) */
    {
        double inv_sqrt_n = 1.0 / sqrt((double)n);
        for (idx_t i = 0; i < n; i++)
            Q[i] = inv_sqrt_n;
    }

    beta[0] = 0.0;

    for (idx_t j = 0; j < lanczos_k; j++) {
        double *qj = &Q[(size_t)j * (size_t)n];
        double *pj = &P[(size_t)j * (size_t)m];

        /* p_j = A * q_j */
        {
            sparse_err_t mv_err = sparse_matvec(A, qj, pj);
            if (mv_err != SPARSE_OK) {
                free(P);
                free(Q);
                free(alpha);
                free(beta);
                sparse_free(At);
                return mv_err;
            }
        }

        /* p_j = p_j - beta_j * p_{j-1} */
        if (j > 0) {
            double *pjm1 = &P[(size_t)(j - 1) * (size_t)m];
            for (idx_t i = 0; i < m; i++)
                pj[i] -= beta[j] * pjm1[i];
        }

        /* Reorthogonalize p_j against p_0..p_{j-1} */
        for (idx_t r = 0; r < j; r++) {
            double *pr = &P[(size_t)r * (size_t)m];
            double dot = 0.0;
            for (idx_t i = 0; i < m; i++)
                dot += pr[i] * pj[i];
            for (idx_t i = 0; i < m; i++)
                pj[i] -= dot * pr[i];
        }

        /* alpha_j = ||p_j|| */
        double anorm = 0.0;
        for (idx_t i = 0; i < m; i++)
            anorm += pj[i] * pj[i];
        anorm = sqrt(anorm);
        alpha[j] = anorm;

        if (anorm > 1e-30) {
            double inv = 1.0 / anorm;
            for (idx_t i = 0; i < m; i++)
                pj[i] *= inv;
        }

        /* r = A^T * p_j - alpha_j * q_j */
        double *qj1 = &Q[(size_t)(j + 1) * (size_t)n];
        {
            sparse_err_t mv_err = sparse_matvec(At, pj, qj1);
            if (mv_err != SPARSE_OK) {
                free(P);
                free(Q);
                free(alpha);
                free(beta);
                sparse_free(At);
                return mv_err;
            }
        }
        for (idx_t i = 0; i < n; i++)
            qj1[i] -= alpha[j] * qj[i];

        /* Reorthogonalize q_{j+1} against q_0..q_j */
        for (idx_t r = 0; r <= j; r++) {
            double *qr = &Q[(size_t)r * (size_t)n];
            double dot = 0.0;
            for (idx_t i = 0; i < n; i++)
                dot += qr[i] * qj1[i];
            for (idx_t i = 0; i < n; i++)
                qj1[i] -= dot * qr[i];
        }

        /* beta_{j+1} = ||q_{j+1}|| */
        double bnorm = 0.0;
        for (idx_t i = 0; i < n; i++)
            bnorm += qj1[i] * qj1[i];
        bnorm = sqrt(bnorm);
        beta[j + 1] = bnorm;

        if (j + 1 < lanczos_k && bnorm > 1e-30) {
            double inv = 1.0 / bnorm;
            for (idx_t i = 0; i < n; i++)
                qj1[i] *= inv;
        }
    }

    sparse_free(At);
    free(P);
    free(Q);

    /* Now we have a lanczos_k x lanczos_k bidiagonal with
     * diag=alpha, superdiag=beta[1..lanczos_k-1] */
    double *bd_super = NULL;
    if (lanczos_k > 1) {
        bd_super = malloc((size_t)(lanczos_k - 1) * sizeof(double));
        if (!bd_super) {
            free(alpha);
            free(beta);
            return SPARSE_ERR_ALLOC;
        }
        for (idx_t i = 0; i < lanczos_k - 1; i++)
            bd_super[i] = beta[i + 1];
    }
    free(beta);

    /* Run bidiagonal SVD iteration on the small lanczos_k x lanczos_k bidiag */
    idx_t max_iter_val = opts ? opts->max_iter : 0;
    double tol_val = opts ? opts->tol : 0.0;
    sparse_err_t err =
        bidiag_svd_iterate(alpha, bd_super, lanczos_k, NULL, 0, NULL, 0, max_iter_val, tol_val);
    free(bd_super);

    if (err != SPARSE_OK) {
        free(alpha);
        return err;
    }

    /* Make non-negative and sort descending */
    for (idx_t i = 0; i < lanczos_k; i++)
        if (alpha[i] < 0.0)
            alpha[i] = -alpha[i];

    for (idx_t i = 0; i < lanczos_k - 1; i++) {
        idx_t best = i;
        for (idx_t j = i + 1; j < lanczos_k; j++)
            if (alpha[j] > alpha[best])
                best = j;
        if (best != i) {
            double tmp = alpha[i];
            alpha[i] = alpha[best];
            alpha[best] = tmp;
        }
    }

    /* Keep only the top kk singular values */
    double *sigma = malloc((size_t)kk * sizeof(double));
    if (!sigma) {
        free(alpha);
        return SPARSE_ERR_ALLOC;
    }
    memcpy(sigma, alpha, (size_t)kk * sizeof(double));
    free(alpha);

    svd->sigma = sigma;
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * SVD applications: rank, pseudoinverse, low-rank approximation
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_svd_rank(const SparseMatrix *A, double tol, idx_t *rank) {
    if (!A || !rank)
        return SPARSE_ERR_NULL;

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    if (err != SPARSE_OK)
        return err;

    /* Default tolerance: eps * max(m,n) * sigma_max */
    if (tol <= 0.0) {
        idx_t maxdim = (svd.m > svd.n) ? svd.m : svd.n;
        tol = 2.2204460492503131e-16 * (double)maxdim * svd.sigma[0];
    }

    idx_t r = 0;
    for (idx_t i = 0; i < svd.k; i++) {
        if (svd.sigma[i] > tol)
            r++;
    }

    *rank = r;
    sparse_svd_free(&svd);
    return SPARSE_OK;
}

sparse_err_t sparse_pinv(const SparseMatrix *A, double tol, double **pinv) {
    if (!pinv)
        return SPARSE_ERR_NULL;
    *pinv = NULL;
    if (!A)
        return SPARSE_ERR_NULL;

    /* Full SVD with U and Vt */
    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1};
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, &opts, &svd);
    if (err != SPARSE_OK)
        return err;

    idx_t m = svd.m, n = svd.n, k = svd.k;

    /* Default tolerance */
    if (tol <= 0.0) {
        idx_t maxdim = (m > n) ? m : n;
        tol = 2.2204460492503131e-16 * (double)maxdim * svd.sigma[0];
    }

    /* A^+ = V * Sigma^+ * U^T, where A^+ is n×m.
     * U is m×k (column-major), Vt is k×n (column-major).
     * V = Vt^T. So A^+ = V * Sigma^+ * U^T.
     *
     * Compute column j of A^+:
     *   A^+[:,j] = sum_{i: sigma_i > tol} (1/sigma_i) * V[:,i] * U[j,i]
     *            = sum_{i} (1/sigma_i) * Vt^T[:,i] * U[j,i]
     */
    size_t nm;
    if (size_mul_overflow((size_t)n, (size_t)m, &nm) || nm > SIZE_MAX / sizeof(double)) {
        sparse_svd_free(&svd);
        return SPARSE_ERR_ALLOC;
    }
    double *result = calloc(nm, sizeof(double)); // NOLINT(clang-analyzer-optin.portability.UnixAPI)
    if (!result) {
        sparse_svd_free(&svd);
        return SPARSE_ERR_ALLOC;
    }

    /* result is n×m column-major: result[col * n + row] */
    for (idx_t i = 0; i < k; i++) {
        if (svd.sigma[i] <= tol)
            continue;
        double inv_sigma = 1.0 / svd.sigma[i];
        /* Outer product: (1/sigma_i) * V[:,i] * U[:,i]^T
         * V[:,i] is column i of V = row i of Vt.
         * In Vt (k×n column-major): Vt[col_j * k + i] = V[i, col_j]... wait.
         * Vt is stored as k×n column-major: element (row_r, col_c) = Vt[col_c * k + row_r].
         * So Vt[row=i, col=j] = Vt[j * k + i] = V^T[i,j] = V[j,i].
         * Thus V[j,i] = Vt[j * k + i]. So column i of V has V[j,i] = Vt[j*k + i] for j=0..n-1.
         *
         * U is m×k column-major: U[col_i * m + row_r] = U[r, i].
         *
         * A^+[row_r, col_c] += inv_sigma * V[row_r, i] * U[col_c, i]
         * result[col_c * n + row_r] += inv_sigma * Vt[row_r * k + i] * U[i * m + col_c]
         */
        for (idx_t c = 0; c < m; c++) {
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            double u_ci = svd.U[(size_t)i * (size_t)m + (size_t)c] * inv_sigma;
            for (idx_t r = 0; r < n; r++) {
                // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
                result[(size_t)c * (size_t)n + (size_t)r] +=
                    svd.Vt[(size_t)r * (size_t)k + (size_t)i] * u_ci;
            }
        }
    }

    *pinv = result;
    sparse_svd_free(&svd);
    return SPARSE_OK;
}

sparse_err_t sparse_svd_lowrank(const SparseMatrix *A, idx_t rank_k, double **lowrank) {
    if (!lowrank)
        return SPARSE_ERR_NULL;
    *lowrank = NULL;
    if (!A)
        return SPARSE_ERR_NULL;

    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);
    idx_t kmax = (m < n) ? m : n;

    if (rank_k <= 0 || rank_k > kmax)
        return SPARSE_ERR_BADARG;

    /* Full SVD with U and Vt */
    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1};
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, &opts, &svd);
    if (err != SPARSE_OK)
        return err;

    /* A_k = sum_{i=0}^{rank_k-1} sigma_i * U[:,i] * Vt[i,:]
     * Result is m×n column-major. */
    size_t mn;
    if (size_mul_overflow((size_t)m, (size_t)n, &mn) || mn > SIZE_MAX / sizeof(double)) {
        sparse_svd_free(&svd);
        return SPARSE_ERR_ALLOC;
    }
    double *result = calloc(mn, sizeof(double)); // NOLINT(clang-analyzer-optin.portability.UnixAPI)
    if (!result) {
        sparse_svd_free(&svd);
        return SPARSE_ERR_ALLOC;
    }

    idx_t k = svd.k;
    double *U_data = svd.U;   /* m×k col-major */
    double *Vt_data = svd.Vt; /* k×n col-major */
    for (idx_t i = 0; i < rank_k && i < k; i++) {
        double si = svd.sigma[i];
        if (si == 0.0)
            break;
        /* Outer product: sigma_i * U[:,i] * Vt[i,:]
         * U[:,i] = U_data[i*m .. i*m+m-1]
         * Vt[i,:] has Vt[row=i, col=j] = Vt_data[j*k + i] for j=0..n-1 */
        for (idx_t j = 0; j < n; j++) {
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            double vt_ij = Vt_data[(size_t)j * (size_t)k + (size_t)i] * si;
            for (idx_t r = 0; r < m; r++) {
                /* NOLINTNEXTLINE(clang-analyzer-security.ArrayBound) */
                double u_ir = U_data[(size_t)i * (size_t)m + (size_t)r];
                result[(size_t)j * (size_t)m + (size_t)r] += u_ir * vt_ij;
            }
        }
    }

    *lowrank = result;
    sparse_svd_free(&svd);
    return SPARSE_OK;
}
