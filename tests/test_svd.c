#include "sparse_bidiag.h"
#include "sparse_dense.h"
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_svd.h"
#include "sparse_svd_internal.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Compute ||A - U*B*V^T|| using explicit U (m×k), V (n×k), and bidiag B.
 * B is k×k upper bidiagonal (diag + superdiag).
 */
static double gk_reconstruction_error(const SparseMatrix *A, const double *U, const double *V,
                                      const double *diag, const double *superdiag, idx_t m, idx_t n,
                                      idx_t k) {
    /* Compute U*B: (m×k) * (k×k upper bidiagonal) = m×k */
    double *UB = calloc((size_t)m * (size_t)k, sizeof(double));
    if (!UB)
        return INFINITY;
    for (idx_t j = 0; j < k; j++) {
        /* B(j,j) = diag[j] */
        for (idx_t i = 0; i < m; i++)
            UB[(size_t)j * (size_t)m + (size_t)i] += U[(size_t)j * (size_t)m + (size_t)i] * diag[j];
        /* B(j, j+1) = superdiag[j] if j < k-1 */
        if (j < k - 1) {
            for (idx_t i = 0; i < m; i++)
                UB[(size_t)(j + 1) * (size_t)m + (size_t)i] +=
                    U[(size_t)j * (size_t)m + (size_t)i] * superdiag[j];
        }
    }

    /* Now compute (U*B)*V^T: m×k * k×n = m×n */
    /* Result[i][j] = sum_p UB[i][p] * V[j][p] (V is n×k col-major) */
    double maxerr = 0.0;
    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = 0; j < n; j++) {
            double val = 0.0;
            for (idx_t p = 0; p < k; p++)
                val += UB[(size_t)p * (size_t)m + (size_t)i] * V[(size_t)p * (size_t)n + (size_t)j];
            double a_val = sparse_get_phys(A, i, j);
            double e = fabs(a_val - val);
            if (e > maxerr)
                maxerr = e;
        }
    }

    free(UB);
    return maxerr;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Golub-Kahan extraction tests (Sprint 8 Day 4)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Extract U, V from 3×3 bidiag and verify reconstruction */
static void test_gk_extract_3x3(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 0, 4.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 1, 2, 6.0);
    sparse_insert(A, 2, 0, 7.0);
    sparse_insert(A, 2, 1, 8.0);
    sparse_insert(A, 2, 2, 9.0);

    sparse_bidiag_t bd;
    sparse_err_t err = sparse_bidiag_factor(A, &bd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t k = 3;
    double *U = calloc(3 * 3, sizeof(double));
    double *V = calloc(3 * 3, sizeof(double));
    ASSERT_NOT_NULL(U);
    ASSERT_NOT_NULL(V);
    if (!U || !V) {
        free(U);
        free(V);
        sparse_bidiag_free(&bd);
        sparse_free(A);
        return;
    }

    ASSERT_ERR(sparse_svd_extract_uv(&bd, U, V), SPARSE_OK);

    double recon = gk_reconstruction_error(A, U, V, bd.diag, bd.superdiag, 3, 3, k);
    printf("    GK 3x3: recon=%.3e\n", recon);
    ASSERT_TRUE(recon < 1e-10);

    free(U);
    free(V);
    sparse_bidiag_free(&bd);
    sparse_free(A);
}

/* Tall 10×5 */
static void test_gk_extract_tall(void) {
    idx_t m = 10, nc = 5;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++)
            if (i == j || (i + j) % 3 == 0)
                sparse_insert(A, i, j, (double)(i * nc + j + 1));

    sparse_bidiag_t bd;
    sparse_err_t err = sparse_bidiag_factor(A, &bd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t k = nc;
    double *U = calloc((size_t)m * (size_t)k, sizeof(double));
    double *V = calloc((size_t)nc * (size_t)k, sizeof(double));
    ASSERT_NOT_NULL(U);
    ASSERT_NOT_NULL(V);
    if (!U || !V) {
        free(U);
        free(V);
        sparse_bidiag_free(&bd);
        sparse_free(A);
        return;
    }

    ASSERT_ERR(sparse_svd_extract_uv(&bd, U, V), SPARSE_OK);

    double recon = gk_reconstruction_error(A, U, V, bd.diag, bd.superdiag, m, nc, k);
    printf("    GK 10x5: recon=%.3e\n", recon);
    ASSERT_TRUE(recon < 1e-10);

    free(U);
    free(V);
    sparse_bidiag_free(&bd);
    sparse_free(A);
}

/* Wide 5×10 (uses transposed bidiag) */
static void test_gk_extract_wide(void) {
    idx_t m = 5, nc = 10;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++)
            if (i == j || j == i + 5)
                sparse_insert(A, i, j, (double)(i + j + 1));

    sparse_bidiag_t bd;
    sparse_err_t err = sparse_bidiag_factor(A, &bd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t k = m; /* min(5, 10) = 5 */
    double *U = calloc((size_t)m * (size_t)k, sizeof(double));
    double *V = calloc((size_t)nc * (size_t)k, sizeof(double));
    ASSERT_NOT_NULL(U);
    ASSERT_NOT_NULL(V);
    if (!U || !V) {
        free(U);
        free(V);
        sparse_bidiag_free(&bd);
        sparse_free(A);
        return;
    }

    ASSERT_ERR(sparse_svd_extract_uv(&bd, U, V), SPARSE_OK);

    if (!bd.transposed) {
        double recon = gk_reconstruction_error(A, U, V, bd.diag, bd.superdiag, m, nc, k);
        printf("    GK 5x10: recon=%.3e\n", recon);
        ASSERT_TRUE(recon < 1e-10);
    } else {
        printf("    GK 5x10 (wide, transposed bidiag): skipping recon check\n");
    }

    free(U);
    free(V);
    sparse_bidiag_free(&bd);
    sparse_free(A);
}

/* Basic SVD compute (singular values only) */
static void test_svd_basic_sigma(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Diagonal matrix: singular values = |diagonal| */
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 1, 1, -5.0);
    sparse_insert(A, 2, 2, 1.0);

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(svd.k, 3);
    ASSERT_NOT_NULL(svd.sigma);
    /* Descending order: 5, 3, 1 */
    printf("    SVD sigma: %.3f, %.3f, %.3f\n", svd.sigma[0], svd.sigma[1], svd.sigma[2]);
    ASSERT_NEAR(svd.sigma[0], 5.0, 1e-10);
    ASSERT_NEAR(svd.sigma[1], 3.0, 1e-10);
    ASSERT_NEAR(svd.sigma[2], 1.0, 1e-10);
    ASSERT_TRUE(svd.U == NULL); /* compute_uv=0 by default */

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Golub-Kahan validation tests (Sprint 8 Day 5)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: check U^T*U ≈ I_k for m×k column-major U */
static double orthogonality_error(const double *Q, idx_t rows, idx_t cols) {
    double maxerr = 0.0;
    for (idx_t i = 0; i < cols; i++) {
        for (idx_t j = 0; j < cols; j++) {
            double dot = 0.0;
            for (idx_t p = 0; p < rows; p++)
                dot += Q[(size_t)i * (size_t)rows + (size_t)p] *
                       Q[(size_t)j * (size_t)rows + (size_t)p];
            double expected = (i == j) ? 1.0 : 0.0;
            double e = fabs(dot - expected);
            if (e > maxerr)
                maxerr = e;
        }
    }
    return maxerr;
}

/* Helper: extract U/V and check reconstruction + orthogonality */
static void validate_gk(const SparseMatrix *A, const char *name) {
    idx_t m = sparse_rows(A);
    idx_t n = sparse_cols(A);
    idx_t k = (m < n) ? m : n;

    sparse_bidiag_t bd;
    sparse_err_t err = sparse_bidiag_factor(A, &bd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK)
        return;

    double *U = calloc((size_t)m * (size_t)k, sizeof(double));
    double *V = calloc((size_t)n * (size_t)k, sizeof(double));
    ASSERT_NOT_NULL(U);
    ASSERT_NOT_NULL(V);
    if (!U || !V) {
        free(U);
        free(V);
        sparse_bidiag_free(&bd);
        return;
    }

    ASSERT_ERR(sparse_svd_extract_uv(&bd, U, V), SPARSE_OK);

    /* For transposed (wide) matrices, bd.diag/superdiag are B_t (A^T's bidiag).
     * gk_reconstruction_error assumes upper-bidiagonal for A, so only enforce
     * reconstruction for the non-transposed case. U/V orthogonality is checked
     * either way. End-to-end wide SVD reconstruction is validated by
     * test_svd_wide_5x10_uv. */
    double recon = NAN;
    if (!bd.transposed)
        recon = gk_reconstruction_error(A, U, V, bd.diag, bd.superdiag, m, n, k);
    double u_orth = orthogonality_error(U, m, k);
    double v_orth = orthogonality_error(V, n, k);

    printf("    GK %s: recon=%.3e, U_orth=%.3e, V_orth=%.3e\n", name, recon, u_orth, v_orth);
    if (!bd.transposed)
        ASSERT_TRUE(recon < 1e-10);
    ASSERT_TRUE(u_orth < 1e-10);
    ASSERT_TRUE(v_orth < 1e-10);

    free(U);
    free(V);
    sparse_bidiag_free(&bd);
}

/* Square 5×5 with orthogonality check */
static void test_gk_square_5x5(void) {
    SparseMatrix *A = sparse_create(5, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 2, 3, 1.0);
    sparse_insert(A, 3, 2, 1.0);
    sparse_insert(A, 3, 3, 5.0);
    sparse_insert(A, 3, 4, 1.0);
    sparse_insert(A, 4, 3, 1.0);
    sparse_insert(A, 4, 4, 6.0);
    validate_gk(A, "5x5");
    sparse_free(A);
}

/* Tall 10×5 with orthogonality */
static void test_gk_tall_ortho(void) {
    idx_t m = 10, nc = 5;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++)
            if (i == j || (i + j) % 3 == 0)
                sparse_insert(A, i, j, (double)(i * nc + j + 1));
    validate_gk(A, "10x5");
    sparse_free(A);
}

/* Wide 5×10 with orthogonality */
static void test_gk_wide_ortho(void) {
    idx_t m = 5, nc = 10;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++)
            if (i == j || j == i + 5)
                sparse_insert(A, i, j, (double)(i + j + 1));
    validate_gk(A, "5x10");
    sparse_free(A);
}

/* Rank-deficient 6×4 (col1 = col0) */
static void test_gk_rank_deficient(void) {
    SparseMatrix *A = sparse_create(6, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 6; i++) {
        sparse_insert(A, i, 0, (double)(i + 1));
        sparse_insert(A, i, 1, (double)(i + 1)); /* duplicate */
        sparse_insert(A, i, 2, (double)(i * 2 + 1));
        sparse_insert(A, i, 3, (double)(i + 3));
    }
    validate_gk(A, "rank-def 6x4");
    sparse_free(A);
}

/* nos4 (100×100 SPD) */
static void test_gk_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    validate_gk(A, "nos4");
    sparse_free(A);
}

/* west0067 (67×67 unsymmetric) */
static void test_gk_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    validate_gk(A, "west0067");
    sparse_free(A);
}

/* 1×1 trivial */
static void test_gk_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 7.0);

    sparse_bidiag_t bd;
    sparse_err_t err = sparse_bidiag_factor(A, &bd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double U_val = 0.0, V_val = 0.0;
    ASSERT_ERR(sparse_svd_extract_uv(&bd, &U_val, &V_val), SPARSE_OK);

    printf("    GK 1x1: U=%.3f, V=%.3f, diag=%.3f\n", U_val, V_val, bd.diag[0]);
    /* U*diag*V^T should equal 7 */
    ASSERT_NEAR(U_val * bd.diag[0] * V_val, 7.0, 1e-10);

    sparse_bidiag_free(&bd);
    sparse_free(A);
}

/* SVD compute with U/V */
static void test_svd_with_uv(void) {
    SparseMatrix *A = sparse_create(4, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 2, 5.0);
    sparse_insert(A, 2, 2, 6.0);
    sparse_insert(A, 3, 0, 1.0);
    sparse_insert(A, 3, 2, 2.0);

    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1};
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, &opts, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(svd.k, 3);
    ASSERT_NOT_NULL(svd.sigma);
    ASSERT_NOT_NULL(svd.U);
    ASSERT_NOT_NULL(svd.Vt);
    if (!svd.U || !svd.Vt) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }

    /* Verify U*diag(sigma)*Vt ≈ A */
    double maxerr = 0.0;
    for (idx_t i = 0; i < 4; i++) {
        for (idx_t j = 0; j < 3; j++) {
            double val = 0.0;
            for (idx_t p = 0; p < 3; p++)
                val += svd.U[(size_t)p * 4 + (size_t)i] * svd.sigma[p] *
                       svd.Vt[(size_t)j * 3 + (size_t)p];
            double e = fabs(sparse_get_phys(A, i, j) - val);
            if (e > maxerr)
                maxerr = e;
        }
    }
    printf("    SVD 4x3 with UV: recon=%.3e, sigma=[%.3f, %.3f, %.3f]\n", maxerr, svd.sigma[0],
           svd.sigma[1], svd.sigma[2]);
    /* Reconstruction should be tight now that bulge chase and 2x2 SVD are fixed */
    ASSERT_TRUE(maxerr < 1e-10);

    /* Singular values positive and descending */
    ASSERT_TRUE(svd.sigma[0] >= svd.sigma[1]);
    ASSERT_TRUE(svd.sigma[1] >= svd.sigma[2]);
    ASSERT_TRUE(svd.sigma[2] >= 0.0);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Bidiagonal SVD iteration tests (Sprint 8 Day 6)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Diagonal bidiagonal: already converged, singular values = |diag| */
static void test_bidiag_svd_diagonal(void) {
    double diag[] = {5.0, -3.0, 1.0};
    double super[] = {0.0, 0.0};
    ASSERT_ERR(bidiag_svd_iterate(diag, super, 3, NULL, 0, NULL, 0, 0, 0), SPARSE_OK);
    /* Should be non-negative */
    ASSERT_NEAR(diag[0], 5.0, 1e-14);
    ASSERT_NEAR(diag[1], 3.0, 1e-14);
    ASSERT_NEAR(diag[2], 1.0, 1e-14);
}

/* 2×2 bidiagonal: known singular values */
static void test_bidiag_svd_2x2(void) {
    /* B = [[3, 1], [0, 4]]
     * B^T*B = [[9, 3], [3, 17]]
     * eigenvalues of B^T*B: singular values = sqrt(eigenvalues) */
    double diag[] = {3.0, 4.0};
    double super[] = {1.0};
    ASSERT_ERR(bidiag_svd_iterate(diag, super, 2, NULL, 0, NULL, 0, 0, 0), SPARSE_OK);

    /* Sort descending for comparison */
    if (diag[0] < diag[1]) {
        double t = diag[0];
        diag[0] = diag[1];
        diag[1] = t;
    }

    /* Analytical: eigenvalues of B^T*B = (26 ± sqrt(100+9))/2... compute numerically */
    double l1, l2;
    eigen2x2(9.0, 3.0, 17.0, &l1, &l2);
    double s1 = sqrt(l2); /* larger */
    double s2 = sqrt(l1); /* smaller */

    printf("    bidiag SVD 2x2: [%.6f, %.6f] expected [%.6f, %.6f]\n", diag[0], diag[1], s1, s2);
    ASSERT_NEAR(diag[0], s1, 1e-10);
    ASSERT_NEAR(diag[1], s2, 1e-10);
}

/* 3×3 bidiagonal with U/V accumulation */
static void test_bidiag_svd_3x3_uv(void) {
    double diag[] = {4.0, 3.0, 2.0};
    double super[] = {1.0, 1.0};

    /* Start with U = I_3, V = I_3 */
    double U[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double V[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    ASSERT_ERR(bidiag_svd_iterate(diag, super, 3, U, 3, V, 3, 0, 0), SPARSE_OK);

    /* Verify U*diag(sigma)*V^T ≈ original B */
    /* Original B: [[4,1,0],[0,3,1],[0,0,2]] */
    double orig[3][3] = {{4, 1, 0}, {0, 3, 1}, {0, 0, 2}};
    double maxerr = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double val = 0.0;
            for (int p = 0; p < 3; p++)
                val += U[p * 3 + i] * diag[p] * V[p * 3 + j];
            double e = fabs(orig[i][j] - val);
            if (e > maxerr)
                maxerr = e;
        }
    }
    printf("    bidiag SVD 3x3 UV recon: %.3e, sigma=[%.4f, %.4f, %.4f]\n", maxerr, diag[0],
           diag[1], diag[2]);
    ASSERT_TRUE(maxerr < 1e-10);
    /* All singular values positive */
    ASSERT_TRUE(diag[0] >= 0.0);
    ASSERT_TRUE(diag[1] >= 0.0);
    ASSERT_TRUE(diag[2] >= 0.0);
}

/* k=1: trivial */
static void test_bidiag_svd_k1(void) {
    double diag[] = {-7.0};
    ASSERT_ERR(bidiag_svd_iterate(diag, NULL, 1, NULL, 0, NULL, 0, 0, 0), SPARSE_OK);
    ASSERT_NEAR(diag[0], 7.0, 1e-14);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SVD convergence tests (Sprint 8 Day 7)
 * ═══════════════════════════════════════════════════════════════════════ */

/* SVD of diagonal: exact singular values */
static void test_svd_diagonal_5x5(void) {
    SparseMatrix *A = sparse_create(5, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 7.0);
    sparse_insert(A, 1, 1, -3.0);
    sparse_insert(A, 2, 2, 5.0);
    sparse_insert(A, 3, 3, 1.0);
    sparse_insert(A, 4, 4, -9.0);

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Descending: 9, 7, 5, 3, 1 */
    printf("    SVD diag 5x5: [%.3f, %.3f, %.3f, %.3f, %.3f]\n", svd.sigma[0], svd.sigma[1],
           svd.sigma[2], svd.sigma[3], svd.sigma[4]);
    ASSERT_NEAR(svd.sigma[0], 9.0, 1e-10);
    ASSERT_NEAR(svd.sigma[1], 7.0, 1e-10);
    ASSERT_NEAR(svd.sigma[2], 5.0, 1e-10);
    ASSERT_NEAR(svd.sigma[3], 3.0, 1e-10);
    ASSERT_NEAR(svd.sigma[4], 1.0, 1e-10);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* SVD sum of squared sigmas = trace(A^T*A) */
static void test_svd_trace_invariant(void) {
    SparseMatrix *A = sparse_create(4, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 2, 5.0);
    sparse_insert(A, 2, 2, 6.0);
    sparse_insert(A, 3, 0, 1.0);
    sparse_insert(A, 3, 2, 2.0);

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Sum of sigma^2 should equal sum of squared entries of A (Frobenius norm squared) */
    double frob_sq = 1 + 4 + 9 + 16 + 25 + 36 + 1 + 4; /* = 96 */
    double sigma_sq_sum = 0.0;
    for (idx_t i = 0; i < svd.k; i++)
        sigma_sq_sum += svd.sigma[i] * svd.sigma[i];

    printf("    SVD trace: sum(sigma^2)=%.3f, ||A||_F^2=%.3f\n", sigma_sq_sum, frob_sq);
    ASSERT_NEAR(sigma_sq_sum, frob_sq, 1e-8);

    /* All positive and descending */
    for (idx_t i = 0; i < svd.k; i++)
        ASSERT_TRUE(svd.sigma[i] >= 0.0);
    for (idx_t i = 1; i < svd.k; i++)
        ASSERT_TRUE(svd.sigma[i] <= svd.sigma[i - 1] + 1e-10);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* Rank-1 matrix: only one nonzero singular value */
static void test_svd_rank1(void) {
    SparseMatrix *A = sparse_create(4, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* rank-1: u*v^T where u=[1,2,3,4], v=[1,1,1] */
    for (idx_t i = 0; i < 4; i++)
        for (idx_t j = 0; j < 3; j++)
            sparse_insert(A, i, j, (double)(i + 1));

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_EQ(err, SPARSE_OK);

    printf("    SVD rank-1: [%.4f, %.4f, %.4f]\n", svd.sigma[0], svd.sigma[1], svd.sigma[2]);
    /* sigma[0] should be ||u||*||v|| = sqrt(30)*sqrt(3) ≈ 9.487 */
    double expected = sqrt(30.0) * sqrt(3.0);
    ASSERT_TRUE(fabs(svd.sigma[0] - expected) < 1e-6);
    /* sigma[1] and sigma[2] should be ~0 */
    ASSERT_TRUE(svd.sigma[1] < 1e-8);
    ASSERT_TRUE(svd.sigma[2] < 1e-8);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* Rank-1 matrix with UV reconstruction */
static void test_svd_rank1_uv(void) {
    SparseMatrix *A = sparse_create(4, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* rank-1: u*v^T where u=[1,2,3,4], v=[1,1,1] */
    for (idx_t i = 0; i < 4; i++)
        for (idx_t j = 0; j < 3; j++)
            sparse_insert(A, i, j, (double)(i + 1));

    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, &opts, &svd);
    ASSERT_EQ(err, SPARSE_OK);

    /* Verify reconstruction: A ≈ U * diag(sigma) * Vt
     * Vt is stored column-major: Vt[j * k + r] = (V^T)_{r,j} */
    double max_err = 0.0;
    idx_t k = svd.k;
    for (idx_t i = 0; i < 4; i++) {
        for (idx_t j = 0; j < 3; j++) {
            double sum = 0.0;
            for (idx_t s = 0; s < k; s++)
                sum += svd.U[(size_t)s * 4 + (size_t)i] * svd.sigma[s] *
                       svd.Vt[(size_t)j * (size_t)k + (size_t)s];
            double expected = (double)(i + 1);
            double e = fabs(sum - expected);
            if (e > max_err)
                max_err = e;
        }
    }
    printf("    SVD rank-1 UV reconstruction error: %.2e\n", max_err);
    ASSERT_TRUE(max_err < 1e-8);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* Rank-2 matrix in 5x5: exactly two nonzero singular values */
static void test_svd_rank2(void) {
    SparseMatrix *A = sparse_create(5, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* A = u1*v1^T + u2*v2^T where:
     * u1=[1,0,1,0,1], v1=[1,1,0,0,0]
     * u2=[0,1,0,1,0], v2=[0,0,1,1,0] */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 3, 2, 1.0);
    sparse_insert(A, 3, 3, 1.0);
    sparse_insert(A, 4, 0, 1.0);
    sparse_insert(A, 4, 1, 1.0);

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_EQ(err, SPARSE_OK);

    printf("    SVD rank-2: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", svd.sigma[0], svd.sigma[1],
           svd.sigma[2], svd.sigma[3], svd.sigma[4]);
    /* Exactly two nonzero singular values */
    ASSERT_TRUE(svd.sigma[0] > 0.1);
    ASSERT_TRUE(svd.sigma[1] > 0.1);
    ASSERT_TRUE(svd.sigma[2] < 1e-10);
    ASSERT_TRUE(svd.sigma[3] < 1e-10);
    ASSERT_TRUE(svd.sigma[4] < 1e-10);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* 10x10 rank-5: five nonzero + five near-zero singular values */
static void test_svd_rank5_in_10x10(void) {
    SparseMatrix *A = sparse_create(10, 10);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Build rank-5 matrix: A = sum_{r=0}^{4} (r+1)*e_r*e_r^T
     * where e_r is the r-th standard basis vector.
     * This is a diagonal matrix with 5 nonzeros. */
    for (idx_t i = 0; i < 5; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_EQ(err, SPARSE_OK);

    /* sigma = [5, 4, 3, 2, 1, 0, 0, 0, 0, 0] (descending) */
    ASSERT_NEAR(svd.sigma[0], 5.0, 1e-10);
    ASSERT_NEAR(svd.sigma[1], 4.0, 1e-10);
    ASSERT_NEAR(svd.sigma[2], 3.0, 1e-10);
    ASSERT_NEAR(svd.sigma[3], 2.0, 1e-10);
    ASSERT_NEAR(svd.sigma[4], 1.0, 1e-10);
    for (idx_t i = 5; i < 10; i++)
        ASSERT_TRUE(svd.sigma[i] < 1e-10);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* Rank-1 square matrix: zero-diagonal chase on square bidiagonal */
static void test_svd_rank1_square(void) {
    SparseMatrix *A = sparse_create(5, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* rank-1: all entries = row index + 1 (u=[1,2,3,4,5], v=[1,1,1,1,1]) */
    for (idx_t i = 0; i < 5; i++)
        for (idx_t j = 0; j < 5; j++)
            sparse_insert(A, i, j, (double)(i + 1));

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_EQ(err, SPARSE_OK);

    /* sigma[0] = ||u||*||v|| = sqrt(55)*sqrt(5) */
    double expected = sqrt(55.0) * sqrt(5.0);
    ASSERT_TRUE(fabs(svd.sigma[0] - expected) < 1e-6);
    for (idx_t i = 1; i < 5; i++)
        ASSERT_TRUE(svd.sigma[i] < 1e-8);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* Rank-1 wide matrix: m < n with zero-diagonal chase */
static void test_svd_rank1_wide(void) {
    SparseMatrix *A = sparse_create(3, 6);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* rank-1: u=[1,2,3], v=[1,1,1,1,1,1] */
    for (idx_t i = 0; i < 3; i++)
        for (idx_t j = 0; j < 6; j++)
            sparse_insert(A, i, j, (double)(i + 1));

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_EQ(err, SPARSE_OK);

    /* sigma[0] = ||u||*||v|| = sqrt(14)*sqrt(6) */
    double expected = sqrt(14.0) * sqrt(6.0);
    printf("    SVD rank-1 wide: sigma[0]=%.4f (expected %.4f)\n", svd.sigma[0], expected);
    ASSERT_TRUE(fabs(svd.sigma[0] - expected) < 1e-6);
    for (idx_t i = 1; i < 3; i++)
        ASSERT_TRUE(svd.sigma[i] < 1e-8);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* Near-singular matrix: diagonal entries approaching machine epsilon */
static void test_svd_near_singular(void) {
    SparseMatrix *A = sparse_create(4, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Diagonal with decreasing entries: 1, 1e-4, 1e-8, 1e-12 */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1e-4);
    sparse_insert(A, 2, 2, 1e-8);
    sparse_insert(A, 3, 3, 1e-12);

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_EQ(err, SPARSE_OK);

    ASSERT_NEAR(svd.sigma[0], 1.0, 1e-10);
    ASSERT_NEAR(svd.sigma[1], 1e-4, 1e-14);
    ASSERT_NEAR(svd.sigma[2], 1e-8, 1e-18);
    ASSERT_NEAR(svd.sigma[3], 1e-12, 1e-22);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* Multiple zero diagonals: bidiagonal with alternating zero/nonzero */
static void test_svd_multi_zero_diag(void) {
    /* Build a matrix whose bidiagonal has multiple near-zero diagonals.
     * Use a rank-2 matrix in 6×6: two outer products with gaps. */
    SparseMatrix *A = sparse_create(6, 6);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* A = diag(3, 0, 2, 0, 1, 0) — three zero singular values */
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 4, 4, 1.0);

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_EQ(err, SPARSE_OK);

    ASSERT_NEAR(svd.sigma[0], 3.0, 1e-10);
    ASSERT_NEAR(svd.sigma[1], 2.0, 1e-10);
    ASSERT_NEAR(svd.sigma[2], 1.0, 1e-10);
    for (idx_t i = 3; i < 6; i++)
        ASSERT_TRUE(svd.sigma[i] < 1e-10);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* Rank-deficient dense matrix: all rows are multiples of first two */
static void test_svd_rank2_dense(void) {
    SparseMatrix *A = sparse_create(5, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Row 0: [1, 2, 3, 4]
     * Row 1: [5, 6, 7, 8]
     * Row 2: [2, 4, 6, 8] = 2 * row 0
     * Row 3: [6, 8, 10, 12] = row 0 + row 1
     * Row 4: [3, 6, 9, 12] = 3 * row 0
     * Rank = 2 */
    double rows[5][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {2, 4, 6, 8}, {6, 8, 10, 12}, {3, 6, 9, 12}};
    for (idx_t i = 0; i < 5; i++)
        for (idx_t j = 0; j < 4; j++)
            sparse_insert(A, i, j, rows[i][j]);

    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1, .max_iter = 0, .tol = 0.0};
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, &opts, &svd);
    ASSERT_EQ(err, SPARSE_OK);

    printf("    SVD rank-2 dense: [%.4f, %.4f, %.4f, %.4f]\n", svd.sigma[0], svd.sigma[1],
           svd.sigma[2], svd.sigma[3]);
    /* Exactly two nonzero singular values */
    ASSERT_TRUE(svd.sigma[0] > 1.0);
    ASSERT_TRUE(svd.sigma[1] > 0.1);
    ASSERT_TRUE(svd.sigma[2] < 1e-10);
    ASSERT_TRUE(svd.sigma[3] < 1e-10);

    /* Verify reconstruction: A ≈ U * diag(sigma) * Vt */
    double max_err = 0.0;
    idx_t k = svd.k;
    for (idx_t i = 0; i < 5; i++) {
        for (idx_t j = 0; j < 4; j++) {
            double sum = 0.0;
            for (idx_t s = 0; s < k; s++)
                sum += svd.U[(size_t)s * 5 + (size_t)i] * svd.sigma[s] *
                       svd.Vt[(size_t)j * (size_t)k + (size_t)s];
            double e = fabs(sum - rows[i][j]);
            if (e > max_err)
                max_err = e;
        }
    }
    printf("    SVD rank-2 dense UV reconstruction error: %.2e\n", max_err);
    ASSERT_TRUE(max_err < 1e-8);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* SuiteSparse rank-deficient: zero some columns of nos4 */
static void test_svd_suitesparse_rank_deficient(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, "matrices/nos4.mtx");
    if (lerr != SPARSE_OK || !A) {
        printf("    SKIP: nos4.mtx not found\n");
        return;
    }
    /* Zero columns 50-99 to create a rank-deficient matrix */
    SparseMatrix *B = sparse_create(sparse_rows(A), sparse_cols(A));
    ASSERT_NOT_NULL(B);
    if (!B) {
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < sparse_rows(A); i++) {
        for (idx_t j = 0; j < 50; j++) {
            double val = sparse_get(A, i, j);
            if (val != 0.0)
                sparse_insert(B, i, j, val);
        }
    }
    sparse_free(A);

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(B, NULL, &svd);
    ASSERT_EQ(err, SPARSE_OK);

    /* Count nonzero singular values */
    idx_t rank = 0;
    for (idx_t i = 0; i < svd.k; i++) {
        if (svd.sigma[i] > 1e-10 * svd.sigma[0])
            rank++;
    }
    printf("    nos4 rank-deficient (cols 0-49 only): rank=%d (of %d)\n", (int)rank, (int)svd.k);
    ASSERT_TRUE(rank <= 50);
    ASSERT_TRUE(rank > 0);
    /* SVD rank estimation should agree */
    idx_t svd_rank = 0;
    err = sparse_svd_rank(B, 0.0, &svd_rank);
    ASSERT_EQ(err, SPARSE_OK);
    printf("    sparse_svd_rank reports: %d\n", (int)svd_rank);
    ASSERT_TRUE(svd_rank <= 50);

    sparse_svd_free(&svd);
    sparse_free(B);
}

/* SVD descending order */
static void test_svd_descending(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 10.0);
    sparse_insert(A, 2, 2, 5.0);

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_NEAR(svd.sigma[0], 10.0, 1e-10);
    ASSERT_NEAR(svd.sigma[1], 5.0, 1e-10);
    ASSERT_NEAR(svd.sigma[2], 1.0, 1e-10);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SVD edge cases (Sprint 8 Day 8)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Zero bidiag superdiag: should deflate immediately */
static void test_bidiag_svd_zero_super(void) {
    double diag[] = {5.0, 3.0, 1.0};
    double super[] = {0.0, 0.0};
    ASSERT_ERR(bidiag_svd_iterate(diag, super, 3, NULL, 0, NULL, 0, 0, 0), SPARSE_OK);
    ASSERT_NEAR(diag[0], 5.0, 1e-14);
    ASSERT_NEAR(diag[1], 3.0, 1e-14);
    ASSERT_NEAR(diag[2], 1.0, 1e-14);
}

/* All-zero matrix */
static void test_svd_all_zero(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* empty — all zeros */

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    for (idx_t i = 0; i < svd.k; i++)
        ASSERT_NEAR(svd.sigma[i], 0.0, 1e-14);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* Repeated singular values */
static void test_svd_repeated(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 5.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 2, 5.0);

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    for (idx_t i = 0; i < 3; i++)
        ASSERT_NEAR(svd.sigma[i], 5.0, 1e-10);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* SVD on 1×1 */
static void test_svd_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, -7.0);

    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1};
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, &opts, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_NEAR(svd.sigma[0], 7.0, 1e-10);
    ASSERT_NOT_NULL(svd.U);
    ASSERT_NOT_NULL(svd.Vt);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* SVD NULL inputs */
static void test_svd_null(void) {
    sparse_svd_t svd;
    ASSERT_ERR(sparse_svd_compute(NULL, NULL, &svd), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_svd_compute(NULL, NULL, NULL), SPARSE_ERR_NULL);
}

/* Larger diagonal: 20×20 */
static void test_svd_diag_20x20(void) {
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(n - i));

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Descending: 20, 19, ..., 1 */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(svd.sigma[i], (double)(n - i), 1e-10);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Full SVD driver tests (Sprint 8 Day 10)
 * ═══════════════════════════════════════════════════════════════════════ */

/* SVD on tall rectangular 10×5 */
static void test_svd_tall_10x5(void) {
    idx_t m = 10, nc = 5;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++)
            if (i == j)
                sparse_insert(A, i, j, (double)(5 - j));

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(svd.k, nc);
    /* Diagonal: sigma = [5, 4, 3, 2, 1] */
    ASSERT_NEAR(svd.sigma[0], 5.0, 1e-10);
    ASSERT_NEAR(svd.sigma[4], 1.0, 1e-10);
    ASSERT_TRUE(svd.sigma[0] >= svd.sigma[1]);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* SVD on wide rectangular 5×10 */
static void test_svd_wide_5x10(void) {
    idx_t m = 5, nc = 10;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        sparse_insert(A, i, i, (double)(m - i));

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(svd.k, m);
    ASSERT_NEAR(svd.sigma[0], 5.0, 1e-10);
    ASSERT_NEAR(svd.sigma[4], 1.0, 1e-10);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* SVD on wide rectangular 5×10 with UV: reconstruction and orthogonality */
static void test_svd_wide_5x10_uv(void) {
    idx_t m = 5, nc = 10;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    /* Non-diagonal wide matrix */
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 3, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 4, -1.0);
    sparse_insert(A, 2, 2, -2.0);
    sparse_insert(A, 2, 5, 0.5);
    sparse_insert(A, 3, 6, 1.5);
    sparse_insert(A, 3, 8, 2.5);
    sparse_insert(A, 4, 7, -3.0);
    sparse_insert(A, 4, 9, 1.0);

    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1};
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, &opts, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t k = svd.k;
    ASSERT_EQ(k, m);
    ASSERT_NOT_NULL(svd.U);
    ASSERT_NOT_NULL(svd.Vt);
    if (!svd.U || !svd.Vt) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }

    /* U orthonormality: U^T * U ≈ I_k */
    for (idx_t p = 0; p < k; p++) {
        for (idx_t q = p; q < k; q++) {
            double dot = 0.0;
            for (idx_t i = 0; i < m; i++)
                dot += svd.U[(size_t)p * (size_t)m + (size_t)i] *
                       svd.U[(size_t)q * (size_t)m + (size_t)i];
            double expected = (p == q) ? 1.0 : 0.0;
            ASSERT_NEAR(dot, expected, 1e-10);
        }
    }

    /* Vt orthonormality: Vt * Vt^T ≈ I_k.
     * Vt is k×nc column-major: Vt[col * k + row] = Vt_matrix[row, col] */
    for (idx_t p = 0; p < k; p++) {
        for (idx_t q = p; q < k; q++) {
            double dot = 0.0;
            for (idx_t j = 0; j < nc; j++)
                dot += svd.Vt[(size_t)j * (size_t)k + (size_t)p] *
                       svd.Vt[(size_t)j * (size_t)k + (size_t)q];
            double expected = (p == q) ? 1.0 : 0.0;
            ASSERT_NEAR(dot, expected, 1e-10);
        }
    }

    /* Reconstruction: U * diag(sigma) * Vt ≈ A */
    double max_err = 0.0;
    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = 0; j < nc; j++) {
            double val = 0.0;
            for (idx_t r = 0; r < k; r++)
                val += svd.U[(size_t)r * (size_t)m + (size_t)i] * svd.sigma[r] *
                       svd.Vt[(size_t)j * (size_t)k + (size_t)r];
            double e = fabs(val - sparse_get(A, i, j));
            if (e > max_err)
                max_err = e;
        }
    }
    printf("    wide 5x10 UV recon: ||U*S*Vt - A||_max = %.3e\n", max_err);
    ASSERT_TRUE(max_err < 1e-10);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* SVD singular-values-only vs with-UV: same sigma */
static void test_svd_sigma_only_vs_uv(void) {
    SparseMatrix *A = sparse_create(4, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 3, 3, 1.0);

    /* Sigma only */
    sparse_svd_t svd1;
    sparse_err_t err1 = sparse_svd_compute(A, NULL, &svd1);
    ASSERT_ERR(err1, SPARSE_OK);
    if (err1 != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    ASSERT_TRUE(svd1.U == NULL);
    ASSERT_TRUE(svd1.Vt == NULL);

    /* With UV */
    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1};
    sparse_svd_t svd2;
    sparse_err_t err2 = sparse_svd_compute(A, &opts, &svd2);
    ASSERT_ERR(err2, SPARSE_OK);
    if (err2 != SPARSE_OK) {
        sparse_svd_free(&svd1);
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(svd2.U);
    ASSERT_NOT_NULL(svd2.Vt);

    /* Same singular values */
    for (idx_t i = 0; i < svd1.k; i++)
        ASSERT_NEAR(svd1.sigma[i], svd2.sigma[i], 1e-10);

    printf("    sigma-only vs UV: sigma=[%.3f, %.3f, %.3f, %.3f]\n", svd1.sigma[0], svd1.sigma[1],
           svd1.sigma[2], svd1.sigma[3]);

    sparse_svd_free(&svd1);
    sparse_svd_free(&svd2);
    sparse_free(A);
}

/* SVD on nos4: singular values are positive, descending */
static void test_svd_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(svd.k, 100);
    /* All positive */
    for (idx_t i = 0; i < svd.k; i++)
        ASSERT_TRUE(svd.sigma[i] >= 0.0);
    /* Descending */
    for (idx_t i = 1; i < svd.k; i++)
        ASSERT_TRUE(svd.sigma[i] <= svd.sigma[i - 1] + 1e-10);

    printf("    SVD nos4: sigma_max=%.3f, sigma_min=%.6f\n", svd.sigma[0], svd.sigma[99]);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* SVD on west0067 */
static void test_svd_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(svd.k, 67);
    ASSERT_TRUE(svd.sigma[0] > 0.0);
    /* Descending */
    for (idx_t i = 1; i < svd.k; i++)
        ASSERT_TRUE(svd.sigma[i] <= svd.sigma[i - 1] + 1e-10);

    printf("    SVD west0067: sigma_max=%.3f, sigma_min=%.6e\n", svd.sigma[0], svd.sigma[66]);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* SVD rank matches QR rank on rank-deficient matrix */
static void test_svd_rank_vs_qr(void) {
    SparseMatrix *A = sparse_create(5, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* col1 = col0, col3 = col2 → rank 2 */
    for (idx_t i = 0; i < 5; i++) {
        sparse_insert(A, i, 0, (double)(i + 1));
        sparse_insert(A, i, 1, (double)(i + 1));
        sparse_insert(A, i, 2, (double)(i * 2 + 1));
        sparse_insert(A, i, 3, (double)(i * 2 + 1));
    }

    /* SVD rank: count sigma > tol */
    sparse_svd_t svd;
    sparse_err_t err = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t svd_rank = 0;
    double tol_svd = 1e-8 * svd.sigma[0];
    for (idx_t i = 0; i < svd.k; i++)
        if (svd.sigma[i] > tol_svd)
            svd_rank++;

    /* QR rank */
    sparse_qr_t qr;
    sparse_err_t qr_err = sparse_qr_factor(A, &qr);
    idx_t qr_rank = (qr_err == SPARSE_OK) ? qr.rank : -1;

    printf("    SVD rank=%d, QR rank=%d\n", (int)svd_rank, (int)qr_rank);
    ASSERT_EQ(svd_rank, qr_rank);

    if (qr_err == SPARSE_OK)
        sparse_qr_free(&qr);
    sparse_svd_free(&svd);
    sparse_free(A);
}

/* SVD NULL svd pointer with valid matrix */
static void test_svd_null_input(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);

    ASSERT_ERR(sparse_svd_compute(A, NULL, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Partial SVD via Lanczos bidiagonalization (Sprint 8 Day 11)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Partial SVD NULL inputs */
static void test_partial_svd_null(void) {
    sparse_svd_t svd;
    ASSERT_ERR(sparse_svd_partial(NULL, 3, NULL, &svd), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_svd_partial(NULL, 3, NULL, NULL), SPARSE_ERR_NULL);
}

/* Partial SVD bad k */
static void test_partial_svd_bad_k(void) {
    SparseMatrix *A = sparse_create(5, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 5; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    sparse_svd_t svd;
    ASSERT_ERR(sparse_svd_partial(A, 0, NULL, &svd), SPARSE_ERR_BADARG);
    ASSERT_ERR(sparse_svd_partial(A, -1, NULL, &svd), SPARSE_ERR_BADARG);
    ASSERT_ERR(sparse_svd_partial(A, 6, NULL, &svd), SPARSE_ERR_BADARG);

    sparse_free(A);
}

/* Partial SVD on diagonal 10×10: top 3 should match full SVD */
static void test_partial_svd_diag_10x10(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(n - i));

    /* Full SVD for reference */
    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Partial: k=3 */
    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 3, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 3);
    ASSERT_EQ(partial.m, n);
    ASSERT_EQ(partial.n, n);

    /* Top 3 singular values should match */
    for (idx_t i = 0; i < 3; i++) {
        printf("    partial sigma[%d]=%.6f, full sigma[%d]=%.6f\n", (int)i, partial.sigma[i],
               (int)i, full.sigma[i]);
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 1e-10);
    }

    /* No U/Vt (partial doesn't compute them) */
    ASSERT_TRUE(partial.U == NULL);
    ASSERT_TRUE(partial.Vt == NULL);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/* Partial SVD k = min(m,n): should match full SVD */
static void test_partial_svd_full_k(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(n - i));

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, n, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, n);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 1e-10);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/* Partial SVD on dense-ish matrix: top k singular values match */
static void test_partial_svd_dense_8x8(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Symmetric positive definite: A(i,j) = 1/(i+j+1) (Hilbert-like) */
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, 1.0 / (double)(i + j + 1));

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t kk = 4;
    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, kk, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, kk);
    printf("    Hilbert 8x8 partial SVD (k=%d):\n", (int)kk);
    for (idx_t i = 0; i < kk; i++) {
        printf("      sigma[%d]: partial=%.8f, full=%.8f\n", (int)i, partial.sigma[i],
               full.sigma[i]);
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 1e-8);
    }

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/* Partial SVD on rectangular tall matrix */
static void test_partial_svd_tall(void) {
    idx_t m = 10, nc = 5;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < nc; i++)
        sparse_insert(A, i, i, (double)(nc - i));

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 3, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 3);
    for (idx_t i = 0; i < 3; i++)
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 1e-10);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/* Partial SVD on wide matrix */
static void test_partial_svd_wide(void) {
    idx_t m = 5, nc = 10;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        sparse_insert(A, i, i, (double)(m - i));

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 3, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 3);
    /* Lanczos on wide matrices converges more slowly; use relative tolerance */
    for (idx_t i = 0; i < 3; i++)
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 0.05 * full.sigma[i]);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/* Partial SVD on nos4 (100×100): top 5 match full SVD */
static void test_partial_svd_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t kk = 5;
    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, kk, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, kk);
    printf("    nos4 partial SVD (k=%d):\n", (int)kk);
    for (idx_t i = 0; i < kk; i++) {
        printf("      sigma[%d]: partial=%.6f, full=%.6f\n", (int)i, partial.sigma[i],
               full.sigma[i]);
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 0.1 * full.sigma[i]);
    }

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Partial SVD validation (Sprint 8 Day 12)
 * ═══════════════════════════════════════════════════════════════════════ */

/* k=1: single largest singular value */
static void test_partial_svd_k1(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(n - i));

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 1, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 1);
    printf("    k=1: partial=%.6f, full=%.6f\n", partial.sigma[0], full.sigma[0]);
    ASSERT_NEAR(partial.sigma[0], full.sigma[0], 1e-10);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/* Rank-deficient matrix: k > rank should still work (extra values near zero) */
static void test_partial_svd_rank_deficient(void) {
    /* 6×4 matrix with rank 2: col0=col1, col2=col3 */
    SparseMatrix *A = sparse_create(6, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 6; i++) {
        sparse_insert(A, i, 0, (double)(i + 1));
        sparse_insert(A, i, 1, (double)(i + 1));
        sparse_insert(A, i, 2, (double)(i * 2 + 1));
        sparse_insert(A, i, 3, (double)(i * 2 + 1));
    }

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* k=4 = min(m,n), but true rank is 2 */
    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 4, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 4);
    /* Top 2 should be non-trivial, bottom 2 near zero */
    printf("    rank-def: sigma=[%.4f, %.4f, %.4f, %.4f]\n", partial.sigma[0], partial.sigma[1],
           partial.sigma[2], partial.sigma[3]);
    ASSERT_TRUE(partial.sigma[0] > 0.1);
    ASSERT_TRUE(partial.sigma[1] > 0.1);
    ASSERT_NEAR(partial.sigma[2], 0.0, 1e-8);
    ASSERT_NEAR(partial.sigma[3], 0.0, 1e-8);

    /* Top 2 match full SVD */
    ASSERT_NEAR(partial.sigma[0], full.sigma[0], 1e-8);
    ASSERT_NEAR(partial.sigma[1], full.sigma[1], 1e-8);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/* Partial SVD on west0067: top singular values match full SVD */
static void test_partial_svd_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t kk = 5;
    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, kk, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, kk);
    printf("    west0067 partial SVD (k=%d):\n", (int)kk);
    for (idx_t i = 0; i < kk; i++) {
        printf("      sigma[%d]: partial=%.6f, full=%.6f\n", (int)i, partial.sigma[i],
               full.sigma[i]);
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 0.1 * full.sigma[i]);
    }

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/* Descending order guaranteed for partial SVD */
static void test_partial_svd_descending(void) {
    idx_t n = 15;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Non-trivial matrix: tridiagonal */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 2.0 * (double)(i + 1));
        if (i + 1 < n)
            sparse_insert(A, i, i + 1, 1.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, 1.0);
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 7, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 7);
    /* All positive */
    for (idx_t i = 0; i < 7; i++)
        ASSERT_TRUE(partial.sigma[i] >= 0.0);
    /* Descending */
    for (idx_t i = 1; i < 7; i++)
        ASSERT_TRUE(partial.sigma[i] <= partial.sigma[i - 1] + 1e-10);

    printf("    descending: sigma[0]=%.4f, sigma[6]=%.4f\n", partial.sigma[0], partial.sigma[6]);

    sparse_svd_free(&partial);
    sparse_free(A);
}

/* Timing comparison: partial vs full SVD on nos4 */
static void test_partial_svd_timing(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    /* Time full SVD */
    clock_t t0 = clock();
    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    double full_time = (double)(clock() - t0) / CLOCKS_PER_SEC;
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Time partial SVD (k=5) */
    t0 = clock();
    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 5, NULL, &partial);
    double partial_time = (double)(clock() - t0) / CLOCKS_PER_SEC;
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    printf("    nos4 timing: full=%.4f s, partial(k=5)=%.4f s\n", full_time, partial_time);

    /* Just verify partial is reasonable — timing is informational */
    ASSERT_EQ(partial.k, 5);
    ASSERT_TRUE(partial.sigma[0] > 0.0);

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/* Partial SVD on dense non-symmetric matrix */
static void test_partial_svd_nonsymmetric(void) {
    idx_t m = 10, nc = 8;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Fill with structured non-symmetric values */
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++)
            if ((i + j) % 3 != 0)
                sparse_insert(A, i, j, (double)(i + 1) / (double)(j + 1));

    sparse_svd_t full;
    sparse_err_t ferr = sparse_svd_compute(A, NULL, &full);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_svd_t partial;
    sparse_err_t perr = sparse_svd_partial(A, 4, NULL, &partial);
    ASSERT_ERR(perr, SPARSE_OK);
    if (perr != SPARSE_OK) {
        sparse_svd_free(&full);
        sparse_free(A);
        return;
    }

    ASSERT_EQ(partial.k, 4);
    printf("    non-symmetric 10x8 partial (k=4):\n");
    for (idx_t i = 0; i < 4; i++) {
        printf("      sigma[%d]: partial=%.6f, full=%.6f\n", (int)i, partial.sigma[i],
               full.sigma[i]);
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 0.05 * full.sigma[i] + 1e-10);
    }

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SVD applications: rank, pseudoinverse, low-rank (Sprint 8 Day 13)
 * ═══════════════════════════════════════════════════════════════════════ */

/* SVD rank: full-rank diagonal */
static void test_svd_rank_full(void) {
    SparseMatrix *A = sparse_create(5, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 5; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    idx_t rank;
    sparse_err_t err = sparse_svd_rank(A, 0.0, &rank);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    printf("    full-rank 5x5: rank=%d\n", (int)rank);
    ASSERT_EQ(rank, 5);

    sparse_free(A);
}

/* SVD rank: rank-deficient matrix */
static void test_svd_rank_deficient(void) {
    /* 5×4 with col0=col1, col2=col3 → rank 2 */
    SparseMatrix *A = sparse_create(5, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 5; i++) {
        sparse_insert(A, i, 0, (double)(i + 1));
        sparse_insert(A, i, 1, (double)(i + 1));
        sparse_insert(A, i, 2, (double)(i * 2 + 1));
        sparse_insert(A, i, 3, (double)(i * 2 + 1));
    }

    idx_t rank;
    sparse_err_t err = sparse_svd_rank(A, 0.0, &rank);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    printf("    rank-deficient 5x4: rank=%d\n", (int)rank);
    ASSERT_EQ(rank, 2);

    sparse_free(A);
}

/* SVD rank: nearly singular (large condition number) */
static void test_svd_rank_nearly_singular(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 2, 2, 1e-14); /* near zero but above machine eps */

    idx_t rank;
    sparse_err_t err = sparse_svd_rank(A, 0.0, &rank);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    printf("    nearly-singular 3x3: rank=%d\n", (int)rank);
    /* Default tol = eps * max(m,n) * sigma_max ≈ 2.2e-16 * 3 * 1 ≈ 6.6e-16
     * sigma_min = 1e-14 > tol, so rank should be 3 */
    ASSERT_EQ(rank, 3);

    /* With explicit tolerance 1e-12, rank = 2 */
    sparse_err_t err2 = sparse_svd_rank(A, 1e-12, &rank);
    ASSERT_ERR(err2, SPARSE_OK);
    if (err2 != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    printf("    nearly-singular 3x3 (tol=1e-12): rank=%d\n", (int)rank);
    ASSERT_EQ(rank, 2);

    sparse_free(A);
}

/* SVD rank: NULL inputs */
static void test_svd_rank_null(void) {
    idx_t rank;
    ASSERT_ERR(sparse_svd_rank(NULL, 0.0, &rank), SPARSE_ERR_NULL);
}

/* Pseudoinverse: diagonal matrix */
static void test_pinv_diagonal(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 2, 2, 5.0);

    double *pinv_data = NULL;
    sparse_err_t err = sparse_pinv(A, 0.0, &pinv_data);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(pinv_data);
    if (!pinv_data) {
        sparse_free(A);
        return;
    }

    /* pinv of diag(2,4,5) = diag(0.5, 0.25, 0.2) */
    /* pinv is 3×3 column-major: pinv[col*3 + row] */
    ASSERT_NEAR(pinv_data[0 * 3 + 0], 0.5, 1e-10);
    ASSERT_NEAR(pinv_data[1 * 3 + 1], 0.25, 1e-10);
    ASSERT_NEAR(pinv_data[2 * 3 + 2], 0.2, 1e-10);
    /* Off-diag should be ~0 */
    ASSERT_NEAR(pinv_data[0 * 3 + 1], 0.0, 1e-10);
    ASSERT_NEAR(pinv_data[1 * 3 + 0], 0.0, 1e-10);

    free(pinv_data);
    sparse_free(A);
}

/* Pseudoinverse: Moore-Penrose condition A * A^+ * A ≈ A */
static void test_pinv_moore_penrose(void) {
    idx_t m = 4, nc = 3;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Non-trivial tall matrix */
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 0, 1.0);
    sparse_insert(A, 3, 2, 2.0);

    double *pinv_data = NULL;
    sparse_err_t err = sparse_pinv(A, 0.0, &pinv_data);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(pinv_data);
    if (!pinv_data) {
        sparse_free(A);
        return;
    }

    /* pinv is nc×m column-major: pinv[col*nc + row] */
    /* Verify A * A^+ * A ≈ A.
     * Step 1: compute B = A * pinv (m×nc * nc×m = m×m) using sparse_get + dense mult
     * Step 2: compute C = B * A (m×m * m×nc = m×nc), compare to A */

    /* Compute A * pinv as dense m×m */
    double *B = calloc((size_t)m * (size_t)m, sizeof(double));
    ASSERT_NOT_NULL(B);
    if (!B) {
        free(pinv_data);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = 0; j < m; j++) {
            double sum = 0.0;
            for (idx_t p = 0; p < nc; p++)
                sum += sparse_get(A, i, p) * pinv_data[(size_t)j * (size_t)nc + (size_t)p];
            B[(size_t)j * (size_t)m + (size_t)i] = sum;
        }
    }

    /* Compute (A * pinv) * A and compare to A */
    double max_err = 0.0;
    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = 0; j < nc; j++) {
            double sum = 0.0;
            for (idx_t p = 0; p < m; p++)
                sum += B[(size_t)p * (size_t)m + (size_t)i] * sparse_get(A, p, j);
            double e = fabs(sum - sparse_get(A, i, j));
            if (e > max_err)
                max_err = e;
        }
    }
    printf("    Moore-Penrose ||A*A^+*A - A||_max = %.3e\n", max_err);
    ASSERT_TRUE(max_err < 1e-10);

    free(B);
    free(pinv_data);
    sparse_free(A);
}

/* Pseudoinverse: NULL inputs */
static void test_pinv_null(void) {
    double *pinv_data = NULL;
    ASSERT_ERR(sparse_pinv(NULL, 0.0, &pinv_data), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_pinv(NULL, 0.0, NULL), SPARSE_ERR_NULL);
}

/* Pseudoinverse: rectangular tall matrix — verify A * A^+ * A ≈ A */
static void test_pinv_rectangular(void) {
    /* 3×2 full column rank */
    idx_t m = 3, nc = 2;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 2, 1, 3.0);

    double *pinv_data = NULL;
    sparse_err_t err = sparse_pinv(A, 0.0, &pinv_data);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(pinv_data);
    if (!pinv_data) {
        sparse_free(A);
        return;
    }

    /* pinv is nc×m = 2×3 column-major.
     * Verify A * A^+ * A ≈ A (first Moore-Penrose condition).
     * Compute B = A * pinv (m×nc * nc×m = m×m), then C = B * A (m×m * m×nc = m×nc). */
    double *B = calloc((size_t)m * (size_t)m, sizeof(double));
    ASSERT_NOT_NULL(B);
    if (!B) {
        free(pinv_data);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < m; j++) {
            double sum = 0.0;
            for (idx_t p = 0; p < nc; p++)
                sum += sparse_get(A, i, p) * pinv_data[(size_t)j * (size_t)nc + (size_t)p];
            B[(size_t)j * (size_t)m + (size_t)i] = sum;
        }

    double max_err = 0.0;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++) {
            double sum = 0.0;
            for (idx_t p = 0; p < m; p++)
                sum += B[(size_t)p * (size_t)m + (size_t)i] * sparse_get(A, p, j);
            double e = fabs(sum - sparse_get(A, i, j));
            if (e > max_err)
                max_err = e;
        }
    printf("    rectangular pinv ||A*A^+*A - A||_max = %.3e\n", max_err);
    ASSERT_TRUE(max_err < 1e-10);

    free(B);
    free(pinv_data);
    sparse_free(A);
}

/* Low-rank approximation: rank-k of diagonal */
static void test_lowrank_diagonal(void) {
    SparseMatrix *A = sparse_create(4, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 10.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 3, 3, 1.0);

    double *lr = NULL;
    sparse_err_t err = sparse_svd_lowrank(A, 2, &lr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(lr);
    if (!lr) {
        sparse_free(A);
        return;
    }

    /* Rank-2 approx of diag(10,5,2,1) = diag(10,5,0,0) */
    /* lr is 4×4 col-major */
    ASSERT_NEAR(lr[0 * 4 + 0], 10.0, 1e-10);
    ASSERT_NEAR(lr[1 * 4 + 1], 5.0, 1e-10);
    ASSERT_NEAR(lr[2 * 4 + 2], 0.0, 1e-10);
    ASSERT_NEAR(lr[3 * 4 + 3], 0.0, 1e-10);

    /* ||A - A_k||_F = sqrt(2^2 + 1^2) = sqrt(5) */
    double frob_err = 0.0;
    for (idx_t i = 0; i < 4; i++)
        for (idx_t j = 0; j < 4; j++) {
            double diff = sparse_get(A, i, j) - lr[(size_t)j * 4 + (size_t)i];
            frob_err += diff * diff;
        }
    frob_err = sqrt(frob_err);
    printf("    lowrank(2) diag: ||A - A_k||_F = %.6f (expected %.6f)\n", frob_err, sqrt(5.0));
    ASSERT_NEAR(frob_err, sqrt(5.0), 1e-10);

    free(lr);
    sparse_free(A);
}

/* Low-rank: error matches theoretical bound */
static void test_lowrank_error_bound(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Tridiagonal */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 2.0 * (double)(i + 1));
        if (i + 1 < n)
            sparse_insert(A, i, i + 1, 1.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, 1.0);
    }

    /* Get full SVD for reference */
    sparse_svd_t svd;
    sparse_err_t serr = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(serr, SPARSE_OK);
    if (serr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t rank_k = 3;
    double *lr = NULL;
    sparse_err_t lrerr = sparse_svd_lowrank(A, rank_k, &lr);
    ASSERT_ERR(lrerr, SPARSE_OK);
    if (lrerr != SPARSE_OK) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(lr);
    if (!lr) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }

    /* ||A - A_k||_F should = sqrt(sum_{i=k}^{n-1} sigma_i^2) */
    double expected_sq = 0.0;
    for (idx_t i = rank_k; i < svd.k; i++)
        expected_sq += svd.sigma[i] * svd.sigma[i];
    double expected = sqrt(expected_sq);

    double actual_sq = 0.0;
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++) {
            double diff = sparse_get(A, i, j) - lr[(size_t)j * (size_t)n + (size_t)i];
            actual_sq += diff * diff;
        }
    double actual = sqrt(actual_sq);

    printf("    lowrank(%d) tridiag: ||A-A_k||_F = %.6f, expected = %.6f\n", (int)rank_k, actual,
           expected);
    ASSERT_NEAR(actual, expected, 1e-8);

    free(lr);
    sparse_svd_free(&svd);
    sparse_free(A);
}

/* Low-rank: NULL and bad args */
static void test_lowrank_errors(void) {
    double *lr = NULL;
    ASSERT_ERR(sparse_svd_lowrank(NULL, 2, &lr), SPARSE_ERR_NULL);

    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);

    ASSERT_ERR(sparse_svd_lowrank(A, 0, &lr), SPARSE_ERR_BADARG);
    ASSERT_ERR(sparse_svd_lowrank(A, 4, &lr), SPARSE_ERR_BADARG);
    ASSERT_ERR(sparse_svd_lowrank(A, -1, &lr), SPARSE_ERR_BADARG);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test suite
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("SVD");

    /* Golub-Kahan extraction (Day 4) */
    RUN_TEST(test_gk_extract_3x3);
    RUN_TEST(test_gk_extract_tall);
    RUN_TEST(test_gk_extract_wide);
    RUN_TEST(test_svd_basic_sigma);

    /* Golub-Kahan validation (Day 5) */
    RUN_TEST(test_gk_square_5x5);
    RUN_TEST(test_gk_tall_ortho);
    RUN_TEST(test_gk_wide_ortho);
    RUN_TEST(test_gk_rank_deficient);
    RUN_TEST(test_gk_nos4);
    RUN_TEST(test_gk_west0067);
    RUN_TEST(test_gk_1x1);
    RUN_TEST(test_svd_with_uv);

    /* Bidiagonal SVD iteration (Day 6) */
    RUN_TEST(test_bidiag_svd_diagonal);
    RUN_TEST(test_bidiag_svd_2x2);
    RUN_TEST(test_bidiag_svd_3x3_uv);
    RUN_TEST(test_bidiag_svd_k1);

    /* SVD convergence (Day 7) */
    RUN_TEST(test_svd_diagonal_5x5);
    RUN_TEST(test_svd_descending);
    RUN_TEST(test_svd_trace_invariant);
    RUN_TEST(test_svd_rank1);
    RUN_TEST(test_svd_rank1_uv);
    RUN_TEST(test_svd_rank2);
    RUN_TEST(test_svd_rank5_in_10x10);
    RUN_TEST(test_svd_rank1_square);
    RUN_TEST(test_svd_rank1_wide);
    RUN_TEST(test_svd_near_singular);
    RUN_TEST(test_svd_multi_zero_diag);
    RUN_TEST(test_svd_rank2_dense);
    RUN_TEST(test_svd_suitesparse_rank_deficient);

    /* SVD edge cases (Day 8) */
    RUN_TEST(test_bidiag_svd_zero_super);
    RUN_TEST(test_svd_all_zero);
    RUN_TEST(test_svd_repeated);
    RUN_TEST(test_svd_1x1);
    RUN_TEST(test_svd_null);
    RUN_TEST(test_svd_diag_20x20);

    /* Full SVD driver (Day 10) */
    RUN_TEST(test_svd_tall_10x5);
    RUN_TEST(test_svd_wide_5x10);
    RUN_TEST(test_svd_wide_5x10_uv);
    RUN_TEST(test_svd_sigma_only_vs_uv);
    RUN_TEST(test_svd_nos4);
    RUN_TEST(test_svd_west0067);
    RUN_TEST(test_svd_rank_vs_qr);
    RUN_TEST(test_svd_null_input);

    /* Partial SVD / Lanczos (Day 11) */
    RUN_TEST(test_partial_svd_null);
    RUN_TEST(test_partial_svd_bad_k);
    RUN_TEST(test_partial_svd_diag_10x10);
    RUN_TEST(test_partial_svd_full_k);
    RUN_TEST(test_partial_svd_dense_8x8);
    RUN_TEST(test_partial_svd_tall);
    RUN_TEST(test_partial_svd_wide);
    RUN_TEST(test_partial_svd_nos4);

    /* Partial SVD validation (Day 12) */
    RUN_TEST(test_partial_svd_k1);
    RUN_TEST(test_partial_svd_rank_deficient);
    RUN_TEST(test_partial_svd_west0067);
    RUN_TEST(test_partial_svd_descending);
    RUN_TEST(test_partial_svd_timing);
    RUN_TEST(test_partial_svd_nonsymmetric);

    /* SVD applications (Day 13) */
    RUN_TEST(test_svd_rank_full);
    RUN_TEST(test_svd_rank_deficient);
    RUN_TEST(test_svd_rank_nearly_singular);
    RUN_TEST(test_svd_rank_null);
    RUN_TEST(test_pinv_diagonal);
    RUN_TEST(test_pinv_moore_penrose);
    RUN_TEST(test_pinv_null);
    RUN_TEST(test_pinv_rectangular);
    RUN_TEST(test_lowrank_diagonal);
    RUN_TEST(test_lowrank_error_bound);
    RUN_TEST(test_lowrank_errors);

    TEST_SUITE_END();
}
