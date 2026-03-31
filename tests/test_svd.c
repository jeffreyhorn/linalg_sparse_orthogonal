#include "sparse_bidiag.h"
#include "sparse_matrix.h"
#include "sparse_svd.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    /* Compute U*B first: (m×k) * (k×k bidiag) = m×k */
    double *UB = calloc((size_t)m * (size_t)k, sizeof(double));
    if (!UB)
        return INFINITY;

    for (idx_t j = 0; j < k; j++) {
        /* UB[:,j] = U[:,j] * diag[j] */
        for (idx_t i = 0; i < m; i++)
            UB[(size_t)j * (size_t)m + (size_t)i] = U[(size_t)j * (size_t)m + (size_t)i] * diag[j];
        /* UB[:,j] += U[:,j-1] * superdiag[j-1] (if j > 0) — no, B is upper bidiag:
         * B(i,j) nonzero for i==j (diag) and i==j-1 (superdiag).
         * So UB[:,j] = U[:,j]*B(j,j) + U[:,j-1]*B(j-1,j) for j>0. Wait...
         * B is k×k with B(i,i)=diag[i], B(i,i+1)=superdiag[i].
         * (U*B)[:,j] = sum_i U[:,i]*B(i,j).
         * B(i,j) is nonzero for i=j (diag[j]) and i=j-1 (superdiag[j-1] if j>0).
         * Wait no: B(i,i+1)=superdiag[i], so B(j-1,j)=superdiag[j-1]. */
    }
    /* Redo: UB = U * B where B is upper bidiagonal */
    memset(UB, 0, (size_t)m * (size_t)k * sizeof(double));
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

    double recon = gk_reconstruction_error(A, U, V, bd.diag, bd.superdiag, m, nc, k);
    printf("    GK 5x10: recon=%.3e\n", recon);
    ASSERT_TRUE(recon < 1e-10);

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
 * Test suite
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("SVD");

    /* Golub-Kahan extraction (Day 4) */
    RUN_TEST(test_gk_extract_3x3);
    RUN_TEST(test_gk_extract_tall);
    RUN_TEST(test_gk_extract_wide);
    RUN_TEST(test_svd_basic_sigma);

    TEST_SUITE_END();
}
