#include "sparse_bidiag.h"
#include "sparse_dense.h"
#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_matrix.h"
#include "sparse_qr.h"
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
 * Cross-feature integration tests (Sprint 8 Day 14)
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * SVD trace identity on nos4: sum(sigma_i^2) == ||A||_F^2.
 * This verifies the bidiagonal QR iteration preserves the Frobenius norm.
 */
static void test_svd_vs_ata_eigenvalues(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    idx_t n = sparse_rows(A);
    ASSERT_EQ(n, 100);

    /* Compute SVD */
    sparse_svd_t svd;
    ASSERT_ERR(sparse_svd_compute(A, NULL, &svd), SPARSE_OK);

    /* Compute A^T*A as dense, get its Frobenius norm and trace.
     * For SPD nos4: A = A^T, so A^T*A = A^2.
     * sum(sigma_i^2) = ||A||_F^2 = trace(A^T*A). */
    double sum_sigma_sq = 0.0;
    for (idx_t i = 0; i < svd.k; i++)
        sum_sigma_sq += svd.sigma[i] * svd.sigma[i];

    /* Compute ||A||_F^2 directly */
    double frob_sq = 0.0;
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++) {
            double v = sparse_get(A, i, j);
            frob_sq += v * v;
        }

    printf("    nos4: sum(sigma^2)=%.6f, ||A||_F^2=%.6f\n", sum_sigma_sq, frob_sq);
    ASSERT_NEAR(sum_sigma_sq, frob_sq, 1e-8 * frob_sq);

    /* Singular values should be positive and descending */
    for (idx_t i = 0; i < svd.k; i++)
        ASSERT_TRUE(svd.sigma[i] >= 0.0);
    for (idx_t i = 1; i < svd.k; i++)
        ASSERT_TRUE(svd.sigma[i] <= svd.sigma[i - 1] + 1e-14);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/**
 * Matrix-free GMRES with ILU preconditioner on steam1.
 * Wrap SparseMatrix in a callback and verify convergence.
 */
static sparse_err_t matvec_wrapper(const void *ctx, idx_t nn, const double *x, double *y) {
    (void)nn;
    return sparse_matvec((const SparseMatrix *)ctx, x, y);
}

static void test_mf_gmres_ilu_steam1(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/steam1.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    idx_t n = sparse_rows(A);

    /* Set up RHS b = A * ones */
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    double *ones = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(ones);
    if (!b || !x || !ones) {
        free(b);
        free(x);
        free(ones);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* Compute ILU(0) preconditioner */
    sparse_ilu_t ilu;
    sparse_err_t ilu_err = sparse_ilu_factor(A, &ilu);
    ASSERT_ERR(ilu_err, SPARSE_OK);
    if (ilu_err != SPARSE_OK) {
        free(b);
        free(x);
        free(ones);
        sparse_free(A);
        return;
    }

    /* Solve with matrix-free GMRES + ILU preconditioner */
    sparse_gmres_opts_t opts = {.tol = 1e-10, .max_iter = 500, .restart = 50};
    sparse_iter_result_t result;
    sparse_err_t err =
        sparse_solve_gmres_mf(matvec_wrapper, A, n, b, x, &opts, sparse_ilu_precond, &ilu, &result);

    printf("    steam1 MF-GMRES+ILU: iters=%d, resid=%.3e, converged=%d\n", (int)result.iterations,
           result.residual_norm, result.converged);

    /* Check convergence and solution accuracy */
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    /* Verify solution: ||x - ones|| should be small */
    double sol_err = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = x[i] - ones[i];
        sol_err += d * d;
    }
    sol_err = sqrt(sol_err) / sqrt((double)n);
    printf("    steam1 ||x - x_true||/sqrt(n) = %.3e\n", sol_err);
    ASSERT_TRUE(sol_err < 1e-6);

    sparse_ilu_free(&ilu);
    free(b);
    free(x);
    free(ones);
    sparse_free(A);
}

/**
 * Partial SVD top-k matches full SVD top-k on bcsstk04.
 */
static void test_partial_vs_full_bcsstk04(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    idx_t n = sparse_rows(A);
    printf("    bcsstk04: n=%d\n", (int)n);

    /* Full SVD */
    sparse_svd_t full;
    ASSERT_ERR(sparse_svd_compute(A, NULL, &full), SPARSE_OK);

    /* Partial SVD: top 5 */
    idx_t kk = 5;
    if (kk > full.k)
        kk = full.k;
    sparse_svd_t partial;
    ASSERT_ERR(sparse_svd_partial(A, kk, NULL, &partial), SPARSE_OK);

    ASSERT_EQ(partial.k, kk);
    printf("    bcsstk04 partial vs full (k=%d):\n", (int)kk);
    for (idx_t i = 0; i < kk; i++) {
        printf("      sigma[%d]: partial=%.6f, full=%.6f\n", (int)i, partial.sigma[i],
               full.sigma[i]);
        /* Allow 10% relative tolerance for Lanczos approximation */
        ASSERT_NEAR(partial.sigma[i], full.sigma[i], 0.1 * full.sigma[i] + 1e-10);
    }

    sparse_svd_free(&full);
    sparse_svd_free(&partial);
    sparse_free(A);
}

/**
 * SVD rank matches QR rank on a rank-deficient matrix.
 */
static void test_svd_rank_matches_qr(void) {
    /* 6×4 matrix with rank 2: col0=col1, col2=col3 */
    idx_t m = 6, nc = 4;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++) {
        double v1 = (double)(i + 1);
        double v2 = (double)(i * 2 + 1);
        sparse_insert(A, i, 0, v1);
        sparse_insert(A, i, 1, v1); /* dup of col 0 */
        sparse_insert(A, i, 2, v2);
        sparse_insert(A, i, 3, v2); /* dup of col 2 */
    }

    idx_t svd_rank;
    ASSERT_ERR(sparse_svd_rank(A, 0.0, &svd_rank), SPARSE_OK);

    sparse_qr_t qr;
    sparse_err_t qr_err = sparse_qr_factor(A, &qr);
    idx_t qr_rank = (qr_err == SPARSE_OK) ? qr.rank : -1;

    printf("    rank-deficient 6x4: SVD rank=%d, QR rank=%d\n", (int)svd_rank, (int)qr_rank);
    ASSERT_EQ(svd_rank, 2);
    ASSERT_EQ(svd_rank, qr_rank);

    if (qr_err == SPARSE_OK)
        sparse_qr_free(&qr);
    sparse_free(A);
}

/**
 * Pseudoinverse on SuiteSparse west0067: A * A^+ * A ≈ A.
 */
static void test_pinv_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    idx_t m = sparse_rows(A);
    idx_t nc = sparse_cols(A);

    double *pinv_data = NULL;
    ASSERT_ERR(sparse_pinv(A, 0.0, &pinv_data), SPARSE_OK);
    ASSERT_NOT_NULL(pinv_data);
    if (!pinv_data) {
        sparse_free(A);
        return;
    }

    /* Compute B = A * pinv (m×nc * nc×m = m×m) */
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

    /* Compute C = B * A and compare to A: ||A*A^+*A - A||_max */
    double max_err = 0.0;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++) {
            double sum = 0.0;
            for (idx_t p = 0; p < m; p++)
                sum += B[(size_t)p * (size_t)m + (size_t)i] * sparse_get(A, p, j);
            double err = fabs(sum - sparse_get(A, i, j));
            if (err > max_err)
                max_err = err;
        }
    printf("    west0067 pinv: ||A*A^+*A - A||_max = %.3e\n", max_err);
    ASSERT_TRUE(max_err < 1e-8);

    free(B);
    free(pinv_data);
    sparse_free(A);
}

/**
 * Low-rank approximation error bound on nos4.
 */
static void test_lowrank_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    idx_t n = sparse_rows(A);

    /* Full SVD for reference */
    sparse_svd_t svd;
    ASSERT_ERR(sparse_svd_compute(A, NULL, &svd), SPARSE_OK);

    /* Rank-10 approximation */
    idx_t rank_k = 10;
    double *lr = NULL;
    ASSERT_ERR(sparse_svd_lowrank(A, rank_k, &lr), SPARSE_OK);
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

    printf("    nos4 lowrank(%d): ||A-A_k||_F = %.6f, expected = %.6f\n", (int)rank_k, actual,
           expected);
    ASSERT_NEAR(actual, expected, 1e-8);

    free(lr);
    sparse_svd_free(&svd);
    sparse_free(A);
}

/**
 * Bidiagonalization + SVD round-trip on west0067 with economy UV.
 */
static void test_svd_reconstruction_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    idx_t m = sparse_rows(A);
    idx_t nc = sparse_cols(A);
    idx_t k = (m < nc) ? m : nc;

    sparse_svd_opts_t opts = {.compute_uv = 1, .economy = 1};
    sparse_svd_t svd;
    ASSERT_ERR(sparse_svd_compute(A, &opts, &svd), SPARSE_OK);
    ASSERT_NOT_NULL(svd.U);
    ASSERT_NOT_NULL(svd.Vt);

    /* Reconstruct A from U*S*Vt and measure error */
    double max_err = 0.0;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++) {
            double sum = 0.0;
            for (idx_t p = 0; p < k; p++)
                sum += svd.U[(size_t)p * (size_t)m + (size_t)i] * svd.sigma[p] *
                       svd.Vt[(size_t)j * (size_t)k + (size_t)p];
            double err = fabs(sum - sparse_get(A, i, j));
            if (err > max_err)
                max_err = err;
        }
    printf("    west0067 SVD reconstruction: ||U*S*Vt - A||_max = %.3e\n", max_err);
    ASSERT_TRUE(max_err < 1e-8);

    sparse_svd_free(&svd);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test suite
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sprint 8 Integration");

    RUN_TEST(test_svd_vs_ata_eigenvalues);
    RUN_TEST(test_mf_gmres_ilu_steam1);
    RUN_TEST(test_partial_vs_full_bcsstk04);
    RUN_TEST(test_svd_rank_matches_qr);
    RUN_TEST(test_pinv_west0067);
    RUN_TEST(test_lowrank_nos4);
    RUN_TEST(test_svd_reconstruction_west0067);

    TEST_SUITE_END();
}
