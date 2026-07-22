#include "sparse_bidiag.h"
#include "sparse_dense.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_bidiag_helpers.h"
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
 * Bidiagonal reduction tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Square 3×3: verify bidiagonal structure and reconstruction */
static void test_bidiag_3x3(void) {
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

    ASSERT_EQ(bd.m, 3);
    ASSERT_EQ(bd.n, 3);
    ASSERT_NOT_NULL(bd.diag);
    ASSERT_NOT_NULL(bd.superdiag);

    double recon = tf_bidiag_reconstruction_max_error(A, &bd);
    printf("    3x3 bidiag recon: %.3e\n", recon);
    ASSERT_TRUE(recon < 1e-10);

    sparse_bidiag_free(&bd);
    sparse_free(A);
}

/* Tall rectangular 10×5 */
static void test_bidiag_tall(void) {
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

    double recon = tf_bidiag_reconstruction_max_error(A, &bd);
    printf("    10x5 bidiag recon: %.3e\n", recon);
    ASSERT_TRUE(recon < 1e-10);

    sparse_bidiag_free(&bd);
    sparse_free(A);
}

/* Wide rectangular 5×10 — handled via internal transpose */
static void test_bidiag_wide(void) {
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

    ASSERT_TRUE(bd.transposed);
    double recon = tf_bidiag_reconstruction_max_error(A, &bd);
    printf("    5x10 bidiag recon: %.3e\n", recon);
    ASSERT_TRUE(recon < 1e-10);

    sparse_bidiag_free(&bd);
    sparse_free(A);
}

/* Wide 3×8 */
static void test_bidiag_wide_3x8(void) {
    SparseMatrix *A = sparse_create(3, 8);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 0, 5, 2.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 1, 6, 3.0);
    sparse_insert(A, 2, 2, 5.0);
    sparse_insert(A, 2, 4, 1.0);
    sparse_insert(A, 2, 7, 2.0);

    sparse_bidiag_t bd;
    sparse_err_t err = sparse_bidiag_factor(A, &bd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double recon = tf_bidiag_reconstruction_max_error(A, &bd);
    printf("    3x8 bidiag recon: %.3e\n", recon);
    ASSERT_TRUE(recon < 1e-10);

    sparse_bidiag_free(&bd);
    sparse_free(A);
}

/* Single-row 1×5 */
static void test_bidiag_single_row(void) {
    SparseMatrix *A = sparse_create(1, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 0, 2, 4.0);

    sparse_bidiag_t bd;
    sparse_err_t err = sparse_bidiag_factor(A, &bd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_TRUE(bd.transposed);
    double recon = tf_bidiag_reconstruction_max_error(A, &bd);
    printf("    1x5 bidiag recon: %.3e\n", recon);
    ASSERT_TRUE(recon < 1e-10);

    sparse_bidiag_free(&bd);
    sparse_free(A);
}

/* Diagonal matrix: superdiag should be ~0 */
static void test_bidiag_diagonal(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    sparse_bidiag_t bd;
    sparse_err_t err = sparse_bidiag_factor(A, &bd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Superdiagonal should be zero for a diagonal matrix */
    for (idx_t i = 0; i < n - 1; i++)
        ASSERT_NEAR(bd.superdiag[i], 0.0, 1e-12);

    /* Diagonal entries should have same magnitude as original (order may differ) */
    double recon = tf_bidiag_reconstruction_max_error(A, &bd);
    ASSERT_TRUE(recon < 1e-10);

    sparse_bidiag_free(&bd);
    sparse_free(A);
}

/* 1×1 matrix */
static void test_bidiag_1x1(void) {
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

    ASSERT_NEAR(fabs(bd.diag[0]), 7.0, 1e-12);

    sparse_bidiag_free(&bd);
    sparse_free(A);
}

/* NULL inputs */
static void test_bidiag_null(void) {
    sparse_bidiag_t bd;
    ASSERT_ERR(sparse_bidiag_factor(NULL, &bd), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_bidiag_factor(NULL, NULL), SPARSE_ERR_NULL);
}

/* nos4 (100×100 SPD): reconstruction error */
static void test_bidiag_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    sparse_bidiag_t bd;
    sparse_err_t err = sparse_bidiag_factor(A, &bd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double recon = tf_bidiag_reconstruction_max_error(A, &bd);
    printf("    nos4 bidiag recon: %.3e\n", recon);
    ASSERT_TRUE(recon < 1e-8);

    sparse_bidiag_free(&bd);
    sparse_free(A);
}

/* Free on zeroed struct */
static void test_bidiag_free_zeroed(void) {
    sparse_bidiag_t bd;
    memset(&bd, 0, sizeof(bd));
    sparse_bidiag_free(&bd); /* should not crash */
}

/* ═══════════════════════════════════════════════════════════════════════
 * Tridiagonal QR hardening tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* n=100 tridiagonal: verify convergence */
static void test_tridiag_n100(void) {
    idx_t n = 100;
    double *diag = malloc((size_t)n * sizeof(double));
    double *sub = malloc((size_t)(n - 1) * sizeof(double));
    ASSERT_NOT_NULL(diag);
    ASSERT_NOT_NULL(sub);
    if (!diag || !sub) {
        free(diag);
        free(sub);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        diag[i] = 2.0;
    for (idx_t i = 0; i < n - 1; i++)
        sub[i] = -1.0;

    ASSERT_ERR(tridiag_qr_eigenvalues(diag, sub, n, 0), SPARSE_OK);

    /* Verify sorted */
    for (idx_t i = 1; i < n; i++)
        ASSERT_TRUE(diag[i] >= diag[i - 1] - 1e-14);

    /* Spot-check bounds: all eigenvalues in (0, 4) for this tridiag */
    ASSERT_TRUE(diag[0] > 0.0);
    ASSERT_TRUE(diag[n - 1] < 4.0);

    double pi = 3.14159265358979323846;
    double expected_min = 2.0 - 2.0 * cos(pi / (double)(n + 1));
    printf("    tridiag n=100: lam_min=%.6e (expected %.6e)\n", diag[0], expected_min);
    ASSERT_NEAR(diag[0], expected_min, 1e-10);

    free(diag);
    free(sub);
}

/* Clustered eigenvalues: all diagonal = 1, tiny off-diagonal */
static void test_tridiag_clustered(void) {
    idx_t n = 20;
    double *diag = malloc((size_t)n * sizeof(double));
    double *sub = malloc((size_t)(n - 1) * sizeof(double));
    ASSERT_NOT_NULL(diag);
    ASSERT_NOT_NULL(sub);
    if (!diag || !sub) {
        free(diag);
        free(sub);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        diag[i] = 1.0;
    for (idx_t i = 0; i < n - 1; i++)
        sub[i] = 1e-8;

    ASSERT_ERR(tridiag_qr_eigenvalues(diag, sub, n, 0), SPARSE_OK);

    /* All eigenvalues should be near 1 */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(diag[i], 1.0, 1e-6);

    free(diag);
    free(sub);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Bidiagonal Reduction & Tridiagonal QR Hardening");

    /* Bidiagonal reduction */
    RUN_TEST(test_bidiag_3x3);
    RUN_TEST(test_bidiag_tall);
    RUN_TEST(test_bidiag_wide);
    RUN_TEST(test_bidiag_wide_3x8);
    RUN_TEST(test_bidiag_single_row);
    RUN_TEST(test_bidiag_diagonal);
    RUN_TEST(test_bidiag_1x1);
    RUN_TEST(test_bidiag_null);
    RUN_TEST(test_bidiag_nos4);
    RUN_TEST(test_bidiag_free_zeroed);

    /* Tridiagonal QR hardening */
    RUN_TEST(test_tridiag_n100);
    RUN_TEST(test_tridiag_clustered);

    TEST_SUITE_END();
}
