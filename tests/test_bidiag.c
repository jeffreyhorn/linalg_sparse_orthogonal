#include "sparse_bidiag.h"
#include "sparse_dense.h"
#include "sparse_matrix.h"
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
 * Compute ||A - U*B*V^T|| where U/V are applied via Householder sequences
 * and B is bidiagonal. Uses explicit dense reconstruction.
 */
static double bidiag_reconstruction_error(const SparseMatrix *A, const sparse_bidiag_t *bd) {
    idx_t m = bd->m;
    idx_t n = bd->n;
    idx_t k = (m < n) ? m : n;
    if (k == 0)
        return 0.0;

    /* Form dense B (m × n, upper bidiagonal) */
    double *B = calloc((size_t)m * (size_t)n, sizeof(double));
    if (!B)
        return INFINITY;
    for (idx_t i = 0; i < k; i++)
        B[(size_t)i * (size_t)m + (size_t)i] = bd->diag[i]; /* col-major */
    for (idx_t i = 0; i < k - 1; i++)
        B[(size_t)(i + 1) * (size_t)m + (size_t)i] = bd->superdiag[i];

    /* Apply U to columns of B: U*B — apply left Householder reflectors
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
            return INFINITY;
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

    double recon = bidiag_reconstruction_error(A, &bd);
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

    double recon = bidiag_reconstruction_error(A, &bd);
    printf("    10x5 bidiag recon: %.3e\n", recon);
    ASSERT_TRUE(recon < 1e-10);

    sparse_bidiag_free(&bd);
    sparse_free(A);
}

/* Wide rectangular 5×10 — skipped, behavior is undefined for m < n per API docs */
static void test_bidiag_wide(void) {
    SparseMatrix *A = sparse_create(5, 10);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 5; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    sparse_bidiag_t bd;
    ASSERT_ERR(sparse_bidiag_factor(A, &bd), SPARSE_ERR_SHAPE);
    printf("    5x10 bidiag: correctly rejected with SPARSE_ERR_SHAPE\n");

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
    double recon = bidiag_reconstruction_error(A, &bd);
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

    double recon = bidiag_reconstruction_error(A, &bd);
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
