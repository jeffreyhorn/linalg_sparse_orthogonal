#include "sparse_dense.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Create / free tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_dense_create_basic(void) {
    dense_matrix_t *M = dense_create(3, 4);
    ASSERT_NOT_NULL(M);
    if (!M)
        return;
    ASSERT_EQ(M->rows, 3);
    ASSERT_EQ(M->cols, 4);
    ASSERT_NOT_NULL(M->data);

    /* Should be zero-initialized */
    for (idx_t j = 0; j < 4; j++)
        for (idx_t i = 0; i < 3; i++)
            ASSERT_NEAR(DENSE_AT(M, i, j), 0.0, 1e-15);

    dense_free(M);
}

static void test_dense_create_empty(void) {
    dense_matrix_t *M = dense_create(0, 0);
    ASSERT_NOT_NULL(M);
    if (!M)
        return;
    ASSERT_EQ(M->rows, 0);
    ASSERT_EQ(M->cols, 0);
    dense_free(M);
}

static void test_dense_create_invalid(void) {
    ASSERT_TRUE(dense_create(-1, 5) == NULL);
    ASSERT_TRUE(dense_create(5, -1) == NULL);
}

static void test_dense_free_null(void) { dense_free(NULL); /* should not crash */ }

/* ═══════════════════════════════════════════════════════════════════════
 * GEMM tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Identity * A = A */
static void test_dense_gemm_identity(void) {
    idx_t n = 3;
    dense_matrix_t *I = dense_create(n, n);
    dense_matrix_t *A = dense_create(n, n);
    dense_matrix_t *C = dense_create(n, n);
    ASSERT_NOT_NULL(I);
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(C);
    if (!I || !A || !C) {
        dense_free(I);
        dense_free(A);
        dense_free(C);
        return;
    }

    /* Set I = identity */
    for (idx_t i = 0; i < n; i++)
        DENSE_AT(I, i, i) = 1.0;

    /* Set A = [[1,2,3],[4,5,6],[7,8,9]] */
    double vals[] = {1, 4, 7, 2, 5, 8, 3, 6, 9}; /* column-major */
    for (idx_t k = 0; k < 9; k++)
        A->data[k] = vals[k];

    ASSERT_ERR(dense_gemm(I, A, C), SPARSE_OK);

    for (idx_t k = 0; k < 9; k++)
        ASSERT_NEAR(C->data[k], A->data[k], 1e-15);

    dense_free(I);
    dense_free(A);
    dense_free(C);
}

/* 2×3 * 3×2 = 2×2 known product */
static void test_dense_gemm_rect(void) {
    dense_matrix_t *A = dense_create(2, 3);
    dense_matrix_t *B = dense_create(3, 2);
    dense_matrix_t *C = dense_create(2, 2);
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    ASSERT_NOT_NULL(C);
    if (!A || !B || !C) {
        dense_free(A);
        dense_free(B);
        dense_free(C);
        return;
    }

    /* A = [[1,2,3],[4,5,6]] (row-major) → column-major */
    DENSE_AT(A, 0, 0) = 1.0;
    DENSE_AT(A, 0, 1) = 2.0;
    DENSE_AT(A, 0, 2) = 3.0;
    DENSE_AT(A, 1, 0) = 4.0;
    DENSE_AT(A, 1, 1) = 5.0;
    DENSE_AT(A, 1, 2) = 6.0;

    /* B = [[7,8],[9,10],[11,12]] */
    DENSE_AT(B, 0, 0) = 7.0;
    DENSE_AT(B, 0, 1) = 8.0;
    DENSE_AT(B, 1, 0) = 9.0;
    DENSE_AT(B, 1, 1) = 10.0;
    DENSE_AT(B, 2, 0) = 11.0;
    DENSE_AT(B, 2, 1) = 12.0;

    ASSERT_ERR(dense_gemm(A, B, C), SPARSE_OK);

    /* C = A*B = [[58,64],[139,154]] */
    ASSERT_NEAR(DENSE_AT(C, 0, 0), 58.0, 1e-12);
    ASSERT_NEAR(DENSE_AT(C, 0, 1), 64.0, 1e-12);
    ASSERT_NEAR(DENSE_AT(C, 1, 0), 139.0, 1e-12);
    ASSERT_NEAR(DENSE_AT(C, 1, 1), 154.0, 1e-12);

    dense_free(A);
    dense_free(B);
    dense_free(C);
}

/* Dimension mismatch */
static void test_dense_gemm_shape_error(void) {
    dense_matrix_t *A = dense_create(2, 3);
    dense_matrix_t *B = dense_create(2, 2); /* A.cols=3 != B.rows=2 */
    dense_matrix_t *C = dense_create(2, 2);
    if (!A || !B || !C) {
        dense_free(A);
        dense_free(B);
        dense_free(C);
        return;
    }

    ASSERT_ERR(dense_gemm(A, B, C), SPARSE_ERR_SHAPE);

    dense_free(A);
    dense_free(B);
    dense_free(C);
}

/* NULL inputs */
static void test_dense_gemm_null(void) {
    dense_matrix_t *A = dense_create(2, 2);
    dense_matrix_t *C = dense_create(2, 2);
    if (A && C) {
        ASSERT_ERR(dense_gemm(NULL, A, C), SPARSE_ERR_NULL);
        ASSERT_ERR(dense_gemm(A, NULL, C), SPARSE_ERR_NULL);
        ASSERT_ERR(dense_gemm(A, A, NULL), SPARSE_ERR_NULL);
    }
    dense_free(A);
    dense_free(C);
}

/* ═══════════════════════════════════════════════════════════════════════
 * GEMV tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Identity * x = x */
static void test_dense_gemv_identity(void) {
    idx_t n = 4;
    dense_matrix_t *I = dense_create(n, n);
    ASSERT_NOT_NULL(I);
    if (!I)
        return;
    for (idx_t i = 0; i < n; i++)
        DENSE_AT(I, i, i) = 1.0;

    double x[] = {1.0, 2.0, 3.0, 4.0};
    double y[4];
    ASSERT_ERR(dense_gemv(I, x, y), SPARSE_OK);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(y[i], x[i], 1e-15);

    dense_free(I);
}

/* Known 2×3 * 3-vector */
static void test_dense_gemv_rect(void) {
    dense_matrix_t *A = dense_create(2, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    /* A = [[1,2,3],[4,5,6]] */
    DENSE_AT(A, 0, 0) = 1.0;
    DENSE_AT(A, 0, 1) = 2.0;
    DENSE_AT(A, 0, 2) = 3.0;
    DENSE_AT(A, 1, 0) = 4.0;
    DENSE_AT(A, 1, 1) = 5.0;
    DENSE_AT(A, 1, 2) = 6.0;

    double x[] = {1.0, 2.0, 3.0};
    double y[2];
    ASSERT_ERR(dense_gemv(A, x, y), SPARSE_OK);

    /* y = A*x = [14, 32] */
    ASSERT_NEAR(y[0], 14.0, 1e-12);
    ASSERT_NEAR(y[1], 32.0, 1e-12);

    dense_free(A);
}

/* NULL inputs */
static void test_dense_gemv_null(void) {
    dense_matrix_t *A = dense_create(2, 2);
    double x[2] = {1.0, 2.0};
    double y[2];
    if (A) {
        ASSERT_ERR(dense_gemv(NULL, x, y), SPARSE_ERR_NULL);
        ASSERT_ERR(dense_gemv(A, NULL, y), SPARSE_ERR_NULL);
        ASSERT_ERR(dense_gemv(A, x, NULL), SPARSE_ERR_NULL);
    }
    dense_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * DENSE_AT macro test
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_dense_at_layout(void) {
    dense_matrix_t *M = dense_create(3, 2);
    ASSERT_NOT_NULL(M);
    if (!M)
        return;

    /* Set elements and verify column-major layout */
    DENSE_AT(M, 0, 0) = 1.0;
    DENSE_AT(M, 1, 0) = 2.0;
    DENSE_AT(M, 2, 0) = 3.0;
    DENSE_AT(M, 0, 1) = 4.0;
    DENSE_AT(M, 1, 1) = 5.0;
    DENSE_AT(M, 2, 1) = 6.0;

    /* Column-major: data should be [1,2,3,4,5,6] */
    ASSERT_NEAR(M->data[0], 1.0, 1e-15);
    ASSERT_NEAR(M->data[1], 2.0, 1e-15);
    ASSERT_NEAR(M->data[2], 3.0, 1e-15);
    ASSERT_NEAR(M->data[3], 4.0, 1e-15);
    ASSERT_NEAR(M->data[4], 5.0, 1e-15);
    ASSERT_NEAR(M->data[5], 6.0, 1e-15);

    dense_free(M);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Givens rotation tests (Sprint 7 Day 5)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Givens zeroes second component */
static void test_givens_basic(void) {
    double c, s;
    givens_compute(3.0, 4.0, &c, &s);

    /* [c s; -s c]^T * [3; 4] = [5; 0] */
    double r0 = c * 3.0 + s * 4.0;
    double r1 = -s * 3.0 + c * 4.0;
    ASSERT_NEAR(r0, 5.0, 1e-14);
    ASSERT_NEAR(r1, 0.0, 1e-14);
}

/* b=0: no rotation needed */
static void test_givens_b_zero(void) {
    double c, s;
    givens_compute(5.0, 0.0, &c, &s);
    ASSERT_NEAR(c, 1.0, 1e-15);
    ASSERT_NEAR(s, 0.0, 1e-15);
}

/* a=0: swap */
static void test_givens_a_zero(void) {
    double c, s;
    givens_compute(0.0, 7.0, &c, &s);
    ASSERT_NEAR(c, 0.0, 1e-15);
    ASSERT_NEAR(s, 1.0, 1e-15);

    double r0 = c * 0.0 + s * 7.0;
    double r1 = -s * 0.0 + c * 7.0;
    ASSERT_NEAR(r0, 7.0, 1e-14);
    ASSERT_NEAR(r1, 0.0, 1e-14);
}

/* Both zero */
static void test_givens_both_zero(void) {
    double c, s;
    givens_compute(0.0, 0.0, &c, &s);
    ASSERT_NEAR(c, 1.0, 1e-15);
    ASSERT_NEAR(s, 0.0, 1e-15);
}

/* Rotation preserves norm */
static void test_givens_norm_preserving(void) {
    double c, s;
    givens_compute(-2.0, 3.0, &c, &s);

    double x[3] = {1.0, 2.0, 3.0};
    double y[3] = {4.0, 5.0, 6.0};

    /* Compute norms before */
    double norm_before = 0.0;
    for (int i = 0; i < 3; i++)
        norm_before += x[i] * x[i] + y[i] * y[i];

    givens_apply_left(c, s, x, y, 3);

    /* Compute norms after */
    double norm_after = 0.0;
    for (int i = 0; i < 3; i++)
        norm_after += x[i] * x[i] + y[i] * y[i];

    ASSERT_NEAR(norm_before, norm_after, 1e-12);
}

/* apply_left zeroes target in a 2-element "row" */
static void test_givens_apply_left(void) {
    double c, s;
    givens_compute(3.0, 4.0, &c, &s);

    double x[1] = {3.0};
    double y[1] = {4.0};
    givens_apply_left(c, s, x, y, 1);
    ASSERT_NEAR(x[0], 5.0, 1e-14);
    ASSERT_NEAR(y[0], 0.0, 1e-14);
}

/* Negative values */
static void test_givens_negative(void) {
    double c, s;
    givens_compute(-3.0, -4.0, &c, &s);

    double r0 = c * (-3.0) + s * (-4.0);
    double r1 = -s * (-3.0) + c * (-4.0);
    ASSERT_NEAR(fabs(r0), 5.0, 1e-14);
    ASSERT_NEAR(r1, 0.0, 1e-14);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 2×2 eigenvalue tests (Sprint 7 Day 5)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Diagonal matrix: eigenvalues = diagonal entries */
static void test_eigen2x2_diagonal(void) {
    double l1, l2;
    eigen2x2(3.0, 0.0, 7.0, &l1, &l2);
    ASSERT_NEAR(l1, 3.0, 1e-14);
    ASSERT_NEAR(l2, 7.0, 1e-14);
}

/* Known symmetric: [[2, 1], [1, 2]] → eigenvalues 1, 3 */
static void test_eigen2x2_known(void) {
    double l1, l2;
    eigen2x2(2.0, 1.0, 2.0, &l1, &l2);
    ASSERT_NEAR(l1, 1.0, 1e-14);
    ASSERT_NEAR(l2, 3.0, 1e-14);
}

/* Identity: eigenvalues both 1 */
static void test_eigen2x2_identity(void) {
    double l1, l2;
    eigen2x2(1.0, 0.0, 1.0, &l1, &l2);
    ASSERT_NEAR(l1, 1.0, 1e-14);
    ASSERT_NEAR(l2, 1.0, 1e-14);
}

/* Nearly equal eigenvalues */
static void test_eigen2x2_nearly_equal(void) {
    double l1, l2;
    eigen2x2(1.0, 1e-10, 1.0, &l1, &l2);
    ASSERT_NEAR(l1, 1.0 - 1e-10, 1e-14);
    ASSERT_NEAR(l2, 1.0 + 1e-10, 1e-14);
}

/* Large values */
static void test_eigen2x2_large(void) {
    double l1, l2;
    eigen2x2(1e10, 1.0, 1e10, &l1, &l2);
    ASSERT_NEAR(l1, 1e10 - 1.0, 1e-4);
    ASSERT_NEAR(l2, 1e10 + 1.0, 1e-4);
}

/* Negative eigenvalues: [[-5, 2], [2, -5]] → -7, -3 */
static void test_eigen2x2_negative(void) {
    double l1, l2;
    eigen2x2(-5.0, 2.0, -5.0, &l1, &l2);
    ASSERT_NEAR(l1, -7.0, 1e-14);
    ASSERT_NEAR(l2, -3.0, 1e-14);
}

/* Zero off-diagonal */
static void test_eigen2x2_zero_offdiag(void) {
    double l1, l2;
    eigen2x2(5.0, 0.0, 2.0, &l1, &l2);
    ASSERT_NEAR(l1, 2.0, 1e-14);
    ASSERT_NEAR(l2, 5.0, 1e-14);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Tridiagonal QR eigenvalue tests (Sprint 7 Day 12)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Diagonal matrix: eigenvalues = diagonal entries */
static void test_tridiag_diagonal(void) {
    double diag[] = {5.0, 3.0, 7.0, 1.0};
    double sub[] = {0.0, 0.0, 0.0};
    ASSERT_ERR(tridiag_qr_eigenvalues(diag, sub, 4, 0), SPARSE_OK);
    /* Should be sorted ascending */
    ASSERT_NEAR(diag[0], 1.0, 1e-14);
    ASSERT_NEAR(diag[1], 3.0, 1e-14);
    ASSERT_NEAR(diag[2], 5.0, 1e-14);
    ASSERT_NEAR(diag[3], 7.0, 1e-14);
}

/* 2×2 tridiagonal: should match eigen2x2 */
static void test_tridiag_2x2(void) {
    double diag[] = {2.0, 5.0};
    double sub[] = {1.0};
    ASSERT_ERR(tridiag_qr_eigenvalues(diag, sub, 2, 0), SPARSE_OK);

    double l1, l2;
    eigen2x2(2.0, 1.0, 5.0, &l1, &l2);
    ASSERT_NEAR(diag[0], l1, 1e-12);
    ASSERT_NEAR(diag[1], l2, 1e-12);
}

/* Known tridiagonal: -1, 2, -1 of size n
 * Eigenvalues: 2 - 2*cos(k*pi/(n+1)) for k=1..n */
static void test_tridiag_known(void) {
    idx_t n = 10;
    double diag[10], sub[9];
    for (idx_t i = 0; i < n; i++)
        diag[i] = 2.0;
    for (idx_t i = 0; i < n - 1; i++)
        sub[i] = -1.0;

    ASSERT_ERR(tridiag_qr_eigenvalues(diag, sub, n, 0), SPARSE_OK);

    /* Compare with analytical eigenvalues */
    double pi = 3.14159265358979323846;
    for (idx_t k = 0; k < n; k++) {
        double expected = 2.0 - 2.0 * cos((double)(k + 1) * pi / (double)(n + 1));
        ASSERT_NEAR(diag[k], expected, 1e-10);
    }
}

/* 1×1 matrix */
static void test_tridiag_1x1(void) {
    double diag[] = {42.0};
    ASSERT_ERR(tridiag_qr_eigenvalues(diag, NULL, 1, 0), SPARSE_OK);
    ASSERT_NEAR(diag[0], 42.0, 1e-15);
}

/* n=0 */
static void test_tridiag_empty(void) {
    ASSERT_ERR(tridiag_qr_eigenvalues(NULL, NULL, 0, 0), SPARSE_OK);
}

/* Larger: n=50, -1,2,-1 tridiagonal */
static void test_tridiag_large(void) {
    idx_t n = 50;
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

    /* Verify sorted ascending */
    for (idx_t i = 1; i < n; i++)
        ASSERT_TRUE(diag[i] >= diag[i - 1] - 1e-14);

    /* Spot-check smallest and largest */
    double pi = 3.14159265358979323846;
    double lam_min = 2.0 - 2.0 * cos(pi / (double)(n + 1));
    double lam_max = 2.0 - 2.0 * cos((double)n * pi / (double)(n + 1));
    printf("    tridiag n=50: lam_min=%.6e (expected %.6e), lam_max=%.6e (expected %.6e)\n",
           diag[0], lam_min, diag[n - 1], lam_max);
    ASSERT_NEAR(diag[0], lam_min, 1e-10);
    ASSERT_NEAR(diag[n - 1], lam_max, 1e-10);

    free(diag);
    free(sub);
}

/* Graded diagonal: eigenvalues spanning many orders of magnitude */
static void test_tridiag_graded(void) {
    double diag[] = {1e-8, 1.0, 1e4, 1e8};
    double sub[] = {1e-10, 1e-5, 1e-2};
    ASSERT_ERR(tridiag_qr_eigenvalues(diag, sub, 4, 0), SPARSE_OK);

    /* Should be sorted and roughly match diagonal entries (off-diag is small) */
    ASSERT_TRUE(diag[0] < diag[1]);
    ASSERT_TRUE(diag[1] < diag[2]);
    ASSERT_TRUE(diag[2] < diag[3]);
    ASSERT_NEAR(diag[0], 1e-8, 1e-6);
    ASSERT_NEAR(diag[3], 1e8, 1e2);
}

/* NULL inputs for n>1 */
static void test_tridiag_null(void) {
    ASSERT_ERR(tridiag_qr_eigenvalues(NULL, NULL, 5, 0), SPARSE_ERR_NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Dense Matrix Utilities");

    /* Create / free */
    RUN_TEST(test_dense_create_basic);
    RUN_TEST(test_dense_create_empty);
    RUN_TEST(test_dense_create_invalid);
    RUN_TEST(test_dense_free_null);

    /* GEMM */
    RUN_TEST(test_dense_gemm_identity);
    RUN_TEST(test_dense_gemm_rect);
    RUN_TEST(test_dense_gemm_shape_error);
    RUN_TEST(test_dense_gemm_null);

    /* GEMV */
    RUN_TEST(test_dense_gemv_identity);
    RUN_TEST(test_dense_gemv_rect);
    RUN_TEST(test_dense_gemv_null);

    /* Layout */
    RUN_TEST(test_dense_at_layout);

    /* Givens rotations (Sprint 7 Day 5) */
    RUN_TEST(test_givens_basic);
    RUN_TEST(test_givens_b_zero);
    RUN_TEST(test_givens_a_zero);
    RUN_TEST(test_givens_both_zero);
    RUN_TEST(test_givens_norm_preserving);
    RUN_TEST(test_givens_apply_left);
    RUN_TEST(test_givens_negative);

    /* 2×2 eigenvalue solver (Sprint 7 Day 5) */
    RUN_TEST(test_eigen2x2_diagonal);
    RUN_TEST(test_eigen2x2_known);
    RUN_TEST(test_eigen2x2_identity);
    RUN_TEST(test_eigen2x2_nearly_equal);
    RUN_TEST(test_eigen2x2_large);
    RUN_TEST(test_eigen2x2_negative);
    RUN_TEST(test_eigen2x2_zero_offdiag);

    /* Tridiagonal QR eigenvalue solver (Sprint 7 Day 12) */
    RUN_TEST(test_tridiag_diagonal);
    RUN_TEST(test_tridiag_2x2);
    RUN_TEST(test_tridiag_known);
    RUN_TEST(test_tridiag_1x1);
    RUN_TEST(test_tridiag_empty);
    RUN_TEST(test_tridiag_large);
    RUN_TEST(test_tridiag_graded);
    RUN_TEST(test_tridiag_null);

    TEST_SUITE_END();
}
