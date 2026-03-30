#include "sparse_dense.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>

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

static void test_dense_free_null(void) {
    dense_free(NULL); /* should not crash */
}

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

    TEST_SUITE_END();
}
