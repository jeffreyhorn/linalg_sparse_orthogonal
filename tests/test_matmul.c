#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════
 * SpMM tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Known 2x2: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]] */
static void test_matmul_2x2(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0); sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 3.0); sparse_insert(A, 1, 1, 4.0);

    SparseMatrix *B = sparse_create(2, 2);
    sparse_insert(B, 0, 0, 5.0); sparse_insert(B, 0, 1, 6.0);
    sparse_insert(B, 1, 0, 7.0); sparse_insert(B, 1, 1, 8.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);

    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 19.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 1), 22.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 0), 43.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 50.0, 0.0);

    sparse_free(A); sparse_free(B); sparse_free(C);
}

/* I * A = A */
static void test_matmul_identity_left(void)
{
    idx_t n = 4;
    SparseMatrix *I = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(I, i, i, 1.0);

    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 2.0); sparse_insert(A, 0, 3, 5.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 0, 1.0); sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 3, 7.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(I, A, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), sparse_nnz(A));

    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 3), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 2), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 3, 3), 7.0, 0.0);

    sparse_free(I); sparse_free(A); sparse_free(C);
}

/* A * I = A */
static void test_matmul_identity_right(void)
{
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 1.0); sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 0, 7.0);

    SparseMatrix *I = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(I, i, i, 1.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, I, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), sparse_nnz(A));

    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 2), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 0), 7.0, 0.0);

    sparse_free(A); sparse_free(I); sparse_free(C);
}

/* Diagonal * A = scaled rows */
static void test_matmul_diag_left(void)
{
    SparseMatrix *D = sparse_create(2, 2);
    sparse_insert(D, 0, 0, 2.0);
    sparse_insert(D, 1, 1, 3.0);

    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0); sparse_insert(A, 0, 1, 4.0);
    sparse_insert(A, 1, 0, 5.0); sparse_insert(A, 1, 1, 6.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(D, A, &C), SPARSE_OK);

    /* Row 0 scaled by 2, row 1 scaled by 3 */
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 1), 8.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 0), 15.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 18.0, 0.0);

    sparse_free(D); sparse_free(A); sparse_free(C);
}

/* A * Diagonal = scaled columns */
static void test_matmul_diag_right(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0); sparse_insert(A, 0, 1, 4.0);
    sparse_insert(A, 1, 0, 5.0); sparse_insert(A, 1, 1, 6.0);

    SparseMatrix *D = sparse_create(2, 2);
    sparse_insert(D, 0, 0, 2.0);
    sparse_insert(D, 1, 1, 3.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, D, &C), SPARSE_OK);

    /* Col 0 scaled by 2, col 1 scaled by 3 */
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 1), 12.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 0), 10.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 18.0, 0.0);

    sparse_free(A); sparse_free(D); sparse_free(C);
}

/* Dimension mismatch → SPARSE_ERR_SHAPE */
static void test_matmul_shape_mismatch(void)
{
    SparseMatrix *A = sparse_create(2, 3);
    SparseMatrix *B = sparse_create(4, 2);  /* 3 != 4 */
    SparseMatrix *C;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_ERR_SHAPE);
    sparse_free(A); sparse_free(B);
}

/* NULL inputs → SPARSE_ERR_NULL */
static void test_matmul_null(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    SparseMatrix *C;
    ASSERT_ERR(sparse_matmul(NULL, A, &C), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_matmul(A, NULL, &C), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_matmul(A, A, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

/* Rectangular: (3x2) * (2x4) → (3x4) */
static void test_matmul_rectangular(void)
{
    SparseMatrix *A = sparse_create(3, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 0, 3.0); sparse_insert(A, 2, 1, 4.0);

    SparseMatrix *B = sparse_create(2, 4);
    sparse_insert(B, 0, 0, 1.0); sparse_insert(B, 0, 2, 2.0);
    sparse_insert(B, 1, 1, 3.0); sparse_insert(B, 1, 3, 4.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    ASSERT_EQ(sparse_rows(C), 3);
    ASSERT_EQ(sparse_cols(C), 4);

    /* C(0,:) = 1*row0(B) = [1, 0, 2, 0] */
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 2), 2.0, 0.0);
    /* C(1,:) = 2*row1(B) = [0, 6, 0, 8] */
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 6.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 3), 8.0, 0.0);
    /* C(2,:) = 3*row0(B) + 4*row1(B) = [3, 12, 6, 16] */
    ASSERT_NEAR(sparse_get_phys(C, 2, 0), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 1), 12.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 2), 6.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 3), 16.0, 0.0);

    sparse_free(A); sparse_free(B); sparse_free(C);
}

/* Cancellation: A * B where entries cancel to zero → no spurious nnz */
static void test_matmul_cancellation(void)
{
    /* A = [[1, 1]], B = [[1], [-1]] → C = [[0]] */
    SparseMatrix *A = sparse_create(1, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);

    SparseMatrix *B = sparse_create(2, 1);
    sparse_insert(B, 0, 0, 1.0);
    sparse_insert(B, 1, 0, -1.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), 0);  /* cancellation → no entry */

    sparse_free(A); sparse_free(B); sparse_free(C);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Sparse Matrix-Matrix Multiply Tests");

    RUN_TEST(test_matmul_2x2);
    RUN_TEST(test_matmul_identity_left);
    RUN_TEST(test_matmul_identity_right);
    RUN_TEST(test_matmul_diag_left);
    RUN_TEST(test_matmul_diag_right);
    RUN_TEST(test_matmul_shape_mismatch);
    RUN_TEST(test_matmul_null);
    RUN_TEST(test_matmul_rectangular);
    RUN_TEST(test_matmul_cancellation);

    TEST_SUITE_END();
}
