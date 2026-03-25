#include "sparse_matrix.h"
#include "sparse_cholesky.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Cholesky factorization tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* 2x2 SPD: A = [[4, 2], [2, 3]] → L = [[2, 0], [1, sqrt(2)]] */
static void test_cholesky_2x2(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 1, 1, 3.0);

    ASSERT_ERR(sparse_cholesky_factor(A), SPARSE_OK);

    /* Verify L values */
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 2.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(A, 1, 0), 1.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), sqrt(2.0), 1e-14);
    /* Upper triangle should be gone */
    ASSERT_NEAR(sparse_get_phys(A, 0, 1), 0.0, 0.0);

    sparse_free(A);
}

/* 3x3 SPD tridiagonal: A = [[4,-1,0],[-1,4,-1],[0,-1,4]] */
static void test_cholesky_3x3_tridiag(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);  sparse_insert(A, 0, 1, -1.0);
    sparse_insert(A, 1, 0, -1.0); sparse_insert(A, 1, 1, 4.0);  sparse_insert(A, 1, 2, -1.0);
    sparse_insert(A, 2, 1, -1.0); sparse_insert(A, 2, 2, 4.0);

    ASSERT_ERR(sparse_cholesky_factor(A), SPARSE_OK);

    /* L should be lower triangular */
    ASSERT_NEAR(sparse_get_phys(A, 0, 1), 0.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 0, 2), 0.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 2), 0.0, 0.0);

    /* L(0,0) = sqrt(4) = 2 */
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 2.0, 1e-14);
    /* L(1,0) = -1/2 = -0.5 */
    ASSERT_NEAR(sparse_get_phys(A, 1, 0), -0.5, 1e-14);
    /* L(1,1) = sqrt(4 - (-0.5)^2) = sqrt(3.75) */
    double l11 = sqrt(4.0 - 0.25);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), l11, 1e-14);

    sparse_free(A);
}

/* Non-SPD matrix → SPARSE_ERR_NOT_SPD */
static void test_cholesky_not_spd(void)
{
    /* A = [[1, 2], [2, 1]] — symmetric but not positive definite (eigenvalues 3, -1) */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 1, 1, 1.0);

    ASSERT_ERR(sparse_cholesky_factor(A), SPARSE_ERR_NOT_SPD);
    sparse_free(A);
}

/* Non-square → SPARSE_ERR_SHAPE */
static void test_cholesky_nonsquare(void)
{
    SparseMatrix *A = sparse_create(2, 3);
    ASSERT_ERR(sparse_cholesky_factor(A), SPARSE_ERR_SHAPE);
    sparse_free(A);
}

/* NULL → SPARSE_ERR_NULL */
static void test_cholesky_null(void)
{
    ASSERT_ERR(sparse_cholesky_factor(NULL), SPARSE_ERR_NULL);
}

/* 1x1 SPD */
static void test_cholesky_1x1(void)
{
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 9.0);
    ASSERT_ERR(sparse_cholesky_factor(A), SPARSE_OK);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 3.0, 1e-14);
    sparse_free(A);
}

/* Identity → L = I */
static void test_cholesky_identity(void)
{
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    ASSERT_ERR(sparse_cholesky_factor(A), SPARSE_OK);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(sparse_get_phys(A, i, i), 1.0, 1e-14);

    sparse_free(A);
}

/* Diagonal SPD → L = diag(sqrt(d_ii)) */
static void test_cholesky_diagonal(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 1, 1, 9.0);
    sparse_insert(A, 2, 2, 16.0);

    ASSERT_ERR(sparse_cholesky_factor(A), SPARSE_OK);

    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 2.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), 3.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(A, 2, 2), 4.0, 1e-14);
    ASSERT_EQ(sparse_nnz(A), 3);

    sparse_free(A);
}

/* Verify L*L^T ≈ A using matvec */
static void test_cholesky_reconstruct(void)
{
    /* A = [[10, 1, 2], [1, 10, 1], [2, 1, 10]] */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 10.0); sparse_insert(A, 0, 1, 1.0); sparse_insert(A, 0, 2, 2.0);
    sparse_insert(A, 1, 0, 1.0);  sparse_insert(A, 1, 1, 10.0); sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 0, 2.0);  sparse_insert(A, 2, 1, 1.0); sparse_insert(A, 2, 2, 10.0);

    /* Save original entries */
    double orig[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            orig[i][j] = sparse_get_phys(A, (idx_t)i, (idx_t)j);

    ASSERT_ERR(sparse_cholesky_factor(A), SPARSE_OK);

    /* Reconstruct: (L*L^T)(i,j) = sum_k L(i,k)*L(j,k) for k <= min(i,j) */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k <= j; k++) {
                double l_ik = sparse_get_phys(A, (idx_t)i, (idx_t)k);
                double l_jk = sparse_get_phys(A, (idx_t)j, (idx_t)k);
                sum += l_ik * l_jk;
            }
            ASSERT_NEAR(sum, orig[i][j], 1e-12);
        }
    }

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Cholesky Factorization Tests");

    RUN_TEST(test_cholesky_2x2);
    RUN_TEST(test_cholesky_3x3_tridiag);
    RUN_TEST(test_cholesky_not_spd);
    RUN_TEST(test_cholesky_nonsquare);
    RUN_TEST(test_cholesky_null);
    RUN_TEST(test_cholesky_1x1);
    RUN_TEST(test_cholesky_identity);
    RUN_TEST(test_cholesky_diagonal);
    RUN_TEST(test_cholesky_reconstruct);

    TEST_SUITE_END();
}
