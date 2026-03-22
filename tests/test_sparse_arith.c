#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_scale tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_scale_identity_by_3(void)
{
    SparseMatrix *A = sparse_create(4, 4);
    for (idx_t i = 0; i < 4; i++)
        sparse_insert(A, i, i, 1.0);

    ASSERT_ERR(sparse_scale(A, 3.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 4);
    for (idx_t i = 0; i < 4; i++)
        ASSERT_NEAR(sparse_get_phys(A, i, i), 3.0, 0.0);

    sparse_free(A);
}

static void test_scale_by_1(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 5.0);
    sparse_insert(A, 1, 2, 7.0);
    sparse_insert(A, 2, 1, -3.0);

    ASSERT_ERR(sparse_scale(A, 1.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 3);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 2), 7.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 2, 1), -3.0, 0.0);

    sparse_free(A);
}

static void test_scale_by_0(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);
    ASSERT_EQ(sparse_nnz(A), 3);

    ASSERT_ERR(sparse_scale(A, 0.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 0);

    /* All entries should be gone */
    for (idx_t i = 0; i < 3; i++)
        for (idx_t j = 0; j < 3; j++)
            ASSERT_NEAR(sparse_get_phys(A, i, j), 0.0, 0.0);

    sparse_free(A);
}

static void test_scale_by_negative_1(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, -2.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 1.0);

    ASSERT_ERR(sparse_scale(A, -1.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 4);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), -4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 0, 1), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 0), -3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), -1.0, 0.0);

    sparse_free(A);
}

static void test_scale_rectangular(void)
{
    SparseMatrix *A = sparse_create(2, 5);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 4, 2.0);
    sparse_insert(A, 1, 2, 3.0);

    ASSERT_ERR(sparse_scale(A, 0.5), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 3);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 0.5, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 0, 4), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 2), 1.5, 0.0);

    sparse_free(A);
}

static void test_scale_null(void)
{
    ASSERT_ERR(sparse_scale(NULL, 2.0), SPARSE_ERR_NULL);
}

static void test_scale_empty_matrix(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_ERR(sparse_scale(A, 5.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 0);
    sparse_free(A);
}

static void test_scale_invalidates_norm(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);

    double norm;
    ASSERT_ERR(sparse_norminf(A, &norm), SPARSE_OK);
    ASSERT_NEAR(norm, 2.0, 0.0);

    ASSERT_ERR(sparse_scale(A, 3.0), SPARSE_OK);

    ASSERT_ERR(sparse_norminf(A, &norm), SPARSE_OK);
    ASSERT_NEAR(norm, 6.0, 0.0);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Sparse Matrix Arithmetic Tests");

    /* Scale */
    RUN_TEST(test_scale_identity_by_3);
    RUN_TEST(test_scale_by_1);
    RUN_TEST(test_scale_by_0);
    RUN_TEST(test_scale_by_negative_1);
    RUN_TEST(test_scale_rectangular);
    RUN_TEST(test_scale_null);
    RUN_TEST(test_scale_empty_matrix);
    RUN_TEST(test_scale_invalidates_norm);

    TEST_SUITE_END();
}
