#include "sparse_matrix.h"
#include "sparse_lu.h"
#include "sparse_reorder.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_permute tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Identity permutation → output equals input */
static void test_permute_identity(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 0, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    idx_t perm[] = {0, 1, 2};
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &B), SPARSE_OK);
    ASSERT_NOT_NULL(B);

    ASSERT_EQ(sparse_nnz(B), sparse_nnz(A));
    ASSERT_NEAR(sparse_get_phys(B, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 0, 2), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 1), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 2, 0), 7.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 2, 2), 9.0, 0.0);

    sparse_free(A);
    sparse_free(B);
}

/* Reverse permutation on diagonal → diagonal reversed */
static void test_permute_reverse_diagonal(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);

    idx_t perm[] = {2, 1, 0};
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &B), SPARSE_OK);
    ASSERT_NOT_NULL(B);

    /* B(0,0) = A(2,2) = 3, B(1,1) = A(1,1) = 2, B(2,2) = A(0,0) = 1 */
    ASSERT_NEAR(sparse_get_phys(B, 0, 0), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 1), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 2, 2), 1.0, 0.0);

    sparse_free(A);
    sparse_free(B);
}

/* Known permutation on small matrix → verify entries */
static void test_permute_known(void)
{
    /* A = [1 2; 3 4], perm = [1, 0] (swap rows and cols)
     * B(0,0) = A(1,1) = 4, B(0,1) = A(1,0) = 3
     * B(1,0) = A(0,1) = 2, B(1,1) = A(0,0) = 1 */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 4.0);

    idx_t perm[] = {1, 0};
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &B), SPARSE_OK);
    ASSERT_NOT_NULL(B);

    ASSERT_NEAR(sparse_get_phys(B, 0, 0), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 0, 1), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 0), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 1), 1.0, 0.0);

    sparse_free(A);
    sparse_free(B);
}

/* Permute preserves nnz */
static void test_permute_preserves_nnz(void)
{
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 2, 2.0);
    sparse_insert(A, 2, 3, 3.0);
    sparse_insert(A, 3, 0, 4.0);

    idx_t perm[] = {3, 0, 1, 2};
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &B), SPARSE_OK);
    ASSERT_NOT_NULL(B);
    ASSERT_EQ(sparse_nnz(B), 4);

    sparse_free(A);
    sparse_free(B);
}

/* NULL inputs → proper error codes */
static void test_permute_null(void)
{
    idx_t perm[] = {0};
    SparseMatrix *B;
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_ERR(sparse_permute(NULL, perm, perm, &B), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_permute(A, NULL, perm, &B), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_permute(A, perm, NULL, &B), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_permute(A, perm, perm, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_bandwidth tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_bandwidth_diagonal(void)
{
    SparseMatrix *A = sparse_create(5, 5);
    for (idx_t i = 0; i < 5; i++)
        sparse_insert(A, i, i, 1.0);
    ASSERT_EQ(sparse_bandwidth(A), 0);
    sparse_free(A);
}

static void test_bandwidth_tridiag(void)
{
    SparseMatrix *A = sparse_create(5, 5);
    for (idx_t i = 0; i < 5; i++) {
        sparse_insert(A, i, i, 2.0);
        if (i > 0) sparse_insert(A, i, i-1, -1.0);
        if (i < 4) sparse_insert(A, i, i+1, -1.0);
    }
    ASSERT_EQ(sparse_bandwidth(A), 1);
    sparse_free(A);
}

static void test_bandwidth_full(void)
{
    /* Entry at (0, 4) gives bandwidth 4 */
    SparseMatrix *A = sparse_create(5, 5);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 4, 1.0);
    sparse_insert(A, 4, 4, 1.0);
    ASSERT_EQ(sparse_bandwidth(A), 4);
    sparse_free(A);
}

static void test_bandwidth_null(void)
{
    ASSERT_EQ(sparse_bandwidth(NULL), 0);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Permute + solve integration test
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_permute_then_solve(void)
{
    /* A = [4 -1 0; -1 4 -1; 0 -1 4], solve A*x = b */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, -1.0);
    sparse_insert(A, 1, 0, -1.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 2, -1.0);
    sparse_insert(A, 2, 1, -1.0);
    sparse_insert(A, 2, 2, 4.0);

    double b_orig[] = {1.0, 2.0, 3.0};

    /* Permute with perm = [2, 0, 1] */
    idx_t perm[] = {2, 0, 1};
    SparseMatrix *PA = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &PA), SPARSE_OK);

    /* Permute the RHS: pb[i] = b[perm[i]] */
    double pb[3];
    for (int i = 0; i < 3; i++) pb[i] = b_orig[perm[i]];

    /* Factor and solve the permuted system */
    ASSERT_ERR(sparse_lu_factor(PA, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    double xp[3];
    ASSERT_ERR(sparse_lu_solve(PA, pb, xp), SPARSE_OK);

    /* Unpermute: x_orig[perm[i]] = xp[i] */
    double x_orig[3];
    for (int i = 0; i < 3; i++) x_orig[perm[i]] = xp[i];

    /* Verify A * x_orig ≈ b_orig */
    double r[3];
    sparse_matvec(A, x_orig, r);
    for (int i = 0; i < 3; i++) r[i] -= b_orig[i];
    double rnorm = 0.0;
    for (int i = 0; i < 3; i++) {
        double a = fabs(r[i]);
        if (a > rnorm) rnorm = a;
    }
    ASSERT_TRUE(rnorm < 1e-12);

    sparse_free(A);
    sparse_free(PA);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Reordering & Permutation Tests");

    /* Permute */
    RUN_TEST(test_permute_identity);
    RUN_TEST(test_permute_reverse_diagonal);
    RUN_TEST(test_permute_known);
    RUN_TEST(test_permute_preserves_nnz);
    RUN_TEST(test_permute_null);

    /* Bandwidth */
    RUN_TEST(test_bandwidth_diagonal);
    RUN_TEST(test_bandwidth_tridiag);
    RUN_TEST(test_bandwidth_full);
    RUN_TEST(test_bandwidth_null);

    /* Integration */
    RUN_TEST(test_permute_then_solve);

    TEST_SUITE_END();
}
