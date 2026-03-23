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
 * RCM tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Check that perm is a valid permutation of 0..n-1 */
static int is_valid_perm(const idx_t *perm, idx_t n)
{
    int *seen = calloc((size_t)n, sizeof(int));
    if (!seen) return 0;
    for (idx_t i = 0; i < n; i++) {
        if (perm[i] < 0 || perm[i] >= n || seen[perm[i]]) {
            free(seen);
            return 0;
        }
        seen[perm[i]] = 1;
    }
    free(seen);
    return 1;
}

/* Arrow matrix: dense first row/column → RCM should reduce bandwidth */
static void test_rcm_arrow(void)
{
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    /* Arrow: row 0 connects to all, col 0 connects to all, diagonal */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 2.0);
        if (i > 0) {
            sparse_insert(A, 0, i, 1.0);
            sparse_insert(A, i, 0, 1.0);
        }
    }

    idx_t bw_before = sparse_bandwidth(A);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_ERR(sparse_reorder_rcm(A, perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(perm, n));

    SparseMatrix *PA = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &PA), SPARSE_OK);
    idx_t bw_after = sparse_bandwidth(PA);

    /* RCM should not increase bandwidth (may reduce or keep same) */
    ASSERT_TRUE(bw_after <= bw_before);

    free(perm);
    sparse_free(A);
    sparse_free(PA);
}

/* Already-banded (tridiagonal) → RCM should not worsen */
static void test_rcm_tridiag(void)
{
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) sparse_insert(A, i, i-1, -1.0);
        if (i < n-1) sparse_insert(A, i, i+1, -1.0);
    }

    idx_t bw_before = sparse_bandwidth(A);
    ASSERT_EQ(bw_before, 1);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_ERR(sparse_reorder_rcm(A, perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(perm, n));

    SparseMatrix *PA = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &PA), SPARSE_OK);
    idx_t bw_after = sparse_bandwidth(PA);
    ASSERT_TRUE(bw_after <= bw_before);

    free(perm);
    sparse_free(A);
    sparse_free(PA);
}

/* Diagonal matrix → any permutation is valid */
static void test_rcm_diagonal(void)
{
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_ERR(sparse_reorder_rcm(A, perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(perm, n));

    /* Bandwidth should remain 0 */
    SparseMatrix *PA = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &PA), SPARSE_OK);
    ASSERT_EQ(sparse_bandwidth(PA), 0);

    free(perm);
    sparse_free(A);
    sparse_free(PA);
}

/* Block diagonal (disconnected graph) → should handle without errors */
static void test_rcm_disconnected(void)
{
    /* Two 3x3 blocks, no connections between them */
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    /* Block 1: rows 0-2 */
    sparse_insert(A, 0, 0, 4.0); sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0); sparse_insert(A, 1, 1, 4.0); sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0); sparse_insert(A, 2, 2, 4.0);
    /* Block 2: rows 3-5 */
    sparse_insert(A, 3, 3, 4.0); sparse_insert(A, 3, 4, 1.0);
    sparse_insert(A, 4, 3, 1.0); sparse_insert(A, 4, 4, 4.0); sparse_insert(A, 4, 5, 1.0);
    sparse_insert(A, 5, 4, 1.0); sparse_insert(A, 5, 5, 4.0);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_ERR(sparse_reorder_rcm(A, perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(perm, n));

    free(perm);
    sparse_free(A);
}

/* RCM + factor + solve produces correct result */
static void test_rcm_solve(void)
{
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    /* Arrow-like: node 0 connects to all, plus diagonal */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(n + 1));
        if (i > 0) {
            sparse_insert(A, 0, i, 1.0);
            sparse_insert(A, i, 0, 1.0);
        }
    }

    /* RCM reorder */
    idx_t *p = malloc((size_t)n * sizeof(idx_t));
    ASSERT_ERR(sparse_reorder_rcm(A, p), SPARSE_OK);

    SparseMatrix *PA = NULL;
    ASSERT_ERR(sparse_permute(A, p, p, &PA), SPARSE_OK);

    /* RHS: b = A * ones */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b    = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++) ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* Permute RHS */
    double *pb = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++) pb[i] = b[p[i]];

    /* Factor and solve permuted system */
    ASSERT_ERR(sparse_lu_factor(PA, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    double *xp = malloc((size_t)n * sizeof(double));
    ASSERT_ERR(sparse_lu_solve(PA, pb, xp), SPARSE_OK);

    /* Unpermute solution */
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++) x[p[i]] = xp[i];

    /* Verify residual */
    double *r = malloc((size_t)n * sizeof(double));
    sparse_matvec(A, x, r);
    double rnorm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        r[i] -= b[i];
        double a = fabs(r[i]);
        if (a > rnorm) rnorm = a;
    }
    ASSERT_TRUE(rnorm < 1e-10);

    free(p); free(ones); free(b); free(pb); free(xp); free(x); free(r);
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

    /* RCM */
    RUN_TEST(test_rcm_arrow);
    RUN_TEST(test_rcm_tridiag);
    RUN_TEST(test_rcm_diagonal);
    RUN_TEST(test_rcm_disconnected);
    RUN_TEST(test_rcm_solve);

    TEST_SUITE_END();
}
