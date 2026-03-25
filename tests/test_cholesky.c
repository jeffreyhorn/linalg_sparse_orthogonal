#include "sparse_matrix.h"
#include "sparse_cholesky.h"
#include "sparse_lu.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

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
 * Cholesky solve tests
 * ═══════════════════════════════════════════════════════════════════════ */

static double vec_norminf(const double *v, idx_t n)
{
    double mx = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double a = fabs(v[i]);
        if (a > mx) mx = a;
    }
    return mx;
}

/* Factor and solve 3x3 SPD → verify x matches known solution */
static void test_cholesky_solve_3x3(void)
{
    /* A = [[4,2,0],[2,5,1],[0,1,3]], b = [8, 14, 10] → x = [1, 2, 3] (by construction) */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0); sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 2.0); sparse_insert(A, 1, 1, 5.0); sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0); sparse_insert(A, 2, 2, 3.0);

    double b[] = {8.0, 15.0, 11.0};  /* A * [1,2,3] */

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);

    double x[3];
    ASSERT_ERR(sparse_cholesky_solve(L, b, x), SPARSE_OK);

    /* Verify via residual: ||b - A*x|| */
    double r[3];
    sparse_matvec(A, x, r);
    for (int i = 0; i < 3; i++) r[i] -= b[i];
    ASSERT_TRUE(vec_norminf(r, 3) < 1e-12);

    sparse_free(A);
    sparse_free(L);
}

/* Factor and solve 5x5 SPD tridiagonal → verify residual */
static void test_cholesky_solve_5x5_tridiag(void)
{
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) { sparse_insert(A, i, i-1, -1.0); sparse_insert(A, i-1, i, -1.0); }
    }

    /* b = A * ones */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(ones);
    ASSERT_NOT_NULL(b);
    for (idx_t i = 0; i < n; i++) ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);

    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x);
    ASSERT_ERR(sparse_cholesky_solve(L, b, x), SPARSE_OK);

    /* x should be ~ones */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-12);

    free(ones); free(b); free(x);
    sparse_free(A); sparse_free(L);
}

/* Factor with AMD → solve → correct result */
static void test_cholesky_solve_amd(void)
{
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(n + 1));
        if (i > 0) { sparse_insert(A, i, i-1, -1.0); sparse_insert(A, i-1, i, -1.0); }
    }

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(ones);
    ASSERT_NOT_NULL(b);
    for (idx_t i = 0; i < n; i++) ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    sparse_cholesky_opts_t opts = { SPARSE_REORDER_AMD };
    ASSERT_ERR(sparse_cholesky_factor_opts(L, &opts), SPARSE_OK);

    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x);
    ASSERT_ERR(sparse_cholesky_solve(L, b, x), SPARSE_OK);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-10);

    free(ones); free(b); free(x);
    sparse_free(A); sparse_free(L);
}

/* Factor with RCM → solve → correct result */
static void test_cholesky_solve_rcm(void)
{
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(n + 1));
        if (i > 0) { sparse_insert(A, i, i-1, -1.0); sparse_insert(A, i-1, i, -1.0); }
    }

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(ones);
    ASSERT_NOT_NULL(b);
    for (idx_t i = 0; i < n; i++) ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    sparse_cholesky_opts_t opts = { SPARSE_REORDER_RCM };
    ASSERT_ERR(sparse_cholesky_factor_opts(L, &opts), SPARSE_OK);

    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x);
    ASSERT_ERR(sparse_cholesky_solve(L, b, x), SPARSE_OK);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-10);

    free(ones); free(b); free(x);
    sparse_free(A); sparse_free(L);
}

/* Solve with NONE → same as plain cholesky_factor */
static void test_cholesky_solve_none(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0); sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 2.0); sparse_insert(A, 1, 1, 5.0); sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0); sparse_insert(A, 2, 2, 3.0);

    double b[] = {8.0, 15.0, 11.0};

    SparseMatrix *L1 = sparse_copy(A);
    sparse_cholesky_opts_t opts = { SPARSE_REORDER_NONE };
    ASSERT_ERR(sparse_cholesky_factor_opts(L1, &opts), SPARSE_OK);
    double x1[3];
    ASSERT_ERR(sparse_cholesky_solve(L1, b, x1), SPARSE_OK);

    SparseMatrix *L2 = sparse_copy(A);
    ASSERT_ERR(sparse_cholesky_factor(L2), SPARSE_OK);
    double x2[3];
    ASSERT_ERR(sparse_cholesky_solve(L2, b, x2), SPARSE_OK);

    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x1[i], x2[i], 1e-14);

    sparse_free(A); sparse_free(L1); sparse_free(L2);
}

/* Solve NULL args */
static void test_cholesky_solve_null(void)
{
    double b[1] = {1.0}, x[1];
    ASSERT_ERR(sparse_cholesky_solve(NULL, b, x), SPARSE_ERR_NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SuiteSparse validation
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: load, Cholesky factor+solve, check residual */
static void cholesky_validate(const char *path, double res_tol,
                               const sparse_cholesky_opts_t *opts)
{
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, path), SPARSE_OK);
    idx_t n = sparse_rows(A);

    double *ones = malloc((size_t)n * sizeof(double));
    double *b    = malloc((size_t)n * sizeof(double));
    double *x    = malloc((size_t)n * sizeof(double));
    double *r    = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(ones);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(r);
    for (idx_t i = 0; i < n; i++) ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    sparse_err_t err;
    if (opts)
        err = sparse_cholesky_factor_opts(L, opts);
    else
        err = sparse_cholesky_factor(L);
    ASSERT_ERR(err, SPARSE_OK);

    ASSERT_ERR(sparse_cholesky_solve(L, b, x), SPARSE_OK);

    sparse_matvec(A, x, r);
    double rnorm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        r[i] -= b[i];
        double a = fabs(r[i]);
        if (a > rnorm) rnorm = a;
    }

    printf("    %s: nnz_L=%d, res=%.2e\n", path, (int)sparse_nnz(L), rnorm);
    ASSERT_TRUE(rnorm < res_tol);

    free(ones); free(b); free(x); free(r);
    sparse_free(A); sparse_free(L);
}

static void test_cholesky_nos4(void)
{
    cholesky_validate(SS_DIR "/nos4.mtx", 1e-10, NULL);
}

static void test_cholesky_bcsstk04(void)
{
    cholesky_validate(SS_DIR "/bcsstk04.mtx", 1e-4, NULL);
}

static void test_cholesky_nos4_amd(void)
{
    sparse_cholesky_opts_t opts = { SPARSE_REORDER_AMD };
    cholesky_validate(SS_DIR "/nos4.mtx", 1e-10, &opts);
}

static void test_cholesky_bcsstk04_rcm(void)
{
    sparse_cholesky_opts_t opts = { SPARSE_REORDER_RCM };
    cholesky_validate(SS_DIR "/bcsstk04.mtx", 1e-4, &opts);
}

/* Compare Cholesky fill-in vs LU fill-in on SPD matrices */
static void test_cholesky_fillin_vs_lu(void)
{
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/nos4.mtx"), SPARSE_OK);

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);
    idx_t chol_nnz = sparse_nnz(L);

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    idx_t lu_nnz = sparse_nnz(LU);

    printf("    nos4: Cholesky nnz=%d, LU nnz=%d\n", (int)chol_nnz, (int)lu_nnz);

    /* Cholesky stores only lower triangle, so nnz should be ≤ LU */
    ASSERT_TRUE(chol_nnz <= lu_nnz);

    sparse_free(A); sparse_free(L); sparse_free(LU);
}

/* Nearly singular SPD: should factor but have large condition */
static void test_cholesky_nearly_singular(void)
{
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (i < n - 1) ? 1.0 : 1e-12);

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);

    /* Solve should work but with reduced accuracy */
    double b[] = {1.0, 1.0, 1.0, 1.0};
    double x[4];
    ASSERT_ERR(sparse_cholesky_solve(L, b, x), SPARSE_OK);

    /* x should be [1, 1, 1, 1e12] approximately */
    ASSERT_NEAR(x[0], 1.0, 1e-6);
    ASSERT_NEAR(x[3], 1e12, 1e6);

    sparse_free(A); sparse_free(L);
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

    /* Solve */
    RUN_TEST(test_cholesky_solve_3x3);
    RUN_TEST(test_cholesky_solve_5x5_tridiag);
    RUN_TEST(test_cholesky_solve_amd);
    RUN_TEST(test_cholesky_solve_rcm);
    RUN_TEST(test_cholesky_solve_none);
    RUN_TEST(test_cholesky_solve_null);

    /* SuiteSparse validation */
    RUN_TEST(test_cholesky_nos4);
    RUN_TEST(test_cholesky_bcsstk04);
    RUN_TEST(test_cholesky_nos4_amd);
    RUN_TEST(test_cholesky_bcsstk04_rcm);
    RUN_TEST(test_cholesky_fillin_vs_lu);

    /* Edge cases */
    RUN_TEST(test_cholesky_nearly_singular);

    TEST_SUITE_END();
}
