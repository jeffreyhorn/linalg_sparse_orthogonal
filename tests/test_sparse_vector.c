#include "sparse_vector.h"
#include "sparse_matrix.h"
#include "sparse_lu.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * vec_norm2 tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_norm2_basic(void)
{
    double v[] = {3.0, 4.0};
    ASSERT_NEAR(vec_norm2(v, 2), 5.0, 1e-15);
}

static void test_norm2_single(void)
{
    double v[] = {-7.0};
    ASSERT_NEAR(vec_norm2(v, 1), 7.0, 1e-15);
}

static void test_norm2_zero(void)
{
    double v[] = {0.0, 0.0, 0.0};
    ASSERT_NEAR(vec_norm2(v, 3), 0.0, 0.0);
}

static void test_norm2_null(void)
{
    ASSERT_NEAR(vec_norm2(NULL, 5), 0.0, 0.0);
    double v[] = {1.0};
    ASSERT_NEAR(vec_norm2(v, 0), 0.0, 0.0);
}

/* ═══════════════════════════════════════════════════════════════════════
 * vec_norminf tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_norminf_basic(void)
{
    double v[] = {1.0, -5.0, 3.0};
    ASSERT_NEAR(vec_norminf(v, 3), 5.0, 0.0);
}

static void test_norminf_null(void)
{
    ASSERT_NEAR(vec_norminf(NULL, 5), 0.0, 0.0);
}

/* ═══════════════════════════════════════════════════════════════════════
 * vec_axpy tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_axpy_basic(void)
{
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {10.0, 20.0, 30.0};
    vec_axpy(2.0, x, y, 3);
    ASSERT_NEAR(y[0], 12.0, 0.0);
    ASSERT_NEAR(y[1], 24.0, 0.0);
    ASSERT_NEAR(y[2], 36.0, 0.0);
}

static void test_axpy_zero_alpha(void)
{
    double x[] = {1.0, 2.0};
    double y[] = {5.0, 6.0};
    vec_axpy(0.0, x, y, 2);
    ASSERT_NEAR(y[0], 5.0, 0.0);
    ASSERT_NEAR(y[1], 6.0, 0.0);
}

static void test_axpy_negative(void)
{
    double x[] = {1.0, 2.0};
    double y[] = {5.0, 6.0};
    vec_axpy(-1.0, x, y, 2);
    ASSERT_NEAR(y[0], 4.0, 0.0);
    ASSERT_NEAR(y[1], 4.0, 0.0);
}

static void test_axpy_null(void)
{
    double y[] = {1.0};
    vec_axpy(1.0, NULL, y, 1);  /* should not crash */
    ASSERT_NEAR(y[0], 1.0, 0.0);
}

/* ═══════════════════════════════════════════════════════════════════════
 * vec_copy tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_copy_basic(void)
{
    double src[] = {1.0, 2.0, 3.0};
    double dst[3] = {0};
    vec_copy(src, dst, 3);
    ASSERT_NEAR(dst[0], 1.0, 0.0);
    ASSERT_NEAR(dst[1], 2.0, 0.0);
    ASSERT_NEAR(dst[2], 3.0, 0.0);
}

static void test_copy_null(void)
{
    double dst[2] = {99.0, 99.0};
    vec_copy(NULL, dst, 2);  /* should not crash */
    ASSERT_NEAR(dst[0], 99.0, 0.0);  /* unchanged */
}

/* ═══════════════════════════════════════════════════════════════════════
 * vec_zero tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_zero_basic(void)
{
    double v[] = {1.0, 2.0, 3.0};
    vec_zero(v, 3);
    ASSERT_NEAR(v[0], 0.0, 0.0);
    ASSERT_NEAR(v[1], 0.0, 0.0);
    ASSERT_NEAR(v[2], 0.0, 0.0);
}

static void test_zero_null(void)
{
    vec_zero(NULL, 5);  /* should not crash */
    ASSERT_TRUE(1);
}

/* ═══════════════════════════════════════════════════════════════════════
 * vec_dot tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_dot_basic(void)
{
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    ASSERT_NEAR(vec_dot(x, y, 3), 32.0, 0.0);  /* 4+10+18 */
}

static void test_dot_orthogonal(void)
{
    double x[] = {1.0, 0.0};
    double y[] = {0.0, 1.0};
    ASSERT_NEAR(vec_dot(x, y, 2), 0.0, 0.0);
}

static void test_dot_null(void)
{
    double x[] = {1.0};
    ASSERT_NEAR(vec_dot(NULL, x, 1), 0.0, 0.0);
    ASSERT_NEAR(vec_dot(x, NULL, 1), 0.0, 0.0);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SpMV: compare against dense matrix-vector multiply
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_matvec_vs_dense(void)
{
    /* Build a 5x5 dense-ish matrix and verify SpMV matches manual computation */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    double dense[5][5] = {
        { 2, -1,  0,  0,  0},
        {-1,  2, -1,  0,  0},
        { 0, -1,  2, -1,  0},
        { 0,  0, -1,  2, -1},
        { 0,  0,  0, -1,  2}
    };
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            if (dense[i][j] != 0.0)
                sparse_insert(A, i, j, dense[i][j]);

    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y_sparse[5] = {0}, y_dense[5] = {0};

    sparse_matvec(A, x, y_sparse);

    /* Dense matvec */
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            y_dense[i] += dense[i][j] * x[j];

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(y_sparse[i], y_dense[i], 1e-14);

    sparse_free(A);
}

static void test_matvec_rectangular(void)
{
    /* 2x3 matrix: [1 2 3; 4 5 6] * [1; 1; 1] = [6; 15] */
    SparseMatrix *A = sparse_create(2, 3);
    sparse_insert(A, 0, 0, 1.0); sparse_insert(A, 0, 1, 2.0); sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 0, 4.0); sparse_insert(A, 1, 1, 5.0); sparse_insert(A, 1, 2, 6.0);

    double x[] = {1.0, 1.0, 1.0};
    double y[2] = {0};
    ASSERT_ERR(sparse_matvec(A, x, y), SPARSE_OK);
    ASSERT_NEAR(y[0], 6.0, 1e-14);
    ASSERT_NEAR(y[1], 15.0, 1e-14);

    sparse_free(A);
}

static void test_matvec_empty(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    double x[] = {1.0, 2.0, 3.0};
    double y[3] = {99.0, 99.0, 99.0};
    ASSERT_ERR(sparse_matvec(A, x, y), SPARSE_OK);
    ASSERT_NEAR(y[0], 0.0, 0.0);
    ASSERT_NEAR(y[1], 0.0, 0.0);
    ASSERT_NEAR(y[2], 0.0, 0.0);
    sparse_free(A);
}

static void test_matvec_large_values(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1e15);
    sparse_insert(A, 1, 1, 1e-15);

    double x[] = {1.0, 1.0};
    double y[2] = {0};
    sparse_matvec(A, x, y);
    ASSERT_NEAR(y[0], 1e15, 1.0);      /* relative precision */
    ASSERT_NEAR(y[1], 1e-15, 1e-30);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Iterative refinement on ill-conditioned system
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_refinement_improves_illcond(void)
{
    /*
     * Build a moderately ill-conditioned 8x8 matrix:
     * A(i,j) = 1/(i+j+1) + 10*delta(i,j)  (Hilbert-like + diagonal boost)
     * Without refinement, residual should be larger than with refinement.
     */
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, 1.0 / (double)(i + j + 1)
                                   + ((i == j) ? 10.0 : 0.0));

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++) b[i] = 1.0;

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    /* Residual before refinement */
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++) r[i] -= b[i];
    double res_before = vec_norminf(r, n);

    /* Refine */
    ASSERT_ERR(sparse_lu_refine(A, LU, b, x, 5, 1e-15), SPARSE_OK);

    /* Residual after refinement */
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++) r[i] -= b[i];
    double res_after = vec_norminf(r, n);

    /* Refinement should not make things worse */
    ASSERT_TRUE(res_after <= res_before + 1e-15);

    /* Both should be small for this mildly ill-conditioned matrix */
    ASSERT_TRUE(res_after < 1e-12);

    free(b); free(x); free(r);
    sparse_free(LU);
    sparse_free(A);
}

static void test_refinement_zero_rhs(void)
{
    /* b = 0 ⟹ x should be 0 */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    SparseMatrix *LU = sparse_copy(A);
    sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12);

    double b[] = {0, 0, 0, 0};
    double x[4] = {0};
    sparse_lu_solve(LU, b, x);
    sparse_lu_refine(A, LU, b, x, 3, 1e-15);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 0.0, 1e-15);

    sparse_free(LU);
    sparse_free(A);
}

static void test_refinement_multiple_rhs(void)
{
    /* Solve the same factorization with two different RHS vectors */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)     sparse_insert(A, i, i-1, -1.0);
        if (i < n - 1) sparse_insert(A, i, i+1, -1.0);
    }

    SparseMatrix *LU = sparse_copy(A);
    sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12);

    double b1[] = {1, 0, 0, 0, 0};
    double b2[] = {0, 0, 0, 0, 1};
    double x1[5], x2[5], r[5];

    sparse_lu_solve(LU, b1, x1);
    sparse_lu_refine(A, LU, b1, x1, 3, 1e-15);
    sparse_matvec(A, x1, r);
    for (idx_t i = 0; i < n; i++) r[i] -= b1[i];
    ASSERT_TRUE(vec_norminf(r, n) < 1e-14);

    sparse_lu_solve(LU, b2, x2);
    sparse_lu_refine(A, LU, b2, x2, 3, 1e-15);
    sparse_matvec(A, x2, r);
    for (idx_t i = 0; i < n; i++) r[i] -= b2[i];
    ASSERT_TRUE(vec_norminf(r, n) < 1e-14);

    sparse_free(LU);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Vector Utilities, SpMV & Refinement Tests");

    /* vec_norm2 */
    RUN_TEST(test_norm2_basic);
    RUN_TEST(test_norm2_single);
    RUN_TEST(test_norm2_zero);
    RUN_TEST(test_norm2_null);

    /* vec_norminf */
    RUN_TEST(test_norminf_basic);
    RUN_TEST(test_norminf_null);

    /* vec_axpy */
    RUN_TEST(test_axpy_basic);
    RUN_TEST(test_axpy_zero_alpha);
    RUN_TEST(test_axpy_negative);
    RUN_TEST(test_axpy_null);

    /* vec_copy */
    RUN_TEST(test_copy_basic);
    RUN_TEST(test_copy_null);

    /* vec_zero */
    RUN_TEST(test_zero_basic);
    RUN_TEST(test_zero_null);

    /* vec_dot */
    RUN_TEST(test_dot_basic);
    RUN_TEST(test_dot_orthogonal);
    RUN_TEST(test_dot_null);

    /* SpMV */
    RUN_TEST(test_matvec_vs_dense);
    RUN_TEST(test_matvec_rectangular);
    RUN_TEST(test_matvec_empty);
    RUN_TEST(test_matvec_large_values);

    /* Iterative refinement */
    RUN_TEST(test_refinement_improves_illcond);
    RUN_TEST(test_refinement_zero_rhs);
    RUN_TEST(test_refinement_multiple_rhs);

    TEST_SUITE_END();
}
