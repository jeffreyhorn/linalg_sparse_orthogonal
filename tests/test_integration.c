#include "sparse_matrix.h"
#include "sparse_lu.h"
#include "sparse_vector.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>
#include <math.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 1: Load MM -> factor -> solve -> check residual -> save result
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_load_factor_solve_save(void)
{
    /* Load the tridiagonal matrix */
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, DATA_DIR "/tridiagonal_20.mtx"), SPARSE_OK);
    ASSERT_NOT_NULL(A);

    idx_t n = sparse_rows(A);
    ASSERT_EQ(n, 20);

    /* Set b = A * [1, 1, ..., 1] so exact solution is x = [1, ..., 1] */
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++) x_exact[i] = 1.0;
    sparse_matvec(A, x_exact, b);

    /* Factor a copy */
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    /* Solve */
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    /* Residual: r = b - A*x */
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++) r[i] = b[i] - r[i];
    double res = vec_norminf(r, n);
    ASSERT_TRUE(res < 1e-12);

    /* Solution accuracy */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-12);

    /* Save the solved result to a temp file and reload */
    ASSERT_ERR(sparse_save_mm(A, "/tmp/integ_tridiag.mtx"), SPARSE_OK);
    SparseMatrix *A2 = NULL;
    ASSERT_ERR(sparse_load_mm(&A2, "/tmp/integ_tridiag.mtx"), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), sparse_nnz(A2));
    ASSERT_EQ(sparse_rows(A2), n);

    sparse_free(A2);
    free(x_exact); free(b); free(x); free(r);
    sparse_free(LU);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 2: Create -> copy -> factor copy -> solve -> refine -> verify
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_create_copy_factor_refine(void)
{
    /* Build a 10x10 diag-dominant matrix programmatically */
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        if (i > 0)     sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1) sparse_insert(A, i, i + 1, -1.0);
        if (i > 1)     sparse_insert(A, i, i - 2, -0.5);
        if (i < n - 2) sparse_insert(A, i, i + 2, -0.5);
    }

    /* RHS: b = A * [1, 2, 3, ..., n] */
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++) x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    /* Copy and factor the copy (preserving A for residual) */
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);

    /* Original should be untouched */
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 10.0, 0.0);

    /* Solve */
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    /* Check pre-refinement residual */
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++) r[i] = b[i] - r[i];
    double res_before = vec_norminf(r, n);

    /* Refine */
    ASSERT_ERR(sparse_lu_refine(A, LU, b, x, 5, 1e-15), SPARSE_OK);

    /* Check post-refinement residual */
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++) r[i] = b[i] - r[i];
    double res_after = vec_norminf(r, n);

    ASSERT_TRUE(res_after <= res_before + 1e-15);
    ASSERT_TRUE(res_after < 1e-13);

    /* Solution accuracy */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    free(x_exact); free(b); free(x); free(r);
    sparse_free(LU);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 3: Multiple solves with same factorization
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_multiple_rhs_same_factorization(void)
{
    /* Load symmetric matrix */
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, DATA_DIR "/symmetric_4.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A);

    /* Factor once */
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);

    /* Solve with 3 different RHS vectors */
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));

    for (int rhs = 0; rhs < 3; rhs++) {
        /* b = A * e_rhs (unit vector) */
        vec_zero(b, n);
        for (idx_t i = 0; i < n; i++) {
            double col_val = sparse_get_phys(A, i, (idx_t)rhs);
            b[i] = col_val;
        }

        ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

        /* Residual check: r = b - A*x */
        sparse_matvec(A, x, r);
        for (idx_t i = 0; i < n; i++) r[i] = b[i] - r[i];
        double res = vec_norminf(r, n);
        ASSERT_TRUE(res < 1e-12);
    }

    free(b); free(x); free(r);
    sparse_free(LU);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 4: Round-trip: create -> save -> load -> compare
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_full_roundtrip(void)
{
    /* Create a matrix with varied structure */
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(i + 1) * 10.0);
        if (i < n - 1) sparse_insert(A, i, i + 1, -1.5);
        if (i > 0)     sparse_insert(A, i, i - 1, 2.3);
    }
    /* Add a few scattered off-diagonals */
    sparse_insert(A, 0, n - 1, 0.01);
    sparse_insert(A, n - 1, 0, -0.01);

    idx_t nnz_orig = sparse_nnz(A);

    /* Save */
    ASSERT_ERR(sparse_save_mm(A, "/tmp/integ_roundtrip.mtx"), SPARSE_OK);

    /* Load */
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_load_mm(&B, "/tmp/integ_roundtrip.mtx"), SPARSE_OK);
    ASSERT_NOT_NULL(B);

    /* Compare */
    ASSERT_EQ(sparse_rows(B), n);
    ASSERT_EQ(sparse_cols(B), n);
    ASSERT_EQ(sparse_nnz(B), nnz_orig);

    /* Element-by-element comparison */
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            ASSERT_NEAR(sparse_get_phys(A, i, j),
                        sparse_get_phys(B, i, j), 1e-14);

    /* Both should produce the same solution */
    double *b = malloc((size_t)n * sizeof(double));
    double *x_a = malloc((size_t)n * sizeof(double));
    double *x_b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++) b[i] = 1.0;

    SparseMatrix *LU_A = sparse_copy(A);
    SparseMatrix *LU_B = sparse_copy(B);
    sparse_lu_factor(LU_A, SPARSE_PIVOT_PARTIAL, 1e-12);
    sparse_lu_factor(LU_B, SPARSE_PIVOT_PARTIAL, 1e-12);
    sparse_lu_solve(LU_A, b, x_a);
    sparse_lu_solve(LU_B, b, x_b);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_a[i], x_b[i], 1e-14);

    free(b); free(x_a); free(x_b);
    sparse_free(LU_A);
    sparse_free(LU_B);
    sparse_free(B);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 5: Load all reference matrices, factor, solve
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_all_reference_matrices(void)
{
    const char *files[] = {
        DATA_DIR "/identity_5.mtx",
        DATA_DIR "/diagonal_10.mtx",
        DATA_DIR "/tridiagonal_20.mtx",
        DATA_DIR "/symmetric_4.mtx",
        DATA_DIR "/bcsstk01.mtx",
        DATA_DIR "/unsymm_5.mtx",
    };
    int nfiles = 6;

    for (int f = 0; f < nfiles; f++) {
        SparseMatrix *A = NULL;
        sparse_err_t err = sparse_load_mm(&A, files[f]);
        ASSERT_ERR(err, SPARSE_OK);
        ASSERT_NOT_NULL(A);

        idx_t n = sparse_rows(A);
        double *x_exact = malloc((size_t)n * sizeof(double));
        double *b = malloc((size_t)n * sizeof(double));
        double *x = malloc((size_t)n * sizeof(double));
        double *r = malloc((size_t)n * sizeof(double));
        for (idx_t i = 0; i < n; i++) x_exact[i] = 1.0;
        sparse_matvec(A, x_exact, b);

        SparseMatrix *LU = sparse_copy(A);
        ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
        ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

        /* Relative residual: ||r|| / ||b|| */
        sparse_matvec(A, x, r);
        for (idx_t i = 0; i < n; i++) r[i] = b[i] - r[i];
        double res = vec_norminf(r, n);
        double bnorm = vec_norminf(b, n);
        double rel_res = (bnorm > 0) ? res / bnorm : res;
        ASSERT_TRUE(rel_res < 1e-10);

        free(x_exact); free(b); free(x); free(r);
        sparse_free(LU);
        sparse_free(A);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 6: Both pivoting strategies produce same answer
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_both_pivots_agree_integration(void)
{
    /* Build a 15x15 matrix with some structure */
    idx_t n = 15;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 20.0);
        if (i > 0) sparse_insert(A, i, i - 1, -2.0);
        if (i < n - 1) sparse_insert(A, i, i + 1, -3.0);
        sparse_insert(A, i, (i + 5) % n, 0.5);
    }

    double *b = malloc((size_t)n * sizeof(double));
    double *x_comp = malloc((size_t)n * sizeof(double));
    double *x_part = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++) b[i] = (double)(i + 1);

    SparseMatrix *LU1 = sparse_copy(A);
    SparseMatrix *LU2 = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU1, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_factor(LU2, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    sparse_lu_solve(LU1, b, x_comp);
    sparse_lu_solve(LU2, b, x_part);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_comp[i], x_part[i], 1e-10);

    free(b); free(x_comp); free(x_part);
    sparse_free(LU1);
    sparse_free(LU2);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 7: Error recovery — handle failures gracefully
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_error_recovery(void)
{
    /* Attempt to factor singular matrix, then successfully factor a good one */
    SparseMatrix *bad = sparse_create(3, 3);
    sparse_insert(bad, 0, 0, 1.0);
    /* rows 1 and 2 are all zero — singular */
    sparse_err_t err = sparse_lu_factor(bad, SPARSE_PIVOT_COMPLETE, 1e-12);
    ASSERT_ERR(err, SPARSE_ERR_SINGULAR);
    sparse_free(bad);

    /* Now factor a good matrix — should work fine */
    SparseMatrix *good = sparse_create(3, 3);
    sparse_insert(good, 0, 0, 4.0); sparse_insert(good, 0, 1, 1.0);
    sparse_insert(good, 1, 0, 1.0); sparse_insert(good, 1, 1, 3.0);
    sparse_insert(good, 2, 2, 2.0);

    SparseMatrix *LU = sparse_copy(good);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    double b[] = {5.0, 4.0, 6.0};
    double x[3];
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    /* Verify */
    double r[3];
    sparse_matvec(good, x, r);
    for (int i = 0; i < 3; i++) r[i] -= b[i];
    ASSERT_TRUE(vec_norminf(r, 3) < 1e-14);

    sparse_free(LU);
    sparse_free(good);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Integration Tests");

    RUN_TEST(test_load_factor_solve_save);
    RUN_TEST(test_create_copy_factor_refine);
    RUN_TEST(test_multiple_rhs_same_factorization);
    RUN_TEST(test_full_roundtrip);
    RUN_TEST(test_all_reference_matrices);
    RUN_TEST(test_both_pivots_agree_integration);
    RUN_TEST(test_error_recovery);

    TEST_SUITE_END();
}
