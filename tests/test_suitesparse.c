#include "sparse_matrix.h"
#include "sparse_lu.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif

#define SS_DIR DATA_DIR "/suitesparse"

/*
 * Helper: load matrix, factor, solve with b = A*ones (exact solution x = ones).
 * Returns 1 on success, 0 on expected failure (singular), -1 on error.
 */
static int solve_and_check(const char *path, sparse_pivot_t pivot,
                           double tol, double res_threshold)
{
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, path);
    if (err != SPARSE_OK) {
        printf("    load failed: %s\n", sparse_strerror(err));
        return -1;
    }

    idx_t n = sparse_rows(A);
    if (n != sparse_cols(A)) {
        printf("    non-square (%d x %d), skipping solve\n", n, sparse_cols(A));
        sparse_free(A);
        return 0;
    }

    /* Generate RHS: b = A * ones(n) so exact solution is x = ones(n) */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b    = malloc((size_t)n * sizeof(double));
    double *x    = malloc((size_t)n * sizeof(double));
    double *r    = malloc((size_t)n * sizeof(double));
    if (!ones || !b || !x || !r) {
        free(ones); free(b); free(x); free(r);
        sparse_free(A);
        return -1;
    }
    for (idx_t i = 0; i < n; i++) ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* Copy for residual check, then factor */
    SparseMatrix *A_orig = sparse_copy(A);
    if (!A_orig) {
        free(ones); free(b); free(x); free(r);
        sparse_free(A);
        return -1;
    }

    err = sparse_lu_factor(A, pivot, tol);
    if (err == SPARSE_ERR_SINGULAR) {
        printf("    singular (pivot=%s)\n",
               pivot == SPARSE_PIVOT_PARTIAL ? "partial" : "complete");
        free(ones); free(b); free(x); free(r);
        sparse_free(A); sparse_free(A_orig);
        return 0;
    }
    if (err != SPARSE_OK) {
        printf("    factor failed: %s\n", sparse_strerror(err));
        free(ones); free(b); free(x); free(r);
        sparse_free(A); sparse_free(A_orig);
        return -1;
    }

    err = sparse_lu_solve(A, b, x);
    if (err != SPARSE_OK) {
        printf("    solve failed: %s\n", sparse_strerror(err));
        free(ones); free(b); free(x); free(r);
        sparse_free(A); sparse_free(A_orig);
        return -1;
    }

    /* Compute residual: r = b - A_orig * x */
    sparse_matvec(A_orig, x, r);
    double res_norm = 0.0;
    double b_norm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double ri = fabs(b[i] - r[i]);
        if (ri > res_norm) res_norm = ri;
        double bi = fabs(b[i]);
        if (bi > b_norm) b_norm = bi;
    }

    double rel_res = (b_norm > 0.0) ? res_norm / b_norm : res_norm;

    printf("    n=%d nnz=%d pivot=%s rel_res=%.3e %s\n",
           n, sparse_nnz(A_orig),
           pivot == SPARSE_PIVOT_PARTIAL ? "partial" : "complete",
           rel_res,
           rel_res < res_threshold ? "OK" : "HIGH");

    int result = (rel_res < res_threshold) ? 1 : 0;

    free(ones); free(b); free(x); free(r);
    sparse_free(A); sparse_free(A_orig);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Per-matrix tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_west0067_partial(void)
{
    int ok = solve_and_check(SS_DIR "/west0067.mtx",
                             SPARSE_PIVOT_PARTIAL, 1e-12, 1e-10);
    ASSERT_EQ(ok, 1);
}

static void test_west0067_complete(void)
{
    int ok = solve_and_check(SS_DIR "/west0067.mtx",
                             SPARSE_PIVOT_COMPLETE, 1e-12, 1e-10);
    ASSERT_EQ(ok, 1);
}

static void test_nos4_partial(void)
{
    int ok = solve_and_check(SS_DIR "/nos4.mtx",
                             SPARSE_PIVOT_PARTIAL, 1e-12, 1e-10);
    ASSERT_EQ(ok, 1);
}

static void test_nos4_complete(void)
{
    int ok = solve_and_check(SS_DIR "/nos4.mtx",
                             SPARSE_PIVOT_COMPLETE, 1e-12, 1e-10);
    ASSERT_EQ(ok, 1);
}

static void test_bcsstk04_partial(void)
{
    int ok = solve_and_check(SS_DIR "/bcsstk04.mtx",
                             SPARSE_PIVOT_PARTIAL, 1e-12, 1e-8);
    ASSERT_EQ(ok, 1);
}

static void test_bcsstk04_complete(void)
{
    int ok = solve_and_check(SS_DIR "/bcsstk04.mtx",
                             SPARSE_PIVOT_COMPLETE, 1e-12, 1e-8);
    ASSERT_EQ(ok, 1);
}

static void test_steam1_partial(void)
{
    int ok = solve_and_check(SS_DIR "/steam1.mtx",
                             SPARSE_PIVOT_PARTIAL, 1e-12, 1e-10);
    ASSERT_EQ(ok, 1);
}

static void test_steam1_complete(void)
{
    int ok = solve_and_check(SS_DIR "/steam1.mtx",
                             SPARSE_PIVOT_COMPLETE, 1e-12, 1e-10);
    ASSERT_EQ(ok, 1);
}

static void test_fs_541_1_partial(void)
{
    int ok = solve_and_check(SS_DIR "/fs_541_1.mtx",
                             SPARSE_PIVOT_PARTIAL, 1e-12, 1e-8);
    ASSERT_EQ(ok, 1);
}

static void test_fs_541_1_complete(void)
{
    int ok = solve_and_check(SS_DIR "/fs_541_1.mtx",
                             SPARSE_PIVOT_COMPLETE, 1e-12, 1e-8);
    ASSERT_EQ(ok, 1);
}

static void test_orsirr_1_partial(void)
{
    int ok = solve_and_check(SS_DIR "/orsirr_1.mtx",
                             SPARSE_PIVOT_PARTIAL, 1e-12, 1e-8);
    ASSERT_EQ(ok, 1);
}

static void test_orsirr_1_complete(void)
{
    int ok = solve_and_check(SS_DIR "/orsirr_1.mtx",
                             SPARSE_PIVOT_COMPLETE, 1e-12, 1e-8);
    ASSERT_EQ(ok, 1);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Norm-only tests for all matrices (always valid, even for singular)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_norminf_all_matrices(void)
{
    int large = getenv("SPARSE_TEST_LARGE") && atoi(getenv("SPARSE_TEST_LARGE")) > 0;
    const char *files[] = {
        SS_DIR "/west0067.mtx",
        SS_DIR "/nos4.mtx",
        SS_DIR "/bcsstk04.mtx",
        SS_DIR "/steam1.mtx",
        SS_DIR "/fs_541_1.mtx",   /* large */
        SS_DIR "/orsirr_1.mtx",   /* large */
    };
    int count = large ? 6 : 4;
    for (int i = 0; i < count; i++) {
        SparseMatrix *A = NULL;
        ASSERT_ERR(sparse_load_mm(&A, files[i]), SPARSE_OK);
        double norm;
        ASSERT_ERR(sparse_norminf(A, &norm), SPARSE_OK);
        ASSERT_TRUE(norm > 0.0);
        sparse_free(A);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Condition number estimation on SuiteSparse matrices
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_condest_suitesparse(void)
{
    const char *files[] = {
        SS_DIR "/west0067.mtx",
        SS_DIR "/nos4.mtx",
        SS_DIR "/bcsstk04.mtx",
        SS_DIR "/steam1.mtx",
    };
    for (int i = 0; i < 4; i++) {
        SparseMatrix *A = NULL;
        ASSERT_ERR(sparse_load_mm(&A, files[i]), SPARSE_OK);

        SparseMatrix *LU = sparse_copy(A);
        ASSERT_NOT_NULL(LU);
        ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

        double cond;
        ASSERT_ERR(sparse_lu_condest(A, LU, &cond), SPARSE_OK);
        ASSERT_TRUE(cond > 0.0);  /* estimate must be positive and finite */

        printf("    %s: condest = %.3e\n", files[i], cond);

        sparse_free(A);
        sparse_free(LU);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("SuiteSparse Matrix Tests");

    /* west0067: 67x67, unsymmetric, chemical engineering */
    RUN_TEST(test_west0067_partial);
    RUN_TEST(test_west0067_complete);

    /* nos4: 100x100, symmetric, structural */
    RUN_TEST(test_nos4_partial);
    RUN_TEST(test_nos4_complete);

    /* bcsstk04: 132x132, symmetric, structural stiffness */
    RUN_TEST(test_bcsstk04_partial);
    RUN_TEST(test_bcsstk04_complete);

    /* steam1: 240x240, unsymmetric, thermal */
    RUN_TEST(test_steam1_partial);
    RUN_TEST(test_steam1_complete);

    /* Large matrices: gated behind SPARSE_TEST_LARGE=1 to avoid slow CI runs
     * (especially under ASan/UBSan). Run with: SPARSE_TEST_LARGE=1 make test */
    if (getenv("SPARSE_TEST_LARGE") && atoi(getenv("SPARSE_TEST_LARGE")) > 0) {
        /* fs_541_1: 541x541, unsymmetric, chemical process */
        RUN_TEST(test_fs_541_1_partial);
        RUN_TEST(test_fs_541_1_complete);

        /* orsirr_1: 1030x1030, unsymmetric, oil reservoir */
        RUN_TEST(test_orsirr_1_partial);
        RUN_TEST(test_orsirr_1_complete);
    } else {
        printf("  [SKIP] Large matrix tests (set SPARSE_TEST_LARGE=1 to enable)\n");
    }

    /* Norm check (small matrices only; includes large when SPARSE_TEST_LARGE=1) */
    RUN_TEST(test_norminf_all_matrices);

    /* Condition number estimation */
    RUN_TEST(test_condest_suitesparse);

    TEST_SUITE_END();
}
