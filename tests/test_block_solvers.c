#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/* Build n×n SPD tridiagonal: diag=4, off-diag=-1 */
static SparseMatrix *make_spd_tridiag(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -1.0);
    }
    return A;
}

/* Compute ||A*x - b||_inf */
static double residual_inf(const SparseMatrix *A, const double *x, const double *b, idx_t n) {
    double *r = malloc((size_t)n * sizeof(double));
    if (!r)
        return INFINITY;
    sparse_matvec(A, x, r);
    double mx = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = fabs(r[i] - b[i]);
        if (d > mx)
            mx = d;
    }
    free(r);
    return mx;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_matvec_block tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_matvec_block_matches_single(void) {
    idx_t n = 6;
    idx_t nrhs = 3;
    SparseMatrix *A = make_spd_tridiag(n);

    double X[18]; /* 6×3 */
    for (idx_t k = 0; k < nrhs; k++)
        for (idx_t i = 0; i < n; i++)
            X[i + n * k] = (double)(i + 1) * (double)(k + 1);

    double Y_block[18], y_single[6];
    ASSERT_ERR(sparse_matvec_block(A, X, nrhs, Y_block), SPARSE_OK);

    /* Verify each column matches sparse_matvec */
    for (idx_t k = 0; k < nrhs; k++) {
        sparse_matvec(A, &X[n * k], y_single);
        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(Y_block[i + n * k], y_single[i], 1e-14);
    }

    sparse_free(A);
}

static void test_matvec_block_null(void) {
    SparseMatrix *A = make_spd_tridiag(3);
    double x = 0, y = 0;
    ASSERT_ERR(sparse_matvec_block(NULL, &x, 1, &y), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_matvec_block(A, NULL, 1, &y), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_matvec_block(A, &x, 1, NULL), SPARSE_ERR_NULL);
    /* nrhs=0 is OK */
    ASSERT_ERR(sparse_matvec_block(A, &x, 0, &y), SPARSE_OK);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Block CG tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Test: Block CG with 3 RHS matches single-RHS CG */
static void test_block_cg_3rhs(void) {
    idx_t n = 10;
    idx_t nrhs = 3;
    SparseMatrix *A = make_spd_tridiag(n);

    double B[30], X_block[30], x_single[10];
    for (idx_t k = 0; k < nrhs; k++)
        for (idx_t i = 0; i < n; i++)
            B[i + n * k] = (double)(i + 1) * (double)(k + 1);

    /* Zero initial guess */
    memset(X_block, 0, sizeof(X_block));

    sparse_iter_opts_t opts = {.max_iter = 1000, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_cg_solve_block(A, B, nrhs, X_block, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    /* Verify each column matches single-RHS CG */
    for (idx_t k = 0; k < nrhs; k++) {
        memset(x_single, 0, sizeof(x_single));
        sparse_solve_cg(A, &B[n * k], x_single, &opts, NULL, NULL, NULL);

        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(X_block[i + n * k], x_single[i], 1e-8);
    }

    sparse_free(A);
}

/* Test: Residual check for all columns */
static void test_block_cg_residuals(void) {
    idx_t n = 15;
    idx_t nrhs = 4;
    SparseMatrix *A = make_spd_tridiag(n);

    double *B = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    double *X = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    ASSERT_NOT_NULL(B);
    ASSERT_NOT_NULL(X);
    if (!B || !X) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }

    for (idx_t k = 0; k < nrhs; k++)
        for (idx_t i = 0; i < n; i++)
            B[i + n * k] = sin((double)(i + k * n));

    sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_cg_solve_block(A, B, nrhs, X, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    /* Check residual for each column */
    for (idx_t k = 0; k < nrhs; k++) {
        double res = residual_inf(A, &X[n * k], &B[n * k], n);
        ASSERT_TRUE(res < 1e-8);
    }

    free(B);
    free(X);
    sparse_free(A);
}

/* Test: Iteration count similar to single-RHS */
static void test_block_cg_iteration_count(void) {
    idx_t n = 20;
    idx_t nrhs = 2;
    SparseMatrix *A = make_spd_tridiag(n);

    double B[40], X_block[40];
    for (idx_t k = 0; k < nrhs; k++)
        for (idx_t i = 0; i < n; i++)
            B[i + n * k] = (double)(i + 1);
    memset(X_block, 0, sizeof(X_block));

    sparse_iter_opts_t opts = {.max_iter = 1000, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t block_result, single_result;

    ASSERT_ERR(sparse_cg_solve_block(A, B, nrhs, X_block, &opts, NULL, NULL, &block_result),
               SPARSE_OK);

    /* Single-RHS for comparison */
    double x_single[20];
    memset(x_single, 0, sizeof(x_single));
    sparse_solve_cg(A, B, x_single, &opts, NULL, NULL, &single_result);

    /* Block CG should converge in similar number of iterations */
    printf("    block_cg iters=%d  single_cg iters=%d\n", (int)block_result.iterations,
           (int)single_result.iterations);
    /* Allow up to 2x iterations (block may be slightly different) */
    ASSERT_TRUE(block_result.iterations <= 2 * single_result.iterations + 5);

    sparse_free(A);
}

/* Test: nrhs=1 matches single CG */
static void test_block_cg_single_rhs(void) {
    idx_t n = 8;
    SparseMatrix *A = make_spd_tridiag(n);

    double b[8], x_block[8], x_single[8];
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);
    memset(x_block, 0, sizeof(x_block));
    memset(x_single, 0, sizeof(x_single));

    sparse_iter_opts_t opts = {.max_iter = 1000, .tol = 1e-10, .verbose = 0};
    ASSERT_ERR(sparse_cg_solve_block(A, b, 1, x_block, &opts, NULL, NULL, NULL), SPARSE_OK);
    ASSERT_ERR(sparse_solve_cg(A, b, x_single, &opts, NULL, NULL, NULL), SPARSE_OK);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_block[i], x_single[i], 1e-10);

    sparse_free(A);
}

/* Test: Null args */
static void test_block_cg_null(void) {
    SparseMatrix *A = make_spd_tridiag(3);
    double b = 0, x = 0;
    ASSERT_ERR(sparse_cg_solve_block(NULL, &b, 1, &x, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_cg_solve_block(A, NULL, 1, &x, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_cg_solve_block(A, &b, 1, NULL, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

/* Test: nrhs=0 returns OK */
static void test_block_cg_nrhs_zero(void) {
    SparseMatrix *A = make_spd_tridiag(3);
    double dummy = 0;
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_cg_solve_block(A, &dummy, 0, &dummy, NULL, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Block GMRES tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Build n×n unsymmetric diag-dominant matrix */
static SparseMatrix *make_unsymmetric(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(n + 1));
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -2.0); /* asymmetric: -2 above, -1 below */
    }
    return A;
}

/* Test: Block GMRES with 3 RHS on unsymmetric matrix */
static void test_block_gmres_3rhs(void) {
    idx_t n = 10;
    idx_t nrhs = 3;
    SparseMatrix *A = make_unsymmetric(n);

    double B[30], X_block[30], x_single[10];
    for (idx_t k = 0; k < nrhs; k++)
        for (idx_t i = 0; i < n; i++)
            B[i + n * k] = (double)(i + 1) * (double)(k + 1);
    memset(X_block, 0, sizeof(X_block));

    sparse_gmres_opts_t opts = {
        .max_iter = 500, .restart = 30, .tol = 1e-10, .verbose = 0, .precond_side = 0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_gmres_solve_block(A, B, nrhs, X_block, &opts, NULL, NULL, &result),
               SPARSE_OK);
    ASSERT_TRUE(result.converged);

    /* Verify each column matches single-RHS GMRES */
    for (idx_t k = 0; k < nrhs; k++) {
        memset(x_single, 0, sizeof(x_single));
        sparse_solve_gmres(A, &B[n * k], x_single, &opts, NULL, NULL, NULL);
        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(X_block[i + n * k], x_single[i], 1e-8);
    }

    sparse_free(A);
}

/* Test: Block GMRES with single RHS matches single GMRES */
static void test_block_gmres_single_rhs(void) {
    idx_t n = 8;
    SparseMatrix *A = make_unsymmetric(n);

    double b[8], x_block[8], x_single[8];
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);
    memset(x_block, 0, sizeof(x_block));
    memset(x_single, 0, sizeof(x_single));

    sparse_gmres_opts_t opts = {
        .max_iter = 500, .restart = 30, .tol = 1e-10, .verbose = 0, .precond_side = 0};
    ASSERT_ERR(sparse_gmres_solve_block(A, b, 1, x_block, &opts, NULL, NULL, NULL), SPARSE_OK);
    ASSERT_ERR(sparse_solve_gmres(A, b, x_single, &opts, NULL, NULL, NULL), SPARSE_OK);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_block[i], x_single[i], 1e-10);

    sparse_free(A);
}

/* Test: Block GMRES residual check */
static void test_block_gmres_residuals(void) {
    idx_t n = 15;
    idx_t nrhs = 4;
    SparseMatrix *A = make_unsymmetric(n);

    double *B = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    double *X = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    ASSERT_NOT_NULL(B);
    ASSERT_NOT_NULL(X);
    if (!B || !X) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }

    for (idx_t k = 0; k < nrhs; k++)
        for (idx_t i = 0; i < n; i++)
            B[i + n * k] = sin((double)(i + k * n));

    sparse_gmres_opts_t opts = {
        .max_iter = 1000, .restart = 30, .tol = 1e-10, .verbose = 0, .precond_side = 0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_gmres_solve_block(A, B, nrhs, X, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    for (idx_t k = 0; k < nrhs; k++) {
        double res = residual_inf(A, &X[n * k], &B[n * k], n);
        ASSERT_TRUE(res < 1e-8);
    }

    free(B);
    free(X);
    sparse_free(A);
}

/* Test: Preconditioned block GMRES with ILU(0) */
static void test_block_gmres_preconditioned(void) {
    idx_t n = 10;
    idx_t nrhs = 2;
    SparseMatrix *A = make_unsymmetric(n);

    /* Factor ILU(0) preconditioner */
    sparse_ilu_t ilu;
    ASSERT_ERR(sparse_ilu_factor(A, &ilu), SPARSE_OK);

    double B[20], X[20];
    for (idx_t k = 0; k < nrhs; k++)
        for (idx_t i = 0; i < n; i++)
            B[i + n * k] = (double)(i + 1);
    memset(X, 0, sizeof(X));

    sparse_gmres_opts_t opts = {
        .max_iter = 500, .restart = 30, .tol = 1e-10, .verbose = 0, .precond_side = 0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_gmres_solve_block(A, B, nrhs, X, &opts, sparse_ilu_precond, &ilu, &result),
               SPARSE_OK);
    ASSERT_TRUE(result.converged);

    for (idx_t k = 0; k < nrhs; k++) {
        double res = residual_inf(A, &X[n * k], &B[n * k], n);
        ASSERT_TRUE(res < 1e-8);
    }

    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* Test: Block GMRES null args */
static void test_block_gmres_null(void) {
    SparseMatrix *A = make_unsymmetric(3);
    double b = 0, x = 0;
    ASSERT_ERR(sparse_gmres_solve_block(NULL, &b, 1, &x, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_gmres_solve_block(A, NULL, 1, &x, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_gmres_solve_block(A, &b, 1, NULL, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

/* Test: Block GMRES nrhs=0 */
static void test_block_gmres_nrhs_zero(void) {
    SparseMatrix *A = make_unsymmetric(3);
    double dummy = 0;
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_gmres_solve_block(A, &dummy, 0, &dummy, NULL, NULL, NULL, &result),
               SPARSE_OK);
    ASSERT_TRUE(result.converged);
    sparse_free(A);
}

/* Test: Block GMRES with restart — convergence on harder system */
static void test_block_gmres_restart(void) {
    /* Larger unsymmetric system that may need restart */
    idx_t n = 30;
    idx_t nrhs = 2;
    SparseMatrix *A = make_unsymmetric(n);

    double *B = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    double *X = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    ASSERT_NOT_NULL(B);
    ASSERT_NOT_NULL(X);
    if (!B || !X) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }

    for (idx_t k = 0; k < nrhs; k++)
        for (idx_t i = 0; i < n; i++)
            B[i + n * k] = (double)(i + 1);

    /* Small restart to force restart cycles */
    sparse_gmres_opts_t opts = {
        .max_iter = 500, .restart = 10, .tol = 1e-10, .verbose = 0, .precond_side = 0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_gmres_solve_block(A, B, nrhs, X, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    for (idx_t k = 0; k < nrhs; k++) {
        double res = residual_inf(A, &X[n * k], &B[n * k], n);
        ASSERT_TRUE(res < 1e-8);
    }

    free(B);
    free(X);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Block Solvers (CG + GMRES)");

    /* Block SpMV */
    RUN_TEST(test_matvec_block_null);
    RUN_TEST(test_matvec_block_matches_single);

    /* Block CG */
    RUN_TEST(test_block_cg_null);
    RUN_TEST(test_block_cg_nrhs_zero);
    RUN_TEST(test_block_cg_single_rhs);
    RUN_TEST(test_block_cg_3rhs);
    RUN_TEST(test_block_cg_residuals);
    RUN_TEST(test_block_cg_iteration_count);

    /* Block GMRES */
    RUN_TEST(test_block_gmres_null);
    RUN_TEST(test_block_gmres_nrhs_zero);
    RUN_TEST(test_block_gmres_single_rhs);
    RUN_TEST(test_block_gmres_3rhs);
    RUN_TEST(test_block_gmres_residuals);
    RUN_TEST(test_block_gmres_preconditioned);
    RUN_TEST(test_block_gmres_restart);

    TEST_SUITE_END();
}
