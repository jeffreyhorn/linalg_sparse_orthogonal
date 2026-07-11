#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"
#include "test_solver_helpers.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int insert_or_free(SparseMatrix **A, idx_t row, idx_t col, double value) {
    sparse_err_t err = sparse_insert(*A, row, col, value);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(*A);
        *A = NULL;
        return 0;
    }
    return 1;
}

static SparseMatrix *build_identity(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        if (!insert_or_free(&A, i, i, 1.0))
            return NULL;
    }
    return A;
}

static SparseMatrix *build_unsym_tridiag(idx_t n, double diag, double upper, double lower) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        if (!insert_or_free(&A, i, i, diag))
            return NULL;
        if (i > 0 && !insert_or_free(&A, i, i - 1, lower))
            return NULL;
        if (i < n - 1 && !insert_or_free(&A, i, i + 1, upper))
            return NULL;
    }
    return A;
}

static int require_allocs(const SparseMatrix *A, const double *B, const double *X) {
    ASSERT_NOT_NULL(A);
    ASSERT_NOT_NULL(B);
    ASSERT_NOT_NULL(X);
    return A && B && X;
}

static void test_block_bicgstab_null_inputs(void) {
    SparseMatrix *A = build_identity(3);
    double B[3] = {1, 2, 3}, X[3] = {0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_bicgstab_solve_block(NULL, B, 1, X, NULL, NULL, NULL, &result),
               SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_bicgstab_solve_block(A, NULL, 1, X, NULL, NULL, NULL, &result),
               SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_bicgstab_solve_block(A, B, 1, NULL, NULL, NULL, NULL, &result),
               SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_block_bicgstab_nrhs_zero(void) {
    SparseMatrix *A = build_identity(3);
    double B[1] = {0}, X[1] = {0};
    sparse_iter_result_t result;

    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_err_t err = sparse_bicgstab_solve_block(A, B, 0, X, NULL, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    sparse_free(A);
}

static void test_block_bicgstab_nrhs_negative(void) {
    SparseMatrix *A = build_identity(3);
    double B[1] = {0}, X[1] = {0};
    sparse_iter_result_t result;

    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    ASSERT_ERR(sparse_bicgstab_solve_block(A, B, -1, X, NULL, NULL, NULL, &result),
               SPARSE_ERR_BADARG);
    sparse_free(A);
}

static void test_block_bicgstab_nonsquare(void) {
    SparseMatrix *A = sparse_create(3, 4);
    if (!A)
        return;
    double B[3] = {1, 2, 3}, X[4] = {0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_bicgstab_solve_block(A, B, 1, X, NULL, NULL, NULL, &result),
               SPARSE_ERR_SHAPE);
    sparse_free(A);
}

static void test_block_bicgstab_2rhs(void) {
    idx_t n = 20;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    double *B = calloc(2 * (size_t)n, sizeof(double));
    double *X = calloc(2 * (size_t)n, sizeof(double));
    if (!require_allocs(A, B, X)) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++) {
        B[i] = (double)(i + 1);
        B[i + n] = sin((double)(i + 1));
    }

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_bicgstab_solve_block(A, B, 2, X, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }
    ASSERT_TRUE(result.converged);

    for (int col = 0; col < 2; col++) {
        double *x_col = X + (size_t)col * (size_t)n;
        double *b_col = B + (size_t)col * (size_t)n;
        double rel_res = tf_relative_residual_l2(A, b_col, x_col, n, HUGE_VAL);
        ASSERT_TRUE(rel_res < 1e-8);
    }

    free(B);
    free(X);
    sparse_free(A);
}

static void test_block_bicgstab_4rhs(void) {
    idx_t n = 15;
    SparseMatrix *A = build_unsym_tridiag(n, 5.0, -1.5, -0.5);

    idx_t nrhs = 4;
    double *B = calloc((size_t)nrhs * (size_t)n, sizeof(double));
    double *X = calloc((size_t)nrhs * (size_t)n, sizeof(double));
    if (!require_allocs(A, B, X)) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }
    for (idx_t j = 0; j < nrhs; j++)
        for (idx_t i = 0; i < n; i++)
            B[(size_t)i + (size_t)j * (size_t)n] = (double)(i + 1) * (double)(j + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_bicgstab_solve_block(A, B, nrhs, X, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }
    ASSERT_TRUE(result.converged);

    for (idx_t j = 0; j < nrhs; j++) {
        double *x_col = X + (size_t)j * (size_t)n;
        double *b_col = B + (size_t)j * (size_t)n;
        double rel_res = tf_relative_residual_l2(A, b_col, x_col, n, HUGE_VAL);
        ASSERT_TRUE(rel_res < 1e-8);
    }

    free(B);
    free(X);
    sparse_free(A);
}

static void test_block_bicgstab_matches_single_rhs(void) {
    idx_t n = 15;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    idx_t nrhs = 3;
    double *B = calloc((size_t)nrhs * (size_t)n, sizeof(double));
    double *X_single = calloc((size_t)nrhs * (size_t)n, sizeof(double));
    double *X_block = calloc((size_t)nrhs * (size_t)n, sizeof(double));
    if (!require_allocs(A, B, X_single) || !X_block) {
        ASSERT_NOT_NULL(X_block);
        free(B);
        free(X_single);
        free(X_block);
        sparse_free(A);
        return;
    }
    for (idx_t j = 0; j < nrhs; j++)
        for (idx_t i = 0; i < n; i++)
            B[(size_t)i + (size_t)j * (size_t)n] = (double)(i + j * n + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};

    for (idx_t j = 0; j < nrhs; j++) {
        const double *bj = B + (size_t)j * (size_t)n;
        double *xj = X_single + (size_t)j * (size_t)n;
        sparse_err_t err = sparse_solve_bicgstab(A, bj, xj, &opts, NULL, NULL, NULL);
        ASSERT_ERR(err, SPARSE_OK);
        if (err != SPARSE_OK) {
            free(B);
            free(X_single);
            free(X_block);
            sparse_free(A);
            return;
        }
    }

    sparse_iter_result_t result;
    sparse_err_t err = sparse_bicgstab_solve_block(A, B, nrhs, X_block, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(B);
        free(X_single);
        free(X_block);
        sparse_free(A);
        return;
    }
    ASSERT_TRUE(result.converged);

    for (idx_t j = 0; j < nrhs; j++)
        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(X_block[(size_t)i + (size_t)j * (size_t)n],
                        X_single[(size_t)i + (size_t)j * (size_t)n], 1e-12);

    free(B);
    free(X_single);
    free(X_block);
    sparse_free(A);
}

static void test_block_bicgstab_mixed_convergence(void) {
    idx_t n = 20;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    double *B = calloc(2 * (size_t)n, sizeof(double));
    double *X = calloc(2 * (size_t)n, sizeof(double));
    if (!require_allocs(A, B, X)) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }

    B[0] = 1.0;
    for (idx_t i = 0; i < n; i++)
        B[i + n] = 1000.0 * sin(10.0 * (double)(i + 1));

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_bicgstab_solve_block(A, B, 2, X, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }
    ASSERT_TRUE(result.converged);

    for (int col = 0; col < 2; col++) {
        double rel_res = tf_relative_residual_l2(A, B + (size_t)col * (size_t)n,
                                                 X + (size_t)col * (size_t)n, n, HUGE_VAL);
        ASSERT_TRUE(rel_res < 1e-8);
    }

    ASSERT_TRUE(result.iterations > 0);

    free(B);
    free(X);
    sparse_free(A);
}

static void test_block_bicgstab_nrhs_1(void) {
    idx_t n = 10;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    double *b = calloc((size_t)n, sizeof(double));
    double *x_single = calloc((size_t)n, sizeof(double));
    double *x_block = calloc((size_t)n, sizeof(double));
    if (!require_allocs(A, b, x_single) || !x_block) {
        ASSERT_NOT_NULL(x_block);
        free(b);
        free(x_single);
        free(x_block);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};

    sparse_iter_result_t r1, r2;
    sparse_err_t err1 = sparse_solve_bicgstab(A, b, x_single, &opts, NULL, NULL, &r1);
    ASSERT_ERR(err1, SPARSE_OK);
    if (err1 != SPARSE_OK) {
        free(b);
        free(x_single);
        free(x_block);
        sparse_free(A);
        return;
    }
    sparse_err_t err2 = sparse_bicgstab_solve_block(A, b, 1, x_block, &opts, NULL, NULL, &r2);
    ASSERT_ERR(err2, SPARSE_OK);
    if (err2 != SPARSE_OK) {
        free(b);
        free(x_single);
        free(x_block);
        sparse_free(A);
        return;
    }

    ASSERT_TRUE(r1.converged);
    ASSERT_TRUE(r2.converged);
    ASSERT_EQ(r1.iterations, r2.iterations);
    ASSERT_NEAR(r1.residual_norm, r2.residual_norm, 1e-14);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_single[i], x_block[i], 1e-14);

    free(b);
    free(x_single);
    free(x_block);
    sparse_free(A);
}

static void test_block_bicgstab_preconditioned(void) {
    idx_t n = 30;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    idx_t nrhs = 2;
    double *B = calloc((size_t)nrhs * (size_t)n, sizeof(double));
    double *X = calloc((size_t)nrhs * (size_t)n, sizeof(double));
    if (!require_allocs(A, B, X)) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }
    for (idx_t j = 0; j < nrhs; j++)
        for (idx_t i = 0; i < n; i++)
            B[(size_t)i + (size_t)j * (size_t)n] = (double)(i + 1) * (double)(j + 1);

    sparse_ilu_t ilu;
    memset(&ilu, 0, sizeof(ilu));
    sparse_err_t ilu_err = sparse_ilu_factor(A, &ilu);
    ASSERT_ERR(ilu_err, SPARSE_OK);
    if (ilu_err != SPARSE_OK) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err =
        sparse_bicgstab_solve_block(A, B, nrhs, X, &opts, sparse_ilu_precond, &ilu, &result);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_ilu_free(&ilu);
        free(B);
        free(X);
        sparse_free(A);
        return;
    }
    ASSERT_TRUE(result.converged);

    for (idx_t j = 0; j < nrhs; j++) {
        double rel_res = tf_relative_residual_l2(A, B + (size_t)j * (size_t)n,
                                                 X + (size_t)j * (size_t)n, n, HUGE_VAL);
        ASSERT_TRUE(rel_res < 1e-8);
    }

    sparse_ilu_free(&ilu);
    free(B);
    free(X);
    sparse_free(A);
}

static void test_block_bicgstab_result_aggregation(void) {
    idx_t n = 10;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    idx_t nrhs = 3;
    double *B = calloc((size_t)nrhs * (size_t)n, sizeof(double));
    double *X = calloc((size_t)nrhs * (size_t)n, sizeof(double));
    if (!require_allocs(A, B, X)) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }
    for (idx_t j = 0; j < nrhs; j++)
        for (idx_t i = 0; i < n; i++)
            B[(size_t)i + (size_t)j * (size_t)n] = (double)(i + 1) * (double)(j + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t block_result;
    sparse_err_t block_err =
        sparse_bicgstab_solve_block(A, B, nrhs, X, &opts, NULL, NULL, &block_result);
    ASSERT_ERR(block_err, SPARSE_OK);
    if (block_err != SPARSE_OK) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }
    ASSERT_TRUE(block_result.converged);

    idx_t max_iters = 0;
    double max_res = 0.0;
    for (idx_t j = 0; j < nrhs; j++) {
        double *bj = B + (size_t)j * (size_t)n;
        double *x_fresh = calloc((size_t)n, sizeof(double));
        ASSERT_NOT_NULL(x_fresh);
        if (!x_fresh) {
            free(B);
            free(X);
            sparse_free(A);
            return;
        }
        sparse_iter_result_t col_result;
        sparse_err_t col_err =
            sparse_solve_bicgstab(A, bj, x_fresh, &opts, NULL, NULL, &col_result);
        ASSERT_ERR(col_err, SPARSE_OK);
        if (col_err != SPARSE_OK) {
            free(x_fresh);
            free(B);
            free(X);
            sparse_free(A);
            return;
        }
        if (col_result.iterations > max_iters)
            max_iters = col_result.iterations;
        if (col_result.residual_norm > max_res)
            max_res = col_result.residual_norm;
        free(x_fresh);
    }

    ASSERT_EQ(block_result.iterations, max_iters);
    ASSERT_NEAR(block_result.residual_norm, max_res, 1e-14);

    free(B);
    free(X);
    sparse_free(A);
}

static sparse_err_t failing_precond(const void *ctx, idx_t n, const double *r, double *z) {
    (void)ctx;
    (void)n;
    (void)r;
    (void)z;
    return SPARSE_ERR_SINGULAR;
}

static void test_block_bicgstab_error_propagation(void) {
    idx_t n = 5;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *B = calloc(2 * (size_t)n, sizeof(double));
    double *X = calloc(2 * (size_t)n, sizeof(double));
    if (!require_allocs(A, B, X)) {
        free(B);
        free(X);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++) {
        B[(size_t)i] = 1.0;
        B[(size_t)i + (size_t)n] = 2.0;
    }

    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err =
        sparse_bicgstab_solve_block(A, B, 2, X, &opts, failing_precond, NULL, &result);

    ASSERT_ERR(err, SPARSE_ERR_SINGULAR);

    free(B);
    free(X);
    sparse_free(A);
}

int main(void) {
    TEST_SUITE_BEGIN("Block BiCGSTAB");

    RUN_TEST(test_block_bicgstab_null_inputs);
    RUN_TEST(test_block_bicgstab_nrhs_zero);
    RUN_TEST(test_block_bicgstab_nrhs_negative);
    RUN_TEST(test_block_bicgstab_nonsquare);
    RUN_TEST(test_block_bicgstab_2rhs);
    RUN_TEST(test_block_bicgstab_4rhs);
    RUN_TEST(test_block_bicgstab_matches_single_rhs);
    RUN_TEST(test_block_bicgstab_mixed_convergence);
    RUN_TEST(test_block_bicgstab_nrhs_1);
    RUN_TEST(test_block_bicgstab_preconditioned);
    RUN_TEST(test_block_bicgstab_result_aggregation);
    RUN_TEST(test_block_bicgstab_error_propagation);

    TEST_SUITE_END();
}
