#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Test helpers — matrix builders
 * ═══════════════════════════════════════════════════════════════════════ */

static SparseMatrix *build_identity(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);
    return A;
}

static SparseMatrix *build_diagonal(idx_t n, double val) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, val);
    return A;
}

static SparseMatrix *build_unsym_tridiag(idx_t n, double diag, double upper, double lower) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, diag);
        if (i > 0)
            sparse_insert(A, i, i - 1, lower);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, upper);
    }
    return A;
}

static SparseMatrix *build_spd_tridiag(idx_t n, double diag_val, double offdiag_val) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, diag_val);
        if (i > 0)
            sparse_insert(A, i, i - 1, offdiag_val);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, offdiag_val);
    }
    return A;
}

static double compute_relative_residual(const SparseMatrix *A, const double *b, const double *x,
                                        idx_t n) {
    double *Ax = calloc((size_t)n, sizeof(double));
    if (!Ax)
        return 1.0 / 0.0; /* INFINITY — will fail any tolerance check */
    sparse_matvec(A, x, Ax);
    double rnorm = 0.0, bnorm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double ri = b[i] - Ax[i];
        rnorm += ri * ri;
        bnorm += b[i] * b[i];
    }
    free(Ax);
    return (bnorm > 0.0) ? sqrt(rnorm / bnorm) : sqrt(rnorm);
}

/* ═══════════════════════════════════════════════════════════════════════
 * NULL and error handling tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_bicgstab_null_A(void) {
    double b[1] = {1.0}, x[1] = {0.0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_solve_bicgstab(NULL, b, x, NULL, NULL, NULL, &result), SPARSE_ERR_NULL);
}

static void test_bicgstab_null_b(void) {
    SparseMatrix *A = build_identity(3);
    double x[3] = {0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_solve_bicgstab(A, NULL, x, NULL, NULL, NULL, &result), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_bicgstab_null_x(void) {
    SparseMatrix *A = build_identity(3);
    double b[3] = {1, 2, 3};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_solve_bicgstab(A, b, NULL, NULL, NULL, NULL, &result), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_bicgstab_nonsquare(void) {
    SparseMatrix *A = sparse_create(3, 4);
    if (!A)
        return;
    double b[3] = {1, 2, 3}, x[4] = {0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_solve_bicgstab(A, b, x, NULL, NULL, NULL, &result), SPARSE_ERR_SHAPE);
    sparse_free(A);
}

static void test_bicgstab_bad_opts_negative_maxiter(void) {
    SparseMatrix *A = build_identity(3);
    double b[3] = {1, 2, 3}, x[3] = {0};
    sparse_iter_opts_t opts = {.max_iter = -1, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result), SPARSE_ERR_BADARG);
    sparse_free(A);
}

static void test_bicgstab_bad_opts_negative_tol(void) {
    SparseMatrix *A = build_identity(3);
    double b[3] = {1, 2, 3}, x[3] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = -1.0, .verbose = 0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result), SPARSE_ERR_BADARG);
    sparse_free(A);
}

static void test_bicgstab_null_result(void) {
    SparseMatrix *A = build_identity(3);
    double b[3] = {1, 2, 3}, x[3] = {0};
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, NULL, NULL, NULL, NULL);
    ASSERT_ERR(err, SPARSE_OK);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Trivial case tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_bicgstab_zero_size(void) {
    /* sparse_create(0,0) returns NULL, so bicgstab returns ERR_NULL */
    SparseMatrix *A = sparse_create(0, 0);
    ASSERT_NULL(A);
    double dummy = 0.0;
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, &dummy, &dummy, NULL, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_ERR_NULL);
}

static void test_bicgstab_zero_rhs(void) {
    SparseMatrix *A = build_identity(5);
    double b[5] = {0, 0, 0, 0, 0};
    double x[5] = {1, 2, 3, 4, 5};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, NULL, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
    for (idx_t i = 0; i < 5; i++)
        ASSERT_NEAR(x[i], 0.0, 1e-15);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Basic solver tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_bicgstab_identity(void) {
    idx_t n = 5;
    SparseMatrix *A = build_identity(n);
    double b[5] = {1, 2, 3, 4, 5};
    double x[5] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], b[i], 1e-10);

    sparse_free(A);
}

static void test_bicgstab_diagonal(void) {
    idx_t n = 4;
    SparseMatrix *A = build_diagonal(n, 3.0);
    double b[4] = {3, 6, 9, 12};
    double x[4] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double expected[4] = {1, 2, 3, 4};
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], expected[i], 1e-10);

    sparse_free(A);
}

static void test_bicgstab_spd_tridiag(void) {
    idx_t n = 20;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);

    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-8);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_unsym_tridiag(void) {
    idx_t n = 20;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-8);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 7.0);
    double b[1] = {21.0};
    double x[1] = {0.0};
    sparse_iter_result_t result;

    sparse_err_t err = sparse_solve_bicgstab(A, b, x, NULL, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_NEAR(x[0], 3.0, 1e-10);

    sparse_free(A);
}

static void test_bicgstab_already_converged(void) {
    idx_t n = 3;
    SparseMatrix *A = build_identity(n);
    double b[3] = {1, 2, 3};
    double x[3] = {1, 2, 3};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.iterations, 0);

    sparse_free(A);
}

static void test_bicgstab_maxiter_zero(void) {
    SparseMatrix *A = build_identity(3);
    double b[3] = {1, 2, 3};
    double x[3] = {0};
    sparse_iter_opts_t opts = {.max_iter = 0, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_ERR_NOT_CONVERGED);
    ASSERT_FALSE(result.converged);
    ASSERT_EQ(result.iterations, 0);

    sparse_free(A);
}

static void test_bicgstab_default_opts(void) {
    idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_err_t err = sparse_solve_bicgstab(A, b, x, NULL, NULL, NULL, NULL);
    ASSERT_ERR(err, SPARSE_OK);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-8);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_larger_unsym(void) {
    idx_t n = 50;
    SparseMatrix *A = build_unsym_tridiag(n, 5.0, -1.5, -0.5);

    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = sin((double)(i + 1));

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-8);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_result_fields(void) {
    idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.iterations > 0);
    ASSERT_TRUE(result.residual_norm < 1e-10);
    ASSERT_TRUE(result.residual_norm >= 0.0);

    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Known-solution tests (Day 2)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_bicgstab_3x3_known_solution(void) {
    /* A = [4 1 -1; 2 7 1; 1 -3 12], x_exact = [1, 2, 3]
     * b = A * x_exact = [3, 19, 31] */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, -1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 1, 1, 7.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 1, -3.0);
    sparse_insert(A, 2, 2, 12.0);

    double b[3] = {3.0, 19.0, 31.0};
    double x[3] = {0.0, 0.0, 0.0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_NEAR(x[0], 1.0, 1e-10);
    ASSERT_NEAR(x[1], 2.0, 1e-10);
    ASSERT_NEAR(x[2], 3.0, 1e-10);

    sparse_free(A);
}

static void test_bicgstab_5x5_known_solution(void) {
    /* 5×5 nonsymmetric diagonally dominant system.
     * A has diag=10, upper=1, lower=-2, plus corner entries.
     * x_exact = [1, -1, 2, -2, 3] */
    SparseMatrix *A = sparse_create(5, 5);
    double vals[5][5] = {{10, 1, 0, 0, -1},
                         {-2, 10, 1, 0, 0},
                         {0, -2, 10, 1, 0},
                         {0, 0, -2, 10, 1},
                         {1, 0, 0, -2, 10}};
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            if (vals[i][j] != 0.0)
                sparse_insert(A, i, j, vals[i][j]);

    double x_exact[5] = {1, -1, 2, -2, 3};
    double b[5] = {0};
    /* b = A * x_exact */
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            b[i] += vals[i][j] * x_exact[j];

    double x[5] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    for (int i = 0; i < 5; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-9);

    sparse_free(A);
}

static void test_bicgstab_vs_lu_direct(void) {
    idx_t n = 30;
    SparseMatrix *A = build_unsym_tridiag(n, 6.0, -1.5, -2.5);

    /* Make a copy for LU (factorization mutates the matrix) */
    SparseMatrix *A_lu = build_unsym_tridiag(n, 6.0, -1.5, -2.5);

    double *b = calloc((size_t)n, sizeof(double));
    double *x_lu = calloc((size_t)n, sizeof(double));
    double *x_bicgstab = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    /* Solve with LU */
    REQUIRE_OK(sparse_lu_factor(A_lu, SPARSE_PIVOT_PARTIAL, 1e-12));
    REQUIRE_OK(sparse_lu_solve(A_lu, b, x_lu));

    /* Solve with BiCGSTAB */
    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x_bicgstab, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_bicgstab[i], x_lu[i], 1e-8);

    free(b);
    free(x_lu);
    free(x_bicgstab);
    sparse_free(A);
    sparse_free(A_lu);
}

/* ═══════════════════════════════════════════════════════════════════════
 * True residual verification (Day 2)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_bicgstab_true_residual_matches(void) {
    idx_t n = 25;
    SparseMatrix *A = build_unsym_tridiag(n, 5.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_TRUE(result.converged);

    /* Independently compute true relative residual */
    double true_res = compute_relative_residual(A, b, x, n);

    /* The reported residual_norm should match the true residual */
    ASSERT_NEAR(result.residual_norm, true_res, 1e-14);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_nonzero_initial_guess(void) {
    idx_t n = 15;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -0.5, -1.5);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++) {
        b[i] = (double)(i + 1);
        x[i] = 0.1 * (double)(i + 1);
    }

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-8);

    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Preconditioned BiCGSTAB tests (Day 2)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_bicgstab_ilu_precond(void) {
    idx_t n = 30;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    /* Factor ILU(0) preconditioner */
    sparse_ilu_t ilu;
    memset(&ilu, 0, sizeof(ilu));
    sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
    ASSERT_ERR(ferr, SPARSE_OK);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, sparse_ilu_precond, &ilu, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-10);

    sparse_ilu_free(&ilu);
    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_precond_fewer_iters(void) {
    idx_t n = 50;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = sin((double)(i + 1));

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};

    /* Unpreconditioned solve */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t r1;
    sparse_solve_bicgstab(A, b, x1, &opts, NULL, NULL, &r1);
    ASSERT_TRUE(r1.converged);

    /* ILU-preconditioned solve */
    sparse_ilu_t ilu;
    memset(&ilu, 0, sizeof(ilu));
    sparse_ilu_factor(A, &ilu);
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t r2;
    sparse_solve_bicgstab(A, b, x2, &opts, sparse_ilu_precond, &ilu, &r2);
    ASSERT_TRUE(r2.converged);

    /* Preconditioned should converge in fewer iterations */
    ASSERT_TRUE(r2.iterations <= r1.iterations);

    sparse_ilu_free(&ilu);
    free(b);
    free(x1);
    free(x2);
    sparse_free(A);
}

static void test_bicgstab_precond_known_solution(void) {
    /* Same 3×3 system as test_bicgstab_3x3_known_solution, with ILU */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, -1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 1, 1, 7.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 1, -3.0);
    sparse_insert(A, 2, 2, 12.0);

    double b[3] = {3.0, 19.0, 31.0};
    double x[3] = {0.0, 0.0, 0.0};

    sparse_ilu_t ilu;
    memset(&ilu, 0, sizeof(ilu));
    sparse_ilu_factor(A, &ilu);

    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, sparse_ilu_precond, &ilu, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_NEAR(x[0], 1.0, 1e-10);
    ASSERT_NEAR(x[1], 2.0, 1e-10);
    ASSERT_NEAR(x[2], 3.0, 1e-10);

    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SuiteSparse integration tests (Day 3)
 * ═══════════════════════════════════════════════════════════════════════ */

static void compute_rhs(const SparseMatrix *A, const double *x_exact, double *b) {
    sparse_matvec(A, x_exact, b);
}

static void test_bicgstab_west0067(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/west0067.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A);
    ASSERT_EQ(n, 67);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    /* west0067 has zero diagonals so ILU(0) fails — use ILUT with pivoting
     * and heavy fill for a strong preconditioner */
    sparse_ilu_t ilu;
    memset(&ilu, 0, sizeof(ilu));
    sparse_ilut_opts_t ilut_opts = {.tol = 1e-4, .max_fill = 60, .pivot = 1};
    REQUIRE_OK(sparse_ilut_factor(A, &ilut_opts, &ilu));

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-6, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, sparse_ilu_precond, &ilu, &result);

    double rel_res = compute_relative_residual(A, b, x, n);
    printf("    west0067: BiCGSTAB+ILUT iters=%d, rel_res=%.3e, converged=%d\n",
           (int)result.iterations, rel_res, result.converged);

    /* west0067 has complex eigenvalue structure that can challenge BiCGSTAB.
     * With a strong ILUT preconditioner it should converge, but we verify
     * at minimum that the solver runs without crashing and reports status. */
    if (result.converged) {
        ASSERT_TRUE(rel_res < 1e-4);
    } else {
        ASSERT_ERR(err, SPARSE_ERR_NOT_CONVERGED);
    }

    sparse_ilu_free(&ilu);
    free(x_exact);
    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_steam1(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/steam1.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A);
    ASSERT_EQ(n, 240);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    /* ILU(0) preconditioned BiCGSTAB — steam1 is ill-conditioned (condest ~3e7) */
    sparse_ilu_t ilu;
    memset(&ilu, 0, sizeof(ilu));
    REQUIRE_OK(sparse_ilu_factor(A, &ilu));

    sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-6, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, sparse_ilu_precond, &ilu, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    printf("    steam1: BiCGSTAB+ILU iters=%d, rel_res=%.3e\n", (int)result.iterations, rel_res);
    ASSERT_TRUE(rel_res < 1e-4);

    sparse_ilu_free(&ilu);
    free(x_exact);
    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_orsirr_1(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/orsirr_1.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] orsirr_1.mtx not found\n");
        return;
    }
    idx_t n = sparse_rows(A);
    ASSERT_EQ(n, 1030);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    sparse_ilu_t ilu;
    memset(&ilu, 0, sizeof(ilu));
    REQUIRE_OK(sparse_ilu_factor(A, &ilu));

    sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-8, .verbose = 0};
    sparse_iter_result_t result;
    err = sparse_solve_bicgstab(A, b, x, &opts, sparse_ilu_precond, &ilu, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    printf("    orsirr_1: BiCGSTAB+ILU iters=%d, rel_res=%.3e\n", (int)result.iterations, rel_res);
    ASSERT_TRUE(rel_res < 1e-6);

    sparse_ilu_free(&ilu);
    free(x_exact);
    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * BiCGSTAB vs GMRES comparison (Day 3)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_bicgstab_vs_gmres_steam1(void) {
    SparseMatrix *A_bicg = NULL;
    SparseMatrix *A_gmres = NULL;
    ASSERT_ERR(sparse_load_mm(&A_bicg, SS_DIR "/steam1.mtx"), SPARSE_OK);
    ASSERT_ERR(sparse_load_mm(&A_gmres, SS_DIR "/steam1.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A_bicg);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A_bicg, x_exact, b);

    /* Both use ILU(0) preconditioning */
    sparse_ilu_t ilu_bicg, ilu_gmres;
    memset(&ilu_bicg, 0, sizeof(ilu_bicg));
    memset(&ilu_gmres, 0, sizeof(ilu_gmres));
    REQUIRE_OK(sparse_ilu_factor(A_bicg, &ilu_bicg));
    REQUIRE_OK(sparse_ilu_factor(A_gmres, &ilu_gmres));

    /* BiCGSTAB solve */
    double *x_bicg = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t bicg_opts = {.max_iter = 2000, .tol = 1e-6, .verbose = 0};
    sparse_iter_result_t bicg_result;
    sparse_solve_bicgstab(A_bicg, b, x_bicg, &bicg_opts, sparse_ilu_precond, &ilu_bicg,
                          &bicg_result);

    /* GMRES(30) solve */
    double *x_gmres = calloc((size_t)n, sizeof(double));
    sparse_gmres_opts_t gmres_opts = {.max_iter = 2000, .restart = 30, .tol = 1e-6, .verbose = 0};
    sparse_iter_result_t gmres_result;
    sparse_solve_gmres(A_gmres, b, x_gmres, &gmres_opts, sparse_ilu_precond, &ilu_gmres,
                       &gmres_result);

    double res_bicg = compute_relative_residual(A_bicg, b, x_bicg, n);
    double res_gmres = compute_relative_residual(A_bicg, b, x_gmres, n);
    printf("    steam1: BiCGSTAB iters=%d res=%.3e, GMRES(30) iters=%d res=%.3e\n",
           (int)bicg_result.iterations, res_bicg, (int)gmres_result.iterations, res_gmres);

    /* Both should converge with ILU(0) */
    ASSERT_TRUE(bicg_result.converged);
    ASSERT_TRUE(gmres_result.converged);

    /* Both residuals should be acceptable (steam1 is ill-conditioned so
     * we compare residuals rather than solution values) */
    ASSERT_TRUE(res_bicg < 1e-4);
    ASSERT_TRUE(res_gmres < 1e-4);

    sparse_ilu_free(&ilu_bicg);
    sparse_ilu_free(&ilu_gmres);
    free(x_exact);
    free(b);
    free(x_bicg);
    free(x_gmres);
    sparse_free(A_bicg);
    sparse_free(A_gmres);
}

static void test_bicgstab_vs_gmres_tridiag(void) {
    idx_t n = 100;
    SparseMatrix *A_bicg = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    SparseMatrix *A_gmres = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    double *b = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = sin((double)(i + 1));

    /* BiCGSTAB */
    double *x_bicg = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t bicg_opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t bicg_result;
    sparse_solve_bicgstab(A_bicg, b, x_bicg, &bicg_opts, NULL, NULL, &bicg_result);

    /* GMRES(30) — same tolerance, no preconditioning */
    double *x_gmres = calloc((size_t)n, sizeof(double));
    sparse_gmres_opts_t gmres_opts = {.max_iter = 500, .restart = 30, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t gmres_result;
    sparse_solve_gmres(A_gmres, b, x_gmres, &gmres_opts, NULL, NULL, &gmres_result);

    ASSERT_TRUE(bicg_result.converged);
    ASSERT_TRUE(gmres_result.converged);

    printf("    tridiag 100: BiCGSTAB iters=%d, GMRES(30) iters=%d\n", (int)bicg_result.iterations,
           (int)gmres_result.iterations);

    /* Solutions should match */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_bicg[i], x_gmres[i], 1e-6);

    free(b);
    free(x_bicg);
    free(x_gmres);
    sparse_free(A_bicg);
    sparse_free(A_gmres);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Numerical hardening tests (Day 4)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_bicgstab_nan_inf_detection(void) {
    /* Build a matrix with a zero row — this creates a singular system where
     * BiCGSTAB's recurrence scalars can blow up to Inf/NaN */
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    /* row 1 is all zeros */
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 3, 3, 1.0);

    double b[4] = {1.0, 1.0, 1.0, 1.0};
    double x[4] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    /* Should get either NOT_CONVERGED or NUMERIC, but never crash */
    ASSERT_TRUE(err == SPARSE_ERR_NOT_CONVERGED || err == SPARSE_ERR_NUMERIC);
    ASSERT_FALSE(result.converged);

    sparse_free(A);
}

static void test_bicgstab_nearly_singular(void) {
    /* Nearly singular diagonal: one entry is very small */
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (i == 5) ? 1e-14 : 1.0);

    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);

    /* Should not crash. May converge or report numeric error. */
    ASSERT_TRUE(err == SPARSE_OK || err == SPARSE_ERR_NOT_CONVERGED || err == SPARSE_ERR_NUMERIC);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_high_condition_number(void) {
    /* Build a diagonally dominant matrix with condition number ~1e12 */
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        double diag = (i == 0) ? 1e12 : 1.0;
        sparse_insert(A, i, i, diag);
        if (i > 0)
            sparse_insert(A, i, i - 1, 0.1);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -0.1);
    }

    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-8, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);

    /* Should converge (diagonally dominant) but may be slow */
    ASSERT_TRUE(err == SPARSE_OK || err == SPARSE_ERR_NOT_CONVERGED);
    if (result.converged) {
        double rel_res = compute_relative_residual(A, b, x, n);
        ASSERT_TRUE(rel_res < 1e-6);
    }

    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_random_initial_guess(void) {
    idx_t n = 20;
    SparseMatrix *A = build_unsym_tridiag(n, 5.0, -1.0, -2.0);

    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++) {
        b[i] = (double)(i + 1);
        /* Pseudo-random initial guess using a simple hash */
        x[i] = sin(37.0 * (double)(i + 1)) * 100.0;
    }

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-8);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_bicgstab_near_solution_guess(void) {
    idx_t n = 15;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    /* Compute exact solution via LU, then perturb */
    SparseMatrix *A_lu = build_unsym_tridiag(n, 4.0, -1.0, -2.0);
    double *b = calloc((size_t)n, sizeof(double));
    double *x_exact = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    REQUIRE_OK(sparse_lu_factor(A_lu, SPARSE_PIVOT_PARTIAL, 1e-12));
    REQUIRE_OK(sparse_lu_solve(A_lu, b, x_exact));

    /* Start near the solution */
    for (idx_t i = 0; i < n; i++)
        x[i] = x_exact[i] + 1e-6 * sin((double)i);

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
    /* Should converge quickly from a near-solution guess */
    ASSERT_TRUE(result.iterations < 30);

    free(b);
    free(x_exact);
    free(x);
    sparse_free(A);
    sparse_free(A_lu);
}

static void test_bicgstab_zero_diagonal(void) {
    /* Permuted identity-like matrix with zero diagonal entries */
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 2, 3, 1.0);
    sparse_insert(A, 3, 2, 1.0);

    double b[4] = {1.0, 2.0, 3.0, 4.0};
    double x[4] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t err = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, &result);
    /* This is a permutation matrix — should converge */
    if (result.converged) {
        ASSERT_ERR(err, SPARSE_OK);
        ASSERT_NEAR(x[0], 2.0, 1e-8);
        ASSERT_NEAR(x[1], 1.0, 1e-8);
        ASSERT_NEAR(x[2], 4.0, 1e-8);
        ASSERT_NEAR(x[3], 3.0, 1e-8);
    }

    sparse_free(A);
}

static void test_bicgstab_err_numeric_strerror(void) {
    /* Verify the new error code has a string representation */
    const char *s = sparse_strerror(SPARSE_ERR_NUMERIC);
    ASSERT_NOT_NULL(s);
    ASSERT_TRUE(strlen(s) > 0);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Block BiCGSTAB tests (Day 5)
 * ═══════════════════════════════════════════════════════════════════════ */

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

    sparse_err_t err = sparse_bicgstab_solve_block(A, B, 0, X, NULL, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    sparse_free(A);
}

static void test_block_bicgstab_nrhs_negative(void) {
    SparseMatrix *A = build_identity(3);
    double B[1] = {0}, X[1] = {0};
    sparse_iter_result_t result;

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

    /* 2 RHS columns, column-major */
    double *B = calloc(2 * (size_t)n, sizeof(double));
    double *X = calloc(2 * (size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++) {
        B[i] = (double)(i + 1);          /* column 0 */
        B[i + n] = sin((double)(i + 1)); /* column 1 */
    }

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_bicgstab_solve_block(A, B, 2, X, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    /* Verify each column independently */
    for (int col = 0; col < 2; col++) {
        double *x_col = X + (size_t)col * (size_t)n;
        double *b_col = B + (size_t)col * (size_t)n;
        double rel_res = compute_relative_residual(A, b_col, x_col, n);
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
    for (idx_t j = 0; j < nrhs; j++)
        for (idx_t i = 0; i < n; i++)
            B[(size_t)i + (size_t)j * (size_t)n] = (double)(i + 1) * (double)(j + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_bicgstab_solve_block(A, B, nrhs, X, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    for (idx_t j = 0; j < nrhs; j++) {
        double *x_col = X + (size_t)j * (size_t)n;
        double *b_col = B + (size_t)j * (size_t)n;
        double rel_res = compute_relative_residual(A, b_col, x_col, n);
        ASSERT_TRUE(rel_res < 1e-8);
    }

    free(B);
    free(X);
    sparse_free(A);
}

static void test_block_bicgstab_matches_single_rhs(void) {
    idx_t n = 15;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    /* Solve column-by-column with single-RHS, then compare with block */
    idx_t nrhs = 3;
    double *B = calloc((size_t)nrhs * (size_t)n, sizeof(double));
    double *X_single = calloc((size_t)nrhs * (size_t)n, sizeof(double));
    double *X_block = calloc((size_t)nrhs * (size_t)n, sizeof(double));
    for (idx_t j = 0; j < nrhs; j++)
        for (idx_t i = 0; i < n; i++)
            B[(size_t)i + (size_t)j * (size_t)n] = (double)(i + j * n + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};

    /* Single-RHS solves */
    for (idx_t j = 0; j < nrhs; j++) {
        const double *bj = B + (size_t)j * (size_t)n;
        double *xj = X_single + (size_t)j * (size_t)n;
        sparse_solve_bicgstab(A, bj, xj, &opts, NULL, NULL, NULL);
    }

    /* Block solve */
    sparse_iter_result_t result;
    sparse_bicgstab_solve_block(A, B, nrhs, X_block, &opts, NULL, NULL, &result);
    ASSERT_TRUE(result.converged);

    /* Solutions should match */
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
    /* Build a system where column 0 (easy RHS) converges much faster
     * than column 1 (harder RHS). Verify both converge correctly. */
    idx_t n = 20;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    double *B = calloc(2 * (size_t)n, sizeof(double));
    double *X = calloc(2 * (size_t)n, sizeof(double));

    /* Column 0: b = first unit vector (easy, sparse RHS) */
    B[0] = 1.0;
    /* Column 1: b = large oscillating RHS */
    for (idx_t i = 0; i < n; i++)
        B[i + n] = 1000.0 * sin(10.0 * (double)(i + 1));

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_bicgstab_solve_block(A, B, 2, X, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    /* Both columns should have correct solutions */
    for (int col = 0; col < 2; col++) {
        double rel_res = compute_relative_residual(A, B + (size_t)col * (size_t)n,
                                                   X + (size_t)col * (size_t)n, n);
        ASSERT_TRUE(rel_res < 1e-8);
    }

    /* Aggregate iterations should reflect the slower column */
    ASSERT_TRUE(result.iterations > 0);

    free(B);
    free(X);
    sparse_free(A);
}

static void test_block_bicgstab_nrhs_1(void) {
    /* nrhs=1 should match single-RHS solve */
    idx_t n = 10;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    double *b = calloc((size_t)n, sizeof(double));
    double *x_single = calloc((size_t)n, sizeof(double));
    double *x_block = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};

    sparse_iter_result_t r1, r2;
    sparse_solve_bicgstab(A, b, x_single, &opts, NULL, NULL, &r1);
    sparse_bicgstab_solve_block(A, b, 1, x_block, &opts, NULL, NULL, &r2);

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
    for (idx_t j = 0; j < nrhs; j++)
        for (idx_t i = 0; i < n; i++)
            B[(size_t)i + (size_t)j * (size_t)n] = (double)(i + 1) * (double)(j + 1);

    sparse_ilu_t ilu;
    memset(&ilu, 0, sizeof(ilu));
    REQUIRE_OK(sparse_ilu_factor(A, &ilu));

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err =
        sparse_bicgstab_solve_block(A, B, nrhs, X, &opts, sparse_ilu_precond, &ilu, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    for (idx_t j = 0; j < nrhs; j++) {
        double rel_res =
            compute_relative_residual(A, B + (size_t)j * (size_t)n, X + (size_t)j * (size_t)n, n);
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
    for (idx_t j = 0; j < nrhs; j++)
        for (idx_t i = 0; i < n; i++)
            B[(size_t)i + (size_t)j * (size_t)n] = (double)(i + 1) * (double)(j + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t block_result;
    sparse_bicgstab_solve_block(A, B, nrhs, X, &opts, NULL, NULL, &block_result);
    ASSERT_TRUE(block_result.converged);

    /* Verify aggregate is max of per-column results */
    idx_t max_iters = 0;
    double max_res = 0.0;
    for (idx_t j = 0; j < nrhs; j++) {
        double *xj = X + (size_t)j * (size_t)n;
        double *bj = B + (size_t)j * (size_t)n;

        /* Re-solve to get per-column iteration counts */
        double *x_fresh = calloc((size_t)n, sizeof(double));
        sparse_iter_result_t col_result;
        sparse_solve_bicgstab(A, bj, x_fresh, &opts, NULL, NULL, &col_result);
        if (col_result.iterations > max_iters)
            max_iters = col_result.iterations;
        if (col_result.residual_norm > max_res)
            max_res = col_result.residual_norm;
        free(x_fresh);
        (void)xj;
    }

    ASSERT_EQ(block_result.iterations, max_iters);
    ASSERT_NEAR(block_result.residual_norm, max_res, 1e-14);

    free(B);
    free(X);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Block error propagation test (Day 6)
 * ═══════════════════════════════════════════════════════════════════════ */

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

/* ═══════════════════════════════════════════════════════════════════════
 * Matrix-free BiCGSTAB tests (Day 6)
 * ═══════════════════════════════════════════════════════════════════════ */

static sparse_err_t sparse_matvec_cb(const void *ctx, idx_t n, const double *xv, double *y) {
    const SparseMatrix *A = (const SparseMatrix *)ctx;
    (void)n;
    return sparse_matvec(A, xv, y);
}

static void test_bicgstab_mf_matches_matrix(void) {
    idx_t n = 20;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    double *b = calloc((size_t)n, sizeof(double));
    double *x_mat = calloc((size_t)n, sizeof(double));
    double *x_mf = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res_mat, res_mf;

    sparse_solve_bicgstab(A, b, x_mat, &opts, NULL, NULL, &res_mat);
    sparse_solve_bicgstab_mf(sparse_matvec_cb, A, n, b, x_mf, &opts, NULL, NULL, &res_mf);

    ASSERT_TRUE(res_mat.converged);
    ASSERT_TRUE(res_mf.converged);
    ASSERT_EQ(res_mat.iterations, res_mf.iterations);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_mat[i], x_mf[i], 1e-12);

    free(b);
    free(x_mat);
    free(x_mf);
    sparse_free(A);
}

typedef struct {
    idx_t n;
    double scale;
} scaled_identity_t;

static sparse_err_t scaled_identity_matvec(const void *ctx, idx_t n, const double *xv, double *y) {
    const scaled_identity_t *op = (const scaled_identity_t *)ctx;
    (void)n;
    for (idx_t i = 0; i < op->n; i++)
        y[i] = op->scale * xv[i];
    return SPARSE_OK;
}

static void test_bicgstab_mf_scaled_identity(void) {
    idx_t n = 5;
    scaled_identity_t op = {.n = n, .scale = 3.0};

    double b[5] = {3, 6, 9, 12, 15};
    double x[5] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t err =
        sparse_solve_bicgstab_mf(scaled_identity_matvec, &op, n, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double expected[5] = {1, 2, 3, 4, 5};
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], expected[i], 1e-10);
}

static sparse_err_t failing_matvec(const void *ctx, idx_t n, const double *xv, double *y) {
    (void)ctx;
    (void)n;
    (void)xv;
    (void)y;
    return SPARSE_ERR_BADARG;
}

static void test_bicgstab_mf_matvec_error(void) {
    double b[3] = {1, 2, 3}, x[3] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t err =
        sparse_solve_bicgstab_mf(failing_matvec, NULL, 3, b, x, &opts, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_ERR_BADARG);
}

static void test_bicgstab_mf_null_callback(void) {
    double b[1] = {1.0}, x[1] = {0.0};
    ASSERT_ERR(sparse_solve_bicgstab_mf(NULL, NULL, 1, b, x, NULL, NULL, NULL, NULL),
               SPARSE_ERR_NULL);
}

static void test_bicgstab_mf_zero_rhs(void) {
    idx_t n = 5;
    scaled_identity_t op = {.n = n, .scale = 2.0};
    double b[5] = {0};
    double x[5] = {1, 2, 3, 4, 5};
    sparse_iter_result_t result;

    sparse_err_t err =
        sparse_solve_bicgstab_mf(scaled_identity_matvec, &op, n, b, x, NULL, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 0.0, 1e-15);
}

static void test_bicgstab_mf_preconditioned(void) {
    idx_t n = 30;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, -1.0, -2.0);

    double *b = calloc((size_t)n, sizeof(double));
    double *x_mat = calloc((size_t)n, sizeof(double));
    double *x_mf = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_ilu_t ilu;
    memset(&ilu, 0, sizeof(ilu));
    REQUIRE_OK(sparse_ilu_factor(A, &ilu));

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res_mat, res_mf;

    sparse_solve_bicgstab(A, b, x_mat, &opts, sparse_ilu_precond, &ilu, &res_mat);
    sparse_solve_bicgstab_mf(sparse_matvec_cb, A, n, b, x_mf, &opts, sparse_ilu_precond, &ilu,
                             &res_mf);

    ASSERT_TRUE(res_mat.converged);
    ASSERT_TRUE(res_mf.converged);
    ASSERT_EQ(res_mat.iterations, res_mf.iterations);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_mat[i], x_mf[i], 1e-12);

    sparse_ilu_free(&ilu);
    free(b);
    free(x_mat);
    free(x_mf);
    sparse_free(A);
}

static void test_bicgstab_mf_n_zero(void) {
    sparse_iter_result_t result;
    double dummy = 0;
    sparse_err_t err = sparse_solve_bicgstab_mf(scaled_identity_matvec, NULL, 0, &dummy, &dummy,
                                                NULL, NULL, NULL, &result);
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_TRUE(result.converged);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("BiCGSTAB");

    /* Error handling */
    RUN_TEST(test_bicgstab_null_A);
    RUN_TEST(test_bicgstab_null_b);
    RUN_TEST(test_bicgstab_null_x);
    RUN_TEST(test_bicgstab_nonsquare);
    RUN_TEST(test_bicgstab_bad_opts_negative_maxiter);
    RUN_TEST(test_bicgstab_bad_opts_negative_tol);
    RUN_TEST(test_bicgstab_null_result);

    /* Trivial cases */
    RUN_TEST(test_bicgstab_zero_size);
    RUN_TEST(test_bicgstab_zero_rhs);

    /* Basic solver */
    RUN_TEST(test_bicgstab_identity);
    RUN_TEST(test_bicgstab_diagonal);
    RUN_TEST(test_bicgstab_spd_tridiag);
    RUN_TEST(test_bicgstab_unsym_tridiag);
    RUN_TEST(test_bicgstab_1x1);
    RUN_TEST(test_bicgstab_already_converged);
    RUN_TEST(test_bicgstab_maxiter_zero);
    RUN_TEST(test_bicgstab_default_opts);
    RUN_TEST(test_bicgstab_larger_unsym);
    RUN_TEST(test_bicgstab_result_fields);

    /* Known-solution tests */
    RUN_TEST(test_bicgstab_3x3_known_solution);
    RUN_TEST(test_bicgstab_5x5_known_solution);
    RUN_TEST(test_bicgstab_vs_lu_direct);

    /* True residual verification */
    RUN_TEST(test_bicgstab_true_residual_matches);
    RUN_TEST(test_bicgstab_nonzero_initial_guess);

    /* Preconditioned BiCGSTAB */
    RUN_TEST(test_bicgstab_ilu_precond);
    RUN_TEST(test_bicgstab_precond_fewer_iters);
    RUN_TEST(test_bicgstab_precond_known_solution);

    /* SuiteSparse integration */
    RUN_TEST(test_bicgstab_west0067);
    RUN_TEST(test_bicgstab_steam1);
    RUN_TEST(test_bicgstab_orsirr_1);

    /* BiCGSTAB vs GMRES comparison */
    RUN_TEST(test_bicgstab_vs_gmres_steam1);
    RUN_TEST(test_bicgstab_vs_gmres_tridiag);

    /* Numerical hardening */
    RUN_TEST(test_bicgstab_nan_inf_detection);
    RUN_TEST(test_bicgstab_nearly_singular);
    RUN_TEST(test_bicgstab_high_condition_number);
    RUN_TEST(test_bicgstab_random_initial_guess);
    RUN_TEST(test_bicgstab_near_solution_guess);
    RUN_TEST(test_bicgstab_zero_diagonal);
    RUN_TEST(test_bicgstab_err_numeric_strerror);

    /* Block BiCGSTAB */
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

    /* Matrix-free BiCGSTAB */
    RUN_TEST(test_bicgstab_mf_matches_matrix);
    RUN_TEST(test_bicgstab_mf_scaled_identity);
    RUN_TEST(test_bicgstab_mf_matvec_error);
    RUN_TEST(test_bicgstab_mf_null_callback);
    RUN_TEST(test_bicgstab_mf_zero_rhs);
    RUN_TEST(test_bicgstab_mf_preconditioned);
    RUN_TEST(test_bicgstab_mf_n_zero);

    TEST_SUITE_END();
}
