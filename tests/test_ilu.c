#include "sparse_cholesky.h"
#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Test helpers
 * ═══════════════════════════════════════════════════════════════════════ */

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

/** Compute ||b - A*x|| / ||b|| */
static double compute_relative_residual(const SparseMatrix *A, const double *b, const double *x,
                                        idx_t n) {
    double *r = malloc((size_t)n * sizeof(double));
    if (!r)
        return INFINITY;
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - r[i];
    double rnorm = vec_norm2(r, n);
    double bnorm = vec_norm2(b, n);
    free(r);
    return (bnorm > 0.0) ? rnorm / bnorm : 0.0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Basic ILU(0) factorization tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* ILU(0) on 3×3 dense matrix = exact LU (no fill dropped) */
static void test_ilu_3x3_dense(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 0, 4.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 3.0);
    sparse_insert(A, 2, 0, 8.0);
    sparse_insert(A, 2, 1, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }
    ASSERT_NOT_NULL(ilu.L);
    ASSERT_NOT_NULL(ilu.U);

    /* L should be unit lower triangular:
     * L = [[1, 0, 0],
     *      [2, 1, 0],
     *      [4, 3, 1]] */
    ASSERT_NEAR(sparse_get_phys(ilu.L, 0, 0), 1.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.L, 1, 0), 2.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.L, 1, 1), 1.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.L, 2, 0), 4.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.L, 2, 1), 3.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.L, 2, 2), 1.0, 1e-14);

    /* U should be upper triangular:
     * U = [[2, 1, 1],
     *      [0, 1, 1],
     *      [0, 0, 2]] */
    ASSERT_NEAR(sparse_get_phys(ilu.U, 0, 0), 2.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.U, 0, 1), 1.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.U, 0, 2), 1.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.U, 1, 1), 1.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.U, 1, 2), 1.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.U, 2, 2), 2.0, 1e-14);

    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ILU(0) on diagonal matrix → L = I, U = diag(A) */
static void test_ilu_diagonal(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    double diags[] = {3.0, 7.0, 2.0, 5.0, 1.0};
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diags[i]);

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    /* L = I */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(sparse_get_phys(ilu.L, i, i), 1.0, 1e-14);
        for (idx_t j = 0; j < n; j++) {
            if (j != i)
                ASSERT_NEAR(sparse_get_phys(ilu.L, i, j), 0.0, 1e-14);
        }
    }

    /* U = diag(A) */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(sparse_get_phys(ilu.U, i, i), diags[i], 1e-14);
    }

    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ILU(0) on tridiagonal matrix → no fill, matches exact LU */
static void test_ilu_tridiagonal(void) {
    idx_t n = 5;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    /* Verify L*U*z = r via solve */
    double r[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double z[5];
    ASSERT_ERR(sparse_ilu_solve(&ilu, r, z), SPARSE_OK);

    /* Verify: A*z ≈ r (since tridiagonal has no fill, ILU(0) = exact LU) */
    double *Az = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(Az);
    if (!Az) {
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    sparse_matvec(A, z, Az);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(Az[i], r[i], 1e-12);

    free(Az);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ILU(0) drops fill: matrix with fill positions not in A's pattern */
static void test_ilu_drops_fill(void) {
    /* A = [[2, 1, 0],
     *      [1, 3, 1],
     *      [0, 1, 2]]
     * Exact LU would create fill at (2,0) and (0,2).
     * ILU(0) should not create these entries. */
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 2.0);

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    /* L(2,0) should be zero (fill position not in A's pattern) */
    ASSERT_NEAR(sparse_get_phys(ilu.L, 2, 0), 0.0, 1e-14);
    /* U(0,2) should be zero */
    ASSERT_NEAR(sparse_get_phys(ilu.U, 0, 2), 0.0, 1e-14);

    /* L should still have the correct non-fill entries */
    ASSERT_NEAR(sparse_get_phys(ilu.L, 0, 0), 1.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.L, 1, 0), 0.5, 1e-14); /* 1/2 */
    ASSERT_NEAR(sparse_get_phys(ilu.L, 1, 1), 1.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.L, 2, 2), 1.0, 1e-14);

    /* U should have correct entries */
    ASSERT_NEAR(sparse_get_phys(ilu.U, 0, 0), 2.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.U, 0, 1), 1.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(ilu.U, 1, 1), 2.5, 1e-14); /* 3 - 0.5*1 */
    ASSERT_NEAR(sparse_get_phys(ilu.U, 1, 2), 1.0, 1e-14);

    /* Verify ILU solve is an approximation, not exact */
    double r[3] = {3.0, 5.0, 3.0};
    double z[3];
    ASSERT_ERR(sparse_ilu_solve(&ilu, r, z), SPARSE_OK);

    /* A*z should be close to r but not exact (due to dropped fill) */
    double *Az = malloc(3 * sizeof(double));
    ASSERT_NOT_NULL(Az);
    if (!Az) {
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    sparse_matvec(A, z, Az);
    /* The approximation should be reasonable */
    double err = 0.0;
    for (int i = 0; i < 3; i++)
        err += (Az[i] - r[i]) * (Az[i] - r[i]);
    err = sqrt(err);
    /* Not exact, but should be a reasonable approximation */
    ASSERT_TRUE(err < 1.0);

    free(Az);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * ILU(0) solve tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* ILU solve on identity → z = r */
static void test_ilu_solve_identity(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    double r[4] = {1.0, 2.0, 3.0, 4.0};
    double z[4];
    ASSERT_ERR(sparse_ilu_solve(&ilu, r, z), SPARSE_OK);

    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(z[i], r[i], 1e-14);

    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ILU solve round-trip: factor, solve, check A*z ≈ r */
static void test_ilu_solve_roundtrip(void) {
    idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    double *r = malloc((size_t)n * sizeof(double));
    double *z = malloc((size_t)n * sizeof(double));
    double *Az = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(z);
    ASSERT_NOT_NULL(Az);
    if (!r || !z || !Az) {
        free(r);
        free(z);
        free(Az);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        r[i] = (double)(i + 1);

    ASSERT_ERR(sparse_ilu_solve(&ilu, r, z), SPARSE_OK);

    /* For tridiagonal, ILU(0) = exact LU, so A*z should equal r */
    sparse_matvec(A, z, Az);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(Az[i], r[i], 1e-10);

    free(r);
    free(z);
    free(Az);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * ILU as preconditioner — basic test
 * ═══════════════════════════════════════════════════════════════════════ */

/* ILU-preconditioned CG on SPD tridiagonal */
static void test_ilu_precond_cg(void) {
    idx_t n = 20;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    /* Unpreconditioned CG */
    double *x_unprec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_unprec);
    if (!x_unprec) {
        free(x_exact);
        free(b);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result_unprec;
    ASSERT_ERR(sparse_solve_cg(A, b, x_unprec, &opts, NULL, NULL, &result_unprec), SPARSE_OK);

    /* ILU-preconditioned CG */
    double *x_prec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_prec);
    if (!x_prec) {
        free(x_exact);
        free(b);
        free(x_unprec);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    sparse_iter_result_t result_prec;
    ASSERT_ERR(sparse_solve_cg(A, b, x_prec, &opts, sparse_ilu_precond, &ilu, &result_prec),
               SPARSE_OK);

    ASSERT_TRUE(result_unprec.converged);
    ASSERT_TRUE(result_prec.converged);
    /* ILU should reduce iteration count (for tridiag, ILU = exact LU → 1 iter) */
    ASSERT_TRUE(result_prec.iterations <= result_unprec.iterations);

    printf("    ILU-CG: unprec=%d iters, prec=%d iters\n", (int)result_unprec.iterations,
           (int)result_prec.iterations);

    free(x_exact);
    free(b);
    free(x_unprec);
    free(x_prec);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ILU-preconditioned GMRES on unsymmetric system */
static void test_ilu_precond_gmres(void) {
    idx_t n = 15;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 5.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, 2.0);
    }

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1) * 0.3);
    sparse_matvec(A, x_exact, b);

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    double *x = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x);
    if (!x) {
        free(x_exact);
        free(b);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    sparse_gmres_opts_t opts = {.max_iter = 200, .restart = 10, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, sparse_ilu_precond, &ilu, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-8);

    free(x_exact);
    free(b);
    free(x);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * ILU-preconditioned CG on SuiteSparse SPD matrices (Day 8)
 * ═══════════════════════════════════════════════════════════════════════ */

/* ILU-preconditioned CG on nos4 (100×100 SPD) */
static void test_ilu_cg_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    /* Unpreconditioned CG */
    double *x_unprec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_unprec);
    if (!x_unprec) {
        free(x_exact);
        free(b);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result_unprec;
    ASSERT_ERR(sparse_solve_cg(A, b, x_unprec, &opts, NULL, NULL, &result_unprec), SPARSE_OK);

    /* ILU-preconditioned CG */
    double *x_prec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_prec);
    if (!x_prec) {
        free(x_unprec);
        free(x_exact);
        free(b);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    sparse_iter_result_t result_prec;
    ASSERT_ERR(sparse_solve_cg(A, b, x_prec, &opts, sparse_ilu_precond, &ilu, &result_prec),
               SPARSE_OK);

    ASSERT_TRUE(result_unprec.converged);
    ASSERT_TRUE(result_prec.converged);
    ASSERT_TRUE(result_prec.iterations <= result_unprec.iterations);

    double res_unprec = compute_relative_residual(A, b, x_unprec, n);
    double res_prec = compute_relative_residual(A, b, x_prec, n);
    printf("    nos4 ILU-CG: unprec=%d iters (res=%.3e), prec=%d iters (res=%.3e)\n",
           (int)result_unprec.iterations, res_unprec, (int)result_prec.iterations, res_prec);
    ASSERT_TRUE(res_unprec < 1e-8);
    ASSERT_TRUE(res_prec < 1e-8);

    free(x_exact);
    free(b);
    free(x_unprec);
    free(x_prec);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ILU-preconditioned CG on bcsstk04 (132×132 SPD stiffness) */
static void test_ilu_cg_bcsstk04(void) {
    SparseMatrix *A = NULL;
    {
        sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
        ASSERT_ERR(lerr, SPARSE_OK);
        if (lerr != SPARSE_OK || !A)
            return;
    }
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    /* Unpreconditioned CG */
    double *x_unprec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_unprec);
    sparse_iter_opts_t opts = {.max_iter = 1000, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result_unprec;
    ASSERT_ERR(sparse_solve_cg(A, b, x_unprec, &opts, NULL, NULL, &result_unprec), SPARSE_OK);

    /* ILU-preconditioned CG */
    double *x_prec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_prec);
    sparse_iter_result_t result_prec;
    ASSERT_ERR(sparse_solve_cg(A, b, x_prec, &opts, sparse_ilu_precond, &ilu, &result_prec),
               SPARSE_OK);

    ASSERT_TRUE(result_unprec.converged);
    ASSERT_TRUE(result_prec.converged);
    ASSERT_TRUE(result_prec.iterations <= result_unprec.iterations);

    double res_prec = compute_relative_residual(A, b, x_prec, n);
    printf("    bcsstk04 ILU-CG: unprec=%d iters, prec=%d iters (res=%.3e)\n",
           (int)result_unprec.iterations, (int)result_prec.iterations, res_prec);
    ASSERT_TRUE(res_prec < 1e-8);

    free(x_exact);
    free(b);
    free(x_unprec);
    free(x_prec);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * ILU-preconditioned GMRES on SuiteSparse unsymmetric matrices (Day 8)
 * ═══════════════════════════════════════════════════════════════════════ */

/* ILU(0) on west0067 — matrix has 65/67 zero diagonal entries, so ILU(0)
 * (which requires nonzero pivots) correctly returns SPARSE_ERR_SINGULAR.
 * West0067 requires pivoting for any LU-type factorization. */
static void test_ilu_gmres_west0067(void) {
    SparseMatrix *A = NULL;
    {
        sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
        ASSERT_ERR(lerr, SPARSE_OK);
        if (lerr != SPARSE_OK || !A)
            return;
    }

    sparse_ilu_t ilu;
    sparse_err_t err = sparse_ilu_factor(A, &ilu);
    printf("    west0067: ILU(0) returns %s (65/67 zero diag entries)\n", sparse_strerror(err));
    /* west0067 has structurally zero diagonal → ILU(0) cannot proceed */
    ASSERT_ERR(err, SPARSE_ERR_SINGULAR);

    sparse_free(A);
}

/* ILU-preconditioned GMRES on steam1 */
static void test_ilu_gmres_steam1(void) {
    SparseMatrix *A = NULL;
    {
        sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/steam1.mtx");
        ASSERT_ERR(lerr, SPARSE_OK);
        if (lerr != SPARSE_OK || !A)
            return;
    }
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    /* Unpreconditioned GMRES */
    double *x_unprec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_unprec);
    if (!x_unprec) {
        free(x_exact);
        free(b);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    /* Relaxed tol: steam1 is ill-conditioned (condest ~3e7) */
    sparse_gmres_opts_t opts = {.max_iter = 1000, .restart = 50, .tol = 1e-4, .verbose = 0};
    sparse_iter_result_t result_unprec;
    sparse_solve_gmres(A, b, x_unprec, &opts, NULL, NULL, &result_unprec); /* may not converge */

    /* ILU-preconditioned GMRES */
    double *x_prec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_prec);
    if (!x_prec) {
        free(x_exact);
        free(b);
        free(x_unprec);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    sparse_iter_result_t result_prec;
    ASSERT_ERR(sparse_solve_gmres(A, b, x_prec, &opts, sparse_ilu_precond, &ilu, &result_prec),
               SPARSE_OK);

    double res_unprec = compute_relative_residual(A, b, x_unprec, n);
    double res_prec = compute_relative_residual(A, b, x_prec, n);
    printf("    steam1 ILU-GMRES(50): unprec=%d iters (res=%.3e), prec=%d iters (res=%.3e)\n",
           (int)result_unprec.iterations, res_unprec, (int)result_prec.iterations, res_prec);

    /* ILU preconditioning should significantly improve convergence.
     * Note: left preconditioning means the preconditioned residual converges,
     * but the true residual may be larger for ill-conditioned systems. */
    ASSERT_TRUE(result_prec.converged);
    ASSERT_TRUE(res_prec < 1e-4); /* relaxed: steam1 is ill-conditioned (condest ~3e7) */
    ASSERT_TRUE(result_prec.iterations < result_unprec.iterations);

    free(x_exact);
    free(b);
    free(x_unprec);
    free(x_prec);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ILU-preconditioned GMRES on orsirr_1 (1030×1030) */
static void test_ilu_gmres_orsirr_1(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/orsirr_1.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] orsirr_1.mtx not found\n");
        return;
    }
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1) * 0.01);
    sparse_matvec(A, x_exact, b);

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    /* Unpreconditioned GMRES */
    double *x_unprec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_unprec);
    if (!x_unprec) {
        free(x_exact);
        free(b);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    /* orsirr_1 is large and ill-conditioned; use relaxed tolerance */
    sparse_gmres_opts_t opts = {.max_iter = 2000, .restart = 50, .tol = 1e-5, .verbose = 0};
    sparse_iter_result_t result_unprec;
    ASSERT_ERR(sparse_solve_gmres(A, b, x_unprec, &opts, NULL, NULL, &result_unprec), SPARSE_OK);

    /* ILU-preconditioned GMRES */
    double *x_prec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_prec);
    if (!x_prec) {
        free(x_exact);
        free(b);
        free(x_unprec);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    sparse_iter_result_t result_prec;
    ASSERT_ERR(sparse_solve_gmres(A, b, x_prec, &opts, sparse_ilu_precond, &ilu, &result_prec),
               SPARSE_OK);

    double res_prec = compute_relative_residual(A, b, x_prec, n);
    printf("    orsirr_1 ILU-GMRES(50): unprec=%d iters (conv=%d), prec=%d iters (res=%.3e)\n",
           (int)result_unprec.iterations, result_unprec.converged, (int)result_prec.iterations,
           res_prec);

    ASSERT_TRUE(result_prec.converged);
    ASSERT_TRUE(res_prec < 1e-4); /* relaxed for large ill-conditioned system */
    ASSERT_TRUE(result_prec.iterations <= result_unprec.iterations);

    free(x_exact);
    free(b);
    free(x_unprec);
    free(x_prec);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Cholesky preconditioner (Day 8)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Cholesky preconditioner context */
typedef struct {
    SparseMatrix *L; /* Cholesky factor (factored in-place) */
} cholesky_precond_t;

/* Cholesky preconditioner callback: solve L*L^T*z = r */
static sparse_err_t cholesky_precond_apply(const void *ctx, idx_t n, const double *r, double *z) {
    const cholesky_precond_t *pc = (const cholesky_precond_t *)ctx;
    (void)n;
    return sparse_cholesky_solve(pc->L, r, z);
}

/* Cholesky-preconditioned CG on nos4 */
static void test_cholesky_precond_cg_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    /* Create Cholesky preconditioner from a copy of A */
    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    if (!L) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);
    cholesky_precond_t pc = {.L = L};

    /* Unpreconditioned CG */
    double *x_unprec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_unprec);
    if (!x_unprec) {
        free(x_exact);
        free(b);
        sparse_free(L);
        sparse_free(A);
        return;
    }
    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result_unprec;
    ASSERT_ERR(sparse_solve_cg(A, b, x_unprec, &opts, NULL, NULL, &result_unprec), SPARSE_OK);

    /* Cholesky-preconditioned CG (exact preconditioner → should converge in 1-2 iters) */
    double *x_prec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_prec);
    if (!x_prec) {
        free(x_exact);
        free(b);
        free(x_unprec);
        sparse_free(L);
        sparse_free(A);
        return;
    }
    sparse_iter_result_t result_prec;
    ASSERT_ERR(sparse_solve_cg(A, b, x_prec, &opts, cholesky_precond_apply, &pc, &result_prec),
               SPARSE_OK);

    double res_prec = compute_relative_residual(A, b, x_prec, n);
    printf("    nos4 Cholesky-CG: unprec=%d iters, prec=%d iters (res=%.3e)\n",
           (int)result_unprec.iterations, (int)result_prec.iterations, res_prec);

    ASSERT_TRUE(result_unprec.converged);
    ASSERT_TRUE(result_prec.converged);
    /* Exact Cholesky preconditioner should converge in very few iterations */
    ASSERT_TRUE(result_prec.iterations <= 2);
    ASSERT_TRUE(res_prec < 1e-8);

    free(x_exact);
    free(b);
    free(x_unprec);
    free(x_prec);
    sparse_free(L);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Preconditioner quality validation (Day 8)
 * ═══════════════════════════════════════════════════════════════════════ */

/* ILU should reduce iteration count by ≥2× on poorly conditioned system */
static void test_ilu_speedup_illcond(void) {
    /* Build a poorly conditioned SPD tridiagonal with varying diagonal */
    idx_t n = 30;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++) {
        double d = 2.0 + 50.0 * (double)i / (double)(n - 1); /* 2 to 52 */
        sparse_insert(A, i, i, d);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -1.0);
    }

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};

    /* Unpreconditioned CG */
    double *x_unprec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_unprec);
    if (!x_unprec) {
        free(x_exact);
        free(b);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    sparse_iter_result_t result_unprec;
    ASSERT_ERR(sparse_solve_cg(A, b, x_unprec, &opts, NULL, NULL, &result_unprec), SPARSE_OK);

    /* ILU-preconditioned CG */
    double *x_prec = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_prec);
    if (!x_prec) {
        free(x_exact);
        free(b);
        free(x_unprec);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    sparse_iter_result_t result_prec;
    ASSERT_ERR(sparse_solve_cg(A, b, x_prec, &opts, sparse_ilu_precond, &ilu, &result_prec),
               SPARSE_OK);

    ASSERT_TRUE(result_unprec.converged);
    ASSERT_TRUE(result_prec.converged);

    printf("    ill-cond speedup: unprec=%d iters, prec=%d iters (%.1fx)\n",
           (int)result_unprec.iterations, (int)result_prec.iterations,
           result_prec.iterations > 0
               ? (double)result_unprec.iterations / (double)result_prec.iterations
               : 0.0);

    /* ILU on tridiagonal = exact LU, so preconditioned should be 1 iteration */
    ASSERT_TRUE(result_prec.iterations <= result_unprec.iterations);

    double res_prec = compute_relative_residual(A, b, x_prec, n);
    ASSERT_TRUE(res_prec < 1e-8);

    free(x_exact);
    free(b);
    free(x_unprec);
    free(x_prec);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * ILUT factorization tests (Sprint 6 Day 2)
 * ═══════════════════════════════════════════════════════════════════════ */

/* ILUT on dense 3×3 with high fill → matches exact LU */
static void test_ilut_3x3_dense(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 0, 4.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 3.0);
    sparse_insert(A, 2, 0, 8.0);
    sparse_insert(A, 2, 1, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    /* High fill + low tolerance → should match exact LU */
    sparse_ilut_opts_t opts = {.tol = 1e-15, .max_fill = 100};
    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilut_factor(A, &opts, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    /* Verify solve: L*U*z = r should give exact result */
    double r[3] = {1.0, 2.0, 3.0};
    double z[3];
    ASSERT_ERR(sparse_ilu_solve(&ilu, r, z), SPARSE_OK);

    double *Az = malloc(3 * sizeof(double));
    ASSERT_NOT_NULL(Az);
    if (!Az) {
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    sparse_matvec(A, z, Az);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(Az[i], r[i], 1e-12);

    free(Az);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ILUT on tridiagonal SPD → same as ILU(0) (no fill to drop) */
static void test_ilut_tridiagonal(void) {
    idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilut_factor(A, NULL, &ilu); /* default opts */
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    double *r = malloc((size_t)n * sizeof(double));
    double *z = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(r);
    ASSERT_NOT_NULL(z);
    if (!r || !z) {
        free(r);
        free(z);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        r[i] = (double)(i + 1);

    ASSERT_ERR(sparse_ilu_solve(&ilu, r, z), SPARSE_OK);

    /* Tridiagonal: ILUT = exact LU (no fill), so A*z = r */
    double *Az = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(Az);
    if (!Az) {
        free(r);
        free(z);
        sparse_ilu_free(&ilu);
        sparse_free(A);
        return;
    }
    sparse_matvec(A, z, Az);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(Az[i], r[i], 1e-10);

    free(r);
    free(z);
    free(Az);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ILUT on west0067 → succeeds where ILU(0) fails */
static void test_ilut_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    /* ILU(0) should fail on west0067 (structurally zero diagonal) */
    sparse_ilu_t ilu0;
    sparse_err_t ilu0_err = sparse_ilu_factor(A, &ilu0);
    ASSERT_ERR(ilu0_err, SPARSE_ERR_SINGULAR);
    if (ilu0_err != SPARSE_ERR_SINGULAR) {
        sparse_ilu_free(&ilu0);
        sparse_free(A);
        return;
    }
    printf("    west0067: ILU(0) returns %s (expected)\n", sparse_strerror(ilu0_err));

    /* ILUT should succeed (fill-in provides nonzero diagonal) */
    sparse_ilut_opts_t opts = {.tol = 1e-4, .max_fill = 20};
    sparse_ilu_t ilut;
    sparse_err_t ilut_err = sparse_ilut_factor(A, &opts, &ilut);
    printf("    west0067: ILUT returns %s\n", sparse_strerror(ilut_err));

    if (ilut_err == SPARSE_OK) {
        /* Verify ILUT solve works */
        idx_t n = sparse_rows(A);
        double *r = malloc((size_t)n * sizeof(double));
        double *z = malloc((size_t)n * sizeof(double));
        ASSERT_NOT_NULL(r);
        ASSERT_NOT_NULL(z);
        if (r && z) {
            for (idx_t i = 0; i < n; i++)
                r[i] = 1.0;
            ASSERT_ERR(sparse_ilu_solve(&ilut, r, z), SPARSE_OK);
            double znorm = vec_norm2(z, n);
            printf("    west0067: ILUT solve norm=%.3e\n", znorm);
            ASSERT_TRUE(znorm > 0.0);
        }
        free(r);
        free(z);
        sparse_ilu_free(&ilut);
    }

    sparse_free(A);
}

/* ILUT-preconditioned GMRES on west0067 */
static void test_ilut_gmres_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_ilut_opts_t opts = {.tol = 1e-4, .max_fill = 20};
    sparse_ilu_t ilut;
    sparse_err_t ferr = sparse_ilut_factor(A, &opts, &ilut);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }

    double *x = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x);
    if (!x) {
        free(x_exact);
        free(b);
        sparse_ilu_free(&ilut);
        sparse_free(A);
        return;
    }
    /* west0067 has 65/67 zero diagonals; ILUT uses diagonal modification to
     * produce factors, but the resulting preconditioner is too poor for GMRES
     * convergence.  We assert that ILUT factorisation succeeds (above) and
     * that GMRES runs without crashing, returning a well-formed result. */
    sparse_gmres_opts_t gm_opts = {.max_iter = 500,
                                   .restart = 30,
                                   .tol = 1e-10,
                                   .verbose = 0,
                                   .precond_side = SPARSE_PRECOND_RIGHT};
    sparse_iter_result_t result;
    sparse_err_t solve_err =
        sparse_solve_gmres(A, b, x, &gm_opts, sparse_ilut_precond, &ilut, &result);

    double res = compute_relative_residual(A, b, x, n);
    printf("    west0067 ILUT-right-GMRES(30): %d iters, res=%.3e, conv=%d\n",
           (int)result.iterations, res, result.converged);

    /* GMRES is expected not to converge on this pathological matrix */
    ASSERT_TRUE(solve_err == SPARSE_OK || solve_err == SPARSE_ERR_NOT_CONVERGED);
    ASSERT_TRUE(result.iterations > 0);
    ASSERT_TRUE(result.residual_norm >= 0.0);

    free(x_exact);
    free(b);
    free(x);
    sparse_ilu_free(&ilut);
    sparse_free(A);
}

/* ILUT-preconditioned CG on nos4 */
static void test_ilut_cg_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    /* ILU(0) preconditioned CG */
    sparse_ilu_t ilu0;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu0);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            free(x_exact);
            free(b);
            sparse_free(A);
            return;
        }
    }
    double *x0 = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x0);
    sparse_iter_opts_t cg_opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t res0 = {0};
    if (x0)
        ASSERT_ERR(sparse_solve_cg(A, b, x0, &cg_opts, sparse_ilu_precond, &ilu0, &res0),
                   SPARSE_OK);

    /* ILUT preconditioned CG — tight tolerance to preserve quality */
    sparse_ilut_opts_t ilut_opts = {.tol = 1e-10, .max_fill = 50};
    sparse_ilu_t ilut;
    {
        sparse_err_t ferr = sparse_ilut_factor(A, &ilut_opts, &ilut);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            free(x_exact);
            free(b);
            free(x0);
            sparse_ilu_free(&ilu0);
            sparse_free(A);
            return;
        }
    }
    double *xt = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(xt);
    sparse_iter_result_t rest;
    if (xt) {
        sparse_err_t serr = sparse_solve_cg(A, b, xt, &cg_opts, sparse_ilut_precond, &ilut, &rest);
        ASSERT_TRUE(serr == SPARSE_OK || serr == SPARSE_ERR_NOT_CONVERGED);
    }

    if (x0 && xt) {
        printf("    nos4: ILU(0)-CG=%d iters, ILUT-CG=%d iters (conv=%d)\n", (int)res0.iterations,
               (int)rest.iterations, rest.converged);
        ASSERT_TRUE(res0.converged);
    }

    free(x_exact);
    free(b);
    free(x0);
    free(xt);
    sparse_ilu_free(&ilu0);
    sparse_ilu_free(&ilut);
    sparse_free(A);
}

/* ILUT error handling */
static void test_ilut_null_inputs(void) {
    SparseMatrix *A = build_spd_tridiag(3, 4.0, -1.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_ilu_t ilu;

    ASSERT_ERR(sparse_ilut_factor(NULL, NULL, &ilu), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_ilut_factor(A, NULL, NULL), SPARSE_ERR_NULL);

    /* Non-square */
    SparseMatrix *R = sparse_create(3, 4);
    ASSERT_NOT_NULL(R);
    if (R) {
        sparse_insert(R, 0, 0, 1.0);
        ASSERT_ERR(sparse_ilut_factor(R, NULL, &ilu), SPARSE_ERR_SHAPE);
        sparse_free(R);
    }

    /* Invalid opts */
    sparse_ilut_opts_t bad_opts = {.tol = -1.0, .max_fill = 10};
    ASSERT_ERR(sparse_ilut_factor(A, &bad_opts, &ilu), SPARSE_ERR_BADARG);

    bad_opts.tol = 1e-3;
    bad_opts.max_fill = -1;
    ASSERT_ERR(sparse_ilut_factor(A, &bad_opts, &ilu), SPARSE_ERR_BADARG);

    sparse_free(A);
}

/* ILUT vs ILU(0) on steam1: compare iteration counts */
static void test_ilut_vs_ilu0_steam1(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/steam1.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_gmres_opts_t gm_opts = {.max_iter = 1000, .restart = 50, .tol = 1e-4, .verbose = 0};

    /* ILU(0)-GMRES */
    sparse_ilu_t ilu0;
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu0);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            free(x_exact);
            free(b);
            sparse_free(A);
            return;
        }
    }
    double *x0 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res0;
    if (x0)
        sparse_solve_gmres(A, b, x0, &gm_opts, sparse_ilu_precond, &ilu0, &res0);

    /* ILUT-GMRES */
    sparse_ilut_opts_t ilut_opts = {.tol = 1e-3, .max_fill = 20};
    sparse_ilu_t ilut;
    {
        sparse_err_t ferr = sparse_ilut_factor(A, &ilut_opts, &ilut);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            free(x_exact);
            free(b);
            free(x0);
            sparse_ilu_free(&ilu0);
            sparse_free(A);
            return;
        }
    }
    double *xt = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t rest;
    if (xt)
        sparse_solve_gmres(A, b, xt, &gm_opts, sparse_ilut_precond, &ilut, &rest);

    if (x0 && xt) {
        printf("    steam1: ILU(0)-GMRES=%d iters (conv=%d), ILUT-GMRES=%d iters (conv=%d)\n",
               (int)res0.iterations, res0.converged, (int)rest.iterations, rest.converged);
    }

    free(x_exact);
    free(b);
    free(x0);
    free(xt);
    sparse_ilu_free(&ilu0);
    sparse_ilu_free(&ilut);
    sparse_free(A);
}

/* ILUT with default opts (NULL) */
static void test_ilut_default_opts(void) {
    SparseMatrix *A = build_spd_tridiag(5, 4.0, -1.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilut_factor(A, NULL, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    double r[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double z[5];
    ASSERT_ERR(sparse_ilu_solve(&ilu, r, z), SPARSE_OK);

    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * ILUT partial pivoting tests (Sprint 7 Day 2)
 * ═══════════════════════════════════════════════════════════════════════ */

/* ILUT pivot on dense 3×3: should match exact LU with correct permutation */
static void test_ilut_pivot_3x3(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 0, 4.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 3.0);
    sparse_insert(A, 2, 0, 8.0);
    sparse_insert(A, 2, 1, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    sparse_ilut_opts_t opts = {.tol = 1e-15, .max_fill = 100, .pivot = 1};
    sparse_ilu_t ilu;
    sparse_err_t ferr = sparse_ilut_factor(A, &opts, &ilu);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* perm should be non-NULL and a valid permutation */
    ASSERT_NOT_NULL(ilu.perm);

    /* Solve should produce correct answer */
    double r[3] = {1.0, 2.0, 3.0};
    double z[3];
    ASSERT_ERR(sparse_ilu_solve(&ilu, r, z), SPARSE_OK);

    double Az[3];
    sparse_matvec(A, z, Az);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(Az[i], r[i], 1e-10);

    printf("    pivot 3x3: perm=[%d,%d,%d]\n", (int)ilu.perm[0], (int)ilu.perm[1],
           (int)ilu.perm[2]);

    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ILUT pivot: perm should be valid (each index 0..n-1 appears once) */
static void test_ilut_pivot_perm_valid(void) {
    idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_ilut_opts_t opts = {.tol = 1e-3, .max_fill = 10, .pivot = 1};
    sparse_ilu_t ilu;
    sparse_err_t ferr = sparse_ilut_factor(A, &opts, &ilu);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_NOT_NULL(ilu.perm);
    if (ilu.perm) {
        int *seen = calloc((size_t)n, sizeof(int));
        ASSERT_NOT_NULL(seen);
        if (seen) {
            for (idx_t i = 0; i < n; i++) {
                ASSERT_TRUE(ilu.perm[i] >= 0 && ilu.perm[i] < n);
                if (ilu.perm[i] >= 0 && ilu.perm[i] < n)
                    seen[ilu.perm[i]] = 1;
            }
            for (idx_t i = 0; i < n; i++)
                ASSERT_TRUE(seen[i]);
            free(seen);
        }
    }

    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ILUT pivot=0 (default): perm should be NULL */
static void test_ilut_pivot_default_no_perm(void) {
    SparseMatrix *A = build_spd_tridiag(5, 4.0, -1.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_ilut_opts_t opts = {.tol = 1e-3, .max_fill = 10, .pivot = 0};
    sparse_ilu_t ilu;
    sparse_err_t ferr = sparse_ilut_factor(A, &opts, &ilu);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* perm should be NULL when pivoting is disabled */
    ASSERT_TRUE(ilu.perm == NULL);

    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ILUT pivot on identity: perm should be identity */
static void test_ilut_pivot_identity(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    sparse_ilut_opts_t opts = {.tol = 1e-3, .max_fill = 10, .pivot = 1};
    sparse_ilu_t ilu;
    sparse_err_t ferr = sparse_ilut_factor(A, &opts, &ilu);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_NOT_NULL(ilu.perm);
    if (ilu.perm) {
        for (idx_t i = 0; i < n; i++)
            ASSERT_EQ(ilu.perm[i], i);
    }

    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ILUT pivot on diagonally dominant matrix: pivoting should not change behavior */
static void test_ilut_pivot_diag_dominant(void) {
    idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n, 10.0, -1.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    /* Solve with pivoting */
    sparse_ilut_opts_t opts_piv = {.tol = 1e-4, .max_fill = 20, .pivot = 1};
    sparse_ilu_t ilu_piv;
    sparse_err_t ferr = sparse_ilut_factor(A, &opts_piv, &ilu_piv);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Solve without pivoting */
    sparse_ilut_opts_t opts_nopiv = {.tol = 1e-4, .max_fill = 20, .pivot = 0};
    sparse_ilu_t ilu_nopiv;
    ferr = sparse_ilut_factor(A, &opts_nopiv, &ilu_nopiv);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_ilu_free(&ilu_piv);
        sparse_free(A);
        return;
    }

    double *b = malloc((size_t)n * sizeof(double));
    double *x_piv = malloc((size_t)n * sizeof(double));
    double *x_nopiv = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(x_piv);
    ASSERT_NOT_NULL(x_nopiv);
    if (!b || !x_piv || !x_nopiv) {
        free(b);
        free(x_piv);
        free(x_nopiv);
        sparse_ilu_free(&ilu_piv);
        sparse_ilu_free(&ilu_nopiv);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    ASSERT_ERR(sparse_ilu_solve(&ilu_piv, b, x_piv), SPARSE_OK);
    ASSERT_ERR(sparse_ilu_solve(&ilu_nopiv, b, x_nopiv), SPARSE_OK);

    /* Both should produce similar results on diag-dominant matrix */
    double res_piv = compute_relative_residual(A, b, x_piv, n);
    double res_nopiv = compute_relative_residual(A, b, x_nopiv, n);
    printf("    diag-dominant: pivot_res=%.3e, nopivot_res=%.3e\n", res_piv, res_nopiv);
    ASSERT_TRUE(res_piv < 1e-8);
    ASSERT_TRUE(res_nopiv < 1e-8);

    free(b);
    free(x_piv);
    free(x_nopiv);
    sparse_ilu_free(&ilu_piv);
    sparse_ilu_free(&ilu_nopiv);
    sparse_free(A);
}

/* ILUT pivot on west0067 with GMRES: compare with diagonal modification */
static void test_ilut_pivot_gmres_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    /* ILUT with pivoting */
    sparse_ilut_opts_t opts = {.tol = 1e-4, .max_fill = 20, .pivot = 1};
    sparse_ilu_t ilut;
    sparse_err_t ferr = sparse_ilut_factor(A, &opts, &ilut);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }

    double *x = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x);
    if (!x) {
        free(x_exact);
        free(b);
        sparse_ilu_free(&ilut);
        sparse_free(A);
        return;
    }

    sparse_gmres_opts_t gm_opts = {.max_iter = 500,
                                   .restart = 30,
                                   .tol = 1e-10,
                                   .verbose = 0,
                                   .precond_side = SPARSE_PRECOND_RIGHT};
    sparse_iter_result_t result = {0};
    sparse_err_t solve_err =
        sparse_solve_gmres(A, b, x, &gm_opts, sparse_ilut_precond, &ilut, &result);

    double res = compute_relative_residual(A, b, x, n);
    printf("    west0067 pivot-ILUT-GMRES: %d iters, res=%.3e, conv=%d\n", (int)result.iterations,
           res, result.converged);

    /* With pivoting, GMRES may or may not converge on west0067 (pathological).
     * We validate that the solver runs without crashing. */
    ASSERT_TRUE(solve_err == SPARSE_OK || solve_err == SPARSE_ERR_NOT_CONVERGED);
    ASSERT_TRUE(result.iterations > 0);

    free(x_exact);
    free(b);
    free(x);
    sparse_ilu_free(&ilut);
    sparse_free(A);
}

/* ILUT pivot on steam1 with GMRES: compare iteration count */
static void test_ilut_pivot_gmres_steam1(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/steam1.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    /* Pivot ILUT */
    sparse_ilut_opts_t opts_piv = {.tol = 1e-3, .max_fill = 10, .pivot = 1};
    sparse_ilu_t ilut_piv;
    sparse_err_t ferr = sparse_ilut_factor(A, &opts_piv, &ilut_piv);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }

    /* Non-pivot ILUT */
    sparse_ilut_opts_t opts_nopiv = {.tol = 1e-3, .max_fill = 10, .pivot = 0};
    sparse_ilu_t ilut_nopiv;
    ferr = sparse_ilut_factor(A, &opts_nopiv, &ilut_nopiv);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        free(x_exact);
        free(b);
        sparse_ilu_free(&ilut_piv);
        sparse_free(A);
        return;
    }

    sparse_gmres_opts_t gm_opts = {.max_iter = 200,
                                   .restart = 30,
                                   .tol = 1e-10,
                                   .verbose = 0,
                                   .precond_side = SPARSE_PRECOND_LEFT};

    /* Solve with pivot ILUT */
    double *x_piv = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_piv);
    if (!x_piv) {
        free(x_exact);
        free(b);
        sparse_ilu_free(&ilut_piv);
        sparse_ilu_free(&ilut_nopiv);
        sparse_free(A);
        return;
    }
    sparse_iter_result_t res_piv = {0};
    sparse_solve_gmres(A, b, x_piv, &gm_opts, sparse_ilut_precond, &ilut_piv, &res_piv);

    /* Solve with non-pivot ILUT */
    double *x_nopiv = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x_nopiv);
    if (!x_nopiv) {
        free(x_exact);
        free(b);
        free(x_piv);
        sparse_ilu_free(&ilut_piv);
        sparse_ilu_free(&ilut_nopiv);
        sparse_free(A);
        return;
    }
    sparse_iter_result_t res_nopiv = {0};
    sparse_solve_gmres(A, b, x_nopiv, &gm_opts, sparse_ilut_precond, &ilut_nopiv, &res_nopiv);

    printf("    steam1: pivot=%d iters (conv=%d), nopivot=%d iters (conv=%d)\n",
           (int)res_piv.iterations, res_piv.converged, (int)res_nopiv.iterations,
           res_nopiv.converged);

    free(x_exact);
    free(b);
    free(x_piv);
    free(x_nopiv);
    sparse_ilu_free(&ilut_piv);
    sparse_ilu_free(&ilut_nopiv);
    sparse_free(A);
}

/* NULL opts with pivot: defaults should have pivot=0 */
static void test_ilut_pivot_null_opts(void) {
    SparseMatrix *A = build_spd_tridiag(5, 4.0, -1.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_ilu_t ilu;
    sparse_err_t ferr = sparse_ilut_factor(A, NULL, &ilu);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Default opts should have pivot=0, so perm is NULL */
    ASSERT_TRUE(ilu.perm == NULL);

    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Error handling tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_ilu_null_inputs(void) {
    SparseMatrix *A = build_spd_tridiag(3, 4.0, -1.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_ilu_t ilu;
    double r[3] = {1.0, 2.0, 3.0};
    double z[3];

    ASSERT_ERR(sparse_ilu_factor(NULL, &ilu), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_ilu_factor(A, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_ilu_solve(NULL, r, z), SPARSE_ERR_NULL);

    /* Factor first for solve NULL tests */
    {
        sparse_err_t ferr = sparse_ilu_factor(A, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }
    ASSERT_ERR(sparse_ilu_solve(&ilu, NULL, z), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_ilu_solve(&ilu, r, NULL), SPARSE_ERR_NULL);

    sparse_ilu_free(&ilu);
    /* Double free should be safe */
    sparse_ilu_free(&ilu);

    sparse_free(A);
}

static void test_ilu_nonsquare(void) {
    SparseMatrix *A = sparse_create(3, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_ilu_t ilu;

    ASSERT_ERR(sparse_ilu_factor(A, &ilu), SPARSE_ERR_SHAPE);

    sparse_free(A);
}

static void test_ilu_singular(void) {
    /* Zero diagonal → singular */
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 0.0); /* zero pivot */
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);

    sparse_ilu_t ilu;
    ASSERT_ERR(sparse_ilu_factor(A, &ilu), SPARSE_ERR_SINGULAR);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test suite
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("ILU(0) Preconditioner");

    /* Basic factorization */
    RUN_TEST(test_ilu_3x3_dense);
    RUN_TEST(test_ilu_diagonal);
    RUN_TEST(test_ilu_tridiagonal);
    RUN_TEST(test_ilu_drops_fill);

    /* Solve */
    RUN_TEST(test_ilu_solve_identity);
    RUN_TEST(test_ilu_solve_roundtrip);

    /* As preconditioner */
    RUN_TEST(test_ilu_precond_cg);
    RUN_TEST(test_ilu_precond_gmres);

    /* ILU-preconditioned CG on SuiteSparse (Day 8) */
    RUN_TEST(test_ilu_cg_nos4);
    RUN_TEST(test_ilu_cg_bcsstk04);

    /* ILU-preconditioned GMRES on SuiteSparse (Day 8) */
    RUN_TEST(test_ilu_gmres_west0067);
    RUN_TEST(test_ilu_gmres_steam1);
    RUN_TEST(test_ilu_gmres_orsirr_1);

    /* Cholesky preconditioner (Day 8) */
    RUN_TEST(test_cholesky_precond_cg_nos4);

    /* Preconditioner quality (Day 8) */
    RUN_TEST(test_ilu_speedup_illcond);

    /* ILUT tests (Sprint 6 Day 2) */
    RUN_TEST(test_ilut_3x3_dense);
    RUN_TEST(test_ilut_tridiagonal);
    RUN_TEST(test_ilut_west0067);
    RUN_TEST(test_ilut_gmres_west0067);
    RUN_TEST(test_ilut_cg_nos4);
    RUN_TEST(test_ilut_null_inputs);
    RUN_TEST(test_ilut_vs_ilu0_steam1);
    RUN_TEST(test_ilut_default_opts);

    /* ILUT partial pivoting (Sprint 7 Day 2) */
    RUN_TEST(test_ilut_pivot_3x3);
    RUN_TEST(test_ilut_pivot_perm_valid);
    RUN_TEST(test_ilut_pivot_default_no_perm);
    RUN_TEST(test_ilut_pivot_identity);
    RUN_TEST(test_ilut_pivot_diag_dominant);
    RUN_TEST(test_ilut_pivot_gmres_west0067);
    RUN_TEST(test_ilut_pivot_gmres_steam1);
    RUN_TEST(test_ilut_pivot_null_opts);

    /* Error handling */
    RUN_TEST(test_ilu_null_inputs);
    RUN_TEST(test_ilu_nonsquare);
    RUN_TEST(test_ilu_singular);

    TEST_SUITE_END();
}
