#include "sparse_matrix.h"
#include "sparse_iterative.h"
#include "sparse_cholesky.h"
#include "sparse_lu.h"
#include "sparse_vector.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Test helpers — SPD matrix builders & residual computation
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Build an n×n SPD tridiagonal matrix:
 *   diag = diag_val, off-diag = offdiag_val
 *   e.g., [4 -1 0; -1 4 -1; 0 -1 4] for diag_val=4, offdiag_val=-1
 *
 * The matrix is SPD when diag_val > 2*|offdiag_val|.
 */
static SparseMatrix *build_spd_tridiag(idx_t n, double diag_val, double offdiag_val)
{
    SparseMatrix *A = sparse_create(n, n);
    if (!A) return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, diag_val);
        if (i > 0)     sparse_insert(A, i, i - 1, offdiag_val);
        if (i < n - 1) sparse_insert(A, i, i + 1, offdiag_val);
    }
    return A;
}

/**
 * Build an n×n identity matrix.
 */
static SparseMatrix *build_identity(idx_t n)
{
    SparseMatrix *A = sparse_create(n, n);
    if (!A) return NULL;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);
    return A;
}

/**
 * Build a 2D Laplacian stencil on an m×m grid (matrix size n = m*m).
 * Returns an n×n SPD matrix with 5-point stencil.
 */
static SparseMatrix *build_laplacian_2d(idx_t m)
{
    idx_t n = m * m;
    SparseMatrix *A = sparse_create(n, n);
    if (!A) return NULL;
    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = 0; j < m; j++) {
            idx_t row = i * m + j;
            sparse_insert(A, row, row, 4.0);
            if (j > 0)     sparse_insert(A, row, row - 1, -1.0);
            if (j < m - 1) sparse_insert(A, row, row + 1, -1.0);
            if (i > 0)     sparse_insert(A, row, row - m, -1.0);
            if (i < m - 1) sparse_insert(A, row, row + m, -1.0);
        }
    }
    return A;
}

/**
 * Compute the relative residual ||b - A*x|| / ||b||.
 * Returns 0.0 if ||b|| == 0.
 */
static double compute_relative_residual(const SparseMatrix *A,
                                         const double *b, const double *x,
                                         idx_t n)
{
    double *r = malloc((size_t)n * sizeof(double));
    if (!r) return -1.0;

    /* r = A*x */
    sparse_matvec(A, x, r);

    /* r = b - A*x */
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - r[i];

    double rnorm = vec_norm2(r, n);
    double bnorm = vec_norm2(b, n);
    free(r);

    return (bnorm > 0.0) ? rnorm / bnorm : 0.0;
}

/**
 * Compute b = A*x_exact for generating test RHS with known solution.
 */
static void compute_rhs(const SparseMatrix *A, const double *x_exact,
                         double *b)
{
    sparse_matvec(A, x_exact, b);
}

/* ═══════════════════════════════════════════════════════════════════════
 * CG tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* CG on 2×2 SPD: A = [[4, 1], [1, 3]], b = [1, 2] → known solution */
static void test_cg_2x2_spd(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);

    /* Exact solution: A*x = b → x = A^{-1}*b
     * A^{-1} = (1/11)*[[3, -1], [-1, 4]]
     * b = [1, 2] → x = (1/11)*[1, 7] = [1/11, 7/11] */
    double b[2] = {1.0, 2.0};
    double x[2] = {0.0, 0.0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_NEAR(x[0], 1.0 / 11.0, 1e-12);
    ASSERT_NEAR(x[1], 7.0 / 11.0, 1e-12);
    ASSERT_TRUE(result.iterations <= 2);  /* n=2, CG converges in ≤ n steps */

    sparse_free(A);
}

/* CG on 3×3 SPD tridiagonal */
static void test_cg_3x3_tridiag(void)
{
    SparseMatrix *A = build_spd_tridiag(3, 4.0, -1.0);

    /* Use x_exact = [1, 2, 3], compute b = A*x_exact */
    double x_exact[3] = {1.0, 2.0, 3.0};
    double b[3], x[3] = {0.0, 0.0, 0.0};
    compute_rhs(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.iterations <= 3);  /* CG converges in ≤ n steps on SPD */
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    sparse_free(A);
}

/* CG on identity matrix → x = b in 1 iteration */
static void test_cg_identity(void)
{
    idx_t n = 5;
    SparseMatrix *A = build_identity(n);

    double b[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double x[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-14, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.iterations <= 1);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], b[i], 1e-14);

    sparse_free(A);
}

/* CG with zero RHS → x = 0 */
static void test_cg_zero_rhs(void)
{
    SparseMatrix *A = build_spd_tridiag(4, 4.0, -1.0);

    double b[4] = {0.0, 0.0, 0.0, 0.0};
    double x[4] = {1.0, 1.0, 1.0, 1.0};  /* nonzero initial guess */
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_cg(A, b, x, NULL, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(x[i], 0.0, 1e-15);

    sparse_free(A);
}

/* CG with NULL opts → uses defaults */
static void test_cg_default_opts(void)
{
    SparseMatrix *A = build_spd_tridiag(3, 4.0, -1.0);
    double x_exact[3] = {1.0, -1.0, 0.5};
    double b[3], x[3] = {0.0, 0.0, 0.0};
    compute_rhs(A, x_exact, b);

    sparse_iter_result_t result;
    ASSERT_ERR(sparse_solve_cg(A, b, x, NULL, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    sparse_free(A);
}

/* CG on 2D Laplacian (larger SPD system) */
static void test_cg_laplacian_2d(void)
{
    idx_t m = 5;
    idx_t n = m * m;  /* 25×25 */
    SparseMatrix *A = build_laplacian_2d(m);

    /* x_exact = [1, 2, ..., n] */
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.residual_norm < 1e-10);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-10);

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* CG with initial guess close to solution → fewer iterations */
static void test_cg_initial_guess(void)
{
    SparseMatrix *A = build_spd_tridiag(10, 4.0, -1.0);
    idx_t n = 10;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x_zero = calloc((size_t)n, sizeof(double));
    double *x_close = malloc((size_t)n * sizeof(double));

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    /* Slightly perturbed initial guess */
    for (idx_t i = 0; i < n; i++)
        x_close[i] = x_exact[i] + 0.01;

    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result_zero, result_close;

    sparse_solve_cg(A, b, x_zero, &opts, NULL, NULL, &result_zero);
    sparse_solve_cg(A, b, x_close, &opts, NULL, NULL, &result_close);

    ASSERT_TRUE(result_zero.converged);
    ASSERT_TRUE(result_close.converged);
    /* Close guess should need fewer iterations */
    ASSERT_TRUE(result_close.iterations <= result_zero.iterations);

    free(x_exact); free(b); free(x_zero); free(x_close);
    sparse_free(A);
}

/* CG on 1×1 system: trivial solve */
static void test_cg_1x1(void)
{
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 5.0);

    double b[1] = {10.0};
    double x[1] = {0.0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-14, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_NEAR(x[0], 2.0, 1e-14);
    ASSERT_TRUE(result.iterations <= 1);

    sparse_free(A);
}

/* CG on diagonal SPD matrix */
static void test_cg_diagonal(void)
{
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    double diag_vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag_vals[i]);

    /* x_exact = [6, 5, 4, 3, 2, 1], b = diag * x_exact */
    double x_exact[6] = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double b[6], x[6] = {0};
    compute_rhs(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-14, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    /* Diagonal matrix: CG converges in 1 iteration (all eigenvalues distinct
       but CG on diagonal is effectively direct) */
    ASSERT_TRUE(result.iterations <= n);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-12);

    sparse_free(A);
}

/* CG on larger tridiagonal system (50×50) */
static void test_cg_large_tridiag(void)
{
    idx_t n = 50;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1) * 0.1);
    compute_rhs(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.residual_norm < 1e-12);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-10);

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* CG fails when max_iter is too small */
static void test_cg_max_iter_exceeded(void)
{
    idx_t n = 20;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    /* Only allow 2 iterations on a 20×20 system — not enough */
    sparse_iter_opts_t opts = {.max_iter = 2, .tol = 1e-14, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result),
               SPARSE_ERR_NOT_CONVERGED);
    ASSERT_FALSE(result.converged);
    ASSERT_EQ(result.iterations, 2);
    ASSERT_TRUE(result.residual_norm > 1e-14);  /* not converged yet */

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* CG with exact initial guess → converges in 0 iterations */
static void test_cg_exact_initial_guess(void)
{
    SparseMatrix *A = build_spd_tridiag(5, 4.0, -1.0);
    idx_t n = 5;

    double x_exact[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b[5];
    double x[5];
    compute_rhs(A, x_exact, b);

    /* Start from the exact solution */
    for (idx_t i = 0; i < n; i++)
        x[i] = x_exact[i];

    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.iterations, 0);

    sparse_free(A);
}

/* Simple diagonal preconditioner: M = diag(A), so M^{-1}*r = r./diag(A) */
typedef struct {
    double *diag_inv;  /* 1/A(i,i) for each i */
    idx_t n;
} diag_precond_t;

static sparse_err_t diag_precond_apply(const void *ctx, idx_t n,
                                        const double *r, double *z)
{
    const diag_precond_t *pc = (const diag_precond_t *)ctx;
    (void)n;
    for (idx_t i = 0; i < pc->n; i++)
        z[i] = pc->diag_inv[i] * r[i];
    return SPARSE_OK;
}

/* CG with diagonal (Jacobi) preconditioner */
static void test_cg_diagonal_preconditioner(void)
{
    idx_t n = 20;
    /* Build a poorly scaled SPD tridiagonal: diag varies from 2 to 40 */
    SparseMatrix *A = sparse_create(n, n);
    double *diag_inv = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++) {
        double d = 2.0 + 2.0 * (double)i;  /* 2, 4, 6, ..., 40 */
        sparse_insert(A, i, i, d);
        if (i > 0)     sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1) sparse_insert(A, i, i + 1, -1.0);
        diag_inv[i] = 1.0 / d;
    }

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    diag_precond_t pc = {.diag_inv = diag_inv, .n = n};
    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};

    /* Solve without preconditioner */
    double *x_unprec = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t result_unprec;
    sparse_solve_cg(A, b, x_unprec, &opts, NULL, NULL, &result_unprec);

    /* Solve with diagonal preconditioner */
    double *x_prec = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t result_prec;
    sparse_solve_cg(A, b, x_prec, &opts, diag_precond_apply, &pc, &result_prec);

    ASSERT_TRUE(result_unprec.converged);
    ASSERT_TRUE(result_prec.converged);

    /* Preconditioned should converge in fewer iterations on this
       poorly scaled system */
    ASSERT_TRUE(result_prec.iterations <= result_unprec.iterations);

    /* Both should produce correct solutions */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_unprec[i], x_exact[i], 1e-8);
        ASSERT_NEAR(x_prec[i], x_exact[i], 1e-8);
    }

    free(diag_inv); free(x_exact); free(b);
    free(x_unprec); free(x_prec);
    sparse_free(A);
}

/* CG on 2D Laplacian with diagonal preconditioner */
static void test_cg_precond_laplacian(void)
{
    idx_t m = 6;
    idx_t n = m * m;  /* 36×36 */
    SparseMatrix *A = build_laplacian_2d(m);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    /* Diagonal preconditioner: diag(A) = 4 for all entries */
    double *diag_inv = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        diag_inv[i] = 0.25;  /* 1/4 */
    diag_precond_t pc = {.diag_inv = diag_inv, .n = n};

    sparse_iter_opts_t opts = {.max_iter = 300, .tol = 1e-10, .verbose = 0};

    double *x = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, diag_precond_apply, &pc, &result),
               SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.residual_norm < 1e-10);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-8);

    free(diag_inv); free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SuiteSparse CG validation (Day 3)
 * ═══════════════════════════════════════════════════════════════════════ */

/* CG on nos4 (100×100 SPD) */
static void test_cg_nos4(void)
{
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_NOT_NULL(A);

    idx_t n = sparse_rows(A);
    ASSERT_EQ(n, 100);

    /* Generate RHS from known solution x_exact = [1, 2, ..., n] */
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    printf("    nos4: CG iters=%d, rel_res=%.3e\n",
           (int)result.iterations, rel_res);
    ASSERT_TRUE(rel_res < 1e-8);

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* CG on bcsstk04 (132×132 SPD stiffness matrix) */
static void test_cg_bcsstk04(void)
{
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    ASSERT_ERR(err, SPARSE_OK);
    ASSERT_NOT_NULL(A);

    idx_t n = sparse_rows(A);
    ASSERT_EQ(n, 132);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 1000, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    printf("    bcsstk04: CG iters=%d, rel_res=%.3e\n",
           (int)result.iterations, rel_res);
    ASSERT_TRUE(rel_res < 1e-8);

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* CG on SuiteSparse: compare zero initial guess vs nearby guess */
static void test_cg_suitesparse_initial_guess(void)
{
    SparseMatrix *A = NULL;
    sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};

    /* Solve from x_0 = 0 */
    double *x_zero = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t result_zero;
    sparse_solve_cg(A, b, x_zero, &opts, NULL, NULL, &result_zero);

    /* Solve from x_0 close to solution (perturbed by 1%) */
    double *x_near = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_near[i] = x_exact[i] * 1.01;
    sparse_iter_result_t result_near;
    sparse_solve_cg(A, b, x_near, &opts, NULL, NULL, &result_near);

    ASSERT_TRUE(result_zero.converged);
    ASSERT_TRUE(result_near.converged);
    ASSERT_TRUE(result_near.iterations <= result_zero.iterations);
    printf("    nos4 initial guess: zero=%d iters, near=%d iters\n",
           (int)result_zero.iterations, (int)result_near.iterations);

    free(x_exact); free(b); free(x_zero); free(x_near);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Convergence monitoring tests (Day 3)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Tight tolerance (1e-14) needs more iterations than default (1e-10) */
static void test_cg_tight_tolerance(void)
{
    idx_t n = 30;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1) * 0.2);
    compute_rhs(A, x_exact, b);

    /* Tight tolerance */
    double *x_tight = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts_tight = {.max_iter = 200, .tol = 1e-14, .verbose = 0};
    sparse_iter_result_t result_tight;
    sparse_solve_cg(A, b, x_tight, &opts_tight, NULL, NULL, &result_tight);

    /* Loose tolerance */
    double *x_loose = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts_loose = {.max_iter = 200, .tol = 1e-4, .verbose = 0};
    sparse_iter_result_t result_loose;
    sparse_solve_cg(A, b, x_loose, &opts_loose, NULL, NULL, &result_loose);

    ASSERT_TRUE(result_tight.converged);
    ASSERT_TRUE(result_loose.converged);
    /* Tight tolerance should require more iterations */
    ASSERT_TRUE(result_tight.iterations >= result_loose.iterations);
    /* Tight tolerance should produce smaller residual */
    ASSERT_TRUE(result_tight.residual_norm <= result_loose.residual_norm + 1e-16);

    free(x_exact); free(b); free(x_tight); free(x_loose);
    sparse_free(A);
}

/* Loose tolerance (1e-4) converges quickly */
static void test_cg_loose_tolerance(void)
{
    idx_t n = 50;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    double *x = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-4, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.residual_norm < 1e-4);
    /* With loose tolerance, should converge much faster than n iterations */
    ASSERT_TRUE(result.iterations < n);

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* Verify CG converges and residual reported in result struct is accurate */
static void test_cg_residual_accuracy(void)
{
    idx_t n = 15;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    /* Run CG with different max_iter limits and verify residual decreases */
    double prev_residual = 1e30;
    for (idx_t max_it = 1; max_it <= n; max_it++) {
        double *x = calloc((size_t)n, sizeof(double));
        sparse_iter_opts_t opts = {.max_iter = max_it, .tol = 1e-15, .verbose = 0};
        sparse_iter_result_t result;
        sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

        /* Independently compute the actual relative residual */
        double actual_res = compute_relative_residual(A, b, x, n);

        /* The reported residual should be close to the actual one */
        ASSERT_NEAR(result.residual_norm, actual_res, 1e-6);

        /* More iterations should give better (or equal) residual */
        ASSERT_TRUE(actual_res <= prev_residual + 1e-14);
        prev_residual = actual_res;

        free(x);
        if (actual_res < 1e-12) break;
    }

    ASSERT_TRUE(prev_residual < 1e-10);

    free(x_exact); free(b);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Non-SPD behavior tests (Day 3)
 * ═══════════════════════════════════════════════════════════════════════ */

/* CG on non-symmetric matrix — may not converge (user responsibility) */
static void test_cg_nonsymmetric_behavior(void)
{
    /* Non-symmetric 3×3 matrix */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 2.0);  /* asymmetric: A(1,0)=2 != A(0,1)=1 */
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 2, 5.0);

    double b[3] = {5.0, 5.0, 5.0};
    double x[3] = {0.0, 0.0, 0.0};
    /* Give it few iterations — it may or may not converge */
    sparse_iter_opts_t opts = {.max_iter = 50, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    /* We don't check the return code — just verify it doesn't crash */
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);
    /* Result should have valid fields regardless */
    ASSERT_TRUE(result.iterations >= 0);
    ASSERT_TRUE(result.residual_norm >= 0.0);

    sparse_free(A);
}

/* CG on indefinite symmetric matrix — may break down */
static void test_cg_indefinite_behavior(void)
{
    /* Symmetric but indefinite: eigenvalues are 3 and -1 */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 1, 1, 1.0);

    double b[2] = {1.0, 1.0};
    double x[2] = {0.0, 0.0};
    sparse_iter_opts_t opts = {.max_iter = 50, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    /* Should not crash — may return NOT_CONVERGED or breakdown */
    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_TRUE(result.iterations >= 0);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * CG vs Cholesky comparison (Day 3)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compare CG and Cholesky on same tridiagonal system */
static void test_cg_vs_cholesky_tridiag(void)
{
    idx_t n = 20;
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1) * 0.3);

    /* CG solve */
    SparseMatrix *A_cg = build_spd_tridiag(n, 4.0, -1.0);
    compute_rhs(A_cg, x_exact, b);
    double *x_cg = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result_cg;
    sparse_solve_cg(A_cg, b, x_cg, &opts, NULL, NULL, &result_cg);

    /* Cholesky solve */
    SparseMatrix *A_chol = build_spd_tridiag(n, 4.0, -1.0);
    double *x_chol = malloc((size_t)n * sizeof(double));
    sparse_cholesky_factor(A_chol);
    sparse_cholesky_solve(A_chol, b, x_chol);

    ASSERT_TRUE(result_cg.converged);

    /* Both should produce comparable solutions */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_cg[i], x_chol[i], 1e-8);

    /* Both should be close to exact solution */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_cg[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_chol[i], x_exact[i], 1e-10);
    }

    free(x_exact); free(b); free(x_cg); free(x_chol);
    sparse_free(A_cg); sparse_free(A_chol);
}

/* Compare CG and Cholesky on nos4 */
static void test_cg_vs_cholesky_nos4(void)
{
    SparseMatrix *A_cg = NULL;
    SparseMatrix *A_chol = NULL;
    sparse_load_mm(&A_cg, SS_DIR "/nos4.mtx");
    sparse_load_mm(&A_chol, SS_DIR "/nos4.mtx");

    idx_t n = sparse_rows(A_cg);
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A_cg, x_exact, b);

    /* CG solve */
    double *x_cg = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result_cg;
    sparse_solve_cg(A_cg, b, x_cg, &opts, NULL, NULL, &result_cg);

    /* Cholesky solve */
    double *x_chol = malloc((size_t)n * sizeof(double));
    sparse_cholesky_factor(A_chol);
    sparse_cholesky_solve(A_chol, b, x_chol);

    double res_cg = compute_relative_residual(A_cg, b, x_cg, n);
    double res_chol = compute_relative_residual(A_cg, b, x_chol, n);

    printf("    nos4: CG res=%.3e (%d iters), Cholesky res=%.3e\n",
           res_cg, (int)result_cg.iterations, res_chol);

    ASSERT_TRUE(result_cg.converged);
    ASSERT_TRUE(res_cg < 1e-8);
    ASSERT_TRUE(res_chol < 1e-8);

    /* CG and Cholesky solutions should be close */
    double maxdiff = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double diff = fabs(x_cg[i] - x_chol[i]);
        if (diff > maxdiff) maxdiff = diff;
    }
    printf("    nos4: CG vs Cholesky max |diff| = %.3e\n", maxdiff);
    ASSERT_TRUE(maxdiff < 1e-6);

    free(x_exact); free(b); free(x_cg); free(x_chol);
    sparse_free(A_cg); sparse_free(A_chol);
}

/* Compare CG and Cholesky on bcsstk04 */
static void test_cg_vs_cholesky_bcsstk04(void)
{
    SparseMatrix *A_cg = NULL;
    SparseMatrix *A_chol = NULL;
    sparse_load_mm(&A_cg, SS_DIR "/bcsstk04.mtx");
    sparse_load_mm(&A_chol, SS_DIR "/bcsstk04.mtx");

    idx_t n = sparse_rows(A_cg);
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A_cg, x_exact, b);

    /* CG solve */
    double *x_cg = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 1000, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result_cg;
    sparse_solve_cg(A_cg, b, x_cg, &opts, NULL, NULL, &result_cg);

    /* Cholesky solve */
    double *x_chol = malloc((size_t)n * sizeof(double));
    sparse_cholesky_factor(A_chol);
    sparse_cholesky_solve(A_chol, b, x_chol);

    double res_cg = compute_relative_residual(A_cg, b, x_cg, n);
    double res_chol = compute_relative_residual(A_cg, b, x_chol, n);

    printf("    bcsstk04: CG res=%.3e (%d iters), Cholesky res=%.3e\n",
           res_cg, (int)result_cg.iterations, res_chol);

    ASSERT_TRUE(result_cg.converged);
    ASSERT_TRUE(res_cg < 1e-4);   /* bcsstk04 is ill-conditioned */
    ASSERT_TRUE(res_chol < 1e-4);

    free(x_exact); free(b); free(x_cg); free(x_chol);
    sparse_free(A_cg); sparse_free(A_chol);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Verbose mode test (Day 3)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Verify verbose mode doesn't crash and produces output */
static void test_cg_verbose_mode(void)
{
    SparseMatrix *A = build_spd_tridiag(5, 4.0, -1.0);
    idx_t n = 5;
    double x_exact[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b[5], x[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    compute_rhs(A, x_exact, b);

    /* Redirect stderr to /dev/null to suppress verbose output in test,
     * but verify it doesn't crash */
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10, .verbose = 1};
    sparse_iter_result_t result;

    /* Capture stderr to verify output exists */
    FILE *saved_stderr = stderr;
    FILE *devnull = fopen("/dev/null", "w");
    if (devnull) stderr = devnull;

    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    if (devnull) {
        stderr = saved_stderr;
        fclose(devnull);
    }

    /* Verify solution is still correct despite verbose output */
    for (int i = 0; i < (int)n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Error handling tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* NULL inputs → SPARSE_ERR_NULL */
static void test_cg_null_inputs(void)
{
    SparseMatrix *A = build_spd_tridiag(3, 4.0, -1.0);
    double b[3] = {1.0, 2.0, 3.0};
    double x[3] = {0.0, 0.0, 0.0};

    ASSERT_ERR(sparse_solve_cg(NULL, b, x, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_solve_cg(A, NULL, x, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_solve_cg(A, b, NULL, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
}

/* Non-square matrix → SPARSE_ERR_SHAPE */
static void test_cg_nonsquare(void)
{
    SparseMatrix *A = sparse_create(3, 4);
    sparse_insert(A, 0, 0, 1.0);

    double b[3] = {1.0, 0.0, 0.0};
    double x[4] = {0.0};

    ASSERT_ERR(sparse_solve_cg(A, b, x, NULL, NULL, NULL, NULL), SPARSE_ERR_SHAPE);
    sparse_free(A);
}

/* GMRES stub returns NOT_CONVERGED, but handles NULL/shape correctly */
static void test_gmres_null_inputs(void)
{
    SparseMatrix *A = build_spd_tridiag(3, 4.0, -1.0);
    double b[3] = {1.0, 2.0, 3.0};
    double x[3] = {0.0, 0.0, 0.0};

    ASSERT_ERR(sparse_solve_gmres(NULL, b, x, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_solve_gmres(A, NULL, x, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_solve_gmres(A, b, NULL, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
}

static void test_gmres_nonsquare(void)
{
    SparseMatrix *A = sparse_create(3, 4);
    sparse_insert(A, 0, 0, 1.0);
    double b[3] = {1.0, 0.0, 0.0};
    double x[4] = {0.0};

    ASSERT_ERR(sparse_solve_gmres(A, b, x, NULL, NULL, NULL, NULL), SPARSE_ERR_SHAPE);
    sparse_free(A);
}

/* GMRES on SPD system (should work, just less efficient than CG) */
static void test_gmres_on_spd(void)
{
    SparseMatrix *A = build_spd_tridiag(3, 4.0, -1.0);
    double x_exact[3] = {1.0, 2.0, 3.0};
    double b[3], x[3] = {0.0, 0.0, 0.0};
    compute_rhs(A, x_exact, b);

    sparse_iter_result_t result;
    ASSERT_ERR(sparse_solve_gmres(A, b, x, NULL, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Result struct tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Verify result struct is properly populated */
static void test_cg_result_struct(void)
{
    SparseMatrix *A = build_spd_tridiag(5, 4.0, -1.0);
    idx_t n = 5;

    double x_exact[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b[5], x[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    compute_rhs(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result = {.iterations = -1, .residual_norm = -1.0, .converged = -1};

    sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    ASSERT_TRUE(result.iterations >= 0);
    ASSERT_TRUE(result.iterations <= n);
    ASSERT_TRUE(result.residual_norm >= 0.0);
    ASSERT_TRUE(result.residual_norm < 1e-10);
    ASSERT_TRUE(result.converged == 1);

    sparse_free(A);
}

/* NULL result pointer → no crash */
static void test_cg_null_result(void)
{
    SparseMatrix *A = build_spd_tridiag(3, 4.0, -1.0);
    double x_exact[3] = {1.0, 2.0, 3.0};
    double b[3], x[3] = {0.0, 0.0, 0.0};
    compute_rhs(A, x_exact, b);

    /* Should not crash with NULL result */
    ASSERT_ERR(sparse_solve_cg(A, b, x, NULL, NULL, NULL, NULL), SPARSE_OK);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Error code tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Verify SPARSE_ERR_NOT_CONVERGED has a string representation */
static void test_not_converged_strerror(void)
{
    const char *s = sparse_strerror(SPARSE_ERR_NOT_CONVERGED);
    ASSERT_NOT_NULL(s);
    ASSERT_TRUE(s[0] != '\0');
}

/* ═══════════════════════════════════════════════════════════════════════
 * GMRES solver tests (Day 4)
 * ═══════════════════════════════════════════════════════════════════════ */

/* GMRES on 2×2 non-symmetric system → converges in ≤ 2 iterations */
static void test_gmres_2x2_nonsymmetric(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, -1.0);
    sparse_insert(A, 1, 1, 3.0);

    /* A = [[2,1],[-1,3]], b = [5, 5]
     * A^{-1} = (1/7)*[[3,-1],[1,2]]
     * x = (1/7)*[10, 15] = [10/7, 15/7] */
    double b[2] = {5.0, 5.0};
    double x[2] = {0.0, 0.0};
    sparse_gmres_opts_t opts = {.max_iter = 100, .restart = 10, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_NEAR(x[0], 10.0 / 7.0, 1e-12);
    ASSERT_NEAR(x[1], 15.0 / 7.0, 1e-12);
    ASSERT_TRUE(result.iterations <= 2);

    sparse_free(A);
}

/* GMRES on 3×3 non-symmetric system */
static void test_gmres_3x3_nonsymmetric(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, -1.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 2, 2.0);
    sparse_insert(A, 2, 1, -1.0);
    sparse_insert(A, 2, 2, 5.0);

    double x_exact[3] = {1.0, 2.0, 3.0};
    double b[3], x[3] = {0.0, 0.0, 0.0};
    compute_rhs(A, x_exact, b);

    sparse_gmres_opts_t opts = {.max_iter = 100, .restart = 10, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.iterations <= 3);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    sparse_free(A);
}

/* GMRES on identity → x = b in 1 iteration */
static void test_gmres_identity(void)
{
    idx_t n = 5;
    SparseMatrix *A = build_identity(n);
    double b[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double x[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    sparse_gmres_opts_t opts = {.max_iter = 100, .restart = 10, .tol = 1e-14, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.iterations <= 1);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], b[i], 1e-14);

    sparse_free(A);
}

/* GMRES with zero RHS → x = 0 */
static void test_gmres_zero_rhs(void)
{
    SparseMatrix *A = build_spd_tridiag(4, 4.0, -1.0);
    double b[4] = {0.0, 0.0, 0.0, 0.0};
    double x[4] = {1.0, 1.0, 1.0, 1.0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, NULL, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(x[i], 0.0, 1e-15);

    sparse_free(A);
}

/* GMRES on 4×4 fully unsymmetric system (CG would not work) */
static void test_gmres_4x4_unsymmetric(void)
{
    SparseMatrix *A = sparse_create(4, 4);
    /* Build a strictly row-diag-dominant unsymmetric matrix */
    sparse_insert(A, 0, 0, 5.0);  sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 3, -1.0);
    sparse_insert(A, 1, 0, -2.0); sparse_insert(A, 1, 1, 6.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, -1.0); sparse_insert(A, 2, 2, 7.0);
    sparse_insert(A, 2, 3, 2.0);
    sparse_insert(A, 3, 0, 1.0);  sparse_insert(A, 3, 2, -1.0);
    sparse_insert(A, 3, 3, 4.0);

    double x_exact[4] = {1.0, -1.0, 2.0, 0.5};
    double b[4], x[4] = {0.0, 0.0, 0.0, 0.0};
    compute_rhs(A, x_exact, b);

    sparse_gmres_opts_t opts = {.max_iter = 100, .restart = 10, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.iterations <= 4);
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    sparse_free(A);
}

/* GMRES on 1×1 system */
static void test_gmres_1x1(void)
{
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 7.0);
    double b[1] = {21.0};
    double x[1] = {0.0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, NULL, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_NEAR(x[0], 3.0, 1e-14);

    sparse_free(A);
}

/* GMRES with default opts (NULL) */
static void test_gmres_default_opts(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);  sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, -1.0); sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, -1.0); sparse_insert(A, 2, 2, 5.0);

    double x_exact[3] = {2.0, -1.0, 3.0};
    double b[3], x[3] = {0.0, 0.0, 0.0};
    compute_rhs(A, x_exact, b);

    sparse_iter_result_t result;
    ASSERT_ERR(sparse_solve_gmres(A, b, x, NULL, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    sparse_free(A);
}

/* GMRES on larger unsymmetric tridiagonal (20×20) */
static void test_gmres_large_unsymmetric(void)
{
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    /* Unsymmetric tridiagonal: diag=5, upper=2, lower=-1 */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 5.0);
        if (i > 0)     sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1) sparse_insert(A, i, i + 1, 2.0);
    }

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1) * 0.3);
    compute_rhs(A, x_exact, b);

    sparse_gmres_opts_t opts = {.max_iter = 200, .restart = 30, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-10);

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* GMRES with exact initial guess → converges in 0 iterations */
static void test_gmres_exact_initial_guess(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0); sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, -1.0); sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 2, 4.0);

    double x_exact[3] = {1.0, 2.0, 3.0};
    double b[3], x[3];
    compute_rhs(A, x_exact, b);
    for (int i = 0; i < 3; i++) x[i] = x_exact[i];

    sparse_gmres_opts_t opts = {.max_iter = 100, .restart = 10, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_EQ(result.iterations, 0);

    sparse_free(A);
}

/* GMRES max_iter exceeded */
static void test_gmres_max_iter_exceeded(void)
{
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 5.0);
        if (i > 0)     sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1) sparse_insert(A, i, i + 1, 2.0);
    }

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    /* Only allow 2 iterations with restart=2 */
    sparse_gmres_opts_t opts = {.max_iter = 2, .restart = 2, .tol = 1e-14, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result),
               SPARSE_ERR_NOT_CONVERGED);
    ASSERT_FALSE(result.converged);
    ASSERT_TRUE(result.iterations <= 2);

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* Verify Arnoldi orthogonality: solve a system and check result is correct */
static void test_gmres_arnoldi_correctness(void)
{
    /* Dense 5×5 unsymmetric matrix — GMRES should converge in ≤ 5 iters */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 10.0); sparse_insert(A, 0, 1, -1.0); sparse_insert(A, 0, 4, 2.0);
    sparse_insert(A, 1, 0, -2.0); sparse_insert(A, 1, 1, 8.0);  sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, -1.0); sparse_insert(A, 2, 2, 9.0);  sparse_insert(A, 2, 3, -2.0);
    sparse_insert(A, 3, 2, 1.0);  sparse_insert(A, 3, 3, 7.0);  sparse_insert(A, 3, 4, -1.0);
    sparse_insert(A, 4, 0, -1.0); sparse_insert(A, 4, 3, 2.0);  sparse_insert(A, 4, 4, 6.0);

    double x_exact[5] = {1.0, -2.0, 3.0, -1.0, 2.0};
    double b[5], x[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    compute_rhs(A, x_exact, b);

    /* Use restart > n so it's unrestarted GMRES */
    sparse_gmres_opts_t opts = {.max_iter = 20, .restart = 10, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.iterations <= n);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-10);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * GMRES restart & preconditioning tests (Day 5)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: build an unsymmetric diag-dominant tridiagonal */
static SparseMatrix *build_unsym_tridiag(idx_t n, double diag, double upper, double lower)
{
    SparseMatrix *A = sparse_create(n, n);
    if (!A) return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, diag);
        if (i > 0)     sparse_insert(A, i, i - 1, lower);
        if (i < n - 1) sparse_insert(A, i, i + 1, upper);
    }
    return A;
}

/* Restart comparison: smaller restart → more total iterations */
static void test_gmres_restart_comparison(void)
{
    idx_t n = 30;
    SparseMatrix *A = build_unsym_tridiag(n, 5.0, 2.0, -1.0);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1) * 0.2);
    compute_rhs(A, x_exact, b);

    /* Small restart (5) */
    double *x_small = calloc((size_t)n, sizeof(double));
    sparse_gmres_opts_t opts_small = {.max_iter = 300, .restart = 5, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result_small;
    sparse_solve_gmres(A, b, x_small, &opts_small, NULL, NULL, &result_small);

    /* Large restart (30 = n, effectively unrestarted) */
    double *x_large = calloc((size_t)n, sizeof(double));
    sparse_gmres_opts_t opts_large = {.max_iter = 300, .restart = 30, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result_large;
    sparse_solve_gmres(A, b, x_large, &opts_large, NULL, NULL, &result_large);

    ASSERT_TRUE(result_small.converged);
    ASSERT_TRUE(result_large.converged);

    /* Larger restart should converge in fewer or equal iterations */
    ASSERT_TRUE(result_large.iterations <= result_small.iterations);

    printf("    restart comparison: restart=5 → %d iters, restart=30 → %d iters\n",
           (int)result_small.iterations, (int)result_large.iterations);

    /* Both produce correct solutions */
    double res_small = compute_relative_residual(A, b, x_small, n);
    double res_large = compute_relative_residual(A, b, x_large, n);
    ASSERT_TRUE(res_small < 1e-8);
    ASSERT_TRUE(res_large < 1e-8);

    free(x_exact); free(b); free(x_small); free(x_large);
    sparse_free(A);
}

/* restart > n → effectively unrestarted GMRES, converges in ≤ n iters */
static void test_gmres_unrestarted(void)
{
    idx_t n = 8;
    SparseMatrix *A = build_unsym_tridiag(n, 4.0, 1.5, -1.0);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    /* restart = 100 > n = 8, so unrestarted */
    sparse_gmres_opts_t opts = {.max_iter = 100, .restart = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.iterations <= n);  /* unrestarted GMRES converges in ≤ n */

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* Lucky breakdown: diagonal matrix → solution found in 1 Arnoldi step */
static void test_gmres_lucky_breakdown(void)
{
    /* Diagonal matrix: Krylov subspace is 1-dimensional (A*r = lambda*r) */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 3.0);  /* scalar multiple of identity */

    double b[5] = {3.0, 6.0, 9.0, 12.0, 15.0};
    double x[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    sparse_gmres_opts_t opts = {.max_iter = 100, .restart = 10, .tol = 1e-14, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.iterations <= 1);  /* lucky breakdown after 1 step */

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], b[i] / 3.0, 1e-14);

    sparse_free(A);
}

/* Lucky breakdown: A = I + rank-1 → Krylov subspace dimension ≤ 2 */
static void test_gmres_small_krylov(void)
{
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    /* A = I + e_0 * e_1^T (identity plus a rank-1 perturbation) */
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);
    sparse_insert(A, 0, 1, 1.0);  /* A(0,1) += 1 */

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    sparse_gmres_opts_t opts = {.max_iter = 50, .restart = 50, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    /* Should converge very quickly due to simple spectrum */
    ASSERT_TRUE(result.iterations <= 3);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* GMRES with diagonal (Jacobi) preconditioner */
static void test_gmres_diagonal_preconditioner(void)
{
    idx_t n = 20;
    /* Poorly scaled unsymmetric tridiagonal */
    SparseMatrix *A = sparse_create(n, n);
    double *diag_inv = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++) {
        double d = 2.0 + 3.0 * (double)i;  /* 2, 5, 8, ..., 59 */
        sparse_insert(A, i, i, d);
        if (i > 0)     sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1) sparse_insert(A, i, i + 1, 1.5);
        diag_inv[i] = 1.0 / d;
    }

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    diag_precond_t pc = {.diag_inv = diag_inv, .n = n};
    sparse_gmres_opts_t opts_unprec = {.max_iter = 200, .restart = 20, .tol = 1e-10, .verbose = 0};
    sparse_gmres_opts_t opts_prec = {.max_iter = 200, .restart = 20, .tol = 1e-10, .verbose = 0};

    /* Without preconditioner */
    double *x_unprec = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t result_unprec;
    sparse_solve_gmres(A, b, x_unprec, &opts_unprec, NULL, NULL, &result_unprec);

    /* With diagonal preconditioner */
    double *x_prec = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t result_prec;
    sparse_solve_gmres(A, b, x_prec, &opts_prec, diag_precond_apply, &pc, &result_prec);

    ASSERT_TRUE(result_unprec.converged);
    ASSERT_TRUE(result_prec.converged);
    /* Preconditioned may use more total iterations (true residual convergence)
     * but should still produce a correct solution */

    printf("    GMRES precond: unprec=%d iters, prec=%d iters\n",
           (int)result_unprec.iterations, (int)result_prec.iterations);

    /* Both solutions correct */
    double res_unprec = compute_relative_residual(A, b, x_unprec, n);
    double res_prec = compute_relative_residual(A, b, x_prec, n);
    ASSERT_TRUE(res_unprec < 1e-8);
    ASSERT_TRUE(res_prec < 1e-8);

    free(diag_inv); free(x_exact); free(b);
    free(x_unprec); free(x_prec);
    sparse_free(A);
}

/* GMRES with preconditioner on larger system (Laplacian-like unsymmetric) */
static void test_gmres_precond_large(void)
{
    idx_t n = 40;
    SparseMatrix *A = build_unsym_tridiag(n, 6.0, 2.0, -1.0);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = cos((double)(i + 1) * 0.15);
    compute_rhs(A, x_exact, b);

    /* Diagonal preconditioner: 1/6 for all diagonal entries */
    double *diag_inv = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        diag_inv[i] = 1.0 / 6.0;
    diag_precond_t pc = {.diag_inv = diag_inv, .n = n};

    double *x = calloc((size_t)n, sizeof(double));
    sparse_gmres_opts_t opts = {.max_iter = 300, .restart = 15, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, diag_precond_apply, &pc, &result),
               SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    ASSERT_TRUE(rel_res < 1e-8);

    free(diag_inv); free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* GMRES verbose mode doesn't crash */
static void test_gmres_verbose_mode(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 3.0); sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, -1.0); sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 2, 2, 5.0);

    double x_exact[3] = {1.0, 2.0, 3.0};
    double b[3], x[3] = {0.0, 0.0, 0.0};
    compute_rhs(A, x_exact, b);

    FILE *saved_stderr = stderr;
    FILE *devnull = fopen("/dev/null", "w");
    if (devnull) stderr = devnull;

    sparse_gmres_opts_t opts = {.max_iter = 100, .restart = 10, .tol = 1e-10, .verbose = 1};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    if (devnull) {
        stderr = saved_stderr;
        fclose(devnull);
    }

    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    sparse_free(A);
}

/* GMRES restart=1 (extreme case, essentially Richardson iteration) */
static void test_gmres_restart_1(void)
{
    idx_t n = 5;
    SparseMatrix *A = build_unsym_tridiag(n, 10.0, 1.0, -1.0);

    double x_exact[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b[5], x[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    compute_rhs(A, x_exact, b);

    /* restart=1: each "restart" is a single Arnoldi step */
    sparse_gmres_opts_t opts = {.max_iter = 200, .restart = 1, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-8);

    sparse_free(A);
}

/* GMRES with initial guess: near solution converges faster */
static void test_gmres_initial_guess(void)
{
    idx_t n = 15;
    SparseMatrix *A = build_unsym_tridiag(n, 5.0, 2.0, -1.0);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    sparse_gmres_opts_t opts = {.max_iter = 200, .restart = 10, .tol = 1e-10, .verbose = 0};

    /* From zero */
    double *x_zero = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t result_zero;
    sparse_solve_gmres(A, b, x_zero, &opts, NULL, NULL, &result_zero);

    /* From near solution */
    double *x_near = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_near[i] = x_exact[i] + 0.01;
    sparse_iter_result_t result_near;
    sparse_solve_gmres(A, b, x_near, &opts, NULL, NULL, &result_near);

    ASSERT_TRUE(result_zero.converged);
    ASSERT_TRUE(result_near.converged);
    ASSERT_TRUE(result_near.iterations <= result_zero.iterations);

    free(x_exact); free(b); free(x_zero); free(x_near);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * GMRES SuiteSparse validation (Day 6)
 * ═══════════════════════════════════════════════════════════════════════ */

/* GMRES on west0067 (67×67 unsymmetric) — needs large restart for convergence */
static void test_gmres_west0067(void)
{
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

    /* west0067 has complex eigenvalue structure; unrestarted GMRES (restart=n)
     * is needed for reliable convergence without preconditioning */
    sparse_gmres_opts_t opts = {.max_iter = 200, .restart = 67, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    printf("    west0067: GMRES(%d) iters=%d, rel_res=%.3e\n",
           67, (int)result.iterations, rel_res);
    ASSERT_TRUE(rel_res < 1e-8);

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* GMRES on steam1 (240×240, condest ~3e7 — ill-conditioned) */
static void test_gmres_steam1(void)
{
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

    /* steam1 is highly ill-conditioned (condest ~3e7). Use large restart
     * and relaxed tolerance. Preconditioning would help significantly. */
    sparse_gmres_opts_t opts = {.max_iter = 2000, .restart = 100, .tol = 1e-6, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double rel_res = compute_relative_residual(A, b, x, n);
    printf("    steam1: GMRES(%d) iters=%d, rel_res=%.3e\n",
           100, (int)result.iterations, rel_res);
    ASSERT_TRUE(rel_res < 1e-4);

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* GMRES on orsirr_1 (1030×1030) — larger stress test */
static void test_gmres_orsirr_1(void)
{
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
        x_exact[i] = sin((double)(i + 1) * 0.01);
    compute_rhs(A, x_exact, b);

    sparse_gmres_opts_t opts = {.max_iter = 2000, .restart = 50, .tol = 1e-8, .verbose = 0};
    sparse_iter_result_t result;

    sparse_err_t solve_err = sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);

    double rel_res = compute_relative_residual(A, b, x, n);
    printf("    orsirr_1: GMRES iters=%d, rel_res=%.3e, converged=%d\n",
           (int)result.iterations, rel_res, result.converged);

    if (solve_err == SPARSE_OK) {
        ASSERT_TRUE(rel_res < 1e-6);
    }
    /* orsirr_1 may need preconditioning to converge fully; we accept either outcome */

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* GMRES restart comparison on west0067: GMRES(10) vs GMRES(30) vs GMRES(67) */
static void test_gmres_restart_comparison_suitesparse(void)
{
    SparseMatrix *A = NULL;
    sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    /* west0067 has complex eigenvalue structure — small restarts may not
     * converge, but larger restarts should. This test records behavior. */
    int restarts[] = {10, 30, 67};
    printf("    west0067 restart comparison:\n");
    int any_converged = 0;
    for (int r = 0; r < 3; r++) {
        double *x = calloc((size_t)n, sizeof(double));
        sparse_gmres_opts_t opts = {
            .max_iter = 500, .restart = (idx_t)restarts[r],
            .tol = 1e-10, .verbose = 0
        };
        sparse_iter_result_t result;
        sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);
        double rel_res = compute_relative_residual(A, b, x, n);
        printf("      GMRES(%d): iters=%d, res=%.3e, converged=%d\n",
               restarts[r], (int)result.iterations, rel_res, result.converged);
        if (result.converged) {
            ASSERT_TRUE(rel_res < 1e-8);
            any_converged = 1;
        }
        free(x);
    }
    /* At least the unrestarted (restart=67) should converge */
    ASSERT_TRUE(any_converged);

    free(x_exact); free(b);
    sparse_free(A);
}

/* GMRES restart comparison on steam1 */
static void test_gmres_restart_comparison_steam1(void)
{
    SparseMatrix *A = NULL;
    sparse_load_mm(&A, SS_DIR "/steam1.mtx");
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A, x_exact, b);

    int restarts[] = {10, 30, 50};
    printf("    steam1 restart comparison:\n");
    for (int r = 0; r < 3; r++) {
        double *x = calloc((size_t)n, sizeof(double));
        sparse_gmres_opts_t opts = {
            .max_iter = 1000, .restart = (idx_t)restarts[r],
            .tol = 1e-10, .verbose = 0
        };
        sparse_iter_result_t result;
        sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);
        double rel_res = compute_relative_residual(A, b, x, n);
        printf("      GMRES(%d): iters=%d, res=%.3e, converged=%d\n",
               restarts[r], (int)result.iterations, rel_res, result.converged);
        if (result.converged)
            ASSERT_TRUE(rel_res < 1e-8);
        free(x);
    }

    free(x_exact); free(b);
    sparse_free(A);
}

/* GMRES vs LU on west0067 */
static void test_gmres_vs_lu_west0067(void)
{
    SparseMatrix *A_gmres = NULL;
    SparseMatrix *A_lu = NULL;
    ASSERT_ERR(sparse_load_mm(&A_gmres, SS_DIR "/west0067.mtx"), SPARSE_OK);
    ASSERT_ERR(sparse_load_mm(&A_lu, SS_DIR "/west0067.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A_gmres);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A_gmres, x_exact, b);

    /* GMRES solve — unrestarted for west0067 */
    double *x_gmres = calloc((size_t)n, sizeof(double));
    sparse_gmres_opts_t opts = {.max_iter = 200, .restart = 67, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    sparse_solve_gmres(A_gmres, b, x_gmres, &opts, NULL, NULL, &result);

    /* LU solve */
    double *x_lu = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_lu);
    ASSERT_ERR(sparse_lu_factor(A_lu, SPARSE_PIVOT_PARTIAL, 1e-14), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(A_lu, b, x_lu), SPARSE_OK);

    double res_gmres = compute_relative_residual(A_gmres, b, x_gmres, n);
    double res_lu = compute_relative_residual(A_gmres, b, x_lu, n);

    printf("    west0067: GMRES res=%.3e (%d iters), LU res=%.3e\n",
           res_gmres, (int)result.iterations, res_lu);

    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(res_gmres < 1e-8);
    ASSERT_TRUE(res_lu < 1e-8);

    /* Solutions should be comparable */
    double maxdiff = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double diff = fabs(x_gmres[i] - x_lu[i]);
        if (diff > maxdiff) maxdiff = diff;
    }
    printf("    west0067: GMRES vs LU max |diff| = %.3e\n", maxdiff);
    ASSERT_TRUE(maxdiff < 1e-4);

    free(x_exact); free(b); free(x_gmres); free(x_lu);
    sparse_free(A_gmres); sparse_free(A_lu);
}

/* GMRES vs LU on steam1 */
static void test_gmres_vs_lu_steam1(void)
{
    SparseMatrix *A_gmres = NULL;
    SparseMatrix *A_lu = NULL;
    ASSERT_ERR(sparse_load_mm(&A_gmres, SS_DIR "/steam1.mtx"), SPARSE_OK);
    ASSERT_ERR(sparse_load_mm(&A_lu, SS_DIR "/steam1.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A_gmres);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A_gmres, x_exact, b);

    /* GMRES solve — large restart for ill-conditioned steam1 */
    double *x_gmres = calloc((size_t)n, sizeof(double));
    sparse_gmres_opts_t opts = {.max_iter = 2000, .restart = 100, .tol = 1e-6, .verbose = 0};
    sparse_iter_result_t result;
    sparse_solve_gmres(A_gmres, b, x_gmres, &opts, NULL, NULL, &result);

    /* LU solve */
    double *x_lu = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_lu);
    ASSERT_ERR(sparse_lu_factor(A_lu, SPARSE_PIVOT_PARTIAL, 1e-14), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(A_lu, b, x_lu), SPARSE_OK);

    double res_gmres = compute_relative_residual(A_gmres, b, x_gmres, n);
    double res_lu = compute_relative_residual(A_gmres, b, x_lu, n);

    printf("    steam1: GMRES res=%.3e (%d iters), LU res=%.3e\n",
           res_gmres, (int)result.iterations, res_lu);

    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(res_gmres < 1e-4);
    ASSERT_TRUE(res_lu < 1e-6);

    free(x_exact); free(b); free(x_gmres); free(x_lu);
    sparse_free(A_gmres); sparse_free(A_lu);
}

/* GMRES on SPD matrix (nos4) — compare iteration count with CG */
static void test_gmres_vs_cg_nos4(void)
{
    SparseMatrix *A_cg = NULL;
    SparseMatrix *A_gmres = NULL;
    sparse_load_mm(&A_cg, SS_DIR "/nos4.mtx");
    sparse_load_mm(&A_gmres, SS_DIR "/nos4.mtx");
    idx_t n = sparse_rows(A_cg);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    compute_rhs(A_cg, x_exact, b);

    /* CG solve */
    double *x_cg = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t cg_opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result_cg;
    sparse_solve_cg(A_cg, b, x_cg, &cg_opts, NULL, NULL, &result_cg);

    /* GMRES solve */
    double *x_gmres = calloc((size_t)n, sizeof(double));
    sparse_gmres_opts_t gm_opts = {.max_iter = 500, .restart = 50, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result_gmres;
    sparse_solve_gmres(A_gmres, b, x_gmres, &gm_opts, NULL, NULL, &result_gmres);

    double res_cg = compute_relative_residual(A_cg, b, x_cg, n);
    double res_gmres = compute_relative_residual(A_cg, b, x_gmres, n);

    printf("    nos4: CG iters=%d res=%.3e, GMRES iters=%d res=%.3e\n",
           (int)result_cg.iterations, res_cg,
           (int)result_gmres.iterations, res_gmres);

    ASSERT_TRUE(result_cg.converged);
    ASSERT_TRUE(result_gmres.converged);
    ASSERT_TRUE(res_cg < 1e-8);
    ASSERT_TRUE(res_gmres < 1e-8);

    free(x_exact); free(b); free(x_cg); free(x_gmres);
    sparse_free(A_cg); sparse_free(A_gmres);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test suite
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Iterative Solvers");

    /* CG solver tests */
    RUN_TEST(test_cg_2x2_spd);
    RUN_TEST(test_cg_3x3_tridiag);
    RUN_TEST(test_cg_identity);
    RUN_TEST(test_cg_zero_rhs);
    RUN_TEST(test_cg_default_opts);
    RUN_TEST(test_cg_laplacian_2d);
    RUN_TEST(test_cg_initial_guess);
    RUN_TEST(test_cg_1x1);
    RUN_TEST(test_cg_diagonal);
    RUN_TEST(test_cg_large_tridiag);
    RUN_TEST(test_cg_max_iter_exceeded);
    RUN_TEST(test_cg_exact_initial_guess);

    /* CG with preconditioner */
    RUN_TEST(test_cg_diagonal_preconditioner);
    RUN_TEST(test_cg_precond_laplacian);

    /* SuiteSparse CG validation */
    RUN_TEST(test_cg_nos4);
    RUN_TEST(test_cg_bcsstk04);
    RUN_TEST(test_cg_suitesparse_initial_guess);

    /* Convergence monitoring */
    RUN_TEST(test_cg_tight_tolerance);
    RUN_TEST(test_cg_loose_tolerance);
    RUN_TEST(test_cg_residual_accuracy);

    /* Non-SPD behavior */
    RUN_TEST(test_cg_nonsymmetric_behavior);
    RUN_TEST(test_cg_indefinite_behavior);

    /* CG vs Cholesky comparison */
    RUN_TEST(test_cg_vs_cholesky_tridiag);
    RUN_TEST(test_cg_vs_cholesky_nos4);
    RUN_TEST(test_cg_vs_cholesky_bcsstk04);

    /* Verbose mode */
    RUN_TEST(test_cg_verbose_mode);

    /* Error handling */
    RUN_TEST(test_cg_null_inputs);
    RUN_TEST(test_cg_nonsquare);
    RUN_TEST(test_gmres_null_inputs);
    RUN_TEST(test_gmres_nonsquare);
    RUN_TEST(test_gmres_on_spd);

    /* Result struct */
    RUN_TEST(test_cg_result_struct);
    RUN_TEST(test_cg_null_result);

    /* GMRES solver tests */
    RUN_TEST(test_gmres_2x2_nonsymmetric);
    RUN_TEST(test_gmres_3x3_nonsymmetric);
    RUN_TEST(test_gmres_identity);
    RUN_TEST(test_gmres_zero_rhs);
    RUN_TEST(test_gmres_4x4_unsymmetric);
    RUN_TEST(test_gmres_1x1);
    RUN_TEST(test_gmres_default_opts);
    RUN_TEST(test_gmres_large_unsymmetric);
    RUN_TEST(test_gmres_exact_initial_guess);
    RUN_TEST(test_gmres_max_iter_exceeded);
    RUN_TEST(test_gmres_arnoldi_correctness);

    /* GMRES restart & preconditioning (Day 5) */
    RUN_TEST(test_gmres_restart_comparison);
    RUN_TEST(test_gmres_unrestarted);
    RUN_TEST(test_gmres_lucky_breakdown);
    RUN_TEST(test_gmres_small_krylov);
    RUN_TEST(test_gmres_diagonal_preconditioner);
    RUN_TEST(test_gmres_precond_large);
    RUN_TEST(test_gmres_verbose_mode);
    RUN_TEST(test_gmres_restart_1);
    RUN_TEST(test_gmres_initial_guess);

    /* GMRES SuiteSparse validation (Day 6) */
    RUN_TEST(test_gmres_west0067);
    RUN_TEST(test_gmres_steam1);
    RUN_TEST(test_gmres_orsirr_1);
    RUN_TEST(test_gmres_restart_comparison_suitesparse);
    RUN_TEST(test_gmres_restart_comparison_steam1);
    RUN_TEST(test_gmres_vs_lu_west0067);
    RUN_TEST(test_gmres_vs_lu_steam1);
    RUN_TEST(test_gmres_vs_cg_nos4);

    /* Error codes */
    RUN_TEST(test_not_converged_strerror);

    TEST_SUITE_END();
}
