#include "sparse_matrix.h"
#include "sparse_iterative.h"
#include "sparse_vector.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#ifdef SPARSE_OPENMP
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <omp.h>
#pragma GCC diagnostic pop
#endif

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

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
 * Compute SpMV y = A*x and return the result in caller-provided y.
 * Also compute a reference result using element-by-element accumulation
 * to verify against.
 */
static void verify_matvec(const SparseMatrix *A, const double *x,
                           idx_t n, double tol)
{
    double *y = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(y);
    ASSERT_ERR(sparse_matvec(A, x, y), SPARSE_OK);

    /* Verify using sparse_get to compute the product independently */
    for (idx_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (idx_t j = 0; j < sparse_cols(A); j++) {
            double aij = sparse_get(A, i, j);
            if (aij != 0.0)
                sum += aij * x[j];
        }
        ASSERT_NEAR(y[i], sum, tol);
    }

    free(y);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SpMV correctness tests (should pass with both serial and OpenMP builds)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Report whether OpenMP is active */
static void test_omp_status(void)
{
#ifdef SPARSE_OPENMP
    int nthreads = omp_get_max_threads();
    printf("    OpenMP ENABLED: max_threads=%d\n", nthreads);
    ASSERT_TRUE(nthreads >= 1);
#else
    printf("    OpenMP DISABLED (serial build)\n");
    ASSERT_TRUE(1);  /* always passes */
#endif
}

/* SpMV on identity matrix */
static void test_spmv_identity(void)
{
    idx_t n = 50;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    double *x = malloc((size_t)n * sizeof(double));
    double *y = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x[i] = (double)(i + 1);

    ASSERT_ERR(sparse_matvec(A, x, y), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(y[i], x[i], 1e-15);

    free(x); free(y);
    sparse_free(A);
}

/* SpMV on tridiagonal matrix — verify element by element */
static void test_spmv_tridiag(void)
{
    idx_t n = 100;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);

    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x[i] = sin((double)(i + 1) * 0.1);

    verify_matvec(A, x, n, 1e-12);

    free(x);
    sparse_free(A);
}

/* SpMV on 2D Laplacian (more complex connectivity) */
static void test_spmv_laplacian(void)
{
    idx_t m = 8;
    idx_t n = m * m;  /* 64×64 */
    SparseMatrix *A = build_laplacian_2d(m);

    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x[i] = (double)(i + 1);

    verify_matvec(A, x, n, 1e-10);

    free(x);
    sparse_free(A);
}

/* SpMV on SuiteSparse nos4 (100×100 SPD) */
static void test_spmv_nos4(void)
{
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/nos4.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A);

    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x[i] = (double)(i + 1);

    verify_matvec(A, x, n, 1e-10);

    free(x);
    sparse_free(A);
}

/* SpMV on SuiteSparse west0067 (67×67 unsymmetric) */
static void test_spmv_west0067(void)
{
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/west0067.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A);

    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x[i] = sin((double)(i + 1) * 0.2);

    verify_matvec(A, x, n, 1e-10);

    free(x);
    sparse_free(A);
}

/* SpMV on SuiteSparse steam1 (240×240) */
static void test_spmv_steam1(void)
{
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/steam1.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A);

    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x[i] = (double)(i + 1);

    verify_matvec(A, x, n, 1e-8);

    free(x);
    sparse_free(A);
}

/* SpMV multiple times — verify reproducibility */
static void test_spmv_reproducible(void)
{
    idx_t n = 80;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);

    double *x = malloc((size_t)n * sizeof(double));
    double *y1 = malloc((size_t)n * sizeof(double));
    double *y2 = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x[i] = (double)(i + 1);

    sparse_matvec(A, x, y1);
    sparse_matvec(A, x, y2);

    /* Results must be bit-identical across calls */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(y1[i], y2[i], 0.0);

    free(x); free(y1); free(y2);
    sparse_free(A);
}

/* SpMV edge case: sparse matrix with only diagonal entries, 2×2 */
static void test_spmv_diagonal_small(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    double x[2] = {2.0, 4.0};
    double y[2];
    ASSERT_ERR(sparse_matvec(A, x, y), SPARSE_OK);
    ASSERT_NEAR(y[0], 6.0, 1e-15);
    ASSERT_NEAR(y[1], 20.0, 1e-15);
    sparse_free(A);
}

/* SpMV edge case: 1×1 matrix */
static void test_spmv_1x1(void)
{
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 7.0);
    double x = 3.0, y = 0.0;
    ASSERT_ERR(sparse_matvec(A, &x, &y), SPARSE_OK);
    ASSERT_NEAR(y, 21.0, 1e-15);
    sparse_free(A);
}

/* SpMV with CG — verify CG still converges correctly with parallel SpMV */
static void test_spmv_cg_integration(void)
{
    idx_t n = 50;
    SparseMatrix *A = build_spd_tridiag(n, 4.0, -1.0);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-8);

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* SpMV with GMRES — verify GMRES still converges correctly with parallel SpMV */
static void test_spmv_gmres_integration(void)
{
    idx_t n = 30;
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
        x_exact[i] = sin((double)(i + 1) * 0.2);
    sparse_matvec(A, x_exact, b);

    sparse_gmres_opts_t opts = {.max_iter = 200, .restart = 30, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-8);

    free(x_exact); free(b); free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test suite
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Parallel SpMV (OpenMP)");

    RUN_TEST(test_omp_status);

    /* SpMV correctness */
    RUN_TEST(test_spmv_identity);
    RUN_TEST(test_spmv_tridiag);
    RUN_TEST(test_spmv_laplacian);
    RUN_TEST(test_spmv_nos4);
    RUN_TEST(test_spmv_west0067);
    RUN_TEST(test_spmv_steam1);
    RUN_TEST(test_spmv_reproducible);

    /* Edge cases */
    RUN_TEST(test_spmv_diagonal_small);
    RUN_TEST(test_spmv_1x1);

    /* Integration with iterative solvers */
    RUN_TEST(test_spmv_cg_integration);
    RUN_TEST(test_spmv_gmres_integration);

    TEST_SUITE_END();
}
