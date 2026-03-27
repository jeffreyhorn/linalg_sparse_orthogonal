#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif

/* ─── Helpers ────────────────────────────────────────────────────────── */

static double vec_norminf(const double *v, idx_t n) {
    double mx = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double a = fabs(v[i]);
        if (a > mx)
            mx = a;
    }
    return mx;
}

/*
 * Load a matrix from file, set up b = A * x_exact where x_exact = [1,1,...,1],
 * factorize, solve, check residual.
 */
static double load_solve_residual(const char *path, sparse_pivot_t pivot) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, path) != SPARSE_OK)
        return -1.0;

    idx_t n = sparse_rows(A);
    if (n != sparse_cols(A)) {
        sparse_free(A);
        return -2.0;
    }

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));

    /* x_exact = [1, 1, ..., 1] */
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;

    /* b = A * x_exact */
    sparse_matvec(A, x_exact, b);

    /* Copy for factorization */
    SparseMatrix *LU = sparse_copy(A);

    sparse_err_t err = sparse_lu_factor(LU, pivot, 1e-12);
    if (err != SPARSE_OK) {
        sparse_free(A);
        sparse_free(LU);
        free(x_exact);
        free(b);
        free(x);
        free(r);
        return -3.0;
    }

    err = sparse_lu_solve(LU, b, x);
    if (err != SPARSE_OK) {
        sparse_free(A);
        sparse_free(LU);
        free(x_exact);
        free(b);
        free(x);
        free(r);
        return -4.0;
    }

    /* Residual: r = A*x - b */
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] -= b[i];
    double res = vec_norminf(r, n);

    sparse_free(A);
    sparse_free(LU);
    free(x_exact);
    free(b);
    free(x);
    free(r);
    return res;
}

/*
 * Same as above but also does iterative refinement.
 */
static double load_solve_refine_residual(const char *path, sparse_pivot_t pivot) {
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, path) != SPARSE_OK)
        return -1.0;

    idx_t n = sparse_rows(A);
    if (n != sparse_cols(A)) {
        sparse_free(A);
        return -2.0;
    }

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;
    sparse_matvec(A, x_exact, b);

    SparseMatrix *LU = sparse_copy(A);
    sparse_lu_factor(LU, pivot, 1e-12);
    sparse_lu_solve(LU, b, x);

    /* Refine */
    sparse_lu_refine(A, LU, b, x, 5, 1e-15);

    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] -= b[i];
    double res = vec_norminf(r, n);

    sparse_free(A);
    sparse_free(LU);
    free(x_exact);
    free(b);
    free(x);
    free(r);
    return res;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Identity matrix
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_identity_complete(void) {
    double res = load_solve_residual(DATA_DIR "/identity_5.mtx", SPARSE_PIVOT_COMPLETE);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_NEAR(res, 0.0, 1e-14);
}

static void test_identity_partial(void) {
    double res = load_solve_residual(DATA_DIR "/identity_5.mtx", SPARSE_PIVOT_PARTIAL);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_NEAR(res, 0.0, 1e-14);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Diagonal matrix
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_diagonal_complete(void) {
    double res = load_solve_residual(DATA_DIR "/diagonal_10.mtx", SPARSE_PIVOT_COMPLETE);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_NEAR(res, 0.0, 1e-13);
}

static void test_diagonal_partial(void) {
    double res = load_solve_residual(DATA_DIR "/diagonal_10.mtx", SPARSE_PIVOT_PARTIAL);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_NEAR(res, 0.0, 1e-13);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Tridiagonal (Poisson 1D, n=20)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_tridiag_complete(void) {
    double res = load_solve_residual(DATA_DIR "/tridiagonal_20.mtx", SPARSE_PIVOT_COMPLETE);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_TRUE(res < 1e-10);
}

static void test_tridiag_partial(void) {
    double res = load_solve_residual(DATA_DIR "/tridiagonal_20.mtx", SPARSE_PIVOT_PARTIAL);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_TRUE(res < 1e-10);
}

static void test_tridiag_refined(void) {
    double res = load_solve_refine_residual(DATA_DIR "/tridiagonal_20.mtx", SPARSE_PIVOT_COMPLETE);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_TRUE(res < 1e-13);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Symmetric matrix (loaded and expanded)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_symmetric_complete(void) {
    double res = load_solve_residual(DATA_DIR "/symmetric_4.mtx", SPARSE_PIVOT_COMPLETE);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_NEAR(res, 0.0, 1e-10);
}

static void test_symmetric_partial(void) {
    double res = load_solve_residual(DATA_DIR "/symmetric_4.mtx", SPARSE_PIVOT_PARTIAL);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_NEAR(res, 0.0, 1e-10);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Structural engineering matrix (bcsstk01-inspired, 6x6 SPD)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_bcsstk01_complete(void) {
    double res = load_solve_residual(DATA_DIR "/bcsstk01.mtx", SPARSE_PIVOT_COMPLETE);
    ASSERT_TRUE(res >= 0.0);
    /* Large entries (~1e6), so absolute residual may be larger */
    ASSERT_TRUE(res < 1e-4);
}

static void test_bcsstk01_refined(void) {
    double res = load_solve_refine_residual(DATA_DIR "/bcsstk01.mtx", SPARSE_PIVOT_COMPLETE);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_TRUE(res < 1e-6);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Small unsymmetric matrix
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_unsymm_complete(void) {
    double res = load_solve_residual(DATA_DIR "/unsymm_5.mtx", SPARSE_PIVOT_COMPLETE);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_NEAR(res, 0.0, 1e-12);
}

static void test_unsymm_partial(void) {
    double res = load_solve_residual(DATA_DIR "/unsymm_5.mtx", SPARSE_PIVOT_PARTIAL);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_NEAR(res, 0.0, 1e-12);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Solution accuracy (compare against known x_exact = [1,...,1])
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_solution_accuracy_diagonal(void) {
    SparseMatrix *A = NULL;
    sparse_load_mm(&A, DATA_DIR "/diagonal_10.mtx");
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;
    sparse_matvec(A, x_exact, b);

    SparseMatrix *LU = sparse_copy(A);
    sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12);
    sparse_lu_solve(LU, b, x);

    /* x should match x_exact closely */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-13);

    sparse_free(A);
    sparse_free(LU);
    free(x_exact);
    free(b);
    free(x);
}

static void test_solution_accuracy_unsymm(void) {
    SparseMatrix *A = NULL;
    sparse_load_mm(&A, DATA_DIR "/unsymm_5.mtx");
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;
    sparse_matvec(A, x_exact, b);

    SparseMatrix *LU = sparse_copy(A);
    sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12);
    sparse_lu_solve(LU, b, x);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-12);

    sparse_free(A);
    sparse_free(LU);
    free(x_exact);
    free(b);
    free(x);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Known Matrix Tests");

    /* Identity */
    RUN_TEST(test_identity_complete);
    RUN_TEST(test_identity_partial);

    /* Diagonal */
    RUN_TEST(test_diagonal_complete);
    RUN_TEST(test_diagonal_partial);

    /* Tridiagonal */
    RUN_TEST(test_tridiag_complete);
    RUN_TEST(test_tridiag_partial);
    RUN_TEST(test_tridiag_refined);

    /* Symmetric */
    RUN_TEST(test_symmetric_complete);
    RUN_TEST(test_symmetric_partial);

    /* Structural engineering (bcsstk01-inspired) */
    RUN_TEST(test_bcsstk01_complete);
    RUN_TEST(test_bcsstk01_refined);

    /* Unsymmetric */
    RUN_TEST(test_unsymm_complete);
    RUN_TEST(test_unsymm_partial);

    /* Solution accuracy */
    RUN_TEST(test_solution_accuracy_diagonal);
    RUN_TEST(test_solution_accuracy_unsymm);

    TEST_SUITE_END();
}
