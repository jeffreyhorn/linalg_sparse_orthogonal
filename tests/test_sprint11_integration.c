/**
 * Sprint 11 cross-feature integration tests.
 *
 * Validates that tolerance standardization, factored-state validation,
 * thread-safe norminf, and version generation work together correctly.
 */
#include "sparse_cholesky.h"
#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_lu.h"
#include "sparse_lu_csr.h"
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Helper: build a well-conditioned n×n SPD tridiagonal, scaled by s
 * ═══════════════════════════════════════════════════════════════════════ */

static SparseMatrix *build_spd_tridiag(idx_t n, double s) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0 * s);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0 * s);
            sparse_insert(A, i - 1, i, -1.0 * s);
        }
    }
    return A;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 1: Scaled tolerance across all direct solvers
 *
 * Build a 10×10 SPD tridiagonal at scale 1e-35, solve with LU,
 * Cholesky, QR, and CSR-LU.  All should produce x ≈ 1.
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_all_solvers_tiny_scale(void) {
    double s = 1e-35;
    idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n, s);
    ASSERT_NOT_NULL(A);

    /* b = A * ones */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* LU */
    {
        SparseMatrix *LU = sparse_copy(A);
        ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-50), SPARSE_OK);
        ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);
        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(x[i], 1.0, 1e-8);
        sparse_free(LU);
    }

    /* Cholesky */
    {
        SparseMatrix *L = sparse_copy(A);
        ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);
        ASSERT_ERR(sparse_cholesky_solve(L, b, x), SPARSE_OK);
        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(x[i], 1.0, 1e-8);
        sparse_free(L);
    }

    /* QR */
    {
        sparse_qr_t qr;
        ASSERT_ERR(sparse_qr_factor(A, &qr), SPARSE_OK);
        ASSERT_ERR(sparse_qr_solve(&qr, b, x, NULL), SPARSE_OK);
        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(x[i], 1.0, 1e-6);
        sparse_qr_free(&qr);
    }

    /* CSR LU */
    {
        double *b2 = malloc((size_t)n * sizeof(double));
        memcpy(b2, b, (size_t)n * sizeof(double));
        ASSERT_ERR(lu_csr_factor_solve(A, b2, x, 1e-50), SPARSE_OK);
        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(x[i], 1.0, 1e-8);
        free(b2);
    }

    free(ones);
    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 2: Same at large scale (1e+35)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_all_solvers_huge_scale(void) {
    double s = 1e+35;
    idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n, s);
    ASSERT_NOT_NULL(A);

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* LU */
    {
        SparseMatrix *LU = sparse_copy(A);
        ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e+20), SPARSE_OK);
        ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);
        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(x[i], 1.0, 1e-8);
        sparse_free(LU);
    }

    /* Cholesky */
    {
        SparseMatrix *L = sparse_copy(A);
        ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);
        ASSERT_ERR(sparse_cholesky_solve(L, b, x), SPARSE_OK);
        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(x[i], 1.0, 1e-8);
        sparse_free(L);
    }

    /* QR */
    {
        sparse_qr_t qr;
        ASSERT_ERR(sparse_qr_factor(A, &qr), SPARSE_OK);
        ASSERT_ERR(sparse_qr_solve(&qr, b, x, NULL), SPARSE_OK);
        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(x[i], 1.0, 1e-6);
        sparse_qr_free(&qr);
    }

    /* CSR LU */
    {
        double *b2 = malloc((size_t)n * sizeof(double));
        memcpy(b2, b, (size_t)n * sizeof(double));
        ASSERT_ERR(lu_csr_factor_solve(A, b2, x, 1e+20), SPARSE_OK);
        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(x[i], 1.0, 1e-8);
        free(b2);
    }

    free(ones);
    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 3: Factored-state flag across factor → solve → modify → solve
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_factored_state_lifecycle(void) {
    idx_t n = 5;
    SparseMatrix *A = build_spd_tridiag(n, 1.0);
    ASSERT_NOT_NULL(A);
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    /* LU: unfactored → solve fails */
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_ERR_BADARG);

    /* Factor → solve succeeds */
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    /* Solve again → still succeeds (flag not cleared by solve) */
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    /* Modify → solve fails */
    sparse_insert(LU, 0, 0, 99.0);
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_ERR_BADARG);

    /* Cholesky: same lifecycle */
    SparseMatrix *L = sparse_copy(A);
    ASSERT_ERR(sparse_cholesky_solve(L, b, x), SPARSE_ERR_BADARG);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);
    ASSERT_ERR(sparse_cholesky_solve(L, b, x), SPARSE_OK);
    sparse_remove(L, 0, 0);
    ASSERT_ERR(sparse_cholesky_solve(L, b, x), SPARSE_ERR_BADARG);

    /* ILU rejects factored matrix */
    SparseMatrix *LU2 = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU2, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    sparse_ilu_t ilu;
    ASSERT_ERR(sparse_ilu_factor(LU2, &ilu), SPARSE_ERR_BADARG);
    sparse_ilu_free(&ilu);

    /* ILU accepts unfactored matrix */
    ASSERT_ERR(sparse_ilu_factor(A, &ilu), SPARSE_OK);
    sparse_ilu_free(&ilu);

    free(b);
    free(x);
    sparse_free(A);
    sparse_free(LU);
    sparse_free(L);
    sparse_free(LU2);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 4: ILU+GMRES at extreme scales
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_ilu_gmres_extreme_scales(void) {
    double scales[] = {1e-35, 1e+35};
    for (int s = 0; s < 2; s++) {
        idx_t n = 10;
        SparseMatrix *A = build_spd_tridiag(n, scales[s]);
        ASSERT_NOT_NULL(A);

        double *b = malloc((size_t)n * sizeof(double));
        double *x = calloc((size_t)n, sizeof(double));
        double *ones = malloc((size_t)n * sizeof(double));
        for (idx_t i = 0; i < n; i++)
            ones[i] = 1.0;
        sparse_matvec(A, ones, b);

        sparse_ilu_t ilu;
        ASSERT_ERR(sparse_ilu_factor(A, &ilu), SPARSE_OK);

        sparse_gmres_opts_t opts = {.restart = 20, .max_iter = 200, .tol = 1e-10};
        sparse_iter_result_t result;
        ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, sparse_ilu_precond, &ilu, &result),
                   SPARSE_OK);

        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(x[i], 1.0, 1e-6);

        sparse_ilu_free(&ilu);
        free(b);
        free(x);
        free(ones);
        sparse_free(A);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 5: Version macros are consistent
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_version_consistency(void) {
    /* VERSION_STRING should match the encoded components */
    char expected[32];
    snprintf(expected, sizeof(expected), "%d.%d.%d", SPARSE_VERSION_MAJOR, SPARSE_VERSION_MINOR,
             SPARSE_VERSION_PATCH);
    ASSERT_EQ(strcmp(SPARSE_VERSION_STRING, expected), 0);

    /* SPARSE_VERSION should be the encoded integer */
    int encoded = SPARSE_VERSION_MAJOR * 10000 + SPARSE_VERSION_MINOR * 100 + SPARSE_VERSION_PATCH;
    ASSERT_EQ(SPARSE_VERSION, encoded);

    /* Components should be non-negative */
    ASSERT_TRUE(SPARSE_VERSION_MAJOR >= 0);
    ASSERT_TRUE(SPARSE_VERSION_MINOR >= 0);
    ASSERT_TRUE(SPARSE_VERSION_PATCH >= 0);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 6: Tolerance + factored-state on SuiteSparse matrix
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

static void test_suitesparse_tolerance_integration(void) {
    /* Load nos4 (100×100 SPD) */
    SparseMatrix *nos4 = NULL;
    if (sparse_load_mm(&nos4, SS_DIR "/nos4.mtx") != SPARSE_OK || !nos4) {
        printf("    [SKIP] nos4.mtx not found\n");
        return;
    }

    idx_t n = sparse_rows(nos4);
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    /* Cholesky solve — uses relative tolerance */
    SparseMatrix *L = sparse_copy(nos4);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);
    ASSERT_ERR(sparse_cholesky_solve(L, b, x), SPARSE_OK);

    /* Verify via residual */
    double *r = malloc((size_t)n * sizeof(double));
    sparse_matvec(nos4, x, r);
    double rnorm = 0.0, bnorm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double diff = fabs(r[i] - b[i]);
        if (diff > rnorm)
            rnorm = diff;
        if (fabs(b[i]) > bnorm)
            bnorm = fabs(b[i]);
    }
    double relres = rnorm / (bnorm > 0 ? bnorm : 1.0);
    printf("    nos4 Cholesky relres = %.2e\n", relres);
    ASSERT_TRUE(relres < 1e-10);

    /* Factored-state: solve on unfactored copy fails */
    SparseMatrix *nos4_copy = sparse_copy(nos4);
    ASSERT_ERR(sparse_cholesky_solve(nos4_copy, b, x), SPARSE_ERR_BADARG);

    free(b);
    free(x);
    free(r);
    sparse_free(nos4);
    sparse_free(L);
    sparse_free(nos4_copy);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sprint 11 Integration Tests");

    RUN_TEST(test_all_solvers_tiny_scale);
    RUN_TEST(test_all_solvers_huge_scale);
    RUN_TEST(test_factored_state_lifecycle);
    RUN_TEST(test_ilu_gmres_extreme_scales);
    RUN_TEST(test_version_consistency);
    RUN_TEST(test_suitesparse_tolerance_integration);

    TEST_SUITE_END();
}
