/**
 * Sprint 12 cross-feature integration tests.
 *
 * Validates that LDL^T factorization with Bunch-Kaufman pivoting integrates
 * correctly with the rest of the library: extreme-scale tolerance, reordering,
 * inertia, LU equivalence, iterative refinement, condition estimation, and
 * SuiteSparse matrices.
 */
#include "sparse_cholesky.h"
#include "sparse_ldlt.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DATA_DIR "tests/data"
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/* Build a KKT matrix K = [H A^T; A 0] of size (nh+nc) x (nh+nc). */
static SparseMatrix *build_kkt(idx_t nh, idx_t nc) {
    idx_t n = nh + nc;
    SparseMatrix *K = sparse_create(n, n);
    for (idx_t i = 0; i < nh; i++) {
        sparse_insert(K, i, i, 4.0);
        if (i > 0) {
            sparse_insert(K, i, i - 1, -1.0);
            sparse_insert(K, i - 1, i, -1.0);
        }
    }
    for (idx_t c = 0; c < nc; c++) {
        idx_t j0 = (c * 2) % nh;
        idx_t j1 = (j0 + 1) % nh;
        sparse_insert(K, nh + c, j0, 1.0);
        sparse_insert(K, j0, nh + c, 1.0);
        sparse_insert(K, nh + c, j1, 1.0);
        sparse_insert(K, j1, nh + c, 1.0);
    }
    return K;
}

/* Build a scaled symmetric indefinite 3x3 matrix. */
static SparseMatrix *build_scaled_indef(double s) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0 * s);
    sparse_insert(A, 0, 1, 1.0 * s);
    sparse_insert(A, 0, 2, -1.0 * s);
    sparse_insert(A, 1, 0, 1.0 * s);
    sparse_insert(A, 1, 1, 3.0 * s);
    sparse_insert(A, 2, 0, -1.0 * s);
    sparse_insert(A, 2, 2, 4.0 * s);
    return A;
}

/* Compute relative residual ||A*x - b|| / ||b||. */
static double relative_residual(const SparseMatrix *A, const double *x, const double *b, idx_t n) {
    double *r = malloc((size_t)n * sizeof(double));
    sparse_matvec(A, x, r);
    double nr = 0.0, nb = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = r[i] - b[i];
        nr += d * d;
        nb += b[i] * b[i];
    }
    free(r);
    return (nb > 0.0) ? sqrt(nr / nb) : sqrt(nr);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 1: Extreme-scale tolerance — factor and solve at 1e-35 and 1e+35
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s12_scaled_tolerance(void) {
    double scales[] = {1e-35, 1e-10, 1.0, 1e10, 1e35};
    int nscales = 5;

    for (int s = 0; s < nscales; s++) {
        SparseMatrix *A = build_scaled_indef(scales[s]);
        sparse_ldlt_t ldlt;
        ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

        double x_exact[] = {1.0, 2.0, 3.0};
        double b[3], x[3];
        sparse_matvec(A, x_exact, b);
        ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);

        for (int i = 0; i < 3; i++)
            ASSERT_NEAR(x[i], x_exact[i], 1e-6);

        sparse_ldlt_free(&ldlt);
        sparse_free(A);
    }
    printf("    scaled tolerance: 5 scales from 1e-35 to 1e+35 OK\n");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 2: KKT pipeline — factor → solve → refine → condest → verify
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s12_kkt_pipeline(void) {
    SparseMatrix *K = build_kkt(20, 8);
    idx_t n = 28;

    /* Factor with AMD */
    sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0};
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor_opts(K, &opts, &ldlt), SPARSE_OK);

    /* Inertia */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 20);
    ASSERT_EQ(neg, 8);

    /* Solve */
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(K, x_exact, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);

    double rr_before = relative_residual(K, x, b, n);

    /* Refine */
    ASSERT_ERR(sparse_ldlt_refine(K, &ldlt, b, x, 5, 1e-15), SPARSE_OK);
    double rr_after = relative_residual(K, x, b, n);
    printf("    KKT pipeline: relres before=%.3e, after=%.3e\n", rr_before, rr_after);
    ASSERT_TRUE(rr_after <= rr_before);
    ASSERT_TRUE(rr_after < 1e-12);

    /* Condest */
    double cond;
    ASSERT_ERR(sparse_ldlt_condest(K, &ldlt, &cond), SPARSE_OK);
    ASSERT_TRUE(cond > 0.0);
    printf("    KKT condest: %.2f\n", cond);

    free(x_exact);
    free(b);
    free(x);
    sparse_ldlt_free(&ldlt);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 3: Reordering equivalence — NONE, AMD, RCM all produce same x
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s12_reorder_equivalence(void) {
    SparseMatrix *K = build_kkt(14, 6);
    idx_t n = 20;

    double *b = malloc((size_t)n * sizeof(double));
    double *x_none = malloc((size_t)n * sizeof(double));
    double *x_amd = malloc((size_t)n * sizeof(double));
    double *x_rcm = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    /* Factor with NONE */
    sparse_ldlt_t ldlt_none;
    ASSERT_ERR(sparse_ldlt_factor(K, &ldlt_none), SPARSE_OK);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt_none, b, x_none), SPARSE_OK);

    /* Factor with AMD */
    sparse_ldlt_opts_t opts_amd = {SPARSE_REORDER_AMD, 0.0};
    sparse_ldlt_t ldlt_amd;
    ASSERT_ERR(sparse_ldlt_factor_opts(K, &opts_amd, &ldlt_amd), SPARSE_OK);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt_amd, b, x_amd), SPARSE_OK);

    /* Factor with RCM */
    sparse_ldlt_opts_t opts_rcm = {SPARSE_REORDER_RCM, 0.0};
    sparse_ldlt_t ldlt_rcm;
    ASSERT_ERR(sparse_ldlt_factor_opts(K, &opts_rcm, &ldlt_rcm), SPARSE_OK);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt_rcm, b, x_rcm), SPARSE_OK);

    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_none[i], x_amd[i], 1e-8);
        ASSERT_NEAR(x_none[i], x_rcm[i], 1e-8);
    }
    printf("    reorder equivalence: NONE/AMD/RCM agree to 1e-8\n");

    free(b);
    free(x_none);
    free(x_amd);
    free(x_rcm);
    sparse_ldlt_free(&ldlt_none);
    sparse_ldlt_free(&ldlt_amd);
    sparse_ldlt_free(&ldlt_rcm);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 4: Inertia matches known eigenstructure
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s12_inertia_eigenstructure(void) {
    /* Diagonal with known eigenvalues: 5 positive, 3 negative */
    idx_t n = 8;
    double eigs[] = {10.0, 5.0, 3.0, 1.0, 0.5, -0.5, -2.0, -7.0};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, eigs[i]);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 5);
    ASSERT_EQ(neg, 3);
    ASSERT_EQ(zero, 0);

    /* D should capture eigenvalues exactly */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(ldlt.D[i], eigs[i], fabs(eigs[i]) * 1e-14);

    printf("    inertia: (5+, 3-, 0) matches diagonal eigenvalues\n");

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 5: LDL^T vs LU equivalence on symmetric system
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s12_ldlt_vs_lu(void) {
    SparseMatrix *K = build_kkt(10, 4);
    idx_t n = 14;

    /* LDL^T */
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(K, &ldlt), SPARSE_OK);

    double *b = malloc((size_t)n * sizeof(double));
    double *x_ldlt = malloc((size_t)n * sizeof(double));
    double *x_lu = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x_ldlt), SPARSE_OK);

    /* LU */
    SparseMatrix *Kcopy = sparse_copy(K);
    ASSERT_ERR(sparse_lu_factor(Kcopy, SPARSE_PIVOT_PARTIAL, 1e-14), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(Kcopy, b, x_lu), SPARSE_OK);

    double max_diff = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = fabs(x_ldlt[i] - x_lu[i]);
        if (d > max_diff)
            max_diff = d;
    }
    printf("    LDL^T vs LU: max|diff| = %.3e\n", max_diff);
    ASSERT_TRUE(max_diff < 1e-8);

    /* Fill-in comparison */
    idx_t nnz_ldlt = sparse_nnz(ldlt.L);
    idx_t nnz_lu = sparse_nnz(Kcopy);
    printf("    fill-in: nnz(L_ldlt)=%d, nnz(LU)=%d, ratio=%.2f\n", (int)nnz_ldlt, (int)nnz_lu,
           (double)nnz_ldlt / (double)nnz_lu);
    ASSERT_TRUE(nnz_ldlt <= nnz_lu);

    free(b);
    free(x_ldlt);
    free(x_lu);
    sparse_ldlt_free(&ldlt);
    sparse_free(Kcopy);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 6: Unfactored struct → solve fails
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s12_unfactored_solve_fails(void) {
    sparse_ldlt_t ldlt;
    memset(&ldlt, 0, sizeof(ldlt));
    double b = 1.0, x = 0.0;

    /* Zeroed struct (n == 0) is a valid empty factorization */
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, &b, &x), SPARSE_OK);
    ASSERT_ERR(sparse_ldlt_refine(NULL, &ldlt, &b, &x, 3, 1e-12), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_ldlt_condest(NULL, &ldlt, &x), SPARSE_ERR_NULL);

    /* n == 0 early-returns OK regardless of A dimensions */
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 1.0);
    ASSERT_ERR(sparse_ldlt_refine(A, &ldlt, &b, &x, 3, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_ldlt_condest(A, &ldlt, &x), SPARSE_OK);

    /* But n > 0 with NULL internal pointers must return BADARG */
    ldlt.n = 1;
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, &b, &x), SPARSE_ERR_BADARG);
    ASSERT_ERR(sparse_ldlt_refine(A, &ldlt, &b, &x, 3, 1e-12), SPARSE_ERR_BADARG);
    ASSERT_ERR(sparse_ldlt_condest(A, &ldlt, &x), SPARSE_ERR_BADARG);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 7: LDL^T vs Cholesky on SuiteSparse SPD matrix
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s12_suitesparse_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    if (lerr != SPARSE_OK || !A) {
        printf("    skip: nos4 not found\n");
        return;
    }

    idx_t n = sparse_rows(A);

    /* LDL^T with AMD */
    sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0};
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor_opts(A, &opts, &ldlt), SPARSE_OK);

    /* Inertia: SPD → all positive */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, n);
    ASSERT_EQ(neg, 0);

    /* Solve + refine */
    double *ones = calloc((size_t)n, sizeof(double));
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    ASSERT_ERR(sparse_ldlt_refine(A, &ldlt, b, x, 3, 1e-15), SPARSE_OK);

    double rr = relative_residual(A, x, b, n);
    printf("    nos4 (n=%d, AMD): relres=%.3e, nnz(L)=%d\n", (int)n, rr, (int)sparse_nnz(ldlt.L));
    ASSERT_TRUE(rr < 1e-12);

    free(ones);
    free(b);
    free(x);
    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 8: Large KKT with full pipeline
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s12_large_kkt_pipeline(void) {
    SparseMatrix *K = build_kkt(200, 80);
    idx_t n = 280;

    sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0};
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor_opts(K, &opts, &ldlt), SPARSE_OK);

    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 200);
    ASSERT_EQ(neg, 80);

    double *ones = calloc((size_t)n, sizeof(double));
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(K, ones, b);

    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    ASSERT_ERR(sparse_ldlt_refine(K, &ldlt, b, x, 5, 1e-15), SPARSE_OK);

    double rr = relative_residual(K, x, b, n);
    printf("    KKT 280x280 (AMD): relres=%.3e, nnz(L)=%d\n", rr, (int)sparse_nnz(ldlt.L));
    ASSERT_TRUE(rr < 1e-12);

    /* Condest should be finite and positive */
    double cond;
    ASSERT_ERR(sparse_ldlt_condest(K, &ldlt, &cond), SPARSE_OK);
    ASSERT_TRUE(cond > 0.0 && cond < 1e15);
    printf("    KKT 280x280 condest: %.2f\n", cond);

    free(ones);
    free(b);
    free(x);
    sparse_ldlt_free(&ldlt);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sprint 12 Integration Tests (LDL^T)");

    RUN_TEST(test_s12_scaled_tolerance);
    RUN_TEST(test_s12_kkt_pipeline);
    RUN_TEST(test_s12_reorder_equivalence);
    RUN_TEST(test_s12_inertia_eigenstructure);
    RUN_TEST(test_s12_ldlt_vs_lu);
    RUN_TEST(test_s12_unfactored_solve_fails);
    RUN_TEST(test_s12_suitesparse_nos4);
    RUN_TEST(test_s12_large_kkt_pipeline);

    TEST_SUITE_END();
}
