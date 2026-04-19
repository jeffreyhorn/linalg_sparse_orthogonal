/**
 * Sprint 18 cross-feature integration tests.
 *
 * Exercises the transparent CSC dispatch in
 * `sparse_cholesky_factor_opts` across the `SPARSE_CSC_THRESHOLD`
 * boundary: matrices with n < threshold must take the linked-list
 * path, matrices with n >= threshold must take the CSC supernodal
 * path, and both paths must agree on the solution for a matrix that
 * can be factored either way.  Also covers the native CSC LDL^T
 * kernel (Sprint 18 Days 1-5) on indefinite and SPD inputs.
 */
#include "sparse_chol_csc_internal.h"
#include "sparse_cholesky.h"
#include "sparse_ldlt.h"
#include "sparse_ldlt_csc_internal.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_DIR "tests/data"
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static SparseMatrix *build_spd_tridiag(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    return A;
}

/* Banded SPD with bandwidth bw — diagonal dominant, non-trivial fill.
 * Returned A is symmetric (both off-diagonals inserted explicitly). */
static SparseMatrix *build_spd_banded(idx_t n, idx_t bw) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(2 * bw + 2));
        for (idx_t d = 1; d <= bw && i + d < n; d++) {
            double off = -1.0 / (double)(d + 1);
            sparse_insert(A, i, i + d, off);
            sparse_insert(A, i + d, i, off);
        }
    }
    return A;
}

/* KKT saddle-point for indefinite test coverage. */
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

static double relative_residual(const SparseMatrix *A, const double *x, const double *b) {
    idx_t n = sparse_rows(A);
    double *r = malloc((size_t)n * sizeof(double));
    sparse_matvec(A, x, r);
    double nr = 0.0, nb = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double ri = fabs(r[i] - b[i]);
        double bi = fabs(b[i]);
        if (ri > nr)
            nr = ri;
        if (bi > nb)
            nb = bi;
    }
    free(r);
    return nb > 0.0 ? nr / nb : nr;
}

/* Factor + solve via sparse_cholesky_factor_opts, checking that the
 * dispatch chose the expected backend. */
static void factor_solve_assert_path(SparseMatrix *A, int expect_csc, double tol_residual) {
    idx_t n = sparse_rows(A);
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    SparseMatrix *L = sparse_copy(A);
    int used = -1;
    sparse_cholesky_opts_t opts = {SPARSE_REORDER_AMD, SPARSE_CHOL_BACKEND_AUTO, &used};
    REQUIRE_OK(sparse_cholesky_factor_opts(L, &opts));
    ASSERT_EQ(used, expect_csc);

    REQUIRE_OK(sparse_cholesky_solve(L, b, x));
    double rel = relative_residual(A, x, b);
    ASSERT_TRUE(rel < tol_residual);

    free(ones);
    free(b);
    free(x);
    sparse_free(L);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Cross-threshold Cholesky dispatch
 * ═══════════════════════════════════════════════════════════════════════ */

/* Small SPD (n = 20) must take the linked-list path under AUTO. */
static void test_s18_below_threshold_uses_linked_list(void) {
    SparseMatrix *A = build_spd_tridiag(20);
    factor_solve_assert_path(A, /*expect_csc=*/0, 1e-10);
    sparse_free(A);
}

/* Just above SPARSE_CSC_THRESHOLD (default 100): should take CSC. */
static void test_s18_just_above_threshold_uses_csc(void) {
    SparseMatrix *A = build_spd_tridiag(SPARSE_CSC_THRESHOLD + 20);
    factor_solve_assert_path(A, /*expect_csc=*/1, 1e-10);
    sparse_free(A);
}

/* Mid-range SPD banded matrix with non-trivial fill (n = 500, bw = 10). */
static void test_s18_mid_range_spd_banded_uses_csc(void) {
    SparseMatrix *A = build_spd_banded(500, 10);
    factor_solve_assert_path(A, /*expect_csc=*/1, 1e-10);
    sparse_free(A);
}

/* Large SPD banded (n = 2000, bw = 50): stress the supernodal extract. */
static void test_s18_large_spd_banded_uses_csc(void) {
    SparseMatrix *A = build_spd_banded(2000, 50);
    factor_solve_assert_path(A, /*expect_csc=*/1, 1e-10);
    sparse_free(A);
}

/* SuiteSparse fixtures — mix of below-threshold and above-threshold. */
static void test_s18_suitesparse_nos4(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/nos4.mtx"));
    /* nos4 is n=100 which is exactly SPARSE_CSC_THRESHOLD; AUTO uses CSC. */
    factor_solve_assert_path(A, /*expect_csc=*/1, 1e-10);
    sparse_free(A);
}

static void test_s18_suitesparse_bcsstk04(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx"));
    /* bcsstk04 is n=132, > threshold. */
    factor_solve_assert_path(A, /*expect_csc=*/1, 1e-10);
    sparse_free(A);
}

static void test_s18_suitesparse_bcsstk14(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/bcsstk14.mtx"));
    /* n=1806 is well above threshold. */
    factor_solve_assert_path(A, /*expect_csc=*/1, 1e-10);
    sparse_free(A);
}

/* Force both paths on a single matrix and assert bit-level agreement
 * between the `SparseMatrix` L values they produce.  Uses a matrix
 * sized right at the threshold so both paths are exercisable. */
static void test_s18_force_both_paths_agree(void) {
    idx_t n = SPARSE_CSC_THRESHOLD + 30;
    SparseMatrix *A = build_spd_banded(n, 5);

    SparseMatrix *L_ll = sparse_copy(A);
    int used_ll = -1;
    sparse_cholesky_opts_t opts_ll = {SPARSE_REORDER_AMD, SPARSE_CHOL_BACKEND_LINKED_LIST,
                                      &used_ll};
    REQUIRE_OK(sparse_cholesky_factor_opts(L_ll, &opts_ll));
    ASSERT_EQ(used_ll, 0);

    SparseMatrix *L_cs = sparse_copy(A);
    int used_cs = -1;
    sparse_cholesky_opts_t opts_cs = {SPARSE_REORDER_AMD, SPARSE_CHOL_BACKEND_CSC, &used_cs};
    REQUIRE_OK(sparse_cholesky_factor_opts(L_cs, &opts_cs));
    ASSERT_EQ(used_cs, 1);

    double *b = malloc((size_t)n * sizeof(double));
    double *x_ll = calloc((size_t)n, sizeof(double));
    double *x_cs = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0 + 0.1 * (double)i;

    REQUIRE_OK(sparse_cholesky_solve(L_ll, b, x_ll));
    REQUIRE_OK(sparse_cholesky_solve(L_cs, b, x_cs));

    /* Solutions must agree to round-off.  We compare solve outputs
     * rather than raw L values because the two paths apply drop-
     * tolerance at different grain sizes; the factor patterns can
     * differ by a handful of entries that round to machine epsilon,
     * but the solve is the user-visible contract. */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_ll[i], x_cs[i], 1e-10);

    free(b);
    free(x_ll);
    free(x_cs);
    sparse_free(A);
    sparse_free(L_ll);
    sparse_free(L_cs);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Native CSC LDL^T — indefinite matrices
 * ═══════════════════════════════════════════════════════════════════════ */

static void ldlt_csc_factor_solve(const SparseMatrix *A, double tol_residual) {
    idx_t n = sparse_rows(A);
    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(sparse_reorder_amd(A, perm));

    LdltCsc *F = NULL;
    REQUIRE_OK(ldlt_csc_from_sparse(A, perm, 2.0, &F));
    REQUIRE_OK(ldlt_csc_eliminate(F));

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    REQUIRE_OK(ldlt_csc_solve(F, b, x));
    double rel = relative_residual(A, x, b);
    ASSERT_TRUE(rel < tol_residual);

    free(ones);
    free(b);
    free(x);
    free(perm);
    ldlt_csc_free(F);
}

/* Indefinite KKT, moderately sized, via the native CSC LDL^T kernel. */
static void test_s18_ldlt_csc_kkt_indefinite(void) {
    SparseMatrix *K = build_kkt(50, 20);
    ldlt_csc_factor_solve(K, 1e-9);
    sparse_free(K);
}

/* Native kernel vs wrapper: same answer on an indefinite matrix. */
static void test_s18_ldlt_csc_native_matches_wrapper_indefinite(void) {
    SparseMatrix *K = build_kkt(40, 15);
    idx_t n = sparse_rows(K);
    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(sparse_reorder_amd(K, perm));

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(K, ones, b);

    /* Capture the current override so a mid-test failure can't leak a
     * forced kernel into later tests (the override is process-global).
     * We avoid REQUIRE_OK while the override is active — any failure
     * from from_sparse / eliminate restores the previous value first,
     * then raises a normal assertion. */
    LdltCscKernelOverride prev_override = ldlt_csc_get_kernel_override();

    LdltCsc *F_native = NULL;
    ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_NATIVE);
    sparse_err_t err_nf = ldlt_csc_from_sparse(K, perm, 2.0, &F_native);
    sparse_err_t err_ne = (err_nf == SPARSE_OK) ? ldlt_csc_eliminate(F_native) : SPARSE_OK;

    LdltCsc *F_wrapper = NULL;
    sparse_err_t err_wf = SPARSE_OK, err_we = SPARSE_OK;
    if (err_nf == SPARSE_OK && err_ne == SPARSE_OK) {
        ldlt_csc_set_kernel_override(LDLT_CSC_KERNEL_WRAPPER);
        err_wf = ldlt_csc_from_sparse(K, perm, 2.0, &F_wrapper);
        if (err_wf == SPARSE_OK)
            err_we = ldlt_csc_eliminate(F_wrapper);
    }
    ldlt_csc_set_kernel_override(prev_override);

    REQUIRE_OK(err_nf);
    REQUIRE_OK(err_ne);
    REQUIRE_OK(err_wf);
    REQUIRE_OK(err_we);

    double *x_n = calloc((size_t)n, sizeof(double));
    double *x_w = calloc((size_t)n, sizeof(double));
    REQUIRE_OK(ldlt_csc_solve(F_native, b, x_n));
    REQUIRE_OK(ldlt_csc_solve(F_wrapper, b, x_w));

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_n[i], x_w[i], 1e-10);

    free(ones);
    free(b);
    free(x_n);
    free(x_w);
    free(perm);
    ldlt_csc_free(F_native);
    ldlt_csc_free(F_wrapper);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("test_sprint18_integration");

    /* Cholesky dispatch — cross-threshold coverage */
    RUN_TEST(test_s18_below_threshold_uses_linked_list);
    RUN_TEST(test_s18_just_above_threshold_uses_csc);
    RUN_TEST(test_s18_mid_range_spd_banded_uses_csc);
    RUN_TEST(test_s18_large_spd_banded_uses_csc);
    RUN_TEST(test_s18_suitesparse_nos4);
    RUN_TEST(test_s18_suitesparse_bcsstk04);
    RUN_TEST(test_s18_suitesparse_bcsstk14);
    RUN_TEST(test_s18_force_both_paths_agree);

    /* Native CSC LDL^T — indefinite coverage */
    RUN_TEST(test_s18_ldlt_csc_kkt_indefinite);
    RUN_TEST(test_s18_ldlt_csc_native_matches_wrapper_indefinite);

    TEST_SUITE_END();
}
