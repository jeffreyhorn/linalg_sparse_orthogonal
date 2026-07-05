#include "sparse_analysis.h"
#include "sparse_cholesky.h"
#include "sparse_csr.h"
#include "sparse_eigs.h"
#include "sparse_iterative.h"
#include "sparse_ldlt.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include "test_integration_fixtures.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 1: Load MM -> factor -> solve -> check residual -> save result
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_load_factor_solve_save(void) {
    /* Load the tridiagonal matrix */
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, DATA_DIR "/tridiagonal_20.mtx"), SPARSE_OK);
    ASSERT_NOT_NULL(A);

    idx_t n = sparse_rows(A);
    ASSERT_EQ(n, 20);

    /* Set b = A * [1, 1, ..., 1] so exact solution is x = [1, ..., 1] */
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;
    sparse_matvec(A, x_exact, b);

    /* Factor a copy */
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    /* Solve */
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    /* Residual: r = b - A*x */
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - r[i];
    double res = vec_norminf(r, n);
    ASSERT_TRUE(res < 1e-12);

    /* Solution accuracy */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-12);

    /* Save the solved result to a temp file and reload */
    ASSERT_ERR(sparse_save_mm(A, tf_tmp("integ_tridiag.mtx")), SPARSE_OK);
    SparseMatrix *A2 = NULL;
    ASSERT_ERR(sparse_load_mm(&A2, tf_tmp("integ_tridiag.mtx")), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), sparse_nnz(A2));
    ASSERT_EQ(sparse_rows(A2), n);

    sparse_free(A2);
    free(x_exact);
    free(b);
    free(x);
    free(r);
    sparse_free(LU);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 2: Create -> copy -> factor copy -> solve -> refine -> verify
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_create_copy_factor_refine(void) {
    /* Build a 10x10 diag-dominant matrix programmatically */
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -1.0);
        if (i > 1)
            sparse_insert(A, i, i - 2, -0.5);
        if (i < n - 2)
            sparse_insert(A, i, i + 2, -0.5);
    }

    /* RHS: b = A * [1, 2, 3, ..., n] */
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    /* Copy and factor the copy (preserving A for residual) */
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);

    /* Original should be untouched */
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 10.0, 0.0);

    /* Solve */
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    /* Check pre-refinement residual */
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - r[i];
    double res_before = vec_norminf(r, n);

    /* Refine */
    ASSERT_ERR(sparse_lu_refine(A, LU, b, x, 5, 1e-15), SPARSE_OK);

    /* Check post-refinement residual */
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - r[i];
    double res_after = vec_norminf(r, n);

    ASSERT_TRUE(res_after <= res_before + 1e-15);
    ASSERT_TRUE(res_after < 1e-13);

    /* Solution accuracy */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    free(x_exact);
    free(b);
    free(x);
    free(r);
    sparse_free(LU);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 3: Multiple solves with same factorization
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_multiple_rhs_same_factorization(void) {
    /* Load symmetric matrix */
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, DATA_DIR "/symmetric_4.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A);

    /* Factor once */
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);

    /* Solve with 3 different RHS vectors */
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));

    for (int rhs = 0; rhs < 3; rhs++) {
        /* b = A * e_rhs (unit vector) */
        vec_zero(b, n);
        for (idx_t i = 0; i < n; i++) {
            double col_val = sparse_get_phys(A, i, (idx_t)rhs);
            b[i] = col_val;
        }

        ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

        /* Residual check: r = b - A*x */
        sparse_matvec(A, x, r);
        for (idx_t i = 0; i < n; i++)
            r[i] = b[i] - r[i];
        double res = vec_norminf(r, n);
        ASSERT_TRUE(res < 1e-12);
    }

    free(b);
    free(x);
    free(r);
    sparse_free(LU);
    sparse_free(A);
}

static void test_reset_perms_invalidates_permuted_lu_shell(void) {
    SparseMatrix *A = sparse_create(2, 2);
    ASSERT_NOT_NULL(A);
    ASSERT_ERR(sparse_insert(A, 0, 1, 1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(A, 1, 0, 1.0), SPARSE_OK);
    ASSERT_ERR(sparse_insert(A, 1, 1, 3.0), SPARSE_OK);

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    const idx_t *rp_before = sparse_row_perm(LU);
    ASSERT_TRUE(rp_before[0] != 0 || rp_before[1] != 1);

    double b[2] = {1.0, 4.0};
    double x[2];
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    ASSERT_ERR(sparse_reset_perms(LU), SPARSE_OK);

    const idx_t *rp = sparse_row_perm(LU);
    const idx_t *irp = sparse_inv_row_perm(LU);
    const idx_t *cp = sparse_col_perm(LU);
    const idx_t *icp = sparse_inv_col_perm(LU);
    for (idx_t i = 0; i < 2; i++) {
        ASSERT_EQ(rp[i], i);
        ASSERT_EQ(irp[i], i);
        ASSERT_EQ(cp[i], i);
        ASSERT_EQ(icp[i], i);
    }
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_ERR_BADARG);

    sparse_free(LU);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 4: Round-trip: create -> save -> load -> compare
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_full_roundtrip(void) {
    /* Create a matrix with varied structure */
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(i + 1) * 10.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -1.5);
        if (i > 0)
            sparse_insert(A, i, i - 1, 2.3);
    }
    /* Add a few scattered off-diagonals */
    sparse_insert(A, 0, n - 1, 0.01);
    sparse_insert(A, n - 1, 0, -0.01);

    idx_t nnz_orig = sparse_nnz(A);

    /* Save */
    ASSERT_ERR(sparse_save_mm(A, tf_tmp("integ_roundtrip.mtx")), SPARSE_OK);

    /* Load */
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_load_mm(&B, tf_tmp("integ_roundtrip.mtx")), SPARSE_OK);
    ASSERT_NOT_NULL(B);

    /* Compare */
    ASSERT_EQ(sparse_rows(B), n);
    ASSERT_EQ(sparse_cols(B), n);
    ASSERT_EQ(sparse_nnz(B), nnz_orig);

    /* Element-by-element comparison */
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            ASSERT_NEAR(sparse_get_phys(A, i, j), sparse_get_phys(B, i, j), 1e-14);

    /* Both should produce the same solution */
    double *b = malloc((size_t)n * sizeof(double));
    double *x_a = malloc((size_t)n * sizeof(double));
    double *x_b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    SparseMatrix *LU_A = sparse_copy(A);
    SparseMatrix *LU_B = sparse_copy(B);
    sparse_lu_factor(LU_A, SPARSE_PIVOT_PARTIAL, 1e-12);
    sparse_lu_factor(LU_B, SPARSE_PIVOT_PARTIAL, 1e-12);
    sparse_lu_solve(LU_A, b, x_a);
    sparse_lu_solve(LU_B, b, x_b);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_a[i], x_b[i], 1e-14);

    free(b);
    free(x_a);
    free(x_b);
    sparse_free(LU_A);
    sparse_free(LU_B);
    sparse_free(B);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 5: Load all reference matrices, factor, solve
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_all_reference_matrices(void) {
    const char *files[] = {
        DATA_DIR "/identity_5.mtx",  DATA_DIR "/diagonal_10.mtx", DATA_DIR "/tridiagonal_20.mtx",
        DATA_DIR "/symmetric_4.mtx", DATA_DIR "/bcsstk01.mtx",    DATA_DIR "/unsymm_5.mtx",
    };
    int nfiles = 6;

    for (int f = 0; f < nfiles; f++) {
        SparseMatrix *A = NULL;
        sparse_err_t err = sparse_load_mm(&A, files[f]);
        ASSERT_ERR(err, SPARSE_OK);
        ASSERT_NOT_NULL(A);

        idx_t n = sparse_rows(A);
        double *x_exact = malloc((size_t)n * sizeof(double));
        double *b = malloc((size_t)n * sizeof(double));
        double *x = malloc((size_t)n * sizeof(double));
        double *r = malloc((size_t)n * sizeof(double));
        for (idx_t i = 0; i < n; i++)
            x_exact[i] = 1.0;
        sparse_matvec(A, x_exact, b);

        SparseMatrix *LU = sparse_copy(A);
        ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
        ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

        /* Relative residual: ||r|| / ||b|| */
        sparse_matvec(A, x, r);
        for (idx_t i = 0; i < n; i++)
            r[i] = b[i] - r[i];
        double res = vec_norminf(r, n);
        double bnorm = vec_norminf(b, n);
        double rel_res = (bnorm > 0) ? res / bnorm : res;
        ASSERT_TRUE(rel_res < 1e-10);

        free(x_exact);
        free(b);
        free(x);
        free(r);
        sparse_free(LU);
        sparse_free(A);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 6: Both pivoting strategies produce same answer
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_both_pivots_agree_integration(void) {
    /* Build a 15x15 matrix with some structure */
    idx_t n = 15;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 20.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -2.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -3.0);
        sparse_insert(A, i, (i + 5) % n, 0.5);
    }

    double *b = malloc((size_t)n * sizeof(double));
    double *x_comp = malloc((size_t)n * sizeof(double));
    double *x_part = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    SparseMatrix *LU1 = sparse_copy(A);
    SparseMatrix *LU2 = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU1, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_factor(LU2, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    sparse_lu_solve(LU1, b, x_comp);
    sparse_lu_solve(LU2, b, x_part);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_comp[i], x_part[i], 1e-10);

    free(b);
    free(x_comp);
    free(x_part);
    sparse_free(LU1);
    sparse_free(LU2);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Workflow 7: Error recovery — handle failures gracefully
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_error_recovery(void) {
    /* Attempt to factor singular matrix, then successfully factor a good one */
    SparseMatrix *bad = sparse_create(3, 3);
    sparse_insert(bad, 0, 0, 1.0);
    /* rows 1 and 2 are all zero — singular */
    sparse_err_t err = sparse_lu_factor(bad, SPARSE_PIVOT_COMPLETE, 1e-12);
    ASSERT_ERR(err, SPARSE_ERR_SINGULAR);
    sparse_free(bad);

    /* Now factor a good matrix — should work fine */
    SparseMatrix *good = sparse_create(3, 3);
    sparse_insert(good, 0, 0, 4.0);
    sparse_insert(good, 0, 1, 1.0);
    sparse_insert(good, 1, 0, 1.0);
    sparse_insert(good, 1, 1, 3.0);
    sparse_insert(good, 2, 2, 2.0);

    SparseMatrix *LU = sparse_copy(good);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    double b[] = {5.0, 4.0, 6.0};
    double x[3];
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    /* Verify */
    double r[3];
    sparse_matvec(good, x, r);
    for (int i = 0; i < 3; i++)
        r[i] -= b[i];
    ASSERT_TRUE(vec_norminf(r, 3) < 1e-14);

    sparse_free(LU);
    sparse_free(good);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 29 Day 6 (Item 4): progress / cancel callback coverage
 *
 * Pins the per-routine contract for `opts.progress_cb` /
 * `opts.progress_user` across LU + Cholesky + LDL^T factor paths:
 *   - emits at least `n` progress events covering [0, n) across the
 *     elimination phase (some emit n+1 if total includes a final
 *     k == n boundary; LDL^T may emit fewer than n for 2x2 pivots
 *     because k advances by 2).
 *   - cancellation semantics are family/path-local:
 *       * LU no-reorder path: bit-identical at step 0
 *       * LU reordered one-shot path: caller matrix preserved via
 *         temporary working copy
 *       * Cholesky no-reorder linked-list path: not bit-identical
 *         because the upper triangle is stripped before the first
 *         emission
 *       * Cholesky reordered one-shot path: caller matrix preserved
 *         via temporary working copy
 *       * LDL^T: input matrix bit-identical because factor writes to
 *         a separate owned factor object
 *   - default-NULL-callback path is bit-identical to Sprint 28.
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_progress_cb_lu_emits(void) {
    const idx_t n = 100;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    integration_progress_counter_t ctx = {.cancel_after_step = -1};
    sparse_lu_opts_t opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_NONE,
        .tol = 1e-12,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    ASSERT_EQ(sparse_lu_factor_opts(A, &opts), SPARSE_OK);
    ASSERT_EQ(ctx.n_calls, n); /* one emission per column k = 0..n-1 */
    ASSERT_EQ(ctx.last_step, n - 1);
    ASSERT_TRUE(ctx.last_phase != NULL);
    ASSERT_TRUE(strcmp(ctx.last_phase, "lu_factor") == 0);
    ASSERT_TRUE(ctx.last_elapsed_s >= 0.0);
    sparse_free(A);
}

static void test_progress_cb_lu_cancel(void) {
    const idx_t n = 100;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    /* Snapshot the original matrix entries to verify bit-identity
     * after cancel-at-step-0. */
    SparseMatrix *A_orig = sparse_copy(A);
    REQUIRE_OK(A_orig ? SPARSE_OK : SPARSE_ERR_ALLOC);

    integration_progress_counter_t ctx = {.cancel_after_step = 0};
    sparse_lu_opts_t opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_NONE,
        .tol = 1e-12,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    ASSERT_EQ(sparse_lu_factor_opts(A, &opts), SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx.n_calls, 1); /* cancelled at the very first emission */

    /* Matrix entries and perms stay unchanged for this no-reorder path,
     * and the cancelled matrix must not be accepted as factored. */
    ASSERT_EQ(sparse_get(A, 0, 0), sparse_get(A_orig, 0, 0));
    for (idx_t i = 0; i < n; i++) {
        ASSERT_TRUE(sparse_get(A, i, i) == sparse_get(A_orig, i, i));
        if (i > 0) {
            ASSERT_TRUE(sparse_get(A, i, i - 1) == sparse_get(A_orig, i, i - 1));
            ASSERT_TRUE(sparse_get(A, i - 1, i) == sparse_get(A_orig, i - 1, i));
        }
    }
    double b_cancel[100];
    double x_cancel[100];
    for (idx_t i = 0; i < n; i++)
        b_cancel[i] = 1.0;
    ASSERT_EQ(sparse_lu_solve(A, b_cancel, x_cancel), SPARSE_ERR_BADARG);

    sparse_free(A);
    sparse_free(A_orig);
}

static void test_progress_cb_lu_cancel_after_reorder_preserves_original_matrix(void) {
    const idx_t n = 100;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    SparseMatrix *A_orig = sparse_copy(A);
    REQUIRE_OK(A_orig ? SPARSE_OK : SPARSE_ERR_ALLOC);

    integration_progress_counter_t ctx = {.cancel_after_step = 0};
    sparse_lu_opts_t opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_AMD,
        .tol = 1e-12,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    ASSERT_EQ(sparse_lu_factor_opts(A, &opts), SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx.n_calls, 1);

    const idx_t *rp = sparse_row_perm(A);
    const idx_t *irp = sparse_inv_row_perm(A);
    const idx_t *cp = sparse_col_perm(A);
    const idx_t *icp = sparse_inv_col_perm(A);
    for (idx_t i = 0; i < n; i++) {
        ASSERT_TRUE(rp[i] == i);
        ASSERT_TRUE(irp[i] == i);
        ASSERT_TRUE(cp[i] == i);
        ASSERT_TRUE(icp[i] == i);
        ASSERT_TRUE(sparse_get(A, i, i) == sparse_get(A_orig, i, i));
        if (i > 0) {
            ASSERT_TRUE(sparse_get(A, i, i - 1) == sparse_get(A_orig, i, i - 1));
            ASSERT_TRUE(sparse_get(A, i - 1, i) == sparse_get(A_orig, i - 1, i));
        }
    }

    double b[100];
    double x[100];
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;
    ASSERT_EQ(sparse_lu_solve(A, b, x), SPARSE_ERR_BADARG);
    sparse_lu_opts_t retry_opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_AMD,
        .tol = 1e-12,
    };
    ASSERT_EQ(sparse_lu_factor_opts(A, &retry_opts), SPARSE_OK);
    ASSERT_EQ(sparse_lu_solve(A, b, x), SPARSE_OK);

    sparse_free(A);
    sparse_free(A_orig);
}

static void
test_lu_refactor_attempt_rejects_existing_reordered_factor_and_preserves_old_factor(void) {
    const idx_t n = 100;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_lu_opts_t factor_opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_AMD,
        .tol = 1e-12,
    };
    ASSERT_EQ(sparse_lu_factor_opts(A, &factor_opts), SPARSE_OK);

    double b_cancel[100];
    double x_before[100];
    double x_after[100];
    for (idx_t i = 0; i < n; i++)
        b_cancel[i] = 1.0;
    ASSERT_EQ(sparse_lu_solve(A, b_cancel, x_before), SPARSE_OK);

    sparse_lu_opts_t retry_opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_NONE,
        .tol = 1e-12,
    };
    ASSERT_EQ(sparse_lu_factor_opts(A, &retry_opts), SPARSE_ERR_BADARG);

    ASSERT_EQ(sparse_lu_solve(A, b_cancel, x_after), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_after[i], x_before[i], 1e-12);

    sparse_free(A);
}

static void test_lu_invalid_reorder_opts_preserve_existing_reordered_factor(void) {
    const idx_t n = 100;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_lu_opts_t factor_opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_AMD,
        .tol = 1e-12,
    };
    ASSERT_EQ(sparse_lu_factor_opts(A, &factor_opts), SPARSE_OK);

    double b[100];
    double x_before[100];
    double x_after[100];
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;
    ASSERT_EQ(sparse_lu_solve(A, b, x_before), SPARSE_OK);

    sparse_lu_opts_t invalid_opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = (sparse_reorder_t)99,
        .tol = 1e-12,
    };
    ASSERT_EQ(sparse_lu_factor_opts(A, &invalid_opts), SPARSE_ERR_BADARG);
    ASSERT_EQ(sparse_lu_solve(A, b, x_after), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_after[i], x_before[i], 1e-12);

    sparse_free(A);
}

static void test_lu_invalid_pivot_opts_preserve_original_matrix_and_allow_retry(void) {
    const idx_t n = 100;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    SparseMatrix *A_orig = sparse_copy(A);
    REQUIRE_OK(A_orig ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_lu_opts_t invalid_opts = {
        .pivot = (sparse_pivot_t)99,
        .reorder = SPARSE_REORDER_AMD,
        .tol = 1e-12,
    };
    ASSERT_EQ(sparse_lu_factor_opts(A, &invalid_opts), SPARSE_ERR_BADARG);

    const idx_t *rp = sparse_row_perm(A);
    const idx_t *irp = sparse_inv_row_perm(A);
    const idx_t *cp = sparse_col_perm(A);
    const idx_t *icp = sparse_inv_col_perm(A);
    for (idx_t i = 0; i < n; i++) {
        ASSERT_TRUE(rp[i] == i);
        ASSERT_TRUE(irp[i] == i);
        ASSERT_TRUE(cp[i] == i);
        ASSERT_TRUE(icp[i] == i);
        ASSERT_TRUE(sparse_get(A, i, i) == sparse_get(A_orig, i, i));
        if (i > 0) {
            ASSERT_TRUE(sparse_get(A, i, i - 1) == sparse_get(A_orig, i, i - 1));
            ASSERT_TRUE(sparse_get(A, i - 1, i) == sparse_get(A_orig, i - 1, i));
        }
    }

    double b[100];
    double x[100];
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;
    ASSERT_EQ(sparse_lu_solve(A, b, x), SPARSE_ERR_BADARG);

    sparse_lu_opts_t retry_opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_AMD,
        .tol = 1e-12,
    };
    ASSERT_EQ(sparse_lu_factor_opts(A, &retry_opts), SPARSE_OK);
    ASSERT_EQ(sparse_lu_solve(A, b, x), SPARSE_OK);

    sparse_free(A);
    sparse_free(A_orig);
}

static void test_progress_cb_cholesky_emits_cancel(void) {
    const idx_t n = 100;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    /* Emit: count all callbacks, force linked-list backend so the
     * per-column scalar emission path runs. */
    integration_progress_counter_t ctx = {.cancel_after_step = -1};
    sparse_cholesky_opts_t opts = {
        .reorder = SPARSE_REORDER_NONE,
        .backend = SPARSE_CHOL_BACKEND_LINKED_LIST,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A, &opts), SPARSE_OK);
    ASSERT_EQ(ctx.n_calls, n);
    ASSERT_TRUE(strcmp(ctx.last_phase, "cholesky_factor") == 0);
    sparse_free(A);

    /* Cancel: rebuild matrix (factor consumed the previous one) and
     * cancel at step=0.  The Cholesky factor strips the upper triangle
     * BEFORE the for-k loop, so cancel-at-step-0 does NOT leave the
     * matrix bit-identical to entry — only the lower triangle is
     * preserved.  The contract is "factor returns SPARSE_ERR_CANCELLED
     * + factored=0" rather than full unmodified-input. */
    A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    integration_progress_counter_t ctx2 = {.cancel_after_step = 0};
    sparse_cholesky_opts_t opts2 = {
        .reorder = SPARSE_REORDER_NONE,
        .backend = SPARSE_CHOL_BACKEND_LINKED_LIST,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx2,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A, &opts2), SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx2.n_calls, 1);
    /* Diagonal preserved: cancellation at step=0 fires before any
     * column-k=0 update writes to L(0, 0). */
    ASSERT_TRUE(sparse_get(A, 0, 0) == 4.0);
    double b_cancel[100];
    double x_cancel[100];
    for (idx_t i = 0; i < n; i++)
        b_cancel[i] = 1.0;
    ASSERT_EQ(sparse_cholesky_solve(A, b_cancel, x_cancel), SPARSE_ERR_BADARG);
    sparse_free(A);
}

static void test_progress_cb_cholesky_csc_emits(void) {
    const idx_t n = 100;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    double b[100];
    double x[100];
    int used_csc = 0;
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    integration_progress_counter_t ctx = {.cancel_after_step = -1};
    sparse_cholesky_opts_t opts = {
        .reorder = SPARSE_REORDER_NONE,
        .backend = SPARSE_CHOL_BACKEND_CSC,
        .used_csc_path = &used_csc,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A, &opts), SPARSE_OK);
    ASSERT_EQ(used_csc, 1);
    ASSERT_EQ(ctx.n_calls, 4);
    ASSERT_EQ(ctx.last_step, 3);
    ASSERT_EQ(ctx.last_total, 4);
    ASSERT_TRUE(strcmp(ctx.last_phase, "cholesky_factor_csc") == 0);
    ASSERT_TRUE(ctx.last_elapsed_s >= 0.0);

    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;
    ASSERT_EQ(sparse_cholesky_solve(A, b, x), SPARSE_OK);

    sparse_free(A);
}

static void test_progress_cb_cholesky_cancel_after_reorder_preserves_original_matrix(void) {
    const idx_t n = 100;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    SparseMatrix *A_orig = sparse_copy(A);
    REQUIRE_OK(A_orig ? SPARSE_OK : SPARSE_ERR_ALLOC);

    integration_progress_counter_t ctx = {.cancel_after_step = 0};
    sparse_cholesky_opts_t opts = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_CHOL_BACKEND_LINKED_LIST,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A, &opts), SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx.n_calls, 1);

    const idx_t *rp = sparse_row_perm(A);
    const idx_t *irp = sparse_inv_row_perm(A);
    const idx_t *cp = sparse_col_perm(A);
    const idx_t *icp = sparse_inv_col_perm(A);
    for (idx_t i = 0; i < n; i++) {
        ASSERT_TRUE(rp[i] == i);
        ASSERT_TRUE(irp[i] == i);
        ASSERT_TRUE(cp[i] == i);
        ASSERT_TRUE(icp[i] == i);
        ASSERT_TRUE(sparse_get(A, i, i) == sparse_get(A_orig, i, i));
        if (i > 0) {
            ASSERT_TRUE(sparse_get(A, i, i - 1) == sparse_get(A_orig, i, i - 1));
            ASSERT_TRUE(sparse_get(A, i - 1, i) == sparse_get(A_orig, i - 1, i));
        }
    }

    double b[100];
    double x[100];
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;
    ASSERT_EQ(sparse_cholesky_solve(A, b, x), SPARSE_ERR_BADARG);

    sparse_cholesky_opts_t retry_opts = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_CHOL_BACKEND_LINKED_LIST,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A, &retry_opts), SPARSE_OK);
    ASSERT_EQ(sparse_cholesky_solve(A, b, x), SPARSE_OK);

    sparse_free(A);
    sparse_free(A_orig);
}

static void test_progress_cb_cholesky_csc_cancel_before_writeback_preserves_original_matrix(void) {
    const idx_t n = 100;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    SparseMatrix *A_orig = NULL;
    double b[100];
    double x[100];
    int used_csc = 0;

    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    A_orig = sparse_copy(A);
    REQUIRE_OK(A_orig ? SPARSE_OK : SPARSE_ERR_ALLOC);

    integration_progress_counter_t ctx = {.cancel_after_step = 3};
    sparse_cholesky_opts_t opts = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_CHOL_BACKEND_CSC,
        .used_csc_path = &used_csc,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A, &opts), SPARSE_ERR_CANCELLED);
    ASSERT_EQ(used_csc, 1);
    ASSERT_EQ(ctx.n_calls, 4);
    ASSERT_EQ(ctx.last_step, 3);
    ASSERT_EQ(ctx.last_total, 4);
    ASSERT_TRUE(strcmp(ctx.last_phase, "cholesky_factor_csc") == 0);

    const idx_t *rp = sparse_row_perm(A);
    const idx_t *irp = sparse_inv_row_perm(A);
    const idx_t *cp = sparse_col_perm(A);
    const idx_t *icp = sparse_inv_col_perm(A);
    for (idx_t i = 0; i < n; i++) {
        ASSERT_TRUE(rp[i] == i);
        ASSERT_TRUE(irp[i] == i);
        ASSERT_TRUE(cp[i] == i);
        ASSERT_TRUE(icp[i] == i);
        ASSERT_TRUE(sparse_get(A, i, i) == sparse_get(A_orig, i, i));
        if (i > 0) {
            ASSERT_TRUE(sparse_get(A, i, i - 1) == sparse_get(A_orig, i, i - 1));
            ASSERT_TRUE(sparse_get(A, i - 1, i) == sparse_get(A_orig, i - 1, i));
        }
    }

    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;
    ASSERT_EQ(sparse_cholesky_solve(A, b, x), SPARSE_ERR_BADARG);

    sparse_cholesky_opts_t retry_opts = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_CHOL_BACKEND_CSC,
        .used_csc_path = NULL,
        .progress_cb = NULL,
        .progress_user = NULL,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A, &retry_opts), SPARSE_OK);
    ASSERT_EQ(sparse_cholesky_solve(A, b, x), SPARSE_OK);

    sparse_free(A);
    sparse_free(A_orig);
}

static void
test_cholesky_refactor_attempt_rejects_existing_reordered_factor_and_preserves_old_factor(void) {
    const idx_t n = 100;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_cholesky_opts_t factor_opts = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_CHOL_BACKEND_LINKED_LIST,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A, &factor_opts), SPARSE_OK);

    double b[100];
    double x_before[100];
    double x_after[100];
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;
    ASSERT_EQ(sparse_cholesky_solve(A, b, x_before), SPARSE_OK);

    sparse_cholesky_opts_t retry_opts = {
        .reorder = SPARSE_REORDER_NONE,
        .backend = SPARSE_CHOL_BACKEND_LINKED_LIST,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A, &retry_opts), SPARSE_ERR_BADARG);
    ASSERT_EQ(sparse_cholesky_solve(A, b, x_after), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_after[i], x_before[i], 1e-12);

    sparse_free(A);
}

static void test_cholesky_reordered_not_spd_preserves_original_matrix(void) {
    SparseMatrix *A = sparse_create(3, 3);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 1, 2, 0.5);
    sparse_insert(A, 2, 1, 0.5);
    sparse_insert(A, 2, 2, 3.0);

    SparseMatrix *A_orig = sparse_copy(A);
    REQUIRE_OK(A_orig ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_cholesky_opts_t opts = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_CHOL_BACKEND_LINKED_LIST,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A, &opts), SPARSE_ERR_NOT_SPD);

    const idx_t *rp = sparse_row_perm(A);
    const idx_t *irp = sparse_inv_row_perm(A);
    const idx_t *cp = sparse_col_perm(A);
    const idx_t *icp = sparse_inv_col_perm(A);
    for (idx_t i = 0; i < 3; i++) {
        ASSERT_TRUE(rp[i] == i);
        ASSERT_TRUE(irp[i] == i);
        ASSERT_TRUE(cp[i] == i);
        ASSERT_TRUE(icp[i] == i);
        for (idx_t j = 0; j < 3; j++)
            ASSERT_TRUE(sparse_get(A, i, j) == sparse_get(A_orig, i, j));
    }

    double b[3] = {1.0, 1.0, 1.0};
    double x[3];
    ASSERT_EQ(sparse_cholesky_solve(A, b, x), SPARSE_ERR_BADARG);

    sparse_free(A);
    sparse_free(A_orig);
}

static void test_cholesky_invalid_backend_preserves_original_matrix_and_allows_retry(void) {
    const idx_t n = 120;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    SparseMatrix *A_orig = sparse_copy(A);
    REQUIRE_OK(A_orig ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_cholesky_opts_t bad_opts = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = (sparse_chol_backend_t)99,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A, &bad_opts), SPARSE_ERR_BADARG);

    const idx_t *rp = sparse_row_perm(A);
    const idx_t *irp = sparse_inv_row_perm(A);
    const idx_t *cp = sparse_col_perm(A);
    const idx_t *icp = sparse_inv_col_perm(A);
    for (idx_t i = 0; i < n; i++) {
        ASSERT_TRUE(rp[i] == i);
        ASSERT_TRUE(irp[i] == i);
        ASSERT_TRUE(cp[i] == i);
        ASSERT_TRUE(icp[i] == i);
        ASSERT_TRUE(sparse_get(A, i, i) == sparse_get(A_orig, i, i));
        if (i > 0) {
            ASSERT_TRUE(sparse_get(A, i, i - 1) == sparse_get(A_orig, i, i - 1));
            ASSERT_TRUE(sparse_get(A, i - 1, i) == sparse_get(A_orig, i - 1, i));
        }
    }

    double b[120];
    double x[120];
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;
    ASSERT_EQ(sparse_cholesky_solve(A, b, x), SPARSE_ERR_BADARG);

    sparse_cholesky_opts_t retry_opts = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_CHOL_BACKEND_CSC,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A, &retry_opts), SPARSE_OK);
    ASSERT_EQ(sparse_cholesky_solve(A, b, x), SPARSE_OK);

    sparse_free(A);
    sparse_free(A_orig);
}

static void test_progress_cb_ldlt_emits_cancel(void) {
    const idx_t n = 100;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    /* Emit: count, force linked-list backend.  LDL^T may use 2x2
     * pivots, so the number of emissions is <= n (each pivot
     * advances k by 1 or 2).  For our diagonally-dominant
     * tridiagonal SPD all pivots are 1x1 and we get exactly n
     * emissions, but the contract is "emissions ≤ n" generically. */
    integration_progress_counter_t ctx = {.cancel_after_step = -1};
    sparse_ldlt_t ldlt = {0};
    sparse_ldlt_opts_t opts = {
        .reorder = SPARSE_REORDER_NONE,
        .tol = 0.0,
        .backend = SPARSE_LDLT_BACKEND_LINKED_LIST,
        .used_csc_path = NULL,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    ASSERT_EQ(sparse_ldlt_factor_opts(A, &opts, &ldlt), SPARSE_OK);
    ASSERT_TRUE(ctx.n_calls > 0);
    ASSERT_TRUE(ctx.n_calls <= n);
    ASSERT_TRUE(strcmp(ctx.last_phase, "ldlt_factor") == 0);
    sparse_ldlt_free(&ldlt);

    /* Cancel: A is untouched by LDL^T (factor writes to a separate
     * ldlt_t struct), so cancel-at-step-0 leaves A bit-identical. */
    SparseMatrix *A_orig = sparse_copy(A);
    REQUIRE_OK(A_orig ? SPARSE_OK : SPARSE_ERR_ALLOC);
    integration_progress_counter_t ctx2 = {.cancel_after_step = 0};
    sparse_ldlt_t ldlt2 = {0};
    sparse_ldlt_opts_t opts2 = {
        .reorder = SPARSE_REORDER_NONE,
        .tol = 0.0,
        .backend = SPARSE_LDLT_BACKEND_LINKED_LIST,
        .used_csc_path = NULL,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx2,
    };
    ASSERT_EQ(sparse_ldlt_factor_opts(A, &opts2, &ldlt2), SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx2.n_calls, 1);
    /* ldlt2 struct freed by the factor; sparse_ldlt_free safe on zeroed remnant. */
    sparse_ldlt_free(&ldlt2);
    /* A unmodified. */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_TRUE(sparse_get(A, i, i) == sparse_get(A_orig, i, i));
    }
    sparse_free(A);
    sparse_free(A_orig);
}

/* Default-NULL-callback bit-identical-to-Sprint-28 contract: verify
 * `opts.progress_cb == NULL` produces the same factorisation result
 * as the no-opts (sparse_lu_factor) entry point. */
static void test_progress_cb_null_default_unchanged(void) {
    const idx_t n = 50;
    SparseMatrix *A1 = integration_build_tridiag_spd(n);
    SparseMatrix *A2 = integration_build_tridiag_spd(n);
    REQUIRE_OK(A1 && A2 ? SPARSE_OK : SPARSE_ERR_ALLOC);

    ASSERT_EQ(sparse_lu_factor(A1, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    sparse_lu_opts_t opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_NONE,
        .tol = 1e-12,
    };
    ASSERT_EQ(sparse_lu_factor_opts(A2, &opts), SPARSE_OK);

    /* Solve A1 x = b and A2 x = b on a known b; results must match. */
    double *b = malloc((size_t)n * sizeof(double));
    double *x1 = malloc((size_t)n * sizeof(double));
    double *x2 = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(b && x1 && x2 ? SPARSE_OK : SPARSE_ERR_ALLOC);
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);
    sparse_lu_solve(A1, b, x1);
    sparse_lu_solve(A2, b, x2);
    for (idx_t i = 0; i < n; i++)
        ASSERT_TRUE(x1[i] == x2[i]); /* bit-identical solution */

    free(b);
    free(x1);
    free(x2);
    sparse_free(A1);
    sparse_free(A2);
}

static void test_cholesky_default_wrapper_matches_default_opts(void) {
    const idx_t n = 50;
    SparseMatrix *A1 = integration_build_tridiag_spd(n);
    SparseMatrix *A2 = integration_build_tridiag_spd(n);
    double *b = NULL;
    double *x1 = NULL;
    double *x2 = NULL;

    REQUIRE_OK(A1 && A2 ? SPARSE_OK : SPARSE_ERR_ALLOC);

    ASSERT_EQ(sparse_cholesky_factor(A1), SPARSE_OK);

    sparse_cholesky_opts_t opts = {
        .reorder = SPARSE_REORDER_NONE,
        .backend = SPARSE_CHOL_BACKEND_AUTO,
        .used_csc_path = NULL,
        .progress_cb = NULL,
        .progress_user = NULL,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A2, &opts), SPARSE_OK);

    b = malloc((size_t)n * sizeof(double));
    x1 = malloc((size_t)n * sizeof(double));
    x2 = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(b && x1 && x2 ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    ASSERT_EQ(sparse_cholesky_solve(A1, b, x1), SPARSE_OK);
    ASSERT_EQ(sparse_cholesky_solve(A2, b, x2), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_TRUE(x1[i] == x2[i]);

    free(b);
    free(x1);
    free(x2);
    sparse_free(A1);
    sparse_free(A2);
}

static void test_ldlt_default_wrapper_matches_default_opts(void) {
    const idx_t n = 50;
    SparseMatrix *A1 = integration_build_tridiag_spd(n);
    SparseMatrix *A2 = integration_build_tridiag_spd(n);
    sparse_ldlt_t ldlt1 = {0};
    sparse_ldlt_t ldlt2 = {0};
    double *b = NULL;
    double *x1 = NULL;
    double *x2 = NULL;

    REQUIRE_OK(A1 && A2 ? SPARSE_OK : SPARSE_ERR_ALLOC);

    ASSERT_EQ(sparse_ldlt_factor(A1, &ldlt1), SPARSE_OK);

    sparse_ldlt_opts_t opts = {
        .reorder = SPARSE_REORDER_NONE,
        .tol = 0.0,
        .backend = SPARSE_LDLT_BACKEND_AUTO,
        .used_csc_path = NULL,
        .progress_cb = NULL,
        .progress_user = NULL,
    };
    ASSERT_EQ(sparse_ldlt_factor_opts(A2, &opts, &ldlt2), SPARSE_OK);

    b = malloc((size_t)n * sizeof(double));
    x1 = malloc((size_t)n * sizeof(double));
    x2 = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(b && x1 && x2 ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    ASSERT_EQ(sparse_ldlt_solve(&ldlt1, b, x1), SPARSE_OK);
    ASSERT_EQ(sparse_ldlt_solve(&ldlt2, b, x2), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_TRUE(x1[i] == x2[i]);

    free(b);
    free(x1);
    free(x2);
    sparse_ldlt_free(&ldlt1);
    sparse_ldlt_free(&ldlt2);
    sparse_free(A1);
    sparse_free(A2);
}

static void test_lu_factor_opts_matches_explicit_analysis_path(void) {
    const idx_t n = 50;
    SparseMatrix *A_opts = integration_build_tridiag_spd(n);
    SparseMatrix *A_analysis = integration_build_tridiag_spd(n);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *b = NULL;
    double *x_opts = NULL;
    double *x_analysis = NULL;

    REQUIRE_OK(A_opts && A_analysis ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_lu_opts_t lu_opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_AMD,
        .tol = 1e-12,
    };
    ASSERT_EQ(sparse_lu_factor_opts(A_opts, &lu_opts), SPARSE_OK);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_LU,
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_analyze(A_analysis, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_analysis, &analysis, &factors), SPARSE_OK);

    b = malloc((size_t)n * sizeof(double));
    x_opts = malloc((size_t)n * sizeof(double));
    x_analysis = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(b && x_opts && x_analysis ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    ASSERT_EQ(sparse_lu_solve(A_opts, b, x_opts), SPARSE_OK);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b, x_analysis), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_opts[i], x_analysis[i], 1e-12);

    free(b);
    free(x_opts);
    free(x_analysis);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_opts);
    sparse_free(A_analysis);
}

static void test_create_from_csr_enters_one_shot_lu_workflow(void) {
    SparseMatrix *A_src = integration_build_unsym_4x4();
    SparseMatrix *A_ctor = integration_build_from_csr_constructor(A_src);
    double x_exact[4] = {1.0, -2.0, 0.5, 3.0};
    double b[4] = {0.0, 0.0, 0.0, 0.0};
    double x[4] = {0.0, 0.0, 0.0, 0.0};
    double r[4] = {0.0, 0.0, 0.0, 0.0};
    sparse_lu_opts_t opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_AMD,
        .tol = 1e-12,
    };

    REQUIRE_OK(A_src && A_ctor ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_matvec(A_src, x_exact, b);

    ASSERT_EQ(sparse_lu_factor_opts(A_ctor, &opts), SPARSE_OK);
    ASSERT_EQ(sparse_lu_solve(A_ctor, b, x), SPARSE_OK);

    sparse_matvec(A_src, x, r);
    for (idx_t i = 0; i < 4; i++) {
        r[i] = b[i] - r[i];
        ASSERT_NEAR(x[i], x_exact[i], 1e-12);
    }
    ASSERT_TRUE(vec_norminf(r, 4) < 1e-12);

    sparse_free(A_ctor);
    sparse_free(A_src);
}

static void test_cholesky_factor_opts_matches_explicit_analysis_path(void) {
    const idx_t n = (idx_t)(SPARSE_CSC_THRESHOLD + 100);
    SparseMatrix *A_opts = integration_build_tridiag_spd(n);
    SparseMatrix *A_analysis = integration_build_tridiag_spd(n);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    int used_csc_path = 0;
    double *b = NULL;
    double *x_opts = NULL;
    double *x_analysis = NULL;

    REQUIRE_OK(A_opts && A_analysis ? SPARSE_OK : SPARSE_ERR_ALLOC);
    ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD);

    sparse_cholesky_opts_t chol_opts = {
        .reorder = SPARSE_REORDER_AMD,
        .used_csc_path = &used_csc_path,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A_opts, &chol_opts), SPARSE_OK);
    ASSERT_EQ(used_csc_path, 1);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_analyze(A_analysis, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_analysis, &analysis, &factors), SPARSE_OK);

    b = malloc((size_t)n * sizeof(double));
    x_opts = malloc((size_t)n * sizeof(double));
    x_analysis = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(b && x_opts && x_analysis ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    ASSERT_EQ(sparse_cholesky_solve(A_opts, b, x_opts), SPARSE_OK);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b, x_analysis), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_opts[i], x_analysis[i], 1e-12);

    free(b);
    free(x_opts);
    free(x_analysis);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_opts);
    sparse_free(A_analysis);
}

static void test_ldlt_factor_opts_matches_explicit_analysis_path(void) {
    const idx_t n = 200;
    SparseMatrix *A_opts = integration_build_tridiag_spd(n);
    SparseMatrix *A_analysis = integration_build_tridiag_spd(n);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    sparse_ldlt_t ldlt = {0};
    double *b = NULL;
    double *x_opts = NULL;
    double *x_analysis = NULL;

    REQUIRE_OK(A_opts && A_analysis ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_ldlt_opts_t ldlt_opts = {
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_ldlt_factor_opts(A_opts, &ldlt_opts, &ldlt), SPARSE_OK);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_LDLT,
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_analyze(A_analysis, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_analysis, &analysis, &factors), SPARSE_OK);

    b = malloc((size_t)n * sizeof(double));
    x_opts = malloc((size_t)n * sizeof(double));
    x_analysis = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(b && x_opts && x_analysis ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    ASSERT_EQ(sparse_ldlt_solve(&ldlt, b, x_opts), SPARSE_OK);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b, x_analysis), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_opts[i], x_analysis[i], 1e-12);
    for (idx_t i = 0; i < n; i++) {
        ASSERT_EQ(ldlt.perm[i], analysis.perm[i]);
        ASSERT_EQ(factors.ldlt_perm[i], analysis.perm[i]);
    }

    free(b);
    free(x_opts);
    free(x_analysis);
    sparse_ldlt_free(&ldlt);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_opts);
    sparse_free(A_analysis);
}

static void test_ldlt_factor_opts_matches_explicit_analysis_path_indefinite_kkt(void) {
    SparseMatrix *A_opts = integration_build_kkt(/*n_top=*/140, /*n_bot=*/10);
    SparseMatrix *A_analysis = integration_build_kkt(/*n_top=*/140, /*n_bot=*/10);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    sparse_ldlt_t ldlt = {0};
    double *x_exact = NULL;
    double *b = NULL;
    double *x_opts = NULL;
    double *x_analysis = NULL;
    int used_csc = -1;
    const idx_t n = 150;

    REQUIRE_OK(A_opts && A_analysis ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_ldlt_opts_t ldlt_opts = {
        .reorder = SPARSE_REORDER_NONE,
        .backend = SPARSE_LDLT_BACKEND_AUTO,
        .used_csc_path = &used_csc,
    };
    ASSERT_EQ(sparse_ldlt_factor_opts(A_opts, &ldlt_opts, &ldlt), SPARSE_OK);
    ASSERT_EQ(used_csc, 1);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_LDLT,
        .reorder = SPARSE_REORDER_NONE,
    };
    ASSERT_EQ(sparse_analyze(A_analysis, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_analysis, &analysis, &factors), SPARSE_OK);

    x_exact = malloc((size_t)n * sizeof(double));
    b = malloc((size_t)n * sizeof(double));
    x_opts = malloc((size_t)n * sizeof(double));
    x_analysis = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b && x_opts && x_analysis ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;
    sparse_matvec(A_opts, x_exact, b);

    ASSERT_EQ(sparse_ldlt_solve(&ldlt, b, x_opts), SPARSE_OK);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b, x_analysis), SPARSE_OK);
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_opts[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_analysis[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_opts[i], x_analysis[i], 1e-10);
    }

    free(x_exact);
    free(b);
    free(x_opts);
    free(x_analysis);
    sparse_ldlt_free(&ldlt);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_opts);
    sparse_free(A_analysis);
}

static void test_public_lifecycle_ldlt_refactor_same_pattern_indefinite_kkt(void) {
    SparseMatrix *A1 = integration_build_kkt(/*n_top=*/140, /*n_bot=*/10);
    SparseMatrix *A2 = integration_build_kkt(/*n_top=*/140, /*n_bot=*/10);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact = NULL;
    double *b1 = NULL;
    double *b2 = NULL;
    double *x1 = NULL;
    double *x2 = NULL;
    const idx_t n = 150;

    REQUIRE_OK(A1 && A2 ? SPARSE_OK : SPARSE_ERR_ALLOC);
    integration_perturb_kkt_values_in_place(A2, /*n_top=*/140, /*n_bot=*/10, /*scale=*/0.2);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_LDLT,
        .reorder = SPARSE_REORDER_NONE,
    };
    ASSERT_EQ(sparse_analyze(A1, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A1, &analysis, &factors), SPARSE_OK);
    ASSERT_NOT_NULL(factors.ldlt_perm);
    ASSERT_NOT_NULL(factors.pivot_size);

    x_exact = malloc((size_t)n * sizeof(double));
    b1 = malloc((size_t)n * sizeof(double));
    b2 = malloc((size_t)n * sizeof(double));
    x1 = malloc((size_t)n * sizeof(double));
    x2 = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b1 && b2 && x1 && x2 ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;

    sparse_matvec(A1, x_exact, b1);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b1, x1), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x1[i], x_exact[i], 1e-10);

    sparse_matvec(A2, x_exact, b2);
    ASSERT_EQ(sparse_refactor_numeric(A2, &analysis, &factors), SPARSE_OK);
    ASSERT_NOT_NULL(factors.ldlt_perm);
    ASSERT_NOT_NULL(factors.pivot_size);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b2, x2), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x2[i], x_exact[i], 1e-10);

    free(x_exact);
    free(b1);
    free(b2);
    free(x1);
    free(x2);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A1);
    sparse_free(A2);
}

static void test_public_lifecycle_ldlt_refactor_same_pattern_indefinite_kkt_amd(void) {
    SparseMatrix *A1 = integration_build_kkt(/*n_top=*/140, /*n_bot=*/10);
    SparseMatrix *A2 = integration_build_kkt(/*n_top=*/140, /*n_bot=*/10);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact = NULL;
    double *b1 = NULL;
    double *b2 = NULL;
    double *x1 = NULL;
    double *x2 = NULL;
    const idx_t n = 150;

    REQUIRE_OK(A1 && A2 ? SPARSE_OK : SPARSE_ERR_ALLOC);
    integration_perturb_kkt_values_in_place(A2, /*n_top=*/140, /*n_bot=*/10, /*scale=*/0.2);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_LDLT,
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_analyze(A1, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A1, &analysis, &factors), SPARSE_OK);
    ASSERT_NOT_NULL(factors.ldlt_perm);
    ASSERT_NOT_NULL(factors.pivot_size);

    x_exact = malloc((size_t)n * sizeof(double));
    b1 = malloc((size_t)n * sizeof(double));
    b2 = malloc((size_t)n * sizeof(double));
    x1 = malloc((size_t)n * sizeof(double));
    x2 = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b1 && b2 && x1 && x2 ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;

    sparse_matvec(A1, x_exact, b1);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b1, x1), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x1[i], x_exact[i], 1e-10);

    sparse_matvec(A2, x_exact, b2);
    ASSERT_EQ(sparse_refactor_numeric(A2, &analysis, &factors), SPARSE_OK);
    ASSERT_NOT_NULL(factors.ldlt_perm);
    ASSERT_NOT_NULL(factors.pivot_size);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b2, x2), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x2[i], x_exact[i], 1e-10);

    free(x_exact);
    free(b1);
    free(b2);
    free(x1);
    free(x2);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A1);
    sparse_free(A2);
}

static void
test_public_lifecycle_ldlt_refactor_rejects_nnz_drift_and_preserves_old_factors_amd(void) {
    SparseMatrix *A_good = integration_build_kkt(/*n_top=*/140, /*n_bot=*/10);
    SparseMatrix *A_bad = NULL;
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact = NULL;
    double *b = NULL;
    double *x = NULL;
    const idx_t n_top = 140;
    const idx_t n_bot = 10;
    const idx_t n = n_top + n_bot;

    REQUIRE_OK(A_good ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_LDLT,
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_analyze(A_good, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_good, &analysis, &factors), SPARSE_OK);
    ASSERT_NOT_NULL(factors.ldlt_perm);
    ASSERT_NOT_NULL(factors.pivot_size);

    x_exact = malloc((size_t)n * sizeof(double));
    b = malloc((size_t)n * sizeof(double));
    x = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b && x ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;
    sparse_matvec(A_good, x_exact, b);

    A_bad = sparse_copy(A_good);
    REQUIRE_OK(A_bad ? SPARSE_OK : SPARSE_ERR_ALLOC);
    ASSERT_EQ(sparse_set(A_bad, 0, n_top, 0.0), SPARSE_OK);
    ASSERT_EQ(sparse_set(A_bad, n_top, 0, 0.0), SPARSE_OK);

    ASSERT_EQ(sparse_refactor_numeric(A_bad, &analysis, &factors), SPARSE_ERR_BADARG);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b, x), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    free(x_exact);
    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_bad);
    sparse_free(A_good);
}

static void test_public_lifecycle_solve_rejects_zeroed_factors(void) {
    const idx_t n = 50;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *b = NULL;
    double *x = NULL;

    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_NONE,
    };
    ASSERT_EQ(sparse_analyze(A, &analysis_opts, &analysis), SPARSE_OK);

    b = malloc((size_t)n * sizeof(double));
    x = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(b && x ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b, x), SPARSE_ERR_BADARG);

    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);
}

static void test_public_lifecycle_solve_rejects_mismatched_analysis_and_preserves_factors(void) {
    const idx_t n = 4;
    SparseMatrix *A_good = integration_build_tridiag_spd(n);
    SparseMatrix *A_lu = integration_build_unsym_4x4();
    SparseMatrix *A_other_n = integration_build_tridiag_spd(5);
    sparse_analysis_t good_analysis = {0};
    sparse_analysis_t lu_analysis = {0};
    sparse_analysis_t other_n_analysis = {0};
    sparse_factors_t factors = {0};
    double x_exact[4] = {1.0, 1.0, 1.0, 1.0};
    double b[4];
    double x[4];

    REQUIRE_OK(A_good && A_lu && A_other_n ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_analysis_opts_t good_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_analysis_opts_t lu_opts = {
        .factor_type = SPARSE_FACTOR_LU,
        .reorder = SPARSE_REORDER_NONE,
    };
    sparse_analysis_opts_t other_n_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_NONE,
    };

    ASSERT_EQ(sparse_analyze(A_good, &good_opts, &good_analysis), SPARSE_OK);
    ASSERT_EQ(sparse_analyze(A_lu, &lu_opts, &lu_analysis), SPARSE_OK);
    ASSERT_EQ(sparse_analyze(A_other_n, &other_n_opts, &other_n_analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_good, &good_analysis, &factors), SPARSE_OK);

    sparse_matvec(A_good, x_exact, b);

    ASSERT_EQ(sparse_factor_solve(&factors, &lu_analysis, b, x), SPARSE_ERR_BADARG);
    ASSERT_EQ(sparse_factor_solve(&factors, &other_n_analysis, b, x), SPARSE_ERR_SHAPE);

    ASSERT_EQ(sparse_factor_solve(&factors, &good_analysis, b, x), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-12);

    sparse_factor_free(&factors);
    sparse_analysis_free(&good_analysis);
    sparse_analysis_free(&lu_analysis);
    sparse_analysis_free(&other_n_analysis);
    sparse_free(A_good);
    sparse_free(A_lu);
    sparse_free(A_other_n);
}

static void test_public_lifecycle_repeated_solve_and_free_zeroed(void) {
    const idx_t n = 40;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact1 = NULL;
    double *x_exact2 = NULL;
    double *b1 = NULL;
    double *b2 = NULL;
    double *x1 = NULL;
    double *x2 = NULL;

    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    REQUIRE_OK(sparse_analyze(A, &analysis_opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A, &analysis, &factors));

    x_exact1 = malloc((size_t)n * sizeof(double));
    x_exact2 = malloc((size_t)n * sizeof(double));
    b1 = malloc((size_t)n * sizeof(double));
    b2 = malloc((size_t)n * sizeof(double));
    x1 = malloc((size_t)n * sizeof(double));
    x2 = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact1 && x_exact2 && b1 && b2 && x1 && x2 ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++) {
        x_exact1[i] = 1.0 + 0.25 * (double)i;
        x_exact2[i] = (i % 2 == 0) ? 2.0 : -1.0;
    }
    sparse_matvec(A, x_exact1, b1);
    sparse_matvec(A, x_exact2, b2);

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b1, x1));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b2, x2));
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x1[i], x_exact1[i], 1e-12);
        ASSERT_NEAR(x2[i], x_exact2[i], 1e-12);
    }

    sparse_factor_free(&factors);
    ASSERT_TRUE(factors.F == NULL);
    ASSERT_TRUE(factors.D == NULL);
    ASSERT_TRUE(factors.D_offdiag == NULL);
    ASSERT_TRUE(factors.pivot_size == NULL);
    ASSERT_TRUE(factors.ldlt_perm == NULL);
    ASSERT_EQ(factors.n, 0);
    ASSERT_TRUE(factors.factor_norm == 0.0);

    sparse_analysis_free(&analysis);
    ASSERT_TRUE(analysis.perm == NULL);
    ASSERT_TRUE(analysis.etree == NULL);
    ASSERT_TRUE(analysis.postorder == NULL);
    ASSERT_TRUE(analysis.sym_L.col_ptr == NULL);
    ASSERT_TRUE(analysis.sym_L.row_idx == NULL);
    ASSERT_TRUE(analysis.sym_U.col_ptr == NULL);
    ASSERT_TRUE(analysis.sym_U.row_idx == NULL);
    ASSERT_EQ(analysis.n, 0);
    ASSERT_EQ(analysis.source_nnz, 0);
    ASSERT_TRUE(analysis.analysis_norm == 0.0);

    /* The lifecycle free entry points are documented as safe on zeroed state. */
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);

    free(x_exact1);
    free(x_exact2);
    free(b1);
    free(b2);
    free(x1);
    free(x2);
    sparse_free(A);
}

static void test_public_lifecycle_refactor_accepts_zeroed_factors(void) {
    const idx_t n = 50;
    SparseMatrix *A1 = integration_build_tridiag_spd(n);
    SparseMatrix *A2 = integration_build_tridiag_spd(n);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact = NULL;
    double *b1 = NULL;
    double *b2 = NULL;
    double *x1 = NULL;
    double *x2 = NULL;

    REQUIRE_OK(A1 && A2 ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_analyze(A1, &analysis_opts, &analysis), SPARSE_OK);

    x_exact = malloc((size_t)n * sizeof(double));
    b1 = malloc((size_t)n * sizeof(double));
    b2 = malloc((size_t)n * sizeof(double));
    x1 = malloc((size_t)n * sizeof(double));
    x2 = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b1 && b2 && x1 && x2 ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;

    sparse_matvec(A1, x_exact, b1);
    ASSERT_EQ(sparse_refactor_numeric(A1, &analysis, &factors), SPARSE_OK);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b1, x1), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x1[i], x_exact[i], 1e-12);

    for (idx_t i = 0; i < n; i++)
        ASSERT_EQ(sparse_set(A2, i, i, 5.0), SPARSE_OK);

    sparse_matvec(A2, x_exact, b2);
    ASSERT_EQ(sparse_refactor_numeric(A2, &analysis, &factors), SPARSE_OK);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b2, x2), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x2[i], x_exact[i], 1e-12);

    free(x_exact);
    free(b1);
    free(b2);
    free(x1);
    free(x2);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A1);
    sparse_free(A2);
}

static void test_public_lifecycle_refactor_rejects_mismatched_existing_factors(void) {
    SparseMatrix *A_lu = integration_build_unsym_4x4();
    SparseMatrix *A_spd = integration_build_tridiag_spd(4);
    SparseMatrix *A_spd_new = integration_build_tridiag_spd(4);
    sparse_analysis_t lu_analysis = {0};
    sparse_analysis_t chol_analysis = {0};
    sparse_factors_t factors = {0};
    double b_lu[4] = {1.0, 2.0, 3.0, 4.0};
    double x_lu[4];
    double r_lu[4];

    REQUIRE_OK(A_lu && A_spd && A_spd_new ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_analysis_opts_t lu_opts = {
        .factor_type = SPARSE_FACTOR_LU,
        .reorder = SPARSE_REORDER_NONE,
    };
    sparse_analysis_opts_t chol_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_NONE,
    };
    ASSERT_EQ(sparse_analyze(A_lu, &lu_opts, &lu_analysis), SPARSE_OK);
    ASSERT_EQ(sparse_analyze(A_spd, &chol_opts, &chol_analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_lu, &lu_analysis, &factors), SPARSE_OK);

    ASSERT_EQ(sparse_refactor_numeric(A_spd_new, &chol_analysis, &factors), SPARSE_ERR_BADARG);

    ASSERT_EQ(sparse_factor_solve(&factors, &lu_analysis, b_lu, x_lu), SPARSE_OK);
    sparse_matvec(A_lu, x_lu, r_lu);
    for (idx_t i = 0; i < 4; i++)
        r_lu[i] = b_lu[i] - r_lu[i];
    ASSERT_TRUE(vec_norminf(r_lu, 4) < 1e-12);

    sparse_factor_free(&factors);
    sparse_analysis_free(&lu_analysis);
    sparse_analysis_free(&chol_analysis);
    sparse_free(A_lu);
    sparse_free(A_spd);
    sparse_free(A_spd_new);
}

static void test_public_lifecycle_refactor_preserves_old_factors_on_failure(void) {
    const idx_t n = 40;
    SparseMatrix *A_good = integration_build_tridiag_spd(n);
    SparseMatrix *A_bad = NULL;
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact = NULL;
    double *b = NULL;
    double *x = NULL;

    REQUIRE_OK(A_good ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_analyze(A_good, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_good, &analysis, &factors), SPARSE_OK);

    x_exact = malloc((size_t)n * sizeof(double));
    b = malloc((size_t)n * sizeof(double));
    x = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b && x ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;
    sparse_matvec(A_good, x_exact, b);

    A_bad = sparse_copy(A_good);
    REQUIRE_OK(A_bad ? SPARSE_OK : SPARSE_ERR_ALLOC);
    ASSERT_EQ(sparse_set(A_bad, 0, 0, -1.0), SPARSE_OK);

    ASSERT_EQ(sparse_refactor_numeric(A_bad, &analysis, &factors), SPARSE_ERR_NOT_SPD);

    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b, x), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-12);

    free(x_exact);
    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_bad);
    sparse_free(A_good);
}

static void test_public_lifecycle_refactor_rejects_nnz_drift_and_preserves_old_factors(void) {
    const idx_t n = 40;
    SparseMatrix *A_good = integration_build_tridiag_spd(n);
    SparseMatrix *A_bad = NULL;
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact = NULL;
    double *b = NULL;
    double *x = NULL;

    REQUIRE_OK(A_good ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_analyze(A_good, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_good, &analysis, &factors), SPARSE_OK);

    x_exact = malloc((size_t)n * sizeof(double));
    b = malloc((size_t)n * sizeof(double));
    x = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b && x ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;
    sparse_matvec(A_good, x_exact, b);

    A_bad = sparse_copy(A_good);
    REQUIRE_OK(A_bad ? SPARSE_OK : SPARSE_ERR_ALLOC);
    ASSERT_EQ(sparse_set(A_bad, 0, 1, 0.0), SPARSE_OK);
    ASSERT_EQ(sparse_set(A_bad, 1, 0, 0.0), SPARSE_OK);

    ASSERT_EQ(sparse_refactor_numeric(A_bad, &analysis, &factors), SPARSE_ERR_BADARG);

    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b, x), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-12);

    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_bad);
    sparse_free(A_good);
    free(x_exact);
    free(b);
    free(x);
}

static void test_public_lifecycle_cholesky_csc_refactor_preserves_old_factors_on_failure(void) {
    const idx_t n = 120;
    SparseMatrix *A_good = integration_build_tridiag_spd(n);
    SparseMatrix *A_bad = NULL;
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact = NULL;
    double *b = NULL;
    double *x = NULL;

    REQUIRE_OK(A_good ? SPARSE_OK : SPARSE_ERR_ALLOC);
    ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_analyze(A_good, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_good, &analysis, &factors), SPARSE_OK);

    x_exact = malloc((size_t)n * sizeof(double));
    b = malloc((size_t)n * sizeof(double));
    x = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b && x ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;
    sparse_matvec(A_good, x_exact, b);

    A_bad = sparse_copy(A_good);
    REQUIRE_OK(A_bad ? SPARSE_OK : SPARSE_ERR_ALLOC);
    ASSERT_EQ(sparse_set(A_bad, 0, 0, -1.0), SPARSE_OK);

    ASSERT_EQ(sparse_refactor_numeric(A_bad, &analysis, &factors), SPARSE_ERR_NOT_SPD);

    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b, x), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-12);

    free(x_exact);
    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_bad);
    sparse_free(A_good);
}

static void
test_public_lifecycle_cholesky_csc_refactor_rejects_nnz_drift_and_preserves_old_factors(void) {
    const idx_t n = 120;
    SparseMatrix *A_good = integration_build_tridiag_spd(n);
    SparseMatrix *A_bad = NULL;
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact = NULL;
    double *b = NULL;
    double *x = NULL;

    REQUIRE_OK(A_good ? SPARSE_OK : SPARSE_ERR_ALLOC);
    ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_analyze(A_good, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_good, &analysis, &factors), SPARSE_OK);

    x_exact = malloc((size_t)n * sizeof(double));
    b = malloc((size_t)n * sizeof(double));
    x = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b && x ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;
    sparse_matvec(A_good, x_exact, b);

    A_bad = sparse_copy(A_good);
    REQUIRE_OK(A_bad ? SPARSE_OK : SPARSE_ERR_ALLOC);
    ASSERT_EQ(sparse_set(A_bad, 0, 1, 0.0), SPARSE_OK);
    ASSERT_EQ(sparse_set(A_bad, 1, 0, 0.0), SPARSE_OK);

    ASSERT_EQ(sparse_refactor_numeric(A_bad, &analysis, &factors), SPARSE_ERR_BADARG);

    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b, x), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-12);

    free(x_exact);
    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_bad);
    sparse_free(A_good);
}

static void test_public_lifecycle_refactor_failure_allows_retry(void) {
    const idx_t n = 40;
    SparseMatrix *A_good = integration_build_tridiag_spd(n);
    SparseMatrix *A_bad = NULL;
    SparseMatrix *A_retry = NULL;
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact_old = NULL;
    double *x_exact_retry = NULL;
    double *b_old = NULL;
    double *b_retry = NULL;
    double *x_old = NULL;
    double *x_retry = NULL;

    REQUIRE_OK(A_good ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_analyze(A_good, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_good, &analysis, &factors), SPARSE_OK);

    x_exact_old = malloc((size_t)n * sizeof(double));
    x_exact_retry = malloc((size_t)n * sizeof(double));
    b_old = malloc((size_t)n * sizeof(double));
    b_retry = malloc((size_t)n * sizeof(double));
    x_old = malloc((size_t)n * sizeof(double));
    x_retry = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact_old && x_exact_retry && b_old && b_retry && x_old && x_retry
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++) {
        x_exact_old[i] = 1.0;
        x_exact_retry[i] = 2.0 - 0.01 * (double)i;
    }
    sparse_matvec(A_good, x_exact_old, b_old);

    A_bad = sparse_copy(A_good);
    A_retry = sparse_copy(A_good);
    REQUIRE_OK(A_bad && A_retry ? SPARSE_OK : SPARSE_ERR_ALLOC);
    ASSERT_EQ(sparse_set(A_bad, 0, 0, -1.0), SPARSE_OK);
    for (idx_t i = 0; i < n; i++) {
        const double diag = sparse_get(A_retry, i, i);
        ASSERT_EQ(sparse_set(A_retry, i, i, diag + 0.5 + 0.01 * (double)i), SPARSE_OK);
    }
    sparse_matvec(A_retry, x_exact_retry, b_retry);

    ASSERT_EQ(sparse_refactor_numeric(A_bad, &analysis, &factors), SPARSE_ERR_NOT_SPD);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b_old, x_old), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_old[i], x_exact_old[i], 1e-12);

    ASSERT_EQ(sparse_refactor_numeric(A_retry, &analysis, &factors), SPARSE_OK);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b_retry, x_retry), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_retry[i], x_exact_retry[i], 1e-12);

    free(x_exact_old);
    free(x_exact_retry);
    free(b_old);
    free(b_retry);
    free(x_old);
    free(x_retry);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_bad);
    sparse_free(A_retry);
    sparse_free(A_good);
}

static void test_public_lifecycle_cholesky_csc_refactor_failure_allows_retry(void) {
    const idx_t n = 120;
    SparseMatrix *A_good = integration_build_tridiag_spd(n);
    SparseMatrix *A_bad = NULL;
    SparseMatrix *A_retry = NULL;
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact_old = NULL;
    double *x_exact_retry = NULL;
    double *b_old = NULL;
    double *b_retry = NULL;
    double *x_old = NULL;
    double *x_retry = NULL;

    REQUIRE_OK(A_good ? SPARSE_OK : SPARSE_ERR_ALLOC);
    ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_analyze(A_good, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_good, &analysis, &factors), SPARSE_OK);

    x_exact_old = malloc((size_t)n * sizeof(double));
    x_exact_retry = malloc((size_t)n * sizeof(double));
    b_old = malloc((size_t)n * sizeof(double));
    b_retry = malloc((size_t)n * sizeof(double));
    x_old = malloc((size_t)n * sizeof(double));
    x_retry = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact_old && x_exact_retry && b_old && b_retry && x_old && x_retry
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++) {
        x_exact_old[i] = 1.0;
        x_exact_retry[i] = 1.5 + 0.002 * (double)i;
    }
    sparse_matvec(A_good, x_exact_old, b_old);

    A_bad = sparse_copy(A_good);
    A_retry = sparse_copy(A_good);
    REQUIRE_OK(A_bad && A_retry ? SPARSE_OK : SPARSE_ERR_ALLOC);
    ASSERT_EQ(sparse_set(A_bad, 0, 0, -1.0), SPARSE_OK);
    for (idx_t i = 0; i < n; i++) {
        const double diag = sparse_get(A_retry, i, i);
        ASSERT_EQ(sparse_set(A_retry, i, i, diag + 0.75 + 0.001 * (double)i), SPARSE_OK);
    }
    sparse_matvec(A_retry, x_exact_retry, b_retry);

    ASSERT_EQ(sparse_refactor_numeric(A_bad, &analysis, &factors), SPARSE_ERR_NOT_SPD);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b_old, x_old), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_old[i], x_exact_old[i], 1e-12);

    ASSERT_EQ(sparse_refactor_numeric(A_retry, &analysis, &factors), SPARSE_OK);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b_retry, x_retry), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_retry[i], x_exact_retry[i], 1e-12);

    free(x_exact_old);
    free(x_exact_retry);
    free(b_old);
    free(b_retry);
    free(x_old);
    free(x_retry);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_bad);
    sparse_free(A_retry);
    sparse_free(A_good);
}

static void test_public_lifecycle_ldlt_refactor_failure_allows_retry_amd(void) {
    const idx_t n_top = 140;
    const idx_t n_bot = 10;
    const idx_t n = n_top + n_bot;
    SparseMatrix *A_good = integration_build_kkt(n_top, n_bot);
    SparseMatrix *A_bad = NULL;
    SparseMatrix *A_retry = NULL;
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact_old = NULL;
    double *x_exact_retry = NULL;
    double *b_old = NULL;
    double *b_retry = NULL;
    double *x_old = NULL;
    double *x_retry = NULL;

    REQUIRE_OK(A_good ? SPARSE_OK : SPARSE_ERR_ALLOC);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_LDLT,
        .reorder = SPARSE_REORDER_AMD,
    };
    ASSERT_EQ(sparse_analyze(A_good, &analysis_opts, &analysis), SPARSE_OK);
    ASSERT_EQ(sparse_factor_numeric(A_good, &analysis, &factors), SPARSE_OK);
    ASSERT_NOT_NULL(factors.ldlt_perm);
    ASSERT_NOT_NULL(factors.pivot_size);

    x_exact_old = malloc((size_t)n * sizeof(double));
    x_exact_retry = malloc((size_t)n * sizeof(double));
    b_old = malloc((size_t)n * sizeof(double));
    b_retry = malloc((size_t)n * sizeof(double));
    x_old = malloc((size_t)n * sizeof(double));
    x_retry = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact_old && x_exact_retry && b_old && b_retry && x_old && x_retry
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++) {
        x_exact_old[i] = 1.0;
        x_exact_retry[i] = 1.0 + 0.005 * (double)i;
    }
    sparse_matvec(A_good, x_exact_old, b_old);

    A_bad = sparse_copy(A_good);
    A_retry = integration_build_kkt(n_top, n_bot);
    REQUIRE_OK(A_bad && A_retry ? SPARSE_OK : SPARSE_ERR_ALLOC);
    ASSERT_EQ(sparse_set(A_bad, 0, n_top, 0.0), SPARSE_OK);
    ASSERT_EQ(sparse_set(A_bad, n_top, 0, 0.0), SPARSE_OK);
    integration_perturb_kkt_values_in_place(A_retry, n_top, n_bot, 0.35);
    sparse_matvec(A_retry, x_exact_retry, b_retry);

    ASSERT_EQ(sparse_refactor_numeric(A_bad, &analysis, &factors), SPARSE_ERR_BADARG);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b_old, x_old), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_old[i], x_exact_old[i], 1e-10);

    ASSERT_EQ(sparse_refactor_numeric(A_retry, &analysis, &factors), SPARSE_OK);
    ASSERT_NOT_NULL(factors.ldlt_perm);
    ASSERT_NOT_NULL(factors.pivot_size);
    ASSERT_EQ(sparse_factor_solve(&factors, &analysis, b_retry, x_retry), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_retry[i], x_exact_retry[i], 1e-10);

    free(x_exact_old);
    free(x_exact_retry);
    free(b_old);
    free(b_retry);
    free(x_old);
    free(x_retry);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_bad);
    sparse_free(A_retry);
    sparse_free(A_good);
}

static void test_public_lifecycle_refactor_same_pattern_matches_one_shot_cholesky(void) {
    const idx_t n = 120;
    SparseMatrix *A_base = integration_build_tridiag_spd(n);
    SparseMatrix *A_refactor1 = integration_build_tridiag_spd(n);
    SparseMatrix *A_refactor2 = integration_build_tridiag_spd(n);
    SparseMatrix *A_one_shot0 = integration_build_tridiag_spd(n);
    SparseMatrix *A_one_shot1 = integration_build_tridiag_spd(n);
    SparseMatrix *A_one_shot2 = integration_build_tridiag_spd(n);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact = NULL;
    double *b0 = NULL;
    double *b1 = NULL;
    double *b2 = NULL;
    double *x_public0 = NULL;
    double *x_public1 = NULL;
    double *x_public2 = NULL;
    double *x_one_shot0 = NULL;
    double *x_one_shot1 = NULL;
    double *x_one_shot2 = NULL;
    int used_csc_path0 = 0;
    int used_csc_path1 = 0;
    int used_csc_path2 = 0;

    REQUIRE_OK(A_base && A_refactor1 && A_refactor2 && A_one_shot0 && A_one_shot1 && A_one_shot2
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);
    ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_cholesky_opts_t chol_opts0 = {
        .reorder = SPARSE_REORDER_AMD,
        .used_csc_path = &used_csc_path0,
    };
    sparse_cholesky_opts_t chol_opts1 = {
        .reorder = SPARSE_REORDER_AMD,
        .used_csc_path = &used_csc_path1,
    };
    sparse_cholesky_opts_t chol_opts2 = {
        .reorder = SPARSE_REORDER_AMD,
        .used_csc_path = &used_csc_path2,
    };

    REQUIRE_OK(sparse_analyze(A_base, &analysis_opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A_base, &analysis, &factors));

    x_exact = malloc((size_t)n * sizeof(double));
    b0 = malloc((size_t)n * sizeof(double));
    b1 = malloc((size_t)n * sizeof(double));
    b2 = malloc((size_t)n * sizeof(double));
    x_public0 = malloc((size_t)n * sizeof(double));
    x_public1 = malloc((size_t)n * sizeof(double));
    x_public2 = malloc((size_t)n * sizeof(double));
    x_one_shot0 = malloc((size_t)n * sizeof(double));
    x_one_shot1 = malloc((size_t)n * sizeof(double));
    x_one_shot2 = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b0 && b1 && b2 && x_public0 && x_public1 && x_public2 && x_one_shot0 &&
                       x_one_shot1 && x_one_shot2
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++) {
        x_exact[i] = 0.5 + 0.125 * (double)i;
        ASSERT_EQ(sparse_set(A_refactor1, i, i, 5.0), SPARSE_OK);
        ASSERT_EQ(sparse_set(A_refactor2, i, i, 6.5 + 0.01 * (double)i), SPARSE_OK);
        ASSERT_EQ(sparse_set(A_one_shot1, i, i, 5.0), SPARSE_OK);
        ASSERT_EQ(sparse_set(A_one_shot2, i, i, 6.5 + 0.01 * (double)i), SPARSE_OK);
    }

    sparse_matvec(A_base, x_exact, b0);
    sparse_matvec(A_refactor1, x_exact, b1);
    sparse_matvec(A_refactor2, x_exact, b2);

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b0, x_public0));

    REQUIRE_OK(sparse_cholesky_factor_opts(A_one_shot0, &chol_opts0));
    ASSERT_EQ(used_csc_path0, 1);
    REQUIRE_OK(sparse_cholesky_solve(A_one_shot0, b0, x_one_shot0));

    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public0[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_one_shot0[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_public0[i], x_one_shot0[i], 1e-12);
    }

    REQUIRE_OK(sparse_refactor_numeric(A_refactor1, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b1, x_public1));

    REQUIRE_OK(sparse_cholesky_factor_opts(A_one_shot1, &chol_opts1));
    ASSERT_EQ(used_csc_path1, 1);
    REQUIRE_OK(sparse_cholesky_solve(A_one_shot1, b1, x_one_shot1));

    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public1[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_one_shot1[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_public1[i], x_one_shot1[i], 1e-12);
    }

    REQUIRE_OK(sparse_refactor_numeric(A_refactor2, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b2, x_public2));

    REQUIRE_OK(sparse_cholesky_factor_opts(A_one_shot2, &chol_opts2));
    ASSERT_EQ(used_csc_path2, 1);
    REQUIRE_OK(sparse_cholesky_solve(A_one_shot2, b2, x_one_shot2));

    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public2[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_one_shot2[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_public2[i], x_one_shot2[i], 1e-12);
    }

    free(x_exact);
    free(b0);
    free(b1);
    free(b2);
    free(x_public0);
    free(x_public1);
    free(x_public2);
    free(x_one_shot0);
    free(x_one_shot1);
    free(x_one_shot2);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_base);
    sparse_free(A_refactor1);
    sparse_free(A_refactor2);
    sparse_free(A_one_shot0);
    sparse_free(A_one_shot1);
    sparse_free(A_one_shot2);
}

static void
test_public_lifecycle_constructor_built_csc_refactor_same_pattern_matches_one_shot_cholesky(void) {
    const idx_t n = (idx_t)(SPARSE_CSC_THRESHOLD + 20);
    SparseMatrix *A_base_src = integration_build_tridiag_spd(n);
    SparseMatrix *A_refactor1_src = integration_build_tridiag_spd(n);
    SparseMatrix *A_refactor2_src = integration_build_tridiag_spd(n);
    SparseMatrix *A_one_shot0_src = integration_build_tridiag_spd(n);
    SparseMatrix *A_one_shot1_src = integration_build_tridiag_spd(n);
    SparseMatrix *A_one_shot2_src = integration_build_tridiag_spd(n);
    SparseMatrix *A_base = NULL;
    SparseMatrix *A_refactor1 = NULL;
    SparseMatrix *A_refactor2 = NULL;
    SparseMatrix *A_one_shot0 = NULL;
    SparseMatrix *A_one_shot1 = NULL;
    SparseMatrix *A_one_shot2 = NULL;
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact = NULL;
    double *b0 = NULL;
    double *b1 = NULL;
    double *b2 = NULL;
    double *x_public0 = NULL;
    double *x_public1 = NULL;
    double *x_public2 = NULL;
    double *x_one_shot0 = NULL;
    double *x_one_shot1 = NULL;
    double *x_one_shot2 = NULL;
    int used_csc_path0 = 0;
    int used_csc_path1 = 0;
    int used_csc_path2 = 0;

    REQUIRE_OK(A_base_src && A_refactor1_src && A_refactor2_src && A_one_shot0_src &&
                       A_one_shot1_src && A_one_shot2_src
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);
    ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD);

    for (idx_t i = 0; i < n; i++) {
        ASSERT_EQ(sparse_set(A_refactor1_src, i, i, 5.0), SPARSE_OK);
        ASSERT_EQ(sparse_set(A_refactor2_src, i, i, 6.5 + 0.01 * (double)i), SPARSE_OK);
        ASSERT_EQ(sparse_set(A_one_shot1_src, i, i, 5.0), SPARSE_OK);
        ASSERT_EQ(sparse_set(A_one_shot2_src, i, i, 6.5 + 0.01 * (double)i), SPARSE_OK);
    }

    A_base = integration_build_from_csc_constructor(A_base_src);
    A_refactor1 = integration_build_from_csc_constructor(A_refactor1_src);
    A_refactor2 = integration_build_from_csc_constructor(A_refactor2_src);
    A_one_shot0 = integration_build_from_csc_constructor(A_one_shot0_src);
    A_one_shot1 = integration_build_from_csc_constructor(A_one_shot1_src);
    A_one_shot2 = integration_build_from_csc_constructor(A_one_shot2_src);

    REQUIRE_OK(A_base && A_refactor1 && A_refactor2 && A_one_shot0 && A_one_shot1 && A_one_shot2
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_cholesky_opts_t chol_opts0 = {
        .reorder = SPARSE_REORDER_AMD,
        .used_csc_path = &used_csc_path0,
    };
    sparse_cholesky_opts_t chol_opts1 = {
        .reorder = SPARSE_REORDER_AMD,
        .used_csc_path = &used_csc_path1,
    };
    sparse_cholesky_opts_t chol_opts2 = {
        .reorder = SPARSE_REORDER_AMD,
        .used_csc_path = &used_csc_path2,
    };

    REQUIRE_OK(sparse_analyze(A_base, &analysis_opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A_base, &analysis, &factors));

    x_exact = malloc((size_t)n * sizeof(double));
    b0 = malloc((size_t)n * sizeof(double));
    b1 = malloc((size_t)n * sizeof(double));
    b2 = malloc((size_t)n * sizeof(double));
    x_public0 = malloc((size_t)n * sizeof(double));
    x_public1 = malloc((size_t)n * sizeof(double));
    x_public2 = malloc((size_t)n * sizeof(double));
    x_one_shot0 = malloc((size_t)n * sizeof(double));
    x_one_shot1 = malloc((size_t)n * sizeof(double));
    x_one_shot2 = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b0 && b1 && b2 && x_public0 && x_public1 && x_public2 && x_one_shot0 &&
                       x_one_shot1 && x_one_shot2
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 0.5 + 0.125 * (double)i;

    sparse_matvec(A_base, x_exact, b0);
    sparse_matvec(A_refactor1, x_exact, b1);
    sparse_matvec(A_refactor2, x_exact, b2);

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b0, x_public0));

    REQUIRE_OK(sparse_cholesky_factor_opts(A_one_shot0, &chol_opts0));
    ASSERT_EQ(used_csc_path0, 1);
    REQUIRE_OK(sparse_cholesky_solve(A_one_shot0, b0, x_one_shot0));

    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public0[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_one_shot0[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_public0[i], x_one_shot0[i], 1e-12);
    }

    REQUIRE_OK(sparse_refactor_numeric(A_refactor1, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b1, x_public1));

    REQUIRE_OK(sparse_cholesky_factor_opts(A_one_shot1, &chol_opts1));
    ASSERT_EQ(used_csc_path1, 1);
    REQUIRE_OK(sparse_cholesky_solve(A_one_shot1, b1, x_one_shot1));

    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public1[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_one_shot1[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_public1[i], x_one_shot1[i], 1e-12);
    }

    REQUIRE_OK(sparse_refactor_numeric(A_refactor2, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b2, x_public2));

    REQUIRE_OK(sparse_cholesky_factor_opts(A_one_shot2, &chol_opts2));
    ASSERT_EQ(used_csc_path2, 1);
    REQUIRE_OK(sparse_cholesky_solve(A_one_shot2, b2, x_one_shot2));

    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public2[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_one_shot2[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_public2[i], x_one_shot2[i], 1e-12);
    }

    free(x_exact);
    free(b0);
    free(b1);
    free(b2);
    free(x_public0);
    free(x_public1);
    free(x_public2);
    free(x_one_shot0);
    free(x_one_shot1);
    free(x_one_shot2);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_base);
    sparse_free(A_refactor1);
    sparse_free(A_refactor2);
    sparse_free(A_one_shot0);
    sparse_free(A_one_shot1);
    sparse_free(A_one_shot2);
    sparse_free(A_base_src);
    sparse_free(A_refactor1_src);
    sparse_free(A_refactor2_src);
    sparse_free(A_one_shot0_src);
    sparse_free(A_one_shot1_src);
    sparse_free(A_one_shot2_src);
}

static void test_public_lifecycle_refactor_small_same_pattern_matches_forced_csc_cholesky(void) {
    const idx_t n = 40;
    SparseMatrix *A_base = integration_build_tridiag_spd(n);
    SparseMatrix *A_refactor1 = integration_build_tridiag_spd(n);
    SparseMatrix *A_refactor2 = integration_build_tridiag_spd(n);
    SparseMatrix *A_one_shot0 = integration_build_tridiag_spd(n);
    SparseMatrix *A_one_shot1 = integration_build_tridiag_spd(n);
    SparseMatrix *A_one_shot2 = integration_build_tridiag_spd(n);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    double *x_exact = NULL;
    double *b0 = NULL;
    double *b1 = NULL;
    double *b2 = NULL;
    double *x_public0 = NULL;
    double *x_public1 = NULL;
    double *x_public2 = NULL;
    double *x_one_shot0 = NULL;
    double *x_one_shot1 = NULL;
    double *x_one_shot2 = NULL;
    int used_csc_path0 = 0;
    int used_csc_path1 = 0;
    int used_csc_path2 = 0;

    REQUIRE_OK(A_base && A_refactor1 && A_refactor2 && A_one_shot0 && A_one_shot1 && A_one_shot2
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);
    ASSERT_TRUE(n < SPARSE_CSC_THRESHOLD);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_cholesky_opts_t chol_opts0 = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_CHOL_BACKEND_CSC,
        .used_csc_path = &used_csc_path0,
    };
    sparse_cholesky_opts_t chol_opts1 = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_CHOL_BACKEND_CSC,
        .used_csc_path = &used_csc_path1,
    };
    sparse_cholesky_opts_t chol_opts2 = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_CHOL_BACKEND_CSC,
        .used_csc_path = &used_csc_path2,
    };

    REQUIRE_OK(sparse_analyze(A_base, &analysis_opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A_base, &analysis, &factors));

    x_exact = malloc((size_t)n * sizeof(double));
    b0 = malloc((size_t)n * sizeof(double));
    b1 = malloc((size_t)n * sizeof(double));
    b2 = malloc((size_t)n * sizeof(double));
    x_public0 = malloc((size_t)n * sizeof(double));
    x_public1 = malloc((size_t)n * sizeof(double));
    x_public2 = malloc((size_t)n * sizeof(double));
    x_one_shot0 = malloc((size_t)n * sizeof(double));
    x_one_shot1 = malloc((size_t)n * sizeof(double));
    x_one_shot2 = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b0 && b1 && b2 && x_public0 && x_public1 && x_public2 && x_one_shot0 &&
                       x_one_shot1 && x_one_shot2
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++) {
        x_exact[i] = 0.5 + 0.125 * (double)i;
        ASSERT_EQ(sparse_set(A_refactor1, i, i, 5.0), SPARSE_OK);
        ASSERT_EQ(sparse_set(A_refactor2, i, i, 6.5 + 0.01 * (double)i), SPARSE_OK);
        ASSERT_EQ(sparse_set(A_one_shot1, i, i, 5.0), SPARSE_OK);
        ASSERT_EQ(sparse_set(A_one_shot2, i, i, 6.5 + 0.01 * (double)i), SPARSE_OK);
    }

    sparse_matvec(A_base, x_exact, b0);
    sparse_matvec(A_refactor1, x_exact, b1);
    sparse_matvec(A_refactor2, x_exact, b2);

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b0, x_public0));

    REQUIRE_OK(sparse_cholesky_factor_opts(A_one_shot0, &chol_opts0));
    ASSERT_EQ(used_csc_path0, 1);
    REQUIRE_OK(sparse_cholesky_solve(A_one_shot0, b0, x_one_shot0));

    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public0[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_one_shot0[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_public0[i], x_one_shot0[i], 1e-12);
    }

    REQUIRE_OK(sparse_refactor_numeric(A_refactor1, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b1, x_public1));

    REQUIRE_OK(sparse_cholesky_factor_opts(A_one_shot1, &chol_opts1));
    ASSERT_EQ(used_csc_path1, 1);
    REQUIRE_OK(sparse_cholesky_solve(A_one_shot1, b1, x_one_shot1));

    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public1[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_one_shot1[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_public1[i], x_one_shot1[i], 1e-12);
    }

    REQUIRE_OK(sparse_refactor_numeric(A_refactor2, &analysis, &factors));
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b2, x_public2));

    REQUIRE_OK(sparse_cholesky_factor_opts(A_one_shot2, &chol_opts2));
    ASSERT_EQ(used_csc_path2, 1);
    REQUIRE_OK(sparse_cholesky_solve(A_one_shot2, b2, x_one_shot2));

    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public2[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_one_shot2[i], x_exact[i], 1e-12);
        ASSERT_NEAR(x_public2[i], x_one_shot2[i], 1e-12);
    }

    free(x_exact);
    free(b0);
    free(b1);
    free(b2);
    free(x_public0);
    free(x_public1);
    free(x_public2);
    free(x_one_shot0);
    free(x_one_shot1);
    free(x_one_shot2);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_base);
    sparse_free(A_refactor1);
    sparse_free(A_refactor2);
    sparse_free(A_one_shot0);
    sparse_free(A_one_shot1);
    sparse_free(A_one_shot2);
}

static void test_public_lifecycle_refactor_same_pattern_matches_one_shot_ldlt(void) {
    const idx_t n_top = (idx_t)(SPARSE_CSC_THRESHOLD + 12);
    const idx_t n_bot = 8;
    const idx_t n = n_top + n_bot;
    SparseMatrix *A_base = integration_build_kkt(n_top, n_bot);
    SparseMatrix *A_refactor1 = integration_build_kkt(n_top, n_bot);
    SparseMatrix *A_refactor2 = integration_build_kkt(n_top, n_bot);
    SparseMatrix *A_one_shot0 = integration_build_kkt(n_top, n_bot);
    SparseMatrix *A_one_shot1 = integration_build_kkt(n_top, n_bot);
    SparseMatrix *A_one_shot2 = integration_build_kkt(n_top, n_bot);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    sparse_ldlt_t ldlt0 = {0};
    sparse_ldlt_t ldlt1 = {0};
    sparse_ldlt_t ldlt2 = {0};
    double *x_exact = NULL;
    double *b0 = NULL;
    double *b1 = NULL;
    double *b2 = NULL;
    double *x_public0 = NULL;
    double *x_public1 = NULL;
    double *x_public2 = NULL;
    double *x_one_shot0 = NULL;
    double *x_one_shot1 = NULL;
    double *x_one_shot2 = NULL;
    int used_csc_path0 = 0;
    int used_csc_path1 = 0;
    int used_csc_path2 = 0;

    REQUIRE_OK(A_base && A_refactor1 && A_refactor2 && A_one_shot0 && A_one_shot1 && A_one_shot2
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);
    ASSERT_TRUE(n >= SPARSE_CSC_THRESHOLD);

    integration_perturb_kkt_values_in_place(A_refactor1, n_top, n_bot, 0.2);
    integration_perturb_kkt_values_in_place(A_refactor2, n_top, n_bot, 0.45);
    integration_perturb_kkt_values_in_place(A_one_shot1, n_top, n_bot, 0.2);
    integration_perturb_kkt_values_in_place(A_one_shot2, n_top, n_bot, 0.45);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_LDLT,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_ldlt_opts_t ldlt_opts0 = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_LDLT_BACKEND_AUTO,
        .used_csc_path = &used_csc_path0,
    };
    sparse_ldlt_opts_t ldlt_opts1 = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_LDLT_BACKEND_AUTO,
        .used_csc_path = &used_csc_path1,
    };
    sparse_ldlt_opts_t ldlt_opts2 = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_LDLT_BACKEND_AUTO,
        .used_csc_path = &used_csc_path2,
    };

    REQUIRE_OK(sparse_analyze(A_base, &analysis_opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A_base, &analysis, &factors));
    ASSERT_NOT_NULL(factors.ldlt_perm);
    ASSERT_NOT_NULL(factors.pivot_size);

    x_exact = malloc((size_t)n * sizeof(double));
    b0 = malloc((size_t)n * sizeof(double));
    b1 = malloc((size_t)n * sizeof(double));
    b2 = malloc((size_t)n * sizeof(double));
    x_public0 = malloc((size_t)n * sizeof(double));
    x_public1 = malloc((size_t)n * sizeof(double));
    x_public2 = malloc((size_t)n * sizeof(double));
    x_one_shot0 = malloc((size_t)n * sizeof(double));
    x_one_shot1 = malloc((size_t)n * sizeof(double));
    x_one_shot2 = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b0 && b1 && b2 && x_public0 && x_public1 && x_public2 && x_one_shot0 &&
                       x_one_shot1 && x_one_shot2
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0 + 0.01 * (double)i;

    sparse_matvec(A_base, x_exact, b0);
    sparse_matvec(A_refactor1, x_exact, b1);
    sparse_matvec(A_refactor2, x_exact, b2);

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b0, x_public0));
    REQUIRE_OK(sparse_ldlt_factor_opts(A_one_shot0, &ldlt_opts0, &ldlt0));
    ASSERT_EQ(used_csc_path0, 1);
    REQUIRE_OK(sparse_ldlt_solve(&ldlt0, b0, x_one_shot0));
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public0[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_one_shot0[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_public0[i], x_one_shot0[i], 1e-10);
    }

    REQUIRE_OK(sparse_refactor_numeric(A_refactor1, &analysis, &factors));
    ASSERT_NOT_NULL(factors.ldlt_perm);
    ASSERT_NOT_NULL(factors.pivot_size);
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b1, x_public1));
    REQUIRE_OK(sparse_ldlt_factor_opts(A_one_shot1, &ldlt_opts1, &ldlt1));
    ASSERT_EQ(used_csc_path1, 1);
    REQUIRE_OK(sparse_ldlt_solve(&ldlt1, b1, x_one_shot1));
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public1[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_one_shot1[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_public1[i], x_one_shot1[i], 1e-10);
    }

    REQUIRE_OK(sparse_refactor_numeric(A_refactor2, &analysis, &factors));
    ASSERT_NOT_NULL(factors.ldlt_perm);
    ASSERT_NOT_NULL(factors.pivot_size);
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b2, x_public2));
    REQUIRE_OK(sparse_ldlt_factor_opts(A_one_shot2, &ldlt_opts2, &ldlt2));
    ASSERT_EQ(used_csc_path2, 1);
    REQUIRE_OK(sparse_ldlt_solve(&ldlt2, b2, x_one_shot2));
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public2[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_one_shot2[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_public2[i], x_one_shot2[i], 1e-10);
    }

    free(x_exact);
    free(b0);
    free(b1);
    free(b2);
    free(x_public0);
    free(x_public1);
    free(x_public2);
    free(x_one_shot0);
    free(x_one_shot1);
    free(x_one_shot2);
    sparse_ldlt_free(&ldlt0);
    sparse_ldlt_free(&ldlt1);
    sparse_ldlt_free(&ldlt2);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_base);
    sparse_free(A_refactor1);
    sparse_free(A_refactor2);
    sparse_free(A_one_shot0);
    sparse_free(A_one_shot1);
    sparse_free(A_one_shot2);
}

static void test_public_lifecycle_refactor_small_same_pattern_matches_forced_csc_ldlt(void) {
    const idx_t n_top = 30;
    const idx_t n_bot = 10;
    const idx_t n = n_top + n_bot;
    SparseMatrix *A_base = integration_build_kkt(n_top, n_bot);
    SparseMatrix *A_refactor1 = integration_build_kkt(n_top, n_bot);
    SparseMatrix *A_refactor2 = integration_build_kkt(n_top, n_bot);
    SparseMatrix *A_one_shot0 = integration_build_kkt(n_top, n_bot);
    SparseMatrix *A_one_shot1 = integration_build_kkt(n_top, n_bot);
    SparseMatrix *A_one_shot2 = integration_build_kkt(n_top, n_bot);
    sparse_analysis_t analysis = {0};
    sparse_factors_t factors = {0};
    sparse_ldlt_t ldlt0 = {0};
    sparse_ldlt_t ldlt1 = {0};
    sparse_ldlt_t ldlt2 = {0};
    double *x_exact = NULL;
    double *b0 = NULL;
    double *b1 = NULL;
    double *b2 = NULL;
    double *x_public0 = NULL;
    double *x_public1 = NULL;
    double *x_public2 = NULL;
    double *x_one_shot0 = NULL;
    double *x_one_shot1 = NULL;
    double *x_one_shot2 = NULL;
    int used_csc_path0 = 0;
    int used_csc_path1 = 0;
    int used_csc_path2 = 0;

    REQUIRE_OK(A_base && A_refactor1 && A_refactor2 && A_one_shot0 && A_one_shot1 && A_one_shot2
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);
    ASSERT_TRUE(n < SPARSE_CSC_THRESHOLD);

    integration_perturb_kkt_values_in_place(A_refactor1, n_top, n_bot, 0.2);
    integration_perturb_kkt_values_in_place(A_refactor2, n_top, n_bot, 0.45);
    integration_perturb_kkt_values_in_place(A_one_shot1, n_top, n_bot, 0.2);
    integration_perturb_kkt_values_in_place(A_one_shot2, n_top, n_bot, 0.45);

    sparse_analysis_opts_t analysis_opts = {
        .factor_type = SPARSE_FACTOR_LDLT,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_ldlt_opts_t ldlt_opts0 = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_LDLT_BACKEND_CSC,
        .used_csc_path = &used_csc_path0,
    };
    sparse_ldlt_opts_t ldlt_opts1 = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_LDLT_BACKEND_CSC,
        .used_csc_path = &used_csc_path1,
    };
    sparse_ldlt_opts_t ldlt_opts2 = {
        .reorder = SPARSE_REORDER_AMD,
        .backend = SPARSE_LDLT_BACKEND_CSC,
        .used_csc_path = &used_csc_path2,
    };

    REQUIRE_OK(sparse_analyze(A_base, &analysis_opts, &analysis));
    REQUIRE_OK(sparse_factor_numeric(A_base, &analysis, &factors));
    ASSERT_NOT_NULL(factors.ldlt_perm);
    ASSERT_NOT_NULL(factors.pivot_size);

    x_exact = malloc((size_t)n * sizeof(double));
    b0 = malloc((size_t)n * sizeof(double));
    b1 = malloc((size_t)n * sizeof(double));
    b2 = malloc((size_t)n * sizeof(double));
    x_public0 = malloc((size_t)n * sizeof(double));
    x_public1 = malloc((size_t)n * sizeof(double));
    x_public2 = malloc((size_t)n * sizeof(double));
    x_one_shot0 = malloc((size_t)n * sizeof(double));
    x_one_shot1 = malloc((size_t)n * sizeof(double));
    x_one_shot2 = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(x_exact && b0 && b1 && b2 && x_public0 && x_public1 && x_public2 && x_one_shot0 &&
                       x_one_shot1 && x_one_shot2
                   ? SPARSE_OK
                   : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0 + 0.01 * (double)i;

    sparse_matvec(A_base, x_exact, b0);
    sparse_matvec(A_refactor1, x_exact, b1);
    sparse_matvec(A_refactor2, x_exact, b2);

    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b0, x_public0));
    REQUIRE_OK(sparse_ldlt_factor_opts(A_one_shot0, &ldlt_opts0, &ldlt0));
    ASSERT_EQ(used_csc_path0, 1);
    REQUIRE_OK(sparse_ldlt_solve(&ldlt0, b0, x_one_shot0));
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public0[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_one_shot0[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_public0[i], x_one_shot0[i], 1e-10);
    }

    REQUIRE_OK(sparse_refactor_numeric(A_refactor1, &analysis, &factors));
    ASSERT_NOT_NULL(factors.ldlt_perm);
    ASSERT_NOT_NULL(factors.pivot_size);
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b1, x_public1));
    REQUIRE_OK(sparse_ldlt_factor_opts(A_one_shot1, &ldlt_opts1, &ldlt1));
    ASSERT_EQ(used_csc_path1, 1);
    REQUIRE_OK(sparse_ldlt_solve(&ldlt1, b1, x_one_shot1));
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public1[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_one_shot1[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_public1[i], x_one_shot1[i], 1e-10);
    }

    REQUIRE_OK(sparse_refactor_numeric(A_refactor2, &analysis, &factors));
    ASSERT_NOT_NULL(factors.ldlt_perm);
    ASSERT_NOT_NULL(factors.pivot_size);
    REQUIRE_OK(sparse_factor_solve(&factors, &analysis, b2, x_public2));
    REQUIRE_OK(sparse_ldlt_factor_opts(A_one_shot2, &ldlt_opts2, &ldlt2));
    ASSERT_EQ(used_csc_path2, 1);
    REQUIRE_OK(sparse_ldlt_solve(&ldlt2, b2, x_one_shot2));
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_public2[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_one_shot2[i], x_exact[i], 1e-10);
        ASSERT_NEAR(x_public2[i], x_one_shot2[i], 1e-10);
    }

    free(x_exact);
    free(b0);
    free(b1);
    free(b2);
    free(x_public0);
    free(x_public1);
    free(x_public2);
    free(x_one_shot0);
    free(x_one_shot1);
    free(x_one_shot2);
    sparse_ldlt_free(&ldlt0);
    sparse_ldlt_free(&ldlt1);
    sparse_ldlt_free(&ldlt2);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A_base);
    sparse_free(A_refactor1);
    sparse_free(A_refactor2);
    sparse_free(A_one_shot0);
    sparse_free(A_one_shot1);
    sparse_free(A_one_shot2);
}

/* SPARSE_ERR_CANCELLED string round-trips through sparse_strerror. */
static void test_progress_cb_strerror(void) {
    const char *s = sparse_strerror(SPARSE_ERR_CANCELLED);
    ASSERT_TRUE(s != NULL);
    ASSERT_TRUE(strstr(s, "cancel") != NULL || strstr(s, "Cancel") != NULL);
}

/* Sprint 29 Day 7 (Item 4 close): progress / cancel coverage for QR,
 * iterative solvers (CG / GMRES / MINRES / BiCGSTAB), and the
 * eigsolver Lanczos + LOBPCG backends.  Pin the per-routine contract
 * for opts.progress_cb / progress_user:
 *   - emits >= 1 progress event during the iterative / factor phase.
 *   - cancellation returns SPARSE_ERR_CANCELLED.
 *   - default-NULL-callback path is bit-identical to Sprint 28.
 *
 * Each routine uses the same `integration_progress_count_cb` helper + a fresh
 * counter context.  Cancel tests set `cancel_after_step = 0` to fire
 * at the first emission, returning SPARSE_ERR_CANCELLED with the
 * library having done at most negligible work. */

static void test_progress_cb_qr_emits_cancel(void) {
    const idx_t m = 80, n = 50;
    SparseMatrix *A = sparse_create(m, n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < n; j++)
            if (i == j || (i + j) % 7 == 0)
                sparse_insert(A, i, j, (double)(i + j + 1));

    integration_progress_counter_t ctx = {.cancel_after_step = -1};
    sparse_qr_opts_t opts = {
        .reorder = SPARSE_REORDER_NONE,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    sparse_qr_t qr = {0};
    ASSERT_EQ(sparse_qr_factor_opts(A, &opts, &qr), SPARSE_OK);
    ASSERT_TRUE(ctx.n_calls > 0);
    ASSERT_TRUE(ctx.n_calls <= n);
    ASSERT_TRUE(strcmp(ctx.last_phase, "qr_factor") == 0);
    sparse_qr_free(&qr);

    /* Cancel at step 0. */
    integration_progress_counter_t ctx2 = {.cancel_after_step = 0};
    sparse_qr_opts_t opts2 = {
        .reorder = SPARSE_REORDER_NONE,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx2,
    };
    sparse_qr_t qr2 = {0};
    ASSERT_EQ(sparse_qr_factor_opts(A, &opts2, &qr2), SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx2.n_calls, 1);
    sparse_qr_free(&qr2);
    sparse_free(A);
}

static void test_progress_cb_cg_emits_cancel(void) {
    const idx_t n = 50;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(b && x ? SPARSE_OK : SPARSE_ERR_ALLOC);
    for (idx_t i = 0; i < n; i++) {
        b[i] = 1.0;
        x[i] = 0.0;
    }

    integration_progress_counter_t ctx = {.cancel_after_step = -1};
    sparse_iter_opts_t opts = {
        .max_iter = 200,
        .tol = 1e-10,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    sparse_err_t rc = sparse_solve_cg(A, b, x, &opts, NULL, NULL, NULL);
    ASSERT_EQ(rc, SPARSE_OK);
    ASSERT_TRUE(ctx.n_calls > 0);
    ASSERT_TRUE(strcmp(ctx.last_phase, "cg") == 0);

    /* Cancel: reset x, set cancel at step 0. */
    for (idx_t i = 0; i < n; i++)
        x[i] = 0.0;
    integration_progress_counter_t ctx2 = {.cancel_after_step = 0};
    sparse_iter_opts_t opts2 = {
        .max_iter = 200,
        .tol = 1e-10,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx2,
    };
    rc = sparse_solve_cg(A, b, x, &opts2, NULL, NULL, NULL);
    ASSERT_EQ(rc, SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx2.n_calls, 1);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_progress_cb_gmres_emits_cancel(void) {
    const idx_t n = 50;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(b && x ? SPARSE_OK : SPARSE_ERR_ALLOC);
    for (idx_t i = 0; i < n; i++) {
        b[i] = 1.0;
        x[i] = 0.0;
    }

    integration_progress_counter_t ctx = {.cancel_after_step = 0};
    sparse_gmres_opts_t opts = {
        .max_iter = 200,
        .restart = 30,
        .tol = 1e-10,
        .precond_side = SPARSE_PRECOND_LEFT,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    sparse_err_t rc = sparse_solve_gmres(A, b, x, &opts, NULL, NULL, NULL);
    ASSERT_EQ(rc, SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx.n_calls, 1);
    ASSERT_TRUE(strcmp(ctx.last_phase, "gmres") == 0);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_progress_cb_minres_emits_cancel(void) {
    const idx_t n = 50;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(b && x ? SPARSE_OK : SPARSE_ERR_ALLOC);
    for (idx_t i = 0; i < n; i++) {
        b[i] = 1.0;
        x[i] = 0.0;
    }

    integration_progress_counter_t ctx = {.cancel_after_step = 0};
    sparse_iter_opts_t opts = {
        .max_iter = 200,
        .tol = 1e-10,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    sparse_err_t rc = sparse_solve_minres(A, b, x, &opts, NULL, NULL, NULL);
    ASSERT_EQ(rc, SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx.n_calls, 1);
    ASSERT_TRUE(strcmp(ctx.last_phase, "minres") == 0);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_progress_cb_bicgstab_emits_cancel(void) {
    const idx_t n = 50;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(b && x ? SPARSE_OK : SPARSE_ERR_ALLOC);
    for (idx_t i = 0; i < n; i++) {
        b[i] = 1.0;
        x[i] = 0.0;
    }

    integration_progress_counter_t ctx = {.cancel_after_step = 0};
    sparse_iter_opts_t opts = {
        .max_iter = 200,
        .tol = 1e-10,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    sparse_err_t rc = sparse_solve_bicgstab(A, b, x, &opts, NULL, NULL, NULL);
    ASSERT_EQ(rc, SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx.n_calls, 1);
    ASSERT_TRUE(strcmp(ctx.last_phase, "bicgstab") == 0);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_progress_cb_lanczos_emits_cancel(void) {
    const idx_t n = 50;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    double vals[3] = {0, 0, 0};
    integration_progress_counter_t ctx = {.cancel_after_step = 0};
    sparse_eigs_t res = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .tol = 1e-10,
        .reorthogonalize = 1,
        .backend = SPARSE_EIGS_BACKEND_LANCZOS,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    sparse_err_t rc = sparse_eigs_sym(A, 3, &opts, &res);
    ASSERT_EQ(rc, SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx.n_calls, 1);
    ASSERT_TRUE(strcmp(ctx.last_phase, "lanczos") == 0);

    sparse_free(A);
}

static void test_progress_cb_lobpcg_emits_cancel(void) {
    const idx_t n = 50;
    SparseMatrix *A = integration_build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    idx_t k = 3;
    double vals[3] = {0, 0, 0};
    double *vecs = calloc((size_t)n * (size_t)k, sizeof(double));
    REQUIRE_OK(vecs ? SPARSE_OK : SPARSE_ERR_ALLOC);
    integration_progress_counter_t ctx = {.cancel_after_step = 0};
    sparse_eigs_t res = {.eigenvalues = vals, .eigenvectors = vecs};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .tol = 1e-10,
        .reorthogonalize = 1,
        .compute_vectors = 1,
        .backend = SPARSE_EIGS_BACKEND_LOBPCG,
        .block_size = k,
        .progress_cb = integration_progress_count_cb,
        .progress_user = &ctx,
    };
    sparse_err_t rc = sparse_eigs_sym(A, k, &opts, &res);
    ASSERT_EQ(rc, SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx.n_calls, 1);
    ASSERT_TRUE(strcmp(ctx.last_phase, "lobpcg") == 0);

    free(vecs);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Integration Tests");

    RUN_TEST(test_load_factor_solve_save);
    RUN_TEST(test_create_copy_factor_refine);
    RUN_TEST(test_multiple_rhs_same_factorization);
    RUN_TEST(test_reset_perms_invalidates_permuted_lu_shell);
    RUN_TEST(test_full_roundtrip);
    RUN_TEST(test_all_reference_matrices);
    RUN_TEST(test_both_pivots_agree_integration);
    RUN_TEST(test_error_recovery);

    /* Sprint 29 Day 6: progress / cancel callbacks (Item 4). */
    RUN_TEST(test_progress_cb_lu_emits);
    RUN_TEST(test_progress_cb_lu_cancel);
    RUN_TEST(test_progress_cb_lu_cancel_after_reorder_preserves_original_matrix);
    RUN_TEST(test_lu_refactor_attempt_rejects_existing_reordered_factor_and_preserves_old_factor);
    RUN_TEST(test_lu_invalid_reorder_opts_preserve_existing_reordered_factor);
    RUN_TEST(test_lu_invalid_pivot_opts_preserve_original_matrix_and_allow_retry);
    RUN_TEST(test_progress_cb_cholesky_emits_cancel);
    RUN_TEST(test_progress_cb_cholesky_csc_emits);
    RUN_TEST(test_progress_cb_cholesky_cancel_after_reorder_preserves_original_matrix);
    RUN_TEST(test_progress_cb_cholesky_csc_cancel_before_writeback_preserves_original_matrix);
    RUN_TEST(
        test_cholesky_refactor_attempt_rejects_existing_reordered_factor_and_preserves_old_factor);
    RUN_TEST(test_cholesky_reordered_not_spd_preserves_original_matrix);
    RUN_TEST(test_cholesky_invalid_backend_preserves_original_matrix_and_allows_retry);
    RUN_TEST(test_progress_cb_ldlt_emits_cancel);
    RUN_TEST(test_progress_cb_null_default_unchanged);
    RUN_TEST(test_cholesky_default_wrapper_matches_default_opts);
    RUN_TEST(test_ldlt_default_wrapper_matches_default_opts);
    RUN_TEST(test_lu_factor_opts_matches_explicit_analysis_path);
    RUN_TEST(test_create_from_csr_enters_one_shot_lu_workflow);
    RUN_TEST(test_cholesky_factor_opts_matches_explicit_analysis_path);
    RUN_TEST(test_ldlt_factor_opts_matches_explicit_analysis_path);
    RUN_TEST(test_ldlt_factor_opts_matches_explicit_analysis_path_indefinite_kkt);
    RUN_TEST(test_public_lifecycle_ldlt_refactor_same_pattern_indefinite_kkt);
    RUN_TEST(test_public_lifecycle_ldlt_refactor_same_pattern_indefinite_kkt_amd);
    RUN_TEST(test_public_lifecycle_ldlt_refactor_rejects_nnz_drift_and_preserves_old_factors_amd);
    RUN_TEST(test_public_lifecycle_solve_rejects_zeroed_factors);
    RUN_TEST(test_public_lifecycle_solve_rejects_mismatched_analysis_and_preserves_factors);
    RUN_TEST(test_public_lifecycle_repeated_solve_and_free_zeroed);
    RUN_TEST(test_public_lifecycle_refactor_accepts_zeroed_factors);
    RUN_TEST(test_public_lifecycle_refactor_rejects_mismatched_existing_factors);
    RUN_TEST(test_public_lifecycle_refactor_preserves_old_factors_on_failure);
    RUN_TEST(test_public_lifecycle_refactor_rejects_nnz_drift_and_preserves_old_factors);
    RUN_TEST(test_public_lifecycle_cholesky_csc_refactor_preserves_old_factors_on_failure);
    RUN_TEST(
        test_public_lifecycle_cholesky_csc_refactor_rejects_nnz_drift_and_preserves_old_factors);
    RUN_TEST(test_public_lifecycle_refactor_failure_allows_retry);
    RUN_TEST(test_public_lifecycle_cholesky_csc_refactor_failure_allows_retry);
    RUN_TEST(test_public_lifecycle_ldlt_refactor_failure_allows_retry_amd);
    RUN_TEST(test_public_lifecycle_refactor_same_pattern_matches_one_shot_cholesky);
    RUN_TEST(
        test_public_lifecycle_constructor_built_csc_refactor_same_pattern_matches_one_shot_cholesky);
    RUN_TEST(test_public_lifecycle_refactor_small_same_pattern_matches_forced_csc_cholesky);
    RUN_TEST(test_public_lifecycle_refactor_same_pattern_matches_one_shot_ldlt);
    RUN_TEST(test_public_lifecycle_refactor_small_same_pattern_matches_forced_csc_ldlt);
    RUN_TEST(test_progress_cb_strerror);

    /* Sprint 29 Day 7: progress / cancel coverage for QR, iterative
     * solvers, and eigsolver Lanczos + LOBPCG backends. */
    RUN_TEST(test_progress_cb_qr_emits_cancel);
    RUN_TEST(test_progress_cb_cg_emits_cancel);
    RUN_TEST(test_progress_cb_gmres_emits_cancel);
    RUN_TEST(test_progress_cb_minres_emits_cancel);
    RUN_TEST(test_progress_cb_bicgstab_emits_cancel);
    RUN_TEST(test_progress_cb_lanczos_emits_cancel);
    RUN_TEST(test_progress_cb_lobpcg_emits_cancel);

    TEST_SUITE_END();
}
