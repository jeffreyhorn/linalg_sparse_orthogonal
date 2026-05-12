#include "sparse_cholesky.h"
#include "sparse_ldlt.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
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
    ASSERT_ERR(sparse_save_mm(A, "/tmp/integ_tridiag.mtx"), SPARSE_OK);
    SparseMatrix *A2 = NULL;
    ASSERT_ERR(sparse_load_mm(&A2, "/tmp/integ_tridiag.mtx"), SPARSE_OK);
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
    ASSERT_ERR(sparse_save_mm(A, "/tmp/integ_roundtrip.mtx"), SPARSE_OK);

    /* Load */
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_load_mm(&B, "/tmp/integ_roundtrip.mtx"), SPARSE_OK);
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
 *   - cancellation at step=0 returns SPARSE_ERR_CANCELLED with the
 *     input matrix bit-identical to entry.
 *   - default-NULL-callback path is bit-identical to Sprint 28.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    idx_t n_calls;
    idx_t cancel_after_step; /* return non-zero when step == cancel_after_step; -1 = never */
    idx_t last_step;
    const char *last_phase;
    double last_elapsed_s;
} progress_counter_t;

static int progress_count_cb(const sparse_progress_t *p, void *user) {
    progress_counter_t *ctx = (progress_counter_t *)user;
    ctx->n_calls++;
    ctx->last_step = p->step;
    ctx->last_phase = p->phase;
    ctx->last_elapsed_s = p->elapsed_s;
    if (ctx->cancel_after_step >= 0 && p->step >= ctx->cancel_after_step)
        return 1;
    return 0;
}

/* Build a 100x100 SPD tridiagonal: diag = 4, sub/super = -1.
 * Diagonally dominant + symmetric → factorable by LU, Cholesky, LDL^T. */
static SparseMatrix *build_tridiag_spd(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    return A;
}

static void test_progress_cb_lu_emits(void) {
    const idx_t n = 100;
    SparseMatrix *A = build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    progress_counter_t ctx = {.cancel_after_step = -1};
    sparse_lu_opts_t opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_NONE,
        .tol = 1e-12,
        .progress_cb = progress_count_cb,
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
    SparseMatrix *A = build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    /* Snapshot the original matrix entries to verify bit-identity
     * after cancel-at-step-0. */
    SparseMatrix *A_orig = sparse_copy(A);
    REQUIRE_OK(A_orig ? SPARSE_OK : SPARSE_ERR_ALLOC);

    progress_counter_t ctx = {.cancel_after_step = 0};
    sparse_lu_opts_t opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_NONE,
        .tol = 1e-12,
        .progress_cb = progress_count_cb,
        .progress_user = &ctx,
    };
    ASSERT_EQ(sparse_lu_factor_opts(A, &opts), SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx.n_calls, 1); /* cancelled at the very first emission */

    /* Matrix must be unmodified: factored flag clear + same entries. */
    ASSERT_EQ(sparse_get(A, 0, 0), sparse_get(A_orig, 0, 0));
    for (idx_t i = 0; i < n; i++) {
        ASSERT_TRUE(sparse_get(A, i, i) == sparse_get(A_orig, i, i));
        if (i > 0) {
            ASSERT_TRUE(sparse_get(A, i, i - 1) == sparse_get(A_orig, i, i - 1));
            ASSERT_TRUE(sparse_get(A, i - 1, i) == sparse_get(A_orig, i - 1, i));
        }
    }

    sparse_free(A);
    sparse_free(A_orig);
}

static void test_progress_cb_cholesky_emits_cancel(void) {
    const idx_t n = 100;
    SparseMatrix *A = build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    /* Emit: count all callbacks, force linked-list backend so the
     * Day-6 emission path runs (CSC supernodal backend's emissions
     * are deferred). */
    progress_counter_t ctx = {.cancel_after_step = -1};
    sparse_cholesky_opts_t opts = {
        .reorder = SPARSE_REORDER_NONE,
        .backend = SPARSE_CHOL_BACKEND_LINKED_LIST,
        .progress_cb = progress_count_cb,
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
    A = build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    progress_counter_t ctx2 = {.cancel_after_step = 0};
    sparse_cholesky_opts_t opts2 = {
        .reorder = SPARSE_REORDER_NONE,
        .backend = SPARSE_CHOL_BACKEND_LINKED_LIST,
        .progress_cb = progress_count_cb,
        .progress_user = &ctx2,
    };
    ASSERT_EQ(sparse_cholesky_factor_opts(A, &opts2), SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx2.n_calls, 1);
    /* Diagonal preserved: cancellation at step=0 fires before any
     * column-k=0 update writes to L(0, 0). */
    ASSERT_TRUE(sparse_get(A, 0, 0) == 4.0);
    sparse_free(A);
}

static void test_progress_cb_ldlt_emits_cancel(void) {
    const idx_t n = 100;
    SparseMatrix *A = build_tridiag_spd(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    /* Emit: count, force linked-list backend.  LDL^T may use 2x2
     * pivots, so the number of emissions is <= n (each pivot
     * advances k by 1 or 2).  For our diagonally-dominant
     * tridiagonal SPD all pivots are 1x1 and we get exactly n
     * emissions, but the contract is "emissions ≤ n" generically. */
    progress_counter_t ctx = {.cancel_after_step = -1};
    sparse_ldlt_t ldlt = {0};
    sparse_ldlt_opts_t opts = {
        .reorder = SPARSE_REORDER_NONE,
        .tol = 0.0,
        .backend = SPARSE_LDLT_BACKEND_LINKED_LIST,
        .used_csc_path = NULL,
        .progress_cb = progress_count_cb,
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
    progress_counter_t ctx2 = {.cancel_after_step = 0};
    sparse_ldlt_t ldlt2 = {0};
    sparse_ldlt_opts_t opts2 = {
        .reorder = SPARSE_REORDER_NONE,
        .tol = 0.0,
        .backend = SPARSE_LDLT_BACKEND_LINKED_LIST,
        .used_csc_path = NULL,
        .progress_cb = progress_count_cb,
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
    SparseMatrix *A1 = build_tridiag_spd(n);
    SparseMatrix *A2 = build_tridiag_spd(n);
    REQUIRE_OK(A1 && A2 ? SPARSE_OK : SPARSE_ERR_ALLOC);

    ASSERT_EQ(sparse_lu_factor(A1, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    sparse_lu_opts_t opts = {
        .pivot = SPARSE_PIVOT_PARTIAL,
        .reorder = SPARSE_REORDER_NONE,
        .tol = 1e-12,
        /* progress_cb = NULL (designated-init default) */
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

/* SPARSE_ERR_CANCELLED string round-trips through sparse_strerror. */
static void test_progress_cb_strerror(void) {
    const char *s = sparse_strerror(SPARSE_ERR_CANCELLED);
    ASSERT_TRUE(s != NULL);
    ASSERT_TRUE(strstr(s, "cancel") != NULL || strstr(s, "Cancel") != NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Integration Tests");

    RUN_TEST(test_load_factor_solve_save);
    RUN_TEST(test_create_copy_factor_refine);
    RUN_TEST(test_multiple_rhs_same_factorization);
    RUN_TEST(test_full_roundtrip);
    RUN_TEST(test_all_reference_matrices);
    RUN_TEST(test_both_pivots_agree_integration);
    RUN_TEST(test_error_recovery);

    /* Sprint 29 Day 6: progress / cancel callbacks (Item 4). */
    RUN_TEST(test_progress_cb_lu_emits);
    RUN_TEST(test_progress_cb_lu_cancel);
    RUN_TEST(test_progress_cb_cholesky_emits_cancel);
    RUN_TEST(test_progress_cb_ldlt_emits_cancel);
    RUN_TEST(test_progress_cb_null_default_unchanged);
    RUN_TEST(test_progress_cb_strerror);

    TEST_SUITE_END();
}
