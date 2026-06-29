/* _POSIX_C_SOURCE 200809L: needed for `setenv` / `unsetenv` used by
 * the `tf_setenv` / `tf_unsetenv` macros from test_framework.h.
 * Must be defined BEFORE any system header is included so glibc's
 * `<features.h>` sees it on first inclusion. */
#if !defined(_WIN32) && (!defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L)
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define _POSIX_C_SOURCE 200809L
#endif
/*
 * Supernodal and writeback proof owner for Cholesky CSC. Core CSC conversion,
 * scalar elimination, solve, and dispatch tests remain in `test_chol_csc.c`.
 */

#include "sparse_analysis.h"
#include "sparse_chol_csc_internal.h"
#include "sparse_cholesky.h"
#include "sparse_ldlt_csc_internal.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_chol_csc_supernodal_helpers.h"

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Supernode detection
 * ═══════════════════════════════════════════════════════════════════════ */

/* ─── Null / arg validation ────────────────────────────────────── */

static void test_detect_supernodes_null_args(void) {
    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_alloc(3, 3, &L));
    idx_t starts[3], sizes[3], count;
    ASSERT_ERR(chol_csc_detect_supernodes(NULL, 4, starts, sizes, &count), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_detect_supernodes(L, 4, NULL, sizes, &count), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_detect_supernodes(L, 4, starts, NULL, &count), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_detect_supernodes(L, 4, starts, sizes, NULL), SPARSE_ERR_NULL);
    /* min_size must be >= 1. */
    ASSERT_ERR(chol_csc_detect_supernodes(L, 0, starts, sizes, &count), SPARSE_ERR_BADARG);
    ASSERT_ERR(chol_csc_detect_supernodes(L, -1, starts, sizes, &count), SPARSE_ERR_BADARG);
    chol_csc_free(L);
}

/* ─── Canonical structures ─────────────────────────────────────── */

/* Diagonal matrix: every column has one entry (its diagonal), so no
 * column-pair satisfies the supernode condition. */
static void test_detect_supernodes_diagonal(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0 + (double)i);
    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &L));

    idx_t *starts, *sizes, count;
    /* min_size = 4 → no supernodes of that size. */
    detect_supernodes_alloc(L, 4, &starts, &sizes, &count);
    ASSERT_EQ(count, 0);
    free(starts);
    free(sizes);

    /* min_size = 1 → every column is a size-1 fundamental supernode. */
    detect_supernodes_alloc(L, 1, &starts, &sizes, &count);
    ASSERT_EQ(count, n);
    for (idx_t i = 0; i < count; i++) {
        ASSERT_EQ(starts[i], i);
        ASSERT_EQ(sizes[i], 1);
    }
    free(starts);
    free(sizes);

    chol_csc_free(L);
    sparse_free(A);
}

/* Dense n x n SPD: a single supernode covering all columns. */
static void test_detect_supernodes_dense(void) {
    idx_t n = 8;
    /* A = I + e*e^T → SPD with diagonal n+1 and off-diagonal 1 (dense). */
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, (i == j) ? (double)(n + 1) : 1.0);

    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &L));

    idx_t *starts, *sizes, count;
    detect_supernodes_alloc(L, 4, &starts, &sizes, &count);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(starts[0], 0);
    ASSERT_EQ(sizes[0], n);
    free(starts);
    free(sizes);

    chol_csc_free(L);
    sparse_free(A);
}

/* Block-diagonal with two 5x5 dense SPD blocks — expect two size-5
 * supernodes. */
static void test_detect_supernodes_block_diagonal(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t b = 0; b < 2; b++) {
        idx_t o = b * 5;
        for (idx_t i = 0; i < 5; i++)
            for (idx_t j = 0; j < 5; j++)
                sparse_insert(A, o + i, o + j, (i == j) ? 6.0 : 1.0);
    }

    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &L));

    idx_t *starts, *sizes, count;
    detect_supernodes_alloc(L, 4, &starts, &sizes, &count);
    ASSERT_EQ(count, 2);
    ASSERT_EQ(starts[0], 0);
    ASSERT_EQ(sizes[0], 5);
    ASSERT_EQ(starts[1], 5);
    ASSERT_EQ(sizes[1], 5);
    free(starts);
    free(sizes);

    chol_csc_free(L);
    sparse_free(A);
}

/* Tridiagonal SPD: the inner columns all have structure {j, j+1} which
 * is a different pattern from {j+1, j+2}, so they do *not* merge.  The
 * last two columns are a special case — col n-2 has pattern {n-2, n-1}
 * and col n-1 has pattern {n-1}, satisfying the supernode invariant
 * (col n-1 size is one less, and the empty tail trivially matches).  So
 * a tridiagonal L yields exactly one size-2 supernode at columns
 * {n-2, n-1} and n-2 singleton supernodes in front. */
static void test_detect_supernodes_tridiagonal(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &L));

    /* min_size = 4 → size-2 trailing supernode doesn't meet threshold. */
    idx_t *starts, *sizes, count;
    detect_supernodes_alloc(L, 4, &starts, &sizes, &count);
    ASSERT_EQ(count, 0);
    free(starts);
    free(sizes);

    /* min_size = 2 → exactly the trailing {n-2, n-1} supernode. */
    detect_supernodes_alloc(L, 2, &starts, &sizes, &count);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(starts[0], n - 2);
    ASSERT_EQ(sizes[0], 2);
    free(starts);
    free(sizes);

    /* min_size = 1 → (n-2) singletons + the trailing size-2 block. */
    detect_supernodes_alloc(L, 1, &starts, &sizes, &count);
    ASSERT_EQ(count, n - 1);
    for (idx_t i = 0; i < n - 2; i++) {
        ASSERT_EQ(starts[i], i);
        ASSERT_EQ(sizes[i], 1);
    }
    ASSERT_EQ(starts[n - 2], n - 2);
    ASSERT_EQ(sizes[n - 2], 2);
    free(starts);
    free(sizes);

    chol_csc_free(L);
    sparse_free(A);
}

/* Reverse arrowhead: dense column 0 causes fill into columns 1, 2, 3
 * so the trailing columns form a dense supernode chain. */
static void test_detect_supernodes_reverse_arrowhead(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 10.0);
    for (idx_t i = 1; i < n; i++) {
        sparse_insert(A, i, 0, 1.0 + 0.1 * (double)i);
        sparse_insert(A, 0, i, 1.0 + 0.1 * (double)i);
        sparse_insert(A, i, i, 1.0);
    }

    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &L));

    /* After fill, columns 0-4 all have structure {j, j+1, ..., n-1},
     * which is the canonical dense-supernode pattern. */
    idx_t *starts, *sizes, count;
    detect_supernodes_alloc(L, 1, &starts, &sizes, &count);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(starts[0], 0);
    ASSERT_EQ(sizes[0], n);
    free(starts);
    free(sizes);

    chol_csc_free(L);
    sparse_free(A);
}

/* ─── SuiteSparse inspection ──────────────────────────────────── */

/* Factor nos4 and bcsstk04, run supernode detection, and print the
 * partition.  The expected counts aren't precisely predictable (they
 * depend on AMD's output ordering), but they should be non-zero and
 * in a reasonable range relative to n. */
static void test_detect_supernodes_suitesparse_report(void) {
    const char *mtx_paths[2] = {SS_DIR "/nos4.mtx", SS_DIR "/bcsstk04.mtx"};
    for (int mi = 0; mi < 2; mi++) {
        SparseMatrix *A = NULL;
        REQUIRE_OK(sparse_load_mm(&A, mtx_paths[mi]));

        sparse_analysis_opts_t opts = {
            .factor_type = SPARSE_FACTOR_CHOLESKY,
            .reorder = SPARSE_REORDER_AMD,
        };
        sparse_analysis_t an = {0};
        REQUIRE_OK(sparse_analyze(A, &opts, &an));

        CholCsc *L = NULL;
        REQUIRE_OK(chol_csc_factor(A, &an, &L));

        idx_t *starts, *sizes, count;
        detect_supernodes_alloc(L, 4, &starts, &sizes, &count);

        idx_t total_cols_in_supernodes = 0;
        idx_t max_size = 0;
        for (idx_t i = 0; i < count; i++) {
            total_cols_in_supernodes += sizes[i];
            if (sizes[i] > max_size)
                max_size = sizes[i];
        }
        printf("    %s (AMD): n=%d nnz(L)=%d supernodes=%d max_size=%d cols_in_super=%d\n",
               mtx_paths[mi], (int)L->n, (int)L->nnz, (int)count, (int)max_size,
               (int)total_cols_in_supernodes);

        /* Non-trivial structural matrices should yield at least one
         * size-4+ supernode. */
        ASSERT_TRUE(count >= 1);
        /* Sanity: supernodes never extend past n. */
        for (idx_t i = 0; i < count; i++)
            ASSERT_TRUE(starts[i] + sizes[i] <= L->n);
        /* Sanity: supernodes are strictly ascending. */
        for (idx_t i = 1; i < count; i++)
            ASSERT_TRUE(starts[i] > starts[i - 1]);

        free(starts);
        free(sizes);
        chol_csc_free(L);
        sparse_analysis_free(&an);
        sparse_free(A);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Supernodal-etree reordering: numeric correctness and corpus-safety
 * contracts
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The tests cover the Liu 1990 / Davis 2006 section 6.5 contract: compose
 * etree postorder into `analysis->perm`, rebuild B, recompute etree/postorder,
 * and preserve numeric solve quality and supernode contiguity on corpus inputs.
 */

/* Assert numeric factorization residuals stay within the established 1e-8
 * envelope after the supernode-grouping permutation. The off path is the
 * baseline; the on path proves the etree-postorder composition does not
 * corrupt numeric factorization. */
static void test_supernodal_postorder_residual_unchanged(void) {
    /* bcsstk04: small SuiteSparse SPD with irregular structure (matches
     * real-matrix behaviour while staying cheap for CI. AMD-permuted
     * factorization has a known well-conditioned residual on this fixture. */
    SparseMatrix *A = NULL;
    sparse_err_t rc = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    if (rc != SPARSE_OK) {
        printf("    skipped (bcsstk04 fixture not loadable: %d)\n", (int)rc);
        return;
    }
    idx_t n = sparse_rows(A);

    double *x_true = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x_off = calloc((size_t)n, sizeof(double));
    double *x_on = calloc((size_t)n, sizeof(double));
    if (!x_true || !b || !x_off || !x_on) {
        free(x_true);
        free(b);
        free(x_off);
        free(x_on);
        sparse_free(A);
        REQUIRE_OK(SPARSE_ERR_ALLOC);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_true[i] = 1.0;
    sparse_matvec(A, x_true, b);

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_analysis_t an_off = {0};
    sparse_factors_t fa_off = {0};
    sparse_analysis_t an_on = {0};
    sparse_factors_t fa_on = {0};
    int env_set = 0;
    double res_off = 0.0;
    double res_on = 0.0;

    /* Off path — explicit rc handling so the cleanup label always
     * runs and unsetenv fires once env_set is true. */
    tf_unsetenv("SPARSE_SUPERNODAL_POSTORDER");
    rc = sparse_analyze(A, &opts, &an_off);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_analyze (env off): rc=%d", (int)rc);
        goto cleanup;
    }
    rc = sparse_factor_numeric(A, &an_off, &fa_off);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_factor_numeric (env off): rc=%d", (int)rc);
        goto cleanup;
    }
    rc = sparse_factor_solve(&fa_off, &an_off, b, x_off);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_factor_solve (env off): rc=%d", (int)rc);
        goto cleanup;
    }
    res_off = compute_rel_residual(A, x_off, b);

    /* On path */
    if (tf_setenv("SPARSE_SUPERNODAL_POSTORDER", "on") != 0) {
        printf("    skipped (setenv SPARSE_SUPERNODAL_POSTORDER failed)\n");
        goto cleanup;
    }
    env_set = 1;
    rc = sparse_analyze(A, &opts, &an_on);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_analyze (env on): rc=%d", (int)rc);
        goto cleanup;
    }
    rc = sparse_factor_numeric(A, &an_on, &fa_on);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_factor_numeric (env on): rc=%d", (int)rc);
        goto cleanup;
    }
    rc = sparse_factor_solve(&fa_on, &an_on, b, x_on);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_factor_solve (env on): rc=%d", (int)rc);
        goto cleanup;
    }
    res_on = compute_rel_residual(A, x_on, b);

    printf("    bcsstk04: residual off=%.3e on=%.3e\n", res_off, res_on);

    /* Both paths must remain inside the 1e-8 envelope. In practice both
     * deliver ~1e-15 on bcsstk04, leaving room for floating-point
     * reordering effects from the postorder composition. */
    ASSERT_TRUE(res_off < 1e-8);
    ASSERT_TRUE(res_on < 1e-8);

cleanup:
    if (env_set)
        tf_unsetenv("SPARSE_SUPERNODAL_POSTORDER");
    sparse_factor_free(&fa_off);
    sparse_factor_free(&fa_on);
    sparse_analysis_free(&an_off);
    sparse_analysis_free(&an_on);
    free(x_true);
    free(b);
    free(x_off);
    free(x_on);
    sparse_free(A);
}

/* Corpus-safety: supernode total_grouped stays within a 25% band of the
 * env-off baseline across the corpus. AMD output is already approximately
 * etree-postordered on most fixtures, so the composition should not move the
 * supernode structure meaningfully. The generous band absorbs small-fixture
 * noise; the durable contract is that the post-pass preserves contiguity. */
static void test_supernodal_postorder_no_supernode_count_regression(void) {
    /* bcsstk14 is large enough for non-trivial supernode structure while
     * remaining small enough for routine CI coverage. */
    SparseMatrix *A = NULL;
    sparse_err_t rc = sparse_load_mm(&A, SS_DIR "/bcsstk14.mtx");
    if (rc != SPARSE_OK) {
        printf("    skipped (bcsstk14 fixture not loadable: %d)\n", (int)rc);
        return;
    }

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_analysis_t an_off = {0};
    sparse_factors_t fa_off = {0};
    sparse_analysis_t an_on = {0};
    sparse_factors_t fa_on = {0};
    int env_set = 0;
    idx_t count_off = 0;
    idx_t count_on = 0;
    idx_t total_off = -1;
    idx_t total_on = -1;

    tf_unsetenv("SPARSE_SUPERNODAL_POSTORDER");
    rc = sparse_analyze(A, &opts, &an_off);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_analyze (env off): rc=%d", (int)rc);
        goto cleanup;
    }
    rc = sparse_factor_numeric(A, &an_off, &fa_off);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_factor_numeric (env off): rc=%d", (int)rc);
        goto cleanup;
    }
    total_off = count_grouped_supernode_columns(fa_off.F, /*min_size=*/4, &count_off);

    if (tf_setenv("SPARSE_SUPERNODAL_POSTORDER", "on") != 0) {
        printf("    skipped (setenv SPARSE_SUPERNODAL_POSTORDER failed)\n");
        goto cleanup;
    }
    env_set = 1;
    rc = sparse_analyze(A, &opts, &an_on);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_analyze (env on): rc=%d", (int)rc);
        goto cleanup;
    }
    rc = sparse_factor_numeric(A, &an_on, &fa_on);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_factor_numeric (env on): rc=%d", (int)rc);
        goto cleanup;
    }
    total_on = count_grouped_supernode_columns(fa_on.F, /*min_size=*/4, &count_on);

    printf("    bcsstk14: supernodes(min=4) off=(count=%d total=%d) on=(count=%d total=%d)\n",
           (int)count_off, (int)total_off, (int)count_on, (int)total_on);

    /* count_grouped_supernode_columns returns -1 on failure; treat as skip. */
    if (total_off < 0 || total_on < 0) {
        printf("    skipped (chol_csc conversion failed)\n");
    } else if (total_off == 0) {
        /* No supernodes met min_size on the off-baseline (this won't
         * happen on bcsstk14 today — total_off=1246 — but the test
         * would be ill-defined under a future library update that
         * shifts the supernode threshold, so guard explicitly).
         * Require total_on == 0 too: env=on cannot legitimately
         * create supernodes when the baseline has none under the
         * same min_size gate. */
        ASSERT_EQ(total_on, 0);
    } else {
        /* 25 % band: |total_on - total_off| <= 0.25 * total_off.
         * The band absorbs measurement variability if the AMD permutation
         * changes across library versions. */
        idx_t delta = total_on > total_off ? total_on - total_off : total_off - total_on;
        ASSERT_TRUE((long long)delta * 4 <= (long long)total_off);
    }

cleanup:
    if (env_set)
        tf_unsetenv("SPARSE_SUPERNODAL_POSTORDER");
    sparse_factor_free(&fa_off);
    sparse_factor_free(&fa_on);
    sparse_analysis_free(&an_off);
    sparse_analysis_free(&an_on);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Dense primitives and supernode-aware elimination
 * ═══════════════════════════════════════════════════════════════════════ */

/* ─── chol_dense_factor: null / badarg ─────────────────────────── */

static void test_chol_dense_factor_null(void) {
    ASSERT_ERR(chol_dense_factor(NULL, 3, 3, 0.0), SPARSE_ERR_NULL);
    double A[4] = {1.0, 0.0, 0.0, 1.0};
    ASSERT_ERR(chol_dense_factor(A, -1, 2, 0.0), SPARSE_ERR_BADARG);
    ASSERT_ERR(chol_dense_factor(A, 2, 1, 0.0), SPARSE_ERR_BADARG); /* lda<n */
    /* n == 0 is a valid no-op. */
    REQUIRE_OK(chol_dense_factor(A, 0, 0, 0.0));
}

/* ─── chol_dense_factor: 1x1 and 2x2 hand-verified ────────────── */

static void test_chol_dense_factor_1x1(void) {
    double A[1] = {9.0};
    REQUIRE_OK(chol_dense_factor(A, 1, 1, 0.0));
    ASSERT_NEAR(A[0], 3.0, 1e-12);
}

static void test_chol_dense_factor_2x2(void) {
    /* A = [[4, 2], [2, 5]], column-major:
     *   A[0,0]=4, A[1,0]=2, A[0,1]=2, A[1,1]=5 → indices [0,1,2,3] = [4, 2, 2, 5]
     * L = [[2, 0], [1, 2]] → A[0,0]=2, A[1,0]=1, A[1,1]=2. */
    double A[4] = {4.0, 2.0, 2.0, 5.0};
    REQUIRE_OK(chol_dense_factor(A, 2, 2, 0.0));
    ASSERT_NEAR(A[0], 2.0, 1e-12); /* L[0,0] */
    ASSERT_NEAR(A[1], 1.0, 1e-12); /* L[1,0] */
    ASSERT_NEAR(A[3], 2.0, 1e-12); /* L[1,1] */
}

/* ─── chol_dense_factor: 4x4 SPD round-trip ────────────────────── */

static void test_chol_dense_factor_4x4(void) {
    /* A = I + ee^T with e = [1,1,1,1] → diagonal 2, off 1. */
    idx_t n = 4;
    double A[16];
    for (idx_t j = 0; j < n; j++)
        for (idx_t i = 0; i < n; i++)
            A[i + j * n] = (i == j) ? 2.0 : 1.0;

    /* Keep a copy for L*L^T verification. */
    double A_orig[16];
    memcpy(A_orig, A, sizeof(A));

    REQUIRE_OK(chol_dense_factor(A, n, n, 0.0));

    /* Verify L*L^T matches A_orig's lower triangle. */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double sum = 0.0;
            for (idx_t k = 0; k <= j; k++)
                sum += A[i + k * n] * A[j + k * n];
            ASSERT_NEAR(sum, A_orig[i + j * n], 1e-12);
        }
    }
}

/* ─── chol_dense_factor: non-SPD detection ─────────────────────── */

static void test_chol_dense_factor_not_spd(void) {
    /* [[-1, 0], [0, 1]] — first diagonal is negative. */
    double A[4] = {-1.0, 0.0, 0.0, 1.0};
    ASSERT_ERR(chol_dense_factor(A, 2, 2, 0.0), SPARSE_ERR_NOT_SPD);

    /* [[1, 2], [2, 1]] — Schur complement becomes negative. */
    double B[4] = {1.0, 2.0, 2.0, 1.0};
    ASSERT_ERR(chol_dense_factor(B, 2, 2, 0.0), SPARSE_ERR_NOT_SPD);
}

/* ─── chol_dense_solve_lower: null / badarg ─────────────────── */

static void test_chol_dense_solve_null(void) {
    double L[4] = {1.0, 0.0, 0.0, 1.0};
    double b[2] = {1.0, 2.0};
    ASSERT_ERR(chol_dense_solve_lower(NULL, 2, 2, b), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_dense_solve_lower(L, 2, 2, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_dense_solve_lower(L, -1, 2, b), SPARSE_ERR_BADARG);
    ASSERT_ERR(chol_dense_solve_lower(L, 2, 1, b), SPARSE_ERR_BADARG);
    /* n == 0 is a no-op. */
    REQUIRE_OK(chol_dense_solve_lower(L, 0, 0, b));
}

/* Forward substitution on a hand-verified 3x3 lower-triangular matrix. */
static void test_chol_dense_solve_lower_3x3(void) {
    /* L = [[2, 0, 0], [1, 3, 0], [0, 2, 4]], column-major.
     * b = [2, 7, 18] → x = [1, 2, 3] (check: L*x = [2*1, 1*1+3*2, 2*2+4*3] = [2,7,16]).
     * Wait: L[2, 1] = 2 and L[2, 2] = 4, so L*x = [2*1, 1*1+3*2, 0*1+2*2+4*3] = [2, 7, 16].
     * Let me recompute: for x = [1, 2, 3], b_0 = L[0,0]*x[0] = 2*1 = 2.
     *   b_1 = L[1,0]*x[0] + L[1,1]*x[1] = 1*1 + 3*2 = 7.
     *   b_2 = L[2,0]*x[0] + L[2,1]*x[1] + L[2,2]*x[2] = 0*1 + 2*2 + 4*3 = 16.
     * So b = [2, 7, 16]. */
    double L[9] = {2.0, 1.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 4.0};
    double b[3] = {2.0, 7.0, 16.0};
    REQUIRE_OK(chol_dense_solve_lower(L, 3, 3, b));
    ASSERT_NEAR(b[0], 1.0, 1e-12);
    ASSERT_NEAR(b[1], 2.0, 1e-12);
    ASSERT_NEAR(b[2], 3.0, 1e-12);
}

static void test_chol_dense_solve_panel_2x2_two_rhs(void) {
    double L[4] = {2.0, 1.0, 0.0, 3.0};
    double panel[4] = {
        2.0,
        6.0,
        7.0,
        15.0,
    };
    REQUIRE_OK(chol_dense_solve_panel(L, 2, 2, panel, 2, 2));
    ASSERT_NEAR(panel[0], 1.0, 1e-12);
    ASSERT_NEAR(panel[1], 3.0, 1e-12);
    ASSERT_NEAR(panel[2], 2.0, 1e-12);
    ASSERT_NEAR(panel[3], 4.0, 1e-12);
}

static void test_supernodal_dense_backend_default_contract(void) {
    const chol_dense_kernels_t *kernels = chol_csc_supernodal_dense_kernels();
    ASSERT_NOT_NULL(kernels);
    if (!kernels)
        return;
    ASSERT_NOT_NULL(kernels->name);
    ASSERT_NOT_NULL(kernels->factor);
    ASSERT_NOT_NULL(kernels->solve_lower);
    ASSERT_NOT_NULL(kernels->solve_panel);
    if (!kernels->name || !kernels->factor || !kernels->solve_lower || !kernels->solve_panel)
        return;

    double A[4] = {4.0, 2.0, 2.0, 5.0};
    REQUIRE_OK(kernels->factor(A, 2, 2, 0.0));
    ASSERT_NEAR(A[0], 2.0, 1e-12);
    ASSERT_NEAR(A[1], 1.0, 1e-12);
    ASSERT_NEAR(A[3], 2.0, 1e-12);

    double b[2] = {2.0, 5.0};
    REQUIRE_OK(kernels->solve_lower(A, 2, 2, b));
    ASSERT_NEAR(b[0], 1.0, 1e-12);
    ASSERT_NEAR(b[1], 2.0, 1e-12);

    double panel[4] = {2.0, 6.0, 5.0, 11.0};
    REQUIRE_OK(kernels->solve_panel(A, 2, 2, panel, 2, 2));
    ASSERT_NEAR(panel[0], 1.0, 1e-12);
    ASSERT_NEAR(panel[1], 3.0, 1e-12);
    ASSERT_NEAR(panel[2], 2.0, 1e-12);
    ASSERT_NEAR(panel[3], 4.0, 1e-12);
}

static void test_supernodal_dense_backend_builtin_env_contract(void) {
    tf_unsetenv("SPARSE_CHOL_DENSE_BACKEND");
    if (tf_setenv("SPARSE_CHOL_DENSE_BACKEND", "builtin") != 0)
        SKIP_TEST("setenv SPARSE_CHOL_DENSE_BACKEND=builtin failed");

    const chol_dense_kernels_t *kernels = chol_csc_supernodal_dense_kernels();
    tf_unsetenv("SPARSE_CHOL_DENSE_BACKEND");

    ASSERT_NOT_NULL(kernels);
    if (!kernels)
        return;
    ASSERT_NOT_NULL(kernels->name);
    if (!kernels->name)
        return;
    ASSERT_TRUE(strcmp(kernels->name, "builtin") == 0);
}

static void test_supernodal_dense_backend_accelerate_env_contract(void) {
    tf_unsetenv("SPARSE_CHOL_DENSE_BACKEND");
    if (tf_setenv("SPARSE_CHOL_DENSE_BACKEND", "accelerate") != 0)
        SKIP_TEST("setenv SPARSE_CHOL_DENSE_BACKEND=accelerate failed");

    const chol_dense_kernels_t *kernels = chol_csc_supernodal_dense_kernels();
    tf_unsetenv("SPARSE_CHOL_DENSE_BACKEND");

    ASSERT_NOT_NULL(kernels);
    if (!kernels)
        return;
    ASSERT_NOT_NULL(kernels->name);
    ASSERT_NOT_NULL(kernels->factor);
    ASSERT_NOT_NULL(kernels->solve_lower);
    ASSERT_NOT_NULL(kernels->solve_panel);
    if (!kernels->name || !kernels->factor || !kernels->solve_lower || !kernels->solve_panel)
        return;

#ifdef __APPLE__
    ASSERT_TRUE(strcmp(kernels->name, "builtin") == 0 || strcmp(kernels->name, "accelerate") == 0);
    if (strcmp(kernels->name, "accelerate") == 0) {
        double A[4] = {4.0, 2.0, 2.0, 5.0};
        REQUIRE_OK(kernels->factor(A, 2, 2, 0.0));
        ASSERT_NEAR(A[0], 2.0, 1e-12);
        ASSERT_NEAR(A[1], 1.0, 1e-12);
        ASSERT_NEAR(A[3], 2.0, 1e-12);

        double b[2] = {2.0, 5.0};
        REQUIRE_OK(kernels->solve_lower(A, 2, 2, b));
        ASSERT_NEAR(b[0], 1.0, 1e-12);
        ASSERT_NEAR(b[1], 2.0, 1e-12);

        double panel[4] = {2.0, 6.0, 5.0, 11.0};
        REQUIRE_OK(kernels->solve_panel(A, 2, 2, panel, 2, 2));
        ASSERT_NEAR(panel[0], 1.0, 1e-12);
        ASSERT_NEAR(panel[1], 3.0, 1e-12);
        ASSERT_NEAR(panel[2], 2.0, 1e-12);
        ASSERT_NEAR(panel[3], 4.0, 1e-12);
    }
#else
    ASSERT_TRUE(strcmp(kernels->name, "builtin") == 0);
#endif
}

static void test_supernodal_dense_backend_external_env_contract(void) {
    tf_unsetenv("SPARSE_CHOL_DENSE_BACKEND");
    if (tf_setenv("SPARSE_CHOL_DENSE_BACKEND", "external") != 0)
        SKIP_TEST("setenv SPARSE_CHOL_DENSE_BACKEND=external failed");

    const chol_dense_kernels_t *kernels = chol_csc_supernodal_dense_kernels();
    tf_unsetenv("SPARSE_CHOL_DENSE_BACKEND");

    ASSERT_NOT_NULL(kernels);
    if (!kernels)
        return;
    ASSERT_NOT_NULL(kernels->name);
    ASSERT_NOT_NULL(kernels->factor);
    ASSERT_NOT_NULL(kernels->solve_lower);
    ASSERT_NOT_NULL(kernels->solve_panel);
    if (!kernels->name || !kernels->factor || !kernels->solve_lower || !kernels->solve_panel)
        return;

    ASSERT_TRUE(strcmp(kernels->name, "builtin") == 0 || strcmp(kernels->name, "accelerate") == 0 ||
                strcmp(kernels->name, "blas-lapack") == 0);
    if (strcmp(kernels->name, "builtin") != 0) {
        double A[4] = {4.0, 2.0, 2.0, 5.0};
        REQUIRE_OK(kernels->factor(A, 2, 2, 0.0));
        ASSERT_NEAR(A[0], 2.0, 1e-12);
        ASSERT_NEAR(A[1], 1.0, 1e-12);
        ASSERT_NEAR(A[3], 2.0, 1e-12);

        double b[2] = {2.0, 5.0};
        REQUIRE_OK(kernels->solve_lower(A, 2, 2, b));
        ASSERT_NEAR(b[0], 1.0, 1e-12);
        ASSERT_NEAR(b[1], 2.0, 1e-12);

        double panel[4] = {2.0, 6.0, 5.0, 11.0};
        REQUIRE_OK(kernels->solve_panel(A, 2, 2, panel, 2, 2));
        ASSERT_NEAR(panel[0], 1.0, 1e-12);
        ASSERT_NEAR(panel[1], 3.0, 1e-12);
        ASSERT_NEAR(panel[2], 2.0, 1e-12);
        ASSERT_NEAR(panel[3], 4.0, 1e-12);
    }
}

/* ─── ldlt_dense_factor (Bunch-Kaufman on column-major storage) ─── */

/* Reconstruct A from factored L, D, D_offdiag, pivot_size and check
 * it matches the original matrix to `tol`.  `A_before` holds the
 * original symmetric values (both triangles); `A_factored` holds L
 * below-diag + 1.0 on the diagonal after `ldlt_dense_factor`. */
static int ldlt_dense_reconstruction_matches(const double *A_before, const double *A_factored,
                                             const double *D, const double *D_offdiag,
                                             const idx_t *pivot_size, idx_t n, idx_t lda,
                                             double tol) {
    /* Build an explicit L and D*L^T product in fresh buffers so the
     * check is obvious.  L is unit lower triangular; D is block
     * diagonal with 1×1 or 2×2 blocks. */
    double *Lfull = calloc((size_t)(n * n), sizeof(double));
    double *DLt = calloc((size_t)(n * n), sizeof(double));
    double *LDLt = calloc((size_t)(n * n), sizeof(double));
    if (Lfull == NULL || DLt == NULL || LDLt == NULL) {
        free(Lfull);
        free(DLt);
        free(LDLt);
        return 0;
    }
    int ok = 1;

    for (idx_t i = 0; i < n; i++) {
        Lfull[i + i * n] = 1.0;
        for (idx_t j = 0; j < i; j++)
            Lfull[i + j * n] = A_factored[i + j * lda];
    }

    /* D * L^T: for each column t of L^T (i.e., row t of L):
     *   (DLt)[k, t] = sum over pivot block at k of D[k..]*Lfull[t, ..]
     * Handle 1×1 and 2×2 blocks separately. */
    for (idx_t t = 0; t < n; t++) {
        idx_t k = 0;
        while (k < n) {
            if (pivot_size[k] == 1) {
                DLt[k + t * n] = D[k] * Lfull[t + k * n];
                k++;
            } else {
                double l_t_k = Lfull[t + k * n];
                double l_t_k1 = Lfull[t + (k + 1) * n];
                DLt[k + t * n] = D[k] * l_t_k + D_offdiag[k] * l_t_k1;
                DLt[(k + 1) + t * n] = D_offdiag[k] * l_t_k + D[k + 1] * l_t_k1;
                k += 2;
            }
        }
    }

    /* L * (D * L^T). */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            double s = 0.0;
            for (idx_t p = 0; p < n; p++)
                s += Lfull[i + p * n] * DLt[p + j * n];
            LDLt[i + j * n] = s;
        }
    }

    /* Compare against A_before on the lower triangle. */
    for (idx_t i = 0; i < n && ok; i++) {
        for (idx_t j = 0; j <= i && ok; j++) {
            double want = A_before[i + j * lda];
            double got = LDLt[i + j * n];
            if (fabs(want - got) > tol)
                ok = 0;
        }
    }

    free(Lfull);
    free(DLt);
    free(LDLt);
    return ok;
}

/* Null / shape checks. */
static void test_ldlt_dense_factor_arg_checks(void) {
    double A[4] = {1, 0, 0, 1};
    double D[2] = {0}, Doff[2] = {0};
    idx_t ps[2] = {0};
    ASSERT_ERR(ldlt_dense_factor(NULL, D, Doff, ps, 2, 2, 0.0, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_dense_factor(A, NULL, Doff, ps, 2, 2, 0.0, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_dense_factor(A, D, NULL, ps, 2, 2, 0.0, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_dense_factor(A, D, Doff, NULL, 2, 2, 0.0, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(ldlt_dense_factor(A, D, Doff, ps, -1, 2, 0.0, NULL), SPARSE_ERR_BADARG);
    ASSERT_ERR(ldlt_dense_factor(A, D, Doff, ps, 2, 1, 0.0, NULL), SPARSE_ERR_BADARG); /* lda<n */
}

/* 4×4 indefinite (diagonal-dominant): factor, then reconstruct A and
 * check round-off.  All 1×1 pivots expected because diag dominance
 * ensures criterion 1. */
static void test_ldlt_dense_factor_4x4_indefinite(void) {
    /* Mix of positive and negative diagonals + modest off-diagonals. */
    double A_init[16] = {4.0, 0.5, 0.3, 0.1, 0.5, -3.0, 0.2, 0.4,
                         0.3, 0.2, 5.0, 0.6, 0.1, 0.4,  0.6, -2.0};
    double A[16];
    memcpy(A, A_init, sizeof(A));

    double D[4] = {0}, Doff[4] = {0};
    idx_t ps[4] = {0};
    double growth = 0.0;
    REQUIRE_OK(ldlt_dense_factor(A, D, Doff, ps, 4, 4, 1e-12, &growth));

    ASSERT_TRUE(ldlt_dense_reconstruction_matches(A_init, A, D, Doff, ps, 4, 4, 1e-10));
    ASSERT_TRUE(growth < 10.0); /* Diagonal-dominant — no large L entries. */
}

/* 2×2 forced: small diagonals + large off-diagonal triggers criterion 4. */
static void test_ldlt_dense_factor_2x2_forced(void) {
    /* A = [[0.1, 1.0], [1.0, 0.3]].  |A[0,0]| = 0.1 < α * 1.0 = 0.64,
     * so criterion 1 fails.  Criterion 2: |A[0,0]| * σ_r = 0.1 * 0 = 0
     * (n=2, sigma_r has no other rows), comparison is 0 >= α * 1.0²
     * = 0.64 → false.  Criterion 3: |A[1,1]| = 0.3 >= α * 0 = 0 → true.
     *
     * Hmm — n=2 with only one off-diagonal, σ_r is 0, so criterion 3
     * would fire (swap-and-1×1).  Use n=3 to actually force a 2×2. */
    double A_init[9] = {0.1, 1.0, 0.2, 1.0, 0.3, 0.1, 0.2, 0.1, 4.0};
    double A[9];
    memcpy(A, A_init, sizeof(A));

    double D[3] = {0}, Doff[3] = {0};
    idx_t ps[3] = {0};
    double growth = 0.0;
    REQUIRE_OK(ldlt_dense_factor(A, D, Doff, ps, 3, 3, 1e-12, &growth));

    ASSERT_EQ(ps[0], 2);
    ASSERT_EQ(ps[1], 2);
    ASSERT_TRUE(fabs(Doff[0]) > 1e-10);
    ASSERT_NEAR(A[1], 0.0, 0.0);
    ASSERT_NEAR(A[3], 0.0, 0.0);
    ASSERT_TRUE(ldlt_dense_reconstruction_matches(A_init, A, D, Doff, ps, 3, 3, 1e-10));
}

/* 6×6 with mixed 1×1 and 2×2 pivots; verify reconstruction and
 * bounded growth. */
static void test_ldlt_dense_factor_6x6_mixed_pivots(void) {
    idx_t n = 6;
    double A_init[36];
    /* Start with diag-dominant and then perturb to trigger a 2×2 */
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++) {
            if (i == j)
                A_init[i + j * n] = (i % 2) ? -3.0 : 5.0;
            else
                A_init[i + j * n] = 0.1 * (double)((i + 1) * (j + 2) % 7);
        }
    /* Force a 2×2 by making the (2, 2) diagonal tiny vs its (2, 3) coupling. */
    A_init[2 + 2 * n] = 0.1;
    A_init[3 + 3 * n] = 0.2;
    A_init[2 + 3 * n] = 1.0;
    A_init[3 + 2 * n] = 1.0;
    /* Ensure symmetry. */
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = i + 1; j < n; j++)
            A_init[i + j * n] = A_init[j + i * n];

    double A[36];
    memcpy(A, A_init, sizeof(A));

    double D[6] = {0}, Doff[6] = {0};
    idx_t ps[6] = {0};
    double growth = 0.0;
    REQUIRE_OK(ldlt_dense_factor(A, D, Doff, ps, n, n, 1e-12, &growth));

    /* Verify we got a 2×2 pivot SOMEWHERE (i.e., at least one ps[k]==2 pair). */
    int has_2x2 = 0;
    for (idx_t k = 0; k + 1 < n; k++) {
        if (ps[k] == 2 && ps[k + 1] == 2) {
            has_2x2 = 1;
            break;
        }
    }
    ASSERT_TRUE(has_2x2);

    ASSERT_TRUE(ldlt_dense_reconstruction_matches(A_init, A, D, Doff, ps, n, n, 1e-9));
    ASSERT_TRUE(growth < 100.0); /* BK bounds growth — sanity check. */
}

/* ─── chol_csc_eliminate_supernodal: dispatch & correctness ─── */

/* On a dense SPD matrix the supernodal path must produce the same
 * factored L as the scalar path. */
static void test_eliminate_supernodal_dense(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, (i == j) ? (double)(n + 1) : 1.0);

    /* Scalar path. */
    CholCsc *L_scalar = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &L_scalar));
    REQUIRE_OK(chol_csc_eliminate(L_scalar));

    /* Supernodal path. */
    CholCsc *L_super = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &L_super));
    REQUIRE_OK(chol_csc_eliminate_supernodal(L_super, 4));
    REQUIRE_OK(chol_csc_validate(L_super));

    /* Structural and numeric equality. */
    ASSERT_EQ(L_scalar->nnz, L_super->nnz);
    for (idx_t j = 0; j <= n; j++)
        ASSERT_EQ(L_scalar->col_ptr[j], L_super->col_ptr[j]);
    for (idx_t p = 0; p < L_scalar->nnz; p++) {
        ASSERT_EQ(L_scalar->row_idx[p], L_super->row_idx[p]);
        ASSERT_NEAR(L_scalar->values[p], L_super->values[p], 1e-12);
    }

    chol_csc_free(L_scalar);
    chol_csc_free(L_super);
    sparse_free(A);
}

/* Block-diagonal SPD: residuals identical to scalar path. */
static void test_eliminate_supernodal_block_diagonal(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t b = 0; b < 2; b++) {
        idx_t o = b * 5;
        for (idx_t i = 0; i < 5; i++)
            for (idx_t j = 0; j < 5; j++)
                sparse_insert(A, o + i, o + j, (i == j) ? 6.0 : 1.0);
    }

    double *x_true = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x_sc = calloc((size_t)n, sizeof(double));
    double *x_sn = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_true[i] = 1.0 + (double)i;
    sparse_matvec(A, x_true, b);

    /* Scalar solve. */
    CholCsc *Ls = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &Ls));
    REQUIRE_OK(chol_csc_eliminate(Ls));
    REQUIRE_OK(chol_csc_solve(Ls, b, x_sc));

    /* Supernodal solve. */
    CholCsc *Ln = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &Ln));
    REQUIRE_OK(chol_csc_eliminate_supernodal(Ln, 4));
    REQUIRE_OK(chol_csc_solve(Ln, b, x_sn));

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_sc[i], x_sn[i], 1e-12);

    free(x_true);
    free(b);
    free(x_sc);
    free(x_sn);
    chol_csc_free(Ls);
    chol_csc_free(Ln);
    sparse_free(A);
}

/* SuiteSparse bcsstk04 (AMD): residual via the supernodal path should
 * match the scalar path exactly. */
static void test_eliminate_supernodal_bcsstk04_amd(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx"));
    idx_t n = sparse_rows(A);

    double *x_true = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_true[i] = 1.0;
    sparse_matvec(A, x_true, b);

    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &opts, &an));

    /* Scalar. */
    CholCsc *Ls = NULL;
    REQUIRE_OK(chol_csc_from_sparse_with_analysis(A, &an, &Ls));
    REQUIRE_OK(chol_csc_eliminate(Ls));
    double *x_sc = calloc((size_t)n, sizeof(double));
    REQUIRE_OK(chol_csc_solve_perm(Ls, an.perm, b, x_sc));

    /* Supernodal. */
    CholCsc *Ln = NULL;
    REQUIRE_OK(chol_csc_from_sparse_with_analysis(A, &an, &Ln));
    REQUIRE_OK(chol_csc_eliminate_supernodal(Ln, 4));
    double *x_sn = calloc((size_t)n, sizeof(double));
    REQUIRE_OK(chol_csc_solve_perm(Ln, an.perm, b, x_sn));

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_sc[i], x_sn[i], 1e-12);

    free(x_true);
    free(b);
    free(x_sc);
    free(x_sn);
    chol_csc_free(Ls);
    chol_csc_free(Ln);
    sparse_analysis_free(&an);
    sparse_free(A);
}

/* Kuu regression check.
 *
 * The scalar CSC kernel once regressed on Kuu (n = 7102) because drop
 * handling spent most of its time in `_platform_memmove` inside
 * `chol_csc_gather`'s `shift_columns_right_of` path. The maintained contract
 * is the in-place write-and-zero-pad fast path: skip the memmove when the
 * pre-allocated slot fits the survivors and every survivor row is already in
 * the slot's row_idx, which `chol_csc_from_sparse_with_analysis` guarantees.
 *
 * The regression check here factors Kuu through `chol_csc_eliminate`
 * (scalar CSC) AND `chol_csc_eliminate_supernodal` and asserts both
 * paths produce solves matching the linked-list reference to the
 * 1e-10 SPD spot-check tolerance.  Exact pivot-level agreement is
 * checked indirectly via the x-vector match since the supernodal
 * path runs the same underlying CholCsc plumbing. */
static void test_chol_csc_kuu_scalar_no_regression(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/Kuu.mtx"));
    idx_t n = sparse_rows(A);

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x_sc = calloc((size_t)n, sizeof(double));
    double *x_sn = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(ones);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(x_sc);
    ASSERT_NOT_NULL(x_sn);

    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    sparse_analysis_opts_t aopts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_AMD,
    };
    sparse_analysis_t an = {0};
    REQUIRE_OK(sparse_analyze(A, &aopts, &an));

    /* Scalar path — the Kuu-regression target. */
    CholCsc *Lsc = NULL;
    REQUIRE_OK(chol_csc_from_sparse_with_analysis(A, &an, &Lsc));
    REQUIRE_OK(chol_csc_eliminate(Lsc));
    REQUIRE_OK(chol_csc_solve_perm(Lsc, an.perm, b, x_sc));

    /* Supernodal path — reference.  Should already be correct from
     * the supernodal CSC implementation. */
    CholCsc *Lsn = NULL;
    REQUIRE_OK(chol_csc_from_sparse_with_analysis(A, &an, &Lsn));
    /* Matches the supernodal min_size used in other test_chol_csc
     * tests (the SPARSE_CSC_SUPERNODE_MIN_SIZE dispatch constant
     * lives in `src/sparse_cholesky.c` and isn't in the public
     * header). */
    REQUIRE_OK(chol_csc_eliminate_supernodal(Lsn, 4));
    REQUIRE_OK(chol_csc_solve_perm(Lsn, an.perm, b, x_sn));

    /* Residual against the original A. */
    double *Ax = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(Ax);
    sparse_matvec(A, x_sc, Ax);
    double rmax = 0.0, bmax = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double r = fabs(Ax[i] - b[i]);
        if (r > rmax)
            rmax = r;
        double bi = fabs(b[i]);
        if (bi > bmax)
            bmax = bi;
    }
    double rel_scalar = bmax > 0.0 ? rmax / bmax : rmax;
    ASSERT_TRUE(rel_scalar < 1e-10);

    /* Cross-check scalar vs supernodal solutions agree to round-off. */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_sc[i], x_sn[i], 1e-9);
    }

    free(ones);
    free(b);
    free(x_sc);
    free(x_sn);
    free(Ax);
    chol_csc_free(Lsc);
    chol_csc_free(Lsn);
    sparse_analysis_free(&an);
    sparse_free(A);
}

/* Null / badarg checks for the supernodal dispatch. */
static void test_eliminate_supernodal_null(void) {
    ASSERT_ERR(chol_csc_eliminate_supernodal(NULL, 4), SPARSE_ERR_NULL);
    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_alloc(3, 3, &L));
    ASSERT_ERR(chol_csc_eliminate_supernodal(L, 0), SPARSE_ERR_BADARG);
    ASSERT_ERR(chol_csc_eliminate_supernodal(L, -1), SPARSE_ERR_BADARG);
    chol_csc_free(L);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Supernode extract / writeback round-trips
 * ═══════════════════════════════════════════════════════════════════════ */

/* Build a dense SPD CSC and round-trip it through
 * extract → writeback with an identity buffer copy.  After writeback
 * the CSC must validate and reproduce the original dense matrix via
 * `chol_csc_to_sparse`.  Values and structure must match bit-for-bit
 * since no dense factor or drop tolerance runs between the two ops. */
static void test_supernode_extract_writeback_dense(void) {
    idx_t n = 6;
    /* Dense SPD via A = I + e*e^T: diagonal n+1, off-diagonal 1.
     * A single supernode covers all n columns. */
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, (i == j) ? (double)(n + 1) : 1.0);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));

    /* Snapshot original CSC values to compare after writeback. */
    idx_t nnz = csc->col_ptr[csc->n];
    double *orig_values = malloc((size_t)nnz * sizeof(double));
    REQUIRE_OK(nnz > 0 ? SPARSE_OK : SPARSE_ERR_ALLOC);
    for (idx_t p = 0; p < nnz; p++)
        orig_values[p] = csc->values[p];

    /* Panel height of the single supernode. */
    idx_t panel_height = chol_csc_supernode_panel_height(csc, 0);
    ASSERT_EQ(panel_height, n);

    double *dense = calloc((size_t)(panel_height * n), sizeof(double));
    idx_t *row_map = calloc((size_t)panel_height, sizeof(idx_t));
    idx_t ph_out = 0;
    REQUIRE_OK(chol_csc_supernode_extract(csc, 0, n, dense, panel_height, row_map, &ph_out));
    ASSERT_EQ(ph_out, n);

    /* row_map[i] == i for a dense supernode covering all columns. */
    for (idx_t i = 0; i < panel_height; i++)
        ASSERT_EQ(row_map[i], i);

    /* Dense buffer should hold A's lower triangle in column-major layout:
     * diag = n+1, off-diag = 1, upper-triangle cells untouched (0 from
     * calloc). */
    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = j; i < n; i++) {
            double expected = (i == j) ? (double)(n + 1) : 1.0;
            ASSERT_TRUE(fabs(dense[i + j * panel_height] - expected) < 1e-15);
        }
    }

    /* Writeback: no dense mutation in between, so CSC should be
     * byte-identical afterwards. */
    REQUIRE_OK(
        chol_csc_supernode_writeback(csc, 0, n, dense, panel_height, row_map, panel_height, 0.0));
    REQUIRE_OK(chol_csc_validate(csc));
    for (idx_t p = 0; p < nnz; p++)
        ASSERT_TRUE(fabs(csc->values[p] - orig_values[p]) < 1e-15);

    free(dense);
    free(row_map);
    free(orig_values);
    chol_csc_free(csc);
    sparse_free(A);
}

/* Block-diagonal SPD: extract each supernode, round-trip each
 * independently, assert the second supernode's writeback does not
 * disturb the first's storage (and vice versa). */
static void test_supernode_extract_writeback_block_diagonal(void) {
    idx_t n = 6;
    /* Two 3×3 dense SPD blocks at (0..2) and (3..5). */
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t b = 0; b < 2; b++) {
        idx_t o = b * 3;
        for (idx_t i = 0; i < 3; i++)
            for (idx_t j = 0; j < 3; j++)
                sparse_insert(A, o + i, o + j, (i == j) ? 4.0 : 1.0);
    }

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));

    idx_t nnz = csc->col_ptr[csc->n];
    double *orig_values = malloc((size_t)nnz * sizeof(double));
    for (idx_t p = 0; p < nnz; p++)
        orig_values[p] = csc->values[p];

    /* Extract + writeback supernode 0 (cols 0..2, panel_height = 3). */
    idx_t ph0 = chol_csc_supernode_panel_height(csc, 0);
    ASSERT_EQ(ph0, 3);
    double *dense0 = calloc((size_t)(ph0 * 3), sizeof(double));
    idx_t *row_map0 = calloc((size_t)ph0, sizeof(idx_t));
    idx_t ph_out0 = 0;
    REQUIRE_OK(chol_csc_supernode_extract(csc, 0, 3, dense0, ph0, row_map0, &ph_out0));

    /* Supernode 1 (cols 3..5, panel_height = 3). */
    idx_t ph1 = chol_csc_supernode_panel_height(csc, 3);
    ASSERT_EQ(ph1, 3);
    double *dense1 = calloc((size_t)(ph1 * 3), sizeof(double));
    idx_t *row_map1 = calloc((size_t)ph1, sizeof(idx_t));
    idx_t ph_out1 = 0;
    REQUIRE_OK(chol_csc_supernode_extract(csc, 3, 3, dense1, ph1, row_map1, &ph_out1));

    /* row_map for supernode 1 is [3, 4, 5]. */
    ASSERT_EQ(row_map1[0], 3);
    ASSERT_EQ(row_map1[1], 4);
    ASSERT_EQ(row_map1[2], 5);

    /* Writeback each supernode.  Each touches only its own columns. */
    REQUIRE_OK(chol_csc_supernode_writeback(csc, 0, 3, dense0, ph0, row_map0, ph0, 0.0));
    REQUIRE_OK(chol_csc_supernode_writeback(csc, 3, 3, dense1, ph1, row_map1, ph1, 0.0));
    REQUIRE_OK(chol_csc_validate(csc));

    for (idx_t p = 0; p < nnz; p++)
        ASSERT_TRUE(fabs(csc->values[p] - orig_values[p]) < 1e-15);

    /* Mutate supernode 0's dense buffer, write back, then extract
     * supernode 1: the second supernode's stored values must still
     * match the originals (independence check). */
    for (idx_t p = 0; p < ph0 * 3; p++)
        dense0[p] = 99.0;
    REQUIRE_OK(chol_csc_supernode_writeback(csc, 0, 3, dense0, ph0, row_map0, ph0, 0.0));
    /* Re-extract supernode 1 — should be unchanged. */
    double *dense1_reread = calloc((size_t)(ph1 * 3), sizeof(double));
    idx_t *row_map1b = calloc((size_t)ph1, sizeof(idx_t));
    idx_t ph_out1b = 0;
    REQUIRE_OK(chol_csc_supernode_extract(csc, 3, 3, dense1_reread, ph1, row_map1b, &ph_out1b));
    for (idx_t p = 0; p < ph1 * 3; p++)
        ASSERT_TRUE(fabs(dense1_reread[p] - dense1[p]) < 1e-15);

    free(dense0);
    free(dense1);
    free(dense1_reread);
    free(row_map0);
    free(row_map1);
    free(row_map1b);
    free(orig_values);
    chol_csc_free(csc);
    sparse_free(A);
}

/* Supernode with below-panel rows: arrow-shaped matrix where the last
 * row touches every column.  After AMD, the factor has a supernode
 * at the trailing diagonal with a non-trivial below-panel.  Verify
 * extract+writeback are still identity there. */
static void test_supernode_extract_writeback_with_below_panel(void) {
    /* Construct a matrix directly whose CSC already has a supernode
     * with below-supernode rows.  Build an SPD matrix whose lower
     * triangle has:
     *   col 0: rows 0, 1, 2, 3 (diag + three panel rows)
     *   col 1: rows 1, 2, 3
     *   col 2: rows 2, 3
     *   col 3: row 3
     * Every column in [0, 2] shares panel row {3}.  With min_size=2
     * we can treat cols [0, 1] as a supernode of size 2, panel
     * height = 4 (rows 0, 1, 2, 3).
     *
     * The entries are chosen so A is SPD (diagonally dominant). */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    double entries[4][4] = {
        {10.0, 1.0, 1.0, 1.0}, /* row 0 */
        {1.0, 10.0, 1.0, 1.0}, /* row 1 */
        {1.0, 1.0, 10.0, 1.0}, /* row 2 */
        {1.0, 1.0, 1.0, 10.0}, /* row 3 */
    };
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, entries[i][j]);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));

    /* Supernode at (0, 1) with panel_height = 4. */
    idx_t s_start = 0, s_size = 2;
    idx_t ph = chol_csc_supernode_panel_height(csc, s_start);
    ASSERT_EQ(ph, n);
    double *dense = calloc((size_t)(ph * s_size), sizeof(double));
    idx_t *row_map = calloc((size_t)ph, sizeof(idx_t));
    idx_t ph_out = 0;
    REQUIRE_OK(chol_csc_supernode_extract(csc, s_start, s_size, dense, ph, row_map, &ph_out));
    ASSERT_EQ(ph_out, n);

    /* Verify the two-column supernode's dense layout.
     * Diagonal block (rows 0..1) × (cols 0..1):
     *   col 0: [A[0,0]=10, A[1,0]=1]
     *   col 1: [ignored upper, A[1,1]=10]
     * Panel (rows 2..3):
     *   col 0: [A[2,0]=1, A[3,0]=1]
     *   col 1: [A[2,1]=1, A[3,1]=1] */
    ASSERT_TRUE(fabs(dense[0 + 0 * ph] - 10.0) < 1e-15);
    ASSERT_TRUE(fabs(dense[1 + 0 * ph] - 1.0) < 1e-15);
    ASSERT_TRUE(fabs(dense[1 + 1 * ph] - 10.0) < 1e-15);
    ASSERT_TRUE(fabs(dense[2 + 0 * ph] - 1.0) < 1e-15);
    ASSERT_TRUE(fabs(dense[3 + 0 * ph] - 1.0) < 1e-15);
    ASSERT_TRUE(fabs(dense[2 + 1 * ph] - 1.0) < 1e-15);
    ASSERT_TRUE(fabs(dense[3 + 1 * ph] - 1.0) < 1e-15);

    /* Writeback: CSC should still validate and the supernode's entries
     * match their original values. */
    idx_t nnz = csc->col_ptr[csc->n];
    double *orig_values = malloc((size_t)nnz * sizeof(double));
    for (idx_t p = 0; p < nnz; p++)
        orig_values[p] = csc->values[p];
    REQUIRE_OK(chol_csc_supernode_writeback(csc, s_start, s_size, dense, ph, row_map, ph, 0.0));
    REQUIRE_OK(chol_csc_validate(csc));
    for (idx_t p = 0; p < nnz; p++)
        ASSERT_TRUE(fabs(csc->values[p] - orig_values[p]) < 1e-15);

    free(dense);
    free(row_map);
    free(orig_values);
    chol_csc_free(csc);
    sparse_free(A);
}

/* lda > panel_height (padded dense buffer): the extract must write
 * only to rows [0, panel_height) of each column; padding rows are left
 * untouched. */
static void test_supernode_extract_lda_padding(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, (i == j) ? (double)(n + 2) : 0.5);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));

    idx_t ph = chol_csc_supernode_panel_height(csc, 0);
    ASSERT_EQ(ph, n);

    idx_t lda = ph + 3; /* padding rows at [ph, lda) */
    double *dense = malloc((size_t)(lda * n) * sizeof(double));
    for (idx_t p = 0; p < lda * n; p++)
        dense[p] = -7.77; /* sentinel — untouched rows keep this */

    idx_t *row_map = calloc((size_t)ph, sizeof(idx_t));
    idx_t ph_out = 0;
    REQUIRE_OK(chol_csc_supernode_extract(csc, 0, n, dense, lda, row_map, &ph_out));
    ASSERT_EQ(ph_out, n);

    /* Padding rows must still hold the sentinel. */
    for (idx_t j = 0; j < n; j++)
        for (idx_t i = ph; i < lda; i++)
            ASSERT_TRUE(dense[i + j * lda] == -7.77);

    /* Lower-triangle entries correct; upper triangle still sentinel. */
    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = 0; i < j; i++)
            ASSERT_TRUE(dense[i + j * lda] == -7.77);
        double expected_diag = (double)(n + 2);
        ASSERT_TRUE(fabs(dense[j + j * lda] - expected_diag) < 1e-15);
        for (idx_t i = j + 1; i < n; i++)
            ASSERT_TRUE(fabs(dense[i + j * lda] - 0.5) < 1e-15);
    }

    /* Writeback under the same padded lda should still produce an
     * identity round-trip on the supernode. */
    idx_t nnz = csc->col_ptr[csc->n];
    double *orig_values = malloc((size_t)nnz * sizeof(double));
    for (idx_t p = 0; p < nnz; p++)
        orig_values[p] = csc->values[p];
    REQUIRE_OK(chol_csc_supernode_writeback(csc, 0, n, dense, lda, row_map, ph, 0.0));
    for (idx_t p = 0; p < nnz; p++)
        ASSERT_TRUE(fabs(csc->values[p] - orig_values[p]) < 1e-15);

    free(dense);
    free(row_map);
    free(orig_values);
    chol_csc_free(csc);
    sparse_free(A);
}

/* Null-arg / out-of-range / insufficient-lda error paths. */
static void test_supernode_extract_error_paths(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 2.0);
    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));

    double dense[16] = {0};
    idx_t row_map[4] = {0};
    idx_t ph_out = 0;

    ASSERT_ERR(chol_csc_supernode_extract(NULL, 0, 1, dense, 4, row_map, &ph_out), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_supernode_extract(csc, 0, 1, NULL, 4, row_map, &ph_out), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_supernode_extract(csc, 0, 1, dense, 4, NULL, &ph_out), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_supernode_extract(csc, 0, 1, dense, 4, row_map, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_supernode_extract(csc, -1, 1, dense, 4, row_map, &ph_out),
               SPARSE_ERR_BADARG);
    ASSERT_ERR(chol_csc_supernode_extract(csc, 0, 0, dense, 4, row_map, &ph_out),
               SPARSE_ERR_BADARG);
    /* s_start + s_size > n */
    ASSERT_ERR(chol_csc_supernode_extract(csc, 2, 3, dense, 4, row_map, &ph_out),
               SPARSE_ERR_BADARG);
    /* lda < panel_height (panel_height = 1 for a diagonal matrix) */
    ASSERT_ERR(chol_csc_supernode_extract(csc, 0, 1, dense, 0, row_map, &ph_out),
               SPARSE_ERR_BADARG);

    /* Writeback: same error paths for null / range / lda. */
    ASSERT_ERR(chol_csc_supernode_writeback(NULL, 0, 1, dense, 4, row_map, 1, 0.0),
               SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_supernode_writeback(csc, 0, 1, NULL, 4, row_map, 1, 0.0), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_supernode_writeback(csc, 0, 1, dense, 4, NULL, 1, 0.0), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_supernode_writeback(csc, -1, 1, dense, 4, row_map, 1, 0.0),
               SPARSE_ERR_BADARG);
    ASSERT_ERR(chol_csc_supernode_writeback(csc, 0, 0, dense, 4, row_map, 1, 0.0),
               SPARSE_ERR_BADARG);
    ASSERT_ERR(chol_csc_supernode_writeback(csc, 2, 3, dense, 4, row_map, 1, 0.0),
               SPARSE_ERR_BADARG);
    /* panel_height < s_size */
    ASSERT_ERR(chol_csc_supernode_writeback(csc, 0, 2, dense, 4, row_map, 1, 0.0),
               SPARSE_ERR_BADARG);

    chol_csc_free(csc);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Supernode diagonal-block factor
 * ═══════════════════════════════════════════════════════════════════════ */

/* Dense 8×8 SPD matrix.  The full matrix is a single supernode, so
 * s_start = 0 (no external cmod).  The helper's factored diagonal
 * block must match the scalar-kernel L factor exactly. */
static void test_supernode_eliminate_diag_dense_8x8(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, (i == j) ? (double)(n + 1) : 1.0);

    /* Pre-factor CSC (A's lower triangle).  This is what the helper
     * receives. */
    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));

    /* Reference factor via the scalar kernel. */
    CholCsc *ref_L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &ref_L));

    /* Extract the single supernode (panel empty when n = s_size). */
    idx_t ph = chol_csc_supernode_panel_height(csc, 0);
    ASSERT_EQ(ph, n);
    double *dense = calloc((size_t)(ph * n), sizeof(double));
    idx_t *row_map = calloc((size_t)ph, sizeof(idx_t));
    idx_t ph_out = 0;
    REQUIRE_OK(chol_csc_supernode_extract(csc, 0, n, dense, ph, row_map, &ph_out));

    /* Run the diag-factor helper.  No external cmod to apply
     * (s_start = 0), just the dense Cholesky on the whole block. */
    REQUIRE_OK(chol_csc_supernode_eliminate_diag(csc, 0, n, dense, ph, row_map, ph, 0.0));

    /* Compare lower triangle of dense to the reference L. */
    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = j; i < n; i++) {
            double ref = chol_csc_value_at(ref_L, i, j);
            double got = dense[i + j * ph];
            ASSERT_TRUE(fabs(got - ref) < 1e-12);
        }
    }

    free(dense);
    free(row_map);
    chol_csc_free(csc);
    chol_csc_free(ref_L);
    sparse_free(A);
}

/* Supernode starting at column 1 (prior col 0 is a size-1 "non-
 * supernode" column that contributes external cmod).  After the
 * helper runs, the factored diagonal block at [1, 5) must match the
 * scalar L's corresponding block. */
static void test_supernode_eliminate_diag_with_external_cmod(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    /* Row 0 / col 0: diagonal 2 plus A[1, 0] = 1 so col 0 contributes
     * to supernode's col 1 via cmod. */
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    /* Cols 1..4 form a dense SPD block (diag 10, off-diag 1). */
    for (idx_t i = 1; i < n; i++)
        for (idx_t j = 1; j < n; j++)
            sparse_insert(A, i, j, (i == j) ? 10.0 : 1.0);

    /* Scalar reference. */
    CholCsc *ref_L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &ref_L));

    /* Build a CSC that has col 0 already factored (so the supernode
     * helper's external cmod reads L[:, 0], not A[:, 0]).  We fake
     * this by overwriting col 0's values in csc with the scalar L
     * values — the structural pattern is the same. */
    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));
    for (idx_t p = csc->col_ptr[0]; p < csc->col_ptr[0 + 1]; p++) {
        idx_t r = csc->row_idx[p];
        csc->values[p] = chol_csc_value_at(ref_L, r, 0);
    }

    /* Extract supernode [1, 5). */
    idx_t s_start = 1, s_size = 4;
    idx_t ph = chol_csc_supernode_panel_height(csc, s_start);
    ASSERT_EQ(ph, s_size); /* No below-panel rows in this layout. */
    double *dense = calloc((size_t)(ph * s_size), sizeof(double));
    idx_t *row_map = calloc((size_t)ph, sizeof(idx_t));
    idx_t ph_out = 0;
    REQUIRE_OK(chol_csc_supernode_extract(csc, s_start, s_size, dense, ph, row_map, &ph_out));

    /* Run diag factor with external cmod. */
    REQUIRE_OK(
        chol_csc_supernode_eliminate_diag(csc, s_start, s_size, dense, ph, row_map, ph, 0.0));

    /* Compare the factored diagonal block to the reference L's
     * [1, 5) × [1, 5) slab. */
    for (idx_t j = 0; j < s_size; j++) {
        for (idx_t i = j; i < s_size; i++) {
            double ref = chol_csc_value_at(ref_L, s_start + i, s_start + j);
            double got = dense[i + j * ph];
            ASSERT_TRUE(fabs(got - ref) < 1e-12);
        }
    }

    free(dense);
    free(row_map);
    chol_csc_free(csc);
    chol_csc_free(ref_L);
    sparse_free(A);
}

/* Block-diagonal SPD (two 3×3 blocks): each block is a supernode,
 * and the blocks are independent — external cmod from col 0 into
 * supernode [3, 6) is all zero because L[3..5, 0] = 0. */
static void test_supernode_eliminate_diag_block_diagonal(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t b = 0; b < 2; b++) {
        idx_t o = b * 3;
        for (idx_t i = 0; i < 3; i++)
            for (idx_t j = 0; j < 3; j++)
                sparse_insert(A, o + i, o + j, (i == j) ? 4.0 : 1.0);
    }

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));
    CholCsc *ref_L = NULL;
    REQUIRE_OK(chol_csc_factor(A, NULL, &ref_L));

    /* Supernode 0: cols [0, 3).  No prior columns → external cmod
     * is empty. */
    {
        idx_t ph = chol_csc_supernode_panel_height(csc, 0);
        ASSERT_EQ(ph, 3);
        double *dense = calloc((size_t)(ph * 3), sizeof(double));
        idx_t *row_map = calloc((size_t)ph, sizeof(idx_t));
        idx_t ph_out = 0;
        REQUIRE_OK(chol_csc_supernode_extract(csc, 0, 3, dense, ph, row_map, &ph_out));
        REQUIRE_OK(chol_csc_supernode_eliminate_diag(csc, 0, 3, dense, ph, row_map, ph, 0.0));
        for (idx_t j = 0; j < 3; j++)
            for (idx_t i = j; i < 3; i++)
                ASSERT_TRUE(fabs(dense[i + j * ph] - chol_csc_value_at(ref_L, i, j)) < 1e-12);
        free(dense);
        free(row_map);
    }

    /* Supernode 1: cols [3, 6).  First overwrite cols [0, 3) of csc
     * with the scalar L so external cmod sees the right L values. */
    for (idx_t c = 0; c < 3; c++) {
        for (idx_t p = csc->col_ptr[c]; p < csc->col_ptr[c + 1]; p++) {
            idx_t r = csc->row_idx[p];
            csc->values[p] = chol_csc_value_at(ref_L, r, c);
        }
    }
    {
        idx_t ph = chol_csc_supernode_panel_height(csc, 3);
        ASSERT_EQ(ph, 3);
        double *dense = calloc((size_t)(ph * 3), sizeof(double));
        idx_t *row_map = calloc((size_t)ph, sizeof(idx_t));
        idx_t ph_out = 0;
        REQUIRE_OK(chol_csc_supernode_extract(csc, 3, 3, dense, ph, row_map, &ph_out));
        REQUIRE_OK(chol_csc_supernode_eliminate_diag(csc, 3, 3, dense, ph, row_map, ph, 0.0));
        for (idx_t j = 0; j < 3; j++)
            for (idx_t i = j; i < 3; i++)
                ASSERT_TRUE(fabs(dense[i + j * ph] - chol_csc_value_at(ref_L, 3 + i, 3 + j)) <
                            1e-12);
        free(dense);
        free(row_map);
    }

    chol_csc_free(csc);
    chol_csc_free(ref_L);
    sparse_free(A);
}

/* Non-SPD matrix: chol_dense_factor must return SPARSE_ERR_NOT_SPD
 * and the helper must surface it.  Use a fully-dense 3×3 so the
 * supernode invariant holds (panel_height == s_size); the negative
 * diagonal is caught inside chol_dense_factor's first step. */
static void test_supernode_eliminate_diag_not_spd(void) {
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    /* Dense 3×3 with A[0,0] = -1 so factorisation breaks immediately.
     * All off-diagonals present so the CSC stores the full lower
     * triangle — s_size = 3 supernode with panel_height = 3. */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            double v;
            if (i == j)
                v = (i == 0) ? -1.0 : 1.0;
            else
                v = 0.3;
            sparse_insert(A, i, j, v);
        }
    }

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));

    idx_t ph = chol_csc_supernode_panel_height(csc, 0);
    ASSERT_EQ(ph, n);
    double *dense = calloc((size_t)(ph * n), sizeof(double));
    idx_t *row_map = calloc((size_t)ph, sizeof(idx_t));
    idx_t ph_out = 0;
    REQUIRE_OK(chol_csc_supernode_extract(csc, 0, n, dense, ph, row_map, &ph_out));

    ASSERT_ERR(chol_csc_supernode_eliminate_diag(csc, 0, n, dense, ph, row_map, ph, 0.0),
               SPARSE_ERR_NOT_SPD);

    free(dense);
    free(row_map);
    chol_csc_free(csc);
    sparse_free(A);
}

/* Error paths: null args, invalid range, insufficient lda / panel_height. */
static void test_supernode_eliminate_diag_error_paths(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 2.0);
    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));

    double dense[16] = {0};
    idx_t row_map[4] = {0};

    ASSERT_ERR(chol_csc_supernode_eliminate_diag(NULL, 0, 1, dense, 4, row_map, 1, 0.0),
               SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_supernode_eliminate_diag(csc, 0, 1, NULL, 4, row_map, 1, 0.0),
               SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_supernode_eliminate_diag(csc, 0, 1, dense, 4, NULL, 1, 0.0),
               SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_supernode_eliminate_diag(csc, -1, 1, dense, 4, row_map, 1, 0.0),
               SPARSE_ERR_BADARG);
    ASSERT_ERR(chol_csc_supernode_eliminate_diag(csc, 0, 0, dense, 4, row_map, 1, 0.0),
               SPARSE_ERR_BADARG);
    /* s_start + s_size > n */
    ASSERT_ERR(chol_csc_supernode_eliminate_diag(csc, 2, 3, dense, 4, row_map, 1, 0.0),
               SPARSE_ERR_BADARG);
    /* panel_height < s_size */
    ASSERT_ERR(chol_csc_supernode_eliminate_diag(csc, 0, 2, dense, 4, row_map, 1, 0.0),
               SPARSE_ERR_BADARG);
    /* lda < panel_height */
    ASSERT_ERR(chol_csc_supernode_eliminate_diag(csc, 0, 1, dense, 0, row_map, 1, 0.0),
               SPARSE_ERR_BADARG);

    chol_csc_free(csc);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Panel solve and full batched path integration
 * ═══════════════════════════════════════════════════════════════════════ */

/* Dense 10×10 SPD factor via the full batched path; compare residual
 * ||A·x - b|| / ||b|| against the scalar kernel's solve. */
static void test_eliminate_supernodal_dense_10x10_residual(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, (i == j) ? (double)(n + 2) : 1.0);

    double *b = malloc((size_t)n * sizeof(double));
    double *x_sc = calloc((size_t)n, sizeof(double));
    double *x_sn = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0 + (double)i;

    CholCsc *Ls = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &Ls));
    REQUIRE_OK(chol_csc_eliminate(Ls));
    REQUIRE_OK(chol_csc_solve(Ls, b, x_sc));

    CholCsc *Ln = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &Ln));
    REQUIRE_OK(chol_csc_eliminate_supernodal(Ln, 4));
    REQUIRE_OK(chol_csc_validate(Ln));
    REQUIRE_OK(chol_csc_solve(Ln, b, x_sn));

    /* Both solves should hit the same x within round-off. */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_sc[i], x_sn[i], 1e-12);

    /* Residual check against the original RHS. */
    double *Ax = calloc((size_t)n, sizeof(double));
    sparse_matvec(A, x_sn, Ax);
    double rr = 0.0, bn = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double r = fabs(Ax[i] - b[i]);
        double bi = fabs(b[i]);
        if (r > rr)
            rr = r;
        if (bi > bn)
            bn = bi;
    }
    ASSERT_TRUE(rr / (bn > 0.0 ? bn : 1.0) < 1e-10);

    free(b);
    free(x_sc);
    free(x_sn);
    free(Ax);
    chol_csc_free(Ls);
    chol_csc_free(Ln);
    sparse_free(A);
}

/* Degenerate min_size = 1: every column forms its own 1×1 supernode.
 * The batched path then runs once per column with s_size = 1, which
 * should reduce numerically to the scalar cdiv + cmod + gather. */
static void test_eliminate_supernodal_size1_matches_scalar(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    /* Dense SPD so every column has plenty of below-diagonal entries;
     * s_size = 1 forces the supernode path to process each column's
     * panel individually through chol_dense_solve_lower(size=1). */
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, (i == j) ? (double)(n + 1) : 0.5);

    CholCsc *Ls = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &Ls));
    REQUIRE_OK(chol_csc_eliminate(Ls));

    CholCsc *Ln = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &Ln));
    REQUIRE_OK(chol_csc_eliminate_supernodal(Ln, 1));
    REQUIRE_OK(chol_csc_validate(Ln));

    ASSERT_TRUE(chol_csc_values_match(Ls, Ln, 1e-12));

    chol_csc_free(Ls);
    chol_csc_free(Ln);
    sparse_free(A);
}

/* Seeded random SPD sweep: generate SPD matrices at several n, factor
 * each via the scalar path and the supernodal path, and assert the
 * factor values match to round-off on every run.
 *
 * We construct A = B + B^T + n·I with B dense and random so that
 * (a) A is SPD (diagonal dominates off-diagonal magnitudes) and
 * (b) A's structure is dense, so the whole matrix is one supernode
 *     and the batched diag + panel path fires exactly once. */
static unsigned int supernodal_rng = 0u;
static unsigned int supernodal_rng_next(void) {
    supernodal_rng = supernodal_rng * 1664525u + 1013904223u;
    return supernodal_rng;
}
static double supernodal_rng_uniform(double lo, double hi) {
    double u = (double)(supernodal_rng_next() & 0x7fffffff) / (double)0x7fffffff;
    return lo + (hi - lo) * u;
}

static void test_eliminate_supernodal_random_spd_sweep(void) {
    /* Size / seed pairs selected to cover small + moderate supernodes
     * (n up to 60 keeps the dense-SPD test cheap under full O(n³)
     * factorisation cost). */
    struct {
        idx_t n;
        unsigned int seed;
    } cases[] = {
        {12, 0xdecade01u}, {12, 0xdecade02u}, {20, 0xdecade03u}, {20, 0xdecade04u},
        {32, 0xdecade05u}, {32, 0xdecade06u}, {48, 0xdecade07u}, {48, 0xdecade08u},
        {60, 0xdecade09u}, {60, 0xdecade0au},
    };
    const size_t ncases = sizeof(cases) / sizeof(cases[0]);

    for (size_t idx = 0; idx < ncases; idx++) {
        idx_t n = cases[idx].n;
        supernodal_rng = cases[idx].seed;

        /* Build A = B + B^T + n·I with B ∈ [-1, 1]^{n×n}.  SPD by
         * diagonal dominance (n >> ||B||_∞). */
        SparseMatrix *A = sparse_create(n, n);
        for (idx_t i = 0; i < n; i++) {
            sparse_insert(A, i, i, (double)n);
            for (idx_t j = 0; j < i; j++) {
                double v = supernodal_rng_uniform(-1.0, 1.0);
                sparse_insert(A, i, j, v);
                sparse_insert(A, j, i, v);
            }
        }

        CholCsc *Ls = NULL;
        REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &Ls));
        REQUIRE_OK(chol_csc_eliminate(Ls));

        CholCsc *Ln = NULL;
        REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &Ln));
        REQUIRE_OK(chol_csc_eliminate_supernodal(Ln, 4));
        REQUIRE_OK(chol_csc_validate(Ln));

        ASSERT_TRUE(chol_csc_values_match(Ls, Ln, 1e-10));

        chol_csc_free(Ls);
        chol_csc_free(Ln);
        sparse_free(A);
    }
}

/* Residual check on bcsstk04 with AMD: the supernodal path's factor
 * followed by a solve against A·x = b must land within 1e-10 relative
 * residual, independently of whether any supernodes were detected. */
static void test_eliminate_supernodal_bcsstk04_residual(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx"));
    idx_t n = sparse_rows(A);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(sparse_reorder_amd(A, perm));

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, perm, 2.0, &L));
    REQUIRE_OK(chol_csc_eliminate_supernodal(L, 4));
    REQUIRE_OK(chol_csc_validate(L));
    REQUIRE_OK(chol_csc_solve_perm(L, perm, b, x));

    double *Ax = calloc((size_t)n, sizeof(double));
    sparse_matvec(A, x, Ax);
    double rr = 0.0, bn = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double r = fabs(Ax[i] - b[i]);
        double bi = fabs(b[i]);
        if (r > rr)
            rr = r;
        if (bi > bn)
            bn = bi;
    }
    ASSERT_TRUE(rr / (bn > 0.0 ? bn : 1.0) < 1e-10);

    free(perm);
    free(ones);
    free(b);
    free(x);
    free(Ax);
    chol_csc_free(L);
    sparse_free(A);
}

/* The full supernodal entry must reject a
 * non-positive stored diagonal before any supernode dispatch or writeback
 * mutation begins. */
static void test_eliminate_supernodal_rejects_nonpositive_stored_diagonal(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            double v = 0.0;
            if (i == j)
                v = (i == 0) ? -1.0 : 4.0;
            else if (labs((long)i - (long)j) == 1)
                v = 0.25;
            if (v != 0.0)
                sparse_insert(A, i, j, v);
        }
    }

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 1.0, &csc));
    ASSERT_TRUE(csc != NULL);
    ASSERT_TRUE(csc->col_ptr[0] < csc->col_ptr[1]);
    ASSERT_TRUE(csc->values[csc->col_ptr[0]] == -1.0);

    ASSERT_ERR(chol_csc_eliminate_supernodal(csc, 2), SPARSE_ERR_NOT_SPD);
    ASSERT_TRUE(csc->values[csc->col_ptr[0]] == -1.0);

    chol_csc_free(csc);
    sparse_free(A);
}

/* Panel helper: trivial null-arg / bad-range checks, plus a
 * panel_rows == 0 fast path that returns SPARSE_OK. */
static void test_supernode_eliminate_panel_error_paths(void) {
    double L_diag[4] = {1.0, 0.5, 0.0, 2.0};
    double panel[4] = {0};

    ASSERT_ERR(chol_csc_supernode_eliminate_panel(NULL, 2, 2, panel, 2, 1), SPARSE_ERR_NULL);
    /* s_size < 1 */
    ASSERT_ERR(chol_csc_supernode_eliminate_panel(L_diag, 0, 2, panel, 2, 1), SPARSE_ERR_BADARG);
    /* lda_diag < s_size */
    ASSERT_ERR(chol_csc_supernode_eliminate_panel(L_diag, 2, 1, panel, 2, 1), SPARSE_ERR_BADARG);
    /* panel_rows < 0 */
    ASSERT_ERR(chol_csc_supernode_eliminate_panel(L_diag, 2, 2, panel, 2, -1), SPARSE_ERR_BADARG);
    /* panel_rows == 0: fast-path SPARSE_OK even with null panel. */
    ASSERT_ERR(chol_csc_supernode_eliminate_panel(L_diag, 2, 2, NULL, 0, 0), SPARSE_OK);
    /* lda_panel < panel_rows */
    ASSERT_ERR(chol_csc_supernode_eliminate_panel(L_diag, 2, 2, panel, 0, 1), SPARSE_ERR_BADARG);
}

static void
test_supernode_eliminate_diag_missing_dense_kernel_descriptor_is_backend_contract_error(void) {
    SparseMatrix *A = sparse_create(2, 2);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));

    idx_t row_map[2] = {0, 1};
    double dense[4] = {4.0, 1.0, 1.0, 3.0};

    chol_csc_supernodal_set_dense_kernels_override_for_test(NULL);
    ASSERT_ERR(chol_csc_supernode_eliminate_diag(csc, 0, 2, dense, 2, row_map, 2, 0.0),
               SPARSE_ERR_BACKEND_CONTRACT);
    chol_csc_supernodal_clear_dense_kernels_override_for_test();

    chol_csc_free(csc);
    sparse_free(A);
}

static void test_supernode_eliminate_diag_missing_factor_kernel_is_backend_contract_error(void) {
    SparseMatrix *A = sparse_create(2, 2);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);

    CholCsc *csc = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &csc));

    idx_t row_map[2] = {0, 1};
    double dense[4] = {4.0, 1.0, 1.0, 3.0};
    static const chol_dense_kernels_t missing_factor = {
        .name = "missing-factor",
        .factor = NULL,
        .solve_lower = chol_dense_solve_lower,
        .solve_panel = chol_dense_solve_panel,
    };

    chol_csc_supernodal_set_dense_kernels_override_for_test(&missing_factor);
    ASSERT_ERR(chol_csc_supernode_eliminate_diag(csc, 0, 2, dense, 2, row_map, 2, 0.0),
               SPARSE_ERR_BACKEND_CONTRACT);
    chol_csc_supernodal_clear_dense_kernels_override_for_test();

    chol_csc_free(csc);
    sparse_free(A);
}

static void test_supernode_eliminate_panel_missing_solve_panel_is_backend_contract_error(void) {
    double L_diag[4] = {2.0, 1.0, 0.0, 2.0};
    double panel[2] = {2.0, 5.0};
    static const chol_dense_kernels_t missing_solve_panel = {
        .name = "missing-solve-panel",
        .factor = chol_dense_factor,
        .solve_lower = chol_dense_solve_lower,
        .solve_panel = NULL,
    };

    chol_csc_supernodal_set_dense_kernels_override_for_test(&missing_solve_panel);
    ASSERT_ERR(chol_csc_supernode_eliminate_panel(L_diag, 2, 2, panel, 1, 1),
               SPARSE_ERR_BACKEND_CONTRACT);
    chol_csc_supernodal_clear_dense_kernels_override_for_test();
}

/* ═══════════════════════════════════════════════════════════════════════
 * Parametrised scalar-to-batched cross-check and boundary coverage
 * ═══════════════════════════════════════════════════════════════════════ */

/* Parametrised cross-check: factor each SPD fixture scalar-vs-batched
 * across two reorder regimes (identity and AMD) and a range of
 * min_size thresholds.  Every combination must yield byte-identical
 * factors.
 *
 * For the SuiteSparse fixtures (nos4, bcsstk04) the cross-check uses
 * `min_size ∈ {4, 16}`.  A `min_size = 1` parametrisation on those
 * sparse matrices would expose a documented limitation of the
 * batched path: the detected-supernode extract uses A's pre-fill
 * column pattern, and a fundamental supernode of size ≥ 2 in A can
 * sit above rows that gain L-fill from prior eliminations.  The
 * `s_size == 1` fast-path (see `chol_csc_eliminate_supernodal`)
 * delegates singletons to the scalar kernel, and `min_size >= 4`
 * exercises the batched branch only on supernodes large enough to
 * benefit from the dense factor / dense solve primitives.  The
 * synthetic dense / block-diagonal fixtures have no fill, so their
 * cross-check sweeps through `min_size ∈ {1, 4, 16}`. */
static void test_supernodal_parametrised_cross_check(void) {
    const char *mtx_paths[] = {SS_DIR "/nos4.mtx", SS_DIR "/bcsstk04.mtx"};
    const idx_t ss_min_sizes[] = {4, 16};
    const idx_t synth_min_sizes[] = {1, 4, 16};
    const size_t n_paths = sizeof(mtx_paths) / sizeof(mtx_paths[0]);
    const size_t n_ss_min = sizeof(ss_min_sizes) / sizeof(ss_min_sizes[0]);
    const size_t n_synth_min = sizeof(synth_min_sizes) / sizeof(synth_min_sizes[0]);

    for (size_t pi = 0; pi < n_paths; pi++) {
        SparseMatrix *A = NULL;
        REQUIRE_OK(sparse_load_mm(&A, mtx_paths[pi]));
        idx_t n = sparse_rows(A);

        /* Run 1: identity permutation. */
        for (size_t mi = 0; mi < n_ss_min; mi++)
            assert_supernodal_matches_scalar(A, NULL, ss_min_sizes[mi], 1e-10, mtx_paths[pi]);

        /* Run 2: AMD fill-reducing reorder. */
        idx_t *perm = malloc((size_t)n * sizeof(idx_t));
        REQUIRE_OK(sparse_reorder_amd(A, perm));
        for (size_t mi = 0; mi < n_ss_min; mi++)
            assert_supernodal_matches_scalar(A, perm, ss_min_sizes[mi], 1e-10, mtx_paths[pi]);
        free(perm);

        sparse_free(A);
    }

    /* Dense synthetic: one big supernode; stresses the batched path. */
    {
        idx_t n = 12;
        SparseMatrix *A = sparse_create(n, n);
        for (idx_t i = 0; i < n; i++)
            for (idx_t j = 0; j < n; j++)
                sparse_insert(A, i, j, (i == j) ? (double)(n + 1) : 0.25);
        for (size_t mi = 0; mi < n_synth_min; mi++)
            assert_supernodal_matches_scalar(A, NULL, synth_min_sizes[mi], 1e-12, "dense12");
        sparse_free(A);
    }

    /* Block-diagonal synthetic: two supernodes, each size 5. */
    {
        idx_t n = 10;
        SparseMatrix *A = sparse_create(n, n);
        for (idx_t b = 0; b < 2; b++) {
            idx_t o = b * 5;
            for (idx_t i = 0; i < 5; i++)
                for (idx_t j = 0; j < 5; j++)
                    sparse_insert(A, o + i, o + j, (i == j) ? 8.0 : 1.0);
        }
        for (size_t mi = 0; mi < n_synth_min; mi++)
            assert_supernodal_matches_scalar(A, NULL, synth_min_sizes[mi], 1e-12, "block10");
        sparse_free(A);
    }
}

/* Boundary supernode test: construct A so the batched loop visits
 * both a singleton supernode (size 1) and a large supernode (size ≥ 4)
 * in the same invocation when called with min_size = 1.
 *
 * Structure:
 *   Col 0       — diagonal-only singleton, isolated from cols [1, n).
 *   Cols 1..n-1 — dense SPD block forming a single fundamental
 *                 supernode of size n-1.
 *
 * With min_size = 1, `chol_csc_detect_supernodes` reports:
 *   supernode 0: start = 0, size = 1    (singleton branch)
 *   supernode 1: start = 1, size = n-1  (batched diag + panel branch)
 *
 * The test then asserts scalar == batched on this matrix, so both the
 * singleton branch (degenerate `s_size == 1`, trivial dense factor,
 * empty panel) and the larger-supernode branch run through the same
 * integrated loop. */
static void test_supernodal_boundary_singleton_plus_large(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    /* Col 0: isolated diagonal. */
    sparse_insert(A, 0, 0, 5.0);
    /* Cols [1, n): dense SPD block. */
    for (idx_t i = 1; i < n; i++)
        for (idx_t j = 1; j < n; j++)
            sparse_insert(A, i, j, (i == j) ? (double)(n + 2) : 1.0);

    /* Confirm the detected partition matches the expected boundary
     * shape (size 1 + size n-1) under min_size = 1. */
    CholCsc *inspect = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &inspect));
    idx_t *starts = malloc((size_t)n * sizeof(idx_t));
    idx_t *sizes = malloc((size_t)n * sizeof(idx_t));
    idx_t count = 0;
    REQUIRE_OK(chol_csc_detect_supernodes(inspect, 1, starts, sizes, &count));
    ASSERT_EQ(count, 2);
    ASSERT_EQ(starts[0], 0);
    ASSERT_EQ(sizes[0], 1);
    ASSERT_EQ(starts[1], 1);
    ASSERT_EQ(sizes[1], n - 1);
    free(starts);
    free(sizes);
    chol_csc_free(inspect);

    /* Scalar == batched on the same matrix; forces the batched loop
     * to run both the singleton and the size-(n-1) supernode. */
    assert_supernodal_matches_scalar(A, NULL, 1, 1e-12, "boundary");

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * CSC to linked-list writeback for transparent dispatch
 * ═══════════════════════════════════════════════════════════════════════ */

/* Dense 5×5 SPD, no reorder: verify round-trip. */
static void test_writeback_roundtrip_dense5_noreorder(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, (i == j) ? (double)(n + 1) : 1.0);
    writeback_roundtrip_check(A, 0, 1e-12);
    sparse_free(A);
}

/* Tridiagonal SPD, no reorder. */
static void test_writeback_roundtrip_tridiag_noreorder(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    writeback_roundtrip_check(A, 0, 1e-12);
    sparse_free(A);
}

/* SuiteSparse nos4 with AMD: verifies the reorder_perm is populated
 * correctly and L matches the scalar reference. */
static void test_writeback_roundtrip_nos4_amd(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/nos4.mtx"));
    writeback_roundtrip_check(A, 1, 1e-10);
    sparse_free(A);
}

/* SuiteSparse bcsstk04 with AMD. */
static void test_writeback_roundtrip_bcsstk04_amd(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx"));
    writeback_roundtrip_check(A, 1, 1e-10);
    sparse_free(A);
}

/* Writeback rejects a matrix that is already factored. */
static void test_writeback_rejects_already_factored(void) {
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 2.0);
    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &L));
    REQUIRE_OK(chol_csc_eliminate(L));

    /* Factor A via scalar so its `factored` flag is 1, then try to
     * writeback on top — should be rejected. */
    REQUIRE_OK(sparse_cholesky_factor(A));
    ASSERT_TRUE(A->factored);
    ASSERT_ERR(chol_csc_writeback_to_sparse(L, A, NULL), SPARSE_ERR_BADARG);

    chol_csc_free(L);
    sparse_free(A);
}

/* Writeback rejects a matrix with non-identity row_perm. */
static void test_writeback_rejects_nonidentity_row_perm(void) {
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 2.0);
    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &L));
    REQUIRE_OK(chol_csc_eliminate(L));

    /* Scramble row_perm so the precondition check fails.  inv_row_perm
     * is also rotated to keep the matrix in a valid (if permuted)
     * state. */
    A->row_perm[0] = 1;
    A->row_perm[1] = 0;
    A->inv_row_perm[0] = 1;
    A->inv_row_perm[1] = 0;
    ASSERT_ERR(chol_csc_writeback_to_sparse(L, A, NULL), SPARSE_ERR_BADARG);

    chol_csc_free(L);
    sparse_free(A);
}

/* Writeback rejects null arguments. */
static void test_writeback_rejects_null(void) {
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 2.0);
    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &L));
    REQUIRE_OK(chol_csc_eliminate(L));

    ASSERT_ERR(chol_csc_writeback_to_sparse(NULL, A, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(chol_csc_writeback_to_sparse(L, NULL, NULL), SPARSE_ERR_NULL);

    chol_csc_free(L);
    sparse_free(A);
}

/* Writeback rejects a size-mismatched target. */
static void test_writeback_rejects_shape_mismatch(void) {
    SparseMatrix *A = sparse_create(3, 3);
    for (idx_t i = 0; i < 3; i++)
        sparse_insert(A, i, i, 2.0);
    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, NULL, 2.0, &L));
    REQUIRE_OK(chol_csc_eliminate(L));

    SparseMatrix *big = sparse_create(5, 5);
    ASSERT_ERR(chol_csc_writeback_to_sparse(L, big, NULL), SPARSE_ERR_SHAPE);

    chol_csc_free(L);
    sparse_free(A);
    sparse_free(big);
}

/* Writeback should publish a solve-ready factored shell with identity internal
 * perms and the external reorder permutation carried in factor state. */
static void test_writeback_publishes_solve_ready_factored_shell(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    REQUIRE_OK(perm ? SPARSE_OK : SPARSE_ERR_ALLOC);
    REQUIRE_OK(sparse_reorder_amd(A, perm));

    CholCsc *L = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, perm, 2.0, &L));
    REQUIRE_OK(chol_csc_eliminate(L));

    SparseMatrix *got = sparse_copy(A);
    ASSERT_TRUE(got != NULL);
    REQUIRE_OK(chol_csc_writeback_to_sparse(L, got, perm));

    ASSERT_TRUE(got->factored);
    ASSERT_TRUE(got->reorder_perm != NULL);
    for (idx_t i = 0; i < n; i++) {
        ASSERT_EQ(got->row_perm[i], i);
        ASSERT_EQ(got->inv_row_perm[i], i);
        ASSERT_EQ(got->col_perm[i], i);
        ASSERT_EQ(got->inv_col_perm[i], i);
        ASSERT_EQ(got->reorder_perm[i], perm[i]);
    }

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    REQUIRE_OK(ones && b && x ? SPARSE_OK : SPARSE_ERR_ALLOC);
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);
    REQUIRE_OK(sparse_cholesky_solve(got, b, x));
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-10);

    free(ones);
    free(b);
    free(x);
    free(perm);
    chol_csc_free(L);
    sparse_free(got);
    sparse_free(A);
}

static void run_supernode_detection_tests(void) {
    RUN_TEST(test_detect_supernodes_null_args);
    RUN_TEST(test_detect_supernodes_diagonal);
    RUN_TEST(test_detect_supernodes_dense);
    RUN_TEST(test_detect_supernodes_block_diagonal);
    RUN_TEST(test_detect_supernodes_tridiagonal);
    RUN_TEST(test_detect_supernodes_reverse_arrowhead);
    RUN_TEST(test_detect_supernodes_suitesparse_report);
}

static void run_supernodal_postorder_tests(void) {
    RUN_TEST(test_supernodal_postorder_residual_unchanged);
    RUN_TEST(test_supernodal_postorder_no_supernode_count_regression);
}

static void run_supernodal_dense_tests(void) {
    RUN_TEST(test_chol_dense_factor_null);
    RUN_TEST(test_chol_dense_factor_1x1);
    RUN_TEST(test_chol_dense_factor_2x2);
    RUN_TEST(test_chol_dense_factor_4x4);
    RUN_TEST(test_chol_dense_factor_not_spd);
    RUN_TEST(test_chol_dense_solve_null);
    RUN_TEST(test_chol_dense_solve_lower_3x3);
    RUN_TEST(test_chol_dense_solve_panel_2x2_two_rhs);
    RUN_TEST(test_supernodal_dense_backend_default_contract);
    RUN_TEST(test_supernodal_dense_backend_builtin_env_contract);
    RUN_TEST(test_supernodal_dense_backend_accelerate_env_contract);
    RUN_TEST(test_supernodal_dense_backend_external_env_contract);

    /* ldlt_dense_factor (BK on column-major) cross-checks */
    RUN_TEST(test_ldlt_dense_factor_arg_checks);
    RUN_TEST(test_ldlt_dense_factor_4x4_indefinite);
    RUN_TEST(test_ldlt_dense_factor_2x2_forced);
    RUN_TEST(test_ldlt_dense_factor_6x6_mixed_pivots);
    RUN_TEST(test_eliminate_supernodal_dense);
    RUN_TEST(test_eliminate_supernodal_block_diagonal);
    RUN_TEST(test_eliminate_supernodal_bcsstk04_amd);
    RUN_TEST(test_chol_csc_kuu_scalar_no_regression);
    RUN_TEST(test_eliminate_supernodal_null);
}

static void run_supernode_extract_writeback_tests(void) {
    RUN_TEST(test_supernode_extract_writeback_dense);
    RUN_TEST(test_supernode_extract_writeback_block_diagonal);
    RUN_TEST(test_supernode_extract_writeback_with_below_panel);
    RUN_TEST(test_supernode_extract_lda_padding);
    RUN_TEST(test_supernode_extract_error_paths);
}

static void run_supernode_diag_factor_tests(void) {
    RUN_TEST(test_supernode_eliminate_diag_dense_8x8);
    RUN_TEST(test_supernode_eliminate_diag_with_external_cmod);
    RUN_TEST(test_supernode_eliminate_diag_block_diagonal);
    RUN_TEST(test_supernode_eliminate_diag_not_spd);
    RUN_TEST(test_supernode_eliminate_diag_error_paths);
}

static void run_supernode_panel_tests(void) {
    RUN_TEST(test_eliminate_supernodal_dense_10x10_residual);
    RUN_TEST(test_eliminate_supernodal_size1_matches_scalar);
    RUN_TEST(test_eliminate_supernodal_random_spd_sweep);
    RUN_TEST(test_eliminate_supernodal_bcsstk04_residual);
    RUN_TEST(test_eliminate_supernodal_rejects_nonpositive_stored_diagonal);
    RUN_TEST(test_supernode_eliminate_panel_error_paths);
    RUN_TEST(
        test_supernode_eliminate_diag_missing_dense_kernel_descriptor_is_backend_contract_error);
    RUN_TEST(test_supernode_eliminate_diag_missing_factor_kernel_is_backend_contract_error);
    RUN_TEST(test_supernode_eliminate_panel_missing_solve_panel_is_backend_contract_error);
}

static void run_supernodal_parametrised_tests(void) {
    RUN_TEST(test_supernodal_parametrised_cross_check);
    RUN_TEST(test_supernodal_boundary_singleton_plus_large);
}

static void run_writeback_tests(void) {
    RUN_TEST(test_writeback_roundtrip_dense5_noreorder);
    RUN_TEST(test_writeback_roundtrip_tridiag_noreorder);
    RUN_TEST(test_writeback_roundtrip_nos4_amd);
    RUN_TEST(test_writeback_roundtrip_bcsstk04_amd);
    RUN_TEST(test_writeback_rejects_already_factored);
    RUN_TEST(test_writeback_rejects_nonidentity_row_perm);
    RUN_TEST(test_writeback_rejects_null);
    RUN_TEST(test_writeback_rejects_shape_mismatch);
    RUN_TEST(test_writeback_publishes_solve_ready_factored_shell);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("chol_csc_supernodal");

    /* Supernode detection */
    run_supernode_detection_tests();

    /* Supernodal etree-postorder corpus safety */
    run_supernodal_postorder_tests();

    /* Dense primitives + supernode-aware elimination */
    run_supernodal_dense_tests();

    /* Supernode extract / writeback plumbing */
    run_supernode_extract_writeback_tests();

    /* Supernode diagonal-block factor */
    run_supernode_diag_factor_tests();

    /* Panel solve + full batched path integration */
    run_supernode_panel_tests();

    /* Parametrised scalar↔batched cross-check + boundary */
    run_supernodal_parametrised_tests();

    /* CSC → linked-list writeback */
    run_writeback_tests();

    TEST_SUITE_END();
}
