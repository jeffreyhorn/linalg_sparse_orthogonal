/*
 * Quotient-graph AMD proof owner.
 *
 * The public `sparse_reorder_amd` wrapper is expected to delegate directly to
 * the internal quotient-graph implementation.  This suite owns that contract:
 *
 *   - argument validation for the internal entry point;
 *   - bit-identical wrapper/helper permutations on selected SuiteSparse
 *     fixtures;
 *   - identical symbolic-Cholesky fill through both entry points; and
 *   - a large regular banded stress fixture that would be unsuitable for the
 *     retired bitset implementation.
 *
 * Historical bitset comparison remains isolated in `benchmarks/bench_amd_qg.c`
 * as a benchmark/reporting foil.  These tests are the maintained in-tree
 * delegation and structural guardrail surface.
 */

#include "sparse_analysis.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_reorder_amd_qg_internal.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* Argument validation lives before the elimination loop and should match the
 * public wrapper contract. */

static void test_amd_qg_null_args(void) {
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_NOT_NULL(A);
    idx_t perm[1] = {0};
    ASSERT_ERR(sparse_reorder_amd_qg(NULL, perm), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_reorder_amd_qg(A, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_amd_qg_rejects_rectangular(void) {
    SparseMatrix *A = sparse_create(3, 5);
    ASSERT_NOT_NULL(A);
    idx_t perm[3] = {0};
    ASSERT_ERR(sparse_reorder_amd_qg(A, perm), SPARSE_ERR_SHAPE);
    sparse_free(A);
}

static void test_amd_qg_singleton(void) {
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_NOT_NULL(A);
    sparse_insert(A, 0, 0, 1.0);
    idx_t perm[1] = {99};
    REQUIRE_OK(sparse_reorder_amd_qg(A, perm));
    ASSERT_EQ(perm[0], 0);
    sparse_free(A);
}

/* ─── Helper: validate perm + compute symbolic Cholesky nnz ─────── */

static int is_valid_permutation(const idx_t *perm, idx_t n) {
    int *seen = calloc((size_t)n, sizeof(int));
    if (!seen)
        return 0;
    for (idx_t i = 0; i < n; i++) {
        idx_t p = perm[i];
        if (p < 0 || p >= n || seen[p]) {
            free(seen);
            return 0;
        }
        seen[p] = 1;
    }
    free(seen);
    return 1;
}

static idx_t symbolic_cholesky_nnz_with_perm(const SparseMatrix *A, const idx_t *perm) {
    SparseMatrix *PA = NULL;
    if (sparse_permute(A, perm, perm, &PA) != SPARSE_OK)
        return -1;
    sparse_analysis_opts_t opts = {
        .factor_type = SPARSE_FACTOR_CHOLESKY,
        .reorder = SPARSE_REORDER_NONE,
    };
    sparse_analysis_t analysis = {0};
    sparse_err_t rc = sparse_analyze(PA, &opts, &analysis);
    idx_t nnz = (rc == SPARSE_OK) ? analysis.sym_L.nnz : (idx_t)-1;
    sparse_analysis_free(&analysis);
    sparse_free(PA);
    return nnz;
}

/* ─── Wrapper delegation: public sparse_reorder_amd vs internal qg ── */

/* The public wrapper and the internal helper should share one code path.  This
 * compare therefore expects bit-identical permutations and identical
 * symbolic-Cholesky fill. */
static void compare_wrapper_vs_qg(const char *fixture, const char *path) {
    SparseMatrix *A = NULL;
    sparse_err_t rc = sparse_load_mm(&A, path);
    if (rc != SPARSE_OK) {
        printf("    skipped (%s not loadable: %d)\n", fixture, (int)rc);
        return;
    }
    idx_t n = sparse_rows(A);

    idx_t *perm_wrapper = malloc((size_t)n * sizeof(idx_t));
    idx_t *perm_qg = malloc((size_t)n * sizeof(idx_t));
    /* Fail-fast on alloc — ASSERT_NOT_NULL is non-fatal in this test
     * framework, so without an early return the subsequent
     * sparse_reorder_amd / _amd_qg calls would receive a NULL perm
     * and likely crash.  Free everything we did allocate (including
     * A) on the unhappy path so the test exits cleanly. */
    if (!perm_wrapper || !perm_qg) {
        free(perm_wrapper);
        free(perm_qg);
        sparse_free(A);
        REQUIRE_OK(SPARSE_ERR_ALLOC);
        return;
    }

    /* Capture rc through both reorder calls and route to a single
     * `cleanup:` label so a failure can't leak perm_wrapper /
     * perm_qg / A into subsequent fixture iterations.  This
     * helper runs across multiple fixtures, so a leak compounds.
     * Reuse the outer `rc` (declared at the sparse_load_mm call
     * above) so we don't shadow it. */
    rc = sparse_reorder_amd(A, perm_wrapper);
    if (rc != SPARSE_OK)
        goto cleanup;
    rc = sparse_reorder_amd_qg(A, perm_qg);
    if (rc != SPARSE_OK)
        goto cleanup;

    /* Both must be valid permutations of [0, n). */
    ASSERT_TRUE(is_valid_permutation(perm_wrapper, n));
    ASSERT_TRUE(is_valid_permutation(perm_qg, n));

    /* Wrapper must produce the bit-identical permutation as the
     * helper it delegates to.  If this ever diverges, the wrapper
     * has grown logic beyond pure forwarding. */
    for (idx_t i = 0; i < n; i++)
        ASSERT_EQ(perm_wrapper[i], perm_qg[i]);

    idx_t nnz_wrapper = symbolic_cholesky_nnz_with_perm(A, perm_wrapper);
    idx_t nnz_qg = symbolic_cholesky_nnz_with_perm(A, perm_qg);
    ASSERT_TRUE(nnz_wrapper > 0);
    ASSERT_TRUE(nnz_qg > 0);
    ASSERT_EQ(nnz_wrapper, nnz_qg);

    printf("    %s (n=%d): wrapper nnz(L) = %d, qg nnz(L) = %d (identical)\n", fixture, (int)n,
           (int)nnz_wrapper, (int)nnz_qg);

cleanup:
    free(perm_wrapper);
    free(perm_qg);
    sparse_free(A);
    REQUIRE_OK(rc);
}

static void test_amd_qg_delegation_nos4(void) { compare_wrapper_vs_qg("nos4", SS_DIR "/nos4.mtx"); }

static void test_amd_qg_delegation_bcsstk04(void) {
    compare_wrapper_vs_qg("bcsstk04", SS_DIR "/bcsstk04.mtx");
}

static void test_amd_qg_delegation_bcsstk14(void) {
    compare_wrapper_vs_qg("bcsstk14", SS_DIR "/bcsstk14.mtx");
}

/* ─── Large regular banded stress fixture ─────────────────────────── */

/* This is the maintained "large regular input" guardrail for qg-AMD.  The
 * banded matrix keeps nnz linear in n while still exercising an input size
 * that is inappropriate for dense bitset-style adjacency. */
static void test_amd_stress_10k_banded(void) {
    /* Banded with bandwidth 5: each row has ≤ 11 nonzeros (5 above,
     * 5 below, 1 diagonal).  nnz ≈ 11 · n = 110 000, comfortably
     * inside the quotient-graph workspace's initial 5·nnz + 6·n + 1
     * allocation. */
    idx_t n = 10000;
    SparseMatrix *A = sparse_create(n, n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    /* Check sparse_insert returns: a partial fixture would surface
     * downstream as misleading timing or validity assertions.  On
     * any failure free A and bail out via REQUIRE_OK before the
     * test continues with an incomplete matrix. */
    for (idx_t i = 0; i < n; i++) {
        sparse_err_t ins_rc = sparse_insert(A, i, i, 1.0);
        for (idx_t k = 1; ins_rc == SPARSE_OK && k <= 5; k++) {
            if (i + k < n) {
                ins_rc = sparse_insert(A, i, i + k, 1.0);
                if (ins_rc == SPARSE_OK)
                    ins_rc = sparse_insert(A, i + k, i, 1.0);
            }
        }
        if (ins_rc != SPARSE_OK) {
            sparse_free(A);
            REQUIRE_OK(ins_rc);
            return;
        }
    }

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    /* Fail-fast on alloc — ASSERT_NOT_NULL is non-fatal in this
     * test framework, so without an early return sparse_reorder_amd
     * would receive a NULL perm and is_valid_permutation would
     * dereference NULL. */
    if (!perm) {
        sparse_free(A);
        REQUIRE_OK(SPARSE_ERR_ALLOC);
        return;
    }

    clock_t t0 = clock();
    sparse_err_t rc = sparse_reorder_amd(A, perm);
    double secs = (double)(clock() - t0) / (double)CLOCKS_PER_SEC;
    REQUIRE_OK(rc);
    ASSERT_TRUE(is_valid_permutation(perm, n));

    printf("    AMD on 10 000×10 000 banded (nnz=%d): %.2f s\n", (int)sparse_nnz(A), secs);

    /* The linked-list matrix backend makes `sparse_build_adj` the dominant
     * cost on this fixture.  Keep the ceiling broad: this is a structural
     * regression guard, not a portable timing benchmark. */
    ASSERT_TRUE(secs < 30.0);

    free(perm);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("quotient-graph AMD wrapper delegation and large-input guardrail");
    RUN_TEST(test_amd_qg_null_args);
    RUN_TEST(test_amd_qg_rejects_rectangular);
    RUN_TEST(test_amd_qg_singleton);
    RUN_TEST(test_amd_qg_delegation_nos4);
    RUN_TEST(test_amd_qg_delegation_bcsstk04);
    RUN_TEST(test_amd_qg_delegation_bcsstk14);
    RUN_TEST(test_amd_stress_10k_banded);
    TEST_SUITE_END();
}
