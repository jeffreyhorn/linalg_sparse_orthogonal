/*
 * Sprint 22 Day 11 — quotient-graph AMD parallel-comparison tests.
 *
 * Day 10 shipped the design block + stub; Day 11 ships the full
 * elimination loop in `src/sparse_reorder_amd_qg.c`.  Today's tests
 * run *both* AMD implementations side-by-side on the SuiteSparse
 * corpus and assert fill-quality parity:
 *
 *   nnz_qg ≤ 1.05 × nnz_bitset
 *
 * The 5%-point margin is the plan's tolerance — pivot tie-breaking
 * differs across the two implementations (the bitset version picks
 * the first-encountered minimum-degree vertex, the quotient-graph
 * version walks the same ordering but with different early
 * termination on degree updates), so bit-identical permutations
 * aren't expected.  Fill nnz under symbolic Cholesky is the headline
 * quality metric.
 *
 * Day 12 swaps the production `sparse_reorder_amd` body to call the
 * quotient-graph helper; today both paths still run side-by-side so
 * any regression in the new implementation surfaces immediately.
 */

#include "sparse_analysis.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_reorder_amd_qg_internal.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ─── Day 10 stub-contract retire ─────────────────────────────────── */

/* The Day-10 "stub returns BADARG" test is removed now that the
 * implementation lands.  NULL / non-square argument validation
 * stays — those checks live ahead of the elimination loop and the
 * tests pin them to the same contract as the bitset
 * sparse_reorder_amd. */

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
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};
    sparse_err_t rc = sparse_analyze(PA, &opts, &analysis);
    idx_t nnz = (rc == SPARSE_OK) ? analysis.sym_L.nnz : (idx_t)-1;
    sparse_analysis_free(&analysis);
    sparse_free(PA);
    return nnz;
}

/* ─── Parallel comparison: bitset AMD vs quotient-graph AMD ──────── */

static void compare_bitset_vs_qg(const char *fixture, const char *path, double max_ratio) {
    SparseMatrix *A = NULL;
    sparse_err_t rc = sparse_load_mm(&A, path);
    if (rc != SPARSE_OK) {
        printf("    skipped (%s not loadable: %d)\n", fixture, (int)rc);
        return;
    }
    idx_t n = sparse_rows(A);

    idx_t *perm_bitset = malloc((size_t)n * sizeof(idx_t));
    idx_t *perm_qg = malloc((size_t)n * sizeof(idx_t));
    ASSERT_NOT_NULL(perm_bitset);
    ASSERT_NOT_NULL(perm_qg);

    REQUIRE_OK(sparse_reorder_amd(A, perm_bitset));
    REQUIRE_OK(sparse_reorder_amd_qg(A, perm_qg));

    /* Both must be valid permutations of [0, n). */
    ASSERT_TRUE(is_valid_permutation(perm_bitset, n));
    ASSERT_TRUE(is_valid_permutation(perm_qg, n));

    idx_t nnz_bitset = symbolic_cholesky_nnz_with_perm(A, perm_bitset);
    idx_t nnz_qg = symbolic_cholesky_nnz_with_perm(A, perm_qg);
    ASSERT_TRUE(nnz_bitset > 0);
    ASSERT_TRUE(nnz_qg > 0);

    double ratio = (double)nnz_qg / (double)nnz_bitset;
    printf("    %s (n=%d): bitset nnz(L) = %d, qg nnz(L) = %d (qg/bitset = %.3f)\n", fixture,
           (int)n, (int)nnz_bitset, (int)nnz_qg, ratio);

    /* Plan target: fill parity within `max_ratio` (typically 1.05). */
    ASSERT_TRUE(ratio <= max_ratio);

    free(perm_bitset);
    free(perm_qg);
    sparse_free(A);
}

static void test_amd_qg_parity_nos4(void) {
    compare_bitset_vs_qg("nos4", SS_DIR "/nos4.mtx", 1.05);
}

static void test_amd_qg_parity_bcsstk04(void) {
    compare_bitset_vs_qg("bcsstk04", SS_DIR "/bcsstk04.mtx", 1.05);
}

static void test_amd_qg_parity_bcsstk14(void) {
    compare_bitset_vs_qg("bcsstk14", SS_DIR "/bcsstk14.mtx", 1.05);
}

/* ═══════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sprint 22 Day 11: quotient-graph AMD vs bitset parity");
    RUN_TEST(test_amd_qg_null_args);
    RUN_TEST(test_amd_qg_rejects_rectangular);
    RUN_TEST(test_amd_qg_singleton);
    RUN_TEST(test_amd_qg_parity_nos4);
    RUN_TEST(test_amd_qg_parity_bcsstk04);
    RUN_TEST(test_amd_qg_parity_bcsstk14);
    TEST_SUITE_END();
}
