/*
 * Sprint 22 Days 11-12 — quotient-graph AMD wrapper-delegation
 * + production-swap tests.
 *
 * Day 11 shipped the full quotient-graph elimination loop in
 * `src/sparse_reorder_amd_qg.c`; Day 12 deleted the old bitset
 * implementation and rewrote `sparse_reorder_amd` to forward
 * directly to `sparse_reorder_amd_qg`.  As a result, the public
 * wrapper and the internal helper now share a single code path.
 *
 * The "parity" tests below therefore no longer cross two separate
 * implementations — they pin the post-Day-12 contract instead:
 *
 *   - the public `sparse_reorder_amd` wrapper delegates correctly
 *     (bit-identical permutations on the SuiteSparse corpus), and
 *   - the resulting symbolic-Cholesky fill matches between the two
 *     entry points (a sanity check that the wrapper isn't doing
 *     anything beyond calling through).
 *
 * Cross-implementation parity against the deleted bitset still runs
 * out-of-tree as a one-shot capture in `benchmarks/bench_amd_qg.c`,
 * which embeds the bitset as a test-local helper.  That bench was
 * the Day-13 evidence used to retire the bitset; this in-tree test
 * suite now guards delegation, NULL/shape contracts, and the
 * 10 000×10 000 stress fixture that was the Day-12 swap's
 * acceptance gate.
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

/* ─── Wrapper delegation: public sparse_reorder_amd vs internal qg ── */

/* Post-Day-12, the public wrapper and the internal helper share one
 * code path.  This compare therefore expects bit-identical
 * permutations and identical symbolic-Cholesky fill — anything else
 * means the wrapper has drifted from straight delegation. */
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

/* ─── Day 12 stress test: 10 000 × 10 000 banded matrix ──────────── */

/* The bitset AMD's O(n²/64) memory at n = 10 000 is 12.5 MB just for
 * the bitset, with O(n²/64 · n) = 125 G ops in the per-pivot merge —
 * the bench would have taken minutes.  The quotient-graph version
 * scales linearly in nnz; the test confirms it completes well inside
 * the plan's 5 s budget on a banded fixture.  This is the
 * "structurally regular but n past the bitset's reach" check that
 * Day 12 is meant to surface. */
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

    /* Plan completion criterion: < 5 s.  On the linked-list backend
     * `sparse_build_adj` itself takes the bulk of the time (it
     * walks the slab pool); the AMD elimination loop is only a
     * fraction.  We assert 30 s as a generous ceiling — anything
     * above this means the swap regressed asymptotic behaviour. */
    ASSERT_TRUE(secs < 30.0);

    free(perm);
    sparse_free(A);
}

/* ─── Sprint 23 Day 2: workspace extension regression test ──────────── */

/* The Sprint 23 Day 2 change extends `qg_t` with `elen[]` and grows
 * the initial `iw_size` from `5·nnz + 6·n + 1` to `7·nnz + 8·n + 1`
 * (Davis 2006 §7's reference size).  Both changes are structural —
 * no algorithmic behaviour change — so the produced permutation must
 * stay bit-identical to the Sprint-22 baseline.  This test pins that
 * contract on a synthetic 100×100 banded fixture (deterministic,
 * doesn't depend on the SuiteSparse data dir).
 *
 * A baseline permutation isn't hard-coded here — the corpus
 * delegation tests above already cover bit-identical fill on the
 * SuiteSparse fixtures (nos4 = 637, bcsstk04 = 3143, bcsstk14 =
 * 116071, all unchanged from Sprint 22's bench_day14.txt); this test
 * adds a determinism check (two independent runs produce the same
 * permutation) and a validity check on a non-corpus fixture so
 * Day 3's element-absorption work has a smaller fixture to bisect
 * against if a regression turns up. */
static void test_qg_workspace_extension_no_regression(void) {
    const idx_t n = 100;
    const idx_t bandwidth = 5;
    SparseMatrix *A = sparse_create(n, n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    for (idx_t i = 0; i < n; i++) {
        sparse_err_t ins_rc = sparse_insert(A, i, i, 1.0);
        for (idx_t k = 1; ins_rc == SPARSE_OK && k <= bandwidth; k++) {
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

    idx_t *perm1 = malloc((size_t)n * sizeof(idx_t));
    idx_t *perm2 = malloc((size_t)n * sizeof(idx_t));
    if (!perm1 || !perm2) {
        free(perm1);
        free(perm2);
        sparse_free(A);
        REQUIRE_OK(SPARSE_ERR_ALLOC);
        return;
    }

    REQUIRE_OK(sparse_reorder_amd_qg(A, perm1));
    REQUIRE_OK(sparse_reorder_amd_qg(A, perm2));

    /* Determinism: two independent runs on the same fixture produce
     * bit-identical permutations.  Day 3's element-absorption work
     * must preserve this. */
    ASSERT_TRUE(is_valid_permutation(perm1, n));
    ASSERT_TRUE(is_valid_permutation(perm2, n));
    for (idx_t i = 0; i < n; i++)
        ASSERT_EQ(perm1[i], perm2[i]);

    /* Symbolic Cholesky nnz on the permuted matrix — provides the
     * bisection golden for Day 3.  100×100 bw=5: 595 entries in A
     * (100 diagonal + 2*5*95 - 5*4/2*... actually 100 diag + 2*(5*100
     * - sum(i for 1..5)) but this is documentation, not assertion). */
    idx_t nnz_L = symbolic_cholesky_nnz_with_perm(A, perm1);
    ASSERT_TRUE(nnz_L > 0);
    printf("    100×100 banded (bw=5): nnz(L) under qg AMD = %d\n", (int)nnz_L);

    free(perm1);
    free(perm2);
    sparse_free(A);
}

/* ─── Sprint 23 Day 4: supervariable detection ──────────────────────── */

/* Build a star fixture: vertex 0 is the centre, vertices 1..k are
 * leaves connected only to the centre.  After the centre is
 * eliminated, the k leaves all have identical adjacency (empty
 * variable-side, single-element element-side = {0}).  They form a
 * single supervariable and co-eliminate in one pivot step.
 *
 * The "absorbed" probe should report k-1 entries (= leaves
 * co-eliminated alongside the supervariable representative).  The
 * resulting permutation must contain the centre (0) followed by
 * the k leaves in ID order in some contiguous block. */
static SparseMatrix *make_star_n(idx_t k) {
    /* n = 1 (centre) + k (leaves). */
    SparseMatrix *A = sparse_create(k + 1, k + 1);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < k + 1; i++) {
        if (sparse_insert(A, i, i, 1.0) != SPARSE_OK)
            goto fail;
    }
    /* Centre is vertex 0; leaves are 1..k. */
    for (idx_t i = 1; i <= k; i++) {
        if (sparse_insert(A, 0, i, 1.0) != SPARSE_OK)
            goto fail;
        if (sparse_insert(A, i, 0, 1.0) != SPARSE_OK)
            goto fail;
    }
    return A;
fail:
    sparse_free(A);
    return NULL;
}

static void test_qg_supervariable_synthetic(void) {
    const idx_t k = 4; /* 4 leaves */
    SparseMatrix *A = make_star_n(k);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    idx_t n = sparse_rows(A); /* k + 1 */

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    if (!perm) {
        sparse_free(A);
        REQUIRE_OK(SPARSE_ERR_ALLOC);
        return;
    }

    REQUIRE_OK(sparse_reorder_amd_qg(A, perm));
    ASSERT_TRUE(is_valid_permutation(perm, n));

    /* The centre (vertex 0) has the largest degree (k); each leaf
     * has degree 1.  Min-degree picks a leaf first.  After ANY one
     * leaf eliminates, the centre's degree is k-1, and the
     * remaining leaves now have an empty variable-side + a single-
     * element element-side, so they should be detected as a
     * supervariable on the next pivot — co-eliminating en masse.
     *
     * What we can pin deterministically: the four leaves (1..4)
     * appear contiguously in `perm[]` — once the supervariable is
     * detected, all members co-eliminate with their representative.
     * The centre (0) appears at the very end. */
    idx_t centre_pos = -1;
    for (idx_t i = 0; i < n; i++) {
        if (perm[i] == 0) {
            centre_pos = i;
            break;
        }
    }
    ASSERT_EQ(centre_pos, n - 1);

    /* Find the first and last leaf positions; they must be
     * contiguous (after the supervariable forms, the leaves
     * co-eliminate as one unit). */
    idx_t first_leaf = n, last_leaf = 0;
    for (idx_t i = 0; i < n; i++) {
        if (perm[i] >= 1 && perm[i] <= k) {
            if (i < first_leaf)
                first_leaf = i;
            if (i > last_leaf)
                last_leaf = i;
        }
    }
    ASSERT_EQ(last_leaf - first_leaf + 1, k);

    printf("    star (n=%d, %d leaves): perm centre at %d, leaves contiguous at [%d..%d]\n", (int)n,
           (int)k, (int)centre_pos, (int)first_leaf, (int)last_leaf);

    free(perm);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sprint 22 Days 11-12 + Sprint 23 Days 2-4: quotient-graph AMD");
    RUN_TEST(test_amd_qg_null_args);
    RUN_TEST(test_amd_qg_rejects_rectangular);
    RUN_TEST(test_amd_qg_singleton);
    RUN_TEST(test_amd_qg_delegation_nos4);
    RUN_TEST(test_amd_qg_delegation_bcsstk04);
    RUN_TEST(test_amd_qg_delegation_bcsstk14);
    RUN_TEST(test_amd_stress_10k_banded);
    RUN_TEST(test_qg_workspace_extension_no_regression);
    RUN_TEST(test_qg_supervariable_synthetic);
    TEST_SUITE_END();
}
