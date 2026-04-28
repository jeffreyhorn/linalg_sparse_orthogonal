/*
 * Sprint 22 Day 1 — graph partitioner skeleton tests.
 *
 * Coverage today:
 *   - `sparse_graph_from_sparse` round-trips through `sparse_graph_free`
 *     cleanly on the SuiteSparse nos4 fixture (smoke test under ASan).
 *   - Same round-trip on a synthetic 1×1 fixture (singleton boundary
 *     for the constructor's degree-0 path).
 *   - Argument validation: rectangular A is rejected with SHAPE,
 *     NULL args produce SPARSE_ERR_NULL.  Both error paths leave G
 *     in the pre-cleared empty state.
 *   - The Day 1 stubs (`sparse_graph_partition`, `sparse_graph_subgraph`)
 *     return SPARSE_ERR_BADARG as documented — locks in the "stub in
 *     progress" signal so a future commit can't silently change the
 *     contract.
 *
 * Note: the constructor's `n == 0` branch is reachable only from
 * internal callers (e.g., the Day 6 recursive ND driver passing an
 * empty subgraph) — `sparse_create(0, 0)` returns NULL, so there's
 * no public-API path to construct a 0×0 SparseMatrix to feed in
 * here.  The branch stays for defensive correctness; coverage
 * arrives once Day 6 lands a caller.
 *
 * Days 2-5 will replace these stubs with real coverage of coarsening,
 * bisection, FM refinement, and vertex-separator extraction.
 */

#include "sparse_graph_internal.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <stdlib.h>
#include <string.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ─── graph_from_sparse / graph_free round-trip ───────────────────── */

static void test_graph_from_sparse_nos4_round_trip(void) {
    SparseMatrix *A = NULL;
    sparse_err_t rc = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    if (rc != SPARSE_OK) {
        printf("    skipped (nos4 fixture not loadable: %d)\n", (int)rc);
        return;
    }

    sparse_graph_t G = {0};
    ASSERT_EQ(sparse_graph_from_sparse(A, &G), SPARSE_OK);

    /* Structural invariants: n matches A; xadj is monotonic; xadj[n]
     * equals the total adjacency length; adjncy entries are in range
     * and contain no self-loops. */
    ASSERT_EQ(G.n, sparse_rows(A));
    ASSERT_NOT_NULL(G.xadj);
    if (G.n > 0)
        ASSERT_NOT_NULL(G.adjncy);
    ASSERT_EQ(G.xadj[0], 0);
    for (idx_t i = 0; i < G.n; i++) {
        ASSERT_TRUE(G.xadj[i + 1] >= G.xadj[i]);
        for (idx_t k = G.xadj[i]; k < G.xadj[i + 1]; k++) {
            idx_t j = G.adjncy[k];
            ASSERT_TRUE(j >= 0 && j < G.n);
            ASSERT_TRUE(j != i);
        }
    }

    /* Day 1 contract: vwgt / ewgt are NULL on the root graph. */
    ASSERT_TRUE(G.vwgt == NULL);
    ASSERT_TRUE(G.ewgt == NULL);

    sparse_graph_free(&G);
    /* Free is idempotent and safe on an empty struct. */
    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Singleton (n == 1, no edges) ─────────────────────────────────── */

static void test_graph_from_sparse_singleton(void) {
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_NOT_NULL(A);
    sparse_insert(A, 0, 0, 1.0); /* diagonal — drops via the self-loop filter */
    sparse_graph_t G = {0};
    ASSERT_EQ(sparse_graph_from_sparse(A, &G), SPARSE_OK);
    ASSERT_EQ(G.n, 1);
    ASSERT_EQ(G.xadj[0], 0);
    ASSERT_EQ(G.xadj[1], 0); /* no neighbours */
    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Rectangular A is rejected with SHAPE ─────────────────────────── */

static void test_graph_from_sparse_rejects_rectangular(void) {
    SparseMatrix *A = sparse_create(3, 5);
    ASSERT_NOT_NULL(A);
    sparse_graph_t G = {0};
    ASSERT_ERR(sparse_graph_from_sparse(A, &G), SPARSE_ERR_SHAPE);
    /* G is left in the empty state by the constructor's pre-clear. */
    ASSERT_EQ(G.n, 0);
    ASSERT_TRUE(G.xadj == NULL);
    sparse_graph_free(&G); /* no-op, but must be safe */
    sparse_free(A);
}

/* ─── NULL handling ────────────────────────────────────────────────── */

static void test_graph_from_sparse_null_args(void) {
    sparse_graph_t G = {0};
    SparseMatrix *A = sparse_create(2, 2);
    ASSERT_NOT_NULL(A);
    ASSERT_ERR(sparse_graph_from_sparse(NULL, &G), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_graph_from_sparse(A, NULL), SPARSE_ERR_NULL);
    sparse_graph_free(NULL); /* NULL-safe per contract */
    sparse_free(A);
}

/* ─── Day 1 stubs return SPARSE_ERR_BADARG ─────────────────────────── */

static void test_graph_partition_is_stub(void) {
    sparse_graph_t G = {0};
    idx_t part[1] = {0};
    idx_t sep = 0;
    ASSERT_ERR(sparse_graph_partition(&G, part, &sep), SPARSE_ERR_BADARG);
}

static void test_graph_subgraph_is_stub(void) {
    sparse_graph_t parent = {0};
    sparse_graph_t child = {0};
    idx_t vs[1] = {0};
    idx_t map[1] = {0};
    ASSERT_ERR(sparse_graph_subgraph(&parent, vs, 1, &child, map), SPARSE_ERR_BADARG);
}

/* ═══════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sprint 22 Day 1: sparse_graph_t round-trip & stubs");

    /* Round-trip + structural invariants */
    RUN_TEST(test_graph_from_sparse_nos4_round_trip);
    RUN_TEST(test_graph_from_sparse_singleton);

    /* Argument validation */
    RUN_TEST(test_graph_from_sparse_rejects_rectangular);
    RUN_TEST(test_graph_from_sparse_null_args);

    /* Day 1 stub contracts */
    RUN_TEST(test_graph_partition_is_stub);
    RUN_TEST(test_graph_subgraph_is_stub);

    TEST_SUITE_END();
}
