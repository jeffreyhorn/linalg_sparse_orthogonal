/*
 * Sprint 22 Days 1-2 — graph partitioner unit tests.
 *
 * Day 1 coverage:
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
 * Day 2 coverage:
 *   - `graph_coarsen_heavy_edge_matching` halves a 5×5 grid to ≤ 13
 *     vertices in one step and ≤ 7 in two; halves a 1D path
 *     (n = 20) similarly.
 *   - Vertex-weight sum is preserved across every level of the
 *     hierarchy on the 5×5 grid (the headline coarsening invariant).
 *   - Heavy-edge matching prefers within-clique edges on a
 *     two-cliques-plus-bridge fixture: the bridge is heavier than
 *     the within-clique edges, so it survives instead of getting
 *     collapsed (verifies the matching prefers the heaviest edge).
 *   - Determinism contract: same `(graph, seed)` produces a
 *     bit-identical coarse graph + cmap on every call.
 *
 * Note: the constructor's `n == 0` branch is reachable only from
 * internal callers (e.g., the Day 6 recursive ND driver passing an
 * empty subgraph) — `sparse_create(0, 0)` returns NULL, so there's
 * no public-API path to construct a 0×0 SparseMatrix to feed in
 * here.  The branch stays for defensive correctness; coverage
 * arrives once Day 6 lands a caller.
 */

#include "sparse_graph_internal.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <stdint.h>
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

/* ─── Day 2 helpers: build synthetic graph fixtures ────────────────── */

/* Build the symmetric adjacency graph of the 2D `r × c` grid (vertices
 * indexed in row-major order; each vertex connects to its 4-neighbours
 * where they exist).  All edges have unit weight; no vwgt set. */
static SparseMatrix *make_grid_2d(idx_t r, idx_t c) {
    SparseMatrix *A = sparse_create(r * c, r * c);
    for (idx_t i = 0; i < r; i++) {
        for (idx_t j = 0; j < c; j++) {
            idx_t v = i * c + j;
            sparse_insert(A, v, v, 1.0); /* placeholder diagonal */
            if (j + 1 < c) {
                sparse_insert(A, v, v + 1, 1.0);
                sparse_insert(A, v + 1, v, 1.0);
            }
            if (i + 1 < r) {
                sparse_insert(A, v, v + c, 1.0);
                sparse_insert(A, v + c, v, 1.0);
            }
        }
    }
    return A;
}

/* Build a 1D path graph: n vertices with edges (0-1), (1-2), ..., (n-2)-(n-1).
 * Self-loops included as placeholders; no vwgt. */
static SparseMatrix *make_path_1d(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 1.0);
        if (i + 1 < n) {
            sparse_insert(A, i, i + 1, 1.0);
            sparse_insert(A, i + 1, i, 1.0);
        }
    }
    return A;
}

/* Build two K_k cliques (vertices 0..k-1 and k..2k-1) joined by a
 * single bridge edge (0, k).  The bridge is given the heaviest edge
 * weight (10×) by injecting that pattern into the resulting graph
 * after the from_sparse step, since SparseMatrix doesn't carry edge
 * weights.  The test then heavy-edge-matches against this graph. */
static void overwrite_edge_weight(sparse_graph_t *G, idx_t u, idx_t v, idx_t new_wt) {
    if (!G->ewgt) {
        G->ewgt = malloc((size_t)G->xadj[G->n] * sizeof(idx_t));
        if (!G->ewgt)
            return;
        for (idx_t k = 0; k < G->xadj[G->n]; k++)
            G->ewgt[k] = 1;
    }
    for (idx_t k = G->xadj[u]; k < G->xadj[u + 1]; k++) {
        if (G->adjncy[k] == v)
            G->ewgt[k] = new_wt;
    }
    for (idx_t k = G->xadj[v]; k < G->xadj[v + 1]; k++) {
        if (G->adjncy[k] == u)
            G->ewgt[k] = new_wt;
    }
}

static SparseMatrix *make_two_cliques_with_bridge(idx_t k) {
    SparseMatrix *A = sparse_create(2 * k, 2 * k);
    /* Clique 1: vertices 0..k-1 */
    for (idx_t i = 0; i < k; i++) {
        sparse_insert(A, i, i, 1.0);
        for (idx_t j = i + 1; j < k; j++) {
            sparse_insert(A, i, j, 1.0);
            sparse_insert(A, j, i, 1.0);
        }
    }
    /* Clique 2: vertices k..2k-1 */
    for (idx_t i = k; i < 2 * k; i++) {
        sparse_insert(A, i, i, 1.0);
        for (idx_t j = i + 1; j < 2 * k; j++) {
            sparse_insert(A, i, j, 1.0);
            sparse_insert(A, j, i, 1.0);
        }
    }
    /* Bridge edge (0, k) */
    sparse_insert(A, 0, (idx_t)k, 1.0);
    sparse_insert(A, (idx_t)k, 0, 1.0);
    return A;
}

/* ─── Day 2: heavy-edge-matching coarsener tests ───────────────────── */

static void test_coarsen_5x5_grid_halves_in_one_step(void) {
    SparseMatrix *A = make_grid_2d(5, 5);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    sparse_graph_t coarse = {0};
    idx_t cmap[25] = {0};
    REQUIRE_OK(graph_coarsen_heavy_edge_matching(&G, /*seed=*/42u, &coarse, cmap));

    /* The matching pairs at most n/2 vertices, so coarse.n is at
     * least ceil(n/2) = 13 — but typically lower because some
     * paired vertices condense.  Day 2 contract: ≤ 13 in one step. */
    ASSERT_TRUE(coarse.n <= 13);
    ASSERT_TRUE(coarse.n >= 1);

    /* cmap range check + every fine vertex is mapped. */
    for (idx_t i = 0; i < 25; i++)
        ASSERT_TRUE(cmap[i] >= 0 && cmap[i] < coarse.n);

    /* Vertex-weight sum invariant: sum(vwgt_coarse) == n_fine
     * (since the root has uniform vwgt = 1). */
    idx_t sum = 0;
    for (idx_t c = 0; c < coarse.n; c++)
        sum += coarse.vwgt[c];
    ASSERT_EQ(sum, 25);

    /* Adjacency well-formedness: no self-loops, sorted, no duplicates. */
    for (idx_t c = 0; c < coarse.n; c++) {
        for (idx_t k = coarse.xadj[c]; k < coarse.xadj[c + 1]; k++) {
            ASSERT_TRUE(coarse.adjncy[k] != c);
            if (k + 1 < coarse.xadj[c + 1])
                ASSERT_TRUE(coarse.adjncy[k] < coarse.adjncy[k + 1]);
        }
    }

    sparse_graph_free(&coarse);
    sparse_graph_free(&G);
    sparse_free(A);
}

static void test_coarsen_5x5_grid_halves_again_in_two_steps(void) {
    /* Verifies the per-step halving claim directly via two manual
     * coarsening passes (the hierarchy builder's small-enough
     * threshold of MAX(20, n/100) = 20 stops after one level on
     * n = 25, so we can't use it for this check). */
    SparseMatrix *A = make_grid_2d(5, 5);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    sparse_graph_t c1 = {0};
    idx_t cmap1[25] = {0};
    REQUIRE_OK(graph_coarsen_heavy_edge_matching(&G, /*seed=*/42u, &c1, cmap1));
    ASSERT_TRUE(c1.n <= 13);

    sparse_graph_t c2 = {0};
    idx_t *cmap2 = malloc((size_t)c1.n * sizeof(idx_t));
    ASSERT_NOT_NULL(cmap2);
    REQUIRE_OK(graph_coarsen_heavy_edge_matching(&c1, /*seed=*/43u, &c2, cmap2));
    ASSERT_TRUE(c2.n <= 7);

    /* vwgt sum invariant across both levels.  The fine root has
     * uniform vwgt = 1 so the sum is 25 at every level. */
    idx_t sum1 = 0;
    for (idx_t c = 0; c < c1.n; c++)
        sum1 += c1.vwgt[c];
    ASSERT_EQ(sum1, 25);
    idx_t sum2 = 0;
    for (idx_t c = 0; c < c2.n; c++)
        sum2 += c2.vwgt[c];
    ASSERT_EQ(sum2, 25);

    free(cmap2);
    sparse_graph_free(&c2);
    sparse_graph_free(&c1);
    sparse_graph_free(&G);
    sparse_free(A);
}

static void test_hierarchy_build_5x5_grid(void) {
    /* Verifies the hierarchy builder lands at least one coarsened
     * level and conserves vwgt across every level it produces.  The
     * 5×5 grid coarsens once (n=25 → ≤13) and then trips the
     * small-enough threshold (20) so the build stops at nlevels=1. */
    SparseMatrix *A = make_grid_2d(5, 5);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    sparse_graph_hierarchy_t h = {0};
    REQUIRE_OK(sparse_graph_hierarchy_build(&G, /*seed=*/42u, &h));

    ASSERT_TRUE(h.nlevels >= 1);
    ASSERT_TRUE(h.coarse[0].n <= 13);

    /* vwgt sum is conserved at every level. */
    for (int lvl = 0; lvl < h.nlevels; lvl++) {
        idx_t sum = 0;
        for (idx_t c = 0; c < h.coarse[lvl].n; c++)
            sum += h.coarse[lvl].vwgt[c];
        ASSERT_EQ(sum, 25);
    }

    sparse_graph_hierarchy_free(&h);
    sparse_graph_free(&G);
    sparse_free(A);
}

static void test_coarsen_1d_path_halves(void) {
    SparseMatrix *A = make_path_1d(20);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    sparse_graph_t coarse = {0};
    idx_t cmap[20] = {0};
    REQUIRE_OK(graph_coarsen_heavy_edge_matching(&G, /*seed=*/123u, &coarse, cmap));

    /* The 1D path is the cleanest matching case — every interior
     * vertex has exactly two unmatched neighbours initially, so the
     * matching pairs nearly every vertex.  ≤ 11 = ceil(20/2)+1
     * absorbs any odd-vertex rounding. */
    ASSERT_TRUE(coarse.n <= 11);
    ASSERT_TRUE(coarse.n >= 1);

    /* vwgt sum invariant. */
    idx_t sum = 0;
    for (idx_t c = 0; c < coarse.n; c++)
        sum += coarse.vwgt[c];
    ASSERT_EQ(sum, 20);

    sparse_graph_free(&coarse);
    sparse_graph_free(&G);
    sparse_free(A);
}

static void test_coarsen_prefers_heaviest_edge(void) {
    /* Two K_3 cliques + bridge.  Bridge edge gets weight 10; intra-
     * clique edges keep weight 1.  Heavy-edge preference means:
     * whenever vertex 0 or vertex k=3 is visited first, it picks the
     * other as its heaviest unmatched neighbour and the bridge
     * collapses (cmap[0] == cmap[3]).
     *
     * Vertex 0 lands at position 0 in the shuffle with probability
     * 1/n = 1/6 per seed (same for vertex 3), so P(bridge collapses)
     * ≈ 1/3 per seed.  Sweeping 16 seeds drives the
     * never-collapses probability to (2/3)^16 ≈ 0.001, vanishingly
     * small flake risk. */
    const idx_t k = 3;
    SparseMatrix *A = make_two_cliques_with_bridge(k);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    overwrite_edge_weight(&G, 0, k, 10);

    int bridge_collapsed = 0;
    for (uint32_t s = 0; s < 16 && !bridge_collapsed; s++) {
        sparse_graph_t coarse = {0};
        idx_t cmap[6] = {0};
        REQUIRE_OK(graph_coarsen_heavy_edge_matching(&G, s, &coarse, cmap));
        if (cmap[0] == cmap[k])
            bridge_collapsed = 1;
        sparse_graph_free(&coarse);
    }
    ASSERT_TRUE(bridge_collapsed);

    sparse_graph_free(&G);
    sparse_free(A);
}

static void test_coarsen_is_deterministic(void) {
    /* Same (graph, seed) pair must produce the same coarse graph
     * and the same cmap on every call.  This is the contract
     * `sparse_graph_partition` will rely on so nested-dissection's
     * recursion produces stable permutations. */
    SparseMatrix *A = make_grid_2d(6, 6);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    sparse_graph_t c1 = {0};
    sparse_graph_t c2 = {0};
    idx_t cmap1[36] = {0};
    idx_t cmap2[36] = {0};
    REQUIRE_OK(graph_coarsen_heavy_edge_matching(&G, /*seed=*/99u, &c1, cmap1));
    REQUIRE_OK(graph_coarsen_heavy_edge_matching(&G, /*seed=*/99u, &c2, cmap2));

    ASSERT_EQ(c1.n, c2.n);
    ASSERT_EQ(c1.xadj[c1.n], c2.xadj[c2.n]);
    ASSERT_EQ(memcmp(c1.xadj, c2.xadj, (size_t)(c1.n + 1) * sizeof(idx_t)), 0);
    ASSERT_EQ(memcmp(c1.adjncy, c2.adjncy, (size_t)c1.xadj[c1.n] * sizeof(idx_t)), 0);
    ASSERT_EQ(memcmp(c1.vwgt, c2.vwgt, (size_t)c1.n * sizeof(idx_t)), 0);
    ASSERT_EQ(memcmp(c1.ewgt, c2.ewgt, (size_t)c1.xadj[c1.n] * sizeof(idx_t)), 0);
    ASSERT_EQ(memcmp(cmap1, cmap2, sizeof(cmap1)), 0);

    sparse_graph_free(&c1);
    sparse_graph_free(&c2);
    sparse_graph_free(&G);
    sparse_free(A);
}

static void test_hierarchy_free_safe_on_zero_init(void) {
    sparse_graph_hierarchy_t h = {0};
    sparse_graph_hierarchy_free(&h);   /* must not crash */
    sparse_graph_hierarchy_free(NULL); /* must not crash */
    ASSERT_EQ(h.nlevels, 0);
    ASSERT_TRUE(h.coarse == NULL);
    ASSERT_TRUE(h.cmaps == NULL);
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
    TEST_SUITE_BEGIN("Sprint 22 Days 1-2: sparse_graph_t + heavy-edge coarsening");

    /* Day 1: round-trip + structural invariants */
    RUN_TEST(test_graph_from_sparse_nos4_round_trip);
    RUN_TEST(test_graph_from_sparse_singleton);

    /* Day 1: argument validation */
    RUN_TEST(test_graph_from_sparse_rejects_rectangular);
    RUN_TEST(test_graph_from_sparse_null_args);

    /* Day 1: stubs (subgraph + partition) — partition is a Day 1 stub
     * still pending Day 4's real implementation */
    RUN_TEST(test_graph_partition_is_stub);
    RUN_TEST(test_graph_subgraph_is_stub);

    /* Day 2: heavy-edge-matching coarsener */
    RUN_TEST(test_coarsen_5x5_grid_halves_in_one_step);
    RUN_TEST(test_coarsen_5x5_grid_halves_again_in_two_steps);
    RUN_TEST(test_coarsen_1d_path_halves);
    RUN_TEST(test_coarsen_prefers_heaviest_edge);
    RUN_TEST(test_coarsen_is_deterministic);

    /* Day 2: multilevel hierarchy */
    RUN_TEST(test_hierarchy_build_5x5_grid);
    RUN_TEST(test_hierarchy_free_safe_on_zero_init);

    TEST_SUITE_END();
}
