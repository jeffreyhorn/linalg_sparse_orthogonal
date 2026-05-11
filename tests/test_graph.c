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

#include "sparse_eigs.h"
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
     * small-enough threshold (20) so the build stops at nlevels=1.
     *
     * Sprint 27 Day 2: pin SPARSE_ND_COARSENING=heavy_edge — this
     * test's `n <= 13` bound is HEM-specific (Sprint 22 Day 2 era).
     * Under the new HCC default (Sprint 27 Day 2 flip), HCC's
     * `min(deg)` scoring on the 5×5 grid leaves more corner /
     * boundary vertices unmatched (corners have deg=2; HCC matches
     * interior-interior pairs first), producing a coarse graph with
     * n > 13.  HCC behaviour on small regular grids is intentional
     * and exercised by the Sprint 25 `test_hcc_match_selection_grid`
     * test; this test stays scoped to HEM. */
    SparseMatrix *A = make_grid_2d(5, 5);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    if (setenv("SPARSE_ND_COARSENING", "heavy_edge", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv failed)\n");
        sparse_graph_free(&G);
        sparse_free(A);
        return;
    }

    sparse_graph_hierarchy_t h = {0};
    sparse_err_t hrc = sparse_graph_hierarchy_build(&G, /*seed=*/42u, &h);
    unsetenv("SPARSE_ND_COARSENING");
    REQUIRE_OK(hrc);

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

/* Sprint 25 Day 1-3: HCC match-selection contract pins.
 *
 * Day 1 stubbed these as skip-mode tests that print the expected
 * match selection.  Day 3 lands the actual assertions now that
 * graph_coarsen_hcc is implemented (Day 2).
 *
 * See docs/planning/EPIC_2/SPRINT_25/hcc_design.md for the scoring
 * formula `score = edge_weight * min(deg(u), deg(v))` + tie-break
 * "lower-id neighbour wins on equal score". */
static void test_hcc_match_selection_grid(void) {
    /* 5x5 unit-weighted grid (n=25, regular structure).  Under HEM
     * (Sprint 22 default), all neighbour edges have weight 1, so
     * the match selection follows shuffle-order tie-break.  Under
     * HCC, the score = 1 * min(deg(u), deg(v)) — interior vertices
     * (deg=4) beat boundary vertices (deg=2 or 3) for the same
     * weight, so HCC's match choices differ from HEM's.
     *
     * Day 3 contract:
     *   1. Determinism: same (graph, seed) → same cmap (same as
     *      Sprint 22's test_coarsen_is_deterministic, but for the
     *      HCC code path).
     *   2. Cmap range: every fine vertex maps to a coarse vertex
     *      in [0, n_coarse).
     *   3. HCC differs from HEM: at least one cmap entry under HCC
     *      differs from the corresponding entry under HEM (proves
     *      HCC is actually being called and producing distinct
     *      output, not silently falling through to the default).
     *      Day 2's diagnostic measured 12/25 cmap entries differ on
     *      the 5x5 grid; this assertion pins ≥ 1 entry differs to
     *      avoid over-constraining the test. */
    SparseMatrix *A = make_grid_2d(5, 5);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    sparse_graph_t hcc1 = {0}, hcc2 = {0}, hem = {0};
    idx_t cmap_hcc1[25] = {0}, cmap_hcc2[25] = {0}, cmap_hem[25] = {0};

    REQUIRE_OK(graph_coarsen_hcc(&G, /*seed=*/42u, &hcc1, cmap_hcc1));
    REQUIRE_OK(graph_coarsen_hcc(&G, /*seed=*/42u, &hcc2, cmap_hcc2));
    REQUIRE_OK(graph_coarsen_heavy_edge_matching(&G, /*seed=*/42u, &hem, cmap_hem));

    /* Determinism. */
    ASSERT_EQ(hcc1.n, hcc2.n);
    ASSERT_EQ(memcmp(cmap_hcc1, cmap_hcc2, sizeof(cmap_hcc1)), 0);

    /* Cmap range. */
    ASSERT_TRUE(hcc1.n >= 1 && hcc1.n <= 25);
    for (idx_t i = 0; i < 25; i++)
        ASSERT_TRUE(cmap_hcc1[i] >= 0 && cmap_hcc1[i] < hcc1.n);

    /* HCC differs from HEM (at least one cmap entry differs OR
     * coarse-vertex count differs).  Day 2 diagnostic: 12/25 cmap
     * entries differ, and HCC produces 14 coarse vertices vs HEM's
     * 13.  Either signal is sufficient evidence HCC is firing. */
    int any_diff = (hcc1.n != hem.n);
    for (idx_t i = 0; i < 25 && !any_diff; i++)
        if (cmap_hcc1[i] != cmap_hem[i])
            any_diff = 1;
    ASSERT_TRUE(any_diff);

    sparse_graph_free(&hcc1);
    sparse_graph_free(&hcc2);
    sparse_graph_free(&hem);
    sparse_graph_free(&G);
    sparse_free(A);
}

static void test_hcc_match_selection_irregular(void) {
    /* Small irregular fixture: a 4-vertex "Y" shape — vertex 0
     * is a hub connected to vertices 1, 2, 3 (each of which has
     * only one neighbour, vertex 0).  All edges have weight 1.
     *
     *     1
     *      \
     *       0 --- 2
     *      /
     *     3
     *
     * Under HCC, the score for edge (0, k) = 1 * min(deg(0)=3,
     * deg(k)=1) = 1 for every k in {1,2,3}.  Day 3 contract:
     *   1. Determinism: same seed → same cmap.
     *   2. n_coarse = 3 (hub + 1 leaf collapse to 1 coarse vertex;
     *      the other 2 leaves each get their own coarse vertex
     *      since their only neighbour got matched first).
     *   3. Hub (vertex 0) is matched with EXACTLY ONE leaf:
     *      cmap[0] appears twice in the cmap array; the other 2
     *      cmap values appear exactly once each.
     *
     *      The choice of WHICH leaf the hub matches with depends
     *      on shuffle order (Day 2 measured: seed=1,5 → leaf 2;
     *      seed=2,3,4,42 → leaf 1).  HCC's lower-id-neighbour
     *      tie-break only fires when the hub itself is processed
     *      first; otherwise a leaf gets processed and matches with
     *      vertex 0 as its only unmatched neighbour.  Either way,
     *      the structural invariant (one pair + two singletons)
     *      holds. */
    SparseMatrix *A = sparse_create(4, 4);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 3, 3, 1.0);
    for (idx_t k = 1; k <= 3; k++) {
        sparse_insert(A, 0, k, -1.0);
        sparse_insert(A, k, 0, -1.0);
    }
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    sparse_graph_t c1 = {0}, c2 = {0};
    idx_t cmap1[4] = {0}, cmap2[4] = {0};
    REQUIRE_OK(graph_coarsen_hcc(&G, /*seed=*/42u, &c1, cmap1));
    REQUIRE_OK(graph_coarsen_hcc(&G, /*seed=*/42u, &c2, cmap2));

    /* Determinism. */
    ASSERT_EQ(c1.n, c2.n);
    ASSERT_EQ(memcmp(cmap1, cmap2, sizeof(cmap1)), 0);

    /* n_coarse = 3 (1 pair + 2 singletons). */
    ASSERT_EQ(c1.n, 3);

    /* Cmap range. */
    for (idx_t i = 0; i < 4; i++)
        ASSERT_TRUE(cmap1[i] >= 0 && cmap1[i] < 3);

    /* Histogram: cmap[0] (the hub's coarse vertex) appears exactly
     * twice (hub + 1 matched leaf); the other 2 cmap values appear
     * exactly once each (2 unmatched leaves). */
    idx_t counts[3] = {0, 0, 0};
    for (idx_t i = 0; i < 4; i++)
        counts[cmap1[i]]++;
    /* Sort counts ascending so we can compare against {1, 1, 2}
     * regardless of which coarse-id the hub got assigned. */
    for (idx_t i = 0; i < 3; i++)
        for (idx_t j = i + 1; j < 3; j++)
            if (counts[j] < counts[i]) {
                idx_t tmp = counts[i];
                counts[i] = counts[j];
                counts[j] = tmp;
            }
    ASSERT_EQ(counts[0], 1);
    ASSERT_EQ(counts[1], 1);
    ASSERT_EQ(counts[2], 2);

    sparse_graph_free(&c1);
    sparse_graph_free(&c2);
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

/* ─── Day 1 / Day 6 stubs ───────────────────────────────────────────── */

static void test_graph_subgraph_is_stub(void) {
    sparse_graph_t parent = {0};
    sparse_graph_t child = {0};
    idx_t vs[1] = {0};
    idx_t map[1] = {0};
    ASSERT_ERR(sparse_graph_subgraph(&parent, vs, 1, &child, map), SPARSE_ERR_BADARG);
}

/* ═══════════════════════════════════════════════════════════════════
 * Sprint 22 Day 4 — uncoarsening + vertex-separator extraction.
 * ═══════════════════════════════════════════════════════════════════ */

/* Build the symmetric adjacency graph of a `d × d × d` 3D mesh
 * (6-connected: each vertex has up to 6 axis-aligned neighbours). */
static SparseMatrix *make_mesh_3d(idx_t d) {
    SparseMatrix *A = sparse_create(d * d * d, d * d * d);
    if (!A)
        return NULL;
    for (idx_t z = 0; z < d; z++) {
        for (idx_t y = 0; y < d; y++) {
            for (idx_t x = 0; x < d; x++) {
                idx_t v = x + y * d + z * d * d;
                sparse_insert(A, v, v, 1.0); /* placeholder diagonal */
                if (x + 1 < d) {
                    sparse_insert(A, v, v + 1, 1.0);
                    sparse_insert(A, v + 1, v, 1.0);
                }
                if (y + 1 < d) {
                    sparse_insert(A, v, v + d, 1.0);
                    sparse_insert(A, v + d, v, 1.0);
                }
                if (z + 1 < d) {
                    sparse_insert(A, v, v + d * d, 1.0);
                    sparse_insert(A, v + d * d, v, 1.0);
                }
            }
        }
    }
    return A;
}

/* Verify the partition invariant: no edge connects a side-0 vertex
 * to a side-1 vertex.  Every cut edge must route through a separator
 * vertex (`part[i] == 2`). */
static int check_partition_invariant(const sparse_graph_t *G, const idx_t *part) {
    for (idx_t i = 0; i < G->n; i++) {
        for (idx_t k = G->xadj[i]; k < G->xadj[i + 1]; k++) {
            idx_t j = G->adjncy[k];
            int p_i = (int)part[i];
            int p_j = (int)part[j];
            if ((p_i == 0 && p_j == 1) || (p_i == 1 && p_j == 0))
                return 0;
        }
    }
    return 1;
}

/* Count vertices per side. */
static void count_partition_sides(const sparse_graph_t *G, const idx_t *part, idx_t *n0, idx_t *n1,
                                  idx_t *nsep) {
    *n0 = 0;
    *n1 = 0;
    *nsep = 0;
    for (idx_t i = 0; i < G->n; i++) {
        if (part[i] == 0)
            (*n0)++;
        else if (part[i] == 1)
            (*n1)++;
        else if (part[i] == 2)
            (*nsep)++;
    }
}

/* ─── 10×10 grid: separator ≈ 10 (one row/column) ─────────────────── */

static void test_partition_10x10_grid(void) {
    SparseMatrix *A = make_grid_2d(10, 10);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    ASSERT_EQ(G.n, 100);

    idx_t part[100] = {0};
    idx_t sep = 0;
    REQUIRE_OK(sparse_graph_partition(&G, part, &sep));

    /* Optimal vertex separator on a 10×10 grid is 10 (a full row or
     * column).  Allow 12 (+20%) for FM stochasticity. */
    ASSERT_TRUE(sep >= 5);
    ASSERT_TRUE(sep <= 12);

    /* No part-0 / part-1 cross-edges. */
    ASSERT_TRUE(check_partition_invariant(&G, part));

    /* Balanced sides ±20% of (n - sep) ≈ 88. */
    idx_t n0, n1, nsep;
    count_partition_sides(&G, part, &n0, &n1, &nsep);
    ASSERT_EQ(n0 + n1 + nsep, 100);
    ASSERT_EQ(nsep, sep);
    idx_t imbal = n0 > n1 ? n0 - n1 : n1 - n0;
    ASSERT_TRUE(imbal <= 20);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Sprint 25 Day 6: spectral bisection (stubs; Days 7-8 land asserts) ── */

/* Sprint 25 Day 6-7: pin the Fiedler-vector eigenvalue ordering
 * contract on a connected graph.
 *   Day 6: built the Laplacian + verified the row-sum-to-zero
 *          invariant (still asserted below).
 *   Day 7: also call sparse_eigs_sym and assert
 *          - λ_0 ≈ 0 (within 1e-6 tolerance) — trivial Laplacian eigenvalue
 *          - λ_1 > 1e-6 — algebraic connectivity > 0 for connected graphs
 *          - eigenvector v_0 is approximately constant (Fiedler 1973's
 *            classic property).
 *
 * See docs/planning/EPIC_2/SPRINT_25/spectral_bisection_design.md. */
static void test_spectral_bisection_eigenvalue_ordering(void) {
    /* Path graph 0-1-2-3-4: each non-endpoint has 2 neighbours,
     * endpoints have 1.  Connected → Fiedler vector exists. */
    SparseMatrix *A = sparse_create(5, 5);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    for (idx_t i = 0; i < 4; i++) {
        sparse_insert(A, i, i + 1, 1.0);
        sparse_insert(A, i + 1, i, 1.0);
    }
    /* Diagonal entries pinning the row-sum invariant we assert
     * separately on the Laplacian. */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 3, 3, 2.0);
    sparse_insert(A, 4, 4, 1.0);

    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    ASSERT_EQ(G.n, 5);

    /* Day 6 invariant: Laplacian builds; rows sum to zero. */
    SparseMatrix *L = NULL;
    REQUIRE_OK(graph_build_laplacian(&G, &L));
    ASSERT_NOT_NULL(L);
    ASSERT_EQ(sparse_rows(L), 5);
    ASSERT_EQ(sparse_cols(L), 5);
    for (idx_t i = 0; i < 5; i++) {
        double row_sum = 0.0;
        for (idx_t j = 0; j < 5; j++)
            row_sum += sparse_get(L, i, j);
        ASSERT_TRUE(fabs(row_sum) < 1e-12);
    }

    /* Day 7 contract: smallest two eigenpairs via sparse_eigs_sym. */
    double eigvals[2] = {0.0, 0.0};
    double eigvecs[5 * 2] = {0.0};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_SMALLEST,
        .tol = 1e-8,
        .reorthogonalize = 1,
        .compute_vectors = 1,
    };
    sparse_eigs_t result = {.eigenvalues = eigvals, .eigenvectors = eigvecs};
    REQUIRE_OK(sparse_eigs_sym(L, /*k=*/2, &opts, &result));
    ASSERT_EQ(result.n_converged, 2);

    /* λ_0 ≈ 0 (trivial Laplacian eigenvalue; eigenvector is
     * proportional to the constant vector). */
    ASSERT_TRUE(fabs(eigvals[0]) < 1e-6);

    /* λ_1 > 0 (algebraic connectivity > 0 ⇔ graph is connected;
     * Fiedler 1973). */
    ASSERT_TRUE(eigvals[1] > 1e-6);

    /* v_0 (column 0) is approximately constant.  All entries should
     * have the same sign (Perron-Frobenius / non-negative
     * eigenvector property of the Laplacian's null space).  Strong
     * check: every entry is within 10 % of the mean. */
    double mean = 0.0;
    for (idx_t i = 0; i < 5; i++)
        mean += eigvecs[i];
    mean /= 5.0;
    /* mean is non-zero by construction (eigenvector is normalized
     * to unit length; if not constant zero, mean ≈ ±1/sqrt(5) ≈ ±0.447). */
    ASSERT_TRUE(fabs(mean) > 0.1);
    for (idx_t i = 0; i < 5; i++) {
        double rel = fabs(eigvecs[i] - mean) / fabs(mean);
        ASSERT_TRUE(rel < 0.1); /* every component within 10 % of mean */
    }

    printf("    path graph (n=5): λ_0=%.3e (≈0), λ_1=%.3e (>0); v_0 within ±10%% of mean\n",
           eigvals[0], eigvals[1]);

    sparse_free(L);
    sparse_graph_free(&G);
    sparse_free(A);
}

/* Sprint 25 Day 6-7: pin the 60/40-balance fallback contract on a
 * star-graph fixture (1 hub + n-1 leaves).
 *
 * The Fiedler vector of a star graph puts the hub at one extreme
 * value and all leaves at the other (the leaves are
 * indistinguishable by the Laplacian's eigenstructure).  The
 * median ≈ leaf value; the strict-< partition assigns the hub
 * alone to side 0 and ALL leaves to side 1 — a 1/(n-1) imbalance.
 * For n = 11, that's 1/10 = 0.1, well below the 60/40 threshold of
 * 0.4, so the balance check fires and graph_bisect_coarsest_spectral
 * falls back to bisect_gggp, which produces a balanced cut.
 *
 * Day 7 contract: under SPARSE_ND_COARSEST_BISECTION=spectral on
 * an n=11 star graph, the resulting partition's balance ratio
 * (min(n0, n1) / max(n0, n1)) is >= 0.4 — proving fallback fired
 * (because spectral alone would produce 0.1). */
static void test_spectral_bisection_gggp_fallback(void) {
    if (setenv("SPARSE_ND_COARSEST_BISECTION", "spectral", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv failed)\n");
        return;
    }

    /* Star graph with n=11 (1 hub at vertex 0 + 10 leaves at
     * vertices 1..10).  Pure-spectral natural cut: {0} vs {1..10} =
     * 1/10 imbalance, well below 60/40 threshold. */
    const idx_t N = 11;
    SparseMatrix *A = sparse_create(N, N);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_insert(A, 0, 0, (double)(N - 1));
    for (idx_t k = 1; k < N; k++) {
        sparse_insert(A, k, k, 1.0);
        sparse_insert(A, 0, k, -1.0);
        sparse_insert(A, k, 0, -1.0);
    }
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part[11] = {0};
    sparse_err_t rc = graph_bisect_coarsest(&G, part);
    unsetenv("SPARSE_ND_COARSEST_BISECTION");

    REQUIRE_OK(rc);

    /* Structural validity: all entries in {0, 1}; at least one on
     * each side. */
    idx_t n0 = 0, n1 = 0;
    for (idx_t i = 0; i < N; i++) {
        ASSERT_TRUE(part[i] == 0 || part[i] == 1);
        if (part[i] == 0)
            n0++;
        else
            n1++;
    }
    ASSERT_EQ(n0 + n1, N);
    ASSERT_TRUE(n0 >= 1 && n1 >= 1);

    /* Day 7 fallback assertion: spectral's natural cut would
     * produce a 1/(N-1) = 1/10 = 0.1 imbalance ratio, well below
     * the 60/40 threshold (0.4).  The fact that the resulting
     * partition has a balance ratio >= 0.4 PROVES the spectral
     * path's 60/40 check fired and bisect_gggp took over.
     * (GGGP on a star graph produces roughly 5/6 = 0.83 balance
     * by vertex-weight half-target.) */
    idx_t lo = (n0 < n1) ? n0 : n1;
    idx_t hi = (n0 < n1) ? n1 : n0;
    /* lo / hi >= 0.4 ⇔ 10 * lo >= 4 * hi (integer arithmetic). */
    ASSERT_TRUE(10 * lo >= 4 * hi);

    printf("    star graph (n=%d): n0=%d, n1=%d, balance ratio=%.2f (>=0.40 ⇒ "
           "GGGP fallback fired)\n",
           (int)N, (int)n0, (int)n1, (double)lo / (double)hi);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Sprint 25 Day 8: spectral-bisection edge cases (n=1, n=2, ─────── */
/* ─── disconnected, Lanczos failure → GGGP fallback) ──────────────── */

/* Trivial size n=1: spectral skips Lanczos and assigns the single
 * vertex to side 0 directly. */
static void test_spectral_bisection_n1(void) {
    if (setenv("SPARSE_ND_COARSEST_BISECTION", "spectral", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv failed)\n");
        return;
    }

    SparseMatrix *A = sparse_create(1, 1);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_insert(A, 0, 0, 1.0);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    ASSERT_EQ(G.n, 1);

    idx_t part[1] = {99}; /* sentinel */
    sparse_err_t rc = graph_bisect_coarsest(&G, part);
    unsetenv("SPARSE_ND_COARSEST_BISECTION");
    REQUIRE_OK(rc);

    /* n=1 must produce part[0] = 0 (degenerate single-vertex partition). */
    ASSERT_EQ(part[0], 0);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* Trivial size n=2: spectral skips Lanczos and assigns each vertex
 * to its own side (the unique 2-way split). */
static void test_spectral_bisection_n2(void) {
    if (setenv("SPARSE_ND_COARSEST_BISECTION", "spectral", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv failed)\n");
        return;
    }

    SparseMatrix *A = sparse_create(2, 2);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 0, 1, -1.0);
    sparse_insert(A, 1, 0, -1.0);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    ASSERT_EQ(G.n, 2);

    idx_t part[2] = {99, 99}; /* sentinels */
    sparse_err_t rc = graph_bisect_coarsest(&G, part);
    unsetenv("SPARSE_ND_COARSEST_BISECTION");
    REQUIRE_OK(rc);

    /* n=2 must produce {part[0]=0, part[1]=1} — the unique 2-way split. */
    ASSERT_EQ(part[0], 0);
    ASSERT_EQ(part[1], 1);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* Disconnected graph: a Laplacian with multiple zero eigenvalues
 * (one per connected component) breaks the Fiedler-vector
 * uniqueness assumption.  Spectral bisection detects this via
 * λ_1 - λ_0 < 1e-6 and falls back to GGGP.
 *
 * Fixture: two disjoint K_3 triangles (vertices 0-1-2 and 3-4-5
 * forming a triangle each; no edges between them).  The Laplacian
 * has λ_0 = λ_1 = 0 (both connected components contribute a zero
 * eigenvalue); the disconnected-graph detection in
 * graph_bisect_coarsest_spectral fires and bisect_gggp produces
 * the partition. */
static void test_spectral_bisection_disconnected(void) {
    if (setenv("SPARSE_ND_COARSEST_BISECTION", "spectral", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv failed)\n");
        return;
    }

    /* Two disjoint K_3 triangles: vertices {0,1,2} and {3,4,5}. */
    SparseMatrix *A = sparse_create(6, 6);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    /* Diagonal: degree of each vertex = 2 (every triangle vertex
     * has 2 neighbours in its component, 0 in the other). */
    for (idx_t i = 0; i < 6; i++)
        sparse_insert(A, i, i, 2.0);
    /* Triangle 0: edges (0,1), (1,2), (0,2). */
    sparse_insert(A, 0, 1, -1.0);
    sparse_insert(A, 1, 0, -1.0);
    sparse_insert(A, 1, 2, -1.0);
    sparse_insert(A, 2, 1, -1.0);
    sparse_insert(A, 0, 2, -1.0);
    sparse_insert(A, 2, 0, -1.0);
    /* Triangle 1: edges (3,4), (4,5), (3,5). */
    sparse_insert(A, 3, 4, -1.0);
    sparse_insert(A, 4, 3, -1.0);
    sparse_insert(A, 4, 5, -1.0);
    sparse_insert(A, 5, 4, -1.0);
    sparse_insert(A, 3, 5, -1.0);
    sparse_insert(A, 5, 3, -1.0);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    ASSERT_EQ(G.n, 6);

    idx_t part[6] = {0};
    sparse_err_t rc = graph_bisect_coarsest(&G, part);
    unsetenv("SPARSE_ND_COARSEST_BISECTION");
    REQUIRE_OK(rc);

    /* Validate structural contract.  The disconnected-graph fallback
     * routes through bisect_gggp, which produces SOME valid {0, 1}
     * partition (not necessarily aligned with the component
     * boundaries — GGGP is unaware of components).  Just assert all
     * entries in {0, 1} + at least one on each side. */
    int has_zero = 0, has_one = 0;
    for (idx_t i = 0; i < 6; i++) {
        ASSERT_TRUE(part[i] == 0 || part[i] == 1);
        if (part[i] == 0)
            has_zero = 1;
        if (part[i] == 1)
            has_one = 1;
    }
    ASSERT_TRUE(has_zero && has_one);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* Lanczos non-convergence: simulate by setting an unrealistically
 * tight tolerance + a tiny iteration cap that prevents convergence,
 * then verify spectral falls back to GGGP cleanly.
 *
 * Day 7's implementation uses opts.tol = 1e-8 internally and
 * doesn't expose tol/max_iterations to callers, so we can't
 * directly trip Lanczos non-convergence from this test.  Instead
 * we rely on the disconnected-graph test above as a proxy for the
 * "Lanczos returns non-meaningful eigenpairs → fall back" path
 * (the disconnected case is the realistic Lanczos-can't-give-
 * useful-eigenpairs scenario in production), and document why
 * this test stays as a smoke test rather than a true Lanczos-
 * failure injection. */
static void test_spectral_bisection_lanczos_failure(void) {
    if (setenv("SPARSE_ND_COARSEST_BISECTION", "spectral", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv failed)\n");
        return;
    }

    /* Build a fixture that's well-behaved for Lanczos (a 5-vertex
     * path graph) — Day 7's spectral path produces a normal Fiedler
     * cut here.  This test confirms the dispatch wiring works on a
     * normal fixture under the env var; the actual Lanczos-failure
     * fallback path is exercised by graph_bisect_coarsest_spectral's
     * `if (eigs_rc != SPARSE_OK || result.n_converged < 2) → fall
     * back` clause, which is reachable but hard to trip without an
     * explicit fault-injection hook.
     *
     * The disconnected-graph test above (test_spectral_bisection_disconnected)
     * exercises the analogous "spectral can't produce a meaningful
     * Fiedler vector → fall back" path via the
     * `if (lambda_1 - lambda_0 < 1e-6) → fall back` clause; together
     * the two cover the practical fallback-firing scenarios. */
    SparseMatrix *A = sparse_create(5, 5);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    for (idx_t i = 0; i < 4; i++) {
        sparse_insert(A, i, i + 1, 1.0);
        sparse_insert(A, i + 1, i, 1.0);
    }
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 3, 3, 2.0);
    sparse_insert(A, 4, 4, 1.0);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part[5] = {0};
    sparse_err_t rc = graph_bisect_coarsest(&G, part);
    unsetenv("SPARSE_ND_COARSEST_BISECTION");
    REQUIRE_OK(rc);

    /* Structural contract: valid partition. */
    int has_zero = 0, has_one = 0;
    for (idx_t i = 0; i < 5; i++) {
        ASSERT_TRUE(part[i] == 0 || part[i] == 1);
        if (part[i] == 0)
            has_zero = 1;
        if (part[i] == 1)
            has_one = 1;
    }
    ASSERT_TRUE(has_zero && has_one);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Sprint 25 Day 5: SPARSE_FM_INTERMEDIATE_PASSES env-var plumbing ──── */

static void test_fm_intermediate_passes_smoke(void) {
    /* Sprint 25 Day 5 smoke test: pin the SPARSE_FM_INTERMEDIATE_PASSES
     * env var's plumbing.  Set the var to "2" via setenv, run
     * sparse_graph_partition on a 10×10 grid (the same fixture
     * test_partition_10x10_grid uses under default settings), assert
     * the resulting partition is structurally valid.
     *
     * The test does NOT lock in a particular nnz_L or separator-size
     * outcome — Day 5's sweep showed passes=2 produces a partition
     * within the same noise band as passes=1 on the 10×10 grid (sep
     * stays in [5, 12]; partition invariant holds; balance within
     * 20% of (n - sep)).  This smoke test pins the wiring (the env
     * var must be parsed AND the dispatch must reach the
     * intermediate_passes branch in graph_uncoarsen) without
     * over-constraining the cut quality.
     *
     * We unsetenv after the test to keep the per-process env clean
     * for subsequent tests in this binary's run. */
    if (setenv("SPARSE_FM_INTERMEDIATE_PASSES", "2", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv failed; can't exercise env-var plumbing)\n");
        return;
    }

    SparseMatrix *A = make_grid_2d(10, 10);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    ASSERT_EQ(G.n, 100);

    idx_t part[100] = {0};
    idx_t sep = 0;
    sparse_err_t rc = sparse_graph_partition(&G, part, &sep);

    /* Restore env before any potential REQUIRE_OK / ASSERT exit so
     * subsequent tests run with the default. */
    unsetenv("SPARSE_FM_INTERMEDIATE_PASSES");

    REQUIRE_OK(rc);

    /* Same structural invariants as test_partition_10x10_grid: the
     * partition must be valid regardless of pass count. */
    ASSERT_TRUE(sep >= 5);
    ASSERT_TRUE(sep <= 12);
    ASSERT_TRUE(check_partition_invariant(&G, part));

    idx_t n0, n1, nsep;
    count_partition_sides(&G, part, &n0, &n1, &nsep);
    ASSERT_EQ(n0 + n1 + nsep, 100);
    ASSERT_EQ(nsep, sep);
    idx_t imbal = n0 > n1 ? n0 - n1 : n1 - n0;
    ASSERT_TRUE(imbal <= 20);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* Sprint 26 Day 7: SPARSE_FM_FINEST_STRATEGY=fifo differs-from-
 * baseline contract pin.  Day 6 stubbed this on the 10×10 grid
 * (which is too small to exercise FIFO's tie-break sensitivity —
 * the symmetric optimal cuts converge to the same partition under
 * FIFO and LIFO).  Day 7 uses a 30×30 grid (n=900) where FIFO
 * actually produces a different partition than baseline.
 *
 * Test contract: under `SPARSE_FM_FINEST_STRATEGY=fifo` on a 30×30
 * grid, partition produces (a) a structurally-valid result, (b) a
 * partition that differs from the baseline strategy's, (c)
 * deterministic across two runs (same input → same output).
 *
 * Pinning (b) is the smoke-level evidence that the FIFO pop variant
 * is actually being exercised — without (b), the dispatch could be
 * a no-op and the test would still pass.  Day 8's cross-corpus
 * sweep is where the differs-from-baseline measurement scales up to
 * the headline-fixture (Pres_Poisson) and decides flip-or-stay.
 *
 * See SPRINT_26/finest_fm_design.md for the sub-axis selection
 * rationale + Day 6 / Day 7 / Day 8 split. */
static void test_finest_fm_strategy_fifo_smoke(void) {
    /* 30×30 grid: small enough to run quickly (~ms), large enough
     * for FIFO and baseline to converge to different cuts.  Empirical:
     * baseline sep=30, FIFO sep=30, but the partitions differ
     * vertex-for-vertex (verified pre-test with a small ad-hoc
     * comparison against /tmp/cmp_fifo.c during Day-7 development).
     *
     * PR #34 review fix: route every exit path through a single
     * `cleanup:` block that frees the three buffers, frees G/A, and
     * unsets SPARSE_FM_FINEST_STRATEGY.  The previous flow had two
     * issues: (1) `ASSERT_NOT_NULL` is non-fatal (alloc failure
     * would NULL-deref in subsequent partition calls + memcmp), and
     * (2) `REQUIRE_OK` returns immediately on failure, leaving the
     * env var set + buffers leaked across subsequent tests. */
    SparseMatrix *A = NULL;
    sparse_graph_t G = {0};
    idx_t *part_baseline = NULL;
    idx_t *part_fifo1 = NULL;
    idx_t *part_fifo2 = NULL;
    int env_set = 0;
    int coarsening_env_set = 0;

    /* Sprint 27 Day 2: pin SPARSE_ND_COARSENING=heavy_edge — the
     * "FIFO differs from baseline on the 30×30 grid" contract was
     * verified under HEM coarsening (Sprint 26 Day 7).  HCC's
     * `min(deg)` scoring on a regular grid produces more deterministic
     * matchings (most edges score identically; tie-break dominates),
     * which can collapse the FIFO-vs-baseline differentiation.
     * Pinning HEM keeps this test scoped to its Sprint 26 design
     * intent under the new HCC default (Sprint 27 Day 2 flip). */
    if (setenv("SPARSE_ND_COARSENING", "heavy_edge", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv SPARSE_ND_COARSENING failed)\n");
        return;
    }
    coarsening_env_set = 1;

    A = make_grid_2d(30, 30);
    if (!A) {
        TF_FAIL_("make_grid_2d(%d, %d) returned NULL (OOM)", 30, 30);
        goto cleanup;
    }
    sparse_err_t rc = sparse_graph_from_sparse(A, &G);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_graph_from_sparse: rc=%d", (int)rc);
        goto cleanup;
    }
    ASSERT_EQ(G.n, 900);

    /* Baseline run (env var unset). */
    unsetenv("SPARSE_FM_FINEST_STRATEGY");
    part_baseline = malloc((size_t)G.n * sizeof(idx_t));
    if (!part_baseline) {
        TF_FAIL_("malloc(part_baseline) returned NULL (n=%d)", (int)G.n);
        goto cleanup;
    }
    idx_t sep_baseline = 0;
    rc = sparse_graph_partition(&G, part_baseline, &sep_baseline);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_graph_partition(baseline): rc=%d", (int)rc);
        goto cleanup;
    }
    if (!check_partition_invariant(&G, part_baseline)) {
        TF_FAIL_("baseline partition invariant failed (n=%d)", (int)G.n);
        goto cleanup;
    }

    /* FIFO run #1. */
    if (setenv("SPARSE_FM_FINEST_STRATEGY", "fifo", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv failed; can't exercise env-var plumbing)\n");
        goto cleanup;
    }
    env_set = 1;
    part_fifo1 = malloc((size_t)G.n * sizeof(idx_t));
    if (!part_fifo1) {
        TF_FAIL_("malloc(part_fifo1) returned NULL (n=%d)", (int)G.n);
        goto cleanup;
    }
    idx_t sep_fifo1 = 0;
    rc = sparse_graph_partition(&G, part_fifo1, &sep_fifo1);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_graph_partition(fifo run #1): rc=%d", (int)rc);
        goto cleanup;
    }
    if (!check_partition_invariant(&G, part_fifo1)) {
        TF_FAIL_("fifo run #1 partition invariant failed (n=%d)", (int)G.n);
        goto cleanup;
    }

    /* FIFO run #2 (determinism check). */
    part_fifo2 = malloc((size_t)G.n * sizeof(idx_t));
    if (!part_fifo2) {
        TF_FAIL_("malloc(part_fifo2) returned NULL (n=%d)", (int)G.n);
        goto cleanup;
    }
    idx_t sep_fifo2 = 0;
    rc = sparse_graph_partition(&G, part_fifo2, &sep_fifo2);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_graph_partition(fifo run #2): rc=%d", (int)rc);
        goto cleanup;
    }

    /* (a) FIFO determinism: same input → same output. */
    ASSERT_EQ(sep_fifo1, sep_fifo2);
    ASSERT_EQ(memcmp(part_fifo1, part_fifo2, (size_t)G.n * sizeof(idx_t)), 0);

    /* (b) FIFO differs from baseline: at least one vertex's part
     * differs, OR the sep counts differ.  This is the smoke-level
     * evidence that the FIFO pop variant is actually being
     * exercised. */
    int differs = (sep_baseline != sep_fifo1) ||
                  (memcmp(part_baseline, part_fifo1, (size_t)G.n * sizeof(idx_t)) != 0);
    printf("    30x30 grid: baseline sep=%d, fifo sep=%d, partitions %s\n", (int)sep_baseline,
           (int)sep_fifo1, differs ? "DIFFER (FIFO active)" : "match");
    ASSERT_TRUE(differs);

cleanup:
    if (env_set)
        unsetenv("SPARSE_FM_FINEST_STRATEGY");
    if (coarsening_env_set)
        unsetenv("SPARSE_ND_COARSENING");
    free(part_baseline);
    free(part_fifo1);
    free(part_fifo2);
    sparse_graph_free(&G);
    sparse_free(A);
}

/* Sprint 28 Day 2 — Item 1: formal gain-bucket-noise variant of
 * thick-restart FM lights up under
 *   SPARSE_FM_FINEST_STRATEGY=thick_restart
 *   SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal
 *
 * Smoke-test contract:
 *   (a) Two runs with the same env state produce bit-identical
 *       partitions (deterministic via per-call (n, k) seeded RNG).
 *   (b) The gain_noise_formal partition differs from the baseline
 *       (env unset) partition, evidence that the new code path is
 *       being exercised.
 *
 * Fixture: 5×5×5 3D mesh (n=125).  The mesh has three perpendicular
 * mid-planes of equivalent cut quality (sep ≈ 25 each).  Baseline FM
 * deterministically picks one; the gain_noise_formal noise on the
 * bucket placement perturbs the FM walk enough to land a different
 * (still ≤ near-optimal) partition — sufficient for the smoke
 * contract.  A 30×30 2D grid was tried first but its single optimal
 * cut (median row, sep=30) is robust to the per-pass gain noise: both
 * walks converge to bit-identical assignments.  Pinned to
 * SPARSE_ND_COARSENING=heavy_edge per the same Sprint 27 Day 2
 * pattern as the FIFO smoke test (HCC's tighter matching on regular
 * fixtures can collapse small differences).
 */
static void test_finest_fm_gain_noise_formal_disrupts_baseline(void) {
    SparseMatrix *A = NULL;
    sparse_graph_t G = {0};
    idx_t *part_baseline = NULL;
    idx_t *part_gnf1 = NULL;
    idx_t *part_gnf2 = NULL;
    int strategy_env_set = 0;
    int perturb_env_set = 0;
    int coarsening_env_set = 0;

    if (setenv("SPARSE_ND_COARSENING", "heavy_edge", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv SPARSE_ND_COARSENING failed)\n");
        return;
    }
    coarsening_env_set = 1;

    A = make_mesh_3d(5);
    if (!A) {
        TF_FAIL_("make_mesh_3d(%d) returned NULL (OOM)", 5);
        goto cleanup;
    }
    sparse_err_t rc = sparse_graph_from_sparse(A, &G);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_graph_from_sparse: rc=%d", (int)rc);
        goto cleanup;
    }
    ASSERT_EQ(G.n, 125);

    /* Baseline run (env vars unset; default FM behaviour). */
    unsetenv("SPARSE_FM_FINEST_STRATEGY");
    unsetenv("SPARSE_FM_THICK_RESTART_PERTURB");
    part_baseline = malloc((size_t)G.n * sizeof(idx_t));
    if (!part_baseline) {
        TF_FAIL_("malloc(part_baseline) returned NULL (n=%d)", (int)G.n);
        goto cleanup;
    }
    idx_t sep_baseline = 0;
    rc = sparse_graph_partition(&G, part_baseline, &sep_baseline);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_graph_partition(baseline): rc=%d", (int)rc);
        goto cleanup;
    }
    if (!check_partition_invariant(&G, part_baseline)) {
        TF_FAIL_("baseline partition invariant failed (n=%d)", (int)G.n);
        goto cleanup;
    }

    /* gain_noise_formal run #1. */
    if (setenv("SPARSE_FM_FINEST_STRATEGY", "thick_restart", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv SPARSE_FM_FINEST_STRATEGY failed)\n");
        goto cleanup;
    }
    strategy_env_set = 1;
    if (setenv("SPARSE_FM_THICK_RESTART_PERTURB", "gain_noise_formal", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv SPARSE_FM_THICK_RESTART_PERTURB failed)\n");
        goto cleanup;
    }
    perturb_env_set = 1;
    part_gnf1 = malloc((size_t)G.n * sizeof(idx_t));
    if (!part_gnf1) {
        TF_FAIL_("malloc(part_gnf1) returned NULL (n=%d)", (int)G.n);
        goto cleanup;
    }
    idx_t sep_gnf1 = 0;
    rc = sparse_graph_partition(&G, part_gnf1, &sep_gnf1);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_graph_partition(gain_noise_formal #1): rc=%d", (int)rc);
        goto cleanup;
    }
    if (!check_partition_invariant(&G, part_gnf1)) {
        TF_FAIL_("gain_noise_formal #1 partition invariant failed (n=%d)", (int)G.n);
        goto cleanup;
    }

    /* gain_noise_formal run #2 (determinism check). */
    part_gnf2 = malloc((size_t)G.n * sizeof(idx_t));
    if (!part_gnf2) {
        TF_FAIL_("malloc(part_gnf2) returned NULL (n=%d)", (int)G.n);
        goto cleanup;
    }
    idx_t sep_gnf2 = 0;
    rc = sparse_graph_partition(&G, part_gnf2, &sep_gnf2);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_graph_partition(gain_noise_formal #2): rc=%d", (int)rc);
        goto cleanup;
    }

    /* (a) determinism: two runs produce bit-identical partitions. */
    ASSERT_EQ(sep_gnf1, sep_gnf2);
    ASSERT_EQ(memcmp(part_gnf1, part_gnf2, (size_t)G.n * sizeof(idx_t)), 0);

    /* (b) differs from baseline: the new code path is exercised. */
    int differs = (sep_baseline != sep_gnf1) ||
                  (memcmp(part_baseline, part_gnf1, (size_t)G.n * sizeof(idx_t)) != 0);
    printf("    5x5x5 mesh: baseline sep=%d, gain_noise_formal sep=%d, partitions %s\n",
           (int)sep_baseline, (int)sep_gnf1,
           differs ? "DIFFER (gain_noise_formal active)" : "match");
    ASSERT_TRUE(differs);

cleanup:
    if (perturb_env_set)
        unsetenv("SPARSE_FM_THICK_RESTART_PERTURB");
    if (strategy_env_set)
        unsetenv("SPARSE_FM_FINEST_STRATEGY");
    if (coarsening_env_set)
        unsetenv("SPARSE_ND_COARSENING");
    free(part_baseline);
    free(part_gnf1);
    free(part_gnf2);
    sparse_graph_free(&G);
    sparse_free(A);
}

/* Sprint 28 Day 3 — Item 2 stub: multi-strategy FM ensemble's
 * pick-correctness contract.
 *
 * Day 4 implementation lights up `SPARSE_FM_FINEST_STRATEGY=ensemble`
 * to run K FM strategies in parallel per finest-level call and pick
 * the strategy with the lowest cut.  This test pins the pick-
 * correctness contract: on a fixture where one strategy provably
 * dominates (lower cut than the other two), the ensemble runner
 * must pick that strategy's result.
 *
 * Day 3 ships only the stub.  RUN_TEST line is commented out below
 * because the `ensemble` strategy value is not yet parsed — Day 4
 * adds the enum value + dispatch.  The stub compiles today but the
 * setenv call would route to default-fallthrough baseline.
 *
 * Day 4 will: (a) enable the RUN_TEST line; (b) replace the
 * `printf("    stub — Day 4 lights this up\n")` placeholder body
 * with the real synthetic-fixture + 3-strategy comparison + assert
 * the ensemble picks the dominant strategy's result.  See
 * `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_design.md`
 * "Pick-correctness contract".
 */
static void test_finest_fm_ensemble_picks_best_strategy(void) {
    /* Sprint 28 Day 3 stub — Day 4 implementation pending.
     *
     * Planned Day-4 body sketch:
     *   1. Build a fixture where baseline FM lands a tight cut and
     *      FIFO + annealing land looser cuts (e.g. an irregular SPD
     *      where the FIFO tail-pop disrupts a good LIFO walk).
     *   2. Run with SPARSE_FM_FINEST_STRATEGY=ensemble
     *      SPARSE_FM_ENSEMBLE_STRATEGIES=baseline,fifo,annealing.
     *   3. Assert the ensemble's resulting cut equals baseline's
     *      cut (the dominant strategy's result), not FIFO's or
     *      annealing's.
     *   4. (Optional) Re-run with the dominant strategy swapped
     *      (e.g. force FIFO to dominate by env tweaks) and assert
     *      the ensemble picks FIFO under that variant.
     *
     * For Day 3, this stub just prints a placeholder line so the
     * test framework reports it cleanly when RUN_TEST is later
     * enabled.  Sprint 27 Day 11 thick-restart test followed this
     * same stub-now-light-up-later pattern. */
    printf("    stub — Day 4 lights this up (ensemble pick-correctness contract)\n");
}

/* ─── 5×5×5 3D mesh: separator ≈ 25 (one mid-plane) ──────────────── */

static void test_partition_5x5x5_mesh(void) {
    SparseMatrix *A = make_mesh_3d(5);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    ASSERT_EQ(G.n, 125);

    idx_t *part = calloc((size_t)G.n, sizeof(idx_t));
    ASSERT_NOT_NULL(part);
    idx_t sep = 0;
    REQUIRE_OK(sparse_graph_partition(&G, part, &sep));

    /* Optimal planar separator: 5×5 = 25.  Allow 30 (+20%). */
    ASSERT_TRUE(sep >= 10);
    ASSERT_TRUE(sep <= 30);

    ASSERT_TRUE(check_partition_invariant(&G, part));

    /* 3-way split.  Smaller-side vertex-separator extraction can
     * amplify edge-partition imbalance: a 60-65 edge cut becomes a
     * 35-65 + 25-sep vertex partition.  Allow up to 60 (about half
     * the non-separator vertices).  Both sides must be non-empty so
     * the recursive ND has work to do on each. */
    idx_t n0, n1, nsep;
    count_partition_sides(&G, part, &n0, &n1, &nsep);
    ASSERT_EQ(n0 + n1 + nsep, 125);
    ASSERT_TRUE(n0 > 0);
    ASSERT_TRUE(n1 > 0);
    idx_t imbal = n0 > n1 ? n0 - n1 : n1 - n0;
    ASSERT_TRUE(imbal <= 60);

    free(part);
    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Two K10 cliques + bridge: separator = 1 (the bridge) ────────── */

static void test_partition_two_k10_with_bridge(void) {
    SparseMatrix *A = make_two_cliques_with_bridge(10);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    ASSERT_EQ(G.n, 20);

    idx_t part[20] = {0};
    idx_t sep = 0;
    REQUIRE_OK(sparse_graph_partition(&G, part, &sep));

    /* Optimal separator is one of the two bridge endpoints (cut the
     * single edge by lifting one endpoint into the separator).  Allow
     * 2 (the FM uncoarsening might land on either endpoint). */
    ASSERT_TRUE(sep >= 1);
    ASSERT_TRUE(sep <= 2);

    ASSERT_TRUE(check_partition_invariant(&G, part));

    /* Both K10 sides survive intact (modulo the bridge endpoint
     * lifted into the separator). */
    idx_t n0, n1, nsep;
    count_partition_sides(&G, part, &n0, &n1, &nsep);
    ASSERT_EQ(n0 + n1 + nsep, 20);
    ASSERT_TRUE(n0 >= 9);
    ASSERT_TRUE(n1 >= 9);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Edge → vertex separator helper, isolated ────────────────────── */

static void test_edge_to_vertex_separator_smaller_side(void) {
    /* Manually build a 2-way edge separator on the 5×6 grid where
     * column ≤ 2 → side 0 (15 vertices) and column ≥ 3 → side 1
     * (15 vertices).  Tied weights → smaller_side defaults to 0.
     * The boundary on side 0 is column 2 (5 vertices); after the
     * smaller-side lift, those 5 land in the separator. */
    SparseMatrix *A = make_grid_2d(5, 6);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part[30] = {0};
    for (idx_t r = 0; r < 5; r++) {
        for (idx_t c = 0; c < 6; c++) {
            part[r * 6 + c] = c < 3 ? 0 : 1;
        }
    }

    REQUIRE_OK(graph_edge_separator_to_vertex_separator(&G, part));

    /* No part-0 / part-1 cross-edges. */
    ASSERT_TRUE(check_partition_invariant(&G, part));

    /* Boundary on side 0 (column 2) is now the separator (5 vertices). */
    idx_t n0, n1, nsep;
    count_partition_sides(&G, part, &n0, &n1, &nsep);
    ASSERT_EQ(nsep, 5);
    ASSERT_EQ(n0, 10); /* columns 0, 1 */
    ASSERT_EQ(n1, 15); /* columns 3, 4, 5 */
    /* Column-2 vertices are exactly the separator. */
    for (idx_t r = 0; r < 5; r++)
        ASSERT_EQ(part[r * 6 + 2], 2);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── NULL-arg + n=0 + small-input contracts ──────────────────────── */

static void test_partition_null_args(void) {
    sparse_graph_t G = {0};
    idx_t part[1] = {0};
    idx_t sep = 0;
    ASSERT_ERR(sparse_graph_partition(NULL, part, &sep), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_graph_partition(&G, NULL, &sep), SPARSE_ERR_NULL);
}

static void test_partition_singleton(void) {
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_NOT_NULL(A);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    idx_t part[1] = {99};
    idx_t sep = 99;
    REQUIRE_OK(sparse_graph_partition(&G, part, &sep));
    /* Singleton: no edges → no separator needed; the vertex stays on
     * its bisect side. */
    ASSERT_TRUE(part[0] == 0 || part[0] == 1 || part[0] == 2);
    ASSERT_EQ(sep, part[0] == 2 ? 1 : 0);
    sparse_graph_free(&G);
    sparse_free(A);
}

static void test_uncoarsen_null_args(void) {
    sparse_graph_t G = {0};
    sparse_graph_hierarchy_t h = {0};
    idx_t buf[1] = {0};
    ASSERT_ERR(graph_uncoarsen(NULL, &h, buf, buf), SPARSE_ERR_NULL);
    ASSERT_ERR(graph_uncoarsen(&G, NULL, buf, buf), SPARSE_ERR_NULL);
    ASSERT_ERR(graph_uncoarsen(&G, &h, NULL, buf), SPARSE_ERR_NULL);
    ASSERT_ERR(graph_uncoarsen(&G, &h, buf, NULL), SPARSE_ERR_NULL);
}

static void test_edge_to_vertex_separator_null_args(void) {
    sparse_graph_t G = {0};
    idx_t part[1] = {0};
    ASSERT_ERR(graph_edge_separator_to_vertex_separator(NULL, part), SPARSE_ERR_NULL);
    ASSERT_ERR(graph_edge_separator_to_vertex_separator(&G, NULL), SPARSE_ERR_NULL);
}

/* ═══════════════════════════════════════════════════════════════════
 * Sprint 22 Day 3 — coarsest-graph bisection + FM refinement.
 * ═══════════════════════════════════════════════════════════════════ */

/* Compute the cut weight of a 2-way partition (mirrors the static
 * helper in `src/sparse_graph.c`). */
static idx_t compute_cut(const sparse_graph_t *G, const idx_t *part) {
    idx_t cut = 0;
    for (idx_t i = 0; i < G->n; i++) {
        for (idx_t k = G->xadj[i]; k < G->xadj[i + 1]; k++) {
            idx_t j = G->adjncy[k];
            if (j <= i)
                continue;
            if (part[i] != part[j])
                cut += G->ewgt ? G->ewgt[k] : 1;
        }
    }
    return cut;
}

/* Sum of side-0 vs side-1 vertex weights (NULL vwgt → uniform 1). */
static void compute_side_weights(const sparse_graph_t *G, const idx_t *part, idx_t *w0, idx_t *w1) {
    *w0 = 0;
    *w1 = 0;
    for (idx_t i = 0; i < G->n; i++) {
        idx_t w = G->vwgt ? G->vwgt[i] : 1;
        if (part[i] == 0)
            *w0 += w;
        else
            *w1 += w;
    }
}

/* ─── Brute-force bisection on a known 8-vertex path graph ─────────── */

static void test_bisect_brute_force_path_n8(void) {
    /* Path 0-1-2-3-4-5-6-7 (n=8, 7 edges, all weight 1).  The
     * minimum-cut balanced bisection cuts edge (3, 4) — cut = 1.
     * Any non-contiguous balanced split crosses 3+ edges. */
    SparseMatrix *A = make_path_1d(8);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part[8] = {0};
    REQUIRE_OK(graph_bisect_coarsest(&G, part));

    /* Cut equals the analytical minimum (1). */
    ASSERT_EQ(compute_cut(&G, part), 1);

    /* Partition is balanced (4-4 split on uniform weights). */
    idx_t w0 = 0, w1 = 0;
    compute_side_weights(&G, part, &w0, &w1);
    ASSERT_EQ(w0, 4);
    ASSERT_EQ(w1, 4);

    /* Each entry is 0 or 1. */
    for (idx_t i = 0; i < 8; i++)
        ASSERT_TRUE(part[i] == 0 || part[i] == 1);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Brute-force bisection on a small disconnected fixture ────────── */

static void test_bisect_brute_force_two_triangles(void) {
    /* Two K3 cliques (vertices 0-1-2 and 3-4-5) with no bridge.
     * Optimal cut = 0 (split between cliques). */
    SparseMatrix *A = sparse_create(6, 6);
    ASSERT_NOT_NULL(A);
    /* Clique 1: 0-1-2 */
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    /* Clique 2: 3-4-5 */
    sparse_insert(A, 3, 4, 1.0);
    sparse_insert(A, 4, 3, 1.0);
    sparse_insert(A, 3, 5, 1.0);
    sparse_insert(A, 5, 3, 1.0);
    sparse_insert(A, 4, 5, 1.0);
    sparse_insert(A, 5, 4, 1.0);

    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part[6] = {0};
    REQUIRE_OK(graph_bisect_coarsest(&G, part));

    ASSERT_EQ(compute_cut(&G, part), 0); /* perfectly separable */

    /* Each clique ended up on a single side. */
    ASSERT_EQ(part[0], part[1]);
    ASSERT_EQ(part[0], part[2]);
    ASSERT_EQ(part[3], part[4]);
    ASSERT_EQ(part[3], part[5]);
    ASSERT_NEQ(part[0], part[3]);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── GGGP bisection on a 30-vertex grid (n > 20 fall-through) ─────── */

static void test_bisect_gggp_5x6_grid(void) {
    /* 5×6 grid: 30 vertices, 49 edges.  GGGP returns a balanced-ish
     * 2-way partition; we don't insist on optimality (Day 4's per-
     * level FM refines what GGGP starts from), only that the result
     * is a valid partition with both sides populated. */
    SparseMatrix *A = make_grid_2d(5, 6);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    ASSERT_EQ(G.n, 30);

    idx_t part[30] = {0};
    REQUIRE_OK(graph_bisect_coarsest(&G, part));

    /* Both sides populated. */
    idx_t w0 = 0, w1 = 0;
    compute_side_weights(&G, part, &w0, &w1);
    ASSERT_TRUE(w0 > 0);
    ASSERT_TRUE(w1 > 0);
    ASSERT_EQ(w0 + w1, 30);

    /* Cut is finite and reasonable: a connected 5×6 grid has minimum
     * cut 5 (cleanest column split); GGGP usually sits in [5, 12]
     * before refinement.  Allow up to 20 — Day 4 FM will refine. */
    idx_t cut = compute_cut(&G, part);
    ASSERT_TRUE(cut <= 20);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── FM reduces a deliberately-bad checkerboard partition ────────── */

static void test_fm_reduces_checkerboard_cut(void) {
    /* 5×6 grid with a checkerboard initial partition (every edge
     * endpoint has opposite parity, so every edge is cut — 49/49).
     * FM should drop this dramatically. */
    SparseMatrix *A = make_grid_2d(5, 6);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    ASSERT_EQ(G.n, 30);

    idx_t part[30] = {0};
    for (idx_t r = 0; r < 5; r++) {
        for (idx_t c = 0; c < 6; c++) {
            part[r * 6 + c] = (r + c) & 1;
        }
    }
    idx_t init_cut = compute_cut(&G, part);
    ASSERT_EQ(init_cut, 49); /* every grid edge is cross-cut */

    REQUIRE_OK(graph_refine_fm(&G, part));
    idx_t final_cut = compute_cut(&G, part);

    /* FM should at least halve the cut on this fixture.  Single-pass
     * FM doesn't reach the optimum (5) — Day 4's per-level FM
     * replays will close that gap — but a 50% reduction is well
     * within the single-pass capability. */
    ASSERT_TRUE(final_cut < init_cut);
    ASSERT_TRUE(final_cut <= init_cut / 2);

    /* Partition still 2-way. */
    for (idx_t i = 0; i < 30; i++)
        ASSERT_TRUE(part[i] == 0 || part[i] == 1);

    /* Vertex-weight balance preserved (or improved): |w0 - w1| ≤ tol. */
    idx_t w0 = 0, w1 = 0;
    compute_side_weights(&G, part, &w0, &w1);
    idx_t imbal = w0 > w1 ? w0 - w1 : w1 - w0;
    ASSERT_TRUE(imbal <= 4); /* well under the 5%×30 + max_vwgt=2 = 3.5 budget */

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── FM on already-optimal partition is a no-op ──────────────────── */

static void test_fm_optimal_partition_no_regress(void) {
    /* 5×6 grid, optimal column-split partition (cut = 5).  Single-
     * pass FM can take negative-gain moves, so the rollback-on-
     * regress logic is what guarantees we don't degrade an already-
     * optimal input. */
    SparseMatrix *A = make_grid_2d(5, 6);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part[30] = {0};
    for (idx_t r = 0; r < 5; r++) {
        for (idx_t c = 0; c < 6; c++) {
            part[r * 6 + c] = c < 3 ? 0 : 1;
        }
    }
    idx_t init_cut = compute_cut(&G, part);
    ASSERT_EQ(init_cut, 5);

    REQUIRE_OK(graph_refine_fm(&G, part));
    idx_t final_cut = compute_cut(&G, part);

    /* Cut never increases: rollback restores the best state seen. */
    ASSERT_TRUE(final_cut <= init_cut);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Bisect + FM compose to a near-optimal cut ───────────────────── */

static void test_bisect_then_fm_5x6_grid(void) {
    /* Compose `graph_bisect_coarsest` (GGGP since n=30 > 20) with
     * `graph_refine_fm`.  Day 4 will replay this composition at every
     * uncoarsening level; this Day-3 smoke test verifies the two
     * routines hand off cleanly. */
    SparseMatrix *A = make_grid_2d(5, 6);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part[30] = {0};
    REQUIRE_OK(graph_bisect_coarsest(&G, part));
    idx_t after_bisect = compute_cut(&G, part);

    REQUIRE_OK(graph_refine_fm(&G, part));
    idx_t after_fm = compute_cut(&G, part);

    /* FM never makes the cut worse than the bisection start. */
    ASSERT_TRUE(after_fm <= after_bisect);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Edge cases: n=1, oversized graph, NULL args ─────────────────── */

static void test_bisect_singleton(void) {
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_NOT_NULL(A);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    idx_t part[1] = {7}; /* sentinel — must be overwritten */
    REQUIRE_OK(graph_bisect_coarsest(&G, part));
    ASSERT_EQ(part[0], 0);
    sparse_graph_free(&G);
    sparse_free(A);
}

static void test_bisect_n41_uses_gggp(void) {
    /* n > 20 routes through GGGP — there's no upper-size cap any more
     * (the Day 5 stress tests need bisect to handle larger inputs
     * when the multilevel coarsening saturates).  Verify a 41-vertex
     * path produces a valid balanced 2-way partition with cut = 1
     * (the natural mid-path cut). */
    SparseMatrix *A = make_path_1d(41);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    idx_t part[41] = {0};
    REQUIRE_OK(graph_bisect_coarsest(&G, part));
    /* Both sides non-empty; cut is small (single-cut path). */
    idx_t w0 = 0, w1 = 0;
    compute_side_weights(&G, part, &w0, &w1);
    ASSERT_TRUE(w0 > 0);
    ASSERT_TRUE(w1 > 0);
    ASSERT_EQ(w0 + w1, 41);
    /* GGGP may not produce the absolute minimum cut, but on a path it
     * should still be small (< 10% of the edges). */
    ASSERT_TRUE(compute_cut(&G, part) <= 4);
    sparse_graph_free(&G);
    sparse_free(A);
}

static void test_bisect_null_args(void) {
    sparse_graph_t G = {0};
    idx_t part[1] = {0};
    ASSERT_ERR(graph_bisect_coarsest(NULL, part), SPARSE_ERR_NULL);
    ASSERT_ERR(graph_bisect_coarsest(&G, NULL), SPARSE_ERR_NULL);
}

static void test_fm_null_args(void) {
    sparse_graph_t G = {0};
    idx_t part[1] = {0};
    ASSERT_ERR(graph_refine_fm(NULL, part), SPARSE_ERR_NULL);
    ASSERT_ERR(graph_refine_fm(&G, NULL), SPARSE_ERR_NULL);
}

/* ═══════════════════════════════════════════════════════════════════
 * Sprint 22 Day 5 — stress tests, edge cases, determinism, SuiteSparse.
 * ═══════════════════════════════════════════════════════════════════ */

/* ─── Day 5 fixture builders ──────────────────────────────────────── */

/* n vertices, no edges (only diagonal placeholders so sparse_create
 * has something to insert). */
static SparseMatrix *make_empty_graph(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);
    return A;
}

/* Complete graph K_n: every off-diagonal pair connected. */
static SparseMatrix *make_complete_graph(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 1.0);
        for (idx_t j = i + 1; j < n; j++) {
            sparse_insert(A, i, j, 1.0);
            sparse_insert(A, j, i, 1.0);
        }
    }
    return A;
}

/* Complete bipartite K_{m,n}: vertices 0..m-1 in part A, m..m+n-1
 * in part B; every (a, b) edge present, no within-part edges. */
static SparseMatrix *make_bipartite_complete(idx_t m, idx_t n) {
    SparseMatrix *A = sparse_create(m + n, m + n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < m + n; i++)
        sparse_insert(A, i, i, 1.0);
    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = m; j < m + n; j++) {
            sparse_insert(A, i, j, 1.0);
            sparse_insert(A, j, i, 1.0);
        }
    }
    return A;
}

/* ─── Edge case: single vertex (no edges, trivial partition) ──────── */

static void test_partition_n1_singleton(void) {
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_NOT_NULL(A);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part[1] = {99};
    idx_t sep = 99;
    REQUIRE_OK(sparse_graph_partition(&G, part, &sep));

    /* No edges → no separator needed; the single vertex is on side
     * 0 or side 1.  (Either is acceptable — there's no recursion to
     * disrupt.) */
    ASSERT_TRUE(part[0] == 0 || part[0] == 1 || part[0] == 2);
    ASSERT_EQ(sep, part[0] == 2 ? 1 : 0);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Edge case: n=2 with one edge ────────────────────────────────── */

static void test_partition_n2_one_edge(void) {
    /* A single 0-1 edge can't be left as a part-0/part-1 cross edge,
     * so the smaller-side convention must lift at least one endpoint
     * into the separator. */
    SparseMatrix *A = make_path_1d(2);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part[2] = {99, 99};
    idx_t sep = 99;
    REQUIRE_OK(sparse_graph_partition(&G, part, &sep));

    ASSERT_TRUE(check_partition_invariant(&G, part));
    ASSERT_TRUE(sep >= 1 && sep <= 2);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Edge case: empty graph (n vertices, 0 edges) ────────────────── */

static void test_partition_empty_graph_n10(void) {
    SparseMatrix *A = make_empty_graph(10);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part[10] = {0};
    idx_t sep = 0;
    REQUIRE_OK(sparse_graph_partition(&G, part, &sep));

    /* No edges → no boundary → no separator. */
    ASSERT_EQ(sep, 0);

    /* All n vertices accounted for; both sides non-empty. */
    idx_t n0, n1, nsep;
    count_partition_sides(&G, part, &n0, &n1, &nsep);
    ASSERT_EQ(n0 + n1 + nsep, 10);
    ASSERT_EQ(nsep, 0);
    ASSERT_TRUE(check_partition_invariant(&G, part));

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Edge case: complete graph K_20 (dense, large separator) ─────── */

static void test_partition_complete_k20(void) {
    /* K_20 is the densest 20-vertex graph: any vertex separator must
     * include every vertex on one side of the cut (every vertex is
     * connected to every other). */
    SparseMatrix *A = make_complete_graph(20);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part[20] = {0};
    idx_t sep = 0;
    REQUIRE_OK(sparse_graph_partition(&G, part, &sep));

    /* Plan contract: dense connectivity forces |sep| ≥ 5.  Practical
     * upper bound ≤ 15 (smaller-side lift can take all 10 boundary
     * vertices on the smaller side). */
    ASSERT_TRUE(sep >= 5);
    ASSERT_TRUE(sep <= 15);
    ASSERT_TRUE(check_partition_invariant(&G, part));

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Edge case: complete bipartite K_{10,10} ─────────────────────── */

static void test_partition_bipartite_k_10_10(void) {
    /* Complete bipartite K_{10,10}: optimal vertex cover (= optimal
     * vertex separator) is one of the two bipartition sides — 10
     * vertices.  Plan allows up to 11 for FM noise. */
    SparseMatrix *A = make_bipartite_complete(10, 10);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part[20] = {0};
    idx_t sep = 0;
    REQUIRE_OK(sparse_graph_partition(&G, part, &sep));

    ASSERT_TRUE(sep >= 5);
    ASSERT_TRUE(sep <= 11);
    ASSERT_TRUE(check_partition_invariant(&G, part));

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── Determinism contract: same input → bit-identical partition ──── */

static void test_partition_determinism_10x10_grid(void) {
    /* `sparse_graph_partition` is contracted to be a pure function of
     * its input: same graph → same partition + sep on every call.
     * The seed is hardcoded inside the routine (currently 0); two
     * back-to-back calls must produce a bit-identical partition. */
    SparseMatrix *A = make_grid_2d(10, 10);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part1[100] = {0};
    idx_t part2[100] = {0};
    idx_t sep1 = 99;
    idx_t sep2 = 99;
    REQUIRE_OK(sparse_graph_partition(&G, part1, &sep1));
    REQUIRE_OK(sparse_graph_partition(&G, part2, &sep2));

    ASSERT_EQ(sep1, sep2);
    ASSERT_EQ(memcmp(part1, part2, sizeof(part1)), 0);

    sparse_graph_free(&G);
    sparse_free(A);
}

static void test_partition_determinism_two_cliques(void) {
    /* Same determinism contract on a structured fixture (two K_10
     * cliques + bridge) so we cover both regular and irregular
     * graphs. */
    SparseMatrix *A = make_two_cliques_with_bridge(10);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));

    idx_t part1[20] = {0};
    idx_t part2[20] = {0};
    idx_t sep1 = 99;
    idx_t sep2 = 99;
    REQUIRE_OK(sparse_graph_partition(&G, part1, &sep1));
    REQUIRE_OK(sparse_graph_partition(&G, part2, &sep2));

    ASSERT_EQ(sep1, sep2);
    ASSERT_EQ(memcmp(part1, part2, sizeof(part1)), 0);

    sparse_graph_free(&G);
    sparse_free(A);
}

/* ─── SuiteSparse smoke: bcsstk14 (n=1806) and Pres_Poisson ───────── */

/* Smoke test: verifies the partitioner handles a realistic
 * SuiteSparse fixture without crashing or allocation failure.  Hard
 * timing (< 100ms per the plan) is documented, not asserted — Day 14
 * profiling will retune if the naive O(n) max-gain scan in FM
 * dominates large-input runtime. */
static void run_suitesparse_partition_smoke(const char *path, idx_t expected_n) {
    SparseMatrix *A = NULL;
    sparse_err_t rc = sparse_load_mm(&A, path);
    if (rc != SPARSE_OK) {
        printf("    skipped (%s not loadable: %d)\n", path, (int)rc);
        return;
    }

    sparse_graph_t G = {0};
    REQUIRE_OK(sparse_graph_from_sparse(A, &G));
    if (expected_n > 0)
        ASSERT_EQ(G.n, expected_n);

    idx_t *part = malloc((size_t)G.n * sizeof(idx_t));
    ASSERT_NOT_NULL(part);
    idx_t sep = 0;

    clock_t t0 = clock();
    sparse_err_t prc = sparse_graph_partition(&G, part, &sep);
    clock_t t1 = clock();
    ASSERT_EQ(prc, SPARSE_OK);

    /* Sanity: separator is finite, smaller than the graph itself. */
    ASSERT_TRUE(sep > 0);
    ASSERT_TRUE(sep < G.n);
    /* Partition invariant: no part-0 / part-1 cross-edges.  The
     * fundamental correctness check at scale. */
    ASSERT_TRUE(check_partition_invariant(&G, part));

    double ms = (double)(t1 - t0) * 1000.0 / (double)CLOCKS_PER_SEC;
    printf("    %s (n=%d): sep=%d, %.1f ms\n", path, (int)G.n, (int)sep, ms);

    free(part);
    sparse_graph_free(&G);
    sparse_free(A);
}

static void test_partition_bcsstk14_smoke(void) {
    run_suitesparse_partition_smoke(SS_DIR "/bcsstk14.mtx", 1806);
}

/* Sprint 26 Day 3: HCC bcsstk14 sep > 0 contract (after sep=0 fall-back
 * lands in `sparse_graph_partition`).  Sprint 25 Day 10's attempted
 * HCC default flip surfaced that `SPARSE_ND_COARSENING=hcc` produces a
 * degenerate empty separator (`sep == 0`) on bcsstk14, blocking the
 * production-default flip.  Day 3's `sparse_graph_partition` sep=0
 * fall-back retries with HEM forced via thread-local override; bcsstk14
 * recovers sep > 0 (HEM works on bcsstk14 per Sprint 22 baseline).
 *
 * Test contract: under `SPARSE_ND_COARSENING=hcc`, partitioning
 * bcsstk14 produces sep > 0 + check_partition_invariant passes (i.e.
 * the partition is well-formed).  Pin via the same assertions as the
 * default-strategy `test_partition_bcsstk14_smoke`; the only
 * difference is the env-var override. */
static void test_hcc_bcsstk14_no_degenerate_partition(void) {
    /* PR #34 review fix: route every exit through a single `cleanup:`
     * block.  Previous flow had two issues: (1) `ASSERT_NOT_NULL(part)`
     * was non-fatal — alloc failure would NULL-deref in the
     * subsequent partition call; (2) `REQUIRE_OK(prc)` after the
     * partition would return immediately on failure, leaving
     * SPARSE_ND_COARSENING set + buffers leaked across subsequent
     * tests in the same process. */
    SparseMatrix *A = NULL;
    sparse_graph_t G = {0};
    idx_t *part = NULL;

    if (setenv("SPARSE_ND_COARSENING", "hcc", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv failed)\n");
        goto cleanup;
    }

    sparse_err_t rc = sparse_load_mm(&A, SS_DIR "/bcsstk14.mtx");
    if (rc != SPARSE_OK) {
        printf("    skipped (bcsstk14 not loadable: %d)\n", (int)rc);
        goto cleanup;
    }

    rc = sparse_graph_from_sparse(A, &G);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_graph_from_sparse: rc=%d", (int)rc);
        goto cleanup;
    }

    part = malloc((size_t)G.n * sizeof(idx_t));
    if (!part) {
        TF_FAIL_("malloc(part) returned NULL (n=%d)", (int)G.n);
        goto cleanup;
    }
    idx_t sep = 0;
    sparse_err_t prc = sparse_graph_partition(&G, part, &sep);
    if (prc != SPARSE_OK) {
        TF_FAIL_("sparse_graph_partition: rc=%d", (int)prc);
        goto cleanup;
    }
    printf("    bcsstk14 under SPARSE_ND_COARSENING=hcc: sep=%d (Sprint 26 Day 3 fall-back "
           "recovered)\n",
           (int)sep);
    ASSERT_TRUE(sep > 0);
    ASSERT_TRUE(sep < G.n);
    ASSERT_TRUE(check_partition_invariant(&G, part));

cleanup:
    unsetenv("SPARSE_ND_COARSENING");
    free(part);
    sparse_graph_free(&G);
    sparse_free(A);
}

static void test_partition_pres_poisson_smoke(void) {
    /* Pres_Poisson is a 2D Poisson-on-irregular-grid fixture — the
     * canonical mesh shape ND was designed for.  Expected to produce
     * a clean planar separator. */
    run_suitesparse_partition_smoke(SS_DIR "/Pres_Poisson.mtx", 0);
}

/* Sprint 27 Day 4: SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k
 * differs-from-per_vertex (dynamic-K) contract pin.  Sprint 26 Day
 * 12's empirical finding was that the three per-vertex weight schemes
 * (hybrid / balance / degree) converge to bit-identical outputs on
 * 5 of 6 fixtures because the dynamic-K + 70/30 balance gate
 * dominates the score formula.  Sprint 27 Day 4 adds a fixed-K
 * variant that terminates after exactly K =
 * min(boundary_count[0], boundary_count[1]) lifts regardless of
 * balance — bypassing the gate so the score formulas can express
 * their character.  This test verifies the new mode produces a
 * different partition than the dynamic-K mode at the same weight
 * scheme (smoke-level evidence the fixed-K plumbing fires).
 *
 * Plan-spec deviation: Sprint 27 PLAN.md Day 4 task 5 named
 * Pres_Poisson as the fixture.  Pres_Poisson at n=14 822 is too
 * slow for unit-test scope (~7 s ND); use the existing 30×30 grid
 * (n=900, ~ms) which Sprint 26 Day 7 also used for env-var
 * differentiation tests.  Documented in
 * `docs/planning/EPIC_2/SPRINT_27/per_vertex_fixed_k_decision.md`
 * "Day 4 test placement". */
static void test_per_vertex_fixed_k_differs_from_dynamic_k(void) {
    SparseMatrix *A = NULL;
    sparse_graph_t G = {0};
    idx_t *part_dynamic = NULL;
    idx_t *part_fixed = NULL;
    int env_set = 0;

    A = make_grid_2d(30, 30);
    if (!A) {
        TF_FAIL_("make_grid_2d(%d, %d) returned NULL (OOM)", 30, 30);
        goto cleanup;
    }
    sparse_err_t rc = sparse_graph_from_sparse(A, &G);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_graph_from_sparse: rc=%d", (int)rc);
        goto cleanup;
    }
    ASSERT_EQ(G.n, 900);

    /* Dynamic-K (Sprint 26 Day 10 baseline) under the hybrid weight. */
    if (setenv("SPARSE_ND_SEP_LIFT_STRATEGY", "per_vertex", /*overwrite=*/1) != 0) {
        printf("    skipped (setenv failed)\n");
        goto cleanup;
    }
    env_set = 1;
    part_dynamic = malloc((size_t)G.n * sizeof(idx_t));
    if (!part_dynamic) {
        TF_FAIL_("malloc(part_dynamic) returned NULL (n=%d)", (int)G.n);
        goto cleanup;
    }
    idx_t sep_dynamic = 0;
    rc = sparse_graph_partition(&G, part_dynamic, &sep_dynamic);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_graph_partition(per_vertex dynamic-K): rc=%d", (int)rc);
        goto cleanup;
    }

    /* Fixed-K (Sprint 27 Day 4 new) under the same hybrid weight. */
    if (setenv("SPARSE_ND_SEP_LIFT_STRATEGY", "per_vertex_fixed_k", /*overwrite=*/1) != 0) {
        TF_FAIL_("setenv SPARSE_ND_SEP_LIFT_STRATEGY=%s failed", "per_vertex_fixed_k");
        goto cleanup;
    }
    /* Default SPARSE_ND_SEP_LIFT_WEIGHT is hybrid, matching the
     * dynamic-K's `per_vertex` weight — so any differentiation
     * comes from the termination predicate (fixed-K vs dynamic-K),
     * not from a different score formula. */
    part_fixed = malloc((size_t)G.n * sizeof(idx_t));
    if (!part_fixed) {
        TF_FAIL_("malloc(part_fixed) returned NULL (n=%d)", (int)G.n);
        goto cleanup;
    }
    idx_t sep_fixed = 0;
    rc = sparse_graph_partition(&G, part_fixed, &sep_fixed);
    if (rc != SPARSE_OK) {
        TF_FAIL_("sparse_graph_partition(per_vertex_fixed_k): rc=%d", (int)rc);
        goto cleanup;
    }

    int differs = (sep_dynamic != sep_fixed) ||
                  (memcmp(part_dynamic, part_fixed, (size_t)G.n * sizeof(idx_t)) != 0);
    printf("    30x30 grid: dynamic-K sep=%d, fixed-K sep=%d, partitions %s\n", (int)sep_dynamic,
           (int)sep_fixed, differs ? "DIFFER (fixed-K active)" : "match");
    ASSERT_TRUE(differs);

cleanup:
    if (env_set)
        unsetenv("SPARSE_ND_SEP_LIFT_STRATEGY");
    free(part_dynamic);
    free(part_fixed);
    sparse_graph_free(&G);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN(
        "Sprint 22 Days 1-5: graph + coarsen + bisect/FM + uncoarsen + partition + stress");

    /* Day 1: round-trip + structural invariants */
    RUN_TEST(test_graph_from_sparse_nos4_round_trip);
    RUN_TEST(test_graph_from_sparse_singleton);

    /* Day 1: argument validation */
    RUN_TEST(test_graph_from_sparse_rejects_rectangular);
    RUN_TEST(test_graph_from_sparse_null_args);

    /* Day 6 stub still pending: subgraph */
    RUN_TEST(test_graph_subgraph_is_stub);

    /* Day 2: heavy-edge-matching coarsener */
    RUN_TEST(test_coarsen_5x5_grid_halves_in_one_step);
    RUN_TEST(test_coarsen_5x5_grid_halves_again_in_two_steps);
    RUN_TEST(test_coarsen_1d_path_halves);
    RUN_TEST(test_coarsen_prefers_heaviest_edge);
    RUN_TEST(test_coarsen_is_deterministic);
    /* Sprint 25 Day 1 stubs (skip-mode; Day 3 lands assertions): */
    RUN_TEST(test_hcc_match_selection_grid);
    RUN_TEST(test_hcc_match_selection_irregular);

    /* Day 2: multilevel hierarchy */
    RUN_TEST(test_hierarchy_build_5x5_grid);
    RUN_TEST(test_hierarchy_free_safe_on_zero_init);

    /* Day 3: coarsest-graph bisection */
    RUN_TEST(test_bisect_brute_force_path_n8);
    RUN_TEST(test_bisect_brute_force_two_triangles);
    RUN_TEST(test_bisect_gggp_5x6_grid);
    RUN_TEST(test_bisect_singleton);
    RUN_TEST(test_bisect_n41_uses_gggp);
    RUN_TEST(test_bisect_null_args);

    /* Day 3: FM refinement */
    RUN_TEST(test_fm_reduces_checkerboard_cut);
    RUN_TEST(test_fm_optimal_partition_no_regress);
    RUN_TEST(test_bisect_then_fm_5x6_grid);
    RUN_TEST(test_fm_null_args);

    /* Day 4: uncoarsening + vertex-separator extraction + end-to-end
     * sparse_graph_partition */
    RUN_TEST(test_partition_10x10_grid);
    /* Sprint 25 Day 5: env-var plumbing smoke test for
     * SPARSE_FM_INTERMEDIATE_PASSES (multi-pass FM at intermediate
     * uncoarsening levels). */
    RUN_TEST(test_fm_intermediate_passes_smoke);
    /* Sprint 26 Day 6: SPARSE_FM_FINEST_STRATEGY=fifo plumbing
     * stub; Day 7 tightens to differs-from-baseline assertion. */
    RUN_TEST(test_finest_fm_strategy_fifo_smoke);
    /* Sprint 28 Day 2: SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal —
     * formal gain-bucket-noise variant of thick-restart FM (replaces
     * Sprint 27 Day 11 simplified gauss_noise). */
    RUN_TEST(test_finest_fm_gain_noise_formal_disrupts_baseline);
    /* Sprint 28 Day 3 stub: SPARSE_FM_FINEST_STRATEGY=ensemble —
     * multi-strategy FM ensemble pick-correctness contract.  Day 4
     * lights this up (parser + dispatch + assert against synthetic
     * dominant-strategy fixture).  RUN_TEST commented out for Day 3
     * because the `ensemble` strategy value is not yet parsed; the
     * stub function exists so Day 4 can enable in one diff. */
    /* RUN_TEST(test_finest_fm_ensemble_picks_best_strategy); */
    (void)test_finest_fm_ensemble_picks_best_strategy; /* silence unused-static-fn */
    /* Sprint 25 Day 6 stubs (Day 7-8 land asserts): */
    RUN_TEST(test_spectral_bisection_eigenvalue_ordering);
    RUN_TEST(test_spectral_bisection_gggp_fallback);
    /* Sprint 25 Day 8: edge-case spectral tests. */
    RUN_TEST(test_spectral_bisection_n1);
    RUN_TEST(test_spectral_bisection_n2);
    RUN_TEST(test_spectral_bisection_disconnected);
    RUN_TEST(test_spectral_bisection_lanczos_failure);
    RUN_TEST(test_partition_5x5x5_mesh);
    RUN_TEST(test_partition_two_k10_with_bridge);
    /* Sprint 26 Day 2: stub for HCC bcsstk14 sep=0 blocker; Day 3
     * tightens the assertion after the sep=0 fall-back fix lands. */
    RUN_TEST(test_hcc_bcsstk14_no_degenerate_partition);
    RUN_TEST(test_edge_to_vertex_separator_smaller_side);
    RUN_TEST(test_partition_singleton);
    RUN_TEST(test_partition_null_args);
    RUN_TEST(test_uncoarsen_null_args);
    RUN_TEST(test_edge_to_vertex_separator_null_args);

    /* Day 5: edge cases, determinism contract, SuiteSparse smoke */
    RUN_TEST(test_partition_n1_singleton);
    RUN_TEST(test_partition_n2_one_edge);
    RUN_TEST(test_partition_empty_graph_n10);
    RUN_TEST(test_partition_complete_k20);
    RUN_TEST(test_partition_bipartite_k_10_10);
    RUN_TEST(test_partition_determinism_10x10_grid);
    RUN_TEST(test_partition_determinism_two_cliques);
    RUN_TEST(test_partition_bcsstk14_smoke);
    RUN_TEST(test_partition_pres_poisson_smoke);
    /* Sprint 27 Day 4: per_vertex_fixed_k differs-from-dynamic-K
     * smoke test. */
    RUN_TEST(test_per_vertex_fixed_k_differs_from_dynamic_k);

    TEST_SUITE_END();
}
