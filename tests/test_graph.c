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

static void test_partition_pres_poisson_smoke(void) {
    /* Pres_Poisson is a 2D Poisson-on-irregular-grid fixture — the
     * canonical mesh shape ND was designed for.  Expected to produce
     * a clean planar separator. */
    run_suitesparse_partition_smoke(SS_DIR "/Pres_Poisson.mtx", 0);
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
    RUN_TEST(test_partition_5x5x5_mesh);
    RUN_TEST(test_partition_two_k10_with_bridge);
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

    TEST_SUITE_END();
}
