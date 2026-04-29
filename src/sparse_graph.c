/*
 * sparse_graph.c — Multilevel graph partitioner for Sprint 22 nested
 *                  dissection.
 *
 * ─── Design block ─────────────────────────────────────────────────────
 *
 * Sprint 22's nested-dissection ordering (`sparse_reorder_nd`, exposed
 * through `SPARSE_REORDER_ND` in Sprint 22 Day 8) recursively bisects
 * the symmetric adjacency graph of A: order interior vertices of each
 * partition first, then separator vertices last.  The fill-reducing
 * power of ND lives entirely in the quality of the bisection — so
 * Sprint 22 Day 1-5 builds a multilevel vertex-separator partitioner
 * before anything else.
 *
 * **Why a multilevel partitioner.**  Direct partitioning of the
 * original graph (O(|V| · |E|) iterations of Kernighan-Lin) blows up
 * on the SuiteSparse PDE-mesh corpus we care about (Pres_Poisson,
 * bcsstk14).  The multilevel approach (Karypis & Kumar 1998, "A Fast
 * and Highly Quality Multilevel Scheme for Partitioning Irregular
 * Graphs", SIAM J. Sci. Comput. 20:359-392 — the METIS paper)
 * coarsens the graph to a manageable size, runs an exact bisection on
 * the coarsest level, then projects the partition back through the
 * hierarchy with refinement at every level.  Total cost is linear in
 * |V| + |E| under heavy-edge-matching coarsening; partition quality
 * matches single-level KL/FM at a fraction of the runtime.
 *
 * **Three-phase pipeline.**
 *
 *   1. **Coarsening (Day 2).**  Heavy-edge matching: walk vertices
 *      in randomised order with a deterministic seed; for each
 *      unmatched vertex, pick the unmatched neighbour with the
 *      heaviest connecting edge; collapse the pair into a single
 *      coarse vertex with summed weight.  Repeat until the coarsest
 *      graph has n_coarsest ≤ MAX(20, n_orig / 100).  The hierarchy
 *      is stored as an array of `sparse_graph_t *` plus a per-level
 *      `cmap[]` array mapping fine vertices to their coarse
 *      preimages.  Heavy-edge matching is preferred over random
 *      matching because it preserves spectral structure (METIS §4 —
 *      the heavier the edge, the more important the connection it
 *      represents in the original problem).
 *
 *   2. **Initial bisection (Day 3).**  At the coarsest level, run a
 *      brute-force minimum-cut bisection (n ≤ 20, ~10^6 partitions
 *      to enumerate — tractable).  For n in (20, 40] fall back to
 *      Greedy Graph-Growing Partition (METIS §3 — pick a peripheral
 *      vertex, BFS until half the vertex weight is consumed).  The
 *      initial partition feeds the FM refinement phase.
 *
 *   3. **Uncoarsen with FM (Days 3-4).**  Walk back up the hierarchy
 *      one level at a time.  At each level, project the coarse
 *      partition through the cmap (each coarse vertex becomes its
 *      fine preimage on the same side) and run a single
 *      Fiduccia-Mattheyses refinement pass (Fiduccia & Mattheyses
 *      1982, "A Linear-Time Heuristic for Improving Network
 *      Partitions") to clean up the projected boundary.  FM is
 *      O(|E|) per pass with rollback-on-regress; replaying it at
 *      each level converges to KL-quality cuts in linear total
 *      cost.  At the final (finest) level, convert the resulting
 *      edge separator to a vertex separator on the smaller side of
 *      the cut (METIS convention — minimises the recursive ND
 *      tree's height inflation).
 *
 * **Vertex-separator output convention.**  `sparse_graph_partition`
 * writes `part[i] ∈ {0, 1, 2}` (0 = left, 1 = right, 2 = separator).
 * The recursive ND driver consumes this 3-way labelling, recurses on
 * the two subgraphs induced by part==0 and part==1, then appends the
 * separator vertices last to the output permutation.  Sprint 22
 * Day 6 implements that recursion.
 *
 * **Small-graph base case.**  Sprint 22 Day 6's recursion stops when
 * a subgraph has n ≤ ND_BASE_THRESHOLD (provisional 100; tuned in
 * Day 9) and falls through to AMD on the subgraph.  The partitioner
 * itself doesn't impose this threshold — it's an ND-driver
 * decision — but the brute-force bisection at the coarsest level
 * gives the partitioner its own micro-fast-path for n ≤ 20.
 *
 * **Determinism.**  Heavy-edge matching's vertex traversal order is
 * pseudo-randomised with a deterministic seed (mirrors Sprint 21
 * LOBPCG's golden-ratio convention).  Same input + same seed = same
 * partition.  Sprint 22 Day 5 locks this in as a contract test.
 *
 * **References.**
 *   - Karypis & Kumar (1998), "A Fast and Highly Quality Multilevel
 *     Scheme for Partitioning Irregular Graphs", SIAM J. Sci.
 *     Comput. 20:359-392.  The METIS paper.
 *   - George (1973), "Nested Dissection of a Regular Finite Element
 *     Mesh", SIAM J. Numer. Anal. 10:345-363.  The original ND
 *     algorithm — establishes the separator-last fill-reducing
 *     argument.
 *   - Fiduccia & Mattheyses (1982), "A Linear-Time Heuristic for
 *     Improving Network Partitions", DAC'82.  The FM refinement
 *     algorithm.
 *
 * **Sprint 22 Day 1 status.**  This file ships the
 * `sparse_graph_t` data structure, the `sparse_graph_from_sparse` /
 * `sparse_graph_free` helpers, and stubs for `sparse_graph_subgraph`
 * and `sparse_graph_partition` (both return SPARSE_ERR_BADARG).
 * Days 2-4 replace the partitioner stub with the multilevel
 * pipeline; the subgraph helper lands on Day 6 alongside the
 * recursive ND driver.
 */

#include "sparse_graph_internal.h"
#include "sparse_matrix_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_graph_from_sparse — build CSR adjacency from a SparseMatrix.
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Thin wrapper over the existing internal `sparse_build_adj` (defined
 * in `src/sparse_matrix_internal.h` and shared with the AMD / RCM
 * paths in `src/sparse_reorder.c`).  That helper already does the
 * symmetrise-A-plus-A^T pass, drops self-loops, deduplicates
 * neighbours, and returns a CSR pair (xadj, adjncy) that matches
 * `sparse_graph_t`'s representation slot-for-slot.  vwgt and ewgt
 * stay NULL — the partitioner treats unweighted as uniform = 1, and
 * Day 2's coarsener will populate weights on derived graphs as it
 * collapses fine vertices.
 */
sparse_err_t sparse_graph_from_sparse(const SparseMatrix *A, sparse_graph_t *G) {
    if (!G)
        return SPARSE_ERR_NULL;

    /* Pre-clear before the A NULL-check so every error path (NULL A,
     * non-square A, allocation failure) leaves G in the empty state.
     * Callers that defensively call `sparse_graph_free(G)` after an
     * error see a no-op, and tests that probe the post-error fields
     * see deterministic NULL / 0 values. */
    G->n = 0;
    G->xadj = NULL;
    G->adjncy = NULL;
    G->vwgt = NULL;
    G->ewgt = NULL;

    if (!A)
        return SPARSE_ERR_NULL;

    idx_t n = sparse_rows(A);
    if (n != sparse_cols(A))
        return SPARSE_ERR_SHAPE;

    if (n == 0) {
        /* Empty graph: allocate xadj of length 1 holding [0] so the
         * (xadj[n] == |adjncy|) invariant holds vacuously. */
        G->xadj = malloc(sizeof(idx_t));
        if (!G->xadj)
            return SPARSE_ERR_ALLOC;
        G->xadj[0] = 0;
        return SPARSE_OK;
    }

    idx_t *xadj = NULL;
    idx_t *adjncy = NULL;
    sparse_err_t rc = sparse_build_adj(A, &xadj, &adjncy);
    if (rc != SPARSE_OK)
        return rc;

    G->n = n;
    G->xadj = xadj;
    G->adjncy = adjncy;
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_graph_free — release owned arrays, reset to empty state.
 * ═══════════════════════════════════════════════════════════════════════
 *
 * NULL-safe per array; safe on a zero-initialised struct (no-op).
 * Intentionally does not free the struct itself — the struct may be
 * stack-allocated by the caller, or reused for another graph.
 */
void sparse_graph_free(sparse_graph_t *G) {
    if (!G)
        return;
    free(G->xadj);
    free(G->adjncy);
    free(G->vwgt);
    free(G->ewgt);
    G->xadj = NULL;
    G->adjncy = NULL;
    G->vwgt = NULL;
    G->ewgt = NULL;
    G->n = 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_graph_subgraph — Day 6 stub.
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Returns SPARSE_ERR_BADARG (the codebase's "stub in progress"
 * signal — no SPARSE_ERR_NOT_IMPL exists in this library).  The real
 * implementation lands alongside the recursive nested-dissection
 * driver on Sprint 22 Day 6, where it's the single caller.
 */
sparse_err_t sparse_graph_subgraph(const sparse_graph_t *parent, const idx_t *vertex_set, idx_t k,
                                   sparse_graph_t *child, idx_t *vertex_id_map_out) {
    (void)parent;
    (void)vertex_set;
    /* Pre-clear the outputs so callers see deterministic empty
     * state on the BADARG return.  Doubles as a clang-tidy hint
     * that the output parameters really are written (the stub
     * otherwise looks like it could take const pointers). */
    if (child) {
        child->n = 0;
        child->xadj = NULL;
        child->adjncy = NULL;
        child->vwgt = NULL;
        child->ewgt = NULL;
    }
    if (vertex_id_map_out && k > 0)
        memset(vertex_id_map_out, 0, (size_t)k * sizeof(idx_t));
    return SPARSE_ERR_BADARG;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Heavy-edge-matching coarsener (Sprint 22 Day 2).
 * ═══════════════════════════════════════════════════════════════════════
 *
 * splitmix64 PRNG (well-known, public-domain) — same generator used
 * by SplittableRandom and many embedded engines.  Stable across
 * compilers / platforms, so `(graph, seed)` deterministically yields
 * the same coarsened graph everywhere. */
static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* In-place Fisher-Yates shuffle of `perm[0..n-1]` (initially the
 * identity). */
static void fisher_yates_shuffle(idx_t *perm, idx_t n, uint32_t seed) {
    uint64_t state = (uint64_t)seed * 0x9E3779B97F4A7C15ULL + 1;
    for (idx_t i = 0; i < n; i++)
        perm[i] = i;
    for (idx_t i = n - 1; i > 0; i--) {
        uint64_t r = splitmix64_next(&state);
        idx_t j = (idx_t)(r % (uint64_t)(i + 1));
        idx_t tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
}

/* Comparator for the (neighbour, weight) pair used by the per-coarse-
 * vertex sort+merge dedup pass. */
typedef struct {
    idx_t nbr;
    idx_t wt;
} coarse_edge_t;

static int cmp_coarse_edge(const void *a, const void *b) {
    idx_t na = ((const coarse_edge_t *)a)->nbr;
    idx_t nb = ((const coarse_edge_t *)b)->nbr;
    return (na > nb) - (na < nb);
}

sparse_err_t graph_coarsen_heavy_edge_matching(const sparse_graph_t *fine, uint32_t seed,
                                               sparse_graph_t *coarse_out, idx_t *cmap_out) {
    if (!fine || !coarse_out)
        return SPARSE_ERR_NULL;

    /* Pre-clear coarse_out so every error path leaves it empty. */
    coarse_out->n = 0;
    coarse_out->xadj = NULL;
    coarse_out->adjncy = NULL;
    coarse_out->vwgt = NULL;
    coarse_out->ewgt = NULL;

    if (fine->n > 0 && !cmap_out)
        return SPARSE_ERR_NULL;

    if (fine->n == 0) {
        /* Empty graph stays empty.  Allocate xadj of length 1 so the
         * (xadj[n] == |adjncy|) invariant holds. */
        coarse_out->xadj = malloc(sizeof(idx_t));
        if (!coarse_out->xadj)
            return SPARSE_ERR_ALLOC;
        coarse_out->xadj[0] = 0;
        return SPARSE_OK;
    }

    idx_t n_fine = fine->n;

    idx_t *perm = malloc((size_t)n_fine * sizeof(idx_t));
    if (!perm)
        return SPARSE_ERR_ALLOC;
    fisher_yates_shuffle(perm, n_fine, seed);

    /* Build cmap by walking vertices in shuffled order, matching each
     * unmatched vertex to its heaviest unmatched neighbour.  -1 means
     * "not yet assigned to a coarse vertex". */
    for (idx_t i = 0; i < n_fine; i++)
        cmap_out[i] = -1;

    idx_t n_coarse = 0;
    for (idx_t p = 0; p < n_fine; p++) {
        idx_t v = perm[p];
        if (cmap_out[v] != -1)
            continue;
        idx_t best_nbr = -1;
        idx_t best_wt = 0; /* edge weights are positive, so 0 is a safe floor */
        for (idx_t k = fine->xadj[v]; k < fine->xadj[v + 1]; k++) {
            idx_t u = fine->adjncy[k];
            if (cmap_out[u] != -1)
                continue;
            idx_t w = fine->ewgt ? fine->ewgt[k] : 1;
            if (w > best_wt) {
                best_wt = w;
                best_nbr = u;
            }
        }
        cmap_out[v] = n_coarse;
        if (best_nbr != -1)
            cmap_out[best_nbr] = n_coarse;
        n_coarse++;
    }
    free(perm);

    /* Every fine vertex is mapped to a coarse vertex by the matching
     * loop above (the first iteration with cmap_out[v] == -1 always
     * fires for n_fine > 0), so n_coarse ≥ 1 here.  Establishing this
     * invariant explicitly suppresses clang-analyzer's worst-case
     * "loop didn't execute → n_coarse = 0" false positive, which
     * would otherwise flag the calloc(0, ...) and the cmap[i]
     * indexings below as out-of-bounds. */
    if (n_coarse <= 0) {
        coarse_out->xadj = malloc(sizeof(idx_t));
        if (!coarse_out->xadj)
            return SPARSE_ERR_ALLOC;
        coarse_out->xadj[0] = 0;
        return SPARSE_OK;
    }

    /* Build the coarse graph.  First pass: aggregate vwgt; count
     * surviving (non-self-loop) coarse-edge incidences per coarse
     * vertex (with duplicates — those are merged in the dedup pass).
     */
    idx_t *vwgt_coarse = calloc((size_t)n_coarse, sizeof(idx_t));
    idx_t *deg_coarse = calloc((size_t)n_coarse, sizeof(idx_t));
    if (!vwgt_coarse || !deg_coarse) {
        free(vwgt_coarse);
        free(deg_coarse);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t i = 0; i < n_fine; i++) {
        idx_t c = cmap_out[i];
        vwgt_coarse[c] += fine->vwgt ? fine->vwgt[i] : 1;
    }
    /* Walk fine adjacency once to count how many slots each coarse
     * vertex's adj list will need in the with-duplicates layout.
     * Each fine edge {i, j} (i < j) contributes to ci and cj if
     * ci != cj. */
    for (idx_t i = 0; i < n_fine; i++) {
        idx_t ci = cmap_out[i];
        for (idx_t k = fine->xadj[i]; k < fine->xadj[i + 1]; k++) {
            idx_t j = fine->adjncy[k];
            if (j <= i)
                continue; /* upper-triangle walk: avoid double-counting */
            idx_t cj = cmap_out[j];
            if (ci == cj)
                continue; /* collapsed edge — drop the would-be self-loop */
            /* cmap_out values are constructed in [0, n_coarse), but the
             * static analyser doesn't track that constraint, so it
             * reports these increments as out-of-bounds.  Suppress the
             * false positive — same pattern as `src/sparse_etree.c`. */
            deg_coarse[ci]++; // NOLINT(clang-analyzer-security.ArrayBound)
            deg_coarse[cj]++; // NOLINT(clang-analyzer-security.ArrayBound)
        }
    }

    /* xadj = prefix sum of deg_coarse.  Allocate adjncy/ewgt at the
     * total degree.  The dedup pass below shrinks these in place. */
    idx_t *xadj = malloc((size_t)(n_coarse + 1) * sizeof(idx_t));
    if (!xadj) {
        free(vwgt_coarse);
        free(deg_coarse);
        return SPARSE_ERR_ALLOC;
    }
    xadj[0] = 0;
    for (idx_t c = 0; c < n_coarse; c++)
        xadj[c + 1] = xadj[c] + deg_coarse[c];
    idx_t total = xadj[n_coarse]; // NOLINT(clang-analyzer-security.ArrayBound)

    /* coarse_edge_t bucket per coarse vertex; pre-laid-out via xadj.
     * Zero-init via calloc so clang-analyzer can see the slots are
     * defined even on the (unreachable) total == 0 path. */
    coarse_edge_t *buckets = calloc((size_t)(total > 0 ? total : 1), sizeof(coarse_edge_t));
    idx_t *cursor = calloc((size_t)n_coarse, sizeof(idx_t));
    if (!buckets || !cursor) {
        free(buckets);
        free(cursor);
        free(xadj);
        free(vwgt_coarse);
        free(deg_coarse);
        return SPARSE_ERR_ALLOC;
    }
    free(deg_coarse);

    /* Pass 2: fill the buckets (with duplicates). */
    for (idx_t i = 0; i < n_fine; i++) {
        idx_t ci = cmap_out[i];
        for (idx_t k = fine->xadj[i]; k < fine->xadj[i + 1]; k++) {
            idx_t j = fine->adjncy[k];
            if (j <= i)
                continue;
            idx_t cj = cmap_out[j];
            if (ci == cj)
                continue;
            idx_t w = fine->ewgt ? fine->ewgt[k] : 1;
            /* xadj[ci] + cursor[ci] is bounded by xadj[ci+1] thanks to
             * the pass-1 degree count, but the analyser doesn't track
             * the relationship.  Same false positive as above. */
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            buckets[xadj[ci] + cursor[ci]++] = (coarse_edge_t){.nbr = cj, .wt = w};
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            buckets[xadj[cj] + cursor[cj]++] = (coarse_edge_t){.nbr = ci, .wt = w};
        }
    }
    free(cursor);

    /* Pass 3: per coarse vertex, sort by neighbour and merge
     * consecutive equal-neighbour entries by summing weights.  Track
     * the compacted offset so we can rebuild xadj at the end. */
    idx_t *new_deg = calloc((size_t)n_coarse, sizeof(idx_t));
    if (!new_deg) {
        free(buckets);
        free(xadj);
        free(vwgt_coarse);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t c = 0; c < n_coarse; c++) {
        idx_t start = xadj[c];
        idx_t len = xadj[c + 1] - start;
        if (len <= 1) {
            new_deg[c] = len;
            continue;
        }
        coarse_edge_t *list = &buckets[start];
        qsort(list, (size_t)len, sizeof(coarse_edge_t), cmp_coarse_edge);
        idx_t write = 0;
        /* `list` spans `len > 1` entries from the buckets allocation;
         * `write` starts at 0 and only ever increases up to `len`, so
         * every `list[write]` and `list[write - 1]` access stays in
         * bounds.  Analyser doesn't track this. */
        list[write++] = list[0]; // NOLINT(clang-analyzer-security.ArrayBound)
        for (idx_t a = 1; a < len; a++) {
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            if (list[a].nbr == list[write - 1].nbr) {
                list[write - 1].wt += list[a].wt;
            } else {
                list[write++] = list[a];
            }
        }
        new_deg[c] = write;
    }

    /* Compact: build the final xadj/adjncy/ewgt arrays. */
    idx_t *final_xadj = malloc((size_t)(n_coarse + 1) * sizeof(idx_t));
    if (!final_xadj) {
        free(buckets);
        free(xadj);
        free(vwgt_coarse);
        free(new_deg);
        return SPARSE_ERR_ALLOC;
    }
    final_xadj[0] = 0;
    for (idx_t c = 0; c < n_coarse; c++)
        final_xadj[c + 1] = final_xadj[c] + new_deg[c];
    idx_t final_total = final_xadj[n_coarse];

    idx_t *final_adjncy = malloc((size_t)(final_total > 0 ? final_total : 1) * sizeof(idx_t));
    idx_t *final_ewgt = malloc((size_t)(final_total > 0 ? final_total : 1) * sizeof(idx_t));
    if (!final_adjncy || !final_ewgt) {
        free(final_adjncy);
        free(final_ewgt);
        free(final_xadj);
        free(buckets);
        free(xadj);
        free(vwgt_coarse);
        free(new_deg);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t c = 0; c < n_coarse; c++) {
        /* xadj[c] is in [0, total] by construction (prefix sum of
         * deg_coarse), so &buckets[xadj[c]] is in bounds.  The
         * analyser conflates this with a paths-where-total-could-be-0
         * scenario the matching loop already rules out. */
        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
        coarse_edge_t *src = &buckets[xadj[c]];
        idx_t dst = final_xadj[c];
        for (idx_t a = 0; a < new_deg[c]; a++) {
            final_adjncy[dst + a] = src[a].nbr;
            final_ewgt[dst + a] = src[a].wt;
        }
    }

    free(buckets);
    free(xadj);
    free(new_deg);

    coarse_out->n = n_coarse;
    coarse_out->xadj = final_xadj;
    coarse_out->adjncy = final_adjncy;
    coarse_out->vwgt = vwgt_coarse;
    coarse_out->ewgt = final_ewgt;
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Multilevel hierarchy (Sprint 22 Day 2).
 * ═══════════════════════════════════════════════════════════════════════ */

void sparse_graph_hierarchy_free(sparse_graph_hierarchy_t *h) {
    if (!h)
        return;
    if (h->coarse) {
        for (int i = 0; i < h->nlevels; i++)
            sparse_graph_free(&h->coarse[i]);
        free(h->coarse);
    }
    if (h->cmaps) {
        for (int i = 0; i < h->nlevels; i++)
            free(h->cmaps[i]);
        free(h->cmaps);
    }
    h->coarse = NULL;
    h->cmaps = NULL;
    h->nlevels = 0;
}

sparse_err_t sparse_graph_hierarchy_build(const sparse_graph_t *root, uint32_t seed,
                                          sparse_graph_hierarchy_t *h) {
    if (!h)
        return SPARSE_ERR_NULL;
    h->nlevels = 0;
    h->coarse = NULL;
    h->cmaps = NULL;
    if (!root)
        return SPARSE_ERR_NULL;

    if (root->n == 0)
        return SPARSE_OK; /* nothing to coarsen */

    idx_t n_root = root->n;
    idx_t base_threshold = n_root / 100;
    if (base_threshold < 20)
        base_threshold = 20;
    /* log2(n) + 5 ceiling; cap at a defensive 64 to avoid pathology
     * on enormous n. */
    int level_cap = 5;
    {
        idx_t v = n_root;
        while (v > 1) {
            v >>= 1;
            level_cap++;
        }
        if (level_cap > 64)
            level_cap = 64;
    }

    /* Grow coarse[] and cmaps[] in place by realloc; pre-allocate a
     * starting slot count to avoid quadratic realloc costs. */
    int cap = 8;
    if (cap > level_cap)
        cap = level_cap;
    h->coarse = calloc((size_t)cap, sizeof(sparse_graph_t));
    h->cmaps = calloc((size_t)cap, sizeof(idx_t *));
    if (!h->coarse || !h->cmaps) {
        sparse_graph_hierarchy_free(h);
        return SPARSE_ERR_ALLOC;
    }

    const sparse_graph_t *prev = root;
    for (int level = 0; level < level_cap; level++) {
        idx_t n_prev = prev->n;
        idx_t *cmap = malloc((size_t)n_prev * sizeof(idx_t));
        if (!cmap) {
            sparse_graph_hierarchy_free(h);
            return SPARSE_ERR_ALLOC;
        }
        sparse_graph_t coarse = {0};
        /* Per-level seed perturbation so each level shuffles its
         * vertices differently (otherwise the same seed picks the
         * same matching pattern at every level). */
        sparse_err_t rc =
            graph_coarsen_heavy_edge_matching(prev, seed + (uint32_t)level, &coarse, cmap);
        if (rc != SPARSE_OK) {
            free(cmap);
            sparse_graph_hierarchy_free(h);
            return rc;
        }
        /* Bail-out checks: stop and keep the levels accumulated so far. */
        idx_t n_coarse = coarse.n;
        int no_progress = (n_coarse * 10 > n_prev * 9); /* > 90% — no halving */
        int small_enough = (n_coarse <= base_threshold);

        /* Grow capacity if needed before stashing this level. */
        if (level >= cap) {
            int new_cap = cap * 2;
            if (new_cap > level_cap)
                new_cap = level_cap;
            sparse_graph_t *new_coarse =
                realloc(h->coarse, (size_t)new_cap * sizeof(sparse_graph_t));
            idx_t **new_cmaps = realloc(h->cmaps, (size_t)new_cap * sizeof(idx_t *));
            if (!new_coarse || !new_cmaps) {
                free(new_coarse ? new_coarse : h->coarse);
                free(new_cmaps ? new_cmaps : h->cmaps);
                h->coarse = NULL;
                h->cmaps = NULL;
                sparse_graph_free(&coarse);
                free(cmap);
                sparse_graph_hierarchy_free(h);
                return SPARSE_ERR_ALLOC;
            }
            /* Zero the newly-grown tail so a downstream cleanup sees
             * NULL pointers in the unused slots. */
            for (int i = cap; i < new_cap; i++) {
                memset(&new_coarse[i], 0, sizeof(sparse_graph_t));
                new_cmaps[i] = NULL;
            }
            h->coarse = new_coarse;
            h->cmaps = new_cmaps;
            cap = new_cap;
        }

        if (level == 0 && no_progress) {
            /* Coarsening made no progress on the very first pass —
             * report empty hierarchy so the caller falls back to
             * single-level partitioning. */
            sparse_graph_free(&coarse);
            free(cmap);
            sparse_graph_hierarchy_free(h);
            return SPARSE_OK;
        }
        if (no_progress) {
            /* Subsequent no-progress: keep the levels we already
             * accumulated; drop this level. */
            sparse_graph_free(&coarse);
            free(cmap);
            break;
        }

        h->coarse[level] = coarse;
        h->cmaps[level] = cmap;
        h->nlevels = level + 1;
        prev = &h->coarse[level];

        if (small_enough)
            break;
    }
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Coarsest-graph bisection + FM refinement (Sprint 22 Day 3).
 * ═══════════════════════════════════════════════════════════════════════
 */

/* Compute the cut weight of a 2-way partition.  Iterates each
 * undirected edge once via the i < j upper-triangle convention. */
static idx_t compute_cut_weight(const sparse_graph_t *G, const idx_t *part) {
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

/* Brute-force minimum-cut bisection for n ≤ 20.  Vertex 0 is fixed
 * to side 0 (the side-swapped mirror has identical cut, so this
 * halves the search), then 2^(n-1) ≤ 524288 patterns are scanned.
 * The lowest-cut pattern that satisfies vertex-weight balance
 * |w0 - w1| ≤ max_vwgt wins. */
static sparse_err_t bisect_brute_force(const sparse_graph_t *G, idx_t *part_out) {
    idx_t n = G->n;
    if (n == 1) {
        part_out[0] = 0;
        return SPARSE_OK;
    }

    idx_t max_vwgt = 1;
    for (idx_t i = 0; i < n; i++) {
        idx_t w = G->vwgt ? G->vwgt[i] : 1;
        if (w > max_vwgt)
            max_vwgt = w;
    }
    /* Tolerance: max_vwgt allows a single-vertex move to balance. */
    idx_t tolerance = max_vwgt;

    int have_best = 0;
    idx_t best_cut = 0;
    uint32_t best_pat = 0;
    /* `mid_pat` is a fallback for the (rare) case where no balanced
     * partition exists within tolerance — pick the most-balanced
     * pattern at the lowest imbalance seen so the routine never
     * returns garbage. */
    int have_mid = 0;
    idx_t best_imbal = 0;
    uint32_t mid_pat = 0;
    idx_t mid_cut = 0;

    uint32_t total_pats = 1U << (uint32_t)(n - 1);
    for (uint32_t p = 0; p < total_pats; p++) {
        uint32_t pattern = p << 1; /* bit 0 = vertex 0's side = 0 */

        idx_t w0 = 0;
        idx_t w1 = 0;
        for (idx_t i = 0; i < n; i++) {
            idx_t w = G->vwgt ? G->vwgt[i] : 1;
            if ((pattern >> (uint32_t)i) & 1U)
                w1 += w;
            else
                w0 += w;
        }
        idx_t imbal = w0 > w1 ? w0 - w1 : w1 - w0;
        idx_t cut = 0;
        for (idx_t i = 0; i < n; i++) {
            uint32_t side_i = (pattern >> (uint32_t)i) & 1U;
            for (idx_t k = G->xadj[i]; k < G->xadj[i + 1]; k++) {
                idx_t j = G->adjncy[k];
                if (j <= i)
                    continue;
                uint32_t side_j = (pattern >> (uint32_t)j) & 1U;
                if (side_i != side_j)
                    cut += G->ewgt ? G->ewgt[k] : 1;
            }
        }

        if (imbal <= tolerance) {
            if (!have_best || cut < best_cut) {
                have_best = 1;
                best_cut = cut;
                best_pat = pattern;
            }
        }
        if (!have_mid || imbal < best_imbal || (imbal == best_imbal && cut < mid_cut)) {
            have_mid = 1;
            best_imbal = imbal;
            mid_pat = pattern;
            mid_cut = cut;
        }
    }

    uint32_t winner = have_best ? best_pat : mid_pat;
    for (idx_t i = 0; i < n; i++)
        part_out[i] = (winner >> (uint32_t)i) & 1U;
    return SPARSE_OK;
}

/* BFS from `start` filling `dist[v]` (-1 if unreachable).  Caller
 * provides scratch queue of length ≥ G->n. */
static void bfs_distances(const sparse_graph_t *G, idx_t start, idx_t *dist, idx_t *queue) {
    for (idx_t i = 0; i < G->n; i++)
        dist[i] = -1;
    dist[start] = 0;
    idx_t head = 0;
    idx_t tail = 0;
    queue[tail++] = start;
    while (head < tail) {
        idx_t v = queue[head++];
        for (idx_t k = G->xadj[v]; k < G->xadj[v + 1]; k++) {
            idx_t u = G->adjncy[k];
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                queue[tail++] = u;
            }
        }
    }
}

/* Greedy Graph-Growing Partition (METIS §3) for n in [21, 40]:
 * find a peripheral vertex via two BFS passes, BFS-grow side 0 from
 * it until half the vertex weight is consumed, leave the rest on
 * side 1.  The resulting partition is often coarsely balanced —
 * Day 4's per-level FM refinement is what actually polishes the
 * cut. */
static sparse_err_t bisect_gggp(const sparse_graph_t *G, idx_t *part_out) {
    idx_t n = G->n;
    idx_t *dist = malloc((size_t)n * sizeof(idx_t));
    idx_t *queue = malloc((size_t)n * sizeof(idx_t));
    int *visited = calloc((size_t)n, sizeof(int));
    if (!dist || !queue || !visited) {
        free(dist);
        free(queue);
        free(visited);
        return SPARSE_ERR_ALLOC;
    }

    /* Two-BFS peripheral-vertex finder. */
    bfs_distances(G, 0, dist, queue);
    idx_t v0 = 0;
    idx_t best_d = 0;
    for (idx_t i = 0; i < n; i++) {
        if (dist[i] > best_d) {
            best_d = dist[i];
            v0 = i;
        }
    }
    bfs_distances(G, v0, dist, queue);
    idx_t v_periph = v0;
    best_d = 0;
    for (idx_t i = 0; i < n; i++) {
        if (dist[i] > best_d) {
            best_d = dist[i];
            v_periph = i;
        }
    }

    idx_t total_vwgt = 0;
    for (idx_t i = 0; i < n; i++)
        total_vwgt += G->vwgt ? G->vwgt[i] : 1;
    idx_t target = total_vwgt / 2;

    for (idx_t i = 0; i < n; i++)
        part_out[i] = 1;

    /* BFS from peripheral; stop assigning to side 0 once target is
     * reached (or surpassed by the most recent push).  Disconnected
     * components beyond the periphery's cluster stay on side 1. */
    idx_t head = 0;
    idx_t tail = 0;
    queue[tail++] = v_periph;
    visited[v_periph] = 1;
    idx_t consumed = 0;
    {
        idx_t w = G->vwgt ? G->vwgt[v_periph] : 1;
        part_out[v_periph] = 0;
        consumed += w;
    }
    while (head < tail && consumed < target) {
        idx_t v = queue[head++];
        for (idx_t k = G->xadj[v]; k < G->xadj[v + 1]; k++) {
            idx_t u = G->adjncy[k];
            if (visited[u])
                continue;
            visited[u] = 1;
            queue[tail++] = u;
            idx_t w = G->vwgt ? G->vwgt[u] : 1;
            if (consumed + w > target + 1 && consumed > 0)
                continue; /* would overshoot; leave on side 1 */
            part_out[u] = 0;
            consumed += w;
        }
    }

    free(visited);
    free(queue);
    free(dist);
    return SPARSE_OK;
}

sparse_err_t graph_bisect_coarsest(const sparse_graph_t *G, idx_t *part_out) {
    if (!G || !part_out)
        return SPARSE_ERR_NULL;
    if (G->n > 40)
        return SPARSE_ERR_BADARG;
    if (G->n == 0)
        return SPARSE_OK;
    if (G->n <= 20)
        return bisect_brute_force(G, part_out);
    return bisect_gggp(G, part_out);
}

sparse_err_t graph_refine_fm(const sparse_graph_t *G, idx_t *part_io) {
    if (!G || !part_io)
        return SPARSE_ERR_NULL;
    if (G->n == 0)
        return SPARSE_OK;

    idx_t n = G->n;

    /* Per-vertex gain = (sum of edge weights to other-side neighbours)
     *                 − (sum of edge weights to same-side neighbours).
     * Moving v flips the cut by -gain[v] (positive gain ⇒ smaller cut). */
    idx_t *gain = malloc((size_t)n * sizeof(idx_t));
    int *locked = calloc((size_t)n, sizeof(int));
    idx_t *best_part = malloc((size_t)n * sizeof(idx_t));
    if (!gain || !locked || !best_part) {
        free(gain);
        free(locked);
        free(best_part);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t v = 0; v < n; v++) {
        idx_t internal = 0;
        idx_t external = 0;
        for (idx_t k = G->xadj[v]; k < G->xadj[v + 1]; k++) {
            idx_t u = G->adjncy[k];
            idx_t w = G->ewgt ? G->ewgt[k] : 1;
            if (part_io[v] == part_io[u])
                internal += w;
            else
                external += w;
        }
        gain[v] = external - internal;
    }

    idx_t cur_cut = compute_cut_weight(G, part_io);
    idx_t best_cut = cur_cut;
    memcpy(best_part, part_io, (size_t)n * sizeof(idx_t));

    /* Side weights for balance tracking. */
    idx_t w0 = 0;
    idx_t w1 = 0;
    idx_t max_vwgt = 1;
    for (idx_t i = 0; i < n; i++) {
        idx_t w = G->vwgt ? G->vwgt[i] : 1;
        if (w > max_vwgt)
            max_vwgt = w;
        if (part_io[i] == 0)
            w0 += w;
        else
            w1 += w;
    }
    idx_t total_vwgt = w0 + w1;
    idx_t init_imbal = w0 > w1 ? w0 - w1 : w1 - w0;
    idx_t max_imbal = total_vwgt / 20; /* 5% of total */
    if (max_imbal < init_imbal)
        max_imbal = init_imbal;
    max_imbal += max_vwgt;

    /* FM main loop.  Pick the highest-gain unlocked vertex (positive
     * or negative — accepting transient cut increases lets FM escape
     * shallow local minima), move it, lock it, update neighbour
     * gains.  After the pass, restore the best (lowest-cut) state
     * seen during the walk. */
    for (idx_t step = 0; step < n; step++) {
        idx_t best_v = -1;
        idx_t best_g = 0;
        int have_candidate = 0;
        for (idx_t v = 0; v < n; v++) {
            if (locked[v])
                continue;
            idx_t v_w = G->vwgt ? G->vwgt[v] : 1;
            idx_t new_w0 = part_io[v] == 0 ? w0 - v_w : w0 + v_w;
            idx_t new_w1 = part_io[v] == 0 ? w1 + v_w : w1 - v_w;
            idx_t new_imbal = new_w0 > new_w1 ? new_w0 - new_w1 : new_w1 - new_w0;
            if (new_imbal > max_imbal)
                continue;
            /* Cache gain[v] in a local so the comparison `v < best_v`
             * isn't on the same line as `gain[v]`; cppcheck otherwise
             * raises a spurious arrayIndexThenCheck on the v compare. */
            idx_t gv = gain[v];
            if (!have_candidate || gv > best_g || (gv == best_g && v < best_v)) {
                have_candidate = 1;
                best_g = gv;
                best_v = v;
            }
        }
        if (!have_candidate)
            break;

        /* Move best_v: cut changes by -gain[best_v]. */
        cur_cut -= best_g;
        idx_t v_w = G->vwgt ? G->vwgt[best_v] : 1;
        idx_t old_side = part_io[best_v];
        idx_t new_side = 1 - old_side;
        if (old_side == 0) {
            w0 -= v_w;
            w1 += v_w;
        } else {
            w0 += v_w;
            w1 -= v_w;
        }
        part_io[best_v] = new_side;
        locked[best_v] = 1;

        /* Update gains for unlocked neighbours.  Neighbours now on
         * the new side: edge (best_v, u) flipped from external to
         * internal for u, so gain[u] -= 2w.  Neighbours on the old
         * side: edge flipped internal → external, gain[u] += 2w. */
        for (idx_t k = G->xadj[best_v]; k < G->xadj[best_v + 1]; k++) {
            idx_t u = G->adjncy[k];
            if (locked[u])
                continue;
            idx_t w = G->ewgt ? G->ewgt[k] : 1;
            if (part_io[u] == new_side)
                gain[u] -= 2 * w;
            else
                gain[u] += 2 * w;
        }

        if (cur_cut < best_cut) {
            best_cut = cur_cut;
            memcpy(best_part, part_io, (size_t)n * sizeof(idx_t));
        }
    }

    /* Roll back to the best state. */
    memcpy(part_io, best_part, (size_t)n * sizeof(idx_t));

    free(gain);
    free(locked);
    free(best_part);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Uncoarsening + vertex-separator extraction (Sprint 22 Day 4).
 * ═══════════════════════════════════════════════════════════════════════
 */

sparse_err_t graph_uncoarsen(const sparse_graph_t *root, const sparse_graph_hierarchy_t *h,
                             const idx_t *coarsest_part, idx_t *root_part_out) {
    if (!root || !h || !coarsest_part || !root_part_out)
        return SPARSE_ERR_NULL;

    /* No coarsening occurred — coarsest_part is on root.  Just copy
     * and run a single FM polish. */
    if (h->nlevels == 0) {
        if (root->n > 0)
            memcpy(root_part_out, coarsest_part, (size_t)root->n * sizeof(idx_t));
        return graph_refine_fm(root, root_part_out);
    }

    /* Two ping-pong buffers sized to the largest level (root). */
    idx_t max_n = root->n;
    for (int i = 0; i < h->nlevels; i++) {
        if (h->coarse[i].n > max_n)
            max_n = h->coarse[i].n;
    }
    idx_t *cur = malloc((size_t)max_n * sizeof(idx_t));
    idx_t *next = malloc((size_t)max_n * sizeof(idx_t));
    if (!cur || !next) {
        free(cur);
        free(next);
        return SPARSE_ERR_ALLOC;
    }

    /* Seed `cur` with the coarsest partition. */
    idx_t coarsest_n = h->coarse[h->nlevels - 1].n;
    if (coarsest_n > 0)
        memcpy(cur, coarsest_part, (size_t)coarsest_n * sizeof(idx_t));

    /* Walk levels from coarsest down to root.  At each step, project
     * `cur` (on coarse[level]) through cmaps[level] onto the next-
     * finer graph (root if level == 0, else coarse[level - 1]) and
     * refine the result with FM. */
    for (int level = h->nlevels - 1; level >= 0; level--) {
        const sparse_graph_t *dst_graph = (level == 0) ? root : &h->coarse[level - 1];
        const idx_t *cmap = h->cmaps[level];
        for (idx_t i = 0; i < dst_graph->n; i++) {
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            next[i] = cur[cmap[i]];
        }
        sparse_err_t rc = graph_refine_fm(dst_graph, next);
        if (rc != SPARSE_OK) {
            free(cur);
            free(next);
            return rc;
        }
        idx_t *tmp = cur;
        cur = next;
        next = tmp;
    }

    if (root->n > 0)
        memcpy(root_part_out, cur, (size_t)root->n * sizeof(idx_t));
    free(cur);
    free(next);
    return SPARSE_OK;
}

sparse_err_t graph_edge_separator_to_vertex_separator(const sparse_graph_t *G, idx_t *part_io) {
    if (!G || !part_io)
        return SPARSE_ERR_NULL;
    if (G->n == 0)
        return SPARSE_OK;

    /* Side weights drive the smaller-side decision (METIS convention). */
    idx_t w0 = 0;
    idx_t w1 = 0;
    for (idx_t i = 0; i < G->n; i++) {
        idx_t w = G->vwgt ? G->vwgt[i] : 1;
        if (part_io[i] == 0)
            w0 += w;
        else
            w1 += w;
    }
    idx_t smaller_side = (w1 < w0) ? 1 : 0;
    idx_t other_side = 1 - smaller_side;

    /* Two-pass: first mark every boundary vertex on the smaller side,
     * then move the marks into part_io.  Splitting the marking from
     * the move keeps the boundary check simple — once we start
     * moving, "neighbour on other side" gets ambiguous. */
    int *is_boundary = calloc((size_t)G->n, sizeof(int));
    if (!is_boundary)
        return SPARSE_ERR_ALLOC;

    for (idx_t i = 0; i < G->n; i++) {
        if (part_io[i] != smaller_side)
            continue;
        for (idx_t k = G->xadj[i]; k < G->xadj[i + 1]; k++) {
            idx_t j = G->adjncy[k];
            if (part_io[j] == other_side) {
                is_boundary[i] = 1;
                break;
            }
        }
    }
    for (idx_t i = 0; i < G->n; i++) {
        if (is_boundary[i])
            part_io[i] = 2;
    }

    free(is_boundary);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_graph_partition — full multilevel pipeline (Sprint 22 Day 4).
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Composes the four phases shipped over Sprint 22 Days 2-4:
 *   1. Build the multilevel coarsening hierarchy (Day 2).
 *   2. Bisect the coarsest level (Day 3) and FM-refine.
 *   3. Uncoarsen back to the root with FM at every level (Day 4).
 *   4. Convert the final 2-way edge separator to a 3-way vertex
 *      separator on the smaller side (Day 4).
 */
sparse_err_t sparse_graph_partition(const sparse_graph_t *G, idx_t *part_out, idx_t *sep_out) {
    if (!G || !part_out)
        return SPARSE_ERR_NULL;
    if (sep_out)
        *sep_out = 0;
    if (G->n == 0)
        return SPARSE_OK;

    sparse_graph_hierarchy_t h = {0};
    sparse_err_t rc = sparse_graph_hierarchy_build(G, /*seed=*/0U, &h);
    if (rc != SPARSE_OK)
        return rc;

    /* Coarsest = last hierarchy level if any coarsening happened, else
     * the root itself (hierarchy.nlevels == 0 means the matching
     * saturated immediately). */
    const sparse_graph_t *coarsest = (h.nlevels > 0) ? &h.coarse[h.nlevels - 1] : G;
    if (coarsest->n > 40) {
        /* Saturated matching on a too-large input — Day 3's bisect can't
         * handle it.  Bubble up the error rather than silently returning
         * a degenerate partition. */
        sparse_graph_hierarchy_free(&h);
        return SPARSE_ERR_BADARG;
    }

    idx_t *coarsest_part = malloc((size_t)coarsest->n * sizeof(idx_t));
    if (!coarsest_part) {
        sparse_graph_hierarchy_free(&h);
        return SPARSE_ERR_ALLOC;
    }
    rc = graph_bisect_coarsest(coarsest, coarsest_part);
    if (rc == SPARSE_OK)
        rc = graph_refine_fm(coarsest, coarsest_part);
    if (rc != SPARSE_OK) {
        free(coarsest_part);
        sparse_graph_hierarchy_free(&h);
        return rc;
    }

    if (h.nlevels == 0) {
        /* Already at root size — just copy the coarsest partition over. */
        memcpy(part_out, coarsest_part, (size_t)G->n * sizeof(idx_t));
    } else {
        rc = graph_uncoarsen(G, &h, coarsest_part, part_out);
        if (rc != SPARSE_OK) {
            free(coarsest_part);
            sparse_graph_hierarchy_free(&h);
            return rc;
        }
    }
    free(coarsest_part);
    sparse_graph_hierarchy_free(&h);

    rc = graph_edge_separator_to_vertex_separator(G, part_out);
    if (rc != SPARSE_OK)
        return rc;

    if (sep_out) {
        idx_t sep = 0;
        for (idx_t i = 0; i < G->n; i++) {
            if (part_out[i] == 2)
                sep++;
        }
        *sep_out = sep;
    }
    return SPARSE_OK;
}
