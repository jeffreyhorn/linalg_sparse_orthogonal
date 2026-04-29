#ifndef SPARSE_GRAPH_INTERNAL_H
#define SPARSE_GRAPH_INTERNAL_H

/**
 * @file sparse_graph_internal.h
 * @brief Internal graph representation + multilevel partitioner for
 *        Sprint 22 nested dissection.
 *
 * The graph used by the partitioner / nested-dissection driver is a
 * pure adjacency-only view of the symmetric pattern of A + A^T.  It
 * deliberately drops everything Sprint 11-21 has been carrying on
 * the linked-list `SparseMatrix`: no values, no per-node mutex
 * bookkeeping, no slab pool — just CSR-like xadj/adjncy plus
 * optional vertex / edge weights for the multilevel hierarchy.
 *
 * Not part of the public API.  Sprint 22 Day 1 establishes the
 * representation; Days 2-5 fill in coarsening / partition / refinement /
 * separator extraction; Days 6-9 build nested dissection on top.
 */

#include "sparse_matrix.h"
#include "sparse_types.h"

#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_graph_t — CSR-style adjacency with optional weights.
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Representation:
 *   - n vertices, indexed 0..n-1.
 *   - xadj[0..n] is the CSR pointer array; vertex i's neighbours
 *     live at adjncy[xadj[i] .. xadj[i+1] - 1].
 *   - adjncy is symmetric (j ∈ adj(i)  ⇔  i ∈ adj(j)) and contains
 *     no self-loops.
 *
 * Optional weights (multilevel hierarchy uses them; the root graph
 * leaves them NULL and the partitioner treats unweighted as
 * uniform-weight = 1):
 *   - vwgt[i] is vertex i's weight (the count of fine-graph
 *     vertices that collapsed into i during coarsening).
 *   - ewgt[k] is the weight of the k-th edge in adjncy (the sum of
 *     fine-graph edge weights aggregated under matching).
 *
 * Memory ownership: the four arrays are owned by the graph struct.
 * `sparse_graph_free` releases them all (NULL-safe per array, so
 * partial-failure paths in the constructor don't leak).
 *
 * Invariants enforced by the constructor (`sparse_graph_from_sparse`
 * and the Day 2 `graph_coarsen_*` helpers):
 *   - n ≥ 0; n == 0 is legal (empty graph).
 *   - xadj[0] == 0; xadj[n] == |adjncy|.
 *   - Each xadj[i+1] - xadj[i] is the degree of vertex i.
 *   - adj lists are sorted ascending and duplicate-free.
 */
typedef struct {
    idx_t n;       /**< Number of vertices. */
    idx_t *xadj;   /**< CSR pointers, length n+1; owned. */
    idx_t *adjncy; /**< Adjacency entries, length xadj[n]; owned. */
    idx_t *vwgt;   /**< Optional vertex weights, length n; NULL = uniform. */
    idx_t *ewgt;   /**< Optional edge weights, length xadj[n]; NULL = uniform. */
} sparse_graph_t;

/**
 * @brief Build the symmetric adjacency graph of A.
 *
 * Wraps the existing internal `sparse_build_adj` helper so the
 * partitioner gets a CSR adjacency without re-implementing the
 * symmetrise-plus-dedup pass.  vwgt and ewgt are left NULL —
 * coarsening (Sprint 22 Day 2) will populate them on derived graphs.
 *
 * @param A    Input matrix.  Must be square (n × n); the partitioner
 *             assumes a symmetric structure for the recursive ND.
 * @param G    Output graph.  Caller-owned struct slot; on success the
 *             struct's fields are populated and must be released via
 *             `sparse_graph_free(G)`.  On failure G's fields are
 *             zero-initialised so a defensive `sparse_graph_free`
 *             is still safe.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or G is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_ALLOC if any internal allocation fails.
 */
sparse_err_t sparse_graph_from_sparse(const SparseMatrix *A, sparse_graph_t *G);

/**
 * @brief Release the four arrays owned by a graph and zero its
 *        scalar fields.  Safe on a zero-initialised struct (no-op).
 *        Safe if any of the four owned pointers is NULL.  After
 *        return the struct is in the empty state and may be reused.
 */
void sparse_graph_free(sparse_graph_t *G);

/**
 * @brief Build a vertex-induced subgraph.
 *
 * Sprint 22 Day 6's recursive nested dissection calls this once per
 * subdivision: given the parent graph and the set of vertex indices
 * (in the parent's numbering) that belong to one side of the
 * vertex-separator partition, the helper produces a child graph
 * containing only those vertices and the edges among them.
 *
 * The child uses 0..|vertex_set|-1 numbering; the caller passes a
 * `vertex_id_map_out` array (length |vertex_set|) so it can recover
 * each child vertex's parent id when assembling the final
 * permutation.
 *
 * @param parent          Parent graph.
 * @param vertex_set      Sorted ascending parent-vertex indices to keep.
 * @param k               Number of vertices in `vertex_set`.
 * @param child           Output child graph; same ownership / failure
 *                        semantics as `sparse_graph_from_sparse`.
 * @param vertex_id_map_out  Output: caller-allocated array of length k;
 *                        on return, `vertex_id_map_out[i] == vertex_set[i]`
 *                        (parent id of child vertex i).  May be NULL if
 *                        the caller doesn't need the map (rare — the
 *                        recursive ND always does).
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any required argument is NULL.
 * @return SPARSE_ERR_BADARG if `vertex_set` contains an out-of-range
 *         index (≥ parent->n) or duplicates.
 * @return SPARSE_ERR_ALLOC on allocation failure.
 */
sparse_err_t sparse_graph_subgraph(const sparse_graph_t *parent, const idx_t *vertex_set, idx_t k,
                                   sparse_graph_t *child, idx_t *vertex_id_map_out);

/* ═══════════════════════════════════════════════════════════════════════
 * Multilevel coarsening (Sprint 22 Day 2).
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Heavy-edge matching (Karypis & Kumar 1998 §4): walk vertices in a
 * deterministic-pseudorandom order driven by `seed`; for each
 * unmatched vertex pick the unmatched neighbour connected by the
 * heaviest edge; collapse the pair into a single coarse vertex with
 * summed weight.  Vertices with no unmatched neighbour become
 * coarse singletons.
 *
 * Heavy-edge is preferred over random matching because it preserves
 * spectral structure — the heavier the edge, the more important the
 * connection it represents in the original problem.  At the
 * partition stage (Sprint 22 Day 3 / Day 4) the resulting coarsest
 * graph then captures enough of the original graph's "shape" that a
 * brute-force minimum-cut bisection on the coarsest level translates
 * into a near-optimal cut after uncoarsening + FM refinement.
 *
 * **Weight aggregation invariants** (asserted by Day 2 tests):
 *   - sum(vwgt_coarse) == sum(vwgt_fine)
 *     (treating absent vwgt as uniform = 1).
 *   - For each coarse edge (c1, c2) with c1 != c2:
 *     ewgt_coarse[c1, c2] = sum of ewgt_fine[i, j] over fine edges
 *     (i, j) with cmap[i] == c1 && cmap[j] == c2 (treating absent
 *     ewgt as uniform = 1).  Parallel fine edges are merged via a
 *     sort-then-scan pass per coarse vertex.
 *   - Self-loops produced by collapsing matched pairs are dropped
 *     (the pair's connecting edge no longer exists in the coarse
 *     graph).
 */

/**
 * @brief One-step heavy-edge-matching coarsening.
 *
 * @param fine        Input graph (n vertices, ≥ 0).
 * @param seed        PRNG seed for the vertex traversal order.  Same
 *                    `(fine, seed)` pair always produces the same
 *                    coarse graph — the determinism contract relied
 *                    on by `sparse_graph_partition` so that nested
 *                    dissection's recursive calls produce stable
 *                    permutations across runs.
 * @param coarse_out  Output graph; pre-cleared on every error path
 *                    (matches `sparse_graph_from_sparse`'s contract).
 *                    On success, owns its own xadj / adjncy / vwgt /
 *                    ewgt arrays — release with `sparse_graph_free`.
 * @param cmap_out    Caller-allocated array of length `fine->n`; on
 *                    success `cmap_out[i]` is the coarse-vertex index
 *                    (in [0, coarse_out->n)) for fine vertex i.  On
 *                    failure, contents are unspecified.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any required pointer is NULL.
 * @return SPARSE_ERR_ALLOC on allocation failure.
 *
 * @note When `fine->n == 0` the coarse graph is also empty (n == 0,
 *       xadj allocated to length 1) and `cmap_out` is untouched.
 */
sparse_err_t graph_coarsen_heavy_edge_matching(const sparse_graph_t *fine, uint32_t seed,
                                               sparse_graph_t *coarse_out, idx_t *cmap_out);

/**
 * @brief Multilevel coarsening hierarchy.
 *
 * Owns the chain of coarsened graphs derived from a caller-supplied
 * root.  The root itself is NOT held by the hierarchy — the caller
 * keeps it across the lifetime of the hierarchy and uses `cmaps[0]`
 * to translate root-vertex indices into the first coarse level.
 *
 * Layout:
 *   - `coarse[0]`  = first coarsening of the caller's root.
 *   - `coarse[i]`  = coarsening of `coarse[i - 1]` for i ≥ 1.
 *   - `cmaps[0]`   = root → coarse[0]   (length = root->n)
 *   - `cmaps[i]`   = coarse[i-1] → coarse[i]   (length = coarse[i-1].n)
 *
 * `nlevels == 0` is legal — it means coarsening was useless on the
 * input (e.g. an isolated-vertex graph where the matching produces
 * only singletons).  In that case both `coarse` and `cmaps` are NULL
 * and the caller can fall back to single-level partitioning.
 */
typedef struct {
    int nlevels;            /**< Number of coarsened levels held. */
    sparse_graph_t *coarse; /**< coarse[0..nlevels-1], owned. */
    idx_t **cmaps;          /**< cmaps[0..nlevels-1], owned. */
} sparse_graph_hierarchy_t;

/**
 * @brief Build a multilevel coarsening hierarchy from a root graph.
 *
 * Coarsens repeatedly until one of three stop conditions fires:
 *   1. `n_coarse <= MAX(20, root->n / 100)` — coarsest level is small
 *      enough for the brute-force / GGGP bisection of Day 3.
 *   2. `n_coarse > 0.9 * n_fine` — a coarsening pass made too little
 *      progress; further coarsening would just churn without
 *      shrinking the problem.
 *   3. `nlevels >= log2(root->n) + 5` — defensive ceiling.
 *
 * The seed is forwarded to each per-level coarsening with a level-
 * dependent perturbation (level-0 uses `seed`, level-1 uses
 * `seed + 1`, etc.) so the same `(root, seed)` pair always produces
 * the same hierarchy.
 *
 * @param root  Input graph (caller-owned, kept alive across the
 *              lifetime of the hierarchy).
 * @param seed  PRNG seed.
 * @param h     Output hierarchy; pre-cleared on every error path.
 *              On success the caller releases it with
 *              `sparse_graph_hierarchy_free`.
 *
 * @return SPARSE_OK on success (including the `nlevels == 0` no-
 *         progress case).
 * @return SPARSE_ERR_NULL if `root` or `h` is NULL.
 * @return SPARSE_ERR_ALLOC on allocation failure.
 */
sparse_err_t sparse_graph_hierarchy_build(const sparse_graph_t *root, uint32_t seed,
                                          sparse_graph_hierarchy_t *h);

/**
 * @brief Release every level + cmap owned by a hierarchy and zero
 *        the struct.  Safe on a zero-initialised struct (no-op).
 *        Does NOT touch the root graph the hierarchy was built from.
 */
void sparse_graph_hierarchy_free(sparse_graph_hierarchy_t *h);

/* ═══════════════════════════════════════════════════════════════════════
 * Coarsest-graph bisection + FM refinement (Sprint 22 Day 3).
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Day 4's uncoarsening pipeline composes these two routines: bisect
 * the coarsest hierarchy level into an initial 2-way partition, then
 * project through the hierarchy with FM at every level until the
 * partition lands at the original (root) graph.  Vertex-separator
 * extraction (Day 4) consumes the final 2-way partition.
 *
 * Both routines operate on edge-separator partitions: `part[i] ∈
 * {0, 1}` means vertex i is on the "left" or "right" side; the cut
 * is the sum of edge weights connecting the two sides.  Day 4's
 * `graph_edge_separator_to_vertex_separator` converts the final
 * edge separator into the 3-way `{0, 1, 2}` form
 * `sparse_graph_partition` returns.
 */

/**
 * @brief Bisect a small graph into a balanced 2-way partition.
 *
 * Routes through one of two strategies based on `G->n`:
 *
 *   - **n ≤ 20** — brute-force minimum-cut enumeration.  Vertex 0 is
 *     fixed to side 0 (the side-swapped mirror has the same cut, so
 *     this halves the search) and the remaining `2^(n-1)` patterns
 *     are scanned; the lowest-cut pattern that satisfies vertex-
 *     weight balance (|w0 - w1| ≤ max_vwgt) wins.
 *
 *   - **20 < n ≤ 40** — Greedy Graph-Growing Partition (METIS §3):
 *     find a peripheral vertex via two BFS passes, BFS-grow side 0
 *     from it until half the vertex weight is consumed, leave the
 *     rest on side 1.  Day 4's per-level FM refines the resulting
 *     (often imbalanced) partition.
 *
 * Larger inputs return `SPARSE_ERR_BADARG` — the multilevel
 * coarsening (`sparse_graph_hierarchy_build`) is contracted to drive
 * `n` down to MAX(20, n_orig / 100) before the partitioner gets here.
 *
 * @param G        Input graph.  Must satisfy `G->n ≤ 40`.
 * @param part_out Caller-allocated array of length `G->n`; written
 *                 on success with `part_out[i] ∈ {0, 1}`.  On failure
 *                 the contents are unspecified.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if `G` or `part_out` is NULL.
 * @return SPARSE_ERR_BADARG if `G->n > 40`.
 * @return SPARSE_ERR_ALLOC on allocation failure.
 */
sparse_err_t graph_bisect_coarsest(const sparse_graph_t *G, idx_t *part_out);

/**
 * @brief Single-pass Fiduccia-Mattheyses refinement of a 2-way partition.
 *
 * Walks unlocked vertices in decreasing-gain order, where
 * `gain(v) = (sum of edge weights to vertices on the other side)
 *          − (sum of edge weights to vertices on the same side)`,
 * and `cut_after_move(v) = cut_before_move - gain(v)`.  Each move
 * locks the vertex for the rest of the pass and updates the gains
 * of its unlocked neighbours.  After every unlocked vertex has been
 * processed (or the balance constraint blocks every remaining
 * candidate), the partition is rolled back to the lowest-cut state
 * seen during the walk — this lets FM escape shallow local minima
 * by accepting transient cut increases that subsequent moves
 * recover.
 *
 * Balance constraint: `|w0 - w1|` may grow up to
 * `max(initial_imbalance, total_vwgt / 20) + max_vwgt`.  This permits
 * restoring balance when the input partition is severely skewed
 * (the `total_vwgt / 20` slack), accommodates a single-vertex move
 * that crosses the centre (the `max_vwgt` slack), and never makes
 * balance worse than the input.
 *
 * @param G       Input graph.
 * @param part_io Length-`G->n` array.  In: initial partition with
 *                `part_io[i] ∈ {0, 1}`.  Out: refined partition
 *                (or unchanged if the pass found no improvement).
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if `G` or `part_io` is NULL.
 * @return SPARSE_ERR_ALLOC on allocation failure.
 */
sparse_err_t graph_refine_fm(const sparse_graph_t *G, idx_t *part_io);

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_graph_partition — multilevel vertex-separator partitioner.
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The Sprint 22 entry point that nested dissection composes against.
 * Three-phase pipeline (Karypis & Kumar 1998; details in
 * `src/sparse_graph.c`'s file-header design block):
 *
 *   1. Coarsen the input graph via heavy-edge matching until the
 *      coarsest level is small enough for an exact bisection
 *      (Sprint 22 Day 2).
 *   2. Bisect the coarsest graph (brute-force min-cut for n ≤ 20,
 *      greedy graph-growing partition otherwise) and refine the
 *      partition with Fiduccia-Mattheyses (Sprint 22 Day 3).
 *   3. Project the partition back through the hierarchy with a per-
 *      level FM refinement, then convert the final edge separator to
 *      a vertex separator on the smaller side of the cut
 *      (Sprint 22 Day 4).
 *
 * On return `part_out[i] ∈ {0, 1, 2}`:
 *   - 0 = "left" partition vertex
 *   - 1 = "right" partition vertex
 *   - 2 = vertex separator
 * and `*sep_out` holds the count of separator vertices.
 *
 * @param G        Input graph.  Modified only via temporary
 *                 internal copies (the multilevel hierarchy lives in
 *                 the partitioner's scratch space, freed before
 *                 return).
 * @param part_out Caller-allocated array of length G->n; written on
 *                 success.  On failure the contents are unspecified.
 * @param sep_out  Output count of separator vertices (the count of
 *                 i such that part_out[i] == 2).  May be NULL if the
 *                 caller doesn't need it.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if G or part_out is NULL.
 * @return SPARSE_ERR_BADARG if G->n < 0 or G is structurally
 *         malformed (xadj[n] != |adjncy|).
 * @return SPARSE_ERR_ALLOC on allocation failure.
 *
 * **Day 1 stub.** Returns SPARSE_ERR_BADARG.  Days 2-4 replace the
 * body with the multilevel pipeline.
 */
sparse_err_t sparse_graph_partition(const sparse_graph_t *G, idx_t *part_out, idx_t *sep_out);

#endif /* SPARSE_GRAPH_INTERNAL_H */
