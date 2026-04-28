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
