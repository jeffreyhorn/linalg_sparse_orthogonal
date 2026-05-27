/*
 * sparse_graph_core.c — graph construction / ownership core extracted
 * from Sprint 43 Day 5's first graph-subsystem decomposition batch.
 *
 * This file owns:
 *   - sparse_graph_from_sparse(...)
 *   - sparse_graph_free(...)
 *   - sparse_graph_subgraph(...)
 *
 * It intentionally stays narrow: no coarsening, no coarse bisection,
 * no FM refinement, and no top-level partition orchestration.
 */

#include "sparse_graph_internal.h"
#include "sparse_matrix_internal.h"

#include <stdlib.h>

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
 * the coarsener will populate weights on derived graphs as it
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
 * sparse_graph_subgraph — build a vertex-induced subgraph.
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Two passes over the parent's adjacency: pass 1 tallies per-vertex
 * degrees so we can prefix-sum `xadj`; pass 2 fills `adjncy` (and
 * `ewgt`, when the parent has them).  The parent → child index map is
 * held in a scratch array indexed by parent-vertex id (length
 * `parent->n`).
 *
 * `vertex_set` must be sorted ascending and duplicate-free — the
 * recursive ND driver constructs it by walking partition labels in
 * vertex order, which trivially produces a sorted set.  The resulting
 * child adjacency lists inherit the parent's CSR sort invariant: each
 * list is in ascending neighbour-id order.
 */
sparse_err_t sparse_graph_subgraph(const sparse_graph_t *parent, const idx_t *vertex_set, idx_t k,
                                   sparse_graph_t *child, idx_t *vertex_id_map_out) {
    if (!parent || !child)
        return SPARSE_ERR_NULL;

    /* Pre-clear child so every error path leaves it empty. */
    child->n = 0;
    child->xadj = NULL;
    child->adjncy = NULL;
    child->vwgt = NULL;
    child->ewgt = NULL;

    if (k > 0 && !vertex_set)
        return SPARSE_ERR_NULL;

    if (k == 0) {
        child->xadj = malloc(sizeof(idx_t));
        if (!child->xadj)
            return SPARSE_ERR_ALLOC;
        child->xadj[0] = 0;
        return SPARSE_OK;
    }

    /* Validate vertex_set: sorted ascending, in [0, parent->n), no dupes. */
    idx_t prev_vid = -1;
    for (idx_t i = 0; i < k; i++) {
        idx_t v = vertex_set[i];
        if (v < 0 || v >= parent->n)
            return SPARSE_ERR_BADARG;
        if (v <= prev_vid)
            return SPARSE_ERR_BADARG;
        prev_vid = v;
    }

    /* Parent → child map: -1 for vertices not in the subset. */
    idx_t *p2c = malloc((size_t)parent->n * sizeof(idx_t));
    if (!p2c)
        return SPARSE_ERR_ALLOC;
    for (idx_t i = 0; i < parent->n; i++)
        p2c[i] = -1;
    for (idx_t i = 0; i < k; i++)
        p2c[vertex_set[i]] = i;

    /* Pass 1: count degrees, prefix-sum into child->xadj. */
    idx_t *xadj = malloc((size_t)(k + 1) * sizeof(idx_t));
    if (!xadj) {
        free(p2c);
        return SPARSE_ERR_ALLOC;
    }
    xadj[0] = 0;
    for (idx_t i = 0; i < k; i++) {
        idx_t v = vertex_set[i];
        idx_t deg = 0;
        for (idx_t pp = parent->xadj[v]; pp < parent->xadj[v + 1]; pp++) {
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            if (p2c[parent->adjncy[pp]] >= 0)
                deg++;
        }
        xadj[i + 1] = xadj[i] + deg;
    }
    idx_t total_edges = xadj[k]; // NOLINT(clang-analyzer-security.ArrayBound)

    /* Pass 2: fill adjncy + (optional) ewgt. */
    idx_t *adjncy = NULL;
    idx_t *ewgt = NULL;
    if (total_edges > 0) {
        adjncy = malloc((size_t)total_edges * sizeof(idx_t));
        if (!adjncy) {
            free(p2c);
            free(xadj);
            return SPARSE_ERR_ALLOC;
        }
        if (parent->ewgt) {
            ewgt = malloc((size_t)total_edges * sizeof(idx_t));
            if (!ewgt) {
                free(p2c);
                free(xadj);
                free(adjncy);
                return SPARSE_ERR_ALLOC;
            }
        }
    }
    idx_t pos = 0;
    for (idx_t i = 0; i < k; i++) {
        idx_t v = vertex_set[i];
        for (idx_t pp = parent->xadj[v]; pp < parent->xadj[v + 1]; pp++) {
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            idx_t cu = p2c[parent->adjncy[pp]];
            if (cu < 0)
                continue;
            /* `adjncy` is allocated iff `total_edges > 0`; the inner
             * branch only executes when at least one edge exists.
             * The static analyser conflates this with the empty-graph
             * path and reports a NULL deref / out-of-bounds. */
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound,clang-analyzer-core.NullDereference)
            adjncy[pos] = cu;
            if (ewgt)
                ewgt[pos] = parent->ewgt[pp];
            pos++;
        }
    }

    /* Optional vwgt copy. */
    idx_t *vwgt = NULL;
    if (parent->vwgt) {
        vwgt = malloc((size_t)k * sizeof(idx_t));
        if (!vwgt) {
            free(p2c);
            free(xadj);
            free(adjncy);
            free(ewgt);
            return SPARSE_ERR_ALLOC;
        }
        for (idx_t i = 0; i < k; i++)
            vwgt[i] = parent->vwgt[vertex_set[i]];
    }

    free(p2c);

    child->n = k;
    child->xadj = xadj;
    child->adjncy = adjncy;
    child->vwgt = vwgt;
    child->ewgt = ewgt;

    if (vertex_id_map_out) {
        for (idx_t i = 0; i < k; i++)
            vertex_id_map_out[i] = vertex_set[i];
    }
    return SPARSE_OK;
}
