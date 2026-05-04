/*
 * sparse_reorder_nd.c — Nested Dissection (Sprint 22 Day 6).
 *
 * George (1973) showed that recursively bisecting a symmetric matrix's
 * adjacency graph and ordering separator vertices last produces an
 * elimination ordering whose fill grows as O(n^{(d-1)/d}) on regular
 * d-dimensional meshes — vs O(n) for AMD on the same input.  This
 * file ships the recursive driver: pull a 3-way `{0, 1, 2}` partition
 * from `sparse_graph_partition` (Sprint 22 Days 1-5), recurse on the
 * two interior subgraphs, then append the separator vertices to the
 * permutation.  The `SPARSE_REORDER_ND` enum value is wired through
 * every factorization's analysis dispatch (Sprint 22 Day 8), so
 * callers can opt into ND either via `sparse_analyze` with
 * `SPARSE_REORDER_ND` or by calling `sparse_reorder_nd` directly the
 * same way they call `sparse_reorder_amd` / `sparse_reorder_rcm`.
 *
 * **Base case (`n ≤ ND_BASE_THRESHOLD`).**  At small subgraph sizes
 * the recursion's separator-last benefit doesn't outweigh the
 * partitioner overhead, so the driver falls back to the natural
 * (identity-on-subgraph) ordering — a deliberately simple Day 6
 * starting point.  Sprint 22 Day 9 retuned the threshold against
 * the SuiteSparse corpus.  Splicing the new quotient-graph AMD
 * into the leaves to lift fill quality is deferred to Sprint 23
 * (see `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 23, item:
 * "ND leaf ordering: call quotient-graph AMD on each leaf-sized
 * subgraph") — closing this is what the Day 14 plan target
 * (ND ≥ 2× AMD nnz(L) reduction on Pres_Poisson) needs.
 *
 * **Permutation contract.**  `perm[new_i] = old_i` (matches the
 * existing AMD / RCM / COLAMD contract from `include/sparse_reorder.h`).
 * The recursive helper takes a `vertex_id_map` argument that
 * translates each subgraph-local index into its root-graph index, so
 * children never need to consult the parent graph during ordering.
 */

#include "sparse_graph_internal.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_reorder_amd_qg_internal.h"
#include "sparse_reorder_nd_internal.h"
#include "sparse_types.h"

#include <stdlib.h>
#include <string.h>

/* Base-case threshold: the recursion stops here and the subgraph's
 * vertices land in the permutation in their natural (subgraph-local)
 * order.  Default 32 from the Day 9 sweep: fill on bcsstk14 (n=1806)
 * and Pres_Poisson (n=14822) is minimised here within 0.1 % of any
 * threshold in {4, 8, 16, 32}, and 32 is significantly faster than
 * the smaller values on Pres_Poisson (recursive partitioning cost
 * dominates beyond the leaves).  See
 * `docs/planning/EPIC_2/SPRINT_22/bench_day9_nd.txt` for the full
 * sweep data.
 *
 * Declared in `src/sparse_reorder_nd_internal.h` (benchmark-only,
 * not thread-safe, no ABI guarantee — see that header) so the Day 9
 * sweep (`benchmarks/bench_reorder.c --nd-threshold N`) can
 * override it from the command line without recompiling the
 * library, but library consumers don't see it.  The Sprint 23
 * follow-up that splices quotient-graph AMD into each leaf will
 * turn this into a real "stop recursing here, run AMD" cutover, at
 * which point the right tuning surface is an opts struct on
 * `sparse_reorder_nd` itself and this global goes away. */
idx_t sparse_reorder_nd_base_threshold = 32;

/* Append `n` vertices from a subgraph to the global permutation in
 * the order they appear in `vertex_id_map`.  Used by the
 * degenerate-partition fallback in `nd_recurse` (when the
 * partitioner returns a separator that consumes everything; the
 * leaf-AMD path now propagates allocation / AMD failures rather
 * than falling back here). */
static void nd_emit_natural(const idx_t *vertex_id_map, idx_t n, idx_t *perm, idx_t *next_pos) {
    /* The caller guarantees `perm` has space for at least `*next_pos + n`
     * entries (the recursion's invariant — each subgraph's vertices fit
     * in their slice of the root permutation).  Static analyser doesn't
     * track this cross-call invariant. */
    for (idx_t i = 0; i < n; i++)
        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound,clang-analyzer-core.uninitialized.Assign)
        perm[*next_pos + i] = vertex_id_map[i];
    *next_pos += n;
}

/* Sprint 23 Day 7: build a temporary `SparseMatrix` from a leaf
 * subgraph's CSR adjacency for `sparse_reorder_amd_qg` to consume.
 *
 * AMD reads only the *symbolic* structure — the numeric values
 * never matter — so each adjacency edge becomes a `value=1.0`
 * entry, with `value=1.0` on the diagonal too (AMD ignores
 * diagonal but `sparse_reorder_amd_qg` works on the symmetric
 * pattern of A + A^T; including the diagonal is harmless and
 * keeps the matrix structurally well-formed for downstream
 * sanity checks).
 *
 * Returns NULL on allocation failure or insert failure.  Caller
 * (the leaf base case in `nd_recurse`) propagates a NULL return
 * as `SPARSE_ERR_ALLOC` to its own caller — Sprint 23 Day 7's
 * silent natural-ordering fallback was replaced per PR #31
 * review (comment 3184299437) so allocation pressure surfaces
 * cleanly instead of silently degrading the ordering. */
static SparseMatrix *nd_subgraph_to_sparse(const sparse_graph_t *G_leaf) {
    idx_t n = G_leaf->n;
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        if (sparse_insert(A, i, i, 1.0) != SPARSE_OK)
            goto fail;
        idx_t row_start = G_leaf->xadj[i];
        idx_t row_end = G_leaf->xadj[i + 1];
        for (idx_t k = row_start; k < row_end; k++) {
            idx_t j = G_leaf->adjncy[k];
            if (sparse_insert(A, i, j, 1.0) != SPARSE_OK)
                goto fail;
        }
    }
    return A;
fail:
    sparse_free(A);
    return NULL;
}

/* The driver is genuinely recursive — each level descends two
 * subgraphs.  Recursion depth is O(log n) on regular meshes (and
 * bounded by graph size in the worst case).  Suppress clang-tidy's
 * misc-no-recursion check; iterative replacement would obscure the
 * algorithm without measurable benefit. */
// NOLINTNEXTLINE(misc-no-recursion)
static sparse_err_t nd_recurse(const sparse_graph_t *G, const idx_t *vertex_id_map, idx_t *perm,
                               idx_t *next_pos) {
    idx_t n = G->n;
    if (n == 0)
        return SPARSE_OK;

    /* Single-vertex base case (also the recursion floor for n=1
     * subgraphs the partitioner can produce after a separator
     * lift).  Avoids a redundant partition call. */
    if (n == 1) {
        perm[*next_pos] = vertex_id_map[0];
        (*next_pos)++;
        return SPARSE_OK;
    }

    /* Sprint 23 Day 7: small-subgraph base case → quotient-graph
     * AMD on the leaf subgraph, splice the per-leaf permutation
     * into the global perm[] via vertex_id_map.  Replaces Sprint
     * 22's natural-ordering base case — gives the leaves
     * minimum-degree quality without burning the partitioner's
     * coarsening cost on subgraphs too small for separator-last
     * to amortise.
     *
     * On any failure path (`nd_subgraph_to_sparse` allocation,
     * `leaf_perm` allocation, or `sparse_reorder_amd_qg` call):
     * propagate the error to the caller rather than silently
     * falling back to natural ordering.  Sprint 23 Day 7's
     * original design fell back-and-returned-OK to keep ND
     * "always-completes" against transient allocation pressure,
     * but per PR #31 review (comment 3184299437) that masked
     * real `SPARSE_ERR_ALLOC` failures from the caller and made
     * `sparse_reorder_nd` look successful while silently
     * degrading.  Propagating gives callers the option to
     * distinguish a successful reorder from a degraded fallback
     * (e.g., they can retry or surface the error). */
    if (n <= sparse_reorder_nd_base_threshold) {
        SparseMatrix *A_leaf = nd_subgraph_to_sparse(G);
        if (!A_leaf)
            return SPARSE_ERR_ALLOC;
        idx_t *leaf_perm = malloc((size_t)n * sizeof(idx_t));
        if (!leaf_perm) {
            sparse_free(A_leaf);
            return SPARSE_ERR_ALLOC;
        }
        sparse_err_t leaf_rc = sparse_reorder_amd_qg(A_leaf, leaf_perm);
        if (leaf_rc != SPARSE_OK) {
            free(leaf_perm);
            sparse_free(A_leaf);
            return leaf_rc;
        }
        /* Splice the leaf-local permutation through vertex_id_map
         * to write global root-graph IDs into perm[]. */
        for (idx_t i = 0; i < n; i++) {
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound,clang-analyzer-core.uninitialized.Assign)
            perm[*next_pos + i] = vertex_id_map[leaf_perm[i]];
        }
        *next_pos += n;
        free(leaf_perm);
        sparse_free(A_leaf);
        return SPARSE_OK;
    }

    /* Partition: 3-way label part[i] ∈ {0, 1, 2}. */
    idx_t *part = malloc((size_t)n * sizeof(idx_t));
    if (!part)
        return SPARSE_ERR_ALLOC;
    idx_t sep_count = 0;
    sparse_err_t rc = sparse_graph_partition(G, part, &sep_count);
    if (rc != SPARSE_OK) {
        free(part);
        return rc;
    }

    /* Tally the three sides. */
    idx_t n0 = 0;
    idx_t n1 = 0;
    for (idx_t i = 0; i < n; i++) {
        if (part[i] == 0)
            n0++;
        else if (part[i] == 1)
            n1++;
    }

    /* Degenerate-partition guard: if either interior side is empty,
     * the recursion would re-enter on the same graph.  Drop to
     * natural ordering on the whole subgraph. */
    if (n0 == 0 || n1 == 0) {
        free(part);
        nd_emit_natural(vertex_id_map, n, perm, next_pos);
        return SPARSE_OK;
    }

    /* Collect the two interior vertex sets in ascending index order
     * so `sparse_graph_subgraph`'s "sorted vertex_set" precondition
     * is met for free. */
    idx_t *vs0 = malloc((size_t)n0 * sizeof(idx_t));
    idx_t *vs1 = malloc((size_t)n1 * sizeof(idx_t));
    if (!vs0 || !vs1) {
        free(part);
        free(vs0);
        free(vs1);
        return SPARSE_ERR_ALLOC;
    }
    {
        idx_t i0 = 0;
        idx_t i1 = 0;
        for (idx_t i = 0; i < n; i++) {
            if (part[i] == 0)
                vs0[i0++] = i;
            else if (part[i] == 1)
                vs1[i1++] = i;
        }
    }

    /* Recurse on side 0. */
    {
        sparse_graph_t G0 = {0};
        idx_t *map0 = malloc((size_t)n0 * sizeof(idx_t));
        if (!map0) {
            free(part);
            free(vs0);
            free(vs1);
            return SPARSE_ERR_ALLOC;
        }
        rc = sparse_graph_subgraph(G, vs0, n0, &G0, NULL);
        if (rc != SPARSE_OK) {
            free(map0);
            free(part);
            free(vs0);
            free(vs1);
            return rc;
        }
        /* `vs0[i] ∈ [0, n)` by construction (we built it from `part`),
         * and `vertex_id_map` has length `n`.  The analyser doesn't
         * track the relationship. */
        for (idx_t i = 0; i < n0; i++)
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound,clang-analyzer-core.uninitialized.Assign)
            map0[i] = vertex_id_map[vs0[i]];
        rc = nd_recurse(&G0, map0, perm, next_pos);
        sparse_graph_free(&G0);
        free(map0);
        if (rc != SPARSE_OK) {
            free(part);
            free(vs0);
            free(vs1);
            return rc;
        }
    }

    /* Recurse on side 1. */
    {
        sparse_graph_t G1 = {0};
        idx_t *map1 = malloc((size_t)n1 * sizeof(idx_t));
        if (!map1) {
            free(part);
            free(vs0);
            free(vs1);
            return SPARSE_ERR_ALLOC;
        }
        rc = sparse_graph_subgraph(G, vs1, n1, &G1, NULL);
        if (rc != SPARSE_OK) {
            free(map1);
            free(part);
            free(vs0);
            free(vs1);
            return rc;
        }
        for (idx_t i = 0; i < n1; i++)
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound,clang-analyzer-core.uninitialized.Assign)
            map1[i] = vertex_id_map[vs1[i]];
        rc = nd_recurse(&G1, map1, perm, next_pos);
        sparse_graph_free(&G1);
        free(map1);
        if (rc != SPARSE_OK) {
            free(part);
            free(vs0);
            free(vs1);
            return rc;
        }
    }

    /* Separator last — the rule that makes ND fill-reducing. */
    for (idx_t i = 0; i < n; i++) {
        if (part[i] == 2) {
            // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
            perm[*next_pos] = vertex_id_map[i];
            (*next_pos)++;
        }
    }

    free(part);
    free(vs0);
    free(vs1);
    return SPARSE_OK;
}

sparse_err_t sparse_reorder_nd(const SparseMatrix *A, idx_t *perm) {
    if (!A || !perm)
        return SPARSE_ERR_NULL;
    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;

    sparse_graph_t G = {0};
    sparse_err_t rc = sparse_graph_from_sparse(A, &G);
    if (rc != SPARSE_OK)
        return rc;

    if (G.n == 0) {
        sparse_graph_free(&G);
        return SPARSE_OK;
    }

    /* Identity vertex_id_map at the root: subgraph-local index i
     * maps to root index i.  Recursive children compose this map
     * with their slice through the parent's vertex set. */
    idx_t *root_map = malloc((size_t)G.n * sizeof(idx_t));
    if (!root_map) {
        sparse_graph_free(&G);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t i = 0; i < G.n; i++)
        root_map[i] = i;

    idx_t next_pos = 0;
    rc = nd_recurse(&G, root_map, perm, &next_pos);

    free(root_map);
    sparse_graph_free(&G);
    return rc;
}
