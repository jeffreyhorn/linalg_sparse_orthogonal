/* Sprint 25 Day 11: feature-test macro to expose `clock_gettime` /
 * `CLOCK_MONOTONIC` for the SPARSE_ND_PROFILE instrumentation
 * below.  Same pattern as Sprint 24 Day 1's SPARSE_QG_PROFILE in
 * `src/sparse_reorder_amd_qg.c`. */
#if !defined(_WIN32) && (!defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 199309L)
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define _POSIX_C_SOURCE 199309L
#endif
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

#include <stdio.h> /* Sprint 25 Day 11: SPARSE_ND_PROFILE stderr trace. */
#include <stdlib.h>
#include <string.h>
#include <time.h> /* Sprint 25 Day 11: SPARSE_ND_PROFILE wall-clock instrumentation. */

/* Sprint 25 Day 11: SPARSE_ND_PROFILE per-phase wall-clock
 * instrumentation.  File-static accumulators (nanoseconds) — `nd_recurse`
 * is recursive and not thread-safe, so accumulating across calls in
 * file-statics is fine and avoids threading a profile struct through
 * every recursion level.  `nd_prof_enabled` is set by `sparse_reorder_nd`
 * once (after parsing the env var) and stays set for the duration of
 * the call; emitted to stderr at the end of the top-level
 * `sparse_reorder_nd` call.  Production overhead is one branch per
 * timed call when `nd_prof_enabled == 0`.  See
 * docs/planning/EPIC_2/SPRINT_25/PLAN.md Day 11 task 1 + the analogous
 * SPARSE_QG_PROFILE pattern in src/sparse_reorder_amd_qg.c. */
static int nd_prof_enabled = 0;
static long long nd_prof_partition_ns = 0;     /* sparse_graph_partition cumulative */
static long long nd_prof_subgraph_ns = 0;      /* sparse_graph_subgraph cumulative */
static long long nd_prof_leaf_amd_ns = 0;      /* sparse_reorder_amd_qg leaf splice cumulative */
static long long nd_prof_leaf_subgraph_ns = 0; /* nd_subgraph_to_sparse cumulative */
static long long nd_prof_emit_natural_ns = 0;  /* nd_emit_natural cumulative */
static long long nd_prof_graph_build_ns = 0;   /* sparse_graph_from_sparse one-shot */
static idx_t nd_prof_partition_calls = 0;
static idx_t nd_prof_leaf_amd_calls = 0;
static idx_t nd_prof_emit_natural_calls = 0;

static long long nd_prof_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

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
        long long lsub_t0 = nd_prof_enabled ? nd_prof_now_ns() : 0;
        SparseMatrix *A_leaf = nd_subgraph_to_sparse(G);
        if (nd_prof_enabled)
            nd_prof_leaf_subgraph_ns += nd_prof_now_ns() - lsub_t0;
        if (!A_leaf)
            return SPARSE_ERR_ALLOC;
        idx_t *leaf_perm = malloc((size_t)n * sizeof(idx_t));
        if (!leaf_perm) {
            sparse_free(A_leaf);
            return SPARSE_ERR_ALLOC;
        }
        long long lamd_t0 = nd_prof_enabled ? nd_prof_now_ns() : 0;
        sparse_err_t leaf_rc = sparse_reorder_amd_qg(A_leaf, leaf_perm);
        if (nd_prof_enabled) {
            nd_prof_leaf_amd_ns += nd_prof_now_ns() - lamd_t0;
            nd_prof_leaf_amd_calls++;
        }
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
    long long part_t0 = nd_prof_enabled ? nd_prof_now_ns() : 0;
    sparse_err_t rc = sparse_graph_partition(G, part, &sep_count);
    if (nd_prof_enabled) {
        nd_prof_partition_ns += nd_prof_now_ns() - part_t0;
        nd_prof_partition_calls++;
    }
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
        long long en_t0 = nd_prof_enabled ? nd_prof_now_ns() : 0;
        nd_emit_natural(vertex_id_map, n, perm, next_pos);
        if (nd_prof_enabled) {
            nd_prof_emit_natural_ns += nd_prof_now_ns() - en_t0;
            nd_prof_emit_natural_calls++;
        }
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
        long long sg0_t0 = nd_prof_enabled ? nd_prof_now_ns() : 0;
        rc = sparse_graph_subgraph(G, vs0, n0, &G0, NULL);
        if (nd_prof_enabled)
            nd_prof_subgraph_ns += nd_prof_now_ns() - sg0_t0;
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
        long long sg1_t0 = nd_prof_enabled ? nd_prof_now_ns() : 0;
        rc = sparse_graph_subgraph(G, vs1, n1, &G1, NULL);
        if (nd_prof_enabled)
            nd_prof_subgraph_ns += nd_prof_now_ns() - sg1_t0;
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

    /* Sprint 25 Day 11: SPARSE_ND_PROFILE per-phase wall-clock
     * breakdown.  Reset accumulators on every entry so consecutive
     * calls produce independent measurements (file-static state is
     * fine since `sparse_reorder_nd` isn't thread-safe — same caveat
     * as `sparse_reorder_nd_base_threshold` per the header). */
    nd_prof_enabled = (getenv("SPARSE_ND_PROFILE") != NULL);
    long long prof_t0 = nd_prof_enabled ? nd_prof_now_ns() : 0;
    if (nd_prof_enabled) {
        nd_prof_partition_ns = 0;
        nd_prof_subgraph_ns = 0;
        nd_prof_leaf_amd_ns = 0;
        nd_prof_leaf_subgraph_ns = 0;
        nd_prof_emit_natural_ns = 0;
        nd_prof_graph_build_ns = 0;
        nd_prof_partition_calls = 0;
        nd_prof_leaf_amd_calls = 0;
        nd_prof_emit_natural_calls = 0;
    }

    sparse_graph_t G = {0};
    long long gb_t0 = nd_prof_enabled ? nd_prof_now_ns() : 0;
    sparse_err_t rc = sparse_graph_from_sparse(A, &G);
    if (nd_prof_enabled)
        nd_prof_graph_build_ns = nd_prof_now_ns() - gb_t0;
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

    if (nd_prof_enabled) {
        long long total_ns = nd_prof_now_ns() - prof_t0;
        long long sum_phase_ns = nd_prof_graph_build_ns + nd_prof_partition_ns +
                                 nd_prof_subgraph_ns + nd_prof_leaf_amd_ns +
                                 nd_prof_leaf_subgraph_ns + nd_prof_emit_natural_ns;
        long long other_ns = total_ns > sum_phase_ns ? total_ns - sum_phase_ns : 0;
        fprintf(stderr,
                "nd-profile n=%d total=%.3fms\n"
                "  graph_build  = %.3fms\n"
                "  partition    = %.3fms (%d calls)\n"
                "  subgraph     = %.3fms\n"
                "  leaf_amd     = %.3fms (%d calls)\n"
                "  leaf_subgraph= %.3fms\n"
                "  emit_natural = %.3fms (%d calls)\n"
                "  other        = %.3fms\n",
                (int)sparse_rows(A), (double)total_ns / 1.0e6,
                (double)nd_prof_graph_build_ns / 1.0e6, (double)nd_prof_partition_ns / 1.0e6,
                (int)nd_prof_partition_calls, (double)nd_prof_subgraph_ns / 1.0e6,
                (double)nd_prof_leaf_amd_ns / 1.0e6, (int)nd_prof_leaf_amd_calls,
                (double)nd_prof_leaf_subgraph_ns / 1.0e6, (double)nd_prof_emit_natural_ns / 1.0e6,
                (int)nd_prof_emit_natural_calls, (double)other_ns / 1.0e6);
    }
    return rc;
}
