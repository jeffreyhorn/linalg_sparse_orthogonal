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
 * instrumentation.  Per-thread accumulators (nanoseconds) — `nd_recurse`
 * is recursive within a single `sparse_reorder_nd` call, so file-static
 * accumulators avoid threading a profile struct through every recursion
 * level.  `_Thread_local` storage class (PR #33 review fix) keeps
 * concurrent `sparse_reorder_nd` calls on different matrices race-free:
 * each thread sees its own `nd_prof_enabled` + accumulators, even when
 * profiling is disabled (the env-var read + zeroing on entry is a write
 * to file-static state that without `_Thread_local` would race).  Same
 * pattern as `last_errno` in `src/sparse_types.c`.  `nd_prof_enabled`
 * is set by `sparse_reorder_nd` once (after parsing the env var) and
 * stays set for the duration of the call; emitted to stderr at the end
 * of the top-level `sparse_reorder_nd` call.  Production overhead is
 * one branch per timed call when `nd_prof_enabled == 0`.  See
 * docs/planning/EPIC_2/SPRINT_25/PLAN.md Day 11 task 1 + the analogous
 * SPARSE_QG_PROFILE pattern in src/sparse_reorder_amd_qg.c. */
static _Thread_local int nd_prof_enabled = 0;
static _Thread_local long long nd_prof_partition_ns = 0;
static _Thread_local long long nd_prof_subgraph_ns = 0;
static _Thread_local long long nd_prof_leaf_amd_ns = 0;
static _Thread_local long long nd_prof_leaf_subgraph_ns = 0;
static _Thread_local long long nd_prof_emit_natural_ns = 0;
static _Thread_local long long nd_prof_graph_build_ns = 0;
static _Thread_local idx_t nd_prof_partition_calls = 0;
static _Thread_local idx_t nd_prof_leaf_amd_calls = 0;
static _Thread_local idx_t nd_prof_emit_natural_calls = 0;

/* Sprint 26 Day 4: per-recursion-depth partition profiling.  Sprint 25
 * Day 11's instrumentation accumulated cumulative partition time across
 * all 301 recursive calls on Pres_Poisson but didn't attribute cost by
 * depth.  Sprint 26 item 5 (FINEST FM annealing/thick-restart) needs to
 * know whether cost concentrates at the root, intermediate levels, or
 * near the base threshold to pick which sub-axis to implement.
 *
 * `MAX_ND_DEPTH = 64` covers any plausible ND recursion depth — a
 * pathological worst-case ND on a path graph of n vertices recurses
 * ~log2(n) levels; n ≤ 10^9 caps depth at ~30; 64 leaves headroom.
 * Bounds-check `depth < MAX_ND_DEPTH` in the partition wrapper; if the
 * recursion exceeds the bound, the overflow accumulates into the
 * MAX_ND_DEPTH-1 bucket (a SPARSE_ND_PROFILE warning fires once).
 * See `docs/planning/EPIC_2/SPRINT_26/PLAN.md` Day 4. */
#define MAX_ND_DEPTH 64
static _Thread_local long long nd_prof_partition_ns_per_depth[MAX_ND_DEPTH] = {0};
static _Thread_local idx_t nd_prof_partition_calls_per_depth[MAX_ND_DEPTH] = {0};
static _Thread_local int nd_prof_depth_overflow_warned = 0;

/* Monotonic-clock timestamp helper.  Returns nanoseconds since an
 * unspecified epoch.  POSIX `clock_gettime(CLOCK_MONOTONIC, ...)` on
 * non-Windows; Windows routes through C11 `timespec_get(..., TIME_UTC)`
 * (PR #33 review fix — mirrors `qg_prof_now_ns()` in
 * `src/sparse_reorder_amd_qg.c`). */
static long long nd_prof_now_ns(void) {
    struct timespec ts;
#ifdef _WIN32
    timespec_get(&ts, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* Base-case threshold: the recursion stops here and the subgraph's
 * vertices go through leaf-AMD instead of further multilevel-
 * partition recursion (Sprint 23 Day 7 introduced the leaf-AMD
 * splice; Sprint 22 used natural ordering at the base case).
 *
 * **Sprint 27 Day 3 flip: 96 → 128 (relaxed flip rule).**  Sprint
 * 26 Day 5 picked t=96 under a strict 1pp regression cap that
 * rejected t=128 by s3rmt3m3 +1.05pp (just barely past the gate).
 * Sprint 27 Day 3 re-swept t ∈ {96, 128, 192, 256} under a relaxed
 * 2pp regression cap + the new Sprint 27 Day 2 HCC + Kuu-safe
 * default coarsening; t=128 is the maximum threshold satisfying
 * the relaxed rule.  Result on Pres_Poisson: ND wall 8 826 ms →
 * 7 079 ms (-19.8 %) with nnz_L +0.5 % (well within 2pp).  Per-
 * fixture: Kuu nnz_L -1.1 % win; bcsstk14 / s3rmt3m3 within
 * +/-0.5 %; nos4 / bcsstk04 bit-stable.  t=192 fails the relaxed
 * rule by Pres_Poisson +2.0 % (right at the 2pp cap); t=256 fails
 * clearly (Pres_Poisson +3.2 %).
 *
 * **Prior history (preserved for traceability).**  Sprint 22 Day 9's
 * original sweep set the default at 32 against natural-ordering
 * leaves; Sprint 23 Day 7 swapped natural for leaf-AMD without
 * re-sweeping; Sprint 25 Day 11's per-phase profile measured
 * `nd_emit_natural` firing 32 times at ~165 ms each on Pres_Poisson;
 * Sprint 26 Day 4's per-recursion-depth profile showed cost
 * concentrating at depths 6-9 (88 % of partition cost on 169
 * small-subgraph calls).  The 32 → 96 flip on Day 5 of Sprint 26
 * cut Pres_Poisson ND wall 38.1 s → 12.2 s (-67.9 %); Sprint 27
 * Day 2 HCC default added another -28 % (8.8 s); Day 3's t=128 flip
 * adds another -19.8 % (7.1 s).  Cumulative wall improvement vs
 * the Sprint 25 baseline (t=32 + HEM) is roughly -81 %.
 *
 * Per-fixture-class advisory: bimodal-degree solid-mechanics SPDs
 * (Kuu's CV=0.425 class) benefit monotonically from larger t —
 * t=256 produces -6.9 % nnz_L on Kuu vs t=96.  Workloads that look
 * more like Kuu than Pres_Poisson can opt in via
 * `bench_reorder --nd-threshold 256` or programmatic
 * `sparse_reorder_nd_base_threshold = 256` per the
 * `sparse_reorder_nd_internal.h` exposure contract.  Default
 * stays at t=128 because Pres_Poisson is the headline fixture and
 * its fill-quality regress at t > 128 fails the corpus flip rule.
 * See `docs/planning/EPIC_2/SPRINT_27/nd_base_threshold_decision.md`
 * for the full sweep matrix + flip-rule application.
 *
 * Declared in `src/sparse_reorder_nd_internal.h` (benchmark-only,
 * not thread-safe, no ABI guarantee — see that header) so the
 * `benchmarks/bench_reorder.c --nd-threshold N` flag can override
 * it from the command line without recompiling the library, but
 * library consumers don't see it. */
idx_t sparse_reorder_nd_base_threshold = 128;

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

/* Sprint 27 Days 7-9: SPARSE_ND_ROOT_BISECT env-var parser.
 *
 * Day 7 landed the parser + dispatch skeleton; Days 8-9 wired the
 * actual root-level spectral path through `nd_recurse` at depth 0
 * (reuses Sprint 25 Day 7's `graph_bisect_coarsest_spectral`
 * Laplacian + Lanczos + Fiedler pipeline at the root level instead
 * of the coarsest level — see the dispatch site in `nd_recurse`).
 *
 * Default `multilevel` preserves Sprint 22 → Sprint 27 Day 6
 * behaviour bit-identically.  `spectral` triggers the root-level
 * path when `n <= SPARSE_ND_ROOT_BISECT_MAX_N` (default 50000).
 * Above the threshold, fall through to `multilevel` (Lanczos at
 * n > 100000 is > 30 s per Sprint 21 Day 5 scaling — not worth
 * the special path on production-scale fixtures). */
typedef enum {
    ND_ROOT_BISECT_MULTILEVEL = 0, /* Sprint 22 default — full pipeline */
    ND_ROOT_BISECT_SPECTRAL = 1,   /* Sprint 27 Day 7-9 — Fiedler at root */
} nd_root_bisect_strategy_t;

static nd_root_bisect_strategy_t parse_nd_root_bisect_strategy(void) {
    const char *env = getenv("SPARSE_ND_ROOT_BISECT");
    if (env && strcmp(env, "spectral") == 0)
        return ND_ROOT_BISECT_SPECTRAL;
    /* Default + unrecognized + "multilevel" all fall through. */
    return ND_ROOT_BISECT_MULTILEVEL;
}

static idx_t parse_nd_root_bisect_max_n(void) {
    idx_t max_n = 50000;
    const char *env = getenv("SPARSE_ND_ROOT_BISECT_MAX_N");
    if (env && *env) {
        char *endp = NULL;
        long v = strtol(env, &endp, 10);
        if (env != endp && *endp == '\0' && v >= 1 && v <= 100000000)
            max_n = (idx_t)v;
    }
    return max_n;
}

/* The driver is genuinely recursive — each level descends two
 * subgraphs.  Recursion depth is O(log n) on regular meshes (and
 * bounded by graph size in the worst case).  Suppress clang-tidy's
 * misc-no-recursion check; iterative replacement would obscure the
 * algorithm without measurable benefit. */
// NOLINTNEXTLINE(misc-no-recursion)
static sparse_err_t nd_recurse(const sparse_graph_t *G, const idx_t *vertex_id_map, idx_t *perm,
                               idx_t *next_pos, int depth) {
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
    sparse_err_t rc = SPARSE_OK;
    long long part_t0 = nd_prof_enabled ? nd_prof_now_ns() : 0;

    /* Sprint 27 Day 8: SPARSE_ND_ROOT_BISECT dispatch.  At depth 0
     * (root call), if the env var is `spectral` AND the graph is
     * within the size threshold, run Lanczos + Fiedler at the root
     * via `graph_bisect_coarsest_spectral` (Sprint 25 Day 7 helper,
     * promoted to internal-API today); convert the 2-way result to a
     * 3-way separator via `graph_edge_separator_to_vertex_separator`
     * (Sprint 22 Day 4); skip the multilevel pipeline.
     *
     * Determinism: Lanczos with fixed tol + reorthogonalization is
     * deterministic given the same Laplacian; output cuts reproduce
     * across runs.
     *
     * Lanczos failure / 60-40 imbalance: `graph_bisect_coarsest_spectral`
     * falls back internally to GGGP (still produces a valid 2-way
     * partition).  The caller never sees the failure; the existing
     * 3-way conversion path runs unconditionally on success.
     *
     * Default-off (env var unset / `multilevel`) leaves the existing
     * multilevel `sparse_graph_partition` path unchanged — Sprint 27
     * Day 6 behaviour preserved bit-identically. */
    int used_root_spectral = 0;
    if (depth == 0) {
        nd_root_bisect_strategy_t root_strategy = parse_nd_root_bisect_strategy();
        idx_t root_max_n = parse_nd_root_bisect_max_n();
        if (root_strategy == ND_ROOT_BISECT_SPECTRAL && n <= root_max_n && n >= 3) {
            rc = graph_bisect_coarsest_spectral(G, part);
            if (rc == SPARSE_OK)
                rc = graph_edge_separator_to_vertex_separator(G, part);
            used_root_spectral = (rc == SPARSE_OK);
        }
    }
    if (!used_root_spectral) {
        rc = sparse_graph_partition(G, part, &sep_count);
    }

    if (nd_prof_enabled) {
        long long elapsed = nd_prof_now_ns() - part_t0;
        nd_prof_partition_ns += elapsed;
        nd_prof_partition_calls++;
        /* Sprint 26 Day 4: per-depth attribution.  Bounds-check
         * overflow into the MAX_ND_DEPTH-1 bucket; warn once. */
        int bucket = (depth < MAX_ND_DEPTH) ? depth : (MAX_ND_DEPTH - 1);
        if (depth >= MAX_ND_DEPTH && !nd_prof_depth_overflow_warned) {
            fprintf(stderr,
                    "nd-profile WARNING: recursion depth %d exceeds MAX_ND_DEPTH=%d; "
                    "accumulating into bucket %d\n",
                    depth, MAX_ND_DEPTH, MAX_ND_DEPTH - 1);
            nd_prof_depth_overflow_warned = 1;
        }
        nd_prof_partition_ns_per_depth[bucket] += elapsed;
        nd_prof_partition_calls_per_depth[bucket]++;
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
        rc = nd_recurse(&G0, map0, perm, next_pos, depth + 1);
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
        rc = nd_recurse(&G1, map1, perm, next_pos, depth + 1);
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
     * calls produce independent measurements (the accumulators are
     * `_Thread_local` per the file-top declaration, so concurrent
     * calls on different matrices race only on the env-var read,
     * not on the accumulator state). */
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
        /* Sprint 26 Day 4: per-depth partition accumulators. */
        for (int d = 0; d < MAX_ND_DEPTH; d++) {
            nd_prof_partition_ns_per_depth[d] = 0;
            nd_prof_partition_calls_per_depth[d] = 0;
        }
        nd_prof_depth_overflow_warned = 0;
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
    rc = nd_recurse(&G, root_map, perm, &next_pos, /*depth=*/0);

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
        /* Sprint 26 Day 4: per-depth partition breakdown.  Emit one
         * line per non-empty depth bucket.  Sized columns: depth (3
         * digits), calls (5 digits), total_ms (10.3 width), avg_ms
         * (8.3 width). */
        fprintf(stderr, "  partition_per_depth:\n");
        fprintf(stderr, "    depth  calls   total_ms      avg_ms\n");
        for (int d = 0; d < MAX_ND_DEPTH; d++) {
            if (nd_prof_partition_calls_per_depth[d] == 0)
                continue;
            double total_ms = (double)nd_prof_partition_ns_per_depth[d] / 1.0e6;
            double avg_ms = total_ms / (double)nd_prof_partition_calls_per_depth[d];
            fprintf(stderr, "    %5d  %5d   %10.3f  %10.3f\n", d,
                    (int)nd_prof_partition_calls_per_depth[d], total_ms, avg_ms);
        }
    }
    return rc;
}
