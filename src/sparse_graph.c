/*
 * sparse_graph.c — Remaining FM / uncoarsening / orchestration slice
 *                  of the Sprint 22 multilevel graph partitioner.
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
 *      graph has n_coarsest ≤ MAX(20, n_orig / divisor) where
 *      divisor defaults to 100 (overridable via the Sprint 24 Day 5
 *      env var `SPARSE_ND_COARSEN_FLOOR_RATIO`).  The hierarchy
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
 *      tree's height inflation; Sprint 24 Day 6 adds a
 *      `balanced_boundary` strategy via `SPARSE_ND_SEP_LIFT_STRATEGY`
 *      that lifts the smaller-boundary side instead).
 *
 * **Vertex-separator output convention.**  `sparse_graph_partition`
 * writes `part[i] ∈ {0, 1, 2}` (0 = left, 1 = right, 2 = separator).
 * The recursive ND driver consumes this 3-way labelling, recurses on
 * the two subgraphs induced by part==0 and part==1, then appends the
 * separator vertices last to the output permutation.  Sprint 22
 * Day 6 implements that recursion.
 *
 * **Small-graph base case.**  Sprint 22 Day 6's recursion stops when
 * a subgraph has n ≤ `sparse_reorder_nd_base_threshold` (default 32
 * from the Day 9 sweep) and emits the subgraph's vertices in
 * natural (subgraph-local) order.  The partitioner itself doesn't
 * impose this threshold — it's an ND-driver decision — but the
 * brute-force bisection at the coarsest level gives the partitioner
 * its own micro-fast-path for n ≤ 20.  The Sprint 22 plan's
 * follow-up of splicing quotient-graph AMD into each leaf is
 * deferred to Sprint 23 (see `docs/planning/EPIC_2/PROJECT_PLAN.md`).
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
 * **Sprint 43 Phase 1 ownership note.**
 *   - graph construction / ownership now lives in
 *     `src/sparse_graph_core.c`
 *   - hierarchy / coarsening now lives in
 *     `src/sparse_graph_coarsen.c`
 *   - coarse bisection now lives in
 *     `src/sparse_graph_bisect.c`
 *   - this file intentionally retains:
 *       - FM refinement
 *       - uncoarsening
 *       - separator lifting
 *       - top-level partition orchestration
 *
 * That split keeps the current extraction phase bounded while
 * preserving the original multilevel partition contract consumed by
 * `src/sparse_reorder_nd.c`.
 */

#include "sparse_graph_fm_buckets.h"
#include "sparse_graph_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Sprint 26 Day 7: thread-local override for the FINEST FM FIFO
 * tie-break.  `graph_uncoarsen` sets this to 1 before invoking
 * `graph_refine_fm` at the finest level when
 * SPARSE_FM_FINEST_STRATEGY=fifo, restores to 0 after.
 * `graph_refine_fm` reads this once on entry and dispatches to
 * `fm_bucket_pop_max_tail` (FIFO) or `fm_bucket_pop_max` (LIFO,
 * Sprint 23 baseline) accordingly.  `_Thread_local` keeps
 * concurrent FM calls race-free.  See
 * `docs/planning/EPIC_2/SPRINT_26/finest_fm_design.md`. */
static _Thread_local int fm_pop_use_tail = 0;

/* Sprint 27 Day 5: thread-local override for annealing-acceptance
 * FM at the finest level.  `graph_uncoarsen` sets this to 1 before
 * invoking `graph_refine_fm` at the finest level when
 * SPARSE_FM_FINEST_STRATEGY=annealing, restores to 0 after.  Day 5
 * lands the parser + dispatch wiring (skeleton); Day 6 implements
 * the acceptance probability `P = exp(-Δgain / T)` overlay in
 * `graph_refine_fm`'s pop-eval-accept loop.  `_Thread_local` keeps
 * concurrent FM calls race-free.  See
 * `docs/planning/EPIC_2/SPRINT_27/annealing_fm_design.md`. */
static _Thread_local int fm_use_annealing = 0;

/* Sprint 27 Day 5: temperature schedule for annealing FM.  Day 5
 * stubs three values; default exponential matches the classical
 * Kirkpatrick-1983 SA formulation (T_k = T_0 × α^k where α ≈ 0.5).
 * Day 6 wires the per-pass T computation; until then all three
 * values fall through to baseline. */
typedef enum {
    FM_ANNEAL_SCHEDULE_LINEAR = 0,
    FM_ANNEAL_SCHEDULE_EXPONENTIAL = 1, /* default */
    FM_ANNEAL_SCHEDULE_COSINE = 2,
} fm_anneal_schedule_t;

static _Thread_local fm_anneal_schedule_t fm_anneal_schedule = FM_ANNEAL_SCHEDULE_EXPONENTIAL;

/* Sprint 27 Day 10: thread-local override for thick-restart FM at
 * the finest level.  `graph_uncoarsen` sets this to 1 before
 * invoking `graph_refine_fm` at the finest level when
 * SPARSE_FM_FINEST_STRATEGY=thick_restart, restored to 0 after.
 * Day 10 lands the parser + dispatch wiring (skeleton); Day 11
 * implements the global-best-tracking + per-pass perturbation
 * overlay in graph_refine_fm.  See
 * `docs/planning/EPIC_2/SPRINT_27/thick_restart_design.md`. */
static _Thread_local int fm_use_thick_restart = 0;

/* Sprint 27 Day 10: perturbation strategy for thick-restart.  Day
 * 10 stubs three values; default random_flip matches the simplest
 * formulation (flip k = 1 % × n random vertices' partition
 * assignments before each pass except the first).  Day 11 wires
 * the per-pass perturbation; until then all three values fall
 * through to baseline. */
typedef enum {
    FM_THICK_RESTART_PERTURB_RANDOM_FLIP = 0, /* default */
    FM_THICK_RESTART_PERTURB_BOUNDARY_SHUFFLE = 1,
    FM_THICK_RESTART_PERTURB_GAUSS_NOISE = 2,
    /* Sprint 28 Day 2: formal gain-bucket-noise variant.  Replaces
     * Sprint 27 Day 11's simplified gauss_noise with the Day-10
     * design's intent (perturb gain comparator, not partition state).
     * Inserts each vertex into the bucket at index
     * `gain[v] + sigma_k * max_weighted_degree * randn()` where
     * sigma_k decays per pass (linear / exponential). */
    FM_THICK_RESTART_PERTURB_GAIN_NOISE_FORMAL = 3,
} fm_thick_restart_perturb_t;

static _Thread_local fm_thick_restart_perturb_t fm_thick_restart_perturb =
    FM_THICK_RESTART_PERTURB_RANDOM_FLIP;

static fm_thick_restart_perturb_t parse_fm_thick_restart_perturb(void) {
    const char *env = getenv("SPARSE_FM_THICK_RESTART_PERTURB");
    if (!env)
        return FM_THICK_RESTART_PERTURB_RANDOM_FLIP;
    if (strcmp(env, "boundary_shuffle") == 0)
        return FM_THICK_RESTART_PERTURB_BOUNDARY_SHUFFLE;
    if (strcmp(env, "gauss_noise") == 0)
        return FM_THICK_RESTART_PERTURB_GAUSS_NOISE;
    if (strcmp(env, "gain_noise_formal") == 0)
        return FM_THICK_RESTART_PERTURB_GAIN_NOISE_FORMAL;
    /* Default + unrecognized + "random_flip" all fall through. */
    return FM_THICK_RESTART_PERTURB_RANDOM_FLIP;
}

/* Sprint 28 Day 2: gain-noise schedule for the formal variant.  Mirrors
 * the Sprint 27 Day 5-6 annealing-schedule axis but applies to
 * sigma_k decay rather than temperature decay.  Default linear
 * (cheaper than exponential to compute, predictable cutoff at last
 * pass).  Read once on entry to graph_refine_fm via the thread-local
 * `fm_gain_noise_schedule` set by graph_uncoarsen at the finest
 * level when fm_thick_restart_perturb == GAIN_NOISE_FORMAL. */
typedef enum {
    FM_GAIN_NOISE_SCHEDULE_LINEAR = 0,      /* default; sigma_k = sigma_0 * (1 - k/K) */
    FM_GAIN_NOISE_SCHEDULE_EXPONENTIAL = 1, /* sigma_k = sigma_0 * 0.5^k */
    FM_GAIN_NOISE_SCHEDULE_COSINE = 2,      /* sigma_k = sigma_0/2 * (1 + cos(πk/K)) */
} fm_gain_noise_schedule_t;

static _Thread_local fm_gain_noise_schedule_t fm_gain_noise_schedule =
    FM_GAIN_NOISE_SCHEDULE_LINEAR;

static fm_gain_noise_schedule_t parse_fm_gain_noise_schedule(void) {
    const char *env = getenv("SPARSE_FM_GAIN_NOISE_SCHEDULE");
    if (!env)
        return FM_GAIN_NOISE_SCHEDULE_LINEAR;
    if (strcmp(env, "exponential") == 0)
        return FM_GAIN_NOISE_SCHEDULE_EXPONENTIAL;
    if (strcmp(env, "cosine") == 0)
        return FM_GAIN_NOISE_SCHEDULE_COSINE;
    /* Default + unrecognized + "linear" all fall through. */
    return FM_GAIN_NOISE_SCHEDULE_LINEAR;
}

/* Sprint 27 Day 11: thick-restart perturbation helper.  Modifies
 * `part[]` in place by flipping some 2-way partition assignments
 * (0 → 1 or 1 → 0) according to the chosen mode.  Called by
 * `graph_uncoarsen`'s finest-level pass loop before each pass except
 * the first when fm_use_thick_restart is set.
 *
 * Mode semantics:
 *   - RANDOM_FLIP (default): flip k = max(1, n/100) random vertices.
 *     O(k) per call.  Cheapest variant.
 *   - BOUNDARY_SHUFFLE: identify boundary vertices (vertices with at
 *     least one cross-edge to the other side), then random-flip ~50 %
 *     of them.  O(|E|) per call.  Targets the FM-relevant region.
 *   - GAUSS_NOISE: Day-10 design described this as "Gaussian noise on
 *     GAIN estimates".  Implementing that requires deeper integration
 *     with graph_refine_fm's gain-init phase.  Day 11 simplification:
 *     random-flip with k drawn proportional to a half-Gaussian
 *     (typical k ≈ n/50, more spread than RANDOM_FLIP).  Documented
 *     deviation in `SPRINT_27/thick_restart_design.md`; the formal
 *     gain-noise variant routes to Sprint 28+ if Day 12's flip-rule
 *     decision motivates it.
 *
 * Determinism: the RNG state is owned by the caller (typically a
 * per-call deterministic seed), passed through and updated
 * in-place.  Same input → same output.
 *
 * Sprint 28 Day 2: GAIN_NOISE_FORMAL is a no-op here — its
 * perturbation lives inside graph_refine_fm at gain-bucket-init
 * time (per-vertex Gaussian noise on bucket placement), not in the
 * partition state.  The graph_uncoarsen anchor-restoration memcpy
 * still fires for that mode (preserves global-best tracking across
 * passes); only the partition-state perturbation is skipped. */
static void thick_restart_perturb(const sparse_graph_t *G, idx_t *part,
                                  fm_thick_restart_perturb_t mode, uint32_t *rng) {
    idx_t n = G->n;
    if (n < 2)
        return;

    /* Sprint 28 Day 2: gain-noise formal variant has no partition-state
     * perturbation step; the noise is applied inside graph_refine_fm. */
    if (mode == FM_THICK_RESTART_PERTURB_GAIN_NOISE_FORMAL)
        return;

    if (mode == FM_THICK_RESTART_PERTURB_BOUNDARY_SHUFFLE) {
        for (idx_t v = 0; v < n; v++) {
            if (part[v] != 0 && part[v] != 1)
                continue;
            int boundary = 0;
            for (idx_t k = G->xadj[v]; k < G->xadj[v + 1]; k++) {
                idx_t u = G->adjncy[k];
                if (part[u] != 0 && part[u] != 1)
                    continue;
                if (part[u] != part[v]) {
                    boundary = 1;
                    break;
                }
            }
            if (!boundary)
                continue;
            /* xorshift32 advance — flip ~50 % of boundary vertices. */
            *rng ^= *rng << 13;
            *rng ^= *rng >> 17;
            *rng ^= *rng << 5;
            if ((*rng & 1U) != 0U)
                part[v] = (idx_t)(1 - part[v]);
        }
        return;
    }

    /* RANDOM_FLIP and GAUSS_NOISE both flip k random vertices; only k
     * differs.  GAUSS_NOISE doubles k vs RANDOM_FLIP per the Day-11
     * simplification documented above. */
    idx_t k = n / 100;
    if (k < 1)
        k = 1;
    if (mode == FM_THICK_RESTART_PERTURB_GAUSS_NOISE) {
        k = n / 50;
        if (k < 2)
            k = 2;
    }
    for (idx_t i = 0; i < k; i++) {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 17;
        *rng ^= *rng << 5;
        idx_t v = (idx_t)((*rng) % (uint32_t)n);
        if (part[v] == 0)
            part[v] = 1;
        else if (part[v] == 1)
            part[v] = 0;
    }
}

/* Sprint 27 Day 6: per-pass index + total passes used by
 * graph_refine_fm to compute the temperature `T_k` for the chosen
 * schedule.  graph_uncoarsen sets these before each finest-level
 * graph_refine_fm call when fm_use_annealing == 1; defaults to
 * (0, 1) which produces a single-pass T = T_0 (max temperature)
 * if read by accident from a non-annealing context.
 *
 * Rationale for thread-local rather than a function parameter:
 * graph_refine_fm's signature is part of the internal-but-stable
 * contract used by `partition_once`, `graph_uncoarsen`, and the
 * separator-lift code.  Threading a new (optional) parameter
 * through all callers would touch many sites; the thread-local
 * pattern is already established for the parallel
 * `fm_pop_use_tail` (FIFO) and `fm_use_annealing` flags. */
static _Thread_local int fm_anneal_pass_idx = 0;
static _Thread_local int fm_anneal_total_passes = 1;

static fm_anneal_schedule_t parse_fm_anneal_schedule(void) {
    const char *env = getenv("SPARSE_FM_ANNEALING_SCHEDULE");
    if (!env)
        return FM_ANNEAL_SCHEDULE_EXPONENTIAL;
    if (strcmp(env, "linear") == 0)
        return FM_ANNEAL_SCHEDULE_LINEAR;
    if (strcmp(env, "cosine") == 0)
        return FM_ANNEAL_SCHEDULE_COSINE;
    /* Default + unrecognized + "exponential" all fall through. */
    return FM_ANNEAL_SCHEDULE_EXPONENTIAL;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Coarsest-graph bisection + FM refinement (Sprint 22 Day 3).
 * ═══════════════════════════════════════════════════════════════════════
 */

/* Compute the cut weight of a 2-way partition.  Iterates each
 * undirected edge once via the i < j upper-triangle convention.
 *
 * Invariant: `part[]` is allocated with at least G->n entries (the
 * caller's responsibility — every site that constructs a partition
 * for a sparse_graph_t allocates G->n idx_t entries).  `G->adjncy[k]`
 * for k in [G->xadj[i], G->xadj[i+1]) yields a vertex index in
 * [0, G->n), so `part[j]` is always in bounds.  clang-analyzer can't
 * track adjncy's bounded-vertex invariant across function-call
 * boundaries; suppress the path-sensitive ArrayBound false positive. */
// NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
static idx_t compute_cut_weight(const sparse_graph_t *G, const idx_t *part) {
    idx_t cut = 0;
    for (idx_t i = 0; i < G->n; i++) {
        for (idx_t k = G->xadj[i]; k < G->xadj[i + 1]; k++) {
            idx_t j = G->adjncy[k];
            if (j <= i)
                continue;
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            if (part[i] != part[j])
                cut += G->ewgt ? G->ewgt[k] : 1;
        }
    }
    return cut;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 23 Day 9: gain-bucket data structure for FM refinement.
 *
 * The implementation lives inline in this TU because Day 10 will
 * tightly couple it with `graph_refine_fm` below, and there are no
 * other consumers — keeping it in one file keeps the FM data path
 * inspectable in a single sweep.  The public API is declared in
 * `src/sparse_graph_fm_buckets.h` so `tests/test_graph_fm_buckets.c`
 * can pin the contract independently of the FM hot loop.
 *
 * Sentinel choice: `-1` for an empty bucket head and for next/prev
 * list ends.  All vertex IDs the caller ever passes are in
 * `[0, n_vertices)`, so `-1` never aliases a real ID.  `idx_t` is
 * signed (`int32_t` / `int64_t` per build config) so sign comparison
 * works without casting.
 * ═══════════════════════════════════════════════════════════════════════
 */

#define FM_BUCKET_EMPTY ((idx_t) - 1)

sparse_err_t fm_bucket_array_init(fm_bucket_array_t *arr, idx_t n_vertices, idx_t max_gain) {
    if (!arr)
        return SPARSE_ERR_NULL;
    if (n_vertices < 0 || max_gain < 0)
        return SPARSE_ERR_BADARG;

    /* Guard size computations against overflow.  Same SIZE_MAX pattern
     * the rest of the codebase uses (e.g. `src/sparse_lu_csr.c` lines
     * 60, 1349) — under-allocation here would produce OOB writes in
     * the heads/next/prev fills below.  The first bound uses IDX_MAX
     * (defined alongside idx_t in include/sparse_types.h) so the
     * guard tracks idx_t's actual range — the migration-to-int64_t
     * path the typedef comment documents stays clean. */
    if (max_gain > (IDX_MAX - 1) / 2)
        return SPARSE_ERR_ALLOC; /* 2*max_gain + 1 would overflow idx_t */
    idx_t num_buckets = 2 * max_gain + 1;
    if ((size_t)num_buckets > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC; /* num_buckets * sizeof(idx_t) overflows size_t */

    /* `next` / `prev` allocate to length max(n_vertices, 1) so a zero-
     * vertex graph still gives malloc a non-zero length (avoids the
     * implementation-defined `malloc(0)` corner case). */
    size_t link_len = (size_t)(n_vertices > 0 ? n_vertices : 1);
    if (link_len > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC; /* link_len * sizeof(idx_t) overflows size_t */

    arr->heads = malloc((size_t)num_buckets * sizeof(idx_t));
    arr->tails = malloc((size_t)num_buckets * sizeof(idx_t));
    arr->counts = calloc((size_t)num_buckets, sizeof(idx_t));
    arr->next = malloc(link_len * sizeof(idx_t));
    arr->prev = malloc(link_len * sizeof(idx_t));
    if (!arr->heads || !arr->tails || !arr->counts || !arr->next || !arr->prev) {
        free(arr->heads);
        free(arr->tails);
        free(arr->counts);
        free(arr->next);
        free(arr->prev);
        arr->heads = NULL;
        arr->tails = NULL;
        arr->counts = NULL;
        arr->next = NULL;
        arr->prev = NULL;
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t i = 0; i < num_buckets; i++) {
        arr->heads[i] = FM_BUCKET_EMPTY;
        arr->tails[i] = FM_BUCKET_EMPTY;
    }
    arr->n_vertices = n_vertices;
    arr->max_gain = max_gain;
    arr->bucket_offset = max_gain;
    arr->num_buckets = num_buckets;
    arr->cursor = FM_BUCKET_EMPTY;
    return SPARSE_OK;
}

void fm_bucket_array_free(fm_bucket_array_t *arr) {
    if (!arr)
        return;
    free(arr->heads);
    free(arr->tails);
    free(arr->counts);
    free(arr->next);
    free(arr->prev);
    arr->heads = NULL;
    arr->tails = NULL;
    arr->counts = NULL;
    arr->next = NULL;
    arr->prev = NULL;
    arr->n_vertices = 0;
    arr->max_gain = 0;
    arr->bucket_offset = 0;
    arr->num_buckets = 0;
    arr->cursor = FM_BUCKET_EMPTY;
}

void fm_bucket_insert(fm_bucket_array_t *arr, idx_t vertex, idx_t gain) {
    idx_t bucket = arr->bucket_offset + gain;
    /* Doubly-linked-list head insert: vertex becomes the new head,
     * old head (if any) hangs off vertex's `next`. */
    arr->prev[vertex] = FM_BUCKET_EMPTY;
    arr->next[vertex] = arr->heads[bucket];
    if (arr->heads[bucket] != FM_BUCKET_EMPTY)
        arr->prev[arr->heads[bucket]] = vertex;
    else
        /* Sprint 26 Day 7: inserting into a previously-empty bucket
         * makes this vertex both the head AND the tail.  Otherwise
         * the existing tail (the first-inserted vertex) stays at
         * the tail end of the chain. */
        arr->tails[bucket] = vertex;
    arr->heads[bucket] = vertex;
    arr->counts[bucket]++;
    if (bucket > arr->cursor)
        arr->cursor = bucket;
}

void fm_bucket_remove(fm_bucket_array_t *arr, idx_t vertex, idx_t gain) {
    idx_t bucket = arr->bucket_offset + gain;
    idx_t p = arr->prev[vertex];
    idx_t n = arr->next[vertex];
    if (p != FM_BUCKET_EMPTY)
        arr->next[p] = n;
    else
        arr->heads[bucket] = n;
    if (n != FM_BUCKET_EMPTY)
        arr->prev[n] = p;
    else
        /* Sprint 26 Day 7: removing the tail vertex (next[v] == EMPTY)
         * means the previous vertex becomes the new tail.  If there
         * was no previous (single-element bucket), tail goes to EMPTY
         * — `p == FM_BUCKET_EMPTY` covers that case too. */
        arr->tails[bucket] = p;
    arr->counts[bucket]--;
    /* Cursor walk-down: if we just emptied the cursor's bucket, slide
     * the cursor down past every empty bucket below it.  Worst-case
     * scan is the full array; amortised cost is O(1) over an FM pass
     * because each bucket is visited at most twice (once descending,
     * once if a later insert lifts the cursor back). */
    if (bucket == arr->cursor) {
        while (arr->cursor >= 0 && arr->counts[arr->cursor] == 0)
            arr->cursor--;
    }
}

sparse_err_t fm_bucket_pop_max(fm_bucket_array_t *arr, idx_t *vertex_out, idx_t *gain_out) {
    if (!arr || !vertex_out || !gain_out)
        return SPARSE_ERR_NULL;
    if (arr->cursor < 0)
        return SPARSE_ERR_BOUNDS;
    idx_t bucket = arr->cursor;
    idx_t v = arr->heads[bucket];
    idx_t g = bucket - arr->bucket_offset;
    fm_bucket_remove(arr, v, g);
    *vertex_out = v;
    *gain_out = g;
    return SPARSE_OK;
}

/* Sprint 26 Day 7: FIFO pop variant — pops the cursor bucket's tail
 * (first-inserted vertex among the equal-gain tie group) instead of
 * the head (most-recently inserted).  Used by FM under
 * SPARSE_FM_FINEST_STRATEGY=fifo to break the saturation Sprint 25
 * Day 5 measured.  See SPRINT_26/finest_fm_design.md. */
sparse_err_t fm_bucket_pop_max_tail(fm_bucket_array_t *arr, idx_t *vertex_out, idx_t *gain_out) {
    if (!arr || !vertex_out || !gain_out)
        return SPARSE_ERR_NULL;
    if (arr->cursor < 0)
        return SPARSE_ERR_BOUNDS;
    idx_t bucket = arr->cursor;
    idx_t v = arr->tails[bucket];
    idx_t g = bucket - arr->bucket_offset;
    fm_bucket_remove(arr, v, g);
    *vertex_out = v;
    *gain_out = g;
    return SPARSE_OK;
}

sparse_err_t graph_refine_fm(const sparse_graph_t *G, idx_t *part_io) {
    if (!G || !part_io)
        return SPARSE_ERR_NULL;
    if (G->n == 0)
        return SPARSE_OK;

    /* Sprint 26 Day 7: pick the pop-strategy based on the thread-
     * local override.  `graph_uncoarsen` sets `fm_pop_use_tail = 1`
     * before invoking `graph_refine_fm` at the finest level under
     * SPARSE_FM_FINEST_STRATEGY=fifo; default 0 = Sprint 23 LIFO
     * pop (head).  Read once on entry to avoid per-iteration
     * branching in the FM hot loop. */
    sparse_err_t (*pop_max)(fm_bucket_array_t *, idx_t *, idx_t *) =
        fm_pop_use_tail ? fm_bucket_pop_max_tail : fm_bucket_pop_max;

    /* Sprint 27 Day 6: annealing-acceptance overlay.  When
     * fm_use_annealing is set, negative-gain pops are subjected to a
     * per-vertex acceptance check `random < exp(g / T)`; rejected
     * vertices are parked on the same per-step skipped list as
     * balance-skipped vertices (re-considered next step).  Default
     * fm_use_annealing == 0 → branch is bypassed; baseline FM
     * behaviour is bit-identical to current master. */
    const int use_annealing = fm_use_annealing;
    const int anneal_debug = use_annealing && getenv("SPARSE_FM_ANNEALING_DEBUG") != NULL;
    double anneal_T = 0.0;
    uint32_t anneal_rng = 0;
    idx_t anneal_worsening_accepted = 0;
    idx_t anneal_worsening_rejected = 0;

    /* Sprint 28 Day 2: formal gain-noise overlay (the Day-10 design's
     * intent that Sprint 27 Day 11 simplified to partition-state random-
     * flip).  When `fm_use_thick_restart` is set AND
     * `fm_thick_restart_perturb == GAIN_NOISE_FORMAL`, perturb the
     * bucket placement via `gain_for_bucket[v] = gain[v] + noise[v]`
     * where `noise[v] = sigma_k * max_weighted_degree * randn()` is
     * sampled once per pass per vertex.  The bucket structure is
     * sized for `2 * max_weighted_degree` to absorb the noise +
     * neighbour-update accumulation.  Cut accounting + neighbour-
     * update gain equation use the TRUE `gain[v]`; only the bucket
     * key carries the noise offset.  Default-off path (gain_for_bucket
     * == NULL) is bit-identical to current master.
     *
     * sigma_k decays per pass under `fm_gain_noise_schedule` (default
     * linear; matches PLAN.md Day 2 task 2).  Pass index k =
     * fm_anneal_pass_idx (reused thread-local from Sprint 27 Day 6;
     * graph_uncoarsen sets it before each finest-level call). */
    const int use_gain_noise_formal =
        fm_use_thick_restart &&
        fm_thick_restart_perturb == FM_THICK_RESTART_PERTURB_GAIN_NOISE_FORMAL;
    const int gain_noise_debug =
        use_gain_noise_formal && getenv("SPARSE_FM_GAIN_NOISE_DEBUG") != NULL;
    double gain_sigma_k = 0.0;
    uint32_t gain_noise_rng = 0;

    idx_t n = G->n;

    /* Per-vertex gain = (sum of edge weights to other-side neighbours)
     *                 − (sum of edge weights to same-side neighbours).
     * Moving v flips the cut by -gain[v] (positive gain ⇒ smaller cut).
     *
     * Sprint 23 Day 10: gain values live in the bucket array (one
     * bucket per gain level; doubly-linked list inside each bucket).
     * Per-step max-find is O(1) amortised via the cursor in the
     * bucket array — replaces Sprint 22's O(n) linear scan, lifts FM
     * total complexity from O(n²) to O(|E|) per pass. */
    idx_t *gain = malloc((size_t)n * sizeof(idx_t));
    int *locked = calloc((size_t)n, sizeof(int));
    int *in_bucket = calloc((size_t)n, sizeof(int));
    idx_t *best_part = malloc((size_t)n * sizeof(idx_t));
    /* `skipped_this_step` is the per-step deferred-pop list: vertices
     * popped but balance-ineligible *this* step.  Re-inserted into the
     * bucket at end-of-step so the next step (with shifted w0/w1)
     * reconsiders them — Sprint 22's per-step full re-scan retried
     * such vertices implicitly; the bucket-FM has to spell it out. */
    idx_t *skipped_this_step = malloc((size_t)n * sizeof(idx_t));
    /* Sprint 28 Day 2: gain-for-bucket array.  Only allocated under
     * the gain_noise_formal overlay; carries the per-vertex noise
     * offset that the bucket structure uses for placement.  NULL on
     * the default code path → bucket inserts read `gain[v]` directly
     * (bit-identical to Sprint 27 master).  calloc-zeroed so the
     * path-sensitive analyzer sees an initialised value at the
     * bucket-insert read site (the noise-init loop below covers all
     * v ∈ [0, n), but the analyzer can't prove that across two
     * separate `if (use_gain_noise_formal)` blocks).  The redundant
     * `n > 0` guard the malloc previously had is dropped because
     * line 1985's `if (G->n == 0) return SPARSE_OK;` already filters
     * the n == 0 case before we reach this allocation. */
    idx_t *gain_for_bucket = NULL;
    if (use_gain_noise_formal) {
        gain_for_bucket = calloc((size_t)n, sizeof(idx_t));
    }
    if (!gain || !locked || !in_bucket || !best_part || !skipped_this_step ||
        (use_gain_noise_formal && !gain_for_bucket)) {
        free(gain);
        free(locked);
        free(in_bucket);
        free(best_part);
        free(skipped_this_step);
        free(gain_for_bucket);
        return SPARSE_ERR_ALLOC;
    }

    /* Initial gains + max weighted degree (drives bucket array sizing).
     * gain[v] always lives in [-weighted_degree(v), +weighted_degree(v)]
     * — both initially (each edge contributes ±w to either internal or
     * external) and after any sequence of neighbour moves (each edge's
     * contribution to gain[v] flips sign when that edge crosses the
     * partition, but the magnitude is unchanged), so max_weighted_degree
     * is a tight bound on |gain| at every point in the FM walk. */
    idx_t max_weighted_degree = 0;
    for (idx_t v = 0; v < n; v++) {
        idx_t internal = 0;
        idx_t external = 0;
        idx_t v_wd = 0;
        for (idx_t k = G->xadj[v]; k < G->xadj[v + 1]; k++) {
            idx_t u = G->adjncy[k];
            idx_t w = G->ewgt ? G->ewgt[k] : 1;
            v_wd += w;
            if (part_io[v] == part_io[u])
                internal += w;
            else
                external += w;
        }
        gain[v] = external - internal;
        if (v_wd > max_weighted_degree)
            max_weighted_degree = v_wd;
    }

    /* Sprint 28 Day 2: oversize the bucket array by 2× when the
     * gain_noise_formal overlay is active.  Per-vertex noise (sampled
     * below) is bounded by `sigma_0 * max_weighted_degree * |randn()|`;
     * with sigma_0 = 0.5 and the Box-Muller-style sampling clamped to
     * ~3σ, |noise| stays within max_weighted_degree.  The post-init
     * `gain_for_bucket[v] = gain[v] + noise[v]` therefore lives in
     * `[-2*max_weighted_degree, +2*max_weighted_degree]`; subsequent
     * neighbour-updates change `gain[v]` by `±2w` per step but keep
     * the noise offset constant, so the bucket key stays in the
     * doubled range throughout.  Default-off path uses
     * max_weighted_degree (bit-identical to Sprint 27 master). */
    idx_t bucket_max_gain = max_weighted_degree;
    if (use_gain_noise_formal) {
        bucket_max_gain = max_weighted_degree * 2;
        if (bucket_max_gain < max_weighted_degree) /* idx_t overflow guard */
            bucket_max_gain = max_weighted_degree;
    }
    fm_bucket_array_t buckets = {0};
    sparse_err_t rc = fm_bucket_array_init(&buckets, n, bucket_max_gain);
    if (rc != SPARSE_OK) {
        free(gain);
        free(locked);
        free(in_bucket);
        free(best_part);
        free(skipped_this_step);
        free(gain_for_bucket);
        return rc;
    }

    /* Sprint 27 Day 6: compute T_k for the current pass under the
     * configured schedule.  T_0 = max_weighted_degree (an upper
     * bound on |gain|; gives moderate initial acceptance for the
     * worst-case worsening move).  Pass index k = fm_anneal_pass_idx
     * (set by graph_uncoarsen before each finest-level call); total
     * passes K = fm_anneal_total_passes (Sprint 23 Day 11 default 3).
     * Schedule formulae per Day-5 design (annealing_fm_design.md):
     *   LINEAR:      T_k = T_0 × (1 − k/K)
     *   EXPONENTIAL: T_k = T_0 × 0.5^k        (Kirkpatrick-1983 §3)
     *   COSINE:      T_k = T_0/2 × (1 + cos(πk/K))
     * Cutoff: when T <= 1.0, all worsening-move probabilities collapse
     * to <~ 0.37, so annealing effectively stops rejecting late in
     * the schedule. */
    if (use_annealing) {
        int K = fm_anneal_total_passes > 0 ? fm_anneal_total_passes : 1;
        int k = fm_anneal_pass_idx;
        if (k < 0)
            k = 0;
        if (k >= K)
            k = K - 1;
        double T0 = (double)max_weighted_degree;
        switch (fm_anneal_schedule) {
        case FM_ANNEAL_SCHEDULE_LINEAR:
            anneal_T = T0 * (1.0 - (double)k / (double)K);
            break;
        case FM_ANNEAL_SCHEDULE_COSINE:
            anneal_T = T0 * 0.5 * (1.0 + cos(3.14159265358979323846 * (double)k / (double)K));
            break;
        case FM_ANNEAL_SCHEDULE_EXPONENTIAL:
        default:
            anneal_T = T0;
            for (int i = 0; i < k; i++)
                anneal_T *= 0.5;
            break;
        }
        /* Per-call deterministic seed: hash of (n, k).  xorshift32
         * needs a non-zero state; bias by + 1 to guarantee that. */
        anneal_rng =
            (uint32_t)(((uint64_t)(uint32_t)n * 31U + (uint32_t)(uint64_t)(unsigned long)k) *
                           2654435761U +
                       1U);
    }

    /* Sprint 28 Day 2: compute sigma_k for the gain_noise_formal
     * overlay.  sigma_0 = 0.5 means the per-vertex noise magnitude
     * is up to ~50% of max_weighted_degree at pass 0 (with Box-Muller
     * sampling clamped to ~3σ); decays per pass under
     * `fm_gain_noise_schedule`.  Same per-call deterministic seed
     * recipe as Day-6 annealing, but biased by 7U to differentiate
     * the RNG stream when the two overlays compose (annealing +
     * gain_noise_formal). */
    if (use_gain_noise_formal) {
        int K = fm_anneal_total_passes > 0 ? fm_anneal_total_passes : 1;
        int k = fm_anneal_pass_idx;
        if (k < 0)
            k = 0;
        if (k >= K)
            k = K - 1;
        const double sigma_0 = 0.5;
        switch (fm_gain_noise_schedule) {
        case FM_GAIN_NOISE_SCHEDULE_EXPONENTIAL:
            gain_sigma_k = sigma_0;
            for (int i = 0; i < k; i++)
                gain_sigma_k *= 0.5;
            break;
        case FM_GAIN_NOISE_SCHEDULE_COSINE:
            gain_sigma_k =
                sigma_0 * 0.5 * (1.0 + cos(3.14159265358979323846 * (double)k / (double)K));
            break;
        case FM_GAIN_NOISE_SCHEDULE_LINEAR:
        default:
            gain_sigma_k = sigma_0 * (1.0 - (double)k / (double)K);
            break;
        }
        gain_noise_rng =
            (uint32_t)(((uint64_t)(uint32_t)n * 31U + (uint32_t)(uint64_t)(unsigned long)k) *
                           2654435761U +
                       7U);
    }

    /* Sprint 28 Day 2: populate gain_for_bucket[v] = gain[v] + noise[v]
     * when the gain_noise_formal overlay is active.  noise[v] is
     * sampled via Box-Muller-style central-limit approximation: sum
     * 12 uniform draws and subtract 6 to approximate N(0, 1).  Cheap,
     * deterministic given the seeded RNG, and accurate to ~3σ for
     * the bucket-placement use case.  The result is clamped to
     * `±max_weighted_degree` so the post-init gain_for_bucket[v]
     * stays within `±2*max_weighted_degree` (matches the doubled
     * bucket sizing above). */
    if (use_gain_noise_formal && gain_for_bucket) {
        const double noise_scale = gain_sigma_k * (double)max_weighted_degree;
        const idx_t noise_clamp = max_weighted_degree;
        for (idx_t v = 0; v < n; v++) {
            double u_sum = 0.0;
            for (int i = 0; i < 12; i++) {
                gain_noise_rng ^= gain_noise_rng << 13;
                gain_noise_rng ^= gain_noise_rng >> 17;
                gain_noise_rng ^= gain_noise_rng << 5;
                u_sum += (double)gain_noise_rng / 4294967296.0;
            }
            double standard_normal = u_sum - 6.0;
            double noise = noise_scale * standard_normal;
            idx_t noise_int;
            if (noise > (double)noise_clamp)
                noise_int = noise_clamp;
            else if (noise < -(double)noise_clamp)
                noise_int = -noise_clamp;
            else
                noise_int = (idx_t)noise;
            gain_for_bucket[v] = gain[v] + noise_int;
        }
    }

    /* Insert every vertex into the bucket initially.  Vertices with
     * negative gain (interior to their side) are still inserted —
     * Sprint 22's FM scanned them too, accepting transient cut
     * increases, and the rollback-to-best-cut step at the end keeps
     * us from finalising a worse partition.
     *
     * Iterate in reverse so the bucket's head-insert pattern leaves
     * the lowest-ID vertex at each bucket's head — this matches
     * Sprint 22's lowest-ID-wins tie-breaking on equal gains, which
     * `test_ldlt_via_nd_dispatch` (bcsstk04 LDL^T no-pivoting)
     * happened to depend on.  Strictly an initial-state invariant —
     * neighbour-update remove+inserts during the FM walk can
     * scatter IDs within a bucket — but downstream tests show this
     * is sufficient to avoid the catastrophic residual blow-up. */
    for (idx_t v = n - 1; v >= 0; v--) {
        /* Sprint 28 Day 2: clang-analyzer-core.uninitialized.Assign
         * false positive — both `gain[v]` (init loop above, lines
         * ~2086-2103) and `gain_for_bucket[v]` (calloc'd + noise-init
         * loop above when use_gain_noise_formal is true) are written
         * for every v ∈ [0, n) before this read.  The analyzer can't
         * track the gain-init loop's write across the intervening
         * thick-restart / annealing branches. */
        // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
        idx_t bucket_key =
            (use_gain_noise_formal && gain_for_bucket) ? gain_for_bucket[v] : gain[v];
        fm_bucket_insert(&buckets, v, bucket_key);
        in_bucket[v] = 1;
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

    /* FM main loop.  pop_max returns the highest-gain still-in-bucket
     * vertex; balance-ineligible pops get parked on the per-step
     * skipped list and re-inserted at end-of-step so they're
     * reconsidered next step (when w0/w1 may have shifted in their
     * favour).  This preserves Sprint 22's "consider every unlocked
     * vertex every step" semantics — the partition tests
     * (tests/test_graph.c) and the bcsstk04 LDL^T no-pivoting
     * residual test (tests/test_reorder_nd.c) both depend on it. */
    for (idx_t step = 0; step < n; step++) {
        idx_t best_v = -1;
        idx_t best_g = 0;
        int have_candidate = 0;
        idx_t skipped_count = 0;
        while (buckets.cursor >= 0) {
            idx_t v = -1;
            idx_t g = 0;
            sparse_err_t pop_rc = pop_max(&buckets, &v, &g);
            if (pop_rc != SPARSE_OK)
                break;
            in_bucket[v] = 0;
            /* Plan invariant: pop_max never returns a locked vertex
             * because vertices get popped (and thus removed) before
             * being marked locked; subsequent neighbour-update
             * propagation also remove-then-reinserts only unlocked
             * neighbours. */
            if (locked[v]) {
                /* Defensive only — should be unreachable. */
                continue;
            }
            idx_t v_w = G->vwgt ? G->vwgt[v] : 1;
            idx_t new_w0 = part_io[v] == 0 ? w0 - v_w : w0 + v_w;
            idx_t new_w1 = part_io[v] == 0 ? w1 + v_w : w1 - v_w;
            idx_t new_imbal = new_w0 > new_w1 ? new_w0 - new_w1 : new_w1 - new_w0;
            if (new_imbal > max_imbal) {
                /* skipped_count ≤ n by construction: each pop sets
                 * `in_bucket[v] = 0`, so the inner while can pop a
                 * given vertex at most once per FM step, capping
                 * skipped_count at the bucket's pre-step occupancy
                 * (≤ n).  clang-analyzer's path-sensitive search
                 * doesn't see this — silence the false positive. */
                // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
                skipped_this_step[skipped_count++] = v;
                continue;
            }
            /* Sprint 27 Day 6: annealing-acceptance overlay.  When
             * fm_use_annealing is set and T > 1, negative-gain pops
             * are accepted with probability `exp(g / T)` and rejected
             * (re-bucketed at end-of-step) with probability `1 - P`.
             * Positive-gain (improving) moves and balance-eligible
             * zero-gain moves bypass this check entirely — they're
             * always accepted, matching baseline FM.  Default
             * `use_annealing == 0` skips the branch. */
            if (use_annealing && g < 0 && anneal_T > 1.0) {
                /* xorshift32 advance — produces a uniform 32-bit
                 * value; convert to [0, 1) by dividing by 2^32. */
                anneal_rng ^= anneal_rng << 13;
                anneal_rng ^= anneal_rng >> 17;
                anneal_rng ^= anneal_rng << 5;
                double r = (double)anneal_rng / 4294967296.0;
                double accept_p = exp((double)g / anneal_T);
                if (r >= accept_p) {
                    anneal_worsening_rejected++;
                    // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
                    skipped_this_step[skipped_count++] = v;
                    continue;
                }
                anneal_worsening_accepted++;
            }
            best_v = v;
            best_g = g;
            have_candidate = 1;
            break;
        }

        /* Move + neighbour gain-update happen first (touches gain[]
         * for all unlocked neighbours, bucket re-shuffle for
         * in_bucket ones), so when we re-insert the skipped vertices
         * below their gain[] reflects the move's effect. */
        if (have_candidate) {
            /* Sprint 28 Day 2: cut accounting uses the TRUE gain
             * (gain[best_v]) rather than the popped bucket key
             * (best_g) when the gain_noise_formal overlay is active —
             * the bucket key carries the per-vertex noise offset and
             * is not the actual cut delta.  Default-off path uses
             * best_g (== gain[best_v] at pop) bit-identically. */
            idx_t cut_delta = (use_gain_noise_formal && gain_for_bucket) ? gain[best_v] : best_g;
            cur_cut -= cut_delta;
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

            /* Neighbour gains: edge (best_v, u) flipped from external
             * to internal (u on new_side) → gain[u] -= 2w; from
             * internal to external (u on old_side) → gain[u] += 2w.
             * gain[] is updated for *every* unlocked neighbour; the
             * bucket re-shuffle only fires for currently-in-bucket
             * neighbours, so skipped-this-step vertices pick up their
             * updated gain when we re-insert them below.
             *
             * Sprint 28 Day 2: when gain_noise_formal is active,
             * gain_for_bucket[u] also shifts by the same ±2w delta
             * (the per-vertex noise offset is constant within a pass;
             * neighbour-update is linear).  Bucket remove+insert uses
             * the noisy key on both sides. */
            for (idx_t k = G->xadj[best_v]; k < G->xadj[best_v + 1]; k++) {
                idx_t u = G->adjncy[k];
                if (locked[u])
                    continue;
                idx_t w = G->ewgt ? G->ewgt[k] : 1;
                idx_t old_g = gain[u];
                idx_t old_bucket_key =
                    (use_gain_noise_formal && gain_for_bucket) ? gain_for_bucket[u] : old_g;
                if (part_io[u] == new_side) {
                    gain[u] -= 2 * w;
                    if (use_gain_noise_formal && gain_for_bucket)
                        gain_for_bucket[u] -= 2 * w;
                } else {
                    gain[u] += 2 * w;
                    if (use_gain_noise_formal && gain_for_bucket)
                        gain_for_bucket[u] += 2 * w;
                }
                if (in_bucket[u]) {
                    fm_bucket_remove(&buckets, u, old_bucket_key);
                    idx_t new_bucket_key =
                        (use_gain_noise_formal && gain_for_bucket) ? gain_for_bucket[u] : gain[u];
                    fm_bucket_insert(&buckets, u, new_bucket_key);
                }
            }

            if (cur_cut < best_cut) {
                best_cut = cur_cut;
                memcpy(best_part, part_io, (size_t)n * sizeof(idx_t));
            }
        }

        /* Re-insert balance-skipped vertices for next-step
         * consideration.  Their gain[] has been updated by the
         * neighbour-update loop above (when applicable), so the
         * bucket placement reflects their current gain.
         *
         * Sprint 28 Day 2: under the gain_noise_formal overlay, the
         * bucket key is `gain_for_bucket[w]` (which carries the per-
         * vertex noise offset and was tracked in lockstep with
         * `gain[w]` above). */
        for (idx_t i = 0; i < skipped_count; i++) {
            idx_t w = skipped_this_step[i];
            if (!locked[w]) {
                idx_t reinsert_key =
                    (use_gain_noise_formal && gain_for_bucket) ? gain_for_bucket[w] : gain[w];
                fm_bucket_insert(&buckets, w, reinsert_key);
                in_bucket[w] = 1;
            }
        }

        if (!have_candidate)
            break;
    }

    /* Roll back to the best state. */
    memcpy(part_io, best_part, (size_t)n * sizeof(idx_t));

    /* Sprint 27 Day 6: emit annealing per-pass stats under
     * SPARSE_FM_ANNEALING_DEBUG=1 (default off; one-branch overhead
     * when off). */
    if (anneal_debug) {
        fprintf(stderr,
                "fm-annealing-debug n=%d pass=%d/%d schedule=%d T=%.3f "
                "worsening_accepted=%d worsening_rejected=%d\n",
                (int)n, fm_anneal_pass_idx, fm_anneal_total_passes, (int)fm_anneal_schedule,
                anneal_T, (int)anneal_worsening_accepted, (int)anneal_worsening_rejected);
    }

    /* Sprint 28 Day 2: emit gain-noise per-pass stats under
     * SPARSE_FM_GAIN_NOISE_DEBUG=1 (default off; one-branch overhead
     * when off). */
    if (gain_noise_debug) {
        fprintf(stderr,
                "fm-gain-noise-debug n=%d pass=%d/%d schedule=%d sigma_k=%.4f best_cut=%d\n",
                (int)n, fm_anneal_pass_idx, fm_anneal_total_passes, (int)fm_gain_noise_schedule,
                gain_sigma_k, (int)best_cut);
    }

    fm_bucket_array_free(&buckets);
    free(gain);
    free(locked);
    free(in_bucket);
    free(best_part);
    free(skipped_this_step);
    free(gain_for_bucket);
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

    /* Sprint 23 Day 11: 3-pass FM at the finest level.  Sprint 22 ran
     * a single FM pass per uncoarsening level; Day 11's exploration
     * (`docs/planning/EPIC_2/SPRINT_23/davis_notes.md` §"Day-11
     * finding") measured end-to-end nnz(L) on Pres_Poisson under
     * SPARSE_FM_FINEST_PASSES = {1, 2, 3, 5} and observed:
     *
     *   - 1 pass: ratio 1.026×, ND wall 47.3 s
     *   - 2 pass: ratio 0.958×, ND wall 41.4 s
     *   - 3 pass: ratio 0.952×, ND wall 40.5 s   ← chosen
     *   - 5 pass: ratio 0.953×, ND wall 41.2 s   (no further win)
     *
     * 3 is the sweet spot: each successive pass converges further
     * toward the FM local optimum on this fixture's separator
     * structure, with diminishing returns past pass 3.  ND/AMD now
     * lands at 0.95× — Pres_Poisson ND beats AMD, the headline
     * fill-quality gate from Sprint 22 onwards.
     *
     * Override via SPARSE_FM_FINEST_PASSES env var (1..16) for
     * regression bisection.  The intermediate-level passes stay at
     * 1 — the multilevel coarsening already gives those levels a
     * mostly-converged input, and adding passes there is wall-time
     * cost without measurable fill win. */
    int finest_passes = 3;
    {
        const char *env = getenv("SPARSE_FM_FINEST_PASSES");
        if (env) {
            /* `strtol` with end-pointer + range checks instead of
             * `atoi`: env-var inputs are user-controlled, and atoi
             * has UB on overflow + silently accepts non-numeric
             * prefixes ("3foo" → 3).  Reject anything that isn't a
             * pure integer in [1, 16] and fall back to the default
             * (3) on any parse / range failure. */
            char *endp = NULL;
            long v = strtol(env, &endp, 10);
            if (env != endp && *endp == '\0' && v >= 1 && v <= 16)
                finest_passes = (int)v;
        }
    }

    /* Sprint 26 Day 6: SPARSE_FM_FINEST_STRATEGY env-var parser
     * stub.  Day 4's per-recursion-level profile identified
     * sub-axis (b) bucket-tie-break (FIFO via tails[]) as the
     * highest-leverage, lowest-risk Item 5 candidate;
     * SPRINT_26/finest_fm_design.md picks `fifo` as the value name
     * for the new pop-from-tail variant.  Day 6 lands the parser
     * + a no-op dispatch (all values fall through to baseline);
     * Day 7 implements `fifo` semantics; Day 8 sweeps + decides
     * whether to flip default.
     *
     * Range: {baseline, fifo, annealing, thick_restart}.  Default
     * `baseline` (Sprint 23 LIFO-on-insertion-order behavior) is
     * preserved bit-identically.  Out-of-range / non-numeric /
     * missing → baseline.  Sub-axes annealing + thick_restart are
     * recognized as valid values for forward-compatibility but
     * unimplemented in Sprint 26 (rejected per Day 4 design); they
     * fall through to baseline.  See SPRINT_26/finest_fm_design.md
     * "Rejected alternatives" for the reasoning. */
    enum {
        FINEST_FM_BASELINE = 0,
        FINEST_FM_FIFO = 1,
        FINEST_FM_ANNEALING = 2,
        FINEST_FM_THICK_RESTART = 3,
        /* Sprint 28 Day 4 — multi-strategy FM ensemble: run K
         * sub-strategies in parallel per finest-level call (default
         * baseline + fifo + annealing per `SPARSE_FM_ENSEMBLE_STRATEGIES`)
         * and pick the lowest-cut result.  See
         * docs/planning/EPIC_2/SPRINT_28/ensemble_fm_design.md. */
        FINEST_FM_ENSEMBLE = 4,
    } finest_strategy = FINEST_FM_BASELINE;
    {
        const char *env = getenv("SPARSE_FM_FINEST_STRATEGY");
        if (env) {
            if (strcmp(env, "fifo") == 0)
                finest_strategy = FINEST_FM_FIFO;
            else if (strcmp(env, "annealing") == 0)
                finest_strategy = FINEST_FM_ANNEALING;
            else if (strcmp(env, "thick_restart") == 0)
                finest_strategy = FINEST_FM_THICK_RESTART;
            else if (strcmp(env, "ensemble") == 0)
                finest_strategy = FINEST_FM_ENSEMBLE;
            /* Unrecognized + "baseline" both fall through to
             * FINEST_FM_BASELINE. */
        }
    }
    /* Sprint 27 Day 5 dispatch update: Day 5 lands the `annealing`
     * skeleton.  Sprint 26 Day 6's design rejected annealing on
     * cost grounds (20-50 % wall expansion); Sprint 26 Day 5's
     * `nd_base_threshold = 96` flip + Sprint 27 Day 3's = 128 flip
     * cumulatively cut Pres_Poisson ND wall 38 s → 7 s, making the
     * wall budget affordable.  Day 5 wires `fm_use_annealing` +
     * `fm_anneal_schedule` thread-locals; Day 6 lands the
     * acceptance-probability overlay + measurement.  `thick_restart`
     * stays unimplemented (Sprint 27 item 6 budget; Days 10-12). */
    fm_anneal_schedule_t anneal_schedule_choice = parse_fm_anneal_schedule();
    fm_thick_restart_perturb_t thick_restart_perturb_choice = parse_fm_thick_restart_perturb();
    /* Sprint 28 Day 2: gain-noise schedule for the formal thick-restart
     * variant.  Only consulted by graph_refine_fm when
     * fm_thick_restart_perturb == GAIN_NOISE_FORMAL; defaults to
     * linear so the default-off code path stays bit-identical. */
    fm_gain_noise_schedule_t gain_noise_schedule_choice = parse_fm_gain_noise_schedule();

    /* Sprint 28 Day 4: multi-strategy FM ensemble strategy list.
     * Parsed from `SPARSE_FM_ENSEMBLE_STRATEGIES` (default
     * "baseline,fifo,annealing"); recognized values are the same
     * `SPARSE_FM_FINEST_STRATEGY` enum names except `ensemble`
     * itself (would recurse) and `thick_restart` (Day-4 scope:
     * ensemble runs single-pass per-strategy, but thick_restart's
     * value comes from multi-pass anchor + perturbation — skipped
     * silently in the ensemble; can be added in a future sprint).
     * Capped at 4 entries (the four supported sub-strategies);
     * de-duplicated by first-occurrence-wins; empty list degenerates
     * to {baseline} so ensemble == baseline matches Sprint 27
     * default.  See docs/planning/EPIC_2/SPRINT_28/ensemble_fm_design.md. */
    int ensemble_strategy_list[4] = {0, 0, 0, 0};
    int ensemble_strategy_count = 0;
    if (finest_strategy == FINEST_FM_ENSEMBLE) {
        const char *env = getenv("SPARSE_FM_ENSEMBLE_STRATEGIES");
        char buf[256];
        /* Oversize-input handling (PR #36 review): the default
         * selector list is < 30 chars; any value approaching the
         * 256-byte buffer signals a malformed env var.  Rather than
         * silently truncate (which could drop a token mid-string and
         * surprise the caller), reject oversize inputs by falling
         * back to the default list — same behaviour the caller would
         * see if they unset the env var. */
        const char *list;
        if (env && *env && strlen(env) >= sizeof(buf)) {
            fprintf(stderr,
                    "fm-ensemble WARNING: SPARSE_FM_ENSEMBLE_STRATEGIES is "
                    "%zu bytes (>= %zu); falling back to default selector\n",
                    strlen(env), sizeof(buf));
            list = "baseline,fifo,annealing";
        } else {
            list = (env && *env) ? env : "baseline,fifo,annealing";
        }
        size_t list_len = strlen(list);
        /* list_len < sizeof(buf) is now guaranteed by the oversize check
         * above; the safety copy preserves it as an invariant for any
         * future caller that ignores the warning. */
        if (list_len >= sizeof(buf))
            list_len = sizeof(buf) - 1;
        memcpy(buf, list, list_len);
        buf[list_len] = '\0';
        /* Portable manual comma-tokenizer (replaces POSIX strtok_r which is
         * not in MSVC's <string.h>; Sprint 28 Day-4 first cut used strtok_r
         * + a `_POSIX_C_SOURCE` feature-test macro which closed the Ubuntu
         * lint but blocked Windows builds — PR #36 review feedback). */
        char *tok = buf;
        while (tok && *tok && ensemble_strategy_count < 4) {
            char *comma = tok;
            while (*comma && *comma != ',')
                comma++;
            int has_more = (*comma == ',');
            if (has_more)
                *comma = '\0';
            while (*tok == ' ' || *tok == '\t')
                tok++;
            size_t tok_len = strlen(tok);
            while (tok_len > 0 && (tok[tok_len - 1] == ' ' || tok[tok_len - 1] == '\t' ||
                                   tok[tok_len - 1] == '\n')) {
                tok[--tok_len] = '\0';
            }
            int strat = -1;
            if (strcmp(tok, "baseline") == 0)
                strat = FINEST_FM_BASELINE;
            else if (strcmp(tok, "fifo") == 0)
                strat = FINEST_FM_FIFO;
            else if (strcmp(tok, "annealing") == 0)
                strat = FINEST_FM_ANNEALING;
            /* `thick_restart` + `ensemble` (recursion) + unrecognized
             * silently skipped; ensemble runs the recognized subset. */
            if (strat >= 0) {
                int dup = 0;
                for (int i = 0; i < ensemble_strategy_count; i++) {
                    if (ensemble_strategy_list[i] == strat) {
                        dup = 1;
                        break;
                    }
                }
                if (!dup)
                    ensemble_strategy_list[ensemble_strategy_count++] = strat;
            }
            tok = has_more ? (comma + 1) : NULL;
        }
        if (ensemble_strategy_count == 0) {
            ensemble_strategy_list[0] = FINEST_FM_BASELINE;
            ensemble_strategy_count = 1;
        }
    }
    const int ensemble_debug =
        finest_strategy == FINEST_FM_ENSEMBLE && getenv("SPARSE_FM_ENSEMBLE_DEBUG") != NULL;
    /* Sprint 26 Day 7 dispatch: `fifo` sets `fm_pop_use_tail = 1`
     * for the finest-level call below (restored to 0 after).
     * Sprint 27 Day 5 adds the parallel `annealing` dispatch
     * (sets `fm_use_annealing = 1` + `fm_anneal_schedule` to the
     * parsed schedule choice; restored after).  Sprint 27 Day 10
     * adds the `thick_restart` dispatch wiring (sets
     * `fm_use_thick_restart = 1` + `fm_thick_restart_perturb`;
     * Day 11 lands the global-best-tracking + perturbation
     * overlay in graph_refine_fm). */

    /* Sprint 25 Day 4: SPARSE_FM_INTERMEDIATE_PASSES extends the
     * Sprint 23 Day 11 multi-pass-FM exploration from the finest
     * uncoarsening level to the second-finest (level == 1) and
     * third-finest (level == 2) levels.  Default 1 = Sprint 23
     * behavior bit-identically (intermediate levels stay single-
     * pass).  Range [1, 10]; out-of-range / non-numeric / missing
     * → default 1.  Same strtol + end-pointer + range-check
     * validation pattern as SPARSE_FM_FINEST_PASSES + Sprint 24's
     * SPARSE_ND_COARSEN_FLOOR_RATIO.  The skipped-vertex re-
     * insertion contract (Sprint 23 Day 10's bcsstk04 LDL^T
     * residual hazard fix in graph_refine_fm) holds across the
     * new pass placements: every FM call uses the same internal
     * re-insertion logic, so multi-pass at intermediate levels
     * inherits the contract automatically.  See
     * docs/planning/EPIC_2/SPRINT_25/PLAN.md Day 4 + Sprint 24
     * RETROSPECTIVE.md "Performance highlights" lesson "multi-
     * pass FM's payoff scales with the cost of a single pass". */
    int intermediate_passes = 1;
    {
        const char *env = getenv("SPARSE_FM_INTERMEDIATE_PASSES");
        if (env) {
            char *endp = NULL;
            long v = strtol(env, &endp, 10);
            if (env != endp && *endp == '\0' && v >= 1 && v <= 10)
                intermediate_passes = (int)v;
        }
    }

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
        /* Pass count per level:
         *   level == 0 (finest)     → finest_passes (Sprint 23 Day 11; default 3)
         *   level == 1 or 2         → intermediate_passes (Sprint 25 Day 4; default 1)
         *   level >= 3 (coarser)    → 1 pass (Sprint 22 default)
         * The intermediate band is the second-finest + third-finest
         * uncoarsening projections — close enough to the finest level
         * that FM refinement has graph structure worth exploring, but
         * distant enough that Sprint 22's single-pass default
         * captured the cost-effective sweet spot until Sprint 23
         * Day 11's multi-pass exploration. */
        int passes;
        if (level == 0) {
            passes = finest_passes;
        } else if (level == 1 || level == 2) {
            passes = intermediate_passes;
        } else {
            passes = 1;
        }
        /* Sprint 26 Day 7: at the finest level (level == 0) under
         * SPARSE_FM_FINEST_STRATEGY=fifo, set the thread-local
         * pop-strategy override so graph_refine_fm uses
         * fm_bucket_pop_max_tail (FIFO; first-inserted wins) instead
         * of the default fm_bucket_pop_max (LIFO; most-recently-
         * inserted wins).  Restore after this level's passes finish. */
        int prev_pop_use_tail = fm_pop_use_tail;
        int prev_use_annealing = fm_use_annealing;
        fm_anneal_schedule_t prev_schedule = fm_anneal_schedule;
        int prev_anneal_pass_idx = fm_anneal_pass_idx;
        int prev_anneal_total_passes = fm_anneal_total_passes;
        int prev_use_thick_restart = fm_use_thick_restart;
        fm_thick_restart_perturb_t prev_thick_restart_perturb = fm_thick_restart_perturb;
        fm_gain_noise_schedule_t prev_gain_noise_schedule = fm_gain_noise_schedule;
        if (level == 0 && finest_strategy == FINEST_FM_FIFO)
            fm_pop_use_tail = 1;
        if (level == 0 && finest_strategy == FINEST_FM_ANNEALING) {
            fm_use_annealing = 1;
            fm_anneal_schedule = anneal_schedule_choice;
            fm_anneal_total_passes = passes;
        }
        if (level == 0 && finest_strategy == FINEST_FM_THICK_RESTART) {
            fm_use_thick_restart = 1;
            fm_thick_restart_perturb = thick_restart_perturb_choice;
            fm_anneal_total_passes = passes; /* reuse pass-count thread-local */
            fm_gain_noise_schedule = gain_noise_schedule_choice;
        }
        /* Sprint 27 Day 11: thick-restart anchor allocation.  Tracks
         * the global-best partition + cut across all passes at the
         * finest level.  Only allocated when fm_use_thick_restart is
         * active (level == 0 + strategy == thick_restart) AND
         * dst_graph->n >= 2 (n=0 / n=1 don't have meaningful
         * partitions to perturb).  Allocation failure falls through
         * to the standard pass loop without thick-restart wrapping —
         * fm_use_thick_restart's behaviour collapses to baseline FM
         * (which is still a valid degraded mode). */
        idx_t *tr_anchor_part = NULL;
        idx_t tr_anchor_cut = 0;
        uint32_t tr_rng = 0;
        const int tr_active = (fm_use_thick_restart && dst_graph->n >= 2);
        if (tr_active) {
            tr_anchor_part = malloc((size_t)dst_graph->n * sizeof(idx_t));
            if (tr_anchor_part) {
                memcpy(tr_anchor_part, next, (size_t)dst_graph->n * sizeof(idx_t));
                tr_anchor_cut = compute_cut_weight(dst_graph, tr_anchor_part);
                /* Per-call deterministic seed: same xorshift32-state
                 * recipe as Day 6 annealing.  Non-zero by construction. */
                tr_rng = (uint32_t)(((uint64_t)(uint32_t)dst_graph->n * 31U +
                                     (uint32_t)(uint64_t)(unsigned long)passes) *
                                        2654435761U +
                                    1U);
            }
        }

        /* Sprint 28 Day 4: ensemble buffers.  Three n-sized partition
         * arrays — `ensemble_start` snapshots the per-pass starting
         * state (carried forward from prior passes), `ensemble_working`
         * is the per-strategy FM scratch, `ensemble_best` holds the
         * lowest-cut partition seen across the K strategies for the
         * current pass.  Allocated only at level==0 under ENSEMBLE
         * mode + n >= 1; allocation failure degrades to baseline FM
         * (matching the tr_anchor failure mode above).  See
         * docs/planning/EPIC_2/SPRINT_28/ensemble_fm_design.md. */
        idx_t *ensemble_start = NULL;
        idx_t *ensemble_working = NULL;
        idx_t *ensemble_best = NULL;
        const int ens_active =
            (level == 0 && finest_strategy == FINEST_FM_ENSEMBLE && dst_graph->n >= 1);
        if (ens_active) {
            ensemble_start = malloc((size_t)dst_graph->n * sizeof(idx_t));
            ensemble_working = malloc((size_t)dst_graph->n * sizeof(idx_t));
            ensemble_best = malloc((size_t)dst_graph->n * sizeof(idx_t));
            if (!ensemble_start || !ensemble_working || !ensemble_best) {
                /* Degrade to baseline FM if any buffer fails. */
                free(ensemble_start);
                free(ensemble_working);
                free(ensemble_best);
                ensemble_start = NULL;
                ensemble_working = NULL;
                ensemble_best = NULL;
            }
        }

        for (int p = 0; p < passes; p++) {
            /* Sprint 27 Day 6: per-pass annealing index.  Set
             * unconditionally so a future caller that enables
             * annealing mid-uncoarsening sees a sensible default;
             * graph_refine_fm only consults fm_anneal_pass_idx when
             * fm_use_annealing == 1.  Sprint 27 Day 10/11: thick-
             * restart also threads pass index for the perturbation
             * RNG advance (only fires for p > 0). */
            fm_anneal_pass_idx = p;
            /* Sprint 27 Day 11: thick-restart restart-from-anchor.
             * Pass 0 starts from `next` as projected from the coarser
             * level (baseline behaviour).  Passes p > 0 copy the
             * global-best anchor back into `next` and apply a
             * perturbation (random_flip / boundary_shuffle / gauss_noise)
             * before the FM walk.  This re-explores the cut landscape
             * from the saved anchor instead of building only on the
             * previous pass's result. */
            if (tr_active && tr_anchor_part && p > 0) {
                memcpy(next, tr_anchor_part, (size_t)dst_graph->n * sizeof(idx_t));
                thick_restart_perturb(dst_graph, next, fm_thick_restart_perturb, &tr_rng);
            }

            /* Sprint 28 Day 4: multi-strategy FM ensemble dispatch.
             * For each strategy in the parsed selector list, reset
             * the FM thread-locals to defaults, set the strategy's
             * specific overrides, clone the partition start state
             * into the working buffer, run graph_refine_fm, score
             * the resulting cut, track the lowest-cut partition.
             * After all strategies finish, copy the winner back into
             * `next` for the next pass.  Single-strategy path below
             * is bypassed via `continue` when this branch fires. */
            if (ens_active && ensemble_start && ensemble_working && ensemble_best) {
                memcpy(ensemble_start, next, (size_t)dst_graph->n * sizeof(idx_t));
                idx_t best_cut = 0;
                int best_strat_idx = 0;
                for (int s = 0; s < ensemble_strategy_count; s++) {
                    int strat = ensemble_strategy_list[s];
                    /* Reset to defaults (cleared between strategies). */
                    fm_pop_use_tail = 0;
                    fm_use_annealing = 0;
                    fm_use_thick_restart = 0;
                    /* Set strategy-specific overrides.  `baseline`
                     * keeps the defaults; `thick_restart` is skipped
                     * by the parser so doesn't appear here. */
                    if (strat == FINEST_FM_FIFO) {
                        fm_pop_use_tail = 1;
                    } else if (strat == FINEST_FM_ANNEALING) {
                        fm_use_annealing = 1;
                        fm_anneal_schedule = anneal_schedule_choice;
                        fm_anneal_total_passes = passes;
                    }
                    /* Clone start state into the working buffer. */
                    memcpy(ensemble_working, ensemble_start, (size_t)dst_graph->n * sizeof(idx_t));
                    sparse_err_t rc = graph_refine_fm(dst_graph, ensemble_working);
                    if (rc != SPARSE_OK) {
                        free(ensemble_start);
                        free(ensemble_working);
                        free(ensemble_best);
                        free(tr_anchor_part);
                        fm_pop_use_tail = prev_pop_use_tail;
                        fm_use_annealing = prev_use_annealing;
                        fm_anneal_schedule = prev_schedule;
                        fm_anneal_pass_idx = prev_anneal_pass_idx;
                        fm_anneal_total_passes = prev_anneal_total_passes;
                        fm_use_thick_restart = prev_use_thick_restart;
                        fm_thick_restart_perturb = prev_thick_restart_perturb;
                        fm_gain_noise_schedule = prev_gain_noise_schedule;
                        free(cur);
                        free(next);
                        return rc;
                    }
                    idx_t cur_cut = compute_cut_weight(dst_graph, ensemble_working);
                    int is_winner = (s == 0) || (cur_cut < best_cut);
                    if (is_winner) {
                        best_cut = cur_cut;
                        best_strat_idx = s;
                        memcpy(ensemble_best, ensemble_working,
                               (size_t)dst_graph->n * sizeof(idx_t));
                    }
                    if (ensemble_debug) {
                        /* `best_so_far` reflects the state at the moment
                         * this strategy ran — multiple per-pass rows can
                         * report best_so_far=1 if a later strategy beats
                         * an earlier one.  To identify the FINAL winner
                         * for a pass, find the highest-index row with
                         * best_so_far=1 (or filter on `pass` and pick
                         * the max-`s` best_so_far=1).  Naming reflects
                         * the running semantic; the older `won` label
                         * implied final ownership which was misleading
                         * (PR #36 review). */
                        fprintf(stderr,
                                "fm-ensemble-debug n=%d pass=%d s=%d strat=%d cut=%d "
                                "best_so_far=%d\n",
                                (int)dst_graph->n, p, s, strat, (int)cur_cut,
                                (s == best_strat_idx) ? 1 : 0);
                    }
                }
                memcpy(next, ensemble_best, (size_t)dst_graph->n * sizeof(idx_t));
                continue; /* skip the single-strategy graph_refine_fm below */
            }

            sparse_err_t rc = graph_refine_fm(dst_graph, next);
            if (rc != SPARSE_OK) {
                free(ensemble_start);
                free(ensemble_working);
                free(ensemble_best);
                free(tr_anchor_part);
                fm_pop_use_tail = prev_pop_use_tail;
                fm_use_annealing = prev_use_annealing;
                fm_anneal_schedule = prev_schedule;
                fm_anneal_pass_idx = prev_anneal_pass_idx;
                fm_anneal_total_passes = prev_anneal_total_passes;
                fm_use_thick_restart = prev_use_thick_restart;
                fm_thick_restart_perturb = prev_thick_restart_perturb;
                fm_gain_noise_schedule = prev_gain_noise_schedule;
                free(cur);
                free(next);
                return rc;
            }
            /* Sprint 27 Day 11: end-of-pass best-cut update.  Compare
             * this pass's cut to the saved global best; if better,
             * promote `next` to the new anchor.  This is the
             * "thick-restart globally-best-tracking" contract that
             * differentiates from Sprint 23 Day 11's per-pass rollback. */
            if (tr_active && tr_anchor_part) {
                idx_t cur_cut = compute_cut_weight(dst_graph, next);
                if (cur_cut < tr_anchor_cut) {
                    memcpy(tr_anchor_part, next, (size_t)dst_graph->n * sizeof(idx_t));
                    tr_anchor_cut = cur_cut;
                }
            }
        }
        /* Sprint 27 Day 11: at end-of-passes, restore the global-best
         * anchor as the final output (in case the last pass landed on
         * a worse cut than an earlier pass). */
        if (tr_active && tr_anchor_part) {
            memcpy(next, tr_anchor_part, (size_t)dst_graph->n * sizeof(idx_t));
            if (getenv("SPARSE_FM_THICK_RESTART_DEBUG")) {
                fprintf(stderr,
                        "fm-thick-restart-debug n=%d passes=%d perturb=%d "
                        "best_cut=%d\n",
                        (int)dst_graph->n, passes, (int)fm_thick_restart_perturb,
                        (int)tr_anchor_cut);
            }
        }
        free(tr_anchor_part);
        free(ensemble_start);
        free(ensemble_working);
        free(ensemble_best);
        fm_pop_use_tail = prev_pop_use_tail;
        fm_use_annealing = prev_use_annealing;
        fm_anneal_schedule = prev_schedule;
        fm_anneal_pass_idx = prev_anneal_pass_idx;
        fm_anneal_total_passes = prev_anneal_total_passes;
        fm_use_thick_restart = prev_use_thick_restart;
        fm_thick_restart_perturb = prev_thick_restart_perturb;
        fm_gain_noise_schedule = prev_gain_noise_schedule;
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

/* Sprint 26 Day 10: separator-lift strategy enum.  Day 10 extends
 * Sprint 24 Day 6's two-value scheme with a third `per_vertex` value:
 * score boundary vertices individually + greedily pick top-K
 * regardless of side (vs the Sprint 22 / 24 side-then-lift heuristics
 * which lift one entire side's boundary).  See
 * `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_design.md`. */
typedef enum {
    SEP_LIFT_SMALLER_WEIGHT = 0,    /* Sprint 22 default — METIS convention */
    SEP_LIFT_BALANCED_BOUNDARY = 1, /* Sprint 24 Day 6 advisory */
    /* Sprint 26 Day 10/12 — per-vertex score + top-K.  Three preset
     * weight schemes per PLAN.md Day 12 task 1: hybrid (default;
     * cross_deg-priority + balance tie-break — Day 10's formula),
     * balance (balance-priority; balance-bonus dominates), degree
     * (low-total-degree priority + balance tie-break).  All three
     * use the same greedy 70/30-balance-respecting top-K selection;
     * only the score formula differs. */
    SEP_LIFT_PER_VERTEX_HYBRID = 2,  /* SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex */
    SEP_LIFT_PER_VERTEX_BALANCE = 3, /* SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_balance */
    SEP_LIFT_PER_VERTEX_DEGREE = 4,  /* SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_degree */
    /* Sprint 27 Day 4 — fixed-K termination instead of the
     * 70/30-balance gate.  K = min(boundary_count[0],
     * boundary_count[1]).  Stacks with the orthogonal
     * SPARSE_ND_SEP_LIFT_WEIGHT={hybrid (default), balance, degree}
     * axis to differentiate the three weight schemes (which Sprint
     * 26 Day 12 found bit-identical on 5 of 6 fixtures because the
     * 70/30 balance gate dominates).  See
     * `docs/planning/EPIC_2/SPRINT_27/per_vertex_fixed_k_decision.md`. */
    SEP_LIFT_PER_VERTEX_FIXED_K = 5, /* SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k */
} sep_lift_strategy_t;

/* Sprint 27 Day 4 — orthogonal weight-scheme axis for the fixed-K
 * variant.  Set via SPARSE_ND_SEP_LIFT_WEIGHT={hybrid, balance,
 * degree}; default hybrid.  Only consulted when strategy ==
 * SEP_LIFT_PER_VERTEX_FIXED_K (the existing per_vertex_* strategies
 * keep their hardcoded weight schemes for backward compatibility
 * with Sprint 26 advisory env-var users). */
typedef enum {
    SEP_LIFT_WEIGHT_HYBRID = 0,
    SEP_LIFT_WEIGHT_BALANCE = 1,
    SEP_LIFT_WEIGHT_DEGREE = 2,
} sep_lift_weight_t;

static sep_lift_strategy_t parse_sep_lift_strategy(void) {
    const char *env = getenv("SPARSE_ND_SEP_LIFT_STRATEGY");
    if (!env)
        return SEP_LIFT_SMALLER_WEIGHT;
    if (strcmp(env, "balanced_boundary") == 0)
        return SEP_LIFT_BALANCED_BOUNDARY;
    if (strcmp(env, "per_vertex") == 0)
        return SEP_LIFT_PER_VERTEX_HYBRID;
    if (strcmp(env, "per_vertex_balance") == 0)
        return SEP_LIFT_PER_VERTEX_BALANCE;
    if (strcmp(env, "per_vertex_degree") == 0)
        return SEP_LIFT_PER_VERTEX_DEGREE;
    if (strcmp(env, "per_vertex_fixed_k") == 0)
        return SEP_LIFT_PER_VERTEX_FIXED_K;
    /* Default + unrecognized + "smaller_weight" all fall through. */
    return SEP_LIFT_SMALLER_WEIGHT;
}

static sep_lift_weight_t parse_sep_lift_weight(void) {
    const char *env = getenv("SPARSE_ND_SEP_LIFT_WEIGHT");
    if (!env)
        return SEP_LIFT_WEIGHT_HYBRID;
    if (strcmp(env, "balance") == 0)
        return SEP_LIFT_WEIGHT_BALANCE;
    if (strcmp(env, "degree") == 0)
        return SEP_LIFT_WEIGHT_DEGREE;
    /* Default + unrecognized + "hybrid" all fall through. */
    return SEP_LIFT_WEIGHT_HYBRID;
}

/* Sprint 26 Day 12 / Sprint 27 Day 4: returns 1 if the strategy is
 * any per_vertex variant (hybrid / balance / degree / fixed_k).
 * Used to gate the per-vertex code path entry. */
static int is_per_vertex_strategy(sep_lift_strategy_t s) {
    return s == SEP_LIFT_PER_VERTEX_HYBRID || s == SEP_LIFT_PER_VERTEX_BALANCE ||
           s == SEP_LIFT_PER_VERTEX_DEGREE || s == SEP_LIFT_PER_VERTEX_FIXED_K;
}

/* Sprint 26 Day 10/12: qsort comparator for per-vertex separator
 * scoring.  Sorts boundary-vertex indices DESCENDING by score
 * (highest score first).  Score is computed by one of three formulas
 * (HYBRID / BALANCE / DEGREE) — see graph_edge_separator_to_vertex_separator.
 *
 * `score` is `int64_t` (PR #34 review fix; was `idx_t`/int32_t):
 * BALANCE and DEGREE schemes use `1000 * (...)` multipliers that can
 * overflow int32 on graphs with vertex degrees approaching ~2M, which
 * would corrupt the qsort ordering and make the comparator non-
 * transitive.  int64_t lifts the worst-case to ~9.2e18 — beyond any
 * plausible graph size in this codebase. */
typedef struct {
    idx_t vertex;
    int64_t score;
} per_vertex_score_t;

static int per_vertex_score_cmp_desc(const void *a, const void *b) {
    const per_vertex_score_t *pa = (const per_vertex_score_t *)a;
    const per_vertex_score_t *pb = (const per_vertex_score_t *)b;
    /* DESCENDING: higher score first.  Tie-break by lower vertex id
     * (deterministic when scores tie). */
    if (pa->score != pb->score)
        return (pa->score < pb->score) ? 1 : -1;
    if (pa->vertex != pb->vertex)
        return (pa->vertex > pb->vertex) ? 1 : -1;
    return 0;
}

sparse_err_t graph_edge_separator_to_vertex_separator(const sparse_graph_t *G, idx_t *part_io) {
    if (!G || !part_io)
        return SPARSE_ERR_NULL;
    if (G->n == 0)
        return SPARSE_OK;

    /* Side weights — Sprint 22 used these to pick the smaller-weight
     * side as the lift target (METIS convention).  Sprint 24 Day 6
     * adds a `balanced_boundary` strategy that picks the side with the
     * smaller boundary count regardless of weight.  Both strategies
     * compute the per-side boundary count up front so the strategy
     * choice is a cheap branch on the same intermediate. */
    idx_t w[2] = {0, 0};
    for (idx_t i = 0; i < G->n; i++) {
        idx_t wi = G->vwgt ? G->vwgt[i] : 1;
        if (part_io[i] == 0)
            w[0] += wi;
        else
            w[1] += wi;
    }

    /* Two-pass: first mark every boundary vertex on each side and
     * accumulate per-side boundary counts + boundary weight, then
     * pick the lift side under the configured strategy and move the
     * boundary marks for that side into part_io.  Splitting the
     * marking from the move keeps the boundary check simple — once we
     * start moving, "neighbour on other side" gets ambiguous. */
    int *is_boundary = calloc((size_t)G->n, sizeof(int));
    if (!is_boundary)
        return SPARSE_ERR_ALLOC;

    idx_t boundary_count[2] = {0, 0};
    idx_t boundary_weight[2] = {0, 0};
    for (idx_t i = 0; i < G->n; i++) {
        idx_t side = part_io[i];
        if (side != 0 && side != 1)
            continue;
        idx_t other = 1 - side;
        for (idx_t k = G->xadj[i]; k < G->xadj[i + 1]; k++) {
            idx_t j = G->adjncy[k];
            if (part_io[j] == other) {
                is_boundary[i] = 1;
                boundary_count[side]++;
                boundary_weight[side] += G->vwgt ? G->vwgt[i] : 1;
                break;
            }
        }
    }

    /* Strategy selection.  Default `smaller_weight` reproduces the
     * Sprint 22 Day 4 behaviour: lift the side with smaller vertex
     * weight.  `balanced_boundary` (Sprint 24 Day 6) lifts the side
     * with the smaller boundary count.  `per_vertex` (Sprint 26 Day
     * 10) scores each boundary vertex individually + greedily picks
     * top-K regardless of side, maintaining the 70/30 post-lift
     * balance check.  All non-default strategies fall back to
     * smaller_weight if the post-lift balance would be worse than
     * 70/30. */
    sep_lift_strategy_t strategy = parse_sep_lift_strategy();
    idx_t smaller_weight_side = (w[1] < w[0]) ? 1 : 0;
    idx_t lift_side = smaller_weight_side;
    int per_vertex_active = 0; /* Sprint 26 Day 10: 1 → use the
                                  per_vertex_lifted[] array below
                                  instead of per-side mass-lift. */
    int *per_vertex_lifted = NULL;

    if (strategy == SEP_LIFT_BALANCED_BOUNDARY) {
        /* Pick the smaller-boundary side; ties go to side 0 to
         * match the smaller_weight tie-break convention. */
        idx_t bb_side = (boundary_count[1] < boundary_count[0]) ? 1 : 0;
        idx_t lift_w = w[bb_side] - boundary_weight[bb_side];
        idx_t other_w = w[1 - bb_side];
        idx_t total_w = lift_w + other_w;
        int balanced = 1;
        if (total_w > 0) {
            idx_t max_w = (lift_w > other_w) ? lift_w : other_w;
            if ((int64_t)10 * (int64_t)max_w > (int64_t)7 * (int64_t)total_w)
                balanced = 0;
        }
        if (balanced)
            lift_side = bb_side;
    } else if (is_per_vertex_strategy(strategy)) {
        /* Sprint 26 Day 10/12 — per-vertex separator scoring with
         * three preset weight schemes.
         *
         * Score formulas (all compute cross_deg + total_deg + side
         * for each boundary vertex; combine via different weights):
         *   - HYBRID  (default per_vertex; Day 10): `2 * cross_deg + balance_bonus`
         *     — cross-degree-dominant; balance is tie-break.
         *   - BALANCE (per_vertex_balance; Day 12 task 1):
         *     `1000 * balance_bonus + cross_deg` — balance dominates;
         *     cross-degree is tie-break.  Expects Kuu-class wins
         *     (irregular SPDs where balanced_boundary already shines).
         *   - DEGREE  (per_vertex_degree; Day 12 task 1):
         *     `1000 * (max_deg - total_deg) + balance_bonus` — low
         *     total-degree dominates; balance is tie-break.  Expects
         *     regular-grid wins by avoiding high-degree separator
         *     vertices.
         *
         * (max_deg in DEGREE is the maximum degree across all boundary
         * vertices on this graph level — used to reverse the sort
         * direction without a negative-int hack.)
         *
         * Selection: sort all boundary vertices by score descending,
         * greedily lift one-by-one while maintaining 70/30 post-lift
         * weight balance.  Stop on imbalance violation; if K=0 (can't
         * lift anything safely), fall back to smaller_weight via the
         * existing lift_side = smaller_weight_side default below.
         *
         * See SPRINT_26/per_vertex_sep_design.md for the full rationale
         * + Day 12 sweep dimensions. */
        idx_t total_boundary = boundary_count[0] + boundary_count[1];
        if (total_boundary > 0) {
            per_vertex_score_t *scored =
                malloc((size_t)total_boundary * sizeof(per_vertex_score_t));
            if (!scored) {
                free(is_boundary);
                return SPARSE_ERR_ALLOC;
            }
            idx_t larger_side = (w[0] >= w[1]) ? 0 : 1;
            /* Sprint 27 Day 4: resolve the score-formula weight
             * scheme.  The four legacy per_vertex_* strategies hardcode
             * their weight; SEP_LIFT_PER_VERTEX_FIXED_K reads the
             * orthogonal SPARSE_ND_SEP_LIFT_WEIGHT axis. */
            sep_lift_weight_t weight;
            switch (strategy) {
            case SEP_LIFT_PER_VERTEX_HYBRID:
            default:
                weight = SEP_LIFT_WEIGHT_HYBRID;
                break;
            case SEP_LIFT_PER_VERTEX_BALANCE:
                weight = SEP_LIFT_WEIGHT_BALANCE;
                break;
            case SEP_LIFT_PER_VERTEX_DEGREE:
                weight = SEP_LIFT_WEIGHT_DEGREE;
                break;
            case SEP_LIFT_PER_VERTEX_FIXED_K:
                weight = parse_sep_lift_weight();
                break;
            }

            /* For DEGREE weight scheme: find max degree among boundary
             * vertices (one-pass pre-scan; small overhead vs the
             * boundary-walk below). */
            idx_t max_deg = 0;
            if (weight == SEP_LIFT_WEIGHT_DEGREE) {
                for (idx_t v = 0; v < G->n; v++) {
                    if (!is_boundary[v])
                        continue;
                    idx_t deg = G->xadj[v + 1] - G->xadj[v];
                    if (deg > max_deg)
                        max_deg = deg;
                }
            }
            idx_t bidx = 0;
            for (idx_t v = 0; v < G->n; v++) {
                if (!is_boundary[v])
                    continue;
                idx_t side = part_io[v];
                idx_t other = 1 - side;
                idx_t cross_deg = 0;
                for (idx_t k = G->xadj[v]; k < G->xadj[v + 1]; k++) {
                    idx_t j = G->adjncy[k];
                    if (part_io[j] == other)
                        cross_deg++;
                }
                idx_t balance_bonus = (side == larger_side) ? 1 : 0;
                /* PR #34 review fix: compute multiplications in int64
                 * before assigning to `score`.  BALANCE / DEGREE schemes'
                 * `1000 * (...)` multipliers can overflow int32 on
                 * graphs with vertex degrees approaching ~2M, which
                 * would corrupt qsort ordering. */
                int64_t score = 0;
                switch (weight) {
                case SEP_LIFT_WEIGHT_HYBRID:
                default:
                    /* cross_deg dominant; balance tie-break. */
                    score = (int64_t)2 * (int64_t)cross_deg + (int64_t)balance_bonus;
                    break;
                case SEP_LIFT_WEIGHT_BALANCE:
                    /* balance dominant; cross_deg tie-break. */
                    score = (int64_t)1000 * (int64_t)balance_bonus + (int64_t)cross_deg;
                    break;
                case SEP_LIFT_WEIGHT_DEGREE: {
                    /* low total-degree dominant; balance tie-break. */
                    idx_t deg = G->xadj[v + 1] - G->xadj[v];
                    score = (int64_t)1000 * (int64_t)(max_deg - deg) + (int64_t)balance_bonus;
                    break;
                }
                }
                scored[bidx].vertex = v;
                scored[bidx].score = score;
                bidx++;
            }
            /* Sort descending by score. */
            qsort(scored, (size_t)total_boundary, sizeof(per_vertex_score_t),
                  per_vertex_score_cmp_desc);

            /* Sprint 27 Day 4: termination predicate split.  The four
             * legacy per_vertex_* strategies use the dynamic-K
             * 70/30-balance gate (Sprint 26 Day 10 contract).
             * SEP_LIFT_PER_VERTEX_FIXED_K terminates after exactly
             * K = min(boundary_count[0], boundary_count[1]) iterations
             * regardless of balance state — Sprint 26 Day 12 found the
             * 70/30 gate fires early enough that the three weight
             * schemes converge to bit-identical outputs on 5 of 6
             * fixtures (the score formula doesn't get to differentiate
             * before the gate stops the lift).  Fixed-K bypasses the
             * gate so the score formulas can express their character. */
            per_vertex_lifted = calloc((size_t)G->n, sizeof(int));
            if (!per_vertex_lifted) {
                free(scored);
                free(is_boundary);
                return SPARSE_ERR_ALLOC;
            }
            idx_t cur_w0 = w[0], cur_w1 = w[1];
            idx_t lifted_count = 0;
            const idx_t fixed_k_target =
                (boundary_count[0] < boundary_count[1]) ? boundary_count[0] : boundary_count[1];
            for (idx_t k = 0; k < total_boundary; k++) {
                idx_t v = scored[k].vertex;
                idx_t side = part_io[v];
                idx_t vw = G->vwgt ? G->vwgt[v] : 1;
                idx_t new_w0 = cur_w0;
                idx_t new_w1 = cur_w1;
                if (side == 0)
                    new_w0 -= vw;
                else
                    new_w1 -= vw;
                if (strategy == SEP_LIFT_PER_VERTEX_FIXED_K) {
                    if (lifted_count >= fixed_k_target)
                        break; /* fixed-K cap hit */
                } else {
                    idx_t total_w = new_w0 + new_w1;
                    if (total_w > 0) {
                        idx_t max_w = (new_w0 > new_w1) ? new_w0 : new_w1;
                        if ((int64_t)10 * (int64_t)max_w > (int64_t)7 * (int64_t)total_w)
                            break; /* would violate 70/30 — stop here */
                    }
                }
                per_vertex_lifted[v] = 1;
                cur_w0 = new_w0;
                cur_w1 = new_w1;
                lifted_count++;
            }
            free(scored);

            /* If we lifted at least one vertex, use the per-vertex
             * mask.  Otherwise fall back to smaller_weight (lift_side
             * already set above). */
            if (lifted_count > 0) {
                per_vertex_active = 1;
            } else {
                free(per_vertex_lifted);
                per_vertex_lifted = NULL;
            }
        }
    }

    if (per_vertex_active) {
        for (idx_t i = 0; i < G->n; i++) {
            if (per_vertex_lifted[i])
                part_io[i] = 2;
        }
        free(per_vertex_lifted);
    } else {
        for (idx_t i = 0; i < G->n; i++) {
            if (is_boundary[i] && part_io[i] == lift_side)
                part_io[i] = 2;
        }
    }

    free(is_boundary);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_graph_partition — remaining multilevel orchestration layer.
 * ═══════════════════════════════════════════════════════════════════════
 *
 * After Sprint 43 Phase 1 extraction, this remaining layer owns only
 * the cross-module sequencing:
 *   1. hierarchy build from `src/sparse_graph_coarsen.c`
 *   2. coarsest split from `src/sparse_graph_bisect.c`
 *   3. FM/uncoarsening + separator lifting in this file
 *   4. sep=0 retry policy via the coarsening module's override seam
 */
static const sparse_graph_t *graph_hierarchy_coarsest(const sparse_graph_t *root,
                                                      const sparse_graph_hierarchy_t *h) {
    return (h->nlevels > 0) ? &h->coarse[h->nlevels - 1] : root;
}

static sparse_err_t graph_partition_seed_coarsest(const sparse_graph_t *coarsest,
                                                  idx_t **coarsest_part_out) {
    idx_t *coarsest_part = malloc((size_t)coarsest->n * sizeof(idx_t));
    if (!coarsest_part)
        return SPARSE_ERR_ALLOC;

    sparse_err_t rc = graph_bisect_coarsest(coarsest, coarsest_part);
    if (rc == SPARSE_OK)
        rc = graph_refine_fm(coarsest, coarsest_part);
    if (rc != SPARSE_OK) {
        free(coarsest_part);
        return rc;
    }

    *coarsest_part_out = coarsest_part;
    return SPARSE_OK;
}

static idx_t graph_partition_count_separator_vertices(const sparse_graph_t *G, const idx_t *part) {
    idx_t sep = 0;
    for (idx_t i = 0; i < G->n; i++) {
        if (part[i] == 2)
            sep++;
    }
    return sep;
}

static int graph_partition_should_retry_with_forced_hem(idx_t sep) {
    return sep == 0 && sparse_graph_coarsening_strategy_current() != COARSENING_HEAVY_EDGE;
}

/* Sprint 26 Day 3: extracted partition body so `sparse_graph_partition`
 * can call it twice — once with the configured strategy, and (if the
 * first pass produces a degenerate sep=0) once more with the
 * `force_hem_override` set.  See `SPRINT_26/hcc_sep_zero_diagnosis.md`. */
static sparse_err_t partition_once(const sparse_graph_t *G, idx_t *part_out, idx_t *sep_out) {
    sparse_graph_hierarchy_t h = {0};
    sparse_err_t rc = sparse_graph_hierarchy_build(G, /*seed=*/0U, &h);
    if (rc != SPARSE_OK)
        return rc;

    const sparse_graph_t *coarsest = graph_hierarchy_coarsest(G, &h);
    /* `graph_bisect_coarsest` handles any `n` (brute force ≤ 20, GGGP
     * above) so we don't need a coarsest-size cap here.  The hierarchy
     * may stop above the 20-vertex target on inputs where heavy-edge
     * matching saturates (e.g. bcsstk14) — GGGP just bisects whatever
     * the hierarchy delivers, and Day 4's per-level FM polishes the
     * uncoarsened result. */

    idx_t *coarsest_part = NULL;
    rc = graph_partition_seed_coarsest(coarsest, &coarsest_part);
    if (rc != SPARSE_OK) {
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

    *sep_out = graph_partition_count_separator_vertices(G, part_out);
    return SPARSE_OK;
}

sparse_err_t sparse_graph_partition(const sparse_graph_t *G, idx_t *part_out, idx_t *sep_out) {
    if (!G || !part_out)
        return SPARSE_ERR_NULL;
    if (sep_out)
        *sep_out = 0;
    if (G->n == 0)
        return SPARSE_OK;

    idx_t sep = 0;
    sparse_err_t rc = partition_once(G, part_out, &sep);
    if (rc != SPARSE_OK)
        return rc;

    /* Sprint 26 Day 3: sep=0 fall-back. If the first pass produced a
     * degenerate empty separator AND the configured strategy was HCC
     * (so HEM hasn't already been tried), force HEM through the
     * coarsening module's internal override seam and re-run the
     * multilevel pipeline. */
    if (graph_partition_should_retry_with_forced_hem(sep)) {
        sparse_graph_force_hem_override_begin();
        rc = partition_once(G, part_out, &sep);
        sparse_graph_force_hem_override_end();
        if (rc != SPARSE_OK)
            return rc;
    }

    if (sep_out)
        *sep_out = sep;
    return SPARSE_OK;
}
