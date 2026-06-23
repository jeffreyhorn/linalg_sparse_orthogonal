/*
 * sparse_graph.c — Remaining uncoarsening / orchestration slice of the
 *                  Sprint 22 multilevel graph partitioner.
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
 * **Small-graph base case.**  Sprint 22 Day 6's recursion originally
 * stopped when a subgraph had n ≤ `sparse_reorder_nd_base_threshold`
 * and emitted the subgraph's vertices in natural order; Sprint 23
 * replaced that leaf path with quotient-graph AMD, and Sprint 86 Day 6
 * raised the current default threshold to 160 after a reviewed-runtime
 * re-sweep.  The partitioner itself doesn't impose this threshold —
 * it's an ND-driver decision — but the brute-force bisection at the
 * coarsest level gives the partitioner its own micro-fast-path for
 * n ≤ 20.
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
 *   - FM refinement now lives in
 *     `src/sparse_graph_refine.c`
 *   - separator lifting now lives in
 *     `src/sparse_graph_separator.c`
 *   - this file intentionally retains:
 *       - uncoarsening
 *       - top-level partition orchestration
 *       - retry / fallback glue
 *
 * That split keeps the current extraction phase bounded while
 * preserving the original multilevel partition contract consumed by
 * `src/sparse_reorder_nd.c`.
 */

#include "sparse_alloc_internal.h"
#include "sparse_graph_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Sprint 44 Day 6 extraction note:
 *   - FM refinement, bucket operations, FM parser helpers, cut-weight
 *     evaluation, and FM thread-local runtime state now live in
 *     `src/sparse_graph_refine.c`
 *   - separator policy / lifting now lives in
 *     `src/sparse_graph_separator.c`
 *   - this file intentionally begins at uncoarsening / orchestration
 *     ownership only
 */

/* ═══════════════════════════════════════════════════════════════════════
 * Uncoarsening + residual orchestration runtime (Sprint 22 Day 4).
 * ═══════════════════════════════════════════════════════════════════════
 */

static int graph_parse_env_int_range(const char *name, int default_value, int min_value,
                                     int max_value) {
    const char *env = getenv(name);
    if (!env)
        return default_value;

    char *endp = NULL;
    long v = strtol(env, &endp, 10);
    if (env != endp && *endp == '\0' && v >= min_value && v <= max_value)
        return (int)v;
    return default_value;
}

static fm_anneal_schedule_t graph_parse_anneal_schedule_compat_override(void) {
    const char *env = getenv("SPARSE_FM_ANNEALING_SCHEDULE");
    if (!env)
        return FM_ANNEAL_SCHEDULE_EXPONENTIAL;
    if (strcmp(env, "linear") == 0)
        return FM_ANNEAL_SCHEDULE_LINEAR;
    if (strcmp(env, "cosine") == 0)
        return FM_ANNEAL_SCHEDULE_COSINE;
    return FM_ANNEAL_SCHEDULE_EXPONENTIAL;
}

static fm_thick_restart_perturb_t graph_parse_thick_restart_perturb_compat_override(void) {
    const char *env = getenv("SPARSE_FM_THICK_RESTART_PERTURB");
    if (!env)
        return FM_THICK_RESTART_PERTURB_RANDOM_FLIP;
    if (strcmp(env, "boundary_shuffle") == 0)
        return FM_THICK_RESTART_PERTURB_BOUNDARY_SHUFFLE;
    if (strcmp(env, "gauss_noise") == 0)
        return FM_THICK_RESTART_PERTURB_GAUSS_NOISE;
    if (strcmp(env, "gain_noise_formal") == 0)
        return FM_THICK_RESTART_PERTURB_GAIN_NOISE_FORMAL;
    return FM_THICK_RESTART_PERTURB_RANDOM_FLIP;
}

static fm_gain_noise_schedule_t graph_parse_gain_noise_schedule_compat_override(void) {
    const char *env = getenv("SPARSE_FM_GAIN_NOISE_SCHEDULE");
    if (!env)
        return FM_GAIN_NOISE_SCHEDULE_LINEAR;
    if (strcmp(env, "exponential") == 0)
        return FM_GAIN_NOISE_SCHEDULE_EXPONENTIAL;
    if (strcmp(env, "cosine") == 0)
        return FM_GAIN_NOISE_SCHEDULE_COSINE;
    return FM_GAIN_NOISE_SCHEDULE_LINEAR;
}

static finest_fm_strategy_t graph_parse_finest_strategy(void) {
    const char *env = getenv("SPARSE_FM_FINEST_STRATEGY");
    if (!env)
        return FINEST_FM_BASELINE;
    if (strcmp(env, "fifo") == 0)
        return FINEST_FM_FIFO;
    if (strcmp(env, "annealing") == 0)
        return FINEST_FM_ANNEALING;
    if (strcmp(env, "thick_restart") == 0)
        return FINEST_FM_THICK_RESTART;
    if (strcmp(env, "ensemble") == 0)
        return FINEST_FM_ENSEMBLE;
    return FINEST_FM_BASELINE;
}

static int graph_parse_ensemble_strategy_list(int out[4]) {
    const char *env = getenv("SPARSE_FM_ENSEMBLE_STRATEGIES");
    char buf[256];
    const char *list;
    int strategy_count = 0;

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
    if (list_len >= sizeof(buf))
        list_len = sizeof(buf) - 1;
    memcpy(buf, list, list_len);
    buf[list_len] = '\0';

    char *tok = buf;
    while (tok && *tok && strategy_count < 4) {
        char *comma = tok;
        while (*comma && *comma != ',')
            comma++;
        int has_more = (*comma == ',');
        if (has_more)
            *comma = '\0';
        while (*tok == ' ' || *tok == '\t')
            tok++;
        size_t tok_len = strlen(tok);
        while (tok_len > 0 &&
               (tok[tok_len - 1] == ' ' || tok[tok_len - 1] == '\t' || tok[tok_len - 1] == '\n')) {
            tok[--tok_len] = '\0';
        }

        int strat = -1;
        if (strcmp(tok, "baseline") == 0)
            strat = FINEST_FM_BASELINE;
        else if (strcmp(tok, "fifo") == 0)
            strat = FINEST_FM_FIFO;
        else if (strcmp(tok, "annealing") == 0)
            strat = FINEST_FM_ANNEALING;

        if (strat >= 0) {
            int dup = 0;
            for (int i = 0; i < strategy_count; i++) {
                if (out[i] == strat) {
                    dup = 1;
                    break;
                }
            }
            if (!dup)
                out[strategy_count++] = strat;
        }
        tok = has_more ? (comma + 1) : NULL;
    }

    if (strategy_count == 0) {
        out[0] = FINEST_FM_BASELINE;
        return 1;
    }
    return strategy_count;
}

static int graph_env_flag_enabled(const char *name) { return getenv(name) != NULL; }

static sparse_graph_fm_policy_t graph_fm_policy_from_compat_env(void) {
    sparse_graph_fm_policy_t policy = {
        .finest_passes = graph_parse_env_int_range("SPARSE_FM_FINEST_PASSES", 3, 1, 16),
        .intermediate_passes = graph_parse_env_int_range("SPARSE_FM_INTERMEDIATE_PASSES", 1, 1, 10),
        .finest_strategy = graph_parse_finest_strategy(),
        .anneal_schedule_choice = graph_parse_anneal_schedule_compat_override(),
        .thick_restart_perturb_choice = graph_parse_thick_restart_perturb_compat_override(),
        .gain_noise_schedule_choice = graph_parse_gain_noise_schedule_compat_override(),
        .ensemble_strategy_list = {0, 0, 0, 0},
        .ensemble_strategy_count = 0,
        .ensemble_debug = 0,
        .anneal_debug = 0,
        .gain_noise_debug = 0,
        .thick_restart_debug = 0,
    };

    if (policy.finest_strategy == FINEST_FM_ENSEMBLE) {
        policy.ensemble_strategy_count =
            graph_parse_ensemble_strategy_list(policy.ensemble_strategy_list);
        policy.ensemble_debug = graph_env_flag_enabled("SPARSE_FM_ENSEMBLE_DEBUG");
    }

    policy.anneal_debug = graph_env_flag_enabled("SPARSE_FM_ANNEALING_DEBUG");
    policy.gain_noise_debug = graph_env_flag_enabled("SPARSE_FM_GAIN_NOISE_DEBUG");
    policy.thick_restart_debug = graph_env_flag_enabled("SPARSE_FM_THICK_RESTART_DEBUG");

    return policy;
}

static int graph_uncoarsen_level_passes(int level, int finest_passes, int intermediate_passes) {
    if (level == 0)
        return finest_passes;
    if (level == 1 || level == 2)
        return intermediate_passes;
    return 1;
}

static sparse_graph_fm_runtime_t
graph_uncoarsen_runtime_for_level(const sparse_graph_fm_runtime_t *prev_runtime,
                                  const sparse_graph_fm_policy_t *policy, int level, int passes) {
    sparse_graph_fm_runtime_t runtime = *prev_runtime;
    if (level != 0)
        return runtime;

    if (policy->finest_strategy == FINEST_FM_FIFO)
        runtime.pop_use_tail = 1;
    if (policy->finest_strategy == FINEST_FM_ANNEALING) {
        runtime.use_annealing = 1;
        runtime.anneal_schedule = policy->anneal_schedule_choice;
        runtime.anneal_total_passes = passes;
        runtime.anneal_debug = policy->anneal_debug;
    }
    if (policy->finest_strategy == FINEST_FM_THICK_RESTART) {
        runtime.use_thick_restart = 1;
        runtime.thick_restart_perturb = policy->thick_restart_perturb_choice;
        runtime.anneal_total_passes = passes;
        runtime.gain_noise_schedule = policy->gain_noise_schedule_choice;
        runtime.gain_noise_debug = policy->gain_noise_debug;
        runtime.thick_restart_debug = policy->thick_restart_debug;
    }
    return runtime;
}

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

    sparse_graph_fm_policy_t policy = graph_fm_policy_from_compat_env();

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
    /* Sprint 27 Day 5 dispatch update: Day 5 lands the `annealing`
     * skeleton.  Sprint 26 Day 6's design rejected annealing on
     * cost grounds (20-50 % wall expansion); Sprint 26 Day 5's
     * `nd_base_threshold = 96` flip + Sprint 27 Day 3's = 128 flip
     * cumulatively cut Pres_Poisson ND wall 38 s → 7 s, making the
     * wall budget affordable.  Day 5 wires `fm_use_annealing` +
     * `fm_anneal_schedule` thread-locals; Day 6 lands the
     * acceptance-probability overlay + measurement.  `thick_restart`
     * stays unimplemented (Sprint 27 item 6 budget; Days 10-12). */
    /* Sprint 28 Day 2: gain-noise schedule for the formal thick-restart
     * variant.  Only consulted by graph_refine_fm when
     * fm_thick_restart_perturb == GAIN_NOISE_FORMAL; defaults to
     * linear so the default-off code path stays bit-identical. */
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
        int passes =
            graph_uncoarsen_level_passes(level, policy.finest_passes, policy.intermediate_passes);
        sparse_graph_fm_runtime_t prev_runtime = {0};
        sparse_graph_fm_runtime_get(&prev_runtime);
        sparse_graph_fm_runtime_t runtime =
            graph_uncoarsen_runtime_for_level(&prev_runtime, &policy, level, passes);
        sparse_graph_fm_runtime_set(&runtime);
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
        const int tr_active = (runtime.use_thick_restart && dst_graph->n >= 2);
        if (tr_active) {
            tr_anchor_part = malloc((size_t)dst_graph->n * sizeof(idx_t));
            if (tr_anchor_part) {
                memcpy(tr_anchor_part, next, (size_t)dst_graph->n * sizeof(idx_t));
                tr_anchor_cut = sparse_graph_compute_cut_weight(dst_graph, tr_anchor_part);
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
            (level == 0 && policy.finest_strategy == FINEST_FM_ENSEMBLE && dst_graph->n >= 1);
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
            runtime.anneal_pass_idx = p;
            sparse_graph_fm_runtime_set(&runtime);
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
                sparse_graph_thick_restart_perturb(dst_graph, next, runtime.thick_restart_perturb,
                                                   &tr_rng);
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
                for (int s = 0; s < policy.ensemble_strategy_count; s++) {
                    int strat = policy.ensemble_strategy_list[s];
                    /* Reset to defaults (cleared between strategies). */
                    sparse_graph_fm_runtime_t strategy_runtime = prev_runtime;
                    strategy_runtime.pop_use_tail = 0;
                    strategy_runtime.use_annealing = 0;
                    strategy_runtime.use_thick_restart = 0;
                    strategy_runtime.anneal_pass_idx = p;
                    strategy_runtime.anneal_total_passes = passes;
                    strategy_runtime.anneal_schedule = policy.anneal_schedule_choice;
                    strategy_runtime.thick_restart_perturb = policy.thick_restart_perturb_choice;
                    strategy_runtime.gain_noise_schedule = policy.gain_noise_schedule_choice;
                    strategy_runtime.anneal_debug = policy.anneal_debug;
                    strategy_runtime.gain_noise_debug = policy.gain_noise_debug;
                    strategy_runtime.thick_restart_debug = policy.thick_restart_debug;
                    /* Set strategy-specific overrides.  `baseline`
                     * keeps the defaults; `thick_restart` is skipped
                     * by the parser so doesn't appear here. */
                    if (strat == FINEST_FM_FIFO) {
                        strategy_runtime.pop_use_tail = 1;
                    } else if (strat == FINEST_FM_ANNEALING) {
                        strategy_runtime.use_annealing = 1;
                    }
                    sparse_graph_fm_runtime_set(&strategy_runtime);
                    /* Clone start state into the working buffer. */
                    memcpy(ensemble_working, ensemble_start, (size_t)dst_graph->n * sizeof(idx_t));
                    sparse_err_t rc = graph_refine_fm(dst_graph, ensemble_working);
                    if (rc != SPARSE_OK) {
                        free(ensemble_start);
                        free(ensemble_working);
                        free(ensemble_best);
                        free(tr_anchor_part);
                        sparse_graph_fm_runtime_set(&prev_runtime);
                        free(cur);
                        free(next);
                        return rc;
                    }
                    idx_t cur_cut = sparse_graph_compute_cut_weight(dst_graph, ensemble_working);
                    int is_winner = (s == 0) || (cur_cut < best_cut);
                    if (is_winner) {
                        best_cut = cur_cut;
                        best_strat_idx = s;
                        memcpy(ensemble_best, ensemble_working,
                               (size_t)dst_graph->n * sizeof(idx_t));
                    }
                    if (policy.ensemble_debug) {
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
                sparse_graph_fm_runtime_set(&prev_runtime);
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
                idx_t cur_cut = sparse_graph_compute_cut_weight(dst_graph, next);
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
            if (runtime.thick_restart_debug) {
                fprintf(stderr,
                        "fm-thick-restart-debug n=%d passes=%d perturb=%d "
                        "best_cut=%d\n",
                        (int)dst_graph->n, passes, (int)runtime.thick_restart_perturb,
                        (int)tr_anchor_cut);
            }
        }
        free(tr_anchor_part);
        free(ensemble_start);
        free(ensemble_working);
        free(ensemble_best);
        sparse_graph_fm_runtime_set(&prev_runtime);
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

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_graph_partition — remaining multilevel orchestration layer.
 * ═══════════════════════════════════════════════════════════════════════
 *
 * After Sprint 43 Phase 1 extraction, this remaining layer owns only
 * the cross-module sequencing:
 *   1. hierarchy build from `src/sparse_graph_coarsen.c`
 *   2. coarsest split from `src/sparse_graph_bisect.c`
 *   3. FM in `src/sparse_graph_refine.c`, uncoarsening in this file,
 *      and separator lifting in `src/sparse_graph_separator.c`
 *   4. sep=0 retry policy via the coarsening module's override seam
 */
static const sparse_graph_t *graph_hierarchy_coarsest(const sparse_graph_t *root,
                                                      const sparse_graph_hierarchy_t *h) {
    return (h->nlevels > 0) ? &h->coarse[h->nlevels - 1] : root;
}

static sparse_err_t graph_partition_seed_coarsest(const sparse_graph_t *coarsest,
                                                  idx_t **coarsest_part_out) {
    idx_t *coarsest_part = NULL;
    sparse_err_t alloc_rc =
        sparse_malloc_idx_array(coarsest->n, sizeof(idx_t), (void **)&coarsest_part);
    if (alloc_rc != SPARSE_OK)
        return alloc_rc;

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

/* Retry policy is keyed to the configured strategy surface, not the
 * effective per-call strategy that graph_coarsen_with_strategy() may
 * choose after HCC's internal CV-based fall-through. A sep==0 first
 * pass under configured HCC still gets one explicit forced-HEM rerun
 * through the override seam below. */
static int graph_partition_should_retry_with_forced_hem(idx_t sep) {
    return sep == 0 && sparse_graph_coarsening_strategy_current() != COARSENING_HEAVY_EDGE;
}

/* Sprint 26 Day 3: extracted partition body so `sparse_graph_partition`
 * can call it twice — once with the configured strategy, and (if the
 * first pass produces a degenerate sep=0 under non-HEM configured
 * policy) once more with the `force_hem_override` set.  See
 * `SPRINT_26/hcc_sep_zero_diagnosis.md`. */
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
     * degenerate empty separator under non-HEM configured policy,
     * force HEM through the coarsening module's internal override
     * seam and re-run the multilevel pipeline. This check is keyed to
     * the configured strategy surface; HCC may already have fallen
     * through to HEM internally on the first pass, but the explicit
     * retry contract still hangs off the non-HEM configuration path. */
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
