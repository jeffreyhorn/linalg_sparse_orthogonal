/*
 * sparse_graph.c — Remaining uncoarsening / separator / orchestration
 *                  slice of the Sprint 22 multilevel graph partitioner.
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
 *   - FM refinement now lives in
 *     `src/sparse_graph_refine.c`
 *   - this file intentionally retains:
 *       - uncoarsening
 *       - separator lifting
 *       - top-level partition orchestration
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

/* Sprint 44 Day 5 extraction note:
 *   - FM refinement, bucket operations, FM parser helpers, cut-weight
 *     evaluation, and FM thread-local runtime state now live in
 *     `src/sparse_graph_refine.c`
 *   - this file intentionally begins at uncoarsening / separator /
 *     orchestration ownership only
 */

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
    fm_anneal_schedule_t anneal_schedule_choice = sparse_graph_parse_fm_anneal_schedule();
    fm_thick_restart_perturb_t thick_restart_perturb_choice =
        sparse_graph_parse_fm_thick_restart_perturb();
    /* Sprint 28 Day 2: gain-noise schedule for the formal thick-restart
     * variant.  Only consulted by graph_refine_fm when
     * fm_thick_restart_perturb == GAIN_NOISE_FORMAL; defaults to
     * linear so the default-off code path stays bit-identical. */
    fm_gain_noise_schedule_t gain_noise_schedule_choice =
        sparse_graph_parse_fm_gain_noise_schedule();

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
        sparse_graph_fm_runtime_t prev_runtime = {0};
        sparse_graph_fm_runtime_t runtime = {0};
        sparse_graph_fm_runtime_get(&prev_runtime);
        runtime = prev_runtime;
        if (level == 0 && finest_strategy == FINEST_FM_FIFO)
            runtime.pop_use_tail = 1;
        if (level == 0 && finest_strategy == FINEST_FM_ANNEALING) {
            runtime.use_annealing = 1;
            runtime.anneal_schedule = anneal_schedule_choice;
            runtime.anneal_total_passes = passes;
        }
        if (level == 0 && finest_strategy == FINEST_FM_THICK_RESTART) {
            runtime.use_thick_restart = 1;
            runtime.thick_restart_perturb = thick_restart_perturb_choice;
            runtime.anneal_total_passes = passes;
            runtime.gain_noise_schedule = gain_noise_schedule_choice;
        }
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
                for (int s = 0; s < ensemble_strategy_count; s++) {
                    int strat = ensemble_strategy_list[s];
                    /* Reset to defaults (cleared between strategies). */
                    sparse_graph_fm_runtime_t strategy_runtime = prev_runtime;
                    strategy_runtime.pop_use_tail = 0;
                    strategy_runtime.use_annealing = 0;
                    strategy_runtime.use_thick_restart = 0;
                    strategy_runtime.anneal_pass_idx = p;
                    strategy_runtime.anneal_total_passes = passes;
                    strategy_runtime.anneal_schedule = anneal_schedule_choice;
                    strategy_runtime.thick_restart_perturb = thick_restart_perturb_choice;
                    strategy_runtime.gain_noise_schedule = gain_noise_schedule_choice;
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
            if (getenv("SPARSE_FM_THICK_RESTART_DEBUG")) {
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
