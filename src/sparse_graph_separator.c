/*
 * sparse_graph_separator.c — Separator-lifting policy and edge-to-
 *                            vertex separator conversion.
 *
 * Phase-2 ownership:
 *   - separator strategy / weight enums
 *   - separator policy parsers
 *   - per-vertex separator scoring helpers
 *   - graph_edge_separator_to_vertex_separator(...)
 *
 * The top-level partition pipeline still lives in `src/sparse_graph.c`;
 * this file owns only the final separator-policy and conversion seam.
 */

#include "sparse_alloc_internal.h"
#include "sparse_graph_internal.h"

#include <stdlib.h>
#include <string.h>

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
    return SEP_LIFT_WEIGHT_HYBRID;
}

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

    idx_t w[2] = {0, 0};
    for (idx_t i = 0; i < G->n; i++) {
        idx_t wi = G->vwgt ? G->vwgt[i] : 1;
        if (part_io[i] == 0)
            w[0] += wi;
        else
            w[1] += wi;
    }

    int *is_boundary = NULL;
    sparse_err_t rc = sparse_calloc_idx_array(G->n, sizeof(*is_boundary), (void **)&is_boundary);
    if (rc != SPARSE_OK)
        return rc;

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

    sep_lift_strategy_t strategy = parse_sep_lift_strategy();
    idx_t smaller_weight_side = (w[1] < w[0]) ? 1 : 0;
    idx_t lift_side = smaller_weight_side;
    int per_vertex_active = 0;
    int *per_vertex_lifted = NULL;

    if (strategy == SEP_LIFT_BALANCED_BOUNDARY) {
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
        idx_t total_boundary = boundary_count[0] + boundary_count[1];
        if (total_boundary > 0) {
            per_vertex_score_t *scored = NULL;
            rc = sparse_malloc_idx_array(total_boundary, sizeof(*scored), (void **)&scored);
            if (rc != SPARSE_OK) {
                free(is_boundary);
                return rc;
            }
            idx_t larger_side = (w[0] >= w[1]) ? 0 : 1;
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
                int64_t score = 0;
                switch (weight) {
                case SEP_LIFT_WEIGHT_HYBRID:
                default:
                    score = (int64_t)2 * (int64_t)cross_deg + (int64_t)balance_bonus;
                    break;
                case SEP_LIFT_WEIGHT_BALANCE:
                    score = (int64_t)1000 * (int64_t)balance_bonus + (int64_t)cross_deg;
                    break;
                case SEP_LIFT_WEIGHT_DEGREE: {
                    idx_t deg = G->xadj[v + 1] - G->xadj[v];
                    score = (int64_t)1000 * (int64_t)(max_deg - deg) + (int64_t)balance_bonus;
                    break;
                }
                }
                scored[bidx].vertex = v;
                scored[bidx].score = score;
                bidx++;
            }
            qsort(scored, (size_t)total_boundary, sizeof(per_vertex_score_t),
                  per_vertex_score_cmp_desc);

            rc = sparse_calloc_idx_array(G->n, sizeof(*per_vertex_lifted),
                                         (void **)&per_vertex_lifted);
            if (rc != SPARSE_OK) {
                free(scored);
                free(is_boundary);
                return rc;
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
                        break;
                } else {
                    idx_t total_w = new_w0 + new_w1;
                    if (total_w > 0) {
                        idx_t max_w = (new_w0 > new_w1) ? new_w0 : new_w1;
                        if ((int64_t)10 * (int64_t)max_w > (int64_t)7 * (int64_t)total_w)
                            break;
                    }
                }
                per_vertex_lifted[v] = 1;
                cur_w0 = new_w0;
                cur_w1 = new_w1;
                lifted_count++;
            }
            free(scored);

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
