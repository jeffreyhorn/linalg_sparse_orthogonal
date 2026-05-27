#include "sparse_graph_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Heavy-edge-matching coarsener (Sprint 22 Day 2).
 * ═══════════════════════════════════════════════════════════════════════
 *
 * splitmix64 PRNG (well-known, public-domain) — same generator used
 * by SplittableRandom and many embedded engines. Stable across
 * compilers / platforms, so `(graph, seed)` deterministically yields
 * the same coarsened graph everywhere.
 */
static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* In-place Fisher-Yates shuffle of `perm[0..n-1]` (initially the
 * identity). */
static void fisher_yates_shuffle(idx_t *perm, idx_t n, uint32_t seed) {
    uint64_t state = (uint64_t)seed * 0x9E3779B97F4A7C15ULL + 1;
    for (idx_t i = 0; i < n; i++)
        perm[i] = i;
    for (idx_t i = n - 1; i > 0; i--) {
        uint64_t r = splitmix64_next(&state);
        idx_t j = (idx_t)(r % (uint64_t)(i + 1));
        idx_t tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
}

/* Comparator for the (neighbour, weight) pair used by the per-coarse-
 * vertex sort+merge dedup pass. */
typedef struct {
    idx_t nbr;
    idx_t wt;
} coarse_edge_t;

static int cmp_coarse_edge(const void *a, const void *b) {
    idx_t na = ((const coarse_edge_t *)a)->nbr;
    idx_t nb = ((const coarse_edge_t *)b)->nbr;
    return (na > nb) - (na < nb);
}

/* Sprint 26 Day 3: thread-local override for the sep=0 fall-back.
 * `sparse_graph_partition` forces HEM temporarily through the
 * begin/end helpers below before retrying a degenerate partition.
 * `_Thread_local` keeps concurrent partition calls race-free. */
static _Thread_local int force_hem_override = 0;

/* Parser for the active coarsening strategy. Exposed through
 * `sparse_graph_coarsening_strategy_current()` so the remaining
 * orchestration layer can query the current strategy without owning
 * the parser implementation directly. */
static coarsening_strategy_t parse_coarsening_strategy(void) {
    if (force_hem_override)
        return COARSENING_HEAVY_EDGE;
    const char *env = getenv("SPARSE_ND_COARSENING");
    if (env && strcmp(env, "heavy_edge") == 0)
        return COARSENING_HEAVY_EDGE;
    if (env && strcmp(env, "hcc") == 0)
        return COARSENING_HCC;
    return COARSENING_HCC;
}

coarsening_strategy_t sparse_graph_coarsening_strategy_current(void) {
    return parse_coarsening_strategy();
}

void sparse_graph_force_hem_override_begin(void) { force_hem_override = 1; }

void sparse_graph_force_hem_override_end(void) { force_hem_override = 0; }

/* Sprint 25 Day 2: strategy-parameterized coarsening core. Both
 * graph_coarsen_heavy_edge_matching (Sprint 22) and graph_coarsen_hcc
 * (Sprint 25) call this with their respective strategy. Only the
 * matching-loop's score function + tie-break differ; the
 * graph-construction passes (vwgt aggregation, deg counting, sort+merge
 * dedup, compaction) are identical and shared. */
static sparse_err_t graph_coarsen_with_strategy(const sparse_graph_t *fine, uint32_t seed,
                                                coarsening_strategy_t strategy,
                                                sparse_graph_t *coarse_out, idx_t *cmap_out) {
    if (!fine || !coarse_out)
        return SPARSE_ERR_NULL;

    coarse_out->n = 0;
    coarse_out->xadj = NULL;
    coarse_out->adjncy = NULL;
    coarse_out->vwgt = NULL;
    coarse_out->ewgt = NULL;

    if (fine->n > 0 && !cmap_out)
        return SPARSE_ERR_NULL;

    if (fine->n == 0) {
        coarse_out->xadj = malloc(sizeof(idx_t));
        if (!coarse_out->xadj)
            return SPARSE_ERR_ALLOC;
        coarse_out->xadj[0] = 0;
        return SPARSE_OK;
    }

    idx_t n_fine = fine->n;

    if (strategy == COARSENING_HCC && n_fine >= 2) {
        double cv_threshold = 0.30;
        const char *env = getenv("SPARSE_ND_COARSENING_CV_FALLTHROUGH");
        if (env && *env) {
            char *endp = NULL;
            double v = strtod(env, &endp);
            if (env != endp && *endp == '\0' && v >= 0.0 && v <= 100.0)
                cv_threshold = v;
        }
        if (cv_threshold > 0.0) {
            double sum = 0.0;
            double sumsq = 0.0;
            for (idx_t i = 0; i < n_fine; i++) {
                double d = (double)(fine->xadj[i + 1] - fine->xadj[i]);
                sum += d;
                sumsq += d * d;
            }
            double mean = sum / (double)n_fine;
            double var = sumsq / (double)n_fine - mean * mean;
            if (var < 0.0)
                var = 0.0;
            double cv = (mean > 0.0) ? sqrt(var) / mean : 0.0;
            if (cv > cv_threshold) {
                if (getenv("SPARSE_HCC_DEBUG")) {
                    fprintf(stderr,
                            "hcc-debug strategy=hcc fell through to heavy_edge: "
                            "n_fine=%d CV=%.3f > threshold=%.3f\n",
                            (int)n_fine, cv, cv_threshold);
                }
                strategy = COARSENING_HEAVY_EDGE;
            }
        }
    }

    idx_t *perm = malloc((size_t)n_fine * sizeof(idx_t));
    if (!perm)
        return SPARSE_ERR_ALLOC;
    fisher_yates_shuffle(perm, n_fine, seed);

    for (idx_t i = 0; i < n_fine; i++)
        cmap_out[i] = -1;

    idx_t n_coarse = 0;
    for (idx_t p = 0; p < n_fine; p++) {
        idx_t v = perm[p];
        if (cmap_out[v] != -1)
            continue;
        idx_t best_nbr = -1;
        if (strategy == COARSENING_HCC) {
            int64_t best_score = 0;
            idx_t deg_v = fine->xadj[v + 1] - fine->xadj[v];
            for (idx_t k = fine->xadj[v]; k < fine->xadj[v + 1]; k++) {
                idx_t u = fine->adjncy[k];
                if (cmap_out[u] != -1)
                    continue;
                idx_t w = fine->ewgt ? fine->ewgt[k] : 1;
                idx_t deg_u = fine->xadj[u + 1] - fine->xadj[u];
                idx_t mind = (deg_v < deg_u) ? deg_v : deg_u;
                int64_t score = (int64_t)w * (int64_t)mind;
                if ((score > best_score) || (best_nbr < 0) ||
                    (score == best_score && u < best_nbr)) {
                    best_score = score;
                    best_nbr = u;
                }
            }
        } else {
            idx_t best_wt = 0;
            for (idx_t k = fine->xadj[v]; k < fine->xadj[v + 1]; k++) {
                idx_t u = fine->adjncy[k];
                if (cmap_out[u] != -1)
                    continue;
                idx_t w = fine->ewgt ? fine->ewgt[k] : 1;
                if (w > best_wt) {
                    best_wt = w;
                    best_nbr = u;
                }
            }
        }
        cmap_out[v] = n_coarse;
        if (best_nbr != -1)
            cmap_out[best_nbr] = n_coarse;
        n_coarse++;
    }
    free(perm);

    if (getenv("SPARSE_HCC_DEBUG")) {
        const char *strategy_name = (strategy == COARSENING_HCC) ? "hcc" : "heavy_edge";
        idx_t *cluster_sizes = (n_coarse > 0) ? calloc((size_t)n_coarse, sizeof(idx_t)) : NULL;
        idx_t matched = 0;
        if (cluster_sizes) {
            for (idx_t i = 0; i < n_fine; i++)
                cluster_sizes[cmap_out[i]]++;
            for (idx_t i = 0; i < n_fine; i++) {
                if (cluster_sizes[cmap_out[i]] > 1)
                    matched++;
            }
            free(cluster_sizes);
        }
        double match_ratio = (n_fine > 0) ? (double)matched / (double)n_fine : 0.0;
        fprintf(stderr, "hcc-debug strategy=%s n_fine=%d n_coarse=%d match_ratio=%.3f\n",
                strategy_name, (int)n_fine, (int)n_coarse, match_ratio);
        for (idx_t i = 0; i < n_fine; i += 16) {
            idx_t end = (i + 16 > n_fine) ? n_fine - 1 : i + 15;
            fprintf(stderr, "hcc-debug cmap[%d..%d] =", (int)i, (int)end);
            for (idx_t j = i; j < n_fine && j < i + 16; j++)
                fprintf(stderr, " %d", (int)cmap_out[j]);
            fprintf(stderr, "\n");
        }
    }

    if (n_coarse <= 0) {
        coarse_out->xadj = malloc(sizeof(idx_t));
        if (!coarse_out->xadj)
            return SPARSE_ERR_ALLOC;
        coarse_out->xadj[0] = 0;
        return SPARSE_OK;
    }

    idx_t *vwgt_coarse = calloc((size_t)n_coarse, sizeof(idx_t));
    idx_t *deg_coarse = calloc((size_t)n_coarse, sizeof(idx_t));
    if (!vwgt_coarse || !deg_coarse) {
        free(vwgt_coarse);
        free(deg_coarse);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t i = 0; i < n_fine; i++) {
        idx_t c = cmap_out[i];
        vwgt_coarse[c] += fine->vwgt ? fine->vwgt[i] : 1;
    }
    for (idx_t i = 0; i < n_fine; i++) {
        idx_t ci = cmap_out[i];
        for (idx_t k = fine->xadj[i]; k < fine->xadj[i + 1]; k++) {
            idx_t j = fine->adjncy[k];
            if (j <= i)
                continue;
            idx_t cj = cmap_out[j];
            if (ci == cj)
                continue;
            deg_coarse[ci]++; // NOLINT(clang-analyzer-security.ArrayBound)
            deg_coarse[cj]++; // NOLINT(clang-analyzer-security.ArrayBound)
        }
    }

    idx_t *xadj = malloc((size_t)(n_coarse + 1) * sizeof(idx_t));
    if (!xadj) {
        free(vwgt_coarse);
        free(deg_coarse);
        return SPARSE_ERR_ALLOC;
    }
    xadj[0] = 0;
    for (idx_t c = 0; c < n_coarse; c++)
        xadj[c + 1] = xadj[c] + deg_coarse[c];
    idx_t total = xadj[n_coarse]; // NOLINT(clang-analyzer-security.ArrayBound)

    coarse_edge_t *buckets = calloc((size_t)(total > 0 ? total : 1), sizeof(coarse_edge_t));
    idx_t *cursor = calloc((size_t)n_coarse, sizeof(idx_t));
    if (!buckets || !cursor) {
        free(buckets);
        free(cursor);
        free(xadj);
        free(vwgt_coarse);
        free(deg_coarse);
        return SPARSE_ERR_ALLOC;
    }
    free(deg_coarse);

    for (idx_t i = 0; i < n_fine; i++) {
        idx_t ci = cmap_out[i];
        for (idx_t k = fine->xadj[i]; k < fine->xadj[i + 1]; k++) {
            idx_t j = fine->adjncy[k];
            if (j <= i)
                continue;
            idx_t cj = cmap_out[j];
            if (ci == cj)
                continue;
            idx_t w = fine->ewgt ? fine->ewgt[k] : 1;
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            buckets[xadj[ci] + cursor[ci]++] = (coarse_edge_t){.nbr = cj, .wt = w};
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            buckets[xadj[cj] + cursor[cj]++] = (coarse_edge_t){.nbr = ci, .wt = w};
        }
    }
    free(cursor);

    idx_t *new_deg = calloc((size_t)n_coarse, sizeof(idx_t));
    if (!new_deg) {
        free(buckets);
        free(xadj);
        free(vwgt_coarse);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t c = 0; c < n_coarse; c++) {
        idx_t start = xadj[c];
        idx_t len = xadj[c + 1] - start;
        if (len <= 1) {
            new_deg[c] = len;
            continue;
        }
        coarse_edge_t *list = &buckets[start];
        qsort(list, (size_t)len, sizeof(coarse_edge_t), cmp_coarse_edge);
        idx_t write = 0;
        list[write++] = list[0]; // NOLINT(clang-analyzer-security.ArrayBound)
        for (idx_t a = 1; a < len; a++) {
            if (list[a].nbr == list[write - 1].nbr) { // NOLINT(clang-analyzer-security.ArrayBound)
                list[write - 1].wt += list[a].wt;
            } else {
                list[write++] = list[a];
            }
        }
        new_deg[c] = write;
    }

    idx_t *final_xadj = malloc((size_t)(n_coarse + 1) * sizeof(idx_t));
    if (!final_xadj) {
        free(buckets);
        free(xadj);
        free(vwgt_coarse);
        free(new_deg);
        return SPARSE_ERR_ALLOC;
    }
    final_xadj[0] = 0;
    for (idx_t c = 0; c < n_coarse; c++)
        final_xadj[c + 1] = final_xadj[c] + new_deg[c];
    idx_t final_total = final_xadj[n_coarse];

    idx_t *final_adjncy = malloc((size_t)(final_total > 0 ? final_total : 1) * sizeof(idx_t));
    idx_t *final_ewgt = malloc((size_t)(final_total > 0 ? final_total : 1) * sizeof(idx_t));
    if (!final_adjncy || !final_ewgt) {
        free(final_adjncy);
        free(final_ewgt);
        free(final_xadj);
        free(buckets);
        free(xadj);
        free(vwgt_coarse);
        free(new_deg);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t c = 0; c < n_coarse; c++) {
        coarse_edge_t *src = &buckets[xadj[c]]; // NOLINT(clang-analyzer-security.ArrayBound)
        idx_t dst = final_xadj[c];
        for (idx_t a = 0; a < new_deg[c]; a++) {
            final_adjncy[dst + a] = src[a].nbr;
            final_ewgt[dst + a] = src[a].wt;
        }
    }

    free(buckets);
    free(xadj);
    free(new_deg);

    coarse_out->n = n_coarse;
    coarse_out->xadj = final_xadj;
    coarse_out->adjncy = final_adjncy;
    coarse_out->vwgt = vwgt_coarse;
    coarse_out->ewgt = final_ewgt;
    return SPARSE_OK;
}

sparse_err_t graph_coarsen_heavy_edge_matching(const sparse_graph_t *fine, uint32_t seed,
                                               sparse_graph_t *coarse_out, idx_t *cmap_out) {
    return graph_coarsen_with_strategy(fine, seed, COARSENING_HEAVY_EDGE, coarse_out, cmap_out);
}

sparse_err_t graph_coarsen_hcc(const sparse_graph_t *fine, uint32_t seed,
                               sparse_graph_t *coarse_out, idx_t *cmap_out) {
    return graph_coarsen_with_strategy(fine, seed, COARSENING_HCC, coarse_out, cmap_out);
}

void sparse_graph_hierarchy_free(sparse_graph_hierarchy_t *h) {
    if (!h)
        return;
    if (h->coarse) {
        for (int i = 0; i < h->nlevels; i++)
            sparse_graph_free(&h->coarse[i]);
        free(h->coarse);
    }
    if (h->cmaps) {
        for (int i = 0; i < h->nlevels; i++)
            free(h->cmaps[i]);
        free(h->cmaps);
    }
    h->coarse = NULL;
    h->cmaps = NULL;
    h->nlevels = 0;
}

sparse_err_t sparse_graph_hierarchy_build(const sparse_graph_t *root, uint32_t seed,
                                          sparse_graph_hierarchy_t *h) {
    if (!h)
        return SPARSE_ERR_NULL;
    h->nlevels = 0;
    h->coarse = NULL;
    h->cmaps = NULL;
    if (!root)
        return SPARSE_ERR_NULL;

    if (root->n == 0)
        return SPARSE_OK;

    idx_t n_root = root->n;
    idx_t divisor = 100;
    {
        const char *env = getenv("SPARSE_ND_COARSEN_FLOOR_RATIO");
        if (env) {
            char *endp = NULL;
            long v = strtol(env, &endp, 10);
            if (env != endp && *endp == '\0' && v >= 1 && v <= 100000)
                divisor = (idx_t)v;
        }
    }
    idx_t base_threshold = n_root / divisor;
    if (base_threshold < 20)
        base_threshold = 20;

    coarsening_strategy_t strategy = parse_coarsening_strategy();

    int level_cap = 5;
    {
        idx_t v = n_root;
        while (v > 1) {
            v >>= 1;
            level_cap++;
        }
        if (level_cap > 64)
            level_cap = 64;
    }

    int cap = 8;
    if (cap > level_cap)
        cap = level_cap;
    h->coarse = calloc((size_t)cap, sizeof(sparse_graph_t));
    h->cmaps = calloc((size_t)cap, sizeof(idx_t *));
    if (!h->coarse || !h->cmaps) {
        sparse_graph_hierarchy_free(h);
        return SPARSE_ERR_ALLOC;
    }

    const sparse_graph_t *prev = root;
    for (int level = 0; level < level_cap; level++) {
        idx_t n_prev = prev->n;
        if (n_prev <= 2)
            break;
        idx_t *cmap = malloc((size_t)n_prev * sizeof(idx_t));
        if (!cmap) {
            sparse_graph_hierarchy_free(h);
            return SPARSE_ERR_ALLOC;
        }
        sparse_graph_t coarse = {0};
        sparse_err_t rc;
        if (strategy == COARSENING_HCC) {
            rc = graph_coarsen_hcc(prev, seed + (uint32_t)level, &coarse, cmap);
        } else {
            rc = graph_coarsen_heavy_edge_matching(prev, seed + (uint32_t)level, &coarse, cmap);
        }
        if (rc != SPARSE_OK) {
            free(cmap);
            sparse_graph_hierarchy_free(h);
            return rc;
        }
        idx_t n_coarse = coarse.n;
        int no_progress = (n_coarse * 10 > n_prev * 9);
        int small_enough = (n_coarse <= base_threshold);

        if (level >= cap) {
            int new_cap = cap * 2;
            if (new_cap > level_cap)
                new_cap = level_cap;
            sparse_graph_t *new_coarse =
                realloc(h->coarse, (size_t)new_cap * sizeof(sparse_graph_t));
            idx_t **new_cmaps = realloc(h->cmaps, (size_t)new_cap * sizeof(idx_t *));
            if (!new_coarse || !new_cmaps) {
                free(new_coarse ? new_coarse : h->coarse);
                free(new_cmaps ? new_cmaps : h->cmaps);
                h->coarse = NULL;
                h->cmaps = NULL;
                sparse_graph_free(&coarse);
                free(cmap);
                sparse_graph_hierarchy_free(h);
                return SPARSE_ERR_ALLOC;
            }
            for (int i = cap; i < new_cap; i++) {
                memset(&new_coarse[i], 0, sizeof(sparse_graph_t));
                new_cmaps[i] = NULL;
            }
            h->coarse = new_coarse;
            h->cmaps = new_cmaps;
            cap = new_cap;
        }

        if (level == 0 && no_progress) {
            sparse_graph_free(&coarse);
            free(cmap);
            sparse_graph_hierarchy_free(h);
            return SPARSE_OK;
        }
        if (no_progress) {
            sparse_graph_free(&coarse);
            free(cmap);
            break;
        }

        h->coarse[level] = coarse;
        h->cmaps[level] = cmap;
        h->nlevels = level + 1;
        prev = &h->coarse[level];

        if (small_enough)
            break;
    }
    return SPARSE_OK;
}
