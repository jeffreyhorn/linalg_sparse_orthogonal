/*
 * sparse_graph_refine.c — FM refinement subsystem extracted from the
 *                         residual multilevel graph partitioner.
 *
 * Phase-2 ownership:
 *   - FM-local thread state
 *   - FM parser/helpers
 *   - cut-weight evaluation shared with uncoarsening
 *   - gain-bucket implementation
 *   - graph_refine_fm(...)
 *
 * `graph_uncoarsen(...)` and top-level orchestration remain in
 * `src/sparse_graph.c`; they interact with this file only through the
 * internal runtime/config seam declared in `sparse_graph_internal.h`.
 */

#include "sparse_alloc_internal.h"
#include "sparse_graph_fm_buckets.h"
#include "sparse_graph_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local int fm_pop_use_tail = 0;
static _Thread_local int fm_use_annealing = 0;
static _Thread_local fm_anneal_schedule_t fm_anneal_schedule = FM_ANNEAL_SCHEDULE_EXPONENTIAL;
static _Thread_local int fm_use_thick_restart = 0;
static _Thread_local fm_thick_restart_perturb_t fm_thick_restart_perturb =
    FM_THICK_RESTART_PERTURB_RANDOM_FLIP;
static _Thread_local fm_gain_noise_schedule_t fm_gain_noise_schedule =
    FM_GAIN_NOISE_SCHEDULE_LINEAR;
static _Thread_local int fm_anneal_debug = 0;
static _Thread_local int fm_gain_noise_debug = 0;
static _Thread_local int fm_thick_restart_debug = 0;
static _Thread_local int fm_anneal_pass_idx = 0;
static _Thread_local int fm_anneal_total_passes = 1;

void sparse_graph_fm_runtime_get(sparse_graph_fm_runtime_t *out) {
    if (!out)
        return;
    out->pop_use_tail = fm_pop_use_tail;
    out->use_annealing = fm_use_annealing;
    out->anneal_schedule = fm_anneal_schedule;
    out->anneal_pass_idx = fm_anneal_pass_idx;
    out->anneal_total_passes = fm_anneal_total_passes;
    out->use_thick_restart = fm_use_thick_restart;
    out->thick_restart_perturb = fm_thick_restart_perturb;
    out->gain_noise_schedule = fm_gain_noise_schedule;
    out->anneal_debug = fm_anneal_debug;
    out->gain_noise_debug = fm_gain_noise_debug;
    out->thick_restart_debug = fm_thick_restart_debug;
}

void sparse_graph_fm_runtime_set(const sparse_graph_fm_runtime_t *state) {
    if (!state)
        return;
    fm_pop_use_tail = state->pop_use_tail;
    fm_use_annealing = state->use_annealing;
    fm_anneal_schedule = state->anneal_schedule;
    fm_anneal_pass_idx = state->anneal_pass_idx;
    fm_anneal_total_passes = state->anneal_total_passes;
    fm_use_thick_restart = state->use_thick_restart;
    fm_thick_restart_perturb = state->thick_restart_perturb;
    fm_gain_noise_schedule = state->gain_noise_schedule;
    fm_anneal_debug = state->anneal_debug;
    fm_gain_noise_debug = state->gain_noise_debug;
    fm_thick_restart_debug = state->thick_restart_debug;
}

void sparse_graph_thick_restart_perturb(const sparse_graph_t *G, idx_t *part,
                                        fm_thick_restart_perturb_t mode, uint32_t *rng) {
    idx_t n = G->n;
    if (n < 2)
        return;

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
            *rng ^= *rng << 13;
            *rng ^= *rng >> 17;
            *rng ^= *rng << 5;
            if ((*rng & 1U) != 0U)
                part[v] = (idx_t)(1 - part[v]);
        }
        return;
    }

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

// NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
idx_t sparse_graph_compute_cut_weight(const sparse_graph_t *G, const idx_t *part) {
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

#define FM_BUCKET_EMPTY ((idx_t) - 1)

sparse_err_t fm_bucket_array_init(fm_bucket_array_t *arr, idx_t n_vertices, idx_t max_gain) {
    if (!arr)
        return SPARSE_ERR_NULL;
    if (n_vertices < 0 || max_gain < 0)
        return SPARSE_ERR_BADARG;

    if (max_gain > (IDX_MAX - 1) / 2)
        return SPARSE_ERR_ALLOC;
    idx_t num_buckets = 2 * max_gain + 1;
    if ((size_t)num_buckets > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;

    size_t link_len = (size_t)(n_vertices > 0 ? n_vertices : 1);
    if (link_len > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;

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
    arr->prev[vertex] = FM_BUCKET_EMPTY;
    arr->next[vertex] = arr->heads[bucket];
    if (arr->heads[bucket] != FM_BUCKET_EMPTY)
        arr->prev[arr->heads[bucket]] = vertex;
    else
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
        arr->tails[bucket] = p;
    arr->counts[bucket]--;
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

    sparse_err_t (*pop_max)(fm_bucket_array_t *, idx_t *, idx_t *) =
        fm_pop_use_tail ? fm_bucket_pop_max_tail : fm_bucket_pop_max;

    const int use_annealing = fm_use_annealing;
    const int anneal_debug = use_annealing && fm_anneal_debug;
    double anneal_T = 0.0;
    uint32_t anneal_rng = 0;
    idx_t anneal_worsening_accepted = 0;
    idx_t anneal_worsening_rejected = 0;

    const int use_gain_noise_formal =
        fm_use_thick_restart &&
        fm_thick_restart_perturb == FM_THICK_RESTART_PERTURB_GAIN_NOISE_FORMAL;
    const int gain_noise_debug = use_gain_noise_formal && fm_gain_noise_debug;
    double gain_sigma_k = 0.0;
    uint32_t gain_noise_rng = 0;

    idx_t n = G->n;
    idx_t *gain = NULL;
    int *locked = NULL;
    int *in_bucket = NULL;
    idx_t *best_part = NULL;
    idx_t *skipped_this_step = NULL;
    idx_t *gain_for_bucket = NULL;
    sparse_err_t alloc_rc = sparse_malloc_idx_array(n, sizeof(*gain), (void **)&gain);
    if (alloc_rc == SPARSE_OK)
        alloc_rc = sparse_calloc_idx_array(n, sizeof(*locked), (void **)&locked);
    if (alloc_rc == SPARSE_OK)
        alloc_rc = sparse_calloc_idx_array(n, sizeof(*in_bucket), (void **)&in_bucket);
    if (alloc_rc == SPARSE_OK)
        alloc_rc = sparse_malloc_idx_array(n, sizeof(*best_part), (void **)&best_part);
    if (alloc_rc == SPARSE_OK)
        alloc_rc =
            sparse_malloc_idx_array(n, sizeof(*skipped_this_step), (void **)&skipped_this_step);
    if (alloc_rc == SPARSE_OK && use_gain_noise_formal) {
        alloc_rc = sparse_calloc_idx_array(n, sizeof(*gain_for_bucket), (void **)&gain_for_bucket);
    }
    if (alloc_rc != SPARSE_OK) {
        free(gain);
        free(locked);
        free(in_bucket);
        free(best_part);
        free(skipped_this_step);
        free(gain_for_bucket);
        return alloc_rc;
    }

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

    idx_t bucket_max_gain = max_weighted_degree;
    if (use_gain_noise_formal) {
        bucket_max_gain = max_weighted_degree * 2;
        if (bucket_max_gain < max_weighted_degree)
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
        anneal_rng =
            (uint32_t)(((uint64_t)(uint32_t)n * 31U + (uint32_t)(uint64_t)(unsigned long)k) *
                           2654435761U +
                       1U);
    }

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

    for (idx_t v = n - 1; v >= 0; v--) {
        // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
        idx_t bucket_key =
            (use_gain_noise_formal && gain_for_bucket) ? gain_for_bucket[v] : gain[v];
        fm_bucket_insert(&buckets, v, bucket_key);
        in_bucket[v] = 1;
    }

    idx_t cur_cut = sparse_graph_compute_cut_weight(G, part_io);
    idx_t best_cut = cur_cut;
    memcpy(best_part, part_io, (size_t)n * sizeof(idx_t));

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
    idx_t max_imbal = total_vwgt / 20;
    if (max_imbal < init_imbal)
        max_imbal = init_imbal;
    max_imbal += max_vwgt;

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
            if (locked[v])
                continue;
            idx_t v_w = G->vwgt ? G->vwgt[v] : 1;
            idx_t new_w0 = part_io[v] == 0 ? w0 - v_w : w0 + v_w;
            idx_t new_w1 = part_io[v] == 0 ? w1 + v_w : w1 - v_w;
            idx_t new_imbal = new_w0 > new_w1 ? new_w0 - new_w1 : new_w1 - new_w0;
            if (new_imbal > max_imbal) {
                // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
                skipped_this_step[skipped_count++] = v;
                continue;
            }
            if (use_annealing && g < 0 && anneal_T > 1.0) {
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

        if (have_candidate) {
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

    memcpy(part_io, best_part, (size_t)n * sizeof(idx_t));

    if (anneal_debug) {
        fprintf(stderr,
                "fm-annealing-debug n=%d pass=%d/%d schedule=%d T=%.3f "
                "worsening_accepted=%d worsening_rejected=%d\n",
                (int)n, fm_anneal_pass_idx, fm_anneal_total_passes, (int)fm_anneal_schedule,
                anneal_T, (int)anneal_worsening_accepted, (int)anneal_worsening_rejected);
    }

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
