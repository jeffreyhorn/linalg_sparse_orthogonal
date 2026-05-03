#ifndef SPARSE_GRAPH_FM_BUCKETS_H
#define SPARSE_GRAPH_FM_BUCKETS_H

/**
 * @file sparse_graph_fm_buckets.h
 * @brief Sprint 23 Day 9 — gain-bucket structure for FM refinement.
 *
 * Day-10 swap target: replaces the O(n) max-gain scan in
 * `graph_refine_fm` (src/sparse_graph.c, Sprint 22 Day 4) with an
 * O(1) max-find via a bucket array indexed by gain.  Day 9 stands
 * up the data structure + unit-tested API; Day 10 wires it into
 * the FM hot loop.
 *
 * Design (METIS Lib/refine.c reference; Karypis-Kumar 1998 §4):
 *
 *   - `heads[]` is an array of length `2·max_gain + 1`.  `heads[i]`
 *     is the vertex ID at the head of the doubly-linked list living
 *     in bucket `i`, or `-1` (sentinel) if the bucket is empty.
 *   - Bucket index for gain `g` is `bucket_offset + g`, with
 *     `bucket_offset = max_gain` so gains in `[-max_gain, +max_gain]`
 *     map to indices in `[0, 2·max_gain]`.
 *   - `next[]` and `prev[]` are per-vertex linkage arrays of length
 *     `n_vertices`.  `next[v]` / `prev[v]` are the IDs of the next /
 *     previous vertex in v's bucket, or `-1` at list ends.  Vertices
 *     not in any bucket have undefined linkage entries (we don't
 *     read them until insert).
 *   - `cursor` tracks the highest non-empty bucket index, or `-1`
 *     if no bucket holds anything.  `pop_max` reads the head of
 *     `heads[cursor]` in O(1); insert may push the cursor up;
 *     remove may walk the cursor down past empty buckets.  The
 *     amortised per-operation cost stays O(1) over an FM pass —
 *     each bucket is visited at most twice by the cursor walk
 *     (once on the way down, once back up if a later insert
 *     repopulates it), and per-pass insert/remove counts are
 *     bounded by `O(|E|)` (one move + per-neighbour gain update).
 *
 * Contract: each vertex is in at most one bucket at any time.  The
 * caller is responsible for tracking whether a vertex is currently
 * inserted (the bucket structure does not maintain a "membership"
 * bit — `insert(v, g_a)` followed by `insert(v, g_b)` without an
 * intervening `remove(v, g_a)` corrupts the linkage).  FM's natural
 * usage matches: vertices enter via initial-population, get popped
 * by `pop_max` (which removes them), and re-enter only via
 * gain-update propagation that always pairs a `remove(v, old_g)`
 * with `insert(v, new_g)`.
 */

#include "sparse_types.h"

/**
 * @brief Bucket-array gain queue for FM refinement.
 *
 * Lifecycle: caller-owned struct slot, zero-initialised on declaration.
 * `fm_bucket_array_init` populates the four arrays; `fm_bucket_array_free`
 * releases them and zeros the scalar fields.  Re-init is allowed (free
 * before re-init, or just zero-init the slot before a fresh init).
 */
typedef struct {
    idx_t *heads;        /**< Per-bucket head vertex; length `num_buckets`. -1 = empty. */
    idx_t *next;         /**< Per-vertex next-in-bucket; length `n_vertices`. */
    idx_t *prev;         /**< Per-vertex prev-in-bucket; length `n_vertices`. */
    idx_t *counts;       /**< Per-bucket vertex count; length `num_buckets`. */
    idx_t n_vertices;    /**< Total vertices in the parent graph. */
    idx_t max_gain;      /**< Bucket array spans gains in [-max_gain, +max_gain]. */
    idx_t bucket_offset; /**< Equals max_gain (so gain g → heads[offset + g]). */
    idx_t num_buckets;   /**< Equals 2·max_gain + 1. */
    idx_t cursor;        /**< Highest non-empty bucket index, or -1 if all empty. */
} fm_bucket_array_t;

/**
 * @brief Allocate the bucket array's internal storage.
 *
 * @param arr           Caller-owned struct slot; expected to be zero-init
 *                      or freed.
 * @param n_vertices    Vertex count for the per-vertex linkage arrays.
 *                      Vertices are addressed by ID in `[0, n_vertices)`.
 * @param max_gain      Inclusive upper bound on absolute gain magnitude.
 *                      Bucket array size is `2·max_gain + 1`.  Must be ≥ 0.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if `arr` is NULL.
 * @return SPARSE_ERR_BADARG if `n_vertices < 0` or `max_gain < 0`.
 * @return SPARSE_ERR_ALLOC on allocation failure (arr's pointers
 *         are NULL on return; `_free` is safe).
 */
sparse_err_t fm_bucket_array_init(fm_bucket_array_t *arr, idx_t n_vertices, idx_t max_gain);

/**
 * @brief Insert vertex `vertex` into the bucket for `gain`.
 *
 * O(1).  Pushes the cursor up if `gain` exceeds the current cursor's
 * gain.  Caller must guarantee the vertex is not already in any
 * bucket (insertion of an already-present vertex corrupts the
 * linkage with no diagnostic).
 *
 * @param arr      Initialised bucket array.
 * @param vertex   Vertex ID in `[0, n_vertices)`.
 * @param gain     Signed gain in `[-max_gain, +max_gain]`.
 */
void fm_bucket_insert(fm_bucket_array_t *arr, idx_t vertex, idx_t gain);

/**
 * @brief Remove vertex `vertex` from the bucket for `gain`.
 *
 * O(1) for the unlink itself; the cursor walk-down (if `vertex` was
 * the last entry in the cursor's bucket) is amortised O(1) over an
 * FM pass.  Caller must pass the same `gain` that was used for the
 * insert (the bucket array doesn't track per-vertex bucket
 * membership).
 *
 * @param arr      Initialised bucket array.
 * @param vertex   Vertex ID in `[0, n_vertices)`.
 * @param gain     Signed gain the vertex was inserted with.
 */
void fm_bucket_remove(fm_bucket_array_t *arr, idx_t vertex, idx_t gain);

/**
 * @brief Pop the highest-gain vertex from the bucket array.
 *
 * O(1) amortised.  Removes the popped vertex from its bucket; walks
 * the cursor down past newly-empty buckets if needed.
 *
 * @param arr          Initialised bucket array.
 * @param vertex_out   Output: popped vertex ID.  Untouched on empty.
 * @param gain_out     Output: popped vertex's gain.  Untouched on empty.
 *
 * @return SPARSE_OK if a vertex was popped.
 * @return SPARSE_ERR_BOUNDS if the bucket array is empty (no vertices
 *         currently inserted).
 */
sparse_err_t fm_bucket_pop_max(fm_bucket_array_t *arr, idx_t *vertex_out, idx_t *gain_out);

/**
 * @brief Release the bucket array's internal storage and zero its
 *        scalar fields.  Safe on a zero-initialised struct (no-op).
 *        Safe to call multiple times.
 */
void fm_bucket_array_free(fm_bucket_array_t *arr);

#endif /* SPARSE_GRAPH_FM_BUCKETS_H */
