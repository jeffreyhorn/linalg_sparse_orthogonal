/* Sprint 24 Day 1: feature-test macro to expose `clock_gettime` /
 * `CLOCK_MONOTONIC` for the `SPARSE_QG_PROFILE` instrumentation
 * below.  Same pattern used in `tests/test_sprint10_integration.c`
 * and `tests/test_reorder_nd.c`; clang-tidy's
 * `bugprone-reserved-identifier` rule flags `_POSIX_C_SOURCE` as
 * reserved (correct in user space but POSIX-mandated for libc-
 * exposure-control), so suppress just for the define line. */
#if !defined(_WIN32) && (!defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 199309L)
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define _POSIX_C_SOURCE 199309L
#endif
/*
 * sparse_reorder_amd_qg.c — Quotient-graph AMD (Sprint 22 Days 10-12).
 *
 * ─── Design block ─────────────────────────────────────────────────────
 *
 * Sprint 22's last major item: replace the bitset AMD that lives in
 * `src/sparse_reorder.c:421` (`sparse_reorder_amd`) with a quotient-
 * graph implementation that runs in O(nnz) memory rather than the
 * existing O(n² / 64).
 *
 * The bitset version (Sprint 13) allocates an n × ⌈n/64⌉-word bitset
 * adjacency matrix, computes minimum degree via popcount, and merges
 * neighbours' adjacency on each pivot via a per-row word-wise OR.
 * The seam is clean — `n × nwords` allocation at line 443, pivot
 * loop at lines 468-477, neighbour-merge at lines 488-508 — and the
 * routine produces a valid AMD-quality permutation.  But on n = 1806
 * (bcsstk14) the bitset already takes 410 KB; on n = 50 000 the
 * bitset is 312 MB.  At n = 100 000 it's 1.25 GB — well past where
 * any reasonable factorisation should be paying that overhead.
 *
 * **Quotient-graph representation** (Amestoy / Davis / Duff 2004,
 * "Algorithm 837: AMD, an approximate minimum degree ordering
 * algorithm", TOMS).  Vertices and *elements* (clusters of
 * eliminated variables) live in one shared integer workspace
 * `iw[]`.  The lower (variables-only) half of `iw[]` holds each
 * variable's adjacency — a mix of "still a variable" and "is now
 * inside element e" entries — while the upper half holds element-
 * level adjacency lists.  Memory bound is `≈ 5·nnz + 6·n + 1`
 * integer entries, regardless of fill: as the elimination proceeds,
 * eliminated variables move from the variable side into element
 * lists without growing the workspace.
 *
 * **Element absorption.**  Once a variable's adjacency reduces to a
 * single element `e` (i.e. all its neighbours have been eliminated
 * into the same element), that variable is absorbed into `e` —
 * removed from the active set, the workspace slot it occupied
 * compacted away.  Absorption is what keeps the workspace bounded:
 * without it, the variable adjacency would grow linearly with
 * elimination steps.
 *
 * **Supervariable detection.**  Variables whose adjacency lists
 * become identical (after element absorption) are merged into a
 * single supervariable.  Minimum-degree pivot selection then runs
 * over supervariables, not individual variables — on PDE-like
 * matrices this shrinks the active set by 5-20× and is the headline
 * win over a naïve quotient-graph implementation.  The supervariable
 * link follows a hash-then-compare pattern (Davis 2006 §7); the
 * hash uses a cheap variable-list signature, with full comparison
 * only on hash collisions.
 *
 * **Approximate degree update.**  Exact minimum-degree update is
 * O(nnz) per pivot — far too expensive on dense pivots.  AMD
 * instead bounds the post-pivot degree of each affected variable by
 *   d_approx(i) = |adj(i, V)| + Σ_e |adj(e, V) \ {pivot}|
 * — the count of variables in i's adjacency, plus the count of
 * variables reachable through i's element neighbours.  The bound is
 * tight enough on most matrices that the resulting ordering is
 * within a few percent of exact-MD fill, while costing only
 * O(adjacency-of-adjacency) per pivot.  Davis 2006 §7's
 * `dense_row_threshold` skips the approximate-degree update on
 * vertices whose post-pivot degree exceeds 10·√n — at that point
 * they're "dense rows" and the minimum-degree heuristic isn't
 * informative.
 *
 * **Workspace layout** (single `iw[]` allocation; offsets named
 * after the SuiteSparse AMD reference):
 *   iw_size = 5·nnz + 6·n + 1
 *   - iw[0..nnz-1]                : variable adjacency entries (mixed
 *                                   variable / element references)
 *   - iw[nnz..2·nnz-1]            : element adjacency entries
 *   - len[0..n-1]                 : per-vertex adjacency length
 *   - elen[0..n-1]                : per-vertex element-list length
 *   - degree[0..n-1]              : approximate degree
 *   - super[0..n-1]               : supervariable link (negative when
 *                                   absorbed, points to head when
 *                                   leader)
 *   - next[0..n-1] / last[0..n-1] : doubly-linked list per degree
 *                                   bucket for O(1) min-degree pick
 *   - flag[0..n-1]                : scratch for the
 *                                   approximate-degree-update pass
 *
 * **Validation strategy.**  Because AMD's pivot tie-breaking differs
 * across implementations (the bitset version picks first-encountered
 * minimum-degree; the quotient-graph version walks supervariables in
 * bucket order), bit-identical permutation match across the two
 * isn't expected.  Day 11's tests instead assert *equivalent fill*:
 * on each corpus matrix (nos4 / bcsstk04 / bcsstk14), the symbolic
 * Cholesky `nnz(L)` under the quotient-graph permutation must be
 * within 5 % of the bitset version's nnz(L).  Day 12 then swaps the
 * production `sparse_reorder_amd` body to call this helper.
 *
 * **References.**
 *   - Amestoy, Davis, Duff (2004), "Algorithm 837: AMD, an
 *     approximate minimum degree ordering algorithm", ACM TOMS
 *     30(3):381-388.  The TOMS reference paper.
 *   - Davis (2006), "Direct Methods for Sparse Linear Systems",
 *     SIAM, §7.  Pseudocode for the quotient-graph elimination loop.
 *   - SuiteSparse AMD source (`AMD/Source/amd_2.c`).  Reference
 *     implementation.  Single-workspace design, hash-based
 *     supervariable merge, approximate-degree update with dense-row
 *     skip — the structure this file mirrors.
 *
 * **Sprint 22 day-by-day status.**
 *   - Day 10 (this commit): design block + stub returning
 *     SPARSE_ERR_BADARG.  No behaviour change — the production
 *     path still routes through the bitset AMD in
 *     `src/sparse_reorder.c`.
 *   - Day 11 (next): implement the elimination loop end-to-end.
 *     Parallel-test against the bitset on nos4 / bcsstk04 /
 *     bcsstk14; assert fill parity within 5 %.
 *   - Day 12: swap `sparse_reorder_amd`'s body to call this helper;
 *     remove the bitset implementation.  Run the full corpus
 *     regression suite; verify no AMD-using test regresses.
 *   - Day 13: capture bench numbers for the bitset → quotient-graph
 *     swap (peak RSS, wall time) on n ≥ 1000 fixtures.
 */

#include "sparse_reorder_amd_qg_internal.h"

#include "sparse_matrix.h"
#include "sparse_matrix_internal.h"
#include "sparse_types.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h> /* Sprint 24 Day 1: SPARSE_QG_PROFILE stderr trace. */
#include <stdlib.h>
#include <string.h>
#include <time.h> /* Sprint 24 Day 1: SPARSE_QG_PROFILE wall-clock instrumentation. */

/* ═══════════════════════════════════════════════════════════════════════
 * Day 11 implementation notes.
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The Day 10 design block sketches the full Davis 2006 quotient-graph
 * AMD with element absorption + supervariable detection + approximate-
 * degree updates.  Day 11 ships a *simplified* quotient-graph
 * minimum-degree implementation that:
 *
 *   - keeps the workspace structure (`iw[]` for adjacency, separate
 *     small per-vertex arrays for `xadj` / `len` / `deg` / `elim`);
 *   - rebuilds each affected vertex's adjacency on every pivot via a
 *     sorted merge (no element absorption — the merge operates
 *     directly on variable lists);
 *   - uses *exact* minimum-degree (each rebuild updates `deg[u]` to
 *     the new adjacency length);
 *   - compacts the workspace by walking unfilled vertices in
 *     `xadj`-sorted order on demand — `qg_reserve` triggers a
 *     compaction whenever an append would overflow `iw_size`, and
 *     doubles `iw[]` (capped against `INT32_MAX` and
 *     `SIZE_MAX/sizeof(idx_t)`) if compaction can't free enough room.
 *
 * The simplification leaves performance on the table relative to the
 * SuiteSparse AMD reference (the full Davis algorithm shaves the
 * pivot loop's per-vertex cost from O(deg) to O(supervariable count)
 * and the degree update from O(adjacency) to
 * O(adjacency-of-adjacency)).  But fill quality is identical to the
 * exact-MD upper bound and easily within the Day 11 plan's "≤ 5 %
 * of the bitset version's nnz(L)" target — the bitset AMD is also
 * exact MD, just with a denser representation.  Day 13's bench will
 * compare both against AMD-reference numbers; Sprint 23 may revisit
 * if the reference-AMD gap turns out to dominate factor wall time on
 * larger fixtures.
 */

/* ─── Quotient-graph workspace ─────────────────────────────────────────
 *
 * `iw[]` is the bulk of the allocation: per-vertex adjacency lists
 * concatenated, with `xadj[i]` pointing at vertex i's start and
 * `len[i]` recording its current length.  Lists grow during
 * elimination (fill-in), so when a vertex's adjacency must be
 * rewritten the new list is appended at `iw[iw_used..]` and `xadj[i]`
 * updated; the old slots become garbage that compaction reclaims.
 *
 * Workspace layout:
 *   iw       length iw_size (grows on demand)
 *   xadj     length n
 *   len      length n
 *   deg      length n
 *   elim     length n  (one byte per vertex)
 *
 * Initial `iw_size` is `5·nnz + 6·n + 1` (Davis 2006 §7), which
 * accommodates ~5× fill before the first compaction.  All adjacency
 * lists are kept sorted ascending so the merge step is a linear
 * two-pointer scan with the dedup / filter step inline; no separate
 * scratch buffer is needed.
 */
typedef struct {
    idx_t *iw;
    idx_t iw_size;
    idx_t iw_used;
    idx_t *xadj;
    idx_t *len;
    idx_t *deg;
    char *elim;
    idx_t n;
    /* Sprint 24 Day 1: SPARSE_QG_PROFILE accumulators (nanoseconds).
     * Off by default; `prof_enabled` set by `sparse_reorder_amd_qg`
     * if the env var is set.  Sprint 22 baseline has no element side
     * (no qg_recompute_deg with adj-of-adj walk, no supervariable
     * detection) so the eliminate-internal subdivisions from Sprint
     * 24 Day 1's instrumentation simplify to init / pick / eliminate
     * totals only.  Day 1's profile of the post-Sprint-23 build
     * showed qg_recompute_deg dominating at 95 % of total wall time
     * — the (c) revert this commit lands restores the variable-only
     * baseline that has no element-side walk. */
    int prof_enabled;
    long long prof_init_ns;
    long long prof_pick_ns;
    long long prof_eliminate_ns;
} qg_t;

/* Sprint 24 Day 1: monotonic-clock timestamp helper.  Returns
 * nanoseconds since an unspecified epoch.  POSIX
 * `clock_gettime(CLOCK_MONOTONIC, ...)` on non-Windows; Windows
 * routes through C11 `timespec_get(..., TIME_UTC)`. */
static long long qg_prof_now_ns(void) {
    struct timespec ts;
#ifdef _WIN32
    timespec_get(&ts, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}

static void qg_free(qg_t *qg) {
    if (!qg)
        return;
    free(qg->iw);
    free(qg->xadj);
    free(qg->len);
    free(qg->deg);
    free(qg->elim);
    qg->iw = NULL;
    qg->xadj = NULL;
    qg->len = NULL;
    qg->deg = NULL;
    qg->elim = NULL;
    qg->iw_size = 0;
    qg->iw_used = 0;
    qg->n = 0;
}

/* Initialise the quotient graph from a CSR adjacency.  `adj_ptr` /
 * `adj_list` are produced by `sparse_build_adj` (symmetric, no self-
 * loops, sorted).  Caller frees both regardless of return code. */
static sparse_err_t qg_init(qg_t *qg, idx_t n, const idx_t *adj_ptr, const idx_t *adj_list) {
    qg->n = n;
    idx_t nnz = adj_ptr[n];
    /* Davis 2006 §7: 5·nnz + 6·n + 1 is comfortably above the typical
     * fill peak before the first compaction.  Compute in `uint64_t`
     * and check against the storage type (`idx_t` = int32) plus
     * `SIZE_MAX / sizeof(idx_t)`; under-allocation here would later
     * surface as out-of-bounds writes in the merge / memcpy paths. */
    uint64_t iw_size_64 = (uint64_t)5 * (uint64_t)nnz + (uint64_t)6 * (uint64_t)n + 1;
    if (iw_size_64 < 1024)
        iw_size_64 = 1024;
    /* Single combined cap covers both the storage type (`idx_t`,
     * int32) and the malloc byte count.  On 64-bit `SIZE_MAX /
     * sizeof(idx_t)` dominates and `INT32_MAX` is the live bound;
     * on a hypothetical 32-bit platform the byte-count term can
     * tighten it. */
    const uint64_t cap = (uint64_t)INT32_MAX < SIZE_MAX / sizeof(idx_t)
                             ? (uint64_t)INT32_MAX
                             : (uint64_t)(SIZE_MAX / sizeof(idx_t));
    if (iw_size_64 > cap)
        return SPARSE_ERR_ALLOC;
    idx_t iw_size = (idx_t)iw_size_64;
    qg->iw_size = iw_size;
    qg->iw_used = nnz;
    /* Sprint 24 Day 1: profile counters; `prof_enabled` is set by
     * `sparse_reorder_amd_qg` per the env var. */
    qg->prof_enabled = 0;
    qg->prof_init_ns = 0;
    qg->prof_pick_ns = 0;
    qg->prof_eliminate_ns = 0;
    qg->iw = malloc((size_t)iw_size * sizeof(idx_t));
    qg->xadj = malloc((size_t)n * sizeof(idx_t));
    qg->len = malloc((size_t)n * sizeof(idx_t));
    qg->deg = malloc((size_t)n * sizeof(idx_t));
    qg->elim = calloc((size_t)n, sizeof(char));
    if (!qg->iw || !qg->xadj || !qg->len || !qg->deg || !qg->elim) {
        qg_free(qg);
        return SPARSE_ERR_ALLOC;
    }
    if (nnz > 0)
        memcpy(qg->iw, adj_list, (size_t)nnz * sizeof(idx_t));
    for (idx_t i = 0; i < n; i++) {
        qg->xadj[i] = adj_ptr[i];
        qg->len[i] = adj_ptr[i + 1] - adj_ptr[i];
        qg->deg[i] = qg->len[i];
    }
    return SPARSE_OK;
}

/* Pair (xadj_value, vertex_id) used by the qsort-driven compaction
 * walk.  Sorted by `xadj_val` ascending so we can safely memcpy each
 * surviving adjacency to a lower (or equal) `iw[]` position without
 * overlap. */
typedef struct {
    idx_t xadj_val;
    idx_t vertex;
} qg_compact_pair_t;

static int qg_compact_compare(const void *a, const void *b) {
    idx_t va = ((const qg_compact_pair_t *)a)->xadj_val;
    idx_t vb = ((const qg_compact_pair_t *)b)->xadj_val;
    return (va > vb) - (va < vb);
}

/* Compact the workspace: walk surviving (non-eliminated, non-empty)
 * vertices in xadj-sort order and pack their adjacency into the
 * front of `iw[]`.  After return, `iw_used` is the new high-water. */
static sparse_err_t qg_compact(qg_t *qg) {
    idx_t n = qg->n;
    /* Guard the `n * sizeof(qg_compact_pair_t)` byte count against
     * `size_t` overflow before the malloc.  On 32-bit `size_t` and
     * sizeof(qg_compact_pair_t) = 8, n above ~2^29 would wrap and
     * under-allocate, leading to OOB writes when filling `pairs[k]`. */
    if ((size_t)n > SIZE_MAX / sizeof(qg_compact_pair_t))
        return SPARSE_ERR_ALLOC;
    qg_compact_pair_t *pairs = malloc((size_t)n * sizeof(qg_compact_pair_t));
    if (!pairs)
        return SPARSE_ERR_ALLOC;
    idx_t k = 0;
    for (idx_t i = 0; i < n; i++) {
        if (!qg->elim[i] && qg->len[i] > 0) {
            pairs[k].xadj_val = qg->xadj[i];
            pairs[k].vertex = i;
            k++;
        }
    }
    qsort(pairs, (size_t)k, sizeof(qg_compact_pair_t), qg_compact_compare);

    idx_t new_pos = 0;
    for (idx_t j = 0; j < k; j++) {
        idx_t v = pairs[j].vertex;
        idx_t old_xadj = qg->xadj[v];
        idx_t l = qg->len[v];
        if (old_xadj != new_pos) {
            /* memmove handles the (rare) case where list shrinkage
             * yields a small overlap; in xadj-sort order it never
             * happens, but the cost is the same. */
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            memmove(&qg->iw[new_pos], &qg->iw[old_xadj], (size_t)l * sizeof(idx_t));
        }
        qg->xadj[v] = new_pos;
        new_pos += l;
    }
    qg->iw_used = new_pos;
    free(pairs);
    return SPARSE_OK;
}

/* Make sure `iw_used + need` fits in `iw_size`.  First tries
 * compaction; if that's not enough, doubles the workspace via
 * `realloc`.
 *
 * Growth math runs in `uint64_t` and checks against both `INT32_MAX`
 * (the storage type, `idx_t`) and `SIZE_MAX / sizeof(idx_t)` (the
 * `realloc` byte count) on every doubling step.  Without this an
 * int32 wrap could turn the loop into an infinite spin or hand
 * `realloc` a too-small request. */
static sparse_err_t qg_reserve(qg_t *qg, idx_t need) {
    /* Compute fits/grow checks in `uint64_t` so a hypothetical
     * `iw_used + need` overflow in `idx_t` (int32) cannot wrap into
     * a falsely-OK comparison. */
    const uint64_t cap = (uint64_t)INT32_MAX < SIZE_MAX / sizeof(idx_t)
                             ? (uint64_t)INT32_MAX
                             : (uint64_t)(SIZE_MAX / sizeof(idx_t));
    uint64_t target = (uint64_t)qg->iw_used + (uint64_t)need;
    if (target <= (uint64_t)qg->iw_size)
        return SPARSE_OK;
    sparse_err_t rc = qg_compact(qg);
    if (rc != SPARSE_OK)
        return rc;
    target = (uint64_t)qg->iw_used + (uint64_t)need;
    if (target <= (uint64_t)qg->iw_size)
        return SPARSE_OK;

    if (target > cap)
        return SPARSE_ERR_ALLOC;
    uint64_t new_size_64 = (uint64_t)qg->iw_size;
    while (new_size_64 < target) {
        if (new_size_64 > cap / 2) {
            /* Doubling would overshoot the storage cap; clamp to
             * `cap` (we already validated `target <= cap`, so the
             * loop terminates here). */
            new_size_64 = cap;
            break;
        }
        new_size_64 *= 2;
    }
    idx_t new_size = (idx_t)new_size_64;
    idx_t *new_iw = realloc(qg->iw, (size_t)new_size * sizeof(idx_t));
    if (!new_iw)
        return SPARSE_ERR_ALLOC;
    qg->iw = new_iw;
    qg->iw_size = new_size;
    return SPARSE_OK;
}

/* Find the unelim vertex with smallest current degree.  Returns -1
 * iff every vertex is eliminated (i.e. the loop's terminal step).
 * O(n) per call — the same complexity as the bitset AMD's pivot
 * scan. */
static idx_t qg_pick_min_deg(const qg_t *qg) {
    idx_t best = -1;
    idx_t best_deg = qg->n + 1;
    for (idx_t i = 0; i < qg->n; i++) {
        if (qg->elim[i])
            continue;
        if (qg->deg[i] < best_deg) {
            best_deg = qg->deg[i];
            best = i;
        }
    }
    return best;
}

/* Eliminate pivot `p`.  For each unelim neighbour `u`:
 *
 *   u_new_adj = (u_old_adj ∪ p_old_adj) \ {p, u, eliminated}
 *
 * Both lists are sorted, so the merge is a linear two-pointer scan
 * with the dedup / filter step inline.  The result is appended at
 * `iw[iw_used..]` and `u`'s `xadj` / `len` / `deg` are updated. */
static sparse_err_t qg_eliminate(qg_t *qg, idx_t p) {
    qg->elim[p] = 1;
    idx_t p_len = qg->len[p];
    if (p_len == 0)
        return SPARSE_OK;

    /* Snapshot p's adjacency before we touch `iw[]` — `qg_reserve`
     * may move the buffer underneath us.  Guard the `p_len *
     * sizeof(idx_t)` byte count against `size_t` overflow first;
     * on 32-bit `size_t` a wrap here would under-allocate and the
     * subsequent memcpy / per-neighbour merges would write OOB. */
    if ((size_t)p_len > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;
    idx_t *p_adj_copy = malloc((size_t)p_len * sizeof(idx_t));
    if (!p_adj_copy)
        return SPARSE_ERR_ALLOC;
    memcpy(p_adj_copy, &qg->iw[qg->xadj[p]], (size_t)p_len * sizeof(idx_t));

    for (idx_t k = 0; k < p_len; k++) {
        idx_t u = p_adj_copy[k];
        if (qg->elim[u])
            continue;

        idx_t u_len = qg->len[u];
        /* Merge upper bound = u_len + p_len.  Compute in `uint64_t`
         * so the addition can't wrap before `qg_reserve` validates
         * it; if the sum exceeds the storage type (idx_t = int32),
         * fail cleanly here.  Reserve before reading u_adj —
         * `qg_reserve` may realloc and invalidate pointers. */
        uint64_t merge_upper_64 = (uint64_t)u_len + (uint64_t)p_len;
        if (merge_upper_64 > (uint64_t)INT32_MAX) {
            free(p_adj_copy);
            return SPARSE_ERR_ALLOC;
        }
        sparse_err_t rc = qg_reserve(qg, (idx_t)merge_upper_64);
        if (rc != SPARSE_OK) {
            free(p_adj_copy);
            return rc;
        }

        idx_t *u_adj = &qg->iw[qg->xadj[u]];
        idx_t *new_adj = &qg->iw[qg->iw_used];
        idx_t new_len = 0;
        idx_t a = 0;
        idx_t b = 0;
        while (a < u_len && b < p_len) {
            idx_t va = u_adj[a];
            idx_t vb = p_adj_copy[b];
            idx_t v;
            if (va == vb) {
                v = va;
                a++;
                b++;
            } else if (va < vb) {
                v = va;
                a++;
            } else {
                v = vb;
                b++;
            }
            if (v != p && v != u && !qg->elim[v])
                new_adj[new_len++] = v;
        }
        while (a < u_len) {
            idx_t v = u_adj[a++];
            if (v != p && v != u && !qg->elim[v])
                new_adj[new_len++] = v;
        }
        while (b < p_len) {
            idx_t v = p_adj_copy[b++];
            if (v != p && v != u && !qg->elim[v])
                new_adj[new_len++] = v;
        }

        qg->xadj[u] = qg->iw_used;
        qg->len[u] = new_len;
        qg->deg[u] = new_len;
        qg->iw_used += new_len;
    }

    /* Mark p's old adjacency as garbage; the next compaction
     * reclaims it. */
    qg->len[p] = 0;
    free(p_adj_copy);
    return SPARSE_OK;
}

sparse_err_t sparse_reorder_amd_qg(const SparseMatrix *A, idx_t *perm) {
    if (!A || !perm)
        return SPARSE_ERR_NULL;
    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;

    idx_t n = sparse_rows(A);
    if (n == 0)
        return SPARSE_OK;
    if (n == 1) {
        perm[0] = 0;
        return SPARSE_OK;
    }

    /* Pre-clear perm so partial-failure paths leave it in a
     * deterministic state. */
    for (idx_t i = 0; i < n; i++)
        perm[i] = 0;

    idx_t *adj_ptr = NULL;
    idx_t *adj_list = NULL;
    sparse_err_t rc = sparse_build_adj(A, &adj_ptr, &adj_list);
    if (rc != SPARSE_OK)
        return rc;

    /* Sprint 24 Day 1: SPARSE_QG_PROFILE wall-clock breakdown.
     * Captures qg_init / per-pivot-pick / per-pivot-eliminate
     * cumulative ns.  The Sprint 22 variable-only quotient graph
     * has no element-side walk in deg recompute, so the
     * eliminate-internal subdivisions Day 1 tracked under the
     * Sprint 23 build don't apply — the totals here are enough
     * to verify the wall-time fix. */
    int prof_enabled = (getenv("SPARSE_QG_PROFILE") != NULL);
    long long prof_t0 = prof_enabled ? qg_prof_now_ns() : 0;

    qg_t qg = {0};
    rc = qg_init(&qg, n, adj_ptr, adj_list);
    free(adj_ptr);
    free(adj_list);
    if (rc != SPARSE_OK)
        return rc;
    qg.prof_enabled = prof_enabled;
    if (prof_enabled)
        qg.prof_init_ns = qg_prof_now_ns() - prof_t0;

    for (idx_t step = 0; step < n; step++) {
        long long pick_t0 = prof_enabled ? qg_prof_now_ns() : 0;
        idx_t p = qg_pick_min_deg(&qg);
        if (prof_enabled)
            qg.prof_pick_ns += qg_prof_now_ns() - pick_t0;
        /* Invariant: at every step there's still at least one
         * unelim vertex (the loop runs `n` steps and we mark
         * exactly one elim per step).  A negative return from
         * `qg_pick_min_deg` means the qg state has been
         * corrupted; assert in debug builds and surface as
         * SPARSE_ERR_BADARG (documented in the header) under
         * NDEBUG so callers can still propagate the failure. */
        assert(p >= 0 && "qg_pick_min_deg: invariant violated — no unelim vertex");
        if (p < 0) {
            qg_free(&qg);
            return SPARSE_ERR_BADARG;
        }
        perm[step] = p;
        long long elim_t0 = prof_enabled ? qg_prof_now_ns() : 0;
        rc = qg_eliminate(&qg, p);
        if (prof_enabled)
            qg.prof_eliminate_ns += qg_prof_now_ns() - elim_t0;
        if (rc != SPARSE_OK) {
            qg_free(&qg);
            return rc;
        }
    }

    if (prof_enabled) {
        long long total_ns = qg.prof_init_ns + qg.prof_pick_ns + qg.prof_eliminate_ns;
        fprintf(stderr,
                "qg-profile n=%d total=%.3fms\n"
                "  init=%.3fms  pick=%.3fms  eliminate=%.3fms\n",
                (int)n, (double)total_ns / 1.0e6, (double)qg.prof_init_ns / 1.0e6,
                (double)qg.prof_pick_ns / 1.0e6, (double)qg.prof_eliminate_ns / 1.0e6);
    }

    qg_free(&qg);
    return SPARSE_OK;
}
