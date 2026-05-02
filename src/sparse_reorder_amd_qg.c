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
#include <stdio.h> /* Day 3: getenv-controlled SPARSE_QG_PROBE stderr trace. */
#include <stdlib.h>
#include <string.h>

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
 *
 * ─── Sprint 23 plan (Days 2-6): full Davis 2006 algorithm ──────────────
 *
 * Sprint 23 closes the four mechanisms Sprint 22's simplification
 * skipped.  Reference: `docs/planning/EPIC_2/SPRINT_23/davis_notes.md`
 * (in-tree design rationale citing book page numbers + algorithm
 * lines).  Workspace + per-vertex state evolves as follows:
 *
 *   Day 2: Extend `qg_t` with `elen[]` (per-vertex element-side
 *          adjacency count) and grow initial `iw_size` from
 *          `5·nnz + 6·n + 1` to `7·nnz + 8·n + 1` (Davis §7's
 *          reference size — buffers the elements-side region).  No
 *          algorithmic change yet; bit-identical fill.
 *   Day 3: Element absorption.  When pivot `p` eliminates, form a
 *          new element `e` (reusing `p`'s slot — Davis's
 *          index-recycling convention) and append `e` to each
 *          neighbour's element-side adjacency.  Variables whose
 *          variable-side adjacency reduces to zero get marked
 *          absorbed; compaction reclaims their slots.
 *   Day 4: Supervariable detection.  Hash signature
 *          `sum_{v in adj(i)} v` per Davis §7.4; bucket by hash,
 *          full compare on hash collision.  Merged supervariables
 *          eliminate together at the next pivot.
 *   Day 5: Approximate-degree formula
 *          `d_approx(i) = |adj_var(i, V \ p)| + |L_p \ {i}|
 *                       + Σ_{e in adj_elem(i) \ L_p}
 *                              |adj(e, V \ p) \ adj(i)|`
 *          (Davis §7.5).  Replaces the Sprint-22 exact recompute.
 *          Conservative bound (`d_approx >= d_exact`) is the test-
 *          side contract Day 6 pins.  Vertices whose post-pivot
 *          `d_approx > 10·√n` skip the update entirely (dense-row
 *          escape).
 *
 * ─── Sprint 23 iw[] layout (post-Day-2) ────────────────────────────────
 *
 * Each variable `i` owns a contiguous slice of `iw[]` of length
 * `len[i]`.  The slice is split into two regions:
 *
 *     iw[xadj[i] ..                   xadj[i] + len[i] - elen[i] - 1]
 *         variable-side adjacency (active variable IDs)
 *     iw[xadj[i] + len[i] - elen[i] .. xadj[i] + len[i] - 1]
 *         element-side adjacency (element IDs from earlier pivots)
 *
 * At init `elen[i] = 0` for every variable — the entire slice is
 * variable-side, matching the Sprint-22 representation byte-for-byte.
 * As pivots eliminate, the elements-side suffix grows; compaction
 * preserves the split (each relocated slice keeps its variable-prefix
 * + element-suffix structure intact via a single memmove of length
 * `len[i]`, since the two regions are contiguous).
 *
 * `super[]` and `super_size[]` (added Day 4) live alongside `xadj[]`
 * etc as length-`n` per-vertex arrays — *not* inside `iw[]`.  The
 * supervariable representative for vertex `i` is `super[i]`; the
 * count of variables in a supervariable is `super_size[rep]` (only
 * meaningful when `rep == super[rep]`).  The minimum-degree pivot
 * scan iterates over representatives only.
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
 * Workspace layout (post-Sprint-23-Day-3):
 *   iw          length iw_size (grows on demand)
 *   xadj        length n
 *   len         length n
 *   elen        length n  (per-vertex element-side adjacency count)
 *   deg         length n
 *   elim        length n  (one byte per vertex)
 *   absorbed    length n  (Day 3: 1 if vertex absorbed by an element)
 *   deg_mark    length n  (Day 3: scratch bitmap for deg recompute)
 *   touched     length n  (Day 3: touched-list for deg-mark cleanup)
 *
 * Each variable's slice in `iw[]` is split: variable-side prefix of
 * length `len[i] - elen[i]`, followed by element-side suffix of
 * length `elen[i]`.  At init `elen[i] == 0` for every variable, so
 * the entire slice is variable-side and the representation is
 * byte-identical to Sprint 22.  Day 3 populates the element-side as
 * element absorption fires.
 *
 * Initial `iw_size` is `7·nnz + 8·n + 1` (Davis 2006 §7's reference
 * form; the extra `2·nnz + 2·n` over Sprint 22's simplified
 * `5·nnz + 6·n + 1` buffers the elements-side region so the first
 * compaction is deferred a few pivots).
 *
 * Element-side encoding: an element ID is just the vertex slot of an
 * already-eliminated variable (Davis's index-recycling convention).
 * To distinguish an element from a variable in `iw[]`'s element-side
 * suffix, check `elim[id]` — eliminated vertices serve as elements.
 * The element's variable-set is stored at `iw[xadj[id]..xadj[id]+len[id]]`.
 *
 * All variable-side adjacency lists are kept sorted ascending; the
 * element-side is sorted by the underlying element-ID (i.e. the
 * eliminated-vertex slot), so both regions remain easy to dedup.
 */
typedef struct {
    idx_t *iw;
    idx_t iw_size;
    idx_t iw_used;
    idx_t iw_peak; /* Day 3: monotonic high-water of iw_used across the run.
                    * Probes element-absorption's workspace flattening — without
                    * absorption iw_used grows steadily through elimination;
                    * with absorption it plateaus once the active set shrinks. */
    idx_t *xadj;
    idx_t *len;
    idx_t *elen; /* Day 2: per-vertex element-side adjacency count. */
    idx_t *deg;
    char *elim;
    char *absorbed; /* Day 3: 1 if vertex was absorbed into an element. */
    char *deg_mark; /* Day 3: scratch bitmap for deg recompute (cleared via `touched[]`). */
    idx_t *touched; /* Day 3: touched-list (entries written to deg_mark). */
    /* Day 3: monotonic list of vertex IDs that go into perm[] right
     * after a pivot but without their own qg_pick_min_deg step.
     * Used for both Davis-style absorption (variable-side empty +
     * one-element element-side) and Day-4 supervariable
     * co-elimination (members of the pivot's supervariable that
     * aren't the representative).  The driver reads the slice
     * `absorbed_list[prev_count, absorbed_count)` after each pivot. */
    idx_t *absorbed_list;
    idx_t absorbed_count;
    /* Day 4: supervariable structure.  `super[i]` is the
     * representative ID of i's supervariable; initially `super[i] =
     * i` (every vertex is its own supervariable).  When variables i
     * and j are merged, `super[j] = i` (or vice versa); only the
     * representative satisfies `super[i] == i` and is visible to
     * `qg_pick_min_deg`.
     *
     * Members are linked in a singly-linked list per supervariable:
     * `super_first[rep]` is the head, `super_next[m]` is the next
     * member, -1 marks list end.  `super_size[rep]` is the member
     * count (only meaningful for representatives). */
    idx_t *super;
    idx_t *super_first;
    idx_t *super_next;
    idx_t *super_size;
    idx_t n;
} qg_t;

static void qg_free(qg_t *qg) {
    if (!qg)
        return;
    free(qg->iw);
    free(qg->xadj);
    free(qg->len);
    free(qg->elen);
    free(qg->deg);
    free(qg->elim);
    free(qg->absorbed);
    free(qg->deg_mark);
    free(qg->touched);
    free(qg->absorbed_list);
    free(qg->super);
    free(qg->super_first);
    free(qg->super_next);
    free(qg->super_size);
    qg->iw = NULL;
    qg->xadj = NULL;
    qg->len = NULL;
    qg->elen = NULL;
    qg->deg = NULL;
    qg->elim = NULL;
    qg->absorbed = NULL;
    qg->deg_mark = NULL;
    qg->touched = NULL;
    qg->absorbed_list = NULL;
    qg->absorbed_count = 0;
    qg->super = NULL;
    qg->super_first = NULL;
    qg->super_next = NULL;
    qg->super_size = NULL;
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
    /* Davis 2006 §7 reference form: `7·nnz + 8·n + 1`.  The extra
     * `2·nnz + 2·n` over Sprint 22's simplified `5·nnz + 6·n + 1`
     * buffers the elements-side region of `iw[]` (Sprint 23 Day 3
     * populates it via element absorption) so the first compaction
     * is deferred a few pivots.  Compute in `uint64_t` and check
     * against the storage type (`idx_t` = int32) plus
     * `SIZE_MAX / sizeof(idx_t)`; under-allocation here would later
     * surface as out-of-bounds writes in the merge / memcpy paths. */
    uint64_t iw_size_64 = (uint64_t)7 * (uint64_t)nnz + (uint64_t)8 * (uint64_t)n + 1;
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
    qg->iw_peak = nnz;
    qg->iw = malloc((size_t)iw_size * sizeof(idx_t));
    qg->xadj = malloc((size_t)n * sizeof(idx_t));
    qg->len = malloc((size_t)n * sizeof(idx_t));
    /* `elen[]` zero-initialised: every variable starts with no
     * element-side adjacency (matches the Sprint-22 representation).
     * Day 3 increments `elen[i]` when an element is appended to i. */
    qg->elen = calloc((size_t)n, sizeof(idx_t));
    qg->deg = malloc((size_t)n * sizeof(idx_t));
    qg->elim = calloc((size_t)n, sizeof(char));
    /* Day 3: absorbed[] is 0-initialised (no vertex starts absorbed).
     * deg_mark[] / touched[] are scratch state used per pivot by
     * qg_recompute_deg; they are cleared via the touched-list so a
     * fresh `calloc` is fine here and per-pivot O(touched) cleanup
     * keeps the bitmap in a known state. */
    qg->absorbed = calloc((size_t)n, sizeof(char));
    qg->deg_mark = calloc((size_t)n, sizeof(char));
    qg->touched = malloc((size_t)n * sizeof(idx_t));
    qg->absorbed_list = malloc((size_t)n * sizeof(idx_t));
    qg->absorbed_count = 0;
    /* Day 4: each vertex starts as its own (singleton) supervariable. */
    qg->super = malloc((size_t)n * sizeof(idx_t));
    qg->super_first = malloc((size_t)n * sizeof(idx_t));
    qg->super_next = malloc((size_t)n * sizeof(idx_t));
    qg->super_size = malloc((size_t)n * sizeof(idx_t));
    if (!qg->iw || !qg->xadj || !qg->len || !qg->elen || !qg->deg || !qg->elim || !qg->absorbed ||
        !qg->deg_mark || !qg->touched || !qg->absorbed_list || !qg->super || !qg->super_first ||
        !qg->super_next || !qg->super_size) {
        qg_free(qg);
        return SPARSE_ERR_ALLOC;
    }
    if (nnz > 0)
        memcpy(qg->iw, adj_list, (size_t)nnz * sizeof(idx_t));
    for (idx_t i = 0; i < n; i++) {
        qg->xadj[i] = adj_ptr[i];
        qg->len[i] = adj_ptr[i + 1] - adj_ptr[i];
        qg->deg[i] = qg->len[i];
        qg->super[i] = i;
        qg->super_first[i] = i;
        qg->super_next[i] = -1;
        qg->super_size[i] = 1;
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

/* Compact the workspace: walk surviving slices (non-empty) in
 * xadj-sort order and pack their adjacency into the front of
 * `iw[]`.  After return, `iw_used` is the new high-water.
 *
 * Sprint 23 Day 3: surviving slices include both active variables
 * (!elim && !absorbed) AND eliminated vertices serving as elements
 * (elim[i] && len[i] > 0).  Absorbed vertices are dropped — their
 * variable-side has been fully absorbed into an element, so their
 * slot can be released.
 *
 * Each slice is `len[i]` entries split into a variable-side prefix
 * of `len[i] - elen[i]` and an element-side suffix of `elen[i]`.
 * Because the two regions are contiguous, the relocation is still a
 * single memmove of `len[i]` entries — the split is preserved
 * automatically.  The debug assertion below catches any future code
 * that lets `elen[i]` exceed `len[i]`. */
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
        /* Keep slices that are still referenced.  Absorbed variables
         * are dropped (their content is fully captured by the
         * element they were absorbed into).  Eliminated vertices
         * with non-zero len[] are elements — they're referenced
         * from active variables' element-side and must stay. */
        if (qg->absorbed[i])
            continue;
        if (qg->len[i] > 0) {
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
        /* Variable/element split invariant: the element-side suffix
         * must fit within the slice. */
        assert(qg->elen[v] <= l);
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

/* Find the unelim, non-absorbed, supervariable-representative vertex
 * with smallest current degree.  Returns -1 iff every vertex is
 * eliminated, absorbed, or non-representative (i.e. the loop's
 * terminal step).  O(active_count) per call.
 *
 * Sprint 23 Day 3 added the absorbed-skip; Day 4 adds the
 * non-representative skip (super[i] != i).  Supervariable members
 * other than the representative are skipped here and instead
 * co-eliminate with the representative when it pivots. */
static idx_t qg_pick_min_deg(const qg_t *qg) {
    idx_t best = -1;
    idx_t best_deg = qg->n + 1;
    for (idx_t i = 0; i < qg->n; i++) {
        if (qg->elim[i] || qg->absorbed[i] || qg->super[i] != i)
            continue;
        if (qg->deg[i] < best_deg) {
            best_deg = qg->deg[i];
            best = i;
        }
    }
    return best;
}

/* Recompute `deg[u]` (true count of distinct active variables u is
 * adjacent to) by walking u's variable-side and expanding each
 * element in u's element-side.  Uses `qg->deg_mark` as a dedup
 * bitmap; `qg->touched` accumulates marked entries so the bitmap
 * can be cleared in O(touched) instead of O(n).
 *
 * Sprint 23 Day 5 will replace this exact recompute with the Davis
 * approximate-degree formula (`d_approx`) for the per-pivot cost
 * win.  Day 3 keeps the exact recompute so fill quality is
 * bit-identical to Sprint 22. */
static void qg_recompute_deg(qg_t *qg, idx_t u) {
    idx_t u_xadj = qg->xadj[u];
    idx_t u_len = qg->len[u];
    idx_t u_elen = qg->elen[u];
    idx_t u_vlen = u_len - u_elen;
    idx_t touched_count = 0;

    /* Variable-side contributes directly. */
    for (idx_t j = 0; j < u_vlen; j++) {
        idx_t v = qg->iw[u_xadj + j];
        if (v == u || qg->elim[v] || qg->absorbed[v])
            continue;
        if (!qg->deg_mark[v]) {
            qg->deg_mark[v] = 1;
            qg->touched[touched_count++] = v;
        }
    }
    /* Element-side: expand each element to its variable-set. */
    for (idx_t j = u_vlen; j < u_len; j++) {
        idx_t e = qg->iw[u_xadj + j];
        idx_t e_xadj = qg->xadj[e];
        idx_t e_vlen = qg->len[e] - qg->elen[e]; /* element's elen is 0 */
        for (idx_t k = 0; k < e_vlen; k++) {
            idx_t v = qg->iw[e_xadj + k];
            if (v == u || qg->elim[v] || qg->absorbed[v])
                continue;
            if (!qg->deg_mark[v]) {
                qg->deg_mark[v] = 1;
                qg->touched[touched_count++] = v;
            }
        }
    }

    qg->deg[u] = touched_count;

    /* Restore the bitmap to all-zero for the next call. */
    for (idx_t i = 0; i < touched_count; i++)
        qg->deg_mark[qg->touched[i]] = 0;
}

/* Sprint 23 Day 4: supervariable detection (Davis 2006 §7.4).
 *
 * Two variables are *indistinguishable* when their adjacencies are
 * identical (after symmetric self-references are accounted for).
 * Indistinguishable variables can be merged into a single
 * supervariable: the minimum-degree pivot scan operates on the
 * representative only, and when the representative pivots all
 * members co-eliminate together.  On PDE-mesh fixtures the
 * supervariable factor is 5–20× by mid-elimination — the headline
 * shrink for the per-pivot active-set scan.
 *
 * The hash signature combines the variable-side and element-side
 * IDs: cheap to compute (O(len)) and a low-collision filter for the
 * O(k²) full-comparison pass within each hash bucket. */
static uint64_t qg_supervariable_hash(const qg_t *qg, idx_t i) {
    idx_t i_xadj = qg->xadj[i];
    idx_t i_len = qg->len[i];
    idx_t i_elen = qg->elen[i];
    idx_t i_vlen = i_len - i_elen;
    uint64_t h_var = 0;
    for (idx_t j = 0; j < i_vlen; j++)
        h_var += (uint64_t)qg->iw[i_xadj + j];
    uint64_t h_elem = 0;
    for (idx_t j = i_vlen; j < i_len; j++)
        h_elem += (uint64_t)qg->iw[i_xadj + j];
    return h_var ^ h_elem;
}

/* Full-compare check: are u and v indistinguishable?  Returns 1 iff
 * the sets `adj(u) ∪ {u}` and `adj(v) ∪ {v}` are equal as multisets
 * (i.e., u in v's adj must be matched by v in u's adj, and the rest
 * of the adjacency lists must be identical).
 *
 * Both lists are sorted ascending so the comparison is a linear
 * two-pointer walk with the mutual self-reference handled inline.
 * Element-sides are compared directly (no self-references — elements
 * are eliminated vertices, not active variables).
 *
 * Caller guarantees u != v and both are non-eliminated, non-absorbed
 * supervariable representatives. */
static int qg_indistinguishable(const qg_t *qg, idx_t u, idx_t v) {
    if (qg->len[u] != qg->len[v])
        return 0;
    if (qg->elen[u] != qg->elen[v])
        return 0;

    idx_t u_vlen = qg->len[u] - qg->elen[u];
    idx_t u_xadj = qg->xadj[u];
    idx_t v_xadj = qg->xadj[v];

    /* Variable-side walk: skip the mutual self-ref (v in u's vars,
     * u in v's vars) and compare the rest. */
    idx_t i = 0, j = 0;
    while (i < u_vlen || j < u_vlen) {
        if (i < u_vlen && qg->iw[u_xadj + i] == v) {
            i++;
            continue;
        }
        if (j < u_vlen && qg->iw[v_xadj + j] == u) {
            j++;
            continue;
        }
        if (i >= u_vlen || j >= u_vlen)
            return 0;
        if (qg->iw[u_xadj + i] != qg->iw[v_xadj + j])
            return 0;
        i++;
        j++;
    }

    /* Element-side: direct compare (sorted; no self-refs). */
    idx_t u_elen = qg->elen[u];
    for (idx_t k = 0; k < u_elen; k++) {
        if (qg->iw[u_xadj + u_vlen + k] != qg->iw[v_xadj + u_vlen + k])
            return 0;
    }
    return 1;
}

/* Merge supervariable b into supervariable a (a stays as the
 * representative, b's members fold into a's list).  Caller
 * guarantees a == super[a] (representative), b == super[b]
 * (representative), a != b. */
static void qg_merge_supervariables_pair(qg_t *qg, idx_t a, idx_t b) {
    /* Walk b's member list, updating super[m] = a. */
    for (idx_t m = qg->super_first[b]; m != -1; m = qg->super_next[m])
        qg->super[m] = a;

    /* Splice b's list onto the front of a's list.  Order within a
     * supervariable doesn't matter algorithmically; we sort by ID
     * at perm[] append time for deterministic output. */
    /* Find b's tail (the last member with super_next == -1). */
    idx_t b_tail = qg->super_first[b];
    while (qg->super_next[b_tail] != -1)
        b_tail = qg->super_next[b_tail];
    qg->super_next[b_tail] = qg->super_first[a];
    qg->super_first[a] = qg->super_first[b];
    qg->super_size[a] += qg->super_size[b];

    /* b is no longer a representative; null its list head and size
     * so a stale qg_pick_min_deg can never accidentally select it
     * (the `super[b] != b` filter is the primary guard). */
    qg->super_first[b] = -1;
    qg->super_size[b] = 0;
}

/* qsort comparator: sort `(hash, vertex)` pairs by hash ascending,
 * then vertex ascending — ID order within a hash bucket gives the
 * pairwise compare a deterministic walk. */
typedef struct {
    uint64_t hash;
    idx_t vertex;
} qg_sv_pair_t;

static int qg_sv_pair_compare(const void *a, const void *b) {
    const qg_sv_pair_t *pa = (const qg_sv_pair_t *)a;
    const qg_sv_pair_t *pb = (const qg_sv_pair_t *)b;
    if (pa->hash < pb->hash)
        return -1;
    if (pa->hash > pb->hash)
        return 1;
    return (pa->vertex > pb->vertex) - (pa->vertex < pb->vertex);
}

/* Run supervariable detection over the candidate set
 * `candidates[0..count)`.  Bucket by hash, full-compare within each
 * bucket, merge confirmed pairs.  Caller guarantees candidates are
 * unique vertex IDs. */
static sparse_err_t qg_merge_supervariables(qg_t *qg, const idx_t *candidates, idx_t count) {
    if (count < 2)
        return SPARSE_OK;

    /* Filter to representatives that are still active. */
    qg_sv_pair_t *pairs = malloc((size_t)count * sizeof(qg_sv_pair_t));
    if (!pairs)
        return SPARSE_ERR_ALLOC;
    idx_t k = 0;
    for (idx_t i = 0; i < count; i++) {
        idx_t c = candidates[i];
        if (qg->elim[c] || qg->absorbed[c])
            continue;
        if (qg->super[c] != c)
            continue;
        pairs[k].hash = qg_supervariable_hash(qg, c);
        pairs[k].vertex = c;
        k++;
    }
    if (k < 2) {
        free(pairs);
        return SPARSE_OK;
    }

    qsort(pairs, (size_t)k, sizeof(qg_sv_pair_t), qg_sv_pair_compare);

    /* Walk sorted pairs; each contiguous same-hash run is a bucket. */
    idx_t bucket_begin = 0;
    while (bucket_begin < k) {
        idx_t bucket_end = bucket_begin + 1;
        while (bucket_end < k && pairs[bucket_end].hash == pairs[bucket_begin].hash)
            bucket_end++;
        if (bucket_end - bucket_begin > 1) {
            /* Pairwise compare within bucket; merge confirmed pairs.
             * The lowest-ID survivor stays as representative. */
            for (idx_t i = bucket_begin; i < bucket_end; i++) {
                idx_t a = pairs[i].vertex;
                /* Skip if a was absorbed into an earlier match this loop. */
                if (qg->super[a] != a)
                    continue;
                for (idx_t j = i + 1; j < bucket_end; j++) {
                    idx_t b = pairs[j].vertex;
                    if (qg->super[b] != b)
                        continue;
                    if (qg_indistinguishable(qg, a, b))
                        qg_merge_supervariables_pair(qg, a, b);
                }
            }
        }
        bucket_begin = bucket_end;
    }

    free(pairs);
    return SPARSE_OK;
}

/* Eliminate pivot `p` (Sprint 23 Day 3 — Davis element representation).
 *
 * Davis Algorithm 7.4 hot loop:
 *   1. Form a new element `e` (reusing p's slot — index recycling).
 *      e's variable-set = (p's variable-side ∪ vars-in-p's-element-side)
 *      \ {p, eliminated, absorbed}.
 *   2. For each neighbour u of p (i.e., variable in e's variable-set):
 *      - Drop p from u's variable-side (if p was a direct neighbour).
 *      - Append e (= p) to u's element-side.
 *      - Recompute deg[u] (exact, via qg_recompute_deg).
 *   3. Check absorption: if u's variable-side is empty AND
 *      u's element-side has converged to a single element, mark
 *      u absorbed.  Absorbed vertices skip qg_pick_min_deg and
 *      have their slots reclaimed by the next compaction.
 *
 * Bit-identical fill rationale: deg[u] is computed as the true
 * distinct-variable count in both Sprint 22's merged representation
 * and Davis's element representation, so qg_pick_min_deg picks the
 * same vertex at each step.  Absorption skips some pivots, but
 * absorbed vertices' adjacency is fully captured by the absorbing
 * element — their pivots would contribute zero new fill in Sprint
 * 22's order anyway, so dropping them preserves nnz(L). */
static int qg_id_compare(const void *a, const void *b) {
    idx_t va = *(const idx_t *)a;
    idx_t vb = *(const idx_t *)b;
    return (va > vb) - (va < vb);
}

static sparse_err_t qg_eliminate(qg_t *qg, idx_t p) {
    /* Day 4: collect p's supervariable members (other than p), mark
     * them eliminated, and append them to `absorbed_list` in ID-sorted
     * order so the driver puts them into perm[] right after p.  This
     * is fill-equivalent to Sprint 22's "pivot u then v then ..." for
     * indistinguishable variables — supervariable elimination just
     * collapses their step into one. */
    idx_t super_extra = qg->super_size[p] - 1;
    if (super_extra > 0) {
        idx_t *members = malloc((size_t)super_extra * sizeof(idx_t));
        if (!members)
            return SPARSE_ERR_ALLOC;
        idx_t mc = 0;
        for (idx_t m = qg->super_first[p]; m != -1; m = qg->super_next[m]) {
            if (m == p)
                continue;
            members[mc++] = m;
        }
        assert(mc == super_extra);
        qsort(members, (size_t)mc, sizeof(idx_t), qg_id_compare);
        for (idx_t i = 0; i < mc; i++) {
            idx_t m = members[i];
            qg->elim[m] = 1;
            qg->len[m] = 0;
            qg->elen[m] = 0;
            qg->deg[m] = 0;
            qg->absorbed_list[qg->absorbed_count++] = m;
        }
        free(members);
    }

    qg->elim[p] = 1;
    idx_t p_len = qg->len[p];
    if (p_len == 0) {
        qg->elen[p] = 0;
        return SPARSE_OK;
    }

    /* Snapshot p's adjacency before we touch `iw[]` — `qg_reserve`
     * may move the buffer underneath us.  Guard the `p_len *
     * sizeof(idx_t)` byte count against `size_t` overflow first. */
    if ((size_t)p_len > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;
    idx_t *p_adj_copy = malloc((size_t)p_len * sizeof(idx_t));
    if (!p_adj_copy)
        return SPARSE_ERR_ALLOC;
    memcpy(p_adj_copy, &qg->iw[qg->xadj[p]], (size_t)p_len * sizeof(idx_t));
    idx_t p_elen = qg->elen[p];
    idx_t p_vlen = p_len - p_elen;

    /* --- Step 1: Compute new element e's variable-set ---
     * e_var_set = (p's vars ∪ vars-via-p's-elements) \ {p, eliminated, absorbed}.
     * Use deg_mark as a temporary bitmap; collect set in `touched[]`.
     * Important: e_var_set replaces the previous "merged adjacency"
     * Sprint 22 stored on each neighbour — it's the new clique
     * formed by p's elimination. */
    idx_t e_count = 0;
    /* Walk p's variable-side. */
    for (idx_t k = 0; k < p_vlen; k++) {
        idx_t v = p_adj_copy[k];
        if (v == p || qg->elim[v] || qg->absorbed[v])
            continue;
        if (!qg->deg_mark[v]) {
            qg->deg_mark[v] = 1;
            qg->touched[e_count++] = v;
        }
    }
    /* Walk p's element-side and expand each. */
    for (idx_t k = p_vlen; k < p_len; k++) {
        idx_t e_old = p_adj_copy[k];
        idx_t e_old_xadj = qg->xadj[e_old];
        idx_t e_old_vlen = qg->len[e_old] - qg->elen[e_old];
        for (idx_t j = 0; j < e_old_vlen; j++) {
            idx_t v = qg->iw[e_old_xadj + j];
            if (v == p || qg->elim[v] || qg->absorbed[v])
                continue;
            if (!qg->deg_mark[v]) {
                qg->deg_mark[v] = 1;
                qg->touched[e_count++] = v;
            }
        }
    }

    /* Snapshot e's variable-set into a fresh buffer.  We need it
     * sorted (for the binary-search-style "is p in u's variable-side"
     * check below, and to keep the compaction's relocation
     * invariants).  `touched[]` is in insertion order, not vertex-ID
     * order, so sort it. */
    idx_t *e_adj = NULL;
    if (e_count > 0) {
        e_adj = malloc((size_t)e_count * sizeof(idx_t));
        if (!e_adj) {
            /* Restore deg_mark before bailing out. */
            for (idx_t i = 0; i < e_count; i++)
                qg->deg_mark[qg->touched[i]] = 0;
            free(p_adj_copy);
            return SPARSE_ERR_ALLOC;
        }
        /* Build sorted e_adj by walking deg_mark in ID order — O(n). */
        idx_t k = 0;
        for (idx_t v = 0; v < qg->n; v++) {
            if (qg->deg_mark[v]) {
                e_adj[k++] = v;
                qg->deg_mark[v] = 0;
            }
        }
        assert(k == e_count);
    } else {
        /* Even when e_count == 0, restore deg_mark for any entries
         * we may have touched (defensive — e_count == 0 here means
         * the touched-list is also empty). */
        for (idx_t i = 0; i < e_count; i++)
            qg->deg_mark[qg->touched[i]] = 0;
    }

    /* --- Step 2-3: Reserve workspace for e's adjacency + write it ---
     * e takes p's slot.  e's adjacency is e_count entries written at
     * iw[iw_used..]; xadj[p] points at the new location, len[p] =
     * e_count, elen[p] = 0.  Old content at p's previous xadj is
     * garbage to be reclaimed by the next compaction. */
    sparse_err_t rc = qg_reserve(qg, e_count);
    if (rc != SPARSE_OK) {
        free(e_adj);
        free(p_adj_copy);
        return rc;
    }
    idx_t e_xadj = qg->iw_used;
    if (e_count > 0) {
        memcpy(&qg->iw[e_xadj], e_adj, (size_t)e_count * sizeof(idx_t));
        qg->iw_used += e_count;
    }
    qg->xadj[p] = e_xadj;
    qg->len[p] = e_count;
    qg->elen[p] = 0;

    /* --- Step 4: For each variable u in e's variable-set, rebuild
     *     u's slice and recompute deg[u] ---
     * u's new slice:
     *   variable-side = u_old_var_side \ {p}  (drop p if direct)
     *   element-side = u_old_elem_side ⋃ {e=p}
     * Followed by absorption check.
     *
     * Note: e_adj may be NULL when e_count == 0; the loop is
     * a no-op in that case. */
    for (idx_t v_idx = 0; v_idx < e_count; v_idx++) {
        idx_t u = e_adj[v_idx];
        idx_t u_xadj_old = qg->xadj[u];
        idx_t u_len_old = qg->len[u];
        idx_t u_elen_old = qg->elen[u];
        idx_t u_vlen_old = u_len_old - u_elen_old;

        /* Reserve u_len_old + 1 for the rebuilt slice (may be a few
         * entries shorter if p was in u's variable-side, but +1 is
         * a safe upper bound). */
        if ((uint64_t)u_len_old + 1 > (uint64_t)INT32_MAX) {
            free(e_adj);
            free(p_adj_copy);
            return SPARSE_ERR_ALLOC;
        }
        rc = qg_reserve(qg, u_len_old + 1);
        if (rc != SPARSE_OK) {
            free(e_adj);
            free(p_adj_copy);
            return rc;
        }
        /* qg_reserve may have moved iw[] — re-fetch u_xadj_old. */
        u_xadj_old = qg->xadj[u];

        idx_t new_xadj = qg->iw_used;
        idx_t new_vlen = 0;
        for (idx_t j = 0; j < u_vlen_old; j++) {
            idx_t v = qg->iw[u_xadj_old + j];
            if (v == p)
                continue; /* drop the just-eliminated pivot */
            /* Day 4: also drop stale eliminated/absorbed entries.
             * Co-eliminated supervariable members were marked
             * elim[]=1 at the top of this function; they're stale
             * here because they were in u's adjacency at slice-
             * build time but are no longer active. */
            if (qg->elim[v] || qg->absorbed[v])
                continue;
            qg->iw[qg->iw_used++] = v;
            new_vlen++;
        }
        idx_t new_elen = 0;
        for (idx_t j = u_vlen_old; j < u_len_old; j++) {
            idx_t e_id = qg->iw[u_xadj_old + j];
            qg->iw[qg->iw_used++] = e_id;
            new_elen++;
        }
        /* Append the new element e (= p's slot) to u's element-side. */
        qg->iw[qg->iw_used++] = p;
        new_elen++;

        qg->xadj[u] = new_xadj;
        qg->len[u] = new_vlen + new_elen;
        qg->elen[u] = new_elen;

        qg_recompute_deg(qg, u);

        /* --- Step 5: Absorption check ---
         * u is absorbed iff u's variable-side is empty AND u's
         * element-side has converged to a single element (Davis §7.3).
         * The single-element case is when new_elen == 1 (only e=p in
         * u's element-side); for new_elen > 1, u still depends on
         * older elements that may bring in distinct variables. */
        if (new_vlen == 0 && new_elen == 1) {
            qg->absorbed[u] = 1;
            qg->absorbed_list[qg->absorbed_count++] = u;
            /* Free u's slot for compaction reclaim — len[u] = 0
             * keeps the slot from being relocated by qg_compact. */
            qg->len[u] = 0;
            qg->elen[u] = 0;
            qg->deg[u] = 0;
        }
    }

    if (qg->iw_used > qg->iw_peak)
        qg->iw_peak = qg->iw_used;

    /* Day 4: detect new supervariable merges among e's neighbours.
     * After p's elimination changed their adjacencies, two
     * previously-distinct variables may now be indistinguishable.
     * Merging them shrinks the active set for subsequent
     * qg_pick_min_deg scans and lets co-elimination happen on the
     * next pivot step. */
    sparse_err_t merge_rc = qg_merge_supervariables(qg, e_adj, e_count);

    free(e_adj);
    free(p_adj_copy);
    return merge_rc;
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

    qg_t qg = {0};
    rc = qg_init(&qg, n, adj_ptr, adj_list);
    free(adj_ptr);
    free(adj_list);
    if (rc != SPARSE_OK)
        return rc;

    /* Day 4: initial supervariable detection.  Vertices with
     * identical original adjacency (e.g., the leaves of a star)
     * form supervariables before any pivot fires.  Run the merge
     * over all n vertices once at startup; subsequent merges only
     * need to consider the per-pivot e_var_set since untouched
     * adjacencies don't change their (non-)indistinguishability. */
    {
        idx_t *all_candidates = malloc((size_t)n * sizeof(idx_t));
        if (!all_candidates) {
            qg_free(&qg);
            return SPARSE_ERR_ALLOC;
        }
        for (idx_t i = 0; i < n; i++)
            all_candidates[i] = i;
        rc = qg_merge_supervariables(&qg, all_candidates, n);
        free(all_candidates);
        if (rc != SPARSE_OK) {
            qg_free(&qg);
            return rc;
        }
    }

    /* Sprint 23 Day 3: each pivot may absorb additional vertices
     * (variables fully subsumed by the new element).  Absorbed
     * vertices appear in `perm[]` immediately after the pivot
     * that absorbed them — their fill contribution is identical
     * to the absorbing element's so the relative order within an
     * absorbed group doesn't affect nnz(L).  The step counter
     * advances by `(1 + absorbed_this_pivot)` per outer iteration. */
    idx_t step = 0;
    while (step < n) {
        idx_t p = qg_pick_min_deg(&qg);
        /* Invariant: at every step there's still at least one
         * non-eliminated, non-absorbed vertex (the loop has
         * placed `step` vertices into perm[] and `step < n`).
         * A negative return from `qg_pick_min_deg` means the qg
         * state has been corrupted; assert in debug builds and
         * surface as SPARSE_ERR_BADARG (documented in the header)
         * under NDEBUG so callers can still propagate the failure. */
        assert(p >= 0 && "qg_pick_min_deg: invariant violated — no unelim vertex");
        if (p < 0) {
            qg_free(&qg);
            return SPARSE_ERR_BADARG;
        }

        idx_t absorbed_before = qg.absorbed_count;
        perm[step++] = p;
        rc = qg_eliminate(&qg, p);
        if (rc != SPARSE_OK) {
            qg_free(&qg);
            return rc;
        }

        /* Append vertices absorbed by THIS pivot to perm[].
         * absorbed_list grows monotonically; the slice
         * [absorbed_before, absorbed_count) is fresh. */
        for (idx_t i = absorbed_before; i < qg.absorbed_count; i++) {
            assert(step < n);
            perm[step++] = qg.absorbed_list[i];
        }
    }

    /* Day 3 workspace probe: print iw_used high-water and absorbed
     * count when `SPARSE_QG_PROBE` is set in the environment.  Used
     * once per sprint to verify element-absorption shrinks the
     * active set; off in production.  No allocation cost when the
     * env var is unset (single getenv call). */
    if (getenv("SPARSE_QG_PROBE")) {
        fprintf(stderr, "qg-probe n=%d iw_peak=%d iw_size=%d absorbed=%d (%.1f%% of n)\n", (int)n,
                (int)qg.iw_peak, (int)qg.iw_size, (int)qg.absorbed_count,
                100.0 * (double)qg.absorbed_count / (double)n);
    }

    qg_free(&qg);
    return SPARSE_OK;
}
