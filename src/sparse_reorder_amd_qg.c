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
#include "sparse_types.h"

sparse_err_t sparse_reorder_amd_qg(const SparseMatrix *A, idx_t *perm) {
    /* Day 10 stub: validate args, return BADARG.  Day 11 replaces
     * the body with the full elimination loop. */
    if (!A || !perm)
        return SPARSE_ERR_NULL;
    if (sparse_rows(A) != sparse_cols(A))
        return SPARSE_ERR_SHAPE;
    /* Pre-clear perm so callers see deterministic empty state on
     * the BADARG return.  Doubles as a clang-tidy hint that perm
     * really is an output parameter (the stub otherwise looks
     * const-able). */
    idx_t n = sparse_rows(A);
    for (idx_t i = 0; i < n; i++)
        perm[i] = 0;
    return SPARSE_ERR_BADARG;
}
