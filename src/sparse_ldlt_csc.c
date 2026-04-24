/**
 * @file sparse_ldlt_csc.c
 * @brief CSC working-format numeric backend for symmetric indefinite
 *        LDL^T factorization.
 *
 * ─── Native kernel ────────────────────────────────────────────────────
 *
 * `ldlt_csc_eliminate` runs column-by-column Bunch-Kaufman directly on
 * the packed CSC arrays: four-criteria pivot selection (α = (1 + √17)/8),
 * in-place symmetric row/column swaps, element-growth tracking, and
 * 1×1 / 2×2 block pivots.  The column-sweep structure mirrors
 * `chol_csc_eliminate`, with the Bunch-Kaufman scan/swap/cmod
 * replacing Cholesky's cdiv at each step.  No linked-list round-trip
 * anywhere — the kernel operates on `CholCsc` row/value arrays plus
 * the auxiliary `D`, `D_offdiag`, `pivot_size`, and `perm` slots
 * defined in `LdltCsc`.
 *
 * Sprint 19 Day 8-9 added a per-row adjacency index (`row_adj[r]`,
 * `row_adj_count[r]`, `row_adj_cap[r]` on `LdltCsc`) populated by
 * `ldlt_csc_eliminate_native` after each column writeback.
 * `ldlt_csc_cmod_unified`'s Phase A (1×1 prior contributions) and
 * Phase B (2×2 cross-terms) iterate `F->row_adj[col]` instead of
 * `[0, step_k)` with a binary search per prior — restoring the
 * sparse-row scaling the linked-list reference's `acc_schur_col`
 * provides via its cross-linked SparseMatrix row chain.
 *
 * Factor times (see `benchmarks/bench_ldlt_csc.c`,
 * `docs/planning/EPIC_2/SPRINT_19/bench_day14.txt`, and
 * `docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md`): 1.7-3.5× faster
 * than the linked-list `sparse_ldlt_factor` on bcsstk04 (1.7×) and
 * bcsstk14 (3.5×), one-shot factor with AMD inside the timed region
 * on both sides.
 *
 * ─── Legacy wrapper path (tests/benchmarks only) ──────────────────────
 *
 * A secondary body, `ldlt_csc_eliminate_wrapper`, remains compiled in
 * so the benchmark and a handful of regression tests can A/B-test the
 * native kernel against the Sprint 17 expand-and-delegate path.  That
 * body builds a full symmetric `SparseMatrix`, calls
 * `sparse_ldlt_factor`, and unpacks the result back into CSC — bit-
 * identical output, ~2× slower on anything larger than bcsstk04.  No
 * production call site reaches it.  Selection between the two is a
 * runtime override (`ldlt_csc_set_kernel_override`); the compile-
 * time `-DLDLT_CSC_USE_NATIVE=0` fallback flips the default for
 * emergency debugging.
 *
 * ─── Storage layout ───────────────────────────────────────────────────
 *
 * `LdltCsc` mirrors `sparse_ldlt_t`'s pivot conventions so the solve
 * code paths can share idioms:
 *
 *   L          — unit lower triangular factor, stored in a `CholCsc`.
 *                Diagonal entries are stored as 1.0 (for CSC invariant
 *                uniformity); below-diagonal entries are the actual L
 *                multipliers.
 *   D          — diagonal of D, length n.  For a 1×1 pivot at step k,
 *                D[k] is the scalar.  For a 2×2 pivot at (k, k+1),
 *                D[k] and D[k+1] hold the 2×2 block's diagonal.
 *   D_offdiag  — off-diagonal of 2×2 pivots, length n.  D_offdiag[k] =
 *                D(k, k+1) = D(k+1, k) when pivot_size[k] == 2.  Zero
 *                for 1×1 pivots.
 *   pivot_size — length n.  1 for a 1×1 pivot at step k, or 2 for both
 *                indices of a 2×2 pair (pivot_size[k] == pivot_size[k+1]
 *                == 2).
 *   perm       — length n, composed symmetric permutation such that
 *                `perm[k]` maps factorization-order index k back to the
 *                user's original row/column index.  Combines any
 *                fill-reducing input permutation with the Bunch-
 *                Kaufman pivot permutation chosen during elimination.
 *
 * ─── Worked solve example ─────────────────────────────────────────────
 *
 * For P·A·P^T = L·D·L^T, solving A·x = b proceeds in five phases:
 *
 *   1. y[i] = b[perm[i]]                        (apply P to b)
 *   2. Solve L·w = y                            (forward on unit L)
 *   3. Solve D·z = w                            (block-diagonal):
 *        1×1 block at k : z[k] = w[k] / D[k]
 *        2×2 block at k : det = D[k]·D[k+1] - D_offdiag[k]²
 *                         z[k]   = ( D[k+1]·w[k]   - D_offdiag[k]·w[k+1]) / det
 *                         z[k+1] = (-D_offdiag[k]·w[k] +      D[k]·w[k+1]) / det
 *   4. Solve L^T·v = z                          (backward on unit L^T)
 *   5. x[perm[i]] = v[i]                        (apply P^T to v)
 *
 * Both triangular sweeps walk the CSC column slice once each, skipping
 * the unit diagonal (same structure as `chol_csc_solve` but without the
 * diagonal division).
 *
 * ─── Batched supernodal LDL^T (Sprint 19 Days 10-14) ──────────────────
 *
 * The Sprint 18 Cholesky batched supernodal path (Days 6-10 on the
 * Cholesky side) does four things per detected supernode: extract the
 * diagonal block + panel into a dense column-major buffer, factor the
 * diagonal block with `chol_dense_factor`, solve the panel via
 * `chol_dense_solve_lower`, then gather the result back.  The LDL^T
 * version mirrors that structure but has one extra constraint: a 2×2
 * pivot block spans two columns (k, k+1), and the batched
 * `eliminate_diag` needs to see both columns together to run
 * Bunch-Kaufman correctly.  That means a 2×2 pivot pair cannot be
 * split across a supernode boundary — if the fundamental-supernode
 * detection would end a supernode at column `end-1` where `end-1` is
 * the first of a 2×2 pair, the boundary must move (either to include
 * `end` too, or to exclude `end-1`).
 *
 * **Two-pass model.**  The atomicity check needs `pivot_size[]`, which
 * only exists after a first factor runs.  The intended workflow for
 * the batched path is therefore: (1) run `ldlt_csc_eliminate_native`
 * once to populate `pivot_size[]`; (2) call
 * `ldlt_csc_detect_supernodes` to identify 2×2-safe supernodes from
 * the factored pivot pattern; (3) on subsequent refactorizations with
 * the same sparsity pattern and a structurally-identical pivot choice
 * (warm state), use the batched path.  The two-pass model matches the
 * Cholesky batched supernodal's analyze-once / factor-many workflow
 * and keeps the detection logic purely structural — no numerical
 * decisions during supernode selection.
 *
 * An alternative "rollback" design, where supernodes are detected
 * during the first factor and rolled back if a 2×2 pivot crosses a
 * tentative boundary, was considered and rejected: it requires two
 * writeback code paths (batched success, rollback to scalar), and
 * the Cholesky two-pass analogue already sets the precedent.
 *
 * **Detection rules (`ldlt_csc_detect_supernodes`).**  Start with the
 * Liu-Ng-Peyton three-condition scan from `chol_csc_detect_supernodes`
 * operating on `F->L`.  After each tentative supernode `[j, end)`:
 *
 *   1. If `end-1` is the first of a 2×2 pair (`pivot_size[end-1] == 2`
 *      AND (`end-1 == 0` or `pivot_size[end-2] == 1`)), extend by 1
 *      so the pair's second column is included — provided the
 *      Liu-Ng-Peyton pattern check between `end-1` and `end` still
 *      holds.  If it doesn't, retract `end-- ` to exclude the 2×2's
 *      first column entirely; the column pair gets factored by the
 *      scalar kernel.
 *
 *   2. At the start of the next iteration, if `j` is the second of a
 *      2×2 pair (`pivot_size[j] == 2` AND `pivot_size[j-1] == 2`),
 *      skip it — it belongs to the 2×2 unit that the prior iteration
 *      already decided about, and starting a supernode mid-pair is
 *      invalid.  Advance `j` by 1 and continue.
 *
 * Size gating by `min_size` happens after boundary adjustment.  Every
 * emitted supernode therefore satisfies: (a) both boundary columns
 * are on a 1×1-or-2×2-pair boundary, and (b) `size >= min_size`.
 *
 * **Sprint 19 Day 11 — Dense LDL^T primitive.**  `ldlt_dense_factor`
 * (in `src/sparse_chol_csc.c`) runs Bunch-Kaufman on a dense column-
 * major buffer, returning the unit lower-triangular L plus per-column
 * D / D_offdiag / pivot_size.  Used by Day 13's `eliminate_diag` per
 * supernode block.
 *
 * **Sprint 19 Day 12 — Extract / writeback plumbing.**
 * `ldlt_csc_supernode_extract` and `_writeback` move the supernode's
 * diagonal block + panel between packed CSC (`F->L`) and a dense
 * column-major scratch.  Writeback applies a per-column drop rule
 * scaled to `|D[k]|` for 1×1 pivots and `|d11| + |d22| + |d21|` for
 * 2×2 pivots, matching the scalar `ldlt_csc_eliminate_native`'s
 * `chol_csc_gather` invocations.  See declarations in
 * `src/sparse_ldlt_csc_internal.h`.
 *
 * **Sprint 19 Day 13 — eliminate_diag / eliminate_panel /
 * eliminate_supernodal.**  Wire the four helpers together:
 *   1. `extract`: A → dense buffer.
 *   2. `eliminate_diag`: external cmod from priors `[0, s_start)`
 *      (handles 1×1 and 2×2 prior cmod), then `ldlt_dense_factor`
 *      on the diagonal slab.
 *   3. `eliminate_panel`: per-row two-phase solve
 *      `D_block^{-1} · L_diag^{-1} · panel_row^T`.
 *   4. `writeback`: dense → CSC with the per-column drop rule.
 * `ldlt_csc_eliminate_supernodal` interleaves this batched path with
 * the scalar `ldlt_csc_eliminate_one_step` for non-supernodal columns
 * (singletons + columns outside detected supernodes).
 *
 * **Production scope (Sprint 19 end).**  The batched supernodal LDL^T
 * path delivers strong speedups on SPD matrices (bcsstk14: 6.83×
 * vs linked-list, bcsstk04: 3.05×, nos4: 2.62× — Day 14 bench).
 * Indefinite matrices that need 2×2 pivots also work *when* the
 * heuristic CSC fill from `ldlt_csc_from_sparse` covers the
 * supernodal cmod's structural fill.  KKT-style saddle points and
 * other matrices with non-trivial off-block structure can produce
 * cmod fill rows that the heuristic slot lacks; the writeback then
 * silently drops them, yielding an incorrect factor.  Same root
 * cause as the Cholesky path's pre-Sprint-18-Day-6 Kuu regression,
 * resolved on the Cholesky side via
 * `chol_csc_from_sparse_with_analysis`.  An
 * `ldlt_csc_from_sparse_with_analysis` mirror is the natural Sprint
 * 20 follow-up — until then, callers needing batched LDL^T on
 * indefinite matrices should fall back to the scalar
 * `ldlt_csc_eliminate` path.  Test coverage: SPD-only batched
 * checks live in `tests/test_sprint19_integration.c`; the indefinite
 * batched path is exercised by the random 30×30 cross-check in
 * `tests/test_ldlt_csc.c` (where heuristic fill happens to suffice).
 */

#include "sparse_ldlt.h"
#include "sparse_ldlt_csc_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ─── Free ───────────────────────────────────────────────────────────── */

void ldlt_csc_free(LdltCsc *m) {
    if (!m)
        return;
    chol_csc_free(m->L);
    free(m->D);
    free(m->D_offdiag);
    free(m->pivot_size);
    free(m->perm);
    /* Sprint 19 Day 8: release per-row adjacency lists.  Each entry in
     * `row_adj` is NULL until a column gets appended to that row.
     * `m->row_adj` is alloc'd with length `max(m->n, 1)` by
     * `ldlt_csc_alloc`, so iterating `[0, m->n)` on a non-NULL
     * `m->row_adj` is always in bounds — clang-analyzer can't see
     * that invariant across the alloc/free boundary. */
    if (m->row_adj) {
        for (idx_t r = 0; r < m->n; r++)
            free(m->row_adj[r]); // NOLINT(clang-analyzer-security.ArrayBound)
        free(m->row_adj);
    }
    free(m->row_adj_count);
    free(m->row_adj_cap);
    free(m);
}

/* ─── Allocate ───────────────────────────────────────────────────────── */

sparse_err_t ldlt_csc_alloc(idx_t n, idx_t initial_nnz, LdltCsc **out) {
    if (!out)
        return SPARSE_ERR_NULL;
    *out = NULL;
    if (n < 0)
        return SPARSE_ERR_BADARG;

    /* Overflow guards for byte counts (n known non-negative above). */
    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    if ((size_t)n > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;

    LdltCsc *m = calloc(1, sizeof(LdltCsc));
    if (!m)
        return SPARSE_ERR_ALLOC;

    m->n = n;
    m->factor_norm = 0.0;

    sparse_err_t err = chol_csc_alloc(n, initial_nnz, &m->L);
    if (err != SPARSE_OK) {
        ldlt_csc_free(m);
        return err;
    }

    /* Always allocate at least one slot so array pointers are non-NULL
     * even for n == 0 — keeps invariant checks simple. */
    size_t alloc_n = n > 0 ? (size_t)n : 1;
    m->D = calloc(alloc_n, sizeof(double));
    m->D_offdiag = calloc(alloc_n, sizeof(double));
    m->pivot_size = calloc(alloc_n, sizeof(idx_t));
    m->perm = calloc(alloc_n, sizeof(idx_t));
    /* Sprint 19 Day 8: row-adjacency index starts with all rows empty.
     * Per-row arrays allocated lazily by `ldlt_csc_row_adj_append` on
     * first append to each row.  `calloc` zeros all three so every
     * row_adj[r] slot is NULL and every count/cap is 0 until written. */
    m->row_adj = calloc(alloc_n, sizeof(idx_t *));
    m->row_adj_count = calloc(alloc_n, sizeof(idx_t));
    m->row_adj_cap = calloc(alloc_n, sizeof(idx_t));
    if (!m->D || !m->D_offdiag || !m->pivot_size || !m->perm || !m->row_adj || !m->row_adj_count ||
        !m->row_adj_cap) {
        ldlt_csc_free(m);
        return SPARSE_ERR_ALLOC;
    }

    /* Defaults: every step is a 1x1 pivot; perm is the identity.  Both
     * are overwritten during elimination in Day 8. */
    for (idx_t i = 0; i < n; i++) {
        m->pivot_size[i] = 1;
        m->perm[i] = i;
    }

    *out = m;
    return SPARSE_OK;
}

/* ─── Row-adjacency append (Sprint 19 Day 8) ─────────────────────── */

sparse_err_t ldlt_csc_row_adj_append(LdltCsc *F, idx_t row, idx_t col) {
    if (!F)
        return SPARSE_ERR_NULL;
    if (row < 0 || row >= F->n || col < 0 || col >= F->n)
        return SPARSE_ERR_BADARG;

    idx_t cap = F->row_adj_cap[row];
    idx_t count = F->row_adj_count[row];
    if (count >= cap) {
        /* Geometric growth (2×), starting at 4 for first-touch rows so
         * short row-adjacency lists don't pay a per-append reallocation
         * when the fill pattern is modest. */
        idx_t new_cap = cap > 0 ? cap * 2 : 4;
        if (new_cap > INT32_MAX)
            return SPARSE_ERR_ALLOC;
        if ((size_t)new_cap > SIZE_MAX / sizeof(idx_t))
            return SPARSE_ERR_ALLOC;
        idx_t *resized = realloc(F->row_adj[row], (size_t)new_cap * sizeof(idx_t));
        if (!resized)
            return SPARSE_ERR_ALLOC;
        F->row_adj[row] = resized;
        F->row_adj_cap[row] = new_cap;
    }
    F->row_adj[row][count] = col;
    F->row_adj_count[row] = count + 1;
    return SPARSE_OK;
}

/* ─── 2×2-aware supernode detection (Sprint 19 Day 10) ─────────────── */

/* Return 1 iff columns `prev` and `prev + 1` belong to the same
 * fundamental supernode of `L` (Liu-Ng-Peyton three-condition check,
 * same test as `chol_csc.c`'s static helper; duplicated here to keep
 * the LDL^T side loosely coupled). */
static int ldlt_csc_same_supernode(const CholCsc *L, idx_t prev) {
    idx_t curr = prev + 1;
    idx_t prev_start = L->col_ptr[prev];
    idx_t prev_end = L->col_ptr[prev + 1];
    idx_t curr_start = L->col_ptr[curr];
    idx_t curr_end = L->col_ptr[curr + 1];

    idx_t prev_size = prev_end - prev_start;
    idx_t curr_size = curr_end - curr_start;

    if (prev_size < 2)
        return 0;
    if (L->row_idx[prev_start + 1] != curr)
        return 0;
    if (curr_size != prev_size - 1)
        return 0;

    idx_t tail_len = curr_size - 1;
    for (idx_t t = 0; t < tail_len; t++) {
        if (L->row_idx[curr_start + 1 + t] != L->row_idx[prev_start + 2 + t])
            return 0;
    }
    return 1;
}

/* A column `k` is "the first of a 2×2 pair" when `pivot_size[k] == 2`
 * and `k == 0 || pivot_size[k-1] == 1`.  The second of the pair
 * always follows immediately at `k + 1`. */
static int ldlt_csc_is_first_of_2x2(const idx_t *pivot_size, idx_t k) {
    if (pivot_size[k] != 2)
        return 0;
    if (k == 0)
        return 1;
    return pivot_size[k - 1] == 1;
}

/* A column `k` is "the second of a 2×2 pair" when both `pivot_size[k]`
 * and `pivot_size[k-1]` are 2 AND `pivot_size[k-1]` itself is the
 * first of a 2×2 (recursively — but the pairing is always adjacent,
 * so this simplifies to checking the window [k-2, k-1, k]).  For our
 * forward scan we call this only when k >= 1. */
static int ldlt_csc_is_second_of_2x2(const idx_t *pivot_size, idx_t k) {
    if (k < 1)
        return 0;
    if (pivot_size[k] != 2 || pivot_size[k - 1] != 2)
        return 0;
    /* pivot_size[k-2] == 1 confirms the 2×2 starts at k-1, not k-2.
     * k == 1 implies k-1 == 0 which is by definition the first of a
     * 2×2 pair, so k is the second. */
    if (k == 1)
        return 1;
    return pivot_size[k - 2] == 1;
}

sparse_err_t ldlt_csc_detect_supernodes(const LdltCsc *F, idx_t min_size, idx_t *super_starts,
                                        idx_t *super_sizes, idx_t *count) {
    if (!F || !F->L || !F->pivot_size || !super_starts || !super_sizes || !count)
        return SPARSE_ERR_NULL;
    if (min_size < 1)
        return SPARSE_ERR_BADARG;

    idx_t n = F->n;
    const CholCsc *L = F->L;
    const idx_t *pivot_size = F->pivot_size;
    idx_t written = 0;
    idx_t j = 0;

    while (j < n) {
        /* Skip a column if it's the second of a 2×2 pair — the prior
         * iteration already decided what to do with the pair, and we
         * must not start a supernode mid-pair (see the Day 10 design
         * block for atomicity). */
        if (ldlt_csc_is_second_of_2x2(pivot_size, j)) {
            j++;
            continue;
        }

        /* Extend the supernode as long as the Liu-Ng-Peyton pattern
         * check allows. */
        idx_t end = j + 1;
        while (end < n && ldlt_csc_same_supernode(L, end - 1))
            end++;

        /* 2×2 atomicity at the upper boundary: `end-1` must not be
         * the first of a 2×2 pair.  If it is, try to extend by 1 to
         * include the pair's second column; if the pattern blocks
         * the extension, retract `end-- ` so the 2×2 stays outside
         * the supernode (scalar handles it). */
        if (end - 1 >= j && ldlt_csc_is_first_of_2x2(pivot_size, end - 1)) {
            if (end < n && ldlt_csc_same_supernode(L, end - 1)) {
                end++;
            } else {
                end--;
            }
        }

        idx_t size = end - j;
        if (size >= min_size) {
            super_starts[written] = j;
            super_sizes[written] = size;
            written++;
        }

        /* Ensure progress even when the atomicity retraction produced
         * an empty supernode (end <= j): advance `j` by 1 so scalar
         * handles this column and we keep scanning. */
        if (end <= j) {
            j = j + 1;
        } else {
            j = end;
        }
    }

    *count = written;
    return SPARSE_OK;
}

/* ─── Convert SparseMatrix → LdltCsc ─────────────────────────────────── */

sparse_err_t ldlt_csc_from_sparse(const SparseMatrix *mat, const idx_t *perm_in, double fill_factor,
                                  LdltCsc **ldlt_out) {
    if (!ldlt_out)
        return SPARSE_ERR_NULL;
    *ldlt_out = NULL;
    if (!mat)
        return SPARSE_ERR_NULL;
    if (mat->rows != mat->cols)
        return SPARSE_ERR_SHAPE;

    /* Reject factored / non-identity-perm matrices up front — the same
     * precondition `sparse_ldlt_factor` enforces.  Without this, the
     * `sparse_is_symmetric` check below walks physical storage while
     * `chol_csc_from_sparse` later walks logical storage via
     * inv_row_perm / inv_col_perm, so the two views could disagree on
     * a matrix with a non-identity internal permutation. */
    idx_t n = mat->rows;
    if (mat->factored)
        return SPARSE_ERR_BADARG;
    {
        const idx_t *rp = sparse_row_perm(mat);
        const idx_t *cp = sparse_col_perm(mat);
        for (idx_t i = 0; i < n; i++) {
            if ((rp && rp[i] != i) || (cp && cp[i] != i))
                return SPARSE_ERR_BADARG;
        }
    }

    /* LDL^T requires a symmetric input; reject non-symmetric A the same
     * way the linked-list `sparse_ldlt_factor` does, with the shared
     * SPARSE_ERR_NOT_SPD code (that enum also covers "not symmetric"
     * per sparse_ldlt.h's documented contract). */
    if (!sparse_is_symmetric(mat, 1e-12))
        return SPARSE_ERR_NOT_SPD;

    /* Build L via the Cholesky CSC converter; this validates perm_in and
     * caches L->factor_norm.  Errors propagate unchanged. */
    CholCsc *L = NULL;
    sparse_err_t err = chol_csc_from_sparse(mat, perm_in, fill_factor, &L);
    if (err != SPARSE_OK)
        return err;

    LdltCsc *m = calloc(1, sizeof(LdltCsc));
    if (!m) {
        chol_csc_free(L);
        return SPARSE_ERR_ALLOC;
    }
    m->n = n;
    m->L = L;
    m->factor_norm = L->factor_norm;

    size_t alloc_n = n > 0 ? (size_t)n : 1;
    m->D = calloc(alloc_n, sizeof(double));
    m->D_offdiag = calloc(alloc_n, sizeof(double));
    m->pivot_size = calloc(alloc_n, sizeof(idx_t));
    m->perm = calloc(alloc_n, sizeof(idx_t));
    /* Sprint 19 Day 8: row-adjacency index — same calloc-zero initial
     * state as `ldlt_csc_alloc` so Day 9's cmod_unified can read
     * row_adj_count[col] == 0 and short-circuit before any column has
     * been factored. */
    m->row_adj = calloc(alloc_n, sizeof(idx_t *));
    m->row_adj_count = calloc(alloc_n, sizeof(idx_t));
    m->row_adj_cap = calloc(alloc_n, sizeof(idx_t));
    if (!m->D || !m->D_offdiag || !m->pivot_size || !m->perm || !m->row_adj || !m->row_adj_count ||
        !m->row_adj_cap) {
        ldlt_csc_free(m);
        return SPARSE_ERR_ALLOC;
    }

    /* Initial pivot_size is 1 everywhere; elimination overwrites. */
    for (idx_t i = 0; i < n; i++)
        m->pivot_size[i] = 1;

    /* Initial perm: the caller-supplied fill-reducing permutation if any,
     * else identity.  Bunch-Kaufman pivoting (Day 8) composes further
     * swaps into this array. */
    if (perm_in) {
        for (idx_t i = 0; i < n; i++)
            m->perm[i] = perm_in[i];
    } else {
        for (idx_t i = 0; i < n; i++)
            m->perm[i] = i;
    }

    *ldlt_out = m;
    return SPARSE_OK;
}

/* ─── Public: symbolic-analysis-aware LDL^T conversion ───────────────── */

/* Sprint 20 Days 1-2: design + implementation.
 *
 * `ldlt_csc_from_sparse_with_analysis` mirrors
 * `chol_csc_from_sparse_with_analysis` (Sprint 18 Day 12 / Sprint 19
 * Day 6) for the LDL^T side.  It pre-allocates the embedded `L` with
 * every column's full sym_L pattern rather than the heuristic
 * `fill_factor × A.nnz` pattern that `ldlt_csc_from_sparse` produces.
 * This closes the indefinite-fill hole documented in the Sprint 19
 * NOTE in `tests/test_sprint19_integration.c` (search for
 * "NOTE on the indefinite supernodal path's current scope"): the
 * batched `ldlt_csc_eliminate_supernodal` writeback silently dropped
 * cmod fill rows on KKT-style saddle points, producing residuals of
 * 1e-2..1e-6 instead of round-off.
 *
 * Symbolic-pattern reuse:
 *   `sparse_analyze` treats `SPARSE_FACTOR_CHOLESKY` and
 *   `SPARSE_FACTOR_LDLT` identically for the symbolic pipeline (see
 *   the shared `case SPARSE_FACTOR_CHOLESKY: case SPARSE_FACTOR_LDLT:`
 *   dispatch in `src/sparse_analysis.c`): both run
 *   `sparse_etree_compute` → `sparse_colcount` →
 *   `sparse_symbolic_cholesky` on the symmetric input and produce
 *   identical `sym_L` patterns.  This function therefore accepts
 *   either type without extra per-column buffering.
 *
 * 2×2-pivot handling — "Option D" in `SPRINT_20/PLAN.md`:
 *   Bunch-Kaufman symmetric swaps during elimination CAN introduce
 *   rows not present in `sym_L(A)` because sym_L is pattern-dependent
 *   and symmetric swaps permute the pattern.  The alternative
 *   approaches considered during Day 1 were:
 *     Option A: reuse `sym_L(A)` as-is, accept that BK 2×2 fill can
 *               overflow.  Incorrect on indefinite inputs — defeats
 *               the purpose of the shim.
 *     Option B: run a dedicated LDL^T symbolic pass that accounts
 *               for potential 2×2 pivot fill.  Requires bespoke
 *               infrastructure; high cost for a niche correctness
 *               case.
 *     Option C: use sym_L(A) + per-column 2× over-allocation to
 *               cover BK 2×2 fill.  Bounded but wasteful on SPD
 *               inputs where no 2×2 pivots occur.
 *     Option D: handle 2×2 pivot fill at the workflow level, not
 *               per-column.  SPD inputs (all 1×1 pivots, no swaps)
 *               use sym_L(A) directly; indefinite inputs run a
 *               scalar pre-pass to resolve BK swaps, symmetrically
 *               permute A by the resulting perm, and run
 *               `sparse_analyze` on the pre-permuted matrix.  After
 *               pre-permutation BK cannot swap again during the
 *               batched factor, so sym_L on the pre-permuted matrix
 *               is complete without over-allocation.
 *   Option D selected: the transparent dispatch added in Sprint 20
 *   Days 4-6 wraps the pre-pass workflow behind
 *   `sparse_ldlt_factor_opts` so public-API callers never see it,
 *   while batched-test helpers (e.g. `s19_supernodal_matches_scalar`)
 *   already use the same two-pass structure.
 *
 * Implementation (Day 2):
 *   1. Delegate the L layout + A-scatter to
 *      `chol_csc_from_sparse_with_analysis` — this sets
 *      `L->sym_L_preallocated = 1` for free and re-uses the
 *      bsearch-into-row-range scatter loop that the Cholesky path
 *      already validated on the Sprint 18 corpus.
 *   2. Wrap the returned `CholCsc *L` in an `LdltCsc` with D /
 *      D_offdiag / pivot_size / perm / row_adj allocated in the
 *      same zero-initialised shape as `ldlt_csc_from_sparse` (the
 *      calloc + identity-perm initialisation below mirrors lines
 *      484-524 of `ldlt_csc_from_sparse`).
 *   3. `pivot_size[i] = 1` default; `perm` copied from
 *      `analysis->perm` when present, else identity — matches the
 *      `ldlt_csc_from_sparse` convention when the caller supplied
 *      a fill-reducing perm.  Bunch-Kaufman pivoting composes
 *      further swaps into this array during eliminate.
 *
 * Day 3 wires the resulting `L->sym_L_preallocated == 1` state into
 * `ldlt_csc_eliminate_supernodal`'s writeback fast-path so the
 * indefinite KKT fixture drops from 1e-2..1e-6 residual to
 * round-off. */
sparse_err_t ldlt_csc_from_sparse_with_analysis(const SparseMatrix *mat,
                                                const sparse_analysis_t *analysis,
                                                LdltCsc **ldlt_out) {
    if (!ldlt_out)
        return SPARSE_ERR_NULL;
    *ldlt_out = NULL;
    if (!mat || !analysis)
        return SPARSE_ERR_NULL;
    if (analysis->type != SPARSE_FACTOR_CHOLESKY && analysis->type != SPARSE_FACTOR_LDLT)
        return SPARSE_ERR_BADARG;
    if (mat->rows != mat->cols)
        return SPARSE_ERR_SHAPE;
    if (mat->rows != analysis->n)
        return SPARSE_ERR_SHAPE;

    /* LDL^T requires a symmetric input; reject non-symmetric `mat`
     * with the same SPARSE_ERR_NOT_SPD code the scalar
     * `ldlt_csc_from_sparse` entry point uses (mirrors the
     * documented contract in the function docstring above). */
    if (!sparse_is_symmetric(mat, 1e-12))
        return SPARSE_ERR_NOT_SPD;

    /* Delegate L layout + sym_L pre-allocation + A-scatter to the
     * Cholesky converter.  Sets `L->sym_L_preallocated = 1` and
     * caches `L->factor_norm` from `analysis->analysis_norm`. */
    CholCsc *L = NULL;
    sparse_err_t err = chol_csc_from_sparse_with_analysis(mat, analysis, &L);
    if (err != SPARSE_OK)
        return err;

    /* Wrap L in an LdltCsc with D / D_offdiag / pivot_size / perm /
     * row_adj zero-initialised, matching `ldlt_csc_from_sparse`. */
    idx_t n = mat->rows;
    LdltCsc *m = calloc(1, sizeof(LdltCsc));
    if (!m) {
        chol_csc_free(L);
        return SPARSE_ERR_ALLOC;
    }
    m->n = n;
    m->L = L;
    m->factor_norm = L->factor_norm;

    size_t alloc_n = n > 0 ? (size_t)n : 1;
    m->D = calloc(alloc_n, sizeof(double));
    m->D_offdiag = calloc(alloc_n, sizeof(double));
    m->pivot_size = calloc(alloc_n, sizeof(idx_t));
    m->perm = calloc(alloc_n, sizeof(idx_t));
    m->row_adj = calloc(alloc_n, sizeof(idx_t *));
    m->row_adj_count = calloc(alloc_n, sizeof(idx_t));
    m->row_adj_cap = calloc(alloc_n, sizeof(idx_t));
    if (!m->D || !m->D_offdiag || !m->pivot_size || !m->perm || !m->row_adj || !m->row_adj_count ||
        !m->row_adj_cap) {
        ldlt_csc_free(m);
        return SPARSE_ERR_ALLOC;
    }

    /* Initial pivot_size is 1 everywhere; elimination overwrites. */
    for (idx_t i = 0; i < n; i++)
        m->pivot_size[i] = 1;

    /* Initial perm: caller-supplied fill-reducing permutation from
     * the analysis, else identity.  Bunch-Kaufman pivoting composes
     * further swaps into this array during eliminate. */
    if (analysis->perm) {
        for (idx_t i = 0; i < n; i++)
            m->perm[i] = analysis->perm[i];
    } else {
        for (idx_t i = 0; i < n; i++)
            m->perm[i] = i;
    }

    *ldlt_out = m;
    return SPARSE_OK;
}

/* ─── Convert LdltCsc → SparseMatrix (L lower triangle only) ─────────── */

sparse_err_t ldlt_csc_to_sparse(const LdltCsc *ldlt, const idx_t *perm_out,
                                SparseMatrix **mat_out) {
    if (!ldlt)
        return SPARSE_ERR_NULL;
    return chol_csc_to_sparse(ldlt->L, perm_out, mat_out);
}

/* ─── Writeback CSC factor → public sparse_ldlt_t ─────────────────────── */

/* Sprint 20 Day 5: transplant a factored LdltCsc into the
 * `sparse_ldlt_t` shape the public API documents.  Mirrors
 * `chol_csc_writeback_to_sparse` on the Cholesky side except that
 * the output is a separately-allocated result struct (not an
 * overwrite of the input matrix) — matching the LDL^T API's
 * separation between input `A` and output `ldlt->L`. */
sparse_err_t ldlt_csc_writeback_to_ldlt(const LdltCsc *F, double tol, sparse_ldlt_t *ldlt_out) {
    if (!F || !ldlt_out)
        return SPARSE_ERR_NULL;
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_NULL;
    if (F->n < 0)
        return SPARSE_ERR_BADARG;

    idx_t n = F->n;

    /* Build the L SparseMatrix column-by-column, mirroring
     * `chol_csc_writeback_to_sparse`'s filter: skip exact zeros
     * (common when the CSC was pre-populated via
     * `ldlt_csc_from_sparse_with_analysis` and some sym_L fill
     * positions never received a non-zero value) and drop below-
     * diagonal entries below `SPARSE_DROP_TOL * |L[j, j]|` so the
     * transplanted `SparseMatrix` sparsity matches what the
     * linked-list kernel publishes.  The diagonal (row_idx[col_ptr[j]]
     * == j by CSC invariant) is inserted whenever its stored value is
     * non-zero, which covers every factor produced by this backend
     * because LDL^T stores a unit-diagonal L (`L[j, j] == 1.0`; see
     * the `unit diagonal` references in `sparse_ldlt_csc.c:86` and
     * the elimination kernels).  The `v == 0.0` filter below
     * therefore never drops the diagonal in practice, and the `i != j`
     * guard on the below-diagonal threshold keeps the unit diagonal
     * from being accidentally filtered by the drop_tol test. */
    SparseMatrix *L_out = NULL;
    if (n > 0) {
        L_out = sparse_create(n, n);
        if (!L_out)
            return SPARSE_ERR_ALLOC;
        const CholCsc *L = F->L;
        for (idx_t j = 0; j < n; j++) {
            idx_t cstart = L->col_ptr[j];
            idx_t cend = L->col_ptr[j + 1];
            if (cstart == cend)
                continue;
            double abs_l_jj = (L->row_idx[cstart] == j) ? fabs(L->values[cstart]) : 0.0;
            double threshold = SPARSE_DROP_TOL * abs_l_jj;
            for (idx_t p = cstart; p < cend; p++) {
                idx_t i = L->row_idx[p];
                double v = L->values[p];
                if (v == 0.0)
                    continue;
                if (i != j && fabs(v) < threshold)
                    continue;
                sparse_err_t ierr = sparse_insert(L_out, i, j, v);
                if (ierr != SPARSE_OK) {
                    sparse_free(L_out);
                    return ierr;
                }
            }
        }
    }

    /* Allocate and copy the auxiliary arrays.  Use alloc_n = max(1, n)
     * so n == 0 is still a valid writeback producing non-NULL
     * pointers (matches ldlt_csc_from_sparse's convention). */
    size_t alloc_n = n > 0 ? (size_t)n : 1;
    double *D = calloc(alloc_n, sizeof(double));
    double *D_off = calloc(alloc_n, sizeof(double));
    int *ps = calloc(alloc_n, sizeof(int));
    idx_t *perm = calloc(alloc_n, sizeof(idx_t));
    if (!D || !D_off || !ps || !perm) {
        free(D);
        free(D_off);
        free(ps);
        free(perm);
        sparse_free(L_out);
        return SPARSE_ERR_ALLOC;
    }
    for (idx_t i = 0; i < n; i++) {
        D[i] = F->D[i];
        D_off[i] = F->D_offdiag[i];
        /* pivot_size values are always 1 or 2 (validated during
         * factor), so narrowing idx_t → int is lossless. */
        ps[i] = (int)F->pivot_size[i];
        perm[i] = F->perm[i];
    }

    ldlt_out->L = L_out;
    ldlt_out->D = D;
    ldlt_out->D_offdiag = D_off;
    ldlt_out->pivot_size = ps;
    ldlt_out->perm = perm;
    ldlt_out->n = n;
    ldlt_out->factor_norm = F->factor_norm;
    ldlt_out->tol = tol;
    return SPARSE_OK;
}

/* ─── Invariant checker ─────────────────────────────────────────────── */

sparse_err_t ldlt_csc_validate(const LdltCsc *ldlt) {
    if (!ldlt)
        return SPARSE_ERR_NULL;
    sparse_err_t err = chol_csc_validate(ldlt->L);
    if (err != SPARSE_OK)
        return err;
    if (ldlt->n < 0)
        return SPARSE_ERR_BADARG;
    if (ldlt->n > 0) {
        if (!ldlt->D || !ldlt->D_offdiag || !ldlt->pivot_size || !ldlt->perm)
            return SPARSE_ERR_BADARG;
    }

    /* pivot_size must be 1 or 2, and 2x2 pivots must cover consecutive
     * indices (pivot_size[i] = pivot_size[i+1] = 2). */
    for (idx_t i = 0; i < ldlt->n; i++) {
        idx_t s = ldlt->pivot_size[i];
        if (s != 1 && s != 2)
            return SPARSE_ERR_BADARG;
        if (s == 2) {
            if (i + 1 >= ldlt->n)
                return SPARSE_ERR_BADARG;
            if (ldlt->pivot_size[i + 1] != 2)
                return SPARSE_ERR_BADARG;
            /* Skip the second index of the 2x2 pair — it's been checked. */
            i++;
        }
    }

    /* perm must be a permutation of [0, n): every index appears exactly
     * once.  Use a small bit vector to detect duplicates / out-of-range. */
    if (ldlt->n == 0)
        return SPARSE_OK;
    char *seen = calloc((size_t)ldlt->n, sizeof(char));
    if (!seen)
        return SPARSE_ERR_ALLOC;
    for (idx_t i = 0; i < ldlt->n; i++) {
        idx_t p = ldlt->perm[i];
        if (p < 0 || p >= ldlt->n || seen[p]) {
            free(seen);
            return SPARSE_ERR_BADARG;
        }
        seen[p] = 1;
    }
    free(seen);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 8: Bunch-Kaufman elimination via the linked-list kernel
 * ═══════════════════════════════════════════════════════════════════════ */

/* Expand the lower-triangle CSC into a full symmetric SparseMatrix —
 * every off-diagonal entry is mirrored across the diagonal so that
 * sparse_ldlt_factor's symmetry check passes. */
static sparse_err_t csc_to_full_symmetric_matrix(const CholCsc *csc, SparseMatrix **out) {
    if (!csc || !out)
        return SPARSE_ERR_NULL;
    *out = NULL;
    idx_t n = csc->n;
    if (n <= 0)
        return SPARSE_ERR_BADARG;

    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return SPARSE_ERR_ALLOC;

    for (idx_t j = 0; j < n; j++) {
        for (idx_t p = csc->col_ptr[j]; p < csc->col_ptr[j + 1]; p++) {
            idx_t i = csc->row_idx[p];
            double v = csc->values[p];
            sparse_err_t ierr = sparse_insert(A, i, j, v);
            if (ierr != SPARSE_OK) {
                sparse_free(A);
                return ierr;
            }
            if (i != j) {
                ierr = sparse_insert(A, j, i, v);
                if (ierr != SPARSE_OK) {
                    sparse_free(A);
                    return ierr;
                }
            }
        }
    }

    *out = A;
    return SPARSE_OK;
}

sparse_err_t ldlt_csc_eliminate_wrapper(LdltCsc *F) {
    if (!F)
        return SPARSE_ERR_NULL;
    idx_t n = F->n;
    if (n <= 0)
        return SPARSE_OK;

    /* Validate only the invariants elimination actually needs.  The
     * full `ldlt_csc_validate(F)` call is too strict here because it
     * delegates to `chol_csc_validate(F->L)`, which rejects non-empty
     * CSC columns that don't start with an explicit diagonal entry —
     * but the linked-list `sparse_ldlt_factor` legitimately accepts
     * A's with a structurally-missing diagonal (treats them as zero
     * and either forms a 2x2 BK pivot or returns SPARSE_ERR_SINGULAR
     * later).  So we keep the safety checks that prevent crashes on
     * partially-initialised inputs but drop the diagonal-first
     * requirement and the sorted/distinct-row-index requirement that
     * full validate imposes. */
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_NULL;
    if (F->L->n != n || !F->L->col_ptr)
        return SPARSE_ERR_BADARG;
    if (F->L->col_ptr[0] != 0)
        return SPARSE_ERR_BADARG;
    idx_t l_nnz = F->L->col_ptr[n];
    if (l_nnz < 0)
        return SPARSE_ERR_BADARG;
    /* Validate every col_ptr entry against [0, l_nnz] before any reads of
     * row_idx/values.  Checking monotonicity alone is not enough — a
     * corrupted col_ptr[j] could be negative or exceed l_nnz while still
     * satisfying col_ptr[j] <= col_ptr[j+1], which would let later loops
     * read past the row_idx/values buffers. */
    for (idx_t j = 0; j < n; j++) {
        idx_t col_start = F->L->col_ptr[j];
        idx_t col_end = F->L->col_ptr[j + 1];
        if (col_start < 0 || col_end < 0 || col_start > col_end || col_start > l_nnz ||
            col_end > l_nnz)
            return SPARSE_ERR_BADARG;
    }
    if (l_nnz > 0 && (!F->L->row_idx || !F->L->values))
        return SPARSE_ERR_NULL;
    for (idx_t p = 0; p < l_nnz; p++) {
        if (F->L->row_idx[p] < 0 || F->L->row_idx[p] >= n)
            return SPARSE_ERR_BADARG;
    }
    /* Confirm perm is a real permutation so we can memcpy into perm_in
     * without risk of a stray index later in the factor path. */
    if ((size_t)n > SIZE_MAX / sizeof(unsigned char))
        return SPARSE_ERR_ALLOC;
    unsigned char *seen = calloc((size_t)n, sizeof(unsigned char));
    if (!seen)
        return SPARSE_ERR_ALLOC;
    for (idx_t j = 0; j < n; j++) {
        idx_t pj = F->perm[j];
        if (pj < 0 || pj >= n || seen[pj]) {
            free(seen);
            return SPARSE_ERR_BADARG;
        }
        seen[pj] = 1;
    }
    free(seen);

    /* Save the pre-elimination perm (fill-reducing) so we can compose it
     * with the Bunch-Kaufman perm chosen during factorization.  Guard
     * `n * sizeof(idx_t)` against size_t overflow on 32-bit platforms
     * (or absurdly large n) before computing the byte count. */
    if ((size_t)n > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;
    size_t perm_bytes = (size_t)n * sizeof(idx_t);
    idx_t *perm_in = malloc(perm_bytes);
    if (!perm_in)
        return SPARSE_ERR_ALLOC;
    memcpy(perm_in, F->perm, perm_bytes);

    /* Expand F->L's stored lower triangle to a full symmetric matrix so
     * the linked-list factor's symmetry check passes. */
    SparseMatrix *A_work = NULL;
    sparse_err_t err = csc_to_full_symmetric_matrix(F->L, &A_work);
    if (err != SPARSE_OK) {
        free(perm_in);
        return err;
    }

    /* Run the linked-list LDL^T factorization on the expanded matrix. */
    sparse_ldlt_t ll = {0};
    err = sparse_ldlt_factor(A_work, &ll);
    sparse_free(A_work);
    if (err != SPARSE_OK) {
        free(perm_in);
        return err;
    }

    /* Copy D / D_offdiag / pivot_size verbatim.  pivot_size widens from
     * the linked-list's int to our idx_t — values are 1 or 2 so the
     * conversion is lossless. */
    for (idx_t k = 0; k < n; k++) {
        F->D[k] = ll.D[k];
        F->D_offdiag[k] = ll.D_offdiag[k];
        F->pivot_size[k] = (idx_t)ll.pivot_size[k];
    }

    /* Compose the permutation: the factorization-order index k in the
     * CSC corresponds to index perm_in[ll.perm[k]] in the user's
     * original coordinate system. */
    if (ll.perm) {
        for (idx_t k = 0; k < n; k++)
            F->perm[k] = perm_in[ll.perm[k]];
    } else {
        for (idx_t k = 0; k < n; k++)
            F->perm[k] = perm_in[k];
    }

    /* Mirror ll.factor_norm into F->factor_norm — same quantity
     * (||A||_inf), just copied for consistency with `sparse_ldlt_t`. */
    F->factor_norm = ll.factor_norm;

    /* `sparse_ldlt_factor` initialises ll.L with a full identity
     * diagonal (`sparse_insert(L, i, i, 1.0)` for every i) before the
     * Bunch-Kaufman sweep, so the CSC conversion below can rely on
     * every column already containing its unit diagonal — no extra
     * injection loop is needed here. */

    /* Replace F->L with a CSC built from ll.L.  The linked-list factor
     * is already complete — no further fill-in will be introduced — so
     * allocate the CSC at exact capacity (fill_factor = 1.0) to avoid
     * a spurious 2x over-allocation on large factors. */
    CholCsc *new_L = NULL;
    err = chol_csc_from_sparse(ll.L, NULL, 1.0, &new_L);
    if (err != SPARSE_OK) {
        sparse_ldlt_free(&ll);
        free(perm_in);
        return err;
    }
    chol_csc_free(F->L);
    F->L = new_L;

    sparse_ldlt_free(&ll);
    free(perm_in);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18 Day 2: In-place symmetric swap primitive
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Algorithm (see header comment for the why):
 *
 *   Normalise i < j.
 *
 *   Phase A (cols [0, i)):
 *     For each column c, look up rows i and j in c's sorted row_idx
 *     slice.  Four cases:
 *       - Both present: swap values in place.
 *       - Only i: rename row_idx[pos_i] = j and bubble forward.
 *       - Only j: rename row_idx[pos_j] = i and bubble backward.
 *       - Neither: no-op.
 *     Column sizes and col_ptr are unchanged.
 *
 *   Phase B (cols [i, j]):
 *     Gather every entry in the block into (row, col, value) triples,
 *     apply σ to both coordinates, reflect to lower triangle if
 *     needed, bucket by new column and insertion-sort each bucket by
 *     row.  Write the result back into the same CSC slots (total block
 *     nnz is preserved so col_ptr[j+1] stays put — only col_ptr[i+1..j]
 *     shift within the block).
 *
 *   Phase C: swap F->D[i] ↔ F->D[j], F->D_offdiag[i] ↔ F->D_offdiag[j],
 *   F->pivot_size[i] ↔ F->pivot_size[j], F->perm[i] ↔ F->perm[j].
 *
 * Cols (j, n) are untouched: row indices in column c > j are all >= c,
 * so they never equal i or j. */

sparse_err_t ldlt_csc_symmetric_swap(LdltCsc *F, idx_t i, idx_t j) {
    if (!F)
        return SPARSE_ERR_NULL;
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_NULL;
    idx_t n = F->n;
    if (i < 0 || i >= n || j < 0 || j >= n)
        return SPARSE_ERR_BADARG;
    if (i == j)
        return SPARSE_OK;
    if (i > j) {
        idx_t tmp = i;
        i = j;
        j = tmp;
    }

    CholCsc *L = F->L;
    idx_t *col_ptr = L->col_ptr;
    idx_t *row_idx = L->row_idx;
    double *values = L->values;

    /* ── Phase A: cols [0, i) — swap row i ↔ row j per column ──────── */
    for (idx_t c = 0; c < i; c++) {
        idx_t start = col_ptr[c];
        idx_t end = col_ptr[c + 1];
        idx_t pos_i = end;
        idx_t pos_j = end;
        /* Linear scan is fine: columns are typically small (O(nnz/n))
         * and the scan stops as soon as row_idx[p] > j since row_idx
         * is sorted ascending. */
        for (idx_t p = start; p < end; p++) {
            idx_t r = row_idx[p];
            if (r == i) {
                pos_i = p;
            } else if (r == j) {
                pos_j = p;
                break; /* j is the larger target; beyond this, no matches. */
            } else if (r > j) {
                break;
            }
        }
        if (pos_i < end && pos_j < end) {
            /* Both present: swap values; row_idx slots keep their order. */
            double tmp = values[pos_i];
            values[pos_i] = values[pos_j];
            values[pos_j] = tmp;
        } else if (pos_i < end) {
            /* Only i present: rename to j and bubble forward to keep
             * row_idx sorted ascending. */
            double v = values[pos_i];
            idx_t p = pos_i;
            while (p + 1 < end && row_idx[p + 1] < j) {
                row_idx[p] = row_idx[p + 1];
                values[p] = values[p + 1];
                p++;
            }
            row_idx[p] = j;
            values[p] = v;
        } else if (pos_j < end) {
            /* Only j present: rename to i and bubble backward. */
            double v = values[pos_j];
            idx_t p = pos_j;
            while (p > start && row_idx[p - 1] > i) {
                row_idx[p] = row_idx[p - 1];
                values[p] = values[p - 1];
                p--;
            }
            row_idx[p] = i;
            values[p] = v;
        }
    }

    /* ── Phase B: cols [i, j] — gather-permute-scatter ─────────────── */
    idx_t block_start = col_ptr[i];
    idx_t block_end = col_ptr[j + 1];
    idx_t block_nnz = block_end - block_start;

    if (block_nnz > 0) {
        /* Temporary buffers: (row, col, value) triples for the gathered
         * block.  Total nnz is preserved through the permutation, so the
         * same block slot holds the rebuilt content without shifting
         * cols (j, n-1]. */
        if ((size_t)block_nnz > SIZE_MAX / sizeof(idx_t))
            return SPARSE_ERR_ALLOC;
        if ((size_t)block_nnz > SIZE_MAX / sizeof(double))
            return SPARSE_ERR_ALLOC;
        idx_t block_width = j - i + 1;
        /* Guard `block_width * sizeof(idx_t)` before the calloc below
         * so a 32-bit `size_t` platform (or a pathologically large n)
         * can't wrap and under-allocate `new_col_count`. */
        if ((size_t)block_width > SIZE_MAX / sizeof(idx_t))
            return SPARSE_ERR_ALLOC;

        idx_t *tmp_rows = malloc((size_t)block_nnz * sizeof(idx_t));
        idx_t *tmp_cols = malloc((size_t)block_nnz * sizeof(idx_t));
        double *tmp_vals = malloc((size_t)block_nnz * sizeof(double));
        idx_t *new_col_count = calloc((size_t)block_width, sizeof(idx_t));
        if (!tmp_rows || !tmp_cols || !tmp_vals || !new_col_count) {
            free(tmp_rows);
            free(tmp_cols);
            free(tmp_vals);
            free(new_col_count);
            return SPARSE_ERR_ALLOC;
        }

        /* Gather: walk cols [i, j], apply σ to (row, col), reflect to
         * lower triangle. */
        idx_t k = 0;
        for (idx_t c = i; c <= j; c++) {
            idx_t cstart = col_ptr[c];
            idx_t cend = col_ptr[c + 1];
            for (idx_t p = cstart; p < cend; p++) {
                idx_t r = row_idx[p];
                double v = values[p];
                idx_t rn = (r == i) ? j : ((r == j) ? i : r);
                idx_t cn = (c == i) ? j : ((c == j) ? i : c);
                /* Lower-triangle reflection by symmetry of the underlying
                 * matrix: (rn, cn) and (cn, rn) hold the same value, so
                 * pick whichever is lower-triangular. */
                if (rn < cn) {
                    idx_t t = rn;
                    rn = cn;
                    cn = t;
                }
                tmp_rows[k] = rn;
                tmp_cols[k] = cn;
                tmp_vals[k] = v;
                /* Invariant: cn ∈ [i, j] after the σ/reflect above, so
                 * `cn - i` ∈ [0, block_width).  The static analyzer
                 * can't prove this through the ternary + conditional
                 * swap chain, so the bound is asserted here for
                 * documentation and the access is NOLINT-suppressed. */
                new_col_count[cn - i]++; // NOLINT(clang-analyzer-security.ArrayBound)
                k++;
            }
        }

        /* Compute the new per-column write cursors within the existing
         * block slot.  Block boundaries (col_ptr[i], col_ptr[j+1]) do
         * not move; only col_ptr[i+1..j] shift within the block. */
        idx_t cursor = block_start;
        idx_t *col_write = new_col_count; /* reuse: turn into write cursors */
        for (idx_t c = 0; c < block_width; c++) {
            idx_t count = new_col_count[c];
            col_ptr[i + c] = cursor;
            col_write[c] = cursor;
            cursor += count;
        }
        /* col_ptr[j + 1] already equals block_end; leave it. */

        /* Scatter: bucket each triple by new column.  `tmp_cols[t]` is
         * written for every t in [0, block_nnz) during the gather loop
         * above (k ends at exactly block_nnz), and `tmp_cols[t]` is
         * always in [i, j] by the σ/reflect invariant — so `c_local` is
         * always a valid index into col_write[0..block_width).  NOLINT
         * suppresses a false positive where the analyser can't see the
         * 1:1 correspondence between the gather and scatter loops. */
        for (idx_t t = 0; t < block_nnz; t++) {
            idx_t c_local =
                tmp_cols[t] - i; // NOLINT(clang-analyzer-core.UndefinedBinaryOperatorResult)
            idx_t pos = col_write[c_local]++;
            row_idx[pos] = tmp_rows[t];
            values[pos] = tmp_vals[t];
        }

        /* Sort each rebuilt column's slot ascending by row.  Insertion
         * sort matches `sort_column_entries` in sparse_chol_csc.c —
         * columns are typically small and nearly sorted after scatter. */
        for (idx_t c = i; c <= j; c++) {
            idx_t cstart = col_ptr[c];
            idx_t cend = (c + 1 <= j) ? col_ptr[c + 1] : block_end;
            for (idx_t p = cstart + 1; p < cend; p++) {
                idx_t key_row = row_idx[p];
                double key_val = values[p];
                idx_t q = p;
                while (q > cstart && row_idx[q - 1] > key_row) {
                    row_idx[q] = row_idx[q - 1];
                    values[q] = values[q - 1];
                    q--;
                }
                row_idx[q] = key_row;
                values[q] = key_val;
            }
        }

        free(tmp_rows);
        free(tmp_cols);
        free(tmp_vals);
        free(new_col_count); /* was aliased as col_write — same buffer. */
    }

    /* ── Phase C: swap auxiliary arrays at positions i and j ──────── */
    {
        double tmp = F->D[i];
        F->D[i] = F->D[j];
        F->D[j] = tmp;
    }
    {
        double tmp = F->D_offdiag[i];
        F->D_offdiag[i] = F->D_offdiag[j];
        F->D_offdiag[j] = tmp;
    }
    {
        idx_t tmp = F->pivot_size[i];
        F->pivot_size[i] = F->pivot_size[j];
        F->pivot_size[j] = tmp;
    }
    {
        idx_t tmp = F->perm[i];
        F->perm[i] = F->perm[j];
        F->perm[j] = tmp;
    }

    /* ── Phase D: swap row-adjacency slots (Sprint 19 Day 9) ────────
     *
     * The swap σ = (i, j) renames row i ↔ row j in every factored
     * column c ∈ [0, i).  `row_adj[i]` lists priors whose entries
     * landed at row i BEFORE the swap; those entries now live at row
     * j, and vice versa — so swap the two rows' entire adjacency
     * lists (pointer, count, cap) in lockstep.
     *
     * Rows other than i and j are unaffected: column c may have had
     * entries at other rows, but those rows' indices didn't change.
     * Hence we only need to swap the two slots, not rebuild the
     * whole index. */
    if (F->row_adj) {
        idx_t *tmp_ptr = F->row_adj[i];
        F->row_adj[i] = F->row_adj[j];
        F->row_adj[j] = tmp_ptr;
    }
    if (F->row_adj_count) {
        idx_t tmp_cnt = F->row_adj_count[i];
        F->row_adj_count[i] = F->row_adj_count[j];
        F->row_adj_count[j] = tmp_cnt;
    }
    if (F->row_adj_cap) {
        idx_t tmp_cap = F->row_adj_cap[i];
        F->row_adj_cap[i] = F->row_adj_cap[j];
        F->row_adj_cap[j] = tmp_cap;
    }

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 18: Native CSC Bunch-Kaufman — scaffolding
 * ═══════════════════════════════════════════════════════════════════════ */

/* ─── Kernel selection & runtime override ──────────────────────────── */

/* Process-scope override.  Default = 0 (LDLT_CSC_KERNEL_DEFAULT) means
 * "use the compile-time default from LDLT_CSC_USE_NATIVE".  Tests and
 * benchmarks set this via `ldlt_csc_set_kernel_override` to exercise a
 * specific path on the current binary. */
static LdltCscKernelOverride g_ldlt_csc_kernel_override = LDLT_CSC_KERNEL_DEFAULT;

void ldlt_csc_set_kernel_override(LdltCscKernelOverride mode) { g_ldlt_csc_kernel_override = mode; }

LdltCscKernelOverride ldlt_csc_get_kernel_override(void) { return g_ldlt_csc_kernel_override; }

sparse_err_t ldlt_csc_eliminate(LdltCsc *F) {
    /* Resolve any DEFAULT override to the compile-time-selected kernel.
     * Writing it as an if + early return (rather than a switch with a
     * separate DEFAULT case) avoids a bugprone-branch-clone lint when
     * LDLT_CSC_USE_NATIVE == 1 makes the DEFAULT and NATIVE bodies
     * identical. */
    LdltCscKernelOverride mode = g_ldlt_csc_kernel_override;
    if (mode == LDLT_CSC_KERNEL_DEFAULT) {
#if LDLT_CSC_USE_NATIVE
        mode = LDLT_CSC_KERNEL_NATIVE;
#else
        mode = LDLT_CSC_KERNEL_WRAPPER;
#endif
    }
    if (mode == LDLT_CSC_KERNEL_NATIVE)
        return ldlt_csc_eliminate_native(F);
    return ldlt_csc_eliminate_wrapper(F);
}

/* ─── Native-kernel workspace lifecycle ────────────────────────────── */

void ldlt_csc_workspace_free(LdltCscWorkspace *ws) {
    if (!ws)
        return;
    free(ws->dense_col);
    free(ws->dense_pattern);
    free(ws->dense_marker);
    free(ws->dense_col_r);
    free(ws->dense_pattern_r);
    free(ws->dense_marker_r);
    free(ws);
}

sparse_err_t ldlt_csc_workspace_alloc(idx_t n, LdltCscWorkspace **out) {
    if (!out)
        return SPARSE_ERR_NULL;
    *out = NULL;
    if (n < 0)
        return SPARSE_ERR_BADARG;

    /* Overflow guards: six length-n arrays. */
    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    if ((size_t)n > SIZE_MAX / sizeof(idx_t))
        return SPARSE_ERR_ALLOC;

    LdltCscWorkspace *ws = calloc(1, sizeof(LdltCscWorkspace));
    if (!ws)
        return SPARSE_ERR_ALLOC;
    ws->n = n;

    /* Allocate at least 1 slot so later subscripts never trip over a
     * null pointer even for n == 0 — matches `chol_csc_workspace_alloc`. */
    size_t alloc_n = n > 0 ? (size_t)n : 1;
    ws->dense_col = calloc(alloc_n, sizeof(double));
    ws->dense_pattern = calloc(alloc_n, sizeof(idx_t));
    ws->dense_marker = calloc(alloc_n, sizeof(int8_t));
    ws->dense_col_r = calloc(alloc_n, sizeof(double));
    ws->dense_pattern_r = calloc(alloc_n, sizeof(idx_t));
    ws->dense_marker_r = calloc(alloc_n, sizeof(int8_t));
    if (!ws->dense_col || !ws->dense_pattern || !ws->dense_marker || !ws->dense_col_r ||
        !ws->dense_pattern_r || !ws->dense_marker_r) {
        ldlt_csc_workspace_free(ws);
        return SPARSE_ERR_ALLOC;
    }

    *out = ws;
    return SPARSE_OK;
}

/* ─── Ported BK pivot scanner (phase 1) ────────────────────────────── */

/* Bunch-Kaufman alpha = (1 + sqrt(17)) / 8 ≈ 0.6404.  Computed once at
 * first call rather than baking in a double literal — matches
 * `sparse_ldlt.c`'s runtime computation, so any future sqrt-precision
 * tweak applies to both kernels. */
static double ldlt_csc_bk_alpha(void) { return (1.0 + sqrt(17.0)) / 8.0; }

/* Scan the dense column accumulator for the largest off-diagonal
 * magnitude at rows i > k.  Returns the magnitude (0.0 if none) and
 * writes the row index of that off-diagonal into *r_out (k when the
 * column has no below-diagonal fill, matching `sparse_ldlt.c`'s
 * sentinel convention).
 *
 * Ported from the inline block at src/sparse_ldlt.c:498-507.  Day 3
 * wires this into `ldlt_csc_eliminate_native`'s column loop.
 */
static double ldlt_csc_bk_scan_offdiag(const double *dense_col, const idx_t *pattern,
                                       idx_t pattern_count, idx_t k, idx_t *r_out) {
    double max_offdiag = 0.0;
    idx_t r = k;
    for (idx_t t = 0; t < pattern_count; t++) {
        idx_t i = pattern[t];
        if (i > k) {
            double mag = fabs(dense_col[i]);
            if (mag > max_offdiag) {
                max_offdiag = mag;
                r = i;
            }
        }
    }
    *r_out = r;
    return max_offdiag;
}

/* ─── Scatter + cmod helpers (Sprint 18 Day 3) ──────────────────── */

/* Scatter the symmetric column `col` at step k into a dense accumulator.
 *
 * At step `step_k`, F->L stores factored L (with unit diagonal) in
 * columns [0, step_k) and A's lower triangle in columns [step_k, n).
 * Scattering "column col" in the symmetric sense means picking up:
 *   - lower-tri stored entries of column col with row >= step_k
 *     (iterate col_ptr[col]..col_ptr[col+1]); and
 *   - reflected upper-tri entries at rows in [step_k, col) — by
 *     symmetry A[col, c] == A[c, col], so for each column c in
 *     [step_k, col) we binary-search for row `col` in c's slice and
 *     place the hit into dense[c].
 *
 * The upper-tri loop is empty when `col == step_k` (primary pivot
 * column k) and non-empty when `col > step_k` (BK phase-2 partner r).
 */
static void ldlt_csc_scatter_symmetric(const CholCsc *L, idx_t col, idx_t step_k, double *dense,
                                       idx_t *pattern, int8_t *marker, idx_t *pattern_count) {
    idx_t cstart = L->col_ptr[col];
    idx_t cend = L->col_ptr[col + 1];
    for (idx_t p = cstart; p < cend; p++) {
        idx_t i = L->row_idx[p];
        if (i < step_k)
            continue;
        dense[i] = L->values[p];
        if (!marker[i]) {
            marker[i] = 1;
            pattern[(*pattern_count)++] = i;
        }
    }
    for (idx_t c = step_k; c < col; c++) {
        idx_t start = L->col_ptr[c];
        idx_t end = L->col_ptr[c + 1];
        idx_t lo = start;
        idx_t hi = end;
        while (lo < hi) {
            idx_t mid = lo + (hi - lo) / 2;
            if (L->row_idx[mid] < col) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo < end && L->row_idx[lo] == col) {
            dense[c] = L->values[lo];
            if (!marker[c]) {
                marker[c] = 1;
                pattern[(*pattern_count)++] = c;
            }
        }
    }
}

/* Binary-search for `target` in the sorted row_idx slice
 * [cstart, cend) of a CSC column.  Returns the L value at that row, or
 * 0.0 if not present. */
static double ldlt_csc_lookup_Lrc(const CholCsc *L, idx_t cstart, idx_t cend, idx_t target) {
    idx_t lo = cstart;
    idx_t hi = cend;
    while (lo < hi) {
        idx_t mid = lo + (hi - lo) / 2;
        if (L->row_idx[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < cend && L->row_idx[lo] == target) {
        return L->values[lo];
    }
    return 0.0;
}

/* Apply cmod contributions from prior factored columns kp in [0, step_k)
 * to the dense accumulator for column `col`.
 *
 * Two passes:
 *
 *   Phase A: for every prior column kp where `L[col, kp] != 0`,
 *   subtract `L[i, kp] * D[kp] * L[col, kp]` from dense[i] for every
 *   stored row i >= step_k in column kp.  D[kp] here is the per-
 *   column diagonal entry (the block-diagonal element for 2×2
 *   priors, NOT the 2×2 inverse).
 *
 *   Phase B: for every 2×2 block pivot pair (kp, kp+1) where
 *   `L[col, kp] != 0` or `L[col, kp+1] != 0`, add the off-diagonal
 *   cross-term `L[i, kp] * D_off[kp] * L[col, kp+1]
 *   + L[i, kp+1] * D_off[kp] * L[col, kp]` to the subtraction.
 *
 * Phase A + Phase B together reproduce the reference acc_schur_col
 * semantics in sparse_ldlt.c exactly.  New rows touched by either pass
 * get appended to the pattern via the marker check.
 *
 * Sprint 19 Day 9 switch: both phases iterate `F->row_adj[col]`
 * instead of `[0, step_k)`.  `row_adj[col]` is populated by
 * `ldlt_csc_eliminate_native` after every column writeback with the
 * list of prior columns that have a stored non-zero at row `col`.
 * Iterating only those columns matches the linked-list reference's
 * `acc_schur_col` sparse-row scaling (the linked-list's cross-linked
 * row iterator visits exactly the columns that contribute); the
 * pre-Sprint-19 kernel iterated every `kp in [0, step_k)` with a
 * binary search per `kp`, giving O(step_k · log nnz) per cmod even
 * when `col`'s row was very sparse. */
static void ldlt_csc_cmod_unified(const LdltCsc *F, idx_t col, idx_t step_k, double *dense,
                                  idx_t *pattern, int8_t *marker, idx_t *pattern_count) {
    const CholCsc *L = F->L;
    idx_t row_adj_count = (col >= 0 && col < F->n) ? F->row_adj_count[col] : 0;
    idx_t *row_adj = (col >= 0 && col < F->n) ? F->row_adj[col] : NULL;

    /* Phase A: per-column diagonal contribution — walk `row_adj[col]`. */
    for (idx_t idx = 0; idx < row_adj_count; idx++) {
        idx_t kp = row_adj[idx];
        /* Defensive: `row_adj` may contain entries ≥ step_k if the
         * caller repopulated across elimination restarts (shouldn't
         * happen today but the guard is free).  Skip them so they
         * don't double-count into a future step. */
        if (kp >= step_k)
            continue;

        idx_t cstart = L->col_ptr[kp];
        idx_t cend = L->col_ptr[kp + 1];
        double L_col_kp = ldlt_csc_lookup_Lrc(L, cstart, cend, col);
        if (L_col_kp == 0.0)
            continue; /* should not happen if row_adj is accurate */
        double factor = F->D[kp] * L_col_kp;
        for (idx_t p = cstart; p < cend; p++) {
            idx_t i = L->row_idx[p];
            if (i < step_k)
                continue;
            if (!marker[i]) {
                marker[i] = 1;
                /* pattern is allocated with n slots and marker-gated
                 * uniqueness bounds *pattern_count by n throughout the
                 * elimination; analyzer can't follow this across the
                 * helper boundary so the access is NOLINT-suppressed. */
                pattern[(*pattern_count)++] = i; // NOLINT(clang-analyzer-security.ArrayBound)
            }
            dense[i] -= L->values[p] * factor;
        }
    }

    /* Phase B: cross-term correction for 2×2 block priors — walk the
     * same `row_adj[col]` list, detecting 2×2 pair membership via
     * `pivot_size`.  Each entry `kp` in the list triggers at most one
     * inner loop:
     *
     *   - If `kp` is the FIRST of a 2×2 pair `(kp, kp+1)`:
     *     `ct2 = D_off[kp] * L[col, kp]` (non-zero since `kp` is in
     *     `row_adj`), and the inner loop runs on column `kp+1`
     *     (updating `dense[i] -= L[i, kp+1] * ct2`).
     *
     *   - If `kp` is the SECOND of a 2×2 pair `(kp-1, kp)`:
     *     `ct1 = D_off[kp-1] * L[col, kp]`, inner loop on column
     *     `kp-1` (updating `dense[i] -= L[i, kp-1] * ct1`).
     *
     * The two inner-loop cases are the row-adj-driven counterparts of
     * the old Phase B's `ct1 != 0` / `ct2 != 0` branches.  If `col`
     * has no stored entry in either member of a 2×2 pair, the pair
     * contributes zero and is correctly skipped (neither member
     * appears in `row_adj[col]`). */
    for (idx_t idx = 0; idx < row_adj_count; idx++) {
        idx_t kp = row_adj[idx];
        if (kp >= step_k)
            continue;
        if (F->pivot_size[kp] != 2)
            continue;

        idx_t pair_other;
        int kp_is_first;
        if (kp + 1 < step_k && F->pivot_size[kp + 1] == 2 &&
            (kp == 0 || F->pivot_size[kp - 1] != 2)) {
            /* kp is the first of a 2×2 at (kp, kp+1). */
            pair_other = kp + 1;
            kp_is_first = 1;
        } else if (kp >= 1 && F->pivot_size[kp - 1] == 2) {
            /* kp is the second of a 2×2 at (kp-1, kp). */
            pair_other = kp - 1;
            kp_is_first = 0;
        } else {
            continue; /* defensive: malformed pivot_size[] */
        }

        idx_t d_off_idx = kp_is_first ? kp : pair_other;
        double d_off = F->D_offdiag[d_off_idx];
        if (d_off == 0.0)
            continue;

        /* L[col, kp] is guaranteed non-zero (kp is in row_adj). */
        idx_t cstart_kp = L->col_ptr[kp];
        idx_t cend_kp = L->col_ptr[kp + 1];
        double L_col_kp = ldlt_csc_lookup_Lrc(L, cstart_kp, cend_kp, col);
        double ct = d_off * L_col_kp; /* ct2 when kp is first, ct1 when kp is second */
        if (ct == 0.0)
            continue;

        /* Inner loop on the OTHER column of the pair. */
        idx_t cstart_o = L->col_ptr[pair_other];
        idx_t cend_o = L->col_ptr[pair_other + 1];
        for (idx_t p = cstart_o; p < cend_o; p++) {
            idx_t i = L->row_idx[p];
            if (i < step_k)
                continue;
            if (!marker[i]) {
                marker[i] = 1;
                pattern[(*pattern_count)++] = i; // NOLINT(clang-analyzer-security.ArrayBound)
            }
            dense[i] -= L->values[p] * ct;
        }
    }
}

/* Sprint 19 Day 9: populate `F->row_adj` for every prior column `col`
 * contributes to, by walking column `col`'s storage in `F->L` after
 * gather.  For each stored row `i > col`, append `col` to
 * `F->row_adj[i]`.  The diagonal entry (row == col) is skipped — a
 * column is not its own prior.
 *
 * Called once per column writeback in `ldlt_csc_eliminate_native`;
 * together with the row-adj-driven cmod this reproduces the linked-
 * list reference's sparse-row iteration without the O(step_k) scan. */
static sparse_err_t ldlt_csc_populate_row_adj(LdltCsc *F, idx_t col) {
    idx_t cstart = F->L->col_ptr[col];
    idx_t cend = F->L->col_ptr[col + 1];
    for (idx_t p = cstart; p < cend; p++) {
        idx_t i = F->L->row_idx[p];
        if (i > col) {
            sparse_err_t err = ldlt_csc_row_adj_append(F, i, col);
            if (err != SPARSE_OK)
                return err;
        }
    }
    return SPARSE_OK;
}

/* Clear the primary accumulator's touched entries.
 *
 * pattern_count is maintained by the column loop to never exceed the
 * allocation size (ws->n); the marker-gated increments in scatter and
 * cmod ensure uniqueness.  Analyzer can't track that through the helper
 * boundary, hence the NOLINT suppression on the access. */
static void ldlt_csc_clear_dense_col(LdltCscWorkspace *ws) {
    for (idx_t t = 0; t < ws->pattern_count; t++) {
        idx_t i = ws->dense_pattern[t]; // NOLINT(clang-analyzer-security.ArrayBound)
        ws->dense_col[i] = 0.0;
        ws->dense_marker[i] = 0;
    }
    ws->pattern_count = 0;
}

/* Clear the partner accumulator's touched entries (same invariants as
 * `ldlt_csc_clear_dense_col`). */
static void ldlt_csc_clear_dense_col_r(LdltCscWorkspace *ws) {
    for (idx_t t = 0; t < ws->pattern_count_r; t++) {
        idx_t i = ws->dense_pattern_r[t]; // NOLINT(clang-analyzer-security.ArrayBound)
        ws->dense_col_r[i] = 0.0;
        ws->dense_marker_r[i] = 0;
    }
    ws->pattern_count_r = 0;
}

/* ─── Native kernel (Sprint 18 Days 3-4: full 1×1 + 2×2) ─────────── */

/* Sprint 19 Day 13 refactor: the body of `ldlt_csc_eliminate_native`'s
 * column loop is extracted here so `ldlt_csc_eliminate_supernodal`
 * (Day 13) can interleave it with the batched supernodal path on
 * non-supernodal columns.
 *
 * One step factors a single 1×1 pivot or a 2×2 pivot pair starting at
 * column `*k_inout`, advancing it by 1 or 2 on success.  All temporary
 * accumulator state lives on `ws`; the caller (eliminate_native /
 * eliminate_supernodal) owns `ws`'s lifecycle.  On error the
 * accumulator may be left dirty — the caller's policy is to free `ws`
 * (and not reuse it) before returning, so clearing is unnecessary. */
static sparse_err_t ldlt_csc_eliminate_one_step(LdltCsc *F, LdltCscWorkspace *ws, idx_t *k_inout,
                                                double drop_tol, double sing_tol, double alpha_bk,
                                                double growth_bound) {
    idx_t n = F->n;
    idx_t k = *k_inout;
    sparse_err_t rc = SPARSE_OK;

    /* ── Scatter + cmod for column k ────────────────────────── */
    ldlt_csc_scatter_symmetric(F->L, k, k, ws->dense_col, ws->dense_pattern, ws->dense_marker,
                               &ws->pattern_count);
    ldlt_csc_cmod_unified(F, k, k, ws->dense_col, ws->dense_pattern, ws->dense_marker,
                          &ws->pattern_count);

    if (!ws->dense_marker[k]) {
        ws->dense_marker[k] = 1;
        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
        ws->dense_pattern[ws->pattern_count++] = k;
    }

    /* ── Phase 1 BK scan ────────────────────────────────────── */
    idx_t r = k;
    double max_offdiag =
        ldlt_csc_bk_scan_offdiag(ws->dense_col, ws->dense_pattern, ws->pattern_count, k, &r);
    double diag_k = ws->dense_col[k];

    int use_2x2 = 0;

    /* ── Phase 2 BK criteria (only when the diagonal is small) ── */
    if (max_offdiag > 0.0 && k + 1 < n && fabs(diag_k) < alpha_bk * max_offdiag) {
        ldlt_csc_scatter_symmetric(F->L, r, k, ws->dense_col_r, ws->dense_pattern_r,
                                   ws->dense_marker_r, &ws->pattern_count_r);
        ldlt_csc_cmod_unified(F, r, k, ws->dense_col_r, ws->dense_pattern_r, ws->dense_marker_r,
                              &ws->pattern_count_r);

        double sigma_r = 0.0;
        for (idx_t t = 0; t < ws->pattern_count_r; t++) {
            idx_t i = ws->dense_pattern_r[t];
            if (i >= k && i != r) {
                double m = fabs(ws->dense_col_r[i]);
                if (m > sigma_r)
                    sigma_r = m;
            }
        }

        if (fabs(diag_k) * sigma_r >= alpha_bk * max_offdiag * max_offdiag) {
            ldlt_csc_clear_dense_col_r(ws);
        } else if (fabs(ws->dense_col_r[r]) >= alpha_bk * sigma_r) {
            rc = ldlt_csc_symmetric_swap(F, k, r);
            if (rc != SPARSE_OK)
                return rc;

            ldlt_csc_clear_dense_col(ws);
            for (idx_t t = 0; t < ws->pattern_count_r; t++) {
                idx_t i = ws->dense_pattern_r[t];
                idx_t m = (i == k) ? r : (i == r ? k : i);
                ws->dense_col[m] = ws->dense_col_r[i];
                if (!ws->dense_marker[m]) {
                    ws->dense_marker[m] = 1;
                    ws->dense_pattern[ws->pattern_count++] = m;
                }
            }
            ldlt_csc_clear_dense_col_r(ws);
            diag_k = ws->dense_col[k];
            if (!ws->dense_marker[k]) {
                ws->dense_marker[k] = 1;
                ws->dense_pattern[ws->pattern_count++] = k;
            }
        } else {
            use_2x2 = 1;
        }
    }

    if (!use_2x2) {
        /* ── 1×1 apply ────────────────────────────────────── */
        if (fabs(diag_k) < sing_tol)
            return SPARSE_ERR_SINGULAR;
        F->D[k] = diag_k;
        F->D_offdiag[k] = 0.0;
        F->pivot_size[k] = 1;

        for (idx_t t = 0; t < ws->pattern_count; t++) {
            idx_t i = ws->dense_pattern[t];
            if (i > k) {
                double v = ws->dense_col[i] / diag_k;
                if (fabs(v) > growth_bound)
                    return SPARSE_ERR_SINGULAR;
                ws->dense_col[i] = v;
            }
        }

        ws->dense_col[k] = 1.0;

        CholCscWorkspace view;
        view.n = ws->n;
        view.dense_col = ws->dense_col;
        view.dense_pattern = ws->dense_pattern;
        view.dense_marker = ws->dense_marker;
        view.pattern_count = ws->pattern_count;
        rc = chol_csc_gather(F->L, k, &view, drop_tol);
        ws->pattern_count = view.pattern_count;
        if (rc != SPARSE_OK)
            return rc;

        rc = ldlt_csc_populate_row_adj(F, k);
        if (rc != SPARSE_OK)
            return rc;

        ldlt_csc_clear_dense_col(ws);
        *k_inout = k + 1;
        return SPARSE_OK;
    }

    /* ── 2×2 block pivot at (k, r) ─────────────────────────── */
    if (r != k + 1) {
        rc = ldlt_csc_symmetric_swap(F, r, k + 1);
        if (rc != SPARSE_OK)
            return rc;

        double tmp_d = ws->dense_col[r];
        ws->dense_col[r] = ws->dense_col[k + 1];
        ws->dense_col[k + 1] = tmp_d;
        int8_t tmp_m = ws->dense_marker[r];
        ws->dense_marker[r] = ws->dense_marker[k + 1];
        ws->dense_marker[k + 1] = tmp_m;
        for (idx_t t = 0; t < ws->pattern_count; t++) {
            if (ws->dense_pattern[t] == r) {
                ws->dense_pattern[t] = k + 1;
            } else if (ws->dense_pattern[t] == k + 1) {
                ws->dense_pattern[t] = r;
            }
        }

        tmp_d = ws->dense_col_r[r];
        ws->dense_col_r[r] = ws->dense_col_r[k + 1];
        ws->dense_col_r[k + 1] = tmp_d;
        tmp_m = ws->dense_marker_r[r];
        ws->dense_marker_r[r] = ws->dense_marker_r[k + 1];
        ws->dense_marker_r[k + 1] = tmp_m;
        for (idx_t t = 0; t < ws->pattern_count_r; t++) {
            if (ws->dense_pattern_r[t] == r) {
                ws->dense_pattern_r[t] = k + 1;
            } else if (ws->dense_pattern_r[t] == k + 1) {
                ws->dense_pattern_r[t] = r;
            }
        }
    }

    double d11 = ws->dense_col[k];
    double d21 = ws->dense_col[k + 1];
    double d22 = ws->dense_col_r[k + 1];
    double det = d11 * d22 - d21 * d21;
    double bscale = fabs(d11) + fabs(d22) + fabs(d21);
    double det_tol = (bscale > 0.0) ? drop_tol * bscale * bscale : sing_tol * sing_tol;
    if (fabs(det) < det_tol)
        return SPARSE_ERR_SINGULAR;
    double inv_det = 1.0 / det;
    double drop_2x2 = (bscale > 0.0) ? drop_tol * bscale : drop_tol;

    F->D[k] = d11;
    F->D[k + 1] = d22;
    F->D_offdiag[k] = d21;
    F->D_offdiag[k + 1] = 0.0;
    F->pivot_size[k] = 2;
    F->pivot_size[k + 1] = 2;

    for (idx_t t = 0; t < ws->pattern_count_r; t++) {
        idx_t i = ws->dense_pattern_r[t];
        if (!ws->dense_marker[i]) {
            ws->dense_marker[i] = 1;
            ws->dense_pattern[ws->pattern_count++] = i;
        }
    }

    for (idx_t t = 0; t < ws->pattern_count; t++) {
        idx_t i = ws->dense_pattern[t];
        if (i <= k + 1)
            continue;
        double s_ik = ws->dense_col[i];
        double s_ik1 = ws->dense_col_r[i];
        double l_ik = (s_ik * d22 - s_ik1 * d21) * inv_det;
        double l_ik1 = (-s_ik * d21 + s_ik1 * d11) * inv_det;
        if (fabs(l_ik) > growth_bound || fabs(l_ik1) > growth_bound)
            return SPARSE_ERR_SINGULAR;
        ws->dense_col[i] = l_ik;
        ws->dense_col_r[i] = l_ik1;
        if (!ws->dense_marker_r[i]) {
            ws->dense_marker_r[i] = 1;
            ws->dense_pattern_r[ws->pattern_count_r++] = i;
        }
    }

    ws->dense_col[k] = 1.0;
    ws->dense_col[k + 1] = 0.0;

    CholCscWorkspace view_k;
    view_k.n = ws->n;
    view_k.dense_col = ws->dense_col;
    view_k.dense_pattern = ws->dense_pattern;
    view_k.dense_marker = ws->dense_marker;
    view_k.pattern_count = ws->pattern_count;
    rc = chol_csc_gather(F->L, k, &view_k, drop_2x2);
    ws->pattern_count = view_k.pattern_count;
    if (rc != SPARSE_OK)
        return rc;

    ws->dense_col_r[k + 1] = 1.0;
    ws->dense_col_r[k] = 0.0;
    if (!ws->dense_marker[k + 1]) {
        ws->dense_marker[k + 1] = 1;
        ws->dense_pattern[ws->pattern_count++] = k + 1;
    }
    if (!ws->dense_marker_r[k + 1]) {
        ws->dense_marker_r[k + 1] = 1;
        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
        ws->dense_pattern_r[ws->pattern_count_r++] = k + 1;
    }
    if (!ws->dense_marker_r[k]) {
        ws->dense_marker_r[k] = 1;
        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
        ws->dense_pattern_r[ws->pattern_count_r++] = k;
    }

    CholCscWorkspace view_k1;
    view_k1.n = ws->n;
    view_k1.dense_col = ws->dense_col_r;
    view_k1.dense_pattern = ws->dense_pattern;
    view_k1.dense_marker = ws->dense_marker;
    view_k1.pattern_count = ws->pattern_count;
    rc = chol_csc_gather(F->L, k + 1, &view_k1, drop_2x2);
    ws->pattern_count = view_k1.pattern_count;
    if (rc != SPARSE_OK)
        return rc;

    rc = ldlt_csc_populate_row_adj(F, k);
    if (rc != SPARSE_OK)
        return rc;
    rc = ldlt_csc_populate_row_adj(F, k + 1);
    if (rc != SPARSE_OK)
        return rc;

    ldlt_csc_clear_dense_col(ws);
    ldlt_csc_clear_dense_col_r(ws);
    *k_inout = k + 2;
    return SPARSE_OK;
}

/* Sprint 18 Days 3-4 ship a complete Bunch-Kaufman kernel directly on
 * CSC storage: scatter + cmod per column (handling both 1×1 and 2×2
 * priors via `ldlt_csc_cmod_unified`), four-criteria pivot selection
 * with an in-place symmetric swap for criteria 3 and 4, 1×1 divide
 * or 2×2 block factor with element-growth tracking against
 * `growth_bound = 1 / (100 * DROP_TOL)` matching the linked-list
 * reference, and gather through `chol_csc_gather`.
 *
 * F->perm is updated in place via the symmetric-swap helper at each
 * BK swap, so no separate "compose with fill-reducing perm" step is
 * needed — by the end of the loop F->perm holds the composition the
 * wrapper produces via its post-factor unpack.
 */
sparse_err_t ldlt_csc_eliminate_native(LdltCsc *F) {
    if (!F)
        return SPARSE_ERR_NULL;
    idx_t n = F->n;
    if (n <= 0)
        return SPARSE_OK;

    /* Same structural input validation the wrapper performs. */
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_NULL;
    if (F->L->n != n || !F->L->col_ptr)
        return SPARSE_ERR_BADARG;
    if (F->L->col_ptr[0] != 0)
        return SPARSE_ERR_BADARG;
    idx_t l_nnz = F->L->col_ptr[n];
    if (l_nnz < 0)
        return SPARSE_ERR_BADARG;
    for (idx_t j = 0; j < n; j++) {
        idx_t col_start = F->L->col_ptr[j];
        idx_t col_end = F->L->col_ptr[j + 1];
        if (col_start < 0 || col_end < 0 || col_start > col_end || col_start > l_nnz ||
            col_end > l_nnz)
            return SPARSE_ERR_BADARG;
    }
    if (l_nnz > 0 && (!F->L->row_idx || !F->L->values))
        return SPARSE_ERR_NULL;
    for (idx_t p = 0; p < l_nnz; p++) {
        if (F->L->row_idx[p] < 0 || F->L->row_idx[p] >= n)
            return SPARSE_ERR_BADARG;
    }

    LdltCscWorkspace *ws = NULL;
    sparse_err_t err = ldlt_csc_workspace_alloc(n, &ws);
    if (err != SPARSE_OK)
        return err;

    /* Tolerances — match sparse_ldlt.c exactly so native / wrapper
     * decisions stay in lockstep on borderline matrices. */
    const double drop_tol = SPARSE_DROP_TOL;
    const double sing_tol = sparse_rel_tol(F->factor_norm, drop_tol);
    const double alpha_bk = ldlt_csc_bk_alpha();
    const double growth_bound = 1.0 / (100.0 * drop_tol);

    sparse_err_t rc = SPARSE_OK;

    idx_t k = 0;
    while (k < n) {
        rc = ldlt_csc_eliminate_one_step(F, ws, &k, drop_tol, sing_tol, alpha_bk, growth_bound);
        if (rc != SPARSE_OK)
            goto cleanup;
    }

cleanup:
    ldlt_csc_clear_dense_col(ws);
    ldlt_csc_clear_dense_col_r(ws);
    ldlt_csc_workspace_free(ws);
    return rc;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 9: LDL^T solve — forward / diagonal-block / backward sweeps
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t ldlt_csc_solve(const LdltCsc *F, const double *b, double *x) {
    if (!F || !b || !x)
        return SPARSE_ERR_NULL;
    idx_t n = F->n;
    if (n == 0)
        return SPARSE_OK;
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_BADARG;

    /* Workspace: y holds the permuted RHS → forward-solved vector;
     * z receives the diagonal-solved vector and is then overwritten
     * in place by the backward sweep. */
    if ((size_t)n > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    double *y = malloc((size_t)n * sizeof(double));
    double *z = malloc((size_t)n * sizeof(double));
    if (!y || !z) {
        free(y);
        free(z);
        return SPARSE_ERR_ALLOC;
    }

    /* Tolerance scaling matches sparse_ldlt_solve: 1x1 singularity
     * against ||A||_inf, 2x2 block singularity against the block's own
     * scale (|d11| + |d22| + |d21|)^2 to handle Schur complements
     * whose magnitude has drifted from the original matrix norm. */
    double solve_tol = SPARSE_DROP_TOL;
    double sing_tol = sparse_rel_tol(F->factor_norm, solve_tol);

    /* Phase 0: y[i] = b[perm[i]]  (apply P to b). */
    for (idx_t i = 0; i < n; i++)
        y[i] = b[F->perm[i]];

    /* Phase 1: Forward solve L*w = y.  L is unit lower triangular in
     * CSC: for each column j left-to-right, skip the unit diagonal
     * (first stored entry) and subtract L[i,j] * y[j] from every row
     * i > j present in column j's slot. */
    for (idx_t j = 0; j < n; j++) {
        idx_t start = F->L->col_ptr[j];
        idx_t end = F->L->col_ptr[j + 1];
        if (start == end || F->L->row_idx[start] != j) {
            /* Missing unit diagonal — Day 8's elimination guarantees it,
             * so this indicates a hand-corrupted CSC.  Fail safely. */
            free(y);
            free(z);
            return SPARSE_ERR_BADARG;
        }
        double yj = y[j];
        for (idx_t p = start + 1; p < end; p++) {
            idx_t i = F->L->row_idx[p];
            y[i] -= F->L->values[p] * yj;
        }
    }

    /* Phase 2: Diagonal solve D*z = w (w has overwritten y). */
    for (idx_t k = 0; k < n;) {
        if (F->pivot_size[k] == 1) {
            if (fabs(F->D[k]) < sing_tol) {
                free(y);
                free(z);
                return SPARSE_ERR_SINGULAR;
            }
            z[k] = y[k] / F->D[k];
            k++;
        } else {
            /* 2x2 block: [[d11, d21], [d21, d22]]; inv = 1/det * [[d22, -d21], [-d21, d11]].
             * `ldlt_csc_validate` guarantees pivot_size[k] == 2 implies
             * pivot_size[k+1] == 2, which in turn requires k+1 < n — but
             * that invariant isn't visible to clang-tidy's path analyser,
             * so we reject a malformed trailing 2x2 pivot here rather than
             * indexing past y/z. */
            if (k + 1 >= n) {
                free(y);
                free(z);
                return SPARSE_ERR_BADARG;
            }
            double d11 = F->D[k];
            double d22 = F->D[k + 1];
            double d21 = F->D_offdiag[k];
            double det = d11 * d22 - d21 * d21;
            double bscale = fabs(d11) + fabs(d22) + fabs(d21);
            double det_tol = (bscale > 0.0) ? solve_tol * bscale * bscale : sing_tol * sing_tol;
            if (fabs(det) < det_tol) {
                free(y);
                free(z);
                return SPARSE_ERR_SINGULAR;
            }
            double y_k1 = y[k + 1];
            z[k] = (d22 * y[k] - d21 * y_k1) / det;
            z[k + 1] = (d11 * y_k1 - d21 * y[k]) / det;
            k += 2;
        }
    }

    /* Phase 3: Backward solve L^T*v = z.  For each column j of L
     * right-to-left, the below-diagonal entries are exactly row j of
     * L^T; accumulate sum_{i>j} L[i,j] * z[i] and subtract from z[j]. */
    for (idx_t j = n - 1; j >= 0; j--) {
        idx_t start = F->L->col_ptr[j];
        idx_t end = F->L->col_ptr[j + 1];
        double sum = 0.0;
        for (idx_t p = start + 1; p < end; p++) {
            idx_t i = F->L->row_idx[p];
            sum += F->L->values[p] * z[i];
        }
        z[j] -= sum;
    }

    /* Phase 4: x[perm[i]] = z[i]  (apply P^T to z). */
    for (idx_t i = 0; i < n; i++)
        x[F->perm[i]] = z[i];

    free(y);
    free(z);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 19 Day 12: batched supernodal LDL^T — extract / writeback
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The implementations mirror the Cholesky helpers in sparse_chol_csc.c
 * (Sprint 18 Days 6 / 10).  See the design block at the top of this
 * file and the doc-comments on the declarations in
 * sparse_ldlt_csc_internal.h for the LDL^T-specific deltas (drop
 * threshold scale and D / D_offdiag / pivot_size handoff). */

/* Local copy of `chol_csc_bsearch_row_map` (static in
 * sparse_chol_csc.c).  Could be shared via the chol internal header,
 * but it's a five-line function and duplicating keeps the LDL^T side
 * loosely coupled. */
static idx_t ldlt_csc_bsearch_row_map(const idx_t *row_map, idx_t panel_height, idx_t target) {
    idx_t lo = 0;
    idx_t hi = panel_height;
    while (lo < hi) {
        idx_t mid = lo + (hi - lo) / 2;
        if (row_map[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < panel_height && row_map[lo] == target) {
        return lo;
    }
    return panel_height;
}

sparse_err_t ldlt_csc_supernode_extract(const LdltCsc *F, idx_t s_start, idx_t s_size,
                                        double *dense, idx_t lda, idx_t *row_map,
                                        idx_t *panel_height_out) {
    if (!F || !dense || !row_map || !panel_height_out)
        return SPARSE_ERR_NULL;
    if (!F->L)
        return SPARSE_ERR_NULL;
    if (s_start < 0 || s_size < 1 || s_start > F->n - s_size)
        return SPARSE_ERR_BADARG;

    const CholCsc *L = F->L;
    idx_t first_start = L->col_ptr[s_start];
    idx_t panel_height = L->col_ptr[s_start + 1] - first_start;
    if (lda < panel_height)
        return SPARSE_ERR_BADARG;
    if (panel_height < s_size)
        return SPARSE_ERR_BADARG;

    /* Seed row_map from the first column.  Supernodal invariant:
     * the first s_size entries are [s_start, ..., s_start + s_size - 1]
     * in order (the diagonal block precedes the shared panel rows). */
    for (idx_t i = 0; i < panel_height; i++)
        row_map[i] = L->row_idx[first_start + i];
    for (idx_t i = 0; i < s_size; i++) {
        if (row_map[i] != s_start + i)
            return SPARSE_ERR_BADARG;
    }
    *panel_height_out = panel_height;

    for (idx_t j = 0; j < s_size; j++) {
        idx_t c = s_start + j;
        idx_t cstart = L->col_ptr[c];
        idx_t cend = L->col_ptr[c + 1];
        for (idx_t p = cstart; p < cend; p++) {
            idx_t row = L->row_idx[p];
            idx_t local = ldlt_csc_bsearch_row_map(row_map, panel_height, row);
            if (local >= panel_height)
                return SPARSE_ERR_BADARG;
            dense[local + j * lda] = L->values[p];
        }
    }

    return SPARSE_OK;
}

sparse_err_t ldlt_csc_supernode_writeback(LdltCsc *F, idx_t s_start, idx_t s_size,
                                          const double *dense, idx_t lda, const idx_t *row_map,
                                          idx_t panel_height, const double *D_block,
                                          const double *D_offdiag_block,
                                          const idx_t *pivot_size_block, double drop_tol) {
    if (!F || !dense || !row_map || !D_block || !D_offdiag_block || !pivot_size_block)
        return SPARSE_ERR_NULL;
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size)
        return SPARSE_ERR_NULL;
    if (s_start < 0 || s_size < 1 || s_start > F->n - s_size)
        return SPARSE_ERR_BADARG;
    if (panel_height < s_size || lda < panel_height)
        return SPARSE_ERR_BADARG;

    CholCsc *L = F->L;

    /* Precompute per-column drop thresholds matching the scalar
     * `chol_csc_gather` invocations in `ldlt_csc_eliminate_native`:
     *   - 1×1 pivot at column j: threshold = drop_tol  (the scalar
     *     path passes raw drop_tol because dense_col[j] == 1.0).
     *   - 2×2 pair (j_first, j_first+1): threshold for both columns =
     *     drop_tol * (|d11| + |d22| + |d21|).  We disambiguate first
     *     vs second via D_offdiag_block (non-zero on first, zero on
     *     second) — robust against adjacent 2×2 pairs where
     *     pivot_size_block alone is ambiguous.
     *
     * Storing thresholds on the stack works because s_size is bounded
     * by SPARSE_MAX_SUPERNODE (set elsewhere); for general n we'd
     * need a heap allocation, but supernodes are sized for cache
     * efficiency well below that bound. */
    if ((size_t)s_size > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    double *thresholds = malloc((size_t)s_size * sizeof(double));
    if (!thresholds)
        return SPARSE_ERR_ALLOC;

    for (idx_t j = 0; j < s_size; j++) {
        if (pivot_size_block[j] == 1) {
            thresholds[j] = drop_tol;
        } else if (pivot_size_block[j] == 2) {
            /* Identify the first of this pair to source d11/d22/d21. */
            idx_t j_first = (D_offdiag_block[j] != 0.0) ? j : j - 1;
            if (j_first < 0 || j_first + 1 >= s_size) {
                free(thresholds);
                return SPARSE_ERR_BADARG;
            }
            double d11 = D_block[j_first];
            double d22 = D_block[j_first + 1];
            double d21 = D_offdiag_block[j_first];
            double bscale = fabs(d11) + fabs(d22) + fabs(d21);
            thresholds[j] = drop_tol * bscale;
        } else {
            free(thresholds);
            return SPARSE_ERR_BADARG;
        }
    }

    /* Per-column gather: walk the existing CSC slot, translate row →
     * local via row_map, write dense[local + j*lda] back into
     * values[p].  Drop below-diagonal entries below threshold; never
     * drop the diagonal (local == j by the supernodal invariant on
     * row_map's first s_size slots). */
    for (idx_t j = 0; j < s_size; j++) {
        idx_t c = s_start + j;
        idx_t cstart = L->col_ptr[c];
        idx_t cend = L->col_ptr[c + 1];
        double threshold = thresholds[j];
        for (idx_t p = cstart; p < cend; p++) {
            idx_t row = L->row_idx[p];
            idx_t local = ldlt_csc_bsearch_row_map(row_map, panel_height, row);
            if (local >= panel_height) {
                free(thresholds);
                return SPARSE_ERR_BADARG;
            }
            double v = dense[local + j * lda];
            if (local != j && fabs(v) < threshold)
                v = 0.0;
            L->values[p] = v;
        }

        /* Distribute the dense-block-factor's auxiliary outputs into
         * the LdltCsc.  These are owned by the caller's local scratch
         * (per the Day 13 plan) and copied verbatim here. */
        F->D[s_start + j] = D_block[j];
        F->D_offdiag[s_start + j] = D_offdiag_block[j];
        F->pivot_size[s_start + j] = pivot_size_block[j];
    }

    free(thresholds);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 19 Day 13: batched supernodal LDL^T — eliminate_diag /
 *                   eliminate_panel / eliminate_supernodal
 * ═══════════════════════════════════════════════════════════════════════
 *
 * See the design block in sparse_ldlt_csc_internal.h for the
 * higher-level architecture and the pivot-stability assumption that
 * lets the batched path skip BK-swap propagation. */

sparse_err_t ldlt_csc_supernode_eliminate_diag(const LdltCsc *F, idx_t s_start, idx_t s_size,
                                               double *dense, idx_t lda, const idx_t *row_map,
                                               idx_t panel_height, double *D_block,
                                               double *D_offdiag_block, idx_t *pivot_size_block,
                                               double tol) {
    if (!F || !F->L || !F->D || !F->D_offdiag || !F->pivot_size || !dense || !row_map || !D_block ||
        !D_offdiag_block || !pivot_size_block)
        return SPARSE_ERR_NULL;
    if (s_start < 0 || s_size < 1 || s_start > F->n - s_size)
        return SPARSE_ERR_BADARG;
    if (panel_height < s_size || lda < panel_height)
        return SPARSE_ERR_BADARG;

    const CholCsc *L = F->L;

    /* Scratch for L[s_start+j, k] (and L[s_start+j, k+1] when k is the
     * first of a 2×2 pair).  Allocated once outside the prior-column
     * loop. */
    if ((size_t)s_size > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    double *L_col_k = malloc((size_t)s_size * sizeof(double));
    double *L_col_k1 = malloc((size_t)s_size * sizeof(double));
    if (!L_col_k || !L_col_k1) {
        free(L_col_k);
        free(L_col_k1);
        return SPARSE_ERR_ALLOC;
    }

    /* External cmod from prior columns [0, s_start).
     *
     * Two prior-column shapes:
     *   - 1×1 pivot at k.  Apply `dense[r, j] -= L[r, k] * D[k] *
     *     L[s_start+j, k]` for every (row, j) where the L values
     *     exist.  Iterate k by 1.
     *   - 2×2 pivot pair (k, k+1).  Apply the four-term outer-product
     *     update:
     *       dense[r, j] -= L[r, k]   * (D[k]   * L[s_start+j, k]
     *                                  + D_off * L[s_start+j, k+1])
     *                   + L[r, k+1] * (D[k+1] * L[s_start+j, k+1]
     *                                  + D_off * L[s_start+j, k]).
     *     The "first of pair" disambiguator (D_offdiag[k] != 0)
     *     matches what `ldlt_csc_eliminate_native` writes — robust
     *     against adjacent 2×2 pairs.  Iterate k by 2 after the pair. */
    for (idx_t k = 0; k < s_start;) {
        idx_t pk = F->pivot_size[k];
        int is_2x2 = (pk == 2 && F->D_offdiag[k] != 0.0);
        if (is_2x2 && k + 1 >= s_start) {
            /* Pair would split the supernode boundary — defensive:
             * `ldlt_csc_detect_supernodes` shouldn't emit a supernode
             * starting on a 2×2 pair's second column, but if some
             * caller did, fall back to treating k as 1×1.  Returning
             * BADARG here would surface the inconsistency loudly. */
            free(L_col_k);
            free(L_col_k1);
            return SPARSE_ERR_BADARG;
        }
        idx_t step = is_2x2 ? 2 : 1;

        /* Collect L[s_start+j, k] (and L[s_start+j, k+1] for 2×2). */
        for (idx_t j = 0; j < s_size; j++)
            L_col_k[j] = 0.0;
        if (is_2x2)
            for (idx_t j = 0; j < s_size; j++)
                L_col_k1[j] = 0.0;

        int saw_super_row = 0;
        idx_t cstart = L->col_ptr[k];
        idx_t cend = L->col_ptr[k + 1];
        for (idx_t p = cstart; p < cend; p++) {
            idx_t row = L->row_idx[p];
            if (row < s_start)
                continue;
            if (row >= s_start + s_size)
                break;
            L_col_k[row - s_start] = L->values[p];
            saw_super_row = 1;
        }
        idx_t cstart1 = 0, cend1 = 0;
        if (is_2x2) {
            cstart1 = L->col_ptr[k + 1];
            cend1 = L->col_ptr[k + 2];
            for (idx_t p = cstart1; p < cend1; p++) {
                idx_t row = L->row_idx[p];
                if (row < s_start)
                    continue;
                if (row >= s_start + s_size)
                    break;
                L_col_k1[row - s_start] = L->values[p];
                saw_super_row = 1;
            }
        }

        if (!saw_super_row) {
            k += step;
            continue;
        }

        if (!is_2x2) {
            double dk = F->D[k];
            for (idx_t p = cstart; p < cend; p++) {
                idx_t row = L->row_idx[p];
                idx_t local = ldlt_csc_bsearch_row_map(row_map, panel_height, row);
                if (local >= panel_height)
                    continue;
                double v_r_k = L->values[p];
                double factor = v_r_k * dk;
                for (idx_t j = 0; j < s_size; j++) {
                    double ljk = L_col_k[j];
                    if (ljk != 0.0)
                        dense[local + j * lda] -= factor * ljk;
                }
            }
        } else {
            double dk = F->D[k];
            double dk1 = F->D[k + 1];
            double doff = F->D_offdiag[k];

            /* Walk column k for L[r, k] entries.  Per-row update:
             *   dense[r, j] -= L[r, k] * (dk * L_col_k[j] + doff * L_col_k1[j]). */
            for (idx_t p = cstart; p < cend; p++) {
                idx_t row = L->row_idx[p];
                idx_t local = ldlt_csc_bsearch_row_map(row_map, panel_height, row);
                if (local >= panel_height)
                    continue;
                double v_r_k = L->values[p];
                for (idx_t j = 0; j < s_size; j++) {
                    double term = dk * L_col_k[j] + doff * L_col_k1[j];
                    if (term != 0.0)
                        dense[local + j * lda] -= v_r_k * term;
                }
            }

            /* Walk column k+1 for L[r, k+1] entries.  Per-row update:
             *   dense[r, j] -= L[r, k+1] * (dk1 * L_col_k1[j] + doff * L_col_k[j]). */
            for (idx_t p = cstart1; p < cend1; p++) {
                idx_t row = L->row_idx[p];
                idx_t local = ldlt_csc_bsearch_row_map(row_map, panel_height, row);
                if (local >= panel_height)
                    continue;
                double v_r_k1 = L->values[p];
                for (idx_t j = 0; j < s_size; j++) {
                    double term = dk1 * L_col_k1[j] + doff * L_col_k[j];
                    if (term != 0.0)
                        dense[local + j * lda] -= v_r_k1 * term;
                }
            }
        }

        k += step;
    }

    free(L_col_k);
    free(L_col_k1);

    /* Mirror the lower triangle of the diagonal block to the upper
     * triangle so `ldlt_dense_factor`'s symmetric-input contract
     * holds (it reads `A[i + r*lda]` for r > i during BK Phase 2
     * sigma_r, which is the upper triangle). */
    for (idx_t j = 0; j < s_size; j++) {
        for (idx_t i = j + 1; i < s_size; i++)
            dense[j + i * lda] = dense[i + j * lda];
    }

    /* Dense Bunch-Kaufman LDL^T on the s_size × s_size diagonal slab.
     * Operates in place; reads/writes only [0, s_size) × [0, s_size).
     * The panel rows below are untouched here — the panel solve in
     * `ldlt_csc_supernode_eliminate_panel` consumes them post-factor. */
    sparse_err_t err = ldlt_dense_factor(dense, D_block, D_offdiag_block, pivot_size_block, s_size,
                                         lda, tol, NULL);
    if (err != SPARSE_OK)
        return err;

    /* Pivot-stability check: the dense factor's BK decisions must
     * match the cached scalar pass.  Mismatch means either the first
     * factor's pivot pattern wasn't preserved (e.g. F was not
     * pre-permuted by an earlier scalar factor) or numerical drift
     * during a refactor changed the BK choice — in either case the
     * caller must fall back to scalar. */
    for (idx_t j = 0; j < s_size; j++) {
        if (pivot_size_block[j] != F->pivot_size[s_start + j])
            return SPARSE_ERR_BADARG;
    }

    return SPARSE_OK;
}

sparse_err_t ldlt_csc_supernode_eliminate_panel(const double *L_diag, const double *D_block,
                                                const double *D_offdiag_block,
                                                const idx_t *pivot_size_block, idx_t s_size,
                                                idx_t lda_diag, double *panel, idx_t lda_panel,
                                                idx_t panel_rows) {
    if (!L_diag || !D_block || !D_offdiag_block || !pivot_size_block)
        return SPARSE_ERR_NULL;
    if (s_size < 1 || lda_diag < s_size || panel_rows < 0)
        return SPARSE_ERR_BADARG;
    if (panel_rows == 0)
        return SPARSE_OK;
    if (!panel)
        return SPARSE_ERR_NULL;
    if (lda_panel < panel_rows)
        return SPARSE_ERR_BADARG;

    if ((size_t)s_size > SIZE_MAX / sizeof(double))
        return SPARSE_ERR_ALLOC;
    double *row_buf = malloc((size_t)s_size * sizeof(double));
    if (!row_buf)
        return SPARSE_ERR_ALLOC;

    /* For each panel row i, two-phase solve:
     *   y = L_diag^{-1} * panel_row_i^T   (forward sub, unit lower triangular)
     *   x = D_block^{-1} * y              (block diagonal: 1×1 div / 2×2 inverse)
     *   panel_row_i = x. */
    for (idx_t i = 0; i < panel_rows; i++) {
        for (idx_t j = 0; j < s_size; j++)
            row_buf[j] = panel[i + j * lda_panel];

        /* Forward sub against unit L_diag.  L[j,j] = 1.0 so division
         * is a no-op but the per-row sum still needs to subtract the
         * lower-triangular contributions. */
        for (idx_t j = 0; j < s_size; j++) {
            double sum = row_buf[j];
            for (idx_t q = 0; q < j; q++)
                sum -= L_diag[j + q * lda_diag] * row_buf[q];
            row_buf[j] = sum;
        }

        /* Block diagonal solve.  Walk pivot blocks left-to-right. */
        idx_t j = 0;
        while (j < s_size) {
            if (pivot_size_block[j] == 1) {
                double dk = D_block[j];
                if (dk == 0.0) {
                    free(row_buf);
                    return SPARSE_ERR_SINGULAR;
                }
                row_buf[j] = row_buf[j] / dk;
                j += 1;
            } else if (pivot_size_block[j] == 2 && j + 1 < s_size && pivot_size_block[j + 1] == 2 &&
                       D_offdiag_block[j] != 0.0) {
                double d11 = D_block[j];
                double d22 = D_block[j + 1];
                double d21 = D_offdiag_block[j];
                double det = d11 * d22 - d21 * d21;
                if (det == 0.0) {
                    free(row_buf);
                    return SPARSE_ERR_SINGULAR;
                }
                double inv_det = 1.0 / det;
                double y0 = row_buf[j];
                double y1 = row_buf[j + 1];
                row_buf[j] = (d22 * y0 - d21 * y1) * inv_det;
                row_buf[j + 1] = (-d21 * y0 + d11 * y1) * inv_det;
                j += 2;
            } else {
                free(row_buf);
                return SPARSE_ERR_BADARG;
            }
        }

        for (idx_t q = 0; q < s_size; q++)
            panel[i + q * lda_panel] = row_buf[q];
    }

    free(row_buf);
    return SPARSE_OK;
}

sparse_err_t ldlt_csc_eliminate_supernodal(LdltCsc *F, idx_t min_size) {
    if (!F)
        return SPARSE_ERR_NULL;
    if (min_size < 1)
        return SPARSE_ERR_BADARG;

    idx_t n = F->n;
    if (n == 0)
        return SPARSE_OK;

    /* Same structural validation as `ldlt_csc_eliminate_native` so
     * misuse surfaces consistently across both entry points. */
    if (!F->L || !F->D || !F->D_offdiag || !F->pivot_size || !F->perm)
        return SPARSE_ERR_NULL;
    if (F->L->n != n || !F->L->col_ptr)
        return SPARSE_ERR_BADARG;

    /* Detect supernodes from cached `F->pivot_size`. */
    idx_t *starts = malloc((size_t)n * sizeof(idx_t));
    idx_t *sizes = malloc((size_t)n * sizeof(idx_t));
    if (!starts || !sizes) {
        free(starts);
        free(sizes);
        return SPARSE_ERR_ALLOC;
    }
    idx_t super_count = 0;
    sparse_err_t err = ldlt_csc_detect_supernodes(F, min_size, starts, sizes, &super_count);
    if (err != SPARSE_OK) {
        free(starts);
        free(sizes);
        return err;
    }

    LdltCscWorkspace *ws = NULL;
    err = ldlt_csc_workspace_alloc(n, &ws);
    if (err != SPARSE_OK) {
        free(starts);
        free(sizes);
        return err;
    }

    const double drop_tol = SPARSE_DROP_TOL;
    const double sing_tol = sparse_rel_tol(F->factor_norm, drop_tol);
    const double alpha_bk = ldlt_csc_bk_alpha();
    const double growth_bound = 1.0 / (100.0 * drop_tol);

    sparse_err_t rc = SPARSE_OK;
    idx_t super_idx = 0;
    idx_t j = 0;
    while (j < n) {
        /* Skip past detected size-1 supernodes; the per-column scalar
         * branch below handles that one column.  The min_size >= 2
         * guard in the size check below is what gates the batched
         * path's structural-pattern requirements (no fill inside a
         * fundamental supernode). */
        if (super_idx < super_count && j == starts[super_idx] && sizes[super_idx] == 1) {
            super_idx++;
        }

        if (super_idx < super_count && j == starts[super_idx] && sizes[super_idx] >= 2) {
            /* ── Batched supernode at column j ───────────────────── */
            idx_t s_start = j;
            idx_t s_size = sizes[super_idx];
            idx_t panel_height = chol_csc_supernode_panel_height(F->L, s_start);
            /* `s_size >= 2` already enforced by the outer `if`; the
             * remaining defensive checks reject malformed `F->L` (e.g.
             * an empty column at `s_start` or a panel shorter than the
             * supernode's diagonal block). */
            if (panel_height < 1 || panel_height < s_size) {
                rc = SPARSE_ERR_BADARG;
                break;
            }
            if ((size_t)panel_height > SIZE_MAX / sizeof(idx_t) ||
                (size_t)s_size > SIZE_MAX / (size_t)panel_height) {
                rc = SPARSE_ERR_ALLOC;
                break;
            }
            size_t dense_cells = (size_t)panel_height * (size_t)s_size;
            if (dense_cells > SIZE_MAX / sizeof(double)) {
                rc = SPARSE_ERR_ALLOC;
                break;
            }
            double *dense = calloc(dense_cells, sizeof(double));
            idx_t *row_map = malloc((size_t)panel_height * sizeof(idx_t));
            double *D_block = malloc((size_t)s_size * sizeof(double));
            double *D_off_block = malloc((size_t)s_size * sizeof(double));
            idx_t *ps_block = malloc((size_t)s_size * sizeof(idx_t));
            if (!dense || !row_map || !D_block || !D_off_block || !ps_block) {
                free(dense);
                free(row_map);
                free(D_block);
                free(D_off_block);
                free(ps_block);
                rc = SPARSE_ERR_ALLOC;
                break;
            }

            idx_t ph_out = 0;
            rc = ldlt_csc_supernode_extract(F, s_start, s_size, dense, panel_height, row_map,
                                            &ph_out);
            if (rc == SPARSE_OK)
                rc = ldlt_csc_supernode_eliminate_diag(F, s_start, s_size, dense, panel_height,
                                                       row_map, panel_height, D_block, D_off_block,
                                                       ps_block, drop_tol);
            if (rc == SPARSE_OK) {
                idx_t panel_rows = panel_height - s_size;
                if (panel_rows > 0)
                    rc = ldlt_csc_supernode_eliminate_panel(dense, D_block, D_off_block, ps_block,
                                                            s_size, panel_height, dense + s_size,
                                                            panel_height, panel_rows);
            }
            if (rc == SPARSE_OK)
                rc = ldlt_csc_supernode_writeback(F, s_start, s_size, dense, panel_height, row_map,
                                                  panel_height, D_block, D_off_block, ps_block,
                                                  drop_tol);
            if (rc == SPARSE_OK) {
                /* Populate row-adjacency for every column in the
                 * supernode so subsequent scalar (or batched) columns
                 * can iterate `row_adj[col]` instead of `[0, k)`. */
                for (idx_t jj = 0; jj < s_size && rc == SPARSE_OK; jj++)
                    rc = ldlt_csc_populate_row_adj(F, s_start + jj);
            }

            free(dense);
            free(row_map);
            free(D_block);
            free(D_off_block);
            free(ps_block);
            if (rc != SPARSE_OK)
                break;

            j += s_size;
            super_idx++;
        } else {
            /* ── Scalar single-column step at j ───────────────────── */
            rc = ldlt_csc_eliminate_one_step(F, ws, &j, drop_tol, sing_tol, alpha_bk, growth_bound);
            if (rc != SPARSE_OK)
                break;
        }
    }

    ldlt_csc_clear_dense_col(ws);
    ldlt_csc_clear_dense_col_r(ws);
    ldlt_csc_workspace_free(ws);
    free(starts);
    free(sizes);
    return rc;
}
