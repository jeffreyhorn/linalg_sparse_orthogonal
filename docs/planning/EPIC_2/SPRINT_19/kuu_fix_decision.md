# Kuu Regression — Fix Decision (Sprint 19 Day 5)

## Background

Sprint 18 Day 12's larger-corpus bench found that scalar CSC
Cholesky regresses to **0.77×** vs linked-list on `Kuu.mtx`
(n = 7102) — the one matrix in the Sprint 18 corpus where scalar
CSC loses.  Every other matrix shows scalar CSC flat or ahead.  The
supernodal path side-steps the problem via full `sym_L`
pre-allocation, so the regression is not user-visible through
`sparse_cholesky_factor_opts` (which dispatches to supernodal at
n ≥ SPARSE_CSC_THRESHOLD = 100), but the scalar kernel should still
be restored to the monotonic n-vs-speedup trend.

Copilot's PR #26 review flagged the same behaviour and the Sprint 18
retrospective deferred the fix to Sprint 19.

## Profile confirmation

`sample` on `/tmp/profile_kuu` (scalar CSC factor of Kuu, 10 iters,
12-second window) attributed user-thread time as follows
(`docs/planning/EPIC_2/SPRINT_19/profile_day5_kuu.txt`):

| function | samples | share of `chol_csc_eliminate` |
|----------|--------:|------------------------------:|
| `chol_csc_eliminate` (total) | 8561 | 100% |
| &nbsp;&nbsp;↳ `chol_csc_gather` | 5319 | 62% |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ `_platform_memmove` (via `shift_columns_right_of`) | **5167** | **60%** |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ `qsort` / `idx_t_cmp` (pattern sort) | 40 | < 1% |
| &nbsp;&nbsp;↳ `chol_csc_cmod` (numeric work) | 3213 | 37% |

**Finding:** 60% of Kuu's scalar CSC factor time is spent in
`memmove` inside `chol_csc_gather`'s `shift_columns_right_of`
path.  Only 37% is real numerical work.  Hypothesis confirmed —
the regression is not drop-tolerance miscalibration, cmod fill-in,
or any secondary cost; it is specifically the per-column shift.

## Options

### Option A — pre-allocate full `sym_L` in `chol_csc_from_sparse`

Match `chol_csc_from_sparse_with_analysis` by computing the symbolic
L pattern up front (via the `sparse_etree` + `sparse_colcount` +
`sparse_symbolic_cholesky` pipeline, same as `sparse_analyze`) and
sizing the CSC's `values` / `row_idx` arrays to `sym_L.nnz` before
elimination runs.  Then eliminate writes into pre-sized slots and
never needs `shift_columns_right_of`.

**Implementation sketch:**

1. Factor the symbolic phase into a static helper
   `compute_sym_L_pattern(const SparseMatrix *mat, const idx_t *perm,
   idx_t **col_ptr_out, idx_t **row_idx_out, idx_t *nnz_out)`.
   Reuses the existing `sparse_etree_compute` +
   `sparse_colcount` + `sparse_symbolic_cholesky` path without
   building a full `sparse_analysis_t`.
2. `chol_csc_from_sparse` calls the helper (when the matrix is
   symmetric; otherwise falls back to the heuristic for defensive
   compatibility with any non-Cholesky caller) and pre-allocates
   with the returned `col_ptr` / `row_idx`.  Values get A's
   lower-triangle entries at their matching positions; fill
   positions initialised to 0.0.
3. `chol_csc_gather` rewrites to write into pre-sized slots
   without `shift_columns_right_of`.  The drop-tolerance rule
   still applies: entries below `drop_tol * |L[j, j]|` get zeroed
   in place (same semantics as `chol_csc_supernode_writeback` from
   Sprint 18 Day 10).  `col_ptr` is immutable post-allocation.
4. Downstream consumers (`chol_csc_solve`, `chol_csc_writeback_to_sparse`)
   already tolerate zero-valued stored entries — Sprint 18 Day 10
   added the `v == 0.0 ? continue` gate in the latter.

**Memory cost on the Sprint 18 + Day 3 corpus:**

| fixture | n | A.nnz | heuristic cap | sym_L cap | ratio |
|---------|--:|------:|--------------:|----------:|------:|
| nos4 | 100 | 594 | 694 | 637 | **0.92×** |
| bcsstk04 | 132 | 3 648 | 3 780 | 3 143 | **0.83×** |
| bcsstk14 | 1 806 | 63 454 | 65 260 | 116 071 | 1.78× |
| s3rmt3m3 | 5 357 | 207 123 | 212 480 | 474 609 | 2.23× |
| Kuu | 7 102 | 340 200 | 347 302 | 406 264 | 1.17× |
| Pres_Poisson | 14 822 | 715 804 | 730 626 | 2 668 793 | **3.65×** |

Small matrices (nos4, bcsstk04): sym_L is **smaller** than the
heuristic (sym_L is lower-triangular only; heuristic is
`fill_factor × A.nnz` which covers the full pattern).  No memory
regression below n = 132.

Larger matrices: sym_L is up to 3.65× larger on Pres_Poisson.  This
is the same cost `chol_csc_from_sparse_with_analysis` already pays,
and the supernodal dispatch has been using it since Sprint 18
Day 11 without user complaints.  Adding it to scalar aligns scalar
with supernodal instead of creating a new memory footprint.

**Complexity:** Medium.  One new static helper, one changed
initialiser, one rewritten `chol_csc_gather`.  Estimated 10-12
hours.  Low risk because `_with_analysis` already proves the
pattern works end-to-end.

### Option B — batch the shifts inside `chol_csc_gather`

Keep the dynamic-growth path but accumulate multiple column deltas
and issue one bulk memmove at the end of a batch (e.g., every
256 columns, or at the end of elimination).  The current per-column
memmove becomes an amortised bulk memmove.

**Complexity:** High.  Requires:
- Queueing per-column survivor lists across the batch (each column
  can shrink or grow; the queued state has to preserve each
  column's final row_idx / values).
- Maintaining the "current col_ptr" view during elimination so
  subsequent columns can bsearch against in-progress columns (the
  left-looking cmod reads stored L(row, k) for k < j).
- Serialising the batched shift at the right moment — premature
  shift invalidates indices the kernel is currently using.
- Preserving `shift_columns_right_of`'s overflow guards (idx_t /
  size_t saturation) across the batched math.

Estimated 20-24 hours.  High risk because the kernel's
left-looking invariant spans across columns that are simultaneously
"waiting for shift" and "already used as reference".

Zero memory cost over the current heuristic (no change to
`chol_csc_from_sparse`).

### Option comparison

| criterion | Option A | Option B |
|-----------|---------:|---------:|
| scalar-vs-supernodal design consistency | aligned | diverges |
| memory overhead (Kuu) | +17% over heuristic | 0 |
| memory overhead (small n < 100) | −8% to −17% | 0 |
| implementation effort | 10-12 hrs | 20-24 hrs |
| risk level | low | high |
| eliminates memmove entirely | yes | no (amortised) |
| precedent in code base | `_with_analysis` + supernodal | none |

## Decision: Option A

**Reasoning:**

1. The memory cost is not additional — `chol_csc_from_sparse_with_analysis`
   already pays it, and the supernodal dispatch has used that path
   in production since Sprint 18 Day 11.  Option A extends the same
   allocation pattern to the scalar kernel; the library will have
   one allocation story instead of two.
2. On the small-matrix range where the heuristic is nominally more
   frugal, sym_L is actually smaller (0.83×–0.92×) because the
   heuristic uses `fill_factor × A.nnz` covering A's full symmetric
   pattern while sym_L stores only the lower triangle.
3. Option A eliminates the memmove entirely (not just amortised),
   giving a predictable ~2.5× speedup on Kuu (the measured 60%
   memmove share disappears) and similar gains on any matrix whose
   gather currently shrinks columns.
4. Option B's risk profile is poor: changing the serialisation of
   left-looking cmod vs gather is the kind of subtle reordering
   that can produce correct residuals on most matrices and wrong
   answers on a narrow class — exactly the profile that pierced
   Sprint 18 Day 12's supernodal sym_L bug.

Expected outcome post-fix:

- Kuu scalar CSC speedup rises from 0.77× to approximately **1.5×
  — 2.0×** (the cmod fraction remains at ~37% of the current
  runtime; removing the 60% memmove share puts scalar CSC above
  linked-list by the ratio of the remaining work).
- Every other fixture in the corpus either stays flat or improves
  (no regression expected).
- Scalar CSC's n-vs-speedup trend becomes monotonic on the
  enlarged corpus.

## Follow-up: Days 6 + 7

- **Day 6:** implement the `compute_sym_L_pattern` helper,
  rewrite `chol_csc_from_sparse` to pre-allocate, rewrite
  `chol_csc_gather` to write-into-slot without shifts, add the
  `test_chol_csc_kuu_scalar_no_regression` test.
- **Day 7:** full-corpus re-bench, regression sweep vs
  `bench_day14.txt`, update `PERF_NOTES.md` Kuu row + narrative,
  memory delta check on the small-matrix corpus.
