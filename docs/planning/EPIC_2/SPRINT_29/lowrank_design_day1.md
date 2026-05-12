# Sprint 29 Day 1 — Item 1: Sparse Low-Rank Without Dense Accumulator (Design)

## Decision

**Pick (b) per-cell outer-product accumulator with the existing `drop_tol` cutoff.**  Replace the m×n dense intermediate (`calloc(m*n, sizeof(double))`) in `sparse_svd_lowrank_sparse()` with a per-`(i, j)` loop that evaluates `sum_{s=0}^{rank_k-1} σ_s · U[i, s] · V^T[s, j]` and inserts cells above `drop_tol` directly into the sparse output.  Memory drops O(m·n) → O(nnz_result); wall stays O(m·n·rank_k) (no algorithmic regression on the dominant inner loop).

Rejected alternatives:
- **(a) Top-K-per-row pruning** — would replace the per-cell `drop_tol` cutoff with an "insert at most K largest-magnitude entries per row" rule.  Cheaper to compute (saves the sum-then-threshold per cell) but changes the semantic contract of the function (existing callers rely on the `drop_tol` cutoff per `tests/test_svd.c::test_lowrank_sparse_vs_dense` which asserts `||A_sparse - A_dense||_F < drop_tol · sqrt(m·n)`).  Rejected to preserve back-compat with existing tests + callers.
- **(c) Hashmap-of-cells accumulator** — maintain a hashmap of `(i, j) → cumulative value` across all rank-1 outer products, threshold + insert at the end.  Has the same final result as (b) but worse cache behaviour (random hashmap access vs sequential `(i, j)` scan) and the temporary hashmap can grow to O(m·n) for dense intermediate states.  Rejected because (b)'s per-cell sum-then-threshold path is both simpler + cache-friendlier.

## Background — Existing Path Memory Profile

`src/sparse_svd.c::sparse_svd_lowrank_sparse()` (the Sprint-17 baseline implementation):

```c
double *accum = calloc(mn, sizeof(double));   /* m × n × 8 bytes */
for (idx_t s = 0; s < rank_k; s++)
    for (idx_t i = 0; i < m; i++) {
        double u_scaled = σ_s · U[i, s];
        for (idx_t j = 0; j < n; j++)
            accum[i*n + j] += u_scaled · Vt[j*k + s];
    }
for (idx_t i = 0; i < m; i++)
    for (idx_t j = 0; j < n; j++)
        if (fabs(accum[i*n + j]) >= drop_tol)
            sparse_insert(out, i, j, accum[i*n + j]);
free(accum);
```

Memory footprint:
- SVD itself: `U` is m·k·8 bytes; `Vt` is k·n·8 bytes; `σ` is k·8 bytes.
- **Dense accumulator: m·n·8 bytes.**  Dominates SVD memory once `min(m, n) > rank_k`.
- Sparse output: O(nnz_result) — for typical SuiteSparse fixtures, `nnz_result << m·n` after thresholding.

Per-fixture intermediate sizes at `rank_k = 50`:

| Fixture | n | m·n·8 bytes (accumulator) | SVD U + Vt + σ (8 bytes each) |
|---|---:|---:|---:|
| nos4 | 100 | 80 KB | 80 KB |
| bcsstk04 | 132 | 139 KB | 106 KB |
| Kuu | 7 102 | 403 MB | 5.7 MB |
| bcsstk14 | 1 806 | 26 MB | 1.4 MB |
| s3rmt3m3 | 5 357 | 230 MB | 4.3 MB |
| **Pres_Poisson** | 14 822 | **1.76 GB** | 11.9 MB |

The accumulator is 100-150× the SVD itself on the larger fixtures.

## New Algorithm — Per-Cell Outer-Product Accumulator

```c
SparseMatrix *out = sparse_create(m, n);
idx_t k = svd.k;
for (idx_t i = 0; i < m; i++) {
    for (idx_t j = 0; j < n; j++) {
        double val = 0.0;
        for (idx_t s = 0; s < rank_k; s++)
            val += σ[s] · U[i + s*m] · Vt[j*k + s];
        if (fabs(val) >= drop_tol)
            sparse_insert(out, i, j, val);
    }
}
```

Memory footprint:
- SVD itself: same as before (U + Vt + σ).
- **No dense accumulator.**  The `val` register holds a single scalar per `(i, j)`.
- Sparse output: O(nnz_result) — same as before.

Wall cost: O(m·n·rank_k) (same as the existing accumulator-fill loop's `for s for i for j` triple).  The trailing `for i for j` threshold-and-insert pass in the existing path is folded into the same triple loop, saving a final m·n pass over the dense buffer.

Expected memory wins (relative to the existing path):

| Fixture | Existing peak (SVD + accumulator) | New peak (SVD only) | Memory delta |
|---|---:|---:|---:|
| nos4 | 160 KB | 80 KB | -50% |
| Kuu | 409 MB | 5.7 MB | -98.6% |
| **Pres_Poisson** | **1.77 GB** | **12 MB** | **-99.3%** |

## Thresholding Strategy — Per-Cell `drop_tol` Cutoff

Decision: **preserve the existing per-cell `drop_tol` cutoff** (`fabs(val) >= drop_tol`).

Rejected: top-K-per-row pruning (e.g., "insert at most K largest-magnitude entries per row").  Per-row pruning would change the semantic contract — existing callers (`tests/test_svd.c::test_lowrank_sparse_vs_dense`) assert `||A_sparse - A_dense||_F < drop_tol · sqrt(m·n)` which holds under the per-cell cutoff but not under per-row pruning (the dropped entries are bounded individually, not aggregated per-row).  Per-row pruning is a useful future axis (would help cases where `drop_tol` is too coarse for `min(m,n)`-class fixtures), but it's a behaviour change rather than a memory optimisation.  Sprint 30+ may revisit if a caller surfaces a need.

## Numeric Comparability vs Existing Path

The new path computes `val[i,j] = sum_s σ_s · U[i,s] · Vt[s,j]` in a different summation order than the existing path:

- **Existing path**: outer loop on `s`; for each rank, accumulate `σ_s · U[i,s] · Vt[s,j]` into the m·n buffer.  Per-cell accumulation order: rank-by-rank, oldest-to-newest.
- **New path**: outer loop on `(i,j)`; for each cell, sum `s=0..rank_k-1` in one pass.  Per-cell accumulation order: rank-by-rank, oldest-to-newest (same).

Both paths add the same floating-point operands in the same order per cell → the per-cell result is bit-identical.  The new path's `for i for j for s` triple loop is just a transposition of the existing `for s for i for j` triple — the inner-loop products are evaluated in different orders ACROSS cells but not WITHIN a cell.  Frobenius residual tolerance in the Day-2 test (`1e-10`) absorbs any cross-platform floating-point reordering.

## Implementation Surface

**Day 1 (this commit):**
- Add `parse_svd_lowrank_outer()` env-var parser recognising `SPARSE_SVD_LOWRANK_OUTER={off (default), on}`.
- Add `sparse_svd_lowrank_outer_product()` internal helper stub: signature accepts the SVD struct + `rank_k` + `drop_tol` + `m` + `n` + output pointer; Day-1 body returns an empty sparse matrix (signals stub state for the failing-as-expected test).
- Dispatch in `sparse_svd_lowrank_sparse()`: when env-on, call the new helper; otherwise route through the existing dense-intermediate path.  Default-off path bit-identical to Sprint 28.
- Failing-as-expected test `tests/test_svd.c::test_sparse_svd_lowrank_outer_product_matches_dense`: 32×32 diagonal synthetic with σ = {10, 5, 1, 0.1, 0, ...}; runs both env-off + env-on; asserts `||A_off - A_on||_F / ||A_off||_F ≤ 1e-10`.  RUN_TEST commented out until Day 2; verified in dry-run to trip the assertion under the Day-1 empty-matrix stub.

**Day 2 (planned):**
- Replace the empty-matrix stub in `sparse_svd_lowrank_outer_product()` with the per-cell accumulator described above.
- Light up the Day-1 RUN_TEST line; assertion should pass.
- Add corpus-safety test `test_sparse_svd_lowrank_outer_product_corpus_safety` over nos4 / bcsstk04 / bcsstk14 at `rank_k ∈ {10, 50}`.
- Extend `bench_svd.c` (or write one-off `/tmp/bench_lowrank_day2.c`) to measure wall + peak memory under both paths; capture to `docs/planning/EPIC_2/SPRINT_29/lowrank_sweep_day2.txt`.
- Flip the production default to `on` if the sweep clears ≥ 50% memory + ≥ 30% wall reduction without corpus regression past 5%.  Otherwise ship as advisory.

## LOC Estimate

- Day 1 (this commit): ~80 LOC (parser + stub + dispatch + test).  Actual landed: ~75 LOC.
- Day 2: ~30 LOC (replace empty-matrix stub with per-cell accumulator) + ~50 LOC corpus-safety test + ~50 LOC bench helper (if not folded into bench_svd.c).  Total ~130 LOC across Day 2.

## What Ships in Sprint 29 Day 1

- `src/sparse_svd.c`: env-var parser + stub helper + dispatch.  Default-off path bit-identical to Sprint 28.
- `tests/test_svd.c`: failing-as-expected stub (RUN_TEST commented out).
- `tests/test_svd.c`: `_POSIX_C_SOURCE 200809L` guard at the top of the file (needed for `tf_setenv` / `tf_unsetenv` macros from test_framework.h — same pattern as the 3 Sprint-28 test files).
- `docs/planning/EPIC_2/SPRINT_29/lowrank_design_day1.md` (this doc).
- All quality checks clean.

## References

- `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 29 Item 1 (lines 777, 790).
- `docs/planning/EPIC_2/SPRINT_29/PLAN.md` Day 1 + Day 2 sections.
- `src/sparse_svd.c::sparse_svd_lowrank_sparse()` lines 1274-1384 — the existing dense-intermediate path.
- `tests/test_svd.c::test_lowrank_sparse_vs_dense` line ~2890 — existing test that pins the `drop_tol`-based semantic contract.
- Sprint 17 / 18 PERF_NOTES.md — SVD low-rank baselines.
