# Sprint 41 Day 6 Artifact: Core Module Migration Batch 2

## Purpose

Record the completion of Sprint 41's planned first-wave hotspot migration
set: the remaining `sparse_etree.c` helper-consolidation batch plus the
cleanup needed to reconcile `sparse_dense.c` with the same shared helper
ownership model.

## Day 6 Batch Scope

The first-wave hotspot list entering Day 6 was:

- `src/sparse_dense.c`
- `src/sparse_svd.c`
- `src/sparse_eigs.c`
- `src/sparse_etree.c`

Day 5 completed the SVD/eigs pair. Day 6 completed the remaining first-wave
work by:

- migrating `src/sparse_etree.c`
- reconciling remaining manual byte-count logic in `src/sparse_dense.c`

This batch stayed within helper consolidation. It did **not** attempt a broad
algorithm refactor.

## `src/sparse_etree.c` Migration Result

### Removed local helper

Day 6 removed the file-local:

- `alloc_would_overflow(...)`

That helper had been the remaining generic n-based allocation guard in the
final untouched first-wave hotspot module.

### Shared helpers adopted

The generic safety seam now uses the shared helper layer instead:

- `sparse_malloc_array(...)`
- `sparse_calloc_array(...)`
- `sparse_count_bytes_overflow(...)`
- `sparse_idx_count_bytes_overflow(...)`
- `sparse_size_add_overflow(...)`
- `sparse_size_to_idx_checked(...)`

### Migrated `etree` families

The Day 6 `etree` migration covered:

- etree/postorder work arrays
- child-list arrays
- marker/tmp scratch arrays
- propagated row-set arrays
- symbolic Cholesky `col_ptr` / `row_idx` sizing
- symbolic LU bridge arrays
- symbolic U-structure accumulation sizing

### What stayed local in `etree`

Day 6 intentionally kept these local:

- symbolic-structure traversal and fill logic
- prefix-sum / monotonicity meaning
- symbolic LU/Cholesky bridge ownership
- algorithm-specific cleanup sequencing

That is the right Sprint 41 boundary:

- generic allocation/overflow mechanics move to the shared layer
- symbolic meaning and traversal behavior stay file-local

## `src/sparse_dense.c` Reconciliation Result

Day 4 had already landed the first proof integration in `dense_create()`, but
several manual byte-count checks still remained. Day 6 aligned those with the
shared helper style:

- `dense_gemm(...)`
  - now uses shared overflow + byte-count helpers for output zeroing
- `dense_gemv(...)`
  - now uses shared byte-count derivation for `y`
- `tridiag_qr_eigenpairs(...)`
  - now uses `sparse_malloc_array(...)` for sort/permutation scratch buffers

This matters because it keeps the first-wave hotspot set internally
consistent: the named dense hotspot no longer mixes an early shared-helper
proof with lingering manual byte arithmetic in adjacent code.

## First-Wave Status After Day 6

After the Day 6 batch:

- `src/sparse_dense.c` — migrated/reconciled
- `src/sparse_svd.c` — migrated
- `src/sparse_eigs.c` — migrated
- `src/sparse_etree.c` — migrated

So the planned first-wave hotspot list is complete.

## Broader `src/` Gap List Handed Forward

The post-Day-6 sweep leaves the next migration queue concentrated in:

- `src/sparse_ic.c`
- `src/sparse_iterative.c`
- `src/sparse_analysis.c`
- `src/sparse_qr.c`
- `src/sparse_graph.c`

The broader migration pressure is now cleaner:

- direct `malloc((size_t)n * sizeof(T))` / `calloc((size_t)n, sizeof(T))`
  families
- manual `SIZE_MAX / sizeof(...)` checks
- remaining `SIZE_MAX - ...` accumulation guards
- modules already using `sparse_size_mul_overflow(...)` but not the source
  wrappers

This is the correct Day 7 handoff: the broader queue is explicit, and it is no
longer mixed with incomplete first-wave hotspot work.

## Validation Result

Because `*.c` changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

## Highest-Value Conclusion

Sprint 41 now has a validated completed first-wave hotspot migration set.
`sparse_etree.c` no longer carries its own generic allocation-overflow helper,
and `sparse_dense.c` no longer carries the first-wave's lingering manual
byte-count drift. That leaves the rest of the sprint free to focus on the
broader `src/` audit and migration queue rather than revisiting the initial
hotspot set.
