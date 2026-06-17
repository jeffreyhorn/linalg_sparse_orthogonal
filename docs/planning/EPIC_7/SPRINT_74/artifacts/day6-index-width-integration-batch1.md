# Sprint 74 Day 6: Index-Width Integration Batch 1

Date: 2026-06-16
Branch: `sprint-74`

## Purpose

Land the first bounded Sprint 74 capability-modernization batch so the width
contract becomes an explicit compile-time surface and the matrix shell uses the
checked width bridge more consistently.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_74/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_74/artifacts/day5-index-scalar-architecture-design.md`
- `include/sparse_types.h`
- `src/sparse_types.c`
- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`
- `include/sparse_matrix.h`
- `src/sparse_matrix.c`
- `tests/test_sparse_matrix.c`

## Day 6 Landing

### 1. The public width contract is now explicit and bounded

The first Sprint 74 batch does not ship a repo-wide 64-bit conversion.

It lands one clearer compile-time width contract in `include/sparse_types.h`:

- `SPARSE_IDX_BITS` selects `32` or `64`
- `idx_t` and `IDX_MAX` now come from that single contract
- `SPARSE_PRIDX` and `SPARSE_SCNIDX` expose matching print/scan format macros
- `_Static_assert` ties the selected macro width back to `sizeof(idx_t)`
- `sparse_idx_bits()` reports the selected width at runtime

The reviewed default build stays exactly where it was: `32`-bit.

### 2. The checked width bridge is now the clearer owner

The batch tightens the internal bridge in:

- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`

The landed follow-through is:

- checked conversion helpers now reject null output pointers
- `sparse_size_to_idx_checked(...)` now compares through `uintmax_t`
- `sparse_malloc_idx_array(...)` and `sparse_calloc_idx_array(...)` now reuse
  the checked conversion helper path directly

That preserves current overflow/failure behavior while making the bridge less
duplicated and easier to trust.

### 3. The matrix shell now uses that bridge on its highest-value width seam

The matrix-shell follow-through stays bounded to:

- `include/sparse_matrix.h`
- `src/sparse_matrix.c`

The landed cleanup is:

- checked shell-buffer allocation / teardown helpers for create/free
- checked permutation byte sizing in copy paths
- checked and saturating `sparse_memory_usage(...)` accumulation
- checked `idx_t` support-buffer allocation in `sparse_matmul(...)`
- width-aware Matrix Market and matrix-print formatting/scanning via
  `SPARSE_PRIDX` / `SPARSE_SCNIDX`

This keeps the public matrix shell stable while making its width-sensitive
allocation and I/O seams more coherent.

### 4. The proof stayed narrow

The only proof expansion is in `tests/test_sparse_matrix.c`.

The new `test_idx_width_contract(...)` proves:

- `sparse_idx_bits()` matches `SPARSE_IDX_BITS`
- `sizeof(idx_t)` matches that width
- the maintained default reviewed build still maps `idx_t` and `IDX_MAX` to
  the expected 32-bit contract values

No broader support or documentation surfaces were forced to move.

## Validation

Because `*.c` and `*.h` changed, I ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 477.93 sec`

Raw touched-surface `wc -l` counts:

- `include/sparse_types.h` = `278`
- `src/sparse_types.c` = `52`
- `include/sparse_matrix.h` = `610`
- `src/sparse_matrix.c` = `1125`
- `src/sparse_alloc_internal.h` = `63`
- `src/sparse_alloc_internal.c` = `60`
- `tests/test_sparse_matrix.c` = `1071`

## Exit State

Sprint 74 Day 6 exits with:

1. one explicit compile-time width contract
2. one clearer checked `idx_t` <-> `size_t` bridge owner
3. one bounded matrix-shell width follow-through
4. one focused width-contract regression in the matrix proof owner
5. one fully validated first capability-modernization landing
