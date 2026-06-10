# Sprint 62 Day 10: Cholesky Preservation Hardening Batch

Date: 2026-06-10
Branch: `sprint-62`

## Purpose

Land the bounded Cholesky one-shot hardening slice designed on Day 9:
strengthen reordered one-shot publication/preservation semantics without
redesigning the broader no-reorder cancel model or widening into the shared
direct lifecycle API.

## Main Result

### 1. Reordered Cholesky one-shot calls now publish back only on success

The landed implementation moved the exact seam Day 9 targeted:

- reordered `sparse_cholesky_factor_opts(...)` now builds a temporary
  symmetrically permuted working copy
- the chosen Cholesky backend factors that working copy in the reordered
  coordinate space
- reordered/factored payload is transplanted back onto the caller-owned matrix
  only after successful completion

That means:

- cancelled reordered one-shot attempts no longer strand the caller matrix in a
  partially reordered intermediate state
- failed reordered attempts also preserve the caller-owned matrix in its
  original coordinate space

### 2. The touched-file fence stayed exact

Touched:

- `include/sparse_cholesky.h`
- `src/sparse_cholesky.c`
- `tests/test_integration.c`

Not widened into:

- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt.c`
- `src/sparse_qr.c`
- `tests/test_cholesky.c`
- docs/example/benchmark surfaces

### 3. Public contract now matches the strengthened reordered behavior

`include/sparse_cholesky.h` now states directly that:

- one-shot Cholesky remains the first-class/default direct path
- stable-pattern repeated runs still belong on the shared direct lifecycle
- reordered one-shot attempts may factor a temporary reordered working copy
  and publish back only on success
- cancelled or failed reordered one-shot attempts leave the caller matrix in
  its original coordinate space

## Implementation Shape

### Internal helper split

The landed shape in `src/sparse_cholesky.c` is:

- `s62_cholesky_factor_reordered_working_copy(...)`
  - builds the reordered working matrix
  - runs the no-reorder backend path on that working copy
  - publishes reordered/factored payload back only after success
- `s62_cholesky_factor_opts_no_reorder(...)`
  - owns the pre-existing no-reorder backend dispatch path
  - exists specifically so the working-copy helper does not recurse through the
    public wrapper
- `s62_cholesky_steal_factor_payload(...)`
  - transplants factored matrix storage/state back to the caller matrix

### Important implementation note

The first cut reused `sparse_cholesky_factor_opts(...)` recursively on the
working copy. That was behaviorally fine but failed `clang-tidy`’s
`misc-no-recursion` rule. The landed version flattened the helper through the
dedicated internal no-reorder function instead.

## Regression Proof

The new public regression stayed in `tests/test_integration.c`:

- `test_progress_cb_cholesky_cancel_after_reorder_preserves_original_matrix`

It proves all three required caller-facing behaviors:

- cancelled reordered Cholesky attempt preserves the original caller matrix
  entries and identity row/col perms
- cancelled matrix remains unfactored and solve is rejected
- later reordered one-shot retry succeeds on the same caller matrix

## Preserved Compatibility

Preserved:

- one-shot Cholesky remains first-class/default
- repeated-run direct solves remain on
  `sparse_analyze()` / `sparse_factor_numeric()` / `sparse_factor_solve()` /
  `sparse_refactor_numeric()`
- no-reorder linked-list cancel-at-step-0 remains compatibility-preserved and
  non-bit-identical
- backend selection remains inside the existing Cholesky wrapper

Strengthened:

- reordered one-shot cancellation/failure no longer publishes partial reordered
  state
- reordered retry remains possible on the original caller matrix

Still deferred:

- no-reorder linked-list cancel bit-identity restoration
- CSC progress callback parity for Cholesky
- LDL^T / QR convergence work
- broader direct-family docs/examples simplification

## Validation

Required gate run:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Result:

- all passed

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 357.85 sec`

Non-blocking validation note:

- the reviewed CMake rebuild again emitted the ordinary
  `bench_eigs_reuse.c` double-promotion warnings while rebuilding that bench
  binary, but the full reviewed path still completed cleanly and passed all
  parity gates

## Exit State

Sprint 62 now has a coherent direct-usability package across:

- LU reordered one-shot preservation hardening
- Cholesky reordered one-shot preservation hardening
- preserved explicit shared direct lifecycle boundary

The remaining Sprint 62 direct-family queue is now smaller and more
compatibility-oriented than the sprint opened with.
