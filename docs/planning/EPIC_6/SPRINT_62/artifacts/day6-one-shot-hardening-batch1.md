# Sprint 62 Day 6: One-Shot Hardening Batch I

Date: 2026-06-10
Branch: `sprint-62`


## Purpose

Land the first bounded direct-usability hardening slice on the highest-value
direct path by making LU one-shot wrappers reject reused matrix state more
explicitly, while preserving the explicit `analysis` / `factors` lifecycle as
the canonical repeated-run path.

## Landed Scope

### Touched files

- `include/sparse_lu.h`
- `src/sparse_lu.c`
- `tests/test_integration.c`

### Untouched by design

- `src/sparse_analysis.c`
- `src/sparse_cholesky.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt.c`
- `src/sparse_qr.c`
- `tests/test_sparse_lu.c`
- broad docs/examples/benchmark surfaces

## Main Implementation Result

### 1. LU one-shot wrappers now reject reused row/column state up front

Both:

- `sparse_lu_factor(...)`
- `sparse_lu_factor_opts(...)`

now call `sparse_matrix_require_original_row_col_state(mat)` before entering
the one-shot factor path.

That means LU one-shot wrappers now reject matrices that have already been:

- reordered
- pivoted
- factored

instead of attempting to re-enter the one-shot wrapper flow on an old factor
container.

### 2. The public LU contract is clearer without widening the lifecycle API

`include/sparse_lu.h` now states more directly that:

- LU one-shot entry points should be called on a fresh matrix or a
  `sparse_copy(...)`
- stable-pattern repeated runs belong on the explicit public lifecycle in
  `sparse_analysis.h`
- reordered/default-compatible LU calls may internally reuse shared lifecycle
  plumbing, but they still keep the same public one-shot matrix-state
  contract

### 3. The old reordered LU factor is now preserved when a caller makes a mistaken second one-shot call

The Day 6 regression proof in `tests/test_integration.c` now verifies:

- reordered LU factorization succeeds
- a second LU one-shot call on that same matrix is rejected with
  `SPARSE_ERR_BADARG`
- the previously built factor remains valid and solves identically before and
  after the rejected call

This keeps the hardening useful to callers without blurring away the
one-shot-versus-lifecycle distinction.

## Compatibility and Safety Notes

### Preserved compatibility

- explicit repeated-run direct solves remain the public:
  - `sparse_analyze()`
  - `sparse_factor_numeric()`
  - `sparse_factor_solve()`
  - `sparse_refactor_numeric()`
  lifecycle
- LU one-shot wrappers remain first-class/default peer entry points
- no hidden copy semantics were introduced

### Important Day 6 detail

The first implementation cut flattened all precondition failures to
`SPARSE_ERR_BADARG`, which incorrectly changed the `NULL` path.

The landed fix preserves the original error split:

- `NULL` still returns `SPARSE_ERR_NULL`
- reused row/column state returns `SPARSE_ERR_BADARG`

## Validation

### Required code-day gate

- `make format`
- `make lint`
- `make test`

All passed.

### Stronger reviewed baseline

- `make quality-review-full`

Passed.

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 222.54 sec`

### Non-blocking note

The reviewed CMake rebuild again emitted ordinary compiler warnings while
rebuilding `bench_eigs_reuse`, but the full reviewed path still completed
cleanly and passed all parity gates.

## Day 6 Exit State

Sprint 62 now has one coherent first hardening batch:

- LU one-shot wrappers reject reused matrix state explicitly
- the public LU contract is clearer about fresh-matrix use versus explicit
  repeated-run lifecycle use
- the highest-value Day 6 integration proof now covers both rejection and
  old-factor preservation
- the batch stayed inside the exact Day 5 touched-file fence
- the landed state passed both the required code-day gate and the full
  reviewed baseline
