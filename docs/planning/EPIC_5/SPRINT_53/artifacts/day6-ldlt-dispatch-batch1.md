# Sprint 53 Day 6: LDL^T Dispatch Batch I

## Purpose

Day 6 makes the LDL^T CSC-vs-linked-list dispatch contract easier to reason
about on the highest-value public path without redesigning the direct-solver
surface. The goal is to centralize selected-backend reasoning, share one
selected-path execution seam across reorder and no-reorder flow, clarify the
public `used_csc_path` wording, and add direct proof that selected-path
telemetry still survives later wrapper validation failure.

## Main Day 6 Result

Day 6 extracted two small helpers in `src/sparse_ldlt.c`:

- `ldlt_dispatch_select_backend(...)`
- `ldlt_factor_selected_backend(...)`

Together they now own:

1. backend selection for `AUTO`, forced linked-list, forced CSC, and the
   `n == 0` empty-matrix exception
2. actual execution of the selected numeric path
3. one shared execution contract across reorder and no-reorder LDL^T wrapper
   flow

This complements the Day 4-5 indefinite CSC work instead of overlapping it:

- Day 4-5 reduced duplicated CSC preparation/completion orchestration
- Day 6 reduces duplicated wrapper-level dispatch reasoning and path
  publication

## Code Changes

### 1. Centralized LDL^T backend selection

`src/sparse_ldlt.c` now implements:

- `ldlt_dispatch_select_backend(...)`

That helper validates the backend enum and resolves:

- `SPARSE_LDLT_BACKEND_LINKED_LIST` -> linked-list
- `SPARSE_LDLT_BACKEND_CSC` -> CSC
- `SPARSE_LDLT_BACKEND_AUTO` -> threshold-based choice
- `n == 0` -> forced linked-list regardless of requested CSC

The empty-matrix exception stays explicit because the CSC scalar pre-pass has
no meaningful empty input to factor.

### 2. Shared selected-backend execution seam

`src/sparse_ldlt.c` now also implements:

- `ldlt_factor_selected_backend(...)`

That helper executes the already-selected path through:

- `ldlt_factor_csc_path(...)` for CSC
- `ldlt_factor_internal(...)` for linked-list

`sparse_ldlt_factor_opts(...)` now uses that same helper in both:

- the reordered/permuted matrix branch
- the no-reorder direct branch

This keeps the selected-backend contract consistent after selection, instead
of repeating similar execution branching in multiple places.

### 3. Public backend-telemetry wording now matches the actual contract better

`include/sparse_ldlt.h` now clarifies two important points:

1. forced `SPARSE_LDLT_BACKEND_CSC` still falls back to linked-list on the
   `n == 0` empty-matrix edge case
2. `used_csc_path` reports the actual selected numeric path, not just the
   caller-requested backend enum

That makes the header more honest about what callers should observe on
success or failure.

## Regression Proof Added

Day 6 added focused public proof in `tests/test_ldlt.c`:

- `test_ldlt_backend_csc_reports_selected_path_before_reorder_error`

The test forces the CSC backend, provides `used_csc_path`, and then triggers a
later invalid reorder enum. The proof checks that:

1. the wrapper still returns `SPARSE_ERR_BADARG`
2. the selected-path telemetry was already published as CSC (`used_csc_path=1`)

This is a better signal than a generic success-path test because it proves the
telemetry survives later wrapper validation failure.

## Preserved Contract

Day 6 intentionally preserved the bounded Sprint 50-53 semantics:

- one-shot LDL^T remains first-class
- repeated direct runs remain analysis/factors-centric
- the scalar BK pre-pass remains the authoritative indefinite permutation
  resolution step
- no raw CSC/native storage is exposed publicly
- no new generic direct handle is introduced

This is dispatch and telemetry cleanup, not a public lifecycle redesign.

## Validation

Because `*.c` / `*.h` changed, Day 6 ran the full required gate:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 120.16 sec`

Day 6 also ran the touched follow-ons justified by the batch:

- `./build/test_ldlt`
- `./build/test_sprint20_integration`
- `./build/test_integration`
- `./build/example_analysis`

Representative direct results:

- `test_ldlt` = `84 / 84`
- `test_sprint20_integration` = `20 / 20`
- `test_integration` = `35 / 35`
- `example_analysis` residual = `4.44e-16`

## What Day 6 Solved

- centralized backend selection for the LDL^T public wrapper
- shared selected-backend execution across reorder and no-reorder flow
- clarified that `used_csc_path` reports the actual selected numeric path
- added direct proof that selected-path telemetry survives later reorder
  validation failure

## What Day 6 Did Not Solve

- LDL^T-specific factor-many benchmark proof is still later work
- broader Cholesky/LDL^T dispatch reconciliation is still later work
- the scalar BK pre-pass still remains the authoritative indefinite
  permutation-resolution step

## Operational Result

Sprint 53 now has a cleaner LDL^T dispatch base before the later
reconciliation and benchmark days:

- one named backend selector
- one named selected-backend execution seam
- more honest public backend telemetry wording
- stronger direct proof around the dispatch contract
