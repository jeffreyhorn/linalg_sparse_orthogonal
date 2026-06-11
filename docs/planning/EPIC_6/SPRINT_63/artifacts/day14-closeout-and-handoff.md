# Sprint 63 Day 14: Closeout and Handoff

Date: 2026-06-10
Branch: `sprint-63`

## Purpose

Package Sprint 63 into a clean validated handoff:

- summarize the landed LU and CSC lifecycle outcomes
- record the preserved compatibility fence
- make the deferred queue explicit
- confirm whether the Sprint 63 Epic 6 plan section needs correction

## Main Result

Sprint 63 now hands off one coherent direct-lifecycle uniformity package
across:

- LU lifecycle follow-through
- Cholesky CSC lifecycle follow-through
- shared repeated-run direct failure-preserve proof on the large-`n`
  CSC-backed Cholesky lane
- compatibility/regression tightening
- docs/example/benchmark/maintainer follow-through
- validated Day 13 close

## Landed Outcomes

### LU

- invalid pivot and reorder enums reject before reorder or factor mutation
  begins
- rejected one-shot LU reuse preserves the previously usable factor
- reordered one-shot LU cancel/failure preserves the caller-owned matrix

### Cholesky / CSC

- invalid reorder and backend enums reject before reorder or factor mutation
  begins
- reordered one-shot Cholesky cancel/failure preserves the caller-owned matrix
- CSC dispatch/publication behavior is more uniform on the touched wrapper path
- the large-`n` CSC-backed repeated-run direct lane now explicitly preserves
  old usable factors on:
  - same-pattern non-SPD refactor failure
  - obvious nnz drift rejection
- the CSC supernodal path now has direct family-local proof for early
  rejection on a stored non-positive diagonal

## Preserved Compatibility Fence

Still true after Sprint 63:

- one-shot direct wrappers remain first-class/default peer entry points
- the explicit repeated-run direct lifecycle remains:
  - `sparse_analyze()`
  - `sparse_factor_numeric()`
  - `sparse_factor_solve()`
  - `sparse_refactor_numeric()`
- repeated direct reuse preserves symbolic/permutation setup, not stale numeric
  factor contents
- failed `sparse_refactor_numeric(...)` calls preserve previous usable factors
  on the public repeated-run direct path
- reordered LU and reordered Cholesky preserve the caller-owned matrix on
  cancel/failure through temporary reordered working copies
- no-reorder linked-list Cholesky cancellation remains intentionally
  non-bit-identical
- LDL^T remains a cleaner separate-owner surface and was not widened just to
  force fake family symmetry

## Validated Baseline

Sprint 63 closes from the Day 13 validated state:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 348.10 sec`

## Deferred Queue

Explicit remaining queue after Sprint 63:

- no-reorder linked-list Cholesky bit-identical cancellation restoration
- CSC progress-callback parity for Cholesky / LDL^T
- broader LDL^T or QR wording follow-through only if a later contradiction
  appears
- later direct-family docs/examples density cleanup outside the bounded
  Sprint 63 surfaces
- later direct-lifecycle/productization work above this now-stabilized LU/CSC
  base

## Project Plan Check

Re-reading `docs/planning/EPIC_6/PROJECT_PLAN.md` showed no Sprint 63
contradiction that requires a plan correction.

The sprint landed inside the intended Epic 6 lane:

- LU lifecycle follow-through
- CSC repeated-run uniformity
- solve/refactor semantics alignment
- validation and closeout

## Exit State

Sprint 63 now hands off a stable validated direct-lifecycle base:

- the highest-value LU and CSC lifecycle seams are closed
- the real family-local compatibility fence is explicit
- the remaining queue is bounded and future-facing rather than ambiguous
