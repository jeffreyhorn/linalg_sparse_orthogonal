# Sprint 62 Day 14: Closeout and Handoff

Date: 2026-06-10
Branch: `sprint-62`

## Purpose

Package Sprint 62 into one clean validated direct-usability handoff for the
next Epic 6 implementation sprint.

## Main Result

Sprint 62 closes as one coherent direct-solver usability and lifecycle package
rather than a set of separate family-local patches:

- LU one-shot wrappers now reject reused row/column state up front
- reordered LU one-shot attempts preserve the caller matrix on cancel/failure
  by factoring on a temporary reordered working copy and publishing back only
  on success
- reordered Cholesky one-shot attempts now follow the same preserved-caller
  rule
- direct-family docs, tutorial guidance, example adoption notes, and
  maintainer guidance now match the shipped LU/Cholesky behavior more directly

## Preserved Compatibility Fence

Sprint 62 reduced surprise without blurring the public direct-workflow
boundary:

- one-shot direct wrappers remain first-class/default peer entry points
- the explicit repeated-run direct lifecycle remains the canonical reuse path:
  - `sparse_analyze()`
  - `sparse_factor_numeric()`
  - `sparse_factor_solve()`
  - `sparse_refactor_numeric()`
- reordered LU and reordered Cholesky now preserve the caller matrix on
  cancel/failure
- no-reorder linked-list Cholesky cancellation remains on its existing
  compatibility lane
- LDL^T remains a cleaner separate-owner surface and did not need forced
  widening just to make the sprint look symmetrical

## Deferred Queue

The remaining direct-usability queue after Sprint 62 is explicit and bounded:

- no-reorder linked-list Cholesky cancellation restoration remains deferred
- broader LDL^T wording or compatibility follow-through only if a later
  contradiction appears
- QR remains mainly a comparison/deferred surface
- broader direct-family docs/examples simplification outside the touched
  high-signal surfaces remains future work
- deeper direct-lifecycle uniformity and CSC/LU follow-through belongs to
  Sprint 63

## Validated Baseline

Sprint 62 closes from the Day 13 validated baseline:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 377.56 sec`

Representative retained proof/output signals:

- `./build/test_integration` -> `43 / 43`
- `./build/test_cholesky` -> `21 / 21`
- `./build/test_ldlt` -> `84 / 84`
- `./build/test_sparse_lu` -> `37 / 37`
- `example_analysis` residual = `4.44e-16`
- `example_basic_solve` residual = `0.00e+00`
- `example_ldlt` relative residual = `1.555e-16`
- `bench_refactor`: `tridiag-200 1.47x`, `tridiag-500 1.25x`,
  `bcsstk04 1.30x`, `nos4 1.41x`
- `bench_refactor_csc nos4`: `speedup_refactor = 0.61x`, residuals
  `8.24e-16` / `7.06e-16`

## Project-Plan Check

Re-reading the Sprint 62 section of
`docs/planning/EPIC_6/PROJECT_PLAN.md` showed the landed branch still matches
the intended scope. No Sprint 62 correction is needed.

## Exit State

Sprint 62 exits with:

- a cleaner and more explicit one-shot direct usability story
- stronger preservation semantics on the highest-value reordered LU/Cholesky
  paths
- a clearer workflow split between one-shot direct solves and the explicit
  repeated-run direct lifecycle
- an explicit deferred queue for Sprint 63 and later Epic 6 follow-through
