# Sprint 42 Day 12 Artifact: Focused Lifecycle Misuse Tests

## Scope

Day 12 implements the bounded focused-test batch defined on Day 11:

- analyze-once misuse tightening
- QR copy-before-use misuse tightening
- SVD copy-before-use misuse tightening

This is a regression-hardening batch. It does not change public API shape or
expand Sprint 42 into broader lifecycle redesign work.

## Landed Test Coverage

### 1. Analyze-once misuse tightening

Target file:

- `tests/test_etree.c`

Added coverage:

- `sparse_analyze(...)` rejects an already-factored matrix with
  `SPARSE_ERR_BADARG`
- `sparse_factor_numeric(...)` rejects a matrix whose row/column permutation
  state is no longer the original identity view

Why this matters:

- these are the main preconditions behind the analyze-once compatibility bridge
- Sprint 42 introduced shared original-state guard helpers and private factor
  seams; Day 12 now pins the caller-facing misuse contract directly

Implementation notes:

- the factored rejection path uses a real Cholesky-factored copy
- the non-identity-state rejection path uses a copied matrix with explicitly
  non-identity row/column permutation state
- the clean original matrix is still used to prove the normal success path

## 2. QR copy-before-use misuse tightening

Target file:

- `tests/test_qr.c`

Added coverage:

- a LU-factored matrix copy is rejected by `sparse_qr_factor(...)` with
  `SPARSE_ERR_BADARG`
- the untouched original matrix still factors successfully via QR

Why this matters:

- QR already documented that it requires the original physical row/column view
- Sprint 42’s compatibility story still depends on the caller using
  `sparse_copy(...)` before matrix-mutating workflows
- Day 12 now enforces that contract with a direct regression instead of leaving
  it only in comments and documentation

## 3. SVD copy-before-use misuse tightening

Target file:

- `tests/test_svd.c`

Added coverage:

- a LU-factored matrix copy is rejected by `sparse_svd_compute(...)`
- the same reused factored copy is rejected by `sparse_svd_partial(...)`
- the untouched original matrix still succeeds through full SVD

Why this matters:

- SVD shares the same original-state requirement as QR through the
  bidiagonalization path
- Day 12 now covers both the full and partial SVD entry points that depend on
  that contract

## Compatibility Meaning

The Day 12 regressions make Sprint 42’s compatibility rule explicit:

- if a caller wants to preserve an original coefficient matrix for later QR,
  SVD, analyze-once, or related original-state-sensitive use
- then the caller should factor or mutate a fresh `sparse_copy(...)`
- not the same matrix object they later want to reuse as an untouched input

This is still the current compatibility story for the direct matrix-mutating
families. Sprint 42 hardens it; it does not replace it yet.

## What Day 12 Deliberately Avoided

This batch did **not** widen into:

- public-handle API tests for future interfaces
- broad lifecycle-test framework work
- benchmark/example misuse expansion
- wider README/tutorial rewrites
- broader cancellation redesign outside the already-touched Day 10 seams

The right result for Day 12 was a small contract-focused regression batch, and
that is what landed.

## Validation

Because `*.c` changed, I ran the required full gate:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

## Day 12 Outcome

Sprint 42 now has direct lifecycle misuse coverage for the highest-value
remaining compatibility-facing seams:

- analyze-once misuse
- QR reused-matrix misuse
- SVD reused-matrix misuse

That closes the Day 11 focused-test design without expanding Sprint 42 beyond
its internal-handle and compatibility-scaffolding mandate.
