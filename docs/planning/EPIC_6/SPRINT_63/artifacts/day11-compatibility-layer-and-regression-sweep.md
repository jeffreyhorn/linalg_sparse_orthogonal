# Sprint 63 Day 11: Compatibility Layer and Regression Sweep

Date: 2026-06-10
Branch: `sprint-63`

## Purpose

Tighten the post-Day-10 direct-lifecycle and CSC compatibility surface without
reopening the implementation batch:

- add the missing family-local CSC regression for the new supernodal
  early-rejection guard
- remove stale public header wording on the touched LU and Cholesky entry
  points
- close the sweep from the strongest reviewed baseline

## Landed Surfaces

Public headers:

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`

Proof:

- `tests/test_chol_csc.c`

## Main Result

Sprint 63 Day 11 closes the remaining post-Day-10 compatibility gap with one
bounded CSC regression and two small header-truthfulness follow-through edits.

The batch stayed inside the Day 11 fence:

- no implementation widening
- no `sparse_analysis` lifecycle redesign
- no docs/example/benchmark widening
- no LDL^T or QR spillover

## Header Truthfulness Follow-Through

The touched LU and Cholesky headers now state the shipped early-rejection
contract directly.

`sparse_lu_factor_opts(...)` now says that invalid pivot or reorder enums are
rejected before reorder or factor mutation begins.

`sparse_cholesky_factor_opts(...)` now says that invalid reorder or backend
enums are rejected before reorder or factor mutation begins.

That aligns the family-local comments with the landed Day 6-Day 10 behavior
instead of forcing callers to infer the safety property from tests or
implementation shape.

## New CSC Regression

`tests/test_chol_csc.c` now adds:

- `test_eliminate_supernodal_rejects_nonpositive_stored_diagonal`

The regression builds a small CSC input with a stored negative diagonal and
proves:

- `chol_csc_eliminate_supernodal(...)` returns `SPARSE_ERR_NOT_SPD`
- the stored diagonal entry remains unchanged at the point of rejection

This is the missing family-local counterpart to the Day 10 public lifecycle
proof:

- the CSC supernodal early-rejection guard is now tested directly
- the rejection path is explicitly proven to fail before downstream supernodal
  mutation work begins

## Validation

Ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `./build/test_chol_csc`
- `./build/test_integration`

Result:

- all passed

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 396.79 sec`

Focused retained proof points:

- `test_chol_csc` passed `140 / 140`
- `test_integration` passed `47 / 47`
- the new Day 11 regression passed on the family-local CSC proof surface:
  - `test_eliminate_supernodal_rejects_nonpositive_stored_diagonal`

## Non-Blocking Note

The reviewed CMake rebuild again emitted the existing
`bench_eigs_reuse.c` double-promotion warnings while rebuilding that bench
binary, but the reviewed path still completed cleanly and passed all parity
gates.

## Exit State

Sprint 63 Day 11 now hands off a smaller final queue:

- lifecycle and CSC compatibility proof is tightened on the highest-signal
  direct proof homes
- touched public header wording now matches the shipped implementation
- the sprint can move into bounded docs/example/benchmark follow-through
  without reopening implementation work
