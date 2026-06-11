# Sprint 63 Day 10: Large-n CSC Cholesky Lifecycle Semantics Batch

Date: 2026-06-10
Branch: `sprint-63`

## Purpose

Land the bounded shared direct lifecycle semantics slice that Sprint 63 Day 9
fixed as the remaining highest-value queue item:

- large-`n` CSC-backed Cholesky refactor failure should preserve the old usable
  factors on the public repeated-run direct lifecycle path
- gross structure drift on that same lane should reject cleanly without
  destroying the existing usable factors

## Landed Surfaces

Implementation:

- `src/sparse_chol_csc_supernodal.c`

Proof:

- `tests/test_integration.c`

## Main Result

Sprint 63 now has explicit large-`n` CSC-backed public lifecycle proof for the
two missing refactor-failure retention cases:

1. same-pattern but no-longer-SPD retry preserves the old usable factors
2. gross nnz drift rejects with `SPARSE_ERR_BADARG` and still preserves the old
   usable factors

The batch stayed inside the Day 9 fence:

- no `sparse_analysis` API redesign
- no benchmark or example widening
- no LDL^T or QR spillover

## Implementation Follow-Through

The only implementation change needed was one small supernodal CSC guard in
`src/sparse_chol_csc_supernodal.c`.

The landed path now checks each non-empty column's stored diagonal before
supernode dispatch begins and returns `SPARSE_ERR_NOT_SPD` when that diagonal
is already non-positive.

That keeps the supernodal CSC path aligned with the scalar CSC path on the
simplest SPD contract instead of letting an already-invalid stored diagonal
proceed deeper into batched elimination.

## New Public Lifecycle Proof

`tests/test_integration.c` now proves two large-`n` CSC-backed Cholesky public
lifecycle contracts at `n = 120`:

- `test_public_lifecycle_cholesky_csc_refactor_preserves_old_factors_on_failure`
- `test_public_lifecycle_cholesky_csc_refactor_rejects_nnz_drift_and_preserves_old_factors`

The non-SPD case now uses the clearest possible trigger:

- set `A_bad(0, 0) = -1.0`

The nnz-drift case removes both symmetric off-diagonal entries:

- `A_bad(0, 1) = 0.0`
- `A_bad(1, 0) = 0.0`

Both tests prove the same retention rule:

- the failing refactor returns the expected error
- the previously built CSC-backed factors remain usable
- a later solve still matches the baseline solution

## Validation

Ran:

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
- `Total Test time (real) = 359.66 sec`

## Non-Blocking Note

The reviewed CMake rebuild again emitted the ordinary
`bench_eigs_reuse.c` double-promotion warnings while rebuilding that bench
binary, but the reviewed path still completed cleanly and passed all parity
gates.

## Exit State

Sprint 63 Day 10 now hands off a much smaller remaining queue:

- the shared direct lifecycle now has explicit large-`n` CSC-backed Cholesky
  failure-preserve proof
- the missing Sprint 63 semantics work is no longer broad lifecycle design
- the next work can stay focused on compatibility and final follow-through
