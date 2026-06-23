# Sprint 84 Day 11: Failure-Path Numerical Proof Batch

## Purpose

Land the bounded failure-path numerical proof batch on the shared public
lifecycle owner.

## Main Result

Sprint 84 Day 11 landed one bounded failure-path proof batch:

- required implementation center:
  - `tests/test_integration.c`
- strongest support-only follow-through that was not needed:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `docs/maintainer_guide.md`

## Landed Surface

The landed retry-after-failure proof widening is explicit now:

- `test_public_lifecycle_refactor_failure_allows_retry`
- `test_public_lifecycle_cholesky_csc_refactor_failure_allows_retry`
- `test_public_lifecycle_ldlt_refactor_failure_allows_retry_amd`

This is a proof-surface widening batch, not a production algorithm batch. It
keeps the Day 11 move:

- test-owned
- centered entirely in the shared public lifecycle proof owner
- bounded to failure-path preservation and retry semantics
- explicit about preserved old-factor usability before retry
- explicit about successful later good same-pattern retry on the same public
  `analysis` / `factors` objects

## Proof Widening

The new linked-list Cholesky retry proof shows:

- a successful initial factor / solve establishes a valid baseline state
- a later refactor on a same-pattern but non-SPD matrix fails with
  `SPARSE_ERR_NOT_SPD`
- that failure preserves the previously valid factor state
- callers can still solve the original system correctly from the preserved
  state
- callers can then refactor a later good same-pattern matrix and recover the
  correct new solution without rebuilding the public lifecycle objects

The new CSC Cholesky retry proof establishes the same retained lifecycle
contract on the forced CSC-backed lane above the dispatch threshold.

The new AMD LDL^T retry proof covers the indefinite public lifecycle lane:

- a rejected nnz-drift refactor failure preserves the previous factor state
- the preserved state still solves the original KKT-like system correctly
- a later good same-pattern retry on the same public objects succeeds
- the retained LDL^T permutation / pivot metadata remains present after the
  successful retry

Together with the pre-existing Day 10 owner surface, `tests/test_integration.c`
now proves:

- callback cancellation and short-circuit behavior
- original-matrix preservation on cancellation and invalid-option failures
- zeroed-state and mismatched-state rejection behavior
- preserved-old-factor behavior after refactor failure
- successful later retry-after-failure behavior on retained shared public
  lifecycle direct-family lanes

## Strongest Clarification

The useful Day 11 clarification is explicit:

- Sprint 84 no longer has a gap around retry-after-failure lifecycle proof on
  the shared public refactor path
- the strongest remaining assurance seam is no longer "can callers recover on
  the same lifecycle objects after a failed refactor"
- family-local direct proof owners did not need to move
- maintainer/support wording did not need to move
- later policy / CI alignment remains separate work

## Validation

The Day 11 batch was validated with:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`

Representative retained outputs:

- `test_integration` = `56 / 56`
- reviewed CMake `Total Test time (real)` = `512.76 sec`
- reviewed CMake `test_reorder_nd` remained the dominant runtime anchor at
  `366.43 sec`

## Exit State

- Sprint 84 now has one landed bounded failure-path numerical proof batch.
- The retained shared public lifecycle owner proves preserved-old-factor solve
  behavior and successful later retry after failed refactor on linked-list
  Cholesky, CSC Cholesky, and AMD LDL^T lanes.
- Later sprint work can stay focused on final proof alignment, full validation
  closeout, and subsequent policy / CI / support-surface work only if bounded
  evidence justifies movement.
