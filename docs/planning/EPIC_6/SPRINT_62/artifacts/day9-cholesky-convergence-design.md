# Sprint 62 Day 9: Cholesky Lifecycle/Wrapper Convergence Design

Date: 2026-06-10
Branch: `sprint-62`

## Purpose

Turn the Day 8 post-LU audit into one exact Day 10 Cholesky implementation
fence, with explicit compatibility rules for what Sprint 62 will strengthen
and what remains deferred.

## Main Design Result

### 1. Day 10 should move only the reordered Cholesky publication seam

The strongest remaining Cholesky usability problem is specific:

- reordered one-shot Cholesky currently publishes the permuted working state
  onto the caller-owned matrix before factorization success is known

That is the part worth moving in Sprint 62.

The broader no-reorder linked-list cancel model is explicitly not the Day 10
target:

- cancel-at-step-0 still strips the upper triangle before the first callback
- restoring full bit-identity there would be a broader compatibility change

So the Day 10 batch should strengthen reordered preservation, not redesign the
entire Cholesky cancel contract.

### 2. The Day 10 touched-file fence is exact

Required:

- `include/sparse_cholesky.h`
- `src/sparse_cholesky.c`
- `tests/test_integration.c`

Optional only if proof burden forces it:

- `tests/test_cholesky.c`

Explicitly deferred:

- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt.c`
- `src/sparse_qr.c`
- broad docs/example/benchmark surfaces

### 3. The implementation model should mirror LU Day 7 only where justified

The justified Cholesky carry-forward from LU Day 7 is:

- use a temporary reordered working copy for reordered one-shot attempts
- publish the reordered/factored payload back to the caller matrix only after
  success

The unjustified carry-forward is:

- trying to erase all in-place mutation from Cholesky
- trying to make every cancellation path bit-identical
- widening into the shared analyze/factor/refactor API

## Public/Internal Contract

### Public/front-door rule after Day 10

`include/sparse_cholesky.h` should state more directly that:

- one-shot Cholesky still belongs on a fresh matrix or `sparse_copy()`
- stable-pattern repeated runs still belong on the explicit shared direct
  lifecycle
- reordered one-shot attempts may factor a temporary reordered working copy
  and publish back only on success

### Internal rule after Day 10

`src/sparse_cholesky.c` should:

- keep backend selection inside the existing Cholesky wrapper
- avoid mutating the caller-owned matrix into reordered form until the chosen
  factor path succeeds
- preserve the existing no-reorder linked-list cancellation semantics

## Regression Obligation

The primary Day 10 proof should live in `tests/test_integration.c` and cover:

- cancelled reordered Cholesky one-shot attempt preserves original caller
  matrix state
- the cancelled matrix is still rejected as unfactored
- a later reordered Cholesky one-shot retry succeeds on the same original
  caller matrix

If family-local detail makes that insufficient, add the smallest necessary
support proof in `tests/test_cholesky.c`, but do not default to widening the
proof surface.

## Preserved Compatibility

- one-shot Cholesky remains a first-class/default entry point
- explicit repeated-run direct solves remain on the shared direct lifecycle
- no-reorder in-place mutation remains part of the Cholesky one-shot identity
- no-reorder linked-list cancel-at-step-0 remains non-bit-identical and
  documented as such

## Explicitly Deferred

- no-reorder linked-list cancel bit-identity restoration
- CSC progress callback parity for Cholesky
- LDL^T or QR code-path convergence work
- broad direct-family docs/examples simplification
- any hidden-copy semantics that erase one-shot mutation globally

## Day 9 Exit State

Sprint 62 now has one exact next code target:

- a bounded reordered Cholesky preservation hardening slice
- a tight touched-file fence
- explicit proof ownership in `tests/test_integration.c`
- explicit compatibility rules for what Sprint 62 still will not change
