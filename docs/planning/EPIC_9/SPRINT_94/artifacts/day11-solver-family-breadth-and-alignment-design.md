# Sprint 94 Day 11 - Solver-Family Breadth and Alignment Design

## Scope
- Assess the validated Day 7 scalar landing and Day 10 index/ABI landing
  against the remaining Sprint 94 capability contract
- Decide whether any further solver-family implementation widening is still
  required
- Freeze the exact Day 12 center and directly forced support surfaces

## Main Result

The validated post-Day-10 baseline does not justify another solver-family
implementation landing.

The bounded Sprint 94 capability claim is already supported on the touched
highest-value lanes:

- the shared matrix-shell scalar seam is now real in both the public helper
  contract and the storage/build implementation owner
- the touched matrix-shell save/load and diagnostic seam now reads as more
  trustworthy under the existing `idx_t` width contract
- the iterative, eigensolver, and QR public scalar seams were already real and
  proof-owned before this design pass

That means the strongest remaining contradiction is support-surface
interpretation, not missing implementation.

## Exact Day 12 Center

- required Day 12 center:
  - `docs/maintainer_guide.md`

- directly forced support-only follow-through only if the wording truly needs
  it:
  - `README.md`

- explicitly not reopened unless a wording contradiction forces movement:
  - solver-family implementation owners
  - proof binaries and focused capability tests
  - package/install/export surfaces

## Why No Further Solver-Family Landing Is Needed

- `include/sparse_types.h` already states the shipped scalar contract remains
  real-only while identifying the touched widened scalar seam truthfully
- `include/sparse_matrix.h` plus `src/sparse_matrix.c` now make the matrix-
  shell scalar contract real on the highest-value shared owner
- `include/sparse_iterative.h`, `include/sparse_eigs.h`, and
  `include/sparse_qr.h` already expose bounded public scalar seams backed by
  focused proof owners
- widening dense or SVD owners now would broaden the Sprint 94 claim beyond
  the bounded scalar/index contract and into later deferred capability work

## Day 12 Alignment Intent

Day 12 should align support wording with the validated landing set:

- maintainers should read the bounded scalar claim as:
  - matrix-shell storage/build seam real
  - matrix-shell diagnostic/load/save width maturity real
  - iterative/eigs/QR public scalar seams real
- support wording should still keep these explicit non-claims:
  - no broad complex support
  - no broad mixed-precision maturity
  - no dense/SVD family-wide scalar widening claim
  - no broader package or platform capability reinterpretation

## Result

- Sprint 94's final code implementation batch is retired by evidence
- Day 12 is fixed to a support-only alignment pass
- the sprint stays inside the bounded capability contract without drifting into
  generic solver-family expansion
