# Sprint 84 Day 5: Oracle / Property / Failure-Path Architecture Design

## Purpose

Define the bounded assurance contract that Sprint 84 will actually land on the
first maintained direct-family external differential lane.

## Main Result

Sprint 84 now has one explicit first implementation contract:

- required implementation center:
  - `tests/test_chol_csc.c`
- support only if the first batch truly forces it:
  - `tests/test_chol_csc_supernodal_helpers.h`
  - `tests/test_framework.h`
  - `tests/test_fuzz.c`
  - `tests/test_integration.c`
  - `tests/test_ldlt.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `README.md`
  - `docs/maintainer_guide.md`

## Ownership Split

The Day 5 ownership split is now fixed:

- maintained external differential harness owner:
  - `tests/test_chol_csc.c`
- deterministic seeded property owner, but not in the first batch:
  - `tests/test_fuzz.c`
- public failure-path, cancellation, and lifecycle-preservation owner, but not
  in the first batch unless forced:
  - `tests/test_integration.c`
- direct-family support comparison owner if the first batch truly forces a
  second family-local seam:
  - `tests/test_ldlt.c`
- iterative/eigensolver retained proof owners, but not first-batch adoption
  owners:
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
- support-surface wording owners only if implementation truly moves the public
  assurance reading:
  - `README.md`
  - `docs/maintainer_guide.md`

## Strongest Clarification

The useful Day 5 clarification is explicit now:

- the first landing should preserve the Sprint 80 oracle fence by keeping the
  maintained external differential lane bounded to the direct-family SPD
  Cholesky path
- it should keep the first maintained comparison test-owned, fixture-backed,
  and family-local inside `tests/test_chol_csc.c`
- it should keep deterministic seeded property coverage in `tests/test_fuzz.c`
  as a separate follow-through seam rather than collapsing that work into the
  first external differential batch
- it should keep cancellation, lifecycle-preservation, and error-path proof
  centered in `tests/test_integration.c` unless the first batch exposes one
  truly local Cholesky-only contradiction
- it should keep iterative/eigs proof owners unchanged in the first batch
  rather than inflating the first maintained external lane into a repo-wide
  adoption claim
- it should not turn benchmarks or examples into correctness owners

## Preserved First-Batch Fence

The preserved first-batch fence is explicit:

- no mandatory heavyweight external stack for normal builds
- no repo-wide claim that every solver now has maintained external proof
- no seeded-property widening folded into the first batch unless the direct
  differential contract truly forces it
- no failure-path/package/platform churn detached from a real landed
  comparison seam
- no benchmark/reporting drift into oracle ownership
- no reopening Sprint 83 capability-surface work

## Exit State

- Sprint 84 now has one bounded oracle/property/failure-path architecture
  contract.
- Ownership between the first maintained direct-family external harness, the
  seeded-property lane, and the public failure-path lane is fixed before Day 6
  begins.
- Later iterative/eigs adoption and broader support/dependency spillover
  remain explicitly outside the first batch.
