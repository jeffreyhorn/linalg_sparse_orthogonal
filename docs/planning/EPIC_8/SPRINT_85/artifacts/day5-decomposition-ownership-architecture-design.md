# Sprint 85 Day 5: Decomposition / Ownership Architecture Design

## Purpose

Define the bounded extraction and ownership contract Sprint 85 will actually
land on the first iterative-source cleanup lane.

## Main Result

Sprint 85 now has one explicit first implementation contract:

- required implementation center:
  - `src/sparse_iterative.c`
- support only if the first batch truly forces it:
  - `tests/test_iterative.c`
  - `tests/test_iterative_handle_helpers.h`
  - `tests/test_integration.c`
  - `docs/maintainer_guide.md`
  - `README.md`

## Ownership Split

The Day 5 ownership split is now fixed:

- iterative-source extraction owner:
  - `src/sparse_iterative.c`
- retained iterative reviewed proof owner after extraction:
  - `tests/test_iterative.c`
- retained iterative test-helper owner only if extraction truly forces proof
  helper movement:
  - `tests/test_iterative_handle_helpers.h`
- shared lifecycle/public repeated-run owner, but not first-batch adoption
  owner unless the extraction exposes a true lifecycle contradiction:
  - `tests/test_integration.c`
- strongest next source-cleanup owner, but not first-batch owner:
  - `src/sparse_chol_csc.c`
- support-surface wording owners only if implementation truly changes the
  maintainer rerun or owner reading:
  - `docs/maintainer_guide.md`
  - `README.md`

## Strongest Clarification

The useful Day 5 clarification is explicit now:

- the first landing should keep the cleanup source-owned inside
  `src/sparse_iterative.c`
- it should reduce local mixed-responsibility concentration by extracting one
  bounded internal helper seam rather than distributing logic across many new
  owners
- it should preserve `tests/test_iterative.c` as the reviewed proof owner
  instead of turning the first cleanup into a test-architecture rewrite
- it should keep `tests/test_integration.c` as a follow-through lifecycle
  owner only if the source extraction actually changes repeated-run or handle
  semantics
- it should keep direct-family source and giant-test cleanup explicitly later,
  not folded into the first iterative lane
- it should not widen into public-header, package/runtime, benchmark, or
  example ownership changes

## Preserved First-Batch Fence

The preserved first-batch fence is explicit:

- no broad algorithm rewriting detached from ownership cleanup
- no proof-owner diffusion by moving helpers without preserving owners
- no repo-wide “cleanup sweep” claim
- no public-header or package/runtime churn detached from the landed source
  seam
- no benchmark/reporting or example drift into correctness ownership
- no reopening Sprint 84 assurance widening as part of the first Sprint 85
  cleanup batch

## Exit State

- Sprint 85 now has one bounded iterative extraction/decomposition contract.
- Ownership between the first iterative source lane, retained proof owners,
  and later direct-family/test follow-through is fixed before Day 6 begins.
- Public-header, runtime, benchmark, example, and broader source/test
  spillover remain explicitly outside the first batch.
