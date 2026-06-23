# Sprint 84 Day 10: Failure-Path Numerical Proof Design

## Purpose

Define the bounded cancellation, error-path, and stress-fixture proof contract
for the most fragile retained public lifecycle guarantees after the Day 9
property-expansion landing.

## Main Result

Sprint 84 now has one explicit third implementation contract:

- required Day 11 center:
  - `tests/test_integration.c`
- strongest support-only follow-through if the failure-path batch truly forces
  it:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `docs/maintainer_guide.md`
- lower-value non-touch surfaces:
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `tests/test_fuzz.c`
  - benchmark and example surfaces
  - package/install/export surfaces

## Exact Day 11 Center

The exact Day 11 implementation center is now fixed to:

- `tests/test_integration.c`

That owner already contains the strongest retained public failure-path seams:

- direct-family cancellation and callback short-circuit coverage
- retry-after-failure and original-matrix-preservation coverage
- public lifecycle solve / refactor rejection and state-preservation coverage
- zeroed-state, mismatched-state, and repeated-solve lifecycle coverage
- QR / iterative / eigensolver cancellation coverage through the shared
  progress-callback proof owner

The highest-value next move is therefore to deepen failure-path numerical proof
inside that existing owner rather than spread the batch across family-local
files or reopen the property owner.

## Best Failure-Path Lane

The strongest Day 10 failure-path lane is now fixed to:

- cancellation and callback short-circuit guarantees on retained public
  workflows
- error-path factor / solve / refactor preservation, especially when callers
  retry after failure
- zeroed-state, mismatched-state, and old-factor-preservation guarantees on
  the shared public lifecycle path

This keeps the batch aligned with the Day 9 exit state:

- Day 9 closed the strongest deterministic property-depth contradiction
- Day 11 should now target the most fragile lifecycle guarantees instead of
  adding more property breadth

## Support-Only Follow-Through

The strongest support-only follow-through is now:

- `tests/test_chol_csc.c`
- `tests/test_ldlt.c`
- `docs/maintainer_guide.md`

Current reading:

- `tests/test_chol_csc.c` and `tests/test_ldlt.c` already own the strongest
  family-local direct proofs and should move only if the integration batch
  exposes one local contradiction that the shared proof owner cannot express
  cleanly
- `tests/test_iterative.c` and `tests/test_eigs.c` already have lower-value
  later-family local surfaces and should stay deferred because
  `tests/test_integration.c` already owns the meaningful callback-cancellation
  contract
- maintainer wording can remain deferred unless the landed batch changes the
  maintained assurance reading in a way the current support text no longer
  describes truthfully

## Preserved Fence

The bounded Day 10 fence is explicit:

- no reopening the Day 6 external differential lane as the main batch center
- no reopening Day 9 seeded-property depth as the main batch center
- no benchmark or example drift into failure-path correctness ownership
- no package/install/export churn detached from a real landed proof seam
- no family-local proof inflation when the shared integration owner already
  carries the authoritative lifecycle semantics

## Exit State

- Sprint 84 now has one exact third implementation contract.
- Day 11 can stay bounded to failure-path numerical proof in
  `tests/test_integration.c`.
- Policy / CI / support-surface alignment remains explicitly deferred until
  after the landed failure-path batch.
