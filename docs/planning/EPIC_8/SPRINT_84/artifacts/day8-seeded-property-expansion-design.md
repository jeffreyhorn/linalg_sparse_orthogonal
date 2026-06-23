# Sprint 84 Day 8: Seeded Property Expansion Design

## Purpose

Define the bounded deterministic property-expansion contract for the
highest-value retained lifecycle seams after the Day 6 external differential
landing.

## Main Result

Sprint 84 now has one explicit second implementation contract:

- required Day 9 center:
  - `tests/test_fuzz.c`
- strongest support-only follow-through if the property batch truly forces it:
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
- strongest support-only wording if the contract truly forces movement:
  - `docs/maintainer_guide.md`
  - `README.md`
- lower-value non-touch surfaces:
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - benchmark and example surfaces
  - package/install/export surfaces

## Exact Day 9 Center

The exact Day 9 implementation center is now fixed to:

- `tests/test_fuzz.c`

That owner already contains the best bounded Sprint 84 property seam:

- deterministic small random LU / Cholesky / QR / SVD properties
- large-`n` Cholesky CSC public lifecycle same-pattern properties
- large-`n` LDL^T CSC public lifecycle same-pattern properties

The highest-value next move is therefore to deepen deterministic lifecycle and
agreement properties inside that existing owner rather than reopen the Day 6
external differential lane or spread the batch across later-family surfaces.

## Best Property Lane

The strongest Day 8 property lane is now fixed to:

- repeated-run public lifecycle invariants on retained large-`n` CSC direct
  flows
- reorder / factor / solve agreement properties that stay inside the current
  public lifecycle surface
- residual and invariance properties on touched retained public flows

This keeps the batch aligned with the Day 7 rerank:

- next seam = deterministic property depth
- later seam = failure-path numerical proof
- later seam = iterative/eigs external adoption

## Support-Only Follow-Through

The strongest support-only follow-through is now:

- `tests/test_integration.c`
- `tests/test_chol_csc.c`
- `tests/test_ldlt.c`

Current reading:

- `tests/test_integration.c` already owns the strongest cancellation, retry,
  and lifecycle-preservation invariants and should move only if the new
  property batch exposes a real public-lifecycle contradiction
- `tests/test_chol_csc.c` and `tests/test_ldlt.c` already own the strongest
  family-local direct proofs and should stay support-only unless Day 9 needs a
  very local helper or invariant adjustment
- maintainer/README wording can remain deferred unless the landed property
  batch truly changes the maintained assurance reading

## Preserved Fence

The bounded Day 8 fence is explicit:

- no reopening the Day 6 external differential lane as the main batch center
- no folding Day 10 / Day 11 failure-path proof into the Day 9 property batch
- no repo-wide property inflation across iterative/eigs just because
  `tests/test_fuzz.c` also touches QR/SVD today
- no benchmark/example drift into oracle ownership
- no support-surface churn detached from a real landed property move

## Exit State

- Sprint 84 now has one exact second implementation contract.
- Day 9 can stay bounded to deterministic seeded-property expansion in
  `tests/test_fuzz.c`.
- Later failure-path numerical proof and later-family assurance adoption remain
  explicitly deferred.
