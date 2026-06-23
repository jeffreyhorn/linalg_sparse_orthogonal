# Sprint 84 Day 7: Post-Landing Audit and Rerank

## Purpose

Re-rank the strongest remaining Sprint 84 contradiction after the Day 6
direct-family external differential landing.

## Main Result

The Day 6 landing closed the strongest first assurance contradiction:

- the repo no longer lacks any maintained external differential lane on the
  highest-value direct-family SPD path
- `tests/test_chol_csc.c` now owns a real bounded external-process
  differential seam on `nos4` and `bcsstk04`
- a second immediate direct-family external batch is not the highest-value
  next move

The strongest remaining Sprint 84 seam is now deterministic seeded-property
expansion.

## Exact Next Center

The exact Day 8 design center is now fixed to:

- `tests/test_fuzz.c`

That file already owns the broadest deterministic property-generator surface
across:

- random LU / Cholesky / QR / SVD property checks
- large-`n` Cholesky CSC public lifecycle same-pattern properties
- large-`n` LDL^T CSC public lifecycle same-pattern properties

The strongest residual contradiction is therefore not “more proof that the
first external lane exists.” It is the limited depth and prioritization of the
deterministic property surface that sits next to the newly landed external
direct-family lane.

## Support-Only Follow-Through

The strongest support-only follow-through is now:

- `tests/test_integration.c`
- `docs/maintainer_guide.md`
- `README.md`

Current reading:

- `tests/test_integration.c` already owns the strongest public cancellation,
  failure-path, retry, and lifecycle-preservation invariants
- `docs/maintainer_guide.md` already carries the authoritative bounded
  proof-owner split after Day 6
- `README.md` remains broadly truthful and does not require movement unless
  the Day 8/9 property batch truly changes the user-visible assurance reading

## Preserved Non-Touch Map

The useful Day 7 clarification is explicit now:

- `tests/test_chol_csc.c` does not need a second immediate external
  differential batch
- `tests/test_ldlt.c` remains a support-only direct-family comparison owner
  rather than the next landing center
- `tests/test_iterative.c` and `tests/test_eigs.c` remain later-family
  assurance follow-through surfaces, not the next move
- benchmark and example surfaces still do not become correctness owners
- CI/docs/package wording can remain deferred unless the next landed batch
  truly moves the maintained assurance reading

## Strongest Clarification

Sprint 84's next contradiction center is no longer “prove that any maintained
external differential lane exists.” It is also not “immediately widen the
external direct-family lane just because the first batch worked.”

It is the bounded deterministic property depth still missing on the retained
public lifecycle surfaces.

That fixes the ordering:

- next seam = seeded-property expansion
- later seam = failure-path numerical proof
- later seam = iterative/eigs external adoption
- not the other way around

## Validation

This was a docs-only rerank day, so no build/test rerun was required.

The rerank was grounded in direct rereads of:

- `tests/test_chol_csc.c`
- `tests/chol_external_dense_reference.py`
- `tests/test_fuzz.c`
- `tests/test_integration.c`
- `tests/test_ldlt.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- `docs/maintainer_guide.md`
- `README.md`

## Exit State

- Sprint 84 now has one explicit post-Day-6 rerank.
- Day 8 can stay bounded to one deterministic seeded-property design lane.
- Support drift is clearly separated from the real next assurance move.
