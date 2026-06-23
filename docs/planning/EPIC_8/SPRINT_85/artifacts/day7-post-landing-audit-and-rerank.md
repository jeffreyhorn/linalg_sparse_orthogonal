# Sprint 85 Day 7: Post-Landing Audit and Rerank

## Purpose

Re-rank the remaining Sprint 85 maintainability contradictions after the Day 6
iterative-source cleanup landing.

## Main Result

The Day 6 landing closed the strongest first maintainability contradiction:

- `src/sparse_iterative.c` no longer stands out as the clear first cleanup
  center
- the repo now has one real bounded iterative-source cleanup seam landed
- a second immediate iterative-only batch is not the highest-value next move

The strongest remaining Sprint 85 seam is now bounded direct-family source
cleanup.

## Exact Next Center

The exact Day 8 design center is now fixed to:

- `src/sparse_chol_csc.c`

Post-Day-6 live hotspot context:

- `src/sparse_iterative.c` = `1854` lines
- `src/sparse_chol_csc.c` = `1841` lines
- `src/sparse_qr.c` = `1563` lines
- `src/sparse_ldlt.c` = `1535` lines

The useful distinction is no longer raw size alone. It is that the iterative
first-lane contradiction has already been reduced in code, while the strongest
next direct-family source hotspot still has not.

## Support-Only Follow-Through

The strongest support-only follow-through is now:

- `tests/test_chol_csc.c`
- `docs/maintainer_guide.md`
- `README.md`

Current reading:

- `tests/test_chol_csc.c` remains the strongest proof-owner concentration but
  does not become the next landing center unless the direct-family source
  cleanup truly forces helper or registration movement
- `docs/maintainer_guide.md` and `README.md` remain truthful and should stay
  deferred unless the next batch changes owner/helper boundaries in a way that
  affects rerun or maintenance guidance

## Preserved Non-Touch Map

The useful Day 7 clarification is explicit now:

- no second immediate iterative cleanup batch as the next center
- no giant-test architecture rewrite before the next source hotspot is designed
- no spillover into `src/sparse_qr.c`, `src/sparse_ldlt.c`, or `tests/test_qr.c`
  as the next center
- no benchmark, example, package, or runtime ownership drift

## Strongest Clarification

Sprint 85's next contradiction center is no longer “do more iterative cleanup
because the first batch worked.” It is also not “jump directly to giant-test
cleanup because the tests are larger than the sources.”

It is the strongest remaining unaddressed source hotspot inside the
direct-family lane.

That fixes the ordering:

- next seam = direct-family source cleanup
- later seam = giant-test architecture cleanup
- later seam = narrower proof/docs alignment only if directly forced

## Validation

This was a docs-only rerank day, so no build/test rerun was required.

The rerank was grounded in direct rereads and live post-Day-6 measurement of:

- `src/sparse_iterative.c`
- `src/sparse_chol_csc.c`
- `src/sparse_qr.c`
- `src/sparse_ldlt.c`
- `tests/test_chol_csc.c`
- `tests/test_qr.c`
- `tests/test_integration.c`
- `tests/test_ldlt.c`
- `tests/test_iterative.c`

## Exit State

- Sprint 85 now has one explicit post-Day-6 rerank.
- Day 8 can stay bounded to one direct-family source design lane.
- Giant-test cleanup remains clearly separated from the real next
  implementation move.
