# Sprint 85 Day 4: First Maintainability Boundary

## Purpose

Fix the first bounded Sprint 85 maintainability implementation fence so the
next design pass can define one real extraction/decomposition contract instead
of another broad cleanup rewrite.

## Main Result

Sprint 85 now has one explicit first implementation fence:

- required first landing:
  - `src/sparse_iterative.c`
- support only if the first landing truly forces it:
  - `tests/test_iterative.c`
  - `tests/test_iterative_handle_helpers.h`
  - `tests/test_integration.c`
  - `docs/maintainer_guide.md`
  - `README.md`
- explicitly deferred from the first landing:
  - `src/sparse_chol_csc.c`
  - `src/sparse_qr.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_eigs.c`
  - `tests/test_chol_csc.c`
  - `tests/test_qr.c`
  - `tests/test_ldlt.c`
  - generic giant-test registration cleanup as a first-batch center
  - reviewed runtime-convergence work
  - benchmark/reporting ownership changes
  - install/package/runtime maturity widening

## Strongest Clarification

The useful Day 4 clarification is now explicit:

- the best first Sprint 85 move is one bounded iterative-source cleanup
- direct-family source cleanup remains the strongest second seam, not part of
  the first batch center
- giant-test architecture cleanup remains real Sprint 85 work, but only after
  the first source cleanup exposes the actual helper seam that should move
- proof-owner tests and maintainer wording stay support-only unless the first
  source landing truly changes helper boundaries or rerun expectations
- benchmark, example, and install/package surfaces remain outside the first
  implementation center

## Preserved First-Batch Fence

The preserved first-batch non-goal fence is explicit now:

- no broad algorithm rewriting detached from ownership cleanup
- no proof dilution from moving helpers without preserving owners
- no repo-wide “cleanup sweep” claim
- no support-surface churn detached from a real landed hotspot seam
- no runtime-tuning or `test_reorder_nd` work inside the first lane
- no reopening Sprint 84's bounded assurance package as part of the first
  Sprint 85 landing

## Exit State

- Sprint 85 now has one bounded first maintainability landing center.
- Day 5 can design one iterative extraction/decomposition architecture
  contract inside that fence.
- Lower-value direct-family source cleanup, giant-test architecture work, and
  broader support/runtime spillover are held back until later lanes.
