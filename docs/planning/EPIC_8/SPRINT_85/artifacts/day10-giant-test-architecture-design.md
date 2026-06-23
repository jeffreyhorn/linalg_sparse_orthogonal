# Sprint 85 Day 10: Giant-Test Architecture Design

## Purpose

Define the exact bounded giant-test cleanup seam Sprint 85 should land next
after the Day 9 direct-family source-owner move.

## Main Result

Sprint 85 now has one explicit third implementation contract:

- required implementation center:
  - `tests/test_chol_csc.c`
- directly forced support surfaces only if the local cleanup truly needs them:
  - `tests/test_chol_csc_supernodal_helpers.h`
  - `docs/maintainer_guide.md`
  - `README.md`

## Exact Day 11 Seam

The strongest bounded cleanup seam is the still-concentrated registration
layout inside `tests/test_chol_csc.c`.

The live file already contains a good local pattern near the end: several
small runner-group helpers now own later supernodal, writeback, and dispatch
registrations, but `main()` still carries a long flat `RUN_TEST(...)` block for
earlier coverage families.

The Day 11 batch should therefore:

- reduce the long `main()` registration concentration in
  `tests/test_chol_csc.c`
- introduce a few additional local runner groups for the earlier coverage
  families that are still registered inline
- keep the cleanup inside the same proof owner rather than redistributing test
  logic across files

## Ownership Split

The Day 10 ownership split is now fixed:

- retained giant proof owner after cleanup:
  - `tests/test_chol_csc.c`
- local helper-header support owner only if the registration split truly needs
  a declaration adjustment:
  - `tests/test_chol_csc_supernodal_helpers.h`
- support-surface wording owners only if the local organization change alters
  maintainer guidance:
  - `docs/maintainer_guide.md`
  - `README.md`

## Strongest Clarification

The useful Day 10 clarification is explicit now:

- Day 11 should be a bounded registration/layout cleanup inside the retained
  Cholesky CSC proof owner
- it should not become a generic helper extraction across proof-owner files
- it should not widen into adjacent hotspots like `tests/test_qr.c`,
  `tests/test_integration.c`, or `tests/test_ldlt.c`
- it should not reopen source-owner cleanup in the direct-family or iterative
  implementation files

## Preserved Non-Goal Fence

The preserved bounded-cleanup reading is explicit:

- no cross-file proof-owner redistribution
- no generic rewrite of Cholesky CSC test logic
- no benchmark/example ownership drift
- no source-family cleanup reopening as part of the test cleanup
- no public/API/package/runtime churn

## Exit State

- Sprint 85 now has one explicit giant-test architecture contract.
- Day 11 can land one bounded `tests/test_chol_csc.c` cleanup without
  reopening generic proof-owner redistribution.
- The strongest maintainability seam after Day 9 is now fixed as local runner
  organization and `main()` registration reduction inside the retained
  Cholesky CSC proof owner.
