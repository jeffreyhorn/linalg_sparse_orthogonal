# Sprint 85 Day 11: Giant-Test Architecture Batch

## Purpose

Land the bounded giant-test cleanup fixed on Day 10 by reducing registration
concentration in `tests/test_chol_csc.c` while preserving local proof
ownership.

## Main Result

The Day 11 landing stayed inside the Day 10 fence:

- required implementation center:
  - `tests/test_chol_csc.c`
- directly forced support surfaces actually needed:
  - none
- not needed in the batch:
  - `tests/test_chol_csc_supernodal_helpers.h`
  - `docs/maintainer_guide.md`
  - `README.md`
  - adjacent proof-owner files

## Landed Surface

The landed batch introduced local runner groups for the earlier coverage
families that were still registered inline in `main()`:

- alloc / growth
- conversion round-trips
- permutations plus fill-factor / norm caching
- symbolic analysis plus validate / edge hardening
- workspace plus elimination scaffolding
- scalar kernel coverage
- solve / residual / shim coverage

`main()` now calls those grouped runners, matching the same local pattern the
file already used later for supernodal, writeback, and dispatch coverage.

## Strongest Clarification

The useful Day 11 clarification is explicit now:

- this was a bounded registration/layout cleanup, not a test-behavior change
- it reduced the concentration of the Cholesky CSC proof owner without
  redistributing test logic across files
- it preserved one-file proof ownership while making the late-file helper
  pattern consistent across the whole suite

## Preserved Non-Goal Fence

The preserved bounded-cleanup reading held:

- no cross-file proof-owner redistribution
- no generic rewrite of Cholesky CSC test logic
- no helper-header churn
- no source-family cleanup reopening
- no public/API/package/runtime churn

## Validation

The landed batch passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

## Exit State

- Sprint 85 now has one landed bounded giant-test architecture cleanup batch.
- `tests/test_chol_csc.c` still owns the same proof surface, but its
  registration layout now follows one consistent local runner-group structure.
- The strongest remaining Sprint 85 seam is later alignment and closeout work,
  not another immediate `main()` registration cleanup in this proof owner.
