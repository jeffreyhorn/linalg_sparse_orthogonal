# Sprint 85 Day 8: Direct-Family Hotspot Design

## Purpose

Define the exact bounded direct-family cleanup seam Sprint 85 will land next on
the `src/sparse_chol_csc.c` hotspot.

## Main Result

Sprint 85 now has one explicit second implementation contract:

- required implementation center:
  - `src/sparse_chol_csc.c`
- directly forced support surfaces if the extraction truly needs them:
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_csc_internal.h`
  - `tests/test_chol_csc.c`
  - `docs/maintainer_guide.md`
  - `README.md`

## Exact Day 9 Seam

The strongest bounded cleanup seam is the embedded dense LDL^T/backend block
currently owned by `src/sparse_chol_csc.c` even though it belongs to the LDL^T
family.

That seam includes:

- `ldlt_dense_factor`
- `ldlt_dense_factor_selected`
- `ldlt_dense_factor_backend_name`
- the associated Accelerate probe and backend-selection helpers

The Day 9 batch should move that seam to the LDL^T CSC owner rather than
perform a generic Cholesky helper redistribution.

## Ownership Split

The Day 8 ownership split is now fixed:

- Cholesky CSC backend owner after cleanup:
  - `src/sparse_chol_csc.c`
- LDL^T dense primitive and backend-selection owner after cleanup:
  - `src/sparse_ldlt_csc.c`
- LDL^T internal declaration owner if needed:
  - `src/sparse_ldlt_csc_internal.h`
- retained proof owner only if symbol movement truly forces follow-through:
  - `tests/test_chol_csc.c`
- support-surface wording owners only if helper ownership movement changes
  maintainer guidance:
  - `docs/maintainer_guide.md`
  - `README.md`

## Strongest Clarification

The useful Day 8 clarification is explicit now:

- Day 9 should reduce mixed family ownership inside `src/sparse_chol_csc.c`
  by moving the LDL^T dense/backend seam to the LDL^T CSC owner
- it should not reopen Cholesky CSC elimination behavior, public contracts, or
  supernodal semantics
- it should not turn the batch into a broad split of every helper block in
  `src/sparse_chol_csc.c`
- giant-test architecture cleanup remains later work unless the symbol move
  truly forces proof-owner follow-through

## Preserved Non-Goal Fence

The preserved bounded-cleanup reading is explicit:

- no generic family-wide refactor across Cholesky and LDL^T
- no public-header or install/package/runtime churn
- no benchmark/example ownership drift
- no giant-test registration rewrite as part of the source move
- no reopening Sprint 84 assurance widening

## Exit State

- Sprint 85 now has one explicit second implementation contract.
- Day 9 can land one bounded mixed-ownership cleanup without reopening generic
  family-wide refactoring.
- Giant-test architecture cleanup remains clearly separated from the next
  source batch.
