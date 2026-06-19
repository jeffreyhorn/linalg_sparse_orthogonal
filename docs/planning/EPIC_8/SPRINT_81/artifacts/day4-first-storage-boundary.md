# Sprint 81 Day 4 - First Storage Boundary

Date: 2026-06-19  
Branch: sprint-81

## Purpose

Fix the first bounded Sprint 81 implementation fence so the coming
compressed-first landing moves one coherent product/storage seam rather than
sprawling into repeated-run workflow, direct-family wrapper, or broad support
surface churn.

## Main Result

Sprint 81 now has one explicit first implementation fence:

- required first landing:
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- support only if the first landing truly forces it:
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_refactor_csc.c`
  - `README.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- explicitly deferred from the first landing:
  - `src/sparse_analysis.c`
  - `src/sparse_cholesky.c`
  - `src/sparse_ldlt.c`
  - `src/sparse_qr.c`
  - broad direct-family wrapper cleanup
  - repeated-run workflow convergence as a first-batch center

## Strongest Day 4 Clarification

The strongest useful Day 4 clarification is now explicit:

- the best first Sprint 81 move is the public matrix-shell construction,
  import, and publication owner
- the repeated-run direct path remains the strongest second seam, not the
  first implementation center
- proof-owner tests and benchmark surfaces remain support-only unless the first
  landing truly changes behavior on those seams

## Preserved First-Batch Fence

The first landing must preserve this non-goal fence:

- no broad API redesign
- no backend or capability reopening
- no generic whole-library workflow rewrite
- no hidden escalation into repeated-run architecture cleanup in the first
  batch
- no broad support-surface churn without an implementation-forced reason

## Exit State

- Sprint 81 now has one explicit first implementation boundary.
- The public matrix-shell owner surfaces are fixed as the first batch center.
- Day 5 can now define one bounded compressed-first implementation contract.
