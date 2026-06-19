# Sprint 81 Day 5 - Compressed-First Architecture Design

Date: 2026-06-19  
Branch: sprint-81

## Purpose

Define the bounded Sprint 81 implementation contract so the first landing
reduces linked-list-first construction/import tax without widening into the
repeated-run direct path, broad wrapper cleanup, or general API redesign.

## Main Result

Sprint 81 now has one explicit first implementation contract:

- required implementation center:
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- support only if the first batch truly forces it:
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
  - `benchmarks/bench_refactor_csc.c`
  - `README.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`

## Ownership Split

The Day 5 ownership split is now fixed:

- compressed-first construction/import owner:
  - `src/sparse_matrix.c`
  - especially the shell lifecycle plus Matrix Market load/build paths
- linked-list compatibility shell owner:
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
  - retained as the mutable compatibility shell, not the only permanent
    product reading
- conversion/publication owner:
  - `src/sparse_matrix.c`
  - especially copy, transpose, save/export, and shell publication paths
- repeated-run workflow reuse owner, but not in the first batch:
  - `src/sparse_analysis.c`

## Compatibility Reading

The strongest useful Day 5 clarification is now explicit:

- the first landing should preserve the existing public `SparseMatrix`
  compatibility shell for callers
- it should reduce linked-list-first tax by making construction/import and
  publication read more like a bounded compressed-first seam internally
- it should not promise that repeated-run direct workflows are converged in
  the same batch

## Preserved First-Batch Fence

The first landing must preserve this non-goal fence:

- no broad public API redesign
- no repo-wide compressed-format rewrite
- no reopening of direct-family wrapper cleanup
- no hidden escalation into `src/sparse_analysis.c`
- no forced docs/examples/header churn unless the implementation truly moves
  the contract

## Exit State

- Sprint 81 now has one explicit compressed-first implementation contract.
- Ownership between the matrix-shell first landing and later repeated-run
  workflow convergence is clear.
- Day 6 can land one bounded construction/import batch without reopening
  design questions.
