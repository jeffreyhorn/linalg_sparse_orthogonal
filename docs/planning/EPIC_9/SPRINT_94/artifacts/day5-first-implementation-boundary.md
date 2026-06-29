# Sprint 94 Day 5: First Implementation Boundary

## Purpose

Fix one bounded first implementation fence so Sprint 94 starts with the
highest-value scalar-contract seam instead of generic capability churn.

## Main Result

Sprint 94 now has one explicit first implementation fence:

- required first landing:
  - `include/sparse_types.h`
  - the matching strongest shared scalar implementation seam behind that
    public type owner

- directly forced support surfaces only if the first landing truly needs them:
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
  - `docs/maintainer_guide.md`

- explicitly later unless the first landing truly forces movement:
  - touched 64-bit and ABI maturity beyond the first scalar seam
  - `include/sparse_dense.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_qr.h`
  - the matching solver-family implementation owners
  - `tests/test_dense.c`
  - `tests/test_qr.c`
  - `tests/test_svd.c`
  - `tests/test_eigs.c`
  - `tests/test_iterative.c`
  - `benchmarks/bench_svd.c`
  - `benchmarks/bench_eigs.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `README.md`
  - `INSTALL.md`
  - `Makefile`
  - `CMakeLists.txt`

## Strongest Clarification

The useful Day 5 clarification is now explicit:

- Sprint 94 should start by widening the bounded public scalar-contract seam
- it should not begin by widening every solver-family owner at once
- it should not reopen broad docs, package wording, or benchmark
  interpretation in the first batch unless the touched scalar seam itself
  truly forces it

## Deferred From The First Landing

The first batch now explicitly defers:

- fake full-library complex support
- broad mixed-precision maturity
- generic family-wide numeric rewriting
- wider-index cleanup detached from the touched scalar seam
- solver-family breadth expansion detached from the first capability landing
- benchmark/reporting widening detached from reviewed proof owners
- support-surface wording churn detached from the first scalar landing

## Exit State

- Sprint 94 has one explicit first implementation boundary.
- The first code landing is fixed to the bounded public scalar-contract seam
  with only the strongest matrix-shell public/support owners as directly
  forced follow-through.
- Day 6 can define the scalar widening implementation contract without
  reopening the ranked first-center choice.
