# Sprint 81 Day 8 - Workflow Convergence Design

Date: 2026-06-19  
Branch: sprint-81

## Purpose

Fix one bounded repeated-run workflow convergence contract so Sprint 81 can
land a second implementation batch that reduces one-shot versus repeated-run
ambiguity on the strongest remaining direct-workflow seam without widening
into a broader solver-family or support-surface rewrite.

## Main Result

Sprint 81 now has one exact second implementation contract:

- required Day 9 center:
  - `src/sparse_analysis.c`
- strongest proof/measurement follow-through only if the implementation truly
  forces it:
  - `tests/test_integration.c`
  - `benchmarks/bench_refactor_csc.c`
- support-only wording only if the implementation truly changes the public
  reading:
  - `include/sparse_analysis.h`
  - `README.md`
  - `docs/maintainer_guide.md`

## Exact Day 9 Seam

The exact Day 9 seam is now fixed:

- reduce the smaller-problem repeated-run direct ambiguity inside
  `sparse_factor_numeric(...)`
- specifically the Cholesky and LDL^T branches that still fall back through
  `build_permuted_copy(...)` before factoring
- keep the batch centered on working-copy preparation and repeated-run factor
  publication, not on another public matrix-shell rewrite

## Strongest Day 8 Clarification

The strongest useful Day 8 clarification is explicit now:

- LU also still uses `build_permuted_copy(...)`, but it is not the best next
  landing center
- widening the batch to LU would turn Day 9 into a broader solver-family
  architecture rewrite instead of one bounded repeated-run convergence pass
- Cholesky and LDL^T are the stronger bounded next seam because those lanes
  already have stronger analysis-backed CSC-aware structure and stronger
  public repeated-run proof and benchmark context

## Preserved Fence

The preserved second-batch fence is explicit too:

- no reopening of the Day 6 matrix-shell construction/import batch
- no broad direct-family wrapper cleanup in `src/sparse_cholesky.c`,
  `src/sparse_ldlt.c`, or `src/sparse_qr.c`
- no generic repeated-run architecture rewrite
- no backend, capability, package, or workflow-lane spill
- no support-surface churn unless the implementation truly forces it

## Exit State

- Sprint 81 now has one explicit repeated-run workflow convergence contract.
- The exact Day 9 touch set is fixed before implementation begins.
- Day 9 can land one bounded workflow-convergence batch without reopening
  matrix-shell, wrapper, or support-surface drift.
