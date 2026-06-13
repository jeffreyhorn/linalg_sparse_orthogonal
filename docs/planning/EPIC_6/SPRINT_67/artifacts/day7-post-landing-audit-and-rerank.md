# Sprint 67 Day 7: Post-Landing Audit And Rerank

Date: 2026-06-13
Branch: `sprint-67`

## Purpose

Rerank the remaining Sprint 67 maintainability seams after the Day 6
graph/reorder landing and decide whether a second graph-only batch is still the
highest-value move.

## Main Rerank Result

The Day 6 landing closed the strongest pure graph/reorder ownership
contradiction.

After Day 6:

- `src/sparse_graph.c` reads more clearly as:
  - uncoarsening orchestration
  - partition sequencing
  - retry/fallback ownership
- `src/sparse_reorder_nd.c` reads more clearly as:
  - ND recursive driver
  - public reorder entry
  - local support helpers around the recursive path

So a second graph-only batch is no longer justified automatically just because
those files are still non-trivial.

## Strongest Remaining Seam

The strongest remaining contradiction has shifted to the duplicated ND
compatibility-policy surface split across:

- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`

The duplicated policy surface still includes parallel compatibility parsing for:

- root-bisect mode
- coarsening mode
- coarsest-bisection mode
- root-bisect max-n
- coarsen floor ratio
- coarsening CV fallthrough
- separator-lift strategy
- separator-lift weight

Why this now ranks above another graph-only batch:

- it is a stronger ownership contradiction than the smaller local
  retry/fallback seam still left in `src/sparse_graph.c`
- it already sits on the CSC/analysis lane that Day 3 ranked second
- it affects the shared analysis/reorder control plane rather than only one
  local orchestration shell

## Residual Graph Seam After Day 6

`src/sparse_graph.c` still has a real local seam in:

- `partition_once(...)`
- `graph_partition_should_retry_with_forced_hem(...)`
- `sparse_graph_partition(...)`

That remains worth tracking, but it is now:

- more local
- lower-risk
- less contradictory than the shared ND policy duplication

So it becomes support/deferred context rather than the mandatory next landing.

## Exact Next Target

The strongest next batch is now:

- shared ND policy normalization across `src/sparse_reorder_nd.c` and
  `src/sparse_analysis.c`

Likely touched surfaces:

- `src/sparse_reorder_nd.c`
- `src/sparse_analysis.c`

Likely proof home:

- `tests/test_reorder_nd.c`
- `tests/test_integration.c`

Support only if the landing truly needs it:

- `src/sparse_reorder_nd_internal.h`
- `include/sparse_analysis.h`

This keeps Sprint 67 aligned with the Day 3 ranking:

1. graph/reorder first
2. CSC/analysis second

But it avoids forcing a fake second graph batch after the first landing already
closed the strongest graph-only contradiction.

## Explicit Non-Widening Fence

The next landing should still not widen into:

- already-extracted graph subsystem files
- `src/sparse_reorder_amd_qg.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- public API redesign
- packaging/platform/build churn

The rerank changes target order, not the bounded-sprint safety contract.

## Exit State

Sprint 67 Day 7 closes with one explicit post-Day-6 order:

1. first graph/reorder landing:
   - closed the strongest pure graph/reorder contradiction
2. strongest remaining seam:
   - shared ND policy normalization across `src/sparse_reorder_nd.c` and
     `src/sparse_analysis.c`
3. graph residuals now lower-priority/local:
   - retry/fallback glue in `src/sparse_graph.c`
4. likely next proof home:
   - `tests/test_reorder_nd.c`
   - `tests/test_integration.c`

That gives Day 8 one exact job:

- define the bounded shared ND policy / CSC-analysis convergence design instead
  of forcing a second graph-only batch
