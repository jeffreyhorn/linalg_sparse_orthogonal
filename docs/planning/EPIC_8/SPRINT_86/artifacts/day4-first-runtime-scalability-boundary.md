# Sprint 86 Day 4: First Runtime and Scalability Boundary

## Purpose

Fix the first bounded Sprint 86 runtime/scalability implementation fence so
the next design pass can define one real ND runtime contract instead of
another broad optimization rewrite.

## Main Result

Sprint 86 now has one explicit first implementation fence:

- required first landing:
  - `src/sparse_reorder_nd.c`
- support only if the first landing truly forces it:
  - `src/sparse_graph.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `tests/test_reorder.c`
  - `docs/maintainer_guide.md`
  - `README.md`
- explicitly deferred from the first landing:
  - `src/sparse_reorder.c`
  - `src/sparse_reorder_amd_qg.c`
  - `tests/test_reorder_amd_qg.c`
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_fillin.c`
  - proof-surface rebalancing as a first-batch center
  - benchmark/comparison follow-through as a first-batch center
  - CI/reviewed-path alignment as a first-batch center
  - install/package/runtime-surface widening
  - generic maintainability decomposition restart

## Strongest Clarification

The useful Day 4 clarification is now explicit:

- the best first Sprint 86 move is one bounded ND orchestration/runtime
  reduction inside `src/sparse_reorder_nd.c`
- graph-pipeline source movement remains allowed only where that first seam
  truly forces it
- reorder/graph proof-owner tests stay support-only unless the runtime landing
  changes their contract or requires tightly scoped proof updates
- benchmark and canonical-reporting surfaces remain outside the first
  implementation center
- CI/reviewed-path alignment remains later work after a real landed runtime
  seam exists

## Preserved First-Batch Fence

The preserved first-batch non-goal fence is explicit now:

- no weakening of correctness proof quality to buy runtime wins
- no broad graph/reorder family rewrite detached from the ND lane
- no generic maintainability decomposition restart
- no benchmark-governance or example-governance drift into correctness
  ownership
- no support-surface churn detached from a real landed runtime seam
- no package/platform maturity claim widening

## Exit State

- Sprint 86 now has one bounded first runtime/scalability landing center.
- Day 5 can design one ND runtime architecture contract inside that fence.
- Later proof-surface rebalancing, graph-pipeline spillover, benchmark
  comparisons, CI alignment, and broader support movement are held back until
  later lanes.
