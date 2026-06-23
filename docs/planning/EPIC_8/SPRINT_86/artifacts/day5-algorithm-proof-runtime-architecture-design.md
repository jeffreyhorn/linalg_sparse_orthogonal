# Sprint 86 Day 5: Algorithm / Proof Runtime Architecture Design

## Purpose

Define the bounded runtime/scalability contract that Sprint 86 will actually
land on the first ND runtime-reduction lane.

## Main Result

Sprint 86 now has one explicit first implementation contract:

- required implementation center:
  - `src/sparse_reorder_nd.c`
- support only if the first batch truly forces it:
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

## Ownership Split

The Day 5 ownership split is now fixed:

- ND runtime-reduction owner:
  - `src/sparse_reorder_nd.c`
- retained reviewed proof owner after the runtime landing:
  - `tests/test_reorder_nd.c`
- graph-pipeline follow-through owners only if the runtime seam truly forces
  algorithmic spillover:
  - `src/sparse_graph.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
- retained graph-family proof owner only if the first batch truly changes
  graph-path behavior:
  - `tests/test_graph.c`
- retained public reorder proof owner only if the first batch changes
  top-level reorder behavior outside the ND-focused reviewed lane:
  - `tests/test_reorder.c`
- benchmark/comparison evidence owners, but not first-batch owners:
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_fillin.c`
- support-surface wording owners only if implementation truly changes the
  maintainer rerun or reviewed-path reading:
  - `docs/maintainer_guide.md`
  - `README.md`

## Strongest Clarification

The useful Day 5 clarification is explicit now:

- the first landing should stay runtime-owned inside `src/sparse_reorder_nd.c`
- it should reduce reviewed runtime concentration by changing one bounded ND
  orchestration/policy seam rather than redistributing work across many new
  owners
- it should preserve `tests/test_reorder_nd.c` as the primary reviewed proof
  owner instead of turning Day 6 into a proof-surface rebalance batch
- it should keep graph-pipeline movement support-only unless the touched ND
  seam genuinely exposes one graph-local bottleneck that must move in the same
  batch
- it should keep benchmarks informative rather than authoritative
- it should keep CI/reviewed-path alignment explicitly later, after a real
  runtime seam lands

## Preserved First-Batch Fence

The preserved first-batch fence is explicit:

- no weakening of correctness proof quality to buy runtime wins
- no broad graph/reorder family rewrite detached from the ND lane
- no proof-surface rebalancing folded into the first batch unless the ND
  runtime seam truly forces it
- no benchmark/reporting or example drift into correctness ownership
- no generic maintainability decomposition restart
- no public docs or package/runtime churn detached from the landed runtime
  seam

## Exit State

- Sprint 86 now has one bounded ND runtime architecture contract.
- Ownership between the first ND runtime lane, retained reviewed proof owner,
  graph-pipeline spillover, and later benchmark/CI follow-through is fixed
  before Day 6 begins.
- Proof-surface rebalancing, benchmark evidence, CI alignment, and broader
  support spillover remain explicitly outside the first batch.
