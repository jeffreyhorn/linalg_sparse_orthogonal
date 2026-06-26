# Sprint 93 Day 5: First Implementation Boundary

## Purpose

Fix one bounded first implementation fence so Sprint 93 starts with the
highest-value ND runtime seam instead of generic graph, threading, or
benchmark churn.

## Main Result

Sprint 93 now has one explicit first implementation fence:

- required first landing:
  - `src/sparse_reorder_nd.c`
  - the matching touched recursion-side and leaf/runtime seam behind the
    reviewed ND owner

- directly forced support surfaces only if the first landing truly needs them:
  - `src/sparse_graph.c`
  - `src/sparse_graph_refine.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `benchmarks/bench_reorder.c`

- explicitly later unless the first landing truly forces movement:
  - `src/sparse_graph_internal.h`
  - `src/sparse_reorder_nd_internal.h`
  - `tests/test_threads.c`
  - `tests/test_omp.c`
  - `benchmarks/bench_amd_qg.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - `CMakeLists.txt`
  - install/export and workflow surfaces

## Strongest Clarification

The useful Day 5 clarification is now explicit:

- Sprint 93 should start by improving the ND recursive runtime seam
- it should not begin by widening every graph-partition, threading, or
  runtime-control owner at once
- it should not reopen proof naming, workflow wording, or broad benchmark
  interpretation in the first batch unless the touched runtime seam itself
  truly forces it

## Deferred From The First Landing

The first batch now explicitly defers:

- broad graph/reorder rewrites
- generic multithreading everywhere
- runtime-control cleanup detached from the touched ND seam
- proof-surface restructuring detached from real reviewed-runtime savings
- benchmark/reporting widening detached from the first runtime landing
- public support-surface wording churn detached from the touched seam

## Exit State

- Sprint 93 has one explicit first implementation boundary.
- The first code landing is fixed to the ND recursive runtime owner with only
  the strongest adjacent graph and proof surfaces as directly forced
  follow-through.
- Day 6 can define the runtime-reduction implementation contract without
  reopening the ranked first-center choice.
