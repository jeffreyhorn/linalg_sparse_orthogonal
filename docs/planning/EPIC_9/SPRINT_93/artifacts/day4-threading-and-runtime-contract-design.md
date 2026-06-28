# Sprint 93 Day 4: Threading and Runtime Contract Design

## Purpose

Separate remaining Sprint 93 debt into algorithmic runtime concentration,
runtime-control complexity, and proof-topology cost so the first
implementation boundary can stay bounded to the highest-value reviewed seam.

## Main Result

Sprint 93 now has one explicit runtime/threading contract:

- algorithmic runtime debt:
  - means repeated work, recursion-side cost, or graph-partition cost on the
    touched reviewed ND lane
  - remains the strongest first-class implementation target

- runtime-control debt:
  - means profile env vars, threshold knobs, or thread-local FM/coarsening
    overrides that are still useful but too diffuse to read as one clean
    runtime model
  - remains a real Sprint 93 target, but sequenced behind the first
    algorithmic seam unless directly forced

- proof-topology debt:
  - means reviewed runtime cost caused by giant binary owners or repeated
    heavy fixture/proof concentration rather than by the algorithm itself
  - remains real Sprint 93 work, but only where rebalancing reduces cost
    without weakening correctness trust

## Strongest Clarification

The useful Day 4 clarification is now explicit:

- Sprint 93 should not treat all remaining runtime debt as a concurrency
  problem
- it should not treat every thread-local override as equally urgent
- it should first improve the touched reviewed ND runtime seam, then tighten
  the runtime-control story only where the same seam still depends on it

## Non-Claim Fence

Sprint 93 now preserves one sharper runtime/threading non-claim fence:

- no fake broad scaling victory
- no fake repo-wide threading maturity claim
- no broad cross-platform runtime parity claim
- no benchmark-superiority claim detached from the reviewed proof owners

## Exact Owner Split

The strongest direct-owner reading is now explicit:

- first-center implementation owners:
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph.c`
  - `src/sparse_graph_refine.c`

- second-center runtime-control owners if truly forced:
  - `src/sparse_graph_internal.h`
  - `src/sparse_reorder_nd_internal.h`
  - adjacent profile / override test coverage in `tests/test_reorder_nd.c`
    and `tests/test_graph.c`

- later proof-only or support-only owners unless the first landing forces
  movement:
  - `tests/test_threads.c`
  - `tests/test_omp.c`
  - `benchmarks/bench_reorder.c`
  - `README.md`
  - `docs/maintainer_guide.md`

## Deferred From The First Runtime Landing

The first batch now explicitly defers:

- generic multithreading everywhere
- broad graph/reorder rewrites detached from the reviewed ND seam
- benchmark-only tuning detached from reviewed proof
- public support-surface wording churn unless the touched runtime seam truly
  changes the contract reading

## Exit State

- Sprint 93 has one explicit threading/runtime contract before code movement.
- Day 5 is fixed to one bounded first implementation fence around the touched
  ND runtime seam.
- Runtime-control cleanup, proof-surface rebalancing, and evidence
  follow-through remain sequenced behind that first center.
