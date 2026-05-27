# Sprint 43 Day 5: Graph Ownership / Construction Extraction Batch 1

## Summary

Day 5 landed the first real Phase-1 graph decomposition batch by extracting the
graph ownership / construction seam out of `src/sparse_graph.c` into a new
dedicated translation unit:

- `src/sparse_graph_core.c`

The extracted seam is intentionally narrow and stable:

- `sparse_graph_from_sparse(...)`
- `sparse_graph_free(...)`
- `sparse_graph_subgraph(...)`

The remaining `src/sparse_graph.c` now starts at the heavier coarsening /
partitioning logic instead of mixing graph object lifecycle with the main
algorithmic monolith.

## Files Changed

- `src/sparse_graph_core.c`
- `src/sparse_graph.c`
- `Makefile`
- `CMakeLists.txt`

## What Moved

### 1. Graph construction

`sparse_graph_from_sparse(...)` moved into `src/sparse_graph_core.c`.

This is the cleanest Phase-1 extraction seam because it owns:

- rectangular-input rejection
- symmetric adjacency construction from `SparseMatrix`
- degree accumulation
- graph object initialization
- error cleanup on partial construction failure

That lifecycle belongs with graph object creation, not with coarsening,
bisection, FM refinement, or separator lifting.

### 2. Graph teardown

`sparse_graph_free(...)` moved into `src/sparse_graph_core.c`.

This keeps graph ownership rules adjacent to graph creation and makes the
decomposition boundary clearer for later hierarchy/coarsening extraction.

### 3. Current subgraph seam

`sparse_graph_subgraph(...)` moved into `src/sparse_graph_core.c`.

Even though it is currently still a stub, it belongs with graph ownership and
construction rather than deeper partition/refinement code.

## What Did Not Move

Day 5 intentionally did **not** move:

- hierarchy lifecycle
- heavy-edge matching / HCC coarsening
- coarse bisection
- FM refinement
- separator lifting
- top-level partition orchestration

That preserves the Phase-1 extraction order set on Days 2-4:

1. graph ownership / construction
2. hierarchy / coarsening
3. coarse bisection
4. later-phase FM / separator cleanup

## Build-System Wiring

Both maintained build systems were expanded in the bounded Day 4 shape:

- `Makefile`
- `CMakeLists.txt`

`src/sparse_graph_core.c` was added explicitly to the library source list while
retaining `src/sparse_graph.c`.

This preserves the current explicit-source ownership model and avoids any
broader build-graph redesign.

## Boundary Quality

The Day 5 extraction preserved the intended shared-vs-local rule:

- shared graph declarations remain in `src/sparse_graph_internal.h`
- no new public headers were introduced
- no public API changes were made
- no algorithmic behavior was intentionally changed

The extraction also removed `sparse_matrix_internal.h` from the remaining
`src/sparse_graph.c`, which is a good Phase-1 signal: the residual monolith is
now less entangled with graph object construction details.

## Validation

Because `*.c` files changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

## Day 5 Outcome

Sprint 43 now has a real first extraction batch in place:

- graph object ownership and construction are no longer buried in the main
  graph/ND monolith
- build wiring already supports the split cleanly
- the remaining monolith begins at the heavier algorithmic seam, which makes
  the upcoming hierarchy/coarsening extraction more straightforward

This is the right Phase-1 result: structural progress without FM/separator
scope creep or public-contract churn.
