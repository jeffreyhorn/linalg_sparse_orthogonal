# Sprint 43 Day 12: Focused Graph Seam Tests

## Summary

Day 12 implemented the bounded graph seam-test batch defined on Day 11.

The work stayed intentionally small:

- replace the stale stub-era subgraph probe with current-era coverage
- add one real successful induced-subgraph contract test
- add two extracted coarse-bisection dispatch seam tests

This protects the new Phase-1 graph module boundaries without turning Sprint 43
into a broad graph test rewrite.

## Implemented Batch

### `tests/test_graph.c`

Added or updated:

- `test_graph_subgraph_argument_validation(...)`
- `test_graph_subgraph_path_slice(...)`
- `count_bipartition_sides(...)`
- `test_bisect_forced_gggp_small_graph(...)`
- `test_bisect_forced_brute_large_graph_falls_back_to_gggp(...)`

## Results

### 1. The extracted graph-core seam now has direct success-path coverage

The stale `test_graph_subgraph_is_stub(...)` era is now gone from the active
coverage story.

The new subgraph tests split the seam cleanly:

- one test keeps the bad-argument coverage
- one test asserts the real successful contract

The successful contract test uses a simple path graph and checks:

- subgraph construction succeeds
- child size is correct
- induced adjacency is exact
- no unexpected weights are introduced
- `vertex_id_map_out` matches the expected parent vertices

That is the right protection for the extracted `src/sparse_graph_core.c`
ownership/construction seam.

### 2. The extracted coarse-bisection dispatch seam is now pinned explicitly

The new dispatch-focused tests protect `src/sparse_graph_bisect.c` at the
module boundary rather than at a vague algorithm-quality level.

Pinned contracts:

- explicit `SPARSE_ND_COARSEST_BISECTION=gggp` on a small graph still produces
  a valid balanced bipartition
- explicit `SPARSE_ND_COARSEST_BISECTION=brute` on an oversized graph still
  returns through the documented safe GGGP fallback path

The oversized fallback test goes one step further and compares the result
against the default large-graph outcome, so the fallback behavior is pinned
more tightly than a generic “partition exists” assertion.

### 3. The batch stayed inside the Sprint 43 Phase-1 boundary

Day 12 intentionally did **not** add:

- FM refinement extraction tests
- separator-lifting extraction tests
- wider ND strategy-matrix expansion
- performance/benchmark-style graph checks

That is the correct boundary for the sprint. The new coverage protects the
ownership/construction and dispatch seams created by the file split, without
reopening broader algorithm test scope.

## Validation

Because `tests/test_graph.c` changed, the full required gate was run:

- `make format`
- `make lint`
- `make test`

All passed.

The authoritative test sweep included the touched graph regressions directly:

- `test_graph_subgraph_argument_validation`
- `test_graph_subgraph_path_slice`
- `test_bisect_forced_gggp_small_graph`
- `test_bisect_forced_brute_large_graph_falls_back_to_gggp`

## Outcome

Sprint 43 Day 12 closes the most important remaining graph seam-test gaps from
the Phase-1 decomposition work:

- extracted graph-core ownership/construction is now covered on the success
  path
- extracted coarse-bisection dispatch is now pinned explicitly
- the batch stayed bounded and behavior-preserving

That is the right test shape before the Sprint 43 validation sweep and closeout
days.
