# Sprint 43 Day 11: Focused Graph Test Design

## Summary

Day 11 audited the current graph / ND regression surface against the Sprint 43
Phase-1 module split and defined the bounded Day 12 seam-test batch.

The key conclusion is that the existing suite already covers the broad public
graph behavior well. The remaining test need is **not** a broad graph rewrite.
It is a small seam-protection batch for the new module boundaries.

## Current Coverage Shape

The current graph-focused suite already gives strong public behavior coverage
through:

- `tests/test_graph.c`
- `tests/test_graph_fm_buckets.c`
- `tests/test_reorder_nd.c`
- `tests/test_reorder_amd_qg.c`

Together these already cover:

- graph construction from sparse matrices
- hierarchy/coarsening behavior
- default and spectral coarse-bisection behavior
- full multilevel partition behavior
- ND integration
- quotient-graph AMD integration

That means Sprint 43 does **not** need new broad algorithm tests on Day 12.

## Highest-Value Gaps

### 1. Successful subgraph coverage is missing

`src/sparse_graph_core.c` now owns a real implementation of:

- `sparse_graph_subgraph(...)`

But the current test surface still only has a stale stub-era negative probe:

- `test_graph_subgraph_is_stub(...)`

This is the clearest Phase-1 seam gap because the ownership/construction slice
is now extracted, but the tests do not yet pin its successful induced-subgraph
behavior directly.

### 2. Extracted coarse-bisection dispatch is not pinned explicitly enough

Current tests already cover:

- default coarse-bisection behavior
- spectral routing
- spectral fallback behavior
- default brute/GGGP size-based behavior

What remains thin after the extraction into `src/sparse_graph_bisect.c` is
explicit dispatch-seam protection for:

- `SPARSE_ND_COARSEST_BISECTION=gggp` on a small graph
- `SPARSE_ND_COARSEST_BISECTION=brute` on an oversized graph, where the
  documented contract is safe fallback to GGGP

That is a seam-protection gap, not an algorithm-quality gap.

### 3. Top-level retry/orchestration is already adequately covered for this sprint

The Day 10 orchestration/retry seam already has enough public safety coverage
through:

- HCC/bcsstk14 sep>0 protection
- partition determinism checks
- ND integration checks

So Day 12 does **not** need another large retry/orchestration-focused batch
unless a very small regression check falls out naturally.

## Day 12 Batch

Recommended Day 12 implementation batch:

### `tests/test_graph.c`

1. Add a real successful `sparse_graph_subgraph(...)` contract test.
   - Use a small simple parent graph.
   - Assert:
     - child graph shape is correct
     - adjacency only contains induced edges
     - optional `vertex_id_map_out` is correct
     - graph ownership/free path stays clean

2. Add a forced-`gggp` dispatch test on a small graph.
   - Set `SPARSE_ND_COARSEST_BISECTION=gggp`.
   - Use a graph small enough that default routing would otherwise choose brute.
   - Assert:
     - valid `{0,1}` partition
     - balance/invariant contract holds
     - behavior remains deterministic

3. Add a forced-`brute` request on an oversized graph.
   - Set `SPARSE_ND_COARSEST_BISECTION=brute`.
   - Use a graph large enough that the documented contract must fall back to
     GGGP.
   - Assert:
     - call succeeds
     - valid `{0,1}` partition is produced
     - balance/invariant contract holds

Optional only if it stays tiny:

4. Rename or retire `test_graph_subgraph_is_stub(...)` so the test names match
   the live implementation era.

## Now vs Later Boundary

### Needed now

- extracted ownership/construction seam protection
- extracted coarse-bisection dispatch protection
- behavior-preserving seam checks for the current Phase-1 split

### Better deferred

- FM refinement extraction-specific tests
- separator-lifting extraction-specific tests
- deeper runtime-strategy matrix expansion
- broader ND performance/benchmark-oriented test work

## Day 11 Outcome

Day 11 leaves Day 12 with a concrete, bounded implementation batch:

- one ownership/construction seam test
- one or two bisection-dispatch seam tests
- no broad graph test rewrite

That is the right shape for Sprint 43 Phase 1.
