# Sprint 93 Day 10: Runtime-Control Cleanup Batch

## Purpose

Land one bounded runtime-control cleanup inside the touched ND owner without
changing current runtime-policy results, widening into graph-policy work, or
pulling proof and benchmark follow-through forward before they are needed.

## Main Result

The Day 10 landing stayed inside the exact Day 9 contract:

- the only code owner touched was `src/sparse_reorder_nd.c`
- no directly forced edits were needed in:
  - `src/sparse_reorder_nd_internal.h`
  - `src/sparse_graph_internal.h`
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `benchmarks/bench_reorder.c`

The landed cleanup is now explicit:

- ND default policy construction is split into:
  - one baseline owner with the shipped default values
  - one compatibility-override application seam
- the graph override begin/end stack is now grouped behind one scoped helper:
  - `nd_graph_override_scope_begin(...)`
  - `nd_graph_override_scope_end(...)`
- `sparse_reorder_nd_with_policy(...)` now applies the touched graph-policy
  override cluster through that one scoped seam instead of manually spelling
  each begin/end call in the main execution path

## Preserved Semantics

The batch preserved the bounded Day 9 runtime-control contract:

- `sparse_reorder_nd_default_policy()` still returns the same effective
  compatibility-default policy surface
- current env names and accepted values stay unchanged
- typed-policy precedence remains unchanged
- current benchmark/test-only hooks stay intact:
  - ND profile override
  - ND base-threshold hook
- ND ordering semantics and runtime-policy results stay unchanged

## Validation

The required code-day validation queue passed cleanly:

- `make format`
- `make lint`
- `make test`

## Exit State

- Sprint 93 now has one landed ND runtime-control cleanup batch.
- The touched ND control model is smaller and sharper without changing current
  runtime-policy behavior.
- The remaining Sprint 93 queue can now move to proof-surface rebalancing and
  bounded runtime evidence from that cleaner touched control seam.
