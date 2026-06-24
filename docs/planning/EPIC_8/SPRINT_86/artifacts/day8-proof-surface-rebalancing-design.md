# Sprint 86 Day 8: Proof-Surface Rebalancing Design

## Purpose

Define the bounded reviewed-surface cleanup Sprint 86 should land next so the
remaining `test_reorder_nd` runtime concentration can fall without weakening
the retained ND correctness proof.

## Main Result

Sprint 86 now has one explicit second implementation contract:

- required Day 9 center:
  - `tests/test_reorder_nd.c`
- directly forced support-only follow-through if the rebalance truly needs it:
  - `CMakeLists.txt`
  - `Makefile`
- strongest adjacent proof-owner follow-through only if the batch exposes a
  real shared-fixture or rerun-contract seam:
  - `tests/test_graph.c`
  - `tests/test_reorder.c`
- strongest support-only wording if the contract truly changes reviewed rerun
  guidance:
  - `docs/maintainer_guide.md`
  - `README.md`
- lower-value non-touch surfaces:
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_fillin.c`

## Exact Day 9 Center

The exact Day 9 implementation center is now fixed to one in-owner rebalance
inside `tests/test_reorder_nd.c`, not a second immediate algorithm batch and
not a build-level binary split.

The decisive Day 8 reason is explicit:

- after Day 6, the remaining contradiction is reviewed proof concentration,
  not unresolved ND threshold policy
- `test_reorder_nd` still accounts for roughly `59%` of the reviewed CMake
  total
- splitting into another binary would add reviewed test-count churn and build
  wiring movement without necessarily reducing sequential reviewed runtime
- the higher-value next seam is to reduce repeated heavy fixture setup inside
  the retained proof owner itself

## Best Rebalance Lane

The strongest Day 8 proof lane is now fixed to:

- extract a small number of local runner/helper seams inside
  `tests/test_reorder_nd.c`
- group the repeated heavy `bcsstk14` policy/comparison tests so that family
  can reuse one fixture load and one bounded comparison context
- group the repeated heavy `Pres_Poisson` policy/comparison tests so that
  family can reuse one fixture load and one bounded comparison context
- keep the later supernodal-postorder advisory/corpus-safety family grouped as
  its own local runner seam
- flatten the long `main()` registration block while preserving the retained
  authoritative ND proof owner

The highest-value local family split is therefore:

- core ND permutation / fill / dispatch contracts
- ND policy and typed-env override contracts on shared heavy fixtures
- supernodal-postorder advisory and corpus-safety contracts

## Support-Only Follow-Through

The strongest support-only follow-through is now:

- `CMakeLists.txt`
- `Makefile`
- `tests/test_graph.c`
- `tests/test_reorder.c`
- `docs/maintainer_guide.md`
- `README.md`

Current reading:

- `CMakeLists.txt` and `Makefile` should stay untouched unless the in-owner
  rebalance unexpectedly proves insufficient and a later true test split
  becomes necessary
- `tests/test_graph.c` and `tests/test_reorder.c` remain adjacent proof owners
  but do not become the next landing center unless the Day 9 rebalance
  exposes a real shared-fixture or rerun-contract seam
- maintainer/README wording should stay deferred unless the landed batch
  truly changes reviewed rerun guidance

## Preserved Fence

The bounded Day 8 fence is explicit:

- no second immediate ND-policy batch
- no graph-family or reorder-family runtime rewrite
- no default build-level test-count expansion as the first proof-surface move
- no redistribution of ND proof ownership into `tests/test_graph.c` or
  `tests/test_reorder.c`
- no benchmark/example drift into correctness ownership
- no CI/reviewed-path alignment folded into the Day 9 batch

## Exit State

- Sprint 86 now has one exact second implementation contract.
- Day 9 can stay bounded to `tests/test_reorder_nd.c` and reduce reviewed
  runtime concentration through fixture/reuse and runner-group rebalancing
  inside the retained ND proof owner.
- Benchmark/comparison follow-through and CI/reviewed-path alignment remain
  explicitly later.
