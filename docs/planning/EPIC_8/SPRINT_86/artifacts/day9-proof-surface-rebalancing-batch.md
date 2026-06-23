# Sprint 86 Day 9: Proof-Surface Rebalancing Batch

## Purpose

Land the bounded ND proof-owner rebalance fixed on Day 8 while preserving the
single retained reviewed correctness owner and measuring the actual reviewed
runtime effect.

## Main Result

The Day 9 landing stayed inside the Day 8 fence:

- required implementation center:
  - `tests/test_reorder_nd.c`
- directly forced support surfaces actually needed:
  - none
- not needed in the batch:
  - `CMakeLists.txt`
  - `Makefile`
  - `tests/test_graph.c`
  - `tests/test_reorder.c`
  - `docs/maintainer_guide.md`
  - `README.md`

## Landed Surface

The landed proof-owner rebalance introduced two bounded local seams inside
`tests/test_reorder_nd.c`:

- cached heavy-fixture copy reuse for repeated ND corpus inputs:
  - `bcsstk14`
  - `Pres_Poisson`
  - `Kuu`
- local runner-group extraction for the major in-file proof families:
  - `run_nd_core_tests()`
  - `run_nd_policy_tests()`
  - `run_nd_supernodal_tests()`

The heavy-fixture rebalance keeps proof isolation intact by caching one loaded
fixture per family and handing each test its own `SparseMatrix` copy via
`sparse_copy()`.

The registration/layout rebalance keeps the retained authoritative ND proof
owner in one file while removing another long flat `main()` block.

## Strongest Clarification

The useful Day 9 clarification is explicit:

- this was a bounded proof-owner reuse/layout landing, not an ND algorithm
  change
- it preserved one-file reviewed proof ownership
- it did not require build-level test-count growth
- it did not justify adjacent proof-owner redistribution

## Reviewed Runtime Result

The authoritative reviewed runtime result on this machine did **not** improve
the remaining long pole:

- Day 6 reviewed anchor:
  - `test_reorder_nd = 138.68 sec`
  - reviewed CMake total = `234.05 sec`
- Day 9 reviewed result:
  - `test_reorder_nd = 144.95 sec`
  - reviewed CMake total = `246.07 sec`

The local proof-owner run was likewise effectively flat:

- reviewed local `test_reorder_nd` suite time = `62.041 s`

The correct reading is therefore:

- the landed rebalance is valid and clean
- but it did not yet reduce the reviewed ND runtime concentration on this run
- the strongest remaining Sprint 86 contradiction is still the reviewed ND
  long pole, not missing proof-owner cleanup

## Preserved Non-Goal Fence

The Day 8 bounded fence held:

- no `src/sparse_reorder_nd.c` reopening
- no graph-family runtime rewrite
- no build-level test split
- no redistribution into `tests/test_graph.c` or `tests/test_reorder.c`
- no benchmark/example drift into correctness ownership
- no docs or CI guidance churn

## Validation

The landed batch passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Reviewed parity remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`

## Exit State

- Sprint 86 now has one landed bounded proof-owner rebalance batch.
- `tests/test_reorder_nd.c` still owns the same reviewed ND proof surface, but
  now does so with local heavy-fixture reuse and clearer runner grouping.
- The strongest remaining Sprint 86 seam is still reviewed ND runtime
  concentration, not another immediate proof-owner layout cleanup.
