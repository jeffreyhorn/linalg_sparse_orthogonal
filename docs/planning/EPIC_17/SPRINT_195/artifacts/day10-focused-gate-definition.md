# Sprint 195 Day 10: Focused Gate Definition

## Purpose

Define the maintained focused validation gate for the selected
`sparse_symbolic_cholesky()` reliability proof.

## Focused Gate

The primary local focused gate is:

```sh
make symbolic-allocation-failure-gate
```

The target builds and runs `test_etree` after executing
`tests/test_symbolic_allocation_failure_gate_registration.py`.

## Gate Coverage

The focused gate covers the selected Sprint 195 proof lane:

- selected `sparse_symbolic_cholesky()` allocation-failure status checks;
- empty and non-empty `sym->col_ptr` failure paths;
- selected partial-state failure checkpoints from `row_idx` through propagated
  row-set allocation;
- failed-output empty-state and repeated-cleanup assertions;
- successful retry after each selected known-5x5 allocation-failure checkpoint;
- existing `test_etree` success fixtures in the same process to catch ordering
  sensitivity.

## CMake Selector

Day 10 added this CTest label wiring:

```cmake
set_tests_properties(test_etree PROPERTIES LABELS "etree;symbolic;allocation_failure")
```

This enables CMake-side focused selection with:

```sh
ctest --test-dir <build-dir> -L symbolic --output-on-failure
```

The broader `ctest -L allocation_failure` selector now includes all maintained
selected allocation-failure lanes, including prior iterative and matmul lanes
plus this selected symbolic Cholesky lane.

## Drift Guard

`tests/test_symbolic_allocation_failure_gate_registration.py` now requires:

- the Make target declaration;
- the Make target dependency on `$(BUILDDIR)/test_etree`;
- the Python guard invocation inside the Make target;
- `add_sparse_test(test_etree)`;
- the `etree;symbolic;allocation_failure` CTest labels;
- the selected failure, cleanup, and retry `RUN_TEST(...)` entries;
- the selected fail-after checkpoint names and counts;
- the cleanup helper assertion call.

## Non-Claims

The gate proves only the selected symbolic Cholesky reliability lane. It does
not claim exhaustive etree, symbolic LU, analysis, direct-solver, sparse matrix,
OS OOM, platform-wide, or concurrent allocation-hook reliability coverage.
