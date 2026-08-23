# Day 7: Allocation Failure Regression Gate

## Purpose

Day 7 promotes the selected Sprint 176 allocation-failure proof from a local
implementation detail into an explicit maintained regression gate.

The selected proof remains intentionally narrow:

- subsystem: iterative repeated-run workspace owner;
- Make owner: `test_iterative`;
- CMake owner: `test_iterative`;
- focused Make gate: `make iterative-allocation-failure-gate`;
- focused CTest selector: `ctest -L allocation_failure`.

## Maintained Make Surface

Day 7 adds:

```sh
make iterative-allocation-failure-gate
```

The target builds and runs `build/test_iterative`, which owns the selected
allocation-failure tests added on Days 5 and 6:

- `test_iter_handle_owner_allocation_failure_leaves_handle_empty`
- `test_cg_handle_workspace_allocation_failure_recovers`
- `test_iter_handle_invalid_prepare_calls_do_not_publish_state`
- `test_gmres_handle_growth_allocation_failure_preserves_existing_workspace`
- `test_minres_handle_growth_allocation_failure_preserves_existing_workspace`

The broad maintained `make test` path is unchanged because `test_iterative`
was already registered in `TEST_SRCS`.

## Maintained CMake Surface

Day 7 labels the existing CTest registration for `test_iterative` with:

```cmake
LABELS "iterative;allocation_failure"
```

This keeps the full CTest count unchanged while allowing focused CMake
validation through:

```sh
ctest --test-dir build/sprint176-day7-cmake -L allocation_failure --output-on-failure
```

The CMake surface remains platform-coherent because it does not add a new test
binary; it only labels an existing test executable that is already part of the
reviewed CMake suite.

## Test Inventory Impact

No test-count updates are required:

- Make `TEST_BINS` count is unchanged.
- CMake `add_test` count is unchanged.
- Windows CTest-count guards are unchanged because the existing
  `test_iterative` registration remains the only CTest row affected.

## Platform Notes

The selected proof uses the private `sparse_alloc_internal` test hook compiled
into the normal static test library. It does not require POSIX-only APIs,
environment-specific allocator behavior, or generated artifacts.

Remaining limitations:

- the proof covers only the iterative repeated-run workspace owner;
- it is not a broad allocation-failure guarantee for every solver family;
- the hook remains private/internal and is not product API.

## Validation

Focused Make gate:

```sh
make iterative-allocation-failure-gate
```

Result: passed.

Required full gate:

```sh
make format && make lint && make test
```

Result: passed.

Focused CMake registration and label gate:

```sh
cmake -S . -B build/sprint176-day7-cmake
cmake --build build/sprint176-day7-cmake --target test_iterative --parallel 1
ctest -N --test-dir build/sprint176-day7-cmake -L allocation_failure
ctest --test-dir build/sprint176-day7-cmake -L allocation_failure --output-on-failure
```

Result: passed.
