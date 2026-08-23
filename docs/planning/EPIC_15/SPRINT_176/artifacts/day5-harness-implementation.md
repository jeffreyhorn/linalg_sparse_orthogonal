# Day 5: Failure Harness Implementation

## Purpose

Implement the deterministic allocation-failure harness designed on Day 4 and
apply it to the selected Sprint 176 subsystem: iterative repeated-run handle
workspace ownership.

## Implemented Hook

Day 5 adds private allocator fault injection in the internal allocation helper
layer:

| File | Change |
| --- | --- |
| `src/sparse_alloc_internal.h` | Declares private test helpers `sparse_alloc_test_fail_after()` and `sparse_alloc_test_reset()`. |
| `src/sparse_alloc_internal.c` | Tracks a disabled-by-default allocation countdown and returns `SPARSE_ERR_ALLOC` before system allocation when the countdown reaches zero. |

Hook semantics:

- `sparse_alloc_test_fail_after(0)` fails the next non-empty helper allocation.
- `sparse_alloc_test_fail_after(1)` permits one helper allocation, then fails
  the next one.
- `sparse_alloc_test_reset()` disables injection.
- zero-sized allocation helper calls and overflow checks preserve existing
  behavior.

The hook is not declared in installed public headers.

## Implemented Tests

The tests are appended to the existing `test_iterative` executable through
`tests/test_iterative_handle_helpers.h`, avoiding any CTest target-count change.

| Test | Failure point | Assertions |
| --- | --- | --- |
| `test_iter_handle_owner_allocation_failure_leaves_handle_empty` | Owner allocation in `s49_iter_handle_ensure()` | prepare returns `SPARSE_ERR_ALLOC`; handle stays empty; repeated free is safe. |
| `test_cg_handle_workspace_allocation_failure_recovers` | CG double-workspace allocation after owner allocation succeeds | prepare returns `SPARSE_ERR_ALLOC`; owner remains cleanup-safe; later prepare succeeds; repeated free resets the handle. |
| `test_gmres_handle_growth_allocation_failure_preserves_existing_workspace` | GMRES larger prepare after successful smaller prepare | failed growth returns `SPARSE_ERR_ALLOC`; existing handle remains non-empty; smaller prepare still succeeds; later larger prepare succeeds after reset. |
| `test_minres_handle_growth_allocation_failure_preserves_existing_workspace` | MINRES larger prepare after successful smaller prepare | failed growth returns `SPARSE_ERR_ALLOC`; existing handle remains non-empty; smaller prepare still succeeds; later larger prepare succeeds after reset. |

## Scope Boundary

This proof supports a narrow allocation-failure statement for public iterative
repeated-run handle preparation and growth. It does not prove allocation-failure
cleanup for one-shot iterative calls, direct solvers, matrix/core allocation
paths, QR/SVD/LDLT/LU CSR, or graph routines.

## Focused Validation

Command:

```sh
make build/test_iterative && build/test_iterative
```

Result:

- `Tests run: 84`
- `Tests failed: 0`
- `Tests skipped: 0`
- `Assertions: 734`

## Required Full Gate

Day 5 modifies `.c` and `.h` files, so the required quality gate is:

```sh
make format && make lint && make test
```

Result: passed.
