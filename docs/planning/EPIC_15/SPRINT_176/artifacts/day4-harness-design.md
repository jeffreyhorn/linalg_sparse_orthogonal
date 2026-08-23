# Day 4: Failure Harness Design

## Purpose

Design deterministic allocation-failure tests and cleanup observability for the
selected Sprint 176 subsystem: the iterative repeated-run workspace owner.
Day 4 defines the harness before implementation so Day 5 can edit source and
tests with a narrow target and clear validation expectations.

## Selected Harness Owner

The harness should live in the existing iterative test surface:

| Surface | Decision |
| --- | --- |
| Primary test target | `tests/test_iterative.c` |
| Existing repeated-run helper tests | `tests/test_iterative_handle_helpers.h` |
| In-scope implementation files | `src/sparse_iterative.c`, `src/sparse_iterative_workspace_internal.c`, `src/sparse_alloc_internal.c` |
| In-scope private headers | `src/sparse_iterative_workspace_internal.h`, `src/sparse_alloc_internal.h` |
| Public header changes | Avoid unless Day 8 documentation needs lifecycle wording. No allocator-control API belongs in installed public headers. |

Using `test_iterative` keeps the proof near existing public repeated-run handle
coverage for CG, GMRES, and MINRES.

## Injection Strategy

Day 5 should add a private allocation-test hook in the internal allocation
helper layer. The hook must remain outside installed public API.

Recommended private test hook semantics:

| Hook | Location | Behavior |
| --- | --- | --- |
| `sparse_alloc_test_fail_after(long remaining)` | `src/sparse_alloc_internal.h` / `.c` | Enables deterministic failure after `remaining` successful helper allocations. A value of `0` fails the next helper allocation. Negative disables injection. |
| `sparse_alloc_test_reset(void)` | `src/sparse_alloc_internal.h` / `.c` | Disables injection and resets counters. |
| internal helper check | `src/sparse_alloc_internal.c` | `sparse_malloc_array()` and `sparse_calloc_array()` consult the hook after overflow checks and before calling system `malloc`/`calloc`. |

The hook is private because `src/sparse_alloc_internal.h` is not part of the
installed public header set. Tests can include it through existing internal
include paths without adding user-facing API.

## Deterministic Failure Points

| Case | Setup | Injection | Expected result | Required cleanup assertion |
| --- | --- | --- | --- | --- |
| Owner allocation failure | Zeroed `sparse_iter_handle_t handle = {0}` | Fail next helper allocation before `sparse_iter_handle_prepare_cg(&handle, n)` | `SPARSE_ERR_ALLOC` | `handle.internal_state == NULL`; `sparse_iter_handle_free(&handle)` is safe and leaves it zeroed. |
| CG workspace allocation failure | Empty handle or freshly allocated owner | Allow owner allocation, fail next double workspace allocation | `SPARSE_ERR_ALLOC` | Handle owner, if allocated, remains cleanup-safe; repeated free leaves handle zeroed. |
| GMRES growth failure after successful small prepare | Prepare smaller GMRES successfully, then fail larger GMRES double-workspace allocation | Fail next helper allocation during larger prepare | `SPARSE_ERR_ALLOC` | Previous handle owner remains usable for original/smaller GMRES solve or prepare; final free resets handle. |
| MINRES growth failure after successful small prepare | Prepare smaller MINRES successfully, then fail larger MINRES double-workspace allocation | Fail next helper allocation during larger prepare | `SPARSE_ERR_ALLOC` | Previous handle owner remains cleanup-safe; successful prepare after hook reset confirms recovery. |

The owner-allocation and CG-workspace cases are the minimum Day 5 proof. GMRES
and MINRES growth cases should be included if the hook and assertions remain
small.

## Setup And Teardown Design

Use a helper pattern in `tests/test_iterative_handle_helpers.h`:

1. Start each failure test with `sparse_alloc_test_reset()`.
2. Initialize `sparse_iter_handle_t handle = {0}` or call
   `sparse_iter_handle_init(&handle)`.
3. Enable injection with the smallest failure count that targets the desired
   allocation.
4. Call the selected public prepare API.
5. Assert the expected `SPARSE_ERR_ALLOC`.
6. Reset injection immediately after the expected failure.
7. Assert caller-visible handle state when possible:
   - empty handle remains `internal_state == NULL` after owner allocation
     failure;
   - non-empty handle remains non-NULL after growth failure;
   - successful re-prepare is possible after hook reset.
8. Call `sparse_iter_handle_free(&handle)` twice and assert
   `handle.internal_state == NULL`.

The double-free assertion uses public lifecycle semantics, not private buffer
inspection.

## Assertion Checklist

| Assertion | Why it matters |
| --- | --- |
| Prepare returns `SPARSE_ERR_ALLOC` under injected failure. | Confirms deterministic allocation failure propagates through public API. |
| Prepare does not return `SPARSE_OK` under injected failure. | Prevents partial success from slipping through. |
| Empty handle stays empty on owner allocation failure. | Proves no dangling owner pointer is published. |
| Existing handle remains cleanup-safe on workspace growth failure. | Proves failed growth does not corrupt old ownership. |
| Hook reset allows later successful prepare. | Proves tests are deterministic and do not poison later test cases. |
| Repeated `sparse_iter_handle_free()` leaves handle zeroed. | Verifies public cleanup idempotence after failure. |
| No public header exposes allocator-test controls. | Preserves unsupported-API boundary. |

## Test Registration Design

The new tests should be added to the existing `test_iterative` suite rather
than creating a new executable:

- Make already builds `tests/test_iterative.c` as part of the maintained test
  set.
- CMake already registers `test_iterative` with `add_sparse_test`.
- Windows CTest count should not change if the tests are appended to the
  existing executable.
- The full C quality gate is required once Day 5 edits `.c` or `.h` files.

## Implementation Constraints

| Constraint | Design response |
| --- | --- |
| No unsupported public API | Keep test controls in `src/sparse_alloc_internal.h`, not `include/`. |
| Deterministic behavior | Use an explicit helper allocation countdown, not real OOM or platform memory limits. |
| Cross-test isolation | Every test resets the hook before and after use. |
| Thread scope | Sprint 176 tests run single-threaded; the hook does not establish thread-safe fault-injection guarantees. |
| Narrow claim | Test names and documentation must say repeated-run workspace handle, not all iterative solvers. |
| Existing capacity preservation | Growth-failure tests should prepare a smaller workspace first, fail a larger prepare, reset the hook, then verify smaller or later successful prepare still works. |

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Hook leaks into user-facing API | Place declarations only in private `src/` internal header; do not install or document as public API. |
| Hook affects normal builds accidentally | Default disabled state must be negative/off; reset at process start and after each test. |
| Test depends on exact allocator call count too tightly | Target public prepare paths with simple count values and document the intended allocation point in test names/comments. |
| Failed growth corrupts prior workspace but tests only free it | Add a recovery prepare after reset to show the handle remains usable. |
| New test changes Windows CTest inventory | Append tests to `test_iterative`; do not add a new registered executable unless Day 7 explicitly updates inventory guards. |

## Day 5 Implementation Checklist

1. Add private allocation fault-injection helpers to `src/sparse_alloc_internal`.
2. Add tests in `tests/test_iterative_handle_helpers.h` for owner allocation
   failure and selected prepare/growth failures.
3. Register the helper tests in the existing repeated-run handle block in
   `tests/test_iterative.c`.
4. Run `make format && make lint && make test` because Day 5 will modify C or
   internal header files.
5. Run any focused `test_iterative` command available from the build before
   the full gate if useful for iteration.

## Day 4 Completion Record

- Failure injection is deterministic and helper-count based.
- Cleanup checks are explicit enough to implement.
- The harness is scoped to iterative repeated-run workspace ownership.
- No unsupported production/public behavior is required.
- Full C quality gates are reserved for Day 5 implementation.
