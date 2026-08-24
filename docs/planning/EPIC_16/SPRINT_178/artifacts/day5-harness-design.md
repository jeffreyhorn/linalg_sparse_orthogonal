# Sprint 178 Day 5: Harness Design

**Sprint:** 178 - Allocation-Failure Proof Batch 2
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_178/`
**Status:** Complete

## Purpose

Design deterministic allocation-failure injection and observation for the
selected `sparse_matmul()` workspace allocation proof. Day 5 decides whether
the existing private fail-at-count hook is sufficient and defines the
test-helper shape for Day 6 implementation.

## Harness Decision

The existing private allocation-failure hook is sufficient.

Sprint 178 should not add public allocation-failure API and should not
redesign the private allocation helper layer. The selected `sparse_matmul()`
workspace allocations already use wrapped allocation helpers, so tests can
target them with `sparse_alloc_test_fail_after()` and
`sparse_alloc_test_reset()`.

## Existing Hook Semantics

| Hook behavior | Sprint 178 use |
| --- | --- |
| `sparse_alloc_test_fail_after(0)` fails the next wrapped allocation once. | Useful for direct first-call failure tests. |
| Positive values decrement once per wrapped allocation and fail when the counter reaches zero. | Used to skip `sparse_create()` shell-buffer allocations and reach `sparse_matmul()` workspace allocations. |
| The hook resets itself after the injected failure. | Tests should still call `sparse_alloc_test_reset()` in cleanup to make fixture failures isolated. |
| The hook is declared in `src/sparse_alloc_internal.h`. | Tests may include this private header as a test-only internal dependency. |
| The hook is not public API. | README/API docs must not mention the hook as a user-facing feature. |

## Selected Allocation Counts

For valid positive-dimension input matrices, `sparse_matmul()` calls
`sparse_create(m, nc)` before selected workspace allocation. The output matrix
shell uses six wrapped helper allocations:

1. `row_headers`
2. `col_headers`
3. `row_perm`
4. `inv_row_perm`
5. `col_perm`
6. `inv_col_perm`

The selected workspace failure sites occur after those six wrapped
allocations:

| Selected site | Wrapped helper call in selected path | Fail-after value |
| --- | --- | ---: |
| `acc` | first workspace helper allocation after output shell creation | 6 |
| `nz_flag` | second workspace helper allocation after output shell creation | 7 |
| `touched` | third workspace helper allocation after output shell creation | 8 |

The fail-after values assume tests create input fixtures before enabling the
hook. Enabling the hook before fixture construction would shift the counts and
make the test invalid.

## Test Helper API Plan

Day 6 should implement test-side helpers, not product API. A suitable helper
shape is:

```c
typedef struct {
    const char *name;
    long fail_after;
} MatmulWorkspaceFailureCase;

static void expect_matmul_workspace_allocation_failure(
    const MatmulWorkspaceFailureCase *failure_case);
```

The helper should:

1. Build deterministic small input matrices before enabling injection.
2. Initialize `SparseMatrix *C = NULL`.
3. Call `sparse_alloc_test_fail_after(failure_case->fail_after)`.
4. Call `sparse_matmul(A, B, &C)`.
5. Assert `SPARSE_ERR_ALLOC`.
6. Assert `C == NULL`.
7. Call `sparse_alloc_test_reset()`.
8. Retry `sparse_matmul(A, B, &C)`.
9. Assert `SPARSE_OK`.
10. Assert the expected product values.
11. Free `C`, `A`, and `B`.
12. Reset the hook again in cleanup before returning.

## Fixture Requirements

The fixture should be small but nontrivial:

- positive dimensions so `sparse_create(m, nc)` performs shell-buffer
  allocation;
- shape-compatible multiplication;
- at least two output entries so product correctness is observable;
- stable input matrices reused after injected failure;
- no hook activation during input fixture construction.

A 2x3 by 3x2 fixture is sufficient:

| Matrix | Nonzero pattern |
| --- | --- |
| `A` | `A(0,0) = 1`, `A(0,2) = 2`, `A(1,1) = 3` |
| `B` | `B(0,0) = 4`, `B(2,1) = 5`, `B(1,0) = 6` |
| Expected `C` | `C(0,0) = 4`, `C(0,1) = 10`, `C(1,0) = 18` |

## Cleanup Observation Strategy

| Invariant | Test observation |
| --- | --- |
| selected failure returns `SPARSE_ERR_ALLOC` | Assert returned error code after injected failure. |
| no partial output publication | Assert `C == NULL` after injected failure. |
| input matrices remain reusable | Retry multiplication with the same `A` and `B` after reset. |
| retry succeeds | Assert `SPARSE_OK` on retry. |
| retry product is correct | Assert selected entries and dimensions in retry output. |
| hook reset is isolated | Reset before retry and in helper cleanup. |
| public API remains unchanged | Keep helpers static inside tests and include only private internal test hook declarations. |

## Reset And Cleanup Rules

Every test path should follow these rules:

- call `sparse_alloc_test_reset()` before returning, including failure
  cleanup paths;
- free input matrices after the retry or failure cleanup;
- free retry output after verification;
- never leave a non-`NULL` output pointer from a failed injected allocation;
- do not enable the hook during input fixture construction.

## Alternatives Rejected

| Alternative | Rejection reason |
| --- | --- |
| Add a public allocation-failure test API | Violates Sprint 177 and Sprint 178 boundaries; the hook is private/internal. |
| Add a `sparse_matmul()`-specific product option or callback | Too much API surface for a test-only proof. |
| Convert direct solver raw allocations first | Higher value but outside the selected subsystem and riskier than needed for this bounded proof. |
| Add broad allocation tracing or leak counters | Larger harness redesign; direct leak proof can remain sanitizer/Valgrind-adjacent while local tests assert no publication and retry. |
| Select `sparse_create()` shell allocation instead | Separate subsystem; not the Day 3 selected target. |

## Day 6 Implementation Notes

Day 6 should:

- add static test helper constants for the selected fail-after values;
- add one helper that exercises failure, no-publication, reset, retry, and
  product verification;
- keep hook usage private to tests;
- avoid public header changes unless absolutely necessary;
- run a focused compile or test command after edits;
- run full `make format && make lint && make test` before sprint closeout if
  C or header files are modified.

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Failure injection remains deterministic | Complete | Fail-after values 6, 7, and 8 target `acc`, `nz_flag`, and `touched` after fixture creation and output shell allocation. |
| Hook semantics stay private/internal | Complete | Design uses existing `src/sparse_alloc_internal.*` hook from tests only. |
| No public product API is added for test injection | Complete | Helper API plan is static test-side code only. |
