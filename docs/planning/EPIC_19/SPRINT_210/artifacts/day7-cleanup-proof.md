# Sprint 210 Day 7: Cleanup Proof

## Purpose

Prove selected linked-list LDLT cleanup behavior after partial allocation,
injected allocation failure, and successful factorization teardown without
broadening the Sprint 210 claim beyond the selected owner.

## Implementation Summary

| File | Change |
| --- | --- |
| `tests/test_ldlt.c` | Added caller-owned `nnz` preservation, success-output double-free safety, repeated failure cleanup, and success cleanup tests. |

No production source changed on Day 7.

## Cleanup Assertions

| Assertion surface | Proof |
| --- | --- |
| Caller-owned fixture | `assert_ldlt_allocation_failure_input_intact(...)` checks dimensions, `nnz == 7`, and all fixture values after injected failure and success cleanup. |
| Failed output handle | `assert_ldlt_failure_output_free_safe(...)` checks failed outputs are empty, then calls `sparse_ldlt_free()` twice and rechecks emptiness. |
| Successful output handle | `assert_ldlt_success_output_free_safe(...)` checks successful outputs own all expected fields, then calls `sparse_ldlt_free()` twice and rechecks emptiness. |
| Hook cleanup | `assert_ldlt_allocation_hook_probe_after_reset()` proves fail-after state does not leak into later allocations. |

## Added Tests

| Test | Purpose |
| --- | --- |
| `test_ldlt_linked_list_allocation_failure_cleanup_repeatable` | Runs all 25 allocation-failure sites twice in reverse order, proving cleanup does not depend on earlier test order or a pristine process state. |
| `test_ldlt_linked_list_success_cleanup_free_safe` | Proves successful linked-list LDLT output can be freed repeatedly and that caller-owned `A` remains intact before and after teardown. |

## Ownership Boundary

The selected owner owns:

- `sparse_ldlt_t` output fields published during linked-list factorization;
- temporary working copy/workspace allocations created inside factorization.

The selected owner does not own:

- caller-provided `SparseMatrix *A`;
- the shared matrix pool allocator generally;
- CSC LDLT handles;
- reorder workspaces;
- solve, refinement, or condition-estimation workspaces.

Day 7 therefore proves cleanup only for the selected linked-list LDLT
factorization path and its already selected fail-after sweep.

## Validation

Commands run:

```sh
clang-format -i tests/test_ldlt.c
make build/test_ldlt
./build/test_ldlt
```

Focused LDLT result:

- `93` tests passed;
- `0` tests failed;
- `0` tests skipped;
- `5925` assertions passed.

Required C-change validation:

```sh
make format
make lint
make test
git diff --check
```

Full validation result:

- `make format` passed.
- `make lint` passed.
- `make test` passed.
- `git diff --check` passed.
