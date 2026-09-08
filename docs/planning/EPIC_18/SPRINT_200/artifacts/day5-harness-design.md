# Sprint 200 Day 5: Harness Reachability Design

## Purpose

Define the minimum deterministic failure-injection work needed to make the
selected `sparse_symbolic_lu()` owner reachable without broad allocator,
matrix, solver, or public API changes.

## Harness Decision

Reuse the existing private allocation hook:

- `sparse_alloc_test_fail_after(...)`;
- `sparse_alloc_test_reset()`;
- `sparse_malloc_array(...)`;
- `sparse_calloc_array(...)`;
- `sparse_malloc_idx_array(...)`;
- `sparse_calloc_idx_array(...)`.

No new allocation hook, environment variable, public API, or test binary is
needed for Sprint 200.

## Minimal Day 6 Implementation Plan

| Owner site | Day 6 action |
| --- | --- |
| `seen = calloc(1, seen_bytes)` | Convert to an internal wrapper allocation so valid-permutation failures are hook-controlled. |
| `inv_perm = malloc(inv_perm_bytes)` | Convert to `sparse_malloc_array((size_t)n, sizeof(idx_t), ...)`. |
| `sym_U->col_ptr = malloc(u_col_ptr_bytes)` | Convert to `sparse_malloc_array(u_col_ptr_len, sizeof(idx_t), ...)`. |
| `sym_U` early initialization | Ensure `sparse_symbolic_free(sym_U)` runs before returning from every U-building allocation failure. |

These are selected symbolic-LU owner changes only. They do not change public
allocator behavior.

## Out-Of-Scope Allocation Paths

| Path | Day 5 decision |
| --- | --- |
| Top-level `sparse_create(n, n)` allocation for temporary `B` | Do not convert in Sprint 200. |
| Matrix-node allocation from `sparse_insert(B, ...)` | Treat as propagated matrix setup failure if observed; do not claim broad matrix allocation proof. |
| Already-covered `sparse_symbolic_cholesky()` internal proof | Reuse as an intermediate failure source but do not re-claim it as a new owner. |

## Failure-Index Plan

Day 6 should establish named failure cases after wrapper conversion. The
planned fixture families are:

| Fixture family | Intended failure points |
| --- | --- |
| Natural L+U | `row_cols`, `parent`, `postorder`, `cc`, selected `sym_full` propagated failures, `u_cnt`, `sym_U->col_ptr`, `sym_U->row_idx`. |
| L-only | `row_cols`, `parent`, `postorder`, `cc`, selected `sym_full` propagated failures. |
| U-only | `row_cols`, `parent`, `postorder`, `cc`, selected `sym_full` propagated failures, `u_cnt`, `sym_U->col_ptr`, `sym_U->row_idx`. |
| Valid permutation | `seen`, `inv_perm`, then the selected natural-order points reached after permutation setup. |

Matrix-internal `sparse_insert(B, ...)` allocation failures may appear between
`row_cols` and etree workspace allocation. If included, they must be named as
propagated matrix setup failures and not counted as broad matrix proof.

## Test Helper Plan

| Helper | Purpose |
| --- | --- |
| `assert_symbolic_lu_failure_outputs_free_safe(...)` | Verify failed requested outputs are empty and remain empty after repeated `sparse_symbolic_free()`. |
| `assert_unsym_3x3_matrix_intact(...)` | Verify caller-owned matrix state after failure. |
| `assert_perm_intact(...)` | Verify caller-owned permutation state after failure. |
| `expect_symbolic_lu_allocation_failure(...)` | Apply fail-after hook, call owner, reset hook, then assert failure invariants. |
| `expect_symbolic_lu_allocation_failure_recovers(...)` | Prove retry success after reset. |

## Registration Plan

Keep the existing `make symbolic-allocation-failure-gate` umbrella target and
extend `tests/test_symbolic_allocation_failure_gate_registration.py` to require
symbolic-LU allocation-failure tests and case names. Keep the existing CTest
label:

`etree;symbolic;allocation_failure`

A separate `symbolic-lu-allocation-failure-gate` target is a fallback only if
review feedback asks for a distinct command.

## Hook Cleanup Rules

1. Call `sparse_alloc_test_reset()` before setting each fail-after case.
2. Store the selected-owner status in a local variable.
3. Call `sparse_alloc_test_reset()` before any assertion macro can early
   return.
4. Reset again before retry calls and at the end of each table-driven case.
5. Keep fixture cleanup after hook reset.

## Validation

Day 5 changed planning documentation only. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.

`git diff --check` is the Day 5 validation command.
