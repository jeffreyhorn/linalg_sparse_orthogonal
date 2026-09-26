# Sprint 210 Day 5: Harness Implementation

## Purpose

Implement the first selected linked-list LDLT allocation-failure harness slice:
make selected output/workspace allocations reachable by the private fail-after
hook, add reusable fixture/assertion helpers, and prove initial deterministic
allocation failures leave outputs clean and allow retry.

## Implementation Summary

| File | Change |
| --- | --- |
| `src/sparse_ldlt.c` | Converted selected linked-list LDLT output arrays, permutation output, and dense elimination workspaces to private allocation wrappers. |
| `tests/test_ldlt.c` | Added allocation hook include, selected fixture helpers, failure-output assertions, input-preservation assertions, hook reset probe, success-baseline assertions, and two initial allocation-failure tests. |

## Wrapper Conversion

The following `ldlt_factor_internal()` allocations now use deterministic private
allocation wrappers:

| Allocation | Wrapper |
| --- | --- |
| `ldlt->D` | `sparse_calloc_idx_array` |
| `ldlt->D_offdiag` | `sparse_calloc_idx_array` |
| `ldlt->pivot_size` | `sparse_calloc_idx_array` |
| `ldlt->perm` | `sparse_malloc_idx_array` |
| `col_acc`, `col_acc_r` | `sparse_calloc_idx_array` |
| `nz_flag`, `nz_flag_r` | `sparse_calloc_idx_array` |
| `nz_list`, `nz_list_r` | `sparse_malloc_idx_array` |

This is a private testability change for the selected linked-list LDLT owner.
It does not create a public allocator API and does not prove CSC LDLT,
reordered LDLT, solve/refine/condest workspaces, all direct solvers, or broad
allocation-failure reliability.

## Fixture And Helper Implementation

| Helper | Role |
| --- | --- |
| `make_ldlt_allocation_failure_matrix()` | Builds a 3x3 SPD tridiagonal fixture before failure injection. |
| `ldlt_linked_list_allocation_opts()` | Forces `SPARSE_LDLT_BACKEND_LINKED_LIST` with no reordering. |
| `assert_ldlt_allocation_failure_input_intact(...)` | Checks caller-owned matrix preservation. |
| `assert_ldlt_failure_output_empty(...)` | Checks failed output is not success-looking. |
| `assert_ldlt_failure_output_free_safe(...)` | Checks repeated `sparse_ldlt_free()` is safe. |
| `assert_ldlt_allocation_hook_probe_after_reset()` | Verifies fail-after hook reset discipline. |
| `assert_ldlt_allocation_success_baseline(...)` | Checks retry success via owned fields and solve residual. |
| `expect_ldlt_allocation_failure(...)` | Runs one failed allocation case. |
| `expect_ldlt_retry_after_allocation_failure(...)` | Runs one failed allocation case and then a successful retry. |

## Initial Failure Cases

| Case | `fail_after` | Assertion coverage |
| --- | ---: | --- |
| `D output array` | 0 | Empty/free-safe output and retry after first selected allocation fails. |
| `D_offdiag output array` | 1 | Cleanup after partial output allocation. |
| `pivot_size output array` | 2 | Cleanup after two output arrays have been allocated. |

## Added Tests

| Test | Purpose |
| --- | --- |
| `test_ldlt_linked_list_allocation_failures_clear_outputs` | Verifies deterministic output-array allocation failures return `SPARSE_ERR_ALLOC`, preserve `A`, clear output, and leave the hook reset. |
| `test_ldlt_linked_list_allocation_failures_recover_on_retry` | Verifies the same fixture succeeds after hook reset for each initial failure case. |

## Day 6 Handoff

Day 6 should extend the sweep beyond the first output-array allocations to:

1. `sparse_copy(A)` propagated setup failure;
2. `L = sparse_create(n, n)` and identity `sparse_insert`;
3. `perm` allocation;
4. dense workspace allocations;
5. selected elimination `sparse_insert` propagation if the fixture reaches it;
6. any observed off-by-one fail-after changes after broader fixture selection.

## Validation

Commands run:

```sh
clang-format -i src/sparse_ldlt.c tests/test_ldlt.c
make build/test_ldlt
./build/test_ldlt
```

Focused LDLT result:

- `91` tests passed;
- `0` tests failed;
- `0` tests skipped;
- `1233` assertions passed.

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
- `make test` passed on rerun after stopping an older duplicate local
  `make test` process that had been running concurrently and caused a
  transient temp-file collision in `test_sparse_matrix`.
- `git diff --check` passed.
