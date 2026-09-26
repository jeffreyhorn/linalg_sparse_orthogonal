# Sprint 210 Day 6: Failure Sweep Tests

## Purpose

Expand the selected linked-list LDLT allocation-failure harness from the Day 5
initial output-array proof to a deterministic fail-after sweep across every
selected-owner allocation site visible through the existing private allocation
hook.

## Implementation Summary

| File | Change |
| --- | --- |
| `tests/test_ldlt.c` | Expanded `ldlt_allocation_failure_cases` from 3 to 25 named fail-after cases and added an explicit count assertion. |

No production source changed on Day 6. The Day 5 wrapper conversion already
made selected linked-list LDLT output and workspace allocations visible to the
private hook.

## Failure Sweep Map

| `fail_after` | Named site | Coverage family |
| ---: | --- | --- |
| 0 | `D output array` | Selected output array. |
| 1 | `D_offdiag output array` | Selected output array. |
| 2 | `pivot_size output array` | Selected output array. |
| 3 | `working copy entry buffer` | Propagated `sparse_copy(A)` setup allocation. |
| 4 | `working copy row headers` | Working-copy matrix shell allocation. |
| 5 | `working copy column headers` | Working-copy matrix shell allocation. |
| 6 | `working copy row permutation` | Working-copy permutation allocation. |
| 7 | `working copy inverse row permutation` | Working-copy permutation allocation. |
| 8 | `working copy column permutation` | Working-copy permutation allocation. |
| 9 | `working copy inverse column permutation` | Working-copy permutation allocation. |
| 10 | `working copy row-tail scratch` | Working-copy build scratch allocation. |
| 11 | `working copy column-tail scratch` | Working-copy build scratch allocation. |
| 12 | `L row headers` | Output `L` matrix shell allocation. |
| 13 | `L column headers` | Output `L` matrix shell allocation. |
| 14 | `L row permutation` | Output `L` permutation allocation. |
| 15 | `L inverse row permutation` | Output `L` permutation allocation. |
| 16 | `L column permutation` | Output `L` permutation allocation. |
| 17 | `L inverse column permutation` | Output `L` permutation allocation. |
| 18 | `permutation output array` | Selected `ldlt->perm` output allocation. |
| 19 | `column accumulator workspace` | Selected dense workspace allocation. |
| 20 | `nonzero flag workspace` | Selected dense workspace allocation. |
| 21 | `nonzero list workspace` | Selected dense workspace allocation. |
| 22 | `pivot-candidate accumulator workspace` | Selected dense workspace allocation. |
| 23 | `pivot-candidate nonzero flag workspace` | Selected dense workspace allocation. |
| 24 | `pivot-candidate nonzero list workspace` | Selected dense workspace allocation. |

## Assertion Coverage

Each named failure case is exercised by both allocation-failure tests:

| Test | Assertion coverage |
| --- | --- |
| `test_ldlt_linked_list_allocation_failures_clear_outputs` | Exact 25-case registration, `SPARSE_ERR_ALLOC`, preserved caller input, empty/free-safe output, and hook reset after each injected failure. |
| `test_ldlt_linked_list_allocation_failures_recover_on_retry` | Retry success after hook reset for every injected failure site, with owned fields populated and solve residual matching the fixture RHS. |

## Boundary Notes

The sweep is intentionally limited to allocations visible through the existing
private allocation hook. `sparse_insert` node-slab allocations inside the shared
matrix pool still use direct `malloc`; converting that pool would create a broad
matrix-allocation proof rather than a selected linked-list LDLT owner proof.
Day 6 therefore does not claim:

- broad matrix pool allocation-failure coverage;
- all `sparse_insert` failure propagation;
- CSC LDLT allocation-failure coverage;
- reorder allocation-failure coverage;
- solve/refine/condest workspace allocation-failure coverage;
- broad direct-solver allocation-failure reliability.

## Validation

Commands run:

```sh
clang-format -i tests/test_ldlt.c
make build/test_ldlt
./build/test_ldlt
```

Focused LDLT result:

- `91` tests passed;
- `0` tests failed;
- `0` tests skipped;
- `3587` assertions passed.

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
