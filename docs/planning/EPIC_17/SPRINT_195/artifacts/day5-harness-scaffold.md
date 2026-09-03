# Sprint 195 Day 5: Harness Scaffold

## Purpose

Add the minimum deterministic harness scaffold for the selected
`sparse_symbolic_cholesky()` reliability proof and prove it builds through a
focused gate.

## Implementation

| Surface | Change |
| --- | --- |
| `src/sparse_etree.c` | Replaced the selected direct `sym->col_ptr` allocation with `sparse_malloc_array(col_ptr_len, sizeof(idx_t), ...)`. |
| `tests/test_etree.c` | Added private allocation-hook access and two scaffold tests for empty and non-empty `col_ptr` failure paths. |
| `Makefile` | Added `symbolic-allocation-failure-gate`. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Added a guard that checks focused gate wiring and required `RUN_TEST(...)` entries. |

## Scaffold Coverage

| Test | Selected allocation path | Assertion focus |
| --- | --- | --- |
| `test_symbolic_cholesky_allocation_hook_reaches_empty_col_ptr` | Empty `n == 0` `sparse_calloc_array` path. | `SPARSE_ERR_ALLOC`, reset before assertion, zeroed symbolic output. |
| `test_symbolic_cholesky_allocation_hook_reaches_nonempty_col_ptr` | Non-empty `sym->col_ptr` wrapper allocation. | `SPARSE_ERR_ALLOC`, reset before assertion, zeroed symbolic output. |

## Focused Gate

New command:

```sh
make symbolic-allocation-failure-gate
```

The gate builds `test_etree`, runs
`tests/test_symbolic_allocation_failure_gate_registration.py`, then runs the
`test_etree` proof-owner binary.

## Notes

The first focused run exposed a fixture issue: `sparse_create(0, 0)` returns
`NULL`, so it did not reach the selected empty symbolic owner path. The empty
test now uses a zeroed stack `SparseMatrix`, which is sufficient for the
`n == 0` path because the selected function checks only dimensions before the
empty `col_ptr` allocation.

## Day 6 Handoff

The scaffold does not yet prove all selected allocation classes. Day 6 should
extend from the passing harness to cover later allocation points:

- `sym->row_idx`;
- child and marker workspace arrays;
- `col_rows` and `col_nrows`;
- propagated row-set allocations inside the postorder loop.

## Validation

Commands run:

```sh
python3 tests/test_symbolic_allocation_failure_gate_registration.py
make symbolic-allocation-failure-gate
make format-check
git diff --check
```

Results:

- registration guard passed;
- `make symbolic-allocation-failure-gate` passed;
- `test_etree` ran 99 tests, 0 failures, 0 skips, and 675 assertions;
- `make format-check` passed;
- `git diff --check` passed.
