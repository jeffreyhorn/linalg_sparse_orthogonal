# Sprint 195 Day 6: Selected Owner Harness Integration

## Purpose

Extend the Day 5 harness scaffold across the selected
`sparse_symbolic_cholesky()` allocation checkpoints while preserving normal
success behavior and process-global hook reset discipline.

## Implementation

| Surface | Change |
| --- | --- |
| `tests/test_etree.c` | Added a local `make_known_5x5_symbolic_matrix()` helper for selected symbolic success and failure fixtures. |
| `tests/test_etree.c` | Added table-driven fail-after support with `SymbolicFailureCase` and `expect_symbolic_cholesky_allocation_failure()`. |
| `tests/test_etree.c` | Added `test_symbolic_cholesky_allocation_failures_clear_partial_state`. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Added the Day 6 `RUN_TEST(...)` entry to the registration guard. |

## Covered Checkpoints

The selected known-5x5 fixture now proves deterministic hook reachability for:

| Fail-after | Allocation class |
| ---: | --- |
| 0 | non-empty `sym->col_ptr` allocation, covered by Day 5. |
| 1 | `sym->row_idx`. |
| 2 | `child_head`. |
| 3 | `child_next`. |
| 4 | `marker`. |
| 5 | `tmp`. |
| 6 | `col_rows`. |
| 7 | `col_nrows`. |
| 8 | first propagated row-set allocation. |

The empty-matrix `col_ptr` path remains covered by
`test_symbolic_cholesky_allocation_hook_reaches_empty_col_ptr`.

## Reset Discipline

All Day 6 failure cases:

1. prepare fixture arrays before injection;
2. call `sparse_alloc_test_reset()`;
3. arm `sparse_alloc_test_fail_after(fail_after)`;
4. store the selected-owner return status in a local;
5. call `sparse_alloc_test_reset()` before assertions;
6. assert `SPARSE_ERR_ALLOC` and empty symbolic output; and
7. reset again after cleanup.

This prevents the process-global allocation hook from contaminating subsequent
tests if an assertion fails.

## Preserved Success Behavior

The focused gate runs the entire existing `test_etree` binary. Existing
symbolic Cholesky success fixtures still pass, including the known 5x5 fixture
whose matrix builder was factored into a local helper.

## Day 7 Handoff

Day 7 should complete the formal failed-allocation regression record by
documenting allocation-site coverage and deciding whether to fold the Day 5
non-empty `col_ptr` case into the Day 6 table or keep it as a named smoke
test.

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
- `test_etree` ran 100 tests, 0 failures, 0 skips, and 748 assertions;
- `make format-check` passed;
- `git diff --check` passed.
