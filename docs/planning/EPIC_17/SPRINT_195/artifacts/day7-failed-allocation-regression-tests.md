# Sprint 195 Day 7: Failed Allocation Regression Tests

## Purpose

Complete the formal deterministic failed-allocation regression coverage for
the selected `sparse_symbolic_cholesky()` owner.

## Implementation

| Surface | Change |
| --- | --- |
| `tests/test_etree.c` | Formalized the partial-state allocation test as `test_symbolic_cholesky_allocation_failures_clear_partial_state`. |
| `tests/test_etree.c` | Added caller-owned matrix preservation assertions after every known-5x5 selected allocation failure. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Guard now checks the formal regression test and every selected fail-after case. |

## Coverage Map

| Allocation class | Fail-after | Covered by |
| --- | ---: | --- |
| Empty `sym->col_ptr` | 0 on empty fixture | `test_symbolic_cholesky_allocation_hook_reaches_empty_col_ptr` |
| Non-empty `sym->col_ptr` | 0 on 1x1 fixture | `test_symbolic_cholesky_allocation_hook_reaches_nonempty_col_ptr` |
| `sym->row_idx` | 1 | `test_symbolic_cholesky_allocation_failures_clear_partial_state` |
| `child_head` | 2 | `test_symbolic_cholesky_allocation_failures_clear_partial_state` |
| `child_next` | 3 | `test_symbolic_cholesky_allocation_failures_clear_partial_state` |
| `marker` | 4 | `test_symbolic_cholesky_allocation_failures_clear_partial_state` |
| `tmp` | 5 | `test_symbolic_cholesky_allocation_failures_clear_partial_state` |
| `col_rows` | 6 | `test_symbolic_cholesky_allocation_failures_clear_partial_state` |
| `col_nrows` | 7 | `test_symbolic_cholesky_allocation_failures_clear_partial_state` |
| Propagated row set | 8 | `test_symbolic_cholesky_allocation_failures_clear_partial_state` |

## Assertion Contract

Each selected failure case now asserts:

- the selected call returns `SPARSE_ERR_ALLOC`;
- the private allocation hook is reset before assertions run;
- caller-visible symbolic output is cleared;
- the known-5x5 input matrix remains caller-owned and unchanged where used;
- the focused gate registration guard still owns the required test entries.

## Deferred Breadth

No selected `sparse_symbolic_cholesky()` allocation class is intentionally
deferred. The proof remains selected-owner-only and excludes
`sparse_symbolic_lu()`, `sparse_analyze()`, standalone etree/postorder/colcount
failure paths, direct solvers, QR, SVD, graph, sparse matrix construction,
real OOM behavior, and concurrent allocation-hook behavior.

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
- `make format-check` passed;
- `git diff --check` passed.
