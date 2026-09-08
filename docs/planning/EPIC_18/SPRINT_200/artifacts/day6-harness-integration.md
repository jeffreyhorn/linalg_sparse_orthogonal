# Sprint 200 Day 6: Harness Integration

## Purpose

Implement the minimal deterministic allocation-failure harness reachability
changes for the selected `sparse_symbolic_lu()` owner.

## Code Changes

| File | Change |
| --- | --- |
| `src/sparse_etree.c` | Converted optional `seen` allocation from direct `calloc` to `sparse_calloc_idx_array(...)`. |
| `src/sparse_etree.c` | Converted optional `inv_perm` allocation from direct `malloc` to `sparse_malloc_idx_array(...)`. |
| `src/sparse_etree.c` | Converted `sym_U->col_ptr` allocation from direct `malloc` to `sparse_malloc_array(...)`. |
| `src/sparse_etree.c` | Cleared partial `sym_U` state on early U-building allocation failures. |

The integration reuses the existing private allocation wrappers and
`sparse_alloc_test_fail_after(...)` hook. It does not add a new public API,
allocator callback, test binary, or environment-controlled behavior.

## Selected-Owner Reachability

| Allocation point | Day 6 status |
| --- | --- |
| `seen` | Hook-controlled after wrapper conversion. |
| `inv_perm` | Hook-controlled after wrapper conversion. |
| `row_cols` | Already hook-controlled. |
| `parent`, `postorder`, `cc` | Already hook-controlled. |
| `sym_full` intermediate allocations | Already hook-controlled through `sparse_symbolic_cholesky()`. |
| `u_cnt` | Already hook-controlled. |
| `sym_U->col_ptr` | Hook-controlled after wrapper conversion. |
| `sym_U->row_idx` | Already hook-controlled. |

## Retained Non-Claims

Day 6 still does not claim:

- `sparse_create()` allocation ownership;
- broad `sparse_insert()` or matrix-construction allocation ownership;
- all `sparse_etree.c` helper allocation failures;
- `sparse_analyze()` lifecycle allocation proof;
- direct-solver allocation proof;
- operating-system OOM behavior;
- concurrent allocation-hook safety;
- hosted CI or platform parity.

## Day 7 Handoff

Day 7 should add failed-allocation tests that prove the now hook-controlled
points return allocation failure without publishing success-looking symbolic
LU outputs.

The first concrete case table should be built around small `test_etree`
fixtures and should reset `sparse_alloc_test_reset()` before assertions or
retry calls.

## Validation

Day 6 modified `src/sparse_etree.c`, so `make format && make lint && make test`
was required before sprint closeout.

Commands run:

```sh
make format
make symbolic-allocation-failure-gate
make lint
make test
git diff --check
```

Results:

- `make format`: PASS.
- `make symbolic-allocation-failure-gate`: PASS; `test_etree` reported 101
  tests, 0 failures, 0 skipped, and 1262 assertions.
- `make lint`: PASS.
- `make test`: PASS.
- `git diff --check`: PASS.
