# Sprint 200 Day 7: Failed-Allocation Tests

## Purpose

Add deterministic allocation-failure regression tests for the selected
`sparse_symbolic_lu()` owner and prove failed calls do not publish
success-looking symbolic outputs.

## Code Changes

| File | Change |
| --- | --- |
| `src/sparse_etree.c` | Clears requested symbolic LU outputs after argument validation and before the first internal allocation. |
| `tests/test_etree.c` | Adds `test_symbolic_lu_allocation_failures_clear_outputs`. |
| `tests/test_symbolic_allocation_failure_gate_registration.py` | Requires the symbolic LU allocation-failure test and its case table. |

The production change aligns symbolic LU with the existing symbolic Cholesky
failure-publication behavior: after a failed allocation, requested output
objects are empty and safe to pass repeatedly to `sparse_symbolic_free(...)`.

## Failure-Index Coverage

The selected owner uses `sparse_create(n, n)` before the symbolic-LU-specific
allocation points. That internal matrix shell consumes six hook-controlled
allocations. Day 7 keeps those setup allocations outside the earned claim and
starts symbolic-LU owner coverage at `fail_after = 6`.

| Case | `fail_after` | Permutation path | Expected status | Output assertion |
| --- | ---: | --- | --- | --- |
| `perm seen` | 6 | yes | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `perm inverse` | 7 | yes | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `row_cols` | 6 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `parent` | 7 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `postorder` | 8 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `cc` | 9 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `sym_full col_ptr` | 10 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `sym_full row_idx` | 11 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `sym_full child_head` | 12 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `sym_full child_next` | 13 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `sym_full marker` | 14 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `sym_full tmp` | 15 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `sym_full col_rows` | 16 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `sym_full col_nrows` | 17 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `sym_full propagated row set` | 18 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `sym_U u_cnt` | 19 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `sym_U col_ptr` | 20 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |
| `sym_U row_idx` | 21 | no | `SPARSE_ERR_ALLOC` | `sym_L` and `sym_U` empty/free-safe. |

## Assertion Scope

Each case:

1. builds the input matrix before enabling allocation failure injection;
2. sets the selected `fail_after` immediately before `sparse_symbolic_lu(...)`;
3. stores the returned status in a local variable;
4. resets the allocation hook before assertions;
5. asserts `SPARSE_ERR_ALLOC`;
6. asserts the caller-owned matrix remains intact;
7. asserts the permutation remains intact for permutation-path cases;
8. asserts both requested symbolic outputs are empty and repeatedly free-safe.

## Retained Non-Claims

Day 7 still does not claim:

- allocation-failure ownership for the internal `sparse_create(n, n)` shell;
- broad `sparse_insert(...)` or node-pool allocation ownership;
- `sparse_analyze(...)` lifecycle allocation proof;
- retry-after-failure success behavior;
- cleanup/leak proof beyond public empty/free-safe output assertions;
- hosted CI or platform parity.

## Validation

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
- `make symbolic-allocation-failure-gate`: PASS; `test_etree` reported 102
  tests, 0 failures, 0 skipped, and 1880 assertions.
- `make lint`: PASS.
- `make test`: PASS.
- `git diff --check`: PASS.

Day 7 modified `.c` files, so the full C quality gate was run.
