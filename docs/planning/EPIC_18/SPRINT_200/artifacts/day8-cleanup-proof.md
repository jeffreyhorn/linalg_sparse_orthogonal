# Sprint 200 Day 8: Cleanup Proof

## Scope

Day 8 extends the selected-owner allocation-failure proof for
`sparse_symbolic_lu()` cleanup behavior. The proof covers the selected symbolic
LU allocation-owner sites identified on Days 3-7.

This artifact does not claim broad allocator leak detection. The local harness
does not expose allocation balance counters, so the direct evidence is limited
to test-visible cleanup invariants:

- failed calls return `SPARSE_ERR_ALLOC`;
- requested outputs are cleared to empty symbolic state;
- empty failed outputs remain safe across repeated `sparse_symbolic_free()`;
- caller-owned matrix and permutation inputs are unchanged;
- the allocation failure hook is reset before assertions;
- a normal wrapper allocation succeeds after each reset.

## Regression Coverage

Day 8 adds `test_symbolic_lu_allocation_failures_cleanup_sweep` to
`tests/test_etree.c`. The test reuses the exact selected-owner failure table
from Day 7 and repeats every failure site twice.

| Failure site | `fail_after` | Permutation path | Cleanup assertion |
| --- | ---: | --- | --- |
| `perm seen` | 6 | yes | outputs empty/free-safe; input and perm unchanged; hook reset probe succeeds |
| `perm inverse` | 7 | yes | outputs empty/free-safe; input and perm unchanged; hook reset probe succeeds |
| `row_cols` | 6 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `parent` | 7 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `postorder` | 8 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `cc` | 9 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `sym_full col_ptr` | 10 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `sym_full row_idx` | 11 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `sym_full child_head` | 12 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `sym_full child_next` | 13 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `sym_full marker` | 14 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `sym_full tmp` | 15 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `sym_full col_rows` | 16 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `sym_full col_nrows` | 17 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `sym_full propagated row set` | 18 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `sym_U u_cnt` | 19 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `sym_U col_ptr` | 20 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |
| `sym_U row_idx` | 21 | no | outputs empty/free-safe; input unchanged; hook reset probe succeeds |

## Harness Guard

`tests/test_symbolic_allocation_failure_gate_registration.py` now requires:

- registration of `test_symbolic_lu_allocation_failures_cleanup_sweep`;
- the selected symbolic LU failure-case table;
- `assert_symbolic_failure_free_safe(&sym_L)`;
- `assert_symbolic_failure_free_safe(&sym_U)`;
- `assert_allocation_hook_probe_after_reset()`.

This keeps the cleanup proof in the focused
`make symbolic-allocation-failure-gate` path.

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
- `make symbolic-allocation-failure-gate`: PASS; `test_etree` reported 103
  tests, 0 failures, 0 skipped, and 3188 assertions.
- `make lint`: PASS.
- `make test`: PASS.
- `git diff --check`: PASS.

Day 8 modified C test code, so the full C quality gate was run.

## Residual Non-Claims

Day 8 does not claim:

- general allocation-failure ownership for every API;
- cleanup coverage for unrelated matrix-construction setup allocations;
- allocator balance accounting;
- sanitizer leak-detection coverage.

Those remain out of scope for Sprint 200 Day 8 unless a future sprint adds an
allocation balance API or sanitizer lane with leak detection enabled.
