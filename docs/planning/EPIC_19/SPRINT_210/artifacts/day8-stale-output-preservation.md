# Sprint 210 Day 8: Stale Output And Preservation

## Purpose

Prove selected linked-list LDLT allocation failures do not corrupt caller-owned
inputs and do not leave stale or ambiguous selected-owner outputs after the
public factor entrypoint resets the output handle.

## Implementation Summary

| File | Change |
| --- | --- |
| `tests/test_ldlt.c` | Added stale-output sentinels, stale-output seeding/assertion helpers, and `test_ldlt_linked_list_allocation_failures_clear_stale_outputs`. |

No production source changed on Day 8.

## Sentinel Strategy

The stale-output regression pre-seeds `sparse_ldlt_t` with static sentinel slots:

- `L` points at a static `SparseMatrix` sentinel;
- `D`, `D_offdiag`, `pivot_size`, and `perm` point at static sentinel arrays;
- `n`, `factor_norm`, and `tol` use non-success scalar sentinels.

The test first asserts the sentinels are present, then injects each Day 6
allocation failure. On failure, it verifies the public entrypoint has cleared
all factor fields to the empty/free-safe state and has not left stale
success-looking output behind.

## Added Regression

| Test | Coverage |
| --- | --- |
| `test_ldlt_linked_list_allocation_failures_clear_stale_outputs` | Runs all 25 fail-after sites with pre-seeded stale outputs; expects `SPARSE_ERR_ALLOC`; verifies `used_csc_path` changes from `-77` to `0`; verifies output fields are empty; verifies caller-owned fixture dimensions, `nnz`, and values are unchanged; probes hook reset. |

## Preservation Evidence

The caller-owned fixture remains externally owned by the test:

- `sparse_ldlt_factor_opts(...)` receives `const SparseMatrix *A`;
- every failure case verifies `sparse_rows(A) == 3`, `sparse_cols(A) == 3`,
  and `sparse_nnz(A) == 7`;
- every fixture value is checked after failure;
- the selected owner never frees `A`; the test frees it after hook reset and
  output assertions.

## Boundary Notes

This proof follows the LDLT public contract documented in `include/sparse_ldlt.h`:
factor functions overwrite the output struct without freeing existing contents,
so callers must call `sparse_ldlt_free()` before reusing a populated factor
object. Day 8 proves stale slots are cleared on failed selected linked-list
factorization; it does not claim safe reuse of a live populated output object
without caller cleanup.

The proof remains selected-owner scoped and does not claim:

- CSC LDLT stale-output behavior;
- reorder-path stale-output behavior;
- solve/refine/condest output preservation;
- broad direct-solver stale-output behavior;
- matrix-pool allocation or publication behavior.

## Validation

Commands run:

```sh
clang-format -i tests/test_ldlt.c
make build/test_ldlt
./build/test_ldlt
```

Focused LDLT result:

- `94` tests passed;
- `0` tests failed;
- `0` tests skipped;
- `6800` assertions passed.

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
