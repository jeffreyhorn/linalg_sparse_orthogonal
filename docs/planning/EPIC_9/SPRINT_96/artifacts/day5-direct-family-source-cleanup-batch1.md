# Sprint 96 Day 5: Direct-Family Source Cleanup Batch 1

## Purpose

Day 5 lands the first bounded direct-family source cleanup from the Day 4
boundary freeze. The batch separates LDLT dense block factor/backend ownership
from the large CSC implementation owner.

## Implementation Summary

Created:

- `src/sparse_ldlt_dense.c`

Updated:

- `src/sparse_ldlt_csc.c`
- `Makefile`
- `CMakeLists.txt`

The new `src/sparse_ldlt_dense.c` owns:

- dense symmetric swap helper
- BLAS/LAPACK integer guard
- runtime external backend probe state
- dynamic-loader symbol handling
- external BLAS/LAPACK-class dense factor wrapper
- `SPARSE_LDLT_DENSE_BACKEND` environment parsing
- `ldlt_dense_factor_backend_name(...)`
- builtin `ldlt_dense_factor(...)`
- `ldlt_dense_factor_selected(...)`

`src/sparse_ldlt_csc.c` now keeps CSC-specific ownership:

- allocation/free
- row-adjacency support
- CSC conversion and writeback
- CSC validation
- linked-list compatibility wrapper
- native sparse Bunch-Kaufman workspace and elimination
- solve path
- top-level supernodal orchestration call site

## Contract Preservation

No public API changed.

The internal declarations remain in `src/sparse_ldlt_csc_internal.h`:

- `ldlt_dense_factor(...)`
- `ldlt_dense_factor_selected(...)`
- `ldlt_dense_factor_backend_name(...)`

Existing call sites continue to compile against the same internal signatures,
including the supernodal call to `ldlt_dense_factor_selected(...)`.

## Build Registration

Registered the new source in both build systems:

- `Makefile`
- `CMakeLists.txt`

The new source is ordered next to the existing LDLT CSC implementation files:

- `src/sparse_ldlt_dense.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_ldlt_csc_supernodal.c`

## Scope Control

This batch intentionally did not change:

- public headers under `include/`
- numerical recurrence logic
- environment-variable semantics for `SPARSE_LDLT_DENSE_BACKEND`
- CSC allocation, conversion, writeback, validation, solve, or native
  elimination behavior
- benchmark drivers
- generated documentation
- proof-owner test registrations

## Stale-Reference Scans

Ran:

```sh
rg -n "ldlt_dense_factor|ldlt_dense_factor_selected|ldlt_dense_factor_backend_name" src include tests
rg -n "sparse_ldlt_dense|sparse_ldlt_csc" Makefile CMakeLists.txt src
rg -n "Sprint 19 Day 11|dense LDL\^T primitive|runtime-selected backend|sparse_ldlt_csc.c:[0-9]+" src/sparse_ldlt_csc.c src/sparse_ldlt_dense.c src/sparse_ldlt_csc_internal.h
```

Results:

- dense/backend implementation is in `src/sparse_ldlt_dense.c`
- build registrations include `src/sparse_ldlt_dense.c`
- `src/sparse_ldlt_csc.c` no longer owns the dense/backend implementation
- internal declarations remain in `src/sparse_ldlt_csc_internal.h`
- no stale `sparse_ldlt_csc.c:<line>` direct line-number reference remains in
  the touched source owner

## Validation

Required code-day chain:

```sh
make format && make lint && make test
```

Result: passed.

Important targeted proof owners covered by the full test run:

- `test_chol_csc`
- `test_ldlt_csc`
- `test_direct_csc_dispatch`
- `test_direct_csc_regression`
- `test_ldlt`
- `test_ldlt_backend_dispatch`

Targeted direct proof results observed during the full test run:

- `test_chol_csc`: 152 tests passed
- `test_ldlt_csc`: 96 tests passed
- `test_direct_csc_dispatch`: 10 tests passed
- `test_direct_csc_regression`: 8 tests passed
- `test_ldlt_backend_dispatch`: 20 tests passed

## Follow-Up For Day 6

Day 6 should reconcile any remaining direct-family cleanup notes after the
extraction:

- inspect `src/sparse_ldlt_csc_internal.h` comments for whether the internal
  header description should mention the new dense source owner
- decide whether any source-local comments in `src/sparse_ldlt_csc.c`,
  `src/sparse_ldlt_dense.c`, or `src/sparse_ldlt_csc_supernodal.c` need
  durable ownership cleanup
- rerun the required full quality chain if Day 6 changes `.c` or `.h` files

## Day 5 Result

The first direct-family cleanup batch landed. LDLT dense/backend ownership is
now separated from the large CSC implementation owner, while internal function
contracts, public API, build registrations, and direct-family behavior remain
stable under the full validation chain.
