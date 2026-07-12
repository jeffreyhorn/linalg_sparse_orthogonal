# Sprint 120 Day 7 Direct Split Implementation

## Purpose

Day 7 implements the direct proof-owner split selected on Day 5 and designed
on Day 6. The split moves QR solve scenario tests out of the giant QR owner
and into a focused QR solve owner while preserving QR-specific residual,
tolerance, reconstruction, SuiteSparse, and bounded QR-vs-LU comparison
contracts.

## Changed Surfaces

| Surface | Change |
|---|---|
| `tests/test_qr_solve.c` | Added focused QR solve scenario test executable. |
| `tests/test_qr.c` | Removed moved QR solve scenario tests and their `RUN_TEST` registrations. Kept remaining QR coverage and helpers still used by reorder/refinement paths. |
| `Makefile` | Added `$(TESTDIR)/test_qr_solve.c` to `TEST_SRCS`. |
| `CMakeLists.txt` | Added `add_sparse_test(test_qr_solve)` immediately after `test_qr`. |
| `docs/planning/EPIC_11/SPRINT_120/WORKING_NOTES.md` | Recorded Day 7 implementation notes and validation requirement. |

## Moved Test Owners

| Test | New owner | Behavior preserved |
|---|---|---|
| `test_qr_solve_square` | `tests/test_qr_solve.c` | QR and LU agreement within `1e-8`; true and reported residual below `1e-10`. |
| `test_qr_solve_overdetermined` | `tests/test_qr_solve.c` | Positive least-squares residual; reported residual normalized by `||b||` matches true relative residual within `1e-8`; loose `1.0` solve tolerance remains explicit. |
| `test_qr_solve_analytical` | `tests/test_qr_solve.c` | Analytical solution `x = 2.0` and residual `sqrt(2.0)` remain checked at `1e-10`. |
| `test_qr_solve_rank_deficient` | `tests/test_qr_solve.c` | Rank-deficient solve keeps `rank == 2` and the existing residual expectation. |
| `test_qr_solve_nos4` | `tests/test_qr_solve.c` | SuiteSparse `nos4` generated-RHS solve keeps the `1e-8` true residual threshold. |
| `test_qr_solve_null_residual` | `tests/test_qr_solve.c` | `sparse_qr_solve(..., NULL)` remains accepted and accuracy remains below `1e-10`. |
| `test_qr_bcsstk04` | `tests/test_qr_solve.c` | Rank, reconstruction, generated-RHS solve, and `1e-4` SuiteSparse tolerance remain visible. |
| `test_qr_west0067` | `tests/test_qr_solve.c` | SuiteSparse `west0067` generated-RHS solve keeps the `1e-8` true residual threshold. |
| `test_qr_vs_lu` | `tests/test_qr_solve.c` | Bounded `nos4` QR-vs-LU residual and max-difference assertions remain local to the QR solve owner. |
| `test_qr_tall_synthetic` | `tests/test_qr_solve.c` | Mixed reconstruction and generated-RHS tall solve assertions remain visible in the focused owner. |

## Helper Ownership

The first split intentionally avoids a broad shared helper header. QR
solve-local helper copies live in `tests/test_qr_solve.c`:

- `qr_solve_idx_count_bytes`;
- `make_qr_solve_exact_rhs`;
- `qr_solve_insert_or_free`;
- `make_qr_solve_duplicate_column_4x3`;
- `qr_solve_reconstruction_error`;
- `assert_qr_solve_reconstruction_below`;
- `qr_solve_rel_residual`;
- `assert_qr_solve_true_residual_below`.

`tests/test_qr.c` keeps its existing helpers where remaining QR reorder,
rank, sparse-mode, economy, and refinement tests still need them.

## Build and CTest Impact

| Surface | Result |
|---|---|
| Makefile membership | `test_qr_solve.c` is registered in `TEST_SRCS`. |
| CMake membership | `test_qr_solve` is registered with `add_sparse_test`. |
| Local CTest count | `ctest -N --test-dir build/quality-review-cmake` reports 55 tests, including `test_qr_solve` as test #21. |
| Existing QR executable | `test_qr` remains registered and passes with 63 tests after the solve-owner move. |
| New QR solve executable | `test_qr_solve` is registered and passes with 10 focused tests. |

## Validation Results

| Command | Result |
|---|---|
| `make format` | Passed. |
| `make build/test_qr build/test_qr_solve && ./build/test_qr && ./build/test_qr_solve` | Passed. `test_qr`: 63 tests, 0 failed. `test_qr_solve`: 10 tests, 0 failed. |
| `make source-list-check` | Passed: 49 library sources. |
| `cmake -S . -B build/quality-review-cmake` | Passed. |
| `cmake --build build/quality-review-cmake --parallel 1 --clean-first` | Passed, including `test_qr` and `test_qr_solve`. |
| `ctest -N --test-dir build/quality-review-cmake` | Passed membership inspection; total tests: 55. |
| `make lint` | Passed. |
| `make test` | Passed: all Makefile tests passed, including `test_qr` and `test_qr_solve`. |

## Rollback Status

Rollback was not needed. If a future regression appears, restore the moved QR
solve functions and `RUN_TEST` registrations to `tests/test_qr.c`, remove
`tests/test_qr_solve.c`, remove its Makefile/CMake registrations, and re-run
the focused QR and required full quality lanes.

## Non-Claims

This implementation does not claim new QR behavior, broad direct solver
parity, external-oracle completeness, platform/package support, public API
expansion, performance improvement, or state-of-the-art validation. It is a
bounded QR solve proof-owner split.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Selected direct split compiles and focused tests pass | Complete: `test_qr` and `test_qr_solve` both compile and pass. |
| Behavior-preserving movement is documented | Complete: moved tests, helper ownership, tolerances, residual semantics, and mixed reconstruction cases are documented. |
| Any new source/test owner is registered in all required build inventories | Complete: Makefile, CMake, CTest membership, source-list check, and full quality are complete. |
