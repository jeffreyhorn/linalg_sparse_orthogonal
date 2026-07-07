# Sprint 110 Day 6 Matrix Market Focused Validation

## Purpose

Day 6 validates the Day 5 Matrix Market source split through focused Matrix I/O
tests, loaded-matrix solver-smoke tests, CMake parity, source-list parity, and
CTest registration no-drift evidence.

## Validated Surfaces

- Matrix Market load/save behavior now owned by `src/sparse_matrix_io.c`.
- Shared bulk-entry construction now owned by
  `src/sparse_matrix_build_internal.c`.
- Public Matrix Market declarations remain in `include/sparse_matrix.h`.
- CMake and Makefile library membership include the new source owners.
- `build-metadata/library_sources.txt` remains in the same reviewed order as
  the Makefile and CMake library source lists.

## Focused Matrix Market Evidence

The selected Matrix Market validation lanes passed through CTest:

- `test_sparse_matrix`;
- `test_sparse_io`.

These lanes cover Matrix Market roundtrip behavior, duplicate-entry
last-write behavior, symmetric and pattern inputs, bad-input parsing, errno
behavior, and logical row/column save behavior after permutation.

## Solver-Smoke Evidence

The selected loaded-matrix solver-smoke lanes passed through CTest:

- `test_integration`;
- `test_suitesparse`;
- `test_csr`;
- `test_qr`.

These lanes prove that matrices loaded through the moved Matrix Market path
still enter sparse matrix workflows, SuiteSparse fixture workflows, CSR/CSC
conversion workflows, and QR solve workflows.

## Build And Registration Evidence

- `make source-list-check` passed with 48 registered library sources.
- `make quality-review-cmake-compile` passed.
- The CMake clean rebuild compiled and linked:
  - `src/sparse_matrix_build_internal.c`;
  - `src/sparse_matrix_io.c`.
- `ctest -N --test-dir build/quality-review-cmake` reported 54 registered
  tests.
- Makefile/CMake test-count parity reported:
  - CMake tests: 54;
  - Makefile tests: 54.
- Selected CTest execution passed 6 of 6 tests:
  - `test_sparse_matrix`;
  - `test_sparse_io`;
  - `test_integration`;
  - `test_suitesparse`;
  - `test_csr`;
  - `test_qr`.

## Drift Review

- No public headers under `include/` were changed for the Matrix Market split.
- No install/export rules were changed.
- No helper target was added.
- No reviewed CTest registration count changed unexpectedly.
- No Windows CTest count expectation was changed.
- No documentation claims Matrix Market I/O became a public API split; this is
  an internal source-ownership split only.

## Closure

Matrix I/O work is closed for Sprint 110. The Day 5 source move has focused
Make evidence, full Make quality-gate evidence, reviewed CMake compile/parity
evidence, selected CTest execution evidence, and no public-surface drift. Day 7
can proceed to eigensolver behavior-owner selection without carrying unresolved
Matrix Market validation debt.
