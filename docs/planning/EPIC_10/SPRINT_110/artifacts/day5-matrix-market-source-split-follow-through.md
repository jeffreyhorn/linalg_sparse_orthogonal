# Sprint 110 Day 5 Matrix Market Source Split Follow-Through

## Purpose

Day 5 implements the Day 4 Matrix Market source-boundary plan without changing
the public API, install headers, reviewed test surfaces, or Matrix Market
behavior.

## Implementation Summary

- Added `src/sparse_matrix_build_internal.c` as the private owner for
  bulk-entry matrix construction.
- Moved `SparseBuildEntry` into `src/sparse_matrix_internal.h` so copy,
  transpose, and Matrix Market load can continue sharing one internal builder.
- Exposed `sparse_matrix_build_from_entries` only through the private internal
  header.
- Added `src/sparse_matrix_io.c` as the owner for `sparse_save_mm` and
  `sparse_load_mm`.
- Left the public declarations for `sparse_save_mm` and `sparse_load_mm` in
  `include/sparse_matrix.h`.
- Kept checked stream printing helpers in the central matrix implementation
  because display/debug functions still share those helpers.
- Registered the new implementation files in Make, CMake, and
  `build-metadata/library_sources.txt`.

## Behavior Preserved

- `sparse_copy`, `sparse_transpose`, and `sparse_load_mm` still share one
  bulk-entry builder.
- Unsorted entry streams are sorted by row, column, and original order.
- Duplicate entries retain the last value in input order.
- Final zero values are omitted from the constructed matrix.
- Matrix Market load still supports symmetric expansion and pattern matrices.
- Matrix Market save still writes logical row/column coordinates through the
  matrix permutation state.
- Matrix Market I/O still preserves existing error-code and errno handling
  expectations.

## Validation Contract

Day 5 modifies implementation `.c` and private `.h` files, so validation must
include:

- source-list parity;
- focused matrix and Matrix Market tests;
- at least one loaded-matrix solver-smoke lane;
- `make format`;
- `make lint`;
- `make test`;
- `git diff --check`.

## Validation Results

- `make source-list-check` passed with 48 registered library sources.
- Focused Make build passed for:
  - `build/test_sparse_matrix`;
  - `build/test_sparse_io`;
  - `build/test_csr`;
  - `build/test_integration`;
  - `build/test_suitesparse`;
  - `build/test_qr`.
- Focused execution passed for:
  - `build/test_sparse_matrix`;
  - `build/test_sparse_io`;
  - `build/test_csr`;
  - `build/test_integration`;
  - `build/test_suitesparse`;
  - `build/test_qr`.
- `make format && make lint && make test` passed.
- `git diff --check` passed.
- Trailing-whitespace scan over Sprint 110 docs and the new Matrix Market
  source files found no matches.

## Downstream Notes

- Day 6 should close the Matrix Market split by recording focused validation
  evidence and any remaining source-boundary drift.
- No downstream sprint may describe Matrix Market I/O as a public API split;
  this is only an internal source-ownership split.
