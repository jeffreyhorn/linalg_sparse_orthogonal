# Sprint 106 Day 7 - QR Householder Extraction

## Purpose

Day 7 implements the Day 6 secondary extraction boundary by moving QR
Householder and sparse-column helper mechanics out of `src/sparse_qr.c` and
into a private helper owner.

## Implementation Summary

Added:

- `src/sparse_qr_internal.h`
- `src/sparse_qr_householder.c`

Updated:

- `src/sparse_qr.c`
- `build-metadata/library_sources.txt`
- `Makefile`
- `CMakeLists.txt`

## Extracted Responsibilities

| responsibility | new owner |
|---|---|
| QR progress timer for callback elapsed time | `src/sparse_qr_householder.c` |
| Householder vector computation | `src/sparse_qr_householder.c` |
| Householder vector application | `src/sparse_qr_householder.c` |
| sparse column extraction for sparse-mode QR | `src/sparse_qr_householder.c` |
| column-sliced Householder application | `src/sparse_qr_householder.c` |
| private QR declarations | `src/sparse_qr_internal.h` |
| QR factorization, solve, rank, refine, and min-norm orchestration | `src/sparse_qr.c` |

The moved helpers were renamed into the private QR namespace:

- `sparse_qr_householder_compute(...)`
- `sparse_qr_householder_apply(...)`
- `sparse_qr_extract_column(...)`
- `sparse_qr_householder_apply_to_column(...)`

## Behavior Boundary

No public API or algorithmic behavior changed. The split only changes private
ownership:

- `include/sparse_qr.h` is unchanged.
- QR option semantics are unchanged.
- QR dense-mode and sparse-mode factorization call the same helper logic
  through the private header.
- Q application still uses the same Householder sequence and beta values.
- Make/CMake test registration is unchanged.

## Metrics

| file | before | after | change |
|---|---:|---:|---:|
| `src/sparse_qr.c` | 1,563 lines | 1,448 lines | -115 |
| `src/sparse_qr_householder.c` | 0 lines | 79 lines | +79 |
| `src/sparse_qr_internal.h` | 0 lines | 16 lines | +16 |

The net line count reduction comes from moving helper code and trimming
duplicated local helper commentary while preserving the private behavior
contract in the new helper names and Day 7 artifact.

## Build Follow-Through

Updated synchronized source membership:

- `build-metadata/library_sources.txt`
- `Makefile`
- `CMakeLists.txt`

The source-list checker now reports:

```text
source-list-check: PASS (44 library sources)
```

## Validation

Focused validation passed:

```sh
python3 scripts/check_library_sources.py
make build/test_qr build/test_colamd build/test_sprint6_integration
./build/test_qr
./build/test_colamd
./build/test_sprint6_integration
```

Focused test summaries:

- `test_qr`: 73 tests, 0 failures, 0 skips, 507 assertions
- `test_colamd`: 70 tests, 0 failures, 0 skips, 260 assertions
- `test_sprint6_integration`: 7 tests, 0 failures, 0 skips, 55 assertions

Required full C gate passed:

```sh
make format && make lint && make test
```

Final output:

```text
All tests passed.
```

Reviewed CMake compile/parity path passed:

```sh
make quality-review-cmake-compile
```

Results:

- CMake tests: 54
- Makefile tests: 54
- test-count parity: passed

Final hygiene passed:

- `git diff --check`
- trailing-whitespace scan across touched source, build-list, and Sprint 106
  planning files
- final `python3 scripts/check_library_sources.py`

## Residual Queue

Day 8 should re-check whether a second secondary source extraction is still
worth doing immediately. Current residual candidates:

- LU CSR grow/validate or dense LU helper ownership
- eigensolver shift-invert/refinement ownership
- iterative handle/workspace ownership
- SVD fixture-first cleanup rather than source split
- linked-list LDLT follow-through after the CSC extraction settles

## Exit State

The QR Householder helper seam is now privately owned, build membership is
synchronized, focused QR-family tests pass, the full C quality gate passes,
and the reviewed CMake compile/count parity path remains exact.
