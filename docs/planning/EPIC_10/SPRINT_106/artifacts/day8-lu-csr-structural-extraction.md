# Sprint 106 Day 8 - LU CSR Structural Extraction

## Goal

Complete the second selected secondary source extraction by moving LU CSR
structural helpers into a private implementation owner without changing public
API behavior, CSR LU numerical behavior, or test registration.

## Boundary Re-check

Day 7 completed the QR Householder and sparse-column helper extraction selected
by the Day 6 boundary. The QR seam now has a private helper owner and passed
focused QR-family, full C quality, and CMake compile/parity validation. No
remaining Day 7 cleanup was needed before moving to the next selected seam.

The Day 8 seam was therefore the LU CSR structural helper boundary:

- `lu_csr_grow(...)`
- `lu_csr_validate(...)`

These helpers are structural rather than algorithmic. They support CSR storage
growth and invariant checking across the plain CSR LU and block CSR LU paths,
so they can move without changing factorization math or public contracts.

## Implementation

- Added `src/sparse_lu_csr_internal.h` as the private LU CSR internal contract.
- Added `src/sparse_lu_csr_struct.c` as the owner for LU CSR storage growth and
  invariant validation.
- Removed the moved helper definitions from `src/sparse_lu_csr.c`.
- Included `sparse_lu_csr_internal.h` from `src/sparse_lu_csr.c`.
- Preserved existing `realloc` failure semantics in `lu_csr_grow(...)`,
  including the assignment of `csr->col_idx` after a successful column-index
  reallocation when the values reallocation fails.
- Left dense LU helper ownership in `src/sparse_lu_csr.c`; dense helpers belong
  to the block-elimination execution path rather than the shared CSR structural
  lifecycle.
- Updated synchronized source membership:
  - `build-metadata/library_sources.txt`
  - `Makefile`
  - `CMakeLists.txt`

## Metrics

| file | before | after | change |
|---|---:|---:|---:|
| `src/sparse_lu_csr.c` | 1,665 lines | 1,594 lines | -71 |
| `src/sparse_lu_csr_struct.c` | 0 lines | 57 lines | +57 |
| `src/sparse_lu_csr_internal.h` | 0 lines | 9 lines | +9 |

## Validation

- Source-list validation passed:
  - `python3 scripts/check_library_sources.py`
  - `source-list-check: PASS (45 library sources)`
- Focused LU CSR-family validation passed:
  - `make build/test_lu_csr build/test_sprint10_integration`
  - `./build/test_lu_csr`: 53 tests, 0 failed, 0 skipped, 1,062,184 assertions
  - `./build/test_sprint10_integration`: 14 tests, 0 failed, 0 skipped, 65 assertions
- Required full C quality gate passed:
  - `make format && make lint && make test`
  - final output: `All tests passed.`
- Reviewed CMake compile/parity path passed:
  - `make quality-review-cmake-compile`
  - CMake tests: 54
  - Makefile tests: 54
  - test-count parity passed

## Residual Risk

- Dense LU helper ownership remains deferred because those helpers are tied to
  the block-elimination path and deserve a separate boundary if extracted.
- The LU CSR elimination body remains large after this structural extraction.
- Eigensolver, iterative solver, SVD, and linked-list LDLT seams remain future
  maintainability candidates.
- Day 9 should shift from source extraction to giant-test fixture ownership and
  test-file maintainability boundaries.

## Exit Criteria

Day 8 exit criteria are satisfied: the second selected source seam has a private
owner, all build-system source references are synchronized, focused LU CSR
coverage passes, the full required C quality gate passes, and reviewed CMake
compile/test-count parity passes.
