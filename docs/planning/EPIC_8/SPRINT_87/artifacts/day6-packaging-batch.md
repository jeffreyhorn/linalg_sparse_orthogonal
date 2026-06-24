# Sprint 87 Day 6: Packaging Batch

## Purpose

Land one bounded packaging/export modernization batch that makes the live build
and install surface read exactly like Sprint 87's maintained static-first
product contract.

## Main Result

Sprint 87's first implementation landing stayed inside the Day 5 fence:

- required implementation center:
  - `CMakeLists.txt`
- directly forced support follow-through actually needed:
  - `tests/test_cmake_install.sh`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
- not needed in the batch:
  - `cmake/SparseConfig.cmake.in`
  - `sparse.pc.in`
  - `tests/test_install.sh` logic changes
  - `examples/cmake_example/CMakeLists.txt`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`

## Landed Surface

The kept packaging win is explicit:

- the generated `SparseConfigVersion.cmake` no longer advertises
  same-major-version compatibility
- `write_basic_package_version_file(...)` now uses `ExactVersion`
- the callsite now carries an inline comment tying that choice to the repo's
  maintained static-first/no-broad-ABI contract

The directly forced support-surface follow-through was narrow:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`

These now state that the exported CMake package version file is
exact-version-only.

## Proof Follow-Through

The directly forced proof follow-through stayed product-owned:

- `tests/test_cmake_install.sh` now validates:
  - `find_package(Sparse ${EXPECTED_VERSION} EXACT REQUIRED)` succeeds
  - `find_package(Sparse ${MISMATCH_VERSION} REQUIRED)` is rejected
- `tests/test_install.sh` remained valid unchanged because the Make/pkg-config
  side of the static-first contract did not change

## Strongest Clarification

The useful Day 6 clarification is now explicit:

- the first Sprint 87 packaging win does not require opening a shared lane
- it comes from making the existing static-first install/export semantics
  stricter and more truthful
- exact package-version identity is now explicit CMake behavior rather than an
  inferred maintainer caveat
- downstream-consumer expansion and workflow follow-through remain later lanes,
  not part of the first batch

## Validation

The landed batch passed:

- `bash tests/test_cmake_install.sh`
- `bash tests/test_install.sh`

Because no `*.c` or `*.h` files changed, `make format`, `make lint`, and
`make test` were not required for this batch.

## Exit State

- Sprint 87 now has one landed bounded packaging/export batch.
- The live CMake package-version semantics now match the maintained
  static-first and no-broad-ABI contract.
- Later Sprint 87 work remains centered on consumer-proof expansion,
  workflow/platform follow-through, and broader support-surface alignment.
