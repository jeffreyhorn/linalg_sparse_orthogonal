# Sprint 133 Day 11 - Downstream CMake Consumer Proof

## Purpose

Day 11 strengthens the installed CMake consumer proof for the selected
static-first package contract. The proof now checks both consumer behavior and
installed CMake package metadata so export drift is caught before it becomes an
ambiguous downstream build failure.

## Implemented Proof

| File | Change |
| --- | --- |
| `tests/test_cmake_install.sh` | Tightened installed header inventory from a lower-bound check to the exact public-header contract and added installed CMake package metadata checks. |
| `docs/maintainer_guide.md` | Documented the expanded CMake install/export proof responsibilities. |

No C source, public headers, package templates, install rules, workflows, or
package-manager files changed on Day 11.

## Added CMake Package Checks

| Check | Implemented behavior |
| --- | --- |
| Exact header inventory | Compares installed `include/sparse/*.h` files against the current source header count plus generated `sparse_config.h`. |
| Static imported target | Requires `Sparse::sparse_lu_ortho` to be exported as `STATIC IMPORTED`. |
| Installed include prefix | Requires the imported target include directory to use `${_IMPORT_PREFIX}/include`. |
| Installed archive prefix | Requires the imported archive location to use `${_IMPORT_PREFIX}/lib/libsparse_lu_ortho.a`. |
| Source-tree path leakage | Scans installed CMake package files for the repository root path and fails on matches. |
| Build-tree path leakage | Scans installed CMake package files for the temporary CMake build path and fails on matches. |

The existing downstream consumer proof remains in place: configure
`examples/cmake_example` with `find_package(Sparse)`, build it, and run the
installed-package consumer executable.

## Validation Evidence

Successful focused run:

```text
--- Checking installed CMake package metadata ---
  [PASS] CMake imported target is static
  [PASS] CMake imported target uses install include prefix
  [PASS] CMake imported archive uses install prefix
  [PASS] CMake package has no source-tree paths
  [PASS] CMake package has no build-tree paths
--- Summary ---
Passed: 21
Failed: 0
Skipped: 0
ALL CMAKE INSTALL TESTS PASSED
```

## Support Boundary

This proof strengthens the local installed CMake consumer story for the
selected static archive package surface. It does not add shared-library
packaging, dynamic ABI compatibility, package-manager support, or reviewed
platform install/export parity.

## Residual CMake Package Queue

| Item | Status |
| --- | --- |
| Compile definitions | No exported compile definitions are currently part of the selected installed target contract. Add an explicit assertion only if future package metadata introduces them. |
| Transitive link dependencies | The installed consumer link/run proof remains sufficient for the current self-contained static archive. Add stricter link-interface checks if external dependencies become part of the public target. |
| CI promotion | `tests/test_cmake_install.sh` remains local install/export proof unless a future sprint promotes it to reviewed CI. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| An installed CMake consumer can prove the selected contract. | Complete | `tests/test_cmake_install.sh` configures, builds, and runs `examples/cmake_example` against the installed package. |
| Build-tree leakage is detected or explicitly ruled out. | Complete | Installed package files are scanned for source-tree and temporary build-tree paths. |
| CMake package support remains aligned with public documentation. | Complete | Maintainer guide records the static imported-target and installed-prefix proof without expanding unsupported package claims. |
