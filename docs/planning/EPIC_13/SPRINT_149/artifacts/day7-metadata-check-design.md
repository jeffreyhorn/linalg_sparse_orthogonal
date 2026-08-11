# Sprint 149 Day 7: Package Metadata Check Design

## Purpose

Design the remaining Windows package metadata checks needed after the Day 6
workflow promotion. The design keeps package-file inspection separate from
downstream consumer proof and preserves the static-first, CMake-scoped Windows
claim.

## Sources Reviewed

| Source | Metadata Role |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Hosted Windows CMake install/downstream validation owner. |
| `CMakeLists.txt` | Defines `sparse_lu_ortho` as an explicit static target and installs `SparseTargets.cmake`, CMake package files, headers, generated version header, and `sparse.pc`. |
| `cmake/SparseConfig.cmake.in` | Includes installed `SparseTargets.cmake`. |
| `sparse.pc.in` | Owns static archive `sparse.pc` metadata text. |
| `tests/test_cmake_install.sh` | Unix local CMake install/export comparison point. |
| `scripts/static_package_deferral_check.sh` | Linux/macOS reviewed package-contract static deferral guard. |

## Installed Windows Package Layout

| Installed Path | Expected Meaning | Day 7 Design |
| --- | --- | --- |
| `lib/sparse_lu_ortho.lib` | Static library artifact for MSVC consumers. | Keep required static library file check. |
| `include/sparse/*.h` | Installed public headers plus generated version header. | Keep fixed 19-header count and add explicit `sparse_version.h` presence check. |
| `lib/cmake/Sparse/SparseConfig.cmake` | Installed package entry point. | Keep required file check. |
| `lib/cmake/Sparse/SparseConfigVersion.cmake` | Installed package version contract. | Keep required file check and verify behavior through exact/mismatch consumers. |
| `lib/cmake/Sparse/SparseTargets.cmake` | Imported target declaration and include metadata. | Keep positive `STATIC IMPORTED` and install-prefix include checks. |
| `lib/cmake/Sparse/SparseTargets-release.cmake` | Visual Studio Release imported artifact location. | Keep `IMPORTED_LOCATION_RELEASE` and installed `.lib` path checks. |
| `lib/pkgconfig/sparse.pc` | Installed static archive metadata. | Add exact `Name`, `Version`, `Cflags`, and `Libs` text checks without executing `pkg-config`. |

## CMake Imported-Target Rules

| Rule | Concrete PowerShell Assertion |
| --- | --- |
| Imported target is static. | `$targetsText.Contains('add_library(Sparse::sparse_lu_ortho STATIC IMPORTED)')` |
| Include metadata is install-prefix relative. | `$targetsText.Contains('INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"')` |
| Release imported location is present. | `$targetsReleaseText.Contains('IMPORTED_LOCATION_RELEASE')` |
| Release imported location points at installed static `.lib`. | `$targetsReleaseText.Contains('${_IMPORT_PREFIX}/lib/sparse_lu_ortho.lib')` |
| No source/build paths leak into installed package files. | Reject native and slash-normalized source/build paths in combined CMake package text. |
| No shared imported metadata appears. | Reject `SHARED IMPORTED`, `MODULE IMPORTED`, and imported `.so`, `.dylib`, or `.dll` locations. |

The Windows lane should continue to inspect `SparseTargets-release.cmake`
because the hosted workflow uses the Visual Studio multi-config generator and
installs the `Release` configuration. The Unix local script should continue to
inspect `SparseTargets-noconfig.cmake` for single-config generators.

## Shared-Artifact Rejection Rules

| Surface | Rule |
| --- | --- |
| Installed tree | No installed `.dll` files under the prefix. |
| CMake package metadata | No shared/module imported targets and no imported `.so`, `.dylib`, or `.dll` paths. |
| Public claim wording | No shared-library, dynamic ABI, runtime-loader, package-manager, or broad Windows parity claim. |

Day 8 should not try to infer whether a `.lib` is an import library from
binary content. The reviewed claim is instead backed by explicit `STATIC
IMPORTED` target metadata plus absence of DLL/shared imported metadata.

## `sparse.pc` Metadata Rules

Windows should continue to install and inspect `sparse.pc` as static package
metadata only. It should not execute `pkg-config` on Windows.

| Rule | Concrete PowerShell Assertion |
| --- | --- |
| Package name is stable. | require `Name: sparse` |
| Description states static archive metadata. | require `Description: Static archive package metadata for sparse linear algebra` |
| Version matches repository `VERSION`. | require `Version: $version` |
| Cflags are include-variable based. | require `Cflags: -I${includedir}` |
| Libs identify the static archive link surface. | require `Libs: -L${libdir} -lsparse_lu_ortho -lm` prefix or exact line with optional trailing whitespace handled. |
| Private dependencies are absent. | reject `Libs.private:` |
| Unsupported wording is absent. | reject shared, soname, dylib, dll, abi, package-manager, and ecosystem package-manager tokens. |

These are text checks over installed metadata. They do not claim Windows
`pkg-config --exists`, `--cflags`, `--libs`, `--static`, `--modversion`, or
downstream compile/link/run parity.

## Header And Version Expectations

| Check | Decision |
| --- | --- |
| Header count | Keep fixed count of 19 for Sprint 149 because the reviewed support contract currently expects 19 installed headers. |
| Version header | Add explicit `include/sparse/sparse_version.h` presence check so version metadata is not hidden inside aggregate header count. |
| CMake version file | Keep `SparseConfigVersion.cmake` file presence and behavior checks. |
| Exact-version behavior | Keep generated installed CMake consumer with `find_package(Sparse $version EXACT REQUIRED)`, configure/build/run, and output checks. |
| Mismatch-version behavior | Keep lower same-major generated project and require configure failure. |

If future source work changes the header count, the workflow should update the
fixed count intentionally in the same change that updates install docs and
evidence.

## Failure-Message Plan

| Failure Area | Message Style |
| --- | --- |
| Package files | `Expected installed package file at <path>.` |
| Static artifact shape | `Expected installed static library at <path>.` / `Unexpected shared-library artifacts installed: ...` |
| CMake package metadata | `Installed CMake package target ...` or `Installed CMake package metadata ...` |
| `sparse.pc` metadata | `Installed sparse.pc ...` |
| Source/build leaks | `Installed CMake package metadata leaked source/build path: <path>` |
| Downstream consumers | Keep existing `Installed example ...`, `Exact version consumer ...`, and mismatch-version messages. |

Metadata failures should occur before downstream configure/build/run steps so
hosted CI points maintainers at package export defects before consumer defects.

## Day 8 Implementation Targets

Day 8 should update `.github/workflows/windows-ci.yml` to add:

1. explicit installed `include/sparse/sparse_version.h` check;
2. `sparse.pc` `Name: sparse` check;
3. `sparse.pc` exact `Version: $version` check;
4. `sparse.pc` `Cflags: -I${includedir}` check;
5. `sparse.pc` static archive `Libs:` check;
6. local YAML/whitespace/unsupported-claim review.

No `.c` or `.h` changes are expected.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Each strengthened check has a concrete command or PowerShell assertion. | Complete | CMake, shared-artifact, `sparse.pc`, header, and version tables define concrete assertions. |
| Text checks avoid unsupported package-manager or shared-ABI wording. | Complete | `sparse.pc` rules inspect metadata without claiming Windows `pkg-config` execution or shared ABI support. |
| Downstream consumer proof remains separate from package-file inspection. | Complete | Failure-message plan keeps metadata checks before downstream consumer steps. |
