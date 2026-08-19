# Sprint 170 Day 6: CMake Package Feasibility Review

## Purpose

Review the CMake static target, install/export metadata, generated package
files, platform-specific downstream proof, and shared-library implications for
the Sprint 170 product decision.

## CMake Static-Target Inventory

The top-level CMake build intentionally rejects shared-library packaging:

```cmake
if(BUILD_SHARED_LIBS)
    message(FATAL_ERROR ...)
endif()
```

The rejection names the missing prerequisites explicitly:

- export/import policy;
- symbol visibility policy;
- dynamic ABI policy;
- Linux SONAME metadata;
- macOS install-name/RPATH metadata;
- Windows DLL/import-library behavior;
- installed shared consumer proof;
- runtime-loader validation.

The maintained target remains an explicit static archive:

```cmake
add_library(sparse_lu_ortho STATIC ...)
```

Target metadata:

- `EXPORT_NAME sparse_lu_ortho`
- `OUTPUT_NAME sparse_lu_ortho`
- public build/install include directories;
- optional public OpenMP and pthread link dependencies when selected;
- `m` linked on non-MSVC platforms;
- `${CMAKE_DL_LIBS}` linked on Unix non-Apple platforms when available.

There is no CMake shared target, shared option, export/import macro,
visibility preset, runtime destination, library destination, SONAME,
install-name, RPATH, DLL, or import-library configuration.

## Install And Export Package Metadata Notes

CMake install/export behavior is static archive scoped:

| Metadata area | Current behavior |
| --- | --- |
| Installed target | `install(TARGETS sparse_lu_ortho EXPORT SparseTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})` |
| Installed headers | Checked-in `include/*.h`, excluding `*.h.in` and checked-in/generated-name collisions. |
| Generated header | Installs generated `sparse_version.h` under `${CMAKE_INSTALL_INCLUDEDIR}/sparse`. |
| CMake package config | Installs `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and `SparseTargets.cmake`. |
| CMake namespace | Exports `Sparse::sparse_lu_ortho`. |
| Package version compatibility | `write_basic_package_version_file(... COMPATIBILITY ExactVersion)`. |
| pkg-config metadata | Installs generated `sparse.pc` from the same static archive template used by Make. |

`cmake/SparseConfig.cmake.in` is intentionally minimal:

```cmake
@PACKAGE_INIT@

include("${CMAKE_CURRENT_LIST_DIR}/SparseTargets.cmake")

check_required_components(Sparse)
```

It does not expose components, shared/static selectors, ABI variables, runtime
loader metadata, or package-manager integration.

## Maintained CMake Install Proof

`tests/test_cmake_install.sh` is the local Unix-side CMake install/export proof.
It verifies:

- configure, build, and install;
- installed static archive;
- no installed `.so`, `.so.*`, `.dylib`, or `.dll` artifacts;
- installed public headers plus generated `sparse_version.h`;
- installed `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and
  `SparseTargets.cmake`;
- installed `sparse.pc`;
- `Sparse::sparse_lu_ortho` is a `STATIC IMPORTED` target;
- imported target include dirs and archive location use the install prefix;
- package metadata has no source-tree or build-tree path leaks;
- package metadata has no shared-library imported metadata;
- package metadata has no unsupported loader or shared/static selector
  metadata;
- `sparse.pc` remains static archive scoped and free of unsupported package or
  ABI wording;
- installed downstream consumer configure/build/run with `find_package`;
- exact installed version configure/build/run;
- mismatched same-major version rejection;
- installed `pkg-config` version metadata on Unix.

This proof validates a real CMake package for installed static consumers. It
does not validate dynamic loader behavior, shared consumers, ABI stability, or
package-manager distribution.

## Platform-Specific CMake Proof Boundaries

| Platform lane | CMake package evidence | Explicit boundary |
| --- | --- | --- |
| Linux reviewed package contract | Runs `tests/test_cmake_install.sh` in CI after installing CMake and `pkg-config`. | Static archive package only; no shared-library, dynamic ABI, runtime-loader, package-manager, or broad platform parity claim. |
| macOS reviewed CMake install/export | Runs `tests/test_cmake_install.sh` plus static deferral guard in macOS CI. | Static CMake package only; no shared-library, dynamic ABI, runtime-loader, package-manager, or broad macOS parity claim. |
| Windows reviewed CMake install/downstream | Configures/builds/installs with Visual Studio, checks installed `.lib`, headers, CMake package files, generated/maintained CMake consumers, exact version, mismatch rejection, and metadata-only `sparse.pc` inspection. | Windows remains CMake-first; no Windows Makefile parity, no Windows `pkg-config` command execution parity, no shared-library support, no dynamic ABI support, no runtime-loader behavior, no package-manager support, and no broad Windows parity claim. |

The Windows lane is the strongest cross-platform CMake-specific evidence, but
it is intentionally not equivalent to Unix Make/`pkg-config` proof.

## Shared-Library CMake Feasibility Risks

| Risk | Severity | Required future work |
| --- | --- | --- |
| Configure-time shared rejection | High | Replace or supplement the rejection only after a selected shared product path exists. |
| No export/import macro | High | Add public API decoration policy, including Windows `__declspec` behavior or a `.def` strategy. |
| No hidden visibility policy | High | Add hidden-by-default visibility and an allowlisted export surface. |
| No dynamic package metadata | High | Decide whether shared and static use one package, separate components, or separate package names. |
| No loader metadata | High | Add Linux SONAME, macOS install-name/RPATH, and Windows DLL/import-library install semantics. |
| No dynamic ABI version policy | High | Separate package/source version from ABI epoch and compatibility rules. |
| No installed shared consumer proof | High | Add downstream tests that link and run against installed shared artifacts. |
| No exported symbol allowlist checks | High | Compare platform exports against an approved public symbol list. |
| Dependency propagation differences | Medium | Decide how `m`, `dl`, pthread, and OpenMP dependencies are represented for dynamic consumers. |
| Static/shared coexistence ambiguity | Medium | Define install collision behavior, target names, and uninstall/package metadata ownership. |

## Package Metadata Guard Points

The following CMake-owned surfaces must continue rejecting unsupported ABI
claims unless the product decision changes:

- `BUILD_SHARED_LIBS=ON` configure behavior;
- `add_library(sparse_lu_ortho STATIC ...)`;
- `install(TARGETS ... ARCHIVE DESTINATION ...)` without runtime/shared
  destinations;
- `SparseTargets.cmake` as `STATIC IMPORTED`;
- `SparseConfigVersion.cmake` exact-version compatibility;
- absence of `SOVERSION`, `WINDOWS_EXPORT_ALL_SYMBOLS`,
  `C_VISIBILITY_PRESET`, `VISIBILITY_INLINES_HIDDEN`, `INSTALL_NAME_DIR`,
  `MACOSX_RPATH`, `IMPORTED_SONAME`, `IMPORTED_IMPLIB`, and static/shared
  component selectors;
- absence of source/build path leaks in installed package metadata;
- Windows metadata-only `sparse.pc` handling.

## Feasibility Finding

CMake is ready and well validated for the current static-first installed
package product. It is not ready for a shared-library ABI product without a
larger coordinated change across target type, symbol visibility, export/import
policy, loader metadata, package version semantics, platform CI, and
documentation.

For Sprint 170 synthesis, CMake evidence supports either:

1. keeping shared-library and dynamic ABI support explicitly deferred; or
2. creating a separate future epic/sprint sequence for a staged shared build
   with an allowlisted ABI and platform loader proof.

It does not support silently enabling `BUILD_SHARED_LIBS` or describing the
current CMake package version as binary ABI compatibility.

## Day 6 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| CMake static-target inventory | Complete | Mapped shared rejection, explicit static target, include/link metadata, and absent shared controls. |
| Install/export package metadata notes | Complete | Recorded installed target, headers, generated version header, package config/version files, target export, and `sparse.pc`. |
| Platform-specific CMake proof boundary notes | Complete | Separated Linux, macOS, and Windows evidence and non-claims. |
| Shared-library CMake feasibility risks | Complete | Listed missing export, visibility, loader, ABI, package, and consumer proof work. |
| Day 6 CMake-feasibility artifact | Complete | This file. |

## Validation

Day 6 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| CMake package behavior is mapped to the decision needs. | Complete | Static target, install/export package metadata, exact version behavior, and generated metadata are documented. |
| Platform evidence boundaries are preserved. | Complete | Linux/macOS Unix-side CMake proof and Windows CMake-first proof are scoped explicitly. |
| Feasibility notes are ready for decision synthesis. | Complete | The artifact supports static-first continuation unless a separate shared ABI proof stack is funded. |
