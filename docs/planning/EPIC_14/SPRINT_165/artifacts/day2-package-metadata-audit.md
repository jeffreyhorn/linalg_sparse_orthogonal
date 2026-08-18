# Sprint 165 Day 2 Package Metadata Audit

## Purpose

Day 2 audits the current CMake, Makefile, pkg-config, install-validation, CI,
and documentation package surfaces for unsupported wording or metadata. This
is an audit and classification day; it does not change package behavior.

## Source Files Inspected

| Surface | Files | Classification |
| --- | --- | --- |
| CMake package target and install/export metadata | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in` | Supported static archive contract with guard coverage |
| Make install and pkg-config metadata | `Makefile`, `sparse.pc.in` | Supported Unix-side static archive contract |
| Local package validation | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh` | Supported proof owners with explicit non-claim checks |
| Hosted package validation | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml` | Linux/macOS reviewed package lanes plus Windows CMake-first package lane |
| Public package docs | `README.md`, `INSTALL.md` | Supported package guidance with shared-library and ABI non-claims |
| Maintainer package docs | `docs/maintainer_guide.md` | Maintainer proof interpretation and package/ABI policy owner |

## CMake Metadata Audit

| Finding | Evidence | Classification | Day 3 Implication |
| --- | --- | --- | --- |
| `BUILD_SHARED_LIBS=ON` is rejected at configure time. | `CMakeLists.txt` top-level guard emits a fatal error naming static archive package surface, shared-library deferral, dynamic ABI policy, platform loader metadata, installed shared consumer proof, and runtime-loader validation blockers. | Supported contract | Preserve as fail-closed shared deferral guard. |
| Library target is explicitly static. | `add_library(sparse_lu_ortho STATIC ...)` in `CMakeLists.txt`. | Supported contract | Static target declaration should remain directly checked. |
| CMake install destination is archive-only. | `install(TARGETS sparse_lu_ortho ... ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})`. | Supported contract | Guard should continue rejecting `RUNTIME DESTINATION` and `LIBRARY DESTINATION` until shared support is selected. |
| Installed CMake package uses generated export files and exact package version compatibility. | `install(EXPORT SparseTargets ...)`, `configure_package_config_file(...)`, and `write_basic_package_version_file(... COMPATIBILITY ExactVersion)`. | Supported contract | Exact package version metadata must not be described as dynamic ABI support. |
| `SparseConfig.cmake.in` is minimal. | Template only includes `SparseTargets.cmake` and calls `check_required_components(Sparse)`. | Supported contract | Audit generated outputs rather than adding policy prose to the config template. |

## pkg-config Metadata Audit

| Finding | Evidence | Classification | Day 3 Implication |
| --- | --- | --- | --- |
| `sparse.pc` is static archive scoped. | `sparse.pc.in` description is `Static archive package metadata for sparse linear algebra`. | Supported contract | Keep exact description checked by local and hosted package lanes. |
| Link flags describe the current installed static archive surface. | `sparse.pc.in` emits `Libs: -L${libdir} -lsparse_lu_ortho -lm @SPARSE_PC_LIBS_EXTRA@`. | Supported contract | Downstream proof should keep validating installed link flags rather than broad package-manager behavior. |
| No private dependency stanza exists in the template. | `sparse.pc.in` has no `Libs.private`. | Supported contract | Existing checks should keep rejecting `Libs.private` under the current self-contained link contract. |
| Unsupported package/ABI wording is absent from installed `.pc` validation. | `tests/test_install.sh`, `tests/test_cmake_install.sh`, and Windows workflow reject or inspect terms such as shared, soname, dylib, dll, abi, and package-manager names. | Supported contract | Preserve term scans while avoiding false positives in non-metadata docs. |

## Install And Downstream Proof Audit

| Proof Owner | Current Coverage | Classification |
| --- | --- | --- |
| `tests/test_install.sh` | Make clean/install/uninstall, static archive, no shared artifacts, installed header count, `sparse.pc`, pkg-config exact version, prefix/libdir/includedir, cflags/libs, static libs parity, no `Libs.private`, static archive description, absent unsupported package/ABI terms, basic downstream consumer, maintained example source, uninstall cleanup. | Supported Unix-side Make/pkg-config proof |
| `tests/test_cmake_install.sh` | CMake configure/build/install, static archive, no shared artifacts, installed headers, CMake package files, static imported target, no shared imported metadata, no unsupported loader/static-shared selector metadata, install-prefix include/archive paths, source/build path leak checks, static `.pc` description and non-claim scan, installed `find_package(Sparse)` example, exact-version success, mismatched-version rejection, pkg-config version. | Supported Unix-side CMake install/export proof |
| `scripts/static_package_deferral_check.sh` | `BUILD_SHARED_LIBS=ON` rejection, explicit static target, archive install metadata, static `.pc` description, no export/import macros, no shared ABI CMake metadata, no static/shared package selectors, public support wording, Windows package non-claim wording, no unselected Windows package execution. | Supported static-first drift guard |
| `examples/cmake_example/` | Maintained installed CMake consumer used by local and hosted install/downstream checks. | Supported downstream fixture |

## CI Package Coverage Map

| Workflow | Package Coverage | Classification |
| --- | --- | --- |
| `.github/workflows/ci.yml` | Linux reviewed static-first package contract runs `tests/test_install.sh`, `tests/test_cmake_install.sh`, and `scripts/static_package_deferral_check.sh`. | Supported reviewed Linux package proof |
| `.github/workflows/macos-ci.yml` | macOS reviewed Make install/pkg-config proof runs `tests/test_install.sh`; reviewed CMake install/export proof runs `tests/test_cmake_install.sh` and `scripts/static_package_deferral_check.sh`. | Supported reviewed macOS static archive proof |
| `.github/workflows/windows-ci.yml` | Windows reviewed CMake install/downstream lane installs static `.lib`, verifies headers, CMake package files, static imported metadata, source/build path absence, no DLL/shared imported metadata, no unsupported loader/static-shared selectors, metadata-only `sparse.pc` content, generated and maintained CMake consumers, exact-version success, and mismatched-version rejection. | Supported Windows CMake-first package proof with explicit Make/pkg-config non-claims |

## Unsupported Wording Register

No immediate unsupported package wording requiring a Day 2 code/doc change was
found in the focused live package surfaces. The current text consistently
treats the maintained package surface as static-first and describes
shared-library packaging, dynamic ABI compatibility, runtime-loader behavior,
package-manager distribution, Windows Makefile parity, and Windows
`pkg-config` execution parity as non-claims or deferred product decisions.

The terms that should remain guarded during later edits are:

- `shared` and `shared-library`
- `ABI` and `dynamic ABI`
- `SONAME`
- `dylib`
- `dll`
- `runtime-loader`
- `Libs.private`
- `BUILD_SHARED_LIBS`
- package-manager names such as Homebrew, apt, dnf, pacman, vcpkg, and Conan
- Windows Makefile parity
- Windows `pkg-config` execution parity

## Validation Gap Register

| Gap | Current Owner | Classification | Follow-up |
| --- | --- | --- | --- |
| Generated CMake export files are validated through install scripts, not committed snapshots. | `tests/test_cmake_install.sh`, Windows workflow | Supported validation model | Day 3 should keep guard design based on generated install output, not source-controlled generated files. |
| Windows `sparse.pc` is inspected but `pkg-config` is not executed. | `.github/workflows/windows-ci.yml` | Deferred product decision | Preserve as metadata-only until a provider and downstream execution proof are selected. |
| Windows Makefile install/uninstall is not validated. | `.github/workflows/windows-ci.yml`, `docs/maintainer_guide.md` | Deferred product decision | Preserve explicit non-claim. |
| Package-manager distribution is not validated. | README, INSTALL, maintainer guide | Deferred product decision | Keep package-manager names out of package metadata and claim surfaces. |
| Shared-library install/export and loader metadata are intentionally absent. | `CMakeLists.txt`, install scripts, static deferral guard | Deferred product decision | Keep fail-closed rejection and absence checks. |

## Day 3 Guard-Design Handoff

Day 3 should design guard hardening around these concrete owners:

1. Keep `BUILD_SHARED_LIBS=ON` as a configure-time failure and preserve the
   blocker wording checked by `scripts/static_package_deferral_check.sh`.
2. Keep CMake install metadata archive-only until shared-library support has a
   reviewed product decision.
3. Keep installed CMake package checks focused on generated output:
   `STATIC IMPORTED`, install-prefix include/archive paths, no source/build
   leaks, no shared imported metadata, and no loader/static-shared selector
   metadata.
4. Keep `sparse.pc` static archive scoped with exact package metadata checks
   and no `Libs.private` under the current self-contained static link
   contract.
5. Keep Windows package validation CMake-first, with `sparse.pc`
   metadata-only inspection and explicit non-claims for Windows Makefile and
   `pkg-config` execution parity.

## Validation Notes

Day 2 changed planning documentation only. No `.c` or `.h` files were changed,
so `make format`, `make lint`, and `make test` are not required for Day 2.

## Completion Check

- CMake and pkg-config metadata behavior is source-backed.
- Unsupported package and ABI claim terms have file/check owners.
- The audit distinguishes supported static archive behavior from deferred
  shared-library, package-manager, dynamic ABI, runtime-loader, and Windows
  parity product decisions.
