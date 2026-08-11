# Sprint 149 Day 2: Windows Package Audit

## Purpose

Compare the current Windows supplemental CMake install/downstream proof with
the reviewed Linux/macOS static-first package validation lanes, classify each
Windows check, and identify the proof gaps that Day 3 promotion criteria must
resolve.

## Current Proof Stack

| Surface | Current Tier | Proof Owner |
| --- | --- | --- |
| Linux static-first package contract | Reviewed | `.github/workflows/ci.yml::package-contract` |
| macOS Make install/`pkg-config` package proof | Reviewed | `.github/workflows/macos-ci.yml::install-and-pkgconfig` |
| macOS CMake install/export package proof | Reviewed | `.github/workflows/macos-ci.yml::cmake-install-export` |
| Windows CMake install/downstream confidence | Supplemental | `.github/workflows/windows-ci.yml::install-and-downstream` |

The Windows lane is closest to `tests/test_cmake_install.sh`, not to the full
Unix `tests/test_install.sh` Make install/`pkg-config` proof. Sprint 149 must
therefore avoid treating Windows CMake package confidence as Windows Makefile
or `pkg-config` parity.

## Windows Check Classification

| Windows Assertion | Linux/macOS Comparison | Classification | Audit Notes |
| --- | --- | --- | --- |
| Visual Studio CMake configure | `tests/test_cmake_install.sh` CMake configure | Equivalent | Same package route class, different generator/toolchain. |
| Visual Studio CMake build | `tests/test_cmake_install.sh` CMake build | Equivalent | Builds the installable static target before install. |
| `cmake --install` | `tests/test_cmake_install.sh` CMake install | Equivalent | Correct Windows install route; not a Makefile install proof. |
| `lib/sparse_lu_ortho.lib` exists | `libsparse_lu_ortho.a` exists | Equivalent | Platform-specific static archive name. |
| No installed `.dll` files | Unix no `.so`, `.so.*`, `.dylib`, `.dll` | Equivalent but narrower | Correct Windows shared-artifact check; Day 3 may decide whether to also inspect `.lib` metadata or imported-target files more directly. |
| Exactly 19 installed headers | Unix installed header count | Equivalent | Matches current public header count including generated version header. |
| CMake package files exist | `SparseConfig.cmake`, `SparseConfigVersion.cmake`, `SparseTargets.cmake` exist | Equivalent | Windows also checks `sparse.pc` file presence as metadata. |
| `SparseTargets` text lacks shared imported metadata | Unix CMake package shared metadata rejection | Equivalent but weaker | Rejects shared/module imports and `.so`/`.dylib`/`.dll` imported locations, but does not explicitly require `STATIC IMPORTED`. |
| `sparse.pc` file exists and has static archive description | Unix `sparse.pc` static metadata checks | Supplemental metadata | Useful metadata proof; not Windows `pkg-config` execution parity. |
| `sparse.pc` lacks `Libs.private` and unsupported package/ABI wording | Unix `sparse.pc` unsupported wording check | Supplemental metadata | Equivalent text guard, intentionally does not invoke Windows `pkg-config`. |
| Installed example configures/builds/runs with `CMAKE_PREFIX_PATH` | `cmake_example` find_package configure/build/run | Equivalent | Strong installed CMake consumer proof. |
| Exact-version consumer configures/builds/runs | `find_package(Sparse VERSION EXACT REQUIRED)` proof | Equivalent | Stronger than configure-only exact-version checks; matches current Unix script behavior. |
| Lower same-major mismatch version fails configure | mismatched-version rejection | Equivalent | Correct fail-closed package-version behavior. |

## Linux/macOS Reviewed Proof Comparison

| Reviewed Proof Area | Linux/macOS Coverage | Windows Current Coverage | Day 2 Classification |
| --- | --- | --- | --- |
| Make install/uninstall | `tests/test_install.sh` installs and uninstalls via Makefile | No Windows Makefile route | Intentionally non-parity |
| `pkg-config` execution | `pkg-config --exists`, variables, cflags/libs, static libs, modversion | Text-only `sparse.pc` checks | Intentionally non-parity for execution; supplemental metadata for file contents |
| CMake install/export | `tests/test_cmake_install.sh` configure/build/install | Visual Studio CMake configure/build/install | Equivalent |
| Static archive install | `.a` archive exists | `.lib` archive exists | Equivalent |
| Header install count | expected source headers plus generated version header | fixed count of 19 headers | Equivalent but fixed-count drift risk |
| CMake imported target type | explicit `STATIC IMPORTED` check in `SparseTargets.cmake` | shared/module rejection only | Missing direct positive check |
| Installed include prefix | checks `_IMPORT_PREFIX}/include` | not directly checked | Missing |
| Imported archive location | checks `_IMPORT_PREFIX}/lib/libsparse_lu_ortho.a` in `SparseTargets-noconfig.cmake` | not directly checked for `.lib` path | Missing |
| Source/build path leaks | rejects source and build directory paths in package files | not directly checked | Missing |
| Static package deferral guard | Linux and macOS run `scripts/static_package_deferral_check.sh` in reviewed package lanes | not run on Windows | Missing or intentionally non-Windows depending on Day 3 criteria |
| Basic generated downstream consumer | `tests/test_install.sh` builds a small generated consumer through `pkg-config` | not present | Missing only if Day 3 wants a non-example CMake consumer |
| Maintained example downstream consumer | both install scripts exercise maintained example | Windows exercises maintained CMake example | Equivalent |
| Exact-version consumer | CMake exact-version configure/build/run | Windows exact-version configure/build/run | Equivalent |
| Mismatch-version rejection | CMake mismatch configure fails | Windows mismatch configure fails | Equivalent |

## Windows-Equivalent Evidence Already Present

- CMake configure/build/install using the Windows-supported Visual Studio
  generator.
- Installed static library file under the install prefix.
- Installed public headers under `include/sparse`.
- Installed CMake package files under `lib/cmake/Sparse`.
- Installed `sparse.pc` as package metadata.
- Shared-artifact rejection for installed DLLs and shared imported metadata.
- Maintained installed CMake example configure/build/run via
  `CMAKE_PREFIX_PATH`.
- Exact-version installed CMake package configure/build/run.
- Mismatch-version fail-closed configure behavior.

## Intentionally Non-Parity Unix Checks

These checks must remain explicit non-claims unless Sprint 149 makes a separate
product decision with evidence:

- Windows Makefile install or uninstall parity.
- Windows execution of `tests/test_install.sh`.
- Windows `pkg-config --exists`, `--cflags`, `--libs`, `--static`, variable,
  or `--modversion` parity.
- Windows `pkg-config` downstream compile/link/run parity.
- Unix shell path, symlink, or `-ef` behavior.
- Package-manager distribution or package-manager resolver behavior.

## Proof Gaps Before Promotion Criteria

| Gap | Why It Matters | Candidate Day 3 Treatment |
| --- | --- | --- |
| No explicit positive `STATIC IMPORTED` check for the installed CMake target. | Shared metadata rejection is useful, but reviewed package promotion should positively assert the expected target type. | Required for reviewed Windows CMake install-validation promotion. |
| No installed include-prefix check in CMake metadata. | Unix CMake proof verifies installed targets point at `_IMPORT_PREFIX}/include` instead of a source-tree path. | Required or explicitly residualized. |
| No installed static archive location check in `SparseTargets-*.cmake`. | Existing `.lib` file check proves the file exists, not that the exported target points at it. | Required or explicitly residualized. |
| No source/build path leak check in installed CMake package files. | Relocatable installed packages should not embed source or build directories. | Required for reviewed CMake package confidence. |
| No Windows static package deferral guard execution. | Linux/macOS reviewed package lanes prove shared-library deferral and unsupported selectors remain absent. | Decide whether to port/run guard on Windows or keep as Linux/macOS reviewed package-contract scope. |
| No uninstall cleanup proof. | `tests/test_install.sh` validates cleanup; Windows CMake install lane currently does not validate uninstall semantics. | Likely non-parity unless CMake uninstall support exists. |
| No basic generated CMake downstream consumer separate from maintained example. | Unix Make/`pkg-config` script has both a generated basic consumer and maintained example. | Optional if exact-version generated CMake consumer is considered sufficient. |
| Header count is fixed at `19`. | Fixed count catches drift but requires manual workflow edits when headers change. | Decide whether to keep fixed contract or derive from source on Windows. |

## Duplicate Or Stronger Windows Checks

- Windows exact-version proof builds and runs a generated consumer, matching
  the Unix CMake install script's strongest version behavior.
- Windows mismatch-version proof fails closed at configure time, matching the
  Unix CMake install script.
- Windows output handling already normalizes PowerShell array output into text
  before matching example output.
- Windows `sparse.pc` text checks are intentionally metadata-only but are
  useful because the file is installed even when Windows `pkg-config` execution
  remains out of scope.

## Wording-Risk Register

| Risk | Problem | Required Guard |
| --- | --- | --- |
| Calling the Windows lane "package parity" | Could imply Makefile, Unix shell, or `pkg-config` parity. | Use "Windows CMake install/downstream" or a narrower reviewed phrase. |
| Calling text checks "`pkg-config` support" | The lane checks `sparse.pc` contents but does not execute `pkg-config`. | Say "`sparse.pc` metadata" unless execution is added. |
| Calling CMake installed consumer proof "package-manager support" | `find_package(Sparse)` uses installed CMake config files, not a package manager. | Keep package-manager support a non-claim. |
| Saying no DLLs means shared ABI is supported | The check proves absence of shared artifacts, not shared-library readiness. | Preserve shared-library and dynamic ABI deferral wording. |
| Promoting Windows install validation without hosted proof | Local review cannot prove hosted MSVC install behavior. | Require passing hosted Windows job evidence before reviewed wording lands. |
| Inferring install validation from CTest count | Sprint 148 CTest promotion is unrelated to package install proof. | Keep CTest and install-validation evidence in separate rows. |

## Day 3 Handoff

Promotion criteria should decide:

1. Whether the existing Windows lane can become a reviewed Windows CMake
   install-validation lane after closing the missing CMake metadata checks.
2. Whether static package deferral proof must run on Windows or can remain a
   Linux/macOS reviewed package-contract guard.
3. Whether `sparse.pc` metadata checks are required while Windows
   `pkg-config` execution remains explicitly out of scope.
4. Whether fixed header count is acceptable for reviewed Windows evidence.
5. Whether the exact-version generated consumer is sufficient as the
   non-example consumer proof.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Each Windows install check is classified as equivalent, supplemental, missing, or intentionally non-parity. | Complete | Windows check classification and proof-gap tables classify every current Windows assertion and major comparison point. |
| Unsupported Unix Makefile and `pkg-config` parity claims remain explicit. | Complete | Intentionally non-parity table and wording-risk register preserve those non-claims. |
| Remaining proof gaps are concrete enough to drive Day 3 criteria. | Complete | Proof-gap table names each missing assertion and candidate Day 3 treatment. |
