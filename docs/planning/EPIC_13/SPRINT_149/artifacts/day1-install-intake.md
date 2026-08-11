# Sprint 149 Day 1: Install Intake

## Purpose

Establish the Sprint 149 baseline for Windows install-validation parity by
separating current reviewed Linux/macOS package proof from the current
supplemental Windows CMake install/downstream lane.

Sprint 149 is not allowed to infer install-validation parity from Sprint 148's
Windows CTest promotion. It must make a separate package-lane product decision.

## Sources Reviewed

| Source | Use In Day 1 |
| --- | --- |
| `docs/planning/EPIC_13/PROJECT_PLAN.md` | Sprint 149 scope, items, deliverables, and 166-hour estimate. |
| `docs/planning/EPIC_13/SPRINT_148/artifacts/day14-closeout-handoff.md` | Confirms Sprint 148 closed staged Windows CTest portability but handed Windows install-validation parity to Sprint 149. |
| `docs/planning/EPIC_13/SPRINT_148/artifacts/day12-docs-alignment.md` | Confirms public docs still preserve Windows install/parity non-claims after test promotion. |
| `.github/workflows/ci.yml` | Linux reviewed static-first package-contract lane. |
| `.github/workflows/macos-ci.yml` | macOS reviewed static-first Make install/`pkg-config` and CMake install/export lanes. |
| `.github/workflows/windows-ci.yml` | Windows reviewed CMake CTest lane and supplemental CMake install/downstream lane. |
| `tests/test_install.sh` | Unix-side Make install/uninstall plus `pkg-config` proof owner. |
| `tests/test_cmake_install.sh` | Unix-side CMake install/export plus installed `find_package(Sparse)` proof owner. |
| `INSTALL.md` | Public install-support interpretation and non-claims. |
| `docs/maintainer_guide.md` | Maintainer ownership rules for package proof and platform interpretation. |

## Current Cross-Platform Install-Proof Inventory

| Platform | Lane | Current Tier | Commands / Checks | Day 1 Interpretation |
| --- | --- | --- | --- | --- |
| Linux | `.github/workflows/ci.yml::package-contract` | Reviewed | `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh`; `bash scripts/static_package_deferral_check.sh` | Strongest package-contract comparison point for Sprint 149. |
| macOS | `.github/workflows/macos-ci.yml::install-and-pkgconfig` | Reviewed | `bash tests/test_install.sh` | Reviewed macOS static-first Make install and `pkg-config` proof. |
| macOS | `.github/workflows/macos-ci.yml::cmake-install-export` | Reviewed | `bash tests/test_cmake_install.sh`; static-first package deferral proof | Reviewed macOS CMake install/export proof for the static archive package contract. |
| Windows | `.github/workflows/windows-ci.yml::build-and-test` | Reviewed CMake subset | CMake configure/build; `ctest -N`; full hosted CTest; `EXPECTED_WINDOWS_CTEST_COUNT=59` | Reviewed Windows CMake-first consumer proof only, not package parity. |
| Windows | `.github/workflows/windows-ci.yml::install-and-downstream` | Supplemental | CMake install; static `.lib`; no DLLs; 19 headers; CMake package files; `sparse.pc` text checks; downstream CMake example; exact-version consumer; mismatch-version rejection | Sprint 149 decision target: promote, split, rename, or retain as supplemental. |

## Unix-Side Install Script Coverage

`tests/test_install.sh` owns local Unix-side Make install/uninstall plus
`pkg-config` behavior:

| Coverage Area | Current Proof |
| --- | --- |
| Install command | `make clean`; `make install PREFIX="$PREFIX"` |
| Static archive shape | `libsparse_lu_ortho.a` exists |
| Shared artifact rejection | rejects `.so`, `.so.*`, `.dylib`, and `.dll` in install tree |
| Header install | installed header count equals source headers plus generated `sparse_version.h` |
| `sparse.pc` presence | installed under `lib/pkgconfig` |
| `pkg-config` resolution | package exists, exact version works, prefix/libdir/includedir variables match install prefix |
| Compile flags | `pkg-config --cflags sparse` points at installed include directory |
| Link flags | `pkg-config --libs sparse` returns installed static archive link flags |
| Static flags | `pkg-config --libs --static sparse` matches current self-contained link flags |
| Metadata wording | no `Libs.private`; static archive description; no unsupported package/ABI wording |
| Downstream consumers | generated basic consumer and maintained `examples/cmake_example/main.c` compile/link/run through `pkg-config` |
| Cleanup | `make uninstall` removes library, headers, and `sparse.pc` |

`tests/test_cmake_install.sh` owns local Unix-side CMake install/export and
installed `find_package(Sparse)` proof. Day 2 should compare it directly to
the Windows supplemental PowerShell lane because that is the closest
cross-platform equivalent.

## Windows Supplemental Lane Snapshot

The current Windows package lane is:

- workflow: `.github/workflows/windows-ci.yml`
- job id: `install-and-downstream`
- job name: `Windows supplemental CMake install/downstream confidence path`
- runner: `windows-2022`
- generator: `Visual Studio 17 2022`
- architecture: `x64`
- build config: `Release`
- install prefix: `$env:RUNNER_TEMP\sparse-install`

Current assertions:

| Evidence Area | Current Assertion |
| --- | --- |
| Install route | configure, build, and `cmake --install` through Visual Studio generator |
| Static archive | installed `lib/sparse_lu_ortho.lib` exists |
| Shared artifact rejection | no installed `.dll` files |
| Headers | exactly 19 headers under `include/sparse` |
| Package files | `SparseConfig.cmake`, `SparseConfigVersion.cmake`, `SparseTargets.cmake`, and `lib/pkgconfig/sparse.pc` exist |
| CMake metadata | rejects `SHARED IMPORTED`, `MODULE IMPORTED`, and imported `.so`, `.dylib`, or `.dll` locations |
| `sparse.pc` static text | requires `Description: Static archive package metadata for sparse linear algebra` |
| Unsupported wording | rejects `Libs.private`, shared-library, ABI, package-manager, and ecosystem package-manager wording |
| Normal consumer | configures, builds, and runs `examples/cmake_example` through `CMAKE_PREFIX_PATH` |
| Exact-version consumer | generated project uses `find_package(Sparse $version EXACT REQUIRED)`, builds, and runs |
| Mismatch-version behavior | generated project using a lower same-major version must fail configure |

## Evidence Fields For Sprint 149

| Evidence Category | Required Fields |
| --- | --- |
| Package files | platform, workflow/job, install route, prefix, static library path, header count, package file list, shared-artifact rejection rule |
| CMake metadata | target type, imported archive location, installed include prefix, source/build path leak check, shared imported metadata rejection, unsupported wording rejection |
| `sparse.pc` metadata | file presence, static archive description, no `Libs.private`, no shared/ABI/package-manager wording, whether `pkg-config` execution is in scope |
| Downstream consumer | source project, configure command, build command, executable path, runtime output checks, installed-package path used |
| Version behavior | exact version requested, exact-version configure/build/run result, mismatch version chosen, mismatch configure failure result |
| Unsupported claims | Windows Makefile parity, Windows `pkg-config` parity, shared-library support, dynamic ABI, runtime-loader behavior, package-manager support, broad Windows parity |
| Hosted proof | workflow name, runner, job URL or pending PR status, pass/fail status, residual if unavailable locally |

## Day 2 Handoff

Day 2 should compare the Windows supplemental lane to Linux/macOS reviewed
package proof one assertion at a time:

1. Identify checks that are true CMake-package equivalents across platforms.
2. Identify Unix-only checks that must remain non-parity on Windows.
3. Identify Windows package checks that are already stronger than the
   Unix-side comparison point.
4. Identify missing checks before any promotion decision is made.
5. Preserve the distinction between CMake-first installed consumer confidence
   and Makefile/`pkg-config` parity.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Sprint scope is tied to current repository files and Sprint 148 handoff. | Complete | Sources reviewed table links Sprint 149 scope to current workflow, script, doc, and Sprint 148 handoff files. |
| Linux/macOS reviewed proof and Windows supplemental proof are separated. | Complete | Cross-platform inventory keeps reviewed Linux/macOS lanes distinct from the supplemental Windows lane. |
| Every install-validation evidence category has an owner and recording format. | Complete | Evidence fields table defines package files, metadata, consumer, version, unsupported-claim, and hosted-proof fields. |
