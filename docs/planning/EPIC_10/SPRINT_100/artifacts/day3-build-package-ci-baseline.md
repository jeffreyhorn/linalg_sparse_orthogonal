# Sprint 100 Day 3 Build, Package & CI Baseline

## Purpose

Day 3 records the current build, package, CI, and platform proof surfaces
before Epic 10 implementation sprints change them. This artifact should be the
initial support-boundary input for Sprint 108 and a reference for all earlier
sprints that touch build, workflow, package, or platform wording.

## Build Surface Map

| surface | owner | current role |
|---|---|---|
| Makefile default build | `Makefile` | local static library build and primary Unix developer workflow |
| Makefile tests | `Makefile`, `tests/*.c` | full Makefile test execution path |
| Makefile reviewed compile path | `make quality-review-compile` | `format-check`, `source-list-check`, `lint` |
| Makefile reviewed path | `make quality-review` | `format-check`, `lint`, `test`, `deadcode-check` |
| strongest local reviewed path | `make quality-review-full` | `quality-review` plus `quality-review-cmake` |
| CMake build | `CMakeLists.txt` | cross-platform static library, tests, benchmarks, examples, install/export |
| CMake reviewed parity | `make quality-review-cmake` | configure, clean build, `ctest -N`, Make/CMake count parity, full `ctest` |
| source-list guard | `make source-list-check` | checks Make/CMake source membership drift |
| dead-code report/check | `make deadcode-report`, `make deadcode-check` | report generation and completeness gate, not a zero-findings gate |

## Package and Install Surface Map

The maintained install contract is static-first.

| surface | owner | current proof |
|---|---|---|
| Make install | `make install PREFIX=...` | installs static archive, headers, generated version header, and `sparse.pc` |
| Make uninstall | `make uninstall PREFIX=...` | removes static archive, headers, generated version header, and `sparse.pc` |
| pkg-config metadata | `sparse.pc.in` | generated as `sparse.pc` for installed static archive consumers |
| CMake install | `cmake --install` via `CMakeLists.txt` | installs static archive, headers, exported target, package config/version files, and `sparse.pc` |
| exported CMake target | `Sparse::sparse_lu_ortho` | installed consumer target for `find_package(Sparse)` |
| CMake package version | `SparseConfigVersion.cmake` | exact-version package behavior |
| Make install validation | `tests/test_install.sh` | Make install/uninstall plus `pkg-config` downstream consumer proof |
| CMake install validation | `tests/test_cmake_install.sh` | CMake install/export, `find_package(Sparse)`, exact-version behavior, and installed consumer proof |

### Make Install Validation Scope

`tests/test_install.sh` validates:

- static library installed
- no shared-library artifacts installed
- public headers plus generated version header installed
- `sparse.pc` installed
- `pkg-config --cflags`, `--libs`, and `--modversion`
- basic installed consumer compiles, links, and runs
- maintained example source compiles, links, and runs through `pkg-config`
- uninstall removes library, headers, and `sparse.pc`

Inherited post-Epic-9 result: `14` passed, `0` failed.

### CMake Install Validation Scope

`tests/test_cmake_install.sh` validates:

- CMake configure, build, and install
- static library installed
- no shared-library artifacts installed
- headers installed
- `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and
  `SparseTargets.cmake` installed
- `sparse.pc` installed
- `examples/cmake_example` configures with `find_package(Sparse)`, builds,
  and runs
- exact installed package version is accepted
- mismatched package version is rejected when a lower same-major version can
  be formed
- `pkg-config --modversion` matches `VERSION`

Inherited post-Epic-9 result: `16` passed, `0` failed, `0` skipped.

## CI Lane Summary

| workflow | job | authority | command or surface |
|---|---|---|---|
| `.github/workflows/ci.yml` | Linux supplemental runtime and bench-fast path | supplemental | `make test`, `make sanitize`, `make asan`, `make bench-build`, `make bench-fast` |
| `.github/workflows/ci.yml` | Linux enforced reviewed CMake parity path | reviewed | `make quality-review-cmake` |
| `.github/workflows/ci.yml` | Linux supplemental ThreadSanitizer coverage | supplemental | TSan thread tests plus TSan+OpenMP eigensolver subset |
| `.github/workflows/ci.yml` | Linux enforced reviewed Makefile compile-quality path | reviewed | `make quality-review-compile` |
| `.github/workflows/ci.yml` | Linux enforced dead-code report and completeness path | reviewed support lane | `make deadcode-report`, `make deadcode-check` |
| `.github/workflows/ci.yml` | Linux supplemental coverage report | supplemental | `make coverage` |
| `.github/workflows/macos-ci.yml` | macOS Apple Clang reviewed path | reviewed narrower platform lane | `make quality-review-compile`, `make quality-review-cmake`, `make wall-check`, `make sanitize` |
| `.github/workflows/macos-ci.yml` | macOS Homebrew GCC leg | supplemental | direct build/test/wall-check with `gcc-15` |
| `.github/workflows/macos-ci.yml` | macOS install/pkg-config confidence path | supplemental | `bash tests/test_install.sh` |
| `.github/workflows/windows-ci.yml` | Windows MSVC CMake consumer subset | reviewed subset | configure, build, `ctest -N`, full `ctest`; expected count `51` |

## Platform Tier Draft

| platform | current tier | supported reading |
|---|---|---|
| Linux Ubuntu | strongest reviewed source of truth | reviewed CMake parity, reviewed Makefile compile-quality, dead-code report/check, plus supplemental runtime, benchmark, sanitizer, and coverage confidence |
| macOS Apple Clang | reviewed narrower platform lane | compile-quality, CMake parity, wall-check, and sanitizer are reviewed for macOS; install/pkg-config confidence remains supplemental |
| macOS Homebrew GCC | supplemental second-compiler confidence | broadens compile/runtime confidence but is not the macOS reviewed source of truth |
| Windows MSVC | reviewed CMake-first subset | validates CMake configure/build/discovery/execution for the maintained Windows consumer story |

## Current Platform and Package Non-Claims

| non-claim | current reason |
|---|---|
| shared-library-first package contract | install proof is static-first and explicitly rejects unexpected shared-library artifacts |
| dynamic ABI guarantee | no ABI-versioning or shared-library runtime-loader proof exists |
| package-manager ecosystem integration | no Homebrew, vcpkg, conda, Debian, RPM, or similar recipe proof is maintained |
| Windows Makefile parity | Windows reviewed lane is CMake-first with MSVC |
| Windows install-validation parity | no reviewed Windows install/export script lane is maintained |
| symmetric Linux/macOS/Windows reviewed parity | platform lanes intentionally have different reviewed scopes |
| broad backend-neutral acceleration maturity | backend work remains bounded; package proof does not imply vendor backend parity |
| portable timing superiority | benchmark lanes are local calibration, not portable product claims |

## Day 3 Interpretation

The repository has a real package story and a mature CI story, but the support
contract is intentionally tiered:

- Linux is the broadest reviewed truth source.
- macOS has a reviewed Apple Clang lane plus supplemental confidence paths.
- Windows has a reviewed CMake-first subset with an expected test count of
  `51`, not broad Makefile or install parity.
- Install/export validation is strong locally, but not symmetric across every
  CI platform.
- Static-first packaging is a maintained claim; shared-library or dynamic ABI
  maturity is not.

Sprint 108 should revisit whether Epic 10 wants to keep this static-first tier
as the permanent support contract or add explicit shared-library/ABI proof.

