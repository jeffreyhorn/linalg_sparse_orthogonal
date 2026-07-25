# Sprint 134 Day 14 - Platform Tier Closeout and Handoff

## Purpose

Day 14 closes Sprint 134 by publishing the final platform support tiers,
staged-exclusion register, residual queue, validation summary, and Sprint 135
handoff material.

## Final Platform Support Truth

| Platform | Reviewed tier | Supplemental tier | Staged/deferred/non-claims |
| --- | --- | --- | --- |
| Linux | Makefile compile-quality, CMake parity, dead-code, and static-first package-contract lane. | Direct runtime, sanitizer, `bench-fast`, TSan, and coverage signals. | Shared-library packaging, dynamic ABI compatibility, runtime-loader behavior, and package-manager support remain non-claims. |
| macOS | Apple Clang compile-quality, CMake parity, wall-check, and sanitizer path. | Homebrew GCC second-compiler path, static-first Make install/`pkg-config`, and CMake install/export confidence. | Full reviewed macOS install/export parity, package-manager support, shared-library packaging, and dynamic ABI compatibility remain non-claims. |
| Windows | MSVC CMake configure/build/`ctest -N`/full `ctest` subset with `EXPECTED_WINDOWS_CTEST_COUNT=54`. | CMake install/downstream confidence for the CMake-first consumer story. | Separate reviewed install-validation parity, Windows Makefile parity, Windows `pkg-config`, package-manager support, shared-library packaging, dynamic ABI compatibility, and pthread/POSIX-backed staged tests remain non-claims or staged. |

## Implemented Workflow Changes

| Workflow | Sprint 134 result |
| --- | --- |
| `.github/workflows/ci.yml` | Added the Linux reviewed static-first package-contract lane running Make install/`pkg-config`, CMake install/export, and static deferral proof. |
| `.github/workflows/macos-ci.yml` | Added the supplemental macOS CMake install/export confidence lane while preserving the existing supplemental Make install/`pkg-config` lane. |
| `.github/workflows/windows-ci.yml` | Added the supplemental Windows CMake install/downstream confidence lane and reinforced staged blocker output for Windows CTest exclusions. |

## Documentation Changes

| Document | Sprint 134 result |
| --- | --- |
| `README.md` | CI summary now distinguishes Linux reviewed package contract, macOS supplemental package confidence, Windows supplemental install/downstream confidence, and staged Windows tests. |
| `INSTALL.md` | Platform and verification sections now describe reviewed/supplemental/local package confidence without widening non-claims. |
| `docs/maintainer_guide.md` | Maintainer notes now include final platform tiers, Windows CTest count, staged blockers, and promotion gates. |
| `docs/planning/EPIC_11/SPRINT_134/WORKING_NOTES.md` | Records the full Sprint 134 decision and validation trail. |

## Final Staged-Exclusion Register

| Staged item | Owner surface | Current blocker | Promotion gate |
| --- | --- | --- | --- |
| Windows `test_threads` | `CMakeLists.txt`, `tests/test_threads.c`, `.github/workflows/windows-ci.yml` | Direct pthread API dependency. | Windows-native thread test or portability wrapper plus intentional CTest count update and hosted MSVC configure/build/execute proof. |
| Windows `test_sprint4_integration` | `CMakeLists.txt`, `tests/test_sprint4_integration.c`, `.github/workflows/windows-ci.yml` | Direct pthread API dependency. | Windows-native integration equivalent or portability wrapper plus intentional CTest count update and hosted MSVC configure/build/execute proof. |
| Windows `test_fuzz` and bounded lifecycle property lane | `CMakeLists.txt`, `tests/test_fuzz.c`, `.github/workflows/windows-ci.yml`, `docs/maintainer_guide.md` | POSIX temp-file APIs through `<unistd.h>`, `mkstemps`, `close`, and `unlink`. | Portable temp-file abstraction or Windows-specific fuzz/property variant plus intentional CTest count update and hosted MSVC proof. |
| Windows Makefile parity | Makefile/wrapper surface | Windows support remains CMake-first. | Separate product decision and maintained Windows Makefile/wrapper proof. |

## Residual Install and Platform Queue

| Residual | Status after Sprint 134 |
| --- | --- |
| Hosted Linux package-contract behavior | Covered by the new reviewed Linux CI lane; final runner-specific proof comes from CI. |
| Hosted macOS supplemental CMake install/export behavior | Pending CI history; keep supplemental until runtime/flake evidence supports any later promotion. |
| Hosted Windows supplemental install/downstream behavior | Pending CI history; keep supplemental until runtime/flake evidence supports any later promotion. |
| Full reviewed macOS install/export parity | Deferred. |
| Separate reviewed Windows install-validation parity | Deferred. |
| Windows staged thread/fuzz/property promotion | Deferred pending source portability or Windows-native equivalents. |
| Package-manager support | Deferred non-claim. |
| Shared-library packaging and dynamic ABI compatibility | Deferred by Sprint 133 product decision. |
| Runtime-loader behavior | Deferred non-claim. |

## Validation Summary

Day 13 provides the integrated validation record. Final closeout evidence:

| Check | Result |
| --- | --- |
| YAML parse for Linux, macOS, and Windows workflows | Passed |
| Shell syntax for package proof scripts | Passed |
| `bash tests/test_install.sh` | Passed 22 checks, 0 failures |
| `bash tests/test_cmake_install.sh` | Passed 21 checks, 0 failures, 0 skips |
| `bash scripts/static_package_deferral_check.sh` | Passed |
| Local CMake configure plus `ctest -N` registration audit | Passed; 57 non-Windows tests |
| Windows CTest reconciliation | `57 - 3 staged Windows exclusions = 54` |
| Final YAML parse for Linux, macOS, and Windows workflows | Passed |
| Final static deferral wording guard | Passed |
| `git diff --check` | Passed |
| focused trailing-whitespace scan | Passed |
| C/header/CMake registration diff scan | No `.c`, `.h`, or `CMakeLists.txt` changes |
| package proof script diff scan | `tests/test_install.sh` changed for robust `pkg-config --cflags` token parsing |
| Final claim scan | Only explicit non-claims and historical count notes found |

The full C quality gate was not required because Sprint 134 did not modify C
sources or public headers. The post-PR CI fix touched only
`tests/test_install.sh` package-proof parsing.

Post-PR CI follow-up validation:

- `bash -n tests/test_install.sh`: passed
- `bash tests/test_install.sh`: 22 checks, 0 failures
- `bash scripts/static_package_deferral_check.sh`: passed

## PR Review Summary Material

Suggested PR summary:

- Added a reviewed Linux static-first package-contract CI lane.
- Added supplemental macOS CMake install/export package confidence.
- Added supplemental Windows CMake install/downstream package confidence.
- Reinforced Windows staged-test boundaries and blockers for pthread/POSIX
  test sources.
- Aligned README, INSTALL, maintainer guide, workflow comments, and Sprint 134
  artifacts with reviewed/supplemental/staged/deferred support tiers.

Suggested validation summary:

- `bash tests/test_install.sh`: 22 checks, 0 failures
- `bash tests/test_cmake_install.sh`: 21 checks, 0 failures, 0 skips
- `bash scripts/static_package_deferral_check.sh`: passed
- workflow YAML parse for Linux/macOS/Windows: passed
- local CMake/CTest registration audit: 57 non-Windows tests, Windows expected
  count remains 54 after staged exclusions
- docs/workflow whitespace and `git diff --check`: passed

## Sprint 135 Handoff

Sprint 135 should start from these constraints:

- Treat Linux as the only reviewed static-first package-contract CI owner.
- Treat macOS package install/export jobs as supplemental until hosted-runner
  history justifies any reviewed parity decision.
- Treat Windows install/downstream proof as supplemental until hosted-runner
  history justifies any reviewed install-validation decision.
- Do not promote Windows staged thread/fuzz/property tests without source
  portability changes, exact CTest count updates, and hosted MSVC proof.
- Preserve Sprint 133 static-first package/ABI non-claims unless a new product
  decision funds shared-library packaging, dynamic ABI compatibility,
  runtime-loader validation, and package metadata ownership.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Final platform support truth is clear to users and maintainers. | Complete | Final support truth table, README/INSTALL/maintainer guide alignment, and workflow comments agree. |
| Staged exclusions have owners, blockers, and support-tier boundaries. | Complete | Staged-exclusion register lists owners, blockers, and promotion gates. |
| Sprint 134 can close without unresolved workflow or support wording drift. | Complete | Day 13 validation and Day 14 closeout record no positive support overclaims. |
