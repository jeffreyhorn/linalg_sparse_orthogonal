# Sprint 134 Day 11 - Windows Staged Lane Follow-Through

## Purpose

Day 11 implements the Day 10 staged-test decision. No staged Windows test is
promoted because the current blockers are source-level portability issues, not
workflow wording gaps.

## Decision

Keep the current Windows reviewed CTest subset unchanged at `54` tests.

Do not promote:

- `test_threads`
- `test_sprint4_integration`
- `test_fuzz`

The reviewed Windows lane remains the MSVC CMake configure/build/CTest subset.
The Day 9 CMake install/downstream job remains supplemental package confidence
and does not affect CTest membership.

## Implemented Follow-Through

| Surface | Update |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Added source-level blocker rationale to the top comment and `ctest -N` output. |
| `README.md` | Clarified that pthread/POSIX-backed staged tests remain outside the reviewed Windows subset. |
| `INSTALL.md` | Added the same staged-test boundary to the Windows platform row. |
| `docs/maintainer_guide.md` | Recorded the pthread/POSIX blockers and future promotion proof gates. |

## CTest Membership Impact

| Surface | Day 11 impact |
| --- | --- |
| `CMakeLists.txt` test registration | Unchanged |
| Windows workflow expected count | Unchanged at `EXPECTED_WINDOWS_CTEST_COUNT=54` |
| Windows staged exclusions | Unchanged: `test_threads`, `test_sprint4_integration`, `test_fuzz` |
| Day 9 supplemental install job | Unchanged and separate from CTest membership |

## Current Blockers

| Staged test | Blocker | Required promotion gate |
| --- | --- | --- |
| `test_threads` | Direct pthread API dependency. | Windows-native thread test or portability wrapper plus hosted MSVC configure/build/execute proof. |
| `test_sprint4_integration` | Direct pthread API dependency. | Windows-native integration equivalent or portability wrapper plus hosted MSVC configure/build/execute proof. |
| `test_fuzz` | POSIX temp-file APIs through `unistd.h`, `mkstemps`, `close`, and `unlink`. | Portable temp-file abstraction or Windows-native fuzz/property variant plus hosted MSVC proof. |

## Validation Evidence

| Check | Result |
| --- | --- |
| YAML parse for `.github/workflows/windows-ci.yml` | Passed |
| `bash scripts/static_package_deferral_check.sh` | Passed |
| Local CMake configure for CTest registration audit | Passed |
| Local `ctest -N` registration count | `57` on this non-Windows host |
| Windows count reconciliation | `57 - 3 staged Windows exclusions = 54` |
| Staged gate scan | Confirmed `if(Threads_FOUND AND NOT WIN32)` and `if(NOT WIN32 AND NOT MSVC)` remain in `CMakeLists.txt` |
| `git diff --check` | Passed |
| focused trailing-whitespace scan | Passed |
| C/header/CMake registration diff scan | No `.c`, `.h`, or `CMakeLists.txt` changes |
| Temporary audit build cleanup | `build-sprint134-day11` removed |

## Residual Queue

| Residual | Status |
| --- | --- |
| Windows thread coverage | Staged pending Windows-native thread proof. |
| Windows Sprint 4 concurrent integration coverage | Staged pending Windows-native integration proof. |
| Windows fuzz/property coverage | Staged pending portable temp-file handling or Windows-specific proof. |
| Windows expected CTest count update | Deferred until a staged test is intentionally promoted. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected Windows staged-lane decision is implemented or explicitly deferred. | Complete | No staged promotions; workflow/docs now state blocker rationale. |
| Test membership and support docs agree. | Complete | `EXPECTED_WINDOWS_CTEST_COUNT=54` remains unchanged and docs name the pthread/POSIX blockers. |
| No staged test is promoted without validation ownership. | Complete | No CMake registration changes were made. |
