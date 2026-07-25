# Sprint 134 Day 10 - Windows Staged Test Re-Audit

## Purpose

Day 10 re-audits Windows CTest membership and the staged thread, integration,
fuzz, and property lanes before Day 11 decides whether to promote or reinforce
any staged exclusions.

## Current Windows CTest Membership

| Surface | Current state |
| --- | --- |
| Reviewed Windows workflow | `.github/workflows/windows-ci.yml` |
| Reviewed Windows job | `Windows enforced reviewed CMake consumer subset (MSVC)` |
| Reviewed commands | CMake configure, CMake build, `ctest -N`, full `ctest` |
| Expected Windows CTest count | `54` |
| Local non-Windows CTest count | `57` |
| Reconciliation | `57 - test_threads - test_sprint4_integration - test_fuzz = 54` |

The Day 9 supplemental Windows install/downstream job does not add CTest
membership. It therefore does not affect the reviewed Windows CTest count.

## Registration Evidence

| Evidence | Result |
| --- | --- |
| `rg -n "add_sparse_test\\(" CMakeLists.txt` count | `57` registrations on non-Windows platforms |
| `cmake -S . -B build-sprint134-day10 -DCMAKE_BUILD_TYPE=Release` | Passed |
| `ctest --test-dir build-sprint134-day10 -N` | `Total Tests: 57` on this non-Windows host |
| Windows thread/integration gate | `if(Threads_FOUND AND NOT WIN32)` |
| Windows fuzz/property gate | `if(NOT WIN32 AND NOT MSVC)` |
| Windows workflow expected count | `EXPECTED_WINDOWS_CTEST_COUNT: "54"` |
| `git diff --check` | Passed |
| focused trailing-whitespace scan | Passed |
| temporary audit build cleanup | `build-sprint134-day10` removed |

## Staged-Exclusion Rationale

| Staged test | Current blocker | Promotion decision |
| --- | --- | --- |
| `test_threads` | Source includes `<pthread.h>` directly and uses pthread APIs. CMake intentionally gates it on `Threads_FOUND AND NOT WIN32` because CMake can find Win32 threads, but the test source is pthread-specific. | Keep staged. Promotion requires a Windows-native thread test or a portability wrapper plus hosted MSVC proof. |
| `test_sprint4_integration` | Source includes `<pthread.h>` directly and exercises concurrent integration paths through pthread workers. It shares the same CMake gate as `test_threads`. | Keep staged. Promotion requires replacing pthread-only orchestration or adding an equivalent Windows-native integration test. |
| `test_fuzz` | Source includes `<unistd.h>` and uses POSIX temp-file operations including `mkstemps`, `close`, and `unlink`. It also owns the bounded lifecycle property lane currently excluded from Windows. | Keep staged. Promotion requires a portable temp-file abstraction or a Windows-specific fuzz/property test variant plus hosted MSVC proof. |

## Candidate Promotion List

| Candidate | Day 10 outcome |
| --- | --- |
| Promote `test_threads` as-is | Rejected; pthread-only source would not compile under MSVC. |
| Promote `test_sprint4_integration` as-is | Rejected; pthread-only source would not compile under MSVC. |
| Promote `test_fuzz` as-is | Rejected; POSIX temp-file APIs and `<unistd.h>` are not MSVC-ready. |
| Add Windows-native equivalents | Candidate for a later sprint or explicit Day 11 decision, but not safe as an automatic promotion. |
| Keep staged exclusions and clarify docs/comments | Selected Day 10 recommendation for Day 11 follow-through. |

## Future Proof Gates

| Gate | Required before promotion |
| --- | --- |
| Source portability | Remove direct pthread/POSIX dependencies from the promoted test or add a Windows-native equivalent. |
| CMake membership update | Update `CMakeLists.txt` gates intentionally and document the exact expected Windows CTest count delta. |
| Hosted MSVC configure/build evidence | Prove the promoted test builds under `windows-2022` with the Visual Studio 2022 generator. |
| Hosted MSVC execution evidence | Prove the promoted test passes under full `ctest`. |
| Support wording update | Update workflow output, README/INSTALL/maintainer docs, and Sprint artifacts so staged tests are not silently described as reviewed. |

## Day 11 Recommendation

Do not promote any staged Windows CTest member on Day 11. Instead, reinforce the
current staged exclusions in workflow comments or support docs if the final
Day 11 audit finds wording drift.

If Day 11 chooses to make a code change anyway, the smallest safe path is a
new Windows-native equivalent test with its own CMake gate and an explicit
CTest count update. Promoting the current pthread/POSIX tests as-is is not
supported by the Day 10 audit.

## Residual Queue

| Residual | Status |
| --- | --- |
| Windows thread coverage | Staged pending Windows-native thread proof. |
| Windows Sprint 4 concurrent integration coverage | Staged pending Windows-native integration proof. |
| Windows fuzz/property coverage | Staged pending portable temp-file handling or Windows-specific property proof. |
| Windows CTest count update | Deferred until a staged test is intentionally promoted. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Windows staged exclusion has current rationale. | Complete | Rationale table covers `test_threads`, `test_sprint4_integration`, and `test_fuzz`. |
| CTest count and candidate promotions are explicit. | Complete | Count remains `54`; all as-is promotions are rejected. |
| Staged tests are not silently described as reviewed. | Complete | Artifact recommends Day 11 reinforcement rather than promotion. |
