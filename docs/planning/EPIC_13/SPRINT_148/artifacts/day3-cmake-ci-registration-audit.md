# Sprint 148 Day 3 CMake, CI, And Expected-Count Audit

## Purpose

Day 3 audits registration and policy ownership for the staged Windows test
surfaces. This artifact defines where each test is registered today, how
Windows excludes it, what CI output enforces the reviewed count, and what
evidence is required before any CMake or workflow change.

No CMake or workflow policy changes are made on Day 3.

## CMake Gate Inventory

| Test | Current Registration Owner | Current Gate | Linkage | Windows Effect |
| --- | --- | --- | --- | --- |
| `test_threads` | `CMakeLists.txt` test section | `if(Threads_FOUND AND NOT WIN32)` | `Threads::Threads` | Excluded from Windows even though CMake may find Win32 threads, because the source uses pthread APIs. |
| `test_sprint4_integration` | `CMakeLists.txt` test section | `if(Threads_FOUND AND NOT WIN32)` | `Threads::Threads` | Excluded from Windows at file level because one lane uses pthread APIs. |
| `test_fuzz` | `CMakeLists.txt` test section | `if(NOT WIN32 AND NOT MSVC)` | default test executable linkage | Excluded from Windows/MSVC because the source uses POSIX temp-file APIs. |

Relevant CMake owners:

```cmake
function(add_sparse_test TEST_NAME)
    add_executable(${TEST_NAME} tests/${TEST_NAME}.c)
    target_link_libraries(${TEST_NAME} PRIVATE sparse_lu_ortho)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()
```

Current CMake comments already state that pthread-backed tests are POSIX-only
and that CMake's `find_package(Threads)` can find Win32 threads even though the
test source uses pthread APIs. Preserve that distinction if a Windows-native or
portable helper path is added.

## Windows Workflow Expected-Count Audit

| Field | Current Value |
| --- | --- |
| Workflow | `.github/workflows/windows-ci.yml` |
| Reviewed job | `Windows enforced reviewed CMake consumer subset (MSVC)` |
| Runner | `windows-2022` |
| Generator | `Visual Studio 17 2022` |
| Architecture | `x64` |
| Build configuration | `Release` |
| Count variable | `EXPECTED_WINDOWS_CTEST_COUNT` |
| Current count | `56` |
| Enumeration command | `ctest --test-dir build -C Release -N` |
| Execution command | `ctest --test-dir build -C Release --output-on-failure` |

The workflow enforces the count by:

1. capturing `ctest -N` output;
2. extracting `Total Tests:`;
3. converting the value to an integer;
4. comparing it to `EXPECTED_WINDOWS_CTEST_COUNT`;
5. failing before execution if the count differs.

Current staged-exclusion output names:

- `test_threads`;
- `test_sprint4_integration`;
- `test_fuzz`, including the bounded lifecycle property lane;
- pthread blockers for `test_threads` and `test_sprint4_integration`;
- POSIX temp-file blockers for `test_fuzz`;
- no reviewed Makefile parity;
- no separate reviewed install-validation lane.

## Cross-Platform Registration Table

| Surface | Makefile/POSIX | Linux CI | macOS CI | Windows CMake |
| --- | --- | --- | --- | --- |
| `test_threads` | Listed in `TEST_SRCS`; special build rule links `-pthread`. | Built and run directly in supplemental TSan lane; also part of Makefile test surface. | Part of reviewed Makefile/CMake local parity where POSIX thread support exists. | Excluded by `Threads_FOUND AND NOT WIN32`. |
| `test_sprint4_integration` | Listed in `TEST_SRCS`; special build rule links `-pthread`. | Part of Makefile test surface. | Part of reviewed Makefile/CMake local parity where POSIX thread support exists. | Excluded by `Threads_FOUND AND NOT WIN32`. |
| `test_fuzz` | Listed in `TEST_SRCS`; built by default test rule. | Part of Makefile test surface. | Part of reviewed Makefile/CMake local parity where POSIX temp-file APIs exist. | Excluded by `NOT WIN32 AND NOT MSVC`. |

Linux and macOS remain the existing POSIX proof owners. Sprint 148 changes must
not remove their coverage unless a replacement proof is explicitly recorded.

## Report And Documentation Update Candidates

| Surface | Current Role | Update Needed If Promoted | Update Needed If Retained Staged |
| --- | --- | --- | --- |
| `.github/workflows/windows-ci.yml` | Reviewed Windows CMake subset and supplemental install/downstream lane. | Update expected count, staged-output text, blocker text, and promoted/remaining test list. | Preserve count and staged-output text; update only if blocker wording becomes more precise. |
| `CMakeLists.txt` | Test registration gates. | Register promoted Windows-compatible paths intentionally and document split/portable gates. | Keep gates and comments aligned with source blockers. |
| `README.md` | Cross-platform CI contract. | Mention only the promoted reviewed Windows CMake coverage. | Preserve staged-test wording. |
| `INSTALL.md` | Supported platform and Windows CMake consumer interpretation. | Update Windows platform row only for reviewed CMake test coverage; keep install parity separate. | Preserve current Windows staged-test interpretation. |
| `docs/maintainer_guide.md` | Support-tier interpretation and staged residuals. | Update registered count, promoted tests, remaining staged tests, and proof boundaries. | Preserve current staged residuals. |
| `tests/corpus/manifests/report_families.tsv` | CI lane definition row. | Update claim scope/non-claims only if reviewed Windows CI row semantics change. | Usually no change; source-controlled row remains advisory CI lane definition. |
| Sprint 148 artifacts | Implementation evidence. | Record before/after CTest counts, hosted run IDs, and promoted tests. | Record retained staged status and blockers. |

## CTest Before/After Evidence Template

Every count-changing CMake/CI update must record:

| Field | Required Value |
| --- | --- |
| Commit SHA | Exact commit under test. |
| Branch or PR | Branch name or PR number. |
| Platform | Windows hosted runner for claim evidence; local platform for preliminary enumeration. |
| Workflow/job | Workflow file and job name when hosted. |
| Generator/config | CMake generator, architecture, and configuration. |
| Before count | `ctest -N` count before the registration change. |
| After count | `ctest -N` count after the registration change. |
| Count delta | Expected numeric delta and reason. |
| Promoted tests | Exact test names newly registered on Windows. |
| Remaining staged tests | Exact test names still excluded. |
| Execution result | Full CTest result after enumeration passes. |
| Non-claims | Explicit note preserving Windows Makefile, `pkg-config`, install-validation, shared-library, ABI, runtime-loader, package-manager, and broad platform non-claims. |

Recommended local preliminary commands:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build -N
ctest --test-dir build --output-on-failure
```

Hosted Windows proof remains required before any reviewed Windows claim is
promoted.

## Expected-Count Change Rules

1. Do not change `EXPECTED_WINDOWS_CTEST_COUNT` before the source blocker,
   CMake gate, and selected Day 4 disposition agree.
2. A count increase must name the new Windows-registered test or split proof
   owner.
3. A count decrease must explain test removal or replacement and preserve
   non-claims.
4. A rename or split must record both removed and added CTest names.
5. The workflow staged-exclusion output must be updated in the same change as
   any promotion.
6. Hosted `windows-2022` CTest enumeration must match the expected count before
   execution is interpreted as reviewed evidence.

## Day 4 Handoff

Day 4 should choose per-test dispositions using both the Day 2 source audit and
this registration audit:

- `test_threads`: shared portable helper, Windows-native proof owner, retained
  staged status, or rejected promotion.
- `test_sprint4_integration`: split non-threaded integration coverage from the
  threaded SuiteSparse lane, add Windows-native threaded proof, retain staged,
  or reject promotion.
- `test_fuzz`: portable temp-file helper, split file-backed parser fuzz from
  property lanes, retain staged, or reject promotion.

No implementation batch should start until the Day 4 matrix defines expected
CMake names, count impact, docs/report impact, and rollback rules.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| All registration owners are known before CMake edits. | Complete | CMake, Makefile/POSIX, Linux CI, macOS CI, and Windows workflow owners are mapped. |
| Expected-count changes require documented before/after evidence. | Complete | Evidence template and count-change rules require before/after counts, deltas, promoted tests, and hosted proof. |
| Windows reviewed and supplemental claims remain separate. | Complete | Documentation/report candidate table keeps Sprint 148 reviewed CMake changes separate from Sprint 149 install-validation parity. |
