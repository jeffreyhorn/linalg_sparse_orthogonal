# Sprint 148 Day 1 Windows Intake

## Purpose

Day 1 re-establishes Sprint 148 scope from the merged Sprint 147 handoff and
the current repository state. The sprint target is Windows staged-test
portability closure for `test_threads`, `test_sprint4_integration`, and
`test_fuzz`; it is not a Windows package, Makefile, `pkg-config`,
shared-library, package-manager, or broad platform parity sprint.

## Source Inputs

| Source | Role |
| --- | --- |
| `docs/planning/EPIC_13/PROJECT_PLAN.md` | Sprint 148 item list and estimates. |
| `docs/planning/EPIC_13/SPRINT_147/artifacts/day7-windows-evidence-gate.md` | Windows staged-test and install-validation evidence gates. |
| `docs/planning/EPIC_13/SPRINT_147/artifacts/day14-closeout-and-windows-handoff.md` | Sprint 148 prerequisite checklist and closeout handoff. |
| `docs/planning/EPIC_13/SPRINT_147/artifacts/day12-quality-surface-map.md` | Validation-owner map and full C gate rules. |
| `docs/planning/EPIC_13/SPRINT_147/artifacts/day13-public-claim-freeze-audit.md` | Public/support wording baseline and non-claims. |
| `CMakeLists.txt` | Current test registration and Windows staging gates. |
| `.github/workflows/windows-ci.yml` | Current reviewed Windows CMake workflow and expected-count enforcement. |
| `tests/test_threads.c` | Staged pthread-backed thread lifecycle surface. |
| `tests/test_sprint4_integration.c` | Staged pthread-backed Sprint 4 integration surface. |
| `tests/test_fuzz.c` | Staged POSIX-temp-file-backed fuzz/property surface. |

## Current Windows Support Tiers

| Tier | Current Surface | Evidence Owner | Sprint 148 Boundary |
| --- | --- | --- | --- |
| Reviewed | `.github/workflows/windows-ci.yml::build-and-test` | CI/platform owner | Reviewed MSVC CMake configure/build/CTest subset on `windows-2022`. |
| Supplemental | `.github/workflows/windows-ci.yml::install-and-downstream` | Platform/package owner | CMake-first install/downstream confidence; preserve as Sprint 149 input. |
| Staged | `test_threads`, `test_sprint4_integration`, `test_fuzz` | Platform/test owner | Sprint 148 may promote, replace, split, retain, or explicitly reject these staged paths. |
| Deferred | Windows Makefile, Windows `pkg-config`, reviewed install-validation parity | Platform/package owner | Keep as non-claims unless Sprint 149 promotes or rejects install-validation parity. |
| Unsupported | Shared-library ABI, dynamic ABI, runtime loader, package-manager distribution | Package/ABI owner | Outside Sprint 148 scope. |

## Current Reviewed Windows CMake Lane

| Field | Current Value |
| --- | --- |
| Workflow | `.github/workflows/windows-ci.yml` |
| Job name | `Windows enforced reviewed CMake consumer subset (MSVC)` |
| Runner | `windows-2022` |
| Generator | `Visual Studio 17 2022` |
| Architecture | `x64` |
| Build configuration | `Release` |
| Expected CTest count | `EXPECTED_WINDOWS_CTEST_COUNT=56` |
| Configure command | `cmake -S . -B build -G "Visual Studio 17 2022" -A x64` |
| Build command | `cmake --build build --config Release` |
| Enumeration command | `ctest --test-dir build -C Release -N` |
| Execution command | `ctest --test-dir build -C Release --output-on-failure` |

The workflow output currently states that Windows staged exclusions remain:
`test_threads`, `test_sprint4_integration`, and `test_fuzz`. It also names the
current blockers as pthread APIs for `test_threads` and
`test_sprint4_integration`, and POSIX temp-file APIs for `test_fuzz`.

## Current CMake Staged Exclusions

| Test | Current CMake Gate | Current Blocker |
| --- | --- | --- |
| `test_threads` | `if(Threads_FOUND AND NOT WIN32)` | Source includes pthread APIs directly. |
| `test_sprint4_integration` | `if(Threads_FOUND AND NOT WIN32)` | Source includes pthread APIs directly. |
| `test_fuzz` | `if(NOT WIN32 AND NOT MSVC)` | Source depends on POSIX temp-file behavior. |

## Staged-Test Audit Template

Day 2 and Day 3 should use this template for each staged test:

| Field | Required Detail |
| --- | --- |
| Test name | `test_threads`, `test_sprint4_integration`, or `test_fuzz`. |
| Current source file | Exact file path and line ranges for blockers. |
| Current CMake gate | Exact condition and registration point. |
| Blocker class | pthread API, POSIX file/temp API, MSVC compile issue, Windows runtime issue, or CTest policy issue. |
| Behavior to preserve | Assertions, lifecycle/property coverage, deterministic seed, cleanup behavior, timeout behavior, and diagnostics. |
| Existing POSIX proof | Linux/macOS coverage that must remain registered or explicitly split. |
| Windows option candidates | Direct portable port, Windows-native equivalent, split proof owner, retained staged status, or rejected promotion. |
| Expected-count impact | No change, +1, split rename, replacement, or explicit no-promotion rationale. |
| Documentation/report impact | README, INSTALL, maintainer guide, workflow output, report rows, and sprint artifact updates. |
| Stop conditions | Missing hosted proof, weakened POSIX proof, unexplained count drift, or unsupported Windows parity wording. |

## Day 1 Stop Conditions

| Stop Condition | Why It Stops Work |
| --- | --- |
| Branch is not based on current `master`. | Windows count and staged gates may be stale. |
| `EXPECTED_WINDOWS_CTEST_COUNT` differs from the Sprint 147 handoff without explanation. | The Day 1 baseline cannot be trusted. |
| A staged test has already changed registration before source blockers are audited. | Promotion would not be evidence-led. |
| A proposed Windows claim lacks hosted Windows proof requirements. | Local Unix proof cannot establish reviewed Windows support. |
| A change would weaken existing Linux/macOS/POSIX proof. | Sprint 148 must add or replace Windows coverage without losing existing coverage. |
| Documentation implies Windows Makefile, `pkg-config`, install-validation parity, shared-library, dynamic ABI, runtime-loader, package-manager, or broad Windows parity. | Sprint 148 support boundaries would be violated. |

## Day 2 Handoff

Day 2 should audit source blockers before deciding implementation strategy. It
should inspect:

- pthread calls, synchronization assumptions, timeout assumptions, and cleanup
  in `tests/test_threads.c`;
- pthread calls and behavior preserved by `tests/test_sprint4_integration.c`;
- POSIX temp-file assumptions, deterministic fuzz seeds, cleanup behavior, and
  bounded property coverage in `tests/test_fuzz.c`.

No CMake expected-count change should be made until Day 2 source blockers and
Day 3 registration policy are both understood.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 148 scope is tied to current files and Sprint 147 handoff artifacts. | Complete | Source inputs table links the Sprint 147 handoff, CMake, workflow, and staged test files. |
| Current Windows reviewed, supplemental, staged, deferred, and unsupported tiers are recorded. | Complete | Support-tier table preserves the current Windows support boundaries. |
| Each staged test has an audit owner and evidence format. | Complete | Staged-test audit template defines required fields for Day 2 and Day 3. |
