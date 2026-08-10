# Sprint 148 Day 4 Portability Decision Matrix

## Purpose

Day 4 selects the Sprint 148 disposition for each staged Windows-excluded test
surface before implementation begins. The decisions combine the Day 2 source
blocker audit with the Day 3 CMake, CI, and expected-count audit.

No CMake registration or `EXPECTED_WINDOWS_CTEST_COUNT` value should change
until the selected implementation has local evidence and a matching
before/after CTest enumeration.

## Decision Inputs

| Input | Source | Decision Relevance |
| --- | --- | --- |
| Current reviewed Windows count is `56` | `.github/workflows/windows-ci.yml` | Count can increase only when a selected test is intentionally registered. |
| `test_threads` is gated by `Threads_FOUND AND NOT WIN32` | `CMakeLists.txt` | Direct pthread use blocks Windows registration. |
| `test_sprint4_integration` is gated by `Threads_FOUND AND NOT WIN32` | `CMakeLists.txt` | One pthread-backed lane hides several non-threaded integration checks from Windows. |
| `test_fuzz` is gated by `NOT WIN32 AND NOT MSVC` | `CMakeLists.txt` | POSIX temp-file APIs block Windows registration; most property behavior is platform-neutral. |
| POSIX Makefile and Linux/macOS CMake lanes already run these tests | Makefile/CMake/CI audit | Existing POSIX proof must not be weakened by Windows promotion. |
| Windows install/downstream proof remains supplemental | Sprint 147 handoff and Sprint 148 plan | Sprint 148 must not imply Windows install-validation parity. |

## Scoring Legend

| Score | Meaning |
| --- | --- |
| Low risk | Localized implementation with limited behavior or registration churn. |
| Medium risk | Requires helper extraction, split proof, or multi-file coordination. |
| High risk | Likely to change behavior, duplicate semantics, or create claim ambiguity. |
| High preservation | Keeps existing Linux/macOS behavior and promotes equivalent Windows evidence. |
| Medium preservation | Keeps core behavior but narrows or splits some proof ownership. |
| Low preservation | Drops, dilutes, or replaces meaningful existing behavior. |

## Per-Test Decision Matrix

### `test_threads`

| Option | Implementation Risk | Behavior Preservation | Hosted Evidence Need | Support-Claim Impact | Decision |
| --- | --- | --- | --- | --- | --- |
| Direct portable test-thread helper | Medium | High | High | Allows the reviewed Windows CMake subset to include thread lifecycle coverage after hosted proof. | Selected |
| Windows-native duplicate test file | Medium | Medium | High | Adds Windows coverage but creates duplicate behavior ownership and drift risk. | Rejected |
| Split proof owner | Medium | Medium | High | Could isolate stress behavior, but adds extra names without reducing the core pthread blocker. | Fallback only |
| Retain staged | Low | High for POSIX, none for Windows | Low | Leaves the Windows staged gap open. | Rejected as primary |

**Selected disposition:** port `test_threads` through a small test-only portable
thread helper that preserves current pthread behavior on POSIX and maps the same
thread lifecycle to a Windows-compatible implementation on MSVC.

**Promotion target:** reviewed Windows CMake registration for `test_threads`,
with existing POSIX proof preserved.

### `test_sprint4_integration`

| Option | Implementation Risk | Behavior Preservation | Hosted Evidence Need | Support-Claim Impact | Decision |
| --- | --- | --- | --- | --- | --- |
| Split non-threaded Sprint 4 integration proof from pthread-backed lane | Low to medium | High for non-threaded coverage, high for retained POSIX thread coverage | High | Promotes hidden non-threaded integration coverage without overclaiming threaded SuiteSparse parity. | Selected |
| Port entire file directly through the thread helper | Medium to high | Medium to high | High | Could promote all Sprint 4 behavior, but ties unrelated non-threaded coverage to thread-helper risk. | Conditional stretch |
| Windows-native duplicate threaded proof | High | Medium | High | Adds drift risk and unclear equivalence with the POSIX SuiteSparse lane. | Rejected |
| Retain staged | Low | High for POSIX, none for Windows | Low | Leaves non-threaded integration checks hidden from Windows. | Rejected as primary |

**Selected disposition:** split proof ownership so non-threaded Sprint 4
integration checks can be registered on Windows while the pthread-backed
SuiteSparse concurrency lane remains POSIX-only unless the Day 5-6 thread helper
lands cleanly enough to reuse without extra risk.

**Promotion target:** reviewed Windows CMake registration for the non-threaded
Sprint 4 integration surface. The threaded SuiteSparse lane remains explicitly
staged unless implemented and proven through the same portable thread helper.

### `test_fuzz`

| Option | Implementation Risk | Behavior Preservation | Hosted Evidence Need | Support-Claim Impact | Decision |
| --- | --- | --- | --- | --- | --- |
| Portable temp-file helper and full `test_fuzz` promotion | Medium | High | High | Promotes parser and deterministic property coverage into the reviewed Windows CMake subset. | Selected |
| Split property and file-backed parser proof | Medium | Medium | High | Promotes platform-neutral fuzz/property coverage while retaining parser temp-file cases as staged. | Fallback |
| Windows-native duplicate temp-file path only | Medium | Medium | High | Solves the blocker but risks local cleanup differences and duplicate helper behavior. | Rejected |
| Retain staged | Low | High for POSIX, none for Windows | Low | Leaves fuzz/property Windows coverage gap open. | Rejected as primary |

**Selected disposition:** implement a test-only portable temp-file helper and
attempt full `test_fuzz` Windows promotion. If MSVC compile or hosted behavior
shows broader issues, split the platform-neutral property/argument tests from
the file-backed parser fuzz cases and keep only the unresolved file-backed
surface staged.

**Promotion target:** reviewed Windows CMake registration for `test_fuzz` if the
full helper path passes; otherwise a named split fuzz/property proof with
explicit residual parser temp-file staging.

## Selected Implementation Targets

| Target | Primary Days | Expected Windows CTest Count Impact | Required Owners |
| --- | --- | --- | --- |
| Portable thread helper plus `test_threads` promotion | Days 5-6 | Tentative `+1` | Test helper, `tests/test_threads.c`, CMake registration, Windows workflow wording/count after evidence |
| Split non-threaded Sprint 4 integration proof | Days 7-8 | Tentative `+1` | New or refactored Sprint 4 test owner, CMake registration, residual threaded-lane wording |
| Portable temp-file helper plus `test_fuzz` promotion | Days 9-10 | Tentative `+1` | Test temp helper, `tests/test_fuzz.c`, CMake registration, Windows workflow wording/count after evidence |

If all selected targets land as Windows-registered CTest entries, the expected
reviewed Windows count should move from `56` to `59`. This remains a planning
number until confirmed by `ctest --test-dir build -C Release -N` on the updated
tree and by hosted Windows CI.

## Rollback Criteria

Roll back a selected promotion or split before claiming reviewed Windows
coverage when any of these conditions occurs:

| Condition | Rollback Action |
| --- | --- |
| The port fails local compile or focused test validation and cannot be fixed within the planned implementation day. | Restore the previous CMake gate and keep the test staged. |
| A POSIX Linux/macOS lane loses behavior, stress coverage, seed coverage, or diagnostics. | Revert or split the implementation so POSIX proof remains intact. |
| Windows `ctest -N` count differs from the planned count without a named added, removed, or renamed CTest entry. | Do not update `EXPECTED_WINDOWS_CTEST_COUNT`; fix registration or document an explicit split. |
| Hosted Windows CI is unavailable or fails for reasons tied to the promoted test. | Keep support-tier docs at "pending hosted proof" and leave the promoted claim out of reviewed wording. |
| A helper changes observable test semantics rather than only platform mechanics. | Reject the helper path and use a split proof or retained-staged disposition. |
| A selected implementation implies Windows Makefile, Windows `pkg-config`, install-validation parity, shared-library ABI, or package-manager support. | Reword docs/workflow comments and keep those surfaces explicit non-claims. |

## Support-Claim Impact Map

| Decision | Claim Allowed After Hosted Proof | Claim Still Not Allowed |
| --- | --- | --- |
| `test_threads` promoted | Reviewed Windows CMake subset includes thread lifecycle coverage for the promoted test. | Broad Windows threading parity, Windows Makefile parity, or runtime backend equivalence. |
| Sprint 4 non-threaded proof promoted | Reviewed Windows CMake subset includes Sprint 4 non-threaded integration coverage. | POSIX pthread SuiteSparse concurrency parity unless the threaded lane is separately ported and proven. |
| `test_fuzz` promoted | Reviewed Windows CMake subset includes deterministic fuzz/property coverage for the promoted test. | Unbounded fuzzing, sanitizer parity, or unsupported temp-file semantics outside the test helper. |
| Any lane retained staged | The retained lane remains a staged residual with source-level blocker documentation. | Reviewed Windows coverage for the retained behavior. |

## Implementation Sequence For Days 5-10

| Day | Implementation Focus | Exit Evidence |
| --- | --- | --- |
| Day 5 | Design the portable thread helper boundary, assertions, cleanup, diagnostics, and CMake registration path. | Thread design artifact with exact helper API and validation checklist. |
| Day 6 | Implement the selected `test_threads` port and focused local checks. | Thread implementation artifact, local focused validation, and no POSIX behavior regression. |
| Day 7 | Design the Sprint 4 non-threaded split and conditional threaded reuse path. | Integration split design artifact with CTest naming and residual-threaded-lane wording. |
| Day 8 | Implement the selected Sprint 4 split or port and focused local checks. | Integration implementation artifact and updated local registration evidence. |
| Day 9 | Design the portable temp-file helper and fuzz fallback split. | Fuzz design artifact with helper semantics, cleanup rules, and fallback split criteria. |
| Day 10 | Implement the selected fuzz promotion or fallback split and focused local checks. | Fuzz implementation artifact and local validation notes. |

## Day 5 Handoff

Day 5 should design a minimal test-only thread helper first. The design should
name the helper API, map POSIX pthread behavior and Windows behavior, define
cleanup/failure handling, and specify whether `test_sprint4_integration` can
reuse the helper later without coupling Day 6 completion to Day 8 work.

## Completion Criteria

- Every staged test has a selected Sprint 148 disposition.
- Selected work is bounded to Days 5-10 plus Day 11 registration and Day 13
  validation.
- Retained or conditional paths remain explicit non-claims.
- Expected-count changes remain blocked until implementation evidence exists.
