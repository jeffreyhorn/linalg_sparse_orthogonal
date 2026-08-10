# Sprint 148 Working Notes

## Goal

Sprint 148 promotes or replaces the staged Windows-excluded test surfaces with
reviewed Windows-compatible coverage while preserving the current support-tier
boundaries.

## Starting Evidence

- Sprint 147 published the Windows staged-test evidence gate and closeout
  handoff.
- The current reviewed Windows lane is the MSVC CMake configure/build/CTest
  subset in `.github/workflows/windows-ci.yml`.
- The current reviewed Windows CTest count is
  `EXPECTED_WINDOWS_CTEST_COUNT=56`.
- `test_threads` and `test_sprint4_integration` are staged out of Windows
  CMake because they use pthread APIs directly.
- `test_fuzz` is staged out of Windows CMake because it depends on POSIX
  temp-file behavior.
- Windows CMake install/downstream confidence remains supplemental and is a
  Sprint 149 product-decision input, not Sprint 148 scope.

## Support-Tier Baseline

| Tier | Current Surface | Sprint 148 Rule |
| --- | --- | --- |
| Reviewed | Windows MSVC CMake configure/build/CTest subset | May change only when a staged test is intentionally promoted, replaced, or removed with hosted proof requirements preserved. |
| Supplemental | Windows CMake install/downstream confidence | Preserve as handoff to Sprint 149; do not promote during Sprint 148. |
| Staged | `test_threads`, `test_sprint4_integration`, `test_fuzz` | Audit and decide per-test disposition before CMake expected-count changes. |
| Deferred | Windows Makefile, Windows `pkg-config`, reviewed install-validation parity | Keep explicit non-claims. |
| Unsupported | Shared-library ABI, dynamic ABI, runtime loader, package-manager distribution | Keep outside Sprint 148 claims. |

## Item-To-Day Owner Map

| Sprint 148 Item | Primary Days | Closeout Owner |
| --- | --- | --- |
| Item 1: Staged Test Audit | Days 1-3 | Day 1 refreshes the baseline; Days 2-3 audit source blockers and registration policy. |
| Item 2: Portability Design | Days 4-5, 7, 9 | Day 4 chooses per-test dispositions; Days 5, 7, and 9 design the selected lanes. |
| Item 3: Thread Test Port | Days 5-6 | Day 5 designs and Day 6 implements the thread lane if selected. |
| Item 4: Fuzz/Property Port | Days 9-10 | Day 9 designs and Day 10 implements the fuzz/property lane if selected. |
| Item 5: CMake/CI Promotion | Days 3, 11, 13 | Day 3 audits registration; Day 11 applies policy; Day 13 validates and records hosted evidence status. |
| Item 6: Documentation Alignment | Day 12 | Day 12 aligns support-tier docs and residual/non-claim wording. |
| Item 7: Validation | Days 6, 8, 10, 11, 13-14 | Focused validation follows each implementation batch; Day 13 integrates and Day 14 closes. |

## Stop Conditions

- A Windows reviewed-coverage claim lacks hosted Windows proof.
- `EXPECTED_WINDOWS_CTEST_COUNT` changes without a before/after enumeration and
  documented reason.
- A staged test is marked promoted without intentional CMake registration.
- A port removes or weakens Linux/macOS/POSIX proof without replacement or an
  explicit retained-staged decision.
- Windows install-validation parity wording is promoted before Sprint 149.
- Documentation implies Windows Makefile, Windows `pkg-config`,
  shared-library, dynamic ABI, runtime-loader, package-manager, or broad
  Windows parity.
- Required local checks or full C quality gates fail.

## Daily Log

### Day 1: Windows Intake

- Re-read the Sprint 148 plan and project-plan section.
- Reviewed Sprint 147 Day 7 Windows evidence gate and Day 14 closeout handoff.
- Created the Sprint 148 artifact directory.
- Captured the current Windows reviewed lane: `.github/workflows/windows-ci.yml`
  job `Windows enforced reviewed CMake consumer subset (MSVC)` on
  `windows-2022`, Visual Studio 17 2022, x64, Release.
- Reconfirmed `EXPECTED_WINDOWS_CTEST_COUNT=56`.
- Reconfirmed staged CMake exclusions: `test_threads` and
  `test_sprint4_integration` gated by `Threads_FOUND AND NOT WIN32`;
  `test_fuzz` gated by `NOT WIN32 AND NOT MSVC`.
- Defined staged-test audit fields and stop conditions for unsupported Windows
  parity, unclear hosted proof, and expected-count drift.
- Day 2 handoff: audit each staged test source for exact pthread/POSIX blockers
  and behavior that must be preserved before choosing any porting direction.

### Day 2: Source Audit

- Audited `tests/test_threads.c` and tied the Windows blocker to direct
  `<pthread.h>`, `pthread_t`, `pthread_create`, and `pthread_join` usage across
  independent LU workers, shared LU/Cholesky solves, stress tests, concurrent
  `sparse_norminf`, norminf-plus-solve, and optional `SPARSE_MUTEX` concurrent
  insertion.
- Audited `tests/test_sprint4_integration.c` and found that only the concurrent
  SuiteSparse Cholesky lane uses pthread directly, while several non-threaded
  Cholesky/CSR/condest integration checks are currently hidden from Windows by
  the file-level CMake gate.
- Audited `tests/test_fuzz.c` and tied the Windows blocker to `<unistd.h>`,
  `mkstemps`, `close`, and `unlink` in the temp-file helper, not to the entire
  deterministic property-test surface.
- Mapped behavior that must be preserved: thread lifecycle/stress semantics,
  SuiteSparse `nos4` integration behavior, parser malformed-input fuzz cases,
  explicit temp-file skip behavior, deterministic seeds, and large CSC
  lifecycle/reorder property checks.
- Listed helper opportunities without choosing implementation: test-thread
  wrapper, split Sprint 4 proof files, portable temp-file helper, and
  file-backed versus property split for fuzz.
- Day 3 handoff: audit CMake and workflow registration against this source
  audit before selecting per-test dispositions or changing the expected Windows
  CTest count.

### Day 3: Registration Audit

- Audited `CMakeLists.txt` test registration and confirmed `test_threads` and
  `test_sprint4_integration` share the `Threads_FOUND AND NOT WIN32` gate plus
  `Threads::Threads` linkage, while `test_fuzz` is gated by
  `NOT WIN32 AND NOT MSVC`.
- Audited `.github/workflows/windows-ci.yml` and confirmed the reviewed Windows
  CMake job enforces `EXPECTED_WINDOWS_CTEST_COUNT=56` by parsing
  `ctest --test-dir build -C Release -N` output before running full CTest.
- Confirmed workflow output still names the staged exclusions and blockers, and
  explicitly preserves no reviewed Makefile parity and no separate reviewed
  install-validation lane.
- Mapped cross-platform registration: Makefile/POSIX includes all three staged
  tests, Linux CI has direct thread TSan coverage plus Makefile test coverage,
  macOS retains reviewed POSIX Make/CMake coverage, and Windows CMake excludes
  the three staged surfaces.
- Identified update candidates for CMake, Windows workflow comments/counts,
  README, INSTALL, maintainer guide, report-family rows, and Sprint 148
  artifacts if a staged surface is promoted or retained staged.
- Defined the CTest before/after evidence template and expected-count change
  rules for Day 11/Day 13 validation.
- Day 4 handoff: choose per-test dispositions only after combining the Day 2
  source blockers with this registration and expected-count audit.

### Day 4: Portability Decision

- Created the per-test portability decision matrix in
  `artifacts/day4-portability-decision-matrix.md`.
- Selected a direct portable test-thread helper for `test_threads`, with the
  existing pthread behavior preserved on POSIX and a Windows-compatible thread
  lifecycle path added for MSVC.
- Selected a split-proof strategy for `test_sprint4_integration`: promote the
  non-threaded Sprint 4 integration coverage on Windows and retain the
  pthread-backed SuiteSparse concurrency lane as POSIX-only unless the thread
  helper lands cleanly enough to reuse without additional risk.
- Selected a portable temp-file helper as the primary `test_fuzz` promotion
  path, with a split property/parser fallback if full MSVC promotion exposes a
  broader blocker.
- Recorded tentative Windows CTest count impact as `+3` only if all selected
  targets land as Windows-registered CTest entries, moving the planning count
  from `56` to `59`; no workflow expected-count change should happen until
  implementation and `ctest -N` evidence agree.
- Defined rollback rules for local compile failures, POSIX behavior weakening,
  CTest count drift, hosted Windows failures, helper semantic changes, and
  accidental support-claim expansion.
- Day 5 handoff: design the minimal test-only thread helper API, cleanup
  behavior, diagnostics, and CMake registration path before editing
  `test_threads`.

### Day 5: Thread Test Port Design

- Created the thread portability design artifact in
  `artifacts/day5-thread-test-port-design.md`.
- Chose a test-only helper header, tentatively `tests/test_thread_helpers.h`,
  that keeps the existing `void *(*)(void *)` worker signature and hides
  pthread versus Win32 thread creation behind `test_thread_create` and
  `test_thread_join`.
- Defined POSIX behavior so `<pthread.h>` moves into the helper and
  `Threads::Threads` remains linked for non-Windows builds.
- Defined Windows behavior around `CreateThread`, `WaitForSingleObject`, and
  `CloseHandle`, with a small adapter that preserves the current worker
  function shape instead of rewriting test bodies.
- Set the Day 6 refactor boundary for `tests/test_threads.c`: replace raw
  pthread includes, handle arrays, create calls, and join calls while preserving
  worker logic, thresholds, stress counts, iteration counts, and the
  `SPARSE_MUTEX` opt-in insert proof.
- Defined the CMake registration plan: promote only `test_threads`, keep
  `test_sprint4_integration` under the existing POSIX-only gate until Days 7-8,
  and update Windows expected-count policy only after `ctest -N` evidence.
- Recorded the single-test expected-count planning delta as `56` to `57` if
  Day 6 successfully promotes `test_threads`.
- Day 6 handoff: implement the helper, refactor `test_threads`, run focused
  CMake/CTest checks, and then run the full C quality gate because `.c`/`.h`
  files will change.

### Day 6: Thread Test Port Implementation

- Created `tests/test_thread_helpers.h` as a test-only portable thread helper
  with pthread-backed POSIX behavior and Win32 `CreateThread` /
  `WaitForSingleObject` / `CloseHandle` behavior.
- Refactored `tests/test_threads.c` to include the helper and replace raw
  `pthread_t`, `pthread_create`, and `pthread_join` usage with
  `test_thread_t`, `test_thread_create`, and `test_thread_join`.
- Preserved the existing worker functions, thread counts, stress iteration
  counts, residual thresholds, diagnostics, and `SPARSE_MUTEX` opt-in insert
  proof.
- Added explicit join return-code assertions so thread join failures are
  reported by the test framework.
- Updated `CMakeLists.txt` so `test_threads` is always registered by CMake,
  while POSIX builds still link `Threads::Threads`; kept
  `test_sprint4_integration` under the existing POSIX-only gate for Days 7-8.
- Focused validation passed: `cmake -S . -B build`,
  `cmake --build build --target test_threads`, and
  `ctest --test-dir build -R '^test_threads$' --output-on-failure`.
- Local POSIX `ctest --test-dir build -N` reports `Total Tests: 59`; the
  planned Windows expected-count delta remains `56` to `57` pending hosted
  Windows enumeration and the Day 11 CI promotion update.
- Required full C gate passed: `make format && make lint && make test`.
- Day 7 handoff: design the Sprint 4 integration split with the helper
  available as a conditional input, but keep non-threaded Windows promotion as
  the primary target and the SuiteSparse threaded lane as separately staged
  unless proven safe.

### Day 7: Sprint 4 Integration Port Design

- Created the Sprint 4 integration portability design artifact in
  `artifacts/day7-sprint4-integration-port-design.md`.
- Mapped all five `tests/test_sprint4_integration.c` checks to their behavior
  owners: four non-threaded Cholesky/CSR/condest integration checks plus one
  concurrent SuiteSparse Cholesky worker lane.
- Selected a direct helper-backed port as the Day 8 primary path because the
  Day 6 `tests/test_thread_helpers.h` implementation preserves the existing
  `void *(*)(void *)` worker signature and keeps pthread mechanics isolated.
- Kept the Day 4 split-proof strategy as the fallback: if the direct
  helper-backed port fails on MSVC or hosted Windows, retain the current
  pthread-backed source as POSIX-only and add a non-threaded portable Sprint 4
  integration proof for Windows.

### Day 8: Sprint 4 Integration Port Implementation

- Refactored `tests/test_sprint4_integration.c` to use
  `tests/test_thread_helpers.h` and removed direct pthread include/API usage.
- Preserved all five Sprint 4 integration tests, including the concurrent
  SuiteSparse Cholesky worker lane with four workers and the existing
  `maxerr < 1e-8` assertion.
- Updated `CMakeLists.txt` to register `test_sprint4_integration` outside the
  Windows gate and link `Threads::Threads` only on non-Windows builds.
- Created the Day 8 implementation evidence artifact in
  `artifacts/day8-sprint4-integration-port-implementation.md`.
- Focused validation passed: `cmake -S . -B build`,
  `cmake --build build --target test_sprint4_integration`, and
  `ctest --test-dir build -R '^test_sprint4_integration$'
  --output-on-failure`.
- Local POSIX `ctest --test-dir build -N` remains `Total Tests: 59`; planned
  Windows CTest delta is `57` to `58` for the Sprint 4 promotion, pending
  hosted Windows proof and Day 11 workflow-count updates.
- Required full C gate passed: `make format && make lint && make test`.
- Day 9 handoff: design the portable `test_fuzz` temp-file strategy before
  editing the fuzz/property lane.

### Day 9: Fuzz And Property Port Design

- Created the fuzz/property portability design artifact in
  `artifacts/day9-fuzz-property-port-design.md`.
- Mapped `tests/test_fuzz.c` into file-backed Matrix Market parser fuzz cases
  and platform-neutral argument/property checks.
- Confirmed the Windows blocker is limited to test temp-file mechanics:
  `<unistd.h>`, `mkstemps`, `close`, and `unlink`.
- Selected an in-file test-only portable temp-file helper as the Day 10 primary
  path, preserving POSIX `mkstemps` behavior and using CRT-safe Windows temp
  path creation plus `remove` cleanup on MSVC.
- Defined cleanup rules: one bounded suite-level `.mtx` path, no extra temp
  files per fuzz case, skip file-backed parser fuzz only when helper creation
  fails, and best-effort cleanup at suite exit.
- Kept the fallback split explicit: if the full helper path fails, promote a
  temp-free property/argument target and retain file-backed parser fuzz as a
  Windows-staged residual.
- Recorded the planned Windows CTest delta as `58` to `59` for the Day 10
  `test_fuzz` promotion, with workflow count and staged wording still owned by
  Day 11.
- No C source changed on Day 9, so no C quality gate was required.
- Day 10 handoff: implement the portable temp-file helper, promote
  `test_fuzz` in CMake if focused validation passes, and then run the full C
  quality gate.

### Day 10: Fuzz And Property Port Implementation

- Refactored `tests/test_fuzz.c` so `<unistd.h>`, `mkstemps`, and `close`
  remain POSIX-only while Windows uses `GetTempPathA`, `GetTempFileNameA`, and
  `MoveFileA` to create a unique `.mtx` temp path.
- Switched fuzz temp cleanup to `remove(fuzz_tmp_path)`, which works for both
  POSIX and Windows and preserves best-effort cleanup semantics.
- Preserved all file-backed Matrix Market fuzz inputs, null/nonexistent-file
  argument checks, deterministic property seeds, large CSC lifecycle checks,
  and property thresholds.
- Promoted `test_fuzz` in `CMakeLists.txt` by removing the
  `NOT WIN32 AND NOT MSVC` registration gate.
- Created the Day 10 implementation evidence artifact in
  `artifacts/day10-fuzz-property-port-implementation.md`.
- Focused validation passed after the final `.mtx` suffix implementation:
  `cmake -S . -B build`, `cmake --build build --target test_fuzz`,
  `ctest --test-dir build -R '^test_fuzz$' --output-on-failure`, and
  `ctest --test-dir build -N`.
- Local POSIX CTest enumeration remains `Total Tests: 59`; planned Windows
  CTest delta is `58` to `59` for the fuzz promotion, and the aggregate Sprint
  148 planned Windows delta is `56` to `59`.
- Required full C gate passed on the final source state:
  `make format && make lint && make test`.
- Day 11 handoff: update Windows workflow expected count and staged wording for
  the three promoted CMake tests while keeping deferred Windows
  Makefile/install/pkg-config claims explicit.

### Day 11: CMake And Windows CI Promotion Batch

- Reconciled the Day 6, Day 8, and Day 10 implementation outcomes and confirmed
  the three planned Windows CMake promotions are complete in `CMakeLists.txt`:
  `test_threads`, `test_sprint4_integration`, and `test_fuzz`.
- Updated `.github/workflows/windows-ci.yml` so
  `EXPECTED_WINDOWS_CTEST_COUNT` moves from `56` to `59`.
- Replaced the old Windows staged-exclusion job output with wording that names
  the promoted portable tests as part of the reviewed CMake CTest surface.
- Preserved the Windows non-claims for reviewed Makefile parity, separate
  reviewed install validation, pkg-config parity, shared-library support, and
  dynamic ABI support.
- Created the Day 11 promotion evidence artifact in
  `artifacts/day11-cmake-ci-promotion-batch.md`.
- Local validation passed: `cmake -S . -B build`, `ctest --test-dir build -N`
  with `Total Tests: 59`, and `git diff --check`.
- No `.c` or `.h` files changed on Day 11, so the full C gate was not rerun;
  Day 10 remains the latest full source gate on this branch.
- Day 12 handoff: update `README.md`, `INSTALL.md`, and
  `docs/maintainer_guide.md` so public/support docs no longer describe
  `test_threads`, `test_sprint4_integration`, or `test_fuzz` as Windows-staged.

### Day 12: Documentation Alignment

- Updated `README.md` so the CI summary says Windows remains CMake-first while
  including the promoted `test_threads`, `test_sprint4_integration`, and
  `test_fuzz` CTest targets in the reviewed subset.
- Updated `INSTALL.md` supported-platform wording for Windows to name the three
  promoted targets and preserve the Makefile/pkg-config/install-parity
  non-claims.
- Updated `docs/maintainer_guide.md` from the old `56` registered Windows CTest
  count to `59`, and replaced stale staged-exclusion language with the
  promoted CMake-subset interpretation.
- Created the Day 12 documentation evidence artifact in
  `artifacts/day12-docs-alignment.md`.
- Documentation validation passed: stale public/support wording search,
  trailing-whitespace check over touched docs and Sprint 148 artifacts, and
  `git diff --check`.
- No `.c` or `.h` files changed on Day 12, so the full C gate was not required.
- Day 13 handoff: run integrated validation over the promoted test surfaces and
  capture hosted Windows evidence availability or pending-proof residuals.

### Day 13: Integrated Validation And Hosted Evidence Intake

- Ran local CMake validation for the promoted Windows CMake surfaces:
  `cmake -S . -B build`,
  `cmake --build build --target test_threads test_sprint4_integration test_fuzz`,
  `ctest --test-dir build -N`, and
  `ctest --test-dir build -R '^(test_threads|test_sprint4_integration|test_fuzz)$'
  --output-on-failure`.
- Confirmed local CTest enumeration remains `Total Tests: 59`.
- Confirmed focused promoted-target CTest passed: `test_threads`,
  `test_sprint4_integration`, and `test_fuzz` all passed.
- Ran the required full C quality gate because the branch includes `.c` and
  `.h` changes: `make format && make lint && make test`.
- Full gate passed; `cppcheck` covered `tests/test_fuzz.c`,
  `tests/test_sprint4_integration.c`, and `tests/test_threads.c` with `_WIN32`
  defined, and `make test` ended with `All tests passed.`
- Checked hosted evidence availability with `gh pr view`; no PR exists for
  `sprint-148`, so hosted Windows proof remains pending PR CI rather than a
  failing or unavailable-check result.
- Created the Day 13 validation evidence artifact in
  `artifacts/day13-integrated-validation.md`.
- Hygiene validation passed: `git diff --check` and trailing-whitespace checks.
- Day 14 handoff: publish final staged-test closure and keep Sprint 149 Windows
  install-validation parity as a separate decision/evidence track.

### Day 14: Closeout Handoff

- Reviewed Sprint 148 artifacts and confirmed every planned artifact from Day 1
  through Day 13 exists.
- Published the final staged-test closure outcome: `test_threads`,
  `test_sprint4_integration`, and `test_fuzz` are promoted into the reviewed
  Windows CMake subset.
- Confirmed the support boundary after Sprint 148 remains narrow:
  `EXPECTED_WINDOWS_CTEST_COUNT=59`, hosted Windows CMake configure/build/CTest
  is the reviewed proof surface, and Windows Makefile/pkg-config/install-parity,
  shared-library, package-manager, runtime-loader, dynamic ABI, and broad
  Windows parity claims remain out of scope.
- Recorded the Sprint 149 handoff: Windows install/downstream confidence
  remains supplemental and install-validation parity must be decided with
  separate evidence.
- Created the Day 14 closeout artifact in
  `artifacts/day14-closeout-handoff.md`.
- Lightweight documentation validation passed: stale public/support
  staged-exclusion wording search, trailing-whitespace check over touched docs
  and Sprint 148 artifacts, and `git diff --check`.
- Retrospective input: the helper-backed approach closed all three staged-test
  blockers without splitting targets; hosted Windows proof remains pending PR
  CI and should be verified during PR review.
