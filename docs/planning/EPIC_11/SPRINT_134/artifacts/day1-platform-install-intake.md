# Sprint 134 Day 1 - Platform Install Intake

## Purpose

Day 1 establishes the Sprint 134 baseline for cross-platform install support,
Windows staged lanes, CI tier follow-through, artifact ownership, and claim
boundaries.

## Project Plan Mapping

| Item | Project-plan scope | Sprint 134 day owners |
| --- | --- | --- |
| 1 | Platform Gap Audit | Days 1-2 |
| 2 | Linux Install CI Decision | Days 3-4 |
| 3 | macOS Install/Export Follow-Through | Days 5-7 |
| 4 | Windows Install Validation Design | Days 8-9 |
| 5 | Windows Staged Test Follow-Through | Days 10-11 |
| 6 | Validation | Days 12-13 |
| 7 | Closeout | Day 14 |

## Inherited Sprint 133 Baseline

| Surface | Inherited truth |
| --- | --- |
| Package contract | Static-first install/export surface is maintained. |
| Shared-library support | Deferred; `BUILD_SHARED_LIBS=ON` is rejected. |
| Dynamic ABI compatibility | Deferred; version metadata is package/source metadata, not ABI policy. |
| Package-manager support | Deferred; no manager recipes or manager-specific consumer proof. |
| Local package gates | `tests/test_install.sh`, `tests/test_cmake_install.sh`, and `scripts/static_package_deferral_check.sh`. |
| Platform install parity | Not broad or symmetric; Linux, macOS, and Windows have distinct support tiers. |

## Current Platform Tier Inventory

| Platform | Current reviewed tier | Supplemental/local tier | Current non-claims |
| --- | --- | --- | --- |
| Linux | Reviewed Makefile compile-quality path, reviewed CMake parity path, and reviewed dead-code report/check path in `.github/workflows/ci.yml`. | Supplemental direct runtime, sanitizer, benchmark, TSan, and coverage jobs. Local package proof scripts remain developer-side unless promoted. | No separate reviewed install CI lane yet; no package-manager or shared-library support. |
| macOS | Reviewed Apple Clang compile-quality, reviewed CMake parity, wall-check, and sanitizer path in `.github/workflows/macos-ci.yml`. | Supplemental Homebrew GCC direct build/test/wall-check and supplemental static-first Make install/`pkg-config` job. | Supplemental Make install/`pkg-config` does not imply reviewed macOS CMake install/export parity or broad platform install parity. |
| Windows | Reviewed MSVC CMake-first consumer subset in `.github/workflows/windows-ci.yml`. | CTest count inspection and explicit staged exclusion output. | No Windows Makefile parity, no separate reviewed install-validation lane, and no reviewed Windows property/fuzz/thread lane for staged exclusions. |

## Workflow Surface Inventory

| File | Current relevant jobs or checks | Sprint 134 questions |
| --- | --- | --- |
| `.github/workflows/ci.yml` | Linux `build-and-test`, `cmake-build-and-test`, `tsan`, `lint`, `deadcode`, and `coverage`. | Should any package proof become a reviewed Linux install CI lane, and what runtime/tooling cost follows? |
| `.github/workflows/macos-ci.yml` | macOS `build-and-test` matrix and separate `install-and-pkgconfig` supplemental job. | Should CMake install/export parity be reviewed, supplemental, or explicitly deferred on macOS? |
| `.github/workflows/windows-ci.yml` | Windows MSVC configure/build, `ctest -N`, expected count check, and full `ctest`. | Should install/downstream proof be implemented, and should any staged exclusions move into reviewed CTest membership? |

## Windows CTest Baseline

| Field | Current state |
| --- | --- |
| Workflow expected count | `EXPECTED_WINDOWS_CTEST_COUNT=54`. |
| Reviewed Windows lane | Configure + build + `ctest -N` + full `ctest` with MSVC CMake. |
| Staged exclusions named in workflow | `test_threads`, `test_sprint4_integration`, `test_fuzz`. |
| Property/fuzz status | `test_fuzz` and bounded lifecycle property coverage remain outside the reviewed Windows subset. |
| Makefile status | Windows Makefile wrappers and dead-code flow remain staged/non-claimed. |
| Install-validation status | No separate reviewed Windows install-validation lane. |

## Candidate Decisions

| Decision | Current default | Day owner |
| --- | --- | --- |
| Promote Linux install proof to reviewed CI | Undecided; local package proof remains developer-side today. | Days 3-4 |
| Add macOS CMake install/export parity | Undecided; current macOS package job is Make install/`pkg-config` supplemental proof. | Days 5-7 |
| Add Windows install/downstream proof | Undecided; current Windows reviewed scope is CMake-first consumer subset. | Days 8-9 |
| Promote Windows staged tests | Undecided; `test_threads`, `test_sprint4_integration`, and `test_fuzz` remain staged today. | Days 10-11 |
| Update support-tier docs/comments | Pending platform decisions. | Day 12 |

## Claim Fences

- Reviewed CI means an explicit workflow lane with clear ownership, expected
  pass criteria, and failure triage expectations.
- Supplemental CI means confidence-building coverage that does not widen the
  primary reviewed support tier.
- Local proof means developer-side evidence unless promoted by a workflow
  decision.
- Staged exclusions remain non-reviewed until CTest membership, expected
  counts, and validation ownership change.
- Deferred install parity is a non-claim, not a hidden support promise.
- Windows CMake-first support does not imply Windows Makefile parity.
- Static-first package proof does not imply shared-library packaging, dynamic
  ABI compatibility, package-manager support, or runtime-loader behavior.

## Day 2 Handoff

Day 2 should perform the formal platform gap audit:

- compare Linux CI jobs against local package proof scripts and decide what
  evidence would be required for reviewed install CI promotion;
- compare macOS supplemental Make install/`pkg-config` proof against missing
  CMake install/export parity;
- compare Windows MSVC CMake subset, expected CTest count, install gaps,
  staged exclusions, and Makefile non-claims;
- identify any wording drift between README, INSTALL, maintainer guide, and
  workflow comments.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 134 project-plan item has a day-level owner. | Complete | Working notes and this artifact map Items 1-7 to Days 1-14. |
| Sprint 133 static-first package truth is preserved before platform changes. | Complete | Inherited baseline and claim fences retain static-first support and explicit non-claims. |
| Linux, macOS, and Windows support tiers are visible before decisions begin. | Complete | Platform tier and workflow inventories record current reviewed, supplemental, local, staged, and deferred states. |
