# Sprint 144 Day 2 Platform Lane Selection

## Purpose

Score the Sprint 144 candidate platform lanes, select exactly one lane for
complete closure, identify a backup lane, and define the selected-lane
promotion criteria before implementation design begins.

## Scoring Method

Scores use a 1-5 scale where `5` is strongest for promotion viability.

| Score axis | Meaning |
| --- | --- |
| User value | How much the lane improves real adoption or support confidence. |
| Current evidence | How much proof already exists in source, scripts, docs, and CI. |
| Blocker severity | How small and well-understood the remaining blockers are. |
| Implementation cost | How likely the lane is to close without broad refactors. |
| CI cost | How easy the lane is to express and maintain in CI. |
| Portability risk | How unlikely the lane is to require platform-specific source churn. |
| Documentation impact | How cleanly public and maintainer docs can describe the new status. |

Higher total score indicates a better Sprint 144 lane because the sprint should
close one gap completely rather than partially address several.

## Lane Scoring Table

| Candidate lane | User value | Current evidence | Blocker severity | Implementation cost | CI cost | Portability risk | Documentation impact | Total | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| macOS reviewed install/export parity | 4 | 5 | 4 | 5 | 5 | 5 | 4 | 32 | Select primary |
| Windows reviewed CMake static install/downstream parity | 5 | 4 | 3 | 4 | 4 | 3 | 4 | 27 | Backup |
| Linux source-of-truth strengthening | 3 | 5 | 5 | 4 | 5 | 5 | 3 | 30 | Defer due lower marginal value |
| Windows staged test portability | 4 | 2 | 1 | 2 | 3 | 1 | 3 | 16 | Defer due source blockers |

## Selected Lane

Sprint 144 selects **macOS reviewed install/export parity** for complete
closure.

The selected scope is intentionally narrow:

- promote the existing macOS static-first Make install/`pkg-config` job from
  supplemental confidence to reviewed macOS install proof;
- promote the existing macOS static-first CMake install/export job from
  supplemental confidence to reviewed macOS install/export proof;
- keep the scope limited to the static-first package contract implemented in
  Sprint 143;
- preserve all shared-library, dynamic ABI, runtime-loader, package-manager,
  static/shared selector, Windows Makefile, Windows `pkg-config`, and Windows
  install-validation non-claims.

## Why macOS Was Selected

macOS is the best Sprint 144 closure candidate because:

1. The hosted workflow already runs the relevant package proof commands:
   `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, and
   `bash scripts/static_package_deferral_check.sh`.
2. The lane does not require public C/header changes.
3. The lane does not require Windows-only local reproduction.
4. The existing jobs already use the maintained static-first package proof
   scripts that Linux reviewed package CI runs.
5. The remaining work is primarily support-tier ownership, CI wording,
   failure-message clarity, report integration, documentation alignment, and
   final validation.
6. The promotion can close a real platform-support ambiguity without implying
   shared-library or dynamic ABI support.

## Backup Lane

The backup lane is **Windows reviewed CMake static install/downstream parity**.

This lane should be used only if macOS promotion becomes blocked by hosted CI
or support-tier evidence. Its scope would be limited to the existing MSVC
CMake-first static install/downstream proof:

- installed static `.lib`;
- installed headers;
- CMake package metadata;
- installed `sparse.pc` static-description metadata check;
- no DLL artifacts;
- installed example configure/build/run;
- exact-version consumer configure/build/run;
- mismatched-version rejection.

The backup would still preserve non-claims for Windows Makefile parity, Windows
`pkg-config` parity, shared libraries, dynamic ABI, package-manager support,
and broader Windows source/test parity.

## Deferred Lane Rationale

| Deferred lane | Defer reason |
| --- | --- |
| Linux source-of-truth strengthening | Linux already has reviewed Makefile compile-quality, reviewed CMake parity, dead-code, and reviewed static-first package-contract proof. It scored well technically, but it does not close the more visible platform-promotion ambiguity left by Sprint 143. |
| Windows staged test portability | The known blockers are source-level: pthread APIs for `test_threads` and `test_sprint4_integration`, and POSIX temp-file APIs for `test_fuzz`. Closing that lane likely requires C/test portability work and intentional CTest count promotion, making it higher-risk than the package-lane closure. |

## Selected-Lane Promotion Criteria

macOS reviewed install/export parity is complete only if all of the following
criteria are met:

1. `.github/workflows/macos-ci.yml` clearly identifies the static-first Make
   install/`pkg-config` and CMake install/export jobs as reviewed macOS
   install/export proof for the static-first package contract.
2. Workflow comments and job names do not imply shared-library packaging,
   dynamic ABI compatibility, runtime-loader compatibility, package-manager
   support, static/shared selectors, or broader platform parity.
3. README, INSTALL, and maintainer guide support-tier wording agrees with the
   promoted macOS static-first package status.
4. Package/report/freshness artifacts reference the macOS reviewed proof
   without treating report rows as fresh hosted-run evidence.
5. The static-first package deferral guard remains in the macOS CMake
   install/export proof path.
6. Locally feasible checks pass for touched docs, scripts, workflow syntax,
   package report indexes, and install/export scripts.
7. Any hosted-only validation dependency is explicitly documented with failure
   ownership.

## Rejection Criteria

Reject macOS promotion and fall back to the Windows CMake static package lane
if any of the following occur:

- macOS hosted proof cannot be represented by CI without ambiguous support-tier
  claims;
- the selected lane requires source portability changes outside package proof;
- static-first package validation cannot remain identical to or stricter than
  Linux package-contract validation;
- documentation cannot distinguish macOS reviewed static-first install/export
  proof from shared-library, dynamic ABI, package-manager, or full platform
  parity claims.

## Validation And Evidence Checklist

| Evidence owner | Required Day 3-14 follow-through |
| --- | --- |
| `.github/workflows/macos-ci.yml` | Update comments/job names/failure context to mark selected static-first package jobs as reviewed macOS install/export proof. |
| `.github/workflows/ci.yml` | Preserve Linux as strongest reviewed source of truth and ensure macOS promotion does not weaken Linux package-contract wording. |
| `.github/workflows/windows-ci.yml` | Preserve Windows reviewed CMake subset and supplemental install/downstream status unless backup lane is activated. |
| `README.md` | Align cross-platform CI contract with selected macOS reviewed static-first install/export proof. |
| `INSTALL.md` | Update supported-platform table and install validation interpretation. |
| `docs/maintainer_guide.md` | Update maintainer support-tier interpretation and package proof ownership. |
| package/report artifacts | Add or update source-owned references if needed, while preserving freshness semantics. |
| validation commands | Run `git diff --check`, relevant workflow syntax checks, package/report checks, and focused install/export checks where feasible. |

## Day 3 Handoff

Day 3 should establish the before-change blocker and evidence baseline for the
selected macOS lane:

- capture current macOS workflow job names and comments;
- identify exact comments/docs that say the macOS package jobs are
  supplemental;
- confirm current package proof commands used by macOS jobs;
- define the expected post-promotion wording;
- keep Windows and Linux wording unchanged except where needed to preserve
  cross-platform consistency.

## Day 2 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Exactly one platform lane is selected for complete closure. | Complete | macOS reviewed install/export parity selected as the primary lane. |
| Non-selected lanes have explicit defer reasons. | Complete | Backup and deferred lane rationale sections above. |
| Selected-lane proof requirements are concrete enough for design work. | Complete | Promotion criteria, rejection criteria, validation checklist, and Day 3 handoff define required proof owners. |
