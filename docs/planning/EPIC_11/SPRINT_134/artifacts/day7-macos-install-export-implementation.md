# Sprint 134 Day 7 - macOS Install/Export Implementation

## Purpose

Day 7 implements the Day 6 decision to add macOS CMake install/export evidence
without promoting macOS to full reviewed install/export parity.

## Implemented Workflow Change

Added a standalone supplemental job to `.github/workflows/macos-ci.yml`:

| Field | Value |
| --- | --- |
| Job id | `cmake-install-export` |
| Job name | `macOS supplemental CMake install/export confidence path` |
| Runner | `macos-latest` |
| Package proof | `bash tests/test_cmake_install.sh` |
| Static deferral proof | `bash scripts/static_package_deferral_check.sh` |

The existing reviewed Apple Clang matrix path, supplemental Homebrew GCC path,
and supplemental Make install/`pkg-config` job are preserved unchanged.

## Support-Tier Wording

Updated the macOS workflow comments, README CI summary, INSTALL platform table,
INSTALL validation interpretation, and maintainer guide to state that macOS now
carries supplemental static-first Make install/`pkg-config` and CMake
install/export confidence.

The wording keeps these boundaries explicit:

- Linux remains the strongest reviewed source of truth and owns the reviewed
  static-first package-contract lane.
- macOS package jobs are supplemental confidence lanes, not reviewed
  install/export parity.
- Windows remains the reviewed CMake-first consumer subset and does not gain a
  separate install-validation claim from this work.
- Shared-library packaging, dynamic ABI compatibility, package-manager support,
  and runtime-loader behavior remain explicit non-claims.

## Local Workflow-Equivalent Evidence

| Check | Result |
| --- | --- |
| `bash -n scripts/static_package_deferral_check.sh` | Passed |
| YAML parse for `.github/workflows/macos-ci.yml` | Passed |
| `git diff --name-only -- '*.c' '*.h'` | No C/header changes |
| `bash tests/test_cmake_install.sh` | Passed 21 checks, 0 failures, 0 skips |
| `bash scripts/static_package_deferral_check.sh` | Passed |
| `git diff --check` | Passed |
| focused trailing-whitespace scan | Passed |

The local runner is not macOS, so this evidence validates the command surfaces
and package contract locally while the added hosted macOS job remains the
source of macOS-specific confidence.

## Residual macOS Install/Export Queue

| Residual | Status |
| --- | --- |
| Reviewed macOS CMake install/export parity | Deferred pending hosted-runner history from the supplemental lane. |
| Homebrew GCC package install/export matrix | Deferred; no current support claim requires it. |
| macOS Make install/`pkg-config` plus CMake install/export consolidation | Deferred until runtime and flake behavior are known. |
| macOS shared-library/dynamic ABI support | Deferred by Sprint 133 product decision. |
| macOS package-manager support | Deferred non-claim. |

## Rollback Notes

If the hosted macOS CMake install/export job fails for runner-specific reasons,
remove only the `cmake-install-export` job and revert the corresponding
supplemental wording. The reviewed Apple Clang path, supplemental Homebrew GCC
path, and supplemental Make install/`pkg-config` job do not depend on the new
job.

If the static deferral guard fails on macOS only, treat the failure as package
wording or build-system drift and do not widen support claims until the cause
is classified.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected macOS decision is implemented or explicitly deferred. | Complete | Added the supplemental `cmake-install-export` job selected on Day 6. |
| Workflow-equivalent evidence exists for touched macOS package surfaces. | Complete | Local CMake install/export proof and static package deferral guard passed. |
| macOS support claims remain narrower than Linux unless reviewed proof exists. | Complete | Docs and workflow comments classify the new job as supplemental confidence, not reviewed parity. |
