# Sprint 144 Day 5 Source And Script Fix Batch

## Purpose

Implement the selected-lane support-tier fix for macOS reviewed static-first
install/export parity while preserving the existing package proof commands and
Sprint 143 static-first package boundaries.

## Implementation Summary

Day 5 updated `.github/workflows/macos-ci.yml` only.

The implementation:

- promotes the macOS Make install/`pkg-config` package job from supplemental
  confidence to reviewed macOS static-first package proof;
- promotes the macOS CMake install/export package job from supplemental
  confidence to reviewed macOS static-first package proof;
- keeps the proof commands unchanged;
- preserves the Homebrew GCC leg as supplemental second-compiler coverage;
- preserves the existing Apple Clang compile-quality, CMake, wall-check, and
  sanitizer lane as reviewed macOS platform proof;
- preserves static-first non-claims from Sprint 143.

## Files Changed

| File | Change | Reason |
| --- | --- | --- |
| `.github/workflows/macos-ci.yml` | Updated top support-tier comment. | Names the selected macOS package jobs as reviewed static-first install/export proof and preserves non-claims. |
| `.github/workflows/macos-ci.yml` | Renamed `install-and-pkgconfig` job from supplemental confidence path to reviewed proof. | CI status now reflects the promoted support tier. |
| `.github/workflows/macos-ci.yml` | Renamed Make install/`pkg-config` proof step from maintained supplemental proof to reviewed proof. | Step output now points to the reviewed proof owner. |
| `.github/workflows/macos-ci.yml` | Renamed `cmake-install-export` job from supplemental confidence path to reviewed proof. | CI status now reflects the promoted support tier. |
| `.github/workflows/macos-ci.yml` | Renamed CMake install/export proof step from maintained supplemental proof to reviewed proof. | Step output now points to the reviewed proof owner. |

## Commands Preserved

Day 5 intentionally did not change selected-lane commands:

```yaml
run: bash tests/test_install.sh
run: bash tests/test_cmake_install.sh
run: bash scripts/static_package_deferral_check.sh
```

Preserving the commands keeps Day 5 as support-tier promotion rather than
package-mechanics churn.

## Source/Script/Build Checklist

| Surface | Status | Notes |
| --- | --- | --- |
| Public headers | Unchanged | No API or ABI work belongs to this lane. |
| `.c` files | Unchanged | No source portability blocker was identified on Day 3. |
| `Makefile` | Unchanged | Make install behavior already passed Day 3 local proof. |
| `tests/test_install.sh` | Unchanged | Existing script remains the Make install/`pkg-config` proof owner. |
| `tests/test_cmake_install.sh` | Unchanged | Existing script remains the CMake install/export proof owner. |
| `scripts/static_package_deferral_check.sh` | Unchanged | Existing guard remains attached to macOS CMake install/export proof. |
| CMake package metadata | Unchanged | No install/export metadata blocker exists for selected lane. |
| `sparse.pc.in` | Unchanged | No `pkg-config` metadata blocker exists for selected lane. |
| `.github/workflows/macos-ci.yml` | Changed | Support-tier comments and job/step names updated. |

## Focused Assertions

The selected workflow now asserts these claims through comments, job names, and
unchanged proof commands:

| Assertion | Evidence |
| --- | --- |
| macOS Make install/`pkg-config` is reviewed static-first package proof. | `install-and-pkgconfig` job name and `bash tests/test_install.sh`. |
| macOS CMake install/export is reviewed static-first package proof. | `cmake-install-export` job name and `bash tests/test_cmake_install.sh`. |
| macOS package proof keeps static-first deferrals executable. | `bash scripts/static_package_deferral_check.sh` remains in the CMake install/export job. |
| macOS Homebrew GCC remains supplemental. | Top comment and matrix leg wording still identify it as supplemental second-compiler coverage. |
| Promotion does not imply wider package claims. | Workflow comments explicitly exclude shared-library packaging, dynamic ABI, runtime-loader compatibility, package-manager support, static/shared selectors, and broader macOS platform parity. |

## Promoted And Staged Surfaces

| Surface | Day 5 status |
| --- | --- |
| macOS Make install/`pkg-config` package proof | Promoted to reviewed static-first install proof. |
| macOS CMake install/export package proof | Promoted to reviewed static-first install/export proof. |
| macOS static package deferral proof | Promoted as part of reviewed CMake install/export proof. |
| macOS Homebrew GCC | Still supplemental. |
| Linux package contract | Still reviewed source of truth; unchanged. |
| Windows CMake install/downstream proof | Still supplemental; unchanged. |
| Windows staged `test_threads`, `test_sprint4_integration`, and `test_fuzz` | Still staged; unchanged. |

## Validation

Day 5 validation should confirm:

1. `.github/workflows/macos-ci.yml` parses as YAML.
2. Selected-lane stale supplemental wording is gone from the macOS package jobs.
3. Non-claim wording remains present for unsupported package and platform
   claims.
4. Whitespace checks pass.

Validation commands:

```bash
ruby -e 'require "yaml"; ARGV.each { |p| YAML.load_file(p) }' .github/workflows/macos-ci.yml
rg -n "supplemental .*install|install.*supplemental|confidence path|not a reviewed macOS install/export|do not claim reviewed install/export" .github/workflows/macos-ci.yml
rg -n "shared-library packaging|dynamic ABI compatibility|runtime-loader compatibility|package-manager support|static/shared selectors|broader macOS platform parity|reviewed static-first" .github/workflows/macos-ci.yml
git diff --check
```

## Hosted Evidence Boundary

This implementation updates the CI lane definition. It does not by itself
prove that the hosted `macos-latest` runner has executed the promoted lane
after the change. Day 7 and Day 12 should keep hosted macOS CI status as the
final proof owner for the reviewed macOS package claim.

## Day 6 Handoff

Day 6 should design CI promotion follow-through:

- confirm whether Linux workflow wording needs a consistency note now that
  macOS package proof is reviewed for the static-first lane;
- confirm Windows workflow wording stays unchanged because the backup lane was
  not activated;
- decide whether macOS package job names need expected failure text or artifact
  output changes;
- define the workflow syntax and support-tier scans that Day 7 should run.

## Day 5 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected source/script blockers are fixed or explicitly rejected with proof. | Complete | Day 3 found no source-level blocker; Day 5 fixed the support-tier workflow blocker. |
| Touched scripts/build files pass focused syntax checks. | Complete | Ruby YAML parse, stale selected-lane wording scan, non-claim scan, and `git diff --check` passed. |
| No unrelated platform lanes are promoted accidentally. | Complete | Linux and Windows workflow support tiers are unchanged; macOS Homebrew GCC remains supplemental. |
