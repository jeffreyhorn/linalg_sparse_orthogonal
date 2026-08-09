# Sprint 144 Day 6 CI Promotion Design

## Purpose

Design CI follow-through for the selected Sprint 144 lane after Day 5 promoted
the macOS static-first package jobs in `.github/workflows/macos-ci.yml`.

## Selected Lane

The selected CI lane is macOS reviewed static-first install/export proof:

- `install-and-pkgconfig`
  - job name: `macOS reviewed static-first install and pkg-config proof`
  - command: `bash tests/test_install.sh`
- `cmake-install-export`
  - job name: `macOS reviewed static-first CMake install/export proof`
  - commands:
    - `bash tests/test_cmake_install.sh`
    - `bash scripts/static_package_deferral_check.sh`

The promoted lane is hosted-runner proof for the static archive package
contract only.

## Workflow State After Day 5

| Workflow | Current state | Day 7 design decision |
| --- | --- | --- |
| `.github/workflows/macos-ci.yml` | Selected package jobs are named reviewed static-first proof; proof commands are unchanged; non-claims are explicit. | Keep commands unchanged. Add only targeted failure-context comments if review shows they would improve triage. |
| `.github/workflows/ci.yml` | Linux remains enforced source-of-truth reviewed baseline with reviewed static-first package contract. | Keep unchanged unless consistency scan finds wording that conflicts with macOS promotion. |
| `.github/workflows/windows-ci.yml` | Windows remains reviewed CMake subset plus supplemental CMake install/downstream confidence; staged tests remain explicit. | Keep unchanged because backup lane was not activated. |

## Exact CI Changes For Selected Lane

Day 5 already made the necessary support-tier changes in macOS CI. Day 7 should
not add new package mechanics unless a validation failure requires it.

| CI surface | Required Day 7 action | Non-action |
| --- | --- | --- |
| macOS top comment | Confirm it names reviewed static-first install/export proof and preserves non-claims. | Do not add shared-library, dynamic ABI, package-manager, or platform-wide claims. |
| macOS package job names | Confirm both selected jobs advertise reviewed static-first proof. | Do not rename Apple Clang or Homebrew GCC jobs unless needed for clarity. |
| macOS package step names | Confirm both selected package steps advertise reviewed proof. | Do not change command bodies. |
| macOS package commands | Preserve existing commands. | Do not add artifacts, retries, or matrix entries without a concrete proof need. |
| Linux workflow | Confirm strongest reviewed source-of-truth wording still makes sense. | Do not weaken Linux package-contract ownership. |
| Windows workflow | Confirm supplemental install/downstream wording remains unchanged. | Do not promote Windows install validation or staged tests. |

## Expected Count And Staged-Exclusion Policy

The selected macOS package promotion does not touch CTest registration or
compiled test staging.

| Count or exclusion | Policy |
| --- | --- |
| Windows `EXPECTED_WINDOWS_CTEST_COUNT=56` | Unchanged. This count is owned by the Windows reviewed CMake subset and is unrelated to macOS package promotion. |
| Windows staged exclusions | Unchanged: `test_threads`, `test_sprint4_integration`, and `test_fuzz` remain outside the reviewed Windows subset. |
| Linux test registration | Unchanged. Linux package contract remains the reviewed source-of-truth package lane. |
| macOS CTest counts | No new expected-count assertion is introduced. The promoted macOS package lane owns install/export script proof, not CTest count governance. |

If future work adds a macOS CTest count gate, it must be a separate reviewed
lane decision with an explicit count owner.

## Failure Message Draft

Day 7 should preserve or add failure context that points maintainers to exact
proof owners:

| Failure area | Preferred failure context |
| --- | --- |
| macOS Make install/`pkg-config` proof | Failure is owned by `tests/test_install.sh` and the reviewed macOS static-first Make install/`pkg-config` lane. |
| macOS CMake install/export proof | Failure is owned by `tests/test_cmake_install.sh` and the reviewed macOS static-first CMake install/export lane. |
| macOS static deferral proof | Failure is owned by `scripts/static_package_deferral_check.sh` and means the static-first package boundary drifted. |
| Cross-workflow claim mismatch | Failure is owned by workflow/docs support-tier wording, not by package mechanics. |

Failure text must continue to exclude:

- shared-library packaging;
- dynamic ABI compatibility;
- runtime-loader compatibility;
- package-manager support;
- static/shared selectors;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows reviewed install-validation parity;
- broader macOS platform parity.

## Artifact Proof Checklist

The selected lane does not need artifact upload. Its reviewed proof is the
hosted job log plus script output:

| Artifact/proof | Day 7 decision |
| --- | --- |
| GitHub Actions macOS job logs | Required hosted proof after PR run. |
| `tests/test_install.sh` output | Required log output for Make install/`pkg-config` proof. |
| `tests/test_cmake_install.sh` output | Required log output for CMake install/export and downstream consumer proof. |
| `scripts/static_package_deferral_check.sh` output | Required log output for no-shared/no-selector deferral proof. |
| Uploaded artifacts | Not required; scripts print enough pass/fail detail and do not generate stable artifacts. |
| Report rows | Source-controlled advisory semantics only; not fresh hosted-run proof. |

## Workflow Validation Checklist

Day 7 should run these checks after any workflow edits:

```bash
ruby -e 'require "yaml"; ARGV.each { |p| YAML.load_file(p) }' \
  .github/workflows/ci.yml \
  .github/workflows/macos-ci.yml \
  .github/workflows/windows-ci.yml
! rg -n "supplemental .*install|install.*supplemental|confidence path|not a reviewed macOS install/export|do not claim reviewed install/export" .github/workflows/macos-ci.yml
rg -n "macOS reviewed static-first|shared-library packaging|dynamic ABI compatibility|runtime-loader compatibility|package-manager support|static/shared selectors|broader macOS platform parity" .github/workflows/macos-ci.yml
rg -n "Windows supplemental CMake install/downstream|Windows reviewed scope remains CMake-first|Windows staged exclusions remain" .github/workflows/windows-ci.yml
rg -n "Linux.*reviewed static-first package|Linux.*source-of-truth" .github/workflows/ci.yml
git diff --check
```

Day 7 should also run focused local package proof only if it changes package
commands or scripts. If Day 7 remains workflow wording only, YAML and claim
scans are sufficient.

## Cross-Workflow Consistency Decisions

| Question | Decision |
| --- | --- |
| Does Linux need a workflow change now that macOS package proof is reviewed? | No. Linux remains strongest reviewed source of truth and already says its package lane is reviewed. |
| Does Windows need a workflow change? | No. Backup lane was not activated; Windows remains reviewed CMake subset plus supplemental install/downstream confidence. |
| Does macOS need artifact uploads? | No. The reviewed proof is the hosted job log and script output. |
| Does macOS need expected counts? | No. The selected lane is install/export proof, not CTest registration governance. |
| Does Day 7 need documentation changes? | Not yet. README, INSTALL, and maintainer guide alignment are Day 9 owners unless Day 7 discovers a workflow contradiction. |

## Day 7 Implementation Plan

Day 7 should:

1. Re-open `.github/workflows/macos-ci.yml` and confirm selected-lane job names,
   comments, and steps are still coherent after Day 5.
2. Add targeted failure-context comments only if they improve ownership without
   broadening claims.
3. Parse all workflow YAML files with Ruby.
4. Run selected-lane stale wording and non-claim scans.
5. Run Linux and Windows preservation scans.
6. Run `git diff --check`.
7. Record Day 7 CI implementation evidence in `WORKING_NOTES.md` and a Day 7
   artifact.

## Day 6 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Workflow changes are tied to selected-lane evidence. | Complete | CI design points macOS package promotion to the exact proof scripts and hosted job logs. |
| Expected counts and staged exclusions have clear ownership. | Complete | Windows CTest count and staged exclusions remain Windows-owned and unchanged; macOS promotion does not introduce counts. |
| CI messages distinguish promoted support from remaining non-claims. | Complete | Failure-message draft and non-claim list keep selected macOS static-first proof separate from unsupported package/platform claims. |
