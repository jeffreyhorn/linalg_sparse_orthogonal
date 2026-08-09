# Sprint 144 Day 7 CI Promotion Implementation

## Purpose

Complete selected-lane workflow implementation follow-through for macOS
reviewed static-first install/export proof. Day 7 verifies that Day 5's
workflow promotion is coherent, preserves proof commands, keeps unrelated
platform lanes unchanged, and passes locally available syntax and proof checks.

## Implementation Decision

No additional workflow edit was required on Day 7.

Day 5 already updated `.github/workflows/macos-ci.yml` to:

- name the Make install/`pkg-config` job as reviewed macOS static-first proof;
- name the CMake install/export job as reviewed macOS static-first proof;
- remove selected-lane supplemental/confidence wording from macOS package jobs;
- preserve non-claims for shared-library packaging, dynamic ABI compatibility,
  runtime-loader compatibility, package-manager support, static/shared
  selectors, and broader macOS platform parity.

Day 7 confirmed that the implementation matches the Day 6 CI design and should
not add new package mechanics, artifacts, retries, matrix entries, or expected
CTest counts.

## Workflow Files

| Workflow | Day 7 status | Notes |
| --- | --- | --- |
| `.github/workflows/macos-ci.yml` | Promoted selected lane | Package proof jobs are reviewed macOS static-first install/export proof. |
| `.github/workflows/ci.yml` | Unchanged | Linux remains the strongest reviewed source-of-truth baseline and reviewed static-first package contract owner. |
| `.github/workflows/windows-ci.yml` | Unchanged | Windows remains reviewed CMake subset plus supplemental CMake install/downstream confidence; staged exclusions remain explicit. |

## Selected-Lane CI Proof Steps

| Job key | Job name | Proof command(s) | Day 7 action |
| --- | --- | --- | --- |
| `install-and-pkgconfig` | `macOS reviewed static-first install and pkg-config proof` | `bash tests/test_install.sh` | Confirmed unchanged and mirrored locally. |
| `cmake-install-export` | `macOS reviewed static-first CMake install/export proof` | `bash tests/test_cmake_install.sh`; `bash scripts/static_package_deferral_check.sh` | Confirmed unchanged and mirrored locally. |

## Expected Count And Exclusion Updates

No expected CTest count or staged-exclusion update was required.

| Surface | Day 7 result |
| --- | --- |
| Windows `EXPECTED_WINDOWS_CTEST_COUNT` | Unchanged at `56`; owned by Windows reviewed CMake subset. |
| Windows staged exclusions | Unchanged: `test_threads`, `test_sprint4_integration`, and `test_fuzz`. |
| macOS package proof | No CTest count introduced; package proof is owned by install/export scripts. |
| Linux package contract | Unchanged; remains reviewed source-of-truth package contract lane. |

## Local Mirrored Proof

The selected macOS CI commands are shell-based and locally feasible. Day 7 ran
them as implementation evidence, while preserving the boundary that hosted
`macos-latest` CI remains the final reviewed-platform proof owner.

| Command | Result |
| --- | --- |
| `bash tests/test_install.sh` | Passed: 23 passed, 0 failed |
| `bash tests/test_cmake_install.sh` | Passed: 26 passed, 0 failed, 0 skipped |
| `bash scripts/static_package_deferral_check.sh` | Passed |

## Workflow Syntax And Support-Tier Checks

| Check | Result |
| --- | --- |
| Ruby YAML parse for `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, and `.github/workflows/windows-ci.yml` | Passed |
| macOS stale selected-lane supplemental wording scan | Passed |
| macOS reviewed static-first and unsupported-claim boundary scan | Passed |
| Linux source-of-truth reviewed package scan | Passed |
| Windows reviewed CMake subset, supplemental install/downstream, and staged-exclusion scan | Passed |
| `git diff --check` | Passed |

Commands:

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

## Failure Message And Non-Claim Review

The macOS package jobs now fail under names that point to exact proof owners:

- `macOS reviewed static-first install and pkg-config proof`;
- `macOS reviewed static-first CMake install/export proof`.

The workflow comments keep these non-claims explicit:

- no shared-library packaging;
- no dynamic ABI compatibility;
- no runtime-loader compatibility;
- no package-manager support;
- no static/shared selectors;
- no broader macOS platform parity.

Windows failure messages still identify staged exclusions and keep Windows
install/downstream proof supplemental. Linux remains unchanged.

## Hosted Evidence Boundary

Day 7 local validation supports implementation confidence, but the reviewed
macOS platform claim is fully earned only when GitHub Actions runs the promoted
macOS jobs on `macos-latest`.

Day 12 should revisit hosted CI status if available and keep this boundary
visible in closeout.

## Day 8 Handoff

Day 8 should evaluate package/report integration for the selected lane:

1. Inspect `tests/corpus/manifests/report_families.tsv` package and CI rows.
2. Decide whether source-controlled report semantics need to mention the new
   macOS reviewed static-first package lane.
3. Preserve the rule that report rows identify proof owners and do not
   manufacture fresh hosted-run evidence.
4. Run package report normalization and freshness checks if report rows change.

## Day 7 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected-lane CI path reflects the intended support tier. | Complete | macOS workflow names selected package jobs as reviewed static-first proof. |
| Failure messages explain remaining staged blockers without implying support. | Complete | macOS non-claims remain explicit; Windows staged-exclusion messages remain unchanged. |
| Workflow syntax and locally feasible mirrored checks pass. | Complete | YAML parse, support-tier scans, local package proofs, static deferral check, and `git diff --check` passed. |
