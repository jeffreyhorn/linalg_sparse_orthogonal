# Day 7: Manifest Promotion Decision

## Purpose

Day 7 decided whether current Sprint 203 evidence supports adding Windows
metadata to `SRT-COMP-QR-INCOMPATIBLE-LS` in the selected target manifest.

## Decision

Re-defer Windows QR incompatible selected comparison freshness promotion.

The branch has valid local evidence and guard coverage, but it does not have
hosted Windows/MSVC proof or inspected hosted Windows artifacts. The selected
target manifest should remain Linux/macOS-only for `qr-incompatible-ls`.

## Evidence Considered

| Evidence | Result | Promotion effect |
| --- | --- | --- |
| Direct local generator | Passed for `qr-incompatible-ls`. | Supports target health, not Windows promotion. |
| Local CMake probe | Passed with local default CMake generator. | Supports probe structure, not MSVC behavior. |
| Target-specific freshness | Passed for all six QR incompatible rows. | Supports local selected freshness. |
| QR Windows path tests | Passed after Day 6 additions. | Reduces artifact matching risk. |
| Selected manifest tests | Passed. | Confirms current source-of-truth metadata remains coherent. |
| Selected workflow tests | Passed. | Confirms current workflow guards remain coherent. |
| Hosted Windows/MSVC run | Not available. | Blocks promotion. |
| Hosted Windows artifact inspection | Not available. | Blocks promotion. |
| Generated support tier | Still `local_only`. | Blocks promotion unless changed with coordinated evidence/docs. |

## Manifest Fields Left Unchanged

| Field | Retained value |
| --- | --- |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness` |
| `workflow_platforms` | `linux;macos` |
| `support_tier` | `local_only` |

## Required Before Promotion

- Hosted Windows configure/build/probe pass using Visual Studio 2022, x64, and
  Release.
- Hosted freshness pass with `--selected-target qr-incompatible-ls`.
- Inspected artifact bundle containing exactly the six required QR
  incompatible files.
- Manifest metadata, generated support tier, claim scope, non-claims, workflow
  guard, public docs, corpus docs, and maintainer guide updated together.

## Retained Non-Claims

The re-deferral preserves no broad QR parity, no broad least-squares parity, no
external-library ecosystem parity, no Windows report freshness, no
package-manager proof, no shared-library ABI proof, no performance
superiority, and no state-of-the-art claim.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 tests/test_normalize_report_index.py` | Passed. |
