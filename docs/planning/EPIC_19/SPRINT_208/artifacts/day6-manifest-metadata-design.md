# Sprint 208 Day 6: Manifest Metadata Design

## Purpose

Map the Day 5 re-deferral decision to exact selected-target manifest state,
workflow metadata expectations, guard ownership, and documentation wording.
Day 6 is a design day; Day 7 owns implementation.

## Selected Decision

Sprint 208 keeps `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` re-deferred for Windows
selected freshness. The latest hosted Windows artifact is accepted as current
guarded workflow evidence, but positive selected manifest metadata for Windows
is not added because generated support tier and non-claim wording still
contradict promotion.

## Required Current Manifest State

| Field | Required value | Reason |
| --- | --- | --- |
| `target_id` | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` | Exact selected Cholesky row identity. |
| `family` | `comparison` | Selected comparison target. |
| `subfamily` | `cholesky_spd_tridiag_5` | Must match generated rows and artifact directory. |
| `target_key` | `cholesky-spd-tridiag-5` | Must match workflow generator and freshness commands. |
| `selection_scope` | `reviewed_cross_platform_selected` | Existing selected comparison scope. |
| `support_tier` | `local_only` | Generated rows and summary still use local-only semantics. |
| `freshness_policy` | `generated_compare_inputs` | Generated comparison row freshness policy. |
| `artifact_pattern` | `build/comparison/cholesky_spd_tridiag_5/study.tsv` | Manifest-owned selected artifact pattern. |
| `required_files` | `project_observations.tsv;baseline_observations.tsv;dependency_status.tsv;study.tsv;summary.md;manifest.tsv` | Exact selected artifact membership. |
| `expected_rows` | `6` | Exact selected Cholesky row count. |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` | Positive manifest metadata remains Linux/macOS only. |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness` | Positive manifest metadata remains Linux/macOS only. |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness` | Positive manifest metadata remains Linux/macOS only. |
| `workflow_platforms` | `linux;macos` | Windows remains absent while re-deferred. |
| `non_claims` | Must include `no Windows report freshness`, `no package-manager proof`, `no shared-library ABI proof`, `no performance superiority`, and `no state-of-the-art claim`. | Keeps retained non-claims explicit. |

## Current Workflow Metadata Map

| Workflow surface | Required state |
| --- | --- |
| Windows workflow file | `.github/workflows/windows-ci.yml` exists and remains outside positive selected manifest metadata. |
| Windows job ID | `selected-comparison-freshness` |
| Windows job name | `Windows selected Cholesky comparison freshness (MSVC)` |
| Generator command | `python scripts/run_external_comparison.py --target cholesky-spd-tridiag-5 --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib` |
| Freshness command | `python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` |
| Artifact name | `sprint190-windows-selected-comparison-cholesky` |
| Upload scope | Exact six files under `build/comparison/cholesky_spd_tridiag_5/` |
| Upload failure behavior | `if-no-files-found: error` |

This workflow metadata is guarded workflow evidence only. It must not be copied
into positive selected target manifest metadata until generated support tier,
claim scope, and non-claims are promoted together.

## Guard Coverage Plan

| Guard | Required Day 7+ behavior |
| --- | --- |
| `test_windows_report_freshness_deferral_keeps_manifest_unselected` | Continue failing if any selected target lists `windows` while Windows report freshness is formally deferred. |
| New selected Cholesky current-state manifest contract | Assert exact current Cholesky row identity, required files, expected rows, expected row IDs, support tier, workflow metadata, claim scope, and non-claims. |
| `test_future_windows_cholesky_metadata_allowlist_accepts_exact_row` | Continue accepting only the exact Cholesky row if future Windows metadata is added. |
| Future Windows Cholesky allowlist negative tests | Extend to reject future promotion if support tier, claim scope, or non-claims still retain re-deferral wording that contradicts promotion. |
| `validate_windows_selected_cholesky_lane` | Continue requiring target-specific generator/freshness commands, artifact name, exact upload files, and no broad comparison upload. |
| `validate_manifest_windows_deferral` | Continue requiring selected manifest to omit Windows platforms while re-deferred. |
| `test_windows_report_freshness_keeps_bounded_cholesky_only` | Continue enforcing bounded workflow evidence, QR incompatible re-deferral, and no broad selected report freshness lane. |

## Implementation Boundary

Day 7 may modify:

- `tests/test_selected_report_targets_manifest.py`;
- `scripts/validate_windows_powershell.py` only if an unguarded metadata drift
  is found;
- `tests/test_validate_windows_powershell.py` only if PowerShell guard
  coverage changes;
- `tests/test_selected_comparison_workflow.py` only if workflow guard coverage
  gaps are found;
- Sprint 208 working notes and artifacts.

Day 7 must not modify:

- `tests/corpus/manifests/selected_report_targets.tsv` to add Windows positive
  metadata;
- generated Cholesky support-tier or non-claim text;
- public docs before the Day 11 documentation calibration pass, except for a
  guard-required marker fix;
- `.c` or `.h` files unless a concrete validation issue requires it.

## Documentation Wording Design

Future Sprint 208 documentation should use this vocabulary:

- "latest hosted Windows Cholesky evidence was reviewed";
- "the exact `cholesky-spd-tridiag-5` workflow path passed";
- "selected Windows Cholesky freshness remains re-deferred";
- "positive selected manifest metadata remains Linux/macOS only";
- "generated support tier and generated non-claims remain the blocker";
- "no broad Windows report freshness, Windows oracle freshness, Windows
  benchmark freshness, QR incompatible Windows freshness, package/ABI support,
  performance claim, release claim, or state-of-the-art claim is promoted."

## Day 6 Outcome

Item 208.3 has an implementation-ready metadata design. The selected manifest
will remain Windows-absent, with Day 7 focused on stronger current-state and
future-promotion guard coverage rather than manifest promotion.

