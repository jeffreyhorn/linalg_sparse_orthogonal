# Sprint 199 Day 1: Windows Freshness Intake

## Purpose

Establish Sprint 199 scope, owner surfaces, current selected Windows Cholesky
freshness state, and initial validation evidence before changing manifest,
normalizer, workflow, or documentation behavior.

## Project-Plan Item Map

| Item | Day 1 owner mapping |
| --- | --- |
| 199.1 Hosted Evidence Review | Hosted Windows run IDs, selected Cholesky artifact files, row IDs, artifact paths, workflow job metadata, and evidence semantics. |
| 199.2 Manifest Promotion Decision | `tests/corpus/manifests/selected_report_targets.tsv`, especially `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`. |
| 199.3 Normalizer Hardening | `scripts/normalize_report_index.py` and `tests/test_normalize_report_index.py`, with emphasis on Windows path normalization, selected-target filtering, missing rows, and stale artifacts. |
| 199.4 Workflow Guard Update | `.github/workflows/windows-ci.yml`, `scripts/validate_windows_powershell.py`, `tests/test_validate_windows_powershell.py`, and `tests/test_selected_comparison_workflow.py`. |
| 199.5 Documentation Calibration | `README.md`, `INSTALL.md`, `tests/corpus/README.md`, and `docs/maintainer_guide.md`. |
| 199.6 Validation | Selected manifest, workflow, PowerShell, normalizer, generator, freshness, docs, and C quality gates based on changed surface. |

## Current Selected Target State

The selected target manifest currently records:

| Field | Current value |
| --- | --- |
| `target_id` | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| `family` | `comparison` |
| `subfamily` | `cholesky_spd_tridiag_5` |
| `target_key` | `cholesky-spd-tridiag-5` |
| `support_tier` | `local_only` |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness` |
| `workflow_platforms` | `linux;macos` |
| `expected_rows` | `6` |

Day 1 does not promote this row. Windows remains absent from
`workflow_platforms` until hosted artifact semantics, selected target metadata,
support tier, and claim contract are reviewed together.

## Windows Workflow Baseline

`.github/workflows/windows-ci.yml` contains a bounded
`selected-comparison-freshness` job named
`Windows selected Cholesky comparison freshness (MSVC)` that:

- configures and builds the library with CMake/MSVC;
- runs `python scripts/run_external_comparison.py --target
  cholesky-spd-tridiag-5 --probe-build-system cmake --cmake-generator "Visual
  Studio 17 2022" --cmake-arch x64 --cmake-config Release --library
  build/Release/sparse_lu_ortho.lib`;
- runs `python scripts/normalize_report_index.py --family comparison
  --require-generated comparison --check-freshness --selected-target
  cholesky-spd-tridiag-5`;
- uploads only the selected Cholesky comparison bundle as
  `sprint190-windows-selected-comparison-cholesky`.

This is a guarded workflow path, not a manifest promotion by itself.

## Preliminary Hosted Evidence

Day 1 identified the latest successful Windows workflow run on `master`:

| Field | Value |
| --- | --- |
| Run ID | `34269219871` |
| URL | `https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/34269219871` |
| Created | `2026-09-08T19:29:02Z` |
| Head SHA | `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450` |
| Conclusion | `success` |
| Selected job | `Windows selected Cholesky comparison freshness (MSVC)` |
| Selected job conclusion | `success` |

The selected artifact was downloaded with:

```sh
gh run download 34269219871 --name sprint190-windows-selected-comparison-cholesky --dir /tmp/sprint199-day1-windows-artifact-check
```

Downloaded files:

- `project_observations.tsv`
- `baseline_observations.tsv`
- `dependency_status.tsv`
- `study.tsv`
- `summary.md`
- `manifest.tsv`

Day 1 records only that the artifact exists and has the expected file set. Day
2 and Day 3 own row-level inspection, artifact-path semantics, freshness
timestamps, and promotion interpretation.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Current selected manifest invariants are valid. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Windows selected Cholesky workflow contract remains guarded. |
| `python3 tests/test_validate_windows_powershell.py` | Passed | PowerShell validator tests cover structural checks, fake PowerShell success, local unavailable behavior, and hosted fail-closed behavior. |
| `python3 tests/test_normalize_report_index.py` | Passed | Existing normalizer tests pass before Sprint 199 hardening. |
| `python3 tests/test_run_external_comparison.py` | Passed | Existing external comparison generator tests pass before Sprint 199 hardening. |
| `make windows-powershell-validate` | Exit `2` | Structural checks passed, but local `pwsh` is unavailable; this is environment residual evidence, not pass evidence. |

No `.c` or `.h` files changed on Day 1, so the full
`make format && make lint && make test` C gate is not required.

## Claim Boundary

Day 1 does not promote selected Windows report freshness. It establishes that
the guarded workflow path exists, the latest hosted `master` run succeeded,
and the selected artifact can be downloaded. Promotion remains blocked on Day
2/Day 3 hosted artifact semantics and Day 4 manifest decision.

Retained non-claims:

- broad Windows report freshness;
- Windows selected oracle freshness;
- Windows selected benchmark freshness;
- QR incompatible Windows comparison promotion;
- broad report-index freshness;
- unselected comparison families;
- package-manager support;
- shared-library package support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- performance superiority;
- external-library parity;
- state-of-the-art status.

## Day 2 Handoff

Day 2 should inspect the downloaded hosted artifact contents for exact row IDs,
artifact paths, target keys, timestamps, generated commit metadata, platform
metadata, and workflow provenance before any manifest promotion decision is
made.
