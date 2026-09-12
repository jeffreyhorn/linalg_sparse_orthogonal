# Day 8: Manifest And Workflow Metadata

## Purpose

Day 8 updated the regression coverage around the selected report target
manifest so the Day 7 Windows QR incompatible re-deferral is enforced by the
standalone manifest test runner.

## Decision

Keep `SRT-COMP-QR-INCOMPATIBLE-LS` Linux/macOS-only and `local_only` until
hosted Windows/MSVC evidence supports promotion. No selected target manifest
row, workflow YAML, public docs, or maintainer docs were promoted on Day 8.

## Test Added

| Test | Enforced contract |
| --- | --- |
| `test_qr_incompatible_manifest_remains_redeferred_for_windows()` | QR incompatible keeps six expected rows, six required files, exact expected row ids, no Windows platform, no Windows workflow file, no reused Cholesky artifact, `support_tier=local_only`, and required non-claims. |

The new test is called from `main()` in
`tests/test_selected_report_targets_manifest.py`.

## Source-Of-Truth State

| Surface | Day 8 state |
| --- | --- |
| `tests/corpus/manifests/selected_report_targets.tsv` | Unchanged; QR incompatible remains Linux/macOS-only. |
| `.github/workflows/windows-ci.yml` | Unchanged; selected comparison workflow remains Cholesky-specific. |
| `tests/test_selected_report_targets_manifest.py` | Updated with QR incompatible re-deferral contract coverage. |

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `python3 -m py_compile tests/test_selected_report_targets_manifest.py` | Passed. |

## Promotion Boundary

This day does not claim Windows QR incompatible freshness. It prevents a future
partial metadata edit from silently adding Windows, reusing the Cholesky
artifact, changing the selected row set, dropping required files, or removing
required non-claims before hosted MSVC evidence exists.
