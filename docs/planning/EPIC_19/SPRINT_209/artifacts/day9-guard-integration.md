# Sprint 209 Day 9: Guard Integration

## Purpose

Integrate the Day 8 QR incompatible re-deferral decision into the Windows
workflow and manifest guard surface.

## Implemented Guard Coverage

| Guard surface | Coverage added |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Freezes the current `SRT-COMP-QR-INCOMPATIBLE-LS` manifest identity, generator, artifact pattern, required-file set, expected rows, row IDs, workflow metadata, claim scope, support tier, and non-claims. |
| `scripts/validate_windows_powershell.py` | Rejects `.github/workflows/windows-ci.yml`, `selected-qr-incompatible-comparison-freshness`, `sprint209-windows-selected-comparison-qr-incompatible`, or `windows` in the QR manifest row while re-deferred. |
| `scripts/validate_windows_powershell.py` | Rejects non-QR selected rows that reference the Sprint 209 QR Windows job or artifact. |
| `tests/test_validate_windows_powershell.py` | Adds regressions for QR workflow-file, job, artifact, platform, target-key, required-file, non-claim, and non-QR metadata leakage drift. |

## Decision Boundary

The Windows workflow may contain the Sprint 209 QR lane only as a bounded
evidence-collection path. The selected report target manifest remains
Linux/macOS-only and `local_only` until a hosted Windows run exists, the exact
six-file QR artifact has been inspected, generated freshness passes, and all
promotion surfaces move together.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `make windows-powershell-guard` | Passed. |

## Closeout

Day 9 completes the guard-integration slice for Item 209.5. Public docs and
maintainer claim-boundary wording remain scheduled for Day 10 and Day 11.
