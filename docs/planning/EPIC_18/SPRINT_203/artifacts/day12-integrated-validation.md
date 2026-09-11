# Day 12: Integrated Validation

## Purpose

Day 12 ran the focused Sprint 203 validation matrix across the selected QR
incompatible generator, selected freshness diagnostics, manifest/workflow/docs
guards, Windows claim-boundary checks, and QR runtime proof owners.

## Validation Commands

| Command | Result | Notes |
| --- | --- | --- |
| `python3 scripts/run_external_comparison.py --target qr-incompatible-ls` | Passed | Regenerated `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`; reported project-vs-baseline comparison passed. |
| `python3 tests/test_run_external_comparison.py` | Passed | External comparison regression suite passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed | Reported freshness ok for 46 rows and the six QR incompatible generated rows as fresh. |
| `python3 tests/test_normalize_report_index.py` | Passed | Normalizer, selected-target, Windows-path, and row-set diagnostic regressions passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Manifest re-deferral contract remains intact. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Linux/macOS selected comparison and Windows Cholesky/QR re-deferral workflow guards passed. |
| `python3 tests/test_selected_performance_docs.py` | Passed | Existing selected-performance documentation markers remain intact. |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Windows structural, claim-boundary, hosted wiring, manifest, deferral, and local `pwsh` residual regressions passed. |
| `python3 scripts/validate_windows_powershell.py` | Structural pass, exit `2` | Claim boundaries passed; local `pwsh` is unavailable, which remains environment residual evidence rather than pass evidence. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py tests/test_run_external_comparison.py` | Passed | Python syntax compilation passed. |
| `build/test_qr_corpus` | Passed | QR corpus proof owner passed 14 tests with zero failures. |
| `build/test_qr_solve` | Passed | QR solve proof owner passed 19 tests, including the incompatible 4x2 dense-reference comparison. |

## Corrected Command

`make test_qr_corpus` was attempted as a focused QR command but is not a
Makefile target. The maintained built binary exists as `build/test_qr_corpus`,
which was run directly and passed.

## Full C Gate Decision

No `.c` or `.h` files were modified in Sprint 203 through Day 12. The full
`make format && make lint && make test` gate was therefore not required by the
sprint instructions for this day.

## Residuals

| Residual | Status |
| --- | --- |
| Hosted Windows/MSVC QR incompatible proof | Still absent; blocks Windows QR promotion. |
| Hosted QR incompatible Windows artifact inspection | Still absent; blocks selected-target manifest promotion. |
| Local PowerShell availability | `pwsh` unavailable locally; structural checks pass and the condition remains an environment residual. |
| Broad Windows report-index freshness | Not promoted. |

## Promotion Boundary

Day 12 validates the current re-deferred state. It does not add Windows QR
workflow metadata, selected target manifest Windows platforms, generated
support-tier promotion, or broader report freshness claims.
