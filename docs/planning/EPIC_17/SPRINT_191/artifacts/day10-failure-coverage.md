# Sprint 191 Day 10: Focused Failure Coverage

## Summary

Day 10 added focused runner-level failure coverage for the selected
`qr-incompatible-ls` comparison family.

The new tests complement Day 9 report-index freshness coverage by proving that
bad project output, bad baseline/project agreement, and tolerance misses are
reported before documentation claims are expanded.

## Added Tests

| Test | Coverage |
| --- | --- |
| `test_qr_incompatible_ls_tolerance_boundaries_pass_and_fail` | Near-tolerance project and baseline observations pass; beyond-tolerance residual, solution norm, and solution values fail. |
| `test_qr_incompatible_ls_study_rows_reject_tolerance_miss` | Project-vs-baseline residual, solution norm, solution values, and max-absolute-delta mismatches fail selected study validation. |
| `test_qr_incompatible_ls_project_parser_rejects_missing_fields` | Missing project probe fields raise structured `project_probe_failed` diagnostics. |

## Existing Coverage Retained

| Existing test | Retained behavior |
| --- | --- |
| `test_qr_incompatible_ls_reference_parser_rejects_malformed_output` | Malformed source-controlled dense QR helper output fails as `baseline_malformed_output`. |
| `test_qr_incompatible_ls_reference_reports_command_failure` | Helper execution failures remain structured `baseline_command_failed` errors. |
| `test_qr_incompatible_ls_dependency_reports_missing_helper` | Missing required helper rows report `baseline_helper_missing` and execution raises `missing_baseline_helper`. |
| `test_qr_incompatible_selected_freshness_rejects_windows_path_stale_rows` | Windows-style artifact paths still select stale QR incompatible rows and fail freshness. |
| `test_qr_incompatible_selected_freshness_rejects_dependency_only_rows` | Dependency/status evidence alone cannot satisfy selected comparison freshness. |

## Failure Semantics

| Error or status | Interpretation |
| --- | --- |
| `metric_tolerance_miss` | Required selected study rows exist but one or more metrics fail tolerance. This is a hard failure. |
| `project_probe_failed` | Project output is missing required fields or has invalid value shape. This is a structured hard failure. |
| `baseline_malformed_output` | Source-controlled helper output is malformed. This is a structured hard failure. |
| `missing_baseline_helper` | Required dense QR helper is absent. This is a structured hard failure. |
| `skip` or `defer` selected rows | Not accepted as proof by selected freshness diagnostics. |
| optional NumPy/SciPy dependency rows | Deferred advisory rows only; not part of selected proof. |

## Validation

| Command | Result |
| --- | --- |
| `python3 -m py_compile tests/test_run_external_comparison.py tests/test_normalize_report_index.py scripts/run_external_comparison.py scripts/normalize_report_index.py` | Pass |
| `python3 tests/test_run_external_comparison.py` | Pass |
| `python3 tests/test_normalize_report_index.py` | Pass |
| `python3 scripts/validate_corpus_schema.py` | Pass |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| `python3 tests/test_selected_comparison_workflow.py` | Pass |
| `make report-index-comparison-freshness` | Pass, 46 normalized rows |

No `.c` or `.h` files changed, so the full C quality gate is not required for
Day 10.

## Day 11 Handoff

Day 11 should update public, maintainer, and corpus documentation from five to
six selected comparison families where appropriate. Any QR incompatible
least-squares wording should keep the claim fixture-local and repeat the broad
QR, broad least-squares, external-library parity, Windows freshness,
package-manager, ABI, performance, and state-of-the-art non-claims.
