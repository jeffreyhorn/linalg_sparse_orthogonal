# Day 10: Normalizer And Freshness Diagnostics

## Purpose

Day 10 validated selected QR incompatible freshness diagnostics for
Windows-style artifact paths and added targeted regression fixtures for the
remaining uncovered row-set failure cases.

## Diagnostic Coverage

| Diagnostic class | Coverage state |
| --- | --- |
| Missing selected artifact | Covered by `test_selected_comparison_target_freshness_accepts_qr_incompatible_subset()` before generated rows are written. |
| Stale selected row with Windows path | Covered by `test_qr_incompatible_selected_freshness_rejects_windows_path_stale_rows()`. |
| Dependency-only or incomplete row set | Covered by `test_qr_incompatible_selected_freshness_rejects_dependency_only_rows()`. |
| Duplicate selected row with Windows path | Added `test_qr_incompatible_selected_freshness_rejects_duplicate_windows_path_rows()`. |
| Unexpected selected row with Windows path | Added `test_qr_incompatible_selected_freshness_rejects_unexpected_windows_path_rows()`. |
| Wrong selected target | Covered by `test_selected_comparison_target_freshness_rejects_wrong_target_rows()`. |

## Tests Added

| Test | Expected diagnostic |
| --- | --- |
| `test_qr_incompatible_selected_freshness_rejects_duplicate_windows_path_rows()` | Fails on `duplicate normalized row_id` for `comparison_qr_overdetermined_incompatible_4x2_project_status_v1`. |
| `test_qr_incompatible_selected_freshness_rejects_unexpected_windows_path_rows()` | Fails selected row-set freshness with the missing expected QR row, the unexpected row id, the selected QR artifact diagnostic, and `--selected-target qr-incompatible-ls` remediation. |

Both tests rewrite the QR incompatible fixture artifact path to
`build\comparison\qr_incompatible_ls\study.tsv` before invoking the normalizer,
so the diagnostic path is Windows-style while the selected target remains
scoped to the QR incompatible artifact.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `python3 -m py_compile tests/test_normalize_report_index.py` | Passed. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Passed; reported freshness ok for 46 rows. |
| `git diff --check -- docs/planning/EPIC_18/SPRINT_203 tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py` | Passed. |

## Promotion Boundary

Day 10 does not promote broad Windows report-index freshness and does not add a
Windows QR workflow lane. The diagnostics are selected-target scoped and keep
the Day 7 re-deferral intact until hosted Windows/MSVC evidence exists.
