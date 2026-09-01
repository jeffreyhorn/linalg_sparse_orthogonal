# Sprint 191 Day 9: Report Index and Freshness Integration

## Summary

Day 9 hardened selected report-index freshness behavior for the new
`qr-incompatible-ls` comparison family.

The production normalizer already uses the selected target manifest as the
authority for row IDs, expected row count, artifacts, and remediation. Day 9
added focused tests so the new family is explicitly covered for target-specific
freshness, Windows-style artifact paths, and incomplete dependency-only proof.

## Freshness Behavior

| Case | Expected result |
| --- | --- |
| No `qr_incompatible_ls` generated artifact exists | Freshness fails with `required generated family missing: comparison`, the exact `build/comparison/qr_incompatible_ls/study.tsv` artifact diagnostic, and `--selected-target qr-incompatible-ls` remediation. |
| All six selected rows exist and are fresh | Target-specific freshness passes. |
| Artifact paths use Windows backslashes | Matching still selects the QR incompatible rows after path normalization. |
| A selected row is stale | Freshness fails with the concrete generated artifact path and target-specific remediation. |
| Only the dependency/status row exists | Freshness fails with `comparison_selected_rows: row_set_mismatch`; dependency evidence alone is not proof. |

## Selected Row Set

The target-specific expected row set is:

| Row ID |
| --- |
| `comparison_qr_overdetermined_incompatible_4x2_project_status_v1` |
| `comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1` |
| `comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1` |
| `comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1` |
| `comparison_qr_overdetermined_incompatible_4x2_solution_values_v1` |
| `comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1` |

## Regression Coverage

| Test | Purpose |
| --- | --- |
| `test_selected_comparison_target_freshness_accepts_qr_incompatible_subset` | Confirms missing-artifact diagnostics, accepted target-specific rows, support tier, artifact path, and non-claims. |
| `test_qr_incompatible_selected_freshness_rejects_windows_path_stale_rows` | Confirms Windows-style artifact paths still match and stale selected rows fail. |
| `test_qr_incompatible_selected_freshness_rejects_dependency_only_rows` | Confirms dependency/status rows cannot satisfy the six-row selected proof contract alone. |

## Scope

This integration keeps the new family selected for Linux/macOS comparison
freshness only. Windows selected comparison metadata remains limited to the
Sprint 190 Cholesky lane until QR incompatible least-squares receives its own
MSVC project-probe proof.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_normalize_report_index.py` | Pass |
| `python3 scripts/validate_corpus_schema.py` | Pass |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| `python3 tests/test_selected_comparison_workflow.py` | Pass |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls` | Pass |
| `python3 tests/test_run_external_comparison.py` | Pass |
| `make report-index-comparison-freshness` | Pass, 46 normalized rows |

No `.c` or `.h` files changed, so the full C quality gate is not required for
Day 9.

## Day 10 Handoff

Day 10 should decide whether to add more runner-level failure injection around
malformed generated outputs or keep the failure surface focused on the
normalizer row-set, stale-row, status, and dependency-only diagnostics now
covered for `qr-incompatible-ls`.
