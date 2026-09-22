# Sprint 209 Day 6: Artifact Inspection Tests

## Purpose

Harden selected `qr-incompatible-ls` artifact inspection before any Windows
manifest promotion decision. The goal is to prove that a hosted Windows QR
artifact must include both the exact six selected rows and the exact required
artifact files.

## Implemented Normalizer Check

`scripts/normalize_report_index.py` now checks selected comparison
`required_files` from `tests/corpus/manifests/selected_report_targets.tsv` when
the selected `study.tsv` exists. For `SRT-COMP-QR-INCOMPATIBLE-LS`, a generated
`study.tsv` without any required sidecar now produces:

```text
freshness: error: comparison_required_files: missing_artifact_file
```

The diagnostic includes:

- `target_id=SRT-COMP-QR-INCOMPATIBLE-LS`;
- `target_key=qr-incompatible-ls`;
- the exact missing logical path, such as
  `build/comparison/qr_incompatible_ls/manifest.tsv`;
- the selected artifact diagnostic
  `artifacts=build/comparison/qr_incompatible_ls/study.tsv`;
- the target-specific remediation command with
  `--selected-target qr-incompatible-ls`.

## Regression Coverage

| Test | Coverage |
| --- | --- |
| `test_qr_incompatible_generated_rows_match_windows_artifact_paths()` | Accepts slash, backslash, mixed, and absolute Windows-style paths for the exact selected QR `study.tsv`. |
| `test_qr_incompatible_generated_rows_reject_near_match_artifact_paths()` | Rejects near-match QR path variants and backup files. |
| `test_qr_incompatible_selected_freshness_rejects_windows_path_stale_rows()` | Rejects stale QR rows when the generated artifact path uses Windows separators. |
| `test_qr_incompatible_selected_freshness_rejects_duplicate_windows_path_rows()` | Rejects duplicate QR rows after Windows path normalization. |
| `test_qr_incompatible_selected_freshness_rejects_unexpected_windows_path_rows()` | Rejects unexpected QR row IDs after Windows path normalization. |
| `test_qr_incompatible_selected_freshness_rejects_missing_required_artifact_file()` | Rejects a QR bundle missing `manifest.tsv` even when `study.tsv` is complete. |
| `test_qr_incompatible_selected_freshness_ignores_unrelated_missing_artifacts()` | Confirms `--selected-target qr-incompatible-ls` does not require unrelated Cholesky sidecars. |

## Local Artifact Proof

Command:

```text
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
```

Result:

```text
external-comparison: wrote build/comparison/qr_incompatible_ls/project_observations.tsv
external-comparison: wrote build/comparison/qr_incompatible_ls/baseline_observations.tsv
external-comparison: wrote build/comparison/qr_incompatible_ls/dependency_status.tsv
external-comparison: wrote build/comparison/qr_incompatible_ls/study.tsv
external-comparison: wrote build/comparison/qr_incompatible_ls/summary.md
external-comparison: wrote build/comparison/qr_incompatible_ls/manifest.tsv
external-comparison: qr-incompatible-ls project-vs-baseline comparison passed
```

Freshness command:

```text
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
```

Result:

```text
normalize-report-index: freshness ok (17 rows)
```

## Observed QR Row Set

- `comparison_qr_overdetermined_incompatible_4x2_project_status_v1`
- `comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1`
- `comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1`
- `comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1`
- `comparison_qr_overdetermined_incompatible_4x2_solution_values_v1`
- `comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1`

All six rows report `status=pass`, `support_tier=local_only`, and the retained
non-claim boundary that includes `no Windows report freshness`.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_normalize_report_index.py` | Passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 tests/test_validate_windows_powershell.py` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py` | Passed. |

## Day 6 Decision

Artifact inspection coverage is complete for the branch-local Day 6 scope. The
repository can now fail incomplete selected QR artifact bundles before manifest
promotion. Actual Windows promotion remains blocked until a hosted Windows run
uploads and passes inspection for
`sprint209-windows-selected-comparison-qr-incompatible`.
