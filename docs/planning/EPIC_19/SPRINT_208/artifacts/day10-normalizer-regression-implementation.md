# Sprint 208 Day 10 Normalizer Regression Implementation

## Scope

Day 10 implemented the Day 9 normalizer regression plan for the bounded
Windows Cholesky selected freshness path. The work adds executable coverage for
the two Cholesky-specific Windows-path row-set risks that were already covered
for QR incompatible but not yet mirrored for the Sprint 208 target.

## Implementation

Changed file:

- `tests/test_normalize_report_index.py`

Added tests:

- `test_cholesky_selected_freshness_rejects_duplicate_windows_path_rows`
- `test_cholesky_selected_freshness_rejects_unexpected_windows_path_rows`

The duplicate-row test:

- generates only `cholesky_spd_tridiag_5` selected comparison rows;
- appends a duplicate of the first row;
- rewrites all row `artifact_path` values to
  `build\comparison\cholesky_spd_tridiag_5\study.tsv`;
- runs selected freshness for `--selected-target cholesky-spd-tridiag-5`;
- verifies stderr reports `duplicate normalized row_id` and the exact
  duplicated row ID `comparison_cholesky_spd_tridiag_5_project_status_v1`.

The unexpected-row test:

- generates only `cholesky_spd_tridiag_5` selected comparison rows;
- replaces the first row ID with
  `comparison_cholesky_spd_tridiag_5_unexpected_metric_v1`;
- rewrites all row `artifact_path` values to the Windows backslash form;
- runs selected freshness for `--selected-target cholesky-spd-tridiag-5`;
- verifies stdout reports `freshness: error:`, `comparison_selected_rows`,
  `row_set_mismatch`, `target_ids=SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`,
  `observed=6`, the missing expected Cholesky row, the unexpected row, the
  selected Cholesky artifact diagnostic, and the selected-target remediation.

## Validation

Commands run:

```sh
python3 tests/test_normalize_report_index.py
python3 -m py_compile tests/test_normalize_report_index.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_validate_windows_powershell.py
python3 tests/test_selected_comparison_workflow.py
```

Results:

- `test-normalize-report-index: ok`
- Python compilation completed without diagnostics.
- `test-selected-report-targets-manifest: ok`
- `test-validate-windows-powershell: ok`
- `test-selected-comparison-workflow: ok`

## Claim Boundary

No selected manifest metadata was promoted. The tests preserve the Sprint 208
re-deferral boundary:

- no `windows` platform was added to `selected_report_targets.tsv`;
- selected Cholesky support tier remains `local_only`;
- `no Windows report freshness` remains required;
- broad Windows report freshness, package, ABI, performance, release, and
  state-of-the-art claims remain unclaimed.

## Completion Notes

Item 208.4 now has executable Cholesky-specific regression coverage for
Windows-style selected freshness row failures. Day 11 can proceed to public
documentation calibration with the manifest, workflow, PowerShell, and
normalizer guard surfaces aligned to re-deferral.
