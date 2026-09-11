# Sprint 202 Day 7: Freshness Regression Fixtures

## Summary

Day 7 expanded selected benchmark freshness regression coverage for the macOS
hosted selected lane and the existing selected `bench_refactor_csc` row.

The new tests are artifact-level fixture mutations against copied generated
reports, keeping the selected scope limited to
`SRT-BENCH-REFACTOR-CSC-NOS4`.

## Added Fixture Coverage

| Case | Test |
| --- | --- |
| Missing selected artifact | `test_missing_selected_benchmark_artifact_fails` |
| Missing selected index row | `test_missing_selected_index_row_fails` |
| Duplicate selected index row | `test_duplicate_selected_index_row_fails` |
| Malformed timestamp metadata | `test_malformed_selected_timestamp_fails` |
| Hosted `runner_context=local` placeholder | `test_hosted_runner_context_cannot_be_local_placeholder` |
| Hosted `report_label=unlabeled` placeholder | `test_hosted_report_label_cannot_be_unlabeled_placeholder` |
| Selected full relative-path drift | `test_selected_relative_path_drift_fails` |
| Selected dot-prefix relative-path drift | `test_selected_relative_path_dot_prefix_drift_fails` |

## Coverage Map

| Required Sprint 202 diagnostic | Coverage |
| --- | --- |
| Passing selected-platform artifact | Covered by the macOS hosted metadata positive fixture. |
| Missing artifact | Covered by removing `bench_refactor_csc.csv`. |
| Stale artifact or metadata | Covered by selected value, timestamp, CSV, and manifest mismatch tests. |
| Duplicate artifact row | Covered by duplicating the selected `index.tsv` row. |
| Malformed metadata | Covered by malformed timestamp and existing schema tests. |
| Deferred artifact | Covered by unselected row local-only tests. |
| Path-normalized artifact boundary | Covered by rejecting full and dot-prefixed selected relative paths. |

## Validation

Passed:

```sh
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_performance_docs.py
```

No `.c` or `.h` files changed, so the full C quality gate was not required.
