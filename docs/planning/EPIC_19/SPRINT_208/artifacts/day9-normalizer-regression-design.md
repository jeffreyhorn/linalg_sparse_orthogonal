# Sprint 208 Day 9 Normalizer Regression Design

## Scope

Day 9 reviewed selected comparison freshness normalization for the bounded
Windows Cholesky evidence path. The Sprint 208 decision remains re-deferral:
the hosted Windows artifact is valid input evidence, but the selected target
manifest does not promote Windows freshness. The normalizer tests therefore
need to prove that Windows-style artifact paths are filtered and diagnosed
correctly without broadening the claim.

## Existing Coverage

Current `tests/test_normalize_report_index.py` coverage already includes:

| Area | Existing coverage |
| --- | --- |
| Cholesky Windows separators | `test_selected_comparison_generated_rows_match_windows_artifact_paths` accepts `build\comparison\cholesky_spd_tridiag_5\study.tsv` and absolute Windows-style suffix paths. |
| Cholesky near-match paths | `test_selected_comparison_generated_rows_reject_near_match_artifact_paths` rejects sibling and suffix-only near matches. |
| Cholesky stale Windows path | `test_selected_comparison_target_freshness_rejects_windows_path_stale_rows` verifies stale rows are still caught when artifact paths use backslashes. |
| Cholesky stale/failed selected rows | `test_selected_comparison_target_freshness_rejects_cholesky_stale_or_failed` covers stale and fail status diagnostics for `--selected-target cholesky-spd-tridiag-5`. |
| Cholesky wrong target rows | `test_selected_comparison_target_freshness_rejects_wrong_target_rows` verifies missing Cholesky rows when only a different selected target is generated. |
| QR duplicate Windows-path rows | `test_qr_incompatible_selected_freshness_rejects_duplicate_windows_path_rows` covers duplicate row IDs under Windows separators for QR incompatible. |
| QR unexpected Windows-path rows | `test_qr_incompatible_selected_freshness_rejects_unexpected_windows_path_rows` covers unexpected row IDs under Windows separators for QR incompatible. |

## Day 10 Implementation Targets

Day 10 should add Cholesky-specific parity for the QR duplicate and unexpected
Windows-path row regressions rather than changing the manifest.

| Fixture | Setup | Expected diagnostic | Purpose |
| --- | --- | --- | --- |
| Cholesky duplicate Windows path row | Generate only `cholesky_spd_tridiag_5`, append a duplicate of the first row, and rewrite all row `artifact_path` values to `build\comparison\cholesky_spd_tridiag_5\study.tsv`. | `duplicate normalized row_id` on stderr with `comparison_cholesky_spd_tridiag_5_project_status_v1`. | Proves duplicate row detection survives Windows separator normalization for the exact Sprint 208 target. |
| Cholesky unexpected Windows path row | Generate only `cholesky_spd_tridiag_5`, replace the first `comparison_row_id` with `comparison_cholesky_spd_tridiag_5_unexpected_metric_v1`, and rewrite artifact paths to backslash form. | stdout contains `freshness: error:`, `comparison_selected_rows`, `row_set_mismatch`, `observed=6`, missing `comparison_cholesky_spd_tridiag_5_project_status_v1`, unexpected `comparison_cholesky_spd_tridiag_5_unexpected_metric_v1`, `artifacts=build/comparison/cholesky_spd_tridiag_5/study.tsv`, and `--selected-target cholesky-spd-tridiag-5`. | Proves selected Cholesky row identity drift is diagnosed under Windows path form. |
| Cholesky artifact-root absolute path | Reuse `selected_comparison_generated_rows` with `D:\a\linalg_sparse_orthogonal\linalg_sparse_orthogonal\build\comparison\cholesky_spd_tridiag_5\study.tsv`. | Row is accepted as selected Cholesky generated evidence. | Existing coverage already handles this; keep as baseline, do not duplicate unless Day 10 touches the matcher. |
| Cholesky missing selected rows | Generate only another selected comparison target while requiring `--selected-target cholesky-spd-tridiag-5`. | Existing wrong-target test reports `observed=0`, Cholesky missing row IDs, and selected remediation. | Existing coverage is sufficient; no new Day 10 test needed unless diagnostics change. |
| Cholesky stale selected rows | Generate only Cholesky with stale commit and Windows separators. | Existing stale Windows-path test reports `source_commit does not match current HEAD` and selected remediation. | Existing coverage is sufficient; no new Day 10 test needed unless stale diagnostics change. |
| Cholesky wrong artifact name | Use a near-match path such as `cholesky_spd_tridiag_50` or `study.tsv.bak`. | Existing path rejection returns no matched selected rows. | Existing coverage is sufficient; no new Day 10 test needed unless matcher semantics change. |

## Helper Reuse Plan

Day 10 should reuse existing test helpers:

- `write_selected_comparison_rows()` for synthetic comparison rows;
- `read_tsv()` and `csv.DictWriter` for direct study-file mutation;
- `COMPARISON_STUDY_FIELDS` for stable TSV output;
- `run_command(..., expect_success=False)` for CLI diagnostics;
- `SELECTED_CHOLESKY_ARTIFACT_DIAGNOSTIC`;
- `SCRIPT` and existing `--family comparison --require-generated comparison
  --check-freshness --selected-target cholesky-spd-tridiag-5` command shape.

Avoid adding a second synthetic writer or changing production normalizer code
unless Day 10 exposes a real diagnostic gap.

## Expected Claim Boundary

The Day 10 regressions should not:

- add `windows` to `selected_report_targets.tsv`;
- change selected Cholesky support tier away from `local_only`;
- remove `no Windows report freshness`;
- claim broad Windows report freshness;
- claim Windows oracle, benchmark, package, ABI, performance, release, or
  state-of-the-art support.

## Validation Plan

After Day 10 implementation, run:

```sh
python3 tests/test_normalize_report_index.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_validate_windows_powershell.py
python3 -m py_compile tests/test_normalize_report_index.py
```

No C source or public header validation is required unless Day 10 changes
`*.c` or `*.h` files.

## Completion Notes

Item 208.4 now has an implementation-ready regression plan. The highest-value
Day 10 additions are Cholesky duplicate and unexpected Windows-path row tests,
because those are the only QR-covered Windows selected freshness failure modes
not yet mirrored for the Sprint 208 Cholesky target.
