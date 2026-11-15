# Sprint 224 Day 9: QR artifact, row and path regressions

## Scope and outcome

The selected Windows QR incompatible least-squares candidate now has hosted-bundle mutations for the exact six-file artifact, six ordered study rows, source state and recorded study path. The valid synthetic hosted QR bundle still passes selected-target freshness. These are local regression fixtures; they do not replace the final changed-head hosted MSVC artifact required by Days 11–14.

The normalizer now applies its strict selected six-file directory check to QR as well as Cholesky, so an unexpected sidecar cannot silently coexist with a valid study. It also rejects a QR study whose six correct row IDs are out of contract order. The selected workflow test requires the upload path entries to equal the six QR paths in order, rejecting a seventh or duplicate entry even if all required path substrings remain present.

## Observed negative-case diagnostics

| Mutation | Observed rejection |
| --- | --- |
| Remove `study.tsv`, project observations, baseline observations or dependency status | `comparison_required_files: missing_artifact_file` |
| Remove `manifest.tsv` or `summary.md`; rename `summary.md` | `comparison_hosted_metadata_mismatch: field=manifest_missing` or `field=summary_missing` |
| Add `debug.log` as a seventh QR sidecar | `comparison_required_files: unexpected_artifact_file` |
| Remove, duplicate, add or substitute a wrong-target study row | `comparison_selected_rows: row_set_mismatch`, or `duplicate normalized row_id` for a duplicate |
| Swap two otherwise valid QR rows | `comparison_hosted_row_order_mismatch` |
| Change study, manifest and summary to the same stale but well-formed source SHA | `source_commit does not match current HEAD` |
| Mark a hosted QR row's worktree dirty | `comparison_dirty_worktree` |
| Record an extra-directory, backup-suffix or Cholesky study path | `comparison_artifact_path_mismatch` |
| Add or duplicate a QR workflow upload path | `upload paths must match exact six-file contract` |
| Rename a required QR workflow upload path | `upload missing` for the required path |
| Replace a QR upload path with an absolute Windows path, backslashes, mixed separators or a near-match name | Upload missing-path or exact six-file contract diagnostic |

The normalizer accepts the expected relative QR study path with forward slashes, backslashes or mixed separators, and an absolute Windows path ending in the same `build/comparison/qr_incompatible_ls/study.tsv` suffix. The workflow upload contract remains the exact six relative paths, so those alternate spellings are rejected there. The Cholesky selected suite remains independent and passes.

## Checks and boundary

`python3 tests/test_normalize_report_index.py`, `python3 tests/test_selected_comparison_workflow.py`, `python3 tests/test_validate_windows_powershell.py`, `python3 tests/test_selected_report_targets_manifest.py` and `python3 tests/test_run_external_comparison.py` passed after the mutations. The first workflow path assertion was adjusted to the actual exact-path diagnostic, then the suite passed. No C or header file changed on Day 9.

Day 10 owns the remaining provenance, claim, workflow job and Cholesky-substitution mutations from the Day 8 matrix. QR Windows freshness is still conditional on a new hosted artifact from the final changed source head and a full six-file audit.
