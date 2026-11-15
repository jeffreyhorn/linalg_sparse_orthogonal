# Sprint 223 Day 9: Artifact, Row and Path Rejection

## Outcome

The selected Windows Cholesky candidate now rejects extra upload paths, missing or renamed required files, extra files in the selected output directory, mismatched source-row artifact paths, dirty hosted rows and malformed selected row sets. A valid six-file, six-row synthetic Cholesky fixture continues to pass. This is local guard evidence; the final Sprint 223 commit still needs its own hosted Windows job and downloadable six-file artifact audit.

Only `cholesky-spd-tridiag-5` gains the new exact-membership and raw-path checks. The Windows QR lane and unrelated benchmark/oracle checks retain their existing behavior. In particular, a benchmark-only strict normalizer check must not demand a missing Cholesky study; the first test run caught that scope error, and the explicit missing-study diagnostic now applies only when the Cholesky target is selected.

## Implemented contract

| Surface | Rule |
| --- | --- |
| Windows selected upload | Both the workflow regression and PowerShell structural guard parse the selected upload `path: |` block and require the ordered six exact paths. A seventh path, duplicate path, missing path or wrong artifact name is rejected. The upload retains `if-no-files-found: error`. |
| Local selected output directory | For selected Cholesky, the normalizer requires the six manifest-listed files when the study exists, reports a missing `study.tsv` when this target is requested explicitly, and rejects any extra child name in that directory. A rename produces both a missing required name and an unexpected new name. |
| Source-row `artifact_path` | The normalizer validates each Cholesky `study.tsv` row's recorded path **before** replacing it with the physical artifact path in normalized output. Relative forward slashes, Windows backslashes, mixed separators and an absolute Windows path ending in the exact selected `build/comparison/cholesky_spd_tridiag_5/study.tsv` are accepted; near-match target, wrong suffix and wrong-family paths are rejected. |
| Source and row identity | Existing stale-commit, missing/duplicate/unexpected/wrong-target row checks remain active. Hosted Cholesky generation now rejects a dirty checkout, and normalization rejects a `hosted_selected` Cholesky row whose recorded worktree state is not `clean`. Local `local_only` evidence keeps its local semantics. |

## Observed negative diagnostics

All listed fixtures returned nonzero. Temporary directory prefixes vary; the stable diagnostic portions are shown.

| Mutation | Observed reason |
| --- | --- |
| Remove `study.tsv` | `comparison_required_files: missing_artifact_file` with `missing=build/comparison/cholesky_spd_tridiag_5/study.tsv` |
| Remove `manifest.tsv` | `comparison_required_files: missing_artifact_file` naming `manifest.tsv` |
| Rename `summary.md` | `comparison_required_files: missing_artifact_file` naming `summary.md`; new name is also `unexpected_artifact_file` |
| Add local `debug.log` | `comparison_required_files: unexpected_artifact_file: ... unexpected=debug.log` |
| Add or duplicate a workflow upload path | `selected-comparison-freshness upload paths must match the exact selected Cholesky six-file contract` |
| Rename workflow artifact | `selected-comparison-freshness missing selected Cholesky token 'name: sprint190-windows-selected-comparison-cholesky'` |
| Record `cholesky_spd_tridiag_50/study.tsv` in one row | `comparison_artifact_path_mismatch` with row ID, recorded path and expected selected path |
| Mark one hosted row `worktree_state=dirty` | `comparison_dirty_worktree` with row ID and `worktree_state=dirty` |
| Stale row source SHA | `source_commit does not match current HEAD` |
| Remove one row | `comparison_selected_rows: row_set_mismatch` with `expected=6; observed=5` and the missing row ID |
| Duplicate a row | `duplicate normalized row_id` with the duplicated ID |
| Replace one ID with an extra or wrong-target ID | `comparison_selected_rows: row_set_mismatch` with the missing and unexpected IDs |

Positive regressions accept forward-slash, backslash, mixed-separator and absolute Windows spellings of the exact study path. The valid fixture reports `normalize-report-index: freshness ok`. The existing near-match unit tests and the new end-to-end raw-path test reject close but incorrect paths.

## Commands and residuals

Passed: `python3 tests/test_normalize_report_index.py`, `python3 tests/test_selected_comparison_workflow.py`, `python3 tests/test_validate_windows_powershell.py`, `python3 tests/test_run_external_comparison.py`, `python3 tests/test_selected_report_targets_manifest.py`, `python3 scripts/validate_corpus_schema.py`, Python syntax checks and `git diff --check`.

Day 10 still owns cross-file platform/compiler, generated support tier, configuration and claim/non-claim agreement. The local selected normalizer does not by itself authenticate a hosted artifact or distinguish a pull-request merge commit from the intended branch head. Days 11-12 must audit those identities against the hosted run/job/artifact and require the exact final tested commit.
