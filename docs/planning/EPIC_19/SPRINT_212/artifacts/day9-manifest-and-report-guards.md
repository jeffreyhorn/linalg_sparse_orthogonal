# Sprint 212 Day 9: Manifest And Report Guards

## Summary

Day 9 binds the Sprint 212 threshold-free benchmark methodology policy to the
selected target manifest and report metadata. The new selected benchmark
manifest helper requires the exact `SRT-BENCH-REFACTOR-CSC-NOS4` identity,
workflow metadata, claim scope, artifact contract, and non-claim tuple.

## Changed Files

| Path | Change |
| --- | --- |
| `tests/test_selected_report_targets_manifest.py` | Adds exact selected benchmark manifest contract constants, an assertion helper, and drift regressions. |
| `docs/planning/EPIC_19/SPRINT_212/WORKING_NOTES.md` | Records Day 9 manifest/report guard implementation and validation evidence. |

## Guard Contract

The selected benchmark manifest row must remain:

- `target_id=SRT-BENCH-REFACTOR-CSC-NOS4`
- `family=benchmark`
- `subfamily=canonical`
- `target_key=bench_refactor_csc`
- `selection_scope=hosted_selected`
- `support_tier=hosted_selected`
- `freshness_policy=generated_local_advisory`
- `generator_command=make bench-canonical-report-freshness`
- `artifact_pattern=build/bench-reports/canonical/bench_refactor_csc.csv`
- `required_files=bench_refactor_csc.csv;index.tsv;manifest.txt`
- `expected_rows=1`
- `expected_row_ids=bench_refactor_csc`
- Linux/macOS workflow file, job, artifact, and platform tuples
- threshold-free Linux/macOS selected freshness claim scope
- the exact selected benchmark non-claim tuple

## Regression Fixtures

| Test | Purpose |
| --- | --- |
| `test_selected_benchmark_manifest_records_distribution_non_claims` | Reuses the exact contract helper so the current row must match all selected benchmark fields, not only contain non-claim substrings. |
| `test_selected_benchmark_manifest_rejects_identity_drift` | Rejects family, subfamily, target key, row meaning, selection scope, support tier, freshness policy, generator command, and artifact-pattern drift. |
| `test_selected_benchmark_manifest_rejects_workflow_metadata_drift` | Rejects workflow file, job, artifact, and platform tuple drift, including Windows expansion. |
| `test_selected_benchmark_manifest_rejects_threshold_claim_scope` | Rejects hosted timing-threshold wording in the selected benchmark claim scope. |
| `test_selected_benchmark_manifest_rejects_missing_non_claim` | Rejects loss of a required selected benchmark non-claim. |
| `test_selected_benchmark_manifest_rejects_extra_threshold_non_claim` | Rejects extra non-claim tuple drift so the row stays exact rather than substring-valid. |

## Schema Confirmation

`tests/corpus/schemas/report_index_fields.md` already describes the selected
benchmark target as threshold-free and names the authoritative selected target,
`baseline=n/a`, `threshold=n/a`, Linux/macOS-only hosted metadata, and no
portable performance, Windows selected benchmark freshness, broad
package-manager distribution, or timing-comparability claim. No schema text
change was required for Day 9.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `python3 tests/test_bench_canonical_freshness.py` | Passed. |
| `python3 scripts/validate_corpus_schema.py` | Passed. |
| `git diff --check` | Passed. |
| `git status --short` | Passed; Day 7-9 script/test edits are tracked and the Sprint 212 planning directory remains untracked until commit time. |
| `git diff --name-only -- '*.c' '*.h'` | Passed; no C/header files changed. |

## Day 9 Outcome

Item 212.4 now covers the selected benchmark freshness and manifest/report
guard layers. The selected benchmark evidence cannot silently broaden to a
hosted timing threshold, Windows selected benchmark freshness, a different
artifact, or a looser non-claim set through selected-target manifest drift.
