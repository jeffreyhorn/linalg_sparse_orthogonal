# Sprint 181 Day 12: Diagnostics And Drift Tests

## Purpose

Day 12 hardens selected-target manifest diagnostics and workflow drift tests.
The focus is failure clarity: malformed manifest rows should name the row that
must be fixed, and workflow drift failures should name the missing job or
upload block instead of failing as broad substring mismatches.

## Schema Diagnostics

Updated `scripts/validate_corpus_schema.py` with selected-target-specific enum
diagnostics. Invalid `selection_scope`, `support_tier`, and
`freshness_policy` values now report the manifest line and `target_id`.

Selected target validation now rejects unsupported artifact patterns:

- `artifact_pattern=none`;
- absolute artifact paths;
- parent-traversing artifact paths such as `../build/...`.

Selected generated-row diagnostics now include `target_id` for:

- invalid `expected_rows`;
- missing `expected_row_ids` on countable rows;
- missing `generator_command`;
- missing `required_files`;
- missing hosted workflow metadata.

## Manifest Regression Tests

Updated `tests/test_selected_report_targets_manifest.py` with row-specific
diagnostic checks for:

- unsupported support tiers;
- unsupported freshness policies;
- missing artifact patterns;
- parent-traversing artifact patterns;
- invalid expected row counts;
- missing expected row IDs;
- missing generated required files;
- missing hosted workflow metadata.

Existing duplicate target ID, duplicate target key, report-family mapping,
artifact/count collision, and unpromoted-family regressions remain in place.

## Workflow Drift Tests

Updated `tests/test_selected_comparison_workflow.py` with in-memory workflow
mutation tests for:

- missing selected report job IDs;
- wrong upload artifact names;
- missing `if-no-files-found: error` in the exact selected upload block;
- broad comparison upload globs.

These tests exercise the exact job and upload-block helpers introduced on Day
10 without changing workflow YAML fixtures on disk.

## Existing Freshness Diagnostics

Day 12 relies on existing `tests/test_normalize_report_index.py` coverage for
selected generated-row freshness failures:

- missing selected oracle/comparison generated rows;
- stale source commits;
- selected expected row mismatches;
- duplicate selected comparison rows;
- unexpected selected comparison rows;
- failed, skipped, and deferred selected comparison rows.

Those tests already distinguish stale generated data from missing generated
data in normalizer diagnostics.

## Validation

Validation run:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 -m py_compile scripts/validate_corpus_schema.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Failure cases name the manifest row or workflow block that must be fixed. | Complete | Selected manifest diagnostics include `target_id`; workflow drift tests assert missing job/upload-block messages. |
| Diagnostics distinguish stale generated data from missing generated data. | Complete | Existing normalizer tests cover stale source commits separately from missing selected generated rows. |
| Tests cover the highest-risk drift paths introduced by manifest ownership. | Complete | Manifest enum/artifact/required-field drift and workflow job/upload drift are covered. |
