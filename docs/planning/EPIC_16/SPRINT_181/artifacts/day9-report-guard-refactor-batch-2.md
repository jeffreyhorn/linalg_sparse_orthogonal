# Sprint 181 Day 9: Report Guard Refactor Batch 2

## Purpose

Day 9 extends selected-target manifest ownership beyond the oracle and
comparison normalizer checks. The batch moves selected benchmark artifact,
required-file, expected-row, freshness-policy, and hosted support-tier checks
behind `tests/corpus/manifests/selected_report_targets.tsv` while preserving
guard-owned methodology checks that are not modeled by the Day 5 schema.

## Code Changes

Updated `scripts/check_bench_canonical_freshness.py` so the selected benchmark
freshness guard reads `SRT-BENCH-REFACTOR-CSC-NOS4` from the selected-target
manifest before validating generated benchmark artifacts.

Manifest-backed benchmark fields now include:

| Manifest field | Guard usage |
| --- | --- |
| `target_id` | Selects exactly one benchmark contract. |
| `family` | Validates generated `report_family`. |
| `target_key` | Validates selected benchmark row identity. |
| `artifact_pattern` | Derives the selected CSV relative path. |
| `required_files` | Drives required artifact presence checks. |
| `expected_rows` | Validated through selected-target manifest tests. |
| `expected_row_ids` | Drives selected benchmark artifact identity. |
| `support_tier` | Drives hosted selected support-tier expectation. |
| `freshness_policy` | Validated as the selected benchmark advisory policy. |
| `workflow_artifact` | Covered by the benchmark manifest regression test. |

The benchmark guard still owns benchmark-specific methodology assertions:

- workload command `tests/data/suitesparse/nos4.mtx --repeat 1`;
- fixture/workload token `nos4.mtx`;
- matrix size `n=100`;
- repeat semantics `configured_repeat_1`;
- warmup, variance, baseline, threshold, and methodology-note tokens;
- selected versus unselected benchmark claim-boundary checks.

Those fields remain local because the Sprint 181 Day 5 manifest schema does
not yet have typed columns for benchmark methodology details.

## Schema Updates

Updated `scripts/validate_corpus_schema.py` so selected target rows now fail
clearly when `artifact_pattern` is `none`.

This closes the missing-artifact-pattern regression from the Day 9 plan and
keeps selected rows from becoming documentation-only declarations without a
concrete artifact contract.

## Test Updates

Updated `tests/test_selected_report_targets_manifest.py` with:

- a missing `artifact_pattern` regression;
- an explicit assertion that the selected-target manifest currently promotes
  only `oracle`, `comparison`, and `benchmark` families.

Updated `tests/test_bench_canonical_freshness.py` with a benchmark manifest
contract regression. The test verifies that the selected benchmark target row
matches the checker contract for family, subfamily, target key, support tier,
freshness policy, generator command, selected relative path, required files,
workflow artifact, and expected row count.

## Non-Promoted Report Families

Day 9 deliberately does not add selected target rows for:

- package;
- CI-only metadata;
- documentation;
- sentinel;
- guardrail;
- dead-code;
- coverage.

Those families remain governed by `report_families.tsv`, generated-local
advisory checks, or documentation/source-controlled metadata. Workflow
metadata on oracle, comparison, and benchmark selected rows does not turn CI or
documentation rows into generated pass evidence.

## Remaining Duplication

Remaining guard-owned fields are justified rather than migrated:

| Area | Reason |
| --- | --- |
| Benchmark methodology fields | The manifest schema has no typed workload, matrix-size, repeat, warmup, variance, baseline, or threshold columns. |
| Workflow YAML structure | Day 10 owns exact job, command, upload-action, and `if-no-files-found` block checks. |
| Oracle solver-family bucket counts | The manifest owns selected oracle total rows and fixture keys, not per-solver-family buckets. |
| Documentation wording | Day 11 owns public/maintainer documentation alignment. |

## Validation

Validation run:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_bench_canonical_freshness.py`
- `python3 -m py_compile scripts/check_bench_canonical_freshness.py scripts/validate_corpus_schema.py tests/test_bench_canonical_freshness.py tests/test_selected_report_targets_manifest.py`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Remaining duplicated selected target lists are reduced or justified. | Complete | Benchmark selected artifact, required files, support tier, and row identity now come from the manifest; methodology and workflow block details remain justified guard-owned fields. |
| Advisory rows remain advisory and do not manufacture pass evidence. | Complete | Manifest tests assert only oracle, comparison, and benchmark are promoted selected families; package, CI, documentation, sentinel, guardrail, dead-code, and coverage remain unselected. |
| Support-tier and freshness-policy checks are manifest-driven. | Complete | Selected target schema validation covers support tier/freshness policy values, and benchmark tests assert the selected benchmark policy and support tier from the manifest. |
