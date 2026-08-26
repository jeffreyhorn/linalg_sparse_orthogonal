# Sprint 181 Day 6: Parser And Schema Checks

## Purpose

Day 6 adds parser and schema validation for
`tests/corpus/manifests/selected_report_targets.tsv`. The goal is to make the
Day 5 prototype a guarded source-controlled authority before later sprint days
move normalizer and workflow checks behind it.

## Parser Surface

The existing structured TSV loader in `scripts/validate_corpus_schema.py` now
loads the selected report target manifest through:

- `read_tsv(path)`;
- `SELECTED_REPORT_TARGET_REQUIRED`;
- `validate_selected_report_targets(path, rows, report_family_rows)`;
- `validate(root)`.

This keeps selected-target parsing consistent with the existing corpus
manifest style and gives downstream guard refactors a stable list of row
dictionaries keyed by the Day 4 schema fields.

## Schema Checks Added

`validate_selected_report_targets` now checks:

| Check | Diagnostic owner |
| --- | --- |
| Required header fields | `require_fields` names missing headers and empty required cells. |
| Duplicate `target_id` | Error names the duplicate ID and first-seen line. |
| Duplicate `family`/`subfamily`/`target_key` | Error names the duplicate tuple and first-seen line. |
| `family`/`subfamily` mapping | Error names rows not represented in `report_families.tsv`. |
| `selection_scope` enum | Error names invalid value and allowed values. |
| `support_tier` enum | Error names invalid value and allowed values. |
| `freshness_policy` enum | Error names invalid value and allowed values. |
| `expected_rows` shape | Error requires a positive integer or `none`. |
| Countable rows without `expected_row_ids` | Error identifies the row as missing row identity metadata. |
| Generated rows without commands/files | Error requires `generator_command` and `required_files`. |
| Hosted rows without workflow metadata | Error requires workflow file, job, artifact, and platform fields. |
| Artifact/command/count collisions | Error names the duplicate artifact/generator key and first-seen line. |
| Cross-family hosted artifact collisions | Error names the workflow artifact tuple and conflicting families. |

## Allowed Values

The selected-target parser keeps its support-tier vocabulary separate from the
broader corpus/report-family enum:

- broad report-family rows still use existing `SUPPORT_TIERS`;
- selected report target rows use `SELECTED_REPORT_SUPPORT_TIERS`;
- `hosted_selected` is accepted only for selected report target rows.

This preserves the Day 4 boundary: selected target metadata may describe
hosted selected evidence without widening `report_families.tsv` family-level
claims.

## Test Coverage

Added `tests/test_selected_report_targets_manifest.py` with focused cases for:

- current manifest success;
- duplicate target IDs;
- duplicate target-key tuples;
- unsupported support tiers;
- malformed `expected_rows`;
- missing hosted workflow metadata;
- missing `report_families.tsv` mapping;
- artifact/generator collisions with count drift.

The tests mutate parsed manifest rows in memory, so they exercise the parser
diagnostics without writing temporary manifest files or changing source data.

## Documentation Hook

`tests/corpus/README.md` now lists
`manifests/selected_report_targets.tsv` in the corpus layout, ownership table,
and row-interpretation guidance. The README describes the selected-target
manifest as target-specific authority that narrows existing report-family
semantics without promoting unselected proof surfaces.

## Day 7 Handoff

Day 7 can now design the report normalizer refactor against a concrete parser
surface:

- selected oracle expected rows and fixture keys can come from
  `SRT-ORACLE-QR-PSVD-LOCAL`;
- selected comparison target keys, expected row IDs, expected counts, and
  required files can come from `SRT-COMP-*` rows;
- selected benchmark artifact identity and required files can come from
  `SRT-BENCH-REFACTOR-CSC-NOS4`;
- workflow guard refactors can use workflow metadata fields while keeping YAML
  structure checks in their existing tests.

Day 7 should avoid changing freshness behavior until the design maps current
normalizer constants to manifest fields one by one.

## Validation

Day 6 changed Python validation code, a Python test, TSV metadata, and docs.
Validation run:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 -m py_compile scripts/validate_corpus_schema.py tests/test_selected_report_targets_manifest.py`
- `make format && make lint && make test`
- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Malformed manifest rows fail clearly. | Complete | Unsupported support tier, bad expected rows, missing hosted metadata, and missing family mapping tests assert diagnostic substrings. |
| Duplicate selected targets fail clearly. | Complete | Duplicate target ID, duplicate target-key tuple, and artifact/count collision tests cover duplicate diagnostics. |
| Parser output can drive report and workflow guards. | Complete | Validator exposes structured row dictionaries with target identity, expected rows, row IDs, required files, workflow metadata, claims, and non-claims. |
