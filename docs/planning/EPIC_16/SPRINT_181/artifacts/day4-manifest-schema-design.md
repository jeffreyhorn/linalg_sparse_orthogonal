# Sprint 181 Day 4: Manifest Schema Design

## Purpose

Day 4 defines the canonical schema for selected report target metadata. The
schema is designed to make selected oracle, comparison, performance, artifact,
expected-row, support-tier, and workflow upload expectations source-controlled
without replacing generated report evidence or YAML structure guards.

## Canonical Manifest Path Decision

The selected report target manifest should live at:

`tests/corpus/manifests/selected_report_targets.tsv`

Rationale:

- `tests/corpus/manifests/` already owns reviewed TSV metadata that feeds
  corpus and report validation.
- TSV keeps the manifest consistent with `fixtures.tsv`, `generators.tsv`,
  `optional_data.tsv`, and `report_families.tsv`.
- The path keeps selected report target metadata close to the existing
  report-family authority while making it a distinct owner.
- The manifest is source-controlled review evidence, not generated output.

## Manifest Authority Boundary

The selected report target manifest owns selected target metadata:

- stable target identity;
- expected generated row counts;
- generator or freshness command strings;
- required artifact files and generated artifact patterns;
- support tier and freshness policy for selected target evidence;
- hosted platform, workflow job, and workflow artifact names where selected;
- claim scope and non-claims attached to selected target evidence.

The manifest does not own:

- exact YAML job structure, step ordering, upload action syntax, or retention;
- generated report contents;
- local machine benchmark values;
- broad report-index freshness outside selected targets;
- unselected oracle, comparison, performance, package-manager, ABI, or
  platform-parity claims.

Guards should use the manifest for target expectations while continuing to
validate workflow structure directly from YAML.

## Schema Columns

| Column | Required | Meaning |
| --- | --- | --- |
| `target_id` | Yes | Stable unique ID for one selected report target. This is the primary duplicate key. |
| `family` | Yes | Report family, aligned with `report_families.tsv` values such as `oracle`, `comparison`, or `benchmark`. |
| `subfamily` | Yes | Existing subfamily or selected target grouping, such as `solver_backed`, `qr_minnorm`, or `canonical`. |
| `target_key` | Yes | Human-facing target key used by commands and docs, such as `qr-minnorm` or `bench_refactor_csc`. |
| `row_meaning` | Yes | Short purpose statement aligned with report-index row semantics. |
| `selection_scope` | Yes | Selection category: `local_selected`, `hosted_selected`, `reviewed_cross_platform_selected`, or `documentation_selected`. |
| `support_tier` | Yes | Support tier copied from the allowed report-family vocabulary. |
| `freshness_policy` | Yes | Freshness policy copied from the allowed report-family vocabulary. |
| `generator_command` | Yes when generated | Canonical command that creates or validates the selected target. Use `manual review` only for documentation selected rows. |
| `artifact_pattern` | Yes | Primary generated or source-controlled path/glob for the selected target. |
| `required_files` | Yes when generated | Semicolon-delimited file names or relative paths required for the target to be complete. |
| `expected_rows` | Yes when countable | Expected generated row count. Use `none` only where the target has no row-count contract. |
| `expected_row_ids` | Optional | Semicolon-delimited expected generated row IDs when the target contract is row-specific. |
| `workflow_file` | Optional | Workflow file that hosts selected evidence, such as `.github/workflows/ci.yml`. |
| `workflow_job` | Optional | Exact workflow job that owns hosted validation for this selected target. |
| `workflow_artifact` | Optional | Exact upload artifact name for hosted selected evidence. |
| `workflow_platforms` | Optional | Semicolon-delimited reviewed hosted platforms for this target, such as `linux` or `linux;macos`. |
| `claim_scope` | Yes | Positive evidence claim supported by the selected target. |
| `non_claims` | Yes | Semicolon-delimited unsupported interpretations that must remain explicit. |
| `owner` | Yes | Maintainer role or owner file family responsible for target updates. |
| `introduced_in` | Yes | Sprint/day or artifact that introduced or selected the target. |

## Allowed Value Sets

| Field | Allowed values for Day 5 prototype |
| --- | --- |
| `family` | Values already present in `report_families.tsv`; Day 5 should initially use `oracle`, `comparison`, and `benchmark`. |
| `selection_scope` | `local_selected`, `hosted_selected`, `reviewed_cross_platform_selected`, `documentation_selected` |
| `support_tier` | Values already present in `report_families.tsv`: `local_only`, `optional_data`, `reviewed_cross_platform`, `hosted_selected` when applicable. |
| `freshness_policy` | Values already present in `report_families.tsv`: `source_controlled`, `generated_compare_inputs`, `generated_local_advisory`, `hosted_ci_external`, `optional_data_skip`. |
| Empty optional fields | Use `none`, not an empty cell, so missing metadata is explicit. |
| Multi-value cells | Use semicolon-delimited values with no trailing semicolon. |

Day 5 should not introduce new support-tier or freshness-policy values unless
the existing vocabulary cannot describe a selected target. If a new value is
needed, it must be added to the validator's allowed set and documented in the
Day 5 artifact.

## Duplicate Detection Rules

The validator should fail on:

- duplicate `target_id`;
- duplicate tuple of `family`, `subfamily`, and `target_key`;
- duplicate tuple of `family`, `subfamily`, `artifact_pattern`, and
  `generator_command` when `expected_rows` differs;
- duplicate hosted tuple of `workflow_file`, `workflow_job`, and
  `workflow_artifact` when the rows describe different target families without
  an explicit multi-target upload scope;
- a manifest target whose `family` and `subfamily` are not represented in
  `report_families.tsv`, unless the row is explicitly marked
  `documentation_selected`.

Duplicate diagnostics should name both target IDs and the conflicting key so a
maintainer can fix the manifest without inspecting every row.

## Required-Field Validation Rules

The validator should fail when:

- any required column is missing from the header;
- a required cell is empty or contains only whitespace;
- `expected_rows` is neither a positive integer nor `none`;
- `expected_row_ids` is `none` for a selected target that currently has
  row-specific normalizer checks;
- generated selected targets have `generator_command` set to `none`;
- generated selected targets have `required_files` set to `none`;
- hosted selected targets omit `workflow_file`, `workflow_job`, or
  `workflow_artifact`;
- `support_tier` or `freshness_policy` is outside the allowed value set;
- `claim_scope` or `non_claims` is missing.

## Artifact And Row Freshness Diagnostics

Manifest-driven freshness checks should report specific failure categories:

| Failure | Diagnostic should include |
| --- | --- |
| Missing required artifact file | `target_id`, `artifact_pattern`, missing file, and remediation command. |
| Missing generated row | `target_id`, expected row ID or expected row count, generated path, and remediation command. |
| Stale generated row | `target_id`, stale source commit/branch field, generated path, and remediation command. |
| Wrong generated command | `target_id`, observed command, expected `generator_command`, and generated path. |
| Expected-row-count mismatch | `target_id`, expected count, observed count, selected generated file or report family. |
| Unsupported support tier | `target_id`, invalid support tier, allowed values, and manifest path. |
| Unsupported freshness policy | `target_id`, invalid freshness policy, allowed values, and manifest path. |
| Duplicate key | conflicting target IDs, duplicate key type, and manifest path. |

Diagnostics should keep the existing fail-closed behavior for selected oracle
and comparison rows. Advisory rows can still be reported as advisory only when
their `freshness_policy` allows advisory semantics.

## Mapping To Report Families

`report_families.tsv` remains the authority for broad report-family semantics:

- origin class;
- default support tier;
- freshness policy vocabulary;
- high-level claim scope;
- broad non-claims;
- owner and introduction history for report families.

`selected_report_targets.tsv` becomes the authority for selected target
instances:

- exactly which oracle, comparison, and performance targets are selected;
- exact expected counts for those selected targets;
- exact generated artifact files expected for those selected targets;
- exact hosted workflow artifact names for selected hosted evidence.

Mapping rules:

1. Every selected manifest row with an executable report family must map to an
   existing `report_families.tsv` `family` and `subfamily` row.
2. The selected target's `support_tier` must be equal to or narrower than the
   mapped report-family row. A selected row cannot widen a family-level claim.
3. The selected target's `freshness_policy` must match the mapped family row
   unless the target is explicitly hosted metadata and the family row already
   permits `hosted_ci_external`.
4. The selected target's `claim_scope` may be more specific than the
   report-family row, but cannot imply broader target, platform, package,
   ABI, release, or performance claims.
5. The selected target's `non_claims` may add specificity, but cannot remove
   protected family-level non-claims.

## Initial Row Families For Day 5

Day 5 should prototype the manifest with these selected target groups:

| Group | Manifest rows | Notes |
| --- | ---: | --- |
| Selected oracle | 1 or more | At minimum, one selected oracle target covering QR/partial-SVD freshness and `52` expected rows. Day 5 may split by solver family if that makes row IDs cleaner. |
| Selected comparison | 4 | One row each for `qr-minnorm`, `qr-compatible-ls`, `partial-svd-diag6-k2`, and `lu-nonsym-square-5`. |
| Selected performance | 1 | One row for `bench_refactor_csc` on `nos4.mtx --repeat 1`. |
| Hosted workflow uploads | represented in rows | Linux oracle/comparison/performance and macOS comparison artifact names should be represented by workflow fields, not separate proof rows unless Day 5 needs multiple platform rows. |

## Day 5 Handoff

Day 5 should add the TSV manifest and a small validator or schema test that
checks the required header, duplicate-key rules, allowed value sets, and basic
mapping to `report_families.tsv`. The prototype should prefer the smallest row
set that removes the current hand-owned selected target lists without widening
claim scope.

## Validation

Day 4 is documentation-only. Validation:

- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Schema fields cover project-plan item 181.2. | Complete | Required columns include duplicate detection keys, required files, expected rows, freshness policy, and support-tier fields. |
| Manifest authority and report-family metadata roles do not conflict. | Complete | Mapping section keeps `report_families.tsv` as family authority and `selected_report_targets.tsv` as selected target authority. |
| Validation failures are specific enough for maintainers to fix rows quickly. | Complete | Diagnostic table names failure categories and required fields for each message. |
