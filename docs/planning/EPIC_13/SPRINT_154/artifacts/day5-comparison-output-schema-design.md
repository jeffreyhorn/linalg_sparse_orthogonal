# Sprint 154 Day 5 Comparison Output Schema Design

## Purpose

Day 5 designs the output schema and report-index meaning for the first narrow
comparison study. The selected target remains
`qr_underdetermined_minnorm_2x4` with the source-controlled
external-process dense reference selected on Day 4.

## Existing Report Contract Inventory

The current report-family contract lives in
`tests/corpus/manifests/report_families.tsv` and is documented by
`tests/corpus/schemas/report_index_fields.md`.

Relevant existing behavior:

- source-controlled report-family rows are advisory or policy rows, not pass
  evidence;
- generated-local oracle rows use `freshness_policy=generated_compare_inputs`;
- missing generated rows must not silently disappear or manufacture pass
  evidence;
- optional-data skip/defer rows must not count as pass evidence;
- local generated outputs under ignored `build/` paths are not hosted CI,
  package, ABI, platform, performance, release, external-library parity, or
  state-of-the-art proof.

There is no dedicated source-controlled `comparison` report family yet. Day 5
therefore designs a comparison schema that can be emitted as a local study
artifact first, then normalized later only if Days 10-11 add a reviewed report
family row and freshness policy.

## Integration Decision

Day 5 selects an artifact-first schema for the first study.

Rationale:

- the selected study covers one fixture and one source-controlled dense
  reference;
- adding a normalized `comparison` report family before seeing generated rows
  would risk overfitting the report contract;
- artifact-first publication can still capture all provenance and status
  fields needed for later normalization;
- Days 10-11 can promote the schema to report-index rows only after the
  harness has produced real output.

The first study should write generated local artifacts under:

- `build/comparison/qr_minnorm/`

Recommended artifact names:

- `study.tsv`: metric-level rows;
- `summary.md`: human-readable narrow study summary;
- `manifest.tsv`: run-level provenance and row-count summary.

If Days 10-11 add normalized rows, the likely future family is:

- `report_family=comparison`;
- `subfamily=qr_minnorm`;
- `row_meaning=narrow_external_process_dense_reference`;
- `row_origin=generated_local`;
- `support_tier=local_only`;
- `freshness_policy=generated_compare_inputs`.

## Row Schema Proposal

`study.tsv` should use tab-separated rows with these fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `comparison_row_id` | yes | Stable row id, e.g. `comparison_qr_underdetermined_minnorm_2x4_solution_values_v1`. |
| `report_family` | yes | `comparison` for generated comparison artifacts, even if not normalized yet. |
| `subfamily` | yes | `qr_minnorm`. |
| `row_kind` | yes | `metric_comparison`, `dependency_status`, or `run_summary`. |
| `fixture_key` | yes | `qr_underdetermined_minnorm_2x4`. |
| `operation` | yes | `minnorm_solve`. |
| `metric` | yes | Stable metric name. |
| `baseline_name` | yes | `qr_external_dense_reference`. |
| `baseline_type` | yes | `external_process_dense_reference`. |
| `baseline_version` | yes | Source commit plus helper path or explicit helper version if added later. |
| `baseline_command` | yes | Exact helper command. |
| `baseline_python_executable` | yes | Resolved Python executable path. |
| `baseline_python_version` | yes | Full Python version string. |
| `project_name` | yes | `sparse_lu_ortho`. |
| `project_version` | yes | Version from `VERSION`. |
| `project_command` | yes | Exact project-side command selected by harness design. |
| `source_commit` | yes | Current Git commit used for the run. |
| `source_branch` | yes | Current branch if available. |
| `worktree_state` | yes | `clean` or `dirty`. |
| `platform` | yes | Platform string. |
| `compiler` | yes | Compiler identity or `unknown` with caveat. |
| `configuration` | yes | Build/configuration description. |
| `expected_value` | yes | Expected value or status for the metric. |
| `project_value` | yes | Observed project value, or empty for dependency-only rows. |
| `baseline_value` | yes | Observed baseline value, or empty for project-only status rows. |
| `delta_value` | yes | Difference metric where applicable. |
| `tolerance_kind` | yes | `absolute`, `status_only`, or another controlled value. |
| `tolerance_value` | yes | Numeric tolerance or empty for status-only. |
| `status` | yes | `pass`, `fail`, `skip`, `defer`, or `error`. |
| `status_reason` | yes | Stable short reason. |
| `caveat` | yes | Human-readable caveat preserving non-claims. |
| `artifact_path` | yes | Path to the generated artifact containing this row. |
| `generated_at_utc` | yes | UTC timestamp for the local run. |
| `support_tier` | yes | `local_only`. |
| `claim_scope` | yes | Fixture-local claim supported by a passing row. |
| `non_claims` | yes | Semicolon-separated non-claims. |

## Selected Metrics

Required metric rows for the first study:

| Metric | Row Kind | Expected | Tolerance |
| --- | --- | --- | --- |
| `project_status` | `metric_comparison` | `SPARSE_SUCCESS` | `status_only` |
| `baseline_status` | `dependency_status` | `success` | `status_only` |
| `residual_norm` | `metric_comparison` | `<=1e-10` | absolute `1e-10` |
| `solution_norm` | `metric_comparison` | `1.0` | absolute `1e-10` |
| `solution_values` | `metric_comparison` | `0.5,0.5,0.5,0.5` | absolute `1e-10` per component |
| `project_vs_baseline_max_abs_delta` | `metric_comparison` | `<=1e-10` | absolute `1e-10` |

The schema deliberately excludes:

- raw QR basis values;
- Q/R entries;
- pivot order;
- rank-threshold policy rows;
- timing, memory, throughput, or performance rows;
- package, ABI, loader, platform, or hosted CI rows.

## Status Semantics

| Status | Meaning | Counts As Proof |
| --- | --- | --- |
| `pass` | Required selected commands ran and selected metric values met tolerance. | Yes, fixture-local only. |
| `fail` | Required commands ran, but selected metric values or expected statuses missed tolerance. | No. |
| `skip` | Optional future dependency was unavailable under an explicit optional policy. | No. |
| `defer` | Optional future baseline or target was intentionally not selected. | No. |
| `error` | Required dependency, command, parser, provenance, or output contract failed. | No. |

Only `pass` may support the narrow fixture-local comparison claim. `skip`,
`defer`, and `error` rows must remain visible if emitted and must not count as
proof.

## Freshness And Stale-Output Policy

Because Day 5 chooses artifact-first output, freshness should be enforced by
the harness itself before report-index promotion:

- generated rows must record `source_commit`;
- generated rows must record `source_branch`;
- generated rows must record `worktree_state`;
- generated rows must record `generated_at_utc`;
- generated rows must record exact baseline and project commands;
- a study artifact should be treated as stale if `source_commit` differs from
  the current worktree commit when the harness is run in check mode;
- a study artifact should be treated as non-interpretable if any selected
  metric row is missing, duplicated, malformed, or non-pass;
- dirty worktree state is allowed only with an explicit caveat and should not
  be used as release proof.

If Days 10-11 add a normalized report family, the preferred freshness policy is
`generated_compare_inputs`, matching the selected oracle family model.

## Unsupported-Claim Checks

The harness and later documentation should reject or flag wording that turns
this study into:

- broad QR correctness;
- broad minimum-norm behavior;
- external-library or ecosystem parity;
- LAPACK, NumPy, or SciPy parity;
- performance superiority;
- hosted CI proof;
- package-manager support;
- shared-library or dynamic ABI support;
- runtime-loader support;
- platform portability;
- state-of-the-art sparse linear algebra evidence.

At schema level, these claims should appear only in `non_claims`, not in
`claim_scope`.

## Day 6 Handoff

Day 6 should design the harness around this schema:

- generate `study.tsv`, `summary.md`, and `manifest.tsv` under
  `build/comparison/qr_minnorm/`;
- collect all required provenance before comparing metrics;
- parse the baseline `OK <value_count>` protocol;
- produce project-side output for the same selected fixture;
- emit required metric rows with stable row ids;
- fail closed on malformed, missing, duplicated, or non-pass selected rows;
- keep optional NumPy/SciPy package rows deferred unless a later policy selects
  them.
