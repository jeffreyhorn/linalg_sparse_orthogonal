# Day 3 QR Comparison Metric Contract

## Purpose

Day 3 defines the selected `qr_overdetermined_compatible_5x3` comparison
fields, tolerances, row-state semantics, and claim boundaries before any
harness or report-index implementation changes.

## Selected Family

| Field | Value |
| --- | --- |
| Target label | `qr-compatible-ls` |
| Fixture key | `qr_overdetermined_compatible_5x3` |
| Operation | `least_squares_solve` |
| Proposed output root | `build/comparison/qr_compatible_ls/` |
| Baseline name | `qr_external_dense_reference` |
| Baseline type | `external_process_dense_reference` |
| Baseline command | `python3 tests/qr_external_dense_reference.py qr_overdetermined_compatible_5x3` |
| Project name | `sparse_lu_ortho` |
| Claim scope | fixture-local QR compatible least-squares comparison only |
| Initial support tier | `local_only` until selected freshness and hosted artifact behavior are implemented |

The source-controlled helper emits:

```text
OK 4
1
-2
0.5
0
```

The solution 2-norm for `[1, -2, 0.5]` is `2.2912878474779199`.

## Selected Rows

| Row ID | Row Kind | Metric | Expected Value | Tolerance Kind | Tolerance Value | Claim Role |
| --- | --- | --- | --- | --- | --- | --- |
| `comparison_qr_overdetermined_compatible_5x3_project_status_v1` | `metric_comparison` | `project_status` | `SPARSE_SUCCESS` | `status_only` | empty | Claim-bearing gate row. |
| `comparison_qr_overdetermined_compatible_5x3_baseline_status_v1` | `dependency_status` | `baseline_status` | `success` | `status_only` | empty | Claim-bearing gate row because the source-controlled baseline must run. |
| `comparison_qr_overdetermined_compatible_5x3_residual_norm_v1` | `metric_comparison` | `residual_norm` | `<=1e-10` and baseline residual `0` | `absolute` | `1e-10` | Claim-bearing residual agreement row. |
| `comparison_qr_overdetermined_compatible_5x3_solution_norm_v1` | `metric_comparison` | `solution_norm` | `2.2912878474779199` | `absolute` | `1e-10` | Claim-bearing solution-norm agreement row. |
| `comparison_qr_overdetermined_compatible_5x3_solution_values_v1` | `metric_comparison` | `solution_values` | `1,-2,0.5` | `absolute_per_component` | `1e-10` | Claim-bearing component agreement row. |
| `comparison_qr_overdetermined_compatible_5x3_project_vs_baseline_max_abs_delta_v1` | `metric_comparison` | `project_vs_baseline_max_abs_delta` | `<=1e-10` | `absolute` | `1e-10` | Claim-bearing aggregate delta row. |

All six rows are selected rows. A complete selected family requires exactly one
row for each selected row ID, no extra selected row IDs for the fixture, and
`status=pass` for every selected row.

## Required Study Fields

The selected family should reuse the existing comparison `study.tsv` schema:

| Field | Requirement |
| --- | --- |
| `comparison_row_id` | Stable selected row ID from the table above. |
| `report_family` | `comparison`. |
| `subfamily` | Proposed value `qr_compatible_ls`. |
| `row_kind` | `metric_comparison` or `dependency_status`. |
| `fixture_key` | `qr_overdetermined_compatible_5x3`. |
| `operation` | `least_squares_solve`. |
| `metric` | Stable metric name from the selected row table. |
| `baseline_name` | `qr_external_dense_reference`. |
| `baseline_type` | `external_process_dense_reference`. |
| `baseline_version` | Source commit plus helper path or helper version. |
| `baseline_command` | Exact helper command. |
| `baseline_python_executable` | Resolved Python executable path. |
| `baseline_python_version` | Full Python version string. |
| `project_name` | `sparse_lu_ortho`. |
| `project_version` | Version from `VERSION`. |
| `project_command` | Exact project-side probe command. |
| `source_commit` | Current Git commit. |
| `source_branch` | Current branch if available. |
| `worktree_state` | `clean` or `dirty`. |
| `platform` | Platform string. |
| `compiler` | Compiler identity or `unknown` with caveat. |
| `configuration` | Build/configuration description. |
| `expected_value` | Expected status or numeric/vector contract. |
| `project_value` | Observed project value, or empty for dependency-only rows. |
| `baseline_value` | Observed baseline value, or empty for project-only rows. |
| `delta_value` | Difference metric where applicable. |
| `tolerance_kind` | `status_only`, `absolute`, or `absolute_per_component`. |
| `tolerance_value` | Numeric tolerance or empty for status-only. |
| `status` | `pass`, `fail`, `skip`, `defer`, or `error`. |
| `status_reason` | Stable short reason. |
| `caveat` | Caveat preserving local generated and non-claim boundaries. |
| `artifact_path` | Generated artifact containing this row. |
| `generated_at_utc` | UTC timestamp. |
| `support_tier` | `local_only` until later promotion is explicitly implemented. |
| `claim_scope` | `fixture-local QR compatible least-squares comparison only`. |
| `non_claims` | Semicolon-separated non-claims. |

## Excluded Fields And Non-Claims

This selected family is compatible overdetermined least-squares evidence. It
does not emit or claim selected rank, nullspace/projector, or minimum-norm
fields.

| Excluded Field Family | Reason |
| --- | --- |
| rank and rank threshold | The selected fixture is not closing global rank policy. |
| nullspace or projector metrics | The fixture does not prove nullspace basis, projector, or orientation behavior. |
| minimum-norm fields | Minimum-norm behavior is already owned by `qr_underdetermined_minnorm_2x4`; this family is compatible least-squares. |
| raw Q/R entries | Raw basis, sign, orientation, pivot, and ordering identity are non-claims. |
| timing or memory metrics | Sprint 160 comparison rows are correctness/freshness rows, not performance proof. |
| package, ABI, platform, release, or hosted parity fields | These are outside the selected QR comparison claim. |

Required non-claims:

- no broad QR parity;
- no LAPACK, NumPy, SciPy, SuiteSparse, Eigen, or external-library ecosystem
  parity;
- no raw QR basis identity, Q sign/orientation, Q/R entry, pivot, or ordering
  claim;
- no global rank-threshold policy;
- no broad rank-deficient solve, nullspace, minimum-norm, or least-squares
  correctness claim beyond named selected rows;
- no optional package pass evidence;
- no macOS or Windows report-index parity;
- no performance, package-manager, shared-library ABI, dynamic-linking,
  release, or state-of-the-art claim.

## Row-State Semantics

| State | Meaning | Counts As Evidence |
| --- | --- | --- |
| `pass` | Required project and baseline commands ran and the selected metric met tolerance. | Yes, fixture-local only. |
| `fail` | Required commands ran but status or metric tolerance failed. | No. |
| `skip` | A non-selected optional path was unavailable under an explicit optional policy. | No. |
| `defer` | A future baseline or target was intentionally not selected. | No. |
| `error` | Required command, parser, provenance, output, or row contract failed. | No. |
| missing row | A selected row ID is absent. | No; fail freshness. |
| duplicate row | A selected row ID appears more than once. | No; fail freshness. |
| unexpected selected row | A generated selected fixture row is outside the expected selected row set. | No; fail freshness. |
| stale row | `source_commit` differs from the current source commit under freshness checking. | No; fail freshness. |
| malformed row | Required fields are empty or unparsable. | No; fail freshness. |

Skip and defer rows must stay visible if emitted, but they must never satisfy
selected freshness or public claim wording.

## Freshness Contract

The generated family is current only when:

- all six selected row IDs are present exactly once;
- every selected row has `status=pass`;
- `source_commit` matches the current worktree commit during freshness checks;
- required provenance fields are populated;
- generated artifact paths point under `build/comparison/qr_compatible_ls/`;
- no selected row reports `skip`, `defer`, `fail`, or `error`;
- optional dependency rows, if any are later added, remain diagnostic and do
  not alter selected row pass/fail status.

## Implementation Handoff

Day 4 should design the harness extension around:

- adding a generator target such as `qr-compatible-ls`;
- writing artifacts under `build/comparison/qr_compatible_ls/`;
- preserving the existing `qr-minnorm` row set and output paths;
- producing the six selected rows defined here;
- failing closed on missing, duplicate, unexpected, stale, malformed, non-pass,
  or required dependency failure states;
- keeping rank, nullspace/projector, and minimum-norm fields out of this
  family unless a later sprint selects those claims explicitly.
